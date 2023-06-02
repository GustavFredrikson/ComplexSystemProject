import numpy as np
import scipy.spatial
import pygame
from queue import PriorityQueue
from tqdm import tqdm


class PygameVisualizer:
    def __init__(
        self, automaton, fps=50, max_window_size=(400, 400), show_numbers=False
    ):
        self.automaton = automaton
        self.fps = fps
        self.max_window_size = max_window_size
        self.show_numbers = show_numbers

        # Calculate cell size based on max window size and grid size
        self.cell_size = min(
            self.max_window_size[0] // self.automaton.grid.shape[1],
            self.max_window_size[1] // self.automaton.grid.shape[0],
        )

        self.colors = {
            0: (255, 255, 255),  # Empty: White
            1: (0, 0, 0),  # Obstacle: Black
            2: (255, 0, 0),  # Pedestrian: Red
            3: (0, 255, 0),  # Exit: Green
            4: (0, 127, 255),  # Chair: Dark Blue
        }
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

    def visualize(self, stepping=False):
        pygame.init()

        # Set the width and height of the grid locations
        WIDTH = self.cell_size
        HEIGHT = self.cell_size
        MARGIN = 5

        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [
            (WIDTH + MARGIN) * self.automaton.grid.shape[1],
            (HEIGHT + MARGIN) * self.automaton.grid.shape[0],
        ]
        screen = pygame.display.set_mode(WINDOW_SIZE)

        done = False
        clock = pygame.time.Clock()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if not self.automaton.pedestrians:
                break

            if stepping:
                if keys[pygame.K_SPACE]:
                    for _ in range(1):
                        self.automaton.step()
            else:
                self.automaton.step()

            screen.fill(self.colors[0])

            for row in range(self.automaton.grid.shape[0]):
                for column in range(self.automaton.grid.shape[1]):
                    if self.automaton.grid[row][column] == 1:  # obstacle
                        color = self.colors[1]
                    elif self.automaton.grid[row][column] == 4:  # chair
                        color = self.colors[4]
                    else:
                        color = self.colors[int(self.automaton.grid[row][column])]

                    pygame.draw.rect(
                        screen,
                        color,
                        [
                            (MARGIN + WIDTH) * column + MARGIN,
                            (MARGIN + HEIGHT) * row + MARGIN,
                            WIDTH,
                            HEIGHT,
                        ],
                    )

                    if (
                        self.automaton.grid[row][column] != 1 and self.show_numbers
                    ):  # Don't display for obstacles
                        text_surface = self.font.render(
                            f"{self.automaton.floor_field[row][column]:.1f}",
                            True,
                            (0, 0, 0),
                        )
                        screen.blit(
                            text_surface,
                            (
                                (MARGIN + WIDTH) * column + MARGIN + 2,
                                (MARGIN + HEIGHT) * row + MARGIN + 2,
                            ),
                        )

            clock.tick(self.fps)
            pygame.display.flip()

        pygame.quit()


class Pedestrian:
    count = 0

    def __init__(self, pos, ca, memory_length):
        self.pos = pos
        self.ca = ca
        self.memory_length = memory_length
        self.memory_grid = ca.floor_field.copy()

        self.id = Pedestrian.count
        Pedestrian.count += 1

    def move(self, new_pos):
        self.memory_grid[self.pos] += 1
        self.pos = new_pos


class Obstacle:
    def __init__(self, pos, is_chair=False, cost=10):
        self.pos = pos
        self.is_chair = is_chair
        self.cost = cost


class CellularAutomaton2D:
    def __init__(
        self, size, exit_pos, panic_prob=0.05, memory_length=5, show_numbers=False
    ):
        self.grid = np.zeros(size)
        self.exit_pos = exit_pos
        self.panic_prob = panic_prob
        self.pedestrians = []
        self.obstacles = []

        self.memory_length = memory_length  # Add a memory_length attribute

        # Initialize floor field values
        x, y = np.indices(size)
        self.floor_field = scipy.spatial.distance.cdist(
            [(exit_pos)], np.column_stack([x.flatten(), y.flatten()])
        ).reshape(size)
        self.floor_field = self.floor_field * 1.5  # Increase diagonal distances
        self.floor_field[exit_pos] = 1
        self.grid[exit_pos] = 3  # exit

    def add_pedestrian(self, pos):
        if self.grid[pos] == 0:
            self.grid[pos] = 2
            self.pedestrians.append(Pedestrian(pos, self, self.memory_length))
            return True
        return False

    def add_obstacle(self, pos, is_chair=False, cost=2):
        if self.grid[pos] == 0:  # only place obstacle on empty spot
            self.grid[pos] = 1 if not is_chair else 4
            self.floor_field[pos] = np.inf
            self.obstacles.append(Obstacle(pos, is_chair, cost))
            return True
        return False

    def _move_pedestrian(self, pedestrian):
        pos = pedestrian.pos
        if np.random.rand() < self.panic_prob:
            return
        neighbors = self._get_neighbors(pos)
        valid_neighbors = [
            (nx, ny) for nx, ny in neighbors if self.grid[nx, ny] in [0, 3, 4]
        ]  # add 4 for chairs
        if not valid_neighbors:
            return

        min_memory_value = min(
            pedestrian.memory_grid[nx, ny]
            for nx, ny in neighbors
            if self.grid[nx, ny] in [0, 3, 4]  # add 4 for chairs
        )  # empty cell or exit

        best_cells = [
            (nx, ny)
            for nx, ny in neighbors
            if self.grid[nx, ny] in [0, 3, 4]  # add 4 for chairs
            and pedestrian.memory_grid[nx, ny] == min_memory_value
        ]

        if best_cells:
            nx, ny = best_cells[np.random.randint(len(best_cells))]
            for obstacle in self.obstacles:
                if (
                    obstacle.pos == (nx, ny) and obstacle.is_chair
                ):  # if the obstacle is a chair
                    chair_neighbors = self._get_neighbors(obstacle.pos)
                    # check if any neighboring cell of chair is empty
                    if any(self.grid[cx, cy] == 0 for cx, cy in chair_neighbors):
                        if np.random.rand() < 1.0 / (
                            2 * pedestrian.memory_grid[nx, ny]
                        ):  # chairs are harder to move
                            obstacle.pos = pos
                            self.grid[pos] = 4
                            break
                    else:  # if all neighboring cells of chair are not empty, pedestrian looks for another cell
                        best_cells.remove((nx, ny))
                        if best_cells:
                            nx, ny = best_cells[np.random.randint(len(best_cells))]
                        else:
                            return
            else:
                self.grid[pos] = 0

            if (nx, ny) == self.exit_pos:
                self.grid[nx, ny] = 3
                self.pedestrians.remove(pedestrian)
            else:
                self.grid[nx, ny] = 2
                pedestrian.move((nx, ny))

    def _get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                if (
                    dx == 0
                    or dy == 0
                    or (
                        not self.is_obstacle((x + dx, y))
                        and not self.is_obstacle((x, y + dy))
                    )
                ):  # Handle diagonal movement through obstacles
                    neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, pos):
        return np.linalg.norm(np.array(pos) - np.array(self.exit_pos))  # type: ignore

    def can_reach_exit(self, pos):
        # A* search algorithm
        visited = set()
        queue = PriorityQueue()
        queue.put((0, pos))

        while not queue.empty():
            cost, curr_pos = queue.get()

            if curr_pos == self.exit_pos:
                return True

            visited.add(curr_pos)

            for neighbor in self._get_neighbors(curr_pos):
                if neighbor not in visited and not self.is_obstacle(neighbor):
                    new_cost = cost + 1  # Assuming a constant cost of 1 for each step
                    priority = new_cost + self.heuristic(neighbor)
                    queue.put((priority, neighbor))

        return False

    def is_obstacle(self, pos):
        return self.grid[pos] == 1 or self.grid[pos] == 4

    def step(self):
        for pedestrian in self.pedestrians:
            self._move_pedestrian(pedestrian)

    def run(self, steps):
        for _ in range(steps):
            self.step()


def main():
    grid_size = (50, 50)
    pct_obstacles = 0.2
    pct_pedestrians = 0.1
    pct_chairs = 0
    show_numbers = False

    nr_obstacles = int(grid_size[0] * grid_size[1] * pct_obstacles)
    nr_pedestrians = int(grid_size[0] * grid_size[1] * pct_pedestrians)
    nr_chairs = int(grid_size[0] * grid_size[1] * pct_chairs)

    exit_pos = (0, grid_size[1] - 1)
    ca = CellularAutomaton2D(grid_size, exit_pos)

    # Add obstacles at random positions
    for _ in range(nr_obstacles):
        while True:  # Continue until a free position is found
            pos = (
                np.random.randint(0, grid_size[0]),
                np.random.randint(0, grid_size[1]),
            )
            if ca.add_obstacle(pos):  # Check if the addition was successful
                break

    # Add chairs at random positions
    for _ in range(nr_chairs):
        while True:
            pos = (
                np.random.randint(0, grid_size[0]),
                np.random.randint(0, grid_size[1]),
            )
            if ca.add_obstacle(pos, is_chair=True):
                break

    # Add pedestrians at random positions
    for _ in tqdm(range(nr_pedestrians), desc="Adding pedestrians"):
        while True:  # Continue until a free position is found
            pos = (
                np.random.randint(0, grid_size[0]),
                np.random.randint(0, grid_size[1]),
            )
            if ca.add_pedestrian(pos):  # Check if the addition was successful
                break

    vis = PygameVisualizer(ca, fps=60, show_numbers=show_numbers)
    vis.visualize(stepping=True)


if __name__ == "__main__":
    main()
