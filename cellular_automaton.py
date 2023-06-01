import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import pygame

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import pygame


class PygameVisualizer:
    def __init__(self, automaton, fps=50, max_window_size=(800, 800)):
        self.automaton = automaton
        self.fps = fps
        self.max_window_size = max_window_size

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
        }

    def visualize(self, steps):
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

        for _ in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            if done:
                break

            self.automaton.step()

            screen.fill(self.colors[0])

            for row in range(self.automaton.grid.shape[0]):
                for column in range(self.automaton.grid.shape[1]):
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
            clock.tick(self.fps)
            pygame.display.flip()

        pygame.quit()


class CellularAutomaton2D:
    def __init__(self, size, exit_pos, panic_prob=0.05):
        self.grid = np.zeros(size)
        self.exit_pos = exit_pos
        self.panic_prob = panic_prob

        # Initialize floor field values
        x, y = np.indices(size)
        self.floor_field = scipy.spatial.distance.cdist(
            [(exit_pos)], np.column_stack([x.flatten(), y.flatten()])
        ).reshape(size)
        self.floor_field = self.floor_field * 1.5  # Increase diagonal distances
        self.floor_field[exit_pos] = 1
        self.grid[exit_pos] = 3  # exit

    def initialize(self, pedestrians, obstacles):
        for p in pedestrians:
            self.grid[p] = 2  # pedestrian
        for o in obstacles:
            self.grid[o] = 1  # obstacle
            self.floor_field[o] = np.inf  # walls

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
                    or (self.grid[x + dx][y] == 0 and self.grid[x][y + dy] == 0)
                ):  # Hanterar diagonalrörelse genom hinder
                    neighbors.append((nx, ny))
        return neighbors

    def _move_pedestrian(self, pos):
        if np.random.rand() < self.panic_prob:
            return  # pedestrian stays in place due to panic
        neighbors = self._get_neighbors(pos)
        min_floor_field = min(
            self.floor_field[nx, ny]
            for nx, ny in neighbors
            if self.grid[nx, ny] in [0, 3]
        )  # empty cell
        best_cells = [
            (nx, ny)
            for nx, ny in neighbors
            if self.grid[nx, ny] in [0, 3]
            and self.floor_field[nx, ny] == min_floor_field
        ]
        if best_cells:
            nx, ny = best_cells[
                np.random.randint(len(best_cells))
            ]  # randomly choose among the best cells
            self.grid[pos] = 0  # old position becomes empty
            if (nx, ny) == self.exit_pos:  # if the pedestrian has reached the exit
                self.grid[pos] = 0  # old position becomes empty
                self.grid[nx, ny] = 3  # refresh exit
                return  # pedestrian is removed from the grid

            else:
                self.grid[nx, ny] = 2  # move pedestrian

    def step(self):
        pedestrians = np.argwhere(self.grid == 2)
        np.random.shuffle(pedestrians)  # randomize order for fairness
        for p in pedestrians:
            self._move_pedestrian(tuple(p))

    def run(self, steps):
        for _ in range(steps):
            self.step()

    def add_pedestrian(self, pos):
        if self.grid[pos] == 0:  # only place pedestrian on empty spot
            self.grid[pos] = 2  # pedestrian
            return True
        return False

    def add_obstacle(self, pos):
        if self.grid[pos] == 0:  # only place obstacle on empty spot
            self.grid[pos] = 1  # obstacle
            self.floor_field[pos] = np.inf  # walls
            return True
        return False

    def remove_pedestrian(self, pos):
        if self.grid[pos] == 2:
            self.grid[pos] = 0  # empty spot

    def remove_obstacle(self, pos):
        if self.grid[pos] == 1:
            self.grid[pos] = 0
            self.floor_field[pos] = scipy.spatial.distance.cdist(
                [(self.exit_pos)], [pos]
            )[0][
                0
            ]  # restore floor field value


if __name__ == "__main__":
    grid_size = (5, 5)
    nr_pedestrians = 5
    nr_obstacles = 5

    exit_pos = (0, grid_size[1] - 1)
    ca = CellularAutomaton2D(grid_size, exit_pos)

    # Add pedestrians at random positions
    for _ in range(nr_pedestrians):
        while True:  # Continue until a free position is found
            pos = (
                np.random.randint(0, grid_size[0]),
                np.random.randint(0, grid_size[1]),
            )
            if ca.add_pedestrian(pos):  # Check if the addition was successful
                break

    # Add obstacles at random positions
    for _ in range(nr_obstacles):
        while True:  # Continue until a free position is found
            pos = (
                np.random.randint(0, grid_size[0]),
                np.random.randint(0, grid_size[1]),
            )
            if ca.add_obstacle(pos):  # Check if the addition was successful
                break

    vis = PygameVisualizer(ca, fps=1)
    vis.visualize(50)
