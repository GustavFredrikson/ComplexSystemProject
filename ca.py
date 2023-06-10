from collections import defaultdict
from collections import deque
from queue import PriorityQueue
import numpy as np


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
        self.exit_positions = exit_pos if isinstance(exit_pos, list) else [exit_pos]

        self.panic_prob = panic_prob
        self.pedestrians = []
        self.obstacles = []
        self.show_numbers = show_numbers
        self.memory_length = memory_length  # Add a memory_length attribute
        self.floor_field = np.full(size, np.inf)
        for exit_pos in self.exit_positions:
            self.floor_field[exit_pos] = 1
            self.grid[exit_pos] = 3  # exit

    def calculate_floor_field(self):
        for exit_pos in self.exit_positions:
            self.floor_field[exit_pos] = 1
            self.grid[exit_pos] = 3

        # BFS from all exit positions
        queue = deque(self.exit_positions)
        while queue:
            x, y = queue.popleft()
            current_val = self.floor_field[x, y]

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
                if (
                    0 <= nx < self.grid.shape[0]
                    and 0 <= ny < self.grid.shape[1]
                    and self.grid[nx, ny] != 1
                    and self.grid[nx, ny] != 4
                ):
                    new_val = (
                        current_val + 1 if dx == 0 or dy == 0 else current_val + 1.5
                    )
                    if new_val < self.floor_field[nx, ny]:
                        self.floor_field[nx, ny] = new_val
                        queue.append((nx, ny))

        for pedestrian in self.pedestrians:
            pedestrian.memory_grid = self.floor_field.copy()

    def add_pedestrian(self, pos):
        if self.grid[pos] == 0:
            self.grid[pos] = 2
            self.pedestrians.append(Pedestrian(pos, self, self.memory_length))
            return True
        return False

    def add_obstacle(self, pos, is_chair=False, cost=2):
        # print(pos)
        if self.grid[pos] == 0:  # only place obstacle on empty spot
            self.grid[pos] = 1 if not is_chair else 4
            self.floor_field[pos] = np.inf
            self.obstacles.append(Obstacle(pos, is_chair, cost))
            return True
        return False

    def _calculate_pedestrian_move(self, pedestrian):
        pos = pedestrian.pos
        if np.random.rand() < self.panic_prob:
            return pos
        neighbors = self._get_neighbors(pos)
        valid_neighbors = [
            (nx, ny) for nx, ny in neighbors if self.grid[nx, ny] in [0, 3, 4]
        ]  # add 4 for chairs
        if not valid_neighbors:
            return pos

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
            return (nx, ny)
        else:
            return pos

    def _execute_pedestrian_move(self, pedestrian, new_pos):
        old_pos = pedestrian.pos
        nx, ny = new_pos

        # Handle movement of chairs
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
                        obstacle.pos = old_pos
                        self.grid[old_pos] = 4
                        break
                else:  # if all neighboring cells of chair are not empty, pedestrian stays put
                    return

        self.grid[old_pos] = 0  # empty the old position

        if new_pos in self.exit_positions:  # updated to check in all exits
            self.pedestrians.remove(pedestrian)
            self.grid[new_pos] = 3  # mark the exit
        else:
            self.grid[new_pos] = 2  # mark the new position
            pedestrian.move(new_pos)

    def step(self):
        moves = defaultdict(list)
        for pedestrian in self.pedestrians:
            new_pos = self._calculate_pedestrian_move(pedestrian)
            moves[new_pos].append(pedestrian)

        for new_pos, pedestrians in moves.items():
            if len(pedestrians) > 1:
                pedestrian = np.random.choice(pedestrians)
            else:
                pedestrian = pedestrians[0]

            self._execute_pedestrian_move(pedestrian, new_pos)

            # För pedestrian som förlorade, stanna kvar på samma plats
            for other_pedestrian in pedestrians:
                if other_pedestrian != pedestrian:
                    self._execute_pedestrian_move(
                        other_pedestrian, other_pedestrian.pos
                    )

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

    def run(self, steps):
        for _ in range(steps):
            self.step()
