import numpy as np
import matplotlib.pyplot as plt


class CellularAutomaton2D:
    def __init__(self, size, exit_pos, panic_prob=0.05):
        self.grid = np.zeros(size)
        self.exit_pos = exit_pos
        self.panic_prob = panic_prob

    def initialize(self, pedestrians, obstacles):
        for p in pedestrians:
            self.grid[p] = 2  # pedestrian
        for o in obstacles:
            self.grid[o] = 1  # obstacle

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
            if (
                0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]
            ):  # within bounds
                neighbors.append((nx, ny))
        return neighbors

    def _move_pedestrian(self, pos):
        if np.random.rand() < self.panic_prob:
            return  # pedestrian stays in place due to panic
        neighbors = self._get_neighbors(pos)
        np.random.shuffle(neighbors)  # randomize order for fairness
        for nx, ny in neighbors:
            if self.grid[nx, ny] == 0:  # empty cell
                self.grid[nx, ny] = 2  # move pedestrian
                self.grid[pos] = 0  # old position becomes empty
                return

    def step(self):
        pedestrians = np.argwhere(self.grid == 2)
        np.random.shuffle(pedestrians)  # randomize order for fairness
        for p in pedestrians:
            self._move_pedestrian(tuple(p))

    def run(self, steps):
        for _ in range(steps):
            self.step()
            self.visualize()

    def visualize(self):
        plt.imshow(self.grid, cmap="viridis")
        plt.show()


if __name__ == "__main__":
    ca = CellularAutomaton2D((10, 10), (5, 5))
    ca.initialize([(0, 0), (9, 9)], [(4, 4), (5, 5), (6, 6)])
    ca.run(10)
