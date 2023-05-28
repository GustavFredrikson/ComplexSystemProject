import numpy as np
from cellular_automaton import CellularAutomaton2D, PygameVisualizer

grid_size = (25, 40)
nr_pedestrians = 10
nr_obstacles = 40

exit_pos = (0, grid_size[1] - 1)
ca = CellularAutomaton2D(grid_size, exit_pos)

pedestrians = [
    (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
    for _ in range(nr_pedestrians)
]

obstacles = []
for _ in range(nr_obstacles):
    while True:
        obstacle_pos = (
            np.random.randint(0, grid_size[0]),
            np.random.randint(0, grid_size[1]),
        )
        if obstacle_pos not in pedestrians:
            obstacles.append(obstacle_pos)
            break  # Found a valid position, break the while loop

ca.initialize(pedestrians, obstacles)

vis = PygameVisualizer(ca, fps=1)
vis.visualize(50)
