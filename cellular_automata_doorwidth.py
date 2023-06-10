import numpy as np
from tqdm import tqdm
from ca import CellularAutomaton2D
from pyvis import PygameVisualizer


def main():
    scenario = 1  # choose scenario
    num_exp = 1  # sätt 1 om 1 simulering ska köras
    results = []
    showscreen = True
    stepping = False
    stepsize = 1
    grid_size = (18, 20)
    exit_pos = [(grid_size[0] // 2 - 1, 0)]
    show_numbers = False

    count = 0
    if scenario == 0:
        for _ in range(num_exp):
            pct_obstacles = 0.2
            pct_pedestrians = 0.05
            pct_chairs = 0.0

            nr_obstacles = int(grid_size[0] * grid_size[1] * pct_obstacles)
            nr_pedestrians = int(grid_size[0] * grid_size[1] * pct_pedestrians)
            nr_chairs = int(grid_size[0] * grid_size[1] * pct_chairs)

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
                    if ca.add_pedestrian(pos):  # Checka if the addition was successful
                        break

            ca.calculate_floor_field()
            vis = PygameVisualizer(ca, fps=60)
            vis.visualize(stepping=True)

    else:  # classroom
        for _ in range(num_exp):
            pct_obstacles = 0.2
            pct_pedestrians = 0.05
            pct_chairs = 0.0

            nr_obstacles = int(grid_size[0] * grid_size[1] * pct_obstacles)
            nr_pedestrians = int(grid_size[0] * grid_size[1] * pct_pedestrians)
            nr_chairs = int(grid_size[0] * grid_size[1] * pct_chairs)

            ca = CellularAutomaton2D(grid_size, exit_pos)

            # Add obstacles as classroom
            for _ in range(nr_obstacles):
                for x in range(0, grid_size[0], 6):
                    for y in range(5, grid_size[1], 3):
                        for a in range(4):
                            pos = (x + a, y)
                            if ca.add_obstacle(
                                pos
                            ):  # Check if the addition was successful
                                break

            # Add pedestrians at random positions
            for _ in tqdm(range(nr_pedestrians), desc="Adding pedestrians"):
                for x in range(0, grid_size[0], 6):
                    for y in range(6, grid_size[1], 3):
                        for a in range(4):
                            ca.add_pedestrian(
                                (x + a, y)
                            )  # Check if the addition was successful

            ca.calculate_floor_field()
            # to show simulation visually
            # To find approximation of mean value of total iteration count
            # uncomment in that case

            vis = PygameVisualizer(ca, fps=30, show_numbers=show_numbers)
            results.append(vis.visualize(stepping, stepsize, showscreen))
            count += 1
            print("Iteration", count)
        print(np.mean(results))


if __name__ == "__main__":
    main()
