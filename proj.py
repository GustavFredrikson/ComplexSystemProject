import numpy as np
from ComplexSystemProject.cellular_automaton import CellularAutomaton2D, MooreNeighborhood

# Define the states
DEAD, OBSTACLE, PEDESTRIAN = 0, 1, 2

# Constants for panic percentage and probability of staying in place
PANIC_PERCENTAGE = 0.05


# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Define the cell update function
def update_fn(cell, neighbors, *args):
    # If the cell is a pedestrian
    if cell == PEDESTRIAN:
        min_dist = float("inf")
        min_dir = None

        for dir, neighbor in neighbors.items():
            if neighbor[0] != OBSTACLE:
                dist = euclidean_distance(
                    neighbor[1], args[0]
                )  # args[0] is the exit position
                if dist < min_dist:
                    min_dist = dist
                    min_dir = dir

        if np.random.rand() < PANIC_PERCENTAGE:
            # Stay in place due to panic
            return cell
        elif min_dir is not None:
            # Move to the neighboring cell with the lowest distance to exit
            return neighbors[min_dir][0]
    return cell


# Initialize the CA
ca = CellularAutomaton2D(
    size=(100, 100),
    neighborhood=MooreNeighborhood(
        EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS
    ),
    rule=update_fn,
    args=((50, 50),),  # The exit is at the center of the grid
    dtype=int,
)

# Run the CA
for _ in range(100):
    ca.evolve()

# Visualize the result
ca.plot()
