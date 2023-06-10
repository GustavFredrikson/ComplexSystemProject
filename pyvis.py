import pygame


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

    def visualize(self, stepping=False, stepsize=10, showscreen=True):
        pygame.init()
        totaliterations = 0
        if not showscreen:
            stepping = False
        # Set the width and height of the grid locations
        WIDTH = self.cell_size
        HEIGHT = self.cell_size
        MARGIN = 1

        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [
            (WIDTH + MARGIN) * self.automaton.grid.shape[1],
            (HEIGHT + MARGIN) * self.automaton.grid.shape[0],
        ]

        if showscreen:
            screen = pygame.display.set_mode(WINDOW_SIZE)

        done = False
        clock = pygame.time.Clock()
        key = ""
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
            if not self.automaton.pedestrians:
                break

            if stepping:
                if key == "space":
                    for _ in range(stepsize):
                        self.automaton.step()
                        totaliterations += 1
                    key = ""
                else:
                    pass
            else:
                self.automaton.step()
                totaliterations += 1

            if showscreen:
                screen.fill(self.colors[0])

            for row in range(self.automaton.grid.shape[0]):
                for column in range(self.automaton.grid.shape[1]):
                    if self.automaton.grid[row][column] == 1:  # obstacle
                        color = self.colors[1]
                    elif self.automaton.grid[row][column] == 4:  # chair
                        color = self.colors[4]
                    else:
                        color = self.colors[int(self.automaton.grid[row][column])]

                    if showscreen:
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
            if showscreen:
                pygame.display.flip()

        pygame.quit()
        return totaliterations
