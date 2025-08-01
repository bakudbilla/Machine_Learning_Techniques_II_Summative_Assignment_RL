import pygame
import sys
import os

CELL_SIZE = 50
GRID_SIZE = 10
WIDTH = CELL_SIZE * GRID_SIZE
HEIGHT = CELL_SIZE * GRID_SIZE + 50  # Smaller extra height for compact legend

BACKGROUND_COLOR = (220, 220, 220)
GRID_COLOR = (180, 180, 180)

class Renderer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Livestock Monitoring Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 18)  # Smaller font

        # Load and scale images (smaller sizes)
        self.image_dir = os.path.join(os.path.dirname(__file__), "../assets")

        self.drone_img = pygame.transform.scale(
            pygame.image.load(os.path.join(self.image_dir, "drone.png")), (40, 40)
        )
        self.animal_img = pygame.transform.scale(
            pygame.image.load(os.path.join(self.image_dir, "cow.png")), (30, 30)
        )
        self.distress_img = pygame.transform.scale(
            pygame.image.load(os.path.join(self.image_dir, "distress_cow.png")), (30, 30)
        )

    def draw_grid(self):
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT - 50))
        for y in range(0, HEIGHT - 50, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

    def draw_drone(self):
        x, y = self.env.drone_pos
        img_width, img_height = self.drone_img.get_size()
        offset_x = (CELL_SIZE - img_width) // 2
        offset_y = (CELL_SIZE - img_height) // 2
        self.screen.blit(self.drone_img, (x * CELL_SIZE + offset_x, y * CELL_SIZE + offset_y))

    def draw_animals(self):
        for a in self.env.animals:
            x, y = a["pos"]
            img = self.distress_img if a["distress"] else self.animal_img
            img_width, img_height = img.get_size()
            offset_x = (CELL_SIZE - img_width) // 2
            offset_y = (CELL_SIZE - img_height) // 2
            self.screen.blit(img, (x * CELL_SIZE + offset_x, y * CELL_SIZE + offset_y))

    def draw_legend(self):
        legend_y = HEIGHT - 45
        spacing = 140

        # Drone
        self.screen.blit(self.drone_img, (10, legend_y))
        label = self.font.render("Drone", True, (0, 0, 0))
        self.screen.blit(label, (55, legend_y + 10))

        # Healthy Animal
        self.screen.blit(self.animal_img, (10 + spacing, legend_y))
        label = self.font.render("Healthy", True, (0, 0, 0))
        self.screen.blit(label, (55 + spacing, legend_y + 10))

        # Distressed Animal
        self.screen.blit(self.distress_img, (10 + 2 * spacing, legend_y))
        label = self.font.render("Distressed", True, (0, 0, 0))
        self.screen.blit(label, (55 + 2 * spacing, legend_y + 10))

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_animals()
        self.draw_drone()
        self.draw_legend()
        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])

    def close(self):
        pygame.quit()
        sys.exit()
