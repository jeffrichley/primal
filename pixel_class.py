import numpy as np
from pygame.sprite import Sprite
from pygame import Surface

# Define a class for our pixels/grid spaces
class Pixel(Sprite):
    def __init__(self, x, y, screen, color, scale):
        super(Pixel, self).__init__()

        self.x = x * scale
        self.y = y * scale

        self.color = color
        self.screen = screen
        self.surf = Surface((scale, scale))
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect()
        self.show()

    # def _change_color(self):
    #     self.color = state_dict[self.state]

    def show(self):
        self.screen.blit(self.surf, (self.x, self.y))

    def change_state(self, color):
        if np.array_equal(self.color, color):
            return
    #     if self.state == state:
    #         return
    #
    #     self.state = state
        self.color = color
    #     self._change_color()
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect()
        self.show()
