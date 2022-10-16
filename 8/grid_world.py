import random
import numpy as np
from typing import Optional, Tuple
from grid_world_state import GridWorldEnvState

class GridWorldEnv:
    def __init__(self, render_mode: Optional[str] = None):
        self.grid_world_size = 20
        self.grid = np.zeros((self.grid_world_size,) * 2, dtype=int)
        self.action_space = ['up', 'down', 'left', 'right']
        self.action_space_n = len(self.action_space) # up, down, left, right
        self.state_space_n = 3 # clear square, dangerous square, win square
        self.state = GridWorldEnvState() #None
        self.damage = 0
        self.damage_threshold = 5
        
        self._generate_world()

        self.stepped_beyond_terminated = None

        self.render_mode = render_mode

        # pygame
        self.screen_size = 800
        self.screen = None

    def sample_action_space(self) -> str:
        return random.choice(self.action_space)

    def step(self, action: str):
        assert action in self.action_space, 'Invalid action'
        assert self.state is not None, 'Call reset before using step method'

        terminated = bool(
            self.state.pos.x < 0
            or self.state.pos.x >= self.grid_world_size
            or self.state.pos.y < 0
            or self.state.pos.y >= self.grid_world_size
            or self.damage > self.damage_threshold
        )

        if not terminated:
            pass
        elif self.stepped_beyond_terminated is None:
            # Homie just died
            self.stepped_beyond_terminated = False
        else:
            assert not self.stepped_beyond_terminated, 'Call reset after terminated is True'
            self.stepped_beyond_terminated = True
        
        if self.render_mode == 'human':
            self._render()

        pass

    def _render(self):
        #assert self.render_mode is not None, 'Render mode must be set to "human" to render'
        #assert self.state is None, "Do not directly call this method; use step() instead" 

        import pygame
        if self.screen is None:
            pygame.init()
            #pygame.display.init() unnecessary?
            self.screen = pygame.display.set_mode((self.screen_size,) * 2)
        
        surf = pygame.Surface((self.screen_size,) * 2)
        BG_COLOR = 205
        surf.fill((BG_COLOR,) * 3)

        square_sz = self.screen_size / self.grid_world_size
        l, t = 0, 0
        for x in range(self.grid_world_size):
            for y in range(self.grid_world_size):
                l = square_sz * y
                t = square_sz * x
                rect = (l, t, square_sz, square_sz)
                square_color, is_bordered = self._get_square_display(x, y)
                
                pygame.draw.rect(surf, square_color, rect, is_bordered)

        self.screen.blit(surf, (0, 0))

        while True:
            pygame.display.update()
            pygame.event.pump()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def _get_square_display(self, x: int, y: int):
        if self.state.pos.x == x and self.state.pos.y == y:
            return ((0, 64, 255), False)
        elif self.grid[x,y] == 1:
            return ((255, 0, 64), False)
        elif self.grid[x,y] == 2:
            return ((64, 255, 0), False)
        else:
            return ((55,) * 3, True)

    def _generate_world(self):
        # Randomly generate goal in bottom right quarter of map
        r_x, r_y = self._generate_random_point(self.grid_world_size / 2)
        self.grid[r_x, r_y] = 2

        # Randomly generate dangerous zones
        DANGER_ZONES = 20
        for _ in range(DANGER_ZONES):
            r_x, r_y = 0, 0
            is_start_pos = lambda x, y: x == 0 and y == 0
            while (is_start_pos(r_x, r_y) or self.grid[r_x, r_y] == 1 or self.grid[r_x, r_y] == 2):
                r_x, r_y = self._generate_random_point()
            
            self.grid[r_x, r_y] = 1
    
    def _generate_random_point(self, lower_bound = 0) -> Tuple[int, int]:
            r_x = random.randint(lower_bound, self.grid_world_size - 1)
            r_y = random.randint(lower_bound, self.grid_world_size - 1)

            return (r_x, r_y)