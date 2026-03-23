import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np
from enum import Enum
import pygame

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridworldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5, render_mode=None): # 5x5grid
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict({
        "agent": spaces.Box(low=0, high=size-1, shape=(2,), dtype=int),
        "target": spaces.Box(low=0, high=size-1, shape=(2,), dtype=int),})
        self.action_space = gym.spaces.Discrete(4)

        self.agent_position = None
        self.target_position = None
        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),}  # Move down (negative y)
        
        self.walls = {
            ((1,0), (1,1)),
            ((1,1), (1,2)),
            ((3,2), (4,2)),
            ((3,3), (3,4))}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.clock = None
        self.window = None


    def _get_obs(self):
        return{"agent": self.agent_position, "target": self.target_position}
    
    def _get_info(self):
        return{"distance":np.linalg.norm(self.agent_position - self.target_position, ord=1)}
    
    def _is_wall(self, old_pos, new_pos):
        x1, y1 = old_pos
        x2, y2 = new_pos

        # 向右移动
        if x2 == x1 + 1 and y1 == y2:
            return ((x1+1, y1), (x1+1, y1+1)) in self.walls
        # 向左移动
        if x2 == x1 - 1 and y1 == y2:
            return ((x1, y1), (x1, y1+1)) in self.walls
        # 向上移动
        if y2 == y1 + 1 and x1 == x2:
            return ((x1, y1+1), (x1+1, y1+1)) in self.walls
        # 向下移动
        if y2 == y1 - 1 and x1 == x2:
            return ((x1, y1), (x1+1, y1)) in self.walls

        return False


    def reset(self, seed: Optional[int]=None, options:Optional[dict]=None):
        super().reset(seed=seed) #Must call this first to seed the random number generator
        # randomly place the agent
        self.agent_position = self.np_random.integers(0, self.size, size=2, dtype=int)
        # randomly place the traget
        self.target_position = self.agent_position # ensure no the same position
        while np.array_equal(self.target_position, self.agent_position):
            self.target_position = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        reward = 0
        old_position = self.agent_position.copy()
        old_distance = np.linalg.norm(old_position - self.target_position, ord=1)
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]
        # Update agent position, ensuring it stays within grid bounds
        new_position = np.clip(self.agent_position + direction, 0, self.size-1)
        # Check if hit teh wall
        hit_wall = self._is_wall(self.agent_position,new_position)
        if hit_wall:
            reward -= 0.1
        else:
            self.agent_position = new_position
        # Check if agent reached the target
        terminated = np.array_equal(self.agent_position, self.target_position)
        truncated = False

        if terminated:
            reward = 3
        else:
            new_distance = np.linalg.norm(self.agent_position - self.target_position, ord=1)
            # Potential-based shaping: strong incentive to approach target
            progress = old_distance - new_distance
            reward += -0.04 + 0.15 * progress  


        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size) 
        # draw the target
        target_pos = tuple((self.target_position * pix_square_size).astype(int))
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(target_pos, 
        (int(pix_square_size), int(pix_square_size))))
        # draw the agent
        agent_pos = tuple(((self.agent_position + 0.5) * pix_square_size).astype(int))
        pygame.draw.circle(canvas, (0, 0, 255), agent_pos, int(pix_square_size / 3))
        # draw the gridlines
        for x in range(self.size+1):
            line_pos = int(pix_square_size * x)
            pygame.draw.line(canvas, 0, (0, line_pos), (self.window_size, line_pos), width=3)
            pygame.draw.line(canvas, 0, (line_pos, 0), (line_pos, self.window_size), width=3)
            
        # draw the walls
        WALL_COLOR = (255, 140, 0)   # 橙色
        WALL_WIDTH = 6               # 粗线条

        for (x1, y1), (x2, y2) in self.walls:
            if x1 == x2:  # 竖直墙
                pygame.draw.line(
                    canvas,
                    WALL_COLOR,
                    (x1 * pix_square_size, y1 * pix_square_size),
                    (x2 * pix_square_size, y2 * pix_square_size),
                    WALL_WIDTH
                )
            else:  # 水平墙
                pygame.draw.line(
                    canvas,
                    WALL_COLOR,
                    (x1 * pix_square_size, y1 * pix_square_size),
                    (x2 * pix_square_size, y2 * pix_square_size),
                    WALL_WIDTH
                )

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else: #rgb_array
            return np.transpose(np.array(pygame.surfarray.array3d(canvas)), axes=(1, 0, 2))
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# 注册环境
gym.register(id="gymnasium_env/GridWorld-v0",
             entry_point=GridworldEnv,
             max_episode_steps=300)