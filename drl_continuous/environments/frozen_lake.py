import os
from typing import Tuple

import numpy as np
import pygame
from gymnasium.core import Env
from gymnasium.spaces import Box
from pygame.image import load
from pygame.transform import scale

from drl_continuous.utils import IMG_DIR, Q_DIR, parse_map_grid
from drl_continuous.utils.definitions import RewardType

SEED = 111

MAPS = {
    "lab0": [
        "🟩 🟩 🟩 🟩 1",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩 ",
    ],
    "lab1": [
        "🟩 🟩 🟩 ⛔ 1",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩 ",
    ],
    "lab2": [
        "🟩 🟩 🟩 ⛔ 1",
        "🟩 ⛔ 🟩 ⛔ 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩 ",
    ],
    "lab3": [
        "🟩 🟩 🟩 ⛔ 1",
        "🟩 ⛔ 🟩 ⛔ 🟩",
        "🟩 ⛔ 🟩 ⛔ 🟩",
        "🟩 ⛔ 🟩 🟩 🟩",
        "🟩 ⛔ 🟩 🟩 🟩 ",
    ],
    "lab4": [
        "🟩 🟩 🟩 ⛔ 1",
        "🟩 ⛔ 🟩 ⛔ 🟩",
        "🟩 ⛔ 🟩 ⛔ 🟩",
        "🟩 ⛔ 🟩 ⛔ 🟩",
        "🟩 ⛔ 🟩 🟩 🟩 ",
    ],
}


class ContinuousFrozenLake(Env):
    """
    The FrozenLake environment is a simple gridworld MDP with a start state, a
    goal state, and holes. For simplicity, the map is assumed to be a squared
    grid.

    The agent can move in the two-dimensional grid with continuous velocities.
    Positive y is downwards, positive x is to the right.
    """

    observation: np.ndarray
    start_obs: np.ndarray
    action: np.ndarray
    reward: float
    prev_observation: np.ndarray
    prev_action: np.ndarray
    num_steps: int

    def __init__(
        self,
        map_name: str = "6x6",
        reward_type: RewardType = RewardType.dense,
        is_slippery: bool = True,
        is_rendered: bool = True,
    ):
        self.reward_type = reward_type
        self.is_slippery = is_slippery
        self.is_rendered = is_rendered

        self.size, self.holes, self.goals = parse_map_grid(MAPS[map_name])
        self.num_goals = len(self.goals)

        self.goal_idx = 0
        self.old_goal = np.array(list(self.goals.values())[0])
        self.goal = np.array(list(self.goals.values())[0])
        self.observation_space = Box(low=0, high=self.size, shape=(2,))
        self.action_space = Box(low=-0.5, high=0.5, shape=(2,))

        self.max_distance = np.linalg.norm(np.array([self.size, self.size]))
        self.reward_range = Box(low=-self.max_distance, high=0, shape=(1,))
        self._max_episode_steps = 100

        self.grid_height = self.size
        self.grid_width = self.size
        self._cell_size = 100
        self.is_pygame_initialized = False
        if self.is_rendered:
            self.init_pygame()

        self.complete_Q = np.load(f"{Q_DIR}/q_tables_{map_name}.npz")[
            "q_table_a1"
        ]
        self.Q = self.complete_Q[self.goal_idx :: 2, :]

    def reward_function(self, obs: np.ndarray) -> Tuple[float, bool]:
        terminated = False
        reward = 0
        log = None

        if self.reward_type == RewardType.dense:
            goal_distance = np.linalg.norm(obs - self.grid2frame(self.goal))
            reward = 1 - goal_distance / self.max_distance
        elif self.reward_type == RewardType.model:
            cell = np.floor(self.frame2grid(obs)).astype(int)
            state = cell[1] * self.size + cell[0]
            reward = np.max(self.Q[state, :])

        if self.num_steps >= self._max_episode_steps:
            log = "MAX STEPS"
            terminated = True
            reward = -50

        # Check for successful termination
        if self.is_inside_cell(obs, self.goal):
            log = "GOAL REACHED"
            reward = 100
            self.old_goal = self.goal
            self.goal_idx += 1
            if self.goal_idx == self.num_goals:
                terminated = True
                self.goal_idx = 0
            else:
                goal_key = list(self.goals.keys())[self.goal_idx]
                self.goal = np.array(self.goals[goal_key])

        # Check for failure termination
        for hole in self.holes:
            if self.is_inside_cell(obs, hole):
                terminated = True
                reward = -10
                log = "HOLE"
                break

        return reward, terminated, log

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        The agent takes a step in the environment.
        """
        self.prev_observation = self.observation
        action = np.clip(
            action, self.action_space.low, self.action_space.high
        )  # Ensure action is within bounds
        new_x = np.clip(self.observation[0] + action[0], 0.1, self.size - 0.1)
        new_y = np.clip(self.observation[1] + action[1], 0.1, self.size - 0.1)

        if self.is_slippery:
            new_x = np.clip(
                new_x + np.random.normal(0, 0.01), 0.1, self.size - 0.1
            )
            new_y = np.clip(
                new_y + np.random.normal(0, 0.01), 0.1, self.size - 0.1
            )

        self.observation = np.array([new_x, new_y])
        self.reward, terminated, log = self.reward_function(self.observation)
        self.num_steps += 1
        return self.observation, self.reward, terminated, False, {"log": log}

    def reset(self) -> Tuple[np.ndarray, dict]:
        super().reset(seed=SEED)
        self.num_steps = 0
        self.observation = np.array(
            [0.5, 0.5]
        )  # Start state: bottom left corner
        self.start_obs = self.observation
        return self.observation, {}

    def is_inside_cell(self, pos: np.ndarray, cell: np.ndarray) -> bool:
        """
        Check if a position is inside a cell.
        """
        cell_coord = self.grid2frame(cell)
        # inside = True
        # if pos[0] < cell_coord[0] - 0.5: inside = False # left
        # if pos[0] > cell_coord[0] + 0.5: inside = False # right
        # if pos[1] < cell_coord[1] - 0.5: inside = False # bottom
        # if pos[1] > cell_coord[1] + 0.5: inside = False # top

        # if inside is True:
        #     print(pos, cell_coord, cell)
        #     print(inside)
        #     print("---")
        return np.linalg.norm(pos - cell_coord) < 0.5

    def frame2grid(self, frame_pos: np.ndarray) -> np.ndarray:
        """
        Convert a frame position to a grid position.
        """
        assert len(frame_pos) == 2
        x, y = frame_pos
        # Flipping y axis
        y = self.size - y
        return np.array([x, y])

    def grid2frame(self, grid_pos: np.ndarray) -> np.ndarray:
        """
        Convert a grid position to a frame position.
        """
        assert len(grid_pos) == 2
        x, y = grid_pos
        # Flipping y axis
        y = self.size - y

        # Centering the position
        x += 0.5
        y -= 0.5

        return np.array([x, y])

    def render(self):
        """
        Renders the environment with the given observation.

        Args:
            obs (np.ndarray): The observation to render.
        """
        if not self.is_pygame_initialized and self.is_rendered:
            self.init_pygame()
            self.is_pygame_initialized = True
        for x in range(0, self.screen_width, self._cell_size):
            for y in range(0, self.screen_height, self._cell_size):
                # Draw the ice
                rect = ((x, y), (self._cell_size, self._cell_size))
                self.screen.blit(self.ice_img, (x, y))
                pygame.draw.rect(self.screen, (200, 240, 240), rect, 1)

        # Draw the holes
        for hole in self.holes:
            hole_x, hole_y = hole
            hole_x = (
                hole_x * self._cell_size
            )  # - self.hole_img.get_width() // 2
            hole_y = (
                hole_y * self._cell_size
            )  # - self.hole_img.get_height() // 2
            if self.is_inside_cell(self.observation, hole):
                self.screen.blit(self.cracked_hole_img, (hole_x, hole_y))
            else:
                self.screen.blit(self.hole_img, (hole_x, hole_y))

        for goal_char, (g_x, g_y) in self.goals.items():
            goal_rect = pygame.Rect(
                g_x * self._cell_size,
                g_y * self._cell_size,
                self._cell_size,
                self._cell_size,
            )
            pygame.draw.rect(self.screen, (255, 215, 0), goal_rect)
            goal_text = self.font.render(str(goal_char), True, (0, 0, 0))
            text_rect = goal_text.get_rect(center=goal_rect.center)
            self.screen.blit(goal_text, text_rect)

        # Draw the agent (it lives in the continuous reference frame with different coordinates)
        # Namely, [0,0] is in the bottom left corner, as if the grid was a cartesian plane
        # Thus, positive y speed is upwards, positive x speed is to the right
        x, y = self.frame2grid(self.observation)
        agent_x = x * self._cell_size - self.agent_img.get_width() // 2
        agent_y = y * self._cell_size - self.agent_img.get_height() // 2
        self.screen.blit(self.agent_img, (agent_x, agent_y))

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(60)

    def init_pygame(self):
        """
        Initialize the Pygame environment.
        """
        pygame.init()
        self.clock = pygame.time.Clock()

        # Screen
        self.screen_width = self.grid_width * self._cell_size
        self.screen_height = self.grid_width * self._cell_size
        self.screen = pygame.display.set_mode(
            (int(self.screen_width), int(self.screen_height))
        )

        # Images
        self.font = pygame.font.SysFont("Arial", 25)  # Crea un oggetto font

        agent_img_path = os.path.join(IMG_DIR, "ita_man.png")
        self.agent_img = scale(
            load(agent_img_path), (self._cell_size, self._cell_size)
        )

        ice_img_path = os.path.join(IMG_DIR, "ice.png")
        self.ice_img = scale(
            load(ice_img_path), (self._cell_size, self._cell_size)
        )

        hole_img_path = os.path.join(IMG_DIR, "hole.png")
        self.hole_img = scale(
            load(hole_img_path), (self._cell_size, self._cell_size)
        )

        cracked_hole_img_path = os.path.join(IMG_DIR, "cracked_hole.png")
        self.cracked_hole_img = scale(
            load(cracked_hole_img_path), (self._cell_size, self._cell_size)
        )

        goal_img_path = os.path.join(IMG_DIR, "goal.png")
        self.goal_img = scale(
            load(goal_img_path), (self._cell_size // 3, self._cell_size // 3)
        )
