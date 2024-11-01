import random
from typing import List

import gymnasium as gym
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(123)
np.random.seed(123)


class StandardReplayBuffer:
    def __init__(self, env_params, capacity=4096):
        self.env_params = env_params
        self.capacity = capacity
        self.size = 0
        self.buffer = {
            "observation": torch.empty([capacity, env_params["obs_dim"]]),
            "action": torch.empty([capacity, env_params["action_dim"]]),
            "reward": torch.empty([capacity, 1]),
            "done": torch.empty([capacity, 1]),
            "new_observation": torch.empty([capacity, env_params["obs_dim"]]),
        }

    def populate(self, env: gym.Env, start_steps: int = 1000) -> None:
        observation = env.reset()[0]
        observation = torch.as_tensor(observation, dtype=torch.float32).to(
            device
        )  # buffer expects tensor
        for i in range(start_steps):
            action = env.action_space.sample()
            action = torch.as_tensor(action, dtype=torch.float32).to(
                device
            )  # buffer expects tensor
            new_observation, reward, terminated, truncated, _ = env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated
            new_observation = torch.as_tensor(
                new_observation, dtype=torch.float32
            ).to(
                device
            )  # buffer expects tensor
            self.store(observation, action, reward, done, new_observation)
            observation = new_observation
            if terminated or truncated:
                observation = env.reset()[0]
                observation = torch.as_tensor(
                    observation, dtype=torch.float32
                ).to(
                    device
                )  # buffer expects tensor

    def store(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        new_observation: torch.Tensor,
    ) -> None:
        index = self.size % self.capacity
        self.buffer["observation"][index] = observation
        self.buffer["action"][index] = action
        self.buffer["reward"][index] = reward
        self.buffer["done"][index] = done
        self.buffer["new_observation"][index] = new_observation
        self.size += 1

    def sample(self, batch_size=32) -> List[torch.Tensor]:
        max_batch_index = min(self.size, self.capacity - 1)
        sampled_indices = random.sample(range(max_batch_index), batch_size)
        observations = self.buffer["observation"][sampled_indices].to(device)
        actions = self.buffer["action"][sampled_indices].to(device)
        rewards = self.buffer["reward"][sampled_indices].to(device)
        dones = self.buffer["done"][sampled_indices].to(device)
        new_observations = self.buffer["new_observation"][sampled_indices].to(
            device
        )
        return [observations, actions, rewards, dones, new_observations]
