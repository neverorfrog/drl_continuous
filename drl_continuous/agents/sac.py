import os
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from drl_continuous.agents.buffer import StandardReplayBuffer
from drl_continuous.agents.networks import Actor, Critic, SquashedGaussianActor

# Seed
SEED = 111
torch.manual_seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    def __init__(
        self,
        name,
        env: gym.Env,
        window=100,
        polyak=0.995,
        pi_lr=0.001,
        q_lr=0.001,
        target_update_freq=2,
        value_update_freq=1,
        policy_update_freq=2,
        alpha=0.5,
        eps=1.0,
        eps_decay=0.4,
        batch_size=64,
        gamma=0.99,
        max_episodes=200,
    ):

        # Hyperparameters
        self.name = name
        self.env = env
        self.window = window
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.target_update_freq = target_update_freq
        self.value_update_freq = value_update_freq
        self.policy_update_freq = policy_update_freq
        self.alpha = alpha
        self.eps = eps
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_episodes = max_episodes

        self.min_alpha = 0.2

        # env params for networks and buffer
        observation = env.reset()[0]
        self.env_params = {
            "obs_dim": observation.shape[0],
            "action_dim": env.action_space.shape[0],
            "action_bound": env.action_space.high[0],
            "max_steps": env._max_episode_steps,
        }

        # Networks
        self.actor: SquashedGaussianActor = SquashedGaussianActor(
            self.env_params
        ).to(device)
        self.critic1: Critic = Critic(self.env_params).to(device)
        self.critic2: Critic = Critic(self.env_params).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=pi_lr
        )
        self.value_optimizer1 = torch.optim.Adam(
            self.critic1.parameters(), lr=q_lr
        )
        self.value_optimizer2 = torch.optim.Adam(
            self.critic2.parameters(), lr=q_lr
        )
        self.value_loss_fn = nn.MSELoss()

        self.target_actor: Actor = deepcopy(self.actor).to(device)
        self.target_critic1: Critic = deepcopy(self.critic1).to(device)
        self.target_critic2: Critic = deepcopy(self.critic2).to(device)

        # Target networks must be updated not directly through the gradients but
        # with polyak averaging
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        for param in self.target_actor.parameters():
            param.requires_grad = False

        # Experience Replay Buffer
        self.memory = StandardReplayBuffer(self.env_params)
        self.start_steps = batch_size

    def train(self):
        # Life stats
        self.ep = 1
        self.training = True
        self.rewards = deque(maxlen=self.window)

        # Populating the experience replay memory
        self.memory.populate(self.env, self.start_steps)

        with tqdm(total=self.max_episodes) as pbar:
            for _ in range(self.max_episodes):
                # ep stats
                self.num_steps = 0
                self.ep_reward = 0

                # ep termination
                done = False

                # starting point
                observation = self.env.reset()[0]

                while not done:
                    new_observation, done, info = self.interaction_step(
                        observation
                    )
                    self.learning_step()
                    observation = new_observation
                    self.num_steps += 1

                self.episode_update(pbar, info)
                if self.ep % 200 == 0:
                    self.save()

    def interaction_step(self, observation: np.ndarray) -> tuple:
        """
        Function responsible for the interaction of the agent with the
        environment. The action is selected by the policy network, then
        performed and the results stored in the replay buffer. It expects a
        numpy array as input.
        """
        observation = torch.as_tensor(observation, dtype=torch.float32).to(
            device
        )
        action = self.select_action(observation).to(device)
        new_observation, reward, terminated, truncated, info = self.env.step(
            action.cpu().numpy()
        )
        done = terminated or truncated
        new_observation = torch.as_tensor(
            new_observation, dtype=torch.float32
        ).to(
            device
        )  # buffer expects tensor
        self.memory.store(observation, action, reward, done, new_observation)
        self.ep_reward += reward
        return new_observation, done, info

    def select_action(self, observation: torch.Tensor) -> torch.Tensor:
        """
        This function selects an action from the policy network. It expects
        to receive a tensor in input.
        """
        with torch.no_grad():
            action, _ = self.actor(observation, with_logprob=False)
        return action

    def learning_step(self) -> bool:
        # Sampling of the minibatch
        batch = self.memory.sample(batch_size=self.batch_size)
        self.alpha = max(
            self.min_alpha, 0.999 * self.alpha
        )  # alpha discounting

        # Learning step
        if self.num_steps % self.value_update_freq == 0:
            self.value_learning_step(batch)
        if self.num_steps % self.policy_update_freq == 0:
            self.policy_learning_step(batch)
        if self.num_steps % self.target_update_freq == 0:
            self.update_target_networks()

    def value_learning_step(self, batch):
        observations, actions, rewards, dones, new_observations = batch

        self.value_optimizer1.zero_grad()
        self.value_optimizer2.zero_grad()

        # Computation of value estimates
        value_estimates1 = self.critic1(observations, actions)
        value_estimates2 = self.critic2(observations, actions)

        # Computation of value targets
        with torch.no_grad():
            actions, log_pi = self.actor(
                new_observations
            )  # (batch_size, action_dim)
            log_pi = log_pi.unsqueeze(1)  # hotfix
            target_values = torch.min(
                self.target_critic1(new_observations, actions),
                self.target_critic2(new_observations, actions),
            )
            targets = rewards + (1 - dones) * self.gamma * (
                target_values - self.alpha * log_pi
            )

        # MSBE
        value_loss1: torch.Tensor = self.value_loss_fn(
            value_estimates1, targets
        )
        value_loss2: torch.Tensor = self.value_loss_fn(
            value_estimates2, targets
        )
        value_loss1.backward()
        self.value_optimizer1.step()
        value_loss2.backward()
        self.value_optimizer2.step()

    def policy_learning_step(self, batch):
        observations, _, _, _, _ = batch
        self.policy_optimizer.zero_grad()

        # Don't waste computational effort
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False

        # Policy Optimization
        estimated_actions, log_pi = self.actor(observations)
        estimated_values: torch.Tensor = torch.min(
            self.critic1(observations, estimated_actions),
            self.critic2(observations, estimated_actions),
        )
        policy_loss: torch.Tensor = (
            self.alpha * log_pi - estimated_values
        ).mean()  # perform gradient ascent
        policy_loss.backward()
        self.policy_optimizer.step()

        # Reactivate computational graph for critic
        for param in self.critic1.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(
                self.target_critic1.parameters(), self.critic1.parameters()
            ):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

            for target, online in zip(
                self.target_critic2.parameters(), self.critic2.parameters()
            ):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

            for target, online in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

    def episode_update(self, pbar: tqdm = None, info: dict = None):
        self.eps = max(0.1, self.eps * self.eps_decay)
        self.rewards.append(self.ep_reward)
        meanreward = np.mean(self.rewards)
        wandb.log({"reward": self.ep_reward, "mean_reward": meanreward})

        if pbar is not None:
            pbar.set_description(
                f"Episode {self.ep} Mean Reward: {meanreward:.2f} Ep_Reward: {self.ep_reward:.2f} Termination: {info['log']}"
            )
            pbar.update(1)
        self.ep += 1

    def evaluate(self, env=None, render: bool = True, num_ep=3):
        mean_reward = 0.0
        if env is None:
            env = self.env

        for i in range(1, num_ep + 1):
            if render:
                print(f"Starting game {i}")

            observation = torch.FloatTensor(env.reset()[0])

            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                with torch.no_grad():
                    action, _ = self.actor(
                        observation, deterministic=True, with_logprob=False
                    )
                observation, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy()
                )
                observation = torch.FloatTensor(observation)
                total_reward += reward
                if render:
                    self.env.render()

            if render:
                print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1 / i) * (total_reward - mean_reward)

        if render:
            print("Mean Reward: ", mean_reward)
        return mean_reward

    def save(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "models", self.name)

        os.makedirs(path, exist_ok=True)

        torch.save(
            self.actor.state_dict(), open(os.path.join(path, "actor.pt"), "wb")
        )
        torch.save(
            self.critic1.state_dict(),
            open(os.path.join(path, "critic1.pt"), "wb"),
        )
        torch.save(
            self.critic2.state_dict(),
            open(os.path.join(path, "critic2.pt"), "wb"),
        )
        print("MODELS SAVED!")

    def load(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "models", self.name)

        self.actor.load_state_dict(
            torch.load(
                open(os.path.join(path, "actor.pt"), "rb"),
                weights_only=True,
                map_location=device,
            )
        )
        self.critic1.load_state_dict(
            torch.load(
                open(os.path.join(path, "critic1.pt"), "rb"),
                weights_only=True,
                map_location=device,
            )
        )
        self.critic2.load_state_dict(
            torch.load(
                open(os.path.join(path, "critic2.pt"), "rb"),
                weights_only=True,
                map_location=device,
            )
        )
        print("MODELS LOADED!")
