from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from drl_continous.agents.buffer import StandardReplayBuffer
from drl_continous.agents.networks import Actor, Critic
import os
from collections import deque

# Seed
SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
              
class TD3:
    def __init__(self, name, env: gym.Env, window = 100,
                 polyak = 1.0, pi_lr = 0.0001, q_lr = 0.0001, target_update_freq = 2, value_update_freq = 1, 
                 eps = 1.0, eps_decay = 0.99, batch_size = 64, gamma=0.99, max_episodes=200, reward_threshold=100,
                 target_noise = 0.2, noise_clip = 0.3, policy_update_freq = 2):

        # Hyperparameters
        self.name = name
        self.env = env
        self.window = window
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.target_update_freq = target_update_freq
        self.policy_update_freq = policy_update_freq
        self.value_update_freq = value_update_freq
        self.eps = eps
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.reward_threshold = reward_threshold
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        
        # env params for networks and buffer
        observation = env.reset()[0]
        self.env_params = {'obs_dim': observation.shape[0], 
                           'action_dim': env.action_space.shape[0], 
                           'action_bound': env.action_space.high[0],
                           'max_steps': env._max_episode_steps}

        # Networks
        self.actor: Actor = Actor(self.env_params).to(device)
        self.critic1: Critic = Critic(self.env_params).to(device)
        self.critic2: Critic = Critic(self.env_params).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.value_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=q_lr)
        self.value_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=q_lr)
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
        
        print("Training...")

        # Life stats
        self.ep = 1
        self.training = True
        self.rewards = deque(maxlen=self.window)
        self.losses = deque(maxlen=self.window)

        # Populating the experience replay memory
        self.populate_buffer()

        while self.training: 

            # ep stats
            num_steps = 0
            self.ep_reward = 0
            self.ep_mean_value_loss = 0.

            # ep termination
            done = False

            # starting point
            observation = self.env.reset()[0]

            while not done:
                new_observation, done = self.interaction_step(observation)
                self.learning_step(num_steps)
                observation = new_observation
                num_steps += 1

            self.episode_update()


    def interaction_step(self, observation):
        action = self.select_action(observation, noise_weight = self.eps)
        new_observation, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        # Storing in the memory
        self.memory.store(observation,action,reward,done,new_observation) 
        # stats
        self.ep_reward += reward
        return new_observation, done
    
    def select_action(self, observation, noise_weight = 0.2):
        with torch.no_grad(): 
            action = self.actor(torch.as_tensor(observation, dtype=torch.float32))
            action += noise_weight * np.random.randn(self.env_params['action_dim'])
            action = np.clip(action, -self.env_params['action_bound'], self.env_params['action_bound'])
        return action
    
    def estimate_best_actions(self, observations):
        '''
        Estimate the best actions for given observations using the target actor
        network with added noise for policy smoothing. 
        
        Args:
            observations (numpy.ndarray or torch.Tensor): The input observations
            for which to estimate actions.
        Returns:
            torch.Tensor: The estimated best actions with added noise, clipped
            to the action bounds.
        '''
        with torch.no_grad(): 
            actions: torch.Tensor = self.target_actor(torch.as_tensor(observations, dtype=torch.float32))
            noise: torch.Tensor = torch.normal(0, self.target_noise, size=actions.shape)
            noise = torch.clip(noise, -self.noise_clip, self.noise_clip)
            actions = torch.clip(actions + noise, -self.env_params['action_bound'], self.env_params['action_bound'])
        return actions
    
    def learning_step(self, num_steps):
        #Sampling of the minibatch
        batch = self.memory.sample(batch_size = self.batch_size)
        if num_steps % self.value_update_freq == 0:
            self.value_learning_step(batch)
        if num_steps % self.policy_update_freq == 0:
            self.policy_learning_step(batch)
        if num_steps % self.target_update_freq == 0:
            self.update_target_networks()
    
    def value_learning_step(self, batch): 
        observations, actions, rewards, dones, new_observations = batch
        
        #Computation of value targets
        with torch.no_grad():
            best_actions = self.estimate_best_actions(new_observations) # (batch_size, action_dim)
            target_values = torch.min(self.target_critic1(new_observations, best_actions), self.target_critic2(new_observations, best_actions))
            targets = rewards + (1 - dones) * self.gamma * target_values
            
        estimations1 = self.critic1(observations, actions)  
        estimations2 = self.critic2(observations, actions)
        value_loss1: torch.Tensor = self.value_loss_fn(estimations1, targets)
        value_loss2: torch.Tensor = self.value_loss_fn(estimations2, targets)
        
        self.ep_mean_value_loss += (1/self.ep)*(value_loss1.item() - self.ep_mean_value_loss) 
               
        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()
        
        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()
        
        
    def policy_learning_step(self, batch):
        observations, _, _, _, _ = batch
        
        #Don't waste computational effort
        for param in self.critic1.parameters():
            param.requires_grad = False
                
        #Policy Optimization
        estimated_actions = self.actor(observations)
        estimated_values: torch.Tensor = self.critic1(observations, estimated_actions)
        policy_loss: torch.Tensor = -estimated_values.mean() # perform gradient ascent
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        #Reactivate computational graph for critic
        for param in self.critic1.parameters():
            param.requires_grad = True
    
    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(self.target_critic1.parameters(), 
                                    self.critic1.parameters()):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)
                
            for target, online in zip(self.target_critic2.parameters(),
                                    self.critic2.parameters()):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)

            for target, online in zip(self.target_actor.parameters(), 
                                    self.actor.parameters()):
                target.data.mul(polyak)
                target.data.add((1 - polyak) * online.data)
        
    def episode_update(self):
        self.eps = max(0.1, self.eps * self.eps_decay)
        self.rewards.append(self.ep_reward)
        self.losses.append(self.ep_mean_value_loss)
        meanreward = np.mean(self.rewards)
        meanloss = np.mean(self.losses)
        print(f'\rEpisode {self.ep} Mean Reward: {meanreward:.2f} Ep_Reward: {self.ep_reward} Mean Loss: {meanloss:.2f}\t\t')
        if self.ep >= self.max_episodes:
            self.training = False
            print("\nEpisode limit reached")
        if meanreward >= self.reward_threshold:
            self.training = False
            print("\nSUCCESS!")
        self.ep += 1
                
    def evaluate(self, env = None, render:bool = True, num_ep = 3):
        mean_reward = 0.
        if env is None: env = self.env
        
        for i in range(1, num_ep+1):
            if render: print(f"Starting game {i}")

            observation = torch.FloatTensor(env.reset()[0]) 
            
            terminated = False
            truncated = False
            total_reward = 0
            
            while not terminated and not truncated:
                action = self.select_action(observation, noise_weight = 0)
                observation, reward, terminated, truncated, _ = env.step(action)
                observation = torch.FloatTensor(observation)
                total_reward += reward
                if render: self.env.render()
                
            if render: print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

        if render: print("Mean Reward: ", mean_reward)
        return mean_reward 
    
    def populate_buffer(self):    
        observation = self.env.reset()[0]
        for _ in range(self.start_steps):
            with torch.no_grad(): 
                action = self.select_action(observation, noise_weight = 1)
            new_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.store(observation,action,reward,done,new_observation)
            observation = new_observation
            if terminated or truncated: 
                observation = self.env.reset()[0]
                
    def save(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here,"models",self.name)
        
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.actor.state_dict(), open(os.path.join(path,"actor.pt"), "wb"))
        torch.save(self.critic1.state_dict(), open(os.path.join(path,"critic1.pt"), "wb"))
        torch.save(self.critic2.state_dict(), open(os.path.join(path,"critic2.pt"), "wb"))
        print("MODELS SAVED!")

    def load(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "models", self.name)
        
        self.actor.load_state_dict(torch.load(open(os.path.join(path, "actor.pt"), "rb"), weights_only=True))
        self.critic1.load_state_dict(torch.load(open(os.path.join(path, "critic1.pt"), "rb"), weights_only=True))
        self.critic2.load_state_dict(torch.load(open(os.path.join(path, "critic2.pt"), "rb"), weights_only=True))
        print("MODELS LOADED!")