import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

def network(sizes, activation, output_activation = nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        activation = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation()]
    return nn.Sequential(*layers)  
  
class Actor(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, env_params, hidden_dims=(256,256), activation=nn.GELU):
        super().__init__()
        # dimensions
        dimensions = [env_params['obs_dim']] + list(hidden_dims) + [env_params['action_dim']]
        self.pi = network(dimensions, activation, nn.Tanh)
        self.action_bound = env_params['action_bound']
        
    def forward(self, obs):    
        # Return output from network scaled to action space limits.
        return self.action_bound * self.pi(obs)
 
LOG_STD_MAX = 2
LOG_STD_MIN = -20   

class SquashedGaussianActor(nn.Module):
    """Stochastic Policy Network."""
    def __init__(self, env_params, hidden_dims=(512,512), activation=nn.GELU):
        super().__init__()
        # dimensions
        dimensions = [env_params['obs_dim']] + list(hidden_dims)
        self.pi = network(dimensions, activation)
        self.action_bound = env_params['action_bound']
        self.mu_layer = nn.Linear(dimensions[-1], env_params['action_dim'])
        self.log_std_layer = nn.Linear(dimensions[-1], env_params['action_dim'])
        
    
    def forward(self, obs, deterministic = False, with_logprob = True):
        output = self.pi(obs)
        mu = self.mu_layer(output)
        log_std = self.log_std_layer(output)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        pi_distribution = torch.distributions.Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = pi_distribution.rsample()
            
        if with_logprob:
            logp_pi = pi_distribution.log_prob(action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
        else:
            logp_pi = None
            
        action = torch.tanh(action)
        action = self.action_bound * action
        
        return action, logp_pi
        
        
class Critic(nn.Module):
    """Parametrized Q Network."""

    def __init__(self, env_params, hidden_dims=(512,512), activation=nn.GELU):
        super().__init__()
        # dimensions
        dimensions = [env_params['obs_dim'] + env_params['action_dim']] + list(hidden_dims) + [1]
        self.q = network(dimensions, activation)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))    