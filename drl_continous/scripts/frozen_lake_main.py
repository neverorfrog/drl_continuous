from drl_continous.environments import ContinuousFrozenLake
from drl_continous.agents import SAC
from drl_continous.utils.definitions import RewardType
import wandb

env_reward_type = RewardType.dense
map_name = "lab"
experiment_name = f"sac-{env_reward_type.name}-{map_name}"

wandb.init(project="drl-continuous-frozenlake", group=experiment_name)
env = ContinuousFrozenLake(map_name = "lab", reward_type = env_reward_type, is_rendered = False, is_slippery = False)
agent = SAC(name = experiment_name, env = env, max_episodes = 3000)
agent.train()
agent.save()
env.is_rendered = True
agent.evaluate()