import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import yaml
from DQN.agent import PERAgent 
# Import simplified gap-follow policy
from utils.gap_follow import gap_follow_action
import os

with open('/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/rl_conf.yaml', 'r') as file:
    config = yaml.safe_load(file)
#PER DQN
lr = config['agent_hyperparameters']['learning_rate']
gamma = config['agent_hyperparameters']['gamma']
epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
batch_size = config['agent_hyperparameters']['batch_size']
memory_size = config['agent_hyperparameters']['memory_size']
epsilon_max = config['agent_hyperparameters']['epsilon_max']
epsilon_min = config['agent_hyperparameters']['epsilon_min']
alpha = config['per_hyperparameters']['alpha']
beta = config['per_hyperparameters']['beta']
priority_epsilon = config['per_hyperparameters']['priority_epsilon']

savepath = config['training_settings']['savepath']
loadpath = config['training_settings']['savepath']


# Env setup
env = gym.make('f110_gym:f110-v0', 
               map=config['env_settings']['map'], 
               map_ext=config['env_settings']['map_ext'], 
               num_agents=2)

obs_space_size = env.observation_space['scans'].shape[1]
action_size = env.action_space.spaces[0].shape[0]

agent = PERAgent(obs_space_size, action_size, memory_size, batch_size, gamma, lr, alpha, beta, priority_epsilon)

num_episodes = config['training_settings']['episodes']
max_steps = config['training_settings']['max_steps']
epsilon = epsilon_max

def blocking_reward(ego_x, opp_x, ego_collision):
    if ego_collision:
        return -10
    return 5 if ego_x > opp_x else -1

for episode in range(num_episodes):
    obs, info = env.reset(options=np.array([
        config['env_settings']['ego_start_pose'],
        config['env_settings']['opp_start_pose']
    ]))

    ego_obs = obs['scans'][0]
    opp_obs = obs['scans'][1]

    total_reward = 0

    for step in range(max_steps):
        # Ego action via epsilon-greedy
        if np.random.rand() < epsilon:
            ego_action = env.action_space.spaces[0].sample()
        else:
            ego_action = agent.get_action(ego_obs)

        # Opponent action
        opp_action = gap_follow_action(opp_obs)

        # Environment step
        actions = np.array([ego_action, opp_action])
        next_obs, _, terminated, truncated, info = env.step(actions)

        # Compute custom reward
        ego_reward = blocking_reward(
            next_obs['poses_x'][0], 
            next_obs['poses_x'][1], 
            next_obs['collisions'][0]
        )

        # Store experience and learn
        agent.memory.store(ego_obs, ego_action, ego_reward, next_obs['scans'][0], terminated)

        if agent.memory.size() > batch_size:
            agent.learn()

        ego_obs = next_obs['scans'][0]
        opp_obs = next_obs['scans'][1]
        total_reward += ego_reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    if (episode + 1) % config['training_settings']['save_freq'] == 0:
        agent.save(os.path.join(savepath, f'dqn_checkpoint_ep{episode+1}.pth'))

agent.save(os.path.join(savepath, 'dqn_final_model.pth'))
env.close()