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

with open('/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/rl_conf.yaml', 'r') as file:
    config = yaml.safe_load(file)

learning_rate = config['agent_hyperparameters']['learning_rate']
gamma = config['agent_hyperparameters']['gamma']
epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
batch_size = config['agent_hyperparameters']['batch_size']
memory_size = config['agent_hyperparameters']['memory_size']
epsilon_max = config['agent_hyperparameters']['epsilon_max']
epsilon_min = config['agent_hyperparameters']['epsilon_min']
savepath = config['training_settings']['savepath']
loadpath = config['training_settings']['savepath']
alpha = config['per_hyperparameters']['alpha']
beta = config['per_hyperparameters']['beta']
priority_epsilon = config['per_hyperparameters']['priority_epsilon']