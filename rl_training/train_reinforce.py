import os
import yaml
import torch
import numpy as np
import gymnasium as gym
from REINFORCE.agent import REINFORCEAgent
from REINFORCE.trainer import Trainer
from utils.rewards import CenterlineProgressReward, SimplePassReward
from utils.gap_follow import gap_follow_action
from f110_gym.envs.rendering import EnvRenderer as ER


with open('/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/configs/reinforce_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

episodes = config['training_settings']['episodes']
max_steps =  config['training_settings']['max_steps']
lr = config['agent_hyperparameters']['learning_rate']
gamma = config['agent_hyperparameters']['gamma']
model_path = config['env_settings']['model_path']
start_poses = config['env_settings']['start_poses']


env = gym.make('f110_gym:f110-v0',
               render_mode='human_fast',
               map_dir=config['env_settings']['map_dir'],
               map=config['env_settings']['map'],
               map_ext=config['env_settings']['map_ext'],
               num_agents=2,
            #    integrator=Integrator.Euler
               )



obs_dim = env.observation_space.shape[1]
act_dim = env.action_space.spaces[0].shape[0]
reward_fn = SimplePassReward()
agent = REINFORCEAgent(obs_dim, 
                       act_dim, 
                       lr, 
                       gamma)


trainer = Trainer(env, agent,None ,True)


if model_path and os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    agent.load_model(model_path)
else:
    print("No model loaded; training from scratch.")

for episode in range(episodes):
    # total_reward = trainer.run_episode(
    #     config['env_settings']['start_poses'],
    #     config['training_settings']['max_steps']
    # )
    
    log_probs, rewards = [], []
    obs, info =  env.reset(options=np.array(start_poses))
    env.render()
    ego_obs = obs[0]
    opp_obs = obs[1]
    
    for step in range(max_steps):
            ego_action_raw, log_prob = agent.select_action(ego_obs)

            bounded      = np.tanh(ego_action_raw) 
            low          = env.action_space.spaces[0].low
            high         = env.action_space.spaces[0].high
            ego_action   = low + 0.5 * (bounded + 1.0) * (high - low)
            opp_action = gap_follow_action(opp_obs)

            actions = np.array([ego_action, opp_action])
            next_obs, info, terminated, truncated, info = env.step(actions)
            

            ego_pose = [info['poses_x'][0], info['poses_y'][0], info['poses_theta'][0]]
            opp_pose = [info['poses_x'][1], info['poses_y'][1], info['poses_theta'][1]]
            ego_collision = info['collisions'][0]
            
            reward = reward_fn(ego_pose, opp_pose, info)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            # ==== Early termination check ====
            if hasattr(reward_fn, 'is_stuck') and reward_fn.is_stuck():
                print(f"Early termination: agent stuck at step {step}")
                break

            ego_obs = next_obs[0]
            opp_obs = next_obs[1]

            if terminated or truncated:
                break
            env.render()
    agent.update_policy(torch.stack(log_probs), rewards)
    total_reward = sum(rewards)
    print(f"Episode {episode}, Total Reward: {total_reward}")
    
    
agent.save_model(model_path)

env.close()