import numpy as np
import torch
from utils.rewards import simple_pass_reward
from utils.gap_follow import gap_follow_action

class Trainer:
    def __init__(self, env, agent, gamma):
        self.env = env
        self.agent = agent
        self.gamma = gamma

    def save_model(self, path):
        self.agent.policy.save(path)
    
    def load_model(self, path, device='cpu'):
        self.agent.policy.load(path, device=device)

    def run_episode(self, start_poses, max_steps):
        obs, _ = self.env.reset(options=np.array(start_poses))
        ego_obs = obs['scans'][0]
        opp_obs = obs['scans'][1]

        log_probs, rewards = [], []
        for step in range(max_steps):
            ego_action, log_prob = self.agent.select_action(ego_obs)
            opp_action = gap_follow_action(opp_obs)

            actions = np.array([ego_action, opp_action])
            next_obs, _, terminated, truncated, _ = self.env.step(actions)

            ego_pose = [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]
            opp_pose = [obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]]
            ego_collision = obs['collisions'][0]
            
            reward = simple_pass_reward(ego_pose, opp_pose, ego_collision)
            
            
            rewards.append(reward)
            log_probs.append(log_prob)

            ego_obs = next_obs['scans'][0]
            opp_obs = next_obs['scans'][1]

            if terminated or truncated:
                break

        returns = self.compute_returns(rewards)
        self.agent.update_policy(torch.stack(log_probs), returns)

        return sum(rewards)

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns
