import numpy as np
import torch
from utils.rewards import simple_pass_reward
from utils.gap_follow import gap_follow_action

class Trainer:
    def __init__(self, env, agent, gamma, render=False):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.render = render

        if self.render:
            # Unwrap to raw F110Env
            unwrapped_env = self.env
            while hasattr(unwrapped_env, "env"):
                unwrapped_env = unwrapped_env.env
            unwrapped_env.add_render_callback(self.render_callback)
        
    def render_callback(self,env_renderer):
    # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].position[::2]
        y = e.cars[0].position[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

    def save_model(self, path):
        self.agent.policy.save(path)
    
    def load_model(self, path, device='cpu'):
        self.agent.policy.load(path, device=device)

    def run_episode(self, start_poses, max_steps):
        obs, info = self.env.reset(options=np.array(start_poses))
        ego_obs = obs[0]
        opp_obs = obs[1]

        log_probs, rewards = [], []
        self.env.render()
        for step in range(max_steps):
            ego_action, log_prob = self.agent.select_action(ego_obs)
            opp_action = gap_follow_action(opp_obs)

            actions = np.array([ego_action, opp_action])
            next_obs, info, terminated, truncated, info = self.env.step(actions)
            

            ego_pose = [info['poses_x'][0], info['poses_y'][0], info['poses_theta'][0]]
            opp_pose = [info['poses_x'][1], info['poses_y'][1], info['poses_theta'][1]]
            ego_collision = info['collisions'][0]
            
            reward = simple_pass_reward(ego_pose, opp_pose, ego_collision)
            
            
            rewards.append(reward)
            log_probs.append(log_prob)

            ego_obs = next_obs[0]
            opp_obs = next_obs[1]

            if terminated or truncated:
                break
            self.env.render()

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
