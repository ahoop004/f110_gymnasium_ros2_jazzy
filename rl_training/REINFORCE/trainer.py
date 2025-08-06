import numpy as np
import torch
from utils.rewards import CenterlineProgressReward
from utils.gap_follow import gap_follow_action
from f110_gym.envs.rendering import EnvRenderer as ER

class Trainer:
    def __init__(self, env, agent, gamma, reward_fn=None, render=False):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.render = render
        self.reward_fn = CenterlineProgressReward("/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/maps/cenerlines/Shanghai_waypoints.csv")

        if self.render:
            # Unwrap to raw F110Env
            unwrapped_env = self.env
            while hasattr(unwrapped_env, "env"):
                unwrapped_env = unwrapped_env.env
            unwrapped_env.add_render_callback(self.render_callback)
            centerline_callback = ER.make_waypoints_callback("/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/maps/cenerlines/Shanghai_waypoints.csv")
            unwrapped_env.add_render_callback(centerline_callback)
        
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
        self.reward_fn.reset()
        self.env.render()
        for step in range(max_steps):
            ego_action_raw, log_prob = self.agent.select_action(ego_obs)

            bounded      = np.tanh(ego_action_raw) 
            low          = self.env.action_space.spaces[0].low
            high         = self.env.action_space.spaces[0].high
            ego_action   = low + 0.5 * (bounded + 1.0) * (high - low)
            
            
            
            opp_action = gap_follow_action(opp_obs)

            actions = np.array([ego_action, opp_action])
            next_obs, info, terminated, truncated, info = self.env.step(actions)
            

            ego_pose = [info['poses_x'][0], info['poses_y'][0], info['poses_theta'][0]]
            opp_pose = [info['poses_x'][1], info['poses_y'][1], info['poses_theta'][1]]
            ego_collision = info['collisions'][0]
            
            reward = self.reward_fn(ego_pose, opp_pose, info)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            # ==== Early termination check ====
            if hasattr(self.reward_fn, 'is_stuck') and self.reward_fn.is_stuck():
                print(f"Early termination: agent stuck at step {step}")
                break

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
