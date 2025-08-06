# rl_training/utils/custom_rewards.py
import math
import numpy as np
from .centerline_helper import CenterlineHelper

class BaseReward:
    """
    Base class for reward functions. Subclasses should override the compute() method.
    """
    def __call__(self, ego_pose, opp_pose, info):
        return self.compute(ego_pose, opp_pose, info)

    def compute(self, ego_pose, opp_pose, info):
        raise NotImplementedError("compute() must be implemented in subclasses.")


class RobustProgressReward(BaseReward):
    """
    Robust reward for F1TENTH RL racing agents.
    - Rewards forward progress along the centerline.
    - Penalizes negative progress (going backwards).
    - Penalizes lateral deviation from the centerline.
    - Penalizes dawdling (no move).
    - Strongly penalizes collisions.
    - Gives a large bonus for lap completion.
    - Rewards being ahead of the opponent in arc-length (overtaking).
    """

    def __init__(self,
                 centerline_csv,
                 step_penalty=-0.05,
                 progress_scale=2.5,
                 lap_reward=10.0,
                 collision_penalty=-10.0,
                 no_move_penalty=-0.02,
                 no_move_thresh=0.01,
                 lateral_penalty=2.0,
                 overtaking_bonus=1.0,
                 no_move_limit=50):
        self.helper = CenterlineHelper(centerline_csv)
        self.step_penalty = step_penalty
        self.progress_scale = progress_scale
        self.lap_reward = lap_reward
        self.collision_penalty = collision_penalty
        self.no_move_penalty = no_move_penalty
        self.no_move_thresh = no_move_thresh
        self.lateral_penalty = lateral_penalty
        self.overtaking_bonus = overtaking_bonus
        self.no_move_limit = no_move_limit

        self.prev_progress = None
        self.prev_lap = 0
        self.no_move_steps = 0

    def reset(self, start_pose=None):
        self.prev_progress = None
        self.prev_lap = 0
        self.no_move_steps = 0

    def compute(self, ego_pose, opp_pose, info):
        ex, ey, _ = ego_pose
        ox, oy, _ = opp_pose
        progress, ego_idx = self.helper.project([ex, ey])
        opp_progress, _ = self.helper.project([ox, oy])
        reward = self.step_penalty

        # 1. Progress reward (can be negative for backward motion)
        if self.prev_progress is not None:
            delta = progress - self.prev_progress

            # Only allow wrap correction if agent is actually close to the start/end
            near_start = progress < 0.1 * self.helper.total_length
            near_end = self.prev_progress > 0.9 * self.helper.total_length
            # Or vice versa for reverse
            near_start_prev = self.prev_progress < 0.1 * self.helper.total_length
            near_end_now = progress > 0.9 * self.helper.total_length

            if delta < -0.5 * self.helper.total_length and near_start and near_end:
                delta += self.helper.total_length
                reward += self.lap_reward
            elif delta > 0.5 * self.helper.total_length and near_start_prev and near_end_now:
                delta -= self.helper.total_length

            reward += self.progress_scale * delta
 

            if abs(delta) < self.no_move_thresh:
                reward += self.no_move_penalty
                self.no_move_steps += 1
            else:
                self.no_move_steps = 0

            # Lateral deviation penalty
            lateral_dev = np.linalg.norm(self.helper.centerline[ego_idx] - np.array([ex, ey]))
            reward -= self.lateral_penalty * lateral_dev

            # Overtaking/Blocking: reward for being ahead of opponent
            # We consider wrap-around for arc-length
            rel_progress = progress - opp_progress
            if rel_progress < -0.5 * self.helper.total_length:
                rel_progress += self.helper.total_length
            elif rel_progress > 0.5 * self.helper.total_length:
                rel_progress -= self.helper.total_length
            if rel_progress > 0:
                reward += self.overtaking_bonus

        self.prev_progress = progress

        # 2. Collision penalty
        if info['collisions'][0]:
            reward += self.collision_penalty

        return reward

    def is_stuck(self):
        return self.no_move_steps >= self.no_move_limit
