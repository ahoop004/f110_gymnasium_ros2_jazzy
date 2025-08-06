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


class SimplePassReward(BaseReward):
    """
    Reward for staying in front of the opponent and penalizing collisions.
    """
    def __init__(self, step_penalty=-0.01, pass_bonus=1.0, collision_penalty=-10.0):
        self.step_penalty = step_penalty
        self.pass_bonus = pass_bonus
        self.collision_penalty = collision_penalty

    def compute(self, ego_pose, opp_pose, info):
        ego_x = ego_pose[0]
        opp_x = opp_pose[0]
        # info['collisions'] is a list with 0/1 values for each agent
        ego_collision = info['collisions'][0]
        reward = 0.0
        # Collision penalty
        if ego_collision:
            return self.collision_penalty

        reward = self.step_penalty
        # Bonus for being ahead of the opponent
        if ego_x > opp_x:
            reward += self.pass_bonus
        elif ego_x < opp_x:
            reward += self.step_penalty
        return reward


class DistanceReward(BaseReward):
    """
    Reward that encourages the ego car to maximize its distance from the opponent.
    """
    def __init__(self, step_penalty=-0.01, scaling=0.1):
        self.step_penalty = step_penalty
        self.scaling = scaling

    def compute(self, ego_pose, opp_pose, info):
        ex, ey, _ = ego_pose
        ox, oy, _ = opp_pose

        # Euclidean distance between ego and opponent
        dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)
        reward = self.step_penalty + self.scaling * dist
        return reward
class FastLapReward(BaseReward):
    """
    Reward designed to minimize lap time.
    - step_penalty: negative reward each step to discourage taking too long.
    - lap_reward: large positive reward for completing a lap (lap_count increment).
    - progress_scale: scales the incremental distance travelled each step to encourage speed.
    - collision_penalty: heavy negative penalty for collisions.
    """
    def __init__(self, step_penalty=-0.05, lap_reward=10.0,
                 progress_scale=1.0, collision_penalty=-10.0, spin_penalty=0.5):
        self.step_penalty = step_penalty
        self.lap_reward = lap_reward
        self.progress_scale = progress_scale
        self.collision_penalty = collision_penalty
        self.prev_lap_count = 0
        self.prev_position = None  # to measure progress
        self.spin_penalty = spin_penalty

    def compute(self, ego_pose, opp_pose, info):
        ex, ey, _ = ego_pose
        reward = self.step_penalty
        # yaw_rate = abs(info['ang_vels_z'][0])
        # reward -= self.spin_penalty * yaw_rate

        # Reward distance travelled since last step (approximates speed/progress)
        if self.prev_position is not None:
            dx = ex - self.prev_position[0]
            dy = ey - self.prev_position[1]
            dist = math.hypot(dx, dy)
            reward += self.progress_scale * dist
        self.prev_position = (ex, ey)

        # Lap completion bonus
        current_lap = int(info['lap_counts'][0])
        if current_lap > self.prev_lap_count:
            reward += self.lap_reward
            self.prev_lap_count = current_lap

        # Collision penalty
        if info['collisions'][0]:
            reward += self.collision_penalty

        return reward
    
class CenterlineProgressReward(BaseReward):
    def __init__(self, centerline_csv, step_penalty=-0.05, progress_scale=1.0, lap_reward=10.0,
                 collision_penalty=-10.0, no_move_penalty=-0.1, no_move_thresh=0.01):
        self.helper = CenterlineHelper(centerline_csv)
        self.step_penalty = step_penalty
        self.progress_scale = progress_scale
        self.lap_reward = lap_reward
        self.collision_penalty = collision_penalty
        self.no_move_penalty = no_move_penalty
        self.no_move_thresh = no_move_thresh
        self.prev_progress = None
        self.prev_lap = 0

    def reset(self, start_pose=None):
        self.prev_progress = None
        self.prev_lap = 0

    def compute(self, ego_pose, opp_pose, info):
        ex, ey, _ = ego_pose
        progress, _ = self.helper.project([ex, ey])
        reward = self.step_penalty

        if self.prev_progress is not None:
            delta = progress - self.prev_progress
            if delta < -0.5 * self.helper.total_length:
                delta += self.helper.total_length
                reward += self.lap_reward
            elif delta > 0.5 * self.helper.total_length:
                delta -= self.helper.total_length

            if abs(delta) < self.no_move_thresh:
                reward += self.no_move_penalty
            reward += self.progress_scale * max(0, delta)

        self.prev_progress = progress

        if info['collisions'][0]:
            reward += self.collision_penalty

        return reward
