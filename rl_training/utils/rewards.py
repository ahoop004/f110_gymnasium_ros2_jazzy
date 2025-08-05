# rl_training/utils/custom_rewards.py

import numpy as np

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

        # Collision penalty
        if ego_collision:
            return self.collision_penalty

        reward = self.step_penalty
        # Bonus for being ahead of the opponent
        if ego_x > opp_x:
            reward += self.pass_bonus
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
