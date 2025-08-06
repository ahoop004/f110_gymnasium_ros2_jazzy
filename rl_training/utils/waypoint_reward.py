"""
Waypoint following reward for F1TENTH RL training.

This reward encourages the agent to sequentially reach a set of waypoints loaded
from a CSV file. Once a waypoint is passed within a specified distance
threshold, the reward marks it as passed and increments the current target.
In conjunction with a rendering callback (see rendering_waypoints.py), the
passed waypoints can be visualized in red, while the first unpassed waypoint
is shown in white and the remaining unpassed waypoints are shown in yellow.

Usage:

    from waypoint_reward import WaypointFollowReward
    reward_fn = WaypointFollowReward("/path/to/waypoints.csv")
    # During training, call reward = reward_fn(ego_pose, opp_pose, info)
    # and use reward_fn.passed_flags to update rendering colors.

"""

import numpy as np


class WaypointFollowReward:
    """Reward function for following a sequence of waypoints.

    The reward encourages progress toward the next waypoint and penalizes lack
    of progress or collisions. When a waypoint is reached (within a threshold
    distance), it is marked as passed and the next waypoint becomes the target.

    Attributes:
        waypoints (np.ndarray): Array of waypoint coordinates of shape (N, 2).
        threshold (float): Distance threshold for considering a waypoint passed.
        step_penalty (float): Penalty applied each step to encourage movement.
        progress_scale (float): Scaling factor for progress toward waypoint.
        collision_penalty (float): Penalty for collision events.
        pass_bonus (float): Bonus reward for passing a waypoint.
        current_idx (int): Index of the current target waypoint.
        prev_distance (float or None): Distance to the target waypoint from the previous step.
        passed_flags (list[bool]): Flags indicating which waypoints have been passed.
    """

    def __init__(
        self,
        waypoints_csv_path: str,
        threshold: float = 1.0,
        step_penalty: float = -0.01,
        progress_scale: float = 1.0,
        collision_penalty: float = -5.0,
        pass_bonus: float = 5.0,
    ) -> None:
        """
        Initialize the reward function.

        Args:
            waypoints_csv_path: Path to a CSV file containing waypoint coordinates
                without a header. The CSV should have at least two columns (x, y).
            threshold: Distance (in meters) within which a waypoint is considered passed.
            step_penalty: Constant penalty applied every step to discourage idling.
            progress_scale: Factor for scaling progress toward the target waypoint.
            collision_penalty: Penalty applied when a collision occurs (taken from info['collisions']).
            pass_bonus: Bonus reward applied when the agent passes a waypoint.
        """
        # Load waypoints from CSV
        coords = []
        with open(waypoints_csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                coords.append([x, y])
        if not coords:
            raise ValueError(
                f"No valid waypoint coordinates found in {waypoints_csv_path}."
            )
        self.waypoints = np.array(coords)
        self.threshold = threshold
        self.step_penalty = step_penalty
        self.progress_scale = progress_scale
        self.collision_penalty = collision_penalty
        self.pass_bonus = pass_bonus
        # State variables
        self.current_idx = 0
        self.prev_distance = None
        self.passed_flags = [False] * len(self.waypoints)

    def reset(self) -> None:
        """Reset the internal state for a new episode."""
        self.current_idx = 0
        self.prev_distance = None
        self.passed_flags = [False] * len(self.waypoints)

    def __call__(self, ego_pose, opp_pose, info) -> float:
        """Compute the waypoint following reward.

        Args:
            ego_pose: List or tuple (x, y, theta) for the ego vehicle.
            opp_pose: List or tuple (x, y, theta) for the opponent vehicle.
            info: Dictionary containing environment info, must include
                info['collisions'][0] for collision status.

        Returns:
            float: The computed reward for this step.
        """
        ex, ey, _ = ego_pose
        # Compute distance to the current target waypoint
        target = self.waypoints[self.current_idx]
        dist = np.linalg.norm(np.array([ex, ey]) - target)
        reward = self.step_penalty

        # Progress toward waypoint: reward positive for reducing distance, negative for increasing
        if self.prev_distance is not None:
            delta = self.prev_distance - dist
            reward += self.progress_scale * delta
        self.prev_distance = dist

        # Check if waypoint is passed
        if dist < self.threshold:
            if not self.passed_flags[self.current_idx]:
                # Mark waypoint as passed and award bonus
                self.passed_flags[self.current_idx] = True
                reward += self.pass_bonus
            # Advance to the next waypoint (do not wrap around by default)
            if self.current_idx < len(self.waypoints) - 1:
                self.current_idx += 1
                self.prev_distance = np.linalg.norm(
                    np.array([ex, ey]) - self.waypoints[self.current_idx]
                )

        # Apply collision penalty
        if info.get("collisions") is not None:
            # info['collisions'] is expected to be a sequence per agent
            collision_flags = info["collisions"]
            if len(collision_flags) > 0 and collision_flags[0]:
                reward += self.collision_penalty

        return reward