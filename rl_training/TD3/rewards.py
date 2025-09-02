# reward_wrapper.py
import math

class RewardWrapper:
    """
    Simple reward wrapper.
    Returns a small alive reward each step until crash,
    otherwise returns a crash penalty.
    """

    def __init__(self, alive_reward: float = 0.01, crash_penalty: float = -10.0, k_progress: float = 0.01):
        self.alive_reward = float(alive_reward)
        self.crash_penalty = float(crash_penalty)
        self.k_progress = float(k_progress)

        # Internal state for progress
        self._prev_pose = None
        
    def reset(self, observations: dict) -> None:
        """Initialize previous pose from the current observations (episode start)."""
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        self._prev_pose = (x, y, th)

    def compute(self, observations: dict) -> float:
        """
        Compute the step reward.
        Expects: 'ego_idx', 'poses_x', 'poses_y', 'poses_theta', 'collisions'
        """
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        crashed = bool(observations["collisions"][ego])

        total = 0.0
        total -= self._alive()
        total += self._crash(crashed)

        if self.k_progress > 0.0:
            total += self.k_progress * self._progress(x, y, th)

        # update state for next step
        self._prev_pose = (x, y, th)
        return float(total)

   

    def _alive(self) -> float:
        return self.alive_reward

    def _crash(self, crashed: bool) -> float:
        if crashed:
            return self.crash_penalty
        return 0.0

    def _progress(self, x: float, y: float, theta_prev: float) -> float:
        """
        Forward progress along previous heading direction.
        Returns 0.0 on the first call (no previous pose).
        """
        if self._prev_pose is None:
            return 0.0

        x_prev, y_prev, th_prev = self._prev_pose
        # displacement since last step
        dx = x - x_prev
        dy = y - y_prev
        # unit heading from previous step
        ux = math.cos(th_prev)
        uy = math.sin(th_prev)
        ds = dx * ux + dy * uy  # projection onto previous heading
        return ds if ds > 0.0 else 0.0
