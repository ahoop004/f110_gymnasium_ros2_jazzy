# reward_wrapper.py
import math
import numpy as np


class RewardWrapper:
    """
    Reward = -alive_reward (time cost)
             + crash_penalty (when crashed)
             + k_progress * forward_progress
             - k_smooth * smoothness_penalty
             + k_dir * directional_alignment

    Notes:
      - forward_progress is the projection of displacement onto previous heading, clipped at 0
      - smoothness_penalty uses L2 on action deltas: dsteer^2 + smooth_vel_weight * dvel^2
      - directional_alignment = cos(angle_to_most_open_lidar_beam) in vehicle frame
    """

    def __init__(
        self,
        alive_reward: float = 0.01,          # per-step time cost (will be subtracted)
        crash_penalty: float = -1.0,        # negative value
        k_progress: float = 0.1,            # scales forward progress (meters per step)

        k_smooth: float = 0.01,              # weight for smoothness penalty (set 0.0 to disable)
        smooth_vel_weight: float = 0.25,     # relative weight for velocity change vs steering change
        k_dir: float = 0.01,                 # weight for directional alignment (set 0.0 to disable)
        fov_rad: float = 4.7,                # LiDAR field of view in radians (e.g., ~270°)
    ):
        self.alive_reward = float(alive_reward)
        self.crash_penalty = float(crash_penalty)
        self.k_progress = float(k_progress)

        self.k_smooth = float(k_smooth)
        self.smooth_vel_weight = float(smooth_vel_weight)

        self.k_dir = float(k_dir)
        self.fov_rad = float(fov_rad)

        # Internal state
        self._prev_pose = None        # (x, y, theta)
        self._prev_action = None      # (steer_norm, vel_norm) in [-1, 1]

    # -------- Public API --------

    def reset(self, observations: dict) -> None:
        """Initialize previous pose (and clear prev action) at episode start."""
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        self._prev_pose = (x, y, th)
        self._prev_action = None

    def compute(self, observations: dict, action=None) -> float:
        """
        Compute the step reward.

        Args:
          observations: expects keys 'ego_idx', 'poses_x', 'poses_y', 'poses_theta', 'collisions', 'scans'
          action: current agent action in normalized units, shape (2,) = [steer, vel] in [-1, 1].
                  If None, smoothness term is skipped this step.

        Returns:
          float reward
        """
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        crashed = bool(observations["collisions"][ego])

        total = 0.0

        # Per-step time cost
        total -= self.alive_reward

        # Crash penalty
        if crashed:
            total += self.crash_penalty

        # Forward progress
        if self.k_progress > 0.0:
            total += self.k_progress * self._progress(x, y)

        # Directional alignment (LiDAR-based)
        if self.k_dir > 0.0:
            scan = np.asarray(observations["scans"][ego], dtype=np.float32)
            total += self.k_dir * self._directional_alignment(scan)

        # Smoothness (action change penalty)
        if self.k_smooth > 0.0 and action is not None:
            total -= self.k_smooth * self._smoothness_penalty(action)

        # Update internal state
        self._prev_pose = (x, y, th)
        if action is not None:
            self._prev_action = self._to_action_array(action)

        return float(total)

    # -------- Private helpers --------

    def _progress(self, x: float, y: float) -> float:
        """Forward progress along previous heading direction (>= 0)."""
        if self._prev_pose is None:
            return 0.0
        x_prev, y_prev, th_prev = self._prev_pose
        dx = x - x_prev
        dy = y - y_prev
        ux = math.cos(th_prev)
        uy = math.sin(th_prev)
        ds = dx * ux + dy * uy
        return ds if ds > 0.0 else 0.0

    def _smoothness_penalty(self, action) -> float:
        """
        L2 penalty on action changes in normalized space.
        Uses only current action if no previous action is available (penalty=0).
        """
        a = self._to_action_array(action)
        if self._prev_action is None or a.shape != (2,):
            return 0.0
        dsteer = float(a[0] - self._prev_action[0])
        dvel   = float(a[1] - self._prev_action[1])
        return dsteer * dsteer + self.smooth_vel_weight * dvel * dvel

    def _directional_alignment(self, scan: np.ndarray) -> float:
        """
        Reward alignment with the most open LiDAR direction.
        Returns cos(alpha*), where alpha* is the angle (in vehicle frame) of the max-range beam.
        """
        if scan.ndim != 1 or scan.size == 0:
            return 0.0
        n = scan.size
        # Map index -> angle assuming symmetric FOV around 0
        angle_min = -self.fov_rad / 2.0
        angle_inc = self.fov_rad / max(1, n - 1)
        idx = int(np.argmax(scan))
        alpha = angle_min + idx * angle_inc
        # Alignment with forward axis is cos(alpha)
        return float(math.cos(alpha))

    @staticmethod
    def _to_action_array(action):
        """Convert action-like to a flat np.array([steer, vel], dtype=float32)."""
        if isinstance(action, np.ndarray):
            arr = action
        elif hasattr(action, "detach") and hasattr(action, "cpu") and hasattr(action, "numpy"):
            arr = action.detach().cpu().numpy()
        else:
            arr = np.asarray(action)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"action must have 2 elements [steer, vel], got shape {arr.shape}")
        return arr
