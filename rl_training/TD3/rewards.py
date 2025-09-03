# reward_wrapper.py
import math
from collections import deque
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
        k_progress: float = 0.01,            # scales forward progress (meters per step)
        k_opp_crash: float = 10.0,

        k_smooth: float = 0.01,              # weight for smoothness penalty (set 0.0 to disable)
        smooth_vel_weight: float = 0.25,     # relative weight for velocity change vs steering change
        k_dir: float = 0.01,                 # weight for directional alignment (set 0.0 to disable)
        fov_rad: float = 4.7,                # LiDAR field of view in radians (e.g., ~270°)
        
        k_reverse: float = 0.2,          # penalty per meter of backward progress (>=0)
        no_progress_window: int = 20,    # steps to look back (e.g., 20 @ 100 Hz = 0.2 s)
        no_progress_eps: float = 0.02,   # meters of forward progress threshold over window
        no_progress_penalty: float = -0.5,
    ):
        self.alive_reward = float(alive_reward)
        self.crash_penalty = float(crash_penalty)
        self.k_progress = float(k_progress)
        self.k_opp_crash = float(k_opp_crash)
         

        self.k_smooth = float(k_smooth)
        self.smooth_vel_weight = float(smooth_vel_weight)

        self.k_dir = float(k_dir)
        self.fov_rad = float(fov_rad)
        self.k_reverse = float(k_reverse)
        self.no_progress_window = int(no_progress_window)
        self.no_progress_eps = float(no_progress_eps)
        self.no_progress_penalty = float(no_progress_penalty)

        # Internal state
        self._prev_pose = None        # (x, y, theta)
        self._prev_action = None      # (steer_norm, vel_norm) in [-1, 1]
        self._ds_hist = deque(maxlen=self.no_progress_window)
        self._prev_collisions = None

    # -------- Public API --------

    def reset(self, observations: dict) -> None:
        """Initialize previous pose (and clear prev action) at episode start."""
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        self._prev_pose = (x, y, th)
        self._prev_action = None
        if "collisions" in observations:
            try:
                self._prev_collisions = np.asarray(observations["collisions"], dtype=np.int8)
            except Exception:
                self._prev_collisions = None
        else:
            self._prev_collisions = None
        self._ds_hist.clear()

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
        if self.k_opp_crash != 0.0 and "collisions" in observations:
            try:
                curr = np.asarray(observations["collisions"], dtype=np.int8)
                if curr.ndim == 0:
                    curr = np.array([int(curr)], dtype=np.int8)

                ego_idx = int(observations["ego_idx"])
                num_agents = int(curr.size)

                # Detect new crashes for *non-ego* agents
                if self._prev_collisions is None or self._prev_collisions.size != num_agents:
                    prev = np.zeros_like(curr)
                else:
                    prev = self._prev_collisions

                # new event = currently collided AND previously not collided
                new_events = (curr.astype(bool) & ~prev.astype(bool))

                # count only opponents
                opp_mask = np.ones(num_agents, dtype=bool)
                opp_mask[ego_idx] = False

                new_opp_crashes = int(np.sum(new_events & opp_mask))

                if new_opp_crashes > 0:
                    total += self.k_opp_crash * float(new_opp_crashes)
            except Exception:
                pass

        total = 0.0

        # Per-step time cost
        total -= self.alive_reward

        # Crash penalty
        if crashed:
            total += self.crash_penalty

        # Forward progress
        # Forward / reverse progress (signed)
        ds = self._progress_signed(x, y)  # meters along previous heading (+ forward, - backward)
        if self.k_progress > 0.0:
            if ds > 0.0:
                total += self.k_progress * ds
            elif ds < 0.0 and self.k_reverse > 0.0:
                total -= self.k_reverse * (-ds)  # penalize reverse distance)

        # Directional alignment (LiDAR-based)
        if self.k_dir > 0.0:
            scan = np.asarray(observations["scans"][ego], dtype=np.float32)
            total += self.k_dir * self._directional_alignment(scan)
        self._ds_hist.append(ds)
        
        if self.no_progress_penalty != 0.0 and len(self._ds_hist) == self._ds_hist.maxlen:
            recent_forward = sum(max(v, 0.0) for v in self._ds_hist)  # only count forward meters
            if recent_forward < self.no_progress_eps:
                total += self.no_progress_penalty
                # Optional: clear to avoid re-triggering every step
                self._ds_hist.clear()

        # Smoothness (action change penalty)
        if self.k_smooth > 0.0 and action is not None:
            total -= self.k_smooth * self._smoothness_penalty(action)

        # Update internal state
        self._prev_pose = (x, y, th)
        if action is not None:
            self._prev_action = self._to_action_array(action)
            
        if "collisions" in observations:
            try:
                self._prev_collisions = np.asarray(observations["collisions"], dtype=np.int8).copy()
            except Exception:
                self._prev_collisions = None

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
    
    def _progress_signed(self, x: float, y: float) -> float:
        """Signed forward progress along the previous heading (meters). >0 forward, <0 backward."""
        if self._prev_pose is None:
            return 0.0
        x_prev, y_prev, th_prev = self._prev_pose
        dx = x - x_prev
        dy = y - y_prev
        ux = math.cos(th_prev)
        uy = math.sin(th_prev)
        return dx * ux + dy * uy

