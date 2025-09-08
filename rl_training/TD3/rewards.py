import math
import numpy as np
from collections import deque

class RewardWrapper:
    """
    Minimal reward shaping for 'make opp crash, stay safe':
      r = + crash_opp_bonus (only when opp newly crashes and ego didn't)
          - crash_ego_penalty (if ego crashed)
          + progress_gain * forward_progress_m
          - reverse_gain  * backward_progress_m
          - alive_cost
          - spin_penalty (if large yaw change while barely moving)
          - no_progress_penalty (if little forward progress over a short window)

    Assumes exactly 2 agents; opponent index = 1 - ego_idx.
    """

    def __init__(
        self,
        alive_cost: float = 0.005,        # per-step time cost
        crash_ego_penalty: float = -20.0,
        crash_opp_bonus: float = 100.0,   # paid once at new opp crash event
        progress_gain: float = 10.0,      # per meter (forward)
        reverse_gain: float = 5.0,       # per meter (backwards)
        # stagnation detector
        no_progress_window: int = 15,    # steps
        no_progress_eps: float = 0.10,   # meters forward within window
        no_progress_penalty: float = -6.0,
    
        spin_yaw_thresh: float = 0.20,   # rad per step (~11.5°)
        spin_move_eps: float = 0.01,     # m per step considered "not moving"
        spin_penalty: float = -0.5,
    ):
        self.alive_cost = float(alive_cost)
        self.crash_ego_penalty = float(crash_ego_penalty)
        self.crash_opp_bonus = float(crash_opp_bonus)
        self.progress_gain = float(progress_gain)
        self.reverse_gain = float(reverse_gain)

        self.no_progress_window = int(no_progress_window)
        self.no_progress_eps = float(no_progress_eps)
        self.no_progress_penalty = float(no_progress_penalty)

        self.spin_yaw_thresh = float(spin_yaw_thresh)
        self.spin_move_eps = float(spin_move_eps)
        self.spin_penalty = float(spin_penalty)

        # Internal state
        self._prev_pose = None                # (x, y, theta)
        self._prev_collisions = None          # np.array([ego, opp])
        self._ds_hist = deque(maxlen=self.no_progress_window)
        self.opp_crashed_now = False

    # ---- Public API ----
    def reset(self, observations: dict) -> None:
        ego = int(observations["ego_idx"])
        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        self._prev_pose = (x, y, th)

        try:
            c = np.asarray(observations.get("collisions", None), dtype=np.int8)
            if c.ndim == 0:
                c = np.array([int(c)], dtype=np.int8)
            self._prev_collisions = c.copy()
        except Exception:
            self._prev_collisions = None

        self._ds_hist.clear()
        self.opp_crashed_now = False

    def compute(self, observations: dict, action=None) -> float:
        ego = int(observations["ego_idx"])
        opp = 1 - ego  # 2-agent assumption

        x = float(observations["poses_x"][ego])
        y = float(observations["poses_y"][ego])
        th = float(observations["poses_theta"][ego])
        self.opp_crashed_now = False

        # Collisions array
        try:
            curr_col = np.asarray(observations["collisions"], dtype=np.int8)
            if curr_col.ndim == 0:
                curr_col = np.array([int(curr_col)], dtype=np.int8)
        except Exception:
            curr_col = None

        total = 0.0

        # 1) Crash terms
        ego_crash = bool(curr_col[ego]) if curr_col is not None else False
        opp_crash = bool(curr_col[opp]) if curr_col is not None else False

        # Pay bonus only on NEW opp crash events, and only if ego didn't crash
        if self._prev_collisions is not None and curr_col is not None:
            new_opp_crash = (curr_col[opp] == 1 and self._prev_collisions[opp] == 0)
        else:
            new_opp_crash = False

        if new_opp_crash and not ego_crash:
            total += self.crash_opp_bonus
            self.opp_crashed_now = True

        if ego_crash:
            total += self.crash_ego_penalty

        # 2) Forward / reverse progress along previous heading
        ds = 0.0
        dth = 0.0
        if self._prev_pose is not None:
            x0, y0, th0 = self._prev_pose
            dx, dy = x - x0, y - y0
            ux, uy = math.cos(th0), math.sin(th0)
            ds = dx * ux + dy * uy  # signed: +forward, -reverse
            dth = self._angle_diff(th, th0)
        # Reward forward; penalize reverse
        if ds > 0.0:
            total += self.progress_gain * ds
        elif ds < 0.0:
            total -= self.reverse_gain * (-ds)

        # 3) Spin deterrent: large yaw change while barely moving
        if abs(dth) >= self.spin_yaw_thresh and abs(ds) <= self.spin_move_eps:
            total += self.spin_penalty

        # 4) Time cost (prevents sitting still)
        total -= self.alive_cost

        # 5) No-progress penalty over a short window (prevents dithering)
        self._ds_hist.append(ds)
        if self.no_progress_penalty != 0.0 and len(self._ds_hist) == self._ds_hist.maxlen:
            recent_forward = sum(max(v, 0.0) for v in self._ds_hist)
            if recent_forward < self.no_progress_eps:
                total += self.no_progress_penalty
                self._ds_hist.clear()

        # Update internal state
        self._prev_pose = (x, y, th)
        if curr_col is not None:
            self._prev_collisions = curr_col.copy()

        return float(total)

    # ---- helpers ----
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Return signed smallest angle difference a-b in [-pi, pi]."""
        d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return d
