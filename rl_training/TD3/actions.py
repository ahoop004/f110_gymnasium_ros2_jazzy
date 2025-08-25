# actions.py
# Translate normalized actor outputs [-1, 1]^2 into env actions [steer_rad, speed_mps]
# with optional governors (e.g., clearance-based speed cap) and rate limits.
#
# Usage (noise handled in the agent, not here):
#   mapper = ActionMapper(
#       config, dt=0.01,
#       steer_bounds=(-0.4189, 0.4189),
#       speed_bounds=(0.0, 3.0)
#   )
#   mapper.reset()  # or mapper.reset(last_action_env=(0.0, 0.0))
#   a_env = mapper.map(a_norm, extras=extras, training=is_training, eval_mode=is_eval)
#
# where:
#   a_norm = np.array([steer_n, speed_n]) in [-1, 1]
#   extras = dict from observation_wrapper.build(...)[1], e.g. contains:
#       "clearance_q10_forward": float in [0,1]
#       "effective_lidar_max": float in meters

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np


def _clip(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _lerp(lo, hi, t):
    return lo + (hi - lo) * t


def _map_norm_to_env(x_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    x_norm in [-1,1] -> env units via linear map.
    """
    t = 0.5 * (x_norm + 1.0)  # -> [0,1]
    return low + t * (high - low)


class ActionMapper:
    """
    Maps normalized actor outputs to physical env actions with:
      - Optional clearance-based speed cap
      - Optional turn-dependent speed reduction
      - Optional steering & acceleration (speed) rate limits
    Exploration noise is handled by the agent before calling map() (Option A).

    Config keys used (with defaults if missing):

    action:
      train_action_noise_std: float (not used here; noise handled in agent)
      policy_noise: float (TD3 target noise; NOT used here)
      noise_clip: float (TD3 target noise clip; NOT used here)
      governors:
        enable_clearance_speed_cap: bool (default True)
        forward_clearance_q: float in [0,1] (default 0.10)  # used upstream in obs; we only consume extras
        min_clearance_m: float (default 0.60)
        min_speed_mps: float (default 0.0)
        max_speed_mps: float (default None -> fallback to speed_bounds[1])
        clearance_gain: float (default 2.0)  # speed [m/s] per meter of clearance beyond threshold
        enable_turn_speed_cap: bool (default False)
        turn_speed_gamma: float in [0,1] (default 0.0)  # scale of reduction with |steer|/steer_max
        enable_steer_rate_limit: bool (default True)
        steer_rate_limit_rad_per_step: float (default 0.02)
        enable_accel_limit: bool (default False)
        accel_limit_mps2: float (default 1.0)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        dt: float,
        steer_bounds: Tuple[float, float],
        speed_bounds: Tuple[float, float],
    ):
        self.cfg = config or {}
        self.dt = float(dt)

        # Bounds
        self.steer_min, self.steer_max = float(steer_bounds[0]), float(steer_bounds[1])
        self.speed_min, self.speed_max = float(speed_bounds[0]), float(speed_bounds[1])

        # Governors
        gov = (self.cfg.get("action") or {}).get("governors") or {}
        self.enable_clearance_cap: bool = bool(gov.get("enable_clearance_speed_cap", True))
        self.min_clearance_m: float = float(gov.get("min_clearance_m", 0.60))
        self.min_speed_mps: float = float(gov.get("min_speed_mps", 0.0))
        self.max_speed_mps: float = float(gov.get("max_speed_mps", self.speed_max))
        self.clearance_gain: float = float(gov.get("clearance_gain", 2.0))  # m/s per m beyond threshold

        self.enable_turn_cap: bool = bool(gov.get("enable_turn_speed_cap", False))
        self.turn_speed_gamma: float = float(gov.get("turn_speed_gamma", 0.0))  # 0 disables

        # Rate limits
        self.enable_steer_rate: bool = bool(gov.get("enable_steer_rate_limit", True))
        self.steer_rate_limit: float = float(gov.get("steer_rate_limit_rad_per_step", 0.02))
        self.enable_accel_limit: bool = bool(gov.get("enable_accel_limit", False))
        self.accel_limit: float = float(gov.get("accel_limit_mps2", 1.0))  # m/s^2

        # Internal state
        self._last_action_env: Optional[np.ndarray] = None  # [steer, speed]
        self._steer_span = self.steer_max - self.steer_min
        self._speed_span = self.speed_max - self.speed_min
        self._steer_max_abs = max(abs(self.steer_min), abs(self.steer_max)) or 1e-6

    def reset(self, last_action_env: Optional[Tuple[float, float]] = None) -> None:
        """Reset internal smoothing/rate-limit state."""
        if last_action_env is None:
            # Initialize to the midpoints to avoid first-step clamping artifacts
            steer0 = 0.5 * (self.steer_min + self.steer_max)
            speed0 = 0.5 * (self.speed_min + self.speed_max)
            self._last_action_env = np.array([steer0, speed0], dtype=np.float32)
        else:
            self._last_action_env = np.array([last_action_env[0], last_action_env[1]], dtype=np.float32)

    def map(
        self,
        a_norm: np.ndarray,
        *,
        extras: Optional[Dict[str, Any]] = None,
        training: bool = False,   # kept for signature symmetry; noise is NOT added here per Option A
        eval_mode: bool = False   # deterministic behavior flag (caps/limits still apply)
    ) -> np.ndarray:
        """
        Convert normalized action to env units with governors & rate limits.
        Returns np.array([steer_rad, speed_mps], dtype=float32).
        """
        # --- 0) Basic validation ---
        if not isinstance(a_norm, np.ndarray):
            a_norm = np.asarray(a_norm, dtype=np.float32)
        a_norm = a_norm.astype(np.float32, copy=False).reshape(-1)
        if a_norm.shape[0] != 2 or not np.isfinite(a_norm).all():
            # Fallback safe action
            steer_safe = 0.0
            speed_safe = max(0.0, self.speed_min)
            return np.array([steer_safe, speed_safe], dtype=np.float32)

        # --- 1) Clip normalized input (noise already added in agent) ---
        a_norm = _clip(a_norm, -1.0, 1.0)

        # --- 2) Map to env units ---
        low = np.array([self.steer_min, self.speed_min], dtype=np.float32)
        high = np.array([self.steer_max, self.speed_max], dtype=np.float32)
        a_env = _map_norm_to_env(a_norm, low, high)  # [steer, speed]

        steer, speed = float(a_env[0]), float(a_env[1])

        # --- 3) Clearance-based speed cap (optional) ---
        if self.enable_clearance_cap and extras is not None:
            clr_q = extras.get("clearance_q10_forward", None)
            eff_max = extras.get("effective_lidar_max", None)
            if isinstance(clr_q, (float, int)) and isinstance(eff_max, (float, int)) and np.isfinite(clr_q) and np.isfinite(eff_max):
                clearance_m = max(0.0, float(clr_q) * float(eff_max))
                # Thresholded linear cap: v_cap = min_speed + gain * max(0, clearance - min_clearance)
                v_cap = self.min_speed_mps + self.clearance_gain * max(0.0, clearance_m - self.min_clearance_m)
                v_cap = float(_clip(v_cap, self.min_speed_mps, self.max_speed_mps))
                speed = min(speed, v_cap)

        # --- 4) Turn-dependent speed reduction (optional) ---
        if self.enable_turn_cap and self.turn_speed_gamma > 0.0 and self._steer_max_abs > 0.0:
            steer_mag = abs(steer) / self._steer_max_abs  # in [0,1] approximately
            # Reduce speed proportionally to steering magnitude
            speed *= max(0.0, 1.0 - self.turn_speed_gamma * steer_mag)

        # --- 5) Rate limits ---
        if self._last_action_env is None:
            # If not reset() yet, initialize now to current target
            self.reset(last_action_env=(steer, speed))

        last_steer, last_speed = float(self._last_action_env[0]), float(self._last_action_env[1])

        # 5a) Steering rate limit (rad/step)
        if self.enable_steer_rate and self.steer_rate_limit > 0.0:
            d_max = float(self.steer_rate_limit)
            steer = float(_clip(steer, last_steer - d_max, last_steer + d_max))

        # 5b) Acceleration limit (m/s^2)
        if self.enable_accel_limit and self.accel_limit > 0.0 and self.dt > 0.0:
            dv_max = float(self.accel_limit * self.dt)
            speed = float(_clip(speed, last_speed - dv_max, last_speed + dv_max))

        # --- 6) Final clip to env bounds ---
        steer = float(_clip(steer, self.steer_min, self.steer_max))
        speed = float(_clip(speed, self.speed_min, self.speed_max))

        out = np.array([steer, speed], dtype=np.float32)
        self._last_action_env = out
        return out
