import numpy as np
from typing import Optional, Literal

class ActionWrapper:
    """
    Maps agent actions from [-1, 1] into environment inputs [steer, speed].
    - mode="direct": action = [delta_norm, v_norm]
    - mode="rates" : action = [deltadot_norm, accel_norm] -> integrate with dt
                     Uses asymmetric accel limits: accel_fwd_max ( + ), accel_brake_max ( − ).

    Decision gating:
      - Fixed frame-skip: set decision_every > 1 and leave can_choose=None in build()
      - Event-driven: pass can_choose={True|False} to build() from env info
    """

    def __init__(
        self,
        steer_min: float,
        steer_max: float,
        vel_min: float,
        vel_max: float,
        *,
        clip: bool = True,
        decision_every: int = 1,           # k-step repeat; 1 = decide every step
        mode: Literal["direct", "rates"] = "direct",
        # Integration params for mode="rates"
        dt: float = 0.01,
        delta_rate_max: float = 2.5,       # rad/s   (±0.025 rad/step @ dt=0.01)
        accel_fwd_max: float = 5.0,        # m/s^2   (positive accel limit)
        accel_brake_max: float = 8.0,      # m/s^2   (magnitude for negative accel)
    ):
        # Validate ranges
        if not (steer_max > steer_min):
            raise ValueError("steer_max must be > steer_min")
        if not (vel_max > vel_min):
            raise ValueError("vel_max must be > vel_min")
        if decision_every < 1:
            raise ValueError("decision_every must be >= 1")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if accel_fwd_max <= 0 or accel_brake_max <= 0:
            raise ValueError("accel_fwd_max and accel_brake_max must be > 0")

        self.steer_min = float(steer_min)
        self.steer_max = float(steer_max)
        self.vel_min   = float(vel_min)
        self.vel_max   = float(vel_max)
        self.clip = bool(clip)

        # Precompute scales (direct mode)
        self._steer_halfspan = 0.5 * (self.steer_max - self.steer_min)
        self._steer_center   = 0.5 * (self.steer_max + self.steer_min)
        self._vel_halfspan   = 0.5 * (self.vel_max   - self.vel_min)
        self._vel_center     = 0.5 * (self.vel_max   + self.vel_min)

        # Decision gating state
        self.decision_every = int(decision_every)
        self._step_count = 0
        self._last_env_action: Optional[np.ndarray] = None  # shape (2,)

        # Mode / rate params
        self.mode = mode
        self.dt = float(dt)
        self.delta_rate_max = float(delta_rate_max)
        self.accel_fwd_max  = float(accel_fwd_max)
        self.accel_brake_max= float(accel_brake_max)

        # Internal state for rates mode (last commanded)
        self._last_cmd_steer: Optional[float] = None
        self._last_cmd_speed: Optional[float] = None

    def reset(self, *, init_steer: float = 0.0, init_speed: float = 0.0):
        """Call at env.reset(). Also resets decision gating."""
        self._step_count = 0
        self._last_env_action = None
        self._last_cmd_steer = float(np.clip(init_steer, self.steer_min, self.steer_max))
        self._last_cmd_speed = float(np.clip(init_speed, self.vel_min,   self.vel_max))

    def build(
        self,
        action,
        *,
        can_choose: Optional[bool] = None,
        force: bool = False,
        cur_steer: Optional[float] = None,   # measured current steering (if available)
        cur_speed: Optional[float] = None,   # measured current speed (recommended)
    ) -> np.ndarray:
        """
        Args:
            action: shape (2,) or (1,2), values in [-1, 1]
            can_choose: if provided and False -> reuse last action; if True -> take new
            force: override gate and accept a new action
            cur_steer/cur_speed: measured values to integrate from (used in mode="rates")

        Returns:
            np.ndarray float32 (2,) = [steer, speed] in env units.
        """
        a = self._to_numpy(action)

        # Normalize shape
        if a.ndim == 1:
            if a.shape[0] != 2:
                raise ValueError(f"1D action must have shape (2,), got {a.shape}")
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"Action must have shape (2,) or (1,2), got {a.shape}")
        if a.shape[0] != 1:
            raise ValueError("This wrapper is step-wise; pass a single action (shape (2,) or (1,2)).")
        if not np.all(np.isfinite(a)):
            raise ValueError("Action contains NaN/Inf")

        # Decide whether to accept a new action or reuse the last one
        take_new = force
        if not take_new:
            if can_choose is not None:
                take_new = bool(can_choose) or (self._last_env_action is None)
            else:
                take_new = (self._last_env_action is None) or (self._step_count % self.decision_every == 0)

        if take_new:
            if self.mode == "direct":
                # Map normalized directly to env units
                steer = self._steer_center + self._steer_halfspan * a[0, 0]
                speed = self._vel_center   + self._vel_halfspan   * a[0, 1]

            elif self.mode == "rates":
                # Map normalized to physical rates
                deltadot = self.delta_rate_max * float(a[0, 0])   # rad/s

                # Asymmetric accel: positive -> accel_fwd_max, negative -> -accel_brake_max
                accel_norm = float(a[0, 1])
                if accel_norm >= 0.0:
                    accel = accel_norm * self.accel_fwd_max        # + m/s^2
                else:
                    accel = accel_norm * self.accel_brake_max      # − m/s^2 (larger magnitude allowed)

                # Choose integration start: measured preferred, else last commanded, else centers
                if cur_steer is not None:
                    steer_prev = float(cur_steer)
                elif self._last_cmd_steer is not None:
                    steer_prev = self._last_cmd_steer
                else:
                    steer_prev = self._steer_center

                if cur_speed is not None:
                    speed_prev = float(cur_speed)
                elif self._last_cmd_speed is not None:
                    speed_prev = self._last_cmd_speed
                else:
                    speed_prev = self._vel_center

                # Integrate one step
                steer = steer_prev + deltadot * self.dt
                speed = speed_prev + accel    * self.dt

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if self.clip:
                steer = np.clip(steer, self.steer_min, self.steer_max)
                speed = np.clip(speed, self.vel_min,   self.vel_max)

            out = np.array([steer, speed], dtype=np.float32)
            self._last_env_action = out
            self._last_cmd_steer = float(out[0])
            self._last_cmd_speed = float(out[1])
        else:
            # Reuse previous env action
            out = self._last_env_action.astype(np.float32, copy=False)

        self._step_count += 1
        return out

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)
