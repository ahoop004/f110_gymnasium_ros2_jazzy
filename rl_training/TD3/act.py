# action_wrapper.py

import numpy as np

class ActionWrapper:
    """
    Maps agent actions from [-1, 1] into environment ranges and clips.
    Output layout matches the sim's expected inputs: (raw_steer, vel).
    """

    def __init__(self, steer_min: float, steer_max: float, vel_min: float, vel_max: float, *, clip: bool = True):
        # Validate ranges
        if not (steer_max > steer_min):
            raise ValueError("steer_max must be > steer_min")
        if not (vel_max > vel_min):
            raise ValueError("vel_max must be > vel_min")

        self.steer_min = float(steer_min)
        self.steer_max = float(steer_max)
        self.vel_min   = float(vel_min)
        self.vel_max   = float(vel_max)
        self.clip = bool(clip)

        # Precompute scales for speed
        self._steer_halfspan = 0.5 * (self.steer_max - self.steer_min)
        self._steer_center   = 0.5 * (self.steer_max + self.steer_min)
        self._vel_halfspan   = 0.5 * (self.vel_max   - self.vel_min)
        self._vel_center     = 0.5 * (self.vel_max   + self.vel_min)

    def build(self, action) -> np.ndarray:
        """
        Args:
            action: shape (2,) or (N, 2), values in [-1, 1]
                    Accepts numpy arrays; PyTorch tensors will be converted via .detach().cpu().numpy() if present.

        Returns:
            np.ndarray float32 of shape (2,) or (N, 2): [raw_steer, vel] in env units.
        """
        a = self._to_numpy(action)

        if a.ndim == 1:
            if a.shape[0] != 2:
                raise ValueError("1D action must have shape (2,), got {}".format(a.shape))
            a = a[None, :]  # (1,2)

        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("Action must have shape (2,) or (N,2), got {}".format(a.shape))

        if not np.all(np.isfinite(a)):
            raise ValueError("Action contains NaN/Inf")

        # Denormalize from [-1,1] to env units
        steer = self._steer_center + self._steer_halfspan * a[:, 0]
        vel   = self._vel_center   + self._vel_halfspan   * a[:, 1]

        if self.clip:
            steer = np.clip(steer, self.steer_min, self.steer_max)
            vel   = np.clip(vel,   self.vel_min,   self.vel_max)

        out = np.stack([steer, vel], axis=-1).astype(np.float32, copy=False)
        return out[0] if out.shape[0] == 1 else out

    @staticmethod
    def _to_numpy(x):
        """Convert numpy or torch tensor to numpy without importing torch as a hard dependency."""
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        # Light-touch torch handling to avoid import if not present
        if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
        # Try array-like conversion
        return np.asarray(x, dtype=np.float32)
