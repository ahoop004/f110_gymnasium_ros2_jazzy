# observation.py
# Build a flat, ego-centric observation vector from the env's dict obs.
# Usage:
#   obs_w = observation_wrapper(config, lidar_beams=1080, fov=4.7)
#   obs_w.reset(first_obs_dict)
#   vec, extras = obs_w.build(step_obs_dict, last_action=(steer, speed), eval_mode=False)

from __future__ import annotations
import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np


class observation_wrapper:
    """
    Converts the environment's dict observation into a flat vector for RL.

    Features (configurable):
      - LiDAR normalization: fixed (e.g., 30.0) or auto-calibrated effective max.
      - Optional LiDAR downsampling via pooling and optional history stacking (k frames).
      - Ego scalars: speed, yaw_rate, sin(theta), cos(theta), last_action (optional).
      - Opponent-relative scalars in ego-frame: dist, sin(bearing), cos(bearing),
        relative speed along line-of-sight, sin(dtheta), cos(dtheta).
      - Clearance summaries: q10 in forward/left/right sectors (from full LiDAR).
      - Save/restore internal state (effective LiDAR max, history).

    Expected env obs_dict keys (per step):
      'ego_idx' (int),
      'scans' (num_agents, beams),
      'poses_x', 'poses_y', 'poses_theta' : each (num_agents,),
      'linear_vels_x', 'linear_vels_y'    : each (num_agents,),
      'ang_vels_z'                        : (num_agents,)

    The wrapper is *purely* observational — it does not modify the env.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        lidar_beams: int = 1080,
        fov: float = 4.7,
        progress: Optional[Any] = None,  # keep for future centerline features
    ):
        self.cfg = self._with_defaults(config)
        self.progress = progress

        # LiDAR geometry
        self.beams = int(lidar_beams)
        self.fov = float(fov)
        # Beam angles from -FOV/2 to +FOV/2 (0 ≈ straight ahead)
        self._angles = np.linspace(-self.fov / 2.0, self.fov / 2.0, self.beams, dtype=np.float32)

        # Downsampling (pooling)
        self.pool = int(self.cfg["obs"]["lidar_downsample"])
        if self.pool < 1:
            raise ValueError("lidar_downsample must be >= 1")
        if self.beams % self.pool != 0:
            raise ValueError(f"lidar_beams ({self.beams}) must be divisible by lidar_downsample ({self.pool})")
        self._pooled_len = self.beams // self.pool

        # History of (normalized, pooled) LiDAR frames
        self.k_hist = int(self.cfg["obs"]["history_k"])
        if self.k_hist < 1:
            raise ValueError("history_k must be >= 1")
        self._lidar_hist: deque[np.ndarray] = deque(maxlen=self.k_hist)

        # LiDAR normalization mode/state
        self._lidar_mode = str(self.cfg["obs"]["lidar_max_mode"]).lower()  # 'fixed' or 'auto'
        self._fixed_max = float(self.cfg["obs"]["lidar_fixed_max"])        # e.g., 30.0
        self._min_eff_max = float(self.cfg["obs"]["lidar_min_effective"])  # safety floor, e.g., 10.0
        self._auto_percentile = float(self.cfg["obs"]["lidar_auto_percentile"])  # e.g., 99.0
        self._eff_max = self._fixed_max if self._lidar_mode == "fixed" else max(self._min_eff_max, 0.9 * self._fixed_max)
        self._eff_ema_alpha = float(self.cfg["obs"]["lidar_effective_ema"])  # smoothing for auto-effective max
        self._freeze = False  # set True during eval to freeze normalization state

        # Scalar normalization bounds
        self.v_pos_max = float(self.cfg["obs"]["v_pos_max"])            # speed normalizer (m/s)
        self.yaw_rate_norm = float(self.cfg["obs"]["yaw_rate_norm"])    # e.g., 10.0 (matches env bounds)
        # Optional last_action normalization (if you pass env-physical actions)
        self.steer_max_abs = float(self.cfg["obs"]["steer_max_abs"])    # radians
        self.speed_max = float(self.cfg["obs"]["speed_max"])            # m/s
        self.last_action_units = str(self.cfg["obs"]["last_action_units"]).lower()  # 'env' or 'normalized'

        # Which scalar groups to include
        scal_cfg = self.cfg["obs"]["include"]
        self.include_last_action = bool(scal_cfg["last_action"])
        self.include_clearance = bool(scal_cfg["clearance"])
        self.include_opp = bool(scal_cfg["opponent"])
        self.include_track = bool(scal_cfg["track"]) and (self.progress is not None)  # only if supplied

        # Sector definitions (degrees)
        sec_cfg = self.cfg["obs"]["sectors"]
        self._fwd_half_deg = float(sec_cfg["forward_half_deg"])
        self._side_lo_deg = float(sec_cfg["side_low_deg"])
        self._side_hi_deg = float(sec_cfg["side_high_deg"])
        self._fwd_idx = self._sector_indices(-self._fwd_half_deg, +self._fwd_half_deg)
        self._left_idx = self._sector_indices(self._side_lo_deg, self._side_hi_deg)
        self._right_idx = self._sector_indices(-self._side_hi_deg, -self._side_lo_deg)

        # Pre-compute scalar feature length
        self._scalar_len = self._compute_scalar_len()

    # ---------------- Public API ----------------

    def reset(self, obs_dict: Optional[Dict[str, Any]] = None) -> None:
        """Clear history and, optionally, warmup auto-cal on first obs."""
        self._lidar_hist.clear()
        if obs_dict is not None:
            # Warmup one frame so history stack doesn't start empty
            ego_idx = int(obs_dict.get("ego_idx", 0))
            raw_scan = np.asarray(obs_dict["scans"][ego_idx], dtype=np.float32)
            norm_full, norm_pooled = self._normalize_lidar(raw_scan, eval_mode=False)
            self._lidar_hist.append(norm_pooled.copy())

    def set_eval(self, freeze: bool) -> None:
        """Freeze or un-freeze normalization/auto-cal (use True during eval)."""
        self._freeze = bool(freeze)

    def obs_dim(self) -> int:
        """Return the length of the produced observation vector."""
        return self._pooled_len * self.k_hist + self._scalar_len

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state (effective max and lidar history)."""
        return {
            "effective_lidar_max": float(self._eff_max),
            "freeze": bool(self._freeze),
            "lidar_hist": [arr.tolist() for arr in list(self._lidar_hist)],
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from get_state()."""
        if "effective_lidar_max" in state:
            self._eff_max = float(state["effective_lidar_max"])
        if "freeze" in state:
            self._freeze = bool(state["freeze"])
        self._lidar_hist.clear()
        for arr in state.get("lidar_hist", []):
            arr_np = np.asarray(arr, dtype=np.float32)
            if arr_np.shape != (self._pooled_len,):
                # Ignore malformed history entries
                continue
            self._lidar_hist.append(arr_np)

    def build(
        self,
        obs_dict: Dict[str, Any],
        *,
        last_action: Optional[Tuple[float, float]] = None,
        eval_mode: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build the flat observation vector.

        Args:
            obs_dict: raw dict from env.step/reset
            last_action: optional (steer, speed) from previous step
            eval_mode: set True to avoid updating auto-cal and normalizers

        Returns:
            obs_vec: 1-D float32 vector
            extras: dict of diagnostics (effective_lidar_max, sector quantiles, etc.)
        """
        ego_idx = int(obs_dict.get("ego_idx", 0))
        scans = np.asarray(obs_dict["scans"], dtype=np.float32)
        poses_x = np.asarray(obs_dict["poses_x"], dtype=np.float32)
        poses_y = np.asarray(obs_dict["poses_y"], dtype=np.float32)
        poses_th = np.asarray(obs_dict["poses_theta"], dtype=np.float32)
        vxs = np.asarray(obs_dict["linear_vels_x"], dtype=np.float32)
        vys = np.asarray(obs_dict["linear_vels_y"], dtype=np.float32) if "linear_vels_y" in obs_dict else np.zeros_like(vxs)
        wzs = np.asarray(obs_dict["ang_vels_z"], dtype=np.float32)

        # --- LiDAR normalization & pooling ---
        raw_scan = scans[ego_idx]
        norm_full, norm_pooled = self._normalize_lidar(raw_scan, eval_mode=eval_mode or self._freeze)

        # Update history, then stack
        self._maybe_init_hist(norm_pooled)
        self._lidar_hist.append(norm_pooled)
        if self.k_hist == 1:
            lidar_feat = norm_pooled
        else:
            # Stack older -> newer (temporal order)
            lidar_feat = np.concatenate(list(self._lidar_hist), axis=0)

        # --- Ego scalars ---
        vx_ego, vy_ego = float(vxs[ego_idx]), float(vys[ego_idx])
        theta_ego = float(poses_th[ego_idx])
        speed = math.hypot(vx_ego, vy_ego) / max(1e-6, self.v_pos_max)
        speed = np.clip(speed, 0.0, 1.0)
        yaw_rate = float(wzs[ego_idx]) / max(1e-6, self.yaw_rate_norm)
        yaw_rate = float(np.clip(yaw_rate, -1.0, 1.0))
        sin_th, cos_th = math.sin(theta_ego), math.cos(theta_ego)

        scalars = [speed, yaw_rate, sin_th, cos_th]

        # Last action (optional)
        if self.include_last_action:
            if last_action is None:
                scalars.extend([0.0, 0.0])
            else:
                steer_prev, speed_prev = float(last_action[0]), float(last_action[1])
                if self.last_action_units == "env":
                    # Normalize by provided physical bounds
                    steer_n = np.clip(steer_prev / max(1e-6, self.steer_max_abs), -1.0, 1.0)
                    speed_n = np.clip(speed_prev / max(1e-6, self.speed_max), -1.0, 1.0)
                else:
                    # Already normalized (e.g., actor output in [-1,1])
                    steer_n = np.clip(steer_prev, -1.0, 1.0)
                    speed_n = np.clip(speed_prev, -1.0, 1.0)
                scalars.extend([float(steer_n), float(speed_n)])

        # --- Opponent-relative (optional) ---
        opp_added = False
        if self.include_opp and scans.shape[0] >= 2:
            # pick the first non-ego as opponent
            opp_idx = 1 if ego_idx == 0 else 0
            if opp_idx != ego_idx and opp_idx < scans.shape[0]:
                x_ego, y_ego = float(poses_x[ego_idx]), float(poses_y[ego_idx])
                x_opp, y_opp = float(poses_x[opp_idx]), float(poses_y[opp_idx])
                dx_w, dy_w = (x_opp - x_ego), (y_opp - y_ego)
                # ego-frame rotation: R(-theta_ego) * [dx, dy]
                c, s = math.cos(theta_ego), math.sin(theta_ego)
                dx_e = c * dx_w + s * dy_w
                dy_e = -s * dx_w + c * dy_w
                dist = math.hypot(dx_w, dy_w)
                # bearing in ego frame
                bearing = math.atan2(dy_e, dx_e)
                sin_b, cos_b = math.sin(bearing), math.cos(bearing)

                # relative LOS speed (project velocity difference onto LOS in *world* frame)
                vx_opp, vy_opp = float(vxs[opp_idx]), float(vys[opp_idx])
                v_rel_wx = vx_opp - vx_ego
                v_rel_wy = vy_opp - vy_ego
                if dist < 1e-6:
                    rel_los = 0.0
                else:
                    rhat_wx, rhat_wy = dx_w / dist, dy_w / dist
                    rel_los = (v_rel_wx * rhat_wx + v_rel_wy * rhat_wy) / max(1e-6, self.v_pos_max)
                rel_los = float(np.clip(rel_los, -1.0, 1.0))

                # heading delta
                theta_opp = float(poses_th[opp_idx])
                dtheta = theta_opp - theta_ego
                sin_dt, cos_dt = math.sin(dtheta), math.cos(dtheta)

                scalars.extend([
                    float(np.clip(dist / max(1e-6, self._eff_max), 0.0, 1.0)),  # dist normalized by effective lidar max
                    sin_b, cos_b,
                    rel_los,
                    sin_dt, cos_dt
                ])
                opp_added = True

        if self.include_opp and not opp_added:
            scalars.extend([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])  # neutral defaults

        # --- Clearance summaries (optional) ---
        extras = {"effective_lidar_max": float(self._eff_max)}
        if self.include_clearance:
            fwd_q10 = float(np.quantile(norm_full[self._fwd_idx], 0.10)) if self._fwd_idx.size else 0.0
            left_q10 = float(np.quantile(norm_full[self._left_idx], 0.10)) if self._left_idx.size else 0.0
            right_q10 = float(np.quantile(norm_full[self._right_idx], 0.10)) if self._right_idx.size else 0.0
            scalars.extend([fwd_q10, left_q10, right_q10])
            extras.update({
                "clearance_q10_forward": fwd_q10,
                "clearance_q10_left": left_q10,
                "clearance_q10_right": right_q10,
            })

        # --- Optional: Track-relative features (only if progress helper provided) ---
        if self.include_track:
            try:
                # Expect your progress helper to expose a method returning (cte_norm, sin_err, cos_err)
                # You can adapt this to your actual API later.
                x_e, y_e = float(poses_x[ego_idx]), float(poses_y[ego_idx])
                cte_norm, sin_err, cos_err = self._centerline_features(x_e, y_e, theta_ego)
                scalars.extend([cte_norm, sin_err, cos_err])
                extras.update({
                    "cte_norm": float(cte_norm),
                    "sin_heading_err": float(sin_err),
                    "cos_heading_err": float(cos_err),
                })
            except Exception:
                # If your helper isn't wired yet, just skip
                scalars.extend([0.0, 0.0, 1.0])

        scalars_arr = np.asarray(scalars, dtype=np.float32)
        obs_vec = np.concatenate([lidar_feat.astype(np.float32, copy=False), scalars_arr], axis=0)
        return obs_vec, extras

    # ---------------- Internals ----------------

    def _normalize_lidar(self, scan: np.ndarray, *, eval_mode: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize raw LiDAR scan to [0,1] using fixed or auto effective max.
        Returns:
            norm_full : normalized full-resolution scan (len=beams)
            norm_pooled: normalized & pooled scan (len=beams//pool)
        """
        scan = np.asarray(scan, dtype=np.float32)
        # Update effective max if auto mode and not frozen/eval
        if self._lidar_mode == "auto" and not eval_mode and not self._freeze:
            p = float(np.percentile(scan, self._auto_percentile))
            target = max(self._min_eff_max, 0.9 * p)
            # EMA update (monotonic enough, but adapts upward when needed)
            self._eff_max = (1.0 - self._eff_ema_alpha) * self._eff_max + self._eff_ema_alpha * target

        eff = max(1e-6, self._eff_max)
        norm_full = np.clip(scan / eff, 0.0, 1.0)

        if self.pool == 1:
            norm_pooled = norm_full
        else:
            # Average pooling with exact divisibility
            norm_pooled = norm_full.reshape(self._pooled_len, self.pool).mean(axis=1)

        return norm_full, norm_pooled

    def _maybe_init_hist(self, norm_pooled: np.ndarray) -> None:
        """If history is empty and k>1, seed it with the current frame."""
        if self.k_hist > 1 and len(self._lidar_hist) == 0:
            for _ in range(self.k_hist - 1):
                self._lidar_hist.append(norm_pooled.copy())

    def _sector_indices(self, deg_lo: float, deg_hi: float) -> np.ndarray:
        """Return beam indices between deg_lo and deg_hi (in ego frame)."""
        rad_lo = math.radians(deg_lo)
        rad_hi = math.radians(deg_hi)
        # monotone increasing self._angles spans [-fov/2, +fov/2]
        mask = (self._angles >= min(rad_lo, rad_hi)) & (self._angles <= max(rad_lo, rad_hi))
        return np.nonzero(mask)[0].astype(np.int32)

    def _compute_scalar_len(self) -> int:
        # Base ego scalars: speed, yaw_rate, sin(theta), cos(theta)
        length = 4
        if self.include_last_action:
            length += 2
        if self.include_opp:
            # dist, sin(bearing), cos(bearing), rel_los_speed, sin(dtheta), cos(dtheta)
            length += 6
        if self.include_clearance:
            # forward_q10, left_q10, right_q10
            length += 3
        if self.include_track:
            # cte_norm, sin(heading_err), cos(heading_err)
            length += 3
        return length

    def _centerline_features(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        """
        Placeholder hook to compute centerline features from a user-supplied helper.
        Expected to return (cte_norm, sin_heading_err, cos_heading_err).
        Replace this body with your actual geometry code (or adapt the call).
        """
        # Example (pseudo): cte, half_w, heading_err = self.progress.metrics(x, y, theta)
        # cte_norm = np.clip(cte / max(half_w, 1e-6), -1.0, 1.0)
        # return float(cte_norm), float(np.sin(heading_err)), float(np.cos(heading_err))
        raise NotImplementedError("Centerline features require a progress helper; set include_track=False or implement this.")

    @staticmethod
    def _with_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing config keys with sensible defaults."""
        cfg = dict(cfg) if cfg is not None else {}

        # Top-level groups
        cfg.setdefault("obs", {})
        cfg.setdefault("obs", cfg["obs"])

        obs = cfg["obs"]
        # Normalization & geometry
        obs.setdefault("lidar_max_mode", "auto")      # 'fixed' or 'auto'
        obs.setdefault("lidar_fixed_max", 30.0)       # meters
        obs.setdefault("lidar_min_effective", 10.0)   # meters, floor for auto mode
        obs.setdefault("lidar_auto_percentile", 99.0) # percentile used for auto-cal
        obs.setdefault("lidar_effective_ema", 0.05)   # smoothing for effective max update

        # Downsample & history
        obs.setdefault("lidar_downsample", 1)         # pooling size (1 = no downsample)
        obs.setdefault("history_k", 1)                # LiDAR frames to stack

        # Scalar normalization
        obs.setdefault("v_pos_max", 3.0)              # m/s for speed normalization (tune to your training cap)
        obs.setdefault("yaw_rate_norm", 10.0)         # matches env bounds
        obs.setdefault("steer_max_abs", 0.4189)       # ~24 degrees; change to your env's vehicle param
        obs.setdefault("speed_max", 3.0)              # m/s; for last_action normalization if last_action_units='env'
        obs.setdefault("last_action_units", "env")    # 'env' (physical) or 'normalized' ([-1,1])

        # Feature toggles
        obs.setdefault("include", {})
        obs["include"].setdefault("last_action", True)
        obs["include"].setdefault("clearance", True)
        obs["include"].setdefault("opponent", True)
        obs["include"].setdefault("track", False)     # requires progress helper

        # Sector definitions (degrees)
        obs.setdefault("sectors", {})
        obs["sectors"].setdefault("forward_half_deg", 15.0)
        obs["sectors"].setdefault("side_low_deg", 60.0)
        obs["sectors"].setdefault("side_high_deg", 90.0)

        cfg["obs"] = obs
        return cfg
