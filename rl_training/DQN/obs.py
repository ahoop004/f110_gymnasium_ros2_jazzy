# observation.py

import numpy as np
import math
from typing import Literal, Optional


class ObservationWrapper:
    """
    Observation builder with configurable LiDAR reduction + speed features.

    Args:
        lidar_max_range: float, max LiDAR range (meters)
        bounds: dict with keys 'x_min','x_max','y_min','y_max'
        vel_max: float, max linear speed to normalize sqrt(vx^2 + vy^2) into [0,1]
        lidar_reduce_mode: one of {"subsample","avg","min"}
        lidar_reduce_factor: int >=1 (e.g., 6 -> 1080->180)
        lidar_num_bins: optional exact output beam count
        lidar_roi_center: optional float in [0,1] (center of ROI along beams)
        lidar_roi_fraction: optional float in (0,1] fraction of beams to keep before reduction
    """

    def __init__(
        self,
        lidar_max_range: float,
        bounds: dict,
        vel_max: float = 20.0,
        lidar_reduce_mode: Literal["subsample", "avg", "min"] = "subsample",
        lidar_reduce_factor: int = 6,
        lidar_num_bins: Optional[int] = None,
        lidar_roi_center: Optional[float] = None,
        lidar_roi_fraction: Optional[float] = None,
    ):
        self.lidar_max_range = float(lidar_max_range)

        self.x_min = float(bounds["x_min"])
        self.x_max = float(bounds["x_max"])
        self.y_min = float(bounds["y_min"])
        self.y_max = float(bounds["y_max"])

        if self.x_max <= self.x_min:
            raise ValueError("Invalid x bounds: x_max must be > x_min")
        if self.y_max <= self.y_min:
            raise ValueError("Invalid y bounds: y_max must be > y_min")

        self._x_span = self.x_max - self.x_min
        self._y_span = self.y_max - self.y_min

        # speed normalization
        if vel_max <= 0:
            raise ValueError("vel_max must be > 0")
        self.vel_max = float(vel_max)

        # LiDAR reduction config
        if lidar_reduce_mode not in {"subsample", "avg", "min"}:
            raise ValueError("lidar_reduce_mode must be 'subsample', 'avg', or 'min'")
        if lidar_reduce_factor < 1:
            raise ValueError("lidar_reduce_factor must be >= 1")
        if lidar_num_bins is not None and lidar_num_bins < 1:
            raise ValueError("lidar_num_bins must be >= 1 when provided")
        if lidar_roi_fraction is not None and not (0.0 < lidar_roi_fraction <= 1.0):
            raise ValueError("lidar_roi_fraction must be in (0,1]")
        if lidar_roi_center is not None and not (0.0 <= lidar_roi_center <= 1.0):
            raise ValueError("lidar_roi_center must be in [0,1]")

        self.lidar_reduce_mode = lidar_reduce_mode
        self.lidar_reduce_factor = int(lidar_reduce_factor)
        self.lidar_num_bins = int(lidar_num_bins) if lidar_num_bins is not None else None
        self.lidar_roi_center = lidar_roi_center
        self.lidar_roi_fraction = lidar_roi_fraction

    # -------------------------- LiDAR helpers --------------------------

    def _apply_lidar_roi(self, scan: np.ndarray) -> np.ndarray:
        if self.lidar_roi_fraction is None:
            return scan

        n = scan.shape[0]
        keep = max(1, int(round(n * self.lidar_roi_fraction)))

        if self.lidar_roi_center is None:
            c = n // 2
        else:
            c = int(round(self.lidar_roi_center * (n - 1)))

        half = keep // 2
        start = max(0, c - half)
        end = min(n, start + keep)
        start = max(0, end - keep)
        return scan[start:end]

    def _pool_groups(self, arr: np.ndarray, num_groups: int, mode: str) -> np.ndarray:
        groups = np.array_split(arr, num_groups)
        if mode == "avg":
            return np.array([g.mean() if g.size else 0.0 for g in groups], dtype=np.float32)
        elif mode == "min":
            return np.array([g.min() if g.size else 0.0 for g in groups], dtype=np.float32)
        else:
            return np.array([g[0] if g.size else 0.0 for g in groups], dtype=np.float32)

    def _reduce_lidar(self, scan: np.ndarray) -> np.ndarray:
        scan = self._apply_lidar_roi(scan)
        n = scan.shape[0]

        if self.lidar_num_bins is not None:
            num_out = min(self.lidar_num_bins, n)
            if num_out <= 0:
                return scan[:0]
            return self._pool_groups(scan, num_out, mode=self.lidar_reduce_mode)

        if self.lidar_reduce_factor <= 1:
            return scan

        if self.lidar_reduce_mode == "subsample":
            return scan[::self.lidar_reduce_factor].astype(np.float32, copy=False)

        num_groups = max(1, int(np.ceil(n / self.lidar_reduce_factor)))
        return self._pool_groups(scan, num_groups, mode=self.lidar_reduce_mode)

    # -------------------------- Main builder --------------------------

    def build(self, observations: dict) -> np.ndarray:
        """
        Expects keys:
          - 'ego_idx', 'scans', 'poses_x', 'poses_y', 'poses_theta',
            'linear_vels_x', 'linear_vels_y', 'ang_vels_z', 'collisions'
        """
        ego = int(observations["ego_idx"])

        scans      = np.asarray(observations["scans"], dtype=np.float32)
        poses_x    = np.asarray(observations["poses_x"], dtype=np.float32)
        poses_y    = np.asarray(observations["poses_y"], dtype=np.float32)
        poses_th   = np.asarray(observations["poses_theta"], dtype=np.float32)
        vxs        = np.asarray(observations["linear_vels_x"], dtype=np.float32)
        vys        = np.asarray(observations["linear_vels_y"], dtype=np.float32)
        wzs        = np.asarray(observations["ang_vels_z"], dtype=np.float32)   # available if you want later
        collisions = np.asarray(observations["collisions"], dtype=np.float32)

        if scans.ndim != 2:
            raise ValueError("scans must be a 2D array-like of shape (N_agents, N_beams)")

        num_agents = poses_x.shape[0]
        if not (0 <= ego < num_agents):
            raise IndexError("ego_idx out of range")

        # Opponent index (first non-ego if any)
        opp = (0 if ego != 0 else 1) if num_agents >= 2 else None

        # LiDAR (ego) → [0,1], then reduce beams
        lidar_raw = np.clip(scans[ego] / self.lidar_max_range, 0.0, 1.0).astype(np.float32, copy=False)
        lidar = self._reduce_lidar(lidar_raw)

        # Ego
        ex = float(poses_x[ego]); ey = float(poses_y[ego]); et = float(poses_th[ego])
        vx = float(vxs[ego]);     vy = float(vys[ego])
        ego_speed = float(np.hypot(vx, vy))
        ego_speed_n = np.clip(ego_speed / self.vel_max, 0.0, 1.0)
        ego_crashed = float(collisions[ego])

        # Opponent (zeros if not present)
        if opp is None:
            ox = oy = ot = 0.0
            opp_speed_n = 0.0
            ocr = 0.0
        else:
            ox = float(poses_x[opp]); oy = float(poses_y[opp]); ot = float(poses_th[opp])
            ovx = float(vxs[opp]);    ovy = float(vys[opp])
            opp_speed = float(np.hypot(ovx, ovy))
            opp_speed_n = np.clip(opp_speed / self.vel_max, 0.0, 1.0)
            ocr = float(collisions[opp])

        # Positions → [-1,1]
        ex_n = np.clip(2.0 * (ex - self.x_min) / self._x_span - 1.0, -1.0, 1.0)
        ey_n = np.clip(2.0 * (ey - self.y_min) / self._y_span - 1.0, -1.0, 1.0)
        ox_n = np.clip(2.0 * (ox - self.x_min) / self._x_span - 1.0, -1.0, 1.0)
        oy_n = np.clip(2.0 * (oy - self.y_min) / self._y_span - 1.0, -1.0, 1.0)

        # Angles → [-1,1] via /pi
        et_n = np.clip(et / math.pi, -1.0, 1.0)
        ot_n = np.clip(ot / math.pi, -1.0, 1.0)

        # Tail includes BOTH speeds now
        tail = np.array(
            [
                # ego
                ex_n, ey_n, et_n, ego_speed_n, ego_crashed,
                # opponent
                ox_n, oy_n, ot_n, opp_speed_n, ocr,
            ],
            dtype=np.float32,
        )

        return np.concatenate([lidar, tail], axis=0).astype(np.float32, copy=False)
