# observation.py

import numpy as np
import math


class ObservationWrapper:


    def __init__(self, lidar_max_range: float, bounds: tuple[float, float, float, float]):

        self.lidar_max_range = lidar_max_range
        self.x_min = bounds['x_min']
        self.x_max = bounds['x_max']
        self.y_min = bounds['y_min']
        self.y_max = bounds['y_max']
        
        if self.x_max <= self.x_min:
            raise ValueError("Invalid x bounds: x_max must be > x_min")
        if self.y_max <= self.y_min:
            raise ValueError("Invalid y bounds: y_max must be > y_min")
        self._x_span = self.x_max - self.x_min
        self._y_span = self.y_max - self.y_min
    
    def build(self, observations: dict) -> np.ndarray:
        """
        Build the normalized observation vector from a raw step() dict.

        Expects keys:
          - 'ego_idx', 'scans', 'poses_x', 'poses_y', 'poses_theta', 'collisions'
        """
        ego = int(observations["ego_idx"])

        # Coerce to arrays (float32 where applicable)
        scans = np.asarray(observations["scans"], dtype=np.float32)         # (N_agents, N_beams)
        poses_x = np.asarray(observations["poses_x"], dtype=np.float32)     # (N_agents,)
        poses_y = np.asarray(observations["poses_y"], dtype=np.float32)     # (N_agents,)
        poses_th = np.asarray(observations["poses_theta"], dtype=np.float32) # (N_agents,)
        collisions = np.asarray(observations["collisions"], dtype=np.float32) # (N_agents,)

        num_agents = poses_x.shape[0]
        if scans.ndim != 2:
            raise ValueError("scans must be a 2D array-like of shape (N_agents, N_beams)")
        if not (0 <= ego < num_agents):
            raise IndexError("ego_idx out of range")

        # Opponent: first non-ego if available, else None
        if num_agents >= 2:
            opp = 0 if ego != 0 else 1
        else:
            opp = None

        # LiDAR (ego), normalized to [0, 1]
        lidar = np.clip(scans[ego] / self.lidar_max_range, 0.0, 1.0).astype(np.float32, copy=False)

        # Ego fields
        ex = float(poses_x[ego])
        ey = float(poses_y[ego])
        et = float(poses_th[ego])
        ego_crashed = float(collisions[ego])

        # Opp fields (zeros if no opponent)
        if opp is None:
            ox = oy = ot = ocr = 0.0
        else:
            ox = float(poses_x[opp])
            oy = float(poses_y[opp])
            ot = float(poses_th[opp])
            ocr = float(collisions[opp])

        # Normalize positions to [-1, 1]
        ex_n = np.clip(2.0 * (ex - self.x_min) / self._x_span - 1.0, -1.0, 1.0)
        ey_n = np.clip(2.0 * (ey - self.y_min) / self._y_span - 1.0, -1.0, 1.0)
        ox_n = np.clip(2.0 * (ox - self.x_min) / self._x_span - 1.0, -1.0, 1.0)
        oy_n = np.clip(2.0 * (oy - self.y_min) / self._y_span - 1.0, -1.0, 1.0)

        # Normalize angles by pi to [-1, 1]
        et_n = np.clip(et / math.pi, -1.0, 1.0)
        ot_n = np.clip(ot / math.pi, -1.0, 1.0)

        tail = np.array(
            [ex_n, ey_n, et_n, ego_crashed, ox_n, oy_n, ot_n, ocr],
            dtype=np.float32,
        )

        return np.concatenate([lidar, tail], axis=0).astype(np.float32, copy=False)

