import numpy as np
import pandas as pd

class CenterlineHelper:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, comment='#', header=None)
        self.centerline = df[[0, 1]].values  # shape (N,2)
        # Compute arc-lengths for each segment
        diffs = np.diff(self.centerline, axis=0)
        self.seg_lengths = np.linalg.norm(diffs, axis=1)
        self.cum_lengths = np.insert(np.cumsum(self.seg_lengths), 0, 0)
        self.total_length = self.cum_lengths[-1]

    def project(self, pos):
        """Project pos (x, y) to centerline. Return (arc_length, nearest_idx)."""
        diffs = self.centerline - pos
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        # For sub-segment accuracy (optional, linear interpolation):
        if min_idx == 0:
            seg_start, seg_end = self.centerline[0], self.centerline[1]
            seg_idx = 0
        else:
            seg_start, seg_end = self.centerline[min_idx-1], self.centerline[min_idx]
            seg_idx = min_idx-1
        seg_vec = seg_end - seg_start
        seg_len = np.linalg.norm(seg_vec)
        if seg_len > 1e-6:
            t = np.dot(pos - seg_start, seg_vec) / (seg_len**2)
            t = np.clip(t, 0, 1)
        else:
            t = 0
        arc_length = self.cum_lengths[seg_idx] + t * seg_len
        return arc_length, min_idx


class WaypointHelper:
    def __init__(self, csv_path, num_waypoints=100):
        df = pd.read_csv(csv_path, comment='#', header=None)
        full_centerline = df[[0, 1]].values
        # Calculate cumulative arc-lengths
        diffs = np.diff(full_centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0)
        total_length = cum_lengths[-1]

        # Evenly spaced arc-lengths for waypoints
        target_lengths = np.linspace(0, total_length, num_waypoints, endpoint=False)
        waypoints = []
        idx = 0
        for tlen in target_lengths:
            while idx < len(cum_lengths) - 1 and cum_lengths[idx+1] < tlen:
                idx += 1
            seg_start = full_centerline[idx]
            seg_end = full_centerline[(idx+1)%len(full_centerline)]
            seg_len = np.linalg.norm(seg_end - seg_start)
            if seg_len == 0:
                waypoints.append(seg_start)
                continue
            t = (tlen - cum_lengths[idx]) / seg_len
            t = np.clip(t, 0, 1)
            point = seg_start + t * (seg_end - seg_start)
            waypoints.append(point)
        self.waypoints = np.array(waypoints)
        self.num_waypoints = num_waypoints
        self.total_length = total_length

    def nearest_waypoint(self, pos):
        dists = np.linalg.norm(self.waypoints - pos, axis=1)
        idx = np.argmin(dists)
        return idx, self.waypoints[idx]
