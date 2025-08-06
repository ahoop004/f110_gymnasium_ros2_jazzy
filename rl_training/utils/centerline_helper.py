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
