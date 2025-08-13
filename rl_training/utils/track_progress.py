# utils/track_progress.py
import numpy as np
from scipy.spatial import cKDTree

class CenterlineProgress:
    """
    Loads (x_m, y_m, w_tr_right_m, w_tr_left_m) CSV and provides:
      - s(x,y): along-track arclength with wrap for closed loops
      - t(x,y): signed lateral offset (+left/-right) in meters
      - wR(s), wL(s): lane half-widths (optional, via interpolation)

    Usage:
        P = CenterlineProgress("/path/Shanghai_map.csv", closed=True)
        s, t = P.project_xy(x, y)
        ds = P.delta_s(s_curr, s_prev)  # wrap-aware forward progress
    """
    def __init__(self, csv_path: str, closed: bool = True):
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Clean columns
        cols = [c.strip().lstrip("#").strip() for c in df.columns]
        df.columns = cols
        if not {"x_m","y_m"}.issubset(df.columns):
            raise ValueError(f"CSV must have x_m,y_m; has {df.columns}")

        self.xy = df[["x_m","y_m"]].to_numpy(dtype=float)
        self.n  = len(self.xy)
        if self.n < 2:
            raise ValueError("Need at least 2 centerline points")

        # Segment vectors & cumulative arclength s
        seg = np.diff(self.xy, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.s = s
        self.L = s[-1]
        self.closed = bool(closed)

        # Unit tangents per segment
        eps = 1e-12
        self.tan = seg / np.maximum(seg_len[:,None], eps)

        # Segment normals (left-hand normal)
        self.nrm = np.stack([-self.tan[:,1], self.tan[:,0]], axis=1)

        # Midpoints & KDTree for fast nearest-segment lookup
        self.mid = (self.xy[:-1] + self.xy[1:]) * 0.5
        self.kd  = cKDTree(self.mid)

        # Optional width fields
        self.has_widths = {"w_tr_right_m","w_tr_left_m"}.issubset(df.columns)
        if self.has_widths:
            self.wR = df["w_tr_right_m"].to_numpy(dtype=float)
            self.wL = df["w_tr_left_m"].to_numpy(dtype=float)
        else:
            self.wR = self.wL = None

    def project_xy(self, x: float, y: float):
        """
        Orthogonally project (x,y) to its nearest centerline segment.
        Returns:
          s_proj: along-track arclength (0..L)
          t: signed lateral offset (+left/-right) in meters
        """
        p = np.array([x,y], dtype=float)

        # Search a few nearest midpoints to guard against corner cases
        d, idxs = self.kd.query(p, k=min(5, len(self.mid)))

        best = None
        for idx in np.atleast_1d(idxs):
            a = self.xy[idx]
            b = self.xy[idx+1]
            ab = b - a
            L2 = np.dot(ab, ab)
            if L2 <= 1e-12:
                continue
            ap = p - a
            t_par = np.clip(np.dot(ap, ab) / L2, 0.0, 1.0)  # segment parameter
            proj = a + t_par * ab
            # Along-track s at projection
            s_proj = self.s[idx] + t_par * np.linalg.norm(ab)
            # Signed lateral using left-hand normal at this segment
            n = self.nrm[idx]
            t_signed = np.dot(p - proj, n)
            cand = (np.linalg.norm(p - proj), s_proj, t_signed)
            if (best is None) or (cand[0] < best[0]):
                best = cand

        if best is None:
            # Degenerate fallback: snap to nearest node
            j = np.argmin(np.linalg.norm(self.xy - p, axis=1))
            return float(self.s[j]), 0.0

        return float(best[1]), float(best[2])

    def delta_s(self, s_curr: float, s_prev: float) -> float:
        """Forward progress, wrap-aware if track is closed."""
        ds = s_curr - s_prev
        if self.closed:
            # Map to shortest forward equivalent in [-L/2, L/2]
            if ds >  0.5*self.L: ds -= self.L
            if ds < -0.5*self.L: ds += self.L
        return ds

    def widths_at_index(self, i: int):
        if not self.has_widths:
            return None, None
        i = int(np.clip(i, 0, self.n-1))
        return self.wR[i], self.wL[i]
