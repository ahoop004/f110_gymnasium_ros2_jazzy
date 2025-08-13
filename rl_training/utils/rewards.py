import math
import numpy as np



LIDAR_LEN = 1080
LIDAR_MAX  = 1.0
def _wrap_angle(a: float) -> float:
    return ((a + np.pi) % (2 * np.pi)) - np.pi

def parse_flat_obs(obs_flat: np.ndarray):
    """
    obs_flat: [ ego_lidar(1080), ego_x, ego_y, ego_theta, ego_col, opp_x, opp_y, opp_col ]
    returns: ego_lidar(np.float32[1080]),
             ego_pose([x,y,theta]),
             ego_collision(bool),
             opp_pose([x,y,theta]),
             opp_collision(bool)
    """
    obs_flat = np.asarray(obs_flat, dtype=np.float32)
    assert obs_flat.ndim == 1 and obs_flat.shape[0] == LIDAR_LEN + 8, \
        f"Expected 1088-D obs, got {obs_flat.shape}"

    ego_lidar = obs_flat[:LIDAR_LEN]

    ego_x     = float(obs_flat[LIDAR_LEN + 0])
    ego_y     = float(obs_flat[LIDAR_LEN + 1])
    ego_th    = _wrap_angle(float(obs_flat[LIDAR_LEN + 2]))
    ego_col   = bool(obs_flat[LIDAR_LEN + 3])

    opp_x     = float(obs_flat[LIDAR_LEN + 4])
    opp_y     = float(obs_flat[LIDAR_LEN + 5])
    opp_th    = _wrap_angle(float(obs_flat[LIDAR_LEN + 6]))
    opp_col   = bool(obs_flat[LIDAR_LEN + 7])

    ego_pose  = [ego_x, ego_y, ego_th]
    opp_pose  = [opp_x, opp_y,opp_th]

    return ego_lidar, ego_pose, ego_col, opp_pose, opp_col

class BaseReward:
    def __call__(self, *args, **kwargs):
        """
        Backward-compatible:
          - New: reward(obs_flat)
          - Old: reward(ego_lidar, ego_pose, ego_collision, opp_pose, opp_collision)
        """
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            ego_lidar, ego_pose, ego_collision, opp_pose, opp_collision = parse_flat_obs(np.asarray(args[0]))
            return self.compute(ego_lidar, ego_pose, ego_collision, opp_pose, opp_collision)
        elif len(args) == 5:
            return self.compute(*args)
        else:
            raise TypeError("Reward expects either (obs_flat) or "
                            "(ego_lidar, ego_pose, ego_collision, opp_pose, opp_collision)")

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
    
def _rot_into_opp_frame(ego_xy, opp_xy, opp_theta):
    # r = R(-opp_theta) * (ego - opp)
    dx, dy = ego_xy[0] - opp_xy[0], ego_xy[1] - opp_xy[1]
    c, s = math.cos(-opp_theta), math.sin(-opp_theta)
    x_rel = c*dx - s*dy
    y_rel = s*dx + c*dy
    return x_rel, y_rel

class _ProgFallback:
    """Fallback: Euclidean displacement only (no centerline)."""
    def __init__(self): self.prev = {"ego":None,"opp":None}; self.cum={"ego":0.0,"opp":0.0}; self.ma_ego=0.0; self.beta=0.8
    def reset(self): self.__init__()
    def _step(self, who, pose):
        x, y = float(pose[0]), float(pose[1])
        if self.prev[who] is None: self.prev[who]=(x,y); return 0.0
        px, py = self.prev[who]; dx, dy = x-px, y-py; self.prev[who]=(x,y); return (dx*dx+dy*dy)**0.5
    def update(self, ego_pose, opp_pose, dt):
        e = self._step("ego", ego_pose); o = self._step("opp", opp_pose)
        self.cum["ego"] += e; self.cum["opp"] += o
        self.ma_ego = self.beta*self.ma_ego + (1-self.beta)*e
        return e, o
    def lead(self): return self.cum["ego"] - self.cum["opp"]

class _Prog:
    """
    Centerline-based progress with signed forward motion.
    Uses dot(delta_pos, track_tangent) to set the sign of progress.
    """
    def __init__(self, progress, beta: float = 0.8, auto_flip_steps: int = 20):
        if progress is None:
            raise ValueError("Centerline _Prog requires a 'progress' projector.")
        self.P = progress
        self.beta = float(beta)
        self.auto_flip_steps = int(auto_flip_steps)
        self.reset()

    def reset(self):
        self._s_prev = {"ego": None, "opp": None}
        self._cum    = {"ego": 0.0,  "opp": 0.0}
        self._ema_abs_dego = 0.0
        self._t_last = {"ego": 0.0,  "opp": 0.0}
        # store previous positions to get a motion vector
        self._p_prev = {"ego": None, "opp": None}
        # auto flip sign if early steps indicate reversed CSV direction
        self._flip = +1.0
        self._auto_buf = []

    def _tangent_at_s(self, s):
        # Unit tangent of the segment containing arclength s
        import numpy as _np
        idx = int(_np.searchsorted(self.P.s, s, side="right") - 1)
        idx = max(0, min(idx, self.P.n - 2))
        return self.P.tan[idx]  # (tx, ty), unit

    def _signed_step(self, who, x, y, s_curr, s_prev):
        # geometric step magnitude from arclength (wrap-aware)
        ds_geom = self.P.delta_s(s_curr, s_prev)
        # motion vector in world frame
        if self._p_prev[who] is None:
            self._p_prev[who] = (x, y)
            return 0.0
        dx = x - self._p_prev[who][0]
        dy = y - self._p_prev[who][1]
        self._p_prev[who] = (x, y)

        # signed component along track tangent
        tx, ty = self._tangent_at_s(s_curr)
        ds_sign = dx * tx + dy * ty

        # combine: use geometric magnitude but give it the sign from the motion vector
        # (this avoids small s-jitter causing sign flips)
        ds = math.copysign(abs(ds_geom), ds_sign if abs(ds_sign) > 1e-6 else ds_geom)
        return ds

    def update(self, ego_pose, opp_pose):
        ex, ey = float(ego_pose[0]), float(ego_pose[1])
        ox, oy = float(opp_pose[0]), float(opp_pose[1])

        e_s, e_t = self.P.project_xy(ex, ey)
        o_s, o_t = self.P.project_xy(ox, oy)

        # init
        if self._s_prev["ego"] is None: self._s_prev["ego"] = e_s
        if self._s_prev["opp"] is None: self._s_prev["opp"] = o_s

        de = self._signed_step("ego", ex, ey, e_s, self._s_prev["ego"])
        do = self._signed_step("opp", ox, oy, o_s, self._s_prev["opp"])

        self._s_prev["ego"] = e_s
        self._s_prev["opp"] = o_s

        # auto-flip CSV direction if early steps are consistently negative
        if len(self._auto_buf) < self.auto_flip_steps:
            self._auto_buf.append(de)
            if len(self._auto_buf) == self.auto_flip_steps:
                mean_de = sum(self._auto_buf) / max(1, len(self._auto_buf))
                if mean_de < 0.0:
                    self._flip = -1.0

        de *= self._flip
        do *= self._flip

        self._cum["ego"] += de
        self._cum["opp"] += do

        self._ema_abs_dego = self.beta * self._ema_abs_dego + (1.0 - self.beta) * abs(de)
        self._t_last["ego"] = e_t
        self._t_last["opp"] = o_t
        return de, do

    def lead(self) -> float:
        return self._cum["ego"] - self._cum["opp"]

    def ema_abs_ego(self) -> float:
        return self._ema_abs_dego

    def last_offsets(self):
        return self._t_last["ego"], self._t_last["opp"]




class CenterlineSafetyProgressReward:
    """
    Centerline progress + safety:
      + forward along-track progress
      + alive bonus
      - lateral offset penalty (soft, squared w/ cap)
      - near-wall penalty (robust LiDAR percentile, ignores zeros)
      - near-opponent bubble penalty
      - hard crash penalty
    """
    def __init__(self,
                 dt: float,
                 progress,                 # CenterlineProgress(closed=True)
                 # Progress
                 w_prog: float = 1.2,      # per meter; raise if needed
                 forward_sign: float = +1.0,  # set -1.0 if CSV direction is opposite
                 # Alive shaping
                 alive_bonus: float = 0.02,
                 # Lead shaping (optional, mild)
                 w_rel_lead: float = 0.0,
                 lead_clip: float = 5.0,
                 # Lateral (centerline) penalty
                 w_lat: float = 0.35,
                 lat_cap: float = 4.0,     # cap on lat_norm^2 to avoid blow-ups
                 default_half_width: float = 1.5,
                 # Wall safety
                 lidar_max: float = LIDAR_MAX,
                 near_wall_dist: float = 0.35/30.0,
                 w_wall: float = 1.0,
                 wall_quantile: float = 0.05,   # 5th percentile distance
                 # Opponent bubble
                 opp_safe_dist: float = 0.7,
                 w_opp: float = 0.8,
                 # Collisions
                 ego_crash_penalty: float = 50.0,
                 opp_crash_bonus: float = 50.0,
                 # Curriculum
                 grace_steps_wall: int = 25,
                 grace_steps_opp: int  = 25,
                 ):
        self.dt = float(dt)
        self.progress = progress
        self.trk = _Prog(progress, beta=0.8) if progress is not None else _ProgFallback()

        self.w_prog = float(w_prog)
        self.forward_sign = float(forward_sign)
        self.alive_bonus = float(alive_bonus)

        self.w_rel_lead = float(w_rel_lead)
        self.lead_clip  = float(lead_clip)

        self.w_lat = float(w_lat)
        self.lat_cap = float(lat_cap)
        self.default_half_width = float(default_half_width)

        self.lidar_max = float(lidar_max)
        self.near_wall_dist = float(near_wall_dist)
        self.w_wall = float(w_wall)
        self.wall_q = float(wall_quantile)

        self.opp_safe_dist = float(opp_safe_dist)
        self.w_opp = float(w_opp)

        self.ego_crash_penalty = float(ego_crash_penalty)
        self.opp_crash_bonus   = float(opp_crash_bonus)

        self.grace_steps_wall = int(grace_steps_wall)
        self.grace_steps_opp  = int(grace_steps_opp)

        self._steps = 0

    def reset(self):
        self.trk.reset()
        self._steps = 0

    def _step_progress(self, ego_pose, opp_pose):
        if isinstance(self.trk, _ProgFallback):
            de, do = self.trk.update(ego_pose, opp_pose, self.dt)
        else:
            de, do = self.trk.update(ego_pose, opp_pose)
        return de, do

    def _widths_at_s(self, s):
        P = self.progress
        try:
            import numpy as _np
            idx = int(_np.searchsorted(P.s, s, side="right") - 1)
            idx = max(0, min(idx, P.n - 2))
            wR, wL = P.widths_at_index(idx)
            if wR is None or wL is None:
                return self.default_half_width, self.default_half_width
            return float(wR), float(wL)
        except Exception:
            return self.default_half_width, self.default_half_width

    def __call__(self, *args):
        # Parse obs
        if len(args) == 1:
            ego_lidar, ego_pose, ego_col, opp_pose, opp_col = parse_flat_obs(np.asarray(args[0]))
        else:
            ego_lidar, ego_pose, ego_col, opp_pose, opp_col = args

        self._steps += 1

        # Hard term: crashes
        if bool(ego_col):
            return -self.ego_crash_penalty
        if bool(opp_col) and self.opp_crash_bonus > 0.0:
            return +self.opp_crash_bonus

        # Progress (along track, wrap-aware)
        dego, dopp = self._step_progress(ego_pose, opp_pose)
        # During the first few steps, ignore tiny negative dego from projection jitter
        if self._steps < 10:
            dego = max(0.0, dego)
        r_prog = self.w_prog * self.forward_sign * dego

        # Alive bonus to avoid suicidal resets
        r_alive = self.alive_bonus

        # Lead shaping (very mild, optional)
        r_lead = 0.0
        if self.w_rel_lead != 0.0 and not isinstance(self.trk, _ProgFallback):
            lead_m = np.clip(self.trk.lead(), -self.lead_clip, self.lead_clip)
            r_lead = self.w_rel_lead * (lead_m / self.lead_clip)

        # Lateral penalty
        if self.progress is not None:
            s_ego, t_ego = self.progress.project_xy(ego_pose[0], ego_pose[1])
            wR, wL = self._widths_at_s(s_ego)
            w_eff = wL if t_ego >= 0.0 else wR
            w_eff = max(0.2, float(w_eff))
            lat_norm = abs(t_ego) / w_eff
            # Huber-like cap on squared term
            lat_term = min(lat_norm * lat_norm, self.lat_cap)
            r_lat = -self.w_lat * lat_term
        else:
            r_lat = 0.0

        # Wall safety (robust percentile, ignore zeros)
        r_wall = 0.0
        if len(ego_lidar) and self._steps >= self.grace_steps_wall:
            rng = np.asarray(ego_lidar, dtype=np.float32)
            rng = np.where((rng <= 0.0) | ~np.isfinite(rng), self.lidar_max, rng)  # treat zeros/NaNs as far
            rng = np.clip(rng, 0.0, self.lidar_max)
            dmin = float(np.quantile(rng, self.wall_q))  # robust near-wall indicator
            if dmin < self.near_wall_dist:
                x = (self.near_wall_dist - dmin) / max(1e-6, self.near_wall_dist)
                r_wall = -self.w_wall * (x ** 2)

        # Opponent bubble (delayed)
        r_opp = 0.0
        if self._steps >= self.grace_steps_opp:
            dx = ego_pose[0] - opp_pose[0]
            dy = ego_pose[1] - opp_pose[1]
            rho = math.hypot(dx, dy)
            if rho < self.opp_safe_dist:
                y = (self.opp_safe_dist - rho) / max(1e-6, self.opp_safe_dist)
                r_opp = -self.w_opp * (y ** 2)
        # inside __call__, after r_opp:
        r_flank = 0.0
        x_rel, y_rel = _rot_into_opp_frame(ego_pose[:2], opp_pose[:2], opp_pose[2])
        # ahead of opp, slightly offset to a side
        if 0.2 <= x_rel <= 1.8 and 0.25 <= abs(y_rel) <= 0.8:
            # soft peak at the middle of the side window
            y_band = max(0.0, 0.8 - abs(abs(y_rel)-0.525))
            r_flank = 0.1 * (x_rel/1.8) * (y_band/0.8)

        # Sum
        r = r_prog + r_alive + r_lead + r_lat + r_wall + r_opp +r_flank
        return float(r)
