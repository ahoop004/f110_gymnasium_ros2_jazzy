import numpy as np

def _safe_get(d, key, default=0.0):
    try:
        v = d.get(key, default)
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def _q10_forward_clearance(extras):
    """
    Your observation_wrapper said it computes clearance summaries from full LiDAR.
    Expect something like extras["clearance_forward_q10"] (rename if needed).
    """
    return float(extras.get("clearance_forward_q10", 10.0))

def _ego_safety_terms(extras, w_wall=1.0, w_yaw=0.2):
    # Smaller forward clearance => bigger penalty
    cf_q10 = _q10_forward_clearance(extras)              # meters, e.g., 0..30
    yaw_rate = abs(float(extras.get("ego_yaw_rate", 0))) # rad/s
    # Normalize a bit (tune 10.0 and 2.0 to your map/vehicle)
    wall_pen = -w_wall * np.exp(-cf_q10 / 6.0)
    yaw_pen  = -w_yaw  * min(yaw_rate / 2.0, 1.5)
    return wall_pen + yaw_pen

def _rel_features(extras):
    """
    Pull relative scalars the wrapper said it exposes:
      dist, sin(bearing), cos(bearing),
      rel_speed_los, sin(dtheta), cos(dtheta)
    Return with safe defaults.
    """
    dist = float(extras.get("opp_dist", 999.0))
    sb   = float(extras.get("opp_bearing_sin", 0.0))
    cb   = float(extras.get("opp_bearing_cos", -1.0))  # -1 => behind by default
    rs   = float(extras.get("opp_rel_speed_los", 0.0))
    sd   = float(extras.get("opp_dtheta_sin", 0.0))
    cd   = float(extras.get("opp_dtheta_cos", 1.0))
    return dist, sb, cb, rs, sd, cd

def _is_opp_crash(info, opp_idx=1):
    """
    Try to read common flags; fall back to None.
    Customize to your env's info schema.
    """
    # Examples that often exist in F1TENTH-style envs:
    crashed_agents = info.get("crashed_agents") or info.get("crashed") or {}
    if isinstance(crashed_agents, dict):
        return bool(crashed_agents.get(opp_idx, False))
    if isinstance(crashed_agents, (list, tuple)):
        try:
            return bool(crashed_agents[opp_idx])
        except Exception:
            pass
    # Vehicle-to-wall/vehicle flags:
    if bool(info.get("opp_wall_collision", False)) or bool(info.get("opp_vehicle_collision", False)):
        return True
    return False

def _infer_opp_crash_fallback(next_obs_dict, eps=0.2):
    """
    Heuristic if env doesn't mark collisions:
    opponent nearly stopped and near map boundary tends to coincide with crash/timeout.
    With only what's in obs_dict, we can check opponent speed magnitude.
    """
    try:
        vx = float(next_obs_dict["linear_vels_x"][1])
        vy = float(next_obs_dict["linear_vels_y"][1])
        speed = (vx*vx + vy*vy) ** 0.5
        return speed < eps
    except Exception:
        return False

def adversarial_block_reward(
    obs_dict,
    next_obs_dict,
    info,
    extras,
    done,
    terminated,
    truncated,
    *,
    w_opp_crash= +5.0,
    w_block_front= +1.0,
    w_close= +0.6,
    w_align= +0.4,
    w_press= +0.8,
    w_safe= +0.8,
    ego_collision_pen=-15.0,
    min_sep=0.8,
    max_effect_dist=6.0
):
    """
    Composite reward to cause opponent to crash while keeping ego safe.
    Returns: scalar float
    """
    r = 0.0

    # 1) Opponent crash bonus
    # CHANGE: pay as soon as detected; do NOT wait for episode end.
    opp_crash = _is_opp_crash(info, opp_idx=1)
    if not opp_crash:
        opp_crash = _infer_opp_crash_fallback(next_obs_dict)  # heuristic if env doesn't flag
    if opp_crash:
        r += w_opp_crash

    # 2) Blocking geometry
    dist, sb, cb, rs, sd, cd = _rel_features(extras)
    if dist < max_effect_dist:
        ahead_term = -cb  # positive when opponent is behind us (we're ahead)
        closeness = np.clip((max_effect_dist - dist) / max_effect_dist, 0.0, 1.0)
        if dist < min_sep:
            closeness *= 0.25

        r += w_block_front * ahead_term
        r += w_close       * closeness
        r += w_align       * max(cd, 0.0) * closeness

        lateral_pressure = np.clip(abs(sb), 0.0, 1.0) * ahead_term * closeness
        r += w_press * lateral_pressure

    # 3) Ego safety & collision penalties
    r += w_safe * _ego_safety_terms(extras, w_wall=1.0, w_yaw=0.2)

    ego_crash = bool(info.get("ego_wall_collision", False) or info.get("ego_vehicle_collision", False))
    if ego_crash:
        r += ego_collision_pen

    # Robustness: ensure finite scalar
    if not np.isfinite(r):
        r = float(0.0)
    return float(r)
