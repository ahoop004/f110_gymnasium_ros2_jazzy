# td3_main.py
# TD3 training loop for the F1TENTH LIMO setup (single ego + gap-follow opponent).
# - Loads config.yaml
# - Creates env, observation wrapper, action mapper, agent, and PER buffer
# - Runs train/eval loops with warmup, periodic checkpoints, and basic logging
#
# Notes:
# - Uses env reward for now. We'll plug in rewards.py next.
# - Stores *normalized* actions in replay (as required by TD3Agent).
# - Opponent uses gap-follow (from your helper if available; else a safe fallback).
#
# Run:
#   python td3_main.py --config ./config.yaml

from __future__ import annotations
import os
import sys
import time
import math
import json
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple
from collections import deque

import numpy as np
import torch
import yaml
import gymnasium as gym

from obs import ObservationWrapper
from act import ActionWrapper
from agents import DQNAgent
from replay_buffer import PrioritizedReplayBuffer
from rewards import RewardWrapper
from map_utils import get_map_bounds

from gap_follow import gap_follow_action



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linspace_bins(n: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n, dtype=np.float32)


def make_action_table(n_steer: int = 7, n_speed: int = 7) -> np.ndarray:
    """
    Returns (n_steer*n_speed, 2) normalized actions in [-1,1]^2.
    For ActionWrapper(mode="rates"), interpret as [deltadot_norm, accel_norm].
    For mode="direct", interpret as [steer_norm, speed_norm].
    """
    s = linspace_bins(n_steer)
    v = linspace_bins(n_speed)
    S, V = np.meshgrid(s, v, indexing="xy")  # (n_speed, n_steer)
    table = np.stack([S.ravel(), V.ravel()], axis=-1)  # (N,2)
    return table.astype(np.float32)




def main(args: Optional[argparse.Namespace] = None):
    # --- Load your YAML config (same path pattern as TD3 main) ---
    cnf_path = '/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/TD3/config.yaml'
    with open(cnf_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Seeds / device
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    # Map / obs wrapper
    map_bounds = get_map_bounds(cfg['env'].get('map_path') + '.yaml')
    lidar_max = cfg["obs"]["lidar_max"]
    obs_w = ObservationWrapper(
        lidar_max, map_bounds,
        lidar_reduce_mode="subsample",
        lidar_reduce_factor=6,
        # vel_max used inside ObservationWrapper if you added it; else ignored.
    )

    # Action wrapper (RATES mode, with asymmetric braking)
    action_low = np.array(cfg["env"]["action_low"], dtype=np.float32)  # [steer_min, speed_min]
    action_high= np.array(cfg["env"]["action_high"], dtype=np.float32) # [steer_max, speed_max]
    aw = ActionWrapper(
        steer_min=float(action_low[0]),
        steer_max=float(action_high[0]),
        vel_min=float(action_low[1]),
        vel_max=float(action_high[1]),
        mode="rates",
        decision_every=10,   # repeat last action for 10 env steps
        dt=0.01,
        delta_rate_max=5.0,  # rad/s
        # asymmetric accel limits:
        accel_fwd_max=3.0,   # m/s^2
        accel_brake_max=8.0, # m/s^2
    )

    # Env
    env = gym.make(
        cfg["env"]["id"],
        render_mode=cfg["env"].get("render_mode", None),
        map_dir=cfg["env"]["map_dir"],
        map=cfg["env"]["map"],
        map_ext=cfg["env"]["map_ext"],
        num_agents=int(cfg["env"]["num_agents"]),
        render_fps=30
    )
    reward_w = RewardWrapper()

    # Discrete actions: 7 x 7 = 49.
    ACTION_TABLE = make_action_table(7, 7)
    N_ACTIONS = ACTION_TABLE.shape[0]

    # Reset / initial obs
    start_poses = cfg["env"].get("start_poses", None)
    if start_poses is not None:
        start_poses = np.array(start_poses, dtype=np.float32)

    obs_dict, info = env.reset(options=start_poses)
    ego = int(obs_dict["ego_idx"])
    vx  = float(obs_dict["linear_vels_x"][ego])
    vy  = float(obs_dict["linear_vels_y"][ego])
    aw.reset(init_steer=0.0, init_speed=float(np.hypot(vx, vy)))

    obs_vec = obs_w.build(obs_dict)
    obs_dim = int(obs_vec.size)

    # Agent + replay
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=N_ACTIONS,
        gamma=cfg["td3"]["gamma"],     # reuse gamma from YAML
        lr=3e-4,
        tau=0.02,                      # aggressive soft update
        eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
        batch_size=int(cfg["train"]["batch_size"]),
        update_after=int(cfg["train"]["update_after"]),
        update_every=int(cfg["train"]["updates_per_step"]),
    )
    replay = PrioritizedReplayBuffer(obs_dim=obs_dim, capacity=int(cfg["per"]["capacity"]), seed=seed)

    # Training controls
    max_episode_steps = int(cfg["env"].get("max_episode_steps", 5000))
    total_steps = int(cfg["train"]["total_steps"])
    global_steps = 0
    episode = 0
    best_eval_return = -1e9

    def run_episode(eval_mode: bool = False) -> Tuple[float, int]:
        nonlocal global_steps
        total_r = 0.0
        steps = 0
        done = False

        obs_dict, info = env.reset(options=start_poses)
        ego = int(obs_dict["ego_idx"])
        vx  = float(obs_dict["linear_vels_x"][ego])
        vy  = float(obs_dict["linear_vels_y"][ego])
        aw.reset(init_steer=0.0, init_speed=float(np.hypot(vx, vy)))
        reward_w.reset(obs_dict)

        while not done and steps < max_episode_steps:
            s = obs_w.build(obs_dict)

            # ε-greedy over 49 actions
            a_idx = agent.select_action(s, eval_mode=eval_mode)
            a_norm = ACTION_TABLE[a_idx]  # (2,) in [-1,1]^2 as [deltadot_norm, accel_norm]

            # Integrate to env inputs
            cur_speed = float(np.hypot(
                float(obs_dict["linear_vels_x"][ego]),
                float(obs_dict["linear_vels_y"][ego]),
            ))
            ego_action_env = aw.build(a_norm, cur_speed=cur_speed, can_choose=info.get("can_choose"))

            # Opponent: gap-follow (as in your TD3 main)
            opp_scan = np.asarray(obs_dict["scans"][1], dtype=np.float32)
            opp_action_env = gap_follow_action(opp_scan).astype(np.float32)

            actions_env = np.stack([ego_action_env, opp_action_env], axis=0).astype(np.float32)

            next_obs_dict, env_rew, terminated, truncated, info = env.step(actions_env)
            done = bool(terminated or truncated)

            s2 = obs_w.build(next_obs_dict)
            # Use your reward wrapper (optionally combine with env reward)
            r = reward_w.compute(next_obs_dict, a_norm)
            if truncated and not terminated:
                r += -15.0  # keep your timeout penalty

            if not eval_mode:
                replay.add(s, a_idx, float(r), s2, bool(terminated))  # DQN uses terminal flag for bootstrapping
                # Learn
                agent.update(replay)
                global_steps += 1

            total_r += float(r)
            steps += 1
            obs_dict = next_obs_dict

            # Optional live render
            env.render()

        return total_r, steps

    # --- Train/Eval loop ---
    print("[DQN] Starting training...")
    while episode < 5000 and global_steps < total_steps:
        episode += 1
        ep_ret, ep_steps = run_episode(eval_mode=False)

        if episode % 20 == 0:
            eval_ret, eval_steps = run_episode(eval_mode=True)
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                # Save online Q
                model_dir = cfg['env'].get('model')
                os.makedirs(model_dir, exist_ok=True)
                torch.save(agent.q.state_dict(), os.path.join(model_dir, "best_dqn.pt"))
                print(f"[SAVE] New best eval return {best_eval_return:.3f}")

            print(f"[EVAL] ret={eval_ret:.3f} steps={eval_steps} best={best_eval_return:.3f}")

        print(f"Ep {episode:04d} [TRAIN] | R: {ep_ret:.2f} | steps: {ep_steps} | buf: {len(replay)} | gstep: {global_steps}")

    env.close()


if __name__ == "__main__":
    main()