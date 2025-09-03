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
from agents import TD3Agent, TD3Config
from replay_buffer import PrioritizedReplayBuffer
from rewards import RewardWrapper
from map_utils import get_map_bounds

from gap_follow import gap_follow_action



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dirs(paths: dict) -> Path:
    run_name = paths.get("run_name", f"td3_run_{int(time.time())}")
    
    base_logs = Path(paths.get("logs_dir", "./logs"))
    # Create per-run folder under logs
    run_dir = base_logs / f"{run_name}_{int(time.time())}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_yaml(d: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def main(args: Optional[argparse.Namespace] = None):
    cnf_path = '/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/TD3/config.yaml'
    


    with open(cnf_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Dirs & seed
    run_dir = ensure_dirs(cfg.get("paths", {}))
    save_yaml(cfg, run_dir / "config.yaml")
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    
    model_path = cfg['env'].get('model')
    
    map_bounds = get_map_bounds(cfg['env'].get('map_path')+'.yaml')
    lidar_max = cfg["obs"]["lidar_max"]
    obs_w = ObservationWrapper(lidar_max,map_bounds)
    
    action_low = np.array(cfg["env"]["action_low"], dtype=np.float32)
    action_high = np.array(cfg["env"]["action_high"], dtype=np.float32)
    act_wrap = ActionWrapper( float(action_low[0]), float(action_high[0]), float(action_low[1]), float(action_high[1]))
    
    
    reward_w = RewardWrapper()


    env = gym.make(
                cfg["env"]["id"],
                render_mode=cfg["env"].get("render_mode", None),
                map_dir=cfg["env"]["map_dir"],
                map=cfg["env"]["map"],
                map_ext=cfg["env"]["map_ext"],
                num_agents=int(cfg["env"]["num_agents"]),
            )
    
    
    max_episode_steps = int(cfg["env"].get("max_episode_steps", 2000))
    
    
    start_poses = cfg["env"].get("start_poses", None)
    if start_poses is not None:
        start_poses = np.array(start_poses, dtype=np.float32)


    obs_dict, info = env.reset(options=start_poses)

    obs_vec= obs_w.build(obs_dict)
    obs_dim = obs_vec.size
    act_dim = int(len(action_low))  # expect 2


    td3_cfg = TD3Config(
        actor_hidden=tuple(cfg["td3"].get("actor_hidden", (128,128))),
        critic_hidden=tuple(cfg["td3"].get("critic_hidden", (128,128))),
        gamma=cfg["td3"]["gamma"],
        tau=cfg["td3"]["tau"],
        actor_lr=cfg["td3"]["actor_lr"],
        critic_lr=cfg["td3"]["critic_lr"],
        policy_freq=cfg["td3"]["policy_freq"],
        policy_noise=cfg["action"]["policy_noise"],
        noise_clip=cfg["action"]["noise_clip"],
        expl_noise_std=cfg["action"]["train_action_noise_std"],
        expl_noise_clip=cfg["action"].get("train_action_noise_clip", 0.5),
        per_alpha=cfg["per"]["alpha"],
        per_beta_init=cfg["per"]["beta_init"],
        per_beta_final=cfg["per"]["beta_final"],
        per_eps=cfg["per"]["priority_epsilon"],
    )
    agent = TD3Agent(obs_dim, act_dim, cfg=td3_cfg)
    
    if os.path.isfile(str(model_path + "best.pt")):
        agent.load(str(model_path + "best.pt"))
        print('loaded')

    buffer = PrioritizedReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        capacity=int(cfg["per"]["capacity"]),
        alpha=cfg["per"]["alpha"],
        priority_eps=cfg["per"]["priority_epsilon"],
        seed=seed,
    )


    total_steps = int(cfg["train"]["total_steps"])
    warmup_steps = int(cfg["train"]["warmup_steps"])
    update_after = int(cfg["train"]["update_after"])
    batch_size = int(cfg["train"]["batch_size"])
    updates_per_step = int(cfg["train"]["updates_per_step"])

    global_steps = 0
    episode = 0
    best_eval_return = -1e9

    def run_episode(eval_mode: bool = False) -> Tuple[float, int]:
        nonlocal global_steps, episode

        total_r = 0.0
        steps = 0
        done = False
        terminated = False
        truncated = False
        
        obs_dict, _ = env.reset(options=start_poses)
        agent.ou_noise.reset()
        agent.reset_action_state()
        
        while not done and steps < max_episode_steps:

            obs_vec_local = obs_w.build(obs_dict)

            # --- Ego action (normalized)
            if (not eval_mode) and (global_steps < warmup_steps):
                act_norm = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            else:
                act_norm = agent.select_action(obs_vec_local, eval_mode=eval_mode)
                
            if not np.all(np.isfinite(act_norm)):
                print("[WARN] Non-finite action from policy; zeroing.")
                act_norm = np.zeros_like(act_norm)

            ego_action = act_wrap.build(act_norm)

            opp_scan = np.asarray(obs_dict["scans"][1], dtype=np.float32)
            opp_action_env = gap_follow_action(opp_scan).astype(np.float32)

            actions_env = np.stack([ego_action, opp_action_env], axis=0).astype(np.float32)

            next_obs_dict, env_rew, terminated, truncated, info = env.step(actions_env)
            
            episode_ended = bool(terminated or truncated)  # for control/printing/etc.
            done_for_td = bool(terminated)
            done = bool(terminated or truncated)

            next_obs_vec = obs_w.build(next_obs_dict)

            r = reward_w.compute(next_obs_dict)

            if not eval_mode:
                buffer.add(obs_vec_local, act_norm, r, next_obs_vec, done_for_td)

                # Learn (after update_after)
                if global_steps >= update_after:
                    for _ in range(updates_per_step):
                        progress = min(1.0, global_steps / max(1, total_steps))
                        _metrics = agent.update(buffer, batch_size, progress=progress)

            # Tally
            total_r += r
            steps += 1
            global_steps += (0 if eval_mode else 1)
            obs_dict = next_obs_dict
            # env.render()
            # env.render()

            if episode_ended or steps >= max_episode_steps:
                break

        return total_r, steps

    # --- Training loop
    print("[TD3] Starting training...")
    while global_steps < total_steps:
        episode += 1
        ep_ret, ep_steps = run_episode(eval_mode=False)

        if episode % 20 == 0:
            # Just crossed an eval boundary: run one eval episode
            
            eval_ret, eval_steps = run_episode(eval_mode=True)
            

            # Save best
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                best_path = str(model_path + "best.pt")
                agent.save(best_path)
                print(f"[SAVE] New best eval return {best_eval_return:.3f}  ")

            print(f"[EVAL] ret={eval_ret:.3f} steps={eval_steps} best={best_eval_return:.3f}")


        print(f"Ep {episode:04d} [TRAIN] | R: {ep_ret:.2f} | steps: {ep_steps} | buf: {len(buffer)} | gstep: {global_steps}")


    env.close()


if __name__ == "__main__":
    main()
