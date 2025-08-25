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

import numpy as np
import torch
import yaml
import gymnasium as gym

from observation import observation_wrapper
from actions import ActionMapper
from agents import TD3Agent, TD3Config
from replay_buffer import PrioritizedReplayBuffer

# --- Optional opponent policy (gap follow) ---
try:
    from utils.gap_follow import gap_follow_action as gap_follow
except Exception:
    # Fallback: drive slowly forward, small steering to center (no-op if no scan)
    def gap_follow(scan_1d: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float32)  # [steer_rad, speed_mps]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(cfg: dict):
    e = gym.make(
        cfg["env"]["id"],
        render_mode=cfg["env"].get("render_mode", None),
        map_dir=cfg["env"]["map_dir"],
        map=cfg["env"]["map"],
        map_ext=cfg["env"]["map_ext"],
        num_agents=int(cfg["env"]["num_agents"]),
    )
    return e


def ensure_dirs(paths: dict) -> Path:
    run_name = paths.get("run_name", f"td3_run_{int(time.time())}")
    base_ckpt = Path(paths.get("checkpoints_dir", "./checkpoints"))
    base_logs = Path(paths.get("logs_dir", "./logs"))
    # Create per-run folder under logs
    run_dir = base_logs / f"{run_name}_{int(time.time())}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_yaml(d: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def save_obs_wrapper_state(obs_w: observation_wrapper, path: Path) -> None:
    state = obs_w.get_state()
    torch.save(state, path)


def load_obs_wrapper_state(obs_w: observation_wrapper, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    obs_w.set_state(state)


def to_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def main(args: Optional[argparse.Namespace] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt) to resume")
    cli = parser.parse_args(None if args is None else [])

    # --- Load config
    with open(cli.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Dirs & seed
    run_dir = ensure_dirs(cfg.get("paths", {}))
    save_yaml(cfg, run_dir / "config.yaml")
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    # --- Env
    env = make_env(cfg)
    max_episode_steps = int(cfg["env"].get("max_episode_steps", 2000))
    start_poses = cfg["env"].get("start_poses", None)
    if start_poses is not None:
        start_poses = np.array(start_poses, dtype=np.float32)

    # Try to get dt from env; default to 0.01
    try:
        dt = float(getattr(env.unwrapped, "timestep", 0.01))
    except Exception:
        dt = 0.01

    # Action bounds (env units)
    action_low = np.array(cfg["env"]["action_low"], dtype=np.float32)
    action_high = np.array(cfg["env"]["action_high"], dtype=np.float32)
    steer_bounds = (float(action_low[0]), float(action_high[0]))
    speed_bounds = (float(action_low[1]), float(action_high[1]))

    # --- Observation wrapper
    obs_w = observation_wrapper(
        cfg, lidar_beams=cfg["env"]["lidar"]["beams"], fov=cfg["env"]["lidar"]["fov"], progress=None
    )

    # --- Action mapper (noise handled in the agent)
    mapper = ActionMapper(cfg, dt=dt, steer_bounds=steer_bounds, speed_bounds=speed_bounds)
    mapper.reset()  # seed last action

    # --- Dimensions
    # Reset env to get first obs and size things
    obs_dict, info = env.reset(options=start_poses)
    obs_w.reset(obs_dict)
    obs_vec, extras = obs_w.build(obs_dict, last_action=None, eval_mode=False)
    obs_dim = int(obs_w.obs_dim())
    act_dim = int(len(action_low))  # expect 2

    # --- Agent & PER
    td3_cfg = TD3Config(
        actor_hidden=tuple(cfg["td3"].get("actor_hidden", (256, 256))),
        critic_hidden=tuple(cfg["td3"].get("critic_hidden", (256, 256))),
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

    buffer = PrioritizedReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        capacity=int(cfg["per"]["capacity"]),
        alpha=cfg["per"]["alpha"],
        priority_eps=cfg["per"]["priority_epsilon"],
        seed=seed,
    )

    # --- Resume checkpoint (optional)
    ckpt_dir = run_dir / "checkpoints"
    if cli.resume and os.path.isfile(cli.resume):
        agent.load(cli.resume)
        # try loading obs wrapper state
        ow_state = Path(cli.resume).with_suffix(".obs.pt")
        if ow_state.exists():
            load_obs_wrapper_state(obs_w, ow_state)
        print(f"[Resume] Loaded checkpoint: {cli.resume}")

    # --- Training params
    total_steps = int(cfg["train"]["total_steps"])
    warmup_steps = int(cfg["train"]["warmup_steps"])
    update_after = int(cfg["train"]["update_after"])
    batch_size = int(cfg["train"]["batch_size"])
    updates_per_step = int(cfg["train"]["updates_per_step"])
    eval_every = int(cfg["train"]["eval_every_steps"])
    save_every = int(cfg["train"]["save_every_steps"])
    render_flag = bool(cfg["train"].get("render", False))

    # --- Loop state
    global_steps = 0
    episode = 0
    best_eval_return = -1e9

    # Helper to run one episode (train or eval)
    def run_episode(eval_mode: bool = False) -> Tuple[float, int]:
        nonlocal global_steps, episode
        obs_dict_local = obs_dict  # start from outer reset if called at the beginning
        if episode == 0 or eval_mode:
            # fresh reset for eval and first train episode
            obs_dict_local, _ = env.reset(options=start_poses)
            obs_w.reset(obs_dict_local)
            mapper.reset()

        total_r = 0.0
        steps = 0
        last_action_env = None
        done = False
        terminated = False
        truncated = False

        while not done and steps < max_episode_steps:
            # Build vector obs and diagnostics
            obs_vec_local, extras_local = obs_w.build(obs_dict_local, last_action=last_action_env, eval_mode=eval_mode)

            # --- Ego action (normalized)
            if (not eval_mode) and (global_steps < warmup_steps):
                a_norm = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            else:
                a_norm = agent.select_action(obs_vec_local, eval_mode=eval_mode)

            # Map to env units (+ governors, rate limits)
            ego_action_env = mapper.map(a_norm, extras=extras_local, training=not eval_mode, eval_mode=eval_mode)

            # Opponent action from current obs (opp index assumed 1)
            try:
                opp_scan = np.asarray(obs_dict_local["scans"][1], dtype=np.float32)
                opp_action_env = gap_follow(opp_scan).astype(np.float32)
            except Exception:
                opp_action_env = np.array([0.0, action_low[1]], dtype=np.float32)

            # Stack actions -> shape (2,2) for the env
            actions_env = np.stack([ego_action_env, opp_action_env], axis=0).astype(np.float32)

            # Step environment
            next_obs_dict, env_rew, terminated, truncated, info = env.step(actions_env)
            done = bool(terminated or truncated)

            # For now: use env reward (per-step constant). We'll replace with rewards.py later.
            # If env returns scalar timestep reward, it's same for all agents; take it as-is.
            r = float(env_rew)

            # Store transition (normalized action)
            next_obs_vec, _ = obs_w.build(next_obs_dict, last_action=ego_action_env, eval_mode=eval_mode)
            if not eval_mode:
                buffer.add(obs_vec_local, a_norm, r, next_obs_vec, done)

                # Learn (after update_after)
                if global_steps >= update_after:
                    for _ in range(updates_per_step):
                        progress = min(1.0, global_steps / max(1, total_steps))
                        _metrics = agent.update(buffer, batch_size, progress=progress)

            # Tally
            total_r += r
            steps += 1
            global_steps += (0 if eval_mode else 1)
            last_action_env = ego_action_env
            obs_dict_local = next_obs_dict

            if render_flag and hasattr(env, "render"):
                try:
                    env.render()
                except Exception:
                    pass

            if done:
                break

        return total_r, steps

    # --- Training loop
    print("[TD3] Starting training...")
    while global_steps < total_steps:
        episode += 1
        ep_ret, ep_steps = run_episode(eval_mode=False)

        if (global_steps // max(1, eval_every)) != ((global_steps - ep_steps) // max(1, eval_every)):
            # Just crossed an eval boundary: run one eval episode
            obs_w.set_eval(True)
            eval_ret, eval_steps = run_episode(eval_mode=True)
            obs_w.set_eval(False)

            # Save best
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                best_path = run_dir / "checkpoints" / "best.pt"
                agent.save(str(best_path))
                save_obs_wrapper_state(obs_w, best_path.with_suffix(".obs.pt"))
                print(f"[SAVE] New best eval return {best_eval_return:.3f}  -> {best_path.name}")

            print(f"[EVAL] ret={eval_ret:.3f} steps={eval_steps} best={best_eval_return:.3f}")

        # Periodic save by steps
        if save_every > 0 and (global_steps % save_every) < ep_steps:
            path = run_dir / "checkpoints" / f"step_{global_steps}.pt"
            agent.save(str(path))
            save_obs_wrapper_state(obs_w, path.with_suffix(".obs.pt"))
            print(f"[SAVE] checkpoint @ step {global_steps}")

        # Console log
        print(f"Ep {episode:04d} [TRAIN] | R: {ep_ret:.2f} | steps: {ep_steps} | buf: {len(buffer)} | gstep: {global_steps}")

    # --- Final save
    final_path = run_dir / "checkpoints" / "final.pt"
    agent.save(str(final_path))
    save_obs_wrapper_state(obs_w, final_path.with_suffix(".obs.pt"))
    print(f"[TD3] Done. Final checkpoint saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
