#!/usr/bin/env python3
"""Rebuild ppo_model_py36.zip without numpy._core / pickle-protocol-5 deps.

Strategy:
  1. Extract policy.pth from the original zip using zipfile (pure bytes —
     no cloudpickle / numpy involved at this step).
  2. Create a fresh SB3 PPO model with the correct architecture and spaces
     (known from ppo.yaml / gaplock_attacker.yaml).
  3. Load the extracted PyTorch weights into the fresh model.
  4. Save with cloudpickle protocol 4 — compatible with Python 3.6.

Run with the numpy-1.x venv (NOT the main .venv which has numpy 2.x):
    /tmp/sb3_compat_venv/bin/python convert_model_py36.py
"""
import functools
import io
import pathlib
import zipfile

import cloudpickle

# ── Patch cloudpickle before importing SB3 ──────────────────────────────────
_orig_dumps = cloudpickle.dumps
cloudpickle.dumps = functools.partial(_orig_dumps, protocol=4)

import torch                          # noqa: E402
import numpy as np                    # noqa: E402
import gymnasium as gym               # noqa: E402
from stable_baselines3 import PPO     # noqa: E402

script_dir = pathlib.Path(__file__).parent
zip_path = script_dir / "ppo_model.zip"
out_path = script_dir / "ppo_model_py36"

# ── Step 1: extract policy weights directly (no cloudpickle) ────────────────
print(f"Reading {zip_path} ...")
with zipfile.ZipFile(zip_path) as z:
    with z.open("policy.pth") as f:
        policy_bytes = f.read()

policy_params = torch.load(io.BytesIO(policy_bytes), map_location="cpu", weights_only=True)
print("Policy keys:", list(policy_params.keys()))

# ── Step 2: create fresh model with correct spaces & architecture ────────────
# obs_dim=119 confirmed by policy_net.0.weight shape
obs_dim = policy_params["mlp_extractor.policy_net.0.weight"].shape[1]
print(f"Detected obs_dim={obs_dim}")

obs_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
)
action_space = gym.spaces.Box(
    low=np.array([-0.46, -1.0], dtype=np.float32),
    high=np.array([0.46, 1.0], dtype=np.float32),
    dtype=np.float32,
)


class _DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self, **kwargs):
        return np.zeros(obs_dim, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(obs_dim, dtype=np.float32), 0.0, True, False, {}


model = PPO(
    "MlpPolicy",
    _DummyEnv(),
    policy_kwargs={"net_arch": [256, 256]},
    verbose=0,
)

# ── Step 3: load extracted weights ───────────────────────────────────────────
missing, unexpected = model.policy.load_state_dict(policy_params, strict=False)
if missing:
    print("WARNING — missing keys:", missing)
if unexpected:
    print("WARNING — unexpected keys:", unexpected)
print("Weights loaded.")

# ── Step 4: save (cloudpickle.dumps is now patched to protocol 4) ─────────────
model.save(str(out_path))
print(f"Saved {out_path}.zip  (numpy 1.x, pickle protocol 4 — Python 3.6 compatible)")
