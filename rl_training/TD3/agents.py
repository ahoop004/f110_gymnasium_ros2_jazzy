# agents.py
# TD3 agent (MLP actor + twin critics) with PER support.
# - Actor outputs normalized actions in [-1, 1]^act_dim.
# - Action noise for exploration is added here (Option A).
# - Target policy smoothing (policy_noise/noise_clip) for TD3 backups.
# - PER: uses min(Q1,Q2) TD-error for priorities; critic loss is IS-weighted.
#
# Expected replay buffer API (PER-capable):
#   add(s, a, r, s2, d, priority=None)
#   sample(batch_size, beta) -> (batch, indices, is_weights)
#       where batch is a dict with:
#           'obs':      np.float32 [B, obs_dim]
#           'actions':  np.float32 [B, act_dim]   (normalized actions in [-1,1])
#           'rewards':  np.float32 [B, 1] or [B]
#           'next_obs': np.float32 [B, obs_dim]
#           'dones':    np.float32 [B, 1] or [B]  (1.0 if terminal else 0.0)
#   update_priorities(indices, new_priorities: np.ndarray)
#
# NOTE: Store *normalized* actions in replay. Mapping to env units is handled elsewhere.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import ActorMLP, TwinCriticMLP


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def soft_update_(net: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), target.parameters()):
            p_t.mul_(1.0 - tau).add_(tau * p)


@dataclass
class TD3Config:
    # model
    actor_hidden: Tuple[int, ...] = (256, 256)
    critic_hidden: Tuple[int, ...] = (256, 256)
    # rl
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.0  # critic L2
    policy_freq: int = 2
    # TD3 target smoothing (normalized action space)
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    # exploration (normalized space)
    expl_noise_std: float = 0.10
    expl_noise_clip: float = 0.5
    # PER
    per_alpha: float = 0.6
    per_beta_init: float = 0.4
    per_beta_final: float = 1.0
    per_eps: float = 1e-6


class TD3Agent:
    """
    TD3 with twin critics and PER support.
    - Actor returns normalized actions in [-1,1]; exploration noise added here.
    - Critic takes normalized actions.
    - Save/Load: state_dict-only (models + optimizers + counters + config).

    Replay expectations:
      - Store normalized actions in replay ('actions' key).
      - Sample returns PER weights and indices for priority updates.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[TD3Config] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg or TD3Config()

        # Networks
        self.actor = ActorMLP(obs_dim, act_dim, hidden_sizes=self.cfg.actor_hidden).to(self.device)
        self.actor_targ = ActorMLP(obs_dim, act_dim, hidden_sizes=self.cfg.actor_hidden).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.critic = TwinCriticMLP(obs_dim, act_dim, hidden_sizes=self.cfg.critic_hidden).to(self.device)
        self.critic_targ = TwinCriticMLP(obs_dim, act_dim, hidden_sizes=self.cfg.critic_hidden).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        # Optims
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr, weight_decay=self.cfg.weight_decay)

        # Step counters
        self.total_updates: int = 0
        self.total_env_steps: int = 0  # (optional) can be set by td3_main if you want annealing by steps

        # PER beta schedule
        self._beta = self.cfg.per_beta_init

        # For logging
        self._last_metrics: Dict[str, float] = {}

    # ---------------- Acting ----------------

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Returns a normalized action in [-1,1]^act_dim.
        Adds Gaussian exploration noise (clipped) when eval_mode=False.
        """
        self.actor.eval()
        with torch.no_grad():
            obs_t = to_tensor(obs, self.device)
            a = self.actor(obs_t).cpu().numpy()
        a = a.reshape(-1)  # (act_dim,)

        if not eval_mode:
            noise = np.random.normal(0.0, self.cfg.expl_noise_std, size=self.act_dim).astype(np.float32)
            if self.cfg.expl_noise_clip is not None:
                noise = np.clip(noise, -self.cfg.expl_noise_clip, self.cfg.expl_noise_clip)
            a = np.clip(a + noise, -1.0, 1.0)

        return a.astype(np.float32)

    # ---------------- Replay helpers (optional) ----------------

    def remember(self, replay, s, a_norm, r, s2, d, priority: Optional[float] = None) -> None:
        """
        Convenience passthrough; expects normalized action a_norm.
        """
        replay.add(s, a_norm, r, s2, d, priority=priority)

    # ---------------- Training / Update ----------------

    def current_beta(self) -> float:
        return float(self._beta)

    def _anneal_beta(self, progress: float) -> None:
        """
        progress in [0,1]; caller can provide training progress.
        """
        progress = float(np.clip(progress, 0.0, 1.0))
        self._beta = self.cfg.per_beta_init + (self.cfg.per_beta_final - self.cfg.per_beta_init) * progress

    def update(
        self,
        replay,
        batch_size: int,
        *,
        progress: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        One TD3 update step (possibly actor update delayed by policy_freq).
        - Samples a PER batch with beta.
        - Critic update every call.
        - Actor + target updates every policy_freq calls.

        Args:
            replay: PER buffer with sample()/update_priorities()
            batch_size: minibatch size
            progress: optional [0,1] for PER beta anneal

        Returns:
            metrics dict (losses, Q stats, beta, etc.)
        """
        if progress is not None:
            self._anneal_beta(progress)

        # ---- Sample batch
        batch, idxs, is_w = replay.sample(batch_size, beta=self._beta)
        obs = to_tensor(batch["obs"], self.device)
        act = to_tensor(batch["actions"], self.device)  # normalized actions in [-1,1]
        rew = to_tensor(batch["rewards"], self.device).view(-1, 1)
        obs2 = to_tensor(batch["next_obs"], self.device)
        done = to_tensor(batch["dones"], self.device).view(-1, 1)

        # Make sure shapes are consistent
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            rew = rew.unsqueeze(0)
            obs2 = obs2.unsqueeze(0)
            done = done.unsqueeze(0)

        B = obs.shape[0]
        is_w = to_tensor(is_w, self.device).view(B, 1)
        gamma = self.cfg.gamma

        # ---- Critic target
        with torch.no_grad():
            # Target action with TD3 smoothing noise
            a2 = self.actor_targ(obs2)
            if self.cfg.policy_noise is not None and self.cfg.policy_noise > 0.0:
                noise = torch.randn_like(a2) * self.cfg.policy_noise
                if self.cfg.noise_clip is not None and self.cfg.noise_clip > 0.0:
                    noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)
                a2 = torch.clamp(a2 + noise, -1.0, 1.0)
            # Target Q
            q1_t, q2_t = self.critic_targ(obs2, a2)
            q_targ = torch.min(q1_t, q2_t)
            y = rew + (1.0 - done) * gamma * q_targ  # (B,1)

        # ---- Critic update (IS-weighted MSE)
        q1, q2 = self.critic(obs, act)
        td1 = q1 - y
        td2 = q2 - y
        critic_loss = (is_w * (td1 ** 2)).mean() + (is_w * (td2 ** 2)).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Update priorities with min TD-error (L1)
        with torch.no_grad():
            td_err = torch.min(td1.abs(), td2.abs()).squeeze(1)  # (B,)
            new_prios = (td_err + self.cfg.per_eps).clamp_min(1e-12).detach().cpu().numpy().astype(np.float32)
        replay.update_priorities(idxs, new_prios)

        metrics: Dict[str, float] = {
            "loss/critic": float(critic_loss.detach().cpu().item()),
            "td/min_abs_mean": float(td_err.mean().cpu().item()),
            "q1/mean": float(q1.mean().detach().cpu().item()),
            "q2/mean": float(q2.mean().detach().cpu().item()),
            "per/beta": float(self._beta),
        }

        # ---- Delayed policy update
        update_actor = (self.total_updates + 1) % self.cfg.policy_freq == 0
        if update_actor:
            # Actor loss: maximize Q1(s, actor(s)) -> minimize -Q1
            a_pi = self.actor(obs)
            actor_loss = -self.critic.q1_only(obs, a_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            # Soft-update targets
            soft_update_(self.actor, self.actor_targ, self.cfg.tau)
            soft_update_(self.critic, self.critic_targ, self.cfg.tau)

            metrics["loss/actor"] = float(actor_loss.detach().cpu().item())
            metrics["update/actor"] = 1.0
        else:
            metrics["update/actor"] = 0.0

        self.total_updates += 1
        self._last_metrics = metrics
        return metrics

    # ---------------- Save / Load ----------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_targ": self.actor_targ.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_targ": self.critic_targ.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "total_updates": self.total_updates,
            "total_env_steps": self.total_env_steps,
            "per_beta": self._beta,
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.actor_targ.load_state_dict(state["actor_targ"], strict=strict)
        self.critic.load_state_dict(state["critic"], strict=strict)
        self.critic_targ.load_state_dict(state["critic_targ"], strict=strict)
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        # Restore counters/config
        if "cfg" in state and isinstance(state["cfg"], dict):
            # Keep current object; update fields present in saved cfg
            for k, v in state["cfg"].items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        self.total_updates = int(state.get("total_updates", 0))
        self.total_env_steps = int(state.get("total_env_steps", 0))
        self._beta = float(state.get("per_beta", self.cfg.per_beta_init))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, strict=strict)
