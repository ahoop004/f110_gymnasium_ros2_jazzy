# replay_buffer_dqn.py
import numpy as np
from typing import Dict, Tuple, Optional

class PrioritizedReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        priority_eps: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(priority_eps)

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity,),        dtype=np.int64)   # <-- action indices [0..N-1]
        self.rewards  = np.zeros((capacity, 1),      dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),      dtype=np.float32) # 1.0 if terminal else 0.0
        self.priorities = np.zeros((capacity,),      dtype=np.float32)

        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, s: np.ndarray, a_idx: int, r: float, s2: np.ndarray, d: bool, priority: Optional[float] = None) -> None:
        i = self.ptr
        self.obs[i] = s
        self.actions[i] = int(a_idx)                       # <-- store discrete index
        self.rewards[i, 0] = float(r)
        self.next_obs[i] = s2
        self.dones[i, 0] = 1.0 if d else 0.0

        p = (self.priorities.max() if self.size > 0 else 1.0) if (priority is None or not np.isfinite(priority)) else float(priority)
        self.priorities[i] = max(p, self.eps)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        assert self.size > 0, "Cannot sample from an empty buffer."
        N = self.size
        ps = self.priorities[:N].astype(np.float32)

        ps_alpha = np.power(ps + self.eps, self.alpha, dtype=np.float32)
        denom = ps_alpha.sum()
        probs = (ps_alpha / denom) if (np.isfinite(denom) and denom > 0.0) else np.full(N, 1.0 / N, dtype=np.float32)

        idxs = self.rng.choice(N, size=int(batch_size), replace=False, p=probs)

        beta = float(beta)
        weights = np.power(N * probs[idxs], -beta, dtype=np.float32)
        weights /= max(weights.max(), 1e-12)

        batch = {
            "obs": self.obs[idxs],
            "actions_idx": self.actions[idxs],           # <-- int64 indices
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }
        return batch, idxs.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray) -> None:
        idxs = idxs.astype(np.int64, copy=False)
        p = np.asarray(new_priorities, dtype=np.float32)
        p = np.where(np.isfinite(p), p, 0.0)
        self.priorities[idxs] = np.maximum(p, self.eps)
