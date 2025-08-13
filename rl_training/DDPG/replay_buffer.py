# replay_buffer.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Any

class PrioritizedExperienceReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer with ring-buffer insertion.

    Stores (priority, experience) pairs, where `experience` can be any Python object
    (e.g., a namedtuple with fields: state, action, reward, next_state, done).

    Sampling:
      p_i = (priority_i + eps)^alpha / sum_j (priority_j + eps)^alpha
      w_i = (N * p_i)^(-beta);  normalized by max_i w_i  (so max weight = 1)
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        alpha: float = 0.6,
        seed: int = 42,
        priority_epsilon: float = 1e-6,
    ):
        assert buffer_size > 0 and batch_size > 0
        self._buffer_size = int(buffer_size)
        self._batch_size = int(batch_size)
        self._alpha = float(alpha)
        self._eps = float(priority_epsilon)

        # Structured array: explicit object dtype for experiences
        self._buffer = np.empty(
            self._buffer_size,
            dtype=[("priority", np.float32), ("experience", object)],
        )
        self._length = 0
        self._next_idx = 0

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._length

    # ---------------------------
    # Insertion (ring buffer)
    # ---------------------------
    def add(self, experience: Any, priority: float | None = None) -> None:
        """
        Add an experience. If `priority` is None, initialize with current max priority (or 1.0 if empty).
        """
        if priority is None:
            if self._length > 0:
                p0 = float(np.max(self._buffer["priority"][:self._length]))
                if not np.isfinite(p0) or p0 <= 0.0:
                    p0 = 1.0
            else:
                p0 = 1.0
        else:
            p0 = float(priority)

        # Clamp priority for safety
        p0 = np.clip(p0, 1e-8, np.finfo(np.float32).max).astype(np.float32)

        self._buffer["priority"][self._next_idx] = p0
        self._buffer["experience"][self._next_idx] = experience

        # Advance ring pointer
        if self._length < self._buffer_size:
            self._length += 1
        self._next_idx = (self._next_idx + 1) % self._buffer_size

    # ---------------------------
    # Sampling
    # ---------------------------
    def sample(self, beta: float = 0.4) -> Tuple[np.ndarray, List[Any], np.ndarray]:
        """
        Sample a batch using PER. Returns:
          idxs: indices into the buffer
          experiences: list of experiences
          weights: importance-sampling weights (float32, shape [B])
        """
        if self._length == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        ps = self._buffer["priority"][: self._length]  # shape [N]
        # Numerically safe probabilities
        ps_alpha = np.power(ps + self._eps, self._alpha, dtype=np.float64)
        den = ps_alpha.sum()
        if den <= 0.0 or not np.isfinite(den):
            # Fallback to uniform
            sampling_probs = np.full(self._length, 1.0 / self._length, dtype=np.float64)
        else:
            sampling_probs = ps_alpha / den  # shape [N]

        # Sample without replacement if we can
        replace = self._length < self._batch_size
        idxs = self._rng.choice(
            self._length, size=self._batch_size, replace=replace, p=sampling_probs
        ).astype(np.int64)

        # IS weights
        beta = float(beta)
        p_selected = sampling_probs[idxs]  # shape [B]
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.power(self._length * p_selected, -beta, dtype=np.float64)
        # Normalize
        m = np.max(weights)
        if not np.isfinite(m) or m <= 0.0:
            weights = np.ones_like(weights, dtype=np.float64)
        else:
            weights = weights / m
        weights = weights.astype(np.float32)

        experiences = [self._buffer["experience"][i] for i in idxs]
        return idxs, experiences, weights

    # ---------------------------
    # Priority updates
    # ---------------------------
    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities at specified indices. Priorities are clamped to keep them positive and finite.
        """
        pr = np.asarray(priorities, dtype=np.float32)
        if pr.ndim > 1:
            pr = pr.reshape(-1)
        pr = np.clip(pr, 1e-8, np.finfo(np.float32).max)

        # Basic safety: mask any NaN/Inf with a small value
        bad = ~np.isfinite(pr)
        if np.any(bad):
            pr[bad] = 1e-6

        self._buffer["priority"][idxs] = pr
