# dqn_agent.py
import numpy as np
import torch, torch.nn.functional as F
from replay_buffer import PrioritizedReplayBuffer
from models import QNet


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: str = "cuda",
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.02,              # aggressive soft update
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        batch_size: int = 256,
        update_after: int = 5_000,
        update_every: int = 50,
        grad_clip: float = 10.0,
        beta: float = 0.4 
    ):
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.q = QNet(obs_dim, n_actions).to(self.device)
        self.qt = QNet(obs_dim, n_actions).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        
        
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.grad_clip = grad_clip

        self.steps = 0
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.n_actions = n_actions

    def epsilon(self) -> float:
        t = min(1.0, self.steps / float(self.eps_decay_steps))
        return self.eps_start + t * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def select_action(self, obs_vec: np.ndarray, eval_mode: bool = False) -> int:
        self.steps += 1
        if (not eval_mode) and (np.random.rand() < self.epsilon()):
            return np.random.randint(self.n_actions)
        x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)  # (1, n_actions)
        return int(q.argmax(dim=1).item())

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.qt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def update(self, replay: PrioritizedReplayBuffer):
        if len(replay) < self.update_after or (self.steps % self.update_every != 0):
            return None

        s, a, r, s2, d = replay.sample(self.batch_size)
        s  = s.to(self.device)
        a  = a.to(self.device).unsqueeze(-1)          # (B,1)
        r  = r.to(self.device)                        # (B,1)
        s2 = s2.to(self.device)
        d  = d.to(self.device)                        # (B,1)

        # Q(s,a)
        q = self.q(s).gather(1, a)                    # (B,1)

        with torch.no_grad():
            # Double-DQN target
            a2 = self.q(s2).argmax(dim=1, keepdim=True)             # (B,1)
            q2 = self.qt(s2).gather(1, a2)                          # (B,1)
            target = r + (1.0 - d) * self.gamma * q2

        loss = F.smooth_l1_loss(q, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.soft_update()
        return float(loss.item())