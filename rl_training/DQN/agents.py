# dqn_agent.py
import numpy as np
import torch, torch.nn.functional as F
from replay_buffer import PrioritizedReplayBuffer
from models import QNet
from actions_discrete import ACTION_TABLE, N_ACTIONS

class DQNAgent:
    def __init__(self, obs_dim, device="cuda", gamma=0.99,
                 lr=3e-4, tau=0.02,  # "aggressive" soft update
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
                 batch_size=256, update_after=5_000, update_every=50):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.q = QNet(obs_dim, N_ACTIONS).to(self.device)
        self.q_targ = QNet(obs_dim, N_ACTIONS).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = PrioritizedReplayBuffer()
        self.gamma = gamma
        self.tau = tau

        # epsilon schedule
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.steps = 0

        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every

    def epsilon(self):
        frac = min(1.0, self.steps / float(self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def select_action(self, obs_vec: np.ndarray, eval_mode=False) -> int:
        self.steps += 1
        if (not eval_mode) and (np.random.rand() < self.epsilon()):
            return np.random.randint(N_ACTIONS)
        x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)  # (1, N_ACTIONS)
        return int(q.argmax(dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buf.push(s, a, r, s2, float(done))

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_targ.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def update(self):
        if len(self.buf) < self.update_after or (self.steps % self.update_every != 0):
            return None
        s, a, r, s2, d = self.buf.sample(self.batch_size)
        s  = s.to(self.device)
        a  = a.to(self.device)
        r  = r.to(self.device).unsqueeze(-1)
        s2 = s2.to(self.device)
        d  = d.to(self.device).unsqueeze(-1)

        # Q(s,a)
        q = self.q(s).gather(1, a.unsqueeze(-1))  # (B,1)
        with torch.no_grad():
            # Double-DQN option (recommended): uncomment next two lines and delete the single-line max
            a2 = self.q(s2).argmax(dim=1, keepdim=True)           # online argmax
            q2 = self.q_targ(s2).gather(1, a2)                    # target eval
            # q2 = self.q_targ(s2).max(dim=1, keepdim=True).values    # basic DQN target

            target = r + (1.0 - d) * self.gamma * q2

        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())
