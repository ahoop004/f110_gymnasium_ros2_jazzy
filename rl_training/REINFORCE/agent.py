import torch
from .policy import GaussianPolicy

class REINFORCEAgent:
    def __init__(self, obs_dim, act_dim, lr):
        self.policy = GaussianPolicy(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs):
        mean, std = self.policy(torch.FloatTensor(obs))
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.numpy(), log_prob

    def update_policy(self, log_probs, returns):
        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
