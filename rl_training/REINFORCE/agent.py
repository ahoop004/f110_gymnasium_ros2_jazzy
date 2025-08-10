import torch
from .policy import GaussianPolicy

class REINFORCEAgent:
    def __init__(self, obs_dim, act_dim, lr, gamma):
        self.gamma = gamma
        self.policy = GaussianPolicy(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs):
        mean, std = self.policy(torch.FloatTensor(obs))
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.numpy(), log_prob

    def update_policy(self, log_probs, rewards):
        returns = self.compute_returns(rewards)
        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns
    
    def load_model(self, path):
        self.policy.load(path)
    
    def save_model(self, path):
        self.policy.save(path)
    
    