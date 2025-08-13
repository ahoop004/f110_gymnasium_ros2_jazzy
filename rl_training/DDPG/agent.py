import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Sequence
# from .observation import RunningNormalizer
from .replay_buffer import PrioritizedExperienceReplayBuffer

# config = Utils.load_yaml_config('/home/ahoope5/Desktop/SUMORL/SUMO-Routing-RL/src/configurations/config.yaml')

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)

class Actor(nn.Module):


    def __init__(self, obs_dim, act_dim, action_low: Sequence[float], action_high: Sequence[float]):
        super(Actor, self).__init__()
        # self.layer1 = nn.Linear(n_observations, 64)
        # self.layer2 = nn.Linear(64, 32)
        # self.layer3 = nn.Linear(32, n_actions)
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)
        
        self.register_buffer("action_low",  torch.tensor(action_low,  dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        self._init_weights()
        
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

 
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # [obs_dim] -> [1, obs_dim]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        t = torch.tanh(self.fc3(x))  # [-1, 1]

        # Affine scale to [low, high] per dimension
        low, high = self.action_low, self.action_high
        action = 0.5 * (high - low) * t + 0.5 * (high + low)
        return action

class Critic(nn.Module):


    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Encode state first
        self.fcs1 = nn.Linear(obs_dim, 128)
        # Then fuse action
        self.fcs2 = nn.Linear(128 + act_dim, 128)
        self.q = nn.Linear(128, 1)

        self._init_weights()
   
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fcs1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fcs2.weight, nonlinearity="relu")
        nn.init.uniform_(self.q.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fcs1.bias)
        nn.init.zeros_(self.fcs2.bias)
        nn.init.zeros_(self.q.bias)

 
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # Ensure batching
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)

        z = F.relu(self.fcs1(obs))
        z = torch.cat([z, act], dim=-1)
        z = F.relu(self.fcs2(z))
        q = self.q(z)
        return q







class DDPGAgent:

  
    def __init__(self,
                 state_size, 
                 action_size, 
                 path,
                 agent_id,
                 
                 action_low,
                 action_high,
                 
                 gamma=None, 
                 tau=None,
                 
                 actor_lr=None, 
                 critic_lr=None,
                 
                 memory_size=None,
                 
                 batch_size=None,
                 
                 alpha=0.6,
                 beta=0.4,
                 priority_epsilon=1e-5,
                 
                 noise_type= 'gaussian',
                 noise_sigma_start=None,
                 noise_sigma_min=None,
                 noise_decay=None,
                 
                 seed=42,
                 
                 ):
            """
        Initialize the PERAgent.

        Args:
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            path (str): Path to save the model.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_max (float): Maximum value of epsilon.
            epsilon_min (float): Minimum value of epsilon.
            memory_size (int): Size of the replay memory.
            batch_size (int): Batch size for training.
        """

            self.path = path

            self.agent_id = agent_id
            
            
            self.obs_dim = int(state_size)
            self.act_dim = int(action_size)
            self.action_low = np.asarray(action_low, dtype=np.float32)
            self.action_high = np.asarray(action_high, dtype=np.float32)
            assert self.action_low.shape == (self.act_dim,)
            assert self.action_high.shape == (self.act_dim,)

          

            self.batch_size = int(batch_size)
            self.gamma = float(gamma)
            self.tau = float(tau)
            self.beta = float(beta)
            self.priority_epsilon = float(priority_epsilon) 
            self.device = torch.device("cpu")
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            self.actor = Actor(self.obs_dim, self.act_dim, self.action_low, self.action_high).to(self.device)
            self.critic = Critic(self.obs_dim, self.act_dim).to(self.device)
            self.actor_target = Actor(self.obs_dim, self.act_dim, self.action_low, self.action_high).to(self.device)
            self.critic_target = Critic(self.obs_dim, self.act_dim).to(self.device)
            
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())


            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.critic_loss = torch.nn.MSELoss(reduction="none")
            
            self.eval_freeze_norm = False

            # alpha = config.get('alpha', 0.6)  # Hyperparameter for prioritized experience replay
            self.memory = PrioritizedExperienceReplayBuffer(batch_size=self.batch_size, buffer_size=memory_size, alpha=alpha)

            if noise_type.lower() == "ou":
                self.noise = OUActionNoise(
                    self.action_low, self.action_high,
                    sigma_start=noise_sigma_start,
                    sigma_min=noise_sigma_min,
                    decay=noise_decay,
                    seed=seed,
                )
            else:
                self.noise = GaussianActionNoise(
                    self.action_low, self.action_high,
                    sigma_start=noise_sigma_start,
                    sigma_min=noise_sigma_min,
                    decay=noise_decay,
                    seed=seed,
                )

            # Misc
            self.global_step = 0
            print("obs_dim", self.obs_dim)
            print("actor in", self.actor.fc1.in_features, "critic in", self.critic.fcs1.in_features)

            
            



    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.

        Args:
            state (array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (array): Next state.
            done (bool): Whether the episode is done.
        """
     

        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.add(experience)
    
  


    def replay(self):
        """
        Train the agent with a batch of experiences.

        Args:
            batch_size (int): Batch size for training.
        """

        # beta = config.get('beta', 0.4) 
        if len(self.memory) < self.batch_size:
            return

        idxs, experiences, is_weights = self.memory.sample(beta=self.beta)

        
        states_np      = np.stack([e.state for e in experiences])
        actions     = np.stack([e.action for e in experiences])
        rewards     = np.array([e.reward for e in experiences], dtype=np.float32).reshape(-1,1)
        next_states_np = np.stack([e.next_state for e in experiences])
        dones       = np.array([e.done for e in experiences], dtype=np.float32).reshape(-1,1)
        
       
        
        states      = torch.as_tensor(states_np,      dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions,     dtype=torch.float32, device=self.device)
        rewards     = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)
        weights_t   = torch.as_tensor(np.array(is_weights), dtype=torch.float32, device=self.device).unsqueeze(-1)

        B = self.batch_size
        assert states.shape      == (B, self.obs_dim), states.shape
        assert next_states.shape == (B, self.obs_dim), next_states.shape
        assert actions.shape     == (B, self.act_dim), actions.shape
        assert rewards.shape     == (B, 1), rewards.shape
        assert dones.shape       == (B, 1), dones.shape

        assert states.dtype == torch.float32
        assert actions.dtype == torch.float32

        dev = next(self.critic.parameters()).device
        for t in (states, next_states, actions, rewards, dones):
            assert t.device == dev, (t.device, dev)

        # model input dims match obs/action dims
        assert self.actor.fc1.in_features  == self.obs_dim
        assert self.critic.fcs1.in_features == self.obs_dim
        assert self.critic.fcs2.in_features == 128 + self.act_dim
        
        def _finite(name, x):
            ok = torch.isfinite(x).all().item()
            if not ok:
                bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:10].cpu().numpy().tolist()
                raise ValueError(f"Non-finite in {name}; examples idx={bad[:5]}")
            return True

        _finite("states", states); _finite("actions", actions)
        _finite("next_states", next_states); _finite("rewards", rewards); _finite("dones", dones)
        #Critic update
        
        with torch.no_grad():
            a_next   = self.actor_target(next_states)
            _finite("a_next", a_next)
            q_next   = self.critic_target(next_states, a_next)                     
            _finite("q_next", q_next)
            target_y = rewards + self.gamma * (1.0 - dones) * q_next               
            _finite("target_y", target_y)

        q_pred = self.critic(states, actions)                                      
        _finite("q_pred",q_pred)
        td_errors = target_y - q_pred                                              
        _finite("td_errors",td_errors)
        critic_loss_elems = (td_errors ** 2)
        critic_loss = (weights_t * critic_loss_elems).mean()
        

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        # optional: torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()
        
        #Actor update
        for p in self.critic.parameters():
            p.requires_grad_(False)
        a_pred = self.actor(states)
        actor_loss = -self.critic(states, a_pred).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()
        
        for p in self.critic.parameters():
            p.requires_grad_(True)
       
       
        new_priorities = td_errors.detach().abs().squeeze(-1).cpu().numpy() + self.priority_epsilon
        self.memory.update_priorities(idxs, new_priorities)
        
        self._soft_update(self.actor_target,  self.actor,  self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

        self.global_step += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss":  float(actor_loss.detach().cpu().item()),
            "mean_td_abs": float(np.mean(new_priorities)),
        }

    def choose_action(self, obs, training: bool = True):
        
        with torch.no_grad():
           
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            # obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            assert obs_t.ndim == 1 and obs_t.shape[0] == self.actor.fc1.in_features, \
                f"obs len {obs_t.shape[0]} != actor.fc1.in_features {self.actor.fc1.in_features}"

            # 2) numeric sanity
            if not torch.isfinite(obs_t).all():
                bad = (~torch.isfinite(obs_t)).nonzero(as_tuple=False).squeeze(-1)[:10].tolist()
                raise ValueError(f"Non-finite in obs at indices: {bad}")

            # 3) dtype/device/contiguity
            assert obs_t.dtype == torch.float32
            assert next(self.actor.parameters()).is_cuda == obs_t.is_cuda
            obs_t = obs_t.contiguous()
            a = self.actor(obs_t)  # already scaled to [low, high] by your Actor
        a = self.noise(a, training=training)  # add noise only when training
        return a.squeeze(0).detach().cpu().numpy()
    
    
    def _soft_update(self, target_net, online_net, tau: float):
        with torch.no_grad():
            for tp, p in zip(target_net.parameters(), online_net.parameters()):
                tp.data.lerp_(p.data, tau)
    
    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()

    def save_model(self, filename: str = "ddpg_checkpoint.pt"):
        os.makedirs(self.path, exist_ok=True)
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            # --- store bounds as plain lists (or torch tensors), NOT numpy arrays
            "action_low":  self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            # metadata
            "gamma": self.gamma,
            "tau": self.tau,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "global_step": getattr(self, "global_step", 0),
            "torch_version": torch.__version__,
        }
        path = os.path.join(self.path, filename)
        torch.save(ckpt, path)

    def load_model(self, filename: str = "ddpg_checkpoint.pt"):
        import numpy as np
        model_path = os.path.join(self.path, filename)
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}, continuing without loading.")
            return False

        ckpt = None
        # 1) Preferred: safe load (weights_only=True)
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e1:
            # 2) Allowlist numpy reconstruct ONLY if you trust this checkpoint
            try:
                from torch.serialization import safe_globals
                with safe_globals([np.core.multiarray._reconstruct, np.dtype]):
                    ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
                print("(info) Loaded with allowlisted NumPy globals.")
            except Exception as e2:
                # 3) Ultimate fallback: weights_only=False (unsafe; trusted sources only)
                print("(warn) Falling back to weights_only=False. Only do this for trusted checkpoints.")
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        # ---- restore ----
        self.actor.load_state_dict(ckpt["actor"], strict=True)
        self.critic.load_state_dict(ckpt["critic"], strict=True)
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]), strict=False)
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]), strict=False)

        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])

        # metadata (bounds may be list or tensor)
        if "action_low" in ckpt and "action_high" in ckpt:
            al = ckpt["action_low"]; ah = ckpt["action_high"]
            if isinstance(al, torch.Tensor): al = al.detach().cpu().tolist()
            if isinstance(ah, torch.Tensor): ah = ah.detach().cpu().tolist()
            self.action_low  = np.array(al, dtype=np.float32)
            self.action_high = np.array(ah, dtype=np.float32)

        self.gamma = float(ckpt.get("gamma", self.gamma))
        self.tau   = float(ckpt.get("tau", self.tau))
        self.obs_dim = int(ckpt.get("obs_dim", self.obs_dim))
        self.act_dim = int(ckpt.get("act_dim", self.act_dim))
        self.global_step = int(ckpt.get("global_step", 0))

        self.actor.to(self.device).train()
        self.critic.to(self.device).train()
        self.actor_target.to(self.device).eval()
        self.critic_target.to(self.device).eval()
        print(f"Loaded checkpoint from {model_path}")
        return True
        
    
    
    




class OUActionNoise:
    """
    Ornstein–Uhlenbeck process: dx = theta*(mu - x)*dt + sigma*sqrt(dt)*N(0, I)
    Kept per-action-dimension.
    """
    def __init__(self, act_low, act_high, mu=0.0, theta=0.15, sigma_start=0.2, sigma_min=0.05,
                 dt=1.0, decay=0.999, seed=42):
        self.low  = np.asarray(act_low, dtype=np.float32)
        self.high = np.asarray(act_high, dtype=np.float32)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.dt = float(dt)
        self.decay = float(decay)
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros_like(self.low, dtype=np.float32)  # OU state

    def reset(self):
        self.x[:] = 0.0

    def __call__(self, action, training: bool = True):
        if not training:
            return action

        a = action.detach().cpu().numpy()
        # OU update
        dx = self.theta * (self.mu - self.x) * self.dt \
             + self.sigma * np.sqrt(self.dt) * self.rng.normal(size=self.x.shape).astype(np.float32)
        self.x = self.x + dx

        a_noisy = np.clip(a + self.x, self.low, self.high)
        self.sigma = max(self.sigma * self.decay, self.sigma_min)

        if isinstance(action, torch.Tensor):
            return torch.from_numpy(a_noisy).to(action.device, dtype=action.dtype)
        return a_noisy
    
class GaussianActionNoise:
    """
    Add N(0, sigma^2) noise to actions, per-dimension.
    Noise lives in *action units* (already scaled to [low, high]).
    """
    def __init__(self, act_low, act_high, sigma_start=0.2, sigma_min=0.05, decay=0.999, seed=42):
        self.low  = np.asarray(act_low, dtype=np.float32)
        self.high = np.asarray(act_high, dtype=np.float32)
        self.sigma = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.decay = float(decay)
        self.rng = np.random.default_rng(seed)

    def __call__(self, action, training: bool = True):
        """
        action: torch.Tensor [B, act_dim] or [act_dim]
        returns: same shape, numpy or torch (matches input type)
        """
        if not training:
            return action

        # work in numpy for sampling
        a = action.detach().cpu().numpy()
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=a.shape).astype(np.float32)
        a_noisy = np.clip(a + noise, self.low, self.high)

        # decay sigma
        self.sigma = max(self.sigma * self.decay, self.sigma_min)

        # return in the same container type as input
        if isinstance(action, torch.Tensor):
            return torch.from_numpy(a_noisy).to(action.device, dtype=action.dtype)
        return a_noisy