import os
import yaml
import numpy as np
import gymnasium as gym

from utils.track_progress import CenterlineProgress


from DDPG.agent import DDPGAgent

from utils.rewards import   CenterlineSafetyProgressReward
from utils.gap_follow import gap_follow_action



csv_path = "/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/maps/cenerlines/Shanghai_map.csv"
P = CenterlineProgress(csv_path, closed=True)

# === Config ===
with open('/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/DDPG/ddpg_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

episodes    = config['training_settings']['episodes']
max_steps   = config['training_settings']['max_steps']
warmup_steps   = config['training_settings'].get('warmup_steps', 0)
eval_every_eps = config['training_settings'].get('eval_interval_episodes', 10)
save_every     = config['training_settings'].get('save_interval_steps', 5000)
seed           = config['training_settings'].get('seed', 42)
render         = config['training_settings'].get('render', False)

env_cfg     = config['env_settings']
model_path  = env_cfg ['model_path']
ckpt_name   = env_cfg.get('checkpoint_filename', 'ddpg_checkpoint.pt')
start_poses = env_cfg ['start_poses']
action_low  = np.array(env_cfg['action_low'],  dtype=np.float32)  # [s_min, v_min]
action_high = np.array(env_cfg['action_high'], dtype=np.float32)  # [s_max, v_max]


# Agent hyperparameters
hp         = config['agent_hyperparameters']
gamma      = hp['gamma']
tau        = hp['tau']
actor_lr   = hp['actor_lr']
critic_lr  = hp['critic_lr']
memory_sz  = hp['memory_size']
batch_sz   = hp['batch_size']
per_alpha  = hp['per']['alpha']
per_beta   = hp['per']['beta']
prio_eps   = hp['per']['priority_epsilon']
noise_type = hp['noise']['type']
noise_sig0 = hp['noise']['sigma_start']
noise_sigm = hp['noise']['sigma_min']
noise_dec  = hp['noise']['decay']

base_dir  = os.path.dirname(model_path) if model_path else os.getcwd()

# === Env === (kept exactly like your current setup)
env = gym.make(
    'f110_gym:f110-v0',
    render_mode='human_fast',
    map_dir=config['env_settings']['map_dir'],
    map=config['env_settings']['map'],
    map_ext=config['env_settings']['map_ext'],
    num_agents=2,
)


obs, info = env.reset(options=np.array(start_poses, dtype=np.float32))

# obs = flatten_obs(obs)
# Build flat obs for the actor from the dict
obs = np.asarray(obs, dtype=np.float32)
# produces your 1088-length flat vector
obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])

# Action space (2-D continuous: [steer, velocity])
n_actions = int(np.asarray(action_low).shape[0])


agent = DDPGAgent(
    state_size=obs_dim,
    action_size=n_actions,
    path=model_path,              # base directory for checkpoints
    agent_id=0,

    action_low=action_low,
    action_high=action_high,

    gamma=gamma,
    tau=tau,

    actor_lr=actor_lr,
    critic_lr=critic_lr,

    memory_size=memory_sz,
    batch_size=batch_sz,

    alpha=per_alpha,
    beta=per_beta,
    priority_epsilon=prio_eps,

    noise_type=noise_type,
    noise_sigma_start=noise_sig0,
    noise_sigma_min=noise_sigm,
    noise_decay=noise_dec,

    seed=seed,
)
ckpt_path = os.path.join(model_path, ckpt_name)
if os.path.exists(ckpt_path):
    agent.load_model('best.pt')

# reward_fn = FrontFlankHerdReward(
#     dt=env.unwrapped.timestep,
#     target_bearing_deg=45.0,   # front-left/right
#     bearing_tol_deg=15.0,
#     target_dist=3.0,
#     dist_band=1.0,
#     w_in_front=0.5,
#     w_bearing=1.0,
#     w_distance=0.5,
#     w_heading_align=0.2,
#     w_speed_match=0.2,
#     crash_bonus=6.0,
#     progress=P,
# )
reward_fn = CenterlineSafetyProgressReward(
    dt=env.unwrapped.timestep,
    progress=P,
    # Make going forward clearly “worth it”
    w_prog=5.0,            # ↑ from 1.2
    alive_bonus=0.5,      # ↑ from 0.02
    # Soften early punishment (curriculum)
    grace_steps_wall=25,  # ↑
    grace_steps_opp=175,   # ↑
    # Make penalties gentler at start
    w_lat=0.25,            # ↓ from 0.35
    lat_cap=3.0,           # ↓ from 4.0
    near_wall_dist=0.30/30,   # ↓ from 0.35
    w_wall=0.30,           # ↓ from 1.0–1.5
    wall_quantile=0.10,    # ↑ from 0.05 (more forgiving)
    opp_safe_dist=0.60,    # ↓ from 0.7
    w_opp=0.30,            # ↓ from 0.8–1.0
    # Leave lead shaping off until it drives
    w_rel_lead=0.0,
)
global_step = 0

best_reward = 0
for episode in range(episodes):
    reward_fn.reset()
    obs, info = env.reset(options=np.array(start_poses, dtype=np.float32))
    # obs = flatten_obs(obs)
    # env.render()
    # Build flat obs for the actor from the dict
    total_r = 0.0
    steps   = 0
    eval_mode = (episode % eval_every_eps == 0 and episode > 0)

    for step in range(max_steps):
        # --- Ego action (DDPG): random during warmup, otherwise actor (+noise only in training) ---
        if not eval_mode and global_step < warmup_steps:
            ego_action = np.random.uniform(low=action_low, high=action_high).astype(np.float32)
        else:
            ego_action = agent.choose_action(obs, training=not eval_mode).astype(np.float32)

        # Opponent (rule-based) from current obs_dict
        opp_action = gap_follow_action(info["scans"][1]).astype(np.float32)

        # Stack actions for both agents -> shape (2, 2)
        actions = np.stack([ego_action, opp_action], axis=0).astype(np.float32)

        # Step env
        next_obs, _, terminated, truncated, info = env.step(actions)

        # Rebuild flat obs for next step

        # Reward and done
        rew  = reward_fn(next_obs)
        r    = float(rew[0] if np.ndim(rew) else rew)
        done = bool(terminated or truncated)
        # print(info['poses_x'], info['poses_y'], info['poses_theta'])
        # Store transition (continuous actions)
        agent.remember(obs, ego_action, r, next_obs, done)

        # Learn (skip during warmup and eval episodes)
        if not eval_mode and global_step >= warmup_steps:
            agent.replay()

        total_r     += r
        steps       += 1
        global_step += 1
        obs     = next_obs
        

        # Optional early termination from reward helper
        if hasattr(reward_fn, 'is_stuck') and reward_fn.is_stuck():
            print(f"Early termination: agent stuck at step {step}")
            break

        if done:
            break
        # env.render()

        # Periodic checkpoint (optional)
        if not eval_mode and save_every and (global_step % save_every == 0):
            agent.save_model(os.path.basename(ckpt_name))
            print(f"Saved checkpoint @ step {global_step}")

            
    mode_str = "EVAL" if eval_mode else "TRAIN"
    print(f"Ep {episode:04d} [{mode_str}] | R: {total_r:.2f} | steps: {steps} | buf: {len(agent.memory)}")
    if total_r > best_reward:
        best_reward = total_r
        agent.save_model(os.path.basename("best.pt"))
        print(f" Best {best_reward:.2f} @ episode {episode}")

# Final save
agent.save_model(os.path.basename(ckpt_name))
env.close()
