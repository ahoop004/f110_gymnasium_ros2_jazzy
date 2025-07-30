import yaml
import gymnasium as gym
from REINFORCE.agent import REINFORCEAgent
from REINFORCE.trainer import Trainer

with open('/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/configs/reinforce_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

env = gym.make('f110_gym:f110-v0', map=config['env_settings']['map'], map_ext=config['env_settings']['map_ext'], num_agents=2)

obs_dim = env.observation_space['scans'].shape[1]
act_dim = env.action_space.spaces[0].shape[0]

agent = REINFORCEAgent(obs_dim, act_dim, config['agent_hyperparameters']['learning_rate'])
trainer = Trainer(env, agent, config['agent_hyperparameters']['gamma'])

for episode in range(config['training_settings']['episodes']):
    total_reward = trainer.run_episode(
        config['env_settings']['start_poses'],
        config['training_settings']['max_steps']
    )
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
