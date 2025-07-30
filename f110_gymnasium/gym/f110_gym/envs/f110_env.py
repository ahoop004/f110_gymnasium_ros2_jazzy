# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
import gymnasium as gym
from gymnasium import error, spaces, utils
import yaml
from PIL import Image

# base classes
from f110_gym.envs.base_classes import Simulator, Integrator

# others
import numpy as np
import os
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH
    
    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            
            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
        
            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
        
            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
            
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft
    """
    metadata = {'render_modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):        
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 42
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # default integrator
        try:
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4
            
        # default LiDAR position
        try:
            self.lidar_dist = kwargs['lidar_dist']
        except:
            self.lidar_dist = 0.0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))
            

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator, lidar_dist=self.lidar_dist)
        self.sim.set_map(self.map_path, self.map_ext)
        
        meta = yaml.safe_load(open('/home/aaron/f110_gymnasium_ros2_jazzy/assets/maps/levine.yaml'))
        R = meta['resolution']
        x0, y0, _ = meta.get('origin', (0.0, 0.0, 0.0))
        img = Image.open('/home/aaron/f110_gymnasium_ros2_jazzy/assets/maps/' + meta['image'])
        width, height = img.size
        x_min = x0
        x_max = x0 + width * R
        y_min = y0
        y_max = y0 + height * R

        # stateful observations for rendering
        self.render_obs = None
        single_action_space = spaces.Box(
            low=np.array([self.params['s_min'], self.params['v_min']], dtype=np.float32),
            high=np.array([self.params['s_max'], self.params['v_max']], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Tuple((single_action_space, single_action_space))
        
        
        scan_space = spaces.Box(low=0.0, high=30.0, shape=(1080,), dtype=np.float32)
        pose_space = spaces.Box(
            low=np.array([x_min, y_min, -np.pi], dtype=np.float32),
            high=np.array([x_max, y_max, np.pi], dtype=np.float32),
            dtype=np.float32
        )
        agent_obs_space = spaces.Dict({
            'scan': scan_space,
            'pose': pose_space,
            'collision': spaces.Discrete(2),
        })
        
        spaces_dict_obs = spaces.Dict({
            'ego_idx': spaces.Discrete(self.num_agents),
            'scans': spaces.Box(low=0.0, high=30.0, shape=(self.num_agents, 1080), dtype=np.float32),
            'poses_x': spaces.Box(low=x_min, high=x_max, shape=(self.num_agents,), dtype=np.float32),
            'poses_y': spaces.Box(low=y_min, high=y_max, shape=(self.num_agents,), dtype=np.float32),
            'poses_theta': spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_agents,), dtype=np.float32),
            'linear_vels_x': spaces.Box(low=self.params['v_min'], high=self.params['v_max'], shape=(self.num_agents,), dtype=np.float32),
            'linear_vels_y': spaces.Box(low=self.params['v_min'], high=self.params['v_max'], shape=(self.num_agents,), dtype=np.float32),
            'ang_vels_z': spaces.Box(low=0.0, high=10.0, shape=(self.num_agents,), dtype=np.float32),
            'collisions': spaces.MultiBinary(self.num_agents),  # 0/1 collision flags
            'lap_times': spaces.Box(low=0.0, high=100000.0, shape=(self.num_agents,), dtype=np.float32),
            'lap_counts': spaces.Box(low=0, high=10, shape=(self.num_agents,), dtype=np.float32),
        })
        
        
        
        
        # self.observation_space = spaces_dict_obs
        self.observation_space = spaces.Box(low=0.0, high=30.0, shape=(self.num_agents, 1080), dtype=np.float32)

        

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done
        
        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2
        
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        
        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations
        
        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']
        
    def clean_obs(self,obs):
        # Fields that should be numpy arrays of float32
        array_fields = [
            'scans', 'poses_x', 'poses_y', 'poses_theta',
            'linear_vels_x', 'linear_vels_y', 'ang_vels_z'
        ]
        for k in array_fields:
            v = obs.get(k)
            if isinstance(v, list) or (isinstance(v, np.ndarray) and v.dtype != np.float32):
                obs[k] = np.array(v, dtype=np.float32)
        # collisions, lap_times, lap_counts are already correct!
        # ego_idx is an int (good)
        return obs

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        
        # call simulation step
        obs = self.sim.step(action)

        
        obs['lap_times'] = self.lap_times.astype(np.float32)
        obs['lap_counts'] = self.lap_counts.astype(np.float32)

        F110Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep
        

        self._update_state(obs)

        # check done
        terminated, toggle_list = self._check_done()
        truncated = False
        # info = {'checkpoint_done': toggle_list}
        info = {k: v for k, v in obs.items() if k != 'scans'}
        info['checkpoint_done'] = toggle_list

        # LiDAR obs: always float32 array
        lidar_obs = np.array(obs['scans'], dtype=np.float32)
        lidar_obs = np.clip(lidar_obs, 0.0, 30.0) 
        return lidar_obs, reward, terminated, truncated, info
  
        # return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        poses = options
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, terminated, truncated, info = self.step(action)
        lidar_obs = obs
        

      
        self.render_obs = {
            'ego_idx': info['ego_idx'],
            'poses_x': info['poses_x'],
            'poses_y': info['poses_y'],
            'poses_theta': info['poses_theta'],
            'lap_times': info['lap_times'],
            'lap_counts': info['lap_counts']
            }
        lidar_obs = np.clip(lidar_obs, 0.0, 30.0) 
        return lidar_obs, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles
        
        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        
        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)
            
        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)
        
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
