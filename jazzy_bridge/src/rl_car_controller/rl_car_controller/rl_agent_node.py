#!/usr/bin/env python3
"""ROS2 node for PPO attacker inference in the gaplock scenario.

Loads policy weights directly from the SB3 zip (PyTorch only, no SB3 import
needed at runtime) to avoid numpy version conflicts.

Subscribes to:
  - /scan (ego LiDAR)
  - ego_racecar/odom (ego odometry)
  - ego_racecar/opp_odom (opponent odometry)

Publishes:
  - /drive (AckermannDriveStamped)

The PPO model was trained with flatten_gaplock_obs producing a 65-dim vector:
  - 54 downsampled LiDAR beams (1080 -> 54, normalized to [0,1], max_range=12m)
  - 3 ego velocity (vx, vy, omega)
  - 3 target velocity (vx, vy, omega)
  - 5 relative pose (rel_x, rel_y, sin(dtheta), cos(dtheta), distance)

Actions are normalized [-1, 1] mapped to:
  - steer: [-0.46, 0.46] rad
  - speed: [-1.0, 1.0] m/s (reverse prevented -> clamped to [0, 1.0])
"""

import io
import os
import zipfile
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d.euler import quat2euler

from rl_car_controller.obs_flatten import flatten_gaplock_obs


class SB3PolicyNet(nn.Module):
    """Reconstructed SB3 ActorCritic policy network (actor only)."""

    def __init__(self, obs_dim: int, act_dim: int = 2, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp_extractor_policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.mlp_extractor_policy_net(obs)
        return self.action_net(latent)

    def load_sb3_state_dict(self, sd: dict):
        """Map SB3 state_dict keys to this network."""
        mapping = {}
        # Map mlp_extractor.policy_net.{i}.{weight,bias} -> mlp_extractor_policy_net.{i}.{weight,bias}
        for key, val in sd.items():
            if key.startswith('mlp_extractor.policy_net.'):
                new_key = key.replace('mlp_extractor.policy_net.', 'mlp_extractor_policy_net.')
                mapping[new_key] = val
            elif key.startswith('action_net.'):
                mapping[key] = val
            elif key == 'log_std':
                mapping[key] = val
            # Skip value_net keys (not needed for inference)
        self.load_state_dict(mapping, strict=True)


class PPOAttackerNode(Node):
    def __init__(self):
        super().__init__('ppo_attacker_node')

        self.declare_parameter(
            'model_path',
            '/home/aaron/f110_gymnasium_ros2_jazzy/from_training/ppo_model.zip',
        )
        self.declare_parameter('steer_min', -0.46)
        self.declare_parameter('steer_max', 0.46)
        self.declare_parameter('speed_min', -1.0)
        self.declare_parameter('speed_max', 1.0)
        self.declare_parameter('prevent_reverse', True)
        self.declare_parameter('lidar_beams', 54)
        self.declare_parameter('lidar_max_range', 12.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.steer_min = self.get_parameter('steer_min').value
        self.steer_max = self.get_parameter('steer_max').value
        self.speed_min = self.get_parameter('speed_min').value
        self.speed_max = self.get_parameter('speed_max').value
        self.prevent_reverse = self.get_parameter('prevent_reverse').value
        self.lidar_beams = self.get_parameter('lidar_beams').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value

        # Action mapping bounds
        self.action_low = np.array([self.steer_min, self.speed_min], dtype=np.float32)
        self.action_high = np.array([self.steer_max, self.speed_max], dtype=np.float32)

        # Obs normalization scales (matching training)
        self.obs_scales = {
            'lidar_range': self.lidar_max_range,
            'speed': 1.0,
            'position': self.lidar_max_range,
        }

        # Expected obs dim: lidar_beams + 3 (ego vel) + 3 (target vel) + 5 (rel pose)
        self.obs_dim = self.lidar_beams + 11

        # State
        self.ego_pose = np.zeros(3, dtype=np.float32)  # x, y, theta
        self.ego_vel = np.zeros(3, dtype=np.float32)    # vx, vy, omega
        self.opp_pose = np.zeros(3, dtype=np.float32)
        self.opp_vel = np.zeros(3, dtype=np.float32)
        self.has_opp_odom = False

        # Load policy weights directly from SB3 zip (no SB3 import needed)
        self.policy = self._load_policy(model_path)
        self.get_logger().info(
            f'Loaded PPO policy from {model_path} '
            f'(obs_dim={self.obs_dim}, lidar_beams={self.lidar_beams})'
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos
        )
        self.ego_odom_sub = self.create_subscription(
            Odometry, 'odom', self.ego_odom_callback, qos
        )
        self.opp_odom_sub = self.create_subscription(
            Odometry, 'opp_odom', self.opp_odom_callback, qos
        )

        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.get_logger().info('PPO attacker node ready')

    def _load_policy(self, zip_path: str) -> SB3PolicyNet:
        """Extract policy weights from SB3 zip and build PyTorch network."""
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('policy.pth') as f:
                buf = io.BytesIO(f.read())
                state_dict = torch.load(buf, map_location='cpu', weights_only=False)

        # Infer architecture from state dict
        first_weight = state_dict['mlp_extractor.policy_net.0.weight']
        obs_dim = first_weight.shape[1]
        if obs_dim != self.obs_dim:
            self.get_logger().warn(
                f'Model obs_dim={obs_dim} differs from expected={self.obs_dim}. '
                f'Adjusting lidar_beams from {self.lidar_beams} to {obs_dim - 11}.'
            )
            self.lidar_beams = obs_dim - 11
            self.obs_dim = obs_dim

        hidden_dims = []
        i = 0
        while f'mlp_extractor.policy_net.{i}.weight' in state_dict:
            hidden_dims.append(state_dict[f'mlp_extractor.policy_net.{i}.weight'].shape[0])
            i += 2  # skip ReLU layers (they have no weights)

        act_dim = state_dict['action_net.weight'].shape[0]

        net = SB3PolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims)
        net.load_sb3_state_dict(state_dict)
        net.eval()
        return net

    def ego_odom_callback(self, msg: Odometry):
        self.ego_pose[0] = msg.pose.pose.position.x
        self.ego_pose[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z], axes='sxyz')
        self.ego_pose[2] = yaw
        self.ego_vel[0] = msg.twist.twist.linear.x
        self.ego_vel[1] = msg.twist.twist.linear.y
        self.ego_vel[2] = msg.twist.twist.angular.z

    def opp_odom_callback(self, msg: Odometry):
        self.opp_pose[0] = msg.pose.pose.position.x
        self.opp_pose[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z], axes='sxyz')
        self.opp_pose[2] = yaw
        self.opp_vel[0] = msg.twist.twist.linear.x
        self.opp_vel[1] = msg.twist.twist.linear.y
        self.opp_vel[2] = msg.twist.twist.angular.z
        self.has_opp_odom = True

    def _downsample_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """Downsample 1080 LiDAR beams to self.lidar_beams."""
        n = len(ranges)
        target = self.lidar_beams
        if n == target:
            return ranges
        indices = np.linspace(0, n - 1, target, dtype=int)
        return ranges[indices]

    def scan_callback(self, msg: LaserScan):
        if not self.has_opp_odom:
            return

        # Process LiDAR
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0
        )
        ranges = np.clip(ranges, 0.0, self.lidar_max_range)

        # Downsample
        ranges_ds = self._downsample_lidar(ranges)

        # Build the observation dict that flatten_gaplock_obs expects.
        central_state = np.array([
            self.ego_pose[0], self.opp_pose[0],   # poses_x
            self.ego_pose[1], self.opp_pose[1],   # poses_y
            self.ego_pose[2], self.opp_pose[2],   # poses_theta
            self.ego_vel[0],  self.opp_vel[0],    # linear_vels_x
            self.ego_vel[1],  self.opp_vel[1],    # linear_vels_y
            self.ego_vel[2],  self.opp_vel[2],    # ang_vels_z
            0.0,              0.0,                 # collisions
        ], dtype=np.float32)

        obs_dict = {
            'scans': ranges_ds,
            'pose': self.ego_pose.copy(),
            'velocity': self.ego_vel[:2].copy(),
            'angular_velocity': float(self.ego_vel[2]),
            'central_state': central_state,
        }

        # Flatten to obs_dim vector
        flat_obs = flatten_gaplock_obs(
            obs_dict,
            target_id='car_1',
            scales=self.obs_scales,
        )

        # Run policy inference
        obs_tensor = torch.from_numpy(flat_obs).unsqueeze(0)
        with torch.no_grad():
            action_norm = self.policy(obs_tensor).squeeze(0).numpy()

        # Clamp to [-1, 1]
        action_norm = np.clip(action_norm, -1.0, 1.0)

        # Prevent reverse
        if self.prevent_reverse and action_norm[1] < 0.0:
            action_norm[1] = 0.0

        # Map normalized [-1, 1] to actual action space
        continuous = self.action_low + (action_norm + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )

        steer = float(np.clip(continuous[0], self.steer_min, self.steer_max))
        speed = float(np.clip(
            continuous[1],
            0.0 if self.prevent_reverse else self.speed_min,
            self.speed_max,
        ))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PPOAttackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
