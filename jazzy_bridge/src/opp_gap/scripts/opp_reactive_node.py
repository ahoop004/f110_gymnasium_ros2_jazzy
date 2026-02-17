#!/usr/bin/env python3
"""Opponent FTG controller node using FollowTheGapPolicy from training."""

import os
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from opp_gap.ftg import FollowTheGapPolicy


# Default FTG params matching ftg_max.yaml from training
DEFAULT_FTG_PARAMS = {
    'max_distance': 10.0,
    'window_size': 4,
    'bubble_radius': 4.5,
    'max_steer': 0.42,
    'min_speed': 0.2,
    'max_speed': 0.95,
    'steering_gain': 0.8,
    'fov': 4.71238898,
    'normalized': False,
    'steer_smooth': 0.6,
    'mode': 'lidar',
    'gap_min_range': 0.4,
    'target_mode': 'center',
    'wall_avoid_kick': 0.02,
    'panic_factor_near': 1.0,
    'panic_factor_very_near': 1.0,
    'use_disparity_extender': True,
    'disparity_threshold': 0.35,
    'vehicle_width': 0.225,
    'safety_margin': 0.08,
    'no_cutback_enabled': True,
    'cutback_clearance': 0.9,
    'cutback_hold_steps': 8,
}


class OppFTGNode(Node):
    def __init__(self):
        super().__init__('opp_ftg_node')

        self.declare_parameter(
            'ftg_config',
            '/home/aaron/f110_gymnasium_ros2_jazzy/from_training/ftg_max.yaml',
        )

        # Load FTG config
        cfg_path = self.get_parameter('ftg_config').get_parameter_value().string_value
        params = dict(DEFAULT_FTG_PARAMS)
        if cfg_path and os.path.isfile(cfg_path):
            with open(cfg_path, 'r') as f:
                loaded = yaml.safe_load(f)
            # ftg_max.yaml has agents.car_1.params structure
            if isinstance(loaded, dict):
                agent_cfg = loaded
                if 'agents' in agent_cfg:
                    for aid, acfg in agent_cfg['agents'].items():
                        if isinstance(acfg, dict) and 'params' in acfg:
                            params.update(acfg['params'])
                            break
                elif 'params' in agent_cfg:
                    params.update(agent_cfg['params'])
                else:
                    params.update(agent_cfg)
            self.get_logger().info(f'Loaded FTG config from {cfg_path}')
        else:
            self.get_logger().warn(f'FTG config not found at {cfg_path}, using defaults')

        self.ftg = FollowTheGapPolicy.from_config(params)
        self.get_logger().info(
            f'FTG policy: max_speed={self.ftg.max_speed}, '
            f'steering_gain={self.ftg.steering_gain}, '
            f'bubble_radius={self.ftg.bubble_radius}'
        )

        # Current velocity from odom
        self.current_velocity = np.zeros(2, dtype=np.float32)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/opp_scan', self.scan_callback, qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, qos
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/opp_drive', 10
        )

    def odom_callback(self, msg: Odometry):
        self.current_velocity[0] = msg.twist.twist.linear.x
        self.current_velocity[1] = msg.twist.twist.linear.y

    def scan_callback(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        ranges = np.clip(ranges, 0.0, msg.range_max)

        obs = {
            'scans': ranges,
            'velocity': self.current_velocity,
        }

        action = self.ftg.get_action(None, obs)
        steer = float(action[0])
        speed = float(action[1])

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OppFTGNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
