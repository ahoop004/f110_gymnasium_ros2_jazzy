#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS2 PPO actor node for the gaplock task.

Loads a Stable Baselines 3 PPO model (ppo_model.zip from from_training/)
and publishes Twist commands at a fixed rate.

Topics (all configurable via ROS2 parameters):
  Subscribed:
    /scan               (sensor_msgs/LaserScan)
    /vicon/Limo_04/...  (geometry_msgs/TransformStamped) — attacker (ego)
    /vicon/Limo_02/...  (geometry_msgs/TransformStamped) — target (defender)
  Published:
    /cmd_vel            (geometry_msgs/Twist)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO

from gaplock_utils import (
    OBS_DIM,
    build_observation,
    init_agent_state,
    update_agent_state,
)


class RLActorNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_actor_ppo_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("primary_topic", "/vicon/Limo_04/Limo_04")
        self.declare_parameter("secondary_topic", "/vicon/Limo_02/Limo_02")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("use_safety", True)
        self.declare_parameter("hard_border", 1.0)
        self.declare_parameter("prevent_reverse", True)
        self.declare_parameter("prevent_reverse_min_speed", 0.01)
        self.declare_parameter("max_pose_age", 0.25)

        # Default checkpoint: from_training/ppo_model.zip relative to this file
        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_ckpt = os.path.join(script_dir, "..", "from_training", "ppo_model.zip")
        self.declare_parameter("ckpt", default_ckpt)

        scan_topic = self.get_parameter("scan_topic").value
        primary_topic = self.get_parameter("primary_topic").value
        secondary_topic = self.get_parameter("secondary_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.use_safety = bool(self.get_parameter("use_safety").value)
        self.hard_border = float(self.get_parameter("hard_border").value)
        self.prevent_reverse = bool(self.get_parameter("prevent_reverse").value)
        self.min_throttle = float(self.get_parameter("prevent_reverse_min_speed").value)
        self.max_pose_age = float(self.get_parameter("max_pose_age").value)
        self.ckpt_path: str = self.get_parameter("ckpt").value

        # ── State ─────────────────────────────────────────────────────────────
        self.last_scan: Optional[np.ndarray] = None
        self.primary_state = init_agent_state()    # attacker (ego)
        self.secondary_state = init_agent_state()  # target (defender)

        # ── Subscriptions / Publisher ──────────────────────────────────────────
        self.create_subscription(LaserScan, scan_topic, self.on_scan, 1)
        self.create_subscription(TransformStamped, primary_topic, self.on_primary, 1)
        self.create_subscription(TransformStamped, secondary_topic, self.on_secondary, 1)
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        # ── Load SB3 model ────────────────────────────────────────────────────
        if not os.path.isfile(self.ckpt_path):
            self.get_logger().fatal(f"PPO checkpoint not found: {self.ckpt_path}")
            raise FileNotFoundError(self.ckpt_path)

        self.model = PPO.load(self.ckpt_path, device="cpu")
        self.get_logger().info(
            f"PPO actor ready | obs_dim={OBS_DIM} | ckpt={self.ckpt_path}"
        )

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.rate_hz, self.on_tick)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_scan(self, msg: LaserScan) -> None:
        self.last_scan = np.asarray(msg.ranges, dtype=np.float32)

    def on_primary(self, msg: TransformStamped) -> None:
        update_agent_state(self.primary_state, msg)

    def on_secondary(self, msg: TransformStamped) -> None:
        update_agent_state(self.secondary_state, msg)

    def on_tick(self) -> None:
        # ── Guard: wait for first data ─────────────────────────────────────
        if (
            self.last_scan is None
            or self.primary_state["pose"] is None
            or self.secondary_state["pose"] is None
        ):
            self.pub_cmd.publish(Twist())
            return

        # ── Guard: stale poses ─────────────────────────────────────────────
        now = self.get_clock().now().nanoseconds * 1e-9
        for label, state in (("primary", self.primary_state), ("secondary", self.secondary_state)):
            stamp = state["stamp"]
            if stamp is None or (now - stamp) > self.max_pose_age:
                self.get_logger().warn(
                    f"{label} pose stale -> zero command",
                    throttle_duration_sec=2.0,
                )
                self.pub_cmd.publish(Twist())
                return

        # ── Guard: safety border ───────────────────────────────────────────
        if self.use_safety:
            sec_y = float(self.secondary_state["pose"][1])
            if abs(sec_y) > self.hard_border:
                self.get_logger().warn(
                    f"Target |y|={sec_y:.2f} > {self.hard_border:.2f} -> stopping",
                    throttle_duration_sec=2.0,
                )
                self.pub_cmd.publish(Twist())
                return

        # ── Inference ──────────────────────────────────────────────────────
        obs_vec = build_observation(self.last_scan, self.primary_state, self.secondary_state)
        action, _ = self.model.predict(obs_vec, deterministic=True)

        steer = float(action[0])
        throttle = float(action[1])
        if self.prevent_reverse:
            throttle = max(throttle, self.min_throttle)

        cmd = Twist()
        cmd.angular.z = steer
        cmd.linear.x = throttle
        self.pub_cmd.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLActorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
