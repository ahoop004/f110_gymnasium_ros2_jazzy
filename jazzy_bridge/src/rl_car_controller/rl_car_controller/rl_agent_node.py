# rl_car_controller/rl_agent_node.py
import os
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class SimpleMLP(nn.Module):
    """Fallback MLP if you only have a state_dict and no model class available."""
    def __init__(self, input_dim: int, hidden: int = 64, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            
        )
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class RLRacecarNode(Node):
    def __init__(self):
        super().__init__('rl_racecar_node')

        # Parameters (overridable via launch/params)
        self.declare_parameter('scan_topic', 'scan')     # relative: /<ns>/scan
        self.declare_parameter('drive_topic', 'drive')   # relative: /<ns>/drive
        self.declare_parameter('model_relpath', '/home/aaron/f110_gymnasium_ros2_jazzy/jazzy_bridge/src/rl_car_controller/rl_car_controller/policy_final.pth')
        self.declare_parameter('hidden_size', 64)       # only used by SimpleMLP
        self.declare_parameter('max_speed', 4.0)         # clamp speed if needed
        self.declare_parameter('max_steer', 0.4189)      # ~24deg

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        model_rel  = self.get_parameter('model_relpath').get_parameter_value().string_value

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.hidden    = int(self.get_parameter('hidden_size').value)

        # Resolve model path in package share
        share_dir  = get_package_share_directory('rl_car_controller')
        self.model_path = os.path.join(share_dir, model_rel)

        # We’ll finalize the model when we know LiDAR input size
        self.model = None
        self.ckpt  = None
        self._load_checkpoint_eager()

        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos)
        self.publisher    = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info(f"Waiting for scans on '{scan_topic}', publishing to '{drive_topic}'")

    # --- Loading logic that tolerates TorchScript, full module, or state_dict ---
    def _load_checkpoint_eager(self):
        # 1) Try TorchScript
        try:
            m = torch.jit.load(self.model_path, map_location='cpu')
            m.eval()
            self.model = m
            self.get_logger().info("Loaded TorchScript model.")
            return
        except Exception:
            pass

        # 2) Fallback to generic torch.load (may return nn.Module or state_dict)
        try:
            self.ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
            if isinstance(self.ckpt, torch.nn.Module):
                self.ckpt.eval()
                self.model = self.ckpt
                self.get_logger().info("Loaded full PyTorch nn.Module.")
            else:
                self.get_logger().info("Loaded checkpoint/state_dict; will build model on first scan.")
        except TypeError:
            # Older torch without weights_only kw
            self.ckpt = torch.load(self.model_path, map_location='cpu')
            if isinstance(self.ckpt, torch.nn.Module):
                self.ckpt.eval()
                self.model = self.ckpt
                self.get_logger().info("Loaded full PyTorch nn.Module (older torch).")
            else:
                self.get_logger().info("Loaded checkpoint/state_dict (older torch); will build model on first scan.")

    def _maybe_build_from_state(self, input_dim: int):
        """If self.model is None but ckpt is a state_dict-like, build SimpleMLP and load."""
        if self.model is not None:
            return
        if self.ckpt is None:
            raise RuntimeError("No checkpoint available to build model.")

        # Decide which dict holds weights
        if isinstance(self.ckpt, dict) and 'state_dict' in self.ckpt:
            state_dict = self.ckpt['state_dict']
        else:
            state_dict = self.ckpt  # could be OrderedDict

        net = SimpleMLP(input_dim=input_dim, hidden=self.hidden, output_dim=2)
        # Try clean load; tolerate mismatched keys with strict=False if needed
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            self.get_logger().warn(f"State dict keys mismatch. missing={list(missing)} unexpected={list(unexpected)}")
        net.eval()
        self.model = net
        self.get_logger().info(f"Instantiated SimpleMLP with input_dim={input_dim} and loaded state dict.")

    # --- Inference ---
    def lidar_callback(self, msg: LaserScan):
        # Normalize ranges to [0,1]
        r = np.asarray(msg.ranges, dtype=np.float32)
        r = np.nan_to_num(r, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        r = np.clip(r, 0.0, msg.range_max)
        norm = r / max(msg.range_max, 1e-6)

        # If model wasn't fully loaded, create it now with the observed LiDAR length
        if self.model is None:
            self._maybe_build_from_state(input_dim=norm.shape[0])

        # Inference
        obs = torch.from_numpy(norm).unsqueeze(0)  # (1, N)
        with torch.no_grad():
            action = self.model(obs)

        # Expect action shape (1,2): [steer, speed]
        a = action.detach().cpu().numpy().reshape(-1)
        steer = float(np.clip(a[0], -self.max_steer, self.max_steer))
        speed = float(np.clip(a[1], -self.max_speed,  self.max_speed))

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = steer
        drive.drive.speed          = speed
        self.publisher.publish(drive)

def main():
    rclpy.init()
    node = RLRacecarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
