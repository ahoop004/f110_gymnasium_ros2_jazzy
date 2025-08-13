# rl_ego_ddpg_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LIDAR_MAX_M = 30.0  # meters; adjust if your simulator differs

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)
        self.register_buffer("action_low",  torch.tensor(action_low,  dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc2.bias); nn.init.zeros_(self.fc3.bias)

    def forward(self, obs):
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        x = F.relu(self.fc1(obs)); x = F.relu(self.fc2(x))
        t = torch.tanh(self.fc3(x))  # [-1,1]
        low, high = self.action_low, self.action_high
        return 0.5 * (high - low) * t + 0.5 * (high + low)

class BoxScaler:
    def __init__(self, low, high, eps=1e-6):
        low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
        self.mid  = (high + low) / 2.0
        self.half = np.maximum((high - low) / 2.0, eps)
    def transform(self, x):
        y = (np.asarray(x, np.float32) - self.mid) / self.half
        return np.clip(y, -1.0, 1.0).astype(np.float32)

def wrap_angle(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi

class RLEgoController(Node):
    def __init__(self):
        super().__init__('rl_ego_controller')

        # ---- Load DDPG checkpoint ----
        ckpt_path = '/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/DDPG/models/ddpg_checkpoint.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        obs_dim = int(ckpt.get("obs_dim", 1088))
        act_dim = int(ckpt.get("act_dim", 2))
        action_low  = np.array(ckpt.get("action_low",  [-0.4189, 0.0]),  dtype=np.float32)
        action_high = np.array(ckpt.get("action_high", [ 0.4189, 7.0]),  dtype=np.float32)

        self.actor = Actor(obs_dim, act_dim, action_low, action_high)
        self.actor.load_state_dict(ckpt["actor"], strict=True)
        self.actor.eval()

        # Optional: load obs scaling bounds if you saved them (recommended)
        self.scaler = None
        if "obs_low" in ckpt and "obs_high" in ckpt:
            self.scaler = BoxScaler(ckpt["obs_low"], ckpt["obs_high"])

        # ---- Subscriptions / publications ----
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        # Change these topic names to your sim’s:
        self.sub_ego_odom = self.create_subscription(Odometry, '/ego_odom', self.ego_odom_cb, 10)
        self.sub_opp_odom = self.create_subscription(Odometry, '/opp_odom', self.opp_odom_cb, 10)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # State buffers
        self.last_scan = None
        self.ego_xyth = np.array([0.0, 0.0, 0.0], np.float32)
        self.opp_xyth = np.array([0.0, 0.0, 0.0], np.float32)
        self.ego_col = 0.0
        self.opp_col = 0.0

        self.get_logger().info("DDPG ROS2 controller ready.")

    def ego_odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Yaw from quaternion
        q = msg.pose.pose.orientation
        # yaw extraction
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.ego_xyth[:] = [x, y, wrap_angle(yaw)]

    def opp_odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.opp_xyth[:] = [x, y, wrap_angle(yaw)]

    def scan_cb(self, msg: LaserScan):
        # 1) Build flat obs (same order as in training env)
        scan = np.asarray(msg.ranges, np.float32)
        scan = np.nan_to_num(scan, nan=LIDAR_MAX_M, posinf=LIDAR_MAX_M, neginf=0.0)
        scan = np.clip(scan, 0.0, LIDAR_MAX_M) / LIDAR_MAX_M  # normalize to [0,1]
        if scan.shape[0] != 1080:
            # If your LiDAR count differs, resample or adjust the network accordingly
            scan = np.resize(scan, 1080)

        obs_flat = np.concatenate([
            scan,
            np.array([
                self.ego_xyth[0], self.ego_xyth[1], self.ego_xyth[2], self.ego_col,
                self.opp_xyth[0], self.opp_xyth[1], self.opp_xyth[2], self.opp_col,
            ], np.float32)
        ], dtype=np.float32)

        # 2) Deterministic scaling (if available)
        x = self.scaler.transform(obs_flat) if self.scaler is not None else obs_flat

        # 3) Inference (deterministic; no exploration noise)
        with torch.no_grad():
            obs_t = torch.from_numpy(x)
            act_t = self.actor(obs_t)
            action = act_t[0].cpu().numpy()

        steering = float(np.clip(action[0], -0.4189, 0.4189))
        speed    = float(max(0.0, action[1]))  # no reverse

        # 4) Publish
        msg_out = AckermannDriveStamped()
        msg_out.drive.steering_angle = steering
        msg_out.drive.speed = speed
        self.pub_drive.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = RLEgoController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
