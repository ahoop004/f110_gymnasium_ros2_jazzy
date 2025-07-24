import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import torch
import numpy as np

class RLRacecarNode(Node):
    def __init__(self):
        super().__init__('rl_racecar_node')
        self.model = torch.load('path/to/your_model.pth', map_location=torch.device('cpu'))
        self.model.eval()
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  # or your car's lidar topic
            self.lidar_callback,
            10
        )
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def lidar_callback(self, msg):
        # Preprocess lidar data as your model expects
        obs = np.array(msg.ranges, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # shape: (1, N)
        with torch.no_grad():
            action = self.model(obs_tensor)
        # Assume action = [steering, speed], modify as needed
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(action[0, 0])
        drive_msg.drive.speed = float(action[0, 1])
        self.publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLRacecarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
