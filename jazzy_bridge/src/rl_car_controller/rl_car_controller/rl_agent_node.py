import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import torch
import numpy as np

from .policy import GaussianPolicy  # or import your own policy class

class RLEgoController(Node):
    def __init__(self):
        super().__init__('rl_ego_controller')
        self.policy = GaussianPolicy(obs_dim=1080, act_dim=2)  # adjust as needed
        self.policy.load_state_dict(torch.load(
            '/home/aaron/f110_gymnasium_ros2_jazzy/jazzy_bridge/src/rl_car_controller/rl_car_controller/policy_final.pth', map_location='cpu'))  # Update path
        self.policy.eval()
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

    def scan_callback(self, msg):
        scan = np.array(msg.ranges, dtype=np.float32)
        with torch.no_grad():
            mean, std = self.policy(torch.from_numpy(scan).unsqueeze(0))
            action = mean[0].numpy()  # Use deterministic action (mean)
        steering, speed = float(action[0]), float(action[1])
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        self.publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLEgoController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
