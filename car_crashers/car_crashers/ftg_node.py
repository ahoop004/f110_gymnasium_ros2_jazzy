import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import numpy as np

from car_crashers.gap_follow import FollowTheGapPolicy

class gap_follower(Node):

    def __init__(self):
        super().__init__('gap_follower')
        #create publisher for driving
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.2, self.update)
        #create subscribers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(LaserScan, '/scan', self.onScan, qos)
        #driving policy initialized
        self.policy = FollowTheGapPolicy(
            min_speed=0.1,
            max_speed=0.5,
            max_steer=1,
            fov=np.deg2rad(360),
            max_distance=12.0,
            bubble_radius=0.1,
            steering_gain=20, #0.6 by default
        )
        #configure state space info
        self.scan = LaserScan()
        self.current_velocity = np.array([0.0, 0.0, 0.0])

    def update(self):
        #preprocess the scan if we need to
        obs = {
            'scans' : np.array(self.scan.ranges),
        }
        if len(obs['scans']) == 0:
            self.get_logger().warn("Empty scan!")
            return
        action = self.policy.get_action(None,obs)
        steering = float(action[0])*2
        speed = float(action[1])
        print(str(speed) + ", " + str(steering))
        #create message
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = steering
        self.pub.publish(cmd)


    def onScan(self, msg):
        self.scan = msg





def main(args=None):
    rclpy.init(args=args)
    robot = gap_follower()

    rclpy.spin(robot)
    #spin is blocking, do cleanup 
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
