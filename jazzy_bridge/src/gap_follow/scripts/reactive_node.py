import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.scan_subscriber = self.create_subscription(
            LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

    def preprocess_lidar(self, ranges):
        proc_ranges = np.array(ranges)
        proc_ranges[proc_ranges > 3.0] = 3.0
        proc_ranges = np.convolve(proc_ranges, np.ones(5)/5, 'same')
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        masked = np.ma.masked_equal(free_space_ranges, 0)
        slices = np.ma.clump_unmasked(masked)
        max_gap = max(slices, key=lambda s: s.stop - s.start)
        return max_gap.start, max_gap.stop

    def find_best_point(self, start_i, end_i, ranges):
        return np.argmax(ranges[start_i:end_i]) + start_i

    def lidar_callback(self, data):
        ranges = np.array(data.ranges)
        proc_ranges = self.preprocess_lidar(ranges)

        # Find closest point to LiDAR
        closest_point = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        bubble_radius = 50
        min_idx = max(closest_point - bubble_radius, 0)
        max_idx = min(closest_point + bubble_radius, len(proc_ranges) - 1)
        proc_ranges[min_idx:max_idx] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best_point = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Calculate steering angle based on best point
        angle = (best_point - len(proc_ranges)/2) * data.angle_increment

        # Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = 1.0

        self.drive_publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
