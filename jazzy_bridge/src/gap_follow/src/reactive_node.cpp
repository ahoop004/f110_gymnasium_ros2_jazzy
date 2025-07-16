#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

using std::placeholders::_1;

class ReactiveFollowGap : public rclcpp::Node {

   public:
    ReactiveFollowGap() : Node("reactive_node") {
        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(lidarscan_topic, 10, std::bind(&ReactiveFollowGap::lidar_callback, this, _1));
    }

   private:
    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";

    std::vector<double> processed_lidar;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;

    void preprocess_lidar(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        processed_lidar.clear();
        int window_size = 5;  // smoothing window

        for (size_t i = 0; i < scan_msg->ranges.size(); i++) {
            double avg = 0.0;
            int count = 0;

            for (int offset = -window_size / 2; offset <= window_size / 2; offset++) {
                int idx = i + offset;
                if (idx >= 0 && idx < (int)scan_msg->ranges.size()) {
                    avg += std::min<float>(scan_msg->ranges[idx], 3.0);
                    count++;
                }
            }

            processed_lidar.push_back(avg / count);
        }
    }

    void create_bubble(int closest_point, int bubble_radius) {
        int start_idx = std::max(closest_point - bubble_radius, 0);
        int end_idx = std::min(closest_point + bubble_radius, (int)processed_lidar.size() - 1);
        for (int i = start_idx; i <= end_idx; i++) {
            processed_lidar[i] = 0;
        }
    }

    std::pair<int, int> find_max_gap() {
        int largest_starting_i = 0;
        int longest_gap = 0;
        int curr_gap = 0;
        int curr_start = 0;

        for (size_t i = 0; i < processed_lidar.size(); i++) {
            if (processed_lidar[i] <= 0.5) {
                curr_gap = 0;
                curr_start = i + 1;
            } else {
                curr_gap++;
                if (curr_gap > longest_gap) {
                    largest_starting_i = curr_start;
                    longest_gap = curr_gap;
                }
            }
        }
        return {largest_starting_i, longest_gap};
    }

    int find_best_point(int starting_i, int gap_distance) {
        return starting_i + gap_distance / 2;  // simple midpoint for stability
    }

    double to_radians(double theta) {
        return M_PI * theta / 180.0;
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        preprocess_lidar(scan_msg);

        // Find closest point to LiDAR
        auto closest_point_iter = std::min_element(processed_lidar.begin(), processed_lidar.end());
        int closest_point = std::distance(processed_lidar.begin(), closest_point_iter);

        // Eliminate all points inside 'bubble'
        create_bubble(closest_point, 30);

        // Find max length gap
        auto max_gap = find_max_gap();

        // Find the best point in the gap
        int best_angle_i = find_best_point(max_gap.first, max_gap.second);

        // Publish Drive message
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.steering_angle = scan_msg->angle_min + best_angle_i * scan_msg->angle_increment;
        RCLCPP_INFO(this->get_logger(), "Steering angle: '%f'", drive_msg.drive.steering_angle);

        if (std::abs(drive_msg.drive.steering_angle) < to_radians(10)) {
            drive_msg.drive.speed = 1.5;
        } else if (std::abs(drive_msg.drive.steering_angle) < to_radians(20)) {
            drive_msg.drive.speed = 1.0;
        } else {
            drive_msg.drive.speed = 0.5;
        }
        publisher_->publish(drive_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}