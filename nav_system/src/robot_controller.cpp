#include "nav_system/robot_controller.hpp"
#include <cmath>

RobotController::RobotController(const std::string & node_name)
: Node(node_name)
{
    // Parameters 
    this->declare_parameter<double>("k_p_linear", k_p_linear_);
    this->declare_parameter<double>("k_p_angular", k_p_angular_);

    // Publishers
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

    // Subscribers
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom_filtered", 10, std::bind(&RobotController::odomCallback, this, std::placeholders::_1));
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu/data", 10, std::bind(&RobotController::imuCallback, this, std::placeholders::_1));

    // Timer (20 Hz control loop)
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&RobotController::timerCallback, this)
    );

    RCLCPP_INFO(this->get_logger(), "Robot controller started");
}

void RobotController::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;
}

void RobotController::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    current_yaw_ = getYaw(msg->orientation);
}

double RobotController::getYaw(const geometry_msgs::msg::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 m(tf_q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return yaw;
}

void RobotController::timerCallback()
{
    // For now: hardcoded target 
    static bool first_run = true;
    if (first_run) {
        target_x_ = 1.0;   // go 1m forward
        target_y_ = 0.0;
        target_yaw_ = 0.0; // face forward
        has_target_ = true;
        first_run = false;
    }

    if (!has_target_) return;

    // Compute errors
    double dx = target_x_ - current_x_;
    double dy = target_y_ - current_y_;
    double distance_error = std::sqrt(dx*dx + dy*dy);

    // Desired heading to target
    double desired_yaw = std::atan2(dy, dx);
    double heading_error = desired_yaw - current_yaw_;
    // Normalize to [-pi, pi]
    while (heading_error > M_PI) heading_error -= 2 * M_PI;
    while (heading_error < -M_PI) heading_error += 2 * M_PI;

    // Control logic
    auto cmd = geometry_msgs::msg::Twist();

    if (distance_error > 0.05) { // 5 cm tolerance
        // Move toward target
        cmd.linear.x = std::min(k_p_linear_ * distance_error, max_linear_speed_);
        cmd.angular.z = std::min(std::max(k_p_angular_ * heading_error, -max_angular_speed_), max_angular_speed_);
    } else {
        // Hold final heading
        double final_heading_error = target_yaw_ - current_yaw_;
        while (final_heading_error > M_PI) final_heading_error -= 2 * M_PI;
        while (final_heading_error < -M_PI) final_heading_error += 2 * M_PI;
        
        cmd.angular.z = std::min(std::max(k_p_angular_ * final_heading_error, -max_angular_speed_), max_angular_speed_);
    }

    cmd_vel_pub_->publish(cmd);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}