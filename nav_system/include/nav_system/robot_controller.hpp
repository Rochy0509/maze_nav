#ifndef ROBOT_CONTROLLER_HPP
#define ROBOT_CONTROLLER_HPP

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

class RobotController : public rclcpp::Node
{
public:
    explicit RobotController(const std::string & node_name = "robot_controller");

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void timerCallback();

    // Current state
    double current_x_ = 0.0;
    double current_y_ = 0.0;
    double current_yaw_ = 0.0;

    // Target state
    double target_x_ = 0.0;
    double target_y_ = 0.0;
    double target_yaw_ = 0.0;
    bool has_target_ = false;

    // Publishers & Subscribers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Control gains
    const double k_p_linear_ = 0.5;   // P gain for linear speed
    const double k_p_angular_ = 1.0;  // P gain for angular speed
    const double max_linear_speed_ = 0.5;   // m/s
    const double max_angular_speed_ = 1.0;  // rad/s

    // Helper: get yaw from quaternion
    double getYaw(const geometry_msgs::msg::Quaternion& q);
};

#endif // ROBOT_CONTROLLER_HPP