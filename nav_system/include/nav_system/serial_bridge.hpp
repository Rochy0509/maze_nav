#ifndef SERIAL_BRIDGE_HPP
#define SERIAL_BRIDGE_HPP

#include <memory>
#include <string>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "libserial/SerialPort.h"

class SerialBridge : public rclcpp::Node
{
public:
    explicit SerialBridge(const std::string & node_name = "arduino_bridge");

private:
    void timer_callback();
    void process_line(const std::string& line);

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mag_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_raw_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr tof_right_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr tof_left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr tof_front_pub_;
    
    //Subscriber
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // Serial interface
    LibSerial::SerialPort serial_;

    size_t count_{0};

};

#endif // SERIAL_BRIDGE_HPP