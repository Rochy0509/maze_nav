#include "nav_system/serial_bridge.hpp"
#include "nav_system/robot_controller.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nlohmann/json.hpp"
#include <string>
#include <cstring>
#include <iostream>
#include <limits>

SerialBridge::SerialBridge(const std::string & node_name) 
: Node("arduino_bridge"), serial_()
{
    this->declare_parameter<std::string>("serial_port", "/dev/ttyACM0");
    this->declare_parameter<int>("baud_rate", 115200);

    std::string port_name = this->get_parameter("serial_port").as_string();
    int baud_rate = this->get_parameter("baud_rate").as_int();

    try{
        serial_.Open(port_name);
        serial_.SetBaudRate(static_cast<LibSerial::BaudRate>(baud_rate));
        serial_.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
        serial_.SetParity(LibSerial::Parity::PARITY_NONE);
        serial_.SetStopBits(LibSerial::StopBits::STOP_BITS_1);
        serial_.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);
        RCLCPP_INFO(this->get_logger(), "Serial port opened: %s at %d baud", port_name.c_str(), baud_rate);  // Fixed message
    } catch (const LibSerial::NotOpen &e){
        RCLCPP_ERROR(this->get_logger(), "Serial error: %s", e.what());
        return;
    }

    imu_raw_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/raw", 10);
    odom_raw_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom/", 10);
    tof_front_pub_ = this->create_publisher<sensor_msgs::msg::Range>("tof/front", 10);
    tof_left_pub_ = this->create_publisher<sensor_msgs::msg::Range>("tof/left", 10);
    tof_right_pub_ = this->create_publisher<sensor_msgs::msg::Range>("tof/right", 10);
    mag_pub_ = this->create_publisher<sensor_msgs::msg::MagneticField>("imu/mag", 10);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&SerialBridge::timer_callback, this)
    );

    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", 10,
    [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        char buffer[128];
        int len = snprintf(buffer, sizeof(buffer), "{\"cmd\":[%.3f,%.3f,%.3f]}\n",
                           msg->linear.x, msg->linear.y, msg->angular.z);
        if (len > 0) {
            try {
                std::string cmd_str(buffer, len);
                serial_.Write(cmd_str);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Serial write error: %s", e.what());
            }
        }
    });
}

void SerialBridge::timer_callback()
{
    if (!serial_.IsOpen()) {
        return;
    }

    try {
        std::string line_buffer;
        while (serial_.IsDataAvailable()) {
            char c;
            serial_.ReadByte(c, 1);
            
            if (c == '\n') {
                if (!line_buffer.empty()) {
                    process_line(line_buffer);
                    line_buffer.clear();
                }
            } else if (c != '\r') {
                line_buffer += c;
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                            "Serial read error: %s", e.what());
    }
}

void SerialBridge::process_line(const std::string& line)
{
    if (line.empty()) return;

    try {
        auto json = nlohmann::json::parse(line);

        if (json.contains("imu_raw")) {
            auto& arr = json["imu_raw"];
            if (arr.size() >= 9) {
                // Accelerometer: g → m/s²
                double ax = arr[0].get<double>() * 9.80665;
                double ay = arr[1].get<double>() * 9.80665;
                double az = arr[2].get<double>() * 9.80665;

                // Gyroscope: deg/s → rad/s
                double gx = arr[3].get<double>() * M_PI / 180.0;
                double gy = arr[4].get<double>() * M_PI / 180.0;
                double gz = arr[5].get<double>() * M_PI / 180.0;

                // Magnetometer: µT
                float mx = arr[6].get<float>();
                float my = arr[7].get<float>();
                float mz = arr[8].get<float>();

                // === Publish IMU 
                auto imu_msg = sensor_msgs::msg::Imu();
                imu_msg.header.stamp = this->now();
                imu_msg.header.frame_id = "imu_link";

                // No orientation
                imu_msg.orientation.w = 1.0;
                imu_msg.orientation.x = imu_msg.orientation.y = imu_msg.orientation.z = 0.0;
                imu_msg.orientation_covariance.fill(-1); 

                imu_msg.angular_velocity.x = gx;
                imu_msg.angular_velocity.y = gy;
                imu_msg.angular_velocity.z = gz;
                imu_msg.angular_velocity_covariance.fill(-1); 

                imu_msg.linear_acceleration.x = ax;
                imu_msg.linear_acceleration.y = ay;
                imu_msg.linear_acceleration.z = az;
                imu_msg.linear_acceleration_covariance.fill(-1);

                imu_raw_pub_->publish(imu_msg);

                // Publish Magnetometer
                auto mag_msg = sensor_msgs::msg::MagneticField();
                mag_msg.header.stamp = this->now();
                mag_msg.header.frame_id = "imu_link";
                mag_msg.magnetic_field.x = mx * 1e-6; 
                mag_msg.magnetic_field.y = my * 1e-6;
                mag_msg.magnetic_field.z = mz * 1e-6;
                mag_msg.magnetic_field_covariance.fill(-1); 

                mag_pub_->publish(mag_msg);

                RCLCPP_DEBUG(this->get_logger(), "IMU: a=(%.2f,%.2f,%.2f) g=(%.2f,%.2f,%.2f) m=(%.1f,%.1f,%.1f) µT",
                    ax, ay, az, gx, gy, gz, mx, my, mz);
            }
        }

        // Encoder parsing
        else if (json.contains("enc")) {
            auto& arr = json["enc"];
            if (arr.size() >= 4) {
                double left_rot  = arr[2].get<double>();
                double right_rot = arr[3].get<double>();

                static double x = 0.0, y = 0.0, theta = 0.0;
                static bool first_run = true;
                static double last_left_rot = 0.0, last_right_rot = 0.0;

                if (first_run) {
                    last_left_rot = left_rot;
                    last_right_rot = right_rot;
                    first_run = false;
                }

                double dl = left_rot - last_left_rot;
                double dr = right_rot - last_right_rot;
                last_left_rot = left_rot;
                last_right_rot = right_rot;

                const double WHEEL_BASE = 0.074; // meters

                double d_center = (dl + dr) / 2.0;
                double d_theta = (dr - dl) / WHEEL_BASE;

                theta += d_theta;
                x += d_center * cos(theta);
                y += d_center * sin(theta);

                auto msg = nav_msgs::msg::Odometry();
                msg.header.stamp = this->now();
                msg.header.frame_id = "odom";
                msg.child_frame_id = "base_link";

                msg.pose.pose.position.x = x;
                msg.pose.pose.position.y = y;
                msg.pose.pose.position.z = 0.0;

                tf2::Quaternion q;
                q.setRPY(0, 0, theta); // roll=0, pitch=0, yaw=theta
                msg.pose.pose.orientation = tf2::toMsg(q);

                // Zero velocity (optional)
                msg.twist.twist.linear.x = 0.0;
                msg.twist.twist.angular.z = 0.0;

                // Covariances
                msg.pose.covariance.fill(0.0);
                msg.twist.covariance.fill(-1);

                odom_raw_pub_->publish(msg);
                RCLCPP_DEBUG(this->get_logger(), "Odom: x=%.3f y=%.3f th=%.3f", x, y, theta);
            }
        }
        
        // === ToF parsing: complete it ===
        else if (json.contains("tof")) {
            auto& arr = json["tof"];
            if (arr.size() >= 3) {
                float right_cm = arr[0].get<float>();
                float left_cm  = arr[1].get<float>();
                float front_cm = arr[2].get<float>();

                auto publish_range = [this](float dist_cm, rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr pub, const std::string& frame) {
                    sensor_msgs::msg::Range msg;
                    msg.header.stamp = this->now();
                    msg.header.frame_id = frame;
                    msg.radiation_type = sensor_msgs::msg::Range::INFRARED;
                    msg.field_of_view = 0.1;   // rad
                    msg.min_range = 0.02f;     // 2 cm
                    msg.max_range = 2.0f;      // 2 m
                    msg.range = (dist_cm <= 0 || dist_cm > 400) ? 
                        std::numeric_limits<float>::quiet_NaN() : 
                        dist_cm / 100.0f;
                    pub->publish(msg);
                };

                publish_range(right_cm, tof_right_pub_, "tof_right_link");
                publish_range(left_cm,  tof_left_pub_,  "tof_left_link");
                publish_range(front_cm, tof_front_pub_, "tof_front_link");
            }
        }

    } catch (const nlohmann::json::parse_error& e) {
        RCLCPP_WARN(this->get_logger(), "JSON parse error at byte %zu: %s", e.byte, e.what());
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "Error processing line: %s", e.what());
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SerialBridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}