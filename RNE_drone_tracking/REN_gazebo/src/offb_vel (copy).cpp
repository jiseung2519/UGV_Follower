#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/Bool.h> 
#include <std_msgs/Int32MultiArray.h> 
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>
#include <angles/angles.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <algorithm>

template <typename T>
T clamp(const T val, const T min_val, const T max_val)
{
    return std::min(std::max(val, min_val), max_val);
}

class DroneFollower
{
public:
    explicit DroneFollower(ros::NodeHandle& nh)
    : nh_(nh),
      target_sub_(nh_.subscribe("/target_centroid_map", 1, &DroneFollower::targetCallback, this)),
      detected_sub_(nh_.subscribe("/yolo/detected", 1, &DroneFollower::detectedCallback, this)),
      bbox_sub_(nh_.subscribe("/yolo/bounding_box_pixels", 1, &DroneFollower::bboxCallback, this)),
      pose_sub_(nh_.subscribe("/mavros/local_position/pose", 1, &DroneFollower::poseCallback, this)),
      state_sub_(nh_.subscribe("/mavros/state", 10, &DroneFollower::stateCallback, this)),
      cmd_pub_(nh_.advertise<geometry_msgs::TwistStamped>("/mavros/setpoint_velocity/cmd_vel", 10)),
      arming_client_(nh_.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming")),
      set_mode_client_(nh_.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode")),
      desired_distance_(3.0),
      desired_absolute_height_(3.0),
      takeoff_height_(3.0),
      max_linear_speed_(1.0),
      max_yaw_speed_(0.175),
      camera_center_x_(424),   // default value
      yaw_gain_(0.00001),
      is_detected_(false),
      is_at_height_(false),
      has_pose_(false),
      has_target_(false),
      initial_drone_z_at_takeoff_(0.0),
      last_detection_time_(ros::Time(0)),
      last_request_(ros::Time::now())
    {
        nh_.param("camera_center_x", camera_center_x_, camera_center_x_);
        nh_.param("yaw_gain", yaw_gain_, yaw_gain_);

        ROS_INFO("DroneFollower node initialized");
    }

    void spin()
    {
        ros::Rate rate(20);
        publishInitialSetpoints(rate);

        while (ros::ok())
        {
            ros::Time now = ros::Time::now();
            ros::spinOnce();

            manageFlightModes(now);

            if (current_state_.armed && current_state_.mode == "OFFBOARD")
            {
                if (!is_at_height_)
                {
                    ascendToTakeoffHeight();
                }
                else if (!is_detected_)
                {
                    searchForTarget();
                }
                else
                {
                    computeAndPublishControl();
                    rate.sleep();
                    continue;
                }
            }
            else
            {
                publishZeroVelocity();
            }

            handleTargetTimeout(now);

            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;

    ros::Subscriber target_sub_;
    ros::Subscriber detected_sub_;
    ros::Subscriber bbox_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber state_sub_;

    ros::Publisher cmd_pub_;

    ros::ServiceClient arming_client_;
    ros::ServiceClient set_mode_client_;

    mavros_msgs::State current_state_;

    geometry_msgs::Point target_pos_;
    geometry_msgs::Point drone_pos_;
    std::vector<int> bbox_;
    double drone_yaw_;

    bool has_target_;
    bool has_pose_;
    bool is_detected_;
    bool is_at_height_;

    double desired_distance_;
    double desired_absolute_height_;
    double takeoff_height_;
    double max_linear_speed_;
    double max_yaw_speed_;

    int camera_center_x_;
    double yaw_gain_;

    ros::Time last_request_;
    ros::Time last_detection_time_;

    geometry_msgs::TwistStamped last_cmd_;
    double initial_drone_z_at_takeoff_;

    void publishInitialSetpoints(ros::Rate& rate)
    {
        geometry_msgs::TwistStamped zero_twist;
        zero_twist.twist.linear.x = 0.0;
        zero_twist.twist.linear.y = 0.0;
        zero_twist.twist.linear.z = 0.0;
        zero_twist.twist.angular.x = 0.0;
        zero_twist.twist.angular.y = 0.0;
        zero_twist.twist.angular.z = 0.0;

        for (int i = 100; ros::ok() && i > 0; --i)
        {
            zero_twist.header.stamp = ros::Time::now();
            cmd_pub_.publish(zero_twist);
            ROS_INFO_THROTTLE(1.0, "Publishing initial setpoint (%d remaining)", i);
            ros::spinOnce();
            rate.sleep();
        }
    }

    void manageFlightModes(const ros::Time& now)
    {
        if (current_state_.mode != "OFFBOARD")
        {
            if ((now - last_request_) > ros::Duration(1.0))
            {
                if (trySetOffboardMode())
                    ROS_INFO("Offboard mode enabled");
                else
                    ROS_WARN("Failed to set OFFBOARD mode");

                last_request_ = now;
            }
        }
        else if (!current_state_.armed)
        {
            if ((now - last_request_) > ros::Duration(1.0))
            {
                if (tryArmVehicle())
                    ROS_INFO("Vehicle armed");
                else
                    ROS_WARN("Failed to arm vehicle");

                last_request_ = now;
            }
        }
    }

    bool trySetOffboardMode()
    {
        mavros_msgs::SetMode set_mode_srv;
        set_mode_srv.request.custom_mode = "OFFBOARD";
        return set_mode_client_.call(set_mode_srv) && set_mode_srv.response.mode_sent;
    }

    bool tryArmVehicle()
    {
        mavros_msgs::CommandBool arm_srv;
        arm_srv.request.value = true;
        return arming_client_.call(arm_srv) && arm_srv.response.success;
    }

    void ascendToTakeoffHeight()
    {
        if (!has_pose_)
        {
            ROS_WARN_THROTTLE(2.0, "Waiting for drone pose...");
            publishZeroVelocity();
            return;
        }

        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.twist.linear.z = 0.5;  // Climb rate

        if (drone_pos_.z >= takeoff_height_)
        {
            is_at_height_ = true;
            initial_drone_z_at_takeoff_ = drone_pos_.z;
            ROS_INFO("Reached takeoff height: %.2f", initial_drone_z_at_takeoff_);
        }

        cmd_pub_.publish(cmd);
    }

    void searchForTarget()
    {
        if (!has_pose_)
        {
            ROS_WARN_THROTTLE(2.0, "Waiting for drone pose...");
            publishZeroVelocity();
            return;
        }

        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();

        double altitude_error = desired_absolute_height_ - drone_pos_.z;
        cmd.twist.linear.z = clamp(altitude_error * 0.5, -0.5, 0.5);
        cmd.twist.angular.z = max_yaw_speed_; // rotate searching

        ROS_INFO_THROTTLE(2.0, "Searching for target...");

        cmd_pub_.publish(cmd);
    }

    void computeAndPublishControl()
    {
        if (!has_pose_ || !has_target_ || bbox_.size() != 4)
            return;

        // Position control: maintain desired horizontal distance
        double dx = target_pos_.x - drone_pos_.x;
        double dy = target_pos_.y - drone_pos_.y;
        double distance_xy = std::hypot(dx, dy);

        double factor = 1.0 - (desired_distance_ / std::max(distance_xy, 0.001));
        double desired_dx = dx * factor;
        double desired_dy = dy * factor;

        double vx = clamp(desired_dx, -max_linear_speed_, max_linear_speed_);
        double vy = clamp(desired_dy, -max_linear_speed_, max_linear_speed_);

        // Altitude control
        double altitude_error = desired_absolute_height_ - drone_pos_.z;
        double vz = clamp(altitude_error * 0.5, -0.5, 0.5);

        // Yaw control based on bounding box center offset
        int bbox_x_min = bbox_[0];
        int bbox_x_max = bbox_[2];
        double bbox_center_x = (bbox_x_min + bbox_x_max) / 2.0;

        double yaw_error_pixels = camera_center_x_ - bbox_center_x;
        // Positive yaw_rate rotates CCW, negative rotates CW
        double yaw_rate = clamp(-yaw_error_pixels * yaw_gain_, -max_yaw_speed_, max_yaw_speed_);

        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.twist.linear.x = vx;
        cmd.twist.linear.y = vy;
        cmd.twist.linear.z = vz;
        cmd.twist.angular.z = yaw_rate;

        last_cmd_ = cmd;
        cmd_pub_.publish(cmd);
    }

    void publishZeroVelocity()
    {
        geometry_msgs::TwistStamped zero_twist;
        zero_twist.header.stamp = ros::Time::now();
        zero_twist.twist.linear.x = 0.0;
        zero_twist.twist.linear.y = 0.0;
        zero_twist.twist.linear.z = 0.0;
        zero_twist.twist.angular.x = 0.0;
        zero_twist.twist.angular.y = 0.0;
        zero_twist.twist.angular.z = 0.0;
        cmd_pub_.publish(zero_twist);
    }

    void handleTargetTimeout(const ros::Time& now)
    {
        if (is_detected_ && (now - last_detection_time_).toSec() > 5.0)
        {
            ROS_WARN_THROTTLE(5.0, "Target lost for over 5 seconds, switching to search mode.");
            is_detected_ = false;
            has_target_ = false;
            bbox_.clear();
        }
    }

    // Callbacks

    void stateCallback(const mavros_msgs::State::ConstPtr& msg)
    {
        current_state_ = *msg;
    }

    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        drone_pos_ = msg->pose.position;

        tf2::Quaternion q(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        drone_yaw_ = yaw;

        has_pose_ = true;
    }

    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        target_pos_ = msg->pose.position;
        has_target_ = true;
    }

    void detectedCallback(const std_msgs::Bool::ConstPtr& msg)
    {
        is_detected_ = msg->data;
        if (is_detected_)
            last_detection_time_ = ros::Time::now();
    }

    void bboxCallback(const std_msgs::Int32MultiArray::ConstPtr& bbox_msg)
    {
        if (bbox_msg->data.size() == 4)
        {
            int x_min = bbox_msg->data[0];
            int y_min = bbox_msg->data[1];
            int x_max = bbox_msg->data[2];
            int y_max = bbox_msg->data[3];

            if (x_min >= x_max || y_min >= y_max)
            {
                ROS_WARN_THROTTLE(2.0, "Received malformed bbox coordinates");
                bbox_.clear();
                return;
            }

            bbox_ = bbox_msg->data;
            ROS_INFO_STREAM_THROTTLE(1.0, "Updated bbox: [" << x_min << ", " << y_min << ", " << x_max << ", " << y_max << "]");
        }
        else
        {
            bbox_.clear();
            ROS_WARN_THROTTLE(2.0, "Received invalid bounding box message (size != 4)");
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_follower");
    ros::NodeHandle nh;

    DroneFollower follower(nh);
    follower.spin();

    return 0;
}
