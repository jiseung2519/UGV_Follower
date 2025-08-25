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
#include <vector>

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
          desired_distance_(2.0), // Changed desired distance to 4.0m for more space
          desired_absolute_height_(2.0),
          takeoff_height_(2.0),
          max_linear_speed_(2),
          max_yaw_speed_(1),
          xy_gain_(0.5),
          z_gain_(0.5),
          yaw_gain_(3),
          camera_center_x_(424),
          is_detected_(false),
          is_at_height_(false),
          has_pose_(false),
          has_target_(false),
          offboard_initialized_(false),
          last_detection_time_(ros::Time(0)),
          last_request_(ros::Time::now())
    {
        ROS_INFO("DroneFollower node initialized with yaw_gain: %.4f", yaw_gain_);
    }

    void spin()
    {
        ros::Rate rate(50);  // 50Hz

        // Wait for FCU connection
        ROS_INFO("Waiting for FCU connection...");
        while(ros::ok() && !current_state_.connected){
            ros::spinOnce();
            rate.sleep();
        }
        ROS_INFO("FCU connected!");

        // Pre-stream setpoints
        ROS_INFO("Publishing initial setpoints...");
        geometry_msgs::TwistStamped hover_cmd = zeroVelocity();
        for (int i = 0; ros::ok() && i < 150; ++i) {
            hover_cmd.header.stamp = ros::Time::now();
            hover_cmd.header.frame_id = "base_link";
            cmd_pub_.publish(hover_cmd);
            ros::spinOnce();
            rate.sleep();
        }

        ROS_INFO("Starting main loop...");

        while (ros::ok())
        {
            ros::Time now = ros::Time::now();
            ros::spinOnce();

            publishSetpoints();
            manageFlightModes(now);
            handleTargetTimeout(now);

            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber target_sub_, detected_sub_, bbox_sub_, pose_sub_, state_sub_;
    ros::Publisher cmd_pub_;
    ros::ServiceClient arming_client_, set_mode_client_;

    mavros_msgs::State current_state_;
    geometry_msgs::Point target_pos_, drone_pos_;
    std::vector<int> bbox_;
    double drone_yaw_;

    bool has_target_, has_pose_, is_detected_, is_at_height_;
    bool offboard_initialized_;

    double desired_distance_, desired_absolute_height_, takeoff_height_;
    double max_linear_speed_, max_yaw_speed_;
    double xy_gain_, z_gain_, yaw_gain_;
    int camera_center_x_;

    ros::Time last_request_, last_detection_time_;
    geometry_msgs::TwistStamped last_cmd_;  // for smoothing

    void publishSetpoints()
    {
        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "base_link";

        if (current_state_.armed && current_state_.mode == "OFFBOARD")
        {
            offboard_initialized_ = true;

            if (!is_at_height_)
                cmd = getTakeoffCommand();
            else if (!is_detected_)
                cmd = getSearchCommand();
            else
                cmd = getFollowCommand();
        }
        else
        {
            cmd = zeroVelocity();
        }

        // Smooth linear velocity
        cmd.twist.linear.x = 0.8 * last_cmd_.twist.linear.x + 0.2 * cmd.twist.linear.x;
        cmd.twist.linear.y = 0.8 * last_cmd_.twist.linear.y + 0.2 * cmd.twist.linear.y;
        cmd.twist.linear.z = 0.8 * last_cmd_.twist.linear.z + 0.2 * cmd.twist.linear.z;
        last_cmd_ = cmd;

        cmd_pub_.publish(cmd);

        ROS_INFO_THROTTLE(2.0, "cmd_vel: [%.2f, %.2f, %.2f] [%.2f]",
                          cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z,
                          cmd.twist.angular.z);
    }

    void manageFlightModes(const ros::Time& now)
    {
        if (!offboard_initialized_ && current_state_.connected)
        {
            if ((now - last_request_) > ros::Duration(2.0))
            {
                if (current_state_.mode != "OFFBOARD") {
                    if (trySetOffboardMode()) ROS_INFO("OFFBOARD request sent");
                    else ROS_WARN("Failed OFFBOARD request");
                } else if (!current_state_.armed) {
                    if (tryArmVehicle()) ROS_INFO("Arm request sent");
                    else ROS_WARN("Failed arm request");
                } else if (current_state_.armed && current_state_.mode == "OFFBOARD") {
                    offboard_initialized_ = true;
                    ROS_INFO("OFFBOARD active and armed");
                }
                last_request_ = now;
            }
        }
    }

    bool trySetOffboardMode()
    {
        mavros_msgs::SetMode srv;
        srv.request.custom_mode = "OFFBOARD";
        return set_mode_client_.call(srv) && srv.response.mode_sent;
    }

    bool tryArmVehicle()
    {
        mavros_msgs::CommandBool srv;
        srv.request.value = true;
        return arming_client_.call(srv) && srv.response.success;
    }

    geometry_msgs::TwistStamped zeroVelocity()
    {
        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "base_link";
        return cmd;
    }

    geometry_msgs::TwistStamped getTakeoffCommand()
    {
        geometry_msgs::TwistStamped cmd = zeroVelocity();
        if (!has_pose_) return cmd;

        double dz = takeoff_height_ - drone_pos_.z;
        cmd.twist.linear.z = clamp(dz * z_gain_, -0.5, 0.5);

        if (drone_pos_.z >= takeoff_height_) {
            is_at_height_ = true;
            ROS_INFO("Reached takeoff height: %.2f", drone_pos_.z);
        }

        return cmd;
    }

    geometry_msgs::TwistStamped getSearchCommand()
    {
        geometry_msgs::TwistStamped cmd = zeroVelocity();
        if (!has_pose_) return cmd;

        double dz = desired_absolute_height_ - drone_pos_.z;
        cmd.twist.linear.z = clamp(dz * z_gain_, -0.4, 0.4);
        cmd.twist.angular.z = 0.15;
        return cmd;
    }

    // --- MODIFIED FUNCTION ---
    geometry_msgs::TwistStamped getFollowCommand()
    {
        geometry_msgs::TwistStamped cmd = zeroVelocity();
        if (!has_pose_ || !has_target_) return cmd;

        // Calculate the vector from drone to target in the map frame
        double dx_map = target_pos_.x - drone_pos_.x;
        double dy_map = target_pos_.y - drone_pos_.y;

        // Calculate the current horizontal distance
        double horizontal_distance = std::sqrt(dx_map*dx_map + dy_map*dy_map);
        
        // Calculate the error in horizontal distance
        // Positive error means the drone is too far away, negative means too close
        double distance_error = horizontal_distance - desired_distance_;

        // Convert distance error to a forward velocity command
        // The drone moves forward or backward to correct the distance
        cmd.twist.linear.x = clamp(distance_error * xy_gain_, -max_linear_speed_, max_linear_speed_);
        
        // Calculate the yaw error to face the target
        double desired_yaw = std::atan2(dy_map, dx_map);
        double yaw_error = angles::shortest_angular_distance(drone_yaw_, desired_yaw);
        
        // Convert yaw error to a rotational velocity command
        cmd.twist.angular.z = clamp(yaw_error * yaw_gain_, -max_yaw_speed_, max_yaw_speed_);

        // Maintain the desired absolute height
        cmd.twist.linear.z = clamp((desired_absolute_height_ - drone_pos_.z) * z_gain_, -0.5, 0.5);

        // Note: We don't need a sideways velocity (linear.y) because the yaw
        // controller will continuously turn the drone to face the target,
        // and the forward velocity will then correct the distance.
        
        return cmd;
    }

    void handleTargetTimeout(const ros::Time& now)
    {
        if (is_detected_ && (now - last_detection_time_).toSec() > 5.0)
        {
            ROS_WARN_THROTTLE(5.0, "Target lost, switching to search mode");
            is_detected_ = false;
            has_target_ = false;
        }
    }

    // --- Callbacks ---
    void stateCallback(const mavros_msgs::State::ConstPtr& msg) { current_state_ = *msg; }

    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        drone_pos_ = msg->pose.position;
        tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, 
                         msg->pose.orientation.z, msg->pose.orientation.w);
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
        if (is_detected_) last_detection_time_ = ros::Time::now();
    }

    void bboxCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
    {
        if (msg->data.size() == 4) bbox_ = msg->data;
        else bbox_.clear();
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_follower");
    ros::NodeHandle nh;

    ROS_INFO("Starting DroneFollower node...");
    DroneFollower follower(nh);
    follower.spin();

    return 0;
}
