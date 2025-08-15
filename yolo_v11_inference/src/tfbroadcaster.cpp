#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <cmath>

std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

geometry_msgs::Quaternion normalizeQuaternion(const geometry_msgs::Quaternion& q) {
    double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    geometry_msgs::Quaternion normalized_q;
    normalized_q.x = q.x / norm;
    normalized_q.y = q.y / norm;
    normalized_q.z = q.z / norm;
    normalized_q.w = q.w / norm;
    return normalized_q;
}

void targetPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
    geometry_msgs::TransformStamped transform_msg;

    transform_msg.header.stamp = pose_msg->header.stamp;
    transform_msg.header.frame_id = "camera_depth_optical_frame";
    transform_msg.child_frame_id = "target_frame";

    transform_msg.transform.translation.x = pose_msg->pose.position.x;
    transform_msg.transform.translation.y = pose_msg->pose.position.y;
    transform_msg.transform.translation.z = pose_msg->pose.position.z;

    transform_msg.transform.rotation = normalizeQuaternion(pose_msg->pose.orientation);

    tf_broadcaster->sendTransform(transform_msg);

    ROS_INFO("Broadcasted TF from camera_link to target_frame: [x: %f, y: %f, z: %f]",
             pose_msg->pose.position.x, pose_msg->pose.position.y, pose_msg->pose.position.z);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "tf_broadcaster");
    ros::NodeHandle nh;

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>();

    ros::Subscriber pose_sub = nh.subscribe("/target_pose", 10, targetPoseCallback);

    ROS_INFO("Target Pose to TF Broadcaster Node is running...");
    ros::spin();

    return 0;
}