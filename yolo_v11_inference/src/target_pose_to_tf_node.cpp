#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <cmath> // for sqrt

// TF Broadcaster 선언
std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

// 쿼터니언 유효성 검사 및 정규화 함수
geometry_msgs::Quaternion normalizeQuaternion(const geometry_msgs::Quaternion& q) {
    double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    geometry_msgs::Quaternion normalized_q;
    if (norm < 1e-6) {
        ROS_WARN("Quaternion norm is too small, resetting to identity quaternion.");
        normalized_q.x = 0.0;
        normalized_q.y = 0.0;
        normalized_q.z = 0.0;
        normalized_q.w = 1.0; // 기본값으로 설정 (항등 회전)
    } else {
        normalized_q.x = q.x / norm;
        normalized_q.y = q.y / norm;
        normalized_q.z = q.z / norm;
        normalized_q.w = q.w / norm;
    }
    return normalized_q;
}


// /target_pose 콜백 함수
void targetPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
    // TransformStamped 메시지 생성
    geometry_msgs::TransformStamped transform_msg;

    // 헤더 설정
    transform_msg.header.frame_id = "camera_depth_optical_frame";       // 부모 프레임
    transform_msg.child_frame_id = "target_frame";       // 자식 프레임

    // 타임스탬프 유효성 확인 및 설정
    if (pose_msg->header.stamp.isZero()) {
        ROS_WARN("Received PoseStamped message with invalid timestamp. Using current time.");
        transform_msg.header.stamp = ros::Time::now();
    } else {
        transform_msg.header.stamp = pose_msg->header.stamp;
    }

    // 위치 설정
    transform_msg.transform.translation.x = pose_msg->pose.position.x;
    transform_msg.transform.translation.y = pose_msg->pose.position.y;
    transform_msg.transform.translation.z = pose_msg->pose.position.z;

    // 정규화된 쿼터니언으로 회전 설정
    transform_msg.transform.rotation = normalizeQuaternion(pose_msg->pose.orientation);

    // TF 브로드캐스트
    tf_broadcaster->sendTransform(transform_msg);

    ROS_INFO("Broadcasted TF from camera_link to target_frame: [x: %f, y: %f, z: %f]",
             pose_msg->pose.position.x, pose_msg->pose.position.y, pose_msg->pose.position.z);
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "target_pose_to_tf_broadcaster");
    ros::NodeHandle nh;

    // TF Broadcaster 초기화
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>();

    // /target_pose 구독
    ros::Subscriber pose_sub = nh.subscribe("/target_pose", 10, targetPoseCallback);

    ROS_INFO("Target Pose to TF Broadcaster Node is running...");
    ros::spin();

    return 0;
}
