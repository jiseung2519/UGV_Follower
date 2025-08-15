#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>

// 퍼블리셔 선언
ros::Publisher pose_pub;

void targetPointsCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // 포인트 클라우드 메시지를 PCL 포인트 클라우드로 변환
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 유효한 포인트가 없는 경우 처리
    if (cloud->points.empty()) {
        ROS_WARN("No points in /target_points. Unable to estimate pose.");
        return;
    }

    // 중심점 계산
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // PoseStamped 메시지 생성
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = cloud_msg->header; // 입력 메시지의 헤더 사용
    pose_msg.pose.position.x = centroid[0];
    pose_msg.pose.position.y = centroid[1];
    pose_msg.pose.position.z = centroid[2];

    // Orientation 설정 (고정 값)
    pose_msg.pose.orientation.x = 0.0; // 고정된 쿼터니언 값 설정
    pose_msg.pose.orientation.y = 0.0;
    pose_msg.pose.orientation.z = 0.0;
    pose_msg.pose.orientation.w = 1.0; // "회전 없음"

    // 퍼블리시
    pose_pub.publish(pose_msg);

    ROS_INFO("Published /target_pose: [x: %f, y: %f, z: %f, orientation: (0, 0, 0, 1)]",
             centroid[0], centroid[1], centroid[2]);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "target_pose_estimation_node");
    ros::NodeHandle nh;

    // /target_points 구독 및 /target_pose 퍼블리셔 설정
    ros::Subscriber target_points_sub = nh.subscribe("/filtered_depth/projected_points", 1, targetPointsCallback);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/target_pose", 1);

    ROS_INFO("Target Pose Estimation Node is running...");
    ros::spin();
    return 0;
}
