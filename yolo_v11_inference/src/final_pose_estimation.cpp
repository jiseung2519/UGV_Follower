#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>

ros::Publisher pose_pub;

void targetPointsCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->points.empty()) {
        ROS_WARN("No points in /target_points. Unable to estimate pose.");
        return;
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = cloud_msg->header;
    pose_msg.pose.position.x = centroid[0];
    pose_msg.pose.position.y = centroid[1];
    pose_msg.pose.position.z = centroid[2];

    pose_msg.pose.orientation.x = 0.0;
    pose_msg.pose.orientation.y = 0.0;
    pose_msg.pose.orientation.z = 0.0;
    pose_msg.pose.orientation.w = 1.0;

    pose_pub.publish(pose_msg);

    ROS_INFO("Published /target_pose: [x: %f, y: %f, z: %f, orientation: (0, 0, 0, 1)]",
             centroid[0], centroid[1], centroid[2]);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pose_estimation_node");
    ros::NodeHandle nh;

    ros::Subscriber target_points_sub = nh.subscribe("/target_points", 1, targetPointsCallback);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/target_pose", 1);

    ROS_INFO("Target Pose Estimation Node is running...");
    ros::spin();
    return 0;
}