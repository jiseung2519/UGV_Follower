#include "target_detector/target_detector.h"

#include <std_msgs/Int32MultiArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

TargetDetector::TargetDetector(ros::NodeHandle &nh) : tf_listener_(tf_buffer_)
{
    // Get parameters from the parameter server
    std::string depth_cam_topic;
    nh.param<std::string>("depth_cam_topic", depth_cam_topic, "/iris/camera/depth/points");

    // Subscribers
    cloud_sub_ = nh.subscribe(depth_cam_topic, 1, &TargetDetector::pointCloudCallback, this);
    local_position_sub_ = nh.subscribe("/mavros/local_position/pose", 1, &TargetDetector::localPositionCallback, this);
    bbox_sub_ = nh.subscribe("/yolo/bounding_box_pixels", 1, &TargetDetector::bboxCallback, this);

    // Publishers
    target_centroid_map_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_centroid_map", 1);
    target_centroid_cam_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_centroid_cam", 1);

    // Initialize bbox flag
    bbox_received_ = false;

    ROS_INFO("TargetDetector node initialized.");
}

void TargetDetector::localPositionCallback(const geometry_msgs::PoseStampedConstPtr &msg)
{
    geometry_msgs::TransformStamped tf_msg;

    tf_msg.header.stamp = ros::Time::now();
    tf_msg.header.frame_id = "map";            // parent frame
    tf_msg.child_frame_id = "drone_base_link"; // child frame

    tf_msg.transform.translation.x = msg->pose.position.x;
    tf_msg.transform.translation.y = msg->pose.position.y;
    tf_msg.transform.translation.z = msg->pose.position.z;

    tf_msg.transform.rotation = msg->pose.orientation;

    broadcaster_.sendTransform(tf_msg);
}

void TargetDetector::bboxCallback(const std_msgs::Int32MultiArray::ConstPtr &msg)
{
    if (msg->data.size() >= 4)
    {
        xmin_ = msg->data[0];
        ymin_ = msg->data[1];
        xmax_ = msg->data[2];
        ymax_ = msg->data[3];
        bbox_received_ = true;
        ROS_INFO("Received bbox: xmin=%d, ymin=%d, xmax=%d, ymax=%d", xmin_, ymin_, xmax_, ymax_);
    }
    else
    {
        ROS_WARN("Received bbox with less than 4 values!");
        bbox_received_ = false;
    }
}

void TargetDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    if (!bbox_received_)
    {
        ROS_WARN_THROTTLE(5, "No YOLO bbox received yet, skipping point cloud processing.");
        return;
    }

    // Convert ROS PointCloud2 to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->points.empty())
    {
        ROS_WARN("No points from depth camera. Unable to estimate pose.");
        return;
    }

    // Camera intrinsics (adjust if needed or load from param)
    const float fx = 454.686f;
    const float fy = 454.686f;
    const float cx = 424.5f;
    const float cy = 240.5f;

    // Filter points inside YOLO bbox
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto &pt : cloud->points)
    {
        if (pt.z <= 0)
            continue;

        int u = static_cast<int>(fx * (pt.x / pt.z) + cx);
        int v = static_cast<int>(fy * (pt.y / pt.z) + cy);

        if (u >= xmin_ && u <= xmax_ && v >= ymin_ && v <= ymax_)
        {
            filtered_cloud->points.push_back(pt);
        }
    }

    if (filtered_cloud->points.empty())
    {
        ROS_WARN("No points found inside the YOLO bounding box.");
        return;
    }

    // Compute centroid of filtered points
    Eigen::Vector4f centroid_filtered;
    pcl::compute3DCentroid(*filtered_cloud, centroid_filtered);

    // Publish centroid in camera frame (for debugging)
    geometry_msgs::PoseStamped pose_cam_msg;
    pose_cam_msg.header = cloud_msg->header;
    pose_cam_msg.pose.position.x = centroid_filtered[0];
    pose_cam_msg.pose.position.y = centroid_filtered[1];
    pose_cam_msg.pose.position.z = centroid_filtered[2];
    pose_cam_msg.pose.orientation.w = 1.0;
    target_centroid_cam_pub_.publish(pose_cam_msg);

    // Transform centroid to map frame and publish
    try
    {
        geometry_msgs::PointStamped centroid_cam;
        centroid_cam.header = cloud_msg->header;
        centroid_cam.point.x = centroid_filtered[0];
        centroid_cam.point.y = centroid_filtered[1];
        centroid_cam.point.z = centroid_filtered[2];

        geometry_msgs::PointStamped centroid_map;
        tf_buffer_.transform(centroid_cam, centroid_map, "map");

        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header = centroid_map.header;
        pose_msg.pose.position = centroid_map.point;
        pose_msg.pose.orientation.w = 1.0;
        target_centroid_map_pub_.publish(pose_msg);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("Transform failed: %s", ex.what());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_detector");
    ros::NodeHandle nh;

    TargetDetector detector(nh);

    ros::spin();
    return 0;
}
