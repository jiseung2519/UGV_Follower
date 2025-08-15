#include "target_detector/target_detector.h"

TargetDetector::TargetDetector(ros::NodeHandle &nh) : tf_listener_(tf_buffer_)
{
    // get parameters from the parameter server
    std::string depth_cam_topic;
    nh.param<std::string>("depth_cam_topic", depth_cam_topic, "/iris/camera/depth/points");

    cloud_sub_ = nh.subscribe(depth_cam_topic, 1, &TargetDetector::pointCloudCallback, this);
    local_position_sub_ = nh.subscribe("/mavros/local_position/pose", 1, &TargetDetector::localPositionCallback, this);
    target_centroid_map_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_centroid_map", 1);
    target_centroid_cam_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_centroid_cam", 1);

    ROS_INFO("TargetDetector_simple node initialized.");
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

void TargetDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    // transform the point cloud message to PCL format
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->points.empty())
    {
        ROS_WARN("No points from depth camera. Unable to estimate pose.");
        return;
    }

    // compute 3D centroid in depth camera frame
    Eigen::Vector4f centroid_depth_camera_frame;
    pcl::compute3DCentroid(*cloud, centroid_depth_camera_frame);

    geometry_msgs::PointStamped centroid_cam;
    centroid_cam.header = cloud_msg->header; // e.g., "camera_depth_optical_frame"
    centroid_cam.point.x = centroid_depth_camera_frame[0];
    centroid_cam.point.y = centroid_depth_camera_frame[1];
    centroid_cam.point.z = centroid_depth_camera_frame[2];

    // Publish centroid in camera frame (optional, for debugging)
    geometry_msgs::PoseStamped pose_cam_msg;
    pose_cam_msg.header = centroid_cam.header;
    pose_cam_msg.pose.position = centroid_cam.point;
    pose_cam_msg.pose.orientation.w = 1.0; // dummy orientation
    target_centroid_cam_pub_.publish(pose_cam_msg);

    try
    {
        // publish centroid in map frame
        geometry_msgs::PointStamped centroid_map;
        tf_buffer_.transform(centroid_cam, centroid_map, "map"); // transform to map frame

        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header = centroid_map.header;
        pose_msg.pose.position = centroid_map.point;
        pose_msg.pose.orientation.w = 1.0; // dummy orientation

        target_centroid_map_pub_.publish(pose_msg);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("Transform failed: %s", ex.what());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_detector_simple");
    ros::NodeHandle nh;

    TargetDetector detector(nh);

    ros::spin();
    return 0;
}
