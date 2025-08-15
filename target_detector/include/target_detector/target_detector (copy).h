#ifndef TARGET_DETECTOR_H
#define TARGET_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int32MultiArray.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

class TargetDetector
{
public:
    TargetDetector(ros::NodeHandle &nh);

private:
    // Subscribers
    ros::Subscriber cloud_sub_;
    ros::Subscriber local_position_sub_;
    ros::Subscriber bbox_sub_;
    ros::Subscriber camera_info_sub_;  // ✅ New for intrinsics

    // Publishers
    ros::Publisher target_centroid_cam_pub_;
    ros::Publisher target_centroid_map_pub_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster broadcaster_;

    // Bounding box
    int xmin_, ymin_, xmax_, ymax_;
    bool bbox_received_;

    // Camera intrinsics
    bool camera_info_received_;   // ✅ New flag
    double fx_, fy_, cx_, cy_;    // ✅ Intrinsic parameters

    // Drone pose (optional for distance calculation)
    geometry_msgs::PoseStamped current_pose_;

    // Callbacks
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    void localPositionCallback(const geometry_msgs::PoseStampedConstPtr &msg);
    void bboxCallback(const std_msgs::Int32MultiArray::ConstPtr &bbox_msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg); // ✅ New
};

#endif // TARGET_DETECTOR_H

