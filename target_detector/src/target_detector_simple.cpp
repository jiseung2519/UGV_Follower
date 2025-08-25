#include "target_detector/target_detector.h"

TargetDetector::TargetDetector(ros::NodeHandle &nh) : tf_listener_(tf_buffer_)
{
    cloud_sub_ = nh.subscribe("/iris/camera/depth/points", 1, &TargetDetector::pointCloudCallback, this);
    local_position_sub_ = nh.subscribe("/mavros/local_position/pose", 1, &TargetDetector::localPositionCallback, this);
    bbox_sub_ = nh.subscribe("/yolo/bounding_box_pixels", 1, &TargetDetector::bboxCallback, this);

    target_centroid_cam_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_centroid_cam", 1);
    target_centroid_map_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_centroid_map", 1);

    bbox_received_ = false;
    xmin_ = ymin_ = xmax_ = ymax_ = 0;

    ROS_INFO("TargetDetector initialized.");
}

void TargetDetector::bboxCallback(const std_msgs::Int32MultiArray::ConstPtr &msg)
{
    if (msg->data.empty())
    {
        // No detections, just mark no bbox received
        bbox_received_ = false;
        boxes_.clear();
        return;
    }

    if (msg->data.size() % 5 != 0)
    {
        ROS_WARN_THROTTLE(5.0, "Invalid bbox array size (%zu), ignoring!", msg->data.size());
        return;
    }

    boxes_.clear();
    for (size_t i = 0; i + 4 < msg->data.size(); i += 5)
    {
        BBox box;
        box.xmin = msg->data[i];
        box.ymin = msg->data[i + 1];
        box.xmax = msg->data[i + 2];
        box.ymax = msg->data[i + 3];
        box.confidence = msg->data[i + 4] / 1000.0f;
        boxes_.push_back(box);
    }
    bbox_received_ = !boxes_.empty();
}


BBox TargetDetector::selectBestBox()
{
    BBox best = boxes_[0];
    for (const auto &b : boxes_)
    {
        if (b.confidence > best.confidence)
            best = b;
    }
    return best;
}

void TargetDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    if (!bbox_received_ || boxes_.empty())
        return;

    BBox box = selectBestBox();
    xmin_ = box.xmin;
    ymin_ = box.ymin;
    xmax_ = box.xmax;
    ymax_ = box.ymax;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->points.empty()) return;

    // Camera intrinsics
    const float fx = 910.712f;   // rounded from 910.7119750976562
    const float fy = 910.912f;   // rounded from 910.9118041992188
    const float cx = 652.760f;   // rounded from 652.760498046875
    const float cy = 373.047f;   // rounded from 373.0473327636719


    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto &pt : cloud->points)
    {
        if (pt.z <= 0) continue;
        int u = static_cast<int>(fx * (pt.x / pt.z) + cx);
        int v = static_cast<int>(fy * (pt.y / pt.z) + cy);
        if (u >= xmin_ && u <= xmax_ && v >= ymin_ && v <= ymax_)
            filtered_cloud->points.push_back(pt);
    }

    if (filtered_cloud->points.empty()) return;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*filtered_cloud, centroid);

    geometry_msgs::PoseStamped pose_cam;
    pose_cam.header = cloud_msg->header;
    pose_cam.pose.position.x = centroid[0];
    pose_cam.pose.position.y = centroid[1];
    pose_cam.pose.position.z = centroid[2];
    pose_cam.pose.orientation.w = 1.0;
    target_centroid_cam_pub_.publish(pose_cam);

    // Transform to map
    try
    {
        geometry_msgs::PointStamped centroid_cam, centroid_map;
        centroid_cam.header = cloud_msg->header;
        centroid_cam.point.x = centroid[0];
        centroid_cam.point.y = centroid[1];
        centroid_cam.point.z = centroid[2];

        tf_buffer_.transform(centroid_cam, centroid_map, "map");

        geometry_msgs::PoseStamped pose_map;
        pose_map.header = centroid_map.header;
        pose_map.pose.position = centroid_map.point;
        pose_map.pose.orientation.w = 1.0;
        target_centroid_map_pub_.publish(pose_map);
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("TF transform failed: %s", ex.what());
    }
}

void TargetDetector::localPositionCallback(const geometry_msgs::PoseStampedConstPtr &msg)
{
    // Only publish if the timestamp is new
    if (msg->header.stamp == last_tf_stamp_)
        return;

    last_tf_stamp_ = msg->header.stamp;

    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = "map";            // parent frame
    tf_msg.child_frame_id = "drone_base_link"; // child frame

    tf_msg.transform.translation.x = msg->pose.position.x;
    tf_msg.transform.translation.y = msg->pose.position.y;
    tf_msg.transform.translation.z = msg->pose.position.z;
    tf_msg.transform.rotation = msg->pose.orientation;

    broadcaster_.sendTransform(tf_msg);
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_detector");
    ros::NodeHandle nh;
    TargetDetector detector(nh);
    ros::spin();
    return 0;
}
