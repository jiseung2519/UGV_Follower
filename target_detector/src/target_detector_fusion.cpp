#include "target_detector/target_detector.h"

TargetDetector::TargetDetector(ros::NodeHandle &nh)
{
    cloud_sub_ = nh.subscribe("/camera/depth/points", 1, &TargetDetector::pointCloudCallback, this);
    bbox_sub_ = nh.subscribe("/yolo/bounding_box_pixels", 1, &TargetDetector::bboxCallback, this);
    target_centroid_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/target_pose", 1);

    ROS_INFO("TargetDetector (Fusion) node initialized.");
}

void TargetDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    latest_cloud_ = cloud_msg;
    computeAndPublishPose();
}

void TargetDetector::bboxCallback(const std_msgs::Int32MultiArrayConstPtr &bbox_msg)
{
    if (bbox_msg->data.size() == 4)
    {
        latest_bbox_ = bbox_msg->data;
    }
    else
    {
        latest_bbox_.clear();
    }
    computeAndPublishPose();
}

void TargetDetector::computeAndPublishPose()
{
    if (!latest_cloud_ || latest_bbox_.empty())
        return;

    // Convert cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*latest_cloud_, *cloud);

    int x1 = latest_bbox_[0];
    int y1 = latest_bbox_[1];
    int x2 = latest_bbox_[2];
    int y2 = latest_bbox_[3];

    std::vector<Eigen::Vector3f> points_in_bbox;

    // Check each point
    for (const auto &pt : cloud->points)
    {
        if (!pcl::isFinite(pt))
            continue;
        // 포인트클라우드에서 projection 좌표계 확인 필요.
        // 일반적으로 organized point cloud는 row=y, col=x로 접근 가능
        // 하지만 여기서는 단순히 bounding box ROI 기반 filtering 예시
        // 실제로는 pixel index 매핑 필요
    }

    // ---- 간단한 버전: organized cloud일 경우 ----
    if (cloud->isOrganized())
    {
        for (int v = y1; v <= y2; ++v)
        {
            for (int u = x1; u <= x2; ++u)
            {
                const auto &pt = cloud->at(u, v);
                if (!pcl::isFinite(pt))
                    continue;
                points_in_bbox.emplace_back(pt.x, pt.y, pt.z);
            }
        }
    }

    if (points_in_bbox.empty())
    {
        ROS_WARN("No valid 3D points in bounding box.");
        return;
    }

    // Compute centroid
    Eigen::Vector3f centroid(0, 0, 0);
    for (const auto &p : points_in_bbox)
        centroid += p;
    centroid /= points_in_bbox.size();

    // Publish pose
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = latest_cloud_->header;
    pose_msg.pose.position.x = centroid.x();
    pose_msg.pose.position.y = centroid.y();
    pose_msg.pose.position.z = centroid.z();

    pose_msg.pose.orientation.x = 0.0;
    pose_msg.pose.orientation.y = 0.0;
    pose_msg.pose.orientation.z = 0.0;
    pose_msg.pose.orientation.w = 1.0;

    pose_pub_.publish(pose_msg);

    ROS_INFO("Published fused target pose: [x: %.2f, y: %.2f, z: %.2f] (from %zu points)",
             centroid.x(), centroid.y(), centroid.z(), points_in_bbox.size());
}

// main
int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_detector_fusion");
    ros::NodeHandle nh;

    TargetDetector detector(nh);

    ros::spin();
    return 0;
}
