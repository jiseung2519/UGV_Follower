#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <algorithm>

// 전역 변수
ros::Publisher pointcloud_pub;
std::vector<int> bounding_box_pixels;

// RGB 카메라 Intrinsic 파라미터
const float fx_rgb = 604.55224609375; // Focal length x
const float fy_rgb = 604.8668823242188; // Focal length y
const float cx_rgb = 328.6473388671875; // Principal point x
const float cy_rgb = 247.44219970703125; // Principal point y

// Translation 오프셋 (15mm)
const float translation_offset_x = 0.015; // 15mm (x 방향)

// 바운딩 박스 축소 또는 확장을 위한 비율 설정
const float bounding_box_scale_factor = 0.5;

// 바운딩 박스 데이터 수신 콜백
void boundingBoxCallback(const std_msgs::Int32MultiArray::ConstPtr& msg) {
    if (msg->data.size() != 4) {
        ROS_WARN("Invalid bounding box received! Expected 4 elements, got %zu", msg->data.size());
        return;
    }

    int x1 = msg->data[0];
    int y1 = msg->data[1];
    int x2 = msg->data[2];
    int y2 = msg->data[3];

    int center_x = (x1 + x2) / 2;
    int center_y = (y1 + y2) / 2;

    int width = x2 - x1;
    int height = y2 - y1;

    int new_width = static_cast<int>(width * bounding_box_scale_factor);
    int new_height = static_cast<int>(height * bounding_box_scale_factor);

    int new_x1 = center_x - new_width / 2;
    int new_y1 = center_y - new_height / 2;
    int new_x2 = center_x + new_width / 2;
    int new_y2 = center_y + new_height / 2;

    new_x1 = std::max(0, new_x1);
    new_y1 = std::max(0, new_y1);
    new_x2 = std::min(639, new_x2);
    new_y2 = std::min(479, new_y2);

    bounding_box_pixels = {new_x1, new_y1, new_x2, new_y2};

    ROS_INFO("Bounding box updated: [%d, %d, %d, %d]", new_x1, new_y1, new_x2, new_y2);
}

// 포인트 클라우드 데이터 수신 콜백
void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if (bounding_box_pixels.size() != 4) {
        ROS_WARN("Bounding box not set. Skipping point cloud filtering.");
        return;
    }

    // 포인트 클라우드 메시지를 PCL 포인트 클라우드로 변환
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 필터링된 포인트를 저장할 클라우드 생성
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    size_t valid_points = 0;
    for (const auto& point : cloud->points) {
        if (!pcl::isFinite(point)) continue;

        // Translation 변환 적용 (x 방향으로 15mm 이동)
        Eigen::Vector3f depth_point(point.x, point.y, point.z);
        depth_point[0] += translation_offset_x;

        // 변환된 점을 RGB 프레임 픽셀로 매핑
        int u = static_cast<int>((depth_point[0] * fx_rgb) / depth_point[2] + cx_rgb);
        int v = static_cast<int>((depth_point[1] * fy_rgb) / depth_point[2] + cy_rgb);

        if (u >= bounding_box_pixels[0] && u <= bounding_box_pixels[2] &&
            v >= bounding_box_pixels[1] && v <= bounding_box_pixels[3]) {
            filtered_cloud->points.push_back(point);
            valid_points++;
        }
    }

    if (filtered_cloud->points.empty()) {
        ROS_WARN("No points found within the bounding box.");
        return;
    }

    sensor_msgs::PointCloud2 output_cloud;
    pcl::toROSMsg(*filtered_cloud, output_cloud);
    output_cloud.header = cloud_msg->header;
    pointcloud_pub.publish(output_cloud);

    ROS_INFO("Filtered %zu points within bounding box.", valid_points);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "target_pointcloud_filter_node");
    ros::NodeHandle nh;

    ros::Subscriber bbox_sub = nh.subscribe("/yolo_world/bounding_box_pixels", 1, boundingBoxCallback);
    ros::Subscriber cloud_sub = nh.subscribe("/camera/depth/color/points", 1, pointCloudCallback);

    pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/target_points", 1);

    ROS_INFO("Target PointCloud Filter Node is running...");
    ros::spin();
    return 0;
}
