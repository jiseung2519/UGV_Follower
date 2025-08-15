#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <algorithm>
#include <thread>
ros::Publisher pointcloud_pub;
std::vector<int> bounding_box_pixels;

const float fx_rgb = 604.55224609375; // Focal length x
const float fy_rgb = 604.8668823242188; // Focal length y
const float cx_rgb = 328.6473388671875; // Principal point x
const float cy_rgb = 247.44219970703125; // Principal point y

// color: 
// [913.2193603515625, 0.0, 638.2164916992188, 0.0, 911.3751220703125, 362.5431213378906, 0.0, 0.0, 1.0]
// height: 720 width: 1280
// depth: 
// [424.618896484375, 0.0, 419.52734375, 0.0, 424.618896484375, 239.46791076660156, 0.0, 0.0, 1.0]
// height: 480, width: 848

// const float fx_rgb = 913.2193603515625; // Focal length x
// const float fy_rgb = 911.3751220703125; // Focal length y
// const float cx_rgb = 638.2164916992188; // Principal point x
// const float cy_rgb = 362.5431213378906; // Principal point y

const float translation_offset_x = 0.015; // 15mm

const float bounding_box_scale_factor = 0.5;

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

    // new_x1 = std::max(0, new_x1);
    // new_y1 = std::max(0, new_y1);
    // new_x2 = std::min(639, new_x2);
    // new_y2 = std::min(479, new_y2);

    bounding_box_pixels = {new_x1, new_y1, new_x2, new_y2};

    ROS_INFO("Bounding box updated: [%d, %d, %d, %d]", new_x1, new_y1, new_x2, new_y2);
}


void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // Check if bounding box information is set. If not, skip processing.
    if (bounding_box_pixels.size() != 4) {
        ROS_WARN("Bounding box not set. Skipping point cloud filtering.");
        return;
    }

    // Convert the incoming ROS PointCloud2 message to a PCL PointCloud format.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Create a PCL PointCloud to store the filtered points.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Determine the number of threads available on the hardware.
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads > 3) {
        num_threads -= 2; // Subtract 2 if more than 2 threads are available.
        ROS_INFO("Detected %zu hardware threads. Using %zu threads after subtracting 2.", num_threads + 2, num_threads);
    } else {
        num_threads = 2; // Default to 2 threads if hardware_concurrency() <= 2.
        ROS_INFO("Detected %zu hardware threads. Using default 2 threads.", num_threads);
    }

    // Create threads and a vector to store results from each thread.
    std::vector<std::thread> threads;
    std::vector<std::vector<pcl::PointXYZRGB>> thread_results(num_threads);

    // Lambda function to process a range of points in the point cloud.
    auto process_range = [&](size_t start, size_t end, size_t thread_id) {
        for (size_t i = start; i < end; ++i) {
            const auto& point = cloud->points[i];
            if (!pcl::isFinite(point)) continue; // Skip invalid points.

            // Apply translation offset to the x-coordinate of the point.
            Eigen::Vector3f depth_point(point.x, point.y, point.z);
            depth_point[0] += translation_offset_x;

            // Map the transformed 3D point to 2D image coordinates.
            int u = static_cast<int>((depth_point[0] * fx_rgb) / depth_point[2] + cx_rgb);
            int v = static_cast<int>((depth_point[1] * fy_rgb) / depth_point[2] + cy_rgb);

            // Check if the 2D point lies within the bounding box.
            if (u >= bounding_box_pixels[0] && u <= bounding_box_pixels[2] &&
                v >= bounding_box_pixels[1] && v <= bounding_box_pixels[3]) {
                thread_results[thread_id].push_back(point); // Add the point to the thread's results.
            }
        }
    };

    // Divide the points among the threads for parallel processing.
    size_t points_per_thread = cloud->points.size() / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * points_per_thread;
        size_t end = (t == num_threads - 1) ? cloud->points.size() : start + points_per_thread;
        threads.emplace_back(process_range, start, end, t);
    }

    // Wait for all threads to complete their work.
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge the results from all threads into the filtered cloud.
    for (const auto& result : thread_results) {
        for (const auto& point : result) {
            filtered_cloud->points.push_back(point);
        }
    }

    // Check if the filtered cloud is empty and handle accordingly.
    if (filtered_cloud->points.empty()) {
        ROS_WARN("No points found within the bounding box.");
        return;
    }

    // Convert the filtered PCL PointCloud back to a ROS PointCloud2 message.
    sensor_msgs::PointCloud2 output_cloud;
    pcl::toROSMsg(*filtered_cloud, output_cloud);
    output_cloud.header = cloud_msg->header;

    // Publish the filtered PointCloud2 message.
    pointcloud_pub.publish(output_cloud);

    ROS_INFO("Filtered %zu points within bounding box.", filtered_cloud->points.size());
}

// void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
//     if (bounding_box_pixels.size() != 4) {
//         ROS_WARN("Bounding box not set. Skipping point cloud filtering.");
//         return;
//     }

//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
//     pcl::fromROSMsg(*cloud_msg, *cloud);

//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

//     size_t valid_points = 0;
//     for (const auto& point : cloud->points) {
//         if (!pcl::isFinite(point)) continue;

//         Eigen::Vector3f depth_point(point.x, point.y, point.z);
//         depth_point[0] += translation_offset_x;

//         int u = static_cast<int>((depth_point[0] * fx_rgb) / depth_point[2] + cx_rgb);
//         int v = static_cast<int>((depth_point[1] * fy_rgb) / depth_point[2] + cy_rgb);

//         if (u >= bounding_box_pixels[0] && u <= bounding_box_pixels[2] &&
//             v >= bounding_box_pixels[1] && v <= bounding_box_pixels[3]) {
//             filtered_cloud->points.push_back(point);
//             valid_points++;
//         }
//     }

//     if (filtered_cloud->points.empty()) {
//         ROS_WARN("No points found within the bounding box.");
//         return;
//     }

//     sensor_msgs::PointCloud2 output_cloud;
//     pcl::toROSMsg(*filtered_cloud, output_cloud);
//     output_cloud.header = cloud_msg->header;
//     pointcloud_pub.publish(output_cloud);

//     ROS_INFO("Filtered %zu points within bounding box.", valid_points);
// }

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_filter_node");
    ros::NodeHandle nh;

    ros::Subscriber bbox_sub = nh.subscribe("/yolo/bounding_box_pixels", 1, boundingBoxCallback);
    // ros::Subscriber cloud_sub = nh.subscribe("/camera/depth_registerd/points", 1, pointCloudCallback);
    ros::Subscriber cloud_sub = nh.subscribe("/camera/depth/color/points", 1, pointCloudCallback);
    // ros::Subscriber cloud_sub = nh.subscribe("/camera/aligned_depth_to_color/image_raw", 1, pointCloudCallback);

    pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/target_points", 1);

    ROS_INFO("Target PointCloud Filter Node is running...");
    ros::spin();
    return 0;
}
