#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32MultiArray
import pcl
import pcl.pcl_visualization
import numpy as np
import pcl.pcl_visualization
from sensor_msgs import point_cloud2

# Initialize publisher and bounding box
pointcloud_pub = None
bounding_box_pixels = []

# Camera parameters
fx_rgb = 604.55224609375  # Focal length x
fy_rgb = 604.8668823242188  # Focal length y
cx_rgb = 328.6473388671875  # Principal point x
cy_rgb = 247.44219970703125  # Principal point y

# Translation offset (in meters)
translation_offset_x = 0.015  # 15mm

# Bounding box scaling factor
bounding_box_scale_factor = 0.5

def bounding_box_callback(msg):
    global bounding_box_pixels
    if len(msg.data) != 4:
        rospy.logwarn("Invalid bounding box received! Expected 4 elements, got %d", len(msg.data))
        return

    x1, y1, x2, y2 = msg.data
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    width = x2 - x1
    height = y2 - y1

    new_width = int(width * bounding_box_scale_factor)
    new_height = int(height * bounding_box_scale_factor)

    new_x1 = max(0, center_x - new_width // 2)
    new_y1 = max(0, center_y - new_height // 2)
    new_x2 = min(639, center_x + new_width // 2)
    new_y2 = min(479, center_y + new_height // 2)

    bounding_box_pixels = [new_x1, new_y1, new_x2, new_y2]

    rospy.loginfo("Bounding box updated: [%d, %d, %d, %d]", new_x1, new_y1, new_x2, new_y2)

def point_cloud_callback(cloud_msg):
    global bounding_box_pixels
    if len(bounding_box_pixels) != 4:
        rospy.logwarn("Bounding box not set. Skipping point cloud filtering.")
        return

    cloud = pcl.PointCloud_PointXYZRGB()
    points = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
    cloud.from_list(points)

    filtered_points = []

    for point in cloud:
        x, y, z, rgb = point
        depth_point = np.array([x + translation_offset_x, y, z])

        u = int((depth_point[0] * fx_rgb) / depth_point[2] + cx_rgb)
        v = int((depth_point[1] * fy_rgb) / depth_point[2] + cy_rgb)

        if (bounding_box_pixels[0] <= u <= bounding_box_pixels[2] and
            bounding_box_pixels[1] <= v <= bounding_box_pixels[3]):
            filtered_points.append([x, y, z, rgb])

    if not filtered_points:
        rospy.logwarn("No points found within the bounding box.")
        return

    filtered_cloud_msg = point_cloud2.create_cloud_xyz32(cloud_msg.header, filtered_points)
    pointcloud_pub.publish(filtered_cloud_msg)

    rospy.loginfo("Filtered %d points within bounding box.", len(filtered_points))

if __name__ == '__main__':
    rospy.init_node('target_pointcloud_filter_node', anonymous=True)

    rospy.Subscriber("/yolo/bounding_box_pixels", Int32MultiArray, bounding_box_callback)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, point_cloud_callback)

    pointcloud_pub = rospy.Publisher("/target_points", PointCloud2, queue_size=1)

    rospy.loginfo("Target PointCloud Filter Node is running...")
    rospy.spin()
