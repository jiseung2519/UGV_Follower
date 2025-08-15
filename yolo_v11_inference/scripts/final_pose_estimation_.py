#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import pcl
import numpy as np
from sensor_msgs import point_cloud2

pose_pub = None

def target_points_callback(cloud_msg):
    # Convert ROS PointCloud2 message to PCL PointCloud
    cloud = pcl.PointCloud_PointXYZRGB()
    points = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
    cloud.from_list(points)

    if cloud.size == 0:
        rospy.logwarn("No points in /target_points. Unable to estimate pose.")
        return

    # Compute the centroid of the point cloud
    centroid = np.mean(np.array(cloud), axis=0)

    # Create and publish PoseStamped message
    pose_msg = PoseStamped()
    pose_msg.header = cloud_msg.header
    pose_msg.pose.position.x = centroid[0]
    pose_msg.pose.position.y = centroid[1]
    pose_msg.pose.position.z = centroid[2]

    # Fixed orientation
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0

    pose_pub.publish(pose_msg)

    rospy.loginfo("Published /target_pose: [x: %f, y: %f, z: %f, orientation: (0, 0, 0, 1)]",
                  centroid[0], centroid[1], centroid[2])

if __name__ == '__main__':
    rospy.init_node('target_pose_estimation_node', anonymous=True)

    rospy.Subscriber("/target_points", PointCloud2, target_points_callback)
    pose_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size=1)

    rospy.loginfo("Target Pose Estimation Node is running...")
    rospy.spin()
