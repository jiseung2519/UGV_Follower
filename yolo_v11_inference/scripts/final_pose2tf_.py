#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import math

# Initialize the transform broadcaster
tf_broadcaster = None

def normalize_quaternion(q):
    norm = math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2)
    normalized_q = TransformStamped().transform.rotation
    normalized_q.x = q.x / norm
    normalized_q.y = q.y / norm
    normalized_q.z = q.z / norm
    normalized_q.w = q.w / norm
    return normalized_q

def target_pose_callback(pose_msg):
    transform_msg = TransformStamped()

    transform_msg.header.stamp = pose_msg.header.stamp
    transform_msg.header.frame_id = "camera_depth_optical_frame"
    transform_msg.child_frame_id = "target_frame"

    transform_msg.transform.translation.x = pose_msg.pose.position.x
    transform_msg.transform.translation.y = pose_msg.pose.position.y
    transform_msg.transform.translation.z = pose_msg.pose.position.z

    transform_msg.transform.rotation = normalize_quaternion(pose_msg.pose.orientation)

    tf_broadcaster.sendTransform(transform_msg)

    rospy.loginfo("Broadcasted TF from camera_link to target_frame: [x: %f, y: %f, z: %f]",
                  pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z)

if __name__ == '__main__':
    rospy.init_node('target_pose_to_tf_broadcaster', anonymous=True)

    tf_broadcaster = tf2_ros.TransformBroadcaster()

    rospy.Subscriber("/target_pose", PoseStamped, target_pose_callback)

    rospy.loginfo("Target Pose to TF Broadcaster Node is running...")
    rospy.spin()
