#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from tf.transformations import euler_from_quaternion

current_pose = PoseStamped()
target_pose = PoseStamped()

yaw_offset = None  # To store initial yaw offset

def pose_callback(msg):
    global current_pose
    current_pose = msg

def waypoint_callback(msg):
    global target_pose
    target_pose = msg

def get_yaw_from_orientation(orientation):
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    _, _, yaw = euler_from_quaternion(quat)
    return yaw

def yaw_error(target_yaw, current_yaw):
    # Normalize to [-pi, pi]
    error = target_yaw - current_yaw
    return math.atan2(math.sin(error), math.cos(error))

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

if __name__ == "__main__":
    rospy.init_node("offb_node_py")

    rospy.Subscriber("/orb_slam3/camera_pose", PoseStamped, callback=pose_callback)
    rospy.Subscriber("/lookahead_waypoint", PoseStamped, callback=waypoint_callback)

    local_vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
    rate = rospy.Rate(20)

    cmd_velocity = TwistStamped()

    # Warm-up: send some zero commands
    for _ in range(100):
        if rospy.is_shutdown():
            break
        cmd_velocity.header.stamp = rospy.Time.now()
        cmd_velocity.header.frame_id = "base_link"
        cmd_velocity.twist.linear.x = 0
        cmd_velocity.twist.linear.y = 0
        cmd_velocity.twist.linear.z = 0
        cmd_velocity.twist.angular.z = 0
        local_vel_pub.publish(cmd_velocity)
        rate.sleep()

    K_P = 0.3
    K_YAW = 1.0

    MAX_VEL_XY = 2.0   # m/s
    MAX_VEL_Z  = 0.5   # m/s
    MAX_YAW_RATE = 0.7 # rad/s

    while not rospy.is_shutdown():
        # Compute initial yaw offset ONCE
        if yaw_offset is None:
            yaw_offset = 0.0 - get_yaw_from_orientation(current_pose.pose.orientation)

        # --- position error ---
        error_x = target_pose.pose.position.x - current_pose.pose.position.x
        error_y = target_pose.pose.position.y - current_pose.pose.position.y
        error_z = target_pose.pose.position.z - current_pose.pose.position.z

        vx = K_P * error_x
        vy = K_P * error_y
        vz = K_P * error_z

        # --- clamp velocity ---
        cmd_velocity.twist.linear.x = clamp(vx, -MAX_VEL_XY, MAX_VEL_XY)
        cmd_velocity.twist.linear.y = clamp(vy, -MAX_VEL_XY, MAX_VEL_XY)
        cmd_velocity.twist.linear.z = clamp(vz, -MAX_VEL_Z, MAX_VEL_Z)

        # --- yaw control ---
        current_yaw = get_yaw_from_orientation(current_pose.pose.orientation) + yaw_offset

        # Target point to face (fixed at 5,0)
        tree_x = 5.0
        tree_y = 0.0
        dx = tree_x - current_pose.pose.position.x
        dy = tree_y - current_pose.pose.position.y
        target_yaw = math.atan2(dy, dx)

        error_yaw = yaw_error(target_yaw, current_yaw)
        wz = K_YAW * error_yaw

        cmd_velocity.twist.angular.z = clamp(wz, -MAX_YAW_RATE, MAX_YAW_RATE)

        # Publish with header
        cmd_velocity.header.stamp = rospy.Time.now()
        cmd_velocity.header.frame_id = "base_link"
        local_vel_pub.publish(cmd_velocity)

        rate.sleep()
