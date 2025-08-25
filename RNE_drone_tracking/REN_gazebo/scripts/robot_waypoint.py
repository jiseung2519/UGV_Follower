#!/usr/bin/env python3
import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

class SquareNavigator:
    def __init__(self):
        rospy.init_node("square_navigator", anonymous=True)

        # Publisher & Subscriber
        self.cmd_pub = rospy.Publisher("/twist_marker_server/cmd_vel", Twist, queue_size=10)
        # self.robot_gt_pub_map_ = rospy.Publisher("/robot/ground_truth", PoseStamped, queue_size=10)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)

        # Waypoints (사각형)
        self.waypoints = [(0, 0), (7, 0), (7, -5), (0, -5)]
        self.current_waypoint = 0

        # 상태 변수
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # 제어 파라미터
        self.dist_tolerance = 0.7  # waypoint 도착 허용 오차
        self.linear_speed = 0.7
        self.k_ang = 1.0           # yaw 보정 gain

    def odom_callback(self, msg):
        # 위치
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Orientation (quaternion → yaw)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # # publish ground truth position of robot in map frame
        # pose_msg = PoseStamped()
        # pose_msg.header.stamp = rospy.Time.now()
        # pose_msg.header.frame_id = "map"   # 명시적으로 map frame

        # pose_msg.pose = msg.pose.pose
        # pose_msg.pose.position.x += 3 #robot starts 3m ahead of drone
        
        # self.robot_gt_pub_map_.publish(pose_msg)

        self.navigate()

    def navigate(self):
        if self.current_waypoint >= len(self.waypoints):
            self.current_waypoint = 0  # 다시 처음으로 (순환)

        goal_x, goal_y = self.waypoints[self.current_waypoint]

        dx = goal_x - self.x
        dy = goal_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx)

        cmd = Twist()

        if distance > self.dist_tolerance:
            # 항상 전진 + yaw 오차 보정
            angle_error = angle_to_goal - self.yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # normalize

            cmd.linear.x = self.linear_speed
            cmd.angular.z = self.k_ang * angle_error
        else:
            rospy.loginfo(f"Reached waypoint {self.current_waypoint}: ({goal_x}, {goal_y})")
            self.current_waypoint += 1  # 다음 웨이포인트

        self.cmd_pub.publish(cmd)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        navigator = SquareNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
