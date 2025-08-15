#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import time

class TrackerController:
    def __init__(self):
        rospy.init_node('tracker_controller')

        rospy.Subscriber("/car_tracker/detection", Float32MultiArray, self.detection_callback)
        self.cmd_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", Twist, queue_size=1)

        self.last_detection_time = rospy.Time.now()
        self.detection_timeout = rospy.Duration(1.0)

        self.image_width = 640
        self.image_height = 480
        self.desired_area = 15000

        self.kp_x = 0.002
        self.kp_y = 0.002
        self.kp_z = 0.001
        self.k_yaw = 0.002

        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20 Hz

        self.current_detection = None

    def detection_callback(self, msg):
        self.current_detection = msg.data
        self.last_detection_time = rospy.Time.now()

    def control_loop(self, event):
        twist = Twist()
        now = rospy.Time.now()

        if self.current_detection and now - self.last_detection_time < self.detection_timeout:
            cx, cy, w, h, depth = self.current_detection
            err_x = cx - self.image_width / 2
            err_y = cy - self.image_height / 2
            err_depth = self.desired_distance - depth

            twist.linear.x = self.kp_x * err_depth    # forward/backward using depth
            twist.linear.y = -self.kp_y * err_x       # left/right
            twist.linear.z = 0                        # keep constant height
            twist.angular.z = -self.k_yaw * err_x     # yaw correction
        else:
            twist.angular.z = 0.3  # Search rotate

        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    TrackerController()
    rospy.spin()
