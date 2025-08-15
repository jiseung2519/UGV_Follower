#!/usr/bin/env python3
import rospy
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped

class MAVROSCommander:
    def __init__(self):
        rospy.init_node('mavros_commander')

        self.set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)

        rospy.sleep(2)
        self.send_initial_setpoints()
        self.arm_and_set_mode()

    def send_initial_setpoints(self):
        pose = PoseStamped()
        pose.pose.position.z = 2.0
        rate = rospy.Rate(20)
        for _ in range(100):  # Send before OFFBOARD to initialize
            self.setpoint_pub.publish(pose)
            rate.sleep()

    def arm_and_set_mode(self):
        self.set_mode(base_mode=0, custom_mode="OFFBOARD")
        rospy.sleep(1)
        self.arming(True)
        rospy.loginfo("Drone armed and set to OFFBOARD mode")

if __name__ == "__main__":
    MAVROSCommander()
    rospy.spin()
