#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.srv import SetModeRequest, CommandBoolRequest # Import request messages
import time

class SimpleTakeoff:
    def __init__(self):
        rospy.init_node('simple_takeoff_node', anonymous=True)
        self.rate = rospy.Rate(20)  # 20 Hz
        
        # MAVROS state subscriber
        self.current_state = State()
        rospy.Subscriber("mavros/state", State, self.state_cb)
        
        # Publisher for setpoint position
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        
        # Service clients for arming and setting mode
        rospy.wait_for_service('mavros/cmd/arming')
        self.arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        rospy.wait_for_service('mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)
        
        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = 0
        self.target_pose.pose.position.y = 0
        self.target_pose.pose.position.z = 3
        
        rospy.loginfo("Simple Takeoff Node Initialized")

    def state_cb(self, msg):
        self.current_state = msg

    def run(self):
        # Wait for MAVROS connection
        rospy.loginfo("Waiting for MAVROS connection...")
        while not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("MAVROS connected!")
        
        # Publish initial setpoints before OFFBOARD mode
        rospy.loginfo("Publishing initial setpoints...")
        for i in range(100):
            self.local_pos_pub.publish(self.target_pose)
            self.rate.sleep()
        
        rospy.loginfo("Initial setpoints published.")
        
        # Create the service request messages explicitly
        offb_set_mode_req = SetModeRequest()
        offb_set_mode_req.custom_mode = 'OFFBOARD'
        
        arm_cmd_req = CommandBoolRequest()
        arm_cmd_req.value = True
        
        last_request = rospy.Time.now()
        
        while not rospy.is_shutdown():
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_request) > rospy.Duration(5.0):
                if self.set_mode_client.call(offb_set_mode_req).mode_sent:
                    rospy.loginfo("Offboard mode enabled")
                last_request = rospy.Time.now()
            
            elif not self.current_state.armed and (rospy.Time.now() - last_request) > rospy.Duration(5.0):
                if self.arming_client.call(arm_cmd_req).success:
                    rospy.loginfo("Vehicle armed")
                last_request = rospy.Time.now()
                
            # Publish setpoint to maintain OFFBOARD mode and altitude
            self.local_pos_pub.publish(self.target_pose)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        takeoff_node = SimpleTakeoff()
        takeoff_node.run()
    except rospy.ROSInterruptException:
        pass