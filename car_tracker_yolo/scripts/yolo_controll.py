#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from car_tracker_yolo.msg import Detection2DArray
from sensor_msgs.msg import Image # For depth image
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations # For quaternion to Euler conversion

class DroneFollower:
    def __init__(self):
        rospy.init_node('drone_follower_node', anonymous=True)

        self.bridge = CvBridge()

        # --- Subscribers ---
        # Subscribe to YOLO detections (using the new topic name)
        self.yolo_sub = rospy.Subscriber('/yolov7/yolov7_detections', Detection2DArray, self.yolo_detections_callback, queue_size=1)
        # Subscribe to drone's current local position
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback, queue_size=1)
        # Subscribe to aligned depth image (CRITICAL for 3D position estimation)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback, queue_size=1)


        # --- Publishers ---
        # Publish velocity commands to MAVROS
        self.velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)

        # --- State Variables ---
        self.current_pose = None  # Stores the latest drone pose
        self.latest_detections = None # Stores the latest YOLO detections
        self.latest_depth_image = None # Stores the latest depth image (OpenCV format)

        # --- Control Parameters (PID gains - starting with P) ---
        # These will need extensive tuning!
        self.kp_x = 0.005  # Proportional gain for x-velocity (left/right) based on pixel error
        self.kp_y = 0.005  # Proportional gain for y-velocity (up/down) based on pixel error
        self.kp_z = 0.5    # Proportional gain for z-velocity (forward/backward) based on distance error
        self.kp_yaw = 0.005 # Proportional gain for yaw rate based on pixel error

        self.max_velocity_x = 0.5 # m/s
        self.max_velocity_y = 0.5 # m/s
        self.max_velocity_z = 0.5 # m/s
        self.max_yaw_rate = 0.3 # rad/s

        # --- Target Configuration ---
        self.target_class_ids = rospy.get_param('~target_class_ids', [0]) # Make sure this matches your model's 'car' or 'drone' ID
        self.target_distance_z_meters = 2.0 # Desired distance to the target in meters (forward from drone)
        self.center_pixel_x = 320 # Assuming 640x480 resolution (center x)
        self.center_pixel_y = 240 # Assuming 640x480 resolution (center y)
        
        # Camera intrinsics (placeholder - should get from /camera/color/camera_info topic)
        # These are usually fx, fy, cx, cy
        self.camera_fx = 614.0  # Example focal length X
        self.camera_fy = 614.0  # Example focal length Y
        self.camera_cx = 320.0  # Example principal point X
        self.camera_cy = 240.0  # Example principal point Y


        # --- Control Loop Timer ---
        self.control_rate = rospy.Rate(30) # Hz, how often to send commands
        self.control_timer = rospy.Timer(self.control_rate.sleep_dur, self.control_loop)

        rospy.loginfo("Drone Follower Node Initialized")

    def pose_callback(self, msg):
        """Callback for the drone's current pose."""
        self.current_pose = msg

    def yolo_detections_callback(self, msg):
        """Callback for YOLO detection messages."""
        self.latest_detections = msg
        
    def depth_callback(self, msg):
        """Callback for the aligned depth image."""
        try:
            # The depth image is typically 16-bit unsigned integer (CV_16UC1)
            # representing depth in millimeters.
            # Convert to float32 meters for calculations.
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Convert depth from mm to meters
            self.latest_depth_image = self.latest_depth_image.astype(np.float32) / 1000.0
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (depth image): {e}")

    def get_target_3d_position(self, detection):
        """
        Estimates the 3D position of the target relative to the camera.
        Requires camera intrinsics and a depth image.
        """
        if self.latest_depth_image is None:
            rospy.logwarn_throttle(5, "No depth image received yet. Estimating Z from fixed distance.")
            # Fallback if no depth image: assume target is at fixed Z distance
            # Map pixel error to relative 3D position
            
            # Convert pixel coordinates to camera frame (simplified for now)
            # x_c = (u - cx) * Z / fx
            # y_c = (v - cy) * Z / fy
            # z_c = Z

            center_x_pixel = detection.center.x
            center_y_pixel = detection.center.y
            
            # Simple conversion based on assumed fixed Z. This is highly inaccurate without depth.
            x_camera_frame = (center_x_pixel - self.camera_cx) * self.target_distance_z_meters / self.camera_fx
            y_camera_frame = (center_y_pixel - self.camera_cy) * self.target_distance_z_meters / self.camera_fy
            z_camera_frame = self.target_distance_z_meters # Assumed distance

            # Note: RealSense camera frame (Z-forward, X-right, Y-down)
            # Drone body frame (X-forward, Y-right, Z-down) -> Need careful transformation
            # For this simple P-controller, we'll try to map camera frame to body frame velocities
            # X_body = Z_camera (forward)
            # Y_body = -X_camera (left/right, assuming camera X is drone -Y)
            # Z_body = -Y_camera (up/down, assuming camera Y is drone -Z)
            
            # Let's return the 3D point in the camera's frame for now, transformation to drone body frame will happen next
            return np.array([z_camera_frame, -x_camera_frame, -y_camera_frame]) # Simplified map to roughly body-aligned
            

        # --- Using Depth Image for more accurate Z ---
        center_x_pixel = int(detection.center.x)
        center_y_pixel = int(detection.center.y)

        # Ensure pixel coordinates are within image bounds
        if 0 <= center_y_pixel < self.latest_depth_image.shape[0] and \
           0 <= center_x_pixel < self.latest_depth_image.shape[1]:
            
            depth_value = self.latest_depth_image[center_y_pixel, center_x_pixel]

            if depth_value > 0 and not np.isnan(depth_value): # Check for valid depth
                z_camera_frame = depth_value
                # Convert pixel coordinates to camera frame 3D point
                # x_c = (u - cx) * Z / fx
                # y_c = (v - cy) * Z / fy
                x_camera_frame = (center_x_pixel - self.camera_cx) * z_camera_frame / self.camera_fx
                y_camera_frame = (center_y_pixel - self.camera_cy) * z_camera_frame / self.camera_fy

                # Transform from RealSense camera frame (Z-forward, X-right, Y-down)
                # to a more drone-body-like frame (X-forward, Y-right, Z-down)
                # This is a simplified rotation. A proper TF transform is better.
                # drone_body_x = camera_z  (forward)
                # drone_body_y = camera_x  (right)
                # drone_body_z = -camera_y (down, so positive is up) - for cmd_vel linear.z
                # For cmd_vel linear.z, usually positive is UP, so we want -Y_camera

                # Target 3D position relative to the drone's body-fixed frame (approximately)
                # x_vel (forward/backward), y_vel (left/right), z_vel (up/down)
                return np.array([z_camera_frame, -x_camera_frame, -y_camera_frame])
            else:
                rospy.logwarn_throttle(2, f"Invalid depth at pixel ({center_x_pixel}, {center_y_pixel}). Using assumed Z.")
                # Fallback to assumed Z if depth is invalid
                x_camera_frame = (center_x_pixel - self.camera_cx) * self.target_distance_z_meters / self.camera_fx
                y_camera_frame = (center_y_pixel - self.camera_cy) * self.target_distance_z_meters / self.camera_fy
                return np.array([self.target_distance_z_meters, -x_camera_frame, -y_camera_frame])
        else:
            rospy.logwarn_throttle(2, f"Target pixel ({center_x_pixel}, {center_y_pixel}) out of depth image bounds. Using assumed Z.")
            x_camera_frame = (center_x_pixel - self.camera_cx) * self.target_distance_z_meters / self.camera_fx
            y_camera_frame = (center_y_pixel - self.camera_cy) * self.target_distance_z_meters / self.camera_fy
            return np.array([self.target_distance_z_meters, -x_camera_frame, -y_camera_frame])

    def control_loop(self, event):
        """
        Main control loop executed periodically.
        Calculates and publishes velocity commands.
        """
        if self.current_pose is None:
            rospy.loginfo_throttle(1, "Waiting for drone pose...")
            return
        if self.latest_detections is None or not self.latest_detections.detections:
            rospy.loginfo_throttle(1, "Waiting for YOLO detections or no target detected...")
            # If no detections, send hover command (zero velocity)
            self.publish_velocity(0, 0, 0, 0)
            return

        # --- Target Selection ---
        # Find the first detection that matches target_class_ids
        target_detection = None
        for det in self.latest_detections.detections:
            if det.class_name in self.model.names and \
               self.model.names.index(det.class_name) in self.target_class_ids:
                target_detection = det
                break # Found a target, break loop (can add more complex logic here)
        
        if target_detection is None:
            rospy.loginfo_throttle(1, "No target object found among detections.")
            self.publish_velocity(0, 0, 0, 0) # Hover if no target
            return

        # --- Estimate Target's 3D Position Relative to Camera ---
        # This function returns a 3D point in a frame roughly aligned with the drone body
        # (X-forward, Y-right, Z-up for linear.z)
        target_pos_body_approx = self.get_target_3d_position(target_detection)
        
        # --- Calculate Errors ---
        # Error in drone's body frame (forward, right, up/down)
        # target_pos_body_approx[0] is roughly forward distance (Z_camera)
        # target_pos_body_approx[1] is roughly right/left error (X_camera)
        # target_pos_body_approx[2] is roughly up/down error (Y_camera)

        # Desired position is directly in front of drone at target_distance_z_meters, and centered in X/Y
        error_x = target_pos_body_approx[0] - self.target_distance_z_meters # Error in forward distance
        error_y = target_pos_body_approx[1] # Error in lateral position (left/right)
        error_z = target_pos_body_approx[2] # Error in vertical position (up/down)

        # Yaw error: Target's x-pixel coordinate relative to image center
        yaw_error_pixel = target_detection.center.x - self.center_pixel_x
        
        # --- Apply P-Controller ---
        vel_x = -self.kp_z * error_x # Negative because if target is too far (+error_x), go forward (positive vel_x)
        vel_y = -self.kp_x * error_y # Negative because if target is too far right (+error_y), go left (negative vel_y)
        vel_z = -self.kp_y * error_z # Negative because if target is too high (+error_z), go down (negative vel_z)

        # Yaw control: if target is right of center, turn right (positive yaw rate)
        vel_yaw = -self.kp_yaw * yaw_error_pixel # Negative because right pixel error means turn right (positive yaw)

        # --- Clamp Velocities to Max Limits ---
        vel_x = np.clip(vel_x, -self.max_velocity_x, self.max_velocity_x)
        vel_y = np.clip(vel_y, -self.max_velocity_y, self.max_velocity_y)
        vel_z = np.clip(vel_z, -self.max_velocity_z, self.max_velocity_z)
        vel_yaw = np.clip(vel_yaw, -self.max_yaw_rate, self.max_yaw_rate)

        rospy.loginfo_throttle(0.5, f"Target: '{target_detection.class_name}' | "
                                    f"Pixel Error: X={target_detection.center.x - self.center_pixel_x:.1f}, Y={target_detection.center.y - self.center_pixel_y:.1f} | "
                                    f"Est. Rel. Pos (X,Y,Z): {target_pos_body_approx[0]:.2f}, {target_pos_body_approx[1]:.2f}, {target_pos_body_approx[2]:.2f} | "
                                    f"Cmd Vel (X,Y,Z,Yaw): {vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f}, {vel_yaw:.2f}")

        self.publish_velocity(vel_x, vel_y, vel_z, vel_yaw)

    def publish_velocity(self, vx, vy, vz, yaw_rate):
        """
        Publishes velocity commands as a TwistStamped message.
        Linear X: Forward/Backward
        Linear Y: Left/Right
        Linear Z: Up/Down
        Angular Z: Yaw Rate
        """
        vel_msg = TwistStamped()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.header.frame_id = 'base_link' # Or 'odom' depending on MAVROS setup

        vel_msg.twist.linear.x = float(vx)
        vel_msg.twist.linear.y = float(vy)
        vel_msg.twist.linear.z = float(vz)
        vel_msg.twist.angular.z = float(yaw_rate) # Yaw rate

        self.velocity_pub.publish(vel_msg)

if __name__ == '__main__':
    try:
        node = DroneFollower()
        rospy.spin() # Keep the node alive
    except rospy.ROSInterruptException:
        pass