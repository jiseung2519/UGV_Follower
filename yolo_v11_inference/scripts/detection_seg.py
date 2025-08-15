#!/usr/bin/env python
import time
import numpy as np
from scipy import stats
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError

# Initialize the ROS node
rospy.init_node("ultralytics")
time.sleep(1)

# Load the YOLO segmentation model
segmentation_model = YOLO("yolo11m-seg.pt")

# Publisher to send detection classes with distances
classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)

# Create a CvBridge instance for converting ROS images to OpenCV format
bridge = CvBridge()

def callback(depth_data):
    """Callback function to process depth image and RGB image."""
    try:
        # Retrieve the RGB image and convert it to OpenCV format
        color_image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        color_image = bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        
        # Convert depth image to OpenCV format
        depth_image = bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")
        
        # Run the YOLO segmentation model on the color image
        result = segmentation_model(color_image)
        
        # Prepare a list to store detected objects and their distances
        detected_objects = []

        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            
            # # Extract mask and calculate average distance for the object
            # mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(bool)
            # obj_depth_values = depth_image[mask]
            # obj_depth_values = obj_depth_values[~np.isnan(obj_depth_values)]
            # avg_distance = np.mean(obj_depth_values) if len(obj_depth_values) else float('inf')
            
            # # Append the object name and distance to the list
            # detected_objects.append(f"{name}: {avg_distance:.2f}m")

            # Only process objects classified as "cup"
            if name.lower() in ["cup", "keyboard"]:
                # Extract mask and calculate average distance for the cup
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(bool)
                obj_depth_values = depth_image[mask]
                obj_depth_values = obj_depth_values[~np.isnan(obj_depth_values)]
                avg_distance = np.mean(obj_depth_values) if len(obj_depth_values) else float('inf')
                obj_depth_values = obj_depth_values[obj_depth_values > 0]  # Filter out zero values
                if len(obj_depth_values):
                    mode_filter_distance = stats.mode(obj_depth_values, keepdims=True)[0][0]
                else:
                    mode_filter_distance = float('inf')                
                # Append the cup's name and distance to the list
                # detected_objects.append(f"{name}: {avg_distance:.2f}mm")
                detected_objects.append(f"{name}: {mode_filter_distance:.2f}mm")

        # Publish the list of objects with their average distances
        classes_pub.publish(String(data=str(detected_objects)))

    except CvBridgeError as e:
        rospy.logerr(f"Error converting images: {e}")

# Subscriber to depth image topic
rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)

# Keep the node running
rospy.spin()
