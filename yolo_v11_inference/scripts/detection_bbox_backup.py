#!/usr/bin/env python

import time
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Initialize the ROS node
rospy.init_node("ultralytics")
time.sleep(1)

# Load the YOLO bounding-box detection model
detection_model = YOLO("yolo11n.pt")  # Use a model trained for bounding boxes only

# Publishers to send detection classes, distances, and visualized image
classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
bbox_center_pub = rospy.Publisher("/ultralytics/detection/bbox_center", Point, queue_size=5)
image_pub = rospy.Publisher("/ultralytics/detection/image_with_boxes", Image, queue_size=5)

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
        
        # Run the YOLO detection model on the color image
        result = detection_model(color_image)
        
        # Prepare a list to store detected objects and their distances
        detected_objects = []

        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            
            # Only process objects classified as "cup"
            if name.lower() in ["cup", "keyboard"]:
                # Extract bounding box coordinates
                box = result[0].boxes.xyxy[index].cpu().numpy()
                x_min, y_min, x_max, y_max = box

                # Calculate the center of the bounding box
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)

                # Get the depth at the bounding box center (with a small area around it for stability)
                depth_image_dilated = cv2.dilate(depth_image, np.ones((5, 5), np.uint8))
                depth_values = depth_image_dilated[max(center_y-3, 0):center_y+4, max(center_x-3, 0):center_x+4]
                depth_values = depth_values[depth_values > 0]  # Filter out zero values
                avg_distance = np.median(depth_values) if len(depth_values) else float('inf')
                
                # Append the cup's name and distance to the list
                detected_objects.append(f"{name}: {avg_distance:.2f}m")

                # Publish the bounding box center and average distance as a Point message
                bbox_center = Point()
                bbox_center.x = center_x
                bbox_center.y = center_y
                bbox_center.z = avg_distance  # Distance in meters
                
                # Publish the center coordinate of the bounding box
                bbox_center_pub.publish(bbox_center)

                # Draw bounding box and label on the image
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                label = f"{name}: {avg_distance:.2f}mm"
                cv2.putText(color_image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Publish the list of detected cups with distances
        if detected_objects:
            classes_pub.publish(String(data=str(detected_objects)))

        # Convert the image with bounding boxes to a ROS Image message and publish it
        image_msg = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        image_pub.publish(image_msg)

    except CvBridgeError as e:
        rospy.logerr(f"Error converting images: {e}")

# Subscriber to depth image topic
rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)

# Keep the node running
rospy.spin()
