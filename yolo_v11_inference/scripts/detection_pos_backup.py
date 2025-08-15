#!/usr/bin/env python

import time
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import cv2
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs

# Initialize the ROS node
rospy.init_node("ultralytics")
time.sleep(1)

# Load the YOLO segmentation model
segmentation_model = YOLO("yolo11s.pt")

# Publishers for distance, RViz marker, point cloud, and segmented image
classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
point_pub = rospy.Publisher("/ultralytics/detection/point", PointStamped, queue_size=5)
marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=5)
pointcloud_pub = rospy.Publisher("/ultralytics/detection/pointcloud", PointCloud2, queue_size=5)
cropped_image_pub = rospy.Publisher("/ultralytics/detection/cropped_depth_image", Image, queue_size=5)

# Create a CvBridge instance for converting ROS images to OpenCV format
bridge = CvBridge()

# Initialize tf2 buffer and listener
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

def callback(depth_data):
    """Callback function to process depth image and RGB image."""
    try:
        # Retrieve the RGB image and convert it to OpenCV format
        color_image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        color_image = bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        
        # Convert depth image to OpenCV format
        depth_image = bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")

        # Create a blank depth mask the same size as depth_image
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        
        # Run the YOLO segmentation model on the color image
        result = segmentation_model(color_image)
        
        # Prepare a list to store detected objects and their distances
        detected_objects = []

        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            
            # Only process objects classified as "cup" or "keyboard"
            if name.lower() in ["cup", "keyboard"]:
                # Extract bounding box coordinates
                box = result[0].boxes.xyxy[index].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, box)
                
                # Copy only the bounding box region from depth_image to masked_depth_image
                masked_depth_image[y_min:y_max, x_min:x_max] = depth_image[y_min:y_max, x_min:x_max] / 1000.0
                
                # Filter out zero values and compute the median depth
                valid_depth_values = masked_depth_image[masked_depth_image > 0]
                if len(valid_depth_values) == 0:
                    continue  # Skip if there are no valid depth values

                # Depth camera intrinsic parameters
                fx, fy = 424.618896484375, 424.618896484375  # Focal lengths
                cx, cy = 419.52734375, 239.46791076660156  # Principal points
                
                # Convert depth values to a point cloud
                points = []
                for v in range(masked_depth_image.shape[0]):
                    for u in range(masked_depth_image.shape[1]):
                        z = masked_depth_image[v, u]
                        if z > 0:  # Skip invalid depth
                            x = (u - cx) * z / fx
                            y = (v - cy) * z / fy
                            points.append((x, y, z))

                # Only proceed if there are valid points
                if len(points) == 0:
                    continue

                # Publish the point cloud data
                header = depth_data.header
                header.frame_id = "camera_depth_optical_frame"  # Set to your camera's frame
                pointcloud_msg = pc2.create_cloud_xyz32(header, points)
                pointcloud_pub.publish(pointcloud_msg)

                # Calculate the centroid of the point cloud
                points_np = np.array(points)
                centroid = points_np.mean(axis=0)
                position_x, position_y, position_z = centroid

                # Publish the 3D position as a PointStamped message for RViz visualization
                point = PointStamped()
                point.header.frame_id = "camera_depth_optical_frame"
                point.point.x = position_x
                point.point.y = position_y
                point.point.z = position_z
                point_pub.publish(point)

                # Visualize the position in RViz using a Marker
                marker = Marker()
                marker.header.frame_id = "camera_depth_optical_frame"
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = position_x
                marker.pose.position.y = position_y
                marker.pose.position.z = position_z
                marker.scale.x = 0.05  # Sphere diameter
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                marker.color.r = 1.0 if name == "cup" else 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0 if name == "keyboard" else 0.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker_pub.publish(marker)

                # Draw bounding box and label on the image
                color = (0, 255, 0) if name == "cup" else (255, 0, 0)
                cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                label = f"{name}: {position_z:.2f}m"
                cv2.putText(color_image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Append the object and depth info to the detected objects list for logging
                detected_objects.append(f"{name}: {position_z:.2f}m")
                
                # Normalize the masked depth image for visualization in RViz
                if np.count_nonzero(masked_depth_image) > 0:
                    min_val, max_val = np.min(masked_depth_image[masked_depth_image > 0]), np.max(masked_depth_image)
                    normalized_depth = cv2.normalize(masked_depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    normalized_depth = normalized_depth.astype(np.uint8)

                    # # Convert to a colorized depth image for better visibility (optional)
                    # colorized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

                    # Convert to ROS Image message and publish
                    masked_depth_msg = bridge.cv2_to_imgmsg(normalized_depth, encoding="8UC1")
                    masked_depth_msg.header = depth_data.header  # Keep original header information
                    cropped_image_pub.publish(masked_depth_msg)

        # Publish detected objects info
        if detected_objects:
            classes_pub.publish(String(data=str(detected_objects)))

    except CvBridgeError as e:
        rospy.logerr(f"Error converting images: %s", e)

# Subscriber to depth image topic
rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)

# Keep the node running
rospy.spin()
