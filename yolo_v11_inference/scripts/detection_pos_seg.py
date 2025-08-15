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
segmentation_model = YOLO("yolo11m-seg.pt")

# Publishers for distance, RViz marker, point cloud, and segmented image
classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
point_pub = rospy.Publisher("/ultralytics/detection/point", PointStamped, queue_size=5)
marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=5)
pointcloud_pub = rospy.Publisher("/ultralytics/detection/pointcloud", PointCloud2, queue_size=5)
segmented_image_pub = rospy.Publisher("/ultralytics/detection/segmented_image", Image, queue_size=5)

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

        # Create a black image for showing only segmented regions
        segmented_image = np.zeros_like(color_image)
        
        # Convert depth image to OpenCV format
        depth_image = bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")
        
        # Run the YOLO segmentation model on the color image
        result = segmentation_model(color_image)
        
        # Prepare a list to store detected objects and their distances
        detected_objects = []

        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            
            # Only process objects classified as "cup" or "keyboard"
            if name.lower() in ["cup", "keyboard"]:
                # Extract the segmentation mask for the detected object
                mask = result[0].masks.data.cpu().numpy()[index].astype(bool)
                
                # Resize the mask to match the color image dimensions if needed
                if mask.shape[:2] != color_image.shape[:2]:
                    mask_color = cv2.resize(mask.astype(np.uint8), (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

                # Resize the mask to match the depth image dimensions if needed
                if mask.shape[:2] != depth_image.shape[:2]:
                    mask_depth = cv2.resize(mask.astype(np.uint8), (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                                          
                # Mask the original image to only show the segmented area
                segmented_image[mask_color] = color_image[mask_color]

                # Mask the depth image for relevant values only
                masked_depth = depth_image[mask_depth]
                masked_depth = masked_depth[masked_depth > 0] / 1000.0  # Convert mm to meters
                if len(masked_depth) == 0:
                    continue  # Skip if no valid depth values are found

                # Calculate the centroid of the segmentation mask
                mask_coords = np.argwhere(mask)
                center_y, center_x = np.mean(mask_coords, axis=0).astype(int)

                # Intrinsic parameters
                fx, fy = 424.618896484375, 424.618896484375  # Focal lengths
                cx, cy = 419.52734375, 239.46791076660156  # Principal points
                
                # Generate 3D points from the masked depth image
                points = []
                for y, x in mask_coords:
                    z = depth_image[y, x] / 1000.0  # Convert depth to meters
                    if z > 0:
                        x_pos = (x - cx) * z / fx
                        y_pos = (y - cy) * z / fy
                        points.append((x_pos, y_pos, z))

                # Publish the point cloud for the segmented area
                header = depth_data.header
                header.frame_id = "camera_depth_optical_frame"
                pointcloud_msg = pc2.create_cloud_xyz32(header, points)
                pointcloud_pub.publish(pointcloud_msg)

                # Transform the centroid point to the RGB frame
                try:
                    # Create a PointStamped message in the depth frame
                    point_depth_frame = PointStamped()
                    point_depth_frame.header.frame_id = "camera_depth_optical_frame"
                    point_depth_frame.header.stamp = rospy.Time.now()
                    depth_value = depth_image[center_y, center_x] / 1000.0  # Depth in meters
                    position_x = (center_x - cx) * depth_value / fx
                    position_y = (center_y - cy) * depth_value / fy
                    position_z = depth_value

                    # Set centroid point
                    point_depth_frame.point.x = position_x
                    point_depth_frame.point.y = position_y
                    point_depth_frame.point.z = position_z

                    # Lookup the transform from depth to RGB frame
                    transform = tf_buffer.lookup_transform("camera_color_optical_frame", "camera_depth_optical_frame", rospy.Time(0))

                    # Transform the point
                    point_rgb_frame = tf2_geometry_msgs.do_transform_point(point_depth_frame, transform)

                    # Extract transformed coordinates
                    position_x = point_rgb_frame.point.x
                    position_y = point_rgb_frame.point.y
                    position_z = point_rgb_frame.point.z

                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                    rospy.logerr("TF Exception: %s", e)
                    continue

                # Publish the transformed point
                point_pub.publish(point_rgb_frame)

                # Visualize the centroid in RViz using a Marker
                marker = Marker()
                marker.header.frame_id = "camera_color_optical_frame"
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

                # Label the centroid position in meters on the segmented image
                label = f"{name}: {position_z:.2f}m"
                cv2.putText(segmented_image, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Append the object and depth info to the detected objects list for logging
                detected_objects.append(f"{name}: {position_z:.2f}m")

        # Publish detected objects info
        if detected_objects:
            classes_pub.publish(String(data=str(detected_objects)))

        # Publish the segmented image (only segmented areas visible)
        segmented_image_msg = bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
        segmented_image_msg.header.frame_id = "camera_color_optical_frame"
        segmented_image_pub.publish(segmented_image_msg)

    except CvBridgeError as e:
        rospy.logerr(f"Error converting images: %s", e)

# Subscriber to depth image topic
rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)

# Keep the node running
rospy.spin()
