#!/usr/bin/env python3

import rospy
import rospkg  # <-- Make sure this is imported
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
# The LD_PRELOAD export here is good, it will apply to this Python process
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
from ultralytics import YOLO

# Import your custom messages
from car_tracker_yolo.msg import Detection2D, Detection2DArray
from geometry_msgs.msg import Point # Assuming you defined center as geometry_msgs/Point

class YOLODetector:
    def __init__(self):
        rospy.init_node('yolo_detector_node', anonymous=True)

        self.bridge = CvBridge()

        # --- CORRECT WAY TO LOAD MODEL PATH IN ROS ---
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('car_tracker_yolo')
        
        # Construct the full path to your model
        # Assuming 'best.pt' is located in '~/catkin_ws/src/car_tracker_yolo/models/'
        model_path = os.path.join(package_path, 'models', 'best.pt')
        
        rospy.loginfo(f"Attempting to load YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        # Use .ckpt_path to confirm the exact path the model loaded from
        rospy.loginfo(f"YOLO model loaded successfully from: {self.model.ckpt_path}") 

        # --- PARAMETERS (Consider using ROS parameters for these for easier tuning) ---
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        
        # IMPORTANT: These are COCO dataset class IDs. 
        # You MUST verify what class IDs your 'best.pt' model was trained with.
        # Check `self.model.names` output below to confirm.
        self.target_class_ids = rospy.get_param('~target_class_ids', [0, 1]) # Example for custom model
        
        self.class_names = self.model.names # Get names mapping from the loaded model
        rospy.loginfo(f"Model's class names: {self.class_names}")
        rospy.loginfo(f"Filtering for target class IDs: {self.target_class_ids}")


        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)

        # Publisher for annotated image (for visualization)
        self.annotated_image_pub = rospy.Publisher('/yolo_detections_image', Image, queue_size=1)

        # Publisher for custom detection messages
        self.detection_pub = rospy.Publisher('/yolo_detections', Detection2DArray, queue_size=1)

        rospy.loginfo("YOLO Detector Node Initialized")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Perform YOLO inference
        # conf parameter filters detections by confidence score
        # verbose=False reduces console spam from Ultralytics itself
        results = self.model(cv_image, verbose=False, conf=self.confidence_threshold)

        # Create a message to hold all detections
        detections_array_msg = Detection2DArray()
        detections_array_msg.header = data.header # Use the image header for timestamp and frame_id

        # Add debug prints to see raw detections before filtering
        if not results or not results[0].boxes:
            rospy.loginfo("No objects detected by the YOLO model in this frame.")
        else:
            rospy.loginfo(f"YOLO detected {len(results[0].boxes)} objects in total before filtering.")
            for *xyxy, conf, cls in results[0].boxes.data:
                class_id = int(cls)
                class_name = "Unknown"
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                rospy.loginfo(f"  RAW Detection: Class='{class_name}' (ID: {class_id}), Confidence={conf:.2f}")

                # Filter for target classes (e.g., car, truck, bus)
                if class_id in self.target_class_ids:
                    rospy.loginfo(f"  --> Adding target detection: Class='{class_name}', Confidence={conf:.2f}")
                    detection = Detection2D()
                    detection.header = data.header
                    detection.class_name = class_name
                    detection.score = float(conf)

                    # Bounding box coordinates (xyxy is [x1, y1, x2, y2])
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Calculate center and size
                    detection.center.x = (x1 + x2) / 2.0
                    detection.center.y = (y1 + y2) / 2.0
                    detection.size_x = float(x2 - x1)
                    detection.size_y = float(y2 - y1)

                    detections_array_msg.detections.append(detection)
            
            if not detections_array_msg.detections:
                rospy.loginfo("No target objects found after applying class ID filter.")


        # Publish custom detection messages
        self.detection_pub.publish(detections_array_msg)

        # For visualization: draw bounding boxes on the image
        # YOLOv8's built-in plotting directly modifies the image
        annotated_frame = results[0].plot()

        # Publish the annotated image
        try:
            image_message = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            image_message.header = data.header # Ensure header matches original image
            self.annotated_image_pub.publish(image_message)
        except Exception as e:
            rospy.logerr(f"CvBridge Error (annotated image): {e}")

if __name__ == '__main__':
    try:
        node = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 