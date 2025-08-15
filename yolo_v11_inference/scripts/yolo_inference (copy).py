#!/usr/bin/env python3
import rospy
import os
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

model = YOLO("/home/usrg/catkin_ws/src/yolo_v11_inference/models/yolo11n.pt")

class YOLONode:
    def __init__(self):
        rospy.init_node("yolo_inference_node")
        
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.detection_pub = rospy.Publisher("/yolo/bbox_overlayed_image", Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("/yolo/bounding_box_pixels", Int32MultiArray, queue_size=10)
        self.gpt_img_on_sub = rospy.Subscriber("/gpt_img_on", Bool, self.gpt_img_on_callback)
        self.gpt_img_pub = rospy.Publisher("/gpt_img", Image, queue_size=10)
        self.bridge = CvBridge()
        
        self.target_classes = {"car"}
        # Boolean flag to control /gpt_img publishing "black_big",
        self.gpt_img_on = False

    def gpt_img_on_callback(self, msg):
        """
        Callback to update the gpt_img_on flag based on the /gpt_img_on topic.
        """
        self.gpt_img_on = msg.data
    def image_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        results = model(cv_image)
        selected_bbox = None
        max_x_diff = -1
        max_y_diff = -1 

        for result in results:
            for obj in result.boxes:
                cls_name = model.names[int(obj.cls)]
                if cls_name in self.target_classes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    # Add each bounding box as a list to the bounding_boxes_list
                    x_diff = abs(x2 - x1)
                    y_diff = abs(y2 - y1)

                    # Select the bounding box with the largest x difference
                    # If x_diff is the same, select based on y_diff
                    # If both are the same, keep the first detected
                    if (x_diff > max_x_diff or 
                        (x_diff == max_x_diff and y_diff > max_y_diff)):
                        max_x_diff = x_diff
                        max_y_diff = y_diff
                        selected_bbox = [x1, y1, x2, y2]
        if selected_bbox:
            x1, y1, x2, y2 = selected_bbox
            # Draw bounding box on the image
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"Selected", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publish the selected bounding box
            bounding_boxes_msg = Int32MultiArray()
            bounding_boxes_msg.data = selected_bbox
            self.bbox_pub.publish(bounding_boxes_msg)
        else:
            # Publish an empty Int32MultiArray
            bounding_boxes_msg = Int32MultiArray()
            self.bbox_pub.publish(bounding_boxes_msg)         
           
        # Publish the annotated image
        bbox_overlayed_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.detection_pub.publish(bbox_overlayed_image)

        if self.gpt_img_on:
            rospy.loginfo("Publishing image to /gpt_img")
            gpt_img = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.gpt_img_pub.publish(gpt_img)
            self.gpt_img_on=False
            # # Get the height and width of the image
            # height, width = cv_image.shape[:2]
            
            # # Crop the upper half of the image
            # upper_half = cv_image[:height // 2, :]
            
            # # Convert the cropped image to a ROS image message
            # gpt_img = self.bridge.cv2_to_imgmsg(upper_half, "bgr8")
            
            # # Publish the cropped image
            # self.gpt_img_pub.publish(gpt_img)
            
            # # Reset the flag
            # self.gpt_img_on = False


if __name__ == "__main__":
    yolo_inference_node = YOLONode()
    rospy.spin()
