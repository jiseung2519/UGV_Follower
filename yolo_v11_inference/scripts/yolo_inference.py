#!/usr/bin/env python3
import rospy
import os
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

model = YOLO("/home/usrg/catkin_ws/src/yolo_v11_inference/models/ugvvv.pt")
device = next(model.model.parameters()).device
rospy.loginfo(f"Using device: {device}")



class YOLONode:
    def __init__(self):
        rospy.init_node("yolo_inference_node")
        
        self.detected_pub = rospy.Publisher("/yolo/detected", Bool, queue_size=1)
        self.image_sub = rospy.Subscriber("/iris/camera/rgb/image_raw", Image, self.image_callback)
        self.detection_pub = rospy.Publisher("/yolo/bbox_overlayed_image", Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("/yolo/bounding_box_pixels", Int32MultiArray, queue_size=10)
        # self.gpt_img_on_sub = rospy.Subscriber("/gpt_img_on", Bool, self.gpt_img_on_callback)
        # self.gpt_img_pub = rospy.Publisher("/gpt_img", Image, queue_size=10)
        self.bridge = CvBridge()

        self.gpt_img_on = False  # flag to control GPT image publishing

    def gpt_img_on_callback(self, msg):
        self.gpt_img_on = msg.data

    def image_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        results = model(cv_image)
        selected_bbox = None
        max_area = -1

        for result in results:
            for obj in result.boxes:
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                area = abs(x2 - x1) * abs(y2 - y1)

                if area > max_area:
                    max_area = area
                    selected_bbox = [x1, y1, x2, y2]

        if selected_bbox:
            x1, y1, x2, y2 = selected_bbox
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, "Selected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bbox_msg = Int32MultiArray()
            bbox_msg.data = selected_bbox
            self.bbox_pub.publish(bbox_msg)
        else:
            self.bbox_pub.publish(Int32MultiArray())  # empty message

        if selected_bbox:
            self.detected_pub.publish(Bool(data=True))
        else:
            self.detected_pub.publish(Bool(data=False))

        # CORRECTED: This line is now outside the if/else block
        self.detection_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        if self.gpt_img_on:
            rospy.loginfo("Publishing image to /gpt_img")
            gpt_img = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.gpt_img_pub.publish(gpt_img)
            self.gpt_img_on = False

if __name__ == "__main__":
    yolo_inference_node = YOLONode()
    rospy.spin()
