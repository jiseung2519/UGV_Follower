#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# Load your custom YOLO model
model = YOLO("/home/usrg/catkin_ws/src/yolo_v11_inference/models/ugvvv.pt")
device = next(model.model.parameters()).device
rospy.loginfo(f"Using device: {device}")

class YOLONode:
    def __init__(self):
        rospy.init_node("yolo_inference_node")

        # Publishers
        self.detected_pub = rospy.Publisher("/yolo/detected", Bool, queue_size=1)
        self.bbox_pub = rospy.Publisher("/yolo/bounding_box_pixels", Int32MultiArray, queue_size=10)
        self.image_pub = rospy.Publisher("/yolo/bbox_overlayed_image", Image, queue_size=10)

        # Subscriber
        self.image_sub = rospy.Subscriber("/iris/camera/rgb/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        results = model(cv_image)
        selected_bbox = None
        max_conf = -1.0

        for result in results:
            for obj in result.boxes:
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                conf = float(obj.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    selected_bbox = [x1, y1, x2, y2, int(conf * 1000)]  # scaled int

        # Draw and publish
        if selected_bbox:
            x1, y1, x2, y2, conf_int = selected_bbox
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"Conf: {conf_int/1000:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bbox_msg = Int32MultiArray()
            bbox_msg.data = selected_bbox
            self.bbox_pub.publish(bbox_msg)
            self.detected_pub.publish(Bool(data=True))
        else:
            self.bbox_pub.publish(Int32MultiArray())
            self.detected_pub.publish(Bool(data=False))

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

if __name__ == "__main__":
    node = YOLONode()
    rospy.spin()
