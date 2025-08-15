#!/usr/bin/env python3
import rospy
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import message_filters
import cv2
from cv_bridge import CvBridge
import numpy as np

class YOLOWithDepthNode:
    def __init__(self):
        rospy.init_node('camera_detector_node')
        self.bridge = CvBridge()
        self.depth_image = None

        # Sync YOLO detection and depth image
        self.det_sub = message_filters.Subscriber("/yolov7/yolov7", Detection2DArray)
        self.depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.det_sub, self.depth_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.detection_callback)

        self.pub = rospy.Publisher("/car_tracker/detection", Float32MultiArray, queue_size=1)
        self.target_class = 0

    def detection_callback(self, detections, depth_msg):
        best_score = -1
        best_detection = None

        for det in detections.detections:
            if len(det.results) == 0:
                continue
            class_id = det.results[0].id
            score = det.results[0].score
            if class_id == self.target_class and score > best_score:
                best_score = score
                best_detection = det

        if best_detection is None:
            return

        cx = int(best_detection.bbox.center.x)
        cy = int(best_detection.bbox.center.y)
        w = best_detection.bbox.size_x
        h = best_detection.bbox.size_y

        # Convert depth image
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth = float(depth_img[cy, cx])
            if np.isnan(depth) or depth == 0.0:
                depth = -1.0  # Invalid depth
        except Exception as e:
            rospy.logwarn("Depth conversion failed: %s", e)
            depth = -1.0

        # Publish detection + depth
        output = Float32MultiArray()
        output.data = [cx, cy, w, h, depth]
        self.pub.publish(output)

if __name__ == "__main__":
    YOLOWithDepthNode()
    rospy.spin()
