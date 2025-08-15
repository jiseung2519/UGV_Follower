#!/usr/bin/env python

import time
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from ultralytics import YOLO
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Initialize the ROS node
rospy.init_node("tensorrt_inference_node")
time.sleep(1)

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT Engine
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

model = YOLO("yolo11n.pt")
model.export(format="engine")  # creates 'yolov8n.engine'

# Load and create the TensorRT engine
engine = load_engine("/home/re540/catkin_ws/yolo11n.engine")
context = engine.create_execution_context()

# Allocate memory buffers
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)

# Initialize the CvBridge
bridge = CvBridge()

# ROS Publishers
classes_pub = rospy.Publisher("/tensorrt/detection/distance", String, queue_size=5)
point_pub = rospy.Publisher("/tensorrt/detection/point", PointStamped, queue_size=5)
marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=5)
pointcloud_pub = rospy.Publisher("/tensorrt/detection/pointcloud", PointCloud2, queue_size=5)
cropped_image_pub = rospy.Publisher("/tensorrt/detection/cropped_depth_image", Image, queue_size=5)

# Preprocess the image for TensorRT
def preprocess_image(image, input_shape=(640, 640)):
    image = cv2.resize(image, input_shape)
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    return image

# Inference function
def infer(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

# ROS Callback function
def callback(depth_data):
    try:
        # Retrieve and preprocess the RGB image
        color_image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        color_image = bridge.imgmsg_to_cv2(color_image_msg, "bgr8")
        input_image = preprocess_image(color_image)
        np.copyto(inputs[0]["host"], input_image.ravel())

        # Run inference
        output = infer(context, bindings, inputs, outputs, stream)

        # Process output (assuming bounding boxes, scores, and classes are in output)
        boxes, scores, classes = process_output(output)  # Define based on model output format

        # Load and process the depth image
        depth_image = bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)

        for box, score, cls in zip(boxes, scores, classes):
            if score < 0.5:  # Confidence threshold
                continue

            name = "cup" if int(cls) == 0 else "keyboard"  # Modify class IDs as necessary
            if name in ["cup", "keyboard"]:
                x_min, y_min, x_max, y_max = map(int, box)
                masked_depth_image[y_min:y_max, x_min:x_max] = depth_image[y_min:y_max, x_min:x_max] / 1000.0
                valid_depth_values = masked_depth_image[masked_depth_image > 0]
                if len(valid_depth_values) == 0:
                    continue

                fx, fy = 424.618896484375, 424.618896484375
                cx, cy = 419.52734375, 239.46791076660156

                points = []
                for v in range(masked_depth_image.shape[0]):
                    for u in range(masked_depth_image.shape[1]):
                        z = masked_depth_image[v, u]
                        if z > 0:
                            x = (u - cx) * z / fx
                            y = (v - cy) * z / fy
                            points.append((x, y, z))

                if len(points) == 0:
                    continue

                header = depth_data.header
                header.frame_id = "camera_depth_optical_frame"
                pointcloud_msg = pc2.create_cloud_xyz32(header, points)
                pointcloud_pub.publish(pointcloud_msg)

                points_np = np.array(points)
                centroid = points_np.mean(axis=0)
                position_x, position_y, position_z = centroid

                point = PointStamped()
                point.header.frame_id = "camera_depth_optical_frame"
                point.point.x = position_x
                point.point.y = position_y
                point.point.z = position_z
                point_pub.publish(point)

                marker = Marker()
                marker.header.frame_id = "camera_depth_optical_frame"
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = position_x
                marker.pose.position.y = position_y
                marker.pose.position.z = position_z
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                marker.color.r = 1.0 if name == "cup" else 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0 if name == "keyboard" else 0.0
                marker.pose.orientation.w = 1.0
                marker_pub.publish(marker)

                color = (0, 255, 0) if name == "cup" else (255, 0, 0)
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{name}: {position_z:.2f}m"
                cv2.putText(color_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if np.count_nonzero(masked_depth_image) > 0:
                    min_val, max_val = np.min(masked_depth_image[masked_depth_image > 0]), np.max(masked_depth_image)
                    normalized_depth = cv2.normalize(masked_depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    masked_depth_msg = bridge.cv2_to_imgmsg(normalized_depth, encoding="8UC1")
                    masked_depth_msg.header = depth_data.header
                    cropped_image_pub.publish(masked_depth_msg)

        if detected_objects:
            classes_pub.publish(String(data=str(detected_objects)))

    except CvBridgeError as e:
        rospy.logerr(f"Error converting images: {e}")

# ROS subscriber
image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)

# ROS loop
rospy.spin()
