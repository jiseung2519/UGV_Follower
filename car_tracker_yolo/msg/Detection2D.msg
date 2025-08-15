# ~/catkin_ws/src/car_tracker_yolo/msg/Detection2D.msg
# Represents a single 2D object detection

std_msgs/Header header
string class_name
float32 score
geometry_msgs/Point center    # Center of the bounding box (x, y pixels)
float32 size_x                # Width of the bounding box (pixels)
float32 size_y                # Height of the bounding box (pixels)