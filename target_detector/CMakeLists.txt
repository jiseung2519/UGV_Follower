cmake_minimum_required(VERSION 3.0.2)
project(target_detector)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
)

find_package(PCL REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS geometry_msgs pcl_conversions pcl_ros roscpp sensor_msgs std_msgs tf2 tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(target_detector_simple src/target_detector_simple.cpp)
target_link_libraries(target_detector_simple ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(target_detector_fusion src/target_detector_fusion.cpp)
# target_link_libraries(target_detector_fusion ${catkin_LIBRARIES} ${PCL_LIBRARIES})
