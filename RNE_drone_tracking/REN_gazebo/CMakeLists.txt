cmake_minimum_required(VERSION 3.0.2)
project(rne_gazebo)

## Compile as C++17
add_compile_options(-std=c++17)

## Find catkin macros and required ROS packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  mavros_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  angles
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    sensor_msgs
    mavros_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    angles
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Add executable and link libraries
add_executable(drone_follower_node src/offb_vel.cpp)

target_link_libraries(drone_follower_node
  ${catkin_LIBRARIES}
)

add_dependencies(drone_follower_node ${catkin_EXPORTED_TARGETS})

