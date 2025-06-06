# Set the minimum version requirements for CMake
cmake_minimum_required(VERSION 3.8)

# Define the project name and specify the language to be used as C and C++
project(ros2_message_file
  VERSION 0.0.0
  DESCRIPTION "Msgs for Ros2"
  LANGUAGES C CXX
)

# Find the necessary dependency packages
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ros2 message
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Msgpoint.msg"
  "msg/MsgBoundingBox.msg"
  "msg/Mask.msg"
  "msg/KeyPoint.msg"
  "msg/KeyPointArray.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
