# Set the minimum version requirements for CMake
cmake_minimum_required(VERSION 3.8)

# Define project name
project(launch
  VERSION 0.0.0
  DESCRIPTION "ROS2"
  LANGUAGES CXX
)

# If the compiler is GNU or Clang, add compilation options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find the ament_cmake package and make sure it is required
find_package(ament_cmake REQUIRED)

# Install the launch folder to the specified location
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}
)

# Finally call ament_package() to generate the necessary package configuration
ament_package()
