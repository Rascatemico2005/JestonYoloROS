cmake_minimum_required(VERSION 3.0.2)
project(yolo_detect)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  Segmentation.msg
  SegmentationArray.msg
  Classification.msg
  ClassificationArray.msg
  PoseArray.msg
  PoseEstimate.msg
  PoseKeypoint.msg
  OBBDetection.msg
  OBBDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)

include_directories(
include
  ${catkin_INCLUDE_DIRS}
)

