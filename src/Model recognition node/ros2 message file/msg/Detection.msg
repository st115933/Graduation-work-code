# Define YOLOv8 test results

# class probability
int32 class_id
string class_name
float64 score

# Tracking name
string id

# Bounding box (enclosing objects in pixels)
ros2 message file/MsgBoundingBox bbox

# Split mask
ros2 message file/Mask mask

# Key points
ros2 message file/KeyPointArray keypoints
