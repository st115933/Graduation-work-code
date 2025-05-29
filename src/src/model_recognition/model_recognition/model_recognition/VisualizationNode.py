import sys
sys.path.append(f'/home/ubuntu/anaconda3/envs/mypytorch/lib/python3.8/site-packages/')
import cv2
import random
import numpy as np
import collections
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from yolov8_msgs.msg import Detection, DetectionArray

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        # Node status initialization
        self.bridge = CvBridge()
        self.class_color = {}  
        self.tolerance = 0.1   # Timestamp matching tolerance (seconds)
        self.tracking_buffer = collections.deque(maxlen=10)  # Tracking result buffer

        # Create publishers
        self.pub_img = self.create_publisher(Image, 'dbg_image', 10)
        self.pub_bb = self.create_publisher(MarkerArray, 'dgb_bb_markers', 10)
        self.pub_kp = self.create_publisher(MarkerArray, 'dgb_kp_markers', 10)

        # Create subscribers
        self.create_subscription(
            Image, 
            'image_raw', 
            self.on_image, 
            qos_profile_sensor_data
        )
        self.create_subscription(
            DetectionArray, 
            'tracking', 
            self.on_tracking, 
            10
        )

    def on_tracking(self, tracking_msg: DetectionArray):
        """Process tracking result messages"""
        # Add tracking results to the buffer
        self.tracking_buffer.append(tracking_msg)

    def on_image(self, img_msg: Image):
        """Process image messages"""
        if not self.tracking_buffer:
            return
            
        # Calculate image timestamp
        img_time = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        
        # Find the tracking result with the closest timestamp
        best_tracking = min(
            self.tracking_buffer, 
            key=lambda t: abs((t.header.stamp.sec + t.header.stamp.nanosec * 1e-9) - img_time)
        )
        
        # Check if the time difference is within tolerance
        tracking_time = best_tracking.header.stamp.sec + best_tracking.header.stamp.nanosec * 1e-9
        if abs(tracking_time - img_time) > self.tolerance:
            return
            
        # Process the image and tracking result pair
        self.process_pair(img_msg, best_tracking)

    def process_pair(self, img_msg: Image, tracking_msg: DetectionArray):
        """Process the image and corresponding tracking results"""
        # Convert image format
        cv_img = self.bridge.imgmsg_to_cv2(img_msg)
        annot = Annotator(cv_img)
        
        # Initialize marker arrays
        bb_markers = MarkerArray()
        kp_markers = MarkerArray()
        
        # Process each tracked object
        for track in tracking_msg.detections:
            cls = track.class_name
            
            # Assign a unique color to each class
            if cls not in self.class_color:
                self.class_color[cls] = tuple(random.choices(range(256), k=3))
            col = self.class_color[cls]
            
            # Extract bounding box information
            x, y, w, h = (
                track.bbox.center.position.x,
                track.bbox.center.position.y,
                track.bbox.size.x,
                track.bbox.size.y
            )
            
            # Calculate bounding box corners
            tl = (int(x - w/2), int(y - h/2))
            br = (int(x + w/2), int(y + h/2))
            
            # Draw bounding box and label
            label = f"{cls} ID:{track.id} ({track.score:.2f})"
            annot.box_label((*tl, *br), label, col)
            
            # Draw segmentation mask
            if track.mask and track.mask.data:
                pts = np.array([[p.x, p.y] for p in track.mask.data], np.int32)
                annot.masks([pts], [col])
            
            # Draw keypoints
            if track.keypoints and track.keypoints.data:
                kpts = [(int(kp.point.x), int(kp.point.y), kp.id) 
                        for kp in track.keypoints.data]
                annot.keypoints(kpts, track.keypoints.data)
        
        # Publish visualization results
        out_img = annot.result()
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(out_img, encoding=img_msg.encoding))
        self.pub_bb.publish(bb_markers)
        self.pub_kp.publish(kp_markers)

def main():
    rclpy.init()
    node = VisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
