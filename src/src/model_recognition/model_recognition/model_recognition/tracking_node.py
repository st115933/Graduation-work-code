#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import os
import yaml
import sys

# Add Ultralytics library path
sys.path.append(f'/home/ubuntu/anaconda3/envs/mypytorch/lib/python3.8/site-packages/')
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import yaml_load, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.engine.results import Boxes
from yolov8_msgs.msg import DetectionArray, Detection
import collections

class TrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        
        # Declare and retrieve tracker configuration parameter
        self.declare_parameter('tracker_config', '')
        tracker_config_path = self.get_parameter('tracker_config').get_parameter_value().string_value
        
        # Initialize tracker
        self.tracker = self._init_tracker(tracker_config_path)
        
        # Initialize CvBridge and detection result buffer
        self.bridge = CvBridge()
        self.det_buffer = collections.deque(maxlen=10)  # Store the last 10 detection results
        
        # Time synchronization threshold (seconds)
        self.sync_threshold = 0.1
        
        # Create subscribers and publisher
        self.image_sub = self.create_subscription(
            Image, 
            'image_raw', 
            self._on_image, 
            10
        )
        
        self.detections_sub = self.create_subscription(
            DetectionArray,
            'detections',
            self._on_detections,
            10
        )
        
        self.tracking_pub = self.create_publisher(
            DetectionArray,
            'tracking',
            10
        )
        
        self.get_logger().info("Object tracking node initialized")

    def _init_tracker(self, cfg_path: str) -> BaseTrack:
        """Initialize Ultralytics tracker based on configuration file"""
        if not os.path.exists(cfg_path):
            self.get_logger().error(f"Tracker config file not found: {cfg_path}")
            return None
        
        try:
            cfg = check_yaml(cfg_path)
            params = IterableSimpleNamespace(**yaml_load(cfg))
            tracker_type = params.tracker_type.lower()
            
            self.get_logger().info(f"Initializing {tracker_type} tracker")
            
            if tracker_type == 'bytetrack':
                return BYTETracker(args=params, frame_rate=params.frame_rate)
            elif tracker_type == 'botsort':
                return BOTSORT(args=params, frame_rate=params.frame_rate)
            else:
                self.get_logger().error(f"Unknown tracker type: {tracker_type}")
                return None
        except Exception as e:
            self.get_logger().error(f"Error initializing tracker: {e}")
            return None

    def _on_detections(self, msg: DetectionArray):
        """Process received detection results"""
        # Store detection results in buffer
        self.det_buffer.append(msg)
        stamp = msg.header.stamp
        self.get_logger().debug(f"Received detections at {stamp.sec}.{stamp.nanosec}")

    def _on_image(self, img_msg: Image):
        """Process received image"""
        if not self.det_buffer:
            return
        
        # Get image timestamp (in float format)
        img_time = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        
        # Find the closest detection in time
        best_det = min(
            self.det_buffer,
            key=lambda det: abs((det.header.stamp.sec + det.header.stamp.nanosec * 1e-9) - img_time)
        )
        
        # Calculate time difference
        det_time = best_det.header.stamp.sec + best_det.header.stamp.nanosec * 1e-9
        delta = abs(det_time - img_time)
        
        # Check if the time difference is within the threshold
        if delta > self.sync_threshold:
            self.get_logger().warn(
                f"Large time delta between image and detections: {delta:.3f}s > {self.sync_threshold}s"
            )
            return
        
        try:
            # Convert image format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        # Prepare input for tracker
        dets = []
        for detection in best_det.detections:
            # Get bounding box information
            bbox = detection.bbox
            cx = bbox.center.position.x
            cy = bbox.center.position.y
            width = bbox.size.x
            height = bbox.size.y
            
            # Convert to top-left and bottom-right coordinates
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = x1 + width
            y2 = y1 + height
            
            # Get detection score and class ID
            score = detection.score
            class_id = detection.class_id
            
            dets.append([x1, y1, x2, y2, score, class_id])
        
        # Convert to numpy array
        dets_arr = np.array(dets)
        
        # Return directly if no detections are available
        if len(dets_arr) == 0:
            return
            
        # Convert to Boxes object
        dets_boxes = Boxes(dets_arr, (img_msg.height, img_msg.width))
        
        # Update tracker
        if self.tracker is not None:
            # Call tracker update
            online_targets = self.tracker.update(dets_boxes, cv_image)
            
            # Prepare message for publishing
            tracking_msg = DetectionArray()
            tracking_msg.header = img_msg.header  # Use the image's timestamp
            
            for target in online_targets:
                # Note: Ultralytics tracker returns 7 values
                if len(target) < 7:
                    continue
                    
                x1, y1, x2, y2, score, class_id, track_id = target[:7]
                
                # Create detection message
                detection = Detection()
                detection.id = str(int(track_id))
                
                # Set bounding box
                detection.bbox.center.position.x = (x1 + x2) / 2
                detection.bbox.center.position.y = (y1 + y2) / 2
                detection.bbox.size.x = x2 - x1
                detection.bbox.size.y = y2 - y1
                
                # Set classification information
                detection.score = float(score)
                detection.class_id = str(int(class_id))
                
                tracking_msg.detections.append(detection)
            
            # Publish tracking results
            self.tracking_pub.publish(tracking_msg)
            self.get_logger().info(f"Published {len(tracking_msg.detections)} tracked objects")

    def destroy_node(self):
        self.get_logger().info("Shutting down tracking node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
