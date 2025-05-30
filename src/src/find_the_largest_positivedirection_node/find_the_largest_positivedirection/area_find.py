#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from yolov8_msgs.msg import DetectionArray, Detection
from cv_bridge import CvBridge, CvBridgeError
# Computer vision imports
import cv2
import numpy as np

# Define work area coordinates (top-left and bottom-right corners)
WORK_X1, WORK_Y1 = 493, 181
WORK_X2, WORK_Y2 = 732, 535

class YoloImageVisualizer(Node):
    """Visualizes YOLO detections and finds max empty square in work area"""
    
    def __init__(self):
        super().__init__('yolo_image_visualizer')
        
        # Initialize OpenCV-ROS bridge
        self.br = CvBridge()
        
        # Store latest detections
        self.detections = []

        # Publisher for max empty square center
        self.center_pub = self.create_publisher(
            DetectionArray,
            '/max_square_center',
            QoSProfile(depth=1)
        )

        # Subscribers for RGB image and YOLO detections
        self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.create_subscription(DetectionArray, '/yolo/detections', self.detections_callback, 10)

        # Create visualization window
        cv2.namedWindow('YOLO Visualizer', cv2.WINDOW_NORMAL)

    def detections_callback(self, msg: DetectionArray):
        """Store latest YOLO detection results"""
        self.detections = msg.detections

    def find_max_empty_square(self, mask: np.ndarray):

        h, w = mask.shape
        mask_list = mask.tolist()  # Convert to list for faster access
        max_size = 0
        max_i = max_j = 0

        # Use rolling array to store previous row
        prev_row = [0] * w

        # Process first row
        row = mask_list[0]
        current_row = []
        for j in range(w):
            if row[j] == 0:
                current = 1
                current_row.append(current)
                if current > max_size:
                    max_size = current
                    max_i, max_j = 0, j
            else:
                current_row.append(0)
        prev_row = current_row

        # Process remaining rows
        for i in range(1, h):
            row = mask_list[i]
            current_row = [0] * w

            # First column
            if row[0] == 0:
                current = 1
                current_row[0] = current
                if current > max_size:
                    max_size = current
                    max_i, max_j = i, 0

            # Rest of columns
            for j in range(1, w):
                if row[j] == 0:
                    # Get values from top, left, and top-left
                    a = prev_row[j]      # top cell
                    b = current_row[j-1] # left cell
                    c = prev_row[j-1]    # top-left cell
                    current = min(a, b, c) + 1
                    current_row[j] = current
                    if current > max_size:
                        max_size = current
                        max_i, max_j = i, j
                else:
                    current_row[j] = 0

            prev_row = current_row

        # Convert from bottom-right to top-left coordinates
        x = max_j - max_size + 1
        y = max_i - max_size + 1
        return x, y, max_size

    def image_callback(self, img_msg: Image):
        """Main processing callback for RGB images"""
        try:
            # Convert ROS image to OpenCV format
            frame = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # 1) Draw work area (blue rectangle + text)
        cv2.rectangle(frame, (WORK_X1, WORK_Y1), (WORK_X2, WORK_Y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Work Area', (WORK_X1, WORK_Y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 2) Create obstacle mask in work area
        h_roi = WORK_Y2 - WORK_Y1
        w_roi = WORK_X2 - WORK_X1
        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        
        # Project each detection bounding box onto mask
        for det in self.detections:
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            w_box, h_box = det.bbox.size.x, det.bbox.size.y
            x1 = int(cx - w_box/2) - WORK_X1
            y1 = int(cy - h_box/2) - WORK_Y1
            x2 = int(cx + w_box/2) - WORK_X1
            y2 = int(cy + h_box/2) - WORK_Y1
            
            # Clamp coordinates to mask boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_roi, x2), min(h_roi, y2)
            
            # Fill detection area with 1 (occupied)
            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 1

        # 3) Find largest empty square in work area
        x_rel, y_rel, size = self.find_max_empty_square(mask)
        abs_x1 = WORK_X1 + x_rel
        abs_y1 = WORK_Y1 + y_rel
        abs_x2 = abs_x1 + size
        abs_y2 = abs_y1 + size

        # 4) Draw all YOLO detection boxes (green)
        for det in self.detections:
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            w_box, h_box = det.bbox.size.x, det.bbox.size.y
            x1, y1 = int(cx - w_box/2), int(cy - h_box/2)
            x2, y2 = int(cx + w_box/2), int(cy + h_box/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{det.class_name}: {det.score:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 5) Draw largest empty square (red) if found
        if size > 0:
            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Max Square', (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 6) Calculate center coordinates and publish
            cx_pix = abs_x1 + size / 2.0
            cy_pix = abs_y1 + size / 2.0

            center_msg = DetectionArray()
            center_msg.header.stamp = self.get_clock().now().to_msg()
            center_msg.header.frame_id = img_msg.header.frame_id

            det_center = Detection()
            det_center.class_id = -1
            det_center.class_name = 'max_square_center'
            det_center.score = 1.0
            det_center.bbox.center.position.x = cx_pix
            det_center.bbox.center.position.y = cy_pix
            det_center.bbox.size.x = det_center.bbox.size.y = 0.0

            center_msg.detections = [det_center]
            self.center_pub.publish(center_msg)

            # Visualize center as red dot
            cv2.circle(frame, (int(cx_pix), int(cy_pix)), 5, (0, 0, 255), -1)

        # 7) Display final visualization
        cv2.imshow('YOLO Visualizer', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloImageVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
