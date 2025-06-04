import sys
sys.path.append(f'/home/ubuntu/anaconda3/envs/mypytorch/lib/python3.8/site-packages/')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolov8_msgs.msg import DetectionArray, Detection, MsgBoundingBox, Mask, KeyPoint, KeyPointArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

class Yolov8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('roi_x', 100),
                ('roi_y', 100),
                ('roi_width', 400),
                ('roi_height', 300),
                ('model_path', 'yolov8m.pt'),
                ('confidence_threshold', 0.5),
                ('device', 'cuda:0')  # Add device selection parameter
            ]
        )
        
        # Get parameter values
        self.roi_x = self.get_parameter('roi_x').value
        self.roi_y = self.get_parameter('roi_y').value
        self.roi_width = self.get_parameter('roi_width').value
        self.roi_height = self.get_parameter('roi_height').value
        model_path = self.get_parameter('model_path').value
        self.conf_thres = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        
        # Initialize YOLOv8 model and move to specified device
        self.model = YOLO(model_path).to(device)
        self.bridge = CvBridge()
        
        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(
            DetectionArray,
            'detections',
            10)
        
        self.get_logger().info(
            f"YOLOv8 node started, using device: {device}, ROI region: "
            f"x={self.roi_x}, y={self.roi_y}, "
            f"width={self.roi_width}, height={self.roi_height}"
        )

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img_height, img_width, _ = cv_image.shape
            
            # Calculate ROI region with boundary protection
            x1 = max(0, self.roi_x)
            y1 = max(0, self.roi_y)
            x2 = min(img_width, self.roi_x + self.roi_width)
            y2 = min(img_height, self.roi_y + self.roi_height)
            
            # Calculate actual ROI dimensions
            actual_roi_width = x2 - x1
            actual_roi_height = y2 - y1
            
            if actual_roi_width <= 0 or actual_roi_height <= 0:
                self.get_logger().warn(
                    f"Invalid ROI region: "
                    f"Calculated dimensions: width={actual_roi_width}, height={actual_roi_height}"
                )
                return
            
            # Extract ROI region
            roi_image = cv_image[y1:y2, x1:x2]
            
            # Perform YOLOv8 inference
            results = self.model(
                roi_image,
                conf=self.conf_thres,
                verbose=False
            )
            
            # Prepare message to publish
            detection_array = DetectionArray()
            detection_array.header = msg.header
            
            # Process detection results
            for result in results:
                # Ensure all result elements exist
                boxes = result.boxes if hasattr(result, 'boxes') else []
                masks = result.masks if hasattr(result, 'masks') else []
                keypoints = result.keypoints if hasattr(result, 'keypoints') else []
                
                # Create iterators, ensuring consistent length
                max_len = max(len(boxes), len(masks), len(keypoints))
                boxes = list(boxes) + [None] * (max_len - len(boxes))
                masks = list(masks) + [None] * (max_len - len(masks))
                keypoints = list(keypoints) + [None] * (max_len - len(keypoints))
                
                for box, mask, kpts in zip(boxes, masks, keypoints):
                    # Skip invalid detections
                    if box is None:
                        continue
                        
                    # Create Detection message
                    detection = Detection()
                    
                    # Fill in basic information
                    detection.class_id = int(box.cls.item())
                    detection.class_name = result.names[detection.class_id]
                    detection.confidence = float(box.conf.item())
                    
                    # Process bounding box 
                    bbox = MsgBoundingBox()
                    
                    # Extract bounding box coordinates 
                    box_data = box.xyxy[0].cpu().numpy()
                    xmin = float(box_data[0]) + x1
                    ymin = float(box_data[1]) + y1
                    xmax = float(box_data[2]) + x1
                    ymax = float(box_data[3]) + y1
                    
                    # Convert to center+size format
                    center_x = (xmin + xmax) / 2.0
                    center_y = (ymin + ymax) / 2.0
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    bbox.center_position_x = center_x
                    bbox.center_position_y = center_y
                    bbox.center_theta = 0.0
                    bbox.size_x = width
                    bbox.size_y = height
                    detection.bbox = bbox
                    
                    # Process mask 
                    if mask is not None:
                        mask_msg = Mask()
                        # Get polygon contour points
                        try:
                            contour_points = mask.xy[0].astype(np.float32)
                            
                            # Convert to global coordinates and add offset
                            contour_points[:, 0] += x1
                            contour_points[:, 1] += y1
                            
                            # Fill contour point data
                            for point in contour_points:
                                mask_msg.data.append(Mask.Msgpoint(x=float(point[0]), y=float(point[1])))
                            
                            mask_msg.height = img_height
                            mask_msg.width = img_width
                            detection.mask = mask_msg
                        except Exception as e:
                            self.get_logger().warn(f"Error processing mask: {str(e)}")
                    
                    # Process keypoints 
                    if kpts is not None:
                        try:
                            kpts_msg = KeyPointArray()
                            kpts_data = kpts.xy[0].cpu().numpy()
                            kpts_conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else [1.0] * len(kpts_data)
                            
                            for i, (point, conf) in enumerate(zip(kpts_data, kpts_conf), 1):
                                kp = KeyPoint()
                                kp.id = i
                                kp.point.x = float(point[0]) + x1
                                kp.point.y = float(point[1]) + y1
                                kp.score = float(conf)
                                kpts_msg.data.append(kp)
                            
                            detection.keypoints = kpts_msg
                        except Exception as e:
                            self.get_logger().warn(f"Error processing keypoints: {str(e)}")
                    
                    detection_array.detections.append(detection)
            
            # Publish detection results
            self.publisher.publish(detection_array)
            self.get_logger().debug(f"Published {len(detection_array.detections)} detection results")
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
