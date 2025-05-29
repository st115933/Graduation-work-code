import sys
sys.path.append(f'/home/ubuntu/anaconda3/envs/mypytorch/lib/python3.8/site-packages/')

from typing import Optional, NamedTuple

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results

from sensor_msgs.msg import Image
from yolov8_msgs.msg import Msgpoint, MsgBoundingBox, Mask, KeyPoint, KeyPointArray, Detection, DetectionArray
from std_srvs.srv import SetBool


class DetectionData(NamedTuple):
    class_id: int
    class_name: str
    score: float
    bbox: Optional[MsgBoundingBox]
    mask: Optional[Mask]
    keypoints: Optional[KeyPointArray]


class Yolov8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')

        # Parameter declaration area
        self.declare_parameter('roi_x', 100)  
        self.declare_parameter('roi_y', 100)  
        self.declare_parameter('roi_width', 400)  
        self.declare_parameter('roi_height', 300)  
        self.declare_parameter('yolomodel', 'yolov8m.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('yoloenable', True)

        # Get the parameter area
        self.roi_x = self.get_parameter('roi_x').get_parameter_value().integer_value
        self.roi_y = self.get_parameter('roi_y').get_parameter_value().integer_value
        self.roi_width = self.get_parameter('roi_width').get_parameter_value().integer_value
        self.roi_height = self.get_parameter('roi_height').get_parameter_value().integer_value
        yolomodel = self.get_parameter('yolomodel').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.yoloenable = self.get_parameter('yoloenable').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.yolo = YOLO(yolomodel)
        self.yolo.fuse()

        # ROS2 publisher and recipient
        self.pub = self.create_publisher(DetectionArray, 'detections', 10)
        self.sub = self.create_subscription(
            Image, 'image_raw', self.handle_image, qos_profile_sensor_data
        )
        self.srv = self.create_service(SetBool, 'yoloenable', self.toggle_yoloenable)

    def toggle_yoloenable(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        self.yoloenable = req.data
        res.success = True
        return res

    # Extract the bounding box and convert it into a ros2 custom message
    def extract_bbox(self, box, x_offset: int, y_offset: int) -> MsgBoundingBox:
        xywh = box.xywh[0]
        bbox = MsgBoundingBox()
        bbox.center.position.x = float(xywh[0]) + x_offset
        bbox.center.position.y = float(xywh[1]) + y_offset
        bbox.size.x = float(xywh[2])
        bbox.size.y = float(xywh[3])
        return bbox

    # Convert mask to ros2 custom message
    def extract_mask(self, mask, h: int, w: int, x_offset: int, y_offset: int) -> Mask:
        m = Mask()

        m.data = [Msgpoint(x=float(x)+x_offset, y=float(y)+y_offset) 
                 for x, y in mask.xy[0].tolist()]
        m.height = h
        m.width = w
        return m

    # Extract key points and convert to ros2 custom messages
    def extract_keypoints(self, points, x_offset: int, y_offset: int) -> KeyPointArray:
        kp_array = KeyPointArray()
        if points.conf is None:
            return kp_array
        for idx, (coord, conf) in enumerate(zip(points.xy[0], points.conf[0]), start=1):
            if conf < self.confidence_threshold:
                continue
            kp = KeyPoint()
            kp.id = idx
            kp.point.x = float(coord[0]) + x_offset
            kp.point.y = float(coord[1]) + y_offset
            kp.score = float(conf)
            kp_array.data.append(kp)
        return kp_array

    def handle_image(self, msg: Image) -> None:
        if not self.yoloenable:
            return

        # Image preprocessing
        img = self.bridge.imgmsg_to_cv2(msg)
        h, w = img.shape[:2]

        # Calculate the coordinates of the actual ROI area (to prevent crossing the border)
        x1 = max(0, self.roi_x)
        y1 = max(0, self.roi_y)
        x2 = min(w, x1 + self.roi_width)
        y2 = min(h, y1 + self.roi_height)
        
        # Crop the ROI area for reasoning
        roi_img = img[y1:y2, x1:x2]
        
        # Reasoning only in the ROI area
        res = self.yolo.predict(source=roi_img, conf=self.confidence_threshold,
                                device=self.device, verbose=False)[0].cpu()

        # Extract and process bounding boxes, masks, and key points from model inference results
        bboxes = [self.extract_bbox(b, x1, y1) for b in res.boxes] if res.boxes else []
        masks = [self.extract_mask(m, h, w, x1, y1) for m in res.masks] if res.masks else []
        keypoints = [self.extract_keypoints(k, x1, y1) for k in res.keypoints] if res.keypoints else []

        detections = []
        for idx, box_data in enumerate(res.boxes):
            det = DetectionData(
                class_id=int(box_data.cls),
                class_name=self.yolo.names[int(box_data.cls)],
                score=float(box_data.conf),
                bbox=bboxes[idx] if idx < len(bboxes) else None,
                mask=masks[idx] if idx < len(masks) else None,
                keypoints=keypoints[idx] if idx < len(keypoints) else None
            )
            detections.append(det)

        # Publish processing results
        out_msg = DetectionArray()
        out_msg.header = msg.header
        for det in detections:
            d = Detection()
            d.class_id = det.class_id
            d.class_name = det.class_name
            d.score = det.score
            if det.bbox:
                d.bbox = det.bbox
            if det.mask:
                d.mask = det.mask
            if det.keypoints:
                d.keypoints = det.keypoints
            out_msg.detections.append(d)

        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()