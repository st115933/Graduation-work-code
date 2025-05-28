import sys
sys.path.append(f'/home/ubuntu/anaconda3/envs/mypytorch/lib/python3.8/site-packages/')
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import yaml_load, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml, check_requirements
from ultralytics.engine.results import Boxes
from sensor_msgs.msg import Image
from yolov8_msgs.msg import DetectionArray, Detection
from std_msgs.msg import Header
from std_srvs.srv import SetBool
import collections


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')

        # parameters
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        yaml_file = self.get_parameter('tracker_config').get_parameter_value().string_value

        # initialize utilities
        self.bridge = CvBridge()
        self.tracker = self._init_tracker(yaml_file)

        # buffers
        self.det_buffer = collections.deque(maxlen=10)  # Cache the last 10 test results

        # subscribers
        self.create_subscription(
            Image, 'image_raw', self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            DetectionArray, 'detections', self._on_detections, 10)

        # publisher
        self._pub = self.create_publisher(DetectionArray, 'tracking', 10)

    def _init_tracker(self, cfg_path: str) -> BaseTrack:
        # ensure dependencies
        check_requirements('lap')
        cfg = check_yaml(cfg_path)
        params = IterableSimpleNamespace(**yaml_load(cfg))
        cls = BYTETracker if params.tracker_type == 'bytetrack' else BOTSORT
        return cls(args=params, frame_rate=params.frame_rate if hasattr(params, 'frame_rate') else 1)

    def _on_detections(self, msg: DetectionArray):
        # Add test results to the buffer
        self.det_buffer.append(msg)

    def _on_image(self, img_msg: Image):
        if not self.det_buffer:
            return

        # Get the timestamp of the image
        img_time = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9

        # Find the test result closest to the image timestamp
        best_det = min(
            self.det_buffer,
            key=lambda det: abs((det.header.stamp.sec + det.header.stamp.nanosec * 1e-9) - img_time)
        )

        # If the time difference exceeds the threshold value, the processing is skipped
        tolerance = 0.1  # Tolerance threshold (seconds)
        det_time = best_det.header.stamp.sec + best_det.header.stamp.nanosec * 1e-9
        if abs(det_time - img_time) > tolerance:
            self.get_logger().warn("No detection within tolerance for the current image.")
            return

        # Convert image cv
        cv_img = self.bridge.imgmsg_to_cv2(img_msg)

        # Build a detection matrix
        dets = []
        for det in best_det.detections:
            x, y = det.bbox.center.position.x, det.bbox.center.position.y
            w, h = det.bbox.size.x, det.bbox.size.y
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2
            dets.append([x1, y1, x2, y2, det.score, det.class_id])

        if not dets:
            return

        dets_arr = Boxes(np.array(dets), (img_msg.height, img_msg.width))
        tracks = self.tracker.update(dets_arr, cv_img)

        # Generate tracking results
        out_msg = DetectionArray()
        out_msg.header = best_det.header  # Use the timestamp of the test result

        for tr in tracks:
            # tr: [x1, y1, x2, y2, score, class_id, track_id]
            bbox = Boxes(tr[:6], (img_msg.height, img_msg.width))
            idx = int(tr[6]) if len(tr) > 6 else None

            base_det = best_det.detections[idx] if idx is not None else Detection()
            new_det = Detection(
                class_id=base_det.class_id,
                class_name=base_det.class_name,
                score=base_det.score,
                bbox=base_det.bbox,
                mask=base_det.mask,
                keypoints=base_det.keypoints,
            )
            # Update the bounding box
            cx, cy, bw, bh = bbox.xywh[0]
            new_det.bbox.center.position.x = float(cx)
            new_det.bbox.center.position.y = float(cy)
            new_det.bbox.size.x = float(bw)
            new_det.bbox.size.y = float(bh)
            new_det.id = str(int(bbox.id)) if bbox.is_track else ''

            out_msg.detections.append(new_det)

        self._pub.publish(out_msg)


def main():
    rclpy.init()
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
