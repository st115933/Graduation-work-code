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
        super().__init__('visualization_Node')
        # state
        self.bridge = CvBridge()
        self.class_color = {}
        self.tolerance = 0.1  # seconds for matching
        self.det_buffer = collections.deque(maxlen=10)

        # pubs
        self.pub_img = self.create_publisher(Image, 'dbg_image', 10)
        self.pub_bb = self.create_publisher(MarkerArray, 'dgb_bb_markers', 10)
        self.pub_kp = self.create_publisher(MarkerArray, 'dgb_kp_markers', 10)

        # subs
        self.create_subscription(Image, 'image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(DetectionArray, 'detections', self.on_detections, 10)

    def on_detections(self, det_msg: DetectionArray):
        # enqueue latest detections
        self.det_buffer.append(det_msg)

    def on_image(self, img_msg: Image):
        if not self.det_buffer:
            return
        # find best-matching detection by timestamp
        img_time = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        best = min(self.det_buffer, key=lambda d: abs((d.header.stamp.sec + d.header.stamp.nanosec * 1e-9) - img_time))
        det_time = best.header.stamp.sec + best.header.stamp.nanosec * 1e-9
        if abs(det_time - img_time) > self.tolerance:
            return
        self.process_pair(img_msg, best)

    def process_pair(self, img_msg: Image, det_msg: DetectionArray):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg)
        annot = Annotator(cv_img)
        bb_markers = MarkerArray()
        kp_markers = MarkerArray()

        for det in det_msg.detections:
            cls = det.class_name
            if cls not in self.class_color:
                self.class_color[cls] = tuple(random.choices(range(256), k=3))
            col = self.class_color[cls]
            # draw box + label
            x, y, w, h = (det.bbox.center.position.x, det.bbox.center.position.y,
                          det.bbox.size.x, det.bbox.size.y)
            tl = (int(x - w/2), int(y - h/2))
            br = (int(x + w/2), int(y + h/2))
            annot.box_label((*tl, *br), f"{cls} {det.id}:{det.score:.3f}", col)
            # draw mask if present
            if det.mask and det.mask.data:
                pts = np.array([[p.x, p.y] for p in det.mask.data], np.int32)
                annot.masks([pts], [col])
            # draw keypoints if present
            if det.keypoints and det.keypoints.data:
                kpts = [(int(kp.point.x), int(kp.point.y), kp.id) for kp in det.keypoints.data]
                annot.keypoints(kpts, det.keypoints.data)

        # publish results
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