#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from yolov8_msgs.msg import DetectionArray
import serial
import threading
import json
import time
import subprocess

# 串口配置（写死）
_PORT = "/dev/ttyUSB0"
_BAUDRATE = 115200

# 固定的速度和加速度
_FIXED_SPD = 0    # 0 = 最大速度
_FIXED_ACC = 10   # 10 * 100 步/s²

# 图像分辨率（示例值，按实际修改）
IMG_W = 800.0  # 根据代码中的区域判断调整为实际值
IMG_H = 600.0  # 根据代码中的区域判断调整为实际值

# C程序路径和参数（需确保a.out会自动完成任务后退出）
AOUT_PATH = "./a.out"  # 确保a.out在当前路径或提供完整路径
MEASURE_COUNT = 5      # 测量次数（需与C程序逻辑匹配）
STEP_DELAY_MSEC = 1000 # 每次测量间隔(毫秒)

# 区域 1 的像素范围（示例值，按实际需求修改）
X_MIN1, X_MAX1 = 0.0, IMG_W / 2   # 左半区
Y_MIN1, Y_MAX1 = 0.0, IMG_H        # 整个高度

# 区域对应的关节角度（弧度制），四个关节都要填写
REGION1_ANGLES = {
    "base":  0.015339808,
    "shoulder": -1.543184673,
    "elbow":    2.684466379,
    "hand":     1.744136156
}
REGION2_ANGLES = {
    "base":  -0.2,
    "shoulder": -0.5,
    "elbow":     1.2,
    "hand":      1.5
}

def initialize_serial(port=_PORT, baudrate=_BAUDRATE):
    ser = serial.Serial(port, baudrate=baudrate, timeout=1)
    ser.setRTS(False)
    ser.setDTR(False)
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser

class CenterSubscriber(Node):
    def __init__(self):
        super().__init__('center_subscriber')

        # 串口初始化 & 读线程
        self.ser = initialize_serial()
        threading.Thread(target=self._read_serial, daemon=True).start()
        self.get_logger().info(f"Serial initialized on {_PORT} @ {_BAUDRATE}")
        
        # 用于存储C程序进程
        self.c_process = None

        # 订阅 DetectionArray
        self.create_subscription(
            DetectionArray,
            '/max_square_center',
            self.listener_callback,
            10)

    def _read_serial(self):
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                self.get_logger().info(f"[RX] {line}")

    def move_joints(self, angles: dict):
        """
        发送 CMD_JOINTS_RAD_CTRL 命令
        angles: 包含 base, shoulder, elbow, hand 四个键
        """
        cmd = {
            "T": 102,
            "base":     angles["base"],
            "shoulder": angles["shoulder"],
            "elbow":    angles["elbow"],
            "hand":     angles["hand"],
            "spd":      _FIXED_SPD,
            "acc":      _FIXED_ACC
        }
        js = json.dumps(cmd)
        self.ser.write(js.encode('utf-8') + b'\n')
        self.get_logger().info(f"[TX] {js}")
    
    def start_and_wait_c_program(self):
        """启动C程序并等待其自动终止"""
        try:
            # 构建命令行参数
            cmd = [AOUT_PATH, str(MEASURE_COUNT), str(STEP_DELAY_MSEC)]
            self.c_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            self.get_logger().info(f"C程序已启动，PID: {self.c_process.pid}")

            self.c_process.wait()
            
            if self.c_process.returncode == 0:
                self.get_logger().info("C程序已正常完成任务")
            else:
                self.get_logger().warning(f"C程序异常终止，返回码: {self.c_process.returncode}")
        except Exception as e:
            self.get_logger().error(f"C程序启动失败: {str(e)}")
        finally:
            self.c_process = None

    def listener_callback(self, msg: DetectionArray):
        if not msg.detections:
            self.get_logger().info('No detections.')
            return

        det = msg.detections[0]
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        self.get_logger().info(f'center_x: {cx:.1f}, center_y: {cy:.1f}')

        # 区域1：左上
        if 493 <= cx <= 613 and 181 <= cy <= 299:
            self.get_logger().info("In REGION 1-> moving to REGION1_ANGLES")
            
            # 执行tj1
            tj1 = {
                "base":  -0.742446701,
                "shoulder": -1.282407939,
                "elbow":    2.37460226,
                "hand":     1.974233274
            }
            self.move_joints(tj1)
            time.sleep(1)
            
            # 执行tj2
            tj2 = {
                "base":  -0.740912721,
                "shoulder": -1.006291397,
                "elbow":    1.81009733,
                "hand":     2.172116796
            }
            self.move_joints(tj2)
            time.sleep(1)  # 等待到位
            
            # 启动并等待C程序
            self.start_and_wait_c_program()
            
            # 执行tj3
            tj3 = {
                "base":  -0.742446701,
                "shoulder": -1.282407939,
                "elbow":    2.37460226,
                "hand":     1.974233274
            }
            self.move_joints(tj3)
            time.sleep(1)
            
            # 执行tj4
            tj4 = {
                "base":  0.010737866,
                "shoulder": -1.555456519,
                "elbow":    3.014272248,
                "hand":     1.520174961
            }
            self.move_joints(tj4)
            time.sleep(15)

        # 区域2：右上
        elif 613 <= cx <= 732 and 181 <= cy <= 299:
            self.get_logger().info("In REGION 2-> moving to REGION1_ANGLES")
            
            # 执行tj1
            tj1 = {
                "base":  0.882038953,
                "shoulder": -1.492563307,
                "elbow":    2.241145931,
                "hand":     2.300971182
            }
            self.move_joints(tj1)
            time.sleep(1)
            
            # 执行tj2
            tj2 = {
                "base":  0.932660319,
                "shoulder": -1.184233168,
                "elbow":    1.727262367,
                "hand":     2.495786742
            }
            self.move_joints(tj2)
            time.sleep(1)  # 等待到位
            
            # 启动并等待C程序
            self.start_and_wait_c_program()
            
            # 执行tj3
            tj3 = {
                "base":  0.882038953,
                "shoulder": -1.492563307,
                "elbow":    2.241145931,
                "hand":     2.300971182
            }
            self.move_joints(tj3)
            time.sleep(1)
            
            # 执行tj4
            tj4 = {
                "base":  0.010737866,
                "shoulder": -1.555456519,
                "elbow":    3.014272248,
                "hand":     1.520174961
            }
            self.move_joints(tj4)
            time.sleep(15)

        # 区域3：左中
        elif 493 <= cx <= 613 and 299 <= cy <= 417:
            self.get_logger().info("In REGION 3-> moving to REGION1_ANGLES")
            
            # 执行tj1
            tj1 = {
                "base":  0.44792239,
                "shoulder": -0.561436968,
                "elbow":    2.153709026,
                "hand":     1.474155537
            }
            self.move_joints(tj1)
            time.sleep(1)
            
            # 执行tj2
            tj2 = {
                "base":  0.434116563,
                "shoulder": -0.516951526,
                "elbow":    1.67357304,
                "hand":     1.831573061
            }
            self.move_joints(tj2)
            time.sleep(1)  # 等待到位
            
            # 启动并等待C程序
            self.start_and_wait_c_program()
            
            # 执行tj3
            tj3 = {
                "base":  0.44792239,
                "shoulder": -0.561436968,
                "elbow":    2.153709026,
                "hand":     1.474155537
            }
            self.move_joints(tj3)
            time.sleep(1)
            
            # 执行tj4
            tj4 = {
                "base":  0.010737866,
                "shoulder": -1.555456519,
                "elbow":    3.014272248,
                "hand":     1.520174961
            }
            self.move_joints(tj4)
            time.sleep(15)

        # 区域4：右中
        elif 613 <= cx <= 732 and 299 <= cy <= 417:
            self.get_logger().info("In REGION 4-> moving to REGION1_ANGLES")
            
            # 执行tj1
            tj1 = {
                "base":  -0.688757374,
                "shoulder": -0.411106851,
                "elbow":    2.116893487,
                "hand":     1.461883691
            }
            self.move_joints(tj1)
            time.sleep(1)
            
            # 执行tj2
            tj2 = {
                "base":  -0.694893297,
                "shoulder": -0.377359274,
                "elbow":    1.584602154,
                "hand":     1.767145868
            }
            self.move_joints(tj2)
            time.sleep(1)  # 等待到位
            
            # 启动并等待C程序
            self.start_and_wait_c_program()
            
            # 执行tj3
            tj3 = {
                "base":  -0.688757374,
                "shoulder": -0.411106851,
                "elbow":    2.116893487,
                "hand":     1.461883691
            }
            self.move_joints(tj3)
            time.sleep(1)
            
            # 执行tj4
            tj4 = {
                "base":  0.010737866,
                "shoulder": -1.555456519,
                "elbow":    3.014272248,
                "hand":     1.520174961
            }
            self.move_joints(tj4)
            time.sleep(15)

        # 区域5：左下
        elif 493 <= cx <= 613 and 417 <= cy <= 535:
            self.get_logger().info("In REGION 5-> moving to REGION1_ANGLES")
            
            # 执行tj1
            tj1 = {
                "base":  0.220893233,
                "shoulder": -0.21322333,
                "elbow":    1.922077927,
                "hand":     1.491029326
            }
            self.move_joints(tj1)
            time.sleep(1)
            
            # 执行tj2
            tj2 = {
                "base":  0.208621387,
                "shoulder": -0.200951483,
                "elbow":    1.434272037,
                "hand":     1.808563349
            }
            self.move_joints(tj2)
            time.sleep(1)  # 等待到位
            
            # 启动并等待C程序
            self.start_and_wait_c_program()
            
            # 执行tj3
            tj3 = {
                "base":  0.220893233,
                "shoulder": -0.21322333,
                "elbow":    1.922077927,
                "hand":     1.491029326
            }
            self.move_joints(tj3)
            time.sleep(1)
            
            # 执行tj4
            tj4 = {
                "base":  0.010737866,
                "shoulder": -1.555456519,
                "elbow":    3.014272248,
                "hand":     1.520174961
            }
            self.move_joints(tj4)
            time.sleep(15)

        # 区域6：右下
        elif 613 <= cx <= 732 and 417 <= cy <= 535:
            self.get_logger().info("In REGION 6 -> moving to REGION1_ANGLES")
            
            # 执行tj1
            tj1 = {
                "base":  -0.423378697,
                "shoulder": -0.049087385,
                "elbow":    1.708854598,
                "hand":     1.503301172
            }
            self.move_joints(tj1)
            time.sleep(1)
            
            # 执行tj2
            tj2 = {
                "base":  -0.435650544,
                "shoulder": -0.032213597,
                "elbow":    1.270136092,
                "hand":     1.759475964
            }
            self.move_joints(tj2)
            time.sleep(1)  # 等待到位
            
            # 启动并等待C程序
            self.start_and_wait_c_program()
            
            # 执行tj3
            tj3 = {
                "base":  -0.423378697,
                "shoulder": -0.049087385,
                "elbow":    1.708854598,
                "hand":     1.503301172
            }
            self.move_joints(tj3)
            time.sleep(1)
            
            # 执行tj4
            tj4 = {
                "base":  0.010737866,
                "shoulder": -1.555456519,
                "elbow":    3.014272248,
                "hand":     1.520174961
            }
            self.move_joints(tj4)
            time.sleep(15)

def main(args=None):
    rclpy.init(args=args)
    node = CenterSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down...")
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
