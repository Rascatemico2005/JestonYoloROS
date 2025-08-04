#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/usr/lib/python3.8/dist-packages/')

import rospy
import cv2
import numpy as np
import os
import time

from ultralytics import YOLO

from sensor_msgs.msg import CompressedImage
from yolo_detect.msg import Detection, DetectionArray

class YoloUltralyticsNode:
    def __init__(self):
        rospy.init_node('yolo_ultralytics_node', anonymous=True)

        # --- 获取ROS参数 ---
        pt_model_path = rospy.get_param('~pt_model_path', '')
        self.engine_model_path = rospy.get_param('~engine_model_path', '')
        self.input_topic = rospy.get_param('~input_topic', '/camera/color/image_raw/compressed')
        self.detection_topic = rospy.get_param('~detection_topic', '/yolo_detected')
        self.annotated_image_topic = rospy.get_param('~annotated_image_topic', '/yolo_detect/camera/color/compressed')
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)

        # --- 初始化FPS计算相关的变量 ---
        # 用于计算处理速度的FPS
        self.proc_prev_time = 0
        self.proc_fps = 0
        
        # 【订阅FPS新增】用于计算图像订阅频率的FPS
        self.sub_prev_time = 0
        self.sub_fps = 0

        # --- 智能加载模型 ---
        if not os.path.exists(self.engine_model_path):
            rospy.logwarn(f"TensorRT engine not found at {self.engine_model_path}. Exporting from .pt model...")
            if not os.path.exists(pt_model_path):
                rospy.logfatal(f".pt model not found at {pt_model_path}. Cannot create engine. Shutting down.")
                return
            
            rospy.loginfo(f"Loading .pt model from {pt_model_path} for export.")
            pt_model = YOLO(pt_model_path)
            pt_model.export(format="engine", half=True, device=0) 
            rospy.loginfo(f"Export complete. Engine saved to {self.engine_model_path}")

        # 加载最终的TensorRT模型进行推理
        rospy.loginfo(f"Loading TensorRT engine from {self.engine_model_path}")
        try:
            self.model = YOLO(self.engine_model_path)
            rospy.loginfo("Ultralytics YOLO model loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLO model. Error: {e}")
            return

        # --- 设置订阅者和发布者 ---
        self.image_sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.detection_pub = rospy.Publisher(self.detection_topic, DetectionArray, queue_size=10)
        self.annotated_image_pub = rospy.Publisher(self.annotated_image_topic, CompressedImage, queue_size=1)

        rospy.loginfo("YOLO Ultralytics node initialized and ready.")

    def image_callback(self, msg):
        # 【订阅FPS新增】计算订阅FPS
        sub_current_time = time.time()
        if self.sub_prev_time > 0:
            time_diff = sub_current_time - self.sub_prev_time
            if time_diff > 0: # 避免除以零
                self.sub_fps = (self.sub_fps * 0.9) + (1.0 / time_diff * 0.1)
        self.sub_prev_time = sub_current_time

        # --- 开始计时处理 ---
        proc_current_time = time.time()
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error decoding compressed image: {e}")
            return
            
        results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        
        # --- 结束计时并计算处理FPS ---
        if self.proc_prev_time > 0:
            time_diff = proc_current_time - self.proc_prev_time
            if time_diff > 0: # 避免除以零
                self.proc_fps = (self.proc_fps * 0.9) + (1.0 / time_diff * 0.1)
        self.proc_prev_time = proc_current_time

        # 在图像上绘制两种FPS信息
        sub_fps_text = f"Sub FPS: {self.sub_fps:.1f}"
        proc_fps_text = f"Proc FPS: {self.proc_fps:.1f}"
        cv2.putText(annotated_frame, sub_fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, proc_fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- 发布带标注的图像 ---
        try:
            annotated_image_msg = CompressedImage()
            annotated_image_msg.header = msg.header
            annotated_image_msg.format = "jpeg"
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                rospy.logerr("Failed to encode annotated image.")
                return
            annotated_image_msg.data = buffer.tobytes()
            self.annotated_image_pub.publish(annotated_image_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing annotated image: {e}")

        # --- 发布检测结果数据 ---
        detection_array_msg = DetectionArray()
        detection_array_msg.header = msg.header
        
        for box in result.boxes:
            det = Detection()
            class_id = int(box.cls)
            det.class_id = class_id
            det.class_name = self.model.names[class_id]
            det.score = float(box.conf)
            x, y, w, h = box.xywh[0]
            det.x = float(x)
            det.y = float(y)
            det.width = float(w)
            det.height = float(h)
            detection_array_msg.detections.append(det)
            
        self.detection_pub.publish(detection_array_msg)

if __name__ == '__main__':
    try:
        node = YoloUltralyticsNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass