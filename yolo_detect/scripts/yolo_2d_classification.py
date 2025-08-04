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
# 关键改动：导入新的自定义分类消息类型
from yolo_detect.msg import Classification, ClassificationArray

class YoloClassifierNode:
    def __init__(self):
        """
        初始化YOLOv8分类ROS节点
        """
        rospy.init_node('yolo_classifier_node', anonymous=True)

        # --- 获取ROS参数 ---
        pt_model_path = rospy.get_param('~pt_model_path', '')
        self.engine_model_path = rospy.get_param('~engine_model_path', '')
        self.input_topic = rospy.get_param('~input_topic', '/camera/color/image_raw/compressed')
        # 改动：发布分类结果的话题
        self.classification_topic = rospy.get_param('~classification_topic', '/yolo_classify')
        self.annotated_image_topic = rospy.get_param('~annotated_image_topic', '/yolo_classify/annotated_image/compressed')
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)

        # --- 初始化FPS计算相关的变量 ---
        self.proc_prev_time = 0
        self.proc_fps = 0
        self.sub_prev_time = 0
        self.sub_fps = 0

        # --- 智能加载模型 (与检测节点逻辑相同) ---
        if not os.path.exists(self.engine_model_path):
            rospy.logwarn(f"TensorRT engine not found at {self.engine_model_path}. Exporting from .pt model...")
            if not os.path.exists(pt_model_path):
                rospy.logfatal(f".pt model not found at {pt_model_path}. Cannot create engine. Shutting down.")
                return
            
            rospy.loginfo(f"Loading .pt model from {pt_model_path} for export.")
            pt_model = YOLO(pt_model_path)
            # 导出为TensorRT引擎
            pt_model.export(format="engine", half=True, device=0) 
            rospy.loginfo(f"Export complete. Engine saved to {self.engine_model_path}")

        # 加载最终的TensorRT模型进行推理
        rospy.loginfo(f"Loading TensorRT engine from {self.engine_model_path}")
        try:
            # 确保你的模型是分类模型 (e.g., yolov8n-cls.pt)
            self.model = YOLO(self.engine_model_path, task='classify')
            rospy.loginfo("Ultralytics YOLO Classification model loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLO model. Error: {e}")
            return

        # --- 设置订阅者和发布者 ---
        self.image_sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        # 改动：发布者使用新的话题和消息类型
        self.classification_pub = rospy.Publisher(self.classification_topic, ClassificationArray, queue_size=10)
        self.annotated_image_pub = rospy.Publisher(self.annotated_image_topic, CompressedImage, queue_size=1)

        rospy.loginfo("YOLO Classifier node initialized and ready.")

    def image_callback(self, msg):
        """
        图像消息的回调函数
        """
        # --- 计算订阅FPS ---
        sub_current_time = time.time()
        if self.sub_prev_time > 0:
            time_diff = sub_current_time - self.sub_prev_time
            if time_diff > 0:
                self.sub_fps = (self.sub_fps * 0.9) + (1.0 / time_diff * 0.1)
        self.sub_prev_time = sub_current_time

        # --- 图像解码和推理 ---
        proc_start_time = time.time()
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error decoding compressed image: {e}")
            return
        
        # verbose=False可以减少终端输出
        results = self.model(cv_image, verbose=False)
        result = results[0]  # 获取第一张图的结果
        
        # --- 创建一个副本用于标注，避免修改原始图像 ---
        annotated_frame = cv_image.copy()

        # --- 计算处理FPS ---
        proc_end_time = time.time()
        time_diff = proc_end_time - proc_start_time
        if time_diff > 0:
            self.proc_fps = (self.proc_fps * 0.9) + (1.0 / time_diff * 0.1)
        
        # --- 在图像上绘制FPS和分类结果 ---
        # 1. 绘制FPS信息
        cv2.putText(annotated_frame, f"Sub FPS: {self.sub_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Proc FPS: {self.proc_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 2. 绘制最佳分类结果
        top1_index = result.probs.top1
        top1_confidence = result.probs.top1conf
        top1_class_name = self.model.names[top1_index]
        
        result_text = f"Class: {top1_class_name} ({top1_confidence:.2f})"
        cv2.putText(annotated_frame, result_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # --- 发布带标注的图像 ---
        try:
            annotated_image_msg = CompressedImage()
            annotated_image_msg.header = msg.header
            annotated_image_msg.format = "jpeg"
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                annotated_image_msg.data = buffer.tobytes()
                self.annotated_image_pub.publish(annotated_image_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing annotated image: {e}")

        # --- 发布分类结果数据 ---
        classification_array_msg = ClassificationArray()
        classification_array_msg.header = msg.header
        
        # 遍历所有类别概率，并发布超过阈值的结果
        for i, prob in enumerate(result.probs.data):
            if prob >= self.confidence_threshold:
                res = Classification()
                res.class_name = self.model.names[i]
                res.probability = float(prob)
                classification_array_msg.results.append(res)
        
        # 按概率降序排序
        classification_array_msg.results.sort(key=lambda x: x.probability, reverse=True)
        
        self.classification_pub.publish(classification_array_msg)

if __name__ == '__main__':
    try:
        node = YoloClassifierNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass