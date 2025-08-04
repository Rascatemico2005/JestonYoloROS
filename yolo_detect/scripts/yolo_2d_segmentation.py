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

from sensor_msgs.msg import Image, CompressedImage, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from yolo_detect.msg import Segmentation, SegmentationArray

class YoloSegmentationCustomNode:
    def __init__(self):
        rospy.init_node('yolo_segmentation_custom_node', anonymous=True)

        self.bridge = CvBridge()

        # --- 获取ROS参数 ---
        pt_model_path = rospy.get_param('~pt_model_path', '')
        self.engine_model_path = rospy.get_param('~engine_model_path', '')
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        
        # 自定义分割消息数组更新输出话题
        self.input_topic = rospy.get_param('~input_topic', '/camera/color/image_raw/compressed')
        self.segmentation_topic = rospy.get_param('~segmentation_topic', '/yolo/segmentation_array')
        self.annotated_image_topic = rospy.get_param('~annotated_image_topic', '/yolo/annotated_image/compressed')

        # --- 初始化FPS计算  ---
        self.proc_prev_time = 0
        self.proc_fps = 0
        self.sub_prev_time = 0
        self.sub_fps = 0

        # --- 智能加载模型 ---
        if not os.path.exists(self.engine_model_path):
            rospy.logwarn(f"TensorRT engine not found at {self.engine_model_path}. Exporting from .pt model...")
            if not os.path.exists(pt_model_path):
                rospy.logfatal(f".pt model not found at {pt_model_path}. Cannot create engine. Shutting down.")
                return
            rospy.loginfo(f"Loading segmentation .pt model from {pt_model_path} for export.")
            pt_model = YOLO(pt_model_path)
            pt_model.export(format="engine", half=True, device=0) 
            rospy.loginfo(f"Export complete. Engine saved to {self.engine_model_path}")
        rospy.loginfo(f"Loading TensorRT segmentation engine from {self.engine_model_path}")
        try:
            self.model = YOLO(self.engine_model_path)
            rospy.loginfo("Ultralytics YOLO segmentation model loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLO model. Error: {e}")
            return

        # --- 设置订阅者和发布者 ---
        self.image_sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)        
        self.segmentation_pub = rospy.Publisher(self.segmentation_topic, SegmentationArray, queue_size=1)
        self.annotated_image_pub = rospy.Publisher(self.annotated_image_topic, CompressedImage, queue_size=1)

        rospy.loginfo("YOLO Custom Segmentation node initialized and ready.")

    def image_callback(self, msg):
        sub_current_time = time.time()
        if self.sub_prev_time > 0:
            time_diff = sub_current_time - self.sub_prev_time
            if time_diff > 0: # 避免除以零
                self.sub_fps = (self.sub_fps * 0.9) + (1.0 / time_diff * 0.1)
        self.sub_prev_time = sub_current_time

        # --- 预处理 ---
        proc_current_time = time.time()
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error decoding compressed image: {e}")
            return

        # --- 模型推理 ---
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
        
        try:
            annotated_image_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_frame, dst_format='jpg')
            annotated_image_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error publishing annotated image: {e}")

        # --- 构建并发布 SegmentationArray 消息 ---
        seg_array_msg = SegmentationArray()
        seg_array_msg.header = msg.header

        if result.masks is not None:
            # 遍历所有检测到的分割对象
            for i in range(len(result.masks.data)):
                seg_msg = Segmentation()
                box = result.boxes[i]
                mask_tensor = result.masks.data[i]

                # 1. 填充类别和置信度信息
                seg_msg.class_id = int(box.cls)
                seg_msg.class_name = self.model.names[seg_msg.class_id]
                seg_msg.score = float(box.conf)

                # 2. 填充边界框 (bbox)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = RegionOfInterest()
                roi.x_offset = x1
                roi.y_offset = y1
                roi.height = y2 - y1
                roi.width = x2 - x1
                seg_msg.bbox = roi

                # 3. 填充该对象的独立掩码 (mask)
                try:
                    # 将单个对象的mask张量转为numpy数组
                    mask_cpu = mask_tensor.cpu().numpy().astype(np.uint8)
                    # 将mask缩放到原图尺寸以保证对齐
                    mask_resized = cv2.resize(mask_cpu, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # 从完整尺寸的mask中裁剪出bbox对应的区域
                    cropped_mask = mask_resized[y1:y2, x1:x2]
                    # 将裁剪后的二值掩码转换为ROS Image消息 (0=背景, 255=前景)
                    mask_ros_msg = self.bridge.cv2_to_imgmsg(cropped_mask * 255, encoding="mono8")
                    seg_msg.mask = mask_ros_msg
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error creating individual mask: {e}")
                    continue # 跳过这个有问题的对象
                # 4. 将填充好的单个分割对象添加到数组中
                seg_array_msg.segmentations.append(seg_msg)
        # 即使没有检测到任何物体，也发布一个空数组
        self.segmentation_pub.publish(seg_array_msg)

if __name__ == '__main__':
    try:
        node = YoloSegmentationCustomNode()
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("YOLO custom segmentation node shutting down.")
