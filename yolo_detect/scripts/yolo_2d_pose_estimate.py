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

# --- 消息类型导入 ---
from sensor_msgs.msg import CompressedImage
# 导入我们新创建的自定义消息
from yolo_detect.msg import PoseKeypoint, PoseEstimate, PoseArray

class YoloPoseNode:
    def __init__(self):
        """
        初始化YOLO姿态估计ROS节点
        """
        rospy.init_node('yolo_pose_node', anonymous=True)

        # --- 获取ROS参数 ---
        # 注意：模型必须是姿态估计模型，例如 yolo11l-pose.pt
        pt_model_path = rospy.get_param('~pt_model_path', 'yolov11l-pose.pt')
        self.engine_model_path = rospy.get_param('~engine_model_path', 'yolov11l-pose.engine')
        self.input_topic = rospy.get_param('~input_topic', '/camera/color/image_raw/compressed')
        self.pose_topic = rospy.get_param('~pose_topic', '/yolo_pose/poses')
        self.annotated_image_topic = rospy.get_param('~annotated_image_topic', '/yolo_pose/image/compressed')
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)

        # --- 初始化FPS计算相关的变量 ---
        self.proc_prev_time = 0
        self.proc_fps = 0
        self.sub_prev_time = 0
        self.sub_fps = 0

        # --- 智能加载模型 (与原脚本逻辑相同) ---
        if not os.path.exists(self.engine_model_path):
            rospy.logwarn(f"TensorRT engine not found at {self.engine_model_path}. Exporting from .pt model...")
            if not os.path.exists(pt_model_path):
                rospy.logfatal(f".pt model not found at {pt_model_path}. Cannot create engine. Shutting down.")
                rospy.signal_shutdown("Model file not found")
                return
            
            rospy.loginfo(f"Loading .pt model from {pt_model_path} for export.")
            pt_model = YOLO(pt_model_path)
            # 导出为TensorRT engine
            pt_model.export(format="engine", half=True, device=0) 
            rospy.loginfo(f"Export complete. Engine saved to {self.engine_model_path}")

        # 加载最终的TensorRT模型进行推理
        rospy.loginfo(f"Loading TensorRT engine from {self.engine_model_path}")
        try:
            self.model = YOLO(self.engine_model_path, task='pose') # 指定任务为 'pose'
            rospy.loginfo("Ultralytics YOLO Pose model loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLO model. Error: {e}")
            rospy.signal_shutdown("Failed to load model")
            return

        # --- 设置订阅者和发布者 ---
        self.image_sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        # 发布PoseArray消息
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseArray, queue_size=10)
        self.annotated_image_pub = rospy.Publisher(self.annotated_image_topic, CompressedImage, queue_size=1)

        rospy.loginfo("YOLO Pose Estimation node initialized and ready. 🚀")

    def image_callback(self, msg):
        """
        处理传入图像的回调函数
        """
        # --- 计算订阅FPS (与原脚本逻辑相同) ---
        sub_current_time = time.time()
        if self.sub_prev_time > 0:
            time_diff = sub_current_time - self.sub_prev_time
            if time_diff > 0:
                self.sub_fps = (self.sub_fps * 0.9) + (1.0 / time_diff * 0.1)
        self.sub_prev_time = sub_current_time

        # --- 图像解码与模型推理 ---
        proc_start_time = time.time()
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error decoding compressed image: {e}")
            return
            
        # verbose=False可以减少不必要的控制台输出
        results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)
        result = results[0]
        annotated_frame = result.plot() # result.plot() 会自动绘制边界框和关键点

        # --- 计算处理FPS (与原脚本逻辑相同) ---
        proc_end_time = time.time()
        time_diff = proc_end_time - proc_start_time
        if time_diff > 0:
            current_proc_fps = 1.0 / time_diff
            self.proc_fps = (self.proc_fps * 0.9) + (current_proc_fps * 0.1)

        # 在图像上绘制FPS信息
        sub_fps_text = f"Sub FPS: {self.sub_fps:.1f}"
        proc_fps_text = f"Proc FPS: {self.proc_fps:.1f}"
        cv2.putText(annotated_frame, sub_fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, proc_fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- 发布带标注的图像 (与原脚本逻辑相同) ---
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

        # --- 发布姿态估计结果数据 ---
        pose_array_msg = PoseArray()
        pose_array_msg.header = msg.header
        
        # 获取检测框和关键点
        boxes = result.boxes
        keypoints = result.keypoints

        # 遍历每个检测到的实例
        for i in range(len(boxes)):
            pose_msg = PoseEstimate()
            box = boxes[i]
            
            # 1. 填充边界框和类别信息
            class_id = int(box.cls)
            pose_msg.class_id = class_id
            pose_msg.class_name = self.model.names[class_id]
            pose_msg.score = float(box.conf)
            x, y, w, h = box.xywh[0]
            pose_msg.x = float(x)
            pose_msg.y = float(y)
            pose_msg.width = float(w)
            pose_msg.height = float(h)
            
            # 2. 填充关键点信息
            kpt = keypoints[i]
            points_xy = kpt.xy[0]       # 获取所有关键点的xy坐标 (Tensor)
            points_conf = kpt.conf[0]   # 获取所有关键点的置信度 (Tensor)

            for j in range(len(points_xy)):
                keypoint_msg = PoseKeypoint()
                keypoint_msg.x = float(points_xy[j][0])
                keypoint_msg.y = float(points_xy[j][1])
                keypoint_msg.score = float(points_conf[j])
                pose_msg.keypoints.append(keypoint_msg)

            pose_array_msg.poses.append(pose_msg)
            
        self.pose_pub.publish(pose_array_msg)

if __name__ == '__main__':
    try:
        node = YoloPoseNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except rospy.ROSInitException as e:
        rospy.logerr(f"Failed to initialize ROS node: {e}")