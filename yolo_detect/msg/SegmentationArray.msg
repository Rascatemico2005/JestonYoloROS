# 定义一帧图像中所有分割对象的数组
std_msgs/Header header  # 消息头，包含时间戳和坐标系信息
Segmentation[] segmentations  # 分割对象数组
