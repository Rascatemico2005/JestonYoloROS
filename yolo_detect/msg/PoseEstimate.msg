# 单个检测到的姿态的完整数据
# 头部信息
int32 class_id
string class_name
float32 score # 边界框的置信度

# 边界框信息 (center_x, center_y, width, height)
float32 x
float32 y
float32 width
float32 height

# 关键点数组
yolo_detect/PoseKeypoint[] keypoints