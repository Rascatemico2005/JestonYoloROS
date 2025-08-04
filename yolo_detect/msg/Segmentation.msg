# 定义单个分割对象的结果

# 类别信息
int32 class_id
string class_name

# 置信度得分
float32 score

# 该对象的边界框 (Region of Interest)
# 方便快速定位，即使我们有精确的掩码
sensor_msgs/RegionOfInterest bbox

# 该对象的分割掩码
# 这是一个小尺寸的、仅包含该对象的二值图像
# 图像尺寸与上面的 bbox 对应
sensor_msgs/Image mask