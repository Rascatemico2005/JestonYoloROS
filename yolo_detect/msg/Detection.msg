# 描述一个检测到的物体

# 物体类别 ID
int64 class_id

# 物体类别名称
string class_name

# 置信度
float64 score

# 边界框 (bounding box) 像素坐标
# (x, y) 是边界框左上角的点
float64 x
float64 y
float64 width
float64 height