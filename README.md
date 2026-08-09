# 视觉感知工作空间

基于ROS + YOLOv7的锥桶检测与视觉感知工作空间。

## 目录结构

```
src/
├── CMakeLists.txt          # catkin 工作区顶层配置
├── .gitignore              # 排除 __pycache__、build、*.bag 等
├── .gitattributes          # LFS 配置
├── camera/                 # 相机相关
│   ├── camera_control_msgs # 相机控制消息
│   ├── dragandbot_common   # Drag&Bot 公共消息
│   └── pylon_camera        # Pylon 相机驱动
└── yolov7_ros/             # YOLOv7 检测包
    ├── detect_ros.py       # 检测节点
    ├── launch/ msg/ rviz/  # 启动/消息/RViz配置
    ├── weights/best.pt     # 模型权重（71MB，LFS）
    └── yolov7/             # YOLOv7 源码
```

## 模块说明

### camera/ — 相机相关

| 包 | 说明 |
| --- | --- |
| `camera_control_msgs` | 相机控制相关的自定义消息（action / msg / srv） |
| `dragandbot_common` | Drag&Bot 机械臂公共消息定义 |
| `pylon_camera` | Basler Pylon 相机 ROS 驱动，支持 GigE / USB 相机 |

### yolov7_ros/ — 目标检测包

| 文件/目录 | 说明 |
| --- | --- |
| `detect_ros.py` | YOLOv7 检测 ROS 节点，订阅图像话题，发布检测结果 |
| `launch/` | 启动文件（`yolov7.launch`） |
| `msg/` | 自定义消息（`ConeArray.msg` / `ConeInfo.msg`） |
| `rviz/` | RViz 可视化配置 |
| `weights/best.pt` | 训练好的模型权重（仅用于yolov7检测锥桶） |
| `yolov7/` | YOLOv7 部分模型源码（models / utils / visualizer） |

### 环境依赖

- Ubuntu 20.04 + ROS Noetic（或兼容的 ROS 版本）
- Python 3.8+
- PyTorch、OpenCV、numpy
- Basler Pylon SDK（使用 `pylon_camera` 时需要）

# 启动 Pylon 相机
roslaunch pylon_camera pylon_camera_node.launch

# 启动 YOLOv7 锥桶检测
roslaunch yolov7_ros yolov7.launch（包含静态TF发布和rviz，定位建图的建议将这两部分注释掉）
```

### 主要话题

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/pylon_camera_node/image_raw` | `sensor_msgs/Image` | 相机原始图像 |
| `/yolov7/yolov7/all_cones` | `yolov7_ros/ConeArray` | 检测到的锥桶（ID、位置、颜色、置信度） |
| `/yolov7/yolov7/all_cones_marker` | `visualization_msgs/MarkerArray` | 锥桶 RViz 可视化 Marker |
| `/yolov7/yolov7/visualization` | `sensor_msgs/Image` | 可视化标注后的检测图像 |
| `/yolov7/yolov7` | `vision_msgs/Detection2DArray` | 通用 2D 检测结果（兼容 vision_msgs） |

## 说明

- 模型权重best.pt体积较大，使用Git LFS管理，克隆后需执行git lfs pull获取完整文件
