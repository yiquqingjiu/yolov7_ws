# ============================================================
# yolov7_ros 包构建配置（catkin）
# ============================================================
cmake_minimum_required(VERSION 3.0.2)
project(yolov7_ros)

# 查找 catkin 及所需依赖包
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs      # 几何消息（Point 等）
  roscpp             # ROS C++ 支持
  rospy              # ROS Python 支持
  sensor_msgs        # 传感器消息（Image 等）
  std_msgs           # 标准消息（Header 等）
  vision_msgs        # 视觉消息（Detection2DArray 等）
  message_generation # 消息生成工具
)

################################################
## 声明 ROS 自定义消息                        ##
################################################
add_message_files(
  FILES
  ConeArray.msg   # 锥桶数组消息
  ConeInfo.msg    # 单个锥桶信息（ID、位置、颜色、置信度）
)

# 生成消息（声明依赖的消息类型包）
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
# 包配置：声明对外依赖
catkin_package(
    CATKIN_DEPENDS message_runtime
)

###########
## 编译  ##
###########
# 指定头文件搜索路径
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## 安装   ##
#############
# 将 Python 节点脚本安装为可执行文件（供 rosrun / roslaunch 调用）
catkin_install_python(PROGRAMS
  detect_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装 launch 启动文件（供 roslaunch yolov7_ros yolov7.launch 调用）
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
