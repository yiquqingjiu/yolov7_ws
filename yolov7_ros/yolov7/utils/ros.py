import rospy
import torch
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, \
    ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D


def create_header(stamp=None):
    """创建消息头"""
    h = Header()
    h.stamp = stamp if stamp is not None else rospy.Time.now()
    return h


def create_detection_msg(img_msg: Image, detections: torch.Tensor,
                         stamp=None) -> Detection2DArray:
    """
    将 YOLO 检测结果封装为 ROS 的消息
    :param img_msg: 原始图像消息
    :param detections: 左上角、右下角坐标、置信度、类别ID
    :param stamp: 图像时间戳
    :returns: 封装好的 Detection2DArray 消息
    """
    detection_array_msg = Detection2DArray()

    # 消息头
    header = create_header(stamp)
    detection_array_msg.header = header
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        single_detection_msg = Detection2D()
        single_detection_msg.header = header

        # 原始图像
        single_detection_msg.source_img = img_msg

        # 检测框
        bbox = BoundingBox2D()
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        cx = int(round(x1 + w / 2))
        cy = int(round(y1 + h / 2))
        bbox.size_x = w
        bbox.size_y = h

        bbox.center = Pose2D()
        bbox.center.x = cx
        bbox.center.y = cy

        single_detection_msg.bbox = bbox

        # 类别 ID 与置信度
        obj_hyp = ObjectHypothesisWithPose()
        obj_hyp.id = int(cls)
        obj_hyp.score = conf
        single_detection_msg.results = [obj_hyp]

        detection_array_msg.detections.append(single_detection_msg)

    return detection_array_msg