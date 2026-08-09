import sys
import os
import math
from typing import Tuple, Union, List

import torch
import cv2
import numpy as np
import rospy

try:
    import rospkg
    _pkg_dir = rospkg.RosPack().get_path("yolov7_ros")
    _yolo_dir = os.path.join(_pkg_dir, "yolov7")
except Exception:
    _yolo_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "yolov7", "src")
if _yolo_dir not in sys.path:
    sys.path.append(_yolo_dir)

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.ros import create_detection_msg
from visualizer import draw_detections

# 导入消息
from yolov7_ros.msg import ConeArray, ConeInfo
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA 
import time


def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip("\n")
            classes.append(line)
    return classes


def rescale(ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
    xscale = target_shape[1] / ori_shape[1]
    yscale = target_shape[0] / ori_shape[0]
    boxes[:, [0, 2]] *= xscale
    boxes[:, [1, 3]] *= yscale
    return boxes


class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        img = img.unsqueeze(0)
        pred = self.model(img)[0]
        detections = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)
        return detections[0] if (detections and len(detections[0]) > 0) else None


class ConeTracker:
    def __init__(self, max_match_dist: float = 1.0, max_age: float = 1.0):
        self.max_match_dist = max_match_dist 
        self.max_age = max_age                
        self.tracks = {}
        self.next_id = 0

    def _now(self, stamp) -> float:
        if hasattr(stamp, "to_sec"):
            return stamp.to_sec()
        return time.time()

    def update(self, cones: List[dict], stamp) -> List[dict]:
        t = self._now(stamp)

        for cone in cones:
            best_id, best_dist = None, self.max_match_dist
            for tid, tr in self.tracks.items():
                if tr["color"] != cone["color"]:
                    continue
                d = float(np.hypot(tr["x"] - cone["x"], tr["y"] - cone["y"]))
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                # 关联到已有轨迹，更新位置
                cone["id"] = best_id
                self.tracks[best_id]["x"] = cone["x"]
                self.tracks[best_id]["y"] = cone["y"]
                self.tracks[best_id]["last_time"] = t
            else:
                # 新目标，分配新 ID
                cone["id"] = self.next_id
                self.tracks[self.next_id] = {
                    "x": cone["x"], "y": cone["y"],
                    "color": cone["color"], "last_time": t}
                self.next_id += 1

        # 删除轨迹（目标丢失）
        expired = [tid for tid, tr in self.tracks.items()
                   if t - tr["last_time"] > self.max_age]
        for tid in expired:
            del self.tracks[tid]

        return cones


class Yolov7Publisher:
    def __init__(self, img_topic: str, weights: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45, pub_topic: str = "yolov7_detections",
                 device: str = "cuda", img_size: Tuple[int, int] = (320, 320),
                 queue_size: int = 1, visualize: bool = False,
                 class_labels: Union[List, None] = None):
        self.img_size = img_size
        self.device = device
        self.class_labels = class_labels
        self.last_time = time.time()
        self.bridge = CvBridge()

        def _param_to_list(name, default):
            """读取参数"""
            import ast
            val = rospy.get_param(name, default)
            if isinstance(val, str):
                return ast.literal_eval(val)
            return list(val)

        # 相机内参矩阵 
        fx = rospy.get_param("~camera/fx", 1378.97951)
        fy = rospy.get_param("~camera/fy", 1378.27001)
        cx = rospy.get_param("~camera/cx", 984.05127)
        cy = rospy.get_param("~camera/cy", 611.00651)
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        # 畸变系数
        self.D = np.array(_param_to_list(
            "~camera/distortion", [-0.097847, 0.228913, -0.000396, 0.001502, 0.0]),
            dtype=np.float32)
        # 相机在车辆坐标系下的平移
        t = _param_to_list("~camera/translation", [0.3, 0.0, 0.5])
        self.T_cam_in_vehicle = np.array(t, dtype=np.float32)
        # 车辆坐标系到相机坐标系
        r = _param_to_list(
            "~camera/rotation",
            [[0, -1, 0], [0, 0, 1], [1, 0, 0]])
        self.R_vehicle_to_cam = np.array(r, dtype=np.float32)
        self.R_cam_to_vehicle_base = self.R_vehicle_to_cam.T

        # 相机到车体
        self.R_cam_to_vehicle = self.R_cam_to_vehicle_base

        # 锥桶参数
        self.cone_class_names = list(class_labels) if (class_labels and len(class_labels) > 0) \
            else ["red_cone", "blue_cone", "yellow_cone"]
        self.cone_class_ids = list(range(len(self.cone_class_names)))
        self.cone_dist_filter = (0.5, 30.0)

        # 目标跟踪器
        self.tracker = ConeTracker(max_match_dist=1.0, max_age=1.0)

        # 自定义消息发布器
        self.cone_array_pub = rospy.Publisher(
            f"{pub_topic}/all_cones",
            ConeArray,
            queue_size=queue_size
        )

        self.cone_marker_pub = rospy.Publisher(
            f"{pub_topic}/all_cones_marker",  
            MarkerArray,
            queue_size=queue_size
        )

        self.det_pub = rospy.Publisher(pub_topic, Detection2DArray, queue_size=queue_size)
        self.vis_topic = f"{pub_topic}/visualization"
        self.vis_pub = rospy.Publisher(self.vis_topic, Image, queue_size=queue_size) if visualize else None

        self.model = YoloV7(weights=weights, conf_thresh=conf_thresh, iou_thresh=iou_thresh, device=device)
        self.img_sub = rospy.Subscriber(img_topic, Image, self.process_img_callback)
        rospy.loginfo("YOLOv7节点初始化完成")


    def create_cone_markers(self, cone_array_msg: ConeArray) -> MarkerArray:
        marker_array = MarkerArray()
        marker_id = 0 

        # 遍历每个锥桶
        for cone in cone_array_msg.cones:
            x = cone.position.x 
            y = cone.position.y 
            z = cone.position.z
            color = cone.color

            # 创建锥桶位置
            pos_marker = Marker()
            pos_marker.header = cone_array_msg.header 
            pos_marker.ns = "cone_position"  
            pos_marker.id = marker_id        
            pos_marker.type = Marker.CUBE    
            pos_marker.action = Marker.ADD   

            pos_marker.pose.position.x = x
            pos_marker.pose.position.y = y
            pos_marker.pose.position.z = z + 0.15  
            pos_marker.pose.orientation.w = 1.0 

            # 锥桶尺寸
            pos_marker.scale.x = 0.2 
            pos_marker.scale.y = 0.2
            pos_marker.scale.z = 0.3  

            # 颜色映射
            if color == "red_cone":
                pos_marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  
            elif color == "blue_cone":
                pos_marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)  
            elif color == "yellow_cone":
                pos_marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)  
            else:
                pos_marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.8)  

            # 生命周期：0.8
            pos_marker.lifetime = rospy.Duration(0.8)
            # 将方体Marker加入数组
            marker_array.markers.append(pos_marker)
            marker_id += 1  

        return marker_array


    def process_img_callback(self, img_msg: Image):
        # 图像转换
        try:
            img_cv = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"图像转换失败：{str(e)}")
            return
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        h_orig, w_orig, _ = img_cv.shape

        # YOLO推理
        img_resized = cv2.resize(img_cv, self.img_size)
        img_transposed = img_resized.transpose((2, 0, 1))[::-1]
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_transposed)).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        detections = self.model.inference(img_tensor)

        # 初始化自定义消息
        cone_array_msg = ConeArray()
        cone_array_msg.header = img_msg.header
        cone_array_msg.header.frame_id = "base_link"

        if detections is None:
            self.cone_array_pub.publish(cone_array_msg)
            if self.vis_pub:
                self.vis_pub.publish(self.bridge.cv2_to_imgmsg(img_cv, encoding="bgr8"))
            return

        # 检测框缩放
        detections[:, :4] = rescale(self.img_size, detections[:, :4], (h_orig, w_orig))
        detections[:, :4] = detections[:, :4].round()

        #计算锥桶车辆坐标系位置
        raw_cones = [] 
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det.tolist()
            class_id = int(class_id)

            if class_id not in self.cone_class_ids:
                continue
            color = self.cone_class_names[self.cone_class_ids.index(class_id)]

            # 计算坐标
            u_bottom = (x1 + x2) / 2.0
            v_bottom = y2
            v_top = y1
            pixel_height = v_bottom - v_top
            if pixel_height < 10:
                continue

            # 得到畸变矫正后的像素坐标
            points_distorted = np.array([[[u_bottom, v_bottom]]], dtype=np.float32)
            points_undistorted = cv2.undistortPoints(points_distorted, self.K, self.D, None, self.K).squeeze()
            u_und, v_und = points_undistorted

            # 相机系射线方向
            d_cam = np.array([
                (u_und - self.K[0, 2]) / self.K[0, 0],
                (self.K[1, 2] - v_und) / self.K[1, 1],
                1.0
            ], dtype=np.float32)
            # 转到车体系
            d_veh = np.dot(self.R_cam_to_vehicle, d_cam)

            O = self.T_cam_in_vehicle

            # 射线与地平面 z=0
            if d_veh[2] > -1e-6:
                continue
            t = -O[2] / d_veh[2]
            if t <= 0:
                continue
            vehicle_point = O + t * d_veh
            vehicle_point[2] = 0.0

            if not (self.cone_dist_filter[0] < vehicle_point[0] < self.cone_dist_filter[1]):
                continue

            raw_cones.append({
                "x": float(vehicle_point[0]),
                "y": float(vehicle_point[1]),
                "z": float(vehicle_point[2]),
                "color": color,
                "conf": float(conf),
            })

        #目标跟踪分配稳定 ID
        raw_cones = self.tracker.update(raw_cones, img_msg.header.stamp)

        #填充自定义消息
        for cone in raw_cones:
            cone_info = ConeInfo()
            cone_info.id = cone["id"]
            cone_info.position.x = cone["x"]
            cone_info.position.y = cone["y"]
            cone_info.position.z = cone["z"]
            cone_info.color = cone["color"]
            cone_info.confidence = cone["conf"]
            cone_array_msg.cones.append(cone_info)

        # 发布自定义ConeArray消息
        self.cone_array_pub.publish(cone_array_msg)
        rospy.loginfo(f"发布 {len(cone_array_msg.cones)} 个锥桶到话题 {self.cone_array_pub.name}")

        # 发布MarkerArray
        cone_markers = self.create_cone_markers(cone_array_msg)
        self.cone_marker_pub.publish(cone_markers)

        # 其他发布和可视化
        det_msg = create_detection_msg(img_msg, detections, img_msg.header.stamp)
        self.det_pub.publish(det_msg)

        if self.vis_pub:
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time)
            self.last_time = current_time

            bboxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2, _, _ in detections.tolist()]
            class_ids = [int(c) for _, _, _, _, _, c in detections.tolist()]
            confs = [float(c) for _, _, _, _, c, _ in detections.tolist()]
            img_vis = draw_detections(img_cv.copy(), bboxes, class_ids, self.class_labels, confs)
            cv2.putText(img_vis, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            vis_msg = self.bridge.cv2_to_imgmsg(img_vis, encoding="bgr8")
            vis_msg.header = img_msg.header
            self.vis_pub.publish(vis_msg)


if __name__ == "__main__":
    rospy.init_node("yolov7_cone_detection", anonymous=True)
    ns = rospy.get_name() + "/"

    # 读取参数
    try:
        weights_path = rospy.get_param(ns + "weights_path")
        classes_path = rospy.get_param(ns + "classes_path")
        img_topic = rospy.get_param(ns + "img_topic") 
        out_topic = rospy.get_param(ns + "out_topic")
        conf_thresh = rospy.get_param(ns + "conf_thresh", 0.5)
        iou_thresh = rospy.get_param(ns + "iou_thresh", 0.45)
        queue_size = rospy.get_param(ns + "queue_size", 1)
        img_size = rospy.get_param(ns + "img_size", 640)
        visualize = rospy.get_param(ns + "visualize", True)
        device = rospy.get_param(ns + "device", "cuda")
    except KeyError as e:
        rospy.logfatal(f"缺少参数：{str(e)}")
        exit(1)

    # 验证文件和设备
    if not os.path.exists(weights_path):
        rospy.logfatal(f"权重文件不存在：{weights_path}")
        exit(1)
    if not os.path.exists(classes_path):
        rospy.logfatal(f"类别文件不存在：{classes_path}")
        exit(1)

    class_labels = parse_classes_file(classes_path)
    rospy.loginfo(f"加载类别：{class_labels}")

    if device not in ["cuda", "cpu"] or (device == "cuda" and not torch.cuda.is_available()):
        rospy.logwarn("CUDA不可用，切换到CPU")
        device = "cpu"

    # 启动节点
    try:
        publisher = Yolov7Publisher(
            img_topic=img_topic,
            weights=weights_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            pub_topic=out_topic,
            device=device,
            img_size=(img_size, img_size),
            queue_size=queue_size,
            visualize=visualize,
            class_labels=class_labels
        )
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"节点失败：{str(e)}")
        exit(1)
