import numpy as np
import cv2
from typing import List, Union


def get_random_color(seed):
    """保证同一类别颜色稳定"""
    gen = np.random.default_rng(seed)
    color = tuple(gen.choice(range(256), size=3))
    color = tuple([int(c) for c in color])
    return color


def draw_detections(img, bboxes, classes, class_labels, confs=None):
    # 按类别固定颜色(OpenCV 是 BGR 顺序)
    # 0=红, 1=蓝, 2=黄
    CLASS_COLORS = [
        (0, 0, 255),    # red_cone  → 红
        (255, 0, 0),    # blue_cone → 蓝
        (0, 255, 255),  # yellow_cone → 黄
    ]
    for i, (bbox, cls) in enumerate(zip(bboxes, classes)):
        x1, y1, x2, y2 = bbox
        color = CLASS_COLORS[int(cls) % len(CLASS_COLORS)]  # 按类别取固定色
        #画检测框
        img = cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3
        )

        if class_labels:
            label = class_labels[int(cls)]
            if confs is not None and i < len(confs):
                label = f"{label}, {confs[i]:.2f}"

            x_text = int(x1)
            y_text = max(15, int(y1 - 10))
            img = cv2.putText(
                img, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA
            )

    return img
