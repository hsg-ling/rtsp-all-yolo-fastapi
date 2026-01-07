import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging
import time
import datetime

logger = logging.getLogger("YOLO-Service")

class YoloService:
    """假处理，仅在帧上绘制时间戳"""
    def __init__(self, model_path: str = "", conf_threshold: float = 0.5):
        """
        此处参数是为实际YOLO处理占位，此处不使用
        :param model_path:
        :param conf_threshold:
        """
        logger.info("假YOLO处理初始化完成")

    def infer_frame(self, frame: np.ndarray, frame_id: int = 0) -> Tuple[np.ndarray, List[Dict]]:
        """
        假处理：绘制当前时间戳，保留延迟统计和日志输出
        :param frame: OpenCV读取的RGB格式帧
        :param frame_id: 帧唯一标识（用于关联日志）
        :return: 绘制了时间戳的帧
        """
        if frame is None or frame.size == 0:
            raise ValueError("输入帧为空，无法处理")

        # 1. 记录假处理开始时间
        infer_start_time = time.time()

        # 2. 绘制当前时间戳
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # 精确到毫秒
        # 在帧上绘制时间戳（红色字体，位置（50, 50））
        cv2.putText(
            frame,
            current_time,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, # 字体大小
            (0, 0, 255), # 红色
            2,  # 线条粗细
            cv2.LINE_AA,
        )

        # 3. 模拟处理耗时
        time.sleep(0.01)

        # 4. 计算延迟(ms)
        infer_end_time = time.time()
        infer_latency = round((infer_end_time - infer_start_time) * 1000, 2)

        # 5. 输出延迟日志（保持原有结构化格式）
        logger.info(
            f"假YOLO处理完成",
            extra={
                "latency": infer_latency,
                "latency_type": "infer",
                "frame_id": frame_id,
            }
        )

        # 6. 返回结果(空)
        return frame, []

# 单例模式（兼容原有调用逻辑）
_yolo_service = None
def get_yolo_service(model_path: str = "", conf_threshold: float = 0.5) -> YoloService:
    # 获取假YOLO处理单例
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YoloService(model_path, conf_threshold)
    return _yolo_service

