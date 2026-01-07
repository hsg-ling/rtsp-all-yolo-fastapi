# app/services/rtsp_processor.py
import queue
from datetime import datetime

import cv2
import subprocess
import threading
import time
import logging
from typing import Optional
from app.core.config import settings
import numpy as np
from ultralytics import YOLO
import torch

logger = logging.getLogger("RTSP-Processor")


class RTSPProcessAndPushService:
    """拉取RTSP1流 → YOLO处理 → 推送到RTSP2服务器"""

    def __init__(
            self,
            input_rtsp: str,  # RTSP1地址（express推送的源）
            output_rtsp: str,  # RTSP2地址（推送到第二个mediaMTX）
            fps: int = 20,
            width: int = 1024,
            height: int = 576,
            yolo_model: Optional[YOLO] = None,
            target_class: Optional[int] = None,

    ):
        self.input_rtsp = input_rtsp
        self.output_rtsp = output_rtsp
        self.fps = fps
        self.width = width
        self.height = height
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo_model = yolo_model.to(self.device)
        self.target_class = target_class
        logger.info(f"camera fps: {self.fps}")

        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None

        self.frame_queue = queue.Queue(maxsize=5)  # 小队列，避免堆积
        self.process_thread = None  # 独立的处理/推流线程

    def _init_ffmpeg_push(self):
        """初始化ffmpeg，将处理后的帧推送到RTSP2服务器（低延迟）"""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()

        # ffmpeg推RTSP命令（零延迟配置）
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # 从标准输入读帧
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-level', '3.0',
            # '-g', '25',  # 关键帧间隔25帧（1秒/帧，核心！）
            '-g', str(self.fps),  # 关键帧间隔=帧率（1秒1个关键帧，解决花屏）
            '-keyint_min', str(self.fps),  # 最小关键帧间隔，避免间隔过大
            '-sc_threshold', '0',  # 禁用场景切换自动插关键帧，保证间隔稳定
            '-fflags', 'nobuffer+flush_packets',  # 立即刷帧，不缓存
            '-max_delay', '100000',  # 最大延迟500ms
            '-flags', 'low_delay',
            '-an',  # 禁用音频
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            '-flush_packets', '1',  # 强制立即刷包
            self.output_rtsp
        ]

        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"FFmpeg推流到RTSP2启动: {self.output_rtsp}")

    def _pull_worker(self):
        """独立拉流线程：只负责拉取RTSP1帧，放入队列"""
        self.cap = cv2.VideoCapture(self.input_rtsp)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 20    # 获取源流帧率，兜底20
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            logger.error("拉流失败")
            self.is_running = False
            return

        while self.is_running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 队列满时丢弃旧帧（避免堆积）
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        # logger.warning("frame queue full, dropping frame")
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)


    # --------------- YOLO 人体检测函数-----------------------
    def detect_human(self, frame):
        """
        使用YOLO模型检测人体，并绘制检测框
        :param frame: 输入的视频帧
        :return: 绘制了检测框的帧
        """


        # 缩放帧以提高处理速度
        resized_frame = cv2.resize(frame, (self.width, self.height))

        # YOLO推理
        results = self.yolo_model(
            resized_frame,
            classes=[self.target_class],
            conf=0.5,
            half=True
        ) # conf, 置信度阈值

        # 绘制检测框和标签
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取框的坐标（xyxy格式）
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 获取置信度
                conf = float(box.conf[0])
                # 绘制矩形框（绿色，线宽2）
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制标签（置信度）
                label = f"Person {conf:.2f}"
                cv2.putText(resized_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return resized_frame



    def _process_worker(self):
        """独立处理/推流线程：从队列取帧，处理后推RTSP2"""
        self._init_ffmpeg_push()
        while self.is_running:
            try:
                # 超时等待队列帧，避免空轮询
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 假YOLO处理 + 推流（原有逻辑）
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # cv2.putText(frame, current_time, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # # 模拟处理帧耗时
            # time.sleep(0.05)
            # frame = cv2.resize(frame, (self.width, self.height))

            frame = self.detect_human(frame)

            try:

                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
            except BrokenPipeError:
                self._init_ffmpeg_push()
                continue

    def start(self):
        """启动两个独立线程"""
        if self.is_running:
            raise RuntimeError("已运行")
        self.is_running = True
        # 启动拉流线程
        self.thread = threading.Thread(target=self._pull_worker, daemon=True)
        self.thread.start()
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.process_thread.start()

    def stop(self):
        """停止处理推流服务"""
        self.is_running = False
        # 清理资源
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        if self.cap:
            self.cap.release()
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("RTSP处理推流服务已停止")


# 全局实例（配置RTSP1和RTSP2地址）
RTSP1_URL = f"rtsp://{settings.c_user}:{settings.c_password}@{settings.c_ip}:{settings.c_port}/Streaming/Channels/101"  # 要拉取的RTSP1地址
RTSP2_URL = "rtsp://localhost:8555/processed_stream"  # 推送到RTSP2的地址
# --------------- YOLO配置 -----------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# 轻量版 yolov8n
yolo_model = YOLO("yolov8n.pt")

# 只检测人体，YOLO中person的类别ID是0
TARGET_CLASS = 0

_processor_service = RTSPProcessAndPushService(RTSP1_URL, RTSP2_URL, yolo_model=yolo_model, target_class=TARGET_CLASS)


def get_processor_service() -> RTSPProcessAndPushService:
    """获取处理推流服务单例"""
    return _processor_service

