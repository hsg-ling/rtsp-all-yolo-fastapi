# app/services/rtsp_service.py
import cv2
import queue
import threading
import time
import logging
import subprocess
import os
from typing import Optional
from .yolo_service import get_yolo_service
from app.core.config import STREAM_CONFIG

logger = logging.getLogger("RTSP-Service")


class RTSPStreamService:
    def __init__(self, input_rtsp: str, fps: int = 30, width: int = 1280, height: int = 720):
        self.input_rtsp = input_rtsp
        self.fps = fps
        self.width = width
        self.height = height

        # 帧队列（供FLV封装使用）
        self.frame_queue = queue.Queue(maxsize=5)  # 减小队列大小降低延迟
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.yolo_service = get_yolo_service()

        # 帧ID计数器
        self.frame_id_counter = 0
        self.log_interval = STREAM_CONFIG["log_interval"]

        # FLV封装相关：ffmpeg子进程+管道
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.flv_pipe = None  # FLV流管道（用于WebSocket读取）

    def _init_ffmpeg_flv(self):
        """初始化ffmpeg，将原始帧封装为FLV流（低延迟）"""
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.terminate()

        # ffmpeg命令：原始RGB帧 → FLV流（零延迟配置）
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # 从标准输入读取帧
            '-c:v', 'h264',
            '-preset', 'ultrafast',  # 最快编码
            '-tune', 'zerolatency',  # 零延迟
            '-fflags', 'nobuffer',  # 禁用缓冲区
            '-flags', 'low_delay',  # 低延迟
            '-movflags', 'frag_keyframe+empty_moov',  # FLV分片关键帧
            '-f', 'flv',  # 输出FLV格式
            'pipe:1'  # 输出到标准输出
        ]

        # 启动ffmpeg，标准输出作为FLV流管道
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        self.flv_pipe = self.ffmpeg_process.stdout
        logger.info("FFmpeg FLV封装进程已启动（低延迟模式）")

    def _stream_worker(self):
        """拉流→处理→FLV封装核心线程"""
        # 初始化RTSP拉流
        self.cap = cv2.VideoCapture(self.input_rtsp)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化拉流缓冲区
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            logger.error(f"无法打开RTSP流: {self.input_rtsp}")
            self.is_running = False
            return

        # 初始化FLV封装
        self._init_ffmpeg_flv()

        logger.info(f"RTSP流已连接，FLV封装启动: {self.input_rtsp}")

        while self.is_running:
            # 1. 拉帧+统计延迟
            pull_start = time.time()
            ret, frame = self.cap.read()
            pull_end = time.time()
            pull_latency = round((pull_end - pull_start) * 1000, 2)

            if not ret or frame is None:
                logger.warning("读取帧失败，重试...")
                time.sleep(0.1)
                continue

            self.frame_id_counter += 1
            current_frame_id = self.frame_id_counter

            # 2. 调整帧大小+假YOLO处理（绘制时间戳）
            frame = cv2.resize(frame, (self.width, self.height))
            try:
                infer_start = time.time()
                processed_frame, _ = self.yolo_service.infer_frame(frame, current_frame_id)
                infer_end = time.time()
                infer_latency = round((infer_end - infer_start) * 1000, 2)
            except Exception as e:
                logger.error(f"假YOLO处理失败: {e}", extra={"frame_id": current_frame_id})
                processed_frame = frame
                infer_latency = 0

            # 3. 计算整体处理延迟
            total_process_latency = round((time.time() - pull_start) * 1000, 2)

            # 4. 按间隔输出延迟日志
            if current_frame_id % self.log_interval == 0:
                logger.info(
                    f"帧整体处理完成",
                    extra={
                        "latency": total_process_latency,
                        "latency_type": "total_process",
                        "frame_id": current_frame_id,
                        "pull_latency": pull_latency,
                        "infer_latency": infer_latency
                    }
                )

            # 5. 将处理后的帧写入ffmpeg标准输入（生成FLV流）
            try:
                self.ffmpeg_process.stdin.write(processed_frame.tobytes())
                self.ffmpeg_process.stdin.flush()
            except BrokenPipeError:
                logger.error("FFmpeg管道断开，重启FLV封装...")
                self._init_ffmpeg_flv()
                continue

            # 控制帧率（避免过快）
            time.sleep(1 / self.fps - (time.time() - pull_start))

    def start_stream(self):
        """启动流处理"""
        if self.is_running:
            raise RuntimeError("流已在运行中")
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()
        logger.info("RTSP流处理+FLV封装已启动")

    def stop_stream(self):
        """停止流处理"""
        self.is_running = False
        # 停止ffmpeg进程
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process = None
        # 停止拉流线程
        if self.thread:
            self.thread.join(timeout=2)
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        # 释放资源
        if self.cap:
            self.cap.release()
        logger.info("RTSP流处理+FLV封装已停止")

    def read_flv_stream(self, chunk_size: int = 4096) -> Optional[bytes]:
        """读取FLV流数据（供WebSocket推送）"""
        if not self.is_running or self.flv_pipe is None:
            return None
        try:
            # 按块读取FLV流（小块传输降低延迟）
            return self.flv_pipe.read(chunk_size)
        except Exception as e:
            logger.error(f"读取FLV流失败: {e}")
            return None


# 全局实例
_stream_service: Optional[RTSPStreamService] = None


def get_stream_service(input_rtsp: str, fps: int = 30, width: int = 1280, height: int = 720) -> RTSPStreamService:
    global _stream_service
    if _stream_service is None:
        _stream_service = RTSPStreamService(input_rtsp, fps, width, height)
    return _stream_service