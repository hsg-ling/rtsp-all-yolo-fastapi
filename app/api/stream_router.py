# app/api/stream_router.py
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
import logging
import subprocess
import asyncio
from app.services.rtsp_processor import get_processor_service, RTSP2_URL

logger = logging.getLogger("FastAPI-Gateway")
router = APIRouter(prefix="/stream", tags=["视频流网关"])


# RTSP2拉流转FLV的ffmpeg进程（每个WebSocket连接独立创建）
class FLVStreamer:
    def __init__(self, rtsp_url: str, fps: int = 20, width: int = 1280, height: int = 720):
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.ffmpeg_process = None

    def start(self):
        """启动ffmpeg：拉RTSP2 → 转FLV流（标准输出）"""
        ffmpeg_cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            # '-c:v', 'libx264',
            '-c:v', 'copy',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '10',
            '-r', '20',
            '-c:a', 'aac',
            '-fflags', 'nobuffer+flush_packets+genpts+discardcorrupt',
            '-flags', 'low_delay',
            '-f', 'flv',
            '-flush_packets', '1', # 每帧立即刷新输出（无积压）
            '-flvflags', 'no_duration_filesize', # 不等待时长 / 文件大小（实时流专用）
            'pipe:1'  # 输出FLV到标准输出
            # 'text.flv'
        ]

        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"FFmpeg拉RTSP2转FLV启动: {self.rtsp_url}")


    def read_chunk(self, chunk_size: int = 4096) -> bytes:
        """读取FLV流块"""
        if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
            return b''
        return self.ffmpeg_process.stdout.read(chunk_size)
    def stop(self):
        """停止ffmpeg进程"""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                # 等待进程退出，超时后强制杀死
                asyncio.wait_for(self.ffmpeg_process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
            finally:
                self.ffmpeg_process = None

# ------------------- 接口定义 -------------------
@router.post("/start-processor")
async def start_processor():
    """启动RTSP处理推流服务（拉RTSP1→处理→推RTSP2）"""
    processor = get_processor_service()
    try:
        processor.start()
        return {"status": "success", "message": "RTSP处理推流服务已启动", "rtsp2_url": RTSP2_URL}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop-processor")
async def stop_processor():
    """停止RTSP处理推流服务"""
    processor = get_processor_service()
    processor.stop()
    return {"status": "success", "message": "RTSP处理推流服务已停止"}


@router.websocket("/ws/flv")
async def websocket_flv_stream(websocket: WebSocket):
    """WebSocket推送FLV流（从RTSP2拉取）"""
    await websocket.accept()
    logger.info(f"前端WebSocket客户端已连接: {id(websocket)}")

    # 1. 检查RTSP处理推流服务是否运行
    processor = get_processor_service()
    if not processor.is_running:
        await websocket.send_text("错误：RTSP处理推流服务未启动，请先调用/stream/start-processor")
        await websocket.close()
        return

    # 2. 创建FLV流转换器（拉RTSP2→转FLV）
    streamer = FLVStreamer(RTSP2_URL)
    streamer.start()




    try:
        # 3. 循环推送FLV流块到前端
        while True:
            chunk = streamer.read_chunk(4 * 1024)
            if not chunk:
                await asyncio.sleep(0.001)
                continue
            logger.info(f"------------成功读取：{len(chunk)/1024} KB 大小flv流")
            await websocket.send_bytes(chunk)
            logger.info(f"成功发送 {len(chunk)/1024} KB flv")

    except WebSocketDisconnect:
        logger.info("前端WebSocket客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket推送失败: {e}")
        await websocket.close(code=1011, reason=str(e))
    finally:
        # 4. 清理资源
        streamer.stop()
        logger.info("FLV流转换器已停止")


@router.get("/status")
async def get_status():
    """获取服务状态"""
    processor = get_processor_service()
    return {
        "processor_running": processor.is_running,
        "rtsp1_url": processor.input_rtsp,
        "rtsp2_url": processor.output_rtsp
    }

@router.get("/webrtc/{id}")
async def get_webrtc(id: str):
    """webrtc传输"""

    logger.info(f"请求ID: {id}")

    # 1.启动RTSP推流服务
    processor = get_processor_service()
    try:
        processor.start()
        logger.info(f"WebRTC rtsp推流服务已启动 rtsp2_url: {RTSP2_URL}")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # time.sleep(2)
    # 2.返回WebRTC连接地址
    return {
        "webrtcUrl": "http://localhost:8899/webrtc_stream"
    }

