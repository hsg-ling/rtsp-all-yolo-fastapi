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

    # # 缓冲区配置
    # CHUNK_SIZE = 256 * 1024 # 块大小
    # BUFFER_MAXSIZE = 10
    # PRE_FILL_THRESHOLD = 2  # 预填充2个块
    # ACTUAL_BITRATE = 5000 * 1024 * 1024   # 实际码率1Mbps
    # SEND_INTERVAL = (CHUNK_SIZE * 8) / ACTUAL_BITRATE  # 计算发送间隔
    #
    # buffer = asyncio.Queue(maxsize=BUFFER_MAXSIZE)
    # stop_event = asyncio.Event()  # 用于通知生产者停止
    # pre_fill_counter = 0  # 生产者成功放入的计数（仅用于预填充阶段）
    # pre_fill_done = asyncio.Event()  # 预填充完成的信号
    # PRE_FILL_TIMEOUT = 1.0  # 最大等待时间（避免卡死）
    # BUFFER_SAFE_WATER = 5  # 缓冲区安全水位：超过则生产者休眠
    # PRODUCER_SLEEP = 0.005  # 生产者休眠时间（给消费者让时间）
    #
    # async def producer():
    #     nonlocal pre_fill_counter
    #     try:
    #         while not stop_event.is_set():
    #             chunk = streamer.read_chunk(CHUNK_SIZE)
    #             if not chunk:
    #                 await asyncio.sleep(0.001)
    #                 continue
    #
    #             # 预填充阶段：优先保证填满阈值，缓冲区满时等待（不丢弃数据）
    #             if not pre_fill_done.is_set():
    #                 # 预填充未完成时，允许等待更长时间（避免丢弃）
    #                 try:
    #                     await asyncio.wait_for(buffer.put(chunk), timeout=1.0)
    #                     pre_fill_counter += 1
    #                     logger.info(f"预填充中，已放入 {pre_fill_counter}/{PRE_FILL_THRESHOLD} 块")
    #                     # 达到阈值则标记预填充完成
    #                     if pre_fill_counter >= PRE_FILL_THRESHOLD:
    #                         await asyncio.sleep(0)
    #                         # await asyncio.sleep(0.001)
    #                         pre_fill_done.set()
    #                 except asyncio.TimeoutError:
    #                     logger.warning("预填充阶段缓冲区满，等待空间...")
    #                     continue  # 继续等待，不丢弃数据
    #             else:
    #                 # 缓冲区超过安全水位，生产者主动休眠
    #                 if buffer.qsize() >= BUFFER_SAFE_WATER:
    #                     await asyncio.sleep(PRODUCER_SLEEP)
    #
    #                 # 预填充完成后，按原有逻辑处理（满缓冲时丢弃旧数据）
    #                 try:
    #                     await asyncio.wait_for(buffer.put(chunk), timeout=0.1)
    #                 except asyncio.TimeoutError:
    #                     if not buffer.empty():
    #                         await buffer.get()
    #                         buffer.task_done()
    #                     await buffer.put(chunk)
    #                     logger.warning("缓冲区已满，丢弃部分数据")
    #
    #             # 每放1块就主动让出事件循环，强制给消费者机会
    #             await asyncio.sleep(0)
    #             logger.info(f"缓冲区大小：{buffer.qsize()}/{BUFFER_MAXSIZE}")
    #     except Exception as e:
    #         logger.error(f"生产者异常：{e}")
    #     finally:
    #         logger.info("生产者已停止")
    #
    # # 启动生产者
    # producer_task = asyncio.create_task(producer())
    #
    # # 等待预填充完成（基于信号+超时）
    # logger.info(f"等待{id(websocket)}预填充至少 {PRE_FILL_THRESHOLD} 个块...")
    # try:
    #     # 等待pre_fill_done信号，最多等PRE_FILL_TIMEOUT秒
    #     await asyncio.wait_for(pre_fill_done.wait(), timeout=PRE_FILL_TIMEOUT)
    #     logger.info(f"预填充完成，实际放入 {pre_fill_counter} 块，当前缓冲：{buffer.qsize()}")
    # except asyncio.TimeoutError:
    #     logger.warning(f"预填充超时（{PRE_FILL_TIMEOUT}秒），当前仅放入 {pre_fill_counter} 块，继续启动消费者")
    #
    # try:
    #     # 消费者，从缓冲区读取数据并发送到WebSocket
    #     while not stop_event.is_set():
    #         try:
    #             # 缓冲区空时最多等待1秒，避免永久阻塞
    #             chunk = await asyncio.wait_for(buffer.get(), timeout=0.5)
    #
    #             # 强制按间隔发送
    #             # await asyncio.sleep(SEND_INTERVAL)
    #
    #             # 发送数据
    #             await websocket.send_bytes(chunk)
    #             buffer.task_done()
    #             logger.info(f"向{id(websocket)}发送数据：{len(chunk)/1024:.2f}KB, 剩余缓冲：{buffer.qsize()}")
    #
    #             # 2. 【关键】先让出事件循环（让生产者有机会生产），再休眠
    #             await asyncio.sleep(0)  # 强制切换任务，让生产者执行
    #             await asyncio.sleep(0.001)  # 缩短休眠时间，加快循环
    #
    #
    #         except asyncio.TimeoutError:
    #             logger.warning("缓冲区空，等待数据...")
    #             await asyncio.sleep(0.1)
    #         except Exception as e:
    #             logger.error(f"发送数据异常：{e}")
    #
    # except WebSocketDisconnect:
    #     logger.info("前端WebSocket客户端断开连接")
    # except asyncio.TimeoutError:
    #     logger.warning("缓冲区长时间无数据，可能流已中断")
    # except Exception as e:
    #     logger.error(f"WebSocket推送失败：{e}")
    #     await websocket.close(code=1011, reason=str(e))
    # finally:
    #     # 清理资源
    #     stop_event.set()    # 通知生产者停止
    #     await producer_task # 等待生产者退出
    #     streamer.stop()
    #     # 清空缓冲区
    #     while not buffer.empty():
    #         buffer.get()
    #         buffer.task_done()
    #     await buffer.join() # 确保所有任务已完成
    #     logger.info("FLV流转换器及缓冲区已清理")



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

