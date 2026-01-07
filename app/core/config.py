import io
import logging
import json
import sys
from datetime import datetime
import os

from fastapi_cloud_cli.config import Settings
from pydantic_settings import BaseSettings, SettingsConfigDict

# 确保日志目录存在
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """自定义JSON日志格式化器，延迟指标"""

    def format(self, record):
        # 基础日志字段
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        # 添加自定义延迟字段
        if hasattr(record, "latency"):
            log_record["latency_ms"] = record.latency
        if hasattr(record, "frame_id"):
            log_record["frame_id"] = record.frame_id
        if hasattr(record, "latency_type"):
            log_record["latency_type"] = record.latency_type  # 延迟类型：pull/infer/total/queue

        return json.dumps(log_record, ensure_ascii=False)


# 配置全局日志
def setup_logging():
    # 全局日志器：同时输出到文件和控制台
    file_handler = logging.FileHandler(f"{LOG_DIR}/stream_latency.log", encoding="utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)

    # 设置JSON格式
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    console_handler.setFormatter(json_formatter)

    # 全局日志配置
    root_logger = logging.getLogger()
    # 移除所有已存在的处理器（防止Uvicorn默认处理器+自定义处理器重复）
    root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 6. 屏蔽无关模块的日志（uvicorn ffmpeg），并关闭其日志传播
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.WARNING)
    uvicorn_logger.propagate = False  # 阻止Uvicorn日志向上传播到root logger

    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.setLevel(logging.WARNING)
    ultralytics_logger.propagate = False


# 初始化日志（增加防重复执行的判断）
if not logging.getLogger().handlers:  # 只有当root logger没有处理器时才初始化
    setup_logging()

# 业务配置
STREAM_CONFIG = {
    "input_rtsp": "rtsp://localhost:8554/stream",
    "fps": 30,
    "width": 1280,
    "height": 720,
    "log_interval": 30,  # 每30帧输出一次详细延迟日志（避免刷屏）
}


class Settings(BaseSettings):
    # 摄像头配置
    c_user: str
    c_password: str
    c_port: int
    c_ip: str

    # 从 .env 文件加载环境变量
    model_config = SettingsConfigDict(
        env_file="app/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,

    )


settings = Settings()
