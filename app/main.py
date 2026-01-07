# app/main.py
from fastapi import FastAPI
from app.api.stream_router import router as stream_router
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import setup_logging

# 初始化日志
setup_logging()

app = FastAPI(title="RTSP网关服务", version="1.0")

# 跨域配置（支持WebSocket）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(stream_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rtsp-gateway"}