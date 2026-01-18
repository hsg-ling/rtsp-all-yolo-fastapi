# 摄像头rtsp流YOLO处理
## 简介
这是一个为摄像头增加YOLO处理以实现目标检测的代码

**实现流程**

后端使用OpenCV拉取摄像头的rtsp流，经过YOLO处理后，用ffmpeg推送至mediamtx上，前端可以通过访问mediamtx供外部访问的路径获取其返回的可播放页面，前端用``<iframe>``标签嵌入

**所使用技术：**

- 后端：fastapi
- 前端：vue3
- 流媒体服务器：mediamtx_v1.15.5
- 拉流转码工具：ffmpeg_7.1.1
- 开发环境：Windows11

## 使用
1. 创建一个项目环境
2. 编写mediamtx配置文件：
   
   以我代码为例：

   ```yml文件
   paths:
     webrtc_stream:  # 这是供前端访问的路径
       source: rtsp://localhost:8555/processed_stream  # 这是后端处理之后推送至mediamtx的路径，这里的8555端口根据自己实际情况调整，供mediamtx读取，也就是我rtsp_process.py中的RTSP2_URL
       sourceProtocol: tcp # 解决花屏的关键
       sourceOnDemand: yes

   # 内网穿透 解决 deadline exceeded while waiting connection 问题
   webrtcICEServers2: [url: stun:stun.l.google.com:19302]   # 在mediamtx配置文件 409 行附近配置ICE服务器
  
3.  确保rtsp和WebRTC服务开启
4.  根据摄像头实际rtsp流地址设置``rtsp_process.py``的RTSP1_URL
5.  在fastapi配置一下项目运行配置，用一开始创建的环境
6.  前端只需访问：``get_webrtc``路由即可，后端会返回可以直接从mediamtx访问的路径
   
