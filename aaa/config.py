# -*- coding: utf-8 -*-
"""
系统配置文件
所有需要根据环境修改的参数都在这里统一配置
"""

# ==================== 数据库配置 ====================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'hzhdatabase',
    'charset': 'utf8mb4'
}

# ==================== 视频输入配置 ====================
# 默认视频路径（当未指定时使用）
# DEFAULT_VIDEO_PATH = "E:/yolovideo/clipvodeo/demo/demo_video.mkv"
# DEFAULT_VIDEO_PATH = "/home/jxsf/judicialVideo/files/demo_video.mkv"
# DEFAULT_VIDEO_PATH = "D:/Desktop/jxsf/clipvideo/merged_video.mkv"
# DEFAULT_VIDEO_PATH = "http://localhost:9001/live/gj3.flv"
# DEFAULT_VIDEO_PATH = "rtmp://localhost:1936/live/gj3"
# DEFAULT_VIDEO_PATH = "rtsp://admin:jxcjdx2025@192.168.1.103:554/cam/realmonitor?channel=1&subtype=0"
DEFAULT_VIDEO_PATH = "rtsp://admin:jxcjdx2025@192.168.1.103:554/Streaming/Channels/101"
# 备用视频路径（可选）
# BACKUP_VIDEO_PATH = "E:/yolovideo/clipvodeo/gj4.mp4"

# ==================== 模型文件路径配置 ====================
MODEL_PATHS = {
    'person_detector': 'yolov8n.pt',          # 人体检测模型
    'wand_detector': 'bat_yolo11s.pt',        # 金属探测仪检测模型
    'pose_detector': 'yolov8n-pose.pt',       # 姿态关节点检测模型
    'hat_detector': 'hat_yolo11s.pt',         # 帽子检测模型
    'item_detector': 'item_yolo11s.pt',       # 违禁物检测模型
    'reid_weights': 'weights/osnet_x0_25_msmt17.pt'  # 跟踪器权重文件
}

# ==================== 文件保存路径配置 ====================
# 截图保存基础路径
SCREENSHOT_BASE_DIR = "D:/code/judicialVideo/files/AlarmScreenShots"
# SCREENSHOT_BASE_DIR = "/home/jxsf/judicialVideo/files/AlarmScreenShots"
# SCREENSHOT_BASE_DIR = "D:/Desktop/jxsf/judicialVideo/files/AlarmScreenShots"
# 报警视频保存基础路径
ALARM_VIDEO_BASE_DIR = "D:/code/judicialVideo/files/AlarmVideo"
# ALARM_VIDEO_BASE_DIR = "/home/jxsf/judicialVideo/files/AlarmVideo"
# ALARM_VIDEO_BASE_DIR = "D:/Desktop/jxsf/judicialVideo/files/AlarmVideo"
# 结果输出目录（用于保存处理后的视频）
RESULTS_DIR = "results"
# 输出视频文件名
OUTPUT_VIDEO_NAME = "all_demo_v2_unshow_1.mp4"

# ==================== FFmpeg 推流配置（MediaMTX RTSP） ====================
# FFmpeg 可执行文件路径
FFMPEG_PATH = r"D:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
# FFMPEG_PATH = "/usr/bin/ffmpeg"
# FFMPEG_PATH = r"D:\Program Files\ffmpeg-2025-11-10-git-133a0bcb13-full_build\bin\ffmpeg.exe"
# RTSP 推流地址（推送到 MediaMTX）
RTMP_URL = "rtsp://localhost:8555/processed_stream"
# RTSP_URL = "rtsp://192.168.1.100:8554/security_camera"  # 远程 MediaMTX 服务器
# 是否启用 RTSP 推流
USE_RTMP_STREAM = True

# ==================== 检测参数配置 ====================
# 安检警戒线X坐标（竖线位置）
CHECK_LINE_X = 980
# 安检位置X坐标（用于判断是否到达安检区域，不显示）
SECURITY_LINE_X = 650
# 安检区域范围（±像素，人员中心x坐标在此范围内才使用虚拟手腕安检棒）
SECURITY_AREA_THRESHOLD = 50
# 穿越判定阈值
DIRECTION_THRESHOLD = 0
# 方向判断：是否启用方向过滤（只对向右走的人报警）####
ENABLE_DIRECTION_FILTER = True ##### 2025-12-28 添加 #####
# 方向判断：需要连续N帧确认方向（避免误判）####
DIRECTION_CONFIRM_FRAMES = 3 ##### 2025-12-28 添加 #####
# 金属探测仪与口袋的IOU阈值（算扫描成功）
WAND_POCKET_IOU = 0.1
# 手腕与帽子的距离阈值（接近即算摘帽）
WRIST_HAT_DIST = 50
# 手腕与违禁物距离阈值（接近即算手上拿着）
WRIST_ITEM_DIST = 80
# 手接近头部的距离阈值（用于脱帽检测）
WRIST_HEAD_DIST = 80
# 检测结果显示时长（秒）
RESULT_DISPLAY_TIME = 1.0

# ==================== 模型检测置信度阈值 ====================
# 人体检测置信度（在代码中硬编码为0，这里保留用于参考）
PERSON_CONF_THRESHOLD = 0.0
# 姿态检测置信度
POSE_CONF_THRESHOLD = 0.1
# 金属探测仪检测置信度
WAND_CONF_THRESHOLD = 0.25
# 帽子检测置信度
HAT_CONF_THRESHOLD = 0.3
# 违禁物检测置信度
ITEM_CONF_THRESHOLD = 0.25
# 金属探测仪检测图像尺寸
WAND_DETECT_IMGSZ = 960

# ==================== 跟踪器参数配置 ====================
TRACKER_CONFIG = {
    'max_cos_dist': 0.3,      # 最大余弦距离
    'max_iou_dist': 0.8,       # 最大IOU距离
    'max_age': 40,             # 最大未匹配帧数（超过此帧数认为目标离开）
    'n_init': 2,               # 初始化所需匹配次数
    'nn_budget': 100,          # 特征库大小
    'mc_lambda': 0.98,         # 运动补偿参数
    'half': False              # 是否使用半精度
}

# ==================== 视频处理参数 ====================
# 是否保存处理后的视频到本地文件
SAVE_VIDEO = False
# 默认FPS（当视频无法读取FPS时使用）
DEFAULT_FPS = 25.0
# 报警视频保存时长（秒，保存报警前N秒）
ALARM_VIDEO_DURATION = 6.0
# 报警视频时间偏移（秒）
ALARM_START_TIME_OFFSET = 2.0  # 报警前2秒
ALARM_END_TIME_OFFSET = 1.0    # 报警后1秒
# 帧缓存大小（用于保存报警视频，3秒 × 25fps = 75帧）
FRAME_BUFFER_SIZE = 150

# ==================== 检测算法参数 ====================
# 人体框与Pose关节点匹配的IOU阈值
PERSON_POSE_IOU_THRESHOLD = 0.3
# 帽子与头部区域匹配的IOU阈值
HAT_HEAD_IOU_THRESHOLD = 0.15
# 金属探测仪与手腕的距离阈值（用于过滤）
WAND_WRIST_DIST_THRESHOLD = 50
# 脱帽检测：帽子消失后的帧数阈值
HAT_REMOVED_FRAMES_THRESHOLD = 5
# 脱帽检测：手靠近头部后的有效帧数阈值
WRIST_NEAR_HEAD_FRAMES_THRESHOLD = 25

# ==================== 监控循环配置 ====================
# 轮询间隔（秒）
DEVICE_CHECK_INTERVAL = 5

# ==================== 其他配置 ====================
# 跳帧检测
DETECTION_SKIP_FRAMES = 2
# 日志输出频率（每N帧输出一次日志）
LOG_INTERVAL = 10

