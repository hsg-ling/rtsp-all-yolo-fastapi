import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import StrongSort
from pathlib import Path
import torch
import os
from collections import defaultdict, deque
from datetime import datetime  # 用于现实世界时间戳
import time
import subprocess
import threading
from config import (
    MODEL_PATHS, DEFAULT_VIDEO_PATH, SCREENSHOT_BASE_DIR, ALARM_VIDEO_BASE_DIR,
    RESULTS_DIR, OUTPUT_VIDEO_NAME, FFMPEG_PATH, RTMP_URL, USE_RTMP_STREAM,
    CHECK_LINE_X, DIRECTION_THRESHOLD, WAND_POCKET_IOU, WRIST_HAT_DIST,
    WRIST_ITEM_DIST, WRIST_HEAD_DIST, RESULT_DISPLAY_TIME,
    POSE_CONF_THRESHOLD, WAND_CONF_THRESHOLD, HAT_CONF_THRESHOLD,
    ITEM_CONF_THRESHOLD, WAND_DETECT_IMGSZ, TRACKER_CONFIG,
    SAVE_VIDEO, DEFAULT_FPS, ALARM_VIDEO_DURATION, ALARM_START_TIME_OFFSET,
    ALARM_END_TIME_OFFSET, FRAME_BUFFER_SIZE, PERSON_POSE_IOU_THRESHOLD,
    HAT_HEAD_IOU_THRESHOLD, WAND_WRIST_DIST_THRESHOLD,
    HAT_REMOVED_FRAMES_THRESHOLD, WRIST_NEAR_HEAD_FRAMES_THRESHOLD,
    DETECTION_SKIP_FRAMES, LOG_INTERVAL, ENABLE_DIRECTION_FILTER,
    DIRECTION_CONFIRM_FRAMES
)

# # 报警类型映射：英文 -> 中文（用于数据库存储）
# ALARM_TYPE_MAPPING = {
#     "detect no pass": "周身扫描不规范",
#     "hat no pass": "安检时未脱帽",
#     "item no pass": "携带违规物品"
# }
#
# def translate_alarm_types(fail_items):
#     """将报警类型从英文转换为中文"""
#     return [ALARM_TYPE_MAPPING.get(item, item) for item in fail_items]

DETECTION_STATE = {
    "draw_boxes": True,
    "enable_alarm": True,
    "callback": None,  # 报警回调函数
    "config": {},  # 设备配置信息
}

STATE_LOCK = threading.Lock()

# def calculate_iou(box1, box2):
#     """计算两个矩形框的IOU"""
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2
#     inter_x1 = max(x1, x3)
#     inter_y1 = max(y1, y3)
#     inter_x2 = min(x2, x4)
#     inter_y2 = min(y2, y4)
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     area1 = (x2 - x1) * (y2 - y1)
#     area2 = (x4 - x3) * (y4 - y3)
#     union_area = area1 + area2 - inter_area
#     return inter_area / union_area if union_area > 0 else 0
#
#
# def estimate_pocket_area(keypoints):
#     # COCO关节点索引：左髋(11)、右髋(12)、左膝(13)、右膝(14)
#     l_hip = keypoints[11]  # (x,y,conf)
#     r_hip = keypoints[12]
#     l_knee = keypoints[13]
#     r_knee = keypoints[14]
#
#     # 过滤无效关节点（置信度低或坐标为空）
#     valid = all([
#         not np.isnan(l_hip[0]), not np.isnan(l_hip[1]), l_hip[2] > 0.2,
#         not np.isnan(r_hip[0]), not np.isnan(r_hip[1]), r_hip[2] > 0.2,
#         not np.isnan(l_knee[0]), not np.isnan(l_knee[1]), l_knee[2] > 0.2,
#         not np.isnan(r_knee[0]), not np.isnan(r_knee[1]), r_knee[2] > 0.2
#     ])
#     if not valid:
#         return None, None
#
#     # 估算口袋区域（基于髋-膝之间，适当放大范围）
#     pocket_height = abs(l_knee[1] - l_hip[1]) * 0.8  # 口袋高度=髋膝距离的80%
#     pocket_width = abs(r_hip[0] - l_hip[0]) * 0.7  # 口袋宽度=髋间距的70%
#
#     # 左口袋：左髋下方，向左偏移
#     l_pocket = [
#         l_hip[0] - pocket_width / 2,  # x1
#         l_hip[1],  # y1（髋部为上边界）
#         l_hip[0] + pocket_width / 2,  # x2
#         l_hip[1] + pocket_height  # y2（下边界=上边界+高度）
#     ]
#     # 右口袋：右髋下方，向右偏移
#     r_pocket = [
#         r_hip[0] - pocket_width / 2,
#         r_hip[1],
#         r_hip[0] + pocket_width / 2,
#         r_hip[1] + pocket_height
#     ]
#     return l_pocket, r_pocket
#
#
# def estimate_pocket_area_from_box(person_box):
#     """
#     基于人体检测框估算裤子口袋区域：
#     - 使用人体框的中下部区域近似代表两侧裤兜位置
#     - 不再依赖关节点，避免关键点检测失败导致口袋丢失
#     """
#     x1, y1, x2, y2 = person_box
#     w = x2 - x1
#     h = y2 - y1
#     if w <= 0 or h <= 0:
#         return None, None
#
#     # 垂直方向：取人体框的中下 40% 区域作为口袋高度范围
#     pocket_top = y1 + h * 0.55
#     pocket_bottom = y1 + h * 0.75
#
#     # 水平方向：左右各取 35% 宽度，中间留出一小段空隙
#     left_x1 = x1 + w * 0.05
#     left_x2 = x1 + w * 0.45
#     right_x1 = x1 + w * 0.55
#     right_x2 = x1 + w * 0.95
#
#     l_pocket = [left_x1, pocket_top, left_x2, pocket_bottom]
#     r_pocket = [right_x1, pocket_top, right_x2, pocket_bottom]
#     return l_pocket, r_pocket
#
#
# def estimate_upper_pocket_area_from_box(person_box):
#     """
#     基于人体检测框估算上半身衣服左右口袋区域：
#     - 使用人体框的上部 25%-45% 区域近似代表两侧上衣口袋位置
#     - 分左右两部分，逻辑与裤子口袋类似
#     """
#     x1, y1, x2, y2 = person_box
#     w = x2 - x1
#     h = y2 - y1
#     if w <= 0 or h <= 0:
#         return None, None
#
#     # 垂直方向：取人体框的上部 25%-45% 区域作为口袋高度范围
#     pocket_top = y1 + h * 0.25
#     pocket_bottom = y1 + h * 0.45
#
#     # 水平方向：左右各取 35% 宽度，中间留出一小段空隙
#     left_x1 = x1 + w * 0.05
#     left_x2 = x1 + w * 0.45
#     right_x1 = x1 + w * 0.55
#     right_x2 = x1 + w * 0.95
#
#     l_upper_pocket = [left_x1, pocket_top, left_x2, pocket_bottom]
#     r_upper_pocket = [right_x1, pocket_top, right_x2, pocket_bottom]
#     return l_upper_pocket, r_upper_pocket

def detect_human(self, frame, draw_boxes=True, config=None):
    global DETECTION_STATE, STATE_LOCK
    # 默认配置
    if config is None:
        config = {}
    frame_count += 1
    current_time = frame_count / fps  # 使用实际FPS计算时间
    frame_h, frame_w = frame.shape[:2]
    run_detection = (frame_count % DETECTION_SKIP_FRAMES == 1)  # 使用配置文件中的跳帧设置

    if not run_detection:
        # 跳帧：不做检测
        # 如果 draw_boxes=True，使用上一帧已绘制好的画面（避免闪烁）
        # 如果 draw_boxes=False，直接使用当前原始帧（不画框）
        if draw_boxes and last_visualized_frame is not None:
            stable_frame = last_visualized_frame
        else:
            stable_frame = frame  # 不画框时使用原始帧
    # -------------------------- 第一步：基础检测 --------------------------
    # 1. 人体检测（用于跟踪）
    person_results = person_detector(frame, verbose=False)
    detections = []
    for res in person_results:
        for box in res.boxes:
            if box.cls.item() == 0:  # 只保留人体
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2, box.conf.item(), 0])
    detections = np.array(detections) if detections else np.empty((0, 6))

    # 2. 人体跟踪（获取带ID的人体框）
    tracked_objects = tracker.update(detections, frame)
    # ------------- 修复：确保 tracked_objects 为可迭代的 list（避免 None / 空） -------------
    if tracked_objects is None:
        tracked_objects = []
    else:
        # 有些实现返回 numpy array，确保可以正常迭代
        try:
            # 如果是 numpy array 且二维，转换为列表-of-lists
            if hasattr(tracked_objects, 'shape') and len(tracked_objects.shape) == 2:
                tracked_objects = tracked_objects.tolist()
        except Exception:
            pass

    tracked_ids = [int(track[4]) for track in tracked_objects if len(track) > 6 and track[6] == 0]  # 有效人体ID列表

    # 3. Pose关节点检测（用于口袋估算、手腕定位，使用配置文件中的置信度阈值）
    pose_results = pose_detector.predict(frame, conf=POSE_CONF_THRESHOLD, verbose=False)
    person_poses = []  # 存储每个人体的关节点：[(人体框, 关节点列表), ...]
    for res in pose_results:
        if not hasattr(res, "keypoints") or res.keypoints is None or res.keypoints.xy is None:
            continue
        kpts_xy = res.keypoints.xy.cpu().numpy()  # (N,17,2) 或 (0,17,2)
        kpts_conf = res.keypoints.conf.cpu().numpy()  # (N,17) 或 (0,17)
        # 检查是否有检测到关节点（第一个维度必须大于0）
        if kpts_xy.shape[0] == 0 or kpts_conf.shape[0] == 0:
            continue
        # 整理关节点为[(x,y,conf), ...]
        keypoints = [(kpts_xy[0][i][0], kpts_xy[0][i][1], kpts_conf[0][i]) for i in range(17)]
        # 获取当前Pose对应的人体框（取res.boxes的第一个框）
        if res.boxes is not None and len(res.boxes) > 0:
            person_box = res.boxes.xyxy[0].tolist()
            person_poses.append((person_box, keypoints))

    # 4. 金属探测仪检测（筛选靠近手腕的框，使用配置文件中的参数）
    wand_results = wand_detector.predict(frame, imgsz=WAND_DETECT_IMGSZ, conf=WAND_CONF_THRESHOLD, verbose=False)
    valid_wands = []  # 有效金属探测仪框（靠近任一手腕）
    wrists = []  # 所有人体的手腕关节点（用于过滤金属探测仪）
    for (p_box, keypoints) in person_poses:
        l_wrist = keypoints[9]  # 左腕(9)、右腕(10)
        r_wrist = keypoints[10]
        if not np.isnan(l_wrist[0]) and l_wrist[2] > 0.2:
            wrists.append((l_wrist[0], l_wrist[1]))
        if not np.isnan(r_wrist[0]) and r_wrist[2] > 0.2:
            wrists.append((r_wrist[0], r_wrist[1]))
    # 过滤金属探测仪：中心与手腕距离筛选
    for res in wand_results:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for (wx, wy) in wrists:
                if ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5 < WAND_WRIST_DIST_THRESHOLD:  # 使用配置文件中的阈值
                    valid_wands.append([x1, y1, x2, y2])
                    break

    # 5. 帽子检测（使用配置文件中的置信度阈值）
    hat_results = hat_detector.predict(frame, conf=HAT_CONF_THRESHOLD, verbose=False)
    hats = []  # 帽子框列表：[(x1,y1,x2,y2), ...]
    for res in hat_results:
        for box in res.boxes:
            hats.append(box.xyxy[0].tolist())

    # === 新增：违禁物检测（整帧检测，再与手腕匹配，使用配置文件中的置信度阈值） ===
    item_results = item_detector.predict(frame, conf=ITEM_CONF_THRESHOLD, verbose=False)
    items = []  # 违禁物框列表
    for res in item_results:
        for box in res.boxes:
            items.append(box.xyxy[0].tolist())

    # -------------------------- 第二步：状态更新 --------------------------
    # 1. 关联人体跟踪ID与Pose关节点（更新口袋估算、手腕位置、违禁物判断）
    if len(tracked_objects) == 0:
        # 没有跟踪对象，本帧跳过状态更新但仍要写视频和显示
        pass
    else:
        for track in tracked_objects:
            # 兼容 track 可能是 list/tuple 或者 numpy 数组行
            if not track:
                continue
            # 有些实现返回 list 长度 < 8 的行，要做保护
            if len(track) < 6:
                continue
            try:
                x1, y1, x2, y2, track_id, _, cls, _ = track
            except Exception:
                # 如果 track 长度不够，就跳过
                continue
            if cls != 0 or track_id not in tracked_ids:
                continue
            track_id = int(track_id)
            current_person_box = [x1, y1, x2, y2]

            # 找到当前跟踪ID对应的Pose关节点（IOU匹配）
            matched_pose = None
            for (p_box, keypoints) in person_poses:
                try:
                    if calculate_iou(current_person_box, p_box) > PERSON_POSE_IOU_THRESHOLD:  # 使用配置文件中的阈值
                        matched_pose = keypoints
                        break
                except Exception:
                    continue
            if matched_pose:
                track_states[track_id]["last_pose"] = matched_pose  # 保存关节点用于口袋估算

            # 2. 口袋扫描检测
            pocket_scan_ok = False
            if len(valid_wands) > 0:
                l_pocket, r_pocket = estimate_pocket_area_from_box(current_person_box)
                if l_pocket or r_pocket:
                    # 检查金属探测仪是否与任一口袋重叠
                    for wand in valid_wands:
                        if l_pocket and calculate_iou(wand, l_pocket) > wand_pocket_iou:
                            pocket_scan_ok = True
                        if r_pocket and calculate_iou(wand, r_pocket) > wand_pocket_iou:
                            pocket_scan_ok = True
            if pocket_scan_ok:
                track_states[track_id]["pocket_scan"] = True  # 扫描成功更新状态

            # 3. 帽子检测 / 脱帽状态机
            state = track_states[track_id]
            had_hat = state["had_hat"]
            hat_removed = state["hat_removed"]
            head_area = None
            head_center = None
            last_pose = state["last_pose"]

            # 3.1 基于关键点估计头部区域
            if last_pose:
                nose = last_pose[0]
                l_ear = last_pose[3]
                r_ear = last_pose[4]
                if not np.isnan(nose[0]) and nose[2] > 0.2:
                    head_radius = max(
                        abs(l_ear[0] - nose[0]) if not np.isnan(l_ear[0]) else 30,
                        abs(r_ear[0] - nose[0]) if not np.isnan(r_ear[0]) else 30
                    )
                    head_area = [
                        nose[0] - head_radius,
                        nose[1] - head_radius * 1.5,
                        nose[0] + head_radius,
                        nose[1] + head_radius * 0.5
                    ]
                    head_center = ((head_area[0] + head_area[2]) / 2,
                                   (head_area[1] + head_area[3]) / 2)

            state["head_area"] = head_area

            # 3.2 判断当前帧"帽子是否在头上"（使用配置文件中的IOU阈值）
            hat_on_head = False
            if head_area:
                for hat in hats:
                    if calculate_iou(head_area, hat) > HAT_HEAD_IOU_THRESHOLD:
                        hat_on_head = True
                        break
            state["hat_on_head"] = hat_on_head
            state["wore_hat"] = hat_on_head  # 供可视化文案使用

            # 3.3 更新“是否曾经戴过帽子”和最近一帧戴帽的帧号
            if hat_on_head:
                had_hat = True
                state["had_hat"] = True
                state["last_hat_on_frame"] = frame_count

            # 3.4 检测“手是否靠近头部”
            if last_pose and head_center is not None:
                l_wrist = last_pose[9]
                r_wrist = last_pose[10]
                current_wrists = []
                if not np.isnan(l_wrist[0]) and l_wrist[2] > 0.2:
                    current_wrists.append((l_wrist[0], l_wrist[1]))
                if not np.isnan(r_wrist[0]) and r_wrist[2] > 0.2:
                    current_wrists.append((r_wrist[0], r_wrist[1]))

                for (wx, wy) in current_wrists:
                    dist = ((wx - head_center[0]) ** 2 + (wy - head_center[1]) ** 2) ** 0.5
                    if dist < WRIST_HEAD_DIST:  # 使用配置文件中的距离阈值
                        state["last_wrist_near_head_frame"] = frame_count
                        break

            # 3.5 判定是否“完成脱帽”
            if had_hat and not hat_removed:
                last_hat_on = state["last_hat_on_frame"]
                last_wrist_near = state["last_wrist_near_head_frame"]
                if last_hat_on is not None and last_wrist_near is not None:
                    frames_since_hat = frame_count - last_hat_on
                    frames_since_wrist = frame_count - last_wrist_near
                    # 条件：手近期靠近过头部，且已有一段时间检测不到帽子在头上（使用配置文件中的阈值）
                    if frames_since_hat > HAT_REMOVED_FRAMES_THRESHOLD and frames_since_wrist < WRIST_NEAR_HEAD_FRAMES_THRESHOLD:
                        hat_removed = True
                        state["hat_removed"] = True

            # 3.6 根据状态机给出 hat_check 结果
            if not had_hat:
                # 从未戴过帽子，直接通过
                hat_check_ok = True
            elif hat_removed:
                # 戴过帽且完成了脱帽动作，通过
                hat_check_ok = True
            else:
                # 戴过帽但尚未检测到脱帽动作，未通过
                hat_check_ok = False

            state["hat_check"] = hat_check_ok  # 更新帽子检测状态

            # === 违禁物检测（判断手上是否拿物品） ===
            has_item = False
            if last_pose and len(items) > 0:
                # 获取手腕点
                l_wrist = last_pose[9]
                r_wrist = last_pose[10]
                wrists_for_item = []
                if not np.isnan(l_wrist[0]) and l_wrist[2] > 0.2:
                    wrists_for_item.append((l_wrist[0], l_wrist[1]))
                if not np.isnan(r_wrist[0]) and r_wrist[2] > 0.2:
                    wrists_for_item.append((r_wrist[0], r_wrist[1]))
                # 距离判断
                for (wx, wy) in wrists_for_item:
                    for item_box in items:
                        item_cx = (item_box[0] + item_box[2]) / 2
                        item_cy = (item_box[1] + item_box[3]) / 2
                        dist = ((wx - item_cx) ** 2 + (wy - item_cy) ** 2) ** 0.5
                        if dist < wrist_item_dist:
                            has_item = True
                            break
                    if has_item:
                        break
            # item_check True 表示通过（即未检测到手上有违禁物）
            track_states[track_id]["item_check"] = not has_item

            # 4. 方向判断（判断是向左走还是向右走）
            center_x = (x1 + x2) / 2
            current_pos = (center_x, (y1 + y2) / 2)
            state = track_states[track_id]

            if track_id in track_last_positions:
                prev_pos = track_last_positions[track_id]
                prev_x = prev_pos[0] if isinstance(prev_pos, (list, tuple)) else prev_pos

                # 计算当前帧的移动方向
                if center_x > prev_x + 5:  # 向右移动（阈值5像素，避免微小抖动）
                    current_direction = 'right'
                elif center_x < prev_x - 5:  # 向左移动
                    current_direction = 'left'
                else:
                    current_direction = None  # 基本不动

                # 更新方向历史记录
                if current_direction:
                    state["direction_history"].append(current_direction)

                # 确认方向（基于最近N帧的历史记录）
                if len(state["direction_history"]) >= DIRECTION_CONFIRM_FRAMES:
                    # 统计最近N帧中哪个方向出现次数最多
                    right_count = state["direction_history"].count('right')
                    left_count = state["direction_history"].count('left')

                    if right_count > left_count:
                        state["direction"] = 'right'
                        state["is_entering"] = True  # 向右走 = 进入
                    elif left_count > right_count:
                        state["direction"] = 'left'
                        state["is_entering"] = False  # 向左走 = 离开
                    # 如果相等，保持之前的方向
            else:
                # 首次出现，初始化方向历史
                state["direction_history"].clear()

            # 5. 警戒线穿越判断（记录通过时间）
            crossed_line = False
            if track_id in track_last_positions:
                prev_pos = track_last_positions[track_id]
                # prev_pos 可能是 tuple 或单值（不同实现），做兼容
                prev_x = prev_pos[0] if isinstance(prev_pos, (list, tuple)) else prev_pos
                if (prev_x < check_line_x - direction_threshold and center_x >= check_line_x) or \
                        (prev_x > check_line_x + direction_threshold and center_x <= check_line_x):
                    crossed_line = True
                    track_states[track_id]["cross_time"] = current_time  # 记录通过时间
            track_last_positions[track_id] = current_pos

    # -------------------------- 第三步：可视化与结果显示 --------------------------
    if draw_boxes:
        # 1. 绘制辅助元素（警戒线、口袋区域、金属探测仪）
        # 警戒线
        cv2.line(frame, (check_line_x, 0), (check_line_x, frame_h), (0, 255, 255), 2)
        cv2.putText(frame, "Security Line", (check_line_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # 口袋区域（裤子口袋 + 上半身口袋）
        if len(tracked_objects) > 0:
            for track in tracked_objects:
                # 安全解包
                if not track or len(track) < 6:
                    continue
                x1, y1, x2, y2, track_id, _, cls, _ = track
                if cls != 0:
                    continue
                # # 裤子口袋（中下部）
                # l_pocket, r_pocket = estimate_pocket_area_from_box([x1, y1, x2, y2])
                # if l_pocket:
                #     cv2.rectangle(frame, (int(l_pocket[0]), int(l_pocket[1])),
                #                   (int(l_pocket[2]), int(l_pocket[3])), (255, 255, 0), 2, cv2.LINE_AA)
                #     cv2.putText(frame, "Pocket", (int(l_pocket[0]), int(l_pocket[1]) - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # if r_pocket:
                #     cv2.rectangle(frame, (int(r_pocket[0]), int(r_pocket[1])),
                #                   (int(r_pocket[2]), int(r_pocket[3])), (255, 255, 0), 2, cv2.LINE_AA)
                #     cv2.putText(frame, "Pocket", (int(r_pocket[0]), int(r_pocket[1]) - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # # 上半身口袋（25%-45%位置）
                # l_upper_pocket, r_upper_pocket = estimate_upper_pocket_area_from_box([x1, y1, x2, y2])
                # if l_upper_pocket:
                #     cv2.rectangle(frame, (int(l_upper_pocket[0]), int(l_upper_pocket[1])),
                #                   (int(l_upper_pocket[2]), int(l_upper_pocket[3])), (255, 200, 0), 2, cv2.LINE_AA)
                #     cv2.putText(frame, "Upper Pocket", (int(l_upper_pocket[0]), int(l_upper_pocket[1]) - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                # if r_upper_pocket:
                #     cv2.rectangle(frame, (int(r_upper_pocket[0]), int(r_upper_pocket[1])),
                #                   (int(r_upper_pocket[2]), int(r_upper_pocket[3])), (255, 200, 0), 2, cv2.LINE_AA)
                #     cv2.putText(frame, "Upper Pocket", (int(r_upper_pocket[0]), int(r_upper_pocket[1]) - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        # 金属探测仪
        for wand in valid_wands:
            cv2.rectangle(frame, (int(wand[0]), int(wand[1])), (int(wand[2]), int(wand[3])), (0, 255, 0), 2)
            cv2.putText(frame, "Metal Wand", (int(wand[0]), int(wand[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # 帽子
        for hat in hats:
            cv2.rectangle(frame, (int(hat[0]), int(hat[1])), (int(hat[2]), int(hat[3])), (255, 0, 255), 2)
            cv2.putText(frame, "Hat", (int(hat[0]), int(hat[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # === 绘制违禁物框（橙色） ===
        for item in items:
            cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (0, 165, 255), 2)
            cv2.putText(frame, "Prohibited", (int(item[0]), int(item[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 2. 绘制人体跟踪框與实时状态
        if len(tracked_objects) > 0:
            for track in tracked_objects:
                if not track or len(track) < 6:
                    continue
                x1, y1, x2, y2, track_id, _, cls, _ = track
                if cls != 0:
                    continue
                track_id = int(track_id)
                state = track_states[track_id]
                # 颜色：全部通过=绿色，部分未通过=黄色，全未通过=红色
                # 现在把 item_check 一起纳入考虑（只要任一未通过就不是全通过）
                if state["pocket_scan"] and state["hat_check"] and state["item_check"]:
                    color = (0, 255, 0)
                elif state["pocket_scan"] or state["hat_check"] or state["item_check"]:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                # 绘制跟踪框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # 实时状态标签（口袋+帽子+违禁物）
                if state["hat_on_head"]:
                    hat_label = "Hat: On"
                elif state["hat_removed"]:
                    hat_label = "Hat: Removed"
                elif state["had_hat"]:
                    # 曾经戴过帽子，但尚未检测到完整脱帽动作
                    hat_label = "Hat: Need Remove"
                else:
                    hat_label = "Hat: None"
                pocket_label = "Pocket: Pass" if state["pocket_scan"] else "Pocket: Fail"
                item_label = "Item: Pass" if state["item_check"] else "Item: Fail"
                # 方向显示
                direction = state.get("direction", None)
                if direction == 'right':
                    direction_label = "Dir: Entering →"
                    direction_color = (0, 255, 0)  # 绿色
                elif direction == 'left':
                    direction_label = "Dir: Leaving ←"
                    direction_color = (0, 165, 255)  # 橙色
                else:
                    direction_label = "Dir: Unknown"
                    direction_color = (128, 128, 128)  # 灰色

                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, direction_label, (int(x1), int(y1) - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, direction_color, 1)
                cv2.putText(frame, hat_label, (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, pocket_label, (int(x1), int(y1) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, item_label, (int(x1), int(y1) - 0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. 显示通过警戒线后的结果（持续1秒）并保存报警（含现实世界时间、截图）
    realtime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for track_id in list(track_states.keys()):
        cross_time = track_states[track_id]["cross_time"]
        if cross_time and (current_time - cross_time) <= result_display_time:
            # 收集未通过项
            fail_items = []
            if not track_states[track_id]["pocket_scan"]:
                fail_items.append("detect no pass")
            if not track_states[track_id]["hat_check"]:
                fail_items.append("hat no pass")
            if not track_states[track_id]["item_check"]:
                fail_items.append("item no pass")

            # 结果文本
            if not fail_items:
                result_text = f"ID:{track_id} all pass"
                text_color = (0, 255, 0)
            else:
                result_text = f"ID:{track_id} no pass: {', '.join(fail_items)}"
                text_color = (0, 0, 255)

                # 方向过滤：如果启用了方向过滤，只对向右走（进入）的人报警
                should_trigger_alarm = True
                if ENABLE_DIRECTION_FILTER:
                    state = track_states[track_id]
                    is_entering = state.get("is_entering", False)
                    direction = state.get("direction", None)

                    if not is_entering:
                        # 向左走（离开），忽略报警
                        should_trigger_alarm = False
                        if direction == 'left':
                            print(f"[方向过滤] ID:{track_id} 向左走（离开），忽略报警")
                        elif direction is None:
                            print(f"[方向过滤] ID:{track_id} 方向未确定，忽略报警")
                    else:
                        # 向右走（进入），正常报警
                        print(f"[方向过滤] ID:{track_id} 向右走（进入），允许报警")

                # 保存报警截图与信息（如果有未通过项，且尚未处理，且方向允许）
                # 注意：截图和数据库保存是同步的，都在这里处理
                if should_trigger_alarm and not track_states[track_id].get("callback_triggered", False):
                    screenshot_path = None

                    # 1. 先截图（如果需要，使用配置文件中的路径）
                    if save_picture and not track_states[track_id]["screenshot_saved"]:
                        # 动态生成截图保存路径（使用配置文件中的基础路径）
                        base_dir = SCREENSHOT_BASE_DIR
                        placename = config.get('placename', 'unknown')
                        devicename = config.get('devicename', 'unknown')
                        date_str = datetime.now().strftime("%Y%m%d")  # 格式：20251207

                        # 构建完整路径
                        screenshot_dir = os.path.join(base_dir, placename, devicename, date_str)
                        os.makedirs(screenshot_dir, exist_ok=True)  # 确保目录存在

                        # 生成截图文件名
                        screenshot_filename = f"alert_frame{frame_count}_ID{track_id}.jpg"
                        screenshot_name = os.path.join(screenshot_dir, screenshot_filename)

                        cv2.imwrite(screenshot_name, frame)

                        # 生成相对路径（从 AlarmScreenShots 开始，用于数据库存储）
                        relative_path = os.path.join("AlarmScreenShots", placename, devicename, date_str, screenshot_filename)
                        # 将路径分隔符统一为 /（Windows 和 Linux 兼容）
                        screenshot_path = relative_path.replace("\\", "/")

                        track_states[track_id]["screenshot_saved"] = True  # 标记已截图，避免重复
                        print(f"[截图] {realtime_str} - ID:{track_id} - 违规项目: {', '.join(fail_items)}")
                        print(f"[截图] 截图已保存: {screenshot_name}")
                        print(f"[截图] 数据库路径: {screenshot_path}")


            # 绘制结果（固定位置显示，避免重叠）
            if draw_boxes:
                display_y = 100 + (track_id % 5) * 30  # 最多同时显示5个结果，换行
                if display_y < frame_h - 30:
                    cv2.putText(frame, result_text, (50, display_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    # 结果背景框（增强可读性）
                    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (50 - 5, display_y - text_size[1] - 5),
                                  (50 + text_size[0] + 5, display_y + 5), (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.putText(frame, result_text, (50, display_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

if __name__ == "__main__":
    main()