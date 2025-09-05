# -*- coding: utf-8 -*-
"""
Posture Monitoring & Pressure Area Protection - Realtime Inference (with class-subset constraints)
Rev C-3  (raw 僅臉部馬賽克、無任何框；annotated 另存一張；自動 POST 到 API 寫 DB)

依賴:
  pip install opencv-python pyttsx3 numpy torch torchvision efficientnet_pytorch mediapipe pyrealsense2 requests

熱鍵:
  ESC: 離開
  L  : 中/英文疊字切換
  V  : 語音開/關
  P  : 姿勢變化自動儲存 開/關
"""

import os
import cv2
import time
import math
import json
import torch
import threading
import pyttsx3
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import requests
from datetime import datetime
from collections import deque
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mysql.connector

# ============================= 可調參數 ============================= #
MODEL_PATH = r'C:\Users\USER\Desktop\pressure_ulcer_posture_recognition\model\runs\exp1_full\efficientnet_depth_model.pth'  # 或 .ckpt
CAPTURE_ROOT = 'captures'; os.makedirs(CAPTURE_ROOT, exist_ok=True)

# --- API 設定 ---
# 可用環境變數 POSTURE_API_URL 覆寫
API_URL = os.environ.get('POSTURE_API_URL', 'http://127.0.0.1:8000/records')
ENABLE_API_POST = True           # False 則不打 API（僅落地存檔）
API_TIMEOUT_SEC = 5              # requests timeout
API_RETRY = 1                    # 失敗重試次數（不含第一次）

IMG_SIZE = 224
CLASS_NAMES = [
    'heel_correct', 'heel_wrong',
    'left_sacrum_correct', 'left_sacrum_wrong',
    'right_sacrum_correct', 'right_sacrum_wrong',
    'supine'
]

INCLUDE_CAMERA_NAME = True

# 顯示/時間設定
STABLE_HOLD_SECONDS    = 2.0
ANALYSIS_CYCLE_SECONDS = 8.0
TAIL_COUNTDOWN_SECONDS = 5
GAP_SECONDS            = 15.0
PATIENT_LOST_TIMEOUT   = 10
FPS_ESTIMATE           = 30.0

# 穩定/移動門檻
WINDOW_SIZE = 30
STABLE_THRESH = {'avg_vel':0.004,'max_vel':0.010,'spatial_std':0.005}
MOVEMENT_DETECT_THRESH = {'avg_vel':0.006,'max_vel':0.014,'spatial_std':0.006}
UNSTABLE_CANCEL = {'avg_vel':0.008,'max_vel':0.020,'spatial_std':0.009}

# 自動儲存
AUTO_SAVE_ON_POSE_CHANGE = False
AUTO_SAVE_CONF = 0.75
POSE_CHANGE_MIN_INTERVAL = 8.0

# 語音冷卻
SPEECH_EVENTS_COOLDOWN = {
    'start_analysis': 3, 'analysis_end': 3, 'pose_change': 6,
    'return_idle': 6, 'movement_detected': 3, 'movement_idle': 5
}

# ROI / 標籤
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS  = 2
LABEL_MARGIN     = 6
LABEL_INNER_PAD  = 4
LABEL_ALPHA      = 0.55
SACRUM_COLOR     = (0, 220, 0)
HEEL_COLOR       = (255, 100, 0)

# 臉部馬賽克：只在偵測到臉時啟用
FACE_EXPAND_X = 20
FACE_EXPAND_Y = 30

# 語言
LANG_EN = 'en'
LANG_ZH = 'zh'
OVERLAY_LANGUAGE = LANG_ZH

# 顯示強化
CLEAR_ROI_WHEN_NO_PERSON = True
ENABLE_ROI_STALE_TIMEOUT = True
ROI_STALE_SECONDS = 5.0

def round2(x):
    try:
        return None if x is None else float(f"{float(x):.2f}")
    except:
        return None

def to_posix(p: str) -> str:
    return p.replace('\\', '/')

# ============================= 文案 ============================= #
TEXT = {
    'waiting':      {LANG_EN:'Waiting for stable posture...', LANG_ZH:'等待穩定姿勢...'},
    'no_person':    {LANG_EN:'No person detected...', LANG_ZH:'未偵測到人員...'},
    'stable_progress': {LANG_EN:'Stable: {cur:.1f}/{target:.1f}s', LANG_ZH:'穩定中: {cur:.1f}/{target:.1f}s'},
    'analysis_time': {LANG_EN:'Analysis Time: {mm:02d}:{ss:02d}', LANG_ZH:'分析時間: {mm:02d}:{ss:02d}'},
    'post_gap_in': {LANG_EN:'Next cycle in {sec} s', LANG_ZH:'下輪偵測於 {sec} 秒後開始'},
    'starting_analysis': {LANG_EN:'Starting analysis.', LANG_ZH:'開始姿勢分析。'},
    'patient_left': {LANG_EN:'Patient left. Returning to idle.', LANG_ZH:'患者離開，回到待機。'},
    'movement_detected': {LANG_EN:'Movement detected. Restarting.', LANG_ZH:'偵測到移動，重新等待穩定。'},
    'movement_idle': {LANG_EN:'Movement detected. Waiting for stability.', LANG_ZH:'偵測到移動，等待穩定。'},
    'analysis_end': {LANG_EN:'Cycle finished. Next in {sec} s.', LANG_ZH:'本輪完成，{sec} 秒後進行下一輪。'},
    'sacrum_roi': {LANG_EN:'Sacrum ROI', LANG_ZH:'薦骨 ROI'},
    'heel_roi':   {LANG_EN:'Heel ROI',   LANG_ZH:'足跟 ROI'},
    'no_data':    {LANG_EN:'NO DATA',    LANG_ZH:'無資料'},
    'lang_switched': {LANG_EN:'Language switched.', LANG_ZH:'語言已切換。'},
    'voice_off':     {LANG_EN:'Voice disabled.',    LANG_ZH:'語音功能關閉。'},
    'voice_on':      {LANG_EN:'Voice enabled.',     LANG_ZH:'語音功能開啟。'},
    'auto_save_off': {LANG_EN:'Auto pose save OFF.',LANG_ZH:'姿勢變化自動儲存關閉。'},
    'auto_save_on':  {LANG_EN:'Auto pose save ON.', LANG_ZH:'姿勢變化自動儲存開啟。'}
}
def T(key, language=None, **kwargs):
    lang = language or OVERLAY_LANGUAGE
    return TEXT.get(key, {}).get(lang, key).format(**kwargs) if key in TEXT else key

# ============================= 語音管理 ============================= #
class SpeechManager:
    def __init__(self):
        self.enabled = True
        self.last_event_time = {}
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
            self.lock = threading.Lock()
        except Exception:
            self.engine = None
            self.enabled = False
    def say(self, text: str):
        if not self.enabled or self.engine is None: return
        def run():
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
        threading.Thread(target=run, daemon=True).start()
    def event(self, tag: str, text: str):
        if not self.enabled: return
        now = time.time()
        cool = SPEECH_EVENTS_COOLDOWN.get(tag, 3)
        if now - self.last_event_time.get(tag, 0) >= cool:
            self.say(text); self.last_event_time[tag] = now
speech = SpeechManager()

# ============================= 工具函式 ============================= #
def load_model_and_classes(model_path, class_names, img_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inferred_names = class_names[:]
    inferred_imgsz = img_size
    if model_path.lower().endswith('.ckpt'):
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        if 'class_names' in ckpt and isinstance(ckpt['class_names'], (list, tuple)):
            inferred_names = list(ckpt['class_names'])
        if 'img_size' in ckpt and isinstance(ckpt['img_size'], int):
            inferred_imgsz = ckpt['img_size']
        num_classes = len(inferred_names)
        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
        model.load_state_dict(state, strict=True)
    else:
        num_classes = len(inferred_names)
        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)
    model.to(device); model.eval()
    return model, inferred_names, inferred_imgsz, device

def parse_sacrum_label(label: str):
    if not label: return '', ''
    if label.startswith('left_sacrum_'):  return 'left',  ('correct' if label.endswith('_correct') else 'wrong')
    if label.startswith('right_sacrum_'): return 'right', ('correct' if label.endswith('_correct') else 'wrong')
    if label == 'supine':                 return 'supine','neutral'
    return '', ''

# ============================= 視覺/ROI ============================= #
mp_pose = mp.solutions.pose
SELECTED_IDS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_HEEL.value,
    mp_pose.PoseLandmark.RIGHT_HEEL.value
]

def anonymize_face_with_box(color_img_bgr, results):
    """
    只在「偵測到臉部關鍵點」時，對臉部區域加馬賽克。
    沒偵測到臉：原圖不變。
    回傳：(masked_img, face_box or None)
    """
    img = color_img_bgr.copy()
    face_box = None
    h,w = img.shape[:2]

    if results is not None and results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        face_ids = [mp_pose.PoseLandmark.NOSE.value,
                    mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE.value,
                    mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value]
        xs,ys=[],[]
        for idx in face_ids:
            x=int(lm[idx].x*w); y=int(lm[idx].y*h)
            if 0<=x<w and 0<=y<h: xs.append(x); ys.append(y)
        if xs and ys:
            x_min = max(min(xs)-FACE_EXPAND_X,0); x_max = min(max(xs)+FACE_EXPAND_X,w)
            y_min = max(min(ys)-FACE_EXPAND_Y,0); y_max = min(max(ys)+FACE_EXPAND_Y,h)
            roi = img[y_min:y_max, x_min:x_max]
            if roi.size>0:
                small = cv2.resize(roi,(max(1,roi.shape[1]//12), max(1,roi.shape[0]//12)))
                mosaic = cv2.resize(small,(roi.shape[1],roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                img[y_min:y_max, x_min:x_max] = mosaic
                face_box = (x_min,y_min,x_max,y_max)
    return img, face_box

def boxes_overlap(a,b):
    if a is None or b is None: return False
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    return not (ax2<=bx1 or bx2<=ax1 or ay2<=by1 or by2<=ay1)

def place_label(roi_box, face_box, occupied, img_w, img_h,label_size,
                margin=LABEL_MARGIN, inner_pad=LABEL_INNER_PAD):
    x1,y1,x2,y2 = roi_box; tw,th = label_size
    candidates=[
        (x1, y1 - th - 2*inner_pad - margin),
        (x1, y1 + margin),
        (x1, y2 + margin),
        (x1, y2 - th - 2*inner_pad - margin),
        (min(x2+margin, img_w - tw - 2*inner_pad - 1), max(y1,0))
    ]
    for tx,ty in candidates:
        bx1=max(tx - inner_pad,0); by1=max(ty - inner_pad,0)
        bx2=min(tx + tw + inner_pad,img_w-1); by2=min(ty + th + inner_pad,img_h-1)
        box=(bx1,by1,bx2,by2)
        if boxes_overlap(box, face_box): continue
        if any(boxes_overlap(box,o) for o in occupied): continue
        if (by2-by1) < th: continue
        return tx,ty+th,bx1,by1,bx2,by2
    tx = x1+4; ty = y1 + (y2-y1)//2
    bx1=max(tx - inner_pad,0); by1=max(ty - th - inner_pad,0)
    bx2=min(tx + tw + inner_pad,img_w-1); by2=min(ty + inner_pad,img_h-1)
    return tx,ty,bx1,by1,bx2,by2

def draw_label(img, roi_box, face_box, occupied, text, color):
    if roi_box == (0,0,0,0): return None
    img_h,img_w = img.shape[:2]
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)
    tx,baseline,bx1,by1,bx2,by2 = place_label(roi_box,face_box,occupied,img_w,img_h,(tw,th))
    overlay = img.copy()
    cv2.rectangle(overlay,(bx1,by1),(bx2,by2),color,-1)
    cv2.addWeighted(overlay,LABEL_ALPHA,img,1-LABEL_ALPHA,0,img)
    cv2.putText(img,text,(tx,baseline),cv2.FONT_HERSHEY_SIMPLEX,LABEL_FONT_SCALE,(0,0,0),LABEL_THICKNESS+1,cv2.LINE_AA)
    cv2.putText(img,text,(tx,baseline),cv2.FONT_HERSHEY_SIMPLEX,LABEL_FONT_SCALE,(255,255,255),LABEL_THICKNESS,cv2.LINE_AA)
    occupied.append((bx1,by1,bx2,by2))
    return (bx1,by1,bx2,by2)

def get_roi_bounds(points, margin=30, img_w=848, img_h=480):
    if not points: return 0,0,0,0
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x_min = max(min(xs)-margin, 0); x_max = min(max(xs)+margin, img_w)
    y_min = max(min(ys)-margin, 0); y_max = min(max(ys)+margin, img_h)
    return int(x_min), int(x_max), int(y_min), int(y_max)

def preprocess_depth_roi(roi_depth):
    if roi_depth.size == 0: return None
    depth_filtered = cv2.medianBlur(roi_depth, 5)
    mask_zero = (depth_filtered == 0)
    if np.any(~mask_zero):
        depth_filtered[mask_zero] = np.median(depth_filtered[~mask_zero])
    else:
        depth_filtered[:] = 0
    valid = depth_filtered[depth_filtered > 0]
    if valid.size>0:
        p5,p95 = np.percentile(valid,5), np.percentile(valid,95)
        clipped = np.clip(depth_filtered, p5, p95)
        inverted = p95 - clipped
        depth_norm = cv2.normalize(inverted, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_filtered, dtype=np.uint8)
    return cv2.resize(depth_norm, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)

# ============================= 推論（子集合約束）============================= #
def prepare_tensor(depth_roi, device):
    if depth_roi is None: return None
    depth_3ch = np.stack([depth_roi]*3, axis=2)
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tfm(depth_3ch).unsqueeze(0).to(device)

@torch.no_grad()
def predict_subset(model, tensor, allowed_indices):
    logits = model(tensor)                          # [1, C]
    mask = torch.full_like(logits, float('-inf'))   # [1, C]
    mask[:, allowed_indices] = logits[:, allowed_indices]
    probs = torch.softmax(mask, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs

def build_status_sentence_cn(class_names, sacrum_idx, heel_idx):
    posture = '未知'; sacrum_status = '未判定'; heel_status = '未判定'
    if sacrum_idx is not None:
        lab = class_names[sacrum_idx]
        if lab.startswith('left_sacrum_'):
            posture = '左側躺'; sacrum_status = '正確' if lab.endswith('_correct') else '錯誤'
        elif lab.startswith('right_sacrum_'):
            posture = '右側躺'; sacrum_status = '正確' if lab.endswith('_correct') else '錯誤'
        elif lab == 'supine':
            posture = '平躺'; sacrum_status = '一般'
    if heel_idx is not None:
        hlab = class_names[heel_idx]
        if hlab == 'heel_correct': heel_status = '正確'
        elif hlab == 'heel_wrong': heel_status = '錯誤'
    return f"目前患者姿勢：{posture}，薦骨：{sacrum_status}，足跟：{heel_status}。"

def compute_stability_metrics(history):
    if len(history) < 2: return None
    arr = np.array(history); coords = arr[..., :2]
    diffs = coords[1:] - coords[:-1]; dist = np.linalg.norm(diffs, axis=2)
    return {'avg_vel': float(dist.mean()),
            'max_vel': float(dist.max()),
            'spatial_std': float(coords.std(axis=0).mean())}

def calc_frame_score(sacrum_prob, heel_prob, metrics):
    if sacrum_prob is None or heel_prob is None or metrics is None: return -1.0
    mean_prob = (sacrum_prob + heel_prob) / 2.0
    penalty = 4.0 * metrics['avg_vel'] + 2.0 * metrics['spatial_std'] + 1.5 * metrics['max_vel']
    return mean_prob - penalty

# ============================= API 寫入 ============================= #
def post_record_to_api(metadata: dict, folder: str):
    """將紀錄送到 FastAPI /records；失敗時印出錯誤但不丟例外。"""
    if not ENABLE_API_POST:
        return

 _

    tries = 1 + max(0, int(API_RETRY))
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(API_URL, json=payload, timeout=API_TIMEOUT_SEC)
            r.raise_for_status()
            j = r.json()
            print(f"[API] write ok id={j.get('id')} (attempt {attempt}/{tries})")
            return
        except Exception as e:
            print(f"[API] write failed (attempt {attempt}/{tries}): {e}")
            if attempt < tries:
                time.sleep(0.8)
            else:
                # 最後一次也失敗就放棄
                return

# ============================= 存檔（含自動打 API）============================= #
def save_capture(metadata, folder):
    # image_paths dict
    image_paths = {
        "rgb_masked_raw": to_posix(os.path.join(folder, 'rgb_masked_raw.jpg')),
        "rgb_masked_annotated": to_posix(os.path.join(folder, 'rgb_masked_annotated.jpg')),
        "sacrum_depth_proc": to_posix(os.path.join(folder, 'sacrum_depth_proc.png')),
        "heel_depth_proc": to_posix(os.path.join(folder, 'heel_depth_proc.png')),
    }

    # flatten into one dict
    record = {
        **metadata,      # unpack metadata keys
        **image_paths,   # unpack image path keys
        "folder": to_posix(folder)
    }

    # also save metadata.json and metadata.txt (your original requirement)
    with open(os.path.join(folder, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, 'metadata.txt'), 'w', encoding='utf-8') as f:
        for k, v in metadata.items():
            f.write(f"{k}={v}\n")

    # insert into MySQL
    insert_record(record)

    def insert_record(record):
    conn = mysql.connector.connect(
        host="localhost",
        user="your_user",
        password="your_password",
        database="your_db"
    )
    cursor = conn.cursor()

    sql = """
    INSERT INTO posture_records2
    (timestamp_iso, sacrum_class, sacrum_side, sacrum_protection,
     sacrum_confidence, heel_class, heel_confidence, camera_serial, camera_name,
     folder, rgb_masked_raw, rgb_masked_annotated, sacrum_depth_proc, heel_depth_proc)
    VALUES (%(timestamp_iso)s, %(sacrum_class)s, %(sacrum_side)s, %(sacrum_protection)s,
            %(sacrum_confidence)s, %(heel_class)s, %(heel_confidence)s, %(camera_serial)s, %(camera_name)s,
            %(folder)s, %(rgb_masked_raw)s, %(rgb_masked_annotated)s, %(sacrum_depth_proc)s, %(heel_depth_proc)s)
    """

    # convert timestamp string → datetime
    if isinstance(record.get("timestamp_iso"), str):
        record["timestamp_iso"] = datetime.fromisoformat(record["timestamp_iso"])

    cursor.execute(sql, record)
    conn.commit()
    cursor.close()
    conn.close()

# ============================= 主程式 ============================= #
def main():
    global IMG_SIZE, CLASS_NAMES, OVERLAY_LANGUAGE

    print('[INIT] Loading model ...')
    model, CLASS_NAMES, IMG_SIZE, DEVICE = load_model_and_classes(MODEL_PATH, CLASS_NAMES, IMG_SIZE)
    print(f'[INIT] Model ready. Classes: {CLASS_NAMES} | IMG_SIZE={IMG_SIZE} | Device={DEVICE}')

    def idxes(names): return [CLASS_NAMES.index(n) for n in names if n in CLASS_NAMES]
    HEEL_ALLOWED_IDX   = idxes(['heel_correct', 'heel_wrong'])
    SACRUM_ALLOWED_IDX = idxes([
        'left_sacrum_correct', 'left_sacrum_wrong',
        'right_sacrum_correct','right_sacrum_wrong',
        'supine'
    ])
    assert len(HEEL_ALLOWED_IDX)>=1 and len(SACRUM_ALLOWED_IDX)>=1, "Allowed class indices missing. Check CLASS_NAMES."

    # Mediapipe
    mp_pose_module = mp.solutions.pose
    pose = mp_pose_module.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                               enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # RealSense
    print('[INIT] Starting RealSense ...')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # 相機資訊
    try:
        dev = profile.get_device()
        try: CAMERA_NAME = dev.get_info(rs.camera_info.name)
        except: CAMERA_NAME = 'Unknown RealSense'
        try: PRODUCT_LINE = dev.get_info(rs.camera_info.product_line)
        except: PRODUCT_LINE = ''
        try: SERIAL_NUMBER = dev.get_info(rs.camera_info.serial_number)
        except: SERIAL_NUMBER = None
    except Exception:
        CAMERA_NAME, PRODUCT_LINE, SERIAL_NUMBER = 'Unknown RealSense', '', None

    camera_meta = {
        'camera_name': CAMERA_NAME,
        'camera_product_line': PRODUCT_LINE,
        'camera_serial': SERIAL_NUMBER
    }
    print(f'[INIT] RealSense device: {CAMERA_NAME} | Serial: {SERIAL_NUMBER}')

    # 狀態
    mode = 'IDLE'
    joint_history = deque(maxlen=WINDOW_SIZE)
    stable_frames = 0
    analysis_start_time = None
    post_gap_start = None
    last_patient_seen_time = None
    last_pose_key = None

    last_tail_count_second = None
    speech_play_start = None
    speech_estimated_duration = 0.0

    best_sample = None
    last_pose_change_save_time = 0.0
    auto_pose_change_enabled = AUTO_SAVE_ON_POSE_CHANGE

    last_sacrum_proc = None
    last_heel_proc = None
    last_roi_update_time = 0.0

    fps_start_time = time.time(); fps_frame_count = 0; fps = 0.0

    print('[RUN] Main loop started. ESC to exit.')
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            img_h, img_w = color_img.shape[:2]

            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)

            # --- 產生「乾淨版」masked（只有臉馬賽克，無任何框/文字）
            masked_clean, face_box = anonymize_face_with_box(color_img, results)
            # 畫面顯示 & 即時標註使用的畫布
            display_img = masked_clean.copy()

            has_person = bool(results.pose_landmarks)

            # 穩定度
            metrics = None
            if mode in ['IDLE', 'ANALYSIS'] and has_person:
                lm = results.pose_landmarks.landmark
                frame_pts = [(lm[pid].x, lm[pid].y, lm[pid].visibility) for pid in SELECTED_IDS]
                joint_history.append(frame_pts)
                metrics = compute_stability_metrics(joint_history)
                if metrics and mode == 'IDLE':
                    is_stable = (
                        metrics['avg_vel']     < STABLE_THRESH['avg_vel'] and
                        metrics['max_vel']     < STABLE_THRESH['max_vel'] and
                        metrics['spatial_std'] < STABLE_THRESH['spatial_std']
                    )
                    if is_stable: stable_frames += 1
                    else:
                        stable_frames = 0
                        moved = (metrics['avg_vel']>MOVEMENT_DETECT_THRESH['avg_vel'] or
                                 metrics['max_vel']>MOVEMENT_DETECT_THRESH['max_vel'] or
                                 metrics['spatial_std']>MOVEMENT_DETECT_THRESH['spatial_std'])
                        if moved: speech.event('movement_idle', T('movement_idle', language=LANG_ZH))
            else:
                if mode == 'IDLE': stable_frames = 0

            now_t = time.time()

            # 狀態機
            if mode == 'IDLE':
                if has_person:
                    stable_seconds = stable_frames / FPS_ESTIMATE
                    if stable_seconds >= STABLE_HOLD_SECONDS:
                        mode = 'ANALYSIS'
                        analysis_start_time = now_t
                        last_patient_seen_time = now_t
                        last_tail_count_second = None
                        stable_frames = 0
                        best_sample = {'score': -1.0,
                                       'sacrum_pred': None, 'sacrum_prob':0.0,
                                       'heel_pred': None, 'heel_prob':0.0,
                                       'rgb_masked_clean': None,  # <— 只存乾淨版
                                       'sacrum_proc': None, 'heel_proc': None,
                                       'sacrum_bbox': (0,0,0,0), 'heel_bbox': (0,0,0,0),
                                       'analysis_elapsed': 0.0, 'pose_key': None}
                        speech.event('start_analysis', T('starting_analysis', language=LANG_ZH))
                else:
                    if CLEAR_ROI_WHEN_NO_PERSON:
                        last_sacrum_proc = None; last_heel_proc = None

            elif mode == 'ANALYSIS':
                elapsed_analysis = now_t - analysis_start_time if analysis_start_time else 0.0
                remain_analysis = ANALYSIS_CYCLE_SECONDS - elapsed_analysis

                if not has_person:
                    if last_patient_seen_time and (now_t - last_patient_seen_time > PATIENT_LOST_TIMEOUT):
                        mode = 'IDLE'; analysis_start_time = None; stable_frames = 0; best_sample = None
                        speech.event('return_idle', T('patient_left', language=LANG_ZH))
                else:
                    last_patient_seen_time = now_t

                if has_person and metrics and (
                    metrics['avg_vel']>UNSTABLE_CANCEL['avg_vel'] or
                    metrics['max_vel']>UNSTABLE_CANCEL['max_vel'] or
                    metrics['spatial_std']>UNSTABLE_CANCEL['spatial_std']
                ):
                    mode = 'IDLE'; analysis_start_time = None; best_sample = None
                    last_tail_count_second = None; joint_history.clear(); stable_frames = 0
                    speech.event('movement_detected', T('movement_detected', language=LANG_ZH)); continue

                if 0 < remain_analysis <= (TAIL_COUNTDOWN_SECONDS + 0.2):
                    sec_int = math.ceil(remain_analysis)
                    if 1 <= sec_int <= TAIL_COUNTDOWN_SECONDS and sec_int != last_tail_count_second:
                        last_tail_count_second = sec_int; speech.say(str(sec_int))

                if remain_analysis <= 0:
                    if best_sample and best_sample['rgb_masked_clean'] is not None:
                        save_capture(
                            CLASS_NAMES,
                            best_sample['rgb_masked_clean'],   # <— 傳乾淨版
                            best_sample['sacrum_proc'],
                            best_sample['heel_proc'],
                            best_sample['sacrum_pred'], best_sample['sacrum_prob'],
                            best_sample['heel_pred'], best_sample['heel_prob'],
                            best_sample['sacrum_bbox'], best_sample['heel_bbox'],
                            face_box,
                            quality_dict={'warnings': []},
                            reason='cycle_best',
                            analysis_elapsed=best_sample['analysis_elapsed'],
                            camera_meta=camera_meta
                        )
                        result_text = build_status_sentence_cn(
                            CLASS_NAMES, best_sample['sacrum_pred'], best_sample['heel_pred']
                        ) + f" 將於 {int(GAP_SECONDS)} 秒後進行下一輪偵測。"
                    else:
                        result_text = f"沒有有效樣本。將於 {int(GAP_SECONDS)} 秒後進行下一輪偵測。"
                    speech.event('analysis_end', result_text)
                    speech_play_start = time.time()
                    speech_estimated_duration = max(3.0, len(result_text)/6.0 + 1.5)
                    mode = 'GAP_WAIT_SPEECH'; joint_history.clear(); stable_frames = 0; continue

            elif mode == 'GAP_WAIT_SPEECH':
                if speech_play_start and (now_t - speech_play_start) >= speech_estimated_duration:
                    mode = 'POST_GAP'; post_gap_start = time.time()

            elif mode == 'POST_GAP':
                if (now_t - post_gap_start) >= GAP_SECONDS:
                    mode = 'IDLE'; stable_frames = 0; joint_history.clear()

            # 推論 (僅在 ANALYSIS 且有人)
            current_sacrum_proc = None; current_heel_proc = None
            sacrum_bbox = (0,0,0,0); heel_bbox = (0,0,0,0)
            occupied_label_boxes = []

            if mode == 'ANALYSIS' and has_person:
                lm = results.pose_landmarks.landmark
                def px(i): return int(lm[i].x * img_w), int(lm[i].y * img_h)
                sacrum_pts = [px(mp_pose_module.PoseLandmark.LEFT_SHOULDER.value),
                              px(mp_pose_module.PoseLandmark.RIGHT_SHOULDER.value),
                              px(mp_pose_module.PoseLandmark.LEFT_HIP.value),
                              px(mp_pose_module.PoseLandmark.RIGHT_HIP.value)]
                heel_pts = [px(mp_pose_module.PoseLandmark.LEFT_KNEE.value),
                            px(mp_pose_module.PoseLandmark.RIGHT_KNEE.value),
                            px(mp_pose_module.PoseLandmark.LEFT_HEEL.value),
                            px(mp_pose_module.PoseLandmark.RIGHT_HEEL.value)]
                sx1,sx2,sy1,sy2 = get_roi_bounds(sacrum_pts, img_w=img_w, img_h=img_h)
                hx1,hx2,hy1,hy2 = get_roi_bounds(heel_pts,   img_w=img_w, img_h=img_h)
                sacrum_bbox = (sx1,sy1,sx2,sy2); heel_bbox = (hx1,hy1,hx2,hy2)

                sacrum_roi_depth = depth_img[sy1:sy2, sx1:sx2]
                heel_roi_depth   = depth_img[hy1:hy2, hx1:hx2]
                sacrum_processed = preprocess_depth_roi(sacrum_roi_depth)
                heel_processed   = preprocess_depth_roi(heel_roi_depth)

                sacrum_tensor = prepare_tensor(sacrum_processed, DEVICE)
                heel_tensor   = prepare_tensor(heel_processed,   DEVICE)

                sacrum_pred = sacrum_prob = None
                heel_pred   = heel_prob   = None

                if sacrum_tensor is not None and len(SACRUM_ALLOWED_IDX)>0:
                    sacrum_pred, sacrum_prob, _ = predict_subset(model, sacrum_tensor, SACRUM_ALLOWED_IDX)
                    sacrum_prob2 = round2(sacrum_prob)
                    if sacrum_prob2 is not None:
                        cv2.rectangle(display_img,(sx1,sy1),(sx2,sy2),SACRUM_COLOR,2)
                        draw_label(display_img, sacrum_bbox, face_box, occupied_label_boxes,
                                   f"Sacrum:{CLASS_NAMES[sacrum_pred]} {sacrum_prob2:.2f}", SACRUM_COLOR)

                if heel_tensor is not None and len(HEEL_ALLOWED_IDX)>0:
                    heel_pred, heel_prob, _ = predict_subset(model, heel_tensor, HEEL_ALLOWED_IDX)
                    heel_prob2 = round2(heel_prob)
                    if heel_prob2 is not None:
                        cv2.rectangle(display_img,(hx1,hy1),(hx2,hy2),HEEL_COLOR,2)
                        draw_label(display_img, heel_bbox, face_box, occupied_label_boxes,
                                   f"Heel:{CLASS_NAMES[heel_pred]} {heel_prob2:.2f}", HEEL_COLOR)

                current_sacrum_proc = sacrum_processed; current_heel_proc = heel_processed

                # 立即更新 ROI 顯示
                if sacrum_processed is not None:
                    last_sacrum_proc = sacrum_processed; last_roi_update_time = now_t
                if heel_processed is not None:
                    last_heel_proc = heel_processed; last_roi_update_time = now_t

                # 挑最佳樣本（注意：只存乾淨版）
                if sacrum_pred is not None and heel_pred is not None and metrics:
                    cur_score = calc_frame_score(sacrum_prob, heel_prob, metrics)
                    if best_sample and cur_score > best_sample['score']:
                        best_sample.update({
                            'score': cur_score,
                            'sacrum_pred': sacrum_pred, 'sacrum_prob': sacrum_prob,
                            'heel_pred': heel_pred, 'heel_prob': heel_prob,
                            'rgb_masked_clean': masked_clean.copy(),     # <— 只存乾淨版
                            'sacrum_proc': current_sacrum_proc,
                            'heel_proc': current_heel_proc,
                            'sacrum_bbox': sacrum_bbox,
                            'heel_bbox': heel_bbox,
                            'analysis_elapsed': time.time() - analysis_start_time if analysis_start_time else 0.0,
                            'pose_key': f"{CLASS_NAMES[sacrum_pred]}|{CLASS_NAMES[heel_pred]}"
                        })

                # 姿勢變化自動儲存（同樣傳乾淨版）
                if auto_pose_change_enabled and sacrum_pred is not None and heel_pred is not None:
                    if sacrum_prob is not None and heel_prob is not None and \
                       sacrum_prob >= AUTO_SAVE_CONF and heel_prob >= AUTO_SAVE_CONF:
                        pose_key = f"{CLASS_NAMES[sacrum_pred]}|{CLASS_NAMES[heel_pred]}"
                        if pose_key != last_pose_key and (time.time() - last_pose_change_save_time) >= POSE_CHANGE_MIN_INTERVAL:
                            save_capture(CLASS_NAMES,
                                         masked_clean,                 # <— 傳乾淨版
                                         current_sacrum_proc,
                                         current_heel_proc,
                                         sacrum_pred, sacrum_prob,
                                         heel_pred, heel_prob,
                                         sacrum_bbox, heel_bbox,
                                         face_box,
                                         quality_dict={'warnings': []},
                                         reason='pose_change',
                                         analysis_elapsed=time.time() - analysis_start_time if analysis_start_time else None,
                                         camera_meta=camera_meta)
                            speech.event('pose_change', build_status_sentence_cn(CLASS_NAMES, sacrum_pred, heel_pred) + ' 已自動儲存。')
                            last_pose_key = pose_key; last_pose_change_save_time = time.time()

            # 介面文字（畫在 display_img 上；masked_clean 不被更動）
            camline = f'CAM:{camera_meta.get("camera_name","Unknown")} SN:{camera_meta.get("camera_serial","")}'
            cv2.putText(display_img, camline, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),2)

            if mode == 'IDLE':
                if not has_person:
                    cv2.putText(display_img, T('no_person'), (10, img_h-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)
                else:
                    cv2.putText(display_img, T('waiting'), (10, img_h-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    stable_seconds = stable_frames / FPS_ESTIMATE
                    cv2.putText(display_img, T('stable_progress', cur=stable_seconds, target=STABLE_HOLD_SECONDS),
                                (10, img_h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            elif mode == 'ANALYSIS' and analysis_start_time:
                elapsed = now_t - analysis_start_time; mm = int(elapsed) // 60; ss = int(elapsed) % 60
                cv2.putText(display_img, T('analysis_time', mm=mm, ss=ss),
                            (img_w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,180),2)
                remain_analysis = ANALYSIS_CYCLE_SECONDS - elapsed
                if 0 < remain_analysis <= (TAIL_COUNTDOWN_SECONDS + 0.2):
                    tail_int = math.ceil(remain_analysis)
                    if 1 <= tail_int <= TAIL_COUNTDOWN_SECONDS:
                        text = str(tail_int)
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
                        cx = (img_w - tw)//2; cy = img_h//2
                        cv2.putText(display_img, text, (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,0), 12, cv2.LINE_AA)
                        cv2.putText(display_img, text, (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,255,255), 6, cv2.LINE_AA)

            elif mode == 'GAP_WAIT_SPEECH':
                info = 'Broadcasting result...'
                (tw,th),_ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cx = (img_w - tw)//2; cy = img_h//2
                cv2.putText(display_img, info, (cx,cy), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(display_img, info, (cx,cy), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            elif mode == 'POST_GAP':
                remain = int(max(0, GAP_SECONDS - (now_t - post_gap_start)))
                text = T('post_gap_in', sec=remain)
                (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cx = (img_w - tw)//2; cy = img_h//2
                cv2.putText(display_img, text, (cx,cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5, cv2.LINE_AA)
                cv2.putText(display_img, text, (cx,cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3, cv2.LINE_AA)

            # FPS / MODE
            fps_frame_count += 1
            if (time.time() - fps_start_time) > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time(); fps_frame_count = 0
            cv2.putText(display_img, f'FPS:{fps:.1f}', (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),2)
            cv2.putText(display_img, f'MODE:{mode}', (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)

            # 顯示（畫的是 display_img；masked_clean 永遠不被塗改）
            cv2.imshow('Main', display_img)

            # ROI side-by-side 視窗（與是否畫框無關）
            show_sacrum = last_sacrum_proc; show_heel = last_heel_proc
            if ENABLE_ROI_STALE_TIMEOUT and last_roi_update_time > 0 and (now_t - last_roi_update_time) > ROI_STALE_SECONDS:
                show_sacrum = None; show_heel = None
            depth_left_gray  = show_sacrum if show_sacrum is not None else np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.uint8)
            depth_right_gray = show_heel  if show_heel  is not None else np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.uint8)
            gray_left_vis = cv2.cvtColor(depth_left_gray,  cv2.COLOR_GRAY2BGR)
            gray_right_vis= cv2.cvtColor(depth_right_gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(gray_left_vis, T('sacrum_roi') if show_sacrum is not None else f"{T('sacrum_roi')} : {T('no_data')}",
                        (8,20), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            cv2.putText(gray_right_vis, T('heel_roi') if show_heel is not None else f"{T('heel_roi')} : {T('no_data')}",
                        (12,20), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            cv2.imshow('Depth ROIs (Gray)', np.hstack([gray_left_vis, gray_right_vis]))
            color_left_vis  = cv2.applyColorMap(depth_left_gray,  cv2.COLORMAP_JET)
            color_right_vis = cv2.applyColorMap(depth_right_gray, cv2.COLORMAP_JET)
            cv2.putText(color_left_vis, T('sacrum_roi') if show_sacrum is not None else f"{T('sacrum_roi')} : {T('no_data')}",
                        (8,20), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            cv2.putText(color_right_vis, T('heel_roi') if show_heel is not None else f"{T('heel_roi')} : {T('no_data')}",
                        (12,20), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            cv2.imshow('Depth ROIs (ColorMap)', np.hstack([color_left_vis, color_right_vis]))

            # 熱鍵
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print('Leaving ...'); break
            elif key in (ord('l'), ord('L')):
                OVERLAY_LANGUAGE = LANG_ZH if OVERLAY_LANGUAGE == LANG_EN else LANG_EN
                speech.say(TEXT['lang_switched'][LANG_ZH])
            elif key in (ord('v'), ord('V')):
                speech.enabled = not speech.enabled
                if speech.enabled: speech.say(TEXT['voice_on'][LANG_ZH])
                else: print(TEXT['voice_off'][LANG_ZH])
            elif key in (ord('p'), ord('P')):
                auto_pose_change_enabled = not auto_pose_change_enabled
                speech.say(TEXT['auto_save_on'][LANG_ZH] if auto_pose_change_enabled else TEXT['auto_save_off'][LANG_ZH])

    finally:
        pipeline.stop(); cv2.destroyAllWindows(); print('Resources released.')

if __name__ == '__main__':
    main()
