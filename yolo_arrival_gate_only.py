# yolo_arrival_gate_only.py
# 只做「到場守門」：讀 bed_roi.json、YOLO 數人頭（床內=1、外環>=1），命中後觸發你的模型腳本
# 並把 gate 相關事件送到 FastAPI /gate-events（DB: gate_events）

import os, json, time, collections, subprocess
import numpy as np
import cv2
import requests
from datetime import datetime

# ====== 環境 / API 設定 ======
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
API_TIMEOUT  = 5

def iso_now():
    return datetime.now().isoformat(timespec='seconds')

def post_gate_event(event: str, info: dict):
    """
    把 gate 事件丟到 /gate-events，符合你目前 /docs 的 schema:
      { "event": "<string>", "info": { ...任意鍵值... } }
    失敗不擋流程，只印警告。
    """
    url = f"{API_BASE_URL}/gate-events"
    payload = {"event": event, "info": info}
    try:
        r = requests.post(url, json=payload, timeout=API_TIMEOUT)
        if r.status_code >= 300:
            print(f"[WARN] POST {url} status={r.status_code} body={r.text[:200]}")
        else:
            # 可視需要打開下面這行看回應
            # print("[OK] /gate-events =>", r.text)
            pass
    except Exception as e:
        print("[WARN] gate-events post failed:", e)

# ====== 可調參數 ======
BED_ROI_CONFIG = 'bed_roi.json'

# YOLO
YOLO_WEIGHTS = 'yolov8n.pt'
YOLO_PERSON_CONF = 0.40

# 到場條件
ARRIVAL_STREAK_FRAMES = 8           # 連續命中幀數
BED_PATIENT_REQUIRED = 1            # 床內必須 1（病人）
NURSE_RING_REQUIRED  = 1            # 外環至少 1（護理師）

# 床內/外環擴張（「床邊範圍微微擴大」）
BED_EXTRA_PAD_PCT_X = 0.02          # 床內 ROI 額外往外擴（相對整幅畫面的比例）
BED_EXTRA_PAD_PCT_Y = 0.02
RING_PAD_PCT_X      = 0.15          # 外環相對「床內 ROI（已外擴後）」的擴張比例
RING_PAD_PCT_Y      = 0.15

# 啟動倒數
START_TIMEOUT_SECONDS = 7.0

# 觸發方式（二選一）：
TRIGGER_MODE = 'subprocess'         # 'http' 或 'subprocess'
TRIGGER_HTTP_URL = 'http://127.0.0.1:5055/trigger/start'
TRIGGER_SUBPROCESS_CMD = ['python', 'realtime_9_API.py']
TRIGGER_SUBPROCESS_ENV = os.environ.copy()

# ====== 語音（pyttsx3） ======
import pyttsx3
class Speech:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            try:
                for v in self.engine.getProperty('voices'):
                    vid = (getattr(v, 'id', '') or '').lower()
                    vname = (getattr(v, 'name', '') or '').lower()
                    if 'zh' in vid or 'chinese' in vname or 'huihui' in vname or 'yaoyao' in vname or 'hanhan' in vname:
                        self.engine.setProperty('voice', v.id)
                        break
            except Exception:
                pass
            self.engine.setProperty('rate', 165)
            self.engine.setProperty('volume', 1.0)
            self.enabled = True
        except Exception:
            self.engine = None
            self.enabled = False
    def say(self, text):
        if not self.enabled or self.engine is None: return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass
speech = Speech()

SPEECH_LINES = {
    'opening': '壓瘡部位辨識系統即將開始。請護理人員至床邊，確認患者已在床上並維持穩定姿勢。若七秒內未到位，系統將自動結束。',
    'arrival': '到場條件成立，系統即將啟動分析。',
    'timeout_no_nurse': '系統結束：七秒內未偵測到護理人員到場。',
    'timeout_no_patient': '系統結束：七秒內未偵測到床上患者。',
    'timeout_none': '系統結束：七秒內未偵測到床上患者與護理人員。'
}

# ====== 工具 ======
def load_bed_roi():
    if not os.path.exists(BED_ROI_CONFIG):
        raise FileNotFoundError(f'找不到 {BED_ROI_CONFIG}，請先執行 bed_roi_calibrate.py 進行人工標定')
    with open(BED_ROI_CONFIG, 'r', encoding='utf-8') as f:
        o = json.load(f)
    r = o.get('bed_roi_norm')
    if not r or len(r) != 4:
        raise ValueError(f'{BED_ROI_CONFIG} 格式不對，應有 "bed_roi_norm": [x1,y1,x2,y2]')
    return tuple(map(float, r))

def clamp01(x): return max(0.0, min(1.0, float(x)))

def expand_norm_roi(norm_roi, pad_x, pad_y):
    x1,y1,x2,y2 = norm_roi
    return (clamp01(x1 - pad_x), clamp01(y1 - pad_y), clamp01(x2 + pad_x), clamp01(y2 + pad_y))

def denorm_roi(roi_norm, w, h):
    x1 = int(clamp01(roi_norm[0]) * w)
    y1 = int(clamp01(roi_norm[1]) * h)
    x2 = int(clamp01(roi_norm[2]) * w)
    y2 = int(clamp01(roi_norm[3]) * h)
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def make_ring_from_bed(bed_norm_for_ring, pad_x, pad_y):
    bx1,by1,bx2,by2 = bed_norm_for_ring
    ox1 = clamp01(bx1 - pad_x); oy1 = clamp01(by1 - pad_y)
    ox2 = clamp01(bx2 + pad_x); oy2 = clamp01(by2 + pad_y)
    return (ox1,oy1,ox2,oy2)

def people_centers_from_yolo(yolo_model, bgr, conf=0.4):
    res = yolo_model.predict(bgr, imgsz=640, conf=conf, classes=[0], verbose=False)[0]
    centers = []
    if res.boxes is not None and res.boxes.xyxy is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2),cf in zip(xyxy,confs):
            cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
            centers.append(((cx,cy), cf, (x1,y1,x2,y2)))
    return centers

def count_bed_and_ring(centers, bed_px, ring_px):
    def in_rect(pt, rect):
        (x,y) = pt; (x1,y1,x2,y2) = rect
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    bed_cnt = 0; ring_cnt = 0
    for (cx,cy),cf,box in centers:
        if in_rect((cx,cy), bed_px): bed_cnt += 1
        elif in_rect((cx,cy), ring_px): ring_cnt += 1
    bed_cnt = 1 if bed_cnt >= 1 else 0  # 床上病人只接受 0/1
    return bed_cnt, ring_cnt

# ====== 觸發你的模型 ======
def trigger_model():
    if TRIGGER_MODE == 'http':
        try:
            requests.post(TRIGGER_HTTP_URL, json={"timeout": 60}, timeout=5)
            print('[TRIGGER] HTTP start sent ->', TRIGGER_HTTP_URL)
        except Exception as e:
            print('[TRIGGER][HTTP] 失敗：', e)
    elif TRIGGER_MODE == 'subprocess':
        print('[TRIGGER] 以 subprocess 啟動你的模型：', ' '.join(TRIGGER_SUBPROCESS_CMD))
        return subprocess.run(TRIGGER_SUBPROCESS_CMD, env=TRIGGER_SUBPROCESS_ENV).returncode
    else:
        print('[TRIGGER] 未定義的 TRIGGER_MODE:', TRIGGER_MODE)

# ====== 主程式 ======
def main():
    # 1) 讀 ROI
    bed_roi_norm = load_bed_roi()
    bed_norm_expanded = expand_norm_roi(bed_roi_norm, BED_EXTRA_PAD_PCT_X, BED_EXTRA_PAD_PCT_Y)
    print('[INIT] 讀到床位 ROI (norm)=', bed_roi_norm, ' | 擴張後=', bed_norm_expanded)

    # 2) YOLO
    from ultralytics import YOLO
    yolo_person = YOLO(YOLO_WEIGHTS)
    print('[INIT] YOLO ready')

    # 3) 影像來源：RealSense（彩色）
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    print('[INIT] RealSense color stream started')

    # 相機資訊
    try:
        dev = profile.get_device()
        try: camera_name = dev.get_info(rs.camera_info.name)
        except: camera_name = 'Unknown RealSense'
        try: camera_serial = dev.get_info(rs.camera_info.serial_number)
        except: camera_serial = None
    except Exception:
        camera_name, camera_serial = 'Unknown RealSense', None

    # 開場語音 + 記一筆 gate 開啟
    speech.say(SPEECH_LINES['opening'])
    post_gate_event('gate_opening', {
        "timestamp_iso": iso_now(),
        "camera_name": camera_name,
        "camera_serial": camera_serial,
        "bed_roi_norm": bed_roi_norm,
        "bed_roi_norm_expanded": bed_norm_expanded,
        "ring_from_bed_pad": [RING_PAD_PCT_X, RING_PAD_PCT_Y],
        "mode": TRIGGER_MODE,
        "cmd": " ".join(TRIGGER_SUBPROCESS_CMD) if TRIGGER_MODE == 'subprocess' else TRIGGER_HTTP_URL
    })

    hits = collections.deque(maxlen=ARRIVAL_STREAK_FRAMES)
    waiting_after_subprocess = False
    start_ts = time.time()
    arrival_confirmed_once = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            h, w = bgr.shape[:2]

            # ROI（像素）
            bx1,by1,bx2,by2 = denorm_roi(bed_norm_expanded, w, h)
            outer_norm = make_ring_from_bed(bed_norm_expanded, RING_PAD_PCT_X, RING_PAD_PCT_Y)
            ox1,oy1,ox2,oy2 = denorm_roi(outer_norm, w, h)

            # YOLO 偵測
            centers = people_centers_from_yolo(yolo_person, bgr, conf=YOLO_PERSON_CONF)
            bed_cnt, ring_cnt = count_bed_and_ring(centers, (bx1,by1,bx2,by2), (ox1,oy1,ox2,oy2))

            nurse_present = (ring_cnt >= NURSE_RING_REQUIRED)
            bed_present   = (bed_cnt == BED_PATIENT_REQUIRED)

            hit = int(bed_present and nurse_present)
            hits.append(hit)
            if hit:
                arrival_confirmed_once = True

            # 視覺化
            vis = bgr.copy()
            cv2.rectangle(vis,(ox1,oy1),(ox2,oy2),(0,160,255),1)
            cv2.rectangle(vis,(bx1,by1),(bx2,by2),(0,220,255),2)
            cv2.putText(vis, f'Bed:{bed_cnt}  Ring:{ring_cnt}', (bx1+6, max(24,by1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,255),2)
            for (cx,cy),cf,(x1,y1,x2,y2) in centers:
                cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(200,200,200),1)
            cv2.putText(vis, f'Streak {sum(hits)}/{len(hits)} (need {ARRIVAL_STREAK_FRAMES})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            elapsed = time.time() - start_ts
            remain = max(0.0, START_TIMEOUT_SECONDS - elapsed)
            cv2.putText(vis, f'Countdown: {remain:0.1f}s', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if waiting_after_subprocess:
                cv2.putText(vis, 'Model finished. Press SPACE to resume gate, or Q to quit.',
                            (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow('YOLO Arrival Gate', vis)

            # ---- 到場成立（連續命中）→ 觸發 ----
            if len(hits) == ARRIVAL_STREAK_FRAMES and sum(hits) == ARRIVAL_STREAK_FRAMES and not waiting_after_subprocess:
                print('[ARRIVAL] 條件成立（床內=1、外環>=1 且連續命中）')
                speech.say(SPEECH_LINES['arrival'])

                # 記一筆 arrival_confirmed
                post_gate_event('arrival_confirmed', {
                    "timestamp_iso": iso_now(),
                    "camera_name": camera_name,
                    "camera_serial": camera_serial,
                    "bed_count": bed_cnt,
                    "ring_count": ring_cnt,
                    "streak": int(sum(hits)),
                    "streak_need": ARRIVAL_STREAK_FRAMES,
                    "bed_required": BED_PATIENT_REQUIRED,
                    "ring_required": NURSE_RING_REQUIRED
                })

                # 先停相機，避免後面模型也要用相機造成衝突
                pipeline.stop()

                # 記 subprocess_start
                if TRIGGER_MODE == 'subprocess':
                    post_gate_event('subprocess_start', {
                        "timestamp_iso": iso_now(),
                        "mode": TRIGGER_MODE, "cmd": " ".join(TRIGGER_SUBPROCESS_CMD)
                    })
                ret = trigger_model()
                if TRIGGER_MODE == 'subprocess':
                    print('[ARRIVAL] 模型子行程結束，return code =', ret)
                    post_gate_event('subprocess_end', {
                        "timestamp_iso": iso_now(),
                        "mode": TRIGGER_MODE, "cmd": " ".join(TRIGGER_SUBPROCESS_CMD),
                        "return_code": ret
                    })
                    waiting_after_subprocess = True
                    # 重新開相機，讓你可以繼續看畫面
                    pipeline = rs.pipeline()
                    pipeline.start(config)
                    hits.clear()

            # ---- 7 秒逾時：判斷原因並結束 ----
            if elapsed >= START_TIMEOUT_SECONDS and not arrival_confirmed_once and not waiting_after_subprocess:
                if bed_present and not nurse_present:
                    speech.say(SPEECH_LINES['timeout_no_nurse'])
                    post_gate_event('timeout_no_nurse', {
                        "timestamp_iso": iso_now(),
                        "camera_name": camera_name,
                        "camera_serial": camera_serial,
                        "elapsed": elapsed,
                        "bed_count": bed_cnt, "ring_count": ring_cnt,
                        "bed_required": BED_PATIENT_REQUIRED, "ring_required": NURSE_RING_REQUIRED
                    })
                elif (not bed_present) and nurse_present:
                    speech.say(SPEECH_LINES['timeout_no_patient'])
                    post_gate_event('timeout_no_patient', {
                        "timestamp_iso": iso_now(),
                        "camera_name": camera_name,
                        "camera_serial": camera_serial,
                        "elapsed": elapsed,
                        "bed_count": bed_cnt, "ring_count": ring_cnt,
                        "bed_required": BED_PATIENT_REQUIRED, "ring_required": NURSE_RING_REQUIRED
                    })
                elif (not bed_present) and (not nurse_present):
                    speech.say(SPEECH_LINES['timeout_none'])
                    post_gate_event('timeout_none', {
                        "timestamp_iso": iso_now(),
                        "camera_name": camera_name,
                        "camera_serial": camera_serial,
                        "elapsed": elapsed,
                        "bed_count": bed_cnt, "ring_count": ring_cnt,
                        "bed_required": BED_PATIENT_REQUIRED, "ring_required": NURSE_RING_REQUIRED
                    })
                else:
                    speech.say(SPEECH_LINES['timeout_none'])
                    post_gate_event('timeout_none', {
                        "timestamp_iso": iso_now(),
                        "camera_name": camera_name,
                        "camera_serial": camera_serial,
                        "elapsed": elapsed,
                        "bed_count": bed_cnt, "ring_count": ring_cnt,
                        "bed_required": BED_PATIENT_REQUIRED, "ring_required": NURSE_RING_REQUIRED
                    })
                print('[TIMEOUT] 7 秒內未達成到場條件，程式結束。')
                time.sleep(0.3)
                break

            # 鍵盤控制
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if waiting_after_subprocess and key == 32:  # SPACE 繼續
                waiting_after_subprocess = False
                hits.clear()
                start_ts = time.time()
                arrival_confirmed_once = False

    finally:
        try: pipeline.stop()
        except: pass
        cv2.destroyAllWindows()
        post_gate_event('gate_quit', {
            "timestamp_iso": iso_now(),
            "camera_name": camera_name,
            "camera_serial": camera_serial
        })
        print('[CLEAN] Gate stopped.')

if __name__ == '__main__':
    main()
