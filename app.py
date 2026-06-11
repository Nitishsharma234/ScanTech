import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import time
import json
import yaml
import os
import threading
from pathlib import Path
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoScan AI · Vehicle Damage Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0B0E14; color: #E2E8F0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem 2.5rem; }

.hero {
    background: linear-gradient(135deg, #0F1923 0%, #1A2332 50%, #0F1923 100%);
    border: 1px solid #1E3A5F; border-radius: 16px;
    padding: 2.8rem 3rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,122,255,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family:'Space Grotesk',sans-serif; font-size:0.72rem; font-weight:600;
    letter-spacing:0.20em; text-transform:uppercase; color:#3B82F6; margin-bottom:0.7rem;
}
.hero-title {
    font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:700;
    color:#F0F6FF; line-height:1.15; margin-bottom:0.8rem;
}
.hero-sub { font-size:0.95rem; color:#8BA0B8; max-width:560px; line-height:1.7; }
.hero-badges { margin-top:1.4rem; display:flex; gap:0.6rem; flex-wrap:wrap; }
.hero-badge {
    display:inline-flex; align-items:center; gap:0.35rem;
    background:#111827; border:1px solid #1E3A5F; border-radius:20px;
    padding:0.3rem 0.85rem; font-size:0.75rem; color:#8BA0B8; font-weight:500;
}

section[data-testid="stSidebar"] { background:#0D1520; border-right:1px solid #1E3A5F; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family:'Space Grotesk',sans-serif; color:#3B82F6;
    font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase; font-weight:600;
}

[data-testid="stFileUploader"] {
    background:#0F1923; border:2px dashed #1E3A5F;
    border-radius:12px; transition:border-color 0.25s;
}
[data-testid="stFileUploader"]:hover { border-color:#3B82F6; }
[data-testid="stFileUploader"] label { color:#8BA0B8 !important; font-size:0.9rem; }

.status-card { border-radius:12px; padding:1.4rem 1.8rem; display:flex; align-items:flex-start; gap:1.1rem; margin-bottom:1.2rem; }
.status-good { background:linear-gradient(135deg,#052E16 0%,#064E1F 100%); border:1px solid #16A34A; }
.status-bad  { background:linear-gradient(135deg,#2D0A0A 0%,#4A0E0E 100%); border:1px solid #DC2626; }
.status-warn { background:linear-gradient(135deg,#2D1800 0%,#4A2800 100%); border:1px solid #D97706; }
.status-icon { font-size:2.1rem; line-height:1; flex-shrink:0; }
.status-label { font-family:'Space Grotesk',sans-serif; font-size:0.68rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.25rem; }
.status-label-good { color:#4ADE80; }
.status-label-bad  { color:#F87171; }
.status-label-warn { color:#FCD34D; }
.status-verdict { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; line-height:1.2; }
.status-verdict-good { color:#86EFAC; }
.status-verdict-bad  { color:#FECACA; }
.status-verdict-warn { color:#FDE68A; }
.status-desc { font-size:0.83rem; color:#94A3B8; margin-top:0.35rem; line-height:1.55; }

.metric-row { display:flex; gap:0.9rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.metric-tile {
    flex:1; min-width:120px; background:#0F1923;
    border:1px solid #1E3A5F; border-radius:10px; padding:1rem 1.2rem;
    transition: border-color 0.2s;
}
.metric-tile:hover { border-color:#2D5A9E; }
.metric-tile-label { font-size:0.67rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#4B6580; margin-bottom:0.35rem; }
.metric-tile-value { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:#E2E8F0; line-height:1; }
.metric-tile-value.accent { color:#3B82F6; }

.det-table { width:100%; border-collapse:collapse; font-size:0.84rem; margin-top:0.6rem; }
.det-table th {
    text-align:left; font-size:0.67rem; font-weight:600;
    letter-spacing:0.12em; text-transform:uppercase; color:#4B6580;
    padding:0.55rem 0.9rem; border-bottom:1px solid #1E3A5F;
}
.det-table td { padding:0.6rem 0.9rem; color:#CBD5E1; border-bottom:1px solid #111827; }
.det-table tr:last-child td { border-bottom: none; }
.det-table tr:hover td { background:#0F1923; }

.conf-pill { display:inline-block; padding:0.15rem 0.65rem; border-radius:20px; font-size:0.78rem; font-weight:600; }
.conf-high { background:#3D1010; color:#F87171; border:1px solid #DC2626; }
.conf-mid  { background:#2D2000; color:#FCD34D; border:1px solid #D97706; }
.conf-low  { background:#0D2010; color:#86EFAC; border:1px solid #16A34A; }

.section-heading {
    font-family:'Space Grotesk',sans-serif; font-size:0.7rem; font-weight:600;
    letter-spacing:0.18em; text-transform:uppercase; color:#3B82F6;
    margin-bottom:0.8rem; padding-bottom:0.4rem; border-bottom:1px solid #1E3A5F;
}
.severity-bar-wrap { background:#111827; border-radius:8px; overflow:hidden; height:10px; margin-top:0.5rem; }
.severity-bar { height:10px; border-radius:8px; transition:width 0.5s ease; }

.stButton > button {
    background:#1D4ED8; color:#F0F6FF; border:none; border-radius:8px;
    font-family:'Space Grotesk',sans-serif; font-weight:600;
    padding:0.55rem 1.5rem; transition:background 0.2s, transform 0.1s;
    letter-spacing:0.02em;
}
.stButton > button:hover { background:#2563EB; color:#fff; transform:translateY(-1px); }
.stButton > button:active { transform:translateY(0); }

.preprocess-badge {
    display:inline-block; background:#0F2A1A; border:1px solid #16A34A;
    border-radius:6px; padding:0.2rem 0.65rem; font-size:0.71rem;
    color:#4ADE80; margin-right:0.4rem; margin-bottom:0.3rem; font-weight:500;
}

.ft-status-running {
    background:#0D1F0D; border:1px solid #16A34A; border-radius:10px;
    padding:0.9rem 1.1rem; font-size:0.82rem; color:#4ADE80; margin-top:0.5rem;
}
.ft-status-done {
    background:#0F2A1A; border:1px solid #16A34A; border-radius:10px;
    padding:0.9rem 1.1rem; font-size:0.82rem; color:#4ADE80; margin-top:0.5rem;
}
.divider { border:none; border-top:1px solid #1E3A5F; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Background Fine-Tune Session State ───────────────────────────────────────
if "ft_thread"       not in st.session_state: st.session_state.ft_thread       = None
if "ft_status"       not in st.session_state: st.session_state.ft_status       = "idle"
if "ft_new_weights"  not in st.session_state: st.session_state.ft_new_weights  = None
if "ft_error_msg"    not in st.session_state: st.session_state.ft_error_msg    = None
if "ft_start_time"   not in st.session_state: st.session_state.ft_start_time   = None

# ─── Poll background thread result ────────────────────────────────────────────
holder = st.session_state.get("ft_state_holder")
if holder and st.session_state.ft_status == "running":
    if holder["status"] == "done":
        st.session_state.ft_status       = "done"
        st.session_state.ft_new_weights  = holder["new_weights"]
        st.session_state.ft_state_holder = None
    elif holder["status"] == "error":
        st.session_state.ft_status       = "error"
        st.session_state.ft_error_msg    = holder["error_msg"]
        st.session_state.ft_state_holder = None

# ─── Active Learning Config ────────────────────────────────────────────────────
PROJECT_ROOT     = "."
FEEDBACK_DIR     = os.path.join(PROJECT_ROOT, "feedback_data")
FEEDBACK_IMAGES  = os.path.join(FEEDBACK_DIR, "images")
FEEDBACK_LABELS  = os.path.join(FEEDBACK_DIR, "labels")
FEEDBACK_CROPS   = os.path.join(FEEDBACK_DIR, "crops")
FEEDBACK_LOG     = os.path.join(FEEDBACK_DIR, "feedback_log.json")
FINETUNE_TRIGGER = 10
CONF_THRESHOLD   = 0.40   # fixed — not exposed to UI

def _ensure_feedback_dirs():
    for d in [FEEDBACK_IMAGES, FEEDBACK_LABELS, FEEDBACK_CROPS]:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "w") as f:
            json.dump([], f)

def _load_feedback_log():
    _ensure_feedback_dirs()
    try:
        with open(FEEDBACK_LOG) as f:
            return json.load(f)
    except Exception:
        return []

def _save_feedback_log(log):
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log, f, indent=2)

def _count_corrections():
    if not os.path.exists(FEEDBACK_CROPS):
        return 0
    return len([f for f in os.listdir(FEEDBACK_CROPS) if f.endswith(".txt")])

def build_feedback_yaml(class_names_dict):
    path = os.path.join(FEEDBACK_DIR, "feedback_data.yaml")
    cfg  = {
        "path":  FEEDBACK_DIR,
        "train": "crops",
        "val":   "crops",
        "names": {int(k): v for k, v in class_names_dict.items()},
        "nc":    len(class_names_dict),
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path

def run_finetune(base_weights, class_names_dict):
    from ultralytics import YOLO as _YOLO
    crop_count = _count_corrections()
    if crop_count == 0:
        raise ValueError("No correction crops found.")
    ft_model      = _YOLO(base_weights)
    feedback_yaml = build_feedback_yaml(class_names_dict)
    run_name      = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ft_model.train(
        data=feedback_yaml,
        epochs=15,
        imgsz=640,
        batch=max(1, min(4, crop_count)),
        device=0,
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),
        name=run_name,
        workers=0,
        freeze=10,
        cache=False,
    )
    return os.path.join(PROJECT_ROOT, "runs", "detect", run_name, "weights", "best.pt")

def _finetune_worker(base_weights, class_names_dict, state_holder):
    try:
        new_weights             = run_finetune(base_weights, class_names_dict)
        state_holder["status"]      = "done"
        state_holder["new_weights"] = new_weights
    except Exception as e:
        state_holder["status"]    = "error"
        state_holder["error_msg"] = str(e)

def save_correction(img_bgr, boxes_yolo, class_names, source_label="upload", model_weights=""):
    _ensure_feedback_dirs()
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = f"correction_{ts}"
    h_img, w_img = img_bgr.shape[:2]

    cv2.imwrite(os.path.join(FEEDBACK_IMAGES, save_name + ".jpg"), img_bgr)
    with open(os.path.join(FEEDBACK_LABELS, save_name + ".txt"), "w") as f:
        for (cls_id, cx, cy, bw, bh) in boxes_yolo:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    crop_records = []
    for idx, (cls_id, cx, cy, bw, bh) in enumerate(boxes_yolo):
        x1 = int((cx - bw / 2) * w_img);  y1 = int((cy - bh / 2) * h_img)
        x2 = int((cx + bw / 2) * w_img);  y2 = int((cy + bh / 2) * h_img)
        pad_x = max(int((x2-x1)*0.20), 10); pad_y = max(int((y2-y1)*0.20), 10)
        rx1 = max(0,x1-pad_x); ry1 = max(0,y1-pad_y)
        rx2 = min(w_img,x2+pad_x); ry2 = min(h_img,y2+pad_y)
        crop = img_bgr[ry1:ry2, rx1:rx2]
        crop_h, crop_w = crop.shape[:2]
        scale  = 640 / max(crop_h, crop_w)
        new_w  = int(crop_w*scale); new_h = int(crop_h*scale)
        canvas = np.zeros((640,640,3), dtype=np.uint8)
        off_x  = (640-new_w)//2;    off_y = (640-new_h)//2
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = cv2.resize(crop,(new_w,new_h))
        bx1r = (x1-rx1)*scale+off_x; by1r = (y1-ry1)*scale+off_y
        bx2r = (x2-rx1)*scale+off_x; by2r = (y2-ry1)*scale+off_y
        new_cx = ((bx1r+bx2r)/2)/640; new_cy = ((by1r+by2r)/2)/640
        new_bw = (bx2r-bx1r)/640;     new_bh = (by2r-by1r)/640
        crop_name = f"{save_name}_crop{idx}"
        cv2.imwrite(os.path.join(FEEDBACK_CROPS, crop_name+".jpg"), canvas)
        with open(os.path.join(FEEDBACK_CROPS, crop_name+".txt"), "w") as f:
            f.write(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n")
        crop_records.append({"cls_id":cls_id,"cls_name":class_names.get(cls_id,str(cls_id)),"crop_file":crop_name})

    log = _load_feedback_log()
    log.append({"timestamp":ts,"source":source_label,"saved_as":save_name,
                "corrections":crop_records,"model_weights":model_weights})
    _save_feedback_log(log)
    return _count_corrections()

# ─── Inference Helpers ─────────────────────────────────────────────────────────
def run_tta_inference(model, img_bgr):
    h, w = img_bgr.shape[:2]
    augmented = [
        img_bgr,
        cv2.flip(img_bgr, 1),
        cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=20),
        cv2.convertScaleAbs(img_bgr, alpha=0.75, beta=0),
    ]
    all_boxes = []
    for idx, aug in enumerate(augmented):
        results = model.predict(source=aug, conf=CONF_THRESHOLD, save=False, verbose=False)
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        aug_h, aug_w = aug.shape[:2]
        for (x1,y1,x2,y2), conf, cls_id in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int)
        ):
            if idx == 1:
                x1, x2 = aug_w - x2, aug_w - x1
            sx = w / aug_w; sy = h / aug_h
            all_boxes.append((int(cls_id), float(conf),
                               float(x1*sx), float(y1*sy), float(x2*sx), float(y2*sy)))
    if not all_boxes:
        return []
    return _nms(all_boxes, iou_thresh=0.40)

def run_sliding_window(model, img_bgr, tile_size=480, overlap=0.3):
    h, w  = img_bgr.shape[:2]
    step  = int(tile_size * (1 - overlap))
    boxes = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2t = min(x+tile_size,w); y2t = min(y+tile_size,h)
            x1t = max(0,x2t-tile_size); y1t = max(0,y2t-tile_size)
            tile = img_bgr[y1t:y2t, x1t:x2t]
            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue
            results = model.predict(source=tile, conf=CONF_THRESHOLD, save=False, verbose=False)
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            th, tw = tile.shape[:2]
            for (bx1,by1,bx2,by2), conf, cls_id in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int)
            ):
                if (bx2-bx1)*(by2-by1) / (tw*th) > 0.70: continue
                if (bx2-bx1) < 10 or (by2-by1) < 10: continue
                boxes.append((int(cls_id), float(conf),
                               float(x1t+bx1), float(y1t+by1),
                               float(x1t+bx2), float(y1t+by2)))
    if not boxes:
        return []
    return _nms(boxes, iou_thresh=0.35)

def _nms(boxes, iou_thresh=0.40):
    boxes.sort(key=lambda b: -b[1])
    kept = []; suppressed = [False]*len(boxes)
    for i, bi in enumerate(boxes):
        if suppressed[i]: continue
        kept.append(bi)
        ai = (bi[4]-bi[2])*(bi[5]-bi[3])
        for j in range(i+1, len(boxes)):
            if suppressed[j]: continue
            bj = boxes[j]
            if bi[0] != bj[0]: continue
            ix1=max(bi[2],bj[2]); iy1=max(bi[3],bj[3])
            ix2=min(bi[4],bj[4]); iy2=min(bi[5],bj[5])
            inter=max(0,ix2-ix1)*max(0,iy2-iy1)
            aj=(bj[4]-bj[2])*(bj[5]-bj[3])
            union=ai+aj-inter
            if union > 0 and inter/union > iou_thresh:
                suppressed[j] = True
    return kept

def filter_garbage_boxes(detections, img_shape, max_area_ratio=0.50):
    h, w  = img_shape[:2]; total = h*w
    good  = []
    for det in detections:
        cls_id, conf, x1, y1, x2, y2 = det
        bw = x2-x1; bh = y2-y1
        if (bw*bh)/total <= max_area_ratio and bw >= w*0.01 and bh >= h*0.01:
            good.append(det)
    return good

def compute_severity(detections, img_shape):
    if not detections:
        return 0.0, "None"
    h, w = img_shape[:2]; total_px = h*w
    confs = [d[1] for d in detections]
    areas = [(d[4]-d[2])*(d[5]-d[3]) for d in detections]
    area_ratio   = min(sum(areas)/total_px, 1.0)
    avg_conf     = float(np.mean(confs))
    count_factor = min(len(detections)/5.0, 1.0)
    score = (0.4*avg_conf + 0.35*area_ratio + 0.25*count_factor)*100
    label = "Severe" if score >= 60 else ("Moderate" if score >= 35 else "Minor")
    return round(score, 1), label

def draw_boxes(img_bgr, detections, names):
    out = img_bgr.copy()
    colors = {"high":(60,60,220), "mid":(30,165,255), "low":(60,200,60)}
    for cls_id, conf, x1, y1, x2, y2 in detections:
        label = names.get(int(cls_id), str(cls_id)).replace("_"," ").title()
        tier  = "high" if conf >= 0.65 else ("mid" if conf >= 0.40 else "low")
        color = colors[tier]
        x1i,y1i,x2i,y2i = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(out, (x1i,y1i), (x2i,y2i), color, 2)
        txt = f"{label}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(out, (x1i, y1i-th-8), (x1i+tw+6, y1i), color, -1)
        cv2.putText(out, txt, (x1i+3, y1i-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    return out

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    st.markdown("---")

    model_path = st.text_input(
        "Model Weights (.pt)",
        value="best.pt",
        help="Absolute path to your trained YOLOv8 weights file",
    )

    st.markdown("### 🎚 Detection Settings")
    use_tta = st.toggle("Test-Time Augmentation", value=True,
                        help="Runs 4 augmented variants and merges results for higher recall.")
    use_sliding = st.toggle("Sliding Window Analysis", value=True,
                            help="Tiles the image to detect small, localized damage regions.")
    tile_size = st.select_slider("Tile Size (px)", options=[320, 480, 640], value=480)

    st.markdown("---")
    st.markdown("### 📋 Confidence Scale")
    st.markdown("""
<small style='color:#8BA0B8; line-height:2.0'>
🔴 <b style='color:#F87171'>High  ≥ 65%</b><br>
🟡 <b style='color:#FCD34D'>Mid   40–65%</b><br>
🟢 <b style='color:#4ADE80'>Low   &lt; 40%</b>
</small>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Auto Fine-Tune Status ──────────────────────────────────────────────────
    st.markdown("### 🔁 Model Learning Status")
    correction_count = _count_corrections()
    next_trigger = FINETUNE_TRIGGER - (correction_count % FINETUNE_TRIGGER) if correction_count % FINETUNE_TRIGGER != 0 else FINETUNE_TRIGGER
    st.caption(f"Corrections saved: **{correction_count}**")
    ft_status = st.session_state.ft_status

    if ft_status == "idle":
        if correction_count > 0:
            st.caption(f"Next auto-retrain in **{next_trigger}** more correction(s).")
        else:
            st.caption("Submit corrections on any image to begin.")

    elif ft_status == "running":
        elapsed = (datetime.now() - st.session_state.ft_start_time).seconds
        st.markdown(f"""
<div class="ft-status-running">
    ⚙️ <b>Fine-tuning in background…</b><br>
    <span style="color:#94A3B8; font-size:0.78rem;">Elapsed: {elapsed}s · App remains fully usable</span>
</div>""", unsafe_allow_html=True)
        time.sleep(0)
        st.rerun()

    elif ft_status == "done":
        new_w = st.session_state.ft_new_weights
        st.markdown(f"""
<div class="ft-status-done">
    ✅ <b>Fine-tune complete!</b><br>
    <span style="color:#94A3B8; font-size:0.76rem;">Model auto-switched to new weights.</span><br>
    <span style="color:#4B6580; font-size:0.71rem; word-break:break-all">{new_w}</span>
</div>""", unsafe_allow_html=True)
        st.success("Re-upload an image to see improved predictions.")
        if st.button("Reset Status", key="ft_reset"):
            st.session_state.ft_status      = "idle"
            st.session_state.ft_new_weights = None
            st.rerun()

    elif ft_status == "error":
        st.error(f"Fine-tune failed:\n{st.session_state.ft_error_msg}")
        if st.button("Reset", key="ft_err_reset"):
            st.session_state.ft_status = "idle"
            st.rerun()

    st.markdown("---")
    st.caption("AutoScan AI  ·  YOLOv8 + TTA + Active Learning")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Visual Inspection System</div>
    <div class="hero-title">AutoScan Vehicle Damage Detection</div>
    <div class="hero-sub">
        Upload a vehicle photograph to receive an instant, detailed damage assessment —
        including annotated bounding boxes, per-region confidence scores, a composite
        severity rating, and a structured breakdown of all detected damage areas.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">🔍 YOLOv8 Detection</span>
        <span class="hero-badge">🔄 Test-Time Augmentation</span>
        <span class="hero-badge">🪟 Sliding Window</span>
        <span class="hero-badge">🧠 Active Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

if st.session_state.ft_status == "done" and st.session_state.ft_new_weights:
    active_model_path = st.session_state.ft_new_weights
    if st.session_state.get("_loaded_weights") != active_model_path:
        load_model.clear()
        st.session_state["_loaded_weights"] = active_model_path
else:
    active_model_path = model_path

model = None
try:
    with st.spinner("Loading model weights…"):
        model = load_model(active_model_path)
except Exception as e:
    st.error(f"**Model failed to load.** Verify the path in the sidebar.\n\n`{e}`")

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Upload Vehicle Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

# ─── Inference & Results ──────────────────────────────────────────────────────
if uploaded and model:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    orig_img   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    preprocess_tags = []

    with st.spinner("Analysing image…"):
        t0 = time.time()

        if use_tta:
            raw_detections = run_tta_inference(model, orig_img)
            preprocess_tags.append("TTA · 4× augments")
        else:
            results = model.predict(source=orig_img, conf=CONF_THRESHOLD, save=False, verbose=False)
            res = results[0]
            raw_detections = []
            if res.boxes is not None and len(res.boxes) > 0:
                for (x1,y1,x2,y2), conf, cls_id in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    res.boxes.cls.cpu().numpy().astype(int)
                ):
                    raw_detections.append((int(cls_id), float(conf),
                                           float(x1), float(y1), float(x2), float(y2)))

        good_dets = filter_garbage_boxes(raw_detections, orig_img.shape)

        if use_sliding:
            sliding_dets = run_sliding_window(model, orig_img, tile_size=tile_size)
            if sliding_dets:
                merged = good_dets + sliding_dets
                good_dets = _nms(merged, iou_thresh=0.35)
                preprocess_tags.append(f"Sliding Window · {tile_size}px tiles")

        raw_detections = good_dets
        elapsed = time.time() - t0

    # Get class names
    _r   = model.predict(source=orig_img[:10,:10], conf=0.9, save=False, verbose=False)
    names = _r[0].names
    st.session_state["names"] = names

    # Removed-box state per image
    img_key = f"{uploaded.name}_{orig_img.shape}"
    if st.session_state.get("_last_img_key") != img_key:
        st.session_state.removed_det_indices = set()
        st.session_state._last_img_key = img_key
    if "removed_det_indices" not in st.session_state:
        st.session_state.removed_det_indices = set()

    active_detections = [d for i,d in enumerate(raw_detections)
                         if i not in st.session_state.removed_det_indices]

    damage_detected               = len(active_detections) > 0
    severity_score, severity_label = compute_severity(active_detections, orig_img.shape)
    max_conf  = max([d[1] for d in active_detections]) if damage_detected else 0.0
    det_count = len(active_detections)
    h, w      = orig_img.shape[:2]
    infer_ms  = f"{elapsed*1000:.0f} ms"

    # ── Preprocessing badges ───────────────────────────────────────────────────
    if preprocess_tags:
        badges = "".join(f'<span class="preprocess-badge">{t}</span>' for t in preprocess_tags)
        st.markdown(f'<div style="margin-bottom:1.2rem">{badges}</div>', unsafe_allow_html=True)

    # ── Metric tiles ───────────────────────────────────────────────────────────
    sev_color = "#F87171" if severity_label=="Severe" else ("#FCD34D" if severity_label=="Moderate" else "#4ADE80")
    st.markdown(f"""
<div class="metric-row">
    <div class="metric-tile">
        <div class="metric-tile-label">Detections</div>
        <div class="metric-tile-value {'accent' if det_count > 0 else ''}">{det_count}</div>
    </div>
    <div class="metric-tile">
        <div class="metric-tile-label">Peak Confidence</div>
        <div class="metric-tile-value">{max_conf:.2f}</div>
    </div>
    <div class="metric-tile">
        <div class="metric-tile-label">Severity Score</div>
        <div class="metric-tile-value" style="color:{sev_color}">{severity_score}</div>
    </div>
    <div class="metric-tile">
        <div class="metric-tile-label">Severity Level</div>
        <div class="metric-tile-value" style="color:{sev_color}; font-size:1.1rem; margin-top:4px">{severity_label}</div>
    </div>
    <div class="metric-tile">
        <div class="metric-tile-label">Inference Time</div>
        <div class="metric-tile-value">{infer_ms}</div>
    </div>
    <div class="metric-tile">
        <div class="metric-tile-label">Resolution</div>
        <div class="metric-tile-value" style="font-size:1rem; margin-top:4px">{w}×{h}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Severity bar ───────────────────────────────────────────────────────────
    bar_color = "#EF4444" if severity_score>=60 else ("#F59E0B" if severity_score>=35 else "#22C55E")
    st.markdown(f"""
<div style="margin-bottom:1.4rem">
    <div style="font-size:0.7rem; color:#4B6580; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.35rem">
        Overall Damage Severity
    </div>
    <div class="severity-bar-wrap">
        <div class="severity-bar" style="width:{severity_score}%; background:{bar_color}"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:0.3rem">
        <span style="font-size:0.72rem; color:#4B6580">{severity_score} / 100</span>
        <span style="font-size:0.72rem; color:{sev_color}; font-weight:600">{severity_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Verdict card ───────────────────────────────────────────────────────────
    if damage_detected:
        if severity_label == "Severe":
            card_cls, lbl_cls, ver_cls, icon = "status-bad",  "status-label-bad",  "status-verdict-bad",  "🚨"
            verdict = "Severe Damage Detected"
        elif severity_label == "Moderate":
            card_cls, lbl_cls, ver_cls, icon = "status-warn", "status-label-warn", "status-verdict-warn", "⚠️"
            verdict = "Moderate Damage Detected"
        else:
            card_cls, lbl_cls, ver_cls, icon = "status-warn", "status-label-warn", "status-verdict-warn", "⚠️"
            verdict = "Minor Damage Detected"
        desc = (f"{det_count} region{'s' if det_count>1 else ''} flagged · "
                f"Peak confidence <b>{max_conf:.0%}</b> · "
                f"Severity score <b>{severity_score}/100</b>. "
                f"Inspect highlighted areas for structural or cosmetic damage.")
    else:
        card_cls, lbl_cls, ver_cls, icon = "status-good", "status-label-good", "status-verdict-good", "✅"
        verdict = "No Damage Detected"
        desc = (f"No damage regions found above the <b>{CONF_THRESHOLD:.0%}</b> confidence threshold. "
                f"Vehicle appears to be in good condition.")

    st.markdown(f"""
<div class="status-card {card_cls}">
    <div class="status-icon">{icon}</div>
    <div>
        <div class="status-label {lbl_cls}">Condition Verdict</div>
        <div class="status-verdict {ver_cls}">{verdict}</div>
        <div class="status-desc">{desc}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Image panels ───────────────────────────────────────────────────────────
    col_orig, col_result = st.columns(2)
    result_img = draw_boxes(orig_img, active_detections, names)

    with col_orig:
        st.markdown('<div class="section-heading">Original Image</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), width="stretch")

    with col_result:
        st.markdown('<div class="section-heading">Detection Overlay</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), width="stretch")

    # ── Detection table ────────────────────────────────────────────────────────
    if damage_detected:
        st.markdown('<div class="section-heading" style="margin-top:1.6rem">Detection Breakdown</div>',
                    unsafe_allow_html=True)
        rows = ""
        for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(active_detections, 1):
            label   = names.get(int(cls_id), str(cls_id)).replace("_"," ").title()
            box_w   = int(x2-x1); box_h = int(y2-y1)
            pill_cls = "conf-high" if conf>=0.65 else ("conf-mid" if conf>=0.40 else "conf-low")
            rows += (f"<tr><td>{i}</td><td>{label}</td>"
                     f"<td><span class='conf-pill {pill_cls}'>{conf:.0%}</span></td>"
                     f"<td>{box_w}×{box_h} px</td><td>{int(x1)}, {int(y1)}</td></tr>")
        st.markdown(f"""
<table class="det-table">
    <thead><tr><th>#</th><th>Damage Type</th><th>Confidence</th><th>Region Size</th><th>Origin (x, y)</th></tr></thead>
    <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2, col_gap = st.columns([1, 1, 2])
    with col_dl1:
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        st.download_button("⬇  Download Result", buf.getvalue(), "autoscan_result.png", "image/png")
    with col_dl2:
        buf2 = io.BytesIO()
        Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)).save(buf2, format="PNG")
        st.download_button("⬇  Download Original", buf2.getvalue(), "autoscan_original.png", "image/png")

    # ── Remove Wrong Boxes ─────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Review & Correct Detections</div>', unsafe_allow_html=True)
    st.caption(
        "Remove any incorrect model detections below. "
        "Saved corrections are used to automatically fine-tune the model in the background."
    )

    if raw_detections:
        with st.expander("🗑  Remove incorrect detection boxes", expanded=False):
            st.caption("Click **Remove** on any box the model got wrong. Changes reflect immediately on the overlay.")
            for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(raw_detections):
                label   = names.get(int(cls_id), str(cls_id)).replace("_"," ").title()
                box_w   = int(x2-x1); box_h = int(y2-y1)
                removed = i in st.session_state.removed_det_indices
                rc1, rc2 = st.columns([6, 1])
                with rc1:
                    style = 'opacity:0.35; text-decoration:line-through;' if removed else ''
                    tag   = ' &nbsp;<i style="color:#4ADE80">removed</i>' if removed else ''
                    st.markdown(
                        f'<div class="fb-log-row" style="{style}">'
                        f'<b>#{i+1}</b> &nbsp; <b style="color:#F87171">{label}</b> &nbsp;·&nbsp; '
                        f'conf {conf:.0%} &nbsp;·&nbsp; {box_w}×{box_h} px{tag}</div>',
                        unsafe_allow_html=True
                    )
                with rc2:
                    if removed:
                        if st.button("↩ Undo", key=f"undo_{i}"):
                            st.session_state.removed_det_indices.discard(i)
                            st.rerun()
                    else:
                        if st.button("✖ Remove", key=f"rm_{i}"):
                            st.session_state.removed_det_indices.add(i)
                            st.rerun()

    # ── Active learning: save removed boxes as negative feedback ──────────────
    if st.session_state.removed_det_indices:
        st.markdown("<br>", unsafe_allow_html=True)
        col_sv, _ = st.columns([1, 3])
        with col_sv:
            if st.button("💾 Save Feedback & Train", key="fb_save", type="primary"):
                # Save the remaining (correct) active detections as ground truth
                if active_detections:
                    boxes_yolo = []
                    for cls_id, conf, x1, y1, x2, y2 in active_detections:
                        cx = ((x1+x2)/2)/w;  cy = ((y1+y2)/2)/h
                        bw_n = (x2-x1)/w;    bh_n = (y2-y1)/h
                        boxes_yolo.append((int(cls_id), cx, cy, bw_n, bh_n))
                    total = save_correction(
                        orig_img, boxes_yolo, names,
                        source_label=uploaded.name, model_weights=model_path
                    )
                    st.success(f"✅ Feedback saved. Total corrections: **{total}**")

                    # Auto-trigger fine-tune
                    if (total % FINETUNE_TRIGGER == 0
                            and st.session_state.ft_status != "running"
                            and st.session_state.get("names")):
                        holder = {"status":"running","new_weights":None,"error_msg":None}
                        st.session_state.ft_state_holder = holder
                        st.session_state.ft_status       = "running"
                        st.session_state.ft_start_time   = datetime.now()
                        t = threading.Thread(
                            target=_finetune_worker,
                            args=(model_path, names, holder),
                            daemon=True,
                        )
                        st.session_state.ft_thread = t
                        t.start()
                        st.info("🚀 Auto fine-tune started in the background. Monitor progress in the sidebar.")
                else:
                    st.warning("All detections removed — nothing to save as ground truth.")

    # ── Correction history ─────────────────────────────────────────────────────
    log_entries = _load_feedback_log()
    if log_entries:
        with st.expander(f"📂 Correction history  ({len(log_entries)} sessions)", expanded=False):
            for entry in reversed(log_entries[-20:]):
                cls_str = ", ".join(c["cls_name"] for c in entry.get("corrections",[]))
                st.markdown(
                    f'<div class="fb-log-row"><b>{entry["timestamp"]}</b>'
                    f' &nbsp;·&nbsp; {entry["source"]}'
                    f' &nbsp;·&nbsp; {cls_str}</div>',
                    unsafe_allow_html=True
                )

elif not uploaded:
    st.markdown("""
<div style="background:#0F1923; border:2px dashed #1E3A5F; border-radius:14px;
            padding:3rem; text-align:center; color:#4B6580; margin-top:1.5rem;">
    <div style="font-size:3rem; margin-bottom:0.8rem">🚗</div>
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem; color:#8BA0B8; margin-bottom:0.5rem;">
        Upload a vehicle image to begin analysis
    </div>
    <div style="font-size:0.82rem; color:#4B6580; line-height:1.7">
        Supported formats: JPG · PNG · BMP<br>
        Detection threshold fixed at 40% confidence · TTA and Sliding Window enabled by default
    </div>
</div>
""", unsafe_allow_html=True)
