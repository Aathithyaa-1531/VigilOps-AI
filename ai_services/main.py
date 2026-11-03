"""
VigilOps AI - FastAPI Microservice Engine
Provides high-performance edge detection with Ultralytics YOLOv8 and ByteTrack object tracking,
dynamic multi-factor risk scoring (Scikit-Learn Random Forest), zero-prompt OSHA SOP retrieval,
and 5-agent LangGraph decision orchestration.
"""

import asyncio
import time
import cv2
import os
import base64
import numpy as np
from typing import List, Dict, Any, Optional

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from agent_brain import run_agent_reasoning
from risk_engine import risk_engine
from rag_sop import retrieve_sop_for_hazard

app = FastAPI(
    title="VigilOps AI Microservice Engine",
    description="Autonomous EHS & Industrial Safety Monitoring API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize YOLO Models (Prioritizing custom fine-tuned weights with fallback)
CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "yolov8n_ppe.pt")
BASE_WEIGHTS = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

yolo_model = None
fallback_model = None

if os.path.exists(CUSTOM_WEIGHTS):
    try:
        yolo_model = YOLO(CUSTOM_WEIGHTS)
        print(f"✅ Loaded fine-tuned YOLOv8 PPE model ({CUSTOM_WEIGHTS}) on {DEVICE}")
    except Exception as e:
        print(f"⚠️ Error loading custom YOLO PPE model: {e}")

if os.path.exists(BASE_WEIGHTS):
    try:
        fallback_model = YOLO(BASE_WEIGHTS)
        print(f"✅ Loaded base Ultralytics YOLOv8 fallback model ({BASE_WEIGHTS})")
    except Exception as e:
        print(f"⚠️ Error loading base YOLO: {e}")

if yolo_model is None:
    yolo_model = fallback_model

# In-memory tracking state across frames
# Key: `source_id:track_id` -> { 'first_seen': float, 'last_seen': float, 'last_box': [], 'violation_start': float, 'speed': float }
TRACKER_STATE: Dict[str, Dict[str, Any]] = {}

class StreamRequest(BaseModel):
    source: str
    mode: Optional[str] = "live"

class FrameAnalysisRequest(BaseModel):
    source_id: str
    hazard_type: Optional[str] = "NO_HELMET"
    duration: Optional[float] = 12.0
    proximity: Optional[float] = 0.88

class Base64DetectRequest(BaseModel):
    image_base64: str
    source_id: Optional[str] = "Surveillance Feed 01"
    duration: Optional[float] = 5.0
    worker_speed: Optional[float] = 1.2

def calculate_proximity(boxA: List[float], boxB: List[float]) -> float:
    """Calculates normalized center-point Euclidean proximity score (0 to 1)."""
    centerA = [(boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0]
    centerB = [(boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0]
    dist = np.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)
    # Closer distance -> higher proximity danger
    proximity_score = max(0.0, 1.0 - (dist / 0.65))
    return round(float(proximity_score), 3)

def evaluate_ppe_for_person(
    frame: np.ndarray,
    xyxy_pixels: List[int],
    detected_objects: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates safety helmet and high-visibility vest compliance for a detected person.
    Multi-modal verification engine combining:
    1. Direct YOLOv8 PPE bounding-box detection with relaxed containment & true box propagation.
    2. Structural edge and specular contour analysis (distinguishing rigid hard hats from hair/skin).
    3. Retroreflective microprismatic stripe gradient detection (ANSI/ISEA 107 safety vests).
    4. Optical colorimetry preventing plain yellow/orange shirts from claiming compliance.
    5. Strict UNKNOWN classification when head/torso is occluded or insufficiently resolved.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = xyxy_pixels
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    pw = x2 - x1
    ph = y2 - y1
    p_norm = [x1 / fw, y1 / fh, x2 / fw, y2 / fh]
    pw_n = p_norm[2] - p_norm[0]
    ph_n = p_norm[3] - p_norm[1]

    # Anatomical sub-boxes tightly localized to head and torso
    head_sub_norm = [
        round(max(0.0, min(1.0, (x1 + 0.18 * pw) / fw)), 3),
        round(max(0.0, min(1.0, y1 / fh)), 3),
        round(max(0.0, min(1.0, (x2 - 0.18 * pw) / fw)), 3),
        round(max(0.0, min(1.0, (y1 + 0.24 * ph) / fh)), 3)
    ]
    torso_sub_norm = [
        round(max(0.0, min(1.0, (x1 + 0.10 * pw) / fw)), 3),
        round(max(0.0, min(1.0, (y1 + 0.22 * ph) / fw)), 3),
        round(max(0.0, min(1.0, (x2 - 0.10 * pw) / fw)), 3),
        round(max(0.0, min(1.0, (y1 + 0.65 * ph) / fh)), 3)
    ]

    if pw < 18 or ph < 40:
        return {
            "has_helmet": True,
            "has_vest": True,
            "helmet_status": "UNKNOWN",
            "vest_status": "UNKNOWN",
            "helmet_conf": 0.50,
            "vest_conf": 0.50,
            "helmet_box_norm": head_sub_norm,
            "vest_box_norm": torso_sub_norm,
            "evaluation_method": "baseline_small_box"
        }

    person_crop = frame[y1:y2, x1:x2]
    head_h = max(4, int(ph * 0.22))
    torso_start = head_h
    torso_end = max(torso_start + 4, int(ph * 0.65))

    # Center-crop head horizontally (inner 60% width) to eliminate background margins
    head_x1 = max(0, int(pw * 0.20))
    head_x2 = min(pw, max(head_x1 + 4, int(pw * 0.80)))
    head_crop = person_crop[0:head_h, head_x1:head_x2]

    # Center-crop torso horizontally (inner 70% width)
    torso_x1 = max(0, int(pw * 0.15))
    torso_x2 = min(pw, max(torso_x1 + 4, int(pw * 0.85)))
    torso_crop = person_crop[torso_start:torso_end, torso_x1:torso_x2]

    # Check for direct overlapping YOLO detection if provided
    direct_helmet_det = False
    direct_no_helmet_det = False
    direct_vest_det = False
    direct_no_vest_det = False
    direct_helmet_box = None
    direct_no_helmet_box = None
    direct_vest_box = None
    direct_no_vest_box = None

    if detected_objects:
        for obj in detected_objects:
            cls = obj.get("class", "").lower()
            box = obj.get("bbox", [])
            c_conf = obj.get("confidence", 0.0)
            if len(box) == 4:
                c_x = (box[0] + box[2]) / 2.0
                c_y = (box[1] + box[3]) / 2.0
                # Relaxed horizontal containment (worker leaning or posture):
                in_x = (p_norm[0] - pw_n * 0.15) <= c_x <= (p_norm[2] + pw_n * 0.15)
                # Head vertical zone (allow slightly above top of person bounding box):
                in_head_y = (p_norm[1] - ph_n * 0.12) <= c_y <= (p_norm[1] + ph_n * 0.38)
                # Torso vertical zone:
                in_torso_y = (p_norm[1] + ph_n * 0.18) <= c_y <= (p_norm[1] + ph_n * 0.72)

                if in_x and in_head_y:
                    if "hard_hat" in cls and c_conf >= 0.25:
                        direct_helmet_det = True
                        direct_helmet_box = box
                    elif "no_helmet" in cls and c_conf >= 0.22:
                        direct_no_helmet_det = True
                        direct_no_helmet_box = box

                if in_x and in_torso_y:
                    # Require minimum 0.40 confidence to prevent ordinary shirts from triggering vest detection
                    if ("reflective" in cls or "jacket" in cls) and c_conf >= 0.40:
                        direct_vest_det = True
                        direct_vest_box = box
                    elif "no_vest" in cls and c_conf >= 0.25:
                        direct_no_vest_det = True
                        direct_no_vest_box = box

    # Occlusion check: worker cut off by frame edge or too low resolution
    head_occluded = bool(y1 <= 4 or ph < 50 or pw < 22 or head_crop.size < 80)
    torso_occluded = bool(y2 >= (fh - 4) or ph < 50 or pw < 22 or torso_crop.size < 120)

    # ==================== 1. CRANIUM & HARD HAT EVALUATION ====================
    chosen_helmet_box = direct_helmet_box if direct_helmet_box else (direct_no_helmet_box if direct_no_helmet_box else head_sub_norm)
    if direct_helmet_det:
        helmet_status = "COMPLIANT"
        has_helmet = True
        helmet_conf = 0.95
    elif direct_no_helmet_det:
        helmet_status = "VIOLATION"
        has_helmet = False
        helmet_conf = 0.90
    elif head_occluded:
        helmet_status = "UNKNOWN"
        has_helmet = True  # Avoid false positive violation when head is truncated by frame
        helmet_conf = 0.50
    elif head_crop.size > 0:
        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_head, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_head, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        mean_grad = float(np.mean(grad_mag))
        std_grad = float(np.std(grad_mag))
        has_rigid_structure = bool(mean_grad > 13.0 and std_grad > 10.0)

        hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        yellow_m = cv2.inRange(hsv_head, (20, 140, 120), (36, 255, 255))
        white_m = cv2.inRange(hsv_head, (0, 0, 215), (180, 28, 255))
        orange_m = cv2.inRange(hsv_head, (5, 145, 125), (18, 255, 255))
        blue_m = cv2.inRange(hsv_head, (98, 130, 95), (128, 255, 255))

        helmet_pix = np.count_nonzero(yellow_m | white_m | orange_m | blue_m)
        head_area = float(head_crop.shape[0] * head_crop.shape[1])
        head_ratio = float(helmet_pix / head_area) if head_area > 0 else 0.0

        if (head_ratio >= 0.16 and has_rigid_structure) or (head_ratio >= 0.28):
            has_helmet = True
            helmet_status = "COMPLIANT"
            helmet_conf = float(min(0.98, max(0.70, 0.55 + head_ratio * 0.8)))
        elif ph >= 70 and head_ratio < 0.08 and (not has_rigid_structure or mean_grad < 9.0):
            # Close visible worker with clear absence of hard hat structure/pigment
            has_helmet = False
            helmet_status = "VIOLATION"
            helmet_conf = float(min(0.90, max(0.65, 0.85 - head_ratio)))
        else:
            # Ambiguous resolution or angle: strictly report UNKNOWN rather than guessing
            has_helmet = True
            helmet_status = "UNKNOWN"
            helmet_conf = 0.50
    else:
        has_helmet = True
        helmet_status = "UNKNOWN"
        helmet_conf = 0.50

    # ==================== 2. TORSO & SAFETY VEST EVALUATION ====================
    chosen_vest_box = direct_vest_box if direct_vest_box else (direct_no_vest_box if direct_no_vest_box else torso_sub_norm)
    if torso_occluded:
        vest_status = "UNKNOWN"
        has_vest = True
        vest_conf = 0.50
    elif direct_no_vest_det:
        vest_status = "VIOLATION"
        has_vest = False
        vest_conf = 0.90
    elif direct_vest_det:
        vest_status = "COMPLIANT"
        has_vest = True
        vest_conf = 0.95
    elif torso_crop.size > 0:
        gray_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
        row_luminance = np.mean(gray_torso, axis=1)
        has_retroreflective_stripe = False
        if len(row_luminance) > 8:
            luminance_step = np.max(np.abs(np.diff(row_luminance)))
            has_retroreflective_stripe = bool(luminance_step > 22.0)

        hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        lime_m = cv2.inRange(hsv_torso, (26, 85, 85), (48, 255, 255))
        orange_m = cv2.inRange(hsv_torso, (8, 110, 110), (22, 255, 255))

        vest_pix = np.count_nonzero(lime_m | orange_m)
        torso_area = float(torso_crop.shape[0] * torso_crop.shape[1])
        torso_ratio = float(vest_pix / torso_area) if torso_area > 0 else 0.0

        # ANSI/ISEA 107 Rule:
        # A true safety vest MUST have retroreflective horizontal banding (high luminance step).
        # Ordinary shirts (even yellow or orange) lack retroreflective bands!
        if has_retroreflective_stripe and (torso_ratio >= 0.10 or direct_vest_det):
            has_vest = True
            vest_status = "COMPLIANT"
            vest_conf = float(min(0.98, max(0.72, 0.60 + torso_ratio * 0.8)))
        elif not has_retroreflective_stripe and torso_ratio < 0.12 and ph >= 65:
            # Plain dark, white, blue, or grey clothing without vest
            has_vest = False
            vest_status = "VIOLATION"
            vest_conf = 0.85
        elif not has_retroreflective_stripe and torso_ratio >= 0.12:
            # Ordinary bright colored shirt (yellow/orange) without retroreflective banding
            if ph >= 65:
                has_vest = False
                vest_status = "VIOLATION"
                vest_conf = 0.80
            else:
                has_vest = True
                vest_status = "UNKNOWN"
                vest_conf = 0.50
        else:
            has_vest = True
            vest_status = "UNKNOWN"
            vest_conf = 0.50
    else:
        has_vest = True
        vest_status = "UNKNOWN"
        vest_conf = 0.50

    return {
        "has_helmet": bool(has_helmet),
        "has_vest": bool(has_vest),
        "helmet_status": helmet_status,
        "vest_status": vest_status,
        "helmet_conf": round(float(helmet_conf), 2),
        "vest_conf": round(float(vest_conf), 2),
        "helmet_box_norm": chosen_helmet_box,
        "vest_box_norm": chosen_vest_box,
        "evaluation_method": "multi_modal_structural_optical"
    }

def run_yolo_inference_on_frame(frame: np.ndarray, source_id: str = "Video Frame", duration: float = 0.0):
    """
    Executes real Ultralytics YOLOv8 inference with ByteTrack object tracking on an OpenCV frame.
    Calculates actual latency, maintains persistent track IDs, detects PPE violations,
    and invokes LangGraph multi-agent decision orchestration.
    """
    global TRACKER_STATE
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO tracking model not initialized")

    h, w = frame.shape[:2]
    now = time.time()

    # Prune stale tracking records (>5s inactivity)
    stale_keys = [k for k, v in TRACKER_STATE.items() if (now - v.get("last_seen", now)) > 5.0]
    for k in stale_keys:
        TRACKER_STATE.pop(k, None)

    # 1. Run actual Ultralytics YOLOv8 tracking with ByteTrack
    t0 = time.perf_counter()
    try:
        results = yolo_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=DEVICE,
            verbose=False,
            conf=0.30,
            iou=0.45
        )
    except Exception as e:
        # Fallback to CPU if MPS or tracker config throws
        try:
            results = yolo_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                device="cpu",
                verbose=False,
                conf=0.30,
                iou=0.45
            )
        except Exception:
            results = yolo_model(frame, device="cpu", verbose=False, conf=0.30, iou=0.45)

    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000.0, 1)

    active_names = yolo_model.names
    has_boxes = results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0
    if not has_boxes and fallback_model is not None:
        try:
            fb_results = fallback_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=DEVICE,
                verbose=False,
                conf=0.25
            )
            if fb_results and len(fb_results) > 0 and fb_results[0].boxes is not None and len(fb_results[0].boxes) > 0:
                results = fb_results
                active_names = fallback_model.names
        except Exception:
            try:
                fb_results = fallback_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device="cpu",
                    verbose=False,
                    conf=0.25
                )
                if fb_results and len(fb_results) > 0 and fb_results[0].boxes is not None and len(fb_results[0].boxes) > 0:
                    results = fb_results
                    active_names = fallback_model.names
            except Exception:
                pass

    boxes_out = []
    persons = []
    machinery = []
    hazards = []
    max_duration = duration
    max_speed = 1.0

    raw_ppe_detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for b in results[0].boxes:
            c_id = int(b.cls[0])
            c_name = active_names.get(c_id, f"class_{c_id}").lower()
            c_xyxy = [float(v) for v in b.xyxy[0]]
            c_norm = [
                round(max(0.0, min(1.0, c_xyxy[0] / w)), 3),
                round(max(0.0, min(1.0, c_xyxy[1] / h)), 3),
                round(max(0.0, min(1.0, c_xyxy[2] / w)), 3),
                round(max(0.0, min(1.0, c_xyxy[3] / h)), 3)
            ]
            raw_ppe_detections.append({
                "class": c_name,
                "confidence": round(float(b.conf[0]), 3),
                "bbox": c_norm
            })

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes_obj = results[0].boxes
        for idx, box in enumerate(boxes_obj):
            cls_id = int(box.cls[0])
            cls_name = active_names.get(cls_id, f"class_{cls_id}").lower()
            conf = round(float(box.conf[0]), 3)
            xyxy = [float(v) for v in box.xyxy[0]]
            pixel_box = [int(v) for v in xyxy]

            # ByteTrack track ID
            track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else (idx + 1)

            norm_box = [
                round(max(0.0, min(1.0, xyxy[0] / w)), 3),
                round(max(0.0, min(1.0, xyxy[1] / h)), 3),
                round(max(0.0, min(1.0, xyxy[2] / w)), 3),
                round(max(0.0, min(1.0, xyxy[3] / h)), 3)
            ]

            state_key = f"{source_id}:{track_id}"
            track_info = TRACKER_STATE.get(state_key, {
                "first_seen": now,
                "last_seen": now,
                "last_box": norm_box,
                "violation_start": None,
                "speed": 1.0
            })

            # Calculate worker movement speed (pixels / dt converted to m/s approximation)
            dt = max(0.01, now - track_info["last_seen"])
            prev_center = [(track_info["last_box"][0] + track_info["last_box"][2])/2.0, (track_info["last_box"][1] + track_info["last_box"][3])/2.0]
            curr_center = [(norm_box[0] + norm_box[2])/2.0, (norm_box[1] + norm_box[3])/2.0]
            displacement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            calc_speed = round(float(min(3.5, max(0.4, (displacement / dt) * 12.0))), 2)

            track_info["last_seen"] = now
            track_info["last_box"] = norm_box
            track_info["speed"] = calc_speed
            max_speed = max(max_speed, calc_speed)

            # Evaluate person PPE with raw detections & occlusion awareness
            if cls_name == "person":
                ppe = evaluate_ppe_for_person(frame, pixel_box, detected_objects=raw_ppe_detections)
                persons.append((track_id, norm_box))

                pw = max(1, pixel_box[2] - pixel_box[0])
                ph = max(1, pixel_box[3] - pixel_box[1])

                # Anatomical sub-boxes tightly localized to head and torso
                head_sub_norm = [
                    round(max(0.0, min(1.0, (xyxy[0] + 0.18 * pw) / w)), 3),
                    round(max(0.0, min(1.0, xyxy[1] / h)), 3),
                    round(max(0.0, min(1.0, (xyxy[2] - 0.18 * pw) / w)), 3),
                    round(max(0.0, min(1.0, (xyxy[1] + 0.24 * ph) / h)), 3)
                ]
                torso_sub_norm = [
                    round(max(0.0, min(1.0, (xyxy[0] + 0.10 * pw) / w)), 3),
                    round(max(0.0, min(1.0, (xyxy[1] + 0.22 * ph) / h)), 3),
                    round(max(0.0, min(1.0, (xyxy[2] - 0.10 * pw) / w)), 3),
                    round(max(0.0, min(1.0, (xyxy[1] + 0.65 * ph) / h)), 3)
                ]

                # 1. Determine Helmet Sub-Box (surrounding actual cranium/helmet)
                head_box = ppe.get("helmet_box_norm", head_sub_norm)
                h_status = ppe.get("helmet_status", "UNKNOWN")
                if h_status == "COMPLIANT":
                    boxes_out.append({
                        "track_id": track_id,
                        "class": "hard_hat",
                        "confidence": ppe["helmet_conf"],
                        "bbox_normalized": head_box,
                        "worker_speed": calc_speed
                    })
                elif h_status == "VIOLATION":
                    if track_info["violation_start"] is None:
                        track_info["violation_start"] = now
                    viol_dur = round(now - track_info["violation_start"], 1)
                    max_duration = max(max_duration, viol_dur)

                    hazards.append({
                        "label": "NO_HELMET",
                        "track_id": track_id,
                        "confidence": ppe["helmet_conf"],
                        "bbox": head_box,
                        "duration": viol_dur
                    })
                    boxes_out.append({
                        "track_id": track_id,
                        "class": "no_helmet",
                        "confidence": ppe["helmet_conf"],
                        "bbox_normalized": head_box,
                        "worker_speed": calc_speed
                    })

                # 2. Determine Vest Sub-Box (surrounding actual torso/vest)
                torso_box = ppe.get("vest_box_norm", torso_sub_norm)
                v_status = ppe.get("vest_status", "UNKNOWN")
                if v_status == "COMPLIANT":
                    boxes_out.append({
                        "track_id": track_id,
                        "class": "reflective_jacket",
                        "confidence": ppe["vest_conf"],
                        "bbox_normalized": torso_box,
                        "worker_speed": calc_speed
                    })
                elif v_status == "VIOLATION":
                    if track_info["violation_start"] is None:
                        track_info["violation_start"] = now
                    viol_dur = round(now - track_info["violation_start"], 1)
                    max_duration = max(max_duration, viol_dur)

                    hazards.append({
                        "label": "NO_VEST",
                        "track_id": track_id,
                        "confidence": ppe["vest_conf"],
                        "bbox": torso_box,
                        "duration": viol_dur
                    })
                    boxes_out.append({
                        "track_id": track_id,
                        "class": "no_vest",
                        "confidence": ppe["vest_conf"],
                        "bbox_normalized": torso_box,
                        "worker_speed": calc_speed
                    })

                # 3. Determine Complete Person Box (Surrounding the visible person, NOT relabeled as no_helmet)
                if h_status == "VIOLATION" and v_status == "VIOLATION":
                    display_class = f"worker #{track_id} (no_helmet & no_vest)"
                elif h_status == "VIOLATION":
                    display_class = f"worker #{track_id} (no_helmet)"
                elif v_status == "VIOLATION":
                    display_class = f"worker #{track_id} (no_vest)"
                elif h_status == "UNKNOWN" or v_status == "UNKNOWN":
                    display_class = f"worker #{track_id} (unknown / verify)"
                else:
                    track_info["violation_start"] = None
                    display_class = f"worker #{track_id} (compliant)"

                TRACKER_STATE[state_key] = track_info

                boxes_out.append({
                    "track_id": track_id,
                    "class": display_class,
                    "confidence": conf,
                    "bbox_pixels": pixel_box,
                    "bbox_normalized": norm_box,
                    "worker_speed": calc_speed,
                    "ppe_evaluation": ppe
                })

            elif cls_name in ["no_helmet", "no_vest", "hard_hat", "reflective_jacket"]:
                # Suppress independent/floating PPE detections: PPE compliance is strictly tied to tracked workers.
                # This prevents non-human objects (shadows, background equipment) from triggering false safety violations.
                pass

            elif cls_name in ["forklift", "truck", "car", "crane"]:
                machinery.append((cls_name, norm_box))
                boxes_out.append({
                    "track_id": track_id,
                    "class": cls_name,
                    "confidence": conf,
                    "bbox_pixels": pixel_box,
                    "bbox_normalized": norm_box,
                    "worker_speed": 0.0
                })
            else:
                boxes_out.append({
                    "track_id": track_id,
                    "class": cls_name,
                    "confidence": conf,
                    "bbox_pixels": pixel_box,
                    "bbox_normalized": norm_box,
                    "worker_speed": 0.0
                })

    # 2. Check Machinery & Overhead Fall-Zone Proximity
    proximity_score = 0.15
    for m_name, m_box in machinery:
        for p_id, p_box in persons:
            prox = calculate_proximity(m_box, p_box)
            if prox > 0.50:
                proximity_score = max(proximity_score, prox)
                hazards.append({
                    "label": "FORKLIFT_HAZARD" if "forklift" in m_name or "truck" in m_name else "CRANE_HAZARD",
                    "track_id": p_id,
                    "confidence": 0.92,
                    "bbox": m_box
                })

    # 3. Invoke LangGraph 5-Agent Brain
    agent_output = run_agent_reasoning(
        source_id=source_id,
        raw_detections=boxes_out,
        duration=max_duration,
        proximity=proximity_score,
        worker_speed=max_speed
    )

    primary_hazard = hazards[0]["label"] if hazards else "COMPLIANT_NORMAL"

    return {
        "source": source_id,
        "latency_ms": latency_ms,
        "detections_count": len(boxes_out),
        "detections": boxes_out,
        "hazards": hazards,
        "primary_hazard": primary_hazard,
        "risk_score": agent_output.get("predictive_risk_score", 0.08),
        "threat_level": agent_output.get("threat_level", "LOW"),
        "breakdown": agent_output.get("breakdown", {}),
        "sop": agent_output.get("retrieved_sop", {}),
        "action_plan": agent_output.get("action_plan", []),
        "operational_stages": agent_output.get("operational_stages", []),
        "logs": agent_output.get("agent_logs", [])
    }

def process_video_feed(source_path: str):
    """
    Background worker ingesting video feed (RTSP or local MP4).
    Decodes video frames through YOLOv8 and LangGraph reasoning pipeline.
    """
    print(f"🎬 Ingesting surveillance stream from: {source_path}")
    cap = None
    if os.path.exists(source_path) or source_path.startswith("rtsp://") or source_path.startswith("http"):
        try:
            cap = cv2.VideoCapture(source_path)
        except Exception as e:
            print(f"OpenCV capture error: {e}")

    if cap and cap.isOpened():
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            frame_idx += 1
            if frame_idx % 15 != 0:  # Sample every 15 frames for real-time tracking
                continue

            try:
                run_yolo_inference_on_frame(frame, source_id=source_path)
            except Exception as err:
                print(f"Frame processing note: {err}")

            time.sleep(0.5)
        cap.release()
    else:
        print(f"⚠️ Source '{source_path}' could not be opened as video stream.")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "VigilOps AI Microservice",
        "yolo_loaded": yolo_model is not None,
        "yolo_classes": list(yolo_model.names.values()) if yolo_model else [],
        "scikit_model": "loaded" if risk_engine.model is not None else "heuristic_calibrated"
    }

@app.post("/api/detect")
async def detect_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    result = run_yolo_inference_on_frame(frame, source_id=file.filename or "Uploaded Frame")
    return result

@app.post("/api/detect-base64")
async def detect_base64_frame(req: Base64DetectRequest):
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    result = run_yolo_inference_on_frame(frame, source_id=req.source_id, duration=req.duration or 0.0)
    return result

@app.post("/api/stream/reset")
def reset_stream_state():
    """Resets tracking cache and smoothed threat scoring."""
    global TRACKER_STATE
    TRACKER_STATE.clear()
    risk_engine.reset()
    return {"status": "tracker_reset_complete"}

@app.post("/api/start-stream")
def start_stream(req: StreamRequest, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(process_video_feed, req.source)
    return {
        "status": "stream_worker_initialized",
        "target": req.source,
        "mode": req.mode
    }

@app.post("/api/analyze-frame")
def analyze_frame(req: FrameAnalysisRequest):
    hazards = [{"class": req.hazard_type, "confidence": 0.95}]
    result = run_agent_reasoning(
        source_id=req.source_id,
        raw_detections=hazards,
        duration=req.duration or 0.0,
        proximity=req.proximity or 0.1
    )
    return {
        "status": "analysis_complete",
        "threat_level": result.get("threat_level"),
        "risk_score": result.get("predictive_risk_score"),
        "sop": result.get("retrieved_sop"),
        "logs": result.get("agent_logs"),
        "operational_stages": result.get("operational_stages")
    }

class VideoAuditRequest(BaseModel):
    video_path: Optional[str] = None
    filename: Optional[str] = None
    inference_target: Optional[str] = "local_m2"

@app.get("/api/cloud-gpu/status")
def get_cloud_gpu_status():
    from cloud_gpu_adapter import cloud_gpu_adapter
    return cloud_gpu_adapter.check_health()

@app.post("/api/audit-video")
async def audit_video_endpoint(req: VideoAuditRequest):
    from video_auditor import video_auditor
    video_path = req.video_path
    if not video_path or not os.path.exists(video_path):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_audit_videos", "industrial_ppe_audit_sample.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
        else:
            raise HTTPException(status_code=404, detail="Audit video file not found")

    result = video_auditor.audit_video_parallel(
        video_path,
        source_id=req.filename or os.path.basename(video_path),
        inference_target=req.inference_target or "local_m2"
    )
    return result

@app.post("/api/audit-video/upload")
async def audit_uploaded_video_file(file: UploadFile = File(...)):
    from video_auditor import video_auditor
    upload_dir = "/tmp/vigilops_audits"
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    result = video_auditor.audit_video_parallel(temp_path, source_id=file.filename)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)