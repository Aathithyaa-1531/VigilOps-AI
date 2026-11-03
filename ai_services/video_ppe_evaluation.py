"""
VigilOps AI - Video PPE Evaluation & Comparative Inspection
Runs both Previous Fine-Tuned and Improved Violation-Enhanced models on the independent industrial audit video.
Validates:
1. Detection of unhelmeted heads (no_helmet).
2. Detection of unvested torsos (no_vest).
3. Prevention of normal shirts being misclassified as reflective jackets.
4. Preserved person detection and hard hat stability.
Saves diagnostic comparison frames to runs/ppe_error_analysis/video_eval/.
"""

import os
import cv2
import json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "..", "sample_audit_videos", "industrial_ppe_audit_sample.mp4")
if not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = os.path.join(BASE_DIR, "..", "frontend", "public", "industrial_ppe_audit_sample.mp4")

OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "ppe_error_analysis", "video_eval")
PREV_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n_ppe_finetuned.pt")
IMPROVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n_ppe_improved.pt")

CLASS_NAMES = {
    0: "hard_hat",
    1: "no_helmet",
    2: "reflective_jacket",
    3: "no_vest",
    4: "person",
    5: "forklift",
    6: "crane"
}

CLASS_COLORS = {
    0: (0, 200, 0),       # hard_hat - green
    1: (0, 0, 255),       # no_helmet - red
    2: (255, 165, 0),     # reflective_jacket - orange
    3: (0, 100, 255),     # no_vest - bright orange-red
    4: (220, 220, 220),   # person - light gray
    5: (255, 255, 0),     # forklift
    6: (255, 0, 255)      # crane
}

def annotate_frame(frame, detections, title):
    annotated = frame.copy()
    for d in detections:
        cid = d["class_id"]
        x1, y1, x2, y2 = d["bbox"]
        conf = d["conf"]
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        lbl = f"{CLASS_NAMES.get(cid, str(cid))} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + w + 4, y1), color, -1)
        text_color = (0, 0, 0) if cid in [0, 2, 5] else (255, 255, 255)
        cv2.putText(annotated, lbl, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        
    # Title badge
    cv2.rectangle(annotated, (10, 10), (350, 42), (20, 20, 20), -1)
    cv2.putText(annotated, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return annotated

def run_video_audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("==================================================================")
    print("🎥 VIGILOPS AI — VIDEO PPE MODEL COMPARISON AUDIT")
    print("==================================================================")
    print(f"Video Source   : {VIDEO_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Total Video Frames: {total_frames} ({total_frames/fps:.1f}s at {fps:.1f} fps)")

    prev_model = YOLO(PREV_MODEL_PATH) if os.path.exists(PREV_MODEL_PATH) else None
    imp_model = YOLO(IMPROVED_MODEL_PATH) if os.path.exists(IMPROVED_MODEL_PATH) else None

    if imp_model is None:
        print(f"❌ Improved model not found at {IMPROVED_MODEL_PATH}")
        return

    prev_stats = defaultdict(int)
    imp_stats = defaultdict(int)

    frame_idx = 0
    saved_samples = 0
    sample_interval = max(1, total_frames // 60) # inspect ~60 frames evenly

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Inference with Prev Model
            prev_dets = []
            if prev_model:
                r_prev = prev_model.predict(frame, conf=0.25, device="mps", verbose=False)[0]
                if r_prev.boxes is not None:
                    for b in r_prev.boxes:
                        cid = int(b.cls[0])
                        conf = float(b.conf[0])
                        xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                        prev_dets.append({"class_id": cid, "conf": conf, "bbox": xyxy})
                        prev_stats[cid] += 1

            # Inference with Improved Model
            imp_dets = []
            r_imp = imp_model.predict(frame, conf=0.25, device="mps", verbose=False)[0]
            if r_imp.boxes is not None:
                for b in r_imp.boxes:
                    cid = int(b.cls[0])
                    conf = float(b.conf[0])
                    xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    imp_dets.append({"class_id": cid, "conf": conf, "bbox": xyxy})
                    imp_stats[cid] += 1

            # Check if any violation detected in this frame
            has_violation = any(d["class_id"] in [1, 3] for d in imp_dets)
            if (has_violation or saved_samples < 8) and saved_samples < 15:
                # Save side-by-side
                f_prev = annotate_frame(frame, prev_dets, "PREVIOUS FINE-TUNED MODEL") if prev_model else frame
                f_imp = annotate_frame(frame, imp_dets, "IMPROVED VIOLATION MODEL")
                
                # Resize for side-by-side if too wide
                h, w = f_imp.shape[:2]
                if w > 960:
                    f_prev = cv2.resize(f_prev, (960, int(h * 960 / w)))
                    f_imp = cv2.resize(f_imp, (960, int(h * 960 / w)))
                
                comparison = np.hstack([f_prev, f_imp])
                out_path = os.path.join(OUTPUT_DIR, f"comparison_frame_{frame_idx:04d}.jpg")
                cv2.imwrite(out_path, comparison)
                saved_samples += 1

        frame_idx += 1

    cap.release()

    print("\n================ VIDEO AUDIT DETECTION SUMMARY ================")
    print(f"{'Class Name':<20} | {'Previous Model Detections':<28} | {'Improved Model Detections':<28}")
    print("-" * 80)
    for cid in range(7):
        cname = CLASS_NAMES[cid]
        print(f"{cname:<20} | {prev_stats[cid]:<28d} | {imp_stats[cid]:<28d}")
    print("=================================================================\n")
    print(f"📸 Saved {saved_samples} side-by-side comparison frames to: {OUTPUT_DIR}")

    summary = {
        "video_evaluated": os.path.basename(VIDEO_PATH),
        "total_frames_audited": frame_idx,
        "sample_frames_analyzed": frame_idx // sample_interval,
        "previous_model_detections": {CLASS_NAMES[k]: v for k, v in prev_stats.items()},
        "improved_model_detections": {CLASS_NAMES[k]: v for k, v in imp_stats.items()},
        "comparison_frames_saved": saved_samples
    }
    with open(os.path.join(OUTPUT_DIR, "video_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_video_audit()

