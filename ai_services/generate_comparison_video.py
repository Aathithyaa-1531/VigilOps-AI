"""
VigilOps AI - Video Prediction Comparison Pipeline
Runs baseline model vs fine-tuned model on the industrial PPE audit video.
Generates side-by-side visual comparison video and per-frame detection analytics.
"""

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "..", "frontend", "public", "industrial_ppe_audit_sample.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "video_comparison")
BASELINE_WEIGHTS = os.path.join(BASE_DIR, "models", "yolov8n_ppe_legacy_backup.pt")
FINETUNED_WEIGHTS = os.path.join(BASE_DIR, "models", "yolov8n_ppe_finetuned.pt")
METRICS_JSON = os.path.join(OUTPUT_DIR, "video_comparison_metrics.json")

CLASS_COLORS = {
    0: (0, 255, 0),       # hard_hat - green
    1: (0, 0, 255),       # no_helmet - red
    2: (255, 165, 0),     # reflective_jacket - orange/cyan
    3: (0, 140, 255),     # no_vest - bright orange/red
    4: (255, 255, 255),   # person - white
    5: (255, 255, 0),     # forklift - yellow
    6: (255, 0, 255)      # crane - magenta
}

def annotate_frame(frame, results, title="Model Predictions"):
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Header bar
    cv2.rectangle(annotated, (0, 0), (w, 40), (15, 23, 42), -1)
    cv2.putText(annotated, title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241, 245, 249), 2)
    
    counts = {i: 0 for i in range(7)}
    boxes_detected = []
    
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.25:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                boxes_detected.append((cid, conf, xyxy))
                counts[cid] = counts.get(cid, 0) + 1
                
                color = CLASS_COLORS.get(cid, (200, 200, 200))
                cname = r.names.get(cid, f"cls_{cid}")
                
                # Draw bounding box
                cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                
                # Label badge
                lbl = f"{cname} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (xyxy[0], max(40, xyxy[1] - 18)), (xyxy[0] + lw + 6, max(40, xyxy[1])), color, -1)
                text_color = (0, 0, 0) if cid in [0, 2, 5] else (255, 255, 255)
                cv2.putText(annotated, lbl, (xyxy[0] + 3, max(52, xyxy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    # Subtitle with stats
    stat_str = f"P:{counts[4]} | Helm:{counts[0]} | NoHelm:{counts[1]} | Vest:{counts[2]} | NoVest:{counts[3]}"
    cv2.putText(annotated, stat_str, (w - 380, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)

    return annotated, counts, boxes_detected

def run_video_comparison(max_frames=180):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return

    # Check fallback if legacy backup isn't created yet
    base_w = BASELINE_WEIGHTS if os.path.exists(BASELINE_WEIGHTS) else os.path.join(BASE_DIR, "models", "yolov8n_ppe.pt")
    fine_w = FINETUNED_WEIGHTS if os.path.exists(FINETUNED_WEIGHTS) else os.path.join(BASE_DIR, "runs", "finetuned_ppe_yolov8_run", "weights", "best.pt")

    print(f"🎬 Initializing comparison on {VIDEO_PATH}")
    print(f"   Baseline Model:   {base_w}")
    print(f"   Fine-Tuned Model: {fine_w}")

    model_baseline = YOLO(base_w)
    model_finetuned = YOLO(fine_w)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w, out_h = 640, 360 # Resized for side-by-side display
    canvas_w, canvas_h = out_w * 2, out_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    side_by_side_path = os.path.join(OUTPUT_DIR, "side_by_side_comparison.mp4")
    out_writer = cv2.VideoWriter(side_by_side_path, fourcc, fps, (canvas_w, canvas_h))

    frame_idx = 0
    baseline_totals = {i: 0 for i in range(7)}
    finetuned_totals = {i: 0 for i in range(7)}
    frame_metrics = []

    print(f"⚡ Processing {max_frames} video frames...")
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for inference and layout
        frame_resized = cv2.resize(frame, (out_w, out_h))

        # Run inference
        res_base = model_baseline.predict(frame_resized, conf=0.25, verbose=False, device="mps")
        res_fine = model_finetuned.predict(frame_resized, conf=0.25, verbose=False, device="mps")

        ann_base, counts_b, boxes_b = annotate_frame(frame_resized, res_base, "BASELINE MODEL (Legacy)")
        ann_fine, counts_f, boxes_f = annotate_frame(frame_resized, res_fine, "FINE-TUNED MODEL (VigilOps AI)")

        for i in range(7):
            baseline_totals[i] += counts_b[i]
            finetuned_totals[i] += counts_f[i]

        # Combine side-by-side
        canvas = np.hstack([ann_base, ann_fine])
        out_writer.write(canvas)

        frame_metrics.append({
            "frame": frame_idx,
            "baseline": counts_b,
            "finetuned": counts_f
        })

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"   Processed {frame_idx}/{max_frames} frames...")

    cap.release()
    out_writer.release()
    print(f"✅ Comparison video saved: {side_by_side_path}")

    # Calculate summary comparison
    names = {0: "hard_hat", 1: "no_helmet", 2: "reflective_jacket", 3: "no_vest", 4: "person", 5: "forklift", 6: "crane"}
    summary = {
        "video_evaluated": os.path.basename(VIDEO_PATH),
        "frames_evaluated": frame_idx,
        "baseline_cumulative_detections": {names[i]: baseline_totals[i] for i in range(7)},
        "finetuned_cumulative_detections": {names[i]: finetuned_totals[i] for i in range(7)},
        "output_video": side_by_side_path
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Video metrics summary saved: {METRICS_JSON}")

if __name__ == "__main__":
    run_video_comparison(max_frames=180)

