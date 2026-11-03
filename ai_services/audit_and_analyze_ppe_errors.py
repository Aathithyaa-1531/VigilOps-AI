"""
VigilOps AI - Systematic PPE Error Analysis & Annotation Audit
Performs rigorous ground-truth vs prediction evaluation on the independent test set.
Diagnoses root causes for missed violations (no_helmet & no_vest).
Saves diagnostic comparison images and failure case collections.
"""

import os
import cv2
import json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMG_DIR = os.path.join(BASE_DIR, "datasets", "ppe_finetune", "images", "test")
TEST_LBL_DIR = os.path.join(BASE_DIR, "datasets", "ppe_finetune", "labels", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n_ppe_finetuned.pt")

RUNS_DIR = os.path.join(BASE_DIR, "runs", "ppe_error_analysis")
NO_HELMET_DIR = os.path.join(RUNS_DIR, "no_helmet")
NO_VEST_DIR = os.path.join(RUNS_DIR, "no_vest")
REPORT_JSON = os.path.join(RUNS_DIR, "error_analysis_metrics.json")

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
    5: (255, 255, 0),     # forklift - cyan/yellow
    6: (255, 0, 255)      # crane - magenta
}

def box_iou(b1, b2):
    # b format: [x1, y1, x2, y2]
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter)

def xywh_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2.0) * w)
    y1 = int((cy - bh / 2.0) * h)
    x2 = int((cx + bw / 2.0) * w)
    y2 = int((cy + bh / 2.0) * h)
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

def draw_boxes(image, boxes, is_prediction=False):
    annotated = image.copy()
    for b in boxes:
        cid = b["class_id"]
        coords = b["xyxy"]
        conf = b.get("conf", None)
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        
        cv2.rectangle(annotated, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
        cname = CLASS_NAMES.get(cid, str(cid))
        lbl = f"{cname} {conf:.2f}" if conf is not None else cname
        
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (coords[0], max(0, coords[1] - 18)), (coords[0] + lw + 4, coords[1]), color, -1)
        text_color = (0, 0, 0) if cid in [0, 2, 5] else (255, 255, 255)
        cv2.putText(annotated, lbl, (coords[0] + 2, max(12, coords[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
    return annotated

def execute_error_analysis():
    os.makedirs(NO_HELMET_DIR, exist_ok=True)
    os.makedirs(NO_VEST_DIR, exist_ok=True)
    
    print(f"🚀 Initializing PPE Error Analysis on Test Set...")
    print(f"   Model Checkpoint: {MODEL_PATH}")
    print(f"   Test Images:      {TEST_IMG_DIR}")
    
    model = YOLO(MODEL_PATH)
    
    test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"   Total test images to evaluate: {len(test_images)}")
    
    stats = {
        "total_test_images": len(test_images),
        "ground_truth_counts": defaultdict(int),
        "prediction_counts": defaultdict(int),
        "true_positives": defaultdict(int),
        "false_positives": defaultdict(int),
        "false_negatives": defaultdict(int),
        "no_helmet_failure_causes": defaultdict(int),
        "no_vest_failure_causes": defaultdict(int),
        "failure_cases": []
    }
    
    # Evaluate every image
    for idx, fname in enumerate(test_images):
        img_path = os.path.join(TEST_IMG_DIR, fname)
        lbl_path = os.path.join(TEST_LBL_DIR, os.path.splitext(fname)[0] + ".txt")
        
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]
        
        # 1. Parse Ground Truth
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        xyxy = xywh_to_xyxy(coords, w, h)
                        gt_boxes.append({
                            "class_id": cid,
                            "norm_box": coords,
                            "xyxy": xyxy,
                            "matched": False
                        })
                        stats["ground_truth_counts"][cid] += 1
        
        # 2. Run Inference
        results = model.predict(img_path, conf=0.15, device="mps", verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for b in results.boxes:
                cid = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                pred_boxes.append({
                    "class_id": cid,
                    "conf": conf,
                    "xyxy": xyxy,
                    "matched": False
                })
                stats["prediction_counts"][cid] += 1
        
        # 3. Match Predictions to Ground Truth (IoU >= 0.4)
        for p in pred_boxes:
            best_iou = 0.0
            best_gt = None
            for g in gt_boxes:
                if g["class_id"] == p["class_id"] and not g["matched"]:
                    iou = box_iou(p["xyxy"], g["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g
            if best_iou >= 0.40 and best_gt is not None:
                p["matched"] = True
                best_gt["matched"] = True
                stats["true_positives"][p["class_id"]] += 1
            else:
                stats["false_positives"][p["class_id"]] += 1
        
        # 4. Diagnose False Negatives (Unmatched Ground Truth)
        unmatched_gt = [g for g in gt_boxes if not g["matched"]]
        for g in unmatched_gt:
            stats["false_negatives"][g["class_id"]] += 1
            cid = g["class_id"]
            
            if cid in [1, 3]: # no_helmet (1) or no_vest (3)
                # Find why it was missed
                cause = "MISSED_ENTIRELY"
                bw = g["xyxy"][2] - g["xyxy"][0]
                bh = g["xyxy"][3] - g["xyxy"][1]
                
                # Check for Small Object
                if bh < 25 or bw < 25:
                    cause = "SMALL_OBJECT"
                else:
                    # Check if predicted with conflicting class (e.g. hard_hat instead of no_helmet)
                    conflicting = False
                    for p in pred_boxes:
                        iou = box_iou(g["xyxy"], p["xyxy"])
                        if iou > 0.35:
                            if cid == 1 and p["class_id"] == 0: # no_helmet predicted as hard_hat
                                cause = "CLASSIFIED_AS_COMPLIANT_PPE (hard_hat)"
                                conflicting = True
                                break
                            elif cid == 3 and p["class_id"] == 2: # no_vest predicted as reflective_jacket
                                cause = "CLASSIFIED_AS_COMPLIANT_PPE (reflective_jacket)"
                                conflicting = True
                                break
                            elif iou < 0.45:
                                cause = "POOR_LOCALIZATION_IOU"
                                conflicting = True
                                break
                    if not conflicting and cause != "SMALL_OBJECT":
                        # Check if model detected it at lower confidence (< 0.25)
                        low_conf_match = False
                        for p in pred_boxes:
                            if p["class_id"] == cid and box_iou(g["xyxy"], p["xyxy"]) >= 0.35:
                                if p["conf"] < 0.25:
                                    cause = f"LOW_CONFIDENCE ({p['conf']:.2f} < 0.25)"
                                    low_conf_match = True
                                    break
                        if not low_conf_match:
                            cause = "FEATURE_REPRESENTATION_DEFICIT"
                
                if cid == 1:
                    stats["no_helmet_failure_causes"][cause] += 1
                else:
                    stats["no_vest_failure_causes"][cause] += 1
                
                # Render side-by-side diagnostic visualization
                img_gt = draw_boxes(orig_img, gt_boxes)
                img_pred = draw_boxes(orig_img, pred_boxes, is_prediction=True)
                
                # Add titles
                cv2.putText(img_gt, f"GROUND TRUTH: {fname}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img_pred, f"PREDICTIONS (conf>=0.15)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
                side_by_side = np.hstack([img_gt, img_pred])
                
                out_folder = NO_HELMET_DIR if cid == 1 else NO_VEST_DIR
                out_name = f"diag_{cause[:20].replace(' ', '_')}_{fname}"
                cv2.imwrite(os.path.join(out_folder, out_name), side_by_side)
                
                stats["failure_cases"].append({
                    "image": fname,
                    "target_class": CLASS_NAMES[cid],
                    "box": g["xyxy"],
                    "dimensions": [bw, bh],
                    "diagnosed_cause": cause
                })
        
        if (idx + 1) % 30 == 0:
            print(f"   Evaluated {idx + 1}/{len(test_images)} images...")

    # Calculate metrics
    metrics_summary = {}
    print("\n================ ERROR ANALYSIS RESULTS ================")
    for cid in range(7):
        cname = CLASS_NAMES[cid]
        gt_cnt = stats["ground_truth_counts"][cid]
        tp = stats["true_positives"][cid]
        fp = stats["false_positives"][cid]
        fn = stats["false_negatives"][cid]
        
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics_summary[cname] = {
            "ground_truth_count": gt_cnt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }
        print(f"  {cname:18s}: GT={gt_cnt:3d} | TP={tp:3d} | FP={fp:3d} | FN={fn:3d} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")
    
    print("\n--- NO_HELMET FAILURE ROOT CAUSES ---")
    for cause, cnt in sorted(stats["no_helmet_failure_causes"].items(), key=lambda x: -x[1]):
        print(f"  {cause:45s}: {cnt:3d} instances ({cnt/max(1, sum(stats['no_helmet_failure_causes'].values()))*100:.1f}%)")
        
    print("\n--- NO_VEST FAILURE ROOT CAUSES ---")
    for cause, cnt in sorted(stats["no_vest_failure_causes"].items(), key=lambda x: -x[1]):
        print(f"  {cause:45s}: {cnt:3d} instances ({cnt/max(1, sum(stats['no_vest_failure_causes'].values()))*100:.1f}%)")
    print("========================================================\n")
    
    # Save full report JSON
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics_summary,
            "no_helmet_causes": dict(stats["no_helmet_failure_causes"]),
            "no_vest_causes": dict(stats["no_vest_failure_causes"]),
            "total_failure_cases_saved": len(stats["failure_cases"])
        }, f, indent=2)
    print(f"📄 Detailed analysis JSON saved to: {REPORT_JSON}")

if __name__ == "__main__":
    execute_error_analysis()

