"""
VigilOps AI - Industrial PPE Retraining Pipeline
Retrains Ultralytics YOLOv8 on Enhanced Photorealistic Dataset + Real CCTV Surveillance Frames.
Applies extensive data augmentations (Mosaic, scale jitter, HSV perturbation, flips).
Evaluates Precision, Recall, mAP50, and mAP50-95 independently per class.
"""

import os
import shutil
import json
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "datasets", "real_ppe_vision", "data.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TARGET_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n_ppe.pt")
METRICS_PATH = os.path.join(MODELS_DIR, "enhanced_ppe_metrics.json")

def train_enhanced_ppe_model(epochs=15, imgsz=640, batch=16):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡ [Retraining Pipeline] Hardware Device: {device} (Apple Silicon Acceleration)")

    base_weights = os.path.join(BASE_DIR, "yolov8n.pt")
    if not os.path.exists(base_weights):
        base_weights = "yolov8n.pt"

    print(f"🚀 Initializing Ultralytics YOLOv8 from base weights: {base_weights}")
    model = YOLO(base_weights)

    print(f"⚡ Fine-tuning YOLOv8 on Enhanced PPE Dataset: {DATA_YAML}")
    print(f"   Epochs: {epochs} | Image Size: {imgsz} | Batch Size: {batch}")
    print(f"   Data Augmentation: Mosaic=1.0, HSV=(0.015, 0.7, 0.4), FlipLR=0.5, Scale=0.5")

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        project=os.path.join(BASE_DIR, "runs"),
        name="enhanced_ppe_yolov8_run",
        exist_ok=True,
        verbose=True,
        # Applied Data Augmentations
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5
    )

    # Locate the best or last trained weights
    run_weights = os.path.join(BASE_DIR, "runs", "enhanced_ppe_yolov8_run", "weights", "best.pt")
    if not os.path.exists(run_weights):
        run_weights = os.path.join(BASE_DIR, "runs", "enhanced_ppe_yolov8_run", "weights", "last.pt")

    if os.path.exists(run_weights):
        shutil.copy(run_weights, TARGET_WEIGHTS)
        print(f"✅ Successfully exported retrained PPE model to: {TARGET_WEIGHTS} ({os.path.getsize(TARGET_WEIGHTS)} bytes)")

    # Execute rigorous validation independently for every class
    print("🔍 Running Comprehensive Class-by-Class Validation on Retrained Model...")
    val_model = YOLO(TARGET_WEIGHTS)
    val_metrics = val_model.val(data=DATA_YAML, imgsz=imgsz, device=device, verbose=True)

    class_results = {}
    print("\n================ INDEPENDENT PER-CLASS EVALUATION ================")
    for i, name in val_model.names.items():
        p = float(val_metrics.box.p[i]) if i < len(val_metrics.box.p) else 0.0
        r = float(val_metrics.box.r[i]) if i < len(val_metrics.box.r) else 0.0
        map50 = float(val_metrics.box.ap50[i]) if i < len(val_metrics.box.ap50) else 0.0
        map50_95 = float(val_metrics.box.ap[i]) if i < len(val_metrics.box.ap) else 0.0
        class_results[name] = {
            "class_id": i,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4)
        }
        print(f"  Class {i} ({name:18s}): P={p:.4f} | R={r:.4f} | mAP50={map50:.4f} | mAP50-95={map50_95:.4f}")
    print("==================================================================\n")

    overall_metrics = {
        "model_architecture": "YOLOv8 Nano (Enhanced PPE Retraining)",
        "dataset": "Enhanced Industrial PPE Vision + CCTV Surveillance Frames",
        "epochs": epochs,
        "imgsz": imgsz,
        "all_classes_mAP50": round(float(val_metrics.box.map50), 4),
        "all_classes_mAP50_95": round(float(val_metrics.box.map), 4),
        "mean_precision": round(float(val_metrics.box.mp), 4),
        "mean_recall": round(float(val_metrics.box.mr), 4),
        "class_breakdown": class_results,
        "weights_path": TARGET_WEIGHTS
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"📊 Training & Per-Class Metrics saved to: {METRICS_PATH}")
    return overall_metrics

if __name__ == "__main__":
    train_enhanced_ppe_model(epochs=15, imgsz=640, batch=16)

