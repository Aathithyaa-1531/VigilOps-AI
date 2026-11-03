"""
VigilOps AI - Improved YOLOv8 PPE Fine-Tuning Pipeline
Trains on the balanced, augmented 7-class PPE dataset with enhanced violation representation.
Evaluates on the untouched independent test set.
Saves best checkpoint to models/yolov8n_ppe_improved.pt and logs all metrics.
"""

import os
import shutil
import json
import time
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "datasets", "ppe_finetune", "data.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
START_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n_ppe_finetuned.pt")
IMPROVED_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n_ppe_improved.pt")
IMPROVED_METRICS_PATH = os.path.join(RUNS_DIR, "ppe_error_analysis", "improved_model_metrics.json")

def execute_improved_training(epochs=5, imgsz=640, batch=16):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUNS_DIR, "ppe_error_analysis"), exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("==================================================================")
    print("🚀 VIGILOPS AI — ENHANCED YOLOv8 PPE MODEL TRAINING")
    print("==================================================================")
    print(f"⚡ Hardware Accelerator : {device} (Apple Silicon GPU)")
    print(f"📁 Dataset Config       : {DATA_YAML}")
    print(f"📦 Starting Checkpoint  : {START_WEIGHTS}")
    print(f"🎯 Target Checkpoint    : {IMPROVED_WEIGHTS}")
    print(f"⚙️ Hyperparameters      : Epochs={epochs}, ImgSz={imgsz}, Batch={batch}")
    print(f"   Optimization Params  : cls=1.0, close_mosaic=5, patience=8")
    print("==================================================================\n")

    if not os.path.exists(START_WEIGHTS):
        raise FileNotFoundError(f"Starting weights not found: {START_WEIGHTS}")

    model = YOLO(START_WEIGHTS)
    start_time = time.time()

    # Execute training
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        project=RUNS_DIR,
        name="improved_ppe_yolov8_run",
        exist_ok=True,
        verbose=True,
        patience=8,
        save=True,
        save_period=3,
        # Augmentation & Loss Calibration
        cls=1.0,            # Balanced classification loss
        close_mosaic=1,     # Turn off mosaic in final epoch for sharp bounding box localization
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5
    )

    elapsed_sec = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} minutes).")

    # Locate and save best checkpoint
    best_weights = os.path.join(RUNS_DIR, "improved_ppe_yolov8_run", "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(RUNS_DIR, "improved_ppe_yolov8_run", "weights", "last.pt")

    if os.path.exists(best_weights):
        shutil.copy(best_weights, IMPROVED_WEIGHTS)
        print(f"🏆 Successfully saved best checkpoint to: {IMPROVED_WEIGHTS} ({os.path.getsize(IMPROVED_WEIGHTS)} bytes)")

    # Run Independent Validation on Untouched Test Set
    print("\n🔍 Evaluating Improved Model on Untouched Test Set (144 images, 741 instances)...")
    eval_model = YOLO(IMPROVED_WEIGHTS)
    test_metrics = eval_model.val(
        data=DATA_YAML,
        split="test",
        imgsz=imgsz,
        device=device,
        project=os.path.join(RUNS_DIR, "ppe_error_analysis"),
        name="improved_test_val",
        exist_ok=True,
        verbose=True
    )

    class_results = {}
    print("\n================ IMPROVED PER-CLASS TEST RESULTS ================")
    for i, name in eval_model.names.items():
        p = float(test_metrics.box.p[i]) if i < len(test_metrics.box.p) else 0.0
        r = float(test_metrics.box.r[i]) if i < len(test_metrics.box.r) else 0.0
        map50 = float(test_metrics.box.ap50[i]) if i < len(test_metrics.box.ap50) else 0.0
        map50_95 = float(test_metrics.box.ap[i]) if i < len(test_metrics.box.ap) else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        class_results[name] = {
            "class_id": i,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4)
        }
        print(f"  {name:18s}: Prec={p:.4f} | Rec={r:.4f} | F1={f1:.4f} | mAP50={map50:.4f} | mAP50-95={map50_95:.4f}")

    all_p = float(test_metrics.box.mp)
    all_r = float(test_metrics.box.mr)
    all_map50 = float(test_metrics.box.map50)
    all_map = float(test_metrics.box.map)
    all_f1 = 2 * (all_p * all_r) / (all_p + all_r) if (all_p + all_r) > 0 else 0.0

    print(f"  {'ALL CLASSES':18s}: Prec={all_p:.4f} | Rec={all_r:.4f} | F1={all_f1:.4f} | mAP50={all_map50:.4f} | mAP50-95={all_map:.4f}")
    print("=================================================================\n")

    summary = {
        "model_architecture": "YOLOv8 Nano (Fine-Tuned Violation-Enhanced PPE)",
        "training_duration_seconds": round(elapsed_sec, 2),
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch,
        "all_classes_mAP50": round(all_map50, 4),
        "all_classes_mAP50_95": round(all_map, 4),
        "mean_precision": round(all_p, 4),
        "mean_recall": round(all_r, 4),
        "mean_f1": round(all_f1, 4),
        "class_breakdown": class_results,
        "best_weights_path": IMPROVED_WEIGHTS,
        "run_dir": os.path.join(RUNS_DIR, "improved_ppe_yolov8_run")
    }

    with open(IMPROVED_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved improved metrics JSON to: {IMPROVED_METRICS_PATH}")

if __name__ == "__main__":
    execute_improved_training(epochs=5, imgsz=640, batch=16)
