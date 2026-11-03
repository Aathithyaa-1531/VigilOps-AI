"""
VigilOps AI - Actual YOLOv8 PPE Fine-Tuning Execution
Trains YOLOv8 Nano on the verified unified 7-class PPE dataset (1,483 images, 7,373 instances).
Applies industrial data augmentations (Mosaic, Mixup, HSV shifts, Flips).
Saves best checkpoints, monitors loss curves, and generates full test evaluation.
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
TARGET_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n_ppe_finetuned.pt")
METRICS_PATH = os.path.join(MODELS_DIR, "finetuned_ppe_metrics.json")

def execute_fine_tuning(epochs=25, imgsz=640, batch=16):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"==================================================================")
    print(f"🚀 VIGILOPS AI — ACTUAL YOLOv8 PPE FINE-TUNING PIPELINE")
    print(f"==================================================================")
    print(f"⚡ Hardware Accelerator : {device} (Apple Silicon GPU)")
    print(f"📁 Dataset Config       : {DATA_YAML}")
    print(f"🎯 Target Checkpoint    : {TARGET_WEIGHTS}")
    print(f"⚙️ Hyperparameters      : Epochs={epochs}, ImgSz={imgsz}, Batch={batch}")
    print(f"   Augmentation Params  : Mosaic=1.0, Mixup=0.15, FlipLR=0.5, HSV=(0.015, 0.7, 0.4)")
    print(f"==================================================================\n")

    # Start from pretrained weights
    pretrained_weights = os.path.join(MODELS_DIR, "yolov8n_ppe.pt")
    if not os.path.exists(pretrained_weights):
        pretrained_weights = os.path.join(BASE_DIR, "yolov8n.pt")

    print(f"📦 Initializing model from checkpoint: {pretrained_weights}")
    model = YOLO(pretrained_weights)

    start_time = time.time()
    
    # Execute actual fine-tuning
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        project=RUNS_DIR,
        name="finetuned_ppe_yolov8_run",
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True,
        save_period=5,
        # Applied Industrial Augmentations
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0
    )

    elapsed_sec = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} minutes).")

    # Locate the best weights
    best_weights = os.path.join(RUNS_DIR, "finetuned_ppe_yolov8_run", "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(RUNS_DIR, "finetuned_ppe_yolov8_run", "weights", "last.pt")

    if os.path.exists(best_weights):
        shutil.copy(best_weights, TARGET_WEIGHTS)
        print(f"🏆 Successfully saved best checkpoint to: {TARGET_WEIGHTS} ({os.path.getsize(TARGET_WEIGHTS)} bytes)")

    # Execute rigorous independent evaluation on the TEST split (144 images, 741 instances)
    print("\n🔍 Running Independent Evaluation on TEST split using fine-tuned model...")
    eval_model = YOLO(TARGET_WEIGHTS)
    test_metrics = eval_model.val(data=DATA_YAML, split="test", imgsz=imgsz, device=device, verbose=True)

    class_results = {}
    print("\n================ INDEPENDENT PER-CLASS TEST RESULTS ================")
    for i, name in eval_model.names.items():
        p = float(test_metrics.box.p[i]) if i < len(test_metrics.box.p) else 0.0
        r = float(test_metrics.box.r[i]) if i < len(test_metrics.box.r) else 0.0
        map50 = float(test_metrics.box.ap50[i]) if i < len(test_metrics.box.ap50) else 0.0
        map50_95 = float(test_metrics.box.ap[i]) if i < len(test_metrics.box.ap) else 0.0
        class_results[name] = {
            "class_id": i,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4)
        }
        print(f"  Class {i} ({name:18s}): P={p:.4f} | R={r:.4f} | mAP50={map50:.4f} | mAP50-95={map50_95:.4f}")
    print("====================================================================\n")

    summary_metrics = {
        "model_architecture": "YOLOv8 Nano (Fine-Tuned Industrial PPE)",
        "training_duration_seconds": round(elapsed_sec, 2),
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch,
        "all_classes_mAP50": round(float(test_metrics.box.map50), 4),
        "all_classes_mAP50_95": round(float(test_metrics.box.map), 4),
        "mean_precision": round(float(test_metrics.box.mp), 4),
        "mean_recall": round(float(test_metrics.box.mr), 4),
        "class_breakdown": class_results,
        "best_weights_path": TARGET_WEIGHTS,
        "run_dir": os.path.join(RUNS_DIR, "finetuned_ppe_yolov8_run")
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"📊 Evaluation metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    execute_fine_tuning(epochs=25, imgsz=640, batch=16)

