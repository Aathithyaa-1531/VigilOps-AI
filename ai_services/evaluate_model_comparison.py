"""
VigilOps AI - 3-Model Independent Test Evaluation Comparison
Compares:
1. Baseline Model (models/yolov8n_ppe_legacy_backup.pt)
2. Previous Fine-Tuned Model (models/yolov8n_ppe_finetuned.pt)
3. Improved Violation-Enhanced Model (models/yolov8n_ppe_improved.pt)

Evaluates on the untouched independent test set (144 images, 741 instances).
Computes exact per-class Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95.
Generates comprehensive comparative JSON and Markdown table.
"""

import os
import json
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "datasets", "ppe_finetune", "data.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "ppe_error_analysis")

MODELS = {
    "Baseline Pretrained": os.path.join(MODELS_DIR, "yolov8n_ppe_legacy_backup.pt"),
    "Previous Fine-Tuned": os.path.join(MODELS_DIR, "yolov8n_ppe_finetuned.pt"),
    "Improved Violation-Enhanced": os.path.join(MODELS_DIR, "yolov8n_ppe_improved.pt")
}

CLASS_NAMES = {
    0: "hard_hat",
    1: "no_helmet",
    2: "reflective_jacket",
    3: "no_vest",
    4: "person",
    5: "forklift",
    6: "crane"
}

def evaluate_models():
    os.makedirs(RUNS_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("==================================================================")
    print("🔬 VIGILOPS AI — 3-MODEL INDEPENDENT TEST SET EVALUATION")
    print("==================================================================")
    print(f"Dataset : {DATA_YAML} (Split: TEST, 144 images, 741 instances)")
    print(f"Device  : {device}")
    print("==================================================================\n")

    full_results = {}

    for label, mpath in MODELS.items():
        if not os.path.exists(mpath):
            print(f"⚠️ Model checkpoint not found for {label}: {mpath}")
            continue

        print(f"\n▶ Evaluating: {label}")
        print(f"  Checkpoint: {mpath} ({os.path.getsize(mpath)} bytes)")

        model = YOLO(mpath)
        val_name = f"eval_{label.lower().replace(' ', '_').replace('-', '_')}"
        
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=640,
            device=device,
            project=RUNS_DIR,
            name=val_name,
            exist_ok=True,
            verbose=False
        )

        class_metrics = {}
        for i, cname in CLASS_NAMES.items():
            p = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
            r = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
            map50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
            map95 = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

            class_metrics[cname] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map95, 4)
            }

        all_p = float(metrics.box.mp)
        all_r = float(metrics.box.mr)
        all_map50 = float(metrics.box.map50)
        all_map95 = float(metrics.box.map)
        all_f1 = 2 * (all_p * all_r) / (all_p + all_r) if (all_p + all_r) > 0 else 0.0

        full_results[label] = {
            "all": {
                "precision": round(all_p, 4),
                "recall": round(all_r, 4),
                "f1_score": round(all_f1, 4),
                "mAP50": round(all_map50, 4),
                "mAP50_95": round(all_map95, 4)
            },
            "classes": class_metrics
        }

    # Save to comparison JSON
    comp_json_path = os.path.join(RUNS_DIR, "three_model_comparison_metrics.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n\n==================================================================")
    print("📊 COMPREHENSIVE 3-MODEL COMPARATIVE TEST METRICS")
    print("==================================================================")
    
    header = f"{'Metric / Class':<22} | {'Baseline Pretrained':<20} | {'Previous Fine-Tuned':<20} | {'Improved Violation':<20}"
    print(header)
    print("-" * len(header))

    key_rows = [
        ("no_helmet Recall", lambda r: f"{r['classes']['no_helmet']['recall']*100:.2f}%"),
        ("no_helmet mAP50", lambda r: f"{r['classes']['no_helmet']['mAP50']*100:.2f}%"),
        ("no_helmet Prec", lambda r: f"{r['classes']['no_helmet']['precision']*100:.2f}%"),
        ("no_vest Recall", lambda r: f"{r['classes']['no_vest']['recall']*100:.2f}%"),
        ("no_vest mAP50", lambda r: f"{r['classes']['no_vest']['mAP50']*100:.2f}%"),
        ("no_vest Prec", lambda r: f"{r['classes']['no_vest']['precision']*100:.2f}%"),
        ("hard_hat Recall", lambda r: f"{r['classes']['hard_hat']['recall']*100:.2f}%"),
        ("hard_hat Prec", lambda r: f"{r['classes']['hard_hat']['precision']*100:.2f}%"),
        ("reflective_jacket Recall", lambda r: f"{r['classes']['reflective_jacket']['recall']*100:.2f}%"),
        ("reflective_jacket Prec", lambda r: f"{r['classes']['reflective_jacket']['precision']*100:.2f}%"),
        ("person Recall", lambda r: f"{r['classes']['person']['recall']*100:.2f}%"),
        ("ALL CLASSES mAP50", lambda r: f"{r['all']['mAP50']*100:.2f}%"),
        ("ALL CLASSES Recall", lambda r: f"{r['all']['recall']*100:.2f}%"),
        ("ALL CLASSES Prec", lambda r: f"{r['all']['precision']*100:.2f}%"),
    ]

    for label, fn in key_rows:
        val_base = fn(full_results["Baseline Pretrained"]) if "Baseline Pretrained" in full_results else "N/A"
        val_prev = fn(full_results["Previous Fine-Tuned"]) if "Previous Fine-Tuned" in full_results else "N/A"
        val_imp = fn(full_results["Improved Violation-Enhanced"]) if "Improved Violation-Enhanced" in full_results else "N/A"
        print(f"{label:<22} | {val_base:<20} | {val_prev:<20} | {val_imp:<20}")
    print("==================================================================\n")
    print(f"📄 Full comparative results saved to: {comp_json_path}")

if __name__ == "__main__":
    evaluate_models()

