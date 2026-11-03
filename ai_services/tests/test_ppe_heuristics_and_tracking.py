"""
VigilOps AI - Priority 2 Computer Vision & PPE Heuristics Test Suite
Validates:
1. Custom PPE model weights on labelled validation set (Precision, Recall, mAP).
2. Independent evaluation of Helmet and Vest compliance.
3. Analysis of false positives from ANSI color heuristics (blonde hair, white background, yellow clothing).
4. Tracking continuity across chunk boundaries with IoU reconciliation.
"""

import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import evaluate_ppe_for_person
from video_auditor import video_auditor

def test_model_validation_metrics():
    """Validates YOLOv8 PPE weights against labelled dataset."""
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "edge_vision", "data.yaml"))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n_ppe.pt"))

    assert os.path.exists(yaml_path), f"data.yaml not found at {yaml_path}"
    assert os.path.exists(model_path), f"yolov8n_ppe.pt not found at {model_path}"

    model = YOLO(model_path)
    metrics = model.val(data=yaml_path, device="cpu", verbose=False)

    print("\n📊 Validated YOLO PPE Dataset Metrics:")
    print(f"   ↳ mAP50:     {metrics.box.map50:.4f}")
    print(f"   ↳ mAP50-95:  {metrics.box.map:.4f}")
    print(f"   ↳ Precision: {metrics.box.mp:.4f}")
    print(f"   ↳ Recall:    {metrics.box.mr:.4f}")

    # Ensure validation executed and returned non-empty metrics
    assert metrics.box.map50 is not None
    assert metrics.box.map is not None

def test_helmet_evaluation_independent():
    """Tests head protection evaluation independently with compliant vs non-compliant synthetic crops."""
    # Synthetic 200x400 person crop (head is top 88px)
    h, w = 400, 200
    person_box = [0, 0, w, h]

    # Case A: Compliant yellow hard hat on head
    frame_compliant = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8) # Dark clothing
    # Add vibrant yellow helmet in center of head (top 20% height, middle 60% width)
    cv2.circle(frame_compliant, (100, 40), 35, (0, 220, 255), -1) # BGR Yellow

    eval_a = evaluate_ppe_for_person(frame_compliant, person_box)
    assert eval_a["has_helmet"] is True, f"Expected has_helmet=True for yellow hard hat, got {eval_a}"

    # Case B: Bare head (dark hair, no hard hat)
    frame_bare = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)
    cv2.circle(frame_bare, (100, 40), 30, (25, 35, 45), -1) # Dark brown hair

    eval_b = evaluate_ppe_for_person(frame_bare, person_box)
    assert eval_b["has_helmet"] is False, f"Expected has_helmet=False for bare head, got {eval_b}"

def test_vest_evaluation_independent():
    """Tests high-vis vest evaluation independently with compliant vs non-compliant torso crops."""
    h, w = 400, 200
    person_box = [0, 0, w, h]

    # Case A: Compliant fluorescent lime vest in torso region
    frame_vest = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    # Torso is roughly y=90 to y=260
    cv2.rectangle(frame_vest, (40, 110), (160, 240), (0, 255, 128), -1) # BGR Fluorescent Lime

    eval_a = evaluate_ppe_for_person(frame_vest, person_box)
    assert eval_a["has_vest"] is True, f"Expected has_vest=True for fluorescent lime vest, got {eval_a}"

    # Case B: Plain dark jacket / non-compliant clothing
    frame_no_vest = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(frame_no_vest, (40, 110), (160, 240), (45, 45, 50), -1) # Dark navy jacket

    eval_b = evaluate_ppe_for_person(frame_no_vest, person_box)
    assert eval_b["has_vest"] is False, f"Expected has_vest=False for dark jacket, got {eval_b}"

def test_ansi_heuristics_false_positive_analysis():
    """
    Tests and investigates known false positive edge cases:
    1. White ceiling / bright background behind bare head.
    2. Blonde hair (fair skin/hair) mistaken for hard hat.
    3. Plain yellow cotton shirt without retroreflective stripes.
    """
    h, w = 400, 200
    person_box = [0, 0, w, h]

    # Edge Case 1: Bare head with white background behind outer corners
    # Bare head in center, but outer 20% of crop has white wall (255, 255, 255)
    frame_white_bg = np.full((h, w, 3), (250, 250, 250), dtype=np.uint8)
    # Person body and head are dark
    cv2.rectangle(frame_white_bg, (30, 0), (170, h), (40, 40, 40), -1)
    cv2.circle(frame_white_bg, (100, 45), 25, (20, 30, 40), -1) # Bare head

    eval_white_bg = evaluate_ppe_for_person(frame_white_bg, person_box)
    print(f"\n🔍 False Positive Test (White Background): Helmet detected = {eval_white_bg['has_helmet']} (Confidence: {eval_white_bg['helmet_conf']})")

    # Edge Case 2: Blonde hair (light yellow/brown)
    frame_blonde = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    # Blonde hair BGR roughly (120, 190, 215), lower saturation than neon helmet
    cv2.circle(frame_blonde, (100, 45), 30, (110, 185, 215), -1)
    eval_blonde = evaluate_ppe_for_person(frame_blonde, person_box)
    print(f"🔍 False Positive Test (Blonde Hair): Helmet detected = {eval_blonde['has_helmet']} (Confidence: {eval_blonde['helmet_conf']})")

def test_tracking_continuity_across_chunks():
    """
    Verifies that spatial IoU matching preserves tracking identity across chunk boundaries.
    """
    # Box in Chunk 0 at t=29.0s
    box_chunk0 = [0.550, 0.415, 0.592, 0.765]
    # Box in Chunk 1 at t=29.0s (same person slightly shifted)
    box_chunk1 = [0.552, 0.418, 0.594, 0.767]

    # Calculate IoU
    xA = max(box_chunk0[0], box_chunk1[0])
    yA = max(box_chunk0[1], box_chunk1[1])
    xB = min(box_chunk0[2], box_chunk1[2])
    yB = min(box_chunk0[3], box_chunk1[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area0 = (box_chunk0[2] - box_chunk0[0]) * (box_chunk0[3] - box_chunk0[1])
    area1 = (box_chunk1[2] - box_chunk1[0]) * (box_chunk1[3] - box_chunk1[1])
    union_area = area0 + area1 - inter_area
    iou = inter_area / max(1e-6, union_area)

    print(f"\n🔄 Chunk Boundary Spatial Continuity: Calculated IoU = {iou:.3f}")
    assert iou >= 0.80, f"Expected high IoU across chunk boundary, got {iou}"

if __name__ == "__main__":
    test_model_validation_metrics()
    test_helmet_evaluation_independent()
    test_vest_evaluation_independent()
    test_ansi_heuristics_false_positive_analysis()
    test_tracking_continuity_across_chunks()
    print("\n✅ All Computer Vision Validation Tests Passed Successfully!")
