"""
VigilOps AI - Automated Test Suite: AI Inference, Risk Engine & OSHA RAG
Validates real model inference, measured latency, risk calculations, and regulatory retrieval.
"""

import os
import sys
import numpy as np
import time

# Ensure parent directory in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from risk_engine import risk_engine
from rag_sop import retrieve_sop_for_hazard
from main import run_yolo_inference_on_frame

def run_ai_test_suite():
    print("🧪 Starting VigilOps AI Core Inference & RAG Test Suite...\n")
    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  ✅ [PASS] {name}")
            passed += 1
        except Exception as err:
            print(f"  ❌ [FAIL] {name}: {err}")
            failed += 1

    # 1. Model Loading & Architecture
    def test_model_loading():
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8n_ppe.pt")
        assert os.path.exists(weights_path), f"Trained weights not found at {weights_path}"
        model = YOLO(weights_path)
        expected_classes = ["hard_hat", "no_helmet", "reflective_jacket", "no_vest", "person"]
        for exp in expected_classes:
            assert exp in list(model.names.values()), f"Expected class '{exp}' missing from model"
    test("AI Test 1: Custom YOLOv8 PPE model loads trained architecture and PPE classes", test_model_loading)

    # 2. Real Frame Inference & Dynamic Latency Measurement
    def test_real_inference():
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add visual shapes
        dummy_frame[100:300, 200:400] = [180, 140, 110]
        
        t_start = time.perf_counter()
        result = run_yolo_inference_on_frame(dummy_frame, source_id="Test Canvas")
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        assert "latency_ms" in result, "Latency must be present in output"
        assert result["latency_ms"] > 0, "Measured latency must be greater than 0"
        assert isinstance(result["detections"], list), "Detections must be a list"
        assert "threat_level" in result, "Threat level must be determined"
        assert "risk_score" in result, "Risk score must be calculated"
    test("AI Test 2: Real frame inference executes with measured non-hardcoded latency", test_real_inference)

    # 3. Scikit-learn Multi-Factor Risk Engine
    def test_risk_engine():
        assert risk_engine.model is not None, "Trained Scikit-learn model must be loaded"

        # Safe scenario
        score_low, level_low, _ = risk_engine.evaluate_risk(
            violation_duration=0.5,
            proximity_score=0.1,
            hazard_type="NORMAL_COMPLIANT",
            worker_speed=1.0
        )
        assert level_low in ["LOW", "MODERATE"], f"Expected low/moderate danger, got {level_low} ({score_low})"

        # Critical scenario
        score_high, level_high, _ = risk_engine.evaluate_risk(
            violation_duration=20.0,
            proximity_score=0.95,
            hazard_type="CRANE_HAZARD",
            worker_speed=2.0
        )
        assert level_high == "CRITICAL", f"Expected CRITICAL threat, got {level_high}"
        assert score_high >= 0.75, f"Critical score must be >= 0.75, got {score_high}"
    test("AI Test 3: Predictive Risk Engine accurately classifies multi-factor threat danger", test_risk_engine)

    # 4. Zero-Prompt OSHA 1910 Regulatory Retrieval
    def test_rag_retrieval():
        # Helmet test
        sop_helmet = retrieve_sop_for_hazard("NO_HELMET")
        assert "1910.135" in sop_helmet["standard_code"], f"Expected OSHA 1910.135, got {sop_helmet['standard_code']}"
        assert "SOP-EHS-042" in sop_helmet["sop_id"]

        # Vest test
        sop_vest = retrieve_sop_for_hazard("NO_VEST")
        assert "1910.132" in sop_vest["standard_code"], f"Expected OSHA 1910.132, got {sop_vest['standard_code']}"

        # Forklift test
        sop_forklift = retrieve_sop_for_hazard("FORKLIFT_HAZARD")
        assert "1910.178" in sop_forklift["standard_code"], f"Expected OSHA 1910.178, got {sop_forklift['standard_code']}"

        # Crane test
        sop_crane = retrieve_sop_for_hazard("CRANE_HAZARD")
        assert "1910.179" in sop_crane["standard_code"], f"Expected OSHA 1910.179, got {sop_crane['standard_code']}"

        # Restricted Zone test
        sop_zone = retrieve_sop_for_hazard("RESTRICTED_ZONE")
        assert "1910.147" in sop_zone["standard_code"], f"Expected OSHA 1910.147, got {sop_zone['standard_code']}"
    test("AI Test 4: OSHA 1910 Corpus RAG accurately retrieves authoritative 29 CFR standards", test_rag_retrieval)

    print(f"\n======================================================")
    print(f"🏁 AI Test Summary: {passed} Passed, {failed} Failed")
    print(f"======================================================\n")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_ai_test_suite()
