"""
VigilOps AI - Automated LangGraph Multi-Agent Orchestration Test
Validates:
1. Compilation of the 5-node LangGraph StateGraph.
2. State transitions across all 5 specialized agents:
   - Detection Validation Agent
   - Risk Assessment Agent
   - Safety Compliance Agent
   - Emergency Response Agent
   - Incident Reporting Agent
3. Human-readable operational decision logs.
4. Escalation triggering on critical threat scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_brain import agent_brain, run_agent_reasoning

def run_langgraph_tests():
    print("=================================================================")
    print("🤖 LANGGRAPH 5-AGENT DECISION ORCHESTRATION TEST")
    print("=================================================================")

    # Test Case 1: Normal Ingestion (Low Risk)
    print("\n[Case 1] Normal compliant sector surveillance:")
    normal_detections = [
        {"class": "person", "track_id": 1, "confidence": 0.91, "bbox_normalized": [0.2, 0.2, 0.4, 0.7]}
    ]
    res1 = run_agent_reasoning(
        source_id="CAM_02: Packaging Sector",
        raw_detections=normal_detections,
        duration=0.0,
        proximity=0.10,
        worker_speed=0.9
    )

    print(f"   ↳ Threat Level: {res1['threat_level']} | Score: {int(res1['predictive_risk_score']*100)}%")
    print(f"   ↳ Escalation Required: {res1['escalation_required']}")
    print(f"   ↳ Operational Stages: {len(res1['operational_stages'])} stages executed")
    print(f"   ↳ Operational Logs: {len(res1['agent_logs'])} entries")
    assert res1['threat_level'] in ["LOW", "MODERATE"]
    assert not res1['escalation_required']
    assert len(res1['operational_stages']) == 5, f"Expected 5 stages but got {len(res1['operational_stages'])}"

    # Test Case 2: Critical Suspended Crane Load Breach
    print("\n[Case 2] Critical suspended crane load perimeter breach:")
    from risk_engine import risk_engine
    risk_engine.reset()
    critical_detections = [
        {"class": "crane", "track_id": 5, "confidence": 0.95, "bbox_normalized": [0.1, 0.05, 0.9, 0.35]},
        {"class": "no_helmet", "track_id": 2, "confidence": 0.92, "bbox_normalized": [0.45, 0.40, 0.55, 0.80]}
    ]
    res2 = run_agent_reasoning(
        source_id="CAM_01: Overhead Crane Hoist",
        raw_detections=critical_detections,
        duration=15.0,
        proximity=0.95,
        worker_speed=1.8
    )

    print(f"   ↳ Threat Level: {res2['threat_level']} | Score: {int(res2['predictive_risk_score']*100)}%")
    print(f"   ↳ Escalation Required: {res2['escalation_required']}")
    print(f"   ↳ Action Plan: {res2['action_plan']}")
    print(f"   ↳ SOP: {res2['retrieved_sop']['standard_code']} — {res2['retrieved_sop']['title']}")

    print("\n📋 Detailed Operational Decision Timeline:")
    for log in res2['agent_logs']:
        print(f"   {log}")

    assert res2['escalation_required'] == True, "Expected escalation on critical incident"
    assert len(res2['action_plan']) >= 2, "Missing action plan steps"

    print("\n-----------------------------------------------------------------")
    print("=================================================================")
    print("🎉 LANGGRAPH 5-AGENT DECISION ORCHESTRATION TEST: PASSED")
    print("=================================================================\n")

if __name__ == "__main__":
    run_langgraph_tests()
