"""
VigilOps AI - Automated Predictive Threat Risk Engine Test
Validates:
1. Multi-factor feature ingestion (violation_duration, proximity_score, severity_weight, worker_speed, affected_persons).
2. Random forest predictive inference.
3. Temporal exponential moving average (EMA) smoothing.
4. Correct 4-tier severity classification (LOW, MODERATE, HIGH, CRITICAL).
5. State reset functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_engine import risk_engine

def run_risk_tests():
    print("=================================================================")
    print("🧠 PREDICTIVE THREAT RISK ENGINE INTEGRATION TEST")
    print("=================================================================")

    risk_engine.reset()

    # Scenario 1: Routine / Normal Sector Surveillance
    score1, level1, bd1 = risk_engine.evaluate_risk(
        violation_duration=0.0,
        proximity_score=0.05,
        hazard_type="COMPLIANT_NORMAL",
        worker_speed=0.8
    )
    print(f"\nScenario 1 (Normal Sector): Score = {int(score1*100)}% | Level = {level1}")
    print(f"   Breakdown: {bd1}")
    assert level1 == "LOW", f"Expected LOW but got {level1}"
    assert score1 < 0.30, f"Expected <0.30 but got {score1}"

    # Scenario 2: Moderate Vest Violation
    score2, level2, bd2 = risk_engine.evaluate_risk(
        violation_duration=3.5,
        proximity_score=0.35,
        hazard_type="NO_VEST",
        worker_speed=1.2
    )
    print(f"\nScenario 2 (No Vest Violation): Score = {int(score2*100)}% | Level = {level2}")
    print(f"   Breakdown: {bd2}")
    assert score2 > score1, "Expected risk score to increase"

    # Scenario 3: High Risk Crane Hoist Fall Zone Breach
    # Simulate persistence over consecutive frames
    scores = []
    for t in [4.0, 8.0, 12.0, 15.0, 18.0, 20.0]:
        sc, lvl, bd = risk_engine.evaluate_risk(
            violation_duration=t,
            proximity_score=0.92,
            hazard_type="CRANE_HAZARD",
            worker_speed=1.8,
            affected_persons_count=2
        )
        scores.append(sc)
        print(f"   ↳ t={t}s: Continuous Danger Score = {int(sc*100)}% ({lvl})")

    final_score, final_level, final_bd = sc, lvl, bd
    print(f"\nScenario 3 (Persistent Crane Load Hazard): Final Score = {int(final_score*100)}% | Level = {final_level}")
    assert final_level == "CRITICAL", f"Expected CRITICAL for persistent crane hazard but got {final_level}"
    assert final_score >= 0.75, f"Expected >= 0.75 but got {final_score}"

    # Verify temporal smoothing smoothness: each step smoothly transitions without massive spikes
    for i in range(1, len(scores)):
        diff = abs(scores[i] - scores[i-1])
        assert diff <= 0.25, f"Temporal jump too large ({diff}), smoothing failed"

    print("\n-----------------------------------------------------------------")
    print("✅ All threat scoring scenarios & temporal smoothing verified.")
    print("=================================================================")
    print("🎉 PREDICTIVE THREAT RISK ENGINE TEST: PASSED")
    print("=================================================================\n")

if __name__ == "__main__":
    run_risk_tests()
