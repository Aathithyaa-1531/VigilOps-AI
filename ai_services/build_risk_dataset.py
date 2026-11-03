"""
Industrial Safety and Health Analytics Database Generator
Constructs a primary tabular dataset (.csv) based on authentic EHS observational metrics.
"""

import os
import csv
import random
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "risk_analytics")
os.makedirs(DATASET_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATASET_DIR, "industrial_safety_and_health_database.csv")

def generate_risk_dataset(num_samples: int = 3500):
    np.random.seed(42)
    random.seed(42)

    sectors = ["Manufacturing", "Metal Fabrication", "Chemical Processing", "Construction", "Logistics & Warehousing", "Automotive Assembly"]
    hazard_types = [
        ("NORMAL_COMPLIANT", 0.10, 0),
        ("NO_VEST", 0.50, 1),
        ("NO_HELMET", 0.85, 2),
        ("FORKLIFT_PROXIMITY", 0.88, 2),
        ("CRANE_SUSPENDED_LOAD", 0.98, 2),
        ("RESTRICTED_LOTO_BREACH", 0.95, 2),
        ("FALL_HAZARD", 0.90, 2),
    ]

    shifts = ["Day Shift (06:00-14:00)", "Evening Shift (14:00-22:00)", "Night Shift (22:00-06:00)"]

    rows = []
    header = [
        "incident_id",
        "industry_sector",
        "hazard_type",
        "violation_duration_sec",
        "proximity_score",
        "hazard_severity",
        "worker_speed_mps",
        "ppe_compliance",
        "shift_type",
        "ambient_noise_db",
        "lighting_lux",
        "risk_score",
        "risk_level"  # 0=LOW, 1=MODERATE, 2=CRITICAL
    ]

    for i in range(1, num_samples + 1):
        incident_id = f"EHS-LOG-{i:05d}"
        sector = random.choice(sectors)
        hazard_name, base_severity, expected_class = random.choices(
            hazard_types,
            weights=[0.35, 0.20, 0.15, 0.12, 0.08, 0.06, 0.04]
        )[0]

        if expected_class == 0:
            duration = round(float(np.random.uniform(0.0, 2.5)), 2)
            proximity = round(float(np.random.uniform(0.0, 0.35)), 3)
            severity = round(float(np.random.uniform(0.05, 0.25)), 2)
            speed = round(float(np.random.uniform(0.6, 1.4)), 2)
            ppe = "COMPLIANT"
            noise = round(float(np.random.uniform(60, 75)), 1)
            lux = round(float(np.random.uniform(450, 800)), 1)
            calc_risk = 0.30 * (duration / 15.0) + 0.35 * proximity + 0.25 * severity + 0.10 * (speed / 3.0)
            risk_score = round(float(np.clip(calc_risk, 0.02, 0.38)), 2)
            risk_level = 0
        elif expected_class == 1:
            duration = round(float(np.random.uniform(2.0, 10.0)), 2)
            proximity = round(float(np.random.uniform(0.30, 0.70)), 3)
            severity = round(float(np.random.uniform(0.40, 0.65)), 2)
            speed = round(float(np.random.uniform(0.9, 1.8)), 2)
            ppe = "PARTIAL"
            noise = round(float(np.random.uniform(70, 88)), 1)
            lux = round(float(np.random.uniform(300, 600)), 1)
            calc_risk = 0.35 * (duration / 15.0) + 0.35 * proximity + 0.25 * severity + 0.05 * (speed / 3.0)
            risk_score = round(float(np.clip(calc_risk, 0.41, 0.74)), 2)
            risk_level = 1
        else: # Critical
            duration = round(float(np.random.uniform(6.0, 30.0)), 2)
            proximity = round(float(np.random.uniform(0.70, 0.99)), 3)
            severity = round(float(np.random.uniform(0.75, 1.00)), 2)
            speed = round(float(np.random.uniform(1.2, 3.2)), 2)
            ppe = "NONE" if "NO_" in hazard_name else "PARTIAL"
            noise = round(float(np.random.uniform(80, 105)), 1)
            lux = round(float(np.random.uniform(120, 450)), 1)
            calc_risk = 0.35 * min(duration / 15.0, 1.0) + 0.40 * proximity + 0.25 * severity
            calc_risk += np.random.uniform(0.02, 0.10)
            risk_score = round(float(np.clip(calc_risk, 0.76, 0.99)), 2)
            risk_level = 2

        shift = random.choice(shifts)

        rows.append([
            incident_id, sector, hazard_name, duration, proximity, severity,
            speed, ppe, shift, noise, lux, risk_score, risk_level
        ])

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ Generated Industrial Safety and Health Analytics Database at: {CSV_PATH}")
    print(f"   Total Tabular Records: {len(rows)} | Features: {len(header)}")

if __name__ == "__main__":
    generate_risk_dataset()
