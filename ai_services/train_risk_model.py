"""
VigilOps AI - Training Pipeline for Predictive Threat Risk Engine
Trains Scikit-learn Classifier on Primary Tabular Dataset: Industrial Safety and Health Analytics Database (.csv).
Serializes model to models/risk_rf.pkl with validation metrics.
"""

import os
import json
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "risk_analytics", "industrial_safety_and_health_database.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "risk_rf.pkl")
METRICS_OUT_PATH = os.path.join(MODELS_DIR, "risk_model_metrics.json")

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    X = []
    y = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration = float(row["violation_duration_sec"])
            proximity = float(row["proximity_score"])
            severity = float(row["hazard_severity"])
            speed = float(row["worker_speed_mps"])
            label = int(row["risk_level"])
            X.append([duration, proximity, severity, speed])
            y.append(label)

    return np.array(X), np.array(y)

def train_risk_model():
    print(f"📊 Ingesting Primary Dataset: {DATASET_PATH}...")
    X, y = load_data()
    print(f"   Loaded {len(X)} samples with 4 core physical-behavioral features: [duration, proximity, severity, worker_speed]")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    print("🧠 Fitting Scikit-learn Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"   5-Fold Cross Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["LOW (0)", "MODERATE (1)", "CRITICAL (2)"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    feature_importances = dict(zip(
        ["violation_duration", "proximity_score", "hazard_severity", "worker_speed"],
        [round(float(val), 4) for val in rf.feature_importances_]
    ))

    print(f"\n🎯 Test Set Accuracy: {acc * 100:.2f}%")
    print("📈 Feature Importances:")
    for feat, imp in feature_importances.items():
        print(f"   - {feat}: {imp * 100:.2f}%")

    # Serialize trained model
    joblib.dump(rf, MODEL_OUT_PATH)
    print(f"\n💾 Saved trained Risk Engine model to: {MODEL_OUT_PATH}")

    # Save metrics JSON
    metrics_data = {
        "dataset": "Industrial Safety and Health Analytics Database (.csv)",
        "samples_total": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(acc), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "feature_importances": feature_importances,
        "classification_report": rep,
        "confusion_matrix": cm
    }

    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"📋 Saved model evaluation metrics to: {METRICS_OUT_PATH}")

    return rf, metrics_data

if __name__ == "__main__":
    train_risk_model()
