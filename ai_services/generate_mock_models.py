import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def generate_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"📁 Target models directory verified: {models_dir}")

    print("⏳ Synthesizing Scikit-learn Multi-Factor Risk Engine...")
    # Training samples: [violation_duration (s), proximity_score (0-1), hazard_severity (0-1), worker_speed (m/s)]
    X_train = np.array([
        # Low risk scenarios (class 0)
        [0.5, 0.1, 0.2, 0.8],
        [1.0, 0.2, 0.3, 0.9],
        [2.0, 0.15, 0.2, 0.7],
        [0.0, 0.0, 0.1, 0.5],
        [3.0, 0.25, 0.3, 1.0],

        # Moderate risk scenarios (class 1)
        [5.0, 0.5, 0.6, 1.2],
        [8.0, 0.6, 0.5, 1.4],
        [10.0, 0.55, 0.7, 1.1],
        [6.0, 0.7, 0.6, 1.5],

        # Critical emergency scenarios (class 2)
        [12.0, 0.85, 0.9, 1.8],
        [15.0, 0.95, 1.0, 2.2],
        [20.0, 0.90, 0.95, 2.0],
        [18.0, 0.88, 1.0, 1.9],
        [25.0, 0.99, 1.0, 2.5]
    ])

    # Target class: 0=LOW, 1=MODERATE, 2=CRITICAL
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    rf_model = RandomForestClassifier(n_estimators=25, random_state=42)
    rf_model.fit(X_train, y_train)

    model_path = os.path.join(models_dir, "risk_rf.pkl")
    joblib.dump(rf_model, model_path)
    print(f"✅ Saved Scikit-learn Risk Engine -> {model_path}")

    # Generate placeholder for temporal Keras model compatibility
    keras_path = os.path.join(models_dir, "temporal_keras.h5")
    with open(keras_path, "w") as f:
        f.write("TEMPORAL_KERAS_MODEL_PLACEHOLDER")
    print(f"✅ Created mock placeholder -> {keras_path}")

if __name__ == "__main__":
    generate_models()
