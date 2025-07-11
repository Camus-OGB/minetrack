from joblib import load
import numpy as np

mlp_model = load("app/models/mlp_model.joblib")  # adapte si nécessaire

def predict_mlp(features: list[float]) -> float:
    X = np.array([features])
    return float(mlp_model.predict_proba(X)[0][1])
