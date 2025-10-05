import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model')

def load_artifacts():
    knn = joblib.load(os.path.join(MODEL_PATH, "knn_model.pkl"))
    mlb_goal = joblib.load(os.path.join(MODEL_PATH, "mlb_goal.pkl"))
    mlb_level = joblib.load(os.path.join(MODEL_PATH, "mlb_level.pkl"))
    le = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    equipment_cols = joblib.load(os.path.join(MODEL_PATH, "equipment_columns.pkl"))
    X_columns_loaded = joblib.load(os.path.join(MODEL_PATH, "feature_columns.pkl"))

    return {
        "knn": knn,
        "mlb_goal": mlb_goal,
        "mlb_level": mlb_level,
        "le": le,
        "scaler": scaler,
        "equipment_cols": equipment_cols,
        "X_columns_loaded": X_columns_loaded
    }
