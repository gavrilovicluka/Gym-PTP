import numpy as np
import pandas as pd
from .loader import load_artifacts
from .scoring import get_programs_with_score
from .normalize_words import normalize_words

artifacts = load_artifacts()
knn = artifacts["knn"]
mlb_goal = artifacts["mlb_goal"]
mlb_level = artifacts["mlb_level"]
le = artifacts["le"]
scaler = artifacts["scaler"]
equipment_cols = artifacts["equipment_cols"]
X_columns_loaded = artifacts["X_columns_loaded"]

def predict(goals, levels, equipment, program_length, time_per_workout):
    # Encode user input
    levels = normalize_words(levels)
    goals = normalize_words(goals)

    user_goal = mlb_goal.transform([goals])
    user_level = mlb_level.transform([levels])
    user_equipment = pd.get_dummies(pd.Series(equipment), prefix="equip").reindex(columns=equipment_cols, fill_value=0)
    user_numeric_df = pd.DataFrame(np.array([[program_length, time_per_workout]]), columns=['program_length', 'time_per_workout'])
    user_numeric_scaled = scaler.transform(user_numeric_df)

    user_X = np.hstack([user_goal, user_level, user_equipment.values, user_numeric_scaled])
    user_X_df = pd.DataFrame(user_X, columns=X_columns_loaded) 

    # KNN prediction
    distances, indices = knn.kneighbors(user_X_df)
    top_programs = le.inverse_transform(indices[0])

    scored_programs = get_programs_with_score(top_programs, distances, goals, levels, equipment, program_length, time_per_workout)

    return sorted(scored_programs, key=lambda x: x["total_score"], reverse=True)
