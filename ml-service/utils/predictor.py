import numpy as np
import pandas as pd
import os
from .loader import load_artifacts
from .scoring import overlap_score, numeric_similarity, WEIGHTS
from .normalize_words import normalize_words

artifacts = load_artifacts()
knn = artifacts["knn"]
mlb_goal = artifacts["mlb_goal"]
mlb_level = artifacts["mlb_level"]
le = artifacts["le"]
scaler = artifacts["scaler"]
equipment_cols = artifacts["equipment_cols"]
X_columns_loaded = artifacts["X_columns_loaded"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'model/programs_clean.csv')
program_summary_df = pd.read_csv(DATA_PATH)

def predict(goals, levels, equipment, program_length, time_per_workout, top_k=3):
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

    scored_programs = []
    for i, prog_name in enumerate(top_programs):
        prog_row = program_summary_df[program_summary_df['title'] == prog_name].iloc[0]
        knn_score = 1 / (1 + distances[0][i])

        score_goal = overlap_score(goals, prog_row['goal'])
        score_level = overlap_score(levels, prog_row['level'])
        score_equipment = overlap_score(equipment, [prog_row['equipment']])
        score_length = numeric_similarity(program_length, prog_row['program_length_orig'])
        score_time = numeric_similarity(time_per_workout, prog_row['time_per_workout_orig'])
        
        total_score = (
            WEIGHTS['knn'] * knn_score +
            WEIGHTS['goal'] * score_goal +
            WEIGHTS['level'] * score_level +
            WEIGHTS['equipment'] * score_equipment +
            WEIGHTS['program_length'] * score_length +
            WEIGHTS['time_per_workout'] * score_time
        )

        scored_programs.append({
            "program": prog_name,
            "total_score": total_score
        })

    return sorted(scored_programs, key=lambda x: x["total_score"], reverse=True)
