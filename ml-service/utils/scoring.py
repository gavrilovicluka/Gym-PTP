import os
import pickle
import joblib
import pandas as pd

WEIGHTS = {
    'knn': 1.0,
    'goal': 1.0,
    'level': 1.0,
    'equipment': 0.8,
    'program_length': 0.6,
    'time_per_workout': 0.6
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'model/programs_clean.pkl')
program_summary_df = pd.read_pickle(DATA_PATH)

def overlap_score(user_list, program_list): 
    if not user_list or not program_list:
        return 0
    return len(set(user_list) & set(program_list)) / len(set(user_list) | set(program_list))

def numeric_similarity(user_value, program_value, scale=1.0):
    return 1 / (1 + abs(user_value - program_value) / scale)

def get_programs_with_score(top_programs, distances, goals, levels, equipment, program_length, time_per_workout):
    scored_programs = []
    for i, prog_name in enumerate(top_programs):
        prog_row = program_summary_df[program_summary_df['title'] == prog_name].iloc[0]
        knn_score = 1 / (1 + distances[0][i])

        score_goal = overlap_score(goals, prog_row['goal'])
        score_level = overlap_score(levels, prog_row['level'])
        score_equipment = overlap_score(equipment, [prog_row['equipment']])
        score_length = numeric_similarity(program_length, prog_row['program_length_orig'])
        score_time = numeric_similarity(time_per_workout, prog_row['time_per_workout_orig'])
        
        # print(goals)
        # print(prog_row['goal'])

        # print(knn_score)
        # print(score_goal)
        # print(score_level)
        # print(score_equipment)
        # print(score_length)
        # print(score_time)

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

    return scored_programs