WEIGHTS = {
    'knn': 1.0,
    'goal': 1.0,
    'level': 1.0,
    'equipment': 0.8,
    'program_length': 0.6,
    'time_per_workout': 0.6
}

def overlap_score(user_list, program_list):
    if not user_list or not program_list:
        return 0
    return len(set(user_list) & set(program_list)) / len(set(user_list) | set(program_list))

def numeric_similarity(user_value, program_value, scale=1.0):
    return 1 / (1 + abs(user_value - program_value) / scale)

def total_score(knn_score, score_goal=0, score_level=0, score_equipment=0, score_length=0, score_time=0):
    total_score = (
        WEIGHTS['knn'] * knn_score +
        WEIGHTS['goal'] * score_goal +
        WEIGHTS['level'] * score_level +
        WEIGHTS['equipment'] * score_equipment +
        WEIGHTS['program_length'] * score_length +
        WEIGHTS['time_per_workout'] * score_time
    )
    return total_score