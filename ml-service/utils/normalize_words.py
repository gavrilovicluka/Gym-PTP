import ast

def normalize_words(words_list):
    if isinstance(words_list, str):
        words_list = ast.literal_eval(words_list)
    return [g.lower().strip() for g in words_list]