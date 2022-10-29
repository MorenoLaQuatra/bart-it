import re
from collections import Counter
import string

def normalize_answer(txt):
    def remove_articles(text):
        return re.sub(r'\b(il|lo|la|i|gli|le|l)\b', ' ', text)
    def remove_prepositions(text):
        return re.sub(r'\b(di|a|da|in|con|su|per|tra|fra)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(
        remove_punc(
            remove_prepositions(remove_articles(lower(txt)))
        )
    )

def normalize_answers(list_of_answers):
    normalized_answers = [ normalize_answer(answer) for answer in list_of_answers ]
    return normalized_answers

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max([metric_fn(prediction, ground_truth) for ground_truth in ground_truths])