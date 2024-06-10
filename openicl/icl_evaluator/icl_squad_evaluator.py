"""Squad Evaluator -> also return precision & recall """
from openicl.icl_evaluator import BaseEvaluator
import re
import string
from collections import Counter


def normalize_answer(s) :
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text) :
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text) :
        return " ".join(text.split())

    def remove_punc(text) :
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text) :
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_scores(prediction, ground_truth) :
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 :
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def f1_score(prediction, ground_truth) :
    f1, _, _ = get_scores(prediction, ground_truth)
    return f1


def precision_score(prediction, ground_truth) :
    _, precision, _ = get_scores(prediction, ground_truth)
    return precision


def recall_score(prediction, ground_truth) :
    _, _, recall = get_scores(prediction, ground_truth)
    return recall


def exact_match_score(prediction, ground_truth) :
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths) :
    scores_for_ground_truths = []
    for ground_truth in ground_truths :
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_score(predictions, references) :
    f1 = precision = recall = exact_match = total = 0

    for pred, ref in zip(predictions, references):
        total += 1
        ground_truths = list(ref["answers"]["text"])
        prediction = pred['prediction_text']
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        precision += metric_max_over_ground_truths(precision_score, prediction, ground_truths)
        recall += metric_max_over_ground_truths(recall_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    precision = 100.0 * precision / total
    recall = 100.0 * recall / total

    return {'exact_match' : exact_match, 'f1' : f1, 'precision' : precision, 'recall' : recall}


class SquadEvaluator(BaseEvaluator) :
    def __init__(self) -> None :
        super().__init__()

    def score(self, predictions, references) :
        assert len(predictions) == len(references)
        p_list = [{'prediction_text' : pred, 'id' : str(i)} for i, pred in
                  enumerate(predictions)]
        r_list = [{'answers' : {'answer_start' : [0], 'text' : [ref]}, 'id' : str(i)} for i, ref in
                  enumerate(references)]
        # metric = evaluate.load('squad')
        scores = compute_score(predictions=p_list, references=r_list)
        return scores
