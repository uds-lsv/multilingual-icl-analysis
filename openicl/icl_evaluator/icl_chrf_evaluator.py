"""ChrF++ evaluator"""
from openicl.icl_evaluator import BaseEvaluator
import evaluate


class ChrfEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        metric = evaluate.load("chrf")
        scores = metric.compute(predictions=predictions, references=references, word_order=2)['score']
        return {'chrf': scores}
