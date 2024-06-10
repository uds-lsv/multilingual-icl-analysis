"""API evaluator"""
from openicl.icl_evaluator import BaseEvaluator
import evaluate


class EMEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.metric = 'exact_match'

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        metric = evaluate.load(self.metric)
        scores = metric.compute(predictions=[p.lower() for p in predictions],
                                references=[r.lower() for r in references])
        return scores
