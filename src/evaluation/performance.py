"""H3 — Model performance evaluation with bootstrap confidence intervals."""

import numpy as np
from loguru import logger

from src.schemas.evaluation import ConfidenceInterval, PerformanceResult


class PerformanceEvaluator:
    """Compute F1 metrics with bootstrap CIs for model comparison.

    Evaluates H3: How do models compare on per-technique and aggregate F1?
    """

    def __init__(self, bootstrap_iterations: int = 10_000, seed: int = 42) -> None:
        self._B = bootstrap_iterations
        self._seed = seed

    def evaluate(
        self,
        gold_labels: dict[str, set[str]],
        predictions: dict[str, set[str]],
        technique_names: list[str],
        model_id: str,
    ) -> PerformanceResult:
        """Compute per-technique P/R/F1 and aggregate metrics with bootstrap CI.

        Args:
            gold_labels: segment_id → set of gold techniques
            predictions: segment_id → set of predicted techniques
            technique_names: List of technique names
            model_id: Model identifier

        Returns:
            PerformanceResult with all metrics.
        """
        segments = sorted(set(gold_labels.keys()) & set(predictions.keys()))

        # Per-technique metrics
        per_technique: dict[str, dict[str, float]] = {}
        for technique in technique_names:
            tp = fp = fn = 0
            for seg in segments:
                gold_has = technique in gold_labels.get(seg, set())
                pred_has = technique in predictions.get(seg, set())
                if gold_has and pred_has:
                    tp += 1
                elif pred_has and not gold_has:
                    fp += 1
                elif gold_has and not pred_has:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_technique[technique] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        # Macro-F1
        f1_values = [per_technique[t]["f1"] for t in technique_names]
        macro_f1 = float(np.mean(f1_values))

        # Micro-F1
        total_tp = sum(
            1
            for seg in segments
            for t in technique_names
            if t in gold_labels.get(seg, set()) and t in predictions.get(seg, set())
        )
        total_fp = sum(
            1
            for seg in segments
            for t in technique_names
            if t not in gold_labels.get(seg, set()) and t in predictions.get(seg, set())
        )
        total_fn = sum(
            1
            for seg in segments
            for t in technique_names
            if t in gold_labels.get(seg, set()) and t not in predictions.get(seg, set())
        )
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r) > 0
            else 0.0
        )

        # Bootstrap CI for macro-F1
        macro_f1_ci = self._bootstrap_macro_f1(
            segments, gold_labels, predictions, technique_names
        )

        return PerformanceResult(
            model_id=model_id,
            per_technique=per_technique,
            macro_f1=round(macro_f1, 4),
            micro_f1=round(micro_f1, 4),
            macro_f1_ci=macro_f1_ci,
            bootstrap_iterations=self._B,
        )

    def _bootstrap_macro_f1(
        self,
        segments: list[str],
        gold_labels: dict[str, set[str]],
        predictions: dict[str, set[str]],
        technique_names: list[str],
    ) -> ConfidenceInterval:
        """Compute bootstrap 95% CI for macro-F1."""
        rng = np.random.RandomState(self._seed)
        n = len(segments)
        bootstrap_f1s = []

        for _ in range(self._B):
            # Resample with replacement
            indices = rng.randint(0, n, size=n)
            sample_segments = [segments[i] for i in indices]

            # Compute macro-F1 on this resample
            f1s = []
            for technique in technique_names:
                tp = fp = fn = 0
                for seg in sample_segments:
                    gold_has = technique in gold_labels.get(seg, set())
                    pred_has = technique in predictions.get(seg, set())
                    if gold_has and pred_has:
                        tp += 1
                    elif pred_has:
                        fp += 1
                    elif gold_has:
                        fn += 1
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                f1s.append(f1)
            bootstrap_f1s.append(np.mean(f1s))

        lower = float(np.percentile(bootstrap_f1s, 2.5))
        upper = float(np.percentile(bootstrap_f1s, 97.5))

        return ConfidenceInterval(lower=round(lower, 4), upper=round(upper, 4))
