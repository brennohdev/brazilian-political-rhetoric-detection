import math

import numpy as np
from loguru import logger

from src.schemas.evaluation import BiasResult, ConfidenceInterval


class BiasEvaluator:
    """Evaluate political bias using minimal pairs and McNemar test.

    For each model × technique, compares detection rates on left-referent
    vs right-referent versions of the same text. Significant asymmetry
    indicates political bias.
    """

    def __init__(
        self, significance_level: float = 0.05, n_tests: int = 18
    ) -> None:
        self._alpha = significance_level
        self._n_tests = n_tests
        self._adjusted_alpha = significance_level / n_tests  # Bonferroni

    def evaluate(
        self,
        left_detected: list[bool],
        right_detected: list[bool],
        model_id: str,
        technique: str,
    ) -> BiasResult:
        """Compute McNemar test and OR for one model-technique pair.

        Args:
            left_detected: Whether technique was detected in left variant (per pair).
            right_detected: Whether technique was detected in right variant (per pair).
            model_id: Model identifier.
            technique: Technique name.

        Returns:
            BiasResult with test statistic, p-value, OR, and CI.
        """
        n = len(left_detected)
        assert len(right_detected) == n

        # Build contingency table
        # b = detected in left but NOT in right (bias against left)
        # c = detected in right but NOT in left (bias against right)
        b = sum(1 for l, r in zip(left_detected, right_detected) if l and not r)
        c = sum(1 for l, r in zip(left_detected, right_detected) if not l and r)

        discordant = b + c

        # McNemar statistic
        if discordant == 0:
            chi2 = 0.0
            p_value = 1.0
        else:
            chi2 = (b - c) ** 2 / discordant
            # Chi-squared with 1 df — approximate p-value
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi2, df=1)

        # Odds Ratio
        if c == 0:
            odds_ratio = float("inf") if b > 0 else 1.0
            or_ci = ConfidenceInterval(lower=0.0, upper=float("inf"))
        elif b == 0:
            odds_ratio = 0.0
            or_ci = ConfidenceInterval(lower=0.0, upper=float("inf"))
        else:
            odds_ratio = b / c
            log_or = math.log(odds_ratio)
            se = math.sqrt(1 / b + 1 / c)
            or_ci = ConfidenceInterval(
                lower=math.exp(log_or - 1.96 * se),
                upper=math.exp(log_or + 1.96 * se),
            )

        # Bonferroni-corrected significance
        corrected_p = min(p_value * self._n_tests, 1.0)
        is_significant = p_value < self._adjusted_alpha

        return BiasResult(
            model_id=model_id,
            technique=technique,
            discordant_pairs=discordant,
            mcnemar_statistic=chi2,
            raw_p_value=p_value,
            corrected_p_value=corrected_p,
            odds_ratio=odds_ratio,
            odds_ratio_ci=or_ci,
            is_significant=is_significant,
        )

    def evaluate_all(
        self,
        pair_results: dict[str, dict[str, tuple[list[bool], list[bool]]]],
        technique_names: list[str],
    ) -> list[BiasResult]:
        """Evaluate all model × technique combinations.

        Args:
            pair_results: Dict of model_id → technique → (left_detected, right_detected)
            technique_names: List of technique names.

        Returns:
            List of BiasResult for all combinations.
        """
        results = []
        for model_id, techniques in pair_results.items():
            for technique in technique_names:
                if technique in techniques:
                    left_det, right_det = techniques[technique]
                    result = self.evaluate(left_det, right_det, model_id, technique)
                    results.append(result)
                    if result.is_significant:
                        direction = "anti-left" if result.odds_ratio > 1 else "anti-right"
                        logger.warning(
                            f"SIGNIFICANT BIAS: {model_id}/{technique} — "
                            f"OR={result.odds_ratio:.2f}, p={result.raw_p_value:.4f} "
                            f"({direction})"
                        )
        return results
