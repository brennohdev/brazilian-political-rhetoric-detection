import numpy as np
from loguru import logger

from src.schemas.evaluation import AgreementResult


class AgreementEvaluator:
    """Compute inter-annotator agreement (Cohen's kappa).

    Evaluates H1: Do annotators achieve sufficient agreement (κ ≥ 0.40)
    on each technique?
    """

    KAPPA_THRESHOLD = 0.40

    def evaluate(
        self,
        labels_a: dict[str, set[str]],
        labels_b: dict[str, set[str]],
        technique_names: list[str],
    ) -> AgreementResult:
        """Compute global and per-technique Cohen's kappa.

        Args:
            labels_a: Annotator A's labels. Dict of segment_id → set of techniques.
            labels_b: Annotator B's labels. Dict of segment_id → set of techniques.
            technique_names: List of technique names.

        Returns:
            AgreementResult with global and per-technique kappa.
        """
        # Align by segment (only segments both annotators labeled)
        common_segments = sorted(set(labels_a.keys()) & set(labels_b.keys()))
        if not common_segments:
            logger.warning("No overlapping segments between annotators")
            return AgreementResult(
                global_kappa=0.0,
                per_technique_kappa={t: 0.0 for t in technique_names},
                observed_agreement=0.0,
                flagged_techniques=technique_names,
            )

        # Compute per-technique kappa
        per_technique_kappa: dict[str, float] = {}
        flagged: list[str] = []

        for technique in technique_names:
            # Binary vectors: 1 if annotator labeled this technique, 0 otherwise
            vec_a = [1 if technique in labels_a[seg] else 0 for seg in common_segments]
            vec_b = [1 if technique in labels_b[seg] else 0 for seg in common_segments]

            kappa = self._compute_kappa(vec_a, vec_b)
            per_technique_kappa[technique] = kappa

            if kappa < self.KAPPA_THRESHOLD:
                flagged.append(technique)

        # Global kappa (multi-label: flatten all technique decisions)
        all_a = []
        all_b = []
        for seg in common_segments:
            for technique in technique_names:
                all_a.append(1 if technique in labels_a[seg] else 0)
                all_b.append(1 if technique in labels_b[seg] else 0)

        global_kappa = self._compute_kappa(all_a, all_b)

        # Observed agreement
        n = len(all_a)
        observed = sum(1 for a, b in zip(all_a, all_b) if a == b) / n

        return AgreementResult(
            global_kappa=global_kappa,
            per_technique_kappa=per_technique_kappa,
            observed_agreement=observed,
            flagged_techniques=flagged,
        )

    @staticmethod
    def _compute_kappa(labels_a: list[int], labels_b: list[int]) -> float:
        """Cohen's kappa for two binary label sequences."""
        n = len(labels_a)
        if n == 0:
            return 0.0

        # Observed agreement
        agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
        p_o = agree / n

        # Expected agreement by chance
        a_pos = sum(labels_a) / n
        b_pos = sum(labels_b) / n
        p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

        if p_e == 1.0:
            return 1.0  # Perfect agreement by chance = perfect agreement

        kappa = (p_o - p_e) / (1 - p_e)
        return round(kappa, 4)
