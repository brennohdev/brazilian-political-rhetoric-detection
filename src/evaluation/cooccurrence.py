from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from loguru import logger


class CooccurrenceAnalyzer:
    """Analyze technique co-occurrence patterns and confusion matrices.

    Reveals:
    - Which techniques are frequently predicted together
    - Which techniques are confused with each other
    - Differences in co-occurrence patterns between models and humans
    """

    def compute_cooccurrence_matrix(
        self,
        labels: dict[str, set[str]],
        technique_names: list[str],
    ) -> np.ndarray:
        """Compute co-occurrence matrix from segment labels.

        Returns an NxN matrix where entry [i,j] = proportion of segments
        containing technique i that also contain technique j.
        """
        n = len(technique_names)
        matrix = np.zeros((n, n))
        tech_idx = {t: i for i, t in enumerate(technique_names)}

        for seg_id, techniques in labels.items():
            present = [t for t in techniques if t in tech_idx]
            for t in present:
                matrix[tech_idx[t], tech_idx[t]] += 1  # Self-count
            for t1, t2 in combinations(present, 2):
                matrix[tech_idx[t1], tech_idx[t2]] += 1
                matrix[tech_idx[t2], tech_idx[t1]] += 1

        # Normalize by row (P(j|i) = co-occur(i,j) / count(i))
        for i in range(n):
            if matrix[i, i] > 0:
                matrix[i, :] /= matrix[i, i]

        return matrix

    def compute_confusion_patterns(
        self,
        gold_labels: dict[str, set[str]],
        predictions: dict[str, set[str]],
        technique_names: list[str],
    ) -> dict[str, dict[str, int]]:
        """Find confusion patterns: when model predicts X, what did the gold say?

        Returns dict of predicted_technique → {gold_technique: count}
        showing what the model confuses.
        """
        confusion: dict[str, Counter] = {t: Counter() for t in technique_names}

        common_segments = set(gold_labels.keys()) & set(predictions.keys())

        for seg in common_segments:
            gold = gold_labels[seg]
            pred = predictions[seg]

            # False positives: model predicted technique X but gold doesn't have it
            for p_tech in pred:
                if p_tech not in gold and p_tech in confusion:
                    # What DID the gold have?
                    if gold:
                        for g_tech in gold:
                            confusion[p_tech][g_tech] += 1
                    else:
                        confusion[p_tech]["(none)"] += 1

        return {t: dict(c) for t, c in confusion.items()}

    def compare_model_cooccurrences(
        self,
        gold_labels: dict[str, set[str]],
        model_predictions: dict[str, dict[str, set[str]]],
        technique_names: list[str],
    ) -> dict[str, float]:
        """Compare co-occurrence patterns between gold and each model.

        Returns Frobenius distance between each model's co-occurrence matrix
        and the gold co-occurrence matrix. Lower = more similar to human patterns.
        """
        gold_matrix = self.compute_cooccurrence_matrix(gold_labels, technique_names)

        distances = {}
        for model_id, preds in model_predictions.items():
            pred_matrix = self.compute_cooccurrence_matrix(preds, technique_names)
            distance = float(np.linalg.norm(gold_matrix - pred_matrix, "fro"))
            distances[model_id] = round(distance, 4)

        return distances

    def top_confusions(
        self,
        gold_labels: dict[str, set[str]],
        predictions: dict[str, set[str]],
        technique_names: list[str],
        top_n: int = 10,
    ) -> list[dict]:
        """Get top N confusion patterns (most common false positive explanations).

        Returns list of dicts with predicted, confused_with, and count.
        """
        confusion = self.compute_confusion_patterns(
            gold_labels, predictions, technique_names
        )

        all_confusions = []
        for predicted, confused_with in confusion.items():
            for actual, count in confused_with.items():
                all_confusions.append({
                    "predicted": predicted,
                    "confused_with": actual,
                    "count": count,
                })

        all_confusions.sort(key=lambda x: -x["count"])
        return all_confusions[:top_n]

    def save_cooccurrence_chart(
        self,
        matrix: np.ndarray,
        technique_names: list[str],
        title: str,
        output_path: str,
    ) -> None:
        """Save co-occurrence matrix as a heatmap figure."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))
            # Use short names for display
            short_names = [t[:15] for t in technique_names]
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=short_names,
                yticklabels=short_names,
                ax=ax,
                vmin=0,
                vmax=1,
            )
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Co-occurrence chart saved: {output_path}")
        except ImportError:
            logger.warning("matplotlib/seaborn not available — skipping chart")
