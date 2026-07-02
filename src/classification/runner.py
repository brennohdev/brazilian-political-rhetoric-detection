from loguru import logger

from src.exceptions import ClassificationError
from src.classification.base import BaseClassifier
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment


class ClassificationRunner:
    """Orchestrates batch classification with failure tracking.

    Responsibilities:
    - Run a classifier on a batch of segments
    - Track failed segments (parse errors, API errors)
    - Halt if failure rate exceeds threshold (data quality check)
    - Report progress
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        failure_threshold: float = 0.05,
    ) -> None:
        self._classifier = classifier
        self._failure_threshold = failure_threshold
        self._results: list[list[Prediction]] = []
        self._failures: list[str] = []  # segment_ids that failed

    def run(self, segments: list[SpeechSegment]) -> list[list[Prediction]]:
        """Classify all segments with progress logging and failure tracking.

        Halts early if failure rate exceeds threshold.

        Args:
            segments: List of segments to classify.

        Returns:
            List of prediction lists (one per segment, in order).
            Failed segments get an empty prediction list.

        Raises:
            ClassificationError: If failure rate exceeds threshold.
        """
        self._results = []
        self._failures = []
        total = len(segments)

        logger.info(
            f"Starting classification with {self._classifier.model_id} "
            f"on {total} segments (failure threshold: {self._failure_threshold:.1%})"
        )

        for i, segment in enumerate(segments):
            try:
                predictions = self._classifier.classify(segment)
                self._results.append(predictions)
            except Exception as e:
                self._failures.append(segment.segment_id)
                self._results.append([])  # Empty predictions for failed segments
                logger.warning(
                    f"Failed on segment {segment.segment_id}: {e}"
                )

            # Check failure threshold periodically (every 50 segments)
            processed = i + 1
            if processed % 50 == 0:
                self._check_threshold(processed, total)
                logger.info(
                    f"  Progress: {processed}/{total} "
                    f"({processed/total*100:.1f}%) — "
                    f"failures: {len(self._failures)}"
                )

        # Final threshold check
        self._check_threshold(total, total)

        logger.info(
            f"Classification complete: {total} segments, "
            f"{len(self._failures)} failures ({self.failure_rate:.1%})"
        )
        return self._results

    def _check_threshold(self, processed: int, total: int) -> None:
        """Check if failure rate exceeds threshold. Raise if so."""
        if processed < 10:
            return  # Don't check with too few samples

        rate = len(self._failures) / processed
        if rate > self._failure_threshold:
            raise ClassificationError(
                f"Failure rate ({rate:.1%}) exceeds threshold "
                f"({self._failure_threshold:.1%}) after {processed}/{total} segments. "
                f"Halting classification.",
                context={
                    "model_id": self._classifier.model_id,
                    "processed": processed,
                    "failures": len(self._failures),
                    "failure_rate": rate,
                    "threshold": self._failure_threshold,
                    "failed_segments": self._failures[-10:],
                },
            )

    @property
    def failure_rate(self) -> float:
        """Current failure rate (failures / total processed)."""
        total = len(self._results)
        return len(self._failures) / max(total, 1)

    @property
    def failed_segments(self) -> list[str]:
        """List of segment IDs that failed classification."""
        return self._failures

    @property
    def results(self) -> list[list[Prediction]]:
        """Access results after running."""
        return self._results
