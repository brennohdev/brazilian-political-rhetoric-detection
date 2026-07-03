import json
from pathlib import Path

from loguru import logger

from src.exceptions import ClassificationError
from src.classification.base import BaseClassifier
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment


class ClassificationRunner:
    """Orchestrates batch classification with failure tracking and checkpointing.

    Responsibilities:
    - Run a classifier on a batch of segments
    - Track failed segments (parse errors, API errors)
    - Halt if failure rate exceeds threshold (data quality check)
    - Save checkpoint every N segments for resumability
    - Report progress
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        failure_threshold: float = 0.05,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 25,
    ) -> None:
        self._classifier = classifier
        self._failure_threshold = failure_threshold
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every = checkpoint_every
        self._results: list[list[Prediction]] = []
        self._failures: list[str] = []

    def run(self, segments: list[SpeechSegment]) -> list[list[Prediction]]:
        """Classify all segments with progress, failure tracking, and checkpointing.

        Resumes from checkpoint if one exists. Halts if failure rate exceeds threshold.
        """
        self._results = []
        self._failures = []

        # Resume from checkpoint if available
        start_idx = 0
        if self._checkpoint_dir:
            start_idx, resumed_results, resumed_failures = self._load_checkpoint()
            if start_idx > 0:
                self._results = resumed_results
                self._failures = resumed_failures
                logger.info(
                    f"Resumed from checkpoint at segment {start_idx}/{len(segments)} "
                    f"({len(resumed_failures)} previous failures)"
                )

        total = len(segments)
        logger.info(
            f"Starting classification with {self._classifier.model_id} "
            f"on {total} segments (from idx {start_idx}, "
            f"failure threshold: {self._failure_threshold:.1%})"
        )

        for i in range(start_idx, total):
            segment = segments[i]
            try:
                import time
                t0 = time.time()
                predictions = self._classifier.classify(segment)
                elapsed = time.time() - t0
                self._results.append(predictions)
                n_techs = len(predictions)
                logger.info(
                    f"  [{i+1}/{total}] ✓ {n_techs} technique{'s' if n_techs != 1 else ''} "
                    f"detected ({elapsed:.1f}s) — {segment.segment_id[-30:]}"
                )
            except Exception as e:
                self._failures.append(segment.segment_id)
                self._results.append([])
                logger.warning(f"  [{i+1}/{total}] ✗ FAILED: {e}")

            processed = i + 1
            if processed % self._checkpoint_every == 0:
                self._check_threshold(processed, total)
                self._save_checkpoint(processed)

        # Final check and cleanup
        self._check_threshold(total, total)
        self._clear_checkpoint()

        logger.info(
            f"Classification complete: {total} segments, "
            f"{len(self._failures)} failures ({self.failure_rate:.1%})"
        )
        return self._results

    def _check_threshold(self, processed: int, total: int) -> None:
        """Check if failure rate exceeds threshold."""
        if processed < 10:
            return
        rate = len(self._failures) / processed
        if rate > self._failure_threshold:
            self._save_checkpoint(processed)
            raise ClassificationError(
                f"Failure rate ({rate:.1%}) exceeds threshold "
                f"({self._failure_threshold:.1%}) after {processed}/{total} segments.",
                context={
                    "model_id": self._classifier.model_id,
                    "processed": processed,
                    "failures": len(self._failures),
                    "failure_rate": rate,
                },
            )

    def _save_checkpoint(self, processed: int) -> None:
        """Save progress to a checkpoint file."""
        if not self._checkpoint_dir:
            return
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"checkpoint_{self._classifier.model_id}.json"

        data = {
            "model_id": self._classifier.model_id,
            "processed": processed,
            "failures": self._failures,
            "predictions": [[p.model_dump() for p in preds] for preds in self._results],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def _load_checkpoint(self) -> tuple[int, list[list[Prediction]], list[str]]:
        """Load checkpoint. Returns (start_idx, results, failures)."""
        if not self._checkpoint_dir:
            return 0, [], []
        path = self._checkpoint_dir / f"checkpoint_{self._classifier.model_id}.json"
        if not path.exists():
            return 0, [], []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model_id") != self._classifier.model_id:
                return 0, [], []
            results = [[Prediction.model_validate(p) for p in preds] for preds in data["predictions"]]
            return data["processed"], results, data["failures"]
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return 0, [], []

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint after successful completion."""
        if not self._checkpoint_dir:
            return
        path = self._checkpoint_dir / f"checkpoint_{self._classifier.model_id}.json"
        if path.exists():
            path.unlink()
            logger.info("Checkpoint cleared (run completed)")

    @property
    def failure_rate(self) -> float:
        total = len(self._results)
        return len(self._failures) / max(total, 1)

    @property
    def failed_segments(self) -> list[str]:
        return self._failures

    @property
    def results(self) -> list[list[Prediction]]:
        return self._results
