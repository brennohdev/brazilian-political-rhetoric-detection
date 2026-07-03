import json
from pathlib import Path

from loguru import logger

from src.exceptions import AnnotationError
from src.schemas.annotation import Annotation


class AnnotationLoader:
    """Load and align annotations from two independent annotators."""

    def load_annotations(self, path: Path) -> list[Annotation]:
        """Load annotations from a JSONL file."""
        if not path.exists():
            raise AnnotationError(
                f"Annotation file not found: {path}",
                context={"path": str(path)},
            )
        annotations = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    annotations.append(Annotation.model_validate_json(stripped))
                except Exception as e:
                    logger.warning(f"Line {line_num} in {path.name}: {e}")
        return annotations

    def align_by_segment(
        self,
        annotations_a: list[Annotation],
        annotations_b: list[Annotation],
    ) -> dict[str, tuple[set[str], set[str]]]:
        """Align annotations by segment_id for agreement computation.

        Returns dict of segment_id → (techniques_a, techniques_b).
        Only includes segments annotated by both.
        """
        # Group by segment
        by_seg_a: dict[str, set[str]] = {}
        for ann in annotations_a:
            by_seg_a.setdefault(ann.segment_id, set()).add(ann.technique)

        by_seg_b: dict[str, set[str]] = {}
        for ann in annotations_b:
            by_seg_b.setdefault(ann.segment_id, set()).add(ann.technique)

        # Find common segments
        common = set(by_seg_a.keys()) & set(by_seg_b.keys())
        logger.info(
            f"Aligned {len(common)} segments "
            f"(A has {len(by_seg_a)}, B has {len(by_seg_b)})"
        )

        return {seg: (by_seg_a[seg], by_seg_b[seg]) for seg in sorted(common)}

    def validate_exclusion(
        self,
        calibration_ids: set[str],
        evaluation_ids: set[str],
    ) -> None:
        """Ensure calibration segments are excluded from evaluation set."""
        overlap = calibration_ids & evaluation_ids
        if overlap:
            raise AnnotationError(
                f"Calibration/evaluation overlap: {len(overlap)} segments "
                f"appear in both sets. These must be excluded from evaluation.",
                context={"overlapping_ids": list(overlap)[:10]},
            )
