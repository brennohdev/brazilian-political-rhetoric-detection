from abc import ABC, abstractmethod

from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment


class BaseClassifier(ABC):
    """Abstract base class for all classifiers.

    Any classifier that detects rhetorical manipulation techniques must
    implement this interface. The ClassificationRunner depends only on
    this abstraction, not concrete implementations.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model (used in Prediction.model_id)."""
        ...

    @abstractmethod
    def classify(self, segment: SpeechSegment) -> list[Prediction]:
        """Classify a single speech segment.

        Returns a list of detected techniques. Returns empty list if
        no techniques are detected (clean segment).

        Args:
            segment: A 3-5 sentence speech segment.

        Returns:
            List of Prediction objects for detected techniques.
        """
        ...

    def classify_batch(self, segments: list[SpeechSegment]) -> list[list[Prediction]]:
        """Classify a batch of segments.

        Default implementation: sequential fallback. Subclasses may
        override for parallelism or batched inference.

        Args:
            segments: List of speech segments.

        Returns:
            List of prediction lists, one per segment (same order).
        """
        return [self.classify(seg) for seg in segments]
