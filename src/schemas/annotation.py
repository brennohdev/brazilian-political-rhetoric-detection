from datetime import datetime

from pydantic import BaseModel, Field


class Annotation(BaseModel):
    """Human label compatible with Prediction schema."""

    segment_id: str
    annotator_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    technique: str
    span: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    is_calibration: bool = False
    round_id: str | None = None


class MinimalPair(BaseModel):
    """Left/right variant pair for H2 bias evaluation."""

    pair_id: str
    original_segment_id: str
    left_variant: str
    right_variant: str
    original_referent: str
    replacement_referent: str
    original_spectrum: str = Field(pattern=r"^(left|right)$")
    replacement_spectrum: str = Field(pattern=r"^(left|right)$")
    flagged_for_review: bool = False
