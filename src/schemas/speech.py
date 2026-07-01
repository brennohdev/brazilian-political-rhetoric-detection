from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class PoliticalSpectrum(StrEnum):
    """Political spectrum classification based on Zucco & Power (2024)."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class SessionType(StrEnum):
    """Parliamentary session types (argumentative sessions only)."""

    GRANDE_EXPEDIENTE = "Grande Expediente"
    ORDEM_DO_DIA = "Ordem do Dia"


class Speech(BaseModel):
    """Raw speech as retrieved from the Chamber API."""

    speech_id: str = Field(description="Unique identifier from the API")
    deputy_name: str
    party: str
    political_spectrum: PoliticalSpectrum
    date: date
    session_type: str
    legislative_phase: str
    transcription: str
    collected_at: datetime = Field(default_factory=datetime.now)

    @field_validator("transcription")
    @classmethod
    def transcription_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Transcription cannot be empty")
        return v


class ProcessedSpeech(BaseModel):
    """Speech after filtering and preprocessing."""

    speech_id: str
    deputy_name: str
    party: str
    political_spectrum: PoliticalSpectrum
    date: date
    session_type: SessionType
    clean_text: str
    sentence_count: int = Field(ge=3)
    original_speech_id: str


class SpeechSegment(BaseModel):
    """A 3-5 sentence passage with full traceability."""

    segment_id: str = Field(
        description="Format: {speech_id}_seg{N:03d}",
        pattern=r"^.+_seg\d{3}$",
    )
    speech_id: str
    deputy_name: str
    party: str
    political_spectrum: PoliticalSpectrum
    text: str
    sentence_count: int = Field(ge=3, le=5)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @field_validator("end_char")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if "start_char" in info.data and v <= info.data["start_char"]:
            raise ValueError("end_char must be greater than start_char")
        return v
