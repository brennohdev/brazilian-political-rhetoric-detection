from pydantic import BaseModel, Field, field_validator


class TechniqueDefinition(BaseModel):
    """One of the 6 rhetorical techniques with examples."""

    name: str
    definition: str
    positive_examples: list[str] = Field(min_length=2, max_length=5)
    negative_examples: list[str] = Field(min_length=1, max_length=3)
    frequency_reference: float = Field(
        ge=0.0, le=1.0, description="Expected frequency from SemEval literature"
    )


class Prediction(BaseModel):
    """Model output for a single detected technique in a segment."""

    technique: str
    span: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    model_id: str
    segment_id: str

    @field_validator("end_offset")
    @classmethod
    def offsets_valid(cls, v: int, info) -> int:
        if "start_offset" in info.data and v <= info.data["start_offset"]:
            raise ValueError("end_offset must be greater than start_offset")
        return v
