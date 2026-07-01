from pydantic import BaseModel, Field


class CollectionConfig(BaseModel):
    """Configuration for speech data collection."""

    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    session_types: list[str] = ["Grande Expediente", "Ordem do Dia"]
    rate_limit_delay: float = Field(ge=0.1, default=1.0)
    max_retries: int = Field(ge=1, default=3)
    output_dir: str = "data/raw"


class SamplingConfig(BaseModel):
    """Configuration for stratified sampling."""

    target_per_stratum: int = Field(ge=10, default=100)
    seed: int = 42
    temporal_periods: int = Field(ge=1, default=4)
    segment_sentences: tuple[int, int] = (3, 5)


class ModelConfig(BaseModel):
    """Configuration for a single classification model."""

    name: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_tokens: int = Field(ge=100, default=2048)
    prompt_template: str
    failure_threshold: float = Field(ge=0.0, le=1.0, default=0.05)
    quantization: str | None = None  # e.g., "Q4" for LLaMA


class EvaluationConfig(BaseModel):
    """Configuration for statistical evaluation."""

    seed: int = 42
    bootstrap_iterations: int = Field(ge=1000, default=10_000)
    significance_level: float = Field(ge=0.001, le=0.1, default=0.05)
    bonferroni_tests: int = Field(ge=1, default=18)  # 3 models × 6 techniques
