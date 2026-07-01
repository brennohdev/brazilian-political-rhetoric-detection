from pydantic import BaseModel, Field


class ConfidenceInterval(BaseModel):
    """A confidence interval with bounds and level."""

    lower: float
    upper: float
    level: float = 0.95


class AgreementResult(BaseModel):
    """H1 — Inter-annotator agreement metrics."""

    global_kappa: float = Field(ge=-1.0, le=1.0)
    per_technique_kappa: dict[str, float]
    observed_agreement: float = Field(ge=0.0, le=1.0)
    flagged_techniques: list[str] = Field(
        default_factory=list, description="Techniques with kappa < 0.40"
    )


class BiasResult(BaseModel):
    """H2 — Bias evaluation for one model-technique pair."""

    model_id: str
    technique: str
    discordant_pairs: int
    mcnemar_statistic: float
    raw_p_value: float
    corrected_p_value: float
    odds_ratio: float
    odds_ratio_ci: ConfidenceInterval
    is_significant: bool


class PerformanceResult(BaseModel):
    """H3 — Performance metrics for one model."""

    model_id: str
    per_technique: dict[str, dict[str, float]]  # technique -> {precision, recall, f1}
    macro_f1: float = Field(ge=0.0, le=1.0)
    micro_f1: float = Field(ge=0.0, le=1.0)
    macro_f1_ci: ConfidenceInterval
    bootstrap_iterations: int = 10_000


class EvaluationResult(BaseModel):
    """Aggregated evaluation output."""

    agreement: AgreementResult
    bias_results: list[BiasResult]
    performance_results: list[PerformanceResult]
    seed: int
    config_hash: str
