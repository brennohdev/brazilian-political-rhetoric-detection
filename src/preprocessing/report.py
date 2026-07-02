from pydantic import BaseModel, Field


class FilterStepStats(BaseModel):
    """Statistics for a single filter step."""

    step_name: str
    input_count: int
    output_count: int
    removed_count: int
    removal_rate: float = Field(ge=0.0, le=1.0)


class FilterReport(BaseModel):
    """Complete filtering report documenting removals at each step.

    This report is persisted alongside the processed data for full traceability
    and feeds directly into the paper's methodology section.
    """

    steps: list[FilterStepStats] = []
    total_input: int = 0
    total_output: int = 0

    def add_step(self, step_name: str, input_count: int, output_count: int) -> None:
        """Record the result of a filter step."""
        removed = input_count - output_count
        rate = removed / input_count if input_count > 0 else 0.0
        self.steps.append(
            FilterStepStats(
                step_name=step_name,
                input_count=input_count,
                output_count=output_count,
                removed_count=removed,
                removal_rate=round(rate, 4),
            )
        )

    def summary(self) -> str:
        """Human-readable summary for logging."""
        lines = ["=" * 60, "PREPROCESSING REPORT", "=" * 60]
        lines.append(f"{'Step':<25} {'Input':>7} {'Output':>7} {'Removed':>8} {'Rate':>7}")
        lines.append("-" * 60)
        for step in self.steps:
            lines.append(
                f"{step.step_name:<25} {step.input_count:>7} "
                f"{step.output_count:>7} {step.removed_count:>8} "
                f"{step.removal_rate:>6.1%}"
            )
        lines.append("-" * 60)
        lines.append(
            f"{'TOTAL':<25} {self.total_input:>7} "
            f"{self.total_output:>7} {self.total_input - self.total_output:>8} "
            f"{(self.total_input - self.total_output) / max(self.total_input, 1):>6.1%}"
        )
        lines.append("=" * 60)
        return "\n".join(lines)
