from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from src.exceptions import ConfigValidationError
from src.schemas.config import (
    CollectionConfig,
    EvaluationConfig,
    ModelConfig,
    SamplingConfig,
)

T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Load and validate YAML configs against Pydantic schemas at startup."""

    def __init__(self, config_dir: Path = Path("configs")) -> None:
        self._config_dir = config_dir

    def load(self, filename: str, schema: type[T]) -> T:
        """Load a YAML file and validate against a Pydantic schema.

        Raises ConfigValidationError if file not found or validation fails.
        """
        path = self._config_dir / filename
        if not path.exists():
            raise ConfigValidationError(
                f"Config file not found: {path}",
                errors=[f"File does not exist: {path}"],
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Invalid YAML in {filename}: {e}",
                errors=[str(e)],
            ) from e
        try:
            return schema.model_validate(raw)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise ConfigValidationError(
                f"Validation failed for {filename}",
                errors=errors,
            ) from e

    def load_collection(self) -> CollectionConfig:
        """Load collection configuration."""
        return self.load("collection.yaml", CollectionConfig)

    def load_sampling(self) -> SamplingConfig:
        """Load sampling configuration."""
        return self.load("sampling.yaml", SamplingConfig)

    def load_models(self) -> list[ModelConfig]:
        """Load model configurations (list of models from YAML)."""
        path = self._config_dir / "models.yaml"
        if not path.exists():
            raise ConfigValidationError(
                f"Config file not found: {path}",
                errors=[f"File does not exist: {path}"],
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Invalid YAML in models.yaml: {e}",
                errors=[str(e)],
            ) from e
        if not isinstance(raw, dict) or "models" not in raw:
            raise ConfigValidationError(
                "models.yaml must contain a 'models' key",
                errors=["Missing 'models' key in models.yaml"],
            )
        models = []
        errors = []
        for i, model_data in enumerate(raw["models"]):
            try:
                models.append(ModelConfig.model_validate(model_data))
            except ValidationError as e:
                for err in e.errors():
                    errors.append(f"models[{i}].{err['loc']}: {err['msg']}")
        if errors:
            raise ConfigValidationError(
                f"Validation failed for models.yaml with {len(errors)} errors",
                errors=errors,
            )
        return models

    def load_evaluation(self) -> EvaluationConfig:
        """Load evaluation configuration."""
        return self.load("evaluation.yaml", EvaluationConfig)

    def validate_all(self) -> None:
        """Validate all config files at startup. Collect all errors and report together.

        Raises ConfigValidationError with aggregated errors if any file fails.
        """
        all_errors: list[str] = []
        configs_to_check: list[tuple[str, type[BaseModel]]] = [
            ("collection.yaml", CollectionConfig),
            ("evaluation.yaml", EvaluationConfig),
        ]
        for filename, schema in configs_to_check:
            try:
                self.load(filename, schema)
            except ConfigValidationError as e:
                all_errors.extend(
                    [f"{filename}: {err}" for err in e.validation_errors]
                )
        # Check models.yaml separately (different structure)
        try:
            self.load_models()
        except ConfigValidationError as e:
            all_errors.extend(
                [f"models.yaml: {err}" for err in e.validation_errors]
            )
        if all_errors:
            raise ConfigValidationError(
                f"Configuration validation failed with {len(all_errors)} error(s)",
                errors=all_errors,
            )
