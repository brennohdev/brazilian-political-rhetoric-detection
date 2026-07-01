class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class CollectionError(PipelineError):
    """Errors during speech collection from the API."""

    pass


class ClassificationError(PipelineError):
    """Errors during model classification."""

    pass


class ConfigValidationError(PipelineError):
    """Errors in configuration file validation."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.validation_errors = errors or []


class AnnotationError(PipelineError):
    """Errors in annotation loading or validation."""

    pass


class TaxonomyError(PipelineError):
    """Errors in taxonomy loading or lookup."""

    pass
