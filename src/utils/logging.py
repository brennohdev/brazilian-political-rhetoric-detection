import sys

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru with structured output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output (with rotation).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="30 days",
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        )
