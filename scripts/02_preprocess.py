import json
from pathlib import Path

from loguru import logger

from src.preprocessing.filters import (
    Deduplicator,
    FormalityRemover,
    LegislativePhaseFilter,
    MinimumLengthFilter,
    MonologueIsolator,
    SpeechFilter,
    TextNormalizer,
)
from src.schemas.speech import Speech
from src.utils.io import save_json, save_jsonl
from src.utils.logging import setup_logging


def load_raw_speeches(raw_dir: Path) -> list[Speech]:
    """Load all raw speech JSON files into validated Speech objects.

    Skips files that fail validation (logs warning) and the checkpoint file.
    """
    speeches: list[Speech] = []
    errors = 0

    json_files = sorted(raw_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {raw_dir}")

    for filepath in json_files:
        if filepath.name.startswith("_"):
            continue  # Skip checkpoint and other internal files

        try:
            content = filepath.read_text(encoding="utf-8")
            speech = Speech.model_validate_json(content)
            speeches.append(speech)
        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"Failed to parse {filepath.name}: {e}")

    if errors > 0:
        logger.warning(f"Total parse errors: {errors} (skipped)")

    return speeches


def main() -> None:
    """Main entry point for preprocessing."""
    setup_logging("INFO", log_file="logs/02_preprocess.log")
    logger.info("=" * 60)
    logger.info("Starting speech preprocessing pipeline")
    logger.info("=" * 60)

    # Load raw speeches
    raw_dir = Path("data/raw")
    speeches = load_raw_speeches(raw_dir)
    logger.info(f"Loaded {len(speeches)} valid speeches")

    # Configure filter pipeline
    # Order matters: phase filter first (cheapest, removes most),
    # then text cleaning, then expensive dedup last.
    filter_steps = [
        LegislativePhaseFilter(),
        MonologueIsolator(),
        FormalityRemover(),
        TextNormalizer(),
        Deduplicator(threshold=0.75),
        MinimumLengthFilter(),
    ]

    logger.info(f"Filter pipeline: {[s.name for s in filter_steps]}")
    logger.info("")

    # Run preprocessing
    speech_filter = SpeechFilter(steps=filter_steps)
    processed = speech_filter.filter_all(speeches)

    logger.info("")
    logger.info(speech_filter.report.summary())

    # Persist results
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed speeches as JSONL
    output_path = output_dir / "speeches.jsonl"
    save_jsonl(processed, output_path)
    logger.info(f"Saved {len(processed)} processed speeches to {output_path}")

    # Save filter report
    report_path = output_dir / "filter_report.json"
    save_json(speech_filter.report, report_path)
    logger.info(f"Saved filter report to {report_path}")

    # Log final statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Input:  {len(speeches)} raw speeches")
    logger.info(f"  Output: {len(processed)} processed speeches")
    logger.info(f"  Removed: {len(speeches) - len(processed)} ({(len(speeches) - len(processed)) / len(speeches) * 100:.1f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
