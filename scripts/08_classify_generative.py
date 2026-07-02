"""08_classify_generative.py — Run GPT-4o and LLaMA classifiers.

Classifies all segments using few-shot generative models.
No annotation needed — uses taxonomy definitions as in-context examples.

Input:  data/segments/segments.jsonl
Output: results/predictions/gpt4o.jsonl
        results/predictions/llama3.jsonl

Usage:
    uv run python scripts/08_classify_generative.py --model gpt4o
    uv run python scripts/08_classify_generative.py --model llama3
    uv run python scripts/08_classify_generative.py --model all
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()  # Load .env file (OPENAI_API_KEY)

from src.classification.prompt_builder import PromptBuilder
from src.classification.runner import ClassificationRunner
from src.config.loader import ConfigLoader
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logging


def run_gpt4o(segments: list[SpeechSegment], prompt_builder: PromptBuilder) -> None:
    """Run GPT-4o classification on all segments."""
    from src.classification.gpt4o import GPT4oClassifier

    loader = ConfigLoader()
    models = loader.load_models()
    gpt_config = next((m for m in models if "gpt" in m.name.lower()), models[0])

    classifier = GPT4oClassifier(config=gpt_config, prompt_builder=prompt_builder)
    runner = ClassificationRunner(classifier, failure_threshold=0.05)

    results = runner.run(segments)

    # Flatten predictions and save
    all_predictions = [p for preds in results for p in preds]
    output_path = Path("results/predictions/gpt4o.jsonl")
    save_jsonl(all_predictions, output_path)

    logger.info(f"GPT-4o: {len(all_predictions)} predictions saved to {output_path}")
    logger.info(f"  Failure rate: {runner.failure_rate:.1%}")
    logger.info(f"  Token usage: {classifier.token_usage}")


def run_llama(segments: list[SpeechSegment], prompt_builder: PromptBuilder) -> None:
    """Run LLaMA classification on all segments."""
    from src.classification.llama import LLaMAClassifier

    loader = ConfigLoader()
    models = loader.load_models()
    llama_config = next((m for m in models if "llama" in m.name.lower()), models[1])

    classifier = LLaMAClassifier(config=llama_config, prompt_builder=prompt_builder)
    runner = ClassificationRunner(classifier, failure_threshold=0.05)

    try:
        results = runner.run(segments)
    finally:
        classifier.close()

    # Flatten predictions and save
    all_predictions = [p for preds in results for p in preds]
    output_path = Path("results/predictions/llama3.jsonl")
    save_jsonl(all_predictions, output_path)

    logger.info(f"LLaMA: {len(all_predictions)} predictions saved to {output_path}")
    logger.info(f"  Failure rate: {runner.failure_rate:.1%}")


def main() -> None:
    """Main entry point for generative classification."""
    parser = argparse.ArgumentParser(description="Run generative classifiers")
    parser.add_argument(
        "--model",
        choices=["gpt4o", "llama3", "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of segments (for testing/pilot)",
    )
    args = parser.parse_args()

    setup_logging("INFO", log_file="logs/08_classify_generative.log")
    logger.info("=" * 60)
    logger.info("Starting generative classification")
    logger.info("=" * 60)

    # Load segments
    segments_path = Path("data/segments/segments.jsonl")
    segments = load_jsonl(segments_path, SpeechSegment)
    logger.info(f"Loaded {len(segments)} segments")

    if args.limit:
        segments = segments[: args.limit]
        logger.info(f"Limited to {len(segments)} segments (--limit)")

    # Build prompt
    taxonomy = TaxonomyRegistry()
    prompt_builder = PromptBuilder(taxonomy)
    logger.info(f"Taxonomy loaded: {taxonomy.technique_names}")

    # Run selected model(s)
    if args.model in ("gpt4o", "all"):
        logger.info("")
        logger.info("--- Running GPT-4o ---")
        run_gpt4o(segments, prompt_builder)

    if args.model in ("llama3", "all"):
        logger.info("")
        logger.info("--- Running LLaMA 3 ---")
        run_llama(segments, prompt_builder)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
