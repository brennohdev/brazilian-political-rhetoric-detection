from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.classification.bertimbau import BERTimbauClassifier
from src.classification.runner import ClassificationRunner
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logging

load_dotenv()


def main() -> None:
    """Main entry point for BERTimbau classification."""
    setup_logging("INFO", log_file="logs/10_classify_bertimbau.log")
    logger.info("=" * 60)
    logger.info("Starting BERTimbau classification")
    logger.info("=" * 60)

    # Load segments
    segments_path = Path("data/segments/segments.jsonl")
    segments = load_jsonl(segments_path, SpeechSegment)
    logger.info(f"Loaded {len(segments)} segments")

    # Load classifier with trained checkpoint
    taxonomy = TaxonomyRegistry()
    checkpoint_path = Path("models/bertimbau/best_model.pt")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run scripts/09_train_bertimbau.py first.")
        return

    classifier = BERTimbauClassifier(
        taxonomy=taxonomy, checkpoint_path=checkpoint_path
    )

    # Run classification
    runner = ClassificationRunner(
        classifier,
        failure_threshold=0.05,
        checkpoint_dir=Path("results/predictions/.checkpoints"),
    )
    results = runner.run(segments)

    # Flatten and save
    all_predictions = [p for preds in results for p in preds]
    output_path = Path("results/predictions/bertimbau.jsonl")
    save_jsonl(all_predictions, output_path)

    logger.info(f"BERTimbau: {len(all_predictions)} predictions saved to {output_path}")
    logger.info(f"  Failure rate: {runner.failure_rate:.1%}")

    logger.info("=" * 60)
    logger.info("BERTIMBAU CLASSIFICATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
