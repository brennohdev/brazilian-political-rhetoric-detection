from pathlib import Path

from loguru import logger

from src.config.loader import ConfigLoader
from src.sampling.segmenter import SpeechSegmenter
from src.schemas.speech import ProcessedSpeech
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logging


def main() -> None:
    """Main entry point for segmentation."""
    setup_logging("INFO", log_file="logs/05_segment.log")
    logger.info("=" * 60)
    logger.info("Starting speech segmentation")
    logger.info("=" * 60)

    # Load config for segment size
    loader = ConfigLoader()
    config = loader.load_sampling()
    min_sent, max_sent = config.segment_sentences
    logger.info(f"Segment size: {min_sent}-{max_sent} sentences")

    # Load sampled speeches
    input_path = Path("data/samples/sample.jsonl")
    speeches = load_jsonl(input_path, ProcessedSpeech)
    logger.info(f"Loaded {len(speeches)} sampled speeches")

    # Segment
    segmenter = SpeechSegmenter(min_sentences=min_sent, max_sentences=max_sent)
    segments = segmenter.segment_batch(speeches)

    # Persist
    output_path = Path("data/segments/segments.jsonl")
    save_jsonl(segments, output_path)
    logger.info(f"Saved {len(segments)} segments to {output_path}")

    # Report
    logger.info("")
    logger.info("=" * 60)
    logger.info("SEGMENTATION COMPLETE")
    logger.info(f"  Input speeches: {len(speeches)}")
    logger.info(f"  Output segments: {len(segments)}")
    logger.info(f"  Avg segments/speech: {len(segments)/len(speeches):.1f}")
    logger.info(f"  Sentence range: {min(s.sentence_count for s in segments)}-{max(s.sentence_count for s in segments)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
