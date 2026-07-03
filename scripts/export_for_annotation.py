import argparse
import csv
import random
from pathlib import Path

from loguru import logger

from src.schemas.speech import SpeechSegment
from src.utils.io import load_jsonl
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Export segments for annotation")
    parser.add_argument("--n", type=int, default=50, help="Number of segments to export")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ids", help="File with specific segment IDs (one per line)")
    args = parser.parse_args()

    setup_logging("INFO")

    # Load all segments
    segments_path = Path("data/segments/segments.jsonl")
    segments = load_jsonl(segments_path, SpeechSegment)
    logger.info(f"Loaded {len(segments)} segments")

    # Select subset
    if args.ids:
        ids_path = Path(args.ids)
        target_ids = set(ids_path.read_text().strip().split("\n"))
        selected = [s for s in segments if s.segment_id in target_ids]
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(segments, min(args.n, len(segments)))

    logger.info(f"Selected {len(selected)} segments for annotation")

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "segment_id",
            "text",
            "deputy_name",
            "party",
            "political_spectrum",
            "has_technique",
            "loaded_language",
            "name_calling",
            "doubt",
            "appeal_to_fear",
            "causal_oversimplification",
            "flag_waving",
            "span_loaded_language",
            "span_name_calling",
            "span_doubt",
            "span_appeal_to_fear",
            "span_causal_oversimplification",
            "span_flag_waving",
            "notes",
        ])

        for seg in selected:
            writer.writerow([
                seg.segment_id,
                seg.text,
                seg.deputy_name,
                seg.party,
                seg.political_spectrum.value,
                "",  # has_technique
                "",  # loaded_language
                "",  # name_calling
                "",  # doubt
                "",  # appeal_to_fear
                "",  # causal_oversimplification
                "",  # flag_waving
                "",  # span_loaded_language
                "",  # span_name_calling
                "",  # span_doubt
                "",  # span_appeal_to_fear
                "",  # span_causal_oversimplification
                "",  # span_flag_waving
                "",  # notes
            ])

    logger.info(f"Exported to {output_path}")
    logger.info(f"Import this CSV into Google Sheets and share with both annotators.")


if __name__ == "__main__":
    main()
