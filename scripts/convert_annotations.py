import argparse
import csv
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.schemas.annotation import Annotation
from src.utils.io import save_jsonl
from src.utils.logging import setup_logging

TECHNIQUE_COLUMNS = {
    "loaded_language": "Loaded Language",
    "name_calling": "Name Calling",
    "doubt": "Doubt",
    "appeal_to_fear": "Appeal to Fear",
    "causal_oversimplification": "Causal Oversimplification",
    "flag_waving": "Flag-Waving",
}

SPAN_COLUMNS = {
    "span_loaded_language": "Loaded Language",
    "span_name_calling": "Name Calling",
    "span_doubt": "Doubt",
    "span_appeal_to_fear": "Appeal to Fear",
    "span_causal_oversimplification": "Causal Oversimplification",
    "span_flag_waving": "Flag-Waving",
}


def convert_csv_to_annotations(
    csv_path: Path, annotator_id: str
) -> list[Annotation]:
    """Convert a CSV export to a list of Annotation objects."""
    annotations = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            segment_id = row.get("segment_id", "").strip()
            if not segment_id:
                continue

            # Check each technique column
            for col_name, technique_name in TECHNIQUE_COLUMNS.items():
                value = row.get(col_name, "0").strip()
                if value == "1":
                    # Get corresponding span
                    span_col = f"span_{col_name.replace('_', '_')}"
                    # Map back to span column name
                    span_key = next(
                        (k for k, v in SPAN_COLUMNS.items() if v == technique_name),
                        None,
                    )
                    span = row.get(span_key, "").strip() if span_key else ""

                    annotations.append(
                        Annotation(
                            segment_id=segment_id,
                            annotator_id=annotator_id,
                            timestamp=datetime.now(),
                            technique=technique_name,
                            span=span if span else None,
                            start_offset=None,
                            end_offset=None,
                            is_calibration=False,
                        )
                    )

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Convert annotation CSV to JSONL")
    parser.add_argument("--annotator", required=True, help="Annotator ID (brenno/gustavo)")
    parser.add_argument("--input", required=True, help="Path to CSV export")
    parser.add_argument("--calibration", action="store_true", help="Mark as calibration round")
    parser.add_argument(
        "--output-dir",
        default="data/annotations/main",
        help="Output directory",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return

    annotations = convert_csv_to_annotations(csv_path, args.annotator)

    if args.calibration:
        for ann in annotations:
            ann.is_calibration = True
        output_dir = Path("data/annotations/calibration")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Map annotator name to file
    file_map = {"brenno": "annotator_a.jsonl", "gustavo": "annotator_b.jsonl"}
    filename = file_map.get(args.annotator, f"annotator_{args.annotator}.jsonl")
    output_path = output_dir / filename

    save_jsonl(annotations, output_path)
    logger.info(f"Converted {len(annotations)} annotations to {output_path}")
    logger.info(f"  Segments with at least one technique: {len(set(a.segment_id for a in annotations))}")


if __name__ == "__main__":
    main()
