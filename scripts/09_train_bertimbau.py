from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.classification.bertimbau import BERTimbauClassifier
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry
from src.utils.io import load_jsonl
from src.utils.logging import setup_logging
from src.utils.seeds import set_global_seed

load_dotenv()


def load_annotated_data(
    annotations_dir: Path, segments_path: Path, technique_names: list[str]
) -> tuple[list[str], list[list[int]]]:
    """Load annotated segments and convert to training format.

    Returns (texts, label_vectors) where each label vector is binary
    (1 if technique present, 0 otherwise).
    """
    # Load segments for text
    segments = load_jsonl(segments_path, SpeechSegment)
    seg_map = {s.segment_id: s.text for s in segments}

    # Load consolidated annotations
    import json

    annotations_path = annotations_dir / "annotations.jsonl"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found at {annotations_path}. "
            "Run annotation process first."
        )

    texts = []
    labels = []
    seen_segments = set()

    with open(annotations_path) as f:
        for line in f:
            ann = json.loads(line)
            seg_id = ann["segment_id"]
            if seg_id in seen_segments:
                continue
            seen_segments.add(seg_id)

            text = seg_map.get(seg_id, "")
            if not text:
                continue

            # Build label vector
            techniques_present = set(ann.get("techniques", []))
            label_vec = [
                1 if t in techniques_present else 0 for t in technique_names
            ]
            texts.append(text)
            labels.append(label_vec)

    return texts, labels


def train_test_split(
    texts: list[str], labels: list[list[int]], seed: int = 42
) -> tuple:
    """Split into train/dev/test (70/15/15)."""
    import random

    n = len(texts)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(n * 0.70)
    dev_end = int(n * 0.85)

    train_idx = indices[:train_end]
    dev_idx = indices[train_end:dev_end]
    test_idx = indices[dev_end:]

    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i] for i in dev_idx],
        [labels[i] for i in dev_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in test_idx],
    )


def main() -> None:
    """Main entry point for BERTimbau training."""
    setup_logging("INFO", log_file="logs/09_train_bertimbau.log")
    logger.info("=" * 60)
    logger.info("Starting BERTimbau fine-tuning")
    logger.info("=" * 60)

    set_global_seed(42)

    taxonomy = TaxonomyRegistry()
    technique_names = taxonomy.technique_names
    logger.info(f"Techniques: {technique_names}")

    # Load annotated data
    annotations_dir = Path("data/annotations/consolidated")
    segments_path = Path("data/segments/segments.jsonl")

    texts, labels = load_annotated_data(annotations_dir, segments_path, technique_names)
    logger.info(f"Loaded {len(texts)} annotated segments")

    # Split
    train_t, train_l, dev_t, dev_l, test_t, test_l = train_test_split(texts, labels)
    logger.info(f"Split: train={len(train_t)}, dev={len(dev_t)}, test={len(test_t)}")

    # Initialize and train
    classifier = BERTimbauClassifier(taxonomy=taxonomy)
    history = classifier.train_model(
        train_texts=train_t,
        train_labels=train_l,
        dev_texts=dev_t,
        dev_labels=dev_l,
        epochs=10,
        lr=2e-5,
        batch_size=16,
    )

    # Tune threshold
    threshold = classifier.tune_threshold(dev_t, dev_l)
    logger.info(f"Final threshold: {threshold:.2f}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best dev F1: {max(history['dev_f1']):.4f}")
    logger.info(f"  Checkpoint: models/bertimbau/best_model.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
