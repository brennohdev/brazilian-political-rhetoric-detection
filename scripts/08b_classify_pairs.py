import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from src.classification.prompt_builder import PromptBuilder
from src.schemas.annotation import MinimalPair
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry
from src.utils.io import load_jsonl
from src.utils.logging import setup_logging

load_dotenv()


class PairPrediction(BaseModel):
    """Prediction for a minimal pair variant."""

    pair_id: str
    variant: str  # "left" or "right"
    technique: str
    span: str
    confidence: float
    model_id: str
    original_segment_id: str


def create_virtual_segment(pair: MinimalPair, variant: str) -> SpeechSegment:
    """Create a virtual SpeechSegment from a pair variant for classification."""
    text = pair.left_variant if variant == "left" else pair.right_variant
    spectrum = "left" if variant == "left" else "right"
    # segment_id must match pattern ^.+_seg\d{3}$
    variant_code = "001" if variant == "left" else "002"

    return SpeechSegment(
        segment_id=f"{pair.pair_id}_{variant}_seg{variant_code}",
        speech_id=pair.pair_id,
        deputy_name="MINIMAL_PAIR",
        party="PAIR",
        political_spectrum=spectrum,
        text=text,
        sentence_count=min(5, max(3, text.count(".") + text.count("!") + text.count("?"))),
        start_char=0,
        end_char=len(text),
    )


def classify_pairs_with_model(
    pairs: list[MinimalPair],
    model_name: str,
    prompt_builder: PromptBuilder,
) -> list[PairPrediction]:
    """Classify both variants of all pairs with a given model."""
    from src.config.loader import ConfigLoader

    loader = ConfigLoader()
    models = loader.load_models()

    if model_name == "gpt4o-mini":
        from src.classification.gpt4o import GPT4oClassifier

        config = next((m for m in models if "gpt" in m.name.lower()), models[0])
        classifier = GPT4oClassifier(
            config=config, prompt_builder=prompt_builder, model_name="gpt-4o-mini"
        )
    elif model_name == "llama3":
        from src.classification.llama import LLaMAClassifier

        config = next((m for m in models if "llama" in m.name.lower()), models[1])
        classifier = LLaMAClassifier(config=config, prompt_builder=prompt_builder)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Resume from checkpoint if available
    model_id = "gpt4o_mini" if model_name == "gpt4o-mini" else "llama3"
    all_predictions, start_pair_idx = _load_checkpoint(model_id)
    if start_pair_idx > 0:
        logger.info(f"Resuming from checkpoint: pair {start_pair_idx}/{len(pairs)} ({len(all_predictions)} predictions)")

    total_tasks = len(pairs) * 2
    completed = start_pair_idx * 2

    for pair_idx in range(start_pair_idx, len(pairs)):
        pair = pairs[pair_idx]
        for variant in ["left", "right"]:
            segment = create_virtual_segment(pair, variant)
            completed += 1

            try:
                import time

                t0 = time.time()
                preds = classifier.classify(segment)
                elapsed = time.time() - t0

                for p in preds:
                    all_predictions.append(
                        PairPrediction(
                            pair_id=pair.pair_id,
                            variant=variant,
                            technique=p.technique,
                            span=p.span,
                            confidence=p.confidence,
                            model_id=p.model_id,
                            original_segment_id=pair.original_segment_id,
                        )
                    )

                n_techs = len(preds)
                logger.info(
                    f"  [{completed}/{total_tasks}] ✓ {pair.pair_id[:16]}/"
                    f"{variant} — {n_techs} techniques ({elapsed:.1f}s)"
                )
            except Exception as e:
                logger.warning(f"  [{completed}/{total_tasks}] ✗ {pair.pair_id[:16]}/{variant}: {e}")

        # Save checkpoint every 25 pairs
        if (pair_idx + 1) % 25 == 0:
            _save_checkpoint(all_predictions, model_id, pair_idx + 1)

    # Final save and clear checkpoint
    _save_checkpoint(all_predictions, model_id, len(pairs))
    _clear_checkpoint(model_id)

    if model_name == "llama3":
        classifier.close()

    return all_predictions


def _save_checkpoint(predictions: list[PairPrediction], model_id: str, pairs_done: int) -> None:
    """Save progress checkpoint."""
    checkpoint_dir = Path("results/predictions/.checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"pairs_checkpoint_{model_id}.json"
    data = {
        "model_id": model_id,
        "pairs_done": pairs_done,
        "predictions": [p.model_dump() for p in predictions],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    logger.info(f"    Checkpoint saved ({pairs_done} pairs done, {len(predictions)} predictions)")


def _load_checkpoint(model_id: str) -> tuple[list[PairPrediction], int]:
    """Load checkpoint if exists. Returns (predictions, start_pair_idx)."""
    checkpoint_dir = Path("results/predictions/.checkpoints")
    path = checkpoint_dir / f"pairs_checkpoint_{model_id}.json"
    if not path.exists():
        return [], 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_id") != model_id:
            return [], 0
        predictions = [PairPrediction.model_validate(p) for p in data["predictions"]]
        return predictions, data["pairs_done"]
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
        return [], 0


def _clear_checkpoint(model_id: str) -> None:
    """Remove checkpoint after successful completion."""
    checkpoint_dir = Path("results/predictions/.checkpoints")
    path = checkpoint_dir / f"pairs_checkpoint_{model_id}.json"
    if path.exists():
        path.unlink()
        logger.info("Checkpoint cleared (pairs classification completed)")


def main() -> None:
    """Main entry point for pair classification."""
    parser = argparse.ArgumentParser(description="Classify minimal pair variants")
    parser.add_argument(
        "--model",
        choices=["gpt4o-mini", "llama3", "all"],
        default="all",
        help="Model to classify pairs with",
    )
    args = parser.parse_args()

    setup_logging("INFO", log_file="logs/08b_classify_pairs.log")
    logger.info("=" * 60)
    logger.info("Classifying minimal pair variants for H2")
    logger.info("=" * 60)

    # Load pairs
    pairs_path = Path("data/minimal_pairs/pairs.jsonl")
    pairs = load_jsonl(pairs_path, MinimalPair)
    logger.info(f"Loaded {len(pairs)} minimal pairs ({len(pairs)*2} total classifications)")

    # Build prompt
    taxonomy = TaxonomyRegistry()
    prompt_builder = PromptBuilder(taxonomy)

    models_to_run = []
    if args.model in ("gpt4o-mini", "all"):
        models_to_run.append("gpt4o-mini")
    if args.model in ("llama3", "all"):
        models_to_run.append("llama3")

    for model_name in models_to_run:
        logger.info(f"\n--- Classifying pairs with {model_name} ---")
        predictions = classify_pairs_with_model(pairs, model_name, prompt_builder)

        # Save results
        output_dir = Path("results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_id = "gpt4o_mini" if model_name == "gpt4o-mini" else "llama3"
        output_path = output_dir / f"pairs_{model_id}.jsonl"

        with output_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(p.model_dump_json() + "\n")

        logger.info(f"Saved {len(predictions)} pair predictions to {output_path}")

        # Summary
        from collections import Counter

        by_variant = Counter(p.variant for p in predictions)
        logger.info(f"  Left variant predictions: {by_variant['left']}")
        logger.info(f"  Right variant predictions: {by_variant['right']}")

    logger.info("\n" + "=" * 60)
    logger.info("PAIR CLASSIFICATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
