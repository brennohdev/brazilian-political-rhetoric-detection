import argparse
import json
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

load_dotenv()  # Load .env file (OPENAI_API_KEY)

from src.classification.prompt_builder import PromptBuilder
from src.classification.runner import ClassificationRunner
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logging


class PilotReport(BaseModel):
    """Pilot study results for documentation."""

    model_id: str
    n_segments: int
    n_predictions: int
    technique_counts: dict[str, int]
    technique_prevalence: dict[str, float]
    segments_with_any_technique: int
    prevalence_any: float
    failure_rate: float
    avg_techniques_per_segment: float


def run_pilot(
    segments: list[SpeechSegment],
    model: str,
    prompt_builder: PromptBuilder,
) -> tuple[list[list[Prediction]], float]:
    """Run pilot classification with selected model.

    Returns (results, failure_rate).
    """
    if model == "gpt4o":
        from src.classification.gpt4o import GPT4oClassifier
        from src.config.loader import ConfigLoader

        loader = ConfigLoader()
        models = loader.load_models()
        config = next((m for m in models if "gpt" in m.name.lower()), models[0])
        classifier = GPT4oClassifier(config=config, prompt_builder=prompt_builder)
    elif model == "gpt4o-mini":
        from src.classification.gpt4o import GPT4oClassifier
        from src.config.loader import ConfigLoader

        loader = ConfigLoader()
        models = loader.load_models()
        config = next((m for m in models if "gpt" in m.name.lower()), models[0])
        classifier = GPT4oClassifier(
            config=config, prompt_builder=prompt_builder, model_name="gpt-4o-mini"
        )
    elif model == "llama3":
        from src.classification.llama import LLaMAClassifier
        from src.config.loader import ConfigLoader

        loader = ConfigLoader()
        models = loader.load_models()
        config = next((m for m in models if "llama" in m.name.lower()), models[1])
        classifier = LLaMAClassifier(config=config, prompt_builder=prompt_builder)
    else:
        raise ValueError(f"Unknown model: {model}")

    runner = ClassificationRunner(classifier, failure_threshold=0.20)  # Relaxed for pilot
    results = runner.run(segments)
    return results, runner.failure_rate


def analyze_results(
    results: list[list[Prediction]],
    model_id: str,
    n_segments: int,
    failure_rate: float,
    technique_names: list[str],
) -> PilotReport:
    """Analyze pilot results and produce report."""
    all_predictions = [p for preds in results for p in preds]

    # Count techniques
    technique_counts = Counter(p.technique for p in all_predictions)
    # Ensure all techniques are represented in counts
    for name in technique_names:
        if name not in technique_counts:
            technique_counts[name] = 0

    # Prevalence = proportion of segments containing each technique
    technique_prevalence: dict[str, float] = {}
    for name in technique_names:
        segments_with = sum(1 for preds in results if any(p.technique == name for p in preds))
        technique_prevalence[name] = segments_with / n_segments

    segments_with_any = sum(1 for preds in results if len(preds) > 0)
    avg_techniques = len(all_predictions) / max(n_segments, 1)

    return PilotReport(
        model_id=model_id,
        n_segments=n_segments,
        n_predictions=len(all_predictions),
        technique_counts=dict(technique_counts),
        technique_prevalence=technique_prevalence,
        segments_with_any_technique=segments_with_any,
        prevalence_any=segments_with_any / n_segments,
        failure_rate=failure_rate,
        avg_techniques_per_segment=avg_techniques,
    )


def main() -> None:
    """Main entry point for pilot study."""
    parser = argparse.ArgumentParser(description="Pilot classification study")
    parser.add_argument(
        "--model",
        choices=["gpt4o", "gpt4o-mini", "llama3"],
        required=True,
        help="Model to use for pilot",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of segments to classify (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for segment selection",
    )
    args = parser.parse_args()

    setup_logging("INFO", log_file="logs/06_pilot.log")
    logger.info("=" * 60)
    logger.info(f"PILOT STUDY — {args.model} on {args.n} segments")
    logger.info("=" * 60)

    # Load segments and select subset
    segments_path = Path("data/segments/segments.jsonl")
    all_segments = load_jsonl(segments_path, SpeechSegment)
    logger.info(f"Loaded {len(all_segments)} total segments")

    rng = random.Random(args.seed)
    pilot_segments = rng.sample(all_segments, min(args.n, len(all_segments)))
    logger.info(f"Selected {len(pilot_segments)} for pilot")

    # Build prompt
    taxonomy = TaxonomyRegistry()
    prompt_builder = PromptBuilder(taxonomy)

    # Run classification
    results, failure_rate = run_pilot(pilot_segments, args.model, prompt_builder)

    # Analyze
    report = analyze_results(
        results=results,
        model_id=args.model,
        n_segments=len(pilot_segments),
        failure_rate=failure_rate,
        technique_names=taxonomy.technique_names,
    )

    # Save outputs
    output_dir = Path("results/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"pilot_report_{args.model}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")

    # Save predictions
    all_preds = [p for preds in results for p in preds]
    preds_path = output_dir / f"pilot_predictions_{args.model}.jsonl"
    save_jsonl(all_preds, preds_path)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PILOT RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Model: {report.model_id}")
    logger.info(f"  Segments classified: {report.n_segments}")
    logger.info(f"  Total predictions: {report.n_predictions}")
    logger.info(f"  Failure rate: {report.failure_rate:.1%}")
    logger.info(f"  Segments with any technique: {report.segments_with_any_technique} ({report.prevalence_any:.1%})")
    logger.info(f"  Avg techniques/segment: {report.avg_techniques_per_segment:.2f}")
    logger.info("")
    logger.info("  Per-technique prevalence:")
    for name, prev in sorted(report.technique_prevalence.items(), key=lambda x: -x[1]):
        count = report.technique_counts.get(name, 0)
        logger.info(f"    {name:<30} {prev:>6.1%} ({count} detections)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
