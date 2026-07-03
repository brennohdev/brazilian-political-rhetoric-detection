import json
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.annotation.loader import AnnotationLoader
from src.evaluation.agreement import AgreementEvaluator
from src.evaluation.bias import BiasEvaluator
from src.evaluation.performance import PerformanceEvaluator
from src.schemas.annotation import MinimalPair
from src.schemas.prediction import Prediction
from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry
from src.utils.io import load_jsonl
from src.utils.logging import setup_logging

load_dotenv()


def load_predictions(model_id: str) -> dict[str, set[str]]:
    """Load predictions as segment_id → set of techniques."""
    path = Path(f"results/predictions/{model_id}.jsonl")
    if not path.exists():
        logger.warning(f"Predictions not found: {path}")
        return {}
    preds = load_jsonl(path, Prediction)
    result: dict[str, set[str]] = {}
    for p in preds:
        result.setdefault(p.segment_id, set()).add(p.technique)
    return result


def evaluate_h1(
    annotations_dir: Path, technique_names: list[str]
) -> dict:
    """H1: Inter-annotator agreement."""
    loader = AnnotationLoader()

    ann_a_path = annotations_dir / "annotator_a.jsonl"
    ann_b_path = annotations_dir / "annotator_b.jsonl"

    if not ann_a_path.exists() or not ann_b_path.exists():
        logger.warning("Annotation files not found. Skipping H1.")
        return {}

    ann_a = loader.load_annotations(ann_a_path)
    ann_b = loader.load_annotations(ann_b_path)

    aligned = loader.align_by_segment(ann_a, ann_b)
    labels_a = {seg: techs[0] for seg, techs in aligned.items()}
    labels_b = {seg: techs[1] for seg, techs in aligned.items()}

    evaluator = AgreementEvaluator()
    result = evaluator.evaluate(labels_a, labels_b, technique_names)

    logger.info(f"H1 — Global κ: {result.global_kappa:.4f}")
    for tech, kappa in result.per_technique_kappa.items():
        flag = " ⚠️ BELOW THRESHOLD" if kappa < 0.40 else ""
        logger.info(f"  {tech:<30} κ={kappa:.4f}{flag}")

    return result.model_dump()


def evaluate_h2(
    technique_names: list[str], model_ids: list[str]
) -> list[dict]:
    """H2: Political bias via minimal pairs."""
    pairs_path = Path("data/minimal_pairs/pairs.jsonl")
    if not pairs_path.exists():
        logger.warning("Minimal pairs not found. Skipping H2.")
        return []

    pairs = load_jsonl(pairs_path, MinimalPair)
    if not pairs:
        logger.warning("No minimal pairs available. Skipping H2.")
        return []

    logger.info(f"Loaded {len(pairs)} minimal pairs for H2")

    # For each model, classify both variants and compare
    # This requires running classification on the pair variants
    # For now, we evaluate based on existing predictions if segments match
    evaluator = BiasEvaluator(significance_level=0.05, n_tests=len(model_ids) * len(technique_names))

    all_results = []
    for model_id in model_ids:
        preds = load_predictions(model_id)
        if not preds:
            continue

        for technique in technique_names:
            # Check which original segments have predictions
            left_detected = []
            right_detected = []

            for pair in pairs:
                # The original segment should have predictions
                seg_id = pair.original_segment_id
                if seg_id in preds:
                    orig_has = technique in preds[seg_id]
                else:
                    orig_has = False

                # For the swapped variant, we'd need to re-classify
                # This is a placeholder — full H2 requires classifying both variants
                left_detected.append(orig_has if pair.original_spectrum == "left" else False)
                right_detected.append(orig_has if pair.original_spectrum == "right" else False)

            if any(left_detected) or any(right_detected):
                result = evaluator.evaluate(left_detected, right_detected, model_id, technique)
                all_results.append(result.model_dump())

    return all_results


def evaluate_h3(
    annotations_dir: Path, technique_names: list[str], model_ids: list[str]
) -> list[dict]:
    """H3: Model performance comparison."""
    # Load gold labels from consolidated annotations
    consolidated_path = annotations_dir / "annotations.jsonl"
    if not consolidated_path.exists():
        logger.warning("Consolidated annotations not found. Skipping H3.")
        return []

    gold: dict[str, set[str]] = {}
    with open(consolidated_path) as f:
        for line in f:
            ann = json.loads(line)
            seg_id = ann["segment_id"]
            techniques = set(ann.get("techniques", []))
            gold[seg_id] = techniques

    logger.info(f"Loaded {len(gold)} gold-labeled segments for H3")

    evaluator = PerformanceEvaluator(bootstrap_iterations=10_000, seed=42)
    results = []

    for model_id in model_ids:
        preds = load_predictions(model_id)
        if not preds:
            logger.warning(f"No predictions for {model_id}. Skipping.")
            continue

        result = evaluator.evaluate(gold, preds, technique_names, model_id)
        results.append(result.model_dump())

        logger.info(f"H3 — {model_id}:")
        logger.info(f"  Macro-F1: {result.macro_f1:.4f} [{result.macro_f1_ci.lower:.4f}, {result.macro_f1_ci.upper:.4f}]")
        logger.info(f"  Micro-F1: {result.micro_f1:.4f}")
        for tech, metrics in result.per_technique.items():
            logger.info(f"  {tech:<30} P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    return results


def main() -> None:
    """Main entry point for evaluation."""
    setup_logging("INFO", log_file="logs/11_evaluate.log")
    logger.info("=" * 60)
    logger.info("Starting evaluation (H1, H2, H3)")
    logger.info("=" * 60)

    taxonomy = TaxonomyRegistry()
    technique_names = taxonomy.technique_names
    model_ids = ["gpt4o_mini", "llama3", "bertimbau"]

    annotations_dir = Path("data/annotations/consolidated")

    # H1: Agreement
    logger.info("\n--- H1: Inter-Annotator Agreement ---")
    h1_results = evaluate_h1(annotations_dir, technique_names)

    # H2: Bias
    logger.info("\n--- H2: Political Bias ---")
    h2_results = evaluate_h2(technique_names, model_ids)

    # H3: Performance
    logger.info("\n--- H3: Model Performance ---")
    h3_results = evaluate_h3(annotations_dir, technique_names, model_ids)

    # Save all results
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"

    evaluation = {
        "h1_agreement": h1_results,
        "h2_bias": h2_results,
        "h3_performance": h3_results,
    }
    output_path.write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")

    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
