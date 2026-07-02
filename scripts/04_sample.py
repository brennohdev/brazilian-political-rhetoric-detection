from pathlib import Path

from loguru import logger

from src.config.loader import ConfigLoader
from src.sampling.stratifier import StratifiedSampler
from src.schemas.speech import ProcessedSpeech
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logging
from src.utils.seeds import set_global_seed


def main() -> None:
    """Main entry point for stratified sampling."""
    setup_logging("INFO", log_file="logs/04_sample.log")
    logger.info("=" * 60)
    logger.info("Starting stratified sampling")
    logger.info("=" * 60)

    # Load config
    loader = ConfigLoader()
    config = loader.load_sampling()
    set_global_seed(config.seed)

    logger.info(f"Target per stratum: {config.target_per_stratum}")
    logger.info(f"Temporal periods: {config.temporal_periods}")
    logger.info(f"Seed: {config.seed}")

    # Load processed speeches
    input_path = Path("data/processed/speeches.jsonl")
    speeches = load_jsonl(input_path, ProcessedSpeech)
    logger.info(f"Loaded {len(speeches)} processed speeches")

    # Run stratified sampling
    sampler = StratifiedSampler(config, max_per_deputy=15)
    sample = sampler.sample(speeches)

    # Persist
    output_path = Path("data/samples/sample.jsonl")
    save_jsonl(sample, output_path)
    logger.info(f"Saved {len(sample)} sampled speeches to {output_path}")

    # Report
    strata = sampler.get_stratum_counts()
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAMPLING COMPLETE")
    logger.info(f"  Total sampled: {len(sample)}")
    logger.info(f"  Strata: {len(strata)}")
    logger.info(f"  Unique deputies: {len(set(s.deputy_name for s in sample))}")
    logger.info(f"  Min stratum size: {min(strata.values())}")
    logger.info(f"  Max stratum size: {max(strata.values())}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
