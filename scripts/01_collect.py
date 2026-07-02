from pathlib import Path

import yaml
from loguru import logger

from src.collection.api_client import ChamberAPIClient
from src.collection.collector import SpeechCollector
from src.config.loader import ConfigLoader
from src.schemas.speech import PoliticalSpectrum
from src.utils.logging import setup_logging


def load_party_spectrum(config_dir: Path = Path("configs")) -> dict[str, PoliticalSpectrum]:
    """Load party-to-spectrum mapping from deputies.yaml."""
    deputies_path = config_dir / "deputies.yaml"
    raw = yaml.safe_load(deputies_path.read_text(encoding="utf-8"))

    mapping: dict[str, PoliticalSpectrum] = {}
    party_spectrum = raw.get("party_spectrum", {})

    for spectrum_name, parties in party_spectrum.items():
        spectrum = PoliticalSpectrum(spectrum_name)
        for party in parties:
            mapping[party] = spectrum

    return mapping


def main() -> None:
    """Main entry point for speech collection."""
    setup_logging("INFO", log_file="logs/01_collect.log")
    logger.info("=" * 60)
    logger.info("Starting speech collection from Chamber of Deputies API")
    logger.info("=" * 60)

    # Load and validate configuration
    loader = ConfigLoader()
    config = loader.load_collection()
    logger.info(f"Collection period: {config.start_date} to {config.end_date}")
    logger.info(f"Session types: {config.session_types}")
    logger.info(f"Output directory: {config.output_dir}")

    # Load party-spectrum mapping
    party_spectrum = load_party_spectrum()
    logger.info(f"Loaded spectrum mapping for {len(party_spectrum)} parties")

    # Initialize API client and collector
    output_dir = Path(config.output_dir)

    with ChamberAPIClient(config) as client:
        # First, fetch the list of deputies for the current legislature
        logger.info("Fetching deputy list from API...")
        deputies = client.fetch_deputies(legislature_id=57)
        logger.info(f"Found {len(deputies)} deputies in legislature 57")

        # Filter deputies to only those with known party-spectrum mapping
        known_deputies = [
            d for d in deputies
            if d.get("siglaPartido") in party_spectrum
        ]
        logger.info(
            f"Filtering to {len(known_deputies)} deputies with known party spectrum "
            f"(excluded {len(deputies) - len(known_deputies)} with unknown parties)"
        )

        # Initialize collector and run
        collector = SpeechCollector(
            client=client,
            config=config,
            output_dir=output_dir,
            party_spectrum=party_spectrum,
        )

        collector.collect_all(known_deputies)

        # Report final stats
        stats = collector.get_stats()
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"  Speeches collected: {stats['collected']}")
        logger.info(f"  Deputies failed: {stats['failures']}")
        logger.info(f"  Deputies skipped (resumed): {stats['skipped']}")
        logger.info(f"  Total time: {stats['elapsed_seconds']}s")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
