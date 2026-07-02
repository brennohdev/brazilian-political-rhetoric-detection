import random
from collections import defaultdict

from loguru import logger

from src.schemas.config import SamplingConfig
from src.schemas.speech import PoliticalSpectrum, ProcessedSpeech


class StratifiedSampler:
    """Produces a stratified sample balanced across political spectrum and time.

    Stratification: 3 spectrums × 4 semesters = 12 strata.
    Per-deputy capping prevents a single deputy from dominating any stratum.
    """

    def __init__(self, config: SamplingConfig, max_per_deputy: int = 15) -> None:
        self._config = config
        self._max_per_deputy = max_per_deputy
        self._strata_counts: dict[tuple[str, str], int] = {}
        self._rng = random.Random(config.seed)

    def sample(self, speeches: list[ProcessedSpeech]) -> list[ProcessedSpeech]:
        """Draw a stratified sample balanced across spectrum × temporal period.

        If a stratum has fewer than target_per_stratum speeches, include all
        available and log a warning.

        Args:
            speeches: Full list of processed speeches to sample from.

        Returns:
            Stratified sample as a list of ProcessedSpeech.
        """
        # Group speeches into strata
        strata: dict[tuple[str, str], list[ProcessedSpeech]] = defaultdict(list)
        for speech in speeches:
            period = self._assign_temporal_period(speech)
            key = (speech.political_spectrum.value, period)
            strata[key] += [speech]

        target = self._config.target_per_stratum
        sampled: list[ProcessedSpeech] = []

        for key in sorted(strata.keys()):
            pool = strata[key]
            spectrum, period = key

            # Apply per-deputy cap to the pool
            capped_pool = self._cap_per_deputy(pool)

            if len(capped_pool) <= target:
                logger.warning(
                    f"Stratum ({spectrum}, {period}): only {len(capped_pool)} "
                    f"speeches available (target: {target}). Including all."
                )
                selected = capped_pool
            else:
                selected = self._rng.sample(capped_pool, target)

            self._strata_counts[key] = len(selected)
            sampled.extend(selected)

            logger.info(
                f"  Stratum ({spectrum:>6}, {period}): "
                f"{len(selected)}/{len(pool)} speeches "
                f"(pool after deputy cap: {len(capped_pool)})"
            )

        logger.info(
            f"Sampling complete: {len(sampled)} speeches from "
            f"{len(strata)} strata (target {target}/stratum)"
        )
        return sampled

    def _assign_temporal_period(self, speech: ProcessedSpeech) -> str:
        """Map speech date to one of 4 semesters.

        Periods: 2023-S1, 2023-S2, 2024-S1, 2024-S2
        """
        year = speech.date.year
        month = speech.date.month
        semester = 1 if month <= 6 else 2
        return f"{year}-S{semester}"

    def _cap_per_deputy(self, speeches: list[ProcessedSpeech]) -> list[ProcessedSpeech]:
        """Cap the number of speeches per deputy within a stratum.

        This prevents a single prolific deputy from dominating the sample,
        ensuring diversity of speakers in the annotated corpus.
        """
        by_deputy: dict[str, list[ProcessedSpeech]] = defaultdict(list)
        for s in speeches:
            by_deputy[s.deputy_name].append(s)

        capped: list[ProcessedSpeech] = []
        for deputy_name, deputy_speeches in by_deputy.items():
            if len(deputy_speeches) <= self._max_per_deputy:
                capped.extend(deputy_speeches)
            else:
                # Randomly select up to max_per_deputy
                capped.extend(
                    self._rng.sample(deputy_speeches, self._max_per_deputy)
                )

        return capped

    def get_stratum_counts(self) -> dict[tuple[str, str], int]:
        """Return actual counts per stratum after sampling."""
        return self._strata_counts
