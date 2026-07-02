"""Speech collection orchestrator with persistence and resumability."""

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.collection.api_client import ChamberAPIClient
from src.schemas.config import CollectionConfig
from src.schemas.speech import PoliticalSpectrum, Speech


class SpeechCollector:
    """Orchestrates speech collection with persistence and resumability.

    Responsibilities:
    - Iterates through deputies and pages
    - Persists each speech as individual JSON
    - Maintains checkpoint for resumability
    - Tracks collection statistics
    """

    CHECKPOINT_FILE = "_checkpoint.json"

    def __init__(
        self,
        client: ChamberAPIClient,
        config: CollectionConfig,
        output_dir: Path,
        party_spectrum: dict[str, PoliticalSpectrum],
    ) -> None:
        self._client = client
        self._config = config
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._party_spectrum = party_spectrum
        self._stats: dict[str, int] = {"collected": 0, "failures": 0, "skipped": 0}
        self._start_time: float = 0.0

    def collect_all(self, deputies: list[dict[str, Any]]) -> None:
        """Collect speeches for all deputies, resuming from last checkpoint.

        Args:
            deputies: List of deputy dicts with at least 'id', 'nome', 'siglaPartido'.
        """
        self._start_time = time.time()
        checkpoint = self._load_checkpoint()

        for deputy in deputies:
            deputy_id = deputy["id"]
            deputy_name = deputy.get("nome", "Unknown")
            party = deputy.get("siglaPartido", "Unknown")

            # Determine political spectrum from party
            spectrum = self._get_spectrum(party)

            # Check checkpoint — skip already-completed deputies
            last_page = checkpoint.get(str(deputy_id), 0)
            if last_page == -1:
                # -1 means fully collected
                self._stats["skipped"] += 1
                continue

            logger.info(
                f"Collecting speeches for {deputy_name} ({party}/{spectrum.value}) "
                f"starting from page {last_page + 1}"
            )

            try:
                self._collect_deputy_speeches(
                    deputy_id=deputy_id,
                    deputy_name=deputy_name,
                    party=party,
                    spectrum=spectrum,
                    start_page=last_page + 1,
                    checkpoint=checkpoint,
                )
                # Mark deputy as fully collected
                checkpoint[str(deputy_id)] = -1
                self._save_checkpoint(checkpoint)
            except Exception as e:
                logger.error(f"Failed to collect speeches for {deputy_name}: {e}")
                self._stats["failures"] += 1
                self._save_checkpoint(checkpoint)
                continue

        elapsed = time.time() - self._start_time
        logger.info(
            f"Collection complete. "
            f"Collected: {self._stats['collected']}, "
            f"Failures: {self._stats['failures']}, "
            f"Skipped (already done): {self._stats['skipped']}, "
            f"Elapsed: {elapsed:.1f}s"
        )

    def _collect_deputy_speeches(
        self,
        deputy_id: int,
        deputy_name: str,
        party: str,
        spectrum: PoliticalSpectrum,
        start_page: int,
        checkpoint: dict[str, int],
    ) -> None:
        """Collect all pages of speeches for a single deputy."""
        page = start_page

        while True:
            response = self._client.fetch_speeches(
                deputy_id=deputy_id,
                start_date=self._config.start_date,
                end_date=self._config.end_date,
                page=page,
            )

            for speech_data in response.data:
                persisted = self._persist_speech(speech_data, deputy_name, party, spectrum)
                if persisted:
                    self._stats["collected"] += 1

            # Update checkpoint after each page
            checkpoint[str(deputy_id)] = page
            self._save_checkpoint(checkpoint)

            if not response.has_next:
                break
            page = response.next_page or page + 1

    def _persist_speech(
        self,
        raw_data: dict[str, Any],
        deputy_name: str,
        party: str,
        spectrum: PoliticalSpectrum,
    ) -> bool:
        """Save a single speech as JSON with deterministic filename.

        Returns True if the speech was persisted, False if skipped.
        """
        transcription = raw_data.get("transcricao", "").strip()
        if not transcription:
            return False  # Skip empty speeches

        # Build a deterministic speech ID from available data
        date_str = raw_data.get("dataHoraInicio", "")
        content_hash = hashlib.md5(transcription[:100].encode()).hexdigest()[:8]
        safe_name = deputy_name.replace(" ", "_").replace("/", "_")
        speech_id = f"{party}_{safe_name}_{date_str[:10]}_{content_hash}"

        # Parse the date field
        speech_date: date
        if date_str:
            speech_date = date.fromisoformat(date_str[:10])
        else:
            speech_date = date(2024, 1, 1)

        # Parse faseEvento (can be dict or string)
        fase_evento = raw_data.get("faseEvento", "Unknown")
        if isinstance(fase_evento, dict):
            legislative_phase = fase_evento.get("titulo", "Unknown")
        else:
            legislative_phase = str(fase_evento) if fase_evento else "Unknown"

        speech = Speech(
            speech_id=speech_id,
            deputy_name=deputy_name,
            party=party,
            political_spectrum=spectrum,
            date=speech_date,
            session_type=raw_data.get("tipSessao", "Unknown"),
            legislative_phase=legislative_phase,
            transcription=transcription,
            collected_at=datetime.now(),
        )

        filename = f"{speech_id}.json"
        filepath = self._output_dir / filename
        filepath.write_text(speech.model_dump_json(indent=2), encoding="utf-8")
        return True

    def _get_spectrum(self, party: str) -> PoliticalSpectrum:
        """Map party to political spectrum. Default to CENTER if unknown."""
        return self._party_spectrum.get(party, PoliticalSpectrum.CENTER)

    def _load_checkpoint(self) -> dict[str, int]:
        """Load checkpoint file for resumability."""
        checkpoint_path = self._output_dir / self.CHECKPOINT_FILE
        if checkpoint_path.exists():
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return {}

    def _save_checkpoint(self, checkpoint: dict[str, int]) -> None:
        """Save checkpoint file."""
        checkpoint_path = self._output_dir / self.CHECKPOINT_FILE
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2), encoding="utf-8"
        )

    def get_stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            **self._stats,
            "elapsed_seconds": round(elapsed, 1),
        }
