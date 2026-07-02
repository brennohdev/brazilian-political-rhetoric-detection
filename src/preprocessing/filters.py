import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.report import FilterReport
from src.schemas.speech import ProcessedSpeech, Speech


class FilterStep(ABC):
    """Abstract filter step — Strategy pattern.

    Each step either:
    - Transforms the speech (returns modified Speech)
    - Excludes the speech (returns None)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for reporting."""
        ...

    @abstractmethod
    def apply(self, speech: Speech) -> Speech | None:
        """Apply filter. Return None to exclude the speech."""
        ...


class LegislativePhaseFilter(FilterStep):
    """Retain only argumentative legislative phases.

    Decision: Filter on `legislative_phase` field (not `session_type` which is
    always "Unknown" in the collected data). Retains "Ordem do Dia" and
    "Breves Comunicações" as these are the argumentative sessions where deputies
    present political positions.

    Excludes: Encerramento, Abertura, Homenagem, Comissão Geral.
    """

    ARGUMENTATIVE_PHASES = {"Ordem do Dia", "Breves Comunicações"}

    @property
    def name(self) -> str:
        return "legislative_phase_filter"

    def apply(self, speech: Speech) -> Speech | None:
        if speech.legislative_phase in self.ARGUMENTATIVE_PHASES:
            return speech
        return None


class MonologueIsolator(FilterStep):
    """Isolate the deputy's first continuous monologue.

    Parliamentary transcriptions may include interjections from the president
    or other deputies (e.g., "O SR. PRESIDENTE (Fulano) - ..."). This filter
    extracts only the main speaker's continuous text before any interruption.

    Pattern: Other speakers are introduced with "O SR. " or "A SRA. " followed
    by a name different from the current speaker.
    """

    # Pattern: "O SR. NAME" or "A SRA. NAME" indicating a different speaker
    SPEAKER_CHANGE_PATTERN = re.compile(
        r"(?:O SR\.|A SRA\.) (?:PRESIDENTE|PRESIDENTA)[\s(]"
    )

    @property
    def name(self) -> str:
        return "monologue_isolation"

    def apply(self, speech: Speech) -> Speech | None:
        text = speech.transcription

        # Find first speaker change (president interjection)
        match = self.SPEAKER_CHANGE_PATTERN.search(text)
        if match:
            # Keep only text before the interjection
            text = text[: match.start()].rstrip()

        if not text.strip():
            return None

        return speech.model_copy(update={"transcription": text})


class FormalityRemover(FilterStep):
    """Remove formal parliamentary address patterns and procedural markers.

    These patterns are metadata/protocol, not rhetorical content:
    - Opening identification: "O SR. FULANO (Partido - UF. Sem revisão do orador.) -"
    - Formal addresses: "Sr. Presidente", "Sras. e Srs. Deputados"
    - Closing formulas: "Muito obrigado, Sr. Presidente."
    - Session markers: "(Sem revisão do orador.)"

    We only remove the opening identification block and revision markers,
    preserving the actual speech content including rhetorical addresses to
    the president (which can be part of the argumentative strategy).
    """

    # Opening block: "O SR. NAME (Party - State. Sem revisão...) -"
    OPENING_PATTERN = re.compile(
        r"^O SR\.\s+.+?\)\s*-\s*", re.DOTALL
    )
    # Alternative opening for women: "A SRA. NAME (...) -"
    OPENING_PATTERN_F = re.compile(
        r"^A SRA\.\s+.+?\)\s*-\s*", re.DOTALL
    )
    # Revision note in the middle of text
    REVISION_NOTE = re.compile(
        r"\(Sem revisão do orador\.?\)", re.IGNORECASE
    )

    @property
    def name(self) -> str:
        return "formality_removal"

    def apply(self, speech: Speech) -> Speech | None:
        text = speech.transcription

        # Remove opening identification block
        text = self.OPENING_PATTERN.sub("", text, count=1)
        text = self.OPENING_PATTERN_F.sub("", text, count=1)

        # Remove revision notes
        text = self.REVISION_NOTE.sub("", text)

        text = text.strip()
        if not text:
            return None

        return speech.model_copy(update={"transcription": text})


class TextNormalizer(FilterStep):
    """Normalize Unicode and collapse whitespace.

    - NFKC normalization (canonical decomposition + compatibility composition)
    - Collapse multiple spaces/newlines into single space
    - Strip leading/trailing whitespace
    - Preserve sentence-ending punctuation
    """

    @property
    def name(self) -> str:
        return "text_normalization"

    def apply(self, speech: Speech) -> Speech | None:
        text = speech.transcription

        # Unicode normalization (NFKC handles compatibility characters)
        text = unicodedata.normalize("NFKC", text)

        # Replace newlines and tabs with spaces
        text = re.sub(r"[\n\r\t]+", " ", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        text = text.strip()
        if not text:
            return None

        return speech.model_copy(update={"transcription": text})


class MinimumLengthFilter(FilterStep):
    """Exclude speeches with fewer than 3 sentences after processing.

    Segmentation requires at least 3 sentences per segment. Speeches shorter
    than this cannot produce even one valid segment and should be excluded.

    Sentence detection uses a simple regex heuristic suitable for Portuguese
    parliamentary text (period, question mark, or exclamation followed by
    space and uppercase letter, or end of text).
    """

    SENTENCE_BOUNDARY = re.compile(
        r"[.!?](?:\s+(?=[A-ZÀ-ÚÇ])|$)"
    )
    MIN_SENTENCES = 3

    @property
    def name(self) -> str:
        return "minimum_length"

    def apply(self, speech: Speech) -> Speech | None:
        text = speech.transcription.strip()
        # Count sentences by splitting on boundaries
        sentences = self.SENTENCE_BOUNDARY.split(text)
        # Filter out empty fragments
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < self.MIN_SENTENCES:
            return None
        return speech


class Deduplicator(FilterStep):
    """TF-IDF cosine similarity deduplication (per-deputy, threshold >= 0.75).

    Some deputies deliver very similar speeches across sessions (template
    speeches, repeated procedural statements). This filter identifies near-
    duplicates using TF-IDF vectorization and cosine similarity.

    Operates per-deputy to avoid cross-deputy comparisons (different deputies
    can legitimately make similar arguments).

    NOTE: This filter is stateful — it must process all speeches for a deputy
    at once. The SpeechFilter orchestrator handles this by calling
    `apply_batch()` instead of `apply()`.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "deduplication"

    def apply(self, speech: Speech) -> Speech | None:
        # Single-speech apply is a no-op (dedup requires batch context)
        return speech

    def apply_batch(self, speeches: list[Speech]) -> list[Speech]:
        """Deduplicate speeches per deputy using TF-IDF cosine similarity."""
        if not speeches:
            return []

        # Group by deputy
        by_deputy: dict[str, list[tuple[int, Speech]]] = {}
        for i, speech in enumerate(speeches):
            by_deputy.setdefault(speech.deputy_name, []).append((i, speech))

        keep_indices: set[int] = set()

        for deputy_name, deputy_speeches in by_deputy.items():
            if len(deputy_speeches) <= 1:
                keep_indices.add(deputy_speeches[0][0])
                continue

            texts = [s.transcription for _, s in deputy_speeches]
            indices = [i for i, _ in deputy_speeches]

            try:
                vectorizer = TfidfVectorizer(
                    max_features=5000,
                    strip_accents="unicode",
                    lowercase=True,
                )
                tfidf_matrix = vectorizer.fit_transform(texts)
                sim_matrix = cosine_similarity(tfidf_matrix)

                # Greedy dedup: keep first occurrence, remove later duplicates
                removed: set[int] = set()
                for j in range(len(texts)):
                    if j in removed:
                        continue
                    keep_indices.add(indices[j])
                    for k in range(j + 1, len(texts)):
                        if k in removed:
                            continue
                        if sim_matrix[j, k] >= self._threshold:
                            removed.add(k)
            except ValueError:
                # If TF-IDF fails (e.g., empty vocabulary), keep all
                for idx, _ in deputy_speeches:
                    keep_indices.add(idx)

        # Preserve original order
        return [speeches[i] for i in sorted(keep_indices)]


class SpeechFilter:
    """Orchestrates composable filter steps in sequence.

    Applies each FilterStep to the speech list, tracks removals per step,
    and produces a FilterReport for documentation.
    """

    def __init__(self, steps: list[FilterStep]) -> None:
        self._steps = steps
        self._report = FilterReport()

    def filter_all(self, speeches: list[Speech]) -> list[ProcessedSpeech]:
        """Apply all filter steps, track removals per step.

        Returns ProcessedSpeech objects with clean_text and sentence_count.
        """
        self._report = FilterReport()
        self._report.total_input = len(speeches)
        current = speeches

        for step in self._steps:
            input_count = len(current)

            if isinstance(step, Deduplicator):
                # Deduplicator needs batch processing
                current = step.apply_batch(current)
            else:
                # Standard per-speech filtering
                filtered = []
                for speech in current:
                    result = step.apply(speech)
                    if result is not None:
                        filtered.append(result)
                current = filtered

            output_count = len(current)
            self._report.add_step(step.name, input_count, output_count)
            logger.info(
                f"  [{step.name}] {input_count} → {output_count} "
                f"(removed {input_count - output_count})"
            )

        self._report.total_output = len(current)

        # Convert to ProcessedSpeech
        processed = self._to_processed(current)
        return processed

    def _to_processed(self, speeches: list[Speech]) -> list[ProcessedSpeech]:
        """Convert filtered Speech objects to ProcessedSpeech with metadata."""
        processed = []
        sentence_boundary = re.compile(r"[.!?](?:\s+(?=[A-ZÀ-ÚÇ])|$)")

        for speech in speeches:
            text = speech.transcription.strip()
            sentences = sentence_boundary.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = len(sentences)

            if sentence_count < 3:
                continue  # Safety check

            # Determine session type from legislative_phase
            from src.schemas.speech import SessionType

            if speech.legislative_phase == "Ordem do Dia":
                session_type = SessionType.ORDEM_DO_DIA
            else:
                session_type = SessionType.GRANDE_EXPEDIENTE

            # Generate new speech_id for processed version
            content_hash = hashlib.md5(text[:200].encode()).hexdigest()[:8]
            processed_id = f"proc_{speech.speech_id[:50]}_{content_hash}"

            processed.append(
                ProcessedSpeech(
                    speech_id=processed_id,
                    deputy_name=speech.deputy_name,
                    party=speech.party,
                    political_spectrum=speech.political_spectrum,
                    date=speech.date,
                    session_type=session_type,
                    clean_text=text,
                    sentence_count=sentence_count,
                    original_speech_id=speech.speech_id,
                )
            )

        return processed

    @property
    def report(self) -> FilterReport:
        """Access the filtering report after running filter_all()."""
        return self._report
