import re

from loguru import logger

from src.schemas.speech import ProcessedSpeech, SpeechSegment


class SpeechSegmenter:
    """Splits processed speeches into 3-5 sentence segments.

    Each segment:
    - Contains 3-5 complete sentences
    - Has a traceable ID: {speech_id}_seg{NNN}
    - Preserves char offsets for span-level annotation
    - Preserves all metadata (deputy, party, spectrum)
    """

    # Sentence boundary: period/question/exclamation followed by
    # whitespace and uppercase letter, or end of string
    SENTENCE_BOUNDARY = re.compile(
        r"(?<=[.!?])\s+(?=[A-ZÀ-ÚÇ])"
    )

    def __init__(self, min_sentences: int = 3, max_sentences: int = 5) -> None:
        self._min = min_sentences
        self._max = max_sentences

    def segment(self, speech: ProcessedSpeech) -> list[SpeechSegment]:
        """Split a single speech into segments.

        Strategy: greedy chunking of `max_sentences` sentences per segment.
        If the remainder has fewer than `min_sentences`, merge with the
        previous segment (if it exists) rather than discarding.

        Returns:
            List of SpeechSegment objects with char offsets.
        """
        text = speech.clean_text
        sentences = self._split_sentences(text)

        if len(sentences) < self._min:
            return []  # Speech too short (shouldn't happen post-filter)

        # Build segments using greedy chunking
        segments: list[SpeechSegment] = []
        i = 0
        seg_num = 0

        while i < len(sentences):
            remaining = len(sentences) - i

            if remaining < self._min:
                # Merge remainder with previous segment if possible
                if segments:
                    # Extend the last segment
                    last = segments[-1]
                    new_text = text[last.start_char:]
                    new_sent_count = last.sentence_count + remaining
                    segments[-1] = SpeechSegment(
                        segment_id=last.segment_id,
                        speech_id=last.speech_id,
                        deputy_name=last.deputy_name,
                        party=last.party,
                        political_spectrum=last.political_spectrum,
                        text=new_text.strip(),
                        sentence_count=min(new_sent_count, self._max + 2),
                        start_char=last.start_char,
                        end_char=len(text),
                    )
                break

            # Take max_sentences (or whatever is left if it's >= min)
            chunk_size = min(self._max, remaining)

            # But if remaining after this chunk would be < min and > 0,
            # reduce chunk size to balance
            leftover = remaining - chunk_size
            if 0 < leftover < self._min:
                # Split more evenly
                chunk_size = remaining // 2
                if chunk_size < self._min:
                    chunk_size = self._min

            chunk_sentences = sentences[i : i + chunk_size]

            # Calculate char offsets
            start_char = text.index(chunk_sentences[0])
            # End char: end of last sentence in chunk
            last_sent = chunk_sentences[-1]
            end_char = text.index(last_sent, start_char) + len(last_sent)

            seg_num += 1
            segment_id = f"{speech.speech_id}_seg{seg_num:03d}"

            segment = SpeechSegment(
                segment_id=segment_id,
                speech_id=speech.speech_id,
                deputy_name=speech.deputy_name,
                party=speech.party,
                political_spectrum=speech.political_spectrum,
                text=text[start_char:end_char].strip(),
                sentence_count=len(chunk_sentences),
                start_char=start_char,
                end_char=end_char,
            )
            segments.append(segment)
            i += chunk_size

        return segments

    def segment_batch(self, speeches: list[ProcessedSpeech]) -> list[SpeechSegment]:
        """Segment all speeches and flatten results.

        Args:
            speeches: List of processed speeches to segment.

        Returns:
            Flat list of all segments across all speeches.
        """
        all_segments: list[SpeechSegment] = []
        empty_count = 0

        for speech in speeches:
            segs = self.segment(speech)
            if not segs:
                empty_count += 1
            all_segments.extend(segs)

        if empty_count > 0:
            logger.warning(
                f"{empty_count} speeches produced no segments "
                f"(fewer than {self._min} sentences)"
            )

        logger.info(
            f"Segmented {len(speeches)} speeches into "
            f"{len(all_segments)} segments "
            f"(avg {len(all_segments)/max(len(speeches),1):.1f} segments/speech)"
        )
        return all_segments

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex boundary detection."""
        sentences = self.SENTENCE_BOUNDARY.split(text)
        # Filter out empty fragments
        return [s.strip() for s in sentences if s.strip()]
