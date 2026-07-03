import hashlib
import re
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.schemas.annotation import MinimalPair
from src.schemas.speech import SpeechSegment
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import setup_logging

load_dotenv()

# Political referents for substitution
LEFT_REFERENTS = [
    "Lula", "PT", "Governo Lula", "governo Lula", "Presidente Lula",
    "petistas", "petista", "esquerdistas", "esquerda", "governo do PT",
    "Gleisi", "Haddad", "MST", "CUT", "PSOL",
]

RIGHT_REFERENTS = [
    "Bolsonaro", "PL", "Governo Bolsonaro", "governo Bolsonaro", "Presidente Bolsonaro",
    "bolsonaristas", "bolsonarista", "direitistas", "direita", "governo do PL",
    "Michelle", "Tarcísio", "agronegócio", "bancada da bala", "NOVO",
]

# Create substitution mappings (left→right and right→left)
LEFT_TO_RIGHT = dict(zip(LEFT_REFERENTS, RIGHT_REFERENTS))
RIGHT_TO_LEFT = dict(zip(RIGHT_REFERENTS, LEFT_REFERENTS))


def find_referents(text: str) -> tuple[list[str], str]:
    """Find political referents in text and determine which spectrum they belong to.

    Returns (found_referents, spectrum) where spectrum is 'left', 'right', or 'none'.
    """
    left_found = []
    right_found = []

    for ref in LEFT_REFERENTS:
        if re.search(r"\b" + re.escape(ref) + r"\b", text):
            left_found.append(ref)

    for ref in RIGHT_REFERENTS:
        if re.search(r"\b" + re.escape(ref) + r"\b", text):
            right_found.append(ref)

    if left_found and not right_found:
        return left_found, "left"
    elif right_found and not left_found:
        return right_found, "right"
    elif left_found and right_found:
        return left_found + right_found, "both"  # Skip mixed segments
    else:
        return [], "none"


def swap_referents(text: str, found: list[str], spectrum: str) -> str:
    """Swap political referents in text to create the counter-factual variant."""
    mapping = LEFT_TO_RIGHT if spectrum == "left" else RIGHT_TO_LEFT
    result = text

    # Sort by length (longest first) to avoid partial replacements
    sorted_refs = sorted(found, key=len, reverse=True)
    for ref in sorted_refs:
        if ref in mapping:
            replacement = mapping[ref]
            result = re.sub(r"\b" + re.escape(ref) + r"\b", replacement, result)

    return result


def build_pair(segment: SpeechSegment, found: list[str], spectrum: str) -> MinimalPair | None:
    """Build a minimal pair from a segment with political referents."""
    swapped_text = swap_referents(segment.text, found, spectrum)

    # Verify the swap actually changed something
    if swapped_text == segment.text:
        return None

    # Determine left/right variants
    if spectrum == "left":
        left_variant = segment.text
        right_variant = swapped_text
        original_referent = ", ".join(found)
        replacement_referent = ", ".join(LEFT_TO_RIGHT.get(r, r) for r in found)
    else:
        right_variant = segment.text
        left_variant = swapped_text
        original_referent = ", ".join(found)
        replacement_referent = ", ".join(RIGHT_TO_LEFT.get(r, r) for r in found)

    pair_id = f"pair_{hashlib.md5(segment.segment_id.encode()).hexdigest()[:12]}"

    return MinimalPair(
        pair_id=pair_id,
        original_segment_id=segment.segment_id,
        left_variant=left_variant,
        right_variant=right_variant,
        original_referent=original_referent,
        replacement_referent=replacement_referent,
        original_spectrum=spectrum if spectrum == "left" else "right",
        replacement_spectrum="right" if spectrum == "left" else "left",
        flagged_for_review=False,
    )


def main() -> None:
    """Main entry point for minimal pairs construction."""
    setup_logging("INFO", log_file="logs/07_build_pairs.log")
    logger.info("=" * 60)
    logger.info("Building minimal pairs for H2 bias evaluation")
    logger.info("=" * 60)

    # Load segments
    segments_path = Path("data/segments/segments.jsonl")
    segments = load_jsonl(segments_path, SpeechSegment)
    logger.info(f"Loaded {len(segments)} segments")

    # Find segments with political referents
    pairs: list[MinimalPair] = []
    skipped_both = 0
    skipped_none = 0
    skipped_no_change = 0

    for segment in segments:
        found, spectrum = find_referents(segment.text)

        if spectrum == "none":
            skipped_none += 1
            continue
        if spectrum == "both":
            skipped_both += 1
            continue

        pair = build_pair(segment, found, spectrum)
        if pair is None:
            skipped_no_change += 1
            continue

        pairs.append(pair)

    # Save
    output_path = Path("data/minimal_pairs/pairs.jsonl")
    save_jsonl(pairs, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("MINIMAL PAIRS CONSTRUCTION COMPLETE")
    logger.info(f"  Total segments analyzed: {len(segments)}")
    logger.info(f"  Pairs created: {len(pairs)}")
    logger.info(f"  Skipped (no referents): {skipped_none}")
    logger.info(f"  Skipped (mixed referents): {skipped_both}")
    logger.info(f"  Skipped (swap had no effect): {skipped_no_change}")
    logger.info(f"  From left-spectrum originals: {sum(1 for p in pairs if p.original_spectrum == 'left')}")
    logger.info(f"  From right-spectrum originals: {sum(1 for p in pairs if p.original_spectrum == 'right')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
