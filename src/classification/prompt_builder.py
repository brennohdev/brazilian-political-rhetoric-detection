from src.schemas.speech import SpeechSegment
from src.taxonomy.registry import TaxonomyRegistry


class PromptBuilder:
    """Constructs few-shot prompts for generative classifiers.

    The prompt structure:
    1. System instruction (role and task definition)
    2. Taxonomy definitions with examples (from taxonomy.yaml)
    3. Output format specification (JSON)
    4. Target segment for classification
    """

    def __init__(self, taxonomy: TaxonomyRegistry) -> None:
        self._taxonomy = taxonomy
        self._taxonomy_section = self._build_taxonomy_section()

    def get_system_message(self) -> str:
        """Return the system message defining the classifier's role."""
        return (
            "You are an expert linguist specialized in rhetorical analysis of "
            "Brazilian Portuguese political discourse. Your task is to identify "
            "rhetorical manipulation techniques in parliamentary speech segments "
            "from the Brazilian Chamber of Deputies.\n\n"
            "You will analyze short text passages (3-5 sentences) and identify "
            "which, if any, of the following 6 manipulation techniques are present. "
            "A segment may contain zero, one, or multiple techniques.\n\n"
            "Be precise: only flag a technique when there is clear evidence in the text. "
            "Neutral political speech without manipulation should return an empty list."
        )

    def build_prompt(self, segment: SpeechSegment) -> str:
        """Build the full user prompt for a given segment.

        Args:
            segment: The speech segment to classify.

        Returns:
            Complete prompt string with taxonomy, format instructions, and segment.
        """
        prompt = (
            f"{self._taxonomy_section}\n\n"
            "---\n\n"
            "## Output Format\n\n"
            "Respond with a JSON array of detected techniques. For each detection, include:\n"
            "- \"technique\": the technique name (exactly as listed above)\n"
            "- \"span\": the exact text span that contains the technique\n"
            "- \"confidence\": your confidence score (0.0 to 1.0)\n\n"
            "If no techniques are detected, respond with an empty array: []\n\n"
            "Example output:\n"
            "```json\n"
            "[\n"
            "  {\"technique\": \"Loaded Language\", \"span\": \"política assassina de empregos\", \"confidence\": 0.9},\n"
            "  {\"technique\": \"Appeal to Fear\", \"span\": \"o Brasil vai virar uma Venezuela\", \"confidence\": 0.85}\n"
            "]\n"
            "```\n\n"
            "---\n\n"
            "## Segment to Analyze\n\n"
            f"```\n{segment.text}\n```\n\n"
            "Analyze the segment above. Respond ONLY with the JSON array."
        )
        return prompt

    def _build_taxonomy_section(self) -> str:
        """Build the taxonomy definitions section of the prompt."""
        lines = ["## Manipulation Techniques Taxonomy\n"]

        for technique in self._taxonomy.get_all_techniques():
            lines.append(f"### {technique.name}")
            lines.append(f"**Definition**: {technique.definition}\n")
            lines.append("**Positive examples** (technique IS present):")
            for ex in technique.positive_examples:
                lines.append(f"  - \"{ex}\"")
            lines.append("\n**Negative examples** (technique is NOT present):")
            for ex in technique.negative_examples:
                lines.append(f"  - \"{ex}\"")
            lines.append("")

        return "\n".join(lines)
