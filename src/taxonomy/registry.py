from pathlib import Path

import yaml

from src.exceptions import TaxonomyError
from src.schemas.prediction import TechniqueDefinition


class TaxonomyRegistry:
    """Loads taxonomy from YAML, validates, and provides formatted output for prompts."""

    EXPECTED_TECHNIQUES = 6

    def __init__(self, taxonomy_path: Path = Path("configs/taxonomy.yaml")) -> None:
        self._techniques: dict[str, TechniqueDefinition] = {}
        self._load(taxonomy_path)

    def _load(self, path: Path) -> None:
        """Load and validate taxonomy YAML.

        Raises TaxonomyError if:
        - File not found
        - YAML is malformed
        - Technique count != 6
        - Any technique fails TechniqueDefinition validation
        """
        if not path.exists():
            raise TaxonomyError(
                f"Taxonomy file not found: {path}",
                context={"path": str(path)},
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise TaxonomyError(
                f"Invalid YAML in taxonomy file: {e}",
                context={"path": str(path), "error": str(e)},
            ) from e

        if not isinstance(raw, dict) or "techniques" not in raw:
            raise TaxonomyError(
                "Taxonomy file must contain a 'techniques' key",
                context={"path": str(path)},
            )

        techniques_data = raw["techniques"]
        if not isinstance(techniques_data, list):
            raise TaxonomyError(
                "'techniques' must be a list",
                context={"path": str(path)},
            )

        if len(techniques_data) != self.EXPECTED_TECHNIQUES:
            raise TaxonomyError(
                f"Expected {self.EXPECTED_TECHNIQUES} techniques, got {len(techniques_data)}",
                context={"path": str(path), "count": len(techniques_data)},
            )

        for i, technique_data in enumerate(techniques_data):
            try:
                technique = TechniqueDefinition.model_validate(technique_data)
                self._techniques[technique.name] = technique
            except Exception as e:
                raise TaxonomyError(
                    f"Invalid technique definition at index {i}: {e}",
                    context={"index": i, "data": technique_data},
                ) from e

    def get_technique(self, name: str) -> TechniqueDefinition:
        """Retrieve a single technique by name.

        Raises TaxonomyError if technique not found.
        """
        if name not in self._techniques:
            available = list(self._techniques.keys())
            raise TaxonomyError(
                f"Technique '{name}' not found. Available: {available}",
                context={"requested": name, "available": available},
            )
        return self._techniques[name]

    def get_all_techniques(self) -> list[TechniqueDefinition]:
        """Return all 6 technique definitions."""
        return list(self._techniques.values())

    def format_for_prompt(self) -> str:
        """Format all techniques for inclusion in classifier prompts.

        Returns a structured string with technique definitions and examples
        suitable for few-shot prompt construction.
        """
        sections: list[str] = []
        for technique in self._techniques.values():
            section = f"## {technique.name}\n"
            section += f"Definition: {technique.definition}\n"
            section += "Positive examples:\n"
            for ex in technique.positive_examples:
                section += f"  - {ex}\n"
            section += "Negative examples:\n"
            for ex in technique.negative_examples:
                section += f"  - {ex}\n"
            sections.append(section)
        return "\n".join(sections)

    @property
    def technique_names(self) -> list[str]:
        """Return list of all technique names."""
        return list(self._techniques.keys())
