from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_jsonl(items: list[BaseModel], path: Path) -> None:
    """Save a list of Pydantic models as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def load_jsonl(path: Path, schema: type[T]) -> list[T]:
    """Load a JSONL file into a list of validated Pydantic models."""
    items: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            items.append(schema.model_validate_json(stripped))
    return items


def save_json(item: BaseModel, path: Path) -> None:
    """Save a single Pydantic model as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(item.model_dump_json(indent=2), encoding="utf-8")


def load_json(path: Path, schema: type[T]) -> T:
    """Load a single JSON file into a validated Pydantic model."""
    content = path.read_text(encoding="utf-8")
    return schema.model_validate_json(content)
