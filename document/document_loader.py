from __future__ import annotations

from pathlib import Path


def load_document(path: str | Path) -> str:
    """Load text from a document file."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
