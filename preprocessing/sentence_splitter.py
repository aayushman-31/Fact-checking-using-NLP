from __future__ import annotations

import re


def split_into_sentences(text: str) -> list[str]:
    """Split a document into sentence-like units."""
    cleaned = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
