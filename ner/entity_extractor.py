from __future__ import annotations

import re


SPACY_MODEL = "en_core_web_sm"

try:
    import spacy

    nlp = spacy.load(SPACY_MODEL)
except (ImportError, OSError):
    nlp = None


def _extract_entities_with_rules(text: str) -> list[dict]:
    """Extract simple proper-name phrases when spaCy is unavailable."""
    ignored = {"a", "an", "the", "this", "that", "policy"}
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return [
        {"text": match, "label": "PROPER_NOUN"}
        for match in dict.fromkeys(matches)
        if match.lower() not in ignored
    ]


def extract_entities(text: str) -> list[dict]:
    """Return entities using spaCy when available, otherwise use simple NLP rules."""
    if nlp is None:
        return _extract_entities_with_rules(text)

    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]
