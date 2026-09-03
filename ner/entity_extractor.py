from __future__ import annotations

import os

import spacy


SPACY_MODEL = "en_core_web_sm"


def _ensure_model() -> None:
    """Download a spaCy model if it is not already installed."""
    try:
        spacy.load(SPACY_MODEL)
    except OSError:
        os.system(f"python -m spacy download {SPACY_MODEL}")


_ensure_model()


nlp = spacy.load(SPACY_MODEL)


def extract_entities(text: str) -> list[dict]:
    """Return entities extracted from a claim in a simple dictionary format."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
        })
    return entities
