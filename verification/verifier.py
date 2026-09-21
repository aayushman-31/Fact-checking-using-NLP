from __future__ import annotations

import re


COVERAGE_TOPICS = {
    "emergency": ["emergency", "urgent care", "er", "hospital", "accident"],
    "dental": ["dental", "teeth", "cleaning", "checkup", "orthodontic"],
    "vision": ["vision", "eye", "glasses", "contact lenses", "optical"],
    "maternity": ["maternity", "newborn", "pregnancy", "delivery"],
    "mental health": ["mental health", "therapy", "counseling", "psychology"],
    "prescription": ["prescription", "medicine", "drug", "medications"],
    "cosmetic": ["cosmetic", "elective", "plastic surgery"],
    "pre existing": ["pre existing", "pre-existing", "medical history", "prior condition"],
}

COVERAGE_VERBS = [
    "cover", "covers", "coverage", "include", "includes", "provided", "provides",
    "offer", "offers", "benefit", "benefits", "pay for", "pays for"
]
EXCLUSION_VERBS = [
    "exclude", "excludes", "excluded", "not covered", "does not cover",
    "no coverage for", "not included", "not eligible for"
]
NO_MENTION_PATTERNS = [
    "does not mention", "not mentioned", "no mention", "not specified",
    "not included in", "not covered in", "not listed"
]


def _normalize(text: str) -> str:
    """Normalize text for lexical matching in a lightweight NLP pipeline."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _extract_topics(text: str) -> set[str]:
    """Return all policy topics mentioned in the text."""
    normalized = _normalize(text)
    topics = set()
    for topic, keywords in COVERAGE_TOPICS.items():
        if any(keyword in normalized for keyword in keywords):
            topics.add(topic)
    return topics


def verify_claim(claim: str, evidence: list[str], entities: list[dict] | None = None) -> str:
    """Use rule-based NLP matching to determine whether policy evidence supports, refutes, or leaves a claim uncertain."""
    if not evidence:
        return "NEI"

    normalized_claim = _normalize(claim)
    claim_topics = _extract_topics(claim)
    claim_has_coverage_intent = any(word in normalized_claim for word in [
        "cover", "covers", "coverage", "include", "includes", "benefit", "benefits",
        "provides", "offer", "offers"
    ])

    if not claim_topics and not claim_has_coverage_intent:
        return "NEI"

    for sentence in evidence:
        sentence_text = _normalize(sentence)
        sentence_topics = _extract_topics(sentence_text)
        shared_topics = claim_topics & sentence_topics

        if not shared_topics and not claim_topics:
            continue

        if any(pattern in sentence_text for pattern in NO_MENTION_PATTERNS):
            return "NEI"

        if any(pattern in sentence_text for pattern in EXCLUSION_VERBS):
            return "REFUTED"

        if any(pattern in sentence_text for pattern in COVERAGE_VERBS):
            return "SUPPORTED"

    if claim_topics and any(_extract_topics(_normalize(sentence)) for sentence in evidence):
        return "SUPPORTED"

    return "NEI"
