from __future__ import annotations

import re


COVERAGE_TOPICS = {
    "emergency": [
        "emergency", "urgent care", "er", "hospital", "accident", "hospitalization",
        "hospitalisation", "inpatient", "emergency care", "emergency room",
    ],
    "dental": ["dental", "teeth", "cleaning", "checkup", "orthodontic"],
    "vision": ["vision", "eye", "glasses", "contact lenses", "optical"],
    "maternity": ["maternity", "newborn", "pregnancy", "delivery"],
    "mental health": ["mental health", "therapy", "counseling", "psychology"],
    "prescription": ["prescription", "medicine", "drug", "medications"],
    "cosmetic": ["cosmetic", "elective", "plastic surgery"],
    "pre existing": ["pre existing", "pre-existing", "medical history", "prior condition"],
}

COVERAGE_VERBS = [
    "cover", "covers", "covered", "coverage", "include", "includes", "included",
    "benefit", "benefits", "pay for", "pays for", "will pay", "shall pay", "payable",
    "reimburse", "reimburses", "indemnify", "provides coverage", "offers coverage",
]
EXCLUSION_VERBS = [
    "exclude", "excludes", "excluded", "not covered", "does not cover", "does not pay",
    "will not pay", "shall not pay", "not payable", "no coverage for", "not included",
    "not eligible for",
]
NO_MENTION_PATTERNS = [
    "does not mention", "not mentioned", "no mention", "not specified",
    "not included in", "not covered in", "not listed"
]
GENERIC_CLAIM_TERMS = {
    "a", "an", "the", "this", "that", "policy", "plan", "insurance", "insurer",
    "does", "do", "did", "is", "are", "can", "could", "would", "will", "should",
    "it", "my", "me", "we", "you", "i", "not", "no", "s", "covers", "cover",
    "coverage", "included", "includes", "include", "benefit", "benefits",
    "provides", "provide", "offers", "offer", "for", "with", "to", "of",
}


def _normalize(text: str) -> str:
    """Normalize text for lexical matching in a lightweight NLP pipeline."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _extract_topics(text: str) -> set[str]:
    """Return all policy topics mentioned in the text."""
    normalized = _normalize(text)
    topics = set()
    for topic, keywords in COVERAGE_TOPICS.items():
        if any(_contains_phrase(normalized, keyword) for keyword in keywords):
            topics.add(topic)
    return topics


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a whole phrase so short aliases do not match inside unrelated words."""
    normalized_phrase = _normalize(phrase).split()
    if not normalized_phrase:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in normalized_phrase) + r"\b"
    return re.search(pattern, text) is not None


def _subject_terms(text: str) -> set[str]:
    """Return meaningful non-topic terms to match coverage needs absent from the topic list."""
    return {
        token for token in _normalize(text).split()
        if token not in GENERIC_CLAIM_TERMS and len(token) > 2
    }


def _has_sufficient_subject_overlap(claim_terms: set[str], evidence_text: str) -> bool:
    """Require strong token overlap for needs outside the predefined topic vocabulary."""
    if not claim_terms:
        return False
    evidence_terms = _subject_terms(evidence_text)
    overlap = claim_terms & evidence_terms
    return len(overlap) / len(claim_terms) >= 0.75


def expand_topic_terms(text: str) -> str:
    """Add topic synonyms to a query so TF-IDF can match policy-specific terminology."""
    topics = _extract_topics(text)
    return " ".join(keyword for topic in topics for keyword in COVERAGE_TOPICS[topic])


def verify_claim_with_evidence(
    claim: str,
    evidence: list[str],
    entities: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """Return a verification label and the relevant evidence sentences used."""
    if not evidence:
        return "NEI", []

    normalized_claim = _normalize(claim)
    claim_topics = _extract_topics(claim)
    claim_has_coverage_intent = any(word in normalized_claim for word in [
        "cover", "covers", "coverage", "include", "includes", "benefit", "benefits",
        "provides", "offer", "offers"
    ])

    if not claim_topics and not claim_has_coverage_intent:
        return "NEI", []

    claim_terms = _subject_terms(claim)
    supporting_sentences = []
    refuting_sentences = []
    unknown_sentences = []

    for sentence in evidence:
        sentence_text = _normalize(sentence)
        sentence_topics = _extract_topics(sentence_text)
        shared_topics = claim_topics & sentence_topics

        if claim_topics and not shared_topics:
            continue
        if not claim_topics and not _has_sufficient_subject_overlap(claim_terms, sentence_text):
            continue

        if any(_contains_phrase(sentence_text, pattern) for pattern in NO_MENTION_PATTERNS):
            unknown_sentences.append(sentence)
            continue

        if any(_contains_phrase(sentence_text, pattern) for pattern in EXCLUSION_VERBS):
            refuting_sentences.append(sentence)
            continue

        if any(_contains_phrase(sentence_text, pattern) for pattern in COVERAGE_VERBS):
            supporting_sentences.append(sentence)

    if supporting_sentences and not refuting_sentences and not unknown_sentences:
        return "SUPPORTED", supporting_sentences
    if refuting_sentences and not supporting_sentences and not unknown_sentences:
        return "REFUTED", refuting_sentences
    if unknown_sentences:
        return "NEI", unknown_sentences
    if supporting_sentences or refuting_sentences:
        return "NEI", supporting_sentences + refuting_sentences

    return "NEI", []


def verify_claim(claim: str, evidence: list[str], entities: list[dict] | None = None) -> str:
    """Return only the verification label for existing callers."""
    verdict, _used_evidence = verify_claim_with_evidence(claim, evidence, entities)
    return verdict
