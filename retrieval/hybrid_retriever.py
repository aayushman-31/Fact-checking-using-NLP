from __future__ import annotations

from retrieval.tfidf_retriever import TfidfRetriever


DEFAULT_EVIDENCE_DB = [
    "The policy covers emergency room visits and hospital stays.",
    "The policy excludes pre-existing conditions during the first 12 months.",
    "The policy includes prescription drug coverage for in-network medications.",
    "The policy summary does not mention dental care benefits.",
    "The policy provides mental health support and therapy visits.",
    "The policy does not cover cosmetic surgery procedures.",
    "The policy includes maternity care and newborn care coverage.",
    "The policy provides preventive care visits at no additional cost.",
]


def retrieve_evidence(
    claim: str,
    entities: list[dict] | None = None,
    top_k: int = 3,
    document_sentences: list[str] | None = None,
) -> list[str]:
    """Return evidence from the document sentences using TF-IDF similarity and optional NER filtering."""
    evidence_corpus = document_sentences if document_sentences else DEFAULT_EVIDENCE_DB

    if not evidence_corpus:
        return []

    if entities:
        entity_strings = {item.get("text", "").lower() for item in entities if item.get("text")}
        filtered = [
            sentence for sentence in evidence_corpus
            if not entity_strings or any(entity in sentence.lower() for entity in entity_strings)
        ]
        evidence_corpus = filtered or evidence_corpus

    retriever = TfidfRetriever(evidence_corpus)
    results = retriever.retrieve(claim, top_k=top_k)
    return [text for text, _score in results]
