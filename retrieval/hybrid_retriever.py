from __future__ import annotations

from retrieval.tfidf_retriever import TfidfRetriever
from verification.verifier import expand_topic_terms


def retrieve_evidence(
    claim: str,
    entities: list[dict] | None = None,
    top_k: int = 5,
    document_sentences: list[str] | None = None,
) -> list[str]:
    """Return evidence from the document sentences using TF-IDF similarity and optional NER filtering."""
    evidence_corpus = document_sentences or []

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
    expanded_query = " ".join(part for part in [claim, expand_topic_terms(claim)] if part)
    results = retriever.retrieve(expanded_query, top_k=top_k)
    return [text for text, _score in results]
