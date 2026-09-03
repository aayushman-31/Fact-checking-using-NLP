from __future__ import annotations

from retrieval.tfidf_retriever import TfidfRetriever


EVIDENCE_DB = [
    "Albert Einstein was born in Ulm, in the German Empire.",
    "Paris is the capital city of France.",
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is in China.",
    "The Earth revolves around the Sun.",
    "William Shakespeare wrote Hamlet.",
    "The Earth is a planet in the Solar System.",
    "France is in Europe.",
    "The capital of England is London.",
    "The Pacific Ocean is the largest ocean on Earth.",
]


def retrieve_evidence(claim: str, entities: list[dict] | None = None, top_k: int = 3) -> list[str]:
    """Return a best-effort ranked list of evidence using TF-IDF retrieval."""
    retriever = TfidfRetriever(EVIDENCE_DB)
    results = retriever.retrieve(claim, top_k=top_k)
    return [text for text, _score in results]
