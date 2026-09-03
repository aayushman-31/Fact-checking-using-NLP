from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Retrieve evidence using TF-IDF similarity."""

    def __init__(self, evidence: list[str]):
        self.evidence = evidence
        if not evidence:
            self.vectorizer = TfidfVectorizer()
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(evidence)

    def retrieve(self, claim: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top relevant evidence sentences with cosine similarity scores."""
        if not self.evidence or self.matrix is None:
            return []

        claim_vector = self.vectorizer.transform([claim])
        scores = cosine_similarity(claim_vector, self.matrix).flatten()
        ranked = sorted(
            [(text, float(score)) for text, score in zip(self.evidence, scores)],
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
