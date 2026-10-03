from __future__ import annotations

import argparse
import re
from pathlib import Path

from document.document_loader import load_document
from preprocessing.cleaner import clean_text
from preprocessing.sentence_splitter import split_into_sentences
from claim_detection.claim_detector import identify_claims
from ner.entity_extractor import extract_entities
from retrieval.hybrid_retriever import retrieve_evidence
from verification.verifier import verify_claim
from output.report_generator import generate_report


_COVERAGE_ACTIONS = {
    "cover", "covers", "covered", "covering", "coverage", "include", "includes",
    "included", "provide", "provides", "provided", "offer", "offers", "benefit",
    "benefits", "pay", "pays",
}
_QUESTION_FILLER = {
    "a", "an", "the", "insurance", "insurer", "policy", "plan", "does", "do",
    "did", "is", "are", "can", "could", "would", "will", "should", "it", "this",
    "that", "my", "me", "we", "you", "i", "not", "no", "for", "to", "of", "and",
    "or", "please", "s",
}
_QUESTION_STARTERS = {"does", "do", "did", "is", "are", "can", "could", "would", "will", "should"}


def _extract_coverage_need(request: str) -> str:
    """Reduce a natural-language coverage question to its benefit phrase."""
    words = re.findall(r"[a-z0-9]+", request.lower())
    if not words:
        return ""

    action_index = next(
        (index for index, word in enumerate(words) if word in _COVERAGE_ACTIONS),
        None,
    )
    if action_index is not None:
        words = words[action_index + 1:]
    elif words[0] in _QUESTION_STARTERS:
        return ""

    while words and words[-1] in {"not", "please"}:
        words.pop()
    need_words = [word for word in words if word not in _QUESTION_FILLER]
    return " ".join(need_words)


def _is_negative_coverage_question(request: str) -> bool:
    """Return whether the user negated the coverage action in their question."""
    words = re.findall(r"[a-z0-9]+", request.lower())
    action_index = next(
        (index for index, word in enumerate(words) if word in _COVERAGE_ACTIONS),
        None,
    )
    if action_index is None:
        return False
    return any(word in {"not", "no", "never"} for word in words[max(0, action_index - 3):action_index])


def _sentence_matches_need(sentence: str, coverage_need: str) -> bool:
    """Match a sentence to the requested coverage need using phrase and token overlap."""
    normalized_need = re.sub(r"[^a-z0-9\s]", " ", coverage_need.lower())
    sentence_lower = sentence.lower()

    if normalized_need in sentence_lower:
        return True

    need_tokens = set(normalized_need.split())
    sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence_lower))
    return bool(need_tokens & sentence_tokens)


def evaluate_policy_match(coverage_need: str, policy_document: str | Path) -> str:
    """Determine whether a policy file matches a requested coverage need."""
    negative_question = _is_negative_coverage_question(coverage_need)
    coverage_need = _extract_coverage_need(coverage_need)
    if not coverage_need:
        return "This policy does not cover that."

    document_path = Path(policy_document)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    raw_text = load_document(document_path)
    cleaned_text = clean_text(raw_text)
    sentences = split_into_sentences(cleaned_text)
    claim = f"The policy covers {coverage_need}."
    entities = extract_entities(claim)
    evidence = retrieve_evidence(
        coverage_need,
        entities,
        top_k=10,
        document_sentences=sentences,
    )
    verdict = verify_claim(claim, evidence, entities)

    if verdict == "SUPPORTED":
        if negative_question:
            return "This policy covers that."
        return "This document matches your needs"

    matched_sentences = [
        sentence for sentence in sentences
        if _sentence_matches_need(sentence, coverage_need)
    ]
    if matched_sentences:
        sentence_text = " ".join(matched_sentences).lower()
        exclusion_patterns = [
            "does not cover",
            "not covered",
            "excludes",
            "excluded",
            "no coverage for",
            "not included",
        ]
        if any(pattern in sentence_text for pattern in exclusion_patterns):
            return "This policy does not cover that."

    return "This policy does not cover that." if verdict != "SUPPORTED" else "This document matches your needs"


def run_document_pipeline(document_path: Path) -> None:
    """Run the full NLP pipeline on the policy document."""
    if not document_path.exists():
        print(f"Document not found: {document_path}")
        print("Place 'the_document.pdf' inside the 'document' directory or provide a document path.")
        return

    raw_text = load_document(document_path)
    cleaned_text = clean_text(raw_text)
    sentences = split_into_sentences(cleaned_text)
    claims = identify_claims(sentences)

    results = []
    for claim in claims:
        entities = extract_entities(claim)
        evidence = retrieve_evidence(claim, entities, document_sentences=sentences)
        verdict = verify_claim(claim, evidence, entities)
        results.append({
            "claim": claim,
            "evidence": evidence,
            "verdict": verdict,
            "entities": entities,
        })

    generate_report(results, document_path.name)


def main(argv: list[str] | None = None) -> None:
    """Run the project either as a policy matcher or the document verification pipeline."""
    parser = argparse.ArgumentParser(description="Policy document claim verification")
    parser.add_argument("--coverage-need", help="Coverage the user is looking for in a policy")
    parser.add_argument(
        "--policy-document",
        default="document/the_document.pdf",
        help="Path to the policy document to inspect",
    )
    args = parser.parse_args(argv)

    if args.coverage_need:
        print(evaluate_policy_match(args.coverage_need, args.policy_document))
        return

    coverage_need = input("Enter the coverage need: ").strip()
    if not coverage_need:
        run_document_pipeline(Path(args.policy_document))
        return

    print(evaluate_policy_match(coverage_need, args.policy_document))


if __name__ == "__main__":
    main()
