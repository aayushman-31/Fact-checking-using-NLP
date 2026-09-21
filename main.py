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
    document_path = Path(policy_document)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    raw_text = load_document(document_path)
    cleaned_text = clean_text(raw_text)
    sentences = split_into_sentences(cleaned_text)
    claim = f"The policy covers {coverage_need.strip()}."
    entities = extract_entities(claim)
    evidence = retrieve_evidence(claim, entities, document_sentences=sentences)
    verdict = verify_claim(claim, evidence, entities)

    if verdict == "SUPPORTED":
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
        print("Please create a file named 'the_document.txt' inside the 'document' directory.")
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
        default="document/the_document.txt",
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
