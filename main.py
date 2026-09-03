from __future__ import annotations

from pathlib import Path

from document.document_loader import load_document
from preprocessing.cleaner import clean_text
from preprocessing.sentence_splitter import split_into_sentences
from claim_detection.claim_detector import identify_claims
from ner.entity_extractor import extract_entities
from retrieval.hybrid_retriever import retrieve_evidence
from verification.verifier import verify_claim
from output.report_generator import generate_report


def main() -> None:
    """Run the full TruthLens pipeline on the input document."""
    document_path = Path("document/the_document.txt")

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
        evidence = retrieve_evidence(claim, entities)
        verdict = verify_claim(claim, evidence, entities)
        results.append({
            "claim": claim,
            "evidence": evidence,
            "verdict": verdict,
            "entities": entities,
        })

    generate_report(results, document_path.name)


if __name__ == "__main__":
    main()
