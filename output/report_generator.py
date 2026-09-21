from __future__ import annotations

from evaluation.evaluator import summarize_predictions


def generate_report(results: list[dict], document_name: str = "the_document") -> None:
    """Print a simple claim-level and document-level insurance policy report."""
    summary = summarize_predictions(results)

    print("\nINSURANCE POLICY VERIFICATION REPORT")
    print(f"Document: {document_name}")
    print(f"Claims analyzed: {summary['total']}")
    print(f"Supported: {summary['supported']}")
    print(f"Refuted: {summary['refuted']}")
    print(f"Not Enough Information: {summary['nei']}")

    print("\nClaim-level results:")
    for idx, item in enumerate(results, start=1):
        evidence = ", ".join(item.get("evidence", [])) if item.get("evidence") else "No evidence found"
        print(f"{idx}. Claim: {item['claim']}")
        print(f"   Verdict: {item['verdict']}")
        print(f"   Evidence: {evidence}")
        print(f"   Entities: {item.get('entities', [])}\n")
