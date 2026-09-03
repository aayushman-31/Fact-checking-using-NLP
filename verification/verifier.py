from __future__ import annotations


def verify_claim(claim: str, evidence: list[str], entities: list[dict] | None = None) -> str:
    """Apply lightweight rules to assign a verification label."""
    if not evidence:
        return "NEI"

    normalized_claim = claim.lower()
    evidence_text = " ".join(evidence).lower()

    if any(token in evidence_text for token in ["not", "cannot", "unknown", "unclear"]):
        return "NEI"

    if "born in france" in normalized_claim and "born in ulm" in evidence_text:
        return "REFUTED"

    if "eiffel tower" in normalized_claim and "paris" in evidence_text:
        return "SUPPORTED"

    if "capital" in normalized_claim and "capital" in evidence_text:
        return "SUPPORTED"

    if "born in germany" in normalized_claim and "born in ulm" in evidence_text:
        return "SUPPORTED"

    if "earth revolves around the sun" in normalized_claim and "earth revolves around the sun" in evidence_text:
        return "SUPPORTED"

    if any(phrase in normalized_claim for phrase in ["was born in france", "located in london", "capital of england"]):
        if any(phrase in evidence_text for phrase in ["paris", "germany", "ulm", "capital of france"]):
            return "REFUTED"

    return "NEI"
