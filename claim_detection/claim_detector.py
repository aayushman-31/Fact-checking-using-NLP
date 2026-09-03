from __future__ import annotations

import re


def identify_claims(sentences: list[str]) -> list[str]:
    """Filter sentence list to likely factual statements."""
    claims: list[str] = []
    for sentence in sentences:
        text = sentence.strip()
        if not text:
            continue
        if text.endswith("?") or text.endswith("!"):
            continue
        if len(text.split()) < 4:
            continue
        if re.search(r"\b(the|this|that|these|those)\b", text.lower()):
            pass
        claims.append(text)
    return claims
