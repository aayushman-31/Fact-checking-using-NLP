from __future__ import annotations

from pathlib import Path

import fitz


def load_document(path: str | Path) -> str:
    """Load text from a PDF or UTF-8 text document."""
    document_path = Path(path)
    if document_path.suffix.lower() == ".pdf":
        with fitz.open(document_path) as document:
            text = "\n".join(page.get_text() for page in document)
        if not text.strip():
            raise ValueError(
                f"No selectable text found in PDF: {document_path}. "
                "Scanned PDFs require OCR before they can be checked."
            )
        return text

    with document_path.open("r", encoding="utf-8") as file:
        return file.read()
