Install dependencies and the spaCy English model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place a text-based insurance policy PDF at `document/the_document.pdf`, then run:

```bash
python main.py
```

Enter the coverage need when prompted. To provide a different PDF or a UTF-8 text file:

```bash
python main.py --coverage-need "emergency room visits" --policy-document "path/to/policy.pdf"
```

PDFs must contain selectable text. Scanned image-only PDFs need OCR before they can be checked. UTF-8 `.txt` documents remain supported.
