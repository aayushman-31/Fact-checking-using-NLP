from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Normalize text by lowercasing and stripping extra whitespace."""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize and normalize a sentence or document."""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    return [token for token in tokens if token not in STOP_WORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply lemmatization to tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> list[str]:
    """Run the standard preprocessing pipeline."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return lemmatize_tokens(tokens)
