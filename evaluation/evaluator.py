from __future__ import annotations

from collections import Counter


def compute_accuracy(predictions: list[str], gold: list[str]) -> float:
    """Compute overall accuracy for a set of predictions."""
    if len(predictions) != len(gold):
        raise ValueError("Predictions and gold labels must have the same length.")
    if not predictions:
        return 0.0
    matches = sum(1 for pred, true in zip(predictions, gold) if pred == true)
    return matches / len(predictions)


def compute_classification_report(predictions: list[str], gold: list[str]) -> dict:
    """Return precision, recall, F1, and support for each label."""
    labels = sorted(set(gold) | set(predictions))
    report: dict[str, dict[str, float | int]] = {}

    for label in labels:
        tp = sum(1 for pred, true in zip(predictions, gold) if pred == label and true == label)
        fp = sum(1 for pred, true in zip(predictions, gold) if pred == label and true != label)
        fn = sum(1 for pred, true in zip(predictions, gold) if pred != label and true == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for item in gold if item == label),
        }

    return report


def summarize_predictions(results: list[dict]) -> dict:
    """Summarize a list of claim-result dictionaries."""
    verdicts = [item["verdict"] for item in results]
    counts = Counter(verdicts)
    return {
        "total": len(results),
        "supported": counts.get("SUPPORTED", 0),
        "refuted": counts.get("REFUTED", 0),
        "nei": counts.get("NEI", 0),
    }
