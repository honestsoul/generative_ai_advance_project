"""
Evaluation metrics for LLM outputs.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    metric: str
    score: float
    details: dict[str, Any] | None = None


def exact_match(prediction: str, reference: str) -> EvalResult:
    """Check for exact string match."""
    score = 1.0 if prediction.strip() == reference.strip() else 0.0
    return EvalResult(metric="exact_match", score=score)


def contains_match(prediction: str, reference: str) -> EvalResult:
    """Check if reference is contained in prediction."""
    score = 1.0 if reference.strip().lower() in prediction.lower() else 0.0
    return EvalResult(metric="contains_match", score=score)


def word_overlap(prediction: str, reference: str) -> EvalResult:
    """Calculate word overlap ratio (Jaccard similarity)."""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())

    if not pred_words or not ref_words:
        return EvalResult(metric="word_overlap", score=0.0)

    intersection = pred_words & ref_words
    union = pred_words | ref_words

    score = len(intersection) / len(union)

    return EvalResult(
        metric="word_overlap",
        score=score,
        details={
            "intersection_size": len(intersection),
            "union_size": len(union),
        },
    )


def length_ratio(prediction: str, reference: str) -> EvalResult:
    """Calculate length ratio (prediction / reference)."""
    pred_len = len(prediction)
    ref_len = len(reference)

    if ref_len == 0:
        return EvalResult(metric="length_ratio", score=0.0)

    ratio = pred_len / ref_len

    return EvalResult(
        metric="length_ratio",
        score=ratio,
        details={
            "prediction_length": pred_len,
            "reference_length": ref_len,
        },
    )


def evaluate_all(
    prediction: str,
    reference: str,
) -> dict[str, EvalResult]:
    """Run all evaluation metrics."""
    return {
        "exact_match": exact_match(prediction, reference),
        "contains_match": contains_match(prediction, reference),
        "word_overlap": word_overlap(prediction, reference),
        "length_ratio": length_ratio(prediction, reference),
    }
