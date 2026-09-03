from __future__ import annotations

from scripts.train_ordinary_fbg_px6d_models import (
    _contact_rates_from_confusion,
    _select_best,
)


def _contact_candidate(
    model_id: str,
    *,
    false_positive_rate: float,
    worst_false_positive_rate: float,
    recall: float,
    worst_recall: float,
    macro_f1: float,
) -> dict[str, object]:
    return {
        "task": "contact",
        "model_id": model_id,
        "no_contact_false_positive_rate": false_positive_rate,
        "worst_fold_no_contact_false_positive_rate": (
            worst_false_positive_rate
        ),
        "active_contact_recall": recall,
        "worst_fold_active_contact_recall": worst_recall,
        "macro_f1": macro_f1,
        "worst_fold_macro_f1": macro_f1 - 0.02,
        "inference_latency_ms_per_frame": 0.1,
    }


def test_contact_rates_are_derived_from_binary_confusion_matrix() -> None:
    false_positive_rate, recall = _contact_rates_from_confusion(
        [[90, 10], [5, 95]]
    )

    assert false_positive_rate == 0.10
    assert recall == 0.95


def test_contact_selection_minimizes_worst_fold_false_activation() -> None:
    high_average_f1 = _contact_candidate(
        "high_average_f1",
        false_positive_rate=0.01,
        worst_false_positive_rate=0.08,
        recall=0.98,
        worst_recall=0.90,
        macro_f1=0.99,
    )
    robust = _contact_candidate(
        "robust",
        false_positive_rate=0.015,
        worst_false_positive_rate=0.02,
        recall=0.97,
        worst_recall=0.89,
        macro_f1=0.97,
    )

    selected = _select_best([high_average_f1, robust], "contact")

    assert selected["model_id"] == "robust"


def test_contact_selection_rejects_low_recall_degenerate_candidate() -> None:
    no_false_positives_but_low_recall = _contact_candidate(
        "degenerate",
        false_positive_rate=0.0,
        worst_false_positive_rate=0.0,
        recall=0.70,
        worst_recall=0.55,
        macro_f1=0.75,
    )
    usable = _contact_candidate(
        "usable",
        false_positive_rate=0.02,
        worst_false_positive_rate=0.03,
        recall=0.96,
        worst_recall=0.85,
        macro_f1=0.95,
    )

    selected = _select_best(
        [no_false_positives_but_low_recall, usable], "contact"
    )

    assert selected["model_id"] == "usable"


def test_position_selection_prefers_stronger_worst_batch() -> None:
    brittle = {
        "task": "position",
        "model_id": "brittle",
        "worst_fold_macro_f1": 0.60,
        "worst_fold_accuracy": 0.65,
        "macro_f1": 0.97,
        "session_majority_voting": {"accuracy": 0.98},
        "inference_latency_ms_per_frame": 0.05,
    }
    robust = {
        "task": "position",
        "model_id": "robust",
        "worst_fold_macro_f1": 0.85,
        "worst_fold_accuracy": 0.88,
        "macro_f1": 0.94,
        "session_majority_voting": {"accuracy": 0.95},
        "inference_latency_ms_per_frame": 0.08,
    }

    selected = _select_best([brittle, robust], "position")

    assert selected["model_id"] == "robust"
