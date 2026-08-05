from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.hybrid_spectrum.optical_candidate_benchmark import (
    CandidateSpec,
    evaluate_hierarchical_predictions,
    grouped_candidate_classification,
)


def test_grouped_candidate_predictions_cover_formal_rows_without_leakage() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.asarray(["a", "b", "c", "d"]), 6)
    folds = np.repeat(np.arange(4), 6)
    target = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), 4)
    features = np.column_stack(
        (target + rng.normal(0.0, 0.03, len(target)), rng.normal(size=len(target)))
    ).astype(np.float32)
    spec = CandidateSpec(
        model_id="logistic_test",
        model_family="linear",
        confidence_source="test",
        factory=lambda seed: LogisticRegression(class_weight="balanced"),
        sample_weight_parameter="sample_weight",
    )

    metrics, predicted = grouped_candidate_classification(
        spec=spec,
        features=features,
        target=target,
        training_and_evaluation_mask=np.ones(len(target), dtype=bool),
        fold_id=folds,
        group_id=groups,
        labels=[0, 1],
        seed=5,
    )

    assert metrics["session_count"] == 4
    assert metrics["accuracy"] > 0.95
    assert all(value is not None for value in predicted)


def test_position_head_can_predict_all_fold_rows() -> None:
    features = np.asarray(
        [[0.0], [0.1], [1.0], [1.1], [0.0], [0.1], [1.0], [1.1]],
        dtype=np.float32,
    )
    target = np.asarray(["P11", "P11", "P22", "P22"] * 2)
    groups = np.repeat(np.asarray(["a", "b", "c", "d"]), 2)
    folds = np.repeat(np.arange(4), 2)
    mask = np.asarray([False, True] * 4)
    spec = CandidateSpec(
        model_id="logistic_test",
        model_family="linear",
        confidence_source="test",
        factory=lambda seed: LogisticRegression(),
        sample_weight_parameter="sample_weight",
    )

    _, predicted = grouped_candidate_classification(
        spec=spec,
        features=features,
        target=target,
        training_and_evaluation_mask=mask,
        fold_id=folds,
        group_id=groups,
        labels=["P11", "P22"],
        seed=5,
        predict_all_fold_rows=True,
    )

    assert all(value is not None for value in predicted)


def test_hierarchical_metrics_penalize_false_and_missed_contact() -> None:
    metrics = evaluate_hierarchical_predictions(
        contact_prediction=np.asarray([0, 1, 0, 1]),
        position_prediction=np.asarray(["P11", "P11", "P22", "P22"]),
        contact_target=np.asarray([0, 0, 1, 1]),
        position_target=np.asarray(["", "", "P22", "P22"]),
        eligible_mask=np.ones(4, dtype=bool),
        labels=["no_contact", "P11", "P22"],
    )

    assert metrics["false_contact_rate"] == pytest.approx(0.5)
    assert metrics["missed_contact_rate"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_hierarchy_ignores_unset_predictions_outside_eligible_mask() -> None:
    metrics = evaluate_hierarchical_predictions(
        contact_prediction=np.asarray([0, 1, None], dtype=object),
        position_prediction=np.asarray(["P11", "P22", None], dtype=object),
        contact_target=np.asarray([0, 1, -1]),
        position_target=np.asarray(["", "P22", ""]),
        eligible_mask=np.asarray([True, True, False]),
        labels=["no_contact", "P22"],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
