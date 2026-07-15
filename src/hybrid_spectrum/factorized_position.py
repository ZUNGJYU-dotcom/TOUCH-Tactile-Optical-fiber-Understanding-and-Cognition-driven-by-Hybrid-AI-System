"""Factorized 3x3 position classification for the TOUCH spectral twin."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone


POSITION_ORDER = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)
AXIS_ORDER = ("1", "2", "3")


def validate_position_labels(labels: Iterable[str]) -> np.ndarray:
    values = np.asarray(tuple(str(label) for label in labels))
    invalid = sorted(set(values.tolist()) - set(POSITION_ORDER))
    if invalid:
        raise ValueError(f"invalid 3x3 position labels: {invalid}")
    return values


def _aligned_probability(
    model: Any,
    values: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=float)
    aligned = np.zeros((len(values), len(labels)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        label_text = str(label)
        if label_text in labels:
            aligned[:, labels.index(label_text)] = raw[:, source_index]
    return aligned


class FactorizedPositionClassifier:
    """Predict 3x3 positions from independent column and row evidence.

    ``Pxy`` follows the physical layout used by the project: ``x`` is the
    left/centre/right fibre column and ``y`` is the top/middle/bottom row.
    The final nine-position probability is the normalized product of the two
    axis probabilities. This preserves the configured array orientation while
    allowing weak P21 evidence to share statistical strength with its row and
    column neighbours.
    """

    def __init__(self, row_estimator: Any, column_estimator: Any | None = None) -> None:
        self.row_estimator = row_estimator
        self.column_estimator = (
            column_estimator if column_estimator is not None else deepcopy(row_estimator)
        )
        self.classes_ = np.asarray(POSITION_ORDER)

    def fit(self, values: np.ndarray, labels: Iterable[str]) -> "FactorizedPositionClassifier":
        array = np.asarray(values)
        targets = validate_position_labels(labels)
        if array.ndim != 2 or len(array) != len(targets):
            raise ValueError("position features must have shape [samples, features]")
        rows = np.asarray([label[2] for label in targets])
        columns = np.asarray([label[1] for label in targets])
        self.row_model_ = clone(self.row_estimator).fit(array, rows)
        self.column_model_ = clone(self.column_estimator).fit(array, columns)
        self.n_features_in_ = int(array.shape[1])
        return self

    def predict_axis_proba(self, values: np.ndarray) -> dict[str, np.ndarray]:
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("position features must have shape [samples, features]")
        return {
            "row": _aligned_probability(self.row_model_, array, AXIS_ORDER),
            "column": _aligned_probability(self.column_model_, array, AXIS_ORDER),
        }

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        axis = self.predict_axis_proba(values)
        joint = np.column_stack(
            [
                axis["row"][:, int(label[2]) - 1]
                * axis["column"][:, int(label[1]) - 1]
                for label in POSITION_ORDER
            ]
        )
        denominator = np.maximum(np.sum(joint, axis=1, keepdims=True), 1.0e-12)
        return joint / denominator

    def predict(self, values: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(values)
        return self.classes_[np.argmax(probability, axis=1)]

    def set_runtime_n_jobs(self, n_jobs: int = 1) -> None:
        """Limit nested tree workers for low-latency single-window inference."""

        for model in (getattr(self, "row_model_", None), getattr(self, "column_model_", None)):
            if model is not None and hasattr(model, "n_jobs"):
                model.n_jobs = int(n_jobs)


__all__ = [
    "AXIS_ORDER",
    "POSITION_ORDER",
    "FactorizedPositionClassifier",
    "validate_position_labels",
]
