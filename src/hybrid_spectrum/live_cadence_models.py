"""Low-latency model helpers for live-cadence spectral classification."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CumulativeOrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Ordinal classifier built from cumulative binary logistic thresholds.

    For ordered classes ``light < normal < hard``, the model estimates
    ``P(y > light)`` and ``P(y > normal)``.  The cumulative probabilities are
    projected to a monotone sequence before conversion to class probabilities.
    """

    def __init__(
        self,
        class_order: Iterable[str] = ("light", "normal", "hard"),
        *,
        C: float = 0.5,
        max_iter: int = 3000,
        random_state: int | None = 42,
    ) -> None:
        self.class_order = tuple(class_order)
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CumulativeOrdinalClassifier":
        values = np.asarray(X)
        labels = np.asarray(y).astype(str)
        if len(self.class_order) < 2:
            raise ValueError("ordinal classification requires at least two classes")
        unknown = sorted(set(labels.tolist()) - set(self.class_order))
        if unknown:
            raise ValueError(f"labels are outside class_order: {unknown}")
        self.classes_ = np.asarray(self.class_order)
        class_index = {label: index for index, label in enumerate(self.class_order)}
        encoded = np.asarray([class_index[label] for label in labels], dtype=np.int32)
        self.threshold_models_: list[Pipeline] = []
        for threshold in range(len(self.class_order) - 1):
            target = (encoded > threshold).astype(np.int32)
            if len(np.unique(target)) != 2:
                raise ValueError("each ordinal threshold needs both binary classes")
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=float(self.C),
                            max_iter=int(self.max_iter),
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            model.fit(values, target)
            self.threshold_models_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "threshold_models_"):
            raise RuntimeError("ordinal classifier is not fitted")
        values = np.asarray(X)
        cumulative = np.column_stack(
            [model.predict_proba(values)[:, 1] for model in self.threshold_models_]
        )
        cumulative = np.minimum.accumulate(cumulative, axis=1)
        probabilities = np.empty(
            (len(values), len(self.class_order)), dtype=np.float64
        )
        probabilities[:, 0] = 1.0 - cumulative[:, 0]
        for index in range(1, len(self.class_order) - 1):
            probabilities[:, index] = cumulative[:, index - 1] - cumulative[:, index]
        probabilities[:, -1] = cumulative[:, -1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        total = np.sum(probabilities, axis=1, keepdims=True)
        return probabilities / np.maximum(total, 1.0e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class OrdinalRegressionClassifier(ClassifierMixin, BaseEstimator):
    """Fit an ordinary regressor to ordered class indices."""

    def __init__(
        self,
        regressor: Any,
        class_order: Iterable[str] = ("light", "normal", "hard"),
        *,
        probability_temperature: float = 0.40,
    ) -> None:
        self.regressor = regressor
        self.class_order = tuple(class_order)
        self.probability_temperature = probability_temperature

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalRegressionClassifier":
        labels = np.asarray(y).astype(str)
        unknown = sorted(set(labels.tolist()) - set(self.class_order))
        if unknown:
            raise ValueError(f"labels are outside class_order: {unknown}")
        mapping = {label: index for index, label in enumerate(self.class_order)}
        target = np.asarray([mapping[label] for label in labels], dtype=np.float32)
        self.regressor_ = clone(self.regressor).fit(np.asarray(X), target)
        self.classes_ = np.asarray(self.class_order)
        return self

    def predict_continuous(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("ordinal regressor is not fitted")
        return np.clip(
            np.asarray(self.regressor_.predict(np.asarray(X)), dtype=float),
            0.0,
            len(self.class_order) - 1,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prediction = self.predict_continuous(X)[:, None]
        centers = np.arange(len(self.class_order), dtype=float)[None, :]
        temperature = max(float(self.probability_temperature), 1.0e-6)
        score = np.exp(-0.5 * ((prediction - centers) / temperature) ** 2)
        return score / np.maximum(np.sum(score, axis=1, keepdims=True), 1.0e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        index = np.rint(self.predict_continuous(X)).astype(int)
        return self.classes_[index]


class SpatialCoordinateRegressorClassifier(ClassifierMixin, BaseEstimator):
    """Predict Pxy through independent continuous x/y coordinate regressors."""

    def __init__(self, regressor: Any, *, probability_scale: float = 0.55) -> None:
        self.regressor = regressor
        self.probability_scale = probability_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpatialCoordinateRegressorClassifier":
        labels = np.asarray(y).astype(str)
        invalid = sorted(
            label
            for label in set(labels.tolist())
            if len(label) != 3
            or not label.startswith("P")
            or label[1] not in "123"
            or label[2] not in "123"
        )
        if invalid:
            raise ValueError(f"invalid Pxy labels: {invalid}")
        values = np.asarray(X)
        x_target = np.asarray([int(label[1]) for label in labels], dtype=np.float32)
        y_target = np.asarray([int(label[2]) for label in labels], dtype=np.float32)
        self.x_regressor_ = clone(self.regressor).fit(values, x_target)
        self.y_regressor_ = clone(self.regressor).fit(values, y_target)
        self.classes_ = np.asarray(
            ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
        )
        return self

    def predict_coordinates(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_regressor_"):
            raise RuntimeError("spatial coordinate model is not fitted")
        values = np.asarray(X)
        x = np.clip(self.x_regressor_.predict(values), 1.0, 3.0)
        y = np.clip(self.y_regressor_.predict(values), 1.0, 3.0)
        return np.column_stack([x, y])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        coordinates = self.predict_coordinates(X)
        centers = np.asarray(
            [[int(label[1]), int(label[2])] for label in self.classes_], dtype=float
        )
        squared_distance = np.sum(
            (coordinates[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        scale = max(float(self.probability_scale), 1.0e-6)
        score = np.exp(-0.5 * squared_distance / (scale**2))
        return score / np.maximum(np.sum(score, axis=1, keepdims=True), 1.0e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(X)
        return self.classes_[np.argmax(probability, axis=1)]


def standardize_sequence_channels(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Scale each frame-feature channel using training data only."""

    train_channels = np.asarray(train, dtype=np.float32).transpose(0, 2, 1)
    mean = np.mean(train_channels, axis=(0, 2), keepdims=True)
    scale = np.std(train_channels, axis=(0, 2), keepdims=True)
    scale = np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)

    def transform(values: np.ndarray) -> np.ndarray:
        channels = np.asarray(values, dtype=np.float32).transpose(0, 2, 1)
        return ((channels - mean) / scale).astype(np.float32)

    return tuple(transform(values) for values in (train, *others))


def set_single_thread_prediction(model: Any) -> Any:
    """Avoid thread-pool startup overhead for one-window desktop inference."""

    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except (AttributeError, TypeError):
            pass
    if hasattr(model, "steps"):
        for _, step in model.steps:
            set_single_thread_prediction(step)
    for attribute in (
        "row_model_",
        "column_model_",
        "estimator_",
        "regressor_",
        "x_regressor_",
        "y_regressor_",
    ):
        child = getattr(model, attribute, None)
        if child is not None:
            set_single_thread_prediction(child)
    return model


__all__ = [
    "CumulativeOrdinalClassifier",
    "OrdinalRegressionClassifier",
    "SpatialCoordinateRegressorClassifier",
    "set_single_thread_prediction",
    "standardize_sequence_channels",
]
