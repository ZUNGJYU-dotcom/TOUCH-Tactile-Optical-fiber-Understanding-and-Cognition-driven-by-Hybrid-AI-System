from __future__ import annotations

import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.hybrid_spectrum.github_static_models import (
    AgreementAwareVotingClassifier,
    PLSDAClassifier,
)


class _ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold: float = 0.0, invert: bool = False) -> None:
        self.threshold = threshold
        self.invert = invert

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ThresholdClassifier":
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        positive = np.asarray(X)[:, 0] >= self.threshold
        if self.invert:
            positive = ~positive
        return np.where(positive, self.classes_[1], self.classes_[0])


class PLSDAClassifierTests(unittest.TestCase):
    def test_multiclass_fit_predict_and_scores(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.asarray(["left"] * 12 + ["center"] * 12 + ["right"] * 12)
        values = np.vstack(
            [
                rng.normal(-2.0, 0.15, size=(12, 8)),
                rng.normal(0.0, 0.15, size=(12, 8)),
                rng.normal(2.0, 0.15, size=(12, 8)),
            ]
        )
        model = PLSDAClassifier(n_components=3).fit(values, labels)

        predicted = model.predict(values)
        scores = model.decision_function(values)

        self.assertGreaterEqual(float(np.mean(predicted == labels)), 0.90)
        self.assertEqual(scores.shape, (36, 3))
        self.assertFalse(hasattr(model, "predict_proba"))


class AgreementAwareVotingClassifierTests(unittest.TestCase):
    def test_majority_vote_and_vote_fraction_diagnostics(self) -> None:
        values = np.asarray([[-1.0], [1.0]])
        labels = np.asarray(["left", "right"])
        model = AgreementAwareVotingClassifier(
            estimators=(
                ("primary", _ThresholdClassifier(0.0)),
                ("support_a", _ThresholdClassifier(0.5)),
                ("support_b", _ThresholdClassifier(2.0)),
            ),
            primary_estimator="primary",
        ).fit(values, labels)

        predicted = model.predict(np.asarray([[1.0]]))
        probabilities = model.predict_proba(np.asarray([[1.0]]))
        diagnostics = model.predict_diagnostics(np.asarray([[1.0]]))[0]

        self.assertEqual(predicted.tolist(), ["right"])
        self.assertAlmostEqual(float(np.max(probabilities)), 2.0 / 3.0)
        self.assertEqual(diagnostics["selected_vote_count"], 2)
        self.assertFalse(diagnostics["unanimous"])
        self.assertEqual(
            diagnostics["confidence_semantics"],
            "ensemble_vote_fraction_not_calibrated",
        )

    def test_primary_estimator_resolves_all_different_multiclass_tie(self) -> None:
        class FixedClassifier(ClassifierMixin, BaseEstimator):
            def __init__(self, label: str) -> None:
                self.label = label

            def fit(self, X: np.ndarray, y: np.ndarray) -> "FixedClassifier":
                self.classes_ = np.unique(y)
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.asarray([self.label] * len(X), dtype=object)

        values = np.asarray([[0.0], [1.0], [2.0]])
        labels = np.asarray(["P11", "P21", "P31"])
        model = AgreementAwareVotingClassifier(
            estimators=(
                ("primary", FixedClassifier("P21")),
                ("left", FixedClassifier("P11")),
                ("right", FixedClassifier("P31")),
            ),
            primary_estimator="primary",
        ).fit(values, labels)

        self.assertEqual(model.predict([[4.0]]).tolist(), ["P21"])
        self.assertAlmostEqual(float(np.max(model.predict_proba([[4.0]]))), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
