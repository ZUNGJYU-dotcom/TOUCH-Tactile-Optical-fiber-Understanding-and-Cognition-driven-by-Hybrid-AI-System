from __future__ import annotations

import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from src.hybrid_spectrum.factorized_position import (
    POSITION_ORDER,
    FactorizedPositionClassifier,
)


class FactorizedPositionClassifierTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> tuple[np.ndarray, np.ndarray]:
        values = []
        labels = []
        for label in POSITION_ORDER:
            column = float(label[1])
            row = float(label[2])
            for offset in (-0.05, 0.0, 0.05):
                values.append([column + offset, row - offset, column * row])
                labels.append(label)
        return np.asarray(values, dtype=float), np.asarray(labels)

    def test_factorized_prediction_preserves_project_array_orientation(self) -> None:
        values, labels = self._dataset()
        estimator = ExtraTreesClassifier(n_estimators=80, random_state=7)
        model = FactorizedPositionClassifier(estimator).fit(values, labels)

        predicted = model.predict(values)

        np.testing.assert_array_equal(predicted, labels)
        self.assertEqual(model.classes_.tolist(), list(POSITION_ORDER))

    def test_probability_is_normalized_over_nine_positions(self) -> None:
        values, labels = self._dataset()
        estimator = ExtraTreesClassifier(n_estimators=40, random_state=11)
        model = FactorizedPositionClassifier(estimator).fit(values, labels)

        probability = model.predict_proba(values[:4])

        self.assertEqual(probability.shape, (4, 9))
        np.testing.assert_allclose(np.sum(probability, axis=1), 1.0)

    def test_invalid_position_label_is_rejected(self) -> None:
        values, labels = self._dataset()
        labels[0] = "P99"
        estimator = ExtraTreesClassifier(n_estimators=10, random_state=3)

        with self.assertRaisesRegex(ValueError, "invalid 3x3 position"):
            FactorizedPositionClassifier(estimator).fit(values, labels)


if __name__ == "__main__":
    unittest.main()
