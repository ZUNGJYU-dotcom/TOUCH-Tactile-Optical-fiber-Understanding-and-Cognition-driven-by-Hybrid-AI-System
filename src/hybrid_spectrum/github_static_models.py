"""Small-sample spectral classifiers used by the GitHub candidate benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Multiclass PLS-DA with explicit, non-probabilistic decision scores.

    PLS-DA is a useful chemometric baseline for high-dimensional spectra with
    relatively few independent samples.  The returned scores are not calibrated
    probabilities and must not be presented as confidence values.
    """

    def __init__(self, n_components: int = 8, max_iter: int = 1000) -> None:
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        values, labels = check_X_y(X, y, dtype=float)
        self.label_binarizer_ = LabelBinarizer()
        encoded = self.label_binarizer_.fit_transform(labels)
        if encoded.ndim == 1:
            encoded = encoded[:, None]
        if encoded.shape[1] == 1 and len(self.label_binarizer_.classes_) == 2:
            encoded = np.column_stack([1.0 - encoded[:, 0], encoded[:, 0]])
        component_count = min(
            int(self.n_components),
            max(values.shape[0] - 1, 1),
            values.shape[1],
        )
        self.model_ = PLSRegression(
            n_components=component_count,
            scale=False,
            max_iter=int(self.max_iter),
        )
        self.model_.fit(values, encoded.astype(float))
        self.classes_ = np.asarray(self.label_binarizer_.classes_, dtype=object)
        self.n_features_in_ = values.shape[1]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("model_", "classes_"))
        values = check_array(X, dtype=float)
        scores = np.asarray(self.model_.predict(values), dtype=float)
        if scores.ndim == 1:
            scores = scores[:, None]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


class AgreementAwareVotingClassifier(ClassifierMixin, BaseEstimator):
    """Hard-voting classifier that exposes model agreement, not fake confidence.

    A deterministic primary estimator resolves the rare all-different tie.  The
    values returned by ``predict_proba`` are vote fractions.  They are useful
    for runtime review flags, but they are deliberately identified as
    uncalibrated ensemble agreement rather than class probabilities.
    """

    confidence_source = "ensemble_vote_fraction_not_calibrated"

    def __init__(
        self,
        estimators: tuple[tuple[str, BaseEstimator], ...],
        primary_estimator: str,
    ) -> None:
        self.estimators = estimators
        self.primary_estimator = primary_estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AgreementAwareVotingClassifier":
        values, labels = check_X_y(X, y, dtype=float)
        names = [name for name, _ in self.estimators]
        if not names:
            raise ValueError("at least one estimator is required")
        if len(names) != len(set(names)):
            raise ValueError("estimator names must be unique")
        if self.primary_estimator not in names:
            raise ValueError("primary_estimator must name one configured estimator")

        self.classes_ = np.unique(labels)
        self.estimators_ = tuple(
            (name, clone(estimator).fit(values, labels))
            for name, estimator in self.estimators
        )
        self.n_features_in_ = values.shape[1]
        return self

    def _member_predictions(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("estimators_", "classes_"))
        values = check_array(X, dtype=float)
        return np.asarray(
            [estimator.predict(values) for _, estimator in self.estimators_],
            dtype=object,
        )

    def _vote(self, member_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        counts = np.zeros(
            (member_predictions.shape[1], len(self.classes_)),
            dtype=float,
        )
        for row in member_predictions:
            for sample_index, label in enumerate(row):
                counts[sample_index, class_to_index[label]] += 1.0

        primary_index = next(
            index
            for index, (name, _) in enumerate(self.estimators_)
            if name == self.primary_estimator
        )
        primary_predictions = member_predictions[primary_index]
        selected: list[object] = []
        for sample_index, row in enumerate(counts):
            tied = np.flatnonzero(row == np.max(row))
            primary_label = primary_predictions[sample_index]
            primary_class_index = class_to_index[primary_label]
            selected_index = (
                primary_class_index
                if primary_class_index in tied
                else int(tied[0])
            )
            selected.append(self.classes_[selected_index])
        return np.asarray(selected, dtype=object), counts

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicted, _ = self._vote(self._member_predictions(X))
        return predicted

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, counts = self._vote(self._member_predictions(X))
        return counts / float(len(self.estimators_))

    def predict_diagnostics(self, X: np.ndarray) -> list[dict[str, object]]:
        member_predictions = self._member_predictions(X)
        predicted, counts = self._vote(member_predictions)
        names = [name for name, _ in self.estimators_]
        diagnostics: list[dict[str, object]] = []
        for sample_index, label in enumerate(predicted):
            class_index = int(np.flatnonzero(self.classes_ == label)[0])
            vote_count = int(counts[sample_index, class_index])
            diagnostics.append(
                {
                    "selected_label": str(label),
                    "member_predictions": {
                        name: str(member_predictions[index, sample_index])
                        for index, name in enumerate(names)
                    },
                    "selected_vote_count": vote_count,
                    "member_count": len(names),
                    "agreement_fraction": vote_count / float(len(names)),
                    "unanimous": vote_count == len(names),
                    "confidence_semantics": self.confidence_source,
                }
            )
        return diagnostics


__all__ = ["AgreementAwareVotingClassifier", "PLSDAClassifier"]
