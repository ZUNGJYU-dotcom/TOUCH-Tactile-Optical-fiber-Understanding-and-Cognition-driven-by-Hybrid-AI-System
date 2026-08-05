"""Leakage-safe candidate models for grouped optical-spectrum recognition.

This module is intentionally offline-only.  It compares compact chemometric,
linear, kernel, and tree classifiers on the frozen latest-primary session
folds without changing the TOUCH runtime model bundle.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .advanced_optical_benchmark import equal_group_weights
from .github_static_models import PLSDAClassifier


EstimatorFactory = Callable[[int], BaseEstimator]


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    model_family: str
    confidence_source: str
    factory: EstimatorFactory
    sample_weight_parameter: str | None


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return fixed candidates chosen for small-session spectral data."""

    return (
        CandidateSpec(
            model_id="extra_trees",
            model_family="tree_ensemble",
            confidence_source="predict_proba_uncalibrated",
            factory=lambda seed: ExtraTreesClassifier(
                n_estimators=192,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            sample_weight_parameter="sample_weight",
        ),
        CandidateSpec(
            model_id="logistic_l2",
            model_family="linear_chemometric",
            confidence_source="decision_score_not_calibrated",
            factory=lambda seed: Pipeline(
                (
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=seed,
                        ),
                    ),
                )
            ),
            sample_weight_parameter="model__sample_weight",
        ),
        CandidateSpec(
            model_id="linear_svm",
            model_family="linear_margin",
            confidence_source="decision_margin_not_probability",
            factory=lambda seed: Pipeline(
                (
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LinearSVC(
                            C=0.25,
                            class_weight="balanced",
                            dual="auto",
                            max_iter=10000,
                            random_state=seed,
                        ),
                    ),
                )
            ),
            sample_weight_parameter="model__sample_weight",
        ),
        CandidateSpec(
            model_id="rbf_svm_pca32",
            model_family="kernel_chemometric",
            confidence_source="decision_margin_not_probability",
            factory=lambda seed: Pipeline(
                (
                    ("scale", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=32,
                            whiten=True,
                            svd_solver="randomized",
                            random_state=seed,
                        ),
                    ),
                    (
                        "model",
                        SVC(
                            C=4.0,
                            kernel="rbf",
                            gamma="scale",
                            class_weight="balanced",
                            cache_size=2048,
                        ),
                    ),
                )
            ),
            sample_weight_parameter="model__sample_weight",
        ),
        CandidateSpec(
            model_id="pls_da_12",
            model_family="pls_chemometric",
            confidence_source="pls_decision_score_not_probability",
            factory=lambda seed: Pipeline(
                (
                    ("scale", StandardScaler()),
                    ("model", PLSDAClassifier(n_components=12)),
                )
            ),
            sample_weight_parameter=None,
        ),
        CandidateSpec(
            model_id="shrinkage_lda",
            model_family="linear_discriminant",
            confidence_source="predict_proba_model_based_uncalibrated",
            factory=lambda seed: Pipeline(
                (
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LinearDiscriminantAnalysis(
                            solver="lsqr",
                            shrinkage="auto",
                        ),
                    ),
                )
            ),
            sample_weight_parameter=None,
        ),
    )


def _latency_ms_per_frame(model: BaseEstimator, features: np.ndarray) -> float:
    subset = features[: min(1024, len(features))]
    if not len(subset):
        return float("nan")
    model.predict(subset[: min(16, len(subset))])
    timings: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        model.predict(subset)
        timings.append((time.perf_counter() - started) * 1000.0 / len(subset))
    return float(np.median(timings))


def _group_vote(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    labels: list[Any],
) -> dict[str, float]:
    true_by_group: list[Any] = []
    predicted_by_group: list[Any] = []
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        true_by_group.append(Counter(true[selected].tolist()).most_common(1)[0][0])
        predicted_by_group.append(
            Counter(predicted[selected].tolist()).most_common(1)[0][0]
        )
    return {
        "group_voting_accuracy": float(
            accuracy_score(true_by_group, predicted_by_group)
        ),
        "group_voting_macro_f1": float(
            f1_score(
                true_by_group,
                predicted_by_group,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def grouped_candidate_classification(
    *,
    spec: CandidateSpec,
    features: np.ndarray,
    target: np.ndarray,
    training_and_evaluation_mask: np.ndarray,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    labels: list[Any],
    seed: int,
    predict_all_fold_rows: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one candidate on frozen session folds.

    ``predict_all_fold_rows`` is used for position heads in a hierarchical
    pipeline.  The head is trained and scored only on true contact rows, but it
    also emits predictions for no-contact rows so the contact gate can decide
    whether those predictions are exposed.
    """

    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target)
    mask = np.asarray(training_and_evaluation_mask, dtype=bool)
    folds = np.asarray(fold_id, dtype=int)
    groups = np.asarray(group_id, dtype=str)
    if not np.all(np.isfinite(features)):
        raise ValueError("candidate feature matrix contains NaN or infinity")
    if np.any(folds[mask] < 0):
        raise ValueError("formal candidate rows must have a non-negative fold")

    predicted = np.full(len(target), None, dtype=object)
    total_training_sec = 0.0
    fold_latencies: list[float] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted(set(folds[mask].tolist())):
        train = mask & (folds != fold)
        evaluated = mask & (folds == fold)
        predicted_rows = (folds == fold) if predict_all_fold_rows else evaluated
        if set(groups[train]).intersection(groups[predicted_rows]):
            raise RuntimeError(f"session leakage detected in fold {fold}")
        model = spec.factory(seed + int(fold))
        fit_kwargs: dict[str, Any] = {}
        if spec.sample_weight_parameter:
            fit_kwargs[spec.sample_weight_parameter] = equal_group_weights(groups[train])
        started = time.perf_counter()
        model.fit(features[train], target[train], **fit_kwargs)
        total_training_sec += time.perf_counter() - started
        predicted[predicted_rows] = model.predict(features[predicted_rows])
        evaluated_prediction = np.asarray(predicted[evaluated], dtype=target.dtype)
        latency = _latency_ms_per_frame(model, features[evaluated])
        fold_latencies.append(latency)
        fold_metrics.append(
            {
                "fold_id": int(fold),
                "accuracy": float(accuracy_score(target[evaluated], evaluated_prediction)),
                "macro_f1": float(
                    f1_score(
                        target[evaluated],
                        evaluated_prediction,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "test_frame_count": int(np.sum(evaluated)),
                "test_session_count": int(len(set(groups[evaluated].tolist()))),
                "latency_ms_per_frame": latency,
            }
        )

    true = target[mask]
    evaluated_prediction = np.asarray(predicted[mask], dtype=target.dtype)
    report = classification_report(
        true,
        evaluated_prediction,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "model_id": spec.model_id,
        "model_family": spec.model_family,
        "confidence_source": spec.confidence_source,
        "accuracy": float(accuracy_score(true, evaluated_prediction)),
        "macro_f1": float(
            f1_score(
                true,
                evaluated_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(
            true, evaluated_prediction, labels=labels
        ).tolist(),
        "classification_report": report,
        "training_time_sec": total_training_sec,
        "inference_latency_ms_per_frame": float(np.mean(fold_latencies)),
        "frame_count": int(np.sum(mask)),
        "session_count": int(len(set(groups[mask].tolist()))),
        "fold_metrics": fold_metrics,
        "uses_equal_session_weight": spec.sample_weight_parameter is not None,
    }
    metrics.update(_group_vote(true, evaluated_prediction, groups[mask], labels))
    return metrics, predicted


def evaluate_hierarchical_predictions(
    *,
    contact_prediction: np.ndarray,
    position_prediction: np.ndarray,
    contact_target: np.ndarray,
    position_target: np.ndarray,
    eligible_mask: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Score no-contact plus P11-P33 as one end-to-end decision."""

    mask = np.asarray(eligible_mask, dtype=bool)
    contact_true = np.asarray(contact_target, dtype=int)
    true_eval = np.where(
        contact_true[mask] == 0,
        "no_contact",
        np.asarray(position_target, dtype=str)[mask],
    )
    # Candidate heads intentionally leave rows outside their formal evaluation
    # domain unset.  Slice first so those sentinel values cannot contaminate
    # the hierarchy's dtype conversion.
    contact_eval = np.asarray(contact_prediction, dtype=object)[mask].astype(int)
    position_eval = np.asarray(position_prediction, dtype=object)[mask].astype(str)
    predicted_eval = np.where(contact_eval == 0, "no_contact", position_eval)
    no_contact = true_eval == "no_contact"
    contact = ~no_contact
    false_contact_rate = float(
        np.mean(predicted_eval[no_contact] != "no_contact")
        if np.any(no_contact)
        else np.nan
    )
    missed_contact_rate = float(
        np.mean(predicted_eval[contact] == "no_contact")
        if np.any(contact)
        else np.nan
    )
    active_correct = contact & (predicted_eval != "no_contact")
    conditional_position_accuracy = float(
        np.mean(predicted_eval[active_correct] == true_eval[active_correct])
        if np.any(active_correct)
        else np.nan
    )
    return {
        "accuracy": float(accuracy_score(true_eval, predicted_eval)),
        "macro_f1": float(
            f1_score(
                true_eval,
                predicted_eval,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "false_contact_rate": false_contact_rate,
        "no_contact_recall": 1.0 - false_contact_rate,
        "missed_contact_rate": missed_contact_rate,
        "contact_recall": 1.0 - missed_contact_rate,
        "conditional_position_accuracy": conditional_position_accuracy,
        "confusion_matrix": confusion_matrix(
            true_eval, predicted_eval, labels=labels
        ).tolist(),
        "labels": labels,
        "frame_count": int(np.sum(mask)),
    }


__all__ = [
    "CandidateSpec",
    "candidate_specs",
    "evaluate_hierarchical_predictions",
    "grouped_candidate_classification",
]
