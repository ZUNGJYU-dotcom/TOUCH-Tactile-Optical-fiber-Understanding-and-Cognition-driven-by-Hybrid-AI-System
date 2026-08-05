"""Grouped model comparison for rich ordinary-FBG optical features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .advanced_optical_benchmark import AlignedOpticalDataset, equal_group_weights
from .rich_optical_features import RichFeatureCache


@dataclass(frozen=True)
class FeatureView:
    values: np.ndarray
    names: np.ndarray


def build_rich_feature_views(
    dataset: AlignedOpticalDataset,
    cache: RichFeatureCache,
) -> Mapping[str, FeatureView]:
    """Build strictly aligned feature views for the ablation benchmark."""

    if not np.array_equal(cache.group_id, dataset.group_id):
        raise ValueError("rich feature group order does not match the formal dataset")
    if not np.array_equal(cache.sample_index, dataset.sample_index):
        raise ValueError("rich feature frame order does not match the formal dataset")
    current_indices = np.flatnonzero(
        np.char.startswith(dataset.peak_feature_names.astype(str), "last__")
    )
    if len(current_indices) != 40:
        raise ValueError(f"expected 40 current peak features, found {len(current_indices)}")
    if dataset.spectrum_features.shape[1] < 192:
        raise ValueError("the formal spectrum dataset lacks the 192-feature view")

    current_values = dataset.peak_features[:, current_indices]
    current_names = dataset.peak_feature_names[current_indices]
    spectrum_values = dataset.spectrum_features[:, :192]
    spectrum_names = dataset.spectrum_feature_names[:192]
    rich_values = cache.features
    rich_names = cache.feature_names
    return {
        "peak_current_40": FeatureView(current_values, current_names),
        "full_spectrum_192": FeatureView(spectrum_values, spectrum_names),
        "rich_optical_physics": FeatureView(rich_values, rich_names),
        "rich_plus_full_spectrum_192": FeatureView(
            np.concatenate((rich_values, spectrum_values), axis=1).astype(
                np.float32, copy=False
            ),
            np.concatenate(
                (
                    np.char.add("rich__", rich_names.astype(str)),
                    np.char.add("spectrum__", spectrum_names.astype(str)),
                )
            ),
        ),
    }


def _optional_lightgbm() -> tuple[Any, Any]:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("lightgbm is not installed") from error
    return LGBMClassifier, LGBMRegressor


def model_available(model_id: str) -> tuple[bool, str]:
    if model_id == "extra_trees":
        return True, ""
    if model_id == "lightgbm":
        try:
            _optional_lightgbm()
        except RuntimeError as error:
            return False, str(error)
        return True, ""
    return False, f"unknown model: {model_id}"


def _make_classifier(
    model_id: str,
    *,
    seed: int,
    estimators: int,
    minimum_leaf_samples: int,
) -> Any:
    if model_id == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_id == "lightgbm":
        LGBMClassifier, _ = _optional_lightgbm()
        return LGBMClassifier(
            n_estimators=max(120, estimators),
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=max(12, minimum_leaf_samples * 6),
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown classifier: {model_id}")


def _make_regressor(
    model_id: str,
    *,
    seed: int,
    estimators: int,
    minimum_leaf_samples: int,
) -> Any:
    if model_id == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    if model_id == "lightgbm":
        _, LGBMRegressor = _optional_lightgbm()
        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=max(160, estimators),
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=max(12, minimum_leaf_samples * 6),
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=0.5,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown regressor: {model_id}")


def _latency_ms_per_frame(estimator: Any, features: np.ndarray) -> float:
    subset = features[: min(1024, len(features))]
    if not len(subset):
        return float("nan")
    _predict(estimator, subset[: min(16, len(subset))])
    samples: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        _predict(estimator, subset)
        samples.append((time.perf_counter() - started) * 1000.0 / len(subset))
    return float(np.median(samples))


def _predict(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Predict without LightGBM's spurious NumPy feature-name warning."""

    if estimator.__class__.__module__.startswith("lightgbm"):
        return np.asarray(estimator.predict(features, validate_features=False))
    return np.asarray(estimator.predict(features))


def _group_vote(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    labels: list[Any],
) -> dict[str, float]:
    voted_true: list[Any] = []
    voted_predicted: list[Any] = []
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        voted_true.append(Counter(true[selected].tolist()).most_common(1)[0][0])
        voted_predicted.append(
            Counter(predicted[selected].tolist()).most_common(1)[0][0]
        )
    return {
        "group_voting_accuracy": float(
            accuracy_score(voted_true, voted_predicted)
        ),
        "group_voting_macro_f1": float(
            f1_score(
                voted_true,
                voted_predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _top_importances(
    importances: list[np.ndarray],
    feature_names: np.ndarray,
    limit: int = 30,
) -> list[dict[str, float | str]]:
    if not importances:
        return []
    mean_importance = np.mean(np.stack(importances), axis=0)
    order = np.argsort(mean_importance)[::-1][:limit]
    return [
        {
            "feature": str(feature_names[index]),
            "importance": float(mean_importance[index]),
        }
        for index in order
        if mean_importance[index] > 0
    ]


def grouped_classification(
    *,
    model_id: str,
    feature_view: FeatureView,
    target: np.ndarray,
    mask: np.ndarray,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    labels: list[Any],
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one classifier on immutable capture-session folds."""

    predicted = np.full(target.shape, None, dtype=object)
    folds = sorted(set(fold_id[mask].tolist()))
    fit_seconds = 0.0
    latencies: list[float] = []
    importances: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        train = mask & (fold_id != fold)
        test = mask & (fold_id == fold)
        if set(group_id[train]).intersection(group_id[test]):
            raise RuntimeError(f"group leakage detected in fold {fold}")
        estimator = _make_classifier(
            model_id,
            seed=seed + int(fold),
            estimators=estimators,
            minimum_leaf_samples=minimum_leaf_samples,
        )
        started = time.perf_counter()
        estimator.fit(
            feature_view.values[train],
            target[train],
            sample_weight=equal_group_weights(group_id[train]),
        )
        fit_seconds += time.perf_counter() - started
        fold_prediction = _predict(estimator, feature_view.values[test])
        predicted[test] = fold_prediction
        latency = _latency_ms_per_frame(estimator, feature_view.values[test])
        latencies.append(latency)
        if hasattr(estimator, "feature_importances_"):
            importances.append(np.asarray(estimator.feature_importances_, dtype=float))
        fold_metrics.append(
            {
                "fold_id": int(fold),
                "accuracy": float(accuracy_score(target[test], fold_prediction)),
                "macro_f1": float(
                    f1_score(
                        target[test],
                        fold_prediction,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "test_frame_count": int(np.sum(test)),
                "test_group_count": int(len(set(group_id[test].tolist()))),
                "latency_ms_per_frame": latency,
            }
        )
        if progress is not None:
            progress(fold_number, len(folds))

    true = target[mask]
    selected_prediction = np.asarray(predicted[mask], dtype=target.dtype)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(true, selected_prediction)),
        "macro_f1": float(
            f1_score(
                true,
                selected_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            true, selected_prediction, labels=labels
        ).tolist(),
        "labels": [str(label) for label in labels],
        "frame_count": int(np.sum(mask)),
        "group_count": int(len(set(group_id[mask].tolist()))),
        "training_time_sec": fit_seconds,
        "inference_latency_ms_per_frame": float(np.mean(latencies)),
        "fold_metrics": fold_metrics,
        "top_feature_importance": _top_importances(
            importances, feature_view.names
        ),
    }
    metrics.update(_group_vote(true, selected_prediction, group_id[mask], labels))
    return metrics, predicted


def grouped_force_regression(
    *,
    model_id: str,
    feature_view: FeatureView,
    target: np.ndarray,
    mask: np.ndarray,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate optical-only Fz estimation on immutable session folds."""

    predicted = np.full(target.shape, np.nan, dtype=float)
    folds = sorted(set(fold_id[mask].tolist()))
    fit_seconds = 0.0
    latencies: list[float] = []
    importances: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        train = mask & (fold_id != fold)
        test = mask & (fold_id == fold)
        if set(group_id[train]).intersection(group_id[test]):
            raise RuntimeError(f"group leakage detected in fold {fold}")
        estimator = _make_regressor(
            model_id,
            seed=seed + int(fold),
            estimators=estimators,
            minimum_leaf_samples=minimum_leaf_samples,
        )
        started = time.perf_counter()
        estimator.fit(
            feature_view.values[train],
            target[train],
            sample_weight=equal_group_weights(group_id[train]),
        )
        fit_seconds += time.perf_counter() - started
        fold_prediction = _predict(estimator, feature_view.values[test])
        predicted[test] = fold_prediction
        latency = _latency_ms_per_frame(estimator, feature_view.values[test])
        latencies.append(latency)
        if hasattr(estimator, "feature_importances_"):
            importances.append(np.asarray(estimator.feature_importances_, dtype=float))
        fold_metrics.append(
            {
                "fold_id": int(fold),
                "mae_n": float(mean_absolute_error(target[test], fold_prediction)),
                "rmse_n": float(
                    np.sqrt(mean_squared_error(target[test], fold_prediction))
                ),
                "r2": float(r2_score(target[test], fold_prediction)),
                "test_frame_count": int(np.sum(test)),
                "test_group_count": int(len(set(group_id[test].tolist()))),
                "latency_ms_per_frame": latency,
            }
        )
        if progress is not None:
            progress(fold_number, len(folds))

    true = target[mask]
    selected_prediction = predicted[mask]
    active = true >= 0.10
    metrics: dict[str, Any] = {
        "mae_n": float(mean_absolute_error(true, selected_prediction)),
        "rmse_n": float(np.sqrt(mean_squared_error(true, selected_prediction))),
        "r2": float(r2_score(true, selected_prediction)),
        "active_force_mae_n": float(
            mean_absolute_error(true[active], selected_prediction[active])
        ),
        "within_0_25_n": float(np.mean(np.abs(true - selected_prediction) <= 0.25)),
        "frame_count": int(np.sum(mask)),
        "group_count": int(len(set(group_id[mask].tolist()))),
        "training_time_sec": fit_seconds,
        "inference_latency_ms_per_frame": float(np.mean(latencies)),
        "fold_metrics": fold_metrics,
        "top_feature_importance": _top_importances(
            importances, feature_view.names
        ),
    }
    return metrics, predicted


__all__ = [
    "FeatureView",
    "build_rich_feature_views",
    "grouped_classification",
    "grouped_force_regression",
    "model_available",
]
