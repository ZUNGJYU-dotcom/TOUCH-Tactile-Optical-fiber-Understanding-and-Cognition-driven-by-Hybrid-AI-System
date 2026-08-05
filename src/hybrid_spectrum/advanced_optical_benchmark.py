"""Leakage-safe feature-view benchmarks for synchronized optical/force data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)


POSITION_ORDER = (
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
)


@dataclass(frozen=True)
class AlignedOpticalDataset:
    """Latest-primary rows represented by both feature pipelines."""

    peak_features: np.ndarray
    peak_feature_names: np.ndarray
    spectrum_features: np.ndarray
    spectrum_feature_names: np.ndarray
    contact_target: np.ndarray
    position_target: np.ndarray
    force_fz_n: np.ndarray
    contact_mask: np.ndarray
    position_mask: np.ndarray
    force_mask: np.ndarray
    fold_id: np.ndarray
    group_id: np.ndarray
    sample_index: np.ndarray


def _row_keys(groups: np.ndarray, indices: np.ndarray) -> list[tuple[str, int]]:
    return [(str(group), int(index)) for group, index in zip(groups, indices)]


def load_aligned_latest_primary(
    fusion_dataset_path: Path,
    spectrum_dataset_path: Path,
) -> AlignedOpticalDataset:
    """Align the two feature matrices by session and frame index.

    The fusion dataset contains auxiliary historical sources. Only the
    latest-primary rows are eligible for formal comparison. The strict PX6D
    dataset contains the same latest-primary frames in a separate feature
    representation, so alignment must be exact rather than positional.
    """

    with np.load(fusion_dataset_path, allow_pickle=False) as payload:
        formal_indices = np.flatnonzero(
            payload["formal_test_eligible"].astype(bool)
            & (payload["fold_id"].astype(int) >= 0)
        )
        peak_features = payload["features"][formal_indices].astype(
            np.float32, copy=False
        )
        peak_feature_names = payload["feature_names"].astype(str)
        contact_target = payload["contact_target"][formal_indices].astype(int)
        position_target = payload["position_target"][formal_indices].astype(str)
        force_fz_n = payload["force_fz_n"][formal_indices].astype(float)
        contact_mask = payload["contact_mask"][formal_indices].astype(bool)
        position_mask = payload["position_mask"][formal_indices].astype(bool)
        force_mask = payload["force_mask"][formal_indices].astype(bool)
        fold_id = payload["fold_id"][formal_indices].astype(int)
        group_id = payload["group_id"][formal_indices].astype(str)
        sample_index = payload["sample_index"][formal_indices].astype(int)

    with np.load(spectrum_dataset_path, allow_pickle=False) as payload:
        spectrum_keys = _row_keys(
            payload["session_id"].astype(str),
            payload["capture_index"].astype(int),
        )
        spectrum_lookup = {key: index for index, key in enumerate(spectrum_keys)}
        if len(spectrum_lookup) != len(spectrum_keys):
            raise ValueError("strict spectrum dataset contains duplicate frame keys")
        fusion_keys = _row_keys(group_id, sample_index)
        missing = [key for key in fusion_keys if key not in spectrum_lookup]
        if missing:
            raise ValueError(
                "strict spectrum dataset is missing aligned frames: "
                + ", ".join(map(str, missing[:3]))
            )
        reorder = np.asarray([spectrum_lookup[key] for key in fusion_keys], dtype=int)
        if len(set(fusion_keys)) != len(fusion_keys):
            raise ValueError("fusion dataset contains duplicate latest-primary keys")
        if len(spectrum_lookup) != len(fusion_keys):
            extra_count = len(spectrum_lookup) - len(fusion_keys)
            raise ValueError(
                f"strict spectrum dataset has {extra_count} unmatched frame(s)"
            )
        spectrum_features = payload["features"][reorder].astype(
            np.float32, copy=False
        )
        spectrum_feature_names = payload["feature_names"].astype(str)
        spectrum_fold = payload["fold_id"][reorder].astype(int)
        spectrum_contact = payload["contact_target"][reorder].astype(int)
        spectrum_position = payload["position_target"][reorder].astype(str)
        spectrum_force = payload["force_fz_n"][reorder].astype(float)

    checks = {
        "fold": np.array_equal(fold_id, spectrum_fold),
        "contact_target": np.array_equal(contact_target, spectrum_contact),
        "position_target": np.array_equal(position_target, spectrum_position),
        "force_target": np.allclose(force_fz_n, spectrum_force, equal_nan=True),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("aligned datasets disagree on " + ", ".join(failed))
    for name, matrix in (
        ("peak feature", peak_features),
        ("spectrum feature", spectrum_features),
    ):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} matrix contains NaN or infinite values")

    return AlignedOpticalDataset(
        peak_features=peak_features,
        peak_feature_names=peak_feature_names,
        spectrum_features=spectrum_features,
        spectrum_feature_names=spectrum_feature_names,
        contact_target=contact_target,
        position_target=position_target,
        force_fz_n=force_fz_n,
        contact_mask=contact_mask,
        position_mask=position_mask,
        force_mask=force_mask,
        fold_id=fold_id,
        group_id=group_id,
        sample_index=sample_index,
    )


def build_feature_views(
    dataset: AlignedOpticalDataset,
) -> Mapping[str, np.ndarray]:
    """Return comparable peak, spectrum, and fused feature views."""

    current_indices = np.flatnonzero(
        np.char.startswith(dataset.peak_feature_names.astype(str), "last__")
    )
    if len(current_indices) != 40:
        raise ValueError(
            "expected 40 current-frame peak features, found "
            f"{len(current_indices)}"
        )
    if dataset.spectrum_features.shape[1] < 264:
        raise ValueError("expected the strict 264-dimensional spectrum feature set")
    spectrum_192 = dataset.spectrum_features[:, :192]
    spectrum_264 = dataset.spectrum_features[:, :264]
    peak_temporal = dataset.peak_features
    return {
        "peak_current_40": peak_temporal[:, current_indices],
        "peak_temporal_483": peak_temporal,
        "full_spectrum_192": spectrum_192,
        "full_spectrum_264": spectrum_264,
        "peak_temporal_plus_spectrum_192": np.concatenate(
            (peak_temporal, spectrum_192), axis=1
        ).astype(np.float32, copy=False),
        "peak_temporal_plus_spectrum_264": np.concatenate(
            (peak_temporal, spectrum_264), axis=1
        ).astype(np.float32, copy=False),
    }


def equal_group_weights(groups: np.ndarray) -> np.ndarray:
    """Give each independent capture session equal total fitting mass."""

    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return weights / max(float(np.mean(weights)), 1.0e-12)


def _majority_vote_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    labels: list[Any],
) -> dict[str, float]:
    group_true: list[Any] = []
    group_predicted: list[Any] = []
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_true.append(Counter(true[selected].tolist()).most_common(1)[0][0])
        group_predicted.append(
            Counter(predicted[selected].tolist()).most_common(1)[0][0]
        )
    return {
        "group_voting_accuracy": float(
            accuracy_score(group_true, group_predicted)
        ),
        "group_voting_macro_f1": float(
            f1_score(
                group_true,
                group_predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _latency_ms_per_frame(estimator: Any, features: np.ndarray) -> float:
    subset = features[: min(1024, len(features))]
    if not len(subset):
        return float("nan")
    estimator.predict(subset[: min(16, len(subset))])
    timings: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        estimator.predict(subset)
        timings.append((time.perf_counter() - start) * 1000.0 / len(subset))
    return float(np.median(timings))


def grouped_extra_trees_classification(
    *,
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    labels: list[Any],
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate a classifier using the preassigned session-group folds."""

    predicted = np.full(target.shape, "", dtype=object)
    fit_seconds = 0.0
    fold_latencies: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(set(fold_id[mask].tolist())):
        train = mask & (fold_id != fold)
        test = mask & (fold_id == fold)
        if set(group_id[train]).intersection(group_id[test]):
            raise RuntimeError(f"group leakage detected in fold {fold}")
        model = ExtraTreesClassifier(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed + int(fold),
            n_jobs=-1,
        )
        start = time.perf_counter()
        model.fit(
            features[train],
            target[train],
            sample_weight=equal_group_weights(group_id[train]),
        )
        fit_seconds += time.perf_counter() - start
        fold_prediction = model.predict(features[test])
        predicted[test] = fold_prediction
        latency = _latency_ms_per_frame(model, features[test])
        fold_latencies.append(latency)
        fold_rows.append(
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
    selected_true = target[mask]
    selected_predicted = np.asarray(predicted[mask], dtype=target.dtype)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(selected_true, selected_predicted)),
        "macro_f1": float(
            f1_score(
                selected_true,
                selected_predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            selected_true, selected_predicted, labels=labels
        ).tolist(),
        "labels": [str(value) for value in labels],
        "training_time_sec": fit_seconds,
        "inference_latency_ms_per_frame": float(np.mean(fold_latencies)),
        "frame_count": int(np.sum(mask)),
        "group_count": int(len(set(group_id[mask].tolist()))),
        "fold_metrics": fold_rows,
    }
    metrics.update(
        _majority_vote_metrics(
            selected_true,
            selected_predicted,
            group_id[mask],
            labels,
        )
    )
    return metrics, predicted


def grouped_extra_trees_force(
    *,
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate Fz regression using exactly the same grouped folds."""

    predicted = np.full(target.shape, np.nan, dtype=float)
    fit_seconds = 0.0
    fold_latencies: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(set(fold_id[mask].tolist())):
        train = mask & (fold_id != fold)
        test = mask & (fold_id == fold)
        if set(group_id[train]).intersection(group_id[test]):
            raise RuntimeError(f"group leakage detected in fold {fold}")
        model = ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features=0.8,
            random_state=seed + int(fold),
            n_jobs=-1,
        )
        start = time.perf_counter()
        model.fit(
            features[train],
            target[train],
            sample_weight=equal_group_weights(group_id[train]),
        )
        fit_seconds += time.perf_counter() - start
        fold_prediction = model.predict(features[test])
        predicted[test] = fold_prediction
        latency = _latency_ms_per_frame(model, features[test])
        fold_latencies.append(latency)
        fold_rows.append(
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
    selected_true = target[mask]
    selected_predicted = predicted[mask]
    active = selected_true >= 0.10
    metrics = {
        "mae_n": float(mean_absolute_error(selected_true, selected_predicted)),
        "rmse_n": float(
            np.sqrt(mean_squared_error(selected_true, selected_predicted))
        ),
        "r2": float(r2_score(selected_true, selected_predicted)),
        "active_force_mae_n": float(
            mean_absolute_error(selected_true[active], selected_predicted[active])
        ),
        "within_0_25_n": float(
            np.mean(np.abs(selected_true - selected_predicted) <= 0.25)
        ),
        "training_time_sec": fit_seconds,
        "inference_latency_ms_per_frame": float(np.mean(fold_latencies)),
        "frame_count": int(np.sum(mask)),
        "group_count": int(len(set(group_id[mask].tolist()))),
        "fold_metrics": fold_rows,
    }
    return metrics, predicted


def contact_recalls(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return the two recalls that expose false-contact behavior."""

    return {
        "no_contact_recall": float(
            recall_score(true, predicted, pos_label=0, zero_division=0)
        ),
        "contact_recall": float(
            recall_score(true, predicted, pos_label=1, zero_division=0)
        ),
    }


__all__ = [
    "AlignedOpticalDataset",
    "POSITION_ORDER",
    "build_feature_views",
    "contact_recalls",
    "equal_group_weights",
    "grouped_extra_trees_classification",
    "grouped_extra_trees_force",
    "load_aligned_latest_primary",
]
