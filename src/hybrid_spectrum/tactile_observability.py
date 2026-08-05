"""Leakage-safe screening of additional tactile observables.

This module deliberately separates correlation screening from calibration.
The current capture protocol was designed around position and Fz, so lateral
force and moment channels are exploratory targets even when grouped scores are
high.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .advanced_optical_benchmark import equal_group_weights


@dataclass(frozen=True)
class MechanicalTargetSpec:
    column: str
    display_name: str
    unit: str
    semantics: str


MECHANICAL_TARGET_SPECS = (
    MechanicalTargetSpec("fx_filtered_n", "Fx", "N", "signed lateral force"),
    MechanicalTargetSpec("fy_filtered_n", "Fy", "N", "signed lateral force"),
    MechanicalTargetSpec(
        "filtered_shear_resultant_n", "Shear magnitude", "N", "lateral force magnitude"
    ),
    MechanicalTargetSpec("mx_filtered_nm", "Mx", "N m", "signed moment"),
    MechanicalTargetSpec("my_filtered_nm", "My", "N m", "signed moment"),
    MechanicalTargetSpec("mz_filtered_nm", "Mz", "N m", "signed moment"),
    MechanicalTargetSpec(
        "filtered_moment_resultant_nm", "Moment magnitude", "N m", "moment magnitude"
    ),
)


@dataclass(frozen=True)
class AlignedMechanicalTargets:
    values: Mapping[str, np.ndarray]
    elapsed_time_sec: np.ndarray
    group_id: np.ndarray
    sample_index: np.ndarray


def load_aligned_mechanical_targets(
    *,
    capture_root: Path,
    group_id: np.ndarray,
    sample_index: np.ndarray,
    target_columns: Sequence[str],
) -> AlignedMechanicalTargets:
    """Align PX6D columns to frozen optical rows by session and capture index."""

    groups = np.asarray(group_id).astype(str)
    indices = np.asarray(sample_index).astype(int)
    if groups.ndim != 1 or groups.shape != indices.shape:
        raise ValueError("group_id and sample_index must be aligned vectors")
    if len(set(zip(groups.tolist(), indices.tolist()))) != len(groups):
        raise ValueError("formal frame keys must be unique")

    requested = tuple(dict.fromkeys(str(column) for column in target_columns))
    values = {column: np.full(len(groups), np.nan, dtype=float) for column in requested}
    elapsed = np.full(len(groups), np.nan, dtype=float)
    for current_group in dict.fromkeys(groups.tolist()):
        force_path = capture_root / current_group / "force_timeseries.csv"
        if not force_path.exists():
            raise FileNotFoundError(f"missing PX6D trace: {force_path}")
        required = ["capture_index", "elapsed_time_sec", *requested]
        frame = pd.read_csv(force_path, usecols=required)
        frame["capture_index"] = pd.to_numeric(
            frame["capture_index"], errors="raise"
        ).astype(int)
        if frame["capture_index"].duplicated().any():
            raise ValueError(f"duplicate capture_index in {force_path}")
        frame = frame.set_index("capture_index", verify_integrity=True)
        output_rows = np.flatnonzero(groups == current_group)
        capture_order = indices[output_rows]
        missing = np.setdiff1d(capture_order, frame.index.to_numpy(dtype=int))
        if len(missing):
            raise ValueError(
                f"{current_group} is missing capture_index {int(missing[0])}"
            )
        selected = frame.reindex(capture_order)
        elapsed[output_rows] = pd.to_numeric(
            selected["elapsed_time_sec"], errors="coerce"
        ).to_numpy(dtype=float)
        for column in requested:
            values[column][output_rows] = pd.to_numeric(
                selected[column], errors="coerce"
            ).to_numpy(dtype=float)

    if not np.all(np.isfinite(elapsed)):
        raise ValueError("aligned elapsed_time_sec contains invalid values")
    invalid = [column for column, array in values.items() if not np.all(np.isfinite(array))]
    if invalid:
        raise ValueError("aligned mechanical targets contain invalid values: " + ", ".join(invalid))
    return AlignedMechanicalTargets(
        values=values,
        elapsed_time_sec=elapsed,
        group_id=groups,
        sample_index=indices,
    )


def validate_force_alignment(
    formal_force_fz_n: np.ndarray,
    aligned_force_fz_n: np.ndarray,
    *,
    tolerance_n: float = 1.0e-5,
) -> float:
    """Return maximum Fz difference and reject timestamp/key misalignment."""

    formal = np.asarray(formal_force_fz_n, dtype=float)
    aligned = np.asarray(aligned_force_fz_n, dtype=float)
    if formal.shape != aligned.shape:
        raise ValueError("formal and aligned Fz arrays have different shapes")
    maximum = float(np.max(np.abs(formal - aligned)))
    if not np.isfinite(maximum) or maximum > tolerance_n:
        raise ValueError(
            f"PX6D alignment check failed: max Fz difference {maximum:.9g} N"
        )
    return maximum


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def _predict_position_median(
    train_target: np.ndarray,
    train_position: np.ndarray,
    test_position: np.ndarray,
    train_weights: np.ndarray,
) -> np.ndarray:
    global_median = _weighted_median(train_target, train_weights)
    medians: dict[str, float] = {}
    for label in sorted(set(train_position.tolist())):
        selected = train_position == label
        medians[str(label)] = _weighted_median(
            train_target[selected], train_weights[selected]
        )
    return np.asarray(
        [medians.get(str(label), global_median) for label in test_position],
        dtype=float,
    )


def _median_within_group_std(target: np.ndarray, groups: np.ndarray) -> float:
    values = [float(np.std(target[groups == group])) for group in sorted(set(groups.tolist()))]
    return float(np.median(values)) if values else float("nan")


def classify_observability(metrics: Mapping[str, float]) -> str:
    """Rate correlational observability without claiming physical calibration."""

    r2 = float(metrics.get("r2", float("nan")))
    skill = float(metrics.get("skill_over_position_baseline", float("nan")))
    within_ratio = float(metrics.get("within_session_to_global_std_ratio", 0.0))
    range_to_noise = float(metrics.get("robust_range_to_idle_6sigma", 0.0))
    if r2 >= 0.70 and skill >= 0.20 and within_ratio >= 0.15 and range_to_noise >= 1.0:
        return "strong_correlational_candidate"
    if r2 >= 0.30 and skill > 0.0 and within_ratio >= 0.08:
        return "exploratory_candidate"
    return "not_supported_by_current_protocol"


def grouped_regression_observability(
    *,
    features: np.ndarray,
    feature_names: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    fold_id: np.ndarray,
    group_id: np.ndarray,
    position_target: np.ndarray,
    contact_target: np.ndarray,
    estimators: int = 72,
    minimum_leaf_samples: int = 2,
    seed: int = 20260801,
) -> tuple[dict[str, Any], np.ndarray]:
    """Screen a mechanical target on immutable session folds."""

    x = np.asarray(features, dtype=np.float32)
    names = np.asarray(feature_names).astype(str)
    y = np.asarray(target, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(y)
    folds = sorted(set(np.asarray(fold_id)[valid].astype(int).tolist()))
    groups = np.asarray(group_id).astype(str)
    positions = np.asarray(position_target).astype(str)
    contacts = np.asarray(contact_target).astype(int)
    predicted = np.full(y.shape, np.nan, dtype=float)
    global_baseline = np.full(y.shape, np.nan, dtype=float)
    position_baseline = np.full(y.shape, np.nan, dtype=float)
    importances: list[np.ndarray] = []
    latencies: list[float] = []
    fit_seconds = 0.0
    fold_rows: list[dict[str, float | int]] = []

    for fold in folds:
        train = valid & (fold_id != fold)
        test = valid & (fold_id == fold)
        if set(groups[train]).intersection(groups[test]):
            raise RuntimeError(f"group leakage detected in fold {fold}")
        weights = equal_group_weights(groups[train])
        median = _weighted_median(y[train], weights)
        global_baseline[test] = median
        position_baseline[test] = _predict_position_median(
            y[train], positions[train], positions[test], weights
        )
        estimator = ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features=0.8,
            random_state=seed + int(fold),
            n_jobs=-1,
        )
        started = time.perf_counter()
        estimator.fit(x[train], y[train], sample_weight=weights)
        fit_seconds += time.perf_counter() - started
        fold_prediction = estimator.predict(x[test])
        predicted[test] = fold_prediction
        latency_input = x[test][: min(1024, int(np.sum(test)))]
        latency_started = time.perf_counter()
        estimator.predict(latency_input)
        latencies.append(
            (time.perf_counter() - latency_started) * 1000.0 / len(latency_input)
        )
        importances.append(np.asarray(estimator.feature_importances_, dtype=float))
        fold_rows.append(
            {
                "fold_id": int(fold),
                "mae": float(mean_absolute_error(y[test], fold_prediction)),
                "r2": float(r2_score(y[test], fold_prediction)),
                "test_frames": int(np.sum(test)),
                "test_sessions": int(len(set(groups[test].tolist()))),
            }
        )

    true = y[valid]
    model = predicted[valid]
    global_base = global_baseline[valid]
    position_base = position_baseline[valid]
    robust_low, robust_high = np.percentile(true, [1.0, 99.0])
    robust_range = float(robust_high - robust_low)
    global_std = float(np.std(true))
    within_std = _median_within_group_std(true, groups[valid])
    idle = true[contacts[valid] == 0]
    idle_std = float(np.std(idle)) if len(idle) else float("nan")
    mae = float(mean_absolute_error(true, model))
    global_mae = float(mean_absolute_error(true, global_base))
    position_mae = float(mean_absolute_error(true, position_base))
    mean_importance = np.mean(np.stack(importances), axis=0)
    top_order = np.argsort(mean_importance)[::-1][:20]
    metrics: dict[str, Any] = {
        "frame_count": int(np.sum(valid)),
        "session_count": int(len(set(groups[valid].tolist()))),
        "target_mean": float(np.mean(true)),
        "target_std": global_std,
        "target_p01": float(robust_low),
        "target_p99": float(robust_high),
        "target_robust_range": robust_range,
        "idle_std": idle_std,
        "median_within_session_std": within_std,
        "within_session_to_global_std_ratio": (
            within_std / global_std if global_std > 0 else 0.0
        ),
        "robust_range_to_idle_6sigma": (
            robust_range / (6.0 * idle_std) if idle_std > 0 else float("inf")
        ),
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(true, model))),
        "r2": float(r2_score(true, model)),
        "normalized_mae_robust_range": mae / robust_range if robust_range > 0 else float("nan"),
        "global_median_baseline_mae": global_mae,
        "position_median_baseline_mae": position_mae,
        "skill_over_global_baseline": 1.0 - mae / global_mae if global_mae > 0 else float("nan"),
        "skill_over_position_baseline": 1.0 - mae / position_mae if position_mae > 0 else float("nan"),
        "training_time_sec": fit_seconds,
        "inference_latency_ms_per_frame": float(np.mean(latencies)),
        "fold_metrics": fold_rows,
        "top_feature_importance": [
            {"feature": str(names[index]), "importance": float(mean_importance[index])}
            for index in top_order
            if mean_importance[index] > 0
        ],
    }
    metrics["observability_status"] = classify_observability(metrics)
    return metrics, predicted


def derive_force_phase_labels(
    *,
    force_fz_n: np.ndarray,
    elapsed_time_sec: np.ndarray,
    group_id: np.ndarray,
    no_contact_threshold_n: float = 0.08,
    slope_threshold_n_per_sec: float = 0.30,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate provisional no-contact/loading/hold/release labels from Fz."""

    force = np.asarray(force_fz_n, dtype=float)
    elapsed = np.asarray(elapsed_time_sec, dtype=float)
    groups = np.asarray(group_id).astype(str)
    if force.shape != elapsed.shape or force.shape != groups.shape:
        raise ValueError("force, elapsed time, and group vectors must align")
    labels = np.full(force.shape, "invalid", dtype="U16")
    slope = np.full(force.shape, np.nan, dtype=float)
    smoothed = np.full(force.shape, np.nan, dtype=float)
    width = max(1, int(smoothing_window))
    for group in dict.fromkeys(groups.tolist()):
        selected = np.flatnonzero(groups == group)
        order = selected[np.argsort(elapsed[selected], kind="stable")]
        current_force = force[order]
        current_time = elapsed[order]
        if width > 1:
            current_smoothed = (
                pd.Series(current_force)
                .rolling(width, center=True, min_periods=1)
                .median()
                .to_numpy(dtype=float)
            )
        else:
            current_smoothed = current_force.copy()
        if len(order) > 1 and np.all(np.diff(current_time) > 0):
            current_slope = np.gradient(current_smoothed, current_time)
        else:
            current_slope = np.zeros_like(current_smoothed)
        current_labels = np.full(len(order), "hold", dtype="U16")
        current_labels[current_smoothed <= no_contact_threshold_n] = "no_contact"
        active = current_smoothed > no_contact_threshold_n
        current_labels[active & (current_slope >= slope_threshold_n_per_sec)] = "loading"
        current_labels[active & (current_slope <= -slope_threshold_n_per_sec)] = "release"
        labels[order] = current_labels
        slope[order] = current_slope
        smoothed[order] = current_smoothed
    valid = np.isfinite(force) & np.isfinite(elapsed) & np.isfinite(slope)
    return labels, slope, valid


__all__ = [
    "AlignedMechanicalTargets",
    "MECHANICAL_TARGET_SPECS",
    "MechanicalTargetSpec",
    "classify_observability",
    "derive_force_phase_labels",
    "grouped_regression_observability",
    "load_aligned_mechanical_targets",
    "validate_force_alignment",
]
