"""Train leakage-safe ordinary-FBG position, contact, and Fz candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception as exc:  # pragma: no cover - depends on the local runtime
    LGBMClassifier = None
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    LIGHTGBM_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.px6d_session_dataset import (  # noqa: E402
    POSITION_ORDER,
    load_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ordinary_fbg_px6d_training.yaml"
TASK_ORDER = ("contact", "position", "force")
PLOT_COLORS = ("#1594c4", "#13a47b", "#e0a12d", "#cb665d")
SUPPORTED_MODEL_TYPES = (
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _equal_session_weights(
    session_ids: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Give every independent capture session equal total training mass."""
    selected = session_ids[mask]
    counts = Counter(selected.tolist())
    weights = np.asarray(
        [1.0 / counts[session_id] for session_id in selected],
        dtype=float,
    )
    return weights / max(float(np.mean(weights)), 1.0e-12)


def _classifier_factory(
    model_type: str,
    *,
    estimator_count: int,
    minimum_leaf_samples: int,
    seed: int,
) -> Any:
    common = {
        "n_estimators": estimator_count,
        "min_samples_leaf": minimum_leaf_samples,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    if model_type == "extra_trees":
        return ExtraTreesClassifier(**common)
    if model_type == "random_forest":
        return RandomForestClassifier(**common)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=estimator_count,
            max_leaf_nodes=31,
            min_samples_leaf=max(10, minimum_leaf_samples),
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )
    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError(
                f"lightgbm is unavailable: {LIGHTGBM_IMPORT_ERROR}"
            )
        return LGBMClassifier(
            n_estimators=estimator_count,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=max(10, minimum_leaf_samples),
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"unsupported classifier: {model_type}")


def _regressor_factory(
    model_type: str,
    *,
    estimator_count: int,
    minimum_leaf_samples: int,
    seed: int,
) -> Any:
    common = {
        "n_estimators": estimator_count,
        "min_samples_leaf": minimum_leaf_samples,
        "random_state": seed,
        "n_jobs": -1,
    }
    if model_type == "extra_trees":
        return ExtraTreesRegressor(**common)
    if model_type == "random_forest":
        return RandomForestRegressor(**common)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=estimator_count,
            max_leaf_nodes=31,
            min_samples_leaf=max(10, minimum_leaf_samples),
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    if model_type == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError(
                f"lightgbm is unavailable: {LIGHTGBM_IMPORT_ERROR}"
            )
        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=estimator_count,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=max(10, minimum_leaf_samples),
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"unsupported regressor: {model_type}")


def _resolve_candidate_model_types(
    requested: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, str]]:
    available: list[str] = []
    skipped: dict[str, str] = {}
    for model_type in dict.fromkeys(requested):
        if model_type not in SUPPORTED_MODEL_TYPES:
            skipped[model_type] = "unsupported_model_type"
            continue
        if model_type == "lightgbm" and LGBMClassifier is None:
            skipped[model_type] = (
                "package_not_available: "
                f"{LIGHTGBM_IMPORT_ERROR or 'unknown import error'}"
            )
            continue
        available.append(model_type)
    if not available:
        raise RuntimeError(
            "no requested candidate model is available; "
            f"requested={list(requested)}, skipped={skipped}"
        )
    return tuple(available), skipped


def _measure_latency_ms(
    estimator: Any,
    features: np.ndarray,
) -> float:
    if len(features) == 0:
        return float("nan")
    sample = features[: min(len(features), 4096)]
    _predict_estimator(estimator, sample[: min(16, len(sample))])
    start = time.perf_counter()
    for _ in range(3):
        _predict_estimator(estimator, sample)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / (3 * len(sample))


def _predict_estimator(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Predict without LightGBM's spurious ndarray feature-name warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but "
                "LGBM(?:Classifier|Regressor) was fitted with feature names"
            ),
            category=UserWarning,
        )
        return np.asarray(estimator.predict(features))


def _serialized_size_mb(estimator: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer, compress=3)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def _classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: list[Any],
    label_names: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(label_names)
    }
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            truth,
            predicted,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            truth,
            predicted,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def _session_majority_vote(
    truth: np.ndarray,
    predicted: np.ndarray,
    session_ids: np.ndarray,
    label_order: list[str],
) -> dict[str, Any]:
    rank = {label: index for index, label in enumerate(label_order)}
    rows: list[dict[str, str]] = []
    for session_id in sorted(np.unique(session_ids)):
        selected = session_ids == session_id
        true_values = Counter(truth[selected].tolist())
        predicted_values = Counter(predicted[selected].tolist())
        true_label = sorted(
            true_values,
            key=lambda label: (-true_values[label], rank.get(label, 999)),
        )[0]
        predicted_label = sorted(
            predicted_values,
            key=lambda label: (
                -predicted_values[label],
                rank.get(label, 999),
            ),
        )[0]
        rows.append(
            {
                "session_id": str(session_id),
                "true_label": str(true_label),
                "predicted_label": str(predicted_label),
            }
        )
    vote_truth = np.asarray([row["true_label"] for row in rows], dtype=str)
    vote_predicted = np.asarray(
        [row["predicted_label"] for row in rows],
        dtype=str,
    )
    return {
        "accuracy": float(accuracy_score(vote_truth, vote_predicted)),
        "macro_f1": float(
            f1_score(
                vote_truth,
                vote_predicted,
                labels=label_order,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            vote_truth,
            vote_predicted,
            labels=label_order,
        ).tolist(),
        "rows": rows,
    }


def _plot_confusion(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    values = np.asarray(matrix, dtype=float)
    fig, axis = plt.subplots(figsize=(8.0, 6.6))
    image = axis.imshow(values, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    threshold = float(np.max(values)) * 0.55 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{int(values[row, column])}",
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "#102236",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_force_scatter(
    truth: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    lower = float(min(np.min(truth), np.min(predicted)))
    upper = float(max(np.max(truth), np.max(predicted)))
    fig, axis = plt.subplots(figsize=(7.2, 6.4))
    image = axis.hexbin(
        truth,
        predicted,
        gridsize=55,
        mincnt=1,
        cmap="viridis",
    )
    axis.plot([lower, upper], [lower, upper], color="#cb665d", linewidth=1.5)
    axis.set_xlabel("PX6D reference Fz (N)")
    axis.set_ylabel("Out-of-fold predicted Fz (N)")
    axis.set_title("Continuous Fz Regression, Grouped by Session")
    fig.colorbar(image, ax=axis, label="Frame count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_comparison(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.8))
    display_names = {
        "extra_trees": "ExtraTrees",
        "random_forest": "RF",
        "lightgbm": "LightGBM",
        "hist_gradient_boosting": "HistGB",
    }
    specifications = (
        ("contact", "macro_f1", "Contact macro-F1", False),
        ("position", "macro_f1", "Position macro-F1", False),
        ("force", "mae_n", "Fz MAE (N), lower is better", True),
    )
    for axis, (task, metric, title, lower_is_better) in zip(
        axes,
        specifications,
    ):
        selected = [row for row in rows if row["task"] == task]
        selected.sort(
            key=lambda row: float(row[metric]),
            reverse=not lower_is_better,
        )
        names = [
            f"{display_names.get(row['model_type'], row['model_type'])}\n"
            f"{row['feature_set'].replace('baseline_relative_', '')} features"
            for row in selected
        ]
        values = [float(row[metric]) for row in selected]
        axis.bar(
            range(len(selected)),
            values,
            color=[PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(selected))],
        )
        axis.set_xticks(range(len(selected)), names, rotation=25, ha="right")
        axis.tick_params(axis="x", labelsize=8, pad=4)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if not lower_is_better:
            axis.set_ylim(0.0, 1.02)
        for index, value in enumerate(values):
            axis.text(
                index,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Strict 20260731 New-Data-Only Candidate Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _fit_grouped_classification(
    *,
    task: str,
    model_type: str,
    feature_set: str,
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    session_ids: np.ndarray,
    fold_ids: np.ndarray,
    feature_indices: np.ndarray,
    labels: list[Any],
    label_names: list[str],
    estimator_count: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    if np.issubdtype(target.dtype, np.number):
        predictions = np.full(len(target), -999, dtype=target.dtype)
    else:
        predictions = np.full(len(target), "", dtype=target.dtype)
    latencies: list[float] = []
    training_seconds = 0.0
    fold_metrics: list[dict[str, Any]] = []
    for fold_id in sorted(np.unique(fold_ids)):
        train_mask = mask & (fold_ids != fold_id)
        test_mask = mask & (fold_ids == fold_id)
        estimator = _classifier_factory(
            model_type,
            estimator_count=estimator_count,
            minimum_leaf_samples=minimum_leaf_samples,
            seed=seed + int(fold_id),
        )
        sample_weight = _equal_session_weights(session_ids, train_mask)
        start = time.perf_counter()
        estimator.fit(
            features[train_mask][:, feature_indices],
            target[train_mask],
            sample_weight=sample_weight,
        )
        training_seconds += time.perf_counter() - start
        fold_prediction = _predict_estimator(
            estimator,
            features[test_mask][:, feature_indices]
        )
        predictions[test_mask] = fold_prediction
        latency = _measure_latency_ms(
            estimator,
            features[test_mask][:, feature_indices],
        )
        latencies.append(latency)
        current_metrics = _classification_metrics(
            target[test_mask],
            fold_prediction,
            labels,
            label_names,
        )
        current_metrics["fold_id"] = int(fold_id)
        current_metrics["test_session_count"] = int(
            len(np.unique(session_ids[test_mask]))
        )
        fold_metrics.append(current_metrics)

    truth = target[mask]
    predicted = predictions[mask]
    metrics = _classification_metrics(
        truth,
        predicted,
        labels,
        label_names,
    )
    metrics.update(
        {
            "task": task,
            "model_type": model_type,
            "feature_set": feature_set,
            "evaluation_validity": "grouped_by_session_id",
            "training_weight_strategy": "equal_session_mass",
            "training_time_sec": training_seconds,
            "inference_latency_ms_per_frame": float(np.mean(latencies)),
            "fold_metrics": fold_metrics,
            "frame_count": int(np.sum(mask)),
            "session_count": int(len(np.unique(session_ids[mask]))),
        }
    )
    if task == "contact":
        metrics["active_contact_recall"] = float(
            recall_score(truth, predicted, pos_label=1, zero_division=0)
        )
        contact_matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        false_positive = contact_matrix[0, 1]
        no_contact_total = float(np.sum(contact_matrix[0]))
        metrics["no_contact_false_positive_rate"] = float(
            false_positive / max(no_contact_total, 1.0)
        )
    if task == "position":
        metrics["session_majority_voting"] = _session_majority_vote(
            truth.astype(str),
            predicted.astype(str),
            session_ids[mask],
            label_names,
        )
    return metrics, predictions


def _fit_grouped_force(
    *,
    model_type: str,
    feature_set: str,
    features: np.ndarray,
    force: np.ndarray,
    mask: np.ndarray,
    session_ids: np.ndarray,
    source_positions: np.ndarray,
    fold_ids: np.ndarray,
    feature_indices: np.ndarray,
    estimator_count: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = np.full(len(force), np.nan, dtype=float)
    latencies: list[float] = []
    training_seconds = 0.0
    fold_metrics: list[dict[str, Any]] = []
    for fold_id in sorted(np.unique(fold_ids)):
        train_mask = mask & (fold_ids != fold_id)
        test_mask = mask & (fold_ids == fold_id)
        estimator = _regressor_factory(
            model_type,
            estimator_count=estimator_count,
            minimum_leaf_samples=minimum_leaf_samples,
            seed=seed + int(fold_id),
        )
        sample_weight = _equal_session_weights(session_ids, train_mask)
        start = time.perf_counter()
        estimator.fit(
            features[train_mask][:, feature_indices],
            force[train_mask],
            sample_weight=sample_weight,
        )
        training_seconds += time.perf_counter() - start
        fold_prediction = _predict_estimator(
            estimator,
            features[test_mask][:, feature_indices]
        )
        predictions[test_mask] = fold_prediction
        latencies.append(
            _measure_latency_ms(
                estimator,
                features[test_mask][:, feature_indices],
            )
        )
        fold_metrics.append(
            {
                "fold_id": int(fold_id),
                "mae_n": float(
                    mean_absolute_error(force[test_mask], fold_prediction)
                ),
                "rmse_n": float(
                    np.sqrt(
                        mean_squared_error(force[test_mask], fold_prediction)
                    )
                ),
                "r2": float(r2_score(force[test_mask], fold_prediction)),
                "test_session_count": int(
                    len(np.unique(session_ids[test_mask]))
                ),
            }
        )

    truth = force[mask]
    predicted = predictions[mask]
    error = predicted - truth
    active = truth >= 0.10
    per_position = {}
    for position in sorted(np.unique(source_positions[mask])):
        selected = mask & (source_positions == position)
        per_position[str(position)] = {
            "mae_n": float(
                mean_absolute_error(force[selected], predictions[selected])
            ),
            "frame_count": int(np.sum(selected)),
            "session_count": int(len(np.unique(session_ids[selected]))),
        }
    slope, intercept = np.polyfit(truth, predicted, deg=1)
    metrics = {
        "task": "force",
        "model_type": model_type,
        "feature_set": feature_set,
        "evaluation_validity": "grouped_by_session_id",
        "training_weight_strategy": "equal_session_mass",
        "mae_n": float(mean_absolute_error(truth, predicted)),
        "median_absolute_error_n": float(
            median_absolute_error(truth, predicted)
        ),
        "rmse_n": float(np.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)),
        "bias_n": float(np.mean(error)),
        "active_force_mae_n": float(
            mean_absolute_error(truth[active], predicted[active])
        ),
        "within_0_25_n": float(np.mean(np.abs(error) <= 0.25)),
        "within_0_50_n": float(np.mean(np.abs(error) <= 0.50)),
        "within_1_00_n": float(np.mean(np.abs(error) <= 1.00)),
        "calibration_slope": float(slope),
        "calibration_intercept_n": float(intercept),
        "per_position": per_position,
        "training_time_sec": training_seconds,
        "inference_latency_ms_per_frame": float(np.mean(latencies)),
        "fold_metrics": fold_metrics,
        "frame_count": int(np.sum(mask)),
        "active_frame_count": int(np.sum(active)),
        "session_count": int(len(np.unique(session_ids[mask]))),
    }
    return metrics, predictions


def _select_best(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    selected = [row for row in rows if row["task"] == task]
    if task == "contact":
        return max(
            selected,
            key=lambda row: (
                float(row["macro_f1"]),
                float(row["active_contact_recall"]),
                -float(row["inference_latency_ms_per_frame"]),
            ),
        )
    if task == "position":
        return max(
            selected,
            key=lambda row: (
                float(row["macro_f1"]),
                float(row["session_majority_voting"]["accuracy"]),
                -float(row["inference_latency_ms_per_frame"]),
            ),
        )
    return min(
        selected,
        key=lambda row: (
            float(row["mae_n"]),
            float(row["rmse_n"]),
            float(row["inference_latency_ms_per_frame"]),
        ),
    )


def _fit_final_model(
    *,
    task: str,
    selected: Mapping[str, Any],
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    session_ids: np.ndarray,
    feature_indices: np.ndarray,
    estimator_count: int,
    minimum_leaf_samples: int,
    seed: int,
) -> Any:
    if task == "force":
        estimator = _regressor_factory(
            str(selected["model_type"]),
            estimator_count=estimator_count,
            minimum_leaf_samples=minimum_leaf_samples,
            seed=seed,
        )
    else:
        estimator = _classifier_factory(
            str(selected["model_type"]),
            estimator_count=estimator_count,
            minimum_leaf_samples=minimum_leaf_samples,
            seed=seed,
        )
    estimator.fit(
        features[mask][:, feature_indices],
        target[mask],
        sample_weight=_equal_session_weights(session_ids, mask),
    )
    return estimator


def _evaluate_independent_challenge(
    *,
    challenge_dataset_dir: Path,
    fitted_models: Mapping[str, Mapping[str, Any]],
    primary_session_ids: np.ndarray,
    primary_feature_names: np.ndarray,
    primary_wavelength_nm: np.ndarray,
    position_labels: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Audit quarantined earlier sessions without fitting on them."""
    challenge_manifest = json.loads(
        (
            challenge_dataset_dir
            / "ordinary_fbg_px6d_dataset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if challenge_manifest.get("selection_role") != "challenge":
        raise ValueError(
            "earlier-session dataset is not marked as isolated challenge data"
        )
    challenge = np.load(
        challenge_dataset_dir / "ordinary_fbg_px6d_dataset.npz",
        allow_pickle=False,
    )
    if not np.array_equal(
        challenge["feature_names"].astype(str),
        primary_feature_names.astype(str),
    ):
        raise ValueError(
            "primary and quarantine feature-name schemas do not match"
        )
    if not np.allclose(
        challenge["wavelength_nm"].astype(float),
        primary_wavelength_nm.astype(float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "primary and quarantine wavelength axes do not match"
        )

    features = challenge["features"]
    session_ids = challenge["session_id"].astype(str)
    primary_ids = set(primary_session_ids.astype(str).tolist())
    overlap = sorted(primary_ids.intersection(session_ids.tolist()))
    if overlap:
        raise ValueError(
            "primary and quarantine datasets overlap by session_id: "
            + ", ".join(overlap)
        )
    if np.any(challenge["fold_id"].astype(int) != -1):
        raise ValueError(
            "quarantine dataset must use fold_id=-1 for all frames"
        )

    contact = challenge["contact_target"].astype(int)
    position = challenge["position_target"].astype(str)
    force = challenge["force_fz_n"].astype(float)
    contact_mask = challenge["contact_training_mask"].astype(bool)
    position_mask = challenge["position_training_mask"].astype(bool)
    force_mask = challenge["force_training_mask"].astype(bool)
    capture_index = challenge["capture_index"].astype(int)
    session_to_position = {
        str(row["session_id"]): (
            "no_contact"
            if str(row["position_label"]) == "unlabeled"
            else str(row["position_label"])
        )
        for row in challenge_manifest["sessions"]
    }
    source_positions = np.asarray(
        [session_to_position[session_id] for session_id in session_ids],
        dtype="<U16",
    )

    contact_model = fitted_models["contact"]
    contact_features = np.asarray(
        contact_model["feature_indices"], dtype=int
    )
    contact_prediction = _predict_estimator(
        contact_model["estimator"],
        features[contact_mask][:, contact_features]
    )
    contact_metrics = _classification_metrics(
        contact[contact_mask],
        contact_prediction,
        [0, 1],
        ["no_contact", "contact"],
    )
    contact_matrix = np.asarray(
        contact_metrics["confusion_matrix"], dtype=float
    )
    contact_metrics.update(
        {
            "evaluation_validity": (
                "diagnostic_only_suspected_label_position_error"
            ),
            "frame_count": int(np.sum(contact_mask)),
            "session_count": int(len(np.unique(session_ids[contact_mask]))),
            "active_contact_recall": float(
                recall_score(
                    contact[contact_mask],
                    contact_prediction,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "no_contact_false_positive_rate": float(
                contact_matrix[0, 1]
                / max(float(np.sum(contact_matrix[0])), 1.0)
            ),
            "inference_latency_ms_per_frame": _measure_latency_ms(
                contact_model["estimator"],
                features[contact_mask][:, contact_features],
            ),
        }
    )

    position_model = fitted_models["position"]
    position_features = np.asarray(
        position_model["feature_indices"], dtype=int
    )
    position_prediction = _predict_estimator(
        position_model["estimator"],
        features[position_mask][:, position_features]
    ).astype(str)
    observed_position_labels = [
        label
        for label in position_labels
        if np.any(position[position_mask] == label)
    ]
    position_metrics = _classification_metrics(
        position[position_mask],
        position_prediction,
        observed_position_labels,
        observed_position_labels,
    )
    position_metrics.update(
        {
            "evaluation_validity": (
                "diagnostic_only_suspected_label_position_error"
            ),
            "evaluation_scope": observed_position_labels,
            "frame_count": int(np.sum(position_mask)),
            "session_count": int(len(np.unique(session_ids[position_mask]))),
            "predicted_outside_challenge_scope_count": int(
                np.sum(~np.isin(position_prediction, observed_position_labels))
            ),
            "session_majority_voting": _session_majority_vote(
                position[position_mask],
                position_prediction,
                session_ids[position_mask],
                observed_position_labels,
            ),
            "inference_latency_ms_per_frame": _measure_latency_ms(
                position_model["estimator"],
                features[position_mask][:, position_features],
            ),
        }
    )

    force_model = fitted_models["force"]
    force_features = np.asarray(force_model["feature_indices"], dtype=int)
    force_prediction = _predict_estimator(
        force_model["estimator"],
        features[force_mask][:, force_features]
    )
    force_truth = force[force_mask]
    force_error = force_prediction - force_truth
    active_force = force_truth >= 0.10
    force_per_position = {}
    for source_position in sorted(np.unique(source_positions[force_mask])):
        selected = force_mask & (source_positions == source_position)
        selected_prediction = _predict_estimator(
            force_model["estimator"],
            features[selected][:, force_features]
        )
        force_per_position[str(source_position)] = {
            "mae_n": float(
                mean_absolute_error(force[selected], selected_prediction)
            ),
            "frame_count": int(np.sum(selected)),
            "session_count": int(len(np.unique(session_ids[selected]))),
        }
    force_metrics = {
        "evaluation_validity": (
            "diagnostic_only_suspected_label_position_error"
        ),
        "mae_n": float(mean_absolute_error(force_truth, force_prediction)),
        "median_absolute_error_n": float(
            median_absolute_error(force_truth, force_prediction)
        ),
        "rmse_n": float(
            np.sqrt(mean_squared_error(force_truth, force_prediction))
        ),
        "r2": float(r2_score(force_truth, force_prediction)),
        "bias_n": float(np.mean(force_error)),
        "active_force_mae_n": float(
            mean_absolute_error(
                force_truth[active_force],
                force_prediction[active_force],
            )
        ),
        "within_0_25_n": float(np.mean(np.abs(force_error) <= 0.25)),
        "within_0_50_n": float(np.mean(np.abs(force_error) <= 0.50)),
        "within_1_00_n": float(np.mean(np.abs(force_error) <= 1.00)),
        "per_position": force_per_position,
        "frame_count": int(np.sum(force_mask)),
        "active_frame_count": int(np.sum(active_force)),
        "session_count": int(len(np.unique(session_ids[force_mask]))),
        "inference_latency_ms_per_frame": _measure_latency_ms(
            force_model["estimator"],
            features[force_mask][:, force_features],
        ),
    }

    frame_predictions = pd.DataFrame(
        {
            "session_id": session_ids,
            "capture_index": capture_index,
            "source_position": source_positions,
            "force_fz_n": force,
            "contact_true": np.where(contact_mask, contact, np.nan),
            "position_true": np.where(position_mask, position, ""),
        }
    )
    frame_predictions["contact_predicted"] = ""
    frame_predictions.loc[contact_mask, "contact_predicted"] = (
        contact_prediction.astype(str)
    )
    frame_predictions["position_predicted"] = ""
    frame_predictions.loc[position_mask, "position_predicted"] = (
        position_prediction
    )
    frame_predictions["force_fz_predicted_n"] = np.nan
    frame_predictions.loc[force_mask, "force_fz_predicted_n"] = (
        force_prediction
    )
    frame_predictions.to_csv(
        output_dir / "quarantined_earlier_session_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _plot_confusion(
        contact_metrics["confusion_matrix"],
        ["no_contact", "contact"],
        "Contact Diagnostic, Quarantined Earlier Sessions",
        output_dir / "quarantine_contact_confusion_matrix.png",
    )
    _plot_confusion(
        position_metrics["confusion_matrix"],
        observed_position_labels,
        "Position Diagnostic, Quarantined Earlier Sessions",
        output_dir / "quarantine_position_confusion_matrix.png",
    )
    _plot_force_scatter(
        force_truth,
        force_prediction,
        output_dir / "quarantine_force_scatter.png",
    )
    return {
        "dataset_id": challenge_manifest["dataset_id"],
        "batch_content_sha256": challenge_manifest[
            "batch_content_sha256"
        ],
        "session_count": int(challenge_manifest["session_count"]),
        "frame_count": int(challenge_manifest["frame_count"]),
        "session_overlap_with_primary": 0,
        "reference_validity": (
            "diagnostic_only_suspected_label_position_error"
        ),
        "position_scope": observed_position_labels,
        "contact": contact_metrics,
        "position": position_metrics,
        "force": force_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ordinary-FBG full-spectrum candidates with strict "
            "session-grouped cross-validation."
        )
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--challenge-dataset-dir",
        type=Path,
        help=(
            "Optional isolated earlier-session quarantine. It is audited only "
            "after final primary models are fitted and is not a valid test set."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    challenge_dataset_dir = (
        args.challenge_dataset_dir.expanduser().resolve()
        if args.challenge_dataset_dir
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if PROJECT_ROOT / "models" in output_dir.parents:
        raise ValueError("candidate training output may not overwrite app models")

    manifest = json.loads(
        (dataset_dir / "ordinary_fbg_px6d_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("historical_data_included") is not False:
        raise ValueError("dataset manifest does not prove historical-data isolation")
    if manifest.get("selection_role") != "primary":
        raise ValueError("formal training requires a primary-role dataset")
    dataset = np.load(
        dataset_dir / "ordinary_fbg_px6d_dataset.npz",
        allow_pickle=False,
    )
    config = load_config(config_path)
    model_config = dict(config.get("models") or {})
    evaluation_config = dict(config.get("evaluation") or {})
    seed = int(evaluation_config.get("random_seed", 42))
    cross_validation_estimator_count = int(
        model_config.get(
            "cross_validation_tree_estimators",
            model_config.get("tree_estimators", 200),
        )
    )
    final_estimator_count = int(
        model_config.get(
            "final_tree_estimators",
            cross_validation_estimator_count,
        )
    )
    requested_candidate_model_types = tuple(
        str(model_type)
        for model_type in model_config.get("candidate_model_types")
        or ("extra_trees", "random_forest")
    )
    candidate_model_types, skipped_candidate_models = (
        _resolve_candidate_model_types(requested_candidate_model_types)
    )
    minimum_leaf_samples = int(
        model_config.get("minimum_leaf_samples", 2)
    )

    features = dataset["features"]
    force = dataset["force_fz_n"].astype(float)
    contact = dataset["contact_target"].astype(int)
    position = dataset["position_target"].astype(str)
    session_ids = dataset["session_id"].astype(str)
    fold_ids = dataset["fold_id"].astype(int)
    force_mask = dataset["force_training_mask"].astype(bool)
    contact_mask = dataset["contact_training_mask"].astype(bool)
    position_mask = dataset["position_training_mask"].astype(bool)
    capture_index = dataset["capture_index"].astype(int)
    release_excluded = dataset["release_tail_excluded"].astype(bool)
    session_to_position = {
        str(row["session_id"]): (
            "no_contact"
            if str(row["position_label"]) == "unlabeled"
            else str(row["position_label"])
        )
        for row in manifest["sessions"]
    }
    source_positions = np.asarray(
        [session_to_position[session_id] for session_id in session_ids],
        dtype="<U16",
    )
    feature_sets = {
        name: np.asarray(indices, dtype=int)
        for name, indices in manifest["feature_sets"].items()
    }
    position_labels = [
        label for label in POSITION_ORDER if np.any(position == label)
    ]

    rows: list[dict[str, Any]] = []
    predictions_by_id: dict[str, np.ndarray] = {}
    for feature_set, feature_indices in feature_sets.items():
        for model_type in candidate_model_types:
            contact_metrics, contact_prediction = (
                _fit_grouped_classification(
                    task="contact",
                    model_type=model_type,
                    feature_set=feature_set,
                    features=features,
                    target=contact,
                    mask=contact_mask,
                    session_ids=session_ids,
                    fold_ids=fold_ids,
                    feature_indices=feature_indices,
                    labels=[0, 1],
                    label_names=["no_contact", "contact"],
                    estimator_count=cross_validation_estimator_count,
                    minimum_leaf_samples=minimum_leaf_samples,
                    seed=seed + 100,
                )
            )
            contact_id = f"contact_{model_type}_{feature_set}"
            contact_metrics["model_id"] = contact_id
            rows.append(contact_metrics)
            predictions_by_id[contact_id] = contact_prediction

            position_metrics, position_prediction = (
                _fit_grouped_classification(
                    task="position",
                    model_type=model_type,
                    feature_set=feature_set,
                    features=features,
                    target=position,
                    mask=position_mask,
                    session_ids=session_ids,
                    fold_ids=fold_ids,
                    feature_indices=feature_indices,
                    labels=position_labels,
                    label_names=position_labels,
                    estimator_count=cross_validation_estimator_count,
                    minimum_leaf_samples=minimum_leaf_samples,
                    seed=seed + 200,
                )
            )
            position_id = f"position_{model_type}_{feature_set}"
            position_metrics["model_id"] = position_id
            rows.append(position_metrics)
            predictions_by_id[position_id] = position_prediction

            force_metrics, force_prediction = _fit_grouped_force(
                model_type=model_type,
                feature_set=feature_set,
                features=features,
                force=force,
                mask=force_mask,
                session_ids=session_ids,
                source_positions=source_positions,
                fold_ids=fold_ids,
                feature_indices=feature_indices,
                estimator_count=cross_validation_estimator_count,
                minimum_leaf_samples=minimum_leaf_samples,
                seed=seed + 300,
            )
            force_id = f"force_{model_type}_{feature_set}"
            force_metrics["model_id"] = force_id
            rows.append(force_metrics)
            predictions_by_id[force_id] = force_prediction

    best = {task: _select_best(rows, task) for task in TASK_ORDER}
    task_targets = {
        "contact": contact,
        "position": position,
        "force": force,
    }
    task_masks = {
        "contact": contact_mask,
        "position": position_mask,
        "force": force_mask,
    }
    fitted_models = {}
    for index, task in enumerate(TASK_ORDER):
        selected = best[task]
        indices = feature_sets[str(selected["feature_set"])]
        model = _fit_final_model(
            task=task,
            selected=selected,
            features=features,
            target=task_targets[task],
            mask=task_masks[task],
            session_ids=session_ids,
            feature_indices=indices,
            estimator_count=final_estimator_count,
            minimum_leaf_samples=minimum_leaf_samples,
            seed=seed + 1000 + index,
        )
        selected["final_model_size_mb"] = _serialized_size_mb(model)
        fitted_models[task] = {
            "model_id": selected["model_id"],
            "feature_set": selected["feature_set"],
            "feature_indices": indices,
            "estimator": model,
        }

    challenge_metrics = None
    challenge_manifest = None
    if challenge_dataset_dir is not None:
        challenge_metrics = _evaluate_independent_challenge(
            challenge_dataset_dir=challenge_dataset_dir,
            fitted_models=fitted_models,
            primary_session_ids=session_ids,
            primary_feature_names=dataset["feature_names"],
            primary_wavelength_nm=dataset["wavelength_nm"],
            position_labels=position_labels,
            output_dir=output_dir,
        )
        challenge_manifest = json.loads(
            (
                challenge_dataset_dir
                / "ordinary_fbg_px6d_dataset_manifest.json"
            ).read_text(encoding="utf-8")
        )

    selection_rows: list[dict[str, Any]] = []
    for role, role_manifest in (
        ("primary", manifest),
        ("quarantine", challenge_manifest),
    ):
        if role_manifest is None:
            continue
        for session in role_manifest["sessions"]:
            selection_rows.append(
                {
                    "session_id": session["session_id"],
                    "position_label": session["position_label"],
                    "started_at_epoch_sec": session[
                        "started_at_epoch_sec"
                    ],
                    "selection_role": role,
                    "included_in_formal_training": role == "primary",
                    "fold_id": session["fold_id"],
                    "frame_count": session["frame_count"],
                    "qa_status": session["qa_status"],
                    "finding_codes": "|".join(session["finding_codes"]),
                    "session_directory": session["session_directory"],
                }
            )
    pd.DataFrame(selection_rows).sort_values(
        ["position_label", "started_at_epoch_sec", "session_id"]
    ).to_csv(
        output_dir / "session_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bundle = {
        "schema_version": "ordinary_fbg_px6d_candidate_bundle_v1",
        "deployment_status": "candidate_not_deployed_pending_live_validation",
        "dataset_id": manifest["dataset_id"],
        "batch_content_sha256": manifest["batch_content_sha256"],
        "historical_data_included": False,
        "formal_split_strategy": "grouped_by_session_id",
        "force_semantics": "continuous_PX6D_reference_Fz_N",
        "position_order": position_labels,
        "wavelength_nm": dataset["wavelength_nm"],
        "feature_names": dataset["feature_names"].astype(str),
        "models": fitted_models,
        "candidate_models_requested": requested_candidate_model_types,
        "candidate_models_evaluated": candidate_model_types,
        "skipped_candidate_models": skipped_candidate_models,
        "cross_validation_tree_estimators": (
            cross_validation_estimator_count
        ),
        "final_tree_estimators": final_estimator_count,
        "baseline_contract": dict(config.get("baseline") or {}),
        "label_contract": dict(config.get("labels") or {}),
    }
    bundle_path = output_dir / "ordinary_fbg_px6d_candidate_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    oof = pd.DataFrame(
        {
            "session_id": session_ids,
            "capture_index": capture_index,
            "fold_id": fold_ids,
            "source_position": source_positions,
            "force_fz_n": force,
            "contact_true": np.where(contact_mask, contact, np.nan),
            "position_true": np.where(position_mask, position, ""),
            "release_tail_excluded": release_excluded,
        }
    )
    best_contact_prediction = predictions_by_id[best["contact"]["model_id"]]
    best_position_prediction = predictions_by_id[best["position"]["model_id"]]
    best_force_prediction = predictions_by_id[best["force"]["model_id"]]
    oof["contact_predicted"] = np.where(
        contact_mask,
        best_contact_prediction,
        "",
    )
    oof["position_predicted"] = np.where(
        position_mask,
        best_position_prediction,
        "",
    )
    oof["force_fz_predicted_n"] = best_force_prediction
    oof.to_csv(
        output_dir / "grouped_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    leaderboard_rows = []
    for row in rows:
        leaderboard_rows.append(
            {
                "task": row["task"],
                "model_id": row["model_id"],
                "model_type": row["model_type"],
                "feature_set": row["feature_set"],
                "evaluation_validity": row["evaluation_validity"],
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "session_voting_accuracy": (
                    row.get("session_majority_voting") or {}
                ).get("accuracy"),
                "mae_n": row.get("mae_n"),
                "rmse_n": row.get("rmse_n"),
                "r2": row.get("r2"),
                "active_force_mae_n": row.get("active_force_mae_n"),
                "training_time_sec": row["training_time_sec"],
                "inference_latency_ms_per_frame": row[
                    "inference_latency_ms_per_frame"
                ],
                "selected": row["model_id"] == best[row["task"]]["model_id"],
            }
        )
    pd.DataFrame(leaderboard_rows).to_csv(
        output_dir / "candidate_model_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_payload = {
        "schema_version": "ordinary_fbg_px6d_training_metrics_v1",
        "dataset_id": manifest["dataset_id"],
        "batch_content_sha256": manifest["batch_content_sha256"],
        "historical_data_included": False,
        "independent_session_count": manifest["session_count"],
        "frame_count": manifest["frame_count"],
        "formal_split_strategy": "grouped_by_session_id",
        "random_frame_split_used": False,
        "selection_rule": manifest.get("selection_rule"),
        "release_tail_excluded_frames": manifest[
            "release_tail_excluded_frames"
        ],
        "candidate_only": True,
        "deployment_status": "candidate_not_deployed_pending_live_validation",
        "candidate_models_requested": requested_candidate_model_types,
        "candidate_models_evaluated": candidate_model_types,
        "skipped_candidate_models": skipped_candidate_models,
        "all_candidates": rows,
        "selected_candidates": {
            task: best[task] for task in TASK_ORDER
        },
        "quarantined_earlier_sessions_diagnostic": challenge_metrics,
        "bundle_path": str(bundle_path),
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(
            metrics_payload,
            ensure_ascii=False,
            indent=2,
            default=_json_value,
        )
        + "\n",
        encoding="utf-8",
    )

    _plot_confusion(
        best["contact"]["confusion_matrix"],
        ["no_contact", "contact"],
        "Contact Detection, Grouped Out-of-Fold",
        output_dir / "contact_confusion_matrix.png",
    )
    _plot_confusion(
        best["position"]["confusion_matrix"],
        position_labels,
        "Nine-Position Recognition, Grouped Out-of-Fold",
        output_dir / "position_confusion_matrix.png",
    )
    _plot_confusion(
        best["position"]["session_majority_voting"]["confusion_matrix"],
        position_labels,
        "Session-Level Position Voting",
        output_dir / "position_session_voting_confusion_matrix.png",
    )
    _plot_force_scatter(
        force[force_mask],
        best_force_prediction[force_mask],
        output_dir / "force_oof_scatter.png",
    )
    _plot_comparison(rows, output_dir / "candidate_model_comparison.png")

    isolation_report = [
        "# Dataset Isolation Report",
        "",
        f"- Dataset ID: `{manifest['dataset_id']}`",
        f"- Source root: `{manifest['source_root']}`",
        "- Historical datasets included: no.",
        f"- Independent sessions: {manifest['session_count']}.",
        f"- Synchronized frames: {manifest['frame_count']}.",
        f"- Batch SHA-256: `{manifest['batch_content_sha256']}`.",
        "- Selection rule: `latest_n_by_position` "
        "(P11=5, P12=5, P21=5).",
        "- Latest P11/P12/P21 sessions enter formal training; their earlier "
        "sessions are quarantined for manual review.",
        "- Formal split: grouped by `session_id`.",
        "- Random frame split: prohibited and not used.",
        "- Source inventory covers only metadata, aligned Fz summaries, and "
        "full spectra under the strict `new data` root.",
    ]
    if challenge_metrics is not None:
        isolation_report.extend(
            [
                f"- Quarantined earlier P11/P12/P21 sessions: "
                f"{challenge_metrics['session_count']} sessions.",
                "- Primary/quarantine session overlap: 0.",
                "- Quarantined sessions never entered cross-validation or "
                "final model fitting.",
                "- Their position labels or physical press locations are "
                "suspected to be incorrect, so their metrics are diagnostic "
                "only and cannot support generalization claims.",
            ]
        )
    (output_dir / "dataset_isolation_report.md").write_text(
        "\n".join(isolation_report) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Ordinary-FBG + PX6D Strict-Batch Training Report",
        "",
        "## Scope",
        "",
        "- This run uses only the force-referenced sessions selected by the "
        "current strict-batch dataset manifest.",
        "- Historical static, temporal, blind-test, and legacy model data are "
        "not included.",
        f"- {manifest['session_count']} independent sessions and "
        f"{manifest['frame_count']} synchronized frames were available.",
        "- Frames are repeated measurements within sessions, not independent "
        "experiments.",
        "- Every formal fold holds out complete `session_id` values.",
        "- Candidate families evaluated: "
        f"{', '.join(candidate_model_types)}.",
        "- This run uses every QA-eligible, force-referenced session selected "
        "by the dataset manifest. Captures without Fz are excluded from this "
        "strict track and are handled only by the fusion track.",
        f"- {manifest['release_tail_excluded_frames']} warning-tail frames were "
        "excluded from formal training.",
        "",
        "## Selected Candidates",
        "",
        f"- Contact: `{best['contact']['model_id']}`, accuracy "
        f"{best['contact']['accuracy']:.4f}, macro-F1 "
        f"{best['contact']['macro_f1']:.4f}, active recall "
        f"{best['contact']['active_contact_recall']:.4f}.",
        f"- Position: `{best['position']['model_id']}`, accuracy "
        f"{best['position']['accuracy']:.4f}, macro-F1 "
        f"{best['position']['macro_f1']:.4f}, session voting accuracy "
        f"{best['position']['session_majority_voting']['accuracy']:.4f}.",
        f"- Continuous Fz: `{best['force']['model_id']}`, MAE "
        f"{best['force']['mae_n']:.4f} N, RMSE "
        f"{best['force']['rmse_n']:.4f} N, R2 "
        f"{best['force']['r2']:.4f}, active MAE "
        f"{best['force']['active_force_mae_n']:.4f} N.",
    ]
    if skipped_candidate_models:
        report.extend(
            [
                "",
                "## Skipped Candidates",
                "",
                *[
                    f"- `{name}`: {reason}."
                    for name, reason in skipped_candidate_models.items()
                ],
            ]
        )
    if challenge_metrics is not None:
        report.extend(
            [
                "",
                "## Earlier-Session Quarantine Audit",
                "",
                "- These earlier P11, P12, and P21 sessions are suspected to "
                "contain position-selection or labeling errors.",
                "- They are not an independent test set and do not enter "
                "candidate ranking or deployment decisions.",
                f"- Diagnostic contact macro-F1: "
                f"{challenge_metrics['contact']['macro_f1']:.4f}.",
                f"- Diagnostic three-position accuracy: "
                f"{challenge_metrics['position']['accuracy']:.4f}; "
                f"macro-F1: "
                f"{challenge_metrics['position']['macro_f1']:.4f}.",
                f"- Diagnostic continuous Fz MAE: "
                f"{challenge_metrics['force']['mae_n']:.4f} N; "
                f"R2: {challenge_metrics['force']['r2']:.4f}.",
                "- The poor, systematic position mismatches support manual "
                "label and press-location review rather than model scoring.",
            ]
        )
    report.extend(
        [
        "",
        "## Deployment Boundary",
        "",
        "- These are candidate models only.",
        "- No active TOUCH model or application configuration was overwritten.",
        "- Live baseline handling, end-to-end latency, release recovery, and "
        "independent-day validation are still required before deployment.",
        "- PX6D Fz is the reference target; the optical model is not yet a "
        "calibrated production force sensor.",
        ]
    )
    (output_dir / "training_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "dataset_id": manifest["dataset_id"],
                "output_dir": str(output_dir),
                "bundle_path": str(bundle_path),
                "selected": {
                    task: best[task]["model_id"] for task in TASK_ORDER
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
