"""Benchmark causal spectral classifiers at the measured live SDK cadence.

This is an offline benchmark. It does not open BaySpec hardware, modify the
desktop UI, consume blind-test files, or replace the deployed model.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.factorized_position import (  # noqa: E402
    FactorizedPositionClassifier,
)
from src.hybrid_spectrum.live_cadence_dataset import (  # noqa: E402
    LiveCadenceDataset,
    build_live_cadence_dataset,
    causal_summary_features,
)
from src.hybrid_spectrum.live_cadence_models import (  # noqa: E402
    CumulativeOrdinalClassifier,
    OrdinalRegressionClassifier,
    SpatialCoordinateRegressorClassifier,
    set_single_thread_prediction,
)


CONTACT_ORDER = ["no_contact", "contact"]
POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
RESPONSE_ORDER = ["light", "normal", "hard"]
SUMMARY_MODELS = (
    "extra_trees_summary",
    "lightgbm_summary",
    "svm_rbf_summary",
    "hist_gradient_boosting_summary",
    "logistic_summary",
    "shrinkage_lda_summary",
)
AEON_MODELS = (
    "quant_multivariate",
    "hydra_multivariate",
    "rdst_multivariate",
    "rstsf_multivariate",
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dynamic_sequence_audit_20260714_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-frames", default="1,2,3,4,6")
    parser.add_argument("--source-frame-interval-sec", type=float, default=0.04)
    parser.add_argument("--live-frame-interval-sec", type=float, default=0.40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-aeon", action="store_true")
    return parser.parse_args()


def metrics_for(
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1_values[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "label_order": labels,
    }


def majority_vote_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    keys: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    vote_truth: list[str] = []
    vote_prediction: list[str] = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        true_values, true_counts = np.unique(truth[mask], return_counts=True)
        pred_values, pred_counts = np.unique(predicted[mask], return_counts=True)
        vote_truth.append(str(true_values[int(np.argmax(true_counts))]))
        vote_prediction.append(str(pred_values[int(np.argmax(pred_counts))]))
    output = metrics_for(np.asarray(vote_truth), np.asarray(vote_prediction), labels)
    output["vote_unit_count"] = len(vote_truth)
    return output


def task_specification(dataset: LiveCadenceDataset, task: str) -> dict[str, Any]:
    contact = dataset.contact_labels == "contact"
    if task == "contact":
        return {
            "mask": np.ones(len(dataset.values), dtype=bool),
            "labels": dataset.contact_labels,
            "order": CONTACT_ORDER,
            "vote_keys": np.char.add(
                np.char.add(dataset.file_ids.astype(str), "::"),
                dataset.stage_labels.astype(str),
            ),
        }
    if task == "position":
        return {
            "mask": contact,
            "labels": dataset.position_labels,
            "order": POSITION_ORDER,
            "vote_keys": dataset.file_ids,
        }
    if task == "response_level":
        return {
            "mask": contact,
            "labels": dataset.stage_labels,
            "order": RESPONSE_ORDER,
            "vote_keys": np.char.add(
                np.char.add(dataset.file_ids.astype(str), "::"),
                dataset.stage_labels.astype(str),
            ),
        }
    raise ValueError(f"unknown task: {task}")


def serialized_size_mb(model: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def one_window_latency(model: Any, values: np.ndarray, repeats: int = 40) -> dict[str, float]:
    if not len(values):
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    model.predict(values[:1])
    elapsed: list[float] = []
    for index in range(repeats):
        sample = values[index % len(values) : index % len(values) + 1]
        started = time.perf_counter()
        model.predict(sample)
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean": float(np.mean(elapsed)),
        "p50": float(np.percentile(elapsed, 50.0)),
        "p95": float(np.percentile(elapsed, 95.0)),
    }


def summary_feature_latency(values: np.ndarray, repeats: int = 80) -> dict[str, float]:
    elapsed: list[float] = []
    for index in range(repeats):
        sample = values[index % len(values) : index % len(values) + 1]
        started = time.perf_counter()
        causal_summary_features(sample)
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean": float(np.mean(elapsed)),
        "p50": float(np.percentile(elapsed, 50.0)),
        "p95": float(np.percentile(elapsed, 95.0)),
    }


def make_summary_model(model_id: str, task: str, seed: int) -> Any:
    if model_id == "extra_trees_summary":
        return ExtraTreesClassifier(
            n_estimators=240,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if model_id == "factorized_extra_trees":
        if task != "position":
            raise ValueError("factorized model is position-only")
        estimator = ExtraTreesClassifier(
            n_estimators=240,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        return FactorizedPositionClassifier(estimator)
    if model_id == "lightgbm_summary":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary" if task == "contact" else "multiclass",
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=6,
            min_child_samples=15,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
            random_state=seed,
        )
    if model_id == "svm_rbf_summary":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=4.0,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_id == "hist_gradient_boosting_summary":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    if model_id == "logistic_summary":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_id == "shrinkage_lda_summary":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        )
    if model_id == "ordinal_logistic":
        if task != "response_level":
            raise ValueError("ordinal model is response-level only")
        return CumulativeOrdinalClassifier(random_state=seed)
    if model_id == "ordinal_hist_gradient_boosting":
        if task != "response_level":
            raise ValueError("ordinal model is response-level only")
        return OrdinalRegressionClassifier(
            HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=seed,
            )
        )
    if model_id == "ordinal_lightgbm_regression":
        if task != "response_level":
            raise ValueError("ordinal model is response-level only")
        from lightgbm import LGBMRegressor

        return OrdinalRegressionClassifier(
            LGBMRegressor(
                objective="regression_l1",
                n_estimators=220,
                learning_rate=0.04,
                num_leaves=15,
                max_depth=6,
                min_child_samples=15,
                colsample_bytree=0.75,
                reg_alpha=0.2,
                reg_lambda=1.0,
                n_jobs=-1,
                verbosity=-1,
                random_state=seed,
            )
        )
    if model_id == "coordinate_extra_trees_regression":
        if task != "position":
            raise ValueError("coordinate model is position-only")
        return SpatialCoordinateRegressorClassifier(
            ExtraTreesRegressor(
                n_estimators=240,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            )
        )
    raise ValueError(f"unknown summary model: {model_id}")


def make_aeon_model(model_id: str, history_frames: int, seed: int) -> Any:
    if model_id == "quant_multivariate":
        from aeon.classification.interval_based import QUANTClassifier

        estimator = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
        depth = max(1, min(3, int(np.floor(np.log2(max(2, history_frames))))))
        return QUANTClassifier(
            interval_depth=depth,
            quantile_divisor=2,
            estimator=estimator,
            random_state=seed,
        )
    if model_id == "hydra_multivariate":
        from aeon.classification.convolution_based import HydraClassifier

        return HydraClassifier(
            n_kernels=8,
            n_groups=32,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if model_id == "rdst_multivariate":
        from aeon.classification.shapelet_based import RDSTClassifier

        return RDSTClassifier(
            max_shapelets=300,
            shapelet_lengths=np.arange(2, history_frames + 1),
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if model_id == "rstsf_multivariate":
        from aeon.classification.interval_based import RSTSF

        return RSTSF(
            n_estimators=80,
            n_intervals=8,
            min_interval_length=3,
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"unknown aeon model: {model_id}")


def sequence_scaler(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = np.asarray(train, dtype=np.float32).transpose(0, 2, 1)
    mean = np.mean(channels, axis=(0, 2), keepdims=True)
    scale = np.std(channels, axis=(0, 2), keepdims=True)
    return mean.astype(np.float32), np.where(scale > 1e-8, scale, 1.0).astype(np.float32)


def scale_sequence(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    channels = np.asarray(values, dtype=np.float32).transpose(0, 2, 1)
    return ((channels - mean) / scale).astype(np.float32)


def evaluate_model(
    *,
    dataset: LiveCadenceDataset,
    summary: np.ndarray,
    task: str,
    model_id: str,
    seed: int,
    feature_latency: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specification = task_specification(dataset, task)
    task_mask = specification["mask"]
    labels = specification["labels"]
    label_order = specification["order"]
    all_predictions = np.empty(len(dataset.values), dtype="<U32")
    tested = np.zeros(len(dataset.values), dtype=bool)
    fold_metrics: list[dict[str, Any]] = []
    training_times: list[float] = []
    model_sizes: list[float] = []
    latency_means: list[float] = []
    latency_p50s: list[float] = []
    latency_p95s: list[float] = []
    rows: list[dict[str, Any]] = []

    for fold_index, test_group in enumerate(sorted(set(dataset.capture_groups.tolist()))):
        train = task_mask & (dataset.capture_groups != test_group)
        test = task_mask & (dataset.capture_groups == test_group)
        if set(dataset.file_ids[train]) & set(dataset.file_ids[test]):
            raise AssertionError("original DAT leakage across grouped fold")
        if model_id in AEON_MODELS:
            mean, scale = sequence_scaler(dataset.values[train])
            train_x = scale_sequence(dataset.values[train], mean, scale)
            test_x = scale_sequence(dataset.values[test], mean, scale)
            model = make_aeon_model(model_id, dataset.history_frames, seed + fold_index)
            input_type = "multivariate_physical_frame_sequence"
        else:
            train_x = summary[train]
            test_x = summary[test]
            model = make_summary_model(model_id, task, seed + fold_index)
            input_type = "causal_temporal_summary"
        started = time.perf_counter()
        model.fit(train_x, labels[train])
        training_times.append(time.perf_counter() - started)
        set_single_thread_prediction(model)
        predicted = np.asarray(model.predict(test_x)).astype(str)
        indices = np.flatnonzero(test)
        all_predictions[indices] = predicted
        tested[indices] = True
        fold = metrics_for(labels[test], predicted, label_order)
        fold["test_group"] = str(test_group)
        fold["num_test_windows"] = int(len(indices))
        fold_metrics.append(fold)
        model_sizes.append(serialized_size_mb(model))
        latency = one_window_latency(model, test_x)
        latency_means.append(latency["mean"])
        latency_p50s.append(latency["p50"])
        latency_p95s.append(latency["p95"])
        for index, predicted_label in zip(indices, predicted):
            rows.append(
                {
                    "history_frames": dataset.history_frames,
                    "task": task,
                    "model_id": model_id,
                    "window_index": int(index),
                    "file_id": str(dataset.file_ids[index]),
                    "capture_group": str(dataset.capture_groups[index]),
                    "target_frame_index": int(dataset.target_frame_indices[index]),
                    "true_label": str(labels[index]),
                    "predicted_label": str(predicted_label),
                }
            )
    truth = labels[tested]
    predicted = all_predictions[tested]
    metrics = metrics_for(truth, predicted, label_order)
    voting = majority_vote_metrics(
        truth, predicted, specification["vote_keys"][tested], label_order
    )
    metrics.update(
        {
            "task": task,
            "model_id": model_id,
            "input_type": input_type,
            "input_shape": [dataset.history_frames, len(dataset.feature_names)],
            "history_frames": dataset.history_frames,
            "history_span_sec": dataset.history_span_sec,
            "cold_start_fill_sec": dataset.cold_start_fill_sec,
            "prediction_update_period_sec": dataset.live_frame_interval_sec,
            "num_windows": int(np.sum(task_mask)),
            "num_files": int(len(set(dataset.file_ids[task_mask].tolist()))),
            "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
            "evaluation_validity": "grouped_by_capture_group_and_file_id",
            "random_frame_split": False,
            "grouped_vote_accuracy": float(voting["accuracy"]),
            "grouped_vote_macro_f1": float(voting["macro_f1"]),
            "minimum_capture_group_macro_f1": float(
                min(item["macro_f1"] for item in fold_metrics)
            ),
            "training_time_sec_mean": float(np.mean(training_times)),
            "model_size_mb_mean": float(np.mean(model_sizes)),
            "feature_latency_ms_p50": float(feature_latency["p50"]),
            "feature_latency_ms_p95": float(feature_latency["p95"]),
            "inference_latency_ms_mean": float(np.mean(latency_means)),
            "inference_latency_ms_p50": float(np.mean(latency_p50s)),
            "inference_latency_ms_p95": float(np.max(latency_p95s)),
            "compute_latency_ms_p95": float(
                feature_latency["p95"] + np.max(latency_p95s)
            ),
            "folds": fold_metrics,
        }
    )
    return metrics, rows


def model_ids_for(task: str, history_frames: int, skip_aeon: bool) -> list[str]:
    if task == "contact":
        models = list(SUMMARY_MODELS)
    elif task == "position":
        models = list(SUMMARY_MODELS) + [
            "factorized_extra_trees",
            "coordinate_extra_trees_regression",
        ]
    else:
        models = list(SUMMARY_MODELS) + [
            "ordinal_logistic",
            "ordinal_hist_gradient_boosting",
            "ordinal_lightgbm_regression",
        ]
    if not skip_aeon and history_frames >= 3:
        models.append("quant_multivariate")
    if not skip_aeon and history_frames >= 6:
        models.extend(("hydra_multivariate", "rdst_multivariate", "rstsf_multivariate"))
    return models


def flatten_row(metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in metrics.items()
        if key not in {"per_class", "confusion_matrix", "folds", "label_order"}
    }
    for label, values in metrics.get("per_class", {}).items():
        for metric_name in ("precision", "recall", "f1", "support"):
            row[f"{label}_{metric_name}"] = values[metric_name]
    return row


def save_confusion(path: Path, result: dict[str, Any]) -> None:
    matrix = np.asarray(result["confusion_matrix"])
    labels = result["label_order"]
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(
        f"{result['task']} | {result['model_id']} | {result['history_frames']} physical frames"
    )
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_tradeoff_plot(path: Path, results: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), sharey=False)
    for axis, task in zip(axes, ("contact", "position", "response_level")):
        subset = [row for row in results if row["task"] == task and "missing_reason" not in row]
        model_ids = sorted(set(row["model_id"] for row in subset))
        for model_id in model_ids:
            rows = sorted(
                [row for row in subset if row["model_id"] == model_id],
                key=lambda item: item["cold_start_fill_sec"],
            )
            axis.plot(
                [row["cold_start_fill_sec"] for row in rows],
                [row["macro_f1"] for row in rows],
                marker="o",
                linewidth=1.4,
                label=model_id.replace("_summary", ""),
            )
        axis.set_title(task.replace("_", " ").title())
        axis.set_xlabel("Cold-start history fill (s)")
        axis.set_ylabel("Grouped CV macro-F1")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rank_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for task in ("contact", "position", "response_level"):
        rows = [
            row
            for row in results
            if row["task"] == task and "missing_reason" not in row
        ]
        ranked[task] = sorted(
            rows,
            key=lambda row: (
                -float(row["macro_f1"]),
                -float(row["minimum_capture_group_macro_f1"]),
                float(row["cold_start_fill_sec"]),
                float(row["compute_latency_ms_p95"]),
            ),
        )
    return ranked


def write_report(
    path: Path,
    results: list[dict[str, Any]],
    ranked: dict[str, list[dict[str, Any]]],
    datasets: dict[int, LiveCadenceDataset],
) -> None:
    lines = [
        "# Live-cadence algorithm benchmark",
        "",
        "- Scope: offline algorithm and latency validation only; BaySpec hardware and UI were not opened or modified.",
        "- Data: 27 independent Sense DAT sequences; blind-test files were not used.",
        "- Formal evaluation: leave-one-capture-group-out (G1/G2/G3), with original file_id exclusivity.",
        "- Random frame/window split: prohibited and not used.",
        "- Labels light/normal/hard remain approximate response levels, not force_N.",
        "- Live cadence: real recorded frames selected every 0.40 s from the 0.04 s audit stream; no interpolated spectra are used.",
        "",
        "## Key finding",
        "",
        "The previous 20 x 40 ms training window spans 0.8 s of dense recorded data, but the stable live SDK produces about one physical spectrum every 0.4 s. Filling that history by interpolation does not add measurement information. This benchmark reports the physical-frame history separately from sub-millisecond model compute.",
        "",
        "## Dataset by physical history",
        "",
        "| Frames | Windows | History span | Cold-start fill | Update period |",
        "|---:|---:|---:|---:|---:|",
    ]
    for history, dataset in sorted(datasets.items()):
        lines.append(
            f"| {history} | {len(dataset.values)} | {dataset.history_span_sec:.2f} s | {dataset.cold_start_fill_sec:.2f} s | {dataset.live_frame_interval_sec:.2f} s |"
        )
    lines.extend(["", "## Preliminary leaders", ""])
    for task, rows in ranked.items():
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Rank | Model | Frames | Macro-F1 | Min-group F1 | Vote accuracy | Compute p95 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for index, row in enumerate(rows[:8], start=1):
            lines.append(
                f"| {index} | {row['model_id']} | {row['history_frames']} | {row['macro_f1']:.4f} | {row['minimum_capture_group_macro_f1']:.4f} | {row['grouped_vote_accuracy']:.4f} | {row['compute_latency_ms_p95']:.3f} ms |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is a candidate comparison, not a deployment decision.",
            "- Accuracy based on thousands of windows does not change the fact that there are only 27 independent dynamic files.",
            "- A lower history count reduces cold-start delay, but contact-onset latency also depends on where an action starts relative to the 0.4 s acquisition tick.",
            "- Compute latency excludes the SDK acquisition time; it is reported separately to avoid attributing hardware cadence to the classifier.",
            "- QUANT and HYDRA are included as installed time-series baselines, but their value must be demonstrated on this unusually short physical-frame history rather than assumed from generic archives.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir()
    history_values = sorted(
        {int(value.strip()) for value in args.history_frames.split(",") if value.strip()}
    )
    if not history_values or history_values[0] < 1:
        raise SystemExit("history frames must be positive integers")

    datasets: dict[int, LiveCadenceDataset] = {}
    results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for history_frames in history_values:
        dataset = build_live_cadence_dataset(
            args.audit_dir,
            history_frames,
            source_frame_interval_sec=args.source_frame_interval_sec,
            live_frame_interval_sec=args.live_frame_interval_sec,
        )
        datasets[history_frames] = dataset
        summary = causal_summary_features(dataset.values)
        feature_latency = summary_feature_latency(dataset.values)
        for task in ("contact", "position", "response_level"):
            for model_id in model_ids_for(task, history_frames, args.skip_aeon):
                print(f"[{history_frames} frames] {task}: {model_id}", flush=True)
                try:
                    metrics, rows = evaluate_model(
                        dataset=dataset,
                        summary=summary,
                        task=task,
                        model_id=model_id,
                        seed=args.random_state,
                        feature_latency=(
                            {"mean": 0.0, "p50": 0.0, "p95": 0.0}
                            if model_id in AEON_MODELS
                            else feature_latency
                        ),
                    )
                    results.append(metrics)
                    prediction_rows.extend(rows)
                except Exception as error:
                    results.append(
                        {
                            "task": task,
                            "model_id": model_id,
                            "history_frames": history_frames,
                            "history_span_sec": dataset.history_span_sec,
                            "cold_start_fill_sec": dataset.cold_start_fill_sec,
                            "evaluation_validity": "grouped_by_capture_group_and_file_id",
                            "missing_reason": f"{type(error).__name__}: {error}",
                        }
                    )
                    print(f"  skipped: {type(error).__name__}: {error}", flush=True)

    ranked = rank_results(results)
    for task, rows in ranked.items():
        for rank, result in enumerate(rows[:3], start=1):
            safe_model = result["model_id"].replace("/", "_")
            save_confusion(
                confusion_dir
                / f"{task}_rank{rank}_{safe_model}_{result['history_frames']}frames.png",
                result,
            )
    save_tradeoff_plot(output_dir / "accuracy_latency_tradeoff.png", results)

    flat_rows = [flatten_row(result) for result in results]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with (output_dir / "live_cadence_model_leaderboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    if prediction_rows:
        with (output_dir / "grouped_predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    payload = {
        "scope": "offline_live_cadence_algorithm_benchmark",
        "hardware_opened": False,
        "ui_modified": False,
        "blind_test_used": False,
        "independent_dat_files": 27,
        "source_frame_interval_sec": args.source_frame_interval_sec,
        "live_frame_interval_sec": args.live_frame_interval_sec,
        "history_frames": history_values,
        "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "results": results,
        "top3": {task: rows[:3] for task, rows in ranked.items()},
    }
    (output_dir / "live_cadence_model_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "live_cadence_algorithm_report.md", results, ranked, datasets
    )
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
