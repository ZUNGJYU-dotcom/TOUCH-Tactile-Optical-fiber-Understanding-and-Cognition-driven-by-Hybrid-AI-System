"""Compare tree, kernel, ROCKET, and 1D-CNN models on static Sense spectra.

This script intentionally separates wavelength-axis sequence learning from
true temporal learning.  The current CSV files are independent 512-point
spectral snapshots, so temporal TCN/CNN-LSTM models are registered as pending
until continuous labelled frame sequences are collected.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.sequence_models import (  # noqa: E402
    ChannelStandardizer,
    SmallSpectral1DCNN,
    build_spectral_multiview_data,
    torch_available,
)
from src.hybrid_spectrum.spatial_fingerprint import (  # noqa: E402
    build_spatial_fingerprint_matrix,
)
from src.hybrid_spectrum.sense_static_dataset import (  # noqa: E402
    assert_dataset_manifest_stable,
    build_static_feature_dataset,
    dataset_source_manifest,
    load_sense_dataset,
    load_training_config,
)


POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
FORCE_ORDER = ["light", "normal", "hard"]
CONTACT_ORDER = ["no_contact", "contact"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "sense_static_training.yaml",
    )
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=ROOT / "config" / "hybrid_spectrum_channels.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incremental-date", type=date.fromisoformat, default=date(2026, 7, 14))
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--minirocket-kernels", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-minirocket", action="store_true")
    return parser.parse_args()


def _task_splits(
    records: tuple[Any, ...],
    incremental_date: date,
    training_eligible: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    kind = np.asarray([record.sample_kind for record in records], dtype=object)
    repeat = np.asarray([record.repeat_index or 0 for record in records], dtype=int)
    capture_date = np.asarray([record.timestamp.date() for record in records], dtype=object)
    cv_group = np.asarray([record.cv_group for record in records], dtype=int)
    manual = kind == "manual_press"
    no_contact = kind == "no_contact"
    if training_eligible is None:
        training_eligible = np.ones(len(records), dtype=bool)
    no_contact = no_contact & np.asarray(training_eligible, dtype=bool)
    old = capture_date < incremental_date
    new = capture_date >= incremental_date

    return {
        "contact_cross_batch": {
            "task": "contact",
            "labels": np.asarray([record.contact_label for record in records], dtype=object),
            "label_order": CONTACT_ORDER,
            "train": (old & no_contact & (cv_group != 5)) | (manual & (repeat <= 4)),
            "validation": (old & no_contact & (cv_group == 5)) | (manual & (repeat == 5)),
            "test": new & (no_contact | manual),
            "validity": "grouped_cross_capture_date_challenge",
            "notes": "Tests old-session training against the 2026-07-14 no-contact and hard batch.",
        },
        "position_cross_batch": {
            "task": "position",
            "labels": np.asarray([record.position_label or "" for record in records], dtype=object),
            "label_order": POSITION_ORDER,
            "train": manual & (repeat <= 4),
            "validation": manual & (repeat == 5),
            "test": manual & (repeat >= 6),
            "validity": "grouped_cross_capture_batch_challenge",
            "notes": "Tests repeats 1-5 against independent repeats 6-15 with all three manual response levels.",
        },
        "position_incremental_holdout": {
            "task": "position",
            "labels": np.asarray([record.position_label or "" for record in records], dtype=object),
            "label_order": POSITION_ORDER,
            "train": manual & (repeat <= 11),
            "validation": manual & (repeat == 12),
            "test": manual & (repeat >= 13),
            "validity": "grouped_late_repeat_holdout",
            "notes": "Measures late-repeat performance on repeats 13-15 with all three response levels.",
        },
        "force_cross_batch": {
            "task": "force",
            "labels": np.asarray([record.manual_force_label or "" for record in records], dtype=object),
            "label_order": FORCE_ORDER,
            "train": manual & (repeat <= 4),
            "validation": manual & (repeat == 5),
            "test": manual & (repeat >= 6),
            "validity": "grouped_cross_capture_batch_challenge",
            "notes": "Tests light/normal/hard repeats 1-5 against independent repeats 6-15.",
        },
        "force_repeat_holdout": {
            "task": "force",
            "labels": np.asarray([record.manual_force_label or "" for record in records], dtype=object),
            "label_order": FORCE_ORDER,
            "train": manual & (repeat <= 3),
            "validation": manual & (repeat == 4),
            "test": manual & (repeat == 5),
            "validity": "grouped_leave_repeats_4_and_5_out",
            "notes": "Balanced early-repeat light/normal/hard snapshot reference test.",
        },
    }


def _validate_split(
    track: str,
    spec: dict[str, Any],
    records: tuple[Any, ...],
) -> dict[str, Any]:
    masks = {name: np.asarray(spec[name], dtype=bool) for name in ("train", "validation", "test")}
    file_sets = {
        name: {records[index].file_id for index in np.flatnonzero(mask)}
        for name, mask in masks.items()
    }
    overlap = (
        (file_sets["train"] & file_sets["validation"])
        | (file_sets["train"] & file_sets["test"])
        | (file_sets["validation"] & file_sets["test"])
    )
    if overlap:
        raise ValueError(f"{track} contains file leakage: {sorted(overlap)[:3]}")
    labels = np.asarray(spec["labels"], dtype=object)
    label_order = list(spec["label_order"])
    audit = {
        "track": track,
        "task": spec["task"],
        "evaluation_validity": spec["validity"],
        "notes": spec["notes"],
        "file_overlap_count": 0,
    }
    for name, mask in masks.items():
        observed = labels[mask]
        missing = sorted(set(label_order) - set(observed.tolist()))
        if missing:
            raise ValueError(f"{track} {name} split misses labels: {missing}")
        audit[f"num_{name}_files"] = int(np.sum(mask))
        audit[f"{name}_label_counts"] = {
            label: int(np.sum(observed == label)) for label in label_order
        }
    return audit


def _classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
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


def _serialized_size_mb(model: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def _predict_latency_ms(model: Any, values: np.ndarray, repeats: int = 5) -> float:
    if values.shape[0] == 0:
        return float("nan")
    model.predict(values[: min(values.shape[0], 4)])
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(values)
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / (repeats * values.shape[0])


def _tree_pipeline(kind: str, random_state: int) -> Pipeline:
    if kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        estimator = ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _svm_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    C=4.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit_sklearn_model(
    model_id: str,
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, Any], Any]:
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_time = time.perf_counter() - started
    predicted = np.asarray(model.predict(test_x), dtype=object)
    metrics = _classification_metrics(test_y, predicted, labels)
    metrics.update(
        {
            "model_id": model_id,
            "training_time_sec": training_time,
            "inference_latency_ms_per_spectrum": _predict_latency_ms(model, test_x),
            "model_size_mb": _serialized_size_mb(model),
            "predict_proba_available": hasattr(model, "predict_proba"),
            "confidence_source": (
                "uncalibrated_predict_proba" if hasattr(model, "predict_proba") else "decision_only"
            ),
        }
    )
    return metrics, model


def _fit_minirocket(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels: list[str],
    kernels: int,
    random_state: int,
) -> tuple[dict[str, Any], Any]:
    from aeon.classification.convolution_based import MiniRocketClassifier

    model = MiniRocketClassifier(
        n_kernels=kernels,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return _fit_sklearn_model(
        "minirocket_multiview",
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        labels,
    )


def _fit_small_cnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels: list[str],
    epochs: int,
    patience: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not torch_available():
        raise ImportError("PyTorch is not installed")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_target = np.asarray([label_to_index[str(value)] for value in train_y], dtype=np.int64)
    validation_target = np.asarray(
        [label_to_index[str(value)] for value in validation_y], dtype=np.int64
    )
    test_target = np.asarray([label_to_index[str(value)] for value in test_y], dtype=np.int64)
    standardizer = ChannelStandardizer.fit(train_x)
    normalized_train = standardizer.transform(train_x)
    normalized_validation = standardizer.transform(validation_x)
    normalized_test = standardizer.transform(test_x)

    train_dataset = TensorDataset(
        torch.from_numpy(normalized_train), torch.from_numpy(train_target)
    )
    loader_generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(32, len(train_dataset)),
        shuffle=True,
        generator=loader_generator,
    )
    model = SmallSpectral1DCNN(
        in_channels=normalized_train.shape[1],
        num_classes=len(labels),
    )
    class_counts = np.bincount(train_target, minlength=len(labels)).astype(float)
    class_weights = class_counts.sum() / np.maximum(class_counts * len(labels), 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1.0e-5
    )

    def infer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(values))
            probability = torch.softmax(logits, dim=1).cpu().numpy()
        prediction = np.asarray([labels[index] for index in np.argmax(probability, axis=1)])
        return prediction, probability

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_macro_f1 = -1.0
    wait = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_values, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values)
            loss = criterion(logits, batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation_prediction, _ = infer(normalized_validation)
        validation_macro_f1 = float(
            f1_score(
                validation_y,
                validation_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        scheduler.step(validation_macro_f1)
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(batch_losses)),
                "validation_macro_f1": validation_macro_f1,
            }
        )
        if validation_macro_f1 > best_macro_f1 + 1.0e-5:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    training_time = time.perf_counter() - started
    model.load_state_dict(best_state)
    predicted, probability = infer(normalized_test)
    metrics = _classification_metrics(test_y, predicted, labels)
    warmup = torch.from_numpy(normalized_test[: min(4, normalized_test.shape[0])])
    with torch.no_grad():
        model(warmup)
    latency_started = time.perf_counter()
    latency_repeats = 20
    with torch.no_grad():
        for _ in range(latency_repeats):
            model(torch.from_numpy(normalized_test))
    latency = (
        (time.perf_counter() - latency_started)
        * 1000.0
        / (latency_repeats * normalized_test.shape[0])
    )
    buffer = io.BytesIO()
    torch.save(best_state, buffer)
    metrics.update(
        {
            "model_id": "small_spectral_1dcnn_multiview",
            "training_time_sec": training_time,
            "inference_latency_ms_per_spectrum": latency,
            "model_size_mb": len(buffer.getvalue()) / (1024.0 * 1024.0),
            "predict_proba_available": True,
            "confidence_source": "uncalibrated_softmax",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
        }
    )
    artifact = {
        "state_dict": best_state,
        "architecture": "SmallSpectral1DCNN",
        "input_semantics": "three_channel_wavelength_axis_not_time",
        "input_shape": [None, int(train_x.shape[1]), int(train_x.shape[2])],
        "label_order": labels,
        "channel_standardizer_mean": standardizer.mean,
        "channel_standardizer_scale": standardizer.scale,
        "history": history,
        "test_probabilities": probability,
        "test_predictions": predicted,
    }
    return metrics, artifact


def _plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
    destination: Path,
) -> None:
    values = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = values.max() * 0.55 if values.size else 0.0
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
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_cnn_history(history: list[dict[str, float]], title: str, destination: Path) -> None:
    epochs = [row["epoch"] for row in history]
    loss = [row["training_loss"] for row in history]
    macro_f1 = [row["validation_macro_f1"] for row in history]
    figure, left = plt.subplots(figsize=(8.0, 4.4))
    right = left.twinx()
    left.plot(epochs, loss, color="#0b84a5", label="Training loss")
    right.plot(epochs, macro_f1, color="#f6c85f", label="Validation macro-F1")
    left.set_xlabel("Epoch")
    left.set_ylabel("Loss")
    right.set_ylabel("Macro-F1")
    left.set_title(title)
    left.grid(alpha=0.2)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _write_leaderboard(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "track",
        "task",
        "model_id",
        "input_type",
        "evaluation_validity",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "training_time_sec",
        "inference_latency_ms_per_spectrum",
        "model_size_mb",
        "confidence_source",
        "status",
        "missing_reason",
    ]
    with (output_dir / "model_leaderboard.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
    lines = [
        "# Static spectral sequence model leaderboard",
        "",
        "Current input is one independent 512-point spectrum per CSV. The 1D-CNN and MiniRocket results below operate along wavelength, not time.",
        "",
        "| Track | Model | Input | Accuracy | Macro-F1 | Latency ms/spectrum | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item.get("track", ""),
            -(float(item.get("macro_f1")) if item.get("macro_f1") not in (None, "") else -1.0),
        ),
    ):
        def value(name: str, digits: int = 4) -> str:
            raw = row.get(name)
            return "" if raw in (None, "") else f"{float(raw):.{digits}f}"

        lines.append(
            f"| {row['track']} | {row['model_id']} | {row['input_type']} | "
            f"{value('accuracy')} | {value('macro_f1')} | "
            f"{value('inference_latency_ms_per_spectrum', 3)} | {row['status']} |"
        )
    (output_dir / "model_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    config = load_training_config(args.config.resolve())
    manifest_before = dataset_source_manifest(config)
    records = tuple(load_sense_dataset(config))
    manifest_after = dataset_source_manifest(config)
    assert_dataset_manifest_stable(manifest_before, manifest_after)
    output_dir.mkdir(parents=True)
    (output_dir / "confusion_matrices").mkdir()
    (output_dir / "training_curves").mkdir()
    (output_dir / "models").mkdir()
    (output_dir / "source_dataset_manifest.json").write_text(
        json.dumps(manifest_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    static_dataset = build_static_feature_dataset(
        records,
        config.get("feature_extraction", config),
        args.channel_config.resolve(),
    )
    spectral = build_spectral_multiview_data(records, static_dataset)
    splits = _task_splits(
        records,
        args.incremental_date,
        np.asarray(static_dataset.training_eligible, dtype=bool),
    )
    audits = {
        track: _validate_split(track, spec, records) for track, spec in splits.items()
    }

    engineered = static_dataset.engineered_matrix
    spatial_fingerprint, _spatial_columns = build_spatial_fingerprint_matrix(
        static_dataset.engineered_matrix,
        static_dataset.engineered_columns,
    )
    current_shape = spectral.values[:, 0, :]
    multiview_flat = spectral.values.reshape(spectral.values.shape[0], -1)
    result_by_track: dict[str, list[dict[str, Any]]] = {}
    leaderboard: list[dict[str, Any]] = []
    cnn_histories: dict[str, list[dict[str, float]]] = {}
    errors: list[dict[str, str]] = []

    for track, spec in splits.items():
        labels = np.asarray(spec["labels"], dtype=object)
        label_order = list(spec["label_order"])
        train_index = np.flatnonzero(spec["train"])
        validation_index = np.flatnonzero(spec["validation"])
        test_index = np.flatnonzero(spec["test"])
        track_results: list[dict[str, Any]] = []

        classical_models: list[tuple[str, str, np.ndarray, Callable[[], Any]]] = [
            (
                "random_forest_engineered",
                "engineered_baseline_relative",
                engineered,
                lambda: _tree_pipeline("random_forest", args.random_state),
            ),
            (
                "extra_trees_engineered",
                "engineered_baseline_relative",
                engineered,
                lambda: _tree_pipeline("extra_trees", args.random_state),
            ),
            (
                "random_forest_current_shape",
                "per_spectrum_normalized_current_shape_512",
                current_shape,
                lambda: _tree_pipeline("random_forest", args.random_state),
            ),
            (
                "extra_trees_spatial_fingerprint",
                "per_family_nine_channel_relative_spatial_fingerprint",
                spatial_fingerprint,
                lambda: _tree_pipeline("extra_trees", args.random_state),
            ),
            (
                "svm_rbf_multiview",
                "three_channel_spectral_multiview_flat",
                multiview_flat,
                lambda: _svm_pipeline(args.random_state),
            ),
        ]
        for model_id, input_type, matrix, builder in classical_models:
            try:
                metrics, _model = _fit_sklearn_model(
                    model_id,
                    builder(),
                    matrix[train_index],
                    labels[train_index],
                    matrix[test_index],
                    labels[test_index],
                    label_order,
                )
                metrics["input_type"] = input_type
                metrics["status"] = "completed"
                track_results.append(metrics)
            except Exception as exc:  # noqa: BLE001 - one failed optional model must not abort the pack
                errors.append({"track": track, "model_id": model_id, "error": repr(exc)})

        if not args.skip_minirocket:
            try:
                metrics, _model = _fit_minirocket(
                    spectral.values[train_index],
                    labels[train_index],
                    spectral.values[test_index],
                    labels[test_index],
                    label_order,
                    args.minirocket_kernels,
                    args.random_state,
                )
                metrics["input_type"] = "three_channel_wavelength_axis_sequence"
                metrics["status"] = "completed"
                track_results.append(metrics)
            except Exception as exc:  # noqa: BLE001
                errors.append({"track": track, "model_id": "minirocket_multiview", "error": repr(exc)})
                leaderboard.append(
                    {
                        "track": track,
                        "task": spec["task"],
                        "model_id": "minirocket_multiview",
                        "input_type": "three_channel_wavelength_axis_sequence",
                        "evaluation_validity": spec["validity"],
                        "status": "skipped",
                        "missing_reason": repr(exc),
                    }
                )

        if not args.skip_cnn:
            try:
                metrics, artifact = _fit_small_cnn(
                    spectral.values[train_index],
                    labels[train_index],
                    spectral.values[validation_index],
                    labels[validation_index],
                    spectral.values[test_index],
                    labels[test_index],
                    label_order,
                    args.epochs,
                    args.patience,
                    args.random_state,
                )
                metrics["input_type"] = "three_channel_wavelength_axis_sequence"
                metrics["status"] = "completed_baseline_not_deployed"
                track_results.append(metrics)
                cnn_histories[track] = artifact["history"]
                import torch

                torch.save(
                    {key: value for key, value in artifact.items() if key != "test_probabilities"},
                    output_dir / "models" / f"{track}_small_spectral_1dcnn.pt",
                )
                _plot_cnn_history(
                    artifact["history"],
                    f"{track}: Small Spectral 1D-CNN",
                    output_dir / "training_curves" / f"{track}_small_1dcnn.png",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"track": track, "model_id": "small_spectral_1dcnn_multiview", "error": repr(exc)}
                )
                leaderboard.append(
                    {
                        "track": track,
                        "task": spec["task"],
                        "model_id": "small_spectral_1dcnn_multiview",
                        "input_type": "three_channel_wavelength_axis_sequence",
                        "evaluation_validity": spec["validity"],
                        "status": "skipped",
                        "missing_reason": repr(exc),
                    }
                )

        result_by_track[track] = track_results
        for metrics in track_results:
            row = {
                "track": track,
                "task": spec["task"],
                "evaluation_validity": spec["validity"],
                "missing_reason": "",
                **{key: metrics.get(key) for key in (
                    "model_id",
                    "input_type",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "training_time_sec",
                    "inference_latency_ms_per_spectrum",
                    "model_size_mb",
                    "confidence_source",
                    "status",
                )},
            }
            leaderboard.append(row)
        completed = [row for row in track_results if row.get("status", "").startswith("completed")]
        if completed:
            best = max(completed, key=lambda item: (item["macro_f1"], item["accuracy"]))
            _plot_confusion_matrix(
                best["confusion_matrix"],
                label_order,
                f"{track}: {best['model_id']}",
                output_dir / "confusion_matrices" / f"{track}_best.png",
            )

    _write_leaderboard(leaderboard, output_dir)
    metrics_payload = {
        "schema_version": "spectral_sequence_model_comparison_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_semantics": "one_independent_static_full_spectrum_snapshot_per_csv",
        "num_independent_csv_files": len(records),
        "input_shape": list(spectral.values.shape),
        "spectral_channel_names": list(spectral.channel_names),
        "true_temporal_results_available": False,
        "baseline_cluster_audit": [
            asdict(item) for item in static_dataset.baseline_cluster_assessments
        ],
        "excluded_no_contact_files": int(
            sum(
                record.sample_kind == "no_contact" and not eligible
                for record, eligible in zip(
                    records, static_dataset.training_eligible, strict=True
                )
            )
        ),
        "split_audit": audits,
        "results": result_by_track,
        "errors": errors,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    audit_lines = [
        "# Data and split audit",
        "",
        f"- Independent CSV spectra: {len(records)}",
        f"- Spectral tensor shape: {list(spectral.values.shape)}",
        "- One CSV is one independent static spectrum; 512 is the wavelength-axis length, not time steps.",
        "- Every split is file-exclusive; random spectrum-level splitting is not used.",
        "",
    ]
    for track, audit in audits.items():
        audit_lines.extend(
            [
                f"## {track}",
                "",
                f"- Validity: {audit['evaluation_validity']}",
                f"- Train / validation / test: {audit['num_train_files']} / {audit['num_validation_files']} / {audit['num_test_files']}",
                f"- File overlap: {audit['file_overlap_count']}",
                f"- Note: {audit['notes']}",
                "",
            ]
        )
    (output_dir / "data_and_split_audit.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )

    temporal_lines = [
        "# Temporal model readiness",
        "",
        "## Current status",
        "",
        "The current dataset contains independent static 512-point spectra. The 512 points are wavelengths, not chronological samples. SmallSpectral1DCNN and MiniRocket therefore learn wavelength-axis patterns.",
        "",
        "## Implemented but intentionally not trained",
        "",
        "- `TemporalTCN`: expects `[batch, spectral_features, time_steps]`.",
        "- `TemporalCNNLSTM`: expects `[batch, spectral_features, time_steps]`.",
        "",
        "## Data required for valid temporal comparison",
        "",
        "- Continuous frame sequences with stable acquisition timestamps.",
        "- File/session IDs and no-contact, onset, hold, release, and recovery intervals.",
        "- Grouped train/test separation by complete acquisition session.",
        "- At least several independent sequences per position and response level.",
        "",
        "No temporal accuracy is reported until these conditions are met.",
    ]
    (output_dir / "temporal_model_readiness.md").write_text(
        "\n".join(temporal_lines) + "\n", encoding="utf-8"
    )

    root_lines = [
        "# Recognition root-cause analysis",
        "",
        "This is an offline model comparison and does not deploy a new model.",
        "",
        "## Evidence to inspect",
        "",
        "- `position_cross_batch` isolates capture-batch drift: all models train on repeats 1-4 and test on repeats 6-15.",
        "- `position_incremental_holdout` measures whether later samples from the new batch can be learned.",
        "- The difference between these tracks estimates domain shift rather than ordinary model capacity.",
        "- `force_cross_batch` tests all three response levels from repeats 6-15, so it measures the same domain shift for response level.",
        "",
        "## Known causes before model selection",
        "",
        "1. Optical baseline and spectrum shape change between acquisition batches.",
        "2. The nine sensing points have unequal sensitivity and coupling patterns.",
        "3. Tree models and spectral CNNs both lose accuracy on an unseen capture batch, so model family is not the sole cause.",
        "4. Light and normal responses have lower spatial signal-to-baseline separation than hard responses.",
        "5. Manual fingertip contact area and exact placement vary between files.",
        "6. One internally stable no-contact cluster has a localized residual shape and is excluded from reference/training use.",
        "",
        "See `model_leaderboard.md` for measured algorithm differences.",
    ]
    (output_dir / "root_cause_analysis.md").write_text(
        "\n".join(root_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps({"output_dir": str(output_dir), "rows": len(leaderboard), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
