"""Train grouped temporal baselines on ordered 9-FBG Sense DAT sequences."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    DynamicWindowDataset,
    build_dynamic_window_dataset,
    load_dynamic_config,
    load_dynamic_feature_sequences,
)
from src.hybrid_spectrum.dynamic_temporal_features import (  # noqa: E402
    SUMMARY_FEATURE_BLOCK_ORDER,
    temporal_summary_features,
)
from src.hybrid_spectrum.sequence_models import (  # noqa: E402
    SmallTemporal1DCNN,
    TemporalCNNLSTM,
    TemporalTCN,
    torch_available,
)


CONTACT_ORDER = ["no_contact", "contact"]
POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
RESPONSE_ORDER = ["light", "normal", "hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minirocket-kernels", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-minirocket", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    return parser.parse_args()


def classification_metrics(
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
    voted_truth: list[str] = []
    voted_prediction: list[str] = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        true_values, true_counts = np.unique(truth[mask], return_counts=True)
        pred_values, pred_counts = np.unique(predicted[mask], return_counts=True)
        voted_truth.append(str(true_values[int(np.argmax(true_counts))]))
        voted_prediction.append(str(pred_values[int(np.argmax(pred_counts))]))
    metrics = classification_metrics(
        np.asarray(voted_truth), np.asarray(voted_prediction), labels
    )
    metrics["vote_unit_count"] = len(voted_truth)
    return metrics


def serialized_size_mb(model: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def predict_latency_ms(model: Any, values: np.ndarray, repeats: int = 5) -> float:
    if len(values) == 0:
        return float("nan")
    model.predict(values[: min(4, len(values))])
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(values)
    return (time.perf_counter() - started) * 1000.0 / (repeats * len(values))


def channel_standardize(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Fit per-frame-feature scaling on training windows only."""

    mean, scale = channel_scaler_statistics(train)
    result = [channel_transform(train, mean, scale)]
    for values in others:
        result.append(channel_transform(values, mean, scale))
    return tuple(np.asarray(item, dtype=np.float32) for item in result)


def channel_scaler_statistics(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = np.asarray(train, dtype=np.float32).transpose(0, 2, 1)
    mean = np.mean(channels, axis=(0, 2), keepdims=True)
    scale = np.std(channels, axis=(0, 2), keepdims=True)
    return mean.astype(np.float32), np.where(scale > 1.0e-8, scale, 1.0).astype(np.float32)


def channel_transform(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    channels = np.asarray(values, dtype=np.float32).transpose(0, 2, 1)
    return ((channels - mean) / scale).astype(np.float32)


def task_specifications(dataset: DynamicWindowDataset) -> dict[str, dict[str, Any]]:
    contact = dataset.contact_labels == "contact"
    return {
        "contact": {
            "mask": np.ones(len(dataset.values), dtype=bool),
            "labels": dataset.contact_labels,
            "label_order": CONTACT_ORDER,
            "vote_keys": np.char.add(
                np.char.add(dataset.file_ids.astype(str), "::"),
                dataset.stage_labels.astype(str),
            ),
            "voting_unit": "file_stage",
        },
        "position": {
            "mask": contact,
            "labels": dataset.position_labels,
            "label_order": POSITION_ORDER,
            "vote_keys": dataset.file_ids,
            "voting_unit": "original_dat_file",
        },
        "response_level": {
            "mask": contact,
            "labels": dataset.stage_labels,
            "label_order": RESPONSE_ORDER,
            "vote_keys": np.char.add(
                np.char.add(dataset.file_ids.astype(str), "::"),
                dataset.stage_labels.astype(str),
            ),
            "voting_unit": "file_stage",
        },
    }


def validate_grouped_fold(
    dataset: DynamicWindowDataset,
    task_mask: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    train_files = set(dataset.file_ids[task_mask & train_mask].tolist())
    test_files = set(dataset.file_ids[task_mask & test_mask].tolist())
    overlap = train_files & test_files
    if overlap:
        raise AssertionError(f"file leakage detected: {sorted(overlap)[:3]}")


def fit_tree_fold(
    model_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, Any, dict[str, float | str | bool]]:
    if model_id == "random_forest_temporal_stats":
        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_id == "extra_trees_temporal_stats":
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_id == "svm_rbf_temporal_stats":
        model = Pipeline(
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
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_id == "logistic_temporal_stats":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown temporal summary model: {model_id}")
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_time = time.perf_counter() - started
    predicted = np.asarray(model.predict(test_x))
    confidence_source = (
        "uncalibrated_tree_vote_probability"
        if model_id in {"random_forest_temporal_stats", "extra_trees_temporal_stats"}
        else (
            "svc_internal_platt_probability"
            if model_id == "svm_rbf_temporal_stats"
            else "multinomial_logistic_probability"
        )
    )
    metadata = {
        "training_time_sec": training_time,
        "inference_latency_ms_per_window": predict_latency_ms(model, test_x),
        "model_size_mb": serialized_size_mb(model),
        "predict_proba_available": True,
        "confidence_source": confidence_source,
    }
    return predicted, model, metadata


def fit_minirocket_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    kernels: int,
    random_state: int,
) -> tuple[np.ndarray, Any, dict[str, float | str | bool]]:
    from aeon.classification.convolution_based import MiniRocketClassifier

    channel_mean, channel_scale = channel_scaler_statistics(train_x)
    normalized_train = channel_transform(train_x, channel_mean, channel_scale)
    normalized_test = channel_transform(test_x, channel_mean, channel_scale)
    model = MiniRocketClassifier(
        n_kernels=kernels,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    started = time.perf_counter()
    model.fit(normalized_train, train_y)
    training_time = time.perf_counter() - started
    predicted = np.asarray(model.predict(normalized_test))
    artifact = {
        "model": model,
        "channel_standardizer_mean": channel_mean,
        "channel_standardizer_scale": channel_scale,
        "input_semantics": "frame_features_by_time",
    }
    metadata = {
        "training_time_sec": training_time,
        "inference_latency_ms_per_window": predict_latency_ms(model, normalized_test),
        "model_size_mb": serialized_size_mb(artifact),
        "predict_proba_available": hasattr(model, "predict_proba"),
        "confidence_source": (
            "minirocket_ridge_probability"
            if hasattr(model, "predict_proba")
            else "decision_function_only"
        ),
    }
    return predicted, artifact, metadata


def fit_deep_fold(
    model_id: str,
    model_factory: Callable[[int, int], Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    labels: list[str],
    epochs: int,
    patience: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not torch_available():
        raise ImportError("PyTorch is not installed")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    label_to_index = {label: index for index, label in enumerate(labels)}

    def encode(values: np.ndarray) -> np.ndarray:
        return np.asarray([label_to_index[str(value)] for value in values], dtype=np.int64)

    selection_mean, selection_scale = channel_scaler_statistics(train_x)
    selection_train = channel_transform(train_x, selection_mean, selection_scale)
    selection_validation = channel_transform(
        validation_x, selection_mean, selection_scale
    )
    train_target = encode(train_y)
    validation_target = encode(validation_y)

    def train_epochs(
        model: Any,
        values: np.ndarray,
        targets: np.ndarray,
        epoch_count: int,
        seed: int,
    ) -> list[float]:
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(values), torch.from_numpy(targets)),
            batch_size=min(64, len(values)),
            shuffle=True,
            generator=generator,
        )
        counts = np.bincount(targets, minlength=len(labels)).astype(float)
        weights = counts.sum() / np.maximum(counts * len(labels), 1.0)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.0e-4)
        losses: list[float] = []
        for _ in range(epoch_count):
            model.train()
            batch_losses: list[float] = []
            for batch_values, batch_target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_values), batch_target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(batch_losses)))
        return losses

    def infer(model: Any, values: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(values))
        indices = torch.argmax(logits, dim=1).cpu().numpy()
        return np.asarray([labels[index] for index in indices])

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    selection_model = model_factory(selection_train.shape[1], len(labels))
    counts = np.bincount(train_target, minlength=len(labels)).astype(float)
    weights = counts.sum() / np.maximum(counts * len(labels), 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        selection_model.parameters(), lr=1.2e-3, weight_decay=2.0e-4
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(selection_train), torch.from_numpy(train_target)),
        batch_size=min(64, len(selection_train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    best_state = copy.deepcopy(selection_model.state_dict())
    best_epoch = 1
    best_macro_f1 = -1.0
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        selection_model.train()
        losses: list[float] = []
        for batch_values, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(selection_model(batch_values), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(selection_model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = infer(selection_model, selection_validation)
        validation_macro_f1 = float(
            f1_score(
                validation_y,
                validation_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_macro_f1": validation_macro_f1,
            }
        )
        if validation_macro_f1 > best_macro_f1 + 1.0e-5:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(selection_model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # Refit for the selected epoch on both non-test capture groups.
    combined_x = np.concatenate([train_x, validation_x], axis=0)
    combined_y = np.concatenate([train_y, validation_y], axis=0)
    final_mean, final_scale = channel_scaler_statistics(combined_x)
    final_train = channel_transform(combined_x, final_mean, final_scale)
    final_test = channel_transform(test_x, final_mean, final_scale)
    final_target = encode(combined_y)
    torch.manual_seed(random_state + 1000)
    final_model = model_factory(final_train.shape[1], len(labels))
    started = time.perf_counter()
    final_losses = train_epochs(
        final_model,
        final_train,
        final_target,
        best_epoch,
        random_state + 1000,
    )
    training_time = time.perf_counter() - started
    predicted = infer(final_model, final_test)
    with torch.no_grad():
        final_model(torch.from_numpy(final_test[: min(4, len(final_test))]))
    repeats = 15
    latency_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            final_model(torch.from_numpy(final_test))
    latency = (
        (time.perf_counter() - latency_started) * 1000.0 / (repeats * len(final_test))
    )
    buffer = io.BytesIO()
    torch.save(final_model.state_dict(), buffer)
    artifact = {
        "model_id": model_id,
        "state_dict": final_model.state_dict(),
        "label_order": labels,
        "input_shape": [None, int(final_train.shape[1]), int(final_train.shape[2])],
        "input_semantics": "frame_features_by_time",
        "channel_standardizer_mean": final_mean,
        "channel_standardizer_scale": final_scale,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "selection_history": history,
        "final_training_loss": final_losses,
    }
    metadata = {
        "training_time_sec": training_time,
        "inference_latency_ms_per_window": latency,
        "model_size_mb": len(buffer.getvalue()) / (1024.0 * 1024.0),
        "predict_proba_available": True,
        "confidence_source": "uncalibrated_softmax",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
    }
    return predicted, artifact, metadata


def plot_confusion(
    matrix: list[list[int]], labels: list[str], title: str, destination: Path
) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = float(values.max()) * 0.55 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(int(values[row, column])),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "#102236",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_history(history: list[dict[str, float]], title: str, destination: Path) -> None:
    figure, left = plt.subplots(figsize=(7.6, 4.2))
    right = left.twinx()
    epochs = [row["epoch"] for row in history]
    left.plot(epochs, [row["training_loss"] for row in history], color="#0072B2")
    right.plot(
        epochs,
        [row["validation_macro_f1"] for row in history],
        color="#D55E00",
    )
    left.set_xlabel("Epoch")
    left.set_ylabel("Training loss")
    right.set_ylabel("Validation macro-F1")
    left.set_title(title)
    left.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    for child in ("models", "confusion_matrices", "training_curves"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    config = load_dynamic_config(args.config)
    sequences = load_dynamic_feature_sequences(config)
    dataset = build_dynamic_window_dataset(sequences, config)
    summary_x = temporal_summary_features(dataset.values)
    groups = sorted(set(dataset.capture_groups.tolist()))
    if groups != ["G1", "G2", "G3"]:
        raise ValueError(f"expected G1/G2/G3 capture groups, got {groups}")

    np.savez_compressed(
        output_dir / "dynamic_sequence_windows.npz",
        X=dataset.values,
        stage_labels=dataset.stage_labels,
        contact_labels=dataset.contact_labels,
        position_labels=dataset.position_labels,
        file_ids=dataset.file_ids,
        capture_groups=dataset.capture_groups,
        window_start_frames=dataset.window_start_frames,
        window_end_frames=dataset.window_end_frames,
        feature_names=np.asarray(dataset.feature_names),
    )

    model_specs: list[tuple[str, str, Any]] = [
        ("random_forest_temporal_stats", "tree", None),
        ("extra_trees_temporal_stats", "tree", None),
        ("svm_rbf_temporal_stats", "scaled_classical", None),
        ("logistic_temporal_stats", "scaled_classical", None),
    ]
    if not args.skip_minirocket:
        model_specs.append(("minirocket_temporal", "minirocket", None))
    deep_specs = [
        ("small_temporal_1dcnn", "deep", SmallTemporal1DCNN),
        ("temporal_cnn_lstm", "deep", TemporalCNNLSTM),
        ("temporal_tcn", "deep", TemporalTCN),
    ]

    all_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    prediction_maps: dict[tuple[str, str], dict[int, str]] = {}
    specifications = task_specifications(dataset)
    for task, specification in specifications.items():
        task_models = list(model_specs)
        if task != "contact" and not args.skip_deep:
            task_models.extend(deep_specs)
        task_mask = np.asarray(specification["mask"], dtype=bool)
        task_labels = np.asarray(specification["labels"])
        label_order = list(specification["label_order"])
        for model_index, (model_id, family, factory) in enumerate(task_models):
            fold_truth: list[np.ndarray] = []
            fold_prediction: list[np.ndarray] = []
            fold_indices: list[np.ndarray] = []
            fold_metadata: list[dict[str, Any]] = []
            status = "complete"
            missing_reason = ""
            try:
                for test_index, test_group in enumerate(groups):
                    test_group_mask = dataset.capture_groups == test_group
                    train_group_mask = ~test_group_mask
                    train_mask = task_mask & train_group_mask
                    test_mask = task_mask & test_group_mask
                    validate_grouped_fold(
                        dataset, task_mask, train_group_mask, test_group_mask
                    )
                    train_indices = np.flatnonzero(train_mask)
                    test_indices = np.flatnonzero(test_mask)
                    train_y = task_labels[train_indices]
                    test_y = task_labels[test_indices]
                    seed = args.random_state + 100 * model_index + test_index
                    if family in {"tree", "scaled_classical"}:
                        predicted, artifact, metadata = fit_tree_fold(
                            model_id,
                            summary_x[train_indices],
                            train_y,
                            summary_x[test_indices],
                            seed,
                        )
                        joblib.dump(
                            {
                                "model": artifact,
                                "model_id": model_id,
                                "task": task,
                                "input_semantics": "temporal_summary_features",
                                "summary_feature_block_order": SUMMARY_FEATURE_BLOCK_ORDER,
                                "frame_feature_names": dataset.feature_names,
                                "time_steps": int(dataset.values.shape[1]),
                            },
                            output_dir
                            / "models"
                            / f"{task}_{model_id}_{test_group}.joblib",
                            compress=3,
                        )
                    elif family == "minirocket":
                        predicted, artifact, metadata = fit_minirocket_fold(
                            dataset.values[train_indices],
                            train_y,
                            dataset.values[test_indices],
                            args.minirocket_kernels,
                            seed,
                        )
                        joblib.dump(
                            artifact,
                            output_dir
                            / "models"
                            / f"{task}_{model_id}_{test_group}.joblib",
                            compress=3,
                        )
                    else:
                        remaining_groups = [group for group in groups if group != test_group]
                        validation_group = remaining_groups[test_index % 2]
                        selection_train_group = remaining_groups[1 - (test_index % 2)]
                        selection_train_indices = np.flatnonzero(
                            task_mask
                            & (dataset.capture_groups == selection_train_group)
                        )
                        validation_indices = np.flatnonzero(
                            task_mask & (dataset.capture_groups == validation_group)
                        )
                        predicted, artifact, metadata = fit_deep_fold(
                            model_id,
                            factory,
                            dataset.values[selection_train_indices],
                            task_labels[selection_train_indices],
                            dataset.values[validation_indices],
                            task_labels[validation_indices],
                            dataset.values[test_indices],
                            label_order,
                            args.epochs,
                            args.patience,
                            seed,
                        )
                        import torch

                        torch.save(
                            artifact,
                            output_dir
                            / "models"
                            / f"{task}_{model_id}_{test_group}.pt",
                        )
                        plot_history(
                            artifact["selection_history"],
                            f"{task} / {model_id} / test {test_group}",
                            output_dir
                            / "training_curves"
                            / f"{task}_{model_id}_{test_group}.png",
                        )
                    metadata["test_group"] = test_group
                    metadata["num_train_windows"] = int(np.sum(train_mask))
                    metadata["num_test_windows"] = int(np.sum(test_mask))
                    fold_metrics = classification_metrics(
                        test_y, np.asarray(predicted), label_order
                    )
                    metadata["test_accuracy"] = fold_metrics["accuracy"]
                    metadata["test_macro_f1"] = fold_metrics["macro_f1"]
                    fold_metadata.append(metadata)
                    fold_truth.append(test_y)
                    fold_prediction.append(np.asarray(predicted))
                    fold_indices.append(test_indices)
            except (ImportError, ModuleNotFoundError) as error:
                status = "skipped"
                missing_reason = str(error)
            except Exception as error:
                status = "failed"
                missing_reason = f"{type(error).__name__}: {error}"

            if status != "complete":
                all_results.append(
                    {
                        "task": task,
                        "model_id": model_id,
                        "model_family": family,
                        "status": status,
                        "missing_reason": missing_reason,
                        "evaluation_validity": "grouped_by_capture_group",
                    }
                )
                continue

            truth = np.concatenate(fold_truth)
            predicted = np.concatenate(fold_prediction)
            indices = np.concatenate(fold_indices)
            order = np.argsort(indices)
            truth = truth[order]
            predicted = predicted[order]
            indices = indices[order]
            metrics = classification_metrics(truth, predicted, label_order)
            voting = majority_vote_metrics(
                truth,
                predicted,
                np.asarray(specification["vote_keys"])[indices],
                label_order,
            )
            metrics.update(
                {
                    "task": task,
                    "model_id": model_id,
                    "model_family": family,
                    "input_shape": [None, int(dataset.values.shape[1]), int(dataset.values.shape[2])],
                    "input_type": (
                        "temporal_summary_features"
                        if family in {"tree", "scaled_classical"}
                        else "raw_frame_feature_sequence"
                    ),
                    "evaluation_validity": "grouped_by_capture_group_and_file_id",
                    "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
                    "independent_dat_files": len(set(dataset.file_ids[task_mask].tolist())),
                    "num_windows": int(np.sum(task_mask)),
                    "voting_unit": specification["voting_unit"],
                    "grouped_vote_accuracy": voting["accuracy"],
                    "grouped_vote_macro_f1": voting["macro_f1"],
                    "grouped_vote_unit_count": voting["vote_unit_count"],
                    "minimum_capture_group_macro_f1": float(
                        min(float(item["test_macro_f1"]) for item in fold_metadata)
                    ),
                    "training_time_sec": float(
                        sum(float(item["training_time_sec"]) for item in fold_metadata)
                    ),
                    "inference_latency_ms_per_window": float(
                        np.mean(
                            [
                                float(item["inference_latency_ms_per_window"])
                                for item in fold_metadata
                            ]
                        )
                    ),
                    "model_size_mb": float(
                        np.mean([float(item["model_size_mb"]) for item in fold_metadata])
                    ),
                    "predict_proba_available": bool(
                        all(bool(item["predict_proba_available"]) for item in fold_metadata)
                    ),
                    "confidence_source": fold_metadata[0]["confidence_source"],
                    "folds": fold_metadata,
                    "status": "complete",
                    "missing_reason": "",
                }
            )
            all_results.append(metrics)
            prediction_maps[(task, model_id)] = {
                int(index): str(label) for index, label in zip(indices, predicted)
            }
            for index, true_label, predicted_label in zip(indices, truth, predicted):
                prediction_rows.append(
                    {
                        "task": task,
                        "model_id": model_id,
                        "window_index": int(index),
                        "file_id": dataset.file_ids[index],
                        "capture_group": dataset.capture_groups[index],
                        "window_start_frame": int(dataset.window_start_frames[index]),
                        "window_end_frame": int(dataset.window_end_frames[index]),
                        "stage_label": dataset.stage_labels[index],
                        "position_label": dataset.position_labels[index],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                    }
                )
            plot_confusion(
                metrics["confusion_matrix"],
                label_order,
                f"{task} / {model_id}",
                output_dir / "confusion_matrices" / f"{task}_{model_id}.png",
            )
            print(
                f"{task:14s} {model_id:30s} "
                f"macro_f1={metrics['macro_f1']:.3f} vote={voting['accuracy']:.3f}",
                flush=True,
            )

    with (output_dir / "window_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    combined_rows = []
    contact_indices = np.flatnonzero(dataset.contact_labels == "contact")
    position_models = sorted(
        model for task, model in prediction_maps if task == "position"
    )
    response_models = sorted(
        model for task, model in prediction_maps if task == "response_level"
    )
    for position_model_id in position_models:
        for response_model_id in response_models:
            position_map = prediction_maps[("position", position_model_id)]
            response_map = prediction_maps[("response_level", response_model_id)]
            available = [
                int(index)
                for index in contact_indices
                if int(index) in position_map and int(index) in response_map
            ]
            exact = [
                position_map[index] == dataset.position_labels[index]
                and response_map[index] == dataset.stage_labels[index]
                for index in available
            ]
            combined_rows.append(
                {
                    "position_model_id": position_model_id,
                    "response_model_id": response_model_id,
                    "combined_position_response_exact_accuracy": float(np.mean(exact)),
                    "num_contact_windows": len(available),
                }
            )
    with (output_dir / "combined_position_response_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    (output_dir / "dynamic_sequence_model_metrics.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    leaderboard_columns = [
        "task",
        "model_id",
        "model_family",
        "input_type",
        "evaluation_validity",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "grouped_vote_accuracy",
        "grouped_vote_macro_f1",
        "minimum_capture_group_macro_f1",
        "inference_latency_ms_per_window",
        "model_size_mb",
        "status",
        "missing_reason",
    ]
    with (output_dir / "dynamic_sequence_model_leaderboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard_columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in leaderboard_columns}
            for row in all_results
        )

    lines = [
        "# Dynamic 9-FBG temporal model comparison",
        "",
        "- The 2,098 overlapping windows come from only 27 independent DAT sequences.",
        "- Formal evaluation leaves out an entire capture group and every source DAT file within it.",
        "- Random window splitting is not used.",
        "- Windows never cross no_contact/light/normal/hard boundaries; release is excluded.",
        "- Light/normal/hard are approximate manual response levels, not calibrated force_N.",
        "- Current results are offline baselines and are not deployed to the digital twin.",
        "",
        "| Task | Model | Accuracy | Macro-F1 | Grouped vote accuracy | Latency ms/window | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        all_results,
        key=lambda item: (
            item.get("task", ""),
            -float(item.get("macro_f1", -1.0)),
        ),
    ):
        value = lambda name: (
            ""
            if row.get(name, "") in ("", None)
            else f"{float(row[name]):.3f}"
        )
        lines.append(
            f"| {row['task']} | {row['model_id']} | {value('accuracy')} | "
            f"{value('macro_f1')} | {value('grouped_vote_accuracy')} | "
            f"{value('inference_latency_ms_per_window')} | {row['status']} |"
        )
    lines.extend(["", "## Combined position and response level", ""])
    for row in sorted(
        combined_rows,
        key=lambda item: -float(item["combined_position_response_exact_accuracy"]),
    )[:10]:
        lines.append(
            f"- position={row['position_model_id']}, response={row['response_model_id']}: "
            f"exact position+level accuracy "
            f"{row['combined_position_response_exact_accuracy']:.3f} "
            f"over {row['num_contact_windows']} contact windows."
        )
    (output_dir / "dynamic_sequence_model_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
