"""Benchmark mature GitHub algorithms on the static 9-FBG spectral dataset.

The 512 samples in every current CSV are wavelength bins, not time steps.  This
script therefore evaluates spectral classifiers and a clearly marked
transductive domain-adaptation upper bound.  It never deploys a model or opens
the BaySpec device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
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
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.github_static_models import PLSDAClassifier  # noqa: E402
from src.hybrid_spectrum.sequence_models import build_spectral_multiview_data  # noqa: E402
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

GITHUB_SOURCES = (
    {
        "algorithm": "MultiRocket-Hydra",
        "repository": "aeon-toolkit/aeon",
        "url": "https://github.com/aeon-toolkit/aeon",
        "role": "wavelength-axis convolution feature benchmark",
    },
    {
        "algorithm": "CORAL via SKADA",
        "repository": "scikit-adaptation/skada",
        "url": "https://github.com/scikit-adaptation/skada",
        "role": "cross-batch domain-alignment upper bound",
    },
    {
        "algorithm": "LightGBM",
        "repository": "microsoft/LightGBM",
        "url": "https://github.com/microsoft/LightGBM",
        "role": "small-sample boosted-tree benchmark",
    },
    {
        "algorithm": "CatBoost",
        "repository": "catboost/catboost",
        "url": "https://github.com/catboost/catboost",
        "role": "optional boosted-tree benchmark when installed",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "sense_static_training.yaml"
    )
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=ROOT / "config" / "hybrid_spectrum_channels.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incremental-date", type=date.fromisoformat, default=date(2026, 7, 14))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mr-hydra-groups", type=int, default=32)
    return parser.parse_args()


def task_splits(records: tuple[Any, ...], incremental_date: date) -> dict[str, dict[str, Any]]:
    kind = np.asarray([record.sample_kind for record in records], dtype=object)
    repeat = np.asarray([record.repeat_index or 0 for record in records], dtype=int)
    capture_date = np.asarray([record.timestamp.date() for record in records], dtype=object)
    manual = kind == "manual_press"
    old = capture_date < incremental_date
    new = capture_date >= incremental_date
    return {
        "position_cross_batch": {
            "task": "position",
            "labels": np.asarray([record.position_label or "" for record in records], dtype=object),
            "label_order": POSITION_ORDER,
            "train": manual & old & (repeat <= 4),
            "validation": manual & old & (repeat == 5),
            "test": manual & new & (repeat >= 6),
        },
        "force_cross_batch": {
            "task": "force",
            "labels": np.asarray(
                [record.manual_force_label or "" for record in records], dtype=object
            ),
            "label_order": FORCE_ORDER,
            "train": manual & old & (repeat <= 4),
            "validation": manual & old & (repeat == 5),
            "test": manual & new & (repeat >= 6),
        },
    }


def classification_metrics(
    truth: np.ndarray, predicted: np.ndarray, label_order: list[str]
) -> dict[str, Any]:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        truth, predicted, labels=label_order, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=label_order, average="macro", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1_values[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(label_order)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=label_order).tolist(),
        "label_order": label_order,
    }


def serialized_size_mb(model: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def prediction_latency_ms(model: Any, values: np.ndarray, repeats: int = 5) -> float:
    model.predict(values[: min(4, len(values))])
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(values)
    return (time.perf_counter() - started) * 1000.0 / (repeats * len(values))


def evaluate_model(
    model_id: str,
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    label_order: list[str],
    *,
    input_type: str,
    algorithm_source: str,
    evaluation_validity: str = "grouped_cross_capture_batch_challenge",
    deployment_ready: str = "candidate_only_not_deployed",
) -> tuple[dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_time = time.perf_counter() - started
    predicted = np.asarray(model.predict(test_x), dtype=object)
    metrics = classification_metrics(test_y, predicted, label_order)
    metrics.update(
        {
            "model_id": model_id,
            "input_type": input_type,
            "algorithm_source": algorithm_source,
            "evaluation_validity": evaluation_validity,
            "deployment_ready": deployment_ready,
            "training_time_sec": training_time,
            "inference_latency_ms_per_spectrum": prediction_latency_ms(model, test_x),
            "model_size_mb": serialized_size_mb(model),
            "predict_proba_available": hasattr(model, "predict_proba"),
            "confidence_source": (
                "native_uncalibrated_predict_proba"
                if hasattr(model, "predict_proba")
                else "decision_score_only"
            ),
            "status": "completed",
        }
    )
    return metrics, predicted


def evaluate_coral_upper_bound(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    label_order: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    from skada import CORAL
    from skada.datasets import DomainAwareDataset

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    normalized_source = scaler.fit_transform(imputer.fit_transform(source_x))
    normalized_target = scaler.transform(imputer.transform(target_x))
    label_to_index = {label: index for index, label in enumerate(label_order)}
    encoded_source = np.asarray([label_to_index[str(value)] for value in source_y], dtype=int)
    encoded_target = np.asarray([label_to_index[str(value)] for value in target_y], dtype=int)
    dataset = DomainAwareDataset(
        [
            (normalized_source, encoded_source, "source"),
            (normalized_target, encoded_target, "target"),
        ]
    )
    packed_x, packed_y, sample_domain = dataset.pack(
        as_sources=["source"], as_targets=["target"], mask_target_labels=True
    )
    model = CORAL(
        base_estimator=SVC(C=4.0, kernel="rbf", gamma="scale", class_weight="balanced")
    )
    started = time.perf_counter()
    model.fit(packed_x, packed_y, sample_domain=sample_domain)
    training_time = time.perf_counter() - started
    encoded_prediction = np.asarray(
        model.predict(normalized_target, sample_domain=-1), dtype=int
    )
    predicted = np.asarray(label_order, dtype=object)[encoded_prediction]
    metrics = classification_metrics(target_y, predicted, label_order)
    started = time.perf_counter()
    for _ in range(5):
        model.predict(normalized_target, sample_domain=-1)
    latency = (time.perf_counter() - started) * 1000.0 / (5 * len(target_x))
    artifact = {"imputer": imputer, "scaler": scaler, "model": model}
    metrics.update(
        {
            "model_id": "coral_svc_transductive_upper_bound",
            "input_type": "selected_static_features_with_unlabeled_target_batch",
            "algorithm_source": "scikit-adaptation/skada",
            "evaluation_validity": "transductive_upper_bound_not_single_frame_final",
            "deployment_ready": "false_requires_unlabeled_target_batch",
            "training_time_sec": training_time,
            "inference_latency_ms_per_spectrum": latency,
            "model_size_mb": serialized_size_mb(artifact),
            "predict_proba_available": False,
            "confidence_source": "decision_score_only",
            "status": "completed_upper_bound_only",
        }
    )
    return metrics, predicted


def plot_confusion(
    matrix: list[list[int]], labels: list[str], title: str, destination: Path
) -> None:
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    image = axis.imshow(values, cmap="Blues")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_comparison(rows: list[dict[str, Any]], destination: Path) -> None:
    completed = [row for row in rows if row.get("macro_f1") is not None]
    tracks = ["position_cross_batch", "force_cross_batch"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for axis, track in zip(axes, tracks, strict=True):
        subset = sorted(
            [row for row in completed if row["track"] == track],
            key=lambda item: float(item["macro_f1"]),
        )
        labels = [str(row["model_id"]).replace("_", "\n") for row in subset]
        values = [float(row["macro_f1"]) for row in subset]
        colors = [
            "#d8a33a" if "transductive" in str(row["evaluation_validity"]) else "#2f7f9f"
            for row in subset
        ]
        axis.barh(labels, values, color=colors)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Macro-F1")
        axis.set_title(track.replace("_", " "))
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Static spectral algorithms: strict cross-batch comparison")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    results: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, str]],
    package_status: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> None:
    columns = [
        "track",
        "task",
        "model_id",
        "input_type",
        "algorithm_source",
        "evaluation_validity",
        "deployment_ready",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "training_time_sec",
        "inference_latency_ms_per_spectrum",
        "model_size_mb",
        "predict_proba_available",
        "confidence_source",
        "status",
        "missing_reason",
    ]
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "github_static_algorithm_benchmark_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_metadata,
        "github_sources": GITHUB_SOURCES,
        "package_status": package_status,
        "results": results,
        "errors": errors,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source_lines = ["# GitHub algorithm sources", ""]
    for item in GITHUB_SOURCES:
        source_lines.extend(
            [
                f"- [{item['algorithm']}]({item['url']}) - `{item['repository']}`; {item['role']}.",
            ]
        )
    source_lines.extend(
        [
            "",
            "Repository maturity does not imply suitability for this sensor. Every candidate is measured on the same file-exclusive cross-batch split.",
        ]
    )
    (output_dir / "github_algorithm_sources.md").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )

    report = [
        "# Static GitHub algorithm benchmark",
        "",
        "## Scope",
        "",
        f"- Independent static CSV spectra: {dataset_metadata['num_independent_csv_files']}",
        "- One CSV is one independent snapshot; 512 values are wavelength bins, not time samples.",
        "- Train/validation use early capture repeats; test uses later independent repeats from the new batch.",
        "- No BaySpec device was opened and no runtime model was replaced.",
        "",
        "## Results",
        "",
    ]
    for track in ("position_cross_batch", "force_cross_batch"):
        report.extend([f"### {track}", "", "| Model | Accuracy | Macro-F1 | Validity | Deployment |", "|---|---:|---:|---|---|"])
        for row in sorted(
            [item for item in rows if item["track"] == track and item.get("macro_f1") is not None],
            key=lambda item: float(item["macro_f1"]),
            reverse=True,
        ):
            report.append(
                f"| {row['model_id']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['evaluation_validity']} | {row['deployment_ready']} |"
            )
        report.append("")
    position_rows = [
        item
        for item in rows
        if item["track"] == "position_cross_batch"
        and item.get("macro_f1") is not None
        and "transductive" not in str(item["evaluation_validity"])
    ]
    force_rows = [
        item
        for item in rows
        if item["track"] == "force_cross_batch"
        and item.get("macro_f1") is not None
        and "transductive" not in str(item["evaluation_validity"])
    ]
    best_position = max(position_rows, key=lambda item: float(item["macro_f1"]))
    best_force = max(force_rows, key=lambda item: float(item["macro_f1"]))
    report.extend(
        [
            "## Decision",
            "",
            f"- Best deployable position candidate in this audit: `{best_position['model_id']}` with macro-F1 `{best_position['macro_f1']:.4f}`.",
            f"- Best deployable force candidate in this audit: `{best_force['model_id']}` with macro-F1 `{best_force['macro_f1']:.4f}`.",
            "- CORAL is reported only as a transductive upper bound because it sees the unlabeled target batch distribution before prediction.",
            "- MultiRocket-Hydra remains a wavelength-axis classifier here; it is not a temporal model.",
            "- A more complex GitHub algorithm is accepted only when it improves strict cross-batch performance without collapsing weak-response classes.",
            "",
            "## Interpretation",
            "",
            "The main recognition limit is capture-session domain shift plus unequal pixel sensitivity, especially for light and normal presses. Model family alone is not the root cause. Dynamic DAT sequences should be evaluated next because onset, hold, release, and residual recovery add information unavailable in a static spectrum.",
            "",
        ]
    )
    if errors:
        report.extend(["## Skipped or failed", ""])
        report.extend(f"- `{item['model_id']}` ({item['track']}): {item['error']}" for item in errors)
        report.append("")
    (output_dir / "benchmark_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "confusion_matrices").mkdir()
    (output_dir / "predictions").mkdir()

    config = load_training_config(args.config.resolve())
    manifest_before = dataset_source_manifest(config)
    records = tuple(load_sense_dataset(config))
    manifest_after = dataset_source_manifest(config)
    assert_dataset_manifest_stable(manifest_before, manifest_after)
    (output_dir / "source_dataset_manifest.json").write_text(
        json.dumps(manifest_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    static = build_static_feature_dataset(
        records, config.get("feature_extraction", config), args.channel_config.resolve()
    )
    spectral = build_spectral_multiview_data(records, static)
    spatial, _ = build_spatial_fingerprint_matrix(
        static.engineered_matrix, static.engineered_columns
    )
    multiview_flat = spectral.values.reshape(len(records), -1)
    splits = task_splits(records, args.incremental_date)

    package_status: dict[str, Any] = {}
    for package in ("aeon", "skada", "lightgbm", "catboost"):
        try:
            module = __import__(package)
            package_status[package] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except ImportError as exc:
            package_status[package] = {"available": False, "reason": str(exc)}

    rows: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    for track, spec in splits.items():
        labels = np.asarray(spec["labels"], dtype=object)
        label_order = list(spec["label_order"])
        source_index = np.flatnonzero(spec["train"] | spec["validation"])
        test_index = np.flatnonzero(spec["test"])
        if not set(label_order).issubset(set(labels[source_index])):
            raise ValueError(f"{track} source split misses a class")
        if not set(label_order).issubset(set(labels[test_index])):
            raise ValueError(f"{track} test split misses a class")
        if {records[index].file_id for index in source_index} & {
            records[index].file_id for index in test_index
        }:
            raise ValueError(f"{track} contains file leakage")

        selected_features = spatial if spec["task"] == "position" else static.engineered_matrix
        selected_name = (
            "nine_fbg_relative_spatial_fingerprint"
            if spec["task"] == "position"
            else "engineered_baseline_relative_features"
        )
        candidates: list[tuple[str, Any, np.ndarray, str, str]] = [
            (
                "extra_trees_selected_features",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            ExtraTreesClassifier(
                                n_estimators=600,
                                max_features="sqrt",
                                class_weight="balanced",
                                random_state=args.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
                selected_features,
                selected_name,
                "scikit-learn baseline",
            ),
            (
                "random_forest_selected_features",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=600,
                                max_features="sqrt",
                                class_weight="balanced_subsample",
                                random_state=args.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
                selected_features,
                selected_name,
                "scikit-learn baseline",
            ),
            (
                "shrinkage_lda_selected_features",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                        ),
                    ]
                ),
                selected_features,
                selected_name,
                "scikit-learn shrinkage discriminant baseline",
            ),
            (
                "shrunken_nearest_centroid_selected_features",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", NearestCentroid(shrink_threshold=0.1)),
                    ]
                ),
                selected_features,
                selected_name,
                "scikit-learn interpretable centroid baseline",
            ),
            (
                "plsda_multiview_spectrum",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", PLSDAClassifier(n_components=8)),
                    ]
                ),
                multiview_flat,
                "three_channel_spectral_multiview_flat",
                "chemometric PLS-DA baseline",
            ),
        ]
        if package_status["lightgbm"]["available"]:
            from lightgbm import LGBMClassifier

            lightgbm_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=500,
                            learning_rate=0.03,
                            num_leaves=15,
                            max_depth=4,
                            min_child_samples=8,
                            subsample=0.9,
                            colsample_bytree=0.8,
                            reg_lambda=2.0,
                            class_weight="balanced",
                            random_state=args.random_state,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    ),
                ]
            )
            candidates.append(
                (
                    "lightgbm_selected_features",
                    lightgbm_pipeline,
                    selected_features,
                    selected_name,
                    "microsoft/LightGBM",
                )
            )
            partner = candidates[2][1] if spec["task"] == "position" else candidates[1][1]
            partner_name = "shrinkage_lda" if spec["task"] == "position" else "random_forest"
            candidates.append(
                (
                    f"soft_voting_{partner_name}_lightgbm",
                    VotingClassifier(
                        estimators=[
                            (partner_name, clone(partner)),
                            ("lightgbm", clone(lightgbm_pipeline)),
                        ],
                        voting="soft",
                    ),
                    selected_features,
                    selected_name,
                    "fixed equal-weight scikit-learn and microsoft/LightGBM ensemble",
                )
            )
        if package_status["catboost"]["available"]:
            from catboost import CatBoostClassifier

            candidates.append(
                (
                    "catboost_selected_features",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                CatBoostClassifier(
                                    iterations=500,
                                    depth=5,
                                    learning_rate=0.03,
                                    loss_function="MultiClass",
                                    auto_class_weights="Balanced",
                                    random_seed=args.random_state,
                                    verbose=False,
                                    allow_writing_files=False,
                                ),
                            ),
                        ]
                    ),
                    selected_features,
                    selected_name,
                    "catboost/catboost",
                )
            )

        track_results: list[dict[str, Any]] = []
        for model_id, model, matrix, input_type, source in candidates:
            try:
                metrics, predicted = evaluate_model(
                    model_id,
                    model,
                    matrix[source_index],
                    labels[source_index],
                    matrix[test_index],
                    labels[test_index],
                    label_order,
                    input_type=input_type,
                    algorithm_source=source,
                )
                track_results.append(metrics)
                with (output_dir / "predictions" / f"{track}_{model_id}.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["file_id", "true_label", "pred_label"])
                    writer.writerows(
                        (records[index].file_id, labels[index], pred)
                        for index, pred in zip(test_index, predicted, strict=True)
                    )
                plot_confusion(
                    metrics["confusion_matrix"],
                    label_order,
                    f"{track}: {model_id}",
                    output_dir / "confusion_matrices" / f"{track}_{model_id}.png",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"track": track, "model_id": model_id, "error": repr(exc)})

        try:
            from aeon.classification.convolution_based import MultiRocketHydraClassifier

            model = MultiRocketHydraClassifier(
                n_kernels=8,
                n_groups=args.mr_hydra_groups,
                class_weight="balanced",
                n_jobs=-1,
                random_state=args.random_state,
            )
            metrics, predicted = evaluate_model(
                "multirocket_hydra_multiview",
                model,
                spectral.values[source_index],
                labels[source_index],
                spectral.values[test_index],
                labels[test_index],
                label_order,
                input_type="three_channel_wavelength_axis_sequence",
                algorithm_source="aeon-toolkit/aeon",
            )
            track_results.append(metrics)
            plot_confusion(
                metrics["confusion_matrix"],
                label_order,
                f"{track}: MultiRocket-Hydra",
                output_dir / "confusion_matrices" / f"{track}_multirocket_hydra.png",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"track": track, "model_id": "multirocket_hydra_multiview", "error": repr(exc)}
            )

        try:
            metrics, predicted = evaluate_coral_upper_bound(
                selected_features[source_index],
                labels[source_index],
                selected_features[test_index],
                labels[test_index],
                label_order,
            )
            track_results.append(metrics)
            plot_confusion(
                metrics["confusion_matrix"],
                label_order,
                f"{track}: CORAL transductive upper bound",
                output_dir / "confusion_matrices" / f"{track}_coral_upper_bound.png",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"track": track, "model_id": "coral_svc_transductive_upper_bound", "error": repr(exc)}
            )

        if not package_status["catboost"]["available"]:
            rows.append(
                {
                    "track": track,
                    "task": spec["task"],
                    "model_id": "catboost_selected_features",
                    "input_type": selected_name,
                    "algorithm_source": "catboost/catboost",
                    "evaluation_validity": "not_evaluated",
                    "deployment_ready": "false",
                    "status": "skipped_package_not_installed",
                    "missing_reason": package_status["catboost"].get("reason", "package not installed"),
                }
            )
        results[track] = track_results
        for metrics in track_results:
            rows.append(
                {
                    "track": track,
                    "task": spec["task"],
                    "missing_reason": "",
                    **{
                        key: metrics.get(key)
                        for key in (
                            "model_id",
                            "input_type",
                            "algorithm_source",
                            "evaluation_validity",
                            "deployment_ready",
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                            "training_time_sec",
                            "inference_latency_ms_per_spectrum",
                            "model_size_mb",
                            "predict_proba_available",
                            "confidence_source",
                            "status",
                        )
                    },
                }
            )

    dataset_metadata = {
        "semantics": "one_independent_static_full_spectrum_snapshot_per_csv",
        "num_independent_csv_files": len(records),
        "spectral_tensor_shape": list(spectral.values.shape),
        "wavelength_bins_are_time_steps": False,
        "file_exclusive": True,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(manifest_after, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    write_outputs(output_dir, rows, results, errors, package_status, dataset_metadata)
    plot_comparison(rows, output_dir / "model_comparison_macro_f1.png")
    print(
        json.dumps(
            {"output_dir": str(output_dir), "rows": len(rows), "errors": errors},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
