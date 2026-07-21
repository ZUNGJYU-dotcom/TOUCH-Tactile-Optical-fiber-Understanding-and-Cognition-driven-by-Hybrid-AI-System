"""Benchmark one-physical-frame classifiers under grouped capture validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dynamic_sequence_models import (  # noqa: E402
    POSITION_ORDER,
    RESPONSE_ORDER,
    classification_metrics,
    majority_vote_metrics,
)
from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    load_dynamic_config,
    load_dynamic_feature_sequences,
)
from src.hybrid_spectrum.dynamic_single_spectrum import (  # noqa: E402
    DynamicSingleSpectrumDataset,
    build_dynamic_single_spectrum_dataset,
)


CONTACT_ORDER = ["no_contact", "contact"]
OFFICIAL_SOURCES = {
    "MiniRocket": "https://www.aeon-toolkit.org/en/stable/examples/transformations/minirocket.html",
    "MultiRocket-Hydra": "https://www.aeon-toolkit.org/en/stable/examples/classification/convolution_based.html",
    "QUANT": "https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.interval_based.QUANTClassifier.html",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-frame-stride", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        default=(
            "extra_trees_engineered,lightgbm_engineered,"
            "extra_trees_spectral_flat,minirocket_spectral,"
            "multirocket_spectral,multirocket_hydra_spectral,quant_spectral"
        ),
    )
    return parser.parse_args()


def _extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=240,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _lightgbm(seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=5,
        min_child_samples=15,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _minirocket(seed: int) -> Any:
    from aeon.classification.convolution_based import MiniRocketClassifier

    return MiniRocketClassifier(
        n_kernels=5000,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )


def _multirocket(seed: int) -> Any:
    from aeon.classification.convolution_based import MultiRocketClassifier

    return MultiRocketClassifier(
        n_kernels=2500,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )


def _multirocket_hydra(seed: int) -> Any:
    from aeon.classification.convolution_based import MultiRocketHydraClassifier

    return MultiRocketHydraClassifier(
        n_kernels=8,
        n_groups=32,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )


def _quant(seed: int) -> Any:
    from aeon.classification.interval_based import QUANTClassifier

    return QUANTClassifier(
        interval_depth=5,
        class_weight="balanced",
        random_state=seed,
    )


MODEL_FACTORIES: dict[str, tuple[str, Callable[[int], Any], str]] = {
    "extra_trees_engineered": ("engineered", _extra_trees, "ExtraTrees"),
    "lightgbm_engineered": ("engineered", _lightgbm, "LightGBM"),
    "extra_trees_spectral_flat": ("spectral_flat", _extra_trees, "ExtraTrees"),
    "minirocket_spectral": ("spectral_3d", _minirocket, "MiniRocket"),
    "multirocket_spectral": ("spectral_3d", _multirocket, "MultiRocket"),
    "multirocket_hydra_spectral": (
        "spectral_3d",
        _multirocket_hydra,
        "MultiRocket-Hydra",
    ),
    "quant_spectral": ("spectral_3d", _quant, "QUANT"),
}


def _input_values(dataset: DynamicSingleSpectrumDataset, input_type: str) -> np.ndarray:
    if input_type == "engineered":
        return dataset.engineered_features
    if input_type == "spectral_flat":
        return dataset.spectral_views.reshape(len(dataset.spectral_views), -1)
    if input_type == "spectral_3d":
        return dataset.spectral_views
    raise ValueError(f"unsupported input type: {input_type}")


def _task_data(
    dataset: DynamicSingleSpectrumDataset,
    task: str,
) -> tuple[np.ndarray, list[str]]:
    if task == "contact":
        return np.ones(len(dataset.stage_labels), dtype=bool), CONTACT_ORDER
    contact = dataset.stage_labels != "no_contact"
    if task == "position":
        return contact, POSITION_ORDER
    if task == "response_level":
        return contact, RESPONSE_ORDER
    raise ValueError(task)


def _task_labels(dataset: DynamicSingleSpectrumDataset, task: str) -> np.ndarray:
    if task == "contact":
        return dataset.contact_labels
    if task == "position":
        return dataset.position_labels
    return dataset.stage_labels


def _one_sample_latency_ms(model: Any, sample: np.ndarray) -> tuple[float, float]:
    model.predict(sample)
    values = []
    for _ in range(12):
        start = time.perf_counter_ns()
        model.predict(sample)
        values.append((time.perf_counter_ns() - start) / 1.0e6)
    return float(np.median(values)), float(np.percentile(values, 95.0))


def _serialized_size_mb(model: Any) -> float:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        joblib.dump(model, path, compress=3)
        return float(path.stat().st_size / (1024.0 * 1024.0))


def _plot_confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(int(values[row, column])), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def evaluate_model(
    dataset: DynamicSingleSpectrumDataset,
    *,
    task: str,
    model_id: str,
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_type, factory, family = MODEL_FACTORIES[model_id]
    values = _input_values(dataset, input_type)
    labels = _task_labels(dataset, task)
    mask, label_order = _task_data(dataset, task)
    truth_parts = []
    prediction_parts = []
    index_parts = []
    fold_metrics = []
    training_times = []
    latency_median = []
    latency_p95 = []
    model_sizes = []
    rows: list[dict[str, Any]] = []
    confidence_sources = []

    for fold_index, test_group in enumerate(("G1", "G2", "G3")):
        train = mask & (dataset.capture_groups != test_group)
        test = mask & (dataset.capture_groups == test_group)
        if set(dataset.file_ids[train]) & set(dataset.file_ids[test]):
            raise AssertionError("source DAT leakage in grouped single-spectrum evaluation")
        model = factory(random_state + fold_index)
        start = time.perf_counter()
        model.fit(values[train], labels[train])
        training_times.append(time.perf_counter() - start)
        if hasattr(model, "n_jobs"):
            try:
                model.n_jobs = 1
            except Exception:
                pass
        predicted = np.asarray(model.predict(values[test])).astype(str)
        indices = np.flatnonzero(test)
        fold = classification_metrics(labels[test], predicted, label_order)
        fold.update({"test_group": test_group, "num_test_frames": int(len(indices))})
        fold_metrics.append(fold)
        truth_parts.append(labels[test])
        prediction_parts.append(predicted)
        index_parts.append(indices)
        median_ms, p95_ms = _one_sample_latency_ms(model, values[indices[:1]])
        latency_median.append(median_ms)
        latency_p95.append(p95_ms)
        model_sizes.append(_serialized_size_mb(model))
        confidence_sources.append(
            "predict_proba"
            if hasattr(model, "predict_proba")
            else "decision_function"
            if hasattr(model, "decision_function")
            else "label_only"
        )
        for index, predicted_label in zip(indices, predicted):
            rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "file_id": str(dataset.file_ids[index]),
                    "capture_group": str(dataset.capture_groups[index]),
                    "frame_index": int(dataset.frame_indices[index]),
                    "stage_label": str(dataset.stage_labels[index]),
                    "position_label": str(dataset.position_labels[index]),
                    "true_label": str(labels[index]),
                    "predicted_label": str(predicted_label),
                }
            )

    truth = np.concatenate(truth_parts).astype(str)
    predicted = np.concatenate(prediction_parts).astype(str)
    indices = np.concatenate(index_parts)
    order = np.argsort(indices)
    truth = truth[order]
    predicted = predicted[order]
    indices = indices[order]
    metrics = classification_metrics(truth, predicted, label_order)
    if task == "position":
        vote_keys = dataset.file_ids[indices]
    else:
        vote_keys = np.char.add(
            np.char.add(dataset.file_ids[indices].astype(str), "::"),
            dataset.stage_labels[indices].astype(str),
        )
    voting = majority_vote_metrics(truth, predicted, vote_keys, label_order)
    metrics.update(
        {
            "task": task,
            "model_id": model_id,
            "model_family": family,
            "input_type": input_type,
            "evaluation_validity": "grouped_by_capture_group_and_file_id",
            "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
            "num_test_frames": int(len(indices)),
            "num_independent_files": int(len(set(dataset.file_ids[indices].tolist()))),
            "minimum_capture_group_macro_f1": min(float(item["macro_f1"]) for item in fold_metrics),
            "grouped_vote_accuracy": float(voting["accuracy"]),
            "grouped_vote_macro_f1": float(voting["macro_f1"]),
            "training_time_sec": float(sum(training_times)),
            "inference_latency_median_ms_per_spectrum": float(np.median(latency_median)),
            "inference_latency_p95_ms_per_spectrum": float(max(latency_p95)),
            "model_size_mb_median_fold": float(np.median(model_sizes)),
            "confidence_source": sorted(set(confidence_sources)),
            "folds": fold_metrics,
            "first_output_latency_estimate_sec": float(0.40 + max(latency_p95) / 1000.0),
        }
    )
    return metrics, rows


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir()

    config = load_dynamic_config(args.config)
    sequences = load_dynamic_feature_sequences(config)
    dataset = build_dynamic_single_spectrum_dataset(
        sequences,
        live_frame_stride=args.live_frame_stride,
    )
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(selected_models) - set(MODEL_FACTORIES))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")

    results = []
    prediction_rows: list[dict[str, Any]] = []
    for task in ("contact", "position", "response_level"):
        for model_id in selected_models:
            print(f"[{task}] {model_id}", flush=True)
            try:
                metrics, rows = evaluate_model(
                    dataset,
                    task=task,
                    model_id=model_id,
                    random_state=args.random_state + len(results) * 17,
                )
                metrics["status"] = "completed"
                metrics["missing_reason"] = ""
                results.append(metrics)
                prediction_rows.extend(rows)
                _, labels = _task_data(dataset, task)
                _plot_confusion(
                    metrics["confusion_matrix"],
                    labels,
                    f"{task}: {model_id}",
                    confusion_dir / f"{task}_{model_id}.png",
                )
            except (ImportError, ModuleNotFoundError) as error:
                results.append(
                    {
                        "task": task,
                        "model_id": model_id,
                        "model_family": MODEL_FACTORIES[model_id][2],
                        "input_type": MODEL_FACTORIES[model_id][0],
                        "evaluation_validity": "not_evaluated",
                        "status": "skipped",
                        "missing_reason": f"package not installed: {error}",
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "task": task,
                        "model_id": model_id,
                        "model_family": MODEL_FACTORIES[model_id][2],
                        "input_type": MODEL_FACTORIES[model_id][0],
                        "evaluation_validity": "not_evaluated",
                        "status": "failed",
                        "missing_reason": f"{type(error).__name__}: {error}",
                    }
                )

    leaderboard_fields = [
        "task",
        "model_id",
        "model_family",
        "input_type",
        "evaluation_validity",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "minimum_capture_group_macro_f1",
        "grouped_vote_accuracy",
        "grouped_vote_macro_f1",
        "training_time_sec",
        "inference_latency_median_ms_per_spectrum",
        "inference_latency_p95_ms_per_spectrum",
        "first_output_latency_estimate_sec",
        "model_size_mb_median_fold",
        "confidence_source",
        "status",
        "missing_reason",
    ]
    with (output_dir / "dynamic_single_spectrum_leaderboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    if prediction_rows:
        with (output_dir / "grouped_predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)

    dataset_summary = {
        "num_frames": int(len(dataset.stage_labels)),
        "num_independent_dat_files": int(len(set(dataset.file_ids.tolist()))),
        "capture_groups": sorted(set(dataset.capture_groups.tolist())),
        "engineered_shape": list(dataset.engineered_features.shape),
        "spectral_multiview_shape": list(dataset.spectral_views.shape),
        "spectral_view_names": list(dataset.spectral_view_names),
        "live_frame_stride": int(dataset.live_frame_stride),
        "recorded_frame_interval_sec": float(config["acquisition"]["frame_interval_sec"]),
        "live_physical_frame_interval_sec": 0.40,
        "stage_counts": {
            label: int(np.sum(dataset.stage_labels == label))
            for label in ("no_contact", "light", "normal", "hard")
        },
        "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
        "random_frame_split_allowed": False,
        "transition_and_release_excluded": True,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps({"dataset": dataset_summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    source_lines = ["# Algorithm sources", ""] + [
        f"- {name}: {url}" for name, url in OFFICIAL_SOURCES.items()
    ]
    (output_dir / "algorithm_sources.md").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )

    completed = [row for row in results if row.get("status") == "completed"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, task in zip(axes, ("contact", "position", "response_level")):
        rows = sorted(
            [row for row in completed if row["task"] == task],
            key=lambda row: float(row["macro_f1"]),
        )
        axis.barh([row["model_id"] for row in rows], [row["macro_f1"] for row in rows], color="#2b8cbe")
        axis.set_title(task)
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("One-physical-frame grouped macro-F1")
    figure.tight_layout()
    figure.savefig(output_dir / "dynamic_single_spectrum_model_comparison.png", dpi=180)
    plt.close(figure)

    report = [
        "# Dynamic one-physical-frame algorithm benchmark",
        "",
        "## Scope and validity",
        "",
        f"- Independent DAT files: {dataset_summary['num_independent_dat_files']}.",
        f"- Stable sampled frames: {dataset_summary['num_frames']}.",
        "- Formal split: leave one capture group out (G1/G2/G3), with source file exclusivity.",
        "- Random frame split is prohibited.",
        "- Transition and release frames are excluded.",
        "- Blind-test files are not used.",
        "- Input is one baseline-relative physical spectrum; wavelength bins are not time steps.",
        "- First-output estimate includes one 0.40 s BaySpec acquisition plus measured model inference.",
        "",
        "## Results",
        "",
    ]
    for task in ("contact", "position", "response_level"):
        report.extend([f"### {task}", "", "| Model | Macro-F1 | Min-group F1 | Vote accuracy | p95 ms | First output s |", "|---|---:|---:|---:|---:|---:|"])
        task_rows = sorted(
            [row for row in completed if row["task"] == task],
            key=lambda row: float(row["macro_f1"]),
            reverse=True,
        )
        for row in task_rows:
            report.append(
                f"| {row['model_id']} | {row['macro_f1']:.4f} | "
                f"{row['minimum_capture_group_macro_f1']:.4f} | "
                f"{row['grouped_vote_accuracy']:.4f} | "
                f"{row['inference_latency_p95_ms_per_spectrum']:.2f} | "
                f"{row['first_output_latency_estimate_sec']:.3f} |"
            )
        report.append("")
    report.extend(
        [
            "## Safety boundary",
            "",
            "- This benchmark does not replace the deployed model.",
            "- Light/normal/hard remain approximate manual response levels, not force_N.",
            "- A candidate must improve grouped accuracy and worst-group stability before live shadow validation.",
        ]
    )
    (output_dir / "dynamic_single_spectrum_benchmark_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
