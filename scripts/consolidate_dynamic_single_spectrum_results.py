"""Consolidate grouped one-spectrum benchmarks and evaluate fixed ensembles."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dynamic_sequence_models import (  # noqa: E402
    POSITION_ORDER,
    RESPONSE_ORDER,
    classification_metrics,
    majority_vote_metrics,
)


CONTACT_ORDER = ["no_contact", "contact"]
KEY_COLUMNS = [
    "file_id",
    "capture_group",
    "frame_index",
    "stage_label",
    "position_label",
    "true_label",
]
ENSEMBLES = {
    "accuracy_contact_vote": {
        "task": "contact",
        "models": [
            "extra_trees_spectral_flat",
            "quant_spectral",
            "lightgbm_engineered",
        ],
    },
    "accuracy_position_vote": {
        "task": "position",
        "models": [
            "multirocket_hydra_spectral",
            "extra_trees_engineered",
            "lightgbm_engineered",
        ],
    },
    "accuracy_response_vote": {
        "task": "response_level",
        "models": [
            "quant_spectral",
            "multirocket_spectral",
            "minirocket_spectral",
        ],
    },
}
PIPELINES = {
    "fast_single_spectrum": {
        "contact": ["extra_trees_spectral_flat"],
        "position": ["extra_trees_engineered"],
        "response_level": ["extra_trees_spectral_flat"],
        "selection_status": "predeclared_latency_first_candidate",
    },
    "accuracy_single_spectrum_ensemble": {
        "contact": ENSEMBLES["accuracy_contact_vote"]["models"],
        "position": ENSEMBLES["accuracy_position_vote"]["models"],
        "response_level": ENSEMBLES["accuracy_response_vote"]["models"],
        "selection_status": "exploratory_oof_selected_not_independently_validated",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--current-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models"
            / "candidates"
            / "dynamic_temporal_shadow_candidate_v3_compact_runtime_pos240.joblib"
        ),
    )
    return parser.parse_args()


def _read_inputs(input_dirs: list[Path]) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    dataset_summary: dict[str, Any] | None = None
    seen: set[tuple[str, str]] = set()
    for directory in input_dirs:
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "grouped_predictions.csv"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        current_summary = dict(payload["dataset"])
        if dataset_summary is None:
            dataset_summary = current_summary
        else:
            for key in (
                "num_frames",
                "num_independent_dat_files",
                "engineered_shape",
                "spectral_multiview_shape",
                "live_frame_stride",
                "split_strategy",
            ):
                if current_summary.get(key) != dataset_summary.get(key):
                    raise ValueError(f"dataset mismatch for {key}: {directory}")
        for result in payload["results"]:
            identity = (str(result.get("task")), str(result.get("model_id")))
            if identity in seen:
                raise ValueError(f"duplicate benchmark result: {identity}")
            seen.add(identity)
            results.append(result)
        frame = pd.read_csv(predictions_path, keep_default_na=False)
        prediction_frames.append(frame)
    if dataset_summary is None:
        raise ValueError("no benchmark inputs were loaded")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return results, predictions, dataset_summary


def _task_order(task: str) -> list[str]:
    if task == "contact":
        return CONTACT_ORDER
    if task == "position":
        return POSITION_ORDER
    if task == "response_level":
        return RESPONSE_ORDER
    raise ValueError(task)


def _task_wide(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    subset = predictions[predictions["task"] == task].copy()
    if subset.empty:
        raise ValueError(f"no predictions for task {task}")
    wide = subset.pivot_table(
        index=KEY_COLUMNS,
        columns="model_id",
        values="predicted_label",
        aggfunc="first",
    ).reset_index()
    return wide


def _majority_labels(wide: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    missing = [model_id for model_id in model_ids if model_id not in wide.columns]
    if missing:
        raise ValueError(f"ensemble models missing from predictions: {missing}")
    output: list[str] = []
    for row in wide[model_ids].astype(str).to_numpy():
        counts = Counter(row)
        maximum = max(counts.values())
        tied = {label for label, count in counts.items() if count == maximum}
        output.append(next(label for label in row if label in tied))
    return np.asarray(output)


def _grouped_metrics(
    wide: pd.DataFrame,
    predicted: np.ndarray,
    task: str,
) -> dict[str, Any]:
    labels = _task_order(task)
    truth = wide["true_label"].astype(str).to_numpy()
    metrics = classification_metrics(truth, predicted, labels)
    folds = []
    for group in ("G1", "G2", "G3"):
        mask = wide["capture_group"].astype(str).to_numpy() == group
        fold = classification_metrics(truth[mask], predicted[mask], labels)
        fold["test_group"] = group
        folds.append(fold)
    if task == "position":
        vote_keys = wide["file_id"].astype(str).to_numpy()
    else:
        vote_keys = (
            wide["file_id"].astype(str) + "::" + wide["stage_label"].astype(str)
        ).to_numpy()
    voting = majority_vote_metrics(truth, predicted, vote_keys, labels)
    metrics.update(
        {
            "minimum_capture_group_macro_f1": min(
                float(fold["macro_f1"]) for fold in folds
            ),
            "grouped_vote_accuracy": float(voting["accuracy"]),
            "grouped_vote_macro_f1": float(voting["macro_f1"]),
            "folds": folds,
        }
    )
    return metrics


def _base_lookup(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(result["task"]), str(result["model_id"])): result
        for result in results
        if result.get("status") == "completed"
    }


def _ensemble_metrics(
    predictions: pd.DataFrame,
    results: list[dict[str, Any]],
    ensemble_id: str,
    specification: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    task = str(specification["task"])
    models = list(specification["models"])
    wide = _task_wide(predictions, task)
    predicted = _majority_labels(wide, models)
    metrics = _grouped_metrics(wide, predicted, task)
    lookup = _base_lookup(results)
    components = [lookup[(task, model_id)] for model_id in models]
    metrics.update(
        {
            "task": task,
            "model_id": ensemble_id,
            "model_family": "majority_vote_ensemble",
            "input_type": "mixed",
            "component_models": models,
            "evaluation_validity": "grouped_oof_exploratory_ensemble",
            "selection_status": "selected_on_same_grouped_oof_predictions",
            "inference_latency_p95_ms_per_spectrum": float(
                sum(
                    float(component["inference_latency_p95_ms_per_spectrum"])
                    for component in components
                )
            ),
            "model_size_mb_median_fold": float(
                sum(float(component["model_size_mb_median_fold"]) for component in components)
            ),
            "first_output_latency_estimate_sec": float(
                0.40
                + sum(
                    float(component["inference_latency_p95_ms_per_spectrum"])
                    for component in components
                )
                / 1000.0
            ),
            "status": "completed",
            "deployment_ready": False,
        }
    )
    output = wide[KEY_COLUMNS].copy()
    output["task"] = task
    output["model_id"] = ensemble_id
    output["predicted_label"] = predicted
    return metrics, output


def _pipeline_task_predictions(
    predictions: pd.DataFrame,
    task: str,
    model_ids: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    wide = _task_wide(predictions, task)
    if len(model_ids) == 1:
        predicted = wide[model_ids[0]].astype(str).to_numpy()
    else:
        predicted = _majority_labels(wide, model_ids)
    return wide, predicted


def _pipeline_metrics(
    predictions: pd.DataFrame,
    results: list[dict[str, Any]],
    pipeline_id: str,
    specification: dict[str, Any],
) -> dict[str, Any]:
    task_predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    lookup = _base_lookup(results)
    p95_total = 0.0
    size_total = 0.0
    for task in ("contact", "position", "response_level"):
        model_ids = list(specification[task])
        task_predictions[task] = _pipeline_task_predictions(
            predictions, task, model_ids
        )
        for model_id in model_ids:
            component = lookup[(task, model_id)]
            p95_total += float(component["inference_latency_p95_ms_per_spectrum"])
            size_total += float(component["model_size_mb_median_fold"])

    contact_wide, contact_predicted = task_predictions["contact"]
    position_wide, position_predicted = task_predictions["position"]
    response_wide, response_predicted = task_predictions["response_level"]
    position_map = {
        (str(row.file_id), int(row.frame_index)): str(label)
        for row, label in zip(position_wide.itertuples(), position_predicted)
    }
    response_map = {
        (str(row.file_id), int(row.frame_index)): str(label)
        for row, label in zip(response_wide.itertuples(), response_predicted)
    }
    truth: list[str] = []
    predicted: list[str] = []
    groups: list[str] = []
    for row, contact_label in zip(contact_wide.itertuples(), contact_predicted):
        key = (str(row.file_id), int(row.frame_index))
        truth.append(
            "no_contact"
            if row.stage_label == "no_contact"
            else f"{row.position_label}|{row.stage_label}"
        )
        predicted.append(
            "no_contact"
            if contact_label == "no_contact"
            else f"{position_map.get(key, 'unknown')}|{response_map.get(key, 'unknown')}"
        )
        groups.append(str(row.capture_group))
    truth_values = np.asarray(truth)
    predicted_values = np.asarray(predicted)
    group_values = np.asarray(groups)
    label_order = sorted(set(truth_values) | set(predicted_values))
    combined = classification_metrics(truth_values, predicted_values, label_order)
    group_f1 = {
        group: float(
            classification_metrics(
                truth_values[group_values == group],
                predicted_values[group_values == group],
                label_order,
            )["macro_f1"]
        )
        for group in ("G1", "G2", "G3")
    }
    tasks = {
        task: _grouped_metrics(wide, labels, task)
        for task, (wide, labels) in task_predictions.items()
    }
    return {
        "pipeline_id": pipeline_id,
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "selection_status": specification["selection_status"],
        "component_models": {
            task: list(specification[task])
            for task in ("contact", "position", "response_level")
        },
        "combined_exact_accuracy": float(combined["accuracy"]),
        "combined_exact_macro_f1": float(combined["macro_f1"]),
        "combined_minimum_capture_group_macro_f1": min(group_f1.values()),
        "combined_group_macro_f1": group_f1,
        "contact_macro_f1": float(tasks["contact"]["macro_f1"]),
        "position_macro_f1": float(tasks["position"]["macro_f1"]),
        "response_level_macro_f1": float(tasks["response_level"]["macro_f1"]),
        "light_recall": float(tasks["response_level"]["per_class"]["light"]["recall"]),
        "normal_recall": float(tasks["response_level"]["per_class"]["normal"]["recall"]),
        "hard_recall": float(tasks["response_level"]["per_class"]["hard"]["recall"]),
        "serial_inference_p95_ms_per_spectrum": p95_total,
        "first_output_latency_estimate_sec": 0.40 + p95_total / 1000.0,
        "model_size_mb_median_fold_sum": size_total,
        "deployment_ready": False,
    }


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: result.get(key, "")
        for key in (
            "task",
            "model_id",
            "model_family",
            "input_type",
            "evaluation_validity",
            "accuracy",
            "macro_f1",
            "minimum_capture_group_macro_f1",
            "grouped_vote_accuracy",
            "grouped_vote_macro_f1",
            "training_time_sec",
            "inference_latency_p95_ms_per_spectrum",
            "first_output_latency_estimate_sec",
            "model_size_mb_median_fold",
            "status",
            "selection_status",
        )
    }
    per_class = result.get("per_class") or {}
    for label in CONTACT_ORDER + POSITION_ORDER + RESPONSE_ORDER:
        row[f"{label}_recall"] = (per_class.get(label) or {}).get("recall", "")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _current_model_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "model_path": str(path)}
    bundle = joblib.load(path)
    grouped = bundle.get("grouped_cv_metrics") or {}
    return {
        "available": True,
        "model_path": str(path.resolve()),
        "schema_version": bundle.get("schema_version"),
        "time_steps": bundle.get("time_steps"),
        "estimated_window_duration_sec": bundle.get("estimated_window_duration_sec"),
        "combined_position_response_exact_accuracy": bundle.get(
            "combined_position_response_exact_accuracy"
        ),
        "contact_macro_f1": (grouped.get("contact") or {}).get("macro_f1"),
        "position_macro_f1": (grouped.get("position") or {}).get("macro_f1"),
        "response_level_macro_f1": (grouped.get("response_level") or {}).get(
            "macro_f1"
        ),
        "deployment_ready": bundle.get("deployment_ready"),
        "comparison_warning": (
            "Temporal overlapping-window metrics are not directly interchangeable "
            "with one-physical-spectrum frame metrics."
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    input_dirs = [path.resolve() for path in args.input_dir]
    results, predictions, dataset_summary = _read_inputs(input_dirs)

    ensemble_results: list[dict[str, Any]] = []
    ensemble_prediction_frames: list[pd.DataFrame] = []
    for ensemble_id, specification in ENSEMBLES.items():
        metrics, frame = _ensemble_metrics(
            predictions, results, ensemble_id, specification
        )
        ensemble_results.append(metrics)
        ensemble_prediction_frames.append(frame)

    pipeline_results = [
        _pipeline_metrics(predictions, results, pipeline_id, specification)
        for pipeline_id, specification in PIPELINES.items()
    ]
    current_model = _current_model_summary(args.current_model.resolve())

    _write_csv(
        output_dir / "model_leaderboard.csv",
        [_flatten_result(result) for result in results],
    )
    _write_csv(
        output_dir / "ensemble_leaderboard.csv",
        [_flatten_result(result) for result in ensemble_results],
    )
    _write_csv(output_dir / "pipeline_comparison.csv", pipeline_results)
    pd.concat(ensemble_prediction_frames, ignore_index=True).to_csv(
        output_dir / "ensemble_grouped_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "dataset": dataset_summary,
        "base_results": results,
        "ensemble_results": ensemble_results,
        "pipeline_results": pipeline_results,
        "current_temporal_model_reference": current_model,
    }
    (output_dir / "consolidated_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    completed = [result for result in results if result.get("status") == "completed"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for axis, task in zip(axes, ("contact", "position", "response_level")):
        rows = [result for result in completed if result.get("task") == task]
        rows += [result for result in ensemble_results if result.get("task") == task]
        rows.sort(key=lambda result: float(result["macro_f1"]))
        colors = [
            "#d0872f" if result.get("model_family") == "majority_vote_ensemble" else "#2b8cbe"
            for result in rows
        ]
        axis.barh(
            [str(result["model_id"]) for result in rows],
            [float(result["macro_f1"]) for result in rows],
            color=colors,
        )
        axis.set_title(task)
        axis.set_xlim(0.65 if task != "contact" else 0.90, 1.0)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Grouped one-spectrum model and exploratory ensemble macro-F1")
    figure.tight_layout()
    figure.savefig(output_dir / "model_and_ensemble_comparison.png", dpi=180)
    plt.close(figure)

    fast = next(
        result for result in pipeline_results if result["pipeline_id"] == "fast_single_spectrum"
    )
    accuracy = next(
        result
        for result in pipeline_results
        if result["pipeline_id"] == "accuracy_single_spectrum_ensemble"
    )
    report = [
        "# Dynamic single-spectrum model consolidation",
        "",
        "## Scope",
        "",
        f"- Independent DAT files: {dataset_summary['num_independent_dat_files']}.",
        f"- Stable frames: {dataset_summary['num_frames']} at live-equivalent stride {dataset_summary['live_frame_stride']}.",
        "- Formal split: leave one complete capture group out (G1/G2/G3), with file exclusivity.",
        "- Blind-test files, transition frames, and release frames are excluded.",
        "- One input sample is a physical spectrum, not an interpolated time history.",
        "",
        "## Main findings",
        "",
        f"- Fast pipeline task macro-F1: contact {fast['contact_macro_f1']:.4f}, position {fast['position_macro_f1']:.4f}, response {fast['response_level_macro_f1']:.4f}.",
        f"- Fast pipeline combined exact accuracy: {fast['combined_exact_accuracy']:.4f}; estimated first output {fast['first_output_latency_estimate_sec']:.3f} s.",
        f"- Accuracy ensemble task macro-F1: contact {accuracy['contact_macro_f1']:.4f}, position {accuracy['position_macro_f1']:.4f}, response {accuracy['response_level_macro_f1']:.4f}.",
        f"- Accuracy ensemble combined exact accuracy: {accuracy['combined_exact_accuracy']:.4f}; estimated first output {accuracy['first_output_latency_estimate_sec']:.3f} s.",
        f"- Accuracy ensemble response recall: light {accuracy['light_recall']:.4f}, normal {accuracy['normal_recall']:.4f}, hard {accuracy['hard_recall']:.4f}.",
        "",
        "## Decision",
        "",
        "- Keep the deployed temporal model unchanged.",
        "- Build the fast ExtraTrees pipeline only as an offline shadow candidate.",
        "- The accuracy ensemble was selected after inspecting the same grouped OOF predictions, so it requires independent captures before deployment.",
        "- MultiRocket/QUANT improve some mean metrics but are less stable across capture groups or slower than the fast tree pipeline.",
        "- Light/normal/hard are approximate manual response levels, not force_N.",
    ]
    if current_model.get("available"):
        report.extend(
            [
                "",
                "## Current temporal reference",
                "",
                f"- Time steps: {current_model.get('time_steps')}; nominal trained window: {current_model.get('estimated_window_duration_sec')} s.",
                f"- Grouped macro-F1: contact {current_model.get('contact_macro_f1'):.4f}, position {current_model.get('position_macro_f1'):.4f}, response {current_model.get('response_level_macro_f1'):.4f}.",
                "- These overlapping-window metrics are context only and are not directly interchangeable with the one-spectrum evaluation.",
            ]
        )
    (output_dir / "dynamic_single_spectrum_consolidated_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
