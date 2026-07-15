"""Measure random-seed stability of deployable dynamic model candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dynamic_sequence_models import (  # noqa: E402
    POSITION_ORDER,
    RESPONSE_ORDER,
    SmallTemporal1DCNN,
    classification_metrics,
    fit_deep_fold,
    fit_minirocket_fold,
    fit_tree_fold,
    majority_vote_metrics,
    temporal_summary_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--minirocket-kernels", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    source = np.load(args.dataset.resolve())
    values = np.asarray(source["X"], dtype=np.float32)
    stage = source["stage_labels"].astype(str)
    position = source["position_labels"].astype(str)
    files = source["file_ids"].astype(str)
    groups = source["capture_groups"].astype(str)
    summary = temporal_summary_features(values)
    contact = stage != "no_contact"
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    candidates = [
        ("position", "extra_trees_temporal_stats", "tree"),
        ("position", "small_temporal_1dcnn", "deep"),
        ("response_level", "extra_trees_temporal_stats", "tree"),
        ("response_level", "minirocket_temporal", "minirocket"),
        ("response_level", "small_temporal_1dcnn", "deep"),
    ]
    rows: list[dict[str, object]] = []
    for task, model_id, family in candidates:
        labels = POSITION_ORDER if task == "position" else RESPONSE_ORDER
        truth_all = position if task == "position" else stage
        vote_keys = (
            files
            if task == "position"
            else np.char.add(np.char.add(files, "::"), stage)
        )
        for seed in seeds:
            fold_truth: list[np.ndarray] = []
            fold_prediction: list[np.ndarray] = []
            fold_keys: list[np.ndarray] = []
            fold_f1: list[float] = []
            for test_index, test_group in enumerate(("G1", "G2", "G3")):
                test_mask = contact & (groups == test_group)
                train_mask = contact & (groups != test_group)
                if family == "tree":
                    predicted, _, _ = fit_tree_fold(
                        model_id,
                        summary[train_mask],
                        truth_all[train_mask],
                        summary[test_mask],
                        seed + test_index,
                    )
                elif family == "minirocket":
                    predicted, _, _ = fit_minirocket_fold(
                        values[train_mask],
                        truth_all[train_mask],
                        values[test_mask],
                        args.minirocket_kernels,
                        seed + test_index,
                    )
                else:
                    remaining = [group for group in ("G1", "G2", "G3") if group != test_group]
                    selection_group = remaining[1 - (test_index % 2)]
                    validation_group = remaining[test_index % 2]
                    selection_mask = contact & (groups == selection_group)
                    validation_mask = contact & (groups == validation_group)
                    predicted, _, _ = fit_deep_fold(
                        model_id,
                        SmallTemporal1DCNN,
                        values[selection_mask],
                        truth_all[selection_mask],
                        values[validation_mask],
                        truth_all[validation_mask],
                        values[test_mask],
                        labels,
                        args.epochs,
                        args.patience,
                        seed + test_index,
                    )
                fold_metric = classification_metrics(
                    truth_all[test_mask], np.asarray(predicted), labels
                )
                fold_f1.append(float(fold_metric["macro_f1"]))
                fold_truth.append(truth_all[test_mask])
                fold_prediction.append(np.asarray(predicted))
                fold_keys.append(vote_keys[test_mask])
            truth = np.concatenate(fold_truth)
            predicted = np.concatenate(fold_prediction)
            keys = np.concatenate(fold_keys)
            metrics = classification_metrics(truth, predicted, labels)
            voting = majority_vote_metrics(truth, predicted, keys, labels)
            row: dict[str, object] = {
                "task": task,
                "model_id": model_id,
                "model_family": family,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "grouped_vote_accuracy": voting["accuracy"],
                "minimum_capture_group_macro_f1": min(fold_f1),
            }
            for label in labels:
                row[f"{label}_recall"] = metrics["per_class"][label]["recall"]
            rows.append(row)
            print(
                f"{task:14s} {model_id:28s} seed={seed} "
                f"macro_f1={metrics['macro_f1']:.3f}",
                flush=True,
            )

    columns = sorted({key for row in rows for key in row})
    with (output_dir / "dynamic_model_seed_stability.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, object]] = []
    for task, model_id, family in candidates:
        selected = [row for row in rows if row["task"] == task and row["model_id"] == model_id]
        macro = np.asarray([float(row["macro_f1"]) for row in selected])
        vote = np.asarray([float(row["grouped_vote_accuracy"]) for row in selected])
        minimum = np.asarray(
            [float(row["minimum_capture_group_macro_f1"]) for row in selected]
        )
        summaries.append(
            {
                "task": task,
                "model_id": model_id,
                "model_family": family,
                "seed_count": len(selected),
                "macro_f1_mean": float(np.mean(macro)),
                "macro_f1_std": float(np.std(macro)),
                "macro_f1_min": float(np.min(macro)),
                "macro_f1_max": float(np.max(macro)),
                "grouped_vote_accuracy_mean": float(np.mean(vote)),
                "minimum_capture_group_macro_f1_mean": float(np.mean(minimum)),
            }
        )
    (output_dir / "dynamic_model_seed_stability.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, task in zip(axes, ("position", "response_level")):
        selected_models = [item for item in summaries if item["task"] == task]
        model_ids = [str(item["model_id"]).replace("_temporal_stats", "") for item in selected_models]
        values_by_model = [
            [
                float(row["macro_f1"])
                for row in rows
                if row["task"] == task and row["model_id"] == item["model_id"]
            ]
            for item in selected_models
        ]
        axis.boxplot(values_by_model, tick_labels=model_ids, showmeans=True)
        axis.set_title(task)
        axis.set_ylabel("Grouped macro-F1")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Dynamic model stability across random seeds")
    figure.tight_layout()
    figure.savefig(output_dir / "dynamic_model_seed_stability.png", dpi=180)
    plt.close(figure)

    report = [
        "# Dynamic model random-seed stability",
        "",
        "All runs use leave-one-capture-group-out evaluation. The 2,098 windows still originate from only 27 independent DAT files.",
        "",
        "| Task | Model | Macro-F1 mean | Std | Min | Grouped vote mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        report.append(
            f"| {item['task']} | {item['model_id']} | {item['macro_f1_mean']:.3f} | "
            f"{item['macro_f1_std']:.3f} | {item['macro_f1_min']:.3f} | "
            f"{item['grouped_vote_accuracy_mean']:.3f} |"
        )
    report.extend(
        [
            "",
            "A high single run is not sufficient for deployment. Candidate selection should consider mean, standard deviation, the worst seed, and the weakest capture-group fold.",
        ]
    )
    (output_dir / "dynamic_model_seed_stability_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
