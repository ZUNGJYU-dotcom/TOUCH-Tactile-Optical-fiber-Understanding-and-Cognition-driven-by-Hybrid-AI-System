"""Train and package a non-primary dynamic temporal shadow candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dynamic_sequence_models import (  # noqa: E402
    CONTACT_ORDER,
    POSITION_ORDER,
    RESPONSE_ORDER,
    classification_metrics,
    majority_vote_metrics,
)
from src.hybrid_spectrum.dynamic_temporal_features import (  # noqa: E402
    SUMMARY_FEATURE_BLOCK_ORDER,
    temporal_summary_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "dynamic_temporal_shadow_candidate_v1.joblib",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tree-count",
        type=int,
        default=700,
        help="ExtraTrees members for contact, position, and response models.",
    )
    return parser.parse_args()


def extra_trees(seed: int, tree_count: int = 700) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(tree_count),
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def rbf_svm(seed: int) -> Pipeline:
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


def aligned_probability(model: Any, values: np.ndarray, labels: list[str]) -> np.ndarray:
    probability = np.asarray(model.predict_proba(values), dtype=float)
    aligned = np.zeros((len(values), len(labels)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        aligned[:, labels.index(str(label))] = probability[:, source_index]
    return aligned


def ensemble_predict(
    tree: Any,
    svm: Any,
    values: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    probability = 0.5 * aligned_probability(tree, values, labels) + 0.5 * aligned_probability(
        svm, values, labels
    )
    return np.asarray(labels)[np.argmax(probability, axis=1)], probability


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


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    model_output = args.model_output.resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset.resolve()
    source = np.load(dataset_path)
    values = np.asarray(source["X"], dtype=np.float32)
    summary = temporal_summary_features(values)
    stage = source["stage_labels"].astype(str)
    contact = np.where(stage == "no_contact", "no_contact", "contact")
    position = source["position_labels"].astype(str)
    files = source["file_ids"].astype(str)
    groups = source["capture_groups"].astype(str)
    feature_names = source["feature_names"].astype(str).tolist()
    contact_mask = stage != "no_contact"

    task_data = {
        "contact": {
            "mask": np.ones(len(values), dtype=bool),
            "truth": contact,
            "labels": CONTACT_ORDER,
            "vote_keys": np.char.add(np.char.add(files, "::"), stage),
            "mode": "tree",
        },
        "position": {
            "mask": contact_mask,
            "truth": position,
            "labels": POSITION_ORDER,
            "vote_keys": files,
            "mode": "ensemble",
        },
        "response_level": {
            "mask": contact_mask,
            "truth": stage,
            "labels": RESPONSE_ORDER,
            "vote_keys": np.char.add(np.char.add(files, "::"), stage),
            "mode": "ensemble",
        },
    }
    metrics_by_task: dict[str, Any] = {}
    cv_predictions: dict[str, dict[int, str]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for task_index, (task, specification) in enumerate(task_data.items()):
        mask = np.asarray(specification["mask"], dtype=bool)
        truth_source = np.asarray(specification["truth"])
        labels = list(specification["labels"])
        fold_truth: list[np.ndarray] = []
        fold_prediction: list[np.ndarray] = []
        fold_indices: list[np.ndarray] = []
        fold_metrics: list[dict[str, Any]] = []
        for fold_index, test_group in enumerate(("G1", "G2", "G3")):
            train = mask & (groups != test_group)
            test = mask & (groups == test_group)
            if set(files[train]) & set(files[test]):
                raise AssertionError("source DAT leakage in shadow candidate evaluation")
            seed = args.random_state + task_index * 100 + fold_index
            tree = extra_trees(seed, args.tree_count).fit(
                summary[train], truth_source[train]
            )
            if specification["mode"] == "ensemble":
                svm = rbf_svm(seed).fit(summary[train], truth_source[train])
                predicted, _ = ensemble_predict(tree, svm, summary[test], labels)
            else:
                predicted = np.asarray(tree.predict(summary[test]))
            indices = np.flatnonzero(test)
            fold_result = classification_metrics(truth_source[test], predicted, labels)
            fold_result["test_group"] = test_group
            fold_result["num_test_windows"] = len(indices)
            fold_metrics.append(fold_result)
            fold_truth.append(truth_source[test])
            fold_prediction.append(predicted)
            fold_indices.append(indices)

        truth = np.concatenate(fold_truth)
        predicted = np.concatenate(fold_prediction)
        indices = np.concatenate(fold_indices)
        order = np.argsort(indices)
        truth = truth[order]
        predicted = predicted[order]
        indices = indices[order]
        metrics = classification_metrics(truth, predicted, labels)
        voting = majority_vote_metrics(
            truth,
            predicted,
            np.asarray(specification["vote_keys"])[indices],
            labels,
        )
        metrics.update(
            {
                "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
                "evaluation_validity": "grouped_by_capture_group_and_file_id",
                "grouped_vote_accuracy": voting["accuracy"],
                "grouped_vote_macro_f1": voting["macro_f1"],
                "minimum_capture_group_macro_f1": min(
                    float(item["macro_f1"]) for item in fold_metrics
                ),
                "folds": fold_metrics,
            }
        )
        metrics_by_task[task] = metrics
        cv_predictions[task] = {
            int(index): str(value) for index, value in zip(indices, predicted)
        }
        for index, true_label, predicted_label in zip(indices, truth, predicted):
            prediction_rows.append(
                {
                    "task": task,
                    "window_index": int(index),
                    "file_id": files[index],
                    "capture_group": groups[index],
                    "stage_label": stage[index],
                    "position_label": position[index],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }
            )
        plot_confusion(
            metrics["confusion_matrix"],
            labels,
            f"Dynamic shadow candidate / {task}",
            output_dir / f"{task}_confusion_matrix.png",
        )

    contact_indices = np.flatnonzero(contact_mask)
    combined_correct = np.asarray(
        [
            cv_predictions["position"][int(index)] == position[index]
            and cv_predictions["response_level"][int(index)] == stage[index]
            for index in contact_indices
        ]
    )
    combined_exact_accuracy = float(np.mean(combined_correct))

    # Final fit uses all 27 independent files and is packaged as shadow-only.
    contact_model = extra_trees(args.random_state, args.tree_count).fit(
        summary, contact
    )
    position_tree = extra_trees(args.random_state + 1, args.tree_count).fit(
        summary[contact_mask], position[contact_mask]
    )
    position_svm = rbf_svm(args.random_state + 1).fit(
        summary[contact_mask], position[contact_mask]
    )
    response_tree = extra_trees(args.random_state + 2, args.tree_count).fit(
        summary[contact_mask], stage[contact_mask]
    )
    response_svm = rbf_svm(args.random_state + 2).fit(
        summary[contact_mask], stage[contact_mask]
    )
    source_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    bundle = {
        "schema_version": "dynamic_temporal_shadow_candidate_v1",
        "status": "shadow_only_not_primary",
        "deployment_ready": False,
        "deployment_blockers": [
            "only_27_independent_DAT_sequences",
            "P21_position_recall_requires_more_data",
            "live_stream_shadow_validation_not_completed",
            "probabilities_not_calibrated",
        ],
        "source_dataset_sha256": source_digest,
        "source_dataset_path": str(dataset_path),
        "independent_dat_files": len(set(files.tolist())),
        "window_count": len(values),
        "time_steps": int(values.shape[1]),
        "frame_interval_sec_estimated": 0.04,
        "estimated_window_duration_sec": float(values.shape[1] * 0.04),
        "frame_feature_names": feature_names,
        "summary_feature_block_order": SUMMARY_FEATURE_BLOCK_ORDER,
        "summary_feature_count": int(summary.shape[1]),
        "input_semantics": "nine_FBG_mixed_shift_intensity_shape_temporal_window",
        "response_level_semantics": "approximate_manual_level_not_force_N",
        "label_order": {
            "contact": CONTACT_ORDER,
            "position": POSITION_ORDER,
            "response_level": RESPONSE_ORDER,
        },
        "ensemble_weights": {"extra_trees": 0.5, "rbf_svm": 0.5},
        "tree_count_per_extra_trees_model": int(args.tree_count),
        "models": {
            "contact_extra_trees": contact_model,
            "position_extra_trees": position_tree,
            "position_rbf_svm": position_svm,
            "response_extra_trees": response_tree,
            "response_rbf_svm": response_svm,
        },
        "grouped_cv_metrics": metrics_by_task,
        "combined_position_response_exact_accuracy": combined_exact_accuracy,
    }
    joblib.dump(bundle, model_output, compress=3)
    joblib.dump(bundle, output_dir / model_output.name, compress=3)

    with (output_dir / "shadow_candidate_cv_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    summary_json = {
        "model_path": str(model_output),
        "model_size_mb": model_output.stat().st_size / (1024.0 * 1024.0),
        "status": bundle["status"],
        "deployment_ready": False,
        "metrics": metrics_by_task,
        "combined_position_response_exact_accuracy": combined_exact_accuracy,
    }
    (output_dir / "dynamic_shadow_candidate_metrics.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Dynamic temporal shadow candidate",
        "",
        "This candidate is packaged for offline and future live shadow validation. It does not replace the current primary software model.",
        "",
        f"- Contact macro-F1: {metrics_by_task['contact']['macro_f1']:.3f}.",
        f"- Position macro-F1: {metrics_by_task['position']['macro_f1']:.3f}.",
        f"- Response-level macro-F1: {metrics_by_task['response_level']['macro_f1']:.3f}.",
        f"- Exact position + response level accuracy: {combined_exact_accuracy:.3f}.",
        f"- P21 recall: {metrics_by_task['position']['per_class']['P21']['recall']:.3f}.",
        f"- ExtraTrees members per model: {args.tree_count}.",
        "- Formal split: leave one complete capture group out; no random window split.",
        "- Source independence: 27 DAT sequences, not 2,098 independent experiments.",
        "- Light/normal/hard remain approximate manual response levels, not force_N.",
        "",
        "## Decision",
        "",
        "Keep as shadow-only. More independent P21 sequences and a live press/release validation are required before promotion.",
    ]
    (output_dir / "dynamic_shadow_candidate_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(model_output)
    print(json.dumps({"combined_exact_accuracy": combined_exact_accuracy}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
