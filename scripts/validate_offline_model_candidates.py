"""Run repeat-grouped CV and a detailed cross-batch position error audit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_spectral_sequence_models import (  # noqa: E402
    _classification_metrics,
    _fit_sklearn_model,
    _fit_small_cnn,
    _tree_pipeline,
)
from src.hybrid_spectrum.sequence_models import build_spectral_multiview_data  # noqa: E402
from src.hybrid_spectrum.sense_static_dataset import (  # noqa: E402
    assert_dataset_manifest_stable,
    build_static_feature_dataset,
    dataset_source_manifest,
    load_sense_dataset,
    load_training_config,
)
from src.hybrid_spectrum.spatial_fingerprint import (  # noqa: E402
    build_spatial_fingerprint_matrix,
)


POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
FORCE_ORDER = ["light", "normal", "hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "sense_static_training.yaml"
    )
    parser.add_argument(
        "--channel-config", type=Path, default=ROOT / "config" / "hybrid_spectrum_channels.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _aggregate_fold_predictions(
    truth: list[str],
    predicted: list[str],
    label_order: list[str],
    fold_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _classification_metrics(
        np.asarray(truth, dtype=object),
        np.asarray(predicted, dtype=object),
        label_order,
    )
    result["num_folds"] = len(fold_metrics)
    result["fold_macro_f1"] = [float(row["macro_f1"]) for row in fold_metrics]
    result["fold_accuracy"] = [float(row["accuracy"]) for row in fold_metrics]
    result["macro_f1_mean"] = float(np.mean(result["fold_macro_f1"]))
    result["macro_f1_std"] = float(np.std(result["fold_macro_f1"]))
    result["accuracy_mean"] = float(np.mean(result["fold_accuracy"]))
    result["accuracy_std"] = float(np.std(result["fold_accuracy"]))
    return result


def _run_repeat_cv(
    *,
    task: str,
    labels: np.ndarray,
    label_order: list[str],
    manual_mask: np.ndarray,
    repeat_index: np.ndarray,
    engineered: np.ndarray,
    spectral_values: np.ndarray,
    epochs: int,
    patience: int,
    random_state: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    model_ids = ["random_forest_engineered", "small_spectral_1dcnn_multiview"]
    predictions = {model_id: [] for model_id in model_ids}
    truths = {model_id: [] for model_id in model_ids}
    fold_metrics = {model_id: [] for model_id in model_ids}
    repeat_folds = [
        [1, 6, 11],
        [2, 7, 12],
        [3, 8, 13],
        [4, 9, 14],
        [5, 10, 15],
    ]
    all_repeats = sorted({repeat for fold in repeat_folds for repeat in fold})
    for fold_index, test_repeats in enumerate(repeat_folds):
        validation_repeats = repeat_folds[(fold_index + 1) % len(repeat_folds)]
        train_repeats = [
            repeat
            for repeat in all_repeats
            if repeat not in {*test_repeats, *validation_repeats}
        ]
        train = manual_mask & np.isin(repeat_index, train_repeats)
        validation = manual_mask & np.isin(repeat_index, validation_repeats)
        test = manual_mask & np.isin(repeat_index, test_repeats)

        tree_metrics, tree = _fit_sklearn_model(
            "random_forest_engineered",
            _tree_pipeline("random_forest", random_state + fold_index),
            engineered[train],
            labels[train],
            engineered[test],
            labels[test],
            label_order,
        )
        tree_prediction = tree.predict(engineered[test]).tolist()
        truths["random_forest_engineered"].extend(labels[test].tolist())
        predictions["random_forest_engineered"].extend(tree_prediction)
        fold_metrics["random_forest_engineered"].append(tree_metrics)

        cnn_metrics, artifact = _fit_small_cnn(
            spectral_values[train],
            labels[train],
            spectral_values[validation],
            labels[validation],
            spectral_values[test],
            labels[test],
            label_order,
            epochs,
            patience,
            random_state + fold_index,
        )
        truths["small_spectral_1dcnn_multiview"].extend(labels[test].tolist())
        predictions["small_spectral_1dcnn_multiview"].extend(
            artifact["test_predictions"].tolist()
        )
        fold_metrics["small_spectral_1dcnn_multiview"].append(cnn_metrics)

    for model_id in model_ids:
        results[model_id] = _aggregate_fold_predictions(
            truths[model_id],
            predictions[model_id],
            label_order,
            fold_metrics[model_id],
        )
        results[model_id]["task"] = task
        results[model_id]["evaluation_validity"] = (
            "five_fold_grouped_by_repeat_index_file_exclusive"
        )
    return results


def _cross_batch_position_audit(
    records: tuple[Any, ...],
    position_labels: np.ndarray,
    manual_mask: np.ndarray,
    repeat_index: np.ndarray,
    spatial_matrix: np.ndarray,
    spatial_columns: tuple[str, ...],
    random_state: int,
    output_dir: Path,
) -> dict[str, Any]:
    train = manual_mask & (repeat_index <= 5)
    test = manual_mask & (repeat_index >= 6)
    model = ExtraTreesClassifier(
        n_estimators=900,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(spatial_matrix[train], position_labels[train])
    predicted = model.predict(spatial_matrix[test])
    probabilities = model.predict_proba(spatial_matrix[test])
    class_to_column = {label: index for index, label in enumerate(model.classes_)}
    rows: list[dict[str, Any]] = []
    for source_index, prediction, probability in zip(
        np.flatnonzero(test), predicted, probabilities, strict=True
    ):
        ordered = sorted(
            ((label, float(probability[class_to_column[label]])) for label in model.classes_),
            key=lambda item: (-item[1], item[0]),
        )
        truth = str(position_labels[source_index])
        rows.append(
            {
                "file_id": records[source_index].file_id,
                "repeat_index": records[source_index].repeat_index,
                "true_position": truth,
                "predicted_position": str(prediction),
                "correct": truth == str(prediction),
                "top1_probability": ordered[0][1],
                "top2_position": ordered[1][0],
                "top2_probability": ordered[1][1],
                "margin": ordered[0][1] - ordered[1][1],
            }
        )
    with (output_dir / "position_cross_batch_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    importance_order = np.argsort(model.feature_importances_)[::-1]
    importance_rows = [
        {
            "rank": rank,
            "feature": spatial_columns[index],
            "importance": float(model.feature_importances_[index]),
        }
        for rank, index in enumerate(importance_order[:40], start=1)
    ]
    with (output_dir / "position_spatial_feature_importance.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(importance_rows[0]))
        writer.writeheader()
        writer.writerows(importance_rows)

    metrics = _classification_metrics(position_labels[test], predicted, POSITION_ORDER)
    error_pairs = Counter(
        (row["true_position"], row["predicted_position"])
        for row in rows
        if not row["correct"]
    )
    metrics["error_pairs"] = [
        {"true": truth, "predicted": prediction, "count": count}
        for (truth, prediction), count in error_pairs.most_common()
    ]
    metrics["num_low_margin_below_0_20"] = sum(row["margin"] < 0.20 for row in rows)
    metrics["num_errors"] = sum(not row["correct"] for row in rows)
    metrics["num_test_files"] = len(rows)
    return metrics


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
    (output_dir / "source_dataset_manifest.json").write_text(
        json.dumps(manifest_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dataset = build_static_feature_dataset(
        records,
        config.get("feature_extraction", config),
        args.channel_config.resolve(),
    )
    (output_dir / "baseline_cluster_audit.json").write_text(
        json.dumps(
            [asdict(item) for item in dataset.baseline_cluster_assessments],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    spectral = build_spectral_multiview_data(records, dataset)
    spatial, spatial_columns = build_spatial_fingerprint_matrix(
        dataset.engineered_matrix, dataset.engineered_columns
    )
    manual = np.asarray([record.sample_kind == "manual_press" for record in records])
    repeat = np.asarray([record.repeat_index or 0 for record in records], dtype=int)
    position = np.asarray([record.position_label or "" for record in records], dtype=object)
    force = np.asarray([record.manual_force_label or "" for record in records], dtype=object)

    position_cv = _run_repeat_cv(
        task="position",
        labels=position,
        label_order=POSITION_ORDER,
        manual_mask=manual,
        repeat_index=repeat,
        engineered=dataset.engineered_matrix,
        spectral_values=spectral.values,
        epochs=args.epochs,
        patience=args.patience,
        random_state=args.random_state,
    )
    force_cv = _run_repeat_cv(
        task="force",
        labels=force,
        label_order=FORCE_ORDER,
        manual_mask=manual,
        repeat_index=repeat,
        engineered=dataset.engineered_matrix,
        spectral_values=spectral.values,
        epochs=args.epochs,
        patience=args.patience,
        random_state=args.random_state + 100,
    )
    cross_batch = _cross_batch_position_audit(
        records,
        position,
        manual,
        repeat,
        spatial,
        spatial_columns,
        args.random_state,
        output_dir,
    )
    payload = {
        "schema_version": "offline_candidate_validation_v1",
        "num_independent_csv_files": len(records),
        "position_repeat_grouped_cv": position_cv,
        "force_repeat_grouped_cv": force_cv,
        "position_cross_batch_spatial_fingerprint": cross_batch,
        "limitations": [
            "unseen_capture_session_position_generalization_remains_limited",
            "cnn_operates_on_wavelength_axis_not_time",
            "manual_response_levels_are_not_force_N",
            "all_files_are_static_snapshots_from_one_sensor_build",
        ],
    }
    (output_dir / "offline_candidate_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Offline candidate validation",
        "",
        f"- Independent CSV spectra: {len(records)}",
        "- All CV folds are separated by complete repeat index.",
        "- SmallSpectral1DCNN uses wavelength-axis patterns, not temporal windows.",
        "",
        "## Repeat-grouped position CV (repeats 1-5)",
        "",
    ]
    for model_id, metrics in position_cv.items():
        lines.append(
            f"- {model_id}: aggregate accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}, fold macro-F1 {metrics['macro_f1_mean']:.4f} +/- {metrics['macro_f1_std']:.4f}"
        )
    lines.extend(["", "## Repeat-grouped response-level CV (repeats 1-5)", ""])
    for model_id, metrics in force_cv.items():
        lines.append(
            f"- {model_id}: aggregate accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}, fold macro-F1 {metrics['macro_f1_mean']:.4f} +/- {metrics['macro_f1_std']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Cross-batch spatial-fingerprint position audit",
            "",
            f"- Accuracy: {cross_batch['accuracy']:.4f}",
            f"- Macro-F1: {cross_batch['macro_f1']:.4f}",
            f"- Errors: {cross_batch['num_errors']} / {cross_batch['num_test_files']}",
            f"- Low-margin files (<0.20): {cross_batch['num_low_margin_below_0_20']}",
            "",
            "The remaining errors should be reviewed in `position_cross_batch_predictions.csv` before deployment.",
            "Light and normal still require an independent new capture batch.",
        ]
    )
    (output_dir / "offline_candidate_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "cross_batch": cross_batch}, indent=2))


if __name__ == "__main__":
    main()
