"""Train a factorized-position v3 dynamic shadow candidate."""

from __future__ import annotations

import argparse
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
from sklearn.metrics import recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dynamic_sequence_models import (  # noqa: E402
    POSITION_ORDER,
    RESPONSE_ORDER,
    classification_metrics,
    majority_vote_metrics,
)
from scripts.train_dynamic_shadow_candidate import (  # noqa: E402
    ensemble_predict,
    extra_trees,
    plot_confusion,
    rbf_svm,
)
from src.hybrid_spectrum.dynamic_temporal_features import (  # noqa: E402
    temporal_summary_features,
)
from src.hybrid_spectrum.factorized_position import (  # noqa: E402
    FactorizedPositionClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "dynamic_temporal_shadow_candidate_v3_factorized_position.joblib",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stability-seeds", type=int, default=5)
    parser.add_argument(
        "--position-trees",
        type=int,
        default=240,
        help="Number of ExtraTrees members per row/column axis model.",
    )
    return parser.parse_args()


def validate_base_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != "dynamic_temporal_shadow_candidate_v2":
        raise ValueError("base model must be the release-aware v2 shadow candidate")
    if bundle.get("status") != "shadow_only_not_primary":
        raise ValueError("base model is missing its shadow-only safety marker")
    if bundle.get("deployment_ready") is not False:
        raise ValueError("base model must remain deployment blocked")
    release = bundle.get("release_guard_grouped_cv", {})
    if release.get("evaluation_validity") != "grouped_by_capture_group_and_file_id":
        raise ValueError("base release guard lacks grouped validation evidence")
    if int(release.get("unsafe_early_trigger_sequence_count", -1)) != 0:
        raise ValueError("base release guard has unsafe early triggers")


def factorized_model(seed: int, tree_count: int) -> FactorizedPositionClassifier:
    estimator = extra_trees(seed)
    estimator.n_estimators = int(tree_count)
    return FactorizedPositionClassifier(estimator)


def grouped_position_evaluation(
    summary: np.ndarray,
    position: np.ndarray,
    files: np.ndarray,
    groups: np.ndarray,
    contact_mask: np.ndarray,
    seed: int,
    tree_count: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_index, test_group in enumerate(("G1", "G2", "G3")):
        train = contact_mask & (groups != test_group)
        test = contact_mask & (groups == test_group)
        if set(files[train]) & set(files[test]):
            raise AssertionError("source DAT leakage in factorized position evaluation")
        model = factorized_model(seed + fold_index, tree_count).fit(
            summary[train], position[train]
        )
        predicted = np.asarray(model.predict(summary[test]))
        indices = np.flatnonzero(test)
        metrics = classification_metrics(position[test], predicted, POSITION_ORDER)
        metrics.update(
            {
                "test_group": test_group,
                "num_test_windows": len(indices),
            }
        )
        fold_metrics.append(metrics)
        truth_parts.append(position[test])
        prediction_parts.append(predicted)
        index_parts.append(indices)
        for index, predicted_label in zip(indices, predicted):
            prediction_rows.append(
                {
                    "task": "position",
                    "window_index": int(index),
                    "file_id": str(files[index]),
                    "capture_group": str(groups[index]),
                    "stage_label": "",
                    "position_label": str(position[index]),
                    "true_label": str(position[index]),
                    "predicted_label": str(predicted_label),
                }
            )
    truth = np.concatenate(truth_parts)
    predicted = np.concatenate(prediction_parts)
    indices = np.concatenate(index_parts)
    order = np.argsort(indices)
    truth = truth[order]
    predicted = predicted[order]
    indices = indices[order]
    metrics = classification_metrics(truth, predicted, POSITION_ORDER)
    voting = majority_vote_metrics(truth, predicted, files[indices], POSITION_ORDER)
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
            "position_inference_mode": "factorized_row_column_probability_product",
        }
    )
    return metrics, indices, predicted, prediction_rows


def grouped_direct_position_baseline(
    summary: np.ndarray,
    position: np.ndarray,
    groups: np.ndarray,
    contact_mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    prediction = np.empty(len(position), dtype="<U3")
    test_mask = np.zeros(len(position), dtype=bool)
    folds = []
    for fold_index, test_group in enumerate(("G1", "G2", "G3")):
        train = contact_mask & (groups != test_group)
        test = contact_mask & (groups == test_group)
        model = extra_trees(seed + fold_index).fit(summary[train], position[train])
        prediction[test] = model.predict(summary[test])
        test_mask |= test
        folds.append(
            classification_metrics(position[test], prediction[test], POSITION_ORDER)
        )
    metrics = classification_metrics(position[test_mask], prediction[test_mask], POSITION_ORDER)
    metrics["minimum_capture_group_macro_f1"] = min(
        float(item["macro_f1"]) for item in folds
    )
    return metrics


def grouped_response_evaluation(
    summary: np.ndarray,
    stage: np.ndarray,
    files: np.ndarray,
    groups: np.ndarray,
    contact_mask: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[int, str], list[dict[str, Any]]]:
    truth_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fold_index, test_group in enumerate(("G1", "G2", "G3")):
        train = contact_mask & (groups != test_group)
        test = contact_mask & (groups == test_group)
        fold_seed = seed + fold_index
        tree = extra_trees(fold_seed).fit(summary[train], stage[train])
        svm = rbf_svm(fold_seed).fit(summary[train], stage[train])
        predicted, _ = ensemble_predict(
            tree,
            svm,
            summary[test],
            RESPONSE_ORDER,
        )
        indices = np.flatnonzero(test)
        fold = classification_metrics(stage[test], predicted, RESPONSE_ORDER)
        fold.update({"test_group": test_group, "num_test_windows": len(indices)})
        fold_metrics.append(fold)
        truth_parts.append(stage[test])
        prediction_parts.append(predicted)
        index_parts.append(indices)
        for index, predicted_label in zip(indices, predicted):
            rows.append(
                {
                    "task": "response_level",
                    "window_index": int(index),
                    "file_id": str(files[index]),
                    "capture_group": str(groups[index]),
                    "stage_label": str(stage[index]),
                    "position_label": "",
                    "true_label": str(stage[index]),
                    "predicted_label": str(predicted_label),
                }
            )
    truth = np.concatenate(truth_parts)
    predicted = np.concatenate(prediction_parts)
    indices = np.concatenate(index_parts)
    order = np.argsort(indices)
    truth = truth[order]
    predicted = predicted[order]
    indices = indices[order]
    metrics = classification_metrics(truth, predicted, RESPONSE_ORDER)
    vote_keys = np.char.add(np.char.add(files[indices], "::"), stage[indices])
    voting = majority_vote_metrics(truth, predicted, vote_keys, RESPONSE_ORDER)
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
    return metrics, {int(i): str(v) for i, v in zip(indices, predicted)}, rows


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    model_output = args.model_output.resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset.resolve()
    with np.load(dataset_path, allow_pickle=False) as source:
        values = np.asarray(source["X"], dtype=np.float32)
        stage = source["stage_labels"].astype(str)
        position = source["position_labels"].astype(str)
        files = source["file_ids"].astype(str)
        groups = source["capture_groups"].astype(str)
    summary = temporal_summary_features(values)
    contact_mask = stage != "no_contact"

    base_bundle = joblib.load(args.base_model.resolve())
    validate_base_bundle(base_bundle)

    position_metrics, position_indices, position_prediction, position_rows = (
        grouped_position_evaluation(
            summary,
            position,
            files,
            groups,
            contact_mask,
            args.random_state + 100,
            args.position_trees,
        )
    )
    direct_metrics = grouped_direct_position_baseline(
        summary,
        position,
        groups,
        contact_mask,
        args.random_state + 100,
    )
    response_metrics, response_prediction, response_rows = grouped_response_evaluation(
        summary,
        stage,
        files,
        groups,
        contact_mask,
        args.random_state + 200,
    )
    position_prediction_by_index = {
        int(index): str(value)
        for index, value in zip(position_indices, position_prediction)
    }
    combined_indices = np.flatnonzero(contact_mask)
    combined_exact = float(
        np.mean(
            [
                position_prediction_by_index[int(index)] == position[index]
                and response_prediction[int(index)] == stage[index]
                for index in combined_indices
            ]
        )
    )

    stability_rows: list[dict[str, Any]] = []
    for stability_index in range(args.stability_seeds):
        seed = args.random_state + stability_index * 31
        metrics, _, _, _ = grouped_position_evaluation(
            summary,
            position,
            files,
            groups,
            contact_mask,
            seed,
            args.position_trees,
        )
        stability_rows.append(
            {
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "p21_recall": metrics["per_class"]["P21"]["recall"],
                "grouped_vote_accuracy": metrics["grouped_vote_accuracy"],
                "minimum_capture_group_macro_f1": metrics[
                    "minimum_capture_group_macro_f1"
                ],
            }
        )

    final_factorized = factorized_model(
        args.random_state + 100, args.position_trees
    ).fit(
        summary[contact_mask],
        position[contact_mask],
    )
    bundle = dict(base_bundle)
    bundle["schema_version"] = "dynamic_temporal_shadow_candidate_v3"
    bundle["status"] = "shadow_only_not_primary"
    bundle["deployment_ready"] = False
    bundle["position_inference_mode"] = (
        "factorized_row_column_probability_product"
    )
    bundle["position_factorization"] = {
        "channel_id_semantics": "Pxy_x_is_column_y_is_row",
        "display_layout": [
            ["P11", "P21", "P31"],
            ["P12", "P22", "P32"],
            ["P13", "P23", "P33"],
        ],
        "joint_probability": "P(row) * P(column), normalized over nine positions",
        "tree_count_per_axis": int(args.position_trees),
    }
    bundle["models"] = dict(base_bundle["models"])
    bundle["models"]["position_factorized"] = final_factorized
    bundle["grouped_cv_metrics"] = dict(base_bundle["grouped_cv_metrics"])
    bundle["grouped_cv_metrics"]["position"] = position_metrics
    bundle["grouped_cv_metrics"]["response_level"] = response_metrics
    bundle["combined_position_response_exact_accuracy"] = combined_exact
    blockers = list(bundle.get("deployment_blockers", []))
    for blocker in (
        "only_27_independent_DAT_sequences",
        "response_level_G1_generalization_is_weak",
        "live_factorized_position_validation_not_completed",
    ):
        if blocker not in blockers:
            blockers.append(blocker)
    bundle["deployment_blockers"] = blockers
    joblib.dump(bundle, model_output, compress=3)
    joblib.dump(bundle, output_dir / model_output.name, compress=3)

    plot_confusion(
        position_metrics["confusion_matrix"],
        POSITION_ORDER,
        "Factorized row/column position candidate",
        output_dir / "factorized_position_confusion_matrix.png",
    )
    plot_confusion(
        response_metrics["confusion_matrix"],
        RESPONSE_ORDER,
        "Response-level ensemble retained in v3",
        output_dir / "response_level_confusion_matrix.png",
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    names = ["Direct 9-class ET", "Factorized row/column ET"]
    macro = [direct_metrics["macro_f1"], position_metrics["macro_f1"]]
    p21 = [
        direct_metrics["per_class"]["P21"]["recall"],
        position_metrics["per_class"]["P21"]["recall"],
    ]
    x = np.arange(2)
    axis.bar(x - 0.18, macro, width=0.36, label="Macro-F1", color="#2f7f9f")
    axis.bar(x + 0.18, p21, width=0.36, label="P21 recall", color="#d59b3d")
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Grouped score")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "factorized_vs_direct_position.png", dpi=180)
    plt.close(figure)

    with (output_dir / "factorized_shadow_cv_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = position_rows + response_rows
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "factorized_position_seed_stability.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stability_rows[0]))
        writer.writeheader()
        writer.writerows(stability_rows)

    stability_macro = np.asarray([row["macro_f1"] for row in stability_rows])
    stability_p21 = np.asarray([row["p21_recall"] for row in stability_rows])
    result = {
        "schema_version": bundle["schema_version"],
        "status": bundle["status"],
        "deployment_ready": False,
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "independent_dat_files": len(set(files.tolist())),
        "window_count": len(values),
        "direct_position_baseline": direct_metrics,
        "factorized_position": position_metrics,
        "response_level_retained_ensemble": response_metrics,
        "combined_position_response_exact_accuracy": combined_exact,
        "position_tree_count_per_axis": int(args.position_trees),
        "stability": {
            "seed_count": len(stability_rows),
            "macro_f1_mean": float(np.mean(stability_macro)),
            "macro_f1_std": float(np.std(stability_macro)),
            "macro_f1_min": float(np.min(stability_macro)),
            "p21_recall_mean": float(np.mean(stability_p21)),
            "p21_recall_min": float(np.min(stability_p21)),
        },
        "model_path": str(model_output),
    }
    (output_dir / "factorized_dynamic_shadow_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Dynamic shadow candidate v3: factorized position",
        "",
        "This remains a shadow-only candidate. It does not drive the operator UI or digital twin.",
        "",
        "## Grouped results",
        "",
        f"- Direct nine-class position macro-F1: {direct_metrics['macro_f1']:.3f}.",
        f"- Factorized position macro-F1: {position_metrics['macro_f1']:.3f}.",
        f"- Direct P21 recall: {direct_metrics['per_class']['P21']['recall']:.3f}.",
        f"- Factorized P21 recall: {position_metrics['per_class']['P21']['recall']:.3f}.",
        f"- Factorized file voting accuracy: {position_metrics['grouped_vote_accuracy']:.3f}.",
        f"- Retained response-level macro-F1: {response_metrics['macro_f1']:.3f}.",
        f"- Combined exact position + response accuracy: {combined_exact:.3f}.",
        f"- Five-seed factorized macro-F1 mean/std: {np.mean(stability_macro):.3f} / {np.std(stability_macro):.3f}.",
        f"- Compact position model: {args.position_trees} trees per row/column axis.",
        "",
        "## Interpretation",
        "",
        "Pxy is factorized into physical column x and row y. This matches the configured display layout and reduces P21 confusion without changing the optical features.",
        "",
        "## Boundaries",
        "",
        "- Formal evaluation leaves out one complete capture group and all of its DAT files.",
        "- The 2,098 windows originate from only 27 independent DAT sequences.",
        "- Light/normal/hard are approximate manual response levels, not calibrated force.",
        "- The response-level G1 fold remains weak; v3 is not deployment ready.",
        "- Live factorized-position validation is still required before it may drive the digital twin.",
    ]
    (output_dir / "factorized_dynamic_shadow_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(model_output)
    print(json.dumps(result["stability"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
