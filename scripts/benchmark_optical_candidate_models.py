"""Benchmark compact spectral classifiers and hierarchical optical inference.

This script is offline-only.  It uses the frozen latest-primary dataset and
its immutable grouped-by-session folds.  It does not alter the TOUCH runtime,
UI, deployed model bundle, or executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    POSITION_ORDER,
    build_feature_views,
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.optical_candidate_benchmark import (  # noqa: E402
    candidate_specs,
    evaluate_hierarchical_predictions,
    grouped_candidate_classification,
)


DEFAULT_FUSION_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_all_data_fusion_20260731_v1"
    / "all_source_fusion_dataset.npz"
)
DEFAULT_SPECTRUM_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260731_latest_primary"
    / "primary"
    / "ordinary_fbg_px6d_dataset.npz"
)
DEFAULT_RICH_LEADERBOARD = (
    PROJECT_ROOT
    / "outputs"
    / "rich_optical_algorithm_benchmark_20260801"
    / "rich_optical_model_leaderboard.csv"
)
DEFAULT_RICH_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "rich_optical_algorithm_benchmark_20260801"
    / "grouped_out_of_fold_predictions.csv"
)
DEFAULT_ADVANCED_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "advanced_optical_feature_benchmark_20260801"
    / "out_of_fold_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "optical_algorithm_and_tactile_information_audit_20260802"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION_DATASET)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM_DATASET)
    parser.add_argument("--rich-leaderboard", type=Path, default=DEFAULT_RICH_LEADERBOARD)
    parser.add_argument("--rich-predictions", type=Path, default=DEFAULT_RICH_PREDICTIONS)
    parser.add_argument(
        "--advanced-predictions", type=Path, default=DEFAULT_ADVANCED_PREDICTIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model ids. By default every configured candidate is run.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _candidate_row(
    *,
    metrics: dict[str, Any],
    task: str,
    feature_view: str,
    feature_count: int,
) -> dict[str, Any]:
    report = metrics.get("classification_report", {})
    row: dict[str, Any] = {
        "result_source": "new_candidate_benchmark",
        "model_id": metrics["model_id"],
        "model_family": metrics["model_family"],
        "feature_view": feature_view,
        "feature_count": feature_count,
        "task": task,
        "split_strategy": "immutable_grouped_by_session_id_5fold",
        "evaluation_validity": "formal_grouped_evaluation",
        "frame_count": metrics["frame_count"],
        "group_count": metrics["session_count"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "group_voting_accuracy": metrics["group_voting_accuracy"],
        "group_voting_macro_f1": metrics["group_voting_macro_f1"],
        "training_time_sec": metrics["training_time_sec"],
        "inference_latency_ms_per_frame": metrics[
            "inference_latency_ms_per_frame"
        ],
        "confidence_source": metrics["confidence_source"],
    }
    if task == "contact":
        row["no_contact_recall"] = float(report.get("0", {}).get("recall", np.nan))
        row["contact_recall"] = float(report.get("1", {}).get("recall", np.nan))
        row["false_contact_rate"] = 1.0 - row["no_contact_recall"]
    for position in POSITION_ORDER:
        row[f"{position.lower()}_recall"] = float(
            report.get(position, {}).get("recall", np.nan)
        )
    return row


def _append_prediction_rows(
    rows: list[pd.DataFrame],
    *,
    dataset: Any,
    model_id: str,
    task: str,
    predicted: np.ndarray,
    evaluated_mask: np.ndarray,
    target: np.ndarray,
) -> None:
    selected = np.flatnonzero(evaluated_mask)
    rows.append(
        pd.DataFrame(
            {
                "model_id": model_id,
                "feature_view": "full_spectrum_192",
                "task": task,
                "session_id": dataset.group_id[selected],
                "capture_index": dataset.sample_index[selected],
                "fold_id": dataset.fold_id[selected],
                "true_value": target[selected],
                "predicted_value": predicted[selected],
            }
        )
    )


def _existing_contact_prediction(
    *,
    path: Path,
    dataset: Any,
    model_id: str,
    feature_view: str,
) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["model_id"] == model_id)
        & (frame["feature_view"] == feature_view)
        & (frame["task"] == "contact")
    ].copy()
    if selected.empty:
        return None
    lookup = {
        (str(row.group_id), int(row.sample_index)): int(float(row.predicted_value))
        for row in selected.itertuples(index=False)
    }
    predicted = np.full(len(dataset.group_id), -1, dtype=int)
    for index, (group, sample) in enumerate(
        zip(dataset.group_id, dataset.sample_index)
    ):
        value = lookup.get((str(group), int(sample)))
        if value is not None:
            predicted[index] = value
    if np.any(predicted[dataset.contact_mask] < 0):
        return None
    return predicted


def _advanced_contact_prediction(
    *,
    path: Path,
    dataset: Any,
    feature_view: str,
) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    selected = frame.loc[frame["feature_view"] == feature_view].copy()
    if selected.empty:
        return None
    lookup = {
        (str(row.group_id), int(row.sample_index)): int(float(row.contact_predicted))
        for row in selected.itertuples(index=False)
        if bool(row.contact_evaluated) and pd.notna(row.contact_predicted)
    }
    predicted = np.full(len(dataset.group_id), -1, dtype=int)
    for index, (group, sample) in enumerate(
        zip(dataset.group_id, dataset.sample_index)
    ):
        value = lookup.get((str(group), int(sample)))
        if value is not None:
            predicted[index] = value
    if np.any(predicted[dataset.contact_mask] < 0):
        return None
    return predicted


def _hierarchy_row(
    *,
    pipeline_id: str,
    contact_model_id: str,
    position_model_id: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "contact_model_id": contact_model_id,
        "position_model_id": position_model_id,
        "split_strategy": "immutable_grouped_by_session_id_5fold",
        "evaluation_validity": "formal_grouped_evaluation",
        **{
            key: metrics[key]
            for key in (
                "accuracy",
                "macro_f1",
                "no_contact_recall",
                "contact_recall",
                "false_contact_rate",
                "missed_contact_rate",
                "conditional_position_accuracy",
                "frame_count",
            )
        },
    }


def _plot_candidate_comparison(leaderboard: pd.DataFrame, output_path: Path) -> None:
    contact = leaderboard.loc[leaderboard["task"] == "contact"].copy()
    position = leaderboard.loc[leaderboard["task"] == "position"].copy()
    model_order = contact.sort_values("macro_f1", ascending=False)["model_id"].tolist()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    colors = ["#168db5", "#66b8b0", "#e0b458", "#d67868", "#788ca2", "#9678b8"]
    x = np.arange(len(model_order))
    axes[0].bar(
        x,
        [
            float(contact.loc[contact["model_id"] == model, "macro_f1"].iloc[0])
            for model in model_order
        ],
        color=colors[: len(model_order)],
    )
    axes[0].set_title("Contact macro-F1")
    axes[1].bar(
        x,
        [
            float(
                contact.loc[contact["model_id"] == model, "no_contact_recall"].iloc[
                    0
                ]
            )
            for model in model_order
        ],
        color=colors[: len(model_order)],
    )
    axes[1].set_title("No-contact recall")
    axes[2].bar(
        x,
        [
            float(position.loc[position["model_id"] == model, "macro_f1"].iloc[0])
            for model in model_order
        ],
        color=colors[: len(model_order)],
    )
    axes[2].set_title("Position macro-F1")
    for axis in axes:
        axis.set_xticks(x, model_order, rotation=28, ha="right")
        axis.set_ylim(0.0, 1.01)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Compact spectral candidates - grouped by 50 capture sessions",
        fontsize=15,
        weight="bold",
    )
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def _plot_hierarchy(hierarchy: pd.DataFrame, output_path: Path) -> None:
    ordered = hierarchy.sort_values("macro_f1", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    colors = ["#c76d62" if value < 0.90 else "#1996b7" for value in ordered["macro_f1"]]
    axes[0].barh(ordered["pipeline_id"], ordered["macro_f1"], color=colors)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("End-to-end 10-class macro-F1")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(
        ordered["pipeline_id"], ordered["false_contact_rate"], color="#d9944e"
    )
    axes[1].set_xlim(0.0, max(0.20, float(ordered["false_contact_rate"].max()) * 1.1))
    axes[1].set_title("Idle false-contact rate (lower is better)")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("Hierarchical contact gate + position head", fontsize=15, weight="bold")
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    leaderboard: pd.DataFrame,
    hierarchy: pd.DataFrame,
    frame_count: int,
    session_count: int,
) -> None:
    contact = leaderboard.loc[leaderboard["task"] == "contact"].sort_values(
        ["macro_f1", "no_contact_recall"], ascending=False
    )
    position = leaderboard.loc[leaderboard["task"] == "position"].sort_values(
        "macro_f1", ascending=False
    )
    best_hierarchy = hierarchy.sort_values(
        ["macro_f1", "no_contact_recall"], ascending=False
    ).iloc[0]
    contact_best = contact.iloc[0]
    position_best = position.iloc[0]
    lines = [
        "# Optical recognition candidate benchmark",
        "",
        "## Formal boundary",
        "",
        f"- Latest-primary aligned frames: {frame_count:,}.",
        f"- Independent capture sessions: {session_count}.",
        "- Every formal score uses the immutable five folds grouped by `session_id`.",
        "- No random frame split is used; adjacent frames from one capture never cross train/test boundaries.",
        "- Only Z-axis force was deliberately applied. Fx/Fy and moments are not counted as tactile targets.",
        "- This benchmark does not modify the UI, runtime model, recording path, deployment bundle, or EXE.",
        "",
        "## Candidate result",
        "",
        f"- Best new contact candidate: `{contact_best['model_id']}`, macro-F1 **{contact_best['macro_f1']:.4f}**, no-contact recall **{contact_best['no_contact_recall']:.4f}**.",
        f"- Best new position candidate: `{position_best['model_id']}`, macro-F1 **{position_best['macro_f1']:.4f}**.",
        f"- Best end-to-end hierarchy: `{best_hierarchy['pipeline_id']}`, 10-class macro-F1 **{best_hierarchy['macro_f1']:.4f}**, idle false-contact rate **{best_hierarchy['false_contact_rate']:.4f}**.",
        "",
        "The hierarchy is the scientifically appropriate comparison: an idle frame must first pass the contact gate before a P11-P33 prediction is exposed. Position-only accuracy can otherwise look excellent while the software still jumps to a location when nothing is touching the sensor.",
        "",
        "## Algorithm interpretation",
        "",
        "- ExtraTrees and LightGBM remain strong nonlinear baselines for the present limited number of independent sessions.",
        "- PLS-DA and shrinkage LDA test whether a compact chemometric latent representation generalizes better than a large tree ensemble.",
        "- Linear and RBF SVM test margin-based recognition; their decision scores are not calibrated probabilities.",
        "- The full-spectrum input already contains baseline log-ratio, shape-normalized residual, and derivative views. This is more informative than using nine peak positions alone.",
        "",
        "## Important limitation",
        "",
        "The 192 wavelength-domain features are not a time series. ROCKET, TCN, InceptionTime, or LSTM must receive consecutive spectra over time, not the 512 wavelength bins of one frame. Those temporal models remain valuable, but they require a separate grouped sequence dataset.",
        "",
        "## Deployment decision",
        "",
        "No candidate is deployed by this script. A candidate should replace the current runtime only after replay tests demonstrate lower idle false positives, correct release recovery, and stable live behavior across independent sessions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    views = build_feature_views(dataset)
    full_spectrum = views["full_spectrum_192"]
    selected_specs = candidate_specs()
    if args.models:
        requested = set(args.models)
        selected_specs = tuple(spec for spec in selected_specs if spec.model_id in requested)
        unknown = requested - {spec.model_id for spec in selected_specs}
        if unknown:
            raise ValueError("unknown candidate model(s): " + ", ".join(sorted(unknown)))

    position_labels = [
        label for label in POSITION_ORDER if np.any(dataset.position_target == label)
    ]
    formal_hierarchy_mask = dataset.contact_mask & (
        (dataset.contact_target == 0) | dataset.position_mask
    )
    metrics_payload: dict[str, Any] = {
        "scope": {
            "frame_count": int(len(dataset.group_id)),
            "contact_evaluation_frame_count": int(np.sum(dataset.contact_mask)),
            "position_evaluation_frame_count": int(np.sum(dataset.position_mask)),
            "independent_session_count": int(len(set(dataset.group_id.tolist()))),
            "split_strategy": "immutable_grouped_by_session_id_5fold",
            "random_frame_split_used": False,
            "deployment_changed": False,
            "only_deliberately_applied_force_axis": "Fz",
        },
        "candidates": {},
        "hierarchical_pipelines": {},
    }
    leaderboard_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    contact_predictions: dict[str, np.ndarray] = {}
    position_predictions: dict[str, np.ndarray] = {}

    total_jobs = len(selected_specs) * 2
    job = 0
    for spec in selected_specs:
        job += 1
        print(f"[{job:02d}/{total_jobs:02d}] contact | {spec.model_id}", flush=True)
        contact_metrics, contact_predicted = grouped_candidate_classification(
            spec=spec,
            features=full_spectrum,
            target=dataset.contact_target,
            training_and_evaluation_mask=dataset.contact_mask,
            fold_id=dataset.fold_id,
            group_id=dataset.group_id,
            labels=[0, 1],
            seed=args.seed,
        )
        contact_predictions[spec.model_id] = contact_predicted
        metrics_payload["candidates"][f"{spec.model_id}__contact"] = contact_metrics
        leaderboard_rows.append(
            _candidate_row(
                metrics=contact_metrics,
                task="contact",
                feature_view="full_spectrum_192",
                feature_count=full_spectrum.shape[1],
            )
        )
        _append_prediction_rows(
            prediction_rows,
            dataset=dataset,
            model_id=spec.model_id,
            task="contact",
            predicted=contact_predicted,
            evaluated_mask=dataset.contact_mask,
            target=dataset.contact_target,
        )

        job += 1
        print(f"[{job:02d}/{total_jobs:02d}] position | {spec.model_id}", flush=True)
        position_metrics, position_predicted = grouped_candidate_classification(
            spec=spec,
            features=full_spectrum,
            target=dataset.position_target,
            training_and_evaluation_mask=dataset.position_mask,
            fold_id=dataset.fold_id,
            group_id=dataset.group_id,
            labels=position_labels,
            seed=args.seed + 100,
            predict_all_fold_rows=True,
        )
        position_predictions[spec.model_id] = position_predicted
        metrics_payload["candidates"][f"{spec.model_id}__position"] = position_metrics
        leaderboard_rows.append(
            _candidate_row(
                metrics=position_metrics,
                task="position",
                feature_view="full_spectrum_192",
                feature_count=full_spectrum.shape[1],
            )
        )
        _append_prediction_rows(
            prediction_rows,
            dataset=dataset,
            model_id=spec.model_id,
            task="position",
            predicted=position_predicted,
            evaluated_mask=dataset.position_mask,
            target=dataset.position_target,
        )

    leaderboard = pd.DataFrame(leaderboard_rows)
    hierarchy_rows: list[dict[str, Any]] = []
    hierarchy_details: dict[str, dict[str, Any]] = {}
    hierarchy_labels = ["no_contact", *position_labels]
    for model_id in contact_predictions:
        metrics = evaluate_hierarchical_predictions(
            contact_prediction=contact_predictions[model_id],
            position_prediction=position_predictions[model_id],
            contact_target=dataset.contact_target,
            position_target=dataset.position_target,
            eligible_mask=formal_hierarchy_mask,
            labels=hierarchy_labels,
        )
        pipeline_id = f"{model_id}_contact__{model_id}_position"
        hierarchy_rows.append(
            _hierarchy_row(
                pipeline_id=pipeline_id,
                contact_model_id=model_id,
                position_model_id=model_id,
                metrics=metrics,
            )
        )
        hierarchy_details[pipeline_id] = metrics

    best_contact_id = str(
        leaderboard.loc[leaderboard["task"] == "contact"]
        .sort_values(["macro_f1", "no_contact_recall"], ascending=False)
        .iloc[0]["model_id"]
    )
    best_position_id = str(
        leaderboard.loc[leaderboard["task"] == "position"]
        .sort_values("macro_f1", ascending=False)
        .iloc[0]["model_id"]
    )
    cross_id = f"{best_contact_id}_contact__{best_position_id}_position"
    if cross_id not in hierarchy_details:
        metrics = evaluate_hierarchical_predictions(
            contact_prediction=contact_predictions[best_contact_id],
            position_prediction=position_predictions[best_position_id],
            contact_target=dataset.contact_target,
            position_target=dataset.position_target,
            eligible_mask=formal_hierarchy_mask,
            labels=hierarchy_labels,
        )
        hierarchy_rows.append(
            _hierarchy_row(
                pipeline_id=cross_id,
                contact_model_id=best_contact_id,
                position_model_id=best_position_id,
                metrics=metrics,
            )
        )
        hierarchy_details[cross_id] = metrics

    existing_contact_sources: list[tuple[str, np.ndarray | None]] = [
        (
            "existing_lightgbm_rich_contact",
            _existing_contact_prediction(
                path=args.rich_predictions.resolve(),
                dataset=dataset,
                model_id="lightgbm",
                feature_view="rich_plus_full_spectrum_192",
            ),
        )
    ]
    peak_temporal = _advanced_contact_prediction(
        path=args.advanced_predictions.resolve(),
        dataset=dataset,
        feature_view="peak_temporal_483",
    )
    full_spectrum_264 = _advanced_contact_prediction(
        path=args.advanced_predictions.resolve(),
        dataset=dataset,
        feature_view="full_spectrum_264",
    )
    dual_confirmation = None
    if peak_temporal is not None and full_spectrum_264 is not None:
        dual_confirmation = (
            (peak_temporal == 1) & (full_spectrum_264 == 1)
        ).astype(int)
    existing_contact_sources.append(
        ("existing_dual_view_confirmation", dual_confirmation)
    )
    for contact_id, contact_prediction in existing_contact_sources:
        if contact_prediction is None:
            continue
        metrics = evaluate_hierarchical_predictions(
            contact_prediction=contact_prediction,
            position_prediction=position_predictions[best_position_id],
            contact_target=dataset.contact_target,
            position_target=dataset.position_target,
            eligible_mask=formal_hierarchy_mask,
            labels=hierarchy_labels,
        )
        pipeline_id = f"{contact_id}__{best_position_id}_position"
        hierarchy_rows.append(
            _hierarchy_row(
                pipeline_id=pipeline_id,
                contact_model_id=contact_id,
                position_model_id=best_position_id,
                metrics=metrics,
            )
        )
        hierarchy_details[pipeline_id] = metrics

    hierarchy = pd.DataFrame(hierarchy_rows).sort_values(
        ["macro_f1", "no_contact_recall"], ascending=False
    )
    metrics_payload["hierarchical_pipelines"] = hierarchy_details
    leaderboard.to_csv(output_dir / "candidate_model_leaderboard.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "candidate_grouped_predictions.csv", index=False
    )
    hierarchy.to_csv(output_dir / "hierarchical_pipeline_comparison.csv", index=False)

    combined = leaderboard.copy()
    if args.rich_leaderboard.exists():
        reference = pd.read_csv(args.rich_leaderboard)
        reference["result_source"] = "existing_grouped_benchmark"
        reference["confidence_source"] = "not_recorded_in_reference_leaderboard"
        combined = pd.concat((combined, reference), ignore_index=True, sort=False)
    combined.to_csv(output_dir / "combined_model_leaderboard.csv", index=False)
    (output_dir / "candidate_model_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _plot_candidate_comparison(
        leaderboard, output_dir / "candidate_model_comparison.png"
    )
    _plot_hierarchy(hierarchy, output_dir / "hierarchical_pipeline_comparison.png")
    _write_report(
        output_dir / "candidate_algorithm_report.md",
        leaderboard=leaderboard,
        hierarchy=hierarchy,
        frame_count=len(dataset.group_id),
        session_count=len(set(dataset.group_id.tolist())),
    )
    print(f"Wrote candidate benchmark: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
