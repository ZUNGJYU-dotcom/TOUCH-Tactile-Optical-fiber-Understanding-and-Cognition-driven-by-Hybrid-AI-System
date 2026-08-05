"""Audit capture-session baseline metadata as a recognition confounder."""

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
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.optical_tactile_information import (  # noqa: E402
    session_baseline_metadata_mask,
)
from src.hybrid_spectrum.rich_optical_benchmark import (  # noqa: E402
    FeatureView,
    build_rich_feature_views,
    grouped_classification,
    grouped_force_regression,
)
from src.hybrid_spectrum.rich_optical_features import (  # noqa: E402
    load_rich_feature_cache,
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
DEFAULT_RICH_CACHE = (
    PROJECT_ROOT
    / "outputs"
    / "rich_optical_algorithm_benchmark_20260801"
    / "rich_optical_feature_cache.npz"
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
    parser.add_argument("--rich-cache", type=Path, default=DEFAULT_RICH_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--minimum-leaf-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _feature_stability(
    values: np.ndarray,
    names: np.ndarray,
    groups: np.ndarray,
    selected: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_groups = np.unique(groups)
    for index in np.flatnonzero(selected):
        total_variance = float(np.var(values[:, index]))
        mean_within = float(
            np.mean(
                [
                    float(np.var(values[groups == group, index]))
                    for group in unique_groups
                ]
            )
        )
        rows.append(
            {
                "feature": str(names[index]),
                "total_variance": total_variance,
                "mean_within_session_variance": mean_within,
                "within_to_total_variance_ratio": (
                    mean_within / total_variance if total_variance > 1.0e-12 else 0.0
                ),
                "excluded_from_response_only_view": True,
                "recommended_role": "qa_and_drift_monitoring_only",
            }
        )
    return pd.DataFrame(rows)


def _add_predictions(
    rows: list[pd.DataFrame],
    *,
    dataset: Any,
    model_id: str,
    view_id: str,
    task: str,
    predicted: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
) -> None:
    indices = np.flatnonzero(mask)
    rows.append(
        pd.DataFrame(
            {
                "model_id": model_id,
                "feature_view": view_id,
                "task": task,
                "session_id": dataset.group_id[indices],
                "sample_index": dataset.sample_index[indices],
                "fold_id": dataset.fold_id[indices],
                "true_value": target[indices],
                "predicted_value": predicted[indices],
            }
        )
    )


def _plot(leaderboard: pd.DataFrame, output_path: Path) -> None:
    panels = (
        ("contact", "macro_f1", "Contact macro-F1", False),
        ("contact", "no_contact_recall", "No-contact recall", False),
        ("position", "macro_f1", "Position macro-F1", False),
        ("force_fz", "mae_n", "Optical Fz MAE (N)", True),
    )
    views = leaderboard["feature_view"].drop_duplicates().tolist()
    labels = {
        "full_spectrum_192": "Full spectrum",
        "rich_plus_full_all": "Rich + spectrum\n(all metadata)",
        "rich_plus_full_no_session_baseline": "Rich + spectrum\n(no session baseline)",
    }
    colors = {"extra_trees": "#1598bd", "lightgbm": "#d57366"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    x = np.arange(len(views))
    width = 0.34
    for axis, (task, metric, title, lower_is_better) in zip(axes.flat, panels):
        finite: list[float] = []
        for model_index, model_id in enumerate(("extra_trees", "lightgbm")):
            values: list[float] = []
            for view_id in views:
                match = leaderboard.loc[
                    (leaderboard["task"] == task)
                    & (leaderboard["model_id"] == model_id)
                    & (leaderboard["feature_view"] == view_id),
                    metric,
                ]
                value = float(match.iloc[0]) if len(match) else float("nan")
                values.append(value)
                if np.isfinite(value):
                    finite.append(value)
            bars = axis.bar(
                x + (model_index - 0.5) * width,
                values,
                width,
                color=colors[model_id],
                label=model_id.replace("_", " ").title(),
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
        axis.set_xticks(x, [labels[view] for view in views])
        axis.set_title(title, weight="bold")
        axis.grid(axis="y", alpha=0.2)
        if finite and not lower_is_better:
            axis.set_ylim(max(0.0, min(finite) - 0.06), 1.01)
        axis.legend(loc="best")
    fig.suptitle(
        "Capture-session baseline metadata ablation\n"
        "Same frames and immutable grouped session folds",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    leaderboard: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    frame_count: int,
    group_count: int,
    excluded_count: int,
) -> None:
    def metric(model: str, view: str, task: str, column: str) -> float:
        return float(
            leaderboard.loc[
                (leaderboard.model_id == model)
                & (leaderboard.feature_view == view)
                & (leaderboard.task == task),
                column,
            ].iloc[0]
        )

    lines = [
        "# Capture-Session Baseline Confound Ablation",
        "",
        "## Boundary",
        "",
        f"- {frame_count:,} synchronized latest-primary frames from {group_count} independent sessions.",
        "- All scores use the preassigned five folds grouped by immutable `session_id`.",
        "- No random frame split, live device access, runtime replacement, UI change, or EXE rebuild was performed.",
        "",
        "## Why this audit was necessary",
        "",
        f"The rich cache contains {excluded_count} `baseline_peak_snr/valid` fields. Their median within-session to total variance ratio is {stability['within_to_total_variance_ratio'].median():.3e}. They are effectively capture-session descriptors rather than current tactile-response measurements.",
        "",
        "Baseline SNR remains useful for QA, drift detection, and domain monitoring. It should not carry the recognition decision unless cross-day and remount holdout proves that it generalizes.",
        "",
        "## Grouped ablation result",
        "",
    ]
    for model in ("extra_trees", "lightgbm"):
        all_view = "rich_plus_full_all"
        clean_view = "rich_plus_full_no_session_baseline"
        lines.extend(
            [
                f"### {model.replace('_', ' ').title()}",
                "",
                f"- Contact macro-F1: {metric(model, all_view, 'contact', 'macro_f1'):.4f} -> {metric(model, clean_view, 'contact', 'macro_f1'):.4f}.",
                f"- No-contact recall: {metric(model, all_view, 'contact', 'no_contact_recall'):.4f} -> {metric(model, clean_view, 'contact', 'no_contact_recall'):.4f}.",
                f"- Position macro-F1: {metric(model, all_view, 'position', 'macro_f1'):.4f} -> {metric(model, clean_view, 'position', 'macro_f1'):.4f}.",
                f"- Optical Fz MAE: {metric(model, all_view, 'force_fz', 'mae_n'):.4f} N -> {metric(model, clean_view, 'force_fz', 'mae_n'):.4f} N.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "The response-only view is the scientifically safer candidate for future deployment work. Baseline-only fields should stay in Diagnostics/QA. Position must also be confirmed with cross-day, remount, and randomized acquisition-order holdouts because grouped session folds cannot remove every acquisition-condition correlation when position and collection order are coupled.",
            "",
            "This audit does not claim a new champion and does not change the current Beta model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    cache = load_rich_feature_cache(
        args.rich_cache.resolve(),
        expected_group_id=dataset.group_id,
        expected_sample_index=dataset.sample_index,
    )
    base_views = build_rich_feature_views(dataset, cache)
    excluded = session_baseline_metadata_mask(cache.feature_names)
    if int(np.sum(excluded)) != 18:
        raise ValueError(f"expected 18 session baseline fields, found {np.sum(excluded)}")
    keep = ~excluded
    spectrum = base_views["full_spectrum_192"]
    all_rich = base_views["rich_optical_physics"]
    clean_rich = FeatureView(cache.features[:, keep], cache.feature_names[keep])
    views = {
        "full_spectrum_192": spectrum,
        "rich_plus_full_all": FeatureView(
            np.concatenate((all_rich.values, spectrum.values), axis=1),
            np.concatenate(
                (
                    np.char.add("rich__", all_rich.names.astype(str)),
                    np.char.add("spectrum__", spectrum.names.astype(str)),
                )
            ),
        ),
        "rich_plus_full_no_session_baseline": FeatureView(
            np.concatenate((clean_rich.values, spectrum.values), axis=1),
            np.concatenate(
                (
                    np.char.add("rich__", clean_rich.names.astype(str)),
                    np.char.add("spectrum__", spectrum.names.astype(str)),
                )
            ),
        ),
    }
    stability = _feature_stability(
        cache.features, cache.feature_names, dataset.group_id, excluded
    )
    stability.to_csv(output_dir / "baseline_metadata_feature_audit.csv", index=False)

    position_labels = [
        label for label in POSITION_ORDER if np.any(dataset.position_target == label)
    ]
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    metrics_payload: dict[str, Any] = {
        "scope": {
            "frame_count": int(len(dataset.group_id)),
            "independent_session_count": int(len(np.unique(dataset.group_id))),
            "split_strategy": "preassigned_grouped_by_session_id_5fold",
            "random_frame_split_used": False,
            "live_hardware_used": False,
            "runtime_changed": False,
            "excluded_session_baseline_feature_count": int(np.sum(excluded)),
        },
        "models": {},
    }
    total = 2 * len(views) * 3
    job = 0
    for model_id in ("extra_trees", "lightgbm"):
        model_payload: dict[str, Any] = {}
        for view_id, view in views.items():
            view_payload: dict[str, Any] = {}
            for task in ("contact", "position", "force_fz"):
                job += 1
                print(f"[{job:02d}/{total:02d}] {model_id} | {view_id} | {task}", flush=True)
                if task == "contact":
                    target = dataset.contact_target
                    mask = dataset.contact_mask
                    metrics, predicted = grouped_classification(
                        model_id=model_id,
                        feature_view=view,
                        target=target,
                        mask=mask,
                        fold_id=dataset.fold_id,
                        group_id=dataset.group_id,
                        labels=[0, 1],
                        estimators=args.estimators,
                        minimum_leaf_samples=args.minimum_leaf_samples,
                        seed=args.seed,
                    )
                    true = target[mask]
                    selected = np.asarray(predicted[mask], dtype=int)
                    metrics["no_contact_recall"] = float(np.mean(selected[true == 0] == 0))
                    metrics["contact_recall"] = float(np.mean(selected[true == 1] == 1))
                elif task == "position":
                    target = dataset.position_target
                    mask = dataset.position_mask
                    metrics, predicted = grouped_classification(
                        model_id=model_id,
                        feature_view=view,
                        target=target,
                        mask=mask,
                        fold_id=dataset.fold_id,
                        group_id=dataset.group_id,
                        labels=position_labels,
                        estimators=args.estimators,
                        minimum_leaf_samples=args.minimum_leaf_samples,
                        seed=args.seed + 100,
                    )
                else:
                    target = dataset.force_fz_n
                    mask = dataset.force_mask
                    metrics, predicted = grouped_force_regression(
                        model_id=model_id,
                        feature_view=view,
                        target=target,
                        mask=mask,
                        fold_id=dataset.fold_id,
                        group_id=dataset.group_id,
                        estimators=args.estimators,
                        minimum_leaf_samples=args.minimum_leaf_samples,
                        seed=args.seed + 200,
                    )
                view_payload[task] = metrics
                rows.append(
                    {
                        "model_id": model_id,
                        "feature_view": view_id,
                        "feature_count": int(view.values.shape[1]),
                        "task": task,
                        "split_strategy": "preassigned_grouped_by_session_id_5fold",
                        "evaluation_validity": "formal_grouped_evaluation",
                        "accuracy": metrics.get("accuracy", np.nan),
                        "macro_f1": metrics.get("macro_f1", np.nan),
                        "no_contact_recall": metrics.get("no_contact_recall", np.nan),
                        "contact_recall": metrics.get("contact_recall", np.nan),
                        "group_voting_accuracy": metrics.get("group_voting_accuracy", np.nan),
                        "mae_n": metrics.get("mae_n", np.nan),
                        "rmse_n": metrics.get("rmse_n", np.nan),
                        "r2": metrics.get("r2", np.nan),
                        "inference_latency_ms_per_frame": metrics["inference_latency_ms_per_frame"],
                    }
                )
                _add_predictions(
                    predictions,
                    dataset=dataset,
                    model_id=model_id,
                    view_id=view_id,
                    task=task,
                    predicted=predicted,
                    mask=mask,
                    target=target,
                )
            model_payload[view_id] = view_payload
        metrics_payload["models"][model_id] = model_payload

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(output_dir / "baseline_confound_ablation_leaderboard.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "baseline_confound_grouped_predictions.csv", index=False
    )
    (output_dir / "baseline_confound_ablation_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _plot(leaderboard, output_dir / "baseline_confound_ablation.png")
    _write_report(
        output_dir / "baseline_confound_ablation_report.md",
        leaderboard,
        stability,
        frame_count=len(dataset.group_id),
        group_count=len(np.unique(dataset.group_id)),
        excluded_count=int(np.sum(excluded)),
    )
    print(f"Completed: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
