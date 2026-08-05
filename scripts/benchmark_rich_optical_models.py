"""Benchmark efficient models and richer optical evidence without deployment."""

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
from src.hybrid_spectrum.rich_optical_benchmark import (  # noqa: E402
    build_rich_feature_views,
    grouped_classification,
    grouped_force_regression,
    model_available,
)
from src.hybrid_spectrum.rich_optical_features import (  # noqa: E402
    build_aligned_rich_feature_cache,
    load_rich_feature_cache,
    save_rich_feature_cache,
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
DEFAULT_CAPTURE_ROOT = Path(
    r"E:\重要文档\实验\柔性传感\光纤\Micro-FBG\普通FBG\data\new data"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rich_optical_algorithm_benchmark_20260801"
VIEW_ORDER = (
    "peak_current_40",
    "full_spectrum_192",
    "rich_optical_physics",
    "rich_plus_full_spectrum_192",
)
MODEL_ORDER = ("extra_trees", "lightgbm")
VIEW_LABELS = {
    "peak_current_40": "Peak summary (40)",
    "full_spectrum_192": "Full spectrum (192)",
    "rich_optical_physics": "Rich optical physics",
    "rich_plus_full_spectrum_192": "Rich + full spectrum",
}
MODEL_LABELS = {"extra_trees": "ExtraTrees", "lightgbm": "LightGBM"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION_DATASET)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM_DATASET)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "ordinary_fbg_px6d_training.yaml",
    )
    parser.add_argument(
        "--qa-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "ordinary_fbg_px6d_collection_20260731_final"
            / "collection_qa_summary.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--minimum-leaf-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--rebuild-rich-cache", action="store_true")
    parser.add_argument("--views", nargs="+", choices=VIEW_ORDER, default=list(VIEW_ORDER))
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _leaderboard_row(
    *,
    view_id: str,
    model_id: str,
    task: str,
    feature_count: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "feature_view": view_id,
        "feature_count": feature_count,
        "model_id": model_id,
        "task": task,
        "split_strategy": "preassigned_grouped_by_session_id_5fold",
        "evaluation_validity": "formal_grouped_evaluation",
        "frame_count": metrics["frame_count"],
        "group_count": metrics["group_count"],
        "training_time_sec": metrics["training_time_sec"],
        "inference_latency_ms_per_frame": metrics[
            "inference_latency_ms_per_frame"
        ],
    }
    for key in (
        "accuracy",
        "macro_f1",
        "no_contact_recall",
        "contact_recall",
        "group_voting_accuracy",
        "group_voting_macro_f1",
        "mae_n",
        "rmse_n",
        "r2",
        "active_force_mae_n",
        "within_0_25_n",
    ):
        row[key] = metrics.get(key, np.nan)
    return row


def _append_predictions(
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
    selected = np.flatnonzero(mask)
    rows.append(
        pd.DataFrame(
            {
                "model_id": model_id,
                "feature_view": view_id,
                "task": task,
                "group_id": dataset.group_id[selected],
                "sample_index": dataset.sample_index[selected],
                "fold_id": dataset.fold_id[selected],
                "true_value": target[selected],
                "predicted_value": predicted[selected],
            }
        )
    )


def _plot_comparison(leaderboard: pd.DataFrame, output_path: Path) -> None:
    panels = (
        ("contact", "macro_f1", "Contact macro-F1", False),
        ("contact", "no_contact_recall", "No-contact recall", False),
        ("position", "macro_f1", "Position macro-F1", False),
        ("force_fz", "mae_n", "Optical Fz MAE (N)", True),
    )
    colors = {"extra_trees": "#1598bd", "lightgbm": "#d57366"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    x = np.arange(len(VIEW_ORDER))
    width = 0.34
    for axis, (task, metric, title, lower_is_better) in zip(axes.flat, panels):
        available_values: list[float] = []
        for model_index, model_id in enumerate(MODEL_ORDER):
            values: list[float] = []
            for view_id in VIEW_ORDER:
                match = leaderboard.loc[
                    (leaderboard["task"] == task)
                    & (leaderboard["model_id"] == model_id)
                    & (leaderboard["feature_view"] == view_id),
                    metric,
                ]
                value = float(match.iloc[0]) if len(match) else float("nan")
                values.append(value)
                if np.isfinite(value):
                    available_values.append(value)
            offset = (model_index - 0.5) * width
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=colors[model_id],
                label=MODEL_LABELS[model_id],
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
        axis.set_xticks(x, [VIEW_LABELS[view] for view in VIEW_ORDER], rotation=18, ha="right")
        axis.set_title(title, weight="bold")
        axis.grid(axis="y", alpha=0.2)
        if available_values and not lower_is_better:
            axis.set_ylim(max(0.0, min(available_values) - 0.06), 1.01)
        axis.legend(loc="best")
    fig.suptitle(
        "Efficient Algorithm and Optical-Feature Ablation\n"
        "Same latest-primary frames, same grouped session folds",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _best(
    leaderboard: pd.DataFrame,
    task: str,
    metric: str,
    *,
    ascending: bool = False,
) -> pd.Series:
    selected = leaderboard.loc[leaderboard["task"] == task].dropna(subset=[metric])
    return selected.sort_values(metric, ascending=ascending).iloc[0]


def _baseline_delta(
    leaderboard: pd.DataFrame,
    *,
    task: str,
    model_id: str,
    metric: str,
    candidate_view: str,
) -> float:
    baseline = leaderboard.query(
        "task == @task and model_id == @model_id and feature_view == 'peak_current_40'"
    )[metric].iloc[0]
    candidate = leaderboard.query(
        "task == @task and model_id == @model_id and feature_view == @candidate_view"
    )[metric].iloc[0]
    return float(candidate - baseline)


def _write_report(
    path: Path,
    leaderboard: pd.DataFrame,
    *,
    frame_count: int,
    group_count: int,
    rich_feature_count: int,
) -> None:
    best_contact = _best(leaderboard, "contact", "macro_f1")
    best_idle = _best(leaderboard, "contact", "no_contact_recall")
    best_position = _best(leaderboard, "position", "macro_f1")
    best_force = _best(leaderboard, "force_fz", "mae_n", ascending=True)
    rich_contact_delta = _baseline_delta(
        leaderboard,
        task="contact",
        model_id="extra_trees",
        metric="macro_f1",
        candidate_view="rich_optical_physics",
    )
    rich_idle_delta = _baseline_delta(
        leaderboard,
        task="contact",
        model_id="extra_trees",
        metric="no_contact_recall",
        candidate_view="rich_optical_physics",
    )
    lines = [
        "# Rich Optical Signal and Efficient Algorithm Benchmark",
        "",
        "## Evaluation boundary",
        "",
        f"- {frame_count:,} synchronized frames from {group_count} independent latest-primary capture sessions.",
        "- Five folds are grouped by capture session; no frame-level random split is used.",
        "- P11, P12, and P21 use only the latest five primary sessions configured by the formal dataset.",
        "- This is an offline candidate benchmark. It does not replace the Beta runtime model.",
        "",
        "## Added optical information",
        "",
        f"The rich view contains {rich_feature_count} current-frame optical features. It retains multiple wavelength estimators, estimator disagreement, peak area and height ratios, FWHM change, skewness change, shape correlation/error, local SNR, common-mode-corrected wavelength shifts, same-fibre gradients, spatial-row gradients, and per-peak residuals.",
        "",
        "These features are label-free measurements derived from the spectrum. They are intended to separate true local contact from global drift, residual deformation, peak-tracking instability, and inter-FBG coupling.",
        "",
        "## Best grouped results",
        "",
        f"- Contact macro-F1: **{best_contact['macro_f1']:.4f}**, `{best_contact['model_id']}` + `{best_contact['feature_view']}`.",
        f"- No-contact recall: **{best_idle['no_contact_recall']:.4f}**, `{best_idle['model_id']}` + `{best_idle['feature_view']}`.",
        f"- Position macro-F1: **{best_position['macro_f1']:.4f}**, `{best_position['model_id']}` + `{best_position['feature_view']}`.",
        f"- Optical Fz MAE: **{best_force['mae_n']:.4f} N**, `{best_force['model_id']}` + `{best_force['feature_view']}`.",
        "",
        "For the same ExtraTrees algorithm, replacing the 40-feature peak summary with rich optical physics changed contact macro-F1 by "
        f"**{rich_contact_delta:+.4f}** and no-contact recall by **{rich_idle_delta:+.4f}**. This isolates the value of added optical evidence from the value of changing algorithms.",
        "",
        "## Interpretation",
        "",
        "Position recognition is already close to saturation in offline grouped evaluation. The main unresolved problem is no-contact/release behavior and live domain drift. Therefore a larger neural network is not automatically the most useful next step.",
        "",
        "The recommended architecture remains hierarchical: first detect credible spectral change/contact, then infer position, then estimate Fz only while contact is credible. A stationary spectrum should be eligible for no-contact recovery even when its absolute spectrum retains a residual offset.",
        "",
        "Inference latency in this report covers only model prediction. The BaySpec acquisition cadence remains the dominant end-to-end delay, so an algorithm that adds several milliseconds is acceptable only if it materially improves grouped-session robustness.",
        "",
        "## GitHub implementations reviewed",
        "",
        "- LightGBM: https://github.com/lightgbm-org/LightGBM - efficient nonlinear tabular baseline used in this benchmark.",
        "- aeon: https://github.com/aeon-toolkit/aeon - mature time-series toolkit containing ROCKET-family classifiers.",
        "- MiniRocket: https://github.com/angus924/minirocket - fast temporal convolutional transform for future true sequence windows.",
        "- tsfresh: https://github.com/blue-yonder/tsfresh - automatic temporal feature extraction and relevance filtering.",
        "- tsai: https://github.com/timeseriesAI/tsai - InceptionTime/ResNet/FCN candidates when independent sequence counts become sufficient.",
        "",
        "MiniRocket, HYDRA, or InceptionTime should be fed a real time axis (consecutive spectra or peak tracks), not the wavelength bins of one spectrum mislabeled as time. Existing project benchmarks already show that compact tree models are stronger and more stable than current CNN candidates for position, so temporal models remain a second-stage experiment rather than an automatic replacement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_shortlist(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# GitHub Algorithm Shortlist",
                "",
                "## Ready for the current frame-level dataset",
                "",
                "1. LightGBM - nonlinear tabular learner; benchmarked directly here.",
                "2. ExtraTrees - current high-speed reference; robust with limited independent sessions.",
                "3. PLS/PCA latent optical scores plus ExtraTrees or LogisticRegression - useful for correlated spectra and domain monitoring.",
                "",
                "## Ready after true sequence-window construction",
                "",
                "1. MiniRocket / MultiRocketHydra through aeon.",
                "2. Compact TCN or InceptionTime through tsai.",
                "3. tsfresh-selected release, hysteresis, slope, plateau, and recovery features.",
                "",
                "## Not recommended now",
                "",
                "- A large CNN trained on individual frames: it does not solve baseline drift and adds deployment cost.",
                "- Random frame splitting: it leaks capture-session identity and inflates accuracy.",
                "- Treating the 512 wavelength samples as a temporal sequence: wavelength and time are different physical axes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and aligning formal latest-primary datasets...", flush=True)
    dataset = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    cache_path = output_dir / "rich_optical_feature_cache.npz"
    if cache_path.exists() and not args.rebuild_rich_cache:
        print(f"Loading rich feature cache: {cache_path}", flush=True)
        cache = load_rich_feature_cache(
            cache_path,
            expected_group_id=dataset.group_id,
            expected_sample_index=dataset.sample_index,
        )
    else:
        print("Extracting rich optical physics features...", flush=True)

        def extraction_progress(done: int, total: int, group: str) -> None:
            print(f"  rich features {done:02d}/{total:02d}: {group}", flush=True)

        cache = build_aligned_rich_feature_cache(
            group_id=dataset.group_id,
            sample_index=dataset.sample_index,
            capture_root=args.capture_root.resolve(),
            channel_config_path=args.channel_config.resolve(),
            training_config_path=args.training_config.resolve(),
            qa_summary_path=args.qa_summary.resolve(),
            progress=extraction_progress,
        )
        save_rich_feature_cache(cache, cache_path)
    views = build_rich_feature_views(dataset, cache)

    metrics_payload: dict[str, Any] = {
        "scope": {
            "frame_count": int(len(dataset.group_id)),
            "independent_session_count": int(len(set(dataset.group_id.tolist()))),
            "split_strategy": "preassigned_grouped_by_session_id_5fold",
            "random_frame_split_used": False,
            "deployment_changed": False,
        },
        "models": {},
        "skipped_models": {},
    }
    leaderboard_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    position_labels = [
        label for label in POSITION_ORDER if np.any(dataset.position_target == label)
    ]

    total_jobs = len(args.models) * len(args.views) * 3
    job = 0
    for model_id in args.models:
        available, reason = model_available(model_id)
        if not available:
            metrics_payload["skipped_models"][model_id] = reason
            print(f"Skipping {model_id}: {reason}", flush=True)
            continue
        model_payload: dict[str, Any] = {}
        for view_id in args.views:
            view = views[view_id]
            view_payload: dict[str, Any] = {}
            for task in ("contact", "position", "force_fz"):
                job += 1
                print(
                    f"[{job:02d}/{total_jobs:02d}] {model_id} | {view_id} | {task}",
                    flush=True,
                )
                if task == "contact":
                    metrics, predicted = grouped_classification(
                        model_id=model_id,
                        feature_view=view,
                        target=dataset.contact_target,
                        mask=dataset.contact_mask,
                        fold_id=dataset.fold_id,
                        group_id=dataset.group_id,
                        labels=[0, 1],
                        estimators=args.estimators,
                        minimum_leaf_samples=args.minimum_leaf_samples,
                        seed=args.seed,
                    )
                    true = dataset.contact_target[dataset.contact_mask]
                    selected_prediction = np.asarray(
                        predicted[dataset.contact_mask], dtype=int
                    )
                    metrics["no_contact_recall"] = float(
                        np.mean(selected_prediction[true == 0] == 0)
                    )
                    metrics["contact_recall"] = float(
                        np.mean(selected_prediction[true == 1] == 1)
                    )
                    mask = dataset.contact_mask
                    target = dataset.contact_target
                elif task == "position":
                    metrics, predicted = grouped_classification(
                        model_id=model_id,
                        feature_view=view,
                        target=dataset.position_target,
                        mask=dataset.position_mask,
                        fold_id=dataset.fold_id,
                        group_id=dataset.group_id,
                        labels=position_labels,
                        estimators=args.estimators,
                        minimum_leaf_samples=args.minimum_leaf_samples,
                        seed=args.seed + 100,
                    )
                    mask = dataset.position_mask
                    target = dataset.position_target
                else:
                    metrics, predicted = grouped_force_regression(
                        model_id=model_id,
                        feature_view=view,
                        target=dataset.force_fz_n,
                        mask=dataset.force_mask,
                        fold_id=dataset.fold_id,
                        group_id=dataset.group_id,
                        estimators=args.estimators,
                        minimum_leaf_samples=args.minimum_leaf_samples,
                        seed=args.seed + 200,
                    )
                    mask = dataset.force_mask
                    target = dataset.force_fz_n

                view_payload[task] = metrics
                leaderboard_rows.append(
                    _leaderboard_row(
                        view_id=view_id,
                        model_id=model_id,
                        task=task,
                        feature_count=view.values.shape[1],
                        metrics=metrics,
                    )
                )
                _append_predictions(
                    prediction_rows,
                    dataset=dataset,
                    model_id=model_id,
                    view_id=view_id,
                    task=task,
                    predicted=predicted,
                    mask=mask,
                    target=target,
                )
                for rank, item in enumerate(metrics["top_feature_importance"], start=1):
                    importance_rows.append(
                        {
                            "model_id": model_id,
                            "feature_view": view_id,
                            "task": task,
                            "rank": rank,
                            **item,
                        }
                    )
            model_payload[view_id] = view_payload
        metrics_payload["models"][model_id] = model_payload

    leaderboard = pd.DataFrame(leaderboard_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    importances = pd.DataFrame(importance_rows)
    leaderboard.to_csv(output_dir / "rich_optical_model_leaderboard.csv", index=False)
    predictions.to_csv(output_dir / "grouped_out_of_fold_predictions.csv", index=False)
    importances.to_csv(output_dir / "top_optical_feature_importance.csv", index=False)
    (output_dir / "rich_optical_model_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _plot_comparison(leaderboard, output_dir / "rich_optical_model_comparison.png")
    _write_report(
        output_dir / "rich_optical_algorithm_report.md",
        leaderboard,
        frame_count=len(dataset.group_id),
        group_count=len(set(dataset.group_id.tolist())),
        rich_feature_count=views["rich_optical_physics"].values.shape[1],
    )
    _write_shortlist(output_dir / "github_algorithm_shortlist.md")
    print(f"Completed: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
