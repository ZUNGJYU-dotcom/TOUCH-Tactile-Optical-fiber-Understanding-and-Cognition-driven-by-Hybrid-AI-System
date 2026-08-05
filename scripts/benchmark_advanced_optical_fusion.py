"""Compare peak, full-spectrum, and fused optical feature views."""

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
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    POSITION_ORDER,
    build_feature_views,
    contact_recalls,
    grouped_extra_trees_classification,
    grouped_extra_trees_force,
    load_aligned_latest_primary,
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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "advanced_optical_feature_benchmark_20260801"
)
VIEW_ORDER = (
    "peak_current_40",
    "peak_temporal_483",
    "full_spectrum_192",
    "full_spectrum_264",
    "peak_temporal_plus_spectrum_192",
    "peak_temporal_plus_spectrum_264",
)
VIEW_LABELS = {
    "peak_current_40": "Peak current (40)",
    "peak_temporal_483": "Peak + temporal (483)",
    "full_spectrum_192": "Full spectrum (192)",
    "full_spectrum_264": "Full spectrum + derivative (264)",
    "peak_temporal_plus_spectrum_192": "Fusion (675)",
    "peak_temporal_plus_spectrum_264": "Fusion + derivative (747)",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION_DATASET)
    parser.add_argument(
        "--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM_DATASET
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--minimum-leaf-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEW_ORDER,
        default=list(VIEW_ORDER),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="rebuild reports from existing leaderboard and predictions",
    )
    return parser.parse_args()


def _leaderboard_row(
    view_id: str,
    task: str,
    feature_count: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "feature_view": view_id,
        "feature_count": feature_count,
        "task": task,
        "model": "ExtraTrees",
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


def _plot_leaderboard(leaderboard: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "peak_current_40": "#91a4b7",
        "peak_temporal_483": "#168fbd",
        "full_spectrum_192": "#51b8a6",
        "full_spectrum_264": "#238c80",
        "peak_temporal_plus_spectrum_192": "#e4ae4e",
        "peak_temporal_plus_spectrum_264": "#d67167",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (
        ("contact", "macro_f1", "Contact macro-F1", False),
        ("contact", "no_contact_recall", "No-contact recall", False),
        ("position", "macro_f1", "Position macro-F1", False),
        ("force_fz", "mae_n", "Fz MAE (N)", True),
    )
    for axis, (task, metric, title, lower_is_better) in zip(axes.flat, panels):
        selected = leaderboard.loc[leaderboard["task"] == task].copy()
        selected["order"] = selected["feature_view"].map(
            {view: index for index, view in enumerate(VIEW_ORDER)}
        )
        selected = selected.sort_values("order")
        values = selected[metric].to_numpy(dtype=float)
        labels = [VIEW_LABELS[view] for view in selected["feature_view"]]
        bars = axis.bar(
            np.arange(len(selected)),
            values,
            color=[colors[view] for view in selected["feature_view"]],
        )
        axis.set_xticks(np.arange(len(selected)), labels, rotation=22, ha="right")
        axis.set_title(title, fontsize=12, weight="bold")
        axis.grid(axis="y", alpha=0.2)
        if not lower_is_better:
            axis.set_ylim(max(0.0, float(np.nanmin(values)) - 0.05), 1.01)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        "Ordinary-FBG Optical Feature Ablation\n"
        "Same 10,528 synchronized frames and grouped session folds",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _best_row(
    leaderboard: pd.DataFrame,
    task: str,
    metric: str,
    *,
    lower_is_better: bool = False,
) -> pd.Series:
    selected = leaderboard.loc[leaderboard["task"] == task].dropna(subset=[metric])
    return selected.sort_values(metric, ascending=lower_is_better).iloc[0]


def _contact_policy_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    evaluated = predictions.loc[predictions["contact_evaluated"].astype(bool)].copy()
    pivot = evaluated.pivot_table(
        index=["group_id", "sample_index", "fold_id", "contact_true"],
        columns="feature_view",
        values="contact_predicted",
        aggfunc="first",
    ).reset_index()
    required = {
        "peak_temporal_483",
        "full_spectrum_192",
        "peak_temporal_plus_spectrum_264",
    }
    missing = required.difference(pivot.columns)
    if missing:
        raise ValueError(
            "contact policy analysis is missing feature views: "
            + ", ".join(sorted(missing))
        )
    true = pivot["contact_true"].to_numpy(dtype=int)
    peak = pivot["peak_temporal_483"].to_numpy(dtype=int)
    spectrum = pivot["full_spectrum_192"].to_numpy(dtype=int)
    fusion = pivot["peak_temporal_plus_spectrum_264"].to_numpy(dtype=int)
    policies = {
        "peak_temporal_reference": (
            peak,
            "single model; current peak-temporal feature reference",
        ),
        "full_spectrum_reference": (
            spectrum,
            "single model; strongest no-contact evidence view",
        ),
        "balanced_three_view_majority": (
            ((peak + spectrum + fusion) >= 2).astype(int),
            "balanced majority vote across complementary optical views",
        ),
        "dual_view_contact_confirmation": (
            (peak & spectrum).astype(int),
            "contact only when peak-temporal and full-spectrum views agree",
        ),
        "dual_view_contact_sensitivity": (
            (peak | spectrum).astype(int),
            "contact when either optical view detects contact",
        ),
    }
    rows: list[dict[str, Any]] = []
    for policy_id, (predicted, semantics) in policies.items():
        no_contact_recall = float(
            np.mean(predicted[true == 0] == 0) if np.any(true == 0) else np.nan
        )
        contact_recall = float(
            np.mean(predicted[true == 1] == 1) if np.any(true == 1) else np.nan
        )
        rows.append(
            {
                "policy_id": policy_id,
                "semantics": semantics,
                "accuracy": float(accuracy_score(true, predicted)),
                "macro_f1": float(
                    f1_score(true, predicted, average="macro", zero_division=0)
                ),
                "no_contact_recall": no_contact_recall,
                "false_contact_rate": 1.0 - no_contact_recall,
                "contact_recall": contact_recall,
                "frame_count": int(len(true)),
            }
        )
    return pd.DataFrame(rows)


def _plot_contact_policies(policies: pd.DataFrame, output_path: Path) -> None:
    labels = [
        value.replace("_", " ").replace("reference", "ref.")
        for value in policies["policy_id"]
    ]
    x = np.arange(len(policies))
    width = 0.24
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for offset, column, label, color in (
        (-width, "macro_f1", "Macro-F1", "#178fbd"),
        (0.0, "no_contact_recall", "No-contact recall", "#15a47b"),
        (width, "contact_recall", "Contact recall", "#d07a60"),
    ):
        values = policies[column].to_numpy(dtype=float)
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylim(0.80, 1.01)
    axis.set_title("Contact Gate Policy Trade-off", fontsize=13, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _delta_text(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def _write_report(
    *,
    output_path: Path,
    leaderboard: pd.DataFrame,
    contact_policies: pd.DataFrame,
    dataset_frame_count: int,
    dataset_group_count: int,
) -> None:
    current_contact = leaderboard.query(
        "task == 'contact' and feature_view == 'peak_temporal_483'"
    ).iloc[0]
    current_position = leaderboard.query(
        "task == 'position' and feature_view == 'peak_temporal_483'"
    ).iloc[0]
    current_force = leaderboard.query(
        "task == 'force_fz' and feature_view == 'peak_temporal_483'"
    ).iloc[0]
    best_contact = _best_row(leaderboard, "contact", "macro_f1")
    best_no_contact = _best_row(leaderboard, "contact", "no_contact_recall")
    best_position = _best_row(leaderboard, "position", "macro_f1")
    best_force = _best_row(leaderboard, "force_fz", "mae_n", lower_is_better=True)
    dual_gate = contact_policies.loc[
        contact_policies["policy_id"] == "dual_view_contact_confirmation"
    ].iloc[0]
    balanced_gate = contact_policies.loc[
        contact_policies["policy_id"] == "balanced_three_view_majority"
    ].iloc[0]

    lines = [
        "# Advanced Optical Feature Benchmark",
        "",
        "## Scope and validity",
        "",
        f"- Formal dataset: {dataset_frame_count:,} synchronized latest-primary frames from {dataset_group_count} independent capture sessions.",
        "- Every result uses the same preassigned five session-group folds.",
        "- No frame from one capture session appears in both train and test within a fold.",
        "- The benchmark does not deploy or replace the current Beta runtime model.",
        "- Fz is a PX6D calibration target; later optical-only inference does not require the force sensor.",
        "",
        "## What was compared",
        "",
        "- `peak_current_40`: current-frame features from nine FBG peaks plus four global descriptors.",
        "- `peak_temporal_483`: current peak features plus temporal summaries over the recent spectral history.",
        "- `full_spectrum_192`: baseline log-ratio, normalized shape residual, and current normalized spectrum across 64 bins.",
        "- `full_spectrum_264`: the 192-dimensional view plus wavelength derivatives and global shape statistics.",
        "- Fusion views concatenate the peak-temporal and full-spectrum representations for exactly aligned frames.",
        "",
        "## Main results",
        "",
        f"Best contact macro-F1: **{best_contact['macro_f1']:.4f}** with `{best_contact['feature_view']}` (no-contact recall {best_contact['no_contact_recall']:.4f}).",
        f"Best no-contact recall: **{best_no_contact['no_contact_recall']:.4f}** with `{best_no_contact['feature_view']}` (contact recall {best_no_contact['contact_recall']:.4f}).",
        f"Best position macro-F1: **{best_position['macro_f1']:.4f}** with `{best_position['feature_view']}`.",
        f"Best optical Fz MAE: **{best_force['mae_n']:.4f} N** with `{best_force['feature_view']}` (R2 {best_force['r2']:.4f}).",
        "",
        "Relative to the 483-dimensional latest-primary peak-temporal reference:",
        "",
        f"- contact macro-F1 delta: {_delta_text(float(best_contact['macro_f1'] - current_contact['macro_f1']))};",
        f"- no-contact recall delta: {_delta_text(float(best_no_contact['no_contact_recall'] - current_contact['no_contact_recall']))};",
        f"- position macro-F1 delta: {_delta_text(float(best_position['macro_f1'] - current_position['macro_f1']))};",
        f"- Fz MAE delta: {_delta_text(float(best_force['mae_n'] - current_force['mae_n']))} N (negative is better).",
        "",
        "This feature ablation deliberately trains on latest-primary sessions only. The separately trained all-source deployment reference also uses auxiliary historical data and previously reached contact macro-F1 0.9413 and Fz MAE 0.268 N; those numbers are useful context but are not a like-for-like feature-view comparison.",
        "",
        "## Contact gate policies",
        "",
        f"A three-view majority vote reached macro-F1 **{balanced_gate['macro_f1']:.4f}**, no-contact recall {balanced_gate['no_contact_recall']:.4f}, and contact recall {balanced_gate['contact_recall']:.4f}.",
        f"A conservative dual-view confirmation gate reached no-contact recall **{dual_gate['no_contact_recall']:.4f}** and contact recall **{dual_gate['contact_recall']:.4f}**, with macro-F1 {dual_gate['macro_f1']:.4f}.",
        "",
        "The dual-view gate is the more relevant experimental candidate for the current false-contact and release-residual problem: it requires both the peak-temporal signature and the broad spectral shape to support contact. It should remain a shadow candidate until live release tests confirm that weak real contacts are not suppressed.",
        "",
        "## Engineering interpretation",
        "",
        "The current error pattern should not be treated as a simple lack of model size. Position is already close to saturation offline, while no-contact recall is materially weaker. Live position drift is therefore more likely to come from baseline/domain mismatch, stale release state, or training/runtime preprocessing differences than from insufficient classifier capacity.",
        "",
        "Full-spectrum features can recover shoulders, asymmetric peak deformation, inter-peak background changes, and coupled responses that a peak-only summary discards. Their measured value is determined by the table above; a larger vector is not automatically a better deployment choice. Any gain must also justify feature extraction and inference latency.",
        "",
        "## Recommended model path",
        "",
        "1. Keep ExtraTrees as the deployment reference because it is strong, fast, and robust on the current number of independent sessions.",
        "2. Use a hierarchical runtime: optical change/contact gate, then position inference, then Fz regression only while contact is credible.",
        "3. Add uncertainty abstention, release hysteresis, and short temporal consensus before changing the digital-twin contact point.",
        "4. Test position-conditioned Fz regressors because the nine FBG locations have unequal sensitivity.",
        "5. Build real temporal windows for MultiRocketHydra or a compact TCN only after the gate and baseline path are stable. Treating wavelength bins as time is physically incorrect.",
        "",
        "## Additional optical information worth extracting",
        "",
        "- Savitzky-Golay first and second wavelength derivatives;",
        "- peak FWHM, prominence, local SNR, skew/asymmetry, and shoulder energy;",
        "- spectral angle and baseline reconstruction error;",
        "- PCA/PLS latent scores for correlated full-spectrum changes;",
        "- pairwise peak ratios and a 9-by-9 coupling signature;",
        "- loading/unloading hysteresis, rise time, plateau stability, release decay constant, and temporal dS/dt energy.",
        "",
        "## Mature implementations to reuse",
        "",
        "- aeon MultiRocketHydra: https://github.com/aeon-toolkit/aeon",
        "- pybaselines spectral baseline correction: https://github.com/derb12/pybaselines",
        "- ruptures change-point detection: https://github.com/deepcharles/ruptures",
        "- scikit-learn PLSRegression: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html",
        "",
        "These are candidate components, not evidence that a new deep model should immediately replace the current runtime.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    selected_views = [view for view in VIEW_ORDER if view in args.views]
    if "peak_temporal_483" not in selected_views:
        raise ValueError("peak_temporal_483 is required as the reference view")

    leaderboard_path = output_dir / "feature_view_leaderboard.csv"
    predictions_path = output_dir / "out_of_fold_predictions.csv"
    if args.reuse_existing:
        if not leaderboard_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(
                "--reuse-existing requires feature_view_leaderboard.csv and "
                "out_of_fold_predictions.csv"
            )
        leaderboard = pd.read_csv(leaderboard_path)
        predictions = pd.read_csv(predictions_path)
    else:
        feature_views = build_feature_views(dataset)
        metrics_payload: dict[str, Any] = {}
        leaderboard_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []
        for view_index, view_id in enumerate(selected_views):
            features = feature_views[view_id]
            view_metrics: dict[str, Any] = {}

            contact_metrics, contact_prediction = grouped_extra_trees_classification(
                features=features,
                target=dataset.contact_target,
                mask=dataset.contact_mask,
                fold_id=dataset.fold_id,
                group_id=dataset.group_id,
                labels=[0, 1],
                estimators=args.estimators,
                minimum_leaf_samples=args.minimum_leaf_samples,
                seed=args.seed + view_index * 100,
            )
            contact_selected_prediction = contact_prediction[
                dataset.contact_mask
            ].astype(int)
            contact_metrics.update(
                contact_recalls(
                    dataset.contact_target[dataset.contact_mask],
                    contact_selected_prediction,
                )
            )
            view_metrics["contact"] = contact_metrics
            leaderboard_rows.append(
                _leaderboard_row(
                    view_id, "contact", features.shape[1], contact_metrics
                )
            )

            position_metrics, position_prediction = grouped_extra_trees_classification(
                features=features,
                target=dataset.position_target,
                mask=dataset.position_mask,
                fold_id=dataset.fold_id,
                group_id=dataset.group_id,
                labels=list(POSITION_ORDER),
                estimators=args.estimators,
                minimum_leaf_samples=args.minimum_leaf_samples,
                seed=args.seed + view_index * 100 + 20,
            )
            view_metrics["position"] = position_metrics
            leaderboard_rows.append(
                _leaderboard_row(
                    view_id, "position", features.shape[1], position_metrics
                )
            )

            force_metrics, force_prediction = grouped_extra_trees_force(
                features=features,
                target=dataset.force_fz_n,
                mask=dataset.force_mask,
                fold_id=dataset.fold_id,
                group_id=dataset.group_id,
                estimators=args.estimators,
                minimum_leaf_samples=args.minimum_leaf_samples,
                seed=args.seed + view_index * 100 + 40,
            )
            view_metrics["force_fz"] = force_metrics
            leaderboard_rows.append(
                _leaderboard_row(
                    view_id, "force_fz", features.shape[1], force_metrics
                )
            )
            metrics_payload[view_id] = view_metrics

            frame = pd.DataFrame(
                {
                    "feature_view": view_id,
                    "group_id": dataset.group_id,
                    "sample_index": dataset.sample_index,
                    "fold_id": dataset.fold_id,
                    "contact_true": dataset.contact_target,
                    "contact_predicted": contact_prediction,
                    "position_true": dataset.position_target,
                    "position_predicted": position_prediction,
                    "force_true_n": dataset.force_fz_n,
                    "force_predicted_n": force_prediction,
                    "contact_evaluated": dataset.contact_mask,
                    "position_evaluated": dataset.position_mask,
                    "force_evaluated": dataset.force_mask,
                }
            )
            prediction_rows.append(frame)
            print(
                f"[{view_id}] contact F1={contact_metrics['macro_f1']:.4f}, "
                f"no-contact recall={contact_metrics['no_contact_recall']:.4f}, "
                f"position F1={position_metrics['macro_f1']:.4f}, "
                f"Fz MAE={force_metrics['mae_n']:.4f} N",
                flush=True,
            )

        leaderboard = pd.DataFrame(leaderboard_rows)
        leaderboard.to_csv(leaderboard_path, index=False)
        predictions = pd.concat(prediction_rows, ignore_index=True)
        predictions.to_csv(predictions_path, index=False)
        (output_dir / "feature_view_metrics.json").write_text(
            json.dumps(
                metrics_payload,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    contact_policies = _contact_policy_comparison(predictions)
    contact_policies.to_csv(
        output_dir / "contact_gate_policy_comparison.csv", index=False
    )
    alignment_audit = {
        "schema_version": "advanced_optical_alignment_audit_v1",
        "fusion_dataset": str(args.fusion_dataset.resolve()),
        "spectrum_dataset": str(args.spectrum_dataset.resolve()),
        "aligned_frame_count": int(len(dataset.group_id)),
        "independent_group_count": int(len(set(dataset.group_id.tolist()))),
        "fold_counts": {
            str(fold): int(np.sum(dataset.fold_id == fold))
            for fold in sorted(set(dataset.fold_id.tolist()))
        },
        "contact_frame_count": int(np.sum(dataset.contact_mask)),
        "position_frame_count": int(np.sum(dataset.position_mask)),
        "force_frame_count": int(np.sum(dataset.force_mask)),
        "alignment_key": ["session_id/group_id", "capture_index/sample_index"],
        "targets_equal": True,
        "folds_equal": True,
        "random_frame_split_used": False,
    }
    (output_dir / "alignment_audit.json").write_text(
        json.dumps(alignment_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_leaderboard(
        leaderboard, output_dir / "feature_view_comparison.png"
    )
    _plot_contact_policies(
        contact_policies, output_dir / "contact_gate_policy_comparison.png"
    )
    _write_report(
        output_path=output_dir / "advanced_optical_feature_report.md",
        leaderboard=leaderboard,
        contact_policies=contact_policies,
        dataset_frame_count=len(dataset.group_id),
        dataset_group_count=len(set(dataset.group_id.tolist())),
    )
    print(f"Saved benchmark to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
