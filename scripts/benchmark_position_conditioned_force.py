"""Benchmark leakage-safe position-conditioned optical force estimators.

This script never changes the deployed model.  It compares the current grouped
OOF force evidence with optical-only candidates that account for the strongly
position-dependent sensitivity of the nine-FBG array.  Held-out position
conditions come from grouped OOF optical position predictions, not from the
test labels or PX6D force trace.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.all_source_training import (  # noqa: E402
    load_fusion_arrays,
)
from src.hybrid_spectrum.force_consistency_audit import (  # noqa: E402
    POSITION_ORDER,
    build_force_consistency_tables,
    infer_position_id,
    representative_sessions,
)
from src.hybrid_spectrum.position_conditioned_force import (  # noqa: E402
    ConditionedForceVariant,
    causal_contact_reset_ema,
    grouped_conditioned_force_oof,
    grouped_position_vote,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_all_data_fusion_20260803_v2"
    / "all_source_fusion_dataset.npz"
)
DEFAULT_HISTORICAL_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_all_data_fusion_20260731_v1"
    / "all_source_fusion_dataset.npz"
)
DEFAULT_TRAINING = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_all_data_fusion_training_20260803_v2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_position_conditioned_force_20260803"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--historical-dataset",
        type=Path,
        default=DEFAULT_HISTORICAL_DATASET,
        help="Historical force sessions used only as explicitly downweighted training data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "ordinary_fbg_all_data_fusion_20260803.yaml",
    )
    parser.add_argument(
        "--grouped-oof",
        type=Path,
        default=DEFAULT_TRAINING / "grouped_oof_predictions.csv",
    )
    parser.add_argument(
        "--contact-gate",
        type=Path,
        default=DEFAULT_TRAINING / "force_contact_gate_oof_predictions.csv",
    )
    parser.add_argument(
        "--position-model-id",
        default="all_sources_temporal_extra_trees",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold_id",
        "array_index",
        "group_id",
        "file_id",
        "sample_index",
        "elapsed_time_sec",
        "true_force_n",
        "raw_optical_force_n",
        "contact_gate_active",
        "gated_force_n",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("contact-gate OOF file missing: " + ", ".join(missing))
    result = frame.copy()
    result["model_id"] = "baseline_current_extra_trees"
    result["raw_force_n"] = pd.to_numeric(
        result["raw_optical_force_n"], errors="raise"
    )
    result["position_id"] = [
        infer_position_id(group_id) or infer_position_id(file_id)
        for group_id, file_id in zip(result["group_id"], result["file_id"])
    ]
    result["position_condition"] = result["position_id"]
    result["position_condition_source"] = "not_applicable_baseline"
    result["force_sensor_used_as_runtime_input"] = False
    result["evaluation_validity"] = "formal_grouped_oof_by_session_id"
    return result[
        [
            "model_id",
            "fold_id",
            "array_index",
            "group_id",
            "file_id",
            "sample_index",
            "elapsed_time_sec",
            "position_id",
            "position_condition",
            "position_condition_source",
            "true_force_n",
            "raw_force_n",
            "contact_gate_active",
            "gated_force_n",
            "force_sensor_used_as_runtime_input",
            "evaluation_validity",
        ]
    ].copy()


def _ema_strategy(predictions: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    result = predictions.copy()
    result["model_id"] = result["model_id"].astype(str) + f"_ema{int(alpha * 100):02d}"
    result["gated_force_n"] = causal_contact_reset_ema(
        result["raw_force_n"].to_numpy(dtype=float),
        result["contact_gate_active"].to_numpy(dtype=bool),
        result["group_id"].astype(str).to_numpy(),
        result["sample_index"].to_numpy(dtype=int),
        alpha=alpha,
    )
    return result


def _global_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    reference = predictions["true_force_n"].to_numpy(dtype=float)
    estimate = predictions["gated_force_n"].to_numpy(dtype=float)
    error = estimate - reference
    slope, intercept = np.polyfit(reference, estimate, 1)
    reference_span = float(np.percentile(reference, 95) - np.percentile(reference, 5))
    estimate_span = float(np.percentile(estimate, 95) - np.percentile(estimate, 5))
    zero_mask = reference <= 0.03
    return {
        "sample_count": int(len(reference)),
        "session_count": int(predictions["group_id"].nunique()),
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "rmse_n": float(mean_squared_error(reference, estimate) ** 0.5),
        "r2": float(r2_score(reference, estimate)),
        "pearson_r": float(np.corrcoef(reference, estimate)[0, 1]),
        "linear_slope_pred_vs_px6d": float(slope),
        "linear_intercept_n": float(intercept),
        "amplitude_ratio_p95_p05": float(estimate_span / reference_span),
        "zero_force_false_response_rate": float(
            np.mean(estimate[zero_mask] > 0.10)
        ),
    }


def _strategy_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    position_parts: list[pd.DataFrame] = []
    session_parts: list[pd.DataFrame] = []
    for model_id, group in predictions.groupby("model_id", sort=False):
        sessions, positions = build_force_consistency_tables(group)
        sessions.insert(0, "model_id", model_id)
        positions.insert(0, "model_id", model_id)
        session_parts.append(sessions)
        position_parts.append(positions)
        global_row = _global_metrics(group)
        p13 = positions[positions["position_id"].astype(str) == "P13"].iloc[0]
        metric_rows.append(
            {
                "model_id": model_id,
                **global_row,
                "p13_mae_n": float(p13["mae_n"]),
                "p13_r2": float(p13["r2"]),
                "p13_pearson_r": float(p13["pearson_r"]),
                "p13_slope": float(p13["linear_slope_pred_vs_px6d"]),
                "p13_amplitude_ratio": float(p13["amplitude_ratio_p95_p05"]),
                "worst_position_mae_n": float(positions["mae_n"].max()),
                "worst_position_r2": float(positions["r2"].min()),
                "consistent_position_count": int(
                    (positions["audit_status"] == "consistent").sum()
                ),
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(position_parts, ignore_index=True),
        pd.concat(session_parts, ignore_index=True),
    )


def _candidate_decision(metrics: pd.DataFrame) -> dict[str, Any]:
    indexed = metrics.set_index("model_id")
    baseline = indexed.loc["baseline_current_extra_trees"]
    candidates = indexed.drop(index=["baseline_current_extra_trees"])
    candidates = candidates.assign(
        score=(candidates["mae_n"] - baseline["mae_n"])
        + 0.8 * (candidates["p13_mae_n"] - baseline["p13_mae_n"])
        - 0.25 * (candidates["r2"] - baseline["r2"])
    )
    best_id = str(candidates["score"].idxmin())
    best = indexed.loc[best_id]
    guardrails = {
        "global_mae_not_worse": bool(best["mae_n"] <= baseline["mae_n"] + 0.003),
        "global_r2_not_worse": bool(best["r2"] >= baseline["r2"] - 0.002),
        "p13_mae_improves_10_percent": bool(
            best["p13_mae_n"] <= baseline["p13_mae_n"] * 0.90
        ),
        "p13_r2_improves": bool(best["p13_r2"] > baseline["p13_r2"]),
        "zero_force_residual_not_worse": bool(
            best["zero_force_false_response_rate"]
            <= baseline["zero_force_false_response_rate"] + 0.005
        ),
        "all_nine_positions_consistent": bool(best["consistent_position_count"] == 9),
    }
    passed = all(guardrails.values())
    return {
        "baseline_model_id": "baseline_current_extra_trees",
        "best_candidate_model_id": best_id,
        "deployment_recommended": passed,
        "decision": "candidate_passes_guardrails" if passed else "retain_current_model",
        "guardrails": guardrails,
        "reason": (
            "All predeclared global, P13, release-residual, and nine-point gates passed."
            if passed
            else "At least one predeclared independent-session guardrail failed; no deployed model is replaced."
        ),
    }


def _stable_plateaus(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.arange(0.0, 5.01, 0.5)
    rows: list[dict[str, Any]] = []
    for (model_id, position_id, group_id), group in predictions.groupby(
        ["model_id", "position_id", "group_id"], sort=False
    ):
        group = group.sort_values("sample_index")
        time = group["elapsed_time_sec"].to_numpy(dtype=float)
        force = group["true_force_n"].to_numpy(dtype=float)
        estimate = group["gated_force_n"].to_numpy(dtype=float)
        dt = np.diff(time, prepend=np.nan)
        df = np.diff(force, prepend=np.nan)
        speed = np.divide(
            np.abs(df),
            dt,
            out=np.full_like(force, np.inf),
            where=np.isfinite(dt) & (dt > 0),
        )
        speed[0] = speed[1] if len(speed) > 1 else math.inf
        stable = (force >= 0.10) & (speed <= 0.20)
        bin_index = np.digitize(force, bins, right=False) - 1
        for index in range(len(bins) - 1):
            selected = stable & (bin_index == index)
            if int(np.sum(selected)) < 3:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "position_id": position_id,
                    "group_id": group_id,
                    "force_bin_low_n": bins[index],
                    "force_bin_high_n": bins[index + 1],
                    "stable_frame_count": int(np.sum(selected)),
                    "px6d_mean_n": float(np.mean(force[selected])),
                    "optical_mean_n": float(np.mean(estimate[selected])),
                }
            )
    plateaus = pd.DataFrame(rows)
    linearity_rows: list[dict[str, Any]] = []
    for (model_id, position_id), group in plateaus.groupby(
        ["model_id", "position_id"], sort=False
    ):
        x = group["px6d_mean_n"].to_numpy(dtype=float)
        y = group["optical_mean_n"].to_numpy(dtype=float)
        if len(x) < 3 or float(np.std(x)) <= 1.0e-12:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        linearity_rows.append(
            {
                "model_id": model_id,
                "position_id": position_id,
                "plateau_count": int(len(group)),
                "plateau_mae_n": float(mean_absolute_error(x, y)),
                "plateau_r2": float(r2_score(x, y)),
                "plateau_slope": float(slope),
                "plateau_intercept_n": float(intercept),
                "plateau_pearson_r": float(np.corrcoef(x, y)[0, 1]),
            }
        )
    return plateaus, pd.DataFrame(linearity_rows)


def _plot_strategy_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["model_id"].str.replace("baseline_current_extra_trees", "baseline")
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    panels = (
        ("mae_n", "Global grouped OOF MAE", "N", None),
        ("r2", "Global grouped OOF R2", "R2", None),
        ("p13_mae_n", "P13 grouped OOF MAE", "N", None),
        ("p13_slope", "P13 force-amplitude slope", "slope", 1.0),
    )
    for axis, (column, title, ylabel, reference) in zip(axes.flat, panels):
        axis.bar(labels, metrics[column], color="#72bdd2")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=28, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
        if reference is not None:
            axis.axhline(reference, color="#d06b5f", linestyle="--", linewidth=1.4)
    figure.suptitle("Position-conditioned optical force benchmark")
    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_trace_comparison(
    predictions: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    candidate_id: str,
    output_path: Path,
) -> None:
    baseline_sessions = sessions[
        sessions["model_id"] == "baseline_current_extra_trees"
    ]
    representatives = representative_sessions(baseline_sessions)
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 10.2), sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER):
        group_id = representatives.get(position_id)
        baseline = predictions[
            (predictions["model_id"] == "baseline_current_extra_trees")
            & (predictions["group_id"] == group_id)
        ].sort_values("sample_index")
        candidate = predictions[
            (predictions["model_id"] == candidate_id)
            & (predictions["group_id"] == group_id)
        ].sort_values("sample_index")
        if baseline.empty or candidate.empty:
            axis.set_axis_off()
            continue
        axis.plot(
            baseline["elapsed_time_sec"],
            baseline["true_force_n"],
            color="#142f44",
            linewidth=2.0,
            label="PX6D Fz",
        )
        axis.plot(
            baseline["elapsed_time_sec"],
            baseline["gated_force_n"],
            color="#d4775f",
            linewidth=1.5,
            alpha=0.85,
            label="current model",
        )
        axis.plot(
            candidate["elapsed_time_sec"],
            candidate["gated_force_n"],
            color="#1397ad",
            linewidth=1.7,
            label="best audit candidate",
        )
        axis.set_title(position_id)
        axis.set_ylim(-0.15, 5.25)
        axis.grid(alpha=0.16)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Nine-point independent-session force trace comparison", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_plateau_linearity(
    plateaus: pd.DataFrame,
    *,
    candidate_id: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.5, 11.0), sharex=True, sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER):
        for model_id, color, marker in (
            ("baseline_current_extra_trees", "#d4775f", "o"),
            (candidate_id, "#1397ad", "s"),
        ):
            group = plateaus[
                (plateaus["model_id"] == model_id)
                & (plateaus["position_id"] == position_id)
            ]
            if group.empty:
                continue
            axis.scatter(
                group["px6d_mean_n"],
                group["optical_mean_n"],
                s=15,
                alpha=0.58,
                color=color,
                marker=marker,
                label="current" if model_id.startswith("baseline") else "candidate",
            )
        axis.plot([0, 5], [0, 5], color="#425466", linestyle="--", linewidth=1.0)
        axis.set_title(position_id)
        axis.set_xlim(0, 5)
        axis.set_ylim(0, 5)
        axis.grid(alpha=0.14)
        axis.set_xlabel("PX6D stable plateau (N)")
        axis.set_ylabel("Optical stable plateau (N)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Stable-plateau linearity audit (analysis only, no calibration leakage)")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(value) for value in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    for values in frame.itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|") for value in values])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _write_report(
    metrics: pd.DataFrame,
    positions: pd.DataFrame,
    condition_audit: dict[str, Any],
    decision: dict[str, Any],
    output_path: Path,
) -> None:
    baseline = metrics.set_index("model_id").loc[decision["baseline_model_id"]]
    candidate = metrics.set_index("model_id").loc[decision["best_candidate_model_id"]]
    comparison = positions[
        positions["model_id"].isin(
            [decision["baseline_model_id"], decision["best_candidate_model_id"]]
        )
    ][
        [
            "model_id",
            "position_id",
            "mae_n",
            "r2",
            "pearson_r",
            "linear_slope_pred_vs_px6d",
            "amplitude_ratio_p95_p05",
            "audit_status",
        ]
    ].copy()
    for column in (
        "mae_n",
        "r2",
        "pearson_r",
        "linear_slope_pred_vs_px6d",
        "amplitude_ratio_p95_p05",
    ):
        comparison[column] = comparison[column].map(lambda value: f"{float(value):.3f}")
    failed = [name for name, passed in decision["guardrails"].items() if not passed]
    text = f"""# Position-conditioned optical force benchmark

## Scope

This is an offline, grouped out-of-fold audit. PX6D Fz is supervision and
evaluation evidence only; it is not a runtime input. No deployed model was
changed. The held-out position condition comes from the selected grouped OOF
optical position model, not from the held-out label.

## Position-condition evidence

- Position model: `{condition_audit['position_model_id']}`
- Formal force sessions with an OOF position vote: {condition_audit['voted_session_count']}/{condition_audit['formal_session_count']}
- Group-vote accuracy against session labels: {condition_audit['group_vote_accuracy']:.3%}
- Unknown condition rate: {condition_audit['unknown_condition_rate']:.3%}

## Decision

- Best audit candidate: `{decision['best_candidate_model_id']}`
- Deployment recommended: `{decision['deployment_recommended']}`
- Decision: `{decision['decision']}`
- Failed guardrails: {', '.join(failed) if failed else 'none'}

Global MAE: {baseline['mae_n']:.4f} N -> {candidate['mae_n']:.4f} N.  Global
R2: {baseline['r2']:.4f} -> {candidate['r2']:.4f}.  P13 MAE:
{baseline['p13_mae_n']:.4f} N -> {candidate['p13_mae_n']:.4f} N.  P13 R2:
{baseline['p13_r2']:.4f} -> {candidate['p13_r2']:.4f}.

## Nine-point comparison

{_markdown_table(comparison)}

## Interpretation

Position conditioning is physically justified because each FBG point has a
different optical sensitivity. It is not enough, however, to improve only one
aggregate number. The candidate must preserve global accuracy, release-to-zero
behavior, and all nine point curves while materially improving the weak P13
sessions. Stable-plateau linearity is reported as an analysis diagnostic only;
no held-out PX6D-derived scale or offset is applied to predictions.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arrays = load_fusion_arrays(args.dataset)
    historical_arrays = load_fusion_arrays(args.historical_dataset)
    grouped_oof = pd.read_csv(args.grouped_oof, low_memory=False)
    gate = pd.read_csv(args.contact_gate, low_memory=False)

    group_votes = grouped_position_vote(
        grouped_oof,
        model_id=args.position_model_id,
    )
    formal_groups = sorted(
        set(
            arrays.group_id[
                arrays.force_mask & arrays.formal_test_eligible & (arrays.fold_id >= 0)
            ].tolist()
        )
    )
    true_group_positions = {
        group_id: infer_position_id(group_id) for group_id in formal_groups
    }
    voted = [group_id for group_id in formal_groups if group_id in group_votes]
    correct = [
        group_id
        for group_id in voted
        if group_votes[group_id] == true_group_positions[group_id]
    ]
    condition_audit = {
        "position_model_id": args.position_model_id,
        "formal_session_count": len(formal_groups),
        "voted_session_count": len(voted),
        "group_vote_accuracy": len(correct) / max(1, len(voted)),
        "unknown_condition_rate": 1.0 - len(voted) / max(1, len(formal_groups)),
    }

    gate_indices = gate["array_index"].astype(int)
    if gate_indices.duplicated().any():
        raise ValueError("contact-gate OOF contains duplicate array indices")
    gate_map = dict(
        zip(gate_indices.tolist(), gate["contact_gate_active"].astype(bool).tolist())
    )
    formal_indices = set(
        np.flatnonzero(
            arrays.force_mask & arrays.formal_test_eligible & (arrays.fold_id >= 0)
        ).tolist()
    )
    missing_gate = sorted(formal_indices - set(gate_map))
    if missing_gate:
        raise ValueError(f"contact gate missing {len(missing_gate)} formal rows")

    variants = (
        ConditionedForceVariant(
            "position_current_extra_trees_leaf2",
            feature_view="current_frame",
            estimators=180,
            minimum_leaf_samples=2,
        ),
        ConditionedForceVariant(
            "position_current_extra_trees_leaf4",
            feature_view="current_frame",
            estimators=180,
            minimum_leaf_samples=4,
        ),
        ConditionedForceVariant(
            "position_temporal_extra_trees_leaf2",
            feature_view="temporal_fusion",
            estimators=120,
            minimum_leaf_samples=2,
        ),
    )
    parts = [_baseline_predictions(gate)]
    for variant in variants:
        print(f"training {variant.model_id}", flush=True)
        parts.append(
            grouped_conditioned_force_oof(
                arrays,
                variant,
                source_policy=config["source_policy"],
                group_position_votes=group_votes,
                gate_active_by_array_index=gate_map,
                random_seed=int(config["evaluation"]["random_seed"]),
            )
        )
    historical_variant = ConditionedForceVariant(
        "position_temporal_extra_trees_leaf2_historical_p13",
        feature_view="temporal_fusion",
        estimators=120,
        minimum_leaf_samples=2,
    )
    print(f"training {historical_variant.model_id}", flush=True)
    parts.append(
        grouped_conditioned_force_oof(
            arrays,
            historical_variant,
            source_policy=config["source_policy"],
            group_position_votes=group_votes,
            gate_active_by_array_index=gate_map,
            auxiliary_arrays=historical_arrays,
            auxiliary_positions=("P13",),
            auxiliary_relative_weight=0.25,
            random_seed=int(config["evaluation"]["random_seed"]),
        )
    )
    raw_predictions = pd.concat(parts, ignore_index=True)
    ema_parts = [
        _ema_strategy(group, alpha=0.65)
        for model_id, group in raw_predictions.groupby("model_id", sort=False)
    ]
    predictions = pd.concat([raw_predictions, *ema_parts], ignore_index=True)
    predictions["position_id"] = predictions["position_id"].astype(str)

    metrics, positions, sessions = _strategy_tables(predictions)
    decision = _candidate_decision(metrics)
    plateaus, plateau_linearity = _stable_plateaus(predictions)

    predictions.to_csv(
        output_dir / "position_conditioned_force_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(output_dir / "strategy_metrics.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(
        output_dir / "metrics_by_position.csv", index=False, encoding="utf-8-sig"
    )
    sessions.to_csv(
        output_dir / "metrics_by_session.csv", index=False, encoding="utf-8-sig"
    )
    plateaus.to_csv(
        output_dir / "stable_plateau_means.csv", index=False, encoding="utf-8-sig"
    )
    plateau_linearity.to_csv(
        output_dir / "stable_plateau_linearity_by_position.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_strategy_metrics(metrics, output_dir / "strategy_comparison.png")
    _plot_trace_comparison(
        predictions,
        sessions,
        candidate_id=decision["best_candidate_model_id"],
        output_path=output_dir / "nine_point_trace_comparison.png",
    )
    _plot_plateau_linearity(
        plateaus,
        candidate_id=decision["best_candidate_model_id"],
        output_path=output_dir / "stable_plateau_linearity.png",
    )
    _write_report(
        metrics,
        positions,
        condition_audit,
        decision,
        output_dir / "position_conditioned_force_report.md",
    )
    payload = {
        "schema_version": "ordinary_fbg_position_conditioned_force_benchmark_v1",
        "dataset": str(args.dataset.resolve()),
        "historical_auxiliary_dataset": str(args.historical_dataset.resolve()),
        "historical_auxiliary_policy": {
            "positions": ["P13"],
            "relative_training_weight": 0.25,
            "formal_test_eligible": False,
            "reason": "Older P13 force sessions broaden observed sensitivity; older P11/P12/P21 are excluded because their labels were previously questioned.",
        },
        "grouped_oof_position_source": str(args.grouped_oof.resolve()),
        "contact_gate_source": str(args.contact_gate.resolve()),
        "evaluation_validity": "formal_grouped_oof_by_session_id",
        "runtime_inputs": "optical_features_and_optically_predicted_position_only",
        "px6d_runtime_input": False,
        "condition_audit": condition_audit,
        "decision": decision,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
