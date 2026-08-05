"""Benchmark leakage-safe nine-point optical/PX6D force alignment.

The force sensor is used as supervision and held-out evaluation evidence only.
Each formal session is evaluated in an outer grouped fold.  Position-specific
force calibration is fitted from inner grouped out-of-fold predictions of the
outer training data, so no held-out PX6D trace calibrates itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
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

from src.hybrid_spectrum.all_source_training import load_fusion_arrays  # noqa: E402
from src.hybrid_spectrum.force_consistency_audit import (  # noqa: E402
    POSITION_ORDER,
    build_force_consistency_tables,
    infer_position_id,
    representative_sessions,
)
from src.hybrid_spectrum.position_conditioned_force import (  # noqa: E402
    grouped_position_vote,
)
from src.hybrid_spectrum.position_force_experts import (  # noqa: E402
    CalibrationSpec,
    PositionExpertVariant,
    nested_grouped_position_expert_oof,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_all_data_fusion_20260803_v2"
    / "all_source_fusion_dataset.npz"
)
DEFAULT_TRAINING = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_all_data_fusion_training_20260803_v2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_nine_point_force_alignment_20260804"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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
    parser.add_argument("--estimators", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError("contact gate predictions missing: " + ", ".join(missing))
    result = frame.copy()
    result["model_id"] = "baseline_current_extra_trees"
    result["base_raw_force_n"] = pd.to_numeric(
        result["raw_optical_force_n"], errors="raise"
    )
    result["calibrated_force_n"] = result["base_raw_force_n"]
    result["position_id"] = [
        infer_position_id(group_id) or infer_position_id(file_id)
        for group_id, file_id in zip(result["group_id"], result["file_id"])
    ]
    result["position_condition"] = result["position_id"]
    result["position_condition_source"] = "not_applicable_baseline"
    result["expert_used"] = "shared_global_model"
    result["force_sensor_used_as_runtime_input"] = False
    result["evaluation_validity"] = "formal_grouped_oof_by_session_id"
    columns = [
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
        "expert_used",
        "true_force_n",
        "base_raw_force_n",
        "calibrated_force_n",
        "contact_gate_active",
        "gated_force_n",
        "force_sensor_used_as_runtime_input",
        "evaluation_validity",
    ]
    return result[columns].copy()


def global_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = frame["true_force_n"].to_numpy(dtype=float)
    estimate = frame["gated_force_n"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(reference, estimate, 1)
    zero = reference <= 0.03
    return {
        "sample_count": int(len(frame)),
        "session_count": int(frame["group_id"].nunique()),
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "rmse_n": float(mean_squared_error(reference, estimate) ** 0.5),
        "r2": float(r2_score(reference, estimate)),
        "pearson_r": float(np.corrcoef(reference, estimate)[0, 1]),
        "linear_slope_pred_vs_px6d": float(slope),
        "linear_intercept_n": float(intercept),
        "zero_force_false_response_rate": float(
            np.mean(estimate[zero] > 0.10) if np.any(zero) else math.nan
        ),
    }


def metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    position_parts: list[pd.DataFrame] = []
    session_parts: list[pd.DataFrame] = []
    for model_id, group in predictions.groupby("model_id", sort=False):
        sessions, positions = build_force_consistency_tables(group)
        sessions.insert(0, "model_id", model_id)
        positions.insert(0, "model_id", model_id)
        session_parts.append(sessions)
        position_parts.append(positions)
        position_slopes = positions["linear_slope_pred_vs_px6d"].to_numpy(dtype=float)
        p13 = positions[positions["position_id"].astype(str) == "P13"].iloc[0]
        model_rows.append(
            {
                "model_id": model_id,
                **global_metrics(group),
                "consistent_position_count": int(
                    (positions["audit_status"] == "consistent").sum()
                ),
                "warning_position_count": int(
                    (positions["audit_status"] == "usable_with_warning").sum()
                ),
                "needs_review_position_count": int(
                    (positions["audit_status"] == "needs_review").sum()
                ),
                "worst_position_mae_n": float(positions["mae_n"].max()),
                "worst_position_r2": float(positions["r2"].min()),
                "mean_abs_position_slope_error": float(
                    np.mean(np.abs(position_slopes - 1.0))
                ),
                "p13_mae_n": float(p13["mae_n"]),
                "p13_r2": float(p13["r2"]),
                "p13_pearson_r": float(p13["pearson_r"]),
                "p13_slope": float(p13["linear_slope_pred_vs_px6d"]),
            }
        )
    return (
        pd.DataFrame(model_rows),
        pd.concat(position_parts, ignore_index=True),
        pd.concat(session_parts, ignore_index=True),
    )


def stable_plateaus(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.arange(0.0, 5.01, 0.5)
    rows: list[dict[str, Any]] = []
    for (model_id, position_id, group_id), group in predictions.groupby(
        ["model_id", "position_id", "group_id"], sort=False
    ):
        group = group.sort_values("sample_index")
        elapsed = group["elapsed_time_sec"].to_numpy(dtype=float)
        reference = group["true_force_n"].to_numpy(dtype=float)
        estimate = group["gated_force_n"].to_numpy(dtype=float)
        dt = np.diff(elapsed, prepend=np.nan)
        df = np.diff(reference, prepend=np.nan)
        speed = np.divide(
            np.abs(df),
            dt,
            out=np.full_like(reference, np.inf),
            where=np.isfinite(dt) & (dt > 0),
        )
        if len(speed) > 1:
            speed[0] = speed[1]
        stable = (reference >= 0.10) & (speed <= 0.20)
        bin_index = np.digitize(reference, bins, right=False) - 1
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
                    "px6d_mean_n": float(np.mean(reference[selected])),
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


def select_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    indexed = metrics.set_index("model_id")
    baseline = indexed.loc["baseline_current_extra_trees"]
    candidates = metrics[metrics["model_id"] != "baseline_current_extra_trees"].copy()
    candidates = candidates.sort_values(
        [
            "consistent_position_count",
            "worst_position_mae_n",
            "mae_n",
            "mean_abs_position_slope_error",
        ],
        ascending=[False, True, True, True],
    )
    best_id = str(candidates.iloc[0]["model_id"])
    best = indexed.loc[best_id]
    guardrails = {
        "all_nine_positions_consistent": bool(best["consistent_position_count"] == 9),
        "global_mae_not_materially_worse": bool(
            best["mae_n"] <= baseline["mae_n"] + 0.02
        ),
        "global_r2_not_materially_worse": bool(best["r2"] >= baseline["r2"] - 0.01),
        "zero_force_residual_not_worse": bool(
            best["zero_force_false_response_rate"]
            <= baseline["zero_force_false_response_rate"] + 0.005
        ),
        "p13_mae_improves_10_percent": bool(
            best["p13_mae_n"] <= baseline["p13_mae_n"] * 0.90
        ),
        "p13_slope_in_consistent_range": bool(0.75 <= best["p13_slope"] <= 1.25),
    }
    passed = all(guardrails.values())
    return {
        "baseline_model_id": "baseline_current_extra_trees",
        "best_candidate_model_id": best_id,
        "deployment_recommended": bool(passed),
        "decision": "candidate_passes_guardrails" if passed else "retain_current_model",
        "guardrails": guardrails,
    }


def plot_trace_comparison(
    predictions: pd.DataFrame,
    sessions: pd.DataFrame,
    candidate_id: str,
    output_path: Path,
) -> None:
    baseline_id = "baseline_current_extra_trees"
    representatives = representative_sessions(sessions[sessions["model_id"] == baseline_id])
    figure, axes = plt.subplots(3, 3, figsize=(15.6, 10.2), sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER):
        group_id = representatives.get(position_id)
        baseline = predictions[
            (predictions["model_id"] == baseline_id)
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
            baseline["elapsed_time_sec"], baseline["true_force_n"],
            color="#243746", linewidth=2.0, label="PX6D Fz",
        )
        axis.plot(
            baseline["elapsed_time_sec"], baseline["gated_force_n"],
            color="#d97762", linewidth=1.45, alpha=0.82, label="current optical",
        )
        axis.plot(
            candidate["elapsed_time_sec"], candidate["gated_force_n"],
            color="#0797ad", linewidth=1.75, label="candidate optical",
        )
        axis.set_title(position_id)
        axis.set_ylim(-0.15, 5.25)
        axis.grid(alpha=0.16)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Nine-point independent-session optical/PX6D force traces", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_regression(predictions: pd.DataFrame, candidate_id: str, output_path: Path) -> None:
    selected = predictions[predictions["model_id"] == candidate_id]
    figure, axes = plt.subplots(3, 3, figsize=(12.4, 11.0), sharex=True, sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER):
        group = selected[selected["position_id"] == position_id]
        x = group["true_force_n"].to_numpy(dtype=float)
        y = group["gated_force_n"].to_numpy(dtype=float)
        axis.hexbin(x, y, gridsize=34, extent=(0, 5, 0, 5), mincnt=1, cmap="GnBu")
        axis.plot([0, 5], [0, 5], color="#d97762", linestyle="--", linewidth=1.2)
        slope, intercept = np.polyfit(x, y, 1)
        axis.plot(
            [0, 5], [intercept, intercept + 5 * slope],
            color="#124a5a", linewidth=1.3,
        )
        axis.set_title(f"{position_id} | slope {slope:.2f}")
        axis.set_xlim(0, 5)
        axis.set_ylim(0, 5)
        axis.grid(alpha=0.12)
        axis.set_xlabel("PX6D Fz (N)")
        axis.set_ylabel("Optical estimate (N)")
    figure.suptitle("Grouped OOF nine-point force regression")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_strategy_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    display = metrics.copy()
    display["short_id"] = display["model_id"].str.replace(
        "position_expert_current_extra_trees_", "expert_", regex=False
    ).str.replace("baseline_current_extra_trees", "baseline", regex=False)
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 9.0))
    panels = (
        ("mae_n", "Global MAE", "N", None),
        ("r2", "Global R2", "R2", None),
        ("worst_position_mae_n", "Worst-position MAE", "N", 0.60),
        ("consistent_position_count", "Consistent positions", "count", 9.0),
    )
    for axis, (column, title, ylabel, reference) in zip(axes.flat, panels):
        axis.bar(display["short_id"], display[column], color="#67b9c9")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=28, labelsize=8)
        axis.grid(axis="y", alpha=0.18)
        if reference is not None:
            axis.axhline(reference, color="#d97762", linestyle="--", linewidth=1.2)
    figure.suptitle("Leakage-safe nine-point force alignment benchmark")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [headers, ["---"] * len(headers)]
    for values in frame.itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|") for value in values])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(
    metrics: pd.DataFrame,
    positions: pd.DataFrame,
    decision: dict[str, Any],
    condition_audit: dict[str, Any],
    output_path: Path,
) -> None:
    indexed = metrics.set_index("model_id")
    baseline = indexed.loc[decision["baseline_model_id"]]
    candidate = indexed.loc[decision["best_candidate_model_id"]]
    comparison = positions[
        positions["model_id"].isin(
            [decision["baseline_model_id"], decision["best_candidate_model_id"]]
        )
    ][
        [
            "model_id", "position_id", "mae_n", "r2", "pearson_r",
            "linear_slope_pred_vs_px6d", "lag_ms", "audit_status",
        ]
    ].copy()
    for column in ("mae_n", "r2", "pearson_r", "linear_slope_pred_vs_px6d", "lag_ms"):
        comparison[column] = comparison[column].map(
            lambda value: "--" if pd.isna(value) else f"{float(value):.3f}"
        )
    failed = [key for key, passed in decision["guardrails"].items() if not passed]
    output_path.write_text(
        f"""# Nine-point optical/PX6D force alignment benchmark

## Scope

This is an optical-only runtime benchmark with PX6D Fz used only as 0-5 N
supervision and held-out evaluation evidence. Formal evaluation is grouped by
independent session. Position-specific calibration is fitted only from inner
OOF predictions inside each outer training fold. A test session never fits its
own scale, offset, curve, or lag.

## Position evidence

- Optical position model: `{condition_audit['position_model_id']}`
- Formal sessions with grouped optical vote: {condition_audit['voted_session_count']}/{condition_audit['formal_session_count']}
- Grouped optical position accuracy: {condition_audit['group_vote_accuracy']:.3%}
- Unknown position rate: {condition_audit['unknown_condition_rate']:.3%}

## Result

- Best audit candidate: `{decision['best_candidate_model_id']}`
- Deployment recommended: `{decision['deployment_recommended']}`
- Failed guardrails: {', '.join(failed) if failed else 'none'}
- Global MAE: {baseline['mae_n']:.4f} N -> {candidate['mae_n']:.4f} N
- Global R2: {baseline['r2']:.4f} -> {candidate['r2']:.4f}
- Worst-position MAE: {baseline['worst_position_mae_n']:.4f} N -> {candidate['worst_position_mae_n']:.4f} N
- P13 MAE: {baseline['p13_mae_n']:.4f} N -> {candidate['p13_mae_n']:.4f} N
- P13 slope: {baseline['p13_slope']:.4f} -> {candidate['p13_slope']:.4f}

## Nine-point comparison

{markdown_table(comparison)}

## Interpretation

MAE measures average force magnitude error. It cannot by itself prove that the
curve follows PX6D, so Pearson correlation, fitted slope, release residual,
lag, and stable-plateau linearity are reported together. A candidate is not
deployable unless all nine positions pass the predeclared consistency checks
without materially degrading global or zero-force behavior.
""",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arrays = load_fusion_arrays(args.dataset)
    grouped_oof = pd.read_csv(args.grouped_oof, low_memory=False)
    gate = pd.read_csv(args.contact_gate, low_memory=False)

    group_votes = grouped_position_vote(grouped_oof, model_id=args.position_model_id)
    formal_mask = arrays.force_mask & arrays.formal_test_eligible & (arrays.fold_id >= 0)
    formal_groups = sorted(set(arrays.group_id[formal_mask].tolist()))
    true_positions = {group: infer_position_id(group) for group in formal_groups}
    voted = [group for group in formal_groups if group in group_votes]
    correct = [group for group in voted if group_votes[group] == true_positions[group]]
    condition_audit = {
        "position_model_id": args.position_model_id,
        "formal_session_count": len(formal_groups),
        "voted_session_count": len(voted),
        "group_vote_accuracy": len(correct) / max(1, len(voted)),
        "unknown_condition_rate": 1.0 - len(voted) / max(1, len(formal_groups)),
    }

    gate_indices = gate["array_index"].astype(int)
    if gate_indices.duplicated().any():
        raise ValueError("contact gate has duplicate array indices")
    gate_map = dict(
        zip(gate_indices.tolist(), gate["contact_gate_active"].astype(bool).tolist())
    )
    missing_gate = sorted(set(np.flatnonzero(formal_mask).tolist()) - set(gate_map))
    if missing_gate:
        raise ValueError(f"contact gate missing {len(missing_gate)} formal samples")

    variant = PositionExpertVariant(
        model_id="position_expert_current_extra_trees",
        feature_view="current_frame",
        estimators=int(args.estimators),
        minimum_leaf_samples=2,
        max_features=0.8,
    )
    calibration_specs = (
        CalibrationSpec("zero_anchor_b50", mode="zero_anchor", ridge_strength=0.02, blend=0.50),
        CalibrationSpec("affine_b25", mode="affine", ridge_strength=0.05, blend=0.25),
        CalibrationSpec("affine_b50", mode="affine", ridge_strength=0.05, blend=0.50),
        CalibrationSpec("affine_b75", mode="affine", ridge_strength=0.05, blend=0.75),
        CalibrationSpec("affine_b100", mode="affine", ridge_strength=0.05, blend=1.00),
    )
    print("training nested grouped position experts", flush=True)
    expert_predictions, parameters, split_audit = nested_grouped_position_expert_oof(
        arrays,
        variant,
        calibration_specs,
        source_policy=config["source_policy"],
        group_position_votes=group_votes,
        gate_active_by_array_index=gate_map,
        random_seed=int(config["evaluation"]["random_seed"]),
    )
    predictions = pd.concat(
        [baseline_predictions(gate), expert_predictions], ignore_index=True
    )
    predictions["position_id"] = predictions["position_id"].astype(str)
    metrics, positions, sessions = metric_tables(predictions)
    plateaus, linearity = stable_plateaus(predictions)
    decision = select_candidate(metrics)
    candidate_id = decision["best_candidate_model_id"]

    predictions.to_csv(
        output_dir / "nine_point_force_alignment_oof_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    metrics.to_csv(output_dir / "strategy_metrics.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(output_dir / "metrics_by_position.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(output_dir / "metrics_by_session.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(
        output_dir / "fold_local_calibration_parameters.csv",
        index=False, encoding="utf-8-sig",
    )
    split_audit.to_csv(output_dir / "nested_split_audit.csv", index=False, encoding="utf-8-sig")
    plateaus.to_csv(output_dir / "stable_plateau_means.csv", index=False, encoding="utf-8-sig")
    linearity.to_csv(
        output_dir / "paper_linear_fit_by_position.csv", index=False, encoding="utf-8-sig"
    )
    plot_strategy_metrics(metrics, output_dir / "strategy_comparison.png")
    plot_trace_comparison(
        predictions, sessions, candidate_id, output_dir / "nine_point_trace_comparison.png"
    )
    plot_regression(
        predictions, candidate_id, output_dir / "nine_point_force_regression.png"
    )
    write_report(
        metrics, positions, decision, condition_audit,
        output_dir / "nine_point_force_alignment_report.md",
    )
    payload = {
        "schema_version": "ordinary_fbg_nine_point_force_alignment_v1",
        "dataset": str(args.dataset.resolve()),
        "evaluation_validity": "formal_nested_grouped_oof_by_session_id",
        "runtime_inputs": "optical_features_and_optical_position_vote_only",
        "px6d_runtime_input": False,
        "force_unit": "N",
        "force_range_n": [0.0, 5.0],
        "condition_audit": condition_audit,
        "decision": decision,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(decision), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
