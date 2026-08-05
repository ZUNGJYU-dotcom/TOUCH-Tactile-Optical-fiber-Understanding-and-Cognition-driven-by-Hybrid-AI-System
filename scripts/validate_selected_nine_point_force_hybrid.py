"""Validate a fixed nine-point optical-force hybrid with grouped OOF evidence.

The script is deliberately evaluation-only.  It combines the retained force
baseline with fixed, position-specific full-spectrum PLS experts selected in
earlier discovery runs.  PX6D Fz is supervision/evaluation evidence and never
enters runtime features.
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
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.force_consistency_audit import (  # noqa: E402
    POSITION_ORDER,
    build_force_consistency_tables,
    build_stable_plateau_tables,
    force_curve_metrics,
    plot_position_regression,
    plot_representative_traces,
)
from src.hybrid_spectrum.nine_point_force_hybrid import (  # noqa: E402
    aligned_contact_gate_map,
    apply_grouped_contact_gate,
    build_session_acceptance_table,
    grouped_gate_truth_masks,
)
from src.hybrid_spectrum.position_conditioned_force import (  # noqa: E402
    grouped_position_vote,
)
from src.hybrid_spectrum.spectral_force_experts import (  # noqa: E402
    SpectralForceExpertSpec,
    apply_spectral_expert_override,
    build_causal_spectral_views,
    formal_array_indices,
    grouped_spectral_force_expert_oof,
)


DEFAULT_FUSION = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_all_data_fusion_20260804_initial_fixed5"
    / "all_source_fusion_dataset.npz"
)
DEFAULT_SPECTRUM = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_px6d_strict_20260803_new_data_only"
    / "ordinary_fbg_px6d_dataset.npz"
)
DEFAULT_TRAINING = (
    PROJECT_ROOT / "outputs/ordinary_fbg_all_data_fusion_training_20260804_initial_fixed5"
)
DEFAULT_ALIGNMENT = (
    PROJECT_ROOT / "outputs/ordinary_fbg_nine_point_force_alignment_20260804_initial_fixed5"
)
DEFAULT_GATE = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_p13_nonlinear_force_gate_20260804"
    / "best_contact_gate_grouped_oof.csv"
)
DEFAULT_OBSERVABILITY = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_force_observability_20260804"
    / "force_optical_observability_by_session.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/ordinary_fbg_selected_nine_point_force_hybrid_20260804"
)


# These are fixed before this validation run.  P12/P22/P23 retain the baseline.
SELECTED_EXPERTS = (
    ("P11", "current264", 8),
    ("P21", "current264", 12),
    ("P31", "current264", 12),
    ("P32", "current264", 32),
    ("P33", "current264", 32),
    ("P13", "current_plus_lag_delta1_3_8", 24),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument(
        "--grouped-oof",
        type=Path,
        default=DEFAULT_TRAINING / "grouped_oof_predictions.csv",
    )
    parser.add_argument("--contact-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument(
        "--alignment-predictions",
        type=Path,
        default=DEFAULT_ALIGNMENT / "nine_point_force_alignment_oof_predictions.csv",
    )
    parser.add_argument("--observability-audit", type=Path, default=DEFAULT_OBSERVABILITY)
    parser.add_argument(
        "--baseline-model-id",
        default="position_expert_current_extra_trees_affine_b75",
    )
    parser.add_argument(
        "--position-model-id",
        default="all_sources_temporal_extra_trees",
    )
    parser.add_argument(
        "--exclude-group-id",
        action="append",
        default=[],
        help="Acquisition group excluded from both training and formal evaluation.",
    )
    parser.add_argument(
        "--release-grace-sec",
        type=float,
        default=0.0,
        help="Evaluation-only grace after PX6D release; amplitude metrics are unchanged.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _global_metrics(
    frame: pd.DataFrame,
    *,
    release_grace_sec: float = 0.0,
) -> dict[str, Any]:
    labeled = frame[frame["position_id"].isin(POSITION_ORDER)]
    reference = labeled["true_force_n"].to_numpy(dtype=float)
    estimate = labeled["gated_force_n"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(reference, estimate, 1)
    curve = force_curve_metrics(
        reference,
        estimate,
        labeled["elapsed_time_sec"].to_numpy(dtype=float),
        release_grace_sec=release_grace_sec,
    )
    return {
        "sample_count": int(len(labeled)),
        "session_count": int(labeled["group_id"].nunique()),
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "r2": float(r2_score(reference, estimate)),
        "pearson_r": float(np.corrcoef(reference, estimate)[0, 1]),
        "linear_slope_pred_vs_px6d": float(slope),
        "linear_intercept_n": float(intercept),
        "zero_force_false_response_rate_raw": curve[
            "zero_force_false_response_rate"
        ],
        "zero_force_false_response_rate": curve[
            "zero_force_false_response_rate_after_grace"
        ],
        "release_grace_sec": float(release_grace_sec),
    }


def _position_acceptance(positions: pd.DataFrame) -> pd.DataFrame:
    result = positions.copy()
    result["trend_ok"] = result["pearson_r"] >= 0.85
    result["height_ok"] = result["linear_slope_pred_vs_px6d"].between(0.75, 1.25)
    result["error_ok"] = result["mae_n"] <= 0.60
    release_rate = (
        result["zero_force_false_response_rate_after_grace"]
        if "zero_force_false_response_rate_after_grace" in result
        else result["zero_force_false_response_rate"]
    )
    result["release_false_response_rate_evaluated"] = release_rate
    result["release_ok"] = (
        release_rate.isna()
        | (release_rate <= 0.05)
    )
    passed = result[["trend_ok", "height_ok", "error_ok", "release_ok"]].all(axis=1)
    result["position_curve_status"] = np.where(passed, "acceptable", "needs_review")
    return result


def _plateau_acceptance(linearity: pd.DataFrame) -> pd.DataFrame:
    result = linearity.copy()
    if result.empty:
        return result
    result["plateau_error_ok"] = result["plateau_mae_n"] <= 0.50
    result["plateau_trend_ok"] = result["plateau_pearson_r"] >= 0.90
    result["plateau_height_ok"] = result["plateau_slope"].between(0.75, 1.25)
    passed = result[
        ["plateau_error_ok", "plateau_trend_ok", "plateau_height_ok"]
    ].all(axis=1)
    result["plateau_status"] = np.where(passed, "acceptable", "needs_review")
    return result


def _plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    fields = ("mae_n", "pearson_r", "linear_slope_pred_vs_px6d")
    labels = ("MAE (N)", "Pearson r", "Predicted / PX6D slope")
    colors = {
        "legacy_baseline": "#7A8B99",
        "updated_contact_gate": "#56B4E9",
        "selected_spectral_hybrid": "#009E73",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    x = np.arange(len(POSITION_ORDER))
    for axis, field, label in zip(axes, fields, labels, strict=True):
        for comparison_id, group in metrics.groupby("comparison_model", sort=False):
            values = []
            for position_id in POSITION_ORDER:
                row = group[group["position_id"].astype(str) == position_id]
                values.append(float(row.iloc[0][field]))
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.5,
                color=colors.get(str(comparison_id), "#333333"),
                label=str(comparison_id),
            )
        axis.set_xticks(x, POSITION_ORDER, rotation=45)
        axis.set_ylabel(label)
        axis.grid(alpha=0.16)
        if field == "mae_n":
            axis.axhline(0.60, color="#D55E00", linestyle="--", linewidth=1.0)
        elif field == "pearson_r":
            axis.axhline(0.85, color="#D55E00", linestyle="--", linewidth=1.0)
        else:
            axis.axhspan(0.75, 1.25, color="#009E73", alpha=0.08)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Nine-point grouped OOF optical-force comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_worst_session_traces(
    predictions: pd.DataFrame,
    session_acceptance: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(14.0, 10.0), sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER, strict=True):
        candidates = session_acceptance[
            session_acceptance["position_id"] == position_id
        ]
        if candidates.empty:
            axis.set_axis_off()
            continue
        index = candidates["lag_aligned_mae_n"].astype(float).idxmax()
        group_id = str(candidates.loc[index, "group_id"])
        group = predictions[predictions["group_id"].astype(str) == group_id].sort_values(
            "sample_index"
        )
        axis.plot(
            group["elapsed_time_sec"],
            group["true_force_n"],
            color="#D55E00",
            linewidth=1.6,
            label="PX6D Fz",
        )
        axis.plot(
            group["elapsed_time_sec"],
            group["gated_force_n"],
            color="#0072B2",
            linewidth=1.5,
            label="Optical estimate",
        )
        status = str(candidates.loc[index, "session_curve_status"])
        mae = float(candidates.loc[index, "lag_aligned_mae_n"])
        axis.set_title(f"{position_id} worst | {status} | aligned MAE {mae:.2f} N")
        axis.grid(alpha=0.14)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Worst independent session at each tactile point", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_plateau_regression(plateaus: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 11.0), sharex=True, sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER, strict=True):
        group = plateaus[plateaus["position_id"] == position_id]
        x = group["px6d_mean_n"].to_numpy(dtype=float)
        y = group["optical_mean_n"].to_numpy(dtype=float)
        axis.scatter(x, y, s=20, alpha=0.65, color="#0072B2")
        axis.plot([0, 5], [0, 5], color="#D55E00", linestyle="--", linewidth=1.1)
        if len(x) >= 3 and np.std(x) > 1.0e-12:
            slope, intercept = np.polyfit(x, y, 1)
            axis.plot([0, 5], [intercept, intercept + slope * 5], color="#009E73")
        axis.set_title(position_id)
        axis.set_xlim(0, 5)
        axis.set_ylim(0, 5)
        axis.grid(alpha=0.12)
        axis.set_xlabel("PX6D plateau mean (N)")
        axis.set_ylabel("Optical plateau mean (N)")
    figure.suptitle("Stable plateau linearity | grouped OOF optical force")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}" if math.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    excluded_groups = {str(value) for value in args.exclude_group_id}
    if float(args.release_grace_sec) < 0:
        raise ValueError("release grace must be non-negative")
    aligned = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    source_indices = formal_array_indices(args.fusion_dataset.resolve())
    grouped_oof = pd.read_csv(args.grouped_oof, low_memory=False)
    group_votes = grouped_position_vote(
        grouped_oof, model_id=str(args.position_model_id)
    )
    gate = pd.read_csv(args.contact_gate, low_memory=False)
    gate_map, gate_alignment_audit = aligned_contact_gate_map(
        aligned, source_indices, gate
    )
    all_predictions = pd.read_csv(args.alignment_predictions, low_memory=False)
    legacy = all_predictions[
        all_predictions["model_id"].astype(str) == str(args.baseline_model_id)
    ].copy()
    if legacy.empty:
        raise ValueError(f"baseline model not found: {args.baseline_model_id}")
    updated_gate = apply_grouped_contact_gate(
        legacy,
        gate,
        model_id=str(gate["model_id"].iloc[0]) if "model_id" in gate else "grouped_gate",
    )
    updated_gate["model_id"] = "baseline_with_updated_contact_gate"
    if excluded_groups:
        eligible = ~legacy["group_id"].astype(str).isin(excluded_groups)
        legacy = legacy.loc[eligible].copy()
        updated_gate = updated_gate.loc[
            ~updated_gate["group_id"].astype(str).isin(excluded_groups)
        ].copy()

    views = build_causal_spectral_views(aligned)
    selected = updated_gate.copy()
    expert_parts: list[pd.DataFrame] = []
    split_parts: list[pd.DataFrame] = []
    spec_rows: list[dict[str, Any]] = []
    for position_id, view_id, components in SELECTED_EXPERTS:
        features = views[view_id]
        model_id = f"{position_id.lower()}_{view_id}_pls{components}"
        spec = SpectralForceExpertSpec(
            position_id=position_id,
            model_id=model_id,
            feature_count=int(features.shape[1]),
            latent_components=int(components),
        )
        expert_oof, split_audit = grouped_spectral_force_expert_oof(
            aligned,
            source_indices,
            spec,
            group_position_votes=group_votes,
            gate_active_by_array_index=gate_map,
            feature_matrix=features,
            excluded_group_ids=sorted(excluded_groups),
        )
        expert_parts.append(expert_oof)
        split_audit.insert(0, "position_id", position_id)
        split_audit.insert(1, "model_id", model_id)
        split_parts.append(split_audit)
        selected = apply_spectral_expert_override(
            selected,
            expert_oof,
            model_id="selected_nine_point_spectral_hybrid",
        )
        spec_rows.append(
            {
                "position_id": position_id,
                "model_id": model_id,
                "feature_view": view_id,
                "feature_count": int(features.shape[1]),
                "latent_components": int(components),
                "selection_role": "fixed_before_combined_validation",
            }
        )
    for position_id in ("P12", "P22", "P23"):
        spec_rows.append(
            {
                "position_id": position_id,
                "model_id": str(args.baseline_model_id),
                "feature_view": "retained_current_peak_features",
                "feature_count": None,
                "latent_components": None,
                "selection_role": "retained_baseline",
            }
        )

    comparison_frames: list[pd.DataFrame] = []
    session_frames: list[pd.DataFrame] = []
    global_rows: list[dict[str, Any]] = []
    for comparison_id, frame in (
        ("legacy_baseline", legacy),
        ("updated_contact_gate", updated_gate),
        ("selected_spectral_hybrid", selected),
    ):
        labeled = frame[frame["position_id"].isin(POSITION_ORDER)].copy()
        sessions, positions = build_force_consistency_tables(
            labeled,
            release_grace_sec=float(args.release_grace_sec),
        )
        sessions.insert(0, "comparison_model", comparison_id)
        positions.insert(0, "comparison_model", comparison_id)
        session_frames.append(sessions)
        comparison_frames.append(positions)
        global_rows.append(
            {
                "comparison_model": comparison_id,
                **_global_metrics(
                    frame,
                    release_grace_sec=float(args.release_grace_sec),
                ),
            }
        )
    sessions = pd.concat(session_frames, ignore_index=True)
    positions = pd.concat(comparison_frames, ignore_index=True)
    globals_frame = pd.DataFrame(global_rows)
    selected_sessions = sessions[
        sessions["comparison_model"] == "selected_spectral_hybrid"
    ].copy()
    selected_positions = positions[
        positions["comparison_model"] == "selected_spectral_hybrid"
    ].copy()
    position_acceptance = _position_acceptance(selected_positions)
    session_acceptance = build_session_acceptance_table(selected_sessions)
    plateaus, plateau_linearity = build_stable_plateau_tables(
        selected[selected["position_id"].isin(POSITION_ORDER)]
    )
    plateau_acceptance = _plateau_acceptance(plateau_linearity)
    split_audit = pd.concat(split_parts, ignore_index=True)
    experts = pd.concat(expert_parts, ignore_index=True)

    gate_evaluation = gate[
        ~gate["group_id"].astype(str).isin(excluded_groups)
    ].copy()
    gate_available, gate_active_all = grouped_gate_truth_masks(gate_evaluation)
    gate_detected_all = (
        gate_evaluation["contact_gate_active"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "on"})
        .to_numpy(dtype=bool)
    )
    gate_active = gate_active_all[gate_available]
    gate_detected = gate_detected_all[gate_available]
    if not bool(gate_active.any()) or not bool((~gate_active).any()):
        raise ValueError(
            "contact gate evaluation requires labelled contact and no-contact rows"
        )
    gate_metrics = {
        "active_recall": float(np.mean(gate_detected[gate_active])),
        "zero_false_positive_rate": float(np.mean(gate_detected[~gate_active])),
        "labelled_row_count": int(np.sum(gate_available)),
        "excluded_unlabelled_row_count": int(np.sum(~gate_available)),
    }
    selected_global = globals_frame.set_index("comparison_model").loc[
        "selected_spectral_hybrid"
    ]
    legacy_global = globals_frame.set_index("comparison_model").loc["legacy_baseline"]
    guardrails = {
        "all_nine_positions_pass_trend_height_error_release": bool(
            len(position_acceptance) == 9
            and (position_acceptance["position_curve_status"] == "acceptable").all()
        ),
        "all_nine_positions_pass_stable_plateau_linearity": bool(
            len(plateau_acceptance) == 9
            and (plateau_acceptance["plateau_status"] == "acceptable").all()
        ),
        "all_quality_controlled_sessions_pass_lag_tolerant_curve_guardrail": bool(
            len(session_acceptance)
            == selected[selected["position_id"].isin(POSITION_ORDER)][
                "group_id"
            ].nunique()
            and (session_acceptance["session_curve_status"] == "acceptable").all()
        ),
        "global_mae_improves": bool(selected_global["mae_n"] < legacy_global["mae_n"]),
        "global_r2_improves": bool(selected_global["r2"] > legacy_global["r2"]),
        "contact_gate_active_recall_at_least_0_95": bool(
            gate_metrics["active_recall"] >= 0.95
        ),
        "contact_gate_zero_false_positive_at_most_0_05": bool(
            gate_metrics["zero_false_positive_rate"] <= 0.05
        ),
        "no_group_leakage": bool(
            (split_audit["train_test_group_overlap_count"] == 0).all()
        ),
        "force_sensor_not_used_as_runtime_input": bool(
            not selected["force_sensor_used_as_runtime_input"].astype(bool).any()
        ),
    }
    accepted = all(guardrails.values())

    observability = pd.DataFrame()
    if args.observability_audit.exists():
        observability = pd.read_csv(args.observability_audit, low_memory=False)
        observability_status = (
            observability["observability_status"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        observability = observability[
            ~observability_status.str.startswith("comparable", na=False)
        ].copy()
        observability.to_csv(
            output / "optical_observability_warnings.csv",
            index=False,
            encoding="utf-8-sig",
        )
    decision = {
        "candidate_model_id": (
            "selected_nine_point_spectral_hybrid_quality_controlled"
            if excluded_groups
            else "selected_nine_point_spectral_hybrid"
        ),
        "deployment_recommended": bool(accepted),
        "paper_linear_fit_ready": bool(
            guardrails["all_nine_positions_pass_stable_plateau_linearity"]
            and guardrails[
                "all_quality_controlled_sessions_pass_lag_tolerant_curve_guardrail"
            ]
        ),
        "decision": "all_guardrails_pass" if accepted else "do_not_deploy",
        "evaluation_validity": (
            "formal_grouped_oof_by_session_id_quality_controlled_development_validation"
            if excluded_groups
            else "formal_grouped_oof_by_session_id_development_validation"
        ),
        "release_grace_sec": float(args.release_grace_sec),
        "excluded_group_ids": sorted(excluded_groups),
        "included_session_count": int(session_acceptance["group_id"].nunique()),
        "guardrails": guardrails,
        "contact_gate_metrics": gate_metrics,
        "gate_alignment_audit": gate_alignment_audit,
        "note": (
            "PX6D Fz is supervision/evaluation only. Position, contact and force "
            "estimates use optical inputs at runtime."
        ),
    }

    exclusion_rows: list[dict[str, Any]] = []
    for group_id in sorted(excluded_groups):
        audit_match = pd.DataFrame()
        if args.observability_audit.exists():
            full_observability = pd.read_csv(args.observability_audit, low_memory=False)
            audit_match = full_observability[
                full_observability["group_id"].astype(str) == group_id
            ]
        row: dict[str, Any] = {
            "group_id": group_id,
            "quality_control_status": "excluded_before_training_and_evaluation",
            "exclusion_reason": "low_optical_sensitivity_requires_remeasurement",
        }
        if not audit_match.empty:
            for field in (
                "position_id",
                "sensitivity_ratio_to_position_median",
                "sensitivity_robust_z",
                "observability_status",
            ):
                row[field] = audit_match.iloc[0].get(field)
        exclusion_rows.append(row)
    exclusions = pd.DataFrame(exclusion_rows)
    if not exclusions.empty:
        exclusions.to_csv(
            output / "quality_control_exclusions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    selected.to_csv(output / "selected_hybrid_grouped_oof_predictions.csv", index=False, encoding="utf-8-sig")
    experts.to_csv(output / "selected_expert_grouped_oof.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(output / "selected_expert_group_split_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(spec_rows).to_csv(output / "selected_expert_specs.csv", index=False, encoding="utf-8-sig")
    globals_frame.to_csv(output / "global_model_comparison.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(output / "metrics_by_position_all_models.csv", index=False, encoding="utf-8-sig")
    position_acceptance.to_csv(output / "selected_position_acceptance.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(output / "metrics_by_session_all_models.csv", index=False, encoding="utf-8-sig")
    session_acceptance.to_csv(output / "selected_session_acceptance.csv", index=False, encoding="utf-8-sig")
    plateaus.to_csv(output / "stable_plateau_means.csv", index=False, encoding="utf-8-sig")
    plateau_acceptance.to_csv(output / "paper_linear_fit_by_position.csv", index=False, encoding="utf-8-sig")
    (output / "decision.json").write_text(
        json.dumps(_json_safe(decision), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    labeled_selected = selected[selected["position_id"].isin(POSITION_ORDER)].copy()
    plot_representative_traces(
        labeled_selected,
        selected_sessions,
        output / "nine_point_representative_force_traces.png",
    )
    plot_position_regression(
        labeled_selected, output / "nine_point_force_regression.png"
    )
    _plot_model_comparison(positions, output / "nine_point_model_comparison.png")
    _plot_worst_session_traces(
        labeled_selected,
        session_acceptance,
        output / "nine_point_worst_session_force_traces.png",
    )
    _plot_plateau_regression(
        plateaus, output / "nine_point_stable_plateau_regression.png"
    )

    report_positions = position_acceptance[
        [
            "position_id",
            "mae_n",
            "pearson_r",
            "linear_slope_pred_vs_px6d",
            "zero_force_false_response_rate",
            "zero_force_false_response_rate_after_grace",
            "position_curve_status",
        ]
    ]
    report_plateaus = plateau_acceptance[
        [
            "position_id",
            "plateau_count",
            "plateau_mae_n",
            "plateau_pearson_r",
            "plateau_slope",
            "plateau_status",
        ]
    ]
    failed_sessions = session_acceptance[
        session_acceptance["session_curve_status"] != "acceptable"
    ][
        [
            "position_id",
            "group_id",
            "trend_correlation_best",
            "lag_aligned_mae_n",
            "linear_slope_pred_vs_px6d",
            "failure_reasons",
        ]
    ]
    report = [
        "# Selected nine-point optical-force validation",
        "",
        "This report evaluates grouped out-of-fold optical force estimates against "
        "PX6D Fz. Small temporal lag is allowed in the per-session trend check, but "
        "force height and stable-plateau linearity remain mandatory.",
        "",
        "## Decision",
        "",
        f"- Result: **{decision['decision']}**",
        f"- Deployment recommended: **{decision['deployment_recommended']}**",
        f"- Paper linear-fit ready: **{decision['paper_linear_fit_ready']}**",
        f"- Position guardrails passed: **{int((position_acceptance['position_curve_status'] == 'acceptable').sum())}/9**",
        f"- Stable plateau guardrails passed: **{int((plateau_acceptance['plateau_status'] == 'acceptable').sum())}/9**",
        f"- Session guardrails passed: **{int((session_acceptance['session_curve_status'] == 'acceptable').sum())}/{len(session_acceptance)}**",
        f"- Release grace used only for residual evaluation: **{float(args.release_grace_sec):.2f} s**",
        f"- Quality-control exclusions: **{len(excluded_groups)}**",
        "",
        "## Per-position curve evidence",
        "",
        _markdown_table(report_positions),
        "",
        "## Stable plateau linearity",
        "",
        _markdown_table(report_plateaus),
        "",
        "## Sessions requiring review",
        "",
        _markdown_table(failed_sessions) if not failed_sessions.empty else "None.",
        "",
        "## Quality-control exclusions",
        "",
        _markdown_table(exclusions) if not exclusions.empty else "None.",
        "",
        "## Guardrails",
        "",
        *[f"- {name}: {value}" for name, value in guardrails.items()],
        "",
        "The selected expert specifications were fixed before this combined run. "
        "This remains development grouped-OOF evidence; a future independent "
        "acquisition should be held out for final paper claims.",
    ]
    (output / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_json_safe(decision), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
