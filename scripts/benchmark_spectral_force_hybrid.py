"""Validate a full-spectrum P13 force expert inside the nine-point pipeline."""

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
    plot_position_regression,
    plot_representative_traces,
)
from src.hybrid_spectrum.position_conditioned_force import (  # noqa: E402
    grouped_position_vote,
)
from src.hybrid_spectrum.spectral_force_experts import (  # noqa: E402
    SpectralForceExpertSpec,
    apply_spectral_expert_override,
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
    PROJECT_ROOT
    / "outputs/ordinary_fbg_all_data_fusion_training_20260804_initial_fixed5"
)
DEFAULT_ALIGNMENT = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_nine_point_force_alignment_20260804_initial_fixed5"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/ordinary_fbg_spectral_force_hybrid_20260804"
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
    parser.add_argument(
        "--contact-gate",
        type=Path,
        default=DEFAULT_TRAINING / "force_contact_gate_oof_predictions.csv",
    )
    parser.add_argument(
        "--alignment-predictions",
        type=Path,
        default=DEFAULT_ALIGNMENT / "nine_point_force_alignment_oof_predictions.csv",
    )
    parser.add_argument(
        "--baseline-model-id",
        default="position_expert_current_extra_trees_affine_b75",
    )
    parser.add_argument(
        "--position-model-id",
        default="all_sources_temporal_extra_trees",
    )
    parser.add_argument("--latent-components", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _global_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    reference = frame["true_force_n"].to_numpy(dtype=float)
    estimate = frame["gated_force_n"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(reference, estimate, 1)
    zero = reference <= 0.03
    return {
        "sample_count": int(len(frame)),
        "session_count": int(frame["group_id"].nunique()),
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "r2": float(r2_score(reference, estimate)),
        "pearson_r": float(np.corrcoef(reference, estimate)[0, 1]),
        "linear_slope_pred_vs_px6d": float(slope),
        "linear_intercept_n": float(intercept),
        "zero_force_false_response_rate": float(
            np.mean(estimate[zero] > 0.10) if np.any(zero) else math.nan
        ),
    }


def _stable_plateaus(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.arange(0.0, 5.01, 0.5)
    rows: list[dict[str, Any]] = []
    for (position_id, group_id), group in frame.groupby(
        ["position_id", "group_id"], sort=False
    ):
        group = group.sort_values("sample_index")
        elapsed = group["elapsed_time_sec"].to_numpy(dtype=float)
        reference = group["true_force_n"].to_numpy(dtype=float)
        estimate = group["gated_force_n"].to_numpy(dtype=float)
        dt = np.diff(elapsed, prepend=np.nan)
        speed = np.divide(
            np.abs(np.diff(reference, prepend=np.nan)),
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
                    "position_id": str(position_id),
                    "group_id": str(group_id),
                    "force_bin_low_n": float(bins[index]),
                    "force_bin_high_n": float(bins[index + 1]),
                    "stable_frame_count": int(np.sum(selected)),
                    "px6d_mean_n": float(np.mean(reference[selected])),
                    "optical_mean_n": float(np.mean(estimate[selected])),
                }
            )
    plateaus = pd.DataFrame(rows)
    lines: list[dict[str, Any]] = []
    for position_id, group in plateaus.groupby("position_id", sort=False):
        x = group["px6d_mean_n"].to_numpy(dtype=float)
        y = group["optical_mean_n"].to_numpy(dtype=float)
        if len(x) < 3 or float(np.std(x)) <= 1.0e-12:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        lines.append(
            {
                "position_id": position_id,
                "plateau_count": int(len(group)),
                "plateau_mae_n": float(mean_absolute_error(x, y)),
                "plateau_r2": float(r2_score(x, y)),
                "plateau_slope": float(slope),
                "plateau_intercept_n": float(intercept),
                "plateau_pearson_r": float(np.corrcoef(x, y)[0, 1]),
            }
        )
    return plateaus, pd.DataFrame(lines)


def _plot_model_comparison(
    position_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = ("mae_n", "pearson_r", "linear_slope_pred_vs_px6d")
    labels = ("MAE (N)", "Pearson r", "Slope")
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    x = np.arange(len(POSITION_ORDER))
    for axis, field, label in zip(axes, metrics, labels, strict=True):
        for model_id, color, marker in (
            ("baseline", "#7A8B99", "o"),
            ("spectral_hybrid", "#0072B2", "s"),
        ):
            rows = position_metrics[position_metrics["comparison_model"] == model_id]
            values = []
            for position in POSITION_ORDER:
                row = rows[rows["position_id"].astype(str) == position]
                values.append(float(row.iloc[0][field]))
            axis.plot(x, values, marker=marker, color=color, label=model_id)
        axis.set_xticks(x, POSITION_ORDER, rotation=45)
        axis.set_ylabel(label)
        axis.grid(alpha=0.17)
        if field == "mae_n":
            axis.axhline(0.60, color="#D55E00", linestyle="--", linewidth=1.0)
        elif field == "pearson_r":
            axis.axhline(0.85, color="#D55E00", linestyle="--", linewidth=1.0)
        else:
            axis.axhspan(0.75, 1.25, color="#009E73", alpha=0.08)
    axes[0].legend(frameon=False)
    figure.suptitle("Nine-point grouped OOF force consistency")
    figure.tight_layout()
    figure.savefig(output_path, dpi=210, facecolor="white")
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
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
    aligned = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    source_indices = formal_array_indices(args.fusion_dataset.resolve())
    grouped_oof = pd.read_csv(args.grouped_oof, low_memory=False)
    group_votes = grouped_position_vote(
        grouped_oof, model_id=str(args.position_model_id)
    )
    gate = pd.read_csv(args.contact_gate, low_memory=False)
    gate_map = dict(
        zip(
            gate["array_index"].astype(int).tolist(),
            gate["contact_gate_active"].astype(bool).tolist(),
        )
    )
    all_predictions = pd.read_csv(args.alignment_predictions, low_memory=False)
    baseline = all_predictions[
        all_predictions["model_id"].astype(str) == str(args.baseline_model_id)
    ].copy()
    if baseline.empty:
        raise ValueError(f"baseline model not found: {args.baseline_model_id}")

    spec = SpectralForceExpertSpec(
        latent_components=int(args.latent_components),
    )
    expert_oof, split_audit = grouped_spectral_force_expert_oof(
        aligned,
        source_indices,
        spec,
        group_position_votes=group_votes,
        gate_active_by_array_index=gate_map,
    )
    hybrid = apply_spectral_expert_override(
        baseline,
        expert_oof,
        model_id="spectral_hybrid_p13_pls12",
    )

    comparison_parts: list[pd.DataFrame] = []
    session_parts: list[pd.DataFrame] = []
    global_rows: list[dict[str, Any]] = []
    for comparison_id, frame in (("baseline", baseline), ("spectral_hybrid", hybrid)):
        sessions, positions = build_force_consistency_tables(frame)
        sessions.insert(0, "comparison_model", comparison_id)
        positions.insert(0, "comparison_model", comparison_id)
        session_parts.append(sessions)
        comparison_parts.append(positions)
        global_rows.append(
            {
                "comparison_model": comparison_id,
                **_global_metrics(frame),
                "consistent_position_count": int(
                    (positions["audit_status"] == "consistent").sum()
                ),
                "needs_review_position_count": int(
                    (positions["audit_status"] == "needs_review").sum()
                ),
                "worst_position_mae_n": float(positions["mae_n"].max()),
            }
        )
    positions = pd.concat(comparison_parts, ignore_index=True)
    sessions = pd.concat(session_parts, ignore_index=True)
    globals_frame = pd.DataFrame(global_rows)
    hybrid_positions = positions[positions["comparison_model"] == "spectral_hybrid"]
    hybrid_global = globals_frame.set_index("comparison_model").loc["spectral_hybrid"]
    baseline_global = globals_frame.set_index("comparison_model").loc["baseline"]
    guardrails = {
        "all_nine_positions_consistent": bool(
            (hybrid_positions["audit_status"] == "consistent").sum() == 9
        ),
        "global_mae_not_materially_worse": bool(
            hybrid_global["mae_n"] <= baseline_global["mae_n"] + 0.02
        ),
        "global_r2_not_materially_worse": bool(
            hybrid_global["r2"] >= baseline_global["r2"] - 0.01
        ),
        "zero_force_residual_not_worse": bool(
            hybrid_global["zero_force_false_response_rate"]
            <= baseline_global["zero_force_false_response_rate"] + 0.005
        ),
        "no_group_leakage": bool(
            (split_audit["train_test_group_overlap_count"] == 0).all()
        ),
    }
    accepted = all(guardrails.values())
    decision = {
        "candidate_model_id": "spectral_hybrid_p13_pls12",
        "deployment_recommended": bool(accepted),
        "decision": "candidate_passes_9_of_9" if accepted else "do_not_deploy",
        "guardrails": guardrails,
        "note": (
            "PX6D Fz is supervision/evaluation only; runtime uses optical spectrum, "
            "optical position vote, and optical contact gate."
        ),
    }

    plateaus, linearity = _stable_plateaus(hybrid)
    expert_oof.to_csv(output / "p13_spectral_expert_oof.csv", index=False, encoding="utf-8-sig")
    hybrid.to_csv(output / "hybrid_grouped_oof_predictions.csv", index=False, encoding="utf-8-sig")
    globals_frame.to_csv(output / "global_model_comparison.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(output / "metrics_by_position.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(output / "metrics_by_session.csv", index=False, encoding="utf-8-sig")
    split_audit.to_csv(output / "group_split_audit.csv", index=False, encoding="utf-8-sig")
    plateaus.to_csv(output / "stable_plateau_means.csv", index=False, encoding="utf-8-sig")
    linearity.to_csv(output / "paper_linear_fit_by_position.csv", index=False, encoding="utf-8-sig")
    (output / "decision.json").write_text(
        json.dumps(_json_safe(decision), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    plot_representative_traces(
        hybrid,
        sessions[sessions["comparison_model"] == "spectral_hybrid"],
        output / "nine_point_representative_force_traces.png",
    )
    plot_position_regression(hybrid, output / "nine_point_force_regression.png")
    _plot_model_comparison(positions, output / "nine_point_model_comparison.png")

    report_positions = hybrid_positions[
        [
            "position_id",
            "mae_n",
            "r2",
            "pearson_r",
            "linear_slope_pred_vs_px6d",
            "lag_ms",
            "audit_status",
            "audit_reason",
        ]
    ].copy()
    report = [
        "# Nine-point spectral force hybrid validation",
        "",
        "This is a grouped out-of-fold validation. Each test session is excluded "
        "from fitting. PX6D Fz is not a runtime input.",
        "",
        "## Decision",
        "",
        f"- Result: **{decision['decision']}**",
        f"- Deployment recommended: **{decision['deployment_recommended']}**",
        f"- Consistent points: **{int((hybrid_positions['audit_status'] == 'consistent').sum())}/9**",
        "- Architecture: existing optical force model for eight points; grouped "
        "full-spectrum PLS expert for optically routed P13 sessions.",
        "",
        "## Per-position evidence",
        "",
        _markdown_table(report_positions),
        "",
        "## Guardrails",
        "",
        *[f"- {name}: {value}" for name, value in guardrails.items()],
        "",
        "The reported linear slope is prediction versus PX6D Fz. Stable-plateau "
        "means are exported separately for later paper fitting.",
    ]
    (output / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_json_safe(decision), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
