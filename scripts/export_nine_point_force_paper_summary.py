"""Export publication-ready nine-point force reconstruction summaries.

The script consumes grouped out-of-file predictions produced by
``validate_selected_nine_point_force_hybrid.py``. It never refits or rescales
held-out predictions; the plotted optical force is exactly the saved OOF value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_nine_point_force_paper_alignment_20260804"
)
POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")

INK = "#17324D"
MUTED = "#61758A"
GRID = "#DCE5EC"
PX6D = "#244A73"
OPTICAL = "#E06B4F"
FIT = "#159A8C"
IDEAL = "#707B86"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.edgecolor": "#9BAEBE",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _representative_groups(session_metrics: pd.DataFrame) -> dict[str, str]:
    selected: dict[str, str] = {}
    for position_id in POSITION_ORDER:
        rows = session_metrics[session_metrics["position_id"] == position_id].copy()
        if rows.empty:
            continue
        target_mae = float(rows["lag_aligned_mae_n"].median())
        target_r = float(rows["trend_correlation_best"].median())
        score = (
            (rows["lag_aligned_mae_n"].astype(float) - target_mae).abs()
            + 0.35
            * (rows["trend_correlation_best"].astype(float) - target_r).abs()
        )
        selected[position_id] = str(rows.loc[score.idxmin(), "group_id"])
    return selected


def _global_selected_metrics(frame: pd.DataFrame) -> pd.Series:
    selected = frame[frame["comparison_model"] == "selected_spectral_hybrid"]
    if selected.empty:
        raise ValueError("selected_spectral_hybrid is missing from global metrics")
    return selected.iloc[0]


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_force_traces(
    predictions: pd.DataFrame,
    sessions: pd.DataFrame,
    positions: pd.DataFrame,
    global_metrics: pd.Series,
    output_stem: Path,
) -> pd.DataFrame:
    representatives = _representative_groups(sessions)
    figure, axes = plt.subplots(3, 3, figsize=(16.2, 10.3), sharey=True)
    exported: list[pd.DataFrame] = []

    for panel_index, (axis, position_id) in enumerate(
        zip(axes.flat, POSITION_ORDER, strict=True)
    ):
        group_id = representatives.get(position_id)
        group = predictions[
            (predictions["position_id"] == position_id)
            & (predictions["group_id"].astype(str) == str(group_id))
        ].sort_values("sample_index")
        metric = positions[positions["position_id"] == position_id]
        if group.empty or metric.empty:
            axis.set_axis_off()
            continue

        time_s = group["elapsed_time_sec"].to_numpy(dtype=float)
        reference = group["true_force_n"].to_numpy(dtype=float)
        estimate = group["gated_force_n"].to_numpy(dtype=float)
        axis.fill_between(time_s, reference, estimate, color=OPTICAL, alpha=0.08, linewidth=0)
        axis.plot(time_s, reference, color=PX6D, linewidth=2.25, label="PX6D Fz")
        axis.plot(
            time_s,
            estimate,
            color=OPTICAL,
            linewidth=1.9,
            label="Optical estimate",
        )

        row = metric.iloc[0]
        axis.text(
            0.02,
            0.94,
            position_id,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=12.5,
            fontweight="bold",
            color=INK,
        )
        axis.text(
            0.98,
            0.94,
            (
                f"r={float(row['pearson_r']):.3f}  "
                f"slope={float(row['linear_slope_pred_vs_px6d']):.3f}\n"
                f"MAE={float(row['mae_n']):.2f} N  "
                f"lag={float(row['lag_ms']):+.0f} ms"
            ),
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=8.3,
            color=MUTED,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
        axis.set_ylim(-0.12, 5.25)
        axis.set_xlim(float(np.min(time_s)), float(np.max(time_s)))
        axis.set_yticks(np.arange(0, 5.1, 1.0))
        axis.grid(True, color=GRID, linewidth=0.65, alpha=0.72)
        axis.spines[["top", "right"]].set_visible(False)
        if panel_index // 3 == 2:
            axis.set_xlabel("Time (s)")
        if panel_index % 3 == 0:
            axis.set_ylabel("Force (N)")

        export = group[
            ["position_id", "group_id", "sample_index", "elapsed_time_sec", "true_force_n", "gated_force_n"]
        ].copy()
        export["trace_role"] = "representative_grouped_oof_session"
        exported.append(export)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    figure.suptitle(
        "Nine-point optical force reconstruction",
        fontsize=19,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.953,
        (
            "Grouped out-of-file validation | 0-5 N shared scale | "
            f"overall r={float(global_metrics['pearson_r']):.3f}, "
            f"R2={float(global_metrics['r2']):.3f}, "
            f"MAE={float(global_metrics['mae_n']):.3f} N, "
            f"slope={float(global_metrics['linear_slope_pred_vs_px6d']):.3f}"
        ),
        ha="center",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )
    figure.text(
        0.01,
        0.008,
        (
            "Panel metrics use all held-out sessions at each position. "
            "PX6D is supervision/evaluation only; optical estimate is runtime input-only."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.90), h_pad=1.2, w_pad=1.0)
    _save_figure(figure, output_stem)
    return pd.concat(exported, ignore_index=True)


def plot_plateau_linearity(
    plateaus: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.6, 11.2), sharex=True, sharey=True)
    for panel_index, (axis, position_id) in enumerate(
        zip(axes.flat, POSITION_ORDER, strict=True)
    ):
        group = plateaus[plateaus["position_id"] == position_id]
        metric = fit_metrics[fit_metrics["position_id"] == position_id]
        if group.empty or metric.empty:
            axis.set_axis_off()
            continue
        x = group["px6d_mean_n"].to_numpy(dtype=float)
        y = group["optical_mean_n"].to_numpy(dtype=float)
        row = metric.iloc[0]
        slope = float(row["plateau_slope"])
        intercept = float(row["plateau_intercept_n"])

        axis.scatter(
            x,
            y,
            s=26,
            facecolor="#6DB9D0",
            edgecolor=PX6D,
            linewidth=0.45,
            alpha=0.78,
            zorder=3,
        )
        axis.plot([0, 5], [0, 5], color=IDEAL, linestyle=(0, (4, 3)), linewidth=1.2)
        axis.plot(
            [0, 5],
            [intercept, intercept + slope * 5],
            color=OPTICAL,
            linewidth=2.0,
        )
        axis.text(
            0.04,
            0.94,
            position_id,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=12.5,
            fontweight="bold",
        )
        axis.text(
            0.96,
            0.94,
            (
                f"n={int(row['plateau_count'])}\n"
                f"r={float(row['plateau_pearson_r']):.3f}\n"
                f"slope={slope:.3f}\n"
                f"MAE={float(row['plateau_mae_n']):.2f} N"
            ),
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=8.2,
            color=MUTED,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
        axis.set_xlim(0, 5.1)
        axis.set_ylim(0, 5.1)
        axis.set_xticks(np.arange(0, 5.1, 1.0))
        axis.set_yticks(np.arange(0, 5.1, 1.0))
        axis.grid(True, color=GRID, linewidth=0.65, alpha=0.72)
        axis.spines[["top", "right"]].set_visible(False)
        if panel_index // 3 == 2:
            axis.set_xlabel("PX6D plateau force (N)")
        if panel_index % 3 == 0:
            axis.set_ylabel("Optical estimate (N)")

    legend_handles = [
        plt.Line2D([0], [0], color=IDEAL, linestyle=(0, (4, 3)), label="Ideal y=x"),
        plt.Line2D([0], [0], color=OPTICAL, linewidth=2, label="Per-position linear fit"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6DB9D0",
            markeredgecolor=PX6D,
            label="Stable plateau mean",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.934),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        "Nine-point stable-plateau force linearity",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        "Grouped out-of-file optical force vs PX6D Fz | transitions excluded | shared 0-5 N axes",
        ha="center",
        va="top",
        fontsize=10.3,
        color=MUTED,
    )
    figure.text(
        0.01,
        0.008,
        "Each marker is a stable force-bin mean from an independent acquisition session; no test-trace rescaling is applied.",
        fontsize=8.2,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.90), h_pad=1.0, w_pad=0.9)
    _save_figure(figure, output_stem)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "paper_summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(input_dir / "selected_hybrid_grouped_oof_predictions.csv")
    sessions = pd.read_csv(input_dir / "selected_session_acceptance.csv")
    positions = pd.read_csv(input_dir / "selected_position_acceptance.csv")
    globals_frame = pd.read_csv(input_dir / "global_model_comparison.csv")
    plateaus = pd.read_csv(input_dir / "stable_plateau_means.csv")
    fit_metrics = pd.read_csv(input_dir / "paper_linear_fit_by_position.csv")
    decision = json.loads((input_dir / "decision.json").read_text(encoding="utf-8"))

    _apply_style()
    representative = plot_force_traces(
        predictions,
        sessions,
        positions,
        _global_selected_metrics(globals_frame),
        output_dir / "nine_point_force_trace_summary",
    )
    plot_plateau_linearity(
        plateaus,
        fit_metrics,
        output_dir / "nine_point_plateau_linearity_summary",
    )

    representative.rename(
        columns={
            "elapsed_time_sec": "time_s",
            "true_force_n": "px6d_fz_n",
            "gated_force_n": "optical_estimated_force_n",
        }
    ).to_csv(
        output_dir / "origin_representative_force_traces.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plateaus.to_csv(
        output_dir / "origin_stable_plateau_force_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    positions.to_csv(
        output_dir / "nine_point_force_alignment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_global = _global_selected_metrics(globals_frame)
    report = [
        "# Nine-point optical force alignment summary",
        "",
        "- Evaluation: grouped out-of-file predictions by acquisition session.",
        f"- Included sessions: {int(decision['included_session_count'])}.",
        f"- Quality-control exclusions: {len(decision['excluded_group_ids'])}.",
        "- Exclusion rule: pre-defined optical observability audit, not test error.",
        "- PX6D Fz is supervision/evaluation only and is not a runtime model input.",
        "- No held-out curve is shifted, stretched, or rescaled after prediction.",
        "",
        "## Overall grouped OOF result",
        "",
        f"- MAE: {float(selected_global['mae_n']):.3f} N",
        f"- R2: {float(selected_global['r2']):.3f}",
        f"- Pearson r: {float(selected_global['pearson_r']):.3f}",
        f"- Predicted-vs-PX6D slope: {float(selected_global['linear_slope_pred_vs_px6d']):.3f}",
        f"- Intercept: {float(selected_global['linear_intercept_n']):.3f} N",
        "",
        "## Guardrails",
        "",
    ]
    for key, value in decision["guardrails"].items():
        report.append(f"- {key}: {bool(value)}")
    position_curve_pass = bool(
        decision["guardrails"].get(
            "all_nine_positions_pass_trend_height_error_release", False
        )
    )
    plateau_pass = bool(
        decision["guardrails"].get(
            "all_nine_positions_pass_stable_plateau_linearity", False
        )
    )
    session_curve_pass = bool(
        decision["guardrails"].get(
            "all_quality_controlled_sessions_pass_lag_tolerant_curve_guardrail",
            False,
        )
    )
    if position_curve_pass and plateau_pass and session_curve_pass:
        interpretation = (
            "All nine position-level curve, stable-plateau and session-level "
            "guardrails pass."
        )
    elif plateau_pass:
        interpretation = (
            "All nine stable-plateau linearity checks pass, but the continuous "
            "curve/release guardrail does not pass at every position or session. "
            "This remains development-validation evidence rather than final "
            "calibration evidence."
        )
    else:
        interpretation = (
            "One or more position-level and stable-plateau guardrails do not "
            "pass. This result is exploratory development evidence and is not "
            "final calibration evidence."
        )
    report.extend(
        [
            "",
            interpretation,
        ]
    )
    (output_dir / "paper_force_alignment_summary.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
