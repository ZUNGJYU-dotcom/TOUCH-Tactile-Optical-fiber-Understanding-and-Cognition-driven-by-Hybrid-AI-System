"""Grouped optical-force consistency audit for all nine ordinary-FBG points.

The audit consumes grouped out-of-fold predictions only.  It deliberately
does not use force-sensor values as model inputs and never calibrates a test
session from its own PX6D trace.  The resulting diagnostics are suitable for
Measurement provenance checks and for deciding whether a force model is ready
to show as current evidence.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .measurement_consistency import estimate_frame_lag


POSITION_ORDER = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)
_POSITION_PATTERN = re.compile(r"(?:^|_)(P[123][123])(?:_|$)")


def infer_position_id(value: Any) -> str | None:
    match = _POSITION_PATTERN.search(str(value or ""))
    return match.group(1) if match else None


def load_grouped_force_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), encoding="utf-8-sig")
    required = {
        "group_id",
        "file_id",
        "sample_index",
        "elapsed_time_sec",
        "true_force_n",
        "gated_force_n",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("grouped force predictions missing: " + ", ".join(missing))
    frame["position_id"] = [
        infer_position_id(group_id) or infer_position_id(file_id)
        for group_id, file_id in zip(frame["group_id"], frame["file_id"])
    ]
    frame = frame[frame["position_id"].isin(POSITION_ORDER)].copy()
    for field in ("elapsed_time_sec", "true_force_n", "gated_force_n"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame[
        np.isfinite(frame["elapsed_time_sec"])
        & np.isfinite(frame["true_force_n"])
        & np.isfinite(frame["gated_force_n"])
    ].copy()
    if frame.empty:
        raise ValueError("no usable nine-point grouped force predictions found")
    return frame.sort_values(["position_id", "group_id", "sample_index"])


def _safe_r2(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    if denominator <= 1.0e-12:
        return None
    return float(1.0 - np.sum((estimate - reference) ** 2) / denominator)


def _safe_correlation(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    if (
        len(reference) < 3
        or float(np.std(reference)) <= 1.0e-12
        or float(np.std(estimate)) <= 1.0e-12
    ):
        return None
    return float(np.corrcoef(reference, estimate)[0, 1])


def _aligned_arrays(
    reference: np.ndarray,
    estimate: np.ndarray,
    lag_frames: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_frames is None or lag_frames == 0:
        return reference, estimate
    if lag_frames > 0:
        return reference[:-lag_frames], estimate[lag_frames:]
    return reference[-lag_frames:], estimate[:lag_frames]


def force_curve_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    elapsed_time_sec: np.ndarray,
    *,
    zero_force_max_n: float = 0.03,
    residual_warning_n: float = 0.10,
    maximum_lag_sec: float = 2.0,
    release_grace_sec: float = 0.0,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    elapsed = np.asarray(elapsed_time_sec, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(estimate) & np.isfinite(elapsed)
    reference = reference[valid]
    estimate = estimate[valid]
    elapsed = elapsed[valid]
    if len(reference) == 0:
        raise ValueError("force curve has no finite paired samples")

    error = estimate - reference
    intervals = np.diff(elapsed)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    median_interval = float(np.median(intervals)) if len(intervals) else math.nan
    lag = estimate_frame_lag(
        reference,
        estimate,
        median_frame_interval_sec=median_interval,
        maximum_lag_sec=maximum_lag_sec,
    )
    aligned_reference, aligned_estimate = _aligned_arrays(
        reference, estimate, lag.get("lag_frames")
    )

    if float(np.std(reference)) > 1.0e-12:
        slope, intercept = np.polyfit(reference, estimate, 1)
    else:
        slope, intercept = math.nan, math.nan
    reference_p95 = float(np.percentile(reference, 95))
    estimate_p95 = float(np.percentile(estimate, 95))
    reference_p05 = float(np.percentile(reference, 5))
    estimate_p05 = float(np.percentile(estimate, 5))
    reference_span = reference_p95 - reference_p05
    estimate_span = estimate_p95 - estimate_p05
    zero_mask = reference <= zero_force_max_n
    zero_after_grace = np.zeros(len(reference), dtype=bool)
    last_active_time: float | None = None
    previous_time: float | None = None
    grace = max(0.0, float(release_grace_sec))
    for index, (time_value, force_value) in enumerate(
        zip(elapsed, reference, strict=True)
    ):
        if previous_time is not None and time_value < previous_time:
            last_active_time = None
        if force_value > zero_force_max_n:
            last_active_time = float(time_value)
        elif last_active_time is None or time_value - last_active_time >= grace:
            zero_after_grace[index] = True
        previous_time = float(time_value)

    return {
        "paired_sample_count": int(len(reference)),
        "duration_sec": float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0,
        "frame_interval_median_sec": median_interval if math.isfinite(median_interval) else None,
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(error**2))),
        "bias_n": float(np.mean(error)),
        "r2": _safe_r2(reference, estimate),
        "pearson_r": _safe_correlation(reference, estimate),
        "linear_slope_pred_vs_px6d": float(slope) if math.isfinite(slope) else None,
        "linear_intercept_n": float(intercept) if math.isfinite(intercept) else None,
        "px6d_p95_n": reference_p95,
        "optical_p95_n": estimate_p95,
        "px6d_p05_n": reference_p05,
        "optical_p05_n": estimate_p05,
        "amplitude_ratio_p95_p05": (
            float(estimate_span / reference_span) if reference_span > 1.0e-12 else None
        ),
        "peak_force_px6d_n": float(np.max(reference)),
        "peak_force_optical_n": float(np.max(estimate)),
        "lag_frames": lag.get("lag_frames"),
        "lag_ms": lag.get("lag_ms"),
        "lag_search_correlation": lag.get("correlation"),
        "lag_aligned_pearson_r": _safe_correlation(
            aligned_reference, aligned_estimate
        ),
        "lag_aligned_mae_n": float(
            np.mean(np.abs(aligned_estimate - aligned_reference))
        ),
        "zero_force_frame_count": int(np.sum(zero_mask)),
        "zero_force_false_response_rate": (
            float(np.mean(estimate[zero_mask] > residual_warning_n))
            if np.any(zero_mask)
            else None
        ),
        "release_grace_sec": grace,
        "zero_force_after_grace_frame_count": int(np.sum(zero_after_grace)),
        "zero_force_false_response_rate_after_grace": (
            float(np.mean(estimate[zero_after_grace] > residual_warning_n))
            if np.any(zero_after_grace)
            else None
        ),
    }


def _audit_status(metrics: dict[str, Any]) -> tuple[str, str]:
    slope = metrics.get("linear_slope_pred_vs_px6d")
    correlation = metrics.get("pearson_r")
    mae = metrics.get("mae_n")
    false_rate = metrics.get("zero_force_false_response_rate")
    reasons: list[str] = []
    if correlation is None or correlation < 0.85:
        reasons.append("weak_curve_similarity")
    if slope is None or slope < 0.75:
        reasons.append("force_amplitude_underestimated")
    elif slope > 1.25:
        reasons.append("force_amplitude_overestimated")
    if mae is None or mae > 0.60:
        reasons.append("high_force_error")
    if false_rate is not None and false_rate > 0.05:
        reasons.append("release_or_zero_force_residual")
    if not reasons:
        return "consistent", ""
    if len(reasons) == 1 and reasons[0] not in {
        "weak_curve_similarity",
        "high_force_error",
    }:
        return "usable_with_warning", ";".join(reasons)
    return "needs_review", ";".join(reasons)


def build_force_consistency_tables(
    predictions: pd.DataFrame,
    *,
    release_grace_sec: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_rows: list[dict[str, Any]] = []
    for (position_id, group_id), group in predictions.groupby(
        ["position_id", "group_id"], sort=False
    ):
        metrics = force_curve_metrics(
            group["true_force_n"].to_numpy(dtype=float),
            group["gated_force_n"].to_numpy(dtype=float),
            group["elapsed_time_sec"].to_numpy(dtype=float),
            release_grace_sec=release_grace_sec,
        )
        status, reason = _audit_status(metrics)
        session_rows.append(
            {
                "position_id": position_id,
                "group_id": group_id,
                "file_id": str(group["file_id"].iloc[0]),
                "fold_id": int(group["fold_id"].iloc[0]),
                "evaluation_validity": "formal_grouped_oof_by_session_id",
                "audit_status": status,
                "audit_reason": reason,
                **metrics,
            }
        )
    sessions = pd.DataFrame(session_rows)

    position_rows: list[dict[str, Any]] = []
    for position_id in POSITION_ORDER:
        group = predictions[predictions["position_id"] == position_id]
        if group.empty:
            continue
        metrics = force_curve_metrics(
            group["true_force_n"].to_numpy(dtype=float),
            group["gated_force_n"].to_numpy(dtype=float),
            group["elapsed_time_sec"].to_numpy(dtype=float),
            release_grace_sec=release_grace_sec,
        )
        position_sessions = sessions[sessions["position_id"] == position_id]
        status, reason = _audit_status(metrics)
        position_rows.append(
            {
                "position_id": position_id,
                "session_count": int(len(position_sessions)),
                "evaluation_validity": "formal_grouped_oof_by_session_id",
                "audit_status": status,
                "audit_reason": reason,
                "session_mae_median_n": float(position_sessions["mae_n"].median()),
                "session_mae_max_n": float(position_sessions["mae_n"].max()),
                "session_slope_median": float(
                    position_sessions["linear_slope_pred_vs_px6d"].median()
                ),
                "session_correlation_median": float(
                    position_sessions["pearson_r"].median()
                ),
                **metrics,
            }
        )
    positions = pd.DataFrame(position_rows)
    positions["position_id"] = pd.Categorical(
        positions["position_id"], POSITION_ORDER, ordered=True
    )
    return sessions.sort_values(["position_id", "group_id"]), positions.sort_values(
        "position_id"
    )


def build_stable_plateau_tables(
    predictions: pd.DataFrame,
    *,
    minimum_force_n: float = 0.10,
    maximum_force_n: float = 5.0,
    maximum_force_speed_n_per_sec: float = 0.20,
    force_bin_width_n: float = 0.50,
    minimum_frames_per_plateau: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize stable force plateaus for paper-oriented linear fitting.

    Transition frames are rejected from the force derivative.  Each retained
    row is a session-local force-bin mean, so long recordings do not dominate
    a position simply because they contain more frames.
    """

    required = {
        "position_id",
        "group_id",
        "sample_index",
        "elapsed_time_sec",
        "true_force_n",
        "gated_force_n",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("stable plateau predictions missing: " + ", ".join(missing))
    width = float(force_bin_width_n)
    if width <= 0:
        raise ValueError("force_bin_width_n must be positive")
    edges = np.arange(0.0, float(maximum_force_n) + width + 1.0e-12, width)
    rows: list[dict[str, Any]] = []
    for (position_id, group_id), group in predictions.groupby(
        ["position_id", "group_id"], sort=False
    ):
        group = group.sort_values("sample_index", kind="stable")
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
        stable = (
            np.isfinite(reference)
            & np.isfinite(estimate)
            & (reference >= float(minimum_force_n))
            & (reference <= float(maximum_force_n))
            & (speed <= float(maximum_force_speed_n_per_sec))
        )
        bin_index = np.digitize(reference, edges, right=False) - 1
        for index in range(len(edges) - 1):
            selected = stable & (bin_index == index)
            if int(np.sum(selected)) < int(minimum_frames_per_plateau):
                continue
            rows.append(
                {
                    "position_id": str(position_id),
                    "group_id": str(group_id),
                    "force_bin_low_n": float(edges[index]),
                    "force_bin_high_n": float(edges[index + 1]),
                    "stable_frame_count": int(np.sum(selected)),
                    "px6d_mean_n": float(np.mean(reference[selected])),
                    "optical_mean_n": float(np.mean(estimate[selected])),
                    "plateau_bias_n": float(
                        np.mean(estimate[selected]) - np.mean(reference[selected])
                    ),
                }
            )
    plateaus = pd.DataFrame(rows)
    if plateaus.empty:
        return plateaus, pd.DataFrame()

    summaries: list[dict[str, Any]] = []
    for position_id in POSITION_ORDER:
        group = plateaus[plateaus["position_id"] == position_id]
        if group.empty:
            continue
        reference = group["px6d_mean_n"].to_numpy(dtype=float)
        estimate = group["optical_mean_n"].to_numpy(dtype=float)
        slope = intercept = correlation = r2 = math.nan
        if len(reference) >= 3 and float(np.std(reference)) > 1.0e-12:
            slope, intercept = np.polyfit(reference, estimate, 1)
            r2_value = _safe_r2(reference, estimate)
            correlation_value = _safe_correlation(reference, estimate)
            r2 = math.nan if r2_value is None else float(r2_value)
            correlation = (
                math.nan if correlation_value is None else float(correlation_value)
            )
        summaries.append(
            {
                "position_id": position_id,
                "session_count": int(group["group_id"].nunique()),
                "plateau_count": int(len(group)),
                "plateau_mae_n": float(np.mean(np.abs(estimate - reference))),
                "plateau_r2": r2,
                "plateau_pearson_r": correlation,
                "plateau_slope": float(slope),
                "plateau_intercept_n": float(intercept),
            }
        )
    return plateaus, pd.DataFrame(summaries)


def representative_sessions(
    sessions: pd.DataFrame,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for position_id in POSITION_ORDER:
        rows = sessions[sessions["position_id"] == position_id]
        if rows.empty:
            continue
        target = float(rows["mae_n"].median())
        index = (rows["mae_n"] - target).abs().idxmin()
        selected[position_id] = str(rows.loc[index, "group_id"])
    return selected


def plot_representative_traces(
    predictions: pd.DataFrame,
    sessions: pd.DataFrame,
    output_path: Path,
) -> None:
    representatives = representative_sessions(sessions)
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 10.2), sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER):
        group_id = representatives.get(position_id)
        group = predictions[
            (predictions["position_id"] == position_id)
            & (predictions["group_id"] == group_id)
        ]
        metric_row = sessions[
            (sessions["position_id"] == position_id)
            & (sessions["group_id"] == group_id)
        ]
        if group.empty:
            axis.set_axis_off()
            continue
        elapsed = group["elapsed_time_sec"].to_numpy(dtype=float)
        axis.plot(
            elapsed,
            group["true_force_n"],
            color="#0072B2",
            linewidth=2.0,
            label="PX6D Fz",
        )
        axis.plot(
            elapsed,
            group["gated_force_n"],
            color="#D55E00",
            linewidth=1.8,
            label="Grouped OOF optical Fz",
        )
        row = metric_row.iloc[0]
        axis.set_title(
            f"{position_id} | MAE {row['mae_n']:.2f} N | r {row['pearson_r']:.2f}",
            fontsize=10,
        )
        axis.set_ylim(-0.15, 5.25)
        axis.grid(alpha=0.17)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Nine-point force-curve consistency | representative independent sessions",
        fontsize=15,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)


def plot_position_regression(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12.2, 11.0), sharex=True, sharey=True)
    for axis, position_id in zip(axes.flat, POSITION_ORDER):
        group = predictions[predictions["position_id"] == position_id]
        x = group["true_force_n"].to_numpy(dtype=float)
        y = group["gated_force_n"].to_numpy(dtype=float)
        axis.hexbin(
            x,
            y,
            gridsize=36,
            extent=(0, 5, 0, 5),
            mincnt=1,
            cmap="Blues",
            linewidths=0,
        )
        axis.plot([0, 5], [0, 5], color="#D55E00", linewidth=1.3, linestyle="--")
        if len(x) > 2 and np.std(x) > 1.0e-12:
            slope, intercept = np.polyfit(x, y, 1)
            axis.plot(
                [0, 5],
                [intercept, intercept + 5 * slope],
                color="#009E73",
                linewidth=1.4,
            )
        axis.set_title(position_id)
        axis.set_xlim(0, 5)
        axis.set_ylim(0, 5)
        axis.grid(alpha=0.12)
        axis.set_xlabel("PX6D Fz (N)")
        axis.set_ylabel("Optical estimate (N)")
    figure.suptitle("Grouped OOF optical-force calibration by tactile point", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)


__all__ = [
    "POSITION_ORDER",
    "build_force_consistency_tables",
    "build_stable_plateau_tables",
    "force_curve_metrics",
    "infer_position_id",
    "load_grouped_force_predictions",
    "plot_position_regression",
    "plot_representative_traces",
    "representative_sessions",
]
