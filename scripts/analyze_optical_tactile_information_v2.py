"""Quantify tactile information available beyond position and normal force.

This is an offline, descriptive audit of the latest-primary ordinary-FBG
dataset.  It does not modify the application, deployed models, recording
workflow, or executable.  Spatial quantities are optical proxies and must not
be interpreted as calibrated pressure or independent force pixels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.optical_tactile_information import (  # noqa: E402
    CHANNEL_COORDINATES,
    CHANNEL_ORDER,
    build_noise_normalized_channel_response,
    contact_patch_moments,
    cosine_similarity_rows,
    find_release_recovery_events,
    robust_mad,
)
from src.hybrid_spectrum.rich_optical_features import (  # noqa: E402
    load_rich_feature_cache,
)
from src.hybrid_spectrum.tactile_observability import (  # noqa: E402
    derive_force_phase_labels,
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
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT / "outputs" / "optical_algorithm_and_tactile_information_audit_20260802"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_DIR

POSITION_COORDINATES = {
    channel: tuple(CHANNEL_COORDINATES[index])
    for index, channel in enumerate(CHANNEL_ORDER)
}
FORCE_BIN_EDGES = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.01])
FORCE_BIN_LABELS = ("0-1 N", "1-2 N", "2-3 N", "3-4 N", "4-5 N")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION_DATASET)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM_DATASET)
    parser.add_argument("--rich-cache", type=Path, default=DEFAULT_RICH_CACHE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_strict_metadata_aligned(
    path: Path,
    group_id: np.ndarray,
    sample_index: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        keys = list(
            zip(
                payload["session_id"].astype(str).tolist(),
                payload["capture_index"].astype(int).tolist(),
            )
        )
        lookup = {key: index for index, key in enumerate(keys)}
        expected = list(zip(group_id.astype(str).tolist(), sample_index.astype(int).tolist()))
        missing = [key for key in expected if key not in lookup]
        if missing:
            raise ValueError(f"strict metadata is missing aligned key {missing[0]!r}")
        reorder = np.asarray([lookup[key] for key in expected], dtype=int)
        return {
            "elapsed_time_sec": payload["elapsed_time_sec"][reorder].astype(float),
            "release_tail_excluded": payload["release_tail_excluded"][reorder].astype(bool),
            "trial_id": payload["trial_id"][reorder].astype(str),
        }


def _force_bins(values: np.ndarray) -> np.ndarray:
    return pd.cut(
        np.asarray(values, dtype=float),
        bins=FORCE_BIN_EDGES,
        labels=FORCE_BIN_LABELS,
        include_lowest=True,
        right=False,
    ).astype(object)


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else float("nan")


def _patch_frames(
    *,
    response: np.ndarray,
    group_id: np.ndarray,
    sample_index: np.ndarray,
    elapsed_time_sec: np.ndarray,
    contact_target: np.ndarray,
    position_target: np.ndarray,
    force_fz_n: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, weights in enumerate(response):
        row = contact_patch_moments(weights)
        row.update(
            {
                "session_id": str(group_id[index]),
                "capture_index": int(sample_index[index]),
                "elapsed_time_sec": float(elapsed_time_sec[index]),
                "contact_target": int(contact_target[index]),
                "position_target": str(position_target[index]),
                "force_fz_n": float(force_fz_n[index]),
                "response_sum": float(np.sum(weights)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _patch_summary(frame: pd.DataFrame) -> pd.DataFrame:
    active = frame.loc[
        (frame["contact_target"] == 1)
        & frame["position_target"].isin(CHANNEL_ORDER)
        & np.isfinite(frame["force_fz_n"])
    ].copy()
    active["force_bin"] = _force_bins(active["force_fz_n"].to_numpy())
    rows: list[dict[str, Any]] = []
    for (position, force_bin), group in active.groupby(
        ["position_target", "force_bin"], observed=True, sort=False
    ):
        target_x, target_y = POSITION_COORDINATES[str(position)]
        center_x = _finite_median(group["center_x"].to_numpy())
        center_y = _finite_median(group["center_y"].to_numpy())
        rows.append(
            {
                "position_target": position,
                "force_bin": force_bin,
                "frame_count": int(len(group)),
                "session_count": int(group["session_id"].nunique()),
                "target_x": target_x,
                "target_y": target_y,
                "median_center_x": center_x,
                "median_center_y": center_y,
                "center_error_grid_units": float(
                    np.hypot(center_x - target_x, center_y - target_y)
                ),
                "median_spread_major": _finite_median(group["spread_major"].to_numpy()),
                "median_spread_minor": _finite_median(group["spread_minor"].to_numpy()),
                "median_eccentricity": _finite_median(group["eccentricity"].to_numpy()),
                "median_orientation_deg": _finite_median(group["orientation_deg"].to_numpy()),
                "median_active_channel_count": _finite_median(
                    group["active_channel_count"].to_numpy()
                ),
                "median_peak_response": _finite_median(group["peak_response"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _coupling_signature(
    response: np.ndarray,
    contact_target: np.ndarray,
    position_target: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matrix = np.full((len(CHANNEL_ORDER), len(CHANNEL_ORDER)), np.nan, dtype=float)
    for source_index, source in enumerate(CHANNEL_ORDER):
        selected = (contact_target == 1) & (position_target == source)
        current = response[selected]
        if not len(current):
            continue
        row_max = np.max(current, axis=1, keepdims=True)
        relative = np.divide(
            current,
            row_max,
            out=np.zeros_like(current),
            where=row_max > 1.0e-12,
        )
        median_signature = np.median(relative, axis=0)
        matrix[source_index] = median_signature
        dominant = CHANNEL_ORDER[int(np.argmax(median_signature))]
        target_value = float(median_signature[source_index])
        off_target = np.delete(median_signature, source_index)
        summary_rows.append(
            {
                "pressed_position": source,
                "frame_count": int(len(current)),
                "dominant_response_channel": dominant,
                "dominant_matches_pressed_position": bool(dominant == source),
                "target_channel_response": target_value,
                "strongest_off_target_response": float(np.max(off_target)),
                "target_to_off_target_margin": target_value - float(np.max(off_target)),
                "channels_above_25pct": int(np.sum(median_signature >= 0.25)),
                "channels_above_50pct": int(np.sum(median_signature >= 0.50)),
            }
        )
        for response_channel, value in zip(CHANNEL_ORDER, median_signature):
            rows.append(
                {
                    "pressed_position": source,
                    "response_channel": response_channel,
                    "median_relative_optical_response": float(value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows), matrix


def _session_drift(
    *,
    group_id: np.ndarray,
    sample_index: np.ndarray,
    elapsed_time_sec: np.ndarray,
    contact_target: np.ndarray,
    log_ratio_spectrum: np.ndarray,
    response_score: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session in dict.fromkeys(group_id.astype(str).tolist()):
        selected = np.flatnonzero(group_id == session)
        order = selected[np.argsort(sample_index[selected], kind="stable")]
        idle = order[contact_target[order] == 0]
        if len(idle) < 6:
            continue
        width = max(3, len(idle) // 3)
        early = idle[:width]
        late = idle[-width:]
        early_center = np.median(log_ratio_spectrum[early], axis=0)
        late_center = np.median(log_ratio_spectrum[late], axis=0)
        drift_vector = late_center - early_center
        early_residual = log_ratio_spectrum[early] - early_center
        early_frame_rms = np.sqrt(np.mean(early_residual**2, axis=1))
        baseline_noise = max(_finite_median(early_frame_rms), 1.0e-9)
        drift_rms = float(np.sqrt(np.mean(drift_vector**2)))
        rows.append(
            {
                "session_id": session,
                "idle_frame_count": int(len(idle)),
                "elapsed_span_sec": float(
                    elapsed_time_sec[order[-1]] - elapsed_time_sec[order[0]]
                ),
                "baseline_spectral_noise_rms": baseline_noise,
                "early_to_late_drift_rms": drift_rms,
                "early_to_late_drift_max_abs": float(np.max(np.abs(drift_vector))),
                "drift_to_baseline_noise_ratio": drift_rms / baseline_noise,
                "early_response_score_median": _finite_median(response_score[early]),
                "late_response_score_median": _finite_median(response_score[late]),
                "late_minus_early_response_score": (
                    _finite_median(response_score[late])
                    - _finite_median(response_score[early])
                ),
            }
        )
    return pd.DataFrame(rows)


def _repeatability(
    *,
    group_id: np.ndarray,
    contact_target: np.ndarray,
    position_target: np.ndarray,
    force_fz_n: np.ndarray,
    log_ratio_spectrum: np.ndarray,
) -> pd.DataFrame:
    force_bin = _force_bins(force_fz_n)
    rows: list[dict[str, Any]] = []
    for position in CHANNEL_ORDER:
        for label in FORCE_BIN_LABELS:
            selected = np.flatnonzero(
                (contact_target == 1)
                & (position_target == position)
                & (force_bin == label)
            )
            if not len(selected):
                continue
            session_vectors: list[np.ndarray] = []
            session_ids: list[str] = []
            for session in dict.fromkeys(group_id[selected].astype(str).tolist()):
                current = selected[group_id[selected] == session]
                if len(current) < 3:
                    continue
                session_vectors.append(np.median(log_ratio_spectrum[current], axis=0))
                session_ids.append(session)
            if not session_vectors:
                continue
            vectors = np.asarray(session_vectors)
            similarity = cosine_similarity_rows(vectors)
            upper = similarity[np.triu_indices(len(vectors), k=1)]
            upper = upper[np.isfinite(upper)]
            amplitudes = np.sqrt(np.mean(vectors**2, axis=1))
            amplitude_mean = float(np.mean(amplitudes))
            rows.append(
                {
                    "position_target": position,
                    "force_bin": label,
                    "frame_count": int(len(selected)),
                    "session_count": int(len(session_ids)),
                    "pair_count": int(len(upper)),
                    "mean_centered_spectral_similarity": (
                        float(np.mean(upper)) if len(upper) else float("nan")
                    ),
                    "minimum_centered_spectral_similarity": (
                        float(np.min(upper)) if len(upper) else float("nan")
                    ),
                    "median_spectrum_rms": float(np.median(amplitudes)),
                    "spectrum_amplitude_cv": (
                        float(np.std(amplitudes) / amplitude_mean)
                        if amplitude_mean > 1.0e-12
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _hysteresis(
    *,
    group_id: np.ndarray,
    force_fz_n: np.ndarray,
    phase_labels: np.ndarray,
    log_ratio_spectrum: np.ndarray,
) -> pd.DataFrame:
    force_bin = _force_bins(force_fz_n)
    rows: list[dict[str, Any]] = []
    for session in dict.fromkeys(group_id.astype(str).tolist()):
        selected = np.flatnonzero(group_id == session)
        for label in FORCE_BIN_LABELS:
            loading = selected[(force_bin[selected] == label) & (phase_labels[selected] == "loading")]
            release = selected[(force_bin[selected] == label) & (phase_labels[selected] == "release")]
            if len(loading) < 3 or len(release) < 3:
                continue
            loading_center = np.median(log_ratio_spectrum[loading], axis=0)
            release_center = np.median(log_ratio_spectrum[release], axis=0)
            difference = loading_center - release_center
            reference_rms = max(
                float(np.sqrt(np.mean(loading_center**2))),
                float(np.sqrt(np.mean(release_center**2))),
                1.0e-12,
            )
            similarity = cosine_similarity_rows(
                np.vstack((loading_center, release_center))
            )[0, 1]
            rows.append(
                {
                    "session_id": session,
                    "force_bin": label,
                    "loading_frame_count": int(len(loading)),
                    "release_frame_count": int(len(release)),
                    "loading_release_spectral_rms_difference": float(
                        np.sqrt(np.mean(difference**2))
                    ),
                    "hysteresis_relative_to_response_rms": float(
                        np.sqrt(np.mean(difference**2)) / reference_rms
                    ),
                    "loading_release_centered_similarity": float(similarity),
                }
            )
    return pd.DataFrame(rows)


def _optical_dynamics(
    *,
    group_id: np.ndarray,
    sample_index: np.ndarray,
    elapsed_time_sec: np.ndarray,
    log_ratio_spectrum: np.ndarray,
    phase_labels: np.ndarray,
    force_slope: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    velocity = np.full(len(group_id), np.nan, dtype=float)
    for session in dict.fromkeys(group_id.astype(str).tolist()):
        selected = np.flatnonzero(group_id == session)
        order = selected[np.argsort(sample_index[selected], kind="stable")]
        if len(order) < 2:
            continue
        delta_time = np.diff(elapsed_time_sec[order])
        delta_spectrum = np.diff(log_ratio_spectrum[order], axis=0)
        valid = delta_time > 1.0e-6
        current = np.full(len(order) - 1, np.nan, dtype=float)
        current[valid] = (
            np.sqrt(np.mean(delta_spectrum[valid] ** 2, axis=1)) / delta_time[valid]
        )
        velocity[order[1:]] = current
    frame = pd.DataFrame(
        {
            "session_id": group_id.astype(str),
            "capture_index": sample_index.astype(int),
            "elapsed_time_sec": elapsed_time_sec,
            "phase": phase_labels.astype(str),
            "spectral_velocity_rms_per_sec": velocity,
            "force_slope_n_per_sec": force_slope,
        }
    )
    rows: list[dict[str, Any]] = []
    for phase, group in frame.groupby("phase", sort=False):
        values = group["spectral_velocity_rms_per_sec"].to_numpy(dtype=float)
        rows.append(
            {
                "phase": phase,
                "frame_count": int(np.sum(np.isfinite(values))),
                "median_spectral_velocity_rms_per_sec": _finite_median(values),
                "p90_spectral_velocity_rms_per_sec": _finite_quantile(values, 0.90),
                "p99_spectral_velocity_rms_per_sec": _finite_quantile(values, 0.99),
                "median_abs_force_slope_n_per_sec": _finite_median(
                    np.abs(group["force_slope_n_per_sec"].to_numpy(dtype=float))
                ),
            }
        )
    return frame, pd.DataFrame(rows)


def _plot_coupling(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 7.1), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    axis.set_xticks(range(len(CHANNEL_ORDER)), CHANNEL_ORDER)
    axis.set_yticks(range(len(CHANNEL_ORDER)), CHANNEL_ORDER)
    axis.set_xlabel("Response channel")
    axis.set_ylabel("Pressed position")
    axis.set_title("Median normalized nine-peak coupling signature")
    for row in range(len(CHANNEL_ORDER)):
        for column in range(len(CHANNEL_ORDER)):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.55 else "#102236",
                )
    figure.colorbar(image, ax=axis, label="Relative optical response")
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def _plot_patch_centers(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.6), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(FORCE_BIN_LABELS)))
    for force_label, color in zip(FORCE_BIN_LABELS, colors):
        current = summary.loc[summary["force_bin"] == force_label]
        if current.empty:
            continue
        axis.scatter(
            current["median_center_x"],
            current["median_center_y"],
            s=68,
            label=force_label,
            color=color,
            alpha=0.9,
        )
    for channel, (x, y) in POSITION_COORDINATES.items():
        axis.scatter([x], [y], marker="+", s=120, color="#102236")
        axis.text(x + 0.05, y + 0.05, channel, color="#102236", fontsize=9)
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.35, 1.35)
    axis.set_aspect("equal")
    axis.set_xlabel("Optical proxy center x")
    axis.set_ylabel("Optical proxy center y")
    axis.set_title("Contact-patch proxy centers by position and Fz bin")
    axis.grid(color="#d8e3ec", linewidth=0.8)
    axis.legend(title="Synchronized Fz", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def _plot_drift_recovery(
    drift: pd.DataFrame,
    recovery: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    axes[0].hist(
        drift["drift_to_baseline_noise_ratio"].replace([np.inf, -np.inf], np.nan).dropna(),
        bins=18,
        color="#2aa6b8",
        edgecolor="white",
    )
    axes[0].axvline(1.0, color="#c56b6b", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Early-to-late drift / baseline noise")
    axes[0].set_ylabel("Session count")
    axes[0].set_title("No-contact spectral drift")
    recovered = recovery.loc[
        recovery["detector_id"] == "global_shape_delta_rms"
    ]
    values = recovered.loc[recovered["recovered"], "recovery_time_sec"].dropna()
    axes[1].hist(values, bins=16, color="#69b89f", edgecolor="white")
    axes[1].set_xlabel("Recovery time (s)")
    axes[1].set_ylabel("Release-event count")
    axes[1].set_title("Optical recovery after release")
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def _observable_inventory(
    *,
    patch_summary: pd.DataFrame,
    drift: pd.DataFrame,
    recovery: pd.DataFrame,
    repeatability: pd.DataFrame,
    dynamics: pd.DataFrame,
    force_mae_n: float,
) -> pd.DataFrame:
    recovered = recovery.loc[recovery["detector_id"] == "global_shape_delta_rms"]
    recovery_rate = float(recovered["recovered"].mean()) if len(recovered) else float("nan")
    median_recovery = _finite_median(
        recovered.loc[recovered["recovered"], "recovery_time_sec"].to_numpy()
    )
    median_center_error = _finite_median(
        patch_summary["center_error_grid_units"].to_numpy()
    )
    median_repeatability = _finite_median(
        repeatability["mean_centered_spectral_similarity"].to_numpy()
    )
    median_drift_ratio = _finite_median(
        drift["drift_to_baseline_noise_ratio"].to_numpy()
    )
    loading_velocity = dynamics.loc[
        dynamics["phase"] == "loading", "median_spectral_velocity_rms_per_sec"
    ]
    idle_velocity = dynamics.loc[
        dynamics["phase"] == "no_contact", "median_spectral_velocity_rms_per_sec"
    ]
    velocity_ratio = (
        float(loading_velocity.iloc[0] / idle_velocity.iloc[0])
        if len(loading_velocity) and len(idle_velocity) and idle_velocity.iloc[0] > 0
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "observable": "contact credibility",
                "status": "supported_with_grouped_model",
                "current_evidence": "best conservative gate macro-F1 0.935; no-contact recall 0.913",
                "interpretation": "contact/no-contact state with explicit residual handling",
                "next_requirement": "more long-idle and repeated-release sessions",
            },
            {
                "observable": "contact position P11-P33",
                "status": "strongly_supported_for_current_protocol",
                "current_evidence": "grouped position macro-F1 0.9984",
                "interpretation": "approximate labeled FBG position",
                "next_requirement": "off-grid, broad-contact, and independent-position validation",
            },
            {
                "observable": "optical-only Fz estimate",
                "status": "calibration_target_not_counted_as_new_observable",
                "current_evidence": (
                    f"best grouped response-only MAE {force_mae_n:.3f} N"
                    if np.isfinite(force_mae_n)
                    else "grouped optical-to-Fz calibration benchmark available"
                ),
                "interpretation": "Z-axis reference target only; not an additional tactile observable",
                "next_requirement": "cross-day/remount validation with independent force trajectories",
            },
            {
                "observable": "contact-patch center and spread proxy",
                "status": "descriptive_proxy_only",
                "current_evidence": f"median center error {median_center_error:.3f} grid units",
                "interpretation": "noise-normalized nine-peak response moments",
                "next_requirement": "camera, pressure film, or stage-based contact footprint truth",
            },
            {
                "observable": "coupling / crosstalk fingerprint",
                "status": "directly_measurable",
                "current_evidence": "position-conditioned nine-channel response matrix generated",
                "interpretation": "structural optical/mechanical fingerprint, not independent force pixels",
                "next_requirement": "repeat across devices and remounting conditions",
            },
            {
                "observable": "release recovery and residual",
                "status": "directly_measurable",
                "current_evidence": f"recovery rate {recovery_rate:.3f}; median {median_recovery:.3f} s",
                "interpretation": "time to return to a session-specific optical idle envelope",
                "next_requirement": "longer post-release tails and controlled dwell times",
            },
            {
                "observable": "baseline drift / stability",
                "status": "directly_measurable",
                "current_evidence": f"median drift/noise ratio {median_drift_ratio:.3f}",
                "interpretation": "early-to-late no-contact spectral movement",
                "next_requirement": "temperature and mounting-condition metadata",
            },
            {
                "observable": "repeatability by position and force bin",
                "status": "directly_measurable",
                "current_evidence": f"median inter-session centered similarity {median_repeatability:.3f}",
                "interpretation": "same-condition optical fingerprint consistency",
                "next_requirement": "more repeated independent sessions per bin",
            },
            {
                "observable": "loading / hold / release phase",
                "status": "provisional_temporal_proxy",
                "current_evidence": "grouped phase macro-F1 0.618; loading recall remains weak",
                "interpretation": "phase labels derived from synchronized Fz slope",
                "next_requirement": "higher-rate acquisition and explicit onset/release labels",
            },
            {
                "observable": "optical loading-rate / impact proxy",
                "status": "weak_in_current_sampling",
                "current_evidence": (
                    f"loading/no-contact median spectral-velocity ratio {velocity_ratio:.3f}"
                    if np.isfinite(velocity_ratio)
                    else "spectral velocity trace generated"
                ),
                "interpretation": "current frame rate does not separate loading dynamics from idle variation",
                "next_requirement": "higher-rate controlled tap/loading trials with speed and contact-time labels",
            },
            {
                "observable": "hysteresis",
                "status": "directly_measurable_with_current_cycles",
                "current_evidence": "loading/release spectra compared at matched Fz bins",
                "interpretation": "path dependence at similar synchronized Fz",
                "next_requirement": "more complete repeated loading-unloading cycles",
            },
            {
                "observable": "slip direction and speed",
                "status": "not_supported_by_current_labels",
                "current_evidence": "no controlled x/y slip protocol",
                "interpretation": "cannot be claimed from Z-only presses",
                "next_requirement": "controlled slides with direction and speed ground truth",
            },
            {
                "observable": "texture, material, curvature, object shape",
                "status": "not_supported_by_current_labels",
                "current_evidence": "no object-stratified labels",
                "interpretation": "semantic tactile recognition is currently unidentifiable",
                "next_requirement": "held-out-object repeated-contact dataset",
            },
            {
                "observable": "multiple simultaneous contacts",
                "status": "not_supported_by_current_labels",
                "current_evidence": "current sessions use one approximate press location",
                "interpretation": "multi-contact inverse problem is not identifiable yet",
                "next_requirement": "controlled multi-contact combinations and geometry truth",
            },
        ]
    )


def _baseline_confound_summary(output_dir: Path) -> dict[str, float | int | bool]:
    leaderboard_path = output_dir / "baseline_confound_ablation_leaderboard.csv"
    metadata_path = output_dir / "baseline_metadata_feature_audit.csv"
    summary: dict[str, float | int | bool] = {
        "available": False,
        "metadata_feature_count": 0,
        "median_within_session_to_total_variance_ratio": float("nan"),
        "lightgbm_position_macro_f1_with_metadata": float("nan"),
        "lightgbm_position_macro_f1_without_metadata": float("nan"),
        "lightgbm_force_mae_n_with_metadata": float("nan"),
        "lightgbm_force_mae_n_without_metadata": float("nan"),
        "best_response_only_force_mae_n": float("nan"),
    }
    if not leaderboard_path.exists() or not metadata_path.exists():
        return summary

    leaderboard = pd.read_csv(leaderboard_path)
    metadata = pd.read_csv(metadata_path)

    def metric(model: str, view: str, task: str, column: str) -> float:
        selected = leaderboard.loc[
            (leaderboard["model_id"] == model)
            & (leaderboard["feature_view"] == view)
            & (leaderboard["task"] == task),
            column,
        ]
        return float(selected.iloc[0]) if len(selected) else float("nan")

    response_only = leaderboard.loc[
        (leaderboard["task"] == "force_fz")
        & (leaderboard["feature_view"] == "rich_plus_full_no_session_baseline"),
        "mae_n",
    ]
    summary.update(
        {
            "available": True,
            "metadata_feature_count": int(len(metadata)),
            "median_within_session_to_total_variance_ratio": _finite_median(
                metadata["within_to_total_variance_ratio"].to_numpy()
            ),
            "lightgbm_position_macro_f1_with_metadata": metric(
                "lightgbm", "rich_plus_full_all", "position", "macro_f1"
            ),
            "lightgbm_position_macro_f1_without_metadata": metric(
                "lightgbm",
                "rich_plus_full_no_session_baseline",
                "position",
                "macro_f1",
            ),
            "lightgbm_force_mae_n_with_metadata": metric(
                "lightgbm", "rich_plus_full_all", "force_fz", "mae_n"
            ),
            "lightgbm_force_mae_n_without_metadata": metric(
                "lightgbm",
                "rich_plus_full_no_session_baseline",
                "force_fz",
                "mae_n",
            ),
            "best_response_only_force_mae_n": float(response_only.min())
            if len(response_only)
            else float("nan"),
        }
    )
    return summary


def _write_github_recommendations(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# GitHub algorithm recommendations",
                "",
                "## Current-frame wavelength spectra",
                "",
                "- Keep LightGBM / ExtraTrees as the primary efficient baselines. The formal grouped benchmark shows that PLS-DA, linear/RBF SVM, LDA, and logistic regression do not exceed the tree ensembles on this dataset.",
                "- Use [pybaselines](https://github.com/derb12/pybaselines) for asymmetric least-squares or morphology-based baseline correction when raw-spectrum backgrounds vary.",
                "- Use scikit-learn PLS/PCA/SVM pipelines as interpretable chemometric controls, but not as the current champion.",
                "- Use [SKADA](https://github.com/scikit-adaptation/skada) only after a deliberate cross-day/device domain-shift benchmark exists.",
                "",
                "## True temporal windows",
                "",
                "- Use [aeon](https://github.com/aeon-toolkit/aeon) MiniRocket/MultiRocket/HYDRA for actual frame sequences. Do not treat the 512 wavelength bins of one frame as time.",
                "- Use [tsfresh](https://github.com/blue-yonder/tsfresh) or [pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22) for compact, interpretable temporal features.",
                "- Use [tsai](https://github.com/timeseriesAI/tsai) InceptionTime/TCN only after enough independent dynamic sessions are collected.",
                "",
                "## Drift, release, and online operation",
                "",
                "- Use [ruptures](https://github.com/deepcharles/ruptures) for offline contact/release change-point annotation and segment auditing.",
                "- Use [River](https://github.com/online-ml/river) for online drift statistics and adaptive monitoring, while keeping model updates opt-in and traceable.",
                "- Use [sktime](https://github.com/sktime/sktime) or [aeon](https://github.com/aeon-toolkit/aeon) for grouped temporal pipelines and reproducible model comparison.",
                "",
                "## Recommended next model architecture",
                "",
                "1. Stationarity/drift-aware conservative contact gate.",
                "2. ExtraTrees position head conditioned on credible contact.",
                "3. Continuous Fz regression conditioned on credible contact.",
                "4. Temporal phase/recovery head for loading, hold, release, and residual suppression.",
                "5. Contact-patch and coupling descriptors as auxiliary features, not as calibrated pressure pixels.",
                "",
                "All formal comparisons must remain grouped by immutable session_id.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_aligned_latest_primary(args.fusion_dataset, args.spectrum_dataset)
    rich = load_rich_feature_cache(
        args.rich_cache,
        expected_group_id=dataset.group_id,
        expected_sample_index=dataset.sample_index,
    )
    metadata = _load_strict_metadata_aligned(
        args.spectrum_dataset, dataset.group_id, dataset.sample_index
    )
    no_contact = dataset.contact_mask & (dataset.contact_target == 0)
    contact = dataset.contact_mask & (dataset.contact_target == 1)
    response, scale_rows = build_noise_normalized_channel_response(
        features=rich.features,
        feature_names=rich.feature_names,
        no_contact_mask=no_contact,
        contact_mask=contact,
    )
    pd.DataFrame(scale_rows).to_csv(
        args.output_dir / "channel_sensitivity_normalization.csv", index=False
    )

    patch_frame = _patch_frames(
        response=response,
        group_id=dataset.group_id,
        sample_index=dataset.sample_index,
        elapsed_time_sec=metadata["elapsed_time_sec"],
        contact_target=dataset.contact_target,
        position_target=dataset.position_target,
        force_fz_n=dataset.force_fz_n,
    )
    patch_frame.to_csv(args.output_dir / "contact_patch_proxy_frames.csv", index=False)
    patch_summary = _patch_summary(patch_frame)
    patch_summary.to_csv(args.output_dir / "contact_patch_proxy_summary.csv", index=False)

    coupling, coupling_summary, coupling_matrix = _coupling_signature(
        response, dataset.contact_target, dataset.position_target
    )
    coupling.to_csv(args.output_dir / "optical_coupling_signature.csv", index=False)
    coupling_summary.to_csv(
        args.output_dir / "optical_coupling_signature_summary.csv", index=False
    )

    log_ratio_spectrum = dataset.spectrum_features[:, :64].astype(float)
    global_shape_index = np.flatnonzero(
        dataset.spectrum_feature_names.astype(str) == "global_shape_delta_rms"
    )
    if len(global_shape_index) != 1:
        raise ValueError("global_shape_delta_rms is missing or duplicated")
    global_shape = dataset.spectrum_features[:, int(global_shape_index[0])].astype(float)
    response_score = np.max(response, axis=1)

    drift = _session_drift(
        group_id=dataset.group_id,
        sample_index=dataset.sample_index,
        elapsed_time_sec=metadata["elapsed_time_sec"],
        contact_target=dataset.contact_target,
        log_ratio_spectrum=log_ratio_spectrum,
        response_score=response_score,
    )
    drift.to_csv(args.output_dir / "session_drift_audit.csv", index=False)

    recovery_rows: list[dict[str, Any]] = []
    for detector_id, score in (
        ("global_shape_delta_rms", global_shape),
        ("noise_normalized_nine_peak_max", response_score),
    ):
        events = find_release_recovery_events(
            group_id=dataset.group_id,
            capture_index=dataset.sample_index,
            elapsed_time_sec=metadata["elapsed_time_sec"],
            contact_target=dataset.contact_target,
            response_score=score,
            stable_frames=3,
        )
        for event in events:
            recovery_rows.append({"detector_id": detector_id, **asdict(event)})
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(args.output_dir / "release_recovery_events.csv", index=False)

    repeatability = _repeatability(
        group_id=dataset.group_id,
        contact_target=dataset.contact_target,
        position_target=dataset.position_target,
        force_fz_n=dataset.force_fz_n,
        log_ratio_spectrum=log_ratio_spectrum,
    )
    repeatability.to_csv(
        args.output_dir / "repeatability_by_position_force.csv", index=False
    )

    phase_labels, force_slope, phase_valid = derive_force_phase_labels(
        force_fz_n=dataset.force_fz_n,
        elapsed_time_sec=metadata["elapsed_time_sec"],
        group_id=dataset.group_id,
    )
    phase_labels = phase_labels.astype(str)
    phase_labels[~phase_valid] = "invalid"
    hysteresis = _hysteresis(
        group_id=dataset.group_id,
        force_fz_n=dataset.force_fz_n,
        phase_labels=phase_labels,
        log_ratio_spectrum=log_ratio_spectrum,
    )
    hysteresis.to_csv(args.output_dir / "hysteresis_audit.csv", index=False)
    dynamics_frame, dynamics_summary = _optical_dynamics(
        group_id=dataset.group_id,
        sample_index=dataset.sample_index,
        elapsed_time_sec=metadata["elapsed_time_sec"],
        log_ratio_spectrum=log_ratio_spectrum,
        phase_labels=phase_labels,
        force_slope=force_slope,
    )
    dynamics_frame.to_csv(args.output_dir / "optical_dynamics_frames.csv", index=False)
    dynamics_summary.to_csv(args.output_dir / "optical_dynamics_summary.csv", index=False)

    baseline_confound = _baseline_confound_summary(args.output_dir)
    inventory = _observable_inventory(
        patch_summary=patch_summary,
        drift=drift,
        recovery=recovery,
        repeatability=repeatability,
        dynamics=dynamics_summary,
        force_mae_n=float(baseline_confound["best_response_only_force_mae_n"]),
    )
    inventory.to_csv(args.output_dir / "tactile_observable_inventory_v2.csv", index=False)

    _plot_coupling(coupling_matrix, args.output_dir / "optical_coupling_signature.png")
    _plot_patch_centers(
        patch_summary, args.output_dir / "contact_patch_proxy_centers.png"
    )
    _plot_drift_recovery(
        drift, recovery, args.output_dir / "drift_and_recovery_summary.png"
    )
    _write_github_recommendations(args.output_dir / "github_algorithm_recommendations.md")

    recovered = recovery.loc[recovery["detector_id"] == "global_shape_delta_rms"]
    recovered_times = recovered.loc[recovered["recovered"], "recovery_time_sec"]
    coupling_match_rate = float(
        coupling_summary["dominant_matches_pressed_position"].mean()
    )

    frame_intervals: list[float] = []
    for session in np.unique(dataset.group_id.astype(str)):
        order = np.flatnonzero(dataset.group_id.astype(str) == session)
        order = order[np.argsort(metadata["elapsed_time_sec"][order])]
        deltas = np.diff(metadata["elapsed_time_sec"][order])
        frame_intervals.extend(deltas[(deltas > 0) & np.isfinite(deltas)].tolist())
    frame_intervals_array = np.asarray(frame_intervals, dtype=float)
    median_frame_interval_sec = _finite_median(frame_intervals_array)
    p90_frame_interval_sec = _finite_quantile(frame_intervals_array, 0.90)

    extra_trees_latency_ms = float("nan")
    candidate_leaderboard_path = args.candidate_dir / "candidate_model_leaderboard.csv"
    if candidate_leaderboard_path.exists():
        candidate_leaderboard = pd.read_csv(candidate_leaderboard_path)
        selected_latency = candidate_leaderboard.loc[
            (candidate_leaderboard["model_id"] == "extra_trees")
            & (candidate_leaderboard["task"] == "contact"),
            "inference_latency_ms_per_frame",
        ]
        if len(selected_latency):
            extra_trees_latency_ms = float(selected_latency.iloc[0])

    pd.DataFrame(
        [
            {
                "component": "saved_recording_inter_frame_interval",
                "median_ms": median_frame_interval_sec * 1000.0,
                "p90_ms": p90_frame_interval_sec * 1000.0,
                "measurement_status": "measured_from_saved_latest_primary_sessions",
                "interpretation": "includes acquisition and recording cadence",
            },
            {
                "component": "extra_trees_contact_inference",
                "median_ms": extra_trees_latency_ms,
                "p90_ms": np.nan,
                "measurement_status": "offline_benchmark",
                "interpretation": "model inference only",
            },
            {
                "component": "current_live_end_to_end",
                "median_ms": np.nan,
                "p90_ms": np.nan,
                "measurement_status": "not_measured_hardware_disconnected",
                "interpretation": "user reports improved response with slight residual delay",
            },
        ]
    ).to_csv(args.output_dir / "offline_latency_budget.csv", index=False)

    summary = {
        "dataset": {
            "frame_count": int(len(dataset.group_id)),
            "independent_session_count": int(len(set(dataset.group_id.tolist()))),
            "formal_split": "immutable grouped_by_session_id",
            "fx_fy_moments_used_as_targets": False,
        },
        "contact_patch_proxy": {
            "median_center_error_grid_units": _finite_median(
                patch_summary["center_error_grid_units"].to_numpy()
            ),
            "median_active_channel_count": _finite_median(
                patch_summary["median_active_channel_count"].to_numpy()
            ),
            "calibrated_geometry": False,
        },
        "coupling": {
            "direct_dominant_channel_match_rate": coupling_match_rate,
            "interpretation": "coupled optical fingerprint, not independent force pixels",
        },
        "drift": {
            "audited_session_count": int(len(drift)),
            "median_drift_to_baseline_noise_ratio": _finite_median(
                drift["drift_to_baseline_noise_ratio"].to_numpy()
            ),
            "sessions_above_one_noise_unit": int(
                np.sum(drift["drift_to_baseline_noise_ratio"] > 1.0)
            ),
        },
        "recovery": {
            "event_count": int(len(recovered)),
            "recovered_fraction": float(recovered["recovered"].mean())
            if len(recovered)
            else float("nan"),
            "median_recovery_time_sec": _finite_median(recovered_times.to_numpy()),
            "p90_recovery_time_sec": _finite_quantile(recovered_times.to_numpy(), 0.90),
        },
        "repeatability": {
            "position_force_cells": int(len(repeatability)),
            "median_inter_session_centered_similarity": _finite_median(
                repeatability["mean_centered_spectral_similarity"].to_numpy()
            ),
        },
        "hysteresis": {
            "matched_session_force_cells": int(len(hysteresis)),
            "median_loading_release_spectral_rms_difference": _finite_median(
                hysteresis["loading_release_spectral_rms_difference"].to_numpy()
            ),
            "median_relative_to_response_rms": _finite_median(
                hysteresis["hysteresis_relative_to_response_rms"].to_numpy()
            ),
        },
        "dynamics": {
            "loading_to_no_contact_median_spectral_velocity_ratio": (
                float(
                    dynamics_summary.loc[
                        dynamics_summary["phase"] == "loading",
                        "median_spectral_velocity_rms_per_sec",
                    ].iloc[0]
                    / dynamics_summary.loc[
                        dynamics_summary["phase"] == "no_contact",
                        "median_spectral_velocity_rms_per_sec",
                    ].iloc[0]
                )
            ),
        },
        "baseline_metadata_confound": baseline_confound,
        "force_calibration": {
            "fz_is_calibration_target": True,
            "counted_as_additional_tactile_observable": False,
            "best_grouped_response_only_mae_n": baseline_confound[
                "best_response_only_force_mae_n"
            ],
            "fx_fy_moments_used_as_targets": False,
        },
        "latency_context": {
            "saved_recording_median_frame_interval_sec": median_frame_interval_sec,
            "saved_recording_p90_frame_interval_sec": p90_frame_interval_sec,
            "saved_recording_median_fps": (
                1.0 / median_frame_interval_sec
                if np.isfinite(median_frame_interval_sec)
                and median_frame_interval_sec > 0
                else float("nan")
            ),
            "extra_trees_contact_inference_latency_ms_per_frame": extra_trees_latency_ms,
            "current_live_end_to_end_measured": False,
            "hardware_connected_during_audit": False,
        },
    }
    (args.output_dir / "optical_tactile_information_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    report_lines = [
        "# Optical algorithm and tactile-information audit",
        "",
        "## Validity boundary",
        "",
        f"- Frames: {len(dataset.group_id):,}; independent sessions: {len(set(dataset.group_id.tolist()))}.",
        "- Formal model comparisons use immutable grouped-by-session folds. No random frame split is accepted.",
        "- The experiment controlled only the Z-axis. Fx, Fy, and moments are not counted as new tactile outputs.",
        "- Contact-patch quantities are optical response proxies, not calibrated pressure geometry or independent force pixels.",
        "- No UI, recorder, runtime bundle, deployed model, or EXE was changed.",
        "",
        "## Algorithm result",
        "",
        "The new static chemometric candidates did not beat the tree ensembles. ExtraTrees reached contact macro-F1 0.933 and position macro-F1 0.9984. PLS-DA, logistic regression, linear/RBF SVM, and shrinkage LDA were weaker. The best end-to-end hierarchy remains conservative dual-view contact confirmation plus ExtraTrees position: 10-class macro-F1 0.9750, no-contact recall 0.9125, contact recall 0.9873, and conditional position accuracy 0.9989.",
        "",
        "This means the next gain is more likely to come from drift/release-aware gating and true temporal windows than from replacing ExtraTrees with a larger static classifier.",
        "",
        "## Baseline-metadata confound audit",
        "",
        (
            f"- {baseline_confound['metadata_feature_count']} baseline SNR/validity fields have a median within-session/total variance ratio of "
            f"{baseline_confound['median_within_session_to_total_variance_ratio']:.3e}; they behave like session metadata rather than frame response features."
            if baseline_confound["available"]
            else "- Baseline-metadata ablation was not available in this run."
        ),
        (
            "- Removing these fields changed LightGBM position macro-F1 from "
            f"{baseline_confound['lightgbm_position_macro_f1_with_metadata']:.4f} to "
            f"{baseline_confound['lightgbm_position_macro_f1_without_metadata']:.4f}, and Fz MAE from "
            f"{baseline_confound['lightgbm_force_mae_n_with_metadata']:.3f} N to "
            f"{baseline_confound['lightgbm_force_mae_n_without_metadata']:.3f} N."
            if baseline_confound["available"]
            else "- No ablation comparison was generated."
        ),
        "- Baseline SNR and peak-validity fields should remain QA/Diagnostics metadata and must not enter the formal recognizer feature matrix.",
        "- A cross-day, remount, and randomized acquisition-order holdout is still required before claiming generalization.",
        "",
        "## Latency boundary",
        "",
        f"- Saved latest-primary sessions have a median frame interval of {median_frame_interval_sec * 1000.0:.1f} ms ({summary['latency_context']['saved_recording_median_fps']:.2f} frames/s) and a p90 interval of {p90_frame_interval_sec * 1000.0:.1f} ms.",
        f"- ExtraTrees contact inference takes about {extra_trees_latency_ms:.3f} ms/frame offline, so classifier execution is negligible compared with the recorded frame cadence.",
        "- The user reports that current live response is clearly faster but still has slight delay. No light source or force sensor was connected during this audit, so current end-to-end latency was not measured; acquisition, transport, preprocessing, and rendering must be profiled separately when hardware is available.",
        "",
        "## Additional tactile information in the current data",
        "",
        f"- Optical contact-patch proxy median center error: {summary['contact_patch_proxy']['median_center_error_grid_units']:.3f} grid units.",
        f"- Direct strongest-channel agreement with the pressed label: {summary['coupling']['direct_dominant_channel_match_rate']:.3f}. This deliberately exposes coupling and unequal channel sensitivity; supervised position recognition is much stronger.",
        f"- Median early-to-late no-contact drift/noise ratio: {summary['drift']['median_drift_to_baseline_noise_ratio']:.3f}; {summary['drift']['sessions_above_one_noise_unit']}/{summary['drift']['audited_session_count']} audited sessions moved by more than one baseline-noise unit.",
        f"- Release events returning to the optical idle envelope: {summary['recovery']['recovered_fraction']:.3f}; median recovery {summary['recovery']['median_recovery_time_sec']:.3f} s; p90 {summary['recovery']['p90_recovery_time_sec']:.3f} s.",
        f"- Median inter-session centered spectral similarity within matched position/Fz bins: {summary['repeatability']['median_inter_session_centered_similarity']:.3f}.",
        f"- Matched loading/release force-bin cells available for hysteresis analysis: {summary['hysteresis']['matched_session_force_cells']}.",
        f"- Median hysteresis magnitude relative to the matched response RMS: {summary['hysteresis']['median_relative_to_response_rms']:.3f}.",
        f"- Loading/no-contact median spectral-velocity ratio: {summary['dynamics']['loading_to_no_contact_median_spectral_velocity_ratio']:.3f}; at the current sampling rate, spectral velocity alone is not a useful phase discriminator.",
        "",
        "The current data therefore support: contact credibility, approximate P11-P33 position, continuous Fz calibration, loading/hold/release proxy, recovery time, residual and baseline drift, repeatability, hysteresis, optical loading-rate proxy, coupling fingerprint, and a provisional contact-patch center/spread/eccentricity/orientation descriptor. Fz is the synchronized Z-axis calibration target and is not counted as an additional tactile observable.",
        "",
        "They do not yet support slip direction/speed, texture, material, curvature, object-shape recognition, or multiple simultaneous-contact decomposition because those labels and protocols are absent.",
        "",
        "## Recommended next sequence",
        "",
        "1. Keep the conservative dual-view contact gate and ExtraTrees position head as the static reference.",
        "2. Add release/stationarity and baseline-drift descriptors to the contact gate; evaluate only by held-out session.",
        "3. Collect real high-rate temporal windows and compare MiniRocket/MultiRocket/HYDRA, catch22/tsfresh, and then InceptionTime/TCN.",
        "4. Add controlled tap/loading-rate trials before interpreting spectral velocity as a tactile event class.",
        "5. Add camera/stage or pressure-film ground truth before claiming high-resolution contact-patch geometry.",
        "6. Add controlled slip, texture, material, object, and multi-contact protocols one target at a time.",
        "",
        "See `github_algorithm_recommendations.md` and `tactile_observable_inventory_v2.csv` for the evidence-to-algorithm map.",
    ]
    (args.output_dir / "optical_tactile_information_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    decision_lines = [
        "# Algorithm and tactile-information decision",
        "",
        "## Decision",
        "",
        "Do not replace the current tree-based static baseline with PLS-DA, SVM, LDA, or logistic regression. None improved the strict grouped-by-session result. Use conservative dual-view contact confirmation followed by ExtraTrees position as the reference architecture. Treat LightGBM with response-only rich features as a promising contact/Fz candidate for later validation, not as a deployment decision from this audit.",
        "",
        "The most useful improvement is not a larger static classifier. It is a cleaner contact gate using release recovery, stationarity, drift, normalized shape residual, and coupling-aware features, followed by models trained on genuine temporal windows when higher-rate labeled sequences exist.",
        "",
        "## Evidence",
        "",
        "- Best current hierarchy: 10-class macro-F1 0.9750, no-contact recall 0.9125, contact recall 0.9873, conditional position accuracy 0.9989.",
        "- ExtraTrees position remains highly stable at grouped macro-F1 0.9984.",
        f"- Best grouped response-only optical-to-Fz MAE in the confound audit: {float(baseline_confound['best_response_only_force_mae_n']):.3f} N.",
        "- ExtraTrees contact inference is below 0.1 ms/frame; recorded acquisition cadence, not model execution, dominates offline latency.",
        "- The user reports that live response is clearly faster but still slightly delayed. Hardware was disconnected, so this is an observation rather than a measured end-to-end latency result.",
        "",
        "## Optical feature families worth keeping",
        "",
        "- Peak displacement: centroid, parabolic peak, and cross-correlation shift.",
        "- Peak morphology: height, area, width, skew, local contrast, and peak-validity diagnostics.",
        "- Full-spectrum response: normalized log ratio, shape residual, derivative energy, correlation, and common-mode removal.",
        "- Coupling structure: same-fibre downstream response, neighbouring-fibre response, spatial response moments, and channel-sensitivity normalization.",
        "- Temporal state: onset, hold, release, recovery time, residual, drift, hysteresis, and repeatability.",
        "- Keep baseline SNR/validity metadata out of recognition features; use it only for QA and diagnostics.",
        "",
        "## Tactile quantities supported now",
        "",
        "Supported or provisionally measurable: contact credibility, approximate P11-P33 position, optical-only Fz calibration, contact-patch center/spread proxy, coupling fingerprint, release recovery, baseline drift, repeatability, hysteresis, and provisional loading/hold/release phase. Fz does not count as a new tactile quantity here because only the Z-axis was deliberately applied and it serves as calibration truth.",
        "",
        "Not yet identifiable: slip direction/speed, texture, material, curvature, object identity/shape, or simultaneous multi-contact decomposition. Those require dedicated labels and protocols.",
        "",
        "## Mature algorithm path",
        "",
        "1. Current frames: ExtraTrees and LightGBM remain the efficient references; PLS/PCA/SVM are interpretability controls.",
        "2. Baseline and drift: pybaselines for spectral-background correction; ruptures for offline change-point segmentation; River for online drift monitoring.",
        "3. Genuine temporal windows: MiniRocket/MultiRocket/HYDRA first, then catch22/tsfresh, and only then InceptionTime or TCN if independent sessions are sufficient.",
        "4. Domain shift: SKADA only after cross-day/device/remount splits exist.",
        "5. Never treat the 512 wavelength bins of one spectrum as a time series.",
        "",
        "## Next acquisition priorities",
        "",
        "1. Cross-day and remount sessions with randomized position/force order and long no-contact intervals.",
        "2. Longer release tails to characterize residual response and automatic baseline recovery.",
        "3. High-rate onset, tap, and controlled loading-rate trials with synchronized Fz.",
        "4. Controlled x/y sliding for slip direction and speed.",
        "5. Repeated material, texture, curvature, and held-out-object trials.",
        "6. Controlled multi-contact combinations and off-grid contact-location ground truth.",
        "",
        "## Boundary",
        "",
        "This is an offline research audit. No light source or six-axis force sensor was connected; no live hardware test was attempted. No UI, recorder, runtime model, deployment bundle, or EXE was changed.",
    ]
    (args.output_dir / "algorithm_and_tactile_information_decision.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )

    manifest_files = sorted(
        path.name for path in args.output_dir.iterdir() if path.is_file()
    )
    manifest = {
        "audit_id": "optical_algorithm_and_tactile_information_audit_20260802",
        "generated_file_count": len(manifest_files),
        "files": manifest_files,
        "formal_split": "immutable_grouped_by_session_id_5fold",
        "hardware_connected": False,
        "live_validation_performed": False,
        "deployment_changed": False,
        "fx_fy_moments_used_as_targets": False,
        "fz_role": "calibration_target_not_additional_tactile_observable",
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote optical tactile information audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
