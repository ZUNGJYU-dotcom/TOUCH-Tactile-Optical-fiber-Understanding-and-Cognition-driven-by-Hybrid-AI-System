"""Session-level optical observability audit for force regression.

The audit uses PX6D Fz only after acquisition to determine whether nominally
similar force ranges produced comparable optical responses.  It is diagnostic
evidence, not a runtime input, sample filter, or per-test-session calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .advanced_optical_benchmark import AlignedOpticalDataset


REQUIRED_SUMMARY_FEATURES = (
    "global_log_ratio_rms",
    "global_shape_delta_rms",
    "global_intensity_log_ratio",
)


def _linear_stats(reference: np.ndarray, response: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(response, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or float(np.std(x)) <= 1.0e-12:
        return {"slope": None, "intercept": None, "pearson_r": None}
    slope, intercept = np.polyfit(x, y, 1)
    correlation = None
    if float(np.std(y)) > 1.0e-12:
        correlation = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": correlation,
    }


def _feature_indices(dataset: AlignedOpticalDataset) -> dict[str, int]:
    names = dataset.spectrum_feature_names.astype(str)
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in REQUIRED_SUMMARY_FEATURES if name not in lookup]
    if missing:
        raise ValueError("spectrum summary features missing: " + ", ".join(missing))
    return {name: int(lookup[name]) for name in REQUIRED_SUMMARY_FEATURES}


def build_session_observability_table(
    dataset: AlignedOpticalDataset,
    *,
    minimum_force_n: float = 0.10,
    maximum_force_n: float = 5.0,
    high_force_threshold_n: float = 4.0,
) -> pd.DataFrame:
    """Quantify force-to-optical sensitivity for each acquisition session."""

    indices = _feature_indices(dataset)
    formal = dataset.force_mask & (dataset.fold_id >= 0)
    active = (
        formal
        & np.isfinite(dataset.force_fz_n)
        & (dataset.force_fz_n >= float(minimum_force_n))
        & (dataset.force_fz_n <= float(maximum_force_n))
    )
    rows: list[dict[str, Any]] = []
    for group_id in dict.fromkeys(dataset.group_id[active].astype(str).tolist()):
        selected = active & (dataset.group_id.astype(str) == str(group_id))
        position_values = dataset.position_target[selected].astype(str)
        if len(position_values) == 0:
            continue
        position_id = str(pd.Series(position_values).mode().iloc[0])
        force = dataset.force_fz_n[selected].astype(float)
        log_rms = dataset.spectrum_features[
            selected, indices["global_log_ratio_rms"]
        ].astype(float)
        shape_rms = dataset.spectrum_features[
            selected, indices["global_shape_delta_rms"]
        ].astype(float)
        intensity_log = np.abs(
            dataset.spectrum_features[
                selected, indices["global_intensity_log_ratio"]
            ].astype(float)
        )
        log_stats = _linear_stats(force, log_rms)
        shape_stats = _linear_stats(force, shape_rms)
        intensity_stats = _linear_stats(force, intensity_log)
        high = force >= float(high_force_threshold_n)
        rows.append(
            {
                "position_id": position_id,
                "group_id": str(group_id),
                "fold_id": int(dataset.fold_id[selected][0]),
                "active_frame_count": int(np.sum(selected)),
                "force_min_n": float(np.min(force)),
                "force_max_n": float(np.max(force)),
                "force_span_n": float(np.max(force) - np.min(force)),
                "global_log_ratio_rms_slope_per_n": log_stats["slope"],
                "global_log_ratio_rms_pearson_r": log_stats["pearson_r"],
                "global_shape_delta_rms_slope_per_n": shape_stats["slope"],
                "global_shape_delta_rms_pearson_r": shape_stats["pearson_r"],
                "absolute_global_intensity_log_ratio_slope_per_n": intensity_stats[
                    "slope"
                ],
                "absolute_global_intensity_log_ratio_pearson_r": intensity_stats[
                    "pearson_r"
                ],
                "high_force_frame_count": int(np.sum(high)),
                "high_force_log_ratio_rms_median": (
                    float(np.median(log_rms[high])) if np.any(high) else None
                ),
                "evaluation_role": "post_acquisition_observability_audit",
                "force_sensor_used_as_runtime_input": False,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no force-labelled sessions available for observability audit")

    frame["position_median_log_ratio_slope_per_n"] = np.nan
    frame["sensitivity_ratio_to_position_median"] = np.nan
    frame["sensitivity_robust_z"] = np.nan
    frame["observability_status"] = "comparable_optical_sensitivity"
    for position_id, group in frame.groupby("position_id", sort=False):
        values = group["global_log_ratio_rms_slope_per_n"].to_numpy(dtype=float)
        finite = np.isfinite(values) & (values > 0)
        if not np.any(finite):
            frame.loc[group.index, "observability_status"] = "manual_review_required"
            continue
        median = float(np.median(values[finite]))
        absolute_deviation = np.abs(values[finite] - median)
        mad = float(np.median(absolute_deviation))
        scale = 1.4826 * mad
        ratio = np.divide(
            values,
            median,
            out=np.full_like(values, np.nan),
            where=median > 1.0e-12,
        )
        robust_z = (
            (values - median) / scale
            if scale > 1.0e-12
            else np.full_like(values, np.nan)
        )
        frame.loc[group.index, "position_median_log_ratio_slope_per_n"] = median
        frame.loc[group.index, "sensitivity_ratio_to_position_median"] = ratio
        frame.loc[group.index, "sensitivity_robust_z"] = robust_z
        low = (ratio < 0.45) | ((ratio < 0.60) & (robust_z < -2.5))
        warning = (~low) & (ratio < 0.75)
        invalid = ~np.isfinite(values) | (values <= 0)
        frame.loc[group.index[warning], "observability_status"] = (
            "optical_sensitivity_warning"
        )
        frame.loc[group.index[low], "observability_status"] = (
            "low_optical_sensitivity"
        )
        frame.loc[group.index[invalid], "observability_status"] = (
            "manual_review_required"
        )
    return frame.sort_values(["position_id", "group_id"], kind="stable")


__all__ = [
    "REQUIRED_SUMMARY_FEATURES",
    "build_session_observability_table",
]
