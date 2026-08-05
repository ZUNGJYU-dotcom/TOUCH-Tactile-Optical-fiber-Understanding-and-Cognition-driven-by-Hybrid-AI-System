"""Small temporal summaries required by the deployed TOUCH runtime."""

from __future__ import annotations

import numpy as np


SUMMARY_FEATURE_BLOCK_ORDER = (
    "mean",
    "std",
    "min",
    "max",
    "p10",
    "p90",
    "first",
    "last",
    "delta",
    "slope",
    "mean_abs_difference",
    "rms_difference",
)


def temporal_summary_features(values: np.ndarray) -> np.ndarray:
    """Summarize ``[windows, time, features]`` using past data only."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < 2:
        raise ValueError("runtime values must have shape [windows, time>=2, features]")
    time_axis = np.linspace(-1.0, 1.0, array.shape[1], dtype=np.float32)
    centered_time = time_axis - float(np.mean(time_axis))
    denominator = max(float(np.sum(centered_time**2)), 1.0e-8)
    centered_values = array - np.mean(array, axis=1, keepdims=True)
    slope = (
        np.sum(centered_values * centered_time[None, :, None], axis=1)
        / denominator
    )
    difference = np.diff(array, axis=1)
    blocks = [
        np.mean(array, axis=1),
        np.std(array, axis=1),
        np.min(array, axis=1),
        np.max(array, axis=1),
        np.percentile(array, 10.0, axis=1),
        np.percentile(array, 90.0, axis=1),
        array[:, 0, :],
        array[:, -1, :],
        array[:, -1, :] - array[:, 0, :],
        slope,
        np.mean(np.abs(difference), axis=1),
        np.sqrt(np.mean(difference**2, axis=1)),
    ]
    return np.nan_to_num(
        np.column_stack(blocks),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


__all__ = ["SUMMARY_FEATURE_BLOCK_ORDER", "temporal_summary_features"]
