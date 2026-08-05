"""Runtime-safe baseline-relative full-spectrum feature extraction."""

from __future__ import annotations

import numpy as np


EPSILON = 1.0e-9


def _downsample_mean(values: np.ndarray, bin_count: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional frame matrix")
    if bin_count <= 0 or bin_count > values.shape[1]:
        raise ValueError("bin_count must be within the spectrum point count")
    edges = np.linspace(0, values.shape[1], bin_count + 1, dtype=int)
    return np.column_stack(
        [
            np.mean(values[:, edges[index] : edges[index + 1]], axis=1)
            for index in range(bin_count)
        ]
    )


def extract_baseline_relative_features(
    intensity: np.ndarray,
    baseline: np.ndarray,
    wavelength_nm: np.ndarray,
    *,
    bin_count: int,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[int, ...]]]:
    """Extract model features without importing training-only dependencies."""

    intensity = np.asarray(intensity, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if intensity.ndim != 2:
        raise ValueError("intensity must have shape [frames, spectrum_points]")
    if baseline.shape != (intensity.shape[1],):
        raise ValueError("baseline must match the spectrum point count")
    if wavelength_nm.shape != baseline.shape:
        raise ValueError("wavelength_nm must match the spectrum point count")

    safe_intensity = np.maximum(intensity, 0.0)
    safe_baseline = np.maximum(baseline, 0.0)
    log_ratio = np.log1p(safe_intensity) - np.log1p(safe_baseline)
    current_shape = safe_intensity / np.maximum(
        np.mean(safe_intensity, axis=1, keepdims=True), EPSILON
    )
    baseline_shape = safe_baseline / max(float(np.mean(safe_baseline)), EPSILON)
    shape_delta = current_shape - baseline_shape
    current_log_shape = np.log1p(safe_intensity)
    current_log_shape -= np.mean(current_log_shape, axis=1, keepdims=True)
    derivative = np.gradient(current_log_shape, wavelength_nm, axis=1)

    blocks = [
        _downsample_mean(log_ratio, bin_count),
        _downsample_mean(shape_delta, bin_count),
        _downsample_mean(current_log_shape, bin_count),
        _downsample_mean(derivative, bin_count),
    ]
    names: list[str] = []
    for prefix in (
        "spectrum_log_ratio_bin",
        "spectrum_shape_delta_bin",
        "spectrum_current_log_shape_bin",
        "spectrum_current_derivative_bin",
    ):
        names.extend(f"{prefix}_{index:03d}" for index in range(1, bin_count + 1))

    centered_current = safe_intensity - np.mean(
        safe_intensity, axis=1, keepdims=True
    )
    centered_baseline = safe_baseline - float(np.mean(safe_baseline))
    correlation_denominator = np.sqrt(
        np.sum(centered_current**2, axis=1)
        * float(np.sum(centered_baseline**2))
    )
    shape_correlation = np.sum(
        centered_current * centered_baseline, axis=1
    ) / np.maximum(correlation_denominator, EPSILON)
    global_summary = np.column_stack(
        [
            np.mean(log_ratio, axis=1),
            np.std(log_ratio, axis=1),
            np.sqrt(np.mean(log_ratio**2, axis=1)),
            np.max(np.abs(log_ratio), axis=1),
            np.sqrt(np.mean(shape_delta**2, axis=1)),
            np.max(np.abs(shape_delta), axis=1),
            shape_correlation,
            np.log(
                np.maximum(np.mean(safe_intensity, axis=1), EPSILON)
                / max(float(np.mean(safe_baseline)), EPSILON)
            ),
        ]
    )
    blocks.append(global_summary)
    names.extend(
        (
            "global_log_ratio_mean",
            "global_log_ratio_std",
            "global_log_ratio_rms",
            "global_log_ratio_max_abs",
            "global_shape_delta_rms",
            "global_shape_delta_max_abs",
            "global_shape_correlation",
            "global_intensity_log_ratio",
        )
    )
    matrix = np.concatenate(blocks, axis=1)
    first_three_count = 3 * bin_count
    feature_sets = {
        "baseline_relative_192": tuple(range(first_three_count)),
        "baseline_relative_264": tuple(range(matrix.shape[1])),
    }
    return matrix, tuple(names), feature_sets


__all__ = ["extract_baseline_relative_features"]
