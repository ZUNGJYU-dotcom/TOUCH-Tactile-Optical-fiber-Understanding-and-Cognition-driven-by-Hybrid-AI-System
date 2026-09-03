"""Lightweight baseline-relative spectral features shared by training and runtime."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .features import PeakWindow


EPSILON = 1.0e-9
LOCAL_FEATURE_SUFFIXES = (
    "centroid_shift_pm",
    "log_area_ratio",
    "log_height_ratio",
    "shape_rmse",
)
GLOBAL_FEATURE_NAMES = (
    "global_log_gain",
    "global_normalized_shape_rms",
    "global_normalized_shape_peak",
    "global_derivative_residual_energy",
)


def baseline_relative_frame_feature_names(
    peak_windows: Iterable[PeakWindow],
) -> tuple[str, ...]:
    """Return the fixed 40-value live frame contract without loading a model."""

    local = tuple(
        f"{window.candidate_id.lower()}_{suffix}"
        for window in peak_windows
        for suffix in LOCAL_FEATURE_SUFFIXES
    )
    return local + GLOBAL_FEATURE_NAMES


def _local_peak_features(
    wavelength_nm: np.ndarray,
    spectra: np.ndarray,
    baseline: np.ndarray,
    window: PeakWindow,
) -> tuple[np.ndarray, tuple[str, ...]]:
    mask = np.abs(wavelength_nm - window.center_nm) <= window.half_width_nm
    if int(np.count_nonzero(mask)) < 5:
        raise ValueError(f"not enough samples near {window.candidate_id}")
    x = wavelength_nm[mask]
    current = spectra[:, mask]
    reference = baseline[mask]
    edge_count = max(1, int(round(len(x) * window.edge_fraction)))
    local_floor = 0.5 * (
        np.mean(current[:, :edge_count], axis=1)
        + np.mean(current[:, -edge_count:], axis=1)
    )
    weights = np.maximum(current - local_floor[:, None], 0.0)
    reference_floor = 0.5 * (
        float(np.mean(reference[:edge_count]))
        + float(np.mean(reference[-edge_count:]))
    )
    reference_weights = np.maximum(reference - reference_floor, 0.0)
    current_area = np.trapezoid(weights, x, axis=1)
    reference_area = max(float(np.trapezoid(reference_weights, x)), EPSILON)
    current_centroid = np.sum(weights * x[None, :], axis=1) / np.maximum(
        np.sum(weights, axis=1), EPSILON
    )
    reference_centroid = float(
        np.sum(reference_weights * x) / max(float(np.sum(reference_weights)), EPSILON)
    )
    current_height = np.max(weights, axis=1)
    reference_height = max(float(np.max(reference_weights)), EPSILON)
    normalized_current = weights / np.maximum(current_area[:, None], EPSILON)
    normalized_reference = reference_weights / reference_area
    shape_rmse = np.sqrt(
        np.mean((normalized_current - normalized_reference[None, :]) ** 2, axis=1)
    )
    prefix = window.candidate_id.lower()
    values = np.column_stack(
        [
            (current_centroid - reference_centroid) * 1000.0,
            np.log(np.maximum(current_area, EPSILON) / reference_area),
            np.log(np.maximum(current_height, EPSILON) / reference_height),
            shape_rmse,
        ]
    )
    names = tuple(f"{prefix}_{suffix}" for suffix in LOCAL_FEATURE_SUFFIXES)
    return values, names


def extract_baseline_relative_frame_features(
    wavelength_nm: np.ndarray,
    spectra: np.ndarray,
    baseline: np.ndarray,
    peak_windows: Iterable[PeakWindow],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]:
    """Extract the identical baseline-relative frame features for offline and live use."""

    wavelength = np.asarray(wavelength_nm, dtype=float)
    frame_spectra = np.asarray(spectra, dtype=float)
    if frame_spectra.ndim == 1:
        frame_spectra = frame_spectra[None, :]
    reference = np.asarray(baseline, dtype=float)
    if frame_spectra.ndim != 2 or frame_spectra.shape[1] != wavelength.size:
        raise ValueError("spectra must have shape [frames, wavelength_samples]")
    if reference.shape != (wavelength.size,):
        raise ValueError("baseline must match the wavelength grid")

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    shift_columns: list[int] = []
    area_columns: list[int] = []
    for window in peak_windows:
        values, names = _local_peak_features(
            wavelength,
            frame_spectra,
            reference,
            window,
        )
        offset = len(feature_names)
        shift_columns.append(offset)
        area_columns.append(offset + 1)
        feature_blocks.append(values)
        feature_names.extend(names)

    spectrum_mean = np.mean(frame_spectra, axis=1)
    baseline_mean = max(float(np.mean(reference)), EPSILON)
    normalized_current = frame_spectra / np.maximum(spectrum_mean[:, None], EPSILON)
    normalized_baseline = reference / baseline_mean
    normalized_residual = normalized_current - normalized_baseline[None, :]
    global_log_gain = np.log(np.maximum(spectrum_mean, EPSILON) / baseline_mean)
    global_shape_rms = np.sqrt(np.mean(normalized_residual**2, axis=1))
    global_shape_peak = np.max(np.abs(normalized_residual), axis=1)
    global_derivative_energy = np.mean(np.diff(normalized_residual, axis=1) ** 2, axis=1)
    feature_blocks.append(
        np.column_stack(
            [
                global_log_gain,
                global_shape_rms,
                global_shape_peak,
                global_derivative_energy,
            ]
        )
    )
    feature_names.extend(GLOBAL_FEATURE_NAMES)
    feature_matrix = np.column_stack(feature_blocks)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    shifts = feature_matrix[:, shift_columns]
    log_areas = feature_matrix[:, area_columns]
    response_components = np.column_stack(
        [
            global_shape_rms,
            np.abs(global_log_gain),
            np.median(np.abs(shifts), axis=1) / 1000.0,
            np.median(np.abs(log_areas), axis=1),
        ]
    )
    component_names = (
        "global_shape_rms",
        "absolute_global_log_gain",
        "median_absolute_centroid_shift_nm",
        "median_absolute_log_area_change",
    )
    return feature_matrix, tuple(feature_names), response_components, component_names


__all__ = [
    "baseline_relative_frame_feature_names",
    "extract_baseline_relative_frame_features",
]
