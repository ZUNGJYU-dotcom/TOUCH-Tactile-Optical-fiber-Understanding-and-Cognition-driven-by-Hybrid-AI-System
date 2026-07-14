"""Quality-aware local FBG peak tracking utilities.

The functions in this module deliberately keep spectral feature tracking
separate from physical P11-P33 channel approval. A stable wavelength-order
candidate is not, by itself, an approved physical channel identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1.0e-12


@dataclass(frozen=True)
class PeakTrackingResult:
    marker_nm: float
    centroid_nm: float
    parabolic_nm: float
    cross_correlation_shift_pm: float
    cross_correlation_coefficient: float
    cross_correlation_reliable: bool
    delta_centroid_pm: float
    delta_parabolic_pm: float
    quality_fused_shift_pm: float
    morphology_estimator_disagreement_pm: float
    quality_fused_shift_reliable: bool
    height_counts: float
    height_ratio: float
    area_counts_nm: float
    area_ratio: float
    fwhm_nm: float
    delta_fwhm_pm: float
    skewness: float
    delta_skewness: float
    local_baseline_counts: float
    peak_snr: float
    baseline_peak_snr: float
    baseline_peak_valid: bool
    edge_margin_nm: float
    estimator_spread_pm: float
    shape_correlation: float
    normalized_shape_rmse: float
    valid_peak: bool
    quality_flags: tuple[str, ...]


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def edge_baseline(values: np.ndarray, edge_fraction: float = 0.2) -> float:
    values = np.asarray(values, dtype=float)
    edge_count = max(1, int(round(values.size * edge_fraction)))
    edge_count = min(edge_count, max(1, values.size // 2))
    edges = np.concatenate([values[:edge_count], values[-edge_count:]])
    return float(np.median(edges))


def _edge_noise(values: np.ndarray, edge_fraction: float) -> float:
    edge_count = max(2, int(round(values.size * edge_fraction)))
    edge_count = min(edge_count, max(1, values.size // 2))
    indices = np.concatenate(
        [np.arange(edge_count), np.arange(values.size - edge_count, values.size)]
    )
    edge_values = values[indices]
    if edge_values.size >= 3:
        fit = np.polyval(np.polyfit(indices.astype(float), edge_values, 1), indices)
        noise = robust_sigma(edge_values - fit)
    else:
        noise = robust_sigma(edge_values)
    return max(float(noise), 1.0)


def _weighted_centroid(x: np.ndarray, y: np.ndarray, baseline: float) -> float:
    weights = np.clip(y - baseline, 0.0, None)
    total = float(np.sum(weights))
    if total <= EPSILON:
        return float(x[int(np.argmax(y))])
    return float(np.sum(x * weights) / total)


def _parabolic_center(x: np.ndarray, y: np.ndarray) -> float:
    index = int(np.argmax(y))
    if index <= 0 or index >= y.size - 1:
        return float(x[index])
    fit_x = x[index - 1 : index + 2]
    fit_y = y[index - 1 : index + 2]
    a, b, _ = np.polyfit(fit_x, fit_y, 2)
    if not np.isfinite(a) or not np.isfinite(b) or a >= 0 or abs(a) <= EPSILON:
        return float(x[index])
    center = float(-b / (2.0 * a))
    return center if float(fit_x[0]) <= center <= float(fit_x[-1]) else float(x[index])


def _fwhm_nm(x: np.ndarray, y: np.ndarray, baseline: float) -> float:
    signal = np.clip(y - baseline, 0.0, None)
    peak = float(np.max(signal))
    if peak <= EPSILON:
        return 0.0
    indices = np.flatnonzero(signal >= 0.5 * peak)
    if indices.size < 2:
        return float(np.median(np.diff(x)))
    return float(x[indices[-1]] - x[indices[0]])


def shape_correlation(current: np.ndarray, baseline: np.ndarray) -> float:
    a = np.asarray(current, dtype=float) - float(np.mean(current))
    b = np.asarray(baseline, dtype=float) - float(np.mean(baseline))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPSILON else 0.0


def peak_metrics(
    x: np.ndarray,
    y: np.ndarray,
    edge_fraction: float = 0.2,
) -> dict[str, float]:
    baseline = edge_baseline(y, edge_fraction=edge_fraction)
    positive = np.clip(y - baseline, 0.0, None)
    area = float(np.trapezoid(positive, x))
    centroid = _weighted_centroid(x, y, baseline)
    total = max(float(np.sum(positive)), EPSILON)
    variance = float(np.sum(positive * (x - centroid) ** 2) / total)
    std = float(np.sqrt(max(variance, 0.0)))
    skew = (
        float(np.sum(positive * ((x - centroid) / std) ** 3) / total)
        if std > EPSILON
        else 0.0
    )
    marker_index = int(np.argmax(y))
    return {
        "centroid_nm": centroid,
        "parabolic_nm": _parabolic_center(x, y),
        "marker_nm": float(x[marker_index]),
        "marker_index": float(marker_index),
        "height_counts": float(np.max(y) - baseline),
        "area_counts_nm": area,
        "fwhm_nm": _fwhm_nm(x, y, baseline),
        "skewness": skew,
        "local_baseline_counts": baseline,
        "edge_noise_counts": _edge_noise(y, edge_fraction=edge_fraction),
    }


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    position = np.arange(values.size, dtype=float)
    if values.size < 3:
        return values - float(np.mean(values))
    trend = np.polyval(np.polyfit(position, values, 1), position)
    return values - trend


def cross_correlation_shift(
    wavelength_nm: np.ndarray,
    current: np.ndarray,
    baseline: np.ndarray,
    maximum_shift_nm: float = 0.5,
) -> tuple[float, float]:
    """Estimate current-minus-baseline shift with overlap-normalized correlation."""

    x = np.asarray(wavelength_nm, dtype=float)
    current_values = _linear_detrend(np.asarray(current, dtype=float))
    baseline_values = _linear_detrend(np.asarray(baseline, dtype=float))
    if x.size < 5 or current_values.size != x.size or baseline_values.size != x.size:
        return float("nan"), 0.0
    step_nm = float(np.median(np.diff(x)))
    maximum_lag = min(
        int(np.floor(maximum_shift_nm / max(step_nm, EPSILON))),
        max(0, x.size - 5),
    )
    lags = np.arange(-maximum_lag, maximum_lag + 1, dtype=int)
    scores = np.full(lags.size, -1.0, dtype=float)
    for index, lag in enumerate(lags):
        if lag > 0:
            left = current_values[lag:]
            right = baseline_values[:-lag]
        elif lag < 0:
            left = current_values[:lag]
            right = baseline_values[-lag:]
        else:
            left = current_values
            right = baseline_values
        left = left - float(np.mean(left))
        right = right - float(np.mean(right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > EPSILON:
            scores[index] = float(np.dot(left, right) / denominator)

    best_index = int(np.argmax(scores))
    lag_samples = float(lags[best_index])
    if 0 < best_index < scores.size - 1:
        previous_value = scores[best_index - 1]
        center_value = scores[best_index]
        next_value = scores[best_index + 1]
        denominator = previous_value - 2.0 * center_value + next_value
        if np.isfinite(denominator) and abs(denominator) > EPSILON:
            correction = 0.5 * (previous_value - next_value) / denominator
            lag_samples += float(np.clip(correction, -1.0, 1.0))
    shift_pm = 1000.0 * lag_samples * step_nm
    return float(shift_pm), float(scores[best_index])


def track_peak(
    wavelength_nm: np.ndarray,
    current: np.ndarray,
    baseline: np.ndarray,
    *,
    edge_fraction: float = 0.2,
    maximum_cross_correlation_shift_nm: float = 0.5,
    minimum_peak_height_counts: float = 100.0,
    minimum_peak_snr: float = 3.0,
    minimum_cross_correlation: float = 0.65,
    maximum_estimator_disagreement_pm: float = 80.0,
    maximum_morphology_estimator_disagreement_pm: float = 30.0,
) -> PeakTrackingResult:
    x = np.asarray(wavelength_nm, dtype=float)
    current_values = np.asarray(current, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    if current_values.shape != x.shape or baseline_values.shape != x.shape:
        raise ValueError("peak window arrays must have identical shapes")
    if x.size < 5 or np.any(np.diff(x) <= 0):
        raise ValueError("peak window requires at least five increasing wavelength samples")

    current_metrics = peak_metrics(x, current_values, edge_fraction=edge_fraction)
    baseline_metrics = peak_metrics(x, baseline_values, edge_fraction=edge_fraction)
    cross_shift_pm, cross_coefficient = cross_correlation_shift(
        x,
        current_values,
        baseline_values,
        maximum_shift_nm=maximum_cross_correlation_shift_nm,
    )
    delta_centroid_pm = 1000.0 * (
        current_metrics["centroid_nm"] - baseline_metrics["centroid_nm"]
    )
    delta_parabolic_pm = 1000.0 * (
        current_metrics["parabolic_nm"] - baseline_metrics["parabolic_nm"]
    )
    morphology_estimator_disagreement_pm = abs(delta_centroid_pm - delta_parabolic_pm)
    quality_fused_shift_pm = float(np.mean([delta_centroid_pm, delta_parabolic_pm]))
    estimates = np.asarray([delta_centroid_pm, delta_parabolic_pm, cross_shift_pm], dtype=float)
    finite_estimates = estimates[np.isfinite(estimates)]
    estimator_spread_pm = (
        float(np.ptp(finite_estimates)) if finite_estimates.size >= 2 else float("inf")
    )
    marker_index = int(round(current_metrics["marker_index"]))
    edge_margin_nm = min(
        current_metrics["marker_nm"] - float(x[0]),
        float(x[-1]) - current_metrics["marker_nm"],
    )
    peak_snr = current_metrics["height_counts"] / max(
        current_metrics["edge_noise_counts"], EPSILON
    )
    baseline_peak_snr = baseline_metrics["height_counts"] / max(
        baseline_metrics["edge_noise_counts"], EPSILON
    )
    flags: list[str] = []
    peak_fatal_flags: set[str] = set()
    baseline_fatal_flags: set[str] = set()
    if current_metrics["height_counts"] < minimum_peak_height_counts:
        flags.append("peak_height_low")
        peak_fatal_flags.add("peak_height_low")
    if peak_snr < minimum_peak_snr:
        flags.append("peak_snr_low")
        peak_fatal_flags.add("peak_snr_low")
    if marker_index <= 0 or marker_index >= x.size - 1:
        flags.append("peak_at_window_edge")
        peak_fatal_flags.add("peak_at_window_edge")
    baseline_marker_index = int(round(baseline_metrics["marker_index"]))
    if baseline_metrics["height_counts"] < minimum_peak_height_counts:
        flags.append("baseline_peak_height_low")
        baseline_fatal_flags.add("baseline_peak_height_low")
    if baseline_peak_snr < minimum_peak_snr:
        flags.append("baseline_peak_snr_low")
        baseline_fatal_flags.add("baseline_peak_snr_low")
    if baseline_marker_index <= 0 or baseline_marker_index >= x.size - 1:
        flags.append("baseline_peak_at_window_edge")
        baseline_fatal_flags.add("baseline_peak_at_window_edge")
    if not np.isfinite(cross_shift_pm) or cross_coefficient < minimum_cross_correlation:
        flags.append("cross_correlation_low")
    if estimator_spread_pm > maximum_estimator_disagreement_pm:
        flags.append("wavelength_estimators_disagree")
    if (
        morphology_estimator_disagreement_pm
        > maximum_morphology_estimator_disagreement_pm
    ):
        flags.append("morphology_shift_estimators_disagree")
    if abs(cross_shift_pm) >= maximum_cross_correlation_shift_nm * 1000.0 * 0.98:
        flags.append("cross_correlation_at_search_limit")

    baseline_peak_valid = not bool(baseline_fatal_flags)
    peak_signal_valid = not bool(peak_fatal_flags or baseline_fatal_flags)
    cross_correlation_reliable = bool(
        peak_signal_valid
        and np.isfinite(cross_shift_pm)
        and cross_coefficient >= minimum_cross_correlation
        and abs(cross_shift_pm)
        < maximum_cross_correlation_shift_nm * 1000.0 * 0.98
    )

    current_mean = max(float(np.mean(current_values)), EPSILON)
    baseline_mean = max(float(np.mean(baseline_values)), EPSILON)
    normalized_shape_rmse = float(
        np.sqrt(
            np.mean(
                (
                    current_values / current_mean
                    - baseline_values / baseline_mean
                )
                ** 2
            )
        )
    )
    return PeakTrackingResult(
        marker_nm=current_metrics["marker_nm"],
        centroid_nm=current_metrics["centroid_nm"],
        parabolic_nm=current_metrics["parabolic_nm"],
        cross_correlation_shift_pm=cross_shift_pm,
        cross_correlation_coefficient=cross_coefficient,
        cross_correlation_reliable=cross_correlation_reliable,
        delta_centroid_pm=delta_centroid_pm,
        delta_parabolic_pm=delta_parabolic_pm,
        quality_fused_shift_pm=quality_fused_shift_pm,
        morphology_estimator_disagreement_pm=morphology_estimator_disagreement_pm,
        quality_fused_shift_reliable=(
            peak_signal_valid
            and np.isfinite(quality_fused_shift_pm)
            and morphology_estimator_disagreement_pm
            <= maximum_morphology_estimator_disagreement_pm
        ),
        height_counts=current_metrics["height_counts"],
        height_ratio=current_metrics["height_counts"]
        / max(baseline_metrics["height_counts"], EPSILON),
        area_counts_nm=current_metrics["area_counts_nm"],
        area_ratio=current_metrics["area_counts_nm"]
        / max(baseline_metrics["area_counts_nm"], EPSILON),
        fwhm_nm=current_metrics["fwhm_nm"],
        delta_fwhm_pm=1000.0
        * (current_metrics["fwhm_nm"] - baseline_metrics["fwhm_nm"]),
        skewness=current_metrics["skewness"],
        delta_skewness=current_metrics["skewness"] - baseline_metrics["skewness"],
        local_baseline_counts=current_metrics["local_baseline_counts"],
        peak_snr=float(peak_snr),
        baseline_peak_snr=float(baseline_peak_snr),
        baseline_peak_valid=baseline_peak_valid,
        edge_margin_nm=float(edge_margin_nm),
        estimator_spread_pm=estimator_spread_pm,
        shape_correlation=shape_correlation(current_values, baseline_values),
        normalized_shape_rmse=normalized_shape_rmse,
        valid_peak=peak_signal_valid,
        quality_flags=tuple(flags),
    )
