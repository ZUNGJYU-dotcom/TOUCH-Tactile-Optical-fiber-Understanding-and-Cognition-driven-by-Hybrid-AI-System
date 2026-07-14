"""Mature, dependency-light FBG wavelength-shift demodulation primitives.

The primary measurand is the Bragg wavelength displacement relative to a
no-contact baseline. Intensity is used only to locate and qualify the spectral
feature. No strain, temperature, displacement, pressure, or force conversion
is performed here.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


EPSILON = 1e-12


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(values: Iterable[float]) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return 0.5 * (clean[middle - 1] + clean[middle])


def _mad_sigma(values: list[float]) -> float:
    center = _median(values)
    if center is None:
        return 0.0
    mad = _median(abs(value - center) for value in values)
    return 1.4826 * float(mad or 0.0)


def _window_pairs(
    wavelength_nm: Iterable[Any],
    intensity: Iterable[Any],
    center_nm: float,
    half_width_nm: float,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for raw_x, raw_y in zip(wavelength_nm, intensity):
        x = _finite(raw_x)
        y = _finite(raw_y)
        if x is None or y is None:
            continue
        if abs(x - center_nm) <= half_width_nm:
            pairs.append((x, y))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _feature_mode(values: list[float], requested: str) -> str:
    if requested in {"peak", "dip"}:
        return requested
    center = _median(values) or 0.0
    peak_span = max(values) - center
    dip_span = center - min(values)
    return "dip" if dip_span > peak_span else "peak"


def estimate_peak_wavelengths(
    wavelength_nm: Iterable[Any],
    intensity: Iterable[Any],
    center_nm: float,
    half_width_nm: float = 0.8,
    feature_mode: str = "auto",
) -> dict[str, Any]:
    """Estimate one FBG center with max, centroid, and parabolic methods."""
    pairs = _window_pairs(wavelength_nm, intensity, center_nm, half_width_nm)
    if len(pairs) < 3:
        return {
            "valid": False,
            "reason": "insufficient_points_in_search_window",
            "point_count": len(pairs),
        }

    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mode = _feature_mode(ys, feature_mode)
    local_center = _median(ys) or 0.0
    feature = [value if mode == "peak" else -value for value in ys]
    peak_index = max(range(len(feature)), key=feature.__getitem__)
    max_peak_nm = xs[peak_index]

    if mode == "dip":
        weights = [max(local_center - value, 0.0) for value in ys]
    else:
        weights = [max(value - local_center, 0.0) for value in ys]
    weight_sum = sum(weights)
    centroid_nm = (
        sum(x * weight for x, weight in zip(xs, weights)) / weight_sum
        if weight_sum > EPSILON
        else max_peak_nm
    )

    parabolic_nm = max_peak_nm
    parabolic_valid = False
    if 0 < peak_index < len(feature) - 1:
        left = feature[peak_index - 1]
        middle = feature[peak_index]
        right = feature[peak_index + 1]
        denominator = left - 2.0 * middle + right
        if abs(denominator) > EPSILON:
            fractional_index = 0.5 * (left - right) / denominator
            fractional_index = max(-1.0, min(1.0, fractional_index))
            left_step = xs[peak_index] - xs[peak_index - 1]
            right_step = xs[peak_index + 1] - xs[peak_index]
            step = 0.5 * (left_step + right_step)
            parabolic_nm = xs[peak_index] + fractional_index * step
            parabolic_valid = True

    edge_values = ys[: max(1, len(ys) // 5)] + ys[-max(1, len(ys) // 5) :]
    local_baseline = _median(edge_values) or local_center
    noise = _mad_sigma(edge_values)
    feature_amplitude = (
        local_baseline - min(ys) if mode == "dip" else max(ys) - local_baseline
    )
    snr = feature_amplitude / max(noise, EPSILON)

    return {
        "valid": True,
        "reason": None,
        "point_count": len(pairs),
        "feature_mode": mode,
        "max_peak_wavelength_nm": max_peak_nm,
        "weighted_centroid_wavelength_nm": centroid_nm,
        "parabolic_peak_wavelength_nm": parabolic_nm,
        "parabolic_fit_valid": parabolic_valid,
        "estimator_disagreement_pm": abs(centroid_nm - parabolic_nm) * 1000.0,
        "peak_intensity_counts": ys[peak_index],
        "local_baseline_counts": local_baseline,
        "peak_snr": snr,
    }


def _interpolate(xs: list[float], ys: list[float], x: float) -> float | None:
    if not xs or len(xs) != len(ys) or x < xs[0] or x > xs[-1]:
        return None
    low = 0
    high = len(xs) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if xs[middle] <= x:
            low = middle
        else:
            high = middle
    x0, x1 = xs[low], xs[high]
    if abs(x1 - x0) <= EPSILON:
        return ys[low]
    fraction = (x - x0) / (x1 - x0)
    return ys[low] * (1.0 - fraction) + ys[high] * fraction


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 5:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    a = [value - left_mean for value in left]
    b = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in a) * sum(value * value for value in b))
    if denominator <= EPSILON:
        return None
    return sum(x * y for x, y in zip(a, b)) / denominator


def cross_correlation_shift_pm(
    wavelength_nm: Iterable[Any],
    intensity: Iterable[Any],
    baseline_wavelength_nm: Iterable[Any],
    baseline_intensity: Iterable[Any],
    center_nm: float,
    half_width_nm: float = 0.8,
    max_shift_pm: float = 500.0,
) -> dict[str, Any]:
    """Track a spectral translation against a frozen baseline spectrum."""
    current_pairs = _window_pairs(wavelength_nm, intensity, center_nm, half_width_nm)
    baseline_pairs = _window_pairs(
        baseline_wavelength_nm,
        baseline_intensity,
        center_nm,
        half_width_nm + max_shift_pm / 1000.0,
    )
    if len(current_pairs) < 9 or len(baseline_pairs) < 9:
        return {"valid": False, "reason": "insufficient_points_for_cross_correlation"}

    current_x = [item[0] for item in current_pairs]
    current_y = [item[1] for item in current_pairs]
    baseline_x = [item[0] for item in baseline_pairs]
    baseline_y_raw = [item[1] for item in baseline_pairs]
    baseline_y: list[float] = []
    retained_x: list[float] = []
    retained_current: list[float] = []
    for x, current_value in zip(current_x, current_y):
        interpolated = _interpolate(baseline_x, baseline_y_raw, x)
        if interpolated is None:
            continue
        retained_x.append(x)
        retained_current.append(current_value)
        baseline_y.append(interpolated)
    if len(retained_x) < 9:
        return {"valid": False, "reason": "baseline_and_current_grids_do_not_overlap"}

    steps = [
        retained_x[index + 1] - retained_x[index]
        for index in range(len(retained_x) - 1)
        if retained_x[index + 1] > retained_x[index]
    ]
    step_nm = _median(steps)
    if step_nm is None or step_nm <= EPSILON:
        return {"valid": False, "reason": "invalid_wavelength_grid"}
    max_lag = max(1, min(len(retained_x) // 3, int(round((max_shift_pm / 1000.0) / step_nm))))

    scores: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            base_segment = baseline_y[: len(baseline_y) - lag or None]
            current_segment = retained_current[lag:]
        else:
            base_segment = baseline_y[-lag:]
            current_segment = retained_current[: len(retained_current) + lag]
        score = _correlation(base_segment, current_segment)
        if score is not None:
            scores[lag] = score
    if not scores:
        return {"valid": False, "reason": "cross_correlation_failed"}

    best_lag = max(scores, key=scores.get)
    subpixel = 0.0
    if best_lag - 1 in scores and best_lag + 1 in scores:
        left = scores[best_lag - 1]
        middle = scores[best_lag]
        right = scores[best_lag + 1]
        denominator = left - 2.0 * middle + right
        if abs(denominator) > EPSILON:
            subpixel = max(-1.0, min(1.0, 0.5 * (left - right) / denominator))
    shift_pm = (best_lag + subpixel) * step_nm * 1000.0
    return {
        "valid": True,
        "reason": None,
        "shift_pm": shift_pm,
        "correlation_coefficient": scores[best_lag],
        "integer_lag": best_lag,
        "subpixel_lag": subpixel,
        "wavelength_step_pm": step_nm * 1000.0,
    }


def wavelength_shift_metrics(
    current_wavelength_nm: Any,
    baseline_wavelength_nm: Any,
    *,
    thresholds_pm: dict[str, Any] | None = None,
    visualization_full_scale_pm: float = 500.0,
    baseline_noise_pm: Any = None,
) -> dict[str, Any]:
    """Convert a tracked Bragg wavelength into an uncalibrated response."""
    current = _finite(current_wavelength_nm)
    baseline = _finite(baseline_wavelength_nm)
    noise_pm = _finite(baseline_noise_pm)
    thresholds = thresholds_pm or {}
    no_contact_max = float(thresholds.get("no_contact_max_abs_shift_pm", 10.0))
    small_max = float(thresholds.get("small_shift_max_abs_pm", 75.0))
    moderate_max = float(thresholds.get("moderate_shift_max_abs_pm", 250.0))
    noise_k = float(thresholds.get("noise_gate_k", 3.0))

    if current is None or baseline is None:
        return {
            "valid": False,
            "response_level": "baseline_required",
            "delta_wavelength_nm": None,
            "delta_wavelength_pm": None,
            "absolute_shift_pm": None,
            "shift_direction": "unknown",
            "wavelength_shift_response_ratio": None,
            "temperature_strain_decoupled": False,
            "quality_flags": ["baseline_wavelength_required"],
        }

    delta_nm = current - baseline
    delta_pm = delta_nm * 1000.0
    absolute_pm = abs(delta_pm)
    effective_no_contact = max(no_contact_max, noise_k * max(noise_pm or 0.0, 0.0))
    if absolute_pm <= effective_no_contact:
        level = "no_contact"
        direction = "stable"
    elif absolute_pm < small_max:
        level = "small_shift"
        direction = "red_shift" if delta_pm > 0 else "blue_shift"
    elif absolute_pm < moderate_max:
        level = "moderate_shift"
        direction = "red_shift" if delta_pm > 0 else "blue_shift"
    else:
        level = "large_shift"
        direction = "red_shift" if delta_pm > 0 else "blue_shift"

    response_ratio = min(1.0, absolute_pm / max(float(visualization_full_scale_pm), EPSILON))
    return {
        "valid": True,
        "response_level": level,
        "delta_wavelength_nm": delta_nm,
        "delta_wavelength_pm": delta_pm,
        "absolute_shift_pm": absolute_pm,
        "shift_direction": direction,
        "wavelength_shift_response_ratio": response_ratio,
        "effective_no_contact_threshold_pm": effective_no_contact,
        "temperature_strain_decoupled": False,
        "quality_flags": [],
    }
