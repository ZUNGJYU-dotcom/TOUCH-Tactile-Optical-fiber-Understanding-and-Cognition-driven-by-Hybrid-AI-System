"""Robust wavelength-order FBG candidate discovery.

Discovery establishes repeatable spectral candidates only. It deliberately does
not approve physical P11-P33 identities, which require labelled point presses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .tracking import robust_sigma


@dataclass(frozen=True)
class DiscoveredPeak:
    marker_wavelength_nm: float
    refined_wavelength_nm: float
    intensity_counts: float
    prominence_counts: float
    width_nm: float
    peak_snr: float
    edge_margin_nm: float
    valid_peak: bool
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class PeakDiscoveryResult:
    expected_peak_count: int
    detection_threshold_counts: float
    noise_sigma_counts: float
    detected_peaks: tuple[DiscoveredPeak, ...]
    selected_peaks: tuple[DiscoveredPeak, ...]
    status: str


def _odd_window(requested: int, sample_count: int, polynomial_order: int) -> int:
    window = max(int(requested), polynomial_order + 2)
    if window % 2 == 0:
        window += 1
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    minimum = polynomial_order + 2
    if minimum % 2 == 0:
        minimum += 1
    return max(minimum, min(window, maximum))


def _parabolic_refinement(x: np.ndarray, y: np.ndarray, index: int) -> float:
    if index <= 0 or index >= y.size - 1:
        return float(x[index])
    local_x = x[index - 1 : index + 2] - x[index]
    local_y = y[index - 1 : index + 2]
    a, b, _ = np.polyfit(local_x, local_y, 2)
    if not np.isfinite(a) or not np.isfinite(b) or a >= 0.0 or abs(a) < 1.0e-12:
        return float(x[index])
    offset = float(-b / (2.0 * a))
    if float(local_x[0]) <= offset <= float(local_x[-1]):
        return float(x[index] + offset)
    return float(x[index])


def discover_wavelength_order_peaks(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    *,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    expected_peak_count: int,
    smoothing_window_points: int = 7,
    smoothing_polynomial_order: int = 2,
    minimum_peak_distance_nm: float = 2.5,
    minimum_absolute_prominence_counts: float = 1000.0,
    minimum_prominence_fraction_of_range: float = 0.03,
    minimum_peak_snr: float = 5.0,
    minimum_edge_margin_nm: float = 0.35,
    minimum_peak_width_nm: float = 0.05,
    maximum_peak_width_nm: float = 2.0,
) -> PeakDiscoveryResult:
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(spectrum, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 9:
        raise ValueError("wavelength and spectrum must be one-dimensional arrays of equal size")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError("peak discovery requires finite values on an increasing wavelength grid")
    mask = (x >= wavelength_min_nm) & (x <= wavelength_max_nm)
    if int(np.count_nonzero(mask)) < 9:
        raise ValueError("peak discovery range contains too few spectrum samples")
    search_x = x[mask]
    search_y = y[mask]
    window = _odd_window(
        smoothing_window_points,
        search_y.size,
        smoothing_polynomial_order,
    )
    smooth = savgol_filter(search_y, window, smoothing_polynomial_order, mode="interp")
    residual_noise = max(robust_sigma(search_y - smooth), 1.0)
    dynamic_range = max(
        float(np.percentile(smooth, 99.0) - np.percentile(smooth, 5.0)),
        1.0,
    )
    prominence_threshold = max(
        float(minimum_absolute_prominence_counts),
        float(minimum_prominence_fraction_of_range) * dynamic_range,
    )
    wavelength_step = float(np.median(np.diff(search_x)))
    minimum_distance_points = max(1, int(np.floor(minimum_peak_distance_nm / wavelength_step)))
    indices, properties = find_peaks(
        smooth,
        distance=minimum_distance_points,
        prominence=prominence_threshold,
    )
    widths_samples = (
        peak_widths(smooth, indices, rel_height=0.5)[0]
        if indices.size
        else np.asarray([], dtype=float)
    )
    peaks: list[DiscoveredPeak] = []
    for order, index in enumerate(indices):
        prominence = float(properties["prominences"][order])
        width_nm = float(widths_samples[order] * wavelength_step)
        marker = float(search_x[index])
        refined = _parabolic_refinement(search_x, smooth, int(index))
        edge_margin = min(marker - float(search_x[0]), float(search_x[-1]) - marker)
        peak_snr = prominence / residual_noise
        flags: list[str] = []
        if peak_snr < minimum_peak_snr:
            flags.append("peak_snr_low")
        if edge_margin < minimum_edge_margin_nm:
            flags.append("peak_near_search_edge")
        if not minimum_peak_width_nm <= width_nm <= maximum_peak_width_nm:
            flags.append("peak_width_out_of_range")
        peaks.append(
            DiscoveredPeak(
                marker_wavelength_nm=marker,
                refined_wavelength_nm=refined,
                intensity_counts=float(search_y[index]),
                prominence_counts=prominence,
                width_nm=width_nm,
                peak_snr=peak_snr,
                edge_margin_nm=edge_margin,
                valid_peak=not flags,
                quality_flags=tuple(flags),
            )
        )

    valid_peaks = [peak for peak in peaks if peak.valid_peak]
    if len(valid_peaks) > expected_peak_count:
        selected = sorted(
            sorted(valid_peaks, key=lambda peak: peak.prominence_counts, reverse=True)[
                :expected_peak_count
            ],
            key=lambda peak: peak.refined_wavelength_nm,
        )
        status = "pass_with_extra_candidates"
    else:
        selected = sorted(valid_peaks, key=lambda peak: peak.refined_wavelength_nm)
        status = "pass" if len(selected) == expected_peak_count else "insufficient_valid_peaks"
    return PeakDiscoveryResult(
        expected_peak_count=int(expected_peak_count),
        detection_threshold_counts=prominence_threshold,
        noise_sigma_counts=residual_noise,
        detected_peaks=tuple(sorted(peaks, key=lambda peak: peak.refined_wavelength_nm)),
        selected_peaks=tuple(selected),
        status=status,
    )


def assign_peaks_to_references(
    peaks: Iterable[DiscoveredPeak],
    references: Iterable[tuple[str, float]],
    *,
    maximum_offset_nm: float,
) -> tuple[list[dict[str, float | str | bool]], list[DiscoveredPeak]]:
    if not np.isfinite(maximum_offset_nm) or maximum_offset_nm < 0.0:
        raise ValueError("maximum_offset_nm must be finite and non-negative")
    valid_peaks = [peak for peak in peaks if peak.valid_peak]
    reference_list = list(references)
    assignments: list[dict[str, float | str | bool]] = [
        {
            "candidate_id": candidate_id,
            "reference_wavelength_nm": float(reference_nm),
            "matched": False,
            "match_status": "missing_peak",
        }
        for candidate_id, reference_nm in reference_list
    ]
    if not reference_list or not valid_peaks:
        return assignments, valid_peaks

    raw_offset = np.asarray(
        [
            [abs(reference_nm - peak.refined_wavelength_nm) for peak in valid_peaks]
            for _, reference_nm in reference_list
        ],
        dtype=float,
    )
    reference_count = len(reference_list)
    peak_count = len(valid_peaks)

    # Solve a threshold-aware assignment. Post-filtering an ordinary Hungarian
    # result can discard a valid edge when missing or extra peaks are present.
    # One private dummy column per reference makes "unmatched" explicit; its
    # large cost enforces lexicographic priority: maximize the number of valid
    # matches first, then minimize their total wavelength offset.
    offset_scale = max(float(maximum_offset_nm), 1.0)
    unmatched_cost = (reference_count + 1) * offset_scale
    forbidden_cost = (reference_count + peak_count + 2) * unmatched_cost
    cost = np.full(
        (reference_count, peak_count + reference_count),
        forbidden_cost,
        dtype=float,
    )
    allowed = raw_offset <= float(maximum_offset_nm)
    cost[:, :peak_count] = np.where(allowed, raw_offset, forbidden_cost)
    for row_index in range(reference_count):
        cost[row_index, peak_count + row_index] = unmatched_cost

    row_indices, column_indices = linear_sum_assignment(cost)
    used_peak_indices: set[int] = set()
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        if column_index >= peak_count:
            continue
        offset = float(raw_offset[row_index, column_index])
        if offset > maximum_offset_nm:
            raise RuntimeError("threshold-aware assignment selected a forbidden peak edge")
        peak = valid_peaks[column_index]
        assignments[row_index].update(
            {
                "matched": True,
                "match_status": "matched",
                "detected_wavelength_nm": peak.refined_wavelength_nm,
                "marker_wavelength_nm": peak.marker_wavelength_nm,
                "offset_nm": offset,
                "prominence_counts": peak.prominence_counts,
                "peak_snr": peak.peak_snr,
                "peak_width_nm": peak.width_nm,
            }
        )
        used_peak_indices.add(int(column_index))
    unmatched = [
        peak for index, peak in enumerate(valid_peaks) if index not in used_peak_indices
    ]
    return assignments, unmatched
