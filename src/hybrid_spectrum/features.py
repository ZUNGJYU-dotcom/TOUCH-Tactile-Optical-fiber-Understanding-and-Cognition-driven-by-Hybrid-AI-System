"""Mixed wavelength, intensity, area and shape feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .dataset import SpectrumSegment
from .tracking import shape_correlation, track_peak


EPSILON = 1.0e-12
GLOBAL_ARRAY_CANDIDATE_IDS = tuple(f"FBG{index:02d}" for index in range(1, 10))

FEATURE_ROW_METADATA_COLUMNS = (
    "trial_id",
    "segment_id",
    "label",
    "phase",
    "frame_id",
    "timestamp",
    "data_source",
    "training_eligible",
    "segment_integrity_verified",
    "segment_fingerprint_sha256",
    "baseline_reference_segment_fingerprint_sha256",
    "backend_session_started_at_epoch",
    "device_id",
    "integration_ms",
    "spectrum_peak_profile",
    "operator_attested_no_contact",
)


@dataclass(frozen=True)
class PeakWindow:
    candidate_id: str
    provisional_channel_id: str
    center_nm: float
    half_width_nm: float
    edge_fraction: float = 0.2
    maximum_cross_correlation_shift_nm: float = 0.5
    minimum_peak_height_counts: float = 100.0
    minimum_peak_snr: float = 3.0
    minimum_cross_correlation: float = 0.65
    maximum_estimator_disagreement_pm: float = 80.0
    maximum_morphology_estimator_disagreement_pm: float = 30.0
    minimum_common_mode_valid_peaks: int = 1


def load_peak_windows(config_path: Path) -> list[PeakWindow]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    configured_required_ids = tuple(
        str(value)
        for value in config.get("operating_scope", {}).get(
            "required_candidate_ids", GLOBAL_ARRAY_CANDIDATE_IDS
        )
    )
    if configured_required_ids != GLOBAL_ARRAY_CANDIDATE_IDS:
        raise ValueError(
            "global-array config must require exactly FBG01 through FBG09 in canonical order"
        )
    feature_config = config.get("feature_extraction", {})
    tracking_config = config.get("peak_tracking_quality", {})
    common_mode_config = tracking_config.get("common_mode_correction", {})
    half_width = float(feature_config.get("peak_search_half_width_nm", 0.9))
    edge_fraction = float(feature_config.get("local_baseline_edge_fraction", 0.2))
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("peak_search_half_width_nm must be finite and positive")
    if not np.isfinite(edge_fraction) or not 0.0 < edge_fraction <= 0.5:
        raise ValueError("local_baseline_edge_fraction must be in (0, 0.5]")

    candidate_items = config.get("wavelength_order_candidates", [])
    if not isinstance(candidate_items, list):
        raise ValueError("wavelength_order_candidates must be a list")
    candidate_ids = tuple(str(item.get("candidate_id", "")) for item in candidate_items)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("wavelength_order_candidates contains duplicate candidate_id values")
    if set(candidate_ids) != set(GLOBAL_ARRAY_CANDIDATE_IDS):
        raise ValueError(
            "wavelength_order_candidates must contain exactly one each of FBG01 through FBG09"
        )
    candidate_by_id = {
        str(item["candidate_id"]): item for item in candidate_items
    }

    windows = []
    for candidate_id in GLOBAL_ARRAY_CANDIDATE_IDS:
        item = candidate_by_id[candidate_id]
        center_nm = float(item["center_nm"])
        if not np.isfinite(center_nm):
            raise ValueError(f"{candidate_id} center_nm must be finite")
        windows.append(
            PeakWindow(
                candidate_id=candidate_id,
                provisional_channel_id=str(item["provisional_channel_id"]),
                center_nm=center_nm,
                half_width_nm=half_width,
                edge_fraction=edge_fraction,
                maximum_cross_correlation_shift_nm=float(
                    tracking_config.get("maximum_cross_correlation_shift_nm", 0.5)
                ),
                minimum_peak_height_counts=float(
                    tracking_config.get("minimum_peak_height_counts", 100.0)
                ),
                minimum_peak_snr=float(tracking_config.get("minimum_peak_snr", 3.0)),
                minimum_cross_correlation=float(
                    tracking_config.get("minimum_cross_correlation", 0.65)
                ),
                maximum_estimator_disagreement_pm=float(
                    tracking_config.get("maximum_estimator_disagreement_pm", 80.0)
                ),
                maximum_morphology_estimator_disagreement_pm=float(
                    tracking_config.get(
                        "maximum_morphology_estimator_disagreement_pm", 30.0
                    )
                ),
                minimum_common_mode_valid_peaks=int(
                    common_mode_config.get("minimum_valid_peaks", 5)
                ),
            )
        )
    for left, right in zip(windows, windows[1:], strict=False):
        if right.center_nm <= left.center_nm:
            raise ValueError("FBG01 through FBG09 centers must be strictly increasing")
        if right.center_nm - left.center_nm <= 2.0 * half_width:
            raise ValueError(
                f"peak search windows overlap for {left.candidate_id} and {right.candidate_id}"
            )
    return windows


def extract_frame_features(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    baseline_spectrum: np.ndarray,
    peak_windows: Iterable[PeakWindow],
) -> dict[str, float]:
    if spectrum.shape != wavelength_nm.shape or baseline_spectrum.shape != wavelength_nm.shape:
        raise ValueError("spectrum, baseline and wavelength arrays must have the same shape")
    windows = list(peak_windows)
    if not windows:
        raise ValueError("at least one peak window is required")
    features: dict[str, float] = {}
    tracked_by_prefix: dict[str, Any] = {}
    for window in windows:
        mask = np.abs(wavelength_nm - window.center_nm) <= window.half_width_nm
        if int(np.count_nonzero(mask)) < 5:
            raise ValueError(f"not enough samples near {window.candidate_id}")
        x = wavelength_nm[mask]
        current = spectrum[mask]
        baseline = baseline_spectrum[mask]
        tracked = track_peak(
            x,
            current,
            baseline,
            edge_fraction=window.edge_fraction,
            maximum_cross_correlation_shift_nm=window.maximum_cross_correlation_shift_nm,
            minimum_peak_height_counts=window.minimum_peak_height_counts,
            minimum_peak_snr=window.minimum_peak_snr,
            minimum_cross_correlation=window.minimum_cross_correlation,
            maximum_estimator_disagreement_pm=window.maximum_estimator_disagreement_pm,
            maximum_morphology_estimator_disagreement_pm=(
                window.maximum_morphology_estimator_disagreement_pm
            ),
        )
        prefix = window.candidate_id.lower()
        tracked_by_prefix[prefix] = tracked
        features[f"{prefix}_delta_centroid_pm"] = tracked.delta_centroid_pm
        features[f"{prefix}_delta_parabolic_pm"] = tracked.delta_parabolic_pm
        features[f"{prefix}_cross_correlation_shift_pm"] = tracked.cross_correlation_shift_pm
        features[f"{prefix}_quality_fused_shift_pm"] = tracked.quality_fused_shift_pm
        features[f"{prefix}_morphology_estimator_disagreement_pm"] = (
            tracked.morphology_estimator_disagreement_pm
        )
        features[f"{prefix}_quality_fused_shift_reliable"] = float(
            tracked.quality_fused_shift_reliable
        )
        features[f"{prefix}_cross_correlation_coefficient"] = tracked.cross_correlation_coefficient
        features[f"{prefix}_cross_correlation_reliable"] = float(
            tracked.cross_correlation_reliable
        )
        features[f"{prefix}_height_ratio"] = tracked.height_ratio
        features[f"{prefix}_area_ratio"] = tracked.area_ratio
        features[f"{prefix}_delta_fwhm_pm"] = tracked.delta_fwhm_pm
        features[f"{prefix}_delta_skewness"] = tracked.delta_skewness
        features[f"{prefix}_shape_correlation"] = tracked.shape_correlation
        features[f"{prefix}_normalized_shape_rmse"] = tracked.normalized_shape_rmse
        features[f"{prefix}_peak_snr"] = tracked.peak_snr
        features[f"{prefix}_baseline_peak_snr"] = tracked.baseline_peak_snr
        features[f"{prefix}_baseline_peak_valid"] = float(tracked.baseline_peak_valid)
        features[f"{prefix}_edge_margin_nm"] = tracked.edge_margin_nm
        features[f"{prefix}_estimator_spread_pm"] = tracked.estimator_spread_pm
        features[f"{prefix}_peak_valid"] = float(tracked.valid_peak)
        features[f"{prefix}_quality_flag_count"] = float(len(tracked.quality_flags))

    valid_shifts = [
        tracked.cross_correlation_shift_pm
        for tracked in tracked_by_prefix.values()
        if tracked.cross_correlation_reliable
        and np.isfinite(tracked.cross_correlation_shift_pm)
    ]
    valid_fused_shifts = [
        tracked.quality_fused_shift_pm
        for tracked in tracked_by_prefix.values()
        if tracked.valid_peak
        and tracked.quality_fused_shift_reliable
        and np.isfinite(tracked.quality_fused_shift_pm)
    ]
    minimum_common_mode_peaks = windows[0].minimum_common_mode_valid_peaks
    common_mode_shift_pm = (
        float(np.median(valid_shifts))
        if len(valid_shifts) >= minimum_common_mode_peaks
        else float("nan")
    )
    features["global_common_mode_shift_pm"] = common_mode_shift_pm
    features["global_common_mode_valid_peak_count"] = float(len(valid_shifts))
    fused_common_mode_shift_pm = (
        float(np.median(valid_fused_shifts))
        if len(valid_fused_shifts) >= minimum_common_mode_peaks
        else float("nan")
    )
    features["global_fused_common_mode_shift_pm"] = fused_common_mode_shift_pm
    features["global_fused_common_mode_valid_peak_count"] = float(
        len(valid_fused_shifts)
    )
    for prefix, tracked in tracked_by_prefix.items():
        features[f"{prefix}_common_mode_corrected_shift_pm"] = (
            tracked.cross_correlation_shift_pm - common_mode_shift_pm
            if np.isfinite(common_mode_shift_pm)
            else float("nan")
        )
        features[f"{prefix}_fused_common_mode_corrected_shift_pm"] = (
            tracked.quality_fused_shift_pm - fused_common_mode_shift_pm
            if np.isfinite(fused_common_mode_shift_pm)
            else float("nan")
        )

    normalized_current = spectrum / max(float(np.mean(spectrum)), EPSILON)
    normalized_baseline = baseline_spectrum / max(float(np.mean(baseline_spectrum)), EPSILON)
    residual = normalized_current - normalized_baseline
    features["global_intensity_ratio"] = float(np.mean(spectrum) / max(float(np.mean(baseline_spectrum)), EPSILON))
    features["global_shape_correlation"] = shape_correlation(spectrum, baseline_spectrum)
    features["global_normalized_residual_rms"] = float(np.sqrt(np.mean(residual**2)))
    features["global_normalized_residual_peak"] = float(np.max(np.abs(residual)))
    features["global_derivative_residual_energy"] = float(np.mean(np.diff(residual) ** 2))
    return features


def extract_feature_rows(
    segments: Iterable[SpectrumSegment],
    peak_windows: Iterable[PeakWindow],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if segment.baseline_spectrum is not None:
            baseline = segment.baseline_spectrum
        elif segment.phase == "no_contact":
            baseline = np.median(segment.intensity_counts, axis=0)
        else:
            raise ValueError(
                f"contact segment {segment.trial_id}/{segment.segment_id} has no baseline spectrum"
            )
        for frame_index, spectrum in enumerate(segment.intensity_counts):
            row: dict[str, Any] = {
                "trial_id": segment.trial_id,
                "segment_id": segment.segment_id,
                "label": segment.label,
                "phase": segment.phase,
                "frame_id": int(segment.frame_ids[frame_index]),
                "timestamp": float(segment.timestamps[frame_index]),
                "data_source": str(segment.metadata.get("data_source", "unknown")),
                "training_eligible": bool(segment.training_eligible),
                "segment_integrity_verified": bool(segment.integrity_verified),
                "segment_fingerprint_sha256": segment.segment_fingerprint_sha256 or "",
                "baseline_reference_segment_fingerprint_sha256": str(
                    segment.metadata.get(
                        "baseline_reference_segment_fingerprint_sha256", ""
                    )
                    or ""
                ),
                "backend_session_started_at_epoch": segment.metadata.get(
                    "backend_session_started_at_epoch", ""
                ),
                "device_id": str(segment.metadata.get("device_id", "") or ""),
                "integration_ms": segment.metadata.get("integration_ms", ""),
                "spectrum_peak_profile": str(
                    segment.metadata.get("spectrum_peak_profile", "") or ""
                ),
                "operator_attested_no_contact": bool(
                    segment.metadata.get("operator_attested_no_contact", False)
                ),
            }
            row.update(extract_frame_features(segment.wavelength_nm, spectrum, baseline, peak_windows))
            rows.append(row)
    return rows
