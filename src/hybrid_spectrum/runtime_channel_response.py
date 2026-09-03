"""Build honest per-frame spatial evidence for the TOUCH runtime.

The operator surface is an inferred contact-probability surface.  The raw
nine-peak response is kept separately because the provisional FBG-to-position
mapping is coupled and is not an independent force-pixel measurement.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .features import PeakWindow


EPSILON = 1.0e-9
OBSERVED_ACTIVE_THRESHOLD = 0.055
OBSERVED_RESPONSE_SCALES = {
    "centroid_shift_pm": 500.0,
    "log_area_ratio": 0.5,
    "log_height_ratio": 0.5,
    "shape_rmse": 0.25,
}


def _finite_unit_interval(value: Any) -> float:
    """Return a JSON-safe probability/amplitude value."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return float(np.clip(numeric, 0.0, 1.0))


def _json_grid(
    values: Mapping[str, float],
    display_rows: Sequence[Sequence[str]],
) -> list[list[float]]:
    return [
        [float(values.get(channel_id, 0.0)) for channel_id in row]
        for row in display_rows
    ]


def build_observed_coupled_spectral_response(
    feature_values: Iterable[float],
    feature_names: Sequence[str],
    peak_windows: Iterable[PeakWindow],
    display_rows: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Return baseline-relative evidence from the current nine FBG windows.

    ``response_ratio`` is a fixed-scale display normalization of measured
    baseline-relative spectral changes.  It is deliberately not normalized by
    the strongest channel in the current frame, so a quiet frame remains quiet.
    """

    values = np.asarray(tuple(feature_values), dtype=float)
    if values.shape != (len(feature_names),):
        raise ValueError("feature values and names must have the same length")
    lookup = {
        str(name): float(value) if np.isfinite(value) else 0.0
        for name, value in zip(feature_names, values, strict=True)
    }

    channels: list[dict[str, Any]] = []
    response_by_channel: dict[str, float] = {}
    for window in peak_windows:
        prefix = window.candidate_id.lower()
        centroid_shift_pm = lookup[f"{prefix}_centroid_shift_pm"]
        log_area_ratio = lookup[f"{prefix}_log_area_ratio"]
        log_height_ratio = lookup[f"{prefix}_log_height_ratio"]
        shape_rmse = lookup[f"{prefix}_shape_rmse"]
        scaled = np.asarray(
            [
                abs(centroid_shift_pm)
                / OBSERVED_RESPONSE_SCALES["centroid_shift_pm"],
                abs(log_area_ratio)
                / OBSERVED_RESPONSE_SCALES["log_area_ratio"],
                abs(log_height_ratio)
                / OBSERVED_RESPONSE_SCALES["log_height_ratio"],
                abs(shape_rmse) / OBSERVED_RESPONSE_SCALES["shape_rmse"],
            ],
            dtype=float,
        )
        evidence_score = float(np.sqrt(np.mean(np.square(scaled))))
        response_ratio = float(np.clip(evidence_score, 0.0, 1.0))
        channel_id = str(window.provisional_channel_id)
        response_by_channel[channel_id] = response_ratio
        channels.append(
            {
                "candidate_id": str(window.candidate_id),
                "provisional_channel_id": channel_id,
                "center_wavelength_nm": float(window.center_nm),
                "centroid_shift_pm": float(centroid_shift_pm),
                "log_area_ratio": float(log_area_ratio),
                "log_height_ratio": float(log_height_ratio),
                "shape_rmse": float(shape_rmse),
                "evidence_score": evidence_score,
                "response_ratio": response_ratio,
            }
        )

    dominant_candidate = max(
        response_by_channel,
        key=response_by_channel.get,
        default=None,
    )
    peak_response = float(max(response_by_channel.values(), default=0.0))
    dominant = (
        dominant_candidate
        if peak_response >= OBSERVED_ACTIVE_THRESHOLD
        else None
    )
    responding_channel_ids = [
        channel_id
        for channel_id, response in response_by_channel.items()
        if response >= OBSERVED_ACTIVE_THRESHOLD
    ]
    return {
        "kind": "observed_coupled_spectral_response",
        "mapping_status": "provisional_fbg_to_position_mapping",
        "semantics": (
            "baseline_relative_coupled_spectral_evidence_not_independent_force_pixels"
        ),
        "normalization": "fixed_display_scales_not_force_calibrated",
        "normalization_scales": dict(OBSERVED_RESPONSE_SCALES),
        "dominant_channel": dominant,
        "peak_response": peak_response,
        "responding_channel_ids": responding_channel_ids,
        "response_grid": _json_grid(response_by_channel, display_rows),
        "channels": channels,
    }


def summarize_coupled_contact_signature(
    observed_response: Mapping[str, Any],
    *,
    low_response_threshold: float,
    nominal_response_threshold: float,
    minimum_low_response_channels: int,
    minimum_nominal_response_channels: int,
    expected_channel_count: int = 9,
) -> dict[str, Any]:
    """Summarize whether a frame contains a distributed FBG response.

    A physical press is represented by the response pattern across all nine
    gratings. A large value from one candidate is not sufficient evidence on
    its own because an isolated peak-tracking disturbance can look similar.
    """

    if not 0.0 <= low_response_threshold <= nominal_response_threshold:
        raise ValueError("coupled-response thresholds are invalid")
    if expected_channel_count < 1:
        raise ValueError("expected channel count must be positive")
    if not 1 <= minimum_low_response_channels <= expected_channel_count:
        raise ValueError("minimum low-response channel count is invalid")
    if not 1 <= minimum_nominal_response_channels <= expected_channel_count:
        raise ValueError("minimum nominal-response channel count is invalid")

    raw_channels = observed_response.get("channels")
    channels = raw_channels if isinstance(raw_channels, list) else []
    scores: list[float] = []
    channel_ids: list[str] = []
    for channel in channels:
        if not isinstance(channel, Mapping):
            continue
        try:
            score = float(channel.get("evidence_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        scores.append(max(0.0, score))
        channel_ids.append(str(channel.get("provisional_channel_id") or ""))

    low_response_channel_ids = [
        channel_id
        for channel_id, score in zip(channel_ids, scores, strict=True)
        if score >= low_response_threshold
    ]
    nominal_response_channel_ids = [
        channel_id
        for channel_id, score in zip(channel_ids, scores, strict=True)
        if score >= nominal_response_threshold
    ]
    ordered_scores = sorted(scores, reverse=True)

    def ranked_score(index: int) -> float:
        return float(ordered_scores[index]) if len(ordered_scores) > index else 0.0

    all_channels_present = bool(
        len(scores) == expected_channel_count
        and len(set(channel_ids)) == expected_channel_count
        and all(channel_ids)
    )
    multichannel_pattern = bool(
        all_channels_present
        and len(low_response_channel_ids) >= minimum_low_response_channels
        and len(nominal_response_channel_ids)
        >= minimum_nominal_response_channels
    )
    score_array = np.asarray(scores, dtype=float)
    return {
        "kind": "nine_fbg_joint_contact_signature",
        "semantics": (
            "all_nine_gratings_evaluated_jointly_not_single_peak_activation"
        ),
        "expected_channel_count": int(expected_channel_count),
        "observed_channel_count": int(len(scores)),
        "all_channels_present": all_channels_present,
        "low_response_threshold": float(low_response_threshold),
        "nominal_response_threshold": float(nominal_response_threshold),
        "minimum_low_response_channels": int(minimum_low_response_channels),
        "minimum_nominal_response_channels": int(
            minimum_nominal_response_channels
        ),
        "low_response_channel_count": int(len(low_response_channel_ids)),
        "nominal_response_channel_count": int(
            len(nominal_response_channel_ids)
        ),
        "low_response_channel_ids": low_response_channel_ids,
        "nominal_response_channel_ids": nominal_response_channel_ids,
        "peak_response": ranked_score(0),
        "second_response": ranked_score(1),
        "third_response": ranked_score(2),
        "mean_response": (
            float(np.mean(score_array)) if score_array.size else 0.0
        ),
        "rms_response": (
            float(np.sqrt(np.mean(np.square(score_array))))
            if score_array.size
            else 0.0
        ),
        "multichannel_pattern": multichannel_pattern,
    }


def build_inferred_contact_probability_surface(
    probabilities: Mapping[str, float],
    *,
    accepted_position_id: str | None,
    deformation: float,
    active: bool,
    position_order: Sequence[str],
    display_rows: Sequence[Sequence[str]],
    position_coordinates: Mapping[str, tuple[float, float]],
    probability_source: str = "trained_position_predict_proba",
    probability_semantics: str = (
        "current_frame_position_posterior_scaled_by_optical_force_"
        "not_measured_pressure"
    ),
    active_threshold: float = 0.055,
) -> dict[str, Any]:
    """Scale the current classifier posterior into a spatial surface.

    No Gaussian or hand-authored spatial spreading is introduced.  Relative
    pixel heights come directly from the current position probabilities; the
    optical force estimate controls only the common deformation amplitude.
    """

    cleaned = {
        channel_id: _finite_unit_interval(
            probabilities.get(channel_id, 0.0)
        )
        for channel_id in position_order
    }
    probability_sum = float(sum(cleaned.values()))
    map_source = str(probability_source)
    map_semantics = str(probability_semantics)
    if probability_sum <= EPSILON and accepted_position_id in position_coordinates:
        cleaned = {
            channel_id: 1.0 if channel_id == accepted_position_id else 0.0
            for channel_id in position_order
        }
        probability_sum = 1.0
        map_source = "accepted_label_fallback_without_predict_proba"
        map_semantics = (
            "accepted_position_fallback_scaled_by_optical_force_"
            "not_measured_pressure"
        )
    if probability_sum > EPSILON:
        cleaned = {
            channel_id: value / probability_sum
            for channel_id, value in cleaned.items()
        }

    dominant_channel = max(cleaned, key=cleaned.get, default=None)
    maximum_probability = float(max(cleaned.values(), default=0.0))
    if not active or maximum_probability <= EPSILON:
        surface_values = {channel_id: 0.0 for channel_id in position_order}
    else:
        amplitude = _finite_unit_interval(deformation)
        surface_values = {
            channel_id: amplitude * probability / maximum_probability
            for channel_id, probability in cleaned.items()
        }

    surface_grid = _json_grid(surface_values, display_rows)
    probability_grid = _json_grid(cleaned, display_rows)
    flat = np.asarray(surface_grid, dtype=float)
    weights = np.asarray(
        [
            surface_values[channel_id]
            for row in display_rows
            for channel_id in row
        ],
        dtype=float,
    )
    points = [
        position_coordinates[channel_id]
        for row in display_rows
        for channel_id in row
    ]
    weight_sum = float(np.sum(weights))
    if weight_sum <= EPSILON:
        centroid_x = 0.0
        centroid_y = 0.0
        spread = 0.0
    else:
        centroid_x = float(
            sum(weight * point[0] for weight, point in zip(weights, points))
            / weight_sum
        )
        centroid_y = float(
            sum(weight * point[1] for weight, point in zip(weights, points))
            / weight_sum
        )
        spread = float(
            np.sqrt(
                sum(
                    weight
                    * (
                        (point[0] - centroid_x) ** 2
                        + (point[1] - centroid_y) ** 2
                    )
                    for weight, point in zip(weights, points)
                )
                / weight_sum
            )
        )

    responding_ids = [
        channel_id
        for channel_id in position_order
        if surface_values[channel_id] >= active_threshold
    ]
    return {
        "map_kind": "inferred_contact_probability",
        "map_source": map_source,
        "map_semantics": map_semantics,
        "accepted_position_id": accepted_position_id,
        "dominant_channel": dominant_channel if active else None,
        "probability_grid": probability_grid,
        "probabilities": cleaned,
        "surface_grid": surface_grid,
        "surface_metrics": {
            "surface_peak": float(np.max(flat)),
            "surface_mean": float(np.mean(flat)),
            "surface_area_active": float(np.mean(flat >= active_threshold)),
            "surface_centroid_x": centroid_x,
            "surface_centroid_y": centroid_y,
            "surface_spread": spread,
            "dominant_channel": dominant_channel if active else None,
            "responding_channel_ids": responding_ids,
        },
    }


__all__ = [
    "OBSERVED_ACTIVE_THRESHOLD",
    "OBSERVED_RESPONSE_SCALES",
    "build_inferred_contact_probability_surface",
    "build_observed_coupled_spectral_response",
    "summarize_coupled_contact_signature",
]
