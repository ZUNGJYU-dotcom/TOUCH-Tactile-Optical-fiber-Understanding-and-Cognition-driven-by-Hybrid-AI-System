"""Tactile-information proxies derived from synchronized optical spectra.

The functions in this module are descriptive and calibration-aware.  They do
not convert optical coupling into independent force pixels, and they do not
claim physical contact geometry without an external spatial reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CHANNEL_ORDER = (
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
)
CHANNEL_COORDINATES = np.asarray(
    [
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
    dtype=float,
)


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan")
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _feature_column(
    features: np.ndarray,
    feature_names: np.ndarray,
    name: str,
) -> np.ndarray:
    matches = np.flatnonzero(np.asarray(feature_names).astype(str) == name)
    if len(matches) != 1:
        raise ValueError(f"expected one feature named {name!r}, found {len(matches)}")
    return np.asarray(features[:, int(matches[0])], dtype=float)


def build_noise_normalized_channel_response(
    *,
    features: np.ndarray,
    feature_names: np.ndarray,
    no_contact_mask: np.ndarray,
    contact_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fuse shift, area, and shape evidence after channel-wise normalization.

    Per-channel baselines and active scales prevent naturally sensitive peaks
    from dominating merely because their raw units or amplitudes are larger.
    The result is a dimensionless optical response proxy, not force.
    """

    values = np.asarray(features, dtype=float)
    idle = np.asarray(no_contact_mask, dtype=bool)
    active = np.asarray(contact_mask, dtype=bool)
    if values.shape[0] != len(idle) or len(idle) != len(active):
        raise ValueError("feature rows and masks must align")
    if not np.any(idle) or not np.any(active):
        raise ValueError("both no-contact and contact rows are required")

    response = np.zeros((len(values), len(CHANNEL_ORDER)), dtype=float)
    scale_rows: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(CHANNEL_ORDER, start=1):
        prefix = f"fbg{channel_index:02d}"
        component_specs = (
            ("shift", f"{prefix}_fused_common_mode_corrected_shift_pm"),
            ("area", f"{prefix}_log_area_ratio"),
            ("shape", f"{prefix}_normalized_shape_rmse"),
        )
        normalized_components: list[np.ndarray] = []
        for component_id, feature_name in component_specs:
            component = np.abs(_feature_column(values, feature_names, feature_name))
            idle_values = component[idle & np.isfinite(component)]
            active_values = component[active & np.isfinite(component)]
            idle_center = float(np.median(idle_values))
            idle_noise = robust_mad(idle_values)
            if not np.isfinite(idle_noise):
                idle_noise = 0.0
            active_q95 = float(np.quantile(active_values, 0.95))
            denominator = max(
                active_q95 - idle_center,
                6.0 * idle_noise,
                1.0e-9,
            )
            normalized = np.clip(
                (np.nan_to_num(component, nan=idle_center) - idle_center)
                / denominator,
                0.0,
                1.5,
            )
            normalized_components.append(normalized)
            scale_rows.append(
                {
                    "channel_id": channel,
                    "component": component_id,
                    "source_feature": feature_name,
                    "idle_center": idle_center,
                    "idle_robust_noise": idle_noise,
                    "active_q95": active_q95,
                    "normalization_denominator": denominator,
                }
            )
        response[:, channel_index - 1] = np.median(
            np.column_stack(normalized_components), axis=1
        )
    return response, scale_rows


def contact_patch_moments(
    weights: np.ndarray,
    *,
    coordinates: np.ndarray = CHANNEL_COORDINATES,
    active_fraction_of_peak: float = 0.25,
) -> dict[str, float | int]:
    """Return center/spread/orientation of a nine-channel optical proxy."""

    values = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    coords = np.asarray(coordinates, dtype=float)
    if values.shape != (len(coords),):
        raise ValueError("weights must contain one value per coordinate")
    total = float(np.sum(values))
    peak = float(np.max(values)) if len(values) else 0.0
    if total <= 1.0e-12 or peak <= 1.0e-12:
        return {
            "center_x": float("nan"),
            "center_y": float("nan"),
            "spread_major": float("nan"),
            "spread_minor": float("nan"),
            "eccentricity": float("nan"),
            "orientation_deg": float("nan"),
            "active_channel_count": 0,
            "peak_response": peak,
        }
    center = np.sum(coords * values[:, None], axis=0) / total
    centered = coords - center
    covariance = (centered * values[:, None]).T @ centered / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_variance = max(float(eigenvalues[order[0]]), 0.0)
    minor_variance = max(float(eigenvalues[order[1]]), 0.0)
    major_vector = eigenvectors[:, order[0]]
    eccentricity = float(
        np.sqrt(max(0.0, 1.0 - minor_variance / max(major_variance, 1.0e-12)))
    )
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "spread_major": float(np.sqrt(major_variance)),
        "spread_minor": float(np.sqrt(minor_variance)),
        "eccentricity": eccentricity,
        "orientation_deg": float(
            np.degrees(np.arctan2(major_vector[1], major_vector[0]))
        ),
        "active_channel_count": int(
            np.sum(values >= active_fraction_of_peak * peak)
        ),
        "peak_response": peak,
    }


@dataclass(frozen=True)
class RecoveryEvent:
    session_id: str
    release_capture_index: int
    release_time_sec: float
    baseline_threshold: float
    response_at_release: float
    recovered: bool
    recovery_time_sec: float
    stable_frame_count: int


def find_release_recovery_events(
    *,
    group_id: np.ndarray,
    capture_index: np.ndarray,
    elapsed_time_sec: np.ndarray,
    contact_target: np.ndarray,
    response_score: np.ndarray,
    stable_frames: int = 3,
) -> list[RecoveryEvent]:
    """Measure optical recovery after each contiguous contact run."""

    groups = np.asarray(group_id).astype(str)
    captures = np.asarray(capture_index, dtype=int)
    elapsed = np.asarray(elapsed_time_sec, dtype=float)
    contact = np.asarray(contact_target, dtype=int)
    score = np.asarray(response_score, dtype=float)
    if not (groups.shape == captures.shape == elapsed.shape == contact.shape == score.shape):
        raise ValueError("recovery inputs must align")
    required_stable = max(1, int(stable_frames))
    events: list[RecoveryEvent] = []
    for group in dict.fromkeys(groups.tolist()):
        selected = np.flatnonzero(groups == group)
        order = selected[np.argsort(captures[selected], kind="stable")]
        current_contact = contact[order]
        current_score = score[order]
        current_time = elapsed[order]
        contact_positions = np.flatnonzero(current_contact == 1)
        first_contact = int(contact_positions[0]) if len(contact_positions) else len(order)
        baseline_positions = np.arange(first_contact)[
            (current_contact[:first_contact] == 0)
            & np.isfinite(current_score[:first_contact])
        ]
        if len(baseline_positions) < required_stable:
            baseline_positions = np.flatnonzero(
                (current_contact == 0) & np.isfinite(current_score)
            )
        idle_values = current_score[baseline_positions]
        if not len(idle_values):
            continue
        idle_center = float(np.median(idle_values))
        idle_noise = robust_mad(idle_values)
        if not np.isfinite(idle_noise):
            idle_noise = 0.0
        threshold = idle_center + max(3.0 * idle_noise, 0.02)
        for position in range(1, len(order)):
            if current_contact[position - 1] != 1 or current_contact[position] == 1:
                continue
            recovered_position: int | None = None
            for candidate in range(position, len(order) - required_stable + 1):
                window = current_score[candidate : candidate + required_stable]
                labels = current_contact[candidate : candidate + required_stable]
                if (
                    np.all(np.isfinite(window))
                    and np.all(window <= threshold)
                    and np.all(labels != 1)
                ):
                    recovered_position = candidate
                    break
            recovered = recovered_position is not None
            recovery_time = (
                float(current_time[recovered_position] - current_time[position - 1])
                if recovered_position is not None
                else float("nan")
            )
            events.append(
                RecoveryEvent(
                    session_id=str(group),
                    release_capture_index=int(captures[order[position]]),
                    release_time_sec=float(current_time[position]),
                    baseline_threshold=threshold,
                    response_at_release=float(current_score[position]),
                    recovered=recovered,
                    recovery_time_sec=recovery_time,
                    stable_frame_count=required_stable,
                )
            )
    return events


def cosine_similarity_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = norms[:, None] * norms[None, :]
    similarity = np.divide(
        centered @ centered.T,
        denominator,
        out=np.full((len(matrix), len(matrix)), np.nan, dtype=float),
        where=denominator > 1.0e-12,
    )
    return similarity


def session_baseline_metadata_mask(feature_names: np.ndarray) -> np.ndarray:
    """Select baseline-quality fields that can encode capture-session identity.

    Baseline SNR and validity are valuable QA evidence, but they do not describe
    the current tactile response. Keeping them out of recognition views avoids
    rewarding a model for identifying a capture session instead of contact.
    """

    names = np.asarray(feature_names).astype(str)
    return np.asarray(
        [
            ("_baseline_peak_snr" in name)
            or ("_baseline_peak_valid" in name)
            for name in names
        ],
        dtype=bool,
    )


__all__ = [
    "CHANNEL_COORDINATES",
    "CHANNEL_ORDER",
    "RecoveryEvent",
    "build_noise_normalized_channel_response",
    "contact_patch_moments",
    "cosine_similarity_rows",
    "find_release_recovery_events",
    "robust_mad",
    "session_baseline_metadata_mask",
]
