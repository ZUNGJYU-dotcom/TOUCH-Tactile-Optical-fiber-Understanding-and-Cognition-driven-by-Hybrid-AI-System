"""Force-invariant nine-FBG spatial fingerprints for position recognition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


EPSILON = 1.0e-8
CHANNEL_IDS = tuple(f"fbg{index:02d}" for index in range(1, 10))
SPATIAL_FEATURE_FAMILIES = (
    "common_mode_corrected_shift_pm",
    "fused_common_mode_corrected_shift_pm",
    "delta_centroid_pm",
    "delta_parabolic_pm",
    "cross_correlation_shift_pm",
    "area_ratio",
    "height_ratio",
    "normalized_shape_rmse",
    "shape_correlation",
    "delta_fwhm_pm",
    "delta_skewness",
)


def _transform_family(values: np.ndarray, family: str) -> np.ndarray:
    result = np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if family in {"area_ratio", "height_ratio"}:
        result = np.log(np.clip(result, 1.0e-5, 1.0e5))
    elif family == "shape_correlation":
        result = 1.0 - np.clip(result, -1.0, 1.0)
    return result


def spatial_fingerprint_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for family in SPATIAL_FEATURE_FAMILIES:
        for representation in ("signed", "magnitude"):
            names.extend(
                f"spatial_{family}_{representation}_{channel_id}"
                for channel_id in CHANNEL_IDS
            )
    return tuple(names)


def spatial_fingerprint_from_engineered(
    engineered: Mapping[str, float],
) -> dict[str, float]:
    """Normalize each response family across the nine FBG channels.

    The per-snapshot normalization suppresses global response amplitude and
    unequal session gain.  Signed and magnitude views retain direction while
    emphasizing the relative spatial coupling pattern used for localization.
    """

    output: dict[str, float] = {}
    for family in SPATIAL_FEATURE_FAMILIES:
        values = _transform_family(
            np.asarray(
                [float(engineered.get(f"{channel_id}_{family}", 0.0)) for channel_id in CHANNEL_IDS],
                dtype=float,
            ),
            family,
        )
        scale = max(float(np.max(np.abs(values))), EPSILON)
        signed = values / scale
        magnitude = np.abs(values) / scale
        for channel_id, value in zip(CHANNEL_IDS, signed, strict=True):
            output[f"spatial_{family}_signed_{channel_id}"] = float(value)
        for channel_id, value in zip(CHANNEL_IDS, magnitude, strict=True):
            output[f"spatial_{family}_magnitude_{channel_id}"] = float(value)
    return output


def build_spatial_fingerprint_matrix(
    engineered_matrix: np.ndarray,
    engineered_columns: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = tuple(engineered_columns)
    rows = [
        spatial_fingerprint_from_engineered(
            {column: float(value) for column, value in zip(columns, row, strict=True)}
        )
        for row in np.asarray(engineered_matrix, dtype=float)
    ]
    feature_names = spatial_fingerprint_feature_names()
    matrix = np.asarray(
        [[row[name] for name in feature_names] for row in rows],
        dtype=float,
    )
    return matrix, feature_names


__all__ = [
    "CHANNEL_IDS",
    "SPATIAL_FEATURE_FAMILIES",
    "build_spatial_fingerprint_matrix",
    "spatial_fingerprint_feature_names",
    "spatial_fingerprint_from_engineered",
]
