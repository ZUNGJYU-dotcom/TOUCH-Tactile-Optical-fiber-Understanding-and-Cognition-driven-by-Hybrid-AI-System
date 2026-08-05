"""Rich optical-physics features aligned to synchronized PX6D captures.

The deployed ordinary-FBG model intentionally uses a compact feature schema.
This module exposes the more complete peak tracking and coupling evidence for
offline, leakage-safe ablation before any deployment decision is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from .all_source_fusion import _latest_baseline
from .features import extract_frame_features, load_peak_windows
from .px6d_session_dataset import (
    _load_session_frame_matrix,
    discover_sessions,
    filter_session_descriptors,
    split_primary_and_challenge_sessions,
)


CANDIDATE_PREFIXES = tuple(f"fbg{index:02d}" for index in range(1, 10))

# Three serial FBGs on each fibre. Downstream bending can affect upstream peaks.
SAME_FIBRE_GROUPS = {
    "fibre_1": ("fbg01", "fbg02", "fbg03"),
    "fibre_2": ("fbg04", "fbg05", "fbg06"),
    "fibre_3": ("fbg07", "fbg08", "fbg09"),
}

# Physical display rows: P11 P21 P31 / P12 P22 P32 / P13 P23 P33.
SPATIAL_ROW_GROUPS = {
    "row_1": ("fbg01", "fbg04", "fbg07"),
    "row_2": ("fbg02", "fbg05", "fbg08"),
    "row_3": ("fbg03", "fbg06", "fbg09"),
}

COUPLING_METRICS = (
    "log_area_ratio",
    "log_height_ratio",
    "quality_fused_shift_pm",
    "fused_common_mode_corrected_shift_pm",
    "delta_fwhm_pm",
    "delta_skewness",
    "normalized_shape_rmse",
)


@dataclass(frozen=True)
class RichFeatureCache:
    features: np.ndarray
    feature_names: np.ndarray
    group_id: np.ndarray
    sample_index: np.ndarray


def _finite_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.max(finite) - np.min(finite)),
        float(finite[-1] - finite[0]),
    )


def add_coupling_features(features: Mapping[str, float]) -> dict[str, float]:
    """Derive label-free coupling features from the nine tracked FBG peaks."""

    values = {str(key): float(value) for key, value in features.items()}
    for prefix in CANDIDATE_PREFIXES:
        area = max(float(values.get(f"{prefix}_area_ratio", np.nan)), 1.0e-12)
        height = max(
            float(values.get(f"{prefix}_height_ratio", np.nan)), 1.0e-12
        )
        values[f"{prefix}_log_area_ratio"] = (
            float(np.log(area)) if np.isfinite(area) else float("nan")
        )
        values[f"{prefix}_log_height_ratio"] = (
            float(np.log(height)) if np.isfinite(height) else float("nan")
        )

    for metric in COUPLING_METRICS:
        metric_values = np.asarray(
            [values.get(f"{prefix}_{metric}", np.nan) for prefix in CANDIDATE_PREFIXES],
            dtype=float,
        )
        finite = metric_values[np.isfinite(metric_values)]
        center = float(np.median(finite)) if len(finite) else 0.0
        mean, std, spread, end_to_end = _finite_summary(metric_values)
        stem = f"coupling_{metric}"
        values[f"{stem}_global_mean"] = mean
        values[f"{stem}_global_median"] = center
        values[f"{stem}_global_std"] = std
        values[f"{stem}_global_range"] = spread
        values[f"{stem}_global_end_to_end"] = end_to_end
        for prefix, value in zip(CANDIDATE_PREFIXES, metric_values):
            values[f"{stem}_{prefix}_residual"] = (
                float(value - center) if np.isfinite(value) else 0.0
            )

        for group_kind, groups in (
            ("same_fibre", SAME_FIBRE_GROUPS),
            ("spatial_row", SPATIAL_ROW_GROUPS),
        ):
            for group_id, prefixes in groups.items():
                group_values = np.asarray(
                    [values.get(f"{prefix}_{metric}", np.nan) for prefix in prefixes],
                    dtype=float,
                )
                group_mean, group_std, group_range, group_gradient = _finite_summary(
                    group_values
                )
                group_stem = f"{stem}_{group_kind}_{group_id}"
                values[f"{group_stem}_mean"] = group_mean
                values[f"{group_stem}_std"] = group_std
                values[f"{group_stem}_range"] = group_range
                values[f"{group_stem}_gradient"] = group_gradient

    nonfinite_count = sum(not np.isfinite(value) for value in values.values())
    values["global_nonfinite_physics_feature_count"] = float(nonfinite_count)
    return {
        key: float(value) if np.isfinite(value) else 0.0
        for key, value in values.items()
    }


def extract_rich_frame_features(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    baseline_spectrum: np.ndarray,
    peak_windows: Any,
) -> dict[str, float]:
    """Extract peak tracking, morphology, quality, and coupling features."""

    base = extract_frame_features(
        np.asarray(wavelength_nm, dtype=float),
        np.asarray(spectrum, dtype=float),
        np.asarray(baseline_spectrum, dtype=float),
        peak_windows,
    )
    return add_coupling_features(base)


def _load_training_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_aligned_rich_feature_cache(
    *,
    group_id: np.ndarray,
    sample_index: np.ndarray,
    capture_root: Path,
    channel_config_path: Path,
    training_config_path: Path,
    qa_summary_path: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> RichFeatureCache:
    """Rebuild rich features for rows already frozen by the formal dataset."""

    groups = np.asarray(group_id).astype(str)
    indices = np.asarray(sample_index).astype(int)
    if groups.shape != indices.shape or groups.ndim != 1:
        raise ValueError("group_id and sample_index must be aligned vectors")
    if len(set(zip(groups.tolist(), indices.tolist()))) != len(groups):
        raise ValueError("aligned frame keys must be unique")

    training_config = _load_training_config(training_config_path)
    data_config = dict(training_config.get("data") or {})
    baseline_config = dict(training_config.get("baseline") or {})
    label_config = dict(training_config.get("labels") or {})
    expected_points = int(data_config.get("expected_spectrum_points", 512))
    no_contact_max_force_n = float(label_config.get("no_contact_max_force_n", 0.03))
    descriptors = discover_sessions(capture_root, qa_summary_path)
    descriptors = filter_session_descriptors(descriptors, data_config)
    primary, _ = split_primary_and_challenge_sessions(
        descriptors,
        dict(data_config.get("primary_selection") or {}),
    )
    descriptor_by_id = {descriptor.session_id: descriptor for descriptor in primary}
    missing_groups = sorted(set(groups.tolist()).difference(descriptor_by_id))
    if missing_groups:
        raise ValueError(
            "formal aligned rows are missing from the selected primary sessions: "
            + ", ".join(missing_groups[:5])
        )

    peak_windows = load_peak_windows(channel_config_path)
    rows: list[np.ndarray | None] = [None] * len(groups)
    feature_names: tuple[str, ...] | None = None
    ordered_groups = list(dict.fromkeys(groups.tolist()))
    for group_number, current_group in enumerate(ordered_groups, start=1):
        descriptor = descriptor_by_id[current_group]
        summary, wavelength_nm, intensity = _load_session_frame_matrix(
            descriptor, expected_points
        )
        force_fz_n = summary["force_fz_n"].to_numpy(dtype=float)
        baseline, _, _ = _latest_baseline(
            force_fz_n,
            intensity,
            baseline_config,
            no_contact_max_force_n,
        )
        capture_lookup = {
            int(capture_index): row_index
            for row_index, capture_index in enumerate(
                summary["capture_index"].to_numpy(dtype=int)
            )
        }
        output_positions = np.flatnonzero(groups == current_group)
        for output_position in output_positions:
            capture_index = int(indices[output_position])
            if capture_index not in capture_lookup:
                raise ValueError(
                    f"{current_group} is missing capture_index {capture_index}"
                )
            raw = extract_rich_frame_features(
                wavelength_nm,
                intensity[capture_lookup[capture_index]],
                baseline,
                peak_windows,
            )
            current_names = tuple(raw)
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError("rich optical feature schema changed between frames")
            rows[output_position] = np.asarray(
                [raw[name] for name in feature_names], dtype=np.float32
            )
        if progress is not None:
            progress(group_number, len(ordered_groups), current_group)

    if feature_names is None or any(row is None for row in rows):
        raise ValueError("rich optical feature extraction did not cover every aligned frame")
    matrix = np.stack(rows).astype(np.float32, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rich optical feature matrix contains NaN or infinite values")
    return RichFeatureCache(
        features=matrix,
        feature_names=np.asarray(feature_names),
        group_id=groups,
        sample_index=indices,
    )


def save_rich_feature_cache(cache: RichFeatureCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("rich_optical_feature_cache_v1"),
        features=cache.features,
        feature_names=cache.feature_names,
        group_id=cache.group_id,
        sample_index=cache.sample_index,
    )


def load_rich_feature_cache(
    path: Path,
    *,
    expected_group_id: np.ndarray | None = None,
    expected_sample_index: np.ndarray | None = None,
) -> RichFeatureCache:
    with np.load(path, allow_pickle=False) as payload:
        schema = str(payload["schema_version"].item())
        if schema != "rich_optical_feature_cache_v1":
            raise ValueError(f"unsupported rich feature cache schema: {schema}")
        cache = RichFeatureCache(
            features=payload["features"].astype(np.float32),
            feature_names=payload["feature_names"].astype(str),
            group_id=payload["group_id"].astype(str),
            sample_index=payload["sample_index"].astype(int),
        )
    if expected_group_id is not None and not np.array_equal(
        cache.group_id, np.asarray(expected_group_id).astype(str)
    ):
        raise ValueError("rich feature cache group order does not match the dataset")
    if expected_sample_index is not None and not np.array_equal(
        cache.sample_index, np.asarray(expected_sample_index).astype(int)
    ):
        raise ValueError("rich feature cache frame order does not match the dataset")
    if cache.features.shape != (len(cache.group_id), len(cache.feature_names)):
        raise ValueError("rich feature cache matrix and metadata dimensions disagree")
    if not np.all(np.isfinite(cache.features)):
        raise ValueError("rich feature cache contains NaN or infinite values")
    return cache


__all__ = [
    "CANDIDATE_PREFIXES",
    "COUPLING_METRICS",
    "RichFeatureCache",
    "SAME_FIBRE_GROUPS",
    "SPATIAL_ROW_GROUPS",
    "add_coupling_features",
    "build_aligned_rich_feature_cache",
    "extract_rich_frame_features",
    "load_rich_feature_cache",
    "save_rich_feature_cache",
]
