"""Build one provenance-rich optical dataset from all ordinary-FBG sources."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from .dynamic_sequence_dataset import extract_baseline_relative_frame_features
from .dynamic_temporal_features import (
    SUMMARY_FEATURE_BLOCK_ORDER,
    temporal_summary_features,
)
from .features import load_peak_windows
from .px6d_session_dataset import (
    SessionDescriptor,
    _load_session_frame_matrix,
    _load_session_recorded_baseline,
    assign_session_folds,
    discover_sessions,
    filter_session_descriptors,
    session_has_force_reference,
    split_primary_and_challenge_sessions,
)
from .sense_static_dataset import (
    baseline_for_record,
    build_static_feature_dataset,
    load_training_config,
    parse_sense_spectrum_csv,
    resample_spectrum,
)


POSITION_ORDER = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)
CONTEXT_FEATURE_NAMES = (
    "temporal_frame_count_log1p",
    "temporal_duration_sec",
    "temporal_context_available",
)


@dataclass(frozen=True)
class AllSourceFusionDataset:
    """Fixed-width optical features plus masked labels and source provenance."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    contact_target: np.ndarray
    position_target: np.ndarray
    force_fz_n: np.ndarray
    contact_mask: np.ndarray
    position_mask: np.ndarray
    force_mask: np.ndarray
    formal_test_eligible: np.ndarray
    fold_id: np.ndarray
    source_id: np.ndarray
    source_role: np.ndarray
    group_id: np.ndarray
    file_id: np.ndarray
    sample_index: np.ndarray
    elapsed_time_sec: np.ndarray
    temporal_frame_count: np.ndarray
    force_out_of_range: np.ndarray
    manifest: pd.DataFrame
    source_inventory: tuple[dict[str, Any], ...]


def load_fusion_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = (
        "paths",
        "source_policy",
        "force_calibration",
        "labels",
        "temporal_features",
        "evaluation",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("fusion config is missing: " + ", ".join(missing))
    return config


def resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _summary_feature_names(
    frame_feature_names: Iterable[str],
) -> tuple[str, ...]:
    names = tuple(str(value) for value in frame_feature_names)
    return tuple(
        f"{block}__{name}"
        for block in SUMMARY_FEATURE_BLOCK_ORDER
        for name in names
    ) + CONTEXT_FEATURE_NAMES


def summarize_windows(
    windows: np.ndarray,
    frame_counts: np.ndarray,
    durations_sec: np.ndarray,
    temporal_context_available: np.ndarray,
) -> np.ndarray:
    """Summarize optical-only windows and append non-label context metadata."""

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("windows must have shape [samples, time, optical_features]")
    summary = temporal_summary_features(values).astype(np.float32)
    context = np.column_stack(
        (
            np.log1p(np.asarray(frame_counts, dtype=np.float32)),
            np.asarray(durations_sec, dtype=np.float32),
            np.asarray(temporal_context_available, dtype=np.float32),
        )
    )
    return np.column_stack((summary, context)).astype(np.float32)


def summarize_trailing_sequence(
    frame_features: np.ndarray,
    elapsed_time_sec: np.ndarray,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create causal trailing summaries for every frame in one capture."""

    values = np.asarray(frame_features, dtype=np.float32)
    elapsed = np.asarray(elapsed_time_sec, dtype=float)
    if values.ndim != 2 or len(values) != len(elapsed):
        raise ValueError("frame features and elapsed time must align")
    if len(values) == 0:
        raise ValueError("cannot summarize an empty sequence")
    if np.any(np.diff(elapsed) < 0.0):
        raise ValueError("elapsed time must be monotonic")

    rows: list[np.ndarray] = []
    counts = np.empty(len(values), dtype=np.int32)
    for end in range(len(values)):
        start = int(np.searchsorted(elapsed, elapsed[end] - window_seconds, side="left"))
        segment = values[start : end + 1]
        counts[end] = len(segment)
        duration = float(elapsed[end] - elapsed[start])
        context_available = len(segment) >= 2 and duration > 0.0
        if len(segment) == 1:
            segment = np.repeat(segment, 2, axis=0)
        rows.append(
            summarize_windows(
                segment[None, :, :],
                np.asarray([counts[end]]),
                np.asarray([duration]),
                np.asarray([context_available]),
            )[0]
        )
    return np.asarray(rows, dtype=np.float32), counts


def _singleton_summary(frame_features: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("static frame features must be two-dimensional")
    repeated = np.repeat(values[:, None, :], 2, axis=1)
    return summarize_windows(
        repeated,
        np.ones(len(values), dtype=np.int32),
        np.zeros(len(values), dtype=np.float32),
        np.zeros(len(values), dtype=bool),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_baseline(
    force_fz_n: np.ndarray,
    intensity: np.ndarray,
    baseline_config: Mapping[str, Any],
    no_contact_max_force_n: float,
    recorded_baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    strategy = str(
        baseline_config.get("strategy", "force_referenced_legacy")
    ).strip().lower()
    if strategy == "initial_fixed_frames":
        return _initial_fixed_frame_baseline(intensity, baseline_config)
    if strategy in {
        "fixed_recorded_runtime_preferred",
        "initial_optical_stable",
    }:
        initial_baseline, candidates = _initial_optical_stable_baseline(
            intensity, baseline_config
        )
        if strategy == "initial_optical_stable":
            return initial_baseline, candidates, "initial_optical_stable"

        if recorded_baseline is not None:
            nrms, correlation = _baseline_similarity(
                recorded_baseline, initial_baseline
            )
            if (
                nrms
                <= float(
                    baseline_config.get(
                        "maximum_recorded_vs_initial_nrms", 0.05
                    )
                )
                and correlation
                >= float(
                    baseline_config.get(
                        "minimum_recorded_vs_initial_correlation", 0.98
                    )
                )
            ):
                return (
                    np.asarray(recorded_baseline, dtype=float).copy(),
                    candidates,
                    "fixed_recorded_runtime_baseline",
                )
        return (
            initial_baseline,
            candidates,
            "initial_optical_stable_fallback",
        )

    frame_count = len(force_fz_n)
    search_count = max(
        int(baseline_config.get("minimum_frames", 5)),
        int(
            np.ceil(
                frame_count
                * float(baseline_config.get("search_fraction", 0.20))
            )
        ),
    )
    search_count = min(frame_count, search_count)
    finite_force = np.isfinite(force_fz_n)
    if not np.any(finite_force):
        candidates = np.arange(search_count, dtype=int)
        return (
            np.median(intensity[candidates], axis=0),
            candidates,
            "initial_unreferenced_frames",
        )
    candidates = np.flatnonzero(
        (np.arange(frame_count) < search_count)
        & finite_force
        & (
            force_fz_n
            <= float(
                baseline_config.get(
                    "maximum_force_n", no_contact_max_force_n
                )
            )
        )
    )
    minimum_frames = int(baseline_config.get("minimum_frames", 5))
    mode = "initial_low_force"
    if len(candidates) < minimum_frames:
        fallback_count = min(
            search_count,
            int(baseline_config.get("fallback_lowest_force_frames", 10)),
        )
        initial_force = np.where(
            finite_force[:search_count],
            force_fz_n[:search_count],
            np.inf,
        )
        candidates = np.argsort(initial_force)[:fallback_count]
        mode = "initial_lowest_force_fallback"
    return np.median(intensity[candidates], axis=0), candidates, mode


def _initial_fixed_frame_baseline(
    intensity: np.ndarray,
    baseline_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Use only the acquisition's explicit pre-contact warm-up frames."""

    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("intensity must be a non-empty frame matrix")
    frame_count = min(
        len(values),
        max(
            1,
            int(
                baseline_config.get(
                    "fixed_initial_frames",
                    baseline_config.get("minimum_frames", 5),
                )
            ),
        ),
    )
    candidates = np.arange(frame_count, dtype=int)
    return (
        np.median(values[candidates], axis=0),
        candidates,
        "initial_fixed_frames",
    )


def _initial_optical_stable_baseline(
    intensity: np.ndarray,
    baseline_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a runtime-available initial baseline without force labels."""

    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("intensity must be a non-empty frame matrix")
    frame_count = len(values)
    minimum_frames = max(1, int(baseline_config.get("minimum_frames", 5)))
    search_count = max(
        minimum_frames,
        int(
            np.ceil(
                frame_count
                * float(baseline_config.get("search_fraction", 0.20))
            )
        ),
    )
    search_count = min(frame_count, search_count)
    initial = values[:search_count]
    center = np.median(initial, axis=0)
    denominator = max(float(np.sqrt(np.mean(center**2))), 1.0e-12)
    distance = np.sqrt(np.mean((initial - center) ** 2, axis=1)) / denominator
    distance_center = float(np.median(distance))
    distance_mad = float(np.median(np.abs(distance - distance_center)))
    threshold = distance_center + float(
        baseline_config.get("initial_stability_mad_multiplier", 3.5)
    ) * max(1.4826 * distance_mad, 1.0e-12)
    candidates = np.flatnonzero(distance <= threshold)
    required = min(minimum_frames, search_count)
    if len(candidates) < required:
        candidates = np.argsort(distance)[:required]
    candidates = np.sort(candidates.astype(int))
    return np.median(initial[candidates], axis=0), candidates


def _baseline_similarity(
    recorded_baseline: np.ndarray,
    initial_baseline: np.ndarray,
) -> tuple[float, float]:
    recorded = np.asarray(recorded_baseline, dtype=float)
    initial = np.asarray(initial_baseline, dtype=float)
    if recorded.shape != initial.shape or not np.all(np.isfinite(recorded)):
        return float("inf"), float("-inf")
    denominator = max(float(np.sqrt(np.mean(recorded**2))), 1.0e-12)
    nrms = float(np.sqrt(np.mean((recorded - initial) ** 2)) / denominator)
    recorded_centered = recorded - float(np.mean(recorded))
    initial_centered = initial - float(np.mean(initial))
    correlation_denominator = float(
        np.sqrt(
            np.sum(recorded_centered**2) * np.sum(initial_centered**2)
        )
    )
    if correlation_denominator <= 1.0e-12:
        correlation = 1.0 if np.allclose(recorded, initial) else 0.0
    else:
        correlation = float(
            np.sum(recorded_centered * initial_centered)
            / correlation_denominator
        )
    return nrms, correlation


def _minimum_run_mask(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Keep only contiguous true runs that satisfy a minimum duration."""

    values = np.asarray(mask, dtype=bool)
    result = np.zeros_like(values)
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and values[stop]:
            stop += 1
        if stop - start >= minimum_frames:
            result[start:stop] = True
        start = stop
    return result


def derive_unreferenced_optical_labels(
    response_components: np.ndarray,
    baseline_indices: np.ndarray,
    position_label: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create conservative training-only labels for captures without Fz.

    These labels are never eligible for formal testing or force regression.
    Stable optical changes provide positive contact/position supervision, low
    baseline-relative changes provide no-contact supervision, and ambiguous
    transition frames remain unlabelled.
    """

    components = np.asarray(response_components, dtype=float)
    baseline = np.asarray(baseline_indices, dtype=int)
    if components.ndim != 2 or not len(components):
        raise ValueError("response_components must be a non-empty matrix")
    if not len(baseline):
        raise ValueError("baseline_indices must not be empty")

    scaled: list[np.ndarray] = []
    minimum_scale = float(config.get("minimum_component_scale", 1.0e-8))
    for component in components.T:
        center = float(np.median(component[baseline]))
        baseline_deviation = np.abs(component[baseline] - center)
        scale = max(
            float(np.percentile(baseline_deviation, 95.0)),
            float(np.median(baseline_deviation)) * 1.4826,
            minimum_scale,
        )
        scaled.append(np.abs(component - center) / scale)
    score = np.median(np.column_stack(scaled), axis=1)
    smoothing_frames = max(1, int(config.get("smoothing_frames", 3)))
    score = (
        pd.Series(score)
        .rolling(smoothing_frames, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )

    contact = np.full(len(score), -1, dtype=np.int8)
    position = np.full(len(score), "", dtype="<U16")
    inactive_threshold = float(config.get("inactive_max_robust_z", 1.5))
    inactive = score <= inactive_threshold
    inactive[baseline] = True
    contact[inactive] = 0

    active_threshold = float("nan")
    if position_label != "unlabeled":
        active_threshold = max(
            float(config.get("active_min_robust_z", 3.0)),
            float(
                np.percentile(
                    score,
                    float(config.get("active_session_percentile", 65.0)),
                )
            ),
        )
        contrast = float(np.percentile(score, 90.0) - np.percentile(score, 30.0))
        active = score >= active_threshold
        active = _minimum_run_mask(
            active,
            max(1, int(config.get("minimum_active_run_frames", 2))),
        )
        if (
            contrast < float(config.get("minimum_session_contrast_z", 1.5))
            or int(np.sum(active))
            < int(config.get("minimum_active_frames", 3))
        ):
            active[:] = False
        contact[active] = 1
        position[active] = position_label
    return contact, position, score, active_threshold


def _build_latest_rows(
    *,
    capture_root: Path,
    qa_summary_path: Path,
    latest_config: Mapping[str, Any],
    channel_config_path: Path,
    fusion_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    descriptors = discover_sessions(capture_root, qa_summary_path)
    data_config = dict(latest_config.get("data") or {})
    descriptors = filter_session_descriptors(descriptors, data_config)
    primary, challenge = split_primary_and_challenge_sessions(
        descriptors,
        dict(data_config.get("primary_selection") or {}),
    )
    primary_ids = {item.session_id for item in primary}
    evaluation_config = dict(fusion_config["evaluation"])
    force_primary = tuple(
        descriptor
        for descriptor in primary
        if session_has_force_reference(descriptor)
    )
    fold_by_session = assign_session_folds(
        force_primary,
        n_splits=int(evaluation_config.get("folds", 5)),
        random_seed=int(evaluation_config.get("random_seed", 42)),
    )
    peak_windows = load_peak_windows(channel_config_path)
    expected_points = int(
        latest_config.get("data", {}).get("expected_spectrum_points", 512)
    )
    labels = dict(fusion_config["labels"])
    force_config = dict(fusion_config["force_calibration"])
    baseline_config = dict(latest_config.get("baseline") or {})
    baseline_config.update(dict(fusion_config.get("latest_baseline") or {}))
    quality_config = dict(latest_config.get("quality") or {})
    no_contact_max = float(labels["no_contact_max_force_n"])
    contact_min = float(labels["contact_min_force_n"])
    position_min = float(labels["position_min_force_n"])
    force_min = float(force_config["minimum_n"])
    force_max = float(force_config["maximum_n"])
    release_warning = str(
        quality_config.get(
            "release_tail_warning_code", "release_recovery_residual"
        )
    )
    release_fraction = float(
        quality_config.get("release_tail_exclusion_fraction", 0.15)
    )
    trailing_seconds = float(
        fusion_config["temporal_features"]["trailing_window_seconds"]
    )
    unreferenced_config = dict(
        fusion_config.get("unreferenced_optical_labels") or {}
    )

    rows: list[dict[str, Any]] = []
    feature_names: tuple[str, ...] | None = None
    for descriptor in descriptors:
        if descriptor.qa_status == "fail":
            continue
        configured_role = (
            "latest_primary"
            if descriptor.session_id in primary_ids
            else "latest_challenge"
        )
        summary, wavelength_nm, intensity = _load_session_frame_matrix(
            descriptor, expected_points
        )
        force = summary["force_fz_n"].to_numpy(dtype=float)
        has_force_reference = bool(np.any(np.isfinite(force)))
        role = configured_role if has_force_reference else "latest_aux_no_force"
        elapsed = summary["elapsed_time_sec"].to_numpy(dtype=float)
        recorded_baseline = _load_session_recorded_baseline(
            descriptor, expected_points
        )
        baseline, baseline_indices, baseline_mode = _latest_baseline(
            force,
            intensity,
            baseline_config,
            no_contact_max,
            recorded_baseline=recorded_baseline,
        )
        initial_baseline, _ = _initial_optical_stable_baseline(
            intensity, baseline_config
        )
        recorded_baseline_nrms = float("nan")
        recorded_baseline_correlation = float("nan")
        if recorded_baseline is not None:
            (
                recorded_baseline_nrms,
                recorded_baseline_correlation,
            ) = _baseline_similarity(recorded_baseline, initial_baseline)
        (
            frame_features,
            current_feature_names,
            response_components,
            _,
        ) = extract_baseline_relative_frame_features(
            wavelength_nm,
            intensity,
            baseline,
            peak_windows,
        )
        if feature_names is None:
            feature_names = current_feature_names
        elif feature_names != current_feature_names:
            raise ValueError("latest session optical feature schema changed")
        temporal, frame_counts = summarize_trailing_sequence(
            frame_features, elapsed, trailing_seconds
        )
        release_excluded = np.zeros(len(force), dtype=bool)
        if release_warning in descriptor.finding_codes:
            release_count = max(1, int(np.ceil(len(force) * release_fraction)))
            release_excluded[-release_count:] = True

        pseudo_contact = np.full(len(force), -1, dtype=np.int8)
        pseudo_position = np.full(len(force), "", dtype="<U16")
        pseudo_active_threshold = float("nan")
        if not has_force_reference:
            (
                pseudo_contact,
                pseudo_position,
                _,
                pseudo_active_threshold,
            ) = derive_unreferenced_optical_labels(
                response_components,
                baseline_indices,
                descriptor.position_label,
                unreferenced_config,
            )

        for index in range(len(force)):
            finite_optical = bool(np.all(np.isfinite(temporal[index])))
            finite_force = bool(np.isfinite(force[index]))
            out_of_range = bool(
                finite_force
                and not (force_min <= force[index] <= force_max)
            )
            valid = finite_optical and not bool(release_excluded[index])
            if has_force_reference:
                contact_target = -1
                if force[index] <= no_contact_max:
                    contact_target = 0
                elif (
                    force[index] >= contact_min
                    and descriptor.position_label != "unlabeled"
                ):
                    contact_target = 1
                position_target = ""
                if (
                    role == "latest_primary"
                    and force[index] >= position_min
                    and descriptor.position_label != "unlabeled"
                ):
                    position_target = descriptor.position_label
                label_origin = "synchronized_px6d_fz"
            else:
                contact_target = int(pseudo_contact[index])
                position_target = str(pseudo_position[index])
                label_origin = "optical_change_pseudo_label_training_only"
            rows.append(
                {
                    "features": temporal[index],
                    "contact_target": contact_target,
                    "position_target": position_target,
                    "force_fz_n": float(force[index]),
                    "contact_mask": valid and contact_target >= 0,
                    "position_mask": valid and bool(position_target),
                    "force_mask": (
                        valid and finite_force and not out_of_range
                    ),
                    "formal_test_eligible": (
                        role == "latest_primary" and has_force_reference
                    ),
                    "fold_id": int(
                        fold_by_session.get(descriptor.session_id, -1)
                    ),
                    "source_id": str(
                        data_config.get("dataset_id")
                        or "ordinary_fbg_px6d_latest"
                    ),
                    "source_role": role,
                    "group_id": descriptor.session_id,
                    "file_id": descriptor.session_id,
                    "sample_index": int(
                        summary.iloc[index]["capture_index"]
                    ),
                    "elapsed_time_sec": float(elapsed[index]),
                    "temporal_frame_count": int(frame_counts[index]),
                    "force_out_of_range": out_of_range,
                    "baseline_mode": baseline_mode,
                    "baseline_frame_count": int(len(baseline_indices)),
                    "recorded_baseline_available": recorded_baseline is not None,
                    "recorded_baseline_vs_initial_nrms": recorded_baseline_nrms,
                    "recorded_baseline_vs_initial_correlation": (
                        recorded_baseline_correlation
                    ),
                    "pseudo_active_threshold": pseudo_active_threshold,
                    "source_quality": descriptor.qa_status,
                    "source_quality_flags": ";".join(
                        descriptor.finding_codes
                    ),
                    "label_origin": label_origin,
                }
            )
    if feature_names is None:
        raise ValueError("no latest synchronized sessions were loaded")
    return rows, feature_names


def _build_dynamic_rows(
    *,
    dynamic_config_path: Path,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    config = yaml.safe_load(dynamic_config_path.read_text(encoding="utf-8")) or {}
    dataset_path = dynamic_config_path.parent.parent / (
        "outputs/dynamic_sequence_dataset_20260714_v1/"
        "dynamic_sequence_windows.npz"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "the frozen dynamic sequence dataset is missing: "
            f"{dataset_path.resolve()}"
        )
    with np.load(dataset_path, allow_pickle=False) as payload:
        windows = payload["X"].astype(np.float32)
        feature_names = tuple(payload["feature_names"].astype(str).tolist())
        stage_labels = payload["stage_labels"].astype(str)
        contact_labels = payload["contact_labels"].astype(str)
        position_labels = payload["position_labels"].astype(str)
        file_ids = payload["file_ids"].astype(str)
        capture_groups = payload["capture_groups"].astype(str)
        starts = payload["window_start_frames"].astype(int)
        ends = payload["window_end_frames"].astype(int)
        quality = payload["sequence_quality_flags"].astype(str)
    frame_interval = float(config["acquisition"]["frame_interval_sec"])
    temporal = summarize_windows(
        windows,
        np.full(len(windows), windows.shape[1], dtype=np.int32),
        np.full(
            len(windows),
            (windows.shape[1] - 1) * frame_interval,
            dtype=np.float32,
        ),
        np.ones(len(windows), dtype=bool),
    )
    rows: list[dict[str, Any]] = []
    for index in range(len(windows)):
        contact_target = 1 if contact_labels[index] == "contact" else 0
        rows.append(
            {
                "features": temporal[index],
                "contact_target": contact_target,
                "position_target": position_labels[index],
                "force_fz_n": float("nan"),
                "contact_mask": True,
                "position_mask": bool(position_labels[index]),
                "force_mask": False,
                "formal_test_eligible": False,
                "fold_id": -1,
                "source_id": "legacy_dynamic_sequence_20260714",
                "source_role": "legacy_dynamic",
                "group_id": f"dynamic::{file_ids[index]}",
                "file_id": file_ids[index],
                "sample_index": int(starts[index]),
                "elapsed_time_sec": float(starts[index] * frame_interval),
                "temporal_frame_count": int(ends[index] - starts[index]),
                "force_out_of_range": False,
                "baseline_mode": "initial_no_contact_sequence_median",
                "baseline_frame_count": int(
                    config["baseline"]["minimum_frames"]
                ),
                "source_quality": "usable_with_warning"
                if quality[index]
                else "ok",
                "source_quality_flags": quality[index],
                "label_origin": f"ordered_sequence_stage::{stage_labels[index]}",
            }
        )
    return rows, feature_names


def _build_static_rows(
    *,
    static_config_path: Path,
    channel_config_path: Path,
    force_min_n: float,
    force_max_n: float,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    config = load_training_config(static_config_path)
    data_root = Path(config["data_root"]).expanduser().resolve()
    dataset_folders = tuple(str(value) for value in config["dataset_folders"].values())
    files = sorted(
        (
            path
            for folder in dataset_folders
            for path in (data_root / folder).rglob("*.csv")
        ),
        key=lambda path: str(path).lower(),
    )
    no_contact_folder = str(config["dataset_folders"]["no_contact"])
    no_contact_files = [
        path
        for path in files
        if path.relative_to(data_root).parts[0] == no_contact_folder
    ]
    no_contact_group_by_path = {
        path: index % int(config.get("cv_folds", 5)) + 1
        for index, path in enumerate(no_contact_files)
    }
    records = [
        parse_sense_spectrum_csv(
            path,
            data_root,
            config,
            no_contact_cv_group=no_contact_group_by_path.get(path),
        )
        for path in files
    ]
    dataset = build_static_feature_dataset(
        records, config.get("feature_extraction") or config, channel_config_path
    )
    peak_windows = load_peak_windows(channel_config_path)
    frame_rows: list[np.ndarray] = []
    for record in dataset.records:
        current = resample_spectrum(record, dataset.common_wavelength_nm)
        baseline, _ = baseline_for_record(
            record,
            dataset.reference_baseline_clusters,
            strategy=dataset.baseline_reference_strategy,
        )
        frame_features, feature_names, _, _ = (
            extract_baseline_relative_frame_features(
                dataset.common_wavelength_nm,
                current,
                baseline,
                peak_windows,
            )
        )
        frame_rows.append(frame_features[0])
    temporal = _singleton_summary(np.asarray(frame_rows, dtype=np.float32))
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(dataset.records):
        eligible = bool(dataset.training_eligible[index])
        role = {
            "manual_press": "legacy_static_manual",
            "gauge_press": "legacy_static_gauge",
            "no_contact": "legacy_static_no_contact",
        }[record.sample_kind]
        force_value = (
            float(record.force_newton)
            if record.force_newton is not None
            else float("nan")
        )
        out_of_range = bool(
            np.isfinite(force_value)
            and not (force_min_n <= force_value <= force_max_n)
        )
        rows.append(
            {
                "features": temporal[index],
                "contact_target": 0
                if record.sample_kind == "no_contact"
                else 1,
                "position_target": record.position_label or "",
                "force_fz_n": force_value,
                "contact_mask": eligible,
                "position_mask": eligible and bool(record.position_label),
                "force_mask": eligible
                and record.sample_kind == "gauge_press"
                and not out_of_range,
                "formal_test_eligible": False,
                "fold_id": -1,
                "source_id": "legacy_static_sense_snapshots",
                "source_role": role,
                "group_id": f"static::repeat::{record.cv_group}",
                "file_id": record.file_id,
                "sample_index": int(record.repeat_index or record.cv_group),
                "elapsed_time_sec": 0.0,
                "temporal_frame_count": 1,
                "force_out_of_range": out_of_range,
                "baseline_mode": dataset.baseline_reference_mode[index],
                "baseline_frame_count": 1,
                "source_quality": "ok"
                if not record.quality_flags
                else "usable_with_warning",
                "source_quality_flags": ";".join(record.quality_flags),
                "label_origin": (
                    "known_gauge_force_newton"
                    if record.sample_kind == "gauge_press"
                    else "folder_position_and_contact"
                ),
            }
        )
    return rows, feature_names


def _source_inventory(
    rows: Iterable[dict[str, Any]],
    blind_root: Path,
) -> tuple[dict[str, Any], ...]:
    source_rows = tuple(rows)
    inventory: list[dict[str, Any]] = []
    roles = sorted({str(row["source_role"]) for row in source_rows})
    for role in roles:
        selected = [row for row in source_rows if row["source_role"] == role]
        inventory.append(
            {
                "source_role": role,
                "sample_count": len(selected),
                "group_count": len({str(row["group_id"]) for row in selected}),
                "file_count": len({str(row["file_id"]) for row in selected}),
                "contact_label_count": sum(
                    bool(row["contact_mask"]) for row in selected
                ),
                "position_label_count": sum(
                    bool(row["position_mask"]) for row in selected
                ),
                "force_fz_label_count_0_to_5_n": sum(
                    bool(row["force_mask"]) for row in selected
                ),
                "force_out_of_range_count": sum(
                    bool(row["force_out_of_range"]) for row in selected
                ),
                "training_use": "eligible_by_task_mask",
            }
        )
    blind_files = (
        tuple(blind_root.rglob("*.dat")) + tuple(blind_root.rglob("*.csv"))
        if blind_root.is_dir()
        else ()
    )
    inventory.append(
        {
            "source_role": "blind_audit",
            "sample_count": 0,
            "group_count": 0,
            "file_count": len(blind_files),
            "contact_label_count": 0,
            "position_label_count": 0,
            "force_fz_label_count_0_to_5_n": 0,
            "force_out_of_range_count": 0,
            "training_use": "untouched_unseen_audit_only",
        }
    )
    return tuple(inventory)


def build_all_source_dataset(
    project_root: Path,
    fusion_config: Mapping[str, Any],
) -> AllSourceFusionDataset:
    paths = dict(fusion_config["paths"])
    source_inclusion = dict(fusion_config.get("source_inclusion") or {})
    latest_config_path = resolve_project_path(
        project_root, paths["latest_training_config"]
    )
    latest_config = yaml.safe_load(
        latest_config_path.read_text(encoding="utf-8")
    ) or {}
    channel_config_path = resolve_project_path(
        project_root, paths["channel_config"]
    )
    latest_rows, latest_names = _build_latest_rows(
        capture_root=resolve_project_path(
            project_root, paths["latest_capture_root"]
        ),
        qa_summary_path=resolve_project_path(
            project_root, paths["latest_qa_summary"]
        ),
        latest_config=latest_config,
        channel_config_path=channel_config_path,
        fusion_config=fusion_config,
    )
    dynamic_rows: list[dict[str, Any]] = []
    dynamic_names = latest_names
    if bool(source_inclusion.get("legacy_dynamic", True)):
        dynamic_rows, dynamic_names = _build_dynamic_rows(
            dynamic_config_path=resolve_project_path(
                project_root, paths["dynamic_training_config"]
            )
        )
    force_config = dict(fusion_config["force_calibration"])
    static_rows: list[dict[str, Any]] = []
    static_names = latest_names
    if bool(source_inclusion.get("legacy_static", True)):
        static_rows, static_names = _build_static_rows(
            static_config_path=resolve_project_path(
                project_root, paths["static_training_config"]
            ),
            channel_config_path=channel_config_path,
            force_min_n=float(force_config["minimum_n"]),
            force_max_n=float(force_config["maximum_n"]),
        )
    if latest_names != dynamic_names or latest_names != static_names:
        raise ValueError(
            "all sources must share the same 40 optical frame feature names"
        )
    rows = latest_rows + dynamic_rows + static_rows
    feature_names = _summary_feature_names(latest_names)
    matrix = np.asarray([row.pop("features") for row in rows], dtype=np.float32)
    if matrix.shape[1] != len(feature_names):
        raise AssertionError("feature matrix and feature names do not align")
    lowered_names = tuple(name.lower() for name in feature_names)
    forbidden = tuple(
        str(value).lower()
        for value in force_config.get("forbidden_input_patterns") or ()
    )
    leaked = [
        name
        for name, lowered in zip(feature_names, lowered_names, strict=True)
        if any(pattern in lowered for pattern in forbidden)
    ]
    if leaked:
        raise AssertionError(
            "force supervision leaked into model inputs: " + ", ".join(leaked[:5])
        )

    manifest = pd.DataFrame(rows)
    blind_root = resolve_project_path(project_root, paths["blind_audit_root"])
    inventory = _source_inventory(rows, blind_root)
    return AllSourceFusionDataset(
        features=matrix,
        feature_names=feature_names,
        contact_target=manifest["contact_target"].to_numpy(dtype=np.int8),
        position_target=manifest["position_target"].to_numpy(dtype=str),
        force_fz_n=manifest["force_fz_n"].to_numpy(dtype=np.float32),
        contact_mask=manifest["contact_mask"].to_numpy(dtype=bool),
        position_mask=manifest["position_mask"].to_numpy(dtype=bool),
        force_mask=manifest["force_mask"].to_numpy(dtype=bool),
        formal_test_eligible=manifest["formal_test_eligible"].to_numpy(
            dtype=bool
        ),
        fold_id=manifest["fold_id"].to_numpy(dtype=np.int8),
        source_id=manifest["source_id"].to_numpy(dtype=str),
        source_role=manifest["source_role"].to_numpy(dtype=str),
        group_id=manifest["group_id"].to_numpy(dtype=str),
        file_id=manifest["file_id"].to_numpy(dtype=str),
        sample_index=manifest["sample_index"].to_numpy(dtype=np.int32),
        elapsed_time_sec=manifest["elapsed_time_sec"].to_numpy(
            dtype=np.float32
        ),
        temporal_frame_count=manifest["temporal_frame_count"].to_numpy(
            dtype=np.int32
        ),
        force_out_of_range=manifest["force_out_of_range"].to_numpy(dtype=bool),
        manifest=manifest,
        source_inventory=inventory,
    )


def save_all_source_dataset(
    dataset: AllSourceFusionDataset,
    output_dir: Path,
    *,
    config_path: Path,
    protected_model_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "all_source_fusion_dataset.npz"
    np.savez_compressed(
        dataset_path,
        features=dataset.features,
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        contact_target=dataset.contact_target,
        position_target=dataset.position_target,
        force_fz_n=dataset.force_fz_n,
        contact_mask=dataset.contact_mask,
        position_mask=dataset.position_mask,
        force_mask=dataset.force_mask,
        formal_test_eligible=dataset.formal_test_eligible,
        fold_id=dataset.fold_id,
        source_id=dataset.source_id,
        source_role=dataset.source_role,
        group_id=dataset.group_id,
        file_id=dataset.file_id,
        sample_index=dataset.sample_index,
        elapsed_time_sec=dataset.elapsed_time_sec,
        temporal_frame_count=dataset.temporal_frame_count,
        force_out_of_range=dataset.force_out_of_range,
    )
    manifest_path = output_dir / "all_source_sample_manifest.csv"
    dataset.manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    inventory_path = output_dir / "all_source_source_inventory.csv"
    pd.DataFrame(dataset.source_inventory).to_csv(
        inventory_path, index=False, encoding="utf-8-sig"
    )
    role_counts = Counter(dataset.source_role.tolist())
    protected_hash = (
        _sha256(protected_model_path)
        if protected_model_path.is_file()
        else None
    )
    summary = {
        "schema_version": "ordinary_fbg_all_data_fusion_dataset_v1",
        "dataset_path": str(dataset_path),
        "sample_manifest_path": str(manifest_path),
        "source_inventory_path": str(inventory_path),
        "sample_count": int(len(dataset.features)),
        "feature_count": int(dataset.features.shape[1]),
        "source_role_counts": dict(sorted(role_counts.items())),
        "formal_primary_sample_count": int(
            np.sum(dataset.formal_test_eligible)
        ),
        "contact_label_count": int(np.sum(dataset.contact_mask)),
        "position_label_count": int(np.sum(dataset.position_mask)),
        "force_fz_label_count_0_to_5_n": int(np.sum(dataset.force_mask)),
        "force_out_of_range_excluded_count": int(
            np.sum(dataset.force_out_of_range)
        ),
        "blind_audit_training_sample_count": 0,
        "force_target_semantics": (
            "Fz supervision only; optical features are the only model inputs"
        ),
        "force_sensor_required_at_inference": False,
        "random_frame_split_allowed": False,
        "formal_split_strategy": "grouped_by_session_id",
        "protected_deployed_model_path": str(protected_model_path),
        "protected_deployed_model_sha256_before_training": protected_hash,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    summary_path = output_dir / "all_source_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "AllSourceFusionDataset",
    "build_all_source_dataset",
    "load_fusion_config",
    "resolve_project_path",
    "save_all_source_dataset",
    "summarize_trailing_sequence",
    "summarize_windows",
]
