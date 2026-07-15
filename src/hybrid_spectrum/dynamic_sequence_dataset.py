"""Dynamic Sense DAT loading, mixed-spectrum features and ordered labelling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .features import PeakWindow, load_peak_windows
from .sense_fast_dat import FastDatLayout, read_sense_fast_dat


EPSILON = 1.0e-9
PRESS_STAGE_ORDER = ("no_contact", "light", "normal", "hard", "release")


@dataclass(frozen=True)
class DynamicSequenceRecord:
    path: Path
    file_id: str
    capture_group: str
    position_label: str
    wavelength_nm: np.ndarray
    spectra: np.ndarray
    timestamps_sec: np.ndarray
    layout: FastDatLayout


@dataclass(frozen=True)
class StageSegment:
    label: str
    start_frame: int
    end_frame: int
    stable_start_frame: int
    stable_end_frame: int
    mean_response: float
    median_response: float
    training_eligible: bool
    quality_flag: str


@dataclass(frozen=True)
class DynamicFeatureSequence:
    record: DynamicSequenceRecord
    baseline_spectrum: np.ndarray
    baseline_frame_count: int
    feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    response_components: np.ndarray
    response_component_names: tuple[str, ...]
    response_score: np.ndarray
    stage_segments: tuple[StageSegment, ...]
    release_observed: bool
    release_recovered: bool
    release_recovery_ratio: float
    segmentation_status: str
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class DynamicWindowDataset:
    """Stable, boundary-safe temporal windows derived from independent DAT files."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    stage_labels: np.ndarray
    contact_labels: np.ndarray
    position_labels: np.ndarray
    file_ids: np.ndarray
    capture_groups: np.ndarray
    window_start_frames: np.ndarray
    window_end_frames: np.ndarray
    sequence_quality_flags: np.ndarray


def load_dynamic_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_root = config_path.parent.parent
    config["_config_path"] = config_path
    config["_project_root"] = project_root
    for field in ("data_root", "reference_spectrum_csv", "peak_config_path"):
        value = Path(str(config[field]))
        if not value.is_absolute():
            value = (project_root / value).resolve()
        config[f"_{field}"] = value
    return config


def load_reference_wavelength_grid(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [[cell.strip() for cell in row] for row in csv.reader(handle)]
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if len(row) >= 2 and row[:2] == ["WL", "Power"]
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"WL,Power header is missing: {path}")
    peak_index = next(
        (
            index
            for index in range(header_index + 1, len(rows))
            if rows[index] and rows[index][0] == "Peak_Count"
        ),
        None,
    )
    if peak_index is None:
        raise ValueError(f"Peak_Count row is missing: {path}")
    wavelength = np.asarray(
        [
            float(row[0])
            for row in rows[header_index + 1 : peak_index]
            if len(row) >= 2 and row[0] and row[1]
        ],
        dtype=float,
    )
    if wavelength.size != 512 or not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("reference wavelength grid must contain 512 increasing values")
    wavelength.setflags(write=False)
    return wavelength


def load_dynamic_records(config: dict[str, Any]) -> tuple[DynamicSequenceRecord, ...]:
    data_root = Path(config["_data_root"])
    wavelength = load_reference_wavelength_grid(Path(config["_reference_spectrum_csv"]))
    group_mapping = config["capture_group_mapping"]
    position_mapping = config["position_mapping"]
    frame_interval = float(config["acquisition"]["frame_interval_sec"])
    records: list[DynamicSequenceRecord] = []
    for path in sorted(data_root.rglob("*.dat"), key=lambda item: str(item).lower()):
        group_name = path.parent.name
        position_name = path.stem
        if group_name not in group_mapping:
            raise ValueError(f"unknown capture group folder: {group_name}")
        if position_name not in position_mapping:
            raise ValueError(f"unknown dynamic position filename: {position_name}")
        decoded = read_sense_fast_dat(path)
        spectra = np.asarray(decoded.spectra, dtype=float)
        spectra.setflags(write=False)
        timestamps = np.arange(len(spectra), dtype=float) * frame_interval
        timestamps.setflags(write=False)
        records.append(
            DynamicSequenceRecord(
                path=path,
                file_id=path.relative_to(data_root).as_posix(),
                capture_group=str(group_mapping[group_name]),
                position_label=str(position_mapping[position_name]),
                wavelength_nm=wavelength,
                spectra=spectra,
                timestamps_sec=timestamps,
                layout=decoded.layout,
            )
        )
    if not records:
        raise FileNotFoundError(f"no DAT files found below {data_root}")
    return tuple(records)


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
    names = (
        f"{prefix}_centroid_shift_pm",
        f"{prefix}_log_area_ratio",
        f"{prefix}_log_height_ratio",
        f"{prefix}_shape_rmse",
    )
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
    feature_names.extend(
        [
            "global_log_gain",
            "global_normalized_shape_rms",
            "global_normalized_shape_peak",
            "global_derivative_residual_energy",
        ]
    )
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


def extract_dynamic_frame_features(
    record: DynamicSequenceRecord,
    peak_windows: Iterable[PeakWindow],
    config: dict[str, Any],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...], np.ndarray, int]:
    baseline_config = config["baseline"]
    frame_interval = float(config["acquisition"]["frame_interval_sec"])
    baseline_frames = int(
        round(float(baseline_config["initial_seconds"]) / frame_interval)
    )
    baseline_frames = max(int(baseline_config["minimum_frames"]), baseline_frames)
    baseline_frames = min(
        int(baseline_config["maximum_frames"]),
        baseline_frames,
        max(2, len(record.spectra) // 5),
    )
    baseline = np.median(record.spectra[:baseline_frames], axis=0)
    (
        feature_matrix,
        feature_names,
        response_components,
        component_names,
    ) = extract_baseline_relative_frame_features(
        record.wavelength_nm,
        record.spectra,
        baseline,
        peak_windows,
    )
    scaled_components: list[np.ndarray] = []
    for component in response_components.T:
        baseline_center = float(np.median(component[:baseline_frames]))
        magnitude = np.abs(component - baseline_center)
        scale = max(
            float(np.percentile(magnitude, 95.0)),
            float(np.percentile(magnitude[:baseline_frames], 95.0)) * 5.0,
            EPSILON,
        )
        scaled_components.append(np.clip(magnitude / scale, 0.0, 1.5))
    response_score = np.median(np.column_stack(scaled_components), axis=1)
    response_score = gaussian_filter1d(
        response_score,
        sigma=float(config["segmentation"]["smoothing_sigma_frames"]),
    )
    baseline.setflags(write=False)
    feature_matrix.setflags(write=False)
    response_components.setflags(write=False)
    response_score.setflags(write=False)
    return (
        feature_matrix,
        tuple(feature_names),
        response_components,
        component_names,
        response_score,
        baseline_frames,
    )


def _detect_release_boundary(
    response: np.ndarray,
    minimum_segment_frames: int,
    config: dict[str, Any],
) -> tuple[int, bool, bool, float]:
    n = len(response)
    minimum_release = max(
        minimum_segment_frames,
        int(round(n * float(config["minimum_release_fraction"]))),
    )
    search_start = max(
        4 * minimum_segment_frames,
        int(round(n * float(config["release_search_start_fraction"]))),
    )
    search_stop = n - minimum_release
    comparison_window = max(12, int(round(n * 0.03)))
    candidates = range(search_start, max(search_start + 1, search_stop + 1))
    best_boundary = max(search_start, search_stop)
    best_drop = float("-inf")
    best_before = float(np.percentile(response[:best_boundary], 90.0))
    for boundary in candidates:
        before = float(
            np.median(response[max(0, boundary - comparison_window) : boundary])
        )
        after = float(
            np.median(response[boundary : min(n, boundary + comparison_window)])
        )
        drop = before - after
        if drop > best_drop:
            best_drop = drop
            best_boundary = boundary
            best_before = before
    initial = float(np.median(response[:minimum_segment_frames]))
    hard_scale = max(best_before - initial, EPSILON)
    release_threshold = max(
        float(config["release_drop_min_absolute"]),
        float(config["release_drop_min_hard_fraction"]) * hard_scale,
    )
    release_observed = bool(best_drop >= release_threshold)
    tail_count = max(minimum_segment_frames, int(round(n * 0.08)))
    tail = float(np.median(response[-tail_count:]))
    recovery_ratio = float(max(tail - initial, 0.0) / hard_scale)
    release_recovered = bool(
        release_observed
        and recovery_ratio <= float(config["recovered_tail_max_hard_fraction"])
    )
    return best_boundary, release_observed, release_recovered, recovery_ratio


def _ordered_four_stage_boundaries(
    response: np.ndarray,
    end_frame: int,
    minimum_segment_frames: int,
    config: dict[str, Any],
) -> tuple[tuple[int, int, int], dict[str, float]]:
    values = response[:end_frame]
    if end_frame < 4 * minimum_segment_frames:
        raise ValueError("pre-release sequence is too short for four ordered stages")

    response_range = max(float(np.ptp(values)), EPSILON)
    gain_window = max(
        int(config.get("transition_gain_min_window_frames", 12)),
        int(round(len(response) * float(config.get("transition_gain_window_fraction", 0.02)))),
    )
    transition_gain = np.zeros(end_frame, dtype=float)
    for boundary in range(gain_window, end_frame - gain_window):
        before = float(np.median(values[boundary - gain_window : boundary]))
        after = float(np.median(values[boundary : boundary + gain_window]))
        transition_gain[boundary] = after - before

    prominence = max(
        float(config.get("transition_peak_min_prominence_absolute", 0.002)),
        response_range
        * float(config.get("transition_peak_min_prominence_fraction", 0.004)),
    )
    peaks, _ = find_peaks(
        transition_gain,
        distance=max(8, minimum_segment_frames // 2),
        prominence=prominence,
    )
    candidates = {
        int(peak)
        for peak in peaks
        if minimum_segment_frames <= peak <= end_frame - minimum_segment_frames
    }

    # Weak light transitions can be smaller than later press transitions. Add
    # local maxima around broad temporal anchors so they remain candidates.
    for anchor_fraction in config.get(
        "transition_candidate_anchor_fractions",
        (0.22, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    ):
        anchor = int(round(end_frame * float(anchor_fraction)))
        radius = 2 * gain_window
        start = max(minimum_segment_frames, anchor - radius)
        stop = min(end_frame - minimum_segment_frames + 1, anchor + radius + 1)
        if stop > start:
            candidates.add(start + int(np.argmax(transition_gain[start:stop])))

    ordered_candidates = sorted(candidates)
    if len(ordered_candidates) < 3:
        raise RuntimeError("not enough positive transition candidates")

    trim_fraction = float(config.get("boundary_plateau_trim_fraction", 0.15))
    minimum_increment = float(config.get("stage_level_min_increment_fraction", 0.025))
    best: tuple[float, tuple[int, int, int], tuple[float, ...], float] | None = None
    second_best_score = float("-inf")
    for first_index, first in enumerate(ordered_candidates[:-2]):
        for second_index in range(first_index + 1, len(ordered_candidates) - 1):
            second = ordered_candidates[second_index]
            for third in ordered_candidates[second_index + 1 :]:
                lengths = np.asarray(
                    [first, second - first, third - second, end_frame - third],
                    dtype=float,
                )
                if float(np.min(lengths)) < minimum_segment_frames:
                    continue

                boundaries = (0, first, second, third, end_frame)
                medians: list[float] = []
                deviations: list[float] = []
                for start, stop in zip(boundaries[:-1], boundaries[1:]):
                    trim = max(4, int(round((stop - start) * trim_fraction)))
                    plateau = values[start + trim : stop - trim]
                    if plateau.size == 0:
                        plateau = values[start:stop]
                    median = float(np.median(plateau))
                    medians.append(median)
                    deviations.append(float(np.median(np.abs(plateau - median))))

                normalized_differences = np.diff(medians) / response_range
                monotonic_shortfall = float(
                    np.sum(np.maximum(minimum_increment - normalized_differences, 0.0))
                )
                nonmonotonic_count = int(np.count_nonzero(normalized_differences <= 0.0))
                contact_lengths = lengths[1:]
                duration_cv = float(np.std(contact_lengths) / max(np.mean(contact_lengths), EPSILON))
                variability = float(
                    sum(min(deviation / response_range, 0.25) for deviation in deviations)
                )
                gain_score = float(
                    (
                        transition_gain[first]
                        + transition_gain[second]
                        + transition_gain[third]
                    )
                    / response_range
                )
                level_span = float((medians[3] - medians[0]) / response_range)
                score = (
                    gain_score
                    + float(config.get("stage_level_span_weight", 0.50)) * level_span
                    - float(config.get("stage_monotonic_shortfall_weight", 8.0))
                    * monotonic_shortfall
                    - float(config.get("stage_nonmonotonic_penalty_weight", 4.0))
                    * nonmonotonic_count
                    - float(config.get("contact_duration_cv_penalty_weight", 0.70))
                    * duration_cv
                    - float(config.get("plateau_variability_penalty_weight", 0.35))
                    * variability
                )
                solution = (
                    score,
                    (int(first), int(second), int(third)),
                    tuple(medians),
                    duration_cv,
                )
                if best is None or score > best[0]:
                    if best is not None:
                        second_best_score = max(second_best_score, best[0])
                    best = solution
                else:
                    second_best_score = max(second_best_score, score)

    if best is None:
        raise RuntimeError("ordered stage transition search failed")
    _, boundaries, medians, duration_cv = best
    gain_fractions = [
        float(transition_gain[boundary] / response_range) for boundary in boundaries
    ]
    diagnostics = {
        "boundary_score": float(best[0]),
        "boundary_score_margin": float(best[0] - second_best_score),
        "contact_duration_cv": duration_cv,
        "minimum_transition_gain_fraction": float(min(gain_fractions)),
        "minimum_stage_increment_fraction": float(
            np.min(np.diff(medians)) / response_range
        ),
    }
    return boundaries, diagnostics


def segment_press_sequence(
    response_score: np.ndarray,
    config: dict[str, Any],
) -> tuple[tuple[StageSegment, ...], bool, bool, float, str, tuple[str, ...]]:
    segmentation = config["segmentation"]
    minimum = int(segmentation["minimum_segment_frames"])
    release_boundary, release_observed, release_recovered, recovery_ratio = (
        _detect_release_boundary(response_score, minimum, segmentation)
    )
    (first, second, third), boundary_diagnostics = _ordered_four_stage_boundaries(
        response_score,
        release_boundary,
        minimum,
        segmentation,
    )
    boundaries = (0, first, second, third, release_boundary, len(response_score))
    trim_fraction = float(segmentation["stable_trim_fraction"])
    guard = int(segmentation["transition_guard_frames"])
    minimum_stable = int(segmentation["minimum_stable_frames"])
    segments: list[StageSegment] = []
    quality_flags: list[str] = []
    means: list[float] = []
    for label, start, end in zip(PRESS_STAGE_ORDER, boundaries[:-1], boundaries[1:]):
        length = end - start
        trim = max(guard, int(round(length * trim_fraction)))
        stable_start = min(end, start + trim)
        stable_end = max(stable_start, end - trim)
        if stable_end - stable_start < minimum_stable:
            center = (start + end) // 2
            half = min(length // 2, max(1, minimum_stable // 2))
            stable_start = max(start, center - half)
            stable_end = min(end, stable_start + minimum_stable)
        eligible = bool(
            stable_end - stable_start >= minimum_stable and label != "release"
        )
        flag = "stable_plateau" if eligible else "excluded_release_or_short_plateau"
        mean_response = float(np.mean(response_score[start:end]))
        median_response = float(np.median(response_score[start:end]))
        means.append(mean_response)
        segments.append(
            StageSegment(
                label=label,
                start_frame=start,
                end_frame=end,
                stable_start_frame=stable_start,
                stable_end_frame=stable_end,
                mean_response=mean_response,
                median_response=median_response,
                training_eligible=eligible,
                quality_flag=flag,
            )
        )
    monotonic = bool(means[0] < means[1] < means[2] < means[3])
    if not monotonic:
        quality_flags.append("press_stage_response_not_monotonic")
    if boundary_diagnostics["minimum_transition_gain_fraction"] < float(
        segmentation.get("weak_transition_gain_fraction", 0.01)
    ):
        quality_flags.append("weak_press_stage_transition")
    if boundary_diagnostics["contact_duration_cv"] > float(
        segmentation.get("contact_duration_cv_warning", 0.65)
    ):
        quality_flags.append("contact_stage_duration_imbalanced")
    if boundary_diagnostics["boundary_score_margin"] < float(
        segmentation.get("boundary_score_margin_warning", 0.005)
    ):
        quality_flags.append("stage_boundary_solution_ambiguous")
    if not release_observed:
        quality_flags.append("release_drop_not_clearly_observed")
    if not release_recovered:
        quality_flags.append("release_residual_above_baseline")
    if any(not segment.training_eligible for segment in segments[:4]):
        quality_flags.append("stable_plateau_too_short")
    status = "good_sequence" if not quality_flags else "usable_with_warning"
    return (
        tuple(segments),
        release_observed,
        release_recovered,
        recovery_ratio,
        status,
        tuple(quality_flags),
    )


def build_dynamic_feature_sequence(
    record: DynamicSequenceRecord,
    peak_windows: Iterable[PeakWindow],
    config: dict[str, Any],
) -> DynamicFeatureSequence:
    (
        feature_matrix,
        feature_names,
        response_components,
        component_names,
        response_score,
        baseline_frames,
    ) = extract_dynamic_frame_features(record, peak_windows, config)
    baseline = np.median(record.spectra[:baseline_frames], axis=0)
    (
        segments,
        release_observed,
        release_recovered,
        recovery_ratio,
        status,
        quality_flags,
    ) = segment_press_sequence(response_score, config)
    baseline.setflags(write=False)
    return DynamicFeatureSequence(
        record=record,
        baseline_spectrum=baseline,
        baseline_frame_count=baseline_frames,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        response_components=response_components,
        response_component_names=component_names,
        response_score=response_score,
        stage_segments=segments,
        release_observed=release_observed,
        release_recovered=release_recovered,
        release_recovery_ratio=recovery_ratio,
        segmentation_status=status,
        quality_flags=quality_flags,
    )


def load_dynamic_feature_sequences(
    config: dict[str, Any],
) -> tuple[DynamicFeatureSequence, ...]:
    records = load_dynamic_records(config)
    peak_windows = load_peak_windows(Path(config["_peak_config_path"]))
    return tuple(
        build_dynamic_feature_sequence(record, peak_windows, config)
        for record in records
    )


def build_dynamic_window_dataset(
    sequences: Iterable[DynamicFeatureSequence],
    config: dict[str, Any],
) -> DynamicWindowDataset:
    """Create ``[windows, time, features]`` data without crossing stage boundaries."""

    source_sequences = tuple(sequences)
    if not source_sequences:
        raise ValueError("at least one dynamic feature sequence is required")
    window_config = config["windowing"]
    time_steps = int(window_config["time_steps"])
    stride = int(window_config["stride_frames"])
    if time_steps < 2 or stride < 1:
        raise ValueError("invalid dynamic window configuration")

    values: list[np.ndarray] = []
    stage_labels: list[str] = []
    contact_labels: list[str] = []
    position_labels: list[str] = []
    file_ids: list[str] = []
    capture_groups: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    quality: list[str] = []
    feature_names = source_sequences[0].feature_names
    for sequence in source_sequences:
        if sequence.feature_names != feature_names:
            raise ValueError("all dynamic sequences must share the same feature order")
        for segment in sequence.stage_segments:
            if not segment.training_eligible or segment.label == "release":
                continue
            final_start = segment.stable_end_frame - time_steps
            if final_start < segment.stable_start_frame:
                continue
            for start in range(segment.stable_start_frame, final_start + 1, stride):
                end = start + time_steps
                if start < segment.start_frame or end > segment.end_frame:
                    raise AssertionError("dynamic window crossed a labelled stage boundary")
                values.append(sequence.feature_matrix[start:end])
                stage_labels.append(segment.label)
                contact_labels.append(
                    "no_contact" if segment.label == "no_contact" else "contact"
                )
                position_labels.append(
                    "" if segment.label == "no_contact" else sequence.record.position_label
                )
                file_ids.append(sequence.record.file_id)
                capture_groups.append(sequence.record.capture_group)
                starts.append(start)
                ends.append(end)
                quality.append(";".join(sequence.quality_flags))
    if not values:
        raise ValueError("no stable dynamic windows could be constructed")

    arrays = {
        "values": np.stack(values).astype(np.float32),
        "stage_labels": np.asarray(stage_labels),
        "contact_labels": np.asarray(contact_labels),
        "position_labels": np.asarray(position_labels),
        "file_ids": np.asarray(file_ids),
        "capture_groups": np.asarray(capture_groups),
        "window_start_frames": np.asarray(starts, dtype=np.int32),
        "window_end_frames": np.asarray(ends, dtype=np.int32),
        "sequence_quality_flags": np.asarray(quality),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return DynamicWindowDataset(
        feature_names=feature_names,
        **arrays,
    )
