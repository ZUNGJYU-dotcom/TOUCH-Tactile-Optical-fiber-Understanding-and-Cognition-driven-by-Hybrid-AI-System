"""Sense CSV ingestion and baseline-relative features for static spectra."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .features import PeakWindow, extract_frame_features, load_peak_windows


EPSILON = 1.0e-9
DEFAULT_TIMESTAMP_FORMATS = ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class SenseSpectrumRecord:
    """One immutable Sense spectrum snapshot and its parsed labels."""

    path: Path
    file_id: str
    timestamp: datetime
    metadata: dict[str, str]
    wavelength_nm: np.ndarray
    intensity_counts: np.ndarray
    peak_wavelength_nm: np.ndarray
    peak_power_counts: np.ndarray
    peak_fwhm_nm: np.ndarray
    sample_kind: str
    position_label: str | None
    manual_force_label: str | None
    gauge_force_label: str | None
    force_newton: float | None
    repeat_index: int | None
    cv_group: int
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def contact_label(self) -> str:
        return "no_contact" if self.sample_kind == "no_contact" else "contact"


@dataclass(frozen=True)
class BaselineCluster:
    """A time-local no-contact reference cluster."""

    cluster_id: str
    center_epoch: float
    record_ids: tuple[str, ...]
    spectra: np.ndarray

    @property
    def median_spectrum(self) -> np.ndarray:
        return np.median(self.spectra, axis=0)


@dataclass(frozen=True)
class BaselineClusterAssessment:
    """Quality and session-anchor interpretation of one no-contact cluster."""

    cluster_id: str
    session_id: str
    status: str
    trusted_for_reference: bool
    eligible_for_no_contact_training: bool
    anchor_cluster_id: str | None
    sample_count: int
    within_cluster_noise_ratio_median: float
    within_cluster_noise_ratio_max: float
    within_cluster_drift_ratio: float
    common_gain_ratio_to_anchor: float | None
    normalized_shape_rms_to_anchor: float | None
    normalized_shape_peak_to_anchor: float | None
    shape_correlation_to_anchor: float | None


@dataclass(frozen=True)
class StaticFeatureDataset:
    """Feature matrices plus immutable source records and baseline references."""

    records: tuple[SenseSpectrumRecord, ...]
    common_wavelength_nm: np.ndarray
    baseline_clusters: tuple[BaselineCluster, ...]
    baseline_cluster_assessments: tuple[BaselineClusterAssessment, ...]
    reference_baseline_clusters: tuple[BaselineCluster, ...]
    baseline_reference_strategy: str
    engineered_matrix: np.ndarray
    engineered_columns: tuple[str, ...]
    full_hybrid_matrix: np.ndarray
    full_hybrid_columns: tuple[str, ...]
    baseline_reference_mode: tuple[str, ...]
    training_eligible: tuple[bool, ...]


def _parse_timestamp(raw: str, path: Path) -> tuple[datetime, str | None]:
    for fmt in DEFAULT_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt), None
        except ValueError:
            continue
    return datetime.fromtimestamp(path.stat().st_mtime), "timestamp_from_file_mtime"


def _parse_repeat_suffix(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)-(\d+)", stem)
    if not match:
        raise ValueError(f"file name does not end with a repeat number: {stem}")
    return match.group(1), int(match.group(2))


def _load_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [[cell.strip() for cell in row] for row in csv.reader(handle)]


def _classify_path(
    path: Path,
    data_root: Path,
    config: dict[str, Any],
) -> tuple[str, str | None, str | None, str | None, float | None, int | None]:
    relative = path.relative_to(data_root)
    parts = relative.parts
    if not parts:
        raise ValueError(f"cannot classify path outside data root: {path}")

    folders = config["dataset_folders"]
    position_mapping = config["position_folder_mapping"]
    force_mapping = config["manual_force_mapping"]
    if parts[0] == folders["no_contact"]:
        return "no_contact", None, None, None, None, None
    if parts[0] == folders["manual_press"]:
        if len(parts) < 3:
            raise ValueError(f"manual press path has no position folder: {relative}")
        position = position_mapping.get(parts[1])
        if position is None:
            raise ValueError(f"unknown manual position folder: {parts[1]}")
        raw_force, repeat_index = _parse_repeat_suffix(path.stem)
        force_label = force_mapping.get(raw_force)
        if force_label is None:
            raise ValueError(f"unknown manual force label: {raw_force}")
        return "manual_press", position, force_label, None, None, repeat_index
    if parts[0] == folders["gauge_press"]:
        if len(parts) < 3:
            raise ValueError(f"gauge path has no position folder: {relative}")
        position = parts[1]
        allowed_positions = set(position_mapping.values())
        if position not in allowed_positions:
            raise ValueError(f"unknown gauge position folder: {position}")
        raw_force, repeat_index = _parse_repeat_suffix(path.stem)
        match = re.fullmatch(r"(\d+(?:\.\d+)?)N", raw_force, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"invalid gauge force label: {raw_force}")
        force_newton = float(match.group(1))
        return (
            "gauge_press",
            position,
            None,
            f"{force_newton:g}N",
            force_newton,
            repeat_index,
        )
    raise ValueError(f"unknown dataset folder: {parts[0]}")


def parse_sense_spectrum_csv(
    path: Path,
    data_root: Path,
    config: dict[str, Any],
    *,
    no_contact_cv_group: int | None = None,
) -> SenseSpectrumRecord:
    """Parse one Sense spectrum export without modifying the source file."""

    rows = _load_csv_rows(path)
    data_header_index = next(
        (index for index, row in enumerate(rows) if len(row) >= 2 and row[:2] == ["WL", "Power"]),
        None,
    )
    if data_header_index is None:
        raise ValueError(f"WL,Power header is missing: {path}")
    metadata = {
        row[0]: row[1]
        for row in rows[:data_header_index]
        if len(row) >= 2 and row[0]
    }
    peak_count_index = next(
        (
            index
            for index in range(data_header_index + 1, len(rows))
            if rows[index] and rows[index][0] == "Peak_Count"
        ),
        None,
    )
    if peak_count_index is None:
        raise ValueError(f"Peak_Count row is missing: {path}")

    wavelength: list[float] = []
    intensity: list[float] = []
    for row in rows[data_header_index + 1 : peak_count_index]:
        if len(row) < 2 or not row[0] or not row[1]:
            continue
        wavelength.append(float(row[0]))
        intensity.append(float(row[1]))

    peak_header_index = peak_count_index + 1
    if peak_header_index >= len(rows) or rows[peak_header_index][:3] != [
        "Peak_WL",
        "Peak_Power",
        "Peak_FWHM",
    ]:
        raise ValueError(f"peak table header is missing: {path}")
    peak_rows = [row for row in rows[peak_header_index + 1 :] if len(row) >= 3 and row[0]]
    peak_wavelength = np.asarray([float(row[0]) for row in peak_rows], dtype=float)
    peak_power = np.asarray([float(row[1]) for row in peak_rows], dtype=float)
    peak_fwhm = np.asarray([float(row[2]) for row in peak_rows], dtype=float)

    wavelength_array = np.asarray(wavelength, dtype=float)
    intensity_array = np.asarray(intensity, dtype=float)
    quality_flags: list[str] = []
    expected_points = int(config.get("expected_spectrum_points", 512))
    expected_peaks = int(config.get("expected_peak_count", 9))
    if wavelength_array.size != expected_points:
        quality_flags.append("unexpected_spectrum_point_count")
    if metadata.get("Data_Count") and int(metadata["Data_Count"]) != wavelength_array.size:
        quality_flags.append("metadata_data_count_mismatch")
    declared_peak_count = int(float(rows[peak_count_index][1])) if len(rows[peak_count_index]) > 1 else -1
    if peak_wavelength.size != expected_peaks or declared_peak_count != peak_wavelength.size:
        quality_flags.append("unexpected_peak_count")
    if not np.all(np.isfinite(wavelength_array)) or not np.all(np.isfinite(intensity_array)):
        quality_flags.append("non_finite_spectrum")
    if wavelength_array.size and not np.all(np.diff(wavelength_array) > 0.0):
        quality_flags.append("non_monotonic_wavelength_grid")
    if intensity_array.size and float(np.max(intensity_array)) >= float(
        config.get("saturation_threshold_counts", 65500.0)
    ):
        quality_flags.append("possible_saturation")
    if peak_power.size and np.any(peak_power <= 0.0):
        quality_flags.append("non_positive_peak_power")

    timestamp, timestamp_flag = _parse_timestamp(metadata.get("Test_Time", ""), path)
    if timestamp_flag:
        quality_flags.append(timestamp_flag)
    sample_kind, position, manual_force, gauge_force, force_newton, repeat_index = _classify_path(
        path,
        data_root,
        config,
    )
    if sample_kind == "no_contact":
        if no_contact_cv_group is None:
            raise ValueError("no_contact_cv_group is required for no-contact records")
        cv_group = int(no_contact_cv_group)
    else:
        if repeat_index is None:
            raise ValueError("contact sample has no repeat index")
        cv_group = int(repeat_index)

    return SenseSpectrumRecord(
        path=path,
        file_id=path.relative_to(data_root).as_posix(),
        timestamp=timestamp,
        metadata=metadata,
        wavelength_nm=wavelength_array,
        intensity_counts=intensity_array,
        peak_wavelength_nm=peak_wavelength,
        peak_power_counts=peak_power,
        peak_fwhm_nm=peak_fwhm,
        sample_kind=sample_kind,
        position_label=position,
        manual_force_label=manual_force,
        gauge_force_label=gauge_force,
        force_newton=force_newton,
        repeat_index=repeat_index,
        cv_group=cv_group,
        quality_flags=tuple(sorted(set(quality_flags))),
    )


def load_training_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = ("data_root", "dataset_folders", "position_folder_mapping", "manual_force_mapping")
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError("training config is missing: " + ", ".join(missing))
    return config


def load_sense_dataset(config: dict[str, Any]) -> list[SenseSpectrumRecord]:
    data_root = Path(config["data_root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Sense data root does not exist: {data_root}")
    files = sorted(data_root.rglob("*.csv"), key=lambda path: str(path).lower())
    no_contact_folder = config["dataset_folders"]["no_contact"]
    no_contact_files = sorted(
        [path for path in files if path.relative_to(data_root).parts[0] == no_contact_folder],
        key=lambda path: path.name,
    )
    no_contact_group_by_path = {
        path: index % int(config.get("cv_folds", 5)) + 1
        for index, path in enumerate(no_contact_files)
    }
    return [
        parse_sense_spectrum_csv(
            path,
            data_root,
            config,
            no_contact_cv_group=no_contact_group_by_path.get(path),
        )
        for path in files
    ]


def dataset_source_manifest(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return a lightweight immutable snapshot of every source CSV.

    Training scripts compare this manifest before and after loading so a file
    that is still being written cannot silently enter a formal evaluation.
    """

    data_root = Path(config["data_root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Sense data root does not exist: {data_root}")
    rows = []
    for path in sorted(data_root.rglob("*.csv"), key=lambda item: str(item).lower()):
        stat = path.stat()
        rows.append(
            {
                "file_id": path.relative_to(data_root).as_posix(),
                "size_bytes": int(stat.st_size),
                "modified_time_ns": int(stat.st_mtime_ns),
            }
        )
    return tuple(rows)


def assert_dataset_manifest_stable(
    before: tuple[dict[str, Any], ...],
    after: tuple[dict[str, Any], ...],
) -> None:
    if before == after:
        return
    before_by_id = {row["file_id"]: row for row in before}
    after_by_id = {row["file_id"]: row for row in after}
    added = sorted(set(after_by_id) - set(before_by_id))
    removed = sorted(set(before_by_id) - set(after_by_id))
    changed = sorted(
        file_id
        for file_id in set(before_by_id) & set(after_by_id)
        if before_by_id[file_id] != after_by_id[file_id]
    )
    raise RuntimeError(
        "source dataset changed while it was being loaded; retry after capture stops "
        f"(added={added[:3]}, removed={removed[:3]}, changed={changed[:3]})"
    )


def build_common_wavelength_grid(
    records: Iterable[SenseSpectrumRecord],
    point_count: int,
) -> np.ndarray:
    records = tuple(records)
    if not records:
        raise ValueError("cannot build a wavelength grid from no records")
    start = max(float(record.wavelength_nm[0]) for record in records)
    stop = min(float(record.wavelength_nm[-1]) for record in records)
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("source spectra do not share a valid wavelength range")
    return np.linspace(start, stop, point_count, dtype=float)


def resample_spectrum(record: SenseSpectrumRecord, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, record.wavelength_nm, record.intensity_counts)


def build_baseline_clusters(
    records: Iterable[SenseSpectrumRecord],
    grid: np.ndarray,
    gap_minutes: float,
) -> tuple[BaselineCluster, ...]:
    baseline_records = sorted(
        (record for record in records if record.sample_kind == "no_contact"),
        key=lambda record: record.timestamp,
    )
    if not baseline_records:
        raise ValueError("at least one no-contact spectrum is required")
    grouped: list[list[SenseSpectrumRecord]] = []
    for record in baseline_records:
        if not grouped:
            grouped.append([record])
            continue
        gap_seconds = (record.timestamp - grouped[-1][-1].timestamp).total_seconds()
        if gap_seconds > gap_minutes * 60.0:
            grouped.append([record])
        else:
            grouped[-1].append(record)
    clusters = []
    for index, group in enumerate(grouped, start=1):
        spectra = np.vstack([resample_spectrum(record, grid) for record in group])
        epochs = np.asarray([record.timestamp.timestamp() for record in group], dtype=float)
        clusters.append(
            BaselineCluster(
                cluster_id=f"baseline_cluster_{index:02d}",
                center_epoch=float(np.median(epochs)),
                record_ids=tuple(record.file_id for record in group),
                spectra=spectra,
            )
        )
    return tuple(clusters)


def _gain_normalized_spectrum_comparison(
    anchor: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, float, float]:
    anchor = np.asarray(anchor, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    scale = max(float(np.mean(np.abs(anchor))), EPSILON)
    valid = np.isfinite(anchor) & np.isfinite(candidate)
    valid &= np.abs(anchor) >= scale * 0.05
    if int(np.sum(valid)) < 16:
        return 1.0, float("inf"), float("inf"), 0.0
    gain = float(np.median(candidate[valid] / np.maximum(np.abs(anchor[valid]), EPSILON)))
    if not np.isfinite(gain) or gain <= EPSILON:
        return gain, float("inf"), float("inf"), 0.0
    corrected = candidate / gain
    residual = (corrected - anchor) / scale
    rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
    peak = float(np.max(np.abs(residual[valid])))
    correlation = (
        float(np.corrcoef(anchor[valid], corrected[valid])[0, 1])
        if float(np.std(anchor[valid])) > EPSILON
        and float(np.std(corrected[valid])) > EPSILON
        else 1.0 if rms <= EPSILON else 0.0
    )
    return gain, rms, peak, correlation


def assess_baseline_clusters(
    clusters: tuple[BaselineCluster, ...],
    quality_config: dict[str, Any] | None = None,
) -> tuple[BaselineClusterAssessment, ...]:
    """Separate session drift from stable post-release residual deformation."""

    quality_config = quality_config or {}
    session_gap_sec = float(quality_config.get("session_gap_minutes", 240.0)) * 60.0
    noise_fail = float(quality_config.get("within_cluster_noise_fail_ratio", 0.04))
    drift_fail = float(quality_config.get("within_cluster_drift_fail_ratio", 0.06))
    rms_fail = float(quality_config.get("normalized_shape_rms_fail", 0.03))
    peak_fail = float(quality_config.get("normalized_shape_peak_fail", 0.18))
    corr_fail = float(quality_config.get("shape_correlation_fail", 0.995))

    assessments: list[BaselineClusterAssessment] = []
    session_index = 0
    previous_epoch: float | None = None
    anchor: BaselineCluster | None = None
    for cluster in sorted(clusters, key=lambda item: item.center_epoch):
        if previous_epoch is None or cluster.center_epoch - previous_epoch > session_gap_sec:
            session_index += 1
            anchor = None
        previous_epoch = cluster.center_epoch
        session_id = f"baseline_session_{session_index:02d}"
        median = cluster.median_spectrum.astype(float)
        scale = max(float(np.mean(np.abs(median))), EPSILON)
        frame_rms = np.sqrt(np.mean((cluster.spectra - median) ** 2, axis=1)) / scale
        noise_median = float(np.median(frame_rms))
        noise_max = float(np.max(frame_rms))
        drift = float(
            np.sqrt(np.mean((cluster.spectra[-1] - cluster.spectra[0]) ** 2))
            / scale
        )

        gain: float | None = None
        rms: float | None = None
        peak: float | None = None
        correlation: float | None = None
        if noise_median >= noise_fail or drift >= drift_fail:
            status = "unstable_no_contact_cluster"
            trusted = False
        elif anchor is None:
            status = "trusted_session_anchor"
            trusted = True
            anchor = cluster
        else:
            gain, rms, peak, correlation = _gain_normalized_spectrum_comparison(
                anchor.median_spectrum,
                median,
            )
            if rms >= rms_fail or peak >= peak_fail or correlation <= corr_fail:
                status = "stable_recovery_residual_biased"
                trusted = False
            else:
                status = "trusted_session_consistent"
                trusted = True

        assessments.append(
            BaselineClusterAssessment(
                cluster_id=cluster.cluster_id,
                session_id=session_id,
                status=status,
                trusted_for_reference=trusted,
                eligible_for_no_contact_training=trusted,
                anchor_cluster_id=anchor.cluster_id if anchor is not None else None,
                sample_count=int(cluster.spectra.shape[0]),
                within_cluster_noise_ratio_median=noise_median,
                within_cluster_noise_ratio_max=noise_max,
                within_cluster_drift_ratio=drift,
                common_gain_ratio_to_anchor=gain,
                normalized_shape_rms_to_anchor=rms,
                normalized_shape_peak_to_anchor=peak,
                shape_correlation_to_anchor=correlation,
            )
        )
    return tuple(assessments)


def baseline_for_record(
    record: SenseSpectrumRecord,
    clusters: tuple[BaselineCluster, ...],
    strategy: str = "linear_interpolation",
) -> tuple[np.ndarray, str]:
    """Return a time-local baseline; no-contact files use leave-one-out medians."""

    if not clusters:
        raise ValueError("baseline clusters are empty")
    if record.sample_kind == "no_contact":
        for cluster in clusters:
            if record.file_id not in cluster.record_ids:
                continue
            local_index = cluster.record_ids.index(record.file_id)
            if cluster.spectra.shape[0] > 1:
                keep = np.ones(cluster.spectra.shape[0], dtype=bool)
                keep[local_index] = False
                return np.median(cluster.spectra[keep], axis=0), f"{cluster.cluster_id}_leave_one_out"
            return cluster.median_spectrum, f"{cluster.cluster_id}_single_reference"

    epoch = record.timestamp.timestamp()
    if strategy == "nearest_trusted_session_anchor":
        nearest = min(clusters, key=lambda cluster: abs(cluster.center_epoch - epoch))
        return nearest.median_spectrum, f"{nearest.cluster_id}_nearest_trusted"
    if len(clusters) == 1 or epoch <= clusters[0].center_epoch:
        return clusters[0].median_spectrum, f"{clusters[0].cluster_id}_nearest"
    if epoch >= clusters[-1].center_epoch:
        return clusters[-1].median_spectrum, f"{clusters[-1].cluster_id}_nearest"
    for left, right in zip(clusters, clusters[1:], strict=False):
        if left.center_epoch <= epoch <= right.center_epoch:
            fraction = (epoch - left.center_epoch) / max(right.center_epoch - left.center_epoch, EPSILON)
            spectrum = (1.0 - fraction) * left.median_spectrum + fraction * right.median_spectrum
            return spectrum, f"linear_interpolation_{left.cluster_id}_{right.cluster_id}"
    return clusters[-1].median_spectrum, f"{clusters[-1].cluster_id}_fallback"


def _downsample_mean(values: np.ndarray, bin_count: int) -> np.ndarray:
    return np.asarray([float(np.mean(chunk)) for chunk in np.array_split(values, bin_count)])


def extract_snapshot_feature_vectors(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    baseline_spectrum: np.ndarray,
    peak_windows: Iterable[PeakWindow],
    full_spectrum_bins: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return engineered and full-hybrid features for one aligned snapshot."""

    engineered = extract_frame_features(
        wavelength_nm,
        spectrum,
        baseline_spectrum,
        peak_windows,
    )
    ratio_log = np.log((spectrum + 1.0) / (baseline_spectrum + 1.0))
    current_shape = spectrum / max(float(np.mean(spectrum)), EPSILON)
    baseline_shape = baseline_spectrum / max(float(np.mean(baseline_spectrum)), EPSILON)
    shape_delta = current_shape - baseline_shape
    current_log_shape = np.log(spectrum + 1.0)
    current_log_shape = current_log_shape - float(np.mean(current_log_shape))
    current_shape_derivative = np.gradient(current_log_shape, wavelength_nm)
    full_hybrid = dict(engineered)
    for index, value in enumerate(
        _downsample_mean(ratio_log, full_spectrum_bins), start=1
    ):
        full_hybrid[f"spectrum_log_ratio_bin_{index:03d}"] = float(value)
    for index, value in enumerate(
        _downsample_mean(shape_delta, full_spectrum_bins), start=1
    ):
        full_hybrid[f"spectrum_shape_delta_bin_{index:03d}"] = float(value)
    for index, value in enumerate(
        _downsample_mean(current_log_shape, full_spectrum_bins), start=1
    ):
        full_hybrid[f"spectrum_current_log_shape_bin_{index:03d}"] = float(value)
    for index, value in enumerate(
        _downsample_mean(current_shape_derivative, full_spectrum_bins), start=1
    ):
        full_hybrid[f"spectrum_current_shape_derivative_bin_{index:03d}"] = float(value)
    return engineered, full_hybrid


def build_static_feature_dataset(
    records: Iterable[SenseSpectrumRecord],
    training_config: dict[str, Any],
    channel_config_path: Path,
) -> StaticFeatureDataset:
    records = tuple(records)
    grid = build_common_wavelength_grid(
        records,
        int(training_config.get("common_grid_points", 512)),
    )
    clusters = build_baseline_clusters(
        records,
        grid,
        float(training_config.get("baseline_cluster_gap_minutes", 15.0)),
    )
    quality_config = dict(training_config.get("baseline_quality") or {})
    assessments = assess_baseline_clusters(clusters, quality_config)
    assessment_by_id = {item.cluster_id: item for item in assessments}
    trusted_ids = {
        item.cluster_id for item in assessments if item.trusted_for_reference
    }
    reference_clusters = tuple(
        cluster for cluster in clusters if cluster.cluster_id in trusted_ids
    )
    if not reference_clusters:
        reference_clusters = clusters
    reference_strategy = str(
        quality_config.get("reference_strategy")
        or "nearest_trusted_session_anchor"
    )
    peak_windows: list[PeakWindow] = load_peak_windows(channel_config_path)
    bin_count = int(training_config.get("full_spectrum_bins", 64))
    engineered_rows: list[dict[str, float]] = []
    hybrid_rows: list[dict[str, float]] = []
    reference_modes: list[str] = []
    training_eligible: list[bool] = []
    for record in records:
        current = resample_spectrum(record, grid)
        baseline, reference_mode = baseline_for_record(
            record,
            reference_clusters,
            strategy=reference_strategy,
        )
        feature_row, extended = extract_snapshot_feature_vectors(
            grid,
            current,
            baseline,
            peak_windows,
            bin_count,
        )
        engineered_rows.append(feature_row)
        hybrid_rows.append(extended)
        reference_modes.append(reference_mode)
        eligible = True
        if record.sample_kind == "no_contact":
            source_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if record.file_id in cluster.record_ids
                ),
                None,
            )
            if source_cluster is not None:
                assessment = assessment_by_id[source_cluster.cluster_id]
                eligible = assessment.eligible_for_no_contact_training
        training_eligible.append(eligible)

    engineered_columns = tuple(sorted(engineered_rows[0]))
    hybrid_columns = tuple(sorted(hybrid_rows[0]))
    engineered_matrix = np.asarray(
        [[row[column] for column in engineered_columns] for row in engineered_rows],
        dtype=float,
    )
    hybrid_matrix = np.asarray(
        [[row[column] for column in hybrid_columns] for row in hybrid_rows],
        dtype=float,
    )
    return StaticFeatureDataset(
        records=records,
        common_wavelength_nm=grid,
        baseline_clusters=clusters,
        baseline_cluster_assessments=assessments,
        reference_baseline_clusters=reference_clusters,
        baseline_reference_strategy=reference_strategy,
        engineered_matrix=engineered_matrix,
        engineered_columns=engineered_columns,
        full_hybrid_matrix=hybrid_matrix,
        full_hybrid_columns=hybrid_columns,
        baseline_reference_mode=tuple(reference_modes),
        training_eligible=tuple(training_eligible),
    )


def dataset_sha256(records: Iterable[SenseSpectrumRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.file_id):
        digest.update(record.file_id.encode("utf-8"))
        with record.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
