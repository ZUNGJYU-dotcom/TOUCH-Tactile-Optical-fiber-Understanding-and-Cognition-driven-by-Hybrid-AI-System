"""Causal datasets that match the slower live BaySpec acquisition cadence.

The lab DAT recordings contain an estimated 40 ms spectrum cadence, while the
current stable SDK path yields a physical spectrum roughly every 400 ms.  This
module deliberately selects real recorded frames at that slower cadence.  It
never invents intermediate spectra by interpolation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dynamic_temporal_features import SUMMARY_FEATURE_BLOCK_ORDER


VALID_STAGE_LABELS = ("no_contact", "light", "normal", "hard")


@dataclass(frozen=True)
class LiveCadenceDataset:
    """Stable target frames with causal physical-frame history."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    stage_labels: np.ndarray
    contact_labels: np.ndarray
    position_labels: np.ndarray
    file_ids: np.ndarray
    capture_groups: np.ndarray
    target_frame_indices: np.ndarray
    source_frame_indices: np.ndarray
    source_frame_interval_sec: float
    live_frame_interval_sec: float
    cadence_factor: int
    history_frames: int

    @property
    def history_span_sec(self) -> float:
        return max(0, self.history_frames - 1) * self.live_frame_interval_sec

    @property
    def cold_start_fill_sec(self) -> float:
        return self.history_frames * self.live_frame_interval_sec


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_live_cadence_dataset(
    audit_dir: Path,
    history_frames: int,
    *,
    source_frame_interval_sec: float = 0.04,
    live_frame_interval_sec: float = 0.40,
) -> LiveCadenceDataset:
    """Build causal windows from real frames at the requested live cadence.

    A target is retained only when the existing audit marks that exact frame as
    a stable training frame.  Earlier context may include a transition or the
    preceding stage because that is information available to a real causal
    classifier at inference time.
    """

    if history_frames < 1:
        raise ValueError("history_frames must be at least one")
    if source_frame_interval_sec <= 0.0 or live_frame_interval_sec <= 0.0:
        raise ValueError("frame intervals must be positive")
    cadence_factor = int(round(live_frame_interval_sec / source_frame_interval_sec))
    if cadence_factor < 1:
        raise ValueError("live cadence cannot be faster than the source cadence")
    represented_interval = cadence_factor * source_frame_interval_sec
    if not np.isclose(represented_interval, live_frame_interval_sec, rtol=0.0, atol=1e-6):
        raise ValueError("live/source frame interval ratio must be an integer")

    root = Path(audit_dir).resolve()
    labels_path = root / "dynamic_frame_labels.csv"
    features_path = root / "dynamic_frame_features.npz"
    if not labels_path.exists() or not features_path.exists():
        raise FileNotFoundError("dynamic audit labels or features are missing")

    rows = _read_label_rows(labels_path)
    with np.load(features_path, allow_pickle=True) as source:
        frames = np.asarray(source["X_frames"], dtype=np.float32)
        file_indices = np.asarray(source["file_indices"], dtype=np.int32)
        frame_indices = np.asarray(source["frame_indices"], dtype=np.int32)
        feature_names = tuple(str(value) for value in source["feature_names"].tolist())
        file_ids = np.asarray(source["file_ids"]).astype(str)
        groups = np.asarray(source["capture_groups"]).astype(str)
        positions = np.asarray(source["position_labels"]).astype(str)

    if len(rows) != len(frames):
        raise ValueError("audit label and feature row counts differ")
    expected_files = file_ids[file_indices]
    expected_groups = groups[file_indices]
    expected_positions = positions[file_indices]
    row_files = np.asarray([row["file_id"] for row in rows], dtype=str)
    row_groups = np.asarray([row["capture_group"] for row in rows], dtype=str)
    row_positions = np.asarray([row["position_label"] for row in rows], dtype=str)
    row_frame_indices = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int32)
    if not np.array_equal(row_files, expected_files):
        raise ValueError("audit file order does not match the feature archive")
    if not np.array_equal(row_groups, expected_groups):
        raise ValueError("audit capture groups do not match the feature archive")
    if not np.array_equal(row_positions, expected_positions):
        raise ValueError("audit positions do not match the feature archive")
    if not np.array_equal(row_frame_indices, frame_indices):
        raise ValueError("audit frame indices do not match the feature archive")

    stage = np.asarray([row["stage_label"] for row in rows], dtype=str)
    stable = np.asarray(
        [_parse_bool(row["stable_training_frame"]) for row in rows], dtype=bool
    )
    windows: list[np.ndarray] = []
    output_stage: list[str] = []
    output_contact: list[str] = []
    output_position: list[str] = []
    output_files: list[str] = []
    output_groups: list[str] = []
    output_targets: list[int] = []
    output_sources: list[np.ndarray] = []

    for file_index, file_id in enumerate(file_ids):
        global_rows = np.flatnonzero(file_indices == file_index)
        if not len(global_rows):
            continue
        local_frames = frame_indices[global_rows]
        order = np.argsort(local_frames)
        global_rows = global_rows[order]
        local_frames = local_frames[order]
        if not np.array_equal(local_frames, np.arange(len(local_frames), dtype=np.int32)):
            raise ValueError(f"non-contiguous frame indices in {file_id}")
        physical_rows = global_rows[::cadence_factor]
        for physical_index in range(history_frames - 1, len(physical_rows)):
            target_row = int(physical_rows[physical_index])
            target_stage = str(stage[target_row])
            if not stable[target_row] or target_stage not in VALID_STAGE_LABELS:
                continue
            source_rows = physical_rows[
                physical_index - history_frames + 1 : physical_index + 1
            ]
            windows.append(frames[source_rows])
            output_stage.append(target_stage)
            output_contact.append(
                "no_contact" if target_stage == "no_contact" else "contact"
            )
            output_position.append(
                "" if target_stage == "no_contact" else str(expected_positions[target_row])
            )
            output_files.append(str(file_id))
            output_groups.append(str(expected_groups[target_row]))
            output_targets.append(int(frame_indices[target_row]))
            output_sources.append(frame_indices[source_rows].copy())

    if not windows:
        raise ValueError("no stable live-cadence windows were constructed")
    arrays = {
        "values": np.stack(windows).astype(np.float32),
        "stage_labels": np.asarray(output_stage),
        "contact_labels": np.asarray(output_contact),
        "position_labels": np.asarray(output_position),
        "file_ids": np.asarray(output_files),
        "capture_groups": np.asarray(output_groups),
        "target_frame_indices": np.asarray(output_targets, dtype=np.int32),
        "source_frame_indices": np.stack(output_sources).astype(np.int32),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return LiveCadenceDataset(
        feature_names=feature_names,
        source_frame_interval_sec=float(source_frame_interval_sec),
        live_frame_interval_sec=float(live_frame_interval_sec),
        cadence_factor=cadence_factor,
        history_frames=history_frames,
        **arrays,
    )


def causal_summary_features(values: np.ndarray) -> np.ndarray:
    """Return the established 12-block summary for histories of length >= 1."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < 1:
        raise ValueError("values must have shape [windows, time>=1, features]")
    if array.shape[1] == 1:
        current = array[:, 0, :]
        zeros = np.zeros_like(current)
        blocks = [
            current,
            zeros,
            current,
            current,
            current,
            current,
            current,
            current,
            zeros,
            zeros,
            zeros,
            zeros,
        ]
        return np.column_stack(blocks).astype(np.float32)

    time_axis = np.linspace(-1.0, 1.0, array.shape[1], dtype=np.float32)
    centered_time = time_axis - float(np.mean(time_axis))
    denominator = max(float(np.sum(centered_time**2)), 1.0e-8)
    centered_values = array - np.mean(array, axis=1, keepdims=True)
    slope = np.sum(centered_values * centered_time[None, :, None], axis=1) / denominator
    difference = np.diff(array, axis=1)
    blocks = [
        np.mean(array, axis=1),
        np.std(array, axis=1),
        np.min(array, axis=1),
        np.max(array, axis=1),
        np.percentile(array, 10.0, axis=1),
        np.percentile(array, 90.0, axis=1),
        array[:, 0, :],
        array[:, -1, :],
        array[:, -1, :] - array[:, 0, :],
        slope,
        np.mean(np.abs(difference), axis=1),
        np.sqrt(np.mean(difference**2, axis=1)),
    ]
    return np.nan_to_num(
        np.column_stack(blocks), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def summary_feature_names(frame_feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{block}__{feature}"
        for block in SUMMARY_FEATURE_BLOCK_ORDER
        for feature in frame_feature_names
    )


__all__ = [
    "LiveCadenceDataset",
    "VALID_STAGE_LABELS",
    "build_live_cadence_dataset",
    "causal_summary_features",
    "summary_feature_names",
]
