"""One-physical-frame spectral views for low-latency dynamic recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dynamic_sequence_dataset import DynamicFeatureSequence


EPSILON = 1.0e-9
STABLE_STAGE_ORDER = ("no_contact", "light", "normal", "hard")


@dataclass(frozen=True)
class DynamicSingleSpectrumDataset:
    engineered_features: np.ndarray
    spectral_views: np.ndarray
    stage_labels: np.ndarray
    contact_labels: np.ndarray
    position_labels: np.ndarray
    file_ids: np.ndarray
    capture_groups: np.ndarray
    frame_indices: np.ndarray
    feature_names: tuple[str, ...]
    spectral_view_names: tuple[str, ...]
    live_frame_stride: int


def stable_live_frame_indices(start: int, end: int, stride: int) -> np.ndarray:
    """Sample a stable half-open segment at an acquisition-aware cadence."""

    start = int(start)
    end = int(end)
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be positive")
    if end <= start:
        return np.empty(0, dtype=np.int32)
    indices = np.arange(start, end, stride, dtype=np.int32)
    final = end - 1
    if not len(indices):
        return np.asarray([final], dtype=np.int32)
    if final - int(indices[-1]) >= max(1, stride // 2):
        indices = np.append(indices, np.int32(final))
    return indices


def baseline_relative_spectral_views(
    spectra: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Return log-ratio, normalized-shape residual, and its wavelength derivative."""

    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    reference = np.asarray(baseline, dtype=np.float64)
    if values.ndim != 2 or reference.shape != (values.shape[1],):
        raise ValueError("spectra and baseline wavelength dimensions must match")

    positive_values = np.maximum(values, EPSILON)
    positive_reference = np.maximum(reference, EPSILON)
    log_ratio = np.log(positive_values / positive_reference[None, :])

    current_mean = np.maximum(np.mean(positive_values, axis=1), EPSILON)
    baseline_mean = max(float(np.mean(positive_reference)), EPSILON)
    normalized_current = positive_values / current_mean[:, None]
    normalized_reference = positive_reference / baseline_mean
    shape_residual = normalized_current - normalized_reference[None, :]
    derivative_residual = np.gradient(shape_residual, axis=1)

    views = np.stack(
        [
            log_ratio,
            shape_residual,
            derivative_residual,
        ],
        axis=1,
    )
    return np.ascontiguousarray(
        np.nan_to_num(views, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float32,
    )


def build_dynamic_single_spectrum_dataset(
    sequences: Iterable[DynamicFeatureSequence],
    *,
    live_frame_stride: int = 10,
) -> DynamicSingleSpectrumDataset:
    """Build stable one-frame samples without crossing files or stage boundaries."""

    source = tuple(sequences)
    if not source:
        raise ValueError("at least one dynamic feature sequence is required")
    if live_frame_stride < 1:
        raise ValueError("live_frame_stride must be positive")

    feature_names = source[0].feature_names
    engineered_blocks: list[np.ndarray] = []
    spectral_blocks: list[np.ndarray] = []
    stage_labels: list[str] = []
    contact_labels: list[str] = []
    position_labels: list[str] = []
    file_ids: list[str] = []
    capture_groups: list[str] = []
    frame_indices: list[int] = []

    for sequence in source:
        if sequence.feature_names != feature_names:
            raise ValueError("all sequences must share the engineered feature order")
        for segment in sequence.stage_segments:
            if (
                segment.label not in STABLE_STAGE_ORDER
                or not segment.training_eligible
            ):
                continue
            indices = stable_live_frame_indices(
                segment.stable_start_frame,
                segment.stable_end_frame,
                live_frame_stride,
            )
            if not len(indices):
                continue
            if np.any(indices < segment.start_frame) or np.any(indices >= segment.end_frame):
                raise AssertionError("single-spectrum sample crossed a stage boundary")

            engineered_blocks.append(sequence.feature_matrix[indices])
            spectral_blocks.append(
                baseline_relative_spectral_views(
                    sequence.record.spectra[indices],
                    sequence.baseline_spectrum,
                )
            )
            count = len(indices)
            stage_labels.extend([segment.label] * count)
            contact_labels.extend(
                ["no_contact" if segment.label == "no_contact" else "contact"]
                * count
            )
            position_labels.extend(
                ["" if segment.label == "no_contact" else sequence.record.position_label]
                * count
            )
            file_ids.extend([sequence.record.file_id] * count)
            capture_groups.extend([sequence.record.capture_group] * count)
            frame_indices.extend(int(index) for index in indices)

    if not engineered_blocks:
        raise ValueError("no stable single-spectrum samples were produced")

    return DynamicSingleSpectrumDataset(
        engineered_features=np.ascontiguousarray(
            np.vstack(engineered_blocks), dtype=np.float32
        ),
        spectral_views=np.ascontiguousarray(
            np.concatenate(spectral_blocks, axis=0), dtype=np.float32
        ),
        stage_labels=np.asarray(stage_labels),
        contact_labels=np.asarray(contact_labels),
        position_labels=np.asarray(position_labels),
        file_ids=np.asarray(file_ids),
        capture_groups=np.asarray(capture_groups),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        feature_names=feature_names,
        spectral_view_names=(
            "log_intensity_ratio",
            "normalized_shape_residual",
            "wavelength_derivative_residual",
        ),
        live_frame_stride=int(live_frame_stride),
    )
