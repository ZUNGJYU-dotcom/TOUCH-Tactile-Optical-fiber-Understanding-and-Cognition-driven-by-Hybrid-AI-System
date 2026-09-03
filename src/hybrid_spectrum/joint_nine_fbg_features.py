"""Runtime-safe joint fingerprint features for all nine ordinary FBG peaks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .baseline_relative_features import extract_baseline_relative_features
from .features import PeakWindow
from .runtime_channel_response import OBSERVED_ACTIVE_THRESHOLD
from .runtime_spectral_features import extract_baseline_relative_frame_features


EPSILON = 1.0e-9
EXPECTED_CHANNEL_COUNT = 9
LOCAL_METRIC_COUNT = 4
LOCAL_RESPONSE_SCALES = np.asarray((500.0, 0.5, 0.5, 0.25), dtype=float)


def extract_joint_nine_fbg_features(
    intensity: np.ndarray,
    baseline: np.ndarray,
    wavelength_nm: np.ndarray,
    peak_windows: Iterable[PeakWindow],
    *,
    bin_count: int = 64,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[int, ...]]]:
    """Combine full-spectrum evidence with an explicit nine-peak fingerprint.

    The per-channel response energies and their cross-channel distribution are
    calculated from wavelength, area, height, and shape changes.  This makes
    the coupled nine-FBG pattern directly available to every fitted task while
    retaining the complete baseline-relative spectrum.
    """

    windows = tuple(peak_windows)
    if len(windows) != EXPECTED_CHANNEL_COUNT:
        raise ValueError("joint fingerprint requires exactly nine FBG windows")
    candidate_ids = tuple(window.candidate_id for window in windows)
    if len(set(candidate_ids)) != EXPECTED_CHANNEL_COUNT:
        raise ValueError("joint fingerprint FBG candidate IDs must be unique")

    full_spectrum, full_names, _ = extract_baseline_relative_features(
        intensity,
        baseline,
        wavelength_nm,
        bin_count=bin_count,
    )
    local, local_names, _, _ = extract_baseline_relative_frame_features(
        wavelength_nm,
        intensity,
        baseline,
        windows,
    )
    expected_local_count = EXPECTED_CHANNEL_COUNT * LOCAL_METRIC_COUNT + 4
    if local.shape[1] != expected_local_count:
        raise ValueError("unexpected nine-FBG local feature count")

    channel_metrics = local[:, : EXPECTED_CHANNEL_COUNT * LOCAL_METRIC_COUNT]
    channel_metrics = channel_metrics.reshape(
        len(local), EXPECTED_CHANNEL_COUNT, LOCAL_METRIC_COUNT
    )
    scaled = np.abs(channel_metrics) / LOCAL_RESPONSE_SCALES[None, None, :]
    response_energy = np.sqrt(np.mean(np.square(scaled), axis=2))
    response_sum = np.sum(response_energy, axis=1, keepdims=True)
    response_fraction = np.divide(
        response_energy,
        response_sum,
        out=np.zeros_like(response_energy),
        where=response_sum > EPSILON,
    )
    response_centered = response_energy - np.mean(
        response_energy, axis=1, keepdims=True
    )
    ordered = np.sort(response_energy, axis=1)[:, ::-1]
    joint_summary = np.column_stack(
        (
            np.mean(response_energy, axis=1),
            np.std(response_energy, axis=1),
            np.sqrt(np.mean(np.square(response_energy), axis=1)),
            ordered[:, 0],
            ordered[:, 1],
            ordered[:, 2],
            ordered[:, -1],
            np.mean(response_energy >= OBSERVED_ACTIVE_THRESHOLD, axis=1),
        )
    )

    joint_blocks = (local, response_energy, response_fraction, response_centered)
    joint = np.column_stack((*joint_blocks, joint_summary))
    joint_names = list(local_names)
    for prefix in (
        "joint_response_energy",
        "joint_response_fraction",
        "joint_response_centered",
    ):
        joint_names.extend(
            f"{prefix}_{window.candidate_id.lower()}" for window in windows
        )
    joint_names.extend(
        (
            "joint_response_mean",
            "joint_response_std",
            "joint_response_rms",
            "joint_response_rank_1",
            "joint_response_rank_2",
            "joint_response_rank_3",
            "joint_response_min",
            "joint_response_active_fraction",
        )
    )
    if joint.shape[1] != len(joint_names):
        raise RuntimeError("joint nine-FBG feature-name contract mismatch")

    matrix = np.column_stack((full_spectrum, joint))
    names = (*full_names, *joint_names)
    full_count = len(full_names)
    joint_indices = tuple(range(full_count, matrix.shape[1]))
    feature_sets = {
        "baseline_relative_264": tuple(range(full_count)),
        "nine_fbg_joint_75": joint_indices,
        "baseline_relative_264_plus_nine_fbg_joint_339": tuple(
            range(matrix.shape[1])
        ),
    }
    return (
        np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
        names,
        feature_sets,
    )


__all__ = ["extract_joint_nine_fbg_features"]
