from __future__ import annotations

import numpy as np
import pytest

from src.hybrid_spectrum.optical_tactile_information import (
    CHANNEL_COORDINATES,
    build_noise_normalized_channel_response,
    contact_patch_moments,
    find_release_recovery_events,
    session_baseline_metadata_mask,
)


def _feature_fixture() -> tuple[np.ndarray, np.ndarray]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for index in range(1, 10):
        for suffix in (
            "fused_common_mode_corrected_shift_pm",
            "log_area_ratio",
            "normalized_shape_rmse",
        ):
            names.append(f"fbg{index:02d}_{suffix}")
            base = np.asarray([0.0, 0.01, 0.02, 0.4, 0.8])
            columns.append(base * index)
    return np.column_stack(columns), np.asarray(names)


def test_channel_response_is_channel_normalized_and_finite() -> None:
    features, names = _feature_fixture()
    response, scale = build_noise_normalized_channel_response(
        features=features,
        feature_names=names,
        no_contact_mask=np.asarray([True, True, True, False, False]),
        contact_mask=np.asarray([False, False, False, True, True]),
    )

    assert response.shape == (5, 9)
    assert np.all(np.isfinite(response))
    assert len(scale) == 27
    assert np.all(response[-1] > response[1])


def test_contact_patch_center_tracks_dominant_pixel() -> None:
    weights = np.zeros(9)
    weights[4] = 1.0
    centered = contact_patch_moments(weights)
    assert centered["center_x"] == pytest.approx(0.0)
    assert centered["center_y"] == pytest.approx(0.0)

    weights = np.zeros(9)
    weights[6] = 1.0
    corner = contact_patch_moments(weights)
    assert corner["center_x"] == pytest.approx(CHANNEL_COORDINATES[6, 0])
    assert corner["center_y"] == pytest.approx(CHANNEL_COORDINATES[6, 1])


def test_release_recovery_requires_stable_frames() -> None:
    events = find_release_recovery_events(
        group_id=np.asarray(["s"] * 9),
        capture_index=np.arange(9),
        elapsed_time_sec=np.arange(9, dtype=float) * 0.1,
        contact_target=np.asarray([0, 0, 1, 1, -1, -1, 0, 0, 0]),
        response_score=np.asarray([0.01, 0.02, 0.8, 0.6, 0.4, 0.08, 0.02, 0.01, 0.01]),
        stable_frames=3,
    )

    assert len(events) == 1
    assert events[0].recovered is True
    assert events[0].recovery_time_sec == pytest.approx(0.3)


def test_session_baseline_metadata_mask_excludes_only_session_fields() -> None:
    names = np.asarray(
        [
            "fbg01_baseline_peak_snr",
            "fbg01_baseline_peak_valid",
            "fbg01_peak_snr",
            "fbg01_edge_margin_nm",
            "global_normalized_residual_rms",
        ]
    )
    assert session_baseline_metadata_mask(names).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
