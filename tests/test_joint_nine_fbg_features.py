from __future__ import annotations

import numpy as np

from src.hybrid_spectrum.features import PeakWindow
from src.hybrid_spectrum.joint_nine_fbg_features import (
    extract_joint_nine_fbg_features,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, tuple[PeakWindow, ...]]:
    wavelength = np.linspace(1526.0, 1562.0, 721)
    centers = np.linspace(1528.0, 1560.0, 9)
    baseline = np.full_like(wavelength, 1500.0)
    windows = []
    for index, center in enumerate(centers, start=1):
        baseline += 12000.0 * np.exp(-0.5 * ((wavelength - center) / 0.18) ** 2)
        windows.append(
            PeakWindow(
                candidate_id=f"FBG{index:02d}",
                provisional_channel_id=f"P{index:02d}",
                center_nm=float(center),
                half_width_nm=0.8,
            )
        )
    return wavelength, baseline, tuple(windows)


def test_joint_feature_contract_and_quiet_frame() -> None:
    wavelength, baseline, windows = _fixture()
    matrix, names, feature_sets = extract_joint_nine_fbg_features(
        baseline[None, :],
        baseline,
        wavelength,
        windows,
    )

    assert matrix.shape == (1, 339)
    assert len(names) == 339
    assert len(feature_sets["baseline_relative_264"]) == 264
    assert len(feature_sets["nine_fbg_joint_75"]) == 75
    assert len(feature_sets["baseline_relative_264_plus_nine_fbg_joint_339"]) == 339
    joint_indices = feature_sets["nine_fbg_joint_75"]
    np.testing.assert_allclose(matrix[:, joint_indices], 0.0, atol=1.0e-10)


def test_joint_fingerprint_uses_every_peak_window() -> None:
    wavelength, baseline, windows = _fixture()
    changed = baseline.copy()
    for index, window in enumerate(windows, start=1):
        mask = np.abs(wavelength - window.center_nm) <= 0.55
        changed[mask] *= 1.0 + 0.01 * index

    matrix, names, _ = extract_joint_nine_fbg_features(
        changed[None, :],
        baseline,
        wavelength,
        windows,
    )
    lookup = dict(zip(names, matrix[0], strict=True))
    for window in windows:
        assert lookup[f"joint_response_energy_{window.candidate_id.lower()}"] > 0.0


def test_joint_fingerprint_rejects_missing_channel() -> None:
    wavelength, baseline, windows = _fixture()
    try:
        extract_joint_nine_fbg_features(
            baseline[None, :],
            baseline,
            wavelength,
            windows[:-1],
        )
    except ValueError as exc:
        assert "exactly nine" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing FBG window was accepted")
