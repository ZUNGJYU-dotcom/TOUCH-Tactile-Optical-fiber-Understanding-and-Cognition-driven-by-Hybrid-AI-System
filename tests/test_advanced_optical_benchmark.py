from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.hybrid_spectrum.advanced_optical_benchmark import (
    build_feature_views,
    equal_group_weights,
    load_aligned_latest_primary,
)


def _write_datasets(tmp_path: Path) -> tuple[Path, Path]:
    fusion_path = tmp_path / "fusion.npz"
    spectrum_path = tmp_path / "spectrum.npz"
    feature_names = np.asarray(
        [f"mean__feature_{index:03d}" for index in range(443)]
        + [f"last__feature_{index:03d}" for index in range(40)]
    )
    np.savez_compressed(
        fusion_path,
        features=np.arange(4 * 483, dtype=np.float32).reshape(4, 483),
        feature_names=feature_names,
        contact_target=np.asarray([0, 1, 1, 0], dtype=np.int8),
        position_target=np.asarray(["", "P11", "P22", ""]),
        force_fz_n=np.asarray([0.0, 1.0, 2.0, 0.0], dtype=np.float32),
        contact_mask=np.ones(4, dtype=bool),
        position_mask=np.asarray([False, True, True, False]),
        force_mask=np.ones(4, dtype=bool),
        formal_test_eligible=np.asarray([True, True, True, False]),
        fold_id=np.asarray([0, 1, 2, -1], dtype=np.int8),
        group_id=np.asarray(["session_a", "session_b", "session_c", "old"]),
        sample_index=np.asarray([10, 20, 30, 40], dtype=np.int32),
    )
    order = np.asarray([2, 0, 1])
    np.savez_compressed(
        spectrum_path,
        features=np.arange(3 * 264, dtype=np.float32).reshape(3, 264)[order],
        feature_names=np.asarray([f"spectrum_{index:03d}" for index in range(264)]),
        contact_target=np.asarray([0, 1, 1], dtype=np.int8)[order],
        position_target=np.asarray(["", "P11", "P22"])[order],
        force_fz_n=np.asarray([0.0, 1.0, 2.0], dtype=np.float32)[order],
        fold_id=np.asarray([0, 1, 2], dtype=np.int8)[order],
        session_id=np.asarray(["session_a", "session_b", "session_c"])[order],
        capture_index=np.asarray([10, 20, 30], dtype=np.int32)[order],
    )
    return fusion_path, spectrum_path


def test_alignment_uses_session_and_frame_keys(tmp_path: Path) -> None:
    fusion_path, spectrum_path = _write_datasets(tmp_path)

    dataset = load_aligned_latest_primary(fusion_path, spectrum_path)

    assert dataset.group_id.tolist() == ["session_a", "session_b", "session_c"]
    assert dataset.sample_index.tolist() == [10, 20, 30]
    assert dataset.spectrum_features[:, 0].tolist() == [0.0, 264.0, 528.0]
    assert dataset.fold_id.tolist() == [0, 1, 2]


def test_feature_views_have_expected_dimensions(tmp_path: Path) -> None:
    fusion_path, spectrum_path = _write_datasets(tmp_path)
    dataset = load_aligned_latest_primary(fusion_path, spectrum_path)

    views = build_feature_views(dataset)

    assert {name: matrix.shape for name, matrix in views.items()} == {
        "peak_current_40": (3, 40),
        "peak_temporal_483": (3, 483),
        "full_spectrum_192": (3, 192),
        "full_spectrum_264": (3, 264),
        "peak_temporal_plus_spectrum_192": (3, 675),
        "peak_temporal_plus_spectrum_264": (3, 747),
    }


def test_equal_group_weights_assign_equal_total_mass() -> None:
    groups = np.asarray(["a", "a", "a", "b"])

    weights = equal_group_weights(groups)

    assert np.sum(weights[groups == "a"]) == pytest.approx(
        np.sum(weights[groups == "b"])
    )
    assert np.mean(weights) == pytest.approx(1.0)
