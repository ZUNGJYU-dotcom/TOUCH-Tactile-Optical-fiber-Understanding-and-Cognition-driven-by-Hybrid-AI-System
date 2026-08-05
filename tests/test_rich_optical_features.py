from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.hybrid_spectrum.rich_optical_features import (
    CANDIDATE_PREFIXES,
    RichFeatureCache,
    add_coupling_features,
    load_rich_feature_cache,
    save_rich_feature_cache,
)


def _base_features() -> dict[str, float]:
    values: dict[str, float] = {}
    for index, prefix in enumerate(CANDIDATE_PREFIXES, start=1):
        values[f"{prefix}_area_ratio"] = float(np.exp(index / 10.0))
        values[f"{prefix}_height_ratio"] = float(np.exp(index / 20.0))
        values[f"{prefix}_quality_fused_shift_pm"] = float(index)
        values[f"{prefix}_fused_common_mode_corrected_shift_pm"] = float(index - 5)
        values[f"{prefix}_delta_fwhm_pm"] = float(index * 2)
        values[f"{prefix}_delta_skewness"] = float(index / 10.0)
        values[f"{prefix}_normalized_shape_rmse"] = float(index / 100.0)
    return values


def test_coupling_features_preserve_physical_group_directions() -> None:
    enriched = add_coupling_features(_base_features())

    assert enriched["fbg01_log_area_ratio"] == pytest.approx(0.1)
    assert enriched["fbg09_log_area_ratio"] == pytest.approx(0.9)
    assert enriched[
        "coupling_log_area_ratio_same_fibre_fibre_1_gradient"
    ] == pytest.approx(0.2)
    assert enriched[
        "coupling_log_area_ratio_spatial_row_row_1_gradient"
    ] == pytest.approx(0.6)
    assert enriched[
        "coupling_quality_fused_shift_pm_fbg01_residual"
    ] == pytest.approx(-4.0)


def test_coupling_features_sanitize_unreliable_tracking_values() -> None:
    values = _base_features()
    values["fbg03_quality_fused_shift_pm"] = float("nan")

    enriched = add_coupling_features(values)

    assert all(np.isfinite(value) for value in enriched.values())
    assert enriched["global_nonfinite_physics_feature_count"] >= 1.0


def test_rich_feature_cache_round_trip_and_alignment_guard(tmp_path: Path) -> None:
    path = tmp_path / "rich_features.npz"
    cache = RichFeatureCache(
        features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        feature_names=np.asarray(["a", "b"]),
        group_id=np.asarray(["g1", "g2"]),
        sample_index=np.asarray([10, 20]),
    )
    save_rich_feature_cache(cache, path)

    loaded = load_rich_feature_cache(
        path,
        expected_group_id=np.asarray(["g1", "g2"]),
        expected_sample_index=np.asarray([10, 20]),
    )

    assert loaded.features.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert loaded.feature_names.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="group order"):
        load_rich_feature_cache(
            path,
            expected_group_id=np.asarray(["g2", "g1"]),
        )
