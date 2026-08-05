from __future__ import annotations

import numpy as np

from src.hybrid_spectrum.force_weighting_benchmark import (
    combine_training_weights,
    force_curve_metrics,
    force_weight_multiplier,
    _session_position_labels,
)
from src.hybrid_spectrum.all_source_training import FusionArrays


def test_linear_weighting_emphasizes_high_force() -> None:
    force = np.asarray([0.0, 1.0, 3.0, 5.0])
    groups = np.asarray(["a", "a", "a", "a"])
    multiplier = force_weight_multiplier("linear4", force, groups)
    assert np.all(np.diff(multiplier) > 0.0)
    assert multiplier[0] == 1.0
    assert multiplier[-1] == 4.0


def test_group_bin_weighting_equalizes_bin_mass() -> None:
    force = np.asarray([0.0, 0.0, 0.0, 1.2, 4.4])
    groups = np.asarray(["a"] * 5)
    multiplier = force_weight_multiplier("group_bin_full", force, groups)
    bin_mass = [
        float(np.sum(multiplier[:3])),
        float(multiplier[3]),
        float(multiplier[4]),
    ]
    assert np.allclose(bin_mass, bin_mass[0])


def test_combined_weights_are_positive_and_normalized() -> None:
    combined = combine_training_weights(
        np.asarray([0.5, 0.5, 2.0, 2.0]),
        np.asarray([0.0, 2.0, 3.0, 5.0]),
        np.asarray(["a", "a", "b", "b"]),
        "square4",
    )
    assert np.all(combined > 0.0)
    assert np.isclose(np.mean(combined), 1.0)


def test_force_curve_metrics_expose_amplitude_and_residual() -> None:
    true = np.asarray([0.0, 0.0, 1.0, 3.0, 5.0])
    predicted = np.asarray([0.2, 0.0, 0.9, 2.5, 4.0])
    metrics = force_curve_metrics(true, predicted)
    assert metrics["paired_frame_count"] == 5
    assert metrics["high_force_frame_count"] == 2
    assert metrics["zero_force_false_response_rate"] == 0.5
    assert 0.0 < metrics["calibration_slope"] < 1.0
    assert 0.0 < metrics["amplitude_ratio_p95_p05"] < 1.0


def test_session_position_recovers_baseline_frame_from_group_id() -> None:
    arrays = FusionArrays(
        features=np.zeros((2, 1), dtype=np.float32),
        feature_names=np.asarray(["last__x"]),
        contact_target=np.zeros(2, dtype=np.int8),
        position_target=np.asarray(["", "P13"]),
        force_fz_n=np.zeros(2, dtype=np.float32),
        contact_mask=np.ones(2, dtype=bool),
        position_mask=np.asarray([False, True]),
        force_mask=np.ones(2, dtype=bool),
        formal_test_eligible=np.ones(2, dtype=bool),
        fold_id=np.zeros(2, dtype=np.int16),
        source_role=np.asarray(["latest_primary", "latest_primary"]),
        group_id=np.asarray(["session_P13_trial_001", "session_P13_trial_001"]),
        file_id=np.asarray(["trace.csv", "trace.csv"]),
        sample_index=np.asarray([0, 1], dtype=np.int32),
        elapsed_time_sec=np.asarray([0.0, 0.1], dtype=np.float32),
    )
    labels = _session_position_labels(arrays, np.asarray([True, True]))
    assert labels.tolist() == ["P13", "P13"]
