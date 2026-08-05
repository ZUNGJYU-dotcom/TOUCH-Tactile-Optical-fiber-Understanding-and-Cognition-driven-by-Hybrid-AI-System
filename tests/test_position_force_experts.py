from __future__ import annotations

import numpy as np

from src.hybrid_spectrum.position_force_experts import (
    fit_affine_force_calibrator,
    group_equal_sample_weights,
)


def test_group_equal_weights_give_each_session_equal_mass() -> None:
    weights = group_equal_sample_weights(
        ["short", "long", "long", "long", "medium", "medium"]
    )
    assert np.isclose(weights.sum(), 1.0)
    np.testing.assert_allclose(
        [
            weights[np.asarray([True, False, False, False, False, False])].sum(),
            weights[np.asarray([False, True, True, True, False, False])].sum(),
            weights[np.asarray([False, False, False, False, True, True])].sum(),
        ],
        [1.0 / 3.0] * 3,
    )


def test_affine_calibrator_recovers_known_force_scale() -> None:
    raw = np.tile(np.linspace(0.2, 2.4, 12), 4)
    true = 1.55 * raw + 0.12
    groups = np.repeat(["g1", "g2", "g3", "g4"], 12)
    calibrator = fit_affine_force_calibrator(
        raw,
        true,
        groups,
        mode="affine",
        ridge_strength=0.0,
    )
    assert calibrator.mode == "affine"
    assert np.isclose(calibrator.slope, 1.55, atol=1.0e-6)
    assert np.isclose(calibrator.intercept_n, 0.12, atol=1.0e-6)
    np.testing.assert_allclose(
        calibrator.predict([0.4, 1.2]),
        [0.74, 1.98],
        atol=1.0e-6,
    )


def test_calibrator_ignores_release_and_out_of_range_force() -> None:
    valid_raw = np.tile(np.linspace(0.2, 2.0, 10), 3)
    valid_true = 1.2 * valid_raw
    raw = np.concatenate([valid_raw, [4.5, 4.5]])
    true = np.concatenate([valid_true, [0.0, 8.0]])
    groups = np.concatenate(
        [np.repeat(["g1", "g2", "g3"], 10), ["bad_release", "bad_range"]]
    )
    calibrator = fit_affine_force_calibrator(
        raw,
        true,
        groups,
        mode="zero_anchor",
        ridge_strength=0.0,
    )
    assert np.isclose(calibrator.slope, 1.2, atol=1.0e-6)
    assert calibrator.training_sample_count == 30
    assert calibrator.training_group_count == 3


def test_calibrator_falls_back_to_identity_with_too_few_groups() -> None:
    calibrator = fit_affine_force_calibrator(
        np.linspace(0.2, 2.0, 30),
        np.linspace(0.3, 3.0, 30),
        ["g1"] * 15 + ["g2"] * 15,
        mode="affine",
    )
    assert calibrator.mode == "identity_insufficient_training_data"
    assert calibrator.slope == 1.0
    assert calibrator.intercept_n == 0.0
