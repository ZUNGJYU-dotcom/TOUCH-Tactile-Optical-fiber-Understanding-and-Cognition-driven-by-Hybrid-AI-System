from __future__ import annotations

import numpy as np

from src.hybrid_spectrum.literature_guided_domain_models import (
    LiteratureGuidedPositionEnsemble,
    StrictCrossDateDataset,
    apply_affine_calibration,
    build_literature_feature_views,
    causal_temporal_summary,
    fit_supervised_osc,
    literature_runtime_contact_features,
    learn_group_oof_blend_weight,
    nonnegative_prediction,
    standard_normal_variate,
)


class _ProbabilityModel:
    def __init__(self, probabilities: list[list[float]]) -> None:
        self.classes_ = np.asarray(["P11", "P22"])
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self._probabilities[: len(values)]


def test_standard_normal_variate_is_finite_and_row_centered() -> None:
    values = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]], dtype=float
    )
    transformed = standard_normal_variate(values)
    assert np.all(np.isfinite(transformed))
    assert np.allclose(np.mean(transformed, axis=1), 0.0, atol=1.0e-6)
    assert np.allclose(transformed[1], 0.0)


def test_robust_snv_handles_impulse_and_constant_rows() -> None:
    values = np.asarray(
        [[0.0, 0.0, 0.0, 20.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0]],
        dtype=float,
    )
    transformed = standard_normal_variate(values, robust=True)
    assert transformed.shape == values.shape
    assert np.all(np.isfinite(transformed))
    assert np.allclose(transformed[1], 0.0)


def test_causal_temporal_summary_does_not_cross_groups() -> None:
    signals = np.asarray([[1.0], [2.0], [100.0], [101.0]], dtype=np.float32)
    groups = np.asarray(["a", "a", "b", "b"])
    sample_index = np.asarray([0, 1, 0, 1])
    summary = causal_temporal_summary(signals, groups, sample_index, window_frames=5)
    assert summary.shape == (4, 4)
    assert summary[0, 0] == 1.0
    assert summary[2, 0] == 100.0
    assert summary[2, 2] == 0.0


def test_feature_views_have_expected_shapes_and_no_nan() -> None:
    rows = 8
    features = np.tile(np.linspace(-1.0, 1.0, 264), (rows, 1)).astype(
        np.float32
    )
    dataset = StrictCrossDateDataset(
        features=features,
        feature_names=np.asarray([f"feature_{index}" for index in range(264)]),
        wavelength_nm=np.linspace(1530.0, 1585.0, 512),
        force_fz_n=np.linspace(0.0, 7.0, rows),
        contact_target=np.asarray([0, 0, 1, 1, 0, 1, 1, 0]),
        position_target=np.asarray(["P11"] * rows),
        force_mask=np.ones(rows, dtype=bool),
        contact_mask=np.ones(rows, dtype=bool),
        position_mask=np.ones(rows, dtype=bool),
        group_id=np.asarray(["a"] * 4 + ["b"] * 4),
        acquisition_date=np.asarray(["d1"] * 4 + ["d2"] * 4),
        sample_index=np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
        elapsed_time_sec=np.arange(rows, dtype=float),
    )
    views = build_literature_feature_views(dataset)
    assert views["reference_full264"].values.shape == (rows, 264)
    assert views["response_raw136"].values.shape == (rows, 136)
    assert views["literature_snv_sg328"].values.shape == (rows, 328)
    assert views["literature_snv_sg_temporal488"].values.shape == (rows, 488)
    assert all(np.all(np.isfinite(view.values)) for view in views.values())


def test_runtime_contact_features_match_offline_last_row() -> None:
    rows = 6
    features = np.tile(np.linspace(-0.8, 1.2, 264), (rows, 1)).astype(
        np.float32
    )
    features += np.arange(rows, dtype=np.float32)[:, None] * 0.01
    dataset = StrictCrossDateDataset(
        features=features,
        feature_names=np.asarray([f"feature_{index}" for index in range(264)]),
        wavelength_nm=np.linspace(1530.0, 1585.0, 512),
        force_fz_n=np.linspace(0.0, 2.0, rows),
        contact_target=np.asarray([0, 0, 1, 1, 1, 0]),
        position_target=np.asarray(["P22"] * rows),
        force_mask=np.ones(rows, dtype=bool),
        contact_mask=np.ones(rows, dtype=bool),
        position_mask=np.ones(rows, dtype=bool),
        group_id=np.asarray(["one"] * rows),
        acquisition_date=np.asarray(["d1"] * rows),
        sample_index=np.arange(rows),
        elapsed_time_sec=np.arange(rows, dtype=float),
    )
    offline = build_literature_feature_views(
        dataset, temporal_window_frames=5
    )["literature_snv_sg_temporal488"].values[-1:]
    runtime = literature_runtime_contact_features(
        features[-5:], temporal_window_frames=5
    )
    assert runtime.shape == (1, 488)
    assert np.allclose(runtime, offline, atol=1.0e-6)


def test_position_ensemble_averages_probability_evidence() -> None:
    raw = _ProbabilityModel([[0.8, 0.2], [0.1, 0.9]])
    normalized = _ProbabilityModel([[0.4, 0.6], [0.7, 0.3]])
    ensemble = LiteratureGuidedPositionEnsemble(raw, normalized)
    strict_features = np.zeros((2, 264), dtype=np.float32)

    probabilities = ensemble.predict_proba(strict_features)

    assert np.allclose(probabilities, [[0.6, 0.4], [0.4, 0.6]])
    assert ensemble.predict(strict_features).tolist() == ["P11", "P22"]


def test_group_oof_blend_weight_is_valid() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.asarray([f"g{index}" for index in range(8)]), 6)
    features = rng.normal(size=(len(groups), 8))
    target = np.maximum(0.0, 1.5 * features[:, 0] + 0.2 * features[:, 1])
    result = learn_group_oof_blend_weight(
        tree_features=features,
        latent_features=features,
        target=target,
        groups=groups,
        folds=4,
        estimators=16,
        maximum_rows_per_group=6,
    )
    assert 0.0 <= result["tree_weight"] <= 1.0
    assert np.isclose(result["tree_weight"] + result["latent_weight"], 1.0)
    assert len(result["folds"]) == 4


def test_nonnegative_prediction_has_no_upper_clip() -> None:
    result = nonnegative_prediction(np.asarray([-1.0, 2.0, 8.5]))
    assert np.allclose(result, [0.0, 2.0, 8.5])


def test_affine_calibration_preserves_values_above_five_newtons() -> None:
    result = apply_affine_calibration(
        np.asarray([0.5, 3.0, 6.0]), slope=1.1, intercept_n=0.2
    )
    assert np.allclose(result, [0.75, 3.5, 6.8])


def test_supervised_osc_uses_train_fit_and_reduces_orthogonal_nuisance() -> None:
    response = np.linspace(-1.0, 1.0, 80)
    nuisance = np.sin(np.linspace(0.0, 8.0 * np.pi, 80)) * 7.0
    train = np.column_stack(
        [
            response + 0.05 * nuisance,
            nuisance,
            0.5 * response - 0.7 * nuisance,
        ]
    )
    test = train[:12] + np.asarray([0.1, 0.2, -0.1])
    transformer, corrected_train, corrected_test = fit_supervised_osc(
        train,
        response,
        test,
        components=1,
    )
    assert transformer.n_components_ == 1
    assert corrected_train.shape == train.shape
    assert corrected_test.shape == test.shape
    assert np.all(np.isfinite(corrected_test))
    before = abs(np.corrcoef(train[:, 1], nuisance)[0, 1])
    after = abs(np.corrcoef(corrected_train[:, 1], nuisance)[0, 1])
    assert after < before
