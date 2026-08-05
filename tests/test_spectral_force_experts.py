from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid_spectrum.advanced_optical_benchmark import AlignedOpticalDataset
from src.hybrid_spectrum.spectral_force_experts import (
    SpectralForceExpertSpec,
    apply_spectral_expert_override,
    build_baseline_conditioned_spectral_views,
    grouped_spectral_force_expert_oof,
    load_group_baseline_spectral_views,
)


def _synthetic_dataset(group_count: int = 4) -> AlignedOpticalDataset:
    group_ids = [f"P13_g{index}" for index in range(group_count)]
    groups = np.repeat(group_ids, 8)
    folds = np.repeat(np.arange(group_count), 8)
    force = np.tile(np.linspace(0.2, 4.0, 8), group_count)
    features = np.zeros((len(force), 264), dtype=np.float32)
    features[:, 0] = force
    features[:, 1] = force**2
    return AlignedOpticalDataset(
        peak_features=np.zeros((len(force), 40), dtype=np.float32),
        peak_feature_names=np.asarray([f"last__f{i}" for i in range(40)]),
        spectrum_features=features,
        spectrum_feature_names=np.asarray([f"s{i}" for i in range(264)]),
        contact_target=np.ones(len(force), dtype=int),
        position_target=np.asarray(["P13"] * len(force)),
        force_fz_n=force,
        contact_mask=np.ones(len(force), dtype=bool),
        position_mask=np.ones(len(force), dtype=bool),
        force_mask=np.ones(len(force), dtype=bool),
        fold_id=folds,
        group_id=groups,
        sample_index=np.tile(np.arange(8), group_count),
    )


def test_grouped_spectral_expert_routes_from_optical_vote_without_leakage() -> None:
    dataset = _synthetic_dataset()
    indices = np.arange(len(dataset.group_id)) + 100
    votes = {group: "P13" for group in set(dataset.group_id.tolist())}
    gates = {int(index): True for index in indices}
    predictions, audit = grouped_spectral_force_expert_oof(
        dataset,
        indices,
        SpectralForceExpertSpec(latent_components=2),
        group_position_votes=votes,
        gate_active_by_array_index=gates,
    )
    assert len(predictions) == len(dataset.group_id)
    assert predictions["array_index"].is_unique
    assert (predictions["position_condition"] == "P13").all()
    assert (audit["train_test_group_overlap_count"] == 0).all()
    assert float(np.mean(np.abs(predictions["spectral_gated_force_n"] - dataset.force_fz_n))) < 0.1


def test_grouped_spectral_expert_accepts_aligned_custom_feature_matrix() -> None:
    dataset = _synthetic_dataset()
    indices = np.arange(len(dataset.group_id)) + 500
    custom = np.column_stack(
        (dataset.force_fz_n, dataset.force_fz_n**2, np.ones(len(indices)))
    )
    votes = {group: "P13" for group in set(dataset.group_id.tolist())}
    gates = {int(index): True for index in indices}
    predictions, audit = grouped_spectral_force_expert_oof(
        dataset,
        indices,
        SpectralForceExpertSpec(feature_count=3, latent_components=2),
        group_position_votes=votes,
        gate_active_by_array_index=gates,
        feature_matrix=custom,
    )
    assert len(predictions) == len(dataset.group_id)
    assert (audit["train_test_group_overlap_count"] == 0).all()
    assert float(
        np.mean(
            np.abs(predictions["spectral_gated_force_n"] - dataset.force_fz_n)
        )
    ) < 0.1


def test_grouped_spectral_expert_excludes_flagged_group_from_train_and_test() -> None:
    dataset = _synthetic_dataset(group_count=5)
    indices = np.arange(len(dataset.group_id)) + 900
    votes = {group: "P13" for group in set(dataset.group_id.tolist())}
    gates = {int(index): True for index in indices}
    excluded_group = "P13_g3"

    predictions, audit = grouped_spectral_force_expert_oof(
        dataset,
        indices,
        SpectralForceExpertSpec(latent_components=2),
        group_position_votes=votes,
        gate_active_by_array_index=gates,
        excluded_group_ids=[excluded_group],
    )

    assert excluded_group not in set(predictions["group_id"].astype(str))
    assert len(predictions) == len(dataset.group_id) - 8
    assert (audit["excluded_group_count"] == 1).all()
    assert (audit["train_test_group_overlap_count"] == 0).all()


def test_baseline_spectral_views_are_group_aligned(tmp_path) -> None:
    dataset = _synthetic_dataset()
    for offset, group_id in enumerate(sorted(set(dataset.group_id.tolist()))):
        source = tmp_path / group_id
        source.mkdir()
        pd.DataFrame(
            {
                "point_index": np.arange(64),
                "baseline_intensity_counts": np.linspace(
                    1000 + 100 * offset, 2000 + 100 * offset, 64
                ),
            }
        ).to_csv(source / "spectrum_timeseries.csv", index=False)
    baseline = load_group_baseline_spectral_views(
        tmp_path, dataset.group_id, bin_count=64
    )
    assert baseline["baseline_log_counts"].shape == (len(dataset.group_id), 64)
    np.testing.assert_allclose(
        baseline["baseline_log_counts"][0],
        baseline["baseline_log_counts"][1],
    )
    conditioned = build_baseline_conditioned_spectral_views(dataset, tmp_path)
    assert conditioned["current264_plus_baseline"].shape[0] == len(
        dataset.group_id
    )
    assert np.isfinite(conditioned["lag_delta_plus_baseline_interaction"]).all()


def test_spectral_override_changes_only_routed_rows_and_respects_gate() -> None:
    baseline = pd.DataFrame(
        {
            "array_index": [1, 2, 3],
            "model_id": ["old"] * 3,
            "base_raw_force_n": [0.2, 0.3, 0.4],
            "calibrated_force_n": [0.2, 0.3, 0.4],
            "gated_force_n": [0.2, 0.3, 0.4],
            "contact_gate_active": [True, False, True],
            "expert_used": ["old"] * 3,
            "position_condition_source": ["old"] * 3,
        }
    )
    expert = pd.DataFrame(
        {
            "array_index": [1, 2],
            "spectral_raw_force_n": [1.5, 2.5],
            "spectral_gated_force_n": [1.5, 0.0],
            "contact_gate_active": [True, False],
            "spectral_expert_model_id": ["pls", "pls"],
        }
    )
    result = apply_spectral_expert_override(baseline, expert, model_id="hybrid")
    np.testing.assert_allclose(result["gated_force_n"], [1.5, 0.0, 0.4])
    assert result["model_id"].eq("hybrid").all()
    assert result.loc[2, "expert_used"] == "old"
