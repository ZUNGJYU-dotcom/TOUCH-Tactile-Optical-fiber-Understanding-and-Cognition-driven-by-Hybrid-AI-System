from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid_spectrum.nine_point_force_hybrid import (
    aligned_contact_gate_map,
    apply_grouped_contact_gate,
    build_session_acceptance_table,
    grouped_gate_truth_masks,
)
from src.hybrid_spectrum.advanced_optical_benchmark import AlignedOpticalDataset


def _aligned_dataset_for_gate_test() -> AlignedOpticalDataset:
    return AlignedOpticalDataset(
        peak_features=np.zeros((3, 1), dtype=float),
        peak_feature_names=np.asarray(["peak"], dtype=str),
        spectrum_features=np.zeros((3, 1), dtype=float),
        spectrum_feature_names=np.asarray(["spectrum"], dtype=str),
        contact_target=np.asarray([0, 1, 1], dtype=int),
        position_target=np.asarray(["P11", "P11", "P11"], dtype=str),
        force_fz_n=np.asarray([0.0, 4.9, 5.2], dtype=float),
        contact_mask=np.asarray([True, True, True], dtype=bool),
        position_mask=np.asarray([True, True, True], dtype=bool),
        force_mask=np.asarray([True, True, False], dtype=bool),
        fold_id=np.asarray([0, 0, 0], dtype=int),
        group_id=np.asarray(["g1", "g1", "g1"], dtype=str),
        sample_index=np.asarray([0, 1, 2], dtype=int),
    )


def test_contact_gate_alignment_skips_non_force_rows() -> None:
    dataset = _aligned_dataset_for_gate_test()
    gate = pd.DataFrame(
        {
            "group_id": ["g1", "g1"],
            "sample_index": [0, 1],
            "contact_gate_active": [False, True],
        }
    )

    result, audit = aligned_contact_gate_map(dataset, [10, 11, 12], gate)

    assert result == {10: False, 11: True}
    assert audit["formal_force_row_count"] == 2
    assert audit["mapped_row_count"] == 2
    assert audit["skipped_non_force_row_count"] == 1


def test_contact_gate_alignment_rejects_missing_force_row() -> None:
    dataset = _aligned_dataset_for_gate_test()
    gate = pd.DataFrame(
        {
            "group_id": ["g1"],
            "sample_index": [0],
            "contact_gate_active": [False],
        }
    )

    with np.testing.assert_raises_regex(
        ValueError, "force-eligible aligned rows"
    ):
        aligned_contact_gate_map(dataset, [10, 11, 12], gate)


def test_grouped_gate_truth_masks_supports_legacy_numeric_target() -> None:
    gate = pd.DataFrame({"contact_target": [0, 1, np.nan, 7]})

    available, active = grouped_gate_truth_masks(gate)

    assert available.tolist() == [True, True, False, False]
    assert active.tolist() == [False, True, False, False]


def test_grouped_gate_truth_masks_excludes_unlabelled_semantic_rows() -> None:
    gate = pd.DataFrame(
        {
            "contact_label_available": [True, True, False, "False", "True"],
            "true_contact_label": [
                "no_contact",
                "contact",
                np.nan,
                "contact",
                "active_contact",
            ],
        }
    )

    available, active = grouped_gate_truth_masks(gate)

    assert available.tolist() == [True, True, False, False, True]
    assert active.tolist() == [False, True, False, False, True]


def test_apply_grouped_contact_gate_replaces_legacy_state() -> None:
    baseline = pd.DataFrame(
        {
            "group_id": ["g1", "g1"],
            "sample_index": [0, 1],
            "calibrated_force_n": [1.5, 2.0],
            "contact_gate_active": [True, True],
            "gated_force_n": [1.5, 2.0],
        }
    )
    gate = pd.DataFrame(
        {
            "group_id": ["g1", "g1"],
            "sample_index": [0, 1],
            "contact_probability": [0.05, 0.90],
            "contact_gate_active": [False, True],
        }
    )
    result = apply_grouped_contact_gate(baseline, gate, model_id="gate")
    assert result["legacy_contact_gate_active"].tolist() == [True, True]
    assert result["contact_gate_active"].tolist() == [False, True]
    assert result["gated_force_n"].tolist() == [0.0, 2.0]
    assert not bool(result["force_sensor_used_as_runtime_input"].any())


def test_session_acceptance_allows_small_lag_but_checks_height() -> None:
    sessions = pd.DataFrame(
        {
            "position_id": ["P11", "P12"],
            "group_id": ["g1", "g2"],
            "pearson_r": [0.72, 0.95],
            "lag_aligned_pearson_r": [0.93, 0.96],
            "lag_aligned_mae_n": [0.25, 0.25],
            "linear_slope_pred_vs_px6d": [0.90, 0.40],
            "zero_force_false_response_rate": [0.02, 0.02],
            "lag_ms": [300.0, 0.0],
        }
    )
    result = build_session_acceptance_table(sessions)
    assert result.iloc[0]["session_curve_status"] == "acceptable"
    assert bool(result.iloc[0]["trend_ok"])
    assert result.iloc[1]["session_curve_status"] == "needs_review"
    assert result.iloc[1]["failure_reasons"] == "height"


def test_session_acceptance_prefers_release_grace_metric() -> None:
    sessions = pd.DataFrame(
        {
            "position_id": ["P12"],
            "group_id": ["g1"],
            "pearson_r": [0.95],
            "lag_aligned_pearson_r": [0.96],
            "lag_aligned_mae_n": [0.20],
            "linear_slope_pred_vs_px6d": [1.02],
            "zero_force_false_response_rate": [0.25],
            "zero_force_false_response_rate_after_grace": [0.02],
            "release_grace_sec": [1.0],
            "lag_ms": [300.0],
        }
    )
    result = build_session_acceptance_table(sessions)
    assert result.iloc[0]["session_curve_status"] == "acceptable"
    assert result.iloc[0]["zero_force_false_response_rate"] == 0.02
    assert result.iloc[0]["release_grace_sec"] == 1.0
