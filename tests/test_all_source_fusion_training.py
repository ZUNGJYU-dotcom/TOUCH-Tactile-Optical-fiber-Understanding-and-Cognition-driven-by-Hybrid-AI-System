from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_spectrum.all_source_training import (
    FusionArrays,
    _split_masks,
    apply_optical_contact_gate,
    feature_indices,
    source_group_weights,
)
from hybrid_spectrum.all_source_fusion import (  # noqa: E402
    _latest_baseline,
    derive_unreferenced_optical_labels,
)


def _small_arrays() -> FusionArrays:
    names = np.asarray(
        [f"last__optical_{index}" for index in range(40)]
        + ["temporal_frame_count_log1p"],
        dtype=str,
    )
    return FusionArrays(
        features=np.zeros((8, len(names)), dtype=np.float32),
        feature_names=names,
        contact_target=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
        position_target=np.asarray(["P11"] * 8, dtype=str),
        force_fz_n=np.linspace(0.0, 1.0, 8, dtype=np.float32),
        contact_mask=np.ones(8, dtype=bool),
        position_mask=np.ones(8, dtype=bool),
        force_mask=np.ones(8, dtype=bool),
        formal_test_eligible=np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=bool,
        ),
        fold_id=np.asarray([0, 0, 1, 1, -1, -1, -1, -1], dtype=np.int16),
        source_role=np.asarray(
            ["latest_primary"] * 4 + ["legacy_dynamic"] * 4,
            dtype=str,
        ),
        group_id=np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=str),
        file_id=np.asarray(["file"] * 8, dtype=str),
        sample_index=np.arange(8, dtype=np.int32),
        elapsed_time_sec=np.arange(8, dtype=np.float32),
    )


def test_current_frame_view_has_only_40_optical_features() -> None:
    arrays = _small_arrays()
    selected = feature_indices(arrays.feature_names, "current_frame")
    assert len(selected) == 40
    assert all(
        arrays.feature_names[index].startswith("last__") for index in selected
    )


def test_grouped_split_has_no_formal_group_overlap() -> None:
    arrays = _small_arrays()
    train, test = _split_masks(
        arrays, arrays.contact_mask, fold=0, source_regime="all_sources"
    )
    assert set(arrays.group_id[train]).isdisjoint(arrays.group_id[test])
    assert set(arrays.group_id[test]) == {"a"}
    assert set(arrays.group_id[train]) == {"b", "c", "d"}


def test_source_weights_balance_groups_without_zero_weight() -> None:
    arrays = _small_arrays()
    selected = np.ones(8, dtype=bool)
    weights = source_group_weights(
        arrays.source_role,
        arrays.group_id,
        selected,
        {
            "latest_primary": {"source_weight": 1.0},
            "legacy_dynamic": {"source_weight": 0.5},
        },
    )
    assert np.all(weights > 0.0)
    assert np.isclose(np.mean(weights), 1.0)
    assert np.isclose(np.sum(weights[arrays.group_id == "a"]), np.sum(weights[arrays.group_id == "b"]))
    assert np.isclose(np.sum(weights[arrays.group_id == "c"]), np.sum(weights[arrays.group_id == "d"]))


def test_optical_contact_gate_outputs_zero_without_contact_evidence() -> None:
    gated = apply_optical_contact_gate(
        np.asarray([0.8, 1.2, 6.0], dtype=float),
        np.asarray([0.20, 0.75, 0.95], dtype=float),
        probability_threshold=0.75,
        no_contact_output_n=0.0,
    )
    assert np.allclose(gated, [0.0, 1.2, 6.0])


def test_unreferenced_labels_keep_transitions_out_of_training() -> None:
    baseline = np.zeros((5, 4), dtype=float)
    transition = np.full((2, 4), 2.0, dtype=float)
    stable_contact = np.full((5, 4), 8.0, dtype=float)
    components = np.vstack((baseline, transition, stable_contact, baseline))
    contact, position, _, _ = derive_unreferenced_optical_labels(
        components,
        np.arange(5),
        "P22",
        {
            "minimum_component_scale": 1.0,
            "inactive_max_robust_z": 1.0,
            "active_min_robust_z": 4.0,
            "active_session_percentile": 60.0,
            "minimum_active_run_frames": 3,
            "minimum_active_frames": 3,
            "minimum_session_contrast_z": 1.0,
            "smoothing_frames": 1,
        },
    )
    assert np.all(contact[:5] == 0)
    assert np.all(contact[5:7] == -1)
    assert np.all(contact[7:12] == 1)
    assert np.all(position[7:12] == "P22")


def test_runtime_baseline_is_independent_of_force_reference() -> None:
    recorded = np.asarray([100.0, 200.0, 150.0])
    intensity = np.vstack(
        (
            recorded * 1.01,
            recorded * 1.02,
            recorded * 1.01,
            recorded * 0.80,
            recorded * 0.70,
        )
    )
    config = {
        "strategy": "fixed_recorded_runtime_preferred",
        "minimum_frames": 3,
        "search_fraction": 0.60,
        "maximum_recorded_vs_initial_nrms": 0.05,
        "minimum_recorded_vs_initial_correlation": 0.98,
    }

    low_force, low_indices, low_mode = _latest_baseline(
        np.asarray([0.0, 0.0, 0.1, 2.0, 3.0]),
        intensity,
        config,
        0.03,
        recorded_baseline=recorded,
    )
    preloaded, preloaded_indices, preloaded_mode = _latest_baseline(
        np.asarray([2.4, 2.5, 2.6, 3.0, 4.0]),
        intensity,
        config,
        0.03,
        recorded_baseline=recorded,
    )

    assert np.array_equal(low_force, recorded)
    assert np.array_equal(preloaded, recorded)
    assert np.array_equal(low_indices, preloaded_indices)
    assert low_mode == "fixed_recorded_runtime_baseline"
    assert preloaded_mode == low_mode


def test_runtime_baseline_falls_back_when_recorded_shape_is_incompatible() -> None:
    initial = np.asarray([100.0, 200.0, 150.0])
    intensity = np.vstack((initial, initial * 1.01, initial * 0.99))
    incompatible = np.asarray([200.0, 100.0, 300.0])

    baseline, indices, mode = _latest_baseline(
        np.asarray([0.0, 0.0, 0.0]),
        intensity,
        {
            "strategy": "fixed_recorded_runtime_preferred",
            "minimum_frames": 3,
            "search_fraction": 1.0,
            "maximum_recorded_vs_initial_nrms": 0.05,
            "minimum_recorded_vs_initial_correlation": 0.98,
        },
        0.03,
        recorded_baseline=incompatible,
    )

    assert np.allclose(baseline, initial)
    assert np.array_equal(indices, [0, 1, 2])
    assert mode == "initial_optical_stable_fallback"


def test_initial_fixed_frame_baseline_uses_only_precontact_frames() -> None:
    initial = np.asarray([100.0, 200.0, 150.0])
    intensity = np.vstack(
        (
            initial,
            initial * 1.01,
            initial * 0.99,
            initial * 0.80,
            initial * 0.70,
        )
    )

    baseline_a, indices_a, mode_a = _latest_baseline(
        np.asarray([0.0, 0.0, 0.0, 2.0, 3.0]),
        intensity,
        {"strategy": "initial_fixed_frames", "fixed_initial_frames": 3},
        0.03,
    )
    baseline_b, indices_b, mode_b = _latest_baseline(
        np.asarray([2.0, 3.0, 4.0, 0.0, 0.0]),
        intensity,
        {"strategy": "initial_fixed_frames", "fixed_initial_frames": 3},
        0.03,
    )

    assert np.allclose(baseline_a, initial)
    assert np.array_equal(baseline_a, baseline_b)
    assert np.array_equal(indices_a, [0, 1, 2])
    assert np.array_equal(indices_a, indices_b)
    assert mode_a == "initial_fixed_frames"
    assert mode_b == mode_a
