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
    assert np.allclose(gated, [0.0, 1.2, 5.0])
