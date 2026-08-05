from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid_spectrum.position_conditioned_force import (
    POSITION_CONDITION_ORDER,
    UNKNOWN_POSITION,
    append_position_condition,
    causal_contact_reset_ema,
    equal_group_auxiliary_weights,
    grouped_position_vote,
    infer_group_position_labels,
    one_hot_position,
)


def test_position_inference_prefers_group_and_has_explicit_unknown() -> None:
    labels = infer_group_position_labels(
        ["session_P13_trial_1", "unlabeled", "session_P22_trial_1"],
        ["P11", "", "P31"],
    )
    assert labels.tolist() == ["P13", UNKNOWN_POSITION, "P22"]


def test_one_hot_position_order_is_stable() -> None:
    encoded = one_hot_position(["P11", "P33", "invalid"])
    assert encoded.shape == (3, len(POSITION_CONDITION_ORDER))
    assert int(np.argmax(encoded[0])) == POSITION_CONDITION_ORDER.index("P11")
    assert int(np.argmax(encoded[1])) == POSITION_CONDITION_ORDER.index("P33")
    assert int(np.argmax(encoded[2])) == POSITION_CONDITION_ORDER.index(
        UNKNOWN_POSITION
    )


def test_append_position_condition_keeps_optical_values() -> None:
    optical = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    combined = append_position_condition(optical, ["P11", "P21"])
    np.testing.assert_allclose(combined[:, :2], optical)
    assert combined.shape[1] == 2 + len(POSITION_CONDITION_ORDER)


def test_grouped_position_vote_uses_only_requested_model() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["chosen", "chosen", "chosen", "other"],
            "task": ["position"] * 4,
            "group_id": ["g1", "g1", "g1", "g1"],
            "predicted_label": ["P12", "P12", "P22", "P33"],
        }
    )
    assert grouped_position_vote(frame, model_id="chosen") == {"g1": "P12"}


def test_causal_ema_resets_without_release_residual() -> None:
    smoothed = causal_contact_reset_ema(
        [0.0, 2.0, 4.0, 4.0, 3.0],
        [False, True, True, False, True],
        ["g"] * 5,
        [0, 1, 2, 3, 4],
        alpha=0.5,
    )
    np.testing.assert_allclose(smoothed, [0.0, 1.0, 2.5, 0.0, 1.5])


def test_causal_ema_tracks_groups_independently() -> None:
    smoothed = causal_contact_reset_ema(
        [2.0, 4.0, 2.0, 4.0],
        [True, True, True, True],
        ["a", "a", "b", "b"],
        [0, 1, 0, 1],
        alpha=0.5,
    )
    np.testing.assert_allclose(smoothed, [1.0, 2.5, 1.0, 2.5])


def test_auxiliary_weights_give_sessions_equal_mass() -> None:
    weights = equal_group_auxiliary_weights(
        ["short", "long", "long", "long"],
        total_mass=0.4,
    )
    np.testing.assert_allclose(weights, [0.2, 0.2 / 3, 0.2 / 3, 0.2 / 3])
    assert np.isclose(weights.sum(), 0.4)
