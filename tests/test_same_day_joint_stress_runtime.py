from __future__ import annotations

import numpy as np

from scripts.audit_same_day_joint_fingerprint_stress import (
    POSITION_ORDER,
    _runtime_contact_state,
    _runtime_position_state,
)


def test_contact_hysteresis_requires_two_low_frames_to_release() -> None:
    probability = np.asarray([0.79, 0.80, 0.70, 0.40, 0.40], dtype=float)
    sessions = np.asarray(["session"] * len(probability), dtype=str)

    state = _runtime_contact_state(probability, sessions)

    np.testing.assert_array_equal(state, [False, True, True, True, False])


def test_position_switch_requires_two_confident_matching_frames() -> None:
    labels = np.asarray(POSITION_ORDER, dtype=str)
    probability = np.zeros((5, len(labels)), dtype=float)
    p11 = list(labels).index("P11")
    p21 = list(labels).index("P21")
    probability[0, p11] = 0.9
    probability[1, p21] = 0.8
    probability[2, p11] = 0.7
    probability[3, p21] = 0.8
    probability[4, p21] = 0.8
    sessions = np.asarray(["session"] * len(probability), dtype=str)

    state = _runtime_position_state(
        probability,
        labels,
        np.ones(len(probability), dtype=bool),
        sessions,
    )

    np.testing.assert_array_equal(state, ["P11", "P11", "P11", "P11", "P21"])
