from __future__ import annotations

import numpy as np
import pytest

from scripts.build_same_day_joint_fingerprint_dataset import (
    _select_session_quiet_baseline,
)


def test_quiet_baseline_uses_only_force_confirmed_quiet_frames() -> None:
    intensity = np.vstack(
        [np.full(4, value, dtype=float) for value in (2, 100, 4, 200, 6)]
    )
    contact_target = np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    contact_mask = np.ones(5, dtype=bool)

    baseline, selected = _select_session_quiet_baseline(
        intensity,
        contact_target,
        contact_mask,
        requested_frame_count=3,
    )

    np.testing.assert_array_equal(selected, [0, 2, 4])
    np.testing.assert_allclose(baseline, np.full(4, 4.0))


def test_quiet_baseline_rejects_insufficient_quiet_evidence() -> None:
    intensity = np.ones((5, 4), dtype=float)
    contact_target = np.asarray([0, 1, 1, 1, 0], dtype=np.int8)
    contact_mask = np.ones(5, dtype=bool)

    with pytest.raises(ValueError, match="fewer than three"):
        _select_session_quiet_baseline(
            intensity,
            contact_target,
            contact_mask,
            requested_frame_count=3,
        )
