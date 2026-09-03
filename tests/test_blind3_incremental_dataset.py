from __future__ import annotations

import numpy as np

from scripts.build_blind3_incremental_dataset import (
    _contact_targets,
    _fold_for_session,
)
from scripts.train_ordinary_fbg_px6d_models import _session_majority_vote
from src.hybrid_spectrum.px6d_session_dataset import POSITION_ORDER


def test_idle_sessions_keep_residual_force_as_no_contact() -> None:
    target, mask = _contact_targets(
        np.asarray([0.0, 0.04, 0.12]),
        active_session=False,
    )
    assert target.tolist() == [0, 0, 0]
    assert mask.tolist() == [True, True, True]


def test_active_sessions_leave_force_hysteresis_band_unlabelled() -> None:
    target, mask = _contact_targets(
        np.asarray([0.02, 0.05, 0.10, 1.0]),
        active_session=True,
    )
    assert target.tolist() == [0, -1, 1, 1]
    assert mask.tolist() == [True, False, True, True]


def test_fold_assignment_keeps_ordered_sequences_whole() -> None:
    assert _fold_for_session(
        ordered_labels=tuple(POSITION_ORDER),
        idle_index=0,
    ) == 0
    assert _fold_for_session(
        ordered_labels=("P11", "P12", "P13"),
        idle_index=0,
    ) == 1
    assert _fold_for_session(ordered_labels=("P11",), idle_index=0) == 2
    assert _fold_for_session(ordered_labels=(), idle_index=2) == 4


def test_session_vote_excludes_mixed_label_sequence_sessions() -> None:
    result = _session_majority_vote(
        truth=np.asarray(["P11", "P21", "P11", "P11"]),
        predicted=np.asarray(["P11", "P21", "P11", "P11"]),
        session_ids=np.asarray(["sequence", "sequence", "single", "single"]),
        label_order=["P11", "P21"],
    )
    assert result["accuracy"] == 1.0
    assert result["evaluated_single_label_session_count"] == 1
    assert result["excluded_mixed_label_session_count"] == 1
    assert result["excluded_mixed_label_sessions"][0]["session_id"] == "sequence"
