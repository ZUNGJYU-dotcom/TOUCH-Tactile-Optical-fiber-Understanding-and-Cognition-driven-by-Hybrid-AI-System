from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_blind_multiepisode_record import (
    _contiguous_runs,
    _majority,
    _parse_answer_suffix,
)


def test_parse_answer_suffix_supports_idle_single_and_ordered_positions() -> None:
    assert _parse_answer_suffix("nocontact") == []
    assert _parse_answer_suffix("p22") == ["P22"]
    assert _parse_answer_suffix("p11_p21_p31") == ["P11", "P21", "P31"]


def test_parse_answer_suffix_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unrecognized answer suffix"):
        _parse_answer_suffix("p11_center_p33")


def test_contiguous_runs_keeps_separate_press_episodes() -> None:
    mask = np.array([False, True, True, False, False, True, False, True, True])
    assert _contiguous_runs(mask) == [(1, 2), (5, 5), (7, 8)]


def test_majority_is_deterministic_and_reports_margin() -> None:
    label, winner, total, share, margin = _majority(["P22", "P31", "P22", "none"])
    assert label == "P22"
    assert (winner, total) == (2, 3)
    assert share == pytest.approx(2 / 3)
    assert margin == pytest.approx(1 / 3)
