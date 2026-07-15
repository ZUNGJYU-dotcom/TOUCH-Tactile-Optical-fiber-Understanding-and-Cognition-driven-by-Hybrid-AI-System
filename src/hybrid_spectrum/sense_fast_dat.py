"""Read BaySpec Sense fast-recording DAT spectra without frame drift.

Sense exports observed in this project contain either 512 spectral words per
record or 513 words where the last word is a recorder synchronization word.
Some older files may also have a 256-word prefix.  Inferring a 512-word layout
from file size alone corrupts 513-word recordings: every decoded spectrum then
starts one pixel later than the previous frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPECTRUM_WORDS = 512


@dataclass(frozen=True)
class FastDatLayout:
    """Detected binary record layout and its continuity evidence."""

    name: str
    record_words: int
    prefix_words: int
    spectrum_words: int
    frame_count: int
    trailing_words: int
    median_adjacent_correlation: float
    score_margin: float


@dataclass(frozen=True)
class SenseFastDatSequence:
    """One immutable decoded Sense fast-recording sequence."""

    path: Path
    spectra: np.ndarray
    auxiliary_words: np.ndarray
    layout: FastDatLayout


@dataclass(frozen=True)
class _LayoutCandidate:
    name: str
    record_words: int
    prefix_words: int
    frame_count: int
    trailing_words: int
    median_adjacent_correlation: float


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_float = np.asarray(left, dtype=float)
    right_float = np.asarray(right, dtype=float)
    left_centered = left_float - float(np.mean(left_float))
    right_centered = right_float - float(np.mean(right_float))
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator > 1.0e-12:
        return float(np.dot(left_centered, right_centered) / denominator)
    raw_denominator = float(np.linalg.norm(left_float) * np.linalg.norm(right_float))
    if raw_denominator <= 1.0e-12:
        return 1.0 if np.array_equal(left_float, right_float) else 0.0
    return float(np.dot(left_float, right_float) / raw_denominator)


def _candidate(
    words: np.ndarray,
    *,
    name: str,
    record_words: int,
    prefix_words: int,
) -> _LayoutCandidate | None:
    available = int(words.size) - prefix_words
    frame_count = available // record_words
    if frame_count < 2 or record_words < SPECTRUM_WORDS:
        return None
    usable = words[
        prefix_words : prefix_words + frame_count * record_words
    ].reshape(frame_count, record_words)
    spectra = usable[:, :SPECTRUM_WORDS]
    pair_count = min(frame_count - 1, 128)
    if pair_count <= 0:
        return None
    pair_indices = np.linspace(
        0,
        frame_count - 2,
        num=pair_count,
        dtype=int,
    )
    correlations = np.asarray(
        [
            _safe_correlation(spectra[index], spectra[index + 1])
            for index in pair_indices
        ],
        dtype=float,
    )
    finite = correlations[np.isfinite(correlations)]
    median_correlation = float(np.median(finite)) if finite.size else -1.0
    return _LayoutCandidate(
        name=name,
        record_words=record_words,
        prefix_words=prefix_words,
        frame_count=frame_count,
        trailing_words=available - frame_count * record_words,
        median_adjacent_correlation=median_correlation,
    )


def detect_fast_dat_layout(words: np.ndarray) -> FastDatLayout:
    """Select the layout with the strongest adjacent-spectrum continuity."""

    raw_words = np.asarray(words)
    candidates = [
        _candidate(
            raw_words,
            name="sense_513_words_512_spectrum_plus_aux",
            record_words=513,
            prefix_words=0,
        ),
        _candidate(
            raw_words,
            name="legacy_512_words_no_prefix",
            record_words=512,
            prefix_words=0,
        ),
        _candidate(
            raw_words,
            name="legacy_512_words_256_word_prefix",
            record_words=512,
            prefix_words=256,
        ),
    ]
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        raise ValueError("DAT file has fewer than two complete spectrum records")
    def selection_score(candidate: _LayoutCandidate) -> float:
        # When two 512-word alignments are numerically indistinguishable,
        # prefer the one that consumes an integer number of complete records.
        # The bonus is far below the continuity gap of a genuinely misaligned
        # 512-vs-513 recording.
        complete_record_bonus = 1.0e-6 if candidate.trailing_words == 0 else 0.0
        return candidate.median_adjacent_correlation + complete_record_bonus

    ranked = sorted(valid, key=selection_score, reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = (
        selection_score(best) - selection_score(runner_up)
        if runner_up is not None
        else 1.0
    )
    return FastDatLayout(
        name=best.name,
        record_words=best.record_words,
        prefix_words=best.prefix_words,
        spectrum_words=SPECTRUM_WORDS,
        frame_count=best.frame_count,
        trailing_words=best.trailing_words,
        median_adjacent_correlation=best.median_adjacent_correlation,
        score_margin=float(margin),
    )


def read_sense_fast_dat(path: str | Path) -> SenseFastDatSequence:
    """Decode every complete spectrum while leaving the source file untouched."""

    dat_path = Path(path)
    words = np.fromfile(dat_path, dtype=">u2")
    layout = detect_fast_dat_layout(words)
    start = layout.prefix_words
    stop = start + layout.frame_count * layout.record_words
    records = words[start:stop].reshape(layout.frame_count, layout.record_words)
    spectra = np.asarray(records[:, : layout.spectrum_words], dtype=np.uint16)
    auxiliary = (
        np.asarray(records[:, layout.spectrum_words :], dtype=np.uint16)
        if layout.record_words > layout.spectrum_words
        else np.empty((layout.frame_count, 0), dtype=np.uint16)
    )
    spectra.setflags(write=False)
    auxiliary.setflags(write=False)
    return SenseFastDatSequence(
        path=dat_path,
        spectra=spectra,
        auxiliary_words=auxiliary,
        layout=layout,
    )
