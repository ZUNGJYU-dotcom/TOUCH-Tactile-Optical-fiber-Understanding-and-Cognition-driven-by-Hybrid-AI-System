"""Grouped evaluation utilities for spectral fingerprint classifiers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def grouped_stratified_holdout(
    labels: Iterable[str],
    groups: Iterable[str],
    test_size: float = 0.25,
    random_state: int = 42,
    attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a group-exclusive holdout containing every class on both sides."""

    y = np.asarray(list(labels), dtype=object)
    group_array = np.asarray(list(groups), dtype=object)
    if y.size != group_array.size or y.size == 0:
        raise ValueError("labels and groups must be non-empty and have equal length")
    classes = set(y.tolist())
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    class_total = Counter(y.tolist())
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + offset,
        )
        train_index, test_index = next(splitter.split(np.zeros(y.size), y, group_array))
        if set(y[train_index].tolist()) != classes or set(y[test_index].tolist()) != classes:
            continue
        if set(group_array[train_index].tolist()) & set(group_array[test_index].tolist()):
            continue
        train_counts = Counter(y[train_index].tolist())
        test_counts = Counter(y[test_index].tolist())
        score = 0.0
        for label in classes:
            target_fraction = class_total[label] / y.size
            score += abs(train_counts[label] / train_index.size - target_fraction)
            score += abs(test_counts[label] / test_index.size - target_fraction)
        if best is None or score < best[0]:
            best = (score, train_index, test_index)
    if best is None:
        raise ValueError(
            "could not create a grouped holdout containing every class; collect more independent trials"
        )
    return best[1], best[2]


def majority_vote(labels: Iterable[str]) -> str:
    counts = Counter(str(label) for label in labels)
    if not counts:
        raise ValueError("cannot vote over an empty label set")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
