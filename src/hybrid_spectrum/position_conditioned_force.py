"""Leakage-safe position-conditioned optical force regression.

The ordinary-FBG array has position-dependent optical sensitivity.  This
module benchmarks that effect without using PX6D force as a runtime input and
without fitting a held-out session to its own force trace.  The position fed
to a held-out fold comes from grouped OOF optical position predictions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .all_source_training import (
    FusionArrays,
    feature_indices,
    source_group_weights,
)
from .force_consistency_audit import POSITION_ORDER, infer_position_id


UNKNOWN_POSITION = "UNKNOWN"
POSITION_CONDITION_ORDER = (UNKNOWN_POSITION, *POSITION_ORDER)


@dataclass(frozen=True)
class ConditionedForceVariant:
    """One optical feature view and regressor configuration."""

    model_id: str
    feature_view: str = "current_frame"
    model_family: str = "extra_trees"
    estimators: int = 180
    minimum_leaf_samples: int = 2
    max_features: float | str = 0.8
    bootstrap: bool = False


def infer_group_position_labels(
    group_ids: Sequence[Any],
    fallback_labels: Sequence[Any] | None = None,
) -> np.ndarray:
    """Return one physical position label per row, with an explicit fallback."""

    groups = np.asarray(group_ids, dtype=str)
    fallback = (
        np.full(len(groups), "", dtype=str)
        if fallback_labels is None
        else np.asarray(fallback_labels, dtype=str)
    )
    if len(groups) != len(fallback):
        raise ValueError("group ids and fallback labels must have equal length")
    labels: list[str] = []
    for group_id, fallback_label in zip(groups, fallback):
        inferred = infer_position_id(group_id)
        if inferred in POSITION_ORDER:
            labels.append(str(inferred))
        elif fallback_label in POSITION_ORDER:
            labels.append(str(fallback_label))
        else:
            labels.append(UNKNOWN_POSITION)
    return np.asarray(labels, dtype=str)


def one_hot_position(
    labels: Sequence[Any],
    classes: Sequence[str] = POSITION_CONDITION_ORDER,
) -> np.ndarray:
    """Encode position labels using a stable, runtime-safe class order."""

    class_order = tuple(str(value) for value in classes)
    if len(set(class_order)) != len(class_order):
        raise ValueError("position classes must be unique")
    lookup = {label: index for index, label in enumerate(class_order)}
    unknown_index = lookup.get(UNKNOWN_POSITION)
    encoded = np.zeros((len(labels), len(class_order)), dtype=np.float32)
    for row, value in enumerate(labels):
        label = str(value)
        index = lookup.get(label, unknown_index)
        if index is None:
            raise ValueError(f"unknown position label without fallback: {label}")
        encoded[row, index] = 1.0
    return encoded


def append_position_condition(
    optical_features: np.ndarray,
    labels: Sequence[Any],
) -> np.ndarray:
    """Append the stable one-hot position condition to optical features."""

    optical = np.asarray(optical_features, dtype=np.float32)
    if optical.ndim != 2:
        raise ValueError("optical feature matrix must be two-dimensional")
    if len(optical) != len(labels):
        raise ValueError("optical rows and position labels must have equal length")
    return np.concatenate([optical, one_hot_position(labels)], axis=1)


def grouped_position_vote(
    predictions: pd.DataFrame,
    *,
    model_id: str,
) -> dict[str, str]:
    """Convert OOF frame position predictions into one deployable group vote."""

    required = {"model_id", "task", "group_id", "predicted_label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("position predictions missing: " + ", ".join(missing))
    selected = predictions[
        (predictions["task"].astype(str) == "position")
        & (predictions["model_id"].astype(str) == str(model_id))
    ]
    votes: dict[str, str] = {}
    for group_id, group in selected.groupby("group_id", sort=False):
        labels = [
            str(value)
            for value in group["predicted_label"].tolist()
            if str(value) in POSITION_ORDER
        ]
        if labels:
            votes[str(group_id)] = Counter(labels).most_common(1)[0][0]
    return votes


def test_position_conditions(
    arrays: FusionArrays,
    test_indices: np.ndarray,
    group_votes: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return deployable fold-test conditions and their provenance."""

    labels: list[str] = []
    sources: list[str] = []
    for index in np.asarray(test_indices, dtype=np.int64):
        group_id = str(arrays.group_id[index])
        voted = str(group_votes.get(group_id, UNKNOWN_POSITION))
        if voted in POSITION_ORDER:
            labels.append(voted)
            sources.append("grouped_oof_optical_position_vote")
        else:
            labels.append(UNKNOWN_POSITION)
            sources.append("unknown_no_position_vote")
    return np.asarray(labels, dtype=str), np.asarray(sources, dtype=str)


def causal_contact_reset_ema(
    raw_force_n: Sequence[float],
    contact_active: Sequence[bool],
    group_ids: Sequence[Any],
    sample_indices: Sequence[int],
    *,
    alpha: float,
) -> np.ndarray:
    """Smooth force causally and reset immediately when contact is inactive."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("EMA alpha must be within (0, 1]")
    raw = np.asarray(raw_force_n, dtype=float)
    active = np.asarray(contact_active, dtype=bool)
    groups = np.asarray(group_ids, dtype=str)
    samples = np.asarray(sample_indices, dtype=np.int64)
    if not (len(raw) == len(active) == len(groups) == len(samples)):
        raise ValueError("EMA inputs must have equal lengths")
    if not np.all(np.isfinite(raw)):
        raise ValueError("EMA force input must be finite")

    output = np.zeros(len(raw), dtype=float)
    order = np.lexsort((samples, groups))
    state_by_group: dict[str, float] = {}
    for index in order:
        group = groups[index]
        if not active[index]:
            state_by_group[group] = 0.0
            output[index] = 0.0
            continue
        previous = state_by_group.get(group, 0.0)
        state = float(alpha) * raw[index] + (1.0 - float(alpha)) * previous
        state = float(np.clip(state, 0.0, 5.0))
        state_by_group[group] = state
        output[index] = state
    return output


def equal_group_auxiliary_weights(
    group_ids: Sequence[Any],
    *,
    total_mass: float,
) -> np.ndarray:
    """Give every historical session equal total mass.

    ``total_mass`` is expressed relative to the total weight of the current
    fold training set.  This keeps historical data explicitly auxiliary even
    when it contains many correlated frames.
    """

    groups = np.asarray(group_ids, dtype=str)
    if len(groups) == 0:
        raise ValueError("auxiliary group ids cannot be empty")
    if not np.isfinite(total_mass) or float(total_mass) <= 0.0:
        raise ValueError("auxiliary total mass must be positive")
    unique_groups = sorted(set(groups.tolist()))
    weights = np.zeros(len(groups), dtype=np.float64)
    group_mass = float(total_mass) / len(unique_groups)
    for group_id in unique_groups:
        selected = groups == group_id
        weights[selected] = group_mass / int(np.sum(selected))
    return weights


def _make_regressor(
    variant: ConditionedForceVariant,
    *,
    random_seed: int,
) -> Any:
    if variant.model_family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(variant.estimators),
            min_samples_leaf=int(variant.minimum_leaf_samples),
            max_features=variant.max_features,
            bootstrap=bool(variant.bootstrap),
            n_jobs=-1,
            random_state=int(random_seed),
        )
    if variant.model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(variant.estimators),
            min_samples_leaf=int(variant.minimum_leaf_samples),
            max_features=variant.max_features,
            bootstrap=True,
            n_jobs=-1,
            random_state=int(random_seed),
        )
    if variant.model_family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(variant.estimators),
            min_samples_leaf=max(2, int(variant.minimum_leaf_samples)),
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=int(random_seed),
        )
    raise ValueError(f"unsupported conditioned regressor: {variant.model_family}")


def grouped_conditioned_force_oof(
    arrays: FusionArrays,
    variant: ConditionedForceVariant,
    *,
    source_policy: Mapping[str, Mapping[str, Any]],
    group_position_votes: Mapping[str, str],
    gate_active_by_array_index: Mapping[int, bool],
    auxiliary_arrays: FusionArrays | None = None,
    auxiliary_positions: Sequence[str] = (),
    auxiliary_relative_weight: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Produce grouped OOF force predictions with no test-force calibration."""

    selected_features = feature_indices(arrays.feature_names, variant.feature_view)
    folds = sorted(
        set(
            arrays.fold_id[
                arrays.force_mask
                & arrays.formal_test_eligible
                & (arrays.fold_id >= 0)
            ].tolist()
        )
    )
    if not folds:
        raise ValueError("no formal grouped force folds found")
    true_conditions = infer_group_position_labels(
        arrays.group_id,
        arrays.position_target,
    )
    auxiliary_matrix: np.ndarray | None = None
    auxiliary_targets: np.ndarray | None = None
    auxiliary_groups: np.ndarray | None = None
    auxiliary_position_text = "none"
    if auxiliary_arrays is not None:
        if not np.array_equal(arrays.feature_names, auxiliary_arrays.feature_names):
            raise ValueError("current and auxiliary feature schemas differ")
        allowed_positions = tuple(str(value) for value in auxiliary_positions)
        invalid_positions = sorted(set(allowed_positions) - set(POSITION_ORDER))
        if invalid_positions:
            raise ValueError(
                "invalid auxiliary positions: " + ", ".join(invalid_positions)
            )
        if not allowed_positions:
            raise ValueError("auxiliary arrays require an explicit position filter")
        if float(auxiliary_relative_weight) <= 0.0:
            raise ValueError("auxiliary arrays require a positive relative weight")
        auxiliary_conditions = infer_group_position_labels(
            auxiliary_arrays.group_id,
            auxiliary_arrays.position_target,
        )
        auxiliary_mask = auxiliary_arrays.force_mask & np.isin(
            auxiliary_conditions,
            allowed_positions,
        )
        if not np.any(auxiliary_mask):
            raise ValueError("no force rows match the auxiliary position filter")
        auxiliary_matrix = append_position_condition(
            auxiliary_arrays.features[auxiliary_mask][:, selected_features],
            auxiliary_conditions[auxiliary_mask],
        )
        auxiliary_targets = auxiliary_arrays.force_fz_n[auxiliary_mask]
        auxiliary_groups = auxiliary_arrays.group_id[auxiliary_mask].astype(str)
        auxiliary_position_text = ",".join(allowed_positions)
    parts: list[pd.DataFrame] = []
    for fold in folds:
        test = (
            arrays.force_mask
            & arrays.formal_test_eligible
            & (arrays.fold_id == int(fold))
        )
        train = (
            arrays.force_mask
            & (
                (
                    arrays.formal_test_eligible
                    & (arrays.fold_id >= 0)
                    & (arrays.fold_id != int(fold))
                )
                | ~arrays.formal_test_eligible
            )
        )
        overlap = set(arrays.group_id[train]).intersection(arrays.group_id[test])
        if overlap:
            raise RuntimeError(
                "group leakage detected: " + ", ".join(sorted(overlap)[:5])
            )
        if auxiliary_groups is not None:
            auxiliary_overlap = set(auxiliary_groups).intersection(
                arrays.group_id[test]
            )
            if auxiliary_overlap:
                raise RuntimeError(
                    "historical auxiliary group leakage detected: "
                    + ", ".join(sorted(auxiliary_overlap)[:5])
                )
        train_indices = np.flatnonzero(train)
        test_indices = np.flatnonzero(test)
        test_conditions, condition_sources = test_position_conditions(
            arrays,
            test_indices,
            group_position_votes,
        )
        train_matrix = append_position_condition(
            arrays.features[train][:, selected_features],
            true_conditions[train],
        )
        test_matrix = append_position_condition(
            arrays.features[test][:, selected_features],
            test_conditions,
        )
        weights = source_group_weights(
            arrays.source_role,
            arrays.group_id,
            train,
            source_policy,
        )
        fit_matrix = train_matrix
        fit_targets = arrays.force_fz_n[train]
        fit_weights = weights[train]
        if auxiliary_matrix is not None:
            if auxiliary_targets is None or auxiliary_groups is None:
                raise RuntimeError("incomplete auxiliary force training state")
            auxiliary_weights = equal_group_auxiliary_weights(
                auxiliary_groups,
                total_mass=float(auxiliary_relative_weight)
                * float(np.sum(fit_weights)),
            )
            fit_matrix = np.vstack([fit_matrix, auxiliary_matrix])
            fit_targets = np.concatenate([fit_targets, auxiliary_targets])
            fit_weights = np.concatenate([fit_weights, auxiliary_weights])
        model = _make_regressor(
            variant,
            random_seed=int(random_seed) + int(fold),
        )
        model.fit(
            fit_matrix,
            fit_targets,
            sample_weight=fit_weights,
        )
        raw = np.clip(model.predict(test_matrix).astype(float), 0.0, 5.0)
        gate_active = np.asarray(
            [bool(gate_active_by_array_index.get(int(index), False)) for index in test_indices],
            dtype=bool,
        )
        gated = np.where(gate_active, raw, 0.0)
        parts.append(
            pd.DataFrame(
                {
                    "model_id": variant.model_id,
                    "fold_id": int(fold),
                    "array_index": test_indices,
                    "group_id": arrays.group_id[test],
                    "file_id": arrays.file_id[test],
                    "sample_index": arrays.sample_index[test],
                    "elapsed_time_sec": arrays.elapsed_time_sec[test],
                    "position_id": true_conditions[test],
                    "position_condition": test_conditions,
                    "position_condition_source": condition_sources,
                    "true_force_n": arrays.force_fz_n[test].astype(float),
                    "raw_force_n": raw,
                    "contact_gate_active": gate_active,
                    "gated_force_n": gated,
                    "force_sensor_used_as_runtime_input": False,
                    "historical_auxiliary_positions": auxiliary_position_text,
                    "historical_auxiliary_relative_weight": float(
                        auxiliary_relative_weight
                    ),
                    "evaluation_validity": "formal_grouped_oof_by_session_id",
                }
            )
        )
    predictions = pd.concat(parts, ignore_index=True)
    return predictions.sort_values(["group_id", "sample_index"]).reset_index(drop=True)
