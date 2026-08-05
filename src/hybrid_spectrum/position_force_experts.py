"""Leakage-safe position-expert force regression and fold-local calibration.

Each ordinary-FBG point can have a different optical sensitivity.  A shared
regressor with a position one-hot feature still shares tree partitions across
all points, so this module provides a true mixture of nine position experts.
The held-out expert is selected from optical position evidence.  Optional
force-scale calibration is fitted only from inner out-of-fold predictions of
the outer training folds; a held-out session never calibrates itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .all_source_training import FusionArrays, feature_indices, source_group_weights
from .force_consistency_audit import POSITION_ORDER
from .position_conditioned_force import (
    UNKNOWN_POSITION,
    infer_group_position_labels,
    test_position_conditions,
)


@dataclass(frozen=True)
class PositionExpertVariant:
    """Optical feature and tree settings for one nine-point expert bank."""

    model_id: str = "position_expert_current_extra_trees"
    feature_view: str = "current_frame"
    estimators: int = 140
    minimum_leaf_samples: int = 2
    max_features: float | str = 0.8


@dataclass(frozen=True)
class CalibrationSpec:
    """One fold-local force-scale correction candidate."""

    suffix: str
    mode: str = "affine"
    ridge_strength: float = 0.02
    blend: float = 0.5


@dataclass(frozen=True)
class AffineForceCalibrator:
    """A bounded affine mapping from raw optical force to PX6D Fz."""

    slope: float
    intercept_n: float
    mode: str
    ridge_strength: float
    training_sample_count: int
    training_group_count: int

    def predict(self, values: Sequence[float]) -> np.ndarray:
        raw = np.asarray(values, dtype=float)
        return np.clip(self.slope * raw + self.intercept_n, 0.0, 5.0)


def group_equal_sample_weights(group_ids: Sequence[Any]) -> np.ndarray:
    """Return weights for which every acquisition session has equal mass."""

    groups = np.asarray(group_ids, dtype=str)
    if len(groups) == 0:
        raise ValueError("group ids cannot be empty")
    weights = np.zeros(len(groups), dtype=float)
    unique = sorted(set(groups.tolist()))
    for group_id in unique:
        selected = groups == group_id
        weights[selected] = 1.0 / (len(unique) * int(np.sum(selected)))
    return weights


def fit_affine_force_calibrator(
    raw_force_n: Sequence[float],
    true_force_n: Sequence[float],
    group_ids: Sequence[Any],
    *,
    mode: str = "affine",
    ridge_strength: float = 0.02,
    minimum_force_n: float = 0.10,
    minimum_groups: int = 3,
) -> AffineForceCalibrator:
    """Fit a bounded group-balanced calibrator around the identity mapping."""

    raw = np.asarray(raw_force_n, dtype=float)
    target = np.asarray(true_force_n, dtype=float)
    groups = np.asarray(group_ids, dtype=str)
    if not (len(raw) == len(target) == len(groups)):
        raise ValueError("calibrator inputs must have equal lengths")
    valid = (
        np.isfinite(raw)
        & np.isfinite(target)
        & (target >= float(minimum_force_n))
        & (target <= 5.0)
    )
    raw = raw[valid]
    target = target[valid]
    groups = groups[valid]
    unique_groups = sorted(set(groups.tolist()))
    if len(raw) < 20 or len(unique_groups) < int(minimum_groups):
        return AffineForceCalibrator(
            slope=1.0,
            intercept_n=0.0,
            mode="identity_insufficient_training_data",
            ridge_strength=float(ridge_strength),
            training_sample_count=int(len(raw)),
            training_group_count=int(len(unique_groups)),
        )

    weights = group_equal_sample_weights(groups)
    ridge = max(0.0, float(ridge_strength))
    if mode == "identity":
        slope, intercept = 1.0, 0.0
    elif mode == "zero_anchor":
        numerator = float(np.sum(weights * raw * target)) + ridge
        denominator = float(np.sum(weights * raw * raw)) + ridge
        slope = numerator / max(denominator, 1.0e-12)
        intercept = 0.0
    elif mode == "affine":
        design = np.column_stack([raw, np.ones(len(raw), dtype=float)])
        weighted_design = design * weights[:, None]
        normal = design.T @ weighted_design
        right = design.T @ (weights * target)
        # Shrink towards y=x.  The intercept receives a slightly stronger
        # prior because contact gating already owns the physical zero state.
        penalty = np.diag([ridge, ridge * 4.0])
        prior = np.asarray([1.0, 0.0], dtype=float)
        solution = np.linalg.solve(normal + penalty, right + penalty @ prior)
        slope, intercept = float(solution[0]), float(solution[1])
    else:
        raise ValueError(f"unsupported calibration mode: {mode}")

    return AffineForceCalibrator(
        slope=float(np.clip(slope, 0.60, 1.80)),
        intercept_n=float(np.clip(intercept, -0.75, 0.75)),
        mode=str(mode),
        ridge_strength=ridge,
        training_sample_count=int(len(raw)),
        training_group_count=int(len(unique_groups)),
    )


def _make_expert(variant: PositionExpertVariant, *, random_seed: int) -> Any:
    return ExtraTreesRegressor(
        n_estimators=int(variant.estimators),
        min_samples_leaf=int(variant.minimum_leaf_samples),
        max_features=variant.max_features,
        n_jobs=-1,
        random_state=int(random_seed),
    )


def _fit_expert_bank(
    arrays: FusionArrays,
    train_mask: np.ndarray,
    positions: np.ndarray,
    selected_features: np.ndarray,
    source_policy: Mapping[str, Mapping[str, Any]],
    variant: PositionExpertVariant,
    *,
    random_seed: int,
) -> dict[str, Any]:
    weights = source_group_weights(
        arrays.source_role,
        arrays.group_id,
        train_mask,
        source_policy,
    )
    bank: dict[str, Any] = {}
    for offset, position_id in enumerate(POSITION_ORDER):
        selected = train_mask & (positions == position_id)
        if len(set(arrays.group_id[selected].tolist())) < 2:
            continue
        model = _make_expert(variant, random_seed=random_seed + offset)
        model.fit(
            arrays.features[selected][:, selected_features],
            arrays.force_fz_n[selected],
            sample_weight=weights[selected],
        )
        bank[position_id] = model

    # Explicit fallback for a missing/uncertain optical position vote.
    global_model = _make_expert(
        variant,
        random_seed=random_seed + len(POSITION_ORDER) + 1,
    )
    global_model.fit(
        arrays.features[train_mask][:, selected_features],
        arrays.force_fz_n[train_mask],
        sample_weight=weights[train_mask],
    )
    bank[UNKNOWN_POSITION] = global_model
    return bank


def _predict_expert_bank(
    bank: Mapping[str, Any],
    optical_features: np.ndarray,
    conditions: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(optical_features, dtype=np.float32)
    labels = np.asarray(conditions, dtype=str)
    if len(matrix) != len(labels):
        raise ValueError("expert features and conditions must have equal rows")
    prediction = np.zeros(len(matrix), dtype=float)
    used = np.full(len(matrix), UNKNOWN_POSITION, dtype=object)
    fallback = bank.get(UNKNOWN_POSITION)
    if fallback is None:
        raise ValueError("expert bank has no global fallback")
    for position_id in sorted(set(labels.tolist())):
        selected = labels == position_id
        expert = bank.get(position_id, fallback)
        prediction[selected] = expert.predict(matrix[selected])
        used[selected] = position_id if position_id in bank else UNKNOWN_POSITION
    return np.clip(prediction, 0.0, 5.0), used.astype(str)


def _training_mask(arrays: FusionArrays, excluded_folds: set[int]) -> np.ndarray:
    formal = arrays.formal_test_eligible & (arrays.fold_id >= 0)
    return arrays.force_mask & (
        (~formal) | (formal & ~np.isin(arrays.fold_id, sorted(excluded_folds)))
    )


def _formal_fold_mask(arrays: FusionArrays, fold_id: int) -> np.ndarray:
    return (
        arrays.force_mask
        & arrays.formal_test_eligible
        & (arrays.fold_id == int(fold_id))
    )


def nested_grouped_position_expert_oof(
    arrays: FusionArrays,
    variant: PositionExpertVariant,
    calibration_specs: Sequence[CalibrationSpec],
    *,
    source_policy: Mapping[str, Mapping[str, Any]],
    group_position_votes: Mapping[str, str],
    gate_active_by_array_index: Mapping[int, bool],
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return outer-fold OOF force estimates plus fold-local calibrator audit."""

    selected_features = feature_indices(arrays.feature_names, variant.feature_view)
    formal = arrays.force_mask & arrays.formal_test_eligible & (arrays.fold_id >= 0)
    folds = sorted(set(arrays.fold_id[formal].astype(int).tolist()))
    if len(folds) < 3:
        raise ValueError("nested force calibration requires at least three folds")
    true_positions = infer_group_position_labels(
        arrays.group_id,
        arrays.position_target,
    )
    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    raw_spec = CalibrationSpec(suffix="raw", mode="identity", blend=0.0)
    specs = (raw_spec, *tuple(calibration_specs))
    for outer_fold in folds:
        outer_test = _formal_fold_mask(arrays, outer_fold)
        outer_train = _training_mask(arrays, {outer_fold})
        outer_test_indices = np.flatnonzero(outer_test)
        overlap = set(arrays.group_id[outer_train]).intersection(
            arrays.group_id[outer_test]
        )
        if overlap:
            raise RuntimeError("outer group leakage detected")

        outer_bank = _fit_expert_bank(
            arrays,
            outer_train,
            true_positions,
            selected_features,
            source_policy,
            variant,
            random_seed=int(random_seed) + 1000 * int(outer_fold),
        )
        outer_conditions, condition_sources = test_position_conditions(
            arrays,
            outer_test_indices,
            group_position_votes,
        )
        outer_raw, expert_used = _predict_expert_bank(
            outer_bank,
            arrays.features[outer_test][:, selected_features],
            outer_conditions,
        )

        inner_parts: list[pd.DataFrame] = []
        for inner_fold in folds:
            if inner_fold == outer_fold:
                continue
            inner_test = _formal_fold_mask(arrays, inner_fold)
            inner_train = _training_mask(arrays, {outer_fold, inner_fold})
            inner_test_indices = np.flatnonzero(inner_test)
            inner_overlap = set(arrays.group_id[inner_train]).intersection(
                arrays.group_id[inner_test]
            )
            if inner_overlap or set(arrays.group_id[inner_train]).intersection(
                arrays.group_id[outer_test]
            ):
                raise RuntimeError("inner or outer group leakage detected")
            inner_bank = _fit_expert_bank(
                arrays,
                inner_train,
                true_positions,
                selected_features,
                source_policy,
                variant,
                random_seed=(
                    int(random_seed)
                    + 1000 * int(outer_fold)
                    + 50 * int(inner_fold)
                ),
            )
            # The calibrator is organized by the known training position.  At
            # inference the matching optical position vote selects it.
            inner_raw, _ = _predict_expert_bank(
                inner_bank,
                arrays.features[inner_test][:, selected_features],
                true_positions[inner_test],
            )
            inner_parts.append(
                pd.DataFrame(
                    {
                        "array_index": inner_test_indices,
                        "inner_fold_id": int(inner_fold),
                        "group_id": arrays.group_id[inner_test],
                        "position_id": true_positions[inner_test],
                        "true_force_n": arrays.force_fz_n[inner_test].astype(float),
                        "raw_force_n": inner_raw,
                    }
                )
            )
            split_rows.append(
                {
                    "outer_fold_id": int(outer_fold),
                    "inner_fold_id": int(inner_fold),
                    "train_group_count": len(set(arrays.group_id[inner_train].tolist())),
                    "validation_group_count": len(set(arrays.group_id[inner_test].tolist())),
                    "outer_test_group_count": len(set(arrays.group_id[outer_test].tolist())),
                    "train_validation_overlap_count": 0,
                    "train_outer_test_overlap_count": 0,
                    "validation_outer_test_overlap_count": len(
                        set(arrays.group_id[inner_test]).intersection(
                            arrays.group_id[outer_test]
                        )
                    ),
                }
            )
        inner_oof = pd.concat(inner_parts, ignore_index=True)

        gate_active = np.asarray(
            [
                bool(gate_active_by_array_index.get(int(index), False))
                for index in outer_test_indices
            ],
            dtype=bool,
        )
        base = {
            "fold_id": int(outer_fold),
            "array_index": outer_test_indices,
            "group_id": arrays.group_id[outer_test],
            "file_id": arrays.file_id[outer_test],
            "sample_index": arrays.sample_index[outer_test],
            "elapsed_time_sec": arrays.elapsed_time_sec[outer_test],
            "position_id": true_positions[outer_test],
            "position_condition": outer_conditions,
            "position_condition_source": condition_sources,
            "expert_used": expert_used,
            "true_force_n": arrays.force_fz_n[outer_test].astype(float),
            "base_raw_force_n": outer_raw,
            "contact_gate_active": gate_active,
            "evaluation_validity": "formal_nested_grouped_oof_by_session_id",
            "force_sensor_used_as_runtime_input": False,
        }

        for spec in specs:
            calibrators: dict[str, AffineForceCalibrator] = {}
            for position_id in (*POSITION_ORDER, UNKNOWN_POSITION):
                selected = (
                    np.ones(len(inner_oof), dtype=bool)
                    if position_id == UNKNOWN_POSITION
                    else inner_oof["position_id"].astype(str).to_numpy() == position_id
                )
                calibrator = fit_affine_force_calibrator(
                    inner_oof.loc[selected, "raw_force_n"],
                    inner_oof.loc[selected, "true_force_n"],
                    inner_oof.loc[selected, "group_id"],
                    mode=spec.mode,
                    ridge_strength=spec.ridge_strength,
                )
                calibrators[position_id] = calibrator
                parameter_rows.append(
                    {
                        "model_id": f"{variant.model_id}_{spec.suffix}",
                        "outer_fold_id": int(outer_fold),
                        "calibration_position": position_id,
                        "mode": calibrator.mode,
                        "blend": float(spec.blend),
                        "ridge_strength": calibrator.ridge_strength,
                        "calibration_slope": calibrator.slope,
                        "calibration_intercept_n": calibrator.intercept_n,
                        "training_sample_count": calibrator.training_sample_count,
                        "training_group_count": calibrator.training_group_count,
                    }
                )

            calibrated = np.zeros(len(outer_raw), dtype=float)
            for position_id in sorted(set(outer_conditions.tolist())):
                selected = outer_conditions == position_id
                calibrator = calibrators.get(
                    position_id,
                    calibrators[UNKNOWN_POSITION],
                )
                mapped = calibrator.predict(outer_raw[selected])
                calibrated[selected] = (
                    outer_raw[selected]
                    + float(spec.blend) * (mapped - outer_raw[selected])
                )
            calibrated = np.clip(calibrated, 0.0, 5.0)
            gated = np.where(gate_active, calibrated, 0.0)
            part = pd.DataFrame(base)
            part["model_id"] = f"{variant.model_id}_{spec.suffix}"
            part["calibrated_force_n"] = calibrated
            part["gated_force_n"] = gated
            prediction_parts.append(part)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = predictions.sort_values(
        ["model_id", "group_id", "sample_index"]
    ).reset_index(drop=True)
    parameters = pd.DataFrame(parameter_rows)
    split_audit = pd.DataFrame(split_rows)
    return predictions, parameters, split_audit


__all__ = [
    "AffineForceCalibrator",
    "CalibrationSpec",
    "PositionExpertVariant",
    "fit_affine_force_calibrator",
    "group_equal_sample_weights",
    "nested_grouped_position_expert_oof",
]
