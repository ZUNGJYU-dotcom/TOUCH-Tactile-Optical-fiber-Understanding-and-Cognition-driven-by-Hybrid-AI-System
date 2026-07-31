"""Grouped evaluation for the ordinary-FBG all-source optical dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - dependency is optional
    LGBMClassifier = None
    LGBMRegressor = None


CONTACT_CLASSES = ("no_contact", "contact")
POSITION_CLASSES = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)


@dataclass(frozen=True)
class FusionArrays:
    """In-memory representation of the immutable fusion NPZ."""

    features: np.ndarray
    feature_names: np.ndarray
    contact_target: np.ndarray
    position_target: np.ndarray
    force_fz_n: np.ndarray
    contact_mask: np.ndarray
    position_mask: np.ndarray
    force_mask: np.ndarray
    formal_test_eligible: np.ndarray
    fold_id: np.ndarray
    source_role: np.ndarray
    group_id: np.ndarray
    file_id: np.ndarray
    sample_index: np.ndarray
    elapsed_time_sec: np.ndarray


@dataclass(frozen=True)
class ModelVariant:
    """One source regime, optical feature view, and estimator family."""

    model_id: str
    source_regime: str
    feature_view: str
    model_family: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fusion_arrays(path: Path) -> FusionArrays:
    with np.load(path, allow_pickle=False) as payload:
        arrays = FusionArrays(
            features=payload["features"].astype(np.float32),
            feature_names=payload["feature_names"].astype(str),
            contact_target=payload["contact_target"].astype(np.int8),
            position_target=payload["position_target"].astype(str),
            force_fz_n=payload["force_fz_n"].astype(np.float32),
            contact_mask=payload["contact_mask"].astype(bool),
            position_mask=payload["position_mask"].astype(bool),
            force_mask=payload["force_mask"].astype(bool),
            formal_test_eligible=payload["formal_test_eligible"].astype(bool),
            fold_id=payload["fold_id"].astype(np.int16),
            source_role=payload["source_role"].astype(str),
            group_id=payload["group_id"].astype(str),
            file_id=payload["file_id"].astype(str),
            sample_index=payload["sample_index"].astype(np.int32),
            elapsed_time_sec=payload["elapsed_time_sec"].astype(np.float32),
        )
    if arrays.features.shape[1] != len(arrays.feature_names):
        raise ValueError("feature matrix and feature names are inconsistent")
    if not np.all(np.isfinite(arrays.features)):
        raise ValueError("fusion feature matrix contains non-finite values")
    if np.any(
        arrays.force_mask
        & (
            ~np.isfinite(arrays.force_fz_n)
            | (arrays.force_fz_n < 0.0)
            | (arrays.force_fz_n > 5.0)
        )
    ):
        raise ValueError("force supervision mask contains a value outside 0-5 N")
    return arrays


def feature_indices(
    names: np.ndarray,
    view: str,
) -> np.ndarray:
    if view == "temporal_fusion":
        return np.arange(len(names), dtype=np.int32)
    if view == "current_frame":
        selected = np.asarray(
            [index for index, name in enumerate(names) if name.startswith("last__")],
            dtype=np.int32,
        )
        if len(selected) != 40:
            raise ValueError(
                f"current-frame view expected 40 optical features, got {len(selected)}"
            )
        return selected
    raise ValueError(f"unknown optical feature view: {view}")


def default_variants() -> tuple[ModelVariant, ...]:
    variants = [
        ModelVariant(
            model_id="latest_primary_temporal_extra_trees",
            source_regime="latest_primary_only",
            feature_view="temporal_fusion",
            model_family="extra_trees",
        ),
        ModelVariant(
            model_id="all_sources_current_extra_trees",
            source_regime="all_sources",
            feature_view="current_frame",
            model_family="extra_trees",
        ),
        ModelVariant(
            model_id="all_sources_temporal_random_forest",
            source_regime="all_sources",
            feature_view="temporal_fusion",
            model_family="random_forest",
        ),
        ModelVariant(
            model_id="all_sources_temporal_extra_trees",
            source_regime="all_sources",
            feature_view="temporal_fusion",
            model_family="extra_trees",
        ),
    ]
    if LGBMClassifier is not None and LGBMRegressor is not None:
        variants.append(
            ModelVariant(
                model_id="all_sources_temporal_lightgbm",
                source_regime="all_sources",
                feature_view="temporal_fusion",
                model_family="lightgbm",
            )
        )
    return tuple(variants)


def _classification_model(
    family: str,
    *,
    estimators: int,
    minimum_leaf_samples: int,
    random_seed: int,
) -> Any:
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_seed,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        )
    if family == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=max(120, estimators),
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=max(10, minimum_leaf_samples * 5),
            subsample=0.9,
            colsample_bytree=0.85,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
            random_state=random_seed,
        )
    raise ValueError(f"unsupported classifier family: {family}")


def _regression_model(
    family: str,
    *,
    estimators: int,
    minimum_leaf_samples: int,
    random_seed: int,
) -> Any:
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features=0.7,
            n_jobs=-1,
            random_state=random_seed,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=minimum_leaf_samples,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_seed,
        )
    if family == "lightgbm" and LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=max(120, estimators),
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=max(10, minimum_leaf_samples * 5),
            subsample=0.9,
            colsample_bytree=0.85,
            n_jobs=-1,
            verbosity=-1,
            random_state=random_seed,
        )
    raise ValueError(f"unsupported regressor family: {family}")


def source_group_weights(
    source_role: np.ndarray,
    group_id: np.ndarray,
    selected: np.ndarray,
    source_policy: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    """Give each source a configured mass and each group equal mass within it."""

    indices = np.flatnonzero(selected)
    weights = np.zeros(len(source_role), dtype=np.float64)
    for role in sorted(set(source_role[indices].tolist())):
        role_indices = indices[source_role[indices] == role]
        role_groups = group_id[role_indices]
        unique_groups = sorted(set(role_groups.tolist()))
        configured_mass = float(source_policy.get(role, {}).get("source_weight", 1.0))
        for group in unique_groups:
            group_indices = role_indices[role_groups == group]
            weights[group_indices] = configured_mass / (
                max(1, len(unique_groups)) * max(1, len(group_indices))
            )
    nonzero = weights[selected]
    if not len(nonzero) or not np.all(nonzero > 0.0):
        raise ValueError("source/group weighting produced an invalid training weight")
    weights[selected] /= float(np.mean(nonzero))
    return weights


def _task_targets(
    arrays: FusionArrays,
    task: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...] | None]:
    if task == "contact":
        labels = np.where(arrays.contact_target == 1, "contact", "no_contact")
        return labels, arrays.contact_mask, CONTACT_CLASSES
    if task == "position":
        return arrays.position_target, arrays.position_mask, POSITION_CLASSES
    if task == "force_fz":
        return arrays.force_fz_n, arrays.force_mask, None
    raise ValueError(f"unknown training task: {task}")


def _split_masks(
    arrays: FusionArrays,
    task_mask: np.ndarray,
    fold: int,
    source_regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    test = task_mask & arrays.formal_test_eligible & (arrays.fold_id == fold)
    formal_train = (
        task_mask
        & arrays.formal_test_eligible
        & (arrays.fold_id >= 0)
        & (arrays.fold_id != fold)
    )
    if source_regime == "latest_primary_only":
        train = formal_train
    elif source_regime == "all_sources":
        auxiliary = task_mask & ~arrays.formal_test_eligible
        train = formal_train | auxiliary
    else:
        raise ValueError(f"unknown source regime: {source_regime}")
    overlap = set(arrays.group_id[train]).intersection(arrays.group_id[test])
    if overlap:
        raise RuntimeError(
            "group leakage detected between training and test: "
            + ", ".join(sorted(overlap)[:5])
        )
    if not np.any(train) or not np.any(test):
        raise ValueError(f"empty grouped split for fold {fold}")
    return train, test


def _classification_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    classes: tuple[str, ...],
    groups: np.ndarray,
) -> dict[str, Any]:
    report = classification_report(
        true,
        predicted,
        labels=list(classes),
        output_dict=True,
        zero_division=0,
    )
    group_true: list[str] = []
    group_predicted: list[str] = []
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_true.append(Counter(true[selected].tolist()).most_common(1)[0][0])
        group_predicted.append(
            Counter(predicted[selected].tolist()).most_common(1)[0][0]
        )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(
                true,
                predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        "group_voting_accuracy": float(
            accuracy_score(group_true, group_predicted)
        ),
        "group_voting_macro_f1": float(
            f1_score(
                group_true,
                group_predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            true, predicted, labels=list(classes)
        ).tolist(),
        "classes": list(classes),
        "test_sample_count": int(len(true)),
        "test_group_count": int(len(set(groups.tolist()))),
    }


def _regression_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    group_true: list[float] = []
    group_predicted: list[float] = []
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_true.append(float(np.median(true[selected])))
        group_predicted.append(float(np.median(predicted[selected])))
    return {
        "mae_n": float(mean_absolute_error(true, predicted)),
        "rmse_n": float(np.sqrt(mean_squared_error(true, predicted))),
        "r2": float(r2_score(true, predicted)),
        "group_median_mae_n": float(
            mean_absolute_error(group_true, group_predicted)
        ),
        "test_sample_count": int(len(true)),
        "test_group_count": int(len(set(groups.tolist()))),
        "true_min_n": float(np.min(true)),
        "true_max_n": float(np.max(true)),
        "predicted_min_n": float(np.min(predicted)),
        "predicted_max_n": float(np.max(predicted)),
    }


def _aggregate_predictions(
    predictions: pd.DataFrame,
    task: str,
    classes: tuple[str, ...] | None,
) -> dict[str, Any]:
    if task == "force_fz":
        return _regression_metrics(
            predictions["true_value"].to_numpy(dtype=float),
            predictions["predicted_value"].to_numpy(dtype=float),
            predictions["group_id"].to_numpy(dtype=str),
        )
    if classes is None:
        raise ValueError("classification classes are required")
    return _classification_metrics(
        predictions["true_label"].to_numpy(dtype=str),
        predictions["predicted_label"].to_numpy(dtype=str),
        classes,
        predictions["group_id"].to_numpy(dtype=str),
    )


def apply_optical_contact_gate(
    raw_force_n: np.ndarray,
    contact_probability: np.ndarray,
    *,
    probability_threshold: float,
    no_contact_output_n: float = 0.0,
) -> np.ndarray:
    """Suppress residual optical force when temporal contact evidence is weak."""

    raw = np.asarray(raw_force_n, dtype=float)
    probability = np.asarray(contact_probability, dtype=float)
    if raw.shape != probability.shape:
        raise ValueError("force and contact probability arrays must have equal shape")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("contact probability threshold must be within 0-1")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(probability)):
        raise ValueError("contact gate inputs must be finite")
    return np.where(
        probability >= probability_threshold,
        np.clip(raw, 0.0, 5.0),
        float(no_contact_output_n),
    )


def evaluate_force_contact_gate(
    arrays: FusionArrays,
    config: Mapping[str, Any],
    predictions: pd.DataFrame,
    selected_model_ids: Mapping[str, str],
    variants: Iterable[ModelVariant],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the deployable optical-only contact gate on every Fz OOF frame."""

    gate_config = dict(
        config.get("force_calibration", {}).get("optical_contact_gate", {})
    )
    if not bool(gate_config.get("enabled", True)):
        raise ValueError("optical contact gate must be enabled for force calibration")
    threshold = float(gate_config.get("probability_threshold", 0.75))
    no_contact_output = float(gate_config.get("no_contact_output_n", 0.0))
    source_policy = dict(config["source_policy"])
    model_config = dict(config["models"])
    evaluation = dict(config["evaluation"])
    random_seed = int(evaluation.get("random_seed", 42))
    estimators = int(model_config.get("tree_estimators", 180))
    minimum_leaf = int(model_config.get("minimum_leaf_samples", 2))
    by_id = {variant.model_id: variant for variant in variants}
    contact_variant = by_id[selected_model_ids["contact"]]
    contact_features = feature_indices(
        arrays.feature_names, contact_variant.feature_view
    )
    contact_targets, contact_mask, _ = _task_targets(arrays, "contact")
    force_model_id = selected_model_ids["force_fz"]
    force_variant = by_id[force_model_id]
    force_oof = predictions[
        (predictions["task"] == "force_fz")
        & (predictions["model_id"] == force_model_id)
    ].copy()
    merge_keys = ["fold_id", "group_id", "file_id", "sample_index"]
    if force_oof.duplicated(merge_keys).any():
        raise ValueError("selected force OOF predictions contain duplicate samples")
    force_oof = force_oof[
        merge_keys
        + [
            "elapsed_time_sec",
            "true_value",
            "raw_predicted_value",
            "predicted_value",
        ]
    ].rename(
        columns={
            "true_value": "true_force_n",
            "raw_predicted_value": "unclipped_raw_force_n",
            "predicted_value": "raw_optical_force_n",
        }
    )

    gate_parts: list[pd.DataFrame] = []
    folds = tuple(
        sorted(
            set(
                arrays.fold_id[
                    arrays.force_mask
                    & arrays.formal_test_eligible
                    & (arrays.fold_id >= 0)
                ].tolist()
            )
        )
    )
    for fold in folds:
        contact_train, _ = _split_masks(
            arrays,
            contact_mask,
            int(fold),
            contact_variant.source_regime,
        )
        force_test = (
            arrays.force_mask
            & arrays.formal_test_eligible
            & (arrays.fold_id == int(fold))
        )
        overlap = set(arrays.group_id[contact_train]).intersection(
            arrays.group_id[force_test]
        )
        if overlap:
            raise RuntimeError(
                "contact gate group leakage detected: "
                + ", ".join(sorted(overlap)[:5])
            )
        weights = source_group_weights(
            arrays.source_role,
            arrays.group_id,
            contact_train,
            source_policy,
        )
        model = _classification_model(
            contact_variant.model_family,
            estimators=estimators,
            minimum_leaf_samples=minimum_leaf,
            random_seed=random_seed + int(fold),
        )
        model.fit(
            arrays.features[contact_train][:, contact_features],
            contact_targets[contact_train],
            sample_weight=weights[contact_train],
        )
        probability = np.asarray(
            model.predict_proba(arrays.features[force_test][:, contact_features]),
            dtype=float,
        )
        class_names = np.asarray(model.classes_, dtype=str)
        matching = np.flatnonzero(class_names == "contact")
        if len(matching) != 1:
            raise ValueError("contact classifier does not expose a contact probability")
        contact_probability = probability[:, int(matching[0])]
        array_indices = np.flatnonzero(force_test)
        gate_parts.append(
            pd.DataFrame(
                {
                    "fold_id": int(fold),
                    "group_id": arrays.group_id[force_test],
                    "file_id": arrays.file_id[force_test],
                    "sample_index": arrays.sample_index[force_test],
                    "contact_probability": contact_probability,
                    "contact_gate_active": contact_probability >= threshold,
                    "contact_label_available": arrays.contact_mask[force_test],
                    "true_contact_label": np.where(
                        arrays.contact_mask[force_test],
                        np.where(
                            arrays.contact_target[force_test] == 1,
                            "contact",
                            "no_contact",
                        ),
                        "",
                    ),
                    "array_index": array_indices,
                }
            )
        )
    contact_oof = pd.concat(gate_parts, ignore_index=True)
    gate_predictions = force_oof.merge(
        contact_oof,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(gate_predictions) != len(force_oof):
        raise RuntimeError(
            "contact gate did not cover every selected force OOF prediction"
        )
    gate_predictions["gated_force_n"] = apply_optical_contact_gate(
        gate_predictions["raw_optical_force_n"].to_numpy(dtype=float),
        gate_predictions["contact_probability"].to_numpy(dtype=float),
        probability_threshold=threshold,
        no_contact_output_n=no_contact_output,
    )
    true_force = gate_predictions["true_force_n"].to_numpy(dtype=float)
    raw_force = gate_predictions["raw_optical_force_n"].to_numpy(dtype=float)
    gated_force = gate_predictions["gated_force_n"].to_numpy(dtype=float)
    groups = gate_predictions["group_id"].to_numpy(dtype=str)
    no_contact_force = float(config["labels"].get("no_contact_max_force_n", 0.03))
    residual_limit = float(config["labels"].get("contact_min_force_n", 0.10))
    active_force = float(config["labels"].get("position_min_force_n", 0.25))
    zero_mask = true_force <= no_contact_force
    active_mask = true_force >= active_force
    labeled_mask = gate_predictions["contact_label_available"].to_numpy(dtype=bool)
    predicted_contact_label = np.where(
        gate_predictions["contact_probability"].to_numpy(dtype=float) >= threshold,
        "contact",
        "no_contact",
    )
    labeled_contact_accuracy = (
        float(
            accuracy_score(
                gate_predictions.loc[labeled_mask, "true_contact_label"],
                predicted_contact_label[labeled_mask],
            )
        )
        if np.any(labeled_mask)
        else np.nan
    )
    metrics = {
        "schema_version": "ordinary_fbg_optical_contact_gate_oof_v1",
        "evaluation_validity": "formal_grouped_by_session_id",
        "force_model_id": force_model_id,
        "contact_model_id": selected_model_ids["contact"],
        "force_feature_view": force_variant.feature_view,
        "contact_feature_view": contact_variant.feature_view,
        "probability_threshold": threshold,
        "no_contact_output_n": no_contact_output,
        "test_sample_count": int(len(gate_predictions)),
        "test_group_count": int(len(set(groups.tolist()))),
        "raw_force_metrics": _regression_metrics(true_force, raw_force, groups),
        "gated_force_metrics": _regression_metrics(true_force, gated_force, groups),
        "zero_force_definition_n_lte": no_contact_force,
        "residual_force_warning_n_gt": residual_limit,
        "zero_force_frame_count": int(np.sum(zero_mask)),
        "raw_zero_force_false_response_rate": float(
            np.mean(raw_force[zero_mask] > residual_limit)
        )
        if np.any(zero_mask)
        else np.nan,
        "gated_zero_force_false_response_rate": float(
            np.mean(gated_force[zero_mask] > residual_limit)
        )
        if np.any(zero_mask)
        else np.nan,
        "active_force_definition_n_gte": active_force,
        "active_force_frame_count": int(np.sum(active_mask)),
        "active_force_suppressed_to_zero_rate": float(
            np.mean(gated_force[active_mask] <= no_contact_output + 1e-9)
        )
        if np.any(active_mask)
        else np.nan,
        "gate_active_rate": float(
            np.mean(gate_predictions["contact_gate_active"].to_numpy(dtype=bool))
        ),
        "contact_labeled_force_frame_count": int(np.sum(labeled_mask)),
        "contact_gate_accuracy_on_labeled_force_frames": labeled_contact_accuracy,
        "force_sensor_used_as_model_input": False,
        "force_sensor_required_at_inference": False,
        "runtime_inputs": ["optical_spectrum_time_series"],
    }
    return gate_predictions, metrics


def grouped_cross_validation(
    arrays: FusionArrays,
    variants: Iterable[ModelVariant],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_config = dict(config["models"])
    evaluation = dict(config["evaluation"])
    source_policy = dict(config["source_policy"])
    folds = tuple(
        sorted(
            set(
                arrays.fold_id[
                    arrays.formal_test_eligible & (arrays.fold_id >= 0)
                ].tolist()
            )
        )
    )
    expected_folds = int(evaluation.get("folds", 5))
    if len(folds) != expected_folds:
        raise ValueError(f"expected {expected_folds} formal folds, found {folds}")
    random_seed = int(evaluation.get("random_seed", 42))
    estimators = int(model_config.get("tree_estimators", 180))
    minimum_leaf = int(model_config.get("minimum_leaf_samples", 2))

    prediction_parts: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    task_metrics: dict[str, Any] = {}
    for task in ("contact", "position", "force_fz"):
        targets, task_mask, classes = _task_targets(arrays, task)
        for variant in variants:
            selected_features = feature_indices(
                arrays.feature_names, variant.feature_view
            )
            started = perf_counter()
            variant_parts: list[pd.DataFrame] = []
            for fold in folds:
                train, test = _split_masks(
                    arrays, task_mask, int(fold), variant.source_regime
                )
                weights = source_group_weights(
                    arrays.source_role,
                    arrays.group_id,
                    train,
                    source_policy,
                )
                if task == "force_fz":
                    model = _regression_model(
                        variant.model_family,
                        estimators=estimators,
                        minimum_leaf_samples=minimum_leaf,
                        random_seed=random_seed + int(fold),
                    )
                else:
                    model = _classification_model(
                        variant.model_family,
                        estimators=estimators,
                        minimum_leaf_samples=minimum_leaf,
                        random_seed=random_seed + int(fold),
                    )
                model.fit(
                    arrays.features[train][:, selected_features],
                    targets[train],
                    sample_weight=weights[train],
                )
                raw_prediction = model.predict(
                    arrays.features[test][:, selected_features]
                )
                base = pd.DataFrame(
                    {
                        "model_id": variant.model_id,
                        "task": task,
                        "fold_id": int(fold),
                        "source_regime": variant.source_regime,
                        "feature_view": variant.feature_view,
                        "model_family": variant.model_family,
                        "group_id": arrays.group_id[test],
                        "file_id": arrays.file_id[test],
                        "sample_index": arrays.sample_index[test],
                        "elapsed_time_sec": arrays.elapsed_time_sec[test],
                    }
                )
                if task == "force_fz":
                    base["true_value"] = targets[test].astype(float)
                    base["raw_predicted_value"] = raw_prediction.astype(float)
                    base["predicted_value"] = np.clip(
                        raw_prediction.astype(float), 0.0, 5.0
                    )
                    base["true_label"] = ""
                    base["predicted_label"] = ""
                    base["confidence"] = np.nan
                else:
                    base["true_value"] = np.nan
                    base["raw_predicted_value"] = np.nan
                    base["predicted_value"] = np.nan
                    base["true_label"] = targets[test].astype(str)
                    base["predicted_label"] = raw_prediction.astype(str)
                    if hasattr(model, "predict_proba"):
                        probability = np.asarray(
                            model.predict_proba(
                                arrays.features[test][:, selected_features]
                            ),
                            dtype=float,
                        )
                        base["confidence"] = np.max(probability, axis=1)
                    else:
                        base["confidence"] = np.nan
                variant_parts.append(base)
                prediction_parts.append(base)
                fit_records.append(
                    {
                        "task": task,
                        "model_id": variant.model_id,
                        "fold_id": int(fold),
                        "train_sample_count": int(np.sum(train)),
                        "test_sample_count": int(np.sum(test)),
                        "train_group_count": int(
                            len(set(arrays.group_id[train].tolist()))
                        ),
                        "test_group_count": int(
                            len(set(arrays.group_id[test].tolist()))
                        ),
                        "group_overlap_count": 0,
                    }
                )
            combined = pd.concat(variant_parts, ignore_index=True)
            key = f"{task}::{variant.model_id}"
            metrics = _aggregate_predictions(combined, task, classes)
            metrics.update(
                {
                    "model_id": variant.model_id,
                    "task": task,
                    "source_regime": variant.source_regime,
                    "feature_view": variant.feature_view,
                    "feature_count": int(len(selected_features)),
                    "model_family": variant.model_family,
                    "cross_validation_fit_time_sec": float(
                        perf_counter() - started
                    ),
                    "split_strategy": "grouped_by_session_id",
                    "evaluation_validity": "formal_grouped_evaluation",
                    "force_sensor_used_as_input": False,
                }
            )
            task_metrics[key] = metrics
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return predictions, pd.DataFrame(fit_records), task_metrics


def leaderboard_from_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in metrics.values():
        report = value.get("classification_report", {})
        rows.append(
            {
                "task": value["task"],
                "model_id": value["model_id"],
                "model_family": value["model_family"],
                "source_regime": value["source_regime"],
                "feature_view": value["feature_view"],
                "feature_count": value["feature_count"],
                "split_strategy": value["split_strategy"],
                "evaluation_validity": value["evaluation_validity"],
                "accuracy": value.get("accuracy", np.nan),
                "macro_f1": value.get("macro_f1", np.nan),
                "group_voting_accuracy": value.get(
                    "group_voting_accuracy", np.nan
                ),
                "group_voting_macro_f1": value.get(
                    "group_voting_macro_f1", np.nan
                ),
                "contact_recall": report.get("contact", {}).get(
                    "recall", np.nan
                ),
                "mae_n": value.get("mae_n", np.nan),
                "rmse_n": value.get("rmse_n", np.nan),
                "r2": value.get("r2", np.nan),
                "group_median_mae_n": value.get(
                    "group_median_mae_n", np.nan
                ),
                "test_sample_count": value["test_sample_count"],
                "test_group_count": value["test_group_count"],
                "cross_validation_fit_time_sec": value[
                    "cross_validation_fit_time_sec"
                ],
                "force_sensor_used_as_input": False,
            }
        )
    frame = pd.DataFrame(rows)
    classification = frame[frame["task"] != "force_fz"].sort_values(
        ["task", "macro_f1", "accuracy"],
        ascending=[True, False, False],
    )
    regression = frame[frame["task"] == "force_fz"].sort_values(
        ["mae_n", "rmse_n"],
        ascending=[True, True],
    )
    return pd.concat([classification, regression], ignore_index=True)


def select_candidate_models(
    leaderboard: pd.DataFrame,
) -> dict[str, str]:
    temporal_eligible = leaderboard[
        (leaderboard["source_regime"] == "all_sources")
        & (leaderboard["feature_view"] == "temporal_fusion")
    ]
    selected: dict[str, str] = {}
    for task in ("contact", "position"):
        rows = temporal_eligible[temporal_eligible["task"] == task].sort_values(
            ["macro_f1", "accuracy"],
            ascending=[False, False],
        )
        selected[task] = str(rows.iloc[0]["model_id"])
    force_rows = leaderboard[
        (leaderboard["source_regime"] == "all_sources")
        & (leaderboard["task"] == "force_fz")
    ].sort_values(
        ["mae_n", "rmse_n"],
        ascending=[True, True],
    )
    selected["force_fz"] = str(force_rows.iloc[0]["model_id"])
    return selected


def fit_candidate_bundle(
    arrays: FusionArrays,
    selected_model_ids: Mapping[str, str],
    variants: Iterable[ModelVariant],
    config: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    force_gate_metrics: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    by_id = {variant.model_id: variant for variant in variants}
    model_config = dict(config["models"])
    source_policy = dict(config["source_policy"])
    random_seed = int(config["evaluation"].get("random_seed", 42))
    estimators = int(model_config.get("final_tree_estimators", 300))
    minimum_leaf = int(model_config.get("minimum_leaf_samples", 2))
    trained: dict[str, Any] = {}
    task_metadata: dict[str, Any] = {}
    for task, model_id in selected_model_ids.items():
        variant = by_id[model_id]
        targets, task_mask, classes = _task_targets(arrays, task)
        selected_features = feature_indices(
            arrays.feature_names, variant.feature_view
        )
        weights = source_group_weights(
            arrays.source_role,
            arrays.group_id,
            task_mask,
            source_policy,
        )
        if task == "force_fz":
            model = _regression_model(
                variant.model_family,
                estimators=estimators,
                minimum_leaf_samples=minimum_leaf,
                random_seed=random_seed,
            )
        else:
            model = _classification_model(
                variant.model_family,
                estimators=estimators,
                minimum_leaf_samples=minimum_leaf,
                random_seed=random_seed,
            )
        started = perf_counter()
        model.fit(
            arrays.features[task_mask][:, selected_features],
            targets[task_mask],
            sample_weight=weights[task_mask],
        )
        fit_time = perf_counter() - started
        latency_indices = np.flatnonzero(task_mask)[: min(2000, np.sum(task_mask))]
        latency_matrix = arrays.features[latency_indices][:, selected_features]
        model.predict(latency_matrix[: min(10, len(latency_matrix))])
        latency_started = perf_counter()
        model.predict(latency_matrix)
        latency_ms = (
            (perf_counter() - latency_started)
            * 1000.0
            / max(1, len(latency_matrix))
        )
        trained[task] = {
            "model": model,
            "feature_indices": selected_features,
            "feature_names": arrays.feature_names[selected_features],
            "classes": np.asarray(classes or (), dtype=str),
        }
        metric_key = f"{task}::{model_id}"
        task_metadata[task] = {
            "model_id": model_id,
            "model_family": variant.model_family,
            "feature_view": variant.feature_view,
            "training_sample_count": int(np.sum(task_mask)),
            "training_group_count": int(
                len(set(arrays.group_id[task_mask].tolist()))
            ),
            "fit_time_sec": float(fit_time),
            "inference_latency_ms_per_sample": float(latency_ms),
            "grouped_cv_metrics": metrics[metric_key],
        }
    bundle = {
        "schema_version": "ordinary_fbg_optical_only_force_candidate_v2",
        "created_for": "candidate_evaluation_only_not_deployed",
        "feature_schema": "nine_peak_shift_intensity_shape_temporal_summary_483",
        "all_feature_names": arrays.feature_names,
        "tasks": trained,
        "task_metadata": task_metadata,
        "force_calibration_contract": {
            "supervision_target": "Fz",
            "unit": "N",
            "training_range_n": [0.0, 5.0],
            "force_sensor_is_runtime_input": False,
            "force_sensor_required_at_inference": False,
            "runtime_inputs": ["optical_spectrum_time_series"],
            "prediction_output": "estimated_compression_fz_n",
            "prediction_clip_range_n": [0.0, 5.0],
            "optical_contact_gate": {
                "enabled": True,
                "contact_task": "contact",
                "force_task": "force_fz",
                "probability_threshold": float(
                    force_gate_metrics["probability_threshold"]
                ),
                "no_contact_output_n": float(
                    force_gate_metrics["no_contact_output_n"]
                ),
                "semantics": (
                    "temporal optical contact probability gates the "
                    "current-frame optical Fz regression"
                ),
                "grouped_oof_metrics": dict(force_gate_metrics),
            },
        },
        "runtime_inference_policy": {
            "step_1": "compute temporal optical contact probability",
            "step_2": (
                "return 0 N when contact probability is below the configured "
                "threshold"
            ),
            "step_3": (
                "otherwise return the clipped current-frame optical Fz estimate"
            ),
            "force_sensor_is_runtime_input": False,
        },
        "data_contract": {
            "formal_evaluation": "latest_primary_grouped_by_session_id",
            "auxiliary_sources": [
                "latest_challenge_contact_and_force_only",
                "legacy_dynamic_contact_and_position",
                "legacy_static_manual_contact_and_position",
                "legacy_static_gauge_low_force_anchor",
                "legacy_static_no_contact",
            ],
            "blind_test_used": False,
            "subjective_manual_force_labels_used_as_newton": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    return {
        "candidate_bundle_path": str(output_path),
        "candidate_bundle_size_mb": float(
            output_path.stat().st_size / (1024.0 * 1024.0)
        ),
        "selected_models": dict(selected_model_ids),
        "task_metadata": task_metadata,
    }


def save_figures(
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    force_gate_predictions: pd.DataFrame,
    output_dir: Path,
    selected: Mapping[str, str],
) -> tuple[Path, ...]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, task, metric, title in (
        (axes[0], "contact", "macro_f1", "Contact macro-F1"),
        (axes[1], "position", "macro_f1", "Position macro-F1"),
        (axes[2], "force_fz", "mae_n", "Fz MAE (N, lower is better)"),
    ):
        rows = leaderboard[leaderboard["task"] == task].copy()
        rows = rows.sort_values(metric, ascending=task == "force_fz")
        labels = [
            value.replace("all_sources_", "all_").replace(
                "latest_primary_", "latest_"
            )
            for value in rows["model_id"]
        ]
        values = rows[metric].to_numpy(dtype=float)
        colors = [
            "#1594c4" if model_id == selected[task] else "#9fc9d8"
            for model_id in rows["model_id"]
        ]
        axis.barh(labels, values, color=colors)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    fig.tight_layout()
    comparison_path = figure_dir / "all_data_model_comparison.png"
    fig.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(comparison_path)

    force = predictions[
        (predictions["task"] == "force_fz")
        & (predictions["model_id"] == selected["force_fz"])
    ].copy()
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        force["true_value"],
        force["predicted_value"],
        s=8,
        alpha=0.28,
        color="#1594c4",
        edgecolors="none",
    )
    axis.plot([0, 5], [0, 5], color="#e07a5f", linewidth=1.8)
    axis.set(
        xlabel="PX6D Fz supervision (N)",
        ylabel="Optical-only estimated Fz (N)",
        title="Grouped out-of-fold force calibration",
        xlim=(0, 5),
        ylim=(0, 5),
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    force_path = figure_dir / "optical_force_calibration_oof.png"
    fig.savefig(force_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(force_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for axis, column, title in (
        (axes[0], "raw_optical_force_n", "Raw optical Fz estimate"),
        (axes[1], "gated_force_n", "Contact-gated optical Fz estimate"),
    ):
        axis.scatter(
            force_gate_predictions["true_force_n"],
            force_gate_predictions[column],
            s=8,
            alpha=0.25,
            color="#1594c4",
            edgecolors="none",
        )
        axis.plot([0, 5], [0, 5], color="#e07a5f", linewidth=1.6)
        axis.set(
            xlabel="PX6D Fz calibration target (N)",
            ylabel="Optical-only estimated Fz (N)",
            title=title,
            xlim=(0, 5),
            ylim=(0, 5),
        )
        axis.grid(alpha=0.2)
    fig.suptitle("Formal grouped OOF optical-to-force calibration")
    fig.tight_layout()
    gated_force_path = figure_dir / "optical_force_contact_gate_oof.png"
    fig.savefig(gated_force_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(gated_force_path)
    return tuple(paths)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_training_outputs(
    *,
    output_dir: Path,
    arrays: FusionArrays,
    config: Mapping[str, Any],
    dataset_path: Path,
    protected_model_path: Path,
    protected_hash_before: str,
    predictions: pd.DataFrame,
    split_audit: pd.DataFrame,
    metrics: Mapping[str, Mapping[str, Any]],
    leaderboard: pd.DataFrame,
    force_gate_predictions: pd.DataFrame,
    force_gate_metrics: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "all_data_model_leaderboard.csv"
    metrics_path = output_dir / "all_data_model_metrics.json"
    predictions_path = output_dir / "grouped_oof_predictions.csv"
    force_gate_predictions_path = (
        output_dir / "force_contact_gate_oof_predictions.csv"
    )
    force_gate_metrics_path = output_dir / "force_contact_gate_metrics.json"
    split_path = output_dir / "grouped_split_audit.csv"
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    force_gate_predictions.to_csv(
        force_gate_predictions_path, index=False, encoding="utf-8-sig"
    )
    split_audit.to_csv(split_path, index=False, encoding="utf-8-sig")
    _write_json(metrics_path, dict(metrics))
    _write_json(force_gate_metrics_path, dict(force_gate_metrics))

    source_counts = Counter(arrays.source_role.tolist())
    task_counts = {
        "contact": int(np.sum(arrays.contact_mask)),
        "position": int(np.sum(arrays.position_mask)),
        "force_fz_0_to_5_n": int(np.sum(arrays.force_mask)),
    }
    inventory_path = output_dir / "data_usage_inventory.md"
    inventory_lines = [
        "# All-source data usage inventory",
        "",
        "The force sensor is used only to provide synchronized Fz supervision.",
        "No force or torque channel is present in the model input feature matrix.",
        "",
        "## Source usage",
        "",
    ]
    source_notes = {
        "latest_primary": "formal grouped test plus training; contact, position, Fz",
        "latest_challenge": "auxiliary training only; contact and Fz, no position",
        "legacy_dynamic": "auxiliary training; contact and position, no Fz",
        "legacy_static_manual": "auxiliary training; contact and position, no Fz",
        "legacy_static_gauge": "auxiliary low-force Fz anchor plus contact/position",
        "legacy_static_no_contact": "auxiliary contact baseline only",
    }
    for role, count in sorted(source_counts.items()):
        inventory_lines.append(
            f"- `{role}`: {count} samples; {source_notes.get(role, '')}"
        )
    inventory_lines.extend(
        [
            "- `blind_audit`: 10 files inventoried, 0 training samples; untouched",
            "",
            "## Task coverage",
            "",
            f"- Contact labels: {task_counts['contact']}",
            f"- Position labels: {task_counts['position']}",
            f"- Valid Fz labels in 0-5 N: {task_counts['force_fz_0_to_5_n']}",
            "",
            "Missing labels are represented by task masks. They are never filled with "
            "fabricated zeros. Subjective light/normal/hard labels are never converted "
            "to newtons.",
        ]
    )
    inventory_path.write_text("\n".join(inventory_lines) + "\n", encoding="utf-8")

    report_path = output_dir / "all_data_training_report.md"
    report_lines = [
        "# Ordinary-FBG all-data optical calibration report",
        "",
        "## Decision",
        "",
        "This run uses the complete valid data inventory without mixing label "
        "semantics. Optical spectra and their time evolution are the only model "
        "inputs. PX6D Fz is calibration supervision only. The resulting force "
        "candidate can run without the force sensor after deployment.",
        "",
        "## Evaluation",
        "",
        "All formal results are grouped by synchronized acquisition session. "
        "The formal test set contains only the latest primary sessions. Earlier "
        "or suspected-label sessions are auxiliary training data and never enter "
        "the formal test folds.",
        "",
        "## Selected candidates",
        "",
    ]
    for task, model_id in selected.items():
        row = leaderboard[
            (leaderboard["task"] == task)
            & (leaderboard["model_id"] == model_id)
        ].iloc[0]
        if task == "force_fz":
            report_lines.append(
                f"- Fz calibration: `{model_id}`; MAE {row['mae_n']:.3f} N, "
                f"RMSE {row['rmse_n']:.3f} N, R2 {row['r2']:.3f}."
            )
        else:
            report_lines.append(
                f"- {task}: `{model_id}`; accuracy {row['accuracy']:.3f}, "
                f"macro-F1 {row['macro_f1']:.3f}."
            )
    report_lines.extend(
        [
            "",
            "## Optical-only force inference policy",
            "",
            "PX6D Fz is used only as the synchronized calibration target during "
            "training. At runtime the contact head first evaluates temporal "
            "optical evidence. A nonzero continuous Fz estimate is emitted only "
            f"when optical contact probability is at least "
            f"{force_gate_metrics['probability_threshold']:.2f}. Otherwise the "
            "output is 0 N. The force sensor is not connected to the inference "
            "pipeline.",
            "",
            f"- Raw optical Fz OOF MAE: "
            f"{force_gate_metrics['raw_force_metrics']['mae_n']:.3f} N.",
            f"- Contact-gated optical Fz OOF MAE: "
            f"{force_gate_metrics['gated_force_metrics']['mae_n']:.3f} N.",
            f"- Raw zero-force residual rate: "
            f"{force_gate_metrics['raw_zero_force_false_response_rate']:.1%}.",
            f"- Gated zero-force residual rate: "
            f"{force_gate_metrics['gated_zero_force_false_response_rate']:.1%}.",
            f"- Active-force frames suppressed to zero: "
            f"{force_gate_metrics['active_force_suppressed_to_zero_rate']:.1%}.",
            "",
            "## Runtime contract",
            "",
            "- Runtime input: optical spectrum time series only.",
            "- Calibration target: PX6D compression Fz, 0-5 N.",
            "- Force sensor required during inference: no.",
            "- Fx, Fy, Mx, My, Mz: not used.",
            "- Candidate status: evaluation only; the deployed model was not replaced.",
            "",
            "The trained output estimates the calibration reference represented by "
            "Fz. It is not yet a traceable metrology claim beyond the tested setup, "
            "mounting, sensor specimen, and 0-5 N range.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    figures = save_figures(
        leaderboard,
        predictions,
        force_gate_predictions,
        output_dir,
        selected,
    )
    protected_hash_after = (
        sha256_file(protected_model_path)
        if protected_model_path.is_file()
        else ""
    )
    if protected_hash_before != protected_hash_after:
        raise RuntimeError("protected deployed model changed during candidate training")
    summary = {
        "schema_version": "ordinary_fbg_all_data_training_summary_v1",
        "dataset_path": str(dataset_path),
        "sample_count": int(len(arrays.features)),
        "feature_count": int(arrays.features.shape[1]),
        "feature_input_contains_force_channel": False,
        "force_supervision": "PX6D Fz only",
        "force_training_range_n": [0.0, 5.0],
        "force_sensor_required_at_inference": False,
        "force_inference_policy": {
            "contact_gate": "temporal optical contact probability",
            "contact_probability_threshold": float(
                force_gate_metrics["probability_threshold"]
            ),
            "active_force_estimator": "current-frame optical spectrum",
            "no_contact_output_n": float(
                force_gate_metrics["no_contact_output_n"]
            ),
            "continuous_output_range_n": [0.0, 5.0],
        },
        "force_contact_gate_oof_metrics": dict(force_gate_metrics),
        "formal_split_strategy": "grouped_by_session_id",
        "random_frame_split_used": False,
        "blind_test_used": False,
        "selected_models": dict(selected),
        "candidate_bundle": dict(candidate_summary),
        "protected_deployed_model_path": str(protected_model_path),
        "protected_model_sha256_before": protected_hash_before,
        "protected_model_sha256_after": protected_hash_after,
        "protected_model_unchanged": protected_hash_before == protected_hash_after,
        "artifacts": {
            "leaderboard": str(leaderboard_path),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "force_contact_gate_predictions": str(force_gate_predictions_path),
            "force_contact_gate_metrics": str(force_gate_metrics_path),
            "split_audit": str(split_path),
            "inventory": str(inventory_path),
            "report": str(report_path),
            "figures": [str(path) for path in figures],
        },
        "ok": True,
    }
    summary_path = output_dir / "training_summary.json"
    _write_json(summary_path, summary)
    return summary
