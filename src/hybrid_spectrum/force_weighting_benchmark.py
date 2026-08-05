"""Grouped force-regression weighting audit for the ordinary-FBG dataset.

The benchmark deliberately leaves the deployed model untouched.  It reuses the
formal session folds and optical-only current-frame feature view so weighting
strategies can be compared without changing the evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .all_source_training import (
    FusionArrays,
    feature_indices,
    source_group_weights,
)
from .force_consistency_audit import infer_position_id


FORCE_WEIGHTING_STRATEGIES: tuple[str, ...] = (
    "baseline",
    "linear2",
    "linear4",
    "square4",
    "group_bin_half",
    "group_bin_full",
)

FORCE_BIN_EDGES_N = np.asarray(
    [0.0, 0.10, 0.50, 1.0, 2.0, 3.0, 4.0, 5.000001],
    dtype=float,
)


@dataclass(frozen=True)
class WeightingBenchmarkResult:
    """Predictions and summaries from one complete grouped benchmark."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    position_metrics: pd.DataFrame
    decision: dict[str, Any]


def force_weight_multiplier(
    strategy: str,
    force_n: np.ndarray,
    group_id: np.ndarray,
) -> np.ndarray:
    """Return a per-frame force weighting multiplier.

    Group/bin strategies equalize the force-range mass inside each independent
    session.  They do not mix sessions or alter the held-out fold assignment.
    """

    force = np.asarray(force_n, dtype=float)
    groups = np.asarray(group_id, dtype=str)
    if force.shape != groups.shape:
        raise ValueError("force and group arrays must have equal shape")
    if not np.all(np.isfinite(force)):
        raise ValueError("force weighting received a non-finite target")
    normalized = np.clip(force, 0.0, 5.0) / 5.0
    if strategy == "baseline":
        return np.ones_like(normalized)
    if strategy == "linear2":
        return 1.0 + normalized
    if strategy == "linear4":
        return 1.0 + 3.0 * normalized
    if strategy == "square4":
        return 1.0 + 3.0 * np.square(normalized)
    if strategy not in {"group_bin_half", "group_bin_full"}:
        raise ValueError(f"unknown force weighting strategy: {strategy}")

    multiplier = np.ones_like(normalized)
    bin_id = np.digitize(force, FORCE_BIN_EDGES_N[1:-1], right=False)
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        present_bins, counts = np.unique(bin_id[selected], return_counts=True)
        if not len(present_bins):
            continue
        count_by_bin = dict(zip(present_bins.tolist(), counts.tolist()))
        equal_bin = np.asarray(
            [
                len(selected)
                / (len(present_bins) * count_by_bin[int(bin_id[index])])
                for index in selected
            ],
            dtype=float,
        )
        if strategy == "group_bin_half":
            equal_bin = 0.5 + 0.5 * equal_bin
        multiplier[selected] = equal_bin
    return multiplier


def combine_training_weights(
    base_weight: np.ndarray,
    force_n: np.ndarray,
    group_id: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Combine source/session weights with the candidate force weighting."""

    base = np.asarray(base_weight, dtype=float)
    multiplier = force_weight_multiplier(strategy, force_n, group_id)
    if base.shape != multiplier.shape:
        raise ValueError("base and force weighting arrays must have equal shape")
    combined = base * multiplier
    mean = float(np.mean(combined))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("combined force weighting is invalid")
    return combined / mean


def _safe_pearson(true: np.ndarray, predicted: np.ndarray) -> float:
    if len(true) < 2 or np.std(true) <= 1.0e-12 or np.std(predicted) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(true, predicted)[0, 1])


def force_curve_metrics(
    true_force_n: np.ndarray,
    predicted_force_n: np.ndarray,
) -> dict[str, float | int]:
    """Return error, curve-shape, high-force, and residual diagnostics."""

    true = np.asarray(true_force_n, dtype=float)
    predicted = np.asarray(predicted_force_n, dtype=float)
    valid = np.isfinite(true) & np.isfinite(predicted)
    true = true[valid]
    predicted = predicted[valid]
    if not len(true):
        raise ValueError("force metric calculation has no paired samples")
    error = predicted - true
    slope = float("nan")
    intercept = float("nan")
    if len(true) >= 2 and np.std(true) > 1.0e-12:
        slope, intercept = np.polyfit(true, predicted, deg=1)
    true_amplitude = float(np.percentile(true, 95) - np.percentile(true, 5))
    predicted_amplitude = float(
        np.percentile(predicted, 95) - np.percentile(predicted, 5)
    )
    amplitude_ratio = (
        predicted_amplitude / true_amplitude
        if true_amplitude > 1.0e-12
        else float("nan")
    )
    high = true >= 3.0
    active = true >= 0.25
    zero = true <= 0.03
    return {
        "paired_frame_count": int(len(true)),
        "mae_n": float(mean_absolute_error(true, predicted)),
        "rmse_n": float(np.sqrt(mean_squared_error(true, predicted))),
        "r2": float(r2_score(true, predicted)),
        "bias_n": float(np.mean(error)),
        "pearson_r": _safe_pearson(true, predicted),
        "calibration_slope": float(slope),
        "calibration_intercept_n": float(intercept),
        "amplitude_ratio_p95_p05": float(amplitude_ratio),
        "active_force_mae_n": float(
            mean_absolute_error(true[active], predicted[active])
        )
        if np.any(active)
        else float("nan"),
        "high_force_frame_count": int(np.sum(high)),
        "high_force_mae_n": float(
            mean_absolute_error(true[high], predicted[high])
        )
        if np.any(high)
        else float("nan"),
        "high_force_bias_n": float(np.mean(error[high]))
        if np.any(high)
        else float("nan"),
        "zero_force_frame_count": int(np.sum(zero)),
        "zero_force_mean_prediction_n": float(np.mean(predicted[zero]))
        if np.any(zero)
        else float("nan"),
        "zero_force_false_response_rate": float(np.mean(predicted[zero] > 0.10))
        if np.any(zero)
        else float("nan"),
    }


def _training_masks(
    arrays: FusionArrays,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    test = (
        arrays.force_mask
        & arrays.formal_test_eligible
        & (arrays.fold_id == fold)
    )
    formal_train = (
        arrays.force_mask
        & arrays.formal_test_eligible
        & (arrays.fold_id >= 0)
        & (arrays.fold_id != fold)
    )
    auxiliary = arrays.force_mask & ~arrays.formal_test_eligible
    train = formal_train | auxiliary
    overlap = set(arrays.group_id[train]).intersection(arrays.group_id[test])
    if overlap:
        raise RuntimeError(
            "group leakage in force weighting benchmark: "
            + ", ".join(sorted(overlap)[:5])
        )
    if not np.any(train) or not np.any(test):
        raise ValueError(f"empty grouped force split for fold {fold}")
    return train, test


def _load_contact_gate(
    contact_gate_predictions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if contact_gate_predictions is None:
        return None
    required = {
        "fold_id",
        "group_id",
        "file_id",
        "sample_index",
        "contact_probability",
    }
    missing = sorted(required.difference(contact_gate_predictions.columns))
    if missing:
        raise ValueError("contact gate predictions missing: " + ", ".join(missing))
    keys = ["fold_id", "group_id", "file_id", "sample_index"]
    frame = contact_gate_predictions[list(required)].copy()
    if frame.duplicated(keys).any():
        raise ValueError("contact gate predictions contain duplicate sample keys")
    return frame


def _position_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (strategy, position), frame in predictions.groupby(
        ["weighting_strategy", "position_label"], sort=True
    ):
        if not position:
            continue
        raw = force_curve_metrics(
            frame["true_force_n"].to_numpy(dtype=float),
            frame["predicted_force_n"].to_numpy(dtype=float),
        )
        gated = force_curve_metrics(
            frame["true_force_n"].to_numpy(dtype=float),
            frame["gated_force_n"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "weighting_strategy": strategy,
                "position_label": position,
                "session_count": int(frame["group_id"].nunique()),
                "frame_count": int(len(frame)),
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"gated_{key}": value for key, value in gated.items()},
            }
        )
    return pd.DataFrame(rows)


def _session_position_labels(
    arrays: FusionArrays,
    selected: np.ndarray,
) -> np.ndarray:
    """Recover the pressed position for baseline frames inside a test session."""

    labels: list[str] = []
    for explicit, group_id, file_id in zip(
        arrays.position_target[selected],
        arrays.group_id[selected],
        arrays.file_id[selected],
    ):
        label = str(explicit)
        if not label:
            label = infer_position_id(group_id) or infer_position_id(file_id) or ""
        labels.append(label)
    return np.asarray(labels, dtype=str)


def _choose_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    indexed = metrics.set_index("weighting_strategy")
    baseline = indexed.loc["baseline"]
    eligible: list[tuple[str, float]] = []
    checks: dict[str, dict[str, bool]] = {}
    for strategy, row in indexed.iterrows():
        if strategy == "baseline":
            continue
        current_checks = {
            "global_gated_mae_guardrail": bool(
                row["gated_mae_n"] <= baseline["gated_mae_n"] * 1.01
            ),
            "global_r2_guardrail": bool(
                row["gated_r2"] >= baseline["gated_r2"] - 0.005
            ),
            "macro_position_guardrail": bool(
                row["macro_position_gated_mae_n"]
                <= baseline["macro_position_gated_mae_n"] * 1.01
            ),
            "zero_residual_guardrail": bool(
                row["gated_zero_force_false_response_rate"]
                <= baseline["gated_zero_force_false_response_rate"] + 0.01
            ),
            "p13_improves_5_percent": bool(
                row["p13_gated_mae_n"] <= baseline["p13_gated_mae_n"] * 0.95
            ),
            "high_force_improves_5_percent": bool(
                row["gated_high_force_mae_n"]
                <= baseline["gated_high_force_mae_n"] * 0.95
            ),
        }
        checks[str(strategy)] = current_checks
        if all(current_checks.values()):
            score = float(
                (baseline["p13_gated_mae_n"] - row["p13_gated_mae_n"])
                + (
                    baseline["gated_high_force_mae_n"]
                    - row["gated_high_force_mae_n"]
                )
                + 0.25
                * (baseline["gated_mae_n"] - row["gated_mae_n"])
            )
            eligible.append((str(strategy), score))
    if eligible:
        recommended = max(eligible, key=lambda item: item[1])[0]
        status = "candidate_passed_guardrails_not_deployed"
        reason = (
            "A weighting candidate improved P13 and high-force error while "
            "passing global, position, R2, and zero-residual guardrails."
        )
    else:
        recommended = "baseline"
        status = "keep_current_baseline"
        reason = (
            "No weighting candidate improved P13 and the high-force region "
            "without violating at least one formal guardrail."
        )
    return {
        "recommendation_status": status,
        "recommended_strategy": recommended,
        "deployed_model_changed": False,
        "reason": reason,
        "candidate_guardrail_checks": checks,
    }


def run_force_weighting_benchmark(
    arrays: FusionArrays,
    config: Mapping[str, Any],
    *,
    contact_gate_predictions: pd.DataFrame | None = None,
    strategies: Sequence[str] = FORCE_WEIGHTING_STRATEGIES,
) -> WeightingBenchmarkResult:
    """Train and evaluate each weighting strategy on identical formal folds."""

    unknown = sorted(set(strategies).difference(FORCE_WEIGHTING_STRATEGIES))
    if unknown:
        raise ValueError("unsupported weighting strategies: " + ", ".join(unknown))
    model_config = dict(config["models"])
    source_policy = dict(config["source_policy"])
    evaluation = dict(config["evaluation"])
    gate_config = dict(
        config.get("force_calibration", {}).get("optical_contact_gate", {})
    )
    probability_threshold = float(gate_config.get("probability_threshold", 0.75))
    no_contact_output = float(gate_config.get("no_contact_output_n", 0.0))
    estimators = int(model_config.get("tree_estimators", 180))
    minimum_leaf = int(model_config.get("minimum_leaf_samples", 2))
    random_seed = int(evaluation.get("random_seed", 42))
    feature_index = feature_indices(arrays.feature_names, "current_frame")
    folds = tuple(
        sorted(
            np.unique(
                arrays.fold_id[
                    arrays.force_mask
                    & arrays.formal_test_eligible
                    & (arrays.fold_id >= 0)
                ]
            ).tolist()
        )
    )
    if len(folds) != int(evaluation.get("folds", 5)):
        raise ValueError("formal fold count does not match the training config")
    contact_gate = _load_contact_gate(contact_gate_predictions)

    prediction_parts: list[pd.DataFrame] = []
    fit_seconds: dict[str, float] = {strategy: 0.0 for strategy in strategies}
    for strategy in strategies:
        for fold in folds:
            train, test = _training_masks(arrays, int(fold))
            base = source_group_weights(
                arrays.source_role,
                arrays.group_id,
                train,
                source_policy,
            )[train]
            weights = combine_training_weights(
                base,
                arrays.force_fz_n[train],
                arrays.group_id[train],
                strategy,
            )
            model = ExtraTreesRegressor(
                n_estimators=estimators,
                min_samples_leaf=minimum_leaf,
                max_features=0.8,
                n_jobs=-1,
                random_state=random_seed + int(fold),
            )
            started = perf_counter()
            model.fit(
                arrays.features[train][:, feature_index],
                arrays.force_fz_n[train],
                sample_weight=weights,
            )
            fit_seconds[strategy] += perf_counter() - started
            prediction = np.clip(
                model.predict(arrays.features[test][:, feature_index]),
                0.0,
                5.0,
            )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "weighting_strategy": strategy,
                        "fold_id": int(fold),
                        "group_id": arrays.group_id[test],
                        "file_id": arrays.file_id[test],
                        "sample_index": arrays.sample_index[test],
                        "elapsed_time_sec": arrays.elapsed_time_sec[test],
                        "position_label": _session_position_labels(arrays, test),
                        "true_force_n": arrays.force_fz_n[test],
                        "predicted_force_n": prediction,
                    }
                )
            )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    keys = ["fold_id", "group_id", "file_id", "sample_index"]
    if contact_gate is not None:
        predictions = predictions.merge(
            contact_gate,
            on=keys,
            how="left",
            validate="many_to_one",
        )
        if predictions["contact_probability"].isna().any():
            raise RuntimeError("contact gate does not cover every benchmark frame")
    else:
        predictions["contact_probability"] = 1.0
    predictions["gated_force_n"] = np.where(
        predictions["contact_probability"].to_numpy(dtype=float)
        >= probability_threshold,
        predictions["predicted_force_n"].to_numpy(dtype=float),
        no_contact_output,
    )

    position_metrics = _position_summary(predictions)
    metric_rows: list[dict[str, Any]] = []
    for strategy, frame in predictions.groupby("weighting_strategy", sort=False):
        raw = force_curve_metrics(
            frame["true_force_n"].to_numpy(dtype=float),
            frame["predicted_force_n"].to_numpy(dtype=float),
        )
        gated = force_curve_metrics(
            frame["true_force_n"].to_numpy(dtype=float),
            frame["gated_force_n"].to_numpy(dtype=float),
        )
        per_position = position_metrics[
            position_metrics["weighting_strategy"] == strategy
        ]
        p13 = per_position[per_position["position_label"] == "P13"]
        worst = per_position.sort_values("gated_mae_n", ascending=False).iloc[0]
        metric_rows.append(
            {
                "weighting_strategy": strategy,
                "training_time_sec": float(fit_seconds[strategy]),
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"gated_{key}": value for key, value in gated.items()},
                "macro_position_raw_mae_n": float(per_position["raw_mae_n"].mean()),
                "macro_position_gated_mae_n": float(
                    per_position["gated_mae_n"].mean()
                ),
                "worst_position": str(worst["position_label"]),
                "worst_position_gated_mae_n": float(worst["gated_mae_n"]),
                "p13_raw_mae_n": float(p13.iloc[0]["raw_mae_n"]),
                "p13_gated_mae_n": float(p13.iloc[0]["gated_mae_n"]),
                "p13_gated_calibration_slope": float(
                    p13.iloc[0]["gated_calibration_slope"]
                ),
                "p13_gated_amplitude_ratio_p95_p05": float(
                    p13.iloc[0]["gated_amplitude_ratio_p95_p05"]
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    decision = _choose_candidate(metrics)
    return WeightingBenchmarkResult(
        predictions=predictions,
        metrics=metrics,
        position_metrics=position_metrics,
        decision=decision,
    )
