"""Compare nonlinear optical force regressors and contact gates for P13.

The benchmark is grouped by acquisition session.  It is model-discovery
evidence only and does not deploy or overwrite the current force model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    AlignedOpticalDataset,
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.spectral_force_experts import (  # noqa: E402
    build_causal_spectral_views,
)

try:  # Optional discovery dependency; the production path remains sklearn-only.
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - exercised only on minimal environments
    LGBMClassifier = None
    LGBMRegressor = None


DEFAULT_FUSION = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_all_data_fusion_20260804_initial_fixed5"
    / "all_source_fusion_dataset.npz"
)
DEFAULT_SPECTRUM = (
    PROJECT_ROOT
    / "outputs/ordinary_fbg_px6d_strict_20260803_new_data_only"
    / "ordinary_fbg_px6d_dataset.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/ordinary_fbg_p13_nonlinear_force_gate_20260804"


@dataclass(frozen=True)
class Candidate:
    model_id: str
    view_id: str
    factory: Callable[[], BaseEstimator]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--position", default="P13")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _regression_candidates() -> list[Candidate]:
    candidates = [
        Candidate(
            "extra_trees_leaf2_f05",
            "current264",
            lambda: ExtraTreesRegressor(
                n_estimators=350,
                min_samples_leaf=2,
                max_features=0.5,
                n_jobs=-1,
                random_state=411,
            ),
        ),
        Candidate(
            "extra_trees_leaf5_f10",
            "current264",
            lambda: ExtraTreesRegressor(
                n_estimators=350,
                min_samples_leaf=5,
                max_features=1.0,
                n_jobs=-1,
                random_state=412,
            ),
        ),
        Candidate(
            "extra_trees_leaf3_delta",
            "current_plus_delta1_3_8",
            lambda: ExtraTreesRegressor(
                n_estimators=350,
                min_samples_leaf=3,
                max_features=0.45,
                n_jobs=-1,
                random_state=413,
            ),
        ),
        Candidate(
            "random_forest_leaf3",
            "current264",
            lambda: RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                max_features=0.7,
                n_jobs=-1,
                random_state=421,
            ),
        ),
        Candidate(
            "hist_gradient_leaf12",
            "current264",
            lambda: HistGradientBoostingRegressor(
                learning_rate=0.045,
                max_iter=320,
                max_leaf_nodes=15,
                min_samples_leaf=12,
                l2_regularization=1.0,
                random_state=431,
            ),
        ),
        Candidate(
            "hist_gradient_leaf20_delta",
            "current_plus_delta1_3_8",
            lambda: HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=320,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.5,
                random_state=432,
            ),
        ),
    ]
    if LGBMRegressor is not None:
        candidates.extend(
            [
                Candidate(
                    "lightgbm_leaves15",
                    "current264",
                    lambda: LGBMRegressor(
                        n_estimators=420,
                        learning_rate=0.025,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_lambda=2.0,
                        random_state=441,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
                Candidate(
                    "lightgbm_leaves15_delta",
                    "current_plus_delta1_3_8",
                    lambda: LGBMRegressor(
                        n_estimators=420,
                        learning_rate=0.025,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.60,
                        reg_lambda=2.0,
                        random_state=442,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    return candidates


def _gate_candidates() -> list[Candidate]:
    candidates = [
        Candidate(
            "gate_extra_trees_leaf2",
            "current264",
            lambda: ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                max_features=0.6,
                class_weight="balanced",
                n_jobs=-1,
                random_state=451,
            ),
        ),
        Candidate(
            "gate_extra_trees_leaf5_delta",
            "current_plus_delta1_3_8",
            lambda: ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=5,
                max_features=0.45,
                class_weight="balanced",
                n_jobs=-1,
                random_state=452,
            ),
        ),
        Candidate(
            "gate_hist_gradient",
            "current264",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.045,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=461,
            ),
        ),
    ]
    if LGBMClassifier is not None:
        candidates.append(
            Candidate(
                "gate_lightgbm_delta",
                "current_plus_delta1_3_8",
                lambda: LGBMClassifier(
                    n_estimators=350,
                    learning_rate=0.025,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.85,
                    colsample_bytree=0.60,
                    reg_lambda=2.0,
                    class_weight="balanced",
                    random_state=462,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )
        )
    return candidates


def _position_masks(dataset: AlignedOpticalDataset, position_id: str) -> tuple[np.ndarray, np.ndarray]:
    formal = dataset.fold_id >= 0
    position = dataset.position_target.astype(str) == str(position_id)
    return formal, position


def _grouped_oof_regression(
    dataset: AlignedOpticalDataset,
    features: np.ndarray,
    candidate: Candidate,
    position_id: str,
) -> pd.DataFrame:
    formal, position = _position_masks(dataset, position_id)
    force_eligible = formal & dataset.force_mask & np.isfinite(dataset.force_fz_n)
    active = (
        force_eligible
        & position
        & (dataset.force_fz_n >= 0.10)
        & (dataset.force_fz_n <= 5.0)
    )
    parts: list[pd.DataFrame] = []
    for fold in sorted(set(dataset.fold_id[active].astype(int).tolist())):
        train = active & (dataset.fold_id != int(fold))
        test = (
            formal
            & position
            & (dataset.force_mask | dataset.contact_mask)
            & np.isfinite(dataset.force_fz_n)
            & (dataset.fold_id == int(fold))
        )
        if set(dataset.group_id[train]).intersection(set(dataset.group_id[test])):
            raise RuntimeError("regression train/test group overlap")
        model = candidate.factory()
        model.fit(features[train], dataset.force_fz_n[train])
        prediction = np.clip(np.asarray(model.predict(features[test])).reshape(-1), 0.0, 5.0)
        parts.append(
            pd.DataFrame(
                {
                    "group_id": dataset.group_id[test].astype(str),
                    "sample_index": dataset.sample_index[test].astype(int),
                    "fold_id": int(fold),
                    "true_force_n": dataset.force_fz_n[test].astype(float),
                    "contact_target": dataset.contact_target[test].astype(int),
                    "raw_force_n": prediction,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _grouped_oof_gate(
    dataset: AlignedOpticalDataset,
    features: np.ndarray,
    candidate: Candidate,
    position_id: str,
) -> pd.DataFrame:
    formal, _ = _position_masks(dataset, position_id)
    eligible = formal & dataset.contact_mask
    target = dataset.contact_target.astype(int)
    parts: list[pd.DataFrame] = []
    for fold in sorted(set(dataset.fold_id[eligible].astype(int).tolist())):
        train = eligible & (dataset.fold_id != int(fold))
        # Fit only on frames with a formal contact label, but predict every
        # formal frame in the held-out sessions. Some synchronized force rows
        # intentionally lack a contact label; they still need an optical-only
        # gate at evaluation and runtime.
        test = formal & (dataset.fold_id == int(fold))
        if set(dataset.group_id[train]).intersection(set(dataset.group_id[test])):
            raise RuntimeError("gate train/test group overlap")
        classes = np.unique(target[train].astype(int))
        if len(classes) < 2:
            probability = np.full(int(np.sum(test)), float(classes[0]), dtype=float)
        else:
            model = candidate.factory()
            model.fit(features[train], target[train].astype(int))
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(features[test]))
                model_classes = np.asarray(model.classes_, dtype=int)
                positive = np.flatnonzero(model_classes == 1)
                probability = (
                    probabilities[:, int(positive[0])]
                    if len(positive)
                    else np.zeros(int(np.sum(test)), dtype=float)
                )
            else:
                probability = np.asarray(model.predict(features[test]), dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "group_id": dataset.group_id[test].astype(str),
                    "sample_index": dataset.sample_index[test].astype(int),
                    "fold_id": int(fold),
                    "true_force_n": dataset.force_fz_n[test].astype(float),
                    "contact_target": target[test].astype(int),
                    "contact_label_available": dataset.contact_mask[test].astype(bool),
                    "position_id": dataset.position_target[test].astype(str),
                    "contact_probability": probability,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _safe_curve_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    correlation = None
    slope = None
    r2 = None
    if len(reference) >= 3 and np.std(reference) > 1.0e-12:
        slope = float(np.polyfit(reference, estimate, 1)[0])
        r2 = float(r2_score(reference, estimate))
        if np.std(estimate) > 1.0e-12:
            correlation = float(np.corrcoef(reference, estimate)[0, 1])
    return {
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "r2": r2,
        "pearson_r": correlation,
        "slope": slope,
    }


def _regression_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    active = predictions["true_force_n"] >= 0.10
    metrics = _safe_curve_metrics(
        predictions.loc[active, "true_force_n"], predictions.loc[active, "raw_force_n"]
    )
    session_rows = []
    for group_id, group in predictions.groupby("group_id"):
        selected = group["true_force_n"] >= 0.10
        if int(selected.sum()) < 3:
            continue
        session_rows.append(
            {"group_id": group_id, **_safe_curve_metrics(
                group.loc[selected, "true_force_n"], group.loc[selected, "raw_force_n"]
            )}
        )
    sessions = pd.DataFrame(session_rows)
    trial008 = sessions[sessions["group_id"].str.contains("trial_008", regex=False)]
    metrics.update(
        {
            "worst_session_mae_n": float(sessions["mae_n"].max()),
            "trial008_mae_n": float(trial008.iloc[0]["mae_n"]) if not trial008.empty else None,
            "trial008_r2": float(trial008.iloc[0]["r2"]) if not trial008.empty else None,
            "trial008_pearson_r": float(trial008.iloc[0]["pearson_r"]) if not trial008.empty else None,
            "trial008_slope": float(trial008.iloc[0]["slope"]) if not trial008.empty else None,
        }
    )
    return metrics


def _gate_summary(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    active = predictions["contact_target"].to_numpy(dtype=int) == 1
    zero = predictions["contact_target"].to_numpy(dtype=int) == 0
    detected = predictions["contact_probability"].to_numpy(dtype=float) >= float(threshold)
    trial008 = predictions["group_id"].str.contains("trial_008", regex=False).to_numpy()
    return {
        "threshold": float(threshold),
        "active_recall": float(np.mean(detected[active])) if np.any(active) else None,
        "zero_false_positive_rate": float(np.mean(detected[zero])) if np.any(zero) else None,
        "trial008_active_recall": (
            float(np.mean(detected[trial008 & active])) if np.any(trial008 & active) else None
        ),
        "trial008_zero_false_positive_rate": (
            float(np.mean(detected[trial008 & zero])) if np.any(trial008 & zero) else None
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    views = build_causal_spectral_views(dataset)

    regression_rows: list[dict[str, Any]] = []
    regression_predictions: dict[str, pd.DataFrame] = {}
    for candidate in _regression_candidates():
        predictions = _grouped_oof_regression(
            dataset, views[candidate.view_id], candidate, str(args.position)
        )
        regression_predictions[candidate.model_id] = predictions
        regression_rows.append(
            {
                "model_id": candidate.model_id,
                "view_id": candidate.view_id,
                "feature_count": int(views[candidate.view_id].shape[1]),
                "evaluation_validity": "grouped_oof_by_session_model_discovery",
                **_regression_summary(predictions),
            }
        )
    regression = pd.DataFrame(regression_rows)
    regression["selection_score"] = (
        regression["mae_n"]
        + 0.35 * regression["worst_session_mae_n"]
        + 0.20 * (1.0 - regression["pearson_r"].clip(0.0, 1.0))
        + 0.20 * (regression["slope"] - 1.0).abs()
    )
    regression = regression.sort_values("selection_score", kind="stable")

    gate_rows: list[dict[str, Any]] = []
    gate_predictions: dict[str, pd.DataFrame] = {}
    for candidate in _gate_candidates():
        predictions = _grouped_oof_gate(
            dataset, views[candidate.view_id], candidate, str(args.position)
        )
        gate_predictions[candidate.model_id] = predictions
        for threshold in (0.35, 0.45, 0.55, 0.65):
            gate_rows.append(
                {
                    "model_id": candidate.model_id,
                    "view_id": candidate.view_id,
                    "evaluation_validity": "grouped_oof_by_session_model_discovery",
                    **_gate_summary(predictions, threshold),
                }
            )
    gates = pd.DataFrame(gate_rows)
    gates["selection_score"] = (
        (1.0 - gates["active_recall"])
        + 1.5 * gates["zero_false_positive_rate"]
        + 0.75 * (1.0 - gates["trial008_active_recall"])
    )
    gates = gates.sort_values("selection_score", kind="stable")

    best_regression_id = str(regression.iloc[0]["model_id"])
    best_gate_id = str(gates.iloc[0]["model_id"])
    best_threshold = float(gates.iloc[0]["threshold"])
    force = regression_predictions[best_regression_id]
    gate = gate_predictions[best_gate_id].copy()
    gate["contact_gate_active"] = (
        gate["contact_probability"].to_numpy(dtype=float) >= best_threshold
    )
    gate["model_id"] = best_gate_id
    gate["threshold"] = best_threshold
    gate["evaluation_validity"] = "grouped_oof_by_session_model_discovery"
    gate["force_sensor_used_as_runtime_input"] = False
    merged = force.merge(
        gate[["group_id", "sample_index", "contact_target", "contact_probability"]],
        on=["group_id", "sample_index"],
        how="inner",
        suffixes=("_force", "_gate"),
        validate="one_to_one",
    )
    merged["gated_force_n"] = np.where(
        merged["contact_probability"] >= best_threshold,
        merged["raw_force_n"],
        0.0,
    )
    combined_metrics = _safe_curve_metrics(
        merged["true_force_n"], merged["gated_force_n"]
    )
    combined_metrics.update(
        {
            "force_model_id": best_regression_id,
            "gate_model_id": best_gate_id,
            "gate_threshold": best_threshold,
            "global_contact_gate_zero_false_positive_rate": float(
                gates.iloc[0]["zero_false_positive_rate"]
            ),
        }
    )

    regression.to_csv(output / "regression_model_comparison.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(output / "contact_gate_comparison.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(
        output / "best_contact_gate_grouped_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    merged.to_csv(output / "best_combined_grouped_oof.csv", index=False, encoding="utf-8-sig")
    (output / "best_combined_metrics.json").write_text(
        json.dumps(_json_safe(combined_metrics), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("FORCE")
    print(regression.head(8).to_string(index=False))
    print("\nGATE")
    print(gates.head(8).to_string(index=False))
    print("\nCOMBINED")
    print(json.dumps(_json_safe(combined_metrics), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
