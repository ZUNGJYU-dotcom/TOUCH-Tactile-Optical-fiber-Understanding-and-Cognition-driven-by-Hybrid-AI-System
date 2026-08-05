"""Session-level audits for grouped rich-optical benchmark predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "model_id",
    "feature_view",
    "task",
    "group_id",
    "sample_index",
    "fold_id",
    "true_value",
    "predicted_value",
}


@dataclass(frozen=True)
class PredictionSpec:
    model_id: str
    feature_view: str


def _select_predictions(
    predictions: pd.DataFrame,
    *,
    task: str,
    spec: PredictionSpec,
    value_kind: str,
) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")
    selected = predictions.loc[
        (predictions["task"] == task)
        & (predictions["model_id"] == spec.model_id)
        & (predictions["feature_view"] == spec.feature_view)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"no predictions for {task}: {spec.model_id}/{spec.feature_view}"
        )
    if selected.duplicated(["group_id", "sample_index"]).any():
        raise ValueError(
            f"duplicate grouped predictions for {task}: "
            f"{spec.model_id}/{spec.feature_view}"
        )
    if value_kind == "float":
        selected["true_value"] = pd.to_numeric(
            selected["true_value"], errors="raise"
        ).astype(float)
        selected["predicted_value"] = pd.to_numeric(
            selected["predicted_value"], errors="raise"
        ).astype(float)
    elif value_kind == "integer":
        selected["true_value"] = pd.to_numeric(
            selected["true_value"], errors="raise"
        ).astype(int)
        selected["predicted_value"] = pd.to_numeric(
            selected["predicted_value"], errors="raise"
        ).astype(int)
    elif value_kind == "string":
        selected["true_value"] = selected["true_value"].astype(str)
        selected["predicted_value"] = selected["predicted_value"].astype(str)
    else:
        raise ValueError(f"unsupported value kind: {value_kind}")
    return selected


def _aligned_pair(
    predictions: pd.DataFrame,
    *,
    task: str,
    baseline: PredictionSpec,
    candidate: PredictionSpec,
    value_kind: str,
) -> pd.DataFrame:
    baseline_rows = _select_predictions(
        predictions, task=task, spec=baseline, value_kind=value_kind
    ).rename(
        columns={
            "true_value": "baseline_true_value",
            "predicted_value": "baseline_predicted_value",
            "fold_id": "baseline_fold_id",
        }
    )
    candidate_rows = _select_predictions(
        predictions, task=task, spec=candidate, value_kind=value_kind
    ).rename(
        columns={
            "true_value": "candidate_true_value",
            "predicted_value": "candidate_predicted_value",
            "fold_id": "candidate_fold_id",
        }
    )
    keep = [
        "group_id",
        "sample_index",
        "baseline_fold_id",
        "baseline_true_value",
        "baseline_predicted_value",
    ]
    other = [
        "group_id",
        "sample_index",
        "candidate_fold_id",
        "candidate_true_value",
        "candidate_predicted_value",
    ]
    aligned = baseline_rows[keep].merge(
        candidate_rows[other],
        on=["group_id", "sample_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(baseline_rows) or len(aligned) != len(candidate_rows):
        raise ValueError(f"baseline and candidate frames do not align for {task}")
    if not np.array_equal(
        aligned["baseline_fold_id"].to_numpy(),
        aligned["candidate_fold_id"].to_numpy(),
    ):
        raise ValueError(f"baseline and candidate folds do not align for {task}")
    baseline_true = aligned["baseline_true_value"].to_numpy()
    candidate_true = aligned["candidate_true_value"].to_numpy()
    if value_kind == "float":
        truth_matches = np.allclose(
            baseline_true.astype(float),
            candidate_true.astype(float),
            rtol=0.0,
            atol=1e-9,
        )
    else:
        truth_matches = np.array_equal(baseline_true, candidate_true)
    if not truth_matches:
        raise ValueError(f"baseline and candidate truth do not align for {task}")
    return aligned


def classification_session_audit(
    predictions: pd.DataFrame,
    *,
    task: str,
    baseline: PredictionSpec,
    candidate: PredictionSpec,
    value_kind: str,
) -> pd.DataFrame:
    """Compare grouped classifier predictions one capture session at a time."""

    aligned = _aligned_pair(
        predictions,
        task=task,
        baseline=baseline,
        candidate=candidate,
        value_kind=value_kind,
    )
    rows: list[dict[str, Any]] = []
    for group_id, group in aligned.groupby("group_id", sort=True):
        true = group["baseline_true_value"].to_numpy()
        baseline_predicted = group["baseline_predicted_value"].to_numpy()
        candidate_predicted = group["candidate_predicted_value"].to_numpy()
        true_counts = pd.Series(true).value_counts()
        dominant_truth = true_counts.index[0]
        baseline_accuracy = float(np.mean(baseline_predicted == true))
        candidate_accuracy = float(np.mean(candidate_predicted == true))
        row: dict[str, Any] = {
                "group_id": str(group_id),
                "fold_id": int(group["baseline_fold_id"].iloc[0]),
                "task": task,
                "frame_count": int(len(group)),
                "dominant_true_label": dominant_truth,
                "dominant_true_ratio": float(true_counts.iloc[0] / len(group)),
                "baseline_model_id": baseline.model_id,
                "baseline_feature_view": baseline.feature_view,
                "candidate_model_id": candidate.model_id,
                "candidate_feature_view": candidate.feature_view,
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "accuracy_delta": candidate_accuracy - baseline_accuracy,
                "baseline_error_count": int(np.sum(baseline_predicted != true)),
                "candidate_error_count": int(np.sum(candidate_predicted != true)),
        }
        if task == "contact" and value_kind == "integer":
            no_contact = true == 0
            active_contact = true == 1
            for label, selected in (
                ("no_contact", no_contact),
                ("active_contact", active_contact),
            ):
                row[f"{label}_frame_count"] = int(np.sum(selected))
                if np.any(selected):
                    baseline_recall = float(
                        np.mean(baseline_predicted[selected] == true[selected])
                    )
                    candidate_recall = float(
                        np.mean(candidate_predicted[selected] == true[selected])
                    )
                else:
                    baseline_recall = np.nan
                    candidate_recall = np.nan
                row[f"baseline_{label}_recall"] = baseline_recall
                row[f"candidate_{label}_recall"] = candidate_recall
                row[f"{label}_recall_delta"] = (
                    candidate_recall - baseline_recall
                    if np.isfinite(candidate_recall) and np.isfinite(baseline_recall)
                    else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def force_session_audit(
    predictions: pd.DataFrame,
    *,
    baseline: PredictionSpec,
    candidate: PredictionSpec,
    active_force_threshold_n: float = 0.10,
) -> pd.DataFrame:
    """Compare grouped optical-only Fz predictions one session at a time."""

    aligned = _aligned_pair(
        predictions,
        task="force_fz",
        baseline=baseline,
        candidate=candidate,
        value_kind="float",
    )
    rows: list[dict[str, Any]] = []
    for group_id, group in aligned.groupby("group_id", sort=True):
        true = group["baseline_true_value"].to_numpy(dtype=float)
        baseline_predicted = group["baseline_predicted_value"].to_numpy(dtype=float)
        candidate_predicted = group["candidate_predicted_value"].to_numpy(dtype=float)
        baseline_error = np.abs(true - baseline_predicted)
        candidate_error = np.abs(true - candidate_predicted)
        active = true >= active_force_threshold_n
        rows.append(
            {
                "group_id": str(group_id),
                "fold_id": int(group["baseline_fold_id"].iloc[0]),
                "task": "force_fz",
                "frame_count": int(len(group)),
                "true_fz_mean_n": float(np.mean(true)),
                "true_fz_max_n": float(np.max(true)),
                "active_frame_count": int(np.sum(active)),
                "baseline_model_id": baseline.model_id,
                "baseline_feature_view": baseline.feature_view,
                "candidate_model_id": candidate.model_id,
                "candidate_feature_view": candidate.feature_view,
                "baseline_mae_n": float(np.mean(baseline_error)),
                "candidate_mae_n": float(np.mean(candidate_error)),
                "mae_improvement_n": float(
                    np.mean(baseline_error) - np.mean(candidate_error)
                ),
                "baseline_active_mae_n": (
                    float(np.mean(baseline_error[active])) if np.any(active) else np.nan
                ),
                "candidate_active_mae_n": (
                    float(np.mean(candidate_error[active])) if np.any(active) else np.nan
                ),
                "within_0_25_n_baseline": float(np.mean(baseline_error <= 0.25)),
                "within_0_25_n_candidate": float(np.mean(candidate_error <= 0.25)),
            }
        )
    return pd.DataFrame(rows)


def delta_summary(
    audit: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> dict[str, float | int]:
    """Summarize how broadly a candidate improves independent sessions."""

    values = pd.to_numeric(audit[metric], errors="coerce").dropna().to_numpy(float)
    if not higher_is_better:
        values = -values
    return {
        "session_count": int(len(values)),
        "improved_sessions": int(np.sum(values > tolerance)),
        "worsened_sessions": int(np.sum(values < -tolerance)),
        "tied_sessions": int(np.sum(np.abs(values) <= tolerance)),
        "median_improvement": float(np.median(values)) if len(values) else np.nan,
        "mean_improvement": float(np.mean(values)) if len(values) else np.nan,
    }


def optical_feature_category(feature: str) -> str:
    """Map detailed optical features to interpretable evidence families."""

    name = feature.lower()
    if "same_fibre" in name:
        return "same_fibre_coupling"
    if "spatial_row" in name:
        return "cross_fibre_spatial_coupling"
    if "baseline_peak_snr" in name or name.endswith("peak_snr"):
        return "spectral_quality_snr"
    if "shape_rmse" in name or "shape_correlation" in name or "residual" in name:
        return "distributed_spectral_shape"
    if "fwhm" in name or "skew" in name or "estimator_disagreement" in name:
        return "local_peak_shape"
    if "centroid" in name or "parabolic" in name or "shift_pm" in name:
        return "wavelength_shift"
    if "log_area" in name or "log_height" in name:
        return "intensity_and_peak_area"
    if "spectrum__" in name or "spectrum_" in name:
        return "full_spectrum_bins"
    if "global" in name or "common_mode" in name:
        return "global_common_mode"
    return "other_optical_evidence"


def feature_category_audit(
    feature_importance: pd.DataFrame,
    *,
    task: str,
    spec: PredictionSpec,
) -> pd.DataFrame:
    selected = feature_importance.loc[
        (feature_importance["task"] == task)
        & (feature_importance["model_id"] == spec.model_id)
        & (feature_importance["feature_view"] == spec.feature_view)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"no feature importance for {task}: {spec.model_id}/{spec.feature_view}"
        )
    selected["evidence_family"] = selected["feature"].map(optical_feature_category)
    grouped = (
        selected.groupby("evidence_family", as_index=False)
        .agg(
            top_feature_count=("feature", "size"),
            summed_importance=("importance", "sum"),
            highest_feature_importance=("importance", "max"),
        )
        .sort_values("summed_importance", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["summed_importance"].sum())
    grouped["top_importance_share"] = (
        grouped["summed_importance"] / total if total > 0 else np.nan
    )
    grouped.insert(0, "task", task)
    grouped.insert(1, "model_id", spec.model_id)
    grouped.insert(2, "feature_view", spec.feature_view)
    return grouped


__all__ = [
    "PredictionSpec",
    "classification_session_audit",
    "delta_summary",
    "feature_category_audit",
    "force_session_audit",
    "optical_feature_category",
]
