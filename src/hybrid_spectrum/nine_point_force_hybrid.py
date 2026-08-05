"""Assembly helpers for leakage-safe nine-point optical force candidates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .advanced_optical_benchmark import AlignedOpticalDataset


def _coerce_boolean_series(values: pd.Series) -> np.ndarray:
    """Convert common CSV boolean encodings without treating ``"False"`` as true."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_known = numeric.notna()
    result = np.zeros(len(values), dtype=bool)
    if bool(numeric_known.any()):
        result[numeric_known.to_numpy()] = (
            numeric.loc[numeric_known].to_numpy(dtype=float) != 0.0
        )
    text = values.astype("string").str.strip().str.lower()
    text_true = text.isin({"true", "yes", "y", "on"})
    result[text_true.to_numpy()] = True
    return result


def grouped_gate_truth_masks(
    gate_predictions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return labelled-row and active-contact masks for old and new gate exports.

    Older grouped OOF files store a numeric ``contact_target``.  Newer exports
    preserve the semantic ``true_contact_label`` and explicitly indicate rows
    without a contact label through ``contact_label_available``.  Unknown rows
    are excluded instead of silently becoming no-contact examples.
    """

    if "contact_label_available" in gate_predictions.columns:
        available = _coerce_boolean_series(
            gate_predictions["contact_label_available"]
        )
    else:
        available = np.ones(len(gate_predictions), dtype=bool)

    if "contact_target" in gate_predictions.columns:
        target = pd.to_numeric(
            gate_predictions["contact_target"], errors="coerce"
        )
        valid = target.isin([0, 1]).to_numpy(dtype=bool)
        active = target.eq(1).fillna(False).to_numpy(dtype=bool)
    elif "true_contact_label" in gate_predictions.columns:
        labels = (
            gate_predictions["true_contact_label"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        active_labels = {"contact", "active", "active_contact"}
        inactive_labels = {"no_contact", "no contact", "inactive"}
        valid = labels.isin(active_labels | inactive_labels).to_numpy(dtype=bool)
        active = labels.isin(active_labels).to_numpy(dtype=bool)
    else:
        raise ValueError(
            "gate predictions require contact_target or true_contact_label"
        )

    available &= valid
    active &= available
    return available, active


def apply_grouped_contact_gate(
    baseline_predictions: pd.DataFrame,
    gate_predictions: pd.DataFrame,
    *,
    model_id: str,
) -> pd.DataFrame:
    """Replace a legacy gate with grouped OOF optical contact evidence."""

    baseline_required = {
        "group_id",
        "sample_index",
        "calibrated_force_n",
        "contact_gate_active",
    }
    gate_required = {
        "group_id",
        "sample_index",
        "contact_probability",
        "contact_gate_active",
    }
    missing = sorted(baseline_required - set(baseline_predictions.columns))
    if missing:
        raise ValueError("baseline predictions missing: " + ", ".join(missing))
    missing = sorted(gate_required - set(gate_predictions.columns))
    if missing:
        raise ValueError("gate predictions missing: " + ", ".join(missing))
    keys = ["group_id", "sample_index"]
    if baseline_predictions.duplicated(keys).any():
        raise ValueError("baseline predictions contain duplicate group/sample rows")
    if gate_predictions.duplicated(keys).any():
        raise ValueError("gate predictions contain duplicate group/sample rows")

    gate = gate_predictions[
        keys + ["contact_probability", "contact_gate_active"]
    ].rename(
        columns={
            "contact_probability": "replacement_contact_probability",
            "contact_gate_active": "replacement_contact_gate_active",
        }
    )
    merged = baseline_predictions.merge(
        gate,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    missing_gate = merged["replacement_contact_gate_active"].isna()
    if bool(missing_gate.any()):
        examples = merged.loc[missing_gate, keys].head(3).to_dict("records")
        raise ValueError(f"grouped contact gate misses baseline rows: {examples}")
    merged["legacy_contact_gate_active"] = merged["contact_gate_active"].astype(bool)
    merged["contact_probability"] = merged[
        "replacement_contact_probability"
    ].to_numpy(dtype=float)
    merged["contact_gate_active"] = merged[
        "replacement_contact_gate_active"
    ].astype(bool)
    merged["gated_force_n"] = np.where(
        merged["contact_gate_active"].to_numpy(dtype=bool),
        merged["calibrated_force_n"].to_numpy(dtype=float),
        0.0,
    )
    merged = merged.drop(
        columns=[
            "replacement_contact_probability",
            "replacement_contact_gate_active",
        ]
    )
    merged["contact_gate_model_id"] = str(model_id)
    merged["force_sensor_used_as_runtime_input"] = False
    merged["evaluation_validity"] = "formal_grouped_oof_by_session_id"
    return merged


def aligned_contact_gate_map(
    dataset: AlignedOpticalDataset,
    array_indices: Sequence[int],
    gate_predictions: pd.DataFrame,
) -> tuple[dict[int, bool], dict[str, Any]]:
    """Align grouped gate rows to force-labelled formal array indices.

    Contact-gate OOF exports intentionally contain only rows eligible for the
    configured force task.  The aligned optical dataset can also contain rows
    outside that force range, so those rows must not be treated as missing
    gate predictions.
    """

    source_indices = np.asarray(array_indices, dtype=int)
    if len(source_indices) != len(dataset.group_id):
        raise ValueError("formal array indices and aligned dataset rows differ")
    keys = ["group_id", "sample_index"]
    if gate_predictions.duplicated(keys).any():
        raise ValueError("gate predictions contain duplicate group/sample rows")
    lookup = {
        (str(row.group_id), int(row.sample_index)): bool(row.contact_gate_active)
        for row in gate_predictions.itertuples(index=False)
    }
    result: dict[int, bool] = {}
    missing: list[tuple[str, int]] = []
    formal_force = dataset.force_mask & (dataset.fold_id >= 0)
    for array_index, group_id, sample_index, force_eligible in zip(
        source_indices,
        dataset.group_id,
        dataset.sample_index,
        formal_force,
        strict=True,
    ):
        if not bool(force_eligible):
            continue
        key = (str(group_id), int(sample_index))
        if key not in lookup:
            missing.append(key)
            continue
        result[int(array_index)] = bool(lookup[key])
    formal_force_count = int(np.sum(formal_force))
    if missing:
        raise ValueError(
            "grouped contact gate misses force-eligible aligned rows: "
            f"{missing[:3]}"
        )
    return result, {
        "aligned_row_count": int(len(source_indices)),
        "formal_force_row_count": formal_force_count,
        "mapped_row_count": int(len(result)),
        "missing_row_count": 0,
        "skipped_non_force_row_count": int(len(source_indices) - formal_force_count),
        "status": "ok",
    }


def build_session_acceptance_table(
    sessions: pd.DataFrame,
    *,
    maximum_lag_aligned_mae_n: float = 0.80,
    minimum_trend_correlation: float = 0.80,
    minimum_slope: float = 0.65,
    maximum_slope: float = 1.35,
    maximum_zero_force_false_response_rate: float = 0.10,
) -> pd.DataFrame:
    """Evaluate per-session trend, height and release behavior."""

    rows: list[dict[str, Any]] = []
    for row in sessions.itertuples(index=False):
        raw_r = getattr(row, "pearson_r", np.nan)
        aligned_r = getattr(row, "lag_aligned_pearson_r", np.nan)
        candidates = [
            float(value)
            for value in (raw_r, aligned_r)
            if value is not None and np.isfinite(value)
        ]
        trend_r = max(candidates) if candidates else np.nan
        lag_mae = getattr(row, "lag_aligned_mae_n", np.nan)
        slope = getattr(row, "linear_slope_pred_vs_px6d", np.nan)
        raw_false_rate = getattr(row, "zero_force_false_response_rate", np.nan)
        false_rate = getattr(
            row,
            "zero_force_false_response_rate_after_grace",
            raw_false_rate,
        )
        trend_ok = bool(np.isfinite(trend_r) and trend_r >= minimum_trend_correlation)
        error_ok = bool(
            lag_mae is not None
            and np.isfinite(lag_mae)
            and lag_mae <= maximum_lag_aligned_mae_n
        )
        height_ok = bool(
            slope is not None
            and np.isfinite(slope)
            and minimum_slope <= slope <= maximum_slope
        )
        release_ok = bool(
            false_rate is None
            or not np.isfinite(false_rate)
            or false_rate <= maximum_zero_force_false_response_rate
        )
        failures = [
            name
            for name, passed in (
                ("trend", trend_ok),
                ("height", height_ok),
                ("lag_aligned_error", error_ok),
                ("release", release_ok),
            )
            if not passed
        ]
        rows.append(
            {
                "position_id": str(row.position_id),
                "group_id": str(row.group_id),
                "trend_correlation_best": trend_r,
                "lag_ms": getattr(row, "lag_ms", np.nan),
                "lag_aligned_mae_n": lag_mae,
                "linear_slope_pred_vs_px6d": slope,
                "zero_force_false_response_rate_raw": raw_false_rate,
                "zero_force_false_response_rate": false_rate,
                "release_grace_sec": getattr(row, "release_grace_sec", 0.0),
                "trend_ok": trend_ok,
                "height_ok": height_ok,
                "lag_aligned_error_ok": error_ok,
                "release_ok": release_ok,
                "session_curve_status": "acceptable" if not failures else "needs_review",
                "failure_reasons": ";".join(failures),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "aligned_contact_gate_map",
    "apply_grouped_contact_gate",
    "build_session_acceptance_table",
    "grouped_gate_truth_masks",
]
