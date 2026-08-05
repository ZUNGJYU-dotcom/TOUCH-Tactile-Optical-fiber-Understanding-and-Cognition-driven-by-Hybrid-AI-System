"""Leakage-safe full-spectrum force experts for sensitivity-shifted positions.

The ordinary-FBG force data show that one tactile point can retain a strong
within-session force trend while its absolute optical sensitivity changes
between sessions.  A latent spectral regressor can use covariance across the
whole spectrum instead of relying on one peak amplitude.  This module keeps
that expert strictly grouped by acquisition session and routes it from an
optical position vote; PX6D force is supervision and evaluation evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from .advanced_optical_benchmark import AlignedOpticalDataset


@dataclass(frozen=True)
class SpectralForceExpertSpec:
    """Configuration for one position-specific latent spectral regressor."""

    position_id: str = "P13"
    model_id: str = "p13_full_spectrum_pls12"
    feature_count: int = 264
    latent_components: int = 12
    minimum_force_n: float = 0.10
    maximum_force_n: float = 5.0


def formal_array_indices(fusion_dataset_path: Any) -> np.ndarray:
    """Return source row indices in the same order as aligned formal rows."""

    with np.load(fusion_dataset_path, allow_pickle=False) as payload:
        return np.flatnonzero(
            payload["formal_test_eligible"].astype(bool)
            & (payload["fold_id"].astype(int) >= 0)
        ).astype(int)


def causal_lag_matrix(
    features: np.ndarray,
    groups: Sequence[Any],
    sample_index: Sequence[int],
    lag: int,
) -> np.ndarray:
    """Return X[t-lag] without crossing acquisition-session boundaries."""

    matrix = np.asarray(features)
    group_values = np.asarray(groups).astype(str)
    indices = np.asarray(sample_index, dtype=int)
    lagged = np.empty_like(matrix)
    for group_id in np.unique(group_values):
        rows = np.flatnonzero(group_values == group_id)
        order = rows[np.argsort(indices[rows], kind="stable")]
        source = np.maximum(np.arange(len(order)) - max(0, int(lag)), 0)
        lagged[order] = matrix[order[source]]
    return lagged


def build_causal_spectral_views(
    dataset: AlignedOpticalDataset,
    *,
    feature_count: int = 264,
) -> dict[str, np.ndarray]:
    """Build current and strictly causal spectral views for temporal models."""

    if dataset.spectrum_features.shape[1] < int(feature_count):
        raise ValueError(
            f"spectrum has {dataset.spectrum_features.shape[1]} features; "
            f"causal views require {feature_count}"
        )
    current = dataset.spectrum_features[:, : int(feature_count)].astype(
        np.float32, copy=False
    )
    lag1 = causal_lag_matrix(current, dataset.group_id, dataset.sample_index, 1)
    lag3 = causal_lag_matrix(current, dataset.group_id, dataset.sample_index, 3)
    lag8 = causal_lag_matrix(current, dataset.group_id, dataset.sample_index, 8)
    return {
        "current264": current,
        "current_plus_lag3": np.concatenate((current, lag3), axis=1),
        "current_plus_delta3": np.concatenate((current, current - lag3), axis=1),
        "current_plus_delta1_3_8": np.concatenate(
            (current, current - lag1, current - lag3, current - lag8), axis=1
        ),
        "current_plus_lag_delta1_3_8": np.concatenate(
            (
                current,
                lag1,
                lag3,
                lag8,
                current - lag1,
                current - lag3,
                current - lag8,
            ),
            axis=1,
        ),
    }


def load_group_baseline_spectral_views(
    data_root: Path,
    groups: Sequence[Any],
    *,
    bin_count: int = 64,
) -> dict[str, np.ndarray]:
    """Load optical-only baseline signatures aligned to acquisition rows.

    Each synchronized capture stores the no-contact baseline spectrum that was
    active during acquisition.  Keeping its absolute log-count shape gives a
    force model a chance to recognize changed optical coupling or exposure
    without using the held-out PX6D trace as a session-specific calibration.
    """

    root = Path(data_root)
    group_values = np.asarray(groups).astype(str)
    if int(bin_count) < 2:
        raise ValueError("baseline spectrum bin count must be at least two")
    signatures: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group_id in dict.fromkeys(group_values.tolist()):
        source = root / str(group_id) / "spectrum_timeseries.csv"
        if not source.exists():
            raise FileNotFoundError(
                f"baseline spectrum source not found for {group_id}: {source}"
            )
        frame = pd.read_csv(
            source,
            usecols=["point_index", "baseline_intensity_counts"],
            low_memory=False,
        )
        frame["point_index"] = pd.to_numeric(frame["point_index"], errors="coerce")
        frame["baseline_intensity_counts"] = pd.to_numeric(
            frame["baseline_intensity_counts"], errors="coerce"
        )
        frame = frame[
            np.isfinite(frame["point_index"])
            & np.isfinite(frame["baseline_intensity_counts"])
        ]
        baseline = (
            frame.groupby("point_index", sort=True)["baseline_intensity_counts"]
            .median()
            .to_numpy(dtype=float)
        )
        if len(baseline) < int(bin_count):
            raise ValueError(
                f"baseline spectrum for {group_id} has only {len(baseline)} points"
            )
        edges = np.linspace(0, len(baseline), int(bin_count) + 1, dtype=int)
        binned = np.asarray(
            [np.mean(baseline[edges[index] : edges[index + 1]]) for index in range(int(bin_count))],
            dtype=float,
        )
        log_counts = np.log1p(np.clip(binned, 0.0, None))
        log_shape = log_counts - float(np.mean(log_counts))
        summary = np.asarray(
            [
                float(np.mean(log_counts)),
                float(np.std(log_counts)),
                float(np.percentile(log_counts, 10)),
                float(np.median(log_counts)),
                float(np.percentile(log_counts, 90)),
                float(np.ptp(log_counts)),
            ],
            dtype=float,
        )
        signatures[str(group_id)] = (log_counts, log_shape, summary)

    log_matrix = np.vstack([signatures[group][0] for group in group_values])
    shape_matrix = np.vstack([signatures[group][1] for group in group_values])
    summary_matrix = np.vstack([signatures[group][2] for group in group_values])
    return {
        "baseline_log_counts": log_matrix.astype(np.float32),
        "baseline_log_shape": shape_matrix.astype(np.float32),
        "baseline_summary": summary_matrix.astype(np.float32),
    }


def build_baseline_conditioned_spectral_views(
    dataset: AlignedOpticalDataset,
    data_root: Path,
) -> dict[str, np.ndarray]:
    """Combine causal response features with the session baseline spectrum."""

    causal = build_causal_spectral_views(dataset)
    baseline = load_group_baseline_spectral_views(data_root, dataset.group_id)
    current_log_ratio = dataset.spectrum_features[:, :64].astype(
        np.float32, copy=False
    )
    interaction = current_log_ratio * baseline["baseline_log_shape"]
    baseline_context = np.concatenate(
        (
            baseline["baseline_log_counts"],
            baseline["baseline_log_shape"],
            baseline["baseline_summary"],
        ),
        axis=1,
    )
    return {
        "current264_plus_baseline": np.concatenate(
            (causal["current264"], baseline_context), axis=1
        ),
        "current264_plus_baseline_interaction": np.concatenate(
            (causal["current264"], baseline_context, interaction), axis=1
        ),
        "lag_delta_plus_baseline": np.concatenate(
            (causal["current_plus_lag_delta1_3_8"], baseline_context), axis=1
        ),
        "lag_delta_plus_baseline_interaction": np.concatenate(
            (
                causal["current_plus_lag_delta1_3_8"],
                baseline_context,
                interaction,
            ),
            axis=1,
        ),
    }


def _optical_conditions(
    groups: Sequence[Any],
    group_position_votes: Mapping[str, str],
) -> np.ndarray:
    return np.asarray(
        [str(group_position_votes.get(str(group), "unknown_position")) for group in groups],
        dtype=str,
    )


def grouped_spectral_force_expert_oof(
    dataset: AlignedOpticalDataset,
    array_indices: Sequence[int],
    spec: SpectralForceExpertSpec,
    *,
    group_position_votes: Mapping[str, str],
    gate_active_by_array_index: Mapping[int, bool],
    feature_matrix: np.ndarray | None = None,
    excluded_group_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict routed formal rows with a grouped full-spectrum PLS expert.

    The model is trained only on force-labelled active rows from the specified
    position.  In an outer fold it is applied to every row whose acquisition
    session was assigned to that position by an optical-only grouped vote.
    This includes no-contact/release rows; the existing optical contact gate
    owns the physical zero state.
    """

    source_indices = np.asarray(array_indices, dtype=int)
    if len(source_indices) != len(dataset.group_id):
        raise ValueError("formal array indices and aligned dataset rows differ")
    if feature_matrix is None:
        if dataset.spectrum_features.shape[1] < int(spec.feature_count):
            raise ValueError(
                f"spectrum has {dataset.spectrum_features.shape[1]} features; "
                f"expert requires {spec.feature_count}"
            )
        features = dataset.spectrum_features[:, : int(spec.feature_count)]
    else:
        features = np.asarray(feature_matrix)
        if features.ndim != 2:
            raise ValueError("spectral expert feature matrix must be two-dimensional")
        if features.shape[0] != len(dataset.group_id):
            raise ValueError("spectral expert feature rows differ from aligned dataset")
        if not np.all(np.isfinite(features)):
            raise ValueError("spectral expert feature matrix contains non-finite values")
    optical_condition = _optical_conditions(dataset.group_id, group_position_votes)
    excluded_groups = {str(value) for value in (excluded_group_ids or ())}
    quality_eligible = np.asarray(
        [str(group_id) not in excluded_groups for group_id in dataset.group_id],
        dtype=bool,
    )
    formal_force = dataset.force_mask & (dataset.fold_id >= 0)
    training_target = (
        formal_force
        & quality_eligible
        & (dataset.position_target.astype(str) == str(spec.position_id))
        & (dataset.force_fz_n >= float(spec.minimum_force_n))
        & (dataset.force_fz_n <= float(spec.maximum_force_n))
    )
    folds = sorted(set(dataset.fold_id[formal_force].astype(int).tolist()))
    if len(folds) < 3:
        raise ValueError("spectral expert requires at least three grouped folds")

    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = training_target & (dataset.fold_id != int(fold))
        routed_test = (
            formal_force
            & quality_eligible
            & (dataset.fold_id == int(fold))
            & (optical_condition == str(spec.position_id))
        )
        train_groups = sorted(set(dataset.group_id[train].tolist()))
        test_groups = sorted(set(dataset.group_id[routed_test].tolist()))
        overlap = sorted(set(train_groups).intersection(test_groups))
        if overlap:
            raise RuntimeError("spectral expert group leakage detected")
        if len(train_groups) < 3 or int(np.sum(routed_test)) == 0:
            audit_rows.append(
                {
                    "fold_id": int(fold),
                    "train_sample_count": int(np.sum(train)),
                    "train_group_count": len(train_groups),
                    "routed_test_sample_count": int(np.sum(routed_test)),
                    "routed_test_group_count": len(test_groups),
                    "train_test_group_overlap_count": 0,
                    "excluded_group_count": len(excluded_groups),
                    "status": "insufficient_training_or_routed_test_data",
                }
            )
            continue

        components = min(
            int(spec.latent_components),
            features.shape[1],
            max(1, int(np.sum(train)) - 1),
        )
        model = PLSRegression(
            n_components=components,
            scale=True,
            max_iter=1000,
            tol=1.0e-06,
        )
        model.fit(features[train], dataset.force_fz_n[train])
        raw = np.asarray(model.predict(features[routed_test])).reshape(-1)
        raw = np.clip(raw, 0.0, float(spec.maximum_force_n))
        routed_indices = np.flatnonzero(routed_test)
        gate_active = np.asarray(
            [
                bool(gate_active_by_array_index.get(int(source_indices[index]), False))
                for index in routed_indices
            ],
            dtype=bool,
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "array_index": source_indices[routed_test],
                    "fold_id": int(fold),
                    "group_id": dataset.group_id[routed_test],
                    "sample_index": dataset.sample_index[routed_test],
                    "position_condition": optical_condition[routed_test],
                    "spectral_raw_force_n": raw,
                    "contact_gate_active": gate_active,
                    "spectral_gated_force_n": np.where(gate_active, raw, 0.0),
                    "spectral_expert_model_id": spec.model_id,
                    "evaluation_validity": "formal_grouped_oof_by_session_id",
                    "force_sensor_used_as_runtime_input": False,
                }
            )
        )
        audit_rows.append(
            {
                "fold_id": int(fold),
                "train_sample_count": int(np.sum(train)),
                "train_group_count": len(train_groups),
                "routed_test_sample_count": int(np.sum(routed_test)),
                "routed_test_group_count": len(test_groups),
                "train_test_group_overlap_count": 0,
                "excluded_group_count": len(excluded_groups),
                "latent_components": components,
                "status": "ok",
            }
        )

    if not prediction_parts:
        raise ValueError("spectral expert produced no grouped OOF predictions")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions["array_index"].duplicated().any():
        raise RuntimeError("spectral expert produced duplicate formal rows")
    return predictions.sort_values("array_index"), pd.DataFrame(audit_rows)


def apply_spectral_expert_override(
    baseline_predictions: pd.DataFrame,
    expert_predictions: pd.DataFrame,
    *,
    model_id: str,
) -> pd.DataFrame:
    """Replace only optically routed rows while preserving all other estimates."""

    required = {
        "array_index",
        "base_raw_force_n",
        "calibrated_force_n",
        "gated_force_n",
        "contact_gate_active",
    }
    missing = sorted(required - set(baseline_predictions.columns))
    if missing:
        raise ValueError("baseline predictions missing: " + ", ".join(missing))
    expert_required = {
        "array_index",
        "spectral_raw_force_n",
        "spectral_gated_force_n",
        "contact_gate_active",
        "spectral_expert_model_id",
    }
    missing = sorted(expert_required - set(expert_predictions.columns))
    if missing:
        raise ValueError("expert predictions missing: " + ", ".join(missing))

    result = baseline_predictions.copy()
    if result["array_index"].duplicated().any():
        raise ValueError("baseline model must contain one row per array index")
    expert = expert_predictions.set_index("array_index")
    result_index = result["array_index"].astype(int)
    routed = result_index.isin(expert.index)
    if not bool(routed.any()):
        raise ValueError("spectral expert does not overlap baseline predictions")
    routed_indices = result_index[routed].to_numpy(dtype=int)
    aligned = expert.loc[routed_indices]
    baseline_gate = result.loc[routed, "contact_gate_active"].astype(bool).to_numpy()
    expert_gate = aligned["contact_gate_active"].astype(bool).to_numpy()
    if not np.array_equal(baseline_gate, expert_gate):
        raise ValueError("baseline and spectral expert contact gates disagree")

    raw = aligned["spectral_raw_force_n"].to_numpy(dtype=float)
    gated = aligned["spectral_gated_force_n"].to_numpy(dtype=float)
    result.loc[routed, "base_raw_force_n"] = raw
    result.loc[routed, "calibrated_force_n"] = raw
    result.loc[routed, "gated_force_n"] = gated
    result.loc[routed, "expert_used"] = aligned[
        "spectral_expert_model_id"
    ].to_numpy(dtype=str)
    result.loc[routed, "position_condition_source"] = (
        "grouped_oof_optical_position_vote"
    )
    result["model_id"] = str(model_id)
    result["force_sensor_used_as_runtime_input"] = False
    result["evaluation_validity"] = "formal_grouped_oof_by_session_id"
    return result


__all__ = [
    "SpectralForceExpertSpec",
    "apply_spectral_expert_override",
    "build_causal_spectral_views",
    "build_baseline_conditioned_spectral_views",
    "causal_lag_matrix",
    "formal_array_indices",
    "grouped_spectral_force_expert_oof",
    "load_group_baseline_spectral_views",
]
