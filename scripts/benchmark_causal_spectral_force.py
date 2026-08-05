"""Benchmark leakage-safe causal spectral force views for weak P13 sessions.

This is a model-discovery benchmark, not a deployment script.  Every feature
at frame t is built from the current spectrum and earlier frames in the same
acquisition session.  Outer-fold PLS models never see the held-out session.
PX6D Fz is used only as supervision and evaluation evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    AlignedOpticalDataset,
    build_feature_views,
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.spectral_force_experts import (  # noqa: E402
    build_baseline_conditioned_spectral_views,
    build_causal_spectral_views,
)


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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/ordinary_fbg_causal_spectral_force_20260804"
DEFAULT_RAW_DATA = PROJECT_ROOT.parents[1] / "data" / "new data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--position", default="P13")
    parser.add_argument("--raw-data-root", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _safe_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | None]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if len(reference) == 0:
        return {"mae_n": None, "r2": None, "pearson_r": None, "slope": None}
    correlation = None
    if len(reference) >= 3 and np.std(reference) > 1.0e-12 and np.std(estimate) > 1.0e-12:
        correlation = float(np.corrcoef(reference, estimate)[0, 1])
    slope = None
    if np.std(reference) > 1.0e-12:
        slope = float(np.polyfit(reference, estimate, 1)[0])
    return {
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "r2": float(r2_score(reference, estimate)) if np.std(reference) > 1.0e-12 else None,
        "pearson_r": correlation,
        "slope": slope,
    }


def _grouped_oof(
    dataset: AlignedOpticalDataset,
    features: np.ndarray,
    position_id: str,
    components: int,
) -> pd.DataFrame:
    formal = dataset.force_mask & (dataset.fold_id >= 0)
    position = dataset.position_target.astype(str) == str(position_id)
    active = formal & position & (dataset.force_fz_n >= 0.10) & (dataset.force_fz_n <= 5.0)
    predictions: list[pd.DataFrame] = []
    for fold in sorted(set(dataset.fold_id[active].astype(int).tolist())):
        train = active & (dataset.fold_id != int(fold))
        test = formal & position & (dataset.fold_id == int(fold))
        train_groups = set(dataset.group_id[train].astype(str).tolist())
        test_groups = set(dataset.group_id[test].astype(str).tolist())
        if train_groups.intersection(test_groups):
            raise RuntimeError("causal spectral benchmark detected group leakage")
        count = min(int(components), features.shape[1], int(np.sum(train)) - 1)
        model = PLSRegression(
            n_components=max(1, count),
            scale=True,
            max_iter=1000,
            tol=1.0e-06,
        )
        model.fit(features[train], dataset.force_fz_n[train])
        raw = np.clip(np.asarray(model.predict(features[test])).reshape(-1), 0.0, 5.0)
        predictions.append(
            pd.DataFrame(
                {
                    "group_id": dataset.group_id[test].astype(str),
                    "sample_index": dataset.sample_index[test].astype(int),
                    "fold_id": int(fold),
                    "true_force_n": dataset.force_fz_n[test].astype(float),
                    "raw_force_n": raw,
                }
            )
        )
    if not predictions:
        raise ValueError(f"no grouped predictions produced for {position_id}")
    return pd.concat(predictions, ignore_index=True)


def _score_predictions(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    active = predictions["true_force_n"].to_numpy(dtype=float) >= 0.10
    aggregate = _safe_metrics(
        predictions.loc[active, "true_force_n"],
        predictions.loc[active, "raw_force_n"],
    )
    zero = predictions["true_force_n"].to_numpy(dtype=float) <= 0.03
    sessions: list[dict[str, Any]] = []
    for group_id, group in predictions.groupby("group_id", sort=False):
        group_active = group["true_force_n"].to_numpy(dtype=float) >= 0.10
        metrics = _safe_metrics(
            group.loc[group_active, "true_force_n"],
            group.loc[group_active, "raw_force_n"],
        )
        sessions.append(
            {
                "group_id": str(group_id),
                "active_sample_count": int(np.sum(group_active)),
                **metrics,
            }
        )
    session_frame = pd.DataFrame(sessions)
    trial008 = session_frame[
        session_frame["group_id"].str.contains("trial_008", regex=False)
    ]
    aggregate.update(
        {
            "zero_force_false_response_rate_raw": (
                float(np.mean(predictions.loc[zero, "raw_force_n"] > 0.10))
                if np.any(zero)
                else None
            ),
            "worst_session_mae_n": float(session_frame["mae_n"].max()),
            "trial008_mae_n": (
                float(trial008.iloc[0]["mae_n"]) if not trial008.empty else None
            ),
            "trial008_r2": float(trial008.iloc[0]["r2"]) if not trial008.empty else None,
            "trial008_pearson_r": (
                float(trial008.iloc[0]["pearson_r"]) if not trial008.empty else None
            ),
            "trial008_slope": (
                float(trial008.iloc[0]["slope"]) if not trial008.empty else None
            ),
        }
    )
    return aggregate, session_frame


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
    causal_views = build_causal_spectral_views(dataset)
    standard_views = build_feature_views(dataset)
    views = dict(causal_views)
    views.update(
        {
            "peak_current_40": standard_views["peak_current_40"],
            "peak_temporal_483": standard_views["peak_temporal_483"],
            "peak_temporal_plus_current264": np.concatenate(
                (standard_views["peak_temporal_483"], causal_views["current264"]),
                axis=1,
            ),
            "peak_current_plus_lag_delta1_3_8": np.concatenate(
                (
                    standard_views["peak_current_40"],
                    causal_views["current_plus_lag_delta1_3_8"],
                ),
                axis=1,
            ),
            "peak_temporal_plus_lag_delta1_3_8": np.concatenate(
                (
                    standard_views["peak_temporal_483"],
                    causal_views["current_plus_lag_delta1_3_8"],
                ),
                axis=1,
            ),
        }
    )
    views.update(
        build_baseline_conditioned_spectral_views(
            dataset, args.raw_data_root.resolve()
        )
    )
    components_grid: Iterable[int] = (8, 12, 16, 24)
    summary_rows: list[dict[str, Any]] = []
    session_parts: list[pd.DataFrame] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for view_id, features in views.items():
        for components in components_grid:
            predictions = _grouped_oof(
                dataset, features, str(args.position), int(components)
            )
            metrics, sessions = _score_predictions(predictions)
            model_id = f"{view_id}_pls{components}"
            predictions_by_model[model_id] = predictions
            summary_rows.append(
                {
                    "model_id": model_id,
                    "position_id": str(args.position),
                    "feature_count": int(features.shape[1]),
                    "latent_components": int(components),
                    "evaluation_validity": "grouped_oof_by_session_model_discovery",
                    "force_sensor_used_as_runtime_input": False,
                    **metrics,
                }
            )
            sessions.insert(0, "model_id", model_id)
            session_parts.append(sessions)

    summary = pd.DataFrame(summary_rows)
    summary["selection_score"] = (
        summary["mae_n"]
        + 0.35 * summary["worst_session_mae_n"]
        + 0.20 * (1.0 - summary["pearson_r"].clip(0.0, 1.0))
        + 0.20 * (summary["slope"] - 1.0).abs()
    )
    summary = summary.sort_values("selection_score", kind="stable")
    sessions = pd.concat(session_parts, ignore_index=True)
    summary.to_csv(output / "causal_spectral_model_comparison.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(output / "causal_spectral_session_metrics.csv", index=False, encoding="utf-8-sig")
    best = _json_safe(summary.iloc[0].to_dict())
    best_model_id = str(summary.iloc[0]["model_id"])
    best_predictions = predictions_by_model[best_model_id].copy()
    best_predictions.insert(0, "model_id", best_model_id)
    best_predictions.insert(1, "position_id", str(args.position))
    best_predictions["evaluation_validity"] = (
        "grouped_oof_by_session_model_discovery"
    )
    best_predictions["force_sensor_used_as_runtime_input"] = False
    best_predictions.to_csv(
        output / "causal_spectral_best_grouped_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "causal_spectral_best_candidate.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(summary.head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
