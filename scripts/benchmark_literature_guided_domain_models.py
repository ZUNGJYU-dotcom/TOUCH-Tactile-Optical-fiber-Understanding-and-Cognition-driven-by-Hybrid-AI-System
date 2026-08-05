#!/usr/bin/env python3
"""Benchmark literature-guided FBG models under strict cross-date holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.literature_guided_domain_models import (  # noqa: E402
    POSITION_ORDER,
    FeatureView,
    apply_affine_calibration,
    build_literature_feature_views,
    classification_metrics,
    equal_group_weights,
    fit_predict_osc_ridge,
    fit_predict_pls,
    fit_predict_ridge,
    group_voting_metrics,
    learn_group_oof_blend_weight,
    load_strict_cross_date_datasets,
    make_extra_trees_classifier,
    make_extra_trees_regressor,
    nonnegative_prediction,
    regression_metrics,
)


DEFAULT_DATE_DATASETS = {
    "20260803": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260803_new_data_only"
    / "ordinary_fbg_px6d_dataset.npz",
    "20260804": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260804_new_batch_only"
    / "ordinary_fbg_px6d_dataset.npz",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_literature_guided_cross_date_20260804"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date-dataset",
        action="append",
        default=[],
        metavar="DATE=PATH",
        help="Strict per-date NPZ; repeat for two or more dates.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--estimators", type=int, default=128)
    parser.add_argument("--force-estimators", type=int, default=160)
    parser.add_argument("--minimum-leaf-samples", type=int, default=2)
    parser.add_argument("--temporal-window-frames", type=int, default=5)
    parser.add_argument("--pls-components", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--osc-components", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_date_datasets(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_DATE_DATASETS)
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATE=PATH, received {value!r}")
        date, raw_path = value.split("=", 1)
        result[date.strip()] = Path(raw_path.strip()).expanduser().resolve()
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def latency_ms_per_row(model: Any, features: np.ndarray) -> float:
    subset = features[: min(1024, len(features))]
    if not len(subset):
        return float("nan")
    model.predict(subset[: min(8, len(subset))])
    timings: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        model.predict(subset)
        timings.append((time.perf_counter() - start) * 1000.0 / len(subset))
    return float(np.median(timings))


def common_row(
    *,
    task: str,
    candidate_id: str,
    feature_view: str,
    model_family: str,
    test_date: str,
    train: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    fit_seconds: float,
    latency: float,
) -> dict[str, Any]:
    return {
        "task": task,
        "candidate_id": candidate_id,
        "feature_view": feature_view,
        "model_family": model_family,
        "test_date": str(test_date),
        "evaluation_validity": "leave_one_date_out_grouped_by_session",
        "train_rows": int(np.sum(train)),
        "test_rows": int(np.sum(test)),
        "train_groups": int(len(np.unique(groups[train]))),
        "test_groups": int(len(np.unique(groups[test]))),
        "fit_seconds": float(fit_seconds),
        "inference_latency_ms_per_row": float(latency),
    }


def train_classifier(
    view: FeatureView,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    *,
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[Any, np.ndarray, float, float]:
    model = make_extra_trees_classifier(
        estimators=estimators,
        minimum_leaf_samples=minimum_leaf_samples,
        seed=seed,
    )
    start = time.perf_counter()
    model.fit(
        view.values[train],
        target[train],
        sample_weight=equal_group_weights(groups[train]),
    )
    fit_seconds = time.perf_counter() - start
    predicted = model.predict(view.values[test])
    latency = latency_ms_per_row(model, view.values[test])
    return model, predicted, fit_seconds, latency


def train_force_tree(
    view: FeatureView,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    *,
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[Any, np.ndarray, float, float]:
    model = make_extra_trees_regressor(
        estimators=estimators,
        minimum_leaf_samples=minimum_leaf_samples,
        seed=seed,
    )
    start = time.perf_counter()
    model.fit(
        view.values[train],
        target[train],
        sample_weight=equal_group_weights(groups[train]),
    )
    fit_seconds = time.perf_counter() - start
    predicted = nonnegative_prediction(model.predict(view.values[test]))
    latency = latency_ms_per_row(model, view.values[test])
    return model, predicted, fit_seconds, latency


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    ignored = {
        "test_date",
        "evaluation_validity",
        "feature_view",
        "model_family",
    }
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in ignored
    ]
    rows: list[dict[str, Any]] = []
    for (task, candidate), selected in frame.groupby(["task", "candidate_id"]):
        row: dict[str, Any] = {
            "task": task,
            "candidate_id": candidate,
            "feature_view": selected["feature_view"].iloc[0],
            "model_family": selected["model_family"].iloc[0],
            "date_holdouts": int(len(selected)),
        }
        for column in numeric:
            values = pd.to_numeric(selected[column], errors="coerce")
            if values.notna().any():
                row[f"mean_{column}"] = float(values.mean())
                lower_is_better = any(
                    token in column
                    for token in (
                        "mae",
                        "rmse",
                        "false_response",
                        "latency",
                        "fit_seconds",
                    )
                )
                row[f"worst_{column}"] = float(
                    values.max() if lower_is_better else values.min()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def best_candidate(aggregate: pd.DataFrame, task: str) -> pd.Series:
    selected = aggregate[aggregate["task"] == task].copy()
    if task == "force":
        guarded = selected[
            selected["mean_slope"].between(0.80, 1.20)
            & (selected["mean_r2"] >= selected["mean_r2"].max() - 0.08)
        ]
        if not guarded.empty:
            selected = guarded
        return selected.sort_values(
            ["mean_mae_n", "worst_mae_n", "mean_low_force_false_response_mean_n"]
        ).iloc[0]
    if task == "position":
        # Treat mean macro-F1 values within one percentage point as tied, then
        # prefer the model that survives the harder acquisition date.  This
        # avoids selecting a brittle model for a negligible mean-score gain.
        near_best = selected[
            selected["mean_macro_f1"] >= selected["mean_macro_f1"].max() - 0.01
        ]
        return near_best.sort_values(
            [
                "worst_macro_f1",
                "mean_group_voting_accuracy",
                "mean_macro_f1",
            ],
            ascending=[False, False, False],
        ).iloc[0]
    return selected.sort_values(
        ["mean_macro_f1", "worst_macro_f1"], ascending=[False, False]
    ).iloc[0]


def plot_force_comparison(aggregate: pd.DataFrame, path: Path) -> None:
    selected = aggregate[aggregate["task"] == "force"].sort_values("mean_mae_n")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = ["#0C91C4" if "literature" in value else "#8CA4B5" for value in selected["candidate_id"]]
    ax.barh(selected["candidate_id"], selected["mean_mae_n"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Leave-one-date-out MAE (N), lower is better")
    ax.set_title("Literature-guided optical force candidates")
    ax.grid(axis="x", color="#DDE7EE", linewidth=0.8)
    for index, value in enumerate(selected["mean_mae_n"]):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def plot_classification_comparison(aggregate: pd.DataFrame, path: Path) -> None:
    selected = aggregate[aggregate["task"].isin(["contact", "position"])].copy()
    candidates = sorted(selected["candidate_id"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharex=False)
    for axis, task, title in zip(
        axes,
        ("contact", "position"),
        ("Contact detection", "Contact position"),
    ):
        task_rows = selected[selected["task"] == task].sort_values("mean_macro_f1")
        axis.barh(task_rows["candidate_id"], task_rows["mean_macro_f1"], color="#48A9A6")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Mean macro-F1")
        axis.set_title(title)
        axis.grid(axis="x", color="#DDE7EE", linewidth=0.8)
    fig.suptitle("Strict leave-one-date-out classification")
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def plot_force_parity(predictions: pd.DataFrame, path: Path, candidate: str) -> None:
    selected = predictions[predictions["candidate_id"] == candidate]
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    for date, rows in selected.groupby("test_date"):
        stride = max(1, len(rows) // 5000)
        sampled = rows.iloc[::stride]
        ax.scatter(
            sampled["force_true_n"],
            sampled["force_predicted_n"],
            s=10,
            alpha=0.28,
            label=str(date),
        )
    maximum = float(
        max(selected["force_true_n"].max(), selected["force_predicted_n"].max())
    )
    ax.plot([0.0, maximum], [0.0, maximum], "--", color="#596C7A", linewidth=1.4)
    ax.set_xlim(0.0, maximum * 1.02)
    ax.set_ylim(0.0, maximum * 1.02)
    ax.set_xlabel("PX6D Fz reference (N)")
    ax.set_ylabel("Optical estimate (N)")
    ax.set_title(f"Best cross-date candidate: {candidate}")
    ax.legend(title="Held-out date")
    ax.grid(color="#DDE7EE", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def write_report(
    *,
    output_dir: Path,
    aggregate: pd.DataFrame,
    detailed: pd.DataFrame,
    date_datasets: dict[str, Path],
    best: dict[str, pd.Series],
    dataset_rows: int,
    dataset_groups: int,
    maximum_force: float,
) -> None:
    force_reference = aggregate[
        aggregate["candidate_id"] == "force_reference_full264_extra_trees"
    ].iloc[0]
    contact_reference = aggregate[
        aggregate["candidate_id"] == "contact_reference_full264_extra_trees"
    ].iloc[0]
    position_reference = aggregate[
        aggregate["candidate_id"] == "position_reference_full264_extra_trees"
    ].iloc[0]
    force_best = best["force"]
    force_improvement = (
        float(force_reference["mean_mae_n"]) - float(force_best["mean_mae_n"])
    ) / float(force_reference["mean_mae_n"])
    contact_gain = float(best["contact"]["mean_macro_f1"]) - float(
        contact_reference["mean_macro_f1"]
    )
    position_gain = float(best["position"]["mean_macro_f1"]) - float(
        position_reference["mean_macro_f1"]
    )
    force_candidate_supported = (
        force_improvement > 0.02
        and float(force_best["mean_r2"]) > float(force_reference["mean_r2"])
        and float(force_best["worst_mae_n"])
        <= 1.03 * float(force_reference["worst_mae_n"])
        and 0.75 <= float(force_best["mean_slope"]) <= 1.20
    )
    force_recommendation = (
        "candidate_for_third_date_blind_validation"
        if force_candidate_supported
        else "retain_reference_until_more_dates_are_available"
    )
    overall_recommendation = (
        "classification_candidates_ready_for_third_date_validation_force_not_replaced"
        if contact_gain > 0.0 or position_gain > 0.0
        else force_recommendation
    )

    lines = [
        "# Literature-guided cross-date algorithm report",
        "",
        "## Scope",
        "",
        "This is an offline candidate benchmark for ordinary-FBG BaySpec spectra. "
        "It does not modify the current Stable/Beta runtime and does not deploy a model.",
        "",
        f"- Rows: {dataset_rows:,}",
        f"- Independent sessions: {dataset_groups}",
        f"- Acquisition dates: {', '.join(sorted(date_datasets))}",
        f"- Observed Fz range: 0 to {maximum_force:.3f} N",
        "- Formal split: leave one complete acquisition date out; sessions never cross train/test.",
        "- Force prediction: non-negative only; no artificial 5 N upper clipping.",
        "",
        "## Literature-derived changes",
        "",
        "Savitzky-Golay smoothing/derivatives preserve local spectral structure while extracting "
        "shape change. Standard normal variate and robust median/MAD views reduce multiplicative "
        "scale and baseline-domain variation. A causal five-frame summary adds short-term optical "
        "dynamics without using future frames. Ridge, PLS and train-only OSC are evaluated beside "
        "ExtraTrees; OSC is not assumed beneficial and must win the held-out-date comparison. "
        "The cascade candidate follows the location-first, force-second structure reported for "
        "AI optical-fiber tactile sensing.",
        "",
        "PDS/calibration transfer and CORAL target-domain alignment were not used in the formal "
        "ranking: PDS needs paired cross-date standards, while target-statistic CORAL would make "
        "the held-out-date result transductive rather than a clean deployment simulation.",
        "",
        "Primary sources: Savitzky-Golay (Analytical Chemistry, 1964), SNV (Applied "
        "Spectroscopy, 1989), PDS calibration transfer (Analytical Chemistry, 1991), OSC "
        "(Chemometrics and Intelligent Laboratory Systems, 1998), Deep CORAL (ECCV 2016), "
        "and Li et al. optical tactile cascade (Advanced Intelligent Systems, 2023).",
        "",
        "- https://pubs.acs.org/doi/10.1021/ac60214a047",
        "- https://opg.optica.org/abstract.cfm?uri=as-43-5-772",
        "- https://pubs.acs.org/doi/10.1021/ac00023a016",
        "- https://doi.org/10.1016/S0169-7439(98)00109-9",
        "- https://arxiv.org/abs/1607.01719",
        "- https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202200460",
        "",
        "## Best candidates",
        "",
        f"- Contact: `{best['contact']['candidate_id']}`, mean macro-F1 "
        f"{best['contact']['mean_macro_f1']:.4f}.",
        f"- Position: `{best['position']['candidate_id']}`, mean macro-F1 "
        f"{best['position']['mean_macro_f1']:.4f}.",
        f"- Contact macro-F1 gain over strict full264 ExtraTrees: {contact_gain:+.4f}.",
        f"- Position macro-F1 gain over strict full264 ExtraTrees: {position_gain:+.4f}.",
        f"- Force: `{force_best['candidate_id']}`, mean MAE "
        f"{force_best['mean_mae_n']:.4f} N, mean R2 {force_best['mean_r2']:.4f}, "
        f"mean low-force false response {force_best['mean_low_force_false_response_mean_n']:.4f} N.",
        f"- Force MAE change from strict 264-feature ExtraTrees reference: {force_improvement:+.1%}.",
        f"- Force recommendation: `{force_recommendation}`.",
        f"- Overall recommendation: `{overall_recommendation}`.",
        "",
        "## Interpretation",
        "",
        "The ranking is evidence for cross-date robustness, not a final generalization claim: "
        "only two acquisition dates are available. A candidate should enter Beta only after a "
        "third-date blind replay and live latency/residual checks. The retained group/date split "
        "is more important than a higher random-window score.",
        "",
        "## Artifacts",
        "",
        "- `literature_guided_detailed_metrics.csv`: every held-out-date result",
        "- `literature_guided_aggregate_metrics.csv`: mean and worst-date leaderboard",
        "- `best_force_cross_date_predictions.csv`: held-out predictions for the best force candidate",
        "- `all_force_cross_date_predictions.csv`: held-out predictions for every force candidate",
        "- `force_cross_date_mae.png`: force comparison",
        "- `classification_cross_date_macro_f1.png`: contact/position comparison",
        "- `best_force_cross_date_parity.png`: unclipped force parity",
    ]
    (output_dir / "literature_guided_algorithm_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    summary = {
        "status": overall_recommendation,
        "force_recommendation": force_recommendation,
        "formal_split": "leave_one_date_out_grouped_by_session",
        "force_upper_clip_applied": False,
        "dates": sorted(date_datasets),
        "dataset_rows": dataset_rows,
        "independent_sessions": dataset_groups,
        "maximum_observed_force_n": maximum_force,
        "best_contact": best["contact"].to_dict(),
        "best_position": best["position"].to_dict(),
        "best_force": force_best.to_dict(),
        "force_reference": force_reference.to_dict(),
        "contact_macro_f1_absolute_gain": contact_gain,
        "position_macro_f1_absolute_gain": position_gain,
        "force_mae_relative_improvement": force_improvement,
        "source_datasets": {key: str(value) for key, value in date_datasets.items()},
    }
    (output_dir / "literature_guided_summary.json").write_text(
        json.dumps(
            json_ready(summary),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    date_datasets = parse_date_datasets(args.date_dataset)
    missing = [str(path) for path in date_datasets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dataset(s): " + ", ".join(missing))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading strict cross-date spectra...", flush=True)
    dataset = load_strict_cross_date_datasets(date_datasets)
    print(
        f"Loaded {len(dataset.features):,} rows, "
        f"{len(np.unique(dataset.group_id))} sessions, "
        f"Fz max {np.nanmax(dataset.force_fz_n[dataset.force_mask]):.3f} N",
        flush=True,
    )
    views = build_literature_feature_views(
        dataset,
        temporal_window_frames=args.temporal_window_frames,
    )
    print(
        "Feature views: "
        + ", ".join(f"{key}={view.values.shape[1]}" for key, view in views.items()),
        flush=True,
    )

    dates = sorted(np.unique(dataset.acquisition_date).tolist())
    metric_rows: list[dict[str, Any]] = []
    force_prediction_rows: list[dict[str, Any]] = []

    classification_views = (
        "reference_full264",
        "response_raw136",
        "literature_snv_sg328",
        "literature_snv_sg_temporal488",
    )

    for date_index, test_date in enumerate(dates):
        print(f"\nHeld-out date: {test_date}", flush=True)
        train_date = dataset.acquisition_date != test_date
        test_date_mask = dataset.acquisition_date == test_date
        if set(dataset.group_id[train_date]).intersection(dataset.group_id[test_date_mask]):
            raise RuntimeError("session leakage across acquisition dates")

        fitted_contact: dict[str, Any] = {}
        fitted_position: dict[str, Any] = {}
        for task, target, task_mask, labels in (
            ("contact", dataset.contact_target, dataset.contact_mask, [0, 1]),
            ("position", dataset.position_target, dataset.position_mask, list(POSITION_ORDER)),
        ):
            train = train_date & task_mask
            test = test_date_mask & task_mask
            for view_index, view_name in enumerate(classification_views):
                view = views[view_name]
                candidate = f"{task}_{view_name}_extra_trees"
                print(f"  {candidate}", flush=True)
                model, predicted, fit_seconds, latency = train_classifier(
                    view,
                    target,
                    train,
                    test,
                    dataset.group_id,
                    estimators=args.estimators,
                    minimum_leaf_samples=args.minimum_leaf_samples,
                    seed=args.seed + date_index * 50 + view_index,
                )
                row = common_row(
                    task=task,
                    candidate_id=candidate,
                    feature_view=view_name,
                    model_family="extra_trees",
                    test_date=test_date,
                    train=train,
                    test=test,
                    groups=dataset.group_id,
                    fit_seconds=fit_seconds,
                    latency=latency,
                )
                row.update(
                    classification_metrics(target[test], predicted, labels=labels)
                )
                if task == "contact":
                    row["no_contact_false_positive_rate"] = 1.0 - row["recall_0"]
                    fitted_contact[view_name] = model
                else:
                    row.update(
                        group_voting_metrics(
                            target[test],
                            predicted,
                            dataset.group_id[test],
                            labels=list(POSITION_ORDER),
                        )
                    )
                    fitted_position[view_name] = model
                metric_rows.append(row)

            if task == "position":
                # The raw response view retains absolute inter-peak evidence,
                # while SNV/SG is more robust to date-wise scale drift.  Equal
                # probability averaging is intentionally parameter-free and
                # therefore cannot overfit the held-out acquisition date.
                raw_model = fitted_position["response_raw136"]
                normalized_model = fitted_position["literature_snv_sg328"]
                raw_probability = raw_model.predict_proba(
                    views["response_raw136"].values[test]
                )
                normalized_probability = normalized_model.predict_proba(
                    views["literature_snv_sg328"].values[test]
                )
                if not np.array_equal(raw_model.classes_, normalized_model.classes_):
                    raise RuntimeError("position ensemble class order mismatch")
                ensemble_probability = 0.5 * (
                    raw_probability + normalized_probability
                )
                ensemble_prediction = raw_model.classes_[
                    np.argmax(ensemble_probability, axis=1)
                ]
                ensemble_candidate = "position_literature_probability_ensemble"
                ensemble_row = common_row(
                    task="position",
                    candidate_id=ensemble_candidate,
                    feature_view="response_raw136+literature_snv_sg328",
                    model_family="equal_probability_ensemble",
                    test_date=test_date,
                    train=train,
                    test=test,
                    groups=dataset.group_id,
                    fit_seconds=float("nan"),
                    latency=latency_ms_per_row(
                        raw_model, views["response_raw136"].values[test]
                    )
                    + latency_ms_per_row(
                        normalized_model,
                        views["literature_snv_sg328"].values[test],
                    ),
                )
                ensemble_row.update(
                    classification_metrics(
                        target[test], ensemble_prediction, labels=labels
                    )
                )
                ensemble_row.update(
                    group_voting_metrics(
                        target[test],
                        ensemble_prediction,
                        dataset.group_id[test],
                        labels=list(POSITION_ORDER),
                    )
                )
                metric_rows.append(ensemble_row)

        force_train = train_date & dataset.force_mask
        force_test = test_date_mask & dataset.force_mask
        y_train = dataset.force_fz_n[force_train]
        y_test = dataset.force_fz_n[force_test]
        force_groups = dataset.group_id[force_train]

        force_models: dict[str, Any] = {}
        force_predictions: dict[str, np.ndarray] = {}
        force_fits: dict[str, float] = {}
        force_latencies: dict[str, float] = {}

        for offset, view_name in enumerate(
            ("reference_full264", "literature_snv_sg_temporal488")
        ):
            candidate = (
                "force_reference_full264_extra_trees"
                if view_name == "reference_full264"
                else "force_literature_temporal_extra_trees"
            )
            print(f"  {candidate}", flush=True)
            model, predicted, fit_seconds, latency = train_force_tree(
                views[view_name],
                dataset.force_fz_n,
                force_train,
                force_test,
                dataset.group_id,
                estimators=args.force_estimators,
                minimum_leaf_samples=args.minimum_leaf_samples,
                seed=args.seed + date_index * 50 + 20 + offset,
            )
            force_models[candidate] = model
            force_predictions[candidate] = predicted
            force_fits[candidate] = fit_seconds
            force_latencies[candidate] = latency

        pls_candidate = "force_literature_snv_sg_pls"
        print(f"  {pls_candidate}", flush=True)
        start = time.perf_counter()
        pls_model, pls_prediction = fit_predict_pls(
            views["literature_snv_sg328"].values[force_train],
            y_train,
            force_groups,
            views["literature_snv_sg328"].values[force_test],
            components=args.pls_components,
        )
        force_models[pls_candidate] = pls_model
        force_predictions[pls_candidate] = pls_prediction
        force_fits[pls_candidate] = time.perf_counter() - start
        force_latencies[pls_candidate] = latency_ms_per_row(
            pls_model,
            views["literature_snv_sg328"].values[force_test],
        )

        ridge_candidate = "force_literature_snv_sg_ridge"
        print(f"  {ridge_candidate}", flush=True)
        start = time.perf_counter()
        ridge_model, ridge_prediction = fit_predict_ridge(
            views["literature_snv_sg328"].values[force_train],
            y_train,
            force_groups,
            views["literature_snv_sg328"].values[force_test],
            alpha=args.ridge_alpha,
        )
        force_models[ridge_candidate] = ridge_model
        force_predictions[ridge_candidate] = ridge_prediction
        force_fits[ridge_candidate] = time.perf_counter() - start
        force_latencies[ridge_candidate] = latency_ms_per_row(
            ridge_model,
            views["literature_snv_sg328"].values[force_test],
        )

        osc_candidate = "force_literature_train_only_osc_ridge"
        print(f"  {osc_candidate}", flush=True)
        start = time.perf_counter()
        osc_model, osc_prediction = fit_predict_osc_ridge(
            views["literature_snv_sg328"].values[force_train],
            y_train,
            force_groups,
            views["literature_snv_sg328"].values[force_test],
            alpha=args.ridge_alpha,
            osc_components=args.osc_components,
        )
        force_models[osc_candidate] = osc_model
        force_predictions[osc_candidate] = osc_prediction
        force_fits[osc_candidate] = time.perf_counter() - start
        force_latencies[osc_candidate] = latency_ms_per_row(
            osc_model,
            views["literature_snv_sg328"].values[force_test],
        )

        blend_candidate = "force_literature_group_oof_tree_pls_blend"
        print(f"  {blend_candidate} (inner grouped OOF)", flush=True)
        start = time.perf_counter()
        blend = learn_group_oof_blend_weight(
            tree_features=views["literature_snv_sg_temporal488"].values[force_train],
            latent_features=views["literature_snv_sg328"].values[force_train],
            target=y_train,
            groups=force_groups,
            estimators=max(48, args.force_estimators // 2),
            minimum_leaf_samples=args.minimum_leaf_samples,
            pls_components=args.pls_components,
            seed=args.seed + date_index * 50 + 30,
        )
        tree_candidate = "force_literature_temporal_extra_trees"
        blend_prediction = nonnegative_prediction(
            blend["tree_weight"] * force_predictions[tree_candidate]
            + blend["latent_weight"] * force_predictions[pls_candidate]
        )
        force_predictions[blend_candidate] = blend_prediction
        force_fits[blend_candidate] = (
            force_fits[tree_candidate]
            + force_fits[pls_candidate]
            + time.perf_counter()
            - start
        )
        force_latencies[blend_candidate] = (
            force_latencies[tree_candidate] + force_latencies[pls_candidate]
        )

        calibrated_tree_candidate = "force_literature_oof_affine_calibrated_tree"
        force_predictions[calibrated_tree_candidate] = apply_affine_calibration(
            force_predictions[tree_candidate],
            slope=blend["tree_calibration_slope"],
            intercept_n=blend["tree_calibration_intercept_n"],
        )
        force_fits[calibrated_tree_candidate] = force_fits[tree_candidate]
        force_latencies[calibrated_tree_candidate] = force_latencies[tree_candidate]

        calibrated_blend_candidate = "force_literature_oof_affine_calibrated_blend"
        force_predictions[calibrated_blend_candidate] = apply_affine_calibration(
            blend_prediction,
            slope=blend["blend_calibration_slope"],
            intercept_n=blend["blend_calibration_intercept_n"],
        )
        force_fits[calibrated_blend_candidate] = force_fits[blend_candidate]
        force_latencies[calibrated_blend_candidate] = force_latencies[blend_candidate]

        cascade_candidate = "force_literature_position_cascade_soft_gate"
        print(f"  {cascade_candidate}", flush=True)
        cascade_start = time.perf_counter()
        cascade_prediction = force_predictions[calibrated_blend_candidate].copy()
        cascade_view = views["literature_snv_sg_temporal488"].values
        contact_model = fitted_contact["literature_snv_sg_temporal488"]
        position_model = fitted_position["literature_snv_sg_temporal488"]
        active_class_index = int(np.flatnonzero(contact_model.classes_ == 1)[0])
        active_probability = contact_model.predict_proba(cascade_view[force_test])[
            :, active_class_index
        ]
        predicted_position = position_model.predict(cascade_view[force_test])
        source_position_train = dataset.position_target[force_train]
        for position_index, position in enumerate(POSITION_ORDER):
            local_train = force_train & (dataset.position_target == position)
            local_test_in_force = predicted_position == position
            if np.sum(local_train) < 50 or not np.any(local_test_in_force):
                continue
            local_model = make_extra_trees_regressor(
                estimators=max(48, args.force_estimators),
                minimum_leaf_samples=args.minimum_leaf_samples,
                seed=args.seed + date_index * 100 + 100 + position_index,
            )
            local_model.fit(
                cascade_view[local_train],
                dataset.force_fz_n[local_train],
                sample_weight=equal_group_weights(dataset.group_id[local_train]),
            )
            cascade_prediction[local_test_in_force] = nonnegative_prediction(
                local_model.predict(cascade_view[force_test][local_test_in_force])
            )
        force_predictions["force_literature_position_cascade_no_gate"] = (
            nonnegative_prediction(cascade_prediction)
        )
        force_fits["force_literature_position_cascade_no_gate"] = (
            time.perf_counter() - cascade_start
        )
        force_latencies["force_literature_position_cascade_no_gate"] = float("nan")

        # Suppress only the low-force residual region. High-force estimates are
        # left untouched so contact gating cannot flatten the calibration slope.
        residual_gate = 0.25 + 0.75 * active_probability
        low_region = cascade_prediction < 1.0
        gated_prediction = cascade_prediction.copy()
        gated_prediction[low_region] *= residual_gate[low_region]
        force_predictions[cascade_candidate] = nonnegative_prediction(gated_prediction)
        force_fits[cascade_candidate] = time.perf_counter() - cascade_start
        force_latencies[cascade_candidate] = float("nan")

        force_specs = {
            "force_reference_full264_extra_trees": (
                "reference_full264",
                "extra_trees",
                {},
            ),
            "force_literature_temporal_extra_trees": (
                "literature_snv_sg_temporal488",
                "extra_trees",
                {},
            ),
            "force_literature_snv_sg_pls": (
                "literature_snv_sg328",
                "pls_regression",
                {},
            ),
            "force_literature_snv_sg_ridge": (
                "literature_snv_sg328",
                "ridge_regression",
                {"ridge_alpha": args.ridge_alpha},
            ),
            "force_literature_train_only_osc_ridge": (
                "literature_snv_sg328",
                "train_only_osc_ridge",
                {
                    "ridge_alpha": args.ridge_alpha,
                    "osc_components": args.osc_components,
                },
            ),
            "force_literature_group_oof_tree_pls_blend": (
                "literature_snv_sg_temporal488+literature_snv_sg328",
                "group_oof_convex_blend",
                blend,
            ),
            "force_literature_oof_affine_calibrated_tree": (
                "literature_snv_sg_temporal488",
                "group_oof_affine_calibrated_extra_trees",
                {
                    "calibration_slope": blend["tree_calibration_slope"],
                    "calibration_intercept_n": blend["tree_calibration_intercept_n"],
                },
            ),
            "force_literature_oof_affine_calibrated_blend": (
                "literature_snv_sg_temporal488+literature_snv_sg328",
                "group_oof_affine_calibrated_blend",
                {
                    "calibration_slope": blend["blend_calibration_slope"],
                    "calibration_intercept_n": blend["blend_calibration_intercept_n"],
                },
            ),
            "force_literature_position_cascade_no_gate": (
                "literature_snv_sg_temporal488",
                "position_then_force_experts",
                {"contact_gate": "none", "position_experts": 9},
            ),
            "force_literature_position_cascade_soft_gate": (
                "literature_snv_sg_temporal488",
                "contact_position_force_cascade",
                {"contact_gate": "soft_predict_proba", "position_experts": 9},
            ),
        }
        force_test_indices = np.flatnonzero(force_test)
        for candidate, predicted in force_predictions.items():
            view_name, family, extras = force_specs[candidate]
            row = common_row(
                task="force",
                candidate_id=candidate,
                feature_view=view_name,
                model_family=family,
                test_date=test_date,
                train=force_train,
                test=force_test,
                groups=dataset.group_id,
                fit_seconds=force_fits[candidate],
                latency=force_latencies[candidate],
            )
            row["force_upper_clip_applied"] = False
            row.update(regression_metrics(y_test, predicted))
            for key, value in extras.items():
                if isinstance(value, (str, int, float, bool)):
                    row[key] = value
            metric_rows.append(row)
            for local_index, global_index in enumerate(force_test_indices):
                force_prediction_rows.append(
                    {
                        "candidate_id": candidate,
                        "test_date": test_date,
                        "group_id": dataset.group_id[global_index],
                        "sample_index": int(dataset.sample_index[global_index]),
                        "position_label": dataset.position_target[global_index],
                        "force_true_n": float(y_test[local_index]),
                        "force_predicted_n": float(predicted[local_index]),
                    }
                )

    detailed = pd.DataFrame(metric_rows)
    aggregate = aggregate_metrics(detailed)
    predictions = pd.DataFrame(force_prediction_rows)
    detailed.to_csv(output_dir / "literature_guided_detailed_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "literature_guided_aggregate_metrics.csv", index=False)
    (output_dir / "literature_guided_aggregate_metrics.json").write_text(
        json.dumps(
            json_ready(aggregate.to_dict(orient="records")),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    best = {task: best_candidate(aggregate, task) for task in ("contact", "position", "force")}
    best_force_id = str(best["force"]["candidate_id"])
    best_predictions = predictions[predictions["candidate_id"] == best_force_id]
    best_predictions.to_csv(
        output_dir / "best_force_cross_date_predictions.csv", index=False
    )
    predictions.to_csv(
        output_dir / "all_force_cross_date_predictions.csv", index=False
    )
    plot_force_comparison(aggregate, output_dir / "force_cross_date_mae.png")
    plot_classification_comparison(
        aggregate, output_dir / "classification_cross_date_macro_f1.png"
    )
    plot_force_parity(
        predictions,
        output_dir / "best_force_cross_date_parity.png",
        best_force_id,
    )
    write_report(
        output_dir=output_dir,
        aggregate=aggregate,
        detailed=detailed,
        date_datasets=date_datasets,
        best=best,
        dataset_rows=len(dataset.features),
        dataset_groups=len(np.unique(dataset.group_id)),
        maximum_force=float(np.nanmax(dataset.force_fz_n[dataset.force_mask])),
    )

    print("\nBest candidates", flush=True)
    for task, row in best.items():
        if task == "force":
            print(
                f"  {task}: {row['candidate_id']} | MAE={row['mean_mae_n']:.4f} N | "
                f"R2={row['mean_r2']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  {task}: {row['candidate_id']} | macro-F1={row['mean_macro_f1']:.4f}",
                flush=True,
            )
    print(f"Outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
