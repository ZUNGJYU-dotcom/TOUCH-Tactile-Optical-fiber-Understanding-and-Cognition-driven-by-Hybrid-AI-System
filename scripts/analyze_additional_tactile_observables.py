"""Screen additional tactile observables from synchronized optical/PX6D data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.rich_optical_benchmark import (  # noqa: E402
    FeatureView,
    build_rich_feature_views,
    grouped_classification,
)
from src.hybrid_spectrum.rich_optical_features import (  # noqa: E402
    load_rich_feature_cache,
)
from src.hybrid_spectrum.tactile_observability import (  # noqa: E402
    MECHANICAL_TARGET_SPECS,
    derive_force_phase_labels,
    grouped_regression_observability,
    load_aligned_mechanical_targets,
    validate_force_alignment,
)


DEFAULT_FUSION_DATASET = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_all_data_fusion_20260731_v1" / "all_source_fusion_dataset.npz"
)
DEFAULT_SPECTRUM_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260731_latest_primary"
    / "primary"
    / "ordinary_fbg_px6d_dataset.npz"
)
DEFAULT_RICH_CACHE = (
    PROJECT_ROOT / "outputs" / "rich_optical_algorithm_benchmark_20260801" / "rich_optical_feature_cache.npz"
)
DEFAULT_CAPTURE_ROOT = PROJECT_ROOT.parents[1] / "data" / "new data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "additional_tactile_observability_20260801"
REGRESSION_VIEWS = (
    "peak_current_40",
    "full_spectrum_192",
    "rich_plus_full_spectrum_192",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION_DATASET)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM_DATASET)
    parser.add_argument("--rich-cache", type=Path, default=DEFAULT_RICH_CACHE)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimators", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _inventory_rows(best_rows: pd.DataFrame, phase_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    best = {row["target_id"]: row for row in best_rows.to_dict("records")}
    rows: list[dict[str, Any]] = [
        {
            "observable": "contact state",
            "type": "supervised optical inference",
            "current_evidence": "formal grouped-by-session benchmark",
            "readiness": "strong baseline; idle false positives still require gating",
            "new_data_needed": "more release and long idle sessions",
        },
        {
            "observable": "contact position P11-P33",
            "type": "supervised optical inference",
            "current_evidence": "formal grouped-by-session benchmark",
            "readiness": "strong for the latest controlled positions",
            "new_data_needed": "off-grid, broad contact, and multi-contact ground truth",
        },
        {
            "observable": "normal force Fz",
            "type": "PX6D-calibrated regression",
            "current_evidence": "formal grouped-by-session benchmark",
            "readiness": "usable research baseline, not final metrology",
            "new_data_needed": "balanced force plateaus and independent calibration runs",
        },
    ]
    for spec in MECHANICAL_TARGET_SPECS:
        result = best.get(spec.column, {})
        rows.append(
            {
                "observable": spec.display_name,
                "type": "exploratory PX6D regression",
                "current_evidence": (
                    f"grouped R2={result.get('r2', float('nan')):.3f}; "
                    f"status={result.get('observability_status', 'missing')}"
                    if result
                    else "missing"
                ),
                "readiness": "correlation screen only; current presses did not control this axis",
                "new_data_needed": "dedicated signed shear/moment loading protocol",
            }
        )
    rows.extend(
        [
            {
                "observable": "loading / hold / release phase",
                "type": "temporal tactile state",
                "current_evidence": f"provisional Fz-derived grouped macro-F1={phase_metrics.get('macro_f1', float('nan')):.3f}",
                "readiness": "useful research proxy; labels are derived from Fz slope",
                "new_data_needed": "operator-marked onset, hold, release, and recovery intervals",
            },
            {
                "observable": "contact patch center, spread, eccentricity, orientation",
                "type": "distributed optical proxy",
                "current_evidence": "nine-peak spatial response vector is available",
                "readiness": "computable proxy, not geometrically calibrated",
                "new_data_needed": "camera/pressure-film or positioning-stage contact footprints",
            },
            {
                "observable": "impact / tap energy and loading rate",
                "type": "temporal optical proxy",
                "current_evidence": "spectral derivatives and synchronized force slope are available",
                "readiness": "analysis-ready, not yet independently labeled",
                "new_data_needed": "controlled taps at several speeds and masses",
            },
            {
                "observable": "stability, drift, residual, recovery time, hysteresis",
                "type": "sensor quality and tactile dynamics",
                "current_evidence": "continuous no-contact, press, and release traces are available",
                "readiness": "directly measurable per session",
                "new_data_needed": "longer repeated loading/unloading cycles for publication metrics",
            },
            {
                "observable": "slip direction / speed",
                "type": "dynamic tactile classification",
                "current_evidence": "no controlled slip labels in the formal dataset",
                "readiness": "not supported yet",
                "new_data_needed": "controlled x/y slides with speed and Fx/Fy labels",
            },
            {
                "observable": "texture, material, curvature, object shape",
                "type": "semantic tactile recognition",
                "current_evidence": "no independent labels in the formal dataset",
                "readiness": "not supported yet",
                "new_data_needed": "object-stratified repeated contacts and held-out objects",
            },
            {
                "observable": "multiple simultaneous contacts",
                "type": "multi-contact inverse problem",
                "current_evidence": "current labels are single approximate contact positions",
                "readiness": "not identifiable from current labels",
                "new_data_needed": "controlled two-point combinations plus contact geometry truth",
            },
        ]
    )
    return rows


def _plot_regression_summary(best_rows: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    labels = best_rows["display_name"].tolist()
    colors = [
        "#2aa6b8" if status == "strong_correlational_candidate" else
        "#e0ad52" if status == "exploratory_candidate" else "#c56b6b"
        for status in best_rows["observability_status"]
    ]
    axes[0].barh(labels, best_rows["r2"], color=colors)
    axes[0].axvline(0.7, color="#607487", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Grouped out-of-fold R2")
    axes[0].set_title("Optical observability of PX6D targets")
    axes[0].invert_yaxis()
    axes[1].barh(labels, best_rows["skill_over_position_baseline"], color=colors)
    axes[1].axvline(0.0, color="#607487", linewidth=1)
    axes[1].set_xlabel("Skill over position-only baseline")
    axes[1].set_title("Does optical evidence add more than position?")
    axes[1].invert_yaxis()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_aligned_latest_primary(args.fusion_dataset, args.spectrum_dataset)
    cache = load_rich_feature_cache(
        args.rich_cache,
        expected_group_id=dataset.group_id,
        expected_sample_index=dataset.sample_index,
    )
    views = build_rich_feature_views(dataset, cache)
    target_columns = ["force_fz_n", *(spec.column for spec in MECHANICAL_TARGET_SPECS)]
    mechanical = load_aligned_mechanical_targets(
        capture_root=args.capture_root,
        group_id=dataset.group_id,
        sample_index=dataset.sample_index,
        target_columns=target_columns,
    )
    max_fz_difference = validate_force_alignment(
        dataset.force_fz_n, mechanical.values["force_fz_n"]
    )

    leaderboard: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    metrics_payload: dict[str, Any] = {}
    total_runs = len(MECHANICAL_TARGET_SPECS) * len(REGRESSION_VIEWS)
    run_number = 0
    for spec in MECHANICAL_TARGET_SPECS:
        for view_id in REGRESSION_VIEWS:
            run_number += 1
            print(f"[{run_number}/{total_runs}] {spec.column} | {view_id}", flush=True)
            view = views[view_id]
            metrics, predicted = grouped_regression_observability(
                features=view.values,
                feature_names=view.names,
                target=mechanical.values[spec.column],
                mask=dataset.force_mask,
                fold_id=dataset.fold_id,
                group_id=dataset.group_id,
                position_target=dataset.position_target,
                contact_target=dataset.contact_target,
                estimators=args.estimators,
                seed=args.seed,
            )
            run_id = f"{spec.column}__extra_trees__{view_id}"
            metrics_payload[run_id] = metrics
            leaderboard.append(
                {
                    "target_id": spec.column,
                    "display_name": spec.display_name,
                    "unit": spec.unit,
                    "semantics": spec.semantics,
                    "model_id": "extra_trees",
                    "feature_view": view_id,
                    **{key: value for key, value in metrics.items() if not isinstance(value, (list, dict))},
                }
            )
            selected = dataset.force_mask & np.isfinite(mechanical.values[spec.column])
            predictions.append(
                pd.DataFrame(
                    {
                        "run_id": run_id,
                        "target_id": spec.column,
                        "session_id": dataset.group_id[selected],
                        "capture_index": dataset.sample_index[selected],
                        "fold_id": dataset.fold_id[selected],
                        "true_value": mechanical.values[spec.column][selected],
                        "predicted_value": predicted[selected],
                    }
                )
            )

    leaderboard_frame = pd.DataFrame(leaderboard).sort_values(
        ["target_id", "r2", "mae"], ascending=[True, False, True]
    )
    best_rows = leaderboard_frame.groupby("target_id", sort=False).head(1).copy()

    phase_labels, force_slope, phase_valid = derive_force_phase_labels(
        force_fz_n=dataset.force_fz_n,
        elapsed_time_sec=mechanical.elapsed_time_sec,
        group_id=dataset.group_id,
    )
    phase_mask = dataset.force_mask & phase_valid
    phase_view = FeatureView(dataset.peak_features, dataset.peak_feature_names)
    phase_metrics, phase_prediction = grouped_classification(
        model_id="extra_trees",
        feature_view=phase_view,
        target=phase_labels,
        mask=phase_mask,
        fold_id=dataset.fold_id,
        group_id=dataset.group_id,
        labels=["no_contact", "loading", "hold", "release"],
        estimators=args.estimators,
        minimum_leaf_samples=2,
        seed=args.seed,
    )
    phase_metrics["label_source"] = "derived_from_synchronized_Fz_and_dFz_dt"
    metrics_payload["force_phase__extra_trees__peak_temporal_483"] = phase_metrics
    phase_selected = np.flatnonzero(phase_mask)
    pd.DataFrame(
        {
            "session_id": dataset.group_id[phase_selected],
            "capture_index": dataset.sample_index[phase_selected],
            "fold_id": dataset.fold_id[phase_selected],
            "force_fz_n": dataset.force_fz_n[phase_selected],
            "force_slope_n_per_sec": force_slope[phase_selected],
            "true_phase": phase_labels[phase_selected],
            "predicted_phase": phase_prediction[phase_selected],
        }
    ).to_csv(args.output_dir / "force_phase_grouped_predictions.csv", index=False)

    inventory = pd.DataFrame(_inventory_rows(best_rows, phase_metrics))
    leaderboard_frame.to_csv(args.output_dir / "wrench_observability_leaderboard.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "grouped_wrench_predictions.csv", index=False
    )
    inventory.to_csv(args.output_dir / "tactile_observable_inventory.csv", index=False)
    (args.output_dir / "additional_tactile_observability_metrics.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "frame_count": int(len(dataset.group_id)),
                    "session_count": int(len(set(dataset.group_id.tolist()))),
                    "split": "immutable grouped-by-session_id folds",
                    "max_fz_alignment_difference_n": max_fz_difference,
                },
                "mechanical_targets": metrics_payload,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    _plot_regression_summary(best_rows, args.output_dir / "wrench_observability_comparison.png")

    lines = [
        "# Additional tactile observability report",
        "",
        "## Scope and validity",
        "",
        f"- Latest-primary frames: {len(dataset.group_id):,}.",
        f"- Independent capture sessions: {len(set(dataset.group_id.tolist()))}.",
        "- Every score uses the frozen grouped-by-session folds; no random frame split is used.",
        f"- Raw PX6D alignment check: maximum Fz difference = {max_fz_difference:.3e} N.",
        "- Fx/Fy and moment channels were not deliberately controlled during these presses. Results below are correlational observability screens, not calibration claims.",
        "",
        "## Mechanical targets beyond Fz",
        "",
        "| Target | Best optical view | R2 | MAE | Skill over position baseline | Within-session variation | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in best_rows.to_dict("records"):
        lines.append(
            f"| {row['display_name']} | {row['feature_view']} | {row['r2']:.3f} | "
            f"{row['mae']:.4g} {row['unit']} | {row['skill_over_position_baseline']:.3f} | "
            f"{row['within_session_to_global_std_ratio']:.3f} | {row['observability_status']} |"
        )
    lines.extend(
        [
            "",
            "A high R2 alone is not enough. The position-only baseline check asks whether the model sees real optical variation or merely learns that a particular contact position usually carried a particular lateral load.",
            "",
            "## Contact dynamics",
            "",
            f"- Provisional no-contact/loading/hold/release macro-F1: {phase_metrics['macro_f1']:.3f}.",
            f"- Accuracy: {phase_metrics['accuracy']:.3f}; grouped voting: {phase_metrics['group_voting_accuracy']:.3f}.",
            "- These phase labels come from synchronized Fz and dFz/dt, so this proves optical observability of the force phase proxy, not independent human annotation.",
            "",
            "## What the current data can support",
            "",
            "The current data already support contact credibility, P11-P33 position, continuous Fz, provisional loading/hold/release state, sensor stability, residual/recovery analysis, and optical contact-patch proxies such as center, spread, eccentricity, and orientation.",
            "",
            "Tangential force, moments, slip, texture, material, curvature, and multi-contact decomposition require dedicated protocols before they can be described as calibrated or generally recognizable. In particular, a high score for Fx/Fy from the present captures may reflect repeatable hand posture or position-specific bias.",
            "",
            "## Recommended hierarchy",
            "",
            "1. Drift- and stationarity-aware contact gate.",
            "2. Optical position/contact-patch estimator.",
            "3. Continuous Fz regression only while contact is credible.",
            "4. Temporal loading/hold/release state for residual suppression and event timing.",
            "5. Add shear, slip, texture, and object-shape heads only after controlled labels exist.",
            "",
            "No UI, runtime deployment, model bundle, or EXE was modified by this analysis.",
        ]
    )
    (args.output_dir / "additional_tactile_observability_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
