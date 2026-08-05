"""Benchmark high-force weighting without replacing the deployed model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.all_source_training import load_fusion_arrays  # noqa: E402
from src.hybrid_spectrum.force_weighting_benchmark import (  # noqa: E402
    FORCE_WEIGHTING_STRATEGIES,
    run_force_weighting_benchmark,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "ordinary_fbg_all_data_fusion_20260803_v2"
        / "all_source_fusion_dataset.npz",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "ordinary_fbg_all_data_fusion_20260803.yaml",
    )
    parser.add_argument(
        "--contact-gate",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "ordinary_fbg_all_data_fusion_training_20260803_v2"
        / "force_contact_gate_oof_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "ordinary_fbg_force_weighting_benchmark_20260803",
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _plot(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["weighting_strategy"].tolist()
    colors = ["#188fb7" if label == "baseline" else "#8fc4d5" for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    panels = (
        ("gated_mae_n", "Global gated MAE", "MAE (N)"),
        ("gated_high_force_mae_n", "High-force gated MAE (>=3 N)", "MAE (N)"),
        ("p13_gated_mae_n", "P13 gated MAE", "MAE (N)"),
        ("p13_gated_calibration_slope", "P13 calibration slope", "slope"),
    )
    for axis, (column, title, ylabel) in zip(axes.flat, panels):
        values = metrics[column].to_numpy(dtype=float)
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.22)
        if "slope" in column:
            axis.axhline(1.0, color="#d36b5f", linewidth=1.4, linestyle="--")
    fig.suptitle("Formal grouped force-weighting audit (deployed model unchanged)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    metrics: pd.DataFrame,
    decision: dict[str, Any],
    output_path: Path,
) -> None:
    baseline = metrics.set_index("weighting_strategy").loc["baseline"]
    recommended = metrics.set_index("weighting_strategy").loc[
        decision["recommended_strategy"]
    ]
    lines = [
        "# Ordinary-FBG force weighting benchmark",
        "",
        "This audit concerns the historical Measurement force curves. It uses "
        "the same optical-only 40-feature current-frame input, five formal "
        "session-grouped folds, source/session weighting, and contact gate as "
        "the current force estimator. No deployed model was changed.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['recommendation_status']}`",
        f"- Recommended strategy: `{decision['recommended_strategy']}`",
        f"- Reason: {decision['reason']}",
        "- The result is an audit candidate only; it is not a deployment action.",
        "",
        "## Baseline versus recommendation",
        "",
        f"- Global gated MAE: {baseline['gated_mae_n']:.4f} N -> "
        f"{recommended['gated_mae_n']:.4f} N",
        f"- Global gated R2: {baseline['gated_r2']:.4f} -> "
        f"{recommended['gated_r2']:.4f}",
        f"- High-force gated MAE: {baseline['gated_high_force_mae_n']:.4f} N -> "
        f"{recommended['gated_high_force_mae_n']:.4f} N",
        f"- P13 gated MAE: {baseline['p13_gated_mae_n']:.4f} N -> "
        f"{recommended['p13_gated_mae_n']:.4f} N",
        f"- P13 calibration slope: {baseline['p13_gated_calibration_slope']:.4f} -> "
        f"{recommended['p13_gated_calibration_slope']:.4f}",
        f"- Zero-force false response rate: "
        f"{baseline['gated_zero_force_false_response_rate']:.4%} -> "
        f"{recommended['gated_zero_force_false_response_rate']:.4%}",
        "",
        "## Interpretation",
        "",
        "A visual mismatch between PX6D and optical force traces can come from "
        "an obsolete recorded-runtime estimate, incomplete formal OOF coverage, "
        "or true force-regression amplitude error. Measurement now separates "
        "those evidence sources. This benchmark addresses only the third cause.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arrays = load_fusion_arrays(args.dataset)
    contact_gate = pd.read_csv(args.contact_gate, low_memory=False)
    result = run_force_weighting_benchmark(
        arrays,
        config,
        contact_gate_predictions=contact_gate,
        strategies=FORCE_WEIGHTING_STRATEGIES,
    )

    result.metrics.to_csv(
        args.output_dir / "force_weighting_results.csv", index=False
    )
    result.position_metrics.to_csv(
        args.output_dir / "force_weighting_position_metrics.csv", index=False
    )
    result.predictions.to_csv(
        args.output_dir / "force_weighting_oof_predictions.csv", index=False
    )
    _plot(result.metrics, args.output_dir / "force_weighting_comparison.png")
    _write_report(
        result.metrics,
        result.decision,
        args.output_dir / "force_weighting_report.md",
    )
    summary = {
        "schema_version": "ordinary_fbg_force_weighting_benchmark_v1",
        "dataset": str(args.dataset.resolve()),
        "split_strategy": "five_fold_grouped_by_session_id",
        "feature_view": "current_frame_optical_only_40",
        "model_family": "extra_trees",
        "strategies": list(FORCE_WEIGHTING_STRATEGIES),
        "decision": result.decision,
        "metrics": [
            {key: _json_value(value) for key, value in row.items()}
            for row in result.metrics.to_dict(orient="records")
        ],
    }
    (args.output_dir / "force_weighting_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                **result.decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
