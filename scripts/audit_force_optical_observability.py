"""Export nine-point session-level force-to-optical observability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.advanced_optical_benchmark import (  # noqa: E402
    load_aligned_latest_primary,
)
from src.hybrid_spectrum.force_observability_audit import (  # noqa: E402
    build_session_observability_table,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/ordinary_fbg_force_observability_20260804"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dataset", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _plot(table, output_path: Path) -> None:
    positions = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
    figure, axes = plt.subplots(3, 3, figsize=(15.2, 10.6), sharey=True)
    colors = {
        "comparable_optical_sensitivity": "#2A9D8F",
        "optical_sensitivity_warning": "#E9C46A",
        "low_optical_sensitivity": "#E76F51",
        "manual_review_required": "#7A8B99",
    }
    for axis, position_id in zip(axes.flat, positions, strict=True):
        rows = table[table["position_id"] == position_id].reset_index(drop=True)
        x = np.arange(len(rows))
        axis.bar(
            x,
            rows["sensitivity_ratio_to_position_median"],
            color=[colors.get(value, "#7A8B99") for value in rows["observability_status"]],
            width=0.72,
        )
        axis.axhline(1.0, color="#264653", linewidth=1.0, linestyle="--")
        axis.axhline(0.60, color="#E76F51", linewidth=0.9, linestyle=":")
        axis.set_title(position_id)
        axis.set_xticks(x, [str(index + 1) for index in x], fontsize=7)
        axis.set_ylim(0.0, 1.65)
        axis.grid(axis="y", alpha=0.16)
        axis.set_xlabel("Session")
        axis.set_ylabel("Sensitivity / position median")
    figure.suptitle(
        "Force-to-optical sensitivity consistency by independent session",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=210, facecolor="white")
    plt.close(figure)


def _markdown_table(frame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.5f}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_aligned_latest_primary(
        args.fusion_dataset.resolve(), args.spectrum_dataset.resolve()
    )
    table = build_session_observability_table(dataset)
    table.to_csv(
        output / "force_optical_observability_by_session.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot(table, output / "force_optical_sensitivity_consistency.png")
    counts = table["observability_status"].value_counts().to_dict()
    flagged = table[
        table["observability_status"] != "comparable_optical_sensitivity"
    ][
        [
            "position_id",
            "group_id",
            "sensitivity_ratio_to_position_median",
            "global_log_ratio_rms_pearson_r",
            "observability_status",
        ]
    ]
    summary = {
        "session_count": int(len(table)),
        "position_count": int(table["position_id"].nunique()),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "force_sensor_used_as_runtime_input": False,
        "runtime_exclusion_applied": False,
    }
    (output / "force_optical_observability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Force-to-optical observability audit",
        "",
        "This post-acquisition audit checks whether equal PX6D force ranges produced "
        "comparable spectral response magnitudes. Fz is evidence only and is not a "
        "runtime model input or a test-session calibration signal.",
        "",
        f"- Independent force sessions: {len(table)}",
        f"- Tactile positions: {table['position_id'].nunique()}/9",
        f"- Low optical sensitivity: {counts.get('low_optical_sensitivity', 0)}",
        f"- Sensitivity warnings: {counts.get('optical_sensitivity_warning', 0)}",
        "- No session was removed by this audit.",
        "",
        "## Flagged sessions",
        "",
        _markdown_table(flagged) if not flagged.empty else "None.",
        "",
        "A low-sensitivity session is physically ambiguous: weak optical response can "
        "represent either low force or reduced coupling/sensitivity. It must remain in "
        "the grouped evaluation and should be recollected with controlled contact "
        "geometry if reliable absolute force is required.",
    ]
    (output / "force_optical_observability_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
