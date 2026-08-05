"""Audit grouped optical/PX6D force consistency across P11-P33."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.force_consistency_audit import (  # noqa: E402
    POSITION_ORDER,
    build_force_consistency_tables,
    load_grouped_force_predictions,
    plot_position_regression,
    plot_representative_traces,
)


DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_all_data_fusion_training_20260803_v2"
    / "force_contact_gate_oof_predictions.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_nine_point_force_consistency_20260803"
)


def _safe(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    headers = [str(column) for column in table.columns]
    rows = [headers, ["---"] * len(headers)]
    for values in table.itertuples(index=False, name=None):
        rows.append(
            [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _position_table_markdown(positions: pd.DataFrame) -> str:
    columns = [
        "position_id",
        "mae_n",
        "r2",
        "pearson_r",
        "linear_slope_pred_vs_px6d",
        "amplitude_ratio_p95_p05",
        "lag_ms",
        "audit_status",
    ]
    table = positions[columns].copy()
    for field in columns[1:-1]:
        table[field] = table[field].map(
            lambda value: "--" if pd.isna(value) else f"{float(value):.3f}"
        )
    return _markdown_table(table)


def _report(
    predictions_path: Path,
    sessions: pd.DataFrame,
    positions: pd.DataFrame,
) -> str:
    worst = positions.sort_values("mae_n", ascending=False).iloc[0]
    inconsistent = positions[positions["audit_status"] != "consistent"]
    worst_sessions = sessions.sort_values("mae_n", ascending=False).head(8).copy()
    risk_columns = [
        "position_id",
        "group_id",
        "mae_n",
        "r2",
        "pearson_r",
        "linear_slope_pred_vs_px6d",
        "audit_reason",
    ]
    for field in ("mae_n", "r2", "pearson_r", "linear_slope_pred_vs_px6d"):
        worst_sessions[field] = worst_sessions[field].map(
            lambda value: "--" if pd.isna(value) else f"{float(value):.3f}"
        )
    return f"""# Ordinary-FBG Nine-Point Optical Force Consistency Audit

## Scope and evidence

This report audits the curves shown by Diagnostics > Measurement. The formal
comparison uses grouped out-of-fold predictions by independent `session_id`:
`{predictions_path.resolve()}`.

PX6D Fz is the synchronized 0-5 N calibration target only. The optical model
receives optical features only. Recorded runtime predictions are historical
evidence and must not be presented as the current model curve.

## Nine-point result

{_position_table_markdown(positions)}

The weakest aggregate point is **{worst['position_id']}** with MAE
{worst['mae_n']:.3f} N, R2 {worst['r2']:.3f}, and fitted amplitude slope
{worst['linear_slope_pred_vs_px6d']:.3f}. Positions currently carrying a
warning or review flag: {', '.join(inconsistent['position_id'].astype(str)) or 'none'}.

## Highest-risk independent sessions

{_markdown_table(worst_sessions[risk_columns])}

## Interpretation

Curve shape and force amplitude are separate checks. A high Pearson
correlation with a slope below 1 means the optical signal follows the press and
release trend but systematically underestimates force. A small positive lag is
acceptable as a physical/acquisition delay; lag compensation is reported for
diagnosis only and is not used to inflate the formal direct score.

The Measurement screen should default to **Best valid evidence**, which resolves
to grouped OOF for sessions present in this formal dataset. Current-model replay
is useful for post-training recordings but is not automatically independent.
Recorded runtime is retained only as a labeled historical reference.

## Decision boundary

No per-session PX6D-derived scale factor is applied to optical predictions.
Doing so would leak the force reference into the test session and create an
artificially matched curve. A new force model should be deployed only if it
improves grouped-session MAE, force amplitude consistency, worst-position
behavior, and zero-force residuals together.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_grouped_force_predictions(args.predictions)
    sessions, positions = build_force_consistency_tables(predictions)

    sessions.to_csv(
        output_dir / "force_consistency_by_session.csv",
        index=False,
        encoding="utf-8-sig",
    )
    positions.to_csv(
        output_dir / "force_consistency_by_position.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_representative_traces(
        predictions,
        sessions,
        output_dir / "nine_point_force_trace_consistency.png",
    )
    plot_position_regression(
        predictions,
        output_dir / "nine_point_force_regression.png",
    )

    payload = {
        "schema_version": "ordinary_fbg_nine_point_force_consistency_v1",
        "evaluation_validity": "formal_grouped_oof_by_session_id",
        "prediction_source_file": str(args.predictions.resolve()),
        "position_order": list(POSITION_ORDER),
        "session_count": int(len(sessions)),
        "position_count": int(len(positions)),
        "positions": [
            {key: _safe(value) for key, value in row.items()}
            for row in positions.to_dict(orient="records")
        ],
    }
    (output_dir / "force_consistency_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "force_consistency_report.md").write_text(
        _report(args.predictions, sessions, positions),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
