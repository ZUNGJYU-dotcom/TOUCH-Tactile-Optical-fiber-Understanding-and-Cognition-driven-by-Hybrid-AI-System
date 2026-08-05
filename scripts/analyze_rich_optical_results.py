"""Audit whether rich optical model gains generalize across capture sessions."""

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

from src.hybrid_spectrum.rich_optical_error_analysis import (  # noqa: E402
    PredictionSpec,
    classification_session_audit,
    delta_summary,
    feature_category_audit,
    force_session_audit,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rich_optical_algorithm_benchmark_20260801"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _overall_metric(
    leaderboard: pd.DataFrame,
    *,
    task: str,
    spec: PredictionSpec,
    metric: str,
) -> float:
    selected = leaderboard.loc[
        (leaderboard["task"] == task)
        & (leaderboard["model_id"] == spec.model_id)
        & (leaderboard["feature_view"] == spec.feature_view),
        metric,
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one leaderboard row for {task}/{spec.model_id}/"
            f"{spec.feature_view}, found {len(selected)}"
        )
    return float(selected.iloc[0])


def _merge_session_audits(
    contact: pd.DataFrame,
    position: pd.DataFrame,
    force: pd.DataFrame,
) -> pd.DataFrame:
    contact_columns = {
        "fold_id": "contact_fold_id",
        "frame_count": "contact_frame_count",
        "dominant_true_label": "contact_true_label",
        "dominant_true_ratio": "contact_true_ratio",
        "baseline_accuracy": "contact_baseline_accuracy",
        "candidate_accuracy": "contact_candidate_accuracy",
        "accuracy_delta": "contact_accuracy_delta",
        "baseline_error_count": "contact_baseline_error_count",
        "candidate_error_count": "contact_candidate_error_count",
        "no_contact_frame_count": "contact_no_contact_frame_count",
        "active_contact_frame_count": "contact_active_contact_frame_count",
    }
    position_columns = {
        "fold_id": "position_fold_id",
        "frame_count": "position_frame_count",
        "dominant_true_label": "position_true_label",
        "dominant_true_ratio": "position_true_ratio",
        "baseline_accuracy": "position_baseline_accuracy",
        "candidate_accuracy": "position_candidate_accuracy",
        "accuracy_delta": "position_accuracy_delta",
        "baseline_error_count": "position_baseline_error_count",
        "candidate_error_count": "position_candidate_error_count",
    }
    shared_drop = [
        "task",
        "baseline_model_id",
        "baseline_feature_view",
        "candidate_model_id",
        "candidate_feature_view",
    ]
    contact_view = contact.drop(columns=shared_drop).rename(columns=contact_columns)
    position_view = position.drop(columns=shared_drop).rename(columns=position_columns)
    force_view = force.drop(columns=shared_drop).rename(
        columns={"fold_id": "force_fold_id", "frame_count": "force_frame_count"}
    )
    merged = contact_view.merge(position_view, on="group_id", how="outer").merge(
        force_view, on="group_id", how="outer"
    )
    high_risk = (
        (merged["contact_candidate_accuracy"].fillna(1.0) < 0.80)
        | (merged["position_candidate_accuracy"].fillna(1.0) < 0.90)
        | (merged["candidate_mae_n"].fillna(0.0) > 0.50)
    )
    medium_risk = (
        (merged["contact_candidate_accuracy"].fillna(1.0) < 0.92)
        | (merged["position_candidate_accuracy"].fillna(1.0) < 0.97)
        | (merged["candidate_mae_n"].fillna(0.0) > 0.35)
    )
    merged["review_priority"] = np.select(
        [high_risk, medium_risk], ["high", "medium"], default="low"
    )
    return merged.sort_values(
        ["review_priority", "group_id"],
        key=lambda column: (
            column.map({"high": 0, "medium": 1, "low": 2})
            if column.name == "review_priority"
            else column
        ),
    ).reset_index(drop=True)


def _plot_session_deltas(
    contact: pd.DataFrame,
    position: pd.DataFrame,
    force: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    panels = (
        (contact, "accuracy_delta", "Contact accuracy delta", True, "percentage points"),
        (position, "accuracy_delta", "Position accuracy delta", True, "percentage points"),
        (force, "mae_improvement_n", "Fz MAE improvement", True, "N"),
    )
    for axis, (frame, metric, title, higher_is_better, unit) in zip(axes, panels):
        selected = frame.dropna(subset=[metric]).sort_values(metric)
        values = selected[metric].to_numpy(float)
        if unit == "percentage points":
            values = values * 100.0
        colors = np.where(values >= 0.0, "#1598bd", "#d57366")
        axis.barh(np.arange(len(values)), values, color=colors, alpha=0.9)
        axis.axvline(0.0, color="#17324d", linewidth=1.0)
        axis.set_title(title, weight="bold")
        axis.set_xlabel(unit)
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.18)
        if not higher_is_better:
            axis.invert_xaxis()
    fig.suptitle(
        "Per-session candidate improvement\n"
        "Blue = candidate improved; coral = candidate worsened",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _format_session_table(
    frame: pd.DataFrame,
    *,
    metric: str,
    columns: list[str],
    ascending: bool,
    limit: int = 8,
) -> list[str]:
    selected = frame.sort_values(metric, ascending=ascending).head(limit)
    if selected.empty:
        return ["No eligible sessions."]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in selected.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _write_decision_report(
    output_path: Path,
    *,
    leaderboard: pd.DataFrame,
    contact: pd.DataFrame,
    position: pd.DataFrame,
    force: pd.DataFrame,
    category_audit: pd.DataFrame,
    summaries: dict[str, Any],
) -> None:
    contact_baseline = PredictionSpec("extra_trees", "peak_current_40")
    contact_candidate = PredictionSpec("lightgbm", "rich_plus_full_spectrum_192")
    position_baseline = PredictionSpec("extra_trees", "peak_current_40")
    position_candidate = PredictionSpec("extra_trees", "full_spectrum_192")
    force_baseline = PredictionSpec("extra_trees", "peak_current_40")
    force_candidate = PredictionSpec("extra_trees", "rich_plus_full_spectrum_192")

    baseline_contact_f1 = _overall_metric(
        leaderboard, task="contact", spec=contact_baseline, metric="macro_f1"
    )
    candidate_contact_f1 = _overall_metric(
        leaderboard, task="contact", spec=contact_candidate, metric="macro_f1"
    )
    baseline_idle = _overall_metric(
        leaderboard,
        task="contact",
        spec=contact_baseline,
        metric="no_contact_recall",
    )
    candidate_idle = _overall_metric(
        leaderboard,
        task="contact",
        spec=contact_candidate,
        metric="no_contact_recall",
    )
    baseline_position_f1 = _overall_metric(
        leaderboard, task="position", spec=position_baseline, metric="macro_f1"
    )
    candidate_position_f1 = _overall_metric(
        leaderboard, task="position", spec=position_candidate, metric="macro_f1"
    )
    baseline_force_mae = _overall_metric(
        leaderboard, task="force_fz", spec=force_baseline, metric="mae_n"
    )
    candidate_force_mae = _overall_metric(
        leaderboard, task="force_fz", spec=force_candidate, metric="mae_n"
    )

    top_contact_families = category_audit.loc[
        category_audit["task"] == "contact"
    ].head(5)
    top_force_families = category_audit.loc[
        category_audit["task"] == "force_fz"
    ].head(5)
    lines = [
        "# Rich Optical Evidence Decision Report",
        "",
        "## Scope and validity",
        "",
        "- This report addresses model recognition and optical evidence only; recording, UI, deployment, and the Beta runtime were not changed.",
        "- Formal results use five folds grouped by immutable capture `session_id`; no random frame split is used.",
        "- The dataset contains 10,528 synchronized frames but only 50 independent capture sessions.",
        "- P11, P12, and P21 use only the latest primary sessions. Earlier questionable captures remain excluded.",
        "",
        "## Direct answer",
        "",
        "Yes. The spectra contain useful information beyond a single peak height or current value. The strongest additional evidence is distributed spectral-shape deformation, full-spectrum wavelength-bin patterns, local peak SNR/quality, wavelength-shift disagreement, and same-fibre/cross-fibre coupling statistics.",
        "",
        "The efficient model choice is task-specific rather than one universal network:",
        "",
        "1. Contact gate: LightGBM with rich optical physics plus the 192-bin full-spectrum view.",
        "2. Position: ExtraTrees with the 192-bin full-spectrum view.",
        "3. Optical Fz estimate: ExtraTrees with rich optical physics plus the full-spectrum view.",
        "4. True sequence recognition: ROCKET/HYDRA or a compact TCN only after consecutive spectra are assembled on a real time axis.",
        "",
        "## Like-for-like grouped results",
        "",
        f"- Contact macro-F1: {baseline_contact_f1:.4f} -> {candidate_contact_f1:.4f} ({candidate_contact_f1 - baseline_contact_f1:+.4f}).",
        f"- No-contact recall: {baseline_idle:.4f} -> {candidate_idle:.4f} ({candidate_idle - baseline_idle:+.4f}).",
        f"- Position macro-F1: {baseline_position_f1:.4f} -> {candidate_position_f1:.4f} ({candidate_position_f1 - baseline_position_f1:+.4f}).",
        f"- Fz MAE: {baseline_force_mae:.4f} N -> {candidate_force_mae:.4f} N ({candidate_force_mae - baseline_force_mae:+.4f} N; lower is better).",
        "",
        "The contact gain is meaningful but not yet production-safe: candidate no-contact recall is still below 0.90. Position is almost saturated offline, so current live position errors are more likely caused by domain/preprocessing drift than insufficient classifier capacity. The Fz gain is small and should not be overstated.",
        "",
        "## Independent-session robustness",
        "",
        f"- Contact: improved {summaries['contact']['improved_sessions']}/{summaries['contact']['session_count']} sessions, worsened {summaries['contact']['worsened_sessions']}, tied {summaries['contact']['tied_sessions']}; median frame-accuracy improvement {summaries['contact']['median_improvement']:+.4f}.",
        f"- No-contact frames within sessions: recall improved in {summaries['no_contact']['improved_sessions']}/{summaries['no_contact']['session_count']} eligible sessions, worsened in {summaries['no_contact']['worsened_sessions']}, tied in {summaries['no_contact']['tied_sessions']}.",
        f"- Active-contact frames within sessions: recall improved in {summaries['active_contact']['improved_sessions']}/{summaries['active_contact']['session_count']} eligible sessions, worsened in {summaries['active_contact']['worsened_sessions']}, tied in {summaries['active_contact']['tied_sessions']}.",
        f"- Position: improved {summaries['position']['improved_sessions']}/{summaries['position']['session_count']} sessions, worsened {summaries['position']['worsened_sessions']}, tied {summaries['position']['tied_sessions']}.",
        f"- Fz: lower MAE in {summaries['force']['improved_sessions']}/{summaries['force']['session_count']} sessions, higher MAE in {summaries['force']['worsened_sessions']}, tied {summaries['force']['tied_sessions']}.",
        "",
        "Perfect session-level majority voting is not proof of transient recognition quality because most capture sessions are dominated by one state. Frame-level no-contact/release behavior remains the stricter criterion.",
        "",
        "### Hardest contact sessions for the candidate",
        "",
        *_format_session_table(
            contact,
            metric="candidate_accuracy",
            columns=[
                "group_id",
                "dominant_true_label",
                "frame_count",
                "baseline_accuracy",
                "candidate_accuracy",
                "accuracy_delta",
            ],
            ascending=True,
        ),
        "",
        "### Hardest position sessions for the candidate",
        "",
        *_format_session_table(
            position,
            metric="candidate_accuracy",
            columns=[
                "group_id",
                "dominant_true_label",
                "frame_count",
                "baseline_accuracy",
                "candidate_accuracy",
                "accuracy_delta",
            ],
            ascending=True,
        ),
        "",
        "### Highest candidate Fz errors",
        "",
        *_format_session_table(
            force,
            metric="candidate_mae_n",
            columns=[
                "group_id",
                "frame_count",
                "true_fz_max_n",
                "baseline_mae_n",
                "candidate_mae_n",
                "mae_improvement_n",
            ],
            ascending=False,
        ),
        "",
        "## What the spectrum adds",
        "",
        "Top contact evidence families among the selected model's 30 leading features:",
        "",
        *[
            f"- `{row.evidence_family}`: {row.top_importance_share:.1%} of top-feature importance."
            for row in top_contact_families.itertuples()
        ],
        "",
        "Top Fz evidence families among the selected model's 30 leading features:",
        "",
        *[
            f"- `{row.evidence_family}`: {row.top_importance_share:.1%} of top-feature importance."
            for row in top_force_families.itertuples()
        ],
        "",
        "This supports a physical interpretation: force is encoded mainly in distributed spectral deformation and relative wavelength changes across FBGs, while position is encoded strongly in the detailed full-spectrum fingerprint.",
        "",
        "## Recommended next algorithm experiment",
        "",
        "Do not start with a larger CNN on individual 512-point spectra. First build consecutive-spectrum windows with shape `[window, wavelength_bin]` and aligned peak tracks. Evaluate MiniRocket/MultiRocketHydra or HYDRA against the current tree pipeline using the same grouped `session_id` folds. This directly targets press/release dynamics and residual drift without pretending that wavelength bins are time steps.",
        "",
        "A practical hierarchy is: stationary/drift-aware contact credibility -> full-spectrum position -> Fz regression only when contact is credible. This isolates the current weakest link instead of allowing a false contact decision to contaminate position and force outputs.",
        "",
        "## Deployment decision",
        "",
        "These are offline candidates only. Do not replace the Beta runtime yet. First replay the grouped out-of-fold predictions through the runtime state machine and collect clean release/no-contact sequences to validate false-contact recovery under live preprocessing.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    predictions = pd.read_csv(output_dir / "grouped_out_of_fold_predictions.csv")
    leaderboard = pd.read_csv(output_dir / "rich_optical_model_leaderboard.csv")
    importance = pd.read_csv(output_dir / "top_optical_feature_importance.csv")

    contact = classification_session_audit(
        predictions,
        task="contact",
        baseline=PredictionSpec("extra_trees", "peak_current_40"),
        candidate=PredictionSpec("lightgbm", "rich_plus_full_spectrum_192"),
        value_kind="integer",
    )
    position = classification_session_audit(
        predictions,
        task="position",
        baseline=PredictionSpec("extra_trees", "peak_current_40"),
        candidate=PredictionSpec("extra_trees", "full_spectrum_192"),
        value_kind="string",
    )
    force = force_session_audit(
        predictions,
        baseline=PredictionSpec("extra_trees", "peak_current_40"),
        candidate=PredictionSpec("extra_trees", "rich_plus_full_spectrum_192"),
    )
    merged = _merge_session_audits(contact, position, force)

    no_contact = contact.dropna(subset=["no_contact_recall_delta"])
    active_contact = contact.dropna(subset=["active_contact_recall_delta"])
    summaries = {
        "contact": delta_summary(
            contact, "accuracy_delta", higher_is_better=True
        ),
        "no_contact": delta_summary(
            no_contact, "no_contact_recall_delta", higher_is_better=True
        ),
        "active_contact": delta_summary(
            active_contact, "active_contact_recall_delta", higher_is_better=True
        ),
        "position": delta_summary(
            position, "accuracy_delta", higher_is_better=True
        ),
        "force": delta_summary(
            force, "mae_improvement_n", higher_is_better=True
        ),
    }

    category_frames = [
        feature_category_audit(
            importance,
            task="contact",
            spec=PredictionSpec("lightgbm", "rich_plus_full_spectrum_192"),
        ),
        feature_category_audit(
            importance,
            task="position",
            spec=PredictionSpec("extra_trees", "full_spectrum_192"),
        ),
        feature_category_audit(
            importance,
            task="force_fz",
            spec=PredictionSpec("extra_trees", "rich_plus_full_spectrum_192"),
        ),
    ]
    category_audit = pd.concat(category_frames, ignore_index=True)

    merged.to_csv(output_dir / "per_session_error_analysis.csv", index=False)
    category_audit.to_csv(output_dir / "optical_information_audit.csv", index=False)
    (output_dir / "per_session_robustness_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _plot_session_deltas(
        contact,
        position,
        force,
        output_dir / "per_session_generalization_audit.png",
    )
    _write_decision_report(
        output_dir / "rich_optical_decision_report.md",
        leaderboard=leaderboard,
        contact=contact,
        position=position,
        force=force,
        category_audit=category_audit,
        summaries=summaries,
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
