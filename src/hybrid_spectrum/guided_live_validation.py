"""Planning and reporting for guided live static-spectrum validation."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .live_shadow_validation import write_shadow_validation_artifacts
from .session_level_calibration import (
    CORE_FEATURE_NAMES,
    PerPositionOrdinalCalibrator,
)


POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
LEVEL_ORDER = ("light", "normal", "hard")
SHADOW_RESPONSE_FIELDS = {
    "shift_abs_mean_pm": "shadow_response_shift_abs_mean_pm",
    "shift_abs_max_pm": "shadow_response_shift_abs_max_pm",
    "normalized_residual_peak": "shadow_response_normalized_residual_peak",
    "normalized_residual_rms": "shadow_response_normalized_residual_rms",
}


def build_trial_plan(
    *,
    repeats: int = 1,
    positions: Sequence[str] = POSITION_ORDER,
    levels: Sequence[str] = LEVEL_ORDER,
    order: str = "randomized",
    seed: int = 20260714,
) -> list[dict[str, Any]]:
    """Build a balanced plan without treating frames as independent trials."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    unknown_positions = sorted(set(positions) - set(POSITION_ORDER))
    unknown_levels = sorted(set(levels) - set(LEVEL_ORDER))
    if unknown_positions:
        raise ValueError(f"unknown positions: {unknown_positions}")
    if unknown_levels:
        raise ValueError(f"unknown levels: {unknown_levels}")
    if order not in {"randomized", "blocked"}:
        raise ValueError("order must be randomized or blocked")

    trials: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        repeat_trials = [
            {
                "repeat": repeat,
                "trial_role": "calibration" if repeat == 1 else "validation",
                "position": position,
                "force_level": level,
                "contact_semantics": "manual_fingertip_static_contact",
                "force_semantics": "approximate_manual_response_level_not_force_N",
            }
            for position in positions
            for level in levels
        ]
        if order == "randomized":
            random.Random(seed + repeat - 1).shuffle(repeat_trials)
        trials.extend(repeat_trials)
    for index, trial in enumerate(trials, start=1):
        trial["trial_index"] = index
        trial["trial_id"] = (
            f"R{int(trial['repeat']):02d}_{trial['position']}_{trial['force_level']}"
        )
    return trials


def _accuracy(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(value is True for value in values) / len(values) if values else None


def _label_error_rate(
    rows: Iterable[dict[str, Any]], key: str, *, expected: str
) -> float | None:
    labels = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(label != expected for label in labels) / len(labels) if labels else None


def _confusion(
    rows: Iterable[dict[str, Any]], *, true_key: str, pred_key: str
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {}
    for row in rows:
        truth = row.get(true_key)
        prediction = row.get(pred_key)
        if truth is None or prediction is None:
            continue
        result.setdefault(str(truth), Counter())[str(prediction)] += 1
    return {truth: dict(counts) for truth, counts in result.items()}


def summarize_guided_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    contact_rows = [row for row in records if row.get("phase") == "contact"]
    validation_rows = [
        row for row in contact_rows if row.get("trial_role") == "validation"
    ]
    release_rows = [
        row
        for row in records
        if row.get("phase") in {"pre_release", "post_release"}
    ]
    temporal_ready = [
        row for row in contact_rows if row.get("shadow_temporal_ready") is True
    ]
    trial_ids = {str(row.get("trial_id")) for row in contact_rows if row.get("trial_id")}
    return {
        "schema_version": "guided_live_static_validation_v1",
        "record_count": len(records),
        "contact_frame_count": len(contact_rows),
        "release_frame_count": len(release_rows),
        "captured_trial_count": len(trial_ids),
        "shadow_temporal_ready_fraction": (
            len(temporal_ready) / len(contact_rows) if contact_rows else None
        ),
        "primary_position_accuracy": _accuracy(contact_rows, "primary_position_correct"),
        "shadow_position_accuracy": _accuracy(contact_rows, "shadow_position_correct"),
        "shadow_temporal_position_accuracy": _accuracy(
            temporal_ready, "shadow_temporal_position_correct"
        ),
        "primary_force_accuracy": _accuracy(contact_rows, "primary_force_correct"),
        "shadow_force_accuracy": _accuracy(contact_rows, "shadow_force_correct"),
        "shadow_temporal_force_accuracy": _accuracy(
            temporal_ready, "shadow_temporal_force_correct"
        ),
        "session_calibrated_validation_force_accuracy": _accuracy(
            validation_rows, "shadow_session_calibrated_force_correct"
        ),
        "primary_release_false_contact_rate": _label_error_rate(
            release_rows, "primary_contact", expected="no_contact"
        ),
        "shadow_release_false_contact_rate": _label_error_rate(
            release_rows, "shadow_contact", expected="no_contact"
        ),
        "shadow_temporal_release_false_contact_rate": _label_error_rate(
            release_rows, "shadow_temporal_contact", expected="no_contact"
        ),
        "shadow_position_confusion": _confusion(
            contact_rows,
            true_key="expected_position",
            pred_key="shadow_position",
        ),
        "shadow_temporal_position_confusion": _confusion(
            temporal_ready,
            true_key="expected_position",
            pred_key="shadow_temporal_position",
        ),
        "shadow_force_confusion": _confusion(
            contact_rows,
            true_key="expected_force",
            pred_key="shadow_force",
        ),
        "shadow_temporal_force_confusion": _confusion(
            temporal_ready,
            true_key="expected_force",
            pred_key="shadow_temporal_force",
        ),
        "evaluation_unit": "operator_labeled_trial_with_unique_spectrum_frames",
        "deployment_decision": "shadow_only_not_promoted",
    }


def fit_session_calibrator_from_guided_records(
    records: list[dict[str, Any]],
    *,
    required_positions: Sequence[str],
) -> dict[str, Any]:
    contact_rows = [
        row
        for row in records
        if row.get("phase") == "contact"
        and row.get("trial_role", "calibration") == "calibration"
    ]
    baseline_tokens = {
        str(row["baseline_spectrum_token"])
        for row in contact_rows
        if row.get("baseline_spectrum_token")
    }
    if len(baseline_tokens) != 1:
        return {
            "ok": False,
            "status": (
                "baseline_token_missing"
                if not baseline_tokens
                else "multiple_baselines_calibration_invalid"
            ),
            "baseline_token_count": len(baseline_tokens),
            "calibration_payload": None,
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in contact_rows:
        trial_id = row.get("trial_id")
        if trial_id:
            grouped.setdefault(str(trial_id), []).append(row)
    samples: list[dict[str, Any]] = []
    for trial_id, rows in grouped.items():
        preferred = [row for row in rows if row.get("shadow_temporal_ready") is True]
        if not preferred:
            preferred = [row for row in rows if row.get("shadow_contact") == "contact"]
        if not preferred:
            preferred = rows
        feature_values: dict[str, float] = {}
        complete = True
        for feature in CORE_FEATURE_NAMES:
            column = SHADOW_RESPONSE_FIELDS[feature]
            values = [
                float(row[column])
                for row in preferred
                if row.get(column) is not None
            ]
            if not values:
                complete = False
                break
            feature_values[feature] = float(np.median(values))
        if not complete:
            continue
        samples.append(
            {
                "trial_id": trial_id,
                "position": str(rows[0].get("expected_position") or ""),
                "level": str(rows[0].get("expected_force") or ""),
                "features": feature_values,
                "num_frames": len(preferred),
            }
        )

    present_pairs = {(sample["position"], sample["level"]) for sample in samples}
    missing_pairs = [
        f"{position}:{level}"
        for position in required_positions
        for level in LEVEL_ORDER
        if (position, level) not in present_pairs
    ]
    if missing_pairs:
        return {
            "ok": False,
            "status": "calibration_trials_incomplete",
            "trial_sample_count": len(samples),
            "missing_pairs": missing_pairs,
            "calibration_payload": None,
        }
    calibrator = PerPositionOrdinalCalibrator.fit(
        samples,
        baseline_token=next(iter(baseline_tokens)),
        required_positions=required_positions,
    )
    warning_positions = [
        position
        for position, quality in calibrator.quality.items()
        if quality.get("status") != "ready"
    ]
    return {
        "ok": True,
        "status": (
            "ready"
            if not warning_positions
            else "ready_with_monotonicity_warnings"
        ),
        "trial_sample_count": len(samples),
        "missing_pairs": [],
        "warning_positions": warning_positions,
        "calibration_payload": calibrator.to_dict(),
    }


def write_guided_validation_artifacts(
    output_dir: Path,
    *,
    trial_plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    baseline_result: dict[str, Any] | None = None,
    capture_errors: list[str] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_summary = write_shadow_validation_artifacts(
        output_dir,
        records,
        baseline_result=baseline_result,
        capture_errors=capture_errors,
    )
    guided_summary = summarize_guided_records(records)
    required_positions = tuple(
        position
        for position in POSITION_ORDER
        if any(
            row.get("position") == position
            and row.get("trial_role", "calibration") == "calibration"
            for row in trial_plan
        )
    )
    calibration_result = fit_session_calibrator_from_guided_records(
        records,
        required_positions=required_positions,
    )
    guided_summary.update(
        {
            "planned_trial_count": len(trial_plan),
            "capture_error_count": len(capture_errors or []),
            "shadow_summary": shadow_summary,
            "run_metadata": run_metadata or {},
            "session_level_calibration": {
                key: value
                for key, value in calibration_result.items()
                if key != "calibration_payload"
            },
        }
    )

    plan_path = output_dir / "guided_trial_plan.csv"
    if trial_plan:
        with plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trial_plan[0]))
            writer.writeheader()
            writer.writerows(trial_plan)
    else:
        plan_path.write_text("", encoding="utf-8-sig")

    (output_dir / "guided_validation_summary.json").write_text(
        json.dumps(guided_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "session_level_calibration_status.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in calibration_result.items()
                if key != "calibration_payload"
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if calibration_result.get("ok"):
        (output_dir / "session_level_calibration_candidate.json").write_text(
            json.dumps(
                calibration_result["calibration_payload"],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    report = [
        "# Guided Live Static-Spectrum Validation",
        "",
        "- Scope: nine approximate fingertip positions and three manual response levels.",
        "- Each row is tied to a unique BaySpec spectrum frame and an operator-labeled trial.",
        "- `light`, `normal`, and `hard` are not calibrated force_N.",
        "- The v7 candidate and its temporal vote remain shadow-only.",
        "- This tool cannot promote or overwrite the deployed model.",
        "",
        "## Progress",
        "",
        f"- Planned trials: {len(trial_plan)}",
        f"- Captured trials: {guided_summary['captured_trial_count']}",
        f"- Contact frames: {guided_summary['contact_frame_count']}",
        f"- Release frames: {guided_summary['release_frame_count']}",
        f"- Temporal-ready fraction: {guided_summary['shadow_temporal_ready_fraction']}",
        f"- Session-level calibration status: {calibration_result['status']}",
        "",
        "## Candidate metrics",
        "",
        f"- Raw shadow position accuracy: {guided_summary['shadow_position_accuracy']}",
        f"- Temporal shadow position accuracy: {guided_summary['shadow_temporal_position_accuracy']}",
        f"- Raw shadow level accuracy: {guided_summary['shadow_force_accuracy']}",
        f"- Temporal shadow level accuracy: {guided_summary['shadow_temporal_force_accuracy']}",
        f"- Session-calibrated validation level accuracy: {guided_summary['session_calibrated_validation_force_accuracy']}",
        f"- Temporal release false-contact rate: {guided_summary['shadow_temporal_release_false_contact_rate']}",
        "",
        "A promotion decision requires a completed independent live run and manual review.",
    ]
    (output_dir / "guided_live_validation_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return guided_summary


__all__ = [
    "LEVEL_ORDER",
    "POSITION_ORDER",
    "build_trial_plan",
    "fit_session_calibrator_from_guided_records",
    "summarize_guided_records",
    "write_guided_validation_artifacts",
]
