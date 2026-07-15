"""Utilities for auditable primary-versus-shadow live model validation.

The shadow candidate is diagnostic only.  This module deliberately records
what both models predicted from the same frame and runtime baseline without
changing the operator display or deployed model bundle.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _label(component: Any) -> str | None:
    value = _mapping(component).get("label")
    return str(value) if value is not None else None


def _number(component: Any, key: str) -> float | None:
    value = _mapping(component).get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _prediction_fields(prefix: str, prediction: Any) -> dict[str, Any]:
    payload = _mapping(prediction)
    contact = _mapping(payload.get("contact"))
    position = _mapping(payload.get("position"))
    force = _mapping(payload.get("force_level"))
    uncertainty = _mapping(payload.get("uncertainty"))
    diagnostics = _mapping(position.get("ensemble_diagnostics"))
    response_features = _mapping(payload.get("response_calibration_features"))
    return {
        f"{prefix}_contact": _label(contact),
        f"{prefix}_contact_confidence": _number(contact, "confidence"),
        f"{prefix}_contact_decision_source": contact.get("decision_source"),
        f"{prefix}_position": _label(position),
        f"{prefix}_position_confidence": _number(position, "confidence"),
        f"{prefix}_position_margin": _number(position, "margin"),
        f"{prefix}_position_confidence_source": position.get("confidence_source"),
        f"{prefix}_force": _label(force),
        f"{prefix}_force_confidence": _number(force, "confidence"),
        f"{prefix}_force_margin": _number(force, "margin"),
        f"{prefix}_force_confidence_source": force.get("confidence_source"),
        f"{prefix}_force_model_scope": payload.get("force_model_scope"),
        f"{prefix}_review_needed": uncertainty.get("review_needed"),
        f"{prefix}_position_agreement": _number(diagnostics, "agreement_fraction"),
        f"{prefix}_position_unanimous": diagnostics.get("unanimous"),
        f"{prefix}_position_member_predictions": (
            json.dumps(diagnostics.get("member_predictions"), ensure_ascii=False, sort_keys=True)
            if diagnostics.get("member_predictions") is not None
            else None
        ),
        f"{prefix}_model_bundle_sha256": payload.get("model_bundle_sha256"),
        f"{prefix}_response_shift_abs_mean_pm": _number(
            response_features, "shift_abs_mean_pm"
        ),
        f"{prefix}_response_shift_abs_max_pm": _number(
            response_features, "shift_abs_max_pm"
        ),
        f"{prefix}_response_normalized_residual_peak": _number(
            response_features, "normalized_residual_peak"
        ),
        f"{prefix}_response_normalized_residual_rms": _number(
            response_features, "normalized_residual_rms"
        ),
    }


def _temporal_fields(temporal: Any) -> dict[str, Any]:
    payload = _mapping(temporal)
    return {
        "shadow_temporal_status": payload.get("status"),
        "shadow_temporal_ready": payload.get("ready"),
        "shadow_temporal_contact": payload.get("contact_label"),
        "shadow_temporal_position": payload.get("position_label"),
        "shadow_temporal_force": payload.get("force_label"),
        "shadow_temporal_position_support": _number(payload, "position_support"),
        "shadow_temporal_force_support": _number(payload, "force_support"),
        "shadow_temporal_contact_votes": payload.get("contact_votes"),
        "shadow_temporal_history_unique_frames": payload.get(
            "history_unique_frames"
        ),
        "shadow_temporal_duplicate_frame_ignored": payload.get(
            "duplicate_frame_ignored"
        ),
        "shadow_temporal_semantics": payload.get("semantics"),
    }


def flatten_shadow_frame(
    frame: dict[str, Any],
    *,
    captured_at: str,
    expected_contact: str | None = None,
    expected_position: str | None = None,
    expected_force: str | None = None,
) -> dict[str, Any]:
    """Flatten one ``/api/global_spectrum_frame`` response for CSV logging."""

    latest = _mapping(frame.get("latest"))
    model_frame = _mapping(frame.get("trained_static_spectral_frame"))
    primary = _mapping(model_frame.get("prediction"))
    shadow_wrapper = _mapping(model_frame.get("shadow_candidate"))
    shadow = _mapping(shadow_wrapper.get("prediction"))
    shadow_temporal = _mapping(shadow_wrapper.get("temporal_stabilization"))
    shadow_session_force = _mapping(shadow_wrapper.get("session_calibrated_force"))
    model_status = _mapping(frame.get("trained_static_spectral_model"))
    shadow_status = _mapping(model_status.get("shadow_candidate"))
    input_status = _mapping(model_frame.get("input"))

    inferred_contact = expected_contact
    if inferred_contact is None and (expected_position is not None or expected_force is not None):
        inferred_contact = "contact"

    record: dict[str, Any] = {
        "captured_at": captured_at,
        "frame_id": latest.get("frame_id", frame.get("frame_id")),
        "frame_timestamp": latest.get("timestamp", frame.get("timestamp")),
        "data_source": latest.get("source"),
        "source_fresh": frame.get("source_fresh"),
        "frame_age_sec": frame.get("frame_age_sec"),
        "baseline_ready": input_status.get("baseline_ready", frame.get("baseline_ready")),
        "baseline_spectrum_status": input_status.get("baseline_spectrum_status"),
        "baseline_spectrum_sample_count": input_status.get(
            "baseline_spectrum_sample_count"
        ),
        "baseline_spectrum_token": input_status.get("baseline_spectrum_token"),
        "model_assisted_display_allowed": frame.get("model_assisted_display_allowed"),
        "model_assisted_display_block_reason": frame.get(
            "model_assisted_display_block_reason"
        ),
        "primary_model_status": model_frame.get("status"),
        "primary_model_reason": model_frame.get("reason"),
        "primary_model_bundle_sha256": model_status.get("model_bundle_sha256"),
        "shadow_status": shadow_wrapper.get("status"),
        "shadow_runtime_role": shadow_wrapper.get("runtime_role"),
        "shadow_drives_operator_ui": shadow_wrapper.get("drives_operator_ui"),
        "shadow_drives_digital_twin": shadow_wrapper.get("drives_digital_twin"),
        "shadow_model_bundle_sha256": shadow_status.get("model_bundle_sha256"),
        "expected_contact": inferred_contact,
        "expected_position": expected_position,
        "expected_force": expected_force,
    }
    record.update(_prediction_fields("primary", primary))
    record.update(_prediction_fields("shadow", shadow))
    record.update(_temporal_fields(shadow_temporal))
    record.update(
        {
            "shadow_session_calibrated_force_status": shadow_session_force.get(
                "status"
            ),
            "shadow_session_calibrated_force": shadow_session_force.get("label"),
            "shadow_session_calibrated_force_score": _number(
                shadow_session_force, "ordinal_score"
            ),
            "shadow_session_calibrated_force_margin": _number(
                shadow_session_force, "margin"
            ),
            "shadow_session_calibrated_force_position": shadow_session_force.get(
                "position"
            ),
            "shadow_session_calibrated_force_confidence_source": shadow_session_force.get(
                "confidence_source"
            ),
            "shadow_session_calibrated_force_drives_ui": shadow_session_force.get(
                "drives_operator_ui"
            ),
            "shadow_session_calibrated_force_drives_twin": shadow_session_force.get(
                "drives_digital_twin"
            ),
        }
    )

    both_ready = bool(primary and shadow and shadow_wrapper.get("ok"))
    record["both_models_ready"] = both_ready
    record["position_models_agree"] = (
        record["primary_position"] == record["shadow_position"]
        if both_ready
        and record["primary_position"] is not None
        and record["shadow_position"] is not None
        else None
    )
    record["force_models_agree"] = (
        record["primary_force"] == record["shadow_force"]
        if both_ready
        and record["primary_force"] is not None
        and record["shadow_force"] is not None
        else None
    )
    for prefix in ("primary", "shadow"):
        record[f"{prefix}_contact_correct"] = (
            record[f"{prefix}_contact"] == inferred_contact
            if inferred_contact is not None and record[f"{prefix}_contact"] is not None
            else None
        )
        record[f"{prefix}_position_correct"] = (
            record[f"{prefix}_position"] == expected_position
            if expected_position is not None and record[f"{prefix}_position"] is not None
            else None
        )
        record[f"{prefix}_force_correct"] = (
            record[f"{prefix}_force"] == expected_force
            if expected_force is not None and record[f"{prefix}_force"] is not None
            else None
        )
    record["shadow_temporal_contact_correct"] = (
        record["shadow_temporal_contact"] == inferred_contact
        if inferred_contact is not None
        and record["shadow_temporal_contact"] is not None
        else None
    )
    record["shadow_temporal_position_correct"] = (
        record["shadow_temporal_position"] == expected_position
        if expected_position is not None
        and record["shadow_temporal_position"] is not None
        else None
    )
    record["shadow_temporal_force_correct"] = (
        record["shadow_temporal_force"] == expected_force
        if expected_force is not None and record["shadow_temporal_force"] is not None
        else None
    )
    record["shadow_session_calibrated_force_correct"] = (
        record["shadow_session_calibrated_force"] == expected_force
        if expected_force is not None
        and record["shadow_session_calibrated_force"] is not None
        else None
    )
    return record


def _optional_mean(values: Iterable[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _accuracy(records: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in records if row.get(key) is not None]
    return sum(bool(value) for value in values) / len(values) if values else None


def summarize_shadow_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize availability, agreement, and optional operator labels."""

    both_ready = [row for row in records if row.get("both_models_ready")]
    summary = {
        "schema_version": "live_static_shadow_validation_v2",
        "record_count": len(records),
        "unique_frame_count": len({row.get("frame_id") for row in records}),
        "source_counts": dict(Counter(str(row.get("data_source")) for row in records)),
        "baseline_ready_count": sum(bool(row.get("baseline_ready")) for row in records),
        "both_models_ready_count": len(both_ready),
        "primary_status_counts": dict(
            Counter(str(row.get("primary_model_status")) for row in records)
        ),
        "shadow_status_counts": dict(
            Counter(str(row.get("shadow_status")) for row in records)
        ),
        "shadow_temporal_status_counts": dict(
            Counter(str(row.get("shadow_temporal_status")) for row in records)
        ),
        "primary_shadow_position_agreement": _accuracy(
            both_ready, "position_models_agree"
        ),
        "primary_shadow_force_agreement": _accuracy(both_ready, "force_models_agree"),
        "shadow_position_mean_agreement_fraction": _optional_mean(
            row.get("shadow_position_agreement") for row in both_ready
        ),
        "shadow_position_unanimous_fraction": _accuracy(
            both_ready, "shadow_position_unanimous"
        ),
        "primary_contact_accuracy": _accuracy(records, "primary_contact_correct"),
        "shadow_contact_accuracy": _accuracy(records, "shadow_contact_correct"),
        "primary_position_accuracy": _accuracy(records, "primary_position_correct"),
        "shadow_position_accuracy": _accuracy(records, "shadow_position_correct"),
        "primary_force_accuracy": _accuracy(records, "primary_force_correct"),
        "shadow_force_accuracy": _accuracy(records, "shadow_force_correct"),
        "shadow_temporal_contact_accuracy": _accuracy(
            records, "shadow_temporal_contact_correct"
        ),
        "shadow_temporal_position_accuracy": _accuracy(
            records, "shadow_temporal_position_correct"
        ),
        "shadow_temporal_force_accuracy": _accuracy(
            records, "shadow_temporal_force_correct"
        ),
        "shadow_session_calibrated_force_accuracy": _accuracy(
            records, "shadow_session_calibrated_force_correct"
        ),
        "shadow_ever_drove_operator_ui": any(
            row.get("shadow_drives_operator_ui") is True for row in records
        ),
        "shadow_ever_drove_digital_twin": any(
            row.get("shadow_drives_digital_twin") is True for row in records
        ),
        "session_calibrated_force_ever_drove_operator_ui": any(
            row.get("shadow_session_calibrated_force_drives_ui") is True
            for row in records
        ),
        "session_calibrated_force_ever_drove_digital_twin": any(
            row.get("shadow_session_calibrated_force_drives_twin") is True
            for row in records
        ),
        "deployment_decision": "shadow_only_not_promoted",
    }
    return summary


def write_shadow_validation_artifacts(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    capture_errors: list[str] | None = None,
    baseline_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_shadow_records(records)
    summary["capture_error_count"] = len(capture_errors or [])
    summary["baseline_request_result"] = baseline_result

    csv_path = output_dir / "live_shadow_predictions.csv"
    if records:
        fieldnames = list(records[0])
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    else:
        csv_path.write_text("", encoding="utf-8-sig")

    (output_dir / "live_shadow_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "capture_errors.json").write_text(
        json.dumps(capture_errors or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Live Static Spectral Shadow Validation",
        "",
        "- The deployed model remained primary and was not overwritten.",
        "- The v7 fused-shift ensemble candidate remained shadow-only.",
        "- Both predictions were computed from the same live spectrum and runtime baseline.",
        "- Temporal results use unique-frame debounce/voting and remain diagnostic-only.",
        "- Position labels are P11-P33; level labels are approximate light/normal/hard responses, not force_N.",
        "",
        "## Capture status",
        "",
        f"- Records: {summary['record_count']}",
        f"- Unique frames: {summary['unique_frame_count']}",
        f"- Frames with both models ready: {summary['both_models_ready_count']}",
        f"- Baseline-ready frames: {summary['baseline_ready_count']}",
        f"- Capture errors: {summary['capture_error_count']}",
        f"- Primary/shadow position agreement: {summary['primary_shadow_position_agreement']}",
        f"- Primary/shadow level agreement: {summary['primary_shadow_force_agreement']}",
        f"- Temporal status counts: {summary['shadow_temporal_status_counts']}",
        f"- Temporal contact accuracy: {summary['shadow_temporal_contact_accuracy']}",
        f"- Temporal position accuracy: {summary['shadow_temporal_position_accuracy']}",
        f"- Temporal level accuracy: {summary['shadow_temporal_force_accuracy']}",
        f"- Session-calibrated level accuracy: {summary['shadow_session_calibrated_force_accuracy']}",
        "",
        "## Safety boundary",
        "",
        f"- Shadow drove Operator UI: {summary['shadow_ever_drove_operator_ui']}",
        f"- Shadow drove digital twin: {summary['shadow_ever_drove_digital_twin']}",
        "- Promotion requires a labeled live validation sequence and is not performed by this tool.",
    ]
    (output_dir / "live_shadow_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "flatten_shadow_frame",
    "summarize_shadow_records",
    "write_shadow_validation_artifacts",
]
