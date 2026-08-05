"""Resolve force-estimate evidence for offline Measurement diagnostics.

Recorded runtime estimates, grouped out-of-fold predictions, and model replay
have different scientific meanings.  This module keeps those sources separate
and returns a common per-frame overlay without changing the live adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .all_source_runtime_adapter import AllSourceOpticalForceAdapter


EVIDENCE_SOURCES = (
    "best_available",
    "grouped_oof",
    "current_model_replay",
    "recorded_runtime",
)

MIN_GROUPED_OOF_TRACE_COVERAGE = 0.60
MIN_GROUPED_OOF_MATCHED_ROWS = 20


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_ids(session_dir: Path) -> set[str]:
    identifiers = {Path(session_dir).name}
    metadata_path = Path(session_dir) / "session_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and payload.get("session_id"):
            identifiers.add(str(payload["session_id"]))
    return identifiers


def _trace_lookup(trace_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for row in trace_rows:
        capture_index = _safe_int(row.get("capture_index"))
        if capture_index is not None:
            lookup[capture_index] = row
    return lookup


def _training_model_id(training_dir: Path) -> str:
    summary_path = training_dir / "training_summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            selected = payload.get("selected_models") or {}
            if selected.get("force_fz"):
                return str(selected["force_fz"])
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            pass
    return "grouped_oof_force_model"


def load_grouped_oof_evidence(
    session_dir: Path,
    trace_rows: list[dict[str, Any]],
    outputs_root: Path,
) -> dict[str, Any]:
    """Load the newest exact-session grouped OOF force prediction series."""

    session_dir = Path(session_dir)
    outputs_root = Path(outputs_root)
    identifiers = _session_ids(session_dir)
    trace_by_index = _trace_lookup(trace_rows)
    candidates = sorted(
        outputs_root.rglob("force_contact_gate_oof_predictions.csv")
        if outputs_root.is_dir()
        else (),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for prediction_path in candidates:
        matched: list[dict[str, str]] = []
        try:
            with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("group_id") or row.get("file_id") or "") in identifiers:
                        matched.append(row)
        except (OSError, UnicodeError, csv.Error):
            continue
        if not matched:
            continue

        overlay: dict[int, dict[str, Any]] = {}
        elapsed_deltas: list[float] = []
        reference_errors: list[float] = []
        fold_ids: set[str] = set()
        for row in matched:
            sample_index = _safe_int(row.get("sample_index"))
            if sample_index is None or sample_index not in trace_by_index:
                continue
            trace_row = trace_by_index[sample_index]
            oof_elapsed = _safe_float(row.get("elapsed_time_sec"))
            trace_elapsed = _safe_float(trace_row.get("elapsed_time_sec"))
            if oof_elapsed is not None and trace_elapsed is not None:
                elapsed_deltas.append(abs(oof_elapsed - trace_elapsed))
            oof_reference = _safe_float(row.get("true_force_n"))
            trace_reference = _safe_float(trace_row.get("reference_fz_n"))
            if oof_reference is not None and trace_reference is not None:
                reference_errors.append(abs(oof_reference - trace_reference))
            gate_active = _safe_bool(row.get("contact_gate_active"))
            overlay[sample_index] = {
                "estimated_fz_n": _safe_float(row.get("gated_force_n")),
                "raw_estimated_fz_n": _safe_float(
                    row.get("raw_optical_force_n")
                    or row.get("unclipped_raw_force_n")
                ),
                "contact_label": "contact" if gate_active else "no_contact",
                "position_label": None,
                "inference_latency_ms": None,
                "model_source": "grouped_oof",
            }
            if row.get("fold_id") not in (None, ""):
                fold_ids.add(str(row["fold_id"]))

        coverage = len(overlay) / max(len(trace_by_index), 1)
        max_elapsed_delta = max(elapsed_deltas, default=None)
        reference_mae = (
            float(np.mean(reference_errors)) if reference_errors else None
        )
        minimum_matched_rows = min(
            MIN_GROUPED_OOF_MATCHED_ROWS,
            max(
                1,
                int(
                    math.ceil(
                        max(len(trace_by_index), 1)
                        * MIN_GROUPED_OOF_TRACE_COVERAGE
                    )
                ),
            ),
        )
        alignment_ok = bool(
            len(overlay) >= minimum_matched_rows
            and coverage >= MIN_GROUPED_OOF_TRACE_COVERAGE
            and (max_elapsed_delta is None or max_elapsed_delta <= 0.35)
            and (reference_mae is None or reference_mae <= 0.05)
        )
        if not overlay or not alignment_ok:
            continue

        training_dir = prediction_path.parent
        artifact_path = (
            training_dir
            / "candidate_models"
            / "ordinary_fbg_optical_only_force_candidate.joblib"
        )
        model_id = _training_model_id(training_dir)
        return {
            "ok": True,
            "overlay": overlay,
            "source": "grouped_oof",
            "label": f"Grouped OOF ({model_id})",
            "evaluation_validity": "formal_grouped_oof_by_session_id",
            "provenance": {
                "prediction_source_file": str(prediction_path.resolve()),
                "training_output_dir": str(training_dir.resolve()),
                "model_artifact_path": (
                    str(artifact_path.resolve()) if artifact_path.is_file() else None
                ),
                "model_sha256": _sha256(artifact_path),
                "model_id": model_id,
                "fold_ids": sorted(fold_ids),
                "session_ids": sorted(identifiers),
                "matched_row_count": len(overlay),
                "trace_row_count": len(trace_by_index),
                "coverage_ratio": coverage,
                "coverage_status": (
                    "complete_trace_coverage"
                    if coverage >= 0.95
                    else "partial_trace_coverage"
                ),
                "unmatched_trace_row_count": max(0, len(trace_by_index) - len(overlay)),
                "maximum_elapsed_alignment_error_sec": max_elapsed_delta,
                "reference_alignment_mae_n": reference_mae,
                "force_sensor_is_runtime_model_input": False,
                "note": (
                    "Each session was excluded from its prediction fold; this is "
                    "the preferred model-evaluation curve when available."
                ),
            },
        }
    return {
        "ok": False,
        "source": "grouped_oof",
        "status": "grouped_oof_not_found_for_session",
        "reason": "No exact-session grouped OOF force predictions were found.",
    }


def _load_spectrum_frames(session_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    spectrum_path = Path(session_dir) / "spectrum_timeseries.csv"
    if not spectrum_path.is_file():
        return {}
    buckets: dict[int, dict[str, list[float]]] = {}
    with spectrum_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            capture_index = _safe_int(row.get("capture_index"))
            wavelength = _safe_float(row.get("wavelength_nm"))
            intensity = _safe_float(row.get("intensity_counts"))
            if capture_index is None or wavelength is None or intensity is None:
                continue
            bucket = buckets.setdefault(
                capture_index,
                {"wavelength": [], "intensity": [], "baseline": []},
            )
            bucket["wavelength"].append(wavelength)
            bucket["intensity"].append(intensity)
            baseline = _safe_float(row.get("baseline_intensity_counts"))
            bucket["baseline"].append(
                baseline if baseline is not None else math.nan
            )

    frames: dict[int, dict[str, np.ndarray]] = {}
    for capture_index, bucket in buckets.items():
        wavelength = np.asarray(bucket["wavelength"], dtype=float)
        intensity = np.asarray(bucket["intensity"], dtype=float)
        baseline = np.asarray(bucket["baseline"], dtype=float)
        if wavelength.size < 5 or wavelength.shape != intensity.shape:
            continue
        order = np.argsort(wavelength)
        frames[capture_index] = {
            "wavelength": wavelength[order],
            "intensity": intensity[order],
            "baseline": baseline[order],
        }
    return frames


def _recorded_baseline(
    frames: dict[int, dict[str, np.ndarray]],
    baseline_frame_count: int,
    *,
    strategy: str = "session_initial_stable_median",
    minimum_stable_frames: int = 5,
    stability_mad_multiplier: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select an optical-only replay baseline without borrowing PX6D labels.

    Stored ``baseline_intensity_counts`` can predate the recording session. It
    is therefore diagnostic evidence, not the default replay baseline. The
    default uses a robust median of stable spectra at the start of this session,
    matching the acquisition protocol while keeping force out of inference.
    """

    ordered_indices = sorted(frames)
    if not ordered_indices:
        raise ValueError("spectrum_timeseries.csv contains no usable spectra")
    first = frames[ordered_indices[0]]
    wavelength = first["wavelength"]
    recorded = first["baseline"]
    recorded_available = bool(
        recorded.shape == wavelength.shape
        and np.all(np.isfinite(recorded))
        and np.any(np.abs(recorded) > 1.0e-12)
    )
    normalized_strategy = str(strategy or "").strip().lower()
    if normalized_strategy not in {
        "session_initial_stable_median",
        "recorded_baseline_intensity_counts",
    }:
        raise ValueError(f"unsupported replay baseline strategy: {strategy}")
    if normalized_strategy == "recorded_baseline_intensity_counts":
        if not recorded_available:
            raise ValueError("recorded baseline is unavailable for this session")
        return wavelength, recorded, {
            "method": "recorded_baseline_intensity_counts",
            "frame_indices": [ordered_indices[0]],
            "frame_count": 1,
            "recorded_baseline_policy": "explicit_historical_diagnostic",
        }

    selected_indices = ordered_indices[: max(1, int(baseline_frame_count))]
    aligned: list[np.ndarray] = []
    aligned_indices: list[int] = []
    for capture_index in selected_indices:
        frame = frames[capture_index]
        current_wavelength = frame["wavelength"]
        current_intensity = frame["intensity"]
        if current_wavelength[0] > wavelength[0] or current_wavelength[-1] < wavelength[-1]:
            continue
        aligned.append(np.interp(wavelength, current_wavelength, current_intensity))
        aligned_indices.append(capture_index)
    if not aligned:
        if recorded_available:
            return wavelength, recorded, {
                "method": "recorded_baseline_intensity_counts_fallback",
                "frame_indices": [ordered_indices[0]],
                "frame_count": 1,
                "recorded_baseline_policy": "fallback_only",
            }
        raise ValueError("no spectra cover the baseline wavelength grid")

    candidates = np.asarray(aligned, dtype=float)
    provisional = np.median(candidates, axis=0)
    intensity_scale = max(float(np.median(np.abs(provisional))), 1.0)
    distances = np.sqrt(np.mean(np.square(candidates - provisional), axis=1))
    normalized_distances = distances / intensity_scale
    distance_median = float(np.median(normalized_distances))
    distance_mad = float(
        1.4826 * np.median(np.abs(normalized_distances - distance_median))
    )
    threshold = max(
        distance_median + float(stability_mad_multiplier) * distance_mad,
        distance_median * 2.0 + 1.0e-9,
        0.005,
    )
    accepted = np.flatnonzero(normalized_distances <= threshold)
    minimum = min(
        len(candidates),
        max(1, int(minimum_stable_frames)),
    )
    if len(accepted) < minimum:
        accepted = np.argsort(normalized_distances)[:minimum]
    accepted = np.sort(accepted)
    accepted_indices = [aligned_indices[int(index)] for index in accepted]
    baseline = np.median(candidates[accepted], axis=0)

    recorded_mean_difference = None
    recorded_max_difference = None
    recorded_rms_difference_ratio = None
    recorded_max_difference_ratio = None
    recorded_consistency_status = "unavailable"
    quality_flags: list[str] = []
    if recorded_available:
        absolute_difference = np.abs(recorded - baseline)
        recorded_mean_difference = float(np.mean(absolute_difference))
        recorded_max_difference = float(np.max(absolute_difference))
        recorded_rms_difference_ratio = float(
            np.sqrt(np.mean(np.square(recorded - baseline))) / intensity_scale
        )
        recorded_max_difference_ratio = float(
            recorded_max_difference / intensity_scale
        )
        mismatch_threshold = max(0.01, 1.5 * threshold)
        if recorded_rms_difference_ratio > mismatch_threshold:
            recorded_consistency_status = "mismatch_warning"
            quality_flags.append("recorded_baseline_mismatch")
        else:
            recorded_consistency_status = "consistent"

    return wavelength, baseline, {
        "method": "session_initial_stable_median",
        "frame_indices": accepted_indices,
        "frame_count": len(accepted_indices),
        "candidate_frame_indices": aligned_indices,
        "candidate_frame_count": len(aligned_indices),
        "stability_distance_median": distance_median,
        "stability_distance_max_accepted": float(
            np.max(normalized_distances[accepted])
        ),
        "stability_threshold": threshold,
        "recorded_baseline_available": recorded_available,
        "recorded_baseline_policy": "diagnostic_only",
        "recorded_baseline_mean_absolute_difference_counts": recorded_mean_difference,
        "recorded_baseline_max_absolute_difference_counts": recorded_max_difference,
        "recorded_baseline_normalized_rms_difference": (
            recorded_rms_difference_ratio
        ),
        "recorded_baseline_normalized_max_difference": (
            recorded_max_difference_ratio
        ),
        "recorded_baseline_consistency_status": recorded_consistency_status,
        "quality_flags": quality_flags,
        "force_sensor_used_for_baseline_selection": False,
    }


def replay_current_model_evidence(
    session_dir: Path,
    trace_rows: list[dict[str, Any]],
    model_path: Path,
    peak_config_path: Path,
    *,
    runtime_recovery_config: dict[str, Any] | None = None,
    runtime_gate_config: dict[str, Any] | None = None,
    baseline_frame_count: int = 10,
    baseline_strategy: str = "session_initial_stable_median",
    baseline_minimum_stable_frames: int = 5,
    baseline_stability_mad_multiplier: float = 3.5,
    adapter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Replay one completed spectrum sequence through an isolated adapter."""

    session_dir = Path(session_dir)
    model_path = Path(model_path)
    peak_config_path = Path(peak_config_path)
    if not model_path.is_file():
        return {
            "ok": False,
            "source": "current_model_replay",
            "status": "current_model_not_found",
            "reason": str(model_path),
        }
    frames = _load_spectrum_frames(session_dir)
    if not frames:
        return {
            "ok": False,
            "source": "current_model_replay",
            "status": "spectrum_timeseries_not_found",
            "reason": "A complete spectrum_timeseries.csv is required for replay.",
        }

    wavelength, baseline, baseline_info = _recorded_baseline(
        frames,
        baseline_frame_count,
        strategy=baseline_strategy,
        minimum_stable_frames=baseline_minimum_stable_frames,
        stability_mad_multiplier=baseline_stability_mad_multiplier,
    )
    factory = adapter_factory or AllSourceOpticalForceAdapter.from_paths
    adapter = factory(
        model_path,
        peak_config_path,
        runtime_recovery_config=runtime_recovery_config,
        runtime_gate_config=runtime_gate_config,
    )
    adapter.set_baseline(wavelength, baseline)

    overlay: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for trace_row in sorted(
        trace_rows, key=lambda row: int(row.get("capture_index") or 0)
    ):
        capture_index = _safe_int(trace_row.get("capture_index"))
        if capture_index is None or capture_index not in frames:
            continue
        frame = frames[capture_index]
        timestamp = _safe_float(trace_row.get("timeline_timestamp_epoch_sec"))
        if timestamp is None:
            timestamp = _safe_float(trace_row.get("elapsed_time_sec"))
        result = adapter.update(
            frame["wavelength"],
            frame["intensity"],
            source_timestamp_sec=timestamp,
        )
        if not result.get("ok"):
            errors.append(
                {
                    "capture_index": capture_index,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                }
            )
            continue
        force = dict(result.get("force_fz") or {})
        contact = dict(result.get("contact") or {})
        position = dict(result.get("position") or {})
        overlay[capture_index] = {
            "estimated_fz_n": _safe_float(
                force.get("estimated_n", result.get("estimated_force_fz_n"))
            ),
            "raw_estimated_fz_n": _safe_float(force.get("raw_estimated_n")),
            "contact_label": contact.get("label"),
            "position_label": position.get("label"),
            "inference_latency_ms": _safe_float(
                result.get("inference_latency_ms")
            ),
            "model_source": result.get("recognition_source") or "current_model_replay",
        }

    coverage = len(overlay) / max(len(trace_rows), 1)
    if not overlay:
        return {
            "ok": False,
            "source": "current_model_replay",
            "status": "model_replay_failed",
            "reason": errors[:5],
        }

    model_modified = model_path.stat().st_mtime
    session_timestamps = [
        value
        for row in trace_rows
        if (value := _safe_float(row.get("timeline_timestamp_epoch_sec"))) is not None
    ]
    session_started = min(session_timestamps, default=None)
    post_training_recording = bool(
        session_started is not None and session_started > model_modified
    )
    validity = (
        "post_training_recording_replay"
        if post_training_recording
        else "model_replay_not_independent_until_proven"
    )
    return {
        "ok": True,
        "overlay": overlay,
        "source": "current_model_replay",
        "label": "Current model replay (session baseline)",
        "evaluation_validity": validity,
        "provenance": {
            "model_artifact_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "peak_config_path": str(peak_config_path.resolve()),
            "matched_row_count": len(overlay),
            "trace_row_count": len(trace_rows),
            "coverage_ratio": coverage,
            "inference_error_count": len(errors),
            "inference_errors_preview": errors[:5],
            "baseline": baseline_info,
            "session_started_epoch_sec": session_started,
            "model_modified_epoch_sec": model_modified,
            "post_training_recording": post_training_recording,
            "force_sensor_is_runtime_model_input": False,
            "note": (
                "The live adapter is not touched. Replay uses the recorded optical "
                "spectra and an isolated model instance."
            ),
        },
    }


def recorded_runtime_evidence(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlay = {
        int(row["capture_index"]): {
            "estimated_fz_n": row.get("optical_estimated_fz_n"),
            "raw_estimated_fz_n": row.get("optical_raw_estimated_fz_n"),
            "contact_label": row.get("contact_label"),
            "position_label": row.get("predicted_position_label"),
            "inference_latency_ms": row.get("model_inference_latency_ms"),
            "model_source": row.get("model_source"),
        }
        for row in trace_rows
        if _safe_int(row.get("capture_index")) is not None
    }
    sources = sorted(
        {str(row["model_source"]) for row in trace_rows if row.get("model_source")}
    )
    return {
        "ok": True,
        "overlay": overlay,
        "source": "recorded_runtime",
        "label": "Historical recorded runtime (capture-time model)",
        "evaluation_validity": "historical_capture_time_output_not_current_model",
        "provenance": {
            "model_sources": sources,
            "matched_row_count": len(overlay),
            "trace_row_count": len(trace_rows),
            "coverage_ratio": len(overlay) / max(len(trace_rows), 1),
            "force_sensor_is_runtime_model_input": False,
            "note": (
                "Values were saved during acquisition and may come from an older "
                "model or older baseline state."
            ),
        },
    }


def resolve_measurement_estimate_evidence(
    session_dir: Path,
    trace_rows: list[dict[str, Any]],
    requested_source: str = "best_available",
    *,
    outputs_root: Path,
    model_path: Path | None = None,
    peak_config_path: Path | None = None,
    runtime_recovery_config: dict[str, Any] | None = None,
    runtime_gate_config: dict[str, Any] | None = None,
    baseline_frame_count: int = 10,
    baseline_strategy: str = "session_initial_stable_median",
    baseline_minimum_stable_frames: int = 5,
    baseline_stability_mad_multiplier: float = 3.5,
    adapter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Resolve one coherent Measurement estimate curve.

    Grouped out-of-fold evidence is preferred for model evaluation. A replay
    through the current model is second, while the values recorded during the
    original acquisition remain an explicitly historical fallback. Sources are
    never combined frame by frame.
    """

    requested = str(requested_source or "best_available").strip().lower()
    if requested not in EVIDENCE_SOURCES:
        return {
            "ok": False,
            "source": requested,
            "requested_source": requested,
            "status": "measurement_estimate_source_invalid",
            "reason": f"unsupported estimate source: {requested}",
        }

    def grouped_oof() -> dict[str, Any]:
        return load_grouped_oof_evidence(
            Path(session_dir),
            trace_rows,
            Path(outputs_root),
        )

    def current_replay() -> dict[str, Any]:
        if model_path is None or peak_config_path is None:
            return {
                "ok": False,
                "source": "current_model_replay",
                "status": "current_model_replay_not_configured",
                "reason": "model_path and peak_config_path are required",
            }
        return replay_current_model_evidence(
            Path(session_dir),
            trace_rows,
            Path(model_path),
            Path(peak_config_path),
            runtime_recovery_config=runtime_recovery_config,
            runtime_gate_config=runtime_gate_config,
            baseline_frame_count=baseline_frame_count,
            baseline_strategy=baseline_strategy,
            baseline_minimum_stable_frames=baseline_minimum_stable_frames,
            baseline_stability_mad_multiplier=baseline_stability_mad_multiplier,
            adapter_factory=adapter_factory,
        )

    resolvers = {
        "grouped_oof": grouped_oof,
        "current_model_replay": current_replay,
        "recorded_runtime": lambda: recorded_runtime_evidence(trace_rows),
    }
    if requested != "best_available":
        result = dict(resolvers[requested]())
        result["requested_source"] = requested
        return result

    attempts: list[dict[str, Any]] = []
    for source in ("grouped_oof", "current_model_replay", "recorded_runtime"):
        result = dict(resolvers[source]())
        attempts.append(
            {
                "source": source,
                "ok": bool(result.get("ok")),
                "status": result.get("status"),
                "reason": result.get("reason"),
            }
        )
        if result.get("ok"):
            result["requested_source"] = requested
            result["resolution_attempts"] = attempts
            return result
    return {
        "ok": False,
        "source": requested,
        "requested_source": requested,
        "status": "measurement_estimate_source_unavailable",
        "reason": "No usable force-estimate evidence source was found.",
        "resolution_attempts": attempts,
    }


__all__ = [
    "EVIDENCE_SOURCES",
    "load_grouped_oof_evidence",
    "recorded_runtime_evidence",
    "replay_current_model_evidence",
    "resolve_measurement_estimate_evidence",
]
