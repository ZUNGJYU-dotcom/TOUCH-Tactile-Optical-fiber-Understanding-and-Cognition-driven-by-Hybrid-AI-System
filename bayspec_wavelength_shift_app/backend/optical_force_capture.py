"""In-process BaySpec spectrum and PX6D reference-force session recorder."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable


AVAILABLE_OUTPUT_STREAMS = ("spectrum", "response", "force")
OUTPUT_STREAM_ALIASES = {
    "spectrum": "spectrum",
    "optical_spectrum": "spectrum",
    "response": "response",
    "tactile_response": "response",
    "time_series": "response",
    "timeseries": "response",
    "force": "force",
    "px6d": "force",
    "px6d_force": "force",
}


SUMMARY_FIELDS = [
    "capture_index",
    "timeline_timestamp_epoch_sec",
    "elapsed_time_sec",
    "capture_timestamp_epoch_sec",
    "position_label",
    "action_label",
    "trial_id",
    "spectrum_source",
    "spectrum_frame_id",
    "spectrum_timestamp_epoch_sec",
    "spectrum_points",
    "force_timestamp_epoch_sec",
    "sync_method",
    "sync_quality",
    "sync_offset_ms",
    "sync_within_target",
    "calibration_sync_ok",
    "force_sample_count",
    "fx_raw_n",
    "fy_raw_n",
    "fz_raw_n",
    "mx_raw_nm",
    "my_raw_nm",
    "mz_raw_nm",
    "fx_zeroed_n",
    "fy_zeroed_n",
    "fz_zeroed_n",
    "mx_zeroed_nm",
    "my_zeroed_nm",
    "mz_zeroed_nm",
    "fx_filtered_n",
    "fy_filtered_n",
    "fz_filtered_n",
    "mx_filtered_nm",
    "my_filtered_nm",
    "mz_filtered_nm",
    "reference_fz_n",
    "reference_fz_display_n",
    "median_reference_fz_n",
    "filtered_reference_fz_n",
    "drift_offset_n",
    "drift_corrected_reference_fz_n",
    "conditioned_reference_fz_n",
    "force_fz_n",
    "stationary_detected",
    "auto_zero_drift_active",
    "force_filter_status",
    "force_resultant_n",
    "shear_resultant_n",
    "moment_resultant_nm",
    "force_utilization_percent",
    "moment_utilization_percent",
    "filtered_force_resultant_n",
    "filtered_shear_resultant_n",
    "filtered_moment_resultant_nm",
    "model_source",
    "model_status",
    "model_ready",
    "model_inference_latency_ms",
    "optical_estimated_fz_n",
    "optical_raw_estimated_fz_n",
    "optical_force_estimate_gated",
    "optical_force_estimate_available",
    "optical_force_estimate_source",
    "optical_force_estimate_unit",
    "optical_force_review_needed",
    "optical_force_review_reasons",
    "predicted_contact_label",
    "predicted_position_label",
    "predicted_response_level",
    "predicted_response_confidence",
]

SPECTRUM_FIELDS = [
    "capture_index",
    "timeline_timestamp_epoch_sec",
    "elapsed_time_sec",
    "spectrum_timestamp_epoch_sec",
    "spectrum_source",
    "spectrum_frame_id",
    "point_index",
    "wavelength_nm",
    "intensity_counts",
    "baseline_intensity_counts",
    "normalized_intensity_ratio",
    "normalization_method",
    "normalization_status",
    "position_label",
    "action_label",
    "trial_id",
]

RESPONSE_FIELDS = [
    "capture_index",
    "timeline_timestamp_epoch_sec",
    "elapsed_time_sec",
    "model_timestamp_epoch_sec",
    "model_source",
    "model_status",
    "model_ready",
    "model_inference_latency_ms",
    "optical_estimated_fz_n",
    "optical_raw_estimated_fz_n",
    "optical_force_estimate_gated",
    "optical_force_estimate_available",
    "optical_force_estimate_source",
    "optical_force_estimate_unit",
    "optical_force_review_needed",
    "optical_force_review_reasons",
    "contact_label",
    "contact_confidence",
    "contact_probability",
    "no_contact_probability",
    "predicted_position_label",
    "position_confidence",
    "response_level",
    "response_level_confidence",
    "response_level_raw_label",
    "response_level_decision_rule",
    "light_probability",
    "normal_probability",
    "hard_probability",
    "operational_state",
    "release_latched",
    "runtime_baseline_revision",
    "position_label",
    "action_label",
    "trial_id",
]

FORCE_FIELDS = [
    "capture_index",
    "timeline_timestamp_epoch_sec",
    "elapsed_time_sec",
    "force_timestamp_epoch_sec",
    "sync_method",
    "sync_quality",
    "sync_offset_ms",
    "sync_within_target",
    "calibration_sync_ok",
    "force_sample_count",
    "force_sequence_start",
    "force_sequence_end",
    "fx_raw_n",
    "fy_raw_n",
    "fz_raw_n",
    "mx_raw_nm",
    "my_raw_nm",
    "mz_raw_nm",
    "fx_zeroed_n",
    "fy_zeroed_n",
    "fz_zeroed_n",
    "mx_zeroed_nm",
    "my_zeroed_nm",
    "mz_zeroed_nm",
    "fx_filtered_n",
    "fy_filtered_n",
    "fz_filtered_n",
    "mx_filtered_nm",
    "my_filtered_nm",
    "mz_filtered_nm",
    "reference_fz_n",
    "reference_fz_display_n",
    "median_reference_fz_n",
    "filtered_reference_fz_n",
    "drift_offset_n",
    "drift_corrected_reference_fz_n",
    "conditioned_reference_fz_n",
    "force_fz_n",
    "stationary_detected",
    "auto_zero_drift_active",
    "force_filter_status",
    "force_resultant_n",
    "shear_resultant_n",
    "moment_resultant_nm",
    "force_utilization_percent",
    "moment_utilization_percent",
    "filtered_force_resultant_n",
    "filtered_shear_resultant_n",
    "filtered_moment_resultant_nm",
    "position_label",
    "action_label",
    "trial_id",
]


def _safe_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_")
    return text[:48] or fallback


def _continuous_force_fz_n(force: dict[str, Any]) -> float | None:
    """Return the continuous non-negative compression Fz used as force truth."""
    for key in (
        "conditioned_reference_fz_n",
        "reference_fz_display_n",
        "drift_corrected_reference_fz_n",
        "reference_fz_n",
    ):
        try:
            value = float(force.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return max(0.0, value)
    return None


def _optical_force_export_fields(model: dict[str, Any]) -> dict[str, Any]:
    """Flatten the optical-only Fz estimate without changing model semantics."""
    force = model.get("force_fz") or {}
    uncertainty = model.get("uncertainty") or {}
    estimated = model.get("estimated_force_fz_n", force.get("estimated_n"))
    raw_estimated = force.get("raw_estimated_n")

    def finite_or_none(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    estimated_value = finite_or_none(estimated)
    raw_estimated_value = finite_or_none(raw_estimated)
    reasons = uncertainty.get("reasons") or []
    if isinstance(reasons, str):
        reason_text = reasons
    else:
        reason_text = "|".join(str(reason) for reason in reasons if reason)
    return {
        "model_inference_latency_ms": finite_or_none(
            model.get("inference_latency_ms")
        ),
        "optical_estimated_fz_n": estimated_value,
        "optical_raw_estimated_fz_n": raw_estimated_value,
        "optical_force_estimate_gated": force.get("gated"),
        "optical_force_estimate_available": estimated_value is not None,
        "optical_force_estimate_source": (
            force.get("runtime_input")
            or model.get("runtime_input")
            or model.get("recognition_source")
            or model.get("model_source")
        ),
        "optical_force_estimate_unit": force.get("unit") or "N",
        "optical_force_review_needed": uncertainty.get("review_needed"),
        "optical_force_review_reasons": reason_text,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _sanitize_serializable(value: Any) -> Any:
    """Replace non-finite numeric leaves before writing JSON or CSV."""
    if hasattr(value, "item"):
        try:
            return _sanitize_serializable(value.item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_serializable(item) for item in value]
    return value


def _finite_numeric_list(values: Any) -> list[float] | None:
    if not isinstance(values, list) or not values:
        return None
    normalized: list[float] = []
    try:
        for value in values:
            number = float(value)
            if not math.isfinite(number):
                return None
            normalized.append(number)
    except (TypeError, ValueError):
        return None
    return normalized


def _csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_serializable(value) for key, value in row.items()}


def _csv_rows_block(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    """Serialize complete CSV rows before touching any session file."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writerows(_csv_safe_row(row) for row in rows)
    return buffer.getvalue()


def _flush_and_sync(handle: Any) -> None:
    """Flush Python and operating-system buffers when the handle is durable."""
    handle.flush()
    try:
        file_descriptor = handle.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError):
        # StringIO and other in-memory test handles have no durable backing.
        return
    os.fsync(file_descriptor)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    encoded = json.dumps(
        _sanitize_serializable(payload),
        ensure_ascii=False,
        indent=2,
        default=_json_default,
        allow_nan=False,
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            _flush_and_sync(handle)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_text_transaction(writes: list[tuple[Any, str]]) -> None:
    """Best-effort all-or-nothing append across the files for one capture frame."""
    checkpoints: list[tuple[Any, int]] = []
    try:
        checkpoints = [(handle, handle.tell()) for handle, _payload in writes]
        for handle, payload in writes:
            handle.write(payload)
        for handle, _payload in writes:
            _flush_and_sync(handle)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for handle, checkpoint in checkpoints:
            try:
                handle.seek(checkpoint)
                handle.truncate()
                _flush_and_sync(handle)
            except BaseException as rollback_exc:
                rollback_errors.append(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        if rollback_errors:
            raise OSError(
                "capture_frame_write_failed_and_rollback_incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise


class OpticalForceCaptureManager:
    """Write user-selected TOUCH streams on one canonical capture timeline."""

    def __init__(
        self,
        *,
        output_root: Path,
        frame_provider: Callable[[], dict[str, Any]],
        force_provider: Callable[[dict[str, Any] | None], dict[str, Any]],
        force_status_provider: Callable[[], dict[str, Any]],
        model_provider: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        provenance_provider: Callable[[], dict[str, Any]] | None = None,
        poll_interval_sec: float = 0.05,
        require_software_tare: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.frame_provider = frame_provider
        self.force_provider = force_provider
        self.force_status_provider = force_status_provider
        self.model_provider = model_provider
        self.provenance_provider = provenance_provider
        self.poll_interval_sec = max(0.01, min(1.0, float(poll_interval_sec)))
        self.require_software_tare = bool(require_software_tare)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_in_progress = False
        self._start_cancel_requested = False
        self._start_done_event = threading.Event()
        self._start_done_event.set()
        self._provenance_at_start: dict[str, Any] | None = None
        self._provenance_start_error: str | None = None
        self._provenance_latest: dict[str, Any] | None = None
        self._provenance_latest_error: str | None = None
        self._recovered_interrupted_sessions = self._recover_interrupted_sessions()
        self._state = self._idle_state()

    def _idle_state(self) -> dict[str, Any]:
        return {
            "running": False,
            "capture_status": "idle",
            "session_id": None,
            "output_directory": None,
            "default_output_root": str(self.output_root),
            "requested_output_root": str(self.output_root),
            "started_at_epoch_sec": None,
            "ended_at_epoch_sec": None,
            "position_label": "unlabeled",
            "action_label": "unlabeled",
            "trial_id": "trial_001",
            "operator_note": "",
            "selected_outputs": list(AVAILABLE_OUTPUT_STREAMS),
            "captured_timeline_frames": 0,
            "captured_paired_frames": 0,
            "spectrum_frame_rows": 0,
            "spectrum_sample_rows": 0,
            "response_rows": 0,
            "force_rows": 0,
            "eligible_spectrum_frames": 0,
            "skipped_duplicate_spectrum_frames": 0,
            "polls_without_valid_spectrum": 0,
            "invalid_spectrum_frames": 0,
            "invalid_force_samples": 0,
            "timeline_timestamp_fallbacks": 0,
            "frames_missing_force_reference": 0,
            "frames_outside_force_sync_tolerance": 0,
            "last_spectrum_frame_id": None,
            "last_sync_quality": None,
            "last_sync_offset_ms": None,
            "maximum_absolute_sync_offset_ms": None,
            "timeline_start_epoch_sec": None,
            "last_error": None,
            "recovered_interrupted_session_count": len(
                getattr(self, "_recovered_interrupted_sessions", [])
            ),
            "recovered_interrupted_session_ids": list(
                getattr(self, "_recovered_interrupted_sessions", [])
            ),
        }

    def _recover_interrupted_sessions(self) -> list[str]:
        """Mark sessions left active by an earlier process as interrupted."""
        if not self.output_root.is_dir():
            return []
        active_statuses = {
            "waiting_for_optical_frame",
            "waiting_for_force_sample",
            "waiting_for_force_reference",
            "recording_selected_streams",
            "force_sync_outside_calibration_tolerance",
            "invalid_optical_frame",
            "invalid_force_sample",
        }
        recovered: list[str] = []
        for output_dir in sorted(self.output_root.iterdir()):
            if not output_dir.is_dir():
                continue
            metadata_path = output_dir / "session_metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            capture_status = str(payload.get("capture_status") or "")
            if not bool(payload.get("running")) and capture_status not in active_statuses:
                continue
            recovered_at = time.time()
            session_id = str(payload.get("session_id") or output_dir.name)
            recovery = {
                "detected_at_epoch_sec": recovered_at,
                "previous_capture_status": capture_status or None,
                "reason": "previous_process_terminated_before_clean_stop",
            }
            payload.update(
                {
                    "running": False,
                    "capture_status": "interrupted_recovered",
                    "ended_at_epoch_sec": payload.get("ended_at_epoch_sec")
                    or recovered_at,
                    "last_error": payload.get("last_error")
                    or "previous_process_terminated_before_clean_stop",
                    "recovery": recovery,
                }
            )
            try:
                _write_json_atomic(metadata_path, payload)
                _write_json_atomic(
                    output_dir / "capture_journal.json",
                    {
                        "schema_version": "touch_capture_journal_v1",
                        "session_id": session_id,
                        "running": False,
                        "capture_status": "interrupted_recovered",
                        "captured_timeline_frames": int(
                            payload.get("captured_timeline_frames") or 0
                        ),
                        "last_spectrum_frame_id": payload.get(
                            "last_spectrum_frame_id"
                        ),
                        "updated_at_epoch_sec": recovered_at,
                        "recovery": recovery,
                    },
                )
            except OSError:
                continue
            recovered.append(session_id)
        return recovered

    def _journal_payload_locked(self) -> dict[str, Any]:
        return {
            "schema_version": "touch_capture_journal_v1",
            "session_id": self._state.get("session_id"),
            "running": bool(self._state.get("running")),
            "capture_status": self._state.get("capture_status"),
            "captured_timeline_frames": int(
                self._state.get("captured_timeline_frames") or 0
            ),
            "captured_paired_frames": int(
                self._state.get("captured_paired_frames") or 0
            ),
            "started_at_epoch_sec": self._state.get("started_at_epoch_sec"),
            "ended_at_epoch_sec": self._state.get("ended_at_epoch_sec"),
            "last_spectrum_frame_id": self._state.get("last_spectrum_frame_id"),
            "last_sync_offset_ms": self._state.get("last_sync_offset_ms"),
            "last_error": self._state.get("last_error"),
            "updated_at_epoch_sec": time.time(),
        }

    def _write_journal_locked(self) -> None:
        output_directory = self._state.get("output_directory")
        if not output_directory:
            return
        _write_json_atomic(
            Path(str(output_directory)) / "capture_journal.json",
            self._journal_payload_locked(),
        )

    @staticmethod
    def _normalize_selected_outputs(values: Any) -> list[str]:
        if values is None:
            return list(AVAILABLE_OUTPUT_STREAMS)
        if isinstance(values, str):
            candidates = [item.strip() for item in values.split(",")]
        elif isinstance(values, (list, tuple, set)):
            candidates = list(values)
        else:
            raise ValueError("selected_outputs must be a list or comma-separated string")
        selected: list[str] = []
        unknown: list[str] = []
        for candidate in candidates:
            token = str(candidate or "").strip().lower()
            normalized = OUTPUT_STREAM_ALIASES.get(token)
            if normalized and normalized not in selected:
                selected.append(normalized)
            elif token and normalized is None:
                unknown.append(token)
        if unknown:
            raise ValueError(
                "unknown selected output(s): " + ", ".join(dict.fromkeys(unknown))
            )
        if not selected:
            raise ValueError("select at least one output: spectrum, response, or force")
        return [name for name in AVAILABLE_OUTPUT_STREAMS if name in selected]

    @staticmethod
    def _resolve_requested_output_root(
        requested: str | Path | None,
        default_root: Path,
    ) -> Path:
        text = str(requested or "").strip()
        candidate = Path(text).expanduser() if text else Path(default_root)
        candidate = candidate.resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        if not candidate.is_dir():
            raise NotADirectoryError(str(candidate))
        probe = candidate / f".touch_write_test_{time.time_ns()}.tmp"
        try:
            probe.write_text("TOUCH", encoding="ascii")
        finally:
            probe.unlink(missing_ok=True)
        return candidate

    def start(
        self,
        *,
        position_label: str = "unlabeled",
        action_label: str = "unlabeled",
        trial_id: str = "trial_001",
        operator_note: str = "",
        output_root: str | Path | None = None,
        selected_outputs: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._start_in_progress:
                return {"ok": False, "status": "capture_start_in_progress", **self.status()}
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "status": "capture_already_running", **self.status()}
            previous_state = dict(self._state)
            self._start_in_progress = True
            self._start_cancel_requested = False
            self._start_done_event.clear()
        result: dict[str, Any]
        try:
            result = self._start_reserved(
                position_label=position_label,
                action_label=action_label,
                trial_id=trial_id,
                operator_note=operator_note,
                output_root=output_root,
                selected_outputs=selected_outputs,
            )
        except Exception as exc:
            with self._lock:
                partial_session_started = self._state.get("started_at_epoch_sec") != (
                    previous_state.get("started_at_epoch_sec")
                )
                if partial_session_started:
                    self._state["running"] = False
                    self._state["capture_status"] = "capture_error"
                    self._state["ended_at_epoch_sec"] = time.time()
                    self._state["last_error"] = (
                        f"capture_start_failed: {type(exc).__name__}: {exc}"
                    )
                    try:
                        self._write_metadata_locked()
                    except Exception as metadata_exc:
                        self._state["last_error"] += (
                            "; metadata_write_failed: "
                            f"{type(metadata_exc).__name__}: {metadata_exc}"
                        )
                else:
                    # A pre-session failure must not rewrite the status and
                    # output path of the most recently completed experiment.
                    self._state = previous_state
            result = {
                "ok": False,
                "status": "capture_start_failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "last_start_error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            with self._lock:
                self._start_in_progress = False
                self._start_cancel_requested = False
                self._start_done_event.set()
        # _start_reserved builds its response while the reservation flag is
        # still set. Refresh the lifecycle fields only after the reservation
        # has been released so API clients never observe a completed start as
        # still being in progress.
        return {**result, **self.status()}

    def _start_cancelled(self) -> bool:
        with self._lock:
            return self._start_cancel_requested

    def _snapshot_provenance(self) -> tuple[dict[str, Any] | None, str | None]:
        if self.provenance_provider is None:
            return None, "provenance_provider_not_configured"
        try:
            snapshot = self.provenance_provider()
            if not isinstance(snapshot, dict):
                raise TypeError("provenance provider must return a dictionary")
            return _sanitize_serializable(snapshot), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _cancelled_start_result(self, output_dir: Path | None = None) -> dict[str, Any]:
        if output_dir is not None:
            try:
                output_dir.rmdir()
            except OSError:
                # A non-empty directory contains useful failure evidence and is
                # intentionally retained instead of being deleted recursively.
                pass
        return {"ok": False, "status": "capture_start_cancelled", **self.status()}

    def _start_reserved(
        self,
        *,
        position_label: str,
        action_label: str,
        trial_id: str,
        operator_note: str,
        output_root: str | Path | None,
        selected_outputs: Any,
    ) -> dict[str, Any]:
        try:
            selected = self._normalize_selected_outputs(selected_outputs)
        except ValueError as exc:
            return {
                "ok": False,
                "status": "capture_output_selection_invalid",
                "reason": str(exc),
                "available_outputs": list(AVAILABLE_OUTPUT_STREAMS),
            }
        if "force" in selected:
            try:
                force_status = dict(self.force_status_provider() or {})
            except Exception as exc:
                return {
                    "ok": False,
                    "status": "px6d_status_unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            if self._start_cancelled():
                return self._cancelled_start_result()
            if not force_status.get("connected"):
                return {
                    "ok": False,
                    "status": "px6d_not_connected",
                    "force_status": force_status,
                }
            if self.require_software_tare and not force_status.get("tare_ready"):
                return {
                    "ok": False,
                    "status": "px6d_software_tare_required",
                    "force_status": force_status,
                }

        try:
            selected_output_root = self._resolve_requested_output_root(
                output_root,
                self.output_root,
            )
        except (OSError, ValueError) as exc:
            return {
                "ok": False,
                "status": "capture_output_directory_invalid",
                "requested_output_root": str(output_root or ""),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if self._start_cancelled():
            return self._cancelled_start_result()

        started = time.time()
        timestamp = datetime.fromtimestamp(started).strftime("%Y%m%d_%H%M%S")
        position = _safe_token(position_label, "unlabeled")
        action = _safe_token(action_label, "unlabeled")
        trial = _safe_token(trial_id, "trial_001")
        base_session_id = f"{timestamp}_{position}_{action}_{trial}"
        session_id = base_session_id
        output_dir = selected_output_root / session_id
        suffix = 2
        while output_dir.exists():
            session_id = f"{base_session_id}_{suffix:02d}"
            output_dir = selected_output_root / session_id
            suffix += 1
        output_dir.mkdir(parents=True, exist_ok=False)
        provenance_at_start, provenance_start_error = self._snapshot_provenance()

        with self._lock:
            if self._start_cancel_requested:
                return self._cancelled_start_result(output_dir)
            self._state = {
                **self._idle_state(),
                "running": True,
                "capture_status": (
                    "waiting_for_optical_frame"
                    if "spectrum" in selected or "response" in selected
                    else "waiting_for_force_sample"
                ),
                "session_id": session_id,
                "output_directory": str(output_dir),
                "requested_output_root": str(selected_output_root),
                "started_at_epoch_sec": started,
                "position_label": str(position_label or "unlabeled"),
                "action_label": str(action_label or "unlabeled"),
                "trial_id": str(trial_id or "trial_001"),
                "operator_note": str(operator_note or ""),
                "selected_outputs": selected,
            }
            self._provenance_at_start = provenance_at_start
            self._provenance_start_error = provenance_start_error
            self._provenance_latest = provenance_at_start
            self._provenance_latest_error = provenance_start_error
            self._stop_event.clear()
            try:
                self._write_metadata_locked(
                    refresh_provenance=False,
                    session_initialized=True,
                )
                self._write_journal_locked()
            except Exception as exc:
                self._state["running"] = False
                self._state["capture_status"] = "capture_error"
                self._state["ended_at_epoch_sec"] = time.time()
                self._state["last_error"] = (
                    "capture_session_initialization_failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return {
                    "ok": False,
                    "status": "capture_session_initialization_failed",
                    **self.status(),
                }
            worker = threading.Thread(
                target=self._run,
                args=(output_dir,),
                name="touch-optical-force-capture",
                daemon=True,
            )
            self._thread = worker
            try:
                worker.start()
            except Exception as exc:
                self._thread = None
                self._state["running"] = False
                self._state["capture_status"] = "capture_error"
                self._state["ended_at_epoch_sec"] = time.time()
                self._state["last_error"] = f"capture_thread_start_failed: {type(exc).__name__}: {exc}"
                try:
                    self._write_metadata_locked()
                    self._write_journal_locked()
                except Exception as metadata_exc:
                    self._state["last_error"] += (
                        "; metadata_write_failed: "
                        f"{type(metadata_exc).__name__}: {metadata_exc}"
                    )
                return {
                    "ok": False,
                    "status": "capture_thread_start_failed",
                    **self.status(),
                }
        return {"ok": True, "status": "capture_started", **self.status()}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            start_in_progress = self._start_in_progress
            if start_in_progress:
                self._start_cancel_requested = True
                self._stop_event.set()
        if start_in_progress and not self._start_done_event.wait(timeout=5.0):
            return {
                "ok": False,
                "status": "capture_start_cancel_timeout",
                "reason": "capture start did not acknowledge cancellation before timeout",
                **self.status(),
            }
        with self._lock:
            thread = self._thread
            running = bool(thread is not None and thread.is_alive())
        if not running:
            current_status = self.status()
            capture_failed = bool(current_status.get("last_error")) or str(
                current_status.get("capture_status") or ""
            ) in {"capture_error", "stop_timeout"}
            return {
                "ok": not capture_failed,
                "status": (
                    "capture_failed"
                    if capture_failed
                    else "capture_start_cancelled"
                    if start_in_progress
                    else "capture_not_running"
                ),
                **current_status,
            }
        self._stop_event.set()
        thread.join(timeout=4.0)
        if thread.is_alive():
            with self._lock:
                self._state["last_error"] = "capture_thread_stop_timeout"
                self._state["capture_status"] = "stop_timeout"
                self._state["running"] = True
            return {"ok": False, "status": "capture_stop_timeout", **self.status()}
        metadata_error: str | None = None
        with self._lock:
            self._state["running"] = False
            self._state["ended_at_epoch_sec"] = time.time()
            if self._state["captured_paired_frames"] > 0:
                self._state["capture_status"] = "complete"
            elif self._state["last_error"]:
                self._state["capture_status"] = "capture_error"
            else:
                self._state["capture_status"] = "no_paired_frames"
            try:
                self._write_metadata_locked()
                self._write_journal_locked()
            except Exception as exc:
                metadata_error = f"metadata_write_failed: {type(exc).__name__}: {exc}"
                existing_error = str(self._state.get("last_error") or "").strip()
                self._state["last_error"] = (
                    f"{existing_error}; {metadata_error}"
                    if existing_error
                    else metadata_error
                )
                self._state["capture_status"] = "capture_error"
        if metadata_error is not None:
            return {
                "ok": False,
                "status": "capture_metadata_write_failed",
                **self.status(),
            }
        return {"ok": True, "status": "capture_stopped", **self.status()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._state)
            payload["start_in_progress"] = self._start_in_progress
            payload["start_cancel_requested"] = self._start_cancel_requested
            payload["worker_alive"] = bool(self._thread is not None and self._thread.is_alive())
        denominator = int(payload.get("eligible_spectrum_frames") or 0)
        paired = int(payload.get("captured_timeline_frames") or 0)
        payload["paired_frame_ratio"] = paired / denominator if denominator else None
        payload["force_semantics"] = "PX6D_reference_Fz_not_optical_force_prediction"
        payload["force_target_mode"] = "continuous_fz_regression"
        payload["force_target_field"] = "force_fz_n"
        payload["force_class_bins_enabled"] = False
        selected = set(payload.get("selected_outputs") or [])
        payload["output_format"] = "selectable_aligned_csv_streams_plus_audit_sidecars"
        payload["timeline_basis"] = (
            "spectrum_ingested_at_host_epoch_clock"
            if {"spectrum", "response"} & selected
            else "px6d_force_host_epoch_clock"
        )
        payload["available_outputs"] = list(AVAILABLE_OUTPUT_STREAMS)
        payload["output_files"] = {
            "spectrum_timeseries": "spectrum_timeseries.csv" if "spectrum" in selected else None,
            "tactile_response_timeseries": (
                "tactile_response_timeseries.csv" if "response" in selected else None
            ),
            "force_timeseries": "force_timeseries.csv" if "force" in selected else None,
            "lossless_combined_frames": "synchronized_frames.jsonl",
            "compatibility_summary": "frame_summary.csv",
            "metadata": "session_metadata.json",
            "recovery_journal": "capture_journal.json",
        }
        return payload

    def _increment(self, field: str, amount: int = 1) -> None:
        with self._lock:
            self._state[field] = int(self._state.get(field) or 0) + amount

    def _run(self, output_dir: Path) -> None:
        frames_path = output_dir / "synchronized_frames.jsonl"
        summary_path = output_dir / "frame_summary.csv"
        spectrum_path = output_dir / "spectrum_timeseries.csv"
        response_path = output_dir / "tactile_response_timeseries.csv"
        force_path = output_dir / "force_timeseries.csv"
        with self._lock:
            selected = set(self._state.get("selected_outputs") or AVAILABLE_OUTPUT_STREAMS)
        optical_timeline = bool({"spectrum", "response"} & selected)
        last_written_key: tuple[Any, ...] | None = None
        last_candidate_key: tuple[Any, ...] | None = None
        last_missing_force_key: tuple[Any, ...] | None = None
        last_outside_sync_key: tuple[Any, ...] | None = None
        last_invalid_spectrum_key: tuple[Any, ...] | None = None
        last_invalid_force_key: tuple[Any, ...] | None = None
        try:
            with ExitStack() as stack:
                jsonl_handle = stack.enter_context(frames_path.open("w", encoding="utf-8"))
                summary_handle = stack.enter_context(
                    summary_path.open("w", newline="", encoding="utf-8-sig")
                )
                summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
                summary_writer.writeheader()
                spectrum_handle = None
                spectrum_writer = None
                if "spectrum" in selected:
                    spectrum_handle = stack.enter_context(
                        spectrum_path.open("w", newline="", encoding="utf-8-sig")
                    )
                    spectrum_writer = csv.DictWriter(spectrum_handle, fieldnames=SPECTRUM_FIELDS)
                    spectrum_writer.writeheader()
                response_handle = None
                response_writer = None
                if "response" in selected:
                    response_handle = stack.enter_context(
                        response_path.open("w", newline="", encoding="utf-8-sig")
                    )
                    response_writer = csv.DictWriter(response_handle, fieldnames=RESPONSE_FIELDS)
                    response_writer.writeheader()
                force_handle = None
                force_writer = None
                if "force" in selected:
                    force_handle = stack.enter_context(
                        force_path.open("w", newline="", encoding="utf-8-sig")
                    )
                    force_writer = csv.DictWriter(force_handle, fieldnames=FORCE_FIELDS)
                    force_writer.writeheader()
                for handle in (
                    jsonl_handle,
                    summary_handle,
                    spectrum_handle,
                    response_handle,
                    force_handle,
                ):
                    if handle is not None:
                        _flush_and_sync(handle)
                while not self._stop_event.is_set():
                    latest: dict[str, Any] | None = None
                    wavelengths: list[Any] = []
                    intensities: list[Any] = []
                    force: dict[str, Any] = {}
                    model: dict[str, Any] = {}
                    if optical_timeline:
                        frame_payload = dict(self.frame_provider() or {})
                        candidate = frame_payload.get("latest")
                        wavelengths_value = (
                            candidate.get("wavelength_nm") if isinstance(candidate, dict) else None
                        )
                        intensities_value = (
                            candidate.get("intensity") if isinstance(candidate, dict) else None
                        )
                        if not isinstance(candidate, dict):
                            self._increment("polls_without_valid_spectrum")
                            self._set_status("waiting_for_optical_frame")
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        latest = candidate
                        key = (
                            latest.get("source"),
                            latest.get("frame_id"),
                            latest.get("ingested_at")
                            or latest.get("timestamp_epoch_sec")
                            or latest.get("timestamp"),
                        )
                        normalized_wavelengths = _finite_numeric_list(wavelengths_value)
                        normalized_intensities = _finite_numeric_list(intensities_value)
                        if (
                            normalized_wavelengths is None
                            or normalized_intensities is None
                            or len(normalized_wavelengths) != len(normalized_intensities)
                        ):
                            if key != last_invalid_spectrum_key:
                                self._increment("invalid_spectrum_frames")
                                last_invalid_spectrum_key = key
                            self._set_status("invalid_optical_frame")
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        wavelengths = normalized_wavelengths
                        intensities = normalized_intensities
                        if key == last_written_key:
                            self._increment("skipped_duplicate_spectrum_frames")
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        if key != last_candidate_key:
                            self._increment("eligible_spectrum_frames")
                            last_candidate_key = key
                        try:
                            timeline_timestamp = float(key[2])
                        except (TypeError, ValueError):
                            timeline_timestamp = time.time()
                            self._increment("timeline_timestamp_fallbacks")
                        if not math.isfinite(timeline_timestamp):
                            timeline_timestamp = time.time()
                            self._increment("timeline_timestamp_fallbacks")
                        if "force" in selected:
                            force = dict(self.force_provider(latest) or {})
                            force_available = bool(force.get("ok")) and (
                                not self.require_software_tare or bool(force.get("tare_ready"))
                            )
                            if not force_available:
                                if key != last_missing_force_key:
                                    self._increment("frames_missing_force_reference")
                                    last_missing_force_key = key
                                self._set_status(
                                    str(force.get("status") or "waiting_for_force_reference")
                                )
                                self._stop_event.wait(self.poll_interval_sec)
                                continue
                            try:
                                sync_offset_ms = float(force.get("sync_offset_ms"))
                            except (TypeError, ValueError):
                                sync_offset_ms = math.nan
                            calibration_sync_valid = (
                                force.get("calibration_sync_ok") is True
                                and force.get("sync_within_target") is True
                                and math.isfinite(sync_offset_ms)
                            )
                            if not calibration_sync_valid:
                                if key != last_outside_sync_key:
                                    self._increment(
                                        "frames_outside_force_sync_tolerance"
                                    )
                                    last_outside_sync_key = key
                                self._set_status(
                                    "force_sync_outside_calibration_tolerance"
                                )
                                self._stop_event.wait(self.poll_interval_sec)
                                continue
                            if _continuous_force_fz_n(force) is None:
                                if key != last_invalid_force_key:
                                    self._increment("invalid_force_samples")
                                    last_invalid_force_key = key
                                self._set_status("invalid_force_sample")
                                self._stop_event.wait(self.poll_interval_sec)
                                continue
                        if "response" in selected:
                            if self.model_provider is None:
                                model = {
                                    "model_source": "not_configured",
                                    "model_status": "model_provider_not_configured",
                                    "model_ready": False,
                                }
                            else:
                                try:
                                    model = dict(self.model_provider(latest) or {})
                                except Exception as exc:  # pragma: no cover - retained in capture
                                    model = {
                                        "model_source": "capture_model_provider",
                                        "model_status": "model_provider_error",
                                        "model_ready": False,
                                        "reason": f"{type(exc).__name__}: {exc}",
                                    }
                    else:
                        force = dict(self.force_provider(None) or {})
                        force_ready = bool(force.get("ok")) and (
                            not self.require_software_tare or bool(force.get("tare_ready"))
                        )
                        if not force_ready:
                            self._set_status(
                                str(force.get("status") or "waiting_for_force_reference")
                            )
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        key = (
                            "px6d_force",
                            force.get("force_sequence_end") or force.get("sequence_id"),
                            force.get("force_timestamp_epoch_sec"),
                        )
                        if _continuous_force_fz_n(force) is None:
                            if key != last_invalid_force_key:
                                self._increment("invalid_force_samples")
                                last_invalid_force_key = key
                            self._set_status("invalid_force_sample")
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        if key == last_written_key:
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        try:
                            timeline_timestamp = float(force.get("force_timestamp_epoch_sec"))
                        except (TypeError, ValueError):
                            timeline_timestamp = time.time()
                            self._increment("timeline_timestamp_fallbacks")
                        if not math.isfinite(timeline_timestamp):
                            timeline_timestamp = time.time()
                            self._increment("timeline_timestamp_fallbacks")
                    with self._lock:
                        if self._state["timeline_start_epoch_sec"] is None:
                            self._state["timeline_start_epoch_sec"] = timeline_timestamp
                        timeline_start = float(self._state["timeline_start_epoch_sec"])
                    elapsed_time = max(0.0, timeline_timestamp - timeline_start)
                    last_written_key = key
                    with self._lock:
                        capture_index = int(self._state["captured_timeline_frames"])
                        labels = {
                            "position_label": self._state["position_label"],
                            "action_label": self._state["action_label"],
                            "trial_id": self._state["trial_id"],
                            "operator_note": self._state["operator_note"],
                        }
                    record: dict[str, Any] = {
                        "capture_index": capture_index,
                        "timeline_timestamp_epoch_sec": timeline_timestamp,
                        "elapsed_time_sec": elapsed_time,
                        "capture_timestamp_epoch_sec": time.time(),
                        **labels,
                    }
                    if "spectrum" in selected and latest is not None:
                        record["spectrum"] = {
                            "source": latest.get("source"),
                            "frame_id": latest.get("frame_id"),
                            "timestamp_epoch_sec": key[2],
                            "wavelength_nm": wavelengths,
                            "intensity_counts": intensities,
                            "baseline_intensity_counts": (
                                latest.get(
                                    "normalization_reference_intensity_counts"
                                )
                                or []
                            ),
                            "normalized_intensity_ratio": (
                                latest.get("normalized_intensity_ratio") or []
                            ),
                            "normalization": (
                                latest.get("spectrum_normalization") or {}
                            ),
                            "spectrum_peaks": latest.get("spectrum_peaks") or [],
                        }
                    if "response" in selected:
                        record["tactile_response"] = model
                    if "force" in selected:
                        record["px6d_reference"] = force
                        record["force_fz_n"] = _continuous_force_fz_n(force)
                    sanitized_record = _sanitize_serializable(record)
                    jsonl_block = (
                        json.dumps(
                            sanitized_record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=_json_default,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    summary_block = _csv_rows_block(
                        [self._summary_row(record)],
                        SUMMARY_FIELDS,
                    )
                    spectrum_rows = self._spectrum_rows(record) if spectrum_writer else []
                    writes: list[tuple[Any, str]] = []
                    if spectrum_handle:
                        writes.append(
                            (
                                spectrum_handle,
                                _csv_rows_block(spectrum_rows, SPECTRUM_FIELDS),
                            )
                        )
                    if response_handle:
                        writes.append(
                            (
                                response_handle,
                                _csv_rows_block(
                                    [self._response_row(record)],
                                    RESPONSE_FIELDS,
                                ),
                            )
                        )
                    if force_handle:
                        writes.append(
                            (
                                force_handle,
                                _csv_rows_block([self._force_row(record)], FORCE_FIELDS),
                            )
                        )
                    # The JSONL manifest is the commit marker and is appended
                    # only after all detailed stream rows have been prepared.
                    writes.extend(
                        [
                            (summary_handle, summary_block),
                            (jsonl_handle, jsonl_block),
                        ]
                    )
                    _write_text_transaction(writes)
                    with self._lock:
                        self._state["captured_timeline_frames"] = capture_index + 1
                        self._state["captured_paired_frames"] = capture_index + 1
                        if spectrum_writer:
                            self._state["spectrum_frame_rows"] = capture_index + 1
                        self._state["spectrum_sample_rows"] = int(
                            self._state.get("spectrum_sample_rows") or 0
                        ) + len(spectrum_rows)
                        if response_writer:
                            self._state["response_rows"] = capture_index + 1
                        if force_writer:
                            self._state["force_rows"] = capture_index + 1
                        if latest is not None:
                            self._state["last_spectrum_frame_id"] = latest.get("frame_id")
                        if force_writer:
                            self._state["last_sync_quality"] = force.get("sync_quality")
                            self._state["last_sync_offset_ms"] = force.get("sync_offset_ms")
                        try:
                            absolute_offset = abs(float(force.get("sync_offset_ms")))
                        except (TypeError, ValueError):
                            absolute_offset = None
                        if absolute_offset is not None and not math.isfinite(absolute_offset):
                            absolute_offset = None
                        if absolute_offset is not None:
                            previous_max = self._state.get(
                                "maximum_absolute_sync_offset_ms"
                            )
                            self._state["maximum_absolute_sync_offset_ms"] = (
                                absolute_offset
                                if previous_max is None
                                else max(float(previous_max), absolute_offset)
                            )
                        self._state["capture_status"] = "recording_selected_streams"
                        self._write_journal_locked()
                    self._stop_event.wait(self.poll_interval_sec)
        except Exception as exc:  # pragma: no cover - surfaced by API and metadata
            with self._lock:
                self._state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._state["capture_status"] = "capture_error"
                self._state["running"] = False
        finally:
            with self._lock:
                current = threading.current_thread()
                if self._thread is current:
                    self._thread = None
                if self._state["running"]:
                    self._state["running"] = False
                    if not self._stop_event.is_set():
                        self._state["capture_status"] = "capture_error"
                        if not self._state.get("last_error"):
                            self._state["last_error"] = (
                                "capture_worker_exited_unexpectedly"
                            )
                self._state["ended_at_epoch_sec"] = time.time()
                try:
                    self._write_metadata_locked()
                    self._write_journal_locked()
                except Exception as exc:  # retain terminal state even if audit I/O fails
                    self._state["capture_status"] = "capture_error"
                    metadata_error = f"metadata_write_failed: {type(exc).__name__}: {exc}"
                    existing_error = str(self._state.get("last_error") or "").strip()
                    self._state["last_error"] = (
                        f"{existing_error}; {metadata_error}"
                        if existing_error
                        else metadata_error
                    )

    @staticmethod
    def _spectrum_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
        spectrum = record["spectrum"]
        wavelengths = list(spectrum.get("wavelength_nm") or [])
        intensities = list(spectrum.get("intensity_counts") or [])
        baseline = list(spectrum.get("baseline_intensity_counts") or [])
        normalized = list(spectrum.get("normalized_intensity_ratio") or [])
        normalization = dict(spectrum.get("normalization") or {})
        labels = {
            "position_label": record["position_label"],
            "action_label": record["action_label"],
            "trial_id": record["trial_id"],
        }
        rows: list[dict[str, Any]] = []
        for point_index, (wavelength, intensity) in enumerate(
            zip(wavelengths, intensities)
        ):
            rows.append(
                {
                    "capture_index": record["capture_index"],
                    "timeline_timestamp_epoch_sec": record[
                        "timeline_timestamp_epoch_sec"
                    ],
                    "elapsed_time_sec": record["elapsed_time_sec"],
                    "spectrum_timestamp_epoch_sec": spectrum.get(
                        "timestamp_epoch_sec"
                    ),
                    "spectrum_source": spectrum.get("source"),
                    "spectrum_frame_id": spectrum.get("frame_id"),
                    "point_index": point_index,
                    "wavelength_nm": wavelength,
                    "intensity_counts": intensity,
                    "baseline_intensity_counts": (
                        baseline[point_index]
                        if point_index < len(baseline)
                        else None
                    ),
                    "normalized_intensity_ratio": (
                        normalized[point_index]
                        if point_index < len(normalized)
                        else None
                    ),
                    "normalization_method": normalization.get("method"),
                    "normalization_status": normalization.get("status"),
                    **labels,
                }
            )
        return rows

    @staticmethod
    def _response_row(record: dict[str, Any]) -> dict[str, Any]:
        model = record.get("tactile_response") or {}
        contact = model.get("contact") or {}
        position = model.get("position") or {}
        response = model.get("response_level") or {}
        contact_probabilities = contact.get("probabilities") or {}
        response_probabilities = response.get("probabilities") or {}
        release_guard = model.get("release_guard") or {}
        optical_force = _optical_force_export_fields(model)
        return {
            "capture_index": record["capture_index"],
            "timeline_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "elapsed_time_sec": record["elapsed_time_sec"],
            "model_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "model_source": model.get("model_source"),
            "model_status": model.get("model_status") or model.get("status"),
            "model_ready": model.get("model_ready", model.get("ready")),
            **optical_force,
            "contact_label": contact.get("label"),
            "contact_confidence": contact.get("confidence"),
            "contact_probability": contact_probabilities.get("contact"),
            "no_contact_probability": contact_probabilities.get("no_contact"),
            "predicted_position_label": position.get("label"),
            "position_confidence": position.get("confidence"),
            "response_level": response.get("label"),
            "response_level_confidence": response.get("confidence"),
            "response_level_raw_label": response.get("raw_label"),
            "response_level_decision_rule": response.get("decision_rule"),
            "light_probability": response_probabilities.get("light"),
            "normal_probability": response_probabilities.get("normal"),
            "hard_probability": response_probabilities.get("hard"),
            "operational_state": model.get("operational_state"),
            "release_latched": release_guard.get("release_latched"),
            "runtime_baseline_revision": model.get("runtime_baseline_revision"),
            "position_label": record["position_label"],
            "action_label": record["action_label"],
            "trial_id": record["trial_id"],
        }

    @staticmethod
    def _force_row(record: dict[str, Any]) -> dict[str, Any]:
        force = record["px6d_reference"]
        raw = force.get("raw") or {}
        zeroed = force.get("zeroed") or {}
        filtered = force.get("filtered_zeroed") or zeroed
        mechanical = force.get("mechanical") or {}
        filtered_mechanical = force.get("filtered_mechanical") or mechanical
        return {
            "capture_index": record["capture_index"],
            "timeline_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "elapsed_time_sec": record["elapsed_time_sec"],
            "force_timestamp_epoch_sec": force.get("force_timestamp_epoch_sec"),
            "sync_method": force.get("sync_method"),
            "sync_quality": force.get("sync_quality"),
            "sync_offset_ms": force.get("sync_offset_ms"),
            "sync_within_target": force.get("sync_within_target"),
            "calibration_sync_ok": force.get("calibration_sync_ok"),
            "force_sample_count": force.get("sample_count"),
            "force_sequence_start": force.get("force_sequence_start"),
            "force_sequence_end": force.get("force_sequence_end"),
            "fx_raw_n": raw.get("fx_n"),
            "fy_raw_n": raw.get("fy_n"),
            "fz_raw_n": raw.get("fz_n"),
            "mx_raw_nm": raw.get("mx_nm"),
            "my_raw_nm": raw.get("my_nm"),
            "mz_raw_nm": raw.get("mz_nm"),
            "fx_zeroed_n": zeroed.get("fx_n"),
            "fy_zeroed_n": zeroed.get("fy_n"),
            "fz_zeroed_n": zeroed.get("fz_n"),
            "mx_zeroed_nm": zeroed.get("mx_nm"),
            "my_zeroed_nm": zeroed.get("my_nm"),
            "mz_zeroed_nm": zeroed.get("mz_nm"),
            "fx_filtered_n": filtered.get("fx_n"),
            "fy_filtered_n": filtered.get("fy_n"),
            "fz_filtered_n": filtered.get("fz_n"),
            "mx_filtered_nm": filtered.get("mx_nm"),
            "my_filtered_nm": filtered.get("my_nm"),
            "mz_filtered_nm": filtered.get("mz_nm"),
            "reference_fz_n": force.get("reference_fz_n"),
            "reference_fz_display_n": force.get("reference_fz_display_n"),
            "median_reference_fz_n": force.get("median_reference_fz_n"),
            "filtered_reference_fz_n": force.get("filtered_reference_fz_n"),
            "drift_offset_n": force.get("drift_offset_n"),
            "drift_corrected_reference_fz_n": force.get(
                "drift_corrected_reference_fz_n"
            ),
            "conditioned_reference_fz_n": force.get(
                "conditioned_reference_fz_n"
            ),
            "force_fz_n": record.get(
                "force_fz_n", _continuous_force_fz_n(force)
            ),
            "stationary_detected": force.get("stationary_detected"),
            "auto_zero_drift_active": force.get("auto_zero_drift_active"),
            "force_filter_status": force.get("force_filter_status"),
            "force_resultant_n": mechanical.get("force_resultant_n"),
            "shear_resultant_n": mechanical.get("shear_resultant_n"),
            "moment_resultant_nm": mechanical.get("moment_resultant_nm"),
            "force_utilization_percent": mechanical.get("force_utilization_percent"),
            "moment_utilization_percent": mechanical.get("moment_utilization_percent"),
            "filtered_force_resultant_n": filtered_mechanical.get(
                "force_resultant_n"
            ),
            "filtered_shear_resultant_n": filtered_mechanical.get(
                "shear_resultant_n"
            ),
            "filtered_moment_resultant_nm": filtered_mechanical.get(
                "moment_resultant_nm"
            ),
            "position_label": record["position_label"],
            "action_label": record["action_label"],
            "trial_id": record["trial_id"],
        }

    def _set_status(self, value: str) -> None:
        with self._lock:
            self._state["capture_status"] = value

    @staticmethod
    def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
        spectrum = record.get("spectrum") or {}
        model = record.get("tactile_response") or {}
        contact = model.get("contact") or {}
        position = model.get("position") or {}
        response = model.get("response_level") or {}
        force = record.get("px6d_reference") or {}
        raw = force.get("raw") or {}
        zeroed = force.get("zeroed") or {}
        filtered = force.get("filtered_zeroed") or zeroed
        mechanical = force.get("mechanical") or {}
        filtered_mechanical = force.get("filtered_mechanical") or mechanical
        optical_force = _optical_force_export_fields(model)
        return {
            "capture_index": record["capture_index"],
            "timeline_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "elapsed_time_sec": record["elapsed_time_sec"],
            "capture_timestamp_epoch_sec": record["capture_timestamp_epoch_sec"],
            "position_label": record["position_label"],
            "action_label": record["action_label"],
            "trial_id": record["trial_id"],
            "spectrum_source": spectrum.get("source"),
            "spectrum_frame_id": spectrum.get("frame_id"),
            "spectrum_timestamp_epoch_sec": spectrum.get("timestamp_epoch_sec"),
            "spectrum_points": len(spectrum.get("wavelength_nm") or []),
            "force_timestamp_epoch_sec": force.get("force_timestamp_epoch_sec"),
            "sync_method": force.get("sync_method"),
            "sync_quality": force.get("sync_quality"),
            "sync_offset_ms": force.get("sync_offset_ms"),
            "sync_within_target": force.get("sync_within_target"),
            "calibration_sync_ok": force.get("calibration_sync_ok"),
            "force_sample_count": force.get("sample_count"),
            "fx_raw_n": raw.get("fx_n"),
            "fy_raw_n": raw.get("fy_n"),
            "fz_raw_n": raw.get("fz_n"),
            "mx_raw_nm": raw.get("mx_nm"),
            "my_raw_nm": raw.get("my_nm"),
            "mz_raw_nm": raw.get("mz_nm"),
            "fx_zeroed_n": zeroed.get("fx_n"),
            "fy_zeroed_n": zeroed.get("fy_n"),
            "fz_zeroed_n": zeroed.get("fz_n"),
            "mx_zeroed_nm": zeroed.get("mx_nm"),
            "my_zeroed_nm": zeroed.get("my_nm"),
            "mz_zeroed_nm": zeroed.get("mz_nm"),
            "fx_filtered_n": filtered.get("fx_n"),
            "fy_filtered_n": filtered.get("fy_n"),
            "fz_filtered_n": filtered.get("fz_n"),
            "mx_filtered_nm": filtered.get("mx_nm"),
            "my_filtered_nm": filtered.get("my_nm"),
            "mz_filtered_nm": filtered.get("mz_nm"),
            "reference_fz_n": force.get("reference_fz_n"),
            "reference_fz_display_n": force.get("reference_fz_display_n"),
            "median_reference_fz_n": force.get("median_reference_fz_n"),
            "filtered_reference_fz_n": force.get("filtered_reference_fz_n"),
            "drift_offset_n": force.get("drift_offset_n"),
            "drift_corrected_reference_fz_n": force.get(
                "drift_corrected_reference_fz_n"
            ),
            "conditioned_reference_fz_n": force.get(
                "conditioned_reference_fz_n"
            ),
            "force_fz_n": record.get(
                "force_fz_n", _continuous_force_fz_n(force)
            ),
            "stationary_detected": force.get("stationary_detected"),
            "auto_zero_drift_active": force.get("auto_zero_drift_active"),
            "force_filter_status": force.get("force_filter_status"),
            "force_resultant_n": mechanical.get("force_resultant_n"),
            "shear_resultant_n": mechanical.get("shear_resultant_n"),
            "moment_resultant_nm": mechanical.get("moment_resultant_nm"),
            "force_utilization_percent": mechanical.get("force_utilization_percent"),
            "moment_utilization_percent": mechanical.get("moment_utilization_percent"),
            "filtered_force_resultant_n": filtered_mechanical.get(
                "force_resultant_n"
            ),
            "filtered_shear_resultant_n": filtered_mechanical.get(
                "shear_resultant_n"
            ),
            "filtered_moment_resultant_nm": filtered_mechanical.get(
                "moment_resultant_nm"
            ),
            "model_source": model.get("model_source"),
            "model_status": model.get("model_status") or model.get("status"),
            "model_ready": model.get("model_ready", model.get("ready")),
            **optical_force,
            "predicted_contact_label": contact.get("label"),
            "predicted_position_label": position.get("label"),
            "predicted_response_level": response.get("label"),
            "predicted_response_confidence": response.get("confidence"),
        }

    @staticmethod
    def _read_stream_timeline(
        path: Path,
        *,
        allow_duplicate_indices: bool = False,
    ) -> tuple[dict[int, tuple[float, float]], list[str]]:
        timeline: dict[int, tuple[float, float]] = {}
        errors: list[str] = []
        if not path.exists():
            return timeline, [f"missing_file:{path.name}"]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                try:
                    capture_index = int(row["capture_index"])
                    timestamp = float(row["timeline_timestamp_epoch_sec"])
                    elapsed = float(row["elapsed_time_sec"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"invalid_timeline_row:{path.name}:{row_number}")
                    continue
                existing = timeline.get(capture_index)
                current = (timestamp, elapsed)
                if existing is not None:
                    if not allow_duplicate_indices:
                        errors.append(f"duplicate_capture_index:{path.name}:{capture_index}")
                    if (
                        abs(existing[0] - current[0]) > 1e-9
                        or abs(existing[1] - current[1]) > 1e-9
                    ):
                        errors.append(
                            f"inconsistent_duplicate_index:{path.name}:{capture_index}"
                        )
                timeline[capture_index] = current
        return timeline, errors

    @staticmethod
    def _read_jsonl_timeline(
        path: Path,
    ) -> tuple[dict[int, tuple[float, float]], list[str]]:
        timeline: dict[int, tuple[float, float]] = {}
        errors: list[str] = []
        if not path.exists():
            return timeline, [f"missing_file:{path.name}"]
        with path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    capture_index = int(row["capture_index"])
                    timestamp = float(row["timeline_timestamp_epoch_sec"])
                    elapsed = float(row["elapsed_time_sec"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    errors.append(f"invalid_timeline_row:{path.name}:{row_number}")
                    continue
                if capture_index in timeline:
                    errors.append(f"duplicate_capture_index:{path.name}:{capture_index}")
                timeline[capture_index] = (timestamp, elapsed)
        return timeline, errors

    def _alignment_audit_locked(self, output_dir: Path) -> dict[str, Any]:
        selected = list(self._state.get("selected_outputs") or [])
        paths = {
            "spectrum": output_dir / "spectrum_timeseries.csv",
            "response": output_dir / "tactile_response_timeseries.csv",
            "force": output_dir / "force_timeseries.csv",
        }
        manifest_timeline, manifest_errors = self._read_jsonl_timeline(
            output_dir / "synchronized_frames.jsonl"
        )
        summary_timeline, summary_errors = self._read_stream_timeline(
            output_dir / "frame_summary.csv"
        )
        timelines: dict[str, dict[int, tuple[float, float]]] = {
            "manifest": manifest_timeline,
            "summary": summary_timeline,
        }
        errors: list[str] = [*manifest_errors, *summary_errors]
        for stream in selected:
            timeline, stream_errors = self._read_stream_timeline(
                paths[stream],
                allow_duplicate_indices=stream == "spectrum",
            )
            timelines[stream] = timeline
            errors.extend(stream_errors)
        reference_stream = "manifest"
        reference = timelines[reference_stream]
        for stream in ["summary", *selected]:
            candidate = timelines.get(stream, {})
            if set(candidate) != set(reference):
                errors.append(f"capture_index_mismatch:{reference_stream}:{stream}")
                continue
            for capture_index, expected in reference.items():
                actual = candidate[capture_index]
                if (
                    abs(expected[0] - actual[0]) > 1e-9
                    or abs(expected[1] - actual[1]) > 1e-9
                ):
                    errors.append(
                        f"timeline_value_mismatch:{reference_stream}:{stream}:{capture_index}"
                    )
                    break
        frame_counts = {stream: len(timeline) for stream, timeline in timelines.items()}
        expected_frames = int(self._state.get("captured_timeline_frames") or 0)
        for stream, count in frame_counts.items():
            if count != expected_frames:
                errors.append(f"frame_count_mismatch:{stream}:{count}:{expected_frames}")
        return {
            "status": (
                "aligned"
                if expected_frames > 0 and not errors
                else "no_frames"
                if expected_frames == 0 and not errors
                else "alignment_error"
            ),
            "all_selected_streams_aligned": bool(expected_frames > 0 and not errors),
            "selected_outputs": selected,
            "canonical_keys": [
                "capture_index",
                "timeline_timestamp_epoch_sec",
                "elapsed_time_sec",
            ],
            "timeline_basis": (
                "spectrum_ingested_at_host_epoch_clock"
                if {"spectrum", "response"} & set(selected)
                else "px6d_force_host_epoch_clock"
            ),
            "expected_timeline_frames": expected_frames,
            "stream_frame_counts": frame_counts,
            "maximum_absolute_force_sync_offset_ms": self._state.get(
                "maximum_absolute_sync_offset_ms"
            ),
            "errors": errors,
        }

    def _write_metadata_locked(
        self,
        *,
        refresh_provenance: bool = True,
        session_initialized: bool = False,
    ) -> None:
        output_directory = self._state.get("output_directory")
        if not output_directory:
            return
        output_dir = Path(str(output_directory))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.status()
        alignment_audit = (
            {
                "status": "session_initialized",
                "all_selected_streams_aligned": False,
                "selected_outputs": list(
                    self._state.get("selected_outputs") or []
                ),
                "expected_timeline_frames": 0,
                "stream_frame_counts": {},
                "errors": [],
            }
            if session_initialized
            else self._alignment_audit_locked(output_dir)
        )
        selected = set(self._state.get("selected_outputs") or [])
        force_status_snapshot_error = None
        if "force" in selected:
            try:
                force_status = dict(self.force_status_provider() or {})
            except Exception as exc:
                force_status = {}
                force_status_snapshot_error = f"{type(exc).__name__}: {exc}"
        else:
            force_status = {}
        if refresh_provenance:
            provenance_at_end, provenance_end_error = self._snapshot_provenance()
            if provenance_at_end is not None:
                self._provenance_latest = provenance_at_end
                self._provenance_latest_error = None
            elif provenance_end_error and self._provenance_latest is None:
                self._provenance_latest_error = provenance_end_error
        payload.update(
            {
                "schema_version": "touch_synchronized_capture_v4",
                "session_initialized": bool(session_initialized),
                "provenance": {
                    "start": self._provenance_at_start,
                    "end": (
                        None
                        if session_initialized
                        else self._provenance_latest
                    ),
                    "start_snapshot_error": self._provenance_start_error,
                    "end_snapshot_error": (
                        None
                        if session_initialized
                        else self._provenance_latest_error
                    ),
                },
                "force_target": {
                    "mode": "continuous_fz_regression",
                    "field": "force_fz_n",
                    "unit": "N",
                    "source": "PX6D conditioned compression Fz",
                    "categorical_bins_enabled": False,
                },
                "paired_measurement_contract": {
                    "version": "touch_optical_force_measurement_v1",
                    "reference_field": "force_fz_n",
                    "reference_source": "PX6D conditioned compression Fz",
                    "estimate_field": "optical_estimated_fz_n",
                    "raw_estimate_field": "optical_raw_estimated_fz_n",
                    "estimate_source": "optical spectrum model only",
                    "unit": "N",
                    "force_sensor_is_model_input": False,
                    "intended_use": (
                        "research calibration, repeatability, lag, recovery, and "
                        "optical-force consistency analysis"
                    ),
                },
                "selection_semantics": (
                    "Only user-selected data streams are written to their dedicated CSV files. "
                    "The JSONL manifest contains those same selected payloads plus common labels."
                ),
                "synchronization_contract": {
                    "canonical_keys": [
                        "capture_index",
                        "timeline_timestamp_epoch_sec",
                        "elapsed_time_sec",
                    ],
                    "optical_driven_timeline": (
                        "Spectrum and tactile-response captures use the spectrum host timestamp. "
                        "PX6D samples are window-median aligned to that timestamp, and frames "
                        "outside the configured calibration tolerance are rejected."
                    ),
                    "force_only_timeline": (
                        "Force-only capture uses the PX6D host timestamp and unique force sequence."
                    ),
                },
                "alignment_audit": alignment_audit,
                "spectrum_payload": (
                    "one row per wavelength point with raw counts and, when "
                    "a stable no-contact reference is ready, aligned I0 and I/I0"
                    if "spectrum" in selected
                    else "not_selected"
                ),
                "spectrum_normalization_contract": {
                    "method": "no_contact_baseline_ratio",
                    "formula": "normalized_intensity_ratio = I(lambda,t) / I0(lambda)",
                    "raw_counts_retained": True,
                    "model_input_source": "raw_intensity",
                    "per_frame_min_max_used": False,
                    "missing_reference_behavior": (
                        "normalized fields remain empty until an accepted "
                        "no-contact baseline spectrum is available"
                    ),
                },
                "tactile_response_payload": (
                    "one model response row per canonical frame, including the optical-only Fz estimate, "
                    "in tactile_response_timeseries.csv"
                    if "response" in selected
                    else "not_selected"
                ),
                "mechanical_payload": (
                    "raw and software-zeroed six-axis values, conditioned Fz stages, and derived resultants"
                    if "force" in selected
                    else "not_selected"
                ),
                "force_conditioning": force_status.get("force_conditioning"),
                "force_status_snapshot_error": force_status_snapshot_error,
                "files": {
                    "spectrum_timeseries_csv": (
                        "spectrum_timeseries.csv" if "spectrum" in selected else None
                    ),
                    "tactile_response_timeseries_csv": (
                        "tactile_response_timeseries.csv" if "response" in selected else None
                    ),
                    "force_timeseries_csv": (
                        "force_timeseries.csv" if "force" in selected else None
                    ),
                    "selected_payload_manifest_jsonl": "synchronized_frames.jsonl",
                    "timeline_summary_csv": "frame_summary.csv",
                    "capture_recovery_journal_json": "capture_journal.json",
                },
            }
        )
        _write_json_atomic(output_dir / "session_metadata.json", payload)
