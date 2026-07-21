"""In-process BaySpec spectrum and PX6D reference-force session recorder."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from datetime import datetime
import json
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
    "reference_fz_n",
    "reference_fz_display_n",
    "median_reference_fz_n",
    "filtered_reference_fz_n",
    "drift_offset_n",
    "drift_corrected_reference_fz_n",
    "conditioned_reference_fz_n",
    "stationary_detected",
    "auto_zero_drift_active",
    "force_filter_status",
    "force_resultant_n",
    "shear_resultant_n",
    "moment_resultant_nm",
    "force_utilization_percent",
    "moment_utilization_percent",
    "model_source",
    "model_status",
    "model_ready",
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
    "reference_fz_n",
    "reference_fz_display_n",
    "median_reference_fz_n",
    "filtered_reference_fz_n",
    "drift_offset_n",
    "drift_corrected_reference_fz_n",
    "conditioned_reference_fz_n",
    "stationary_detected",
    "auto_zero_drift_active",
    "force_filter_status",
    "force_resultant_n",
    "shear_resultant_n",
    "moment_resultant_nm",
    "force_utilization_percent",
    "moment_utilization_percent",
    "position_label",
    "action_label",
    "trial_id",
]


def _safe_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_")
    return text[:48] or fallback


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


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
        poll_interval_sec: float = 0.05,
        require_software_tare: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.frame_provider = frame_provider
        self.force_provider = force_provider
        self.force_status_provider = force_status_provider
        self.model_provider = model_provider
        self.poll_interval_sec = max(0.01, min(1.0, float(poll_interval_sec)))
        self.require_software_tare = bool(require_software_tare)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
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
            "frames_missing_force_reference": 0,
            "last_spectrum_frame_id": None,
            "last_sync_quality": None,
            "last_sync_offset_ms": None,
            "maximum_absolute_sync_offset_ms": None,
            "timeline_start_epoch_sec": None,
            "last_error": None,
        }

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
        for candidate in candidates:
            normalized = OUTPUT_STREAM_ALIASES.get(str(candidate or "").strip().lower())
            if normalized and normalized not in selected:
                selected.append(normalized)
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
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "status": "capture_already_running", **self.status()}
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
            force_status = dict(self.force_status_provider() or {})
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

        with self._lock:
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
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(output_dir,),
                name="touch-optical-force-capture",
                daemon=True,
            )
            self._thread.start()
        return {"ok": True, "status": "capture_started", **self.status()}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            running = bool(thread is not None and thread.is_alive())
        if not running:
            return {"ok": True, "status": "capture_not_running", **self.status()}
        self._stop_event.set()
        thread.join(timeout=4.0)
        with self._lock:
            if thread.is_alive():
                self._state["last_error"] = "capture_thread_stop_timeout"
            self._state["running"] = False
            self._state["ended_at_epoch_sec"] = time.time()
            if self._state["captured_paired_frames"] > 0:
                self._state["capture_status"] = "complete"
            elif self._state["last_error"]:
                self._state["capture_status"] = "capture_error"
            else:
                self._state["capture_status"] = "no_paired_frames"
            self._write_metadata_locked()
        return {"ok": True, "status": "capture_stopped", **self.status()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._state)
        denominator = int(payload.get("eligible_spectrum_frames") or 0)
        paired = int(payload.get("captured_timeline_frames") or 0)
        payload["paired_frame_ratio"] = paired / denominator if denominator else None
        payload["force_semantics"] = "PX6D_reference_Fz_not_optical_force_prediction"
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
                        if (
                            not isinstance(candidate, dict)
                            or not isinstance(wavelengths_value, list)
                            or not isinstance(intensities_value, list)
                            or not wavelengths_value
                            or len(wavelengths_value) != len(intensities_value)
                        ):
                            self._increment("polls_without_valid_spectrum")
                            self._set_status("waiting_for_optical_frame")
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        latest = candidate
                        wavelengths = wavelengths_value
                        intensities = intensities_value
                        key = (
                            latest.get("source"),
                            latest.get("frame_id"),
                            latest.get("ingested_at")
                            or latest.get("timestamp_epoch_sec")
                            or latest.get("timestamp"),
                        )
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
                        if "force" in selected:
                            force = dict(self.force_provider(latest) or {})
                            force_ready = bool(force.get("ok")) and (
                                not self.require_software_tare or bool(force.get("tare_ready"))
                            )
                            if not force_ready:
                                if key != last_missing_force_key:
                                    self._increment("frames_missing_force_reference")
                                    last_missing_force_key = key
                                self._set_status(
                                    str(force.get("status") or "waiting_for_force_reference")
                                )
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
                        if key == last_written_key:
                            self._stop_event.wait(self.poll_interval_sec)
                            continue
                        try:
                            timeline_timestamp = float(force.get("force_timestamp_epoch_sec"))
                        except (TypeError, ValueError):
                            timeline_timestamp = time.time()
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
                            "spectrum_peaks": latest.get("spectrum_peaks") or [],
                        }
                    if "response" in selected:
                        record["tactile_response"] = model
                    if "force" in selected:
                        record["px6d_reference"] = force
                    jsonl_handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
                        + "\n"
                    )
                    summary_writer.writerow(self._summary_row(record))
                    spectrum_rows = self._spectrum_rows(record) if spectrum_writer else []
                    if spectrum_writer:
                        spectrum_writer.writerows(spectrum_rows)
                    if response_writer:
                        response_writer.writerow(self._response_row(record))
                    if force_writer:
                        force_writer.writerow(self._force_row(record))
                    jsonl_handle.flush()
                    summary_handle.flush()
                    if spectrum_handle:
                        spectrum_handle.flush()
                    if response_handle:
                        response_handle.flush()
                    if force_handle:
                        force_handle.flush()
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
                    self._stop_event.wait(self.poll_interval_sec)
        except Exception as exc:  # pragma: no cover - surfaced by API and metadata
            with self._lock:
                self._state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._state["capture_status"] = "capture_error"
                self._state["running"] = False
        finally:
            with self._lock:
                if self._state["running"] and self._stop_event.is_set():
                    self._state["running"] = False
                self._state["ended_at_epoch_sec"] = time.time()
                self._write_metadata_locked()

    @staticmethod
    def _spectrum_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
        spectrum = record["spectrum"]
        labels = {
            "position_label": record["position_label"],
            "action_label": record["action_label"],
            "trial_id": record["trial_id"],
        }
        return [
            {
                "capture_index": record["capture_index"],
                "timeline_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
                "elapsed_time_sec": record["elapsed_time_sec"],
                "spectrum_timestamp_epoch_sec": spectrum.get("timestamp_epoch_sec"),
                "spectrum_source": spectrum.get("source"),
                "spectrum_frame_id": spectrum.get("frame_id"),
                "point_index": point_index,
                "wavelength_nm": wavelength,
                "intensity_counts": intensity,
                **labels,
            }
            for point_index, (wavelength, intensity) in enumerate(
                zip(
                    spectrum.get("wavelength_nm") or [],
                    spectrum.get("intensity_counts") or [],
                )
            )
        ]

    @staticmethod
    def _response_row(record: dict[str, Any]) -> dict[str, Any]:
        model = record.get("tactile_response") or {}
        contact = model.get("contact") or {}
        position = model.get("position") or {}
        response = model.get("response_level") or {}
        contact_probabilities = contact.get("probabilities") or {}
        response_probabilities = response.get("probabilities") or {}
        release_guard = model.get("release_guard") or {}
        return {
            "capture_index": record["capture_index"],
            "timeline_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "elapsed_time_sec": record["elapsed_time_sec"],
            "model_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "model_source": model.get("model_source"),
            "model_status": model.get("model_status") or model.get("status"),
            "model_ready": model.get("model_ready", model.get("ready")),
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
        mechanical = force.get("mechanical") or {}
        return {
            "capture_index": record["capture_index"],
            "timeline_timestamp_epoch_sec": record["timeline_timestamp_epoch_sec"],
            "elapsed_time_sec": record["elapsed_time_sec"],
            "force_timestamp_epoch_sec": force.get("force_timestamp_epoch_sec"),
            "sync_method": force.get("sync_method"),
            "sync_quality": force.get("sync_quality"),
            "sync_offset_ms": force.get("sync_offset_ms"),
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
            "stationary_detected": force.get("stationary_detected"),
            "auto_zero_drift_active": force.get("auto_zero_drift_active"),
            "force_filter_status": force.get("force_filter_status"),
            "force_resultant_n": mechanical.get("force_resultant_n"),
            "shear_resultant_n": mechanical.get("shear_resultant_n"),
            "moment_resultant_nm": mechanical.get("moment_resultant_nm"),
            "force_utilization_percent": mechanical.get("force_utilization_percent"),
            "moment_utilization_percent": mechanical.get("moment_utilization_percent"),
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
        mechanical = force.get("mechanical") or {}
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
            "stationary_detected": force.get("stationary_detected"),
            "auto_zero_drift_active": force.get("auto_zero_drift_active"),
            "force_filter_status": force.get("force_filter_status"),
            "force_resultant_n": mechanical.get("force_resultant_n"),
            "shear_resultant_n": mechanical.get("shear_resultant_n"),
            "moment_resultant_nm": mechanical.get("moment_resultant_nm"),
            "force_utilization_percent": mechanical.get("force_utilization_percent"),
            "moment_utilization_percent": mechanical.get("moment_utilization_percent"),
            "model_source": model.get("model_source"),
            "model_status": model.get("model_status") or model.get("status"),
            "model_ready": model.get("model_ready", model.get("ready")),
            "predicted_contact_label": contact.get("label"),
            "predicted_position_label": position.get("label"),
            "predicted_response_level": response.get("label"),
            "predicted_response_confidence": response.get("confidence"),
        }

    @staticmethod
    def _read_stream_timeline(path: Path) -> tuple[dict[int, tuple[float, float]], list[str]]:
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
                if existing is not None and (
                    abs(existing[0] - current[0]) > 1e-9
                    or abs(existing[1] - current[1]) > 1e-9
                ):
                    errors.append(f"inconsistent_duplicate_index:{path.name}:{capture_index}")
                timeline[capture_index] = current
        return timeline, errors

    def _alignment_audit_locked(self, output_dir: Path) -> dict[str, Any]:
        selected = list(self._state.get("selected_outputs") or [])
        paths = {
            "spectrum": output_dir / "spectrum_timeseries.csv",
            "response": output_dir / "tactile_response_timeseries.csv",
            "force": output_dir / "force_timeseries.csv",
        }
        timelines: dict[str, dict[int, tuple[float, float]]] = {}
        errors: list[str] = []
        for stream in selected:
            timeline, stream_errors = self._read_stream_timeline(paths[stream])
            timelines[stream] = timeline
            errors.extend(stream_errors)
        reference_stream = selected[0] if selected else None
        reference = timelines.get(reference_stream or "", {})
        for stream in selected[1:]:
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

    def _write_metadata_locked(self) -> None:
        output_directory = self._state.get("output_directory")
        if not output_directory:
            return
        output_dir = Path(str(output_directory))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.status()
        alignment_audit = self._alignment_audit_locked(output_dir)
        selected = set(self._state.get("selected_outputs") or [])
        force_status = (
            dict(self.force_status_provider() or {}) if "force" in selected else {}
        )
        payload.update(
            {
                "schema_version": "touch_synchronized_capture_v2",
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
                        "PX6D samples are window-median aligned to that timestamp."
                    ),
                    "force_only_timeline": (
                        "Force-only capture uses the PX6D host timestamp and unique force sequence."
                    ),
                },
                "alignment_audit": alignment_audit,
                "spectrum_payload": (
                    "one row per wavelength point in spectrum_timeseries.csv"
                    if "spectrum" in selected
                    else "not_selected"
                ),
                "tactile_response_payload": (
                    "one model response row per canonical frame in tactile_response_timeseries.csv"
                    if "response" in selected
                    else "not_selected"
                ),
                "mechanical_payload": (
                    "raw and software-zeroed six-axis values, conditioned Fz stages, and derived resultants"
                    if "force" in selected
                    else "not_selected"
                ),
                "force_conditioning": force_status.get("force_conditioning"),
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
                },
            }
        )
        (output_dir / "session_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
