"""In-process BaySpec spectrum and PX6D reference-force session recorder."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable


SUMMARY_FIELDS = [
    "capture_index",
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
    "force_resultant_n",
    "shear_resultant_n",
    "moment_resultant_nm",
    "force_utilization_percent",
    "moment_utilization_percent",
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
    """Write unique full-spectrum frames with aligned six-axis force labels."""

    def __init__(
        self,
        *,
        output_root: Path,
        frame_provider: Callable[[], dict[str, Any]],
        force_provider: Callable[[dict[str, Any] | None], dict[str, Any]],
        force_status_provider: Callable[[], dict[str, Any]],
        poll_interval_sec: float = 0.05,
        require_software_tare: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.frame_provider = frame_provider
        self.force_provider = force_provider
        self.force_status_provider = force_status_provider
        self.poll_interval_sec = max(0.01, min(1.0, float(poll_interval_sec)))
        self.require_software_tare = bool(require_software_tare)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = self._idle_state()

    @staticmethod
    def _idle_state() -> dict[str, Any]:
        return {
            "running": False,
            "capture_status": "idle",
            "session_id": None,
            "output_directory": None,
            "started_at_epoch_sec": None,
            "ended_at_epoch_sec": None,
            "position_label": "unlabeled",
            "action_label": "unlabeled",
            "trial_id": "trial_001",
            "operator_note": "",
            "captured_paired_frames": 0,
            "eligible_spectrum_frames": 0,
            "skipped_duplicate_spectrum_frames": 0,
            "polls_without_valid_spectrum": 0,
            "frames_missing_force_reference": 0,
            "last_spectrum_frame_id": None,
            "last_sync_quality": None,
            "last_sync_offset_ms": None,
            "last_error": None,
        }

    def start(
        self,
        *,
        position_label: str = "unlabeled",
        action_label: str = "unlabeled",
        trial_id: str = "trial_001",
        operator_note: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "status": "capture_already_running", **self.status()}
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

        started = time.time()
        timestamp = datetime.fromtimestamp(started).strftime("%Y%m%d_%H%M%S")
        position = _safe_token(position_label, "unlabeled")
        action = _safe_token(action_label, "unlabeled")
        trial = _safe_token(trial_id, "trial_001")
        base_session_id = f"{timestamp}_{position}_{action}_{trial}"
        session_id = base_session_id
        output_dir = self.output_root / session_id
        suffix = 2
        while output_dir.exists():
            session_id = f"{base_session_id}_{suffix:02d}"
            output_dir = self.output_root / session_id
            suffix += 1
        output_dir.mkdir(parents=True, exist_ok=False)

        with self._lock:
            self._state = {
                **self._idle_state(),
                "running": True,
                "capture_status": "waiting_for_paired_frame",
                "session_id": session_id,
                "output_directory": str(output_dir),
                "started_at_epoch_sec": started,
                "position_label": str(position_label or "unlabeled"),
                "action_label": str(action_label or "unlabeled"),
                "trial_id": str(trial_id or "trial_001"),
                "operator_note": str(operator_note or ""),
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
        paired = int(payload.get("captured_paired_frames") or 0)
        payload["paired_frame_ratio"] = paired / denominator if denominator else None
        payload["force_semantics"] = "PX6D_reference_Fz_not_optical_force_prediction"
        payload["output_format"] = "full_spectrum_jsonl_plus_flat_summary_csv"
        return payload

    def _increment(self, field: str, amount: int = 1) -> None:
        with self._lock:
            self._state[field] = int(self._state.get(field) or 0) + amount

    def _run(self, output_dir: Path) -> None:
        frames_path = output_dir / "synchronized_frames.jsonl"
        summary_path = output_dir / "frame_summary.csv"
        last_written_key: tuple[Any, ...] | None = None
        last_candidate_key: tuple[Any, ...] | None = None
        last_missing_force_key: tuple[Any, ...] | None = None
        try:
            with frames_path.open("w", encoding="utf-8") as jsonl_handle, summary_path.open(
                "w", newline="", encoding="utf-8-sig"
            ) as csv_handle:
                writer = csv.DictWriter(csv_handle, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                while not self._stop_event.is_set():
                    frame_payload = dict(self.frame_provider() or {})
                    latest = frame_payload.get("latest")
                    wavelengths = latest.get("wavelength_nm") if isinstance(latest, dict) else None
                    intensities = latest.get("intensity") if isinstance(latest, dict) else None
                    if (
                        not isinstance(latest, dict)
                        or not isinstance(wavelengths, list)
                        or not isinstance(intensities, list)
                        or not wavelengths
                        or len(wavelengths) != len(intensities)
                    ):
                        self._increment("polls_without_valid_spectrum")
                        self._set_status("waiting_for_optical_frame")
                        self._stop_event.wait(self.poll_interval_sec)
                        continue
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
                    force = dict(self.force_provider(latest) or {})
                    force_ready = bool(force.get("ok")) and (
                        not self.require_software_tare or bool(force.get("tare_ready"))
                    )
                    if not force_ready:
                        if key != last_missing_force_key:
                            self._increment("frames_missing_force_reference")
                            last_missing_force_key = key
                        self._set_status(str(force.get("status") or "waiting_for_force_reference"))
                        self._stop_event.wait(self.poll_interval_sec)
                        continue

                    last_written_key = key
                    with self._lock:
                        capture_index = int(self._state["captured_paired_frames"])
                        labels = {
                            "position_label": self._state["position_label"],
                            "action_label": self._state["action_label"],
                            "trial_id": self._state["trial_id"],
                            "operator_note": self._state["operator_note"],
                        }
                    record = {
                        "capture_index": capture_index,
                        "capture_timestamp_epoch_sec": time.time(),
                        **labels,
                        "spectrum": {
                            "source": latest.get("source"),
                            "frame_id": latest.get("frame_id"),
                            "timestamp_epoch_sec": key[2],
                            "wavelength_nm": wavelengths,
                            "intensity_counts": intensities,
                            "spectrum_peaks": latest.get("spectrum_peaks") or [],
                        },
                        "px6d_reference": force,
                    }
                    jsonl_handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default)
                        + "\n"
                    )
                    writer.writerow(self._summary_row(record))
                    jsonl_handle.flush()
                    csv_handle.flush()
                    with self._lock:
                        self._state["captured_paired_frames"] = capture_index + 1
                        self._state["last_spectrum_frame_id"] = latest.get("frame_id")
                        self._state["last_sync_quality"] = force.get("sync_quality")
                        self._state["last_sync_offset_ms"] = force.get("sync_offset_ms")
                        self._state["capture_status"] = "recording_paired_frames"
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

    def _set_status(self, value: str) -> None:
        with self._lock:
            self._state["capture_status"] = value

    @staticmethod
    def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
        spectrum = record["spectrum"]
        force = record["px6d_reference"]
        raw = force.get("raw") or {}
        zeroed = force.get("zeroed") or {}
        mechanical = force.get("mechanical") or {}
        return {
            "capture_index": record["capture_index"],
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
            "force_resultant_n": mechanical.get("force_resultant_n"),
            "shear_resultant_n": mechanical.get("shear_resultant_n"),
            "moment_resultant_nm": mechanical.get("moment_resultant_nm"),
            "force_utilization_percent": mechanical.get("force_utilization_percent"),
            "moment_utilization_percent": mechanical.get("moment_utilization_percent"),
        }

    def _write_metadata_locked(self) -> None:
        output_directory = self._state.get("output_directory")
        if not output_directory:
            return
        output_dir = Path(str(output_directory))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.status()
        payload.update(
            {
                "schema_version": "touch_optical_px6d_sync_v1",
                "spectrum_payload": "full wavelength and intensity arrays in synchronized_frames.jsonl",
                "mechanical_payload": "raw and software-zeroed Fx/Fy/Fz/Mx/My/Mz plus derived resultants",
                "files": {
                    "full_frames_jsonl": "synchronized_frames.jsonl",
                    "frame_summary_csv": "frame_summary.csv",
                },
            }
        )
        (output_dir / "session_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
