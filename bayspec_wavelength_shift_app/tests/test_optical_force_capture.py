from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.optical_force_capture import (
    OpticalForceCaptureManager,
    _continuous_force_fz_n,
    _flush_and_sync,
    _write_text_transaction,
)


class FakeSpectrumSource:
    def __init__(self) -> None:
        self.frame_id = 0

    def frame(self) -> dict:
        self.frame_id += 1
        timestamp = time.time()
        return {
            "latest": {
                "source": "synthetic_test",
                "frame_id": self.frame_id,
                "ingested_at": timestamp,
                "wavelength_nm": [1546.8, 1546.9, 1547.0],
                "intensity": [100.0, 140.0, 105.0],
            }
        }


def aligned_force(record: dict | None) -> dict:
    timestamp = float(record["ingested_at"]) if isinstance(record, dict) else time.time()
    return {
        "ok": True,
        "status": "synced",
        "tare_ready": True,
        "sync_method": "window_median",
        "sync_quality": "excellent",
        "sync_offset_ms": 2.5,
        "sync_within_target": True,
        "calibration_sync_ok": True,
        "sample_count": 4,
        "force_sequence_start": int(timestamp * 1000),
        "force_sequence_end": int(timestamp * 1000),
        "force_timestamp_epoch_sec": timestamp + 0.0025,
        "raw": {
            "fx_n": 0.1,
            "fy_n": 0.2,
            "fz_n": -1.0,
            "mx_nm": 0.01,
            "my_nm": 0.02,
            "mz_nm": 0.03,
        },
        "zeroed": {
            "fx_n": 0.05,
            "fy_n": 0.10,
            "fz_n": -0.80,
            "mx_nm": 0.005,
            "my_nm": 0.010,
            "mz_nm": 0.015,
        },
        "filtered_zeroed": {
            "fx_n": 0.04,
            "fy_n": 0.08,
            "fz_n": -0.79,
            "mx_nm": 0.004,
            "my_nm": 0.008,
            "mz_nm": 0.012,
        },
        "reference_fz_n": 0.80,
        "reference_fz_display_n": 0.80,
        "median_reference_fz_n": 0.82,
        "filtered_reference_fz_n": 0.81,
        "drift_offset_n": 0.01,
        "drift_corrected_reference_fz_n": 0.80,
        "conditioned_reference_fz_n": 0.80,
        "stationary_detected": False,
        "auto_zero_drift_active": False,
        "force_filter_status": "contact_or_motion_filter_frozen",
        "mechanical": {
            "force_resultant_n": 0.808,
            "shear_resultant_n": 0.112,
            "moment_resultant_nm": 0.019,
            "force_utilization_percent": 1.6,
            "moment_utilization_percent": 0.75,
        },
        "filtered_mechanical": {
            "force_resultant_n": 0.795,
            "shear_resultant_n": 0.089,
            "moment_resultant_nm": 0.015,
            "force_utilization_percent": 1.58,
            "moment_utilization_percent": 0.60,
        },
    }


def model_response(record: dict) -> dict:
    return {
        "model_source": "dynamic_temporal_v3_test",
        "model_status": "shadow_ready",
        "model_ready": True,
        "contact": {
            "label": "contact",
            "confidence": 0.91,
            "probabilities": {"no_contact": 0.09, "contact": 0.91},
        },
        "position": {
            "label": "P21",
            "confidence": 0.62,
            "visual_label": "P22",
            "visual_confidence": 0.82,
            "visual_margin": 0.41,
            "raw_label": "P13",
        },
        "response_level": {
            "label": "normal",
            "confidence": 0.75,
            "raw_label": "normal",
            "decision_rule": "test",
            "probabilities": {"light": 0.15, "normal": 0.75, "hard": 0.10},
        },
        "operational_state": "active_contact",
        "release_guard": {"release_latched": False},
        "runtime_baseline_revision": 2,
        "estimated_force_fz_n": 0.73,
        "force_fz": {
            "estimated_n": 0.73,
            "raw_estimated_n": 0.77,
            "visual_drive_n": 0.71,
            "unit": "N",
            "gated": False,
            "runtime_input": "optical_spectrum_time_series",
            "calibration_supervision": "PX6D Fz",
        },
        "uncertainty": {
            "review_needed": True,
            "reasons": ["position_confidence_low"],
        },
        "inference_latency_ms": 3.4,
        "digital_twin": {
            "visual_active": True,
            "position_id": "P22",
            "drive_force_n": 0.71,
            "position_source": "contact_episode_position_lock",
        },
    }


def unique_timeline(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["capture_index"]): (
            float(row["timeline_timestamp_epoch_sec"]),
            float(row["elapsed_time_sec"]),
        )
        for row in rows
    }


class OpticalForceCaptureTests(unittest.TestCase):
    def test_nondurable_transaction_defers_flush_until_batch_commit(self) -> None:
        class TrackingBuffer(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.flush_count = 0

            def flush(self) -> None:
                self.flush_count += 1
                super().flush()

        handle = TrackingBuffer()
        _write_text_transaction([(handle, "frame-one\n")], durable=False)
        self.assertEqual(handle.getvalue(), "frame-one\n")
        self.assertEqual(handle.flush_count, 0)

        _write_text_transaction([(handle, "frame-two\n")], durable=True)
        self.assertEqual(handle.getvalue(), "frame-one\nframe-two\n")
        self.assertEqual(handle.flush_count, 1)

    def test_deferred_response_rows_preserve_raw_capture_rate_and_audit_state(
        self,
    ) -> None:
        def deferred_response(_record: dict) -> dict:
            return {
                "model_source": "deployed_test_model",
                "model_status": "capture_response_deferred",
                "model_ready": False,
                "capture_response_source": "deferred_for_high_rate_capture",
                "capture_response_frame_match": False,
                "capture_response_deferred_reason": (
                    "same_frame_runtime_prediction_not_cached"
                ),
                "raw_spectrum_authoritative": True,
                "offline_reconstruction_supported": True,
            }

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                model_provider=deferred_response,
                poll_interval_sec=0.01,
                durable_flush_interval_sec=0.25,
                durable_flush_frame_count=10,
            )
            self.assertTrue(manager.start()["ok"])
            time.sleep(0.08)
            stopped = manager.stop()

            frame_count = int(stopped["captured_timeline_frames"])
            self.assertGreater(frame_count, 0)
            self.assertEqual(stopped["response_deferred_rows"], frame_count)
            self.assertEqual(stopped["response_same_frame_cache_rows"], 0)
            self.assertIsNotNone(stopped["captured_frame_rate_hz"])
            self.assertEqual(stopped["last_durable_capture_index"], frame_count - 1)
            self.assertGreaterEqual(stopped["durable_flush_count"], 1)

            output_dir = Path(stopped["output_directory"])
            with (output_dir / "tactile_response_timeseries.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["capture_response_source"],
                "deferred_for_high_rate_capture",
            )
            self.assertEqual(row["capture_response_frame_match"], "False")
            self.assertEqual(row["raw_spectrum_authoritative"], "True")
            self.assertEqual(row["offline_reconstruction_supported"], "True")
            self.assertEqual(row["model_timestamp_epoch_sec"], "")

    def test_response_only_retains_raw_spectrum_in_lossless_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                model_provider=lambda _record: {
                    "model_status": "capture_response_deferred",
                    "model_ready": False,
                    "capture_response_source": "deferred_for_high_rate_capture",
                    "raw_spectrum_authoritative": True,
                    "offline_reconstruction_supported": True,
                },
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["response"])["ok"])
            time.sleep(0.04)
            stopped = manager.stop()

            output_dir = Path(stopped["output_directory"])
            self.assertFalse((output_dir / "spectrum_timeseries.csv").exists())
            with (output_dir / "synchronized_frames.jsonl").open(
                encoding="utf-8"
            ) as handle:
                first = json.loads(handle.readline())
            self.assertEqual(first["spectrum"]["wavelength_nm"], [1546.8, 1546.9, 1547.0])
            self.assertEqual(first["spectrum"]["intensity_counts"], [100.0, 140.0, 105.0])

    def test_force_target_preserves_continuous_fz_without_class_binning(self) -> None:
        self.assertAlmostEqual(
            _continuous_force_fz_n({"conditioned_reference_fz_n": 0.347}),
            0.347,
        )
        self.assertAlmostEqual(
            _continuous_force_fz_n({"conditioned_reference_fz_n": 1.284}),
            1.284,
        )
        self.assertEqual(
            _continuous_force_fz_n({"conditioned_reference_fz_n": -0.015}),
            0.0,
        )

    def test_capture_rejects_missing_software_tare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": False},
            )
            result = manager.start()
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "px6d_software_tare_required")

    def test_capture_reports_force_status_provider_failure(self) -> None:
        def unavailable_status() -> dict:
            raise OSError("serial status unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=unavailable_status,
            )
            result = manager.start(selected_outputs=["force"])

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "px6d_status_unavailable")
            self.assertIn("serial status unavailable", result["reason"])
            self.assertFalse(manager.status()["worker_alive"])

    def test_concurrent_start_requests_launch_only_one_capture_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_status_entered = threading.Event()
            release_first_status = threading.Event()
            status_call_lock = threading.Lock()
            status_call_count = 0

            def force_status() -> dict:
                nonlocal status_call_count
                with status_call_lock:
                    status_call_count += 1
                    call_number = status_call_count
                if call_number == 1:
                    first_status_entered.set()
                    release_first_status.wait(timeout=2.0)
                return {"connected": True, "tare_ready": True}

            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=force_status,
                poll_interval_sec=0.01,
            )
            results: list[dict] = []
            results_lock = threading.Lock()

            def start_capture() -> None:
                result = manager.start(trial_id="concurrent_start")
                with results_lock:
                    results.append(result)

            first_caller = threading.Thread(target=start_capture)
            first_caller.start()
            self.assertTrue(first_status_entered.wait(timeout=2.0))
            second_result = manager.start(trial_id="concurrent_start")
            with results_lock:
                results.append(second_result)
            release_first_status.set()
            first_caller.join(timeout=3.0)

            try:
                self.assertEqual(len(results), 2)
                self.assertEqual(sum(bool(result.get("ok")) for result in results), 1)
                started = next(result for result in results if result.get("ok"))
                self.assertFalse(started["start_in_progress"])
                self.assertFalse(started["start_cancel_requested"])
                self.assertTrue(started["worker_alive"])
                rejected = next(result for result in results if not result.get("ok"))
                self.assertIn(
                    rejected["status"],
                    {"capture_start_in_progress", "capture_already_running"},
                )
            finally:
                manager.stop()

    def test_stop_cancels_start_before_capture_thread_can_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            force_status_entered = threading.Event()
            release_force_status = threading.Event()

            def slow_force_status() -> dict:
                force_status_entered.set()
                release_force_status.wait(timeout=2.0)
                return {"connected": True, "tare_ready": True}

            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=slow_force_status,
                poll_interval_sec=0.01,
            )
            start_result: dict = {}
            stop_result: dict = {}

            def start_capture() -> None:
                start_result.update(manager.start(trial_id="cancel_during_start"))

            def stop_capture() -> None:
                stop_result.update(manager.stop())

            starter = threading.Thread(target=start_capture)
            stopper = threading.Thread(target=stop_capture)
            starter.start()
            self.assertTrue(force_status_entered.wait(timeout=2.0))
            stopper.start()

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if manager.status()["start_cancel_requested"]:
                    break
                time.sleep(0.005)
            self.assertTrue(manager.status()["start_cancel_requested"])
            release_force_status.set()
            starter.join(timeout=3.0)
            stopper.join(timeout=3.0)

            self.assertFalse(starter.is_alive())
            self.assertFalse(stopper.is_alive())
            self.assertFalse(start_result["ok"])
            self.assertEqual(start_result["status"], "capture_start_cancelled")
            self.assertTrue(stop_result["ok"])
            self.assertEqual(stop_result["status"], "capture_start_cancelled")
            final_status = manager.status()
            self.assertFalse(final_status["running"])
            self.assertFalse(final_status["worker_alive"])
            self.assertFalse(final_status["start_in_progress"])
            self.assertFalse(final_status["start_cancel_requested"])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_stop_timeout_does_not_report_a_live_writer_as_stopped(self) -> None:
        class StuckCaptureThread:
            def __init__(self) -> None:
                self.join_timeout: float | None = None

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                self.join_timeout = timeout

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            stuck = StuckCaptureThread()
            with manager._lock:
                manager._thread = stuck  # type: ignore[assignment]
                manager._state["running"] = True
                manager._state["capture_status"] = "recording_selected_streams"

            result = manager.stop()

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "capture_stop_timeout")
            self.assertTrue(result["running"])
            self.assertEqual(result["capture_status"], "stop_timeout")
            self.assertEqual(result["last_error"], "capture_thread_stop_timeout")
            self.assertEqual(stuck.join_timeout, 4.0)

    def test_concurrent_stop_requests_leave_one_complete_valid_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["spectrum"])["ok"])
            time.sleep(0.04)
            results: list[dict] = []
            results_lock = threading.Lock()

            def stop_capture() -> None:
                result = manager.stop()
                with results_lock:
                    results.append(result)

            callers = [threading.Thread(target=stop_capture) for _ in range(8)]
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join(timeout=3.0)

            self.assertTrue(all(not caller.is_alive() for caller in callers))
            self.assertEqual(len(results), len(callers))
            self.assertTrue(all(result["ok"] for result in results), results)
            self.assertTrue(
                all(
                    result["status"] in {"capture_stopped", "capture_not_running"}
                    for result in results
                ),
                results,
            )
            final_status = manager.status()
            self.assertFalse(final_status["running"])
            self.assertFalse(final_status["worker_alive"])
            output_dir = Path(final_status["output_directory"])
            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["capture_status"], "complete")
            self.assertTrue(metadata["alignment_audit"]["all_selected_streams_aligned"])

    def test_repeated_start_stop_cycles_do_not_reuse_session_or_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            session_ids: set[str] = set()
            for _index in range(8):
                started = manager.start(
                    selected_outputs=["spectrum"],
                    trial_id="repeat_cycle",
                )
                self.assertTrue(started["ok"], started)
                time.sleep(0.025)
                stopped = manager.stop()
                self.assertTrue(stopped["ok"], stopped)
                self.assertFalse(stopped["worker_alive"])
                self.assertGreater(stopped["captured_timeline_frames"], 0)
                session_ids.add(str(stopped["session_id"]))

            self.assertEqual(len(session_ids), 8)
            self.assertEqual(len(list(Path(temporary).iterdir())), 8)

    def test_unexpected_worker_exit_clears_running_and_thread_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "unexpected_exit"
            output_dir.mkdir()

            def terminate_worker() -> dict:
                raise SystemExit("simulated worker termination")

            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=terminate_worker,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            with manager._lock:
                manager._thread = threading.current_thread()
                manager._state["running"] = True
                manager._state["capture_status"] = "recording_selected_streams"
                manager._state["output_directory"] = str(output_dir)
                manager._state["selected_outputs"] = ["spectrum"]

            with self.assertRaises(SystemExit):
                manager._run(output_dir)

            status = manager.status()
            self.assertFalse(status["running"])
            self.assertFalse(status["worker_alive"])
            self.assertEqual(status["capture_status"], "capture_error")
            self.assertEqual(
                status["last_error"],
                "capture_worker_exited_unexpectedly",
            )

    def test_stop_after_worker_failure_does_not_report_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            with manager._lock:
                manager._thread = None
                manager._state["running"] = False
                manager._state["capture_status"] = "capture_error"
                manager._state["last_error"] = "OSError: disk full"

            result = manager.stop()

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "capture_failed")
            self.assertEqual(result["capture_status"], "capture_error")
            self.assertEqual(result["last_error"], "OSError: disk full")

    def test_metadata_replace_is_atomic_and_preserves_previous_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "atomic_metadata"
            output_dir.mkdir()
            metadata_path = output_dir / "session_metadata.json"
            metadata_path.write_text('{"sentinel":"previous"}', encoding="utf-8")
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            with manager._lock:
                manager._state["output_directory"] = str(output_dir)
                manager._state["selected_outputs"] = ["spectrum"]

            with patch(
                "backend.optical_force_capture.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    with manager._lock:
                        manager._write_metadata_locked()

            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                '{"sentinel":"previous"}',
            )
            self.assertEqual(list(output_dir.glob(".session_metadata.*.tmp")), [])

    def test_frame_write_transaction_rolls_back_all_files_after_partial_failure(self) -> None:
        class FailOnceBuffer(io.StringIO):
            fail_next_write = False

            def write(self, value: str) -> int:
                if self.fail_next_write:
                    self.fail_next_write = False
                    raise OSError("simulated disk full")
                return super().write(value)

        first = io.StringIO()
        second = FailOnceBuffer()
        first.write("first_header\n")
        second.write("second_header\n")
        second.fail_next_write = True

        with self.assertRaisesRegex(OSError, "simulated disk full"):
            _write_text_transaction(
                [
                    (first, "first_frame\n"),
                    (second, "second_frame\n"),
                ]
            )

        self.assertEqual(first.getvalue(), "first_header\n")
        self.assertEqual(second.getvalue(), "second_header\n")

    def test_frame_write_transaction_flushes_operating_system_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.csv"
            second_path = Path(temporary) / "second.csv"
            with first_path.open("w+", encoding="utf-8") as first, second_path.open(
                "w+", encoding="utf-8"
            ) as second, patch(
                "backend.optical_force_capture.os.fsync",
                wraps=os.fsync,
            ) as fsync:
                _write_text_transaction(
                    [
                        (first, "first_frame\n"),
                        (second, "second_frame\n"),
                    ]
                )

            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(first_path.read_text(encoding="utf-8"), "first_frame\n")
            self.assertEqual(second_path.read_text(encoding="utf-8"), "second_frame\n")

    def test_flush_and_sync_supports_in_memory_test_handles(self) -> None:
        handle = io.StringIO()
        handle.write("frame")
        _flush_and_sync(handle)
        self.assertEqual(handle.getvalue(), "frame")

    def test_start_persists_session_identity_and_capture_journal_before_return(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.05,
            )
            started = manager.start(selected_outputs=["spectrum"])
            self.assertTrue(started["ok"], started)
            try:
                output_dir = Path(started["output_directory"])
                metadata_path = output_dir / "session_metadata.json"
                journal_path = output_dir / "capture_journal.json"

                self.assertTrue(metadata_path.is_file())
                self.assertTrue(journal_path.is_file())
                with manager._lock:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    journal = json.loads(journal_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["session_id"], started["session_id"])
                self.assertEqual(journal["session_id"], started["session_id"])
                self.assertTrue(metadata["running"])
                self.assertTrue(journal["running"])
            finally:
                manager.stop()

    def test_manager_recovers_session_left_running_by_previous_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "20260729_interrupted"
            output_dir.mkdir()
            metadata_path = output_dir / "session_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": "touch_synchronized_capture_v4",
                        "session_id": "interrupted-session",
                        "running": True,
                        "capture_status": "recording_selected_streams",
                        "captured_timeline_frames": 7,
                        "last_spectrum_frame_id": 41,
                    }
                ),
                encoding="utf-8",
            )

            manager = OpticalForceCaptureManager(
                output_root=root,
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )

            recovered = json.loads(metadata_path.read_text(encoding="utf-8"))
            journal = json.loads(
                (output_dir / "capture_journal.json").read_text(encoding="utf-8")
            )
            self.assertFalse(recovered["running"])
            self.assertEqual(recovered["capture_status"], "interrupted_recovered")
            self.assertEqual(
                recovered["recovery"]["reason"],
                "previous_process_terminated_before_clean_stop",
            )
            self.assertEqual(journal["capture_status"], "interrupted_recovered")
            status = manager.status()
            self.assertEqual(status["recovered_interrupted_session_count"], 1)
            self.assertEqual(
                status["recovered_interrupted_session_ids"],
                ["interrupted-session"],
            )

    def test_worker_write_failure_is_reported_without_committing_a_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            with patch(
                "backend.optical_force_capture._write_text_transaction",
                side_effect=OSError("simulated disk full"),
            ):
                started = manager.start(selected_outputs=["spectrum"])
                self.assertTrue(started["ok"])
                deadline = time.monotonic() + 2.0
                while manager.status()["worker_alive"] and time.monotonic() < deadline:
                    time.sleep(0.005)

            status = manager.status()
            self.assertFalse(status["worker_alive"])
            self.assertFalse(status["running"])
            self.assertEqual(status["capture_status"], "capture_error")
            self.assertIn("simulated disk full", status["last_error"])
            self.assertEqual(status["captured_timeline_frames"], 0)
            output_dir = Path(status["output_directory"])
            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["alignment_audit"]["status"], "no_frames")

            stopped = manager.stop()
            self.assertFalse(stopped["ok"])
            self.assertEqual(stopped["status"], "capture_failed")

    def test_unexpected_start_exception_is_returned_and_releases_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            with patch.object(
                manager,
                "_start_reserved",
                side_effect=OSError("simulated session directory failure"),
            ):
                result = manager.start(selected_outputs=["spectrum"])

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "capture_start_failed")
            self.assertIn("simulated session directory failure", result["last_start_error"])
            self.assertEqual(result["capture_status"], "idle")
            self.assertFalse(result["start_in_progress"])
            self.assertFalse(result["start_cancel_requested"])
            self.assertFalse(result["worker_alive"])

    def test_failed_new_start_preserves_last_completed_session_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["spectrum"])["ok"])
            time.sleep(0.035)
            completed = manager.stop()
            self.assertEqual(completed["capture_status"], "complete")

            with patch.object(
                manager,
                "_start_reserved",
                side_effect=OSError("new output unavailable"),
            ):
                failed = manager.start(selected_outputs=["spectrum"])

            self.assertFalse(failed["ok"])
            self.assertEqual(failed["status"], "capture_start_failed")
            self.assertEqual(failed["capture_status"], "complete")
            self.assertEqual(failed["session_id"], completed["session_id"])
            self.assertEqual(failed["output_directory"], completed["output_directory"])

    def test_thread_start_and_metadata_failures_are_returned_not_raised(self) -> None:
        class StartFailureThread:
            def is_alive(self) -> bool:
                return False

            def start(self) -> None:
                raise RuntimeError("thread launch denied")

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            with patch(
                "backend.optical_force_capture.threading.Thread",
                return_value=StartFailureThread(),
            ), patch.object(
                manager,
                "_write_metadata_locked",
                side_effect=[None, OSError("metadata path unavailable")],
            ):
                result = manager.start(selected_outputs=["spectrum"])

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "capture_thread_start_failed")
            self.assertEqual(result["capture_status"], "capture_error")
            self.assertIn("thread launch denied", result["last_error"])
            self.assertIn("metadata_write_failed", result["last_error"])

    def test_session_writes_full_spectrum_and_flat_six_axis_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                model_provider=model_response,
                poll_interval_sec=0.01,
            )
            started = manager.start(
                position_label="P22",
                action_label="continuous_px6d_fz_reference",
                trial_id="trial_007",
            )
            self.assertTrue(started["ok"])
            time.sleep(0.08)
            stopped = manager.stop()
            self.assertGreater(stopped["captured_paired_frames"], 0)
            output_dir = Path(stopped["output_directory"])

            with (output_dir / "synchronized_frames.jsonl").open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
            self.assertEqual(first["position_label"], "P22")
            self.assertEqual(first["action_label"], "continuous_px6d_fz_reference")
            self.assertEqual(len(first["spectrum"]["wavelength_nm"]), 3)
            self.assertEqual(first["px6d_reference"]["zeroed"]["fz_n"], -0.80)
            self.assertEqual(first["force_fz_n"], 0.80)

            with (output_dir / "frame_summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["position_label"], "P22")
            self.assertEqual(row["sync_quality"], "excellent")
            self.assertEqual(float(row["reference_fz_n"]), 0.80)
            self.assertEqual(float(row["filtered_reference_fz_n"]), 0.81)
            self.assertEqual(float(row["drift_offset_n"]), 0.01)
            self.assertEqual(float(row["conditioned_reference_fz_n"]), 0.80)
            self.assertEqual(float(row["force_fz_n"]), 0.80)
            self.assertEqual(row["force_filter_status"], "contact_or_motion_filter_frozen")
            self.assertEqual(float(row["fx_zeroed_n"]), 0.05)
            self.assertEqual(float(row["fx_filtered_n"]), 0.04)
            self.assertEqual(float(row["filtered_force_resultant_n"]), 0.795)
            self.assertEqual(row["predicted_position_label"], "P22")
            self.assertEqual(row["display_contact_active"], "True")
            self.assertEqual(row["display_position_label"], "P22")
            self.assertEqual(row["formal_position_label"], "P21")
            self.assertEqual(row["raw_position_label"], "P13")
            self.assertAlmostEqual(float(row["display_optical_force_n"]), 0.71)
            self.assertEqual(row["predicted_response_level"], "normal")
            self.assertAlmostEqual(float(row["optical_estimated_fz_n"]), 0.73)
            self.assertAlmostEqual(float(row["optical_raw_estimated_fz_n"]), 0.77)
            self.assertEqual(row["optical_force_estimate_gated"], "False")
            self.assertEqual(row["optical_force_estimate_available"], "True")
            self.assertEqual(
                row["optical_force_estimate_source"],
                "optical_spectrum_time_series",
            )
            self.assertEqual(row["optical_force_estimate_unit"], "N")
            self.assertEqual(row["optical_force_review_needed"], "True")
            self.assertEqual(
                row["optical_force_review_reasons"], "position_confidence_low"
            )
            self.assertAlmostEqual(float(row["model_inference_latency_ms"]), 3.4)

            spectrum_path = output_dir / "spectrum_timeseries.csv"
            response_path = output_dir / "tactile_response_timeseries.csv"
            force_path = output_dir / "force_timeseries.csv"
            self.assertTrue(spectrum_path.exists())
            self.assertTrue(response_path.exists())
            self.assertTrue(force_path.exists())
            spectrum_timeline = unique_timeline(spectrum_path)
            response_timeline = unique_timeline(response_path)
            force_timeline = unique_timeline(force_path)
            self.assertEqual(spectrum_timeline, response_timeline)
            self.assertEqual(spectrum_timeline, force_timeline)

            with response_path.open(encoding="utf-8-sig", newline="") as handle:
                response_row = next(csv.DictReader(handle))
            self.assertEqual(response_row["response_level"], "normal")
            self.assertAlmostEqual(float(response_row["normal_probability"]), 0.75)
            self.assertAlmostEqual(
                float(response_row["optical_estimated_fz_n"]), 0.73
            )
            self.assertAlmostEqual(
                float(response_row["optical_raw_estimated_fz_n"]), 0.77
            )
            self.assertEqual(response_row["predicted_position_label"], "P22")
            self.assertEqual(response_row["display_position_label"], "P22")
            self.assertEqual(response_row["formal_position_label"], "P21")
            self.assertEqual(response_row["raw_position_label"], "P13")
            self.assertAlmostEqual(
                float(response_row["display_optical_force_n"]), 0.71
            )

            with force_path.open(encoding="utf-8-sig", newline="") as handle:
                force_row = next(csv.DictReader(handle))
            self.assertEqual(float(force_row["force_fz_n"]), 0.80)

            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], "touch_synchronized_capture_v4")
            self.assertIn("provenance", metadata)
            self.assertEqual(
                metadata["provenance"]["start_snapshot_error"],
                "provenance_provider_not_configured",
            )
            self.assertEqual(
                metadata["provenance"]["end_snapshot_error"],
                "provenance_provider_not_configured",
            )
            self.assertEqual(metadata["force_target"]["field"], "force_fz_n")
            self.assertEqual(metadata["force_target"]["unit"], "N")
            self.assertFalse(metadata["force_target"]["categorical_bins_enabled"])
            self.assertEqual(
                metadata["paired_measurement_contract"]["estimate_field"],
                "optical_estimated_fz_n",
            )
            self.assertFalse(
                metadata["paired_measurement_contract"][
                    "force_sensor_is_model_input"
                ]
            )
            self.assertEqual(metadata["capture_status"], "complete")
            self.assertTrue(metadata["alignment_audit"]["all_selected_streams_aligned"])
            self.assertEqual(
                metadata["alignment_audit"]["stream_frame_counts"]["manifest"],
                stopped["captured_timeline_frames"],
            )
            self.assertEqual(
                metadata["alignment_audit"]["stream_frame_counts"]["summary"],
                stopped["captured_timeline_frames"],
            )

    def test_alignment_audit_detects_truncated_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["spectrum"])["ok"])
            time.sleep(0.045)
            stopped = manager.stop()
            self.assertGreater(stopped["captured_timeline_frames"], 0)
            output_dir = Path(stopped["output_directory"])
            (output_dir / "synchronized_frames.jsonl").write_text("", encoding="utf-8")

            with manager._lock:
                audit = manager._alignment_audit_locked(output_dir)

            self.assertEqual(audit["status"], "alignment_error")
            self.assertFalse(audit["all_selected_streams_aligned"])
            self.assertEqual(audit["stream_frame_counts"]["manifest"], 0)
            self.assertTrue(
                any(error.startswith("capture_index_mismatch:manifest:") for error in audit["errors"]),
                audit,
            )

    def test_session_metadata_snapshots_provenance_at_start_and_end(self) -> None:
        snapshots = iter(
            [
                {
                    "software": {
                        "version": "0.16.0",
                        "build_id": "acceptance-remediation",
                        "source_commit": "abc123",
                    },
                    "baseline": {"token": "baseline-start"},
                },
                {
                    "software": {
                        "version": "0.16.0",
                        "build_id": "acceptance-remediation",
                        "source_commit": "abc123",
                    },
                    "baseline": {"token": "baseline-end"},
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {
                    "connected": True,
                    "tare_ready": True,
                },
                provenance_provider=lambda: next(snapshots),
                poll_interval_sec=0.01,
            )
            self.assertTrue(
                manager.start(
                    selected_outputs=["spectrum", "force"],
                    position_label="P22",
                )["ok"]
            )
            time.sleep(0.04)
            stopped = manager.stop()
            metadata = json.loads(
                (
                    Path(stopped["output_directory"])
                    / "session_metadata.json"
                ).read_text(encoding="utf-8")
            )

        provenance = metadata["provenance"]
        self.assertEqual(metadata["schema_version"], "touch_synchronized_capture_v4")
        self.assertEqual(provenance["start"]["baseline"]["token"], "baseline-start")
        self.assertEqual(provenance["end"]["baseline"]["token"], "baseline-end")
        self.assertIsNone(provenance["start_snapshot_error"])
        self.assertIsNone(provenance["end_snapshot_error"])

    def test_nonfinite_spectrum_is_rejected_once_per_frame(self) -> None:
        invalid_frame = {
            "latest": {
                "source": "invalid_test",
                "frame_id": 1,
                "ingested_at": time.time(),
                "wavelength_nm": [1546.8, float("nan"), 1547.0],
                "intensity": [100.0, 140.0, 105.0],
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=lambda: invalid_frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["spectrum"])["ok"])
            time.sleep(0.04)
            stopped = manager.stop()

            self.assertEqual(stopped["captured_timeline_frames"], 0)
            self.assertEqual(stopped["invalid_spectrum_frames"], 1)
            self.assertEqual(stopped["capture_status"], "no_paired_frames")

    def test_nonfinite_force_truth_is_not_recorded(self) -> None:
        invalid_force = {
            "ok": True,
            "status": "synced",
            "tare_ready": True,
            "force_sequence_end": 1,
            "force_timestamp_epoch_sec": time.time(),
            "conditioned_reference_fz_n": float("nan"),
            "reference_fz_display_n": float("nan"),
            "drift_corrected_reference_fz_n": float("nan"),
            "reference_fz_n": float("nan"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=lambda _record: invalid_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["force"])["ok"])
            time.sleep(0.04)
            stopped = manager.stop()

            self.assertEqual(stopped["captured_timeline_frames"], 0)
            self.assertEqual(stopped["invalid_force_samples"], 1)
            self.assertEqual(stopped["capture_status"], "no_paired_frames")

    def test_nonfinite_sync_offset_is_rejected_for_calibration(self) -> None:
        def force_with_nonfinite_offset(record: dict | None) -> dict:
            payload = aligned_force(record)
            payload["sync_offset_ms"] = float("nan")
            return payload

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=force_with_nonfinite_offset,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            self.assertTrue(
                manager.start(selected_outputs=["spectrum", "force"])["ok"]
            )
            time.sleep(0.04)
            stopped = manager.stop()

            self.assertEqual(stopped["captured_timeline_frames"], 0)
            self.assertGreaterEqual(
                stopped["frames_outside_force_sync_tolerance"], 1
            )
            self.assertEqual(stopped["capture_status"], "no_paired_frames")

    def test_optional_nonfinite_values_never_write_nan_tokens(self) -> None:
        def force_with_nonfinite_diagnostic(record: dict | None) -> dict:
            payload = aligned_force(record)
            payload["optional_diagnostic_value"] = float("nan")
            return payload

        def model_with_nonfinite_probability(record: dict) -> dict:
            payload = model_response(record)
            payload["contact"]["probabilities"]["contact"] = float("nan")
            return payload

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=force_with_nonfinite_diagnostic,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                model_provider=model_with_nonfinite_probability,
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start()["ok"])
            time.sleep(0.045)
            stopped = manager.stop()
            self.assertGreater(stopped["captured_timeline_frames"], 0)
            output_dir = Path(stopped["output_directory"])

            for path in output_dir.iterdir():
                if path.suffix not in {".json", ".jsonl", ".csv"}:
                    continue
                text = path.read_text(encoding="utf-8-sig")
                self.assertNotIn("NaN", text, path.name)
                self.assertNotIn("Infinity", text, path.name)
            with (output_dir / "synchronized_frames.jsonl").open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
            self.assertIsNone(
                first["tactile_response"]["contact"]["probabilities"]["contact"]
            )

    def test_force_disconnect_during_final_metadata_does_not_lose_audit(self) -> None:
        status_calls = 0

        def status_then_disconnect() -> dict:
            nonlocal status_calls
            status_calls += 1
            if status_calls == 1:
                return {"connected": True, "tare_ready": True}
            raise OSError("force sensor disconnected")

        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=status_then_disconnect,
                poll_interval_sec=0.01,
            )
            self.assertTrue(manager.start(selected_outputs=["force"])["ok"])
            time.sleep(0.04)
            stopped = manager.stop()

            self.assertTrue(stopped["ok"])
            output_dir = Path(stopped["output_directory"])
            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertIn("force sensor disconnected", metadata["force_status_snapshot_error"])
            self.assertTrue(metadata["alignment_audit"]["all_selected_streams_aligned"])

    def test_all_seven_nonempty_output_selections_write_only_requested_csvs(self) -> None:
        combinations = [
            ["spectrum"],
            ["response"],
            ["force"],
            ["spectrum", "response"],
            ["spectrum", "force"],
            ["response", "force"],
            ["spectrum", "response", "force"],
        ]
        filenames = {
            "spectrum": "spectrum_timeseries.csv",
            "response": "tactile_response_timeseries.csv",
            "force": "force_timeseries.csv",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, selection in enumerate(combinations):
                manager = OpticalForceCaptureManager(
                    output_root=root / f"default_{index}",
                    frame_provider=FakeSpectrumSource().frame,
                    force_provider=aligned_force,
                    force_status_provider=lambda: {"connected": True, "tare_ready": True},
                    model_provider=model_response,
                    poll_interval_sec=0.01,
                )
                started = manager.start(
                    selected_outputs=selection,
                    output_root=root / f"chosen_{index}",
                    trial_id=f"combo_{index}",
                )
                self.assertTrue(started["ok"], selection)
                time.sleep(0.045)
                stopped = manager.stop()
                self.assertGreater(stopped["captured_timeline_frames"], 0, selection)
                output_dir = Path(stopped["output_directory"])
                self.assertEqual(output_dir.parent, (root / f"chosen_{index}").resolve())
                for stream, filename in filenames.items():
                    self.assertEqual((output_dir / filename).exists(), stream in selection)
                metadata = json.loads(
                    (output_dir / "session_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["selected_outputs"], selection)
                self.assertEqual(
                    metadata["timeline_basis"],
                    "px6d_force_host_epoch_clock"
                    if selection == ["force"]
                    else "spectrum_ingested_at_host_epoch_clock",
                )
                self.assertTrue(
                    metadata["alignment_audit"]["all_selected_streams_aligned"],
                    metadata["alignment_audit"],
                )

    def test_sustained_capture_keeps_all_streams_frame_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                model_provider=model_response,
                poll_interval_sec=0.01,
            )
            manager.poll_interval_sec = 0.001
            self.assertTrue(manager.start()["ok"])
            deadline = time.monotonic() + 5.0
            while (
                manager.status()["captured_timeline_frames"] < 300
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            stopped = manager.stop()

            self.assertTrue(stopped["ok"], stopped)
            frame_count = int(stopped["captured_timeline_frames"])
            self.assertGreaterEqual(frame_count, 300)
            output_dir = Path(stopped["output_directory"])
            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metadata["alignment_audit"]["all_selected_streams_aligned"])
            self.assertEqual(
                metadata["alignment_audit"]["stream_frame_counts"],
                {
                    "manifest": frame_count,
                    "summary": frame_count,
                    "spectrum": frame_count,
                    "response": frame_count,
                    "force": frame_count,
                },
            )

            with (output_dir / "spectrum_timeseries.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                spectrum_row_count = sum(1 for _row in csv.DictReader(handle))
            self.assertEqual(spectrum_row_count, frame_count * 3)

    def test_rejects_empty_selection_and_unwritable_output_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = OpticalForceCaptureManager(
                output_root=root,
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            empty = manager.start(selected_outputs=[])
            self.assertFalse(empty["ok"])
            self.assertEqual(empty["status"], "capture_output_selection_invalid")

            misspelled = manager.start(selected_outputs=["spectrum", "froce"])
            self.assertFalse(misspelled["ok"])
            self.assertEqual(
                misspelled["status"],
                "capture_output_selection_invalid",
            )
            self.assertIn("unknown selected output(s): froce", misspelled["reason"])

            not_a_directory = root / "file.txt"
            not_a_directory.write_text("occupied", encoding="utf-8")
            invalid = manager.start(
                selected_outputs=["spectrum"],
                output_root=not_a_directory,
            )
            self.assertFalse(invalid["ok"])
            self.assertEqual(invalid["status"], "capture_output_directory_invalid")

    def test_spectrum_only_does_not_require_px6d_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=lambda _: {"ok": False},
                force_status_provider=lambda: {"connected": False, "tare_ready": False},
                poll_interval_sec=0.01,
            )
            started = manager.start(selected_outputs=["spectrum"])
            self.assertTrue(started["ok"])
            time.sleep(0.035)
            stopped = manager.stop()
            self.assertGreater(stopped["captured_timeline_frames"], 0)


if __name__ == "__main__":
    unittest.main()
