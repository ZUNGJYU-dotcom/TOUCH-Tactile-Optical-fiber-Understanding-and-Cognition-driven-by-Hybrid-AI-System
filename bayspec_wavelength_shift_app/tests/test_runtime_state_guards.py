from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from backend import main as backend_main
from backend.main import SenseExportWatcher, _model_display_source_gate
from bridge import BaySpecWavelengthShiftBridge
from desktop_launcher import (
    health_payload_is_expected,
    load_application_when_ready,
    port_is_free,
    run_backend,
    run_self_test,
    select_backend_port,
    startup_document,
    stop_owned_backend,
    wait_until_ready,
)
from process_lifecycle import (
    attach_process_to_kill_on_close_job,
    close_windows_handle,
)
from sdk_live import DEFAULT_INTERVAL_MS, BaySpecSdkLiveReader


class _AliveThread:
    def is_alive(self) -> bool:
        return True


class _JoinForbiddenThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        raise AssertionError("idempotent start must not join the active worker")


class _NeverStoppingThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return None


class _BlockingJoinThread:
    def __init__(self) -> None:
        self.join_entered = threading.Event()
        self.release_join = threading.Event()
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_entered.set()
        if self.release_join.wait(timeout=timeout):
            self._alive = False


class _ThreadStartFailure:
    def is_alive(self) -> bool:
        return False

    def start(self) -> None:
        raise RuntimeError("thread launch denied")


class _OrphanableSdkProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self.wait_called = False

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return ('{"type":"spectrum","counts":[1,2,3]}\n', "")

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_called = True
        self.returncode = -15
        return self.returncode


class _CompletedSdkProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout_payload = stdout
        self.stderr_payload = stderr
        self.returncode = returncode
        self.pid = 12346

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return self.stdout_payload, self.stderr_payload

    def wait(self, timeout=None):
        return self.returncode


class _UnkillableSdkProcess:
    pid = 12347
    returncode = None

    def poll(self):
        return None

    def kill(self) -> None:
        raise OSError("access denied")

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("unkillable-helper", timeout)


class _BridgeStub:
    def __init__(self) -> None:
        self.payloads = []

    def ingest(self, _payload):
        self.payloads.append(_payload)
        return {"ok": True}


class DesktopBackendShutdownTests(unittest.TestCase):
    def test_owned_backend_shutdown_waits_for_worker_exit(self) -> None:
        class ServerStub:
            should_exit = False
            force_exit = False

        server = ServerStub()

        def backend_loop() -> None:
            while not server.should_exit:
                time.sleep(0.005)

        worker = threading.Thread(target=backend_loop)
        worker.start()
        stopped = stop_owned_backend(
            {"server": server},  # type: ignore[dict-item]
            worker,
            graceful_timeout_s=0.5,
            force_timeout_s=0.1,
        )

        self.assertTrue(stopped)
        self.assertTrue(server.should_exit)
        self.assertFalse(worker.is_alive())

    def test_uvicorn_system_exit_is_recorded_as_startup_failure(self) -> None:
        class FailingServer:
            def __init__(self, _config) -> None:
                self.should_exit = False
                self.force_exit = False

            def run(self) -> None:
                raise SystemExit(1)

        holder: dict[str, object] = {}
        fake_uvicorn = SimpleNamespace(
            Config=lambda *_args, **_kwargs: object(),
            Server=FailingServer,
        )
        with patch(
            "desktop_launcher.load_uvicorn_module",
            return_value=fake_uvicorn,
        ), patch("desktop_launcher.write_log") as write_log:
            run_backend(8640, holder)

        self.assertEqual(holder["startup_error"], "SystemExit: 1")
        self.assertIn("SystemExit: 1", str(holder["startup_traceback"]))
        write_log.assert_called_once()

    def test_dead_backend_thread_fails_readiness_without_full_timeout(self) -> None:
        class DeadThread:
            @staticmethod
            def is_alive() -> bool:
                return False

        started = time.perf_counter()
        with patch(
            "desktop_launcher._read_expected_health",
            side_effect=ConnectionRefusedError("bind failed"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Backend exited before becoming ready: SystemExit: 1",
            ):
                wait_until_ready(
                    "http://127.0.0.1:8640/api/health",
                    timeout_s=10.0,
                    backend_thread=DeadThread(),  # type: ignore[arg-type]
                    server_holder={"startup_error": "SystemExit: 1"},
                )

        self.assertLess(time.perf_counter() - started, 0.5)

    def test_expected_existing_backend_wins_a_port_start_race(self) -> None:
        class DeadThread:
            @staticmethod
            def is_alive() -> bool:
                return False

        with patch("desktop_launcher._read_expected_health", return_value=True):
            wait_until_ready(
                "http://127.0.0.1:8640/api/health",
                timeout_s=1.0,
                backend_thread=DeadThread(),  # type: ignore[arg-type]
                server_holder={"startup_error": "SystemExit: 1"},
            )

    def test_startup_document_is_lightweight_and_operator_facing(self) -> None:
        document = startup_document(Path(__file__).resolve().parents[1])

        self.assertIn("<strong>TOUCH</strong>", document)
        self.assertIn("<h1>TOUCH</h1>", document)
        self.assertIn(
            "Tactile Optical-fiber Understanding and Cognition driven by "
            "Hybrid-AI System",
            document,
        )
        self.assertIn('<span class="status-title">Starting</span>', document)
        self.assertIn("Preparing workspace", document)
        self.assertIn('src="data:image/png;base64,', document)
        self.assertIn('class="logo-stage"', document)
        self.assertIn('class="contact-ball"', document)
        self.assertIn('class="body-shape morph-shell"', document)
        self.assertIn('class="cavity-wall"', document)
        self.assertIn("@keyframes contact-press", document)
        self.assertIn("@keyframes shell-morph", document)
        self.assertIn("@keyframes elastic-recoil", document)
        self.assertIn("@keyframes cavity-capture", document)
        self.assertIn("@keyframes haptic-pulse", document)
        self.assertIn("@keyframes settle-glow", document)
        self.assertIn("#78E2FE", document)
        self.assertIn("#FE9985", document)
        self.assertIn("#075C73", document)
        self.assertNotIn("startup-pocket", document)
        self.assertNotIn("startup-pocket-depth", document)
        self.assertNotIn("#023D50", document)
        self.assertNotIn('class="grip-lobes"', document)
        self.assertNotIn('class="indentation-mask"', document)
        self.assertIn("scaleX(0.84) scaleY(1.12)", document)
        self.assertNotIn('stroke="#159DBE" stroke-width="14"', document)
        self.assertIn('class="progress"', document)
        self.assertNotIn("file:///", document)
        self.assertNotIn("uvicorn", document)
        self.assertNotIn("model bundle", document)

    def test_visible_startup_window_navigates_after_backend_is_ready(self) -> None:
        class AliveThread:
            @staticmethod
            def is_alive() -> bool:
                return True

        class Window:
            loaded_url: str | None = None

            def load_url(self, url: str) -> None:
                self.loaded_url = url

            def load_html(self, _html: str) -> None:
                raise AssertionError("successful startup must not load the error page")

        window = Window()
        ownership = {"owns_backend": True}
        with patch("desktop_launcher.wait_until_ready"), patch(
            "desktop_launcher.write_log"
        ):
            load_application_when_ready(
                window,
                app_root=Path(__file__).resolve().parents[1],
                app_url="http://127.0.0.1:8640/?desktop=1",
                health_url="http://127.0.0.1:8640/api/health",
                backend_thread=AliveThread(),  # type: ignore[arg-type]
                server_holder={},
                ownership=ownership,
                started_at=time.perf_counter(),
            )

        self.assertEqual(window.loaded_url, "http://127.0.0.1:8640/?desktop=1")
        self.assertTrue(ownership["owns_backend"])

    def test_startup_failure_stays_visible_with_a_close_action(self) -> None:
        class Window:
            loaded_html = ""

            def load_url(self, _url: str) -> None:
                raise AssertionError("failed startup must not navigate to the app")

            def load_html(self, html: str) -> None:
                self.loaded_html = html

        window = Window()
        with patch(
            "desktop_launcher.wait_until_ready",
            side_effect=RuntimeError("backend failed"),
        ), patch("desktop_launcher.write_log"):
            load_application_when_ready(
                window,
                app_root=Path(__file__).resolve().parents[1],
                app_url="http://127.0.0.1:8640/?desktop=1",
                health_url="http://127.0.0.1:8640/api/health",
                backend_thread=None,
                server_holder={},
                ownership={"owns_backend": False},
                started_at=time.perf_counter(),
            )

        self.assertIn("Unable to start", window.loaded_html)
        self.assertIn("close_window", window.loaded_html)


class RuntimeResponsivenessConfigTests(unittest.TestCase):
    def test_temporal_history_preroll_is_enabled_for_live_runtime(self) -> None:
        config = backend_main._load_runtime_baseline_recovery_config()

        self.assertTrue(config["prime_temporal_history_with_baseline"])
        self.assertEqual(config["baseline_preroll_frames"], 20)


class ManualIngestRequestLimitTests(unittest.IsolatedAsyncioTestCase):
    class StreamingRequest:
        def __init__(self, chunks: list[bytes], content_length: int | None = None) -> None:
            self._chunks = chunks
            self.stream_calls = 0
            self.headers = (
                {"content-length": str(content_length)}
                if content_length is not None
                else {}
            )

        async def stream(self):
            self.stream_calls += 1
            for chunk in self._chunks:
                yield chunk

    async def test_declared_oversized_ingest_is_rejected_before_body_streaming(self) -> None:
        request = self.StreamingRequest([], content_length=33)
        with patch.object(backend_main, "MAX_MANUAL_INGEST_BODY_BYTES", 32):
            response = await backend_main.ingest(request)  # type: ignore[arg-type]

        self.assertEqual(response.status_code, 413)
        self.assertEqual(request.stream_calls, 0)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "ingest_request_too_large")

    async def test_chunked_oversized_csv_is_rejected_during_streaming(self) -> None:
        request = self.StreamingRequest([b"x" * 20, b"y" * 20])
        with patch.object(backend_main, "MAX_MANUAL_INGEST_BODY_BYTES", 32):
            response = await backend_main.ingest_csv(request)  # type: ignore[arg-type]

        self.assertEqual(response.status_code, 413)
        self.assertEqual(request.stream_calls, 1)
        payload = json.loads(response.body)
        self.assertEqual(payload["maximum_body_bytes"], 32)

    async def test_valid_json_ingest_still_reaches_bridge(self) -> None:
        request = self.StreamingRequest(
            [json.dumps({"channel_id": "P22", "intensity_counts": 123.0}).encode("utf-8")]
        )
        with patch.object(
            backend_main,
            "_manual_ingest_source_conflict",
            return_value=None,
        ), patch.object(
            backend_main.bridge,
            "ingest",
            return_value={"ok": True, "records_ingested": 1},
        ) as ingest:
            response = await backend_main.ingest(request)  # type: ignore[arg-type]

        self.assertTrue(response["ok"])
        ingest.assert_called_once_with({"channel_id": "P22", "intensity_counts": 123.0})


class CaptureApiThreadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_start_disk_work_does_not_block_event_loop(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class JsonRequest:
            async def json(self) -> dict:
                return {
                    "position_label": "P22",
                    "trial_id": "event_loop_probe",
                    "selected_outputs": ["spectrum"],
                }

        def slow_start(**_kwargs) -> dict:
            entered.set()
            release.wait(timeout=1.0)
            return {"ok": True, "status": "capture_started"}

        timer = threading.Timer(0.35, release.set)
        timer.start()
        try:
            with patch.object(
                backend_main.optical_force_capture,
                "start",
                side_effect=slow_start,
            ):
                started_at = time.monotonic()
                request_task = asyncio.create_task(
                    backend_main.px6d_capture_start(JsonRequest())  # type: ignore[arg-type]
                )
                await asyncio.sleep(0.04)
                event_loop_delay = time.monotonic() - started_at
                self.assertTrue(entered.is_set())
                self.assertLess(event_loop_delay, 0.15)
                release.set()
                result = await asyncio.wait_for(request_task, timeout=1.0)
        finally:
            release.set()
            timer.cancel()

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "capture_started")

    async def test_capture_start_rejects_malformed_json_without_starting(self) -> None:
        class InvalidJsonRequest:
            async def json(self) -> dict:
                raise ValueError("broken JSON")

        with patch.object(backend_main.optical_force_capture, "start") as start:
            result = await backend_main.px6d_capture_start(  # type: ignore[arg-type]
                InvalidJsonRequest()
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "capture_request_invalid")
        start.assert_not_called()

    async def test_capture_start_rejects_non_object_json(self) -> None:
        class ListJsonRequest:
            async def json(self) -> list[str]:
                return ["spectrum"]

        with patch.object(backend_main.optical_force_capture, "start") as start:
            result = await backend_main.px6d_capture_start(  # type: ignore[arg-type]
                ListJsonRequest()
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "capture_request_invalid")
        start.assert_not_called()

    async def test_force_zero_is_rejected_while_capture_is_running(self) -> None:
        with patch.object(
            backend_main.optical_force_capture,
            "status",
            return_value={"running": True, "start_in_progress": False, "worker_alive": True},
        ), patch.object(backend_main.px6d_reader, "tare") as tare:
            result = backend_main.px6d_tare(duration_sec=1.0)

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "recording_active")
        self.assertEqual(
            result["reason"],
            "stop_synchronized_recording_before_force_zero",
        )
        tare.assert_not_called()

    async def test_capture_start_is_rejected_while_force_zero_control_is_busy(self) -> None:
        class JsonRequest:
            async def json(self) -> dict:
                return {
                    "position_label": "P22",
                    "trial_id": "zero_guard_probe",
                    "selected_outputs": ["spectrum", "force"],
                }

        acquired = backend_main.PX6D_CAPTURE_CONTROL_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            with patch.object(backend_main.optical_force_capture, "start") as start:
                result = await backend_main.px6d_capture_start(  # type: ignore[arg-type]
                    JsonRequest()
                )
        finally:
            backend_main.PX6D_CAPTURE_CONTROL_LOCK.release()

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "force_zero_in_progress")
        start.assert_not_called()


class SdkLiveReaderStateTests(unittest.TestCase):
    def _reader(self) -> BaySpecSdkLiveReader:
        return BaySpecSdkLiveReader(_BridgeStub(), Path("."))

    def test_default_interval_uses_low_latency_idle_budget(self) -> None:
        reader = self._reader()

        self.assertEqual(DEFAULT_INTERVAL_MS, 10)
        self.assertEqual(reader.interval_ms, DEFAULT_INTERVAL_MS)

    def test_one_frame_helper_does_not_sleep_after_final_frame(self) -> None:
        helper_source = (
            Path(__file__).resolve().parents[1]
            / "sdk_probe"
            / "BaySpecSdkStream.cs"
        ).read_text(encoding="utf-8")

        self.assertIn("if (maxFrames <= 0 || frame < maxFrames)", helper_source)
        self.assertIn("Thread.Sleep(intervalMs);", helper_source)

    def test_start_is_idempotent_while_session_is_active(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.thread = _JoinForbiddenThread()
        reader.channel_id = "P22"

        status = reader.start(channel_id="P22", interval_ms=100, integration=40000)

        self.assertTrue(status["active"])
        self.assertEqual(status["freshness"], "waiting_for_sdk_frame")

    def test_pre_frame_helper_failure_is_reported_as_error(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.last_error = "failed to start SDK helper"

        status = reader.status()

        self.assertEqual(status["freshness"], "error")

    def test_frame_freshness_tracks_measured_acquisition_cycle(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.thread = _JoinForbiddenThread()
        reader.interval_ms = 20
        reader.last_acquisition_duration_ms = 320.0
        reader.last_frame_time = 100.0

        with patch("sdk_live.time.time", return_value=101.0):
            live = reader.status()
        with patch("sdk_live.time.time", return_value=101.3):
            stale = reader.status()

        self.assertAlmostEqual(live["frame_freshness_limit_sec"], 1.25)
        self.assertEqual(live["freshness"], "live")
        self.assertEqual(stale["freshness"], "stale")

    def test_frame_freshness_expands_for_intentionally_slow_sampling(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.thread = _JoinForbiddenThread()
        reader.interval_ms = 2000
        reader.last_frame_time = 100.0

        with patch("sdk_live.time.time", return_value=106.0):
            status = reader.status()

        self.assertAlmostEqual(status["frame_freshness_limit_sec"], 7.0)
        self.assertEqual(status["freshness"], "live")

    def test_missing_helper_fails_without_fake_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = BaySpecSdkLiveReader(_BridgeStub(), Path(temp_dir))

            result = reader.start()

        self.assertFalse(result["ok"])
        self.assertEqual(result["operation_status"], "helper_unavailable")
        self.assertFalse(result["active"])
        self.assertFalse(result["worker_alive"])
        self.assertFalse(result["stop_requested"])
        self.assertEqual(result["lifecycle_status"], "unavailable")

    def test_thread_launch_failure_rolls_back_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            helper = Path(temp_dir) / "sdk_probe" / "BaySpecSdkStream.exe"
            helper.parent.mkdir(parents=True)
            helper.touch()
            reader = BaySpecSdkLiveReader(_BridgeStub(), Path(temp_dir))
            with patch("sdk_live.threading.Thread", return_value=_ThreadStartFailure()):
                result = reader.start()

        self.assertFalse(result["ok"])
        self.assertEqual(result["operation_status"], "start_failed")
        self.assertFalse(result["active"])
        self.assertFalse(result["requested_active"])
        self.assertFalse(result["worker_alive"])
        self.assertEqual(result["lifecycle_status"], "start_failed")

    def test_restart_reaps_orphan_helper_before_new_worker_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            helper = Path(temp_dir) / "sdk_probe" / "BaySpecSdkStream.exe"
            helper.parent.mkdir(parents=True)
            helper.touch()
            orphan = _OrphanableSdkProcess()
            reader = BaySpecSdkLiveReader(_BridgeStub(), Path(temp_dir))
            reader.process = orphan
            with patch("sdk_live.threading.Thread", return_value=_ThreadStartFailure()):
                result = reader.start()

        self.assertTrue(orphan.kill_called)
        self.assertTrue(orphan.wait_called)
        self.assertFalse(result["process_running"])
        self.assertEqual(result["operation_status"], "start_failed")

    def test_restart_is_blocked_when_orphan_helper_cannot_be_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            helper = Path(temp_dir) / "sdk_probe" / "BaySpecSdkStream.exe"
            helper.parent.mkdir(parents=True)
            helper.touch()
            reader = BaySpecSdkLiveReader(_BridgeStub(), Path(temp_dir))
            reader.process = _UnkillableSdkProcess()
            with patch("sdk_live.threading.Thread") as worker_factory:
                result = reader.start()

        self.assertFalse(result["ok"])
        self.assertEqual(result["operation_status"], "orphan_cleanup_failed")
        self.assertTrue(result["process_running"])
        worker_factory.assert_not_called()

    def test_stop_timeout_is_reported_and_blocks_restart(self) -> None:
        reader = self._reader()
        reader.thread = _NeverStoppingThread()
        reader.desired_active = True
        reader.lifecycle_status = "running"

        stopped = reader.stop()
        restarted = reader.start()

        self.assertFalse(stopped["ok"])
        self.assertEqual(stopped["operation_status"], "stop_timeout")
        self.assertTrue(stopped["worker_alive"])
        self.assertEqual(stopped["freshness"], "stopping")
        self.assertFalse(restarted["ok"])
        self.assertEqual(
            restarted["operation_status"],
            "previous_worker_stop_timeout",
        )

    def test_stop_waits_for_in_progress_start_cancellation(self) -> None:
        reader = self._reader()
        previous_worker = _BlockingJoinThread()
        reader.thread = previous_worker
        start_result: dict[str, object] = {}
        stop_result: dict[str, object] = {}

        start_thread = threading.Thread(
            target=lambda: start_result.update(reader.start()),
        )
        start_thread.start()
        self.assertTrue(previous_worker.join_entered.wait(timeout=1.0))

        stop_thread = threading.Thread(
            target=lambda: stop_result.update(reader.stop()),
        )
        stop_thread.start()
        time.sleep(0.05)
        self.assertTrue(stop_thread.is_alive())
        self.assertTrue(reader.status()["start_cancel_requested"])

        previous_worker.release_join.set()
        start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

        self.assertFalse(start_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(start_result["operation_status"], "start_cancelled")
        self.assertEqual(stop_result["operation_status"], "start_cancelled")
        self.assertFalse(stop_result["start_in_progress"])
        self.assertFalse(stop_result["start_cancel_requested"])
        self.assertFalse(stop_result["worker_alive"])

    def test_stale_generation_spectrum_is_not_ingested(self) -> None:
        bridge = _BridgeStub()
        reader = BaySpecSdkLiveReader(bridge, Path("."))
        reader.desired_active = True
        reader.generation = 4
        reader._stop_event.clear()

        accepted = reader._handle_message(
            {"type": "spectrum", "counts": [1.0, 2.0, 3.0]},
            generation=3,
        )

        self.assertFalse(accepted)
        self.assertEqual(bridge.payloads, [])
        self.assertEqual(reader.stale_session_frame_count, 1)

    def test_current_generation_spectrum_is_ingested(self) -> None:
        bridge = _BridgeStub()
        reader = BaySpecSdkLiveReader(bridge, Path("."))
        reader.desired_active = True
        reader.generation = 4
        reader._stop_event.clear()
        with patch.object(reader, "_wavelength_grid", return_value=[1546.8, 1546.9, 1547.0]):
            accepted = reader._handle_message(
                {"type": "spectrum", "counts": [1.0, 2.0, 3.0]},
                generation=4,
            )

        self.assertTrue(accepted)
        self.assertEqual(len(bridge.payloads), 1)
        self.assertEqual(reader.frame_count, 1)
        self.assertEqual(reader.received_frame_count, 1)

    def test_stop_cannot_overtake_an_accepted_frame_commit(self) -> None:
        bridge = _BridgeStub()
        commit_entered = threading.Event()
        release_commit = threading.Event()

        def blocking_ingest(payload):
            commit_entered.set()
            release_commit.wait(timeout=1.0)
            bridge.payloads.append(payload)
            return {"ok": True}

        bridge.ingest = blocking_ingest
        reader = BaySpecSdkLiveReader(bridge, Path("."))
        reader.desired_active = True
        reader.generation = 7
        reader._stop_event.clear()
        accepted: dict[str, bool] = {}
        stopped: dict[str, object] = {}

        with patch.object(
            reader,
            "_wavelength_grid",
            return_value=[1546.8, 1546.9, 1547.0],
        ):
            ingest_thread = threading.Thread(
                target=lambda: accepted.setdefault(
                    "value",
                    reader._handle_message(
                        {"type": "spectrum", "counts": [1.0, 2.0, 3.0]},
                        generation=7,
                    ),
                )
            )
            ingest_thread.start()
            self.assertTrue(commit_entered.wait(timeout=1.0))

            stop_thread = threading.Thread(
                target=lambda: stopped.update(reader.stop()),
            )
            stop_thread.start()
            time.sleep(0.05)
            self.assertTrue(stop_thread.is_alive())

            release_commit.set()
            ingest_thread.join(timeout=1.0)
            stop_thread.join(timeout=1.0)

        self.assertTrue(accepted["value"])
        self.assertFalse(ingest_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(len(bridge.payloads), 1)
        self.assertFalse(stopped["requested_active"])

    def test_unexpected_sdk_worker_exit_reaps_active_helper_process(self) -> None:
        reader = self._reader()
        process = _OrphanableSdkProcess()
        reader.desired_active = True
        reader.generation = 3
        reader.thread = threading.current_thread()
        reader._stop_event.clear()

        with patch("sdk_live.subprocess.Popen", return_value=process), patch.object(
            reader,
            "_handle_message",
            side_effect=SystemExit("simulated SDK worker termination"),
        ):
            with self.assertRaises(SystemExit):
                reader._supervisor_loop(generation=3)

        status = reader.status()
        self.assertTrue(process.terminate_called)
        self.assertTrue(process.wait_called)
        self.assertFalse(process.kill_called)
        self.assertIsNone(reader.process)
        self.assertFalse(status["worker_alive"])
        self.assertFalse(status["requested_active"])
        self.assertEqual(status["lifecycle_status"], "worker_exited")

    def test_helper_stdout_flood_is_bounded_and_process_is_reaped(self) -> None:
        reader = self._reader()
        reader.MAX_HELPER_STDOUT_CHARS = 4096
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys,time; "
                    "sys.stdout.write('x' * 200000); "
                    "sys.stdout.flush(); time.sleep(5)"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )

        started = time.monotonic()
        stdout, _stderr, state = reader._collect_helper_output(process, timeout=2.0)

        self.assertEqual(state["overflow_stream"], "stdout")
        self.assertFalse(state["timed_out"])
        self.assertGreater(state["stdout_chars"], reader.MAX_HELPER_STDOUT_CHARS)
        self.assertLessEqual(len(stdout), reader.MAX_HELPER_STDOUT_CHARS)
        self.assertIsNotNone(process.poll())
        self.assertLess(time.monotonic() - started, 2.0)

    def test_hung_helper_is_killed_and_reaped_at_timeout(self) -> None:
        reader = self._reader()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )

        _stdout, _stderr, state = reader._collect_helper_output(process, timeout=0.1)

        self.assertTrue(state["timed_out"])
        self.assertIsNone(state["overflow_stream"])
        self.assertIsNotNone(process.poll())

    @unittest.skipUnless(os.name == "nt", "Windows job objects are Windows-only")
    def test_kill_on_close_job_reaps_native_helper_if_parent_cleanup_is_forced(
        self,
    ) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        handle = None
        try:
            handle = attach_process_to_kill_on_close_job(process.pid)
            self.assertIsNotNone(handle)

            close_windows_handle(handle)
            handle = None
            process.wait(timeout=3.0)

            self.assertIsNotNone(process.poll())
        finally:
            close_windows_handle(handle)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3.0)

    def test_supervisor_rejects_helper_message_and_spectrum_floods(self) -> None:
        for name, lines in {
            "too_many_messages": [
                json.dumps({"type": "status", "index": index}) for index in range(9)
            ]
            + [json.dumps({"type": "spectrum", "counts": [1, 2, 3]})],
            "duplicate_spectrum": [
                json.dumps({"type": "spectrum", "counts": [1, 2, 3]}),
                json.dumps({"type": "spectrum", "counts": [4, 5, 6]}),
            ],
            "non_object_json": [json.dumps(["spectrum", 1, 2, 3])],
        }.items():
            with self.subTest(name=name):
                bridge = _BridgeStub()
                reader = BaySpecSdkLiveReader(bridge, Path("."))
                reader.desired_active = True
                reader.generation = 3
                reader._stop_event.clear()
                process = _CompletedSdkProcess("\n".join(lines) + "\n")
                with patch("sdk_live.subprocess.Popen", return_value=process), patch.object(
                    reader._stop_event,
                    "wait",
                    return_value=True,
                ):
                    reader._supervisor_loop(generation=3)

                self.assertEqual(bridge.payloads, [])
                self.assertEqual(reader.invalid_helper_message_count, 1)
                self.assertEqual(reader.restart_count, 1)

    def test_spectrum_rejects_nonfinite_and_oversized_counts(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.generation = 4
        reader._stop_event.clear()

        nonfinite = reader._handle_message(
            {"type": "spectrum", "counts": [1.0, float("nan"), 3.0]},
            generation=4,
        )
        reader.MAX_SPECTRUM_POINTS = 2
        oversized = reader._handle_message(
            {"type": "spectrum", "counts": [1.0, 2.0, 3.0]},
            generation=4,
        )

        self.assertFalse(nonfinite)
        self.assertFalse(oversized)
        self.assertEqual(reader.bridge.payloads, [])
        self.assertIn("point limit", reader.last_error or "")

    def test_explicit_helper_error_is_preserved_with_exit_code(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.generation = 5
        reader._stop_event.clear()
        process = _CompletedSdkProcess(
            json.dumps({"type": "error", "message": "no USB20BS device found"})
            + "\n",
            returncode=2,
        )
        with patch("sdk_live.subprocess.Popen", return_value=process), patch.object(
            reader._stop_event,
            "wait",
            return_value=True,
        ):
            reader._supervisor_loop(generation=5)

        self.assertEqual(reader.bridge.payloads, [])
        self.assertIn("no USB20BS device found", reader.last_error or "")
        self.assertIn("exited with code 2", reader.last_error or "")
        self.assertEqual(reader.consecutive_failure_count, 1)

    def test_successful_cycle_interval_includes_acquisition_time(self) -> None:
        reader = self._reader()
        reader.desired_active = True
        reader.generation = 5
        reader.interval_ms = 100
        reader._stop_event.clear()
        process = _CompletedSdkProcess("")
        output = (
            json.dumps({"type": "spectrum", "counts": [1.0, 2.0, 3.0]}) + "\n",
            "",
            {
                "timed_out": False,
                "overflow_stream": None,
                "stdout_chars": 64,
                "stderr_chars": 0,
                "read_errors": [],
                "cleanup_error": None,
            },
        )
        observed_delays: list[float] = []

        def delayed_collect(*_args, **_kwargs):
            time.sleep(0.08)
            return output

        def capture_delay(delay: float) -> bool:
            observed_delays.append(delay)
            return True

        with patch("sdk_live.subprocess.Popen", return_value=process), patch.object(
            reader,
            "_collect_helper_output",
            side_effect=delayed_collect,
        ), patch.object(
            reader,
            "_wavelength_grid",
            return_value=[1546.8, 1546.9, 1547.0],
        ), patch.object(reader._stop_event, "wait", side_effect=capture_delay):
            reader._supervisor_loop(generation=5)

        self.assertEqual(reader.frame_count, 1)
        self.assertEqual(len(observed_delays), 1)
        # Windows may overshoot the mocked 80 ms acquisition sleep by a few
        # milliseconds. Keep enough tolerance to verify that acquisition time
        # is subtracted without making the assertion scheduler-sensitive.
        self.assertGreaterEqual(observed_delays[0], 0.015)
        self.assertLess(observed_delays[0], 0.06)
        self.assertGreater(reader.last_acquisition_duration_ms or 0.0, 70.0)


class ModelDisplaySourceGateTests(unittest.TestCase):
    def test_replay_is_allowed_when_live_sources_are_stopped(self) -> None:
        gate = _model_display_source_gate(
            {"source": "static_http_ingest"},
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertFalse(gate["source_fresh"])
        self.assertEqual(gate["model_input_source_mode"], "held_replay_or_http")

    def test_stale_sdk_frame_is_blocked(self) -> None:
        gate = _model_display_source_gate(
            {"source": "bayspec_direct_usb20bs_sdk"},
            {"active": False, "freshness": "stopped"},
            {"active": True, "freshness": "stale"},
        )
        self.assertFalse(gate["model_input_source_allowed"])
        self.assertEqual(gate["selected_live_source"], "sdk")

    def test_fresh_sdk_frame_is_allowed(self) -> None:
        gate = _model_display_source_gate(
            {"source": "bayspec_direct_usb20bs_sdk"},
            {"active": False, "freshness": "stopped"},
            {"active": True, "freshness": "live"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertTrue(gate["source_fresh"])

    def test_mismatched_buffered_source_is_blocked_during_live_session(self) -> None:
        gate = _model_display_source_gate(
            {"source": "static_http_ingest"},
            {"active": True, "freshness": "live"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertFalse(gate["model_input_source_allowed"])
        self.assertEqual(gate["selected_live_source"], "unmatched_live_source")

    def test_pixel_index_axis_is_diagnostics_only(self) -> None:
        gate = _model_display_source_gate(
            {
                "source": "static_http_ingest",
                "peak_axis_type": "pixel_index",
                "qa_status": "ok",
            },
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertFalse(gate["wavelength_axis_valid"])
        self.assertFalse(gate["formal_spectrum_input_allowed"])
        self.assertFalse(gate["operator_display_valid"])

    def test_wavelength_grid_fallback_flag_blocks_formal_recognition(self) -> None:
        gate = _model_display_source_gate(
            {
                "source": "static_http_ingest",
                "peak_axis_type": "wavelength_nm",
                "qa_status": "warning",
                "qa_flags": ["using_pixel_index_fallback"],
            },
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertEqual(
            gate["wavelength_axis_blockers"],
            ["using_pixel_index_fallback"],
        )
        self.assertFalse(gate["formal_spectrum_input_allowed"])

    def test_physical_wavelength_axis_remains_eligible(self) -> None:
        gate = _model_display_source_gate(
            {
                "source": "static_http_ingest",
                "peak_axis_type": "wavelength_nm",
                "qa_status": "ok",
                "qa_flags": [],
            },
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertTrue(gate["wavelength_axis_valid"])
        self.assertTrue(gate["formal_spectrum_input_allowed"])
        self.assertTrue(gate["operator_display_valid"])

    def test_invalid_spectrum_qa_is_diagnostics_only(self) -> None:
        gate = _model_display_source_gate(
            {
                "source": "static_http_ingest",
                "peak_axis_type": "wavelength_nm",
                "qa_status": "invalid",
                "qa_flags": ["spectrum_length_mismatch"],
            },
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertTrue(gate["wavelength_axis_valid"])
        self.assertFalse(gate["qa_valid"])
        self.assertFalse(gate["formal_spectrum_input_allowed"])
        self.assertFalse(gate["operator_display_valid"])


class ExportWatcherSessionTests(unittest.TestCase):
    @staticmethod
    def _dat_record(record_index: int, record_words: int = 513) -> np.ndarray:
        spectrum = (
            np.arange(512, dtype=np.uint32) + 1000 + record_index * 7
        ) % 65535
        if record_words == 512:
            return spectrum.astype(">u2")
        return np.concatenate(
            [spectrum, np.asarray([record_index], dtype=np.uint32)]
        ).astype(">u2")

    def test_start_clears_previous_session_freshness(self) -> None:
        watcher = SenseExportWatcher()
        watcher.last_signature = ("old.csv", 1, 2)
        watcher.last_ingest_time = 123.0
        watcher.last_attempt_time = 123.0
        watcher.last_file = "old.csv"
        watcher.last_file_mtime = 123.0
        watcher.last_result = {"ok": True}
        watcher.ingest_count = 99

        with patch.object(backend_main.bridge, "latest_export_file", return_value=None):
            status = watcher.start("P22", None, 0.35)
            watcher.stop()

        self.assertEqual(status["freshness"], "waiting_for_export")
        self.assertIsNone(status["last_ingest_time"])
        self.assertIsNone(status["last_file"])
        self.assertEqual(status["ingest_count"], 0)
        self.assertEqual(watcher.last_signature, ("old.csv", 1, 2))

    def test_configuration_change_starts_clean_session(self) -> None:
        watcher = SenseExportWatcher()
        watcher.thread = _AliveThread()
        watcher.active = True
        watcher.channel_id = "P22"
        watcher.export_root = "old-root"
        watcher.last_signature = ("old.csv", 1, 2)
        watcher.last_ingest_time = time.time()

        status = watcher.start("P23", "new-root", 0.35)

        self.assertEqual(status["channel_id"], "P23")
        self.assertEqual(status["freshness"], "waiting_for_export")
        self.assertIsNone(watcher.last_signature)
        self.assertGreater(status["acquisition_session_id"], 0)

    def test_old_export_file_cannot_be_reported_live(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        watcher.last_ingest_time = time.time()
        watcher.last_file_mtime = time.time() - 120.0

        status = watcher.status()

        self.assertEqual(status["freshness"], "stale")
        self.assertGreater(status["seconds_since_last_file_update"], 100.0)

    def test_stop_waits_for_inflight_ingest_barrier(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        watcher.ingest_in_progress = True
        watcher.ingest_lock.acquire()
        result_holder = {}

        worker = threading.Thread(
            target=lambda: result_holder.update(watcher.stop()),
            daemon=True,
        )
        worker.start()
        time.sleep(0.03)
        self.assertTrue(worker.is_alive())

        watcher.ingest_in_progress = False
        watcher.ingest_lock.release()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertTrue(result_holder["ok"])
        self.assertFalse(result_holder["active"])

    def test_thread_launch_failure_rolls_back_active_state(self) -> None:
        watcher = SenseExportWatcher()
        with patch(
            "backend.main.threading.Thread",
            return_value=_ThreadStartFailure(),
        ):
            result = watcher.start("P22", None, 0.35)

        self.assertFalse(result["ok"])
        self.assertEqual(result["operation_status"], "start_failed")
        self.assertFalse(result["active"])
        self.assertFalse(result["worker_alive"])
        self.assertFalse(result["stop_requested"])

    def test_stop_interrupts_long_poll_wait_and_joins_worker(self) -> None:
        watcher = SenseExportWatcher()
        with patch.object(backend_main.bridge, "latest_export_file", return_value=None):
            started = watcher.start("P22", None, 5.0)
            self.assertTrue(started["ok"])
            deadline = time.monotonic() + 1.0
            while not watcher.status()["worker_alive"] and time.monotonic() < deadline:
                time.sleep(0.005)

            before_stop = time.monotonic()
            stopped = watcher.stop()
            elapsed = time.monotonic() - before_stop

        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["operation_status"], "stopped")
        self.assertFalse(stopped["active"])
        self.assertFalse(stopped["worker_alive"])
        self.assertFalse(stopped["stop_requested"])
        self.assertLess(elapsed, 0.75)

    def test_restart_is_blocked_when_previous_worker_did_not_stop(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        watcher.thread = _NeverStoppingThread()

        stopped = watcher.stop()
        restarted = watcher.start("P22", None, 0.35)

        self.assertFalse(stopped["ok"])
        self.assertEqual(stopped["operation_status"], "stop_timeout")
        self.assertTrue(stopped["worker_alive"])
        self.assertFalse(restarted["ok"])
        self.assertEqual(
            restarted["operation_status"],
            "previous_worker_stop_timeout",
        )

    def test_dat_progress_ignores_partial_growth_until_record_is_complete(self) -> None:
        watcher = SenseExportWatcher()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Record_test.dat"
            first = self._dat_record(0)
            second = self._dat_record(1)
            third = self._dat_record(2)
            path.write_bytes(first.tobytes() + second.tobytes())
            watcher._remember_dat_progress(
                path,
                {
                    "dat_frame_count": 2,
                    "dat_record_words": 513,
                    "dat_header_bytes": 0,
                },
            )

            with path.open("ab") as handle:
                handle.write(third[:173].tobytes())
            self.assertFalse(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

            with path.open("ab") as handle:
                handle.write(third[173:].tobytes())
            self.assertTrue(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

    def test_initial_dat_waits_for_enough_layout_evidence(self) -> None:
        watcher = SenseExportWatcher()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Record_test.dat"
            first = self._dat_record(0)
            second = self._dat_record(1)
            third = self._dat_record(2)
            path.write_bytes(first.tobytes() + second.tobytes())

            self.assertLess(path.stat().st_size, watcher.MIN_INITIAL_DAT_BYTES)
            self.assertFalse(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

            with path.open("ab") as handle:
                handle.write(third.tobytes())
            self.assertTrue(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

    def test_truncated_reused_dat_path_reenters_layout_bootstrap(self) -> None:
        watcher = SenseExportWatcher()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Record_reused.dat"
            records = [self._dat_record(index) for index in range(5)]
            path.write_bytes(b"".join(record.tobytes() for record in records))
            watcher._remember_dat_progress(
                path,
                {
                    "dat_frame_count": 5,
                    "dat_record_words": 513,
                    "dat_header_bytes": 0,
                },
            )

            path.write_bytes(records[0].tobytes() + records[1].tobytes())
            self.assertFalse(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

            with path.open("ab") as handle:
                handle.write(records[2].tobytes())
            self.assertTrue(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

    def test_incomplete_initial_dat_does_not_consume_file_signature(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Record_partial.dat"
            path.write_bytes(
                self._dat_record(0).tobytes() + self._dat_record(1).tobytes()
            )
            with patch.object(
                backend_main.bridge,
                "latest_export_file",
                return_value=path,
            ), patch.object(
                backend_main.bridge,
                "ingest_export_file",
            ) as ingest_export, patch.object(
                watcher.stop_event,
                "wait",
                return_value=True,
            ):
                watcher._loop()

            ingest_export.assert_not_called()
            self.assertIsNone(watcher.last_signature)
            self.assertEqual(watcher.incomplete_dat_skip_count, 1)

    def test_dat_progress_detects_rewritten_last_complete_record(self) -> None:
        watcher = SenseExportWatcher()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Record_test.dat"
            first = self._dat_record(0)
            second = self._dat_record(1)
            path.write_bytes(first.tobytes() + second.tobytes())
            watcher._remember_dat_progress(
                path,
                {
                    "dat_frame_count": 2,
                    "dat_record_words": 513,
                    "dat_header_bytes": 0,
                },
            )

            rewritten = self._dat_record(9)
            with path.open("r+b") as handle:
                handle.seek(first.nbytes)
                handle.write(rewritten.tobytes())

            self.assertTrue(
                watcher._dat_export_has_new_complete_frame(path, path.stat().st_size)
            )

    def test_text_export_must_remain_unchanged_during_stability_window(self) -> None:
        watcher = SenseExportWatcher()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Spectrum_test.csv"
            path.write_text("wavelength,intensity\n1546.8,100\n", encoding="utf-8")
            signature = watcher._file_signature(path)[:3]

            def mutate_during_wait(_timeout: float) -> bool:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write("1546.9,101\n")
                return False

            with patch.object(
                watcher.stop_event,
                "wait",
                side_effect=mutate_during_wait,
            ):
                confirmed = watcher._confirm_stable_text_export(path, signature)

            self.assertIsNone(confirmed)

    def test_text_export_removed_during_stability_window_is_not_ingested(self) -> None:
        watcher = SenseExportWatcher()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Spectrum_test.csv"
            path.write_text("wavelength,intensity\n1546.8,100\n", encoding="utf-8")
            signature = watcher._file_signature(path)[:3]

            def remove_during_wait(_timeout: float) -> bool:
                path.unlink()
                return False

            with patch.object(
                watcher.stop_event,
                "wait",
                side_effect=remove_during_wait,
            ):
                confirmed = watcher._confirm_stable_text_export(path, signature)

            self.assertIsNone(confirmed)

    def test_successful_inflight_ingest_is_deduplicated_after_stop(self) -> None:
        watcher = SenseExportWatcher()
        ingest_started = threading.Event()
        release_ingest = threading.Event()
        stop_result: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Spectrum_test.csv"
            path.write_text("wavelength,intensity\n1546.8,100\n", encoding="utf-8")

            def blocking_ingest(*_args, **_kwargs) -> dict[str, object]:
                ingest_started.set()
                self.assertTrue(release_ingest.wait(timeout=1.0))
                return {"ok": True, "records_ingested": 1}

            with patch.object(
                backend_main.bridge,
                "latest_export_file",
                return_value=path,
            ), patch.object(
                backend_main.bridge,
                "ingest_export_file",
                side_effect=blocking_ingest,
            ):
                started = watcher.start("P22", temp_dir, 0.1)
                self.assertTrue(started["ok"])
                self.assertTrue(ingest_started.wait(timeout=1.0))
                stopper = threading.Thread(
                    target=lambda: stop_result.update(watcher.stop()),
                    daemon=True,
                )
                stopper.start()
                time.sleep(0.03)
                self.assertTrue(stopper.is_alive())
                release_ingest.set()
                stopper.join(timeout=1.0)

            self.assertFalse(stopper.is_alive())
            self.assertTrue(stop_result["ok"])
            self.assertEqual(watcher.last_signature, watcher._file_signature(path)[:3])
            self.assertEqual(watcher.ingest_count, 0)

    def test_poll_exception_is_counted_as_failed_ingest_attempt(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        with patch.object(
            backend_main.bridge,
            "latest_export_file",
            side_effect=OSError("export directory unavailable"),
        ), patch.object(watcher.stop_event, "wait", return_value=True):
            watcher._loop()

        self.assertEqual(watcher.failed_ingest_count, 1)
        self.assertIsNotNone(watcher.last_attempt_time)
        self.assertIn("export directory unavailable", watcher.last_error or "")


class FrontendRequestLifecycleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (Path(backend_main.FRONTEND_ROOT) / "app.js").read_text(
            encoding="utf-8"
        )

    def test_capture_status_poll_has_an_independent_inflight_guard(self) -> None:
        self.assertIn("px6dCapturePollInFlight: false", self.source)
        self.assertRegex(
            self.source,
            r"!state\.pageVisible\s*\|\|\s*"
            r"state\.px6dCaptureRequestInFlight\s*\|\|\s*"
            r"state\.px6dCapturePollInFlight",
        )
        self.assertIn("state.px6dCapturePollInFlight = false", self.source)

    def test_live_frame_delivery_uses_low_latency_nonoverlapping_polling(self) -> None:
        self.assertIn("const LIVE_MODEL_POLL_INTERVAL_MS = 25;", self.source)
        self.assertIn(
            'const sourceIntervalSec = source === "direct_sdk" ? 0.01 : 0.1;',
            self.source,
        )
        self.assertIn("frameRequestInFlight: false", self.source)
        self.assertIn("if (state.frameRequestInFlight) {", self.source)
        self.assertIn("if (force) state.forcedFrameRequestQueued = true;", self.source)

    def test_stopping_live_input_clears_the_committed_visual_frame(self) -> None:
        self.assertIn("function clearStoppedLivePresentation()", self.source)
        self.assertIn("state.frame = idleFrame;", self.source)
        self.assertIn("snapDisplayedFrameToCurrentTargets();", self.source)
        self.assertIn("if (!stopping) await fetchFrame({ force: true });", self.source)

    def test_contact_animation_has_fast_attack_and_stable_release(self) -> None:
        self.assertIn("const THREE_ATTENUATION_EASING = 28.0;", self.source)
        self.assertIn("const THREE_DEFORMATION_EASING = 26.0;", self.source)
        self.assertIn("const THREE_SPATIAL_EASING = 24.0;", self.source)
        self.assertIn("const THREE_ATTENUATION_RELEASE_EASING = 16.0;", self.source)
        self.assertIn("const THREE_DEFORMATION_RELEASE_EASING = 14.0;", self.source)

    def test_capture_commands_invalidate_stale_status_poll(self) -> None:
        self.assertIn("px6dCapturePollController: null", self.source)
        self.assertIn("px6dCaptureStatusEpoch: 0", self.source)
        self.assertIn("function invalidatePx6dCaptureStatusPoll()", self.source)
        self.assertIn(
            "state.px6dCapturePollController.abort()",
            self.source,
        )
        self.assertIn(
            "if (requestEpoch !== state.px6dCaptureStatusEpoch) return;",
            self.source,
        )
        self.assertGreaterEqual(
            self.source.count("invalidatePx6dCaptureStatusPoll();"),
            2,
        )

    def test_frame_context_invalidation_aborts_obsolete_request(self) -> None:
        self.assertIn("frameRequestController: null", self.source)
        self.assertIn(
            "state.frameRequestController.abort()",
            self.source,
        )
        self.assertGreaterEqual(
            self.source.count("signal: requestController.signal"),
            2,
        )

    def test_request_abort_uses_standard_abort_error_semantics(self) -> None:
        self.assertIn("const relayAbort = () => controller.abort();", self.source)
        self.assertNotIn('controller.abort("request_timeout")', self.source)
        self.assertNotIn('controller.abort("request_cancelled")', self.source)

    def test_all_frontend_http_calls_use_the_timeout_wrapper(self) -> None:
        # requestJSON is the only place allowed to call fetch directly. This
        # prevents setup, source-switch, and diagnostics commands from hanging
        # forever when a device endpoint stops responding.
        self.assertEqual(self.source.count("await fetch("), 1)
        self.assertIn(
            'await requestJSON("/api/reset?keep_baseline=true", { method: "POST" })',
            self.source,
        )
        self.assertIn('console.error("[input source change]", error);', self.source)

    def test_hidden_page_suspends_polling_and_aborts_obsolete_requests(self) -> None:
        self.assertIn('pageVisible: document.visibilityState !== "hidden"', self.source)
        self.assertIn("function handlePageVisibilityChange()", self.source)
        self.assertIn(
            'document.addEventListener("visibilitychange", handlePageVisibilityChange)',
            self.source,
        )
        self.assertGreaterEqual(self.source.count("if (!state.pageVisible"), 3)
        self.assertIn("invalidatePx6dReferenceRequest();", self.source)
        self.assertIn("invalidatePx6dCaptureStatusPoll();", self.source)
        self.assertIn("fetchFrame({ force: true });", self.source)

    def test_client_schedulers_and_boot_are_idempotent(self) -> None:
        self.assertIn("function startClientSchedulers()", self.source)
        self.assertIn("if (state.clientSchedulersStarted) return;", self.source)
        self.assertIn("if (state.bootStarted) return;", self.source)
        self.assertIn("loadRuntimeCapabilities()", self.source)
        self.assertIn('"/api/health"', self.source)
        self.assertIn('boot().catch((error) => {', self.source)

    def test_frontend_recovers_an_already_running_live_source(self) -> None:
        self.assertIn("const LIVE_SOURCE_PROBE_INTERVAL_MS = 1500;", self.source)
        self.assertIn("async function reconcileLiveSourceState", self.source)
        self.assertIn(
            '"/api/frame?trace_limit=1&include_spectrum=false"',
            self.source,
        )
        self.assertIn(
            "const liveSourceActive = await reconcileLiveSourceState({ force: true });",
            self.source,
        )
        self.assertIn("if (!sourceActive) return;", self.source)
        self.assertIn("invalidateLiveSourceProbe();", self.source)

    def test_backend_imports_from_app_directory_without_external_pythonpath(self) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["TOUCH_PX6D_AUTO_START"] = "false"
        completed = subprocess.run(
            [
                sys.executable,
                "-E",
                "-c",
                (
                    "from backend.main import health; "
                    "payload = health(); "
                    "assert payload['ok'] is True; "
                    "assert payload['backend_contract_version'] == "
                    "'touch_current_runtime_api_v1'"
                ),
            ],
            cwd=str(backend_main.APP_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_fastapi_runtime_uses_lifespan_cleanup(self) -> None:
        backend_source = Path(backend_main.__file__).read_text(encoding="utf-8")
        self.assertIn("lifespan=application_lifespan", backend_source)
        self.assertNotIn('@app.on_event("startup")', backend_source)
        self.assertNotIn('@app.on_event("shutdown")', backend_source)

    def test_fastapi_lifespan_runs_startup_and_shutdown_once(self) -> None:
        events: list[str] = []

        async def exercise_lifespan() -> None:
            async with backend_main.application_lifespan(backend_main.app):
                events.append("inside")

        with patch.object(
            backend_main,
            "startup_reference_sources",
            side_effect=lambda: events.append("startup"),
        ) as startup, patch.object(
            backend_main,
            "shutdown_live_sources",
            side_effect=lambda: events.append("shutdown"),
        ) as shutdown:
            asyncio.run(exercise_lifespan())

        self.assertEqual(events, ["startup", "inside", "shutdown"])
        startup.assert_called_once_with()
        shutdown.assert_called_once_with()


class AcquisitionSourceMutualExclusionTests(unittest.TestCase):
    def test_source_switch_requires_a_fully_quiescent_previous_source(self) -> None:
        for field in (
            "active",
            "requested_active",
            "worker_alive",
            "process_running",
            "ingest_in_progress",
            "start_in_progress",
            "start_cancel_requested",
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    backend_main._live_source_stop_completed(
                        {"ok": True, field: True, "stop_requested": False}
                    )
                )
        self.assertFalse(
            backend_main._live_source_stop_completed({"ok": False})
        )
        self.assertTrue(
            backend_main._live_source_stop_completed(
                {
                    "ok": True,
                    "active": False,
                    "requested_active": False,
                    "worker_alive": False,
                    "process_running": False,
                    "ingest_in_progress": False,
                }
            )
        )

    def test_manual_import_is_blocked_while_sdk_owns_bridge(self) -> None:
        stopped = {
            "active": False,
            "requested_active": False,
            "worker_alive": False,
            "process_running": False,
            "ingest_in_progress": False,
            "start_in_progress": False,
            "start_cancel_requested": False,
        }
        sdk_active = {**stopped, "active": True, "requested_active": True}
        with patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=sdk_active,
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.bridge,
            "ingest_latest_export",
        ) as ingest_export:
            result = backend_main.ingest_latest_export(
                channel_id="P22",
                export_root=None,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "live_source_active")
        self.assertEqual(result["busy_sources"], ["sdk_live"])
        ingest_export.assert_not_called()

    def test_manual_import_runs_when_live_sources_are_quiescent(self) -> None:
        stopped = {
            "active": False,
            "requested_active": False,
            "worker_alive": False,
            "process_running": False,
            "ingest_in_progress": False,
            "start_in_progress": False,
            "start_cancel_requested": False,
        }
        with patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.bridge,
            "ingest_latest_export",
            return_value={"ok": True, "records_ingested": 1},
        ) as ingest_export:
            result = backend_main.ingest_latest_export(
                channel_id="P22",
                export_root=None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "bayspec_sense_export_ingest_once")
        ingest_export.assert_called_once_with(root=None, channel_id="P22")

    def test_concurrent_source_switches_share_one_lifecycle_guard(self) -> None:
        first_reset_entered = threading.Event()
        release_first_reset = threading.Event()
        export_request_reached_sdk_stop = threading.Event()
        reset_count = 0
        reset_count_lock = threading.Lock()
        stopped = {
            "ok": True,
            "active": False,
            "worker_alive": False,
            "ingest_in_progress": False,
        }

        def reset_session() -> dict:
            nonlocal reset_count
            with reset_count_lock:
                reset_count += 1
                call_number = reset_count
            if call_number == 1:
                first_reset_entered.set()
                release_first_reset.wait(timeout=2.0)
            return {"ok": True, "baseline_invalidated": True}

        def stop_sdk() -> dict:
            export_request_reached_sdk_stop.set()
            return dict(stopped)

        results: dict[str, dict] = {}
        with patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=dict(stopped),
        ), patch.object(
            backend_main.export_watcher,
            "stop",
            return_value=dict(stopped),
        ), patch.object(
            backend_main.sdk_live_reader,
            "start",
            return_value={"ok": True, "active": True},
        ), patch.object(
            backend_main.sdk_live_reader,
            "stop",
            side_effect=stop_sdk,
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=dict(stopped),
        ), patch.object(
            backend_main.export_watcher,
            "start",
            return_value={"ok": True, "active": True},
        ), patch.object(
            backend_main,
            "_begin_acquisition_session",
            side_effect=reset_session,
        ):
            sdk_thread = threading.Thread(
                target=lambda: results.setdefault(
                    "sdk",
                    backend_main.sdk_start(
                        channel_id="P22",
                        interval_ms=100,
                        integration=40000,
                    ),
                )
            )
            export_thread = threading.Thread(
                target=lambda: results.setdefault(
                    "export",
                    backend_main.export_watch_start(
                        channel_id="P22",
                        export_root=None,
                        interval_sec=0.35,
                    ),
                )
            )
            sdk_thread.start()
            self.assertTrue(first_reset_entered.wait(timeout=2.0))
            export_thread.start()
            self.assertFalse(export_request_reached_sdk_stop.wait(timeout=0.15))
            release_first_reset.set()
            sdk_thread.join(timeout=3.0)
            export_thread.join(timeout=3.0)

        self.assertFalse(sdk_thread.is_alive())
        self.assertFalse(export_thread.is_alive())
        self.assertTrue(export_request_reached_sdk_stop.is_set())
        self.assertEqual(reset_count, 2)
        self.assertTrue(results["sdk"]["ok"])
        self.assertTrue(results["export"]["ok"])

    def test_export_watch_start_stops_sdk_first(self) -> None:
        watcher_status = {"active": True, "freshness": "waiting_for_export"}
        sdk_status = {"active": False, "freshness": "stopped"}
        with patch.object(backend_main.sdk_live_reader, "stop", return_value=sdk_status) as stop_sdk, patch.object(
            backend_main.export_watcher, "start", return_value=watcher_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.export_watch_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
            )

        stop_sdk.assert_called_once_with()
        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])
        self.assertEqual(result["sdk_live"], sdk_status)

    def test_live_export_source_stops_sdk_first(self) -> None:
        watcher_status = {"active": True, "freshness": "waiting_for_export"}
        sdk_status = {"active": False, "freshness": "stopped"}
        with patch.object(backend_main.sdk_live_reader, "stop", return_value=sdk_status) as stop_sdk, patch.object(
            backend_main.export_watcher, "start", return_value=watcher_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
                control_sense=False,
                source="export_watch",
            )

        stop_sdk.assert_called_once_with()
        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertEqual(result["sdk_live"], sdk_status)

    def test_sdk_start_invalidates_previous_session_baseline(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        stopped_watch = {"ok": True, "active": False, "ingest_in_progress": False}
        with patch.object(
            backend_main.export_watcher,
            "stop",
            return_value=stopped_watch,
        ), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.sdk_start(channel_id="P22", interval_ms=100, integration=40000)

        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])

    def test_duplicate_sdk_start_preserves_current_baseline(self) -> None:
        sdk_status = {
            "active": True,
            "freshness": "live",
            "channel_id": "P22",
            "interval_ms": 100,
            "integration": 40000,
            "sensor_mode": backend_main.DEFAULT_SENSOR_MODE,
        }
        with patch.object(
            backend_main.sdk_live_reader, "status", return_value=sdk_status
        ), patch.object(
            backend_main.sdk_live_reader, "start"
        ) as start_sdk, patch.object(
            backend_main, "_begin_acquisition_session"
        ) as reset_session:
            result = backend_main.sdk_start(
                channel_id="P22", interval_ms=100, integration=40000
            )

        start_sdk.assert_not_called()
        reset_session.assert_not_called()
        self.assertFalse(
            result["acquisition_session_reset"]["baseline_invalidated"]
        )
        self.assertEqual(result["mode"], "bayspec_direct_sdk_already_running")

    def test_duplicate_live_start_uses_same_clamped_sdk_interval(self) -> None:
        sdk_status = {
            "active": True,
            "freshness": "live",
            "channel_id": "P22",
            "interval_ms": 2000,
            "integration": backend_main.DEFAULT_INTEGRATION_US,
            "sensor_mode": backend_main.DEFAULT_SENSOR_MODE,
        }
        with patch.object(
            backend_main.sdk_live_reader, "status", return_value=sdk_status
        ), patch.object(
            backend_main.sdk_live_reader, "stop"
        ) as stop_sdk, patch.object(
            backend_main.sdk_live_reader, "start"
        ) as start_sdk, patch.object(
            backend_main, "_begin_acquisition_session"
        ) as reset_session:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=5.0,
                control_sense=False,
                source="direct_sdk",
            )

        stop_sdk.assert_not_called()
        start_sdk.assert_not_called()
        reset_session.assert_not_called()
        self.assertFalse(
            result["acquisition_session_reset"]["baseline_invalidated"]
        )
        self.assertEqual(result["mode"], "bayspec_live_twin_already_running")

    def test_direct_live_start_invalidates_previous_session_baseline(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        stopped_watch = {"ok": True, "active": False, "ingest_in_progress": False}
        with patch.object(
            backend_main.export_watcher,
            "stop",
            return_value=stopped_watch,
        ), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.1,
                control_sense=False,
                source="direct_sdk",
            )

        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])

    def test_direct_live_start_uses_20_ms_low_latency_interval(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        stopped_watch = {"ok": True, "active": False, "ingest_in_progress": False}
        with patch.object(
            backend_main.export_watcher,
            "stop",
            return_value=stopped_watch,
        ), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ) as start_sdk, patch.object(
            backend_main.bridge, "reset", return_value={"ok": True}
        ):
            backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.02,
                control_sense=False,
                source="direct_sdk",
            )

        start_sdk.assert_called_once_with(
            channel_id="P22",
            interval_ms=20,
            integration=backend_main.DEFAULT_INTEGRATION_US,
            sensor_mode=backend_main.DEFAULT_SENSOR_MODE,
        )

    def test_export_watch_start_is_blocked_when_sdk_does_not_stop(self) -> None:
        sdk_status = {
            "ok": False,
            "operation_status": "stop_timeout",
            "worker_alive": True,
            "stop_requested": True,
        }
        with patch.object(
            backend_main.sdk_live_reader,
            "stop",
            return_value=sdk_status,
        ), patch.object(
            backend_main.export_watcher,
            "start",
        ) as start_watcher, patch.object(
            backend_main,
            "_begin_acquisition_session",
        ) as reset_session:
            result = backend_main.export_watch_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["mode"], "live_source_switch_blocked")
        self.assertFalse(result["acquisition_session_reset"]["baseline_invalidated"])
        start_watcher.assert_not_called()
        reset_session.assert_not_called()

    def test_sdk_reconfigure_is_blocked_when_old_worker_does_not_stop(self) -> None:
        existing_status = {
            "active": True,
            "worker_alive": True,
            "channel_id": "P22",
            "interval_ms": 100,
            "integration": 40000,
        }
        stopped_status = {
            "ok": False,
            "operation_status": "stop_timeout",
            "worker_alive": True,
            "stop_requested": True,
        }
        with patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=existing_status,
        ), patch.object(
            backend_main.sdk_live_reader,
            "stop",
            return_value=stopped_status,
        ), patch.object(
            backend_main.sdk_live_reader,
            "start",
        ) as start_sdk, patch.object(
            backend_main,
            "_begin_acquisition_session",
        ) as reset_session:
            result = backend_main.sdk_start(
                channel_id="P23",
                interval_ms=100,
                integration=40000,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "previous_live_source_not_stopped")
        start_sdk.assert_not_called()
        reset_session.assert_not_called()

    def test_sdk_stop_endpoint_propagates_timeout_failure(self) -> None:
        stopped_status = {
            "ok": False,
            "operation_status": "stop_timeout",
            "worker_alive": True,
            "stop_requested": True,
        }
        with patch.object(
            backend_main.sdk_live_reader,
            "stop",
            return_value=stopped_status,
        ):
            result = backend_main.sdk_stop()

        self.assertFalse(result["ok"])
        self.assertEqual(result["sdk_live"], stopped_status)

    def test_duplicate_export_watch_start_preserves_current_baseline(self) -> None:
        sdk_status = {"ok": True, "active": False, "freshness": "stopped"}
        watcher_status = {
            "active": True,
            "worker_alive": True,
            "ingest_in_progress": False,
            "channel_id": "P22",
            "export_root": None,
            "interval_sec": 0.35,
        }
        with patch.object(
            backend_main.sdk_live_reader,
            "stop",
            return_value=sdk_status,
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=watcher_status,
        ), patch.object(
            backend_main.export_watcher,
            "start",
        ) as start_watcher, patch.object(
            backend_main,
            "_begin_acquisition_session",
        ) as reset_session:
            result = backend_main.export_watch_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "sense_export_watch_already_running")
        self.assertFalse(result["acquisition_session_reset"]["baseline_invalidated"])
        start_watcher.assert_not_called()
        reset_session.assert_not_called()

    def test_export_watch_stop_endpoint_propagates_quiescence_failure(self) -> None:
        stopped_status = {
            "ok": False,
            "operation_status": "stop_timeout",
            "active": False,
            "ingest_in_progress": True,
        }
        with patch.object(
            backend_main.export_watcher,
            "stop",
            return_value=stopped_status,
        ):
            result = backend_main.export_watch_stop()

        self.assertFalse(result["ok"])
        self.assertEqual(result["export_watcher"], stopped_status)


class FastDatIngestRegressionTests(unittest.TestCase):
    def test_fast_dat_uses_decoded_spectrum_length_for_wavelength_grid(self) -> None:
        class LayoutStub:
            frame_count = 2
            prefix_words = 16
            record_words = 516
            spectrum_words = 512
            trailing_words = 0
            median_adjacent_correlation = 0.99
            score_margin = 0.5
            name = "test_layout"

        class SequenceStub:
            layout = LayoutStub()
            spectra = np.vstack(
                [
                    np.linspace(100.0, 200.0, 512),
                    np.linspace(110.0, 220.0, 512),
                ]
            )

        test_bridge = BaySpecWavelengthShiftBridge()
        wavelength_grid = np.linspace(1526.0, 1562.0, 512).tolist()
        with patch(
            "bridge.read_sense_fast_dat",
            return_value=SequenceStub(),
        ), patch.object(
            test_bridge,
            "_latest_wavelength_grid",
            return_value=wavelength_grid,
        ) as find_grid, patch.object(
            test_bridge,
            "ingest",
            return_value={"ok": True, "records_ingested": 1},
        ) as ingest:
            result = test_bridge.ingest_fast_record_dat("synthetic.dat")

        find_grid.assert_called_once_with(Path("."), expected_points=512)
        payload = ingest.call_args.args[0]
        channel = payload["channels"][0]
        self.assertEqual(len(channel["intensity"]), 512)
        self.assertEqual(len(channel["wavelength_nm"]), 512)
        self.assertEqual(result["dat_frame_count"], 2)
        self.assertEqual(result["dat_parser"], "test_layout")


class BridgeHistoryRetentionTests(unittest.TestCase):
    @staticmethod
    def _spectrum_payload(frame_index: int) -> dict[str, object]:
        wavelength = np.linspace(1546.1, 1547.65, 64)
        center = 1546.89 + 0.0001 * (frame_index % 7)
        intensity = 1200.0 + 5000.0 * np.exp(
            -0.5 * ((wavelength - center) / 0.11) ** 2
        )
        return {
            "timestamp": frame_index * 0.1,
            "channels": [
                {
                    "channel_id": "P22",
                    "wavelength_nm": wavelength.tolist(),
                    "intensity": intensity.tolist(),
                }
            ],
        }

    def test_old_records_release_full_spectrum_but_keep_scalar_history(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(
            max_records_per_channel=8,
            max_spectrum_records_per_channel=3,
        )

        for frame_index in range(12):
            result = test_bridge.ingest(self._spectrum_payload(frame_index))
            self.assertTrue(result["ok"])

        records = list(test_bridge.records_by_channel["P22"])
        self.assertEqual(len(records), 8)
        self.assertEqual(len(test_bridge.all_records), 8)
        self.assertEqual(
            sum(bool(record.get("full_spectrum_retained")) for record in records),
            3,
        )
        for record in records[:-3]:
            self.assertNotIn("wavelength_nm", record)
            self.assertNotIn("intensity", record)
            self.assertNotIn("spectrum_peaks", record)
            self.assertTrue(record["spectrum_payload_evicted"])
            self.assertIn("response_level", record)
            self.assertIn("qa_status", record)
        for record in records[-3:]:
            self.assertIn("wavelength_nm", record)
            self.assertIn("intensity", record)
            self.assertFalse(record["spectrum_payload_evicted"])

        latest = test_bridge.latest("P22", include_spectrum=True)
        self.assertEqual(len(latest["wavelength_nm"]), 64)
        self.assertEqual(len(latest["intensity"]), 64)

    def test_accelerated_long_run_history_stays_structurally_bounded(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(
            max_records_per_channel=120,
            max_spectrum_records_per_channel=8,
        )

        for frame_index in range(600):
            test_bridge.ingest(self._spectrum_payload(frame_index))

        records = list(test_bridge.records_by_channel["P22"])
        self.assertEqual(len(records), 120)
        self.assertEqual(len(test_bridge.all_records), 120)
        self.assertEqual(
            sum("wavelength_nm" in record for record in records),
            8,
        )
        self.assertEqual(
            sum("spectrum_peaks" in record for record in records),
            8,
        )
        with patch.object(test_bridge, "latest_export_file", return_value=None):
            policy = test_bridge.status()["history_policy"]
        self.assertEqual(policy["scalar_records_per_channel"], 120)
        self.assertEqual(policy["mixed_trace_records"], 120)
        self.assertEqual(policy["full_spectrum_records_per_channel"], 8)
        self.assertEqual(
            policy["full_spectrum_records_retained_by_channel"],
            {"P22": 8},
        )


class BridgeStatusIsolationTests(unittest.TestCase):
    def test_slow_status_filesystem_scan_does_not_block_frame_ingest(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=20)
        scan_started = threading.Event()
        release_scan = threading.Event()
        status_done = threading.Event()
        ingest_done = threading.Event()
        errors: list[BaseException] = []

        def slow_latest_export_file(*_args: object, **_kwargs: object) -> None:
            scan_started.set()
            release_scan.wait(timeout=3.0)
            return None

        def read_status() -> None:
            try:
                test_bridge.status()
            except BaseException as exc:  # pragma: no cover - asserted below.
                errors.append(exc)
            finally:
                status_done.set()

        def ingest_frame() -> None:
            try:
                test_bridge.ingest(
                    BridgeHistoryRetentionTests._spectrum_payload(0)
                )
            except BaseException as exc:  # pragma: no cover - asserted below.
                errors.append(exc)
            finally:
                ingest_done.set()

        with patch.object(
            test_bridge,
            "latest_export_file",
            side_effect=slow_latest_export_file,
        ), patch.object(
            test_bridge,
            "_sense_process_status_cached",
            return_value={"running": False, "method": "test"},
        ):
            status_thread = threading.Thread(target=read_status, daemon=True)
            status_thread.start()
            self.assertTrue(scan_started.wait(timeout=1.0))
            ingest_thread = threading.Thread(target=ingest_frame, daemon=True)
            ingest_thread.start()
            try:
                self.assertTrue(
                    ingest_done.wait(timeout=0.75),
                    "frame ingest was blocked by slow status-only filesystem I/O",
                )
            finally:
                release_scan.set()
                ingest_thread.join(timeout=2.0)
                status_thread.join(timeout=2.0)

        self.assertTrue(status_done.is_set())
        self.assertFalse(errors)


class BridgeResetTests(unittest.TestCase):
    @staticmethod
    def _stable_spectrum_records(
        spectrum: np.ndarray,
        *,
        frame_count: int = 30,
    ) -> list[dict[str, object]]:
        wavelength = np.linspace(1526.5, 1561.5, spectrum.size)
        records = []
        for index in range(frame_count):
            deterministic_noise = 0.25 * np.sin(
                np.linspace(0.0, 4.0 * np.pi, spectrum.size) + index * 0.07
            )
            frame = spectrum + deterministic_noise
            records.append(
                {
                    "timestamp": index * 0.1,
                    "intensity_counts": float(np.max(frame)),
                    "centroid_wavelength_nm": 1544.34,
                    "wavelength_nm": wavelength.tolist(),
                    "intensity": frame.tolist(),
                }
            )
        return records

    def test_recent_baseline_uses_minimum_sample_count_beyond_time_window(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=50)
        wavelength = [1540.0 + index * 0.1 for index in range(24)]
        test_bridge.records_by_channel["P22"] = [
            {
                "timestamp": float(index),
                "intensity_counts": 1000.0 + index,
                "centroid_wavelength_nm": 1546.89,
                "wavelength_nm": wavelength,
                "intensity": [1000.0 + index for _ in wavelength],
            }
            for index in range(30)
        ]

        recent = test_bridge._recent_records_for_baseline(
            "P22",
            minimum_samples=30,
        )

        self.assertEqual(len(recent), 30)
        self.assertEqual(recent[0]["timestamp"], 0.0)

    def test_keep_baseline_reset_clears_temporal_tracking(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=20)
        test_bridge.baseline_wavelength_by_channel["P22"] = 1546.89
        test_bridge.previous_tracked_wavelength_by_channel["P22"] = 1547.10

        result = test_bridge.reset(keep_baseline=True)

        self.assertTrue(result["ok"])
        self.assertEqual(test_bridge.baseline_wavelength_by_channel["P22"], 1546.89)
        self.assertNotIn("P22", test_bridge.previous_tracked_wavelength_by_channel)

    def test_stable_local_recovery_residual_is_rejected_against_session_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 9000.0 + 3200.0 * np.exp(-0.5 * (x / 0.24) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)

        first = test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        self.assertTrue(first["current_runtime_spectrum_baseline_ready"])
        self.assertEqual(
            first["baseline_anchor_comparison_by_channel"]["P22"]["status"],
            "trusted_anchor_initialized",
        )
        accepted_before = np.asarray(
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"]
        )

        local_residual = clean.copy()
        local_residual[48:78] -= 2600.0
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(
            local_residual
        )
        rejected = test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        self.assertTrue(rejected["ok"])
        self.assertTrue(rejected["current_runtime_spectrum_baseline_rejected"])
        self.assertFalse(rejected["current_runtime_spectrum_baseline_ready"])
        self.assertEqual(
            rejected["current_runtime_spectrum_baseline_status"],
            "recovery_residual_detected",
        )
        self.assertEqual(
            rejected["baseline_anchor_comparison_by_channel"]["P22"]["status"],
            "recovery_residual_detected",
        )
        np.testing.assert_allclose(
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"],
            accepted_before,
        )

    def test_common_gain_change_does_not_look_like_local_recovery_residual(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 8500.0 + 2800.0 * np.exp(-0.5 * (x / 0.28) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)
        test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(
            clean * 1.06
        )
        result = test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        self.assertTrue(result["current_runtime_spectrum_baseline_ready"])
        comparison = result["baseline_anchor_comparison_by_channel"]["P22"]
        self.assertNotEqual(comparison["status"], "recovery_residual_detected")
        self.assertAlmostEqual(comparison["common_gain_ratio"], 1.06, places=3)

    def test_explicit_operator_baseline_can_replace_a_stale_session_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 9000.0 + 3200.0 * np.exp(-0.5 * (x / 0.24) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)
        test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        new_released_state = clean.copy()
        new_released_state[48:78] -= 2600.0
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(
            new_released_state
        )
        accepted = test_bridge.set_baseline(
            {
                "channel_id": "P22",
                "minimum_recent_samples": 30,
                "replace_trusted_session_anchor": True,
            }
        )

        self.assertTrue(accepted["current_runtime_spectrum_baseline_ready"])
        self.assertTrue(accepted["trusted_session_anchor_replaced"])
        comparison = accepted["baseline_anchor_comparison_by_channel"]["P22"]
        self.assertEqual(
            comparison["status"],
            "trusted_anchor_replaced_by_operator_attestation",
        )
        self.assertEqual(comparison["previous_status"], "recovery_residual_detected")
        np.testing.assert_allclose(
            test_bridge.trusted_baseline_anchor_spectrum_by_channel["P22"][
                "intensity"
            ],
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"],
        )

    def test_new_acquisition_session_clears_trusted_baseline_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 9000.0 + 3000.0 * np.exp(-0.5 * (x / 0.25) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)
        test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )
        self.assertIn("P22", test_bridge.trusted_baseline_anchor_spectrum_by_channel)

        test_bridge.reset(keep_baseline=False)

        self.assertNotIn("P22", test_bridge.trusted_baseline_anchor_spectrum_by_channel)
        self.assertNotIn("P22", test_bridge.baseline_anchor_comparison_by_channel)

    def test_runtime_release_reanchor_preserves_trusted_session_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        wavelength = np.linspace(1528.0, 1560.0, 128)
        trusted = 8000.0 + 2500.0 * np.exp(
            -0.5 * ((wavelength - 1546.9) / 0.25) ** 2
        )
        recovered = trusted.copy()
        recovered[54:74] *= 0.96
        test_bridge.trusted_baseline_anchor_spectrum_by_channel["P22"] = {
            "wavelength_nm": wavelength.tolist(),
            "intensity": trusted.tolist(),
        }

        result = test_bridge.set_runtime_recovery_spectrum_baseline(
            "P22",
            wavelength,
            recovered,
            sample_count=14,
            span_sec=5.2,
            shape_motion_rms=0.0016,
            common_gain_motion=0.0004,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["trusted_session_anchor_preserved"])
        self.assertEqual(
            test_bridge.baseline_spectrum_status_by_channel["P22"],
            "stable_post_release_recovery_baseline",
        )
        np.testing.assert_allclose(
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"],
            recovered,
        )
        np.testing.assert_allclose(
            test_bridge.trusted_baseline_anchor_spectrum_by_channel["P22"][
                "intensity"
            ],
            trusted,
        )

    def test_runtime_startup_baseline_initializes_trusted_session_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        wavelength = np.linspace(1528.0, 1560.0, 128)
        baseline = 8200.0 + 2700.0 * np.exp(
            -0.5 * ((wavelength - 1546.9) / 0.25) ** 2
        )

        result = test_bridge.set_runtime_startup_spectrum_baseline(
            "P22",
            wavelength,
            baseline,
            sample_count=5,
            span_sec=0.16,
            shape_motion_rms=0.0004,
            common_gain_motion=0.0002,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["trusted_session_anchor_initialized"])
        self.assertEqual(
            test_bridge.baseline_spectrum_status_by_channel["P22"],
            "stable_current_session_startup_baseline",
        )
        self.assertEqual(
            test_bridge.baseline_spectrum_semantic_role_by_channel["P22"],
            "automatic_current_session_startup_no_contact",
        )
        np.testing.assert_allclose(
            test_bridge.trusted_baseline_anchor_spectrum_by_channel["P22"][
                "intensity"
            ],
            baseline,
        )


class UnifiedBaselineEndpointTests(unittest.TestCase):
    def test_new_acquisition_session_invalidates_previous_attestation(self) -> None:
        previous_attestation = {
            "confirmed": True,
            "attested_at_epoch_sec": 123.0,
            "attested_by": "operator",
            "force_evidence": {"force_fz_n": 0.0},
            "status": "operator_and_force_reference_confirmed",
        }
        with patch.object(
            backend_main,
            "GLOBAL_BASELINE_ATTESTATION",
            previous_attestation,
        ), patch.object(
            backend_main,
            "_reset_current_runtime",
            return_value={"ok": True},
        ), patch.object(
            backend_main.bridge,
            "reset",
            return_value={"ok": True},
        ):
            result = backend_main._begin_acquisition_session()

            self.assertFalse(
                backend_main.GLOBAL_BASELINE_ATTESTATION["confirmed"]
            )
            self.assertIsNone(
                backend_main.GLOBAL_BASELINE_ATTESTATION["attested_by"]
            )
            self.assertEqual(
                result["baseline_attestation"]["invalidation_reason"],
                "new_acquisition_session",
            )

    def test_api_reset_invalidates_attestation_only_when_baseline_is_cleared(self) -> None:
        previous_attestation = {
            "confirmed": True,
            "attested_at_epoch_sec": 123.0,
            "attested_by": "operator",
            "force_evidence": None,
            "status": "operator_confirmed_force_reference_unavailable",
        }
        with patch.object(
            backend_main,
            "GLOBAL_BASELINE_ATTESTATION",
            previous_attestation.copy(),
        ), patch.object(
            backend_main,
            "_reset_current_runtime",
            return_value={"ok": True},
        ), patch.object(
            backend_main.bridge,
            "reset",
            return_value={"ok": True},
        ):
            kept = backend_main.reset(keep_baseline=True)
            self.assertNotIn("baseline_attestation", kept)
            self.assertTrue(
                backend_main.GLOBAL_BASELINE_ATTESTATION["confirmed"]
            )

            cleared = backend_main.reset(keep_baseline=False)
            self.assertFalse(
                backend_main.GLOBAL_BASELINE_ATTESTATION["confirmed"]
            )
            self.assertEqual(
                cleared["baseline_attestation"]["invalidation_reason"],
                "api_reset_without_baseline",
            )

    def test_global_baseline_requires_explicit_no_contact_attestation(self) -> None:
        with patch.object(
            backend_main.bridge,
            "set_global_candidate_baseline",
        ) as set_candidate_baseline, patch.object(
            backend_main.bridge,
            "set_baseline",
        ) as set_model_baseline:
            result = backend_main.set_global_candidate_baseline(minimum_frames=30)

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["status"],
            "operator_no_contact_attestation_required",
        )
        self.assertTrue(result["baseline_unchanged"])
        set_model_baseline.assert_not_called()
        set_candidate_baseline.assert_not_called()

    def test_untared_force_offset_does_not_block_operator_baseline(self) -> None:
        candidate_result = {"ok": True, "frame_count": 30}
        model_result = {
            "ok": True,
            "baseline_set": True,
            "current_runtime_spectrum_baseline_ready": True,
            "current_runtime_spectrum_baseline_status": "stable_post_release_recovery_baseline",
        }
        with patch.object(
            backend_main.px6d_reader,
            "status",
            return_value={
                "connected": True,
                "tare_ready": False,
                "sample_fresh": True,
            },
        ), patch.object(
            backend_main,
            "_px6d_reference_for_record",
            return_value={
                "ok": True,
                "tare_ready": False,
                "sample_fresh": True,
                "force_fz_n": 1.5,
            },
        ), patch.object(
            backend_main.bridge,
            "set_global_candidate_baseline",
            return_value=candidate_result,
        ), patch.object(
            backend_main.bridge,
            "set_baseline",
            return_value=model_result,
        ):
            result = backend_main.set_global_candidate_baseline(
                minimum_frames=30,
                no_contact_attested=True,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["no_contact_attestation"]["status"],
            "operator_confirmed_force_reference_pending_zero",
        )

    def test_global_baseline_passes_required_frames_to_model_baseline(self) -> None:
        candidate_result = {"ok": True, "frame_count": 30}
        model_result = {
            "ok": True,
            "baseline_set": True,
            "current_runtime_spectrum_baseline_ready": True,
            "current_runtime_spectrum_baseline_status": "stable_post_release_recovery_baseline",
            "baseline_spectrum_sample_count_by_channel": {"P22": 30},
            "baseline_spectrum_span_sec_by_channel": {"P22": 3.0},
            "baseline_spectrum_noise_ratio_by_channel": {"P22": 0.001},
            "baseline_spectrum_drift_ratio_by_channel": {"P22": 0.001},
        }
        with patch.object(
            backend_main.bridge,
            "set_global_candidate_baseline",
            return_value=candidate_result,
        ), patch.object(
            backend_main.bridge,
            "set_baseline",
            return_value=model_result,
        ) as set_model_baseline:
            result = backend_main.set_global_candidate_baseline(
                minimum_frames=30,
                no_contact_attested=True,
            )

        set_model_baseline.assert_called_once_with(
            {
                "channel_id": "P22",
                "baseline_method": "frozen_baseline",
                "minimum_recent_samples": 30,
                "replace_trusted_session_anchor": True,
            }
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["candidate_display_baseline_ok"])
        self.assertTrue(result["current_runtime_spectrum_baseline"]["ok"])

    def test_global_baseline_rejects_partial_model_baseline(self) -> None:
        with patch.object(
            backend_main.bridge,
            "set_global_candidate_baseline",
            return_value={"ok": True, "frame_count": 30},
        ) as set_candidate_baseline, patch.object(
            backend_main.bridge,
            "set_baseline",
            return_value={
                "ok": True,
                "baseline_set": True,
                "current_runtime_spectrum_baseline_ready": False,
                "current_runtime_spectrum_baseline_status": "insufficient_recovery_baseline_frames",
                "baseline_spectrum_sample_count_by_channel": {"P22": 6},
            },
        ):
            result = backend_main.set_global_candidate_baseline(
                minimum_frames=30,
                no_contact_attested=True,
            )

        self.assertFalse(result["ok"])
        self.assertIn("insufficient_recovery_baseline_frames", result["message"])
        set_candidate_baseline.assert_not_called()


class DesktopLauncherIdentityTests(unittest.TestCase):
    def test_bound_port_is_not_free_even_when_it_is_not_accepting_connections(
        self,
    ) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held_socket:
            held_socket.bind(("127.0.0.1", 0))
            occupied_port = int(held_socket.getsockname()[1])

            self.assertFalse(port_is_free(occupied_port))

    def test_stale_preferred_port_uses_first_free_fallback(self) -> None:
        with patch(
            "desktop_launcher.port_is_free",
            side_effect=lambda port: port == 8641,
        ), patch(
            "desktop_launcher.backend_is_ready",
            return_value=False,
        ), patch("desktop_launcher.write_log"):
            port, should_start = select_backend_port(8640, candidate_count=3)

        self.assertEqual(port, 8641)
        self.assertTrue(should_start)

    def test_compatible_backend_on_preferred_port_is_reused(self) -> None:
        with patch(
            "desktop_launcher.port_is_free",
            return_value=False,
        ), patch(
            "desktop_launcher.backend_is_ready",
            return_value=True,
        ), patch("desktop_launcher.write_log"):
            port, should_start = select_backend_port(8640, candidate_count=3)

        self.assertEqual(port, 8640)
        self.assertFalse(should_start)

    def test_source_self_test_loads_runtime_without_hardware_autostart(self) -> None:
        with patch.dict("os.environ", {}, clear=False), patch(
            "desktop_launcher.write_log"
        ) as write_log:
            exit_code = run_self_test()

        self.assertEqual(exit_code, 0)
        self.assertIn("\"ok\": true", write_log.call_args.args[0])

    def test_expected_backend_identity_is_accepted(self) -> None:
        self.assertTrue(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH",
                    "mode": "standalone_touch_high_sensitivity_300us_spectral_runtime",
                    "backend_contract_version": "touch_current_runtime_api_v1",
                    "default_operator_recognition": "ordinary_fbg_same_day_joint_nine_fbg_beta_v4",
                    "runtime_model": {
                        "loaded": True,
                        "runtime_role": "deployed_current_model_only",
                        "active_runtime_schema": "same_day_joint_nine_fbg_v4",
                        "active_dataset_id": "ordinary_fbg_20260902_same_day_joint_fingerprint_v2",
                        "current_only_bundle": True,
                        "legacy_fallback_enabled": False,
                    },
                    "recognition_runtime": {
                        "active_model_id": "ordinary_fbg_same_day_joint_nine_fbg_beta_v4",
                        "switchable": False,
                        "model_count": 1,
                    },
                }
            )
        )

    def test_legacy_backend_without_contract_version_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH",
                    "mode": "standalone_touch_high_sensitivity_300us_spectral_runtime",
                    "default_operator_recognition": "ordinary_fbg_same_day_joint_nine_fbg_beta_v4",
                }
            )
        )

    def test_backend_with_mismatched_contract_version_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH",
                    "mode": "standalone_touch_high_sensitivity_300us_spectral_runtime",
                    "backend_contract_version": "touch_current_runtime_api_v0",
                    "default_operator_recognition": "ordinary_fbg_same_day_joint_nine_fbg_beta_v4",
                }
            )
        )

    def test_other_touch_backend_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Optical Intensity Twin",
                    "mode": "standalone_bayspec_optical_intensity",
                    "trained_static_model_primary": False,
                }
            )
        )


class OperatorQaProjectionTests(unittest.TestCase):
    def test_estimator_disagreement_is_diagnostics_only_in_trained_ui(self) -> None:
        app_js = (
            backend_main.FRONTEND_ROOT / "app.js"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'new Set(["wavelength_estimator_disagreement"])',
            app_js,
        )
        self.assertIn("diagnostic_only_qa_flags: diagnosticOnlyRawQaFlags", app_js)
        self.assertIn("raw_qa_flags: rawQaFlags", app_js)
        self.assertIn("...operatorRawQaFlags", app_js)


class CaptureProvenanceContractTests(unittest.TestCase):
    @staticmethod
    def _capture_runtime_frame() -> dict:
        return {
            "frame_id": 77,
            "timestamp": 1788315000.125,
            "ingested_at": 1788315000.125,
            "source": "bayspec_sdk_live",
            "wavelength_nm": [1540.0, 1540.1, 1540.2],
            "intensity": [1000.0, 1040.0, 1010.0],
        }

    def test_capture_response_uses_only_an_exact_same_frame_runtime_cache(
        self,
    ) -> None:
        latest = self._capture_runtime_frame()
        frame_key = backend_main._current_runtime_frame_key(latest)
        cached_prediction = {
            "ok": True,
            "status": "ready",
            "classification_model_source": "deployed_test_model",
            "force_model_source": "deployed_test_force_model",
            "position": {"label": "P11", "confidence": 0.91},
        }
        with (
            patch.object(
                backend_main,
                "_model_display_source_gate",
                return_value={"formal_spectrum_input_allowed": True},
            ),
            patch.object(backend_main, "CURRENT_RUNTIME_LAST_FRAME_KEY", frame_key),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_LAST_PAYLOAD",
                cached_prediction,
            ),
            patch.object(backend_main, "_predict_current_runtime") as predict,
        ):
            payload = backend_main._capture_temporal_response(latest)

        predict.assert_not_called()
        self.assertTrue(payload["model_ready"])
        self.assertTrue(payload["capture_response_frame_match"])
        self.assertEqual(
            payload["capture_response_source"],
            "same_frame_runtime_cache",
        )
        self.assertEqual(payload["position"]["label"], "P11")

    def test_capture_response_defers_instead_of_blocking_on_model_inference(
        self,
    ) -> None:
        latest = self._capture_runtime_frame()
        with (
            patch.object(
                backend_main,
                "_model_display_source_gate",
                return_value={"formal_spectrum_input_allowed": True},
            ),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_LAST_FRAME_KEY",
                ("different-frame",),
            ),
            patch.object(backend_main, "CURRENT_RUNTIME_LAST_PAYLOAD", None),
            patch.object(backend_main, "_predict_current_runtime") as predict,
            patch.object(backend_main, "_current_runtime_status") as full_status,
        ):
            started = time.perf_counter()
            payload = backend_main._capture_temporal_response(latest)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        predict.assert_not_called()
        full_status.assert_not_called()
        self.assertLess(elapsed_ms, 50.0)
        self.assertFalse(payload["model_ready"])
        self.assertFalse(payload["capture_response_frame_match"])
        self.assertEqual(payload["status"], "capture_response_deferred")
        self.assertEqual(
            payload["capture_response_source"],
            "deferred_for_high_rate_capture",
        )
        self.assertTrue(payload["raw_spectrum_authoritative"])
        self.assertTrue(payload["offline_reconstruction_supported"])

    def test_capture_response_rejects_qa_invalid_frame_without_model_state_change(
        self,
    ) -> None:
        latest = {
            "frame_id": 55,
            "source": "static_http_ingest",
            "peak_axis_type": "wavelength_nm",
            "qa_status": "invalid",
            "qa_flags": ["spectrum_length_mismatch"],
        }
        stopped = {"active": False, "freshness": "stopped"}

        with (
            patch.object(
                backend_main.export_watcher,
                "status",
                return_value=stopped,
            ),
            patch.object(
                backend_main.sdk_live_reader,
                "status",
                return_value=stopped,
            ),
            patch.object(backend_main, "_predict_current_runtime") as predict,
        ):
            payload = backend_main._capture_temporal_response(latest)

        predict.assert_not_called()
        self.assertFalse(payload["formal_spectrum_input_allowed"])
        self.assertFalse(payload["model_ready"])
        self.assertEqual(payload["status"], "current_runtime_source_blocked")
        self.assertEqual(
            payload["reason"],
            "spectrum_qa_invalid_for_formal_recognition",
        )

    def test_optical_device_identity_uses_bridge_module_resolvers(self) -> None:
        with (
            patch.object(
                backend_main,
                "_current_runtime_baseline_token",
                return_value=(
                    "baseline-token",
                    {"ok": True, "baseline_spectrum_status": "ready"},
                ),
            ),
            patch.object(
                backend_main,
                "_current_runtime_status",
                return_value={"loaded": True},
            ),
            patch.object(
                backend_main.sdk_live_reader,
                "status",
                return_value={"source": "test_sdk", "active": False},
            ),
            patch.object(
                backend_main.px6d_reader,
                "status",
                return_value={"connected": False},
            ),
            patch.object(
                backend_main,
                "configured_device_id",
                return_value="F1871328",
            ),
            patch.object(
                backend_main,
                "configured_sense_export_root",
                return_value=Path("C:/Sense/Spectrum_Data"),
            ),
            patch.object(
                backend_main,
                "_artifact_identity",
                side_effect=lambda path: {"path": str(path)},
            ),
        ):
            payload = backend_main._capture_provenance_snapshot()

        self.assertEqual(
            payload["optical_device"]["configured_device_id"],
            "F1871328",
        )
        self.assertEqual(
            payload["optical_device"]["configured_sense_export_root"],
            "C:\\Sense\\Spectrum_Data",
        )


class CurrentRuntimeStatusSnapshotTests(unittest.TestCase):
    def test_runtime_status_reads_mutable_state_under_one_lock(self) -> None:
        lock_state = {"held": False}

        class TrackingLock:
            def __enter__(self):
                self.assert_not_held()
                lock_state["held"] = True
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                lock_state["held"] = False

            @staticmethod
            def assert_not_held() -> None:
                if lock_state["held"]:
                    raise AssertionError("runtime status lock entered recursively")

        class GuardedAdapter:
            @staticmethod
            def _require_lock() -> None:
                if not lock_state["held"]:
                    raise AssertionError("runtime adapter read outside status snapshot lock")

            @property
            def bundle(self):
                self._require_lock()
                return {
                    "schema_version": "snapshot-test-v1",
                    "feature_schema": {"fields": ["spectrum"]},
                }

            @property
            def force_min_n(self):
                self._require_lock()
                return 0.0

            @property
            def force_calibrated_max_n(self):
                self._require_lock()
                return 5.0

            @property
            def classification_model_source(self):
                self._require_lock()
                return "snapshot_classifier"

            @property
            def force_model_source(self):
                self._require_lock()
                return "snapshot_force_model"

        with (
            patch.object(backend_main, "CURRENT_RUNTIME_LOCK", TrackingLock()),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_ADAPTER",
                GuardedAdapter(),
            ),
            patch.object(backend_main, "CURRENT_RUNTIME_UNIQUE_FRAME_COUNT", 17),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_STARTUP_BASELINE_STATUS",
                {"status": "ready", "ready": True, "frame_count": 5},
            ),
        ):
            status = backend_main._current_runtime_status()

        self.assertFalse(lock_state["held"])
        self.assertTrue(status["loaded"])
        self.assertEqual(status["schema_version"], "snapshot-test-v1")
        self.assertEqual(status["unique_frame_count"], 17)
        self.assertEqual(
            status["runtime_startup_baseline"]["state"]["frame_count"],
            5,
        )
        self.assertEqual(
            status["classification_model_source"],
            "snapshot_classifier",
        )


class SpectrumFrameIdentityTests(unittest.TestCase):
    def test_distinct_spectra_with_same_scalar_summary_have_distinct_keys(self) -> None:
        wavelength = [1540.0, 1541.0, 1542.0, 1543.0, 1544.0]
        first = [1.0, 2.0, 3.0, 4.0, 5.0]
        second = [1.0, 3.0, 3.0, 3.0, 5.0]

        self.assertEqual(sum(first), sum(second))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[len(first) // 2], second[len(second) // 2])
        self.assertEqual(first[-1], second[-1])
        self.assertNotEqual(
            backend_main._spectrum_fingerprint(wavelength, first),
            backend_main._spectrum_fingerprint(wavelength, second),
        )

    def test_late_concurrent_frame_cannot_rewind_runtime_state(self) -> None:
        wavelength = [1540.0 + 0.1 * index for index in range(32)]
        baseline_intensity = [1000.0 + index for index in range(32)]
        newer = {
            "frame_id": 102,
            "timestamp": 200.2,
            "source": "bayspec_direct_usb20bs_sdk",
            "wavelength_nm": wavelength,
            "intensity": [value * 0.98 for value in baseline_intensity],
        }
        late = {
            "frame_id": 101,
            "timestamp": 200.1,
            "source": "bayspec_direct_usb20bs_sdk",
            "wavelength_nm": wavelength,
            "intensity": [value * 0.99 for value in baseline_intensity],
        }
        baseline = {
            "wavelength_nm": wavelength,
            "intensity": baseline_intensity,
        }
        baseline_token = backend_main._spectrum_token(
            wavelength,
            baseline_intensity,
        )

        class RecordingAdapter:
            def __init__(self) -> None:
                self.timestamps: list[float | None] = []

            def update(self, _wavelength, _intensity, *, source_timestamp_sec):
                self.timestamps.append(source_timestamp_sec)
                return {"ok": True, "status": "ready", "sequence": len(self.timestamps)}

            @staticmethod
            def consume_pending_runtime_baseline_update():
                return None

        adapter = RecordingAdapter()
        pair = {"ok": True, "latest": newer, "baseline": baseline}
        with (
            patch.object(backend_main, "CURRENT_RUNTIME_ADAPTER", adapter),
            patch.object(
                backend_main,
                "_ensure_current_runtime_startup_baseline",
                return_value=pair,
            ),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_BASELINE_TOKEN",
                baseline_token,
            ),
            patch.object(backend_main, "CURRENT_RUNTIME_LAST_FRAME_KEY", None),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_LAST_SOURCE_TIMESTAMP_SEC",
                None,
            ),
            patch.object(backend_main, "CURRENT_RUNTIME_UNIQUE_FRAME_COUNT", 0),
            patch.object(backend_main, "CURRENT_RUNTIME_LAST_PAYLOAD", None),
            patch.object(
                backend_main,
                "CURRENT_RUNTIME_PREDICTION_CACHE",
                OrderedDict(),
            ),
        ):
            first = backend_main._predict_current_runtime(newer)
            second = backend_main._predict_current_runtime(late)

        self.assertEqual(adapter.timestamps, [200.2])
        self.assertEqual(first["unique_frame_count"], 1)
        self.assertEqual(second["unique_frame_count"], 1)
        self.assertTrue(second["out_of_order_frame_ignored"])
        self.assertEqual(second["ignored_source_timestamp_sec"], 200.1)
        self.assertEqual(
            second["latest_processed_source_timestamp_sec"],
            200.2,
        )


if __name__ == "__main__":
    unittest.main()
