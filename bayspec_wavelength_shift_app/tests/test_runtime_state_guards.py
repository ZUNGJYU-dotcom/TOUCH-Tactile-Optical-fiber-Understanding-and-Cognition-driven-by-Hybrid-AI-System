from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from backend import main as backend_main
from backend.main import SenseExportWatcher, _model_display_source_gate
from bridge import BaySpecWavelengthShiftBridge
from desktop_launcher import (
    health_payload_is_expected,
    run_backend,
    run_self_test,
    stop_owned_backend,
    wait_until_ready,
)
from sdk_live import BaySpecSdkLiveReader
from src.hybrid_spectrum.session_level_calibration import (
    CORE_FEATURE_NAMES,
    PerPositionOrdinalCalibrator,
)


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
        with patch("desktop_launcher.uvicorn.Config", return_value=object()), patch(
            "desktop_launcher.uvicorn.Server", FailingServer
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


class SdkLiveReaderStateTests(unittest.TestCase):
    def _reader(self) -> BaySpecSdkLiveReader:
        return BaySpecSdkLiveReader(_BridgeStub(), Path("."))

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
        self.assertGreaterEqual(observed_delays[0], 0.019)
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


class StaticCandidateShadowTests(unittest.TestCase):
    def test_shadow_candidate_never_claims_operator_or_twin_control(self) -> None:
        class CandidateStub:
            def predict(self, *_args, **_kwargs):
                return {
                    "position": {"label": "P22"},
                    "force_level": {"label": "normal"},
                }

        wavelength = np.linspace(1528.0, 1560.0, 64).tolist()
        intensity = np.linspace(1000.0, 2000.0, 64).tolist()
        with patch.object(
            backend_main,
            "STATIC_SPECTRAL_CANDIDATE_PREDICTOR",
            CandidateStub(),
        ):
            result = backend_main._predict_static_spectral_shadow(
                wavelength,
                intensity,
                wavelength,
                intensity,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "shadow_ready")
        self.assertFalse(result["drives_operator_ui"])
        self.assertFalse(result["drives_digital_twin"])
        self.assertEqual(result["prediction"]["position"]["label"], "P22")

    def test_session_calibrated_force_is_baseline_bound_and_shadow_only(self) -> None:
        samples = [
            {
                "position": "P22",
                "level": level,
                "features": {name: value for name in CORE_FEATURE_NAMES},
            }
            for level, value in (("light", 1.0), ("normal", 2.0), ("hard", 4.0))
        ]
        calibrator = PerPositionOrdinalCalibrator.fit(
            samples,
            baseline_token="baseline-a",
            required_positions=("P22",),
        )
        prediction = {
            "response_calibration_features": {
                name: 2.2 for name in CORE_FEATURE_NAMES
            }
        }
        temporal = {
            "ready": True,
            "contact_label": "contact",
            "position_label": "P22",
        }
        with patch.object(
            backend_main,
            "STATIC_SPECTRAL_SESSION_CALIBRATOR",
            calibrator,
        ):
            result = backend_main._apply_session_level_calibration(
                prediction,
                temporal,
                baseline_token="baseline-a",
            )
            mismatch = backend_main._apply_session_level_calibration(
                prediction,
                temporal,
                baseline_token="baseline-b",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["label"], "normal")
        self.assertFalse(result["drives_operator_ui"])
        self.assertFalse(result["drives_digital_twin"])
        self.assertEqual(mismatch["status"], "baseline_mismatch_calibration_invalidated")

    def test_exact_spectrum_token_changes_with_one_sample(self) -> None:
        wavelength = [1.0, 2.0, 3.0]
        first = backend_main._spectrum_token(wavelength, [10.0, 20.0, 30.0])
        repeated = backend_main._spectrum_token(wavelength, [10.0, 20.0, 30.0])
        changed = backend_main._spectrum_token(wavelength, [10.0, 20.0, 30.001])

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)


class DynamicTemporalShadowEndpointTests(unittest.TestCase):
    @staticmethod
    def _spectrum_pair(*, frame_id: int, timestamp: float) -> dict:
        wavelength = np.linspace(1528.0, 1560.0, 64).tolist()
        baseline_intensity = np.linspace(1000.0, 2000.0, 64).tolist()
        current_intensity = [
            value + frame_id * 0.25 for value in baseline_intensity
        ]
        return {
            "ok": True,
            "latest": {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source": "static_http_ingest",
                "wavelength_nm": wavelength,
                "intensity": current_intensity,
            },
            "baseline": {
                "wavelength_nm": wavelength,
                "intensity": baseline_intensity,
            },
        }

    def test_unique_frames_advance_history_and_duplicate_poll_is_ignored(self) -> None:
        class AdapterStub:
            def __init__(self) -> None:
                self.bundle = {
                    "schema_version": "dynamic_temporal_shadow_candidate_v2",
                    "status": "shadow_only_not_primary",
                    "release_guard_grouped_cv": {
                        "unsafe_early_release_trigger_count": 0,
                    },
                }
                self.clear_count = 0
                self.baseline_count = 0
                self.run_inference_flags: list[bool] = []

            def clear(self) -> None:
                self.clear_count += 1

            def set_baseline(self, _wavelength, _intensity) -> None:
                self.baseline_count += 1

            def consume_pending_runtime_baseline_update(self):
                return None

            def update(
                self,
                _wavelength,
                _intensity,
                *,
                run_inference: bool,
                physical_frame: bool = True,
                external_no_contact_hint: bool | None = None,
                source_timestamp_sec: float | None = None,
            ):
                self.run_inference_flags.append(run_inference)
                return {
                    "status": (
                        "shadow_ready" if run_inference else "inference_stride_hold"
                    ),
                    "ready": run_inference,
                }

        adapter = AdapterStub()
        first_pair = self._spectrum_pair(frame_id=1, timestamp=1.0)
        second_pair = self._spectrum_pair(frame_id=2, timestamp=1.04)
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ), patch.object(
            backend_main.bridge,
            "spectral_model_input",
            side_effect=[first_pair, first_pair, second_pair],
        ):
            backend_main._reset_dynamic_temporal_shadow("test_setup")
            first = backend_main._predict_dynamic_temporal_shadow()
            duplicate = backend_main._predict_dynamic_temporal_shadow()
            second = backend_main._predict_dynamic_temporal_shadow()

        self.assertEqual(adapter.baseline_count, 1)
        self.assertEqual(adapter.run_inference_flags, [True, True])
        self.assertEqual(first["unique_frame_count"], 1)
        self.assertTrue(first["inference_executed_this_frame"])
        self.assertTrue(duplicate["duplicate_frame_ignored"])
        self.assertEqual(duplicate["unique_frame_count"], 1)
        self.assertEqual(second["unique_frame_count"], 2)
        self.assertTrue(second["inference_executed_this_frame"])
        for result in (first, duplicate, second):
            self.assertFalse(result["drives_operator_ui"])
            self.assertFalse(result["drives_digital_twin"])
            self.assertEqual(
                result["runtime_role"],
                "shadow_only_not_driving_digital_twin",
            )

    def test_slow_physical_frame_is_resampled_to_model_time_scale(self) -> None:
        class AdapterStub:
            bundle = {"frame_interval_sec_estimated": 0.04}

            def __init__(self) -> None:
                self.calls: list[dict] = []

            def clear(self) -> None:
                self.calls.clear()

            def set_baseline(self, _wavelength, _intensity) -> None:
                return None

            def consume_pending_runtime_baseline_update(self):
                return None

            def update(
                self,
                _wavelength,
                _intensity,
                *,
                run_inference: bool,
                physical_frame: bool,
                external_no_contact_hint: bool | None,
                source_timestamp_sec: float | None = None,
            ) -> dict:
                self.calls.append(
                    {
                        "run_inference": run_inference,
                        "physical_frame": physical_frame,
                        "external_no_contact_hint": external_no_contact_hint,
                    }
                )
                return {"status": "shadow_ready", "ready": True}

        adapter = AdapterStub()
        first_pair = self._spectrum_pair(frame_id=1, timestamp=1.0)
        second_pair = self._spectrum_pair(frame_id=2, timestamp=1.4)
        first_pair["latest"]["response_level"] = "no_contact"
        first_pair["latest"]["qa_status"] = "ok"
        second_pair["latest"]["response_level"] = "no_contact"
        second_pair["latest"]["qa_status"] = "ok"
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ), patch.object(
            backend_main.bridge,
            "spectral_model_input",
            side_effect=[first_pair, second_pair],
        ):
            backend_main._reset_dynamic_temporal_shadow("test_resample")
            first = backend_main._predict_dynamic_temporal_shadow()
            second = backend_main._predict_dynamic_temporal_shadow()

        self.assertEqual(first["temporal_resample_steps"], 1)
        self.assertEqual(second["temporal_resample_steps"], 10)
        self.assertEqual(len(adapter.calls), 11)
        self.assertEqual(sum(call["physical_frame"] for call in adapter.calls), 2)
        self.assertTrue(adapter.calls[-1]["run_inference"])
        self.assertTrue(adapter.calls[-1]["external_no_contact_hint"])

    def test_rejected_runtime_baseline_is_rolled_back_in_adapter(self) -> None:
        pair = self._spectrum_pair(frame_id=7, timestamp=7.0)
        original_baseline = np.asarray(pair["baseline"]["intensity"], dtype=float)
        recovered_baseline = original_baseline * 1.05

        class AdapterStub:
            bundle = {"frame_interval_sec_estimated": 0.04}

            def __init__(self) -> None:
                self.baselines: list[np.ndarray] = []
                self.pending = {
                    "wavelength_nm": np.asarray(
                        pair["baseline"]["wavelength_nm"], dtype=float
                    ),
                    "intensity": recovered_baseline,
                    "sample_count": 8,
                    "span_sec": 1.0,
                }

            def clear(self) -> None:
                return None

            def set_baseline(self, _wavelength, intensity) -> None:
                self.baselines.append(np.asarray(intensity, dtype=float).copy())

            def consume_pending_runtime_baseline_update(self):
                pending, self.pending = self.pending, None
                return pending

            def update(
                self,
                _wavelength,
                _intensity,
                *,
                run_inference: bool,
                physical_frame: bool,
                external_no_contact_hint: bool | None,
                source_timestamp_sec: float | None = None,
            ) -> dict:
                return {
                    "status": "runtime_reference_reanchored",
                    "ready": False,
                }

        adapter = AdapterStub()
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ), patch.object(
            backend_main.bridge,
            "spectral_model_input",
            return_value=pair,
        ), patch.object(
            backend_main.bridge,
            "set_runtime_recovery_spectrum_baseline",
            return_value={
                "ok": False,
                "status": "runtime_recovery_baseline_invalid",
            },
        ):
            backend_main._reset_dynamic_temporal_shadow("test_rollback")
            result = backend_main._predict_dynamic_temporal_shadow()

        self.assertTrue(result["ok"])
        self.assertEqual(len(adapter.baselines), 2)
        np.testing.assert_allclose(adapter.baselines[0], original_baseline)
        np.testing.assert_allclose(adapter.baselines[1], original_baseline)
        update = result["runtime_baseline_update"]
        self.assertFalse(update["ok"])
        self.assertTrue(update["adapter_baseline_rollback_applied"])
        self.assertEqual(
            update["rollback_baseline_token"],
            backend_main._spectrum_token(
                pair["baseline"]["wavelength_nm"],
                pair["baseline"]["intensity"],
            ),
        )
        self.assertEqual(result["baseline_token"], update["rollback_baseline_token"])

    def test_global_frame_does_not_run_dynamic_shadow_by_default(self) -> None:
        stopped = {"active": False, "freshness": "stopped"}
        with patch.object(
            backend_main.bridge,
            "frame",
            return_value={"ok": True, "latest": None},
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sense_controller,
            "status",
            return_value={"ok": True},
        ), patch.object(
            backend_main,
            "_predict_static_spectral_frame",
            return_value={"ok": False, "status": "not_requested"},
        ), patch.object(
            backend_main,
            "_predict_dynamic_temporal_shadow",
        ) as predict_dynamic:
            result = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=False,
                include_shadow=False,
                include_dynamic_shadow=False,
            )

        predict_dynamic.assert_not_called()
        shadow = result["dynamic_temporal_shadow"]
        self.assertEqual(shadow["status"], "dynamic_shadow_not_requested")
        self.assertFalse(shadow["drives_operator_ui"])
        self.assertFalse(shadow["drives_digital_twin"])

    def test_temporal_display_adapter_maps_active_prediction_to_twin_contract(self) -> None:
        payload = {
            "ok": True,
            "status": "shadow_ready",
            "prediction": {
                "ready": True,
                "status": "shadow_ready",
                "history_frames": 20,
                "required_frames": 20,
                "frame_counter": 31,
                "contact": {"label": "contact", "confidence": 0.94},
                "position": {"label": "P21", "confidence": 0.81},
                "response_level": {"label": "normal", "confidence": 0.76},
                "operational_state": "active_contact",
                "release_guard": {"release_latched": False},
                "digital_twin_proxy": {
                    "active": True,
                    "position_id": "P21",
                    "response_level": "normal",
                    "deformation_proxy": 0.58,
                    "surface_grid": [
                        [0.31, 0.58, 0.31],
                        [0.16, 0.31, 0.16],
                        [0.04, 0.08, 0.04],
                    ],
                    "surface_metrics": {
                        "surface_peak": 0.58,
                        "surface_centroid_x": 0.0,
                        "surface_centroid_y": 1.0,
                        "dominant_channel": "P21",
                    },
                    "visualization_semantics": "single_finger_contact_patch",
                    "physical_output_semantics": "uncalibrated_manual_response_level",
                },
            },
        }

        result = backend_main._dynamic_temporal_display_prediction(payload)

        self.assertTrue(result["ok"])
        self.assertTrue(result["drives_operator_ui"])
        self.assertTrue(result["drives_digital_twin"])
        self.assertFalse(result["deployment_ready"])
        prediction = result["prediction"]
        self.assertEqual(prediction["position"]["label"], "P21")
        self.assertEqual(prediction["force_level"]["label"], "normal")
        self.assertEqual(prediction["digital_twin"]["position_id"], "P21")
        self.assertEqual(prediction["digital_twin"]["deformation_proxy"], 0.58)

    def test_temporal_display_adapter_keeps_release_at_zero_deformation(self) -> None:
        payload = {
            "ok": True,
            "status": "released_residual_latched",
            "prediction": {
                "ready": True,
                "status": "released_residual_latched",
                "contact": {"label": "no_contact", "confidence": None},
                "position": None,
                "response_level": None,
                "operational_state": "no_contact_after_confirmed_release",
                "release_guard": {"release_latched": True},
                "digital_twin_proxy": {
                    "active": False,
                    "position_id": None,
                    "response_level": "no_contact",
                    "deformation_proxy": 0.0,
                    "surface_grid": [[0.0, 0.0, 0.0] for _ in range(3)],
                    "surface_metrics": {"surface_peak": 0.0},
                },
            },
        }

        result = backend_main._dynamic_temporal_display_prediction(payload)

        self.assertTrue(result["ok"])
        prediction = result["prediction"]
        self.assertEqual(prediction["contact"]["label"], "no_contact")
        self.assertFalse(prediction["digital_twin"]["active"])
        self.assertEqual(prediction["digital_twin"]["deformation_proxy"], 0.0)

    def test_temporal_display_adapter_blocks_warming_window(self) -> None:
        result = backend_main._dynamic_temporal_display_prediction(
            {
                "ok": True,
                "status": "window_warming_up",
                "prediction": {
                    "ready": False,
                    "status": "window_warming_up",
                    "history_frames": 7,
                    "required_frames": 20,
                },
            }
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "temporal_window_not_ready")
        self.assertEqual(result["history_frames"], 7)
        self.assertFalse(result["drives_digital_twin"])

    def test_global_frame_selects_temporal_prediction_in_validation_mode(self) -> None:
        stopped = {"active": False, "freshness": "stopped"}
        temporal_payload = {
            "ok": True,
            "status": "shadow_ready",
            "prediction": {
                "ready": True,
                "status": "shadow_ready",
                "contact": {"label": "contact", "confidence": 0.93},
                "position": {"label": "P32", "confidence": 0.79},
                "response_level": {"label": "light", "confidence": 0.71},
                "digital_twin_proxy": {
                    "active": True,
                    "position_id": "P32",
                    "response_level": "light",
                    "deformation_proxy": 0.28,
                    "surface_grid": [
                        [0.01, 0.04, 0.08],
                        [0.04, 0.12, 0.28],
                        [0.01, 0.04, 0.08],
                    ],
                    "surface_metrics": {
                        "surface_peak": 0.28,
                        "dominant_channel": "P32",
                    },
                },
            },
        }
        with patch.object(
            backend_main.bridge,
            "frame",
            return_value={"ok": True, "latest": None},
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sense_controller,
            "status",
            return_value={"ok": True},
        ), patch.object(
            backend_main,
            "_predict_static_spectral_frame",
            return_value={"ok": True, "status": "ready", "prediction": {"source": "static"}},
        ) as predict_static, patch.object(
            backend_main,
            "_predict_dynamic_temporal_shadow",
            return_value=temporal_payload,
        ):
            result = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=False,
                include_shadow=False,
                include_dynamic_shadow=True,
                temporal_validation_mode=True,
            )

        self.assertTrue(result["temporal_validation_mode"])
        self.assertTrue(result["model_assisted_display_allowed"])
        self.assertEqual(
            result["active_spectral_model_source"],
            "dynamic_temporal_v3_validation",
        )
        self.assertEqual(
            result["active_spectral_prediction"]["position"]["label"],
            "P32",
        )
        self.assertEqual(
            result["active_spectral_prediction"]["digital_twin"]["deformation_proxy"],
            0.28,
        )
        predict_static.assert_not_called()
        self.assertEqual(
            result["trained_static_spectral_frame"]["status"],
            "skipped_temporal_validation_mode",
        )

    def test_reset_clears_cached_dynamic_trial_state(self) -> None:
        class AdapterStub:
            bundle = {}

            def __init__(self) -> None:
                self.clear_count = 0

            def clear(self) -> None:
                self.clear_count += 1

        adapter = AdapterStub()
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ):
            backend_main.DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN = "old-baseline"
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY = ("old",)
            backend_main.DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT = 99
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD = {"old": True}
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC = 123.0
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM = np.ones(3)
            result = backend_main._reset_dynamic_temporal_shadow("baseline_replaced")

        self.assertTrue(result["ok"])
        self.assertEqual(adapter.clear_count, 1)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY)
        self.assertEqual(backend_main.DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT, 0)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM)


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
        self.assertIn('await requestJSON(\n      "/api/reset?keep_baseline=false"', self.source)
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
        self.assertIn("refresh: false", self.source)
        self.assertIn('boot().catch((error) => {', self.source)

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
                    "'trained_static_spectrum_api_v2'"
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
            "integration": 40000,
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

        self.assertTrue(first["static_model_spectrum_baseline_ready"])
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
        self.assertTrue(rejected["static_model_spectrum_baseline_rejected"])
        self.assertFalse(rejected["static_model_spectrum_baseline_ready"])
        self.assertEqual(
            rejected["static_model_spectrum_baseline_status"],
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

        self.assertTrue(result["static_model_spectrum_baseline_ready"])
        comparison = result["baseline_anchor_comparison_by_channel"]["P22"]
        self.assertNotEqual(comparison["status"], "recovery_residual_detected")
        self.assertAlmostEqual(comparison["common_gain_ratio"], 1.06, places=3)

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


class UnifiedBaselineEndpointTests(unittest.TestCase):
    def test_global_baseline_passes_required_frames_to_model_baseline(self) -> None:
        candidate_result = {"ok": True, "frame_count": 30}
        model_result = {
            "ok": True,
            "baseline_set": True,
            "static_model_spectrum_baseline_ready": True,
            "static_model_spectrum_baseline_status": "stable_post_release_recovery_baseline",
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
            result = backend_main.set_global_candidate_baseline(minimum_frames=30)

        set_model_baseline.assert_called_once_with(
            {
                "channel_id": "P22",
                "baseline_method": "frozen_baseline",
                "minimum_recent_samples": 30,
            }
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["candidate_display_baseline_ok"])
        self.assertTrue(result["static_model_spectrum_baseline"]["ok"])

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
                "static_model_spectrum_baseline_ready": False,
                "static_model_spectrum_baseline_status": "insufficient_recovery_baseline_frames",
                "baseline_spectrum_sample_count_by_channel": {"P22": 6},
            },
        ):
            result = backend_main.set_global_candidate_baseline(minimum_frames=30)

        self.assertFalse(result["ok"])
        self.assertIn("insufficient_recovery_baseline_frames", result["message"])
        set_candidate_baseline.assert_not_called()


class DesktopLauncherIdentityTests(unittest.TestCase):
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
                    "app": "TOUCH System Trained Static Spectrum Twin",
                    "mode": "standalone_bayspec_trained_static_spectrum_twin",
                    "backend_contract_version": "trained_static_spectrum_api_v2",
                    "trained_static_model_primary": True,
                }
            )
        )

    def test_legacy_backend_without_contract_version_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Trained Static Spectrum Twin",
                    "mode": "standalone_bayspec_trained_static_spectrum_twin",
                    "trained_static_model_primary": True,
                }
            )
        )

    def test_backend_with_mismatched_contract_version_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Trained Static Spectrum Twin",
                    "mode": "standalone_bayspec_trained_static_spectrum_twin",
                    "backend_contract_version": "trained_static_spectrum_api_v1",
                    "trained_static_model_primary": True,
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
        self.assertIn("carrier_qa_flags: rawQaFlags", app_js)


if __name__ == "__main__":
    unittest.main()
