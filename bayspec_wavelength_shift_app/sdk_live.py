"""Direct BaySpec USB20BS SDK live reader.

The vendor USB DLL is 32-bit and exposes C++ instance methods, so the 64-bit
Python backend cannot load it directly. A small x86 helper process reads frames
from the DLL and streams JSON lines to this module.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from bridge import configured_device_id, configured_sense_export_root
from spectrum_processing import SpectrumDisplayProcessor


DEFAULT_INTEGRATION_US = 5000
DEFAULT_INTERVAL_MS = 20


class BaySpecSdkLiveReader:
    HELPER_TIMEOUT_SEC = 12.0
    MAX_HELPER_STDOUT_CHARS = 256 * 1024
    MAX_HELPER_STDERR_CHARS = 64 * 1024
    MAX_HELPER_MESSAGES = 8
    MAX_SPECTRUM_POINTS = 16_384

    def __init__(self, bridge: Any, app_root: Path) -> None:
        self.bridge = bridge
        self.app_root = Path(app_root)
        config_root = (
            self.app_root / "config"
            if (self.app_root / "config").exists()
            else self.app_root.parent / "config"
        )
        self.spectrum_processor = SpectrumDisplayProcessor(
            config_root / "spectrum_processing.yaml"
        )
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.channel_id = "P22"
        self.interval_ms = DEFAULT_INTERVAL_MS
        self.integration = DEFAULT_INTEGRATION_US
        self.started_at: float | None = None
        self.last_frame_time: float | None = None
        self.last_status: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_stderr: str | None = None
        self.frame_count = 0
        self.received_frame_count = 0
        self.rejected_frame_count = 0
        self.last_result: dict[str, Any] | None = None
        self.wavelength_grid_cache: list[float] | None = None
        self.desired_active = False
        self.generation = 0
        self.restart_count = 0
        self.last_exit_code: int | None = None
        self.acquisition_strategy = "single_frame_process_restart"
        self.lifecycle_status = "stopped"
        self.last_operation_status = "idle"
        self.consecutive_failure_count = 0
        self.retry_backoff_sec = 0.0
        self.stale_session_frame_count = 0
        self.helper_timeout_count = 0
        self.helper_output_overflow_count = 0
        self.invalid_helper_message_count = 0
        self.last_helper_stdout_chars = 0
        self.last_helper_stderr_chars = 0
        self.last_acquisition_duration_ms: float | None = None
        self.last_cycle_delay_ms: float | None = None
        self._start_in_progress = False
        self._start_cancel_requested = False
        self._start_done_event = threading.Event()
        self._start_done_event.set()
        self._stop_event = threading.Event()

    @property
    def helper_path(self) -> Path:
        return self.app_root / "sdk_probe" / "BaySpecSdkStream.exe"

    def status(self) -> dict[str, Any]:
        with self.lock:
            worker_alive = bool(self.thread is not None and self.thread.is_alive())
            running = self.process is not None and self.process.poll() is None
            now = time.time()
            age = now - self.last_frame_time if self.last_frame_time is not None else None
            active = bool(self.desired_active and worker_alive)
            if self._start_in_progress and not active:
                freshness = "starting"
            elif not self.desired_active and worker_alive:
                freshness = "stopping"
            elif not self.desired_active:
                freshness = "stopped"
            elif not worker_alive:
                freshness = "error"
            elif self.last_frame_time is None:
                freshness = "error" if self.last_error else "waiting_for_sdk_frame"
            elif age is not None and age <= max(3.0, self.interval_ms / 1000.0 * 12):
                freshness = "live"
            else:
                freshness = "stale"
            return {
                "active": active,
                "requested_active": self.desired_active,
                "acquisition_session_id": self.generation,
                "worker_alive": worker_alive,
                "stop_requested": bool(self._stop_event.is_set() and worker_alive),
                "start_in_progress": self._start_in_progress,
                "start_cancel_requested": self._start_cancel_requested,
                "lifecycle_status": self.lifecycle_status,
                "last_operation_status": self.last_operation_status,
                "process_running": running,
                "source": "direct_bayspec_usb20bs_sdk_helper",
                "acquisition_strategy": self.acquisition_strategy,
                "channel_id": self.channel_id,
                "interval_ms": self.interval_ms,
                "integration": self.integration,
                "helper_path": str(self.helper_path),
                "helper_exists": self.helper_path.exists(),
                "process_id": self.process.pid if running and self.process is not None else None,
                "started_at": self.started_at,
                "last_frame_time": self.last_frame_time,
                "seconds_since_last_frame": age,
                "freshness": freshness,
                "frame_count": self.frame_count,
                "received_frame_count": self.received_frame_count,
                "rejected_frame_count": self.rejected_frame_count,
                "restart_count": self.restart_count,
                "consecutive_failure_count": self.consecutive_failure_count,
                "retry_backoff_sec": self.retry_backoff_sec,
                "stale_session_frame_count": self.stale_session_frame_count,
                "helper_timeout_count": self.helper_timeout_count,
                "helper_output_overflow_count": self.helper_output_overflow_count,
                "invalid_helper_message_count": self.invalid_helper_message_count,
                "last_helper_stdout_chars": self.last_helper_stdout_chars,
                "last_helper_stderr_chars": self.last_helper_stderr_chars,
                "last_acquisition_duration_ms": self.last_acquisition_duration_ms,
                "last_cycle_delay_ms": self.last_cycle_delay_ms,
                "helper_output_limits": {
                    "stdout_chars": self.MAX_HELPER_STDOUT_CHARS,
                    "stderr_chars": self.MAX_HELPER_STDERR_CHARS,
                    "messages": self.MAX_HELPER_MESSAGES,
                    "spectrum_points": self.MAX_SPECTRUM_POINTS,
                },
                "last_exit_code": self.last_exit_code,
                "last_status": self.last_status,
                "last_error": self.last_error,
                "last_stderr": self.last_stderr,
                "last_result": self.last_result,
                "spectrum_processing": self.spectrum_processor.status(),
            }

    def processing_status(self) -> dict[str, Any]:
        return self.spectrum_processor.status()

    def update_processing_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self.spectrum_processor.update_settings(updates)

    def capture_background(self) -> dict[str, Any]:
        return self.spectrum_processor.capture_background()

    def clear_background(self) -> dict[str, Any]:
        return self.spectrum_processor.clear_background()

    def start(
        self,
        channel_id: str = "P22",
        interval_ms: int = DEFAULT_INTERVAL_MS,
        integration: int = DEFAULT_INTEGRATION_US,
    ) -> dict[str, Any]:
        with self.lock:
            # Start is intentionally idempotent. A duplicate UI/API request must
            # not wait on the live worker or turn a healthy session into an
            # apparent stopping error.
            worker_alive = bool(self.thread is not None and self.thread.is_alive())
            if self.desired_active and worker_alive:
                self.last_operation_status = "already_running"
                return {
                    "ok": True,
                    "operation_status": "already_running",
                    **self.status(),
                }
            if self._start_in_progress:
                return {
                    "ok": False,
                    "operation_status": "start_in_progress",
                    **self.status(),
                }
            if self.desired_active and not worker_alive:
                self.desired_active = False
                self.lifecycle_status = "worker_missing"
                self.last_error = "SDK acquisition worker is not running"
            self._start_in_progress = True
            self._start_cancel_requested = False
            self._start_done_event.clear()
            self.lifecycle_status = "starting"
            self.last_operation_status = "starting"
            previous_thread = self.thread
        result: dict[str, Any]
        try:
            result = self._start_reserved(
                channel_id=channel_id,
                interval_ms=interval_ms,
                integration=integration,
                previous_thread=previous_thread,
            )
        finally:
            with self.lock:
                self._start_in_progress = False
                self._start_cancel_requested = False
                self._start_done_event.set()
        return {**result, **self.status()}

    def _start_reserved(
        self,
        *,
        channel_id: str,
        interval_ms: int,
        integration: int,
        previous_thread: threading.Thread | None,
    ) -> dict[str, Any]:
        if previous_thread is not None and previous_thread.is_alive():
            previous_thread.join(timeout=3.0)
            if previous_thread.is_alive():
                with self.lock:
                    self.last_error = "previous SDK acquisition worker is still stopping"
                    self.lifecycle_status = "stop_timeout"
                    self.last_operation_status = "previous_worker_stop_timeout"
                return {
                    "ok": False,
                    "operation_status": "previous_worker_stop_timeout",
                    **self.status(),
                }
        with self.lock:
            orphan_process = self.process
            if orphan_process is not None and orphan_process.poll() is not None:
                if self.process is orphan_process:
                    self.process = None
                orphan_process = None
        if orphan_process is not None:
            cleanup_error = self._kill_and_reap(orphan_process)
            process_still_alive = orphan_process.poll() is None
            with self.lock:
                if not process_still_alive and self.process is orphan_process:
                    self.process = None
                if cleanup_error or process_still_alive:
                    self.desired_active = False
                    self.lifecycle_status = "orphan_cleanup_failed"
                    self.last_operation_status = "orphan_cleanup_failed"
                    self.last_error = (
                        "previous SDK helper could not be reaped before restart"
                        + (f": {cleanup_error}" if cleanup_error else "")
                    )
                    return {
                        "ok": False,
                        "operation_status": "orphan_cleanup_failed",
                        **self.status(),
                    }
        with self.lock:
            requested_channel = channel_id
            requested_interval = max(20, min(int(interval_ms), 2000))
            requested_integration = max(1, int(integration))
            if self._start_cancel_requested:
                self.lifecycle_status = "stopped"
                self.last_operation_status = "start_cancelled"
                return {
                    "ok": False,
                    "operation_status": "start_cancelled",
                    **self.status(),
                }
            # The session may have been started by another request while an old
            # worker was finishing.
            if self.desired_active and self.thread is not None and self.thread.is_alive():
                self.lifecycle_status = "running"
                self.last_operation_status = "already_running"
                return {
                    "ok": True,
                    "operation_status": "already_running",
                    **self.status(),
                }

            self.channel_id = requested_channel
            self.interval_ms = requested_interval
            self.integration = requested_integration

            helper = self.helper_path
            if not helper.exists():
                self.last_error = f"SDK helper not found: {helper}"
                self.lifecycle_status = "unavailable"
                self.last_operation_status = "helper_unavailable"
                return {
                    "ok": False,
                    "operation_status": "helper_unavailable",
                    **self.status(),
                }

            self.last_error = None
            self.last_stderr = None
            self.last_status = None
            self.last_result = None
            self.last_frame_time = None
            self.wavelength_grid_cache = None
            self.spectrum_processor.reset_session()
            self.frame_count = 0
            self.received_frame_count = 0
            self.rejected_frame_count = 0
            self.restart_count = 0
            self.consecutive_failure_count = 0
            self.retry_backoff_sec = 0.0
            self.stale_session_frame_count = 0
            self.helper_timeout_count = 0
            self.helper_output_overflow_count = 0
            self.invalid_helper_message_count = 0
            self.last_helper_stdout_chars = 0
            self.last_helper_stderr_chars = 0
            self.last_acquisition_duration_ms = None
            self.last_cycle_delay_ms = None
            self.last_exit_code = None
            self.started_at = time.time()
            self.desired_active = True
            self._stop_event.clear()
            self.generation += 1
            generation = self.generation
            worker = threading.Thread(
                target=self._supervisor_loop,
                args=(generation,),
                name=f"bayspec-sdk-session-{generation}",
                daemon=True,
            )
            self.thread = worker
            try:
                worker.start()
            except Exception as exc:
                self.thread = None
                self.desired_active = False
                self.generation += 1
                self.started_at = None
                self.lifecycle_status = "start_failed"
                self.last_operation_status = "start_failed"
                self.last_error = (
                    "SDK acquisition worker failed to start: "
                    f"{type(exc).__name__}: {exc}"
                )
                return {
                    "ok": False,
                    "operation_status": "start_failed",
                    **self.status(),
                }
            self.lifecycle_status = "running"
            self.last_operation_status = "started"
            return {
                "ok": True,
                "operation_status": "started",
                **self.status(),
            }

    def stop(self) -> dict[str, Any]:
        with self.lock:
            start_in_progress = self._start_in_progress
            if start_in_progress:
                self._start_cancel_requested = True
                self.desired_active = False
                self._stop_event.set()
                self.generation += 1
                self.lifecycle_status = "stop_requested"
                self.last_operation_status = "start_cancel_requested"
        if start_in_progress and not self._start_done_event.wait(timeout=5.0):
            with self.lock:
                self.lifecycle_status = "stop_timeout"
                self.last_operation_status = "start_cancel_timeout"
                self.last_error = "SDK start did not acknowledge cancellation before timeout"
            return {
                "ok": False,
                "operation_status": "start_cancel_timeout",
                **self.status(),
            }
        with self.lock:
            worker = self.thread
            worker_alive = bool(worker is not None and worker.is_alive())
            process = self.process
            process_alive = bool(process is not None and process.poll() is None)
            if not worker_alive and not process_alive:
                self.desired_active = False
                self._stop_event.set()
                self.process = None
                self.thread = None
                self.lifecycle_status = "stopped"
                self.last_operation_status = (
                    "start_cancelled" if start_in_progress else "already_stopped"
                )
                return {
                    "ok": True,
                    "operation_status": self.last_operation_status,
                    **self.status(),
                }
            self.desired_active = False
            self._stop_event.set()
            self.generation += 1
            self.lifecycle_status = "stop_requested"
            self.last_operation_status = "stop_requested"
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            except Exception as exc:
                with self.lock:
                    self.last_error = (
                        "SDK helper termination failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        with self.lock:
            worker_alive = bool(worker is not None and worker.is_alive())
            process_alive = bool(process is not None and process.poll() is None)
            if worker_alive or process_alive:
                self.lifecycle_status = "stop_timeout"
                self.last_operation_status = "stop_timeout"
                self.last_error = "SDK acquisition did not stop before timeout"
                return {
                    "ok": False,
                    "operation_status": "stop_timeout",
                    **self.status(),
                }
            if self.thread is worker:
                self.thread = None
            if self.process is process:
                self.process = None
            self.lifecycle_status = "stopped"
            self.last_operation_status = "stopped"
        return {
            "ok": True,
            "operation_status": "stopped",
            **self.status(),
        }

    @staticmethod
    def _kill_and_reap(process: subprocess.Popen[str], timeout: float = 2.0) -> str | None:
        """Kill a helper and wait for its OS resources to be released."""
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=timeout)
            return None
        except Exception as exc:  # pragma: no cover - platform-specific process failure
            return f"{type(exc).__name__}: {exc}"

    def _collect_helper_output(
        self,
        process: subprocess.Popen[str],
        *,
        timeout: float,
    ) -> tuple[str, str, dict[str, Any]]:
        """Drain helper pipes without allowing a faulty process to grow memory."""
        stdout_stream = getattr(process, "stdout", None)
        stderr_stream = getattr(process, "stderr", None)
        if stdout_stream is None or stderr_stream is None:
            # Compatibility path for test doubles. Real helpers always use both
            # redirected pipes and therefore take the bounded streaming path.
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return stdout or "", stderr or "", {
                    "timed_out": False,
                    "overflow_stream": None,
                    "stdout_chars": len(stdout or ""),
                    "stderr_chars": len(stderr or ""),
                    "read_errors": [],
                    "cleanup_error": None,
                }
            except subprocess.TimeoutExpired:
                cleanup_error = self._kill_and_reap(process)
                return "", "", {
                    "timed_out": True,
                    "overflow_stream": None,
                    "stdout_chars": 0,
                    "stderr_chars": 0,
                    "read_errors": [],
                    "cleanup_error": cleanup_error,
                }

        chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
        states: dict[str, dict[str, Any]] = {
            "stdout": {"chars": 0, "stored": 0, "error": None},
            "stderr": {"chars": 0, "stored": 0, "error": None},
        }
        limits = {
            "stdout": self.MAX_HELPER_STDOUT_CHARS,
            "stderr": self.MAX_HELPER_STDERR_CHARS,
        }
        overflow_event = threading.Event()
        overflow_stream: list[str] = []

        def drain(name: str, stream: Any) -> None:
            state = states[name]
            limit = limits[name]
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    state["chars"] += len(chunk)
                    remaining = limit - state["stored"]
                    if remaining > 0:
                        kept = chunk[:remaining]
                        chunks[name].append(kept)
                        state["stored"] += len(kept)
                    if state["chars"] > limit and not overflow_event.is_set():
                        overflow_stream.append(name)
                        overflow_event.set()
            except Exception as exc:  # pipe closure can race with forced shutdown
                state["error"] = f"{type(exc).__name__}: {exc}"

        readers = [
            threading.Thread(
                target=drain,
                args=("stdout", stdout_stream),
                name="bayspec-sdk-stdout-drain",
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=("stderr", stderr_stream),
                name="bayspec-sdk-stderr-drain",
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + timeout
        timed_out = False
        cleanup_error: str | None = None
        while process.poll() is None:
            if overflow_event.wait(timeout=0.02):
                cleanup_error = self._kill_and_reap(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                cleanup_error = self._kill_and_reap(process)
                break
        if process.poll() is not None:
            try:
                process.wait(timeout=1.0)
            except Exception as exc:  # pragma: no cover - process is already observed dead
                cleanup_error = cleanup_error or f"{type(exc).__name__}: {exc}"

        for reader in readers:
            reader.join(timeout=1.0)
        for name, reader, stream in zip(
            ("stdout", "stderr"),
            readers,
            (stdout_stream, stderr_stream),
        ):
            if reader.is_alive():
                try:
                    stream.close()
                except Exception:
                    pass
                reader.join(timeout=0.5)
                if reader.is_alive() and states[name]["error"] is None:
                    states[name]["error"] = "pipe reader did not stop"

        return "".join(chunks["stdout"]), "".join(chunks["stderr"]), {
            "timed_out": timed_out,
            "overflow_stream": overflow_stream[0] if overflow_stream else None,
            "stdout_chars": int(states["stdout"]["chars"]),
            "stderr_chars": int(states["stderr"]["chars"]),
            "read_errors": [
                f"{name}: {state['error']}"
                for name, state in states.items()
                if state["error"]
            ],
            "cleanup_error": cleanup_error,
        }

    def _supervisor_loop(self, generation: int) -> None:
        """Acquire one hardware frame per helper process and restart safely.

        The vendor snapshot call is stable for a single frame but can access-
        violate when called repeatedly on the same SDK object. Process-level
        isolation keeps the backend alive and makes every failed acquisition
        observable instead of leaving a stale process marked as live.
        """
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            while True:
                cycle_started = time.monotonic()
                with self.lock:
                    if (
                        self._stop_event.is_set()
                        or not self.desired_active
                        or generation != self.generation
                    ):
                        return
                    helper = self.helper_path
                    interval_ms = self.interval_ms
                    integration = self.integration
                args = [
                    str(helper),
                    "--interval-ms",
                    "20",
                    "--integration",
                    str(integration),
                    "--frames",
                    "1",
                ]
                process: subprocess.Popen[str] | None = None
                acquisition_failed = False
                frame_ingested = False
                try:
                    process = subprocess.Popen(
                        args,
                        cwd=str(helper.parent),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=creationflags,
                    )
                    with self.lock:
                        self.process = process
                        if (
                            generation != self.generation
                            or not self.desired_active
                            or self._stop_event.is_set()
                        ):
                            return
                    stdout, stderr, output_state = self._collect_helper_output(
                        process,
                        timeout=self.HELPER_TIMEOUT_SEC,
                    )
                    with self.lock:
                        if (
                            generation != self.generation
                            or not self.desired_active
                            or self._stop_event.is_set()
                        ):
                            if self.process is process:
                                self.process = None
                            return
                        self.last_helper_stdout_chars = int(output_state["stdout_chars"])
                        self.last_helper_stderr_chars = int(output_state["stderr_chars"])
                    if output_state["timed_out"]:
                        acquisition_failed = True
                        with self.lock:
                            self.last_exit_code = process.returncode
                            self.last_error = "SDK one-frame helper timed out"
                            self.helper_timeout_count += 1
                            if output_state["cleanup_error"]:
                                self.last_error += f"; cleanup: {output_state['cleanup_error']}"
                            if self.process is process:
                                self.process = None
                    elif output_state["overflow_stream"]:
                        acquisition_failed = True
                        with self.lock:
                            stream_name = output_state["overflow_stream"]
                            self.last_exit_code = process.returncode
                            self.last_error = (
                                f"SDK helper {stream_name} exceeded the safety limit"
                            )
                            self.helper_output_overflow_count += 1
                            if output_state["cleanup_error"]:
                                self.last_error += f"; cleanup: {output_state['cleanup_error']}"
                            if self.process is process:
                                self.process = None
                    elif output_state["read_errors"]:
                        acquisition_failed = True
                        with self.lock:
                            self.last_exit_code = process.returncode
                            self.last_error = (
                                "SDK helper pipe read failed: "
                                + "; ".join(output_state["read_errors"])
                            )
                            if self.process is process:
                                self.process = None
                    if acquisition_failed:
                        with self.lock:
                            self.last_stderr = stderr.strip() or None
                    if acquisition_failed:
                        messages = []
                    else:
                        messages = []
                        invalid_message = False
                        nonempty_lines = [
                            line.strip().lstrip("\ufeff")
                            for line in stdout.splitlines()
                            if line.strip()
                        ]
                        if len(nonempty_lines) > self.MAX_HELPER_MESSAGES:
                            invalid_message = True
                            with self.lock:
                                self.last_error = (
                                    "SDK helper returned too many messages: "
                                    f"{len(nonempty_lines)}"
                                )
                        else:
                            for line in nonempty_lines:
                                try:
                                    message = json.loads(line)
                                except json.JSONDecodeError as exc:
                                    invalid_message = True
                                    with self.lock:
                                        self.last_error = f"invalid SDK JSON: {exc}"
                                    break
                                if not isinstance(message, dict):
                                    invalid_message = True
                                    with self.lock:
                                        self.last_error = "SDK message must be a JSON object"
                                    break
                                messages.append(message)
                        spectrum_count = sum(
                            message.get("type") == "spectrum" for message in messages
                        )
                        error_message = next(
                            (
                                message
                                for message in messages
                                if message.get("type") == "error"
                            ),
                            None,
                        )
                        if not invalid_message and error_message is not None:
                            invalid_message = True
                            with self.lock:
                                self.last_error = str(
                                    error_message.get("message") or error_message
                                )
                        elif not invalid_message and spectrum_count != 1:
                            invalid_message = True
                            with self.lock:
                                self.last_error = (
                                    "SDK one-frame helper returned "
                                    f"{spectrum_count} spectrum messages"
                                )
                        if invalid_message:
                            acquisition_failed = True
                            messages = []
                            with self.lock:
                                self.invalid_helper_message_count += 1
                    for message in messages:
                        try:
                            frame_ingested = (
                                self._handle_message(message, generation=generation)
                                or frame_ingested
                            )
                        except Exception as exc:  # keep the acquisition supervisor alive
                            acquisition_failed = True
                            with self.lock:
                                self.last_error = (
                                    "SDK frame processing failed: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                    with self.lock:
                        self.last_exit_code = process.returncode
                        self.last_stderr = stderr.strip() or None
                        self.restart_count += 1
                        if process.returncode not in (0, None):
                            acquisition_failed = True
                            exit_error = (
                                "SDK one-frame helper exited with code "
                                f"{process.returncode}"
                            )
                            self.last_error = (
                                f"{self.last_error}; {exit_error}"
                                if self.last_error
                                else exit_error
                            )
                        elif acquisition_failed:
                            pass
                        elif not messages:
                            acquisition_failed = True
                            if self.last_error is None:
                                self.last_error = "SDK one-frame helper returned no spectrum"
                        elif not frame_ingested:
                            acquisition_failed = True
                            if self.last_error is None:
                                self.last_error = "SDK spectrum was not accepted"
                        if self.process is process:
                            self.process = None
                except OSError as exc:
                    acquisition_failed = True
                    with self.lock:
                        if self.process is process:
                            self.process = None
                        self.last_error = f"failed to start SDK helper: {exc}"
                        self.restart_count += 1
                except Exception as exc:  # defensive guard around vendor/helper output
                    acquisition_failed = True
                    if process is not None and process.poll() is None:
                        try:
                            process.kill()
                            process.wait(timeout=2.0)
                        except Exception:
                            pass
                    with self.lock:
                        if self.process is process:
                            self.process = None
                        self.last_error = (
                            "unexpected SDK acquisition failure: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self.restart_count += 1

                with self.lock:
                    if generation != self.generation or not self.desired_active:
                        return
                    if acquisition_failed:
                        self.consecutive_failure_count += 1
                        base_backoff = max(0.25, interval_ms / 1000.0)
                        self.retry_backoff_sec = min(
                            2.0,
                            base_backoff
                            * (2 ** min(self.consecutive_failure_count - 1, 3)),
                        )
                    else:
                        self.consecutive_failure_count = 0
                        self.retry_backoff_sec = 0.0
                    elapsed_sec = max(0.0, time.monotonic() - cycle_started)
                    self.last_acquisition_duration_ms = elapsed_sec * 1000.0
                    if acquisition_failed:
                        delay_sec = max(0.02, self.retry_backoff_sec)
                    else:
                        # The helper is process-isolated for vendor SDK
                        # stability. If starting, acquiring, and reaping that
                        # process already consumed the requested interval,
                        # begin the next isolated acquisition immediately
                        # instead of adding another fixed 20 ms idle period.
                        delay_sec = max(0.0, interval_ms / 1000.0 - elapsed_sec)
                    self.last_cycle_delay_ms = delay_sec * 1000.0
                if self._stop_event.wait(delay_sec):
                    return
        finally:
            orphan_process: subprocess.Popen[str] | None = None
            with self.lock:
                current = threading.current_thread()
                if self.thread is current:
                    self.thread = None
                    if self.desired_active and generation == self.generation:
                        self.desired_active = False
                        self.lifecycle_status = "worker_exited"
                        self.last_operation_status = "worker_exited"
                        if self.last_error is None:
                            self.last_error = "SDK acquisition worker exited unexpectedly"
                    elif not self.desired_active:
                        self.lifecycle_status = (
                            "stopped_after_timeout"
                            if self.lifecycle_status == "stop_timeout"
                            else "stopped"
                        )
                if self.process is not None:
                    if self.process.poll() is None:
                        orphan_process = self.process
                    self.process = None
            if orphan_process is not None:
                try:
                    orphan_process.terminate()
                    try:
                        orphan_process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        orphan_process.kill()
                        orphan_process.wait(timeout=2.0)
                except Exception as exc:
                    with self.lock:
                        cleanup_error = (
                            "SDK orphan helper cleanup failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self.last_error = (
                            f"{self.last_error}; {cleanup_error}"
                            if self.last_error
                            else cleanup_error
                        )

    def _stdout_loop(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                with self.lock:
                    self.last_error = f"invalid SDK JSON: {exc}"
                continue
            self._handle_message(message)
        with self.lock:
            if self.process is process:
                code = process.poll()
                if code not in (None, 0):
                    self.last_error = f"SDK helper exited with code {code}"

    def _stderr_loop(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            line = line.strip()
            if line:
                with self.lock:
                    self.last_stderr = line

    def _wavelength_grid(self, expected_points: int) -> list[float] | None:
        if self.wavelength_grid_cache and len(self.wavelength_grid_cache) == expected_points:
            return self.wavelength_grid_cache
        try:
            grid = self.bridge._latest_wavelength_grid(  # noqa: SLF001 - bridge owns the Sense grid parser.
                configured_sense_export_root(),
                expected_points=expected_points,
            )
        except Exception:
            grid = None
        if grid and len(grid) == expected_points:
            self.wavelength_grid_cache = grid
            return grid
        return None

    def _handle_message(
        self,
        message: dict[str, Any],
        *,
        generation: int | None = None,
    ) -> bool:
        message_type = message.get("type")
        with self.lock:
            if generation is not None and (
                generation != self.generation
                or not self.desired_active
                or self._stop_event.is_set()
            ):
                if message_type == "spectrum":
                    self.stale_session_frame_count += 1
                return False
        if message_type == "status":
            with self.lock:
                self.last_status = message
            return False
        if message_type == "error":
            with self.lock:
                self.last_error = str(message.get("message") or message)
            return False
        if message_type != "spectrum":
            return False

        counts = message.get("counts")
        if not isinstance(counts, list) or not counts:
            with self.lock:
                self.last_error = "SDK spectrum frame has no counts"
            return False
        if len(counts) > self.MAX_SPECTRUM_POINTS:
            with self.lock:
                self.last_error = (
                    "SDK spectrum frame exceeds the point limit: "
                    f"{len(counts)} > {self.MAX_SPECTRUM_POINTS}"
                )
            return False
        try:
            intensity = [float(value) for value in counts]
        except (TypeError, ValueError):
            with self.lock:
                self.last_error = "SDK spectrum counts are not numeric"
            return False
        if not all(math.isfinite(value) for value in intensity):
            with self.lock:
                self.last_error = "SDK spectrum counts must be finite"
            return False

        wavelength_grid = self._wavelength_grid(len(intensity))
        processed = self.spectrum_processor.process(intensity)
        channel_payload: dict[str, Any] = {
            "channel_id": self.channel_id,
            "intensity": intensity,
            "display_intensity": processed["display_intensity"],
            "overlay_intensity": processed["overlay_intensity"],
            "spectrum_processing": processed["spectrum_processing"],
            "model_input_source": "raw_intensity",
            "spectrum_points": len(intensity),
            "sdk_frame_index": message.get("frame_index"),
            "sdk_timestamp_ms": message.get("timestamp_ms"),
            "sdk_snapshot_result": message.get("snapshot_result"),
            "integration_ms": self.integration / 1000.0,
        }
        if wavelength_grid:
            channel_payload["wavelength_nm"] = wavelength_grid
            channel_payload["peak_axis_type"] = "wavelength_nm"
            channel_payload["spectrum_x_unit"] = "wavelength_nm"
        else:
            peak_index = int(message.get("peak_index") or 0)
            peak_index = max(0, min(peak_index, len(intensity) - 1))
            channel_payload["intensity_counts"] = intensity[peak_index]
            channel_payload["peak_pixel_index"] = peak_index
            channel_payload["peak_axis_type"] = "pixel_index"
            channel_payload["spectrum_x_unit"] = "pixel_index"
            channel_payload["peak_selection_method"] = "sdk_global_pixel_peak_no_wavelength_grid"

        # Keep the final session check and bridge commit under the same reader
        # lock. Otherwise stop/reset can invalidate the session after the check
        # but before bridge.ingest(), allowing one stale frame to overwrite a
        # freshly reset baseline or source buffer.
        with self.lock:
            if generation is not None and (
                generation != self.generation
                or not self.desired_active
                or self._stop_event.is_set()
            ):
                self.stale_session_frame_count += 1
                return False
            result = self.bridge.ingest(
                {
                    "source": "bayspec_direct_usb20bs_sdk",
                    "device_id": configured_device_id(),
                    "channels": [channel_payload],
                }
            )
            self.received_frame_count += 1
            self.last_result = result
            if result.get("ok"):
                self.frame_count += 1
                self.last_frame_time = time.time()
                self.last_error = None
            else:
                self.rejected_frame_count += 1
                self.last_error = str(result.get("reason"))
            return bool(result.get("ok"))
