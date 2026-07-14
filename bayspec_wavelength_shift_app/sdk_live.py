"""Direct BaySpec USB20BS SDK live reader.

The vendor USB DLL is 32-bit and exposes C++ instance methods, so the 64-bit
Python backend cannot load it directly. A small x86 helper process reads frames
from the DLL and streams JSON lines to this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from bridge import DEFAULT_DEVICE_ID, DEFAULT_SENSE_ROOT


class BaySpecSdkLiveReader:
    def __init__(self, bridge: Any, app_root: Path) -> None:
        self.bridge = bridge
        self.app_root = Path(app_root)
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.channel_id = "P22"
        self.interval_ms = 100
        self.integration = 40000
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

    @property
    def helper_path(self) -> Path:
        return self.app_root / "sdk_probe" / "BaySpecSdkStream.exe"

    def status(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            now = time.time()
            age = now - self.last_frame_time if self.last_frame_time is not None else None
            if not self.desired_active:
                freshness = "stopped"
            elif self.last_frame_time is None:
                freshness = "waiting_for_sdk_frame"
            elif age is not None and age <= max(3.0, self.interval_ms / 1000.0 * 12):
                freshness = "live"
            else:
                freshness = "stale"
            return {
                "active": self.desired_active,
                "acquisition_session_id": self.generation,
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
                "last_exit_code": self.last_exit_code,
                "last_status": self.last_status,
                "last_error": self.last_error,
                "last_stderr": self.last_stderr,
                "last_result": self.last_result,
            }

    def start(self, channel_id: str = "P22", interval_ms: int = 100, integration: int = 40000) -> dict[str, Any]:
        with self.lock:
            previous_thread = self.thread
        if previous_thread is not None and previous_thread.is_alive():
            previous_thread.join(timeout=3.0)
            if previous_thread.is_alive():
                with self.lock:
                    self.last_error = "previous SDK acquisition worker is still stopping"
                return self.status()
        with self.lock:
            requested_channel = channel_id
            requested_interval = max(20, min(int(interval_ms), 2000))
            requested_integration = max(1, int(integration))
            if self.desired_active:
                return self.status()

            self.channel_id = requested_channel
            self.interval_ms = requested_interval
            self.integration = requested_integration

            helper = self.helper_path
            if not helper.exists():
                self.last_error = f"SDK helper not found: {helper}"
                return self.status()

            self.last_error = None
            self.last_stderr = None
            self.last_status = None
            self.last_result = None
            self.last_frame_time = None
            self.wavelength_grid_cache = None
            self.frame_count = 0
            self.received_frame_count = 0
            self.rejected_frame_count = 0
            self.restart_count = 0
            self.last_exit_code = None
            self.started_at = time.time()
            self.desired_active = True
            self.generation += 1
            generation = self.generation
            self.thread = threading.Thread(
                target=self._supervisor_loop,
                args=(generation,),
                daemon=True,
            )
            self.thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.desired_active = False
            self.generation += 1
            process = self.process
            worker = self.thread
            self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        with self.lock:
            if self.thread is worker and (worker is None or not worker.is_alive()):
                self.thread = None
        return self.status()

    def _supervisor_loop(self, generation: int) -> None:
        """Acquire one hardware frame per helper process and restart safely.

        The vendor snapshot call is stable for a single frame but can access-
        violate when called repeatedly on the same SDK object. Process-level
        isolation keeps the backend alive and makes every failed acquisition
        observable instead of leaving a stale process marked as live.
        """
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while True:
            with self.lock:
                if not self.desired_active or generation != self.generation:
                    self.process = None
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
                    if generation != self.generation or not self.desired_active:
                        process.terminate()
                        return
                    self.process = process
                stdout, stderr = process.communicate(timeout=12.0)
                with self.lock:
                    if generation != self.generation or not self.desired_active:
                        if self.process is process:
                            self.process = None
                        return
                messages = []
                for line in stdout.splitlines():
                    line = line.strip().lstrip("\ufeff")
                    if not line:
                        continue
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        with self.lock:
                            self.last_error = f"invalid SDK JSON: {exc}"
                for message in messages:
                    self._handle_message(message)
                with self.lock:
                    self.last_exit_code = process.returncode
                    self.last_stderr = stderr.strip() or None
                    self.restart_count += 1
                    if process.returncode not in (0, None):
                        self.last_error = f"SDK one-frame helper exited with code {process.returncode}"
                    elif not any(message.get("type") == "spectrum" for message in messages):
                        self.last_error = "SDK one-frame helper returned no spectrum"
                    if self.process is process:
                        self.process = None
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
                with self.lock:
                    self.last_exit_code = process.returncode
                    self.last_error = "SDK one-frame helper timed out"
                    self.restart_count += 1
                    if self.process is process:
                        self.process = None
            except OSError as exc:
                with self.lock:
                    self.process = None
                    self.last_error = f"failed to start SDK helper: {exc}"
                    self.restart_count += 1
            time.sleep(max(0.02, interval_ms / 1000.0))

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
                DEFAULT_SENSE_ROOT / "Spectrum_Data",
                expected_points=expected_points,
            )
        except Exception:
            grid = None
        if grid and len(grid) == expected_points:
            self.wavelength_grid_cache = grid
            return grid
        return None

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "status":
            with self.lock:
                self.last_status = message
            return
        if message_type == "error":
            with self.lock:
                self.last_error = str(message.get("message") or message)
            return
        if message_type != "spectrum":
            return

        counts = message.get("counts")
        if not isinstance(counts, list) or not counts:
            with self.lock:
                self.last_error = "SDK spectrum frame has no counts"
            return
        try:
            intensity = [float(value) for value in counts]
        except (TypeError, ValueError):
            with self.lock:
                self.last_error = "SDK spectrum counts are not numeric"
            return

        wavelength_grid = self._wavelength_grid(len(intensity))
        channel_payload: dict[str, Any] = {
            "channel_id": self.channel_id,
            "intensity": intensity,
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

        result = self.bridge.ingest(
            {
                "source": "bayspec_direct_usb20bs_sdk",
                "device_id": DEFAULT_DEVICE_ID,
                "channels": [channel_payload],
            }
        )
        with self.lock:
            self.received_frame_count += 1
            self.last_result = result
            if result.get("ok"):
                self.frame_count += 1
                self.last_frame_time = time.time()
                self.last_error = None
            else:
                self.rejected_frame_count += 1
                self.last_error = str(result.get("reason"))
