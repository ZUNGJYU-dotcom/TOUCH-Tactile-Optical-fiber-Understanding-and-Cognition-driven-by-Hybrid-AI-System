"""PX6D six-axis reference-force reader for synchronized TOUCH acquisition.

The reader uses the vendor USB/RS485 frame contract without issuing hardware
calibration commands.  Tare is implemented entirely in software so raw sensor
values remain available for traceability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import multiprocessing as mp
import queue
import statistics
import struct
import threading
import time
from typing import Any, Iterable

from process_lifecycle import (
    attach_process_to_kill_on_close_job as _attach_process_to_kill_on_close_job,
    close_windows_handle as _close_windows_handle,
)

try:
    import serial
    from serial.tools import list_ports as serial_list_ports
except Exception:  # pragma: no cover - exposed through status()
    serial = None
    serial_list_ports = None


AXIS_NAMES = ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm")
FRAME_HEADER = b"\xAA\x55"
FRAME_LENGTH = 29
COMMAND_READ_FRAME = 0x05
COMMAND_VERSION = 0x07
RESPONSE_DATA = 0x03


def _optional_usb_identifier(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return int(value.strip(), 0)
    return int(value)


def _serial_port_identity(port_info: object) -> dict[str, Any]:
    return {
        "port": str(getattr(port_info, "device", "") or ""),
        "description": str(getattr(port_info, "description", "") or ""),
        "hwid": str(getattr(port_info, "hwid", "") or ""),
        "vid": getattr(port_info, "vid", None),
        "pid": getattr(port_info, "pid", None),
        "serial_number": str(
            getattr(port_info, "serial_number", "") or ""
        ),
        "manufacturer": str(getattr(port_info, "manufacturer", "") or ""),
    }


def _close_serial_port(port: Any) -> None:
    """Best-effort close that also asks Windows serial reads to cancel."""

    if port is None:
        return
    for method_name in ("cancel_read", "cancel_write"):
        method = getattr(port, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
    try:
        port.close()
    except Exception:
        pass


def crc8(data: bytes | bytearray | Iterable[int], initial: int = 0) -> int:
    """Return the PX6D CRC-8 (polynomial 0x07, initial value 0)."""

    value = int(initial) & 0xFF
    for item in data:
        value ^= int(item) & 0xFF
        for _ in range(8):
            if value & 0x80:
                value = ((value << 1) ^ 0x07) & 0xFF
            else:
                value = (value << 1) & 0xFF
    return value


def command_packet(command: int, argument: int, device_id: int = 0x7F) -> bytes:
    body = bytes(
        [0xAA, 0x55, int(device_id) & 0x7F, int(command) & 0xFF, int(argument) & 0xFF]
    )
    return body + bytes([crc8(body)])


def parse_data_frame(frame: bytes) -> tuple[float, float, float, float, float, float]:
    if len(frame) != FRAME_LENGTH:
        raise ValueError(f"PX6D frame length must be {FRAME_LENGTH}, got {len(frame)}")
    if frame[:2] != FRAME_HEADER:
        raise ValueError("PX6D frame header mismatch")
    if frame[3] != RESPONSE_DATA:
        raise ValueError(f"PX6D response command must be 0x03, got 0x{frame[3]:02X}")
    if crc8(frame[:-1]) != frame[-1]:
        raise ValueError("PX6D CRC mismatch")
    values = struct.unpack("<6f", frame[4:28])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("PX6D frame contains a non-finite axis value")
    return values


def _read_serial_packet(
    port: Any,
    *,
    minimum_length: int,
    timeout_sec: float,
) -> bytes | None:
    buffer = bytearray()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        waiting = int(getattr(port, "in_waiting", 0) or 0)
        chunk = port.read(max(1, min(256, waiting or minimum_length)))
        if chunk:
            buffer.extend(chunk)
        start = buffer.find(FRAME_HEADER)
        if start < 0:
            if len(buffer) > 2:
                del buffer[:-1]
            continue
        if start > 0:
            del buffer[:start]
        if len(buffer) >= minimum_length:
            return bytes(buffer[:minimum_length])
    return None


def _emit_serial_worker_message(output_queue: Any, payload: dict[str, Any]) -> None:
    """Keep the newest worker evidence without allowing queue backpressure."""

    try:
        output_queue.put_nowait(payload)
        return
    except queue.Full:
        pass
    except Exception:
        return
    try:
        output_queue.get_nowait()
    except Exception:
        pass
    try:
        output_queue.put_nowait(payload)
    except Exception:
        pass


def _px6d_serial_worker(
    config: dict[str, Any],
    output_queue: Any,
    stop_event: Any,
) -> None:
    """Own the native serial handle in a disposable process.

    Some Windows USB serial drivers can leave a native read blocked after a
    cable or device reset. Keeping that handle outside the API process lets the
    parent terminate and recreate this worker without freezing TOUCH.
    """

    port = None
    try:
        if serial is None:
            raise RuntimeError("pyserial_not_installed")
        _emit_serial_worker_message(
            output_queue,
            {"kind": "connecting", "timestamp_epoch_sec": time.time()},
        )
        port = serial.Serial(
            str(config["port"]),
            int(config["baud_rate"]),
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.01,
            write_timeout=0.25,
        )
        port.reset_input_buffer()
        port.reset_output_buffer()
        stop_event.wait(float(config["device_settle_sec"]))
        if stop_event.is_set():
            return

        firmware: str | None = None
        handshake_attempts = max(1, int(config["handshake_attempts"]))
        for attempt in range(handshake_attempts):
            if stop_event.is_set():
                return
            port.reset_input_buffer()
            port.write(
                command_packet(
                    COMMAND_VERSION,
                    0x01,
                    int(config["device_id"]),
                )
            )
            port.flush()
            raw = _read_serial_packet(
                port,
                minimum_length=13,
                timeout_sec=float(config["handshake_timeout_sec"]),
            )
            if (
                raw is not None
                and len(raw) >= 13
                and raw[3] == COMMAND_VERSION
                and crc8(raw[:-1]) == raw[-1]
            ):
                firmware = (
                    raw[5:12].decode("ascii", errors="ignore").rstrip("\x00")
                    or None
                )
                break
            if attempt + 1 < handshake_attempts:
                stop_event.wait(0.05)
        if firmware is None:
            raise TimeoutError("PX6D firmware handshake returned no valid response")

        _emit_serial_worker_message(
            output_queue,
            {
                "kind": "connected",
                "firmware_version": firmware,
                "timestamp_epoch_sec": time.time(),
            },
        )
        poll_hz = max(1.0, min(100.0, float(config["poll_hz"])))
        poll_interval = 1.0 / poll_hz
        while not stop_event.is_set():
            loop_started = time.monotonic()
            request_started = time.perf_counter()
            port.write(
                command_packet(
                    COMMAND_READ_FRAME,
                    0x01,
                    int(config["device_id"]),
                )
            )
            port.flush()
            frame = _read_serial_packet(
                port,
                minimum_length=FRAME_LENGTH,
                timeout_sec=float(config["read_timeout_sec"]),
            )
            if frame is None:
                raise TimeoutError("PX6D frame timeout")
            try:
                values = parse_data_frame(frame)
            except Exception as exc:
                _emit_serial_worker_message(
                    output_queue,
                    {
                        "kind": "invalid_frame",
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp_epoch_sec": time.time(),
                    },
                )
                continue
            _emit_serial_worker_message(
                output_queue,
                {
                    "kind": "sample",
                    "timestamp_epoch_sec": time.time(),
                    "timestamp_monotonic_sec": time.monotonic(),
                    "values": list(values),
                    "roundtrip_ms": (
                        time.perf_counter() - request_started
                    )
                    * 1000.0,
                },
            )
            remaining = poll_interval - (time.monotonic() - loop_started)
            if remaining > 0:
                stop_event.wait(remaining)
    except BaseException as exc:
        if not stop_event.is_set():
            _emit_serial_worker_message(
                output_queue,
                {
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp_epoch_sec": time.time(),
                },
            )
    finally:
        _close_serial_port(port)
        _emit_serial_worker_message(
            output_queue,
            {"kind": "stopped", "timestamp_epoch_sec": time.time()},
        )


@dataclass(frozen=True)
class Px6dSample:
    sequence_id: int
    timestamp_epoch_sec: float
    timestamp_monotonic_sec: float
    fx_n: float
    fy_n: float
    fz_n: float
    mx_nm: float
    my_nm: float
    mz_nm: float
    roundtrip_ms: float

    def raw_values(self) -> tuple[float, float, float, float, float, float]:
        return tuple(float(getattr(self, name)) for name in AXIS_NAMES)


@dataclass(frozen=True)
class ConditionedFzSample:
    """Per-sample Fz conditioning result retained for timestamp alignment."""

    sequence_id: int
    median_reference_fz_n: float
    low_pass_reference_fz_n: float
    drift_offset_n: float
    drift_corrected_reference_fz_n: float
    conditioned_reference_fz_n: float
    stationary_detected: bool
    auto_zero_drift_active: bool
    filter_status: str


@dataclass(frozen=True)
class FilteredAxesSample:
    """Median-despiked and low-pass software-zeroed six-axis values."""

    sequence_id: int
    median_zeroed_values: tuple[float, ...]
    filtered_zeroed_values: tuple[float, ...]


class Px6dReader:
    """Background polling reader with software tare and timestamp alignment."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        payload = dict(config or {})
        self.configured_port = str(payload.get("port") or "COM3")
        self.port = self.configured_port
        self.auto_detect_port = bool(payload.get("auto_detect_port", True))
        self.usb_vid = _optional_usb_identifier(payload.get("usb_vid"))
        self.usb_pid = _optional_usb_identifier(payload.get("usb_pid"))
        self.usb_serial_number = str(
            payload.get("usb_serial_number")
            or payload.get("serial_number")
            or ""
        ).strip()
        self.port_description_contains = str(
            payload.get("port_description_contains")
            or payload.get("description_contains")
            or ""
        ).strip()
        self.baud_rate = int(payload.get("baud_rate") or 921600)
        self.device_id = int(payload.get("device_id") or 0x7F)
        self.poll_hz = max(1.0, min(100.0, float(payload.get("poll_hz") or 50.0)))
        self.read_timeout_sec = max(0.02, float(payload.get("read_timeout_sec") or 0.20))
        self.reconnect_interval_sec = max(
            0.1, float(payload.get("reconnect_interval_sec") or 1.0)
        )
        self.reconnect_max_interval_sec = max(
            self.reconnect_interval_sec,
            float(payload.get("reconnect_max_interval_sec") or 8.0),
        )
        self.reconnect_backoff_multiplier = max(
            1.0, float(payload.get("reconnect_backoff_multiplier") or 2.0)
        )
        self.port_busy_backoff_sec = min(
            self.reconnect_max_interval_sec,
            max(
                self.reconnect_interval_sec,
                float(payload.get("port_busy_backoff_sec") or 5.0),
            ),
        )
        self.isolate_serial_process = bool(payload.get("isolate_process", False))
        self.worker_watchdog_sec = max(
            self.read_timeout_sec + 0.30,
            float(payload.get("worker_watchdog_sec") or 1.5),
        )
        self.device_settle_sec = max(
            0.05, float(payload.get("device_settle_sec") or 0.20)
        )
        self.worker_stop_timeout_sec = max(
            0.10, float(payload.get("worker_stop_timeout_sec") or 0.50)
        )
        self.handshake_timeout_sec = max(
            0.10, float(payload.get("handshake_timeout_sec") or 0.40)
        )
        self.handshake_attempts = max(
            1, min(5, int(payload.get("handshake_attempts") or 3))
        )
        self.history_seconds = max(10.0, float(payload.get("history_seconds") or 300.0))
        self.compression_sign = -1.0 if float(payload.get("compression_sign") or -1.0) < 0 else 1.0
        self.filter_alpha = min(1.0, max(0.01, float(payload.get("filter_alpha") or 0.25)))
        median_window = max(1, int(payload.get("median_window_samples") or 5))
        self.median_window_samples = median_window if median_window % 2 else median_window + 1
        self.force_deadband_n = max(0.0, float(payload.get("force_deadband_n") or 0.015))
        self.stationary_window_sec = max(
            0.25, float(payload.get("stationary_window_sec") or 1.0)
        )
        self.stationary_std_max_n = max(
            0.0001, float(payload.get("stationary_std_max_n") or 0.008)
        )
        self.stationary_range_max_n = max(
            self.stationary_std_max_n,
            float(payload.get("stationary_range_max_n") or 0.030),
        )
        self.stationary_slope_max_n_per_sec = max(
            0.0001,
            float(payload.get("stationary_slope_max_n_per_sec") or 0.025),
        )
        self.auto_zero_drift_enabled = bool(
            payload.get("auto_zero_drift_enabled", True)
        )
        self.auto_zero_hold_sec = max(
            0.25, float(payload.get("auto_zero_hold_sec") or 1.5)
        )
        self.auto_zero_capture_limit_n = max(
            self.force_deadband_n,
            float(payload.get("auto_zero_capture_limit_n") or 0.060),
        )
        self.auto_zero_release_reacquire_limit_n = max(
            self.auto_zero_capture_limit_n,
            float(payload.get("auto_zero_release_reacquire_limit_n") or 0.30),
        )
        self.auto_zero_alpha = min(
            0.25, max(0.0001, float(payload.get("auto_zero_alpha") or 0.015))
        )
        self.maximum_drift_offset_n = max(
            self.auto_zero_capture_limit_n,
            float(payload.get("maximum_drift_offset_n") or 0.50),
        )
        self.auto_tare_on_start = bool(payload.get("auto_tare_on_start", False))
        self.auto_tare_duration_sec = max(
            0.25, float(payload.get("auto_tare_duration_sec") or 1.0)
        )
        self.auto_tare_max_std_n = max(
            0.01, float(payload.get("auto_tare_max_std_n") or 0.12)
        )
        self.sync_window_sec = max(0.01, float(payload.get("sync_window_sec") or 0.25))
        self.sync_max_age_sec = max(0.05, float(payload.get("sync_max_age_sec") or 1.0))
        self.force_full_scale_per_axis_n = max(
            0.01, float(payload.get("force_full_scale_per_axis_n") or 50.0)
        )
        self.moment_full_scale_per_axis_nm = max(
            0.001, float(payload.get("moment_full_scale_per_axis_nm") or 2.0)
        )
        self.warning_utilization_percent = min(
            100.0, max(1.0, float(payload.get("warning_utilization_percent") or 90.0))
        )
        self.sync_excellent_max_offset_ms = max(
            0.0, float(payload.get("sync_excellent_max_offset_ms") or 50.0)
        )
        self.sync_good_max_offset_ms = max(
            self.sync_excellent_max_offset_ms,
            float(payload.get("sync_good_max_offset_ms") or 150.0),
        )
        self.sync_acceptable_max_offset_ms = max(
            self.sync_good_max_offset_ms,
            float(payload.get("sync_acceptable_max_offset_ms") or 250.0),
        )

        history_limit = max(500, int(self.poll_hz * self.history_seconds))
        self._samples: deque[Px6dSample] = deque(maxlen=history_limit)
        self._conditioned_samples: dict[int, ConditionedFzSample] = {}
        self._filtered_axis_samples: dict[int, FilteredAxesSample] = {}
        self._axis_median_windows: dict[str, deque[float]] = {
            axis: deque(maxlen=self.median_window_samples) for axis in AXIS_NAMES
        }
        self._filtered_zeroed_axes: dict[str, float | None] = {
            axis: None for axis in AXIS_NAMES
        }
        self._median_reference_window: deque[float] = deque(
            maxlen=self.median_window_samples
        )
        self._stationary_window: deque[tuple[float, float]] = deque()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial = None
        self._connecting_serial = None
        self._serial_worker_process = None
        self._serial_worker_queue = None
        self._serial_worker_stop_event = None
        self._serial_worker_pid: int | None = None
        self._serial_worker_job_handle: int | None = None
        self._serial_worker_restarts = 0
        self._serial_worker_forced_terminations = 0
        self._serial_worker_last_message_monotonic: float | None = None
        self._running = False
        self._connected = False
        self._lifecycle_status = "idle"
        self._firmware_version: str | None = None
        self._last_error: str | None = None
        self._sequence = 0
        self._valid_frames = 0
        self._invalid_frames = 0
        self._connection_attempts = 0
        self._consecutive_connection_failures = 0
        self._current_reconnect_delay_sec = 0.0
        self._next_reconnect_epoch_sec: float | None = None
        self._port_busy_detected = False
        self._connection_error_kind: str | None = None
        self._active_port: str | None = self.configured_port
        self._port_detection_status = "not_checked"
        self._detected_device_identity: dict[str, Any] | None = None
        self._available_serial_ports: list[dict[str, Any]] = []
        self._port_scan_count = 0
        self._last_port_scan_epoch_sec: float | None = None
        self._started_at_epoch_sec: float | None = None
        self._tare_values: tuple[float, float, float, float, float, float] | None = None
        self._tare_timestamp_epoch_sec: float | None = None
        self._tare_status = "required"
        self._tare_fz_std_n: float | None = None
        self._tare_sample_count = 0
        self._filtered_reference_fz_n: float | None = None
        self._drift_offset_n = 0.0
        self._stationary_since_epoch_sec: float | None = None
        self._stationary_detected = False
        self._auto_zero_drift_active = False
        self._filter_status = "tare_required"
        self._auto_zero_frozen = False
        self._auto_zero_freeze_reason: str | None = None

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                stopping = self._stop_event.is_set()
                return {
                    "ok": not stopping,
                    "operation_status": (
                        "stop_in_progress" if stopping else "already_running"
                    ),
                    **self.status(),
                }
            if serial is None:
                self._thread = None
                self._running = False
                self._connected = False
                self._lifecycle_status = "unavailable"
                self._last_error = "pyserial_not_installed"
                return {
                    "ok": False,
                    "operation_status": "dependency_unavailable",
                    **self.status(),
                }
            self._close_serial_locked()
            self._thread = None
            self._stop_event.clear()
            self._clear_reconnect_backoff_locked()
            self._running = True
            self._connected = False
            self._lifecycle_status = "running"
            self._started_at_epoch_sec = time.time()
            worker = threading.Thread(
                target=self._run,
                name="px6d-reader",
                daemon=True,
            )
            self._thread = worker
            try:
                worker.start()
            except Exception as exc:
                self._thread = None
                self._running = False
                self._connected = False
                self._lifecycle_status = "start_failed"
                self._last_error = (
                    f"px6d_reader_start_failed: {type(exc).__name__}: {exc}"
                )
                return {
                    "ok": False,
                    "operation_status": "start_failed",
                    **self.status(),
                }
        return {"ok": True, "operation_status": "started", **self.status()}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = None
                self._running = False
                self._connected = False
                self._lifecycle_status = "stopped"
                self._close_serial_locked()
                return {
                    "ok": True,
                    "operation_status": "already_stopped",
                    **self.status(),
                }
            self._stop_event.set()
            self._lifecycle_status = "stop_requested"
            self._connected = False
            # Closing the handle first interrupts an in-flight serial read on
            # drivers that would otherwise outlive the join timeout.
            self._close_serial_locked()
        self._stop_isolated_worker()
        thread.join(timeout=2.0)
        with self._lock:
            if thread.is_alive():
                self._running = True
                self._connected = False
                self._lifecycle_status = "stop_timeout"
                self._last_error = "px6d_reader_stop_timeout"
                return {
                    "ok": False,
                    "operation_status": "stop_timeout",
                    **self.status(),
                }
            if self._thread is thread:
                self._thread = None
            self._running = False
            self._connected = False
            self._lifecycle_status = "stopped"
            self._close_serial_locked()
        return {"ok": True, "operation_status": "stopped", **self.status()}

    @staticmethod
    def _classify_connection_error(error: object) -> str:
        text = str(error or "").strip().lower()
        if any(
            token in text
            for token in (
                "filenotfounderror",
                "file not found",
                "no such file",
                "cannot find",
                "device_not_detected",
                "port_not_detected",
                "system cannot find",
            )
        ):
            return "port_not_found"
        if any(
            token in text
            for token in (
                "permissionerror",
                "access is denied",
                "access denied",
                "permission denied",
                "拒绝访问",
                "winerror 5",
            )
        ):
            return "port_busy_or_permission_denied"
        if any(
            token in text
            for token in (
                "filenotfounderror",
                "file not found",
                "no such file",
                "cannot find",
                "系统找不到",
            )
        ):
            return "port_not_found"
        if "handshake" in text:
            return "handshake_timeout"
        if "timeout" in text:
            return "device_timeout"
        return "serial_connection_error"

    def _identity_matches(self, identity: dict[str, Any]) -> bool:
        if self.usb_serial_number and (
            str(identity.get("serial_number") or "").casefold()
            != self.usb_serial_number.casefold()
        ):
            return False
        if self.usb_vid is not None and identity.get("vid") != self.usb_vid:
            return False
        if self.usb_pid is not None and identity.get("pid") != self.usb_pid:
            return False
        if self.port_description_contains and (
            self.port_description_contains.casefold()
            not in str(identity.get("description") or "").casefold()
        ):
            return False
        return bool(
            self.usb_serial_number
            or self.usb_vid is not None
            or self.usb_pid is not None
            or self.port_description_contains
        )

    def _resolve_active_port(self) -> str:
        if not self.auto_detect_port:
            with self._lock:
                self._active_port = self.configured_port
                self.port = self.configured_port
                self._port_detection_status = "auto_detection_disabled"
            return self.configured_port
        if serial_list_ports is None:
            with self._lock:
                self._active_port = self.configured_port
                self.port = self.configured_port
                self._port_detection_status = "list_ports_unavailable"
            return self.configured_port

        try:
            identities = [
                _serial_port_identity(port_info)
                for port_info in serial_list_ports.comports()
            ]
        except Exception as exc:
            with self._lock:
                self._active_port = self.configured_port
                self.port = self.configured_port
                self._port_detection_status = "port_scan_failed"
                self._last_error = (
                    f"PX6D serial port scan failed: {type(exc).__name__}: {exc}"
                )
            return self.configured_port

        configured = self.configured_port.casefold()
        identity_configured = bool(
            self.usb_serial_number
            or self.usb_vid is not None
            or self.usb_pid is not None
            or self.port_description_contains
        )
        matching = (
            [item for item in identities if self._identity_matches(item)]
            if identity_configured
            else [
                item
                for item in identities
                if str(item.get("port") or "").casefold() == configured
            ]
        )
        selected = next(
            (
                item
                for item in matching
                if str(item.get("port") or "").casefold() == configured
            ),
            matching[0] if matching else None,
        )
        with self._lock:
            self._port_scan_count += 1
            self._last_port_scan_epoch_sec = time.time()
            self._available_serial_ports = identities
            self._detected_device_identity = dict(selected) if selected else None
            if selected is not None:
                active_port = str(selected["port"])
                self._active_port = active_port
                self.port = active_port
                self._port_detection_status = (
                    "matched_configured_port"
                    if active_port.casefold() == configured
                    else "matched_usb_identity_on_new_port"
                )
                return active_port

            if not identity_configured:
                self._active_port = self.configured_port
                self.port = self.configured_port
                self._port_detection_status = "configured_port_unverified"
                return self.configured_port

            self._active_port = None
            self.port = self.configured_port
            self._port_detection_status = (
                "device_not_detected"
                if not identities
                else "configured_device_identity_not_detected"
            )
        raise FileNotFoundError(
            "PX6D device_not_detected; waiting for USB identity "
            f"VID={self.usb_vid!r}, PID={self.usb_pid!r}, "
            f"serial={self.usb_serial_number!r}"
        )

    def _clear_reconnect_backoff_locked(self) -> None:
        self._consecutive_connection_failures = 0
        self._current_reconnect_delay_sec = 0.0
        self._next_reconnect_epoch_sec = None
        self._port_busy_detected = False
        self._connection_error_kind = None

    def _record_connection_failure(self, error: object) -> float:
        with self._lock:
            return self._record_connection_failure_locked(error)

    def _record_connection_failure_locked(self, error: object) -> float:
        error_text = str(error or "PX6D serial connection error")
        error_kind = self._classify_connection_error(error_text)
        self._consecutive_connection_failures += 1
        base_delay = (
            self.port_busy_backoff_sec
            if error_kind == "port_busy_or_permission_denied"
            else self.reconnect_interval_sec
        )
        exponent = max(0, self._consecutive_connection_failures - 1)
        delay = min(
            self.reconnect_max_interval_sec,
            base_delay * (self.reconnect_backoff_multiplier**exponent),
        )
        self._current_reconnect_delay_sec = float(delay)
        self._next_reconnect_epoch_sec = time.time() + float(delay)
        self._port_busy_detected = error_kind == "port_busy_or_permission_denied"
        self._connection_error_kind = error_kind
        self._connected = False
        if not self._stop_event.is_set():
            self._lifecycle_status = "reconnecting"
            self._last_error = error_text
        return float(delay)

    def _wait_before_reconnect(self) -> None:
        with self._lock:
            delay = max(0.0, float(self._current_reconnect_delay_sec))
            expected_deadline = self._next_reconnect_epoch_sec
        self._stop_event.wait(delay)
        with self._lock:
            if (
                self._next_reconnect_epoch_sec == expected_deadline
                and (
                    expected_deadline is None
                    or time.time() >= expected_deadline
                    or self._stop_event.is_set()
                )
            ):
                self._next_reconnect_epoch_sec = None

    def _run(self) -> None:
        current_worker = threading.current_thread()
        try:
            if serial is None:
                with self._lock:
                    self._lifecycle_status = "unavailable"
                    self._last_error = "pyserial_not_installed"
                return

            if self.isolate_serial_process:
                self._run_isolated_serial()
                return

            auto_tare_pending = self.auto_tare_on_start and self._tare_values is None
            while not self._stop_event.is_set():
                try:
                    self._connect()
                    auto_tare_start = time.monotonic()
                    while not self._stop_event.is_set():
                        loop_started = time.monotonic()
                        sample = self._request_sample()
                        self._append_sample(sample)
                        if (
                            auto_tare_pending
                            and time.monotonic() - auto_tare_start >= self.auto_tare_duration_sec
                        ):
                            result = self.tare(
                                duration_sec=self.auto_tare_duration_sec,
                                max_std_n=self.auto_tare_max_std_n,
                                wait_for_new_samples=False,
                            )
                            if result.get("ok"):
                                auto_tare_pending = False
                            else:
                                auto_tare_start = time.monotonic()
                        remaining = (1.0 / self.poll_hz) - (time.monotonic() - loop_started)
                        if remaining > 0:
                            self._stop_event.wait(remaining)
                except Exception as exc:  # pragma: no cover - hardware path
                    with self._lock:
                        if not self._stop_event.is_set():
                            self._record_connection_failure_locked(
                                f"{type(exc).__name__}: {exc}"
                            )
                        self._close_serial_locked()
                    self._wait_before_reconnect()
        finally:
            self._stop_isolated_worker()
            with self._lock:
                stop_requested = self._stop_event.is_set()
                previous_lifecycle = self._lifecycle_status
                self._running = False
                self._connected = False
                self._close_serial_locked()
                if previous_lifecycle == "unavailable":
                    self._lifecycle_status = "unavailable"
                elif previous_lifecycle == "stop_timeout":
                    self._lifecycle_status = "stopped_after_timeout"
                elif stop_requested:
                    self._lifecycle_status = "stopped"
                else:
                    self._lifecycle_status = "worker_exited"
                    self._last_error = (
                        self._last_error
                        or "PX6D reader worker exited unexpectedly"
                    )
                if self._thread is current_worker:
                    self._thread = None

    def _isolated_worker_config(self, active_port: str) -> dict[str, Any]:
        return {
            "port": active_port,
            "baud_rate": self.baud_rate,
            "device_id": self.device_id,
            "poll_hz": self.poll_hz,
            "read_timeout_sec": self.read_timeout_sec,
            "device_settle_sec": self.device_settle_sec,
            "handshake_timeout_sec": self.handshake_timeout_sec,
            "handshake_attempts": self.handshake_attempts,
        }

    def _start_isolated_worker(self) -> None:
        active_port = self._resolve_active_port()
        context = mp.get_context("spawn")
        output_queue = context.Queue(maxsize=256)
        stop_event = context.Event()
        process = context.Process(
            target=_px6d_serial_worker,
            args=(self._isolated_worker_config(active_port), output_queue, stop_event),
            name="px6d-serial-worker",
            daemon=True,
        )
        with self._lock:
            self._connection_attempts += 1
            self._connected = False
            self._lifecycle_status = "connecting"
            self._serial_worker_queue = output_queue
            self._serial_worker_stop_event = stop_event
            self._serial_worker_process = process
            self._serial_worker_last_message_monotonic = time.monotonic()
        try:
            process.start()
        except BaseException:
            with self._lock:
                if self._serial_worker_process is process:
                    self._serial_worker_process = None
                    self._serial_worker_queue = None
                    self._serial_worker_stop_event = None
                    self._serial_worker_pid = None
            try:
                output_queue.close()
                output_queue.cancel_join_thread()
            except Exception:
                pass
            raise
        job_handle = _attach_process_to_kill_on_close_job(int(process.pid or 0))
        with self._lock:
            self._serial_worker_pid = process.pid
            self._serial_worker_job_handle = job_handle

    def _stop_isolated_worker(self) -> None:
        with self._lock:
            process = self._serial_worker_process
            output_queue = self._serial_worker_queue
            stop_event = self._serial_worker_stop_event
            job_handle = self._serial_worker_job_handle
            self._serial_worker_process = None
            self._serial_worker_queue = None
            self._serial_worker_stop_event = None
            self._serial_worker_pid = None
            self._serial_worker_job_handle = None
        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass
        forced = False
        if process is not None:
            try:
                process.join(timeout=self.worker_stop_timeout_sec)
            except Exception:
                pass
            try:
                alive = process.is_alive()
            except Exception:
                alive = False
            if alive:
                forced = True
                _close_windows_handle(job_handle)
                job_handle = None
                try:
                    process.join(timeout=self.worker_stop_timeout_sec)
                except Exception:
                    pass
                try:
                    alive = process.is_alive()
                except Exception:
                    alive = False
            if alive:
                try:
                    process.terminate()
                except Exception:
                    pass
                try:
                    process.join(timeout=self.worker_stop_timeout_sec)
                except Exception:
                    pass
            try:
                alive = process.is_alive()
            except Exception:
                alive = False
            if alive and hasattr(process, "kill"):
                forced = True
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.join(timeout=self.worker_stop_timeout_sec)
                except Exception:
                    pass
        _close_windows_handle(job_handle)
        if output_queue is not None:
            try:
                output_queue.close()
                output_queue.cancel_join_thread()
            except Exception:
                pass
        if forced:
            with self._lock:
                self._serial_worker_forced_terminations += 1

    def _process_isolated_message(self, message: dict[str, Any]) -> bool:
        kind = str(message.get("kind") or "")
        with self._lock:
            self._serial_worker_last_message_monotonic = time.monotonic()
        if kind == "connecting":
            with self._lock:
                self._connected = False
                self._lifecycle_status = "connecting"
            return False
        if kind == "connected":
            with self._lock:
                self._connected = True
                self._lifecycle_status = "running"
                self._firmware_version = message.get("firmware_version") or None
                self._last_error = None
            return False
        if kind == "invalid_frame":
            with self._lock:
                self._invalid_frames += 1
                self._last_error = str(message.get("error") or "PX6D invalid frame")
            return False
        if kind == "error":
            with self._lock:
                self._record_connection_failure_locked(
                    str(message.get("error") or "PX6D worker error")
                )
            return True
        if kind == "stopped":
            return True
        if kind != "sample":
            return False
        values = tuple(float(value) for value in (message.get("values") or ()))
        if len(values) != len(AXIS_NAMES) or not all(
            math.isfinite(value) for value in values
        ):
            with self._lock:
                self._invalid_frames += 1
                self._last_error = "PX6D worker emitted invalid axis values"
            return False
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        sample = Px6dSample(
            sequence,
            float(message.get("timestamp_epoch_sec") or time.time()),
            float(message.get("timestamp_monotonic_sec") or time.monotonic()),
            *values,
            float(message.get("roundtrip_ms") or 0.0),
        )
        self._append_sample(sample)
        return False

    def _run_isolated_serial(self) -> None:
        auto_tare_pending = self.auto_tare_on_start and self._tare_values is None
        while not self._stop_event.is_set():
            try:
                self._start_isolated_worker()
                auto_tare_start = time.monotonic()
                restart_required = False
                failure_registered = False
                while not self._stop_event.is_set() and not restart_required:
                    with self._lock:
                        output_queue = self._serial_worker_queue
                        process = self._serial_worker_process
                        last_message = self._serial_worker_last_message_monotonic
                    if output_queue is None or process is None:
                        restart_required = True
                        break
                    try:
                        message = output_queue.get(timeout=0.10)
                    except queue.Empty:
                        message = None
                    except (EOFError, OSError, ValueError):
                        restart_required = True
                        message = None
                    if isinstance(message, dict):
                        last_message = time.monotonic()
                        restart_required = self._process_isolated_message(message)
                        failure_registered = (
                            failure_registered
                            or str(message.get("kind") or "") == "error"
                        )
                    if auto_tare_pending and self._connected:
                        if (
                            time.monotonic() - auto_tare_start
                            >= self.auto_tare_duration_sec
                        ):
                            result = self.tare(
                                duration_sec=self.auto_tare_duration_sec,
                                max_std_n=self.auto_tare_max_std_n,
                                wait_for_new_samples=False,
                            )
                            if result.get("ok"):
                                auto_tare_pending = False
                            else:
                                auto_tare_start = time.monotonic()
                    try:
                        process_alive = process.is_alive()
                    except Exception:
                        process_alive = False
                    silent_for = (
                        time.monotonic() - last_message
                        if last_message is not None
                        else math.inf
                    )
                    if not process_alive:
                        restart_required = True
                        with self._lock:
                            if not self._stop_event.is_set():
                                if not failure_registered:
                                    self._record_connection_failure_locked(
                                        self._last_error
                                        or "PX6D serial worker exited unexpectedly"
                                    )
                                    failure_registered = True
                    elif silent_for > self.worker_watchdog_sec:
                        restart_required = True
                        with self._lock:
                            if not failure_registered:
                                self._record_connection_failure_locked(
                                    "PX6D serial worker watchdog timeout "
                                    f"after {silent_for:.2f}s"
                                )
                                failure_registered = True
                self._stop_isolated_worker()
                if not self._stop_event.is_set():
                    with self._lock:
                        self._serial_worker_restarts += 1
                        if not failure_registered:
                            self._record_connection_failure_locked(
                                "PX6D serial worker restart requested"
                            )
                    self._wait_before_reconnect()
            except BaseException as exc:
                self._stop_isolated_worker()
                with self._lock:
                    if not self._stop_event.is_set():
                        self._record_connection_failure_locked(
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._serial_worker_restarts += 1
                self._wait_before_reconnect()

    def _connect(self) -> None:
        active_port = self._resolve_active_port()
        with self._lock:
            self._connection_attempts += 1
        port = serial.Serial(
            active_port,
            self.baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.01,
            write_timeout=0.25,
        )
        with self._lock:
            if self._stop_event.is_set():
                try:
                    port.close()
                finally:
                    raise InterruptedError("PX6D connection cancelled by stop request")
            self._connecting_serial = port
        try:
            port.reset_input_buffer()
            port.reset_output_buffer()
            time.sleep(0.10)
            firmware = self._query_version(port)
        except Exception:
            with self._lock:
                if self._connecting_serial is port:
                    self._connecting_serial = None
            try:
                port.close()
            except Exception:
                pass
            raise
        with self._lock:
            if self._stop_event.is_set():
                if self._connecting_serial is port:
                    self._connecting_serial = None
                try:
                    port.close()
                finally:
                    raise InterruptedError("PX6D connection cancelled by stop request")
            if self._connecting_serial is port:
                self._connecting_serial = None
            self._serial = port
            self._connected = True
            self._lifecycle_status = "running"
            self._firmware_version = firmware
            self._last_error = None

    def _query_version(self, port: Any) -> str | None:
        port.write(command_packet(COMMAND_VERSION, 0x01, self.device_id))
        port.flush()
        raw = self._read_packet(port, minimum_length=13, timeout_sec=0.40)
        if raw is None or len(raw) < 13 or raw[3] != COMMAND_VERSION:
            raise TimeoutError("PX6D firmware handshake returned no valid response")
        if crc8(raw[:-1]) != raw[-1]:
            raise ValueError("PX6D firmware response CRC mismatch")
        return raw[5:12].decode("ascii", errors="ignore").rstrip("\x00") or None

    def _request_sample(self) -> Px6dSample:
        port = self._serial
        if port is None:
            raise RuntimeError("PX6D serial port is not open")
        started = time.perf_counter()
        port.write(command_packet(COMMAND_READ_FRAME, 0x01, self.device_id))
        port.flush()
        frame = self._read_packet(
            port,
            minimum_length=FRAME_LENGTH,
            timeout_sec=self.read_timeout_sec,
        )
        if frame is None:
            raise TimeoutError("PX6D frame timeout")
        try:
            values = parse_data_frame(frame)
        except Exception:
            with self._lock:
                self._invalid_frames += 1
            raise
        now_epoch = time.time()
        now_monotonic = time.monotonic()
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return Px6dSample(sequence, now_epoch, now_monotonic, *values, roundtrip_ms)

    @staticmethod
    def _read_packet(port: Any, *, minimum_length: int, timeout_sec: float) -> bytes | None:
        return _read_serial_packet(
            port,
            minimum_length=minimum_length,
            timeout_sec=timeout_sec,
        )

    def _append_sample(self, sample: Px6dSample) -> None:
        with self._lock:
            self._clear_reconnect_backoff_locked()
            evicted_sequence = (
                self._samples[0].sequence_id
                if len(self._samples) == self._samples.maxlen
                else None
            )
            self._samples.append(sample)
            if evicted_sequence is not None:
                self._conditioned_samples.pop(evicted_sequence, None)
                self._filtered_axis_samples.pop(evicted_sequence, None)
            self._valid_frames += 1
            filtered_axes = self._filter_zeroed_axes_locked(sample)
            self._filtered_axis_samples[sample.sequence_id] = filtered_axes
            reference = self._reference_fz_from_sample_locked(sample)
            conditioned = self._condition_reference_fz_locked(sample, reference)
            self._conditioned_samples[sample.sequence_id] = conditioned

    def _filter_zeroed_axes_locked(self, sample: Px6dSample) -> FilteredAxesSample:
        raw = sample.raw_values()
        tare = self._tare_values or (0.0,) * len(AXIS_NAMES)
        zeroed = tuple(value - offset for value, offset in zip(raw, tare))
        median_values: list[float] = []
        filtered_values: list[float] = []
        for axis, value in zip(AXIS_NAMES, zeroed):
            window = self._axis_median_windows[axis]
            window.append(float(value))
            median_value = float(statistics.median(window))
            previous = self._filtered_zeroed_axes[axis]
            filtered_value = (
                median_value
                if previous is None
                else float(previous) + self.filter_alpha * (median_value - float(previous))
            )
            self._filtered_zeroed_axes[axis] = filtered_value
            median_values.append(median_value)
            filtered_values.append(filtered_value)
        return FilteredAxesSample(
            sequence_id=sample.sequence_id,
            median_zeroed_values=tuple(median_values),
            filtered_zeroed_values=tuple(filtered_values),
        )

    def _condition_reference_fz_locked(
        self,
        sample: Px6dSample,
        reference_fz_n: float,
    ) -> ConditionedFzSample:
        self._median_reference_window.append(float(reference_fz_n))
        median_reference = statistics.median(self._median_reference_window)
        if self._filtered_reference_fz_n is None:
            self._filtered_reference_fz_n = median_reference
        else:
            self._filtered_reference_fz_n += self.filter_alpha * (
                median_reference - self._filtered_reference_fz_n
            )
        low_pass_reference = float(self._filtered_reference_fz_n)

        timestamp = float(sample.timestamp_epoch_sec)
        self._stationary_window.append((timestamp, low_pass_reference))
        cutoff = timestamp - self.stationary_window_sec
        while self._stationary_window and self._stationary_window[0][0] < cutoff:
            self._stationary_window.popleft()

        stationary = False
        if len(self._stationary_window) >= 3:
            span = self._stationary_window[-1][0] - self._stationary_window[0][0]
            values = [value for _, value in self._stationary_window]
            value_std = statistics.pstdev(values) if len(values) > 1 else 0.0
            value_range = max(values) - min(values)
            slope = (
                abs(values[-1] - values[0]) / span
                if span > 1e-9
                else math.inf
            )
            stationary = (
                span >= self.stationary_window_sec * 0.80
                and value_std <= self.stationary_std_max_n
                and value_range <= self.stationary_range_max_n
                and slope <= self.stationary_slope_max_n_per_sec
            )

        before_drift_correction = low_pass_reference - self._drift_offset_n
        near_zero = abs(before_drift_correction) <= self.auto_zero_capture_limit_n
        # This reference channel is intentionally compression-only. A stable
        # negative residual is therefore release-side zero drift, not a valid
        # press. Let the baseline reacquire that direction without widening
        # the positive capture band, which would erase a real light load.
        release_side_drift = (
            before_drift_correction < -self.auto_zero_capture_limit_n
            and abs(before_drift_correction)
            <= self.auto_zero_release_reacquire_limit_n
        )
        auto_zero_active = False
        if self._tare_values is None:
            self._stationary_since_epoch_sec = None
            filter_status = "tare_required"
        elif self._auto_zero_frozen:
            self._stationary_since_epoch_sec = None
            filter_status = (
                f"auto_zero_frozen:{self._auto_zero_freeze_reason}"
                if self._auto_zero_freeze_reason
                else "auto_zero_frozen"
            )
        elif not self.auto_zero_drift_enabled:
            self._stationary_since_epoch_sec = None
            filter_status = "filtered"
        elif stationary and (near_zero or release_side_drift):
            if self._stationary_since_epoch_sec is None:
                self._stationary_since_epoch_sec = timestamp
            held_for = max(0.0, timestamp - self._stationary_since_epoch_sec)
            if held_for >= self.auto_zero_hold_sec:
                proposed_offset = self._drift_offset_n + self.auto_zero_alpha * (
                    low_pass_reference - self._drift_offset_n
                )
                self._drift_offset_n = max(
                    -self.maximum_drift_offset_n,
                    min(self.maximum_drift_offset_n, proposed_offset),
                )
                auto_zero_active = True
                filter_status = (
                    "stationary_release_drift_tracking"
                    if release_side_drift
                    else "stationary_drift_tracking"
                )
            else:
                filter_status = "stationary_hold"
        else:
            self._stationary_since_epoch_sec = None
            filter_status = "contact_or_motion_filter_frozen"

        drift_corrected = low_pass_reference - self._drift_offset_n
        conditioned_reference = (
            0.0 if abs(drift_corrected) <= self.force_deadband_n else drift_corrected
        )
        self._stationary_detected = stationary
        self._auto_zero_drift_active = auto_zero_active
        self._filter_status = filter_status
        return ConditionedFzSample(
            sequence_id=sample.sequence_id,
            median_reference_fz_n=float(median_reference),
            low_pass_reference_fz_n=low_pass_reference,
            drift_offset_n=float(self._drift_offset_n),
            drift_corrected_reference_fz_n=float(drift_corrected),
            conditioned_reference_fz_n=float(conditioned_reference),
            stationary_detected=stationary,
            auto_zero_drift_active=auto_zero_active,
            filter_status=filter_status,
        )

    def set_auto_zero_frozen(
        self,
        frozen: bool,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Freeze drift adaptation while preserving filtering and raw samples."""

        with self._lock:
            self._auto_zero_frozen = bool(frozen)
            self._auto_zero_freeze_reason = (
                str(reason or "external_guard") if frozen else None
            )
            self._stationary_since_epoch_sec = None
            if frozen:
                self._auto_zero_drift_active = False
                self._filter_status = (
                    f"auto_zero_frozen:{self._auto_zero_freeze_reason}"
                )
        return {
            "ok": True,
            "auto_zero_frozen": self._auto_zero_frozen,
            "auto_zero_freeze_reason": self._auto_zero_freeze_reason,
        }

    def _reset_conditioner_locked(self) -> None:
        self._conditioned_samples.clear()
        self._filtered_axis_samples.clear()
        for window in self._axis_median_windows.values():
            window.clear()
        for axis in AXIS_NAMES:
            self._filtered_zeroed_axes[axis] = None
        self._median_reference_window.clear()
        self._stationary_window.clear()
        self._filtered_reference_fz_n = None
        self._drift_offset_n = 0.0
        self._stationary_since_epoch_sec = None
        self._stationary_detected = False
        self._auto_zero_drift_active = False
        self._filter_status = "warming_up"

    def _close_serial_locked(self) -> None:
        ports = (self._serial, self._connecting_serial)
        self._serial = None
        self._connecting_serial = None
        closed_ids: set[int] = set()
        for port in ports:
            if port is None or id(port) in closed_ids:
                continue
            closed_ids.add(id(port))
            _close_serial_port(port)

    def tare(
        self,
        *,
        duration_sec: float = 1.0,
        max_std_n: float | None = None,
        wait_for_new_samples: bool = True,
    ) -> dict[str, Any]:
        """Capture a stable software zero; no hardware calibration command is sent."""

        duration_sec = max(0.25, min(5.0, float(duration_sec)))
        maximum_std = self.auto_tare_max_std_n if max_std_n is None else max(0.01, float(max_std_n))
        starting_sequence: int | None = None
        if wait_for_new_samples:
            starting_sequence = self.latest_sequence
            started_monotonic = time.monotonic()
            deadline = time.monotonic() + duration_sec + 1.0
            while time.monotonic() < deadline:
                if time.monotonic() - started_monotonic >= duration_sec:
                    break
                time.sleep(0.02)
        cutoff = time.time() - duration_sec
        with self._lock:
            candidates = [
                sample
                for sample in self._samples
                if (
                    sample.sequence_id > starting_sequence
                    if starting_sequence is not None
                    else sample.timestamp_epoch_sec >= cutoff
                )
            ]
        minimum_samples = max(8, int(self.poll_hz * duration_sec * 0.40))
        if len(candidates) < minimum_samples:
            with self._lock:
                self._tare_status = "not_enough_samples"
            return {
                "ok": False,
                "status": "not_enough_samples",
                "sample_count": len(candidates),
                "required_samples": minimum_samples,
            }
        fz_values = [sample.fz_n for sample in candidates]
        fz_std = statistics.pstdev(fz_values) if len(fz_values) > 1 else 0.0
        if fz_std > maximum_std:
            with self._lock:
                self._tare_status = "unstable_signal"
            return {
                "ok": False,
                "status": "unstable_signal",
                "sample_count": len(candidates),
                "fz_std_n": fz_std,
                "maximum_std_n": maximum_std,
            }
        medians = tuple(
            statistics.median(float(getattr(sample, axis)) for sample in candidates)
            for axis in AXIS_NAMES
        )
        with self._lock:
            self._tare_values = medians
            self._tare_timestamp_epoch_sec = time.time()
            self._tare_status = "ready"
            self._tare_fz_std_n = fz_std
            self._tare_sample_count = len(candidates)
            # A new zero changes the meaning of every zero-relative value. Do
            # not expose pre-tare raw frames through the new zero or attach
            # them to a later optical frame as if they belonged to this zero.
            self._samples.clear()
            self._reset_conditioner_locked()
        return {
            "ok": True,
            "status": "ready",
            "sample_count": len(candidates),
            "fz_std_n": fz_std,
            "tare": dict(zip(AXIS_NAMES, medians)),
            "software_only": True,
            "history_reset": True,
            "sampling_scope": (
                "post_request_samples" if wait_for_new_samples else "recent_samples"
            ),
        }

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return self._samples[-1].sequence_id if self._samples else 0

    def _history_span_sec(self) -> float:
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            return self._samples[-1].timestamp_epoch_sec - self._samples[0].timestamp_epoch_sec

    def _reference_fz_from_sample_locked(self, sample: Px6dSample) -> float:
        tare_fz = self._tare_values[2] if self._tare_values is not None else 0.0
        return self.compression_sign * (sample.fz_n - tare_fz)

    def _mechanical_metrics(self, zeroed: tuple[float, ...]) -> dict[str, Any]:
        fx, fy, fz, mx, my, mz = (float(value) for value in zeroed)
        force_resultant = math.sqrt(fx * fx + fy * fy + fz * fz)
        shear_resultant = math.sqrt(fx * fx + fy * fy)
        moment_resultant = math.sqrt(mx * mx + my * my + mz * mz)
        force_utilization = 100.0 * max(abs(fx), abs(fy), abs(fz)) / self.force_full_scale_per_axis_n
        moment_utilization = (
            100.0
            * max(abs(mx), abs(my), abs(mz))
            / self.moment_full_scale_per_axis_nm
        )
        peak_utilization = max(force_utilization, moment_utilization)
        return {
            "force_resultant_n": force_resultant,
            "shear_resultant_n": shear_resultant,
            "moment_resultant_nm": moment_resultant,
            "force_utilization_percent": force_utilization,
            "moment_utilization_percent": moment_utilization,
            "peak_utilization_percent": peak_utilization,
            "utilization_status": (
                "warning" if peak_utilization >= self.warning_utilization_percent else "ok"
            ),
            "force_full_scale_per_axis_n": self.force_full_scale_per_axis_n,
            "moment_full_scale_per_axis_nm": self.moment_full_scale_per_axis_nm,
        }

    def _sync_quality(self, sync_offset_ms: float) -> str:
        offset = abs(float(sync_offset_ms))
        if offset <= self.sync_excellent_max_offset_ms:
            return "excellent"
        if offset <= self.sync_good_max_offset_ms:
            return "good"
        if offset <= self.sync_acceptable_max_offset_ms:
            return "acceptable"
        return "poor"

    def _sample_payload(
        self,
        sample: Px6dSample,
        *,
        tare_values: tuple[float, float, float, float, float, float] | None,
        conditioned: ConditionedFzSample | None,
        filtered_axes: FilteredAxesSample | None,
        drift_offset_n: float,
        tare_status: str,
        tare_fz_std_n: float | None,
        tare_sample_count: int,
    ) -> dict[str, Any]:
        raw = sample.raw_values()
        tare = tare_values or (0.0,) * 6
        zeroed = tuple(value - offset for value, offset in zip(raw, tare))
        reference_fz = self.compression_sign * zeroed[2]
        if filtered_axes is None:
            filtered_axes = FilteredAxesSample(
                sequence_id=sample.sequence_id,
                median_zeroed_values=tuple(float(value) for value in zeroed),
                filtered_zeroed_values=tuple(float(value) for value in zeroed),
            )
        if conditioned is None:
            conditioned = ConditionedFzSample(
                sequence_id=sample.sequence_id,
                median_reference_fz_n=reference_fz,
                low_pass_reference_fz_n=reference_fz,
                drift_offset_n=drift_offset_n,
                drift_corrected_reference_fz_n=reference_fz - drift_offset_n,
                conditioned_reference_fz_n=(
                    0.0
                    if abs(reference_fz - drift_offset_n) <= self.force_deadband_n
                    else reference_fz - drift_offset_n
                ),
                stationary_detected=False,
                auto_zero_drift_active=False,
                filter_status="filter_history_unavailable",
            )
        mechanical = self._mechanical_metrics(zeroed)
        return {
            "sequence_id": sample.sequence_id,
            "timestamp_epoch_sec": sample.timestamp_epoch_sec,
            "timestamp_monotonic_sec": sample.timestamp_monotonic_sec,
            "raw": dict(zip(AXIS_NAMES, raw)),
            "zeroed": dict(zip(AXIS_NAMES, zeroed)),
            "median_zeroed": dict(
                zip(AXIS_NAMES, filtered_axes.median_zeroed_values)
            ),
            "filtered_zeroed": dict(
                zip(AXIS_NAMES, filtered_axes.filtered_zeroed_values)
            ),
            "reference_fz_n": reference_fz,
            "reference_fz_display_n": max(
                0.0, conditioned.conditioned_reference_fz_n
            ),
            "force_fz_n": max(0.0, conditioned.conditioned_reference_fz_n),
            "median_reference_fz_n": conditioned.median_reference_fz_n,
            "filtered_reference_fz_n": conditioned.low_pass_reference_fz_n,
            "drift_offset_n": conditioned.drift_offset_n,
            "drift_corrected_reference_fz_n": (
                conditioned.drift_corrected_reference_fz_n
            ),
            "conditioned_reference_fz_n": conditioned.conditioned_reference_fz_n,
            "stationary_detected": conditioned.stationary_detected,
            "auto_zero_drift_active": conditioned.auto_zero_drift_active,
            "force_filter_status": conditioned.filter_status,
            "compression_sign": self.compression_sign,
            "roundtrip_ms": sample.roundtrip_ms,
            "tare_ready": tare_values is not None,
            "tare_status": tare_status,
            "tare_fz_std_n": tare_fz_std_n,
            "tare_sample_count": tare_sample_count,
            "mechanical": mechanical,
            "filtered_mechanical": self._mechanical_metrics(
                filtered_axes.filtered_zeroed_values
            ),
        }

    def _sample_payload_locked(self, sample: Px6dSample) -> dict[str, Any]:
        return self._sample_payload(
            sample,
            tare_values=self._tare_values,
            conditioned=self._conditioned_samples.get(sample.sequence_id),
            filtered_axes=self._filtered_axis_samples.get(sample.sequence_id),
            drift_offset_n=self._drift_offset_n,
            tare_status=self._tare_status,
            tare_fz_std_n=self._tare_fz_std_n,
            tare_sample_count=self._tare_sample_count,
        )

    def latest(self) -> dict[str, Any]:
        with self._lock:
            sample = self._samples[-1] if self._samples else None
            payload = self._sample_payload_locked(sample) if sample is not None else None
        return {"ok": sample is not None, "status": self.status(), "sample": payload}

    def trace(self, limit: int = 500) -> dict[str, Any]:
        limit = max(1, min(20000, int(limit)))
        with self._lock:
            samples = list(self._samples)[-limit:]
            payload_inputs = [
                (
                    sample,
                    self._conditioned_samples.get(sample.sequence_id),
                    self._filtered_axis_samples.get(sample.sequence_id),
                )
                for sample in samples
            ]
            tare_values = self._tare_values
            drift_offset_n = self._drift_offset_n
            tare_status = self._tare_status
            tare_fz_std_n = self._tare_fz_std_n
            tare_sample_count = self._tare_sample_count
        rows = [
            self._sample_payload(
                sample,
                tare_values=tare_values,
                conditioned=conditioned,
                filtered_axes=filtered_axes,
                drift_offset_n=drift_offset_n,
                tare_status=tare_status,
                tare_fz_std_n=tare_fz_std_n,
                tare_sample_count=tare_sample_count,
            )
            for sample, conditioned, filtered_axes in payload_inputs
        ]
        return {"ok": True, "count": len(rows), "samples": rows, "status": self.status()}

    def synchronized_snapshot(
        self,
        timestamp_epoch_sec: float | None,
        *,
        window_sec: float | None = None,
    ) -> dict[str, Any]:
        if timestamp_epoch_sec is None:
            return {"ok": False, "status": "spectrum_timestamp_missing"}
        target = float(timestamp_epoch_sec)
        half_window = self.sync_window_sec if window_sec is None else max(0.01, float(window_sec))
        with self._lock:
            samples = list(self._samples)
            tare = self._tare_values or (0.0,) * 6
            conditioned_by_sequence = dict(self._conditioned_samples)
            filtered_axes_by_sequence = dict(self._filtered_axis_samples)
            auto_zero_frozen = self._auto_zero_frozen
            auto_zero_freeze_reason = self._auto_zero_freeze_reason
        if not samples:
            return {"ok": False, "status": "px6d_sample_missing"}
        selected = [
            sample for sample in samples if abs(sample.timestamp_epoch_sec - target) <= half_window
        ]
        method = "window_median"
        if not selected:
            nearest = min(samples, key=lambda sample: abs(sample.timestamp_epoch_sec - target))
            age = abs(nearest.timestamp_epoch_sec - target)
            if age > self.sync_max_age_sec:
                return {
                    "ok": False,
                    "status": "px6d_sample_too_far_from_spectrum",
                    "nearest_offset_ms": (nearest.timestamp_epoch_sec - target) * 1000.0,
                    "sync_within_target": False,
                    "calibration_sync_ok": False,
                }
            selected = [nearest]
            method = "nearest_sample"
        raw_medians = tuple(
            statistics.median(float(getattr(sample, axis)) for sample in selected)
            for axis in AXIS_NAMES
        )
        zeroed = tuple(value - offset for value, offset in zip(raw_medians, tare))
        reference_fz = self.compression_sign * zeroed[2]
        selected_filtered_axes = [
            filtered_axes_by_sequence[sample.sequence_id]
            for sample in selected
            if sample.sequence_id in filtered_axes_by_sequence
        ]
        if selected_filtered_axes:
            median_zeroed = tuple(
                statistics.median(item.median_zeroed_values[index] for item in selected_filtered_axes)
                for index in range(len(AXIS_NAMES))
            )
            filtered_zeroed = tuple(
                statistics.median(item.filtered_zeroed_values[index] for item in selected_filtered_axes)
                for index in range(len(AXIS_NAMES))
            )
        else:
            median_zeroed = tuple(float(value) for value in zeroed)
            filtered_zeroed = tuple(float(value) for value in zeroed)
        selected_conditioned = [
            conditioned_by_sequence[sample.sequence_id]
            for sample in selected
            if sample.sequence_id in conditioned_by_sequence
        ]
        if selected_conditioned:
            median_reference_fz = statistics.median(
                item.median_reference_fz_n for item in selected_conditioned
            )
            filtered_reference_fz = statistics.median(
                item.low_pass_reference_fz_n for item in selected_conditioned
            )
            drift_offset = statistics.median(
                item.drift_offset_n for item in selected_conditioned
            )
            drift_corrected_reference_fz = statistics.median(
                item.drift_corrected_reference_fz_n for item in selected_conditioned
            )
            conditioned_reference_fz = statistics.median(
                item.conditioned_reference_fz_n for item in selected_conditioned
            )
            latest_conditioned = max(
                selected_conditioned, key=lambda item: item.sequence_id
            )
        else:
            median_reference_fz = reference_fz
            filtered_reference_fz = reference_fz
            drift_offset = 0.0
            drift_corrected_reference_fz = reference_fz
            conditioned_reference_fz = (
                0.0 if abs(reference_fz) <= self.force_deadband_n else reference_fz
            )
            latest_conditioned = None
        center_timestamp = statistics.median(sample.timestamp_epoch_sec for sample in selected)
        sync_offset_ms = (center_timestamp - target) * 1000.0
        sequence_ids = [sample.sequence_id for sample in selected]
        calibration_sync_ok = (
            abs(sync_offset_ms) <= self.sync_acceptable_max_offset_ms
        )
        return {
            "ok": True,
            "status": (
                "synced"
                if calibration_sync_ok
                else "outside_calibration_sync_tolerance"
            ),
            "sync_method": method,
            "sample_count": len(selected),
            "force_sequence_start": min(sequence_ids),
            "force_sequence_end": max(sequence_ids),
            "spectrum_timestamp_epoch_sec": target,
            "force_timestamp_epoch_sec": center_timestamp,
            "sync_offset_ms": sync_offset_ms,
            "sync_quality": self._sync_quality(sync_offset_ms),
            "sync_within_target": calibration_sync_ok,
            "calibration_sync_ok": calibration_sync_ok,
            "window_half_width_sec": half_window,
            "raw": dict(zip(AXIS_NAMES, raw_medians)),
            "zeroed": dict(zip(AXIS_NAMES, zeroed)),
            "median_zeroed": dict(zip(AXIS_NAMES, median_zeroed)),
            "filtered_zeroed": dict(zip(AXIS_NAMES, filtered_zeroed)),
            "reference_fz_n": reference_fz,
            "reference_fz_display_n": max(0.0, conditioned_reference_fz),
            "force_fz_n": max(0.0, conditioned_reference_fz),
            "median_reference_fz_n": median_reference_fz,
            "filtered_reference_fz_n": filtered_reference_fz,
            "drift_offset_n": drift_offset,
            "drift_corrected_reference_fz_n": drift_corrected_reference_fz,
            "conditioned_reference_fz_n": conditioned_reference_fz,
            "stationary_detected": bool(
                latest_conditioned and latest_conditioned.stationary_detected
            ),
            "auto_zero_drift_active": bool(
                latest_conditioned and latest_conditioned.auto_zero_drift_active
            ),
            "auto_zero_frozen": auto_zero_frozen,
            "auto_zero_freeze_reason": auto_zero_freeze_reason,
            "force_filter_status": (
                latest_conditioned.filter_status
                if latest_conditioned is not None
                else "filter_history_unavailable"
            ),
            "compression_sign": self.compression_sign,
            "tare_ready": self._tare_values is not None,
            "tare_status": self._tare_status,
            "tare_fz_std_n": self._tare_fz_std_n,
            "tare_sample_count": self._tare_sample_count,
            "mechanical": self._mechanical_metrics(zeroed),
            "filtered_mechanical": self._mechanical_metrics(filtered_zeroed),
            "semantics": "PX6D_reference_Fz_not_optical_force_prediction",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            worker_alive = bool(
                self._thread is not None and self._thread.is_alive()
            )
            serial_process = self._serial_worker_process
            try:
                serial_worker_alive = bool(
                    serial_process is not None and serial_process.is_alive()
                )
            except Exception:
                serial_worker_alive = False
            latest = self._samples[-1] if self._samples else None
            now = time.time()
            age = now - latest.timestamp_epoch_sec if latest is not None else None
            freshness_limit = max(0.5, 3.0 / self.poll_hz)
            sample_fresh = bool(
                latest is not None
                and worker_alive
                and self._connected
                and age is not None
                and 0.0 <= age <= freshness_limit
            )
            elapsed = (
                latest.timestamp_epoch_sec - self._samples[0].timestamp_epoch_sec
                if latest is not None and len(self._samples) > 1
                else None
            )
            observed_hz = (
                (len(self._samples) - 1) / elapsed
                if elapsed is not None and elapsed > 0
                else None
            )
            next_reconnect_in_sec = (
                max(0.0, self._next_reconnect_epoch_sec - now)
                if self._next_reconnect_epoch_sec is not None
                else None
            )
            return {
                "running": self._running,
                "worker_alive": worker_alive,
                "connected": self._connected,
                "connection_in_progress": bool(
                    self._connecting_serial is not None
                    or (
                        self.isolate_serial_process
                        and serial_worker_alive
                        and not self._connected
                    )
                ),
                "lifecycle_status": self._lifecycle_status,
                "stop_requested": self._stop_event.is_set(),
                "port": self._active_port or self.configured_port,
                "configured_port": self.configured_port,
                "active_port": self._active_port,
                "auto_detect_port": self.auto_detect_port,
                "port_detection_status": self._port_detection_status,
                "detected_device_identity": (
                    dict(self._detected_device_identity)
                    if self._detected_device_identity is not None
                    else None
                ),
                "available_serial_ports": [
                    dict(item) for item in self._available_serial_ports
                ],
                "port_scan_count": self._port_scan_count,
                "last_port_scan_epoch_sec": self._last_port_scan_epoch_sec,
                "baud_rate": self.baud_rate,
                "firmware_version": self._firmware_version,
                "valid_frame_count": self._valid_frames,
                "invalid_frame_count": self._invalid_frames,
                "connection_attempts": self._connection_attempts,
                "consecutive_connection_failures": (
                    self._consecutive_connection_failures
                ),
                "connection_error_kind": self._connection_error_kind,
                "port_busy_detected": self._port_busy_detected,
                "reconnect_delay_sec": self._current_reconnect_delay_sec,
                "next_reconnect_in_sec": next_reconnect_in_sec,
                "reconnect_max_interval_sec": self.reconnect_max_interval_sec,
                "serial_isolation_enabled": self.isolate_serial_process,
                "serial_worker_alive": serial_worker_alive,
                "serial_worker_pid": self._serial_worker_pid,
                "serial_worker_job_guard_active": bool(
                    self._serial_worker_job_handle
                ),
                "serial_worker_restart_count": self._serial_worker_restarts,
                "serial_worker_forced_termination_count": (
                    self._serial_worker_forced_terminations
                ),
                "serial_worker_watchdog_sec": self.worker_watchdog_sec,
                "auto_zero_frozen": self._auto_zero_frozen,
                "auto_zero_freeze_reason": self._auto_zero_freeze_reason,
                "last_error": self._last_error,
                "last_sample_age_sec": age,
                "sample_fresh": sample_fresh,
                "sample_freshness_limit_sec": freshness_limit,
                "configured_poll_hz": self.poll_hz,
                "observed_sample_hz": observed_hz,
                "tare_ready": self._tare_values is not None,
                "tare_status": self._tare_status,
                "tare_timestamp_epoch_sec": self._tare_timestamp_epoch_sec,
                "tare_fz_std_n": self._tare_fz_std_n,
                "tare_sample_count": self._tare_sample_count,
                "compression_sign": self.compression_sign,
                "primary_axis": "Fz",
                "force_full_scale_per_axis_n": self.force_full_scale_per_axis_n,
                "moment_full_scale_per_axis_nm": self.moment_full_scale_per_axis_nm,
                "warning_utilization_percent": self.warning_utilization_percent,
                "force_conditioning": {
                    "median_window_samples": self.median_window_samples,
                    "low_pass_alpha": self.filter_alpha,
                    "deadband_n": self.force_deadband_n,
                    "stationary_window_sec": self.stationary_window_sec,
                    "stationary_std_max_n": self.stationary_std_max_n,
                    "stationary_range_max_n": self.stationary_range_max_n,
                    "stationary_slope_max_n_per_sec": (
                        self.stationary_slope_max_n_per_sec
                    ),
                    "auto_zero_drift_enabled": self.auto_zero_drift_enabled,
                    "auto_zero_hold_sec": self.auto_zero_hold_sec,
                    "auto_zero_capture_limit_n": self.auto_zero_capture_limit_n,
                    "auto_zero_release_reacquire_limit_n": (
                        self.auto_zero_release_reacquire_limit_n
                    ),
                    "auto_zero_alpha": self.auto_zero_alpha,
                    "maximum_drift_offset_n": self.maximum_drift_offset_n,
                    "current_drift_offset_n": self._drift_offset_n,
                    "stationary_detected": self._stationary_detected,
                    "auto_zero_drift_active": self._auto_zero_drift_active,
                    "filter_status": self._filter_status,
                    "raw_values_retained": True,
                    "all_six_axes_filtered": True,
                    "filtered_axis_names": list(AXIS_NAMES),
                },
                "sync_quality_thresholds_ms": {
                    "excellent": self.sync_excellent_max_offset_ms,
                    "good": self.sync_good_max_offset_ms,
                    "acceptable": self.sync_acceptable_max_offset_ms,
                },
                "software_tare_only": True,
                "hardware_calibration_command_used": False,
            }
