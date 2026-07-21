"""PX6D six-axis reference-force reader for synchronized TOUCH acquisition.

The reader uses the vendor USB/RS485 frame contract without issuing hardware
calibration commands.  Tare is implemented entirely in software so raw sensor
values remain available for traceability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import struct
import threading
import time
from typing import Any, Iterable

try:
    import serial
except Exception:  # pragma: no cover - exposed through status()
    serial = None


AXIS_NAMES = ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm")
FRAME_HEADER = b"\xAA\x55"
FRAME_LENGTH = 29
COMMAND_READ_FRAME = 0x05
COMMAND_VERSION = 0x07
RESPONSE_DATA = 0x03


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


class Px6dReader:
    """Background polling reader with software tare and timestamp alignment."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        payload = dict(config or {})
        self.port = str(payload.get("port") or "COM3")
        self.baud_rate = int(payload.get("baud_rate") or 921600)
        self.device_id = int(payload.get("device_id") or 0x7F)
        self.poll_hz = max(1.0, min(100.0, float(payload.get("poll_hz") or 50.0)))
        self.read_timeout_sec = max(0.02, float(payload.get("read_timeout_sec") or 0.20))
        self.reconnect_interval_sec = max(
            0.1, float(payload.get("reconnect_interval_sec") or 1.0)
        )
        self.history_seconds = max(10.0, float(payload.get("history_seconds") or 300.0))
        self.compression_sign = -1.0 if float(payload.get("compression_sign") or -1.0) < 0 else 1.0
        self.filter_alpha = min(1.0, max(0.01, float(payload.get("filter_alpha") or 0.25)))
        self.auto_tare_on_start = bool(payload.get("auto_tare_on_start", True))
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
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial = None
        self._running = False
        self._connected = False
        self._firmware_version: str | None = None
        self._last_error: str | None = None
        self._sequence = 0
        self._valid_frames = 0
        self._invalid_frames = 0
        self._connection_attempts = 0
        self._started_at_epoch_sec: float | None = None
        self._tare_values: tuple[float, float, float, float, float, float] | None = None
        self._tare_timestamp_epoch_sec: float | None = None
        self._tare_status = "required"
        self._tare_fz_std_n: float | None = None
        self._tare_sample_count = 0
        self._filtered_reference_fz_n: float | None = None

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop_event.clear()
            self._running = True
            self._started_at_epoch_sec = time.time()
            self._thread = threading.Thread(
                target=self._run,
                name="px6d-com3-reader",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._running = False
            self._connected = False
            self._close_serial_locked()
        return self.status()

    def _run(self) -> None:
        if serial is None:
            with self._lock:
                self._running = False
                self._last_error = "pyserial_not_installed"
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
                    self._connected = False
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._close_serial_locked()
                self._stop_event.wait(self.reconnect_interval_sec)
        with self._lock:
            self._running = False
            self._connected = False
            self._close_serial_locked()

    def _connect(self) -> None:
        with self._lock:
            self._connection_attempts += 1
        port = serial.Serial(
            self.port,
            self.baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.01,
            write_timeout=0.25,
        )
        try:
            port.reset_input_buffer()
            port.reset_output_buffer()
            time.sleep(0.10)
            firmware = self._query_version(port)
        except Exception:
            port.close()
            raise
        with self._lock:
            self._serial = port
            self._connected = True
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

    def _append_sample(self, sample: Px6dSample) -> None:
        with self._lock:
            self._samples.append(sample)
            self._valid_frames += 1
            reference = self._reference_fz_from_sample_locked(sample)
            if self._filtered_reference_fz_n is None:
                self._filtered_reference_fz_n = reference
            else:
                self._filtered_reference_fz_n += self.filter_alpha * (
                    reference - self._filtered_reference_fz_n
                )

    def _close_serial_locked(self) -> None:
        port, self._serial = self._serial, None
        if port is None:
            return
        try:
            port.close()
        except Exception:
            pass

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
        if wait_for_new_samples:
            starting_sequence = self.latest_sequence
            deadline = time.monotonic() + duration_sec + 1.0
            while time.monotonic() < deadline:
                if self.latest_sequence > starting_sequence and self._history_span_sec() >= duration_sec:
                    break
                time.sleep(0.02)
        cutoff = time.time() - duration_sec
        with self._lock:
            candidates = [sample for sample in self._samples if sample.timestamp_epoch_sec >= cutoff]
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
            self._filtered_reference_fz_n = 0.0
        return {
            "ok": True,
            "status": "ready",
            "sample_count": len(candidates),
            "fz_std_n": fz_std,
            "tare": dict(zip(AXIS_NAMES, medians)),
            "software_only": True,
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

    def _sample_payload_locked(self, sample: Px6dSample) -> dict[str, Any]:
        raw = sample.raw_values()
        tare = self._tare_values or (0.0,) * 6
        zeroed = tuple(value - offset for value, offset in zip(raw, tare))
        reference_fz = self.compression_sign * zeroed[2]
        mechanical = self._mechanical_metrics(zeroed)
        return {
            "sequence_id": sample.sequence_id,
            "timestamp_epoch_sec": sample.timestamp_epoch_sec,
            "timestamp_monotonic_sec": sample.timestamp_monotonic_sec,
            "raw": dict(zip(AXIS_NAMES, raw)),
            "zeroed": dict(zip(AXIS_NAMES, zeroed)),
            "reference_fz_n": reference_fz,
            "reference_fz_display_n": max(0.0, reference_fz),
            "filtered_reference_fz_n": self._filtered_reference_fz_n,
            "compression_sign": self.compression_sign,
            "roundtrip_ms": sample.roundtrip_ms,
            "tare_ready": self._tare_values is not None,
            "tare_status": self._tare_status,
            "tare_fz_std_n": self._tare_fz_std_n,
            "tare_sample_count": self._tare_sample_count,
            "mechanical": mechanical,
        }

    def latest(self) -> dict[str, Any]:
        with self._lock:
            sample = self._samples[-1] if self._samples else None
            payload = self._sample_payload_locked(sample) if sample is not None else None
        return {"ok": sample is not None, "status": self.status(), "sample": payload}

    def trace(self, limit: int = 500) -> dict[str, Any]:
        limit = max(1, min(20000, int(limit)))
        with self._lock:
            samples = list(self._samples)[-limit:]
            rows = [self._sample_payload_locked(sample) for sample in samples]
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
                }
            selected = [nearest]
            method = "nearest_sample"
        raw_medians = tuple(
            statistics.median(float(getattr(sample, axis)) for sample in selected)
            for axis in AXIS_NAMES
        )
        zeroed = tuple(value - offset for value, offset in zip(raw_medians, tare))
        reference_fz = self.compression_sign * zeroed[2]
        center_timestamp = statistics.median(sample.timestamp_epoch_sec for sample in selected)
        sync_offset_ms = (center_timestamp - target) * 1000.0
        sequence_ids = [sample.sequence_id for sample in selected]
        return {
            "ok": True,
            "status": "synced",
            "sync_method": method,
            "sample_count": len(selected),
            "force_sequence_start": min(sequence_ids),
            "force_sequence_end": max(sequence_ids),
            "spectrum_timestamp_epoch_sec": target,
            "force_timestamp_epoch_sec": center_timestamp,
            "sync_offset_ms": sync_offset_ms,
            "sync_quality": self._sync_quality(sync_offset_ms),
            "sync_within_target": abs(sync_offset_ms) <= self.sync_acceptable_max_offset_ms,
            "window_half_width_sec": half_window,
            "raw": dict(zip(AXIS_NAMES, raw_medians)),
            "zeroed": dict(zip(AXIS_NAMES, zeroed)),
            "reference_fz_n": reference_fz,
            "reference_fz_display_n": max(0.0, reference_fz),
            "compression_sign": self.compression_sign,
            "tare_ready": self._tare_values is not None,
            "tare_status": self._tare_status,
            "tare_fz_std_n": self._tare_fz_std_n,
            "tare_sample_count": self._tare_sample_count,
            "mechanical": self._mechanical_metrics(zeroed),
            "semantics": "PX6D_reference_Fz_not_optical_force_prediction",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            latest = self._samples[-1] if self._samples else None
            now = time.time()
            age = now - latest.timestamp_epoch_sec if latest is not None else None
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
            return {
                "running": self._running,
                "connected": self._connected,
                "port": self.port,
                "baud_rate": self.baud_rate,
                "firmware_version": self._firmware_version,
                "valid_frame_count": self._valid_frames,
                "invalid_frame_count": self._invalid_frames,
                "connection_attempts": self._connection_attempts,
                "last_error": self._last_error,
                "last_sample_age_sec": age,
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
                "sync_quality_thresholds_ms": {
                    "excellent": self.sync_excellent_max_offset_ms,
                    "good": self.sync_good_max_offset_ms,
                    "acceptable": self.sync_acceptable_max_offset_ms,
                },
                "software_tare_only": True,
                "hardware_calibration_command_used": False,
            }
