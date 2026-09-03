from __future__ import annotations

import struct
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.px6d_reader import (
    AXIS_NAMES,
    Px6dReader,
    Px6dSample,
    command_packet,
    crc8,
    parse_data_frame,
)
from backend import main as backend_main


def make_sample(
    sequence_id: int,
    timestamp: float,
    *,
    fz_n: float,
    fx_n: float = 0.01,
    fy_n: float = -0.02,
    mx_nm: float = 0.001,
    my_nm: float = -0.002,
    mz_nm: float = 0.003,
) -> Px6dSample:
    return Px6dSample(
        sequence_id=sequence_id,
        timestamp_epoch_sec=timestamp,
        timestamp_monotonic_sec=timestamp,
        fx_n=fx_n,
        fy_n=fy_n,
        fz_n=fz_n,
        mx_nm=mx_nm,
        my_nm=my_nm,
        mz_nm=mz_nm,
        roundtrip_ms=20.0,
    )


class Px6dProtocolContractTests(unittest.TestCase):
    def test_crc_and_version_command_match_vendor_protocol(self) -> None:
        self.assertEqual(command_packet(0x07, 0x01), bytes.fromhex("AA 55 7F 07 01 D0"))

    def test_six_axis_data_frame_is_little_endian_float32(self) -> None:
        expected = (1.25, -2.5, 3.75, -0.125, 0.25, -0.5)
        body = bytes.fromhex("AA 55 7F 03") + struct.pack("<6f", *expected)
        frame = body + bytes([crc8(body)])
        parsed = parse_data_frame(frame)
        for actual, target in zip(parsed, expected):
            self.assertAlmostEqual(actual, target, places=6)

    def test_crc_failure_is_rejected(self) -> None:
        body = bytes.fromhex("AA 55 7F 03") + struct.pack("<6f", *(0.0,) * 6)
        with self.assertRaisesRegex(ValueError, "CRC"):
            parse_data_frame(body + b"\xFF")


class Px6dReferenceForceTests(unittest.TestCase):
    def test_status_marks_samples_fresh_only_while_connected_and_current(self) -> None:
        class LiveThread:
            @staticmethod
            def is_alive() -> bool:
                return True

        reader = Px6dReader({"poll_hz": 50.0, "auto_tare_on_start": False})
        reader._append_sample(make_sample(1, time.time(), fz_n=0.80))
        with reader._lock:
            reader._thread = LiveThread()  # type: ignore[assignment]
            reader._running = True
            reader._connected = True

        live_status = reader.status()
        self.assertTrue(live_status["sample_fresh"])
        self.assertEqual(live_status["sample_freshness_limit_sec"], 0.5)

        with reader._lock:
            reader._connected = False
        disconnected_status = reader.status()
        self.assertFalse(disconnected_status["sample_fresh"])

        with reader._lock:
            reader._connected = True
            reader._samples.clear()
            reader._append_sample(make_sample(2, time.time() - 2.0, fz_n=0.80))
        stale_status = reader.status()
        self.assertFalse(stale_status["sample_fresh"])

    def test_missing_pyserial_is_reported_before_worker_launch(self) -> None:
        reader = Px6dReader({"auto_tare_on_start": False})
        with patch("backend.px6d_reader.serial", None):
            result = reader.start()

        self.assertFalse(result["ok"])
        self.assertEqual(result["operation_status"], "dependency_unavailable")
        self.assertEqual(result["lifecycle_status"], "unavailable")
        self.assertEqual(result["last_error"], "pyserial_not_installed")
        self.assertFalse(result["running"])
        self.assertFalse(result["worker_alive"])

    def test_start_failure_rolls_back_reader_lifecycle(self) -> None:
        class FailingThread:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

            def start(self) -> None:
                raise RuntimeError("thread launch failed")

            def is_alive(self) -> bool:
                return False

        reader = Px6dReader({"auto_tare_on_start": False})
        with patch("backend.px6d_reader.threading.Thread", FailingThread):
            result = reader.start()

        self.assertFalse(result["ok"])
        self.assertEqual(result["operation_status"], "start_failed")
        self.assertFalse(result["running"])
        self.assertFalse(result["worker_alive"])
        self.assertEqual(result["lifecycle_status"], "start_failed")
        self.assertIn("px6d_reader_start_failed", result["last_error"])

    def test_stop_timeout_preserves_live_worker_and_blocks_restart(self) -> None:
        class StuckThread:
            def __init__(self) -> None:
                self.join_timeout: float | None = None

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                self.join_timeout = timeout

        class FakePort:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        reader = Px6dReader({"auto_tare_on_start": False})
        worker = StuckThread()
        port = FakePort()
        with reader._lock:
            reader._thread = worker  # type: ignore[assignment]
            reader._serial = port
            reader._running = True
            reader._connected = True
            reader._lifecycle_status = "running"

        stopped = reader.stop()

        self.assertFalse(stopped["ok"])
        self.assertEqual(stopped["operation_status"], "stop_timeout")
        self.assertTrue(stopped["running"])
        self.assertTrue(stopped["worker_alive"])
        self.assertFalse(stopped["connected"])
        self.assertEqual(stopped["lifecycle_status"], "stop_timeout")
        self.assertEqual(stopped["last_error"], "px6d_reader_stop_timeout")
        self.assertTrue(port.closed)
        self.assertEqual(worker.join_timeout, 2.0)

        restarted = reader.start()
        self.assertFalse(restarted["ok"])
        self.assertEqual(restarted["operation_status"], "stop_in_progress")

    def test_stop_closes_serial_before_waiting_for_worker(self) -> None:
        class FakePort:
            def __init__(self) -> None:
                self.closed = False
                self.cancel_read_called = False
                self.cancel_write_called = False

            def cancel_read(self) -> None:
                self.cancel_read_called = True

            def cancel_write(self) -> None:
                self.cancel_write_called = True

            def close(self) -> None:
                self.closed = True

        class JoiningThread:
            def __init__(self, port: FakePort) -> None:
                self.port = port
                self.alive = True
                self.serial_was_closed_before_join = False

            def is_alive(self) -> bool:
                return self.alive

            def join(self, timeout: float | None = None) -> None:
                self.serial_was_closed_before_join = self.port.closed
                self.alive = False

        reader = Px6dReader({"auto_tare_on_start": False})
        port = FakePort()
        worker = JoiningThread(port)
        with reader._lock:
            reader._thread = worker  # type: ignore[assignment]
            reader._serial = port
            reader._running = True
            reader._connected = True
            reader._lifecycle_status = "running"

        result = reader.stop()

        self.assertTrue(result["ok"])
        self.assertEqual(result["operation_status"], "stopped")
        self.assertTrue(worker.serial_was_closed_before_join)
        self.assertTrue(port.cancel_read_called)
        self.assertTrue(port.cancel_write_called)
        self.assertFalse(result["running"])
        self.assertFalse(result["worker_alive"])
        self.assertEqual(result["lifecycle_status"], "stopped")

    def test_stop_interrupts_serial_handle_during_firmware_handshake(self) -> None:
        class FakePort:
            def __init__(self) -> None:
                self.closed = threading.Event()

            def reset_input_buffer(self) -> None:
                return None

            def reset_output_buffer(self) -> None:
                return None

            def close(self) -> None:
                self.closed.set()

        port = FakePort()

        class FakeSerialModule:
            EIGHTBITS = 8
            PARITY_NONE = "N"
            STOPBITS_ONE = 1

            @staticmethod
            def Serial(*args, **kwargs):
                return port

        reader = Px6dReader(
            {
                "auto_tare_on_start": False,
                "reconnect_interval_sec": 0.1,
            }
        )
        handshake_started = threading.Event()

        def blocked_handshake(active_port) -> str:
            handshake_started.set()
            if not active_port.closed.wait(timeout=2.0):
                raise TimeoutError("test handshake was not interrupted")
            raise OSError("serial handle closed during stop")

        reader._query_version = blocked_handshake  # type: ignore[method-assign]
        with patch("backend.px6d_reader.serial", FakeSerialModule):
            started = reader.start()
            self.assertTrue(started["ok"])
            self.assertTrue(handshake_started.wait(timeout=1.0))
            in_progress = reader.status()
            self.assertTrue(in_progress["connection_in_progress"])

            stopped = reader.stop()

        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["operation_status"], "stopped")
        self.assertTrue(port.closed.is_set())
        self.assertFalse(stopped["connection_in_progress"])
        self.assertFalse(stopped["connected"])
        self.assertFalse(stopped["running"])
        self.assertFalse(stopped["worker_alive"])
        self.assertEqual(stopped["lifecycle_status"], "stopped")
        self.assertIsNone(stopped["last_error"])

    def test_expected_serial_close_during_stop_is_not_reported_as_device_error(self) -> None:
        class FakePort:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        reader = Px6dReader(
            {
                "auto_tare_on_start": False,
                "poll_hz": 100.0,
                "reconnect_interval_sec": 0.1,
            }
        )
        entered_request = threading.Event()
        port = FakePort()

        def fake_connect() -> None:
            with reader._lock:
                reader._serial = port
                reader._connected = True
                reader._lifecycle_status = "running"

        def fake_request_sample() -> Px6dSample:
            entered_request.set()
            reader._stop_event.wait(timeout=1.0)
            raise OSError("serial handle closed by stop")

        reader._connect = fake_connect  # type: ignore[method-assign]
        reader._request_sample = fake_request_sample  # type: ignore[method-assign]

        started = reader.start()
        self.assertTrue(started["ok"])
        self.assertTrue(entered_request.wait(timeout=1.0))
        stopped = reader.stop()

        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["operation_status"], "stopped")
        self.assertFalse(stopped["running"])
        self.assertFalse(stopped["worker_alive"])
        self.assertIsNone(stopped["last_error"])
        self.assertTrue(port.closed)

    def test_outer_worker_finalizer_clears_state_after_nonstandard_exit(self) -> None:
        reader = Px6dReader({"auto_tare_on_start": False})

        def terminate_worker() -> None:
            raise SystemExit("simulated driver-level worker exit")

        reader._connect = terminate_worker  # type: ignore[method-assign]
        with reader._lock:
            reader._thread = threading.current_thread()
            reader._running = True
            reader._connected = False
            reader._lifecycle_status = "running"
        with patch("backend.px6d_reader.serial", object()):
            with self.assertRaises(SystemExit):
                reader._run()

        status = reader.status()
        self.assertFalse(status["running"])
        self.assertFalse(status["worker_alive"])
        self.assertFalse(status["connected"])
        self.assertEqual(status["lifecycle_status"], "worker_exited")
        self.assertEqual(
            status["last_error"],
            "PX6D reader worker exited unexpectedly",
        )

    def test_software_tare_and_compression_sign_produce_positive_reference_fz(self) -> None:
        reader = Px6dReader(
            {
                "poll_hz": 20.0,
                "compression_sign": -1,
                "auto_tare_max_std_n": 0.05,
            }
        )
        now = time.time()
        for index in range(20):
            reader._append_sample(
                make_sample(index + 1, now - 0.95 + index * 0.04, fz_n=0.80)
            )

        tare = reader.tare(duration_sec=1.0, wait_for_new_samples=False)
        self.assertTrue(tare["ok"])
        self.assertTrue(tare["history_reset"])
        self.assertEqual(reader.trace()["count"], 0)
        reader._append_sample(make_sample(21, now + 0.05, fz_n=0.30))

        latest = reader.latest()["sample"]
        self.assertAlmostEqual(latest["raw"]["fz_n"], 0.30, places=6)
        self.assertAlmostEqual(latest["zeroed"]["fz_n"], -0.50, places=6)
        self.assertAlmostEqual(latest["reference_fz_n"], 0.50, places=6)
        self.assertAlmostEqual(latest["reference_fz_display_n"], 0.50, places=6)
        self.assertAlmostEqual(latest["force_fz_n"], 0.50, places=6)
        self.assertTrue(latest["tare_ready"])
        self.assertAlmostEqual(latest["mechanical"]["force_resultant_n"], 0.50, delta=0.03)
        self.assertIn("shear_resultant_n", latest["mechanical"])
        self.assertEqual(latest["mechanical"]["utilization_status"], "ok")
        self.assertIsNotNone(latest["tare_fz_std_n"])

    def test_manual_tare_uses_only_samples_captured_after_request(self) -> None:
        reader = Px6dReader(
            {
                "poll_hz": 20.0,
                "compression_sign": -1,
                "auto_tare_max_std_n": 0.05,
            }
        )
        now = time.time()
        for index in range(20):
            reader._append_sample(
                make_sample(index + 1, now - 1.0 + index * 0.04, fz_n=8.0)
            )

        def append_new_zero_samples() -> None:
            for index in range(10):
                time.sleep(0.025)
                reader._append_sample(
                    make_sample(21 + index, time.time(), fz_n=1.0)
                )

        producer = threading.Thread(target=append_new_zero_samples)
        producer.start()
        tare = reader.tare(duration_sec=0.25, wait_for_new_samples=True)
        producer.join(timeout=1.0)

        self.assertTrue(tare["ok"])
        self.assertEqual(tare["sampling_scope"], "post_request_samples")
        self.assertAlmostEqual(tare["tare"]["fz_n"], 1.0, places=6)
        trace = reader.trace()
        self.assertTrue(
            all(
                abs(float(sample["raw"]["fz_n"]) - 1.0) < 1e-9
                for sample in trace["samples"]
            )
        )

    def test_spectrum_timestamp_uses_window_median_force_label(self) -> None:
        reader = Px6dReader(
            {
                "compression_sign": -1,
                "sync_window_sec": 0.05,
                "sync_max_age_sec": 0.5,
            }
        )
        target = time.time()
        reader._tare_values = (0.0, 0.0, 0.80, 0.0, 0.0, 0.0)
        reader._tare_status = "ready"
        reader._append_sample(make_sample(1, target - 0.02, fz_n=0.30))
        reader._append_sample(make_sample(2, target, fz_n=0.20))
        reader._append_sample(make_sample(3, target + 0.02, fz_n=0.10))

        aligned = reader.synchronized_snapshot(target)
        self.assertTrue(aligned["ok"])
        self.assertEqual(aligned["sync_method"], "window_median")
        self.assertEqual(aligned["sample_count"], 3)
        self.assertAlmostEqual(aligned["raw"]["fz_n"], 0.20, places=6)
        self.assertAlmostEqual(aligned["reference_fz_n"], 0.60, places=6)
        self.assertAlmostEqual(
            aligned["force_fz_n"],
            max(0.0, aligned["conditioned_reference_fz_n"]),
            places=6,
        )
        self.assertEqual(aligned["sync_quality"], "excellent")
        self.assertTrue(aligned["sync_within_target"])
        self.assertEqual(aligned["force_sequence_start"], 1)
        self.assertEqual(aligned["force_sequence_end"], 3)
        self.assertEqual(
            aligned["semantics"], "PX6D_reference_Fz_not_optical_force_prediction"
        )
        self.assertIn("filtered_reference_fz_n", aligned)
        self.assertEqual(set(aligned["filtered_zeroed"]), set(AXIS_NAMES))
        self.assertIn("filtered_mechanical", aligned)
        self.assertIn("drift_corrected_reference_fz_n", aligned)
        self.assertIn("conditioned_reference_fz_n", aligned)
        self.assertIn("force_filter_status", aligned)

    def test_stationary_near_zero_drift_is_corrected_but_contact_is_preserved(self) -> None:
        reader = Px6dReader(
            {
                "poll_hz": 20.0,
                "compression_sign": -1,
                "median_window_samples": 5,
                "filter_alpha": 0.35,
                "force_deadband_n": 0.010,
                "stationary_window_sec": 0.50,
                "stationary_std_max_n": 0.020,
                "stationary_range_max_n": 0.060,
                "stationary_slope_max_n_per_sec": 0.20,
                "auto_zero_hold_sec": 0.40,
                "auto_zero_capture_limit_n": 0.080,
                "auto_zero_alpha": 0.15,
            }
        )
        reader._tare_values = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        reader._tare_status = "ready"
        started = time.time()
        noise = (0.0, 0.001, -0.001, 0.0005, -0.0005)
        for index in range(80):
            drift_n = 0.040 * index / 79.0
            reader._append_sample(
                make_sample(
                    index + 1,
                    started + index * 0.05,
                    fz_n=1.0 - drift_n + noise[index % len(noise)],
                )
            )

        rest = reader.latest()["sample"]
        self.assertGreater(rest["reference_fz_n"], 0.035)
        self.assertGreater(rest["drift_offset_n"], 0.020)
        self.assertLess(abs(rest["conditioned_reference_fz_n"]), 0.020)
        self.assertTrue(rest["stationary_detected"])
        offset_before_contact = rest["drift_offset_n"]

        for index in range(20):
            reader._append_sample(
                make_sample(
                    81 + index,
                    started + (80 + index) * 0.05,
                    fz_n=0.50,
                )
            )
        contact = reader.latest()["sample"]
        self.assertGreater(contact["conditioned_reference_fz_n"], 0.40)
        self.assertLess(
            abs(contact["drift_offset_n"] - offset_before_contact),
            0.006,
        )
        self.assertEqual(
            contact["force_filter_status"], "contact_or_motion_filter_frozen"
        )

    def test_median_stage_rejects_isolated_force_spike(self) -> None:
        reader = Px6dReader(
            {
                "compression_sign": -1,
                "median_window_samples": 5,
                "filter_alpha": 1.0,
                "auto_zero_drift_enabled": False,
            }
        )
        reader._tare_values = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        reader._tare_status = "ready"
        started = time.time()
        for index in range(8):
            reader._append_sample(
                make_sample(index + 1, started + index * 0.02, fz_n=1.0)
            )
        reader._append_sample(make_sample(9, started + 0.18, fz_n=0.0))
        reader._append_sample(make_sample(10, started + 0.20, fz_n=1.0))
        latest = reader.latest()["sample"]
        self.assertAlmostEqual(latest["reference_fz_n"], 0.0, places=6)
        self.assertAlmostEqual(latest["filtered_reference_fz_n"], 0.0, places=6)
        self.assertAlmostEqual(latest["conditioned_reference_fz_n"], 0.0, places=6)

    def test_all_six_axes_are_median_despiked_and_low_pass_filtered(self) -> None:
        reader = Px6dReader(
            {
                "median_window_samples": 5,
                "filter_alpha": 0.25,
                "auto_zero_drift_enabled": False,
            }
        )
        reader._tare_values = (0.0,) * 6
        reader._tare_status = "ready"
        started = time.time()
        for index in range(8):
            reader._append_sample(
                make_sample(
                    index + 1,
                    started + index * 0.02,
                    fx_n=0.0,
                    fy_n=0.0,
                    fz_n=0.0,
                    mx_nm=0.0,
                    my_nm=0.0,
                    mz_nm=0.0,
                )
            )
        reader._append_sample(
            make_sample(
                9,
                started + 0.18,
                fx_n=5.0,
                fy_n=-6.0,
                fz_n=7.0,
                mx_nm=0.5,
                my_nm=-0.6,
                mz_nm=0.7,
            )
        )

        latest = reader.latest()["sample"]
        self.assertEqual(latest["zeroed"]["fx_n"], 5.0)
        self.assertEqual(latest["zeroed"]["mz_nm"], 0.7)
        for axis in ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm"):
            self.assertAlmostEqual(latest["median_zeroed"][axis], 0.0, places=9)
            self.assertAlmostEqual(latest["filtered_zeroed"][axis], 0.0, places=9)
        self.assertAlmostEqual(
            latest["filtered_mechanical"]["force_resultant_n"], 0.0, places=9
        )
        conditioning = reader.status()["force_conditioning"]
        self.assertTrue(conditioning["all_six_axes_filtered"])
        self.assertEqual(conditioning["filtered_axis_names"], list(AXIS_NAMES))

    def test_release_side_drift_reacquires_zero_without_erasing_positive_contact(self) -> None:
        reader = Px6dReader(
            {
                "poll_hz": 20.0,
                "compression_sign": -1,
                "median_window_samples": 5,
                "filter_alpha": 0.35,
                "force_deadband_n": 0.010,
                "stationary_window_sec": 0.50,
                "stationary_std_max_n": 0.020,
                "stationary_range_max_n": 0.060,
                "stationary_slope_max_n_per_sec": 0.20,
                "auto_zero_hold_sec": 0.40,
                "auto_zero_capture_limit_n": 0.060,
                "auto_zero_release_reacquire_limit_n": 0.30,
                "auto_zero_alpha": 0.15,
            }
        )
        reader._tare_values = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        reader._tare_status = "ready"
        started = time.time()

        # A higher raw Fz maps to a negative, release-side compression residual.
        release_tracking_seen = False
        for index in range(80):
            reader._append_sample(
                make_sample(index + 1, started + index * 0.05, fz_n=1.15)
            )
            release_tracking_seen = release_tracking_seen or (
                reader.latest()["sample"]["force_filter_status"]
                == "stationary_release_drift_tracking"
            )
        released = reader.latest()["sample"]
        self.assertLess(released["reference_fz_n"], -0.10)
        self.assertLess(abs(released["conditioned_reference_fz_n"]), 0.025)
        self.assertTrue(release_tracking_seen)
        release_offset = released["drift_offset_n"]

        # A positive compression above the near-zero capture band is preserved.
        for index in range(30):
            reader._append_sample(
                make_sample(81 + index, started + (80 + index) * 0.05, fz_n=0.80)
            )
        contact = reader.latest()["sample"]
        self.assertGreater(contact["conditioned_reference_fz_n"], 0.25)
        self.assertLess(abs(contact["drift_offset_n"] - release_offset), 0.006)
        self.assertEqual(
            contact["force_filter_status"], "contact_or_motion_filter_frozen"
        )

    def test_stale_force_is_not_attached_to_spectrum(self) -> None:
        reader = Px6dReader({"sync_window_sec": 0.01, "sync_max_age_sec": 0.10})
        now = time.time()
        reader._append_sample(make_sample(1, now - 2.0, fz_n=0.80))
        aligned = reader.synchronized_snapshot(now)
        self.assertFalse(aligned["ok"])
        self.assertEqual(aligned["status"], "px6d_sample_too_far_from_spectrum")

    def test_trace_serialization_does_not_block_live_sample_append(self) -> None:
        reader = Px6dReader({"auto_tare_on_start": False})
        started = time.time()
        with reader._lock:
            reader._tare_values = (0.0,) * 6
            reader._tare_status = "ready"
        for index in range(40):
            reader._append_sample(
                make_sample(index + 1, started + index * 0.02, fz_n=0.25)
            )

        serialization_started = threading.Event()
        release_serialization = threading.Event()
        writer_completed = threading.Event()
        original_payload = reader._sample_payload
        trace_result: dict[str, object] = {}
        trace_error: list[BaseException] = []

        def blocking_payload(sample, **kwargs):
            if not serialization_started.is_set():
                serialization_started.set()
                if not release_serialization.wait(timeout=2.0):
                    raise TimeoutError("trace serialization test was not released")
            return original_payload(sample, **kwargs)

        def run_trace() -> None:
            try:
                trace_result.update(reader.trace(limit=40))
            except BaseException as exc:  # pragma: no cover - surfaced below
                trace_error.append(exc)

        def append_live_sample() -> None:
            reader._append_sample(
                make_sample(41, started + 0.82, fz_n=0.30)
            )
            writer_completed.set()

        reader._sample_payload = blocking_payload  # type: ignore[method-assign]
        trace_thread = threading.Thread(target=run_trace)
        writer_thread = threading.Thread(target=append_live_sample)
        trace_thread.start()
        try:
            self.assertTrue(serialization_started.wait(timeout=1.0))
            writer_thread.start()
            self.assertTrue(
                writer_completed.wait(timeout=0.5),
                "trace JSON serialization held the acquisition lock",
            )
        finally:
            release_serialization.set()
            trace_thread.join(timeout=2.0)
            writer_thread.join(timeout=2.0)
            reader._sample_payload = original_payload  # type: ignore[method-assign]

        self.assertFalse(trace_thread.is_alive())
        self.assertFalse(writer_thread.is_alive())
        self.assertEqual(trace_error, [])
        self.assertEqual(trace_result["count"], 40)
        rows = trace_result["samples"]
        self.assertEqual(rows[-1]["sequence_id"], 40)
        self.assertEqual(reader.latest_sequence, 41)

    def test_trace_rows_share_one_tare_metadata_snapshot(self) -> None:
        reader = Px6dReader({"auto_tare_on_start": False})
        started = time.time()
        with reader._lock:
            reader._tare_values = (0.0,) * 6
            reader._tare_status = "ready"
            reader._tare_sample_count = 12
        for index in range(3):
            reader._append_sample(
                make_sample(index + 1, started + index * 0.02, fz_n=0.25)
            )

        serialization_started = threading.Event()
        release_serialization = threading.Event()
        original_payload = reader._sample_payload
        trace_result: dict[str, object] = {}

        def blocking_payload(sample, **kwargs):
            if not serialization_started.is_set():
                serialization_started.set()
                release_serialization.wait(timeout=2.0)
            return original_payload(sample, **kwargs)

        reader._sample_payload = blocking_payload  # type: ignore[method-assign]
        trace_thread = threading.Thread(
            target=lambda: trace_result.update(reader.trace(limit=3))
        )
        trace_thread.start()
        try:
            self.assertTrue(serialization_started.wait(timeout=1.0))
            with reader._lock:
                reader._tare_values = (1.0,) * 6
                reader._tare_status = "replacement_zero"
                reader._tare_sample_count = 99
        finally:
            release_serialization.set()
            trace_thread.join(timeout=2.0)
            reader._sample_payload = original_payload  # type: ignore[method-assign]

        self.assertFalse(trace_thread.is_alive())
        rows = trace_result["samples"]
        self.assertTrue(all(row["tare_status"] == "ready" for row in rows))
        self.assertTrue(all(row["tare_sample_count"] == 12 for row in rows))
        self.assertTrue(
            all(abs(row["zeroed"]["fz_n"] - 0.25) < 1e-9 for row in rows)
        )


class Px6dIntegrationContractTests(unittest.TestCase):
    def test_latest_api_separates_historical_sample_from_live_ready_sample(self) -> None:
        historical = {
            "ok": True,
            "status": {
                "connected": False,
                "sample_fresh": False,
                "last_sample_age_sec": 4.0,
            },
            "sample": {"force_fz_n": 0.72},
        }
        with patch.object(
            backend_main.px6d_reader,
            "latest",
            return_value=historical,
        ):
            result = backend_main.px6d_latest()

        self.assertTrue(result["ok"])
        self.assertTrue(result["sample_present"])
        self.assertFalse(result["sample_fresh"])
        self.assertFalse(result["sample_ready"])

    def test_px6d_api_propagates_start_and_stop_failures(self) -> None:
        with patch.object(
            backend_main.px6d_reader,
            "start",
            return_value={
                "ok": False,
                "operation_status": "dependency_unavailable",
                "running": False,
            },
        ):
            start_result = backend_main.px6d_start()
        with patch.object(
            backend_main.px6d_reader,
            "stop",
            return_value={
                "ok": False,
                "operation_status": "stop_timeout",
                "running": True,
            },
        ):
            stop_result = backend_main.px6d_stop()

        self.assertFalse(start_result["ok"])
        self.assertEqual(
            start_result["operation_status"], "dependency_unavailable"
        )
        self.assertFalse(stop_result["ok"])
        self.assertEqual(stop_result["operation_status"], "stop_timeout")
        self.assertTrue(stop_result["running"])

    def test_environment_can_disable_px6d_autostart_for_safe_secondary_instance(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TOUCH_PX6D_ENABLED": "true",
                "TOUCH_PX6D_AUTO_START": "false",
                "TOUCH_PX6D_PORT": "COM19",
            },
            clear=False,
        ):
            config = backend_main._load_px6d_reference_config()

        self.assertTrue(config["enabled"])
        self.assertFalse(config["auto_start"])
        self.assertEqual(config["port"], "COM19")
        self.assertTrue(config["isolate_process"])
        self.assertGreater(config["worker_watchdog_sec"], 0.0)

    def test_production_px6d_status_exposes_process_isolation_contract(self) -> None:
        reader = Px6dReader(
            {
                "auto_tare_on_start": False,
                "isolate_process": True,
                "worker_watchdog_sec": 1.5,
            }
        )

        status = reader.status()

        self.assertTrue(status["serial_isolation_enabled"])
        self.assertFalse(status["serial_worker_alive"])
        self.assertFalse(status["serial_worker_job_guard_active"])
        self.assertEqual(status["serial_worker_restart_count"], 0)
        self.assertEqual(status["serial_worker_forced_termination_count"], 0)
        self.assertEqual(status["consecutive_connection_failures"], 0)
        self.assertFalse(status["port_busy_detected"])
        self.assertIsNone(status["connection_error_kind"])
        self.assertIsNone(status["next_reconnect_in_sec"])

    def test_px6d_port_follows_exact_usb_identity_after_com_renumber(self) -> None:
        reader = Px6dReader(
            {
                "port": "COM3",
                "auto_detect_port": True,
                "usb_vid": 0x1A86,
                "usb_pid": 0x55D3,
                "usb_serial_number": "5C7B025505",
                "port_description_contains": "CH343",
            }
        )
        detected = SimpleNamespace(
            device="COM7",
            description="USB-Enhanced-SERIAL CH343 (COM7)",
            hwid="USB VID:PID=1A86:55D3 SER=5C7B025505",
            vid=0x1A86,
            pid=0x55D3,
            serial_number="5C7B025505",
            manufacturer="wch.cn",
        )
        with patch(
            "backend.px6d_reader.serial_list_ports.comports",
            return_value=[detected],
        ):
            active_port = reader._resolve_active_port()

        status = reader.status()
        self.assertEqual(active_port, "COM7")
        self.assertEqual(status["configured_port"], "COM3")
        self.assertEqual(status["active_port"], "COM7")
        self.assertEqual(
            status["port_detection_status"],
            "matched_usb_identity_on_new_port",
        )
        self.assertEqual(
            status["detected_device_identity"]["serial_number"],
            "5C7B025505",
        )

    def test_px6d_port_discovery_does_not_open_an_unrelated_serial_device(self) -> None:
        reader = Px6dReader(
            {
                "port": "COM3",
                "auto_detect_port": True,
                "usb_vid": 0x1A86,
                "usb_pid": 0x55D3,
                "usb_serial_number": "5C7B025505",
            }
        )
        unrelated = SimpleNamespace(
            device="COM3",
            description="USB-Enhanced-SERIAL CH343 (COM3)",
            hwid="USB VID:PID=1A86:55D3 SER=OTHER",
            vid=0x1A86,
            pid=0x55D3,
            serial_number="OTHER",
            manufacturer="wch.cn",
        )
        with patch(
            "backend.px6d_reader.serial_list_ports.comports",
            return_value=[unrelated],
        ):
            with self.assertRaises(FileNotFoundError):
                reader._resolve_active_port()

        status = reader.status()
        self.assertIsNone(status["active_port"])
        self.assertEqual(
            status["port_detection_status"],
            "configured_device_identity_not_detected",
        )

    def test_missing_port_is_not_misreported_as_port_busy(self) -> None:
        kind = Px6dReader._classify_connection_error(
            "SerialException: could not open port 'COM3': "
            "FileNotFoundError(2, 'The system cannot find the file specified')"
        )
        self.assertEqual(kind, "port_not_found")

    def test_runtime_status_self_heals_an_unexpected_reader_exit(self) -> None:
        worker_exited = {
            "worker_alive": False,
            "stop_requested": False,
            "lifecycle_status": "worker_exited",
        }
        restarted = {
            "ok": True,
            "worker_alive": True,
            "lifecycle_status": "running",
        }
        with (
            patch.dict(
                backend_main.PX6D_REFERENCE_CONFIG,
                {"enabled": True, "auto_start": True},
            ),
            patch.object(
                backend_main.px6d_reader,
                "status",
                return_value=worker_exited,
            ),
            patch.object(
                backend_main.px6d_reader,
                "start",
                return_value=restarted,
            ) as start_mock,
        ):
            result = backend_main._px6d_runtime_status()

        start_mock.assert_called_once_with()
        self.assertTrue(result["worker_alive"])

    def test_port_busy_failures_use_bounded_backoff_without_changing_api_state(self) -> None:
        reader = Px6dReader(
            {
                "auto_tare_on_start": False,
                "reconnect_interval_sec": 0.5,
                "reconnect_max_interval_sec": 6.0,
                "reconnect_backoff_multiplier": 2.0,
                "port_busy_backoff_sec": 3.0,
            }
        )

        first_delay = reader._record_connection_failure(
            "SerialException: could not open port 'COM3': "
            "PermissionError(13, 'Access is denied')"
        )
        second_delay = reader._record_connection_failure(
            "SerialException: could not open port 'COM3': "
            "PermissionError(13, 'Access is denied')"
        )
        third_delay = reader._record_connection_failure(
            "SerialException: could not open port 'COM3': "
            "PermissionError(13, 'Access is denied')"
        )
        status = reader.status()

        self.assertEqual(first_delay, 3.0)
        self.assertEqual(second_delay, 6.0)
        self.assertEqual(third_delay, 6.0)
        self.assertEqual(status["consecutive_connection_failures"], 3)
        self.assertEqual(
            status["connection_error_kind"], "port_busy_or_permission_denied"
        )
        self.assertTrue(status["port_busy_detected"])
        self.assertEqual(status["reconnect_delay_sec"], 6.0)
        self.assertGreater(status["next_reconnect_in_sec"], 0.0)
        self.assertLessEqual(status["next_reconnect_in_sec"], 6.0)

    def test_valid_sample_resets_reconnect_backoff_immediately(self) -> None:
        reader = Px6dReader(
            {
                "auto_tare_on_start": False,
                "reconnect_interval_sec": 0.5,
                "reconnect_max_interval_sec": 6.0,
                "port_busy_backoff_sec": 3.0,
            }
        )
        reader._record_connection_failure(
            "SerialException: could not open port 'COM3': PermissionError(13)"
        )

        reader._append_sample(make_sample(1, time.time(), fz_n=0.2))
        status = reader.status()

        self.assertEqual(status["consecutive_connection_failures"], 0)
        self.assertEqual(status["reconnect_delay_sec"], 0.0)
        self.assertIsNone(status["next_reconnect_in_sec"])
        self.assertIsNone(status["connection_error_kind"])
        self.assertFalse(status["port_busy_detected"])

    def test_config_and_ui_keep_reference_force_semantics_explicit(self) -> None:
        config_text = (PROJECT_ROOT / "config" / "px6d_reference.yaml").read_text(
            encoding="utf-8"
        )
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        javascript = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        backend = (APP_ROOT / "backend" / "main.py").read_text(encoding="utf-8")

        self.assertIn("PX6D Reference Fz", config_text)
        self.assertIn("Force Sensor", html)
        self.assertIn("Conditioned compression Fz", html)
        self.assertIn("Sensor raw Fz (signed)", html)
        self.assertIn("filtered_zeroed", javascript)
        self.assertIn("filtered_mechanical", javascript)
        self.assertIn("synchronized_ground_truth_reference", config_text)
        self.assertIn("poll_interval_sec: 0.01", config_text)
        self.assertIn("durable_flush_interval_sec: 0.25", config_text)
        self.assertIn("durable_flush_frame_count: 10", config_text)
        self.assertIn("px6dReferenceFz", html)
        self.assertIn("Zero Fz", html)
        self.assertIn("MECHANICAL REFERENCE", html)
        self.assertIn("OPTICAL–MECHANICAL SYNC", html)
        self.assertIn("Data recording", html)
        self.assertIn("diagnosticPx6dFx", html)
        self.assertIn("px6dCaptureStartButton", html)
        self.assertIn("px6dCaptureSpectrum", html)
        self.assertIn("px6dCaptureResponse", html)
        self.assertIn("px6dCaptureForce", html)
        self.assertIn("px6dCaptureOutputRoot", html)
        self.assertIn("px6dCaptureBrowseButton", html)
        self.assertIn("diagnosticPx6dFilterStatus", html)
        self.assertIn("/api/px6d/tare", javascript)
        self.assertIn("/api/px6d/latest", javascript)
        self.assertIn("/api/px6d/reconnect", backend)
        self.assertIn("px6dConnectionDisplayState", javascript)
        self.assertIn("waiting for sensor", javascript)
        self.assertIn("port in use", javascript)
        self.assertIn("/api/px6d_capture/start", javascript)
        self.assertIn("selected_outputs: selectedOutputs", javascript)
        self.assertIn("captured_frame_rate_hz", javascript)
        self.assertIn("Frames / rate", html)
        self.assertIn("output_root: px6dCaptureOutputRoot", javascript)
        self.assertIn("choose_output_directory", javascript)
        self.assertIn("drift_offset_n", javascript)
        self.assertIn("function finiteNumberOrNull(value)", javascript)
        self.assertNotIn("const sampleAge = Number(status?.last_sample_age_sec)", javascript)
        self.assertIn(
            "const liveForceVisible = connected && fresh && Number.isFinite(referenceValue)",
            javascript,
        )
        self.assertIn("const currentForceReady = connected && tareReady && fresh", javascript)
        self.assertIn("const displayedCompressionFz = liveForceVisible", javascript)
        self.assertIn('"px6d_reference": _px6d_reference_for_record', backend)
        self.assertIn("OpticalForceCaptureManager", backend)


if __name__ == "__main__":
    unittest.main()
