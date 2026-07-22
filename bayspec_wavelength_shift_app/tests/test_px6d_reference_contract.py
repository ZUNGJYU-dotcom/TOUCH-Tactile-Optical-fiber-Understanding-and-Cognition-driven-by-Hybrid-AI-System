from __future__ import annotations

import struct
import sys
import threading
import time
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
        self.assertEqual(reader.trace()["count"], 0)

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


class Px6dIntegrationContractTests(unittest.TestCase):
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
        self.assertIn("/api/px6d_capture/start", javascript)
        self.assertIn("selected_outputs: selectedOutputs", javascript)
        self.assertIn("output_root: px6dCaptureOutputRoot", javascript)
        self.assertIn("choose_output_directory", javascript)
        self.assertIn("drift_offset_n", javascript)
        self.assertIn("function finiteNumberOrNull(value)", javascript)
        self.assertNotIn("const sampleAge = Number(status?.last_sample_age_sec)", javascript)
        self.assertIn('"px6d_reference": _px6d_reference_for_record', backend)
        self.assertIn("OpticalForceCaptureManager", backend)


if __name__ == "__main__":
    unittest.main()
