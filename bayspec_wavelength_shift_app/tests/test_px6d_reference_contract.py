from __future__ import annotations

import struct
import sys
import time
from pathlib import Path
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.px6d_reader import (
    Px6dReader,
    Px6dSample,
    command_packet,
    crc8,
    parse_data_frame,
)


def make_sample(sequence_id: int, timestamp: float, *, fz_n: float) -> Px6dSample:
    return Px6dSample(
        sequence_id=sequence_id,
        timestamp_epoch_sec=timestamp,
        timestamp_monotonic_sec=timestamp,
        fx_n=0.01,
        fy_n=-0.02,
        fz_n=fz_n,
        mx_nm=0.001,
        my_nm=-0.002,
        mz_nm=0.003,
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
        reader._append_sample(make_sample(21, now + 0.05, fz_n=0.30))

        latest = reader.latest()["sample"]
        self.assertAlmostEqual(latest["raw"]["fz_n"], 0.30, places=6)
        self.assertAlmostEqual(latest["zeroed"]["fz_n"], -0.50, places=6)
        self.assertAlmostEqual(latest["reference_fz_n"], 0.50, places=6)
        self.assertAlmostEqual(latest["reference_fz_display_n"], 0.50, places=6)
        self.assertTrue(latest["tare_ready"])
        self.assertAlmostEqual(latest["mechanical"]["force_resultant_n"], 0.50, delta=0.03)
        self.assertIn("shear_resultant_n", latest["mechanical"])
        self.assertEqual(latest["mechanical"]["utilization_status"], "ok")
        self.assertIsNotNone(latest["tare_fz_std_n"])

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
        self.assertEqual(aligned["sync_quality"], "excellent")
        self.assertTrue(aligned["sync_within_target"])
        self.assertEqual(aligned["force_sequence_start"], 1)
        self.assertEqual(aligned["force_sequence_end"], 3)
        self.assertEqual(
            aligned["semantics"], "PX6D_reference_Fz_not_optical_force_prediction"
        )

    def test_stale_force_is_not_attached_to_spectrum(self) -> None:
        reader = Px6dReader({"sync_window_sec": 0.01, "sync_max_age_sec": 0.10})
        now = time.time()
        reader._append_sample(make_sample(1, now - 2.0, fz_n=0.80))
        aligned = reader.synchronized_snapshot(now)
        self.assertFalse(aligned["ok"])
        self.assertEqual(aligned["status"], "px6d_sample_too_far_from_spectrum")


class Px6dIntegrationContractTests(unittest.TestCase):
    def test_config_and_ui_keep_reference_force_semantics_explicit(self) -> None:
        config_text = (PROJECT_ROOT / "config" / "px6d_reference.yaml").read_text(
            encoding="utf-8"
        )
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        javascript = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        backend = (APP_ROOT / "backend" / "main.py").read_text(encoding="utf-8")

        self.assertIn("PX6D Reference Fz", config_text)
        self.assertIn("synchronized_ground_truth_reference", config_text)
        self.assertIn("px6dReferenceFz", html)
        self.assertIn("Zero Fz", html)
        self.assertIn("MECHANICAL REFERENCE", html)
        self.assertIn("OPTICAL–MECHANICAL SYNC", html)
        self.assertIn("SYNCHRONIZED DATA RECORDING", html)
        self.assertIn("diagnosticPx6dFx", html)
        self.assertIn("px6dCaptureStartButton", html)
        self.assertIn("px6dCaptureSpectrum", html)
        self.assertIn("px6dCaptureResponse", html)
        self.assertIn("px6dCaptureForce", html)
        self.assertIn("px6dCaptureOutputRoot", html)
        self.assertIn("px6dCaptureBrowseButton", html)
        self.assertIn("/api/px6d/tare", javascript)
        self.assertIn("/api/px6d/latest", javascript)
        self.assertIn("/api/px6d_capture/start", javascript)
        self.assertIn("selected_outputs: selectedOutputs", javascript)
        self.assertIn("output_root: px6dCaptureOutputRoot", javascript)
        self.assertIn("choose_output_directory", javascript)
        self.assertIn('"px6d_reference": _px6d_reference_for_record', backend)
        self.assertIn("OpticalForceCaptureManager", backend)


if __name__ == "__main__":
    unittest.main()
