from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.optical_force_capture import OpticalForceCaptureManager


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
    timestamp = float(record["ingested_at"])
    return {
        "ok": True,
        "status": "synced",
        "tare_ready": True,
        "sync_method": "window_median",
        "sync_quality": "excellent",
        "sync_offset_ms": 2.5,
        "sample_count": 4,
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
        "reference_fz_n": 0.80,
        "reference_fz_display_n": 0.80,
        "mechanical": {
            "force_resultant_n": 0.808,
            "shear_resultant_n": 0.112,
            "moment_resultant_nm": 0.019,
            "force_utilization_percent": 1.6,
            "moment_utilization_percent": 0.75,
        },
    }


class OpticalForceCaptureTests(unittest.TestCase):
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

    def test_session_writes_full_spectrum_and_flat_six_axis_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
                poll_interval_sec=0.01,
            )
            started = manager.start(
                position_label="P22",
                action_label="normal_press",
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
            self.assertEqual(first["action_label"], "normal_press")
            self.assertEqual(len(first["spectrum"]["wavelength_nm"]), 3)
            self.assertEqual(first["px6d_reference"]["zeroed"]["fz_n"], -0.80)

            with (output_dir / "frame_summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["position_label"], "P22")
            self.assertEqual(row["sync_quality"], "excellent")
            self.assertEqual(float(row["reference_fz_n"]), 0.80)
            self.assertEqual(float(row["fx_zeroed_n"]), 0.05)

            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], "touch_optical_px6d_sync_v1")
            self.assertEqual(metadata["capture_status"], "complete")


if __name__ == "__main__":
    unittest.main()
