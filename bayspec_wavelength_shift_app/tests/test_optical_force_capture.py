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
    timestamp = float(record["ingested_at"]) if isinstance(record, dict) else time.time()
    return {
        "ok": True,
        "status": "synced",
        "tare_ready": True,
        "sync_method": "window_median",
        "sync_quality": "excellent",
        "sync_offset_ms": 2.5,
        "sample_count": 4,
        "force_sequence_start": int(timestamp * 1000),
        "force_sequence_end": int(timestamp * 1000),
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


def model_response(record: dict) -> dict:
    return {
        "model_source": "dynamic_temporal_v3_test",
        "model_status": "shadow_ready",
        "model_ready": True,
        "contact": {
            "label": "contact",
            "confidence": 0.91,
            "probabilities": {"no_contact": 0.09, "contact": 0.91},
        },
        "position": {"label": "P22", "confidence": 0.82},
        "response_level": {
            "label": "normal",
            "confidence": 0.75,
            "raw_label": "normal",
            "decision_rule": "test",
            "probabilities": {"light": 0.15, "normal": 0.75, "hard": 0.10},
        },
        "operational_state": "active_contact",
        "release_guard": {"release_latched": False},
        "runtime_baseline_revision": 2,
    }


def unique_timeline(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["capture_index"]): (
            float(row["timeline_timestamp_epoch_sec"]),
            float(row["elapsed_time_sec"]),
        )
        for row in rows
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
                model_provider=model_response,
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
            self.assertEqual(row["predicted_position_label"], "P22")
            self.assertEqual(row["predicted_response_level"], "normal")

            spectrum_path = output_dir / "spectrum_timeseries.csv"
            response_path = output_dir / "tactile_response_timeseries.csv"
            force_path = output_dir / "force_timeseries.csv"
            self.assertTrue(spectrum_path.exists())
            self.assertTrue(response_path.exists())
            self.assertTrue(force_path.exists())
            spectrum_timeline = unique_timeline(spectrum_path)
            response_timeline = unique_timeline(response_path)
            force_timeline = unique_timeline(force_path)
            self.assertEqual(spectrum_timeline, response_timeline)
            self.assertEqual(spectrum_timeline, force_timeline)

            with response_path.open(encoding="utf-8-sig", newline="") as handle:
                response_row = next(csv.DictReader(handle))
            self.assertEqual(response_row["response_level"], "normal")
            self.assertAlmostEqual(float(response_row["normal_probability"]), 0.75)

            metadata = json.loads(
                (output_dir / "session_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], "touch_synchronized_capture_v2")
            self.assertEqual(metadata["capture_status"], "complete")
            self.assertTrue(metadata["alignment_audit"]["all_selected_streams_aligned"])

    def test_all_seven_nonempty_output_selections_write_only_requested_csvs(self) -> None:
        combinations = [
            ["spectrum"],
            ["response"],
            ["force"],
            ["spectrum", "response"],
            ["spectrum", "force"],
            ["response", "force"],
            ["spectrum", "response", "force"],
        ]
        filenames = {
            "spectrum": "spectrum_timeseries.csv",
            "response": "tactile_response_timeseries.csv",
            "force": "force_timeseries.csv",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, selection in enumerate(combinations):
                manager = OpticalForceCaptureManager(
                    output_root=root / f"default_{index}",
                    frame_provider=FakeSpectrumSource().frame,
                    force_provider=aligned_force,
                    force_status_provider=lambda: {"connected": True, "tare_ready": True},
                    model_provider=model_response,
                    poll_interval_sec=0.01,
                )
                started = manager.start(
                    selected_outputs=selection,
                    output_root=root / f"chosen_{index}",
                    trial_id=f"combo_{index}",
                )
                self.assertTrue(started["ok"], selection)
                time.sleep(0.045)
                stopped = manager.stop()
                self.assertGreater(stopped["captured_timeline_frames"], 0, selection)
                output_dir = Path(stopped["output_directory"])
                self.assertEqual(output_dir.parent, (root / f"chosen_{index}").resolve())
                for stream, filename in filenames.items():
                    self.assertEqual((output_dir / filename).exists(), stream in selection)
                metadata = json.loads(
                    (output_dir / "session_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["selected_outputs"], selection)
                self.assertEqual(
                    metadata["timeline_basis"],
                    "px6d_force_host_epoch_clock"
                    if selection == ["force"]
                    else "spectrum_ingested_at_host_epoch_clock",
                )
                self.assertTrue(
                    metadata["alignment_audit"]["all_selected_streams_aligned"],
                    metadata["alignment_audit"],
                )

    def test_rejects_empty_selection_and_unwritable_output_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = OpticalForceCaptureManager(
                output_root=root,
                frame_provider=FakeSpectrumSource().frame,
                force_provider=aligned_force,
                force_status_provider=lambda: {"connected": True, "tare_ready": True},
            )
            empty = manager.start(selected_outputs=[])
            self.assertFalse(empty["ok"])
            self.assertEqual(empty["status"], "capture_output_selection_invalid")

            not_a_directory = root / "file.txt"
            not_a_directory.write_text("occupied", encoding="utf-8")
            invalid = manager.start(
                selected_outputs=["spectrum"],
                output_root=not_a_directory,
            )
            self.assertFalse(invalid["ok"])
            self.assertEqual(invalid["status"], "capture_output_directory_invalid")

    def test_spectrum_only_does_not_require_px6d_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = OpticalForceCaptureManager(
                output_root=Path(temporary),
                frame_provider=FakeSpectrumSource().frame,
                force_provider=lambda _: {"ok": False},
                force_status_provider=lambda: {"connected": False, "tare_ready": False},
                poll_interval_sec=0.01,
            )
            started = manager.start(selected_outputs=["spectrum"])
            self.assertTrue(started["ok"])
            time.sleep(0.035)
            stopped = manager.stop()
            self.assertGreater(stopped["captured_timeline_frames"], 0)


if __name__ == "__main__":
    unittest.main()
