from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, filename: str):
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script_module(
    "prepare_ordinary_fbg_px6d_collection",
    "prepare_ordinary_fbg_px6d_collection.py",
)
validate = load_script_module(
    "validate_ordinary_fbg_px6d_session",
    "validate_ordinary_fbg_px6d_session.py",
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class OrdinaryFbgPx6dCollectionToolingTests(unittest.TestCase):
    def test_manifest_is_balanced_and_avoids_adjacent_position_repeats(self) -> None:
        rows = prepare.build_manifest(
            session_date="20260730",
            formal_repeats=8,
            seed=20260730,
        )
        action_rows = [
            row for row in rows if str(row["position_label"]).startswith("P")
        ]
        counts = prepare._position_counts(rows)

        self.assertEqual(len(rows), 87)
        self.assertEqual(len(action_rows), 81)
        self.assertEqual(set(counts.values()), {9})
        self.assertEqual(
            sum(row["phase"] == "baseline_checkpoint" for row in rows),
            6,
        )
        self.assertTrue(
            all(
                current["position_label"] != previous["position_label"]
                for previous, current in zip(action_rows, action_rows[1:])
            )
        )
        self.assertTrue(
            all(
                row["action_label"] == "continuous_px6d_fz_reference"
                for row in rows
            )
        )

    def test_validator_accepts_a_complete_synchronized_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session_dir = Path(temporary) / "20260730_P22_PILOT"
            session_dir.mkdir()
            frame_count = 20
            metadata = {
                "schema_version": "touch_synchronized_capture_v4",
                "session_id": "synthetic_qa_session",
                "trial_id": "20260730_P22_PILOT",
                "position_label": "P22",
                "action_label": "continuous_px6d_fz_reference",
                "selected_outputs": ["spectrum", "force"],
                "capture_status": "complete",
            }
            (session_dir / "session_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            summary_rows = []
            force_rows = []
            spectrum_rows = []
            for index in range(frame_count):
                timestamp = 1_800_000_000.0 + index * 0.4
                if index < 4 or index >= 16:
                    force_fz_n = 0.005
                elif index < 7:
                    force_fz_n = 0.1 * (index - 3)
                elif index < 13:
                    force_fz_n = 0.4
                else:
                    force_fz_n = 0.4 - 0.1 * (index - 12)
                summary_rows.append(
                    {
                        "capture_index": index,
                        "timeline_timestamp_epoch_sec": timestamp,
                        "elapsed_time_sec": index * 0.4,
                        "position_label": "P22",
                        "trial_id": "20260730_P22_PILOT",
                    }
                )
                force_row: dict[str, object] = {
                    "capture_index": index,
                    "force_timestamp_epoch_sec": timestamp + 0.01,
                    "sync_offset_ms": 10.0,
                    "calibration_sync_ok": "true",
                    "force_fz_n": force_fz_n,
                }
                for field in validate.AXIS_FIELDS:
                    force_row[field] = (
                        -force_fz_n if field == "fz_zeroed_n" else force_fz_n
                    )
                force_rows.append(force_row)
                for point_index in range(512):
                    spectrum_rows.append(
                        {
                            "capture_index": index,
                            "point_index": point_index,
                            "wavelength_nm": 1539.0 + point_index * (43.0 / 511.0),
                            "intensity_counts": 1000.0 + point_index + force_fz_n,
                        }
                    )

            write_csv(
                session_dir / "frame_summary.csv",
                list(summary_rows[0]),
                summary_rows,
            )
            write_csv(
                session_dir / "force_timeseries.csv",
                list(force_rows[0]),
                force_rows,
            )
            write_csv(
                session_dir / "spectrum_timeseries.csv",
                list(spectrum_rows[0]),
                spectrum_rows,
            )

            result = validate.audit_session(
                session_dir,
                minimum_frames=20,
                maximum_sync_offset_ms=250.0,
            )

        self.assertEqual(result["qa_status"], "pass")
        self.assertEqual(result["formal_group_id"], "synthetic_qa_session")
        self.assertEqual(
            result["position_trial_key"],
            "P22:20260730_P22_PILOT",
        )
        self.assertEqual(result["frame_count"], frame_count)
        self.assertEqual(result["spectrum_frame_count"], frame_count)
        self.assertEqual(result["minimum_spectrum_points"], 512)
        self.assertEqual(result["valid_force_ratio"], 1.0)
        self.assertEqual(result["sync_pass_ratio"], 1.0)
        self.assertEqual(result["maximum_observed_sync_offset_ms"], 10.0)
        self.assertEqual(result["findings"], [])


if __name__ == "__main__":
    unittest.main()
