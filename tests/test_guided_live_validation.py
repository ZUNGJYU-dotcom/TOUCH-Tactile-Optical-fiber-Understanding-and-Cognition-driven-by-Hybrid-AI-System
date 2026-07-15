from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from src.hybrid_spectrum.guided_live_validation import (
    build_trial_plan,
    fit_session_calibrator_from_guided_records,
    summarize_guided_records,
    write_guided_validation_artifacts,
)


class GuidedLiveValidationTests(unittest.TestCase):
    def test_balanced_plan_contains_every_position_level_pair(self) -> None:
        plan = build_trial_plan(repeats=2, order="randomized", seed=9)

        self.assertEqual(len(plan), 54)
        pairs = {
            (row["repeat"], row["position"], row["force_level"]) for row in plan
        }
        self.assertEqual(len(pairs), 54)
        self.assertEqual(plan[0]["trial_index"], 1)
        self.assertEqual(
            sum(row["trial_role"] == "calibration" for row in plan),
            27,
        )
        self.assertEqual(
            sum(row["trial_role"] == "validation" for row in plan),
            27,
        )

    def test_summary_separates_contact_accuracy_and_release_residual(self) -> None:
        rows = [
            {
                "phase": "contact",
                "trial_id": "R01_P22_normal",
                "primary_position_correct": False,
                "shadow_position_correct": True,
                "shadow_temporal_ready": True,
                "shadow_temporal_position_correct": True,
                "primary_force_correct": True,
                "shadow_force_correct": True,
                "shadow_temporal_force_correct": True,
                "expected_position": "P22",
                "expected_force": "normal",
                "shadow_position": "P22",
                "shadow_temporal_position": "P22",
                "shadow_force": "normal",
                "shadow_temporal_force": "normal",
            },
            {
                "phase": "post_release",
                "trial_id": "R01_P22_normal",
                "primary_contact": "contact",
                "shadow_contact": "no_contact",
                "shadow_temporal_contact": "no_contact",
            },
        ]

        summary = summarize_guided_records(rows)

        self.assertEqual(summary["captured_trial_count"], 1)
        self.assertEqual(summary["shadow_position_accuracy"], 1.0)
        self.assertEqual(summary["primary_release_false_contact_rate"], 1.0)
        self.assertEqual(summary["shadow_temporal_release_false_contact_rate"], 0.0)

    def test_plan_only_artifacts_are_valid(self) -> None:
        plan = build_trial_plan(
            repeats=1,
            positions=("P22",),
            levels=("light", "hard"),
            order="blocked",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = write_guided_validation_artifacts(
                output,
                trial_plan=plan,
                records=[],
                run_metadata={"mode": "plan_only"},
            )

            self.assertEqual(summary["planned_trial_count"], 2)
            self.assertTrue((output / "guided_trial_plan.csv").exists())
            self.assertTrue((output / "guided_validation_summary.json").exists())
            self.assertTrue((output / "guided_live_validation_report.md").exists())

    def test_completed_trials_create_baseline_bound_calibration(self) -> None:
        rows = []
        for level, value in (("light", 1.0), ("normal", 2.0), ("hard", 4.0)):
            for frame in range(3):
                rows.append(
                    {
                        "phase": "contact",
                        "trial_id": f"R01_P22_{level}",
                        "expected_position": "P22",
                        "expected_force": level,
                        "baseline_spectrum_token": "baseline-a",
                        "shadow_temporal_ready": True,
                        "shadow_contact": "contact",
                        "shadow_response_shift_abs_mean_pm": value + frame * 0.01,
                        "shadow_response_shift_abs_max_pm": value + frame * 0.01,
                        "shadow_response_normalized_residual_peak": value + frame * 0.01,
                        "shadow_response_normalized_residual_rms": value + frame * 0.01,
                    }
                )

        result = fit_session_calibrator_from_guided_records(
            rows,
            required_positions=("P22",),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["trial_sample_count"], 3)
        self.assertEqual(
            result["calibration_payload"]["baseline_token"],
            "baseline-a",
        )

    def test_multiple_baselines_reject_calibration(self) -> None:
        rows = [
            {"phase": "contact", "baseline_spectrum_token": "a"},
            {"phase": "contact", "baseline_spectrum_token": "b"},
        ]

        result = fit_session_calibrator_from_guided_records(
            rows,
            required_positions=("P22",),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "multiple_baselines_calibration_invalid")


if __name__ == "__main__":
    unittest.main()
