from __future__ import annotations

import unittest

from src.hybrid_spectrum.session_level_calibration import (
    CORE_FEATURE_NAMES,
    PerPositionOrdinalCalibrator,
    extract_response_core_features,
)


def feature_row(value: float) -> dict[str, float]:
    return {name: value for name in CORE_FEATURE_NAMES}


class SessionLevelCalibrationTests(unittest.TestCase):
    def test_extracts_four_core_features(self) -> None:
        engineered = {
            **{
                f"fbg{index:02d}_fused_common_mode_corrected_shift_pm": (-1) ** index * index
                for index in range(1, 10)
            },
            "global_normalized_residual_peak": 0.4,
            "global_normalized_residual_rms": 0.2,
        }

        result = extract_response_core_features(engineered)

        self.assertEqual(set(result), set(CORE_FEATURE_NAMES))
        self.assertEqual(result["shift_abs_max_pm"], 9.0)
        self.assertEqual(result["shift_abs_mean_pm"], 5.0)

    def test_position_specific_ordinal_prediction_and_baseline_guard(self) -> None:
        samples = []
        for level, value in (("light", 1.0), ("normal", 2.0), ("hard", 4.0)):
            samples.append(
                {
                    "position": "P22",
                    "level": level,
                    "features": feature_row(value),
                }
            )
        model = PerPositionOrdinalCalibrator.fit(
            samples,
            baseline_token="baseline-a",
            required_positions=("P22",),
        )

        normal = model.predict(
            "P22",
            feature_row(2.2),
            baseline_token="baseline-a",
        )
        mismatch = model.predict(
            "P22",
            feature_row(2.2),
            baseline_token="baseline-b",
        )

        self.assertTrue(normal["ok"])
        self.assertEqual(normal["label"], "normal")
        self.assertEqual(normal["force_semantics"], "approximate_manual_response_level_not_force_N")
        self.assertFalse(mismatch["ok"])
        self.assertEqual(mismatch["status"], "baseline_mismatch_calibration_invalidated")

    def test_serialization_and_nonmonotonic_warning(self) -> None:
        samples = []
        for level, value in (("light", 1.0), ("normal", 4.0), ("hard", 3.0)):
            samples.append(
                {
                    "position": "P22",
                    "level": level,
                    "features": feature_row(value),
                }
            )
        model = PerPositionOrdinalCalibrator.fit(
            samples,
            baseline_token="baseline-a",
            required_positions=("P22",),
        )
        restored = PerPositionOrdinalCalibrator.from_dict(model.to_dict())

        self.assertEqual(
            restored.quality["P22"]["status"],
            "ready_with_monotonicity_warning",
        )
        self.assertEqual(restored.baseline_token, "baseline-a")


if __name__ == "__main__":
    unittest.main()
