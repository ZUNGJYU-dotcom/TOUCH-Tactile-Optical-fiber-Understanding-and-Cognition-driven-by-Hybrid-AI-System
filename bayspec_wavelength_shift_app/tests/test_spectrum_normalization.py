from __future__ import annotations

import unittest

from bridge import normalize_spectrum_to_baseline_ratio


class SpectrumNormalizationTests(unittest.TestCase):
    def test_exact_grid_uses_pointwise_no_contact_ratio(self) -> None:
        result = normalize_spectrum_to_baseline_ratio(
            [1540.0, 1541.0, 1542.0],
            [50.0, 200.0, 450.0],
            [1540.0, 1541.0, 1542.0],
            [100.0, 200.0, 300.0],
            minimum_reference_counts=10.0,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["normalized_intensity_ratio"],
            [0.5, 1.0, 1.5],
        )

    def test_reference_is_interpolated_to_current_wavelength_axis(self) -> None:
        result = normalize_spectrum_to_baseline_ratio(
            [1540.5, 1541.5],
            [150.0, 250.0],
            [1540.0, 1541.0, 1542.0],
            [100.0, 200.0, 300.0],
            minimum_reference_counts=10.0,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["normalization_reference_intensity_counts"],
            [150.0, 250.0],
        )
        self.assertEqual(result["normalized_intensity_ratio"], [1.0, 1.0])

    def test_low_reference_regions_are_not_divided(self) -> None:
        result = normalize_spectrum_to_baseline_ratio(
            [1540.0, 1541.0, 1542.0],
            [5.0, 200.0, 300.0],
            [1540.0, 1541.0, 1542.0],
            [5.0, 200.0, 300.0],
            minimum_reference_counts=100.0,
            minimum_valid_fraction=0.80,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["status"],
            "insufficient_valid_reference_points",
        )

    def test_method_is_not_per_frame_min_max(self) -> None:
        baseline = [100.0, 200.0, 400.0]
        result = normalize_spectrum_to_baseline_ratio(
            [1540.0, 1541.0, 1542.0],
            [200.0, 400.0, 800.0],
            [1540.0, 1541.0, 1542.0],
            baseline,
            minimum_reference_counts=10.0,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["normalized_intensity_ratio"], [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
