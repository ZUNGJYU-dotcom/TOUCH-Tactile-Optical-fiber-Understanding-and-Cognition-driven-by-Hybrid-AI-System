from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_live_fbg_cluster import select_nine_peak_cluster  # noqa: E402


class LiveFbgClusterDiscoveryTests(unittest.TestCase):
    def test_selects_realistic_nine_peak_cluster_and_rejects_broad_background(self) -> None:
        wavelength = np.linspace(1523.0, 1607.0, 841)
        baseline = 7000.0 + 2500.0 * np.exp(-((wavelength - 1548.0) / 22.0) ** 2)
        spectrum = baseline.copy()
        expected = np.asarray([1528.0, 1532.1, 1536.2, 1540.1, 1544.3, 1547.8, 1551.7, 1555.8, 1559.8])
        for index, center in enumerate(expected):
            amplitude = 26000.0 + (index % 3) * 5000.0
            spectrum += amplitude * np.exp(-0.5 * ((wavelength - center) / 0.32) ** 2)
        spectrum += 8500.0 * np.exp(-0.5 * ((wavelength - 1572.0) / 2.4) ** 2)
        spectrum += 6000.0 * np.exp(-0.5 * ((wavelength - 1590.0) / 3.2) ** 2)

        indices, helper = select_nine_peak_cluster(wavelength, spectrum, 1526.5, 1561.5)
        discovered = wavelength[indices]

        self.assertEqual(len(discovered), 9)
        self.assertTrue(np.allclose(discovered, expected, atol=0.12))
        self.assertLess(helper["audit"]["cluster_spacing_cv"], 0.12)
        self.assertLess(float(np.max(discovered)), 1561.5)

    def test_rejects_incomplete_cluster(self) -> None:
        wavelength = np.linspace(1523.0, 1607.0, 841)
        spectrum = np.full_like(wavelength, 7000.0)
        for center in [1528.0, 1532.0, 1536.0, 1540.0, 1544.0, 1548.0, 1552.0, 1556.0]:
            spectrum += 30000.0 * np.exp(-0.5 * ((wavelength - center) / 0.32) ** 2)

        with self.assertRaisesRegex(ValueError, "Only 8 narrow stable candidates"):
            select_nine_peak_cluster(wavelength, spectrum, 1526.5, 1561.5)

    def test_first_peak_anchor_prevents_later_cluster_shift(self) -> None:
        wavelength = np.linspace(1523.0, 1607.0, 4097)
        spectrum = 7000.0 + 900.0 * np.sin((wavelength - 1523.0) / 8.0)
        real_centers = np.asarray([1527.8, 1532.0, 1536.2, 1540.1, 1544.3, 1547.8, 1551.7, 1555.8, 1559.9])
        for center in real_centers:
            spectrum += 26000.0 * np.exp(-0.5 * ((wavelength - center) / 0.24) ** 2)
        spectrum += 52000.0 * np.exp(-0.5 * ((wavelength - 1564.1) / 0.24) ** 2)

        indices, helper = select_nine_peak_cluster(
            wavelength,
            spectrum,
            1526.5,
            1565.0,
            expected_first_peak_nm=1528.0,
            first_peak_tolerance_nm=1.5,
        )

        discovered = wavelength[indices]
        self.assertAlmostEqual(float(discovered[0]), 1527.8, delta=0.2)
        self.assertLess(helper["audit"]["selected_first_peak_offset_nm"], 0.5)


if __name__ == "__main__":
    unittest.main()
