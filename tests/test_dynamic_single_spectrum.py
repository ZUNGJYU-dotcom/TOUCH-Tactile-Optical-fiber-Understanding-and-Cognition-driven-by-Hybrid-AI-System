from __future__ import annotations

import unittest

import numpy as np

from src.hybrid_spectrum.dynamic_single_spectrum import (
    baseline_relative_spectral_views,
    stable_live_frame_indices,
)


class DynamicSingleSpectrumTests(unittest.TestCase):
    def test_stable_indices_respect_half_open_segment(self) -> None:
        indices = stable_live_frame_indices(5, 31, 10)
        np.testing.assert_array_equal(indices, np.asarray([5, 15, 25, 30]))
        self.assertTrue(np.all(indices >= 5))
        self.assertTrue(np.all(indices < 31))

    def test_spectral_views_are_zero_at_baseline(self) -> None:
        baseline = np.linspace(100.0, 200.0, 32)
        views = baseline_relative_spectral_views(baseline, baseline)
        self.assertEqual(views.shape, (1, 3, 32))
        np.testing.assert_allclose(views, 0.0, atol=1.0e-7)

    def test_log_ratio_preserves_global_intensity_change(self) -> None:
        baseline = np.linspace(100.0, 200.0, 32)
        views = baseline_relative_spectral_views(0.8 * baseline, baseline)
        self.assertTrue(np.all(views[0, 0] < 0.0))
        np.testing.assert_allclose(views[0, 1], 0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
