from __future__ import annotations

import unittest

import numpy as np

from src.hybrid_spectrum.sense_static_dataset import (
    BaselineCluster,
    assess_baseline_clusters,
)


class BaselineClusterQualityTests(unittest.TestCase):
    @staticmethod
    def _cluster(
        cluster_id: str,
        epoch: float,
        spectrum: np.ndarray,
        count: int = 8,
    ) -> BaselineCluster:
        frames = np.vstack(
            [
                spectrum
                + 0.2
                * np.sin(np.linspace(0.0, 4.0 * np.pi, spectrum.size) + index * 0.1)
                for index in range(count)
            ]
        )
        return BaselineCluster(
            cluster_id=cluster_id,
            center_epoch=epoch,
            record_ids=tuple(f"{cluster_id}_{index}" for index in range(count)),
            spectra=frames,
        )

    def test_stable_local_residual_is_not_a_training_baseline(self) -> None:
        x = np.linspace(-1.0, 1.0, 128)
        clean = 9000.0 + 3200.0 * np.exp(-0.5 * (x / 0.24) ** 2)
        residual = clean.copy()
        residual[45:78] -= 2600.0
        assessments = assess_baseline_clusters(
            (
                self._cluster("baseline_cluster_01", 0.0, clean),
                self._cluster("baseline_cluster_02", 3600.0, residual),
            ),
            {"session_gap_minutes": 240.0},
        )

        self.assertEqual(assessments[0].status, "trusted_session_anchor")
        self.assertEqual(assessments[1].status, "stable_recovery_residual_biased")
        self.assertFalse(assessments[1].trusted_for_reference)
        self.assertFalse(assessments[1].eligible_for_no_contact_training)

    def test_common_gain_change_remains_trusted(self) -> None:
        x = np.linspace(-1.0, 1.0, 128)
        clean = 8500.0 + 2600.0 * np.exp(-0.5 * (x / 0.28) ** 2)
        assessments = assess_baseline_clusters(
            (
                self._cluster("baseline_cluster_01", 0.0, clean),
                self._cluster("baseline_cluster_02", 3600.0, clean * 1.07),
            ),
            {"session_gap_minutes": 240.0},
        )

        self.assertEqual(assessments[1].status, "trusted_session_consistent")
        self.assertTrue(assessments[1].trusted_for_reference)
        self.assertAlmostEqual(
            assessments[1].common_gain_ratio_to_anchor or 0.0,
            1.07,
            places=3,
        )

    def test_cross_session_shape_change_starts_a_new_anchor(self) -> None:
        x = np.linspace(-1.0, 1.0, 128)
        first = 9000.0 + 2800.0 * np.exp(-0.5 * (x / 0.25) ** 2)
        later = 8200.0 + 3500.0 * np.exp(-0.5 * ((x - 0.08) / 0.31) ** 2)
        assessments = assess_baseline_clusters(
            (
                self._cluster("baseline_cluster_01", 0.0, first),
                self._cluster("baseline_cluster_02", 8.0 * 3600.0, later),
            ),
            {"session_gap_minutes": 240.0},
        )

        self.assertEqual(assessments[1].status, "trusted_session_anchor")
        self.assertEqual(assessments[1].session_id, "baseline_session_02")
        self.assertTrue(assessments[1].trusted_for_reference)


if __name__ == "__main__":
    unittest.main()
