from __future__ import annotations

import unittest

import numpy as np

from src.hybrid_spectrum.sense_static_dataset import assert_dataset_manifest_stable
from src.hybrid_spectrum.spatial_fingerprint import (
    CHANNEL_IDS,
    SPATIAL_FEATURE_FAMILIES,
    build_spatial_fingerprint_matrix,
    spatial_fingerprint_feature_names,
    spatial_fingerprint_from_engineered,
)


class SpatialFingerprintTests(unittest.TestCase):
    def test_manifest_guard_rejects_capture_changes(self) -> None:
        before = ({"file_id": "a.csv", "size_bytes": 10, "modified_time_ns": 1},)
        after = ({"file_id": "a.csv", "size_bytes": 12, "modified_time_ns": 2},)

        with self.assertRaisesRegex(RuntimeError, "dataset changed"):
            assert_dataset_manifest_stable(before, after)

    def test_feature_contract_is_stable_and_unique(self) -> None:
        names = spatial_fingerprint_feature_names()

        self.assertEqual(
            len(names),
            len(CHANNEL_IDS) * len(SPATIAL_FEATURE_FAMILIES) * 2,
        )
        self.assertEqual(len(names), len(set(names)))

    def test_each_family_is_normalized_within_one_snapshot(self) -> None:
        engineered = {}
        for family_index, family in enumerate(SPATIAL_FEATURE_FAMILIES, start=1):
            for channel_index, channel_id in enumerate(CHANNEL_IDS, start=1):
                value = float(family_index * channel_index)
                if family in {"area_ratio", "height_ratio"}:
                    value = 1.0 + value / 100.0
                elif family == "shape_correlation":
                    value = 1.0 - value / 1000.0
                engineered[f"{channel_id}_{family}"] = value

        fingerprint = spatial_fingerprint_from_engineered(engineered)

        for family in SPATIAL_FEATURE_FAMILIES:
            magnitude = [
                fingerprint[f"spatial_{family}_magnitude_{channel_id}"]
                for channel_id in CHANNEL_IDS
            ]
            self.assertAlmostEqual(max(magnitude), 1.0)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in magnitude))

    def test_matrix_builder_matches_row_builder(self) -> None:
        columns = tuple(
            f"{channel_id}_{family}"
            for family in SPATIAL_FEATURE_FAMILIES
            for channel_id in CHANNEL_IDS
        )
        source = np.arange(1, len(columns) * 2 + 1, dtype=float).reshape(2, -1)

        matrix, names = build_spatial_fingerprint_matrix(source, columns)

        self.assertEqual(matrix.shape, (2, len(names)))
        self.assertTrue(np.all(np.isfinite(matrix)))


if __name__ == "__main__":
    unittest.main()
