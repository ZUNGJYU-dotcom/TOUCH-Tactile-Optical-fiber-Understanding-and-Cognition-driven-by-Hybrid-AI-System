from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from src.hybrid_spectrum.live_cadence_dataset import (
    build_live_cadence_dataset,
    causal_summary_features,
)
from src.hybrid_spectrum.live_cadence_models import (
    CumulativeOrdinalClassifier,
    OrdinalRegressionClassifier,
    SpatialCoordinateRegressorClassifier,
)


class LiveCadenceDatasetTests(unittest.TestCase):
    def _audit_dir(self, root: Path) -> Path:
        audit = root / "audit"
        audit.mkdir()
        file_ids = np.asarray(["G1/P22.dat", "G2/P22.dat"])
        groups = np.asarray(["G1", "G2"])
        positions = np.asarray(["P22", "P22"])
        frames = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
        file_indices = np.repeat(np.arange(2), 20)
        frame_indices = np.tile(np.arange(20), 2)
        np.savez_compressed(
            audit / "dynamic_frame_features.npz",
            X_frames=frames,
            file_indices=file_indices,
            frame_indices=frame_indices,
            feature_names=np.asarray(["a", "b", "c"]),
            file_ids=file_ids,
            capture_groups=groups,
            position_labels=positions,
        )
        with (audit / "dynamic_frame_labels.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_id",
                    "capture_group",
                    "position_label",
                    "frame_index",
                    "time_sec_estimated",
                    "stage_label",
                    "stable_training_frame",
                    "response_score_labeling_only",
                ],
            )
            writer.writeheader()
            for file_index, file_id in enumerate(file_ids):
                for frame in range(20):
                    stage = "no_contact" if frame < 10 else "light"
                    writer.writerow(
                        {
                            "file_id": file_id,
                            "capture_group": groups[file_index],
                            "position_label": positions[file_index],
                            "frame_index": frame,
                            "time_sec_estimated": frame * 0.1,
                            "stage_label": stage,
                            "stable_training_frame": frame in {4, 8, 12, 16},
                            "response_score_labeling_only": 0.0,
                        }
                    )
        return audit

    def test_uses_only_real_causal_frames_at_live_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit = self._audit_dir(Path(temporary))
            dataset = build_live_cadence_dataset(
                audit,
                history_frames=2,
                source_frame_interval_sec=0.1,
                live_frame_interval_sec=0.4,
            )
        self.assertEqual(dataset.cadence_factor, 4)
        self.assertEqual(dataset.values.shape[1:], (2, 3))
        self.assertTrue(np.all(np.diff(dataset.source_frame_indices, axis=1) == 4))
        self.assertTrue(
            np.all(dataset.source_frame_indices[:, -1] == dataset.target_frame_indices)
        )
        self.assertEqual(set(dataset.capture_groups.tolist()), {"G1", "G2"})
        self.assertEqual(dataset.history_span_sec, 0.4)
        self.assertEqual(dataset.cold_start_fill_sec, 0.8)

    def test_one_frame_summary_has_established_shape(self) -> None:
        values = np.ones((5, 1, 3), dtype=np.float32)
        summary = causal_summary_features(values)
        self.assertEqual(summary.shape, (5, 36))
        self.assertTrue(np.all(np.isfinite(summary)))

    def test_cumulative_ordinal_probabilities_are_valid(self) -> None:
        rng = np.random.default_rng(42)
        x = np.concatenate(
            [rng.normal(level, 0.1, size=(20, 2)) for level in (0.0, 1.0, 2.0)]
        )
        y = np.repeat(["light", "normal", "hard"], 20)
        model = CumulativeOrdinalClassifier().fit(x, y)
        probability = model.predict_proba(x)
        self.assertEqual(probability.shape, (60, 3))
        np.testing.assert_allclose(np.sum(probability, axis=1), 1.0)
        self.assertGreater(np.mean(model.predict(x) == y), 0.9)

    def test_ordinal_regression_preserves_response_order(self) -> None:
        rng = np.random.default_rng(12)
        x = np.concatenate(
            [rng.normal(level, 0.08, size=(20, 2)) for level in (0.0, 1.0, 2.0)]
        )
        y = np.repeat(["light", "normal", "hard"], 20)
        model = OrdinalRegressionClassifier(
            ExtraTreesRegressor(n_estimators=40, random_state=1)
        ).fit(x, y)
        self.assertGreater(np.mean(model.predict(x) == y), 0.95)
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1.0)

    def test_spatial_coordinate_regression_returns_valid_pxy(self) -> None:
        rng = np.random.default_rng(4)
        labels = np.asarray(
            ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
        )
        coordinates = np.asarray([[int(v[1]), int(v[2])] for v in labels])
        x = np.repeat(coordinates, 8, axis=0) + rng.normal(0.0, 0.03, (72, 2))
        y = np.repeat(labels, 8)
        model = SpatialCoordinateRegressorClassifier(
            ExtraTreesRegressor(n_estimators=40, random_state=2)
        ).fit(x, y)
        self.assertGreater(np.mean(model.predict(x) == y), 0.95)
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
