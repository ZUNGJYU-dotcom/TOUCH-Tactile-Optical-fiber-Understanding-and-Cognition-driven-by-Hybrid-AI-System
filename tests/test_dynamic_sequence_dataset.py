from __future__ import annotations

import unittest

import numpy as np

from src.hybrid_spectrum.dynamic_sequence_dataset import (
    DynamicFeatureSequence,
    DynamicSequenceRecord,
    build_dynamic_window_dataset,
    segment_press_sequence,
)
from src.hybrid_spectrum.sense_fast_dat import FastDatLayout


def _config() -> dict:
    return {
        "segmentation": {
            "minimum_segment_frames": 20,
            "minimum_release_fraction": 0.10,
            "release_search_start_fraction": 0.50,
            "release_drop_min_absolute": 0.12,
            "release_drop_min_hard_fraction": 0.20,
            "recovered_tail_max_hard_fraction": 0.35,
            "stable_trim_fraction": 0.18,
            "transition_guard_frames": 5,
            "minimum_stable_frames": 12,
            "transition_gain_window_fraction": 0.02,
            "transition_gain_min_window_frames": 10,
            "transition_candidate_anchor_fractions": [0.22, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        }
    }


class DynamicSequenceSegmentationTests(unittest.TestCase):
    def test_segments_ordered_press_and_recovered_release(self) -> None:
        rng = np.random.default_rng(42)
        levels = [0.01, 0.18, 0.45, 0.95, 0.04]
        lengths = [80, 65, 70, 75, 90]
        response = np.concatenate(
            [level + rng.normal(0.0, 0.008, length) for level, length in zip(levels, lengths)]
        )
        segments, observed, recovered, ratio, status, flags = segment_press_sequence(
            response,
            _config(),
        )
        self.assertEqual([segment.label for segment in segments], ["no_contact", "light", "normal", "hard", "release"])
        expected = np.cumsum(lengths)[:-1]
        actual = [segment.end_frame for segment in segments[:-1]]
        for measured, target in zip(actual, expected):
            self.assertAlmostEqual(measured, int(target), delta=15)
        self.assertTrue(observed)
        self.assertTrue(recovered)
        self.assertLess(ratio, 0.10)
        self.assertEqual(status, "good_sequence")
        self.assertEqual(flags, ())

    def test_release_residual_is_not_training_eligible_no_contact(self) -> None:
        response = np.concatenate(
            [
                np.full(70, 0.01),
                np.full(70, 0.20),
                np.full(70, 0.48),
                np.full(70, 0.95),
                np.full(70, 0.72),
            ]
        )
        segments, _, recovered, _, status, flags = segment_press_sequence(
            response,
            _config(),
        )
        self.assertFalse(recovered)
        self.assertFalse(segments[-1].training_eligible)
        self.assertIn("release_residual_above_baseline", flags)
        self.assertEqual(status, "usable_with_warning")

    def test_weak_light_plateau_is_not_absorbed_into_no_contact(self) -> None:
        rng = np.random.default_rng(7)
        lengths = [166, 70, 73, 88, 100]
        levels = [0.02, 0.11, 0.45, 0.94, 0.05]
        response = np.concatenate(
            [level + rng.normal(0.0, 0.006, length) for level, length in zip(levels, lengths)]
        )
        segments, _, _, _, _, _ = segment_press_sequence(response, _config())
        measured = [segment.end_frame for segment in segments[:4]]
        expected = list(np.cumsum(lengths)[:4])
        for actual, target in zip(measured, expected):
            self.assertAlmostEqual(actual, int(target), delta=12)
        self.assertLess(segments[0].mean_response, segments[1].mean_response)
        self.assertLess(segments[1].mean_response, segments[2].mean_response)
        self.assertLess(segments[2].mean_response, segments[3].mean_response)

    def test_windows_stay_inside_stable_stage_and_exclude_release(self) -> None:
        response = np.concatenate(
            [np.full(70, level) for level in (0.01, 0.20, 0.50, 0.95, 0.03)]
        )
        segments, observed, recovered, ratio, status, flags = segment_press_sequence(
            response,
            _config(),
        )
        frame_count = len(response)
        record = DynamicSequenceRecord(
            path=None,  # type: ignore[arg-type]
            file_id="G1/P22.dat",
            capture_group="G1",
            position_label="P22",
            wavelength_nm=np.arange(512, dtype=float),
            spectra=np.zeros((frame_count, 512), dtype=float),
            timestamps_sec=np.arange(frame_count, dtype=float) * 0.04,
            layout=FastDatLayout(
                name="test",
                record_words=513,
                prefix_words=0,
                spectrum_words=512,
                frame_count=frame_count,
                trailing_words=0,
                median_adjacent_correlation=1.0,
                score_margin=1.0,
            ),
        )
        features = np.arange(frame_count * 3, dtype=float).reshape(frame_count, 3)
        sequence = DynamicFeatureSequence(
            record=record,
            baseline_spectrum=np.zeros(512),
            baseline_frame_count=20,
            feature_matrix=features,
            feature_names=("a", "b", "c"),
            response_components=np.zeros((frame_count, 1)),
            response_component_names=("score",),
            response_score=response,
            stage_segments=segments,
            release_observed=observed,
            release_recovered=recovered,
            release_recovery_ratio=ratio,
            segmentation_status=status,
            quality_flags=flags,
        )
        config = {"windowing": {"time_steps": 12, "stride_frames": 4}}
        dataset = build_dynamic_window_dataset((sequence,), config)
        self.assertNotIn("release", dataset.stage_labels.tolist())
        self.assertEqual(dataset.values.shape[1:], (12, 3))
        for start, end, label in zip(
            dataset.window_start_frames,
            dataset.window_end_frames,
            dataset.stage_labels,
        ):
            segment = next(item for item in segments if item.label == label)
            self.assertGreaterEqual(int(start), segment.stable_start_frame)
            self.assertLessEqual(int(end), segment.stable_end_frame)


if __name__ == "__main__":
    unittest.main()
