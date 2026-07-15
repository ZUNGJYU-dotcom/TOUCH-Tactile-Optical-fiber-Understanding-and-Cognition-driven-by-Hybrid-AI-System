from __future__ import annotations

import unittest

from src.hybrid_spectrum.temporal_prediction import TemporalStaticPredictionStabilizer


def prediction(contact: str, position: str | None = None, level: str | None = None) -> dict:
    return {
        "contact": {"label": contact},
        "position": (
            {
                "label": position,
                "ensemble_diagnostics": {"agreement_fraction": 1.0},
            }
            if position is not None
            else None
        ),
        "force_level": {"label": level} if level is not None else None,
    }


class TemporalStaticPredictionStabilizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TemporalStaticPredictionStabilizer(
            window_size=5,
            minimum_contact_frames=3,
            release_frames=2,
        )

    def update(self, frame: int, value: dict, baseline: str = "baseline-a") -> dict:
        return self.model.update(
            frame_id=frame,
            prediction=value,
            baseline_token=baseline,
            timestamp=float(frame),
        )

    def test_contact_requires_three_unique_frames_and_votes_position_level(self) -> None:
        self.assertEqual(
            self.update(1, prediction("contact", "P21", "light"))["status"],
            "contact_warming_up",
        )
        self.update(2, prediction("contact", "P21", "normal"))
        result = self.update(3, prediction("contact", "P21", "normal"))

        self.assertEqual(result["status"], "stable_contact")
        self.assertEqual(result["position_label"], "P21")
        self.assertEqual(result["force_label"], "normal")
        self.assertAlmostEqual(result["force_support"], 2.0 / 3.0)

    def test_duplicate_poll_is_not_additional_evidence(self) -> None:
        first = self.update(10, prediction("contact", "P22", "hard"))
        duplicate = self.update(10, prediction("contact", "P22", "hard"))

        self.assertEqual(first["history_unique_frames"], 1)
        self.assertEqual(duplicate["history_unique_frames"], 1)
        self.assertTrue(duplicate["duplicate_frame_ignored"])

    def test_two_release_frames_clear_position_and_level(self) -> None:
        for frame in range(1, 4):
            self.update(frame, prediction("contact", "P13", "hard"))
        held = self.update(4, prediction("no_contact"))
        released = self.update(5, prediction("no_contact"))

        self.assertEqual(held["contact_label"], "contact")
        self.assertEqual(released["status"], "stable_no_contact")
        self.assertIsNone(released["position_label"])
        self.assertIsNone(released["force_label"])

    def test_baseline_change_clears_old_contact_history(self) -> None:
        for frame in range(1, 4):
            self.update(frame, prediction("contact", "P33", "hard"))
        result = self.update(
            4,
            prediction("no_contact"),
            baseline="baseline-b",
        )

        self.assertEqual(result["status"], "stable_no_contact")
        self.assertEqual(result["history_unique_frames"], 1)

    def test_repress_after_release_requires_three_fresh_contact_frames(self) -> None:
        for frame in range(1, 4):
            self.update(frame, prediction("contact", "P13", "hard"))
        self.update(4, prediction("no_contact"))
        released = self.update(5, prediction("no_contact"))

        first_repress = self.update(6, prediction("contact", "P22", "light"))
        second_repress = self.update(7, prediction("contact", "P22", "light"))
        third_repress = self.update(8, prediction("contact", "P22", "light"))

        self.assertEqual(released["history_unique_frames"], 2)
        self.assertEqual(first_repress["status"], "contact_warming_up")
        self.assertEqual(second_repress["status"], "contact_warming_up")
        self.assertEqual(third_repress["status"], "stable_contact")
        self.assertEqual(third_repress["position_label"], "P22")
        self.assertEqual(third_repress["force_label"], "light")


if __name__ == "__main__":
    unittest.main()
