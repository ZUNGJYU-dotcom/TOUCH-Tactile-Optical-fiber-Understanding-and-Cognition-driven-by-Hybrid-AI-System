from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectrum_processing import SpectrumDisplayProcessor


class SpectrumDisplayProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / "settings.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def processor(self, config_path: Path | None = None) -> SpectrumDisplayProcessor:
        return SpectrumDisplayProcessor(
            config_path=config_path,
            user_settings_path=self.settings_path,
        )

    def test_display_processing_never_mutates_raw_model_input(self) -> None:
        processor = self.processor()
        raw = np.linspace(100.0, 180.0, 512)
        raw[250:256] += 400.0
        original = raw.copy()

        result = processor.process(raw)

        np.testing.assert_allclose(raw, original)
        self.assertEqual(len(result["display_intensity"]), 512)
        self.assertTrue(result["spectrum_processing"]["raw_retained"])
        self.assertEqual(
            result["spectrum_processing"]["model_input_source"],
            "raw_intensity",
        )
        self.assertTrue(
            result["spectrum_processing"]["normalization_requested"]
        )
        self.assertNotEqual(result["display_intensity"], original.tolist())

    def test_overlay_contains_previous_processed_frame(self) -> None:
        processor = self.processor()
        first = processor.process(np.arange(32, dtype=float) + 10)
        second = processor.process(np.arange(32, dtype=float) + 20)

        self.assertEqual(first["overlay_intensity"], [])
        self.assertEqual(
            second["overlay_intensity"],
            first["display_intensity"],
        )

    def test_background_subtraction_requires_explicit_reference(self) -> None:
        processor = self.processor()
        processor.update_settings(
            {
                "subtract_background": True,
                "baseline_correction": False,
                "spectrum_smoothing": False,
            }
        )
        missing = processor.process([10, 20, 30])
        self.assertIn(
            "background_reference_required",
            missing["spectrum_processing"]["warnings"],
        )

        captured = processor.capture_background()
        self.assertTrue(captured["ok"])
        result = processor.process([15, 25, 35])
        self.assertEqual(result["display_intensity"], [5.0, 5.0, 5.0])

    def test_invalid_user_values_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "processing.yaml"
            config_path.write_text(
                "spectrum_processing:\n"
                "  integration_us: 1\n"
                "  smoothing_window: 200\n",
                encoding="utf-8",
            )
            processor = self.processor(config_path)
            settings = processor.status()["settings"]
            self.assertEqual(settings["integration_us"], 1)
            self.assertEqual(settings["sensor_mode"], 0)
            self.assertLessEqual(settings["smoothing_window"], 31)
            self.assertEqual(settings["smoothing_window"] % 2, 1)

    def test_normalization_display_preference_can_be_disabled(self) -> None:
        processor = self.processor()
        status = processor.update_settings({"normalize_spectrum": False})
        self.assertFalse(status["settings"]["normalize_spectrum"])

        result = processor.process([100.0, 110.0, 120.0])
        self.assertFalse(
            result["spectrum_processing"]["normalization_requested"]
        )


if __name__ == "__main__":
    unittest.main()
