from __future__ import annotations

from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent


class SpectrumProcessingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frontend = (APP_ROOT / "frontend" / "app.js").read_text(
            encoding="utf-8"
        )
        cls.index = (APP_ROOT / "frontend" / "index.html").read_text(
            encoding="utf-8"
        )
        cls.backend = (APP_ROOT / "backend" / "main.py").read_text(
            encoding="utf-8"
        )
        cls.sdk_live = (APP_ROOT / "sdk_live.py").read_text(encoding="utf-8")
        cls.bridge = (APP_ROOT / "bridge.py").read_text(encoding="utf-8")

    def test_settings_controls_and_api_routes_are_connected(self) -> None:
        for control_id in (
            "spectrumIntegrationSelect",
            "spectrumOverlayToggle",
            "spectrumBackgroundToggle",
            "spectrumBaselineToggle",
            "spectrumSmoothingToggle",
            "spectrumNormalizationToggle",
            "spectrumCaptureBackgroundButton",
            "spectrumClearBackgroundButton",
        ):
            self.assertIn(f'id="{control_id}"', self.index)
            self.assertIn(control_id, self.frontend)

        for route in (
            "/api/spectrum_processing/status",
            "/api/spectrum_processing/settings",
            "/api/spectrum_processing/background/capture",
            "/api/spectrum_processing/background/clear",
        ):
            self.assertIn(route, self.frontend)
            self.assertIn(route, self.backend)

    def test_live_request_carries_persisted_integration(self) -> None:
        self.assertIn(
            "state.spectrumProcessing?.settings?.integration_us",
            self.frontend,
        )
        self.assertIn("integration=${integration}", self.frontend)
        self.assertIn("DEFAULT_INTEGRATION_US = 5000", self.sdk_live)

    def test_display_processing_does_not_replace_raw_model_input(self) -> None:
        self.assertIn('"intensity": intensity', self.sdk_live)
        self.assertIn('"display_intensity": processed["display_intensity"]', self.sdk_live)
        self.assertIn('"overlay_intensity": processed["overlay_intensity"]', self.sdk_live)
        self.assertIn('"model_input_source": "raw_intensity"', self.sdk_live)
        self.assertIn('record["intensity"] = spectrum_intensity', self.bridge)
        self.assertIn('record["display_intensity"] = display_intensity', self.bridge)
        self.assertIn('"model_input_source": "raw_intensity"', self.bridge)
        self.assertIn('"applied_to_model_input": False', self.bridge)

    def test_compact_frame_response_does_not_duplicate_display_arrays(self) -> None:
        self.assertIn('clean.pop("display_intensity", None)', self.bridge)
        self.assertIn('clean.pop("overlay_intensity", None)', self.bridge)
        self.assertIn('clean.pop("normalized_intensity_ratio", None)', self.bridge)
        self.assertIn(
            'clean.pop("normalization_reference_intensity_counts", None)',
            self.bridge,
        )


if __name__ == "__main__":
    unittest.main()
