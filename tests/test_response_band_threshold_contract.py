from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "bayspec_wavelength_shift_app"


class ResponseBandThresholdContractTests(unittest.TestCase):
    def test_light_band_is_shifted_up_and_shared_across_layers(self) -> None:
        config = (PROJECT_ROOT / "config" / "bayspec_wavelength_shift_channels.yaml").read_text(
            encoding="utf-8"
        )
        backend = (APP_ROOT / "backend" / "main.py").read_text(encoding="utf-8")
        frontend = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        styles = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("no_contact_max_ratio: 0.25", config)
        self.assertIn("light_max_ratio: 0.80", config)
        self.assertIn("normal_max_ratio: 0.90", config)

        self.assertIn('raw.get("no_contact_max_ratio", 0.25)', backend)
        self.assertIn('raw.get("light_max_ratio", 0.80)', backend)
        self.assertIn('raw.get("normal_max_ratio", 0.90)', backend)
        self.assertIn("no_contact_max, light_max, normal_max = 0.25, 0.80, 0.90", backend)

        self.assertIn("noContactMax: 0.25", frontend)
        self.assertIn("smallMax: 0.80", frontend)
        self.assertIn("moderateMax: 0.90", frontend)
        self.assertIn(
            "0.5 * (RESPONSE_BAND_THRESHOLDS.noContactMax + RESPONSE_BAND_THRESHOLDS.smallMax)",
            frontend,
        )
        self.assertIn(
            "0.5 * (RESPONSE_BAND_THRESHOLDS.smallMax + RESPONSE_BAND_THRESHOLDS.moderateMax)",
            frontend,
        )

        self.assertIn("var(--response-no-contact-end, 25%)", styles)
        self.assertIn("var(--response-small-end, 80%)", styles)
        self.assertIn("var(--response-moderate-end, 90%)", styles)


if __name__ == "__main__":
    unittest.main()
