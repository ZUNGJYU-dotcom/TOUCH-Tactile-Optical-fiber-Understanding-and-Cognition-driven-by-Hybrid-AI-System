from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "bayspec_wavelength_shift_app"


class ResponseBandThresholdContractTests(unittest.TestCase):
    def test_thresholds_remain_internal_while_operator_gauge_is_continuous(self) -> None:
        config = (PROJECT_ROOT / "config" / "bayspec_wavelength_shift_channels.yaml").read_text(
            encoding="utf-8"
        )
        backend = (APP_ROOT / "backend" / "main.py").read_text(encoding="utf-8")
        frontend = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        styles = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

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
        self.assertIn("function opticalForceAxisStep(maximumN)", frontend)
        self.assertIn("function updateOpticalForceDisplayMaximum(valueN)", frontend)
        self.assertIn('"Optical Fz estimate (N)"', frontend)
        self.assertIn('"continuous_optical_force_n"', frontend)
        self.assertIn('trace_response_semantics: "canonical_operator_display_force_n"', frontend)
        self.assertIn("continuous_optical_fz_no_fixed_upper_limit", backend)
        self.assertIn("upper_limit_applied", backend)
        self.assertIn("#dfeef4 0%", styles)
        self.assertIn("#c8665f 100%", styles)
        self.assertNotIn("var(--response-no-contact-end", styles)


if __name__ == "__main__":
    unittest.main()
