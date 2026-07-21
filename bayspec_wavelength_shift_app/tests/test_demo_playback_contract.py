from pathlib import Path
import unittest

from backend.main import (
    _demo_envelope,
    _operator_response_band_thresholds,
    _response_level_from_shift_ratio,
    simulated_array_frame,
)


APP_ROOT = Path(__file__).resolve().parents[1]


class DemoPlaybackContractTests(unittest.TestCase):
    def test_response_bands_raise_and_widen_light_range(self) -> None:
        thresholds = _operator_response_band_thresholds()
        self.assertEqual(thresholds["no_contact_max"], 0.25)
        self.assertEqual(thresholds["light_max"], 0.80)
        self.assertEqual(thresholds["normal_max"], 0.90)
        self.assertEqual(_response_level_from_shift_ratio(0.249), "no_contact")
        self.assertEqual(_response_level_from_shift_ratio(0.25), "small_shift")
        self.assertEqual(_response_level_from_shift_ratio(0.79), "small_shift")
        self.assertEqual(_response_level_from_shift_ratio(0.80), "moderate_shift")
        self.assertEqual(_response_level_from_shift_ratio(0.90), "large_shift")

    def test_center_contact_uses_light_release_hard_timeline(self) -> None:
        light = [_demo_envelope("center_press", step) for step in range(0, 5)]
        released = [_demo_envelope("center_press", step) for step in range(5, 25)]
        hard = [_demo_envelope("center_press", step) for step in range(25, 45)]
        final_release = [_demo_envelope("center_press", step) for step in range(45, 50)]

        self.assertTrue(all(0.25 <= value < 0.80 for value in light))
        self.assertTrue(all(value == 0.0 for value in released))
        self.assertTrue(all(value >= 0.90 for value in hard))
        self.assertGreater(final_release[0], final_release[-1])
        self.assertEqual(final_release[-1], 0.0)

    def test_staged_point_contacts_share_timeline_and_target_channel(self) -> None:
        expected_channels = {
            "p21_contact": "P21",
            "p12_contact": "P12",
            "p32_contact": "P32",
        }
        for scenario, channel_id in expected_channels.items():
            with self.subTest(scenario=scenario):
                self.assertLess(_demo_envelope(scenario, 4), 0.80)
                self.assertEqual(_demo_envelope(scenario, 5), 0.0)
                self.assertEqual(_demo_envelope(scenario, 24), 0.0)
                self.assertGreaterEqual(_demo_envelope(scenario, 25), 0.90)
                self.assertGreaterEqual(_demo_envelope(scenario, 44), 0.90)
                self.assertEqual(_demo_envelope(scenario, 49), 0.0)

                frame = simulated_array_frame(scenario, step=25)
                self.assertEqual(frame["dominant_channel"], channel_id)
                self.assertEqual(frame["frame_sync_status"], "synced")

    def test_contact_and_slide_actions_release_to_zero(self) -> None:
        self.assertEqual(_demo_envelope("center_press", 49), 0.0)
        self.assertEqual(_demo_envelope("broad_fingertip_contact", 13), 0.0)
        self.assertEqual(_demo_envelope("vertical_slide_p11_p12_p13", 11), 0.0)
        self.assertEqual(_demo_envelope("horizontal_slide_p11_p21_p31", 11), 0.0)
        self.assertEqual(_demo_envelope("diagonal_slide_p11_p22_p33", 11), 0.0)

    def test_frontend_exposes_single_and_five_second_loop_controls(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="demoSingleButton"', html)
        self.assertIn("Play once", html)
        self.assertIn("Start 5 s loop", html)
        self.assertIn("const DEMO_ARRAY_LOOP_INTERVAL_MS = 5000;", app_js)
        self.assertIn("center_press: 50", app_js)
        self.assertIn("p21_contact: 50", app_js)
        self.assertIn("p12_contact: 50", app_js)
        self.assertIn("p32_contact: 50", app_js)
        self.assertIn('data-array-scenario="p21_contact"', html)
        self.assertIn('data-array-scenario="p12_contact"', html)
        self.assertIn('data-array-scenario="p32_contact"', html)
        self.assertIn(
            "state.arrayDemoActionCompletedAt + DEMO_ARRAY_LOOP_INTERVAL_MS",
            app_js,
        )
        self.assertIn("state.arrayDemoActionCompletedAt = performance.now();", app_js)
        self.assertIn("Wait five seconds after release before repeating", html)
        self.assertIn('const frameScenario = actionFinished ? "no_contact"', app_js)
        self.assertIn('"released · baseline running"', app_js)
        self.assertIn("? performance.now() + demoArrayStepIntervalMs()", app_js)
        self.assertIn('playbackMode: "single"', app_js)


if __name__ == "__main__":
    unittest.main()
