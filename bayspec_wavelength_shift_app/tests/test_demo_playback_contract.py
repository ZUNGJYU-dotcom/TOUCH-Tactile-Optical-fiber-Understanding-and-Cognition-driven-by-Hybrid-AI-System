from pathlib import Path
import unittest

from backend.main import (
    _demo_envelope,
    _operator_response_band_thresholds,
    _response_level_from_shift_ratio,
    recorded_demo_library,
    simulated_array_frame,
)


APP_ROOT = Path(__file__).resolve().parents[1]


class DemoPlaybackContractTests(unittest.TestCase):
    def test_recorded_demo_asset_is_ready(self) -> None:
        status = recorded_demo_library.status()
        self.assertTrue(status["available"])
        self.assertEqual(status["status"], "recorded_real_spectrum_ready")
        self.assertEqual(status["capture_date"], "2026-07-31")
        self.assertEqual(status["spectrum_points"], 512)
        self.assertEqual(status["frames_per_position"], 50)
        discovery = status["peak_discovery"]
        self.assertEqual(
            discovery["method"],
            "automatic_no_contact_wavelength_order_discovery",
        )
        self.assertEqual(discovery["status"], "pass")
        self.assertEqual(discovery["selected_peak_count"], 9)

    def test_recorded_demo_peak_labels_follow_auto_discovered_peaks(self) -> None:
        frame = simulated_array_frame("center_press", step=25)
        spectrum = frame["spectrum"]
        peaks = spectrum["peaks"]
        expected_channels = [
            "P11",
            "P12",
            "P13",
            "P21",
            "P22",
            "P23",
            "P31",
            "P32",
            "P33",
        ]

        self.assertEqual(
            spectrum["spectrum_peak_profile"],
            "recorded_auto_discovered_9fbg",
        )
        self.assertEqual(
            spectrum["spectrum_peak_mapping_status"],
            "wavelength_order_assignment",
        )
        self.assertEqual(len(peaks), 9)
        self.assertEqual(
            [peak["provisional_channel_id"] for peak in peaks],
            expected_channels,
        )
        self.assertTrue(all(peak["candidate_mapping"] for peak in peaks))
        self.assertFalse(spectrum["physical_channel_mapping_final"])
        tracked = [float(peak["tracked_wavelength_nm"]) for peak in peaks]
        references = [float(peak["target_wavelength_nm"]) for peak in peaks]
        self.assertEqual(tracked, sorted(tracked))
        self.assertTrue(
            all(abs(value - reference) <= 0.85 for value, reference in zip(tracked, references))
        )

    def test_demo_frame_uses_same_frame_recorded_spectrum_and_force(self) -> None:
        frame = simulated_array_frame("center_press", step=25)
        spectrum = frame["spectrum"]

        self.assertEqual(
            spectrum["spectrum_type"],
            "recorded real BaySpec 512-point spectrum",
        )
        self.assertEqual(len(spectrum["wavelength_nm"]), 512)
        self.assertEqual(len(spectrum["intensity"]), 512)
        self.assertEqual(
            frame["demo_source_session_id"], spectrum["source_session_id"]
        )
        self.assertEqual(
            frame["demo_source_capture_index"], spectrum["source_capture_index"]
        )
        self.assertAlmostEqual(
            frame["recorded_reference_force_fz_n"],
            spectrum["reference_force_fz_n"],
        )
        self.assertEqual(
            frame["recorded_source_sync_status"],
            "same_frame_spectrum_and_px6d_fz",
        )
        self.assertNotIn("synthetic", spectrum["spectrum_type"].lower())
        self.assertNotIn("synthetic", spectrum["source_note"].lower())

    def test_latest_primary_sessions_are_used_for_recollected_positions(self) -> None:
        expected_prefixes = {
            "p21_contact": "20260731_112602_P21_",
            "p12_contact": "20260731_112749_P12_",
        }
        for scenario, prefix in expected_prefixes.items():
            with self.subTest(scenario=scenario):
                frame = simulated_array_frame(scenario, step=25)
                self.assertTrue(frame["demo_source_session_id"].startswith(prefix))

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

    def test_every_frontend_demo_entry_uses_recorded_spectrum_endpoint(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        demo_start = app_js.index("async function injectDemoFrame")
        demo_end = app_js.index("async function runDemoTransition", demo_start)
        demo_source = app_js[demo_start:demo_end]

        self.assertIn('level === "no_contact" ? "no_contact" : "center_press"', demo_source)
        self.assertIn("injectArrayDemoFrame", demo_source)
        self.assertNotIn("generateDemoSpectrum", app_js)
        self.assertNotIn("synthetic simulated spectrum", app_js)
        self.assertIn('label: "RECORDED"', app_js)
        self.assertIn('short: "REC"', app_js)
        self.assertIn('recordedDemoActive ? "RECORDED" : "SIMULATED"', app_js)
        self.assertIn('data-demo-level="no_contact"', html)
        self.assertNotIn("Legacy P22 fallback", html)

    def test_frontend_uses_auto_discovered_markers_for_recorded_demo(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="spectrumReferenceLegend"', html)
        self.assertIn('peakProfile === "recorded_auto_discovered_9fbg"', app_js)
        self.assertIn('? "auto-discovered reference"', app_js)
        self.assertIn(
            "peak.provisional_channel_id || peak.channel_id || peak.candidate_id",
            app_js,
        )
        self.assertIn(
            "spectrum_peak_profile: spectrum?.spectrum_peak_profile",
            app_js,
        )


if __name__ == "__main__":
    unittest.main()
