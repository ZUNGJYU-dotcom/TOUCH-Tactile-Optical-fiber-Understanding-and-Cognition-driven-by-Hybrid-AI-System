from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "bayspec_wavelength_shift_app"
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bridge import BaySpecWavelengthShiftBridge, CHANNEL_ORDER  # noqa: E402
from backend import main as backend_main  # noqa: E402
from src.array_surface.surface_mapper import map_surface, matrices_from_channels  # noqa: E402
from src.wavelength_shift.demodulator import (  # noqa: E402
    cross_correlation_shift_pm,
    estimate_peak_wavelengths,
    wavelength_shift_metrics,
)


def gaussian_spectrum(center_nm: float, *, start: float = 1546.0, stop: float = 1547.8, step: float = 0.005):
    wavelengths = []
    counts = []
    point_count = int(round((stop - start) / step)) + 1
    for index in range(point_count):
        wavelength = start + index * step
        wavelengths.append(wavelength)
        counts.append(900.0 + 30000.0 * math.exp(-0.5 * ((wavelength - center_nm) / 0.055) ** 2))
    return wavelengths, counts


class WavelengthDemodulatorTests(unittest.TestCase):
    def test_peak_estimators_track_subpixel_center(self) -> None:
        wavelengths, counts = gaussian_spectrum(1546.913)
        result = estimate_peak_wavelengths(wavelengths, counts, 1546.89)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["weighted_centroid_wavelength_nm"], 1546.913, places=3)
        self.assertAlmostEqual(result["parabolic_peak_wavelength_nm"], 1546.913, places=3)

    def test_cross_correlation_preserves_shift_sign(self) -> None:
        baseline_x, baseline_y = gaussian_spectrum(1546.89)
        red_x, red_y = gaussian_spectrum(1547.01)
        blue_x, blue_y = gaussian_spectrum(1546.81)
        red = cross_correlation_shift_pm(red_x, red_y, baseline_x, baseline_y, 1546.89)
        blue = cross_correlation_shift_pm(blue_x, blue_y, baseline_x, baseline_y, 1546.89)
        self.assertTrue(red["valid"])
        self.assertTrue(blue["valid"])
        self.assertAlmostEqual(red["shift_pm"], 120.0, delta=6.0)
        self.assertAlmostEqual(blue["shift_pm"], -80.0, delta=6.0)

    def test_response_levels_use_absolute_shift_but_keep_direction(self) -> None:
        small = wavelength_shift_metrics(1546.94, 1546.89)
        moderate = wavelength_shift_metrics(1546.74, 1546.89)
        large = wavelength_shift_metrics(1547.20, 1546.89)
        self.assertEqual(small["response_level"], "small_shift")
        self.assertEqual(small["shift_direction"], "red_shift")
        self.assertEqual(moderate["response_level"], "moderate_shift")
        self.assertEqual(moderate["shift_direction"], "blue_shift")
        self.assertEqual(large["response_level"], "large_shift")
        self.assertFalse(large["temperature_strain_decoupled"])


class BaySpecBridgeCoreLogicTests(unittest.TestCase):
    @staticmethod
    def ingest_spectrum(bridge: BaySpecWavelengthShiftBridge, center_nm: float, timestamp: float) -> dict:
        wavelengths, counts = gaussian_spectrum(center_nm, start=1543.4, stop=1545.3)
        return bridge.ingest(
            {
                "timestamp": timestamp,
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": wavelengths,
                        "intensity": counts,
                        "intensity_counts": max(counts),
                    }
                ],
            }
        )

    def test_runtime_channel_order_uses_physical_display_rows(self) -> None:
        self.assertEqual(
            CHANNEL_ORDER,
            ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"],
        )

    def test_baseline_then_shift_produces_delta_lambda(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        self.ingest_spectrum(bridge, 1544.339792, 0.0)
        baseline = bridge.set_baseline({"channel_id": "P22"})
        self.assertTrue(baseline["ok"])
        self.ingest_spectrum(bridge, 1544.459792, 0.1)
        latest = bridge.latest("P22")
        self.assertAlmostEqual(latest["delta_wavelength_pm"], 120.0, delta=12.0)
        self.assertEqual(latest["response_level"], "moderate_shift")
        self.assertEqual(latest["shift_direction"], "red_shift")
        self.assertFalse(latest["temperature_strain_decoupled"])
        self.assertNotIn("baseline_wavelength_not_ready", latest["qa_flags"])
        self.assertEqual(bridge.frame()["array_frame"]["mode"], "p22_fallback")

    def test_frame_id_advances_on_ingest_not_poll(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        self.ingest_spectrum(bridge, 1544.339792, 0.0)
        first = bridge.frame()
        second = bridge.frame()
        self.assertEqual(first["frame_id"], second["frame_id"])
        self.ingest_spectrum(bridge, 1544.359792, 0.1)
        self.assertGreater(bridge.frame()["frame_id"], second["frame_id"])

    def test_full_spectrum_exposes_nine_real_fbg_candidates_without_enabling_array(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        wavelengths = [1523.1596 + index * (1607.0144 - 1523.1596) / 511 for index in range(512)]
        centers = [1527.813917, 1532.074029, 1536.272630, 1540.087209, 1544.339792, 1547.790240, 1551.672060, 1555.766698, 1559.838208]
        counts = [6500.0 for _ in wavelengths]
        for peak_number, center in enumerate(centers):
            amplitude = 28000.0 + 2500.0 * (peak_number % 3)
            counts = [
                value + amplitude * math.exp(-0.5 * ((wavelength - center) / 0.30) ** 2)
                for wavelength, value in zip(wavelengths, counts)
            ]
        result = bridge.ingest(
            {
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": wavelengths,
                        "intensity": counts,
                    }
                ]
            }
        )
        self.assertTrue(result["ok"])
        latest = bridge.latest("P22")
        self.assertEqual(latest["spectrum_peak_profile"], "current_real_9fbg_candidate")
        self.assertEqual(len(latest["spectrum_peaks"]), 9)
        self.assertEqual([peak["channel_id"] for peak in latest["spectrum_peaks"]], ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"])
        self.assertEqual(
            [peak["candidate_id"] for peak in latest["spectrum_peaks"]],
            [f"FBG{index:02d}" for index in range(1, 10)],
        )
        self.assertEqual(
            [peak["provisional_channel_id"] for peak in latest["spectrum_peaks"]],
            ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"],
        )
        self.assertTrue(
            all(not peak["physical_channel_mapping_final"] for peak in latest["spectrum_peaks"])
        )
        self.assertTrue(
            all(
                math.isfinite(peak["candidate_delta_wavelength_pm"])
                and math.isfinite(peak["candidate_absolute_shift_pm"])
                for peak in latest["spectrum_peaks"]
            )
        )
        self.assertAlmostEqual(latest["demodulation_wavelength_nm"], 1544.339792, places=6)
        self.assertEqual(bridge.frame()["array_frame"]["mode"], "p22_fallback")

    def test_out_of_window_spectrum_is_not_a_valid_response(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        result = bridge.ingest(
            {
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": [1500.0, 1500.1, 1500.2],
                        "intensity": [100.0, 300.0, 120.0],
                    }
                ]
            }
        )
        self.assertTrue(result["ok"])
        latest = bridge.latest("P22")
        self.assertIn("p22_peak_not_found_near_candidate_wavelength", latest["qa_flags"])
        self.assertEqual(latest["response_level"], "uncertain")
        self.assertEqual(bridge.frame()["array_frame"]["mode"], "no_valid_channel")

    def test_global_spectrum_endpoint_declares_p22_as_transport_only(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        wavelengths = [
            1523.1596 + index * (1607.0144 - 1523.1596) / 511
            for index in range(512)
        ]
        counts = [6500.0 for _ in wavelengths]
        for center in (
            1527.813917,
            1532.074029,
            1536.272630,
            1540.087209,
            1544.339792,
            1547.790240,
            1551.672060,
            1555.766698,
            1559.838208,
        ):
            counts = [
                value + 30000.0 * math.exp(-0.5 * ((wavelength - center) / 0.30) ** 2)
                for wavelength, value in zip(wavelengths, counts)
            ]
        bridge.ingest(
            {
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": wavelengths,
                        "intensity": counts,
                    }
                ]
            }
        )

        with patch.object(backend_main, "bridge", bridge):
            payload = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=True,
            )

        self.assertEqual(payload["scope"], "global_3x3_hybrid_spectral_fingerprint")
        self.assertIsNone(payload["selected_channel"])
        self.assertEqual(payload["carrier_channel_id"], "P22")
        self.assertEqual(
            payload["carrier_channel_role"],
            "legacy_full_spectrum_transport_only",
        )
        self.assertEqual(
            payload["global_candidate_ids"],
            [f"FBG{index:02d}" for index in range(1, 10)],
        )
        self.assertFalse(payload["physical_channel_mapping_final"])
        self.assertEqual(payload["global_candidate_summary"]["valid_candidate_count"], 9)
        self.assertIn(
            payload["global_candidate_summary"]["dominant_candidate_id"],
            [f"FBG{index:02d}" for index in range(1, 10)],
        )
        self.assertEqual(
            payload["latest"]["carrier_channel_role"],
            "legacy_full_spectrum_transport_only",
        )
        self.assertEqual(
            payload["latest"]["global_candidate_ids"],
            [f"FBG{index:02d}" for index in range(1, 10)],
        )
        self.assertTrue(payload["global_frame_qa"]["candidate_contract_complete"])
        self.assertFalse(payload["global_frame_qa"]["formal_recognition_allowed"])
        self.assertEqual(
            payload["candidate_contract_complete"],
            payload["global_frame_qa"]["candidate_contract_complete"],
        )
        self.assertEqual(payload["baseline_ready"], payload["global_frame_qa"]["baseline_ready"])
        self.assertEqual(payload["source_fresh"], payload["global_frame_qa"]["source_fresh"])
        self.assertEqual(
            payload["formal_recognition_allowed"],
            payload["global_frame_qa"]["formal_recognition_allowed"],
        )
        self.assertEqual(payload["blockers"], payload["global_frame_qa"]["blockers"])

    def test_frontend_global_normalization_marks_stale_frames(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        styles_css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        self.assertIn('status === "stale_frame"', app_js)
        self.assertIn("globalFrameQa.source_fresh !== false", app_js)
        self.assertIn('"stale_frame"', app_js)
        self.assertIn("Global nine-FBG fingerprint is cached or stale", app_js)
        self.assertIn("GLOBAL_PROXY_FULL_SCALE_PM", app_js)
        self.assertIn("globalCandidateSpatialProxy(validPeaks, rawTrace, baselineStatsByCandidate)", app_js)
        self.assertIn("candidateTraceBaselineStats", app_js)
        self.assertIn("globalCandidateBaselineStatsMap", app_js)
        self.assertIn("globalEventTraceRecords", app_js)
        self.assertIn('trace_response_semantics: "residual_compensated_global_event_peak"', app_js)
        self.assertIn("const modelResponseTrace", app_js)
        self.assertIn('trained_model_visual_response_percent', app_js)
        self.assertIn('global_residual_compensated_event_peak', app_js)
        self.assertIn('Trained model visual response (%)', app_js)
        self.assertIn('Global event peak |Δλ| (pm)', app_js)
        self.assertIn("GLOBAL_EVENT_MIN_CONDITIONING_FRAMES", app_js)
        self.assertIn("contactEvidence", app_js)
        self.assertIn("session_global_no_contact_baseline", app_js)
        self.assertIn("Contact frames must never recenter it", app_js)
        self.assertIn("response_allowed: responseAllowed", app_js)
        self.assertIn("frameResponseIsUsable", app_js)
        self.assertIn("requestSequence < state.lastCommittedFrameRequest", app_js)
        self.assertIn("frameRequestInFlight", app_js)
        self.assertIn("forcedFrameRequestQueued", app_js)
        self.assertIn("if (state.frameRequestInFlight)", app_js)
        self.assertIn("provisional global spectral spatial proxy", app_js)
        self.assertIn("global_spectrum_provisional_spatial_proxy", app_js)
        self.assertIn("RESPONSE_BAND_THRESHOLDS", app_js)
        self.assertIn("normalizedSurfaceResponseRatio(surfaceMetrics, record)", app_js)
        self.assertNotIn(
            "globalEventPeakShiftPm / WAVELENGTH_SHIFT_FULL_SCALE_PM",
            app_js,
        )
        self.assertIn("var(--response-small-end, 34%)", styles_css)
        self.assertIn("var(--response-moderate-end, 70%)", styles_css)
        self.assertIn("Normalized visual response: no contact below", app_js)
        self.assertIn("responseLevelFromSurfaceValue(proxyResponseRatio)", app_js)
        self.assertIn("const trainedModelTrace", app_js)
        self.assertIn("trace: trainedModelTrace || normalizedEventTrace", app_js)
        self.assertIn('trainedModelDisplay ? "Visual response" : "Peak |Δλ|"', app_js)
        self.assertIn("formatPercent(surfacePeakValue, 1)", app_js)
        self.assertIn("function snapDisplayedFrameToCurrentTargets()", app_js)
        self.assertIn("state.smoothSurfaceVisualPeak = state.targetSurfaceVisualPeak", app_js)
        self.assertIn("snapDisplayedFrameToCurrentTargets();", app_js)
        self.assertIn("hasActiveSurfaceResponse && (", app_js)
        self.assertIn("surfaceHasActiveResponse", app_js)
        self.assertIn('ARRAY_ONE_SHOT_SCENARIOS = new Set(["tap", "release"])', app_js)

    def test_operator_spectrum_entry_preserves_its_full_label(self) -> None:
        index_html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        styles_css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

        self.assertIn('class="chip spectrum-open-chip">Spectrum</span>', index_html)
        self.assertIn(".operator-mode .summary-hud .spectrum-open-chip", styles_css)
        self.assertIn("grid-template-columns: minmax(0, 1fr) auto !important;", styles_css)
        self.assertIn("min-width: 76px;", styles_css)

    def test_operator_qa_status_has_the_widest_status_column(self) -> None:
        index_html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        styles_css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

        self.assertIn('class="qa-status-item"><small>QA status</small>', index_html)
        self.assertIn(
            "grid-template-columns: minmax(105px, 0.9fr) minmax(115px, 1fr) minmax(158px, 1.35fr) !important;",
            styles_css,
        )
        self.assertIn(".operator-mode #topQaStatus", styles_css)

    def test_global_candidate_baseline_uses_complete_recent_nine_peak_frames(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        wavelengths = [
            1523.1596 + index * (1607.0144 - 1523.1596) / 511
            for index in range(512)
        ]
        centers = (
            1527.813917,
            1532.074029,
            1536.272630,
            1540.087209,
            1544.339792,
            1547.790240,
            1551.672060,
            1555.766698,
            1559.838208,
        )

        def spectrum(shift_nm: float) -> list[float]:
            counts = [6500.0 for _ in wavelengths]
            for center in centers:
                counts = [
                    value
                    + 30000.0
                    * math.exp(-0.5 * ((wavelength - (center + shift_nm)) / 0.30) ** 2)
                    for wavelength, value in zip(wavelengths, counts)
                ]
            return counts

        for _ in range(3):
            bridge.ingest(
                {
                    "channels": [
                        {
                            "channel_id": "P22",
                            "wavelength_nm": wavelengths,
                            "intensity": spectrum(0.0),
                        }
                    ]
                }
            )

        stale_ingest = time.time() - 10.7
        for index, record in enumerate(bridge.records_by_channel["P22"]):
            record["ingested_at"] = stale_ingest + index * 0.3
            record["source"] = "unit_test_live"

        stale_baseline = bridge.set_global_candidate_baseline(minimum_frames=3)
        self.assertFalse(stale_baseline["ok"])
        self.assertEqual(stale_baseline["reason"], "global_baseline_source_frames_stale")

        base_ingest = time.time() - 0.7
        for index, record in enumerate(bridge.records_by_channel["P22"]):
            record["ingested_at"] = base_ingest + index * 0.3

        baseline = bridge.set_global_candidate_baseline(minimum_frames=3)
        self.assertTrue(baseline["ok"])
        self.assertEqual(baseline["candidate_ids"], [f"FBG{index:02d}" for index in range(1, 10)])

        bridge.ingest(
            {
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": wavelengths,
                        "intensity": spectrum(0.04),
                    }
                ]
            }
        )
        latest = bridge.latest("P22")
        self.assertEqual(len(latest["spectrum_peaks"]), 9)
        self.assertTrue(
            all(
                peak["candidate_reference_status"]
                == "session_global_no_contact_baseline"
                for peak in latest["spectrum_peaks"]
            )
        )
        self.assertTrue(
            all(
                25.0 <= peak["candidate_delta_wavelength_pm"] <= 55.0
                for peak in latest["spectrum_peaks"]
            )
        )

    def test_global_candidate_baseline_rejects_unstable_peak_positions(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        wavelengths = [
            1523.1596 + index * (1607.0144 - 1523.1596) / 511
            for index in range(512)
        ]
        centers = (
            1527.813917,
            1532.074029,
            1536.272630,
            1540.087209,
            1544.339792,
            1547.790240,
            1551.672060,
            1555.766698,
            1559.838208,
        )

        for shift_nm in (0.0, 0.04, 0.0):
            counts = [6500.0 for _ in wavelengths]
            for center in centers:
                counts = [
                    value
                    + 30000.0
                    * math.exp(
                        -0.5 * ((wavelength - (center + shift_nm)) / 0.30) ** 2
                    )
                    for wavelength, value in zip(wavelengths, counts)
                ]
            bridge.ingest(
                {
                    "channels": [
                        {
                            "channel_id": "P22",
                            "wavelength_nm": wavelengths,
                            "intensity": counts,
                        }
                    ]
                }
            )

        base_ingest = time.time() - 0.7
        for index, record in enumerate(bridge.records_by_channel["P22"]):
            record["ingested_at"] = base_ingest + index * 0.3
            record["source"] = "unit_test_live"

        baseline = bridge.set_global_candidate_baseline(minimum_frames=3)
        self.assertFalse(baseline["ok"])
        self.assertEqual(baseline["reason"], "global_candidate_baseline_unstable")
        self.assertTrue(baseline["unstable_candidates"])

    def test_global_candidate_extraction_rejects_smooth_ramp_false_peaks(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        wavelengths = [
            1523.1596 + index * (1607.0144 - 1523.1596) / 511
            for index in range(512)
        ]
        smooth_ramp = [1000.0 + index * 50.0 for index in range(512)]
        peaks = bridge._extract_candidate_spectrum_peaks(wavelengths, smooth_ramp)

        self.assertEqual(len(peaks), 9)
        self.assertTrue(all(peak["valid"] is False for peak in peaks))
        self.assertTrue(
            all(
                "peak_near_search_edge" in peak["quality_flags"]
                or "peak_prominence_low" in peak["quality_flags"]
                for peak in peaks
            )
        )


class WavelengthArraySimulationTests(unittest.TestCase):
    SCENARIO_STEPS = {
        "no_contact": 1,
        "center_press": 14,
        "off_center_fingertip_contact": 14,
        "broad_fingertip_contact": 14,
        "vertical_slide_p11_p12_p13": 12,
        "horizontal_slide_p11_p21_p31": 12,
        "diagonal_slide_p11_p22_p33": 12,
        "tap": 10,
        "release": 12,
    }

    def test_no_contact_and_center_contact_shift_peaks_not_peak_height(self) -> None:
        idle = backend_main.simulated_array_frame("no_contact", step=0)
        contact = backend_main.simulated_array_frame("center_press", step=7)
        self.assertLess(idle["peak_wavelength_shift_pm"], 1.0)
        self.assertGreater(contact["peak_wavelength_shift_pm"], 50.0)
        idle_channels = {item["channel_id"]: item for item in idle["channels"]}
        contact_channels = {item["channel_id"]: item for item in contact["channels"]}
        for channel_id in CHANNEL_ORDER:
            self.assertEqual(
                idle_channels[channel_id]["intensity_counts"],
                contact_channels[channel_id]["intensity_counts"],
            )
            self.assertEqual(contact_channels[channel_id]["relative_intensity"], 1.0)
            self.assertEqual(contact_channels[channel_id]["attenuation_ratio"], 0.0)
        self.assertFalse(contact_channels["P22"]["simulated_intensity_modulation_enabled"])
        self.assertGreater(contact_channels["P22"]["tracked_wavelength_nm"], contact_channels["P22"]["baseline_wavelength_nm"])
        self.assertNotIn("same_fiber_downstream_optical_coupling", contact["coupling_sources"])

    def test_synthetic_spectrum_peak_centers_follow_channel_shifts(self) -> None:
        frame = backend_main.simulated_array_frame("center_press", step=7)
        channels = {item["channel_id"]: item for item in frame["channels"]}
        peaks = {item["channel_id"]: item for item in frame["spectrum"]["peaks"]}
        self.assertAlmostEqual(
            peaks["P22"]["peak_wavelength_nm"],
            channels["P22"]["tracked_wavelength_nm"],
            places=6,
        )
        self.assertEqual(frame["frame_sync_status"], "synced")
        self.assertIn("wavelength-shift", frame["surface_title"])

    def test_synthetic_spectrum_peak_height_is_invariant_across_scenarios(self) -> None:
        frames = [
            backend_main.simulated_array_frame("no_contact", step=0),
            backend_main.simulated_array_frame("center_press", step=7),
            backend_main.simulated_array_frame("broad_fingertip_contact", step=7),
            backend_main.simulated_array_frame("vertical_slide_p11_p12_p13", step=5),
        ]
        maxima = [max(frame["spectrum"]["intensity"]) for frame in frames]
        for frame, maximum in zip(frames, maxima):
            self.assertAlmostEqual(maximum, 44000.0, delta=0.01)
            self.assertFalse(frame["spectrum"]["intensity_modulation_enabled"])
            self.assertEqual(frame["spectrum"]["peak_height_mode"], "fixed_per_channel")
            self.assertEqual(frame["spectrum"]["frame_render_semantics"], "replace_previous_spectrum")

    def test_every_demo_frame_is_pure_wavelength_translation(self) -> None:
        for scenario, step_count in self.SCENARIO_STEPS.items():
            for step in range(step_count):
                with self.subTest(scenario=scenario, step=step):
                    frame = backend_main.simulated_array_frame(scenario, step=step)
                    spectrum = frame["spectrum"]
                    wavelengths = spectrum["wavelength_nm"]
                    counts = spectrum["intensity"]
                    self.assertEqual(len(wavelengths), 4301)
                    self.assertEqual(len(counts), 4301)
                    self.assertTrue(all(math.isfinite(value) for value in counts))
                    self.assertAlmostEqual(max(counts), 44000.0, delta=0.01)
                    self.assertFalse(spectrum["intensity_modulation_enabled"])
                    for channel in frame["channels"]:
                        self.assertEqual(
                            channel["intensity_counts"],
                            channel["baseline_intensity_counts"],
                        )
                        self.assertEqual(channel["relative_intensity"], 1.0)
                        self.assertEqual(channel["attenuation_ratio"], 0.0)
                        self.assertAlmostEqual(
                            channel["tracked_wavelength_nm"]
                            - channel["baseline_wavelength_nm"],
                            channel["delta_wavelength_pm"] / 1000.0,
                            places=12,
                        )

    def test_release_finishes_at_no_contact(self) -> None:
        complete = backend_main.simulated_array_frame("release", step=9)
        self.assertEqual(complete["surface_metrics"]["event_interpretation"], "no_contact_after_release")
        self.assertEqual(complete["surface_metrics"]["responding_channel_count"], 0)

        late_frame = backend_main.simulated_array_frame("release", step=25)
        self.assertEqual(late_frame["surface_metrics"]["event_interpretation"], "no_contact_after_release")
        self.assertEqual(late_frame["surface_metrics"]["responding_channel_count"], 0)

        late_tap = backend_main.simulated_array_frame("tap", step=25)
        self.assertEqual(late_tap["surface_metrics"]["event_interpretation"], "no_contact_after_tap")
        self.assertEqual(late_tap["surface_metrics"]["responding_channel_count"], 0)


class SurfaceMapperCoreLogicTests(unittest.TestCase):
    @staticmethod
    def channel(channel_id: str, x: float, y: float, response: float) -> dict:
        return {
            "channel_id": channel_id,
            "enabled": True,
            "valid": True,
            "x": x,
            "y": y,
            "wavelength_shift_response_ratio": response,
            "response_value": response,
            "qa_status": "ok",
        }

    def test_surface_never_exceeds_strongest_channel(self) -> None:
        surface = map_surface(
            [self.channel("P11", -0.25, 0.0, 0.80), self.channel("P21", 0.25, 0.0, 0.70)]
        )
        self.assertLessEqual(surface["surface_metrics"]["surface_peak"], 0.80 + 1e-12)
        self.assertEqual(surface["responding_channel_count"], 2)

    def test_matrix_orientation_matches_physical_layout(self) -> None:
        order = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
        channels = [self.channel(channel_id, 0.0, 0.0, index / 10.0) for index, channel_id in enumerate(order, 1)]
        matrix = matrices_from_channels(channels)["array_response_3x3"]
        self.assertEqual(matrix[0], [0.1, 0.2, 0.3])
        self.assertEqual(matrix[1], [0.4, 0.5, 0.6])
        self.assertEqual(matrix[2], [0.7, 0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
