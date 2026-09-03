from __future__ import annotations

import math
from pathlib import Path
import re
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

    def test_oversized_spectrum_is_rejected_before_allocating_channel_history(self) -> None:
        bridge = BaySpecWavelengthShiftBridge(max_spectrum_points=32)
        result = bridge.ingest(
            {
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": [1540.0 + index * 0.01 for index in range(33)],
                        "intensity": [1000.0] * 33,
                    }
                ]
            }
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["records_ingested"], 0)
        self.assertEqual(result["records_rejected"], 1)
        self.assertEqual(result["rejections"][0]["reason"], "spectrum_point_limit_exceeded")
        self.assertEqual(dict(bridge.records_by_channel), {})
        self.assertEqual(
            bridge.status()["ingest_rejection_counts"]["spectrum_point_limit_exceeded"],
            1,
        )

    def test_channel_flood_is_bounded_across_single_and_repeated_payloads(self) -> None:
        bridge = BaySpecWavelengthShiftBridge(
            max_channel_buffers=4,
            max_channels_per_payload=4,
        )
        first = bridge.ingest(
            {
                "channels": [
                    {"channel_id": f"C{index}", "intensity_counts": 1000.0 + index}
                    for index in range(10)
                ]
            }
        )
        self.assertTrue(first["ok"])
        self.assertEqual(first["records_ingested"], 4)
        self.assertEqual(first["records_rejected"], 6)
        self.assertEqual(len(bridge.records_by_channel), 4)

        second = bridge.ingest(
            {"channels": [{"channel_id": "C_new", "intensity_counts": 2000.0}]}
        )
        self.assertFalse(second["ok"])
        self.assertEqual(second["rejections"][0]["reason"], "channel_buffer_limit_exceeded")
        self.assertEqual(len(bridge.records_by_channel), 4)

    def test_invalid_or_misaligned_spectrum_arrays_are_rejected_not_silently_compacted(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        nonfinite = bridge.ingest(
            {
                "channel_id": "P22",
                "wavelength_nm": [1546.8, float("nan"), 1547.0],
                "intensity": [100.0, 140.0, 105.0],
            }
        )
        mismatch = bridge.ingest(
            {
                "channel_id": "P22",
                "wavelength_nm": [1546.8, 1546.9],
                "intensity": [100.0],
            }
        )

        self.assertFalse(nonfinite["ok"])
        self.assertEqual(
            nonfinite["rejections"][0]["reason"],
            "spectrum_contains_nonfinite_or_nonnumeric_value",
        )
        self.assertFalse(mismatch["ok"])
        self.assertEqual(mismatch["rejections"][0]["reason"], "spectrum_axis_length_mismatch")
        self.assertEqual(dict(bridge.records_by_channel), {})

    def test_duplicate_channel_in_one_frame_is_explicitly_rejected(self) -> None:
        bridge = BaySpecWavelengthShiftBridge()
        result = bridge.ingest(
            {
                "channels": [
                    {"channel_id": "P22", "intensity_counts": 1000.0},
                    {"channel_id": "P22", "intensity_counts": 1001.0},
                ]
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["records_ingested"], 1)
        self.assertEqual(result["records_rejected"], 1)
        self.assertEqual(result["rejections"][0]["reason"], "duplicate_channel_id_in_frame")
        self.assertEqual(len(bridge.records_by_channel["P22"]), 1)

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

        self.assertEqual(payload["scope"], "optical_contact_position_and_continuous_fz")
        self.assertIsNone(payload["selected_channel"])
        self.assertEqual(payload["carrier_channel_id"], "P22")
        self.assertEqual(
            payload["carrier_channel_role"],
            "full_spectrum_transport_for_current_runtime",
        )
        self.assertFalse(payload["physical_channel_mapping_final"])
        self.assertEqual(payload["global_candidate_summary"]["valid_candidate_count"], 9)
        self.assertIn(
            payload["global_candidate_summary"]["dominant_candidate_id"],
            [f"FBG{index:02d}" for index in range(1, 10)],
        )
        self.assertEqual(
            payload["latest"]["carrier_channel_role"],
            "full_spectrum_transport_for_current_runtime",
        )
        self.assertEqual(
            [
                peak["candidate_id"]
                for peak in payload["latest"]["spectrum_peaks"]
                if peak.get("candidate_mapping")
            ],
            [f"FBG{index:02d}" for index in range(1, 10)],
        )
        self.assertTrue(payload["global_frame_qa"]["candidate_contract_complete"])
        self.assertFalse(payload["global_frame_qa"]["formal_recognition_allowed"])
        self.assertEqual(
            payload["global_candidate_summary"]["candidate_contract_complete"],
            payload["global_frame_qa"]["candidate_contract_complete"],
        )
        self.assertEqual(
            payload["global_candidate_summary"]["baseline_ready"],
            payload["global_frame_qa"]["candidate_baseline_ready"],
        )
        self.assertEqual(payload["source_fresh"], payload["global_frame_qa"]["source_fresh"])
        self.assertEqual(
            payload["formal_recognition_allowed"],
            payload["global_frame_qa"]["formal_recognition_allowed"],
        )
        self.assertEqual(payload["blockers"], payload["global_frame_qa"]["blockers"])
        self.assertEqual(
            payload["active_spectral_model_source"],
            payload["runtime_model"]["classification_model_source"],
        )
        self.assertEqual(
            payload["runtime_model"]["runtime_role"],
            "deployed_current_model_only",
        )
        self.assertEqual(
            payload["operator_visualization_frame"]["contract_version"],
            "touch_operator_visualization_v2",
        )

    def test_global_endpoint_does_not_run_model_for_qa_invalid_frame(self) -> None:
        latest = {
            "frame_id": 901,
            "timestamp": 12.5,
            "ingested_at": time.time(),
            "source": "static_http_ingest",
            "peak_axis_type": "wavelength_nm",
            "qa_status": "invalid",
            "qa_flags": ["spectrum_length_mismatch"],
            "wavelength_nm": [1540.0, 1540.1, 1540.2],
            "intensity": [100.0, 110.0, 105.0],
            "spectrum_peaks": [],
        }
        frame = {
            "frame_id": 901,
            "latest": latest,
            "trace": [],
            "array_frame": None,
        }
        stopped = {"active": False, "freshness": "stopped"}

        with (
            patch.object(backend_main.bridge, "frame", return_value=frame),
            patch.object(backend_main.export_watcher, "status", return_value=stopped),
            patch.object(backend_main.sdk_live_reader, "status", return_value=stopped),
            patch.object(backend_main, "_predict_current_runtime") as predict,
        ):
            payload = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=True,
            )

        predict.assert_not_called()
        self.assertFalse(payload["formal_recognition_allowed"])
        self.assertFalse(payload["global_frame_qa"]["runtime_baseline_ready"])
        self.assertEqual(
            payload["model_assisted_display_block_reason"],
            "spectrum_qa_invalid_for_formal_recognition",
        )
        self.assertFalse(payload["operator_visualization_frame"]["prediction_ready"])

    def test_same_live_frame_is_neutralized_when_source_turns_stale(self) -> None:
        latest = {
            "frame_id": 902,
            "timestamp": 12.6,
            "ingested_at": time.time(),
            "source": "bayspec_direct_sdk",
            "peak_axis_type": "wavelength_nm",
            "qa_status": "ok",
            "qa_flags": [],
            "wavelength_nm": [1540.0, 1540.1, 1540.2],
            "intensity": [100.0, 110.0, 105.0],
            "spectrum_peaks": [],
        }
        frame = {
            "frame_id": 902,
            "latest": latest,
            "trace": [],
            "array_frame": None,
        }
        watcher_stopped = {"active": False, "freshness": "stopped"}
        sdk_fresh = {
            "active": True,
            "freshness": "live",
            "acquisition_session_id": 44,
        }
        sdk_stale = {
            "active": True,
            "freshness": "stale",
            "acquisition_session_id": 44,
        }
        runtime_status = {
            "loaded": True,
            "classification_model_source": "current_classifier",
            "force_model_source": "current_force_model",
            "runtime_startup_baseline": {"state": {"ready": True}},
        }
        active_prediction = {
            "ok": True,
            "status": "ready",
            "classification_model_source": "current_classifier",
            "force_model_source": "current_force_model",
            "contact": {
                "label": "contact",
                "confidence": 0.93,
                "contact_probability": 0.96,
            },
            "position": {"label": "P21", "visual_label": "P21"},
            "force_fz": {"estimated_n": 2.0, "visual_drive_n": 2.0},
            "digital_twin": {
                "active": True,
                "visual_active": True,
                "position_id": "P21",
                "surface_grid": [
                    [0.05, 0.82, 0.04],
                    [0.03, 0.20, 0.02],
                    [0.01, 0.02, 0.01],
                ],
                "surface_metrics": {
                    "surface_peak": 0.82,
                    "surface_mean": 0.13,
                    "surface_area_active": 2 / 9,
                    "dominant_channel": "P21",
                },
            },
        }

        with (
            patch.object(
                backend_main.bridge,
                "frame",
                side_effect=lambda **_kwargs: {
                    **frame,
                    "latest": dict(latest),
                    "trace": list(frame["trace"]),
                },
            ),
            patch.object(
                backend_main.export_watcher,
                "status",
                return_value=watcher_stopped,
            ),
            patch.object(
                backend_main.sdk_live_reader,
                "status",
                side_effect=[sdk_fresh, sdk_stale],
            ),
            patch.object(
                backend_main,
                "_current_runtime_status",
                return_value=runtime_status,
            ),
            patch.object(
                backend_main,
                "_predict_current_runtime",
                return_value=active_prediction,
            ) as predict,
        ):
            fresh_payload = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=True,
            )
            stale_payload = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=True,
            )

        self.assertEqual(predict.call_count, 1)
        fresh_contract = fresh_payload["operator_visualization_frame"]
        stale_contract = stale_payload["operator_visualization_frame"]
        self.assertEqual(fresh_contract["frame_id"], stale_contract["frame_id"])
        self.assertEqual(fresh_contract["source_session_id"], 44)
        self.assertEqual(stale_contract["source_session_id"], 44)
        self.assertTrue(fresh_contract["prediction_ready"])
        self.assertEqual(fresh_contract["response_state"], "contact")
        self.assertGreater(
            fresh_contract["surface"]["surface_metrics"]["surface_peak"],
            0.0,
        )
        self.assertFalse(stale_contract["prediction_ready"])
        self.assertEqual(stale_contract["response_state"], "no_contact")
        self.assertEqual(
            stale_contract["response_block_reason"],
            "stale_or_mismatched_live_source",
        )
        self.assertEqual(
            stale_contract["surface"]["surface_metrics"]["surface_peak"],
            0.0,
        )
        self.assertEqual(
            stale_contract["surface"]["surface_grid"],
            [[0.0, 0.0, 0.0] for _ in range(3)],
        )

    def test_operator_contract_reports_concrete_runtime_model_sources(self) -> None:
        contract = backend_main._build_operator_visualization_frame(
            {
                "frame_id": 42,
                "timestamp": 12.5,
                "source": "unit_test_spectrum",
            },
            {
                "ok": True,
                "status": "ready",
                "classification_model_source": "classification_candidate_v3",
                "force_model_source": "force_candidate_v4",
                "contact": {"label": "no_contact"},
                "digital_twin": {"active": False, "visual_active": False},
            },
            ready=True,
            block_reason=None,
            source_session_id=17,
        )

        self.assertEqual(contract["model_source"], "classification_candidate_v3")
        self.assertEqual(
            contract["classification_model_source"],
            "classification_candidate_v3",
        )
        self.assertEqual(contract["force_model_source"], "force_candidate_v4")
        self.assertEqual(
            contract["logical_model_id"],
            "ordinary_fbg_same_day_joint_nine_fbg_beta_v4",
        )
        self.assertEqual(contract["source_session_id"], 17)

    def test_operator_contract_does_not_report_sync_without_a_source_frame(self) -> None:
        contract = backend_main._build_operator_visualization_frame(
            None,
            {
                "ok": False,
                "status": "current_runtime_source_blocked",
            },
            ready=False,
            block_reason="stale_or_mismatched_live_source",
        )

        self.assertIsNone(contract["frame_id"])
        self.assertEqual(contract["response_state"], "no_contact")
        self.assertEqual(contract["response_block_reason"], "no_source_frame")
        self.assertEqual(contract["sync"]["status"], "no_frame")
        self.assertIsNone(contract["sync"]["spectrum_frame_id"])
        self.assertIsNone(contract["sync"]["surface_frame_id"])
        self.assertEqual(
            contract["surface"]["surface_grid"],
            [[0.0, 0.0, 0.0] for _ in range(3)],
        )

    def test_frontend_global_normalization_marks_stale_frames(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        styles_css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        self.assertIn('status === "stale_frame"', app_js)
        self.assertIn('"stale_frame"', app_js)
        self.assertIn(
            'const GLOBAL_RECOGNITION_SCOPE = "optical_contact_position_and_continuous_fz"',
            app_js,
        )
        self.assertIn(
            'const CURRENT_RUNTIME_MODEL_SOURCE = "ordinary_fbg_same_day_joint_nine_fbg_beta_v4"',
            app_js,
        )
        self.assertIn("function normalizeCanonicalVisualizationFrame(frame, contract)", app_js)
        self.assertIn("function appendCurrentRuntimeTrace(rawRecord, contract)", app_js)
        self.assertIn('trace_response_semantics: "canonical_operator_display_force_n"', app_js)
        self.assertIn("contract?.force?.display_n", app_js)
        self.assertIn("operator_visualization_frame", app_js)
        self.assertIn("frameResponseIsUsable", app_js)
        self.assertIn("requestSequence < state.lastCommittedFrameRequest", app_js)
        self.assertIn("frameRequestInFlight", app_js)
        self.assertIn("forcedFrameRequestQueued", app_js)
        self.assertIn("if (state.frameRequestInFlight)", app_js)
        self.assertNotIn("globalCandidateSpatialProxy(validPeaks, rawTrace, baselineStatsByCandidate)", app_js)
        self.assertIn("#dfeef4 0%", styles_css)
        self.assertIn("#63b7b7 34%", styles_css)
        self.assertIn("#f0d57a 67%", styles_css)
        self.assertIn("#c8665f 100%", styles_css)
        self.assertNotIn("var(--response-no-contact-end", styles_css)
        self.assertIn("function opticalForceAxisStep(maximumN)", app_js)
        self.assertIn("function updateOpticalForceDisplayMaximum(valueN)", app_js)
        self.assertIn('"Optical Fz estimate (N)"', app_js)
        self.assertIn("function snapDisplayedFrameToCurrentTargets()", app_js)
        self.assertIn("state.smoothSurfaceVisualPeak = state.targetSurfaceVisualPeak", app_js)
        self.assertIn("snapDisplayedFrameToCurrentTargets();", app_js)
        self.assertIn('state.arrayDemoPlaybackMode !== "loop"', app_js)
        self.assertIn('const frameScenario = actionFinished ? "no_contact"', app_js)

    def test_operator_contract_keeps_semantic_inferred_and_observed_positions_separate(
        self,
    ) -> None:
        latest = {
            "frame_id": 42,
            "timestamp": 12.5,
            "source": "unit_test_spectrum",
        }
        prediction = {
            "ok": True,
            "status": "ready",
            "contact": {"label": "contact", "contact_probability": 0.94},
            "position": {"label": "P11", "visual_label": "P11"},
            "force_fz": {"visual_drive_n": 1.2, "estimated_n": 1.2},
            "observed_coupled_spectral_response": {
                "dominant_channel": "P33",
                "response_grid": [
                    [0.04, 0.06, 0.08],
                    [0.03, 0.05, 0.09],
                    [0.02, 0.04, 0.72],
                ],
            },
            "digital_twin": {
                "active": True,
                "visual_active": True,
                "position_id": "P11",
                "inferred_dominant_channel": "P21",
                "deformation_proxy": 0.7,
                "surface_grid": [
                    [0.30, 0.70, 0.10],
                    [0.10, 0.15, 0.05],
                    [0.04, 0.03, 0.02],
                ],
                "inferred_contact_probability_grid": [
                    [0.22, 0.52, 0.08],
                    [0.08, 0.06, 0.02],
                    [0.01, 0.005, 0.005],
                ],
                "raw_inferred_contact_probabilities": {
                    "P11": 0.28,
                    "P21": 0.62,
                    "P31": 0.10,
                },
                "inferred_contact_probabilities": {
                    "P11": 0.22,
                    "P21": 0.70,
                    "P31": 0.08,
                },
                "surface_metrics": {
                    "surface_peak": 0.70,
                    "surface_mean": 0.165,
                    "surface_area_active": 5 / 9,
                    "dominant_channel": "P21",
                },
            },
        }

        contract = backend_main._build_operator_visualization_frame(
            latest,
            prediction,
            ready=True,
            block_reason=None,
        )

        self.assertEqual(contract["contract_version"], "touch_operator_visualization_v2")
        self.assertEqual(contract["surface"]["semantic_position_id"], "P11")
        self.assertEqual(contract["surface"]["inferred_dominant_channel"], "P21")
        self.assertEqual(contract["surface"]["observed_dominant_channel"], "P33")
        self.assertEqual(
            contract["surface"]["raw_inferred_contact_probabilities"]["P21"],
            0.62,
        )
        self.assertEqual(
            contract["surface"]["smoothed_inferred_contact_probabilities"]["P21"],
            0.70,
        )
        self.assertEqual(contract["surface"]["surface_metrics"]["dominant_channel"], "P21")
        self.assertEqual(
            contract["surface"]["observed_coupled_spectral_response"]["dominant_channel"],
            "P33",
        )
        self.assertEqual(contract["sync"]["surface_frame_id"], 42)
        self.assertEqual(contract["sync"]["spectrum_frame_id"], 42)

    def test_operator_contract_masks_stale_inference_when_visual_contact_is_inactive(
        self,
    ) -> None:
        latest = {
            "frame_id": 43,
            "timestamp": 12.6,
            "source": "unit_test_spectrum",
        }
        prediction = {
            "ok": True,
            "status": "ready",
            "contact": {
                "label": "contact",
                "confidence": 0.82,
                "contact_probability": 0.82,
            },
            "position": {
                "label": "P23",
                "visual_label": "P23",
                "confidence": 0.91,
                "visual_confidence": 0.91,
                "raw_probabilities": {"P23": 0.91, "P22": 0.09},
                "visual_probabilities": {"P23": 0.88, "P22": 0.12},
            },
            "force_fz": {
                "estimated_n": 0.42,
                "continuous_estimated_n": 0.42,
                "visual_drive_n": 0.0,
            },
            "digital_twin": {
                "active": False,
                "visual_active": False,
                "position_id": "P23",
                "inferred_dominant_channel": "P23",
                "surface_grid": [
                    [0.02, 0.06, 0.10],
                    [0.04, 0.15, 0.35],
                    [0.03, 0.20, 0.82],
                ],
                "inferred_contact_probability_grid": [
                    [0.01, 0.03, 0.06],
                    [0.02, 0.10, 0.28],
                    [0.02, 0.14, 0.88],
                ],
                "raw_inferred_contact_probabilities": {"P23": 0.91},
                "inferred_contact_probabilities": {"P23": 0.88},
                "observed_coupled_spectral_response": {
                    "dominant_channel": "P32",
                    "response_grid": [
                        [0.01, 0.02, 0.03],
                        [0.04, 0.05, 0.06],
                        [0.07, 0.08, 0.09],
                    ],
                },
                "surface_metrics": {
                    "surface_peak": 0.82,
                    "surface_mean": 0.20,
                    "surface_area_active": 0.44,
                    "surface_centroid_x": 0.6,
                    "surface_centroid_y": -0.7,
                    "surface_spread": 0.31,
                    "dominant_channel": "P23",
                    "coupling_status": "stale_contact_value",
                },
            },
        }

        contract = backend_main._build_operator_visualization_frame(
            latest,
            prediction,
            ready=True,
            block_reason=None,
        )

        self.assertTrue(contract["prediction_ready"])
        self.assertEqual(contract["response_state"], "no_contact")
        self.assertEqual(contract["contact"]["label"], "no_contact")
        self.assertIsNone(contract["position"]["display_label"])
        self.assertIsNone(contract["position"]["formal_label"])
        self.assertEqual(contract["position"]["raw_probabilities"], {})
        self.assertEqual(contract["position"]["smoothed_visual_probabilities"], {})
        self.assertEqual(
            contract["surface"]["surface_grid"],
            [[0.0, 0.0, 0.0] for _ in range(3)],
        )
        self.assertEqual(
            contract["surface"]["inferred_contact_probability_grid"],
            [[0.0, 0.0, 0.0] for _ in range(3)],
        )
        self.assertEqual(contract["surface"]["raw_inferred_contact_probabilities"], {})
        self.assertEqual(
            contract["surface"]["smoothed_inferred_contact_probabilities"], {}
        )
        metrics = contract["surface"]["surface_metrics"]
        self.assertEqual(metrics["surface_peak"], 0.0)
        self.assertEqual(metrics["surface_mean"], 0.0)
        self.assertEqual(metrics["surface_area_active"], 0.0)
        self.assertEqual(metrics["surface_centroid_x"], 0.0)
        self.assertEqual(metrics["surface_centroid_y"], 0.0)
        self.assertEqual(metrics["surface_spread"], 0.0)
        self.assertIsNone(metrics["dominant_channel"])
        self.assertEqual(metrics["coupling_status"], "optical_model_no_contact")
        self.assertEqual(contract["trace_sample"]["surface_peak"], 0.0)
        self.assertEqual(contract["trace_sample"]["value_n"], 0.0)
        self.assertEqual(
            contract["surface"]["observed_coupled_spectral_response"]["dominant_channel"],
            "P32",
        )
        self.assertEqual(contract["sync"]["spectrum_frame_id"], 43)

    def test_operator_spectrum_entry_uses_accessible_card_without_redundant_icon(self) -> None:
        index_html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        spectrum_card = index_html.split('class="hud-card summary-hud"', 1)[1].split(
            "</div>", 1
        )[0]
        self.assertIn('role="button"', spectrum_card)
        self.assertIn('tabindex="0"', spectrum_card)
        self.assertIn('aria-label="Open spectrum drawer"', spectrum_card)
        self.assertNotIn("spectrum-open-chip", spectrum_card)

    def test_operator_status_strip_uses_compact_icons(self) -> None:
        index_html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        styles_css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

        self.assertIn('class="status-item qa-status-item"', index_html)
        self.assertIn('data-lucide="shield-check"', index_html)
        self.assertIn(".operator-mode .status-strip .status-item", styles_css)
        self.assertIn("min-height: 34px !important;", styles_css)
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
    SPECTRAL_CHANNEL_ORDER = [
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
    SCENARIO_STEPS = {
        "no_contact": 1,
        "center_press": 50,
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
        contact = backend_main.simulated_array_frame("center_press", step=30)
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

    def test_recorded_spectrum_peaks_follow_auto_discovered_wavelength_order(self) -> None:
        frame = backend_main.simulated_array_frame("center_press", step=30)
        spectrum = frame["spectrum"]
        peaks = spectrum["peaks"]
        self.assertEqual(len(peaks), 9)
        self.assertEqual(
            [item["provisional_channel_id"] for item in peaks],
            self.SPECTRAL_CHANNEL_ORDER,
        )
        references = [
            float(item["candidate_reference_wavelength_nm"]) for item in peaks
        ]
        self.assertEqual(references, sorted(references))
        for peak in peaks:
            self.assertLessEqual(
                abs(
                    float(peak["peak_wavelength_nm"])
                    - float(peak["candidate_reference_wavelength_nm"])
                ),
                0.85,
            )
            self.assertEqual(
                peak["peak_assignment_method"],
                "automatic_no_contact_discovery_then_local_tracking",
            )
            self.assertFalse(peak["physical_channel_mapping_final"])
        self.assertEqual(
            spectrum["spectrum_peak_profile"], "recorded_auto_discovered_9fbg"
        )
        self.assertEqual(
            spectrum["spectrum_peak_mapping_status"],
            "wavelength_order_assignment",
        )
        self.assertEqual(frame["frame_sync_status"], "synced")
        self.assertIn("wavelength-shift", frame["surface_title"])

    def test_recorded_spectrum_replaces_each_frame_and_preserves_measured_counts(self) -> None:
        frames = [
            backend_main.simulated_array_frame("no_contact", step=0),
            backend_main.simulated_array_frame("center_press", step=30),
            backend_main.simulated_array_frame("broad_fingertip_contact", step=7),
            backend_main.simulated_array_frame("vertical_slide_p11_p12_p13", step=5),
        ]
        maxima = [max(frame["spectrum"]["intensity"]) for frame in frames]
        for frame, maximum in zip(frames, maxima):
            self.assertTrue(math.isfinite(maximum))
            self.assertGreater(maximum, 0.0)
            self.assertTrue(frame["spectrum"]["intensity_modulation_enabled"])
            self.assertEqual(frame["spectrum"]["peak_height_mode"], "recorded_counts")
            self.assertEqual(frame["spectrum"]["frame_render_semantics"], "replace_previous_spectrum")
        self.assertGreater(len({round(value, 6) for value in maxima}), 1)

    def test_every_recorded_demo_frame_has_valid_auto_discovered_peak_contract(self) -> None:
        for scenario, step_count in self.SCENARIO_STEPS.items():
            for step in range(step_count):
                with self.subTest(scenario=scenario, step=step):
                    frame = backend_main.simulated_array_frame(scenario, step=step)
                    spectrum = frame["spectrum"]
                    wavelengths = spectrum["wavelength_nm"]
                    counts = spectrum["intensity"]
                    peaks = spectrum["peaks"]
                    self.assertEqual(len(wavelengths), 512)
                    self.assertEqual(len(counts), 512)
                    self.assertTrue(all(math.isfinite(value) for value in counts))
                    self.assertGreater(max(counts), 0.0)
                    self.assertEqual(len(peaks), 9)
                    self.assertEqual(
                        [item["provisional_channel_id"] for item in peaks],
                        self.SPECTRAL_CHANNEL_ORDER,
                    )
                    self.assertEqual(
                        spectrum["spectrum_peak_profile"],
                        "recorded_auto_discovered_9fbg",
                    )
                    self.assertEqual(
                        spectrum["frame_render_semantics"],
                        "replace_previous_spectrum",
                    )
                    self.assertFalse(spectrum["physical_channel_mapping_final"])

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
