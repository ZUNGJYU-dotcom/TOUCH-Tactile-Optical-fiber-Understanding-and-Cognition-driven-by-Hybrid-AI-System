from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from backend import main as backend_main
from backend.main import SenseExportWatcher, _model_display_source_gate
from bridge import BaySpecWavelengthShiftBridge
from desktop_launcher import health_payload_is_expected
from src.hybrid_spectrum.session_level_calibration import (
    CORE_FEATURE_NAMES,
    PerPositionOrdinalCalibrator,
)


class _AliveThread:
    def is_alive(self) -> bool:
        return True


class RuntimeResponsivenessConfigTests(unittest.TestCase):
    def test_temporal_history_preroll_is_enabled_for_live_runtime(self) -> None:
        config = backend_main._load_runtime_baseline_recovery_config()

        self.assertTrue(config["prime_temporal_history_with_baseline"])
        self.assertEqual(config["baseline_preroll_frames"], 20)


class ModelDisplaySourceGateTests(unittest.TestCase):
    def test_replay_is_allowed_when_live_sources_are_stopped(self) -> None:
        gate = _model_display_source_gate(
            {"source": "static_http_ingest"},
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertFalse(gate["source_fresh"])
        self.assertEqual(gate["model_input_source_mode"], "held_replay_or_http")

    def test_stale_sdk_frame_is_blocked(self) -> None:
        gate = _model_display_source_gate(
            {"source": "bayspec_direct_usb20bs_sdk"},
            {"active": False, "freshness": "stopped"},
            {"active": True, "freshness": "stale"},
        )
        self.assertFalse(gate["model_input_source_allowed"])
        self.assertEqual(gate["selected_live_source"], "sdk")

    def test_fresh_sdk_frame_is_allowed(self) -> None:
        gate = _model_display_source_gate(
            {"source": "bayspec_direct_usb20bs_sdk"},
            {"active": False, "freshness": "stopped"},
            {"active": True, "freshness": "live"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertTrue(gate["source_fresh"])

    def test_mismatched_buffered_source_is_blocked_during_live_session(self) -> None:
        gate = _model_display_source_gate(
            {"source": "static_http_ingest"},
            {"active": True, "freshness": "live"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertFalse(gate["model_input_source_allowed"])
        self.assertEqual(gate["selected_live_source"], "unmatched_live_source")


class StaticCandidateShadowTests(unittest.TestCase):
    def test_shadow_candidate_never_claims_operator_or_twin_control(self) -> None:
        class CandidateStub:
            def predict(self, *_args, **_kwargs):
                return {
                    "position": {"label": "P22"},
                    "force_level": {"label": "normal"},
                }

        wavelength = np.linspace(1528.0, 1560.0, 64).tolist()
        intensity = np.linspace(1000.0, 2000.0, 64).tolist()
        with patch.object(
            backend_main,
            "STATIC_SPECTRAL_CANDIDATE_PREDICTOR",
            CandidateStub(),
        ):
            result = backend_main._predict_static_spectral_shadow(
                wavelength,
                intensity,
                wavelength,
                intensity,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "shadow_ready")
        self.assertFalse(result["drives_operator_ui"])
        self.assertFalse(result["drives_digital_twin"])
        self.assertEqual(result["prediction"]["position"]["label"], "P22")

    def test_session_calibrated_force_is_baseline_bound_and_shadow_only(self) -> None:
        samples = [
            {
                "position": "P22",
                "level": level,
                "features": {name: value for name in CORE_FEATURE_NAMES},
            }
            for level, value in (("light", 1.0), ("normal", 2.0), ("hard", 4.0))
        ]
        calibrator = PerPositionOrdinalCalibrator.fit(
            samples,
            baseline_token="baseline-a",
            required_positions=("P22",),
        )
        prediction = {
            "response_calibration_features": {
                name: 2.2 for name in CORE_FEATURE_NAMES
            }
        }
        temporal = {
            "ready": True,
            "contact_label": "contact",
            "position_label": "P22",
        }
        with patch.object(
            backend_main,
            "STATIC_SPECTRAL_SESSION_CALIBRATOR",
            calibrator,
        ):
            result = backend_main._apply_session_level_calibration(
                prediction,
                temporal,
                baseline_token="baseline-a",
            )
            mismatch = backend_main._apply_session_level_calibration(
                prediction,
                temporal,
                baseline_token="baseline-b",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["label"], "normal")
        self.assertFalse(result["drives_operator_ui"])
        self.assertFalse(result["drives_digital_twin"])
        self.assertEqual(mismatch["status"], "baseline_mismatch_calibration_invalidated")

    def test_exact_spectrum_token_changes_with_one_sample(self) -> None:
        wavelength = [1.0, 2.0, 3.0]
        first = backend_main._spectrum_token(wavelength, [10.0, 20.0, 30.0])
        repeated = backend_main._spectrum_token(wavelength, [10.0, 20.0, 30.0])
        changed = backend_main._spectrum_token(wavelength, [10.0, 20.0, 30.001])

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)


class DynamicTemporalShadowEndpointTests(unittest.TestCase):
    @staticmethod
    def _spectrum_pair(*, frame_id: int, timestamp: float) -> dict:
        wavelength = np.linspace(1528.0, 1560.0, 64).tolist()
        baseline_intensity = np.linspace(1000.0, 2000.0, 64).tolist()
        current_intensity = [
            value + frame_id * 0.25 for value in baseline_intensity
        ]
        return {
            "ok": True,
            "latest": {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source": "static_http_ingest",
                "wavelength_nm": wavelength,
                "intensity": current_intensity,
            },
            "baseline": {
                "wavelength_nm": wavelength,
                "intensity": baseline_intensity,
            },
        }

    def test_unique_frames_advance_history_and_duplicate_poll_is_ignored(self) -> None:
        class AdapterStub:
            def __init__(self) -> None:
                self.bundle = {
                    "schema_version": "dynamic_temporal_shadow_candidate_v2",
                    "status": "shadow_only_not_primary",
                    "release_guard_grouped_cv": {
                        "unsafe_early_release_trigger_count": 0,
                    },
                }
                self.clear_count = 0
                self.baseline_count = 0
                self.run_inference_flags: list[bool] = []

            def clear(self) -> None:
                self.clear_count += 1

            def set_baseline(self, _wavelength, _intensity) -> None:
                self.baseline_count += 1

            def consume_pending_runtime_baseline_update(self):
                return None

            def update(
                self,
                _wavelength,
                _intensity,
                *,
                run_inference: bool,
                physical_frame: bool = True,
                external_no_contact_hint: bool | None = None,
                source_timestamp_sec: float | None = None,
            ):
                self.run_inference_flags.append(run_inference)
                return {
                    "status": (
                        "shadow_ready" if run_inference else "inference_stride_hold"
                    ),
                    "ready": run_inference,
                }

        adapter = AdapterStub()
        first_pair = self._spectrum_pair(frame_id=1, timestamp=1.0)
        second_pair = self._spectrum_pair(frame_id=2, timestamp=1.04)
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ), patch.object(
            backend_main.bridge,
            "spectral_model_input",
            side_effect=[first_pair, first_pair, second_pair],
        ):
            backend_main._reset_dynamic_temporal_shadow("test_setup")
            first = backend_main._predict_dynamic_temporal_shadow()
            duplicate = backend_main._predict_dynamic_temporal_shadow()
            second = backend_main._predict_dynamic_temporal_shadow()

        self.assertEqual(adapter.baseline_count, 1)
        self.assertEqual(adapter.run_inference_flags, [True, True])
        self.assertEqual(first["unique_frame_count"], 1)
        self.assertTrue(first["inference_executed_this_frame"])
        self.assertTrue(duplicate["duplicate_frame_ignored"])
        self.assertEqual(duplicate["unique_frame_count"], 1)
        self.assertEqual(second["unique_frame_count"], 2)
        self.assertTrue(second["inference_executed_this_frame"])
        for result in (first, duplicate, second):
            self.assertFalse(result["drives_operator_ui"])
            self.assertFalse(result["drives_digital_twin"])
            self.assertEqual(
                result["runtime_role"],
                "shadow_only_not_driving_digital_twin",
            )

    def test_slow_physical_frame_is_resampled_to_model_time_scale(self) -> None:
        class AdapterStub:
            bundle = {"frame_interval_sec_estimated": 0.04}

            def __init__(self) -> None:
                self.calls: list[dict] = []

            def clear(self) -> None:
                self.calls.clear()

            def set_baseline(self, _wavelength, _intensity) -> None:
                return None

            def consume_pending_runtime_baseline_update(self):
                return None

            def update(
                self,
                _wavelength,
                _intensity,
                *,
                run_inference: bool,
                physical_frame: bool,
                external_no_contact_hint: bool | None,
                source_timestamp_sec: float | None = None,
            ) -> dict:
                self.calls.append(
                    {
                        "run_inference": run_inference,
                        "physical_frame": physical_frame,
                        "external_no_contact_hint": external_no_contact_hint,
                    }
                )
                return {"status": "shadow_ready", "ready": True}

        adapter = AdapterStub()
        first_pair = self._spectrum_pair(frame_id=1, timestamp=1.0)
        second_pair = self._spectrum_pair(frame_id=2, timestamp=1.4)
        first_pair["latest"]["response_level"] = "no_contact"
        first_pair["latest"]["qa_status"] = "ok"
        second_pair["latest"]["response_level"] = "no_contact"
        second_pair["latest"]["qa_status"] = "ok"
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ), patch.object(
            backend_main.bridge,
            "spectral_model_input",
            side_effect=[first_pair, second_pair],
        ):
            backend_main._reset_dynamic_temporal_shadow("test_resample")
            first = backend_main._predict_dynamic_temporal_shadow()
            second = backend_main._predict_dynamic_temporal_shadow()

        self.assertEqual(first["temporal_resample_steps"], 1)
        self.assertEqual(second["temporal_resample_steps"], 10)
        self.assertEqual(len(adapter.calls), 11)
        self.assertEqual(sum(call["physical_frame"] for call in adapter.calls), 2)
        self.assertTrue(adapter.calls[-1]["run_inference"])
        self.assertTrue(adapter.calls[-1]["external_no_contact_hint"])

    def test_global_frame_does_not_run_dynamic_shadow_by_default(self) -> None:
        stopped = {"active": False, "freshness": "stopped"}
        with patch.object(
            backend_main.bridge,
            "frame",
            return_value={"ok": True, "latest": None},
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sense_controller,
            "status",
            return_value={"ok": True},
        ), patch.object(
            backend_main,
            "_predict_static_spectral_frame",
            return_value={"ok": False, "status": "not_requested"},
        ), patch.object(
            backend_main,
            "_predict_dynamic_temporal_shadow",
        ) as predict_dynamic:
            result = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=False,
                include_shadow=False,
                include_dynamic_shadow=False,
            )

        predict_dynamic.assert_not_called()
        shadow = result["dynamic_temporal_shadow"]
        self.assertEqual(shadow["status"], "dynamic_shadow_not_requested")
        self.assertFalse(shadow["drives_operator_ui"])
        self.assertFalse(shadow["drives_digital_twin"])

    def test_temporal_display_adapter_maps_active_prediction_to_twin_contract(self) -> None:
        payload = {
            "ok": True,
            "status": "shadow_ready",
            "prediction": {
                "ready": True,
                "status": "shadow_ready",
                "history_frames": 20,
                "required_frames": 20,
                "frame_counter": 31,
                "contact": {"label": "contact", "confidence": 0.94},
                "position": {"label": "P21", "confidence": 0.81},
                "response_level": {"label": "normal", "confidence": 0.76},
                "operational_state": "active_contact",
                "release_guard": {"release_latched": False},
                "digital_twin_proxy": {
                    "active": True,
                    "position_id": "P21",
                    "response_level": "normal",
                    "deformation_proxy": 0.58,
                    "surface_grid": [
                        [0.31, 0.58, 0.31],
                        [0.16, 0.31, 0.16],
                        [0.04, 0.08, 0.04],
                    ],
                    "surface_metrics": {
                        "surface_peak": 0.58,
                        "surface_centroid_x": 0.0,
                        "surface_centroid_y": 1.0,
                        "dominant_channel": "P21",
                    },
                    "visualization_semantics": "single_finger_contact_patch",
                    "physical_output_semantics": "uncalibrated_manual_response_level",
                },
            },
        }

        result = backend_main._dynamic_temporal_display_prediction(payload)

        self.assertTrue(result["ok"])
        self.assertTrue(result["drives_operator_ui"])
        self.assertTrue(result["drives_digital_twin"])
        self.assertFalse(result["deployment_ready"])
        prediction = result["prediction"]
        self.assertEqual(prediction["position"]["label"], "P21")
        self.assertEqual(prediction["force_level"]["label"], "normal")
        self.assertEqual(prediction["digital_twin"]["position_id"], "P21")
        self.assertEqual(prediction["digital_twin"]["deformation_proxy"], 0.58)

    def test_temporal_display_adapter_keeps_release_at_zero_deformation(self) -> None:
        payload = {
            "ok": True,
            "status": "released_residual_latched",
            "prediction": {
                "ready": True,
                "status": "released_residual_latched",
                "contact": {"label": "no_contact", "confidence": None},
                "position": None,
                "response_level": None,
                "operational_state": "no_contact_after_confirmed_release",
                "release_guard": {"release_latched": True},
                "digital_twin_proxy": {
                    "active": False,
                    "position_id": None,
                    "response_level": "no_contact",
                    "deformation_proxy": 0.0,
                    "surface_grid": [[0.0, 0.0, 0.0] for _ in range(3)],
                    "surface_metrics": {"surface_peak": 0.0},
                },
            },
        }

        result = backend_main._dynamic_temporal_display_prediction(payload)

        self.assertTrue(result["ok"])
        prediction = result["prediction"]
        self.assertEqual(prediction["contact"]["label"], "no_contact")
        self.assertFalse(prediction["digital_twin"]["active"])
        self.assertEqual(prediction["digital_twin"]["deformation_proxy"], 0.0)

    def test_temporal_display_adapter_blocks_warming_window(self) -> None:
        result = backend_main._dynamic_temporal_display_prediction(
            {
                "ok": True,
                "status": "window_warming_up",
                "prediction": {
                    "ready": False,
                    "status": "window_warming_up",
                    "history_frames": 7,
                    "required_frames": 20,
                },
            }
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "temporal_window_not_ready")
        self.assertEqual(result["history_frames"], 7)
        self.assertFalse(result["drives_digital_twin"])

    def test_global_frame_selects_temporal_prediction_in_validation_mode(self) -> None:
        stopped = {"active": False, "freshness": "stopped"}
        temporal_payload = {
            "ok": True,
            "status": "shadow_ready",
            "prediction": {
                "ready": True,
                "status": "shadow_ready",
                "contact": {"label": "contact", "confidence": 0.93},
                "position": {"label": "P32", "confidence": 0.79},
                "response_level": {"label": "light", "confidence": 0.71},
                "digital_twin_proxy": {
                    "active": True,
                    "position_id": "P32",
                    "response_level": "light",
                    "deformation_proxy": 0.28,
                    "surface_grid": [
                        [0.01, 0.04, 0.08],
                        [0.04, 0.12, 0.28],
                        [0.01, 0.04, 0.08],
                    ],
                    "surface_metrics": {
                        "surface_peak": 0.28,
                        "dominant_channel": "P32",
                    },
                },
            },
        }
        with patch.object(
            backend_main.bridge,
            "frame",
            return_value={"ok": True, "latest": None},
        ), patch.object(
            backend_main.export_watcher,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sdk_live_reader,
            "status",
            return_value=stopped,
        ), patch.object(
            backend_main.sense_controller,
            "status",
            return_value={"ok": True},
        ), patch.object(
            backend_main,
            "_predict_static_spectral_frame",
            return_value={"ok": True, "status": "ready", "prediction": {"source": "static"}},
        ) as predict_static, patch.object(
            backend_main,
            "_predict_dynamic_temporal_shadow",
            return_value=temporal_payload,
        ):
            result = backend_main.global_spectrum_frame(
                trace_limit=8,
                include_spectrum=False,
                include_shadow=False,
                include_dynamic_shadow=True,
                temporal_validation_mode=True,
            )

        self.assertTrue(result["temporal_validation_mode"])
        self.assertTrue(result["model_assisted_display_allowed"])
        self.assertEqual(
            result["active_spectral_model_source"],
            "dynamic_temporal_v3_validation",
        )
        self.assertEqual(
            result["active_spectral_prediction"]["position"]["label"],
            "P32",
        )
        self.assertEqual(
            result["active_spectral_prediction"]["digital_twin"]["deformation_proxy"],
            0.28,
        )
        predict_static.assert_not_called()
        self.assertEqual(
            result["trained_static_spectral_frame"]["status"],
            "skipped_temporal_validation_mode",
        )

    def test_reset_clears_cached_dynamic_trial_state(self) -> None:
        class AdapterStub:
            bundle = {}

            def __init__(self) -> None:
                self.clear_count = 0

            def clear(self) -> None:
                self.clear_count += 1

        adapter = AdapterStub()
        with patch.object(
            backend_main,
            "DYNAMIC_TEMPORAL_SHADOW_ADAPTER",
            adapter,
        ):
            backend_main.DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN = "old-baseline"
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY = ("old",)
            backend_main.DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT = 99
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD = {"old": True}
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC = 123.0
            backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM = np.ones(3)
            result = backend_main._reset_dynamic_temporal_shadow("baseline_replaced")

        self.assertTrue(result["ok"])
        self.assertEqual(adapter.clear_count, 1)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY)
        self.assertEqual(backend_main.DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT, 0)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC)
        self.assertIsNone(backend_main.DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM)


class ExportWatcherSessionTests(unittest.TestCase):
    def test_start_clears_previous_session_freshness(self) -> None:
        watcher = SenseExportWatcher()
        watcher.thread = _AliveThread()
        watcher.last_signature = ("old.csv", 1, 2)
        watcher.last_ingest_time = 123.0
        watcher.last_attempt_time = 123.0
        watcher.last_file = "old.csv"
        watcher.last_file_mtime = 123.0
        watcher.last_result = {"ok": True}
        watcher.ingest_count = 99

        status = watcher.start("P22", None, 0.35)

        self.assertEqual(status["freshness"], "waiting_for_export")
        self.assertIsNone(status["last_ingest_time"])
        self.assertIsNone(status["last_file"])
        self.assertEqual(status["ingest_count"], 0)
        self.assertEqual(watcher.last_signature, ("old.csv", 1, 2))

    def test_configuration_change_starts_clean_session(self) -> None:
        watcher = SenseExportWatcher()
        watcher.thread = _AliveThread()
        watcher.active = True
        watcher.channel_id = "P22"
        watcher.export_root = "old-root"
        watcher.last_signature = ("old.csv", 1, 2)
        watcher.last_ingest_time = time.time()

        status = watcher.start("P23", "new-root", 0.35)

        self.assertEqual(status["channel_id"], "P23")
        self.assertEqual(status["freshness"], "waiting_for_export")
        self.assertIsNone(watcher.last_signature)
        self.assertGreater(status["acquisition_session_id"], 0)

    def test_old_export_file_cannot_be_reported_live(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        watcher.last_ingest_time = time.time()
        watcher.last_file_mtime = time.time() - 120.0

        status = watcher.status()

        self.assertEqual(status["freshness"], "stale")
        self.assertGreater(status["seconds_since_last_file_update"], 100.0)


class AcquisitionSourceMutualExclusionTests(unittest.TestCase):
    def test_export_watch_start_stops_sdk_first(self) -> None:
        watcher_status = {"active": True, "freshness": "waiting_for_export"}
        sdk_status = {"active": False, "freshness": "stopped"}
        with patch.object(backend_main.sdk_live_reader, "stop", return_value=sdk_status) as stop_sdk, patch.object(
            backend_main.export_watcher, "start", return_value=watcher_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.export_watch_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
            )

        stop_sdk.assert_called_once_with()
        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])
        self.assertEqual(result["sdk_live"], sdk_status)

    def test_live_export_source_stops_sdk_first(self) -> None:
        watcher_status = {"active": True, "freshness": "waiting_for_export"}
        sdk_status = {"active": False, "freshness": "stopped"}
        with patch.object(backend_main.sdk_live_reader, "stop", return_value=sdk_status) as stop_sdk, patch.object(
            backend_main.export_watcher, "start", return_value=watcher_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
                control_sense=False,
                source="export_watch",
            )

        stop_sdk.assert_called_once_with()
        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertEqual(result["sdk_live"], sdk_status)

    def test_sdk_start_invalidates_previous_session_baseline(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        with patch.object(backend_main.export_watcher, "stop"), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.sdk_start(channel_id="P22", interval_ms=100, integration=40000)

        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])

    def test_direct_live_start_invalidates_previous_session_baseline(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        with patch.object(backend_main.export_watcher, "stop"), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.1,
                control_sense=False,
                source="direct_sdk",
            )

        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])


class BridgeResetTests(unittest.TestCase):
    @staticmethod
    def _stable_spectrum_records(
        spectrum: np.ndarray,
        *,
        frame_count: int = 30,
    ) -> list[dict[str, object]]:
        wavelength = np.linspace(1526.5, 1561.5, spectrum.size)
        records = []
        for index in range(frame_count):
            deterministic_noise = 0.25 * np.sin(
                np.linspace(0.0, 4.0 * np.pi, spectrum.size) + index * 0.07
            )
            frame = spectrum + deterministic_noise
            records.append(
                {
                    "timestamp": index * 0.1,
                    "intensity_counts": float(np.max(frame)),
                    "centroid_wavelength_nm": 1544.34,
                    "wavelength_nm": wavelength.tolist(),
                    "intensity": frame.tolist(),
                }
            )
        return records

    def test_recent_baseline_uses_minimum_sample_count_beyond_time_window(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=50)
        wavelength = [1540.0 + index * 0.1 for index in range(24)]
        test_bridge.records_by_channel["P22"] = [
            {
                "timestamp": float(index),
                "intensity_counts": 1000.0 + index,
                "centroid_wavelength_nm": 1546.89,
                "wavelength_nm": wavelength,
                "intensity": [1000.0 + index for _ in wavelength],
            }
            for index in range(30)
        ]

        recent = test_bridge._recent_records_for_baseline(
            "P22",
            minimum_samples=30,
        )

        self.assertEqual(len(recent), 30)
        self.assertEqual(recent[0]["timestamp"], 0.0)

    def test_keep_baseline_reset_clears_temporal_tracking(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=20)
        test_bridge.baseline_wavelength_by_channel["P22"] = 1546.89
        test_bridge.previous_tracked_wavelength_by_channel["P22"] = 1547.10

        result = test_bridge.reset(keep_baseline=True)

        self.assertTrue(result["ok"])
        self.assertEqual(test_bridge.baseline_wavelength_by_channel["P22"], 1546.89)
        self.assertNotIn("P22", test_bridge.previous_tracked_wavelength_by_channel)

    def test_stable_local_recovery_residual_is_rejected_against_session_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 9000.0 + 3200.0 * np.exp(-0.5 * (x / 0.24) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)

        first = test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        self.assertTrue(first["static_model_spectrum_baseline_ready"])
        self.assertEqual(
            first["baseline_anchor_comparison_by_channel"]["P22"]["status"],
            "trusted_anchor_initialized",
        )
        accepted_before = np.asarray(
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"]
        )

        local_residual = clean.copy()
        local_residual[48:78] -= 2600.0
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(
            local_residual
        )
        rejected = test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        self.assertTrue(rejected["ok"])
        self.assertTrue(rejected["static_model_spectrum_baseline_rejected"])
        self.assertFalse(rejected["static_model_spectrum_baseline_ready"])
        self.assertEqual(
            rejected["static_model_spectrum_baseline_status"],
            "recovery_residual_detected",
        )
        self.assertEqual(
            rejected["baseline_anchor_comparison_by_channel"]["P22"]["status"],
            "recovery_residual_detected",
        )
        np.testing.assert_allclose(
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"],
            accepted_before,
        )

    def test_common_gain_change_does_not_look_like_local_recovery_residual(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 8500.0 + 2800.0 * np.exp(-0.5 * (x / 0.28) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)
        test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(
            clean * 1.06
        )
        result = test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )

        self.assertTrue(result["static_model_spectrum_baseline_ready"])
        comparison = result["baseline_anchor_comparison_by_channel"]["P22"]
        self.assertNotEqual(comparison["status"], "recovery_residual_detected")
        self.assertAlmostEqual(comparison["common_gain_ratio"], 1.06, places=3)

    def test_new_acquisition_session_clears_trusted_baseline_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        x = np.linspace(-1.0, 1.0, 128)
        clean = 9000.0 + 3000.0 * np.exp(-0.5 * (x / 0.25) ** 2)
        test_bridge.records_by_channel["P22"] = self._stable_spectrum_records(clean)
        test_bridge.set_baseline(
            {"channel_id": "P22", "minimum_recent_samples": 30}
        )
        self.assertIn("P22", test_bridge.trusted_baseline_anchor_spectrum_by_channel)

        test_bridge.reset(keep_baseline=False)

        self.assertNotIn("P22", test_bridge.trusted_baseline_anchor_spectrum_by_channel)
        self.assertNotIn("P22", test_bridge.baseline_anchor_comparison_by_channel)

    def test_runtime_release_reanchor_preserves_trusted_session_anchor(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=100)
        wavelength = np.linspace(1528.0, 1560.0, 128)
        trusted = 8000.0 + 2500.0 * np.exp(
            -0.5 * ((wavelength - 1546.9) / 0.25) ** 2
        )
        recovered = trusted.copy()
        recovered[54:74] *= 0.96
        test_bridge.trusted_baseline_anchor_spectrum_by_channel["P22"] = {
            "wavelength_nm": wavelength.tolist(),
            "intensity": trusted.tolist(),
        }

        result = test_bridge.set_runtime_recovery_spectrum_baseline(
            "P22",
            wavelength,
            recovered,
            sample_count=14,
            span_sec=5.2,
            shape_motion_rms=0.0016,
            common_gain_motion=0.0004,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["trusted_session_anchor_preserved"])
        self.assertEqual(
            test_bridge.baseline_spectrum_status_by_channel["P22"],
            "stable_post_release_recovery_baseline",
        )
        np.testing.assert_allclose(
            test_bridge.baseline_spectrum_by_channel["P22"]["intensity"],
            recovered,
        )
        np.testing.assert_allclose(
            test_bridge.trusted_baseline_anchor_spectrum_by_channel["P22"][
                "intensity"
            ],
            trusted,
        )


class UnifiedBaselineEndpointTests(unittest.TestCase):
    def test_global_baseline_passes_required_frames_to_model_baseline(self) -> None:
        candidate_result = {"ok": True, "frame_count": 30}
        model_result = {
            "ok": True,
            "baseline_set": True,
            "static_model_spectrum_baseline_ready": True,
            "static_model_spectrum_baseline_status": "stable_post_release_recovery_baseline",
            "baseline_spectrum_sample_count_by_channel": {"P22": 30},
            "baseline_spectrum_span_sec_by_channel": {"P22": 3.0},
            "baseline_spectrum_noise_ratio_by_channel": {"P22": 0.001},
            "baseline_spectrum_drift_ratio_by_channel": {"P22": 0.001},
        }
        with patch.object(
            backend_main.bridge,
            "set_global_candidate_baseline",
            return_value=candidate_result,
        ), patch.object(
            backend_main.bridge,
            "set_baseline",
            return_value=model_result,
        ) as set_model_baseline:
            result = backend_main.set_global_candidate_baseline(minimum_frames=30)

        set_model_baseline.assert_called_once_with(
            {
                "channel_id": "P22",
                "baseline_method": "frozen_baseline",
                "minimum_recent_samples": 30,
            }
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["candidate_display_baseline_ok"])
        self.assertTrue(result["static_model_spectrum_baseline"]["ok"])

    def test_global_baseline_rejects_partial_model_baseline(self) -> None:
        with patch.object(
            backend_main.bridge,
            "set_global_candidate_baseline",
            return_value={"ok": True, "frame_count": 30},
        ) as set_candidate_baseline, patch.object(
            backend_main.bridge,
            "set_baseline",
            return_value={
                "ok": True,
                "baseline_set": True,
                "static_model_spectrum_baseline_ready": False,
                "static_model_spectrum_baseline_status": "insufficient_recovery_baseline_frames",
                "baseline_spectrum_sample_count_by_channel": {"P22": 6},
            },
        ):
            result = backend_main.set_global_candidate_baseline(minimum_frames=30)

        self.assertFalse(result["ok"])
        self.assertIn("insufficient_recovery_baseline_frames", result["message"])
        set_candidate_baseline.assert_not_called()


class DesktopLauncherIdentityTests(unittest.TestCase):
    def test_expected_backend_identity_is_accepted(self) -> None:
        self.assertTrue(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Trained Static Spectrum Twin",
                    "mode": "standalone_bayspec_trained_static_spectrum_twin",
                    "trained_static_model_primary": True,
                }
            )
        )

    def test_other_touch_backend_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Optical Intensity Twin",
                    "mode": "standalone_bayspec_optical_intensity",
                    "trained_static_model_primary": False,
                }
            )
        )


class OperatorQaProjectionTests(unittest.TestCase):
    def test_estimator_disagreement_is_diagnostics_only_in_trained_ui(self) -> None:
        app_js = (
            backend_main.FRONTEND_ROOT / "app.js"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'new Set(["wavelength_estimator_disagreement"])',
            app_js,
        )
        self.assertIn("diagnostic_only_qa_flags: diagnosticOnlyRawQaFlags", app_js)
        self.assertIn("carrier_qa_flags: rawQaFlags", app_js)


if __name__ == "__main__":
    unittest.main()
