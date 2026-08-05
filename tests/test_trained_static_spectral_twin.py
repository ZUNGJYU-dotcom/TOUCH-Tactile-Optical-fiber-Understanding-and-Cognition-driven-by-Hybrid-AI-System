from __future__ import annotations

import csv
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "bayspec_wavelength_shift_app"
for path in (ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend import main as backend_main  # noqa: E402
from src.hybrid_spectrum.static_model_adapter import StaticSpectralPredictor  # noqa: E402


FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "static_spectra"


def read_spectrum(path: Path) -> tuple[list[float], list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    start = next(index for index, row in enumerate(rows) if row[:2] == ["WL", "Power"]) + 1
    stop = next(index for index, row in enumerate(rows[start:], start) if row and row[0] == "Peak_Count")
    wavelength = [float(row[0]) for row in rows[start:stop] if len(row) >= 2 and row[0] and row[1]]
    intensity = [float(row[1]) for row in rows[start:stop] if len(row) >= 2 and row[0] and row[1]]
    return wavelength, intensity


class TrainedStaticSpectrumTwinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = StaticSpectralPredictor(
            ROOT / "models" / "static_spectral_recognition_bundle.joblib"
        )
        cls.baseline_wavelength, cls.baseline_intensity = read_spectrum(
            FIXTURE_ROOT / "baseline_no_contact_late.csv"
        )

    def direct_prediction(self, filename: str) -> dict:
        wavelength, intensity = read_spectrum(FIXTURE_ROOT / filename)
        return self.predictor.predict(
            wavelength,
            intensity,
            baseline_wavelength_nm=self.baseline_wavelength,
            baseline_intensity_counts=self.baseline_intensity,
        )

    def test_probability_unavailable_is_json_safe_and_requires_review(self) -> None:
        class PredictOnlyModel:
            def predict(self, _matrix):
                return ["contact"]

        result = StaticSpectralPredictor._predict_with_probabilities(
            {
                "model": PredictOnlyModel(),
                "model_id": "predict_only_test",
                "feature_set": "engineered",
            },
            [[0.0]],
        )

        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["margin"])
        self.assertEqual(result["confidence_source"], "unavailable")
        self.assertTrue(result["review_needed"])

    def test_strong_baseline_relative_change_overrides_legacy_no_contact(self) -> None:
        model_contact = {
            "label": "no_contact",
            "confidence": 1.0,
            "margin": 1.0,
            "probabilities": {"contact": 0.0, "no_contact": 1.0},
            "confidence_source": "uncalibrated_predict_proba",
            "probability_calibrated": False,
            "review_needed": False,
        }
        evidence = self.predictor._baseline_relative_contact_evidence(
            {
                "global_normalized_residual_rms": 0.060,
                "global_normalized_residual_peak": 0.310,
                "global_derivative_residual_energy": 0.0008,
                "global_shape_correlation": 0.996,
            }
        )

        result = self.predictor._resolve_contact_decision(model_contact, evidence)

        self.assertEqual(result["label"], "contact")
        self.assertEqual(result["decision_source"], "baseline_relative_spectral_change_gate")
        self.assertTrue(result["model_rule_disagreement"])
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["review_needed"])

    def test_model_contact_without_baseline_change_is_suppressed(self) -> None:
        model_contact = {
            "label": "contact",
            "confidence": 0.98,
            "margin": 0.96,
            "probabilities": {"contact": 0.98, "no_contact": 0.02},
            "confidence_source": "uncalibrated_predict_proba",
            "probability_calibrated": False,
            "review_needed": False,
        }
        evidence = self.predictor._baseline_relative_contact_evidence(
            {
                "global_normalized_residual_rms": 0.006,
                "global_normalized_residual_peak": 0.030,
                "global_derivative_residual_energy": 0.00001,
                "global_shape_correlation": 0.99998,
            }
        )

        result = self.predictor._resolve_contact_decision(model_contact, evidence)

        self.assertEqual(result["label"], "no_contact")
        self.assertEqual(
            result["decision_source"],
            "model_contact_suppressed_without_baseline_change",
        )
        self.assertTrue(result["model_rule_disagreement"])

    def test_representative_no_contact_is_below_physical_gate(self) -> None:
        prediction = self.predictor.predict(
            self.baseline_wavelength,
            self.baseline_intensity,
            baseline_wavelength_nm=self.baseline_wavelength,
            baseline_intensity_counts=self.baseline_intensity,
        )

        evidence = prediction["contact"]["baseline_relative_evidence"]
        self.assertFalse(evidence["strong_contact_evidence"])
        self.assertFalse(evidence["supporting_contact_evidence"])
        self.assertEqual(prediction["contact"]["label"], "no_contact")

    def test_representative_manual_position_and_level_contract(self) -> None:
        expected = {
            "P22_hard_manual.csv": ("P22", "hard", 0.0, 0.0),
            "P13_light_manual.csv": ("P13", "light", -1.0, -1.0),
            "P31_normal_manual.csv": ("P31", "normal", 1.0, 1.0),
        }
        for filename, contract in expected.items():
            with self.subTest(filename=filename):
                prediction = self.direct_prediction(filename)
                self.assertEqual(prediction["contact"]["label"], "contact")
                self.assertEqual(prediction["position"]["label"], contract[0])
                self.assertEqual(prediction["force_level"]["label"], contract[1])
                self.assertEqual(prediction["digital_twin"]["center_x"], contract[2])
                self.assertEqual(prediction["digital_twin"]["center_y"], contract[3])
                self.assertTrue(prediction["force_model_scope"].startswith("position_conditioned:"))
                self.assertEqual(
                    prediction["position"]["confidence_source"],
                    "uncalibrated_predict_proba",
                )
                self.assertFalse(prediction["position"]["probability_calibrated"])
                self.assertEqual(
                    prediction["uncertainty"]["policy"],
                    "diagnostic_only_does_not_change_prediction",
                )

    def test_position_is_stable_across_early_and_late_recovery_baselines(self) -> None:
        expected_positions = {
            "P22_hard_manual.csv": "P22",
            "P13_light_manual.csv": "P13",
            "P31_normal_manual.csv": "P31",
        }
        for baseline_filename in ("baseline_no_contact.csv", "baseline_no_contact_late.csv"):
            baseline_wavelength, baseline_intensity = read_spectrum(FIXTURE_ROOT / baseline_filename)
            for filename, expected_position in expected_positions.items():
                with self.subTest(baseline=baseline_filename, filename=filename):
                    wavelength, intensity = read_spectrum(FIXTURE_ROOT / filename)
                    prediction = self.predictor.predict(
                        wavelength,
                        intensity,
                        baseline_wavelength_nm=baseline_wavelength,
                        baseline_intensity_counts=baseline_intensity,
                    )
                    self.assertEqual(prediction["position"]["label"], expected_position)

    def test_no_contact_suppresses_deformation(self) -> None:
        prediction = self.predictor.predict(
            self.baseline_wavelength,
            self.baseline_intensity,
            baseline_wavelength_nm=self.baseline_wavelength,
            baseline_intensity_counts=self.baseline_intensity,
        )
        self.assertEqual(prediction["contact"]["label"], "no_contact")
        self.assertFalse(prediction["digital_twin"]["active"])
        self.assertEqual(prediction["digital_twin"]["deformation_proxy"], 0.0)

    def test_backend_blocks_inference_until_runtime_baseline_exists(self) -> None:
        backend_main._reset_current_runtime("unit_test_new_session")
        backend_main.bridge.reset(keep_baseline=False)
        baseline_frame = {
            "channel_id": "P22",
            "wavelength_nm": self.baseline_wavelength,
            "intensity": self.baseline_intensity,
            "source": "unit_test_fixture",
        }
        backend_main.bridge.ingest({**baseline_frame, "timestamp": 1000.0})
        blocked = backend_main._predict_current_runtime()
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["status"], "startup_baseline_collecting")

        insufficient_result = backend_main.bridge.set_baseline(
            {"channel_id": "P22", "baseline_method": "frozen_baseline"}
        )
        self.assertTrue(insufficient_result["ok"])
        self.assertFalse(insufficient_result["current_runtime_spectrum_baseline_ready"])
        self.assertEqual(
            insufficient_result["current_runtime_spectrum_baseline_status"],
            "insufficient_recovery_baseline_frames",
        )
        still_blocked = backend_main._predict_current_runtime()
        self.assertFalse(still_blocked["ok"])

        for index in range(1, 20):
            backend_main.bridge.ingest(
                {**baseline_frame, "timestamp": 1000.0 + 0.04 * index}
            )
        baseline_result = backend_main.bridge.set_baseline(
            {"channel_id": "P22", "baseline_method": "frozen_baseline"}
        )
        self.assertTrue(baseline_result["current_runtime_spectrum_baseline_ready"])
        self.assertEqual(
            baseline_result["current_runtime_spectrum_baseline_status"],
            "stable_post_release_recovery_baseline",
        )
        ready = backend_main._predict_current_runtime()
        self.assertTrue(ready["ok"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["runtime_role"], "deployed_current_model_only")
        self.assertEqual(ready["contact"]["label"], "no_contact")
        self.assertEqual(
            baseline_result["baseline_spectrum_semantic_role_by_channel"]["P22"],
            "post_press_release_recovery_no_contact",
        )

    def test_backend_builds_current_session_startup_baseline_automatically(self) -> None:
        backend_main._reset_current_runtime("unit_test_startup_baseline")
        backend_main.bridge.reset(keep_baseline=False)
        baseline_frame = {
            "channel_id": "P22",
            "wavelength_nm": self.baseline_wavelength,
            "intensity": self.baseline_intensity,
            "source": "unit_test_startup_fixture",
        }

        result = None
        for index in range(5):
            backend_main.bridge.ingest(
                {**baseline_frame, "timestamp": 2000.0 + 0.04 * index}
            )
            result = backend_main._predict_current_runtime()

        self.assertIsNotNone(result)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready")
        pair = backend_main.bridge.spectral_model_input(channel_id="P22")
        self.assertTrue(pair["ok"])
        self.assertEqual(
            pair["baseline_spectrum_status"],
            "stable_current_session_startup_baseline",
        )
        self.assertEqual(
            pair["baseline_spectrum_semantic_role"],
            "automatic_current_session_startup_no_contact",
        )
        self.assertIn(
            "P22",
            backend_main.bridge.trusted_baseline_anchor_spectrum_by_channel,
        )

    def test_global_api_contract_keeps_model_position_and_physical_map_semantics_separate(self) -> None:
        backend_main.export_watcher.stop()
        backend_main.sdk_live_reader.stop()
        backend_main.bridge.reset(keep_baseline=False)
        baseline_frame = {
            "channel_id": "P22",
            "wavelength_nm": self.baseline_wavelength,
            "intensity": self.baseline_intensity,
            "source": "unit_test_fixture",
        }
        for index in range(24):
            backend_main.bridge.ingest(
                {**baseline_frame, "timestamp": 1000.0 + 0.04 * index}
            )
        baseline_result = backend_main.bridge.set_baseline(
            {"channel_id": "P22", "baseline_method": "frozen_baseline"}
        )
        self.assertTrue(baseline_result["current_runtime_spectrum_baseline_ready"])

        # A current-runtime integration probe must use the same wavelength grid
        # as its frozen baseline. Legacy static fixtures use a different grid.
        backend_main.bridge.ingest(
            {
                "channel_id": "P22",
                "wavelength_nm": self.baseline_wavelength,
                "intensity": self.baseline_intensity,
                "timestamp": 2000.0,
                "source": "unit_test_fixture",
            }
        )
        frame = backend_main.global_spectrum_frame(
            trace_limit=20,
            include_spectrum=True,
        )
        self.assertFalse(frame["physical_channel_mapping_final"])
        self.assertEqual(
            frame["scope"],
            "optical_contact_position_and_continuous_fz",
        )
        self.assertEqual(
            frame["active_spectral_model_source"],
            "ordinary_fbg_all_data_beta_v1",
        )
        self.assertEqual(
            frame["runtime_model"]["runtime_role"],
            "deployed_current_model_only",
        )
        contract = frame["operator_visualization_frame"]
        self.assertEqual(
            contract["contract_version"],
            "touch_operator_visualization_v1",
        )
        self.assertEqual(contract["model_source"], "ordinary_fbg_all_data_beta_v1")
        self.assertTrue(contract["prediction_ready"])
        self.assertEqual(contract["response_state"], "no_contact")
        self.assertFalse(contract["force"]["upper_limit_applied"])
        sync_ids = {
            contract["sync"][name]
            for name in (
                "spectrum_frame_id",
                "surface_frame_id",
                "trace_frame_id",
                "summary_frame_id",
                "force_frame_id",
            )
        }
        self.assertEqual(sync_ids, {contract["frame_id"]})
        self.assertNotIn("trained_static_spectral_prediction", frame)
        self.assertNotIn("trained_static_spectral_shadow", frame)
        backend_main.bridge.reset(keep_baseline=False)


if __name__ == "__main__":
    unittest.main()
