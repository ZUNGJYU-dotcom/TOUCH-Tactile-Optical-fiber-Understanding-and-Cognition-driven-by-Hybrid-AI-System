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
        backend_main.bridge.reset(keep_baseline=False)
        baseline_frame = {
            "channel_id": "P22",
            "wavelength_nm": self.baseline_wavelength,
            "intensity": self.baseline_intensity,
            "source": "unit_test_fixture",
        }
        backend_main.bridge.ingest({**baseline_frame, "timestamp": 1000.0})
        blocked = backend_main._predict_static_spectral_frame()
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["status"], "baseline_required")

        insufficient_result = backend_main.bridge.set_baseline(
            {"channel_id": "P22", "baseline_method": "frozen_baseline"}
        )
        self.assertTrue(insufficient_result["ok"])
        self.assertFalse(insufficient_result["static_model_spectrum_baseline_ready"])
        self.assertEqual(
            insufficient_result["static_model_spectrum_baseline_status"],
            "insufficient_recovery_baseline_frames",
        )
        still_blocked = backend_main._predict_static_spectral_frame()
        self.assertFalse(still_blocked["ok"])

        for index in range(1, 20):
            backend_main.bridge.ingest(
                {**baseline_frame, "timestamp": 1000.0 + 0.04 * index}
            )
        baseline_result = backend_main.bridge.set_baseline(
            {"channel_id": "P22", "baseline_method": "frozen_baseline"}
        )
        self.assertTrue(baseline_result["static_model_spectrum_baseline_ready"])
        self.assertEqual(
            baseline_result["static_model_spectrum_baseline_status"],
            "stable_post_release_recovery_baseline",
        )
        ready = backend_main._predict_static_spectral_frame()
        self.assertTrue(ready["ok"])
        self.assertEqual(ready["prediction"]["contact"]["label"], "no_contact")
        self.assertEqual(
            ready["input"]["baseline_spectrum_semantic_role"],
            "post_press_release_recovery_no_contact",
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
        self.assertTrue(baseline_result["static_model_spectrum_baseline_ready"])

        wavelength, intensity = read_spectrum(FIXTURE_ROOT / "P22_hard_manual.csv")
        backend_main.bridge.ingest(
            {
                "channel_id": "P22",
                "wavelength_nm": wavelength,
                "intensity": intensity,
                "timestamp": 2000.0,
                "source": "unit_test_fixture",
            }
        )
        frame = backend_main.global_spectrum_frame(
            trace_limit=20,
            include_spectrum=True,
            include_shadow=True,
        )
        prediction = frame["trained_static_spectral_prediction"]

        self.assertTrue(frame["model_assisted_display_allowed"])
        self.assertEqual(prediction["contact"]["label"], "contact")
        self.assertEqual(prediction["position"]["label"], "P22")
        self.assertEqual(prediction["force_level"]["label"], "hard")
        self.assertTrue(prediction["digital_twin"]["active"])
        self.assertFalse(frame["physical_channel_mapping_final"])
        self.assertEqual(
            frame["scope"],
            "global_3x3_hybrid_spectral_fingerprint",
        )
        model_status = frame["trained_static_spectral_model"]
        self.assertEqual(len(model_status["observed_model_feature_windows_nm"]), 9)
        self.assertLess(
            model_status["observed_model_feature_window_range_nm"][0],
            1529.0,
        )
        self.assertFalse(model_status["future_3x3_target_plan_active"])
        self.assertFalse(frame["trained_static_spectral_frame"]["cache_hit"])
        shadow = frame["trained_static_spectral_shadow"]
        self.assertEqual(shadow["status"], "shadow_ready")
        self.assertFalse(shadow["drives_operator_ui"])
        self.assertFalse(shadow["drives_digital_twin"])
        self.assertIsNotNone(frame["trained_static_spectral_shadow_prediction"])
        self.assertEqual(
            frame["trained_static_spectral_model"]["shadow_candidate"]["runtime_role"],
            "shadow_only_not_driving_digital_twin",
        )
        cached_frame = backend_main.global_spectrum_frame(
            trace_limit=20,
            include_spectrum=False,
            include_shadow=True,
        )
        self.assertTrue(cached_frame["trained_static_spectral_frame"]["cache_hit"])
        self.assertEqual(
            cached_frame["trained_static_spectral_prediction"],
            prediction,
        )
        self.assertEqual(
            cached_frame["trained_static_spectral_shadow_prediction"],
            frame["trained_static_spectral_shadow_prediction"],
        )
        operator_frame = backend_main.global_spectrum_frame(
            trace_limit=20,
            include_spectrum=False,
            include_shadow=False,
        )
        self.assertEqual(
            operator_frame["trained_static_spectral_shadow"]["status"],
            "shadow_not_requested",
        )
        self.assertIsNone(
            operator_frame["trained_static_spectral_shadow_prediction"]
        )
        self.assertFalse(
            operator_frame["trained_static_spectral_shadow"]["drives_operator_ui"]
        )
        backend_main.bridge.reset(keep_baseline=False)


if __name__ == "__main__":
    unittest.main()
