from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from src.hybrid_spectrum.dynamic_shadow_adapter import (
    DynamicTemporalShadowAdapter,
    ReleaseResidualGuard,
    RuntimeBaselineRecoveryGuard,
    load_dynamic_shadow_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_MODEL = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "dynamic_temporal_shadow_candidate_v1.joblib"
)
V2_MODEL = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "dynamic_temporal_shadow_candidate_v2.joblib"
)
V3_MODEL = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "dynamic_temporal_shadow_candidate_v3_compact_runtime_pos240.joblib"
)
PEAK_CONFIG = PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml"


class ReleaseResidualGuardTests(unittest.TestCase):
    def test_release_requires_hard_arm_then_exit_and_event(self) -> None:
        guard = ReleaseResidualGuard(
            {
                "enabled": True,
                "hard_arm_probability": 0.50,
                "hard_arm_frames": 4,
                "hard_exit_probability": 0.30,
                "hard_exit_frames": 2,
                "release_event_probability": 0.60,
                "baseline_reset_required_after_latch": True,
            }
        )
        for _ in range(3):
            state = guard.update(
                hard_probability=0.90,
                release_event_probability=0.95,
            )
            self.assertFalse(state["armed"])
            self.assertFalse(state["release_latched"])
        state = guard.update(
            hard_probability=0.90,
            release_event_probability=0.05,
        )
        self.assertTrue(state["armed"])
        self.assertFalse(state["release_latched"])

        state = guard.update(
            hard_probability=0.20,
            release_event_probability=0.95,
        )
        self.assertFalse(state["release_latched"])
        state = guard.update(
            hard_probability=0.20,
            release_event_probability=0.95,
        )
        self.assertTrue(state["just_latched"])
        self.assertTrue(state["release_latched"])

        guard.reset()
        self.assertFalse(
            guard.snapshot(
                release_event_probability=None,
                just_latched=False,
            )["release_latched"]
        )

    def test_latched_release_rearms_after_quiet_then_sustained_contact(self) -> None:
        guard = ReleaseResidualGuard(
            {
                "enabled": True,
                "hard_arm_probability": 0.50,
                "hard_arm_frames": 1,
                "hard_exit_probability": 0.30,
                "hard_exit_frames": 1,
                "release_event_probability": 0.60,
                "auto_rearm_enabled": True,
                "auto_rearm_quiet_frames": 2,
                "auto_rearm_contact_frames": 3,
                "auto_rearm_contact_probability": 0.85,
                "auto_rearm_max_release_probability": 0.30,
            }
        )
        guard.update(
            hard_probability=0.90,
            release_event_probability=0.05,
            contact_probability=0.98,
            raw_contact_label="contact",
        )
        latched = guard.update(
            hard_probability=0.10,
            release_event_probability=0.95,
            contact_probability=0.90,
            raw_contact_label="contact",
        )
        self.assertTrue(latched["release_latched"])

        for _ in range(2):
            quiet = guard.update(
                hard_probability=0.0,
                release_event_probability=None,
                contact_probability=0.05,
                raw_contact_label="no_contact",
            )
        self.assertTrue(quiet["auto_rearm_ready"])

        for _ in range(2):
            held = guard.update(
                hard_probability=0.10,
                release_event_probability=0.05,
                contact_probability=0.96,
                raw_contact_label="contact",
            )
            self.assertTrue(held["release_latched"])
        rearmed = guard.update(
            hard_probability=0.10,
            release_event_probability=0.05,
            contact_probability=0.96,
            raw_contact_label="contact",
        )
        self.assertTrue(rearmed["just_rearmed"])
        self.assertFalse(rearmed["release_latched"])

    def test_latched_release_does_not_rearm_from_weak_or_release_like_contact(self) -> None:
        guard = ReleaseResidualGuard(
            {
                "enabled": True,
                "hard_arm_probability": 0.50,
                "hard_arm_frames": 1,
                "hard_exit_probability": 0.30,
                "hard_exit_frames": 1,
                "release_event_probability": 0.60,
                "auto_rearm_enabled": True,
                "auto_rearm_quiet_frames": 1,
                "auto_rearm_contact_frames": 2,
                "auto_rearm_contact_probability": 0.85,
                "auto_rearm_max_release_probability": 0.30,
            }
        )
        guard.update(
            hard_probability=0.90,
            release_event_probability=0.05,
            contact_probability=0.98,
            raw_contact_label="contact",
        )
        guard.update(
            hard_probability=0.10,
            release_event_probability=0.95,
            contact_probability=0.90,
            raw_contact_label="contact",
        )
        guard.update(
            hard_probability=0.0,
            release_event_probability=None,
            contact_probability=0.05,
            raw_contact_label="no_contact",
        )

        weak = guard.update(
            hard_probability=0.10,
            release_event_probability=0.05,
            contact_probability=0.70,
            raw_contact_label="contact",
        )
        release_like = guard.update(
            hard_probability=0.10,
            release_event_probability=0.80,
            contact_probability=0.98,
            raw_contact_label="contact",
        )
        self.assertTrue(weak["release_latched"])
        self.assertTrue(release_like["release_latched"])
        self.assertFalse(release_like["just_rearmed"])


class RuntimeBaselineRecoveryGuardTests(unittest.TestCase):
    @staticmethod
    def _guard() -> RuntimeBaselineRecoveryGuard:
        return RuntimeBaselineRecoveryGuard(
            {
                "suppress_after_physical_frames": 2,
                "reanchor_after_physical_frames": 3,
                "release_probability_threshold": 0.40,
            }
        )

    def test_stable_release_reanchors_only_after_physical_frames(self) -> None:
        guard = self._guard()
        baseline = np.linspace(1000.0, 2000.0, 64)
        for index in range(2):
            state, recovered = guard.observe(
                baseline + index * 0.01,
                physical_frame=True,
                external_no_contact_hint=True,
                release_event_probability=0.70,
                position_confidence=0.20,
            )
            self.assertIsNone(recovered)
        self.assertTrue(state["suppress_contact"])

        state, recovered = guard.observe(
            baseline + 0.02,
            physical_frame=True,
            external_no_contact_hint=True,
            release_event_probability=0.70,
            position_confidence=0.20,
        )
        self.assertTrue(state["runtime_reference_reanchored"])
        self.assertIsNotNone(recovered)
        np.testing.assert_allclose(recovered, baseline + 0.01, atol=1.0e-9)

    def test_interpolated_frames_do_not_advance_recovery(self) -> None:
        guard = self._guard()
        spectrum = np.linspace(1000.0, 2000.0, 64)
        for _ in range(20):
            state, recovered = guard.observe(
                spectrum,
                physical_frame=False,
                external_no_contact_hint=True,
                release_event_probability=0.90,
                position_confidence=0.10,
            )
        self.assertEqual(state["stable_release_physical_frames"], 0)
        self.assertIsNone(recovered)

    def test_held_contact_without_release_evidence_is_never_reanchored(self) -> None:
        guard = self._guard()
        spectrum = np.linspace(1000.0, 2000.0, 64)
        for _ in range(8):
            state, recovered = guard.observe(
                spectrum,
                physical_frame=True,
                external_no_contact_hint=True,
                release_event_probability=0.10,
                position_confidence=0.20,
            )
        self.assertFalse(state["suppress_contact"])
        self.assertFalse(state["runtime_reference_reanchored"])
        self.assertIsNone(recovered)

    @staticmethod
    def _production_guard() -> RuntimeBaselineRecoveryGuard:
        return RuntimeBaselineRecoveryGuard(
            {
                "quiet_hold_sec": 5.0,
                "minimum_quiet_physical_frames": 8,
                "max_shape_motion_rms": 0.0035,
                "max_common_gain_motion": 0.0030,
                "activity_shape_motion_rms": 0.0060,
                "activity_common_gain_motion": 0.0060,
                "contact_probability_arm": 0.65,
                "contact_arm_physical_frames": 2,
                "stationary_rest_fallback_enabled": True,
            }
        )

    def test_stationarity_alone_does_not_clear_a_held_contact(self) -> None:
        guard = self._production_guard()
        baseline = np.linspace(1000.0, 2000.0, 64)
        held = baseline.copy()
        held[20:44] *= 0.82
        recovered = None
        for index in range(16):
            state, recovered = guard.observe(
                held,
                physical_frame=True,
                external_no_contact_hint=False,
                release_event_probability=0.05,
                position_confidence=0.88,
                contact_probability=0.96,
                contact_label="contact",
                baseline_spectrum=baseline,
                timestamp_sec=index * 0.4,
            )
        self.assertTrue(state["contact_armed"])
        self.assertFalse(state["release_transition_detected"])
        self.assertFalse(state["runtime_reference_reanchored"])
        self.assertIsNone(recovered)

    def test_stable_raw_no_contact_fallback_recovers_missed_release_edge(self) -> None:
        guard = self._production_guard()
        baseline = np.linspace(1000.0, 2000.0, 64)
        residual = baseline.copy()
        residual[20:44] *= 0.93
        recovered = None
        for index in range(15):
            state, recovered = guard.observe(
                residual + 0.01 * np.sin(np.linspace(0.0, np.pi, 64) + index),
                physical_frame=True,
                external_no_contact_hint=True,
                release_event_probability=0.05,
                position_confidence=0.20,
                contact_probability=0.95,
                contact_label="contact",
                baseline_spectrum=baseline,
                timestamp_sec=index * 0.4,
            )
            if recovered is not None:
                break
        self.assertTrue(state["runtime_reference_reanchored"])
        self.assertIn("stationary_rest_fallback", state["release_evidence"])
        self.assertEqual(
            state["recovery_candidate_kind"],
            "stationary_rest_recovery",
        )
        self.assertFalse(state["release_transition_detected"])
        self.assertGreaterEqual(state["quiet_elapsed_sec"], 5.0)
        self.assertIsNotNone(recovered)

        state, second = guard.observe(
            residual,
            physical_frame=True,
            external_no_contact_hint=True,
            release_event_probability=0.05,
            position_confidence=0.20,
            contact_probability=0.95,
            contact_label="contact",
            baseline_spectrum=recovered,
            timestamp_sec=6.0,
        )
        self.assertTrue(state["fallback_locked_until_new_contact"])
        self.assertTrue(state["runtime_rest_latched"])
        self.assertTrue(state["suppress_contact"])
        self.assertFalse(state["stable_release_candidate"])
        self.assertIsNone(second)

        pressed = residual.copy()
        pressed[18:46] *= 0.80
        for index in range(2):
            state, second = guard.observe(
                pressed,
                physical_frame=True,
                external_no_contact_hint=False,
                release_event_probability=0.02,
                position_confidence=0.85,
                contact_probability=0.96,
                contact_label="contact",
                baseline_spectrum=recovered,
                timestamp_sec=6.4 + index * 0.4,
            )
        self.assertFalse(state["fallback_locked_until_new_contact"])
        self.assertFalse(state["runtime_rest_latched"])
        self.assertFalse(state["suppress_contact"])
        self.assertTrue(state["contact_armed"])


@unittest.skipUnless(V1_MODEL.exists(), "dynamic v1 candidate artifact is unavailable")
class DynamicTemporalShadowAdapterSmokeTests(unittest.TestCase):
    @staticmethod
    def _synthetic_spectrum() -> tuple[np.ndarray, np.ndarray]:
        wavelength = np.linspace(1526.5, 1561.5, 512, dtype=float)
        spectrum = np.full_like(wavelength, 1500.0)
        for center in (1527.813917, 1532.074029, 1536.272630, 1540.087209,
                       1544.339792, 1547.790240, 1551.672060, 1555.766698,
                       1559.838208):
            spectrum += 8000.0 * np.exp(-0.5 * ((wavelength - center) / 0.18) ** 2)
        return wavelength, spectrum

    def test_real_v1_bundle_keeps_shadow_safety_marker(self) -> None:
        bundle = load_dynamic_shadow_bundle(V1_MODEL)
        self.assertEqual(bundle["status"], "shadow_only_not_primary")
        self.assertFalse(bundle["deployment_ready"])

    def test_baseline_and_warmup_contract(self) -> None:
        adapter = DynamicTemporalShadowAdapter.from_paths(V1_MODEL, PEAK_CONFIG)
        wavelength, baseline = self._synthetic_spectrum()
        before = adapter.update(wavelength, baseline)
        self.assertEqual(before["status"], "baseline_required")
        self.assertFalse(before["ready"])

        adapter.set_baseline(wavelength, baseline)
        for index in range(adapter.time_steps - 1):
            result = adapter.update(wavelength, baseline)
            self.assertEqual(result["status"], "window_warming_up")
            self.assertEqual(result["history_frames"], index + 1)
        result = adapter.update(wavelength, baseline)
        self.assertEqual(result["status"], "shadow_ready")
        self.assertTrue(result["ready"])
        self.assertEqual(result["mode"], "shadow_only_not_primary")
        self.assertIn(result["contact"]["label"], {"no_contact", "contact"})

    def test_stride_hold_keeps_every_frame_in_the_temporal_window(self) -> None:
        adapter = DynamicTemporalShadowAdapter.from_paths(V1_MODEL, PEAK_CONFIG)
        wavelength, baseline = self._synthetic_spectrum()
        adapter.set_baseline(wavelength, baseline)
        for _ in range(adapter.time_steps - 1):
            adapter.update(wavelength, baseline, run_inference=False)
        waiting = adapter.update(wavelength, baseline, run_inference=False)
        self.assertEqual(waiting["status"], "inference_stride_hold")
        self.assertEqual(waiting["history_frames"], adapter.time_steps)

        inferred = adapter.update(wavelength, baseline, run_inference=True)
        self.assertEqual(inferred["status"], "shadow_ready")
        held = adapter.update(wavelength, baseline, run_inference=False)
        self.assertEqual(held["status"], "shadow_stride_hold")
        self.assertTrue(held["held_from_previous_unique_frame"])
        self.assertEqual(held["history_frames"], adapter.time_steps)

    def test_baseline_preroll_makes_first_new_frame_inferable(self) -> None:
        adapter = DynamicTemporalShadowAdapter.from_paths(
            V1_MODEL,
            PEAK_CONFIG,
            runtime_recovery_config={
                "prime_temporal_history_with_baseline": True,
                "baseline_preroll_frames": 20,
            },
        )
        wavelength, baseline = self._synthetic_spectrum()
        adapter.set_baseline(wavelength, baseline)

        result = adapter.update(wavelength, baseline)

        self.assertEqual(result["status"], "shadow_ready")
        self.assertTrue(result["ready"])
        self.assertEqual(result["history_frames"], adapter.time_steps)
        self.assertTrue(result["baseline_preroll_enabled"])
        self.assertEqual(result["baseline_preroll_frames"], adapter.time_steps)


@unittest.skipUnless(V2_MODEL.exists(), "dynamic v2 candidate artifact is unavailable")
class DynamicTemporalShadowV2SafetyTests(unittest.TestCase):
    def test_v2_requires_zero_grouped_early_release_triggers(self) -> None:
        bundle = load_dynamic_shadow_bundle(V2_MODEL)
        self.assertEqual(bundle["schema_version"], "dynamic_temporal_shadow_candidate_v2")
        self.assertEqual(
            bundle["release_guard_grouped_cv"][
                "unsafe_early_trigger_sequence_count"
            ],
            0,
        )

        unsafe = dict(bundle)
        unsafe["release_guard_grouped_cv"] = dict(
            bundle["release_guard_grouped_cv"]
        )
        unsafe["release_guard_grouped_cv"][
            "unsafe_early_trigger_sequence_count"
        ] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.joblib"
            joblib.dump(unsafe, path)
            with self.assertRaisesRegex(ValueError, "unsafe pre-release triggers"):
                load_dynamic_shadow_bundle(path)


@unittest.skipUnless(V3_MODEL.exists(), "dynamic v3 candidate artifact is unavailable")
class DynamicTemporalShadowV3FactorizedPositionTests(unittest.TestCase):
    def test_v3_loads_with_factorized_position_contract(self) -> None:
        bundle = load_dynamic_shadow_bundle(V3_MODEL)

        self.assertEqual(bundle["schema_version"], "dynamic_temporal_shadow_candidate_v3")
        self.assertEqual(
            bundle["position_inference_mode"],
            "factorized_row_column_probability_product",
        )
        self.assertIn("position_factorized", bundle["models"])
        self.assertFalse(bundle["deployment_ready"])

    def test_v3_adapter_reports_factorized_position_when_contact(self) -> None:
        adapter = DynamicTemporalShadowAdapter.from_paths(V3_MODEL, PEAK_CONFIG)
        wavelength, baseline = DynamicTemporalShadowAdapterSmokeTests._synthetic_spectrum()
        adapter.set_baseline(wavelength, baseline)
        result = None
        for _ in range(adapter.time_steps):
            result = adapter.update(wavelength, baseline)
        self.assertIsNotNone(result)
        if result["contact"]["label"] == "contact":
            self.assertEqual(
                result["position"]["inference_mode"],
                "factorized_row_column_probability_product",
            )


if __name__ == "__main__":
    unittest.main()
