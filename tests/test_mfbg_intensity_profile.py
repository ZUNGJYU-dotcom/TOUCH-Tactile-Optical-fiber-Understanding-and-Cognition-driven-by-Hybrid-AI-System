from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "bayspec_wavelength_shift_app"
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.mfbg_intensity import (  # noqa: E402
    MfbgIntensityDemodulator,
    frame_to_channel_rows,
    frame_to_wide_row,
    load_profile,
)
from src.mfbg_intensity.config import CHANNEL_ORDER, DISPLAY_ROWS  # noqa: E402


class MfbgIntensityProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profile()
        cls.wavelength_nm = np.linspace(1538.0, 1582.0, 4401)

    def spectrum(self, scales: dict[str, float] | None = None) -> np.ndarray:
        scales = scales or {}
        intensity = np.full_like(self.wavelength_nm, 500.0)
        for channel_id, channel in self.profile.channels.items():
            scale = float(scales.get(channel_id, 1.0))
            intensity += (
                12000.0
                * scale
                * np.exp(
                    -0.5
                    * (
                        (self.wavelength_nm - channel.demodulation_wavelength_nm)
                        / 0.12
                    )
                    ** 2
                )
            )
        return intensity

    def engine_with_baseline(self) -> MfbgIntensityDemodulator:
        engine = MfbgIntensityDemodulator(self.profile)
        result = engine.set_baseline(
            self.wavelength_nm,
            [self.spectrum() for _ in range(self.profile.baseline_minimum_frames)],
        )
        self.assertTrue(result["ok"])
        return engine

    def test_profile_has_physical_orientation_and_explicit_safety_boundaries(self) -> None:
        self.assertEqual(self.profile.channel_order, CHANNEL_ORDER)
        self.assertEqual(self.profile.display_rows, DISPLAY_ROWS)
        self.assertEqual(len(self.profile.channels), 9)
        self.assertFalse(self.profile.real_3x3_enabled)
        summary = self.profile.summary()
        self.assertFalse(summary["calibrated_force"])
        self.assertFalse(summary["force_N_output"])
        self.assertEqual(
            summary["surface_semantics"],
            "raw_coupled_optical_attenuation_proxy",
        )

    def test_same_fiber_downstream_paths_are_configured(self) -> None:
        paths = self.profile.raw["coupling"]["same_fiber_directed_paths"]
        self.assertEqual(paths["fiber_1"], ["P13", "P12", "P11"])
        self.assertEqual(paths["fiber_2"], ["P23", "P22", "P21"])
        self.assertEqual(paths["fiber_3"], ["P33", "P32", "P31"])

    def test_baseline_is_grouped_and_requires_enough_frames(self) -> None:
        engine = MfbgIntensityDemodulator(self.profile)
        too_short = engine.set_baseline(
            self.wavelength_nm,
            [self.spectrum() for _ in range(self.profile.baseline_minimum_frames - 1)],
        )
        self.assertFalse(too_short["ok"])
        self.assertEqual(too_short["reason"], "insufficient_baseline_frames")
        self.assertFalse(engine.baseline_ready)

    def test_single_channel_attenuation_produces_nine_vector_and_surface(self) -> None:
        engine = self.engine_with_baseline()
        frame = engine.analyze_spectrum(
            self.wavelength_nm,
            self.spectrum({"P22": 0.40}),
        )
        self.assertEqual(frame["dominant_channel"], "P22")
        self.assertEqual(frame["responding_channel_ids"], ["P22"])
        self.assertEqual(len(frame["attenuation_vector"]), 9)
        self.assertGreater(frame["channel_map"]["P22"]["attenuation_ratio"], 0.50)
        self.assertEqual(frame["contact_region_count"], 1)
        self.assertTrue(frame["surface_grid"])
        self.assertFalse(frame["real_3x3_enabled"])
        self.assertEqual(frame["configured_channel_count"], 9)
        self.assertEqual(frame["analyzed_channel_count"], 9)
        self.assertEqual(frame["real_enabled_channel_count"], 0)
        self.assertEqual(frame["real_enabled_channel_ids"], [])
        self.assertEqual(
            frame["runtime_activation_status"],
            "disabled_pending_measured_wavelengths_and_baseline",
        )
        self.assertFalse(frame["force_N_output"])

    def test_separated_responses_remain_multiple_regions(self) -> None:
        engine = self.engine_with_baseline()
        frame = engine.analyze_spectrum(
            self.wavelength_nm,
            self.spectrum({"P11": 0.35, "P33": 0.45}),
        )
        self.assertEqual(set(frame["responding_channel_ids"]), {"P11", "P33"})
        self.assertEqual(frame["contact_region_count"], 2)
        self.assertEqual(
            {tuple(region["channel_ids"]) for region in frame["contact_regions"]},
            {("P11",), ("P33",)},
        )

    def test_intensity_rise_is_warning_not_false_contact(self) -> None:
        engine = self.engine_with_baseline()
        frame = engine.analyze_spectrum(
            self.wavelength_nm,
            self.spectrum({"P22": 1.35}),
        )
        p22 = frame["channel_map"]["P22"]
        self.assertIn("intensity_rise_anomaly", p22["qa_flags"])
        self.assertFalse(p22["responding"])
        self.assertEqual(p22["response_state"], "no_contact")

    def test_measured_wavelength_has_priority_without_changing_target(self) -> None:
        original = self.profile.channels["P22"]
        measured = replace(original, measured_wavelength_nm=1560.22)
        self.assertEqual(measured.target_wavelength_nm, 1560.0)
        self.assertEqual(measured.demodulation_wavelength_nm, 1560.22)

    def test_recording_adapters_preserve_channel_metrics(self) -> None:
        engine = self.engine_with_baseline()
        frame = engine.analyze_spectrum(
            self.wavelength_nm,
            self.spectrum({"P12": 0.55}),
        )
        rows = frame_to_channel_rows(frame)
        wide = frame_to_wide_row(frame)
        self.assertEqual(len(rows), 9)
        self.assertEqual({row["channel_id"] for row in rows}, set(CHANNEL_ORDER))
        self.assertIn("P12_attenuation_ratio", wide)
        self.assertIn("contact_regions_json", wide)

    def test_api_routes_are_registered_without_replacing_existing_frame_api(self) -> None:
        from backend import main as backend_main

        paths = {
            route.path
            for route in backend_main.app.routes
            if getattr(route, "path", None)
        }
        mfbg_paths = {
            route.path
            for route in backend_main.mfbg_intensity_router.routes
            if getattr(route, "path", None)
        }
        self.assertIn("/api/frame", paths)
        self.assertIn("/api/mfbg-intensity/profile", mfbg_paths)
        self.assertIn("/api/mfbg-intensity/analyze-spectrum", mfbg_paths)
        self.assertIn(
            "/api/mfbg-intensity/baseline-from-recent-bayspec-frames",
            mfbg_paths,
        )
        health = backend_main.health()
        self.assertEqual(
            health["active_runtime_sensor_profile"],
            "ordinary_fbg_hybrid_spectral",
        )
        self.assertEqual(
            health["future_primary_sensor_profile"],
            "mfbg_intensity_3x3",
        )
        self.assertTrue(health["sensor_profile_isolation"])
        frontend = (
            PROJECT_ROOT
            / "bayspec_wavelength_shift_app"
            / "frontend"
            / "index.html"
        ).read_text(encoding="utf-8")
        frontend_js = (
            PROJECT_ROOT
            / "bayspec_wavelength_shift_app"
            / "frontend"
            / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn('id="mfbgFuturePrimary"', frontend)
        self.assertIn('id="mfbgRealArrayStatus"', frontend)
        profile_status = frontend.split(
            'id="sensorProfileStatusItem"', 1
        )[1].split("</span>", 1)[0]
        self.assertIn("<small>Profile</small>", profile_status)
        self.assertIn('id="surfaceModeChip">Ordinary FBG</strong>', profile_status)
        self.assertNotIn('data-lucide="activity"', profile_status)
        self.assertNotIn("Temporal validation", profile_status)
        self.assertIn('setText("surfaceModeChip", currentProfileLabel)', frontend_js)
        self.assertIn(
            "mFBG intensity profile is integrated and awaiting real calibration.",
            frontend_js,
        )
        operator_frame_update = frontend_js.split(
            "function updateUI(inputFrame)", 1
        )[1].split("\nfunction ", 1)[0]
        self.assertNotIn('"surfaceModeChip"', operator_frame_update)

    def test_api_is_fail_closed_until_real_nine_channel_runtime_is_ready(self) -> None:
        from backend import mfbg_intensity_api

        activation = mfbg_intensity_api._runtime_activation()
        self.assertFalse(activation["real_runtime_ready"])
        self.assertFalse(activation["real_3x3_enabled"])
        self.assertEqual(activation["real_enabled_channel_count"], 0)
        self.assertEqual(
            activation["reason"],
            "mfbg_real_3x3_disabled",
        )

        blocked = mfbg_intensity_api._analysis_gate(diagnostic_preview=False)
        self.assertIsNotNone(blocked)
        self.assertFalse(blocked["ok"])
        self.assertFalse(blocked["operator_eligible"])
        self.assertFalse(blocked["recording_eligible"])
        self.assertIsNone(
            mfbg_intensity_api._analysis_gate(diagnostic_preview=True)
        )

    def test_health_exposes_traceable_release_identity(self) -> None:
        from backend import main as backend_main

        payload = backend_main.health()
        self.assertTrue(payload["version"])
        self.assertTrue(payload["build_id"])
        self.assertIn("release", payload)
        self.assertIn("version_manifest", payload["release"])
        self.assertEqual(
            payload["source_commit"],
            payload["release"]["source_commit"],
        )

    def test_beta_health_exposes_one_non_switchable_current_model(self) -> None:
        from backend import main as backend_main

        with patch.object(backend_main, "ALL_SOURCE_BETA_ENABLED", True):
            payload = backend_main.health()

        runtime = payload["recognition_runtime"]
        self.assertEqual(
            runtime["active_model_id"],
            "ordinary_fbg_all_data_beta_v1",
        )
        self.assertEqual(runtime["display_name"], "All-data spectral model")
        self.assertFalse(runtime["switchable"])
        self.assertFalse(runtime["legacy_models_enabled"])
        self.assertFalse(payload["old_model_fallback_enabled"])


if __name__ == "__main__":
    unittest.main()
