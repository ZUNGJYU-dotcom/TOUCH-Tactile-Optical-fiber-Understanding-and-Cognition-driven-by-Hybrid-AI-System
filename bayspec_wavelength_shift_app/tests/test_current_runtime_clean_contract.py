from __future__ import annotations

import json
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent


class CurrentRuntimeCleanContractTests(unittest.TestCase):
    def test_backend_and_launcher_expose_one_current_model(self) -> None:
        backend = (APP_ROOT / "backend" / "main.py").read_text(encoding="utf-8")
        launcher = (APP_ROOT / "desktop_launcher.py").read_text(encoding="utf-8")

        self.assertIn("ordinary_fbg_current_runtime.joblib", backend)
        self.assertIn('"runtime_role": "deployed_current_model_only"', backend)
        self.assertIn('"model_count": 1', backend)
        self.assertIn('recognition_runtime.get("model_count") == 1', launcher)
        for stale_name in (
            "_predict_all_source_beta",
            "_predict_static_spectral_frame",
            "_predict_dynamic_temporal_shadow",
            "old_model_fallback_enabled",
            "legacy_visual_fallback_allowed",
            "legacy_models_enabled",
        ):
            self.assertNotIn(stale_name, backend)
            self.assertNotIn(stale_name, launcher)

    def test_frontend_consumes_only_the_canonical_visualization_contract(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn("function normalizeCanonicalVisualizationFrame", app_js)
        self.assertIn("appendCurrentRuntimeTrace(rawRecord, contract)", app_js)
        self.assertIn("operatorVisualizationContract(record, arrayFrame, runtimeFrame)", app_js)
        for stale_name in (
            "all_source_beta_model",
            "trained_static_spectral_prediction",
            "dynamic_temporal_shadow",
            "normalizeCanonicalBetaVisualizationFrame",
            "trainedStaticModelSurface",
            "betaForceDisplayEnabled",
        ):
            self.assertNotIn(stale_name, app_js)

    def test_frontend_runtime_status_text_has_no_known_encoding_damage(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn('"raw -Fz = compression"', app_js)
        self.assertIn('`#${sequenceStart}-${sequenceEnd}`', app_js)
        self.assertIn('baseline_required: "set reference baseline"', app_js)
        for damaged_fragment in ("鈭扚", "鈥?", "位0"):
            self.assertNotIn(damaged_fragment, app_js)

    def test_frontend_render_key_tracks_backend_unique_physical_frames(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn("function sourceFrameRenderKey(rawFrame)", app_js)
        self.assertIn("active_spectral_prediction?.unique_frame_count", app_js)
        self.assertIn("runtime_prediction?.unique_frame_count", app_js)
        self.assertIn("runtimeUniqueFrame", app_js)
        self.assertIn("sdk_live?.acquisition_session_id", app_js)
        self.assertIn("export_watcher?.acquisition_session_id", app_js)
        self.assertIn("contract?.source_session_id", app_js)

    def test_frontend_render_key_repaints_when_source_gate_becomes_stale(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn("const gateSignature = [", app_js)
        self.assertIn('rawFrame?.source_fresh === true ? "fresh" : "stale"', app_js)
        self.assertIn(
            'rawFrame?.operator_display_valid === true ? "display" : "blocked"',
            app_js,
        )
        self.assertIn("model_assisted_display_block_reason", app_js)
        self.assertIn("visualization?.response_allowed", app_js)
        self.assertNotIn(
            "opticalActive && Boolean(state.lastRenderedSourceFrameKey)",
            app_js,
        )
        self.assertIn("opticalActive && frameSourceIsFresh(state.frame)", app_js)

    def test_frozen_build_contains_only_the_deployed_model_and_explicit_modules(self) -> None:
        spec = (APP_ROOT / "desktop_launcher_beta_all_data.spec").read_text(
            encoding="utf-8"
        )

        self.assertIn("ordinary_fbg_current_runtime.joblib", spec)
        self.assertNotIn("models/candidates", spec.replace("\\", "/"))
        self.assertNotIn('collect_submodules("src.hybrid_spectrum")', spec)
        self.assertNotIn("Tree(", spec)
        self.assertNotIn("str(REPO_ROOT / \"src\")", spec)
        self.assertNotIn("str(REPO_ROOT / \"config\")", spec)

    def test_release_manifest_identifies_the_single_model_beta(self) -> None:
        manifest_path = APP_ROOT / "release_manifests" / "beta" / "VERSION.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertRegex(manifest["version"], r"^\d+\.\d+\.\d+-beta$")
        self.assertEqual(manifest["runtime_model_count"], 1)
        self.assertEqual(
            manifest["deployed_model_source"],
            "ordinary_fbg_current_runtime",
        )
        self.assertEqual(manifest["runtime_model_packaging"], "deployed_only")
        self.assertEqual(
            manifest["operator_visualization_contract"],
            "touch_operator_visualization_v2",
        )
        self.assertEqual(
            manifest["contact_map_runtime"],
            "current_frame_gated_position_posterior_ema_scaled_by_optical_force",
        )
        self.assertEqual(
            manifest["diagnostics_observed_map"],
            "current_frame_nine_fbg_coupled_spectral_response",
        )
        self.assertFalse(manifest["legacy_gaussian_contact_map"])

    def test_stable_manifest_is_an_isolated_promotion_of_validated_beta(self) -> None:
        beta = json.loads(
            (APP_ROOT / "release_manifests" / "beta" / "VERSION.json").read_text(
                encoding="utf-8"
            )
        )
        stable = json.loads(
            (APP_ROOT / "release_manifests" / "stable" / "VERSION.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertRegex(stable["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(stable["release_channel"], "stable")
        self.assertIn(beta["version"], stable["promotion_source"])
        self.assertEqual(stable["runtime_model_count"], 1)
        self.assertEqual(stable["legacy_runtime_model_count"], 0)
        self.assertEqual(stable["deployed_model_source"], beta["deployed_model_source"])
        self.assertEqual(stable["runtime_model_packaging"], "deployed_only")
        self.assertEqual(stable["runtime_release_policy"], beta["runtime_release_policy"])
        self.assertEqual(
            stable["runtime_frame_order_policy"],
            beta["runtime_frame_order_policy"],
        )
        self.assertEqual(
            stable["bayspec_acquisition_settings_scope"],
            "stable_isolated_user_settings",
        )
        self.assertEqual(stable["live_idle_source_false_activation_rate"], 0.0)
        self.assertFalse(stable["live_release_final_contact_latched"])


if __name__ == "__main__":
    unittest.main()
