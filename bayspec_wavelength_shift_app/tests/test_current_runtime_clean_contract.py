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
            "single_current_runtime",
        )


if __name__ == "__main__":
    unittest.main()
