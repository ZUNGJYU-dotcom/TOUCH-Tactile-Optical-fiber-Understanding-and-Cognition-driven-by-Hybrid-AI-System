from pathlib import Path
import json
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent


class AcceptanceRemediationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frontend = (APP_ROOT / "frontend" / "app.js").read_text(
            encoding="utf-8"
        )
        cls.html = (APP_ROOT / "frontend" / "index.html").read_text(
            encoding="utf-8"
        )
        cls.backend = (APP_ROOT / "backend" / "main.py").read_text(
            encoding="utf-8"
        )
        cls.bridge = (APP_ROOT / "bridge.py").read_text(encoding="utf-8")
        cls.force_capture = (
            APP_ROOT / "backend" / "optical_force_capture.py"
        ).read_text(encoding="utf-8")
        cls.mfbg_api = (
            APP_ROOT / "backend" / "mfbg_intensity_api.py"
        ).read_text(encoding="utf-8")
        cls.launcher_spec = (
            APP_ROOT / "desktop_launcher.spec"
        ).read_text(encoding="utf-8")
        cls.launcher = (
            APP_ROOT / "desktop_launcher.py"
        ).read_text(encoding="utf-8")
        cls.version = (
            PROJECT_ROOT / "VERSION.json"
        ).read_text(encoding="utf-8")
        cls.version_payload = json.loads(cls.version)

    def test_invalid_measurement_is_atomically_neutralized(self) -> None:
        self.assertIn("function atomicNeutralFrame(frame)", self.frontend)
        self.assertIn(
            'presentation_status: "invalid_input_neutral"',
            self.frontend,
        )
        self.assertIn("operator_frame_atomic_neutralized: true", self.frontend)
        self.assertIn("diagnostic_raw_array_frame: sourceArrayFrame", self.frontend)

    def test_measured_frames_snap_all_visible_outputs_to_one_frame(self) -> None:
        self.assertIn(
            "function snapDisplayedFrameToCurrentTargets()",
            self.frontend,
        )
        self.assertIn(
            "snapDisplayedFrameToCurrentTargets();",
            self.frontend,
        )
        self.assertIn("drawVisibleCharts();", self.frontend)

    def test_three_runtime_is_demand_driven_and_disposed(self) -> None:
        self.assertIn("function requestThreeAnimation()", self.frontend)
        self.assertEqual(
            self.frontend.count("requestAnimationFrame(animate)"),
            1,
        )
        self.assertIn("function disposeThreeRuntime()", self.frontend)
        self.assertIn(
            'window.addEventListener("beforeunload", disposeThreeRuntime',
            self.frontend,
        )
        self.assertIn("renderer?.forceContextLoss?.()", self.frontend)

    def test_async_geometry_loads_schedule_a_render_frame(self) -> None:
        thumb_setup = self.frontend.split(
            "async function setupThumbHolderModel()", 1
        )[1].split("async function setupWholeHandModel()", 1)[0]
        whole_hand_setup = self.frontend.split(
            "async function setupWholeHandModel()", 1
        )[1].split("function updateGeometryDisplayMode(", 1)[0]

        self.assertIn("applyThumbSceneLayout();", thumb_setup)
        self.assertIn("requestThreeAnimation();", thumb_setup)
        self.assertIn("applyThumbSceneLayout();", whole_hand_setup)
        self.assertIn("requestThreeAnimation();", whole_hand_setup)

    def test_simulation_is_local_only_and_defaults_to_single_playback(self) -> None:
        demo_start = self.frontend.index("async function injectDemoFrame")
        demo_end = self.frontend.index(
            "async function injectArrayDemoFrame",
            demo_start,
        )
        demo_source = self.frontend[demo_start:demo_end]
        self.assertNotIn("/api/ingest", demo_source)
        self.assertIn(
            '{ resetTrajectory = true, playbackMode = "single" }',
            self.frontend,
        )

    def test_five_finger_view_discloses_shared_response_semantics(self) -> None:
        self.assertIn("Shared 3x3 Contact Map", self.html)
        self.assertIn(
            "One response is mirrored across the five fingertip views",
            self.html,
        )
        self.assertIn("shared response proxy", self.frontend)

    def test_current_all_data_model_is_the_only_operator_runtime(self) -> None:
        self.assertIn(
            '"active_model_id": "ordinary_fbg_all_data_beta_v1"',
            self.backend,
        )
        self.assertIn(
            '"model_count": 1',
            self.backend,
        )
        self.assertIn(
            '"switchable": False',
            self.backend,
        )
        self.assertIn("OPERATOR_VISUALIZATION_CONTRACT_VERSION", self.backend)
        self.assertIn("operator_visualization_frame", self.backend)
        self.assertNotIn("dynamic_temporal_v3_validation", self.backend)
        self.assertNotIn("trained_static_spectral_shadow", self.backend)
        self.assertIn(
            '"recognition_scope": "optical_contact_position_and_continuous_fz"',
            self.bridge,
        )
        self.assertIn(
            '"carrier_channel_role": "full_spectrum_transport_for_current_runtime"',
            self.bridge,
        )
        self.assertNotIn("trained_static_full_spectrum_classifier", self.bridge)
        self.assertNotIn("manual light/normal/hard response level", self.bridge)

    def test_baseline_requires_operator_release_attestation(self) -> None:
        self.assertIn("no_contact_attested: bool = False", self.backend)
        self.assertIn(
            "operator_no_contact_attestation_required",
            self.backend,
        )
        self.assertIn('"baseline_unchanged": True', self.backend)

    def test_force_capture_rejects_frames_outside_calibration_sync(self) -> None:
        self.assertIn('"calibration_sync_ok"', self.force_capture)
        self.assertIn("calibration_sync_valid = (", self.force_capture)
        self.assertIn(
            'force.get("calibration_sync_ok") is True',
            self.force_capture,
        )
        self.assertIn(
            'force.get("sync_within_target") is True',
            self.force_capture,
        )
        self.assertIn("math.isfinite(sync_offset_ms)", self.force_capture)

    def test_capture_metadata_has_release_and_calibration_provenance(self) -> None:
        self.assertIn("touch_synchronized_capture_v4", self.force_capture)
        self.assertIn('"provenance": {', self.force_capture)
        self.assertIn("provenance_provider=_capture_provenance_snapshot", self.backend)
        self.assertIn('"configuration_artifacts"', self.backend)
        self.assertIn('"operator_attestation"', self.backend)
        self.assertIn('"channel_grid"', self.backend)

    def test_health_and_frozen_app_expose_one_release_identity(self) -> None:
        self.assertIn('"release": dict(RELEASE_IDENTITY)', self.backend)
        self.assertIn('"source_commit": RELEASE_IDENTITY.get("source_commit")', self.backend)
        self.assertIn('"operator_recognition_model": version_payload.get(', self.backend)
        self.assertIn('"temporal_candidate_role": version_payload.get(', self.backend)
        self.assertIn('"capture_schema": version_payload.get("capture_schema")', self.backend)
        self.assertIn('app_dir.parent / "VERSION.json"', self.launcher_spec)
        release_channel = str(self.version_payload.get("release_channel") or "")
        version = str(self.version_payload.get("version") or "")
        expected_version = (
            re.compile(r"^\d+\.\d+\.\d+$")
            if release_channel == "stable"
            else re.compile(r"^\d+\.\d+\.\d+-beta$")
        )
        self.assertRegex(version, expected_version)
        self.assertTrue(str(self.version_payload.get("build_id") or "").strip())
        self.assertIn(release_channel, {"stable", "beta_research"})

    def test_frozen_build_excludes_unused_research_frameworks(self) -> None:
        self.assertIn("deployment_excludes = [", self.launcher_spec)
        for package in ("torch", "tensorflow", "sympy", "matplotlib", "pandas"):
            self.assertIn(f'"{package}"', self.launcher_spec)
        self.assertIn("excludes=deployment_excludes", self.launcher_spec)

    def test_frozen_capture_default_is_outside_replaceable_bundle(self) -> None:
        self.assertIn('Path.home() / "Documents"', self.launcher)
        self.assertIn('documents_root / "TOUCH" / "captures"', self.launcher)

    def test_capture_is_journaled_and_durably_flushed(self) -> None:
        self.assertIn("def _flush_and_sync(handle: Any)", self.force_capture)
        self.assertIn('"capture_journal.json"', self.force_capture)
        self.assertIn('"touch_capture_journal_v1"', self.force_capture)
        self.assertIn("os.fsync", self.force_capture)

    def test_disabled_mfbg_profile_is_fail_closed(self) -> None:
        self.assertIn("def _analysis_gate(*, diagnostic_preview: bool)", self.mfbg_api)
        self.assertIn('"mfbg_real_3x3_disabled"', self.mfbg_api)
        self.assertIn('"operator_eligible": False', self.mfbg_api)
        self.assertIn('"recording_eligible": False', self.mfbg_api)
        self.assertIn(
            "fail_closed_unless_real_ready_or_explicit_diagnostic_preview",
            self.mfbg_api,
        )


if __name__ == "__main__":
    unittest.main()
