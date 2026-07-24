import json
from pathlib import Path
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "bayspec_wavelength_shift_app"


class RobotNanoHandIntegrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        cls.loader_js = (APP_ROOT / "frontend" / "model_loader.js").read_text(
            encoding="utf-8"
        )
        cls.html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        cls.css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        cls.config = yaml.safe_load(
            (REPO_ROOT / "config" / "thumb_holder_scene.yaml").read_text(
                encoding="utf-8"
            )
        )
        cls.hand_audit = json.loads(
            (
                APP_ROOT
                / "frontend"
                / "assets"
                / "models"
                / "robot_nano_hand_sensorized.audit.json"
            ).read_text(encoding="utf-8")
        )

    def test_official_model_asset_and_mit_provenance_are_vendored(self) -> None:
        model_dir = APP_ROOT / "frontend" / "assets" / "models"
        body = model_dir / "robot_nano_hand_body.glb"
        license_file = model_dir / "robot_nano_hand_LICENSE.txt"

        self.assertTrue(body.is_file())
        self.assertGreater(body.stat().st_size, 1_000_000)
        self.assertEqual(body.read_bytes()[:4], b"glTF")
        self.assertTrue(license_file.is_file())
        license_text = license_file.read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("TheRobotStudio/robot-nano-hand", license_text)

    def test_whole_hand_scene_keeps_current_modified_thumb(self) -> None:
        whole = self.config["whole_hand_scene"]
        self.assertTrue(whole["enabled"])
        self.assertEqual(
            whole["asset_url"],
            "/static/assets/models/robot_nano_hand_sensorized.glb",
        )
        self.assertEqual(whole["source_license"], "MIT")
        self.assertEqual(
            whole["source_repository_url"],
            "https://github.com/TheRobotStudio/robot-nano-hand",
        )
        self.assertEqual(len(whole["modified_thumb_root_matrix_row_major"]), 16)
        self.assertTrue(
            (
                APP_ROOT
                / "frontend"
                / "assets"
                / "models"
                / "thumb_holder.stl"
            ).is_file()
        )

    def test_sensorized_whole_hand_asset_and_four_recess_audit_exist(self) -> None:
        model_dir = APP_ROOT / "frontend" / "assets" / "models"
        sensorized = model_dir / "robot_nano_hand_sensorized.glb"
        audit = model_dir / "robot_nano_hand_sensorized.audit.json"

        self.assertTrue(sensorized.is_file())
        self.assertGreater(sensorized.stat().st_size, 1_000_000)
        self.assertEqual(sensorized.read_bytes()[:4], b"glTF")
        self.assertTrue(audit.is_file())
        audit_text = audit.read_text(encoding="utf-8")
        for finger_id in ("index", "middle", "ring", "little"):
            self.assertIn(f'"{finger_id}"', audit_text)
        self.assertIn(
            '"geometry_status": "solidworks_tilted_flush_max_area_recesses_integrated"',
            audit_text,
        )
        self.assertIn('"replacement_method": "closed_solidworks_cad_component"', audit_text)

    def test_five_finger_sensor_array_is_explicit_and_synchronized(self) -> None:
        sensor_array = self.config["finger_sensor_array"]
        self.assertTrue(sensor_array["enabled"])
        self.assertEqual(
            sensor_array["geometry_status"],
            "solidworks_tilted_flush_max_area_recesses_integrated",
        )
        self.assertEqual(sensor_array["demo_sync_mode"], "synchronized_with_thumb")
        self.assertEqual(sensor_array["data_status"], "synchronized_demo_only")
        self.assertEqual(
            set(sensor_array["fingers"]),
            {"thumb", "index", "middle", "ring", "little"},
        )
        for finger_id in ("index", "middle", "ring", "little"):
            finger = sensor_array["fingers"][finger_id]
            self.assertTrue(finger["enabled"])
            self.assertEqual(len(finger["center_model"]), 3)
            self.assertEqual(len(finger["longitudinal_axis_model"]), 3)
            self.assertEqual(len(finger["outward_normal_model"]), 3)
            self.assertGreater(finger["slot_length_mm"], finger["slot_width_mm"])
            self.assertGreaterEqual(finger["slot_length_mm"], 12.4)
            self.assertGreaterEqual(finger["slot_width_mm"], 11.0)
            self.assertGreater(finger["slot_depth_mm"], 0)
            self.assertIn("cad_slot_source", finger)

    def test_four_finger_solidworks_slot_assets_are_vendored(self) -> None:
        slot_dir = (
            APP_ROOT
            / "frontend"
            / "assets"
            / "models"
            / "four_finger_cad_slots"
        )
        self.assertTrue((slot_dir / "manifest.json").is_file())
        for finger_id in ("index", "middle", "ring", "little"):
            slot_path = slot_dir / f"{finger_id}_local_flat_slot.stl"
            self.assertTrue(slot_path.is_file())
            self.assertGreater(slot_path.stat().st_size, 100_000)

    def test_runtime_sensor_layout_matches_integrated_cad_audit(self) -> None:
        fingers = self.config["finger_sensor_array"]["fingers"]
        for finger_id in ("index", "middle", "ring", "little"):
            runtime = fingers[finger_id]
            cad = self.hand_audit["slots"][finger_id]
            for runtime_key, audit_key in (
                ("center_model", "slot_center_model"),
                ("longitudinal_axis_model", "longitudinal_axis_model"),
                ("outward_normal_model", "outward_normal_model"),
            ):
                for actual, expected in zip(runtime[runtime_key], cad[audit_key]):
                    self.assertAlmostEqual(actual, expected, places=5)
            self.assertAlmostEqual(
                runtime["slot_length_mm"],
                cad["slot_length_model_mm"],
                places=5,
            )
            self.assertAlmostEqual(
                runtime["slot_width_mm"],
                cad["slot_width_model_mm"],
                places=5,
            )
            self.assertAlmostEqual(
                runtime["slot_depth_mm"],
                cad["slot_depth_model_mm"],
                places=5,
            )
            self.assertAlmostEqual(
                runtime["sensor_thickness_scale"] * 0.48,
                runtime["slot_depth_mm"],
                places=5,
            )

    def test_thumb_dynamic_surface_keeps_verified_slot_pose_and_inward_depth(self) -> None:
        whole = self.config["whole_hand_scene"]
        slot = self.config["sensor_slot_transform"]

        # The verified thumb slot is offset toward local +X in whole-hand mode.
        self.assertGreater(whole["sensor_local_lift"][0], 0)
        # The elastomer thickness must extend into the recess, not over the heat surface.
        self.assertLess(slot["surface_scene_scale"][1], 0)

    def test_finger_selector_scopes_spectrum_and_array_views(self) -> None:
        self.assertIn('id="fingerFocusSelect"', self.html)
        for finger_id in ("thumb", "index", "middle", "ring", "little", "all"):
            self.assertIn(f'value="{finger_id}"', self.html)
        self.assertIn("setupFingerSensorGroups()", self.app_js)
        self.assertIn("applyFingerSensorLayout()", self.app_js)
        self.assertIn("fingerFocusSelect?.addEventListener", self.app_js)
        self.assertIn('setText("spectrumOverviewTitle"', self.app_js)
        self.assertIn('setText("footprintTitle"', self.app_js)

    def test_fingertip_closeup_navigation_and_camera_transition_contract(self) -> None:
        for element_id in ("previousFingerButton", "nextFingerButton"):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("finger-closeup-nav", self.css)
        self.assertIn("backdrop-filter: blur(14px)", self.css)
        self.assertIn("function fingerCloseupPose", self.app_js)
        self.assertIn("FINGER_CLOSEUP_DISTANCE_SCALE", self.app_js)
        self.assertIn("sensorBackNormal", self.app_js)
        self.assertIn('"sensor-back"', self.app_js)
        self.assertIn("closeupViewSide", self.app_js)
        self.assertIn("function beginCameraTransition", self.app_js)
        self.assertIn("function updateCameraTransition", self.app_js)
        self.assertIn("function settleOrbitControlsDamping", self.app_js)
        self.assertIn("const cameraTransitionUpdated = updateCameraTransition(timestamp)", self.app_js)
        self.assertIn("if (!cameraTransitionUpdated) controls?.update()", self.app_js)
        self.assertIn("function fingerIdFromPointerEvent", self.app_js)
        self.assertIn("function fingerIdFromProjectedRegion", self.app_js)
        self.assertIn("function projectedFingerHitRegions", self.app_js)
        self.assertIn("fingerHitRegions", self.app_js)
        self.assertIn("function ensureFingerInteractionProxy", self.app_js)
        self.assertIn("isFingerInteractionProxy", self.app_js)
        self.assertIn("function visibleObjectBounds", self.app_js)
        self.assertIn("FINGER_CLICK_MAX_MOVEMENT_PX", self.app_js)
        self.assertIn(
            'setSelectedFinger(fingerFocusSelect.value, { focusCamera: true })',
            self.app_js,
        )
        self.assertIn("cycleFingerCloseup(-1)", self.app_js)
        self.assertIn("cycleFingerCloseup(1)", self.app_js)
        self.assertIn('const FINGER_NAVIGATION_ORDER = [...FINGER_ORDER, "all"]', self.app_js)
        self.assertIn("FINGER_OVERVIEW_DURATION_MS", self.app_js)
        self.assertIn("navigationVisible: keepFingerNavigation && wholeHandMode", self.app_js)
        self.assertIn("controls.enabled = false", self.app_js)
        self.assertIn("controls.enabled = true", self.app_js)

    def test_three_geometry_modes_remain_available(self) -> None:
        for element_id in (
            "settingsWholeHandButton",
            "settingsThumbHolderButton",
            "settingsSurfaceOnlyButton",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn('"whole_hand"', self.app_js)
        self.assertIn('"thumb_holder"', self.app_js)
        self.assertIn('"surface_only"', self.app_js)
        self.assertEqual(
            self.config["thumb_holder_scene"]["default_geometry_mode"],
            "whole_hand",
        )

    def test_whole_hand_loader_repairs_missing_vertex_normals(self) -> None:
        self.assertIn("loadRobotNanoHandModel", self.loader_js)
        self.assertIn('child.geometry.getAttribute("normal")', self.loader_js)
        self.assertIn("child.geometry.computeVertexNormals()", self.loader_js)
        self.assertIn("THREE.DoubleSide", self.loader_js)


if __name__ == "__main__":
    unittest.main()
