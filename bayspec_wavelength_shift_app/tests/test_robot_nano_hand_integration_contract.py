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
        cls.config = yaml.safe_load(
            (REPO_ROOT / "config" / "thumb_holder_scene.yaml").read_text(
                encoding="utf-8"
            )
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
        self.assertIn('"geometry_status": "four_fingertip_recesses_carved"', audit_text)

    def test_five_finger_sensor_array_is_explicit_and_synchronized(self) -> None:
        sensor_array = self.config["finger_sensor_array"]
        self.assertTrue(sensor_array["enabled"])
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

    def test_thumb_dynamic_surface_stays_above_the_slot_body(self) -> None:
        whole = self.config["whole_hand_scene"]
        slot = self.config["sensor_slot_transform"]

        # The thumb surface normal maps to local -X in whole-hand mode.
        self.assertLess(whole["sensor_local_lift"][0], 0)
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
