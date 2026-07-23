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

    def test_whole_hand_loader_repairs_missing_vertex_normals(self) -> None:
        self.assertIn("loadRobotNanoHandModel", self.loader_js)
        self.assertIn('child.geometry.getAttribute("normal")', self.loader_js)
        self.assertIn("child.geometry.computeVertexNormals()", self.loader_js)
        self.assertIn("THREE.DoubleSide", self.loader_js)


if __name__ == "__main__":
    unittest.main()
