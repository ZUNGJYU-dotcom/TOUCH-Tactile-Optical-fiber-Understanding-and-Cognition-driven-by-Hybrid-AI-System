from __future__ import annotations

import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


class OperatorResponsePresentationContractTests(unittest.TestCase):
    def test_operator_response_labels_are_pressure_levels_not_scene_names(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        presentation = app_js.split("function surfaceContactPresentation", 1)[1].split(
            "function clampArrayCoord", 1
        )[0]

        self.assertIn('primary: levelLabel(level)', presentation)
        self.assertIn('primary: scenarioState', presentation)
        self.assertNotIn('primary: `${position}', presentation)
        self.assertNotIn('primary: "Single-finger contact patch"', presentation)
        self.assertNotIn('primary: "Fingertip tap"', presentation)
        self.assertIn('small_shift: "light"', app_js)
        self.assertIn('moderate_shift: "normal"', app_js)
        self.assertIn('large_shift: "hard"', app_js)

    def test_scene_floor_grid_is_visible_but_stays_below_the_model(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn('new THREE.GridHelper(11.5, 36, "#73a9c5", "#bdd6e4")', app_js)
        self.assertIn('sceneGrid.material.opacity = 0.48', app_js)
        self.assertIn('sceneGrid.position.y = boundsMinY - clearance', app_js)


if __name__ == "__main__":
    unittest.main()
