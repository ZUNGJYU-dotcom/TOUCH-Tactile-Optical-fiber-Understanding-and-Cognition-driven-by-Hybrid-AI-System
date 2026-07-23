from __future__ import annotations

import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


class OperatorResponsePresentationContractTests(unittest.TestCase):
    def test_operator_response_state_is_contact_not_force_band(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        presentation = app_js.split("function surfaceContactPresentation", 1)[1].split(
            "function clampArrayCoord", 1
        )[0]

        self.assertIn('primary: "contact"', presentation)
        self.assertIn('primary: "no_contact"', presentation)
        self.assertNotIn("primary: levelLabel(level)", presentation)
        self.assertIn(
            'primary: scenarioState === "no_contact" ? "no_contact" : "contact"',
            presentation,
        )
        self.assertNotIn("primary: scenarioState,", presentation)
        self.assertNotIn('primary: `${position}', presentation)
        self.assertNotIn('primary: "Single-finger contact patch"', presentation)
        self.assertNotIn('primary: "Fingertip tap"', presentation)

    def test_scene_floor_grid_is_visible_but_stays_below_the_model(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn('new THREE.GridHelper(11.5, 36, "#73a9c5", "#bdd6e4")', app_js)
        self.assertIn("sceneGrid.material.opacity = 0.48", app_js)
        self.assertIn("sceneGrid.position.y = boundsMinY - clearance", app_js)

    def test_operator_response_gauge_is_continuous(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        labels_start = html.index('class="response-band-labels"')
        labels_end = html.index("</div>", labels_start)
        labels = html[labels_start:labels_end]

        for label in ("0%", "25%", "50%", "75%", "100%"):
            self.assertIn(f"<span>{label}</span>", labels)
        for legacy_label in (
            "none",
            "light",
            "normal",
            "hard",
            "small",
            "moderate",
            "large",
        ):
            self.assertNotIn(f"<span>{legacy_label}</span>", labels)
        self.assertIn("Continuous normalized optical response from 0% to 100%", app_js)
        self.assertIn("${formatPercent(peak, 0)} optical response", app_js)
        self.assertNotIn("levelLabel(responseBandLevel)", app_js)
        self.assertIn("responseBandValue.setAttribute(", app_js)
        self.assertIn('"aria-label"', app_js)

    def test_contact_footprint_separates_contact_from_coupled_neighbors(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        styles = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        footprint = app_js.split("function updateOperatorFootprint", 1)[1].split(
            "function syncPrimaryCommandLabels", 1
        )[0]

        self.assertIn("COUPLED_CHANNEL_VISIBILITY_THRESHOLD = 0.05", app_js)
        self.assertIn("value >= RESPONSE_BAND_THRESHOLDS.noContactMax", footprint)
        self.assertIn("const contactChannels = []", footprint)
        self.assertIn("const coupledNeighborChannels = []", footprint)
        self.assertIn('cell.classList.toggle("coupled-neighbor", coupledNeighbor)', footprint)
        self.assertIn("contact threshold", footprint)
        self.assertIn("coupledNeighborChannels.length", footprint)
        self.assertIn('"coupled neighbor below contact threshold"', footprint)
        self.assertIn('footprintNote.setAttribute("aria-label", noteDetail)', footprint)
        self.assertIn("contact", footprint)
        self.assertNotIn("responding pixels", footprint)
        self.assertIn(".operator-mode .mini-pixel.coupled-neighbor", styles)


if __name__ == "__main__":
    unittest.main()
