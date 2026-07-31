from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class OperatorIconSimplificationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        self.app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    def test_operator_commands_use_local_lucide_icons_with_accessible_names(self) -> None:
        self.assertIn('/static/vendor/lucide/lucide.min.js', self.html)
        for button_id in (
            "liveTwinButton",
            "exportWatchButton",
            "baselineButton",
            "pauseButton",
            "demoMenuButton",
            "operatorDiagnosticsButton",
            "settingsButton",
        ):
            self.assertIn(f'id="{button_id}"', self.html)
        self.assertGreaterEqual(self.html.count('class="ui-icon-button"'), 5)
        self.assertGreaterEqual(self.html.count('data-tooltip='), 8)
        self.assertGreaterEqual(self.html.count('aria-label='), 8)

    def test_fullscreen_is_a_four_corner_icon_not_visible_button_copy(self) -> None:
        fullscreen_start = self.html.index('id="surfaceFullscreenButton"')
        fullscreen_end = self.html.index('</button>', fullscreen_start)
        fullscreen_markup = self.html[fullscreen_start:fullscreen_end]

        self.assertIn('data-lucide="maximize"', fullscreen_markup)
        self.assertIn('data-lucide="minimize"', fullscreen_markup)
        self.assertNotIn('>Fullscreen<', fullscreen_markup)
        self.assertIn('aria-label="Enter tactile surface fullscreen"', fullscreen_markup)

    def test_dynamic_command_updates_preserve_icon_nodes(self) -> None:
        self.assertIn("function setCommandButtonLabel(button, label)", self.app_js)
        self.assertIn('button.querySelector(".command-label")', self.app_js)
        self.assertIn("setCommandButtonDescription(button, busyLabel)", self.app_js)
        self.assertNotIn('liveTwinButton.textContent = liveActive', self.app_js)
        self.assertNotIn('pauseButton.textContent = state.paused', self.app_js)

    def test_operator_hides_redundant_stage_chrome_and_keeps_tooltips(self) -> None:
        self.assertIn(".lucide-ready .operator-mode .ui-icon-button .command-label", self.css)
        self.assertIn(".operator-mode .ui-icon-button[data-tooltip]::after", self.css)
        self.assertIn(".operator-mode .stage-actions .level-badge", self.css)
        self.assertIn(".operator-mode .three-mount .pressure-scale", self.css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr)) !important", self.css)

    def test_operator_uses_apple_industrial_structure_without_recoloring(self) -> None:
        marker = "Apple-inspired industrial structure"
        self.assertIn(marker, self.css)
        structural_pass = self.css[self.css.index(marker):]
        self.assertIn('font-family: -apple-system, BlinkMacSystemFont', structural_pass)
        self.assertIn("backdrop-filter: saturate(160%) blur(16px)", structural_pass)
        self.assertNotIn("--bg:", structural_pass)
        self.assertNotIn("background: #f5f5f7", structural_pass)
        self.assertNotIn("background: #0071e3", structural_pass)
        self.assertIn("Surface Summary", self.html)
        self.assertIn("Contact Map", self.html)
        self.assertIn("Optical Response", self.html)

    def test_operator_contact_map_discloses_shared_proxy_scope(self) -> None:
        self.assertIn(
            '<div id="footprintTitle" class="section-label">Shared 3x3 Contact Map</div>',
            self.html,
        )
        self.assertGreaterEqual(
            self.app_js.count('setText("footprintTitle", "Shared 3x3 Contact Map")'),
            1,
        )
        self.assertIn(
            "One response is mirrored across the five fingertip views",
            self.html,
        )
        self.assertNotIn('setText("footprintTitle", `${scope} 9-FBG Fingerprint`)', self.app_js)


if __name__ == "__main__":
    unittest.main()
