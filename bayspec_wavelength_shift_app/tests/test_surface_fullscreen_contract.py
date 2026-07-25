from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class SurfaceFullscreenContractTests(unittest.TestCase):
    def test_fullscreen_control_and_escape_contract_exist(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="surfaceFullscreenButton"', html)
        self.assertIn("requestFullscreen", app_js)
        self.assertIn("document.exitFullscreen", app_js)
        self.assertIn('event.key !== "Escape"', app_js)
        self.assertIn('setSurfaceFullscreen(false)', app_js)

    def test_fullscreen_css_hides_everything_except_model_canvas(self) -> None:
        css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

        self.assertIn(".app-shell.surface-fullscreen-active > :not(.dashboard)", css)
        self.assertIn(".dashboard > :not(.twin-stage):not(.right-panel)", css)
        self.assertIn(".twin-stage > :not(.surface-cockpit-grid)", css)
        self.assertIn(".surface-cockpit-grid > :not(.three-mount)", css)
        self.assertIn(".three-mount .scene-caption", css)
        self.assertIn(".right-panel > :not(.operator-summary-card)", css)
        self.assertIn(".fullscreen-hidden-summary-field", css)
        self.assertIn('class="fullscreen-hidden-summary-field"', html)
        self.assertIn('class="fullscreen-brand-mark"', html)
        self.assertIn('src="/static/touch_system_icon.png"', html)
        self.assertIn(".app-shell.surface-fullscreen-active .fullscreen-brand-mark", css)
        self.assertIn("pointer-events: none", css)
        self.assertIn("width: 100vw !important", css)
        self.assertIn("height: 100vh !important", css)


if __name__ == "__main__":
    unittest.main()
