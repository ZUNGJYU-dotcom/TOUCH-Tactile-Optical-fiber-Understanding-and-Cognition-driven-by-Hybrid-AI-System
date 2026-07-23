from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class OperatorResponsiveLayoutContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        cls.launcher = (APP_ROOT / "desktop_launcher.py").read_text(encoding="utf-8")
        cls.app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        cls.html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    def test_operator_tracks_shrink_without_hiding_core_panels(self) -> None:
        final_lock = self.css.index("/* Final cascade lock for the simplified Operator workspace. */")
        responsive_css = self.css[final_lock:]
        self.assertRegex(
            responsive_css,
            re.compile(
                r"\.operator-mode \.dashboard\s*\{[^}]*"
                r"grid-template-columns:\s*"
                r"clamp\(174px, 15\.5vw, 210px\)\s*"
                r"minmax\(0, 1fr\)\s*"
                r"clamp\(224px, 20vw, 278px\)",
                re.S,
            ),
        )
        self.assertIn(".operator-mode .dashboard > *", responsive_css)
        self.assertIn("min-width: 0 !important", responsive_css)
        self.assertIn("overflow: hidden !important", responsive_css)

    def test_desktop_starts_restored_then_supports_true_fullscreen(self) -> None:
        self.assertIn("maximized=False", self.launcher)
        self.assertIn("window.toggle_fullscreen()", self.launcher)
        self.assertIn("width=1180", self.launcher)
        self.assertIn("height=760", self.launcher)
        self.assertIn("min_size=(1024, 680)", self.launcher)
        self.assertNotIn("fullscreen=True", self.launcher)
        self.assertNotIn("resizable=False", self.launcher)

    def test_scaled_restored_windows_use_compact_command_and_evidence_tracks(self) -> None:
        final_lock = self.css.index("/* Final cascade lock for the simplified Operator workspace. */")
        responsive_css = self.css[final_lock:]

        self.assertIn("@media (max-width: 1920px)", responsive_css)
        self.assertIn(
            "grid-template-columns: 170px 390px minmax(252px, 1fr) !important",
            responsive_css,
        )
        self.assertRegex(
            responsive_css,
            re.compile(
                r"clamp\(160px, 13\.5vw, 190px\)\s*"
                r"minmax\(0, 1fr\)\s*"
                r"clamp\(194px, 17vw, 232px\)",
                re.S,
            ),
        )
        self.assertIn(
            "grid-template-columns: 160px 350px minmax(0, 1fr) !important",
            responsive_css,
        )
        self.assertIn(
            "grid-template-columns: 180px minmax(0, 1fr) 300px !important",
            responsive_css,
        )
        self.assertIn(
            "flex: 0.9 1 112px !important",
            responsive_css,
        )
        self.assertIn(".operator-mode .status-strip .qa-status-item", responsive_css)
        self.assertIn(".operator-mode .status-strip .status-item:nth-child(2) svg", responsive_css)
        self.assertIn(".operator-mode .spectrum-open-chip", responsive_css)
        self.assertIn("width: 38px !important", responsive_css)

    def test_idle_spectrum_hides_only_empty_kpis(self) -> None:
        final_lock = self.css.index("/* Final cascade lock for the simplified Operator workspace. */")
        responsive_css = self.css[final_lock:]

        self.assertIn(
            '.operator-mode .summary-hud[data-measurement-state="no_data"] > .optical-kpi-grid',
            responsive_css,
        )
        idle_rule = responsive_css.split(
            '.operator-mode .summary-hud[data-measurement-state="no_data"] > .optical-kpi-grid',
            1,
        )[1].split("}", 1)[0]
        self.assertIn("display: none !important", idle_rule)
        self.assertNotIn(
            '.operator-mode .summary-hud[data-measurement-state="current"] > .optical-kpi-grid',
            responsive_css,
        )

    def test_operator_spectrum_preview_is_not_gated_by_drawer_visibility(self) -> None:
        draw_charts = self.app_js.split("function drawVisibleCharts()", 1)[1].split(
            "function updateChartSmoothing", 1
        )[0]
        preview_call = draw_charts.index("drawOpticalPreview(state.smoothSpectrumRecord)")
        visibility_gate = draw_charts.index("const spectrumVisible")
        self.assertLess(preview_call, visibility_gate)

    def test_operator_spectrum_has_a_state_driven_idle_overlay(self) -> None:
        self.assertIn('class="optical-preview-empty-state"', self.html)
        self.assertIn("No spectrum frame", self.html)
        self.assertIn(
            '.operator-mode .summary-hud[data-measurement-state="no_data"] .optical-preview-empty-state',
            self.css,
        )
        preview_function = self.app_js.split("function drawOpticalPreview(record)", 1)[1].split(
            "function drawHeatmap", 1
        )[0]
        self.assertNotIn('ctx.fillText("No spectrum frame"', preview_function)

    def test_trace_idle_state_is_status_only(self) -> None:
        draw_trace = self.app_js.split("function drawTrace(records)", 1)[1].split(
            "function globalEventResponseFromTrace", 1
        )[0]
        self.assertIn('"No response history"', draw_trace)
        self.assertIn('ctx.fillText("IDLE"', draw_trace)
        self.assertNotIn("Start Live, Watch, or Response", draw_trace)

    def test_active_state_kpis_use_compact_values_with_units_in_labels(self) -> None:
        for label in ("Now · pm", "Peak · pm", "λ · nm", "Δλ · pm", "|Δλ| · pm"):
            self.assertIn(label, self.html)
        self.assertIn("function setCompactMetric", self.app_js)
        self.assertIn('setCompactMetric(\n    "traceCurrentValue"', self.app_js)
        self.assertIn('setCompactMetric(\n    "tracePeakValue"', self.app_js)
        self.assertIn('setCompactMetric(\n    "metricIntensity"', self.app_js)
        self.assertIn("formatCompactNumber(compactPeakWavelength, 1)", self.app_js)
        self.assertIn('setCompactMetric(\n    "metricRelative"', self.app_js)
        self.assertIn('setCompactMetric(\n    "metricLoss"', self.app_js)

    def test_compact_diagnostics_header_uses_complete_icon_commands(self) -> None:
        compact_media = self.css.split("@media (max-width: 1100px)", 1)[1]
        self.assertIn(
            ".lucide-ready .diagnostics-mode .control-actions .ui-icon-button .command-label",
            compact_media,
        )
        self.assertIn("display: none !important", compact_media)
        self.assertIn('title="Peak wavelength (nm)"', self.html)

    def test_operator_active_state_uses_short_source_and_coupling_tokens(self) -> None:
        self.assertIn('label: operatorMode ? "LOCAL"', self.app_js)
        coupling_block = self.app_js.split('"surfaceRuleSource"', 1)[1].split(");", 1)[0]
        self.assertNotIn('"coupled response"', coupling_block)
        self.assertGreaterEqual(coupling_block.count('? "coupled"'), 3)

    def test_operator_response_gauge_is_continuous_not_categorical(self) -> None:
        band_markup = self.html.split('<div class="operator-band-card">', 1)[1].split(
            '<nav class="diagnostic-tabs', 1
        )[0]
        self.assertIn("Optical Response", band_markup)
        self.assertIn("Continuous normalized optical response", band_markup)
        for label in ("0%", "25%", "50%", "75%", "100%"):
            self.assertIn(f"<span>{label}</span>", band_markup)
        for legacy_label in ("<span>light</span>", "<span>normal</span>", "<span>hard</span>"):
            self.assertNotIn(legacy_label, band_markup)

        update_block = self.app_js.split(
            'const marker = document.getElementById("responseBandMarker");', 1
        )[1].split('let note = "raw coupled response";', 1)[0]
        self.assertIn("Continuous normalized optical response from 0% to 100%", update_block)
        self.assertIn("optical response", update_block)
        self.assertNotIn("responseBandLevel", update_block)
        self.assertNotIn("levelLabel(", update_block)

        track_rule = self.css.split(".operator-mode .response-band-track {", 1)[1].split("}", 1)[0]
        self.assertNotIn("--response-small-end", track_rule)
        labels_rule = self.css.split(".operator-mode .response-band-labels {", 1)[1].split("}", 1)[0]
        self.assertIn("repeat(5, minmax(0, 1fr))", labels_rule)

    def test_operator_surface_state_uses_contact_not_force_bands(self) -> None:
        presentation = self.app_js.split("function surfaceContactPresentation(", 1)[1].split(
            "function clampArrayCoord", 1
        )[0]
        self.assertIn('primary: "contact"', presentation)
        self.assertNotIn("primary: levelLabel(", presentation)

        summary = self.app_js.split("const displaySurfaceResponseLevel =", 1)[1].split(
            'setText("surfaceResponseLevel"', 1
        )[0]
        self.assertIn('? "contact"', summary)
        self.assertIn(': "no_contact"', summary)
        self.assertNotIn("levelLabel(", summary)


if __name__ == "__main__":
    unittest.main()
