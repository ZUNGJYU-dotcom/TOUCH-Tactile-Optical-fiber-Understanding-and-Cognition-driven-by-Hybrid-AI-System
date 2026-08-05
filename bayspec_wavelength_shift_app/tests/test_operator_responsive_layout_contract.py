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

    def test_narrow_operator_evidence_cards_cannot_overlap(self) -> None:
        guard = self.css.index("/* Responsive evidence-rail guard.")
        responsive_css = self.css[guard:]

        left_panel_rule = responsive_css.split(".operator-mode .left-panel {", 1)[1].split(
            "}", 1
        )[0]
        self.assertIn(
            "grid-template-rows: 144px minmax(264px, 1fr) auto !important",
            left_panel_rule,
        )
        self.assertIn("overflow-y: auto !important", left_panel_rule)
        self.assertIn("overflow-x: hidden !important", left_panel_rule)

        summary_rule = responsive_css.split(".operator-mode .summary-hud {", 1)[1].split(
            "}", 1
        )[0]
        self.assertIn("height: auto !important", summary_rule)
        self.assertIn("min-height: 264px !important", summary_rule)
        self.assertIn("overflow: hidden !important", summary_rule)

        kpi_rule = responsive_css.split(
            ".operator-mode .summary-hud > .optical-kpi-grid {", 1
        )[1].split("}", 1)[0]
        self.assertIn("min-height: 97px !important", kpi_rule)
        self.assertIn("grid-template-rows: repeat(2, 46px) !important", kpi_rule)

        compact_css = responsive_css.split("@media (max-width: 1100px)", 1)[1]
        self.assertIn(
            "grid-template-rows: 138px minmax(238px, 1fr) auto !important",
            compact_css,
        )
        self.assertIn("min-height: 238px !important", compact_css)
        self.assertIn("grid-template-rows: repeat(2, 42px) !important", compact_css)

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
        self.assertNotIn("spectrum-open-chip", self.html)
        self.assertNotIn(".operator-mode .spectrum-open-chip", responsive_css)

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
        overlay = self.html.split('class="optical-preview-empty-state"', 1)[1].split(
            "</div>", 1
        )[0]
        self.assertIn("No spectrum frame", overlay)
        self.assertNotIn("data-lucide", overlay)
        self.assertNotIn("IDLE", overlay)
        self.assertIn(
            '.operator-mode .summary-hud[data-measurement-state="no_data"] .optical-preview-empty-state',
            self.css,
        )
        preview_function = self.app_js.split("function drawOpticalPreview(record)", 1)[1].split(
            "function drawHeatmap", 1
        )[0]
        self.assertNotIn('ctx.fillText("No spectrum frame"', preview_function)

    def test_compact_evidence_rail_uses_single_line_sidebar_typography(self) -> None:
        compact_pass = self.css.split("/* iOS clarity pass 12:", 1)[1]
        self.assertIn(
            "container: operator-left-evidence-rail / inline-size",
            compact_pass,
        )
        title_rule = compact_pass.split(
            ".operator-mode .left-panel .hud-title-row h2 {", 1
        )[1].split("}", 1)[0]
        self.assertIn("font-size: 15px !important", title_rule)
        self.assertIn("text-overflow: ellipsis", title_rule)
        self.assertIn("white-space: nowrap", title_rule)

        kpi_rule = compact_pass.split(
            ".operator-mode .optical-kpi-grid span {", 1
        )[1].split("}", 1)[0]
        self.assertIn("font-size: 9px !important", kpi_rule)
        self.assertIn("white-space: nowrap", kpi_rule)

        self.assertIn(
            "@container operator-left-evidence-rail (max-width: 165px)",
            compact_pass,
        )
        narrow_rule = compact_pass.split(
            "@container operator-left-evidence-rail (max-width: 165px)", 1
        )[1].split(
            ".operator-mode .left-panel .hud-title-row h2 {", 1
        )[1].split("}", 1)[0]
        self.assertIn("font-size: 14px !important", narrow_rule)

    def test_empty_spectrum_uses_compact_state_specific_track(self) -> None:
        compact_pass = self.css.split("/* iOS clarity pass 13:", 1)[1]
        empty_selector = (
            '.operator-mode .left-panel:has(> '
            '.summary-hud[data-measurement-state="no_data"])'
        )
        self.assertIn(empty_selector, compact_pass)

        desktop_rule = compact_pass.split(empty_selector + " {", 1)[1].split("}", 1)[0]
        self.assertIn(
            "grid-template-rows: 144px clamp(214px, 29vh, 232px) auto !important",
            desktop_rule,
        )
        self.assertIn("align-content: start !important", desktop_rule)

        short_window = compact_pass.split("@media (max-height: 720px)", 1)[1]
        short_rule = short_window.split(empty_selector + " {", 1)[1].split("}", 1)[0]
        self.assertIn("grid-template-rows: 132px 210px 92px !important", short_rule)

        self.assertIn(
            "grid-template-rows: 144px minmax(264px, 1fr) auto !important",
            self.css,
        )

    def test_operator_shell_removes_global_graph_paper_but_keeps_scene_grid(self) -> None:
        clarity_pass = self.css.split("/* iOS clarity pass 14:", 1)[1]
        shell_rule = clarity_pass.split(".operator-mode {", 1)[1].split("}", 1)[0]
        self.assertIn("background: #f4f7fa !important", shell_rule)
        self.assertIn("background-image: none !important", shell_rule)

        pseudo_rule = clarity_pass.split(
            ".operator-mode.app-shell::before {", 1
        )[1].split("}", 1)[0]
        self.assertIn("content: none !important", pseudo_rule)
        self.assertIn("display: none !important", pseudo_rule)

        self.assertNotIn(".three-mount", clarity_pass)
        self.assertIn(".operator-mode .three-mount {", self.css)

    def test_operator_evidence_cards_do_not_use_industrial_accent_rails(self) -> None:
        clarity_pass = self.css.split("/* iOS clarity pass 15:", 1)[1]
        self.assertIn(".operator-mode .hud-card", clarity_pass)
        self.assertIn(".operator-mode .operator-summary-card", clarity_pass)
        self.assertIn(".operator-mode .operator-footprint-card", clarity_pass)
        self.assertIn(".operator-mode .operator-band-card", clarity_pass)
        self.assertIn("background: #ffffff !important", clarity_pass)
        self.assertIn("background-image: none !important", clarity_pass)
        self.assertNotIn("linear-gradient", clarity_pass)
        self.assertNotIn("inset", clarity_pass)

        self.assertIn(
            '#surfaceResponseLevel[data-response-tone="hard"]',
            self.css,
        )

    def test_trace_idle_state_is_status_only(self) -> None:
        draw_trace = self.app_js.split("function drawTrace(records)", 1)[1].split(
            "function globalEventResponseFromTrace", 1
        )[0]
        self.assertIn('"No response history"', draw_trace)
        self.assertNotIn('ctx.fillText("IDLE"', draw_trace)
        self.assertNotIn("Start Live, Watch, or Response", draw_trace)

        trace_kpis = self.app_js.split("function updateTraceKpis(records)", 1)[1].split(
            "function drawTrace(records)", 1
        )[0]
        self.assertIn('setText("traceHistoryValue", "--")', trace_kpis)
        self.assertNotIn('setText("traceHistoryValue", "IDLE")', trace_kpis)

    def test_beta_trace_uses_continuous_optical_force_before_contact_gate(self) -> None:
        self.assertIn("function traceOpticalForceN(item)", self.app_js)
        self.assertIn('rawValue === null || rawValue === undefined || rawValue === ""', self.app_js)
        self.assertIn("const displayForceN = finiteNumberOrNull(contract?.force?.display_n)", self.app_js)
        self.assertIn('trace_response_semantics: "canonical_operator_display_force_n"', self.app_js)
        draw_trace = self.app_js.split("function drawTrace(records)", 1)[1].split(
            "function globalEventResponseFromTrace", 1
        )[0]
        self.assertIn("updateOpticalForceDisplayMaximum", draw_trace)
        self.assertIn("forceAxisMax", draw_trace)
        self.assertNotIn("OPTICAL_FORCE_DISPLAY_MAX_N", draw_trace)
        self.assertIn('"Optical Fz estimate (N)"', draw_trace)
        self.assertIn('"continuous_optical_force_n"', draw_trace)
        self.assertIn('"Collecting optical force history"', draw_trace)
        self.assertIn("if (yMin < 0 || opticalForceTrace)", draw_trace)

    def test_active_state_kpis_use_compact_values_with_shared_trace_unit(self) -> None:
        for label in ("Now", "Peak", "λ · nm", "Δλ · pm", "|Δλ| · pm"):
            self.assertIn(label, self.html)
        self.assertNotIn("Now · pm", self.html)
        self.assertNotIn("Peak · pm", self.html)
        self.assertIn('setText("traceCurrentLabel", "Now")', self.app_js)
        self.assertIn('setText("tracePeakLabel", "Peak")', self.app_js)
        self.assertIn('"Optical Fz estimate (N)"', self.app_js)
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
        self.assertIn(
            '? state.displayMode === "operator" ? "LOCAL" : "SIMULATED"',
            self.app_js,
        )
        coupling_block = self.app_js.split('"surfaceRuleSource"', 1)[1].split(");", 1)[0]
        self.assertIn('!measurementAvailable', coupling_block)
        self.assertIn('? "--"', coupling_block)
        self.assertNotIn('"coupled response"', coupling_block)
        self.assertGreaterEqual(coupling_block.count('? "coupled"'), 3)

        coupling_markup = self.html.split('id="surfaceRuleSource"', 1)[1].split(
            "</strong>", 1
        )[0]
        self.assertIn(">--", coupling_markup)

    def test_source_status_uses_only_state_dot_and_text(self) -> None:
        source_status = self.html.split(
            '<span class="status-item" title="Data source">', 1
        )[1].split("</span>", 1)[0]
        self.assertNotIn("data-lucide", source_status)
        self.assertIn('<strong id="sourceChip">', source_status)

        live_command = self.html.split('id="liveTwinButton"', 1)[1].split(
            "</button>", 1
        )[0]
        self.assertIn('data-lucide="radio-tower"', live_command)

    def test_frontend_status_text_has_no_mojibake_separators(self) -> None:
        for damaged_text in ("路", "卤"):
            self.assertNotIn(damaged_text, self.app_js)
        for source_label in ("SDK | idle", "Watch | idle", "HTTP | idle"):
            self.assertIn(source_label, self.app_js)

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

    def test_short_native_window_keeps_bottom_evidence_inside_its_tracks(self) -> None:
        pass5 = self.css.index("/* iOS clarity pass 5:")
        compact_css = self.css[pass5:]
        self.assertIn(
            "grid-template-rows: 132px minmax(0, 1fr) 92px !important",
            compact_css,
        )
        self.assertIn(
            "grid-template-rows: 176px minmax(0, 1fr) 88px !important",
            compact_css,
        )
        self.assertIn(".operator-mode .operator-force-readout > div", compact_css)
        self.assertIn("height: 12px !important", compact_css)

    def test_operator_evidence_titles_use_sidebar_typography(self) -> None:
        pass16 = self.css.split("/* iOS clarity pass 16:", 1)[1]
        self.assertIn(
            ".operator-mode .left-panel .hud-title-row h2",
            pass16,
        )
        self.assertIn("font-size: 13px !important", pass16)
        self.assertIn("font-weight: 650 !important", pass16)
        self.assertIn(
            "@container operator-left-evidence-rail (max-width: 165px)",
            pass16,
        )
        self.assertIn("font-size: 12px !important", pass16)

    def test_operator_text_lines_keep_full_glyph_height_at_minimum_width(self) -> None:
        pass18 = self.css.split("/* iOS clarity pass 18:", 1)[1]
        self.assertIn(
            ".operator-mode .left-panel .hud-title-row h2",
            pass18,
        )
        self.assertIn("min-height: 17px", pass18)
        self.assertIn("line-height: 1.2 !important", pass18)
        self.assertIn(".operator-mode .section-label", pass18)
        self.assertIn("min-height: 15px", pass18)
        self.assertIn(
            ".operator-mode .operator-summary-card .operator-summary-grid span",
            pass18,
        )
        self.assertIn("line-height: 14px !important", pass18)
        self.assertIn(".operator-mode .mini-pixel strong", pass18)
        self.assertIn("line-height: 15px !important", pass18)

    def test_operator_status_and_right_evidence_use_single_grouped_surfaces(self) -> None:
        pass19 = self.css.split("/* iOS clarity pass 19:", 1)[1]
        self.assertIn(".operator-mode .status-strip", pass19)
        self.assertIn("gap: 0 !important", pass19)
        self.assertIn("background: #f6f9fb", pass19)
        self.assertIn(
            ".operator-mode .status-strip .status-item + .status-item::before",
            pass19,
        )
        self.assertIn("background: #dde6ec", pass19)

        normal_right_rail = (
            ".app-shell.operator-mode:not(.surface-fullscreen-active)"
            ":not(.surface-only-view)"
        )
        self.assertIn(normal_right_rail, pass19)
        self.assertIn(".right-panel", pass19)
        self.assertIn("background: #ffffff !important", pass19)
        self.assertIn("> .operator-summary-card", pass19)
        self.assertIn("> .operator-footprint-card", pass19)
        self.assertIn("> .operator-band-card", pass19)
        self.assertIn("box-shadow: none !important", pass19)
        self.assertIn("border-top: 1px solid #e2eaf0 !important", pass19)


if __name__ == "__main__":
    unittest.main()
