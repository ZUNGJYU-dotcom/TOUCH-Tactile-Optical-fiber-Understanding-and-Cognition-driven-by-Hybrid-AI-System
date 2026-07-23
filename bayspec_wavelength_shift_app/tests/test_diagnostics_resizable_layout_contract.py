from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class DiagnosticsResizableLayoutContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        cls.css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        cls.js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    def test_diagnostics_rail_has_an_accessible_drag_separator(self) -> None:
        self.assertIn('id="diagnosticsPanelResizer"', self.html)
        self.assertIn('role="separator"', self.html)
        self.assertIn('aria-orientation="vertical"', self.html)
        self.assertIn('data-lucide="grip-vertical"', self.html)

    def test_width_is_persisted_and_supports_pointer_and_keyboard_resize(self) -> None:
        self.assertIn('DIAGNOSTICS_PANEL_WIDTH_STORAGE_KEY', self.js)
        self.assertIn('setPointerCapture(event.pointerId)', self.js)
        self.assertIn('event.key === "ArrowLeft"', self.js)
        self.assertIn('event.key === "ArrowRight"', self.js)
        self.assertIn('storedValue === null || storedValue.trim() === ""', self.js)
        self.assertIn('window.localStorage.setItem(DIAGNOSTICS_PANEL_WIDTH_STORAGE_KEY', self.js)

    def test_recording_form_reflows_without_horizontal_clipping(self) -> None:
        self.assertIn('container: diagnostics-rail / inline-size', self.css)
        self.assertIn('@container diagnostics-rail (max-width: 359px)', self.css)
        self.assertIn('min-inline-size: 0', self.css)
        self.assertIn('grid-template-columns: minmax(0, 1fr)', self.css)
        self.assertIn('grid-template-columns: repeat(auto-fit, minmax(104px, 1fr))', self.css)
        self.assertIn('overflow-wrap: anywhere', self.css)

    def test_recording_commands_are_short_and_do_not_repeat_context(self) -> None:
        self.assertIn('<summary><span>Data recording</span>', self.html)
        self.assertIn('<span>Start</span>', self.html)
        self.assertIn('<span>Stop</span>', self.html)
        self.assertNotIn('Start linked recording', self.html)
        self.assertNotIn('linked recording', self.js.lower())
        self.assertNotIn('Stop &amp; save', self.html)

    def test_recording_uses_continuous_px6d_force_instead_of_action_classes(self) -> None:
        self.assertNotIn('id="px6dCaptureAction"', self.html)
        self.assertIn('<span>Force Sensor</span>', self.html)
        self.assertIn('id="px6dCaptureForceValue"', self.html)
        self.assertIn('action_label: "continuous_px6d_fz_reference"', self.js)
        self.assertIn('setText(\n    "px6dCaptureForceValue"', self.js)

    def test_recording_stream_names_are_plain_and_describe_saved_data(self) -> None:
        self.assertIn('<legend>Save</legend>', self.html)
        self.assertIn('value="spectrum" checked /> Spectrum</label>', self.html)
        self.assertIn('value="response" checked /> Recognition</label>', self.html)
        self.assertIn('value="force" checked /> Force</label>', self.html)

    def test_recording_workflow_has_preflight_position_duration_and_next_trial(self) -> None:
        for element_id in (
            "px6dCaptureSpectrumReady",
            "px6dCaptureForceReady",
            "px6dCapturePositionReady",
            "px6dCaptureFolderReady",
            "px6dCaptureDuration",
            "px6dCaptureNextTrialButton",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        expected_order = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
        offsets = [self.html.index(f'data-capture-position="{channel}"') for channel in expected_order]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("function updateCaptureReadiness()", self.js)
        self.assertIn("function nextCaptureTrialId()", self.js)
        self.assertIn("Check ${readiness.missing.join", self.js)
        self.assertIn('.diagnostic-capture-position-grid button:disabled', self.css)
        self.assertIn('.capture-summary-status[data-state="error"]', self.css)

    def test_heatmap_title_stays_on_one_line_in_compact_panel(self) -> None:
        self.assertIn('<h2 class="heatmap-title">2D Response Map</h2>', self.html)
        self.assertIn('id="heatmapChip" class="chip explanatory-copy"', self.html)
        self.assertIn('.diagnostics-mode .heatmap-panel .heatmap-title', self.css)
        self.assertIn('white-space: nowrap', self.css)

    def test_widened_diagnostics_rail_cannot_overlap_surface_metrics(self) -> None:
        self.assertIn('const DIAGNOSTICS_CENTER_MIN_WIDTH_PX = 520;', self.js)
        self.assertIn(
            'const DIAGNOSTICS_CENTER_COMPACT_MIN_WIDTH_PX = 420;',
            self.js,
        )
        self.assertIn(
            'const DIAGNOSTICS_COMPACT_BREAKPOINT_PX = 1100;',
            self.js,
        )
        self.assertIn(
            'grid-template-columns: 180px minmax(520px, 1fr) var(--diagnostics-right-width)',
            self.css,
        )
        self.assertIn(
            '.diagnostics-mode .surface-cockpit-grid,\n'
            '.diagnostics-mode .three-mount,\n'
            '.diagnostics-mode .heatmap-panel,\n'
            '.diagnostics-mode .heatmap-canvas',
            self.css,
        )
        self.assertIn('min-height: 0 !important;', self.css)
        self.assertIn('overflow-y: auto;', self.css)
        self.assertIn('.diagnostics-mode .stage-metrics,', self.css)
        self.assertIn('z-index: 3;', self.css)

    def test_center_stage_has_a_paint_boundary_and_narrow_reflow(self) -> None:
        collision_guard = self.css.split(
            '/* Diagnostics center collision guard.',
            1,
        )[1]
        self.assertIn('container: diagnostics-stage / inline-size;', collision_guard)
        self.assertIn(
            'grid-template-rows: auto minmax(260px, 1fr) auto auto !important;',
            collision_guard,
        )
        self.assertIn('min-height: 260px !important;', collision_guard)
        self.assertIn('contain: layout paint;', collision_guard)
        self.assertIn(
            'grid-template-rows: auto 506px auto auto !important;',
            collision_guard,
        )
        self.assertIn('height: 100%;', collision_guard)
        self.assertIn('height: 100% !important;', collision_guard)
        self.assertIn('align-self: stretch;', collision_guard)
        self.assertIn('@container diagnostics-stage (max-width: 700px)', collision_guard)
        self.assertIn('grid-template-columns: minmax(0, 1fr);', collision_guard)
        self.assertIn('.diagnostics-mode #tactileSurfaceTitle', collision_guard)
        self.assertIn('white-space: nowrap;', collision_guard)
        self.assertIn('justify-content: flex-start;', collision_guard)
        self.assertIn('@container diagnostics-stage (max-width: 419px)', collision_guard)
        self.assertIn(
            'grid-template-columns: minmax(0, 1fr) !important;',
            collision_guard,
        )
        self.assertIn(
            'grid-template-columns: repeat(2, minmax(0, 1fr)) !important;',
            collision_guard,
        )
        self.assertIn('min-height: 506px !important;', collision_guard)
        self.assertIn('height: 506px;', collision_guard)
        self.assertIn('aspect-ratio: auto;', collision_guard)

    def test_twin_and_heatmap_use_one_proportional_resize_rule(self) -> None:
        collision_guard = self.css.split(
            '/* Diagnostics center collision guard.',
            1,
        )[1]
        self.assertIn(
            'grid-template-columns: minmax(0, 1.43fr) minmax(0, 1fr) !important;',
            collision_guard,
        )
        self.assertIn('class="heatmap-canvas-viewport"', self.html)
        self.assertIn(
            '.diagnostics-mode .heatmap-canvas-viewport {',
            collision_guard,
        )
        self.assertIn('display: grid !important;', collision_guard)
        self.assertIn('align-content: stretch;', collision_guard)
        self.assertIn('position: relative;', collision_guard)
        self.assertIn(
            '.diagnostics-mode .heatmap-canvas {\n'
            '  position: absolute;\n'
            '  inset: 0;\n'
            '  width: 100% !important;\n'
            '  height: 100% !important;',
            collision_guard,
        )
        self.assertIn('min-width: 0;', collision_guard)
        self.assertIn('aspect-ratio: auto;', collision_guard)

    def test_narrow_window_width_bounds_never_force_dashboard_overflow(self) -> None:
        self.assertIn(
            'dashboardWidth <= DIAGNOSTICS_COMPACT_BREAKPOINT_PX',
            self.js,
        )
        self.assertIn(
            '? DIAGNOSTICS_CENTER_COMPACT_MIN_WIDTH_PX',
            self.js,
        )
        self.assertIn(
            'const available = Math.max(\n    0,',
            self.js,
        )
        self.assertIn(
            'const min = Math.min(DIAGNOSTICS_PANEL_MIN_WIDTH_PX, available);',
            self.js,
        )

    def test_explanatory_copy_is_hidden_but_operational_status_remains(self) -> None:
        self.assertIn('<body class="minimal-copy">', self.html)
        self.assertIn('.minimal-copy .explanatory-copy', self.css)
        self.assertIn('display: none !important', self.css)
        self.assertIn('id="px6dCaptureStatus"', self.html)
        self.assertIn('id="px6dCaptureOutput"', self.html)
        self.assertIn('id="commandFeedbackText"', self.html)

    def test_operator_spectrum_is_compact_and_freed_space_has_direct_readouts(self) -> None:
        self.assertIn('class="hud-card operator-current-hud"', self.html)
        self.assertIn('<span>Contact</span>', self.html)
        self.assertIn('id="operatorContactValue"', self.html)
        self.assertIn('<span>Position</span>', self.html)
        self.assertIn('id="operatorPositionValue"', self.html)
        self.assertIn('aria-label="Force sensor"', self.html)
        self.assertIn('<span>Force Sensor</span>', self.html)
        self.assertEqual(self.html.count('id="px6dReferenceFz"'), 1)
        self.assertIn('grid-template-rows: 144px clamp(238px, 34vh, 310px)', self.css)
        self.assertIn('max-height: 310px !important', self.css)
        self.assertIn('setText("operatorPositionValue", operatorPosition)', self.js)
        self.assertIn('"operatorContactValue",', self.js)

    def test_fullscreen_summary_reuses_force_readout_without_clipping(self) -> None:
        self.assertEqual(self.html.count('id="px6dReferenceFz"'), 1)
        self.assertIn('const operatorForceReadout = document.querySelector(".operator-force-readout")', self.js)
        self.assertIn('state.surfaceFullscreenActive ? operatorSummaryCardNode : operatorCurrentHud', self.js)
        self.assertIn('.app-shell.surface-fullscreen-active .operator-summary-card', self.css)
        self.assertIn('height: auto !important', self.css)
        self.assertIn('overflow-y: auto !important', self.css)

    def test_recording_has_a_dedicated_early_workspace(self) -> None:
        self.assertIn('data-tooltip="Data recording"', self.html)
        self.assertIn('data-tooltip="Input diagnostics"', self.html)
        self.assertIn('data-tooltip="Mechanical reference"', self.html)
        self.assertIn(
            'class="prediction-card diagnostics-only diagnostic-card diagnostic-capture-card" data-diagnostic-group="recording"',
            self.html,
        )
        self.assertNotIn('class="diagnostic-subsection diagnostic-capture-card"', self.html)
        signal_offset = self.html.index('data-diagnostic-tab="signal"')
        recording_offset = self.html.index('data-diagnostic-tab="recording"')
        surface_offset = self.html.index('data-diagnostic-tab="surface"')
        self.assertLess(signal_offset, recording_offset)
        self.assertLess(recording_offset, surface_offset)
        self.assertEqual(self.html.count('data-diagnostic-group="acquisition"'), 1)
        self.assertEqual(self.html.count('data-diagnostic-group="recording"'), 1)

    def test_demo_scenario_selection_keeps_diagnostics_panel_expanded(self) -> None:
        self.assertIn(
            "function closeOperatorDemoMenuAfterScenarioSelection()",
            self.js,
        )
        self.assertIn(
            'if (state.displayMode === "diagnostics") return;',
            self.js,
        )
        self.assertEqual(
            self.js.count("closeOperatorDemoMenuAfterScenarioSelection();"),
            2,
        )
        self.assertNotIn(
            'stopDemoAutoplay();\n    setDemoMenuOpen(false);',
            self.js,
        )

    def test_active_toolbar_states_recolor_icons_without_inserting_dots(self) -> None:
        for selector in (
            ".operator-mode #liveTwinButton.active-watch::before",
            ".operator-mode #demoMenuButton.demo-active::before",
            ".operator-mode .control-actions button.command-busy::before",
        ):
            start = self.css.index(selector)
            block = self.css[start : self.css.index("}", start) + 1]
            self.assertIn("content: none", block)
            self.assertIn("display: none", block)
            self.assertNotIn("width:", block)
            self.assertNotIn("flex:", block)
        self.assertIn(
            ".operator-mode #demoMenuButton.demo-active svg",
            self.css,
        )
        self.assertIn(
            ".operator-mode .control-actions button.command-busy svg",
            self.css,
        )

    def test_response_trace_header_only_shows_a_plain_paused_state(self) -> None:
        self.assertIn(
            'id="signalQaSummary" class="explanatory-copy" hidden aria-hidden="true"',
            self.html,
        )
        self.assertIn(
            'id="traceChip" class="chip" aria-live="polite" hidden>PAUSED</span>',
            self.html,
        )
        self.assertIn('traceChip.hidden = !state.paused;', self.js)
        self.assertIn('traceChip.textContent = state.paused ? "PAUSED" : "";', self.js)
        self.assertNotIn('traceChip.textContent = `${traceScopeLabel} · ${streamLabel}`;', self.js)
        self.assertIn('.operator-mode .response-hud #traceChip[hidden]', self.css)
        self.assertIn('.operator-mode .response-hud #traceChip:not([hidden])', self.css)
        self.assertIn('.response-hud #traceChip:not([hidden])', self.css)
        self.assertIn('position: absolute', self.css)
        diagnostics_summary = self.css.index('.diagnostics-mode .response-hud #signalQaSummary')
        diagnostics_summary_block = self.css[
            diagnostics_summary : self.css.index('}', diagnostics_summary) + 1
        ]
        self.assertIn('display: none !important', diagnostics_summary_block)


if __name__ == "__main__":
    unittest.main()
