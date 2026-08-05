from pathlib import Path
import re
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class DiagnosticsWorkspaceSimplificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        cls.css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        cls.js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    def test_eight_workspaces_use_a_scrollable_navigation_row(self) -> None:
        self.assertEqual(self.html.count("data-diagnostic-tab="), 8)
        self.assertRegex(
            self.css,
            re.compile(
                r"\.diagnostics-mode \.diagnostic-tabs\s*\{[^}]*"
                r"display:\s*flex\s*!important[^}]*"
                r"overflow-x:\s*auto",
                re.S,
            ),
        )
        self.assertIn("flex: 1 0 52px", self.css)
        self.assertIn("min-width: 52px", self.css)
        self.assertIn('button.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" })', self.js)

    def test_compact_diagnostics_rail_keeps_all_workspace_tabs_visible(self) -> None:
        pass5 = self.css.index("/* iOS clarity pass 5:")
        compact_css = self.css[pass5:]
        self.assertIn("@container diagnostics-rail (max-width: 359px)", compact_css)
        self.assertIn("flex: 1 1 36px", compact_css)
        self.assertIn("min-width: 36px", compact_css)
        self.assertIn("font-size: 8.5px", compact_css)

    def test_diagnostics_evidence_rail_uses_compact_single_line_titles(self) -> None:
        pass20 = self.css.index("/* iOS clarity pass 20:")
        compact_css = self.css[pass20:]
        self.assertIn(
            "container: diagnostics-left-evidence-rail / inline-size",
            compact_css,
        )
        title_rule = compact_css.split(
            ".diagnostics-mode .left-panel .hud-title-row h2 {", 1
        )[1].split("}", 1)[0]
        self.assertIn("font-size: 14px !important", title_rule)
        self.assertIn("text-overflow: ellipsis", title_rule)
        self.assertIn("white-space: nowrap", title_rule)
        self.assertIn(
            "@container diagnostics-left-evidence-rail (max-width: 185px)",
            compact_css,
        )

    def test_diagnostics_spectrum_kpis_have_complete_line_boxes(self) -> None:
        pass20 = self.css.index("/* iOS clarity pass 20:")
        compact_css = self.css[pass20:]
        grid_rule = compact_css.split(
            ".diagnostics-mode .summary-hud > div.optical-kpi-grid {", 1
        )[1].split("}", 1)[0]
        self.assertIn("min-height: 146px !important", grid_rule)
        self.assertIn(
            "grid-template-rows: repeat(2, minmax(67px, 1fr)) !important",
            grid_rule,
        )
        self.assertIn("overflow: visible !important", grid_rule)
        label_rule = compact_css.split(
            ".diagnostics-mode .optical-kpi-grid span {", 1
        )[1].split("}", 1)[0]
        self.assertIn("font-size: 10px !important", label_rule)
        self.assertIn("text-overflow: ellipsis", label_rule)
        self.assertIn("white-space: nowrap", label_rule)

    def test_operator_camera_navigation_does_not_clutter_diagnostics(self) -> None:
        self.assertRegex(
            self.css,
            re.compile(
                r"\.diagnostics-mode \.finger-closeup-nav\s*\{[^}]*"
                r"display:\s*none\s*!important",
                re.S,
            ),
        )

    def test_workspace_tabs_use_compact_labels_with_accessible_full_names(self) -> None:
        for visible_label in (
            "Signal",
            "Record",
            "Measure",
            "Surface",
            "Demo",
            "Input",
            "Force",
            "3D",
        ):
            self.assertIn(f"<span>{visible_label}</span>", self.html)
        for accessible_name in (
            "Signal diagnostics",
            "Data recording",
            "Measurement analysis",
            "Surface diagnostics",
            "Demo controls",
            "Input diagnostics",
            "Mechanical reference",
            "Geometry diagnostics",
        ):
            self.assertIn(f'aria-label="{accessible_name}"', self.html)
        for icon_name in (
            "activity",
            "circle-dot",
            "chart-no-axes-combined",
            "grid-3x3",
            "play",
            "radio",
            "gauge",
            "boxes",
        ):
            self.assertIn(f'data-lucide="{icon_name}"', self.html)

    def test_recording_is_not_nested_inside_input(self) -> None:
        self.assertIn('recording: ".diagnostic-capture-card"', self.js)
        acquisition_start = self.html.index('data-diagnostic-group="acquisition"')
        recording_start = self.html.index('data-diagnostic-group="recording"')
        acquisition_close = self.html.index("</details>", self.html.index('class="diagnostic-subsection diagnostic-source-adapter"', acquisition_start))
        acquisition_close = self.html.index("</details>", acquisition_close + 1)
        self.assertLess(acquisition_close, recording_start)

    def test_demo_workspace_collapses_the_otherwise_empty_card_stack(self) -> None:
        self.assertIn(
            'cardStack.classList.toggle("diagnostic-stack-empty", nextWorkspace === "demo")',
            self.js,
        )
        self.assertRegex(
            self.css,
            re.compile(
                r"\.diagnostics-mode \.diagnostic-card-stack\.diagnostic-stack-empty\s*\{[^}]*"
                r"display:\s*none\s*!important",
                re.S,
            ),
        )

    def test_surface_status_and_overview_are_one_primary_card(self) -> None:
        self.assertNotIn("diagnostic-surface-state\"", self.html)
        card_start = self.html.index('class="prediction-card diagnostics-only diagnostic-card diagnostic-metrics-card"')
        card_end = self.html.index('</details>', self.html.index('class="diagnostic-subsection diagnostic-advanced-surface"', card_start))
        primary_card = self.html[card_start:card_end]
        self.assertIn('id="diagnosticSurfaceResponseLevel"', primary_card)
        self.assertIn('id="surfaceQualityStatus"', primary_card)
        self.assertIn('id="surfacePeakDiagnostic"', primary_card)
        self.assertIn('id="eventInterpretation"', primary_card)

    def test_low_priority_surface_fields_remain_available_under_advanced(self) -> None:
        advanced_start = self.html.index('class="diagnostic-subsection diagnostic-advanced-surface"')
        advanced_end = self.html.index('</details>', advanced_start)
        advanced = self.html[advanced_start:advanced_end]
        for field_id in (
            "surfaceResponseType",
            "surfaceCouplingPath",
            "surfaceForceDecoupled",
            "surfaceLocalMapStatus",
            "responseLevel",
            "liveFreshness",
            "surfaceText",
            "responseText",
        ):
            self.assertIn(f'id="{field_id}"', advanced)

    def test_surface_workspace_is_one_task_card_with_deferred_supporting_evidence(self) -> None:
        self.assertEqual(self.html.count('data-diagnostic-group="surface"'), 1)
        self.assertNotIn("diagnostic-array-card", self.html)
        self.assertNotIn("diagnostic-coupling-card", self.html)
        card_start = self.html.index('class="prediction-card diagnostics-only diagnostic-card diagnostic-metrics-card"')
        signal_start = self.html.index('data-diagnostic-group="signal"', card_start)
        surface_card = self.html[card_start:signal_start]
        self.assertIn('class="diagnostic-subsection diagnostic-array-status"', surface_card)
        self.assertIn('class="diagnostic-subsection diagnostic-coupling-raw"', surface_card)
        for field_id in (
            "realActiveChannels",
            "simulatedChannels",
            "disabledChannels",
            "arrayFrameSyncMirror",
            "qaWarningCount",
            "rawCoupledViewButton",
            "channelGrid",
            "arrayRawTable",
            "arrayJsonPreview",
        ):
            self.assertIn(f'id="{field_id}"', surface_card)

    def test_workspace_switching_keeps_one_default_card_per_task(self) -> None:
        self.assertIn('surface: ".diagnostic-metrics-card"', self.js)
        self.assertIn("defaultCard.open = true", self.js)
        self.assertIn("if (other !== card && other.open) other.open = false", self.js)

    def test_acquisition_health_and_adapter_share_one_task_card(self) -> None:
        self.assertEqual(self.html.count('data-diagnostic-group="acquisition"'), 1)
        self.assertNotIn("diagnostic-bridge-card", self.html)
        card_start = self.html.index('class="prediction-card diagnostics-only diagnostic-card diagnostic-frame-card"')
        card_end = self.html.index('</details>', self.html.index('class="diagnostic-subsection diagnostic-source-adapter"', card_start))
        acquisition_card = self.html[card_start:card_end]
        self.assertIn("ACQUISITION DIAGNOSTICS", acquisition_card)
        for field_id in (
            "diagnosticSourceState",
            "diagnosticStreamState",
            "diagnosticFrameState",
            "diagnosticBaselineState",
            "diagnosticFrameSyncState",
            "diagnosticSpectrumState",
            "diagnosticQaState",
            "diagnosticAxisState",
            "senseProcess",
            "frameAge",
            "watchStatus",
            "latestExport",
            "watchIngestCount",
            "channelsSeen",
            "watchError",
        ):
            self.assertIn(f'id="{field_id}"', acquisition_card)

    def test_acquisition_primary_health_precedes_advanced_evidence(self) -> None:
        card_start = self.html.index('class="prediction-card diagnostics-only diagnostic-card diagnostic-frame-card"')
        headline = self.html.index("diagnostic-acquisition-headline", card_start)
        kpis = self.html.index("diagnostic-acquisition-kpi-grid", card_start)
        advanced = self.html.index("diagnostic-advanced-frame", card_start)
        adapter = self.html.index("diagnostic-source-adapter", card_start)
        self.assertLess(headline, kpis)
        self.assertLess(kpis, advanced)
        self.assertLess(advanced, adapter)


if __name__ == "__main__":
    unittest.main()
