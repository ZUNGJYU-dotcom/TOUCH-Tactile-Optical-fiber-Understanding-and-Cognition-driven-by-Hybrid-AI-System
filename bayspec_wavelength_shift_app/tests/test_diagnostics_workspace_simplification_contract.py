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

    def test_six_workspaces_use_one_stable_navigation_row(self) -> None:
        self.assertEqual(self.html.count("data-diagnostic-tab="), 6)
        self.assertRegex(
            self.css,
            re.compile(
                r"\.diagnostics-mode \.diagnostic-tabs\s*\{[^}]*"
                r"grid-template-columns:\s*repeat\(6, minmax\(0, 1fr\)\)",
                re.S,
            ),
        )

    def test_workspace_tabs_use_compact_labels_with_accessible_full_names(self) -> None:
        for visible_label in ("Signal", "Surface", "Demo", "Input", "Force", "Geometry"):
            self.assertIn(f"<span>{visible_label}</span>", self.html)
        for accessible_name in (
            "Signal diagnostics",
            "Surface diagnostics",
            "Demo controls",
            "Acquisition diagnostics",
            "Mechanical reference and synchronized recording",
            "Geometry diagnostics",
        ):
            self.assertIn(f'aria-label="{accessible_name}"', self.html)
        for icon_name in ("activity", "grid-3x3", "play", "radio", "gauge", "boxes"):
            self.assertIn(f'data-lucide="{icon_name}"', self.html)

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
