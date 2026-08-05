from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "bayspec_wavelength_shift_app" / "frontend"
BACKEND_MAIN = PROJECT_ROOT / "bayspec_wavelength_shift_app" / "backend" / "main.py"


class MeasurementDiagnosticsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")
        cls.javascript = (FRONTEND_ROOT / "app.js").read_text(encoding="utf-8")
        cls.styles = (FRONTEND_ROOT / "styles.css").read_text(encoding="utf-8")
        cls.backend = BACKEND_MAIN.read_text(encoding="utf-8")

    def test_measurement_workspace_is_diagnostics_only(self) -> None:
        diagnostic_start = self.html.index(
            '<nav class="diagnostic-tabs diagnostics-only"'
        )
        operator_markup = self.html[:diagnostic_start]
        self.assertNotIn("measurementComparisonCanvas", operator_markup)
        self.assertNotIn('data-diagnostic-tab="measurement"', operator_markup)
        self.assertIn('data-diagnostic-tab="measurement"', self.html)
        self.assertIn(
            'class="prediction-card diagnostics-only diagnostic-card diagnostic-measurement-card"',
            self.html,
        )
        self.assertIn('data-diagnostic-group="measurement"', self.html)

    def test_measurement_controls_and_metrics_are_present(self) -> None:
        for element_id in (
            "measurementRootInput",
            "measurementSessionSelect",
            "measurementEstimateSource",
            "measurementRefreshButton",
            "measurementAnalyzeButton",
            "measurementComparisonCanvas",
            "measurementEvidence",
            "measurementValidity",
            "measurementBaseline",
            "measurementPairedSamples",
            "measurementCycleCount",
            "measurementMae",
            "measurementRmse",
            "measurementCorrelation",
            "measurementAmplitudeSlope",
            "measurementLag",
            "measurementAcquisitionRate",
            "measurementInferenceLatency",
            "measurementRecovery",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn(".diagnostic-measurement-card", self.styles)

    def test_analysis_is_manual_and_not_part_of_boot(self) -> None:
        self.assertIn(
            'measurementRefreshButton?.addEventListener("click", refreshMeasurementSessions)',
            self.javascript,
        )
        self.assertIn(
            'measurementAnalyzeButton?.addEventListener("click", analyzeSelectedMeasurementSession)',
            self.javascript,
        )
        boot_source = self.javascript[self.javascript.index("async function boot()") :]
        self.assertNotIn("refreshMeasurementSessions()", boot_source)
        self.assertNotIn("analyzeSelectedMeasurementSession()", boot_source)

    def test_frontend_uses_offline_measurement_endpoints(self) -> None:
        self.assertIn('"/api/measurement/analyze"', self.javascript)
        self.assertIn("/api/measurement/sessions", self.javascript)
        self.assertIn('nextWorkspace === "measurement"', self.javascript)
        self.assertIn(
            "drawMeasurementComparison(state.measurementResult?.trace || [])",
            self.javascript,
        )
        self.assertIn("estimate_source: estimateSource", self.javascript)
        self.assertIn("analysis_estimated_fz_n", self.javascript)
        self.assertNotIn("measurementRecordedLegendItem", self.javascript)

    def test_backend_contract_keeps_analysis_off_capture_path(self) -> None:
        self.assertIn('@app.get("/api/measurement/sessions")', self.backend)
        self.assertIn('@app.post("/api/measurement/analyze")', self.backend)
        self.assertIn("await asyncio.to_thread(", self.backend)
        self.assertIn('"status": "recording_still_active"', self.backend)
        self.assertIn("analyze_measurement_session", self.backend)
        self.assertIn("_measurement_estimate_evidence", self.backend)
        self.assertIn("resolve_measurement_estimate_evidence", self.backend)
        self.assertIn("_downsample_measurement_trace", self.backend)


if __name__ == "__main__":
    unittest.main()
