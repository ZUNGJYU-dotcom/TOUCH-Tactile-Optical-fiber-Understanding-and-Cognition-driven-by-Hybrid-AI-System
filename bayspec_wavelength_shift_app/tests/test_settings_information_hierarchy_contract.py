from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
HTML = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
JS = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
CSS = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")


class SettingsInformationHierarchyContractTests(unittest.TestCase):
    def test_settings_keeps_only_surface_presentation_controls(self) -> None:
        settings = HTML.split('id="settingsPanel"', 1)[1].split("</aside>", 1)[0]
        self.assertIn('id="settingsThumbHolderButton"', settings)
        self.assertIn('id="settingsSurfaceOnlyButton"', settings)
        self.assertIn('id="settingsResetCameraButton"', settings)
        self.assertNotIn('id="settingsOperatorModeButton"', settings)
        self.assertNotIn('id="settingsDiagnosticsModeButton"', settings)
        self.assertNotIn('id="settingsSpectrumButton"', settings)
        self.assertNotIn('id="settingsTemporalValidationButton"', settings)
        self.assertNotIn('id="settingsStaticFallbackButton"', settings)

    def test_recognition_runtime_lives_under_acquisition_diagnostics(self) -> None:
        acquisition = HTML.split('data-diagnostic-group="acquisition"', 1)[1].split(
            'data-diagnostic-group="geometry"', 1
        )[0]
        self.assertIn("Recognition runtime", acquisition)
        self.assertIn('id="recognitionModeStatus"', acquisition)
        self.assertNotIn('id="settingsTemporalValidationButton"', acquisition)
        self.assertNotIn('id="settingsStaticFallbackButton"', acquisition)
        self.assertNotIn('id="legacyRecognitionRuntimeControls"', acquisition)
        self.assertLess(
            acquisition.index("Recognition runtime"),
            acquisition.index("Source adapter"),
        )

    def test_diagnostics_exposes_one_current_runtime_without_model_switches(self) -> None:
        self.assertIn('"Current optical model"', JS)
        self.assertIn('runtime?.display_name || "Current optical model"', JS)
        self.assertNotIn("legacyRecognitionRuntimeControls", JS)
        self.assertNotIn("settingsTemporalValidationButton", JS)
        self.assertNotIn("settingsStaticFallbackButton", JS)
        self.assertNotIn("temporalValidationMode", JS)

    def test_removed_workspace_controls_leave_no_dead_js_bindings(self) -> None:
        self.assertNotIn("settingsOperatorModeButton", JS)
        self.assertNotIn("settingsDiagnosticsModeButton", JS)
        self.assertNotIn("settingsSpectrumButton", JS)
        self.assertNotIn("legacyRecognitionRuntimeControls", JS)

    def test_compact_settings_and_diagnostic_runtime_styles_exist(self) -> None:
        self.assertIn("width: min(320px, calc(100vw - 24px));", CSS)
        self.assertIn(".diagnostics-mode .diagnostic-runtime-content", CSS)
        self.assertIn(".diagnostics-mode .diagnostic-runtime-status", CSS)
        self.assertNotIn(".diagnostics-mode .diagnostic-runtime-buttons", CSS)

    def test_settings_focus_moves_inside_and_returns_to_its_opener(self) -> None:
        self.assertIn("let settingsPanelOpener = null", JS)
        settings_function = JS.split("function setSettingsPanelOpen", 1)[1].split(
            "spectrumToggleButton?.addEventListener", 1
        )[0]
        self.assertIn("settingsPanelOpener = document.activeElement", settings_function)
        self.assertIn("settingsCloseButton?.focus()", settings_function)
        self.assertIn("restoreFocus && opener?.isConnected", settings_function)
        self.assertIn("setSettingsPanelOpen(false, false)", JS)

    def test_spectrum_empty_state_is_short_and_non_redundant(self) -> None:
        drawer = HTML.split('id="spectrumDrawer"', 1)[1].split("</aside>", 1)[0]
        self.assertIn("Waiting for frame", drawer)
        self.assertIn("NO FRAME", drawer)
        self.assertIn("Start acquisition or load a spectrum frame.", drawer)
        self.assertNotIn("Start Live, Watch, ingest", drawer)
        self.assertIn('? "NO FRAME"', JS)
        self.assertIn('? "Waiting for frame"', JS)


if __name__ == "__main__":
    unittest.main()
