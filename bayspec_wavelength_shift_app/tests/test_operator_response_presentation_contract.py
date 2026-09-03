from __future__ import annotations

import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


class OperatorResponsePresentationContractTests(unittest.TestCase):
    def test_operator_visuals_use_one_current_runtime_contract(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        canonical = app_js.split(
            "function normalizeCanonicalVisualizationFrame", 1
        )[1].split("function normalizeGlobalSpectrumFrame", 1)[0]
        active_prediction = app_js.split("function activeModelPrediction", 1)[1].split(
            "function opticalForcePresentation", 1
        )[0]

        self.assertIn(
            'OPERATOR_VISUALIZATION_CONTRACT_VERSION = "touch_operator_visualization_v2"',
            app_js,
        )
        self.assertIn("appendCurrentRuntimeTrace(rawRecord, contract)", canonical)
        self.assertIn("operator_visualization_frame: contract", canonical)
        self.assertIn("contract?.force?.display_n", canonical)
        self.assertIn("contract?.surface?.surface_grid", canonical)
        self.assertIn(
            "contract?.surface?.inferred_contact_probability_grid",
            canonical,
        )
        self.assertIn(
            "contract?.surface?.observed_coupled_spectral_response",
            canonical,
        )
        self.assertIn("inferred_contact_probability: inferredProbability", canonical)
        self.assertIn(
            "observed_coupled_response_ratio: observedResponseRatio",
            canonical,
        )
        self.assertIn(
            "contract?.surface?.raw_inferred_contact_probabilities",
            canonical,
        )
        self.assertIn(
            "contract?.surface?.smoothed_inferred_contact_probabilities",
            canonical,
        )
        self.assertIn("force_frame_id: sync.force_frame_id ?? frameId", canonical)
        self.assertNotIn("trained_static_spectral_prediction", canonical)
        self.assertNotIn("dynamic_temporal_shadow", canonical)
        self.assertIn("operatorVisualizationContract(record, arrayFrame)", active_prediction)
        self.assertIn("return contract?.prediction || null", active_prediction)
        self.assertNotIn("all_source_beta_model", app_js)
        self.assertNotIn("trainedStaticModelSurface", app_js)

    def test_operator_summary_follows_the_same_inferred_dominant_as_the_map(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        render = app_js.split("function updateUI(inputFrame)", 1)[1].split(
            "function normalizeCanonicalVisualizationFrame", 1
        )[0]

        self.assertIn(
            "surfaceMetrics.inferred_dominant_channel || surfaceMetrics.dominant_channel || record?.model_position_id",
            render,
        )
        self.assertNotIn(
            "(currentModelDisplay ? record?.model_position_id : null) || surfaceMetrics.dominant_channel",
            render,
        )

    def test_diagnostics_channel_grid_separates_observed_and_inferred_values(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        channel_grid = app_js.split("function updateChannelGrid", 1)[1].split(
            "function updateOperatorFootprint", 1
        )[0]

        self.assertIn("item.observed_coupled_response_ratio", channel_grid)
        self.assertIn("item.inferred_contact_probability", channel_grid)
        self.assertIn("observed coupled", channel_grid)
        self.assertIn("inferred probability", channel_grid)

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

    def test_current_runtime_force_gauge_uses_canonical_optical_estimate(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        styles = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

        helper = app_js.split("function opticalForcePresentation", 1)[1].split(
            "function activeModelDisplayName", 1
        )[0]
        gauge = app_js.split("const forcePresentation = opticalForcePresentation", 1)[1].split(
            'let note = "raw coupled response";', 1
        )[0]

        self.assertIn("operatorVisualizationContract(record, arrayFrame, runtimeFrame)", helper)
        self.assertIn("contract.force || {}", helper)
        self.assertIn('"/api/health"', app_js)
        self.assertIn("loadRuntimeCapabilities()", app_js)
        self.assertIn("force.display_n", helper)
        self.assertIn("OPTICAL_FORCE_CALIBRATED_MAX_N = 5", app_js)
        self.assertIn("OPTICAL_FORCE_MIN_DISPLAY_MAX_N = 5", app_js)
        self.assertIn("updateOpticalForceDisplayMaximum", gauge)
        self.assertIn("above_calibrated_range", gauge)
        self.assertNotIn("OPTICAL_FORCE_DISPLAY_MAX_N", app_js)
        self.assertIn('"Estimated Force"', gauge)
        self.assertIn("estimatedForceN.toFixed(2)", gauge)
        self.assertIn('marker.setAttribute("role", "meter")', gauge)
        self.assertIn('marker.setAttribute("aria-valuenow"', gauge)
        self.assertIn("Estimated Force", html)
        self.assertIn(".operator-band-card.force-estimate-mode", styles)

    def test_low_force_visual_drive_is_not_reblocked_by_operator_ui(self) -> None:
        app_js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        canonical = app_js.split("function normalizeCanonicalVisualizationFrame", 1)[1].split(
            "function normalizeGlobalSpectrumFrame", 1
        )[0]
        presentation = app_js.split("function surfaceContactPresentation", 1)[1].split(
            "function clampArrayCoord", 1
        )[0]
        force = app_js.split("function opticalForcePresentation", 1)[1].split(
            "function activeModelDisplayName", 1
        )[0]

        self.assertIn("const responseAllowed = contract.response_allowed === true", canonical)
        self.assertIn("contract?.surface?.surface_grid", canonical)
        self.assertIn("(twin.visual_active ?? twin.active) === true", presentation)
        self.assertIn("force.display_n", force)
        self.assertIn("valueN,", force)

    def test_px6d_reference_is_hidden_from_operator_but_kept_in_diagnostics(self) -> None:
        html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        styles = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")

        self.assertIn(
            'class="hud-card operator-current-hud" aria-label="Force sensor" '
            'aria-hidden="true" hidden',
            html,
        )
        self.assertIn(
            'class="px6d-reference-row operator-force-readout" aria-label="Force sensor" '
            'aria-hidden="true" hidden',
            html,
        )
        self.assertIn('id="diagnosticPx6dFz"', html)
        self.assertIn('id="diagnosticPx6dTareButton"', html)
        self.assertIn("html body .operator-current-hud[hidden]", styles)
        self.assertIn("html body .operator-force-readout[hidden]", styles)

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
