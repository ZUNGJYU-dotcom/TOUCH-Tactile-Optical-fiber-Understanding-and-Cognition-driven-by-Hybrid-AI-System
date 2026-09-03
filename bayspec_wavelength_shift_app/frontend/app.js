import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadRobotNanoHandModel, loadThumbHolderModel } from "./model_loader.js?v=five-finger-sensor-array-v2";

const channelSelect = document.getElementById("channelSelect");
const inputSourceSelect = document.getElementById("inputSourceSelect");
const liveTwinButton = document.getElementById("liveTwinButton");
const exportWatchButton = document.getElementById("exportWatchButton");
const baselineButton = document.getElementById("baselineButton");
const ingestExportButton = document.getElementById("ingestExportButton");
const pauseButton = document.getElementById("pauseButton");
const resetButton = document.getElementById("resetButton");
const traceCanvas = document.getElementById("traceCanvas");
const opticalPreviewCanvas = document.getElementById("opticalPreviewCanvas");
const spectrumCanvas = document.getElementById("spectrumCanvas");
const selectedSpectrumCanvas = document.getElementById("selectedSpectrumCanvas");
const heatmapCanvas = document.getElementById("heatmapCanvas");
const threeMount = document.getElementById("threeMount");
const channelGrid = document.getElementById("channelGrid");
const demoStatusChip = document.getElementById("demoStatusChip");
const demoSingleButton = document.getElementById("demoSingleButton");
const demoAutoButton = document.getElementById("demoAutoButton");
const demoResetButton = document.getElementById("demoResetButton");
const demoSpeedControl = document.getElementById("demoSpeedControl");
const demoSpeedValue = document.getElementById("demoSpeedValue");
const demoStepButtons = Array.from(document.querySelectorAll(".demo-step"));
const arrayDemoStepButtons = Array.from(document.querySelectorAll(".array-demo-step"));
const nodeDebugButton = document.getElementById("nodeDebugButton");
const operatorModeButton = document.getElementById("operatorModeButton");
const diagnosticsModeButton = document.getElementById("diagnosticsModeButton");
const physicalProxyModeButton = document.getElementById("physicalProxyModeButton");
const responseTerrainModeButton = document.getElementById("responseTerrainModeButton");
const wholeHandModeButton = document.getElementById("wholeHandModeButton");
const thumbHolderModeButton = document.getElementById("thumbHolderModeButton");
const surfaceOnlyModeButton = document.getElementById("surfaceOnlyModeButton");
const fingerFocusControl = document.getElementById("fingerFocusControl");
const fingerFocusSelect = document.getElementById("fingerFocusSelect");
const previousFingerButton = document.getElementById("previousFingerButton");
const nextFingerButton = document.getElementById("nextFingerButton");
const surfaceFullscreenButton = document.getElementById("surfaceFullscreenButton");
const thumbAlignmentSaveButton = document.getElementById("thumbAlignmentSaveButton");
const thumbAlignmentResetButton = document.getElementById("thumbAlignmentResetButton");
const thumbShowSlotButton = document.getElementById("thumbShowSlotButton");
const thumbHideMeshButton = document.getElementById("thumbHideMeshButton");
const thumbWireframeButton = document.getElementById("thumbWireframeButton");
const rawCoupledViewButton = document.getElementById("rawCoupledViewButton");
const idealIndependentViewButton = document.getElementById("idealIndependentViewButton");
const couplingCompensatedViewButton = document.getElementById("couplingCompensatedViewButton");
const layoutCheckButton = document.getElementById("layoutCheckButton");
const layoutCheckOverlay = document.getElementById("layoutCheckOverlay");
const appShell = document.querySelector(".app-shell");
const dashboard = document.querySelector(".dashboard");
const diagnosticsPanelResizer = document.getElementById("diagnosticsPanelResizer");
const arrayRawTable = document.getElementById("arrayRawTable");
const arrayJsonPreview = document.getElementById("arrayJsonPreview");
const demoMenuButton = document.getElementById("demoMenuButton");
const spectrumToggleButton = document.getElementById("spectrumToggleButton");
const spectrumCloseButton = document.getElementById("spectrumCloseButton");
const spectrumDrawer = document.getElementById("spectrumDrawer");
const settingsButton = document.getElementById("settingsButton");
const settingsPanel = document.getElementById("settingsPanel");
const settingsCloseButton = document.getElementById("settingsCloseButton");
const settingsWholeHandButton = document.getElementById("settingsWholeHandButton");
const settingsThumbHolderButton = document.getElementById("settingsThumbHolderButton");
const settingsSurfaceOnlyButton = document.getElementById("settingsSurfaceOnlyButton");
const settingsResetCameraButton = document.getElementById("settingsResetCameraButton");
const spectrumSensorModeSelect = document.getElementById("spectrumSensorModeSelect");
const spectrumIntegrationInput = document.getElementById("spectrumIntegrationInput");
const spectrumOverlayToggle = document.getElementById("spectrumOverlayToggle");
const spectrumBackgroundToggle = document.getElementById("spectrumBackgroundToggle");
const spectrumBaselineToggle = document.getElementById("spectrumBaselineToggle");
const spectrumSmoothingToggle = document.getElementById("spectrumSmoothingToggle");
const spectrumNormalizationToggle = document.getElementById("spectrumNormalizationToggle");
const spectrumCaptureBackgroundButton = document.getElementById("spectrumCaptureBackgroundButton");
const spectrumClearBackgroundButton = document.getElementById("spectrumClearBackgroundButton");
const spectrumProcessingStatus = document.getElementById("spectrumProcessingStatus");
const spectrumBackgroundStatus = document.getElementById("spectrumBackgroundStatus");
const operatorDiagnosticsButton = document.getElementById("operatorDiagnosticsButton");
const demoModule = document.querySelector(".demo-module");
const opticalSummaryCard = document.querySelector(".summary-hud");
const operatorCurrentHud = document.querySelector(".operator-current-hud");
const operatorSummaryCardNode = document.querySelector(".operator-summary-card");
const operatorForceReadout = document.querySelector(".operator-force-readout");
const commandFeedback = document.getElementById("commandFeedback");
const commandFeedbackText = document.getElementById("commandFeedbackText");
const operatorAlert = document.getElementById("operatorAlert");
const operatorAlertSeverity = document.getElementById("operatorAlertSeverity");
const operatorAlertMessage = document.getElementById("operatorAlertMessage");
const operatorAlertDiagnosticsButton = document.getElementById("operatorAlertDiagnosticsButton");
const px6dTareButton = document.getElementById("px6dTareButton");
const diagnosticPx6dTareButton = document.getElementById("diagnosticPx6dTareButton");
const px6dCaptureTareButton = document.getElementById("px6dCaptureTareButton");
const px6dCapturePosition = document.getElementById("px6dCapturePosition");
const px6dCapturePositionButtons = Array.from(document.querySelectorAll("[data-capture-position]"));
const px6dCaptureTrial = document.getElementById("px6dCaptureTrial");
const px6dCaptureNextTrialButton = document.getElementById("px6dCaptureNextTrialButton");
const px6dCaptureNote = document.getElementById("px6dCaptureNote");
const px6dCaptureSpectrum = document.getElementById("px6dCaptureSpectrum");
const px6dCaptureResponse = document.getElementById("px6dCaptureResponse");
const px6dCaptureForce = document.getElementById("px6dCaptureForce");
const px6dCaptureOutputRoot = document.getElementById("px6dCaptureOutputRoot");
const px6dCaptureBrowseButton = document.getElementById("px6dCaptureBrowseButton");
const px6dCaptureStartButton = document.getElementById("px6dCaptureStartButton");
const px6dCaptureStopButton = document.getElementById("px6dCaptureStopButton");
const measurementRootInput = document.getElementById("measurementRootInput");
const measurementSessionSelect = document.getElementById("measurementSessionSelect");
const measurementEstimateSource = document.getElementById("measurementEstimateSource");
const measurementRefreshButton = document.getElementById("measurementRefreshButton");
const measurementAnalyzeButton = document.getElementById("measurementAnalyzeButton");
const measurementComparisonCanvas = document.getElementById("measurementComparisonCanvas");
const diagnosticAccordionCards = Array.from(
  document.querySelectorAll(".right-panel details.diagnostic-card, .right-panel > details.demo-module")
);
const diagnosticTabButtons = Array.from(document.querySelectorAll("[data-diagnostic-tab]"));
const diagnosticGroupedCards = Array.from(document.querySelectorAll("[data-diagnostic-group]"));
const desktopTitlebar = document.getElementById("desktopTitlebar");
const desktopMinimizeButton = document.getElementById("desktopMinimizeButton");
const desktopMaximizeButton = document.getElementById("desktopMaximizeButton");
const desktopCloseButton = document.getElementById("desktopCloseButton");
let spectrumDrawerOpener = null;
let settingsPanelOpener = null;
let desktopWindowCommandFailure = null;
let operatorDiagnosticsAttentionFromQa = false;

function activateDesktopChrome() {
  document.body.classList.add("pywebview-desktop");
}

function refreshDiagnosticsAttention() {
  const attentionRequired =
    operatorDiagnosticsAttentionFromQa || Boolean(desktopWindowCommandFailure);
  operatorDiagnosticsButton?.classList.toggle("attention-required", attentionRequired);
  operatorDiagnosticsButton?.setAttribute(
    "aria-label",
    attentionRequired ? "Diagnostics, attention required" : "Diagnostics"
  );
  if (operatorDiagnosticsButton) {
    operatorDiagnosticsButton.dataset.tooltip = attentionRequired
      ? "Diagnostics - attention"
      : "Diagnostics";
  }
}

function setDesktopWindowCommandFailure(commandName, error = null) {
  desktopWindowCommandFailure = commandName
    ? {
        command: commandName,
        reason: String(error?.message || error?.status || error || "desktop command failed"),
        timestamp: Date.now(),
      }
    : null;
  document.body.classList.toggle(
    "desktop-window-command-error",
    Boolean(desktopWindowCommandFailure)
  );
  refreshDiagnosticsAttention();
}

async function invokeDesktopWindowCommand(commandName) {
  const desktopApi = window.pywebview?.api;
  if (!desktopApi || typeof desktopApi[commandName] !== "function") {
    if (document.body.classList.contains("pywebview-desktop")) {
      setDesktopWindowCommandFailure(commandName, "desktop API unavailable");
    }
    return null;
  }
  try {
    const result = await desktopApi[commandName]();
    if (result?.ok === false) {
      setDesktopWindowCommandFailure(commandName, result.status);
    } else {
      setDesktopWindowCommandFailure(null);
    }
    return result;
  } catch (error) {
    console.error(`[desktop-window:${commandName}]`, error);
    setDesktopWindowCommandFailure(commandName, error);
    return null;
  }
}

window.addEventListener("pywebviewready", activateDesktopChrome);

desktopMinimizeButton?.addEventListener("click", () => {
  // The native shell animates a cached window snapshot, so WebGL does not
  // reflow or repaint while the complete window travels to the taskbar.
  void invokeDesktopWindowCommand("minimize_window");
});

desktopMaximizeButton?.addEventListener("click", async () => {
  const result = await invokeDesktopWindowCommand("toggle_maximize_window");
  if (!result?.ok) return;
  const maximized = Boolean(result.maximized);
  desktopMaximizeButton.classList.toggle("is-maximized", maximized);
  desktopMaximizeButton.setAttribute("aria-label", maximized ? "Restore window" : "Maximize window");
  desktopMaximizeButton.title = maximized ? "Restore" : "Maximize";
});

desktopTitlebar?.addEventListener("dblclick", (event) => {
  if (event.target.closest("button")) return;
  desktopMaximizeButton?.click();
});

desktopCloseButton?.addEventListener("click", () => {
  void invokeDesktopWindowCommand("close_window");
});

function refreshLucideIcons() {
  if (!window.lucide?.createIcons) return;
  window.lucide.createIcons({
    attrs: {
      "aria-hidden": "true",
      focusable: "false",
      "stroke-width": 1.8,
    },
  });
  document.documentElement.classList.add("lucide-ready");
}

function setCommandButtonLabel(button, label) {
  if (!button) return;
  const labelNode = button.querySelector(".command-label");
  if (labelNode) labelNode.textContent = label;
  else button.textContent = label;
}

function setCommandButtonDescription(button, label, tooltip = label) {
  if (!button) return;
  setCommandButtonLabel(button, label);
  button.setAttribute("aria-label", label);
  button.dataset.tooltip = tooltip;
}

refreshLucideIcons();

const DEMO_BASELINE = 42000;
const DEMO_TARGET_WAVELENGTH_NM = 1546.7124;
const DEMO_BASELINE_WAVELENGTH_NM = 1546.89;
const WAVELENGTH_SHIFT_FULL_SCALE_PM = 500;
const RESPONSE_BAND_THRESHOLDS = {
  noContactMax: 0.25,
  smallMax: 0.80,
  moderateMax: 0.90,
};
const COUPLED_CHANNEL_VISIBILITY_THRESHOLD = 0.05;

function applyResponseBandThresholds(raw = {}) {
  const noContactMax = Number(raw?.no_contact_max);
  const smallMax = Number(raw?.light_max);
  const moderateMax = Number(raw?.normal_max);
  if (
    Number.isFinite(noContactMax) &&
    Number.isFinite(smallMax) &&
    Number.isFinite(moderateMax) &&
    0 < noContactMax &&
    noContactMax < smallMax &&
    smallMax < moderateMax &&
    moderateMax < 1
  ) {
    RESPONSE_BAND_THRESHOLDS.noContactMax = noContactMax;
    RESPONSE_BAND_THRESHOLDS.smallMax = smallMax;
    RESPONSE_BAND_THRESHOLDS.moderateMax = moderateMax;
  }
}
const DEMO_ARRAY_STEP_INTERVAL_MS = 100;
const DEMO_ARRAY_LOOP_INTERVAL_MS = 5000;
const DEMO_FRAME_SCHEDULER_INTERVAL_MS = 25;
const DEMO_PLAYBACK_RATE_STORAGE_KEY = "touch-response-playback-rate";
const DIAGNOSTICS_PANEL_WIDTH_STORAGE_KEY = "touch-diagnostics-panel-width";
const DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX = 420;
const DIAGNOSTICS_PANEL_MIN_WIDTH_PX = 280;
const DIAGNOSTICS_PANEL_MAX_WIDTH_PX = 680;
// The Surface workspace contains both the 3D proxy and the 2D map. Keeping a
// real minimum here prevents a widened diagnostics rail from collapsing those
// two views into an overlapping layout.
const DIAGNOSTICS_CENTER_MIN_WIDTH_PX = 520;
const DIAGNOSTICS_CENTER_COMPACT_MIN_WIDTH_PX = 420;
const DIAGNOSTICS_COMPACT_BREAKPOINT_PX = 1100;
const DEMO_PLAYBACK_RATE_MIN = 0.1;
const DEMO_PLAYBACK_RATE_MAX = 2.0;
// Apply a newly acquired contact frame quickly, then keep the established
// slower release constants below. This reduces visual onset latency without
// turning release residuals into an abrupt or unstable animation.
const THREE_ATTENUATION_EASING = 28.0;
const THREE_DEFORMATION_EASING = 26.0;
const THREE_ATTENUATION_RELEASE_EASING = 16.0;
const THREE_DEFORMATION_RELEASE_EASING = 14.0;
const THREE_SURFACE_RELEASE_EASING = 18.0;
const THREE_SPATIAL_EASING = 24.0;
const THREE_SETTLE_EPSILON = 0.00035;
const THREE_MAX_DEVICE_PIXEL_RATIO = 1.25;
const THREE_GEOMETRY_UPDATE_INTERVAL_MS = 33;
const THREE_NORMAL_UPDATE_INTERVAL_MS = 100;
const CHART_UPDATE_INTERVAL_MS = 50;
const THREE_SLOT_BODY_THICKNESS = 0.48;
// The contact face carries the visible indentation. The elastomer sidewall
// follows continuously, while the bottom only moves slightly inside the slot.
const THREE_SLOT_BOTTOM_DEFORMATION_FOLLOW = 0.02;
// Preserve a visible compressed core at maximum response. Without this limit
// the contact face can pass through the bottom skin and appear hollow.
const THREE_SLOT_MIN_COMPRESSED_THICKNESS = 0.12;
const THREE_SLOT_MAX_LOCAL_DEPRESSION =
  (THREE_SLOT_BODY_THICKNESS - THREE_SLOT_MIN_COMPRESSED_THICKNESS) /
  (1 - THREE_SLOT_BOTTOM_DEFORMATION_FOLLOW);
const THREE_SLOT_MAX_LOCAL_BODY_Y =
  THREE_SLOT_BODY_THICKNESS +
  THREE_SLOT_MAX_LOCAL_DEPRESSION * THREE_SLOT_BOTTOM_DEFORMATION_FOLLOW;
const SURFACE_GRID_VISUAL_GAMMA = 0.68;
const DEMO_SURFACE_VISUAL_PEAK_FLOORS = {
  off_center_fingertip_contact: 0.52,
  vertical_slide_p11_p12_p13: 0.56,
  horizontal_slide_p11_p21_p31: 0.56,
  diagonal_slide_p11_p22_p33: 0.56,
  broad_fingertip_contact: 0.58,
  tap: 0.58,
  release: 0.34,
};
const CHART_EASING = 8.5;
const CHART_SETTLE_COUNTS = 0.55;
const TRACE_WINDOW_POINTS = 120;
const DEMO_TRACE_WINDOW_POINTS = 80;
// Poll often enough that a completed SDK frame is normally observed within
// one display frame. Requests never overlap and duplicate physical frames are
// cached by the backend, so this does not repeat model inference.
const LIVE_MODEL_POLL_INTERVAL_MS = 25;
const LIVE_SOURCE_PROBE_INTERVAL_MS = 1500;
const PX6D_UI_POLL_INTERVAL_MS = 100;
const PX6D_CAPTURE_POLL_INTERVAL_MS = 500;
const ARRAY_DISPLAY_ROWS = [
  ["P11", "P21", "P31"],
  ["P12", "P22", "P32"],
  ["P13", "P23", "P33"],
];
const ARRAY_DISPLAY_ORDER = ARRAY_DISPLAY_ROWS.flat();
const FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"];
const FINGER_RIGHT_NAVIGATION_SEQUENCE = [
  "all",
  "thumb",
  "little",
  "ring",
  "middle",
  "index",
  "thumb",
  "all",
];
const FINGER_CLOSEUP_DURATION_MS = 1050;
const FINGER_OVERVIEW_DURATION_MS = 1250;
const FINGER_CLOSEUP_FOV = 40;
const FINGER_CLOSEUP_DISTANCE_SCALE = 4.0;
const WHOLE_HAND_OVERVIEW_FOV = 40;
const WHOLE_HAND_OVERVIEW_MARGIN = 0.88;
const WHOLE_HAND_OVERVIEW_VERTICAL_BIAS = 0.13;
const FINGER_CLICK_MAX_MOVEMENT_PX = 7;
const FINGER_LABELS = {
  thumb: "Thumb",
  index: "Index",
  middle: "Middle",
  ring: "Ring",
  little: "Little",
  all: "All fingers",
};
const ARRAY_CHANNEL_COORDS = {
  P11: { x: -1, y: 1 },
  P21: { x: 0, y: 1 },
  P31: { x: 1, y: 1 },
  P12: { x: -1, y: 0 },
  P22: { x: 0, y: 0 },
  P32: { x: 1, y: 0 },
  P13: { x: -1, y: -1 },
  P23: { x: 0, y: -1 },
  P33: { x: 1, y: -1 },
};
const THREE_SURFACE_ARRAY_TO_SCENE_X_SIGN = 1;
const THREE_SURFACE_ARRAY_TO_SCENE_Z_SIGN = -1;
const THREE_SURFACE_AXIS_MODE = "swap_array_xy";
const WAVELENGTH_PLAN_ORDER = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"];
const GLOBAL_RECOGNITION_SCOPE = "optical_contact_position_and_continuous_fz";
const CURRENT_RUNTIME_MODEL_SOURCE = "ordinary_fbg_same_day_joint_nine_fbg_beta_v4";
const OPERATOR_VISUALIZATION_CONTRACT_VERSION = "touch_operator_visualization_v2";
const OPTICAL_FORCE_CALIBRATED_MAX_N = 5;
const OPTICAL_FORCE_MIN_DISPLAY_MAX_N = 5;
const GLOBAL_CANDIDATE_IDS = Array.from({ length: 9 }, (_, index) => `FBG${String(index + 1).padStart(2, "0")}`);
const GLOBAL_PROXY_FULL_SCALE_PM = 75;
const GLOBAL_PROXY_VISIBLE_MIN = 0.12;
const GLOBAL_EVENT_RESIDUAL_QUANTILE = 0.2;
const GLOBAL_EVENT_DEADBAND_PM = 3;
const GLOBAL_EVENT_MIN_CONDITIONING_FRAMES = 12;
const GLOBAL_EVENT_SINGLE_PEAK_TRIGGER_PM = 18;
const GLOBAL_EVENT_PRIMARY_TRIGGER_PM = 8;
const GLOBAL_EVENT_SECONDARY_TRIGGER_PM = 3;

function currentRuntimeModelSource(frame = null, contract = null) {
  return String(
    contract?.classification_model_source ||
    contract?.model_source ||
    frame?.classification_model_source ||
    frame?.runtime_model?.classification_model_source ||
    frame?.active_spectral_model_source ||
    CURRENT_RUNTIME_MODEL_SOURCE
  );
}

function opticalForceAxisStep(maximumN) {
  const roughStep = Math.max(OPTICAL_FORCE_MIN_DISPLAY_MAX_N, Number(maximumN) || 0) / 4;
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(roughStep, 0.001)));
  const normalized = roughStep / magnitude;
  const nice = normalized <= 1
    ? 1
    : normalized <= 2
      ? 2
      : normalized <= 2.5
        ? 2.5
        : normalized <= 5
          ? 5
          : 10;
  return nice * magnitude;
}

function updateOpticalForceDisplayMaximum(valueN) {
  const currentMaximum = Math.max(
    OPTICAL_FORCE_MIN_DISPLAY_MAX_N,
    Number(state.opticalForceDisplayMaxN) || 0
  );
  const value = Number(valueN);
  if (!Number.isFinite(value) || value <= currentMaximum) return currentMaximum;
  const requiredMaximum = Math.max(OPTICAL_FORCE_MIN_DISPLAY_MAX_N, value * 1.08);
  const step = opticalForceAxisStep(requiredMaximum);
  state.opticalForceDisplayMaxN = Math.max(
    currentMaximum,
    Math.ceil(requiredMaximum / step) * step
  );
  return state.opticalForceDisplayMaxN;
}

function formatForceScaleTick(valueN, includeUnit = false) {
  const value = Math.max(0, Number(valueN) || 0);
  const decimals = value < 10 && Math.abs(value - Math.round(value)) > 1.0e-6 ? 2 : 0;
  const text = Number(value.toFixed(decimals)).toString();
  return includeUnit ? `${text} N` : text;
}
const ARRAY_SLIDE_STEPS = {
  // 5.0 s at 1.0x: light 0.5 s, release 2.0 s, hard 2.0 s,
  // then a 0.5 s smooth release so loop boundaries remain physical.
  center_press: 50,
  p21_contact: 50,
  p12_contact: 50,
  p32_contact: 50,
  off_center_fingertip_contact: 14,
  vertical_slide_p11_p12_p13: 12,
  horizontal_slide_p11_p21_p31: 12,
  diagonal_slide_p11_p22_p33: 12,
  broad_fingertip_contact: 14,
  tap: 10,
  release: 12,
};

function normalizedDemoPlaybackRate(value) {
  if (value === null || value === undefined || value === "") return 1;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 1;
  return Math.max(DEMO_PLAYBACK_RATE_MIN, Math.min(DEMO_PLAYBACK_RATE_MAX, parsed));
}

function storedDemoPlaybackRate() {
  try {
    return normalizedDemoPlaybackRate(window.localStorage.getItem(DEMO_PLAYBACK_RATE_STORAGE_KEY));
  } catch {
    return 1;
  }
}

function demoArrayStepIntervalMs() {
  return DEMO_ARRAY_STEP_INTERVAL_MS / normalizedDemoPlaybackRate(state.demoPlaybackRate);
}

function globalCandidatePeaks(record) {
  const peaks = (Array.isArray(record?.spectrum_peaks) ? record.spectrum_peaks : [])
    .filter((peak) => peak?.candidate_mapping && GLOBAL_CANDIDATE_IDS.includes(String(peak?.candidate_id || "")));
  const byId = new Map(peaks.map((peak) => [String(peak.candidate_id), peak]));
  return GLOBAL_CANDIDATE_IDS.map((candidateId) => byId.get(candidateId)).filter(Boolean);
}

function isGlobalSpectrumFrame(frame, record = frame?.latest) {
  return (
    frame?.scope === GLOBAL_RECOGNITION_SCOPE &&
    frame?.candidate_contract_version === "global_9fbg_candidate_frame_v1" &&
    frame?.physical_channel_mapping_final === false &&
    globalCandidatePeaks(record).length === GLOBAL_CANDIDATE_IDS.length
  );
}

function candidateShiftPm(peak) {
  const direct = Number(peak?.candidate_delta_wavelength_pm);
  if (Number.isFinite(direct)) return direct;
  const tracked = Number(peak?.tracked_wavelength_nm ?? peak?.peak_wavelength_nm);
  const reference = Number(
    peak?.candidate_reference_wavelength_nm ?? peak?.candidate_measured_wavelength_nm
  );
  return Number.isFinite(tracked) && Number.isFinite(reference) ? (tracked - reference) * 1000 : Number.NaN;
}

function dominantGlobalCandidate(record) {
  const peaks = globalCandidatePeaks(record).filter((peak) => peak?.valid !== false);
  const preferredCandidateId = String(record?.dominant_candidate_id || "");
  const preferred = peaks.find((peak) => peak?.candidate_id === preferredCandidateId);
  if (preferred) return preferred;
  return peaks.reduce((best, peak) => {
    if (!best) return peak;
    return Math.abs(candidateShiftPm(peak)) > Math.abs(candidateShiftPm(best)) ? peak : best;
  }, null);
}

function globalSpectralProxyValue(absShiftPm) {
  const value = Number(absShiftPm);
  if (!Number.isFinite(value) || value <= 0) return 0;
  const ratio = Math.max(0, Math.min(1, value / GLOBAL_PROXY_FULL_SCALE_PM));
  return value >= 10 ? Math.max(GLOBAL_PROXY_VISIBLE_MIN, ratio) : ratio;
}

function centeredGlobalProxySurfaceGrid(proxyValue) {
  const v = Math.max(0, Math.min(1, Number(proxyValue) || 0));
  if (v <= 0) {
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  }
  return [
    [v * 0.35, v * 0.55, v * 0.35],
    [v * 0.55, v, v * 0.55],
    [v * 0.35, v * 0.55, v * 0.35],
  ];
}

function resetRuntimeTraceHistory() {
  state.runtimeTraceRecords = [];
  state.runtimeTraceSource = null;
  state.opticalForceDisplayMaxN = OPTICAL_FORCE_MIN_DISPLAY_MAX_N;
}

function appendCurrentRuntimeTrace(rawRecord, contract) {
  const sourceName = String(rawRecord?.source || contract?.source || CURRENT_RUNTIME_MODEL_SOURCE);
  const sourceSession = contract?.source_session_id ?? rawRecord?.source_session_id ?? "";
  const source = `${sourceName}|${String(sourceSession)}`;
  if (state.runtimeTraceSource && state.runtimeTraceSource !== source) {
    resetRuntimeTraceHistory();
  }
  state.runtimeTraceSource = source;
  const frameId = contract?.frame_id ?? rawRecord?.frame_id ?? rawRecord?.spectrum_frame_id ?? null;
  const timestamp = contract?.timestamp ?? rawRecord?.timestamp ?? Date.now() / 1000;
  const displayForceN = finiteNumberOrNull(contract?.force?.display_n);
  const surfacePeak = Math.max(
    0,
    Math.min(1, finiteNumberOrNull(contract?.trace_sample?.surface_peak) ?? 0)
  );
  const record = {
    frame_id: frameId,
    timestamp,
    source,
    recognition_scope: GLOBAL_RECOGNITION_SCOPE,
    visualization_contract_version: OPERATOR_VISUALIZATION_CONTRACT_VERSION,
    trained_model_visual_response_ratio: surfacePeak,
    trained_model_response_level: String(contract?.response_state || "no_contact"),
    trained_model_position_id: contract?.position?.display_label || null,
    estimated_force_fz_n: displayForceN,
    optical_force_trace_n: displayForceN === null ? 0 : Math.max(0, displayForceN),
    trained_model_review_needed: contract?.quality?.review_needed === true,
    trace_response_semantics: "canonical_operator_display_force_n",
  };
  const previous = state.runtimeTraceRecords[state.runtimeTraceRecords.length - 1];
  if (previous && frameId !== null && String(previous.frame_id) === String(frameId)) {
    state.runtimeTraceRecords[state.runtimeTraceRecords.length - 1] = record;
  } else {
    state.runtimeTraceRecords.push(record);
    if (state.runtimeTraceRecords.length > TRACE_WINDOW_POINTS) {
      state.runtimeTraceRecords.splice(
        0,
        state.runtimeTraceRecords.length - TRACE_WINDOW_POINTS
      );
    }
  }
  return state.runtimeTraceRecords.map((item) => ({ ...item }));
}

function candidateTraceBaselineStats(candidateId, trace = []) {
  const recentPeaks = (Array.isArray(trace) ? trace : [])
    .slice(-TRACE_WINDOW_POINTS)
    .map((item) => globalCandidatePeaks(item).find((peak) => peak?.candidate_id === candidateId))
    .filter(
      (peak) =>
        peak?.candidate_reference_status === "session_global_no_contact_baseline" &&
        Number.isFinite(candidateShiftPm(peak))
    );
  if (!recentPeaks.length) {
    return { centerPm: 0, robustSigmaPm: 0, sampleCount: 0, conditioned: false };
  }
  const noiseValues = recentPeaks
    .map((peak) => Number(peak?.candidate_baseline_noise_pm))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  return {
    // candidate_delta_wavelength_pm is already relative to the frozen
    // session no-contact baseline. Contact frames must never recenter it.
    centerPm: 0,
    robustSigmaPm: noiseValues.length ? Math.max(0, quantile(noiseValues, 0.5)) : 0,
    sampleCount: recentPeaks.length,
    conditioned: recentPeaks.length >= GLOBAL_EVENT_MIN_CONDITIONING_FRAMES,
  };
}

function globalCandidateBaselineStatsMap(trace = []) {
  return new Map(
    GLOBAL_CANDIDATE_IDS.map((candidateId) => [candidateId, candidateTraceBaselineStats(candidateId, trace)])
  );
}

function globalCandidateSpatialProxy(peaks = [], trace = [], baselineStatsByCandidate = null) {
  const preliminaryEntries = peaks
    .filter((peak) => peak?.valid !== false && ARRAY_CHANNEL_COORDS[peak?.provisional_channel_id])
    .map((peak) => {
      const currentShiftPm = candidateShiftPm(peak);
      const rawAbsoluteShiftPm = Math.abs(currentShiftPm);
      const baselineStats = baselineStatsByCandidate?.get(peak.candidate_id) ||
        candidateTraceBaselineStats(peak.candidate_id, trace);
      const baselineNoisePm = Number(peak?.candidate_baseline_noise_pm);
      return {
        peak,
        candidateId: peak.candidate_id,
        channelId: peak.provisional_channel_id,
        currentShiftPm,
        rawAbsoluteShiftPm,
        residualCenterPm: baselineStats.centerPm,
        robustSigmaPm: baselineStats.robustSigmaPm,
        conditioningSampleCount: baselineStats.sampleCount,
        conditioned: baselineStats.conditioned,
        baselineNoisePm,
        centeredDeltaPm: Number.isFinite(currentShiftPm)
          ? currentShiftPm - baselineStats.centerPm
          : 0,
        ...ARRAY_CHANNEL_COORDS[peak.provisional_channel_id],
      };
    });

  const conditionedEntries = preliminaryEntries.filter((entry) => entry.conditioned);
  const commonModePm = conditionedEntries.length
    ? quantile(
      conditionedEntries.map((entry) => entry.centeredDeltaPm).sort((a, b) => a - b),
      0.5
    )
    : 0;
  let entries = preliminaryEntries.map((entry) => {
    const deadbandPm = Math.max(
      GLOBAL_EVENT_DEADBAND_PM,
      Number.isFinite(entry.baselineNoisePm) ? entry.baselineNoisePm * 3 : 0,
      Number.isFinite(entry.robustSigmaPm) ? entry.robustSigmaPm * 3 : 0
    );
    const localDeltaPm = entry.conditioned
      ? Math.abs(entry.centeredDeltaPm - commonModePm)
      : 0;
    const eventShiftPm = Math.max(0, localDeltaPm - deadbandPm);
    return {
      ...entry,
      commonModePm,
      localDeltaPm,
      deadbandPm,
      eventShiftPm,
      proxyValue: globalSpectralProxyValue(eventShiftPm),
    };
  });
  const rankedEvidence = entries
    .map((entry) => entry.eventShiftPm)
    .filter(Number.isFinite)
    .sort((a, b) => b - a);
  const primaryEvidencePm = rankedEvidence[0] || 0;
  const secondaryEvidencePm = rankedEvidence[1] || 0;
  const contactEvidence = conditionedEntries.length === GLOBAL_CANDIDATE_IDS.length && (
    primaryEvidencePm >= GLOBAL_EVENT_SINGLE_PEAK_TRIGGER_PM ||
    (primaryEvidencePm >= GLOBAL_EVENT_PRIMARY_TRIGGER_PM && secondaryEvidencePm >= GLOBAL_EVENT_SECONDARY_TRIGGER_PM)
  );
  if (!contactEvidence) {
    entries = entries.map((entry) => ({ ...entry, eventShiftPm: 0, proxyValue: 0 }));
  }

  const byChannel = new Map(entries.map((entry) => [entry.channelId, entry]));
  const surfaceGrid = ARRAY_DISPLAY_ROWS.map((row) =>
    row.map((channelId) => Math.max(0, Math.min(1, byChannel.get(channelId)?.proxyValue || 0)))
  );
  const dominantEntry = entries.reduce(
    (best, entry) => (!best || entry.eventShiftPm > best.eventShiftPm ? entry : best),
    null
  );
  const eventPeakShiftPm = dominantEntry?.eventShiftPm || 0;
  const rawPeakShiftPm = entries.reduce(
    (maximum, entry) => Math.max(maximum, Number(entry.rawAbsoluteShiftPm) || 0),
    0
  );
  const activeThresholdPm = Math.max(GLOBAL_EVENT_DEADBAND_PM, eventPeakShiftPm * 0.18);
  const respondingEntries = entries.filter(
    (entry) => entry.eventShiftPm >= activeThresholdPm && entry.proxyValue >= 0.025
  );
  const weightedEntries = entries.filter((entry) => entry.eventShiftPm > 0);
  const weightSum = weightedEntries.reduce((sum, entry) => sum + entry.eventShiftPm, 0);
  const centroidX = weightSum > 0
    ? weightedEntries.reduce((sum, entry) => sum + entry.x * entry.eventShiftPm, 0) / weightSum
    : 0;
  const centroidY = weightSum > 0
    ? weightedEntries.reduce((sum, entry) => sum + entry.y * entry.eventShiftPm, 0) / weightSum
    : 0;
  const radialVariance = weightSum > 0
    ? weightedEntries.reduce(
      (sum, entry) => sum + entry.eventShiftPm * ((entry.x - centroidX) ** 2 + (entry.y - centroidY) ** 2),
      0
    ) / weightSum
    : 0;
  const spread = weightSum > 0 ? Math.max(0.24, Math.min(1.15, 0.24 + 0.42 * Math.sqrt(radialVariance))) : 0.24;
  const proxyValues = entries.map((entry) => entry.proxyValue);

  return {
    entries,
    surfaceGrid,
    dominantEntry,
    eventPeakShiftPm,
    rawPeakShiftPm,
    surfacePeak: Math.max(0, ...proxyValues),
    surfaceMean: proxyValues.length
      ? proxyValues.reduce((sum, value) => sum + value, 0) / proxyValues.length
      : 0,
    centroidX,
    centroidY,
    spread,
    respondingEntries,
    activeArea: respondingEntries.length / Math.max(1, GLOBAL_CANDIDATE_IDS.length),
    conditioningReady: conditionedEntries.length === GLOBAL_CANDIDATE_IDS.length,
    conditioningSampleCount: conditionedEntries.length
      ? Math.min(...conditionedEntries.map((entry) => entry.conditioningSampleCount))
      : 0,
    commonModePm,
    contactEvidence,
    primaryEvidencePm,
    secondaryEvidencePm,
  };
}

function globalEventTraceRecords(records = [], baselineStatsByCandidate = null) {
  const items = Array.isArray(records) ? records : [];
  const stats = baselineStatsByCandidate || globalCandidateBaselineStatsMap(items);
  return items.map((item) => {
    const peaks = globalCandidatePeaks(item).filter(
      (peak) => peak?.valid !== false && Number.isFinite(candidateShiftPm(peak))
    );
    if (peaks.length !== GLOBAL_CANDIDATE_IDS.length) {
      return {
        ...item,
        recognition_scope: GLOBAL_RECOGNITION_SCOPE,
        global_event_absolute_shift_pm: null,
        surface_peak: null,
        contact_evidence_passed: false,
        trace_response_semantics: "incomplete_global_candidate_frame",
      };
    }
    const proxy = globalCandidateSpatialProxy(peaks, items, stats);
    return {
      ...item,
      recognition_scope: GLOBAL_RECOGNITION_SCOPE,
      global_event_absolute_shift_pm: proxy.eventPeakShiftPm,
      surface_peak: proxy.surfacePeak,
      contact_evidence_passed: proxy.contactEvidence,
      global_common_mode_pm: proxy.commonModePm,
      trace_response_semantics: "residual_compensated_global_event_peak",
    };
  });
}

function suppressGlobalSpatialProxy(proxy = {}, reason = "response_not_allowed") {
  const entries = (Array.isArray(proxy.entries) ? proxy.entries : []).map((entry) => ({
    ...entry,
    eventShiftPm: 0,
    proxyValue: 0,
  }));
  return {
    ...proxy,
    entries,
    surfaceGrid: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    dominantEntry: null,
    eventPeakShiftPm: 0,
    surfacePeak: 0,
    surfaceMean: 0,
    centroidX: 0,
    centroidY: 0,
    spread: 0.24,
    respondingEntries: [],
    activeArea: 0,
    contactEvidence: false,
    primaryEvidencePm: 0,
    secondaryEvidencePm: 0,
    responseSuppressedReason: reason,
  };
}

function isModelPositionLevelMode(mode) {
  return String(mode || "") === "all_source_optical_force_position";
}

function operatorVisualizationContract(...values) {
  for (const value of values) {
    const candidate = value?.operator_visualization_frame || value;
    if (candidate?.contract_version === OPERATOR_VISUALIZATION_CONTRACT_VERSION) {
      return candidate;
    }
  }
  return null;
}

function activeModelPrediction(record = null, arrayFrame = null) {
  const contract = operatorVisualizationContract(record, arrayFrame);
  return contract?.prediction || null;
}

function opticalForcePresentation(record = null, arrayFrame = null, runtimeFrame = null) {
  const contract = operatorVisualizationContract(record, arrayFrame, runtimeFrame);
  if (contract) {
    const force = contract.force || {};
    const rawValue = finiteNumberOrNull(force.display_n);
    const valueN = rawValue === null ? null : Math.max(0, rawValue);
    const calibratedRange = Array.isArray(force.calibrated_range_n)
      ? force.calibrated_range_n
      : [0, OPTICAL_FORCE_CALIBRATED_MAX_N];
    const calibratedMaxN = Number.isFinite(Number(calibratedRange[1]))
      ? Number(calibratedRange[1])
      : OPTICAL_FORCE_CALIBRATED_MAX_N;
    return {
      enabled: true,
      valueN,
      prediction: contract.prediction || null,
      source: String(contract.model_source || CURRENT_RUNTIME_MODEL_SOURCE),
      calibratedMaxN,
      rangeStatus: String(force.range_status || "unavailable"),
    };
  }
  return {
    enabled: false,
    valueN: null,
    prediction: null,
    source: CURRENT_RUNTIME_MODEL_SOURCE,
    calibratedMaxN: OPTICAL_FORCE_CALIBRATED_MAX_N,
    rangeStatus: "unavailable",
  };
}

function activeModelDisplayName() {
  return state.activeRecognitionModelLabel || "Current optical model";
}

function simulatedScenarioStateLabel(arrayFrame = {}, surfaceMetrics = {}) {
  const interpretation = String(surfaceMetrics?.event_interpretation || "").toLowerCase();
  const peak = Number(surfaceMetrics?.surface_peak);
  const responding = Number(
    surfaceMetrics?.responding_channel_count ?? surfaceMetrics?.active_channel_count ?? 0
  );

  const noActiveContact =
    interpretation.startsWith("no_contact") ||
    interpretation.includes("no active contact") ||
    (Number.isFinite(peak) && peak < 0.05 && responding === 0);
  if (noActiveContact) return "no_contact";
  return levelLabel(responseLevelFromSurfaceValue(peak));
}

function surfaceContactPresentation({
  arrayFrame = {},
  surfaceMetrics = {},
  record = null,
  arrayMode = "",
  measurementAvailable = false,
  heldMeasurement = false,
} = {}) {
  if (!measurementAvailable) {
    return { active: false, primary: "no_contact", secondary: "Optical response level" };
  }

  if (isModelPositionLevelMode(arrayMode)) {
    const prediction = activeModelPrediction(record, arrayFrame);
    const twin = prediction?.digital_twin || {};
    const active = (twin.visual_active ?? twin.active) === true;
    if (!active) {
      return {
        active: false,
        primary: "no_contact",
        secondary: heldMeasurement ? "Optical response level | held" : "Optical response level",
      };
    }
    return {
      active: true,
      primary: "contact",
      secondary: heldMeasurement ? "Optical response level | held" : "Optical response level",
    };
  }

  if (String(arrayMode || "").startsWith("global_spectrum_")) {
    const complete = globalCandidatePeaks(record).length === GLOBAL_CANDIDATE_IDS.length;
    const peak = Number(surfaceMetrics?.surface_peak);
    const responding = Number(surfaceMetrics?.responding_channel_count || 0);
    const active = complete && Number.isFinite(peak) && peak >= 0.02 && responding > 0;
    return {
      active,
      primary: !complete ? "uncertain" : active ? "contact" : "no_contact",
      secondary: heldMeasurement ? "Optical response level | held" : "Optical response level",
    };
  }

  const peak = Number(surfaceMetrics?.surface_peak ?? record?.wavelength_shift_response_ratio ?? record?.response_value);
  const respondingRaw = surfaceMetrics?.responding_channel_count ?? surfaceMetrics?.active_channel_count;
  const responding = Number(respondingRaw);
  const hasRespondingCount = Number.isFinite(responding);
  const active = Number.isFinite(peak) && peak >= 0.02 && (!hasRespondingCount || responding > 0);
  const scenarioState = simulatedScenarioStateLabel(arrayFrame, surfaceMetrics);

  if (!active) {
    return { active: false, primary: "no_contact", secondary: "Optical response level" };
  }

  const fallbackLike = ["p22_fallback", "single_point_p22", "no_valid_channel", ""].includes(String(arrayMode || ""));
  if (fallbackLike) {
    return {
      active: true,
      primary: "contact",
      secondary: heldMeasurement ? "Optical response level | held" : "Optical response level",
    };
  }

  return {
    active: true,
    primary: scenarioState === "no_contact" ? "no_contact" : "contact",
    secondary: heldMeasurement ? "Optical response level | held" : "Optical response level",
  };
}

function clampArrayCoord(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(-1.25, Math.min(1.25, numeric));
}

function arrayCoordToSurfaceScene(x, y) {
  if (THREE_SURFACE_AXIS_MODE === "swap_array_xy") {
    return {
      x: THREE_SURFACE_ARRAY_TO_SCENE_X_SIGN * (clampArrayCoord(y) / 1.25) * 3.5,
      z: THREE_SURFACE_ARRAY_TO_SCENE_Z_SIGN * (clampArrayCoord(x) / 1.25) * 2.5,
    };
  }
  return {
    x: THREE_SURFACE_ARRAY_TO_SCENE_X_SIGN * (clampArrayCoord(x) / 1.25) * 3.5,
    z: THREE_SURFACE_ARRAY_TO_SCENE_Z_SIGN * (clampArrayCoord(y) / 1.25) * 2.5,
  };
}

function surfaceSceneToArrayCoord(x, z) {
  if (THREE_SURFACE_AXIS_MODE === "swap_array_xy") {
    return {
      x: clampArrayCoord(THREE_SURFACE_ARRAY_TO_SCENE_Z_SIGN * (z / 2.5) * 1.25),
      y: clampArrayCoord(THREE_SURFACE_ARRAY_TO_SCENE_X_SIGN * (x / 3.5) * 1.25),
    };
  }
  return {
    x: clampArrayCoord(THREE_SURFACE_ARRAY_TO_SCENE_X_SIGN * (x / 3.5) * 1.25),
    y: clampArrayCoord(THREE_SURFACE_ARRAY_TO_SCENE_Z_SIGN * (z / 2.5) * 1.25),
  };
}

const state = {
  paused: false,
  frame: null,
  activeRecognitionModelLabel: "Loading...",
  lastRenderedSourceFrameKey: null,
  selectedChannel: "P22",
  smoothAttenuation: 0,
  smoothDeformation: 0,
  smoothSurfaceVisualPeak: 0,
  smoothSurfaceCentroidX: 0,
  smoothSurfaceCentroidY: 0,
  smoothSurfaceSpread: 0.34,
  smoothSurfaceActiveArea: 0,
  targetAttenuation: 0,
  targetDeformation: 0,
  targetSurfaceVisualPeak: 0,
  targetSurfaceCentroidX: 0,
  targetSurfaceCentroidY: 0,
  targetSurfaceSpread: 0.34,
  targetSurfaceActiveArea: 0,
  lastThreeFrameMs: 0,
  lastGeometryUpdateMs: 0,
  lastGeometryNormalUpdateMs: 0,
  geometryDeltaAccumulator: 0,
  lastChartUpdateMs: 0,
  chartDeltaAccumulator: 0,
  threeNeedsRefresh: true,
  targetTraceRecords: [],
  smoothTraceRecords: [],
  runtimeTraceRecords: [],
  runtimeTraceSource: null,
  opticalForceDisplayMaxN: OPTICAL_FORCE_MIN_DISPLAY_MAX_N,
  targetSpectrumRecord: null,
  smoothSpectrumRecord: null,
  chartsNeedRefresh: true,
  currentArrayFrame: null,
  currentSurfaceGrid: null,
  currentSurfaceMetrics: null,
  trajectoryHistory: [],
  nodeDebugExpanded: false,
  arrayDemoActive: false,
  arrayDemoScenario: null,
  arrayDemoStep: 0,
  arrayDemoTraceRecords: [],
  dataStreamActive: false,
  exportWatchActive: false,
  sdkLiveActive: false,
  liveRequested: false,
  spectrumProcessing: {
    settings: {
      integration_us: 300,
      sensor_mode: 0,
      update_overlay_data: true,
      subtract_background: false,
      baseline_correction: true,
      spectrum_smoothing: true,
      normalize_spectrum: true,
      model_input_source: "raw_intensity",
    },
    background_reference_ready: false,
    raw_spectrum_retained: true,
    model_input_source: "raw_intensity",
    display_input_source: "display_intensity",
  },
  demoModeActive: false,
  demoAutoplay: false,
  demoTimer: null,
  demoStepIndex: 0,
  demoCurrentLevel: null,
  demoPlaybackRate: storedDemoPlaybackRate(),
  arrayDemoNextStepAt: 0,
  arrayDemoCycleStartedAt: 0,
  arrayDemoActionCompletedAt: 0,
  arrayDemoPlaybackMode: null,
  arrayDemoActionComplete: false,
  nextLiveModelPollAt: 0,
  nextLiveSourceProbeAt: 0,
  liveSourceProbeInFlight: false,
  liveSourceProbeController: null,
  displayMode: "operator",
  surfaceRenderMode: "physical_proxy",
  geometryDisplayMode: "thumb_holder",
  selectedFinger: "all",
  fingerNavigationIndex: 0,
  fingerCloseupActive: false,
  surfaceFullscreenActive: false,
  surfaceNativeFullscreenEntered: false,
  thumbSceneConfig: null,
  thumbModelStatus: "not_loaded",
  thumbModelMessage: "--",
  thumbModelAssetUrl: "/static/assets/models/thumb_holder.stl",
  wholeHandModelStatus: "not_loaded",
  wholeHandModelMessage: "--",
  wholeHandModelAssetUrl: "/static/assets/models/robot_nano_hand_sensorized.glb",
  couplingView: "raw_coupled_response",
  layoutCheckVisible: false,
  commandPending: null,
  commandFeedbackTimer: null,
  diagnosticTab: "signal",
  frameModeEpoch: 0,
  frameRequestSequence: 0,
  lastCommittedFrameRequest: 0,
  frameRequestInFlight: false,
  frameRequestController: null,
  forcedFrameRequestQueued: false,
  px6dRequestInFlight: false,
  px6dRequestController: null,
  px6dLatest: null,
  px6dUiStatus: null,
  px6dAligned: null,
  px6dOpticalFrame: null,
  px6dCaptureRequestInFlight: false,
  px6dCapturePollInFlight: false,
  px6dCapturePollController: null,
  px6dCaptureStatusEpoch: 0,
  px6dCaptureStatus: null,
  measurementRequestInFlight: false,
  measurementSessions: [],
  measurementResult: null,
  pageVisible: document.visibilityState !== "hidden",
  animationFrameId: null,
  controlsInteracting: false,
  controlsSettlingUntil: 0,
  bootStarted: false,
  bootComplete: false,
  clientSchedulersStarted: false,
  frameSchedulerId: null,
  px6dSchedulerId: null,
  px6dCaptureSchedulerId: null,
};

let scene;
let camera;
let renderer;
let controls;
let sceneAmbientLight;
let sceneKeyLight;
let sceneFillLight;
let surfaceGeometry;
let surfaceBasePositions;
let surfaceMesh;
let bodyBasePositions;
let bodyMesh;
let bodyWireMesh;
let wireMesh;
let sceneGrid;
let surfaceGridBasePositions;
let bottomGridMesh;
let bottomGridBasePositions;
let thumbModelRoot;
let thumbHolderObject;
let wholeHandRoot;
let wholeHandBodyObject;
let sensorSurfaceGroup;
let slotOutlineHelper;
let grooveReferenceHelper;
const fingerSensorGroups = new Map();
const fingerRaycaster = new THREE.Raycaster();
const fingerPointer = new THREE.Vector2();
let fingerPointerDown = null;
let cameraTransition = null;
let resizeToken = 0;
let windowResizeSettleTimer = 0;
let windowResizeActive = false;

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function px6dConnectionDisplayState(status = {}) {
  if (status.connected === true) return "connected";
  const lifecycle = String(status.lifecycle_status || "").toLowerCase();
  const errorKind = String(status.connection_error_kind || "").toLowerCase();
  const detection = String(status.port_detection_status || "").toLowerCase();
  if (status.connection_in_progress === true || lifecycle === "connecting") {
    return "connecting";
  }
  if (errorKind === "port_busy_or_permission_denied") {
    return "port in use";
  }
  if (
    errorKind === "port_not_found" ||
    detection === "device_not_detected" ||
    detection === "configured_device_identity_not_detected"
  ) {
    return "waiting for sensor";
  }
  if (lifecycle === "reconnecting") return "reconnecting";
  if (lifecycle === "stopped" || status.stop_requested === true) return "stopped";
  if (lifecycle === "unavailable") return "unavailable";
  return status.running === true ? "reconnecting" : "stopped";
}

function px6dCaptureIsBusy() {
  const capture = state.px6dCaptureStatus || {};
  return Boolean(
    capture.running === true ||
    capture.start_in_progress === true ||
    capture.worker_alive === true ||
    state.px6dCaptureRequestInFlight
  );
}

function updatePx6dZeroButtonState() {
  const connected = state.px6dUiStatus?.connected === true;
  const captureBusy = px6dCaptureIsBusy();
  const commandBusy = state.commandPending !== null;
  const disabled = !connected || captureBusy || commandBusy;
  const title = captureBusy
    ? "Stop recording before zeroing the force sensor"
    : commandBusy
      ? "Wait for the current command to finish"
      : connected
        ? "Zero force sensor"
        : "Force sensor is offline";
  [px6dTareButton, diagnosticPx6dTareButton, px6dCaptureTareButton].forEach((button) => {
    if (!button) return;
    button.disabled = disabled;
    button.title = title;
  });
}

function updatePx6dPanel(payload = {}) {
  if (Object.prototype.hasOwnProperty.call(payload, "px6d_reference")) {
    state.px6dAligned = payload?.px6d_reference || null;
    const latest = payload?.latest;
    state.px6dOpticalFrame = latest && typeof latest === "object"
      ? {
          frameId: latest.frame_id ?? "--",
          timestamp: latest.ingested_at ?? latest.timestamp_epoch_sec ?? latest.timestamp ?? null,
          source: latest.source || "optical spectrum",
        }
      : null;
  }
  const status = payload?.px6d_status || payload?.status || state.px6dLatest?.status || {};
  const pair = state.px6dAligned;
  const aligned = pair?.ok === true ? pair : null;
  const sample = payload?.sample || state.px6dLatest?.sample || null;
  const mechanicalSource = aligned || sample || {};
  const zeroed = mechanicalSource?.zeroed || {};
  const filteredZeroed = mechanicalSource?.filtered_zeroed || zeroed;
  const mechanical = mechanicalSource?.mechanical || {};
  const candidateMechanical = mechanicalSource?.filtered_mechanical || mechanical;
  const referenceValue = finiteNumberOrNull(
    aligned?.force_fz_n ??
    sample?.force_fz_n ??
    aligned?.reference_fz_display_n ??
    sample?.reference_fz_display_n ??
    sample?.conditioned_reference_fz_n ??
    sample?.filtered_reference_fz_n
  );
  const rawFzValue = finiteNumberOrNull(aligned?.raw?.fz_n ?? sample?.raw?.fz_n);
  const connected = status?.connected === true;
  const tareReady = Boolean(aligned?.tare_ready ?? sample?.tare_ready ?? status?.tare_ready);
  const sampleAge = finiteNumberOrNull(status?.last_sample_age_sec);
  const freshnessLimit = finiteNumberOrNull(status?.sample_freshness_limit_sec) ?? 0.5;
  const fresh = status?.sample_fresh === true || (
    connected &&
    Number.isFinite(sampleAge) &&
    sampleAge >= 0 &&
    sampleAge <= freshnessLimit
  );
  const liveForceVisible = connected && fresh && Number.isFinite(referenceValue);
  const currentForceReady = connected && tareReady && fresh;
  const rawFz = fresh ? rawFzValue : null;
  const currentFilteredZeroed = currentForceReady ? filteredZeroed : {};
  const displayedMechanical = currentForceReady ? candidateMechanical : {};
  const connectionState = px6dConnectionDisplayState(status);
  const stateLabel = !connected
    ? connectionState
    : !tareReady
      ? "tare required"
      : fresh
        ? "ready"
        : "stale";
  const stateTone = connected && tareReady && fresh ? "ready" : "warning";
  state.px6dUiStatus = {
    connected,
    tareReady,
    fresh,
    liveForceVisible,
    stateLabel,
  };

  const displayedCompressionFz = liveForceVisible
    ? Math.max(0, referenceValue)
    : null;
  setText(
    "px6dReferenceFz",
    Number.isFinite(displayedCompressionFz) ? `${displayedCompressionFz.toFixed(3)} N` : "-- N"
  );
  setText("px6dReferenceStatus", stateLabel);
  const statusElement = document.getElementById("px6dReferenceStatus");
  if (statusElement) statusElement.dataset.state = stateTone;
  updatePx6dZeroButtonState();

  const px6dPort = status.active_port || status.port || status.configured_port || "COM3";
  setText(
    "diagnosticPx6dConnection",
    connected ? `${px6dPort} | connected` : `${px6dPort} | ${connectionState}`
  );
  setText("diagnosticPx6dFirmware", `Firmware ${status?.firmware_version || "--"}`);
  setText("diagnosticPx6dRawFz", Number.isFinite(rawFz) ? `${rawFz.toFixed(3)} N` : "--");
  setText(
    "diagnosticPx6dReferenceFz",
    Number.isFinite(displayedCompressionFz) ? `${displayedCompressionFz.toFixed(3)} N` : "-- N"
  );
  setText(
    "px6dCaptureForceValue",
    Number.isFinite(displayedCompressionFz) ? displayedCompressionFz.toFixed(3) : "--"
  );
  const axisSpecs = [
    ["diagnosticPx6dFx", currentFilteredZeroed?.fx_n, 3],
    ["diagnosticPx6dFy", currentFilteredZeroed?.fy_n, 3],
    ["diagnosticPx6dFz", displayedCompressionFz, 3],
    ["diagnosticPx6dMx", currentFilteredZeroed?.mx_nm, 4],
    ["diagnosticPx6dMy", currentFilteredZeroed?.my_nm, 4],
    ["diagnosticPx6dMz", currentFilteredZeroed?.mz_nm, 4],
  ];
  axisSpecs.forEach(([id, value, digits]) => {
    const numeric = finiteNumberOrNull(value);
    setText(id, Number.isFinite(numeric) ? numeric.toFixed(digits) : "--");
  });
  const derivedSpecs = [
    ["diagnosticPx6dForceResultant", displayedMechanical?.force_resultant_n, "N", 3],
    ["diagnosticPx6dShearResultant", displayedMechanical?.shear_resultant_n, "N", 3],
    ["diagnosticPx6dMomentResultant", displayedMechanical?.moment_resultant_nm, "N*m", 4],
    ["diagnosticPx6dUtilization", displayedMechanical?.peak_utilization_percent, "%", 1],
  ];
  derivedSpecs.forEach(([id, value, unit, digits]) => {
    const numeric = finiteNumberOrNull(value);
    setText(id, Number.isFinite(numeric) ? `${numeric.toFixed(digits)} ${unit}` : `-- ${unit}`);
  });
  const utilizationElement = document.getElementById("diagnosticPx6dUtilization");
  if (utilizationElement) utilizationElement.dataset.healthTone = displayedMechanical?.utilization_status === "warning" ? "warning" : "ok";
  setText("diagnosticPx6dTareStatus", tareReady ? "ready" : status?.tare_status || sample?.tare_status || "required");
  const tareNoise = finiteNumberOrNull(aligned?.tare_fz_std_n ?? sample?.tare_fz_std_n ?? status?.tare_fz_std_n);
  setText("diagnosticPx6dTareNoise", Number.isFinite(tareNoise) ? `${tareNoise.toFixed(4)} N` : "--");
  const filteredFz = currentForceReady
    ? finiteNumberOrNull(aligned?.filtered_reference_fz_n ?? sample?.filtered_reference_fz_n)
    : null;
  const driftOffset = finiteNumberOrNull(aligned?.drift_offset_n ?? sample?.drift_offset_n ?? status?.force_conditioning?.current_drift_offset_n);
  setText("diagnosticPx6dFilteredFz", Number.isFinite(filteredFz) ? `${filteredFz.toFixed(4)} N` : "--");
  setText("diagnosticPx6dDriftOffset", currentForceReady && Number.isFinite(driftOffset) ? `${driftOffset.toFixed(4)} N` : "--");
  setText(
    "diagnosticPx6dFilterStatus",
    aligned?.force_filter_status || sample?.force_filter_status || status?.force_conditioning?.filter_status || "--"
  );
  const observedRate = finiteNumberOrNull(status?.observed_sample_hz);
  setText("diagnosticPx6dSampleRate", Number.isFinite(observedRate) ? `${observedRate.toFixed(1)} Hz` : "--");
  setText("diagnosticPx6dSampleAge", Number.isFinite(sampleAge) ? `${(sampleAge * 1000).toFixed(0)} ms` : "--");
  const compressionSign = finiteNumberOrNull(status?.compression_sign);
  setText(
    "diagnosticPx6dCompressionSign",
    Number.isFinite(compressionSign)
      ? compressionSign < 0
        ? "raw -Fz = compression"
        : "raw +Fz = compression"
      : "--"
  );
  const forceRange = finiteNumberOrNull(status?.force_full_scale_per_axis_n);
  const momentRange = finiteNumberOrNull(status?.moment_full_scale_per_axis_nm);
  setText("diagnosticPx6dForceRange", Number.isFinite(forceRange) ? `+/-${forceRange.toFixed(0)} N / axis` : "--");
  setText("diagnosticPx6dMomentRange", Number.isFinite(momentRange) ? `+/-${momentRange.toFixed(1)} N*m / axis` : "--");

  const syncOffset = finiteNumberOrNull(pair?.sync_offset_ms);
  const syncQuality = aligned?.sync_quality || (state.px6dOpticalFrame ? "not paired" : "waiting");
  const syncQualityElement = document.getElementById("diagnosticPx6dSyncQuality");
  if (syncQualityElement) {
    syncQualityElement.textContent = syncQuality;
    syncQualityElement.dataset.syncTone = aligned?.sync_quality || "waiting";
  }
  setText("diagnosticPx6dSyncOffset", Number.isFinite(syncOffset) ? `${syncOffset.toFixed(1)} ms` : "-- ms");
  setText("diagnosticOpticalFrameId", state.px6dOpticalFrame ? String(state.px6dOpticalFrame.frameId ?? "--") : "NO FRAME");
  const sequenceStart = finiteNumberOrNull(aligned?.force_sequence_start);
  const sequenceEnd = finiteNumberOrNull(aligned?.force_sequence_end);
  setText(
    "diagnosticForceFrameId",
    Number.isFinite(sequenceStart)
      ? sequenceStart === sequenceEnd
        ? `#${sequenceStart}`
        : `#${sequenceStart}-${sequenceEnd}`
      : sample?.sequence_id
        ? `#${sample.sequence_id}`
        : "--"
  );
  setText("diagnosticPx6dSyncMethod", aligned?.sync_method || "--");
  const syncSampleCount = finiteNumberOrNull(aligned?.sample_count);
  setText("diagnosticPx6dSyncSamples", Number.isFinite(syncSampleCount) ? String(syncSampleCount) : "--");
  const syncNote = aligned
    ? `Paired by host timestamp. ${syncQuality} alignment at ${Number.isFinite(syncOffset) ? Math.abs(syncOffset).toFixed(1) : "--"} ms; PX6D remains an external reference.`
    : state.px6dOpticalFrame
      ? `Optical frame present, but force pairing is unavailable: ${pair?.status || "force frame missing"}.`
      : "Waiting for an optical spectrum frame. PX6D can continue independently.";
  setText("diagnosticPx6dSync", syncNote);
  setText(
    "diagnosticPx6dError",
    status?.last_error || (tareReady ? "conditioned display active; raw six-axis values retained" : "stable no-load zero pending")
  );
  updateCaptureReadiness();
}

async function fetchPx6dReference() {
  if (!state.pageVisible || state.px6dRequestInFlight) return;
  state.px6dRequestInFlight = true;
  const requestController = new AbortController();
  state.px6dRequestController = requestController;
  try {
    const payload = await requestJSON(
      "/api/px6d/latest",
      { cache: "no-store" },
      { timeoutMs: 1200, signal: requestController.signal }
    );
    state.px6dLatest = payload;
    updatePx6dPanel(payload);
  } catch (error) {
    if (error?.name === "AbortError" && requestController.signal.aborted) return;
    updatePx6dPanel({
      status: {
        connected: false,
        port: "COM3",
        last_error: commandErrorMessage(error, "PX6D reference unavailable"),
      },
    });
  } finally {
    if (state.px6dRequestController === requestController) {
      state.px6dRequestController = null;
      state.px6dRequestInFlight = false;
    }
  }
}

const CAPTURE_STREAM_LABELS = {
  spectrum: "Spectrum",
  response: "Recognition",
  force: "Force",
};

function selectedCaptureOutputs() {
  return [
    [px6dCaptureSpectrum, "spectrum"],
    [px6dCaptureResponse, "response"],
    [px6dCaptureForce, "force"],
  ].filter(([control]) => control?.checked).map(([, value]) => value);
}

function setCaptureReadinessCell(id, value, readinessState) {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = value;
  element.dataset.state = readinessState;
}

function formatCaptureDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function updateCapturePositionSelection(value = px6dCapturePosition?.value || "") {
  if (px6dCapturePosition && px6dCapturePosition.value !== value) {
    px6dCapturePosition.value = value;
  }
  px6dCapturePositionButtons.forEach((button) => {
    const selected = button.dataset.capturePosition === value;
    button.setAttribute("aria-pressed", selected ? "true" : "false");
  });
}

function nextCaptureTrialId() {
  if (!px6dCaptureTrial) return;
  const current = String(px6dCaptureTrial.value || "trial_001").trim();
  const match = current.match(/^(.*?)(\d+)$/);
  if (!match) {
    px6dCaptureTrial.value = `${current || "trial"}_002`;
    return;
  }
  const width = match[2].length;
  const next = String(Number.parseInt(match[2], 10) + 1).padStart(width, "0");
  px6dCaptureTrial.value = `${match[1]}${next}`;
}

function updateCaptureReadiness() {
  const selected = selectedCaptureOutputs();
  const opticalRequired = selected.includes("spectrum") || selected.includes("response");
  const forceRequired = selected.includes("force");
  const opticalActive = Boolean(
    state.sdkLiveActive || state.exportWatchActive || state.dataStreamActive || state.liveRequested
  );
  const opticalReady = opticalActive && frameSourceIsFresh(state.frame);
  const forceState = state.px6dUiStatus || {};
  const forceReady = Boolean(forceState.connected && forceState.tareReady && forceState.fresh);
  const position = String(px6dCapturePosition?.value || "");
  const folderReady = Boolean(String(px6dCaptureOutputRoot?.value || "").trim());

  setCaptureReadinessCell(
    "px6dCaptureSpectrumReady",
    opticalRequired ? (opticalReady ? "Ready" : "Waiting") : "Off",
    opticalRequired ? (opticalReady ? "ready" : "waiting") : "off"
  );
  const forceLabel = !forceRequired
    ? "Off"
    : !forceState.connected
      ? "Offline"
      : !forceState.tareReady
        ? "Zero"
        : !forceState.fresh
          ? "Stale"
          : "Ready";
  setCaptureReadinessCell(
    "px6dCaptureForceReady",
    forceLabel,
    !forceRequired ? "off" : forceReady ? "ready" : forceState.connected ? "waiting" : "error"
  );
  setCaptureReadinessCell(
    "px6dCapturePositionReady",
    position ? (position === "unlabeled" ? "Unlabeled" : position) : "Select",
    position ? (position === "unlabeled" ? "waiting" : "ready") : "waiting"
  );
  setCaptureReadinessCell(
    "px6dCaptureFolderReady",
    folderReady ? "Ready" : "Select",
    folderReady ? "ready" : "waiting"
  );

  const missing = [];
  if (!selected.length) missing.push("data");
  if (opticalRequired && !opticalReady) missing.push("spectrum");
  if (forceRequired && !forceReady) missing.push("force");
  if (!position) missing.push("position");
  if (!folderReady) missing.push("folder");
  const ready = missing.length === 0;
  const payload = state.px6dCaptureStatus || {};
  const running = payload.running === true;
  const captured = Number(payload.captured_timeline_frames ?? payload.captured_paired_frames ?? 0);
  const saved = !running && captured > 0 && Number.isFinite(Number(payload.ended_at_epoch_sec));
  const failed = !running && Boolean(payload.last_error);
  const summaryStatus = running ? "Recording" : failed ? "Error" : saved ? "Saved" : ready ? "Ready" : "Check setup";
  const summaryState = running ? "recording" : failed ? "error" : saved ? "saved" : ready ? "ready" : "waiting";
  setText("px6dCaptureSummaryStatus", summaryStatus);
  const summaryElement = document.getElementById("px6dCaptureSummaryStatus");
  if (summaryElement) summaryElement.dataset.state = summaryState;
  setText("px6dCaptureStatus", summaryStatus);
  const statusElement = document.getElementById("px6dCaptureStatus");
  if (statusElement) statusElement.title = String(payload.capture_status || "idle").replaceAll("_", " ");
  if (px6dCaptureStartButton) {
    const commandBusy = state.commandPending !== null;
    px6dCaptureStartButton.disabled = running || state.px6dCaptureRequestInFlight || commandBusy;
    px6dCaptureStartButton.dataset.readiness = ready ? "ready" : "setup-required";
    px6dCaptureStartButton.title = commandBusy
      ? "Wait for force zero to finish"
      : ready
        ? "Start recording"
        : `Check ${missing.join(", ")}`;
  }
  return { ready, running, selected, missing };
}

function updatePx6dCapturePanel(payload = {}) {
  state.px6dCaptureStatus = payload;
  const running = payload?.running === true;
  const captured = Number(payload?.captured_timeline_frames ?? payload?.captured_paired_frames ?? 0);
  const capturedFrameRate = Number(payload?.captured_frame_rate_hz);
  const ratioValue = payload?.paired_frame_ratio;
  const ratio = ratioValue === null || ratioValue === undefined ? NaN : Number(ratioValue);
  const syncOffset = Number(payload?.last_sync_offset_ms);
  const startedAt = Number(payload?.started_at_epoch_sec);
  const endedAt = running ? Date.now() / 1000 : Number(payload?.ended_at_epoch_sec);
  const duration = Number.isFinite(startedAt) && Number.isFinite(endedAt) ? Math.max(0, endedAt - startedAt) : 0;
  setText("px6dCaptureDuration", formatCaptureDuration(duration));
  setText(
    "px6dCaptureFrames",
    Number.isFinite(capturedFrameRate)
      ? `${captured} / ${capturedFrameRate.toFixed(1)} Hz`
      : String(captured)
  );
  const selectedOutputs = Array.isArray(payload?.selected_outputs) ? payload.selected_outputs : selectedCaptureOutputs();
  const syncRelevant = selectedOutputs.includes("force") && selectedOutputs.some((value) => value !== "force");
  setText("px6dCaptureRatio", syncRelevant && Number.isFinite(ratio) ? `${(ratio * 100).toFixed(1)}%` : "N/A");
  const ratioElement = document.getElementById("px6dCaptureRatio");
  if (ratioElement) {
    ratioElement.title = payload?.last_sync_quality
      ? `${payload.last_sync_quality} | ${Number.isFinite(syncOffset) ? `${syncOffset.toFixed(1)} ms` : "--"}`
      : "";
  }
  const output = payload?.output_directory;
  const selectedText = selectedOutputs.length
    ? selectedOutputs.map((value) => CAPTURE_STREAM_LABELS[value] || value).join(" + ")
    : "No data selected";
  const outputMessage = payload?.last_error
    ? payload.last_error
    : output
      ? running
        ? `Recording: ${selectedText}`
        : captured > 0
          ? `Saved: ${output}`
          : `Folder: ${output}`
      : "Ready.";
  setText("px6dCaptureOutput", outputMessage);
  const outputElement = document.getElementById("px6dCaptureOutput");
  if (outputElement) outputElement.title = output ? String(output) : outputMessage;
  if (px6dCaptureOutputRoot && document.activeElement !== px6dCaptureOutputRoot) {
    const preferredRoot = payload?.requested_output_root || payload?.default_output_root;
    if (preferredRoot && !px6dCaptureOutputRoot.dataset.userEdited) {
      px6dCaptureOutputRoot.value = preferredRoot;
    }
    if (preferredRoot && measurementRootInput && !measurementRootInput.dataset.userEdited) {
      measurementRootInput.value = preferredRoot;
    }
  }
  const locked = running || state.px6dCaptureRequestInFlight;
  [
    px6dCapturePosition,
    px6dCaptureTrial,
    px6dCaptureNote,
    px6dCaptureSpectrum,
    px6dCaptureResponse,
    px6dCaptureForce,
    px6dCaptureOutputRoot,
    px6dCaptureBrowseButton,
    px6dCaptureNextTrialButton,
  ].forEach((control) => {
    if (control) control.disabled = locked;
  });
  px6dCapturePositionButtons.forEach((button) => {
    button.disabled = locked;
  });
  updateCapturePositionSelection();
  if (px6dCaptureStopButton) px6dCaptureStopButton.disabled = !running || state.px6dCaptureRequestInFlight;
  updatePx6dZeroButtonState();
  updateCaptureReadiness();
}

async function fetchPx6dCaptureStatus() {
  if (
    !state.pageVisible ||
    state.px6dCaptureRequestInFlight ||
    state.px6dCapturePollInFlight
  ) return;
  state.px6dCapturePollInFlight = true;
  const requestEpoch = state.px6dCaptureStatusEpoch;
  const requestController = new AbortController();
  state.px6dCapturePollController = requestController;
  try {
    const payload = await requestJSON(
      "/api/px6d_capture/status",
      { cache: "no-store" },
      { timeoutMs: 1200, signal: requestController.signal }
    );
    if (requestEpoch !== state.px6dCaptureStatusEpoch) return;
    updatePx6dCapturePanel(payload);
  } catch (error) {
    if (error?.name === "AbortError" && requestController.signal.aborted) return;
    if (requestEpoch !== state.px6dCaptureStatusEpoch) return;
    updatePx6dCapturePanel({
      ...(state.px6dCaptureStatus || {}),
      running: false,
      capture_status: "capture API unavailable",
      last_error: commandErrorMessage(error, "capture API unavailable"),
    });
  } finally {
    if (state.px6dCapturePollController === requestController) {
      state.px6dCapturePollController = null;
      state.px6dCapturePollInFlight = false;
    }
  }
}

function invalidatePx6dCaptureStatusPoll() {
  state.px6dCaptureStatusEpoch += 1;
  if (state.px6dCapturePollController && !state.px6dCapturePollController.signal.aborted) {
    state.px6dCapturePollController.abort();
  }
}

function formatMeasurementMetric(value, digits = 2, unit = "") {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return `${numeric.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function setMeasurementStatus(label, detail = null) {
  setText("measurementStatus", label);
  if (detail !== null) setText("measurementOutput", detail);
}

function setMeasurementBusy(busy) {
  state.measurementRequestInFlight = Boolean(busy);
  if (measurementRefreshButton) measurementRefreshButton.disabled = busy;
  if (measurementSessionSelect) measurementSessionSelect.disabled = busy;
  if (measurementEstimateSource) measurementEstimateSource.disabled = busy;
  if (measurementAnalyzeButton) {
    measurementAnalyzeButton.disabled = busy || !measurementSessionSelect?.value;
  }
}

function clearMeasurementMetrics() {
  [
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
  ].forEach((id) => setText(id, "--"));
}

function drawMeasurementComparison(trace = []) {
  if (!measurementComparisonCanvas) return;
  const bounds = measurementComparisonCanvas.getBoundingClientRect();
  if (bounds.width < 20 || bounds.height < 20) return;
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  measurementComparisonCanvas.width = Math.max(1, Math.round(bounds.width * scale));
  measurementComparisonCanvas.height = Math.max(1, Math.round(bounds.height * scale));
  const context = measurementComparisonCanvas.getContext("2d");
  if (!context) return;
  context.setTransform(scale, 0, 0, scale, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfdff";
  context.fillRect(0, 0, width, height);

  const records = Array.isArray(trace) ? trace : [];
  const times = records.map((row, index) => {
    const value = Number(row?.elapsed_time_sec);
    return Number.isFinite(value) ? value : index;
  });
  const series = [
    {
      field: "reference_fz_n",
      color: "#0072b2",
      values: records.map((row) => Number(row?.reference_fz_n)),
      dash: [],
    },
    {
      field: "analysis_estimated_fz_n",
      color: "#d55e00",
      values: records.map((row) => Number(
        row?.analysis_estimated_fz_n ?? row?.optical_estimated_fz_n
      )),
      dash: [],
    },
  ];
  const finiteValues = series.flatMap((item) => item.values.filter(Number.isFinite));
  if (!records.length || !finiteValues.length) {
    context.fillStyle = "#60798b";
    context.font = "600 12px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("Select a completed recording", width / 2, height / 2);
    return;
  }

  const padding = { left: 38, right: 12, top: 20, bottom: 28 };
  const chartWidth = Math.max(1, width - padding.left - padding.right);
  const chartHeight = Math.max(1, height - padding.top - padding.bottom);
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const timeSpan = Math.max(maxTime - minTime, 1e-9);
  let minValue = Math.min(0, ...finiteValues);
  let maxValue = Math.max(0, ...finiteValues);
  if (Math.abs(maxValue - minValue) < 1e-9) maxValue = minValue + 1;
  const margin = Math.max((maxValue - minValue) * 0.08, 0.02);
  minValue -= margin;
  maxValue += margin;
  const valueSpan = maxValue - minValue;
  const xFor = (value) => padding.left + ((value - minTime) / timeSpan) * chartWidth;
  const yFor = (value) => padding.top + (1 - (value - minValue) / valueSpan) * chartHeight;

  context.lineWidth = 1;
  context.strokeStyle = "#dce8ef";
  context.fillStyle = "#6b8191";
  context.font = "500 9px system-ui, sans-serif";
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let index = 0; index <= 4; index += 1) {
    const fraction = index / 4;
    const y = padding.top + fraction * chartHeight;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    const value = maxValue - fraction * valueSpan;
    context.fillText(value.toFixed(valueSpan < 2 ? 2 : 1), padding.left - 5, y);
  }
  context.textAlign = "center";
  context.textBaseline = "top";
  for (let index = 0; index <= 4; index += 1) {
    const fraction = index / 4;
    const x = padding.left + fraction * chartWidth;
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, height - padding.bottom);
    context.stroke();
    context.fillText((minTime + fraction * timeSpan).toFixed(1), x, height - padding.bottom + 6);
  }
  context.save();
  context.translate(10, padding.top + chartHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillStyle = "#506b7e";
  context.font = "700 9px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "top";
  context.fillText("Fz (N)", 0, 0);
  context.restore();

  series.forEach((item) => {
    context.strokeStyle = item.color;
    context.lineWidth = 1.8;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.setLineDash(item.dash || []);
    context.beginPath();
    let drawing = false;
    item.values.forEach((value, index) => {
      if (!Number.isFinite(value) || !Number.isFinite(times[index])) {
        drawing = false;
        return;
      }
      const x = xFor(times[index]);
      const y = yFor(value);
      if (!drawing) context.moveTo(x, y);
      else context.lineTo(x, y);
      drawing = true;
    });
    context.stroke();
  });
  context.setLineDash([]);
}

function renderMeasurementResult(payload = {}) {
  state.measurementResult = payload;
  const summary = payload?.summary || {};
  const data = summary?.data || {};
  const direct = summary?.direct_comparison || {};
  const lag = summary?.lag || {};
  const cadence = summary?.cadence || {};
  const cycles = Array.isArray(payload?.cycle_rows) ? payload.cycle_rows : [];
  const recoveryValues = cycles
    .map((cycle) => Number(cycle?.release_recovery_ratio))
    .filter(Number.isFinite);
  const meanRecovery = recoveryValues.length
    ? recoveryValues.reduce((sum, value) => sum + value, 0) / recoveryValues.length
    : NaN;

  const selectedSource = String(
    data?.analysis_estimate_source
      || payload?.selected_estimate_source
      || "recorded_runtime"
  );
  const selectedLabel = String(
    data?.analysis_estimate_label || selectedSource.replaceAll("_", " ")
  );
  const evaluationValidity = String(data?.evaluation_validity || "unknown");
  setText("measurementEvidence", selectedLabel);
  setText("measurementValidity", evaluationValidity.replaceAll("_", " "));
  setText("measurementSelectedLegend", selectedLabel);
  const baseline = payload?.estimate_evidence?.provenance?.baseline || {};
  const baselineConsistency = String(
    baseline?.recorded_baseline_consistency_status || ""
  );
  const baselineLabel = selectedSource === "current_model_replay"
    ? `${String(baseline?.method || "session baseline").replaceAll("_", " ")}${
      baselineConsistency === "mismatch_warning" ? " | old baseline differs" : ""
    }`
    : selectedSource === "recorded_runtime"
      ? "capture-time baseline"
      : "not used";
  setText("measurementBaseline", baselineLabel);
  const evidenceElement = document.getElementById("measurementEvidence");
  if (evidenceElement) evidenceElement.title = selectedLabel;
  const validityElement = document.getElementById("measurementValidity");
  if (validityElement) validityElement.title = evaluationValidity.replaceAll("_", " ");
  const baselineElement = document.getElementById("measurementBaseline");
  if (baselineElement) {
    const normalizedDifference = Number(
      baseline?.recorded_baseline_normalized_rms_difference
    );
    baselineElement.title = Number.isFinite(normalizedDifference)
      ? `Stored-vs-session baseline RMS difference: ${(normalizedDifference * 100).toFixed(2)}%`
      : baselineLabel;
  }

  setText("measurementPairedSamples", String(data?.paired_count ?? 0));
  setText("measurementCycleCount", String(cycles.length));
  setText("measurementMae", formatMeasurementMetric(direct?.mae_n, 3, "N"));
  setText("measurementRmse", formatMeasurementMetric(direct?.rmse_n, 3, "N"));
  setText("measurementCorrelation", formatMeasurementMetric(direct?.pearson_r, 3));
  setText(
    "measurementAmplitudeSlope",
    formatMeasurementMetric(direct?.linear_slope_pred_vs_reference, 3),
  );
  setText("measurementLag", formatMeasurementMetric(lag?.lag_ms, 0, "ms"));
  setText("measurementAcquisitionRate", formatMeasurementMetric(cadence?.acquisition_rate_hz, 2, "Hz"));
  setText("measurementInferenceLatency", formatMeasurementMetric(cadence?.model_inference_latency_median_ms, 1, "ms"));
  setText("measurementRecovery", Number.isFinite(meanRecovery) ? `${(meanRecovery * 100).toFixed(1)}%` : "--");

  const comparisonStatus = String(data?.comparison_status || "unknown");
  const forceAvailable = Number(data?.valid_reference_count || 0) > 0;
  setMeasurementStatus(
    forceAvailable ? "Complete" : "Optical only",
    forceAvailable
      ? `${selectedLabel} | ${data?.paired_count ?? 0} aligned samples | ${cycles.length} detected cycle${cycles.length === 1 ? "" : "s"}`
      : "PX6D reference was not recorded; force comparison was skipped."
  );
  const output = document.getElementById("measurementOutput");
  if (output) output.title = comparisonStatus.replaceAll("_", " ");
  drawMeasurementComparison(payload?.trace || []);
}

function populateMeasurementSessions(sessions = [], selectedPath = "") {
  state.measurementSessions = Array.isArray(sessions) ? sessions : [];
  if (!measurementSessionSelect) return;
  measurementSessionSelect.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = state.measurementSessions.length
    ? "Select a completed recording"
    : "No completed recordings found";
  measurementSessionSelect.appendChild(placeholder);
  state.measurementSessions.forEach((session) => {
    const option = document.createElement("option");
    option.value = String(session?.path || "");
    const position = String(session?.position_label || "unlabeled");
    const trial = String(session?.trial_id || "--");
    const frames = Number(session?.captured_timeline_frames);
    option.textContent = `${position} | ${trial}${Number.isFinite(frames) ? ` | ${frames} frames` : ""}`;
    option.title = String(session?.session_id || session?.path || "");
    measurementSessionSelect.appendChild(option);
  });
  if (selectedPath && state.measurementSessions.some((session) => session?.path === selectedPath)) {
    measurementSessionSelect.value = selectedPath;
  }
  if (measurementAnalyzeButton) measurementAnalyzeButton.disabled = !measurementSessionSelect.value;
}

async function refreshMeasurementSessions() {
  if (state.measurementRequestInFlight) return;
  const root = String(measurementRootInput?.value || "").trim();
  const previousSelection = measurementSessionSelect?.value || "";
  setMeasurementBusy(true);
  setMeasurementStatus("Searching", "Scanning the selected folder for completed recordings.");
  try {
    const suffix = root ? `?root=${encodeURIComponent(root)}` : "";
    const payload = await requestJSON(
      `/api/measurement/sessions${suffix}`,
      { cache: "no-store" },
      { timeoutMs: 10000 }
    );
    if (measurementRootInput && !measurementRootInput.dataset.userEdited && payload?.root) {
      measurementRootInput.value = payload.root;
    }
    populateMeasurementSessions(payload?.sessions || [], previousSelection);
    clearMeasurementMetrics();
    state.measurementResult = null;
    drawMeasurementComparison([]);
    setMeasurementStatus(
      payload?.session_count ? "Ready" : "No sessions",
      payload?.session_count
        ? `${payload.session_count} completed recording${payload.session_count === 1 ? "" : "s"} found.`
        : "No frame_summary.csv or synchronized_frames.jsonl found in this folder."
    );
  } catch (error) {
    populateMeasurementSessions([]);
    setMeasurementStatus("Error", commandErrorMessage(error, "Unable to scan recordings"));
  } finally {
    setMeasurementBusy(false);
  }
}

async function analyzeSelectedMeasurementSession() {
  if (state.measurementRequestInFlight || !measurementSessionSelect?.value) return;
  setMeasurementBusy(true);
  const estimateSource = String(measurementEstimateSource?.value || "best_available");
  setMeasurementStatus("Analyzing", "Resolving one traceable optical-force estimate and comparing it with PX6D Fz.");
  try {
    const payload = await requestJSON(
      "/api/measurement/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_dir: measurementSessionSelect.value,
          estimate_source: estimateSource,
        }),
      },
      { timeoutMs: 120000 }
    );
    renderMeasurementResult(payload);
  } catch (error) {
    clearMeasurementMetrics();
    drawMeasurementComparison([]);
    setMeasurementStatus("Error", commandErrorMessage(error, "Measurement analysis failed"));
  } finally {
    setMeasurementBusy(false);
  }
}

function invalidatePx6dReferenceRequest() {
  if (state.px6dRequestController && !state.px6dRequestController.signal.aborted) {
    state.px6dRequestController.abort();
  }
}

function setCompactMetric(id, compactValue, fullValue = compactValue) {
  const element = document.getElementById(id);
  if (!element) return;
  const compact = compactValue === null || compactValue === undefined ? "--" : String(compactValue);
  const full = fullValue === null || fullValue === undefined ? compact : String(fullValue);
  element.textContent = compact;
  element.title = full !== compact ? full : "";
  element.setAttribute("aria-label", full);
}

function formatCompactNumber(value, digits = 1, signed = false) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  const prefix = signed && number > 0 ? "+" : "";
  const fixed = number.toFixed(digits).replace(/\.0+$/, "");
  return `${prefix}${fixed}`;
}

function setHealthState(id, value, tone = "neutral") {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = value;
  element.dataset.healthTone = tone;
}

function useDarkCanvas() {
  return appShell?.classList.contains("dark-operator-mode");
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(2)} s`;
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatPm(value, digits = 1, signed = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const number = Number(value);
  const prefix = signed && number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(digits)} pm`;
}

function levelClass(level) {
  const clean = String(level || "");
  if (clean.includes("hard") || clean.includes("large_shift")) return "hard";
  if (clean.includes("normal") || clean.includes("moderate_shift")) return "normal";
  if (clean.includes("light") || clean.includes("small_shift")) return "light";
  if (clean.includes("no_contact")) return "no-contact";
  if (clean.includes("anomaly") || clean.includes("required") || clean.includes("uncertain")) return "warning";
  return "";
}

function levelLabel(level) {
  const clean = String(level || "waiting");
  const labels = {
    no_contact: "no_contact",
    small_shift: "light",
    moderate_shift: "normal",
    large_shift: "hard",
    light_press: "light",
    normal_press: "normal",
    hard_press: "hard",
    baseline_required: "set reference baseline",
    wavelength_tracking_warning: "wavelength tracking warning",
    uncertain: "uncertain",
  };
  if (clean.startsWith("baseline_collecting")) return "collecting baseline";
  return labels[clean] || clean.replaceAll("_", " ");
}

function responseLevelFromSurfaceValue(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return "waiting";
  if (v < RESPONSE_BAND_THRESHOLDS.noContactMax) return "no_contact";
  if (v < RESPONSE_BAND_THRESHOLDS.smallMax) return "small_shift";
  if (v < RESPONSE_BAND_THRESHOLDS.moderateMax) return "moderate_shift";
  return "large_shift";
}

function normalizedSurfaceResponseRatio(surfaceMetrics = {}, record = null) {
  const value = Number(
    surfaceMetrics?.surface_peak ?? record?.wavelength_shift_response_ratio ?? record?.response_value
  );
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : Number.NaN;
}

function operatorQaLabel(value, record = {}) {
  const status = String(value || "").trim();
  if (status === "stale_frame") return "STALE";
  if (
    !status ||
    status === "ok" ||
    status === "synced" ||
    status === "simulated" ||
    status === "ok_with_manual_wavelength" ||
    status === "model_baseline_ready"
  ) {
    return "OK";
  }
  if (status.includes("baseline_required")) return "BASELINE";
  if (status === "model_low_confidence_warning") return "REVIEW";
  if (status === "warning") return "CHECK";
  return "REVIEW";
}

function operatorFreshnessLabel(value) {
  const key = String(value || "").toLowerCase();
  if (key.includes("fresh") || key === "live") return "fresh";
  if (key.includes("stale")) return "stale";
  if (key.includes("error")) return "error";
  if (key.includes("wait")) return "waiting";
  if (key.includes("stop")) return "stopped";
  return key.replace(/_/g, " ") || "waiting";
}

function acquisitionDisplayState(watcher = {}, sdkLive = {}) {
  if (state.demoModeActive) {
    const demoRecord = state.frame?.latest || {};
    const demoArrayFrame = state.frame?.array_frame || state.currentArrayFrame;
    if (state.arrayDemoActive && isRecordedDemoSpectrum(demoRecord, demoArrayFrame)) {
      return {
        label: "RECORDED",
        short: "REC",
        tone: "demo",
        detail: "Recorded 512-point spectrum with synchronized Fz; response surface is a visual proxy.",
      };
    }
    return {
      label: "SIMULATED",
      short: "SIM",
      tone: "demo",
      detail: "Simulated response. Not live data and not calibrated.",
    };
  }

  const sdkActive = Boolean(sdkLive?.active);
  const watchActive = Boolean(watcher?.active);
  if (sdkActive) {
    const freshness = operatorFreshnessLabel(sdkLive?.freshness || "waiting_for_sdk_frame");
    if (freshness === "fresh") return { label: "SDK | live", short: "LIVE", tone: "live", detail: "Fresh direct SDK frames" };
    if (freshness === "stale") return { label: "SDK | stale", short: "STALE", tone: "error", detail: "SDK is active but frames are stale" };
    if (freshness === "error") return { label: "SDK | error", short: "ERROR", tone: "error", detail: sdkLive?.last_error || "SDK acquisition error" };
    return { label: "SDK | waiting", short: "WAIT", tone: "waiting", detail: "Waiting for the first SDK frame" };
  }

  if (watchActive) {
    const freshness = operatorFreshnessLabel(watcher?.freshness || "waiting_for_export");
    if (freshness === "fresh") return { label: "Watch | live", short: "LIVE", tone: "live", detail: "Fresh Sense export frames" };
    if (freshness === "stale") return { label: "Watch | stale", short: "STALE", tone: "error", detail: "Sense export data is stale" };
    if (freshness === "error") return { label: "Watch | error", short: "ERROR", tone: "error", detail: watcher?.last_error || "Export watch error" };
    return { label: "Watch | waiting", short: "WAIT", tone: "waiting", detail: "Waiting for a new Sense export" };
  }

  if (state.liveRequested) {
    return { label: "SDK | waiting", short: "WAIT", tone: "waiting", detail: "Live start accepted; waiting for acquisition" };
  }

  const idleLabels = {
    direct_sdk: "SDK | idle",
    export_watch: "Watch | idle",
    http_ingest: "HTTP | idle",
  };
  return {
    label: idleLabels[inputSourceSelect?.value] || "Input | idle",
    short: "IDLE",
    tone: "idle",
    detail: "Acquisition stopped",
  };
}

function frameHasMeasurement(frame) {
  const record = frame?.latest;
  if (!record || frame?.mode === "operator_idle" || record?.source === "operator_idle") return false;
  if (frame?.array_frame?.mode === "simulated_array_demo") return true;
  if (frame?.scope === GLOBAL_RECOGNITION_SCOPE) {
    const spectrum = spectrumArrays(record);
    return spectrum.intensity.length > 0 && spectrum.xValues.length === spectrum.intensity.length;
  }
  return [record.tracked_wavelength_nm, record.delta_wavelength_pm, record.wavelength_shift_response_ratio].some(
    (value) => value !== null && value !== undefined && Number.isFinite(Number(value))
  );
}

function frameSourceIsFresh(frame) {
  if (frame?.array_frame?.mode === "simulated_array_demo") return true;
  if (frame?.scope === GLOBAL_RECOGNITION_SCOPE) {
    return frame?.global_frame_qa?.operator_display_valid === true;
  }
  const qaStatus = String(frame?.latest?.qa_status || "").toLowerCase();
  return !qaStatus.includes("stale");
}

function currentModelDisplayReady(frame) {
  const contract = operatorVisualizationContract(frame, frame?.latest, frame?.array_frame);
  return contract?.response_allowed === true && Boolean(contract.prediction);
}

function frameResponseIsUsable(frame) {
  if (!frameHasMeasurement(frame)) return false;
  if (frame?.array_frame?.mode === "simulated_array_demo") return true;
  if (!frameSourceIsFresh(frame)) return false;
  if (frame?.scope === GLOBAL_RECOGNITION_SCOPE) return frame?.latest?.response_allowed === true;
  return true;
}

function zeroSurfaceGridLike(grid) {
  if (!Array.isArray(grid)) return [];
  return grid.map((row) => (Array.isArray(row) ? row.map(() => 0) : []));
}

function neutralResponseRecord(record = {}) {
  return {
    ...record,
    diagnostic_raw_response: {
      response_value: record?.response_value ?? null,
      wavelength_shift_response_ratio: record?.wavelength_shift_response_ratio ?? null,
      delta_wavelength_pm: record?.delta_wavelength_pm ?? null,
      absolute_shift_pm: record?.absolute_shift_pm ?? null,
      response_level: record?.response_level ?? null,
      model_position_id: record?.model_position_id ?? null,
      model_force_level: record?.model_force_level ?? null,
    },
    response_allowed: false,
    response_value: 0,
    wavelength_shift_response_ratio: 0,
    delta_wavelength_pm: 0,
    absolute_shift_pm: 0,
    response_level: "no_contact",
    dominant_channel: null,
    provisional_dominant_channel: null,
    model_position_id: null,
    model_position_confidence: null,
    model_force_level: null,
    model_force_confidence: null,
  };
}

function atomicNeutralFrame(frame) {
  if (!frame || !frameHasMeasurement(frame) || frameResponseIsUsable(frame)) return frame;
  const sourceArrayFrame = frame?.array_frame || {};
  const sourceMetrics = sourceArrayFrame?.surface_metrics || frame?.surface_metrics || {};
  const sourceGrid = sourceArrayFrame?.surface_grid || frame?.surface_grid || [];
  const responseBlockReason =
    frame?.latest?.response_block_reason ||
    sourceArrayFrame?.response_block_reason ||
    "operator_response_not_usable";
  const channels = (sourceArrayFrame?.channels || frame?.channel_grid || []).map((channel) => ({
    ...channel,
    diagnostic_raw_response_value: channel?.response_value ?? channel?.wavelength_shift_response_ratio ?? null,
    response_value: 0,
    wavelength_shift_response_ratio: 0,
    response_enabled: false,
  }));
  const surfaceMetrics = {
    ...sourceMetrics,
    surface_peak: 0,
    surface_mean: 0,
    surface_area_active: 0,
    surface_area_active_percent: 0,
    surface_centroid_x: null,
    surface_centroid_y: null,
    surface_spread: 0,
    dominant_channel: null,
    provisional_dominant_channel: null,
    responding_channel_count: 0,
    responding_channel_ids: [],
    responding_candidate_ids: [],
    contact_evidence_passed: false,
    quality_status: responseBlockReason,
  };
  const surfaceGrid = zeroSurfaceGridLike(sourceGrid);
  const sourceTrace = Array.isArray(frame?.trace) ? frame.trace : [];
  const neutralTrace = sourceTrace.length
    ? sourceTrace.map((item, index) =>
        index === sourceTrace.length - 1 ? neutralResponseRecord(item) : item
      )
    : [];
  const arrayFrame = {
    ...sourceArrayFrame,
    mode: "invalid_input_neutral",
    presentation_status: "invalid_input_neutral",
    response_allowed: false,
    response_block_reason: responseBlockReason,
    surface_metrics: surfaceMetrics,
    surface_grid: surfaceGrid,
    channels,
    dominant_channel: null,
    peak_shift_response: 0,
    peak_wavelength_shift_pm: 0,
  };
  return {
    ...frame,
    model_assisted_display_allowed: false,
    operator_frame_atomic_neutralized: true,
    presentation_status: "invalid_input_neutral",
    latest: neutralResponseRecord(frame.latest),
    trace: neutralTrace,
    channel_grid: channels,
    array_frame: arrayFrame,
    surface_grid: surfaceGrid,
    surface_metrics: surfaceMetrics,
    diagnostic_raw_array_frame: sourceArrayFrame,
  };
}

function invalidateFrameRequestContext() {
  state.frameModeEpoch += 1;
  if (state.frameRequestController && !state.frameRequestController.signal.aborted) {
    state.frameRequestController.abort();
  }
}

function setDataSourceDisplay(sourceState) {
  const sourceChip = document.getElementById("sourceChip");
  if (!sourceChip || !sourceState) return;
  sourceChip.textContent = sourceState.label;
  sourceChip.dataset.sourceTone = sourceState.tone;
  sourceChip.title = sourceState.detail;
}

function updateSurfaceFrameState(frame, watcher = {}, sdkLive = {}) {
  const sourceState = acquisitionDisplayState(watcher, sdkLive);
  const heldMeasurement = sourceState.tone === "idle" && frameHasMeasurement(frame);
  const label = state.paused ? "HOLD" : heldMeasurement ? "HELD" : sourceState.short || "IDLE";
  const tone = state.paused ? "paused" : heldMeasurement ? "hold" : sourceState.tone;
  setText("surfaceFrameState", label);
  const element = document.getElementById("surfaceFrameState");
  if (element) element.dataset.streamTone = tone;
}

function updateOperatorStreamSummary() {
  const traceChip = document.getElementById("traceChip");
  if (traceChip) {
    traceChip.hidden = !state.paused;
    traceChip.textContent = state.paused ? "PAUSED" : "";
    traceChip.dataset.streamTone = state.paused ? "paused" : "idle";
    traceChip.title = state.paused ? "Display paused" : "";
  }
}

function operatorAlertState({ record, sourceState, operatorQa, measurementAvailable }) {
  const qaFlags = Array.isArray(record?.qa_flags)
    ? record.qa_flags.map((item) => String(item).toLowerCase())
    : String(record?.qa_flags || "")
        .split(",")
        .map((item) => item.trim().toLowerCase())
        .filter(Boolean);
  const responseLevel = String(record?.response_level || "").toLowerCase();
  const qaStatus = String(record?.qa_status || "").toLowerCase();
  const compactQa = String(operatorQa || "").toLowerCase();
  const globalFrame = record?.recognition_scope === GLOBAL_RECOGNITION_SCOPE;
  const globalBaselineReady = globalFrame && record?.global_frame_qa?.baseline_ready === true;

  if (sourceState?.tone === "error") {
    return { tone: "error", severity: "ALARM", message: `${sourceState.label}. Check the BaySpec connection.` };
  }
  if (
    measurementAvailable &&
    !globalBaselineReady &&
    (responseLevel === "baseline_required" || String(record?.baseline_status || "").includes("required"))
  ) {
    return {
      tone: "warning",
      severity: "WARNING",
      message: globalFrame
        ? "Global FBG01-FBG09 baseline required. Remove contact, wait for a stable spectrum, then press Set baseline."
        : "Reference baseline required. Remove contact, then press Set baseline.",
    };
  }
  if (
    measurementAvailable &&
    qaFlags.some((flag) => flag.includes("correlation") || flag.includes("wavelength_jump") || flag.includes("tracking"))
  ) {
    return { tone: "warning", severity: "WARNING", message: "Bragg wavelength tracking needs review. Open Diagnostics." };
  }
  if (measurementAvailable && (qaStatus === "invalid" || compactQa.includes("review"))) {
    return { tone: "error", severity: "ALARM", message: "Frame rejected by QA. Review Diagnostics." };
  }
  if (
    measurementAvailable &&
    (qaStatus === "model_low_confidence_warning" || compactQa.includes("model confidence"))
  ) {
    return {
      tone: "warning",
      severity: "RECOGNITION",
      message: "Recognition confidence is low. Hold the contact briefly or review model confidence in Diagnostics.",
    };
  }
  if (measurementAvailable && (qaStatus === "warning" || compactQa.includes("warning"))) {
    return { tone: "warning", severity: "WARNING", message: "Signal quality warning. Review Diagnostics." };
  }
  if (sourceState?.tone === "waiting") {
    return { tone: "waiting", severity: "WAITING", message: "Waiting for the first spectrum frame." };
  }
  return null;
}

function updateOperatorAlert(context) {
  const alertState = operatorAlertState(context);
  operatorDiagnosticsAttentionFromQa = ["warning", "error"].includes(alertState?.tone);
  refreshDiagnosticsAttention();
  if (operatorDiagnosticsButton) {
    operatorDiagnosticsButton.dataset.tooltip =
      operatorDiagnosticsAttentionFromQa || Boolean(desktopWindowCommandFailure)
      ? "Diagnostics - attention"
      : "Diagnostics";
  }
  if (operatorAlert) {
    operatorAlert.classList.remove("visible");
    operatorAlert.setAttribute("aria-hidden", "true");
    delete operatorAlert.dataset.tone;
  }
}

function traceAttenuationPercent(item) {
  const modelResponseRatio = Number(item?.trained_model_visual_response_ratio);
  if (Number.isFinite(modelResponseRatio)) {
    return Math.max(0, Math.min(100, modelResponseRatio * 100));
  }
  if (item?.visualization_contract_version === OPERATOR_VISUALIZATION_CONTRACT_VERSION) {
    return Number.NaN;
  }
  const clampTraceValue = (value) => Math.max(-WAVELENGTH_SHIFT_FULL_SCALE_PM, Math.min(WAVELENGTH_SHIFT_FULL_SCALE_PM, value));
  if (item?.recognition_scope === GLOBAL_RECOGNITION_SCOPE) {
    const eventShiftPm = Number(item?.global_event_absolute_shift_pm);
    if (Number.isFinite(eventShiftPm)) return clampTraceValue(Math.max(0, eventShiftPm));
  }
  const globalShifts = globalCandidatePeaks(item)
    .map((peak) => Math.abs(candidateShiftPm(peak)))
    .filter(Number.isFinite);
  if (globalShifts.length === GLOBAL_CANDIDATE_IDS.length) {
    return clampTraceValue(Math.max(...globalShifts));
  }
  const directRaw = item?.delta_wavelength_pm ?? item?.peak_wavelength_shift_pm;
  const direct = directRaw == null ? Number.NaN : Number(directRaw);
  if (Number.isFinite(direct)) return clampTraceValue(direct);
  const trackedRaw = item?.tracked_wavelength_nm ?? item?.peak_wavelength_nm;
  const baselineRaw = item?.baseline_wavelength_nm;
  const tracked = trackedRaw == null ? Number.NaN : Number(trackedRaw);
  const baseline = baselineRaw == null ? Number.NaN : Number(baselineRaw);
  if (Number.isFinite(tracked) && Number.isFinite(baseline)) {
    return clampTraceValue((tracked - baseline) * 1000);
  }
  const responseRaw = item?.wavelength_shift_response_ratio ?? item?.response_value ?? item?.surface_peak;
  const responseRatio = responseRaw == null ? Number.NaN : Number(responseRaw);
  if (Number.isFinite(responseRatio)) return clampTraceValue(responseRatio * WAVELENGTH_SHIFT_FULL_SCALE_PM);
  return Number.NaN;
}

function traceOpticalForceN(item) {
  const rawValue = item?.optical_force_trace_n;
  if (rawValue === null || rawValue === undefined || rawValue === "") return Number.NaN;
  const value = Number(rawValue);
  return Number.isFinite(value)
    ? Math.max(0, value)
    : Number.NaN;
}

function traceDisplayValue(item) {
  const forceN = traceOpticalForceN(item);
  return Number.isFinite(forceN) ? forceN : traceAttenuationPercent(item);
}

function quantile(sortedValues, q) {
  if (!Array.isArray(sortedValues) || !sortedValues.length) return Number.NaN;
  const clamped = Math.max(0, Math.min(1, Number(q) || 0));
  const index = (sortedValues.length - 1) * clamped;
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (index - lo);
}

function globalEventResponseFromTrace(currentAbsShiftPm, trace = []) {
  const current = Number(currentAbsShiftPm);
  if (!Number.isFinite(current) || current <= 0) {
    return { eventShiftPm: 0, residualFloorPm: 0, residualCompensated: true };
  }
  const recent = (Array.isArray(trace) ? trace : [])
    .slice(-TRACE_WINDOW_POINTS)
    .map((item) => Math.abs(traceAttenuationPercent(item)))
    .filter(Number.isFinite)
    .filter((value) => value >= 0);
  if (recent.length < 6) {
    return { eventShiftPm: current, residualFloorPm: 0, residualCompensated: false };
  }
  const sorted = recent.slice().sort((a, b) => a - b);
  const residualFloorPm = Math.max(0, quantile(sorted, GLOBAL_EVENT_RESIDUAL_QUANTILE));
  const eventShiftPm = Math.max(0, current - residualFloorPm - GLOBAL_EVENT_DEADBAND_PM);
  return {
    eventShiftPm,
    residualFloorPm,
    residualCompensated: true,
  };
}

function recordFromAttenuationPercent(percent, timestamp = Date.now(), channelId = "P22") {
  const responseRatio = Math.max(0, Math.min(100, Number(percent) || 0)) / 100;
  const shiftPm = responseRatio * WAVELENGTH_SHIFT_FULL_SCALE_PM;
  const trackedWavelength = DEMO_BASELINE_WAVELENGTH_NM + shiftPm / 1000;
  return {
    timestamp,
    channel_id: channelId,
    intensity_counts: DEMO_BASELINE,
    baseline_intensity_counts: DEMO_BASELINE,
    baseline_wavelength_nm: DEMO_BASELINE_WAVELENGTH_NM,
    tracked_wavelength_nm: trackedWavelength,
    peak_wavelength_nm: trackedWavelength,
    delta_wavelength_nm: shiftPm / 1000,
    delta_wavelength_pm: shiftPm,
    absolute_shift_pm: Math.abs(shiftPm),
    wavelength_shift_response_ratio: responseRatio,
    response_value: responseRatio,
    shift_direction: shiftPm > 0 ? "red_shift" : shiftPm < 0 ? "blue_shift" : "stable",
    response_level: responseLevelFromSurfaceValue(responseRatio),
  };
}

function makeIdleFrame() {
  const now = Date.now();
  const latest = {
    ...recordFromAttenuationPercent(0, now, state.selectedChannel),
    channel_id: null,
    recognition_scope: GLOBAL_RECOGNITION_SCOPE,
    candidate_contract_version: "global_9fbg_candidate_frame_v1",
    global_candidate_ids: [...GLOBAL_CANDIDATE_IDS],
    physical_channel_mapping_final: false,
    intensity_counts: null,
    baseline_intensity_counts: null,
    baseline_wavelength_nm: null,
    tracked_wavelength_nm: null,
    delta_wavelength_nm: null,
    delta_wavelength_pm: null,
    absolute_shift_pm: null,
    wavelength_shift_response_ratio: null,
    source: "operator_idle",
    response_level: "idle",
    baseline_status: "idle",
    qa_status: "ok_with_manual_wavelength",
    target_wavelength_nm: DEMO_TARGET_WAVELENGTH_NM,
    measured_wavelength_nm: 1546.89,
    demodulation_wavelength_nm: 1546.89,
    measured_wavelength_source: "manual_user_observation",
    measured_wavelength_status: "provisional_manual",
    peak_source: "manual_measured_wavelength_override",
    spectrum_unavailable_reason: "Idle state. Start live, watch, ingest, or select a response scenario to show spectrum evidence.",
  };
  const channelGrid = GLOBAL_CANDIDATE_IDS.map((candidateId) => ({
    channel_id: candidateId,
    candidate_id: candidateId,
    enabled: true,
    valid: false,
    qa_status: "no_data",
    response_level: "no_data",
    wavelength_shift_response_ratio: null,
    delta_wavelength_pm: null,
    tracked_wavelength_nm: null,
    intensity_counts: null,
    baseline_intensity_counts: null,
  }));
  const surfaceMetrics = {
    surface_peak: null,
    surface_mean: null,
    surface_area_active: null,
    surface_centroid_x: null,
    surface_centroid_y: null,
    surface_spread: null,
    dominant_channel: null,
    enabled_channel_count: 9,
    responding_channel_count: 0,
    responding_channel_ids: [],
    quality_status: "idle",
    coupling_status: "global fingerprint idle",
    event_interpretation: "idle / waiting for global 9-FBG spectrum",
    num_changed_peaks: null,
  };
  return {
    ok: true,
    mode: "operator_idle",
    scope: GLOBAL_RECOGNITION_SCOPE,
    candidate_contract_version: "global_9fbg_candidate_frame_v1",
    global_candidate_ids: [...GLOBAL_CANDIDATE_IDS],
    physical_channel_mapping_final: false,
    frame_id: "idle",
    timestamp: now,
    selected_channel: null,
    latest,
    trace: [],
    channel_grid: channelGrid,
    array_frame: {
      mode: "global_spectrum_idle",
      frame_id: "idle",
      timestamp: now,
      surface_metrics: surfaceMetrics,
      surface_grid: [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      channels: channelGrid,
      coupling_status: "global fingerprint idle",
    },
    surface_grid: [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    surface_metrics: surfaceMetrics,
    status: { buffered_records: 0, channels_seen: [] },
    export_watcher: { active: false, freshness: "idle" },
    sdk_live: { active: false, freshness: "idle" },
    sense_control: {},
  };
}

function clearStoppedLivePresentation() {
  invalidateFrameRequestContext();
  resetRuntimeTraceHistory();
  state.dataStreamActive = false;
  state.exportWatchActive = false;
  state.sdkLiveActive = false;
  state.liveRequested = false;
  const idleFrame = makeIdleFrame();
  state.frame = idleFrame;
  updateUI(idleFrame);
  snapDisplayedFrameToCurrentTargets();
}

function responseText(record) {
  if (!record) return "Waiting for decoded BaySpec Bragg wavelength input.";
  const qaFlags = Array.isArray(record.qa_flags) ? record.qa_flags : [];
  if (record?.recognition_scope === GLOBAL_RECOGNITION_SCOPE) {
    const contract = operatorVisualizationContract(record);
    const prediction = contract?.prediction || null;
    const activeStatus = String(contract?.prediction_status || record?.active_spectral_model_status || "");
    const modelName = activeModelDisplayName();
    if (contract?.response_allowed === true && prediction?.digital_twin) {
      const twin = prediction.digital_twin;
      if ((twin.visual_active ?? twin.active) !== true) {
        return `${modelName}: no active contact; deformation is suppressed.`;
      }
      const position =
        prediction?.position?.visual_label ||
        prediction?.position?.label ||
        "unknown position";
      const estimatedForce = finiteNumberOrNull(contract?.force?.display_n);
      return estimatedForce !== null
        ? `${modelName}: ${position}; optical Fz estimate ${estimatedForce.toFixed(2)} N.`
        : `${modelName}: ${position}; optical Fz estimate unavailable.`;
    }
    if (activeStatus === "baseline_required") {
      return `${modelName}: set a stable no-contact baseline.`;
    }
    if (["current_runtime_model_source_blocked", "spectrum_required"].includes(activeStatus)) {
      return `${modelName}: waiting for a fresh wavelength-grid frame.`;
    }
    if (record?.active_spectral_model_loaded === true) {
      return `${modelName}: input waiting (${activeStatus || "no frame"}).`;
    }
    return "Current deployed model unavailable. Diagnostics contains the load error.";
  }
  if (record.qa_status === "invalid") {
    return `Frame QA invalid: ${qaFlags.join(", ") || "quality rule failed"}. Do not use this frame as a press response.`;
  }
  if (record.peak_axis_type === "pixel_index") {
    return "Wavelength calibration is missing. Pixel-index spectra cannot provide physical wavelength response.";
  }
  if (record.response_level === "baseline_required") {
    return record?.hybrid_spectral_response_available
      ? "Full spectrum received. Set a no-contact baseline before comparing wavelength, intensity, area, and shape features."
      : "Spectrum received. Set a no-contact baseline before interpreting Bragg wavelength shift.";
  }
  if (record.qa_status === "warning") {
    return `Frame QA warning: ${qaFlags.join(", ") || "review peak tracking"}. Wavelength shift remains an uncalibrated strain/temperature-coupled optical response.`;
  }
  if (record?.hybrid_spectral_response_available) {
    return "Measured response contains mixed wavelength, intensity, area, and spectral-shape changes. Point identity requires the deployed full-spectrum model.";
  }
  return "Diagnostic response: signed Bragg wavelength shift is not calibrated force and remains strain/temperature coupled.";
}
function liveResponseText(record, watcher, sdkLive) {
  const sdkFreshness = String(sdkLive?.freshness || "");
  if (sdkLive?.active && sdkFreshness === "stale") {
    return "Direct SDK stream is stale: the BaySpec helper is running but no fresh spectrum frame arrived recently.";
  }
  const freshness = String(watcher?.freshness || "");
  if (watcher?.active && freshness === "stale") {
    return "Live file is stale: Sense is not writing new DAT frames now. Restart fast recording or Start live twin before expecting press response.";
  }
  return responseText(record);
}

function wavelengthSummary(record) {
  const target = Number(record?.target_wavelength_nm);
  const measured = Number(record?.measured_wavelength_nm);
  const demod = Number(record?.demodulation_wavelength_nm);
  const selected = Number(record?.tracked_wavelength_nm ?? record?.peak_wavelength_nm);
  const pixelIndex = Number(record?.peak_pixel_index);
  if (record?.peak_axis_type === "pixel_index") {
    return Number.isFinite(pixelIndex) ? `pixel ${Math.round(pixelIndex)}; no wavelength grid` : "pixel axis; no wavelength grid";
  }
  if (record?.peak_source === "manual_measured_wavelength_override" && Number.isFinite(demod) && Number.isFinite(selected)) {
    return `manual ${demod.toFixed(3)} / peak ${selected.toFixed(3)} nm`;
  }
  if (Number.isFinite(target) && Number.isFinite(measured) && Number.isFinite(selected)) {
    return `target ${target.toFixed(1)} / measured ${measured.toFixed(3)} / peak ${selected.toFixed(3)} nm`;
  }
  if (Number.isFinite(target) && Number.isFinite(selected)) {
    return `target ${target.toFixed(1)} / peak ${selected.toFixed(3)} nm`;
  }
  if (Number.isFinite(selected)) return `${selected.toFixed(3)} nm`;
  return "--";
}

function attenuation(record) {
  const value = Number(record?.wavelength_shift_response_ratio ?? record?.response_value);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function smoothStep(edge0, edge1, value) {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function visualDeformationStrength(record) {
  const raw = attenuation(record);
  if (raw < 0.018) return 0;
  if (raw < 0.30) {
    return 0.14 + 0.20 * smoothStep(0.05, 0.30, raw);
  }
  if (raw < 0.70) {
    return 0.46 + 0.26 * smoothStep(0.30, 0.70, raw);
  }
  return 0.88 + 0.12 * smoothStep(0.70, 0.92, raw);
}

function colorForAttenuation(value) {
  const t = clamp01(value);
  const color = new THREE.Color();
  if (t < 0.12) {
    return color.lerpColors(new THREE.Color("#7fb9c8"), new THREE.Color("#5aaea6"), t / 0.12);
  }
  if (t < 0.34) {
    return color.lerpColors(new THREE.Color("#5aaea6"), new THREE.Color("#4aa3a2"), (t - 0.12) / 0.22);
  }
  if (t < 0.58) {
    return color.lerpColors(new THREE.Color("#4aa3a2"), new THREE.Color("#d9b86c"), (t - 0.34) / 0.24);
  }
  if (t < 0.80) {
    return color.lerpColors(new THREE.Color("#d9b86c"), new THREE.Color("#cf8a5d"), (t - 0.58) / 0.22);
  }
  return color.lerpColors(new THREE.Color("#cf8a5d"), new THREE.Color("#9f3d37"), (t - 0.80) / 0.20);
}

function visualColorStrength(rawAttenuation, localResponseProfile = 1) {
  const raw = clamp01(rawAttenuation);
  const profile = clamp01(localResponseProfile);
  if (raw < 0.055) return 0.018 + 0.018 * profile;

  const radial = Math.pow(profile, 0.58);
  const lightLift = 0.28 * radial * smoothStep(0.055, 0.13, raw);
  const shoulder = 0.11 * raw * Math.pow(profile, 0.24);
  const centerBoost = 0.18 * radial * smoothStep(0.08, 0.72, raw);
  return clamp01(raw * (0.16 + 0.92 * radial) + lightLift + shoulder + centerBoost);
}

function visualSurfacePeakForFrame(rawPeak, record, arrayFrame) {
  const raw = clamp01(Number.isFinite(rawPeak) ? rawPeak : attenuation(record));
  const scenario = String(arrayFrame?.scenario || "");
  const isSimulatedArrayDemo = String(arrayFrame?.mode || "") === "simulated_array_demo";
  if (!isSimulatedArrayDemo || scenario === "no_contact") {
    return raw;
  }
  const floor = DEMO_SURFACE_VISUAL_PEAK_FLOORS[scenario];
  if (!Number.isFinite(floor)) return raw;
  const floorActivation = smoothStep(0.035, 0.18, raw);
  return Math.max(raw, floor * floorActivation);
}

function visualDeformationFromSurfacePeak(visualPeak, rawPeak = visualPeak) {
  const visual = clamp01(visualPeak);
  const raw = clamp01(rawPeak);
  if (raw < 0.035 && visual < 0.10) return 0;
  if (visual < 0.055) return 0;
  return clamp01(0.20 + 0.82 * smoothStep(0.055, 0.86, visual));
}

function currentSurfaceVisualPeak() {
  const metricPeak = Number(state.currentSurfaceMetrics?.surface_peak);
  return Math.max(
    0,
    clamp01(Number.isFinite(metricPeak) ? metricPeak : 0),
    clamp01(state.smoothSurfaceVisualPeak || 0),
    clamp01(state.targetSurfaceVisualPeak || 0)
  );
}

function dampToward(current, target, lambda, deltaSeconds) {
  if (!Number.isFinite(current)) return target;
  if (!Number.isFinite(target)) return current;
  const dt = Math.max(0, Math.min(0.05, deltaSeconds));
  return current + (target - current) * (1 - Math.exp(-lambda * dt));
}

function cloneTraceRecords(records = []) {
  return (records || []).map((item) => ({
    ...item,
    intensity_counts: Number(item?.intensity_counts),
    baseline_intensity_counts: Number(item?.baseline_intensity_counts),
    trained_model_visual_response_ratio: Number(item?.trained_model_visual_response_ratio),
    optical_force_trace_n: item?.optical_force_trace_n === null || item?.optical_force_trace_n === undefined
      ? null
      : Number(item.optical_force_trace_n),
  }));
}

function alignSmoothTraceRecords(previous, target) {
  const previousRecords = Array.isArray(previous) ? previous : [];
  const offset = target.length - previousRecords.length;
  return target.map((item, index) => {
    const prior = previousRecords[index - offset];
    const priorValue = Number(prior?.intensity_counts);
    const targetValue = Number(item?.intensity_counts);
    const priorModelRatio = Number(prior?.trained_model_visual_response_ratio);
    const targetModelRatio = Number(item?.trained_model_visual_response_ratio);
    const priorForceN = traceOpticalForceN(prior);
    const targetForceN = traceOpticalForceN(item);
    return {
      ...item,
      intensity_counts: Number.isFinite(priorValue) ? priorValue : targetValue,
      trained_model_visual_response_ratio: Number.isFinite(priorModelRatio)
        ? priorModelRatio
        : targetModelRatio,
      optical_force_trace_n: Number.isFinite(priorForceN)
        ? priorForceN
        : targetForceN,
    };
  });
}

function cloneSpectrumRecord(record) {
  if (!record) return null;
  const clone = { ...record };
  [
    "wavelength_nm",
    "spectrum_wavelength_nm",
    "intensity",
    "spectrum_counts",
    "display_intensity",
    "overlay_intensity",
  ].forEach((key) => {
    if (Array.isArray(record?.[key])) {
      clone[key] = record[key].map((value) => Number(value));
    }
  });
  if (Array.isArray(record?.spectrum_peaks)) {
    clone.spectrum_peaks = record.spectrum_peaks.map((item) => ({ ...item }));
  }
  clone.intensity_counts = Number(record?.intensity_counts);
  return clone;
}

function alignSmoothSpectrumRecord(previous, target) {
  // Never blend spectrum bins vertically. When a peak moves, per-bin fading
  // makes the old peak shrink and the new peak grow, which falsely resembles
  // intensity demodulation. Each acquired/demo spectrum replaces the prior one.
  return cloneSpectrumRecord(target);
}

function setChartTargets(trace, record) {
  state.targetTraceRecords = cloneTraceRecords(trace);
  state.targetSpectrumRecord = cloneSpectrumRecord(record);
  if (!state.smoothTraceRecords.length || state.smoothTraceRecords.length !== state.targetTraceRecords.length) {
    state.smoothTraceRecords = alignSmoothTraceRecords(state.smoothTraceRecords, state.targetTraceRecords);
  }
  state.smoothSpectrumRecord = alignSmoothSpectrumRecord(state.smoothSpectrumRecord, state.targetSpectrumRecord);
  state.chartsNeedRefresh = true;
  requestThreeAnimation();
}

function drawVisibleCharts() {
  drawTrace(state.smoothTraceRecords);
  // The compact Operator preview is independent from the full spectrum drawer.
  // Keep it live at all times while the heavier overview/zoom canvases remain
  // gated behind Diagnostics or the explicitly opened drawer.
  drawOpticalPreview(state.smoothSpectrumRecord);
  const diagnosticsVisible = state.displayMode === "diagnostics";
  const spectrumVisible = diagnosticsVisible || spectrumDrawer?.classList.contains("open");
  if (spectrumVisible) {
    drawSpectrum(state.smoothSpectrumRecord);
    drawSelectedPeakZoom(state.smoothSpectrumRecord);
  }
  if (diagnosticsVisible) drawHeatmap(state.currentArrayFrame);
}

function updateChartSmoothing(deltaSeconds) {
  if (!state.chartsNeedRefresh) return;
  let maxDelta = 0;

  if (state.targetTraceRecords.length) {
    if (state.smoothTraceRecords.length !== state.targetTraceRecords.length) {
      state.smoothTraceRecords = alignSmoothTraceRecords(state.smoothTraceRecords, state.targetTraceRecords);
    }
    state.smoothTraceRecords = state.targetTraceRecords.map((item, index) => {
      const isLatestPoint = index === state.targetTraceRecords.length - 1;
      const targetValue = Number(item?.intensity_counts);
      const currentValue = Number(state.smoothTraceRecords[index]?.intensity_counts);
      let nextValue = Number.isFinite(currentValue) ? currentValue : targetValue;
      const targetShiftPm = traceAttenuationPercent(item);
      const currentShiftPm = traceAttenuationPercent(state.smoothTraceRecords[index]);
      const modelResponseTrace = Number.isFinite(Number(item?.trained_model_visual_response_ratio));
      const targetForceN = traceOpticalForceN(item);
      let nextShiftPm = Number.isFinite(currentShiftPm) ? currentShiftPm : targetShiftPm;
      if (Number.isFinite(targetValue)) {
        nextValue = isLatestPoint ? targetValue : dampToward(nextValue, targetValue, CHART_EASING, deltaSeconds);
        const delta = Math.abs(targetValue - nextValue);
        if (delta < CHART_SETTLE_COUNTS) nextValue = targetValue;
        maxDelta = Math.max(maxDelta, delta);
      }
      if (Number.isFinite(targetShiftPm)) {
        nextShiftPm = isLatestPoint
          ? targetShiftPm
          : dampToward(nextShiftPm, targetShiftPm, CHART_EASING, deltaSeconds);
        const shiftDelta = Math.abs(targetShiftPm - nextShiftPm);
        if (shiftDelta < 0.08) nextShiftPm = targetShiftPm;
        maxDelta = Math.max(maxDelta, shiftDelta);
      }
      const nextRatio = Number.isFinite(nextShiftPm)
        ? Math.max(0, Math.min(1, Math.abs(nextShiftPm) / (modelResponseTrace ? 100 : WAVELENGTH_SHIFT_FULL_SCALE_PM)))
        : item?.wavelength_shift_response_ratio;
      const baselineWavelength = Number(item?.baseline_wavelength_nm);
      const trackedWavelength = Number.isFinite(nextShiftPm) && Number.isFinite(baselineWavelength)
        ? baselineWavelength + nextShiftPm / 1000
        : item?.tracked_wavelength_nm;
      return {
        ...item,
        intensity_counts: nextValue,
        trained_model_visual_response_ratio: modelResponseTrace
          ? nextRatio
          : item?.trained_model_visual_response_ratio,
        // Keep quantitative force evidence on the acquisition cadence. Extra
        // chart easing would hide low-force changes and add display latency.
        optical_force_trace_n: Number.isFinite(targetForceN)
          ? targetForceN
          : item?.optical_force_trace_n,
        tracked_wavelength_nm: trackedWavelength,
        delta_wavelength_pm: Number.isFinite(nextShiftPm) ? nextShiftPm : item?.delta_wavelength_pm,
        delta_wavelength_nm: Number.isFinite(nextShiftPm) ? nextShiftPm / 1000 : item?.delta_wavelength_nm,
        absolute_shift_pm: Number.isFinite(nextShiftPm) ? Math.abs(nextShiftPm) : item?.absolute_shift_pm,
        wavelength_shift_response_ratio: nextRatio,
        response_value: nextRatio,
      };
    });
  }

  if (state.targetSpectrumRecord) {
    state.smoothSpectrumRecord = cloneSpectrumRecord(state.targetSpectrumRecord);
  }

  drawVisibleCharts();
  state.chartsNeedRefresh = maxDelta > CHART_SETTLE_COUNTS;
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawGrid(ctx, width, height) {
  const darkCanvas = useDarkCanvas();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = darkCanvas ? "#030a15" : "#fbfdff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = darkCanvas ? "rgba(78, 156, 215, 0.20)" : "#dbe9f3";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i += 1) {
    const y = (height * i) / 6;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 0; i <= 7; i += 1) {
    const x = (width * i) / 7;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function updateTraceKpis(records) {
  const items = Array.isArray(records) ? records : [];
  const metricItems = state.targetTraceRecords.length ? state.targetTraceRecords : items;
  const opticalForceTrace = metricItems.some((item) => Number.isFinite(traceOpticalForceN(item)));
  const values = metricItems.map(traceDisplayValue).filter(Number.isFinite);
  const modelResponseTrace = !opticalForceTrace && metricItems.some((item) =>
    Number.isFinite(Number(item?.trained_model_visual_response_ratio))
  );
  const measurementAvailable = opticalForceTrace
    ? frameHasMeasurement(state.frame) && frameSourceIsFresh(state.frame)
    : frameResponseIsUsable(state.frame);
  const globalFrame = state.frame?.scope === GLOBAL_RECOGNITION_SCOPE;
  const globalEventPeakPm = Number(state.frame?.surface_metrics?.global_event_absolute_shift_pm);
  const surfacePeakPm = Number(state.frame?.array_frame?.peak_wavelength_shift_pm);
  const current = measurementAvailable
    ? opticalForceTrace && metricItems.length
      ? traceOpticalForceN(metricItems[metricItems.length - 1])
      : modelResponseTrace && metricItems.length
        ? traceAttenuationPercent(metricItems[metricItems.length - 1])
      : globalFrame && Number.isFinite(globalEventPeakPm)
        ? Math.max(0, globalEventPeakPm)
        : Number.isFinite(surfacePeakPm)
          ? Math.max(-WAVELENGTH_SHIFT_FULL_SCALE_PM, Math.min(WAVELENGTH_SHIFT_FULL_SCALE_PM, surfacePeakPm))
          : traceAttenuationPercent(state.frame?.latest)
    : Number.NaN;
  const peak = measurementAvailable && values.length ? Math.max(...values.map(Math.abs)) : Number.NaN;

  const unit = opticalForceTrace ? "N" : modelResponseTrace ? "%" : "pm";
  const currentDigits = opticalForceTrace ? 2 : Math.abs(current) >= 100 ? 0 : 1;
  const peakDigits = opticalForceTrace ? 2 : Math.abs(peak) >= 100 ? 0 : 1;
  setText("traceCurrentLabel", "Now");
  setText("tracePeakLabel", "Peak");
  setCompactMetric(
    "traceCurrentValue",
    formatCompactNumber(current, currentDigits),
    Number.isFinite(current) ? `${current.toFixed(currentDigits)} ${unit}` : "--"
  );
  setCompactMetric(
    "tracePeakValue",
    formatCompactNumber(peak, peakDigits),
    Number.isFinite(peak) ? `${peak.toFixed(peakDigits)} ${unit}` : "--"
  );

  if (!measurementAvailable) {
    setText("traceHistoryValue", "--");
    return;
  }

  const timestamped = metricItems
    .map((item) => Number(item?.timestamp ?? item?.time ?? item?.time_s))
    .filter(Number.isFinite);
  let history = `${metricItems.length} pts`;
  if (timestamped.length >= 2) {
    const rawSpan = Math.max(...timestamped) - Math.min(...timestamped);
    const usesMilliseconds = Math.max(...timestamped) > 100000000000;
    const spanSeconds = usesMilliseconds ? rawSpan / 1000 : rawSpan;
    if (Number.isFinite(spanSeconds) && spanSeconds > 0) {
      history = spanSeconds < 10 ? `${spanSeconds.toFixed(1)} s` : `${Math.round(spanSeconds)} s`;
    }
  }
  setText("traceHistoryValue", history);
}

function drawTrace(records) {
  const { ctx, width, height } = setupCanvas(traceCanvas);
  drawGrid(ctx, width, height);
  let plotRecords = Array.isArray(records) ? records : [];
  updateTraceKpis(plotRecords);
  const opticalForceTrace = plotRecords.some((item) => Number.isFinite(traceOpticalForceN(item)));
  const modelResponseTrace = !opticalForceTrace && plotRecords.some((item) =>
    Number.isFinite(Number(item?.trained_model_visual_response_ratio))
  );
  if (plotRecords.length === 1 && Number.isFinite(traceDisplayValue(plotRecords[0]))) {
    plotRecords = [plotRecords[0], plotRecords[0]];
  }
  if (plotRecords.length < 2) {
    ctx.fillStyle = useDarkCanvas() ? "#6ea7bf" : "#71889d";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(
      frameHasMeasurement(state.frame)
        ? opticalForceTrace
          ? "Collecting optical force history"
          : modelResponseTrace
            ? "Collecting model response history"
            : "Collecting wavelength-shift history"
        : "No response history",
      18,
      30
    );
    return;
  }
  const values = plotRecords.map(traceDisplayValue).filter(Number.isFinite);
  if (!values.length) return;
  const globalEventTrace = !opticalForceTrace && !modelResponseTrace && state.frame?.scope === GLOBAL_RECOGNITION_SCOPE;
  const forceAxisMax = opticalForceTrace
    ? updateOpticalForceDisplayMaximum(Math.max(...values))
    : null;
  const thresholds = opticalForceTrace
    ? [0.25, 0.5, 0.75].map((ratio) => ratio * forceAxisMax)
    : globalEventTrace
    ? [
        GLOBAL_PROXY_FULL_SCALE_PM * RESPONSE_BAND_THRESHOLDS.noContactMax,
        GLOBAL_PROXY_FULL_SCALE_PM * RESPONSE_BAND_THRESHOLDS.smallMax,
        GLOBAL_PROXY_FULL_SCALE_PM * RESPONSE_BAND_THRESHOLDS.moderateMax,
      ]
    : modelResponseTrace
      ? [
          100 * RESPONSE_BAND_THRESHOLDS.noContactMax,
          100 * RESPONSE_BAND_THRESHOLDS.smallMax,
          100 * RESPONSE_BAND_THRESHOLDS.moderateMax,
        ]
      : [
        WAVELENGTH_SHIFT_FULL_SCALE_PM * RESPONSE_BAND_THRESHOLDS.noContactMax,
        WAVELENGTH_SHIFT_FULL_SCALE_PM * RESPONSE_BAND_THRESHOLDS.smallMax,
        WAVELENGTH_SHIFT_FULL_SCALE_PM * RESPONSE_BAND_THRESHOLDS.moderateMax,
      ];
  const minimumAxis = opticalForceTrace
    ? forceAxisMax
    : modelResponseTrace
      ? 100
      : globalEventTrace
        ? GLOBAL_PROXY_FULL_SCALE_PM
        : 300;
  const maximumAbs = Math.max(minimumAxis, ...values.map(Math.abs), ...thresholds);
  const axisStep = opticalForceTrace ? opticalForceAxisStep(forceAxisMax) : modelResponseTrace ? 25 : globalEventTrace ? 10 : 50;
  const calculatedAxis = Math.ceil(maximumAbs / axisStep) * axisStep;
  const axisLimit = opticalForceTrace
    ? forceAxisMax
    : globalEventTrace
    ? calculatedAxis
    : Math.min(WAVELENGTH_SHIFT_FULL_SCALE_PM, calculatedAxis);
  const yMin = opticalForceTrace || globalEventTrace || modelResponseTrace ? 0 : -axisLimit;
  const yMax = axisLimit;
  const plotLeft = 28;
  const plotRight = Math.max(plotLeft + 1, width - 8);
  const plotTop = 8;
  const plotBottom = Math.max(plotTop + 1, height - 20);
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;
  const xOf = (i) => plotLeft + (plotRecords.length <= 1 ? 0 : (i / (plotRecords.length - 1)) * plotWidth);
  const yOf = (v) => plotBottom - ((v - yMin) / (yMax - yMin)) * plotHeight;

  const darkCanvas = useDarkCanvas();
  ctx.save();
  ctx.font = "10px system-ui, sans-serif";
  const responseBands = opticalForceTrace
    ? []
    : globalEventTrace || modelResponseTrace
    ? [
        { low: 0, high: thresholds[0], light: "rgba(117, 151, 170, 0.05)", dark: "rgba(117, 151, 170, 0.08)" },
        { low: thresholds[0], high: thresholds[1], light: "rgba(76, 171, 170, 0.055)", dark: "rgba(76, 171, 170, 0.08)" },
        { low: thresholds[1], high: thresholds[2], light: "rgba(218, 184, 108, 0.065)", dark: "rgba(218, 184, 108, 0.08)" },
        { low: thresholds[2], high: yMax, light: "rgba(173, 77, 73, 0.055)", dark: "rgba(173, 77, 73, 0.08)" },
      ]
    : [
        { low: -thresholds[0], high: thresholds[0], light: "rgba(117, 151, 170, 0.05)", dark: "rgba(117, 151, 170, 0.08)" },
        { low: thresholds[0], high: thresholds[1], light: "rgba(76, 171, 170, 0.055)", dark: "rgba(76, 171, 170, 0.08)" },
        { low: thresholds[1], high: thresholds[2], light: "rgba(218, 184, 108, 0.065)", dark: "rgba(218, 184, 108, 0.08)" },
        { low: thresholds[2], high: yMax, light: "rgba(173, 77, 73, 0.055)", dark: "rgba(173, 77, 73, 0.08)" },
      ];
  responseBands.forEach((band) => {
    const visibleLow = Math.max(yMin, band.low);
    const visibleHigh = Math.min(yMax, band.high);
    if (visibleHigh <= visibleLow) return;
    const yTop = yOf(visibleHigh);
    const yBottom = yOf(visibleLow);
    ctx.fillStyle = darkCanvas ? band.dark : band.light;
    ctx.fillRect(plotLeft, yTop, plotWidth, yBottom - yTop);
  });
  const thresholdColors = opticalForceTrace
    ? thresholds.map(() => darkCanvas ? "#607f96" : "#9cb3c4")
    : darkCanvas
      ? ["#44c7d6", "#e7c75f", "#b95b5b"]
      : ["#5baec1", "#d9b86c", "#9f3f46"];
  thresholds.forEach((line, idx) => {
    ctx.strokeStyle = thresholdColors[idx];
    ctx.globalAlpha = 0.48;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(plotLeft, yOf(line));
    ctx.lineTo(plotRight, yOf(line));
    ctx.stroke();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = thresholdColors[idx];
    ctx.textAlign = "right";
    ctx.fillText(Number.isInteger(line) ? `${line}` : line.toFixed(1), plotLeft - 4, yOf(line) + 3);
  });
  if (yMin < 0 || opticalForceTrace) {
    ctx.strokeStyle = darkCanvas ? "rgba(180, 204, 222, 0.70)" : "rgba(92, 117, 136, 0.72)";
    ctx.globalAlpha = 0.9;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(plotLeft, yOf(0));
    ctx.lineTo(plotRight, yOf(0));
    ctx.stroke();
    ctx.fillStyle = darkCanvas ? "#9cb5c8" : "#60778a";
    ctx.textAlign = "right";
    ctx.fillText("0", plotLeft - 4, yOf(0) + 3);
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);

  const lineColor = darkCanvas ? "#9b42ff" : "#0b91d2";
  const validPoints = plotRecords
    .map((item, idx) => ({ x: xOf(idx), y: yOf(traceDisplayValue(item)) }))
    .filter((point) => Number.isFinite(point.y));
  if (validPoints.length) {
    const areaGradient = ctx.createLinearGradient(0, plotTop, 0, plotBottom);
    areaGradient.addColorStop(0, darkCanvas ? "rgba(155, 66, 255, 0.20)" : "rgba(11, 145, 210, 0.17)");
    areaGradient.addColorStop(1, darkCanvas ? "rgba(155, 66, 255, 0.01)" : "rgba(11, 145, 210, 0.01)");
    ctx.fillStyle = areaGradient;
    ctx.beginPath();
    ctx.moveTo(validPoints[0].x, plotBottom);
    validPoints.forEach((point) => ctx.lineTo(point.x, point.y));
    ctx.lineTo(validPoints[validPoints.length - 1].x, plotBottom);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  validPoints.forEach((point, idx) => {
    if (idx === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();
  const latestValue = traceDisplayValue(plotRecords[plotRecords.length - 1]);
  if (Number.isFinite(latestValue)) {
    ctx.fillStyle = lineColor;
    ctx.beginPath();
    ctx.arc(plotRight, yOf(latestValue), 3, 0, Math.PI * 2);
    ctx.fill();
    if (opticalForceTrace) {
      traceCanvas.dataset.currentForceN = latestValue.toFixed(4);
      delete traceCanvas.dataset.currentShiftPm;
      delete traceCanvas.dataset.currentResponsePercent;
    } else if (modelResponseTrace) {
      traceCanvas.dataset.currentResponsePercent = latestValue.toFixed(3);
      delete traceCanvas.dataset.currentShiftPm;
      delete traceCanvas.dataset.currentForceN;
    } else {
      traceCanvas.dataset.currentShiftPm = latestValue.toFixed(3);
      delete traceCanvas.dataset.currentResponsePercent;
      delete traceCanvas.dataset.currentForceN;
    }
  } else {
    delete traceCanvas.dataset.currentShiftPm;
    delete traceCanvas.dataset.currentResponsePercent;
    delete traceCanvas.dataset.currentForceN;
  }
  ctx.fillStyle = darkCanvas ? "#88a8c5" : "#536f86";
  ctx.textAlign = "left";
  ctx.font = "9px system-ui, sans-serif";
  const traceScope = opticalForceTrace
    ? "Optical Fz estimate (N)"
    : modelResponseTrace
      ? "Trained model visual response (%)"
    : globalEventTrace
    ? "Global event peak |shift| (pm)"
    : state.arrayDemoActive
      ? "Surface peak |shift| (pm)"
      : `${state.frame?.selected_channel || state.selectedChannel} shift (pm)`;
  traceCanvas.dataset.scope = opticalForceTrace
    ? "continuous_optical_force_n"
    : modelResponseTrace
      ? "trained_model_visual_response_percent"
    : globalEventTrace
    ? "global_residual_compensated_event_peak"
    : state.arrayDemoActive
      ? "surface_peak"
      : "selected_channel";
  ctx.fillText(traceScope, plotLeft, height - 6);
  ctx.restore();
}

function spectrumArrays(record) {
  const axisIsPixel = record?.peak_axis_type === "pixel_index" || record?.spectrum_x_unit === "pixel_index";
  let xValues = Array.isArray(record?.wavelength_nm)
    ? record.wavelength_nm
    : Array.isArray(record?.spectrum_wavelength_nm)
      ? record.spectrum_wavelength_nm
      : [];
  const rawIntensity = Array.isArray(record?.intensity)
    ? record.intensity
    : Array.isArray(record?.spectrum_counts)
      ? record.spectrum_counts
      : [];
  const processedIntensity = Array.isArray(record?.display_intensity)
    ? record.display_intensity
    : [];
  const normalizedIntensity = Array.isArray(record?.normalized_intensity_ratio)
    ? record.normalized_intensity_ratio
    : [];
  const normalizationStatus = String(record?.spectrum_normalization?.status || "");
  const normalizedDisplay = Boolean(
    state.spectrumProcessing?.settings?.normalize_spectrum !== false &&
    normalizedIntensity.length === rawIntensity.length &&
    normalizedIntensity.length &&
    normalizedIntensity.every((value) => Number.isFinite(Number(value))) &&
    ["ready", "ready_with_invalid_reference_points"].includes(normalizationStatus)
  );
  const intensity = normalizedDisplay
    ? normalizedIntensity
    : processedIntensity.length === rawIntensity.length && processedIntensity.length
      ? processedIntensity
      : rawIntensity;
  const candidateOverlay = Array.isArray(record?.overlay_intensity)
    ? record.overlay_intensity
    : [];
  const overlayIntensity = !normalizedDisplay && candidateOverlay.length === intensity.length
    ? candidateOverlay
    : [];
  if (axisIsPixel && intensity.length) {
    xValues = intensity.map((_, idx) => idx);
  } else if (intensity.length && xValues.length !== intensity.length) {
    const startNm = Number(record?.spectrum_start_nm);
    const endNm = Number(record?.spectrum_end_nm);
    if (Number.isFinite(startNm) && Number.isFinite(endNm)) {
      xValues = intensity.map((_, idx) => startNm + ((endNm - startNm) * idx) / Math.max(intensity.length - 1, 1));
    } else {
      xValues = intensity.map((_, idx) => idx);
    }
  }
  return {
    xValues,
    intensity,
    overlayIntensity,
    rawIntensity,
    normalizedIntensity,
    axisIsPixel,
    normalizedDisplay,
    displayProcessed: intensity === processedIntensity && processedIntensity.length > 0,
  };
}

function normalizedSpectrumBounds(values) {
  const finite = values.map(Number).filter(Number.isFinite);
  if (!finite.length) return { yMin: 0.8, yMax: 1.2 };
  const minimum = Math.min(1, ...finite);
  const maximum = Math.max(1, ...finite);
  const span = Math.max(maximum - minimum, 0.10);
  return {
    yMin: Math.max(0, minimum - span * 0.15),
    yMax: maximum + span * 0.15,
  };
}

function drawNormalizedReferenceLine(ctx, xStart, xEnd, y, { label = true } = {}) {
  ctx.save();
  ctx.strokeStyle = useDarkCanvas()
    ? "rgba(222, 233, 239, 0.55)"
    : "rgba(71, 100, 121, 0.48)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(xStart, y);
  ctx.lineTo(xEnd, y);
  ctx.stroke();
  ctx.setLineDash([]);
  if (label) {
    ctx.fillStyle = useDarkCanvas() ? "#b8cbd7" : "#587084";
    ctx.font = "9px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("I/I0 = 1", xStart + 4, Math.max(10, y - 4));
  }
  ctx.restore();
}

function isSyntheticWavelengthSpectrum(record) {
  const spectrumType = String(record?.spectrum_type || "").toLowerCase();
  const source = String(record?.source || "").toLowerCase();
  if (spectrumType.includes("recorded real") || source.includes("20260731_")) return false;
  return spectrumType.includes("synthetic") || source.includes("synthetic_");
}

function isRecordedDemoSpectrum(record, arrayFrame = state.currentArrayFrame) {
  const spectrumType = String(record?.spectrum_type || "").toLowerCase();
  const dataMode = String(record?.demo_data_mode || arrayFrame?.demo_data_mode || "").toLowerCase();
  return spectrumType.includes("recorded real") || dataMode.includes("recorded_real_spectrum");
}

function fixedSyntheticSpectrumMaximum(record, fallbackValues = []) {
  const peakBaselines = (Array.isArray(record?.spectrum_peaks) ? record.spectrum_peaks : [])
    .flatMap((peak) => [Number(peak?.baseline_intensity_counts), Number(peak?.intensity_counts)])
    .filter(Number.isFinite);
  const fallback = fallbackValues.map(Number).filter(Number.isFinite);
  return Math.max(1, ...peakBaselines, ...fallback);
}

function spectrumUnavailableMessage(record) {
  if (record?.spectrum_unavailable_reason) return record.spectrum_unavailable_reason;
  if (state.currentArrayFrame?.mode === "simulated_array_demo") {
    return state.displayMode === "operator"
      ? "Spectrum unavailable for this local response frame"
      : "Spectrum unavailable for this simulated frame";
  }
  return "Spectrum unavailable for this frame";
}

function formatWavelength(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(digits)} nm` : "--";
}

function wavelengthPlanText(plan, compact = false) {
  const start = Number(plan?.wavelength_start_nm);
  const stop = Number(plan?.wavelength_stop_nm);
  const spacing = Number(plan?.wavelength_spacing_nm);
  const status = String(plan?.status || "preliminary_target_plan").replace(/_/g, " ");
  if (Number.isFinite(start) && Number.isFinite(stop) && Number.isFinite(spacing)) {
    return compact
      ? `${start.toFixed(0)}-${stop.toFixed(0)} nm, ${spacing.toFixed(0)} nm preliminary`
      : `Target wavelength plan: ${start.toFixed(0)} to ${stop.toFixed(0)} nm, ${spacing.toFixed(0)} nm spacing, ${status}.`;
  }
  return compact ? "target wavelength plan preliminary" : "Target wavelength plan: preliminary.";
}

function targetPeakMarkers() {
  const channels = state.currentArrayFrame?.channels || [];
  const markers = channels
    .map((channel) => {
      const target = Number(channel?.target_wavelength_nm);
      if (!Number.isFinite(target)) return null;
      return {
        channel_id: channel.channel_id,
        target_wavelength_nm: target,
        measured_wavelength_nm: Number(channel?.measured_wavelength_nm),
        demodulation_wavelength_nm: Number(channel?.demodulation_wavelength_nm),
        enabled: Boolean(channel?.enabled),
        valid: Boolean(channel?.valid),
      };
    })
    .filter(Boolean);
  return markers.sort((a, b) => {
    const ai = WAVELENGTH_PLAN_ORDER.indexOf(a.channel_id);
    const bi = WAVELENGTH_PLAN_ORDER.indexOf(b.channel_id);
    if (ai !== -1 && bi !== -1) return ai - bi;
    return a.target_wavelength_nm - b.target_wavelength_nm;
  });
}

function drawSpectrum(record) {
  drawOpticalPreview(record);
  const { ctx, width, height } = setupCanvas(spectrumCanvas);
  drawGrid(ctx, width, height);
  const { xValues, intensity, overlayIntensity, axisIsPixel, normalizedDisplay } = spectrumArrays(record);
  if (!xValues.length || xValues.length !== intensity.length) {
    ctx.fillStyle = useDarkCanvas() ? "#7fb4ce" : "#71889d";
    ctx.fillText(spectrumUnavailableMessage(record), 18, 32);
    return;
  }
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const plottedValues = overlayIntensity.length
    ? intensity.concat(overlayIntensity)
    : intensity;
  const normalizedBounds = normalizedDisplay
    ? normalizedSpectrumBounds(plottedValues)
    : null;
  const yMin = normalizedBounds?.yMin ?? 0;
  const yMax = normalizedBounds?.yMax ?? (
    isSyntheticWavelengthSpectrum(record)
      ? fixedSyntheticSpectrumMaximum(record, plottedValues) * 1.08
      : Math.max(Math.max(...plottedValues) * 1.08, Number(record?.baseline_intensity_counts) || 1)
  );
  const plotLeft = 6;
  const plotRight = width - 6;
  const plotTop = 46;
  const plotBottom = height - 8;
  const xOf = (v) => plotLeft + ((v - xMin) / Math.max(xMax - xMin, 1e-9)) * (plotRight - plotLeft);
  const yOf = (v) => plotBottom - ((v - yMin) / Math.max(yMax - yMin, 1e-9)) * (plotBottom - plotTop);

  const darkCanvas = useDarkCanvas();
  if (normalizedDisplay && 1 >= yMin && 1 <= yMax) {
    drawNormalizedReferenceLine(ctx, plotLeft, plotRight, yOf(1));
  }
  if (overlayIntensity.length) {
    ctx.strokeStyle = darkCanvas ? "rgba(165, 193, 210, 0.48)" : "rgba(94, 117, 134, 0.42)";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    xValues.forEach((xValue, idx) => {
      const x = xOf(xValue);
      const y = yOf(Number(overlayIntensity[idx]) || 0);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.strokeStyle = isSyntheticWavelengthSpectrum(record) ? "#18a99a" : darkCanvas ? "#32f0a4" : "#13a56f";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  xValues.forEach((xValue, idx) => {
    const x = xOf(xValue);
    const y = yOf(Number(intensity[idx]) || 0);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const peakProfile = String(record?.spectrum_peak_profile || "").toLowerCase();
  const candidatePeakProfile =
    peakProfile === "current_real_9fbg_candidate" ||
    peakProfile === "recorded_auto_discovered_9fbg";
  const markers = axisIsPixel || candidatePeakProfile ? [] : targetPeakMarkers();
  if (markers.length) {
    ctx.save();
    markers.forEach((marker) => {
      const xValue = Number(marker.target_wavelength_nm);
      if (!Number.isFinite(xValue) || xValue < xMin || xValue > xMax) return;
      const x = xOf(xValue);
      const isDominantTarget = marker.channel_id === record?.dominant_channel || marker.channel_id === record?.selected_channel;
      const order = Math.max(0, WAVELENGTH_PLAN_ORDER.indexOf(marker.channel_id));
      const labelY = 16 + (order % 3) * 11;
      ctx.strokeStyle = isDominantTarget ? "#b45850" : marker.enabled ? (darkCanvas ? "#83d8ff" : "#7fa1b8") : darkCanvas ? "#3f5f75" : "#b5c2cc";
      ctx.globalAlpha = isDominantTarget ? 0.95 : marker.enabled ? 0.46 : 0.28;
      ctx.lineWidth = isDominantTarget ? 1.8 : marker.enabled ? 1.05 : 0.85;
      ctx.setLineDash(marker.enabled ? [5, 4] : [2, 5]);
      ctx.beginPath();
      ctx.moveTo(x, plotTop - 5);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = isDominantTarget ? "#8f3f39" : marker.enabled ? (darkCanvas ? "#b7eaff" : "#517187") : darkCanvas ? "#6f8294" : "#8798a5";
      ctx.font = isDominantTarget ? "bold 11px Segoe UI" : marker.enabled ? "bold 9px Segoe UI" : "9px Segoe UI";
      ctx.fillText(String(marker.channel_id || ""), Math.min(width - 24, x + 3), labelY);
    });
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  const peaks = Array.isArray(record?.spectrum_peaks) ? record.spectrum_peaks : [];
  if (peaks.length) {
    peaks.forEach((peak, peakIndex) => {
      const marker = axisIsPixel
        ? Number(peak.peak_pixel_index)
        : Number(peak.tracked_wavelength_nm ?? peak.peak_wavelength_nm ?? peak.target_wavelength_nm);
      if (!Number.isFinite(marker) || marker < xMin || marker > xMax) return;
      const x = xOf(marker);
      const peakDisplayId = peak.provisional_channel_id || peak.channel_id || peak.candidate_id;
      const peakIdentity = peak.candidate_id || peak.channel_id || peak.provisional_channel_id;
      const isDominant =
        peakDisplayId === record?.dominant_channel ||
        peakIdentity === record?.dominant_candidate_id ||
        peak.dominant;
      const role = String(peak.affected_role || "");
      const roles = Array.isArray(peak.coupling_roles) ? peak.coupling_roles.join(" ") : role;
      const isSameFiber = roles.includes("same_fiber");
      const isCrossFiber = roles.includes("cross_fiber") || roles.includes("shared_elastomer") || roles.includes("force_transfer");
      const isUnknown = roles.includes("unknown");
      ctx.strokeStyle = isDominant ? "#b45850" : isSameFiber ? "#d29119" : isCrossFiber ? "#18a4bd" : isUnknown ? "#7f8b94" : "#8eb6cc";
      ctx.globalAlpha = isDominant ? 0.95 : isSameFiber ? 0.76 : isCrossFiber ? 0.68 : isUnknown ? 0.48 : 0.32;
      ctx.lineWidth = isDominant ? 1.8 : isSameFiber || isCrossFiber ? 1.4 : 1;
      ctx.setLineDash(isDominant ? [] : isSameFiber ? [8, 4] : isCrossFiber ? [3, 3] : [4, 5]);
      ctx.beginPath();
      ctx.moveTo(x, plotTop - 5);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      const nearestIndex = xValues.reduce(
        (bestIndex, value, index) => Math.abs(value - marker) < Math.abs(xValues[bestIndex] - marker) ? index : bestIndex,
        0
      );
      const markerY = yOf(Number(intensity[nearestIndex]) || 0);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(x, markerY, isDominant ? 3.4 : 2.2, 0, Math.PI * 2);
      ctx.fill();
      if (peak.candidate_mapping) {
        ctx.globalAlpha = 0.92;
        ctx.font = isDominant ? "bold 10px Segoe UI" : "9px Segoe UI";
        ctx.fillStyle = isDominant ? "#8f3f39" : darkCanvas ? "#b7d7e8" : "#46677e";
        ctx.fillText(String(peakDisplayId || ""), Math.min(width - 36, x + 3), 15 + (peakIndex % 3) * 11);
      }
    });
    ctx.globalAlpha = 1;
  } else {
    const peak = axisIsPixel
      ? Number(record?.peak_pixel_index)
      : Number(record?.tracked_wavelength_nm ?? record?.peak_wavelength_nm);
    const target = axisIsPixel ? Number.NaN : Number(record?.target_wavelength_nm);
    const marker = Number.isFinite(peak) ? peak : target;
    if (Number.isFinite(marker) && marker >= xMin && marker <= xMax) {
      const x = xOf(marker);
      ctx.strokeStyle = "#d89713";
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, plotTop - 5);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function drawOpticalPreview(record) {
  if (!opticalPreviewCanvas) return;
  const { ctx, width, height } = setupCanvas(opticalPreviewCanvas);
  const darkCanvas = useDarkCanvas();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = darkCanvas ? "#07111d" : "#f8fbfd";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = darkCanvas ? "rgba(90, 145, 180, 0.16)" : "rgba(111, 143, 166, 0.14)";
  ctx.lineWidth = 1;
  for (let index = 1; index < 4; index += 1) {
    const x = (width * index) / 4;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let index = 1; index < 4; index += 1) {
    const y = (height * index) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  const { xValues, intensity, overlayIntensity, axisIsPixel, normalizedDisplay } = spectrumArrays(record);
  if (!xValues.length || xValues.length !== intensity.length) {
    // The state-driven DOM overlay owns the concise no-frame message. Keep
    // only the grid in the canvas so empty-state text is never duplicated.
    return;
  }

  const numericIntensity = intensity
    .concat(overlayIntensity)
    .map(Number)
    .filter(Number.isFinite);
  if (!numericIntensity.length) return;
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMinRaw = Math.min(...numericIntensity);
  const yMaxRaw = Math.max(...numericIntensity);
  const yPadding = Math.max((yMaxRaw - yMinRaw) * 0.12, Math.abs(yMaxRaw) * 0.015, 1);
  const syntheticSpectrum = isSyntheticWavelengthSpectrum(record);
  opticalPreviewCanvas.dataset.frameRenderSemantics = syntheticSpectrum
    ? String(record?.frame_render_semantics || "replace_previous_spectrum")
    : "measured_frame_replacement";
  opticalPreviewCanvas.dataset.peakHeightMode = syntheticSpectrum
    ? String(record?.peak_height_mode || "fixed_per_channel")
    : "measured";
  opticalPreviewCanvas.dataset.intensityModulationEnabled = String(
    syntheticSpectrum ? Boolean(record?.intensity_modulation_enabled) : false
  );
  opticalPreviewCanvas.dataset.intensityVariationMode = syntheticSpectrum ? "fixed_demo_peak_height" : "measured_raw";
  opticalPreviewCanvas.dataset.responseSemantics = String(
    record?.spectral_evidence_semantics || (syntheticSpectrum ? "wavelength_translation_only_demo" : "measured_spectrum")
  );
  const normalizedBounds = normalizedDisplay
    ? normalizedSpectrumBounds(numericIntensity)
    : null;
  const yMin = normalizedBounds?.yMin ?? (
    syntheticSpectrum ? 0 : Math.max(0, yMinRaw - yPadding)
  );
  const yMax = normalizedBounds?.yMax ?? (
    syntheticSpectrum
      ? fixedSyntheticSpectrumMaximum(record, numericIntensity) * 1.08
      : yMaxRaw + yPadding
  );
  const left = 7;
  const right = Math.max(left + 1, width - 7);
  const top = 9;
  const bottom = Math.max(top + 1, height - 18);
  const xOf = (value) => left + ((value - xMin) / Math.max(xMax - xMin, 1e-9)) * (right - left);
  const yOf = (value) => bottom - ((value - yMin) / Math.max(yMax - yMin, 1e-9)) * (bottom - top);

  if (normalizedDisplay && 1 >= yMin && 1 <= yMax) {
    drawNormalizedReferenceLine(ctx, left, right, yOf(1), { label: false });
  }

  const fill = ctx.createLinearGradient(0, top, 0, bottom);
  fill.addColorStop(0, darkCanvas ? "rgba(37, 197, 185, 0.24)" : "rgba(18, 158, 148, 0.20)");
  fill.addColorStop(1, darkCanvas ? "rgba(37, 197, 185, 0.02)" : "rgba(18, 158, 148, 0.01)");
  ctx.fillStyle = fill;
  ctx.beginPath();
  xValues.forEach((value, index) => {
    const x = xOf(value);
    const y = yOf(Number(intensity[index]) || 0);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.closePath();
  ctx.fill();

  if (overlayIntensity.length) {
    ctx.strokeStyle = darkCanvas ? "rgba(155, 188, 205, 0.55)" : "rgba(93, 116, 132, 0.48)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    xValues.forEach((value, index) => {
      const x = xOf(value);
      const y = yOf(Number(overlayIntensity[index]) || 0);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.strokeStyle = darkCanvas ? "#33d6c8" : "#138f92";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  xValues.forEach((value, index) => {
    const x = xOf(value);
    const y = yOf(Number(intensity[index]) || 0);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Candidate real FBGs are shown as compact dots and staggered labels. Avoid
  // full-height marker lines here: they obscure the measured spectrum and can
  // look like extra spectral peaks in the narrow operator panel.
  const previewPeaks = (Array.isArray(record?.spectrum_peaks) ? record.spectrum_peaks : [])
    .filter((peak) => peak?.candidate_mapping);
  opticalPreviewCanvas.dataset.candidatePeakCount = String(previewPeaks.length);
  previewPeaks.forEach((peak, peakIndex) => {
    const marker = Number(
      peak.tracked_wavelength_nm ?? peak.peak_wavelength_nm ?? peak.candidate_measured_wavelength_nm
    );
    if (!Number.isFinite(marker) || marker < xMin || marker > xMax) return;
    const nearestIndex = xValues.reduce(
      (bestIndex, value, index) => Math.abs(value - marker) < Math.abs(xValues[bestIndex] - marker) ? index : bestIndex,
      0
    );
    const x = xOf(marker);
    const y = yOf(Number(intensity[nearestIndex]) || 0);
    const peakDisplayId = peak.candidate_id || peak.channel_id;
    const isDominant = peakDisplayId === (record?.dominant_candidate_id || record?.dominant_channel);
    ctx.fillStyle = isDominant ? "#9f3f46" : darkCanvas ? "#8ddbd4" : "#247b82";
    ctx.beginPath();
    ctx.arc(x, y, isDominant ? 2.8 : 2.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = isDominant ? "bold 8px system-ui, sans-serif" : "7px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(peakDisplayId || ""), x, Math.max(8, top + 8 + (peakIndex % 3) * 9));
  });

  ctx.fillStyle = darkCanvas ? "#7798aa" : "#6b8193";
  ctx.font = "9px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(axisIsPixel ? "pixel" : `${xMin.toFixed(0)} nm`, left, height - 5);
  ctx.textAlign = "right";
  ctx.fillText(axisIsPixel ? `${Math.round(xMax)}` : `${xMax.toFixed(0)} nm`, right, height - 5);
}

function drawSelectedPeakZoom(record) {
  const { ctx, width, height } = setupCanvas(selectedSpectrumCanvas);
  drawGrid(ctx, width, height);
  const { xValues, intensity, axisIsPixel, normalizedDisplay } = spectrumArrays(record);
  if (!xValues.length || xValues.length !== intensity.length) {
    ctx.fillStyle = useDarkCanvas() ? "#7fb4ce" : "#71889d";
    ctx.fillText(spectrumUnavailableMessage(record), 18, 30);
    return;
  }
  const selected = record?.dominant_candidate_id || record?.dominant_channel || record?.selected_channel || record?.channel_id || state.selectedChannel;
  const peaks = Array.isArray(record?.spectrum_peaks) ? record.spectrum_peaks : [];
  const selectedPeak =
    peaks.find((peak) => (peak.candidate_id || peak.channel_id) === selected) ||
    peaks.find((peak) => peak.dominant) ||
    {};
  const marker = axisIsPixel
    ? Number(selectedPeak.peak_pixel_index ?? record?.peak_pixel_index)
    : Number(
        selectedPeak.tracked_wavelength_nm ??
          selectedPeak.peak_wavelength_nm ??
          selectedPeak.target_wavelength_nm ??
          record?.tracked_wavelength_nm ??
          record?.peak_wavelength_nm ??
          record?.target_wavelength_nm
      );
  const fullMin = Math.min(...xValues);
  const fullMax = Math.max(...xValues);
  const halfWindow = axisIsPixel ? 35 : 0.55;
  const xMin = Number.isFinite(marker) ? Math.max(fullMin, marker - halfWindow) : fullMin;
  const xMax = Number.isFinite(marker) ? Math.min(fullMax, marker + halfWindow) : fullMax;
  const visible = xValues
    .map((xValue, index) => ({ xValue, yValue: Number(intensity[index]) || 0 }))
    .filter((point) => point.xValue >= xMin && point.xValue <= xMax);
  if (!visible.length) {
    ctx.fillStyle = useDarkCanvas() ? "#7fb4ce" : "#71889d";
    ctx.fillText("Selected peak outside current spectrum window", 18, 30);
    return;
  }
  const visibleValues = visible.map((point) => point.yValue);
  const normalizedBounds = normalizedDisplay
    ? normalizedSpectrumBounds(visibleValues)
    : null;
  const yMin = normalizedBounds?.yMin ?? (
    isSyntheticWavelengthSpectrum(record) ? 0 : Math.min(...visibleValues) * 0.94
  );
  const yMax = normalizedBounds?.yMax ?? (
    isSyntheticWavelengthSpectrum(record)
      ? fixedSyntheticSpectrumMaximum(record, visibleValues) * 1.08
      : Math.max(...visibleValues) * 1.06
  );
  const plotLeft = 8;
  const plotRight = width - 8;
  const plotTop = 8;
  const plotBottom = height - 8;
  const xOf = (v) => plotLeft + ((v - xMin) / Math.max(xMax - xMin, 1e-9)) * (plotRight - plotLeft);
  const yOf = (v) => plotBottom - ((v - yMin) / Math.max(yMax - yMin, 1e-9)) * (plotBottom - plotTop);
  const darkCanvas = useDarkCanvas();
  if (normalizedDisplay && 1 >= yMin && 1 <= yMax) {
    drawNormalizedReferenceLine(ctx, plotLeft, plotRight, yOf(1), { label: false });
  }
  ctx.strokeStyle = darkCanvas ? "#32f0a4" : "#169e79";
  ctx.lineWidth = 2;
  ctx.beginPath();
  visible.forEach((point, index) => {
    const x = xOf(point.xValue);
    const y = yOf(point.yValue);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (Number.isFinite(marker)) {
    const x = xOf(marker);
    ctx.strokeStyle = "#d89713";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(x, plotTop);
    ctx.lineTo(x, plotBottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawHeatmap(arrayFrame) {
  const { ctx, width, height } = setupCanvas(heatmapCanvas);
  drawGrid(ctx, width, height);
  const grid = arrayFrame?.surface_grid;
  const metrics = arrayFrame?.surface_metrics || {};
  if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) {
    ctx.fillStyle = "#71889d";
    ctx.fillText("Waiting for Bragg wavelength-shift surface", 18, 32);
    return;
  }
  const rows = grid.length;
  const cols = grid[0].length;
  const coordToCanvas = (x, y) => ({
    x: ((x + 1.25) / 2.5) * width,
    y: ((1.25 - y) / 2.5) * height,
  });
  const channelMap = new Map((arrayFrame?.channels || []).map((channel) => [channel.channel_id, channel]));
  const fallbackLike = ["p22_fallback", "single_point_p22", "no_valid_channel"].includes(String(arrayFrame?.mode || ""));
  const fallbackP22 = channelMap.get("P22");
  const channels = fallbackLike
    ? [{ ...(fallbackP22 || {}), channel_id: "P22", x: 0, y: 0 }]
    : ARRAY_DISPLAY_ORDER.map((channelId) => channelMap.get(channelId)).filter(Boolean);
  ctx.fillStyle = "#edf8fb";
  ctx.fillRect(0, 0, width, height);
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  channels
    .filter((channel) => channel.valid && Number(channel.wavelength_shift_response_ratio ?? channel.response_value) > 0.012)
    .forEach((channel) => {
      const response = Math.max(0, Math.min(1, Number(channel.wavelength_shift_response_ratio ?? channel.response_value) || 0));
      const point = coordToCanvas(Number(channel.x) || 0, Number(channel.y) || 0);
      const color = colorForAttenuation(response);
      const rgb = `${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}`;
      const radius = Math.max(width, height) * (fallbackLike ? 0.40 : 0.30);
      const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
      gradient.addColorStop(0, `rgba(${rgb}, ${0.36 + 0.48 * response})`);
      gradient.addColorStop(0.45, `rgba(${rgb}, ${0.20 + 0.26 * response})`);
      gradient.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
    });
  ctx.restore();

  const peakValue = Number(metrics.surface_peak) || 0;
  const cxMetric = Number(metrics.surface_centroid_x);
  const cyMetric = Number(metrics.surface_centroid_y);
  if (Number.isFinite(cxMetric) && Number.isFinite(cyMetric) && peakValue > 0.04) {
    const centerPoint = coordToCanvas(cxMetric, cyMetric);
    const spread = Math.max(0.18, Number(metrics.surface_spread) || 0.32);
    [
      { threshold: 0.05, color: "#5aaea6", scale: 1.34 },
      { threshold: 0.30, color: "#d9b86c", scale: 0.92 },
      { threshold: 0.70, color: "#9f3d37", scale: 0.58 },
    ].forEach((contour) => {
      if (peakValue < contour.threshold) return;
      const ratio = Math.max(0.18, 1 - contour.threshold / Math.max(peakValue, 1e-6));
      const rx = Math.min(width * 0.46, Math.max(18, width * 0.16 * spread * contour.scale + width * 0.18 * ratio));
      const ry = Math.min(height * 0.46, Math.max(14, height * 0.18 * spread * contour.scale + height * 0.22 * ratio));
      ctx.strokeStyle = contour.color;
      ctx.globalAlpha = 0.58;
      ctx.lineWidth = contour.threshold >= 0.70 ? 1.6 : 1.2;
      ctx.beginPath();
      ctx.ellipse(centerPoint.x, centerPoint.y, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  channels.forEach((channel) => {
    const point = coordToCanvas(Number(channel.x) || 0, Number(channel.y) || 0);
    ctx.beginPath();
    ctx.arc(point.x, point.y, channel.valid ? 4.2 : 3.2, 0, Math.PI * 2);
    ctx.fillStyle = channel.valid ? "rgba(16, 34, 54, 0.70)" : "rgba(255, 255, 255, 0.78)";
    ctx.strokeStyle = channel.enabled ? "rgba(49, 83, 107, 0.62)" : "rgba(154, 176, 194, 0.42)";
    ctx.lineWidth = channel.valid ? 1.5 : 1;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "rgba(49, 83, 107, 0.58)";
    ctx.font = "10px Segoe UI";
    ctx.fillText(channel.channel_id, point.x + 6, point.y - 5);
  });

  const cx = Number(metrics.surface_centroid_x);
  const cy = Number(metrics.surface_centroid_y);
  if (Number.isFinite(cx) && Number.isFinite(cy)) {
    const center = coordToCanvas(cx, cy);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = "#102236";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 7, 0, Math.PI * 2);
    ctx.stroke();
    if (state.trajectoryHistory.length > 1) {
      ctx.strokeStyle = "#102236";
      ctx.globalAlpha = 0.72;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      state.trajectoryHistory.forEach((item, index) => {
        const point = coordToCanvas(item.x, item.y);
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  ctx.fillStyle = "#102236";
  ctx.font = "11px Segoe UI";
  ctx.fillText(`peak ${formatPercent(metrics.surface_peak, 1)} | active area ${formatPercent(metrics.surface_area_active, 1)}`, 12, height - 12);
}

const GROOVE_SUPERELLIPSE_POWER = 3.65;
const DEFAULT_SLOT_BOUNDARY_PROFILE = [
  0.998, 0.986, 0.954, 0.931, 0.910, 0.895, 0.886, 0.879,
  0.872, 0.867, 0.868, 0.878, 0.894, 0.910, 0.945, 0.976,
  0.994, 0.984, 0.966, 0.948, 0.940, 0.944, 0.960, 0.982,
  1.022, 1.068, 1.068, 1.000, 1.000, 1.000, 1.000, 1.000,
  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.063, 1.062,
  1.022, 0.983, 0.949, 0.934, 0.932, 0.943, 0.962, 0.991,
  0.993, 0.993, 0.948, 0.926, 0.904, 0.891, 0.885, 0.885,
  0.888, 0.894, 0.899, 0.909, 0.921, 0.934, 0.963, 0.991,
];
const DEFAULT_SLOT_BOUNDARY_POINTS = [
  [-3.3888, -1.7352], [-3.2007, -1.8122], [-3.0104, -1.8832], [-2.8184, -1.9492],
  [-2.6251, -2.0106], [-2.4308, -2.0684], [-2.2354, -2.1221], [-2.0392, -2.1726],
  [-1.8424, -2.2200], [-1.6448, -2.2642], [-1.4467, -2.3052], [-1.2480, -2.3437],
  [-1.0477, -2.3709], [-0.8467, -2.3909], [-0.6451, -2.4045], [-0.4432, -2.4118],
  [-0.2412, -2.4130], [-0.0393, -2.4087], [0.1625, -2.3993], [0.3640, -2.3848],
  [0.5651, -2.3654], [0.7657, -2.3402], [0.9655, -2.3092], [1.1643, -2.2716],
  [1.3617, -2.2269], [1.5574, -2.1745], [1.7508, -2.1137], [1.9414, -2.0434],
  [2.1279, -1.9621], [2.3090, -1.8685], [2.4833, -1.7615], [2.6487, -1.6402],
  [2.8029, -1.5034], [2.9424, -1.3504], [3.0649, -1.1821], [3.1677, -0.9999],
  [3.2499, -0.8066], [3.3107, -0.6047], [3.3520, -0.3975], [3.3754, -0.1871],
  [3.3761, 0.0245], [3.3597, 0.2356], [3.3259, 0.4443], [3.2726, 0.6486],
  [3.1986, 0.8455], [3.1045, 1.0329], [2.9913, 1.2081], [2.8596, 1.3686],
  [2.7126, 1.5137], [2.5533, 1.6438], [2.3842, 1.7595], [2.2075, 1.8620],
  [2.0246, 1.9518], [1.8372, 2.0308], [1.6462, 2.0996], [1.4526, 2.1600],
  [1.2569, 2.2123], [1.0595, 2.2574], [0.8609, 2.2961], [0.6614, 2.3286],
  [0.4610, 2.3552], [0.2600, 2.3769], [0.0587, 2.3934], [-0.1430, 2.4055],
  [-0.3449, 2.4119], [-0.5469, 2.4130], [-0.7488, 2.4082], [-0.9505, 2.3971],
  [-1.1518, 2.3796], [-1.3522, 2.3537], [-1.5506, 2.3138], [-1.7484, 2.2710],
  [-1.9456, 2.2253], [-2.1422, 2.1765], [-2.3379, 2.1242], [-2.5327, 2.0682],
  [-2.7266, 2.0087], [-2.9192, 1.9450], [-3.1103, 1.8765], [-3.2995, 1.8022],
  [-3.3888, 1.6540], [-3.3888, 1.4421], [-3.3888, 1.2303], [-3.3888, 1.0185],
  [-3.3888, 0.8067], [-3.3888, 0.5949], [-3.3888, 0.3830], [-3.3888, 0.1712],
  [-3.3888, -0.0406], [-3.3888, -0.2524], [-3.3888, -0.4642], [-3.3888, -0.6761],
  [-3.3888, -0.8879], [-3.3888, -1.0997], [-3.3888, -1.3115], [-3.3888, -1.5233],
];

function currentSlotBoundaryProfile() {
  const configured = state.thumbSceneConfig?.sensor_slot_transform?.boundary_profile?.radial_scale;
  const profile = Array.isArray(configured) && configured.length >= 8 ? configured : DEFAULT_SLOT_BOUNDARY_PROFILE;
  return profile.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
}

function slotBoundaryProfileScale(theta) {
  const profile = currentSlotBoundaryProfile();
  if (!profile.length) return 1;
  const normalized = ((theta % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
  const scaled = (normalized / (Math.PI * 2)) * profile.length;
  const index0 = Math.floor(scaled) % profile.length;
  const index1 = (index0 + 1) % profile.length;
  const t = scaled - Math.floor(scaled);
  return profile[index0] * (1 - t) + profile[index1] * t;
}

function currentSlotBoundaryPoints(width, depth, fallbackSegments = 128) {
  const configured = state.thumbSceneConfig?.sensor_slot_transform?.boundary_points_local;
  const source = Array.isArray(configured) && configured.length >= 8 ? configured : DEFAULT_SLOT_BOUNDARY_POINTS;
  const points = source
    .map((point) => [Number(point?.[0]), Number(point?.[1])])
    .filter(([x, z]) => Number.isFinite(x) && Number.isFinite(z));
  if (points.length >= 8) return points;
  const generated = [];
  for (let ia = 0; ia < fallbackSegments; ia += 1) {
    const theta = (Math.PI * 2 * ia) / fallbackSegments;
    generated.push(grooveBoundaryPoint(width, depth, theta, 1));
  }
  return generated;
}

function grooveBoundaryPoint(width, depth, theta, radialScale = 1) {
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const exponent = 2 / GROOVE_SUPERELLIPSE_POWER;
  const boundaryScale = slotBoundaryProfileScale(theta);
  const x = (width / 2) * Math.sign(cosT) * Math.pow(Math.abs(cosT), exponent) * radialScale * boundaryScale;
  const z = (depth / 2) * Math.sign(sinT) * Math.pow(Math.abs(sinT), exponent) * radialScale * boundaryScale;
  return [x, z];
}

function createOvalSurfaceGeometry(width, depth, radialSegments = 42, angularSegments = 128) {
  const positions = [0, 0, 0];
  const colors = [0.48, 0.78, 0.90];
  const indices = [];
  const boundary = currentSlotBoundaryPoints(width, depth, angularSegments);
  const segmentCount = boundary.length;

  for (let ir = 1; ir <= radialSegments; ir += 1) {
    const r = ir / radialSegments;
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const [bx, bz] = boundary[ia];
      const x = bx * r;
      const z = bz * r;
      positions.push(x, 0, z);
      colors.push(0.48, 0.78, 0.90);
    }
  }

  for (let ia = 0; ia < segmentCount; ia += 1) {
    indices.push(0, 1 + ia, 1 + ((ia + 1) % segmentCount));
  }

  for (let ir = 1; ir < radialSegments; ir += 1) {
    const inner = 1 + (ir - 1) * segmentCount;
    const outer = 1 + ir * segmentCount;
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const next = (ia + 1) % segmentCount;
      indices.push(inner + ia, outer + ia, outer + next, inner + ia, outer + next, inner + next);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(indices);
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function createOvalElastomerBodyGeometry(width, depth, thickness, radialSegments = 36, angularSegments = 128, sideSegments = 7) {
  const positions = [];
  const colors = [];
  const indices = [];
  const bottomY = thickness;
  const boundary = currentSlotBoundaryPoints(width, depth, angularSegments);
  const segmentCount = boundary.length;

  function vertex(x, y, z) {
    const index = positions.length / 3;
    positions.push(x, y, z);
    colors.push(0.56, 0.80, 0.90);
    return index;
  }

  const bottomCenter = vertex(0, bottomY, 0);
  const bottomRings = [];
  for (let ir = 1; ir <= radialSegments; ir += 1) {
    const r = ir / radialSegments;
    const ring = [];
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const [bx, bz] = boundary[ia];
      const x = bx * r;
      const z = bz * r;
      ring.push(vertex(x, bottomY, z));
    }
    bottomRings.push(ring);
  }
  for (let ia = 0; ia < segmentCount; ia += 1) {
    indices.push(bottomCenter, bottomRings[0][(ia + 1) % segmentCount], bottomRings[0][ia]);
  }
  for (let ir = 0; ir < bottomRings.length - 1; ir += 1) {
    const inner = bottomRings[ir];
    const outer = bottomRings[ir + 1];
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const next = (ia + 1) % segmentCount;
      indices.push(inner[ia], inner[next], outer[next], inner[ia], outer[next], outer[ia]);
    }
  }

  const sideRows = [];
  for (let iy = 0; iy <= sideSegments; iy += 1) {
    const t = iy / sideSegments;
    const y = thickness * t;
    const ring = [];
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const [x, z] = boundary[ia];
      ring.push(vertex(x, y, z));
    }
    sideRows.push(ring);
  }
  for (let iy = 0; iy < sideSegments; iy += 1) {
    const top = sideRows[iy];
    const bottom = sideRows[iy + 1];
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const next = (ia + 1) % segmentCount;
      indices.push(top[ia], bottom[ia], bottom[next], top[ia], bottom[next], top[next]);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(indices);
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function createOvalSurfaceGridGeometry(width, depth, radialSegments = 9, angularSegments = 96, spokes = 16) {
  const positions = [];
  const boundary = currentSlotBoundaryPoints(width, depth, angularSegments);
  const segmentCount = boundary.length;

  for (let ir = 1; ir <= radialSegments; ir += 1) {
    const r = ir / radialSegments;
    for (let ia = 0; ia < segmentCount; ia += 1) {
      const next = (ia + 1) % segmentCount;
      const [bx0, bz0] = boundary[ia];
      const [bx1, bz1] = boundary[next];
      const x0 = bx0 * r;
      const z0 = bz0 * r;
      const x1 = bx1 * r;
      const z1 = bz1 * r;
      positions.push(x0, 0, z0);
      positions.push(x1, 0, z1);
    }
  }

  for (let ia = 0; ia < spokes; ia += 1) {
    const index = Math.round((ia / Math.max(1, spokes)) * segmentCount) % segmentCount;
    const [x, z] = boundary[index];
    positions.push(0, 0, 0);
    positions.push(x, 0, z);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

function createGrooveReferenceLineGeometry(slotTransform = {}) {
  const position = vectorFromConfig(slotTransform.position, [0.546, 0.674, -0.007]);
  const scale = vectorFromConfig(slotTransform.surface_scene_scale, [0.482, 0.268, 0.460]);
  const lift = Number(slotTransform.vertical_lift ?? 0.22);
  const points = currentSlotBoundaryPoints(7, 5, 96);
  const positions = [];
  const lipX = position[0] + (Number.isFinite(lift) ? lift : 0.22) + 0.006;
  for (const [localX, localZ] of points) {
    positions.push(lipX, position[1] + localX * scale[0], position[2] + localZ * scale[2]);
  }
  if (points.length) {
    const [localX, localZ] = points[0];
    positions.push(lipX, position[1] + localX * scale[0], position[2] + localZ * scale[2]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

function topSurfaceShape(x, z, deformation, maxTopDepth) {
  const surfaceValue = surfaceValueAtScene(x, z);
  if (surfaceValue !== null) {
    const patchValue = centroidContactPatchValueAtScene(x, z);
    const deformationValue = patchValue === null ? surfaceValue : patchValue;
    const colorValue = patchValue === null ? surfaceValue : Math.max(patchValue, surfaceValue * 0.52);
    const peak = Math.max(0.05, currentSurfaceVisualPeak(), surfaceValue, deformationValue);
    const normalized = Math.max(0, Math.min(1, colorValue / peak));
    return {
      centerBasin: normalized,
      localDepression: safeSlotDepression(maxTopDepth * Math.pow(Math.max(0, deformationValue), 0.78)),
    };
  }
  const r2 = (x * x) / 6.8 + (z * z) / 4.3;
  const centerBasin = Math.exp(-1.35 * r2);
  const broadBending = Math.exp(-0.34 * r2);
  const profile = 0.68 * centerBasin + 0.32 * broadBending;
  const rimSoftening = 0.035 * deformation * Math.exp(-Math.pow(Math.sqrt(r2) - 1.28, 2) / 0.10);
  return {
    centerBasin,
    localDepression: safeSlotDepression(maxTopDepth * deformation * profile - rimSoftening),
  };
}

function safeSlotDepression(value) {
  return Math.max(0, Math.min(THREE_SLOT_MAX_LOCAL_DEPRESSION, Number(value) || 0));
}

function safeSlotBodyY(value) {
  return Math.max(0, Math.min(THREE_SLOT_MAX_LOCAL_BODY_Y, Number(value) || 0));
}

function bodyElasticShape(x, y, z, deformation, maxTopDepth) {
  const throughThickness = Math.max(
    0,
    Math.min(1, y / THREE_SLOT_BODY_THICKNESS)
  );
  const thicknessFollow =
    1 -
    (1 - THREE_SLOT_BOTTOM_DEFORMATION_FOLLOW) *
      smootherStep(throughThickness);
  const surfaceValue = surfaceValueAtScene(x, z);
  if (surfaceValue !== null) {
    const patchValue = centroidContactPatchValueAtScene(x, z);
    const deformationValue = patchValue === null ? surfaceValue : patchValue;
    const colorValue = patchValue === null ? surfaceValue : Math.max(patchValue, surfaceValue * 0.52);
    const peak = Math.max(0.05, currentSurfaceVisualPeak(), surfaceValue, deformationValue);
    const normalized = Math.max(0, Math.min(1, colorValue / peak));
    const fullSurfaceDepression = safeSlotDepression(
      maxTopDepth * Math.pow(Math.max(0, deformationValue), 0.78)
    );
    const localDepression = fullSurfaceDepression * thicknessFollow;
    return {
      centerBasin: normalized,
      localY: safeSlotBodyY(y + localDepression),
    };
  }
  const r2 = (x * x) / 7.2 + (z * z) / 4.8;
  const centerBasin = Math.exp(-1.18 * r2);
  const broadBending = Math.exp(-0.28 * r2);
  const profile = 0.62 * centerBasin + 0.38 * broadBending;
  const fullSurfaceDepression = safeSlotDepression(
    maxTopDepth * deformation * profile
  );
  const localDepression = fullSurfaceDepression * thicknessFollow;
  return {
    centerBasin,
    localY: safeSlotBodyY(y + localDepression),
  };
}

function vectorFromConfig(values, fallback = [0, 0, 0]) {
  if (!Array.isArray(values)) return fallback.slice();
  return [Number(values[0] ?? fallback[0]) || 0, Number(values[1] ?? fallback[1]) || 0, Number(values[2] ?? fallback[2]) || 0];
}

function radiansFromDegrees(values, fallback = [0, 0, 0]) {
  return vectorFromConfig(values, fallback).map((value) => THREE.MathUtils.degToRad(value));
}

function setObjectTransform(object, transform = {}, fallback = {}) {
  if (!object) return;
  const position = vectorFromConfig(transform.position, fallback.position || [0, 0, 0]);
  const rotation = radiansFromDegrees(transform.rotation_deg, fallback.rotation_deg || [0, 0, 0]);
  const scale = vectorFromConfig(transform.scale, fallback.scale || [1, 1, 1]);
  object.position.set(position[0], position[1], position[2]);
  object.rotation.set(rotation[0], rotation[1], rotation[2]);
  object.scale.set(scale[0] || 1, scale[1] || 1, scale[2] || 1);
  object.visible = transform.visible !== false;
}

function setObjectMatrixFromRowMajor(object, values) {
  if (!object || !Array.isArray(values) || values.length !== 16) return false;
  const numeric = values.map(Number);
  if (!numeric.every(Number.isFinite)) return false;
  object.matrixAutoUpdate = false;
  object.matrix.set(...numeric);
  object.matrixWorldNeedsUpdate = true;
  return true;
}

function normalizedConfigVector(values, fallback) {
  const vector = new THREE.Vector3(...vectorFromConfig(values, fallback));
  if (vector.lengthSq() < 1e-9) {
    vector.set(...fallback);
  }
  return vector.normalize();
}

function selectedFingerLabel() {
  return FINGER_LABELS[state.selectedFinger] || FINGER_LABELS.thumb;
}

function fingerConfig(fingerId) {
  const arrayConfig = state.thumbSceneConfig?.finger_sensor_array;
  const geometryId = arrayConfig?.semantic_to_geometry_id?.[fingerId] || fingerId;
  return arrayConfig?.fingers?.[geometryId] || null;
}

function ensureFingerInteractionProxy(group, fingerId) {
  if (!group) return null;
  let proxy = group.children.find((child) => child.userData?.isFingerInteractionProxy);
  if (!proxy) {
    proxy = new THREE.Mesh(
      new THREE.BoxGeometry(10.4, 6.0, 6.8),
      new THREE.MeshBasicMaterial({
        color: "#ffffff",
        transparent: true,
        opacity: 0,
        depthTest: false,
        depthWrite: false,
      })
    );
    proxy.name = "finger_sensor_interaction_proxy";
    proxy.userData.isFingerInteractionProxy = true;
    proxy.frustumCulled = false;
    group.add(proxy);
  }
  proxy.userData.fingerId = fingerId;
  return proxy;
}

function visibleObjectBounds(root) {
  const bounds = new THREE.Box3();
  const childBounds = new THREE.Box3();
  root?.updateWorldMatrix(true, true);
  root?.traverse((child) => {
    if (
      !child.visible ||
      child.userData?.isFingerInteractionProxy ||
      !child.geometry
    ) {
      return;
    }
    if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
    if (!child.geometry.boundingBox) return;
    childBounds.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld);
    bounds.union(childBounds);
  });
  return bounds;
}

function disposeObject3D(root) {
  if (!root) return;
  const disposedGeometry = new Set();
  const disposedMaterials = new Set();
  const disposedTextures = new Set();
  root.traverse((child) => {
    const geometry = child.geometry;
    if (geometry?.dispose && !disposedGeometry.has(geometry)) {
      disposedGeometry.add(geometry);
      geometry.dispose();
    }
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.filter(Boolean).forEach((material) => {
      if (disposedMaterials.has(material)) return;
      disposedMaterials.add(material);
      Object.values(material).forEach((value) => {
        if (value?.isTexture && value.dispose && !disposedTextures.has(value)) {
          disposedTextures.add(value);
          value.dispose();
        }
      });
      material.dispose?.();
    });
  });
}

function setupFingerSensorGroups() {
  for (const [fingerId, group] of fingerSensorGroups.entries()) {
    if (fingerId !== "thumb") group.parent?.remove(group);
  }
  fingerSensorGroups.clear();
  if (!sensorSurfaceGroup) return;
  fingerSensorGroups.set("thumb", sensorSurfaceGroup);
  sensorSurfaceGroup.userData.fingerId = "thumb";
  sensorSurfaceGroup.traverse((child) => {
    child.userData.fingerId = "thumb";
  });
  ensureFingerInteractionProxy(sensorSurfaceGroup, "thumb");

  for (const fingerId of FINGER_ORDER.filter((item) => item !== "thumb")) {
    const config = fingerConfig(fingerId);
    if (!config || config.enabled === false) continue;
    const group = sensorSurfaceGroup.clone(true);
    group.name = `sensor_slot_surface_group_${fingerId}`;
    group.userData.fingerId = fingerId;
    group.traverse((child) => {
      child.userData.fingerId = fingerId;
      if (child.name === "sensor_slot_surface_outline" && child.material) {
        child.material = child.material.clone();
        child.material.depthTest = true;
        child.renderOrder = 7;
      }
    });
    ensureFingerInteractionProxy(group, fingerId);
    fingerSensorGroups.set(fingerId, group);
    wholeHandRoot?.add(group);
  }
  updateFingerSensorFocusStyles();
}

function applyFingerSensorLayout() {
  const arrayConfig = state.thumbSceneConfig?.finger_sensor_array || {};
  const arrayEnabled = arrayConfig.enabled !== false;
  for (const fingerId of FINGER_ORDER.filter((item) => item !== "thumb")) {
    const group = fingerSensorGroups.get(fingerId);
    const config = fingerConfig(fingerId);
    if (!group || !config) continue;
    if (group.parent !== wholeHandRoot) wholeHandRoot?.add(group);
    const center = new THREE.Vector3(...vectorFromConfig(config.center_model, [0, 0, 0]));
    const outward = normalizedConfigVector(config.outward_normal_model, [0, 0, -1]);
    const longitudinal = normalizedConfigVector(config.longitudinal_axis_model, [0, 1, 0]);
    longitudinal.addScaledVector(outward, -longitudinal.dot(outward)).normalize();
    const inward = outward.clone().multiplyScalar(-1);
    const transverse = new THREE.Vector3().crossVectors(longitudinal, inward).normalize();
    const basis = new THREE.Matrix4().makeBasis(longitudinal, inward, transverse);
    const offset = Number(config.surface_offset_mm ?? 0.08);

    group.matrixAutoUpdate = true;
    group.position.copy(center).addScaledVector(outward, Number.isFinite(offset) ? offset : 0.08);
    group.quaternion.setFromRotationMatrix(basis);
    group.scale.set(
      Math.max(0.1, Number(config.slot_length_mm ?? 15) / 7),
      Math.max(0.1, Number(config.sensor_thickness_scale ?? 1.35)),
      Math.max(0.1, Number(config.slot_width_mm ?? 9) / 5)
    );
    group.visible = arrayEnabled && config.enabled !== false;
    group.userData.slotCenterModel = center.toArray();
    group.userData.longitudinalAxisModel = longitudinal.toArray();
    group.userData.outwardNormalModel = outward.toArray();
    group.userData.inwardNormalModel = inward.toArray();
    group.userData.transverseAxisModel = transverse.toArray();
    group.userData.demoSyncMode = arrayConfig.demo_sync_mode || "synchronized_with_thumb";
  }
  updateFingerSensorFocusStyles();
}

function updateFingerSensorFocusStyles() {
  const selected = state.selectedFinger;
  for (const [fingerId, group] of fingerSensorGroups.entries()) {
    const focused = selected === "all" || selected === fingerId;
    group.userData.focused = focused;
    group.traverse((child) => {
      if (child.name !== "sensor_slot_surface_outline" || !child.material) return;
      child.material.color.set(focused ? "#188fb8" : "#6ea9b8");
      child.material.opacity = focused ? 0.78 : 0.22;
      child.material.needsUpdate = true;
    });
  }
}

function updateFingerScopeLabels() {
  const label = selectedFingerLabel();
  const scope = state.selectedFinger === "all" ? "All fingers" : label;
  setText("spectrumOverviewTitle", `${scope} 9-FBG spectrum`);
  setText("footprintTitle", "Shared 3x3 Contact Map");
  setText(
    "footprintNote",
    state.selectedFinger === "all"
      ? "One response is mirrored across the five fingertip views"
      : `${label} inspection view uses the shared response proxy`
  );
  if (fingerFocusSelect && fingerFocusSelect.value !== state.selectedFinger) {
    fingerFocusSelect.value = state.selectedFinger;
  }
  if (fingerFocusControl) {
    fingerFocusControl.dataset.selectedFinger = state.selectedFinger;
  }
  if (state.geometryDisplayMode === "whole_hand") {
    // Keep the workspace title stable while the selector changes inspection
    // scope. The whole-hand scene remains a five-finger surface even when one
    // fingertip is brought into close-up.
    setText("tactileSurfaceTitle", "Five-Finger Tactile Surface");
    setText(
      "surfaceProxyCaption",
      state.selectedFinger === "all"
        ? "Shared response proxy across five fingertip views"
        : `${label} view | shared response proxy`
    );
  }
  for (const elementId of ["levelBadge", "heatmapChip"]) {
    const element = document.getElementById(elementId);
    if (!element) continue;
    const current = String(element.textContent || "").trim();
    const suffix = current.replace(
      /^(?:Thumb|Index|Middle|Ring|Little|All fingers)\s+/i,
      ""
    );
    element.textContent = `${scope} ${suffix || "surface"}`;
  }
}

function setFingerCloseupNavigationVisible(
  visible,
  { closeup = state.selectedFinger !== "all" } = {}
) {
  const show = Boolean(visible && state.geometryDisplayMode === "whole_hand");
  const closeupMode = Boolean(show && closeup);
  if (previousFingerButton) previousFingerButton.hidden = !show;
  if (nextFingerButton) nextFingerButton.hidden = !show;
  if (threeMount) {
    threeMount.dataset.cameraMode = closeupMode ? "finger-closeup" : "overview";
    threeMount.dataset.focusedFinger = show ? state.selectedFinger : "";
    if (!closeupMode) threeMount.dataset.closeupViewSide = "";
  }
}

function smootherStep(value) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function settleOrbitControlsDamping() {
  if (!controls) return;
  const dampingEnabled = controls.enableDamping;
  controls.enableDamping = false;
  controls.update();
  controls.enableDamping = dampingEnabled;
}

function cancelCameraTransition() {
  cameraTransition = null;
  if (controls) {
    settleOrbitControlsDamping();
    controls.enabled = true;
  }
}

function beginCameraTransition(
  pose,
  {
    closeup = false,
    navigationVisible = closeup,
    durationMs = FINGER_CLOSEUP_DURATION_MS,
  } = {}
) {
  if (!camera || !controls || !pose?.position || !pose?.target) return false;
  settleOrbitControlsDamping();
  const startPosition = camera.position.clone();
  const endPosition = pose.position.clone();
  const startTarget = controls.target.clone();
  const endTarget = pose.target.clone();
  const startUp = camera.up.clone().normalize();
  const endUp = (pose.up || camera.up).clone().normalize();
  const pathLength = Math.max(0.001, startPosition.distanceTo(endPosition));
  const lateral = new THREE.Vector3().crossVectors(
    endPosition.clone().sub(startPosition),
    endUp
  );
  if (lateral.lengthSq() < 1e-5) {
    lateral.crossVectors(endPosition.clone().sub(startPosition), new THREE.Vector3(0, 1, 0));
  }
  if (lateral.lengthSq() > 1e-5) lateral.normalize();
  const controlPoint = startPosition
    .clone()
    .lerp(endPosition, 0.5)
    .addScaledVector(lateral, Math.min(0.34, pathLength * 0.07))
    .addScaledVector(endUp, Math.min(0.20, pathLength * 0.045));

  cameraTransition = {
    active: true,
    startedAt: performance.now(),
    durationMs: Math.max(320, Number(durationMs) || FINGER_CLOSEUP_DURATION_MS),
    startPosition,
    controlPoint,
    endPosition,
    startTarget,
    endTarget,
    startUp,
    endUp,
    startFov: camera.fov,
    endFov: Number(pose.fov) || camera.fov,
    minDistance: Number(pose.minDistance) || controls.minDistance,
    maxDistance: Number(pose.maxDistance) || controls.maxDistance,
    closeup,
  };
  controls.enabled = false;
  state.fingerCloseupActive = closeup;
  setFingerCloseupNavigationVisible(navigationVisible, { closeup });
  if (threeMount) {
    threeMount.dataset.closeupViewSide = closeup ? pose.viewSide || "" : "";
  }
  requestThreeAnimation();
  return true;
}

function updateCameraTransition(timestamp) {
  if (!cameraTransition?.active || !camera || !controls) return false;
  const elapsed = Math.max(0, timestamp - cameraTransition.startedAt);
  const progress = Math.min(1, elapsed / cameraTransition.durationMs);
  const eased = smootherStep(progress);
  const inverse = 1 - eased;

  camera.position
    .copy(cameraTransition.startPosition)
    .multiplyScalar(inverse * inverse)
    .addScaledVector(cameraTransition.controlPoint, 2 * inverse * eased)
    .addScaledVector(cameraTransition.endPosition, eased * eased);
  controls.target.lerpVectors(cameraTransition.startTarget, cameraTransition.endTarget, eased);
  camera.up
    .lerpVectors(cameraTransition.startUp, cameraTransition.endUp, eased)
    .normalize();
  camera.fov =
    cameraTransition.startFov +
    (cameraTransition.endFov - cameraTransition.startFov) * eased;
  camera.updateProjectionMatrix();
  camera.lookAt(controls.target);

  if (progress >= 1) {
    controls.minDistance = cameraTransition.minDistance;
    controls.maxDistance = cameraTransition.maxDistance;
    settleOrbitControlsDamping();
    controls.enabled = true;
    cameraTransition = null;
  }
  return true;
}

function fingerCloseupPose(fingerId) {
  const group = fingerSensorGroups.get(fingerId);
  if (!group || !group.visible || !wholeHandRoot?.visible) return null;
  const bounds = visibleObjectBounds(group);
  if (bounds.isEmpty()) return null;

  const center = bounds.getCenter(new THREE.Vector3());
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const origin = group.localToWorld(new THREE.Vector3(0, 0, 0));
  const sensorFrontNormal = group
    .localToWorld(new THREE.Vector3(0, -1, 0))
    .sub(origin)
    .normalize();
  const longitudinal = group
    .localToWorld(new THREE.Vector3(1, 0, 0))
    .sub(origin)
    .normalize();

  const handBounds = visibleObjectBounds(wholeHandRoot);
  if (!handBounds.isEmpty()) {
    const handCenter = handBounds.getCenter(new THREE.Vector3());
    const radial = center.clone().sub(handCenter);
    if (
      radial.lengthSq() > 1e-5 &&
      sensorFrontNormal.dot(radial) < 0
    ) {
      sensorFrontNormal.negate();
    }
  }
  if (longitudinal.dot(camera.up) < 0) longitudinal.negate();

  const sensorBackNormal = sensorFrontNormal.clone().negate();
  const transverse = new THREE.Vector3().crossVectors(
    longitudinal,
    sensorBackNormal
  );
  if (transverse.lengthSq() < 1e-5) transverse.set(1, 0, 0);
  transverse.normalize();
  const radius = Math.max(0.22, sphere.radius);
  const distance = Math.max(
    1.18,
    Math.min(3.6, radius * FINGER_CLOSEUP_DISTANCE_SCALE)
  );
  const position = center
    .clone()
    .addScaledVector(sensorBackNormal, distance)
    .addScaledVector(transverse, distance * 0.12)
    .addScaledVector(longitudinal, distance * 0.07);

  return {
    position,
    target: center,
    up: longitudinal,
    fov: FINGER_CLOSEUP_FOV,
    minDistance: Math.max(0.38, distance * 0.42),
    maxDistance: Math.max(4.2, distance * 3.8),
    sensorFrontNormal,
    sensorBackNormal,
    viewSide: "sensor-back",
    distance,
  };
}

function focusFingerCloseup(fingerId, { durationMs = FINGER_CLOSEUP_DURATION_MS } = {}) {
  if (!FINGER_ORDER.includes(fingerId)) return false;
  if (state.geometryDisplayMode !== "whole_hand") {
    updateGeometryDisplayMode("whole_hand");
    requestAnimationFrame(() => focusFingerCloseup(fingerId, { durationMs }));
    return true;
  }
  const pose = fingerCloseupPose(fingerId);
  if (!pose) return false;
  return beginCameraTransition(pose, { closeup: true, durationMs });
}

function cycleFingerCloseup(horizontalDirection) {
  const lastIndex = FINGER_RIGHT_NAVIGATION_SEQUENCE.length - 1;
  const currentIndex = Number.isInteger(state.fingerNavigationIndex)
    ? Math.max(0, Math.min(lastIndex, state.fingerNavigationIndex))
    : defaultFingerNavigationIndex(state.selectedFinger);
  const nextIndex =
    horizontalDirection > 0
      ? currentIndex >= lastIndex
        ? 1
        : currentIndex + 1
      : currentIndex <= 0
        ? lastIndex - 1
        : currentIndex - 1;
  setSelectedFinger(FINGER_RIGHT_NAVIGATION_SEQUENCE[nextIndex], {
    focusCamera: true,
    navigationIndex: nextIndex,
  });
}

function defaultFingerNavigationIndex(fingerId) {
  const normalized = String(fingerId || "thumb").toLowerCase();
  const index = FINGER_RIGHT_NAVIGATION_SEQUENCE.indexOf(normalized);
  return index >= 0 ? index : 1;
}

function setSelectedFinger(
  fingerId,
  { focusCamera = false, navigationIndex = null } = {}
) {
  const normalized = String(fingerId || "thumb").toLowerCase();
  state.selectedFinger = [...FINGER_ORDER, "all"].includes(normalized) ? normalized : "thumb";
  state.fingerNavigationIndex = Number.isInteger(navigationIndex)
    ? Math.max(
        0,
        Math.min(FINGER_RIGHT_NAVIGATION_SEQUENCE.length - 1, navigationIndex)
      )
    : defaultFingerNavigationIndex(state.selectedFinger);
  updateFingerSensorFocusStyles();
  updateFingerScopeLabels();
  if (focusCamera) {
    if (state.selectedFinger === "all") {
      applyThumbCameraConfig({
        animated: true,
        keepFingerNavigation: true,
      });
    } else {
      focusFingerCloseup(state.selectedFinger);
    }
  }
}

async function loadThumbSceneConfig() {
  const fallback = {
    thumb_holder_scene: {
      default_geometry_mode: "whole_hand",
      model_asset_url: "",
      fallback_asset_url: "/static/assets/models/thumb_holder.stl",
      fallback_placeholder_enabled: true,
    },
    whole_hand_scene: {
      enabled: true,
      asset_url: "/static/assets/models/robot_nano_hand_sensorized.glb",
      fallback_asset_url: "/static/assets/models/robot_nano_hand_body.glb",
      source_repository_url: "https://github.com/TheRobotStudio/robot-nano-hand",
      source_license: "MIT",
      body_opacity: 0.42,
      body_transform: {
        scale: [0.034, 0.034, 0.034],
        rotation_deg: [0, 0, 0],
        position: [-1.182552, -0.106274, 1.721579],
      },
      modified_thumb_root_matrix_row_major: [
        0.39546837, -0.18512019, -7.85824822, 8.78642365,
        4.55237826, 6.41969769, 0.07786798, 5.78247301,
        6.40797836, -4.54927926, 0.42965253, -107.2841714,
        0, 0, 0, 1,
      ],
      sensor_local_lift: [0.22, 0, 0],
      camera: {
        position: [4.7, 2.2, -8.8],
        target: [0, 0, 0],
      },
    },
    finger_sensor_array: {
      enabled: true,
      geometry_status: "solidworks_tilted_flush_max_area_recesses_integrated",
      manufacturing_status: "provisional_trial_dimensions",
      data_status: "synchronized_demo_only",
      default_selected_finger: "all",
      demo_sync_mode: "synchronized_with_thumb",
      spectrum_scope_mode: "selected_finger",
      array_scope_mode: "selected_finger",
      semantic_to_geometry_id: {
        thumb: "thumb",
        index: "little",
        middle: "ring",
        ring: "middle",
        little: "index",
      },
      source_asset_url: "/static/assets/models/robot_nano_hand_sensorized.glb",
      original_asset_url: "/static/assets/models/robot_nano_hand_body.glb",
      fingers: {
        thumb: {
          label: "Thumb",
          enabled: true,
          sensor_source: "existing_thumb_slot",
        },
        index: {
          label: "Index",
          enabled: true,
          center_model: [79.112326, 63.501709, -38.861167],
          longitudinal_axis_model: [0.185475, 0.940719, -0.283983],
          outward_normal_model: [-0.260970, -0.231463, -0.937187],
          slot_length_mm: 12.444752,
          slot_width_mm: 11.087143,
          slot_depth_mm: 0.724058,
          sensor_thickness_scale: 1.508455,
          surface_offset_mm: 0.04,
        },
        middle: {
          label: "Middle",
          enabled: true,
          center_model: [54.689092, 82.199535, -40.460975],
          longitudinal_axis_model: [0.070253, 0.834482, -0.546539],
          outward_normal_model: [-0.214731, -0.522397, -0.825223],
          slot_length_mm: 13.353811,
          slot_width_mm: 12.161507,
          slot_depth_mm: 0.763075,
          sensor_thickness_scale: 1.589739,
          surface_offset_mm: 0.04,
        },
        ring: {
          label: "Ring",
          enabled: true,
          center_model: [23.346004, 101.594176, -18.297977],
          longitudinal_axis_model: [0.007739, 0.988024, -0.154105],
          outward_normal_model: [-0.070573, -0.153186, -0.985674],
          slot_length_mm: 13.961495,
          slot_width_mm: 12.714933,
          slot_depth_mm: 0.7978,
          sensor_thickness_scale: 1.662083,
          surface_offset_mm: 0.04,
        },
        little: {
          label: "Little",
          enabled: true,
          center_model: [-10.236739, 92.434478, -23.299121],
          longitudinal_axis_model: [-0.106891, 0.971511, -0.211518],
          outward_normal_model: [0.024749, -0.210072, -0.977373],
          slot_length_mm: 14.527000,
          slot_width_mm: 11.786057,
          slot_depth_mm: 0.877102,
          sensor_thickness_scale: 1.827296,
          surface_offset_mm: 0.04,
        },
      },
    },
    thumb_model_transform: {
      scale: [1, 1, 1],
      rotation_deg: [0, 0, 90],
      position: [0, -0.55, 0],
      holder_local_rotation_deg: [0, 0, -14.098],
      holder_local_position: [0, 0.06, 0],
      holder_local_scale: [1, 1, 1],
      opacity: 0.34,
      visible: true,
      wireframe: false,
    },
    sensor_slot_transform: {
      coordinate_space: "thumb_model_local",
      slot_shape: "stl_irregular_oval",
      position: [0.546, 0.674, -0.007],
      rotation_deg: [0, 0, 90],
      vertical_lift: 0.22,
      surface_scene_scale: [0.482, 0.268, 0.460],
      boundary_profile: {
        source: "stl_diff_groove_faces",
        angular_samples: 64,
        radial_scale: DEFAULT_SLOT_BOUNDARY_PROFILE,
      },
      boundary_points_local: DEFAULT_SLOT_BOUNDARY_POINTS,
      width_mm: 10,
      height_mm: 7,
      depth_mm: 1.0,
      z_offset_mm: 0.08,
      visible: true,
      opacity: 1,
    },
    visual_style: {},
  };
  let resolved = fallback;
  try {
    const payload = await requestJSON(
      "/api/thumb_scene_config",
      { cache: "no-store" },
      { timeoutMs: 5000 }
    );
    if (payload?.ok && payload?.config) {
      resolved = { ...fallback };
      Object.entries(payload.config).forEach(([key, value]) => {
        resolved[key] = value && typeof value === "object" && !Array.isArray(value)
          ? { ...(fallback[key] || {}), ...value }
          : value;
      });
    }
  } catch {
    // The embedded fallback keeps the scene available while the backend starts.
  }
  state.thumbSceneConfig = resolved;
  state.thumbModelAssetUrl = fallback.thumb_holder_scene.fallback_asset_url;
  state.wholeHandModelAssetUrl = resolved.whole_hand_scene?.asset_url || fallback.whole_hand_scene.asset_url;
  const configuredFinger = String(
    resolved.finger_sensor_array?.default_selected_finger || "all"
  ).toLowerCase();
  state.selectedFinger = [...FINGER_ORDER, "all"].includes(configuredFinger)
    ? configuredFinger
    : "all";
  state.fingerNavigationIndex = defaultFingerNavigationIndex(state.selectedFinger);
  if (fingerFocusSelect) fingerFocusSelect.value = state.selectedFinger;
  state.thumbModelMessage = "using local STL thumb holder fallback";
  const configuredMode = state.thumbSceneConfig?.thumb_holder_scene?.default_geometry_mode;
  state.geometryDisplayMode = ["whole_hand", "thumb_holder", "surface_only"].includes(configuredMode)
    ? configuredMode
    : "whole_hand";
  populateThumbAlignmentPanel();
}

function alignmentInput(id, value) {
  const element = document.getElementById(id);
  if (element) element.value = value ?? "";
}

function populateThumbAlignmentPanel() {
  const config = state.thumbSceneConfig || {};
  const model = config.thumb_model_transform || {};
  const slot = config.sensor_slot_transform || {};
  const modelPosition = vectorFromConfig(model.position, [0, -0.55, 0]);
  const modelRotation = vectorFromConfig(model.rotation_deg, [0, 0, 90]);
  const modelScale = vectorFromConfig(model.scale, [1, 1, 1]);
  const slotPosition = vectorFromConfig(slot.position, [0.546, 0.674, -0.007]);
  const slotRotation = vectorFromConfig(slot.rotation_deg, [0, 0, 90]);
  const slotScale = vectorFromConfig(slot.surface_scene_scale, [0.482, 0.268, 0.460]);
  [
    ["thumbModelVisible", model.visible !== false],
    ["thumbSurfaceVisible", slot.visible !== false],
    ["thumbWireframeToggle", Boolean(model.wireframe)],
  ].forEach(([id, checked]) => {
    const element = document.getElementById(id);
    if (element) element.checked = Boolean(checked);
  });
  alignmentInput("thumbModelScaleX", modelScale[0]);
  alignmentInput("thumbModelScaleY", modelScale[1]);
  alignmentInput("thumbModelScaleZ", modelScale[2]);
  alignmentInput("thumbModelRotX", modelRotation[0]);
  alignmentInput("thumbModelRotY", modelRotation[1]);
  alignmentInput("thumbModelRotZ", modelRotation[2]);
  alignmentInput("thumbModelPosX", modelPosition[0]);
  alignmentInput("thumbModelPosY", modelPosition[1]);
  alignmentInput("thumbModelPosZ", modelPosition[2]);
  alignmentInput("thumbSlotPosX", slotPosition[0]);
  alignmentInput("thumbSlotPosY", slotPosition[1]);
  alignmentInput("thumbSlotPosZ", slotPosition[2]);
  alignmentInput("thumbSlotRotX", slotRotation[0]);
  alignmentInput("thumbSlotRotY", slotRotation[1]);
  alignmentInput("thumbSlotRotZ", slotRotation[2]);
  alignmentInput("thumbSlotScaleX", slotScale[0]);
  alignmentInput("thumbSlotScaleY", slotScale[1]);
  alignmentInput("thumbSlotScaleZ", slotScale[2]);
  alignmentInput("thumbSlotWidthMm", slot.width_mm ?? 10);
  alignmentInput("thumbSlotHeightMm", slot.height_mm ?? 7);
  alignmentInput("thumbSlotDepthMm", slot.depth_mm ?? 1);
  alignmentInput("thumbModelOpacity", model.opacity ?? 0.34);
  setText("thumbModelLoadStatus", state.thumbModelStatus || "not_loaded");
  setText("thumbModelAssetUrl", state.thumbModelAssetUrl || "--");
  setText("thumbModelMessage", state.thumbModelMessage || "--");
  const assetStatus = document.getElementById("thumbModelAssetUrl");
  const modelMessage = document.getElementById("thumbModelMessage");
  if (assetStatus) assetStatus.title = state.thumbModelAssetUrl || "--";
  if (modelMessage) modelMessage.title = state.thumbModelMessage || "--";
}

function numberInputValue(id, fallback = 0) {
  const value = Number(document.getElementById(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function collectThumbAlignmentConfig() {
  return {
    thumb_holder_scene: {
      default_geometry_mode: state.geometryDisplayMode,
      model_asset_url: "",
      fallback_asset_url: "/static/assets/models/thumb_holder.stl",
      fallback_placeholder_enabled: false,
      model_load_policy: "glb_then_stl_else_blocked",
    },
    thumb_model_transform: {
      visible: document.getElementById("thumbModelVisible")?.checked !== false,
      wireframe: Boolean(document.getElementById("thumbWireframeToggle")?.checked),
      scale: [numberInputValue("thumbModelScaleX", 1), numberInputValue("thumbModelScaleY", 1), numberInputValue("thumbModelScaleZ", 1)],
      rotation_deg: [numberInputValue("thumbModelRotX", 0), numberInputValue("thumbModelRotY", 0), numberInputValue("thumbModelRotZ", 90)],
      position: [numberInputValue("thumbModelPosX", 0), numberInputValue("thumbModelPosY", -0.55), numberInputValue("thumbModelPosZ", 0)],
      holder_local_rotation_deg: vectorFromConfig(state.thumbSceneConfig?.thumb_model_transform?.holder_local_rotation_deg, [0, 0, -14.098]),
      holder_local_position: vectorFromConfig(state.thumbSceneConfig?.thumb_model_transform?.holder_local_position, [0, 0.06, 0]),
      holder_local_scale: vectorFromConfig(state.thumbSceneConfig?.thumb_model_transform?.holder_local_scale, [1, 1, 1]),
      opacity: numberInputValue("thumbModelOpacity", 0.34),
    },
    sensor_slot_transform: {
      coordinate_space: "thumb_model_local",
      slot_shape: "stl_irregular_oval",
      visible: document.getElementById("thumbSurfaceVisible")?.checked !== false,
      position: [numberInputValue("thumbSlotPosX", 0.546), numberInputValue("thumbSlotPosY", 0.674), numberInputValue("thumbSlotPosZ", -0.007)],
      rotation_deg: [numberInputValue("thumbSlotRotX", 0), numberInputValue("thumbSlotRotY", 0), numberInputValue("thumbSlotRotZ", 90)],
      vertical_lift: Number(state.thumbSceneConfig?.sensor_slot_transform?.vertical_lift ?? 0.22),
      surface_scene_scale: [numberInputValue("thumbSlotScaleX", 0.482), numberInputValue("thumbSlotScaleY", 0.268), numberInputValue("thumbSlotScaleZ", 0.460)],
      boundary_profile: state.thumbSceneConfig?.sensor_slot_transform?.boundary_profile || {
        source: "stl_diff_groove_faces",
        angular_samples: 64,
        radial_scale: DEFAULT_SLOT_BOUNDARY_PROFILE,
      },
      boundary_points_local: state.thumbSceneConfig?.sensor_slot_transform?.boundary_points_local || DEFAULT_SLOT_BOUNDARY_POINTS,
      width_mm: numberInputValue("thumbSlotWidthMm", 10),
      height_mm: numberInputValue("thumbSlotHeightMm", 7),
      depth_mm: numberInputValue("thumbSlotDepthMm", 1),
      z_offset_mm: 0.08,
    },
  };
}

function resetThumbAlignmentConfig() {
  state.thumbSceneConfig = {
    ...(state.thumbSceneConfig || {}),
    thumb_holder_scene: {
      ...(state.thumbSceneConfig?.thumb_holder_scene || {}),
      default_geometry_mode: "thumb_holder",
      model_asset_url: "",
      fallback_asset_url: "/static/assets/models/thumb_holder.stl",
      fallback_placeholder_enabled: false,
      model_load_policy: "glb_then_stl_else_blocked",
    },
    thumb_model_transform: {
      visible: true,
      wireframe: false,
      scale: [1, 1, 1],
      rotation_deg: [0, 0, 90],
      position: [0, -0.55, 0],
      holder_local_rotation_deg: [0, 0, -14.098],
      holder_local_position: [0, 0.06, 0],
      holder_local_scale: [1, 1, 1],
      opacity: 0.42,
    },
    sensor_slot_transform: {
      coordinate_space: "thumb_model_local",
      slot_shape: "stl_irregular_oval",
      visible: true,
      position: [0.546, 0.674, -0.007],
      rotation_deg: [0, 0, 90],
      vertical_lift: 0.22,
      surface_scene_scale: [0.482, 0.268, 0.460],
      boundary_profile: {
        source: "stl_diff_groove_faces",
        angular_samples: 64,
        radial_scale: DEFAULT_SLOT_BOUNDARY_PROFILE,
      },
      boundary_points_local: DEFAULT_SLOT_BOUNDARY_POINTS,
      width_mm: 10,
      height_mm: 7,
      depth_mm: 1,
      z_offset_mm: 0.08,
    },
  };
  state.geometryDisplayMode = "thumb_holder";
  populateThumbAlignmentPanel();
  updateGeometryDisplayMode("thumb_holder");
}

function placeSceneGridBelowModel() {
  if (!sceneGrid) return;
  const wholeHandMode = state.geometryDisplayMode === "whole_hand";
  const thumbMode = state.geometryDisplayMode === "thumb_holder";
  const referenceObject = wholeHandMode ? wholeHandRoot : thumbMode ? thumbModelRoot : sensorSurfaceGroup;
  const clearance = wholeHandMode ? 0.30 : thumbMode ? 0.52 : 0.38;
  let boundsMinY = wholeHandMode ? -3.9 : thumbMode ? -1.62 : -0.66;

  if (referenceObject) {
    referenceObject.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(referenceObject);
    if (!bounds.isEmpty() && Number.isFinite(bounds.min.y)) {
      boundsMinY = bounds.min.y;
    }
  }

  sceneGrid.position.y = boundsMinY - clearance;
  sceneGrid.userData.referenceBoundsMinY = boundsMinY;
  sceneGrid.userData.clearance = clearance;
}

function applyThumbSceneLayout() {
  const config = state.thumbSceneConfig || {};
  const modelTransform = config.thumb_model_transform || {};
  const slotTransform = config.sensor_slot_transform || {};
  const wholeHandConfig = config.whole_hand_scene || {};
  const wholeHandMode = state.geometryDisplayMode === "whole_hand";
  const thumbMode = state.geometryDisplayMode === "thumb_holder";
  const physicalMode = wholeHandMode || thumbMode;

  if (wholeHandRoot) {
    wholeHandRoot.matrixAutoUpdate = true;
    setObjectTransform(wholeHandRoot, wholeHandConfig.body_transform || {}, {
      position: [-1.182552, -0.106274, 1.721579],
      rotation_deg: [0, 0, 0],
      scale: [0.034, 0.034, 0.034],
    });
    wholeHandRoot.visible = wholeHandMode && wholeHandConfig.enabled !== false;
  }
  if (wholeHandBodyObject) {
    wholeHandBodyObject.visible = wholeHandMode && wholeHandConfig.enabled !== false;
  }
  if (thumbModelRoot) {
    if (wholeHandMode && wholeHandRoot) {
      if (thumbModelRoot.parent !== wholeHandRoot) wholeHandRoot.add(thumbModelRoot);
      const matrixApplied = setObjectMatrixFromRowMajor(
        thumbModelRoot,
        wholeHandConfig.modified_thumb_root_matrix_row_major
      );
      if (!matrixApplied) {
        thumbModelRoot.matrixAutoUpdate = true;
        setObjectTransform(thumbModelRoot, {}, {
          position: [8.786424, 5.782473, -107.284171],
          rotation_deg: [0, 0, 0],
          scale: [7.87037, 7.87037, 7.87037],
        });
      }
    } else {
      if (thumbModelRoot.parent !== scene) scene.add(thumbModelRoot);
      thumbModelRoot.matrixAutoUpdate = true;
      setObjectTransform(thumbModelRoot, modelTransform, {
        position: [0, -0.55, 0],
        rotation_deg: [0, 0, 90],
        scale: [1, 1, 1],
      });
      thumbModelRoot.updateMatrix();
    }
    thumbModelRoot.visible = physicalMode && modelTransform.visible !== false;
  }
  if (thumbHolderObject) {
    setObjectTransform(
      thumbHolderObject,
      {
        visible: modelTransform.visible !== false,
        position: vectorFromConfig(modelTransform.holder_local_position, [0, 0.06, 0]),
        rotation_deg: vectorFromConfig(modelTransform.holder_local_rotation_deg, [0, 0, -14.098]),
        scale: vectorFromConfig(modelTransform.holder_local_scale, [1, 1, 1]),
      },
      {
        position: [0, 0, 0],
        rotation_deg: [0, 0, 0],
        scale: [1, 1, 1],
      }
    );
    thumbHolderObject.visible = physicalMode && modelTransform.visible !== false;
  }
  if (sensorSurfaceGroup) {
    if (physicalMode) {
      const localToThumb = slotTransform.coordinate_space === "thumb_model_local" && thumbModelRoot;
      const desiredParent = localToThumb ? thumbModelRoot : scene;
      if (sensorSurfaceGroup.parent !== desiredParent) desiredParent.add(sensorSurfaceGroup);
      const position = vectorFromConfig(slotTransform.position, localToThumb ? [0.546, 0.674, -0.007] : [0, -1.52, -0.05]);
      const rotation = radiansFromDegrees(slotTransform.rotation_deg, localToThumb ? [0, 0, 90] : [0, 0, 0]);
      const scale = vectorFromConfig(slotTransform.surface_scene_scale, localToThumb ? [0.482, 0.268, 0.460] : [0.34, 0.34, 0.24]);
      sensorSurfaceGroup.position.set(position[0], position[1], position[2]);
      if (localToThumb) {
        if (wholeHandMode) {
          const localLift = vectorFromConfig(wholeHandConfig.sensor_local_lift, [0.22, 0, 0]);
          sensorSurfaceGroup.position.add(new THREE.Vector3(localLift[0], localLift[1], localLift[2]));
        } else {
          const lift = Number(slotTransform.vertical_lift ?? 0.22);
          if (Number.isFinite(lift) && Math.abs(lift) > 1e-6) {
            thumbModelRoot.updateMatrixWorld(true);
            const parentWorldQuaternion = thumbModelRoot.getWorldQuaternion(new THREE.Quaternion());
            const localLift = new THREE.Vector3(0, lift, 0).applyQuaternion(parentWorldQuaternion.invert());
            sensorSurfaceGroup.position.add(localLift);
          }
        }
      }
      sensorSurfaceGroup.rotation.set(rotation[0], rotation[1], rotation[2]);
      sensorSurfaceGroup.scale.set(scale[0] || 0.34, scale[1] || 0.17, scale[2] || 0.36);
      sensorSurfaceGroup.visible = slotTransform.visible !== false;
    } else {
      if (sensorSurfaceGroup.parent !== scene) scene.add(sensorSurfaceGroup);
      sensorSurfaceGroup.position.set(0, 0, 0);
      sensorSurfaceGroup.rotation.set(0, 0, 0);
      // The inspection view has a full-width stage, so the pad can be larger
      // while retaining enough edge clearance for rotation and deformation.
      sensorSurfaceGroup.scale.set(-0.92, -0.92, 0.92);
      sensorSurfaceGroup.visible = true;
    }
  }
  if (grooveReferenceHelper) {
    const localToThumb = physicalMode && slotTransform.coordinate_space === "thumb_model_local" && thumbModelRoot;
    const desiredParent = localToThumb ? thumbModelRoot : scene;
    if (grooveReferenceHelper.parent !== desiredParent) desiredParent.add(grooveReferenceHelper);
    grooveReferenceHelper.geometry?.dispose?.();
    grooveReferenceHelper.geometry = createGrooveReferenceLineGeometry(slotTransform);
    grooveReferenceHelper.position.set(0, 0, 0);
    grooveReferenceHelper.rotation.set(0, 0, 0);
    grooveReferenceHelper.scale.set(1, 1, 1);
    grooveReferenceHelper.visible = physicalMode && slotTransform.visible !== false;
  }
  if (slotOutlineHelper) {
    slotOutlineHelper.visible = physicalMode;
  }
  applyFingerSensorLayout();
  updateFingerScopeLabels();
  placeSceneGridBelowModel();
  state.threeNeedsRefresh = true;
}

function wholeHandOverviewPose(configuredPose) {
  if (!wholeHandRoot?.visible || !camera) return null;
  const bounds = visibleObjectBounds(wholeHandRoot);
  if (bounds.isEmpty()) return null;

  const center = bounds.getCenter(new THREE.Vector3());
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  if (!Number.isFinite(sphere.radius) || sphere.radius <= 0) return null;

  const viewDirection = configuredPose.position
    .clone()
    .sub(configuredPose.target);
  if (viewDirection.lengthSq() < 1e-6) viewDirection.set(0.46, 0.22, -0.86);
  viewDirection.normalize();

  const verticalHalfFov = THREE.MathUtils.degToRad(
    WHOLE_HAND_OVERVIEW_FOV * 0.5
  );
  const horizontalHalfFov = Math.atan(
    Math.tan(verticalHalfFov) * Math.max(0.25, camera.aspect || 1)
  );
  const viewUp = configuredPose.up
    .clone()
    .addScaledVector(viewDirection, -configuredPose.up.dot(viewDirection));
  if (viewUp.lengthSq() < 1e-6) viewUp.set(0, 1, 0);
  viewUp.normalize();
  const viewRight = new THREE.Vector3()
    .crossVectors(viewUp, viewDirection)
    .normalize();
  const tanVertical = Math.max(0.08, Math.tan(verticalHalfFov));
  const tanHorizontal = Math.max(0.08, Math.tan(horizontalHalfFov));
  let distance = sphere.radius * 1.35;
  const cornerOffsets = [];
  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) {
        const offset = new THREE.Vector3(x, y, z).sub(center);
        cornerOffsets.push(offset);
        const depthOffset = offset.dot(viewDirection);
        const verticalDistance =
          depthOffset +
          (Math.abs(offset.dot(viewUp)) * WHOLE_HAND_OVERVIEW_MARGIN) /
            tanVertical;
        const horizontalDistance =
          depthOffset +
          (Math.abs(offset.dot(viewRight)) * WHOLE_HAND_OVERVIEW_MARGIN) /
            tanHorizontal;
        distance = Math.max(
          distance,
          verticalDistance,
          horizontalDistance
        );
      }
    }
  }
  distance += sphere.radius * 0.06;

  let minProjectedX = Number.POSITIVE_INFINITY;
  let maxProjectedX = Number.NEGATIVE_INFINITY;
  let minProjectedY = Number.POSITIVE_INFINITY;
  let maxProjectedY = Number.NEGATIVE_INFINITY;
  for (const offset of cornerOffsets) {
    const depth = Math.max(
      sphere.radius * 0.12,
      distance - offset.dot(viewDirection)
    );
    const projectedX = offset.dot(viewRight) / (depth * tanHorizontal);
    const projectedY = offset.dot(viewUp) / (depth * tanVertical);
    minProjectedX = Math.min(minProjectedX, projectedX);
    maxProjectedX = Math.max(maxProjectedX, projectedX);
    minProjectedY = Math.min(minProjectedY, projectedY);
    maxProjectedY = Math.max(maxProjectedY, projectedY);
  }
  const projectedMidX = (minProjectedX + maxProjectedX) * 0.5;
  const projectedMidY = (minProjectedY + maxProjectedY) * 0.5;
  const framedCenter = center
    .clone()
    .addScaledVector(
      viewRight,
      projectedMidX * distance * tanHorizontal
    )
    .addScaledVector(
      viewUp,
      (projectedMidY + WHOLE_HAND_OVERVIEW_VERTICAL_BIAS) *
        distance *
        tanVertical
    );

  return {
    position: framedCenter.clone().addScaledVector(viewDirection, distance),
    target: framedCenter,
    up: configuredPose.up.clone(),
    fov: WHOLE_HAND_OVERVIEW_FOV,
    minDistance: Math.max(2.4, sphere.radius * 0.78),
    maxDistance: Math.max(18, distance * 2.2),
  };
}

function applyThumbCameraConfig({
  animated = false,
  keepFingerNavigation = true,
} = {}) {
  if (!camera || !controls) return;
  const wholeHandMode = state.geometryDisplayMode === "whole_hand";
  const thumbMode = state.geometryDisplayMode === "thumb_holder";
  const cameraConfig = state.thumbSceneConfig?.scene_camera || {};
  const wholeHandCamera = state.thumbSceneConfig?.whole_hand_scene?.camera || {};
  const configuredPosition = wholeHandMode
    ? vectorFromConfig(wholeHandCamera.position, [4.7, 2.2, -8.8])
    : thumbMode
      ? vectorFromConfig(cameraConfig.position, [5.2, 2.7, 4.8])
      : [0, 9.3, 5.8];
  const configuredTarget = wholeHandMode
    ? vectorFromConfig(wholeHandCamera.target, [0, 0, 0])
    : thumbMode
      ? vectorFromConfig(cameraConfig.target, [0, -1.0, 0])
      : [0, -0.18, 0];
  const configuredPose = {
    position: new THREE.Vector3(
      configuredPosition[0],
      configuredPosition[1],
      configuredPosition[2]
    ),
    target: new THREE.Vector3(
      configuredTarget[0],
      configuredTarget[1],
      configuredTarget[2]
    ),
    up: new THREE.Vector3(
      thumbMode || wholeHandMode ? 0 : -1,
      thumbMode || wholeHandMode ? 1 : 0,
      0
    ),
    fov: wholeHandMode ? WHOLE_HAND_OVERVIEW_FOV : thumbMode ? 43 : 39,
    minDistance: wholeHandMode ? 5.0 : thumbMode ? 4.0 : 7.2,
    maxDistance: wholeHandMode ? 18.0 : thumbMode ? 13.0 : 14.0,
  };
  const pose =
    wholeHandMode && state.selectedFinger === "all"
      ? wholeHandOverviewPose(configuredPose) || configuredPose
      : configuredPose;
  if (animated) {
    beginCameraTransition(pose, {
      closeup: false,
      navigationVisible: keepFingerNavigation && wholeHandMode,
      durationMs: wholeHandMode ? FINGER_OVERVIEW_DURATION_MS : 900,
    });
    return;
  }
  cancelCameraTransition();
  state.fingerCloseupActive = false;
  setFingerCloseupNavigationVisible(
    keepFingerNavigation && wholeHandMode,
    { closeup: false }
  );
  camera.up.copy(pose.up);
  camera.fov = pose.fov;
  camera.updateProjectionMatrix();
  controls.minDistance = pose.minDistance;
  controls.maxDistance = pose.maxDistance;
  camera.position.copy(pose.position);
  controls.target.copy(pose.target);
  camera.lookAt(controls.target);
  controls.update();
}

function applySceneLighting(physicalMode) {
  const wholeHandMode = state.geometryDisplayMode === "whole_hand";
  if (sceneAmbientLight) {
    sceneAmbientLight.intensity = wholeHandMode ? 0.72 : physicalMode ? 2.2 : 0.85;
  }
  if (sceneKeyLight) {
    sceneKeyLight.intensity = wholeHandMode ? 5.4 : physicalMode ? 2.8 : 3.4;
    sceneKeyLight.position.set(
      wholeHandMode ? 4.5 : physicalMode ? 3 : 4.5,
      wholeHandMode ? 6.5 : physicalMode ? 6 : 7.5,
      wholeHandMode ? -6.0 : physicalMode ? 4 : 5.8
    );
  }
  if (sceneFillLight) {
    sceneFillLight.intensity = wholeHandMode ? 1.8 : physicalMode ? 0.9 : 1.2;
    sceneFillLight.position.set(
      wholeHandMode ? -4.8 : -3.5,
      wholeHandMode ? 2.4 : 2.0,
      wholeHandMode ? 5.5 : -4.0
    );
  }
}

function exposeThreeDebugHandle() {
  if (typeof window === "undefined") return;
  window.__bayspec3dDebug = {
    setCamera(position = [5.2, 2.7, 4.8], target = [0, -1.0, 0]) {
      if (!camera || !controls) return false;
      const p = vectorFromConfig(position, [5.2, 2.7, 4.8]);
      const t = vectorFromConfig(target, [0, -1.0, 0]);
      camera.position.set(p[0], p[1], p[2]);
      controls.target.set(t[0], t[1], t[2]);
      camera.lookAt(controls.target);
      controls.update();
      state.threeNeedsRefresh = true;
      return true;
    },
    setHolderLocalRotation(rotationDeg = [0, 0, 0], position = [0, 0, 0]) {
      state.thumbSceneConfig = {
        ...(state.thumbSceneConfig || {}),
        thumb_model_transform: {
          ...(state.thumbSceneConfig?.thumb_model_transform || {}),
          holder_local_rotation_deg: vectorFromConfig(rotationDeg, [0, 0, 0]),
          holder_local_position: vectorFromConfig(position, [0, 0, 0]),
        },
      };
      applyThumbSceneLayout();
      return state.thumbSceneConfig.thumb_model_transform;
    },
    setSensorSlotTransform(slotTransform = {}) {
      state.thumbSceneConfig = {
        ...(state.thumbSceneConfig || {}),
        sensor_slot_transform: {
          ...(state.thumbSceneConfig?.sensor_slot_transform || {}),
          ...slotTransform,
        },
      };
      applyThumbSceneLayout();
      return state.thumbSceneConfig.sensor_slot_transform;
    },
    getThumbSceneConfig() {
      return state.thumbSceneConfig;
    },
    getSceneGridPlacement() {
      return {
        gridY: sceneGrid?.position?.y ?? null,
        referenceBoundsMinY: sceneGrid?.userData?.referenceBoundsMinY ?? null,
        clearance: sceneGrid?.userData?.clearance ?? null,
      };
    },
    setSelectedFinger(fingerId) {
      setSelectedFinger(fingerId);
      return state.selectedFinger;
    },
    focusFinger(fingerId, durationMs = FINGER_CLOSEUP_DURATION_MS) {
      const normalized = String(fingerId || "").toLowerCase();
      if (!FINGER_ORDER.includes(normalized)) return false;
      setSelectedFinger(normalized);
      return focusFingerCloseup(normalized, { durationMs });
    },
    showWholeHandOverview(animated = true) {
      setSelectedFinger("all");
      applyThumbCameraConfig({
        animated,
        keepFingerNavigation: true,
      });
      return true;
    },
    getCameraTransitionState() {
      return {
        active: Boolean(cameraTransition?.active),
        closeup: Boolean(state.fingerCloseupActive),
        selectedFinger: state.selectedFinger,
        cameraMode: threeMount?.dataset.cameraMode || "overview",
        cameraPosition: camera?.position?.toArray?.() || null,
        cameraTarget: controls?.target?.toArray?.() || null,
      };
    },
    getFingerSensorStatus() {
      const sensors = {};
      for (const [fingerId, group] of fingerSensorGroups.entries()) {
        group.updateMatrixWorld(true);
        const bounds = new THREE.Box3().setFromObject(group);
        const worldCenter = new THREE.Vector3();
        group.getWorldPosition(worldCenter);
        sensors[fingerId] = {
          visible: group.visible,
          focused: Boolean(group.userData.focused),
          parent: group.parent?.name || null,
          worldCenter: worldCenter.toArray(),
          bounds:
            bounds.isEmpty()
              ? null
              : {
                  min: bounds.min.toArray(),
                  max: bounds.max.toArray(),
                },
          demoSyncMode: group.userData.demoSyncMode || "shared_geometry",
        };
      }
      return {
        selectedFinger: state.selectedFinger,
        geometryDisplayMode: state.geometryDisplayMode,
        sensors,
      };
    },
    getModelStatus() {
      const wholeBounds =
        wholeHandRoot && wholeHandRoot.visible
          ? new THREE.Box3().setFromObject(wholeHandRoot)
          : null;
      return {
        geometryDisplayMode: state.geometryDisplayMode,
        thumbModelStatus: state.thumbModelStatus,
        thumbModelMessage: state.thumbModelMessage,
        wholeHandModelStatus: state.wholeHandModelStatus,
        wholeHandModelMessage: state.wholeHandModelMessage,
        wholeHandRootVisible: wholeHandRoot?.visible ?? false,
        wholeHandBodyPresent: Boolean(wholeHandBodyObject),
        wholeHandBodyVisible: wholeHandBodyObject?.visible ?? false,
        wholeHandBounds:
          wholeBounds && !wholeBounds.isEmpty()
            ? {
                min: wholeBounds.min.toArray(),
                max: wholeBounds.max.toArray(),
              }
            : null,
      };
    },
  };
}

async function setupThumbHolderModel() {
  if (!scene || !thumbModelRoot) return;
  const result = await loadThumbHolderModel(state.thumbSceneConfig || {});
  if (thumbHolderObject) {
    thumbModelRoot.remove(thumbHolderObject);
    disposeObject3D(thumbHolderObject);
  }
  thumbHolderObject = result.object;
  if (thumbHolderObject) {
    thumbModelRoot.add(thumbHolderObject);
  }
  state.thumbModelStatus = result.status;
  state.thumbModelMessage = result.message;
  state.thumbModelAssetUrl = result.assetUrl || state.thumbModelAssetUrl;
  state.thumbModelFallbackUsed = Boolean(result.fallback);
  populateThumbAlignmentPanel();
  applyThumbSceneLayout();
  requestThreeAnimation();
}

async function setupWholeHandModel() {
  if (!scene || !wholeHandRoot) return;
  const result = await loadRobotNanoHandModel(state.thumbSceneConfig || {});
  if (wholeHandBodyObject) {
    wholeHandRoot.remove(wholeHandBodyObject);
    disposeObject3D(wholeHandBodyObject);
  }
  wholeHandBodyObject = result.object;
  if (wholeHandBodyObject) {
    wholeHandRoot.add(wholeHandBodyObject);
  }
  state.wholeHandModelStatus = result.status;
  state.wholeHandModelMessage = result.message;
  state.wholeHandModelAssetUrl = result.assetUrl || state.wholeHandModelAssetUrl;
  if (threeMount) {
    threeMount.dataset.wholeHandStatus = result.status;
    threeMount.dataset.wholeHandAsset = state.wholeHandModelAssetUrl;
  }
  if (!wholeHandBodyObject) {
    console.warn(`[whole-hand] ${result.message || "model unavailable"}`);
  }
  applyThumbSceneLayout();
  requestThreeAnimation();
}

function updateGeometryDisplayMode(mode) {
  state.geometryDisplayMode = ["whole_hand", "thumb_holder", "surface_only"].includes(mode)
    ? mode
    : "thumb_holder";
  const wholeHandMode = state.geometryDisplayMode === "whole_hand";
  const thumbMode = state.geometryDisplayMode === "thumb_holder";
  const surfaceOnlyMode = state.geometryDisplayMode === "surface_only";
  const physicalMode = wholeHandMode || thumbMode;
  if (threeMount) threeMount.dataset.geometryMode = state.geometryDisplayMode;
  appShell?.classList.toggle("surface-only-view", surfaceOnlyMode);
  appShell?.classList.toggle("thumb-holder-view", thumbMode);
  appShell?.classList.toggle("whole-hand-view", wholeHandMode);
  wholeHandModeButton?.classList.toggle("active", wholeHandMode);
  thumbHolderModeButton?.classList.toggle("active", state.geometryDisplayMode === "thumb_holder");
  surfaceOnlyModeButton?.classList.toggle("active", surfaceOnlyMode);
  settingsWholeHandButton?.classList.toggle("active", wholeHandMode);
  settingsThumbHolderButton?.classList.toggle("active", state.geometryDisplayMode === "thumb_holder");
  settingsSurfaceOnlyButton?.classList.toggle("active", surfaceOnlyMode);
  wholeHandModeButton?.setAttribute("aria-pressed", String(wholeHandMode));
  thumbHolderModeButton?.setAttribute("aria-pressed", String(thumbMode));
  surfaceOnlyModeButton?.setAttribute("aria-pressed", String(surfaceOnlyMode));
  settingsWholeHandButton?.setAttribute("aria-pressed", String(wholeHandMode));
  settingsThumbHolderButton?.setAttribute("aria-pressed", String(thumbMode));
  settingsSurfaceOnlyButton?.setAttribute("aria-pressed", String(surfaceOnlyMode));
  setText(
    "geometryModeStatus",
    wholeHandMode ? "Whole hand mode" : thumbMode ? "Thumb holder mode" : "Planar surface mode"
  );
  setText(
    "tactileSurfaceTitle",
    wholeHandMode ? "Robot Hand Tactile Surface" : thumbMode ? "Thumb Tactile Surface" : "Planar Tactile Surface"
  );
  setText(
    "surfaceProxyCaption",
    wholeHandMode
      ? "Modified thumb sensor response"
      : thumbMode
        ? "Sensor slot wavelength response"
        : "Footprint-aligned wavelength response"
  );
  if (fingerFocusControl) {
    fingerFocusControl.hidden = !wholeHandMode;
  }
  updateFingerScopeLabels();
  if (surfaceMesh?.material) {
    surfaceMesh.material.roughness = physicalMode ? 0.36 : 0.50;
    surfaceMesh.material.emissiveIntensity = physicalMode ? 0.08 : 0.015;
    surfaceMesh.material.needsUpdate = true;
  }
  if (bodyMesh?.material) {
    bodyMesh.material.opacity = physicalMode ? 0.46 : 0.62;
    bodyMesh.material.needsUpdate = true;
  }
  if (settingsResetCameraButton) {
    settingsResetCameraButton.textContent = wholeHandMode
      ? "Reset whole hand view"
      : thumbMode
        ? "Reset thumb view"
        : "Reset aligned surface view";
  }
  applyThumbSceneLayout();
  applyThumbCameraConfig();
  applySceneLighting(physicalMode);
  updateSurfaceRenderMode(state.surfaceRenderMode);
  // Geometry mode changes also change the stage layout. Resize after the new
  // CSS geometry has settled so the WebGL canvas never keeps the old column width.
  requestAnimationFrame(() => {
    resizeThree();
    requestAnimationFrame(() => {
      resizeThree();
      state.threeNeedsRefresh = true;
    });
  });
}

function refreshFullscreenSurfaceLayout() {
  requestAnimationFrame(() => {
    resizeThree();
    state.threeNeedsRefresh = true;
    requestAnimationFrame(() => {
      resizeThree();
      state.threeNeedsRefresh = true;
    });
  });
}

function applySurfaceFullscreenState(active) {
  state.surfaceFullscreenActive = Boolean(active);
  const forceReadoutTarget = state.surfaceFullscreenActive ? operatorSummaryCardNode : operatorCurrentHud;
  if (operatorForceReadout && forceReadoutTarget && operatorForceReadout.parentElement !== forceReadoutTarget) {
    forceReadoutTarget.appendChild(operatorForceReadout);
  }
  appShell?.classList.toggle("surface-fullscreen-active", state.surfaceFullscreenActive);
  document.documentElement.classList.toggle("surface-fullscreen-document", state.surfaceFullscreenActive);
  document.body.classList.toggle("surface-fullscreen-document", state.surfaceFullscreenActive);
  surfaceFullscreenButton?.classList.toggle("active", state.surfaceFullscreenActive);
  surfaceFullscreenButton?.setAttribute("aria-pressed", String(state.surfaceFullscreenActive));
  surfaceFullscreenButton?.setAttribute(
    "aria-label",
    state.surfaceFullscreenActive ? "Exit tactile surface fullscreen" : "Enter tactile surface fullscreen"
  );
  if (surfaceFullscreenButton) {
    surfaceFullscreenButton.dataset.tooltip = state.surfaceFullscreenActive ? "Exit fullscreen" : "Fullscreen";
  }
  surfaceFullscreenButton?.setAttribute(
    "title",
    state.surfaceFullscreenActive
      ? "Press Esc to exit full-screen tactile surface"
      : "Show only the interactive thumb tactile surface; press Esc to exit"
  );
  refreshFullscreenSurfaceLayout();
}

async function setSurfaceFullscreen(active) {
  const nextActive = Boolean(active);
  if (nextActive === state.surfaceFullscreenActive) return;
  if (nextActive) {
    updateDisplayMode("operator");
    setSettingsPanelOpen(false);
    setDemoMenuOpen(false);
    setSpectrumDrawerOpen(false);
    applySurfaceFullscreenState(true);
    if (document.documentElement.requestFullscreen && !document.fullscreenElement) {
      try {
        await document.documentElement.requestFullscreen();
        state.surfaceNativeFullscreenEntered = true;
      } catch {
        state.surfaceNativeFullscreenEntered = false;
      }
    }
    return;
  }

  applySurfaceFullscreenState(false);
  if (document.fullscreenElement && document.exitFullscreen) {
    try {
      await document.exitFullscreen();
    } catch {
      // CSS viewport mode has already been closed; native exit is best effort.
    }
  }
  state.surfaceNativeFullscreenEntered = false;
}

function projectedFingerHitRegions(rect) {
  const regions = [];
  const cameraUp = camera.up.clone().normalize();

  for (const [fingerId, group] of fingerSensorGroups.entries()) {
    if (!group.visible) continue;
    group.updateWorldMatrix(true, false);
    const center = group.getWorldPosition(new THREE.Vector3());
    const worldScale = group.getWorldScale(new THREE.Vector3());
    const worldRadius =
      4.7 * Math.max(Math.abs(worldScale.x), Math.abs(worldScale.z), 0.1);
    const projectedCenter = center.clone().project(camera);
    if (projectedCenter.z < -1 || projectedCenter.z > 1) continue;

    const projectedEdge = center
      .clone()
      .addScaledVector(cameraUp, worldRadius)
      .project(camera);
    const centerX = rect.left + ((projectedCenter.x + 1) * 0.5) * rect.width;
    const centerY = rect.top + ((1 - projectedCenter.y) * 0.5) * rect.height;
    const projectedRadius =
      Math.abs(projectedEdge.y - projectedCenter.y) * rect.height * 0.5;
    const hitRadius = Math.max(30, Math.min(72, projectedRadius + 12));
    regions.push({ fingerId, centerX, centerY, hitRadius });
  }

  if (threeMount) {
    threeMount.dataset.fingerHitRegions = JSON.stringify(
      regions.map(({ fingerId, centerX, centerY, hitRadius }) => ({
        fingerId,
        centerX: Math.round(centerX * 10) / 10,
        centerY: Math.round(centerY * 10) / 10,
        hitRadius: Math.round(hitRadius * 10) / 10,
      }))
    );
  }
  return regions;
}

function fingerIdFromProjectedRegion(event, rect, regions = null) {
  let nearestFinger = null;
  let nearestScore = Number.POSITIVE_INFINITY;
  for (const { fingerId, centerX, centerY, hitRadius } of (
    regions || projectedFingerHitRegions(rect)
  )) {
    const distance = Math.hypot(event.clientX - centerX, event.clientY - centerY);
    const score = distance / hitRadius;
    if (score <= 1 && score < nearestScore) {
      nearestFinger = fingerId;
      nearestScore = score;
    }
  }
  return nearestFinger;
}

function fingerIdFromPointerEvent(event) {
  if (
    state.geometryDisplayMode !== "whole_hand" ||
    !renderer?.domElement ||
    !camera
  ) {
    return null;
  }
  const rect = renderer.domElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const projectedRegions = projectedFingerHitRegions(rect);
  fingerPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  fingerPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  fingerRaycaster.setFromCamera(fingerPointer, camera);

  const targets = [];
  for (const group of fingerSensorGroups.values()) {
    if (!group.visible) continue;
    group.traverse((child) => {
      if (
        child.visible &&
        child.isMesh &&
        child.userData?.isFingerInteractionProxy
      ) {
        targets.push(child);
      }
    });
  }
  const hit = fingerRaycaster.intersectObjects(targets, false)[0];
  return (
    hit?.object?.userData?.fingerId ||
    fingerIdFromProjectedRegion(event, rect, projectedRegions) ||
    null
  );
}

function handleFingerPointerDown(event) {
  if (event.button !== 0 || !event.isPrimary) return;
  // WebView2 can intermittently render the CSS grab cursor as transparent.
  // Keep a stable native arrow while rotating and use a hand only for a real
  // fingertip hit target.
  renderer.domElement.style.cursor = "default";
  fingerPointerDown = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
  };
}

function handleFingerPointerUp(event) {
  if (!fingerPointerDown || fingerPointerDown.pointerId !== event.pointerId) return;
  const movement = Math.hypot(
    event.clientX - fingerPointerDown.x,
    event.clientY - fingerPointerDown.y
  );
  fingerPointerDown = null;
  renderer.domElement.style.cursor = "default";
  if (movement > FINGER_CLICK_MAX_MOVEMENT_PX) return;
  const fingerId = fingerIdFromPointerEvent(event);
  if (fingerId) setSelectedFinger(fingerId, { focusCamera: true });
}

function handleFingerPointerMove(event) {
  if (!renderer?.domElement || fingerPointerDown) return;
  renderer.domElement.style.cursor = fingerIdFromPointerEvent(event) ? "pointer" : "default";
}

function initThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color("#fafdff");

  camera = new THREE.PerspectiveCamera(43, 1, 0.1, 100);
  camera.position.set(6.0, 4.8, 7.8);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, THREE_MAX_DEVICE_PIXEL_RATIO));
  renderer.setSize(threeMount.clientWidth, threeMount.clientHeight);
  threeMount.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, -0.05, 0);
  controls.minDistance = 4.0;
  controls.maxDistance = 13.0;
  applyThumbCameraConfig();
  controls.addEventListener("start", () => {
    state.controlsInteracting = true;
    requestThreeAnimation();
  });
  controls.addEventListener("change", requestThreeAnimation);
  controls.addEventListener("end", () => {
    state.controlsInteracting = false;
    state.controlsSettlingUntil = performance.now() + 360;
    requestThreeAnimation();
  });
  renderer.domElement.addEventListener("pointerdown", handleFingerPointerDown);
  renderer.domElement.addEventListener("pointerup", handleFingerPointerUp);
  renderer.domElement.addEventListener("pointermove", handleFingerPointerMove);
  renderer.domElement.addEventListener("pointerleave", () => {
    fingerPointerDown = null;
    renderer.domElement.style.cursor = "default";
  });

  sceneAmbientLight = new THREE.AmbientLight("#ffffff", 2.2);
  scene.add(sceneAmbientLight);
  sceneKeyLight = new THREE.DirectionalLight("#ffffff", 2.8);
  sceneKeyLight.position.set(3, 6, 4);
  scene.add(sceneKeyLight);
  sceneFillLight = new THREE.DirectionalLight("#dcefff", 0.9);
  sceneFillLight.position.set(-3.5, 2, -4);
  scene.add(sceneFillLight);
  applySceneLighting(state.geometryDisplayMode !== "surface_only");

  sceneGrid = new THREE.GridHelper(11.5, 36, "#73a9c5", "#bdd6e4");
  sceneGrid.position.y = -1.04;
  sceneGrid.material.transparent = true;
  sceneGrid.material.opacity = 0.48;
  scene.add(sceneGrid);

  wholeHandRoot = new THREE.Group();
  wholeHandRoot.name = "robot_nano_hand_root";
  wholeHandRoot.visible = false;
  scene.add(wholeHandRoot);

  thumbModelRoot = new THREE.Group();
  thumbModelRoot.name = "thumb_model_root";
  scene.add(thumbModelRoot);

  grooveReferenceHelper = new THREE.Line(
    createGrooveReferenceLineGeometry(state.thumbSceneConfig?.sensor_slot_transform || {}),
    new THREE.LineBasicMaterial({
      color: "#ff8a3d",
      transparent: true,
      opacity: 0.62,
      depthTest: false,
    })
  );
  grooveReferenceHelper.name = "stl_diff_groove_lip_reference";
  grooveReferenceHelper.renderOrder = 8;
  scene.add(grooveReferenceHelper);

  sensorSurfaceGroup = new THREE.Group();
  sensorSurfaceGroup.name = "sensor_slot_surface_group";
  scene.add(sensorSurfaceGroup);

  slotOutlineHelper = new THREE.LineSegments(
    createOvalSurfaceGridGeometry(7.18, 5.16, 1, 128, 0),
    new THREE.LineBasicMaterial({
      color: "#37a8c7",
      transparent: true,
      opacity: 0.22,
      depthTest: true,
    })
  );
  slotOutlineHelper.name = "sensor_slot_surface_outline";
  slotOutlineHelper.position.set(0, 0.035, 0);
  sensorSurfaceGroup.add(slotOutlineHelper);

  const bodyGeometry = createOvalElastomerBodyGeometry(
    7,
    5,
    THREE_SLOT_BODY_THICKNESS,
    26,
    96,
    5
  );
  bodyBasePositions = bodyGeometry.attributes.position.array.slice();
  bodyMesh = new THREE.Mesh(
    bodyGeometry,
    new THREE.MeshStandardMaterial({
      vertexColors: true,
      color: "#9bc8dd",
      transparent: true,
      opacity: 0.46,
      roughness: 0.62,
      metalness: 0.02,
      side: THREE.DoubleSide,
    })
  );
  bodyMesh.renderOrder = 0;
  sensorSurfaceGroup.add(bodyMesh);

  bodyWireMesh = new THREE.Mesh(
    bodyGeometry,
    new THREE.MeshBasicMaterial({
      color: "#4f8998",
      transparent: true,
      opacity: 0.24,
      wireframe: true,
      depthTest: true,
    })
  );
  bodyWireMesh.renderOrder = 1;
  // Keep the engineering triangulation out of the normal tactile view. Through
  // the translucent elastomer it reads as radial spokes around a pressed patch.
  bodyWireMesh.visible = false;
  sensorSurfaceGroup.add(bodyWireMesh);

  surfaceGeometry = createOvalSurfaceGeometry(7, 5, 32, 96);
  surfaceBasePositions = surfaceGeometry.attributes.position.array.slice();
  const colors = [];
  const vertexCount = surfaceGeometry.attributes.position.count;
  for (let i = 0; i < vertexCount; i += 1) {
    colors.push(0.48, 0.78, 0.90);
  }
  surfaceGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  surfaceMesh = new THREE.Mesh(
    surfaceGeometry,
    new THREE.MeshStandardMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      roughness: 0.36,
      metalness: 0.03,
      emissive: "#f7fcff",
      emissiveIntensity: 0.08,
      side: THREE.DoubleSide,
    })
  );
  surfaceMesh.renderOrder = 2;
  sensorSurfaceGroup.add(surfaceMesh);

  const gridGeometry = createOvalSurfaceGridGeometry(7, 5, 9, 128, 0);
  surfaceGridBasePositions = gridGeometry.attributes.position.array.slice();
  wireMesh = new THREE.LineSegments(
    gridGeometry,
    new THREE.LineBasicMaterial({
      color: "#e4f1f3",
      transparent: true,
      opacity: 0.10,
      depthTest: true,
    })
  );
  wireMesh.renderOrder = 4;
  sensorSurfaceGroup.add(wireMesh);

  const bottomGridGeometry = createOvalSurfaceGridGeometry(7, 5, 9, 128, 0);
  bottomGridBasePositions = bottomGridGeometry.attributes.position.array.slice();
  bottomGridMesh = new THREE.LineSegments(
    bottomGridGeometry,
    new THREE.LineBasicMaterial({
      color: "#6cb8c4",
      transparent: true,
      opacity: 0.16,
      depthTest: true,
    })
  );
  bottomGridMesh.renderOrder = 1;
  sensorSurfaceGroup.add(bottomGridMesh);

  setupFingerSensorGroups();
  setupThumbHolderModel();
  setupWholeHandModel();
  applyThumbSceneLayout();
  resizeThree();
  requestThreeAnimation();
}

function resizeThree() {
  if (!renderer || !camera) return;
  const width = Math.max(1, threeMount.clientWidth);
  const height = Math.max(1, threeMount.clientHeight);
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  requestThreeAnimation();
}

function updateThree(record, arrayFrame = null) {
  const metrics = arrayFrame?.surface_metrics || {};
  const peak = Number(metrics.surface_peak);
  const rawPeak = Number.isFinite(peak) ? Math.max(0, Math.min(1, peak)) : attenuation(record);
  const atomicNeutral =
    arrayFrame?.presentation_status === "invalid_input_neutral" ||
    arrayFrame?.mode === "invalid_input_neutral";
  const visualPeak = atomicNeutral ? 0 : visualSurfacePeakForFrame(rawPeak, record, arrayFrame);
  state.currentSurfaceGrid = Array.isArray(arrayFrame?.surface_grid) ? arrayFrame.surface_grid : null;
  state.currentSurfaceMetrics = metrics;
  const cx = Number(metrics.surface_centroid_x);
  const cy = Number(metrics.surface_centroid_y);
  const spread = Number(metrics.surface_spread);
  const activeArea = Number(metrics.surface_area_active);
  state.targetSurfaceVisualPeak = visualPeak;
  state.targetSurfaceCentroidX = Number.isFinite(cx) ? Math.max(-1.25, Math.min(1.25, cx)) : 0;
  state.targetSurfaceCentroidY = Number.isFinite(cy) ? Math.max(-1.25, Math.min(1.25, cy)) : 0;
  state.targetSurfaceSpread = Number.isFinite(spread) ? Math.max(0.18, Math.min(1.20, spread)) : 0.34;
  state.targetSurfaceActiveArea = Number.isFinite(activeArea) ? Math.max(0, Math.min(1, activeArea)) : 0;
  state.targetAttenuation = Number.isFinite(peak) ? visualPeak : attenuation(record);
  state.targetDeformation = Number.isFinite(peak)
    ? visualDeformationFromSurfacePeak(visualPeak, rawPeak)
    : visualDeformationStrength(record);
  if (atomicNeutral) {
    state.targetAttenuation = 0;
    state.targetDeformation = 0;
    state.smoothAttenuation = 0;
    state.smoothDeformation = 0;
    state.smoothSurfaceVisualPeak = 0;
    state.smoothSurfaceCentroidX = 0;
    state.smoothSurfaceCentroidY = 0;
    state.smoothSurfaceSpread = state.targetSurfaceSpread;
    state.smoothSurfaceActiveArea = 0;
    if (surfaceGeometry) {
      applyThreeGeometry(0, 0, { recomputeNormals: true });
    }
  }
  state.threeNeedsRefresh = true;
  requestThreeAnimation();
}

function surfaceValueAtScene(x, z) {
  const grid = state.currentSurfaceGrid;
  if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) return null;
  const rows = grid.length;
  const cols = grid[0].length;
  const arrayCoord = surfaceSceneToArrayCoord(x, z);
  const arrayX = arrayCoord.x;
  const arrayY = arrayCoord.y;
  const gx = ((arrayX + 1.25) / 2.5) * (cols - 1);
  const gy = ((1.25 - arrayY) / 2.5) * (rows - 1);
  const x0 = Math.max(0, Math.min(cols - 1, Math.floor(gx)));
  const y0 = Math.max(0, Math.min(rows - 1, Math.floor(gy)));
  const x1 = Math.max(0, Math.min(cols - 1, x0 + 1));
  const y1 = Math.max(0, Math.min(rows - 1, y0 + 1));
  const tx = gx - x0;
  const ty = gy - y0;
  const v00 = Number(grid[y0][x0]) || 0;
  const v10 = Number(grid[y0][x1]) || 0;
  const v01 = Number(grid[y1][x0]) || 0;
  const v11 = Number(grid[y1][x1]) || 0;
  const top = v00 * (1 - tx) + v10 * tx;
  const bottom = v01 * (1 - tx) + v11 * tx;
  const rawValue = Math.max(0, Math.min(1, top * (1 - ty) + bottom * ty));
  const rawPeak = Math.max(1e-6, Number(state.currentSurfaceMetrics?.surface_peak) || rawValue);
  const normalized = rawValue / rawPeak;
  if (normalized < 0.012) return 0;
  const visualPeak = Math.max(rawPeak, currentSurfaceVisualPeak());
  return Math.max(0, Math.min(1, visualPeak * Math.pow(Math.min(1, normalized), SURFACE_GRID_VISUAL_GAMMA)));
}

function centroidContactPatchValueAtScene(x, z) {
  const metrics = state.currentSurfaceMetrics || {};
  const cx = Number(state.smoothSurfaceCentroidX);
  const cy = Number(state.smoothSurfaceCentroidY);
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  const peak = Math.max(0, currentSurfaceVisualPeak(), Number(metrics.surface_peak) || 0);
  if (peak < 0.035) return null;

  const sceneCenter = arrayCoordToSurfaceScene(cx, cy);
  const sceneCx = sceneCenter.x;
  const sceneCz = sceneCenter.z;
  const spread = Math.max(0.20, Math.min(1.12, Number(state.smoothSurfaceSpread) || 0.34));
  const activeArea = Math.max(0, Math.min(1, Number(state.smoothSurfaceActiveArea) || 0));
  const broadness = Math.sqrt(activeArea);
  const sigmaX = 0.40 + 0.62 * spread + 0.58 * broadness;
  const sigmaZ = 0.34 + 0.52 * spread + 0.48 * broadness;
  const dx = (x - sceneCx) / Math.max(0.12, sigmaX);
  const dz = (z - sceneCz) / Math.max(0.12, sigmaZ);
  const localPatch = peak * Math.exp(-0.5 * (dx * dx + dz * dz));
  const shoulderDx = (x - sceneCx) / Math.max(0.12, sigmaX * 1.85);
  const shoulderDz = (z - sceneCz) / Math.max(0.12, sigmaZ * 1.85);
  const softShoulder = peak * 0.18 * Math.exp(-0.5 * (shoulderDx * shoulderDx + shoulderDz * shoulderDz));
  return Math.max(0, Math.min(1, localPatch + softShoulder));
}

function applyThreeGeometry(raw, deformation, { recomputeNormals = false } = {}) {
  if (!surfaceGeometry) return;
  const amp = raw;
  const maxTopDepth = 0.90;
  const physicalMode = state.surfaceRenderMode !== "response_terrain";
  const positions = surfaceGeometry.attributes.position;
  const colors = surfaceGeometry.attributes.color;
  const positionArray = positions.array;
  const colorArray = colors.array;
  const tempColor = new THREE.Color();
  const lightCenterColor = new THREE.Color("#d9c56f");

  if (wireMesh?.material) {
    const inspectionMode = state.geometryDisplayMode === "surface_only";
    wireMesh.visible = true;
    wireMesh.material.color.set(inspectionMode ? "#496f7a" : "#e4f1f3");
    wireMesh.material.opacity = inspectionMode
      ? raw >= 0.70
        ? 0.16
        : raw >= 0.30
          ? 0.12
          : 0.07
      : raw >= 0.70
        ? 0.065
        : raw >= 0.30
          ? 0.050
          : 0.035;
  }

  for (let i = 0; i < positions.count; i += 1) {
    const baseIndex = i * 3;
    const x = surfaceBasePositions[baseIndex];
    const z = surfaceBasePositions[baseIndex + 2];
    const shape = topSurfaceShape(x, z, deformation, maxTopDepth);
    const surfaceY = physicalMode ? shape.localDepression : -shape.localDepression * 0.82;
    positionArray[baseIndex + 1] = surfaceY;
    const heat = Math.max(0, Math.min(1, visualColorStrength(amp, shape.centerBasin)));
    tempColor.copy(colorForAttenuation(heat));
    const lightResponse =
      smoothStep(0.055, 0.105, amp) *
      (1 - smoothStep(0.19, 0.34, amp));
    const lightCenterWarmth =
      0.52 *
      Math.pow(shape.centerBasin, 1.45) *
      lightResponse;
    if (lightCenterWarmth > 0.001) {
      tempColor.lerp(lightCenterColor, lightCenterWarmth);
    }
    colorArray[baseIndex] = tempColor.r;
    colorArray[baseIndex + 1] = tempColor.g;
    colorArray[baseIndex + 2] = tempColor.b;
  }
  positions.needsUpdate = true;
  colors.needsUpdate = true;
  if (recomputeNormals) surfaceGeometry.computeVertexNormals();

  const gridGeometry = wireMesh?.geometry;
  if (gridGeometry && surfaceGridBasePositions) {
    const inspectionMode = state.geometryDisplayMode === "surface_only";
    const surfaceLineOffset = inspectionMode ? -0.018 : 0.032;
    const gridPositions = gridGeometry.attributes.position;
    const gridPositionArray = gridPositions.array;
    for (let i = 0; i < gridPositions.count; i += 1) {
      const baseIndex = i * 3;
      const x = surfaceGridBasePositions[baseIndex];
      const z = surfaceGridBasePositions[baseIndex + 2];
      const shape = topSurfaceShape(x, z, deformation, maxTopDepth);
      const gridY = physicalMode
        ? shape.localDepression + surfaceLineOffset
        : -shape.localDepression * 0.82 + surfaceLineOffset;
      gridPositionArray[baseIndex + 1] = gridY;
    }
    gridPositions.needsUpdate = true;
  }

  const lowerGridGeometry = bottomGridMesh?.geometry;
  if (lowerGridGeometry && bottomGridBasePositions) {
    const lowerPositions = lowerGridGeometry.attributes.position;
    const lowerPositionArray = lowerPositions.array;
    for (let i = 0; i < lowerPositions.count; i += 1) {
      const baseIndex = i * 3;
      const x = bottomGridBasePositions[baseIndex];
      const z = bottomGridBasePositions[baseIndex + 2];
      const shape = bodyElasticShape(
        x,
        THREE_SLOT_BODY_THICKNESS,
        z,
        deformation,
        maxTopDepth
      );
      const bottomDeflection =
        shape.localY - THREE_SLOT_BODY_THICKNESS;
      const lowerY = physicalMode
        ? THREE_SLOT_BODY_THICKNESS + bottomDeflection * 0.20 - 0.010
        : THREE_SLOT_BODY_THICKNESS - bottomDeflection * 0.50;
      lowerPositionArray[baseIndex + 1] = lowerY;
    }
    lowerPositions.needsUpdate = true;
    if (recomputeNormals) lowerGridGeometry.computeBoundingSphere();
    if (bottomGridMesh.material) {
      bottomGridMesh.material.opacity = raw >= 0.70 ? 0.17 : raw >= 0.30 ? 0.145 : 0.105;
    }
  }

  const bodyGeometry = bodyMesh?.geometry;
  if (bodyGeometry && bodyBasePositions) {
    const bodyPositions = bodyGeometry.attributes.position;
    const bodyColors = bodyGeometry.attributes.color;
    const bodyPositionArray = bodyPositions.array;
    const bodyColorArray = bodyColors?.array;
    for (let i = 0; i < bodyPositions.count; i += 1) {
      const baseIndex = i * 3;
      const x = bodyBasePositions[baseIndex];
      const y = bodyBasePositions[baseIndex + 1];
      const z = bodyBasePositions[baseIndex + 2];
      const shape = bodyElasticShape(x, y, z, deformation, maxTopDepth);
      const bodyY = physicalMode ? shape.localY : y + (y - shape.localY) * 0.48;
      bodyPositionArray[baseIndex + 1] = bodyY;
      if (bodyColorArray) {
        const heat = Math.max(0, Math.min(1, 0.60 * visualColorStrength(amp, shape.centerBasin) + 0.08));
        tempColor.copy(colorForAttenuation(heat));
        bodyColorArray[baseIndex] = tempColor.r;
        bodyColorArray[baseIndex + 1] = tempColor.g;
        bodyColorArray[baseIndex + 2] = tempColor.b;
      }
    }
    bodyPositions.needsUpdate = true;
    if (bodyColors) bodyColors.needsUpdate = true;
    if (recomputeNormals) bodyGeometry.computeVertexNormals();
  }
}

function requestThreeAnimation() {
  if (!state.pageVisible || !renderer || state.animationFrameId !== null) return;
  state.animationFrameId = requestAnimationFrame(animate);
}

function animate(timestamp = 0) {
  state.animationFrameId = null;
  if (!state.pageVisible || !renderer || !scene || !camera) return;
  if (windowResizeActive) {
    state.lastThreeFrameMs = timestamp;
    return;
  }
  const previous = state.lastThreeFrameMs || timestamp;
  const deltaSeconds = Math.max(0.001, Math.min(0.08, (timestamp - previous) / 1000 || 1 / 60));
  state.lastThreeFrameMs = timestamp;

  const playbackDeltaSeconds = state.demoModeActive
    ? deltaSeconds * normalizedDemoPlaybackRate(state.demoPlaybackRate)
    : deltaSeconds;

  state.chartDeltaAccumulator += playbackDeltaSeconds;
  if (state.paused) {
    if (state.chartsNeedRefresh) {
      drawVisibleCharts();
      state.chartsNeedRefresh = false;
    }
    state.chartDeltaAccumulator = 0;
    state.lastChartUpdateMs = timestamp;
  } else if (
    state.chartsNeedRefresh &&
    (!state.lastChartUpdateMs || timestamp - state.lastChartUpdateMs >= CHART_UPDATE_INTERVAL_MS)
  ) {
    const chartDeltaSeconds = Math.max(0.001, state.chartDeltaAccumulator);
    state.chartDeltaAccumulator = 0;
    state.lastChartUpdateMs = timestamp;
    updateChartSmoothing(chartDeltaSeconds);
  } else if (!state.chartsNeedRefresh) {
    state.chartDeltaAccumulator = 0;
  }

  state.geometryDeltaAccumulator = state.paused
    ? 0
    : state.geometryDeltaAccumulator + playbackDeltaSeconds;
  if (surfaceGeometry && !state.paused) {
    const attenuationDelta = Math.abs(state.targetAttenuation - state.smoothAttenuation);
    const deformationDelta = Math.abs(state.targetDeformation - state.smoothDeformation);
    const surfacePeakDelta = Math.abs(state.targetSurfaceVisualPeak - state.smoothSurfaceVisualPeak);
    const centroidDelta =
      Math.abs(state.targetSurfaceCentroidX - state.smoothSurfaceCentroidX) +
      Math.abs(state.targetSurfaceCentroidY - state.smoothSurfaceCentroidY);
    const spreadDelta = Math.abs(state.targetSurfaceSpread - state.smoothSurfaceSpread);
    const activeAreaDelta = Math.abs(state.targetSurfaceActiveArea - state.smoothSurfaceActiveArea);
    const geometryNeedsUpdate =
      state.threeNeedsRefresh ||
      attenuationDelta > THREE_SETTLE_EPSILON ||
      deformationDelta > THREE_SETTLE_EPSILON ||
      surfacePeakDelta > THREE_SETTLE_EPSILON ||
      centroidDelta > THREE_SETTLE_EPSILON ||
      spreadDelta > THREE_SETTLE_EPSILON ||
      activeAreaDelta > THREE_SETTLE_EPSILON;
    if (
      geometryNeedsUpdate &&
      (!state.lastGeometryUpdateMs || timestamp - state.lastGeometryUpdateMs >= THREE_GEOMETRY_UPDATE_INTERVAL_MS)
    ) {
      const geometryDeltaSeconds = Math.max(0.001, state.geometryDeltaAccumulator);
      state.geometryDeltaAccumulator = 0;
      state.lastGeometryUpdateMs = timestamp;
      const attenuationEasing = state.targetAttenuation < state.smoothAttenuation
        ? THREE_ATTENUATION_RELEASE_EASING
        : THREE_ATTENUATION_EASING;
      const deformationEasing = state.targetDeformation < state.smoothDeformation
        ? THREE_DEFORMATION_RELEASE_EASING
        : THREE_DEFORMATION_EASING;
      const surfacePeakEasing = state.targetSurfaceVisualPeak < state.smoothSurfaceVisualPeak
        ? THREE_SURFACE_RELEASE_EASING
        : THREE_DEFORMATION_EASING;
      state.smoothAttenuation = dampToward(state.smoothAttenuation, state.targetAttenuation, attenuationEasing, geometryDeltaSeconds);
      state.smoothDeformation = dampToward(state.smoothDeformation, state.targetDeformation, deformationEasing, geometryDeltaSeconds);
      state.smoothSurfaceVisualPeak = dampToward(
        state.smoothSurfaceVisualPeak,
        state.targetSurfaceVisualPeak,
        surfacePeakEasing,
        geometryDeltaSeconds
      );
      state.smoothSurfaceCentroidX = dampToward(
        state.smoothSurfaceCentroidX,
        state.targetSurfaceCentroidX,
        THREE_SPATIAL_EASING,
        geometryDeltaSeconds
      );
      state.smoothSurfaceCentroidY = dampToward(
        state.smoothSurfaceCentroidY,
        state.targetSurfaceCentroidY,
        THREE_SPATIAL_EASING,
        geometryDeltaSeconds
      );
      state.smoothSurfaceSpread = dampToward(state.smoothSurfaceSpread, state.targetSurfaceSpread, THREE_SPATIAL_EASING, geometryDeltaSeconds);
      state.smoothSurfaceActiveArea = dampToward(
        state.smoothSurfaceActiveArea,
        state.targetSurfaceActiveArea,
        THREE_SPATIAL_EASING,
        geometryDeltaSeconds
      );
      if (Math.abs(state.targetAttenuation - state.smoothAttenuation) < THREE_SETTLE_EPSILON) {
        state.smoothAttenuation = state.targetAttenuation;
      }
      if (Math.abs(state.targetDeformation - state.smoothDeformation) < THREE_SETTLE_EPSILON) {
        state.smoothDeformation = state.targetDeformation;
      }
      if (Math.abs(state.targetSurfaceVisualPeak - state.smoothSurfaceVisualPeak) < THREE_SETTLE_EPSILON) {
        state.smoothSurfaceVisualPeak = state.targetSurfaceVisualPeak;
      }
      if (Math.abs(state.targetSurfaceCentroidX - state.smoothSurfaceCentroidX) < THREE_SETTLE_EPSILON) {
        state.smoothSurfaceCentroidX = state.targetSurfaceCentroidX;
      }
      if (Math.abs(state.targetSurfaceCentroidY - state.smoothSurfaceCentroidY) < THREE_SETTLE_EPSILON) {
        state.smoothSurfaceCentroidY = state.targetSurfaceCentroidY;
      }
      if (Math.abs(state.targetSurfaceSpread - state.smoothSurfaceSpread) < THREE_SETTLE_EPSILON) {
        state.smoothSurfaceSpread = state.targetSurfaceSpread;
      }
      if (Math.abs(state.targetSurfaceActiveArea - state.smoothSurfaceActiveArea) < THREE_SETTLE_EPSILON) {
        state.smoothSurfaceActiveArea = state.targetSurfaceActiveArea;
      }
      const recomputeNormals =
        !state.lastGeometryNormalUpdateMs ||
        timestamp - state.lastGeometryNormalUpdateMs >= THREE_NORMAL_UPDATE_INTERVAL_MS;
      applyThreeGeometry(state.smoothAttenuation, state.smoothDeformation, { recomputeNormals });
      if (recomputeNormals) state.lastGeometryNormalUpdateMs = timestamp;
      state.threeNeedsRefresh =
        Math.abs(state.targetAttenuation - state.smoothAttenuation) > THREE_SETTLE_EPSILON ||
        Math.abs(state.targetDeformation - state.smoothDeformation) > THREE_SETTLE_EPSILON ||
        Math.abs(state.targetSurfaceVisualPeak - state.smoothSurfaceVisualPeak) > THREE_SETTLE_EPSILON ||
        Math.abs(state.targetSurfaceCentroidX - state.smoothSurfaceCentroidX) > THREE_SETTLE_EPSILON ||
        Math.abs(state.targetSurfaceCentroidY - state.smoothSurfaceCentroidY) > THREE_SETTLE_EPSILON ||
        Math.abs(state.targetSurfaceSpread - state.smoothSurfaceSpread) > THREE_SETTLE_EPSILON ||
        Math.abs(state.targetSurfaceActiveArea - state.smoothSurfaceActiveArea) > THREE_SETTLE_EPSILON;
    } else if (!geometryNeedsUpdate) {
      state.geometryDeltaAccumulator = 0;
    }
  }

  const cameraTransitionUpdated = updateCameraTransition(timestamp);
  if (!cameraTransitionUpdated) controls?.update();
  renderer?.render(scene, camera);
  const controlsSettling = state.controlsInteracting || timestamp < state.controlsSettlingUntil;
  if (
    cameraTransition?.active ||
    controlsSettling ||
    state.chartsNeedRefresh ||
    state.threeNeedsRefresh
  ) {
    requestThreeAnimation();
  }
}

function sortedChannels(grid) {
  const byId = new Map((grid || []).map((item) => [item.channel_id, item]));
  const ordered = ARRAY_DISPLAY_ORDER.map((id) => byId.get(id)).filter(Boolean);
  const extra = (grid || []).filter((item) => !ARRAY_DISPLAY_ORDER.includes(item.channel_id));
  return ordered.concat(extra);
}

function updateChannelSelectorLabels(channels) {
  if (!channelSelect) return;
  const byId = new Map((channels || []).map((item) => [item.channel_id, item]));
  Array.from(channelSelect.options).forEach((option) => {
    const channelId = option.value;
    const channel = byId.get(channelId);
    const target = Number(channel?.target_wavelength_nm);
    const measured = Number(channel?.measured_wavelength_nm);
    if (Number.isFinite(target)) {
      option.textContent = Number.isFinite(measured)
        ? `${channelId} | target ${target.toFixed(1)} nm | measured ${measured.toFixed(1)} nm`
        : `${channelId} | target ${target.toFixed(1)} nm`;
    } else {
      option.textContent = channelId;
    }
  });
}

const DIAGNOSTIC_WORKSPACES = new Set(["signal", "recording", "measurement", "surface", "demo", "acquisition", "reference", "geometry"]);
const DIAGNOSTIC_DEFAULT_CARD = {
  signal: ".diagnostic-channel-card",
  recording: ".diagnostic-capture-card",
  measurement: ".diagnostic-measurement-card",
  surface: ".diagnostic-metrics-card",
  demo: ".demo-module",
  acquisition: ".diagnostic-frame-card",
  reference: ".diagnostic-reference-card",
  geometry: ".diagnostic-alignment-card",
};

function diagnosticsPanelWidthBounds() {
  const dashboardWidth = dashboard?.getBoundingClientRect().width || window.innerWidth;
  const leftPanel = dashboard?.querySelector(".left-panel");
  const leftWidth = leftPanel?.getBoundingClientRect().width || 180;
  const dashboardStyle = dashboard ? window.getComputedStyle(dashboard) : null;
  const gap = Number.parseFloat(dashboardStyle?.columnGap || dashboardStyle?.gap || "8") || 8;
  const centerMinimum = dashboardWidth <= DIAGNOSTICS_COMPACT_BREAKPOINT_PX
    ? DIAGNOSTICS_CENTER_COMPACT_MIN_WIDTH_PX
    : DIAGNOSTICS_CENTER_MIN_WIDTH_PX;
  const available = Math.max(
    0,
    Math.floor(dashboardWidth - leftWidth - centerMinimum - (gap * 2))
  );
  const min = Math.min(DIAGNOSTICS_PANEL_MIN_WIDTH_PX, available);
  const max = Math.max(min, Math.min(DIAGNOSTICS_PANEL_MAX_WIDTH_PX, available));
  return { min, max };
}

function storedDiagnosticsPanelWidth() {
  try {
    const storedValue = window.localStorage.getItem(DIAGNOSTICS_PANEL_WIDTH_STORAGE_KEY);
    if (storedValue === null || storedValue.trim() === "") {
      return DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX;
    }
    const value = Number(storedValue);
    return Number.isFinite(value) ? value : DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX;
  } catch (_error) {
    return DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX;
  }
}

function setDiagnosticsPanelWidth(width, { persist = true } = {}) {
  if (!dashboard || !diagnosticsPanelResizer) return DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX;
  const { min, max } = diagnosticsPanelWidthBounds();
  const requested = Number(width);
  const nextWidth = Math.round(Math.max(min, Math.min(max, Number.isFinite(requested) ? requested : DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX)));
  dashboard.style.setProperty("--diagnostics-right-width", `${nextWidth}px`);
  diagnosticsPanelResizer.setAttribute("aria-valuemin", String(min));
  diagnosticsPanelResizer.setAttribute("aria-valuemax", String(max));
  diagnosticsPanelResizer.setAttribute("aria-valuenow", String(nextWidth));
  diagnosticsPanelResizer.setAttribute("aria-valuetext", `${nextWidth} pixels`);
  if (persist) {
    try {
      window.localStorage.setItem(DIAGNOSTICS_PANEL_WIDTH_STORAGE_KEY, String(nextWidth));
    } catch (_error) {
      // A blocked localStorage must not disable resizing for this session.
    }
  }
  cancelAnimationFrame(resizeToken);
  resizeToken = requestAnimationFrame(() => {
    resizeThree();
    state.chartsNeedRefresh = true;
    state.threeNeedsRefresh = true;
  });
  return nextWidth;
}

function initializeDiagnosticsPanelResize() {
  if (!dashboard || !diagnosticsPanelResizer) return;
  setDiagnosticsPanelWidth(storedDiagnosticsPanelWidth(), { persist: false });

  let dragState = null;
  diagnosticsPanelResizer.addEventListener("pointerdown", (event) => {
    if (state.displayMode !== "diagnostics" || event.button !== 0) return;
    event.preventDefault();
    dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: Number(diagnosticsPanelResizer.getAttribute("aria-valuenow")) || DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX,
    };
    diagnosticsPanelResizer.setPointerCapture(event.pointerId);
    diagnosticsPanelResizer.classList.add("is-resizing");
    document.body.classList.add("diagnostics-panel-resizing");
  });

  diagnosticsPanelResizer.addEventListener("pointermove", (event) => {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    setDiagnosticsPanelWidth(dragState.startWidth + dragState.startX - event.clientX, { persist: false });
  });

  const finishResize = (event) => {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    if (diagnosticsPanelResizer.hasPointerCapture(event.pointerId)) {
      diagnosticsPanelResizer.releasePointerCapture(event.pointerId);
    }
    dragState = null;
    diagnosticsPanelResizer.classList.remove("is-resizing");
    document.body.classList.remove("diagnostics-panel-resizing");
    setDiagnosticsPanelWidth(Number(diagnosticsPanelResizer.getAttribute("aria-valuenow")));
  };
  diagnosticsPanelResizer.addEventListener("pointerup", finishResize);
  diagnosticsPanelResizer.addEventListener("pointercancel", finishResize);

  diagnosticsPanelResizer.addEventListener("keydown", (event) => {
    const currentWidth = Number(diagnosticsPanelResizer.getAttribute("aria-valuenow")) || DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX;
    const { min, max } = diagnosticsPanelWidthBounds();
    let nextWidth = null;
    if (event.key === "ArrowLeft") nextWidth = currentWidth + 24;
    else if (event.key === "ArrowRight") nextWidth = currentWidth - 24;
    else if (event.key === "Home") nextWidth = min;
    else if (event.key === "End") nextWidth = max;
    if (nextWidth === null) return;
    event.preventDefault();
    setDiagnosticsPanelWidth(nextWidth);
  });

  diagnosticsPanelResizer.addEventListener("dblclick", () => {
    setDiagnosticsPanelWidth(DIAGNOSTICS_PANEL_DEFAULT_WIDTH_PX);
  });
}

function updateDiagnosticWorkspace(workspace) {
  const nextWorkspace = DIAGNOSTIC_WORKSPACES.has(workspace) ? workspace : "signal";
  state.diagnosticTab = nextWorkspace;
  diagnosticTabButtons.forEach((button) => {
    const active = button.dataset.diagnosticTab === nextWorkspace;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    if (active) {
      requestAnimationFrame(() => {
        button.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
      });
    }
  });
  diagnosticGroupedCards.forEach((card) => {
    const visible = card.dataset.diagnosticGroup === nextWorkspace;
    card.classList.toggle("diagnostic-group-hidden", !visible);
    card.setAttribute("aria-hidden", String(!visible));
  });
  const defaultCard = document.querySelector(DIAGNOSTIC_DEFAULT_CARD[nextWorkspace]);
  if (defaultCard instanceof HTMLDetailsElement) {
    defaultCard.open = true;
  }
  const cardStack = document.querySelector(".diagnostic-card-stack");
  if (state.displayMode === "diagnostics" && cardStack) {
    cardStack.classList.toggle("diagnostic-stack-empty", nextWorkspace === "demo");
    cardStack.scrollTop = 0;
  }
  if (nextWorkspace === "geometry") {
    const geometryControls = document.querySelector(".geometry-controls-scroll");
    if (geometryControls) geometryControls.scrollTop = 0;
  }
  if (nextWorkspace === "measurement") {
    requestAnimationFrame(() => {
      drawMeasurementComparison(state.measurementResult?.trace || []);
    });
  }
}

function updateDisplayMode(mode) {
  state.displayMode = mode === "diagnostics" ? "diagnostics" : "operator";
  if (state.displayMode === "operator" && state.surfaceRenderMode === "response_terrain") {
    updateSurfaceRenderMode("physical_proxy");
  }
  appShell?.classList.toggle("diagnostics-mode", state.displayMode === "diagnostics");
  appShell?.classList.toggle("operator-mode", state.displayMode !== "diagnostics");
  operatorModeButton?.classList.toggle("active", state.displayMode === "operator");
  diagnosticsModeButton?.classList.toggle("active", state.displayMode === "diagnostics");
  if (scene) {
    scene.background = new THREE.Color("#fafdff");
  }
  if (sceneGrid) {
    sceneGrid.visible = true;
    sceneGrid.material.opacity = state.displayMode === "operator" ? 0.48 : 0.30;
  }
  if (state.displayMode === "diagnostics") {
    spectrumDrawer?.classList.remove("open");
    spectrumDrawer?.setAttribute("aria-hidden", "true");
    demoModule?.removeAttribute("open");
    demoMenuButton?.classList.remove("menu-open");
    demoMenuButton?.setAttribute("aria-expanded", "false");
    setSettingsPanelOpen(false);
    setDiagnosticsPanelWidth(storedDiagnosticsPanelWidth(), { persist: false });
    updateDiagnosticWorkspace(state.diagnosticTab);
  }
  syncCommandAvailability();
  requestAnimationFrame(() => {
    resizeThree();
    state.chartsNeedRefresh = true;
    if (state.displayMode === "diagnostics") {
      drawHeatmap(state.currentArrayFrame);
    }
    requestAnimationFrame(() => {
      resizeThree();
      state.threeNeedsRefresh = true;
    });
  });
}

function updateSurfaceRenderMode(mode) {
  state.surfaceRenderMode = mode === "response_terrain" ? "response_terrain" : "physical_proxy";
  physicalProxyModeButton?.classList.toggle("active", state.surfaceRenderMode === "physical_proxy");
  responseTerrainModeButton?.classList.toggle("active", state.surfaceRenderMode === "response_terrain");
  if (state.surfaceRenderMode === "response_terrain") {
    setText("surfaceProxyCaption", "Bragg wavelength-shift terrain");
  } else {
    setText(
      "surfaceProxyCaption",
      state.geometryDisplayMode === "whole_hand"
        ? "Modified thumb sensor response"
        : state.geometryDisplayMode === "thumb_holder"
          ? "Sensor slot response surface"
          : "Footprint-aligned wavelength response"
    );
  }
  state.threeNeedsRefresh = true;
}

function couplingViewLabel(value) {
  if (value === "independent_ideal_response") return "Ideal decoupled simulation, not current data";
  if (value === "coupling_compensated_response") return "Coupling-compensated response, disabled until calibration";
  return "Raw coupled Bragg wavelength-shift surface";
}

function updateCouplingView(mode) {
  state.couplingView = mode === "independent_ideal_response" ? "independent_ideal_response" : "raw_coupled_response";
  rawCoupledViewButton?.classList.toggle("active", state.couplingView === "raw_coupled_response");
  idealIndependentViewButton?.classList.toggle("active", state.couplingView === "independent_ideal_response");
  couplingCompensatedViewButton?.classList.remove("active");
  setText("couplingDiagnosticNote", `${couplingViewLabel(state.couplingView)}. Calibration matrix missing for true local-response inversion.`);
  setText(
    "couplingViewBadge",
    state.couplingView === "independent_ideal_response"
      ? "Ideal simulation only, not measured"
      : "Coupled wavelength response"
  );
  if (state.arrayDemoActive && state.arrayDemoScenario) {
    state.arrayDemoStep = 0;
    fetchFrame();
  }
}

function updateArrayDiagnostics(arrayFrame, record, measurementAvailable = true) {
  const channels = sortedChannels(arrayFrame?.channels || []);
  const realActive = channels.filter((item) => item.enabled && item.valid && item.qa_status !== "simulated").length;
  const simulated = channels.filter((item) => item.valid && item.qa_status === "simulated").length;
  const disabled = channels.filter((item) => !item.enabled).length;
  const warnings = channels.filter((item) => {
    const status = String(item.qa_status || "");
    return status && !["ok", "ok_with_manual_wavelength", "simulated", "no_data"].includes(status);
  }).length;
  const mode = String(arrayFrame?.mode || "p22_fallback");
  const recordedDemo = isRecordedDemoSpectrum(record, arrayFrame);
  const currentRuntimeMode = isModelPositionLevelMode(mode);
  const globalSpectrumMode = mode.startsWith("global_spectrum_");
  const fallbackLike = mode === "p22_fallback" || mode === "no_valid_channel" || mode === "";
  setText("realActiveChannels", globalSpectrumMode ? String(realActive) : fallbackLike ? (measurementAvailable ? "1" : "0") : String(realActive));
  setText("simulatedChannels", mode === "simulated_array_demo" ? String(simulated || channels.length) : "0");
  setText("disabledChannels", mode === "simulated_array_demo" ? "0" : String(disabled));
  setText("qaWarningCount", String(warnings));
  setText(
    "surfaceSource",
    mode === "simulated_array_demo"
      ? recordedDemo ? "recorded spectrum + response proxy" : "simulated 3x3 array"
      : currentRuntimeMode
        ? "current 512-point spectrum"
      : globalSpectrumMode
        ? measurementAvailable
          ? "global nine-FBG spectrum"
          : "global nine-FBG - no frame"
      : fallbackLike
        ? measurementAvailable
          ? "P22 fallback"
          : "P22 fallback - no frame"
        : "future real 3x3 disabled"
  );
  setText(
    "surfaceRuleSource",
    !measurementAvailable
      ? "--"
      : mode === "simulated_array_demo"
      ? "coupled"
      : currentRuntimeMode
        ? "coupled"
      : globalSpectrumMode
        ? "coupled"
      : fallbackLike
        ? "P22 fallback"
        : arrayFrame?.coupling_view
          ? couplingViewLabel(arrayFrame.coupling_view)
          : "Bragg wavelength shift"
  );
  setText(
    "surfaceQualityNote",
    mode === "simulated_array_demo"
      ? recordedDemo
        ? "Data source: recorded 512-point BaySpec spectrum; surface remains a response proxy"
        : "Data source: simulated 3x3 array; mechanically coupled wavelength shift"
      : currentRuntimeMode
        ? "Data source: current full-spectrum runtime model; approximate fingertip position and continuous optical force"
      : globalSpectrumMode
        ? "Data source: global FBG01-FBG09 spectrum; provisional wavelength-order spatial proxy"
      : fallbackLike
        ? measurementAvailable
          ? "Data source: P22 fallback; real 3x3 array disabled"
          : "Data source: P22 fallback; no wavelength frame"
        : "Future real 3x3 mode is disabled"
  );

  if (arrayRawTable) {
    const rows = channels
      .map(
        (item) => {
          const target = formatWavelength(item.target_wavelength_nm, 1);
          const measured = Number.isFinite(Number(item.measured_wavelength_nm)) ? ` / meas ${formatWavelength(item.measured_wavelength_nm, 1)}` : "";
          return `<tr><td>${item.channel_id}</td><td>${formatNumber(item.x, 1)}, ${formatNumber(item.y, 1)}</td><td>${target}${measured}</td><td>${formatPm(item.delta_wavelength_pm, 1, true)}</td><td>${formatNumber(item.wavelength_shift_response_ratio ?? item.response_value, 3)}</td><td>${item.coupling_status || "--"}</td></tr>`;
        }
      )
      .join("");
    arrayRawTable.innerHTML = `<thead><tr><th>Channel</th><th>x,y</th><th>Reference nm</th><th>Shift pm</th><th>Normalized</th><th>Coupling</th></tr></thead><tbody>${rows}</tbody>`;
  }
  if (arrayJsonPreview) {
    const preview = {
      mode: arrayFrame?.mode,
      scenario: arrayFrame?.scenario,
      matrix_rows: ARRAY_DISPLAY_ROWS,
      surface_metrics: arrayFrame?.surface_metrics || {},
      coupling_sources: arrayFrame?.coupling_sources || [],
      observed_changed_channels: arrayFrame?.observed_changed_channels || [],
      primary_observed_channel: arrayFrame?.primary_observed_channel,
      secondary_observed_channels: arrayFrame?.secondary_observed_channels || [],
    };
    arrayJsonPreview.textContent = JSON.stringify(preview, null, 2);
  }
}

function updateChannelGrid(grid, selected) {
  channelGrid.innerHTML = "";
  for (const item of sortedChannels(grid)) {
    const cell = document.createElement("div");
    cell.className = `channel-cell${item.channel_id === selected || item.valid ? " active" : ""}${item.enabled ? "" : " disabled"}`;
    const observedResponse = Number(item.observed_coupled_response_ratio);
    const inferredProbability = Number(item.inferred_contact_probability);
    const response = Number.isFinite(observedResponse)
      ? observedResponse
      : Number(item.wavelength_shift_response_ratio ?? item.response_value);
    if (Number.isFinite(response)) {
      const color = colorForAttenuation(Math.max(0, Math.min(1, response)));
      cell.style.background = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.16)`;
    }
    const status = item.valid ? item.qa_status || "valid" : item.enabled ? "no data" : "disabled";
    cell.innerHTML = `<strong>${item.channel_id}</strong><span>x,y ${formatNumber(item.x, 1)}, ${formatNumber(item.y, 1)}</span><span>observed coupled ${formatNumber(observedResponse, 3)}</span><span>inferred probability ${formatNumber(inferredProbability, 3)}</span><span>${item.coupling_status || status}</span>`;
    channelGrid.appendChild(cell);
  }
}

function updateOperatorFootprint(arrayFrame, record, surfaceMetrics, arrayMode, measurementAvailable = true, heldMeasurement = false, runtimeFrame = null) {
  const channelIds = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"];
  const channels = new Map((arrayFrame?.channels || []).map((item) => [item.channel_id, item]));
  const candidatePeaks = globalCandidatePeaks(record);
  const currentRuntimeView = isModelPositionLevelMode(arrayMode);
  const globalCandidateView =
    !currentRuntimeView &&
    record?.recognition_scope === GLOBAL_RECOGNITION_SCOPE &&
    record?.physical_channel_mapping_final === false;
  const candidatesByProvisionalChannel = new Map(
    candidatePeaks.map((peak) => [peak.provisional_channel_id, peak])
  );
  const spatialChannelsByCandidate = new Map(
    (arrayFrame?.channels || [])
      .filter((item) => item?.candidate_id)
      .map((item) => [item.candidate_id, item])
  );
  const dominantCandidateId = surfaceMetrics?.contact_evidence_passed
    ? dominantGlobalCandidate(record)?.candidate_id || null
    : null;
  setText("footprintTitle", "Contact Map");
  const fallbackLike = !arrayFrame || ["p22_fallback", "single_point_p22", "no_valid_channel", ""].includes(String(arrayFrame.mode || ""));
  const globalEventPeakShiftPm = Number(surfaceMetrics?.global_event_absolute_shift_pm ?? record?.absolute_shift_pm);
  const responseBlockReason = String(record?.response_block_reason || "");
  const unavailableResponseText = responseBlockReason === "stale_source_frame"
    ? "Response unavailable - stale frame"
    : responseBlockReason === "global_candidate_baseline_required"
      ? "Response unavailable - set baseline"
      : "No current response";
  // The band, summary, footprint, and surface deformation must share one
  // normalized response. Global live frames use the 75 pm spatial proxy scale,
  // not the former 500 pm single-channel display scale.
  const peak = measurementAvailable
    ? normalizedSurfaceResponseRatio(surfaceMetrics, record)
    : Number.NaN;
  const hasActiveSurfaceResponse =
    Number.isFinite(peak) && peak >= RESPONSE_BAND_THRESHOLDS.noContactMax;
  const contactChannels = [];
  const coupledNeighborChannels = [];
  for (const id of channelIds) {
    const cell = document.getElementById(`footprint${id}`);
    if (!cell) continue;
    const candidatePeak = candidatesByProvisionalChannel.get(id);
    const candidateId = candidatePeak?.candidate_id;
    const spatialChannel = candidateId ? spatialChannelsByCandidate.get(candidateId) : null;
    const channel = globalCandidateView ? spatialChannel : channels.get(id);
    let value = measurementAvailable
      ? globalCandidateView
        ? Number(spatialChannel?.wavelength_shift_response_ratio ?? spatialChannel?.response_value)
        : Number(channel?.wavelength_shift_response_ratio ?? channel?.response_value)
      : Number.NaN;
    if (!Number.isFinite(value)) value = fallbackLike && id === "P22" ? peak : NaN;
    value = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : NaN;
    const active = Number.isFinite(value) && value >= RESPONSE_BAND_THRESHOLDS.noContactMax;
    const coupledNeighbor =
      Number.isFinite(value) &&
      value >= COUPLED_CHANNEL_VISIBILITY_THRESHOLD &&
      !active;
    if (active) contactChannels.push(id);
    else if (coupledNeighbor) coupledNeighborChannels.push(id);
    cell.classList.toggle("active", active);
    cell.classList.toggle("coupled-neighbor", coupledNeighbor);
    cell.classList.toggle(
      "dominant",
      hasActiveSurfaceResponse && (
        globalCandidateView
          ? candidateId === dominantCandidateId || id === surfaceMetrics?.dominant_channel
          : id === (surfaceMetrics?.dominant_channel || "P22")
      )
    );
    const title = cell.querySelector("strong");
    if (title) title.textContent = id;
    const label = cell.querySelector("span");
    if (label) label.textContent = Number.isFinite(value) ? formatPercent(value, 0) : "--";
    cell.setAttribute(
      "aria-label",
      Number.isFinite(value)
        ? `${id}, ${formatPercent(value, 0)} response, ${active ? "above contact threshold" : coupledNeighbor ? "coupled neighbor below contact threshold" : "below response threshold"}`
        : `${id}, no response value`
    );
    if (Number.isFinite(value)) {
      const color = colorForAttenuation(value);
      const alpha = 0.16 + 0.58 * Math.min(1, value);
      cell.style.background = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, ${alpha.toFixed(2)})`;
      cell.style.borderColor = active ? `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.82)` : "";
    } else {
      cell.style.background = "";
      cell.style.borderColor = "";
    }
  }

  const forcePresentation = opticalForcePresentation(record, arrayFrame, runtimeFrame);
  const currentRuntimeForce = forcePresentation.enabled;
  const estimatedForceN = forcePresentation.valueN;
  const forceScaleMaxN = currentRuntimeForce
    ? updateOpticalForceDisplayMaximum(estimatedForceN)
    : null;
  const responseBandRatio =
    currentRuntimeForce && estimatedForceN !== null
      ? Math.max(0, Math.min(1, estimatedForceN / forceScaleMaxN))
      : peak;
  setText("responseBandTitle", currentRuntimeForce ? "Estimated Force" : "Optical Response");
  document.querySelector(".operator-band-card")?.classList.toggle("force-estimate-mode", currentRuntimeForce);
  const responseBandLabels = Array.from(
    document.querySelectorAll(".operator-band-card .response-band-labels span")
  );
  const responseBandLabelText = currentRuntimeForce
    ? [0, 0.25, 0.5, 0.75, 1].map((ratio, index) =>
        formatForceScaleTick(forceScaleMaxN * ratio, index === 0 || index === 4)
      )
    : ["0%", "25%", "50%", "75%", "100%"];
  responseBandLabels.forEach((label, index) => {
    label.textContent = responseBandLabelText[index] || "";
  });
  const marker = document.getElementById("responseBandMarker");
  if (marker) {
    const track = marker.parentElement;
    if (track) {
      track.setAttribute(
        "aria-label",
        currentRuntimeForce
          ? `Force estimated from the optical spectrum; auto-ranging from 0 to ${formatForceScaleTick(forceScaleMaxN, true)}`
          : "Continuous normalized optical response from 0% to 100%; visual proxy only"
      );
    }
    marker.classList.toggle("unavailable", !measurementAvailable);
    const aboveCalibratedRange = currentRuntimeForce && estimatedForceN !== null && (
      forcePresentation.rangeStatus === "above_calibrated_range" ||
      estimatedForceN > forcePresentation.calibratedMaxN
    );
    marker.classList.toggle("above-calibrated-range", aboveCalibratedRange);
    const markerPercent = measurementAvailable && Number.isFinite(responseBandRatio)
      ? Math.max(0, Math.min(100, responseBandRatio * 100))
      : 0;
    marker.style.left = `${markerPercent.toFixed(1)}%`;
    marker.style.transform = `translate(${-markerPercent.toFixed(1)}%, -50%)`;
    marker.textContent = currentRuntimeForce
      ? measurementAvailable && estimatedForceN !== null
        ? `${estimatedForceN.toFixed(2)} N`
        : "-- N"
      : measurementAvailable
        ? formatPercent(peak, 0)
        : "--";
    marker.setAttribute("role", "meter");
    marker.setAttribute("aria-valuemin", "0");
    marker.setAttribute(
      "aria-valuemax",
      currentRuntimeForce ? String(forceScaleMaxN) : "100"
    );
    if (measurementAvailable) {
      const meterValue = currentRuntimeForce && estimatedForceN !== null
        ? estimatedForceN
        : Math.max(0, Math.min(100, peak * 100));
      marker.setAttribute("aria-valuenow", meterValue.toFixed(2));
      marker.setAttribute(
        "aria-valuetext",
        currentRuntimeForce && estimatedForceN !== null
          ? `${estimatedForceN.toFixed(2)} newtons estimated from the optical spectrum${aboveCalibratedRange ? "; above the currently validated range" : ""}`
          : formatPercent(peak, 0)
      );
      marker.title = aboveCalibratedRange
        ? `Above currently validated 0-${formatForceScaleTick(forcePresentation.calibratedMaxN, true)} range`
        : "";
    } else {
      marker.removeAttribute("aria-valuenow");
      marker.setAttribute("aria-valuetext", "No current measurement");
      marker.title = "";
    }
  }
  const runtimePrediction = currentRuntimeView
    ? activeModelPrediction(record, arrayFrame)
    : null;
  const responseBandValue = document.getElementById("responseBandValue");
  const compactResponseBandText = measurementAvailable
    ? currentRuntimeForce && estimatedForceN !== null
      ? `${estimatedForceN.toFixed(2)} N${heldMeasurement ? " - held" : ""}`
      : `${formatPercent(peak, 0)} optical response${heldMeasurement ? " - held" : ""}`
    : currentRuntimeForce ? "-- N" : unavailableResponseText;
  setText("responseBandValue", compactResponseBandText);
  if (responseBandValue) {
    const eventPeakShift = globalCandidateView
      ? globalEventPeakShiftPm
      : arrayFrame?.peak_wavelength_shift_pm ?? record?.absolute_shift_pm;
    const evidenceDetail = runtimePrediction
      ? currentRuntimeForce && estimatedForceN !== null
        ? `${activeModelDisplayName(record, arrayFrame)}; ${estimatedForceN.toFixed(2)} N estimated from spectrum`
        : `${activeModelDisplayName(record, arrayFrame)}; ${formatPercent(peak, 0)} normalized visual response`
      : `${formatPercent(peak, 0)} normalized response; ${formatPm(eventPeakShift, 1)} event peak |shift|`;
    responseBandValue.title = measurementAvailable ? evidenceDetail : unavailableResponseText;
    responseBandValue.setAttribute(
      "aria-label",
      measurementAvailable ? `${compactResponseBandText}. ${evidenceDetail}` : unavailableResponseText
    );
  }
  let note = "raw coupled response";
  let noteDetail = "Raw coupled wavelength response";
  if (!measurementAvailable) note = noteDetail = unavailableResponseText;
  else if (heldMeasurement) note = noteDetail = "Last frame held; acquisition stopped";
  else if (arrayMode === "simulated_array_demo") {
    note = contactChannels.length
      ? `${contactChannels.length} contact - ${coupledNeighborChannels.length} coupled`
      : coupledNeighborChannels.length
        ? `Below threshold - ${coupledNeighborChannels.length} coupled`
        : "No active channels";
    noteDetail = contactChannels.length
      ? `${contactChannels.length} ${contactChannels.length === 1 ? "channel" : "channels"} above the ${formatPercent(RESPONSE_BAND_THRESHOLDS.noContactMax, 0)} contact threshold; ${coupledNeighborChannels.length} coupled ${coupledNeighborChannels.length === 1 ? "neighbor" : "neighbors"} below it`
      : coupledNeighborChannels.length
        ? `${coupledNeighborChannels.length} coupled ${coupledNeighborChannels.length === 1 ? "channel" : "channels"} visible below the ${formatPercent(RESPONSE_BAND_THRESHOLDS.noContactMax, 0)} contact threshold`
        : "No active channels; baseline or recovery state";
  }
  else if (isModelPositionLevelMode(arrayMode)) {
    note = runtimePrediction?.digital_twin?.active
      ? `Model position ${runtimePrediction?.position?.label || "--"}; broad manual fingertip contact domain`
      : `${activeModelDisplayName(record, arrayFrame)} - no active contact`;
  }
  else if (globalCandidateView) {
    note = candidatePeaks.length === GLOBAL_CANDIDATE_IDS.length
      ? "Provisional FBG-to-position proxy; labelled point-press calibration pending"
      : `${candidatePeaks.length}/9 candidates; incomplete global frame`;
  }
  else if (fallbackLike) note = "P22 fallback, real 3x3 disabled";
  if (arrayMode !== "simulated_array_demo") noteDetail = note;
  setText("footprintNote", note);
  const footprintNote = document.getElementById("footprintNote");
  if (footprintNote) {
    footprintNote.title = noteDetail;
    footprintNote.setAttribute("aria-label", noteDetail);
  }
}

function syncPrimaryCommandLabels() {
  const liveActive = state.exportWatchActive || state.sdkLiveActive;
  if (liveTwinButton && !liveTwinButton.classList.contains("command-busy")) {
    setCommandButtonDescription(
      liveTwinButton,
      liveActive ? "Stop live" : "Live",
      liveActive ? "Stop live" : "Start live"
    );
  }
  liveTwinButton?.classList.toggle("active-watch", liveActive);
  liveTwinButton?.setAttribute("aria-pressed", liveActive ? "true" : "false");
  if (liveTwinButton) {
    liveTwinButton.title = liveActive ? "Stop live acquisition" : "Start direct BaySpec SDK acquisition";
  }

  if (exportWatchButton && !exportWatchButton.classList.contains("command-busy")) {
    setCommandButtonDescription(
      exportWatchButton,
      state.exportWatchActive ? "Stop watch" : "Watch exports",
      state.exportWatchActive ? "Stop watch" : "Watch exports"
    );
  }
  exportWatchButton?.classList.toggle("active-watch", state.exportWatchActive);
  exportWatchButton?.setAttribute("aria-pressed", state.exportWatchActive ? "true" : "false");
  if (exportWatchButton) {
    exportWatchButton.title = state.exportWatchActive
      ? "Stop watching the Sense export folder"
      : "Watch the Sense export folder for new spectra";
  }

  if (baselineButton && !baselineButton.classList.contains("command-busy")) {
    setCommandButtonDescription(baselineButton, "Set baseline");
  }
  if (ingestExportButton && !ingestExportButton.classList.contains("command-busy")) {
    setCommandButtonDescription(ingestExportButton, "Ingest", "Ingest latest export");
  }
  if (resetButton && !resetButton.classList.contains("command-busy")) {
    setCommandButtonDescription(resetButton, "Reset trace");
  }
  syncCommandAvailability();
}

function syncCommandAvailability() {
  if (state.commandPending) return;

  const measurementAvailable = frameHasMeasurement(state.frame);
  const baselineReady =
    measurementAvailable &&
    frameSourceIsFresh(state.frame) &&
    !state.paused &&
    !state.demoModeActive &&
    !state.arrayDemoActive;
  const holdReady = Boolean(
    state.demoModeActive ||
      state.arrayDemoActive ||
      state.exportWatchActive ||
      state.sdkLiveActive ||
      state.liveRequested ||
      (state.dataStreamActive && measurementAvailable)
  );

  if (baselineButton) {
    baselineButton.disabled = !baselineReady;
    baselineButton.title = baselineReady
      ? "Release contact, wait for the spectrum to stabilize, then set the recovery-state baseline"
      : state.demoModeActive || state.arrayDemoActive
        ? "Global baseline is unavailable while a local response is active"
        : !frameSourceIsFresh(state.frame)
          ? "Global baseline requires fresh live spectrum frames"
          : state.paused
            ? "Resume display updates before setting a baseline"
            : "Baseline requires one fresh full-spectrum frame";
  }

  if (pauseButton) {
    pauseButton.disabled = !holdReady;
    pauseButton.title = holdReady
      ? state.paused
        ? "Resume display updates"
        : "Hold the current display without stopping acquisition"
      : "Hold becomes available when a data stream or local response is active";
  }

  demoMenuButton?.classList.toggle("demo-active", state.demoModeActive || state.arrayDemoActive);
  demoMenuButton?.setAttribute("aria-pressed", String(state.demoModeActive || state.arrayDemoActive));
}

function updateWatcherUI(watcher, sdkLive) {
  const watchActive = Boolean(watcher?.active);
  const sdkActive = Boolean(sdkLive?.active);
  state.exportWatchActive = watchActive;
  state.sdkLiveActive = sdkActive;
  syncPrimaryCommandLabels();
  const freshness = sdkActive
    ? sdkLive?.freshness || "waiting_for_sdk_frame"
    : watcher?.freshness || (watchActive ? "waiting_for_export" : "stopped");
  setDataSourceDisplay(acquisitionDisplayState(watcher, sdkLive));
  setText("watchStatus", sdkActive ? `SDK active (${sdkLive?.interval_ms || "--"} ms)` : watchActive ? `watch active (${formatNumber(watcher?.interval_sec, 2)} s)` : "stopped");
  setText("liveFreshness", freshness);
  setText("frameAge", formatSeconds(sdkActive ? sdkLive?.seconds_since_last_frame : watcher?.seconds_since_last_ingest));
  setText("watchIngestCount", sdkActive ? sdkLive?.frame_count ?? "--" : watcher?.ingest_count ?? "--");
  setText("watchError", sdkActive ? sdkLive?.last_error || "none" : watcher?.last_error || "none");
  if (sdkLive?.spectrum_processing) {
    renderSpectrumProcessingStatus(sdkLive.spectrum_processing);
  }
}

function setDemoStatus(text, mode = "ready") {
  if (!demoStatusChip) return;
  demoStatusChip.textContent = text;
  demoStatusChip.className = mode;
}

function updateDemoControls() {
  demoStepButtons.forEach((button) => {
    button.classList.toggle("demo-active", button.dataset.demoLevel === state.demoCurrentLevel);
  });
  arrayDemoStepButtons.forEach((button) => {
    button.classList.toggle("demo-active", button.dataset.arrayScenario === state.arrayDemoScenario && state.arrayDemoActive);
  });
  if (nodeDebugButton) {
    nodeDebugButton.textContent = state.nodeDebugExpanded ? "Hide nodes" : "Node debug";
    nodeDebugButton.classList.toggle("demo-active", state.nodeDebugExpanded);
  }
  if (demoAutoButton) {
    demoAutoButton.textContent = state.demoAutoplay ? "Stop 5 s loop" : "Start 5 s loop";
    demoAutoButton.classList.toggle("demo-active", state.demoAutoplay);
  }
  if (demoSingleButton) {
    demoSingleButton.classList.toggle(
      "demo-active",
      state.arrayDemoPlaybackMode === "single" && state.arrayDemoActive && !state.arrayDemoActionComplete
    );
  }
  if (demoSpeedControl) demoSpeedControl.value = String(state.demoPlaybackRate);
  if (demoSpeedValue) demoSpeedValue.value = `${state.demoPlaybackRate.toFixed(1)}x`;
  appShell?.classList.toggle("local-response-active", state.demoModeActive || state.arrayDemoActive);
  syncCommandAvailability();
}

function setDemoPlaybackRate(value, { persist = true } = {}) {
  state.demoPlaybackRate = normalizedDemoPlaybackRate(value);
  // Playback speed also controls the cadence of released baseline frames so
  // the trace and spectrum keep advancing during the idle part of a loop.
  state.arrayDemoNextStepAt = performance.now() + demoArrayStepIntervalMs();
  if (persist) {
    try {
      window.localStorage.setItem(DEMO_PLAYBACK_RATE_STORAGE_KEY, String(state.demoPlaybackRate));
    } catch {}
  }
  if (state.demoTimer) {
    clearTimeout(state.demoTimer);
    state.demoTimer = null;
  }
  updateDemoControls();
}

function stopDemoAutoplay() {
  if (state.demoTimer) {
    clearTimeout(state.demoTimer);
    state.demoTimer = null;
  }
  state.demoAutoplay = false;
  if (state.arrayDemoPlaybackMode === "loop") {
    // Stop means finish the current physical action, release, then remain idle.
    state.arrayDemoPlaybackMode = "single";
  }
  updateDemoControls();
  if (state.demoModeActive) {
    setDemoStatus("demo", "running");
  } else {
    setDemoStatus("ready", "ready");
  }
}

function snapDisplayedFrameToCurrentTargets() {
  state.smoothTraceRecords = cloneTraceRecords(state.targetTraceRecords);
  state.smoothSpectrumRecord = cloneSpectrumRecord(state.targetSpectrumRecord);
  state.smoothAttenuation = state.targetAttenuation;
  state.smoothDeformation = state.targetDeformation;
  state.smoothSurfaceVisualPeak = state.targetSurfaceVisualPeak;
  state.smoothSurfaceCentroidX = state.targetSurfaceCentroidX;
  state.smoothSurfaceCentroidY = state.targetSurfaceCentroidY;
  state.smoothSurfaceSpread = state.targetSurfaceSpread;
  state.smoothSurfaceActiveArea = state.targetSurfaceActiveArea;

  drawVisibleCharts();
  if (surfaceGeometry) {
    applyThreeGeometry(state.smoothAttenuation, state.smoothDeformation, {
      recomputeNormals: true,
    });
  }
  state.chartsNeedRefresh = false;
  state.threeNeedsRefresh = false;
}

function setPaused(paused) {
  const nextPaused = Boolean(paused);
  if (nextPaused !== state.paused) invalidateFrameRequestContext();
  state.paused = nextPaused;
  if (state.paused) {
    // Hold is an atomic visual snapshot: charts, labels, footprint, band, and
    // 3D geometry must all represent the same committed frame.
    snapDisplayedFrameToCurrentTargets();
  }
  setCommandButtonDescription(
    pauseButton,
    state.paused ? "Resume display" : "Hold display",
    state.paused ? "Resume" : "Hold display"
  );
  if (pauseButton) {
    pauseButton.title = state.paused
      ? "Resume the live display"
      : "Hold the current display without stopping acquisition";
  }
  pauseButton.classList.toggle("active-pause", state.paused);
  pauseButton.setAttribute("aria-pressed", state.paused ? "true" : "false");
  syncCommandAvailability();
  if (state.frame) {
    updateSurfaceFrameState(state.frame, state.frame.export_watcher || {}, state.frame.sdk_live || {});
    updateOperatorStreamSummary(
      state.frame,
      state.frame.export_watcher || {},
      state.frame.sdk_live || {},
      document.getElementById("topQaStatus")?.textContent || "OK"
    );
  }
}

async function stopLiveInputsForDemo() {
  await requestJSON(
    "/api/live/stop?control_sense=false",
    { method: "POST" },
    { timeoutMs: 12000 }
  );
  state.exportWatchActive = false;
  state.sdkLiveActive = false;
  state.liveRequested = false;
}

async function ensureLiveInputsStoppedForDemo() {
  const captureRunning = state.frame?.optical_force_capture?.running === true;
  if (state.exportWatchActive || state.sdkLiveActive || state.liveRequested || captureRunning) {
    throw new Error(
      captureRunning
        ? "Stop recording before opening a simulated response."
        : "Stop live acquisition before opening a simulated response."
    );
  }
}

async function prepareDemoMode({ reset = false } = {}) {
  if (reset || !state.demoModeActive || state.arrayDemoActive) invalidateFrameRequestContext();
  setPaused(false);
  await ensureLiveInputsStoppedForDemo();
  inputSourceSelect.value = "http_ingest";
  state.arrayDemoActive = false;
  state.arrayDemoScenario = null;
  state.dataStreamActive = true;
  state.demoModeActive = true;
  setDemoStatus(state.demoAutoplay ? "auto" : "demo", state.demoAutoplay ? "auto" : "running");
}

async function injectDemoFrame(level, { reset = false } = {}) {
  const scenario = level === "no_contact" ? "no_contact" : "center_press";
  await injectArrayDemoFrame(scenario, {
    resetTrajectory: reset,
    playbackMode: "single",
  });
  state.demoCurrentLevel = level;
  updateDemoControls();
}

async function runDemoTransition(action, button = null) {
  try {
    await action();
    return true;
  } catch (error) {
    console.warn("[demo-transition]", error);
    if (button) {
      button.classList.add("command-error");
      window.setTimeout(() => button.classList.remove("command-error"), 4500);
    }
    setCommandFeedback(
      commandErrorMessage(error, "Unable to switch playback while acquisition is stopping"),
      "error",
      { autoHideMs: 7000 }
    );
    return false;
  }
}

function resetArrayDemoTraceHistory(channelId = "P22") {
  const now = Date.now();
  state.arrayDemoTraceRecords = Array.from({ length: 10 }, (_, index) =>
    recordFromAttenuationPercent(0, now - (10 - index) * 120, channelId)
  );
}

function appendArrayDemoTrace(arrayFrame) {
  const peakRatio = Math.max(0, Math.min(1, Number(arrayFrame?.surface_metrics?.surface_peak ?? arrayFrame?.peak_shift_response ?? 0)));
  const peakPercent = peakRatio * 100;
  const now = Date.now();
  const channelId = arrayFrame?.surface_metrics?.dominant_channel || arrayFrame?.dominant_channel || "P22";
  if (!state.arrayDemoTraceRecords.length) resetArrayDemoTraceHistory(channelId);
  state.arrayDemoTraceRecords.push(recordFromAttenuationPercent(peakPercent, now, channelId));
  if (state.arrayDemoTraceRecords.length > DEMO_TRACE_WINDOW_POINTS) {
    state.arrayDemoTraceRecords.splice(0, state.arrayDemoTraceRecords.length - DEMO_TRACE_WINDOW_POINTS);
  }
  return state.arrayDemoTraceRecords.map((item) => ({ ...item }));
}

function frameFromArrayDemo(arrayFrame) {
  const spectrum = arrayFrame?.spectrum || {};
  const dominantChannel =
    spectrum?.dominant_channel ||
    arrayFrame?.dominant_channel ||
    arrayFrame?.surface_metrics?.dominant_channel ||
    "P22";
  const selected =
    (arrayFrame?.channels || []).find((channel) => channel.channel_id === dominantChannel) ||
    (arrayFrame?.channels || []).find((channel) => channel.channel_id === "P22") ||
    (arrayFrame?.channels || [])[0] ||
    {};
  const spectrumPeaks = Array.isArray(spectrum?.peaks) ? spectrum.peaks : [];
  return {
    ok: true,
    mode: arrayFrame?.demo_data_mode || arrayFrame?.mode || "recorded_real_spectrum_reference",
    frame_id: arrayFrame?.frame_id,
    timestamp: arrayFrame?.timestamp,
    selected_channel: selected.channel_id || dominantChannel,
    latest: {
      ...selected,
      source: spectrum?.source_session_id || arrayFrame?.demo_source_session_id || "recorded_demo_reference",
      demodulation_mode: "fbg_wavelength_shift",
      response_basis: "recorded spectrum with synchronized Fz visual proxy",
      peak_axis_type: spectrum?.axis_type === "wavelength_nm" ? "wavelength_nm" : "recorded",
      spectrum_x_unit: spectrum?.axis_type || "wavelength_nm",
      wavelength_nm: spectrum?.wavelength_nm || [],
      spectrum_wavelength_nm: spectrum?.wavelength_nm || [],
      intensity: spectrum?.intensity || [],
      spectrum_counts: spectrum?.intensity || [],
      spectrum_peaks: spectrumPeaks,
      spectrum_peak_profile: spectrum?.spectrum_peak_profile,
      spectrum_peak_mapping_status: spectrum?.spectrum_peak_mapping_status,
      peak_assignment_method: spectrum?.peak_assignment_method,
      physical_channel_mapping_final: Boolean(spectrum?.physical_channel_mapping_final),
      spectrum_points: Array.isArray(spectrum?.intensity) ? spectrum.intensity.length : 0,
      spectrum_type: spectrum?.spectrum_type || "recorded spectrum unavailable",
      spectrum_source_note: spectrum?.source_note || "recorded spectrum unavailable",
      peak_height_mode: spectrum?.peak_height_mode || "recorded_counts",
      intensity_modulation_enabled: Boolean(spectrum?.intensity_modulation_enabled),
      frame_render_semantics: spectrum?.frame_render_semantics || "replace_previous_spectrum",
      wavelength_plan: spectrum?.wavelength_plan || arrayFrame?.wavelength_plan,
      spectrum_frame_id: arrayFrame?.spectrum_frame_id,
      surface_frame_id: arrayFrame?.surface_frame_id,
      trace_frame_id: arrayFrame?.trace_frame_id,
      frame_sync_status: arrayFrame?.frame_sync_status || "synced",
      frame_id: arrayFrame?.frame_id,
      timestamp: arrayFrame?.timestamp,
      dominant_channel: dominantChannel,
      selected_channel: selected.channel_id || dominantChannel,
      coupling_view: arrayFrame?.coupling_view || state.couplingView,
      coupling_status: selected.coupling_status || arrayFrame?.coupling_status || "uncalibrated_mechanically_coupled_wavelength_shift",
      coupling_compensated: Boolean(selected.coupling_compensated),
      coupling_sources: arrayFrame?.coupling_sources || selected.coupling_sources || [],
      observed_changed_channels: arrayFrame?.observed_changed_channels || [],
      primary_observed_channel: arrayFrame?.primary_observed_channel,
      secondary_observed_channels: arrayFrame?.secondary_observed_channels || [],
      possible_cross_fiber_coupling: Boolean(arrayFrame?.possible_cross_fiber_coupling || selected.possible_cross_fiber_coupling),
      possible_same_fiber_coupling: Boolean(arrayFrame?.possible_same_fiber_coupling || selected.possible_same_fiber_coupling),
      local_response_estimate_available: Boolean(arrayFrame?.local_response_estimate_available || selected.local_response_estimate_available),
      coupling_model_note: arrayFrame?.coupling_model_note,
      demo_data_mode: arrayFrame?.demo_data_mode,
      demo_reference_kind: arrayFrame?.demo_reference_kind,
      demo_source_session_id: arrayFrame?.demo_source_session_id,
      demo_source_capture_index: arrayFrame?.demo_source_capture_index,
      demo_source_elapsed_time_sec: arrayFrame?.demo_source_elapsed_time_sec,
      demo_source_position: arrayFrame?.demo_source_position,
      recorded_reference_force_fz_n: arrayFrame?.recorded_reference_force_fz_n,
      estimated_force_fz_n: arrayFrame?.estimated_force_fz_n,
      optical_force_estimate: arrayFrame?.optical_force_estimate,
      active_spectral_model_source: arrayFrame?.active_spectral_model_source,
    },
    trace: appendArrayDemoTrace(arrayFrame),
    channel_grid: arrayFrame?.channels || [],
    array_frame: arrayFrame,
    surface_grid: arrayFrame?.surface_grid,
    surface_metrics: arrayFrame?.surface_metrics,
    surface_note: arrayFrame?.surface_note,
    estimated_force_fz_n: arrayFrame?.estimated_force_fz_n,
    optical_force_estimate: arrayFrame?.optical_force_estimate,
    status: {
      buffered_records: 0,
      channels_seen: [spectrum?.source_session_id || "recorded_demo_reference"],
    },
    export_watcher: { active: false, freshness: "recorded" },
    sdk_live: { active: false, freshness: "recorded" },
    sense_control: {},
  };
}

async function injectArrayDemoFrame(
  scenario,
  { resetTrajectory = true, playbackMode = "single" } = {}
) {
  invalidateFrameRequestContext();
  const demoEpoch = state.frameModeEpoch;
  stopDemoAutoplay();
  await ensureLiveInputsStoppedForDemo();
  setPaused(false);
  inputSourceSelect.value = "http_ingest";
  state.demoModeActive = true;
  state.arrayDemoActive = true;
  state.arrayDemoScenario = scenario;
  state.arrayDemoPlaybackMode = playbackMode === "single" ? "single" : "loop";
  state.demoAutoplay = state.arrayDemoPlaybackMode === "loop";
  state.arrayDemoActionComplete = false;
  state.arrayDemoCycleStartedAt = performance.now();
  state.arrayDemoActionCompletedAt = 0;
  if (resetTrajectory) {
    state.arrayDemoStep = 0;
    state.trajectoryHistory = [];
    resetArrayDemoTraceHistory();
  }
  const stepCount = ARRAY_SLIDE_STEPS[scenario] || 1;
  const step = Math.min(state.arrayDemoStep, Math.max(0, stepCount - 1));
  const data = await requestJSON(
    `/api/array_demo/frame?scenario=${encodeURIComponent(scenario)}&step=${step}&coupling_view=${encodeURIComponent(state.couplingView)}`,
    { cache: "no-store" },
    { timeoutMs: 7000 }
  );
  if (data?.ok === false || !data?.array_frame) {
    throw new Error(data?.reason || data?.status || "Simulated response is unavailable.");
  }
  if (demoEpoch !== state.frameModeEpoch) return;
  const arrayFrame = data.array_frame;
  arrayFrame.scenario = scenario;
  setDemoStatus(
    state.arrayDemoPlaybackMode === "loop" ? "5 s loop" : "playing once",
    state.arrayDemoPlaybackMode === "loop" ? "auto" : "running"
  );
  updateDemoControls();
  const frame = frameFromArrayDemo(arrayFrame);
  state.frame = frame;
  updateUI(frame);
  state.arrayDemoNextStepAt = performance.now() + demoArrayStepIntervalMs();
  state.arrayDemoStep = Math.min(stepCount, state.arrayDemoStep + 1);
}

async function injectArrayDemoFrameAtStep(scenario, step = 0) {
  invalidateFrameRequestContext();
  const demoEpoch = state.frameModeEpoch;
  stopDemoAutoplay();
  await ensureLiveInputsStoppedForDemo();
  setPaused(true);
  inputSourceSelect.value = "http_ingest";
  state.demoModeActive = true;
  state.arrayDemoActive = true;
  state.arrayDemoScenario = scenario;
  state.arrayDemoPlaybackMode = "single";
  state.arrayDemoActionComplete = true;
  state.arrayDemoActionCompletedAt = performance.now();
  state.demoAutoplay = false;
  state.arrayDemoStep = Math.max(0, Number(step) || 0);
  state.trajectoryHistory = [];
  resetArrayDemoTraceHistory();
  const data = await requestJSON(
    `/api/array_demo/frame?scenario=${encodeURIComponent(scenario)}&step=${state.arrayDemoStep}&coupling_view=${encodeURIComponent(state.couplingView)}`,
    { cache: "no-store" },
    { timeoutMs: 7000 }
  );
  if (data?.ok === false || !data?.array_frame) {
    throw new Error(data?.reason || data?.status || "Simulated response is unavailable.");
  }
  if (demoEpoch !== state.frameModeEpoch) return;
  const arrayFrame = data.array_frame;
  arrayFrame.scenario = scenario;
  setDemoStatus(simulatedScenarioStateLabel(arrayFrame, arrayFrame?.surface_metrics || {}), "running");
  updateDemoControls();
  const frame = frameFromArrayDemo(arrayFrame);
  state.frame = frame;
  updateUI(frame);
  state.arrayDemoNextStepAt = performance.now() + demoArrayStepIntervalMs();
  state.arrayDemoStep += 1;
}

function updateDemoReadout(record) {
  const recordSource = String(record?.source || "");
  const isDemoRecord = state.demoModeActive || recordSource.includes("demo");
  if (!isDemoRecord) {
    setText("demoCurrentLevel", "off");
    setText("demoBaseline", "--");
    setText("demoCurrentIntensity", "--");
    setText("demoCurrentAttenuation", "--");
    return;
  }
  if (state.arrayDemoActive) {
    setText(
      "demoCurrentLevel",
      simulatedScenarioStateLabel(state.currentArrayFrame || {}, state.currentArrayFrame?.surface_metrics || {})
    );
    setText("demoBaseline", formatWavelength(record?.baseline_wavelength_nm, 5));
    setText("demoCurrentIntensity", formatWavelength(record?.tracked_wavelength_nm ?? record?.peak_wavelength_nm, 5));
    setText("demoCurrentAttenuation", formatPm(record?.delta_wavelength_pm, 1, true));
    setText("demoReadoutNote", "Simulated 3x3 Bragg wavelength-shift surface; not live data and not a measured 3x3 array.");
    return;
  }
  setText("demoCurrentLevel", levelLabel(record?.response_level));
  setText("demoBaseline", formatWavelength(record?.baseline_wavelength_nm ?? DEMO_BASELINE_WAVELENGTH_NM, 5));
  setText("demoCurrentIntensity", formatWavelength(record?.tracked_wavelength_nm ?? record?.peak_wavelength_nm, 5));
  setText("demoCurrentAttenuation", formatPm(record?.delta_wavelength_pm, 1, true));
  setText(
    "demoReadoutNote",
    state.demoAutoplay
      ? "Auto demo is cycling synchronized wavelength-shift and shifted-spectrum frames; not live data."
      : "Manual demo frame with synchronized Bragg wavelength-shift input."
  );
}

function frameSyncDiagnostics(frame, record, trace, arrayFrame) {
  const surfaceId = arrayFrame?.surface_frame_id ?? arrayFrame?.frame_id ?? record?.surface_frame_id ?? frame?.frame_id;
  const spectrumId = record?.spectrum_frame_id ?? record?.frame_id;
  const lastTrace = Array.isArray(trace) && trace.length ? trace[trace.length - 1] : null;
  const traceId = record?.trace_frame_id ?? lastTrace?.frame_id ?? frame?.frame_id;
  const timestamp = record?.timestamp ?? arrayFrame?.timestamp ?? frame?.timestamp ?? null;
  if (frame?.mode === "operator_idle" || record?.source === "operator_idle") {
    return { surfaceId, spectrumId: null, traceId, timestamp, status: "idle", hasSpectrum: false };
  }
  const hasSurface = Array.isArray(arrayFrame?.surface_grid) || Array.isArray(frame?.surface_grid);
  const spectrum = spectrumArrays(record);
  const hasSpectrum = spectrum.intensity.length > 0 && spectrum.xValues.length === spectrum.intensity.length;
  let status = "synced";
  if (!hasSurface && hasSpectrum) status = "surface_missing";
  else if (hasSurface && !hasSpectrum) status = "spectrum_missing";
  else if (
    surfaceId !== undefined &&
    spectrumId !== undefined &&
    traceId !== undefined &&
    (String(surfaceId) !== String(spectrumId) || String(surfaceId) !== String(traceId))
  ) {
    status = "frame_mismatch";
  }
  return {
    surfaceId,
    spectrumId,
    traceId,
    timestamp,
    status: status === "synced" ? arrayFrame?.frame_sync_status || record?.frame_sync_status || status : status,
    hasSpectrum,
  };
}

function updateUI(inputFrame) {
  const frame = atomicNeutralFrame(inputFrame);
  if (state.frame === inputFrame && frame !== inputFrame) state.frame = frame;
  const record = frame?.latest;
  const trace = frame?.trace || [];
  const status = frame?.status || {};
  const watcher = frame?.export_watcher || {};
  const sdkLive = frame?.sdk_live || {};
  const senseControl = frame?.sense_control || {};
  const arrayFrame = frame?.array_frame || null;
  applyResponseBandThresholds(
    frame?.response_band_thresholds || arrayFrame?.response_band_thresholds || {}
  );
  const surfaceMetrics = arrayFrame?.surface_metrics || {};
  const syncDiag = frameSyncDiagnostics(frame, record, trace, arrayFrame);
  const currentModelDisplay = currentModelDisplayReady(frame);
  const opticalForceDisplay = opticalForcePresentation(record, arrayFrame, frame);
  const currentModelSource = opticalForceDisplay.source;
  const currentRuntimeActive = opticalForceDisplay.enabled;
  const opticalForceEstimate = opticalForceDisplay.valueN;
  const opticalForceEstimateField = document.getElementById("opticalForceEstimateField");
  if (opticalForceEstimateField) opticalForceEstimateField.hidden = !currentRuntimeActive;
  const opticalForceEstimateLabel = opticalForceEstimateField?.querySelector("span");
  if (opticalForceEstimateLabel) opticalForceEstimateLabel.textContent = "Estimated Force";
  setText(
    "opticalForceEstimate",
    currentRuntimeActive && opticalForceEstimate !== null
      ? `${opticalForceEstimate.toFixed(2)} N`
      : "-- N"
  );
  state.currentArrayFrame = arrayFrame;
  state.currentSurfaceGrid = Array.isArray(arrayFrame?.surface_grid) ? arrayFrame.surface_grid : null;
  state.currentSurfaceMetrics = surfaceMetrics;
  const wavelengthPlan = arrayFrame?.wavelength_plan || record?.wavelength_plan || {};
  setText("wavelengthPlanChip", wavelengthPlanText(wavelengthPlan, true));
  const cx = Number(surfaceMetrics.surface_centroid_x);
  const cy = Number(surfaceMetrics.surface_centroid_y);
  if (Number.isFinite(cx) && Number.isFinite(cy) && Number(surfaceMetrics.surface_peak) > 0.03) {
    state.trajectoryHistory.push({ x: cx, y: cy });
    if (state.trajectoryHistory.length > 50) state.trajectoryHistory.shift();
  }

  const globalRecognitionFrame = frame?.scope === GLOBAL_RECOGNITION_SCOPE;
  const completeGlobalRecognitionFrame = isGlobalSpectrumFrame(frame, record);
  const globalDominantPeak = globalRecognitionFrame ? dominantGlobalCandidate(record) : null;
  const dominantChannel = (
    currentModelDisplay
      ? surfaceMetrics.inferred_dominant_channel || surfaceMetrics.dominant_channel || record?.model_position_id
      : null
  ) || surfaceMetrics.dominant_channel || record?.dominant_channel || globalDominantPeak?.candidate_id || arrayFrame?.dominant_channel || frame?.selected_channel || state.selectedChannel;
  const measurementAvailable = frameHasMeasurement(frame);
  const responseAvailable = frameResponseIsUsable(frame);
  const sourceDisplayState = acquisitionDisplayState(watcher, sdkLive);
  const compactQaStatus = record?.qa_status || surfaceMetrics.quality_status || syncDiag.status || "ok";
  const operatorQa = measurementAvailable
    ? operatorQaLabel(compactQaStatus, record)
    : sourceDisplayState.tone === "error"
      ? "SOURCE ERROR"
      : sourceDisplayState.tone === "waiting"
        ? "WAITING"
        : "NO FRAME";
  const qaTone = !measurementAvailable
    ? sourceDisplayState.tone === "error"
      ? "review"
      : sourceDisplayState.tone === "waiting"
        ? "warning"
        : "idle"
    : operatorQa.startsWith("OK")
      ? "ok"
       : operatorQa === "CHECK" || operatorQa === "BASELINE" || operatorQa === "STALE"
         ? "warning"
        : "review";
  const spectrumIsSynthetic = isSyntheticWavelengthSpectrum(record);
  const candidateSpectrumPeaks = (Array.isArray(record?.spectrum_peaks) ? record.spectrum_peaks : [])
    .filter((peak) => peak?.candidate_mapping && peak?.valid !== false);
  const hybridSpectrumEvidence = String(record?.spectral_evidence_semantics || "") === "mixed_wavelength_intensity_shape";
  const arrayMode = String(arrayFrame?.mode || (globalRecognitionFrame ? "global_spectrum_invalid" : "p22_fallback"));
  const isFallbackLikeFrame = !globalRecognitionFrame && (arrayMode === "p22_fallback" || arrayMode === "no_valid_channel");
  const heldMeasurement = measurementAvailable && sourceDisplayState.tone === "idle" && !state.demoModeActive;
  const displayRecord = measurementAvailable ? record : null;
  const measurementState = !measurementAvailable ? "no_data" : heldMeasurement ? "held" : state.demoModeActive ? "demo" : "current";
  if (opticalSummaryCard) opticalSummaryCard.dataset.measurementState = measurementState;
  updateOperatorAlert({ record, sourceState: sourceDisplayState, operatorQa, measurementAvailable });
  setText("topQaStatus", operatorQa);
  const topQaElement = document.getElementById("topQaStatus");
  if (topQaElement) {
    topQaElement.dataset.qaTone = qaTone;
    topQaElement.setAttribute(
      "aria-label",
      operatorQa === "OK"
        ? "Quality status OK"
        : `Quality status ${operatorQa.toLowerCase()}. Open Diagnostics for details.`
    );
  }

  const recordedDemoActive = state.arrayDemoActive && isRecordedDemoSpectrum(displayRecord, arrayFrame);
  const sourceName = state.demoModeActive
    ? recordedDemoActive ? "RECORDED DATA" : "SIMULATION"
    : inputSourceSelect?.value === "export_watch"
      ? "SENSE WATCH"
      : inputSourceSelect?.value === "http_ingest"
        ? "HTTP INGEST"
        : "BAYSPEC SDK";
  const sourceHealthTone = sourceDisplayState.tone === "error"
    ? "error"
    : sourceDisplayState.tone === "waiting"
      ? "warning"
      : sourceDisplayState.tone === "demo"
        ? "demo"
        : sourceDisplayState.tone === "live"
          ? "ok"
          : "neutral";
  const frameStateLabel = !measurementAvailable
    ? "NO FRAME"
    : heldMeasurement || state.paused
      ? "HELD"
      : state.demoModeActive
        ? recordedDemoActive ? "RECORDED" : "SIMULATED"
        : "AVAILABLE";
  const frameStateTone = !measurementAvailable
    ? "neutral"
    : heldMeasurement || state.paused
      ? "warning"
      : state.demoModeActive
        ? "demo"
        : "ok";
  const baselineStatusKey = String(displayRecord?.baseline_status || "").toLowerCase();
  const currentRuntimeBaselineReady =
    displayRecord?.active_spectral_model_status === "ready" ||
    baselineStatusKey.includes("current_runtime_baseline_ready");
  const globalBaselineReady = currentRuntimeBaselineReady || (
    globalRecognitionFrame &&
    globalCandidatePeaks(displayRecord).length === GLOBAL_CANDIDATE_IDS.length &&
    globalCandidatePeaks(displayRecord).every(
      (peak) => peak?.candidate_reference_status === "session_global_no_contact_baseline"
    )
  );
  const baselineValueAvailable = globalRecognitionFrame
    ? globalBaselineReady
    : Number.isFinite(Number(displayRecord?.baseline_intensity_counts));
  const baselineStateLabel = !measurementAvailable
    ? "NOT AVAILABLE"
    : baselineStatusKey.includes("collect")
      ? "COLLECTING"
      : baselineStatusKey.includes("required") || !baselineValueAvailable
        ? "REQUIRED"
        : "READY";
  const baselineStateTone = !measurementAvailable
    ? "neutral"
    : baselineStateLabel === "READY"
      ? "ok"
      : "warning";
  const frameSyncLabel = !measurementAvailable ? "NO FRAME" : String(syncDiag.status || "UNKNOWN").toUpperCase();
  const frameSyncTone = !measurementAvailable
    ? "neutral"
    : syncDiag.status === "synced"
      ? "ok"
      : "warning";
  const spectrumStateLabel = !measurementAvailable
    ? "NO FRAME"
    : !syncDiag.hasSpectrum
      ? "MISSING"
      : spectrumIsSynthetic
        ? "SIMULATED"
        : "AVAILABLE";
  const spectrumStateTone = !measurementAvailable
    ? "neutral"
    : !syncDiag.hasSpectrum
      ? "warning"
      : spectrumIsSynthetic
        ? "demo"
        : "ok";
  const axisStateLabel = !measurementAvailable
    ? "--"
    : record?.peak_axis_type === "pixel_index"
      ? "PIXEL"
      : "WAVELENGTH";
  const axisStateTone = !measurementAvailable
    ? "neutral"
    : record?.peak_axis_type === "pixel_index"
      ? "warning"
      : "ok";
  setHealthState("diagnosticSourceState", sourceName, sourceHealthTone);
  setHealthState("diagnosticStreamState", sourceDisplayState.short || "IDLE", sourceHealthTone);
  setHealthState("diagnosticFrameState", frameStateLabel, frameStateTone);
  setHealthState("diagnosticBaselineState", baselineStateLabel, baselineStateTone);
  setHealthState("diagnosticFrameSyncState", frameSyncLabel, frameSyncTone);
  setHealthState("diagnosticSpectrumState", spectrumStateLabel, spectrumStateTone);
  setHealthState("diagnosticAxisState", axisStateLabel, axisStateTone);
  setHealthState(
    "diagnosticQaState",
    operatorQa,
    qaTone === "review" ? "error" : qaTone === "idle" ? "neutral" : qaTone
  );
  updateSurfaceFrameState(frame, watcher, sdkLive);
  setText(
    "peakMapChip",
    candidateSpectrumPeaks.length
      ? `${candidateSpectrumPeaks.length}-peak candidate`
      : record?.peak_source === "manual_measured_wavelength_override" && Number.isFinite(Number(record?.demodulation_wavelength_nm))
      ? `manual ${Number(record.demodulation_wavelength_nm).toFixed(2)} nm`
      : status?.peak_map_status?.display?.replace("Peak map: ", "") || "target"
  );
  setText(
    "spectrumChip",
    !syncDiag.hasSpectrum
      ? "NO FRAME"
      : spectrumIsSynthetic
        ? state.displayMode === "operator" ? "local 9-FBG" : "simulated 9-FBG"
        : record?.peak_axis_type === "pixel_index"
          ? "pixel fallback"
          : candidateSpectrumPeaks.length
            ? `${candidateSpectrumPeaks.length} candidate FBG peaks`
            : `${dominantChannel} peak`
  );
  setText(
    "spectrumStatusMain",
    !syncDiag.hasSpectrum
      ? "Waiting for frame"
      : spectrumIsSynthetic
        ? state.displayMode === "operator" ? "Local 9-FBG frame" : "Simulated 9-FBG frame"
        : record?.peak_axis_type === "pixel_index"
          ? "Pixel-axis frame"
          : candidateSpectrumPeaks.length
            ? `Wavelength-axis | ${candidateSpectrumPeaks.length} candidate FBG peaks`
            : "Wavelength-axis frame"
  );
  if (spectrumDrawer) spectrumDrawer.dataset.spectrumState = syncDiag.hasSpectrum ? "available" : "missing";
  setText("selectedSpectrumTitle", `Selected peak zoom: ${dominantChannel || "--"}`);
  setText("selectedSpectrumStatus", syncDiag.hasSpectrum ? `Selected peak zoom follows ${dominantChannel}` : "Selected peak zoom unavailable for this frame");
  setText("selectedSpectrumChip", dominantChannel || "--");
  setText(
    "couplingViewBadge",
    arrayFrame?.coupling_view === "independent_ideal_response"
      ? "Ideal simulation only, not measured"
      : "Coupled wavelength response"
  );
  setText(
    "spectrumAxisNote",
    record?.peak_axis_type === "pixel_index"
      ? "Pixel index | optical intensity (counts)"
      : spectrumIsSynthetic
        ? state.displayMode === "operator" ? "Wavelength (nm) | optical intensity (counts)" : "Wavelength (nm) | optical intensity (counts) | simulated"
        : "Wavelength (nm) | reflected intensity (counts)"
  );
  const spectrumPeakProfile = String(record?.spectrum_peak_profile || "").toLowerCase();
  setText(
    "spectrumReferenceLegend",
    spectrumPeakProfile === "recorded_auto_discovered_9fbg"
      ? "auto-discovered reference"
      : spectrumPeakProfile === "current_real_9fbg_candidate"
        ? "tracked wavelength-order reference"
        : "preliminary target"
  );
  setText(
    "spectrumDiagnosticNote",
    syncDiag.status === "spectrum_missing"
      ? "Spectrum unavailable for this frame. Surface can still update from decoded channel intensity."
      : record?.peak_axis_type === "pixel_index"
      ? "Wavelength calibration missing, using pixel index fallback."
      : `Frame sync status: ${syncDiag.status}`
  );
  const globalDominantShiftPm = globalDominantPeak ? candidateShiftPm(globalDominantPeak) : Number.NaN;
  const globalEventShiftPm = Number(surfaceMetrics.global_event_absolute_shift_pm ?? displayRecord?.absolute_shift_pm);
  const globalDominantWavelength = Number(globalDominantPeak?.tracked_wavelength_nm ?? globalDominantPeak?.peak_wavelength_nm);
  const compactPeakWavelength = globalRecognitionFrame
    ? globalDominantWavelength
    : Number(displayRecord?.tracked_wavelength_nm ?? displayRecord?.peak_wavelength_nm);
  const compactSignedShift = Number(globalRecognitionFrame ? globalEventShiftPm : displayRecord?.delta_wavelength_pm);
  const compactAbsoluteShift = Number(globalRecognitionFrame ? globalEventShiftPm : displayRecord?.absolute_shift_pm);
  setCompactMetric(
    "metricIntensity",
    formatCompactNumber(compactPeakWavelength, 1),
    formatWavelength(compactPeakWavelength, 5)
  );
  setCompactMetric(
    "metricRelative",
    formatCompactNumber(compactSignedShift, 1, true),
    formatPm(compactSignedShift, 1, true)
  );
  setCompactMetric(
    "metricLoss",
    formatCompactNumber(compactAbsoluteShift, 1),
    formatPm(compactAbsoluteShift, 1)
  );

  const observedFrameChannel = globalRecognitionFrame
    ? "GLOBAL 9-FBG"
    : displayRecord?.channel_id || dominantChannel || frame?.selected_channel || state.selectedChannel;
  setText("stageChannel", observedFrameChannel || "--");
  setText("dominantChannel", measurementAvailable ? dominantChannel || "--" : "--");
  setText("selectedView", syncDiag.status || "--");
  setText("stageAttenuation", formatPm(globalRecognitionFrame ? globalEventShiftPm : displayRecord?.delta_wavelength_pm, 1, true));
  setText("stageRelative", globalRecognitionFrame ? "event wavelength shift after residual compensation" : displayRecord?.shift_direction || "--");
  setText(
    "stagePeak",
    measurementAvailable
      ? globalRecognitionFrame
        ? `${dominantChannel || "--"} ${formatWavelength(globalDominantWavelength, 5)}`
        : wavelengthSummary(record)
      : "--"
  );

  const badge = document.getElementById("levelBadge");
  const fingerScope = selectedFingerLabel();
  const selectedFingerScope =
    state.selectedFinger === "all" ? "All fingers" : fingerScope;
  badge.textContent = currentModelDisplay
    ? `${selectedFingerScope} spectrum model`
    : globalRecognitionFrame
    ? `${selectedFingerScope} spectral fingerprint`
    : isFallbackLikeFrame
      ? `${selectedFingerScope} synchronized fallback`
      : arrayMode === "simulated_array_demo"
        ? `${selectedFingerScope} coupled response`
        : `${selectedFingerScope} surface`;
  badge.className = `level-badge ${levelClass(record?.response_level)}`;
  const contactPresentation = surfaceContactPresentation({
    arrayFrame,
    surfaceMetrics,
    record,
    arrayMode,
    measurementAvailable: responseAvailable,
    heldMeasurement,
  });
  setText("scenePrimaryCaption", contactPresentation.primary);
  setText("surfaceModeNote", contactPresentation.secondary);
  setText(
    "heatmapChip",
    currentModelDisplay
      ? `${selectedFingerScope} | ${dominantChannel || "--"}`
      : globalRecognitionFrame
      ? `${selectedFingerScope} spatial proxy`
      : isFallbackLikeFrame
        ? `${selectedFingerScope} synchronized fallback`
        : arrayMode === "simulated_array_demo"
          ? `${selectedFingerScope} array response`
          : `${selectedFingerScope} surface`
  );
  setText(
    "heatmapAxisNote",
    currentModelDisplay
      ? "Approximate broad-fingertip contact region from the current 512-point spectrum model."
      : globalRecognitionFrame
      ? "Provisional FBG01-FBG09 wavelength-order spatial proxy. Verify physical P11-P33 positions with labelled point presses."
      : isFallbackLikeFrame
      ? "Single-point P22 fallback surface. 3x3 array not yet enabled."
      : contactPresentation.active
        ? "Single fingertip contact patch with dense-array mechanical and optical coupling."
        : "No active contact. Raw coupled array response is at baseline or recovery."
  );

  const response = document.getElementById("responseLevel");
  const surfaceStateLevel =
    !responseAvailable
      ? measurementAvailable && !frameSourceIsFresh(frame) ? "stale_frame" : "idle"
      : currentModelDisplay
        ? record?.response_level
      : globalRecognitionFrame
        ? responseLevelFromSurfaceValue(surfaceMetrics.surface_peak)
      : arrayMode === "simulated_array_demo"
      ? responseLevelFromSurfaceValue(surfaceMetrics.surface_peak)
      : record?.response_level;
  response.textContent = globalRecognitionFrame
    ? !responseAvailable
      ? levelLabel(surfaceStateLevel)
      : completeGlobalRecognitionFrame
      ? levelLabel(surfaceStateLevel)
      : "global spectrum incomplete"
    : levelLabel(surfaceStateLevel);
  response.style.color = levelClass(surfaceStateLevel) === "hard" ? "#bf4f49" : levelClass(surfaceStateLevel) === "light" ? "#119b69" : "#0b91d2";
  setText("responseText", measurementAvailable ? liveResponseText(record, watcher, sdkLive) : "No wavelength frame. Start Live, Watch, ingest, or select a response scenario.");
  setText("diagnosticCurrentChannel", measurementAvailable ? observedFrameChannel || "--" : "--");
  setText(
    "diagnosticFrameSource",
    !measurementAvailable
      ? "--"
      : arrayMode === "simulated_array_demo"
        ? isRecordedDemoSpectrum(displayRecord, arrayFrame)
          ? "recorded spectrum"
          : "simulated response"
        : displayRecord?.source || "wavelength frame"
  );
  const diagnosticFrameSourceElement = document.getElementById("diagnosticFrameSource");
  if (diagnosticFrameSourceElement) {
    diagnosticFrameSourceElement.title = diagnosticFrameSourceElement.textContent || "";
  }
  setText(
    "baselineIntensity",
    formatWavelength(
      globalRecognitionFrame
        ? globalDominantPeak?.candidate_reference_wavelength_nm
        : displayRecord?.baseline_wavelength_nm,
      5
    )
  );
  setText(
    "currentIntensity",
    formatWavelength(
      globalRecognitionFrame
        ? globalDominantWavelength
        : displayRecord?.tracked_wavelength_nm ?? displayRecord?.peak_wavelength_nm,
      5
    )
  );
  setText("deltaIntensity", formatNumber(displayRecord?.cross_correlation_coefficient, 3));
  setText("relativeIntensity", formatPm(globalRecognitionFrame ? globalEventShiftPm : displayRecord?.delta_wavelength_pm, 2, true));
  setText(
    "attenuationRatio",
    formatNumber(
      globalRecognitionFrame
        ? normalizedSurfaceResponseRatio(surfaceMetrics, displayRecord)
        : displayRecord?.wavelength_shift_response_ratio,
      3
    )
  );
  setText("intensityLoss", formatPm(displayRecord?.wavelength_estimator_disagreement_pm, 2));
  setText("responseBasis", displayRecord?.wavelength_tracking_method || displayRecord?.response_basis || "--");
  setText("p22TargetWavelength", formatWavelength(record?.target_wavelength_nm, 4));
  setText("p22MeasuredWavelength", formatWavelength(record?.measured_wavelength_nm, 4));
  setText("p22DemodulationWavelength", formatWavelength(record?.demodulation_wavelength_nm, 4));
  setText("p22MeasuredSource", record?.measured_wavelength_source || "--");
  setText("p22MeasuredStatus", record?.measured_wavelength_status || "--");
  setText("p22PeakSource", record?.peak_source || "--");
  ["p22MeasuredSource", "p22MeasuredStatus", "p22PeakSource"].forEach((id) => {
    const element = document.getElementById(id);
    if (element) element.title = element.textContent || "";
  });
  setText(
    "surfaceResponseType",
    globalRecognitionFrame
      ? "Global nine-FBG mixed spectral fingerprint"
      : isFallbackLikeFrame
        ? "Single-point P22 wavelength-shift fallback"
        : "Raw coupled Bragg wavelength-shift surface"
  );
  setText(
    "surfaceCouplingPath",
    globalRecognitionFrame
      ? "not spatially mapped"
      : isFallbackLikeFrame
        ? "P22 fallback"
        : arrayFrame?.possible_cross_fiber_coupling || arrayFrame?.possible_same_fiber_coupling
          ? "coupled response"
          : "not resolved"
  );
  setText("surfaceForceDecoupled", displayRecord?.temperature_strain_decoupled ? "Yes" : "No");
  setText("surfaceLocalMapStatus", arrayFrame?.local_response_estimate_available ? "simulation-only estimate" : "Not available yet");
  setText("attenuationPercent", formatPm(globalRecognitionFrame ? globalEventShiftPm : displayRecord?.absolute_shift_pm, 1));
  const globalNoisePm = Number(globalDominantPeak?.candidate_baseline_noise_pm);
  setText("baselineNoise", formatPm(globalRecognitionFrame ? globalNoisePm : displayRecord?.baseline_wavelength_noise_pm, 2));
  setText("baselineStatus", displayRecord?.baseline_status || "--");
  setText("qaStatus", measurementAvailable ? record?.qa_status || "--" : "--");
  setText("peakAxisType", record?.peak_axis_type || "--");
  const liveFrameCount =
    sdkLive?.active && sdkLive?.frame_count !== null && sdkLive?.frame_count !== undefined
      ? sdkLive.frame_count
      : record?.fast_record_frame_count;
  setText("datFrameCount", liveFrameCount === null || liveFrameCount === undefined ? "--" : `${Math.round(Number(liveFrameCount)).toLocaleString()}`);
  setText("polarityStatus", record?.shift_direction || "--");
  setText("integrationMs", record?.integration_ms === null || record?.integration_ms === undefined ? "--" : `${formatNumber(record.integration_ms, 1)} ms`);
  setText("surfaceFrameId", !measurementAvailable || syncDiag.surfaceId === undefined || syncDiag.surfaceId === null ? "--" : String(syncDiag.surfaceId));
  setText("spectrumFrameId", !measurementAvailable || syncDiag.spectrumId === undefined || syncDiag.spectrumId === null ? "--" : String(syncDiag.spectrumId));
  setText("traceFrameId", !measurementAvailable || syncDiag.traceId === undefined || syncDiag.traceId === null ? "--" : String(syncDiag.traceId));
  setText("frameSyncStatus", measurementAvailable ? syncDiag.status : "no frame");
  setText("arrayFrameSyncMirror", syncDiag.status);
  setText("peakAxisTypeMirror", record?.peak_axis_type || (spectrumIsSynthetic ? "wavelength grid" : "--"));
  setText(
    "lastUpdateTimestamp",
    measurementAvailable && syncDiag.timestamp
      ? new Date(Number(syncDiag.timestamp) * (Number(syncDiag.timestamp) < 10000000000 ? 1000 : 1)).toLocaleTimeString()
      : "--"
  );
  const displaySurfaceResponseLevel = !measurementAvailable
    ? "idle"
    : !responseAvailable
      ? frameSourceIsFresh(frame) ? "response unavailable" : "stale frame"
    : globalRecognitionFrame && !completeGlobalRecognitionFrame
      ? "global spectrum incomplete"
      : contactPresentation.active
        ? "contact"
        : "no_contact";
  setText("surfaceResponseLevel", displaySurfaceResponseLevel);
  setText("diagnosticSurfaceResponseLevel", displaySurfaceResponseLevel);
  const surfacePeakValue = Number(
    globalRecognitionFrame
      ? surfaceMetrics.surface_peak
      : surfaceMetrics.surface_peak ?? record?.wavelength_shift_response_ratio ?? record?.response_value
  );
  const surfaceHasActiveResponse =
    responseAvailable &&
    Number.isFinite(surfacePeakValue) &&
    surfacePeakValue >= RESPONSE_BAND_THRESHOLDS.noContactMax;
  const rawResponseTone = levelClass(record?.response_level);
  const surfaceResponseTone = !responseAvailable
    ? "idle"
    : rawResponseTone === "warning"
      ? "warning"
      : levelClass(responseLevelFromSurfaceValue(surfacePeakValue)) || "idle";
  const operatorSummaryCard = document.querySelector(".operator-summary-card");
  if (operatorSummaryCard) operatorSummaryCard.dataset.responseTone = surfaceResponseTone;
  const surfaceResponseLevelElement = document.getElementById("surfaceResponseLevel");
  if (surfaceResponseLevelElement) surfaceResponseLevelElement.dataset.responseTone = surfaceResponseTone;
  setText(
    "surfaceText",
    !measurementAvailable
      ? "No wavelength frame"
      : !responseAvailable
        ? frameSourceIsFresh(frame) ? "Response unavailable" : "Stale frame | response disabled"
      : heldMeasurement
        ? "Last frame held"
        : globalRecognitionFrame
          ? completeGlobalRecognitionFrame
            ? "Provisional global spatial response proxy"
            : "Incomplete global FBG spectrum"
        : isFallbackLikeFrame
      ? "P22 fallback"
      : arrayMode === "simulated_array_demo"
        ? isRecordedDemoSpectrum(displayRecord, arrayFrame)
          ? "Recorded spectrum response proxy"
          : "Simulated 3x3 coupled response"
        : "Coupled Bragg wavelength-shift response"
  );
  setText(
    "surfaceActiveChannels",
    !responseAvailable
      ? "--"
      : globalRecognitionFrame
        ? String(surfaceMetrics.responding_channel_count ?? 0)
      : isFallbackLikeFrame
      ? "1"
      : arrayMode === "simulated_array_demo"
        ? String(surfaceMetrics.responding_channel_count ?? surfaceMetrics.active_channel_count ?? "--")
        : surfaceMetrics.active_channel_count ?? "--"
  );
  const operatorPosition = surfaceHasActiveResponse
    ? globalRecognitionFrame
      ? surfaceMetrics.dominant_channel || "--"
      : dominantChannel || surfaceMetrics.dominant_channel || "--"
    : "--";
  setText("surfaceDominantChannel", operatorPosition);
  const dominantChannelRecord = (arrayFrame?.channels || []).find((item) => item.channel_id === (surfaceMetrics.dominant_channel || dominantChannel));
  const surfacePeakShiftPm = Number(
    (globalRecognitionFrame ? globalEventShiftPm : null) ??
      arrayFrame?.peak_wavelength_shift_pm ??
      dominantChannelRecord?.absolute_shift_pm ??
      record?.absolute_shift_pm
  );
  setText(
    "surfacePeakLabel",
    currentRuntimeActive
      ? "Optical response"
      : currentModelDisplay
        ? "Visual response"
        : "Peak |shift|"
  );
  const surfacePeakElement = document.getElementById("surfacePeak");
  if (surfacePeakElement) {
    surfacePeakElement.title = currentRuntimeActive
      ? "Digital-twin response normalized from the optical-only continuous Fz estimate"
      : currentModelDisplay
        ? "Trained model visual response proxy; not calibrated force"
        : "Peak absolute Bragg wavelength shift";
  }
  setText(
    "surfacePeak",
    responseAvailable
      ? currentModelDisplay
        ? formatPercent(surfacePeakValue, 1)
        : formatPm(surfacePeakShiftPm, 1)
      : "--"
  );
  setText("surfaceMean", responseAvailable ? formatPercent(surfaceMetrics.surface_mean, 1) : "--");
  setText("surfaceActiveArea", responseAvailable ? formatPercent(surfaceMetrics.surface_area_active, 1) : "--");
  setText(
    "surfaceCenter",
    surfaceHasActiveResponse
      ? `${formatNumber(surfaceMetrics.surface_centroid_x, 2)}, ${formatNumber(surfaceMetrics.surface_centroid_y, 2)}`
      : "--"
  );
  setText("surfaceSpread", surfaceHasActiveResponse ? formatNumber(surfaceMetrics.surface_spread, 3) : "--");
  setText("surfacePeakDiagnostic", measurementAvailable ? formatPm(surfacePeakShiftPm, 1) : "--");
  setText("surfaceActiveAreaDiagnostic", measurementAvailable ? formatPercent(surfaceMetrics.surface_area_active, 1) : "--");
  setText(
    "surfaceCenterDiagnostic",
    measurementAvailable
      ? `${formatNumber(surfaceMetrics.surface_centroid_x, 2)}, ${formatNumber(surfaceMetrics.surface_centroid_y, 2)}`
      : "--"
  );
  setText(
    "surfaceDominantChannelDiagnostic",
    measurementAvailable
      ? globalRecognitionFrame
        ? surfaceMetrics.dominant_channel || "--"
        : surfaceMetrics.dominant_channel || dominantChannel || "--"
      : "--"
  );
  setText("surfaceEntropy", measurementAvailable && !globalRecognitionFrame ? formatNumber(surfaceMetrics.surface_entropy, 3) : "--");
  setText("surfaceLRAsymmetry", measurementAvailable && !globalRecognitionFrame ? formatNumber(surfaceMetrics.left_right_asymmetry, 3) : "--");
  setText("surfaceTBAsymmetry", measurementAvailable && !globalRecognitionFrame ? formatNumber(surfaceMetrics.top_bottom_asymmetry, 3) : "--");
  const surfaceQualityRaw = String(surfaceMetrics.quality_status || "--");
  const surfaceQualityDisplay = arrayMode === "simulated_array_demo" ? "simulated" : surfaceQualityRaw;
  const surfaceQualityKey = surfaceQualityDisplay.toLowerCase();
  const surfaceQualityTone = surfaceQualityKey.includes("invalid") || surfaceQualityKey.includes("error")
    ? "error"
    : surfaceQualityKey.includes("warning")
      ? "warning"
      : surfaceQualityKey.includes("simulated")
        ? "demo"
        : surfaceQualityDisplay === "--"
          ? "neutral"
          : "ok";
  const surfaceQualityLabel = arrayMode === "simulated_array_demo"
    ? state.displayMode === "operator" ? "LOCAL" : "SIMULATED"
    : operatorQaLabel(surfaceQualityDisplay);
  setHealthState("surfaceQualityStatus", surfaceQualityLabel, surfaceQualityTone);
  const surfaceQualityElement = document.getElementById("surfaceQualityStatus");
  if (surfaceQualityElement) {
    surfaceQualityElement.title = surfaceQualityRaw !== surfaceQualityDisplay
      ? `Raw surface quality: ${surfaceQualityRaw}`
      : "";
  }
  setText(
    "couplingStatus",
    currentModelDisplay
      ? "trained full-spectrum fingerprint"
      : globalRecognitionFrame
      ? "global fingerprint; provisional spatial proxy"
      : surfaceMetrics.coupling_status || arrayFrame?.coupling_status || record?.coupling_status || "--"
  );
  setText("primaryObservedChannel", surfaceMetrics.primary_observed_channel || "--");
  setText(
    "numChangedPeaks",
    measurementAvailable
      ? candidateSpectrumPeaks.length || surfaceMetrics.num_changed_peaks || "--"
      : "--"
  );
  setText("numChangedPeaksDiagnostic", measurementAvailable ? surfaceMetrics.num_changed_peaks ?? "--" : "--");
  setText("coupledPeakAttenuationSum", formatNumber(surfaceMetrics.coupled_shift_response_sum, 3));
  setText(
    "eventInterpretation",
    measurementAvailable
      ? currentModelDisplay
        ? "model-driven manual fingertip position and approximate response level"
      : globalRecognitionFrame
        ? "global wavelength-order candidates; no spatial contact inference"
        : surfaceMetrics.event_interpretation || "--"
      : "idle / no wavelength frame"
  );
  const secondaryCouplingPaths = Array.isArray(surfaceMetrics.secondary_coupling_paths)
    ? surfaceMetrics.secondary_coupling_paths
    : Array.isArray(arrayFrame?.secondary_coupling_paths)
      ? arrayFrame.secondary_coupling_paths
      : [];
  const couplingPathPreview = secondaryCouplingPaths.length
    ? `${secondaryCouplingPaths.slice(0, 2).join("; ")}${secondaryCouplingPaths.length > 2 ? ` (+${secondaryCouplingPaths.length - 2})` : ""}`
    : Number(surfaceMetrics.num_changed_peaks) > 1
      ? "direct coupled neighbors"
      : "none detected";
  setText("surfaceCouplingPathsDiagnostic", measurementAvailable ? couplingPathPreview : "--");
  const couplingPathElement = document.getElementById("surfaceCouplingPathsDiagnostic");
  if (couplingPathElement) {
    couplingPathElement.title = measurementAvailable && secondaryCouplingPaths.length
      ? secondaryCouplingPaths.join("; ")
      : "";
  }
  setText(
    "surfaceMaxCouplingHopDiagnostic",
    measurementAvailable
      ? String(surfaceMetrics.max_coupling_hop_depth ?? arrayFrame?.max_coupling_hop_depth ?? 0)
      : "--"
  );
  updateOperatorFootprint(arrayFrame, record, surfaceMetrics, arrayMode, responseAvailable, heldMeasurement, frame);
  const secondaryObserved = Array.isArray(surfaceMetrics.secondary_observed_channels)
    ? surfaceMetrics.secondary_observed_channels
    : Array.isArray(surfaceMetrics.secondary_changed_channels)
      ? surfaceMetrics.secondary_changed_channels
      : [];
  setText("secondaryChangedChannels", secondaryObserved.join(", ") || "--");
  const secondaryChannelsElement = document.getElementById("secondaryChangedChannels");
  if (secondaryChannelsElement) {
    secondaryChannelsElement.title = secondaryObserved.join(", ");
  }
  setText(
    "responseText",
    globalRecognitionFrame
      ? liveResponseText(record, watcher, sdkLive)
      : Number(surfaceMetrics.num_changed_peaks) > 1
      ? "Coupled response"
      : liveResponseText(record, watcher, sdkLive)
  );
  updateArrayDiagnostics(arrayFrame, record, measurementAvailable);
  setText(
    "trajectoryNote",
    globalRecognitionFrame
      ? "trajectory disabled until physical P11-P33 mapping is approved"
      : isFallbackLikeFrame
      ? "trajectory limited by single-point fallback"
      : arrayMode === "simulated_array_demo"
        ? isRecordedDemoSpectrum(displayRecord, arrayFrame)
          ? "recorded-reference contact center trajectory"
          : "simulated contact center trajectory"
        : "contact center trajectory from enabled wavelength-shift channels"
  );

  setText(
    "senseProcess",
    senseControl?.sense_window_found
      ? "running"
      : status?.sense_process?.running === true
        ? "running"
        : status?.sense_process?.running === false
          ? "not found"
          : "--"
  );
  const latestFile = watcher?.last_file || status?.latest_export_file;
  setText(
    "latestExport",
    sdkLive?.active
      ? `${sdkLive?.source || "direct SDK"}`
      : latestFile
        ? latestFile.split(/[\\/]/).pop()
        : "none"
  );
  setText("bufferedRecords", status?.buffered_records ?? "--");
  setText("channelsSeen", (status?.channels_seen || []).join(", ") || "none");
  updateWatcherUI(watcher, sdkLive);
  updateOperatorStreamSummary(frame, watcher, sdkLive, operatorQa);
  updateDemoReadout(record);
  updatePx6dPanel(frame || {});

  setChartTargets(trace, record);
  updateThree(record, arrayFrame);
  updateChannelSelectorLabels(arrayFrame?.channels || frame?.channel_grid || []);
  updateChannelGrid(arrayFrame?.channels || frame?.channel_grid || [], frame?.selected_channel);
  // Labels commit atomically per source frame, while charts and geometry ease
  // toward the same frame targets. Synchronous redraw and normal recomputation
  // here caused the post-review live UI regression.
  state.chartsNeedRefresh = true;
  state.threeNeedsRefresh = true;
  requestThreeAnimation();
}

function normalizeCanonicalVisualizationFrame(frame, contract) {
  const rawRecord = frame.latest || {};
  const prediction = contract.prediction || null;
  const runtimeModelSource = currentRuntimeModelSource(frame, contract);
  const responseAllowed = contract.response_allowed === true;
  const responseState = responseAllowed && contract.response_state === "contact"
    ? "contact"
    : "no_contact";
  const surfaceActive = responseState === "contact";
  const positionId = responseState === "contact"
    ? String(contract?.position?.display_label || contract?.surface?.position_id || "") || null
    : null;
  const suppliedGrid = Array.isArray(contract?.surface?.surface_grid)
    ? contract.surface.surface_grid
    : [];
  const surfaceGrid = ARRAY_DISPLAY_ROWS.map((row, rowIndex) =>
    row.map((_channelId, columnIndex) => {
      const value = finiteNumberOrNull(suppliedGrid?.[rowIndex]?.[columnIndex]);
      return surfaceActive
        ? Math.max(0, Math.min(1, value ?? 0))
        : 0;
    })
  );
  const metricSource = contract?.surface?.surface_metrics || {};
  const inferredDominantChannel = surfaceActive
    ? String(
        metricSource.inferred_dominant_channel ||
        metricSource.dominant_channel ||
        contract?.surface?.inferred_dominant_channel ||
        ""
      ) || null
    : null;
  const observedResponse = contract?.surface?.observed_coupled_spectral_response || {};
  const observedDominantChannel = String(
    metricSource.observed_dominant_channel ||
    contract?.surface?.observed_dominant_channel ||
    observedResponse?.dominant_channel ||
    ""
  ) || null;
  const rawProbabilityGrid = Array.isArray(
    contract?.surface?.inferred_contact_probability_grid
  )
    ? contract.surface.inferred_contact_probability_grid
    : [];
  const suppliedProbabilityGrid = ARRAY_DISPLAY_ROWS.map((row, rowIndex) =>
    row.map((_channelId, columnIndex) => {
      const value = finiteNumberOrNull(rawProbabilityGrid?.[rowIndex]?.[columnIndex]);
      return surfaceActive ? Math.max(0, Math.min(1, value ?? 0)) : 0;
    })
  );
  const observedGrid = Array.isArray(observedResponse?.response_grid)
    ? observedResponse.response_grid
    : [];
  const flatGrid = surfaceGrid.flat();
  const surfacePeak = surfaceActive
    ? Math.max(0, Math.min(1, finiteNumberOrNull(metricSource.surface_peak) ?? Math.max(...flatGrid)))
    : 0;
  const surfaceMean = surfaceActive
    ? Math.max(0, finiteNumberOrNull(metricSource.surface_mean) ?? (flatGrid.reduce((sum, value) => sum + value, 0) / flatGrid.length))
    : 0;
  const respondingChannelIds = surfaceActive
    ? ARRAY_DISPLAY_ORDER.filter((channelId) => {
        const coordinate = ARRAY_CHANNEL_COORDS[channelId];
        const rowIndex = 1 - Number(coordinate?.y || 0);
        const columnIndex = Number(coordinate?.x || 0) + 1;
        return (surfaceGrid?.[rowIndex]?.[columnIndex] || 0) >= 0.055;
      })
    : [];
  const activeArea = surfaceActive
    ? Math.max(
        0,
        Math.min(
          1,
          finiteNumberOrNull(metricSource.surface_area_active) ??
            (respondingChannelIds.length / ARRAY_DISPLAY_ORDER.length)
        )
      )
    : 0;
  const displayForceN = finiteNumberOrNull(contract?.force?.display_n);
  const surfaceMetrics = {
    ...metricSource,
    surface_peak: surfacePeak,
    surface_mean: surfaceMean,
    surface_area_active: activeArea,
    surface_area_active_percent: activeArea * 100,
    surface_centroid_x: surfaceActive
      ? finiteNumberOrNull(metricSource.surface_centroid_x) ?? 0
      : 0,
    surface_centroid_y: surfaceActive
      ? finiteNumberOrNull(metricSource.surface_centroid_y) ?? 0
      : 0,
    surface_spread: surfaceActive
      ? finiteNumberOrNull(metricSource.surface_spread) ?? 0.24
      : 0,
    dominant_channel: inferredDominantChannel,
    semantic_position_id: positionId,
    inferred_dominant_channel: inferredDominantChannel,
    observed_dominant_channel: observedDominantChannel,
    enabled_channel_count: ARRAY_DISPLAY_ORDER.length,
    responding_channel_count: respondingChannelIds.length,
    responding_channel_ids: respondingChannelIds,
    coupling_status: String(metricSource.coupling_status || "optical_model_no_contact"),
    quality_status: responseAllowed
      ? "current_runtime_model"
      : String(contract.response_block_reason || "current_runtime_model_not_ready"),
    recognition_source: runtimeModelSource,
    estimated_force_fz_n: displayForceN,
  };
  const channels = ARRAY_DISPLAY_ROWS.flatMap((row, rowIndex) =>
    row.map((channelId, columnIndex) => {
      const coordinate = ARRAY_CHANNEL_COORDS[channelId] || { x: null, y: null };
      const responseValue = surfaceGrid[rowIndex][columnIndex];
      const inferredProbability = Math.max(
        0,
        Math.min(
          1,
          finiteNumberOrNull(suppliedProbabilityGrid?.[rowIndex]?.[columnIndex]) ?? 0
        )
      );
      const observedResponseRatio = Math.max(
        0,
        Math.min(
          1,
          finiteNumberOrNull(observedGrid?.[rowIndex]?.[columnIndex]) ?? 0
        )
      );
      return {
        channel_id: channelId,
        provisional_channel_id: channelId,
        x: coordinate.x,
        y: coordinate.y,
        enabled: true,
        valid: true,
        response_enabled: surfaceActive,
        wavelength_shift_response_ratio: responseValue,
        response_value: responseValue,
        inferred_contact_probability: inferredProbability,
        observed_coupled_response_ratio: observedResponseRatio,
        qa_status: surfaceActive ? "current_runtime_model_response" : "current_runtime_model_waiting",
        spectral_mapping_status: "provisional_fbg_mapping_coupled_observation",
      };
    })
  );
  const rawQaFlags = Array.isArray(rawRecord.qa_flags) ? rawRecord.qa_flags : [];
  const diagnosticOnlyQaFlags = new Set(["wavelength_estimator_disagreement"]);
  const diagnosticOnlyRawQaFlags = rawQaFlags.filter((flag) =>
    diagnosticOnlyQaFlags.has(String(flag))
  );
  const operatorRawQaFlags = rawQaFlags.filter((flag) =>
    !diagnosticOnlyQaFlags.has(String(flag))
  );
  const blockReason = responseAllowed ? null : String(
    contract.response_block_reason || "current_runtime_model_not_ready"
  );
  const qaFlags = [...new Set([
    ...operatorRawQaFlags,
    ...(blockReason ? [blockReason] : []),
  ])];
  const latest = {
    ...rawRecord,
    operator_visualization_frame: contract,
    visualization_contract_version: OPERATOR_VISUALIZATION_CONTRACT_VERSION,
    recognition_scope: GLOBAL_RECOGNITION_SCOPE,
    response_allowed: responseAllowed,
    response_block_reason: blockReason,
    response_level: responseState,
    response_value: surfacePeak,
    wavelength_shift_response_ratio: surfacePeak,
    dominant_channel: inferredDominantChannel,
    provisional_dominant_channel: observedDominantChannel,
    model_position_id: positionId,
    active_spectral_prediction: prediction,
    active_spectral_model_source: runtimeModelSource,
    active_spectral_model_status: String(contract.prediction_status || "unavailable"),
    active_spectral_model_loaded: frame?.runtime_model?.loaded === true,
    active_spectral_model_progress: frame?.runtime_model?.warming || null,
    estimated_force_fz_n: displayForceN,
    optical_force_estimate: responseAllowed ? contract.force || null : null,
    baseline_status: responseAllowed
      ? "current_runtime_baseline_ready"
      : "current_runtime_baseline_required",
    qa_status: responseAllowed
      ? String(contract?.quality?.raw_qa_status || "model_baseline_ready")
      : "current_runtime_model_not_ready",
    qa_flags: qaFlags,
    raw_qa_flags: rawQaFlags,
    diagnostic_only_qa_flags: diagnosticOnlyRawQaFlags,
    model_review_needed: contract?.quality?.review_needed === true,
  };
  const sync = contract.sync || {};
  const frameId = contract.frame_id ?? frame.frame_id ?? rawRecord.frame_id ?? null;
  const arrayFrame = {
    mode: responseAllowed
      ? "all_source_optical_force_position"
      : "all_source_optical_force_waiting",
    frame_id: frameId,
    timestamp: contract.timestamp ?? frame.timestamp ?? rawRecord.timestamp ?? null,
    surface_frame_id: sync.surface_frame_id ?? frameId,
    spectrum_frame_id: sync.spectrum_frame_id ?? frameId,
    trace_frame_id: sync.trace_frame_id ?? frameId,
    summary_frame_id: sync.summary_frame_id ?? frameId,
    force_frame_id: sync.force_frame_id ?? frameId,
    frame_sync_status: String(sync.status || "synced"),
    operator_visualization_frame: contract,
    surface_metrics: surfaceMetrics,
    surface_grid: surfaceGrid,
    inferred_contact_probability_grid: suppliedProbabilityGrid,
    observed_coupled_spectral_response: observedResponse,
    channels,
    coupling_status: surfaceMetrics.coupling_status,
    physical_channel_mapping_final: false,
    trained_position_label_mapping_available: responseAllowed,
    surface_visualization_semantics: String(
      contract?.surface?.map_semantics ||
      "runtime_position_posterior_ema_scaled_by_optical_force_not_measured_pressure"
    ),
    raw_inferred_contact_probabilities: surfaceActive
      ? contract?.surface?.raw_inferred_contact_probabilities || {}
      : {},
    smoothed_inferred_contact_probabilities: surfaceActive
      ? contract?.surface?.smoothed_inferred_contact_probabilities || {}
      : {},
    local_response_estimate_available: surfaceActive,
    provisional_spatial_proxy_available: false,
    response_allowed: responseAllowed,
    response_block_reason: blockReason,
    active_spectral_prediction: prediction,
    active_spectral_model_source: runtimeModelSource,
    estimated_force_fz_n: displayForceN,
    optical_force_estimate: responseAllowed ? contract.force || null : null,
  };
  return {
    ...frame,
    selected_channel: null,
    operator_visualization_frame: contract,
    visualization_contract_version: OPERATOR_VISUALIZATION_CONTRACT_VERSION,
    model_assisted_display_allowed: responseAllowed,
    model_assisted_display_block_reason: blockReason,
    active_spectral_prediction: prediction,
    active_spectral_model_source: runtimeModelSource,
    active_spectral_model_status: String(contract.prediction_status || "unavailable"),
    latest,
    trace: appendCurrentRuntimeTrace(rawRecord, contract),
    channel_grid: channels,
    array_frame: arrayFrame,
    surface_grid: surfaceGrid,
    surface_metrics: surfaceMetrics,
  };
}

function normalizeGlobalSpectrumFrame(frame) {
  if (frame?.scope !== GLOBAL_RECOGNITION_SCOPE) return frame;
  const canonicalContract = operatorVisualizationContract(frame, frame?.latest);
  if (canonicalContract) {
    return normalizeCanonicalVisualizationFrame(frame, canonicalContract);
  }
  return atomicNeutralFrame({
    ...frame,
    active_spectral_model_source: CURRENT_RUNTIME_MODEL_SOURCE,
    active_spectral_model_status:
      frame?.runtime_model?.loaded === true
        ? "waiting_for_canonical_frame"
        : "current_runtime_unavailable",
    model_assisted_display_allowed: false,
    model_assisted_display_block_reason: "canonical_visualization_frame_required",
  });
}

function sourceFrameRenderKey(rawFrame) {
  const latest = rawFrame?.latest || {};
  const visualization = rawFrame?.operator_visualization_frame || {};
  const frameId = latest.frame_id ?? rawFrame?.frame_id;
  const timestamp = latest.timestamp ?? rawFrame?.timestamp;
  const source = latest.source ?? rawFrame?.source ?? "unknown";
  const sourceSession =
    rawFrame?.sdk_live?.acquisition_session_id ??
    rawFrame?.export_watcher?.acquisition_session_id ??
    "";
  const runtimeUniqueFrame =
    rawFrame?.active_spectral_prediction?.unique_frame_count ??
    rawFrame?.runtime_prediction?.unique_frame_count ??
    latest?.active_spectral_prediction?.unique_frame_count ??
    latest?.runtime_prediction?.unique_frame_count;
  const gateSignature = [
    rawFrame?.source_fresh === true ? "fresh" : "stale",
    rawFrame?.operator_display_valid === true ? "display" : "blocked",
    String(rawFrame?.model_assisted_display_block_reason || ""),
    visualization?.prediction_ready === true ? "prediction-ready" : "prediction-waiting",
    visualization?.response_allowed === true ? "response-allowed" : "response-blocked",
    String(visualization?.response_state || ""),
    String(visualization?.response_block_reason || ""),
  ].join(":");
  if (
    frameId == null &&
    timestamp == null &&
    runtimeUniqueFrame == null &&
    !gateSignature
  ) return null;
  return [
    source,
    String(sourceSession),
    String(frameId ?? ""),
    String(timestamp ?? ""),
    String(runtimeUniqueFrame ?? ""),
    gateSignature,
  ].join("|");
}

function invalidateLiveSourceProbe() {
  state.liveSourceProbeController?.abort();
  state.liveSourceProbeController = null;
  state.liveSourceProbeInFlight = false;
}

async function reconcileLiveSourceState({ force = false } = {}) {
  if (!state.pageVisible) return false;
  const now = performance.now();
  if (!force && now < state.nextLiveSourceProbeAt) {
    return state.exportWatchActive || state.sdkLiveActive;
  }
  if (state.liveSourceProbeInFlight) {
    return state.exportWatchActive || state.sdkLiveActive;
  }

  state.nextLiveSourceProbeAt = now + LIVE_SOURCE_PROBE_INTERVAL_MS;
  state.liveSourceProbeInFlight = true;
  const controller = new AbortController();
  state.liveSourceProbeController = controller;
  try {
    const payload = await requestJSON(
      "/api/frame?trace_limit=1&include_spectrum=false",
      { cache: "no-store" },
      { timeoutMs: 4000, signal: controller.signal }
    );
    updateWatcherUI(payload?.export_watcher || {}, payload?.sdk_live || {});
    return state.exportWatchActive || state.sdkLiveActive;
  } catch (error) {
    if (error?.name !== "AbortError") {
      console.warn("[live source probe]", error);
    }
    return state.exportWatchActive || state.sdkLiveActive;
  } finally {
    if (state.liveSourceProbeController === controller) {
      state.liveSourceProbeController = null;
    }
    state.liveSourceProbeInFlight = false;
  }
}

async function fetchFrame({ force = false } = {}) {
  if (!state.pageVisible) return;
  if (state.paused && !force) return;
  const now = performance.now();
  if (!force && state.demoModeActive && !state.arrayDemoActive) return;
  if (!force && !state.arrayDemoActive && now < state.nextLiveModelPollAt) return;
  if (
    !force &&
    state.arrayDemoActive &&
    state.arrayDemoScenario &&
    now < state.arrayDemoNextStepAt
  ) {
    return;
  }
  if (!force && state.arrayDemoActive && state.arrayDemoActionComplete) {
    if (state.arrayDemoPlaybackMode !== "loop") return;
    const nextCycleAt = state.arrayDemoActionCompletedAt + DEMO_ARRAY_LOOP_INTERVAL_MS;
    if (now >= nextCycleAt) {
      state.arrayDemoStep = 0;
      state.arrayDemoActionComplete = false;
      state.arrayDemoCycleStartedAt = now;
      state.arrayDemoActionCompletedAt = 0;
      state.trajectoryHistory = [];
      setDemoStatus("5 s loop", "auto");
    }
  }
  if (state.frameRequestInFlight) {
    // Keep at most one expensive spectrum/model request in flight. Commands may
    // queue one forced refresh, while routine polling simply skips this tick.
    if (force) state.forcedFrameRequestQueued = true;
    return;
  }
  if (!state.arrayDemoActive) {
    state.nextLiveModelPollAt = now + LIVE_MODEL_POLL_INTERVAL_MS;
  }
  state.frameRequestInFlight = true;
  const requestController = new AbortController();
  state.frameRequestController = requestController;
  const requestSequence = ++state.frameRequestSequence;
  const requestEpoch = state.frameModeEpoch;
  const commitFrame = (frame) => {
    if (requestEpoch !== state.frameModeEpoch) return false;
    if (requestSequence < state.lastCommittedFrameRequest) return false;
    if (state.paused && !force) return false;
    state.lastCommittedFrameRequest = requestSequence;
    state.frame = frame;
    updateUI(frame);
    return true;
  };
  try {
    if (state.arrayDemoActive && state.arrayDemoScenario) {
      const stepCount = ARRAY_SLIDE_STEPS[state.arrayDemoScenario] || 1;
      const actionFinished = state.arrayDemoStep >= stepCount;
      const frameScenario = actionFinished ? "no_contact" : state.arrayDemoScenario;
      const step = actionFinished ? 0 : state.arrayDemoStep;
      const data = await requestJSON(
        `/api/array_demo/frame?scenario=${encodeURIComponent(frameScenario)}&step=${step}&coupling_view=${encodeURIComponent(state.couplingView)}`,
        { cache: "no-store" },
        { timeoutMs: 7000, signal: requestController.signal }
      );
      if (data?.ok === false || !data?.array_frame) {
        throw new Error(data?.reason || data?.status || "Simulated response is unavailable.");
      }
      if (requestEpoch !== state.frameModeEpoch) return;
      if (!actionFinished) {
        state.arrayDemoStep += 1;
      }
      const frame = frameFromArrayDemo(data.array_frame);
      commitFrame(frame);
      if (actionFinished) {
        if (!state.arrayDemoActionComplete) {
          state.arrayDemoActionCompletedAt = performance.now();
        }
        state.arrayDemoActionComplete = true;
        state.arrayDemoNextStepAt = state.arrayDemoPlaybackMode === "loop"
          ? performance.now() + demoArrayStepIntervalMs()
          : Number.POSITIVE_INFINITY;
        setDemoStatus(
          state.arrayDemoPlaybackMode === "loop" ? "released - baseline running" : "complete - released",
          "ready"
        );
        updateDemoControls();
      } else {
        state.arrayDemoNextStepAt = performance.now() + demoArrayStepIntervalMs();
      }
      return;
    }
    if (!state.dataStreamActive && !state.demoModeActive && !state.exportWatchActive && !state.sdkLiveActive && !state.liveRequested) {
      const sourceActive = await reconcileLiveSourceState({ force });
      if (!sourceActive) return;
    }
    const traceLimit = state.demoModeActive ? DEMO_TRACE_WINDOW_POINTS : TRACE_WINDOW_POINTS;
    const rawFrame = await requestJSON(
      `/api/global_spectrum_frame?trace_limit=${traceLimit}&include_spectrum=true`,
      { cache: "no-store" },
      { timeoutMs: 7000, signal: requestController.signal }
    );
    const renderKey = sourceFrameRenderKey(rawFrame);
    if (!force && renderKey && renderKey === state.lastRenderedSourceFrameKey) {
      return;
    }
    const frame = normalizeGlobalSpectrumFrame(rawFrame);
    if (commitFrame(frame)) {
      state.lastRenderedSourceFrameKey = renderKey;
      updateCaptureReadiness();
    }
  } catch (error) {
    if (error?.name === "AbortError" && requestController.signal.aborted) return;
    if (requestEpoch !== state.frameModeEpoch || requestSequence < state.lastCommittedFrameRequest) return;
    updateOperatorStreamSummary();
    setText("responseText", String(error));
  } finally {
    if (state.frameRequestController === requestController) {
      state.frameRequestController = null;
    }
    state.frameRequestInFlight = false;
    if (state.forcedFrameRequestQueued) {
      state.forcedFrameRequestQueued = false;
      window.setTimeout(() => fetchFrame({ force: true }), 0);
    }
  }
}

async function requestJSON(url, options = {}, { timeoutMs = 10000, signal = null } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  // Abort without a custom string reason so fetch consistently rejects with a
  // standards-defined AbortError. The flags below distinguish cancellation
  // from timeout without relying on browser-specific rejection values.
  const relayAbort = () => controller.abort();
  if (signal?.aborted) relayAbort();
  else signal?.addEventListener("abort", relayAbort, { once: true });
  const timeoutToken = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const rawText = await response.text();
    let payload = {};
    if (rawText) {
      try {
        payload = JSON.parse(rawText);
      } catch {
        payload = { message: rawText };
      }
    }
    if (!response.ok || payload?.ok === false) {
      const detail = payload?.detail || payload?.message || payload?.error || payload?.status || `HTTP ${response.status}`;
      throw new Error(String(detail));
    }
    return payload;
  } catch (error) {
    if (error?.name === "AbortError") {
      if (!timedOut && signal?.aborted) {
        const cancelled = new Error("Request cancelled because the display context changed.");
        cancelled.name = "AbortError";
        throw cancelled;
      }
      throw new Error("Command timed out. Check the device connection and try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutToken);
    signal?.removeEventListener("abort", relayAbort);
  }
}

function renderSpectrumProcessingStatus(payload = state.spectrumProcessing) {
  const status = payload?.spectrum_processing || payload || {};
  const settings = {
    ...(state.spectrumProcessing?.settings || {}),
    ...(status.settings || {}),
  };
  state.spectrumProcessing = {
    ...state.spectrumProcessing,
    ...status,
    settings,
  };

  if (spectrumSensorModeSelect) {
    spectrumSensorModeSelect.value = String(Number(settings.sensor_mode) === 1 ? 1 : 0);
  }
  if (spectrumIntegrationInput) {
    spectrumIntegrationInput.value = String(Number(settings.integration_us) || 300);
  }
  if (spectrumOverlayToggle) {
    spectrumOverlayToggle.checked = Boolean(settings.update_overlay_data);
  }
  if (spectrumBackgroundToggle) {
    spectrumBackgroundToggle.checked = Boolean(settings.subtract_background);
  }
  if (spectrumBaselineToggle) {
    spectrumBaselineToggle.checked = Boolean(settings.baseline_correction);
  }
  if (spectrumSmoothingToggle) {
    spectrumSmoothingToggle.checked = Boolean(settings.spectrum_smoothing);
  }
  if (spectrumNormalizationToggle) {
    spectrumNormalizationToggle.checked = Boolean(settings.normalize_spectrum);
  }

  const enabledSteps = [];
  if (settings.baseline_correction) enabledSteps.push("baseline");
  if (settings.spectrum_smoothing) enabledSteps.push("smooth");
  if (settings.update_overlay_data) enabledSteps.push("overlay");
  if (settings.subtract_background) enabledSteps.push("dark");
  if (settings.normalize_spectrum) {
    const normalizationStatus = String(
      state.frame?.latest?.spectrum_normalization?.status || ""
    );
    enabledSteps.unshift(
      ["ready", "ready_with_invalid_reference_points"].includes(normalizationStatus)
        ? "I/I0"
        : "I/I0 waiting"
    );
  }
  if (spectrumProcessingStatus) {
    spectrumProcessingStatus.textContent = enabledSteps.length
      ? `Display: ${enabledSteps.join(" + ")}`
      : "Display: raw";
    spectrumProcessingStatus.title =
      "Recognition retains raw intensity. Recording keeps raw counts and adds I0 plus I/I0 when the no-contact reference is ready.";
  }
  if (spectrumBackgroundStatus) {
    spectrumBackgroundStatus.textContent = status.background_reference_ready
      ? `Dark: ${status.background_reference_points || "ready"} points`
      : "Dark: none";
  }
}

async function loadSpectrumProcessingStatus() {
  try {
    const payload = await requestJSON(
      "/api/spectrum_processing/status",
      { cache: "no-store" },
      { timeoutMs: 4000 }
    );
    renderSpectrumProcessingStatus(payload);
    return payload;
  } catch (error) {
    console.warn("[spectrum processing status]", error);
    renderSpectrumProcessingStatus();
    return null;
  }
}

async function loadRuntimeCapabilities() {
  try {
    const payload = await requestJSON(
      "/api/health",
      { cache: "no-store" },
      { timeoutMs: 4000 }
    );
    const runtime = payload?.recognition_runtime || {};
    state.activeRecognitionModelLabel = String(
      runtime?.display_name || "Current optical model"
    );
    setText("recognitionModeStatus", state.activeRecognitionModelLabel);
    return payload;
  } catch (error) {
    console.warn("[runtime capabilities]", error);
    state.activeRecognitionModelLabel = "Runtime unavailable";
    setText("recognitionModeStatus", state.activeRecognitionModelLabel);
    return null;
  }
}

async function updateSpectrumProcessingSettings(updates, { restartLive = false } = {}) {
  const payload = await requestJSON(
    "/api/spectrum_processing/settings",
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    },
    { timeoutMs: 5000 }
  );
  renderSpectrumProcessingStatus(payload);

  if (restartLive && state.sdkLiveActive) {
    const channel = encodeURIComponent(state.selectedChannel);
    const integration = Number(
      payload?.spectrum_processing?.settings?.integration_us
      || state.spectrumProcessing?.settings?.integration_us
      || 300
    );
    const sensorMode = Number(
      payload?.spectrum_processing?.settings?.sensor_mode
      ?? state.spectrumProcessing?.settings?.sensor_mode
      ?? 0
    );
    const livePayload = await requestJSON(
      `/api/live/start?channel_id=${channel}&interval_sec=0.01&integration=${integration}&sensor_mode=${sensorMode}&control_sense=false&source=direct_sdk`,
      { method: "POST" },
      { timeoutMs: 12000 }
    );
    updateWatcherUI(livePayload?.watcher || {}, livePayload?.sdk_live || {});
    await fetchFrame({ force: true });
  }
  return payload;
}

async function postJSON(url, payload = {}) {
  return requestJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function loadMfbgProfileStatus() {
  try {
    const payload = await requestJSON(
      "/api/mfbg-intensity/profiles",
      { cache: "no-store" },
      { timeoutMs: 4000 }
    );
    const profiles = payload?.profiles || {};
    const current = profiles[payload?.active_runtime_profile] || {};
    const future = profiles[payload?.future_primary_profile] || {};
    const currentProfileId = String(payload?.active_runtime_profile || "");
    const currentIsMfbg =
      current.sensor_family === "micro_fiber_bragg_grating" ||
      currentProfileId.toLowerCase().startsWith("mfbg");
    const currentProfileLabel = currentIsMfbg ? "mFBG intensity" : "Ordinary FBG";
    setText("surfaceModeChip", currentProfileLabel);
    const sensorProfileStatusItem = document.getElementById("sensorProfileStatusItem");
    if (sensorProfileStatusItem) {
      sensorProfileStatusItem.title = currentIsMfbg
        ? "Active profile: mFBG optical intensity."
        : "Active profile: Ordinary FBG. mFBG intensity profile is integrated and awaiting real calibration.";
    }
    setText(
      "mfbgCurrentRuntime",
      currentProfileLabel
    );
    setText("mfbgFuturePrimary", future.display_name || "mFBG 3x3");
    setText(
      "mfbgSignalMode",
      future.demodulation_mode === "spectral_window_intensity"
        ? "Optical intensity"
        : future.demodulation_mode || "Pending"
    );
    setText(
      "mfbgRealArrayStatus",
      future.real_3x3_enabled ? "Enabled" : "Disabled"
    );
    const realArrayStatus = document.getElementById("mfbgRealArrayStatus");
    if (realArrayStatus) {
      realArrayStatus.dataset.healthTone = future.real_3x3_enabled ? "ok" : "warning";
    }
  } catch {
    setText("surfaceModeChip", "Ordinary FBG");
    const sensorProfileStatusItem = document.getElementById("sensorProfileStatusItem");
    if (sensorProfileStatusItem) {
      sensorProfileStatusItem.title = "Active sensor profile. Profile status is temporarily unavailable.";
    }
    setText("mfbgFuturePrimary", "mFBG profile unavailable");
    setText("mfbgSignalMode", "--");
    setText("mfbgRealArrayStatus", "Disabled");
  }
}

function commandErrorMessage(error, fallback = "Command failed") {
  const message = String(error?.message || error || fallback).replace(/^Error:\s*/i, "").trim();
  return message.length > 150 ? `${message.slice(0, 147)}...` : message || fallback;
}

function hideCommandFeedback() {
  if (!commandFeedback) return;
  commandFeedback.classList.remove("visible");
  commandFeedback.setAttribute("aria-hidden", "true");
}

function setCommandFeedback(message, tone = "info", { autoHideMs = 0 } = {}) {
  if (!commandFeedback || !commandFeedbackText) return;
  if (state.commandFeedbackTimer) {
    window.clearTimeout(state.commandFeedbackTimer);
    state.commandFeedbackTimer = null;
  }
  commandFeedbackText.textContent = message;
  commandFeedback.dataset.tone = tone;
  commandFeedback.classList.add("visible");
  commandFeedback.setAttribute("aria-hidden", "false");
  if (autoHideMs > 0) {
    state.commandFeedbackTimer = window.setTimeout(() => {
      hideCommandFeedback();
      state.commandFeedbackTimer = null;
    }, autoHideMs);
  }
}

function setCommandControlsLocked(locked) {
  [liveTwinButton, exportWatchButton, baselineButton, pauseButton, ingestExportButton, resetButton, demoMenuButton]
    .filter(Boolean)
    .forEach((button) => {
      button.disabled = locked;
    });
}

async function runCommand({ id, button, busyLabel, busyMessage, successMessage, action }) {
  if (!button || typeof action !== "function") return null;
  if (state.commandPending) {
    setCommandFeedback(`Finish ${state.commandPending} before starting another command.`, "info", { autoHideMs: 2200 });
    return null;
  }

  state.commandPending = id;
  setDemoMenuOpen(false);
  setSpectrumDrawerOpen(false);
  setSettingsPanelOpen(false);
  setCommandControlsLocked(true);
  button.classList.add("command-busy");
  button.setAttribute("aria-busy", "true");
  setCommandButtonDescription(button, busyLabel);
  setCommandFeedback(busyMessage, "busy");

  try {
    const result = await action();
    const message = typeof successMessage === "function" ? successMessage(result) : successMessage;
    setCommandFeedback(message || "Command completed.", "success", { autoHideMs: 2400 });
    return result;
  } catch (error) {
    console.error(`[${id}]`, error);
    button.classList.add("command-error");
    window.setTimeout(() => button.classList.remove("command-error"), 4500);
    setCommandFeedback(commandErrorMessage(error), "error", { autoHideMs: 7000 });
    return null;
  } finally {
    button.classList.remove("command-busy");
    button.removeAttribute("aria-busy");
    state.commandPending = null;
    setCommandControlsLocked(false);
    syncPrimaryCommandLabels();
  }
}

function leaveDemoMode() {
  invalidateFrameRequestContext();
  stopDemoAutoplay();
  state.demoModeActive = false;
  state.arrayDemoActive = false;
  state.arrayDemoScenario = null;
  state.arrayDemoTraceRecords = [];
  state.demoCurrentLevel = null;
  state.arrayDemoNextStepAt = 0;
  state.arrayDemoCycleStartedAt = 0;
  state.arrayDemoActionCompletedAt = 0;
  state.arrayDemoPlaybackMode = null;
  state.arrayDemoActionComplete = false;
  state.trajectoryHistory = [];
  setDemoStatus("ready", "ready");
  updateDemoControls();
}

channelSelect.addEventListener("change", () => {
  state.selectedChannel = channelSelect.value;
  if (state.dataStreamActive || state.demoModeActive || state.exportWatchActive || state.sdkLiveActive || state.liveRequested) fetchFrame();
  else {
    const idleFrame = makeIdleFrame();
    state.frame = idleFrame;
    updateUI(idleFrame);
  }
});

baselineButton.addEventListener("click", () => {
  if (!window.confirm("Confirm that the tactile surface is fully released and stable.")) {
    return;
  }
  runCommand({
    id: "baseline",
    button: baselineButton,
    busyLabel: "Setting...",
    busyMessage: "Checking stable post-release spectra and building the recovery-state baseline...",
    successMessage: (result) => {
      const modelBaseline = result?.current_runtime_spectrum_baseline;
      if (modelBaseline?.ok) {
        const sampleCount = Number(modelBaseline?.sample_count);
        return Number.isFinite(sampleCount)
          ? `Stable recovery baseline set from ${sampleCount} full-spectrum frames.`
          : "Stable post-release recovery baseline set.";
      }
      const frameCount = Number(result?.frame_count);
      return Number.isFinite(frameCount)
        ? `Global FBG01-FBG09 baseline set from ${frameCount} frames.`
        : "Global FBG01-FBG09 display baseline set.";
    },
    action: async () => {
      leaveDemoMode();
      resetRuntimeTraceHistory();
      state.dataStreamActive = true;
      const result = await requestJSON(
        "/api/global_candidate_baseline?minimum_frames=30&no_contact_attested=true&attested_by=operator_ui",
        { method: "POST" },
        { timeoutMs: 12000 }
      );
      await fetchFrame({ force: true });
      return result;
    },
  });
});

ingestExportButton.addEventListener("click", () => {
  runCommand({
    id: "export ingest",
    button: ingestExportButton,
    busyLabel: "Loading...",
    busyMessage: "Loading the latest Sense export...",
    successMessage: "Latest export loaded.",
    action: async () => {
      leaveDemoMode();
      resetRuntimeTraceHistory();
      state.dataStreamActive = true;
      const result = await requestJSON(
        `/api/ingest_latest_export?channel_id=${encodeURIComponent(state.selectedChannel)}`,
        { method: "POST" },
        { timeoutMs: 12000 }
      );
      await fetchFrame({ force: true });
      return result;
    },
  });
});

exportWatchButton.addEventListener("click", () => {
  const stopping = state.exportWatchActive;
  runCommand({
    id: stopping ? "stop watch" : "start watch",
    button: exportWatchButton,
    busyLabel: stopping ? "Stopping..." : "Starting...",
    busyMessage: stopping ? "Stopping Sense export watch..." : "Starting Sense export watch...",
    successMessage: stopping
      ? "Export watch stopped."
      : "Export watch started. Set a stable recovery baseline before recognition.",
    action: async () => {
      leaveDemoMode();
      setPaused(false);
      state.dataStreamActive = true;
      if (stopping) {
        await requestJSON("/api/export_watch/stop", { method: "POST" });
        clearStoppedLivePresentation();
      } else {
        resetRuntimeTraceHistory();
        if (state.sdkLiveActive) {
          await requestJSON("/api/sdk/stop", { method: "POST" });
        }
        const channel = encodeURIComponent(state.selectedChannel);
        await requestJSON(`/api/export_watch/start?channel_id=${channel}&interval_sec=0.35`, { method: "POST" });
      }
      if (!stopping) await fetchFrame({ force: true });
    },
  });
});

liveTwinButton.addEventListener("click", () => {
  const stopping = state.exportWatchActive || state.sdkLiveActive;
  const previousLiveRequested = state.liveRequested;
  runCommand({
    id: stopping ? "stop live" : "start live",
    button: liveTwinButton,
    busyLabel: stopping ? "Stopping..." : "Connecting...",
    busyMessage: stopping ? "Stopping live acquisition..." : "Connecting to BaySpec acquisition...",
    successMessage: stopping
      ? "Live acquisition stopped."
      : () => (state.sdkLiveActive || state.exportWatchActive
        ? "Live acquisition active. Set a stable recovery baseline before recognition."
        : "Live start accepted; waiting for signal."),
    action: async () => {
      leaveDemoMode();
      setPaused(false);
      state.dataStreamActive = true;
      state.liveRequested = !stopping;
      const channel = encodeURIComponent(state.selectedChannel);
      try {
        if (stopping) {
          await requestJSON("/api/live/stop?control_sense=false", { method: "POST" }, { timeoutMs: 12000 });
          clearStoppedLivePresentation();
        } else {
          resetRuntimeTraceHistory();
          const source = inputSourceSelect.value === "export_watch" ? "export_watch" : "direct_sdk";
          const controlSense = source === "export_watch" ? "true" : "false";
          const sourceIntervalSec = source === "direct_sdk" ? 0.01 : 0.1;
          const integration = Number(state.spectrumProcessing?.settings?.integration_us) || 300;
          const sensorMode = Number(state.spectrumProcessing?.settings?.sensor_mode) === 1 ? 1 : 0;
          await requestJSON(
            `/api/live/start?channel_id=${channel}&interval_sec=${sourceIntervalSec}&integration=${integration}&sensor_mode=${sensorMode}&control_sense=${controlSense}&source=${source}`,
            { method: "POST" },
            { timeoutMs: 12000 }
          );
        }
        if (!stopping) await fetchFrame({ force: true });
      } catch (error) {
        state.liveRequested = previousLiveRequested;
        throw error;
      }
    },
  });
});

inputSourceSelect.addEventListener("change", async () => {
  try {
    leaveDemoMode();
    resetRuntimeTraceHistory();
    state.dataStreamActive = true;
    if (inputSourceSelect.value !== "export_watch" && state.exportWatchActive) {
      await requestJSON(
        "/api/export_watch/stop",
        { method: "POST" },
        { timeoutMs: 12000 }
      );
      state.exportWatchActive = false;
    }
    if (inputSourceSelect.value !== "direct_sdk" && state.sdkLiveActive) {
      await requestJSON(
        "/api/sdk/stop",
        { method: "POST" },
        { timeoutMs: 12000 }
      );
      state.sdkLiveActive = false;
    }
    await fetchFrame({ force: true });
  } catch (error) {
    console.error("[input source change]", error);
    setCommandFeedback(commandErrorMessage(error, "Input source change failed"), "error", {
      autoHideMs: 7000,
    });
    syncPrimaryCommandLabels();
  }
});

pauseButton.addEventListener("click", () => {
  const paused = !state.paused;
  setPaused(paused);
  setCommandFeedback(paused ? "Display updates paused; acquisition state is unchanged." : "Display updates resumed.", "info", {
    autoHideMs: 2400,
  });
});

resetButton.addEventListener("click", () => {
  runCommand({
    id: "reset buffer",
    button: resetButton,
    busyLabel: "Clearing...",
    busyMessage: "Clearing the trace buffer...",
    successMessage: "Trace buffer cleared; baseline retained.",
    action: async () => {
      leaveDemoMode();
      resetRuntimeTraceHistory();
      setPaused(false);
      const acquisitionContinues = Boolean(
        state.exportWatchActive || state.sdkLiveActive || state.liveRequested
      );
      await requestJSON("/api/reset?keep_baseline=true", { method: "POST" });
      state.dataStreamActive = acquisitionContinues;
      if (acquisitionContinues) {
        await fetchFrame({ force: true });
      } else {
        const idleFrame = makeIdleFrame();
        state.frame = idleFrame;
        updateUI(idleFrame);
      }
    },
  });
});

demoStepButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    stopDemoAutoplay();
    closeOperatorDemoMenuAfterScenarioSelection();
    await runDemoTransition(
      () => injectDemoFrame(button.dataset.demoLevel, { reset: true }),
      button
    );
  });
});

arrayDemoStepButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    closeOperatorDemoMenuAfterScenarioSelection();
    await runDemoTransition(
      () => injectArrayDemoFrame(button.dataset.arrayScenario || "center_press", {
        resetTrajectory: true,
        playbackMode: "single",
      }),
      button
    );
  });
});

window.__baySpecDemoHooks = {
  injectArrayDemoFrame,
  injectArrayDemoFrameAtStep,
  setPlaybackRate: (value) => setDemoPlaybackRate(value),
  getPlaybackRate: () => state.demoPlaybackRate,
  getStepIntervalMs: () => demoArrayStepIntervalMs(),
  getLoopIntervalMs: () => DEMO_ARRAY_LOOP_INTERVAL_MS,
  getPlaybackState: () => ({
    mode: state.arrayDemoPlaybackMode,
    actionComplete: state.arrayDemoActionComplete,
    scenario: state.arrayDemoScenario,
    step: state.arrayDemoStep,
  }),
  getSchedulerIntervalMs: () => DEMO_FRAME_SCHEDULER_INTERVAL_MS,
};

window.__touchValidationHooks = {
  fetchCurrentFrame: async () => {
    state.dataStreamActive = true;
    await fetchFrame({ force: true });
  },
};

nodeDebugButton?.addEventListener("click", () => {
  state.nodeDebugExpanded = !state.nodeDebugExpanded;
  updateDemoControls();
  updateChannelGrid(state.currentArrayFrame?.channels || state.frame?.channel_grid || [], state.selectedChannel);
});

operatorModeButton?.addEventListener("click", () => {
  updateDisplayMode("operator");
});

diagnosticsModeButton?.addEventListener("click", () => {
  updateDisplayMode("diagnostics");
});

diagnosticTabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    updateDiagnosticWorkspace(button.dataset.diagnosticTab || "signal");
  });
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = diagnosticTabButtons.indexOf(button);
    let nextIndex = currentIndex;
    if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = diagnosticTabButtons.length - 1;
    else if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + diagnosticTabButtons.length) % diagnosticTabButtons.length;
    else nextIndex = (currentIndex + 1) % diagnosticTabButtons.length;
    diagnosticTabButtons[nextIndex]?.focus();
    diagnosticTabButtons[nextIndex]?.click();
  });
});

operatorDiagnosticsButton?.addEventListener("click", () => {
  updateDisplayMode("diagnostics");
});

function openExclusiveDiagnosticCard(card, { scroll = false } = {}) {
  if (!card) return;
  diagnosticAccordionCards.forEach((other) => {
    if (other !== card && other.open) other.open = false;
  });
  card.open = true;
  if (scroll) {
    requestAnimationFrame(() => scrollDiagnosticCardToTop(card));
  }
}

function scrollDiagnosticCardToTop(card) {
  const panel = card?.parentElement;
  if (!panel) return;
  const panelRect = panel.getBoundingClientRect();
  const cardRect = card.getBoundingClientRect();
  const targetTop = Math.max(0, panel.scrollTop + cardRect.top - panelRect.top - 2);
  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  panel.scrollTo({ top: targetTop, behavior: reducedMotion ? "auto" : "smooth" });
}

diagnosticAccordionCards.forEach((card) => {
  card.addEventListener("toggle", () => {
    if (!card.open || state.displayMode !== "diagnostics") return;
    diagnosticAccordionCards.forEach((other) => {
      if (other !== card && other.open) other.open = false;
    });
    requestAnimationFrame(() => scrollDiagnosticCardToTop(card));
  });
});

operatorAlertDiagnosticsButton?.addEventListener("click", () => {
  updateDisplayMode("diagnostics");
  openExclusiveDiagnosticCard(document.querySelector(".diagnostic-frame-card"), { scroll: true });
});

function setSpectrumDrawerOpen(open, restoreFocus = true) {
  const wasOpen = spectrumDrawer?.classList.contains("open");
  if (open) {
    if (!wasOpen && document.activeElement instanceof HTMLElement) spectrumDrawerOpener = document.activeElement;
    setDemoMenuOpen(false);
    setSettingsPanelOpen(false, false);
  }
  spectrumDrawer?.classList.toggle("open", open);
  spectrumDrawer?.setAttribute("aria-hidden", open ? "false" : "true");
  spectrumToggleButton?.setAttribute("aria-expanded", open ? "true" : "false");
  opticalSummaryCard?.setAttribute("aria-expanded", open ? "true" : "false");
  state.chartsNeedRefresh = true;
  requestAnimationFrame(() => {
    drawSpectrum(state.smoothSpectrumRecord || state.targetSpectrumRecord);
    drawSelectedPeakZoom(state.smoothSpectrumRecord || state.targetSpectrumRecord);
    if (open && !wasOpen) {
      spectrumCloseButton?.focus();
    } else if (!open && wasOpen) {
      const opener = spectrumDrawerOpener;
      spectrumDrawerOpener = null;
      if (restoreFocus && opener?.isConnected) opener.focus();
    }
  });
}

function setDemoMenuOpen(open) {
  if (!demoModule) return;
  if (open) {
    setSpectrumDrawerOpen(false, false);
    setSettingsPanelOpen(false, false);
    demoModule.setAttribute("open", "");
  } else {
    demoModule.removeAttribute("open");
  }
  demoMenuButton?.classList.toggle("menu-open", open);
  demoMenuButton?.setAttribute("aria-expanded", open ? "true" : "false");
}

function closeOperatorDemoMenuAfterScenarioSelection() {
  if (state.displayMode === "diagnostics") return;
  setDemoMenuOpen(false);
}

function setSettingsPanelOpen(open, restoreFocus = true) {
  const wasOpen = settingsPanel?.classList.contains("open");
  if (open) {
    if (!wasOpen && document.activeElement instanceof HTMLElement) settingsPanelOpener = document.activeElement;
    setDemoMenuOpen(false);
    setSpectrumDrawerOpen(false, false);
  }
  settingsPanel?.classList.toggle("open", open);
  settingsPanel?.setAttribute("aria-hidden", open ? "false" : "true");
  settingsButton?.classList.toggle("settings-open", open);
  settingsButton?.setAttribute("aria-expanded", open ? "true" : "false");
  requestAnimationFrame(() => {
    if (open && !wasOpen) {
      settingsCloseButton?.focus();
    } else if (!open && wasOpen) {
      const opener = settingsPanelOpener;
      settingsPanelOpener = null;
      if (restoreFocus && opener?.isConnected) opener.focus();
    }
  });
}

spectrumToggleButton?.addEventListener("click", () => {
  setSpectrumDrawerOpen(!spectrumDrawer?.classList.contains("open"));
});

spectrumCloseButton?.addEventListener("click", () => {
  setSpectrumDrawerOpen(false);
});

opticalSummaryCard?.addEventListener("click", () => {
  if (state.displayMode !== "operator") return;
  setSpectrumDrawerOpen(true);
});

opticalSummaryCard?.addEventListener("keydown", (event) => {
  if (state.displayMode !== "operator" || (event.key !== "Enter" && event.key !== " ")) return;
  event.preventDefault();
  setSpectrumDrawerOpen(true);
});

demoMenuButton?.addEventListener("click", () => {
  setDemoMenuOpen(!demoModule?.open);
});

demoSpeedControl?.addEventListener("input", () => {
  setDemoPlaybackRate(demoSpeedControl.value);
});

settingsButton?.addEventListener("click", () => {
  const willOpen = !settingsPanel?.classList.contains("open");
  setSettingsPanelOpen(willOpen);
  if (willOpen) void loadSpectrumProcessingStatus();
});

settingsCloseButton?.addEventListener("click", () => {
  setSettingsPanelOpen(false);
});

settingsWholeHandButton?.addEventListener("click", () => {
  updateGeometryDisplayMode("whole_hand");
  setSettingsPanelOpen(false);
});

settingsThumbHolderButton?.addEventListener("click", () => {
  updateGeometryDisplayMode("thumb_holder");
  setSettingsPanelOpen(false);
});

settingsSurfaceOnlyButton?.addEventListener("click", () => {
  updateGeometryDisplayMode("surface_only");
  setSettingsPanelOpen(false);
});

settingsResetCameraButton?.addEventListener("click", () => {
  applyThumbCameraConfig();
  state.threeNeedsRefresh = true;
});

spectrumSensorModeSelect?.addEventListener("change", async () => {
  spectrumSensorModeSelect.disabled = true;
  try {
    const sensorMode = Number(spectrumSensorModeSelect.value) === 1 ? 1 : 0;
    await updateSpectrumProcessingSettings(
      { sensor_mode: sensorMode },
      { restartLive: true }
    );
    setCommandFeedback(
      `BaySpec mode set to ${sensorMode === 0 ? "High sensitivity" : "High dynamic range"}.`,
      "success",
      { autoHideMs: 2400 }
    );
  } catch (error) {
    console.error("[BaySpec sensor mode]", error);
    setCommandFeedback(commandErrorMessage(error, "Sensor mode update failed"), "error", {
      autoHideMs: 7000,
    });
    await loadSpectrumProcessingStatus();
  } finally {
    spectrumSensorModeSelect.disabled = false;
  }
});

spectrumIntegrationInput?.addEventListener("change", async () => {
  spectrumIntegrationInput.disabled = true;
  try {
    const integrationUs = Math.max(
      1,
      Math.min(10000000, Math.round(Number(spectrumIntegrationInput.value) || 300))
    );
    await updateSpectrumProcessingSettings(
      { integration_us: integrationUs },
      { restartLive: true }
    );
    setCommandFeedback(
      `Exposure set to ${integrationUs} µs.`,
      "success",
      { autoHideMs: 2400 }
    );
  } catch (error) {
    console.error("[spectrum exposure]", error);
    setCommandFeedback(commandErrorMessage(error, "Exposure update failed"), "error", {
      autoHideMs: 7000,
    });
    await loadSpectrumProcessingStatus();
  } finally {
    spectrumIntegrationInput.disabled = false;
  }
});

[
  [spectrumOverlayToggle, "update_overlay_data"],
  [spectrumBackgroundToggle, "subtract_background"],
  [spectrumBaselineToggle, "baseline_correction"],
  [spectrumSmoothingToggle, "spectrum_smoothing"],
  [spectrumNormalizationToggle, "normalize_spectrum"],
].forEach(([control, settingName]) => {
  control?.addEventListener("change", async () => {
    control.disabled = true;
    try {
      await updateSpectrumProcessingSettings({ [settingName]: control.checked });
    } catch (error) {
      console.error(`[spectrum setting: ${settingName}]`, error);
      setCommandFeedback(commandErrorMessage(error, "Spectrum display update failed"), "error", {
        autoHideMs: 7000,
      });
      await loadSpectrumProcessingStatus();
    } finally {
      control.disabled = false;
    }
  });
});

spectrumCaptureBackgroundButton?.addEventListener("click", async () => {
  spectrumCaptureBackgroundButton.disabled = true;
  try {
    const payload = await requestJSON(
      "/api/spectrum_processing/background/capture",
      { method: "POST" },
      { timeoutMs: 5000 }
    );
    renderSpectrumProcessingStatus(payload);
    setCommandFeedback(
      "Dark frame captured. Enable Subtract background when appropriate.",
      "success",
      { autoHideMs: 3200 }
    );
  } catch (error) {
    setCommandFeedback(commandErrorMessage(error, "Dark-frame capture failed"), "error", {
      autoHideMs: 7000,
    });
  } finally {
    spectrumCaptureBackgroundButton.disabled = false;
  }
});

spectrumClearBackgroundButton?.addEventListener("click", async () => {
  spectrumClearBackgroundButton.disabled = true;
  try {
    const payload = await requestJSON(
      "/api/spectrum_processing/background/clear",
      { method: "POST" },
      { timeoutMs: 5000 }
    );
    renderSpectrumProcessingStatus(payload);
    setCommandFeedback("Dark frame cleared.", "success", { autoHideMs: 2200 });
  } catch (error) {
    setCommandFeedback(commandErrorMessage(error, "Unable to clear dark frame"), "error", {
      autoHideMs: 7000,
    });
  } finally {
    spectrumClearBackgroundButton.disabled = false;
  }
});

document.addEventListener("pointerdown", (event) => {
  const target = event.target;
  if (settingsPanel?.classList.contains("open") && !settingsPanel.contains(target) && !settingsButton?.contains(target)) {
    setSettingsPanelOpen(false, false);
  }
  if (demoModule?.open && !demoModule.contains(target) && !demoMenuButton?.contains(target)) {
    setDemoMenuOpen(false);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (state.surfaceFullscreenActive) {
    event.preventDefault();
    void setSurfaceFullscreen(false);
    return;
  }
  setSettingsPanelOpen(false);
  setDemoMenuOpen(false);
  setSpectrumDrawerOpen(false);
});

physicalProxyModeButton?.addEventListener("click", () => {
  updateSurfaceRenderMode("physical_proxy");
});

responseTerrainModeButton?.addEventListener("click", () => {
  updateSurfaceRenderMode("response_terrain");
});

wholeHandModeButton?.addEventListener("click", () => {
  updateGeometryDisplayMode("whole_hand");
});

fingerFocusSelect?.addEventListener("change", () => {
  setSelectedFinger(fingerFocusSelect.value, { focusCamera: true });
});

previousFingerButton?.addEventListener("click", () => {
  cycleFingerCloseup(-1);
});

nextFingerButton?.addEventListener("click", () => {
  cycleFingerCloseup(1);
});

thumbHolderModeButton?.addEventListener("click", () => {
  updateGeometryDisplayMode("thumb_holder");
});

surfaceOnlyModeButton?.addEventListener("click", () => {
  updateGeometryDisplayMode("surface_only");
});

surfaceFullscreenButton?.addEventListener("click", () => {
  void setSurfaceFullscreen(!state.surfaceFullscreenActive);
});

document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && state.surfaceNativeFullscreenEntered) {
    state.surfaceNativeFullscreenEntered = false;
    applySurfaceFullscreenState(false);
  } else {
    refreshFullscreenSurfaceLayout();
  }
});

thumbAlignmentSaveButton?.addEventListener("click", async () => {
  const payload = collectThumbAlignmentConfig();
  try {
    const result = await postJSON("/api/thumb_scene_config", payload);
    if (result.config) {
      state.thumbSceneConfig = result.config;
      populateThumbAlignmentPanel();
      applyThumbSceneLayout();
      setText("thumbAlignmentSaveStatus", "saved");
    } else {
      setText("thumbAlignmentSaveStatus", result.reason || "save failed");
    }
  } catch (error) {
    setText("thumbAlignmentSaveStatus", error?.message || "save failed");
  }
});

thumbAlignmentResetButton?.addEventListener("click", () => {
  resetThumbAlignmentConfig();
  setText("thumbAlignmentSaveStatus", "alignment reset locally; click Save Alignment to persist");
});

thumbShowSlotButton?.addEventListener("click", () => {
  const element = document.getElementById("thumbSurfaceVisible");
  if (element) element.checked = true;
  state.thumbSceneConfig = {
    ...(state.thumbSceneConfig || {}),
    ...collectThumbAlignmentConfig(),
  };
  applyThumbSceneLayout();
  populateThumbAlignmentPanel();
});

thumbHideMeshButton?.addEventListener("click", () => {
  const element = document.getElementById("thumbModelVisible");
  if (element) element.checked = !(element.checked);
  state.thumbSceneConfig = {
    ...(state.thumbSceneConfig || {}),
    ...collectThumbAlignmentConfig(),
  };
  applyThumbSceneLayout();
  populateThumbAlignmentPanel();
  thumbHideMeshButton.textContent = document.getElementById("thumbModelVisible")?.checked === false ? "Show Thumb Mesh" : "Hide Thumb Mesh";
});

thumbWireframeButton?.addEventListener("click", () => {
  const element = document.getElementById("thumbWireframeToggle");
  if (element) element.checked = !(element.checked);
  state.thumbSceneConfig = {
    ...(state.thumbSceneConfig || {}),
    ...collectThumbAlignmentConfig(),
  };
  applyThumbSceneLayout();
  populateThumbAlignmentPanel();
  setupThumbHolderModel();
});

document.querySelectorAll("[data-thumb-align-input]").forEach((input) => {
  input.addEventListener("change", () => {
    state.thumbSceneConfig = {
      ...(state.thumbSceneConfig || {}),
      ...collectThumbAlignmentConfig(),
    };
    applyThumbSceneLayout();
    populateThumbAlignmentPanel();
  });
});

const geometryControlGroups = Array.from(document.querySelectorAll(".geometry-control-group"));
geometryControlGroups.forEach((group) => {
  group.addEventListener("toggle", () => {
    if (!group.open || group.classList.contains("geometry-visibility-group")) return;
    geometryControlGroups.forEach((other) => {
      if (other !== group && !other.classList.contains("geometry-visibility-group")) other.open = false;
    });
  });
});

rawCoupledViewButton?.addEventListener("click", () => {
  updateCouplingView("raw_coupled_response");
});

idealIndependentViewButton?.addEventListener("click", () => {
  updateCouplingView("independent_ideal_response");
});

couplingCompensatedViewButton?.addEventListener("click", () => {
  setText("couplingDiagnosticNote", "Mechanical coupling inversion is disabled until an experimentally identified coupling matrix is provided.");
});

layoutCheckButton?.addEventListener("click", () => {
  state.layoutCheckVisible = !state.layoutCheckVisible;
  layoutCheckOverlay?.classList.toggle("hidden", !state.layoutCheckVisible);
  layoutCheckButton.textContent = state.layoutCheckVisible ? "Hide layout check" : "Show layout check";
});

demoAutoButton?.addEventListener("click", async () => {
  if (state.demoAutoplay) {
    stopDemoAutoplay();
    return;
  }
  await runDemoTransition(
    () => injectArrayDemoFrame(state.arrayDemoScenario || "center_press", {
      resetTrajectory: true,
      playbackMode: "loop",
    }),
    demoAutoButton
  );
});

demoSingleButton?.addEventListener("click", async () => {
  await runDemoTransition(
    () => injectArrayDemoFrame(state.arrayDemoScenario || "center_press", {
      resetTrajectory: true,
      playbackMode: "single",
    }),
    demoSingleButton
  );
});

demoResetButton?.addEventListener("click", async () => {
  stopDemoAutoplay();
  state.demoModeActive = false;
  state.demoCurrentLevel = null;
  await runDemoTransition(
    () => injectDemoFrame("no_contact", { reset: true }),
    demoResetButton
  );
});

async function performPx6dSoftwareZero() {
  if (state.commandPending) return;
  if (px6dCaptureIsBusy()) {
    setCommandFeedback("Stop recording before zeroing the force sensor.", "warning", { autoHideMs: 4500 });
    updatePx6dZeroButtonState();
    return;
  }
  state.commandPending = "PX6D software zero";
  [px6dTareButton, diagnosticPx6dTareButton, px6dCaptureTareButton].forEach((button) => {
    if (!button) return;
    button.disabled = true;
    const label = button.querySelector("span");
    if (label) label.textContent = "Zeroing...";
    else button.textContent = "Zeroing...";
  });
  try {
    const result = await requestJSON(
      "/api/px6d/tare?duration_sec=1.0",
      { method: "POST" },
      { timeoutMs: 3500 }
    );
    setCommandFeedback(
      `PX6D software zero set from ${result.sample_count || 0} stable samples.`,
      "success",
      { autoHideMs: 2600 }
    );
    await fetchPx6dReference();
  } catch (error) {
    setCommandFeedback(commandErrorMessage(error, "PX6D software zero failed"), "error", { autoHideMs: 6000 });
  } finally {
    state.commandPending = null;
    if (px6dTareButton) px6dTareButton.textContent = "Zero Fz";
    if (diagnosticPx6dTareButton) {
      const label = diagnosticPx6dTareButton.querySelector("span");
      if (label) label.textContent = "Zero six-axis reference";
    }
    if (px6dCaptureTareButton) {
      const label = px6dCaptureTareButton.querySelector("span");
      if (label) label.textContent = "Zero";
    }
    updatePx6dZeroButtonState();
    updateCaptureReadiness();
    refreshLucideIcons();
  }
}

px6dTareButton?.addEventListener("click", performPx6dSoftwareZero);
diagnosticPx6dTareButton?.addEventListener("click", performPx6dSoftwareZero);
px6dCaptureTareButton?.addEventListener("click", performPx6dSoftwareZero);

px6dCaptureOutputRoot?.addEventListener("input", () => {
  px6dCaptureOutputRoot.dataset.userEdited = "true";
  updateCaptureReadiness();
});

measurementRootInput?.addEventListener("input", () => {
  measurementRootInput.dataset.userEdited = "true";
});

measurementSessionSelect?.addEventListener("change", () => {
  if (measurementAnalyzeButton) {
    measurementAnalyzeButton.disabled =
      state.measurementRequestInFlight || !measurementSessionSelect.value;
  }
});

measurementEstimateSource?.addEventListener("change", () => {
  state.measurementResult = null;
  clearMeasurementMetrics();
  drawMeasurementComparison([]);
  setMeasurementStatus("Ready", "Analyze to load the selected evidence curve.");
});

measurementRefreshButton?.addEventListener("click", refreshMeasurementSessions);
measurementAnalyzeButton?.addEventListener("click", analyzeSelectedMeasurementSession);

px6dCapturePositionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (button.disabled || !px6dCapturePosition) return;
    updateCapturePositionSelection(button.dataset.capturePosition || "");
    updateCaptureReadiness();
  });
});

px6dCaptureNextTrialButton?.addEventListener("click", () => {
  nextCaptureTrialId();
  px6dCaptureTrial?.focus();
  px6dCaptureTrial?.select();
});

[px6dCaptureSpectrum, px6dCaptureResponse, px6dCaptureForce].forEach((control) => {
  control?.addEventListener("change", updateCaptureReadiness);
});

px6dCaptureBrowseButton?.addEventListener("click", async () => {
  if (state.px6dCaptureRequestInFlight) return;
  const chooser = window.pywebview?.api?.choose_output_directory;
  if (typeof chooser !== "function") {
    setCommandFeedback(
      "Folder browsing is available in the desktop app. Enter the full folder path here in browser mode.",
      "warning",
      { autoHideMs: 5000 }
    );
    px6dCaptureOutputRoot?.focus();
    return;
  }
  try {
    const result = await chooser(px6dCaptureOutputRoot?.value || "");
    if (result?.ok && result?.path && px6dCaptureOutputRoot) {
      px6dCaptureOutputRoot.value = result.path;
      px6dCaptureOutputRoot.dataset.userEdited = "true";
      updateCaptureReadiness();
      setCommandFeedback("Capture folder selected.", "success", { autoHideMs: 1800 });
    }
  } catch (error) {
    setCommandFeedback(commandErrorMessage(error, "Unable to choose capture folder"), "error", { autoHideMs: 5000 });
  }
});

px6dCaptureStartButton?.addEventListener("click", async () => {
  if (state.px6dCaptureRequestInFlight) return;
  if (state.commandPending) {
    setCommandFeedback("Wait for force zero to finish.", "warning", { autoHideMs: 4000 });
    return;
  }
  const selectedOutputs = selectedCaptureOutputs();
  if (!selectedOutputs.length) {
    setCommandFeedback("Select data to save.", "warning", { autoHideMs: 4000 });
    return;
  }
  const readiness = updateCaptureReadiness();
  if (!readiness.ready) {
    setCommandFeedback(`Check ${readiness.missing.join(", ")}.`, "warning", { autoHideMs: 4000 });
    return;
  }
  invalidatePx6dCaptureStatusPoll();
  state.px6dCaptureRequestInFlight = true;
  updatePx6dCapturePanel(state.px6dCaptureStatus || {});
  try {
    const payload = await requestJSON(
      "/api/px6d_capture/start",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          position_label: px6dCapturePosition?.value || "unlabeled",
          action_label: "continuous_px6d_fz_reference",
          trial_id: px6dCaptureTrial?.value || "trial_001",
          operator_note: px6dCaptureNote?.value || "",
          output_root: px6dCaptureOutputRoot?.value?.trim() || null,
          selected_outputs: selectedOutputs,
        }),
      },
      { timeoutMs: 2500 }
    );
    updatePx6dCapturePanel(payload);
    setCommandFeedback(
      "Recording started.",
      "success",
      { autoHideMs: 2400 }
    );
  } catch (error) {
    setCommandFeedback(commandErrorMessage(error, "Unable to start recording"), "error", { autoHideMs: 6000 });
  } finally {
    state.px6dCaptureRequestInFlight = false;
    await fetchPx6dCaptureStatus();
  }
});

px6dCaptureStopButton?.addEventListener("click", async () => {
  if (state.px6dCaptureRequestInFlight) return;
  invalidatePx6dCaptureStatusPoll();
  state.px6dCaptureRequestInFlight = true;
  updatePx6dCapturePanel(state.px6dCaptureStatus || {});
  try {
    const payload = await requestJSON(
      "/api/px6d_capture/stop",
      { method: "POST" },
      { timeoutMs: 5000 }
    );
    updatePx6dCapturePanel(payload);
    if (Number(payload.captured_timeline_frames ?? payload.captured_paired_frames ?? 0) > 0) {
      nextCaptureTrialId();
    }
    const savedFrameRate = Number(payload.captured_frame_rate_hz);
    const savedRateText = Number.isFinite(savedFrameRate)
      ? ` at ${savedFrameRate.toFixed(1)} Hz`
      : "";
    setCommandFeedback(
      `Saved ${payload.captured_timeline_frames ?? payload.captured_paired_frames ?? 0} frames${savedRateText}.`,
      Number(payload.captured_timeline_frames ?? payload.captured_paired_frames ?? 0) > 0 ? "success" : "warning",
      { autoHideMs: 4000 }
    );
  } catch (error) {
    setCommandFeedback(commandErrorMessage(error, "Unable to stop recording"), "error", { autoHideMs: 6000 });
  } finally {
    state.px6dCaptureRequestInFlight = false;
    await fetchPx6dCaptureStatus();
  }
});

window.addEventListener("resize", () => {
  windowResizeActive = true;
  window.clearTimeout(windowResizeSettleTimer);
  windowResizeSettleTimer = window.setTimeout(() => {
    cancelAnimationFrame(resizeToken);
    resizeToken = requestAnimationFrame(() => {
      if (state.displayMode === "diagnostics") {
        setDiagnosticsPanelWidth(Number(diagnosticsPanelResizer?.getAttribute("aria-valuenow")), { persist: false });
      }
      resizeThree();
      if (state.displayMode === "diagnostics" && state.diagnosticTab === "measurement") {
        drawMeasurementComparison(state.measurementResult?.trace || []);
      }
      state.chartsNeedRefresh = true;
      state.threeNeedsRefresh = true;
      windowResizeActive = false;
    });
  }, 120);
});

function handlePageVisibilityChange() {
  state.pageVisible = document.visibilityState !== "hidden";
  if (!state.pageVisible) {
    if (state.animationFrameId !== null) {
      cancelAnimationFrame(state.animationFrameId);
      state.animationFrameId = null;
    }
    state.forcedFrameRequestQueued = false;
    invalidateFrameRequestContext();
    invalidateLiveSourceProbe();
    invalidatePx6dReferenceRequest();
    invalidatePx6dCaptureStatusPoll();
    state.lastThreeFrameMs = 0;
    state.chartDeltaAccumulator = 0;
    state.geometryDeltaAccumulator = 0;
    return;
  }
  state.lastThreeFrameMs = 0;
  state.lastChartUpdateMs = 0;
  state.lastGeometryUpdateMs = 0;
  state.nextLiveModelPollAt = 0;
  state.nextLiveSourceProbeAt = 0;
  state.chartsNeedRefresh = true;
  state.threeNeedsRefresh = true;
  requestThreeAnimation();
  fetchFrame({ force: true });
  fetchPx6dReference();
  fetchPx6dCaptureStatus();
}

document.addEventListener("visibilitychange", handlePageVisibilityChange);

function disposeThreeRuntime() {
  if (state.animationFrameId !== null) {
    cancelAnimationFrame(state.animationFrameId);
    state.animationFrameId = null;
  }
  if (state.frameSchedulerId) window.clearInterval(state.frameSchedulerId);
  if (state.px6dSchedulerId) window.clearInterval(state.px6dSchedulerId);
  if (state.px6dCaptureSchedulerId) window.clearInterval(state.px6dCaptureSchedulerId);
  state.frameRequestController?.abort();
  invalidateLiveSourceProbe();
  state.px6dRequestController?.abort();
  state.px6dCapturePollController?.abort();
  controls?.dispose();
  disposeObject3D(scene);
  renderer?.dispose();
  renderer?.forceContextLoss?.();
}

window.addEventListener("beforeunload", disposeThreeRuntime, { once: true });

function startClientSchedulers() {
  if (state.clientSchedulersStarted) return;
  state.clientSchedulersStarted = true;
  state.frameSchedulerId = window.setInterval(fetchFrame, DEMO_FRAME_SCHEDULER_INTERVAL_MS);
  state.px6dSchedulerId = window.setInterval(fetchPx6dReference, PX6D_UI_POLL_INTERVAL_MS);
  state.px6dCaptureSchedulerId = window.setInterval(
    fetchPx6dCaptureStatus,
    PX6D_CAPTURE_POLL_INTERVAL_MS
  );
}

async function boot() {
  if (state.bootStarted) return;
  state.bootStarted = true;
  initializeDiagnosticsPanelResize();
  await Promise.all([
    loadThumbSceneConfig(),
    loadRuntimeCapabilities(),
  ]);
  setDemoPlaybackRate(state.demoPlaybackRate, { persist: false });
  updateDisplayMode("operator");
  updateGeometryDisplayMode(state.geometryDisplayMode);
  updateSurfaceRenderMode("physical_proxy");
  updateCouplingView("raw_coupled_response");
  initThree();
  setSelectedFinger(state.selectedFinger);
  exposeThreeDebugHandle();
  const idleFrame = makeIdleFrame();
  state.frame = idleFrame;
  updateUI(idleFrame);
  await loadSpectrumProcessingStatus();
  void loadMfbgProfileStatus();
  await fetchPx6dReference();
  await fetchPx6dCaptureStatus();
  const liveSourceActive = await reconcileLiveSourceState({ force: true });
  if (liveSourceActive) {
    await fetchFrame({ force: true });
  }
  startClientSchedulers();
  state.bootComplete = true;
}

boot().catch((error) => {
  state.bootStarted = false;
  state.bootComplete = false;
  console.error("[boot]", error);
  setText("responseText", "Interface initialization failed");
  setCommandFeedback(
    commandErrorMessage(error, "Interface initialization failed"),
    "error"
  );
});
