"""FastAPI backend for the standalone BaySpec FBG wavelength-shift app."""

from __future__ import annotations

import ctypes
import copy
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


APP_ROOT = Path(os.environ.get("BAYSPEC_WAVELENGTH_APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
FRONTEND_ROOT = APP_ROOT / "frontend"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from bridge import bridge
from bridge import BAYSPEC_CHANNEL_CONFIG, CHANNEL_ORDER
from backend.optical_force_capture import OpticalForceCaptureManager
from backend.px6d_reader import Px6dReader
from sdk_live import BaySpecSdkLiveReader
from src.array_surface.surface_mapper import SurfaceConfig, map_surface, matrices_from_channels
from src.hybrid_spectrum.dynamic_shadow_adapter import DynamicTemporalShadowAdapter
from src.hybrid_spectrum.static_model_adapter import StaticSpectralPredictor
from src.hybrid_spectrum.session_level_calibration import (
    POSITION_ORDER as CALIBRATION_POSITION_ORDER,
    PerPositionOrdinalCalibrator,
)
from src.hybrid_spectrum.temporal_prediction import TemporalStaticPredictionStabilizer

try:
    import yaml
except Exception:  # pragma: no cover - diagnostics fallback
    yaml = None


WM_COMMAND = 0x0111
IDOK = 1
IDYES = 6
SENSE_CMD_STOP = 32853
SENSE_CMD_FAST_RECORDING = 32879
PROJECT_ROOT = APP_ROOT if getattr(sys, "frozen", False) else APP_ROOT.parent
THUMB_SCENE_CONFIG_PATH = PROJECT_ROOT / "config" / "thumb_holder_scene.yaml"
STATIC_SPECTRAL_MODEL_PATH = PROJECT_ROOT / "models" / "static_spectral_recognition_bundle.joblib"
STATIC_SPECTRAL_CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "static_spectral_recognition_bundle_v7_fused_shift.joblib"
)
DYNAMIC_TEMPORAL_SHADOW_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "dynamic_temporal_shadow_candidate_v3_compact_runtime_pos240.joblib"
)
DYNAMIC_TEMPORAL_SHADOW_INFERENCE_STRIDE = 1
DYNAMIC_TEMPORAL_PEAK_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml"
)
RUNTIME_CONTACT_STATE_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "runtime_contact_state.yaml"
)
PX6D_REFERENCE_CONFIG_PATH = PROJECT_ROOT / "config" / "px6d_reference.yaml"
STATIC_SPECTRAL_MODEL_LOCK = threading.Lock()
STATIC_SPECTRAL_MODEL_CACHE_KEY: tuple | None = None
STATIC_SPECTRAL_MODEL_CACHE_VALUE: dict | None = None
STATIC_SPECTRAL_SHADOW_STABILIZER = TemporalStaticPredictionStabilizer(
    window_size=5,
    minimum_contact_frames=3,
    release_frames=2,
    minimum_position_support=0.60,
    minimum_level_support=0.60,
)
STATIC_SPECTRAL_SESSION_CALIBRATION_LOCK = threading.Lock()
STATIC_SPECTRAL_SESSION_CALIBRATOR: PerPositionOrdinalCalibrator | None = None
STATIC_SPECTRAL_SESSION_CALIBRATION_SOURCE: dict[str, Any] = {}
DYNAMIC_TEMPORAL_SHADOW_LOCK = threading.Lock()
DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN: str | None = None
DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY: tuple[Any, ...] | None = None
DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT = 0
DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD: dict[str, Any] | None = None
DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC: float | None = None
DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM: np.ndarray | None = None
DYNAMIC_TEMPORAL_SHADOW_MAX_RESAMPLE_STEPS = 12


def _load_px6d_reference_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": True,
        "auto_start": True,
        "port": "COM3",
        "baud_rate": 921600,
        "device_id": 127,
        "poll_hz": 50.0,
        "read_timeout_sec": 0.20,
        "reconnect_interval_sec": 1.0,
        "history_seconds": 300.0,
        "compression_sign": -1,
        "filter_alpha": 0.25,
        "auto_tare_on_start": True,
        "auto_tare_duration_sec": 1.0,
        "auto_tare_max_std_n": 0.12,
        "sync_window_sec": 0.25,
        "sync_max_age_sec": 1.0,
        "force_full_scale_per_axis_n": 50.0,
        "moment_full_scale_per_axis_nm": 2.0,
        "warning_utilization_percent": 90.0,
        "sync_excellent_max_offset_ms": 50.0,
        "sync_good_max_offset_ms": 150.0,
        "sync_acceptable_max_offset_ms": 250.0,
        "capture_output_directory": "data/px6d_synchronized",
        "capture_poll_interval_sec": 0.05,
        "capture_require_software_tare": True,
    }
    if yaml is None or not PX6D_REFERENCE_CONFIG_PATH.exists():
        return defaults
    try:
        payload = yaml.safe_load(
            PX6D_REFERENCE_CONFIG_PATH.read_text(encoding="utf-8")
        ) or {}
    except Exception:
        return defaults
    serial_config = payload.get("serial") or {}
    signal_config = payload.get("signal") or {}
    tare_config = payload.get("software_tare") or {}
    sync_config = payload.get("synchronization") or {}
    mechanical_config = payload.get("mechanical") or {}
    capture_config = payload.get("capture") or {}
    defaults.update(
        {
            "enabled": bool(payload.get("enabled", defaults["enabled"])),
            "auto_start": bool(payload.get("auto_start", defaults["auto_start"])),
            "port": serial_config.get("port", defaults["port"]),
            "baud_rate": serial_config.get("baud_rate", defaults["baud_rate"]),
            "device_id": serial_config.get("device_id", defaults["device_id"]),
            "poll_hz": serial_config.get("poll_hz", defaults["poll_hz"]),
            "read_timeout_sec": serial_config.get(
                "read_timeout_sec", defaults["read_timeout_sec"]
            ),
            "reconnect_interval_sec": serial_config.get(
                "reconnect_interval_sec", defaults["reconnect_interval_sec"]
            ),
            "history_seconds": signal_config.get(
                "history_seconds", defaults["history_seconds"]
            ),
            "compression_sign": signal_config.get(
                "compression_sign", defaults["compression_sign"]
            ),
            "filter_alpha": signal_config.get(
                "filter_alpha", defaults["filter_alpha"]
            ),
            "auto_tare_on_start": tare_config.get(
                "auto_tare_on_start", defaults["auto_tare_on_start"]
            ),
            "auto_tare_duration_sec": tare_config.get(
                "duration_sec", defaults["auto_tare_duration_sec"]
            ),
            "auto_tare_max_std_n": tare_config.get(
                "maximum_fz_std_n", defaults["auto_tare_max_std_n"]
            ),
            "sync_window_sec": sync_config.get(
                "force_window_half_width_sec", defaults["sync_window_sec"]
            ),
            "sync_max_age_sec": sync_config.get(
                "maximum_nearest_sample_age_sec", defaults["sync_max_age_sec"]
            ),
            "force_full_scale_per_axis_n": mechanical_config.get(
                "force_full_scale_per_axis_n",
                defaults["force_full_scale_per_axis_n"],
            ),
            "moment_full_scale_per_axis_nm": mechanical_config.get(
                "moment_full_scale_per_axis_nm",
                defaults["moment_full_scale_per_axis_nm"],
            ),
            "warning_utilization_percent": mechanical_config.get(
                "warning_utilization_percent",
                defaults["warning_utilization_percent"],
            ),
            "sync_excellent_max_offset_ms": sync_config.get(
                "excellent_max_offset_ms",
                defaults["sync_excellent_max_offset_ms"],
            ),
            "sync_good_max_offset_ms": sync_config.get(
                "good_max_offset_ms", defaults["sync_good_max_offset_ms"]
            ),
            "sync_acceptable_max_offset_ms": sync_config.get(
                "acceptable_max_offset_ms",
                defaults["sync_acceptable_max_offset_ms"],
            ),
            "capture_output_directory": capture_config.get(
                "output_directory", defaults["capture_output_directory"]
            ),
            "capture_poll_interval_sec": capture_config.get(
                "poll_interval_sec", defaults["capture_poll_interval_sec"]
            ),
            "capture_require_software_tare": capture_config.get(
                "require_software_tare",
                defaults["capture_require_software_tare"],
            ),
        }
    )
    return defaults


def _spectrum_fingerprint(
    wavelength: list[float], intensity: list[float]
) -> tuple[Any, ...]:
    midpoint = len(intensity) // 2
    return (
        len(intensity),
        round(float(wavelength[0]), 9),
        round(float(wavelength[-1]), 9),
        round(float(intensity[0]), 6),
        round(float(intensity[midpoint]), 6),
        round(float(intensity[-1]), 6),
        round(float(sum(intensity)), 3),
    )


def _spectrum_token(wavelength: list[float], intensity: list[float]) -> str:
    """Return an exact baseline identity used to invalidate live calibration."""

    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(wavelength)))
    digest.update(struct.pack(f"<{len(wavelength)}d", *map(float, wavelength)))
    digest.update(struct.pack(f"<{len(intensity)}d", *map(float, intensity)))
    return digest.hexdigest()


def _clear_session_level_calibration(reason: str) -> dict[str, Any]:
    global STATIC_SPECTRAL_SESSION_CALIBRATOR
    global STATIC_SPECTRAL_SESSION_CALIBRATION_SOURCE

    with STATIC_SPECTRAL_SESSION_CALIBRATION_LOCK:
        was_loaded = STATIC_SPECTRAL_SESSION_CALIBRATOR is not None
        STATIC_SPECTRAL_SESSION_CALIBRATOR = None
        STATIC_SPECTRAL_SESSION_CALIBRATION_SOURCE = {
            "cleared_reason": reason,
            "cleared_at": time.time(),
        }
    return {
        "ok": True,
        "was_loaded": was_loaded,
        "status": "session_level_calibration_cleared",
        "reason": reason,
        "runtime_role": "shadow_diagnostic_only",
    }


def _session_level_calibration_status() -> dict[str, Any]:
    with STATIC_SPECTRAL_SESSION_CALIBRATION_LOCK:
        calibrator = STATIC_SPECTRAL_SESSION_CALIBRATOR
        source = copy.deepcopy(STATIC_SPECTRAL_SESSION_CALIBRATION_SOURCE)
    return {
        "loaded": calibrator is not None,
        "schema_version": calibrator.schema_version if calibrator is not None else None,
        "baseline_token": calibrator.baseline_token if calibrator is not None else None,
        "calibrated_position_count": len(calibrator.anchors) if calibrator is not None else 0,
        "runtime_role": "shadow_diagnostic_only",
        "drives_operator_ui": False,
        "drives_digital_twin": False,
        "force_semantics": "approximate_manual_response_level_not_force_N",
        "source": source,
    }


def _apply_session_level_calibration(
    prediction: dict[str, Any],
    temporal: dict[str, Any],
    *,
    baseline_token: str,
) -> dict[str, Any]:
    with STATIC_SPECTRAL_SESSION_CALIBRATION_LOCK:
        calibrator = STATIC_SPECTRAL_SESSION_CALIBRATOR
    base = {
        "runtime_role": "shadow_diagnostic_only",
        "drives_operator_ui": False,
        "drives_digital_twin": False,
        "force_semantics": "approximate_manual_response_level_not_force_N",
    }
    if calibrator is None:
        return {**base, "ok": False, "status": "session_calibration_not_loaded"}
    if temporal.get("contact_label") != "contact":
        return {
            **base,
            "ok": True,
            "status": "stable_no_contact",
            "label": None,
        }
    position = temporal.get("position_label")
    if not temporal.get("ready") or not position:
        return {
            **base,
            "ok": False,
            "status": "waiting_for_stable_temporal_position",
            "label": None,
        }
    features = prediction.get("response_calibration_features")
    if not isinstance(features, dict):
        return {
            **base,
            "ok": False,
            "status": "response_calibration_features_missing",
            "label": None,
        }
    calibrated = calibrator.predict(
        str(position),
        features,
        baseline_token=baseline_token,
    )
    calibrated.update(base)
    return calibrated


def _current_runtime_baseline_token() -> tuple[str | None, dict[str, Any]]:
    pair = bridge.spectral_model_input(channel_id="P22")
    if not pair.get("ok"):
        return None, pair
    baseline = pair["baseline"]
    token = _spectrum_token(
        baseline["wavelength_nm"],
        baseline["intensity"],
    )
    return token, pair


def _load_runtime_baseline_recovery_config() -> dict[str, Any]:
    if yaml is None or not RUNTIME_CONTACT_STATE_CONFIG_PATH.exists():
        return {}
    payload = yaml.safe_load(
        RUNTIME_CONTACT_STATE_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}
    section = payload.get("runtime_baseline_recovery", payload)
    return dict(section) if isinstance(section, dict) else {}


def _load_response_level_postprocess_config() -> dict[str, Any]:
    if yaml is None or not RUNTIME_CONTACT_STATE_CONFIG_PATH.exists():
        return {}
    payload = yaml.safe_load(
        RUNTIME_CONTACT_STATE_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}
    section = payload.get("response_level_postprocess", {})
    return dict(section) if isinstance(section, dict) else {}


try:
    STATIC_SPECTRAL_PREDICTOR = StaticSpectralPredictor(STATIC_SPECTRAL_MODEL_PATH)
    STATIC_SPECTRAL_MODEL_ERROR = None
except Exception as exc:  # pragma: no cover - exposed through diagnostics
    STATIC_SPECTRAL_PREDICTOR = None
    STATIC_SPECTRAL_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
try:
    STATIC_SPECTRAL_CANDIDATE_PREDICTOR = StaticSpectralPredictor(
        STATIC_SPECTRAL_CANDIDATE_MODEL_PATH
    )
    STATIC_SPECTRAL_CANDIDATE_ERROR = None
except Exception as exc:  # pragma: no cover - exposed through diagnostics
    STATIC_SPECTRAL_CANDIDATE_PREDICTOR = None
    STATIC_SPECTRAL_CANDIDATE_ERROR = f"{type(exc).__name__}: {exc}"
try:
    DYNAMIC_TEMPORAL_SHADOW_ADAPTER = DynamicTemporalShadowAdapter.from_paths(
        DYNAMIC_TEMPORAL_SHADOW_MODEL_PATH,
        DYNAMIC_TEMPORAL_PEAK_CONFIG_PATH,
        runtime_recovery_config=_load_runtime_baseline_recovery_config(),
        response_level_config=_load_response_level_postprocess_config(),
    )
    DYNAMIC_TEMPORAL_SHADOW_ERROR = None
except Exception as exc:  # pragma: no cover - exposed through diagnostics
    DYNAMIC_TEMPORAL_SHADOW_ADAPTER = None
    DYNAMIC_TEMPORAL_SHADOW_ERROR = f"{type(exc).__name__}: {exc}"


def _reset_dynamic_temporal_shadow(reason: str) -> dict[str, Any]:
    global DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN
    global DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY
    global DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT
    global DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD
    global DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC
    global DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM

    with DYNAMIC_TEMPORAL_SHADOW_LOCK:
        if DYNAMIC_TEMPORAL_SHADOW_ADAPTER is not None:
            DYNAMIC_TEMPORAL_SHADOW_ADAPTER.clear()
        DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN = None
        DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY = None
        DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT = 0
        DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD = None
        DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC = None
        DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM = None
    return {
        "ok": True,
        "status": "dynamic_temporal_shadow_reset",
        "reason": reason,
        "runtime_role": "shadow_only_not_driving_digital_twin",
    }


def _dynamic_temporal_shadow_status() -> dict[str, Any]:
    adapter = DYNAMIC_TEMPORAL_SHADOW_ADAPTER
    bundle = adapter.bundle if adapter is not None else {}
    grouped = bundle.get("release_guard_grouped_cv", {})
    return {
        "loaded": adapter is not None,
        "model_path": str(DYNAMIC_TEMPORAL_SHADOW_MODEL_PATH),
        "model_error": DYNAMIC_TEMPORAL_SHADOW_ERROR,
        "schema_version": bundle.get("schema_version"),
        "status": bundle.get("status"),
        "deployment_ready": False,
        "runtime_role": "shadow_only_not_driving_digital_twin",
        "drives_operator_ui": False,
        "drives_digital_twin": False,
        "inference_stride_unique_frames": DYNAMIC_TEMPORAL_SHADOW_INFERENCE_STRIDE,
        "temporal_window_frames": bundle.get("time_steps"),
        "model_frame_interval_sec": bundle.get("frame_interval_sec_estimated"),
        "physical_frame_resampling_enabled": True,
        "maximum_resample_steps_per_physical_frame": (
            DYNAMIC_TEMPORAL_SHADOW_MAX_RESAMPLE_STEPS
        ),
        "runtime_contact_state_config_path": str(
            RUNTIME_CONTACT_STATE_CONFIG_PATH
        ),
        "runtime_baseline_recovery": _load_runtime_baseline_recovery_config(),
        "release_guard_grouped_detection_rate": grouped.get(
            "release_sequence_detection_rate"
        ),
        "release_guard_unsafe_early_triggers": grouped.get(
            "unsafe_early_trigger_sequence_count"
        ),
        "response_level_semantics": "approximate_manual_level_not_force_N",
    }


def _predict_dynamic_temporal_shadow(
    latest_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate each unique spectrum on the model's trained temporal scale."""

    global DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN
    global DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY
    global DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT
    global DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD
    global DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC
    global DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM

    adapter = DYNAMIC_TEMPORAL_SHADOW_ADAPTER
    if adapter is None:
        return {
            "ok": False,
            "status": "dynamic_shadow_unavailable",
            "reason": DYNAMIC_TEMPORAL_SHADOW_ERROR,
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }
    pair = bridge.spectral_model_input(channel_id="P22")
    if not pair.get("ok"):
        return {
            "ok": False,
            "status": "baseline_required" if pair.get("current_ready") else "spectrum_required",
            "reason": pair.get("reason"),
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }
    latest = pair["latest"]
    if (
        isinstance(latest_override, dict)
        and isinstance(latest_override.get("wavelength_nm"), list)
        and isinstance(latest_override.get("intensity"), list)
        and latest_override.get("wavelength_nm")
        and len(latest_override["wavelength_nm"]) == len(latest_override["intensity"])
    ):
        # Synchronized recording passes the exact spectrum frame being written.
        # This avoids a one-frame race if the live SDK publishes again between
        # the capture read and temporal-model inference.
        latest = latest_override
    baseline = pair["baseline"]
    baseline_token = _spectrum_token(
        baseline["wavelength_nm"],
        baseline["intensity"],
    )
    frame_key = (
        latest.get("frame_id"),
        latest.get("timestamp"),
        latest.get("source"),
        _spectrum_fingerprint(latest["wavelength_nm"], latest["intensity"]),
    )
    started = time.perf_counter()
    try:
        with DYNAMIC_TEMPORAL_SHADOW_LOCK:
            if baseline_token != DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN:
                adapter.set_baseline(
                    baseline["wavelength_nm"],
                    baseline["intensity"],
                )
                DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN = baseline_token
                DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY = None
                DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT = 0
                DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD = None
                DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC = None
                DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM = None
            if (
                frame_key == DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY
                and DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD is not None
            ):
                cached = copy.deepcopy(DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD)
                cached["duplicate_frame_ignored"] = True
                cached["cache_lookup_latency_ms"] = (
                    time.perf_counter() - started
                ) * 1000.0
                return cached

            DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT += 1
            run_inference = (
                DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT
                % DYNAMIC_TEMPORAL_SHADOW_INFERENCE_STRIDE
                == 0
            )
            current_spectrum = np.asarray(latest["intensity"], dtype=float)
            try:
                current_timestamp_sec = float(latest.get("timestamp"))
            except (TypeError, ValueError):
                current_timestamp_sec = None
            expected_interval_sec = max(
                0.01,
                float(adapter.bundle.get("frame_interval_sec_estimated") or 0.04),
            )
            source_interval_sec: float | None = None
            if (
                current_timestamp_sec is not None
                and DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC is not None
            ):
                candidate_interval = (
                    current_timestamp_sec
                    - DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC
                )
                if 0.0 < candidate_interval <= 5.0:
                    source_interval_sec = candidate_interval
            resample_steps = 1
            if (
                source_interval_sec is not None
                and DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM is not None
                and DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM.shape
                == current_spectrum.shape
            ):
                resample_steps = max(
                    1,
                    min(
                        DYNAMIC_TEMPORAL_SHADOW_MAX_RESAMPLE_STEPS,
                        int(round(source_interval_sec / expected_interval_sec)),
                    ),
                )

            raw_response_level = str(latest.get("response_level") or "").lower()
            raw_qa_status = str(latest.get("qa_status") or "").lower()
            external_no_contact_hint = bool(
                raw_response_level == "no_contact"
                and raw_qa_status not in {"invalid", "error", "stale"}
            )
            previous_spectrum = DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM
            prediction: dict[str, Any] | None = None
            for step_index in range(1, resample_steps + 1):
                is_physical_frame = step_index == resample_steps
                if previous_spectrum is None or resample_steps == 1:
                    model_spectrum = current_spectrum
                else:
                    fraction = step_index / resample_steps
                    model_spectrum = previous_spectrum + fraction * (
                        current_spectrum - previous_spectrum
                    )
                prediction = adapter.update(
                    latest["wavelength_nm"],
                    model_spectrum,
                    run_inference=bool(run_inference and is_physical_frame),
                    physical_frame=is_physical_frame,
                    external_no_contact_hint=(
                        external_no_contact_hint if is_physical_frame else None
                    ),
                    source_timestamp_sec=(
                        current_timestamp_sec if is_physical_frame else None
                    ),
                )
            if prediction is None:  # pragma: no cover - defensive contract guard
                raise RuntimeError("temporal resampler produced no model frame")
            runtime_baseline_update = None
            pending_baseline = adapter.consume_pending_runtime_baseline_update()
            if pending_baseline is not None:
                runtime_baseline_update = (
                    bridge.set_runtime_recovery_spectrum_baseline(
                        "P22",
                        pending_baseline["wavelength_nm"],
                        pending_baseline["intensity"],
                        sample_count=int(pending_baseline.get("sample_count") or 1),
                        span_sec=float(pending_baseline.get("span_sec") or 0.0),
                        shape_motion_rms=pending_baseline.get("shape_motion_rms"),
                        common_gain_motion=pending_baseline.get(
                            "common_gain_motion"
                        ),
                        policy=str(
                            pending_baseline.get("policy")
                            or "multi_evidence_release_then_spectral_stationarity"
                        ),
                    )
                )
                if runtime_baseline_update.get("ok"):
                    baseline_token = _spectrum_token(
                        pending_baseline["wavelength_nm"].tolist(),
                        pending_baseline["intensity"].tolist(),
                    )
                    DYNAMIC_TEMPORAL_SHADOW_BASELINE_TOKEN = baseline_token
            DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_TIMESTAMP_SEC = current_timestamp_sec
            DYNAMIC_TEMPORAL_SHADOW_LAST_SOURCE_SPECTRUM = current_spectrum.copy()
            payload = {
                "ok": True,
                "status": prediction.get("status"),
                "prediction": prediction,
                "inference_executed_this_frame": run_inference,
                "unique_frame_count": DYNAMIC_TEMPORAL_SHADOW_UNIQUE_FRAME_COUNT,
                "inference_latency_ms": (time.perf_counter() - started) * 1000.0,
                "duplicate_frame_ignored": False,
                "cache_lookup_latency_ms": 0.0,
                "physical_frame_resampling_enabled": True,
                "temporal_resample_steps": resample_steps,
                "model_frame_interval_ms": expected_interval_sec * 1000.0,
                "source_frame_interval_ms": (
                    source_interval_sec * 1000.0
                    if source_interval_sec is not None
                    else None
                ),
                "external_no_contact_hint": external_no_contact_hint,
                "baseline_token": baseline_token,
                "runtime_baseline_update": runtime_baseline_update,
                "runtime_role": "shadow_only_not_driving_digital_twin",
                "drives_operator_ui": False,
                "drives_digital_twin": False,
                "deployment_ready": False,
                "response_level_semantics": "approximate_manual_level_not_force_N",
            }
            DYNAMIC_TEMPORAL_SHADOW_LAST_FRAME_KEY = frame_key
            DYNAMIC_TEMPORAL_SHADOW_LAST_PAYLOAD = copy.deepcopy(payload)
            return payload
    except Exception as exc:
        return {
            "ok": False,
            "status": "dynamic_shadow_inference_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }


def _dynamic_temporal_display_prediction(
    dynamic_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Adapt the validated temporal candidate to the existing twin contract.

    The saved artifact keeps its shadow-only safety metadata.  This adapter is
    only exposed when the caller explicitly requests temporal validation mode,
    so the runtime can be tested without relabelling the candidate as a final
    deployment or a calibrated-force model.
    """

    payload = dynamic_payload if isinstance(dynamic_payload, dict) else {}
    prediction = payload.get("prediction")
    if not payload.get("ok") or not isinstance(prediction, dict):
        return {
            "ok": False,
            "status": str(payload.get("status") or "dynamic_temporal_unavailable"),
            "reason": payload.get("reason"),
            "prediction": None,
            "runtime_role": "operator_validation_candidate",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }
    if prediction.get("ready") is not True:
        return {
            "ok": False,
            "status": str(prediction.get("status") or "window_warming_up"),
            "reason": "temporal_window_not_ready",
            "prediction": None,
            "history_frames": prediction.get("history_frames"),
            "required_frames": prediction.get("required_frames"),
            "runtime_role": "operator_validation_candidate",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }

    proxy = prediction.get("digital_twin_proxy")
    if not isinstance(proxy, dict):
        return {
            "ok": False,
            "status": "dynamic_temporal_proxy_missing",
            "reason": "digital_twin_proxy_missing",
            "prediction": None,
            "runtime_role": "operator_validation_candidate",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }

    contact = prediction.get("contact") or {
        "label": "no_contact",
        "confidence": None,
        "probabilities": {},
    }
    position = prediction.get("position")
    response_level = prediction.get("response_level")
    position_confidence = (
        float(position.get("confidence"))
        if isinstance(position, dict) and position.get("confidence") is not None
        else None
    )
    response_confidence = (
        float(response_level.get("confidence"))
        if isinstance(response_level, dict)
        and response_level.get("confidence") is not None
        else None
    )
    uncertainty_reasons: list[str] = []
    if proxy.get("active") is True:
        if position_confidence is not None and position_confidence < 0.55:
            uncertainty_reasons.append("low_position_confidence")
        if response_confidence is not None and response_confidence < 0.55:
            uncertainty_reasons.append("low_response_level_confidence")

    force_level = response_level or {
        "label": "no_contact",
        "confidence": contact.get("confidence"),
        "probabilities": {},
    }
    display_prediction = {
        "schema_version": "dynamic_temporal_validation_display_v1",
        "recognition_source": "dynamic_temporal_v3_validation",
        "contact": contact,
        "position": position,
        # Keep the legacy key for the existing frontend contract. Its semantics
        # remain an approximate response level, never a calibrated force value.
        "force_level": force_level,
        "response_level": response_level,
        "force_model_scope": "approximate_manual_response_level_not_force_N",
        "response_level_semantics": "approximate_manual_level_not_force_N",
        "operational_state": prediction.get("operational_state"),
        "release_guard": prediction.get("release_guard"),
        "runtime_baseline_recovery": prediction.get(
            "runtime_baseline_recovery"
        ),
        "digital_twin": {
            "active": bool(proxy.get("active")),
            "position_id": proxy.get("position_id"),
            "force_level": proxy.get("response_level"),
            "response_level": proxy.get("response_level"),
            "deformation_proxy": float(proxy.get("deformation_proxy") or 0.0),
            "surface_grid": proxy.get("surface_grid"),
            "surface_metrics": proxy.get("surface_metrics"),
            "visualization_semantics": proxy.get("visualization_semantics"),
            "physical_output_semantics": proxy.get("physical_output_semantics"),
        },
        "uncertainty": {
            "review_needed": bool(uncertainty_reasons),
            "reasons": uncertainty_reasons,
        },
        "temporal_window": {
            "history_frames": prediction.get("history_frames"),
            "required_frames": prediction.get("required_frames"),
            "frame_counter": prediction.get("frame_counter"),
        },
    }
    return {
        "ok": True,
        "status": "temporal_validation_ready",
        "prediction": display_prediction,
        "runtime_role": "operator_validation_candidate",
        "drives_operator_ui": True,
        "drives_digital_twin": True,
        "deployment_ready": False,
        "validation_only": True,
    }


def _static_spectral_model_status() -> dict:
    observed_windows = (
        {
            str(window.candidate_id): float(window.center_nm)
            for window in STATIC_SPECTRAL_PREDICTOR.peak_windows
        }
        if STATIC_SPECTRAL_PREDICTOR is not None
        else {}
    )
    observed_centers = list(observed_windows.values())
    return {
        "loaded": STATIC_SPECTRAL_PREDICTOR is not None,
        "model_path": str(STATIC_SPECTRAL_MODEL_PATH),
        "model_error": STATIC_SPECTRAL_MODEL_ERROR,
        "model_bundle_sha256": (
            STATIC_SPECTRAL_PREDICTOR.bundle_sha256
            if STATIC_SPECTRAL_PREDICTOR is not None
            else None
        ),
        "recognition_scope": "manual_fingertip_static_spectrum_position_and_level",
        "contact_geometry": "broad_approximate_manual_fingertip_contact",
        "position_classes": ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"],
        "force_classes": ["light", "normal", "hard"],
        "force_semantics": "approximate_manual_response_level_not_force_N",
        "training_no_contact_semantics": "post_press_release_recovery_no_contact",
        "runtime_baseline_contract": "stable_multiframe_post_release_recovery_baseline_required",
        "evaluation_scope": "single_session_leave_one_repeat_index_out_baseline",
        "confidence_source": "uncalibrated_predict_proba",
        "uncertainty_policy": "diagnostic_only_does_not_change_prediction",
        "observed_model_feature_windows_nm": observed_windows,
        "observed_model_feature_window_range_nm": (
            [min(observed_centers), max(observed_centers)]
            if observed_centers
            else None
        ),
        "observed_model_feature_window_status": "trained_current_ordinary_fbg_dataset",
        "future_3x3_target_plan_active": False,
        "shadow_candidate": {
            "loaded": STATIC_SPECTRAL_CANDIDATE_PREDICTOR is not None,
            "model_path": str(STATIC_SPECTRAL_CANDIDATE_MODEL_PATH),
            "model_error": STATIC_SPECTRAL_CANDIDATE_ERROR,
            "model_bundle_sha256": (
                STATIC_SPECTRAL_CANDIDATE_PREDICTOR.bundle_sha256
                if STATIC_SPECTRAL_CANDIDATE_PREDICTOR is not None
                else None
            ),
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "deployment_ready": False,
            "baseline_contract": "same_current_session_baseline_as_primary_model",
            "candidate_id": "v7_fused_common_mode_corrected_shift",
            "confidence_source": "ensemble_vote_fraction_not_calibrated",
            "promotion_gate": "labeled_live_position_and_level_validation_required",
            "temporal_stabilization": {
                "runtime_role": "shadow_diagnostic_only",
                "window_unique_frames": 5,
                "minimum_contact_frames": 3,
                "release_frames": 2,
            },
            "session_level_calibration": _session_level_calibration_status(),
        },
    }


def _predict_static_spectral_shadow(
    latest_wavelength: list[float],
    latest_intensity: list[float],
    baseline_wavelength: list[float],
    baseline_intensity: list[float],
) -> dict:
    """Run the candidate beside the deployed model without controlling the UI."""

    if STATIC_SPECTRAL_CANDIDATE_PREDICTOR is None:
        return {
            "ok": False,
            "status": "candidate_unavailable",
            "reason": STATIC_SPECTRAL_CANDIDATE_ERROR,
            "runtime_role": "shadow_only_not_driving_digital_twin",
        }
    started = time.perf_counter()
    try:
        prediction = STATIC_SPECTRAL_CANDIDATE_PREDICTOR.predict(
            latest_wavelength,
            latest_intensity,
            baseline_wavelength_nm=baseline_wavelength,
            baseline_intensity_counts=baseline_intensity,
        )
        return {
            "ok": True,
            "status": "shadow_ready",
            "prediction": prediction,
            "inference_latency_ms": (time.perf_counter() - started) * 1000.0,
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "shadow_inference_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "runtime_role": "shadow_only_not_driving_digital_twin",
        }


def _predict_static_spectral_frame(*, include_shadow: bool = False) -> dict:
    global STATIC_SPECTRAL_MODEL_CACHE_KEY, STATIC_SPECTRAL_MODEL_CACHE_VALUE

    pair = bridge.spectral_model_input(channel_id="P22")
    if STATIC_SPECTRAL_PREDICTOR is None:
        return {
            "ok": False,
            "status": "model_unavailable",
            "reason": STATIC_SPECTRAL_MODEL_ERROR or "static spectral model is not loaded",
            "input": {
                key: pair.get(key)
                for key in (
                    "current_ready",
                    "baseline_ready",
                    "baseline_spectrum_status",
                    "baseline_spectrum_sample_count",
                    "baseline_spectrum_span_sec",
                )
            },
        }
    if not pair.get("ok"):
        return {
            "ok": False,
            "status": "baseline_required" if pair.get("current_ready") else "spectrum_required",
            "reason": pair.get("reason"),
            "input": {
                key: pair.get(key)
                for key in (
                    "current_ready",
                    "baseline_ready",
                    "baseline_spectrum_status",
                    "baseline_spectrum_sample_count",
                    "baseline_spectrum_span_sec",
                )
            },
        }
    latest = pair["latest"]
    baseline = pair["baseline"]
    latest_intensity = latest["intensity"]
    latest_wavelength = latest["wavelength_nm"]
    baseline_intensity = baseline["intensity"]
    baseline_wavelength = baseline["wavelength_nm"]

    baseline_token = _spectrum_token(baseline_wavelength, baseline_intensity)

    cache_key = (
        bool(include_shadow),
        latest.get("frame_id"),
        latest.get("timestamp"),
        latest.get("source"),
        _spectrum_fingerprint(latest_wavelength, latest_intensity),
        pair.get("baseline_spectrum_sample_count"),
        pair.get("baseline_spectrum_status"),
        baseline_token,
    )
    started = time.perf_counter()
    try:
        with STATIC_SPECTRAL_MODEL_LOCK:
            if (
                STATIC_SPECTRAL_MODEL_CACHE_KEY == cache_key
                and STATIC_SPECTRAL_MODEL_CACHE_VALUE is not None
            ):
                cached = copy.deepcopy(STATIC_SPECTRAL_MODEL_CACHE_VALUE)
                cached["cache_hit"] = True
                cached["cache_lookup_latency_ms"] = (time.perf_counter() - started) * 1000.0
                return cached
            prediction = STATIC_SPECTRAL_PREDICTOR.predict(
                latest_wavelength,
                latest_intensity,
                baseline_wavelength_nm=baseline_wavelength,
                baseline_intensity_counts=baseline_intensity,
            )
            if include_shadow:
                shadow_candidate = _predict_static_spectral_shadow(
                    latest_wavelength,
                    latest_intensity,
                    baseline_wavelength,
                    baseline_intensity,
                )
                if shadow_candidate.get("ok") and isinstance(
                    shadow_candidate.get("prediction"), dict
                ):
                    shadow_candidate["temporal_stabilization"] = (
                        STATIC_SPECTRAL_SHADOW_STABILIZER.update(
                            frame_id=latest.get("frame_id")
                            or latest.get("timestamp"),
                            prediction=shadow_candidate["prediction"],
                            baseline_token=baseline_token,
                            timestamp=latest.get("timestamp"),
                        )
                    )
                    shadow_candidate["session_calibrated_force"] = (
                        _apply_session_level_calibration(
                            shadow_candidate["prediction"],
                            shadow_candidate["temporal_stabilization"],
                            baseline_token=baseline_token,
                        )
                    )
            else:
                shadow_candidate = {
                    "ok": False,
                    "status": "shadow_not_requested",
                    "runtime_role": "shadow_only_not_driving_digital_twin",
                    "drives_operator_ui": False,
                    "drives_digital_twin": False,
                    "request_hint": "set include_shadow=true for validation",
                }
            result = {
                "ok": True,
                "status": "ready",
                "prediction": prediction,
                "shadow_candidate": shadow_candidate,
                "inference_latency_ms": (time.perf_counter() - started) * 1000.0,
                "cache_hit": False,
                "cache_lookup_latency_ms": 0.0,
                "input": {
                    "current_ready": True,
                    "baseline_ready": True,
                    "baseline_spectrum_sample_count": pair.get("baseline_spectrum_sample_count"),
                    "baseline_spectrum_noise_ratio": pair.get("baseline_spectrum_noise_ratio"),
                    "baseline_spectrum_drift_ratio": pair.get("baseline_spectrum_drift_ratio"),
                    "baseline_spectrum_span_sec": pair.get("baseline_spectrum_span_sec"),
                    "baseline_spectrum_status": pair.get("baseline_spectrum_status"),
                    "baseline_spectrum_semantic_role": pair.get(
                        "baseline_spectrum_semantic_role"
                    ),
                    "baseline_spectrum_token": baseline_token,
                    "frame_id": latest.get("frame_id"),
                    "timestamp": latest.get("timestamp"),
                    "spectrum_points": len(latest["intensity"]),
                },
            }
            STATIC_SPECTRAL_MODEL_CACHE_KEY = cache_key
            STATIC_SPECTRAL_MODEL_CACHE_VALUE = copy.deepcopy(result)
            return result
    except Exception as exc:
        return {
            "ok": False,
            "status": "inference_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "input": {"current_ready": True, "baseline_ready": True},
        }


def _model_display_source_gate(
    latest: dict | None,
    watcher_status: dict,
    sdk_status: dict,
) -> dict:
    """Decide whether the latest spectrum may drive the trained display.

    Replay/HTTP frames remain usable while both live sources are stopped. Once
    a live source is active, however, only a fresh frame from that same source
    may drive the contact state and deformation. This prevents a cached press
    frame from remaining visible after acquisition stalls or reconnects.
    """

    sdk_active = bool(sdk_status.get("active"))
    watcher_active = bool(watcher_status.get("active"))
    live_source_active = sdk_active or watcher_active
    source = str((latest or {}).get("source") or "").strip().lower()

    selected_live_source = None
    source_fresh = False
    if "bayspec_direct" in source or "sdk" in source:
        selected_live_source = "sdk"
        source_fresh = sdk_active and sdk_status.get("freshness") == "live"
    elif "sense" in source or "export" in source or "watch" in source:
        selected_live_source = "watcher"
        source_fresh = watcher_active and watcher_status.get("freshness") in {
            "fresh",
            "live",
        }
    elif live_source_active:
        # A live acquisition is running, but the buffered record came from a
        # different source. Hold it for diagnostics and suppress deformation.
        selected_live_source = "unmatched_live_source"

    model_input_source_allowed = not live_source_active or source_fresh
    return {
        "live_source_active": live_source_active,
        "selected_live_source": selected_live_source,
        "source_fresh": source_fresh,
        "model_input_source_allowed": model_input_source_allowed,
        "model_input_source_mode": (
            "fresh_live"
            if source_fresh
            else "held_replay_or_http"
            if not live_source_active
            else "stale_or_mismatched_live"
        ),
    }


def _default_thumb_scene_config() -> dict:
    return {
        "thumb_holder_scene": {
            "enabled": True,
            "default_geometry_mode": "thumb_holder",
            "model_asset_url": "",
            "fallback_asset_url": "/static/assets/models/thumb_holder.stl",
            "fallback_placeholder_enabled": False,
            "model_load_policy": "glb_then_stl_else_blocked",
            "note": (
                "Thumb holder uses an oval-like groove insert. Response is uncalibrated "
                "Bragg wavelength displacement with no calibrated-force output."
            ),
        },
        "thumb_model_transform": {
            "scale": [1.0, 1.0, 1.0],
            "rotation_deg": [0.0, 0.0, 90.0],
            "position": [0.0, -0.55, 0.0],
            "holder_local_rotation_deg": [0.0, 0.0, -14.098],
            "holder_local_position": [0.0, 0.06, 0.0],
            "holder_local_scale": [1.0, 1.0, 1.0],
            "opacity": 0.34,
            "wireframe": False,
            "visible": True,
        },
        "sensor_slot_transform": {
            "coordinate_space": "thumb_model_local",
            "slot_shape": "stl_irregular_oval",
            "position": [0.546, 0.674, -0.007],
            "rotation_deg": [0.0, 0.0, 90.0],
            "vertical_lift": 0.22,
            "width_mm": 10.0,
            "height_mm": 7.0,
            "depth_mm": 1.0,
            "z_offset_mm": 0.08,
            "surface_scene_scale": [0.482, 0.268, 0.460],
            "boundary_profile": {
                "source": "stl_diff_groove_faces",
                "angular_samples": 64,
                "radial_scale": [
                    0.998,
                    0.986,
                    0.954,
                    0.931,
                    0.910,
                    0.895,
                    0.886,
                    0.879,
                    0.872,
                    0.867,
                    0.868,
                    0.878,
                    0.894,
                    0.910,
                    0.945,
                    0.976,
                    0.994,
                    0.984,
                    0.966,
                    0.948,
                    0.940,
                    0.944,
                    0.960,
                    0.982,
                    1.022,
                    1.068,
                    1.068,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.000,
                    1.063,
                    1.062,
                    1.022,
                    0.983,
                    0.949,
                    0.934,
                    0.932,
                    0.943,
                    0.962,
                    0.991,
                    0.993,
                    0.993,
                    0.948,
                    0.926,
                    0.904,
                    0.891,
                    0.885,
                    0.885,
                    0.888,
                    0.894,
                    0.899,
                    0.909,
                    0.921,
                    0.934,
                    0.963,
                    0.991,
                ],
                "note": "Extracted by differencing the original STL against the grooved STL; the groove is a non-regular capsule-like recess.",
            },
            "visible": True,
            "opacity": 1.0,
        },
        "visual_style": {
            "thumb_material_color": "#d9e1e7",
            "thumb_slot_color": "#8fd0dc",
            "slot_outline_color": "#37a8c7",
            "response_surface_label": "Sensor slot surface",
        },
    }


def _thumb_scene_config() -> dict:
    if THUMB_SCENE_CONFIG_PATH.exists() and yaml is not None:
        try:
            loaded = yaml.safe_load(THUMB_SCENE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            default = _default_thumb_scene_config()
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(default.get(key), dict):
                    default[key].update(value)
                else:
                    default[key] = value
            return default
        except Exception:
            pass
    return _default_thumb_scene_config()


def _write_thumb_scene_config(payload: dict) -> dict:
    if yaml is None:
        return {"ok": False, "reason": "pyyaml not available"}
    current = _thumb_scene_config()
    allowed_sections = {"thumb_holder_scene", "thumb_model_transform", "sensor_slot_transform", "visual_style"}
    for key, value in payload.items():
        if key in allowed_sections and isinstance(value, dict):
            current.setdefault(key, {}).update(value)
    THUMB_SCENE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    THUMB_SCENE_CONFIG_PATH.write_text(
        yaml.safe_dump(current, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {"ok": True, "config": current, "config_path": str(THUMB_SCENE_CONFIG_PATH)}


class SenseWindowController:
    """Small Windows message bridge for Sense 20/20 live export control."""

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None

    def _windows(self) -> list[dict]:
        if self.user32 is None:
            return []
        windows: list[dict] = []
        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd: int, _lparam: int) -> bool:
            if not self.user32.IsWindowVisible(hwnd):
                return True
            length = self.user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title:
                windows.append({"hwnd": int(hwnd), "title": title})
            return True

        self.user32.EnumWindows(enum_proc_type(callback), 0)
        return windows

    def _find_window(self, text: str) -> dict | None:
        text_lower = text.lower()
        for window in self._windows():
            if text_lower in window["title"].lower():
                return window
        return None

    def status(self) -> dict:
        sense = self._find_window("Sense 20/20")
        dialog = self._find_window("Fast Recording Mode")
        return {
            "ok": True,
            "method": "windows_message_control",
            "available": self.user32 is not None,
            "sense_window_found": sense is not None,
            "sense_window_title": sense["title"] if sense else None,
            "fast_recording_dialog_found": dialog is not None,
            "fast_recording_dialog_title": dialog["title"] if dialog else None,
        }

    def _post_command(self, hwnd: int, command_id: int) -> bool:
        if self.user32 is None:
            return False
        return bool(self.user32.PostMessageW(int(hwnd), WM_COMMAND, int(command_id), 0))

    def send_sense_command(self, command_id: int) -> dict:
        sense = self._find_window("Sense 20/20")
        if sense is None:
            return {"ok": False, "reason": "Sense 20/20 window not found", "command_id": command_id}
        posted = self._post_command(sense["hwnd"], command_id)
        return {
            "ok": posted,
            "command_id": command_id,
            "sense_window_title": sense["title"],
            "reason": None if posted else "PostMessageW failed",
        }

    def confirm_live_start_dialogs(self, timeout_sec: float = 5.0) -> dict:
        deadline = time.time() + max(0.1, timeout_sec)
        events: list[dict] = []
        while time.time() < deadline:
            folder_dialog = self._find_window("Select or Create A Folder")
            if folder_dialog is not None:
                posted = self._post_command(folder_dialog["hwnd"], IDOK)
                events.append({"dialog_title": folder_dialog["title"], "action": "IDOK", "posted": posted})
                time.sleep(0.25)
                continue
            fast_dialog = self._find_window("Fast Recording Mode")
            if fast_dialog is not None:
                posted = self._post_command(fast_dialog["hwnd"], IDYES)
                events.append({"dialog_title": fast_dialog["title"], "action": "IDYES", "posted": posted})
                time.sleep(0.25)
                continue
            time.sleep(0.1)
        return {
            "ok": True,
            "events": events,
            "confirmed_any": bool(events),
            "reason": None if events else "no live-start dialog found; Sense may have started without prompt",
        }

    def start_fast_recording(self, ensure_stopped: bool = True) -> dict:
        stop_result = None
        if ensure_stopped:
            stop_result = self.send_sense_command(SENSE_CMD_STOP)
            time.sleep(0.2)
        start_result = self.send_sense_command(SENSE_CMD_FAST_RECORDING)
        confirm_result = self.confirm_live_start_dialogs(timeout_sec=5.0) if start_result.get("ok") else None
        return {
            "ok": bool(start_result.get("ok")),
            "mode": "sense_fast_recording_start",
            "pre_stop": stop_result,
            "start_command": start_result,
            "confirm_dialog": confirm_result,
            "note": "Sense normal scan should be stopped before fast recording operations.",
        }

    def stop_scan(self) -> dict:
        result = self.send_sense_command(SENSE_CMD_STOP)
        result.update({"mode": "sense_stop_scan"})
        return result


class SenseExportWatcher:
    """Poll Sense 20/20 exports and ingest wavelength-calibrated spectra."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active = False
        self.channel_id = "P22"
        self.export_root: str | None = None
        self.interval_sec = 0.35
        self.last_signature: tuple[str, int, int] | None = None
        self.last_result: dict | None = None
        self.last_error: str | None = None
        self.last_file: str | None = None
        self.last_file_mtime: float | None = None
        self.last_attempt_time: float | None = None
        self.last_ingest_time: float | None = None
        self.started_at: float | None = None
        self.ingest_count = 0
        self.failed_ingest_count = 0
        self.thread: threading.Thread | None = None
        self.generation = 0

    def status(self) -> dict:
        with self.lock:
            now = time.time()
            age = now - self.last_ingest_time if self.last_ingest_time is not None else None
            file_age = (
                max(0.0, now - self.last_file_mtime)
                if self.last_file_mtime is not None
                else None
            )
            freshness_limit = max(3.0, self.interval_sec * 12)
            if not self.active:
                freshness = "stopped"
            elif self.last_ingest_time is None:
                freshness = "waiting_for_export"
            elif (
                age is not None
                and age <= freshness_limit
                and (file_age is None or file_age <= freshness_limit)
            ):
                freshness = "live"
            else:
                freshness = "stale"
            return {
                "active": self.active,
                "acquisition_session_id": self.generation,
                "channel_id": self.channel_id,
                "export_root": self.export_root,
                "interval_sec": self.interval_sec,
                "last_file": self.last_file,
                "last_file_mtime": self.last_file_mtime,
                "last_attempt_time": self.last_attempt_time,
                "last_ingest_time": self.last_ingest_time,
                "seconds_since_last_ingest": age,
                "seconds_since_last_file_update": file_age,
                "freshness_limit_sec": freshness_limit,
                "freshness": freshness,
                "last_error": self.last_error,
                "ingest_count": self.ingest_count,
                "failed_ingest_count": self.failed_ingest_count,
                "last_result": self.last_result,
                "source": "sense_export_file_polling",
            }

    def start(self, channel_id: str, export_root: str | None, interval_sec: float) -> dict:
        with self.lock:
            requested_root = (
                str(Path(export_root).expanduser().resolve())
                if export_root
                else None
            )
            current_root = (
                str(Path(self.export_root).expanduser().resolve())
                if self.export_root
                else None
            )
            configuration_changed = (
                str(channel_id) != self.channel_id or requested_root != current_root
            )
            starting_new_session = not self.active or configuration_changed
            self.channel_id = channel_id
            self.export_root = export_root
            self.interval_sec = max(0.1, min(float(interval_sec), 5.0))
            if starting_new_session:
                # Freshness is session-scoped. Preserve last_signature so an
                # unchanged old export is not re-ingested as a new live frame.
                self.started_at = time.time()
                self.last_result = None
                self.last_file = None
                self.last_file_mtime = None
                self.last_attempt_time = None
                self.last_ingest_time = None
                self.ingest_count = 0
                self.failed_ingest_count = 0
                self.generation += 1
                if configuration_changed:
                    # A signature from another root/channel must never suppress
                    # the first export of the newly selected source.
                    self.last_signature = None
            self.active = True
            self.last_error = None
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()
            return self.status()

    def stop(self) -> dict:
        with self.lock:
            self.active = False
            self.generation += 1
            return self.status()

    def _loop(self) -> None:
        while True:
            with self.lock:
                active = self.active
                channel_id = self.channel_id
                export_root = self.export_root
                interval = self.interval_sec
                generation = self.generation
            if not active:
                time.sleep(0.2)
                continue
            try:
                latest = bridge.latest_export_file(root=export_root)
                if latest is None:
                    with self.lock:
                        self.last_error = "no CSV/TXT export file found"
                        self.last_file = None
                    time.sleep(interval)
                    continue
                stat = latest.stat()
                signature = (str(latest), int(stat.st_mtime_ns), int(stat.st_size))
                if signature != self.last_signature:
                    # Give Sense a brief moment to finish writing the file.
                    time.sleep(0.05)
                    with self.lock:
                        if not self.active or generation != self.generation:
                            continue
                    result = bridge.ingest_export_file(
                        latest,
                        channel_id=channel_id,
                        source="bayspec_sense2020_export_watch",
                    )
                    with self.lock:
                        if not self.active or generation != self.generation:
                            continue
                        self.last_attempt_time = time.time()
                        self.last_result = result
                        self.last_error = None if result.get("ok") else str(result.get("reason"))
                        self.last_file = str(latest)
                        self.last_file_mtime = stat.st_mtime
                        if result.get("ok"):
                            self.last_signature = signature
                            self.last_ingest_time = self.last_attempt_time
                            self.ingest_count += int(result.get("records_ingested") or 0)
                        else:
                            self.failed_ingest_count += 1
                time.sleep(interval)
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                time.sleep(interval)


export_watcher = SenseExportWatcher()
sense_controller = SenseWindowController()
sdk_live_reader = BaySpecSdkLiveReader(bridge=bridge, app_root=APP_ROOT)
PX6D_REFERENCE_CONFIG = _load_px6d_reference_config()
px6d_reader = Px6dReader(PX6D_REFERENCE_CONFIG)


def _px6d_reference_for_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if not PX6D_REFERENCE_CONFIG.get("enabled", True):
        return {"ok": False, "status": "disabled"}
    if not isinstance(record, dict):
        return px6d_reader.synchronized_snapshot(time.time())
    timestamp = record.get("ingested_at")
    if timestamp is None:
        timestamp = record.get("timestamp_epoch_sec")
    if timestamp is None:
        timestamp = record.get("timestamp")
    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        return {"ok": False, "status": "spectrum_timestamp_missing"}
    return px6d_reader.synchronized_snapshot(timestamp_value)


def _capture_spectrum_frame() -> dict[str, Any]:
    return bridge.frame(channel_id="P22", trace_limit=1, include_spectrum=True)


def _capture_temporal_response(latest: dict[str, Any]) -> dict[str, Any]:
    payload = _predict_dynamic_temporal_shadow(latest_override=latest)
    prediction = payload.get("prediction")
    if not isinstance(prediction, dict):
        return {
            "model_source": "dynamic_temporal_v3",
            "model_status": payload.get("status") or "dynamic_temporal_unavailable",
            "model_ready": False,
            "reason": payload.get("reason"),
            "inference_latency_ms": payload.get("inference_latency_ms"),
        }
    return {
        "model_source": "dynamic_temporal_v3",
        "model_status": payload.get("status") or prediction.get("status"),
        "model_ready": bool(payload.get("ok") and prediction.get("ready")),
        "inference_latency_ms": payload.get("inference_latency_ms"),
        "inference_executed_this_frame": payload.get("inference_executed_this_frame"),
        "temporal_resample_steps": payload.get("temporal_resample_steps"),
        **prediction,
    }


def _resolve_capture_output_root() -> Path:
    runtime_override = os.environ.get("TOUCH_CAPTURE_OUTPUT_ROOT")
    if runtime_override:
        return Path(runtime_override).expanduser().resolve()
    configured = Path(str(PX6D_REFERENCE_CONFIG.get("capture_output_directory") or "data/px6d_synchronized"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


optical_force_capture = OpticalForceCaptureManager(
    output_root=_resolve_capture_output_root(),
    frame_provider=_capture_spectrum_frame,
    force_provider=_px6d_reference_for_record,
    force_status_provider=px6d_reader.status,
    model_provider=_capture_temporal_response,
    poll_interval_sec=float(PX6D_REFERENCE_CONFIG.get("capture_poll_interval_sec") or 0.05),
    require_software_tare=bool(
        PX6D_REFERENCE_CONFIG.get("capture_require_software_tare", True)
    ),
)

WAVELENGTH_PLAN_ORDER = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"]
SIMULATED_FBG_WAVELENGTHS = {
    channel_id: 1540.0 + 5.0 * index for index, channel_id in enumerate(WAVELENGTH_PLAN_ORDER)
}
SIMULATED_FBG_I0 = {
    "P11": 38500.0,
    "P21": 42000.0,
    "P31": 39800.0,
    "P12": 40500.0,
    "P22": 44000.0,
    "P32": 39200.0,
    "P13": 37200.0,
    "P23": 41000.0,
    "P33": 38600.0,
}
COUPLING_SOURCE_DEFAULTS = [
    "shared_elastomer_mechanical_coupling",
    "fingertip_contact_area_coverage",
]
DEFAULT_SAME_FIBER_PATHS = [
    ["P11", "P12", "P13"],
    ["P21", "P22", "P23"],
    ["P31", "P32", "P33"],
]
DEFAULT_CROSS_FIBER_NEIGHBOR_LINKS = [
    ["P11", "P21"],
    ["P21", "P31"],
    ["P12", "P22"],
    ["P22", "P32"],
    ["P13", "P23"],
    ["P23", "P33"],
]
COUPLING_EXPLANATION = (
    "Dense fingertip-scale Micro-FBG sensing units share one flexible surface, so one "
    "fingertip contact can shift several Bragg wavelengths through contact-area coverage "
    "and mechanical coupling. Wavelength-shift mode does not reuse the intensity edition's "
    "directional optical-loss cascade. A measured calibration matrix is still required for "
    "local-response decoupling."
)


def _operator_response_band_thresholds() -> dict[str, float | str]:
    """Return one validated response-band contract for API and demo logic."""

    raw = BAYSPEC_CHANNEL_CONFIG.get("operator_response_bands", {}) or {}
    no_contact_max = float(raw.get("no_contact_max_ratio", 0.25))
    light_max = float(raw.get("light_max_ratio", 0.80))
    normal_max = float(raw.get("normal_max_ratio", 0.90))
    if not (0.0 < no_contact_max < light_max < normal_max < 1.0):
        no_contact_max, light_max, normal_max = 0.25, 0.80, 0.90
    return {
        "no_contact_max": no_contact_max,
        "light_max": light_max,
        "normal_max": normal_max,
        "semantics": str(
            raw.get("semantics")
            or "uncalibrated_normalized_visual_response_not_force_N"
        ),
    }


def _response_level_from_shift_ratio(response_ratio: float) -> str:
    thresholds = _operator_response_band_thresholds()
    if response_ratio < float(thresholds["no_contact_max"]):
        return "no_contact"
    if response_ratio < float(thresholds["light_max"]):
        return "small_shift"
    if response_ratio < float(thresholds["normal_max"]):
        return "moderate_shift"
    return "large_shift"


def _simulated_target_wavelength(channel_id: str) -> float:
    configured = (BAYSPEC_CHANNEL_CONFIG.get("channels", {}).get(channel_id, {}) or {}).get("target_wavelength_nm")
    try:
        number = float(configured)
    except (TypeError, ValueError):
        plan = _array_wavelength_plan()
        try:
            index = WAVELENGTH_PLAN_ORDER.index(channel_id)
            number = plan["start_nm"] + index * plan["spacing_nm"]
        except ValueError:
            number = float(SIMULATED_FBG_WAVELENGTHS.get(channel_id, 1560.0))
    return number if math.isfinite(number) else float(SIMULATED_FBG_WAVELENGTHS.get(channel_id, 1550.0))


def _array_wavelength_plan() -> dict[str, float | str | int]:
    raw = BAYSPEC_CHANNEL_CONFIG.get("array_wavelength_plan", {}) or {}
    start_nm = float(raw.get("wavelength_start_nm") or 1540.0)
    stop_nm = float(raw.get("wavelength_stop_nm") or 1580.0)
    spacing_nm = float(raw.get("wavelength_spacing_nm") or 5.0)
    peak_count = int(raw.get("number_of_fbg_peaks") or 9)
    return {
        "wavelength_unit": str(raw.get("wavelength_unit") or "nm"),
        "start_nm": start_nm,
        "stop_nm": stop_nm,
        "spacing_nm": spacing_nm,
        "number_of_fbg_peaks": peak_count,
        "status": str(raw.get("status") or "preliminary_target_plan"),
        "note": str(raw.get("note") or "Target wavelengths are preliminary and should be replaced by measured peak wavelengths after fabrication."),
    }


def _array_wavelength_plan_payload() -> dict:
    plan = _array_wavelength_plan()
    return {
        "wavelength_unit": plan["wavelength_unit"],
        "wavelength_start_nm": plan["start_nm"],
        "wavelength_stop_nm": plan["stop_nm"],
        "wavelength_spacing_nm": plan["spacing_nm"],
        "number_of_fbg_peaks": plan["number_of_fbg_peaks"],
        "status": plan["status"],
        "note": plan["note"],
        "channel_order": WAVELENGTH_PLAN_ORDER,
        "target_wavelengths_nm": {channel_id: _simulated_target_wavelength(channel_id) for channel_id in WAVELENGTH_PLAN_ORDER},
    }


_CENTER_CONTACT_ENVELOPE = (
    # 0.0-0.5 s: light fingertip contact.
    0.30,
    0.42,
    0.52,
    0.56,
    0.56,
    # 0.5-2.5 s: fully released while the spectrum keeps advancing.
    *((0.0,) * 20),
    # 2.5-4.5 s: hard contact.
    0.90,
    0.90,
    *((0.94,) * 18),
    # 4.5-5.0 s: smooth final release before the next loop.
    0.74,
    0.48,
    0.22,
    0.08,
    0.0,
)

_STAGED_POINT_CONTACT_CENTERS = {
    "center_press": (0.0, 0.0),
    "p21_contact": (0.0, 1.0),
    "p12_contact": (-1.0, 0.0),
    "p32_contact": (1.0, 0.0),
}


def _demo_envelope(scenario: str, step: int) -> float:
    step = max(0, int(step))
    if scenario == "no_contact":
        return 0.0
    if scenario == "tap":
        cycle = min(step, 9)
        return [0.04, 0.16, 0.92, 0.62, 0.24, 0.08, 0.03, 0.0, 0.0, 0.0][cycle]
    if scenario == "release":
        cycle = min(step, 11)
        return max(0.0, 0.84 * (1.0 - min(cycle, 8) / 8.0))
    if scenario in _STAGED_POINT_CONTACT_CENTERS:
        return _CENTER_CONTACT_ENVELOPE[min(step, len(_CENTER_CONTACT_ENVELOPE) - 1)]
    if scenario in {"off_center_fingertip_contact", "broad_fingertip_contact"}:
        cycle = step % 14
        if cycle < 5:
            return 0.10 + 0.76 * (cycle / 4.0)
        if cycle < 10:
            return 0.86
        return max(0.0, 0.86 * (1.0 - (cycle - 9) / 4.0))
    cycle = step % 12
    if cycle < 3:
        return 0.38 + 0.16 * cycle
    if cycle < 9:
        return 0.78
    return max(0.0, 0.78 * (1.0 - (cycle - 8) / 3.0))


def _path_point(points: list[tuple[float, float]], step: int, cycle_length: int = 12) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0]
    cycle = max(0, int(step)) % max(cycle_length, 2)
    phase = cycle / max(cycle_length - 1, 1)
    scaled = phase * (len(points) - 1)
    index = min(len(points) - 2, int(math.floor(scaled)))
    local = scaled - index
    x0, y0 = points[index]
    x1, y1 = points[index + 1]
    return x0 + (x1 - x0) * local, y0 + (y1 - y0) * local


def _coupling_config() -> dict:
    section = BAYSPEC_CHANNEL_CONFIG.get("array_coupling", {}) or {}
    same_fiber_section = section.get("same_fiber_bend_loss", {}) or {}
    secondary_section = section.get("cross_fiber_secondary_cascade", {}) or {}
    sources = section.get("coupling_sources") or COUPLING_SOURCE_DEFAULTS
    if not isinstance(sources, list):
        sources = COUPLING_SOURCE_DEFAULTS
    raw_paths = same_fiber_section.get("channel_paths") or DEFAULT_SAME_FIBER_PATHS
    same_fiber_paths = []
    if isinstance(raw_paths, list):
        for raw_path in raw_paths:
            if not isinstance(raw_path, (list, tuple)):
                continue
            path = [str(channel_id) for channel_id in raw_path if str(channel_id) in CHANNEL_ORDER]
            if len(path) >= 2:
                same_fiber_paths.append(path)
    if not same_fiber_paths:
        same_fiber_paths = [path[:] for path in DEFAULT_SAME_FIBER_PATHS]
    raw_neighbor_links = secondary_section.get("neighbor_links") or DEFAULT_CROSS_FIBER_NEIGHBOR_LINKS
    cross_fiber_neighbor_links = []
    if isinstance(raw_neighbor_links, list):
        for raw_link in raw_neighbor_links:
            if not isinstance(raw_link, (list, tuple)) or len(raw_link) != 2:
                continue
            source_id, target_id = (str(raw_link[0]), str(raw_link[1]))
            if source_id not in CHANNEL_ORDER or target_id not in CHANNEL_ORDER or source_id == target_id:
                continue
            canonical = sorted([source_id, target_id])
            if canonical not in cross_fiber_neighbor_links:
                cross_fiber_neighbor_links.append(canonical)
    if not cross_fiber_neighbor_links:
        cross_fiber_neighbor_links = [sorted(link) for link in DEFAULT_CROSS_FIBER_NEIGHBOR_LINKS]
    try:
        transfer_factor = float(same_fiber_section.get("one_step_transfer_factor", 0.26))
    except (TypeError, ValueError):
        transfer_factor = 0.26
    try:
        step_decay = float(same_fiber_section.get("additional_step_decay", 0.65))
    except (TypeError, ValueError):
        step_decay = 0.65
    try:
        minimum_carrier = float(secondary_section.get("min_mechanical_carrier_attenuation", 0.01))
    except (TypeError, ValueError):
        minimum_carrier = 0.01
    try:
        minimum_secondary = float(secondary_section.get("min_secondary_contribution", 0.003))
    except (TypeError, ValueError):
        minimum_secondary = 0.003
    try:
        maximum_carrier = float(secondary_section.get("max_mechanical_carrier_attenuation", 0.35))
    except (TypeError, ValueError):
        maximum_carrier = 0.35
    try:
        mechanical_transfer_factor = float(secondary_section.get("neighbor_mechanical_transfer_factor", 0.22))
    except (TypeError, ValueError):
        mechanical_transfer_factor = 0.22
    try:
        mechanical_distance_sigma = float(secondary_section.get("neighbor_distance_sigma", 0.82))
    except (TypeError, ValueError):
        mechanical_distance_sigma = 0.82
    return {
        "enabled": bool(section.get("enabled", False)),
        "model_type": str(section.get("model_type") or "none"),
        "simulation_model_type": str(section.get("simulation_model_type") or "staged_mechanical_then_optical_proxy"),
        "propagation_direction": str(section.get("propagation_direction") or "bottom_to_top"),
        "same_fiber_bend_loss_enabled": bool(same_fiber_section.get("enabled", True)),
        "same_fiber_paths": same_fiber_paths,
        "same_fiber_affected_direction": str(same_fiber_section.get("affected_direction") or "toward_path_start"),
        "same_fiber_one_step_transfer_factor": max(0.0, min(1.0, transfer_factor)),
        "same_fiber_additional_step_decay": max(0.0, min(1.0, step_decay)),
        "cross_fiber_secondary_cascade_enabled": bool(secondary_section.get("enabled", True)),
        "cross_fiber_secondary_max_mechanical_hops": 1,
        "cross_fiber_neighbor_links": cross_fiber_neighbor_links,
        "cross_fiber_neighbor_mechanical_transfer_factor": max(0.0, min(1.0, mechanical_transfer_factor)),
        "cross_fiber_neighbor_distance_sigma": max(0.05, mechanical_distance_sigma),
        "cross_fiber_secondary_min_carrier": max(0.0, min(1.0, minimum_carrier)),
        "cross_fiber_secondary_min_contribution": max(0.0, min(1.0, minimum_secondary)),
        "cross_fiber_secondary_max_carrier": max(0.0, min(0.92, maximum_carrier)),
        "calibration_matrix_file": section.get("calibration_matrix_file"),
        "current_display": str(section.get("current_display") or section.get("default_display") or "raw_coupled_response"),
        "default_display": str(section.get("default_display") or section.get("current_display") or "raw_coupled_response"),
        "coupling_status": str(section.get("coupling_status") or "uncalibrated_dense_array_coupled_response"),
        "coupling_compensated": bool(section.get("coupling_compensated", False)),
        "coupling_sources": [str(item) for item in sources],
        "calibration_required": bool(section.get("calibration_required", True)),
    }


def _is_downstream(source: dict, target: dict, propagation_direction: str) -> bool:
    sx, sy = float(source["x"]), float(source["y"])
    tx, ty = float(target["x"]), float(target["y"])
    if propagation_direction == "right_to_left":
        return tx < sx - 1e-6
    if propagation_direction == "top_to_bottom":
        return ty < sy - 1e-6
    if propagation_direction == "bottom_to_top":
        return ty > sy + 1e-6
    return tx > sx + 1e-6


def _is_same_fiber(source: dict, target: dict) -> bool:
    # P11-P12-P13, P21-P22-P23, and P31-P32-P33 are the three physical fibers.
    return abs(float(source["x"]) - float(target["x"])) < 1e-6


def _is_cross_fiber_neighbor_link(source_id: str, target_id: str, cfg: dict) -> bool:
    pair = {source_id, target_id}
    return any(pair == set(link) for link in cfg["cross_fiber_neighbor_links"])


def _same_fiber_cascade_steps(
    source_id: str,
    target_id: str,
    source: dict,
    target: dict,
    cfg: dict,
) -> int | None:
    """Return cascade distance, zero for the unaffected direction, or None across fibers."""
    direction = cfg["same_fiber_affected_direction"]
    for path in cfg["same_fiber_paths"]:
        if source_id not in path or target_id not in path:
            continue
        source_index = path.index(source_id)
        target_index = path.index(target_id)
        if direction == "toward_path_end":
            return target_index - source_index if target_index > source_index else 0
        return source_index - target_index if target_index < source_index else 0

    if not _is_same_fiber(source, target):
        return None
    if not _is_downstream(source, target, cfg["propagation_direction"]):
        return 0
    distance = math.sqrt(
        (float(target["x"]) - float(source["x"])) ** 2
        + (float(target["y"]) - float(source["y"])) ** 2
    )
    return max(1, int(round(distance)))


def _coupled_observed_attenuation(local_by_channel: dict[str, float], coords: dict[str, tuple[float, float]]) -> dict[str, dict]:
    cfg = _coupling_config()
    observed: dict[str, dict] = {}
    for target_id in CHANNEL_ORDER:
        tx, ty = coords[target_id]
        target_info = {"x": tx, "y": ty}
        local = max(0.0, float(local_by_channel.get(target_id, 0.0)))
        coupled_extra = 0.0
        same_fiber_cascade_sources = set()
        cross_fiber_mechanical_sources = set()
        cross_fiber_mechanical_carrier_extra = 0.0
        roles = {"primary_peak"} if local >= 0.18 else set()
        for source_id, source_local in local_by_channel.items():
            source_local = max(0.0, float(source_local))
            if source_id == target_id or source_local < 0.04:
                continue
            sx, sy = coords[source_id]
            source_info = {"x": sx, "y": sy}
            distance = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
            same_fiber_steps = _same_fiber_cascade_steps(
                source_id,
                target_id,
                source_info,
                target_info,
                cfg,
            )
            same_fiber = same_fiber_steps is not None
            cross_fiber = not same_fiber
            cross_fiber_neighbor = cross_fiber and _is_cross_fiber_neighbor_link(source_id, target_id, cfg)
            if cross_fiber:
                neighbor_factor = cfg["cross_fiber_neighbor_mechanical_transfer_factor"] if cross_fiber_neighbor else 0.0
                neighbor_sigma = cfg["cross_fiber_neighbor_distance_sigma"]
            else:
                neighbor_factor = 0.12
                neighbor_sigma = 0.82
            neighbor = neighbor_factor * source_local * math.exp(
                -(distance * distance) / (2.0 * neighbor_sigma * neighbor_sigma)
            )
            same_fiber_cascade = 0.0
            if (
                cfg["same_fiber_bend_loss_enabled"]
                and same_fiber_steps is not None
                and same_fiber_steps > 0
            ):
                same_fiber_cascade = (
                    cfg["same_fiber_one_step_transfer_factor"]
                    * source_local
                    * cfg["same_fiber_additional_step_decay"] ** (same_fiber_steps - 1)
                )
            shared_elastomer = 0.055 * source_local * math.exp(-(distance * distance) / (2.0 * 1.55 * 1.55))
            spreading = 0.10 * source_local * math.exp(-(distance * distance) / (2.0 * 0.62 * 0.62))
            mechanical_carrier = max(neighbor, shared_elastomer, spreading)
            contribution = max(neighbor, same_fiber_cascade, shared_elastomer, spreading)
            if contribution <= 0.01:
                continue
            coupled_extra += contribution
            if same_fiber_cascade >= 0.015:
                roles.add("same_fiber_bend_loss_cascade_peak")
                same_fiber_cascade_sources.add(source_id)
            if cross_fiber_neighbor and neighbor >= 0.015:
                roles.add("cross_fiber_mechanical_coupled_peak")
            if cross_fiber_neighbor and mechanical_carrier >= cfg["cross_fiber_secondary_min_carrier"]:
                cross_fiber_mechanical_carrier_extra += mechanical_carrier
                cross_fiber_mechanical_sources.add(source_id)
            if shared_elastomer >= 0.015:
                roles.add("shared_elastomer_deformation_peak")
            if distance <= 1.05 and spreading >= 0.015:
                roles.add("force_transfer_ball_spreading_peak")
        observed_value = max(local, min(0.92, local + min(coupled_extra, 0.46)))
        if observed_value >= 0.05 and not roles:
            roles.add("unknown_coupled_peak")
        role_list = sorted(roles) or ["unchanged_peak"]
        observed[target_id] = {
            "local_response_estimate": local,
            "observed_attenuation_ratio": max(0.0, min(0.92, observed_value)),
            "coupling_extra": max(0.0, observed_value - local),
            "coupling_roles": role_list,
            "coupling_sources": cfg["coupling_sources"],
            "same_fiber_cascade_source_channels": sorted(same_fiber_cascade_sources),
            "cross_fiber_mechanical_source_channels": sorted(cross_fiber_mechanical_sources),
            "cross_fiber_mechanical_carrier_extra": min(
                cfg["cross_fiber_secondary_max_carrier"],
                cross_fiber_mechanical_carrier_extra,
            ),
            "secondary_cascade_extra": 0.0,
            "secondary_coupling_paths": [],
            "coupling_hop_depth": 1 if coupled_extra > 0.0 else 0,
            "possible_cross_fiber_coupling": any("cross_fiber" in role or "shared_elastomer" in role or "force_transfer" in role for role in role_list),
            "possible_same_fiber_coupling": any("same_fiber" in role for role in role_list),
            "local_response_estimate_available": False,
        }

    # One bounded same-row hop to an explicitly adjacent fiber may seed a
    # directional optical cascade on that fiber. It is intentionally non-recursive:
    # P13 -> P23 -> P22/P21 is allowed, but the derived optical response cannot
    # start another mechanical hop.
    if cfg["cross_fiber_secondary_cascade_enabled"]:
        for intermediate_id in CHANNEL_ORDER:
            intermediate = observed[intermediate_id]
            carrier = float(intermediate.get("cross_fiber_mechanical_carrier_extra") or 0.0)
            origin_ids = list(intermediate.get("cross_fiber_mechanical_source_channels") or [])
            if carrier < cfg["cross_fiber_secondary_min_carrier"] or not origin_ids:
                continue
            sx, sy = coords[intermediate_id]
            source_info = {"x": sx, "y": sy}
            for target_id in CHANNEL_ORDER:
                if target_id == intermediate_id:
                    continue
                tx, ty = coords[target_id]
                target_info = {"x": tx, "y": ty}
                steps = _same_fiber_cascade_steps(
                    intermediate_id,
                    target_id,
                    source_info,
                    target_info,
                    cfg,
                )
                if steps is None or steps <= 0:
                    continue
                secondary = (
                    cfg["same_fiber_one_step_transfer_factor"]
                    * carrier
                    * cfg["same_fiber_additional_step_decay"] ** (steps - 1)
                )
                if secondary < cfg["cross_fiber_secondary_min_contribution"]:
                    continue
                target = observed[target_id]
                previous = float(target["observed_attenuation_ratio"])
                updated = max(0.0, min(0.92, previous + secondary))
                applied = max(0.0, updated - previous)
                if applied <= 0.0:
                    continue
                target["observed_attenuation_ratio"] = updated
                target["coupling_extra"] = max(0.0, updated - float(target["local_response_estimate"]))
                target["secondary_cascade_extra"] = float(target.get("secondary_cascade_extra") or 0.0) + applied
                target["coupling_hop_depth"] = max(2, int(target.get("coupling_hop_depth") or 0))
                roles = set(target.get("coupling_roles") or [])
                roles.discard("unchanged_peak")
                roles.add("cross_fiber_to_same_fiber_secondary_cascade_peak")
                target["coupling_roles"] = sorted(roles)
                cascade_sources = set(target.get("same_fiber_cascade_source_channels") or [])
                cascade_sources.add(intermediate_id)
                target["same_fiber_cascade_source_channels"] = sorted(cascade_sources)
                paths = set(target.get("secondary_coupling_paths") or [])
                paths.update(f"{origin_id}->{intermediate_id}->{target_id}" for origin_id in origin_ids)
                target["secondary_coupling_paths"] = sorted(paths)
                target["possible_cross_fiber_coupling"] = True
                target["possible_same_fiber_coupling"] = True
    return observed


def _simulated_array_channels(scenario: str, step: int = 0, coupling_view: str = "raw_coupled_response") -> list[dict]:
    coords = {
        channel: (
            float((BAYSPEC_CHANNEL_CONFIG.get("channels", {}).get(channel, {}) or {}).get("x", 0)),
            float((BAYSPEC_CHANNEL_CONFIG.get("channels", {}).get(channel, {}) or {}).get("y", 0)),
        )
        for channel in CHANNEL_ORDER
    }
    aliases = {
        "left_press": "vertical_slide_p11_p12_p13",
        "right_press": "horizontal_slide_p11_p21_p31",
        "diagonal_slide": "diagonal_slide_p11_p22_p33",
        "p22_center_press": "center_press",
        "multi_point_press": "broad_fingertip_contact",
    }
    scenario = aliases.get(scenario, scenario)
    envelope = _demo_envelope(scenario, step)
    if scenario == "no_contact":
        centers = [(0.0, 0.0, 0.0, 0.52)]
    elif scenario in _STAGED_POINT_CONTACT_CENTERS:
        cx, cy = _STAGED_POINT_CONTACT_CENTERS[scenario]
        centers = [(cx, cy, envelope, 0.52)]
    elif scenario == "off_center_fingertip_contact":
        centers = [(0.52, 0.28, 0.84 * envelope, 0.58)]
    elif scenario == "vertical_slide_p11_p12_p13":
        path = [(-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0)]
        x, y = _path_point(path, step, cycle_length=12)
        centers = [(x, y, envelope, 0.52)]
    elif scenario == "horizontal_slide_p11_p21_p31":
        path = [(-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]
        x, y = _path_point(path, step, cycle_length=12)
        centers = [(x, y, envelope, 0.52)]
    elif scenario == "diagonal_slide_p11_p22_p33":
        path = [(-1.0, 1.0), (0.0, 0.0), (1.0, -1.0)]
        x, y = _path_point(path, step, cycle_length=12)
        centers = [(x, y, envelope, 0.52)]
    elif scenario == "broad_fingertip_contact":
        centers = [(0.06, -0.18, 0.86 * envelope, 0.76)]
    elif scenario == "tap":
        centers = [(0.0, 0.0, envelope, 0.52)]
    elif scenario == "release":
        centers = [(0.0, 0.0, envelope, 0.52)]
    else:
        centers = [(0.0, 0.0, envelope, 0.52)]
    local_by_channel: dict[str, float] = {}
    for channel_id in CHANNEL_ORDER:
        x, y = coords[channel_id]
        local_response = 0.0
        for cx, cy, amp, sigma in centers:
            local_response += amp * math.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma)
            )
        local_by_channel[channel_id] = max(0.0, min(0.96, local_response))

    cfg = _coupling_config()
    shift_cfg = BAYSPEC_CHANNEL_CONFIG.get("wavelength_shift_demodulation", {}) or {}
    full_scale_pm = float(shift_cfg.get("visualization_full_scale_pm") or 500.0)
    channels: list[dict] = []
    for channel_id in CHANNEL_ORDER:
        x, y = coords[channel_id]
        response_ratio = local_by_channel[channel_id]
        baseline = SIMULATED_FBG_I0.get(channel_id, 42000.0)
        # Pure wavelength-shift demo: contact translates the FBG peak but does
        # not modulate its simulated height. Intensity remains spectral evidence.
        intensity = baseline
        target_wavelength = _simulated_target_wavelength(channel_id)
        delta_wavelength_pm = response_ratio * full_scale_pm
        shifted_wavelength = target_wavelength + delta_wavelength_pm / 1000.0
        channels.append(
            {
                "channel_id": channel_id,
                "display_name": channel_id,
                "enabled": True,
                "valid": True,
                "x": x,
                "y": y,
                "target_wavelength_nm": target_wavelength,
                "baseline_wavelength_nm": target_wavelength,
                "tracked_wavelength_nm": shifted_wavelength,
                "peak_wavelength_nm": shifted_wavelength,
                "delta_wavelength_nm": delta_wavelength_pm / 1000.0,
                "delta_wavelength_pm": delta_wavelength_pm,
                "absolute_shift_pm": delta_wavelength_pm,
                "shift_direction": "stable" if response_ratio < 0.02 else "red_shift",
                "wavelength_shift_response_ratio": response_ratio,
                "observed_wavelength_shift_response_ratio": response_ratio,
                "wavelength_tracking_method": "simulated_bragg_shift",
                "peak_pixel_index": None,
                "intensity_counts": intensity,
                "baseline_intensity_counts": baseline,
                "relative_intensity": intensity / max(baseline, 1e-12),
                "attenuation_ratio": 1.0 - intensity / max(baseline, 1e-12),
                "intensity_loss_db": -10.0
                * math.log10(max(intensity / max(baseline, 1e-12), 1e-12)),
                "observed_intensity_counts": intensity,
                "observed_relative_intensity": intensity / max(baseline, 1e-12),
                "observed_attenuation_ratio": 1.0 - intensity / max(baseline, 1e-12),
                "observed_loss_db": -10.0
                * math.log10(max(intensity / max(baseline, 1e-12), 1e-12)),
                "local_response_estimate": response_ratio,
                "coupling_extra": 0.0,
                "coupling_roles": ["mechanically_coupled_shift"]
                if response_ratio >= 0.02
                else [],
                "coupling_sources": [
                    "shared_elastomer_mechanical_coupling",
                    "fingertip_contact_area_coverage",
                ],
                "same_fiber_cascade_source_channels": [],
                "cross_fiber_mechanical_source_channels": [],
                "cross_fiber_mechanical_carrier_extra": 0.0,
                "secondary_cascade_extra": 0.0,
                "secondary_coupling_paths": [],
                "coupling_hop_depth": 0,
                "coupling_compensated": False,
                "coupling_status": "debug_independent_simulation"
                if coupling_view == "independent_ideal_response"
                else "uncalibrated_mechanically_coupled_wavelength_shift",
                "possible_cross_fiber_coupling": coupling_view != "independent_ideal_response",
                "possible_same_fiber_coupling": False,
                "local_response_estimate_available": coupling_view == "independent_ideal_response",
                "response_level": _response_level_from_shift_ratio(response_ratio),
                "response_basis": "simulated_bragg_wavelength_shift",
                "simulated_peak_height_mode": "fixed_at_baseline",
                "simulated_intensity_modulation_enabled": False,
                "response_interpretation": (
                    "simulated uncalibrated wavelength displacement; not force calibrated"
                ),
                "qa_status": "simulated",
                "qa_flags": [
                    "simulated_array_demo",
                    "simulated_mechanically_coupled_wavelength_shift",
                ],
                "response_value": response_ratio,
            }
        )
    return channels


def _generate_synthetic_9fbg_spectrum(channels: list[dict], frame_id: int, timestamp: float) -> dict:
    plan = _array_wavelength_plan()
    start_nm = float(plan["start_nm"]) - 1.5
    end_nm = float(plan["stop_nm"]) + 1.5
    # Ten-picometre raster prevents a translated narrow FBG peak from appearing
    # to change height merely because its center falls between coarse samples.
    points = int(round((end_nm - start_nm) / 0.01)) + 1
    wavelengths = [start_nm + (end_nm - start_nm) * i / (points - 1) for i in range(points)]
    counts: list[float] = []
    peaks = []
    dominant = max(
        channels,
        key=lambda item: float(item.get("wavelength_shift_response_ratio") or 0.0),
        default={},
    )
    dominant_channel = str(dominant.get("channel_id") or "P22")

    def broad_background(wavelength: float) -> float:
        return 580.0 + 70.0 * math.sin(wavelength * 2.2) + 35.0 * math.sin(
            wavelength * 7.7
        )

    # Precompute one fixed profile per FBG. The peak height and width are
    # invariant across demo frames; only its center wavelength is translated.
    # Blending from the static background to the channel I0 avoids a sloped
    # background or sub-grid peak center looking like intensity modulation.
    channel_profiles: list[dict] = []
    sigma = 0.050
    grid_step_nm = (end_nm - start_nm) / max(points - 1, 1)
    for channel in channels:
        channel_id = str(channel.get("channel_id") or "")
        target = float(
            channel.get("target_wavelength_nm")
            or _simulated_target_wavelength(channel_id)
        )
        shifted = float(channel.get("peak_wavelength_nm") or target)
        baseline = float(
            channel.get("baseline_intensity_counts")
            or SIMULATED_FBG_I0.get(channel_id, 42000.0)
        )
        nearest_index = max(
            0,
            min(points - 1, int(round((shifted - start_nm) / grid_step_nm))),
        )
        nearest_wavelength = wavelengths[nearest_index]
        sampled_shape = math.exp(
            -((nearest_wavelength - shifted) ** 2) / (2.0 * sigma * sigma)
        )
        channel_profiles.append(
            {
                "shifted": shifted,
                "baseline": baseline,
                "sampled_shape": sampled_shape,
            }
        )

    for wavelength in wavelengths:
        value = broad_background(wavelength)
        for profile in channel_profiles:
            shifted = float(profile["shifted"])
            baseline = float(profile["baseline"])
            sampled_shape = max(float(profile["sampled_shape"]), 1e-12)
            shape = math.exp(
                -((wavelength - shifted) ** 2) / (2.0 * sigma * sigma)
            )
            blend = min(1.0, shape / sampled_shape)
            value += max(0.0, baseline - broad_background(wavelength)) * blend
        counts.append(max(80.0, value))

    for channel in channels:
        channel_id = str(channel.get("channel_id") or "")
        baseline = float(channel.get("baseline_intensity_counts") or SIMULATED_FBG_I0.get(channel_id, 42000.0))
        current = float(channel.get("intensity_counts") or baseline)
        relative = float(channel.get("relative_intensity") or current / max(baseline, 1e-12))
        attenuation = float(channel.get("attenuation_ratio") or max(0.0, 1.0 - relative))
        target = float(channel.get("target_wavelength_nm") or _simulated_target_wavelength(channel_id))
        shifted = float(channel.get("peak_wavelength_nm") or target)
        peaks.append(
            {
                "channel_id": channel_id,
                "target_wavelength_nm": target,
                "measured_wavelength_nm": channel.get("measured_wavelength_nm"),
                "demodulation_wavelength_nm": channel.get("demodulation_wavelength_nm") or target,
                "baseline_wavelength_nm": channel.get("baseline_wavelength_nm") or target,
                "tracked_wavelength_nm": shifted,
                "peak_wavelength_nm": shifted,
                "delta_wavelength_nm": channel.get("delta_wavelength_nm"),
                "delta_wavelength_pm": channel.get("delta_wavelength_pm"),
                "absolute_shift_pm": channel.get("absolute_shift_pm"),
                "shift_direction": channel.get("shift_direction"),
                "wavelength_shift_response_ratio": channel.get(
                    "wavelength_shift_response_ratio"
                ),
                "observed_wavelength_shift_response_ratio": channel.get(
                    "observed_wavelength_shift_response_ratio"
                ),
                "wavelength_tracking_method": channel.get("wavelength_tracking_method"),
                "intensity_counts": current,
                "baseline_intensity_counts": baseline,
                "relative_intensity": relative,
                "attenuation_ratio": attenuation,
                "intensity_loss_db": -10.0 * math.log10(max(relative, 1e-12)),
                "observed_intensity_counts": channel.get("observed_intensity_counts"),
                "observed_relative_intensity": channel.get("observed_relative_intensity"),
                "observed_attenuation_ratio": channel.get("observed_attenuation_ratio"),
                "observed_loss_db": channel.get("observed_loss_db"),
                "local_response_estimate": channel.get("local_response_estimate"),
                "coupling_extra": channel.get("coupling_extra"),
                "coupling_roles": channel.get("coupling_roles") or ["unchanged_peak"],
                "affected_role": (channel.get("coupling_roles") or ["unchanged_peak"])[0],
                "same_fiber_cascade_source_channels": channel.get("same_fiber_cascade_source_channels") or [],
                "cross_fiber_mechanical_source_channels": channel.get("cross_fiber_mechanical_source_channels") or [],
                "cross_fiber_mechanical_carrier_extra": channel.get("cross_fiber_mechanical_carrier_extra") or 0.0,
                "secondary_cascade_extra": channel.get("secondary_cascade_extra") or 0.0,
                "secondary_coupling_paths": channel.get("secondary_coupling_paths") or [],
                "coupling_hop_depth": channel.get("coupling_hop_depth") or 0,
                "coupling_status": channel.get("coupling_status") or "uncalibrated_dense_array_coupled_response",
                "coupling_sources": channel.get("coupling_sources") or COUPLING_SOURCE_DEFAULTS,
                "possible_cross_fiber_coupling": bool(channel.get("possible_cross_fiber_coupling")),
                "possible_same_fiber_coupling": bool(channel.get("possible_same_fiber_coupling")),
                "local_response_estimate_available": bool(channel.get("local_response_estimate_available")),
                "qa_status": "simulated",
                "simulated": True,
                "dominant": channel_id == dominant_channel,
            }
        )

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "mode": "synthetic_simulated_9fbg_wavelength_shift_spectrum",
        "spectrum_type": "synthetic simulated wavelength-shift spectrum",
        "axis_type": "wavelength_nm",
        "wavelength_nm": [round(value, 4) for value in wavelengths],
        "intensity": [round(value, 2) for value in counts],
        "peaks": peaks,
        "dominant_channel": dominant_channel,
        "selected_channel": dominant_channel,
        "peak_height_mode": "fixed_per_channel",
        "intensity_modulation_enabled": False,
        "frame_render_semantics": "replace_previous_spectrum",
        "source_note": (
            "synthetic 9-FBG wavelength-shift spectrum with fixed peak heights; "
            "peak centers move and each frame replaces the prior spectrum; not real BaySpec data"
        ),
        "wavelength_plan": _array_wavelength_plan_payload(),
    }


def simulated_array_frame(scenario: str, step: int = 0, coupling_view: str = "raw_coupled_response") -> dict:
    surface_cfg = BAYSPEC_CHANNEL_CONFIG.get("surface", {})
    coupling_view = coupling_view if coupling_view in {"raw_coupled_response", "independent_ideal_response", "coupling_compensated_response"} else "raw_coupled_response"
    if coupling_view == "coupling_compensated_response":
        coupling_view = "raw_coupled_response"
    surface_input_mode = "coupling_compensated_surface" if coupling_view == "independent_ideal_response" else "raw_coupled_response_surface"
    config = SurfaceConfig(
        grid_size=int(surface_cfg.get("surface_grid_size") or 80),
        sigma=float(surface_cfg.get("surface_sigma") or 0.65),
        clip_min=float(surface_cfg.get("surface_clip_min") or 0.0),
        clip_max=float(surface_cfg.get("surface_clip_max") or 1.0),
        active_threshold=float(surface_cfg.get("active_threshold") or 0.05),
        active_absolute_threshold=float(surface_cfg.get("active_absolute_threshold") or 0.10),
        active_relative_threshold=float(surface_cfg.get("active_relative_threshold") or 0.25),
        surface_input_mode=surface_input_mode,
    )
    scenario_aliases = {
        "left_press": "vertical_slide_p11_p12_p13",
        "right_press": "horizontal_slide_p11_p21_p31",
        "diagonal_slide": "diagonal_slide_p11_p22_p33",
        "p22_center_press": "center_press",
        "multi_point_press": "broad_fingertip_contact",
    }
    scenario = scenario_aliases.get(scenario, scenario)
    scenario_labels = {
        "center_press": "Center fingertip contact",
        "p21_contact": "P21 fingertip contact",
        "p12_contact": "P12 fingertip contact",
        "p32_contact": "P32 fingertip contact",
        "off_center_fingertip_contact": "Off-center fingertip contact",
        "vertical_slide_p11_p12_p13": "Vertical fingertip slide",
        "horizontal_slide_p11_p21_p31": "Horizontal fingertip slide",
        "diagonal_slide_p11_p22_p33": "Diagonal fingertip slide",
        "broad_fingertip_contact": "Broad fingertip contact",
        "tap": "Fingertip tap",
        "release": "Fingertip release",
        "no_contact": "no contact",
    }
    timestamp = time.time()
    frame_id = int(timestamp * 1000)
    cfg = _coupling_config()
    channels = _simulated_array_channels(scenario, step=step, coupling_view=coupling_view)
    surface = map_surface(channels, config=config)
    matrices = matrices_from_channels(channels)
    spectrum = _generate_synthetic_9fbg_spectrum(channels, frame_id=frame_id, timestamp=timestamp)
    dominant_channel = spectrum.get("dominant_channel") or surface["surface_metrics"].get("dominant_channel") or "P22"
    selected = next((channel for channel in channels if channel.get("channel_id") == dominant_channel), channels[0])
    display_label = scenario_labels.get(scenario, scenario)
    changed_channels = [
        channel
        for channel in channels
        if float(channel.get("wavelength_shift_response_ratio") or 0.0)
        >= float(surface_cfg.get("active_threshold") or 0.02)
    ]
    primary_observed_channel = surface["surface_metrics"].get("primary_observed_channel") or dominant_channel
    secondary_observed_channels = [channel.get("channel_id") for channel in changed_channels if channel.get("channel_id") != primary_observed_channel]
    possible_cross_fiber = any(bool(channel.get("possible_cross_fiber_coupling")) for channel in changed_channels)
    possible_same_fiber = any(bool(channel.get("possible_same_fiber_coupling")) for channel in changed_channels)
    secondary_coupling_paths = sorted(
        {
            path
            for channel in channels
            for path in (channel.get("secondary_coupling_paths") or [])
        }
    )
    max_coupling_hop_depth = max(
        [int(channel.get("coupling_hop_depth") or 0) for channel in channels],
        default=0,
    )
    local_estimate_available = coupling_view == "independent_ideal_response"
    peak_shift_response = max(
        [float(channel.get("wavelength_shift_response_ratio") or 0.0) for channel in channels],
        default=0.0,
    )
    peak_wavelength_shift_pm = max(
        [float(channel.get("absolute_shift_pm") or 0.0) for channel in channels],
        default=0.0,
    )
    if scenario == "tap":
        contact_patch_semantics = "short-lived single-fingertip contact patch"
    elif "slide" in scenario:
        contact_patch_semantics = "moving single-fingertip contact patch"
    elif scenario == "broad_fingertip_contact":
        contact_patch_semantics = "broad single-fingertip contact patch"
    elif scenario in {*_STAGED_POINT_CONTACT_CENTERS, "off_center_fingertip_contact"}:
        contact_patch_semantics = "single-fingertip contact patch"
    else:
        contact_patch_semantics = "no contact or release state"
    active_threshold = float(surface_cfg.get("active_threshold") or 0.02)
    responding_channel_ids = [str(channel.get("channel_id")) for channel in changed_channels if channel.get("channel_id")]
    if scenario == "release" and (peak_shift_response >= active_threshold or responding_channel_ids):
        contact_patch_semantics = "releasing single-fingertip contact patch"
    if scenario == "tap":
        if peak_shift_response >= 0.35:
            event_interpretation = "tap_peak"
        elif peak_shift_response >= active_threshold or responding_channel_ids:
            event_interpretation = "tap_decay"
        else:
            event_interpretation = "no_contact_after_tap"
    elif scenario == "release":
        event_interpretation = (
            "release_decay"
            if peak_shift_response >= active_threshold or responding_channel_ids
            else "no_contact_after_release"
        )
    elif scenario == "no_contact" or not responding_channel_ids:
        event_interpretation = "no_contact / baseline / no active contact"
    elif "slide" in scenario:
        event_interpretation = "single-finger contact patch sliding across coupled array"
    elif scenario == "broad_fingertip_contact":
        event_interpretation = "broad single-finger contact patch covering multiple pixels"
    elif scenario in {*_STAGED_POINT_CONTACT_CENTERS, "off_center_fingertip_contact"}:
        event_interpretation = "single-finger contact patch with coupled neighboring pixels"
    else:
        event_interpretation = (
            surface["surface_metrics"].get("event_interpretation")
            or "raw coupled Bragg wavelength-shift response"
        )
    surface["surface_metrics"].update(
        {
            "enabled_channel_count": len([channel for channel in channels if channel.get("enabled")]),
            "responding_channel_count": len(responding_channel_ids),
            "responding_channel_ids": responding_channel_ids,
            "active_channel_count": len(responding_channel_ids),
            "event_interpretation": event_interpretation,
            "secondary_coupling_paths": secondary_coupling_paths,
            "max_coupling_hop_depth": max_coupling_hop_depth,
        }
    )
    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "surface_frame_id": frame_id,
        "spectrum_frame_id": frame_id,
        "trace_frame_id": frame_id,
        "frame_sync_status": "synced",
        "last_update_timestamp": timestamp,
        "mode": "simulated_array_demo",
        "scenario": scenario,
        "scenario_label": display_label,
        "display_label": display_label,
        "step": step,
        "surface_display_mode": "surface",
        "surface_input_mode": surface_input_mode,
        "coupling_view": coupling_view,
        "coupling_status": "debug_independent_simulation"
        if coupling_view == "independent_ideal_response"
        else "uncalibrated_mechanically_coupled_wavelength_shift",
        "coupling_compensated": False,
        "coupling_sources": [
            "shared_elastomer_mechanical_coupling",
            "fingertip_contact_area_coverage",
        ],
        "coupling_model_note": (
            "Wavelength-shift simulation uses contact-area coverage and shared-elastomer "
            "mechanical coupling only; it does not reuse intensity attenuation cascades."
        ),
        "observed_changed_channels": [channel.get("channel_id") for channel in changed_channels],
        "primary_observed_channel": primary_observed_channel,
        "secondary_observed_channels": secondary_observed_channels,
        "possible_cross_fiber_coupling": possible_cross_fiber,
        "possible_same_fiber_coupling": possible_same_fiber,
        "secondary_coupling_paths": secondary_coupling_paths,
        "max_coupling_hop_depth": max_coupling_hop_depth,
        "local_response_estimate_available": local_estimate_available,
        "array_mode": "simulated_demo_only",
        "array_status": "simulated only, not real data",
        "wavelength_plan": _array_wavelength_plan_payload(),
        "dominant_channel": dominant_channel,
        "peak_attenuation": peak_shift_response,
        "peak_shift_response": peak_shift_response,
        "peak_wavelength_shift_pm": peak_wavelength_shift_pm,
        "contact_patch_semantics": contact_patch_semantics,
        "selected_channel": dominant_channel,
        "channels": channels,
        "spectrum": spectrum,
        "trace": [
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "channel_id": dominant_channel,
                "intensity_counts": selected.get("intensity_counts"),
                "baseline_intensity_counts": selected.get("baseline_intensity_counts"),
                "relative_intensity": selected.get("relative_intensity"),
                "attenuation_ratio": selected.get("attenuation_ratio"),
                "tracked_wavelength_nm": selected.get("tracked_wavelength_nm"),
                "baseline_wavelength_nm": selected.get("baseline_wavelength_nm"),
                "delta_wavelength_pm": selected.get("delta_wavelength_pm"),
                "absolute_shift_pm": selected.get("absolute_shift_pm"),
                "shift_direction": selected.get("shift_direction"),
                "wavelength_shift_response_ratio": selected.get(
                    "wavelength_shift_response_ratio"
                ),
            }
        ],
        "array_response_3x3": matrices["array_response_3x3"],
        "array_quality_3x3": matrices["array_quality_3x3"],
        "valid_channel_mask_3x3": matrices["valid_channel_mask_3x3"],
        "surface_grid": surface["surface_grid"],
        "grid_x": surface["grid_x"],
        "grid_y": surface["grid_y"],
        "surface_metrics": surface["surface_metrics"],
        "response_band_thresholds": _operator_response_band_thresholds(),
        "surface_note": (
            "simulated mechanically coupled wavelength-shift response, not real measured array data"
        ),
        "surface_title": "Raw coupled Bragg wavelength-shift surface",
        "surface_subtitle": (
            "Simulated uncalibrated wavelength displacement. Strain and temperature are not decoupled."
        ),
    }


app = FastAPI(title="TOUCH System Trained Static Spectrum Twin", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


@app.on_event("startup")
def startup_reference_sources() -> None:
    if PX6D_REFERENCE_CONFIG.get("enabled", True) and PX6D_REFERENCE_CONFIG.get(
        "auto_start", True
    ):
        px6d_reader.start()


@app.on_event("shutdown")
def shutdown_live_sources() -> None:
    optical_force_capture.stop()
    sdk_live_reader.stop()
    export_watcher.stop()
    px6d_reader.stop()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "app": "TOUCH System Trained Static Spectrum Twin",
        "mode": "standalone_bayspec_trained_static_spectrum_twin",
        "previous_p22_pd_voltage_app": "kept_separate",
        "optical_intensity_edition": "kept_separate",
        "demodulation_mode": "trained_static_full_spectrum_classifier",
        "recognition_scope": "manual_fingertip_static_spectrum_position_and_level",
        "primary_signal": "512_point_bayspec_full_spectrum_plus_stable_recovery_baseline",
        "diagnostic_spectrum_scope": "global_9fbg_wavelength_intensity_area_shape",
        "carrier_channel_id": "P22",
        "carrier_channel_role": "full_spectrum_transport_for_trained_model",
        "temperature_strain_decoupled": False,
        "array_mode": "global_spectrum_unmapped",
        "physical_channel_mapping_final": False,
        "real_3x3_enabled": False,
        "trained_static_model_primary": True,
        "default_operator_recognition": "dynamic_temporal_v3_validation",
        "dynamic_temporal_validation_primary": True,
        "static_spectral_fallback_available": STATIC_SPECTRAL_PREDICTOR is not None,
        "position_output_semantics": "approximate_manual_fingertip_contact_region",
        "response_level_semantics": "approximate_manual_light_normal_hard_not_force_N",
        "response_band_thresholds": _operator_response_band_thresholds(),
        "not_pd_voltage": True,
        "calibrated_physical_output": False,
        "trained_static_spectral_model": _static_spectral_model_status(),
        "dynamic_temporal_shadow": _dynamic_temporal_shadow_status(),
        "array_wavelength_plan": _array_wavelength_plan_payload(),
        "ui_style": "lab_light_digital_twin_like_previous_app",
        "status": bridge.status(),
        "export_watcher": export_watcher.status(),
        "sdk_live": sdk_live_reader.status(),
        "sense_control": sense_controller.status(),
        "px6d_reference": px6d_reader.status(),
        "optical_force_capture": optical_force_capture.status(),
    }


@app.get("/api/status")
def status() -> dict:
    result = bridge.status()
    result["px6d_reference"] = px6d_reader.status()
    result["optical_force_capture"] = optical_force_capture.status()
    return result


@app.get("/api/px6d/status")
def px6d_status() -> dict:
    return {"ok": True, "mode": "px6d_reference_force", **px6d_reader.status()}


@app.post("/api/px6d/start")
def px6d_start() -> dict:
    status_payload = px6d_reader.start()
    return {"ok": True, "mode": "px6d_reference_force", **status_payload}


@app.post("/api/px6d/stop")
def px6d_stop() -> dict:
    status_payload = px6d_reader.stop()
    return {"ok": True, "mode": "px6d_reference_force", **status_payload}


@app.post("/api/px6d/tare")
def px6d_tare(duration_sec: float = Query(default=1.0, ge=0.25, le=5.0)) -> dict:
    result = px6d_reader.tare(duration_sec=duration_sec)
    result.update(
        {
            "mode": "px6d_software_tare",
            "hardware_calibration_command_used": False,
        }
    )
    return result


@app.get("/api/px6d/latest")
def px6d_latest() -> dict:
    result = px6d_reader.latest()
    result["sample_ready"] = bool(result.pop("ok", False))
    result["ok"] = True
    result["mode"] = "px6d_reference_force"
    return result


@app.get("/api/px6d/trace")
def px6d_trace(limit: int = Query(default=500, ge=1, le=20000)) -> dict:
    result = px6d_reader.trace(limit=limit)
    result["mode"] = "px6d_reference_force_trace"
    return result


@app.get("/api/px6d_capture/status")
def px6d_capture_status() -> dict:
    return {"ok": True, "mode": "optical_px6d_synchronized_capture", **optical_force_capture.status()}


@app.post("/api/px6d_capture/start")
async def px6d_capture_start(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    result = optical_force_capture.start(
        position_label=str(payload.get("position_label") or "unlabeled"),
        action_label=str(payload.get("action_label") or "unlabeled"),
        trial_id=str(payload.get("trial_id") or "trial_001"),
        operator_note=str(payload.get("operator_note") or ""),
        output_root=payload.get("output_root"),
        selected_outputs=payload.get("selected_outputs"),
    )
    return {"mode": "optical_px6d_synchronized_capture", **result}


@app.post("/api/px6d_capture/stop")
def px6d_capture_stop() -> dict:
    result = optical_force_capture.stop()
    return {"mode": "optical_px6d_synchronized_capture", **result}


@app.get("/api/shadow/session_level_calibration")
def shadow_session_level_calibration_status() -> dict:
    current_token, pair = _current_runtime_baseline_token()
    status_payload = _session_level_calibration_status()
    status_payload.update(
        {
            "ok": True,
            "current_baseline_ready": current_token is not None,
            "current_baseline_token": current_token,
            "baseline_matches": bool(
                current_token is not None
                and status_payload.get("baseline_token") == current_token
            ),
            "baseline_status": pair.get("baseline_spectrum_status"),
        }
    )
    return status_payload


@app.post("/api/shadow/session_level_calibration")
async def load_shadow_session_level_calibration(request: Request) -> dict:
    global STATIC_SPECTRAL_SESSION_CALIBRATOR
    global STATIC_SPECTRAL_SESSION_CALIBRATION_SOURCE

    try:
        request_payload = await request.json()
    except Exception:
        return {"ok": False, "status": "request_body_must_be_json"}
    payload = request_payload.get("calibration", request_payload)
    try:
        calibrator = PerPositionOrdinalCalibrator.from_dict(payload)
    except Exception as exc:
        return {
            "ok": False,
            "status": "invalid_session_calibration_payload",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    missing_positions = sorted(
        set(CALIBRATION_POSITION_ORDER) - set(calibrator.anchors)
    )
    if missing_positions:
        return {
            "ok": False,
            "status": "incomplete_position_calibration",
            "missing_positions": missing_positions,
        }
    current_token, pair = _current_runtime_baseline_token()
    if current_token is None:
        return {
            "ok": False,
            "status": "current_runtime_baseline_required",
            "baseline_status": pair.get("baseline_spectrum_status"),
        }
    if calibrator.baseline_token != current_token:
        return {
            "ok": False,
            "status": "calibration_baseline_token_mismatch",
            "calibration_baseline_token": calibrator.baseline_token,
            "current_baseline_token": current_token,
        }
    with STATIC_SPECTRAL_SESSION_CALIBRATION_LOCK:
        STATIC_SPECTRAL_SESSION_CALIBRATOR = calibrator
        STATIC_SPECTRAL_SESSION_CALIBRATION_SOURCE = {
            "loaded_at": time.time(),
            "source": request_payload.get("source") or "local_api",
            "trial_count": request_payload.get("trial_count"),
        }
    return {
        "ok": True,
        "status": "session_level_calibration_loaded_shadow_only",
        **_session_level_calibration_status(),
    }


@app.delete("/api/shadow/session_level_calibration")
def delete_shadow_session_level_calibration() -> dict:
    return _clear_session_level_calibration("manual_api_clear")


@app.post("/api/reset")
def reset(keep_baseline: bool = Query(default=True)) -> dict:
    result = bridge.reset(keep_baseline=keep_baseline)
    result["dynamic_temporal_shadow_reset"] = _reset_dynamic_temporal_shadow(
        "api_reset"
    )
    if not keep_baseline:
        result["session_level_calibration_reset"] = (
            _clear_session_level_calibration("baseline_reset")
        )
    result.update({"mode": "bayspec_wavelength_shift_reset"})
    return result


@app.post("/api/ingest")
async def ingest(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "reason": "request body must be JSON"}
    result = bridge.ingest(payload)
    result.update({"mode": "bayspec_wavelength_shift_json_ingest"})
    return result


@app.post("/api/ingest_csv")
async def ingest_csv(
    request: Request,
    channel_id: str = Query(default="P22"),
    device_id: str = Query(default="F1871328"),
) -> dict:
    body = await request.body()
    text = body.decode("utf-8", errors="ignore")
    result = bridge.ingest_csv_text(text, channel_id=channel_id, device_id=device_id)
    result.update({"mode": "bayspec_wavelength_shift_csv_ingest"})
    return result


@app.post("/api/ingest_latest_export")
def ingest_latest_export(
    channel_id: str = Query(default="P22"),
    export_root: str | None = Query(default=None),
) -> dict:
    result = bridge.ingest_latest_export(root=export_root, channel_id=channel_id)
    result.update({"mode": "bayspec_sense_export_ingest_once"})
    return result


def _begin_acquisition_session() -> dict:
    """Clear held spectra and baselines before a new live source session.

    The deployed classifier is conditioned on a stable no-contact recovery
    baseline from the current acquisition session. Reusing a baseline across
    reconnects, source switches, or remounts would make otherwise valid model
    output scientifically ambiguous.
    """

    calibration_reset = _clear_session_level_calibration(
        "new_acquisition_session"
    )
    dynamic_shadow_reset = _reset_dynamic_temporal_shadow(
        "new_acquisition_session"
    )
    result = bridge.reset(keep_baseline=False)
    return {
        **result,
        "baseline_invalidated": True,
        "baseline_requirement": "stable_current_session_post_release_recovery",
        "session_level_calibration_reset": calibration_reset,
        "dynamic_temporal_shadow_reset": dynamic_shadow_reset,
    }


@app.post("/api/export_watch/start")
def export_watch_start(
    channel_id: str = Query(default="P22"),
    export_root: str | None = Query(default=None),
    interval_sec: float = Query(default=0.35, ge=0.1, le=5.0),
) -> dict:
    sdk_status = sdk_live_reader.stop()
    session_reset = _begin_acquisition_session()
    status = export_watcher.start(channel_id=channel_id, export_root=export_root, interval_sec=interval_sec)
    return {
        "ok": True,
        "mode": "sense_export_watch_started",
        "export_watcher": status,
        "sdk_live": sdk_status,
        "acquisition_session_reset": session_reset,
    }


@app.post("/api/export_watch/stop")
def export_watch_stop() -> dict:
    status = export_watcher.stop()
    return {"ok": True, "mode": "sense_export_watch_stopped", "export_watcher": status}


@app.get("/api/export_watch/status")
def export_watch_status() -> dict:
    return {"ok": True, "mode": "sense_export_watch_status", "export_watcher": export_watcher.status()}


@app.post("/api/sdk/start")
def sdk_start(
    channel_id: str = Query(default="P22"),
    interval_ms: int = Query(default=100, ge=20, le=2000),
    integration: int = Query(default=40000, ge=1, le=10000000),
) -> dict:
    export_watcher.stop()
    session_reset = _begin_acquisition_session()
    status = sdk_live_reader.start(channel_id=channel_id, interval_ms=interval_ms, integration=integration)
    return {
        "ok": bool(status.get("active")),
        "mode": "bayspec_direct_sdk_started",
        "sdk_live": status,
        "acquisition_session_reset": session_reset,
        "live_signal_path": (
            "BaySpec USB20BS SDK helper -> spectrum frames -> Bragg peak tracking "
            "-> wavelength-shift digital twin"
        ),
    }


@app.post("/api/sdk/stop")
def sdk_stop() -> dict:
    status = sdk_live_reader.stop()
    return {"ok": True, "mode": "bayspec_direct_sdk_stopped", "sdk_live": status}


@app.get("/api/sdk/status")
def sdk_status() -> dict:
    return {"ok": True, "mode": "bayspec_direct_sdk_status", "sdk_live": sdk_live_reader.status()}


@app.get("/api/sense/status")
def sense_status() -> dict:
    return {"ok": True, "mode": "sense_window_control_status", "sense_control": sense_controller.status()}


@app.post("/api/sense/start_fast_recording")
def sense_start_fast_recording(ensure_stopped: bool = Query(default=True)) -> dict:
    result = sense_controller.start_fast_recording(ensure_stopped=ensure_stopped)
    return result


@app.post("/api/sense/stop")
def sense_stop() -> dict:
    return sense_controller.stop_scan()


@app.post("/api/live/start")
def live_start(
    channel_id: str = Query(default="P22"),
    export_root: str | None = Query(default=None),
    interval_sec: float = Query(default=0.1, ge=0.1, le=5.0),
    control_sense: bool = Query(default=True),
    source: str = Query(default="direct_sdk"),
) -> dict:
    if source == "direct_sdk":
        export_watcher.stop()
        session_reset = _begin_acquisition_session()
        sdk_status = sdk_live_reader.start(
            channel_id=channel_id,
            interval_ms=max(20, int(interval_sec * 1000)),
            integration=40000,
        )
        return {
            "ok": bool(sdk_status.get("active")),
            "mode": "bayspec_live_twin_started",
            "sdk_live": sdk_status,
            "acquisition_session_reset": session_reset,
            "sense_control": {"ok": True, "mode": "sense_control_not_required_for_direct_sdk"},
            "live_signal_path": (
                "BaySpec USB20BS SDK helper -> spectrum frames -> Bragg peak tracking "
                "-> wavelength-shift digital twin"
            ),
        }

    sdk_status = sdk_live_reader.stop()
    session_reset = _begin_acquisition_session()
    watch_status = export_watcher.start(channel_id=channel_id, export_root=export_root, interval_sec=interval_sec)
    sense_result = sense_controller.start_fast_recording(ensure_stopped=True) if control_sense else {
        "ok": True,
        "mode": "sense_control_skipped",
    }
    return {
        "ok": bool(watch_status.get("active")),
        "mode": "bayspec_live_twin_started",
        "export_watcher": watch_status,
        "sdk_live": sdk_status,
        "acquisition_session_reset": session_reset,
        "sense_control": sense_result,
        "live_signal_path": (
            "Sense fast recording export -> spectrum watcher -> Bragg peak tracking "
            "-> wavelength-shift digital twin"
        ),
    }


@app.post("/api/live/stop")
def live_stop(control_sense: bool = Query(default=True)) -> dict:
    sdk_status = sdk_live_reader.stop()
    watch_status = export_watcher.stop()
    sense_result = sense_controller.stop_scan() if control_sense else {"ok": True, "mode": "sense_control_skipped"}
    return {
        "ok": True,
        "mode": "bayspec_live_twin_stopped",
        "sdk_live": sdk_status,
        "export_watcher": watch_status,
        "sense_control": sense_result,
    }


@app.post("/api/baseline")
async def set_baseline(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    result = bridge.set_baseline(payload)
    if result.get("baseline_set") or result.get("static_model_spectrum_baseline_ready"):
        result["dynamic_temporal_shadow_reset"] = _reset_dynamic_temporal_shadow(
            "runtime_baseline_replaced"
        )
    result.update({"mode": "bayspec_wavelength_baseline_set"})
    return result


@app.post("/api/global_candidate_baseline")
def set_global_candidate_baseline(
    minimum_frames: int = Query(default=30, ge=3, le=500),
) -> dict:
    model_baseline = bridge.set_baseline(
        {
            "channel_id": "P22",
            "baseline_method": "frozen_baseline",
            "minimum_recent_samples": minimum_frames,
        }
    )
    model_baseline_ready = bool(
        model_baseline.get("static_model_spectrum_baseline_ready")
    )
    result = (
        bridge.set_global_candidate_baseline(minimum_frames=minimum_frames)
        if model_baseline_ready
        else {
            "ok": False,
            "reason": model_baseline.get("static_model_spectrum_baseline_status")
            or model_baseline.get("reason")
            or "static_model_spectrum_baseline_not_ready",
            "candidate_baseline_skipped": True,
        }
    )
    candidate_baseline_ok = bool(result.get("ok"))
    baseline_ready = candidate_baseline_ok and model_baseline_ready
    failure_reason = None
    if not candidate_baseline_ok:
        failure_reason = str(result.get("reason") or "global_candidate_baseline_not_ready")
    elif not model_baseline_ready:
        failure_reason = str(
            model_baseline.get("static_model_spectrum_baseline_status")
            or model_baseline.get("reason")
            or "static_model_spectrum_baseline_not_ready"
        )
    result.update(
        {
            "ok": baseline_ready,
            "message": (
                None
                if baseline_ready
                else f"Baseline not ready: {failure_reason}. Keep the sensor released and try again."
            ),
            "mode": "global_9fbg_candidate_display_baseline",
            "recognition_scope": "global_3x3_hybrid_spectral_fingerprint",
            "physical_channel_mapping_final": False,
            "formal_model_baseline": False,
            "candidate_display_baseline_ok": candidate_baseline_ok,
            "static_model_spectrum_baseline": {
                "ok": model_baseline_ready,
                "baseline_set": bool(model_baseline.get("baseline_set")),
                "role": "post_press_release_recovery_no_contact_full_spectrum_baseline",
                "status": model_baseline.get("static_model_spectrum_baseline_status"),
                "sample_count": (
                    model_baseline.get("baseline_spectrum_sample_count_by_channel", {})
                    or {}
                ).get("P22"),
                "span_sec": (
                    model_baseline.get("baseline_spectrum_span_sec_by_channel", {})
                    or {}
                ).get("P22"),
                "noise_ratio": (
                    model_baseline.get("baseline_spectrum_noise_ratio_by_channel", {})
                    or {}
                ).get("P22"),
                "drift_ratio": (
                    model_baseline.get("baseline_spectrum_drift_ratio_by_channel", {})
                    or {}
                ).get("P22"),
                "reason": (
                    None
                    if model_baseline_ready
                    else model_baseline.get("static_model_spectrum_baseline_status")
                    or model_baseline.get("reason")
                ),
            },
        }
    )
    if baseline_ready:
        result["session_level_calibration_reset"] = (
            _clear_session_level_calibration("runtime_baseline_replaced")
        )
        result["dynamic_temporal_shadow_reset"] = (
            _reset_dynamic_temporal_shadow("runtime_baseline_replaced")
        )
    return result


@app.get("/api/latest")
def latest(
    channel_id: str | None = Query(default=None),
    include_spectrum: bool = Query(default=False),
) -> dict:
    return {
        "ok": True,
        "mode": "bayspec_wavelength_shift_latest",
        "channel_id": channel_id,
        "latest": bridge.latest(channel_id=channel_id, include_spectrum=include_spectrum),
    }


@app.get("/api/trace")
def trace(
    channel_id: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=20000),
    include_spectrum: bool = Query(default=False),
) -> dict:
    result = bridge.trace(channel_id=channel_id, limit=limit, include_spectrum=include_spectrum)
    result.update({"mode": "bayspec_wavelength_shift_trace"})
    return result


@app.get("/api/frame")
def frame(
    channel_id: str = Query(default="P22"),
    trace_limit: int = Query(default=600, ge=1, le=20000),
    include_spectrum: bool = Query(default=True),
) -> dict:
    result = bridge.frame(channel_id=channel_id, trace_limit=trace_limit, include_spectrum=include_spectrum)
    result["export_watcher"] = export_watcher.status()
    result["sdk_live"] = sdk_live_reader.status()
    result["sense_control"] = sense_controller.status()
    result["response_band_thresholds"] = _operator_response_band_thresholds()
    result["px6d_reference"] = _px6d_reference_for_record(result.get("latest"))
    result["px6d_status"] = px6d_reader.status()
    result["optical_force_capture"] = optical_force_capture.status()
    return result


@app.get("/api/global_spectrum_frame")
def global_spectrum_frame(
    trace_limit: int = Query(default=8, ge=1, le=20000),
    include_spectrum: bool = Query(default=True),
    include_shadow: bool = Query(default=False),
    include_dynamic_shadow: bool = Query(default=False),
    temporal_validation_mode: bool = Query(default=False),
) -> dict:
    """Expose one full-spectrum carrier frame for joint nine-FBG processing.

    The hardware reader currently stores the raw spectrum on a legacy P22
    carrier record. This endpoint makes that transport detail explicit and
    prevents global recognition clients from treating P22 as their input scope.
    """

    temporal_validation_enabled = temporal_validation_mode is True
    watcher_status = export_watcher.status()
    sdk_status = sdk_live_reader.status()
    result = bridge.frame(
        channel_id="P22",
        trace_limit=trace_limit,
        include_spectrum=include_spectrum,
    )
    latest = result.get("latest")
    source_gate = _model_display_source_gate(latest, watcher_status, sdk_status)
    global_candidate_summary = {
        "valid_candidate_count": 0,
        "expected_candidate_count": 9,
        "dominant_candidate_id": None,
        "peak_absolute_shift_pm": None,
        "candidate_reference_status": "provisional_no_contact_reference",
        "physical_channel_mapping_final": False,
    }
    global_frame_qa = {
        "candidate_contract_complete": False,
        "baseline_ready": False,
        "source_fresh": bool(source_gate["source_fresh"]),
        "live_source_active": source_gate["live_source_active"],
        "selected_live_source": source_gate["selected_live_source"],
        "model_input_source_allowed": source_gate["model_input_source_allowed"],
        "model_input_source_mode": source_gate["model_input_source_mode"],
        "frame_age_sec": None,
        "display_available": False,
        "formal_recognition_allowed": False,
        "blockers": ["no_global_spectrum_frame"],
    }
    if isinstance(latest, dict):
        latest = dict(latest)
        candidate_peaks = [
            peak
            for peak in (latest.get("spectrum_peaks") or [])
            if isinstance(peak, dict) and peak.get("candidate_mapping")
        ]
        valid_candidates = [peak for peak in candidate_peaks if peak.get("valid") is True]
        dominant_candidate = max(
            valid_candidates,
            key=lambda peak: float(peak.get("candidate_absolute_shift_pm") or 0.0),
            default=None,
        )
        candidate_statuses = {
            str(peak.get("candidate_reference_status") or "unknown")
            for peak in valid_candidates
        }
        baseline_ready = (
            len(valid_candidates) == 9
            and candidate_statuses == {"session_global_no_contact_baseline"}
        )
        received_at = latest.get("ingested_at")
        try:
            frame_age_sec = max(0.0, time.time() - float(received_at))
        except (TypeError, ValueError):
            frame_age_sec = None
        source_fresh = bool(source_gate["source_fresh"])
        candidate_contract_complete = (
            len(candidate_peaks) == 9
            and len(valid_candidates) == 9
            and [peak.get("candidate_id") for peak in candidate_peaks]
            == [f"FBG{index:02d}" for index in range(1, 10)]
        )
        blockers: list[str] = []
        if not candidate_contract_complete:
            blockers.append("incomplete_or_invalid_global_candidate_set")
        if not baseline_ready:
            blockers.append("global_candidate_baseline_not_ready")
        if source_gate["live_source_active"] and not source_fresh:
            blockers.append("stale_or_cached_global_frame")
        response_allowed = candidate_contract_complete and baseline_ready and source_fresh
        blockers.extend(
            [
                "physical_p11_p33_mapping_not_approved",
                "deployable_global_model_not_attached",
            ]
        )
        global_candidate_summary.update(
            {
                "valid_candidate_count": len(valid_candidates),
                "dominant_candidate_id": (
                    dominant_candidate.get("candidate_id")
                    if dominant_candidate is not None
                    else None
                ),
                "peak_absolute_shift_pm": (
                    dominant_candidate.get("candidate_absolute_shift_pm")
                    if dominant_candidate is not None
                    else None
                ),
                "candidate_reference_status": (
                    next(iter(candidate_statuses))
                    if len(candidate_statuses) == 1
                    else "mixed_or_incomplete_reference"
                ),
                "baseline_ready": baseline_ready,
                "responding_candidate_count": (
                    sum(
                        1
                        for peak in valid_candidates
                        if float(peak.get("candidate_absolute_shift_pm") or 0.0) >= 10.0
                    )
                    if response_allowed
                    else 0
                ),
                "response_allowed": response_allowed,
            }
        )
        global_frame_qa = {
            "candidate_contract_complete": candidate_contract_complete,
            "baseline_ready": baseline_ready,
            "source_fresh": source_fresh,
            "live_source_active": source_gate["live_source_active"],
            "selected_live_source": source_gate["selected_live_source"],
            "model_input_source_allowed": source_gate["model_input_source_allowed"],
            "model_input_source_mode": source_gate["model_input_source_mode"],
            "frame_age_sec": frame_age_sec,
            "display_available": bool(candidate_peaks),
            "response_allowed": response_allowed,
            "formal_recognition_allowed": False,
            "blockers": blockers,
        }
        latest.update(
            {
                "recognition_scope": "global_3x3_hybrid_spectral_fingerprint",
                "candidate_contract_version": "global_9fbg_candidate_frame_v1",
                "global_candidate_ids": [f"FBG{index:02d}" for index in range(1, 10)],
                "carrier_channel_id": "P22",
                "carrier_channel_role": "legacy_full_spectrum_transport_only",
                "physical_channel_mapping_final": False,
                "global_candidate_summary": global_candidate_summary,
                "source_fresh": source_fresh,
                "frame_age_sec": frame_age_sec,
                "response_allowed": response_allowed,
                "response_block_reason": next(
                    (
                        reason
                        for reason in (
                            "incomplete_or_invalid_global_candidate_set",
                            "global_candidate_baseline_not_ready",
                            "stale_or_cached_global_frame",
                        )
                        if reason in blockers
                    ),
                    None,
                ),
            }
        )
        result["latest"] = latest
    # The temporal validation model is the active recognizer in this mode.
    # Running the legacy static ensemble as well adds close to a second of
    # avoidable CPU work per live frame. Keep it available only when explicitly
    # requested for diagnostics or when static fallback is the selected mode.
    static_inference_requested = bool(not temporal_validation_enabled or include_shadow)
    if static_inference_requested:
        static_model_frame = _predict_static_spectral_frame(
            include_shadow=bool(include_shadow)
        )
    else:
        static_model_frame = {
            "ok": False,
            "status": "skipped_temporal_validation_mode",
            "reason": "dynamic_temporal_model_is_active",
            "inference_skipped": True,
        }
    static_prediction = static_model_frame.get("prediction") if static_model_frame.get("ok") else None
    static_shadow = (
        static_model_frame.get("shadow_candidate")
        if isinstance(static_model_frame.get("shadow_candidate"), dict)
        else None
    )
    static_shadow_prediction = (
        static_shadow.get("prediction")
        if static_shadow is not None and static_shadow.get("ok")
        else None
    )
    dynamic_requested = bool(include_dynamic_shadow is True or temporal_validation_enabled)
    if not dynamic_requested:
        dynamic_temporal_shadow = {
            "ok": False,
            "status": "dynamic_shadow_not_requested",
            "request_hint": "set include_dynamic_shadow=true for diagnostic validation",
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }
    elif not source_gate["model_input_source_allowed"]:
        dynamic_temporal_shadow = {
            "ok": False,
            "status": "dynamic_shadow_source_blocked",
            "reason": "stale_or_mismatched_live_source",
            "runtime_role": "shadow_only_not_driving_digital_twin",
            "drives_operator_ui": False,
            "drives_digital_twin": False,
        }
    else:
        dynamic_temporal_shadow = _predict_dynamic_temporal_shadow()
    dynamic_temporal_display = _dynamic_temporal_display_prediction(
        dynamic_temporal_shadow
    )
    static_model_assisted_display_allowed = bool(
        static_model_frame.get("ok") and source_gate["model_input_source_allowed"]
    )
    temporal_model_assisted_display_allowed = bool(
        temporal_validation_enabled
        and source_gate["model_input_source_allowed"]
        and dynamic_temporal_display.get("ok")
    )
    if temporal_validation_enabled:
        model_assisted_display_allowed = temporal_model_assisted_display_allowed
        active_spectral_prediction = dynamic_temporal_display.get("prediction")
        active_spectral_model_source = "dynamic_temporal_v3_validation"
        active_spectral_model_status = dynamic_temporal_display.get("status")
        active_spectral_model_expected = True
        active_spectral_model_loaded = DYNAMIC_TEMPORAL_SHADOW_ADAPTER is not None
        active_spectral_model_progress = {
            "history_frames": dynamic_temporal_display.get("history_frames"),
            "required_frames": dynamic_temporal_display.get("required_frames"),
        }
    else:
        model_assisted_display_allowed = static_model_assisted_display_allowed
        active_spectral_prediction = static_prediction
        active_spectral_model_source = "static_spectral_model"
        active_spectral_model_status = static_model_frame.get("status")
        active_spectral_model_expected = bool(
            _static_spectral_model_status().get("loaded")
        )
        active_spectral_model_loaded = active_spectral_model_expected
        active_spectral_model_progress = None
    if model_assisted_display_allowed:
        model_assisted_display_block_reason = None
    elif not source_gate["model_input_source_allowed"]:
        model_assisted_display_block_reason = "stale_or_mismatched_live_source"
    elif temporal_validation_enabled:
        model_assisted_display_block_reason = str(
            dynamic_temporal_display.get("reason")
            or dynamic_temporal_display.get("status")
            or "dynamic_temporal_model_not_ready"
        )
    elif static_model_frame.get("ok"):
        model_assisted_display_block_reason = "stale_or_mismatched_live_source"
    else:
        model_assisted_display_block_reason = str(
            static_model_frame.get("reason") or static_model_frame.get("status") or "model_not_ready"
        )
    if isinstance(latest, dict):
        latest["trained_static_spectral_prediction"] = static_prediction
        latest["trained_static_spectral_model_status"] = static_model_frame.get("status")
        latest["active_spectral_prediction"] = active_spectral_prediction
        latest["active_spectral_model_source"] = active_spectral_model_source
        latest["active_spectral_model_status"] = active_spectral_model_status
        latest["active_spectral_model_loaded"] = active_spectral_model_loaded
        latest["active_spectral_model_progress"] = active_spectral_model_progress
        result["latest"] = latest
    result.update(
        {
            "mode": "bayspec_global_9fbg_spectrum_frame",
            "scope": "global_3x3_hybrid_spectral_fingerprint",
            "selected_channel": None,
            "carrier_channel_id": "P22",
            "carrier_channel_role": "legacy_full_spectrum_transport_only",
            "candidate_contract_version": "global_9fbg_candidate_frame_v1",
            "global_candidate_ids": [f"FBG{index:02d}" for index in range(1, 10)],
            "physical_channel_mapping_final": False,
            "global_candidate_summary": global_candidate_summary,
            "global_frame_qa": global_frame_qa,
            "candidate_contract_complete": global_frame_qa.get(
                "candidate_contract_complete"
            ),
            "baseline_ready": global_frame_qa.get("baseline_ready"),
            "source_fresh": global_frame_qa.get("source_fresh"),
            "frame_age_sec": global_frame_qa.get("frame_age_sec"),
            "display_available": global_frame_qa.get("display_available"),
            "formal_recognition_allowed": global_frame_qa.get(
                "formal_recognition_allowed"
            ),
            "model_assisted_display_allowed": model_assisted_display_allowed,
            "model_assisted_display_block_reason": model_assisted_display_block_reason,
            "temporal_validation_mode": temporal_validation_enabled,
            "temporal_model_assisted_display_allowed": (
                temporal_model_assisted_display_allowed
            ),
            "static_model_assisted_display_allowed": (
                static_model_assisted_display_allowed
            ),
            "active_spectral_model_expected": active_spectral_model_expected,
            "active_spectral_model_loaded": active_spectral_model_loaded,
            "active_spectral_model_source": active_spectral_model_source,
            "active_spectral_model_status": active_spectral_model_status,
            "active_spectral_model_progress": active_spectral_model_progress,
            "active_spectral_prediction": active_spectral_prediction,
            "trained_static_spectral_model": _static_spectral_model_status(),
            "trained_static_spectral_frame": static_model_frame,
            "trained_static_spectral_prediction": static_prediction,
            "trained_static_spectral_shadow": static_shadow,
            "trained_static_spectral_shadow_prediction": static_shadow_prediction,
            "dynamic_temporal_shadow": dynamic_temporal_shadow,
            "dynamic_temporal_display": dynamic_temporal_display,
            "response_band_thresholds": _operator_response_band_thresholds(),
            "blockers": global_frame_qa.get("blockers", []),
            "export_watcher": watcher_status,
            "sdk_live": sdk_status,
            "sense_control": sense_controller.status(),
            "px6d_reference": _px6d_reference_for_record(result.get("latest")),
            "px6d_status": px6d_reader.status(),
            "optical_force_capture": optical_force_capture.status(),
        }
    )
    return result


@app.get("/api/thumb_scene_config")
def thumb_scene_config() -> dict:
    config = _thumb_scene_config()
    asset_url = str(config.get("thumb_holder_scene", {}).get("model_asset_url") or "")
    fallback_asset_url = str(config.get("thumb_holder_scene", {}).get("fallback_asset_url") or "")
    asset_path = None
    if asset_url.startswith("/static/"):
        asset_path = FRONTEND_ROOT / asset_url.removeprefix("/static/")
    fallback_asset_path = None
    if fallback_asset_url.startswith("/static/"):
        fallback_asset_path = FRONTEND_ROOT / fallback_asset_url.removeprefix("/static/")
    return {
        "ok": True,
        "mode": "thumb_holder_scene_config",
        "config": config,
        "config_path": str(THUMB_SCENE_CONFIG_PATH),
        "model_asset_url": asset_url,
        "model_asset_exists": bool(asset_path and asset_path.exists()),
        "model_asset_path": str(asset_path) if asset_path else None,
        "fallback_asset_url": fallback_asset_url,
        "fallback_asset_exists": bool(fallback_asset_path and fallback_asset_path.exists()),
        "fallback_asset_path": str(fallback_asset_path) if fallback_asset_path else None,
        "real_3x3_enabled": False,
        "force_N_output": False,
        "calibrated_physical_output": False,
    }


@app.post("/api/thumb_scene_config")
async def save_thumb_scene_config(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "reason": "request body must be JSON"}
    result = _write_thumb_scene_config(payload)
    result.update({"mode": "thumb_holder_scene_config_saved", "real_3x3_enabled": False, "force_N_output": False})
    return result


@app.get("/api/array_demo/frame")
def array_demo_frame(
    scenario: str = Query(default="center_press"),
    step: int = Query(default=0, ge=0, le=1000),
    coupling_view: str = Query(default="raw_coupled_response"),
) -> dict:
    frame = simulated_array_frame(scenario=scenario, step=step, coupling_view=coupling_view)
    return {
        "ok": True,
        "mode": "simulated_array_demo",
        "scenario": scenario,
        "coupling_view": coupling_view,
        "display_label": frame.get("display_label"),
        "dominant_channel": frame.get("dominant_channel"),
        "peak_attenuation": frame.get("peak_attenuation"),
        "peak_shift_response": frame.get("peak_shift_response"),
        "peak_wavelength_shift_pm": frame.get("peak_wavelength_shift_pm"),
        "contact_patch_semantics": frame.get("contact_patch_semantics"),
        "coupling_status": frame.get("coupling_status"),
        "surface_metrics": frame.get("surface_metrics"),
        "frame_sync_status": frame.get("frame_sync_status"),
        "array_frame": frame,
        "surface_grid": frame.get("surface_grid"),
        "surface_note": frame.get("surface_note"),
        "response_band_thresholds": _operator_response_band_thresholds(),
        "message": (
            "mechanically coupled wavelength-shift simulation, not real calibrated 3x3 data"
        ),
    }
