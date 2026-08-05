"""FastAPI backend for the TOUCH current optical runtime."""

from __future__ import annotations

import asyncio
import csv
import ctypes
import copy
from contextlib import asynccontextmanager
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


APP_ROOT = Path(os.environ.get("BAYSPEC_WAVELENGTH_APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
FRONTEND_ROOT = APP_ROOT / "frontend"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from bridge import bridge
from bridge import (
    BAYSPEC_CHANNEL_CONFIG,
    CHANNEL_CONFIG_PATH,
    CHANNEL_ORDER,
    configured_device_id,
    configured_sense_export_root,
    normalize_spectrum_to_baseline_ratio,
)
from backend.optical_force_capture import OpticalForceCaptureManager
from backend.mfbg_intensity_api import router as mfbg_intensity_router
from backend.px6d_reader import Px6dReader
from backend.recorded_demo import RecordedDemoLibrary
from sdk_live import (
    DEFAULT_INTEGRATION_US,
    DEFAULT_INTERVAL_MS,
    BaySpecSdkLiveReader,
)
from src.array_surface.surface_mapper import SurfaceConfig, map_surface, matrices_from_channels
from src.hybrid_spectrum.all_source_runtime_adapter import (
    AllSourceOpticalForceAdapter,
)
from src.hybrid_spectrum.measurement_consistency import (
    MeasurementAnalysisConfig,
    analyze_measurement_session,
    load_measurement_config,
    load_measurement_trace,
)
from src.hybrid_spectrum.measurement_estimate_sources import (
    EVIDENCE_SOURCES,
    resolve_measurement_estimate_evidence,
)

try:
    import yaml
except Exception:  # pragma: no cover - diagnostics fallback
    yaml = None


WM_COMMAND = 0x0111
IDOK = 1
IDYES = 6
SENSE_CMD_STOP = 32853
SENSE_CMD_FAST_RECORDING = 32879
MAX_MANUAL_INGEST_BODY_BYTES = 8 * 1024 * 1024


class IngestRequestTooLarge(ValueError):
    pass


async def _read_limited_ingest_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > MAX_MANUAL_INGEST_BODY_BYTES:
            raise IngestRequestTooLarge("manual ingest request exceeds 8 MiB")

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_MANUAL_INGEST_BODY_BYTES:
            raise IngestRequestTooLarge("manual ingest request exceeds 8 MiB")
    return bytes(body)
PROJECT_ROOT = APP_ROOT if getattr(sys, "frozen", False) else APP_ROOT.parent
CURRENT_RUNTIME_ENABLED = True
THUMB_SCENE_CONFIG_PATH = PROJECT_ROOT / "config" / "thumb_holder_scene.yaml"
HYBRID_SPECTRUM_CHANNEL_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml"
)
CURRENT_RUNTIME_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "deployed"
    / "ordinary_fbg_current_runtime.joblib"
)
RUNTIME_CONTACT_STATE_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "runtime_contact_state.yaml"
)
PX6D_REFERENCE_CONFIG_PATH = PROJECT_ROOT / "config" / "px6d_reference.yaml"
MEASUREMENT_ANALYSIS_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "measurement_analysis.yaml"
)
VERSION_PATH = PROJECT_ROOT / "VERSION.json"
LIVE_SOURCE_CONTROL_LOCK = threading.RLock()
CURRENT_RUNTIME_LOCK = threading.Lock()
CURRENT_RUNTIME_BASELINE_TOKEN: str | None = None
CURRENT_RUNTIME_LAST_FRAME_KEY: tuple[Any, ...] | None = None
CURRENT_RUNTIME_UNIQUE_FRAME_COUNT = 0
CURRENT_RUNTIME_LAST_PAYLOAD: dict[str, Any] | None = None
CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES: list[dict[str, Any]] = []
CURRENT_RUNTIME_STARTUP_BASELINE_LAST_FRAME_KEY: tuple[Any, ...] | None = None
CURRENT_RUNTIME_STARTUP_BASELINE_STATUS: dict[str, Any] = {
    "status": "not_started",
    "ready": False,
    "frame_count": 0,
}
GLOBAL_BASELINE_ATTESTATION: dict[str, Any] = {
    "confirmed": False,
    "attested_at_epoch_sec": None,
    "attested_by": None,
    "force_evidence": None,
    "status": "not_attested",
}


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256_file(path),
    }


def _release_identity() -> dict[str, Any]:
    try:
        version_payload = json.loads(VERSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        version_payload = {}

    source_commit = str(
        version_payload.get("source_commit")
        or os.environ.get("TOUCH_SOURCE_COMMIT")
        or ""
    ).strip()
    source_branch = str(
        version_payload.get("source_branch")
        or os.environ.get("TOUCH_SOURCE_BRANCH")
        or ""
    ).strip()
    if not source_commit and not getattr(sys, "frozen", False):
        try:
            source_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            source_commit = ""
    if not source_branch and not getattr(sys, "frozen", False):
        try:
            source_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            source_branch = ""

    return {
        "product": version_payload.get("product") or "TOUCH",
        "edition": version_payload.get("edition"),
        "version": version_payload.get("version") or "unknown",
        "build_id": version_payload.get("build_id") or "unknown",
        "release_date": version_payload.get("release_date"),
        "release_channel": version_payload.get("release_channel"),
        "source_commit": source_commit or None,
        "source_branch": source_branch or None,
        "source_worktree_state": version_payload.get("source_worktree_state")
        or ("packaged" if getattr(sys, "frozen", False) else "runtime_source_checkout"),
        "operator_recognition_model": version_payload.get(
            "operator_recognition_model"
        ),
        "temporal_candidate_role": version_payload.get("temporal_candidate_role"),
        "baseline_policy": version_payload.get("baseline_policy"),
        "capture_schema": version_payload.get("capture_schema"),
        "default_capture_directory": version_payload.get(
            "default_capture_directory"
        ),
        "hardware_validation_status": version_payload.get(
            "hardware_validation_status"
        ),
        "version_manifest": _artifact_identity(VERSION_PATH),
        "frozen_executable": bool(getattr(sys, "frozen", False)),
    }


RELEASE_IDENTITY = _release_identity()


def _optional_environment_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _apply_px6d_environment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    enabled = _optional_environment_bool("TOUCH_PX6D_ENABLED")
    auto_start = _optional_environment_bool("TOUCH_PX6D_AUTO_START")
    port = os.environ.get("TOUCH_PX6D_PORT")
    if enabled is not None:
        config["enabled"] = enabled
    if auto_start is not None:
        config["auto_start"] = auto_start
    if port and port.strip():
        config["port"] = port.strip()
    return config


def _load_px6d_reference_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": True,
        "auto_start": True,
        "port": "COM3",
        "auto_detect_port": True,
        "usb_vid": 0x1A86,
        "usb_pid": 0x55D3,
        "usb_serial_number": "5C7B025505",
        "port_description_contains": "CH343",
        "baud_rate": 921600,
        "device_id": 127,
        "poll_hz": 50.0,
        "read_timeout_sec": 0.20,
        "reconnect_interval_sec": 1.0,
        "reconnect_max_interval_sec": 8.0,
        "reconnect_backoff_multiplier": 2.0,
        "port_busy_backoff_sec": 5.0,
        "isolate_process": True,
        "worker_watchdog_sec": 1.5,
        "worker_stop_timeout_sec": 0.50,
        "device_settle_sec": 0.20,
        "handshake_timeout_sec": 0.40,
        "handshake_attempts": 3,
        "history_seconds": 300.0,
        "compression_sign": -1,
        "filter_alpha": 0.25,
        "median_window_samples": 5,
        "force_deadband_n": 0.015,
        "stationary_window_sec": 1.0,
        "stationary_std_max_n": 0.008,
        "stationary_range_max_n": 0.030,
        "stationary_slope_max_n_per_sec": 0.025,
        "auto_zero_drift_enabled": True,
        "auto_zero_hold_sec": 1.5,
        "auto_zero_capture_limit_n": 0.060,
        "auto_zero_release_reacquire_limit_n": 0.30,
        "auto_zero_alpha": 0.015,
        "maximum_drift_offset_n": 0.50,
        "auto_tare_on_start": False,
        "auto_tare_duration_sec": 1.0,
        "auto_tare_max_std_n": 0.12,
        "sync_window_sec": 0.25,
        "sync_max_age_sec": 0.25,
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
        return _apply_px6d_environment_overrides(defaults)
    try:
        payload = yaml.safe_load(
            PX6D_REFERENCE_CONFIG_PATH.read_text(encoding="utf-8")
        ) or {}
    except Exception:
        return _apply_px6d_environment_overrides(defaults)
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
            "auto_detect_port": serial_config.get(
                "auto_detect_port", defaults["auto_detect_port"]
            ),
            "usb_vid": serial_config.get("usb_vid", defaults["usb_vid"]),
            "usb_pid": serial_config.get("usb_pid", defaults["usb_pid"]),
            "usb_serial_number": serial_config.get(
                "usb_serial_number",
                serial_config.get("serial_number", defaults["usb_serial_number"]),
            ),
            "port_description_contains": serial_config.get(
                "port_description_contains",
                serial_config.get(
                    "description_contains",
                    defaults["port_description_contains"],
                ),
            ),
            "baud_rate": serial_config.get("baud_rate", defaults["baud_rate"]),
            "device_id": serial_config.get("device_id", defaults["device_id"]),
            "poll_hz": serial_config.get("poll_hz", defaults["poll_hz"]),
            "read_timeout_sec": serial_config.get(
                "read_timeout_sec", defaults["read_timeout_sec"]
            ),
            "reconnect_interval_sec": serial_config.get(
                "reconnect_interval_sec", defaults["reconnect_interval_sec"]
            ),
            "reconnect_max_interval_sec": serial_config.get(
                "reconnect_max_interval_sec",
                defaults["reconnect_max_interval_sec"],
            ),
            "reconnect_backoff_multiplier": serial_config.get(
                "reconnect_backoff_multiplier",
                defaults["reconnect_backoff_multiplier"],
            ),
            "port_busy_backoff_sec": serial_config.get(
                "port_busy_backoff_sec",
                defaults["port_busy_backoff_sec"],
            ),
            "isolate_process": serial_config.get(
                "isolate_process", defaults["isolate_process"]
            ),
            "worker_watchdog_sec": serial_config.get(
                "worker_watchdog_sec", defaults["worker_watchdog_sec"]
            ),
            "worker_stop_timeout_sec": serial_config.get(
                "worker_stop_timeout_sec",
                defaults["worker_stop_timeout_sec"],
            ),
            "device_settle_sec": serial_config.get(
                "device_settle_sec", defaults["device_settle_sec"]
            ),
            "handshake_timeout_sec": serial_config.get(
                "handshake_timeout_sec", defaults["handshake_timeout_sec"]
            ),
            "handshake_attempts": serial_config.get(
                "handshake_attempts", defaults["handshake_attempts"]
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
            "median_window_samples": signal_config.get(
                "median_window_samples", defaults["median_window_samples"]
            ),
            "force_deadband_n": signal_config.get(
                "deadband_n", defaults["force_deadband_n"]
            ),
            "stationary_window_sec": signal_config.get(
                "stationary_window_sec", defaults["stationary_window_sec"]
            ),
            "stationary_std_max_n": signal_config.get(
                "stationary_std_max_n", defaults["stationary_std_max_n"]
            ),
            "stationary_range_max_n": signal_config.get(
                "stationary_range_max_n", defaults["stationary_range_max_n"]
            ),
            "stationary_slope_max_n_per_sec": signal_config.get(
                "stationary_slope_max_n_per_sec",
                defaults["stationary_slope_max_n_per_sec"],
            ),
            "auto_zero_drift_enabled": signal_config.get(
                "auto_zero_drift_enabled", defaults["auto_zero_drift_enabled"]
            ),
            "auto_zero_hold_sec": signal_config.get(
                "auto_zero_hold_sec", defaults["auto_zero_hold_sec"]
            ),
            "auto_zero_capture_limit_n": signal_config.get(
                "auto_zero_capture_limit_n",
                defaults["auto_zero_capture_limit_n"],
            ),
            "auto_zero_release_reacquire_limit_n": signal_config.get(
                "auto_zero_release_reacquire_limit_n",
                defaults["auto_zero_release_reacquire_limit_n"],
            ),
            "auto_zero_alpha": signal_config.get(
                "auto_zero_alpha", defaults["auto_zero_alpha"]
            ),
            "maximum_drift_offset_n": signal_config.get(
                "maximum_drift_offset_n", defaults["maximum_drift_offset_n"]
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
    return _apply_px6d_environment_overrides(defaults)


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


def _load_runtime_startup_baseline_config() -> dict[str, Any]:
    if yaml is None or not RUNTIME_CONTACT_STATE_CONFIG_PATH.exists():
        return {}
    payload = yaml.safe_load(
        RUNTIME_CONTACT_STATE_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}
    section = payload.get("runtime_startup_baseline", {})
    return dict(section) if isinstance(section, dict) else {}


def _load_all_source_runtime_gate_config() -> dict[str, Any]:
    if yaml is None or not RUNTIME_CONTACT_STATE_CONFIG_PATH.exists():
        return {}
    payload = yaml.safe_load(
        RUNTIME_CONTACT_STATE_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}
    section = payload.get("all_source_runtime_gate", {})
    return dict(section) if isinstance(section, dict) else {}


try:
    CURRENT_RUNTIME_ADAPTER = AllSourceOpticalForceAdapter.from_paths(
        CURRENT_RUNTIME_MODEL_PATH,
        HYBRID_SPECTRUM_CHANNEL_CONFIG_PATH,
        runtime_recovery_config=_load_runtime_baseline_recovery_config(),
        runtime_gate_config=_load_all_source_runtime_gate_config(),
    )
    CURRENT_RUNTIME_ERROR = None
except Exception as exc:  # pragma: no cover - exposed through diagnostics
    CURRENT_RUNTIME_ADAPTER = None
    CURRENT_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"


def _reset_current_runtime(reason: str) -> dict[str, Any]:
    global CURRENT_RUNTIME_BASELINE_TOKEN
    global CURRENT_RUNTIME_LAST_FRAME_KEY
    global CURRENT_RUNTIME_UNIQUE_FRAME_COUNT
    global CURRENT_RUNTIME_LAST_PAYLOAD
    global CURRENT_RUNTIME_STARTUP_BASELINE_LAST_FRAME_KEY
    global CURRENT_RUNTIME_STARTUP_BASELINE_STATUS

    with CURRENT_RUNTIME_LOCK:
        if CURRENT_RUNTIME_ADAPTER is not None:
            CURRENT_RUNTIME_ADAPTER.clear()
        CURRENT_RUNTIME_BASELINE_TOKEN = None
        CURRENT_RUNTIME_LAST_FRAME_KEY = None
        CURRENT_RUNTIME_UNIQUE_FRAME_COUNT = 0
        CURRENT_RUNTIME_LAST_PAYLOAD = None
        CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES.clear()
        CURRENT_RUNTIME_STARTUP_BASELINE_LAST_FRAME_KEY = None
        CURRENT_RUNTIME_STARTUP_BASELINE_STATUS = {
            "status": "not_started",
            "ready": False,
            "frame_count": 0,
            "reason": reason,
        }
    return {
        "ok": True,
        "status": "current_runtime_reset",
        "reason": reason,
        "runtime_role": "deployed_current_model_only",
    }


def _current_runtime_status() -> dict[str, Any]:
    adapter = CURRENT_RUNTIME_ADAPTER
    bundle = adapter.bundle if adapter is not None else {}
    observed_training_range_n = (
        [float(adapter.force_min_n), float(adapter.force_calibrated_max_n)]
        if adapter is not None
        else [0.0, None]
    )
    return {
        "enabled": CURRENT_RUNTIME_ENABLED,
        "loaded": adapter is not None,
        "model_path": str(CURRENT_RUNTIME_MODEL_PATH),
        "model_error": CURRENT_RUNTIME_ERROR,
        "schema_version": bundle.get("schema_version"),
        "feature_schema": bundle.get("feature_schema"),
        "runtime_role": "deployed_current_model_only",
        "drives_operator_ui": CURRENT_RUNTIME_ENABLED,
        "drives_digital_twin": CURRENT_RUNTIME_ENABLED,
        "runtime_input": "optical_spectrum_time_series",
        "force_sensor_is_runtime_model_input": False,
        "estimated_force_output": "continuous_optical_fz_no_fixed_upper_limit",
        "observed_training_force_range_n": observed_training_range_n,
        # Retained for older UI consumers; this is the observed training range,
        # not a runtime output clip.
        "validated_force_range_n": observed_training_range_n,
        "force_above_observed_training_range": (
            "reported_as_unvalidated_extrapolation_without_upper_clip"
        ),
        "classification_model_source": (
            adapter.classification_model_source if adapter is not None else None
        ),
        "force_model_source": (
            adapter.force_model_source if adapter is not None else None
        ),
        "unique_frame_count": CURRENT_RUNTIME_UNIQUE_FRAME_COUNT,
        "runtime_contact_gate": _load_all_source_runtime_gate_config(),
        "runtime_startup_baseline": {
            "config": _load_runtime_startup_baseline_config(),
            "state": copy.deepcopy(CURRENT_RUNTIME_STARTUP_BASELINE_STATUS),
        },
        "runtime_baseline_recovery": _load_runtime_baseline_recovery_config(),
    }


OPERATOR_VISUALIZATION_CONTRACT_VERSION = "touch_operator_visualization_v1"
OPERATOR_VISUALIZATION_DISPLAY_ROWS = (
    ("P11", "P21", "P31"),
    ("P12", "P22", "P32"),
    ("P13", "P23", "P33"),
)


def _operator_finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _operator_surface_grid(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        return [[0.0, 0.0, 0.0] for _ in range(3)]
    grid: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return [[0.0, 0.0, 0.0] for _ in range(3)]
        grid.append(
            [
                max(0.0, min(1.0, _operator_finite_float(item, 0.0) or 0.0))
                for item in row
            ]
        )
    return grid


def _build_operator_visualization_frame(
    latest: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
    *,
    ready: bool,
    block_reason: str | None,
) -> dict[str, Any]:
    """Publish the only frame allowed to drive Operator visuals.

    Operator charts, summary, footprint and 3-D geometry consume this contract
    exclusively. A warming or blocked current model becomes a neutral frame
    instead of silently falling back to a different data source.
    """

    record = latest if isinstance(latest, dict) else {}
    payload = prediction if isinstance(prediction, dict) else {}
    prediction_ready = bool(
        ready
        and payload.get("ok") is True
        and payload.get("status") == "ready"
    )
    twin = payload.get("digital_twin") if prediction_ready else {}
    twin = twin if isinstance(twin, dict) else {}
    position = payload.get("position") if prediction_ready else {}
    position = position if isinstance(position, dict) else {}
    contact = payload.get("contact") if prediction_ready else {}
    contact = contact if isinstance(contact, dict) else {}
    force = payload.get("force_fz") if prediction_ready else {}
    force = force if isinstance(force, dict) else {}
    raw_metrics = twin.get("surface_metrics")
    raw_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}

    visual_active = bool(
        prediction_ready
        and (twin.get("visual_active", twin.get("active")) is True)
    )
    grid = _operator_surface_grid(twin.get("surface_grid"))
    if not visual_active:
        grid = [[0.0, 0.0, 0.0] for _ in range(3)]

    flat_grid = [item for row in grid for item in row]
    display_position = (
        str(
            twin.get("position_id")
            or position.get("visual_label")
            or position.get("label")
            or ""
        )
        or None
    )
    display_force_n = _operator_finite_float(force.get("visual_drive_n"))
    if display_force_n is None:
        display_force_n = _operator_finite_float(twin.get("drive_force_n"))
    if display_force_n is None:
        display_force_n = _operator_finite_float(force.get("estimated_n"))
    if prediction_ready and not visual_active:
        display_force_n = 0.0
    if not prediction_ready:
        display_force_n = None

    estimated_force_n = (
        _operator_finite_float(force.get("estimated_n"))
        if prediction_ready
        else None
    )
    continuous_force_n = (
        _operator_finite_float(force.get("continuous_estimated_n"))
        if prediction_ready
        else None
    )
    source = str(record.get("source") or "unknown")
    timestamp = _operator_finite_float(record.get("timestamp"))
    raw_frame_id = record.get("frame_id", record.get("spectrum_frame_id"))
    frame_id: Any = raw_frame_id
    if frame_id is None:
        frame_id = f"{source}|{timestamp if timestamp is not None else 'unknown'}"

    response_allowed = prediction_ready
    response_state = "contact" if visual_active else "no_contact"
    peak = (
        _operator_finite_float(raw_metrics.get("surface_peak"), max(flat_grid))
        if prediction_ready
        else 0.0
    )
    mean = (
        _operator_finite_float(
            raw_metrics.get("surface_mean"),
            sum(flat_grid) / len(flat_grid),
        )
        if prediction_ready
        else 0.0
    )
    active_area = (
        _operator_finite_float(
            raw_metrics.get("surface_area_active"),
            sum(1 for value in flat_grid if value >= 0.055) / len(flat_grid),
        )
        if prediction_ready
        else 0.0
    )
    responding_channel_ids = [
        channel_id
        for row_index, row in enumerate(OPERATOR_VISUALIZATION_DISPLAY_ROWS)
        for column_index, channel_id in enumerate(row)
        if grid[row_index][column_index] >= 0.055
    ]
    metrics = {
        "surface_peak": peak or 0.0,
        "surface_mean": mean or 0.0,
        "surface_area_active": active_area or 0.0,
        "surface_area_active_percent": (active_area or 0.0) * 100.0,
        "surface_centroid_x": (
            _operator_finite_float(raw_metrics.get("surface_centroid_x"), 0.0)
            or 0.0
        ),
        "surface_centroid_y": (
            _operator_finite_float(raw_metrics.get("surface_centroid_y"), 0.0)
            or 0.0
        ),
        "surface_spread": (
            _operator_finite_float(raw_metrics.get("surface_spread"), 0.24)
            or 0.24
        ),
        "dominant_channel": display_position if visual_active else None,
        "enabled_channel_count": 9,
        "responding_channel_count": len(responding_channel_ids),
        "responding_channel_ids": responding_channel_ids,
        "coupling_status": str(
            raw_metrics.get("coupling_status")
            or (
                "optical_model_continuous_force_proxy"
                if visual_active
                else "optical_model_no_contact"
            )
        ),
    }
    resolved_block_reason = None if prediction_ready else (
        block_reason
        or str(payload.get("reason") or payload.get("status") or "current_runtime_model_not_ready")
    )
    return {
        "contract_version": OPERATOR_VISUALIZATION_CONTRACT_VERSION,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "source": source,
        "source_kind": "full_spectrum_optical_frame",
        "model_source": "ordinary_fbg_all_data_beta_v1",
        "prediction_status": str(payload.get("status") or "unavailable"),
        "prediction_ready": prediction_ready,
        "response_allowed": response_allowed,
        "response_block_reason": resolved_block_reason,
        "response_state": response_state,
        "contact": {
            "label": str(contact.get("label") or response_state),
            "visual_active": visual_active,
            "confidence": _operator_finite_float(contact.get("confidence")),
            "contact_probability": _operator_finite_float(
                contact.get("contact_probability")
            ),
        },
        "position": {
            "display_label": display_position if visual_active else None,
            "formal_label": position.get("label") if prediction_ready else None,
            "confidence": _operator_finite_float(position.get("confidence")),
            "visual_confidence": _operator_finite_float(
                position.get("visual_confidence")
            ),
        },
        "force": {
            "display_n": display_force_n,
            "estimated_n": estimated_force_n,
            "continuous_estimated_n": continuous_force_n,
            "unit": "N",
            "range_status": force.get("range_status") if prediction_ready else None,
            "calibrated_range_n": (
                force.get("calibrated_range_n") if prediction_ready else None
            ),
            "upper_limit_applied": False,
            "model_source": force.get("model_source") if prediction_ready else None,
        },
        "surface": {
            "active": visual_active,
            "position_id": display_position if visual_active else None,
            "deformation_proxy": (
                _operator_finite_float(twin.get("deformation_proxy"), 0.0)
                if visual_active
                else 0.0
            ),
            "drive_force_n": display_force_n,
            "surface_grid": grid,
            "surface_metrics": metrics,
        },
        "trace_sample": {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "value_n": display_force_n,
            "surface_peak": peak or 0.0,
            "response_state": response_state,
        },
        "sync": {
            "status": "synced",
            "spectrum_frame_id": frame_id,
            "surface_frame_id": frame_id,
            "trace_frame_id": frame_id,
            "summary_frame_id": frame_id,
            "force_frame_id": frame_id,
        },
        "quality": {
            "raw_qa_status": record.get("qa_status"),
            "raw_qa_flags": list(record.get("qa_flags") or []),
            "review_needed": bool(
                prediction_ready
                and isinstance(payload.get("uncertainty"), dict)
                and payload["uncertainty"].get("review_needed") is True
            ),
        },
        "prediction": copy.deepcopy(payload) if prediction_ready else None,
    }


def _startup_baseline_motion(
    previous: np.ndarray,
    current: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    eps = 1.0e-6
    delta = np.log(np.maximum(current[valid_mask], eps)) - np.log(
        np.maximum(previous[valid_mask], eps)
    )
    common_gain = float(np.median(delta))
    shape_motion = float(np.sqrt(np.mean(np.square(delta - common_gain))))
    return shape_motion, abs(common_gain)


def _runtime_startup_force_evidence(
    timestamp_sec: float | None,
) -> dict[str, Any]:
    reader = globals().get("px6d_reader")
    if reader is None:
        return {"available": False, "status": "force_reader_unavailable"}
    try:
        status = reader.status()
    except Exception as exc:
        return {
            "available": False,
            "status": "force_status_error",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    available = bool(
        status.get("connected")
        and status.get("sample_fresh")
        and status.get("tare_ready")
    )
    if not available:
        return {
            "available": False,
            "status": "force_reference_not_ready",
            "connected": bool(status.get("connected")),
            "sample_fresh": bool(status.get("sample_fresh")),
            "tare_ready": bool(status.get("tare_ready")),
        }
    try:
        snapshot = reader.synchronized_snapshot(timestamp_sec or time.time())
    except Exception as exc:
        return {
            "available": False,
            "status": "force_snapshot_error",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    force_n = _operator_finite_float(snapshot.get("force_fz_n"))
    return {
        "available": bool(snapshot.get("ok") and force_n is not None),
        "status": snapshot.get("status"),
        "force_fz_n": force_n,
        "sync_offset_ms": snapshot.get("sync_offset_ms"),
    }


def _ensure_current_runtime_startup_baseline(
    latest_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a trusted baseline from stable frames in this acquisition session."""

    global CURRENT_RUNTIME_STARTUP_BASELINE_LAST_FRAME_KEY
    global CURRENT_RUNTIME_STARTUP_BASELINE_STATUS

    pair = bridge.spectral_model_input(channel_id="P22")
    if pair.get("ok"):
        with CURRENT_RUNTIME_LOCK:
            CURRENT_RUNTIME_STARTUP_BASELINE_STATUS = {
                "status": "ready",
                "ready": True,
                "frame_count": int(
                    pair.get("baseline_spectrum_sample_count") or 0
                ),
                "baseline_status": pair.get("baseline_spectrum_status"),
            }
        return pair

    config = _load_runtime_startup_baseline_config()
    if not bool(config.get("enabled", True)):
        return pair

    latest = latest_override if isinstance(latest_override, dict) else pair.get("latest")
    if not isinstance(latest, dict):
        return pair
    wavelength_values = latest.get("wavelength_nm")
    intensity_values = latest.get("intensity")
    if (
        not isinstance(wavelength_values, list)
        or not isinstance(intensity_values, list)
        or len(wavelength_values) < 16
        or len(wavelength_values) != len(intensity_values)
    ):
        return pair

    wavelength = np.asarray(wavelength_values, dtype=float)
    intensity = np.asarray(intensity_values, dtype=float)
    if (
        not np.all(np.isfinite(wavelength))
        or not np.all(np.isfinite(intensity))
        or not np.all(np.diff(wavelength) > 0.0)
        or float(np.max(np.abs(intensity))) <= 1.0e-9
    ):
        return pair

    try:
        source_timestamp_sec = float(latest.get("timestamp"))
        if not math.isfinite(source_timestamp_sec):
            source_timestamp_sec = None
    except (TypeError, ValueError):
        source_timestamp_sec = None
    frame_key = (
        latest.get("frame_id"),
        latest.get("timestamp"),
        latest.get("source"),
        _spectrum_fingerprint(wavelength_values, intensity_values),
    )
    minimum_frames = max(3, int(config.get("minimum_frames", 5)))
    minimum_span_sec = max(0.0, float(config.get("minimum_span_sec", 0.12)))
    max_shape_motion = max(
        0.0, float(config.get("max_shape_motion_rms", 0.0060))
    )
    max_common_gain_motion = max(
        0.0, float(config.get("max_common_gain_motion", 0.0060))
    )
    maximum_force_abs_n = max(
        0.0, float(config.get("maximum_force_abs_n", 0.10))
    )
    require_force_release = bool(
        config.get("require_force_release_if_available", True)
    )
    policy = str(
        config.get("policy") or "current_session_stable_five_frame_median"
    )
    observed_monotonic_sec = time.monotonic()

    with CURRENT_RUNTIME_LOCK:
        if frame_key == CURRENT_RUNTIME_STARTUP_BASELINE_LAST_FRAME_KEY:
            return {
                **pair,
                "startup_baseline_status": copy.deepcopy(
                    CURRENT_RUNTIME_STARTUP_BASELINE_STATUS
                ),
            }
        CURRENT_RUNTIME_STARTUP_BASELINE_LAST_FRAME_KEY = frame_key

        force_evidence = _runtime_startup_force_evidence(source_timestamp_sec)
        force_n = _operator_finite_float(force_evidence.get("force_fz_n"))
        if (
            require_force_release
            and force_evidence.get("available")
            and force_n is not None
            and abs(force_n) > maximum_force_abs_n
        ):
            CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES.clear()
            CURRENT_RUNTIME_STARTUP_BASELINE_STATUS = {
                "status": "startup_baseline_waiting_for_release",
                "ready": False,
                "frame_count": 0,
                "force_fz_n": force_n,
                "maximum_force_abs_n": maximum_force_abs_n,
                "policy": policy,
            }
            return {
                **pair,
                "startup_baseline_status": copy.deepcopy(
                    CURRENT_RUNTIME_STARTUP_BASELINE_STATUS
                ),
            }

        CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES.append(
            {
                "wavelength_nm": wavelength.copy(),
                "intensity": intensity.copy(),
                "timestamp_sec": source_timestamp_sec,
                "observed_monotonic_sec": observed_monotonic_sec,
            }
        )
        maximum_buffer_frames = max(minimum_frames + 3, minimum_frames * 2)
        if len(CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES) > maximum_buffer_frames:
            del CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES[:-maximum_buffer_frames]

        frame_count = len(CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES)
        if frame_count < minimum_frames:
            CURRENT_RUNTIME_STARTUP_BASELINE_STATUS = {
                "status": "startup_baseline_collecting",
                "ready": False,
                "frame_count": frame_count,
                "minimum_frames": minimum_frames,
                "force_gate": force_evidence,
                "policy": policy,
            }
            return {
                **pair,
                "startup_baseline_status": copy.deepcopy(
                    CURRENT_RUNTIME_STARTUP_BASELINE_STATUS
                ),
            }

        candidate = CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES[-minimum_frames:]
        reference_x = candidate[-1]["wavelength_nm"]
        aligned = np.vstack(
            [
                np.interp(reference_x, item["wavelength_nm"], item["intensity"])
                for item in candidate
            ]
        )
        median_spectrum = np.median(aligned, axis=0)
        bright_floor = max(
            1.0,
            float(np.percentile(np.abs(median_spectrum), 95)) * 0.01,
        )
        valid_mask = np.isfinite(median_spectrum) & (
            np.abs(median_spectrum) >= bright_floor
        )
        if int(np.sum(valid_mask)) < 16:
            valid_mask = np.isfinite(median_spectrum)
        motions = [
            _startup_baseline_motion(
                aligned[index - 1], aligned[index], valid_mask
            )
            for index in range(1, aligned.shape[0])
        ]
        shape_motion_rms = max((item[0] for item in motions), default=0.0)
        common_gain_motion = max((item[1] for item in motions), default=0.0)
        source_timestamps = [
            float(item["timestamp_sec"])
            for item in candidate
            if item.get("timestamp_sec") is not None
        ]
        source_span = (
            max(source_timestamps) - min(source_timestamps)
            if len(source_timestamps) >= 2
            else 0.0
        )
        monotonic_span = float(
            candidate[-1]["observed_monotonic_sec"]
            - candidate[0]["observed_monotonic_sec"]
        )
        span_sec = max(float(source_span), monotonic_span)
        stable = bool(
            span_sec >= minimum_span_sec
            and shape_motion_rms <= max_shape_motion
            and common_gain_motion <= max_common_gain_motion
        )
        if not stable:
            CURRENT_RUNTIME_STARTUP_BASELINE_STATUS = {
                "status": "startup_baseline_waiting_for_stability",
                "ready": False,
                "frame_count": frame_count,
                "candidate_frame_count": minimum_frames,
                "span_sec": span_sec,
                "shape_motion_rms": shape_motion_rms,
                "common_gain_motion": common_gain_motion,
                "max_shape_motion_rms": max_shape_motion,
                "max_common_gain_motion": max_common_gain_motion,
                "force_gate": force_evidence,
                "policy": policy,
            }
            return {
                **pair,
                "startup_baseline_status": copy.deepcopy(
                    CURRENT_RUNTIME_STARTUP_BASELINE_STATUS
                ),
            }

        baseline_result = bridge.set_runtime_startup_spectrum_baseline(
            "P22",
            reference_x,
            median_spectrum,
            sample_count=minimum_frames,
            span_sec=span_sec,
            shape_motion_rms=shape_motion_rms,
            common_gain_motion=common_gain_motion,
            policy=policy,
        )
        CURRENT_RUNTIME_STARTUP_BASELINE_STATUS = {
            "status": baseline_result.get("status"),
            "ready": bool(baseline_result.get("ok")),
            "frame_count": minimum_frames,
            "span_sec": span_sec,
            "shape_motion_rms": shape_motion_rms,
            "common_gain_motion": common_gain_motion,
            "force_gate": force_evidence,
            "policy": policy,
        }
        if baseline_result.get("ok"):
            CURRENT_RUNTIME_STARTUP_BASELINE_FRAMES.clear()

    refreshed = bridge.spectral_model_input(channel_id="P22")
    return {
        **refreshed,
        "startup_baseline_status": copy.deepcopy(
            CURRENT_RUNTIME_STARTUP_BASELINE_STATUS
        ),
    }


def _predict_current_runtime(
    latest_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the sole deployed optical model on each unique spectrum frame."""

    global CURRENT_RUNTIME_BASELINE_TOKEN
    global CURRENT_RUNTIME_LAST_FRAME_KEY
    global CURRENT_RUNTIME_UNIQUE_FRAME_COUNT
    global CURRENT_RUNTIME_LAST_PAYLOAD

    adapter = CURRENT_RUNTIME_ADAPTER
    if adapter is None:
        return {
            "ok": False,
            "status": "current_runtime_model_unavailable",
            "reason": CURRENT_RUNTIME_ERROR,
            "runtime_role": "deployed_current_model_only",
        }

    pair = _ensure_current_runtime_startup_baseline(latest_override)
    if not pair.get("ok"):
        startup_status = pair.get("startup_baseline_status")
        startup_state = (
            str(startup_status.get("status"))
            if isinstance(startup_status, dict) and startup_status.get("status")
            else None
        )
        blocked_status = (
            startup_state
            or ("baseline_required" if pair.get("current_ready") else "spectrum_required")
        )
        return {
            "ok": False,
            "status": blocked_status,
            "reason": startup_status or pair.get("reason"),
            "runtime_role": "deployed_current_model_only",
        }
    latest = pair["latest"]
    if (
        isinstance(latest_override, dict)
        and isinstance(latest_override.get("wavelength_nm"), list)
        and isinstance(latest_override.get("intensity"), list)
        and latest_override.get("wavelength_nm")
        and len(latest_override["wavelength_nm"])
        == len(latest_override["intensity"])
    ):
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
        _spectrum_fingerprint(
            latest["wavelength_nm"],
            latest["intensity"],
        ),
    )
    started = time.perf_counter()
    try:
        with CURRENT_RUNTIME_LOCK:
            if baseline_token != CURRENT_RUNTIME_BASELINE_TOKEN:
                adapter.set_baseline(
                    baseline["wavelength_nm"],
                    baseline["intensity"],
                )
                CURRENT_RUNTIME_BASELINE_TOKEN = baseline_token
                CURRENT_RUNTIME_LAST_FRAME_KEY = None
                CURRENT_RUNTIME_UNIQUE_FRAME_COUNT = 0
                CURRENT_RUNTIME_LAST_PAYLOAD = None
            if (
                frame_key == CURRENT_RUNTIME_LAST_FRAME_KEY
                and CURRENT_RUNTIME_LAST_PAYLOAD is not None
            ):
                cached = copy.deepcopy(CURRENT_RUNTIME_LAST_PAYLOAD)
                cached["duplicate_frame_ignored"] = True
                cached["cache_lookup_latency_ms"] = (
                    time.perf_counter() - started
                ) * 1000.0
                return cached

            try:
                source_timestamp_sec = float(latest.get("timestamp"))
            except (TypeError, ValueError):
                source_timestamp_sec = None
            prediction = adapter.update(
                latest["wavelength_nm"],
                latest["intensity"],
                source_timestamp_sec=source_timestamp_sec,
            )
            runtime_baseline_update = None
            pending_baseline = adapter.consume_pending_runtime_baseline_update()
            if pending_baseline is not None:
                try:
                    runtime_baseline_update = (
                        bridge.set_runtime_recovery_spectrum_baseline(
                            "P22",
                            pending_baseline["wavelength_nm"],
                            pending_baseline["intensity"],
                            sample_count=int(
                                pending_baseline.get("sample_count") or 1
                            ),
                            span_sec=float(
                                pending_baseline.get("span_sec") or 0.0
                            ),
                            shape_motion_rms=pending_baseline.get(
                                "shape_motion_rms"
                            ),
                            common_gain_motion=pending_baseline.get(
                                "common_gain_motion"
                            ),
                            policy=str(
                                pending_baseline.get("policy")
                                or "multi_evidence_release_then_spectral_stationarity"
                            ),
                        )
                    )
                except Exception as exc:  # keep adapter and bridge references aligned
                    runtime_baseline_update = {
                        "ok": False,
                        "status": "runtime_recovery_baseline_commit_error",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                if runtime_baseline_update.get("ok"):
                    CURRENT_RUNTIME_BASELINE_TOKEN = _spectrum_token(
                        pending_baseline["wavelength_nm"].tolist(),
                        pending_baseline["intensity"].tolist(),
                    )
                else:
                    adapter.set_baseline(
                        baseline["wavelength_nm"],
                        baseline["intensity"],
                    )
                    CURRENT_RUNTIME_BASELINE_TOKEN = baseline_token
                    CURRENT_RUNTIME_LAST_FRAME_KEY = None
                    CURRENT_RUNTIME_LAST_PAYLOAD = None
                    runtime_baseline_update = {
                        **runtime_baseline_update,
                        "adapter_baseline_rollback_applied": True,
                        "rollback_baseline_token": baseline_token,
                    }
            CURRENT_RUNTIME_UNIQUE_FRAME_COUNT += 1
            payload = {
                **prediction,
                "runtime_role": "deployed_current_model_only",
                "drives_operator_ui": True,
                "drives_digital_twin": True,
                "unique_frame_count": CURRENT_RUNTIME_UNIQUE_FRAME_COUNT,
                "backend_inference_latency_ms": (
                    time.perf_counter() - started
                )
                * 1000.0,
                "duplicate_frame_ignored": False,
                "cache_lookup_latency_ms": 0.0,
                "runtime_baseline_update": runtime_baseline_update,
            }
            CURRENT_RUNTIME_LAST_FRAME_KEY = frame_key
            CURRENT_RUNTIME_LAST_PAYLOAD = copy.deepcopy(payload)
            return payload
    except Exception as exc:
        return {
            "ok": False,
            "status": "current_runtime_model_inference_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "runtime_role": "deployed_current_model_only",
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
    qa_status = str((latest or {}).get("qa_status") or "").strip().lower()
    qa_flags = {
        str(flag).strip().lower()
        for flag in ((latest or {}).get("qa_flags") or [])
        if str(flag).strip()
    }
    peak_axis_type = str((latest or {}).get("peak_axis_type") or "").strip().lower()
    hardware_config = BAYSPEC_CHANNEL_CONFIG.get("hardware", {}) or {}
    require_wavelength_grid = bool(
        hardware_config.get("require_wavelength_grid_for_formal_recognition", True)
    )
    wavelength_axis_blockers = {
        "wavelength_grid_missing",
        "using_pixel_index_fallback",
        "pixel_peak_fallback",
    }
    wavelength_axis_valid = bool(
        not require_wavelength_grid
        or (
            peak_axis_type != "pixel_index"
            and not qa_flags.intersection(wavelength_axis_blockers)
        )
    )

    selected_live_source = None
    source_fresh = False
    is_live_origin = False
    if "bayspec_direct" in source or "sdk" in source:
        selected_live_source = "sdk"
        is_live_origin = True
        source_fresh = sdk_active and sdk_status.get("freshness") == "live"
    elif "sense" in source or "export" in source or "watch" in source:
        selected_live_source = "watcher"
        is_live_origin = True
        source_fresh = watcher_active and watcher_status.get("freshness") in {
            "fresh",
            "live",
        }
    elif live_source_active:
        # A live acquisition is running, but the buffered record came from a
        # different source. Hold it for diagnostics and suppress deformation.
        selected_live_source = "unmatched_live_source"

    replay_or_http_allowed = bool(
        latest
        and not live_source_active
        and not is_live_origin
        and source not in {"", "operator_idle"}
    )
    model_input_source_allowed = bool(source_fresh or replay_or_http_allowed)
    qa_valid = (
        qa_status not in {"invalid", "error", "stale"}
        and wavelength_axis_valid
    )
    formal_spectrum_input_allowed = bool(
        model_input_source_allowed and qa_valid
    )
    operator_display_valid = bool(
        latest and formal_spectrum_input_allowed
    )
    return {
        "live_source_active": live_source_active,
        "selected_live_source": selected_live_source,
        "source_fresh": source_fresh,
        "is_live_origin": is_live_origin,
        "qa_valid": qa_valid,
        "wavelength_axis_valid": wavelength_axis_valid,
        "wavelength_axis_blockers": sorted(
            qa_flags.intersection(wavelength_axis_blockers)
        ),
        "formal_spectrum_input_allowed": formal_spectrum_input_allowed,
        "operator_display_valid": operator_display_valid,
        "model_input_source_allowed": model_input_source_allowed,
        "model_input_source_mode": (
            "fresh_live"
            if source_fresh
            else "held_replay_or_http"
            if replay_or_http_allowed
            else "stopped_live_source"
            if is_live_origin and not live_source_active
            else "stale_or_mismatched_live"
        ),
    }


def _default_thumb_scene_config() -> dict:
    return {
        "thumb_holder_scene": {
            "enabled": True,
            "default_geometry_mode": "whole_hand",
            "model_asset_url": "",
            "fallback_asset_url": "/static/assets/models/thumb_holder.stl",
            "fallback_placeholder_enabled": False,
            "model_load_policy": "glb_then_stl_else_blocked",
            "note": (
                "Thumb holder uses an oval-like groove insert. Response is uncalibrated "
                "Bragg wavelength displacement with no calibrated-force output."
            ),
        },
        "whole_hand_scene": {
            "enabled": True,
            "asset_url": "/static/assets/models/robot_nano_hand_sensorized.glb",
            "fallback_asset_url": "/static/assets/models/robot_nano_hand_body.glb",
            "source_name": "Robot Nano Hand",
            "source_repository_url": "https://github.com/TheRobotStudio/robot-nano-hand",
            "source_step_file": "STEP/Complete_Hand_06_Final_Print_03e.STEP",
            "source_license": "MIT",
            "body_excludes_original_thumb_tip": True,
            "modified_thumb_tip_asset": "/static/assets/models/thumb_holder.stl",
            "body_opacity": 0.42,
            "body_transform": {
                "position": [-1.182552, -0.106274, 1.721579],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale": [0.034, 0.034, 0.034],
            },
            "modified_thumb_root_matrix_row_major": [
                0.3954683696,
                -0.1851201911,
                -7.8582482177,
                8.7864236452,
                4.5523782645,
                6.4196976939,
                0.0778679840,
                5.7824730088,
                6.4079783608,
                -4.5492792617,
                0.4296525265,
                -107.2841714018,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "sensor_local_lift": [0.22, 0.0, 0.0],
            "camera": {
                "position": [4.7, 2.2, -8.8],
                "target": [0.0, 0.0, 0.0],
            },
            "note": (
                "Official MIT-licensed hand assembly with the original thumb tip removed "
                "and the existing TOUCH thumb sensor fitted in its place."
            ),
        },
        "finger_sensor_array": {
            "enabled": True,
            "geometry_status": "five_finger_sensorized_prototype",
            "data_status": "synchronized_demo_only",
            "default_selected_finger": "all",
            "demo_sync_mode": "synchronized_with_thumb",
            "spectrum_scope_mode": "selected_finger",
            "array_scope_mode": "selected_finger",
            "semantic_to_geometry_id": {
                "thumb": "thumb",
                "index": "little",
                "middle": "ring",
                "ring": "middle",
                "little": "index",
            },
            "source_asset_url": "/static/assets/models/robot_nano_hand_sensorized.glb",
            "original_asset_url": "/static/assets/models/robot_nano_hand_body.glb",
            "fingers": {
                "thumb": {
                    "label": "Thumb",
                    "enabled": True,
                    "sensor_source": "existing_thumb_slot",
                },
                "index": {
                    "label": "Index",
                    "enabled": True,
                    "mesh_component_id": 22,
                    "center_model": [79.087, 62.010, -39.369],
                    "longitudinal_axis_model": [0.0, 1.0, 0.0],
                    "outward_normal_model": [-0.274, -0.275, -0.922],
                    "slot_length_mm": 18.0,
                    "slot_width_mm": 11.5,
                    "slot_depth_tolerance_mm": 2.2,
                    "sensor_thickness_scale": 1.35,
                    "surface_offset_mm": 0.08,
                },
                "middle": {
                    "label": "Middle",
                    "enabled": True,
                    "mesh_component_id": 16,
                    "center_model": [55.573, 82.611, -41.918],
                    "longitudinal_axis_model": [0.086328, 0.877303, -0.472110],
                    "outward_normal_model": [-0.116884, -0.461688, -0.879308],
                    "slot_length_mm": 18.5,
                    "slot_width_mm": 11.5,
                    "slot_depth_tolerance_mm": 2.4,
                    "sensor_thickness_scale": 1.35,
                    "surface_offset_mm": 0.08,
                },
                "ring": {
                    "label": "Ring",
                    "enabled": True,
                    "mesh_component_id": 10,
                    "center_model": [23.129, 101.826, -19.171],
                    "longitudinal_axis_model": [-0.026465, 0.993782, -0.108153],
                    "outward_normal_model": [-0.086703, -0.110065, -0.990135],
                    "slot_length_mm": 18.5,
                    "slot_width_mm": 11.5,
                    "slot_depth_tolerance_mm": 2.2,
                    "sensor_thickness_scale": 1.35,
                    "surface_offset_mm": 0.08,
                },
                "little": {
                    "label": "Little",
                    "enabled": True,
                    "mesh_component_id": 4,
                    "center_model": [-9.320, 92.498, -24.449],
                    "longitudinal_axis_model": [-0.105852, 0.983709, -0.145299],
                    "outward_normal_model": [0.095880, -0.135343, -0.986149],
                    "slot_length_mm": 17.5,
                    "slot_width_mm": 10.8,
                    "slot_depth_tolerance_mm": 2.4,
                    "sensor_thickness_scale": 1.35,
                    "surface_offset_mm": 0.08,
                },
            },
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
            "whole_hand_material_color": "#d6e0e6",
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
    allowed_sections = {
        "thumb_holder_scene",
        "whole_hand_scene",
        "finger_sensor_array",
        "thumb_model_transform",
        "sensor_slot_transform",
        "visual_style",
    }
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

    # Layout detection needs multiple adjacent spectra. Waiting for three
    # nominal 512-word frames prevents a growing 513-word recording from being
    # misread as two complete 512-word frames during its first milliseconds.
    MIN_INITIAL_DAT_BYTES = 3 * 512 * 2

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
        self.skipped_duplicate_export_count = 0
        self.incomplete_dat_skip_count = 0
        self.unstable_file_skip_count = 0
        self.last_dat_progress: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self.generation = 0
        self.ingest_lock = threading.Lock()
        self.ingest_in_progress = False
        self.stop_event = threading.Event()
        self.lifecycle_status = "stopped"
        self.last_operation_status = "idle"

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
                "skipped_duplicate_export_count": self.skipped_duplicate_export_count,
                "incomplete_dat_skip_count": self.incomplete_dat_skip_count,
                "unstable_file_skip_count": self.unstable_file_skip_count,
                "last_dat_progress": copy.deepcopy(self.last_dat_progress),
                "worker_alive": bool(self.thread is not None and self.thread.is_alive()),
                "ingest_in_progress": self.ingest_in_progress,
                "stop_requested": self.stop_event.is_set(),
                "lifecycle_status": self.lifecycle_status,
                "last_operation_status": self.last_operation_status,
                "last_result": self.last_result,
                "source": "sense_export_file_polling",
            }

    def start(self, channel_id: str, export_root: str | None, interval_sec: float) -> dict:
        with self.lock:
            existing_worker_alive = bool(self.thread is not None and self.thread.is_alive())
            if existing_worker_alive and not self.active:
                self.lifecycle_status = "stop_timeout"
                self.last_operation_status = "previous_worker_stop_timeout"
                self.last_error = "Previous Sense export watcher is still stopping"
                return {
                    "ok": False,
                    "operation_status": "previous_worker_stop_timeout",
                    **self.status(),
                }
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
                self.skipped_duplicate_export_count = 0
                self.incomplete_dat_skip_count = 0
                self.unstable_file_skip_count = 0
                self.generation += 1
                self.stop_event.clear()
                if configuration_changed:
                    # A signature from another root/channel must never suppress
                    # the first export of the newly selected source.
                    self.last_signature = None
                    self.last_dat_progress = None
            self.active = True
            self.last_error = None
            self.lifecycle_status = "starting"
            self.last_operation_status = "starting"
            if self.thread is None or not self.thread.is_alive():
                worker = threading.Thread(
                    target=self._loop,
                    name="sense-export-watcher",
                    daemon=True,
                )
                self.thread = worker
                try:
                    worker.start()
                except Exception as exc:
                    self.thread = None
                    self.active = False
                    self.stop_event.clear()
                    self.lifecycle_status = "start_failed"
                    self.last_operation_status = "start_failed"
                    self.last_error = (
                        "Sense export watcher failed to start: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return {
                        "ok": False,
                        "operation_status": "start_failed",
                        **self.status(),
                    }
            self.lifecycle_status = "running"
            self.last_operation_status = "started"
            return {"ok": True, "operation_status": "started", **self.status()}

    def stop(self) -> dict:
        with self.lock:
            worker = self.thread
            if (
                not self.active
                and not self.ingest_in_progress
                and (worker is None or not worker.is_alive())
            ):
                self.stop_event.clear()
                self.thread = None
                self.lifecycle_status = "stopped"
                self.last_operation_status = "already_stopped"
                return {
                    "ok": True,
                    "operation_status": "already_stopped",
                    **self.status(),
                }
            self.active = False
            self.generation += 1
            self.stop_event.set()
            self.lifecycle_status = "stopping"
            self.last_operation_status = "stopping"
        # Wait for a file parse already in progress. Source-switch endpoints
        # reset the shared bridge only after this quiescence barrier returns.
        acquired = self.ingest_lock.acquire(timeout=5.0)
        if not acquired:
            with self.lock:
                self.last_error = "Sense export ingest did not stop before timeout"
                self.lifecycle_status = "stop_timeout"
                self.last_operation_status = "stop_timeout"
            return {
                "ok": False,
                "operation_status": "stop_timeout",
                **self.status(),
            }
        self.ingest_lock.release()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        worker_alive = bool(worker is not None and worker.is_alive())
        if worker_alive:
            with self.lock:
                self.last_error = "Sense export watcher did not stop before timeout"
                self.lifecycle_status = "stop_timeout"
                self.last_operation_status = "stop_timeout"
            return {
                "ok": False,
                "operation_status": "stop_timeout",
                **self.status(),
            }
        with self.lock:
            if self.thread is worker:
                self.thread = None
            self.stop_event.clear()
            self.lifecycle_status = "stopped"
            self.last_operation_status = "stopped"
        return {"ok": True, "operation_status": "stopped", **self.status()}

    @staticmethod
    def _file_signature(path: Path) -> tuple[str, int, int, float]:
        stat = path.stat()
        return (
            str(path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            float(stat.st_mtime),
        )

    def _confirm_stable_text_export(
        self,
        path: Path,
        signature: tuple[str, int, int],
    ) -> tuple[str, int, int, float] | None:
        if self.stop_event.wait(0.05):
            return None
        try:
            confirmed = self._file_signature(path)
        except OSError:
            return None
        if confirmed[:3] != signature:
            return None
        return confirmed

    @staticmethod
    def _dat_frame_digest(
        path: Path,
        *,
        record_words: int,
        prefix_words: int,
        frame_count: int,
    ) -> str | None:
        if record_words <= 0 or prefix_words < 0 or frame_count <= 0:
            return None
        byte_offset = (prefix_words + (frame_count - 1) * record_words) * 2
        byte_count = record_words * 2
        try:
            with path.open("rb") as handle:
                handle.seek(byte_offset)
                frame_bytes = handle.read(byte_count)
        except OSError:
            return None
        if len(frame_bytes) != byte_count:
            return None
        return hashlib.sha256(frame_bytes).hexdigest()[:24]

    def _dat_export_has_new_complete_frame(self, path: Path, file_size: int) -> bool:
        with self.lock:
            progress = copy.deepcopy(self.last_dat_progress)
        if not progress or progress.get("path") != str(path):
            return (
                int(file_size) >= self.MIN_INITIAL_DAT_BYTES
                and int(file_size) % 2 == 0
            )
        record_words = int(progress.get("record_words") or 0)
        prefix_words = int(progress.get("prefix_words") or 0)
        previous_count = int(progress.get("frame_count") or 0)
        if record_words <= 0:
            return True
        available_words = max(0, int(file_size) // 2 - prefix_words)
        frame_count = available_words // record_words
        if frame_count < previous_count:
            # Sense may truncate and reuse the same recording path. Treat that
            # as a fresh layout bootstrap instead of parsing two partial 513-
            # word records as a valid 512-word file.
            return (
                int(file_size) >= self.MIN_INITIAL_DAT_BYTES
                and int(file_size) % 2 == 0
            )
        if frame_count > previous_count:
            return True
        digest = self._dat_frame_digest(
            path,
            record_words=record_words,
            prefix_words=prefix_words,
            frame_count=frame_count,
        )
        return bool(digest and digest != progress.get("frame_digest"))

    def _remember_dat_progress(self, path: Path, result: dict[str, Any]) -> None:
        frame_count = int(result.get("dat_frame_count") or 0)
        record_words = int(result.get("dat_record_words") or 0)
        prefix_words = int(result.get("dat_header_bytes") or 0) // 2
        digest = self._dat_frame_digest(
            path,
            record_words=record_words,
            prefix_words=prefix_words,
            frame_count=frame_count,
        )
        if frame_count <= 0 or record_words <= 0 or digest is None:
            return
        self.last_dat_progress = {
            "path": str(path),
            "frame_count": frame_count,
            "record_words": record_words,
            "prefix_words": prefix_words,
            "frame_digest": digest,
        }

    def _loop(self) -> None:
        current_worker = threading.current_thread()
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    active = self.active
                    channel_id = self.channel_id
                    export_root = self.export_root
                    interval = self.interval_sec
                    generation = self.generation
                if not active:
                    break
                try:
                    latest = bridge.latest_export_file(root=export_root)
                    if latest is None:
                        with self.lock:
                            if self.active and generation == self.generation:
                                self.last_error = "no CSV/TXT export file found"
                                self.last_file = None
                        if self.stop_event.wait(interval):
                            break
                        continue
                    signature_state = self._file_signature(latest)
                    signature = signature_state[:3]
                    file_mtime = signature_state[3]
                    if signature != self.last_signature:
                        is_dat = latest.suffix.lower() == ".dat"
                        if is_dat:
                            with self.lock:
                                has_dat_progress = bool(
                                    self.last_dat_progress
                                    and self.last_dat_progress.get("path") == str(latest)
                                )
                            if not self._dat_export_has_new_complete_frame(
                                latest,
                                signature[2],
                            ):
                                with self.lock:
                                    if self.active and generation == self.generation:
                                        self.last_file = str(latest)
                                        self.last_file_mtime = file_mtime
                                        if has_dat_progress:
                                            self.last_signature = signature
                                            self.skipped_duplicate_export_count += 1
                                        else:
                                            self.incomplete_dat_skip_count += 1
                                if self.stop_event.wait(interval):
                                    break
                                continue
                        else:
                            confirmed = self._confirm_stable_text_export(
                                latest,
                                signature,
                            )
                            if confirmed is None:
                                with self.lock:
                                    if self.active and generation == self.generation:
                                        self.unstable_file_skip_count += 1
                                if self.stop_event.wait(interval):
                                    break
                                continue
                            signature_state = confirmed
                            signature = confirmed[:3]
                            file_mtime = confirmed[3]
                        with self.lock:
                            if not self.active or generation != self.generation:
                                continue
                        with self.ingest_lock:
                            with self.lock:
                                if not self.active or generation != self.generation:
                                    continue
                                self.ingest_in_progress = True
                            try:
                                result = bridge.ingest_export_file(
                                    latest,
                                    channel_id=channel_id,
                                    source="bayspec_sense2020_export_watch",
                                )
                            finally:
                                with self.lock:
                                    self.ingest_in_progress = False
                        with self.lock:
                            same_source = (
                                str(self.channel_id) == str(channel_id)
                                and self.export_root == export_root
                            )
                            if result.get("ok") and same_source:
                                self.last_signature = signature
                                if is_dat:
                                    self._remember_dat_progress(latest, result)
                            if not self.active or generation != self.generation:
                                continue
                            self.last_attempt_time = time.time()
                            self.last_result = result
                            self.last_error = None if result.get("ok") else str(result.get("reason"))
                            self.last_file = str(latest)
                            self.last_file_mtime = file_mtime
                            if result.get("ok"):
                                self.last_ingest_time = self.last_attempt_time
                                self.ingest_count += int(result.get("records_ingested") or 0)
                            else:
                                self.failed_ingest_count += 1
                    if self.stop_event.wait(interval):
                        break
                except Exception as exc:
                    with self.lock:
                        if self.active and generation == self.generation:
                            self.last_attempt_time = time.time()
                            self.failed_ingest_count += 1
                            self.last_error = str(exc)
                    if self.stop_event.wait(interval):
                        break
        finally:
            with self.lock:
                if self.thread is current_worker:
                    self.thread = None
                unexpected_exit = self.active and not self.stop_event.is_set()
                self.active = False
                if unexpected_exit:
                    self.last_error = self.last_error or "Sense export watcher stopped unexpectedly"
                    self.lifecycle_status = "worker_exited"
                    self.last_operation_status = "worker_exited"
                elif self.lifecycle_status != "stop_timeout":
                    self.lifecycle_status = "stopped"


export_watcher = SenseExportWatcher()
sense_controller = SenseWindowController()
sdk_live_reader = BaySpecSdkLiveReader(bridge=bridge, app_root=APP_ROOT)
PX6D_REFERENCE_CONFIG = _load_px6d_reference_config()
px6d_reader = Px6dReader(PX6D_REFERENCE_CONFIG)
recorded_demo_library = RecordedDemoLibrary(APP_ROOT / "assets" / "demo")


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
    frame = bridge.frame(channel_id="P22", trace_limit=1, include_spectrum=True)
    latest = frame.get("latest")
    if not isinstance(latest, dict):
        return frame

    model_input = bridge.spectral_model_input("P22")
    baseline = model_input.get("baseline")
    if not model_input.get("baseline_ready") or not isinstance(baseline, dict):
        return frame

    normalization_config = dict(
        BAYSPEC_CHANNEL_CONFIG.get("spectrum_normalization", {}) or {}
    )
    normalized = normalize_spectrum_to_baseline_ratio(
        list(latest.get("wavelength_nm") or []),
        list(latest.get("intensity") or []),
        list(baseline.get("wavelength_nm") or []),
        list(baseline.get("intensity") or []),
        minimum_reference_counts=float(
            normalization_config.get("minimum_reference_counts") or 100.0
        ),
        minimum_valid_fraction=float(
            normalization_config.get("minimum_valid_fraction") or 0.80
        ),
    )
    if normalized.get("ok"):
        latest["normalized_intensity_ratio"] = list(
            normalized["normalized_intensity_ratio"]
        )
        latest["normalization_reference_intensity_counts"] = list(
            normalized["normalization_reference_intensity_counts"]
        )
    latest["spectrum_normalization"] = {
        **dict(latest.get("spectrum_normalization") or {}),
        **{
            key: value
            for key, value in normalized.items()
            if key
            not in {
                "normalized_intensity_ratio",
                "normalization_reference_intensity_counts",
            }
        },
        "raw_retained": True,
        "model_input_source": "raw_intensity",
        "applied_to_model_input": False,
    }
    return frame


def _capture_temporal_response(latest: dict[str, Any]) -> dict[str, Any]:
    prediction = _predict_current_runtime(latest_override=latest)
    return {
        "model_source": "ordinary_fbg_all_data_beta_v1",
        "model_status": prediction.get("status"),
        "model_ready": bool(
            prediction.get("ok") and prediction.get("status") == "ready"
        ),
        "inference_latency_ms": prediction.get(
            "backend_inference_latency_ms",
            prediction.get("inference_latency_ms"),
        ),
        "estimated_force_fz_n": prediction.get("estimated_force_fz_n"),
        "force_sensor_is_runtime_input": False,
        "runtime_input": "optical_spectrum_time_series",
        "drives_operator_ui": True,
        "drives_digital_twin": True,
        **prediction,
    }


def _resolve_capture_output_root() -> Path:
    runtime_override = os.environ.get("TOUCH_CAPTURE_OUTPUT_ROOT")
    if runtime_override:
        return Path(runtime_override).expanduser().resolve()
    configured = Path(str(PX6D_REFERENCE_CONFIG.get("capture_output_directory") or "data/px6d_synchronized"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _capture_provenance_snapshot() -> dict[str, Any]:
    baseline_token, model_pair = _current_runtime_baseline_token()
    model_status = _current_runtime_status()
    sdk_status = sdk_live_reader.status()
    force_status = px6d_reader.status()
    attested_by = str(
        GLOBAL_BASELINE_ATTESTATION.get("attested_by") or ""
    ).strip()
    configuration_paths = {
        "bayspec_channel_config": Path(CHANNEL_CONFIG_PATH),
        "runtime_contact_state": RUNTIME_CONTACT_STATE_CONFIG_PATH,
        "px6d_reference": PX6D_REFERENCE_CONFIG_PATH,
        "thumb_scene": THUMB_SCENE_CONFIG_PATH,
        "mfbg_intensity_profile": PROJECT_ROOT / "config" / "mfbg_intensity_3x3.yaml",
        "hybrid_spectrum_channels": HYBRID_SPECTRUM_CHANNEL_CONFIG_PATH,
    }
    return {
        "software": dict(RELEASE_IDENTITY),
        "operator": {
            "operator_id": attested_by or "not_provided",
            "source": (
                "baseline_no_contact_attestation"
                if attested_by
                else "not_recorded"
            ),
        },
        "specimen": {
            "specimen_id": os.environ.get("TOUCH_SPECIMEN_ID")
            or "not_provided",
            "fabrication_batch": os.environ.get("TOUCH_SPECIMEN_BATCH")
            or "not_provided",
            "mounting_state": os.environ.get("TOUCH_SPECIMEN_MOUNTING")
            or "not_recorded",
        },
        "environment": {
            "temperature_c": None,
            "humidity_percent": None,
            "status": "not_recorded",
        },
        "active_runtime_sensor_profile": "ordinary_fbg_hybrid_spectral",
        "future_sensor_profile": {
            "profile_id": "mfbg_intensity_3x3",
            "real_3x3_enabled": False,
        },
        "model": {
            "runtime_role": "primary_operator_and_digital_twin",
            "model_path": model_status.get("model_path"),
            "model_bundle_sha256": _artifact_identity(
                CURRENT_RUNTIME_MODEL_PATH
            ).get("sha256"),
            "loaded": model_status.get("loaded"),
            "confidence_source": model_status.get("confidence_source"),
            "evaluation_scope": model_status.get("evaluation_scope"),
        },
        "configuration_artifacts": {
            name: _artifact_identity(path)
            for name, path in configuration_paths.items()
        },
        "baseline": {
            "ready": bool(model_pair.get("ok")),
            "token": baseline_token,
            "status": model_pair.get("baseline_spectrum_status")
            or model_pair.get("reason"),
            "operator_attestation": dict(GLOBAL_BASELINE_ATTESTATION),
        },
        "optical_device": {
            "configured_device_id": configured_device_id(),
            "configured_sense_export_root": str(
                configured_sense_export_root()
            ),
            **{
                key: sdk_status.get(key)
                for key in (
                    "source",
                    "active",
                    "channel_id",
                    "integration",
                    "interval_ms",
                    "freshness",
                    "acquisition_session_id",
                    "frame_count",
                    "helper_path",
                )
            },
        },
        "force_device": {
            key: force_status.get(key)
            for key in (
                "connected",
                "port",
                "baud_rate",
                "firmware_version",
                "tare_ready",
                "tare_timestamp_epoch_sec",
                "compression_sign",
                "primary_axis",
                "sample_fresh",
                "configured_poll_hz",
            )
        },
        "force_calibration": {
            "hardware_calibration_certificate_id": "not_provided",
            "hardware_calibration_date": "not_provided",
            "hardware_calibration_traceability": "not_recorded",
            "software_tare_ready": force_status.get("tare_ready"),
            "software_tare_timestamp_epoch_sec": force_status.get(
                "tare_timestamp_epoch_sec"
            ),
            "hardware_calibration_command_used": force_status.get(
                "hardware_calibration_command_used",
                False,
            ),
        },
        "channel_grid": {
            "channel_order": list(CHANNEL_ORDER),
            "display_rows": [
                ["P11", "P21", "P31"],
                ["P12", "P22", "P32"],
                ["P13", "P23", "P33"],
            ],
            "physical_mapping_final": False,
            "five_finger_view_semantics": (
                "shared_3x3_response_visualization_not_five_independent_sensors"
            ),
        },
        "recording_contract": {
            "formal_spectrum_requires_physical_wavelength_axis": True,
            "pixel_index_fallback_allowed_for_formal_recognition": False,
            "real_five_finger_simultaneous_measurement_enabled": False,
        },
    }


optical_force_capture = OpticalForceCaptureManager(
    output_root=_resolve_capture_output_root(),
    frame_provider=_capture_spectrum_frame,
    force_provider=_px6d_reference_for_record,
    force_status_provider=px6d_reader.status,
    model_provider=_capture_temporal_response,
    provenance_provider=_capture_provenance_snapshot,
    poll_interval_sec=float(PX6D_REFERENCE_CONFIG.get("capture_poll_interval_sec") or 0.05),
    require_software_tare=bool(
        PX6D_REFERENCE_CONFIG.get("capture_require_software_tare", True)
    ),
)
PX6D_CAPTURE_CONTROL_LOCK = threading.Lock()

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
            or "normalized_surface_activity_thresholds_only"
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


def _simulated_array_channels(
    scenario: str,
    step: int = 0,
    coupling_view: str = "raw_coupled_response",
    response_amplitude: float | None = None,
) -> list[dict]:
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
    if response_amplitude is not None:
        envelope = max(0.0, min(1.0, float(response_amplitude)))
    if scenario == "no_contact":
        centers = [(0.0, 0.0, 0.0, 0.52)]
    elif scenario in _STAGED_POINT_CONTACT_CENTERS:
        cx, cy = _STAGED_POINT_CONTACT_CENTERS[scenario]
        centers = [(cx, cy, envelope, 0.52)]
    elif scenario == "off_center_fingertip_contact":
        # The recorded reference for this scenario is P23. Keep the visual
        # contact center aligned with that measured position.
        centers = [(0.0, -0.72, 0.84 * envelope, 0.58)]
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
    desired_force_n = 5.0 * _demo_envelope(scenario, step)
    recorded_reference = recorded_demo_library.reference_frame(
        scenario=scenario,
        step=step,
        desired_force_n=desired_force_n,
    )
    recorded_force_n = float(recorded_reference["reference_force_fz_n"])
    recorded_response_ratio = float(recorded_reference["response_ratio"])
    channels = _simulated_array_channels(
        scenario,
        step=step,
        coupling_view=coupling_view,
        response_amplitude=recorded_response_ratio,
    )
    surface = map_surface(channels, config=config)
    matrices = matrices_from_channels(channels)
    spectrum = recorded_demo_library.spectrum_payload(
        recorded_reference,
        channels=channels,
        frame_id=frame_id,
        timestamp=timestamp,
    )
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
        "recorded_source_sync_status": "same_frame_spectrum_and_px6d_fz",
        "last_update_timestamp": timestamp,
        "mode": "simulated_array_demo",
        "demo_data_mode": "recorded_real_spectrum_reference",
        "demo_reference_kind": recorded_reference.get("demo_reference_kind"),
        "demo_source_session_id": recorded_reference.get("source_session_id"),
        "demo_source_capture_index": recorded_reference.get("source_capture_index"),
        "demo_source_elapsed_time_sec": recorded_reference.get(
            "source_elapsed_time_sec"
        ),
        "demo_source_position": recorded_reference.get("source_position"),
        "demo_dataset_id": recorded_reference.get("dataset_id"),
        "demo_capture_date": recorded_reference.get("capture_date"),
        "recorded_reference_force_fz_n": recorded_force_n,
        "estimated_force_fz_n": recorded_force_n,
        "optical_force_estimate": {
            "estimated_n": recorded_force_n,
            "source": "recorded_synchronized_px6d_reference_for_demo",
            "calibrated_force_output": False,
        },
        "active_spectral_model_source": "recorded_demo_reference",
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
        "array_mode": "recorded_spectrum_visual_proxy_demo",
        "array_status": "recorded real spectrum with visual surface proxy",
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
            "Visual surface proxy driven by the synchronized recorded Fz reference; "
            "the displayed 512-point spectrum is recorded real BaySpec data."
        ),
        "surface_title": "Raw coupled Bragg wavelength-shift surface",
        "surface_subtitle": (
            "Recorded spectrum reference with force-scaled visual deformation. "
            "Not a measured pressure field."
        ),
    }


def startup_reference_sources() -> None:
    if PX6D_REFERENCE_CONFIG.get("enabled", True) and PX6D_REFERENCE_CONFIG.get(
        "auto_start", True
    ):
        px6d_reader.start()


def _px6d_runtime_status() -> dict[str, Any]:
    status_payload = px6d_reader.status()
    should_self_heal = bool(
        PX6D_REFERENCE_CONFIG.get("enabled", True)
        and PX6D_REFERENCE_CONFIG.get("auto_start", True)
        and not status_payload.get("worker_alive")
        and not status_payload.get("stop_requested")
        and status_payload.get("lifecycle_status") == "worker_exited"
    )
    if should_self_heal:
        return px6d_reader.start()
    return status_payload


def shutdown_live_sources() -> None:
    with LIVE_SOURCE_CONTROL_LOCK:
        optical_force_capture.stop()
        px6d_reader.set_auto_zero_frozen(False)
        sdk_live_reader.stop()
        export_watcher.stop()
        px6d_reader.stop()


@asynccontextmanager
async def application_lifespan(_app: FastAPI):
    startup_reference_sources()
    try:
        yield
    finally:
        shutdown_live_sources()


app = FastAPI(
    title="TOUCH",
    version=str(RELEASE_IDENTITY.get("version") or "unknown"),
    lifespan=application_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(mfbg_intensity_router)
app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")

CURRENT_RECOGNITION_SCOPE = "optical_contact_position_and_continuous_fz"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict:
    runtime_status = _current_runtime_status()
    return {
        "ok": True,
        "app": "TOUCH",
        "version": RELEASE_IDENTITY.get("version"),
        "build_id": RELEASE_IDENTITY.get("build_id"),
        "source_commit": RELEASE_IDENTITY.get("source_commit"),
        "release": dict(RELEASE_IDENTITY),
        "mode": "standalone_touch_all_data_spectral_runtime",
        "backend_contract_version": "touch_current_runtime_api_v1",
        "previous_p22_pd_voltage_app": "kept_separate",
        "optical_intensity_edition": "kept_separate",
        "demodulation_mode": "current_optical_force_runtime",
        "recognition_scope": CURRENT_RECOGNITION_SCOPE,
        "primary_signal": "bayspec_full_spectrum_time_series",
        "diagnostic_spectrum_scope": "global_9fbg_wavelength_intensity_area_shape",
        "carrier_channel_id": "P22",
        "carrier_channel_role": "full_spectrum_transport_for_current_runtime",
        "temperature_strain_decoupled": False,
        "array_mode": "global_spectrum_unmapped",
        "physical_channel_mapping_final": False,
        "real_3x3_enabled": False,
        "default_operator_recognition": "ordinary_fbg_all_data_beta_v1",
        "mfbg_intensity_profile_available": True,
        "future_primary_sensor_profile": "mfbg_intensity_3x3",
        "active_runtime_sensor_profile": "ordinary_fbg_hybrid_spectral",
        "sensor_profile_isolation": True,
        "position_output_semantics": "approximate_manual_fingertip_contact_region",
        "response_level_semantics": "continuous_optical_fz_estimate_no_fixed_upper_limit",
        "response_band_thresholds": _operator_response_band_thresholds(),
        "not_pd_voltage": True,
        "calibrated_physical_output": False,
        "runtime_model": runtime_status,
        "observed_training_force_range_n": runtime_status[
            "observed_training_force_range_n"
        ],
        "validated_force_range_n": runtime_status[
            "observed_training_force_range_n"
        ],
        "force_above_observed_training_range": (
            "reported_as_unvalidated_extrapolation_without_upper_clip"
        ),
        "force_sensor_is_runtime_model_input": False,
        "array_wavelength_plan": _array_wavelength_plan_payload(),
        "ui_style": "lab_light_digital_twin_like_previous_app",
        "status": bridge.status(),
        "export_watcher": export_watcher.status(),
        "sdk_live": sdk_live_reader.status(),
        "sense_control": sense_controller.status(),
        "px6d_reference": _px6d_runtime_status(),
        "optical_force_capture": optical_force_capture.status(),
        "recorded_demo": recorded_demo_library.status(),
        "recognition_runtime": {
            "active_model_id": "ordinary_fbg_all_data_beta_v1",
            "display_name": "Current all-data spectral model",
            "switchable": False,
            "model_count": 1,
        },
    }


@app.get("/api/status")
def status() -> dict:
    result = bridge.status()
    result["px6d_reference"] = _px6d_runtime_status()
    result["optical_force_capture"] = optical_force_capture.status()
    return result


@app.get("/api/px6d/status")
def px6d_status() -> dict:
    return {"ok": True, "mode": "px6d_reference_force", **_px6d_runtime_status()}


@app.post("/api/px6d/start")
def px6d_start() -> dict:
    status_payload = px6d_reader.start()
    return {"ok": True, "mode": "px6d_reference_force", **status_payload}


@app.post("/api/px6d/stop")
def px6d_stop() -> dict:
    status_payload = px6d_reader.stop()
    return {"ok": True, "mode": "px6d_reference_force", **status_payload}


@app.post("/api/px6d/reconnect")
def px6d_reconnect() -> dict:
    stop_payload = px6d_reader.stop()
    if not stop_payload.get("ok", False):
        return {
            "ok": False,
            "mode": "px6d_reference_force",
            "operation_status": "reconnect_stop_failed",
            **stop_payload,
        }
    start_payload = px6d_reader.start()
    return {
        "mode": "px6d_reference_force",
        "operation_status": (
            "reconnect_started"
            if start_payload.get("ok", False)
            else "reconnect_start_failed"
        ),
        **start_payload,
    }


@app.post("/api/px6d/tare")
def px6d_tare(duration_sec: float = Query(default=1.0, ge=0.25, le=5.0)) -> dict:
    if not PX6D_CAPTURE_CONTROL_LOCK.acquire(blocking=False):
        return {
            "ok": False,
            "mode": "px6d_software_tare",
            "status": "capture_control_busy",
            "reason": "wait_for_recording_or_force_zero_command_to_finish",
            "hardware_calibration_command_used": False,
        }
    try:
        capture_status = optical_force_capture.status()
        if any(
            bool(capture_status.get(field))
            for field in ("running", "start_in_progress", "worker_alive")
        ):
            return {
                "ok": False,
                "mode": "px6d_software_tare",
                "status": "recording_active",
                "reason": "stop_synchronized_recording_before_force_zero",
                "hardware_calibration_command_used": False,
            }
        result = px6d_reader.tare(duration_sec=duration_sec)
        result.update(
            {
                "mode": "px6d_software_tare",
                "hardware_calibration_command_used": False,
            }
        )
        return result
    finally:
        PX6D_CAPTURE_CONTROL_LOCK.release()


@app.get("/api/px6d/latest")
def px6d_latest() -> dict:
    _px6d_runtime_status()
    result = px6d_reader.latest()
    sample_present = bool(result.pop("ok", False))
    sample_fresh = bool((result.get("status") or {}).get("sample_fresh"))
    result["sample_present"] = sample_present
    result["sample_fresh"] = sample_fresh
    result["sample_ready"] = bool(sample_present and sample_fresh)
    result["ok"] = True
    result["mode"] = "px6d_reference_force"
    return result


@app.get("/api/px6d/trace")
def px6d_trace(limit: int = Query(default=500, ge=1, le=20000)) -> dict:
    _px6d_runtime_status()
    result = px6d_reader.trace(limit=limit)
    result["mode"] = "px6d_reference_force_trace"
    return result


def _measurement_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _measurement_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_measurement_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _measurement_default_root() -> Path:
    capture_status = optical_force_capture.status()
    configured = (
        capture_status.get("requested_output_root")
        or capture_status.get("default_output_root")
    )
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (Path.home() / "Documents" / "TOUCH" / "captures").resolve()


def _measurement_has_trace(session_dir: Path) -> bool:
    return any(
        (session_dir / filename).is_file()
        for filename in ("frame_summary.csv", "synchronized_frames.jsonl")
    )


def _measurement_csv_fields(session_dir: Path) -> set[str]:
    summary_path = session_dir / "frame_summary.csv"
    if not summary_path.is_file():
        return set()
    try:
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            return set(next(csv.reader(handle), []))
    except OSError:
        return set()


def _measurement_session_metadata(session_dir: Path) -> dict[str, Any]:
    metadata_path = session_dir / "session_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                metadata = payload
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata = {}
    fields = _measurement_csv_fields(session_dir)
    selected_outputs = metadata.get("selected_outputs")
    selected_outputs = selected_outputs if isinstance(selected_outputs, list) else []
    modified_at = max(
        (
            candidate.stat().st_mtime
            for candidate in (
                session_dir / "frame_summary.csv",
                session_dir / "synchronized_frames.jsonl",
                metadata_path,
            )
            if candidate.is_file()
        ),
        default=session_dir.stat().st_mtime,
    )
    return {
        "session_id": str(metadata.get("session_id") or session_dir.name),
        "path": str(session_dir.resolve()),
        "position_label": str(metadata.get("position_label") or "unlabeled"),
        "trial_id": str(metadata.get("trial_id") or "--"),
        "started_at_epoch_sec": metadata.get("started_at_epoch_sec"),
        "ended_at_epoch_sec": metadata.get("ended_at_epoch_sec"),
        "captured_timeline_frames": metadata.get("captured_timeline_frames"),
        "selected_outputs": selected_outputs,
        "has_force_reference": bool(
            "force" in selected_outputs
            or "force_fz_n" in fields
            or (session_dir / "force_timeseries.csv").is_file()
        ),
        "has_optical_estimate": bool(
            "optical_estimated_fz_n" in fields
            or (session_dir / "synchronized_frames.jsonl").is_file()
        ),
        "modified_at_epoch_sec": modified_at,
    }


def _measurement_sessions(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not root.exists() or not root.is_dir():
        return []
    candidates = [root] if _measurement_has_trace(root) else [
        child
        for child in root.iterdir()
        if child.is_dir() and _measurement_has_trace(child)
    ]
    sessions = [_measurement_session_metadata(candidate) for candidate in candidates]
    sessions.sort(
        key=lambda item: float(item.get("modified_at_epoch_sec") or 0.0),
        reverse=True,
    )
    return sessions[:limit]


def _downsample_measurement_trace(
    rows: list[dict[str, Any]], limit: int = 600
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        selected = rows
    else:
        indices = np.linspace(0, len(rows) - 1, num=limit, dtype=int)
        selected = [rows[int(index)] for index in np.unique(indices)]
    fields = (
        "capture_index",
        "elapsed_time_sec",
        "reference_fz_n",
        "analysis_estimated_fz_n",
        "analysis_raw_estimated_fz_n",
        "analysis_estimate_source",
        "recorded_runtime_optical_estimated_fz_n",
        "optical_estimated_fz_n",
        "force_phase",
        "cycle_id",
    )
    return [
        {field: _measurement_json_safe(row.get(field)) for field in fields}
        for row in selected
    ]


def _measurement_estimate_evidence(
    session_dir: Path,
    trace_rows: list[dict[str, Any]],
    requested_source: str,
    analysis_config: MeasurementAnalysisConfig,
) -> dict[str, Any]:
    """Resolve one scientifically coherent force-estimate curve.

    The automatic order favors grouped out-of-fold predictions. Current-model
    replay is second because a recording may also have been used to train that
    model. Recorded runtime output remains available as historical evidence,
    but is never mixed frame-by-frame with either newer source.
    """

    return resolve_measurement_estimate_evidence(
        session_dir,
        trace_rows,
        requested_source,
        outputs_root=PROJECT_ROOT / "outputs",
        model_path=CURRENT_RUNTIME_MODEL_PATH,
        peak_config_path=HYBRID_SPECTRUM_CHANNEL_CONFIG_PATH,
        runtime_recovery_config=_load_runtime_baseline_recovery_config(),
        runtime_gate_config=_load_all_source_runtime_gate_config(),
        baseline_frame_count=analysis_config.replay_baseline_frame_count,
        baseline_strategy=analysis_config.replay_baseline_strategy,
        baseline_minimum_stable_frames=(
            analysis_config.replay_baseline_minimum_stable_frames
        ),
        baseline_stability_mad_multiplier=(
            analysis_config.replay_baseline_stability_mad_multiplier
        ),
    )


@app.get("/api/px6d_capture/status")
def px6d_capture_status() -> dict:
    capture_status = optical_force_capture.status()
    force_status = _px6d_runtime_status()
    if (
        not capture_status.get("running")
        and not capture_status.get("start_in_progress")
        and force_status.get("auto_zero_frozen")
        and force_status.get("auto_zero_freeze_reason") == "synchronized_recording"
    ):
        px6d_reader.set_auto_zero_frozen(False)
    return {
        "ok": True,
        "mode": "optical_px6d_synchronized_capture",
        **capture_status,
    }


@app.post("/api/px6d_capture/start")
async def px6d_capture_start(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        return {
            "ok": False,
            "mode": "optical_px6d_synchronized_capture",
            "status": "capture_request_invalid",
            "reason": f"request body must be JSON: {type(exc).__name__}",
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "mode": "optical_px6d_synchronized_capture",
            "status": "capture_request_invalid",
            "reason": "request body must be a JSON object",
        }
    def start_with_force_zero_guard() -> dict:
        if not PX6D_CAPTURE_CONTROL_LOCK.acquire(blocking=False):
            return {
                "ok": False,
                "status": "force_zero_in_progress",
                "reason": "wait_for_force_zero_to_finish_before_recording",
            }
        try:
            px6d_reader.set_auto_zero_frozen(
                True,
                reason="synchronized_recording",
            )
            try:
                result = optical_force_capture.start(
                    position_label=str(payload.get("position_label") or "unlabeled"),
                    action_label=str(payload.get("action_label") or "unlabeled"),
                    trial_id=str(payload.get("trial_id") or "trial_001"),
                    operator_note=str(payload.get("operator_note") or ""),
                    output_root=payload.get("output_root"),
                    selected_outputs=payload.get("selected_outputs"),
                )
            except Exception:
                px6d_reader.set_auto_zero_frozen(False)
                raise
            if not result.get("ok"):
                px6d_reader.set_auto_zero_frozen(False)
            return result
        finally:
            PX6D_CAPTURE_CONTROL_LOCK.release()

    result = await asyncio.to_thread(start_with_force_zero_guard)
    return {"mode": "optical_px6d_synchronized_capture", **result}


@app.post("/api/px6d_capture/stop")
def px6d_capture_stop() -> dict:
    try:
        result = optical_force_capture.stop()
    except Exception:
        if not optical_force_capture.status().get("running"):
            px6d_reader.set_auto_zero_frozen(False)
        raise
    if not result.get("running"):
        px6d_reader.set_auto_zero_frozen(False)
    return {"mode": "optical_px6d_synchronized_capture", **result}


@app.get("/api/measurement/sessions")
def measurement_sessions(
    root: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=250),
) -> dict:
    selected_root = (
        Path(root).expanduser().resolve()
        if root and str(root).strip()
        else _measurement_default_root()
    )
    sessions = _measurement_sessions(selected_root, limit=limit)
    return _measurement_json_safe(
        {
            "ok": True,
            "mode": "diagnostics_measurement_analysis",
            "root": str(selected_root),
            "session_count": len(sessions),
            "sessions": sessions,
        }
    )


@app.post("/api/measurement/analyze")
async def measurement_analyze(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        return {
            "ok": False,
            "mode": "diagnostics_measurement_analysis",
            "status": "analysis_request_invalid",
            "reason": f"request body must be JSON: {type(exc).__name__}",
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "mode": "diagnostics_measurement_analysis",
            "status": "analysis_request_invalid",
            "reason": "request body must be a JSON object",
        }
    requested_estimate_source = str(
        payload.get("estimate_source") or "best_available"
    ).strip().lower()
    if requested_estimate_source not in EVIDENCE_SOURCES:
        return {
            "ok": False,
            "mode": "diagnostics_measurement_analysis",
            "status": "measurement_estimate_source_invalid",
            "reason": f"unsupported estimate source: {requested_estimate_source}",
            "supported_estimate_sources": list(EVIDENCE_SOURCES),
        }
    raw_session_dir = str(payload.get("session_dir") or "").strip()
    if not raw_session_dir:
        return {
            "ok": False,
            "mode": "diagnostics_measurement_analysis",
            "status": "session_required",
            "reason": "select a completed recording session",
        }
    session_dir = Path(raw_session_dir).expanduser().resolve()
    if not session_dir.is_dir() or not _measurement_has_trace(session_dir):
        return {
            "ok": False,
            "mode": "diagnostics_measurement_analysis",
            "status": "measurement_trace_not_found",
            "reason": "frame_summary.csv or synchronized_frames.jsonl was not found",
        }
    capture_status = optical_force_capture.status()
    active_output = capture_status.get("output_directory")
    if capture_status.get("running") and active_output:
        try:
            active_matches = Path(str(active_output)).expanduser().resolve() == session_dir
        except OSError:
            active_matches = False
        if active_matches:
            return {
                "ok": False,
                "mode": "diagnostics_measurement_analysis",
                "status": "recording_still_active",
                "reason": "stop and save the recording before analysis",
            }
    try:
        config = load_measurement_config(
            MEASUREMENT_ANALYSIS_CONFIG_PATH
            if MEASUREMENT_ANALYSIS_CONFIG_PATH.is_file()
            else None
        )
        trace_rows = await asyncio.to_thread(load_measurement_trace, session_dir)
        evidence = await asyncio.to_thread(
            _measurement_estimate_evidence,
            session_dir,
            trace_rows,
            requested_estimate_source,
            config,
        )
        if not evidence.get("ok"):
            return _measurement_json_safe(
                {
                    "ok": False,
                    "mode": "diagnostics_measurement_analysis",
                    "status": evidence.get(
                        "status", "measurement_estimate_source_unavailable"
                    ),
                    "reason": evidence.get("reason"),
                    "requested_estimate_source": requested_estimate_source,
                    "evidence": evidence,
                }
            )
        result = await asyncio.to_thread(
            analyze_measurement_session,
            session_dir,
            config,
            estimate_overlay=evidence.get("overlay"),
            estimate_source_info=evidence,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "ok": False,
            "mode": "diagnostics_measurement_analysis",
            "status": "analysis_failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return _measurement_json_safe(
        {
            "ok": True,
            "mode": "diagnostics_measurement_analysis",
            "status": "complete",
            "requested_estimate_source": requested_estimate_source,
            "selected_estimate_source": evidence.get("source"),
            "estimate_evidence": {
                key: value
                for key, value in evidence.items()
                if key != "overlay"
            },
            "session": _measurement_session_metadata(session_dir),
            "summary": result["summary"],
            "cycle_rows": result["cycle_rows"],
            "trace": _downsample_measurement_trace(result["trace_rows"]),
        }
    )


@app.post("/api/reset")
def reset(keep_baseline: bool = Query(default=True)) -> dict:
    with LIVE_SOURCE_CONTROL_LOCK:
        result = bridge.reset(keep_baseline=keep_baseline)
        result["runtime_model_reset"] = _reset_current_runtime("api_reset")
        result.update({"mode": "bayspec_wavelength_shift_reset"})
        return result


@app.post("/api/ingest")
async def ingest(request: Request) -> Any:
    try:
        body = await _read_limited_ingest_body(request)
        payload = json.loads(body.decode("utf-8"))
    except IngestRequestTooLarge as exc:
        return JSONResponse(
            status_code=413,
            content={
                "ok": False,
                "status": "ingest_request_too_large",
                "reason": str(exc),
                "maximum_body_bytes": MAX_MANUAL_INGEST_BODY_BYTES,
            },
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"ok": False, "reason": "request body must be JSON"}
    with LIVE_SOURCE_CONTROL_LOCK:
        conflict = _manual_ingest_source_conflict()
        if conflict is not None:
            return conflict
        result = bridge.ingest(payload)
        result.update({"mode": "bayspec_wavelength_shift_json_ingest"})
        return result


@app.post("/api/ingest_csv")
async def ingest_csv(
    request: Request,
    channel_id: str = Query(default="P22"),
    device_id: str | None = Query(default=None),
) -> Any:
    try:
        body = await _read_limited_ingest_body(request)
    except IngestRequestTooLarge as exc:
        return JSONResponse(
            status_code=413,
            content={
                "ok": False,
                "status": "ingest_request_too_large",
                "reason": str(exc),
                "maximum_body_bytes": MAX_MANUAL_INGEST_BODY_BYTES,
            },
        )
    text = body.decode("utf-8", errors="ignore")
    with LIVE_SOURCE_CONTROL_LOCK:
        conflict = _manual_ingest_source_conflict()
        if conflict is not None:
            return conflict
        result = bridge.ingest_csv_text(
            text,
            channel_id=channel_id,
            device_id=device_id,
        )
        result.update({"mode": "bayspec_wavelength_shift_csv_ingest"})
        return result


@app.post("/api/ingest_latest_export")
def ingest_latest_export(
    channel_id: str = Query(default="P22"),
    export_root: str | None = Query(default=None),
) -> dict:
    with LIVE_SOURCE_CONTROL_LOCK:
        conflict = _manual_ingest_source_conflict()
        if conflict is not None:
            return conflict
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

    runtime_model_reset = _reset_current_runtime(
        "new_acquisition_session"
    )
    result = bridge.reset(keep_baseline=False)
    return {
        **result,
        "baseline_invalidated": True,
        "baseline_requirement": "stable_current_session_post_release_recovery",
        "runtime_model_reset": runtime_model_reset,
    }


def _sdk_session_matches(
    status: dict[str, Any],
    *,
    channel_id: str,
    interval_ms: int,
    integration: int,
) -> bool:
    """Return true when a start request already describes the live session."""

    return bool(
        status.get("active")
        and str(status.get("channel_id") or "") == str(channel_id)
        and int(status.get("interval_ms") or 0) == int(interval_ms)
        and int(status.get("integration") or 0) == int(integration)
    )


def _export_watch_session_matches(
    status: dict[str, Any],
    *,
    channel_id: str,
    export_root: str | None,
    interval_sec: float,
) -> bool:
    requested_root = (
        str(Path(export_root).expanduser().resolve()) if export_root else None
    )
    current_root_value = status.get("export_root")
    current_root = (
        str(Path(current_root_value).expanduser().resolve())
        if current_root_value
        else None
    )
    return bool(
        status.get("active")
        and status.get("worker_alive", True)
        and str(status.get("channel_id") or "") == str(channel_id)
        and current_root == requested_root
        and abs(float(status.get("interval_sec") or 0.0) - float(interval_sec))
        <= 1e-9
    )


def _unchanged_acquisition_session() -> dict[str, Any]:
    return {
        "ok": True,
        "baseline_invalidated": False,
        "status": "already_running_same_session",
        "reason": "idempotent_start_request",
    }


def _live_source_stop_completed(status: dict[str, Any]) -> bool:
    """Return true only when a source switch can safely reset shared state."""

    if status.get("ok") is False:
        return False
    # Treat any remaining producer or in-flight ingest as live, even when a
    # buggy adapter forgot to set stop_requested or optimistically returned
    # ok=True. A source switch is safe only after the old source is quiescent.
    return not any(
        bool(status.get(field))
        for field in (
            "active",
            "requested_active",
            "worker_alive",
            "process_running",
            "ingest_in_progress",
            "start_in_progress",
            "start_cancel_requested",
        )
    )


def _manual_ingest_source_conflict() -> dict[str, Any] | None:
    """Reject one-shot imports while a background producer owns the bridge."""

    sources = {
        "sdk_live": sdk_live_reader.status(),
        "export_watch": export_watcher.status(),
    }
    busy_sources = [
        name
        for name, status in sources.items()
        if not _live_source_stop_completed({"ok": True, **status})
    ]
    if not busy_sources:
        return None
    return {
        "ok": False,
        "mode": "manual_ingest_blocked",
        "status": "live_source_active",
        "reason": "stop_live_source_before_manual_ingest",
        "busy_sources": busy_sources,
        "source_status": sources,
    }


def _source_switch_blocked(
    *,
    source: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": False,
        "mode": "live_source_switch_blocked",
        "status": "previous_live_source_not_stopped",
        "reason": f"{source}_stop_failed",
        "message": (
            f"Cannot switch acquisition source because {source} did not stop "
            "cleanly. Retry stop before starting another source."
        ),
        source: status,
        "acquisition_session_reset": {
            "ok": False,
            "baseline_invalidated": False,
            "status": "not_started",
            "reason": "previous_live_source_not_stopped",
        },
    }


def _serialized_live_source_control(operation):
    """Serialize source switching and baseline-invalidating operations.

    FastAPI executes synchronous routes in a worker pool. Without one shared
    guard, simultaneous SDK/export requests can both pass their preflight
    checks, reset the session twice, and leave two producers writing into the
    same bridge. The re-entrant lock also keeps direct test calls and lifespan
    shutdown on the same lifecycle contract.
    """

    @wraps(operation)
    def guarded(*args, **kwargs):
        with LIVE_SOURCE_CONTROL_LOCK:
            return operation(*args, **kwargs)

    return guarded


@app.post("/api/export_watch/start")
@_serialized_live_source_control
def export_watch_start(
    channel_id: str = Query(default="P22"),
    export_root: str | None = Query(default=None),
    interval_sec: float = Query(default=0.35, ge=0.1, le=5.0),
) -> dict:
    sdk_status = sdk_live_reader.stop()
    if not _live_source_stop_completed(sdk_status):
        return _source_switch_blocked(source="sdk_live", status=sdk_status)
    existing_watch = export_watcher.status()
    requested_interval_sec = max(0.1, min(float(interval_sec), 5.0))
    if _export_watch_session_matches(
        existing_watch,
        channel_id=channel_id,
        export_root=export_root,
        interval_sec=requested_interval_sec,
    ):
        return {
            "ok": True,
            "mode": "sense_export_watch_already_running",
            "export_watcher": existing_watch,
            "sdk_live": sdk_status,
            "acquisition_session_reset": _unchanged_acquisition_session(),
        }
    if existing_watch.get("active") or existing_watch.get("ingest_in_progress"):
        stopped_watch = export_watcher.stop()
        if not _live_source_stop_completed(stopped_watch):
            return _source_switch_blocked(
                source="export_watcher",
                status=stopped_watch,
            )
    session_reset = _begin_acquisition_session()
    status = export_watcher.start(
        channel_id=channel_id,
        export_root=export_root,
        interval_sec=requested_interval_sec,
    )
    return {
        "ok": bool(status.get("ok", status.get("active")) and status.get("active")),
        "mode": "sense_export_watch_started",
        "export_watcher": status,
        "sdk_live": sdk_status,
        "acquisition_session_reset": session_reset,
    }


@app.post("/api/export_watch/stop")
@_serialized_live_source_control
def export_watch_stop() -> dict:
    status = export_watcher.stop()
    return {
        "ok": _live_source_stop_completed(status),
        "mode": "sense_export_watch_stopped",
        "export_watcher": status,
    }


@app.get("/api/export_watch/status")
def export_watch_status() -> dict:
    return {"ok": True, "mode": "sense_export_watch_status", "export_watcher": export_watcher.status()}


@app.post("/api/sdk/start")
@_serialized_live_source_control
def sdk_start(
    channel_id: str = Query(default="P22"),
    interval_ms: int = Query(default=DEFAULT_INTERVAL_MS, ge=20, le=2000),
    integration: int = Query(default=DEFAULT_INTEGRATION_US, ge=1, le=10000000),
) -> dict:
    existing_status = sdk_live_reader.status()
    requested_interval_ms = max(20, min(int(interval_ms), 2000))
    requested_integration = max(1, int(integration))
    if _sdk_session_matches(
        existing_status,
        channel_id=channel_id,
        interval_ms=requested_interval_ms,
        integration=requested_integration,
    ):
        return {
            "ok": True,
            "mode": "bayspec_direct_sdk_already_running",
            "sdk_live": existing_status,
            "acquisition_session_reset": _unchanged_acquisition_session(),
        }
    if existing_status.get("active") or existing_status.get("worker_alive"):
        stopped_status = sdk_live_reader.stop()
        if not _live_source_stop_completed(stopped_status):
            return _source_switch_blocked(
                source="sdk_live",
                status=stopped_status,
            )
    stopped_watch = export_watcher.stop()
    if not _live_source_stop_completed(stopped_watch):
        return _source_switch_blocked(
            source="export_watcher",
            status=stopped_watch,
        )
    session_reset = _begin_acquisition_session()
    status = sdk_live_reader.start(
        channel_id=channel_id,
        interval_ms=requested_interval_ms,
        integration=requested_integration,
    )
    return {
        "ok": bool(status.get("ok", status.get("active")) and status.get("active")),
        "mode": "bayspec_direct_sdk_started",
        "sdk_live": status,
        "acquisition_session_reset": session_reset,
        "live_signal_path": (
            "BaySpec USB20BS SDK helper -> spectrum frames -> Bragg peak tracking "
            "-> wavelength-shift digital twin"
        ),
    }


@app.post("/api/sdk/stop")
@_serialized_live_source_control
def sdk_stop() -> dict:
    status = sdk_live_reader.stop()
    return {
        "ok": _live_source_stop_completed(status),
        "mode": "bayspec_direct_sdk_stopped",
        "sdk_live": status,
    }


@app.get("/api/sdk/status")
def sdk_status() -> dict:
    return {"ok": True, "mode": "bayspec_direct_sdk_status", "sdk_live": sdk_live_reader.status()}


@app.get("/api/spectrum_processing/status")
def spectrum_processing_status() -> dict:
    return {
        "ok": True,
        "mode": "display_only_spectrum_processing",
        "spectrum_processing": sdk_live_reader.processing_status(),
    }


@app.patch("/api/spectrum_processing/settings")
async def spectrum_processing_settings(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "status": "invalid_settings_payload",
            "message": "Spectrum processing settings must be a JSON object.",
        }
    status = sdk_live_reader.update_processing_settings(payload)
    return {
        "ok": True,
        "mode": "display_only_spectrum_processing",
        "spectrum_processing": status,
    }


@app.post("/api/spectrum_processing/background/capture")
def spectrum_processing_capture_background() -> dict:
    result = sdk_live_reader.capture_background()
    return {
        **result,
        "mode": "display_only_spectrum_processing",
        "spectrum_processing": sdk_live_reader.processing_status(),
    }


@app.post("/api/spectrum_processing/background/clear")
def spectrum_processing_clear_background() -> dict:
    result = sdk_live_reader.clear_background()
    return {
        **result,
        "mode": "display_only_spectrum_processing",
        "spectrum_processing": sdk_live_reader.processing_status(),
    }


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
@_serialized_live_source_control
def live_start(
    channel_id: str = Query(default="P22"),
    export_root: str | None = Query(default=None),
    interval_sec: float = Query(default=0.02, ge=0.02, le=5.0),
    integration: int = DEFAULT_INTEGRATION_US,
    control_sense: bool = Query(default=True),
    source: str = Query(default="direct_sdk"),
) -> dict:
    if source == "direct_sdk":
        requested_interval_ms = max(20, min(int(interval_sec * 1000), 2000))
        requested_integration = max(1000, min(int(integration), 1000000))
        existing_status = sdk_live_reader.status()
        if _sdk_session_matches(
            existing_status,
            channel_id=channel_id,
            interval_ms=requested_interval_ms,
            integration=requested_integration,
        ):
            return {
                "ok": True,
                "mode": "bayspec_live_twin_already_running",
                "sdk_live": existing_status,
                "acquisition_session_reset": _unchanged_acquisition_session(),
                "sense_control": {
                    "ok": True,
                    "mode": "sense_control_not_required_for_direct_sdk",
                },
            }
        if existing_status.get("active") or existing_status.get("worker_alive"):
            stopped_status = sdk_live_reader.stop()
            if not _live_source_stop_completed(stopped_status):
                return _source_switch_blocked(
                    source="sdk_live",
                    status=stopped_status,
                )
        stopped_watch = export_watcher.stop()
        if not _live_source_stop_completed(stopped_watch):
            return _source_switch_blocked(
                source="export_watcher",
                status=stopped_watch,
            )
        session_reset = _begin_acquisition_session()
        sdk_status = sdk_live_reader.start(
            channel_id=channel_id,
            interval_ms=requested_interval_ms,
            integration=requested_integration,
        )
        return {
            "ok": bool(
                sdk_status.get("ok", sdk_status.get("active"))
                and sdk_status.get("active")
            ),
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
    if not _live_source_stop_completed(sdk_status):
        return _source_switch_blocked(source="sdk_live", status=sdk_status)
    existing_watch = export_watcher.status()
    # Export watching is disk-bound and gains nothing from the SDK's 20 ms
    # post-frame idle budget. Keep its established 100 ms lower bound.
    requested_interval_sec = max(0.1, min(float(interval_sec), 5.0))
    if _export_watch_session_matches(
        existing_watch,
        channel_id=channel_id,
        export_root=export_root,
        interval_sec=requested_interval_sec,
    ):
        return {
            "ok": True,
            "mode": "bayspec_live_twin_already_running",
            "export_watcher": existing_watch,
            "sdk_live": sdk_status,
            "acquisition_session_reset": _unchanged_acquisition_session(),
            "sense_control": {
                "ok": True,
                "mode": "sense_control_unchanged_existing_watch",
            },
        }
    if existing_watch.get("active") or existing_watch.get("ingest_in_progress"):
        stopped_watch = export_watcher.stop()
        if not _live_source_stop_completed(stopped_watch):
            return _source_switch_blocked(
                source="export_watcher",
                status=stopped_watch,
            )
    session_reset = _begin_acquisition_session()
    watch_status = export_watcher.start(
        channel_id=channel_id,
        export_root=export_root,
        interval_sec=requested_interval_sec,
    )
    sense_result = sense_controller.start_fast_recording(ensure_stopped=True) if control_sense else {
        "ok": True,
        "mode": "sense_control_skipped",
    }
    return {
        "ok": bool(
            watch_status.get("ok", watch_status.get("active"))
            and watch_status.get("active")
        ),
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
@_serialized_live_source_control
def live_stop(control_sense: bool = Query(default=True)) -> dict:
    sdk_status = sdk_live_reader.stop()
    watch_status = export_watcher.stop()
    sense_result = sense_controller.stop_scan() if control_sense else {"ok": True, "mode": "sense_control_skipped"}
    return {
        "ok": (
            _live_source_stop_completed(sdk_status)
            and _live_source_stop_completed(watch_status)
            and sense_result.get("ok") is not False
        ),
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
    with LIVE_SOURCE_CONTROL_LOCK:
        result = bridge.set_baseline(payload)
        if result.get("baseline_set") or result.get("current_runtime_spectrum_baseline_ready"):
            result["runtime_model_reset"] = _reset_current_runtime(
                "runtime_baseline_replaced"
            )
        result.update({"mode": "bayspec_wavelength_baseline_set"})
        return result


@app.post("/api/global_candidate_baseline")
@_serialized_live_source_control
def set_global_candidate_baseline(
    minimum_frames: int = Query(default=30, ge=3, le=500),
    no_contact_attested: bool = False,
    attested_by: str | None = None,
) -> dict:
    global GLOBAL_BASELINE_ATTESTATION

    force_status = px6d_reader.status()
    force_evidence = _px6d_reference_for_record(bridge.latest(channel_id="P22"))
    force_connected = bool(force_status.get("connected"))
    force_tare_ready = bool(
        force_status.get("tare_ready") and force_evidence.get("tare_ready")
    )
    force_sample_fresh = bool(
        force_status.get("sample_fresh")
        and force_evidence.get("sample_fresh", True)
    )
    force_reference_enforced = bool(
        force_connected and force_tare_ready and force_sample_fresh
    )
    force_fz_n = force_evidence.get("force_fz_n")
    try:
        force_near_zero = abs(float(force_fz_n)) <= 0.10
    except (TypeError, ValueError):
        force_near_zero = False
    if not no_contact_attested:
        GLOBAL_BASELINE_ATTESTATION = {
            "confirmed": False,
            "attested_at_epoch_sec": None,
            "attested_by": None,
            "force_evidence": force_evidence,
            "status": "operator_no_contact_attestation_required",
        }
        return {
            "ok": False,
            "status": "operator_no_contact_attestation_required",
            "reason": "operator_no_contact_attestation_required",
            "message": "Confirm that the tactile surface is released before setting baseline.",
            "baseline_unchanged": True,
            "no_contact_attestation": copy.deepcopy(GLOBAL_BASELINE_ATTESTATION),
        }
    if force_reference_enforced and not force_near_zero:
        GLOBAL_BASELINE_ATTESTATION = {
            "confirmed": False,
            "attested_at_epoch_sec": time.time(),
            "attested_by": str(attested_by or "operator"),
            "force_evidence": force_evidence,
            "status": "force_reference_not_released",
        }
        return {
            "ok": False,
            "status": "force_reference_not_released",
            "reason": "force_reference_not_released",
            "message": "Force reference is not near zero. Release contact before setting baseline.",
            "baseline_unchanged": True,
            "no_contact_attestation": copy.deepcopy(GLOBAL_BASELINE_ATTESTATION),
        }
    GLOBAL_BASELINE_ATTESTATION = {
        "confirmed": True,
        "attested_at_epoch_sec": time.time(),
        "attested_by": str(attested_by or "operator"),
        "force_evidence": force_evidence,
        "status": (
            "operator_and_force_reference_confirmed"
            if force_reference_enforced
            else "operator_confirmed_force_reference_pending_zero"
            if force_connected and not force_tare_ready
            else "operator_confirmed_force_reference_stale"
            if force_connected
            else "operator_confirmed_force_reference_unavailable"
        ),
    }
    model_baseline = bridge.set_baseline(
        {
            "channel_id": "P22",
            "baseline_method": "frozen_baseline",
            "minimum_recent_samples": minimum_frames,
        }
    )
    model_baseline_ready = bool(
        model_baseline.get("current_runtime_spectrum_baseline_ready")
    )
    result = (
        bridge.set_global_candidate_baseline(minimum_frames=minimum_frames)
        if model_baseline_ready
        else {
            "ok": False,
            "reason": model_baseline.get("current_runtime_spectrum_baseline_status")
            or model_baseline.get("reason")
            or "current_runtime_spectrum_baseline_not_ready",
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
            model_baseline.get("current_runtime_spectrum_baseline_status")
            or model_baseline.get("reason")
            or "current_runtime_spectrum_baseline_not_ready"
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
            "recognition_scope": CURRENT_RECOGNITION_SCOPE,
            "physical_channel_mapping_final": False,
            "formal_model_baseline": False,
            "no_contact_attestation": copy.deepcopy(
                GLOBAL_BASELINE_ATTESTATION
            ),
            "candidate_display_baseline_ok": candidate_baseline_ok,
            "current_runtime_spectrum_baseline": {
                "ok": model_baseline_ready,
                "baseline_set": bool(model_baseline.get("baseline_set")),
                "role": "post_press_release_recovery_no_contact_full_spectrum_baseline",
                "status": model_baseline.get("current_runtime_spectrum_baseline_status"),
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
                    else model_baseline.get("current_runtime_spectrum_baseline_status")
                    or model_baseline.get("reason")
                ),
            },
        }
    )
    if baseline_ready:
        result["runtime_model_reset"] = _reset_current_runtime(
            "runtime_baseline_replaced"
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
) -> dict:
    """Return one atomic frame driven by the sole deployed spectral model."""

    watcher_status = export_watcher.status()
    sdk_status = sdk_live_reader.status()
    result = bridge.frame(
        channel_id="P22",
        trace_limit=trace_limit,
        include_spectrum=include_spectrum,
    )
    raw_latest = result.get("latest")
    latest = dict(raw_latest) if isinstance(raw_latest, dict) else None
    source_gate = _model_display_source_gate(
        latest,
        watcher_status,
        sdk_status,
    )

    candidate_peaks = [
        peak
        for peak in ((latest or {}).get("spectrum_peaks") or [])
        if isinstance(peak, dict) and peak.get("candidate_mapping")
    ]
    valid_candidates = [
        peak for peak in candidate_peaks if peak.get("valid") is True
    ]
    dominant_candidate = max(
        valid_candidates,
        key=lambda peak: float(
            peak.get("candidate_absolute_shift_pm") or 0.0
        ),
        default=None,
    )
    candidate_statuses = {
        str(peak.get("candidate_reference_status") or "unknown")
        for peak in valid_candidates
    }
    candidate_contract_complete = bool(
        len(candidate_peaks) == 9
        and len(valid_candidates) == 9
        and [peak.get("candidate_id") for peak in candidate_peaks]
        == [f"FBG{index:02d}" for index in range(1, 10)]
    )
    candidate_baseline_ready = bool(
        len(valid_candidates) == 9
        and candidate_statuses == {"session_global_no_contact_baseline"}
    )
    global_candidate_summary = {
        "valid_candidate_count": len(valid_candidates),
        "expected_candidate_count": 9,
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
        "physical_channel_mapping_final": False,
        "baseline_ready": candidate_baseline_ready,
        "candidate_contract_complete": candidate_contract_complete,
        "diagnostic_only": True,
    }

    if source_gate["model_input_source_allowed"]:
        runtime_prediction = _predict_current_runtime(
            latest_override=latest,
        )
    else:
        runtime_prediction = {
            "ok": False,
            "status": "current_runtime_source_blocked",
            "reason": (
                "wavelength_grid_required_for_formal_recognition"
                if not source_gate["wavelength_axis_valid"]
                else "stale_or_mismatched_live_source"
            ),
            "runtime_role": "deployed_current_model_only",
        }
    runtime_ready = bool(
        runtime_prediction.get("ok")
        and runtime_prediction.get("status") == "ready"
        and source_gate["operator_display_valid"]
    )
    block_reason = (
        None
        if runtime_ready
        else str(
            runtime_prediction.get("reason")
            or runtime_prediction.get("status")
            or "current_runtime_not_ready"
        )
    )
    operator_visualization_frame = _build_operator_visualization_frame(
        latest,
        runtime_prediction,
        ready=runtime_ready,
        block_reason=block_reason,
    )

    try:
        frame_age_sec = (
            max(0.0, time.time() - float(latest.get("ingested_at")))
            if latest is not None
            else None
        )
    except (TypeError, ValueError):
        frame_age_sec = None

    global_frame_qa = {
        "candidate_contract_complete": candidate_contract_complete,
        "candidate_baseline_ready": candidate_baseline_ready,
        "candidate_diagnostics_only": True,
        "source_fresh": bool(source_gate["source_fresh"]),
        "live_source_active": source_gate["live_source_active"],
        "selected_live_source": source_gate["selected_live_source"],
        "model_input_source_allowed": source_gate[
            "model_input_source_allowed"
        ],
        "wavelength_axis_valid": source_gate["wavelength_axis_valid"],
        "wavelength_axis_blockers": source_gate[
            "wavelength_axis_blockers"
        ],
        "formal_spectrum_input_allowed": source_gate[
            "formal_spectrum_input_allowed"
        ],
        "model_input_source_mode": source_gate["model_input_source_mode"],
        "operator_display_valid": source_gate["operator_display_valid"],
        "frame_age_sec": frame_age_sec,
        "display_available": latest is not None,
        "runtime_baseline_ready": runtime_prediction.get("status")
        not in {"baseline_required", "spectrum_required"},
        "formal_recognition_allowed": runtime_ready,
        "response_allowed": runtime_ready,
        "blockers": [] if runtime_ready else [block_reason],
    }

    if latest is not None:
        latest.update(
            {
                "recognition_scope": CURRENT_RECOGNITION_SCOPE,
                "carrier_channel_id": "P22",
                "carrier_channel_role": (
                    "full_spectrum_transport_for_current_runtime"
                ),
                "physical_channel_mapping_final": False,
                "global_candidate_summary": global_candidate_summary,
                "source_fresh": bool(source_gate["source_fresh"]),
                "operator_display_valid": bool(
                    source_gate["operator_display_valid"]
                ),
                "frame_age_sec": frame_age_sec,
                "response_allowed": runtime_ready,
                "response_block_reason": block_reason,
                "runtime_prediction": runtime_prediction,
                "active_spectral_prediction": runtime_prediction,
                "active_spectral_model_source": (
                    "ordinary_fbg_all_data_beta_v1"
                ),
                "active_spectral_model_status": runtime_prediction.get(
                    "status"
                ),
                "operator_visualization_frame": (
                    operator_visualization_frame
                ),
            }
        )
        result["latest"] = latest

    runtime_status = _current_runtime_status()
    result.update(
        {
            "mode": "touch_current_spectral_runtime_frame",
            "scope": CURRENT_RECOGNITION_SCOPE,
            "selected_channel": None,
            "carrier_channel_id": "P22",
            "carrier_channel_role": (
                "full_spectrum_transport_for_current_runtime"
            ),
            "physical_channel_mapping_final": False,
            "global_candidate_summary": global_candidate_summary,
            "global_frame_qa": global_frame_qa,
            "source_fresh": global_frame_qa["source_fresh"],
            "operator_display_valid": global_frame_qa[
                "operator_display_valid"
            ],
            "frame_age_sec": frame_age_sec,
            "formal_recognition_allowed": runtime_ready,
            "model_assisted_display_allowed": runtime_ready,
            "model_assisted_display_block_reason": block_reason,
            "active_spectral_model_expected": True,
            "active_spectral_model_loaded": runtime_status["loaded"],
            "active_spectral_model_source": (
                "ordinary_fbg_all_data_beta_v1"
            ),
            "active_spectral_model_status": runtime_prediction.get(
                "status"
            ),
            "active_spectral_model_progress": {
                key: runtime_prediction.get(key)
                for key in (
                    "history_frames",
                    "history_duration_sec",
                    "temporal_window_sec",
                    "unique_frame_count",
                )
            },
            "active_spectral_prediction": runtime_prediction,
            "runtime_model": runtime_status,
            "runtime_prediction": runtime_prediction,
            "operator_visualization_frame": operator_visualization_frame,
            "response_band_thresholds": (
                _operator_response_band_thresholds()
            ),
            "blockers": global_frame_qa["blockers"],
            "export_watcher": watcher_status,
            "sdk_live": sdk_status,
            "sense_control": sense_controller.status(),
            "px6d_reference": _px6d_reference_for_record(
                result.get("latest")
            ),
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
    whole_hand_asset_url = str(config.get("whole_hand_scene", {}).get("asset_url") or "")
    asset_path = None
    if asset_url.startswith("/static/"):
        asset_path = FRONTEND_ROOT / asset_url.removeprefix("/static/")
    fallback_asset_path = None
    if fallback_asset_url.startswith("/static/"):
        fallback_asset_path = FRONTEND_ROOT / fallback_asset_url.removeprefix("/static/")
    whole_hand_asset_path = None
    if whole_hand_asset_url.startswith("/static/"):
        whole_hand_asset_path = FRONTEND_ROOT / whole_hand_asset_url.removeprefix("/static/")
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
        "whole_hand_asset_url": whole_hand_asset_url,
        "whole_hand_asset_exists": bool(whole_hand_asset_path and whole_hand_asset_path.exists()),
        "whole_hand_asset_path": str(whole_hand_asset_path) if whole_hand_asset_path else None,
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
    capture_status = optical_force_capture.status()
    if capture_status.get("running") or capture_status.get("start_in_progress"):
        return {
            "ok": False,
            "mode": "simulation_blocked",
            "status": "recording_active",
            "reason": "stop_synchronized_recording_before_simulation",
        }
    sdk_status = sdk_live_reader.status()
    watcher_status = export_watcher.status()
    if sdk_status.get("active") or watcher_status.get("active"):
        return {
            "ok": False,
            "mode": "simulation_blocked",
            "status": "live_source_active",
            "reason": "stop_live_acquisition_before_simulation",
        }
    try:
        frame = simulated_array_frame(
            scenario=scenario,
            step=step,
            coupling_view=coupling_view,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {
            "ok": False,
            "mode": "recorded_demo_unavailable",
            "status": "recorded_real_spectrum_asset_required",
            "reason": str(exc),
            "recorded_demo": recorded_demo_library.status(),
        }
    return {
        "ok": True,
        "mode": "recorded_real_spectrum_reference_demo",
        "demo_data_mode": frame.get("demo_data_mode"),
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
        "recorded_demo": recorded_demo_library.status(),
        "message": (
            "recorded real BaySpec spectrum with synchronized Fz-driven visual proxy"
        ),
    }
