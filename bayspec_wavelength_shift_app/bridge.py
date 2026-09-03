"""BaySpec full-spectrum transport for the current TOUCH runtime.

The deployed runtime consumes the complete optical spectrum time series and
produces contact position plus a continuous optical Fz estimate. P22 is only
the transport channel identifier; it is not a separate recognition model.
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - packaged app normally includes PyYAML.
    yaml = None


DEFAULT_SENSE_ROOT = Path(r"D:\APPs\Sense 2020 V1.6.3.3\Sense 2020 V1.6.3.3")
DEFAULT_DEVICE_ID = "F1871328"
APP_ROOT = Path(os.environ.get("BAYSPEC_WAVELENGTH_APP_ROOT", Path(__file__).resolve().parent)).resolve()
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.sense_fast_dat import read_sense_fast_dat

ARRAY_MODE = "global_spectrum_unmapped"
DEFAULT_CHANNEL_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
EPSILON = 1e-12
CURRENT_RECOGNITION_SCOPE = "optical_contact_position_and_continuous_fz"
CURRENT_CARRIER_ROLE = "full_spectrum_transport_for_current_runtime"
CURRENT_GLOBAL_BASELINE_SCOPE = "current_runtime_global_spectrum_baseline"
DEFAULT_CHANNEL_CONFIG = {
    "app": {
        "mode": "standalone_touch_high_sensitivity_300us_spectral_runtime",
        "edition": "current_all_data_spectral_runtime",
        "recognition_scope": CURRENT_RECOGNITION_SCOPE,
        "array_mode": ARRAY_MODE,
        "default_channel": None,
        "carrier_channel": "P22",
        "carrier_channel_role": CURRENT_CARRIER_ROLE,
        "primary_signal": "512_point_bayspec_full_spectrum_time_series",
        "physical_output": "disabled",
        "calibrated_force_output": False,
        "calibrated_pressure_output": False,
        "strain_temperature_decoupled": False,
    },
    "hardware": {
        "sense_root": str(DEFAULT_SENSE_ROOT),
        "sense_export_subdirectory": "Spectrum_Data",
        "device_id": DEFAULT_DEVICE_ID,
        "sense_root_environment_variable": "TOUCH_SENSE_ROOT",
        "device_id_environment_variable": "TOUCH_BAYSPEC_DEVICE_ID",
        "require_wavelength_grid_for_formal_recognition": True,
    },
    "spectrum_normalization": {
        "enabled": True,
        "method": "no_contact_baseline_ratio",
        "minimum_reference_counts": 100.0,
        "minimum_valid_fraction": 0.80,
        "output_field": "normalized_intensity_ratio",
        "model_input_source": "raw_intensity",
    },
    "array_wavelength_plan": {
        "wavelength_unit": "nm",
        "wavelength_start_nm": 1540.0,
        "wavelength_stop_nm": 1580.0,
        "wavelength_spacing_nm": 5.0,
        "number_of_fbg_peaks": 9,
        "status": "preliminary_target_plan",
        "note": "Target wavelengths are preliminary and should be replaced by measured peak wavelengths after fabrication.",
    },
    "channels": {
        channel: {"enabled": False, "role": "future_array_placeholder"} for channel in DEFAULT_CHANNEL_ORDER
    },
    "response_thresholds": {
        "no_contact_max_attenuation": 0.05,
        "light_press_max_attenuation": 0.30,
        "normal_press_max_attenuation": 0.70,
        "intensity_rise_relative_threshold": 1.05,
        "near_saturation_relative_threshold": 0.15,
    },
    "peak_extraction_method": {
        "default": "weighted_centroid_intensity",
        "options": [
            "top_peak_max",
            "top5_mean",
            "top5_mean_peak_marker",
            "window_integrated_intensity",
            "weighted_centroid_intensity",
            "parabolic_peak_fit",
        ],
    },
    "auto_peak_finder": {
        "enabled": True,
        "wavelength_min_nm": 1538.0,
        "wavelength_max_nm": 1582.0,
        "expected_peak_count": 9,
        "min_peak_distance_nm": 3.0,
        "max_peak_distance_nm": 7.0,
        "min_prominence_ratio": 0.03,
        "min_height_ratio": 0.05,
        "smoothing_window_points": 7,
        "max_allowed_offset_nm": 1.5,
        "peak_width_min_nm": 0.02,
        "peak_width_max_nm": 2.0,
        "peak_snr_min": 3.0,
        "min_abs_peak_intensity_counts": 100.0,
        "min_match_peak_snr": 0.75,
        "reject_zero_intensity_peaks": True,
        "reject_dark_region_peaks": True,
        "reject_peak_width_abnormal": "warning_only",
        "baseline_subtraction": True,
        "allow_manual_review": True,
        "discovery_profiles": {
            "single_p22_existing": {
                "description": "Current single-point P22 BaySpec/Sense data, not the future 3x3 target plan.",
                "expected_peak_count": 1,
                "channels": {"P22": {"target_wavelength_nm": 1546.7124}},
                "wavelength_min_nm": 1544.0,
                "wavelength_max_nm": 1549.0,
                "max_allowed_offset_nm": 1.5,
            },
            "future_3x3_plan": {
                "description": "Future preliminary 9-FBG 3x3 target wavelength plan.",
                "expected_peak_count": 9,
                "wavelength_min_nm": 1538.0,
                "wavelength_max_nm": 1582.0,
                "max_allowed_offset_nm": 1.5,
            },
        },
    },
    "single_p22_existing": {
        "profile": "single_p22_existing",
        "description": "Current single-point P22 BaySpec/Sense fallback profile, separate from the future 3x3 wavelength plan.",
        "channels": {
            "P22": {
                "enabled": True,
                "target_wavelength_nm": 1546.7124,
                "measured_wavelength_nm": 1544.339792,
                "measured_wavelength_source": "live_100_frame_nine_peak_cluster_order_candidate",
                "measured_wavelength_status": "provisional_cluster_candidate",
                "clean_no_contact_confirmed": False,
                "search_half_width_nm": 1.0,
                "peak_search_half_width_nm": 1.0,
                "approval_status": "provisional_pending_labelled_p22_press_confirmation",
            }
        },
    },
    "current_real_9fbg_candidate": {
        "profile": "current_real_9fbg_candidate",
        "status": "candidate_pending_labelled_point_press_confirmation",
        "source": "live_direct_sdk_100_frame_no_contact_cluster_20260712",
        "mapping_basis": "ascending_fabrication_order_candidate",
        "expected_peak_count": 9,
        "expected_first_peak_nm": 1528.0,
        "first_peak_tolerance_nm": 1.5,
        "discovery_search_start_nm": 1526.5,
        "discovery_search_stop_nm": 1561.5,
        "search_half_width_nm": 1.0,
        "apply_to_real_array": False,
        "channels": {
            "P11": {"candidate_id": "FBG01", "candidate_measured_wavelength_nm": 1527.813917},
            "P12": {"candidate_id": "FBG02", "candidate_measured_wavelength_nm": 1532.074029},
            "P13": {"candidate_id": "FBG03", "candidate_measured_wavelength_nm": 1536.272630},
            "P21": {"candidate_id": "FBG04", "candidate_measured_wavelength_nm": 1540.087209},
            "P22": {"candidate_id": "FBG05", "candidate_measured_wavelength_nm": 1544.339792},
            "P23": {"candidate_id": "FBG06", "candidate_measured_wavelength_nm": 1547.790240},
            "P31": {"candidate_id": "FBG07", "candidate_measured_wavelength_nm": 1551.672060},
            "P32": {"candidate_id": "FBG08", "candidate_measured_wavelength_nm": 1555.766698},
            "P33": {"candidate_id": "FBG09", "candidate_measured_wavelength_nm": 1559.838208},
        },
    },
    "p22_spectral_feature_mode": {"mode": "auto", "options": ["peak", "dip", "auto"]},
    "baseline": {
        "default_mode": "frozen_baseline",
        "available_modes": ["manual_latest", "frozen_baseline"],
        "rolling_window_sec": 2.0,
        "rolling_min_samples": 20,
        "update_when_attenuation_below": 0.04,
        "noise_warning_ratio": 0.04,
        "noise_fail_ratio": 0.12,
        "trusted_session_anchor": {
            "enabled": True,
            "normalized_shape_rms_warning": 0.015,
            "normalized_shape_rms_fail": 0.03,
            "normalized_shape_peak_warning": 0.08,
            "normalized_shape_peak_fail": 0.18,
            "shape_correlation_warning": 0.999,
            "shape_correlation_fail": 0.995,
        },
    },
    "quality": {
        "dark_count_max": 20,
        "saturation_count_min": 65000,
        "saturation_fraction_warning": 0.02,
        "low_signal_relative_warning": 0.05,
        "peak_edge_margin_points": 3,
    },
}
DEFAULT_CHANNEL_CONFIG["channels"]["P22"] = {
    "enabled": True,
    "role": "single_point_demo",
    "target_wavelength_nm": 1560.0,
    "search_half_width_nm": 1.0,
    "peak_search_half_width_nm": 1.0,
    "peak_top_n_average": 5,
}
for _channel_id, (_x, _y, _target_nm) in {
    "P11": (-1, 1, 1540.0),
    "P12": (-1, 0, 1545.0),
    "P13": (-1, -1, 1550.0),
    "P21": (0, 1, 1555.0),
    "P22": (0, 0, 1560.0),
    "P23": (0, -1, 1565.0),
    "P31": (1, 1, 1570.0),
    "P32": (1, 0, 1575.0),
    "P33": (1, -1, 1580.0),
}.items():
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("display_name", _channel_id)
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("x", _x)
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("y", _y)
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("target_wavelength_nm", _target_nm)
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("search_half_width_nm", 1.0)
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("peak_search_half_width_nm", 1.0)
    DEFAULT_CHANNEL_CONFIG["channels"][_channel_id].setdefault("baseline_method", "rolling_median")


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_candidates() -> list[Path]:
    env_path = os.environ.get("BAYSPEC_WAVELENGTH_CHANNEL_CONFIG")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            APP_ROOT / "config" / "bayspec_wavelength_shift_channels.yaml",
            PROJECT_ROOT / "config" / "bayspec_wavelength_shift_channels.yaml",
            APP_ROOT.parent / "config" / "bayspec_wavelength_shift_channels.yaml",
        ]
    )
    return candidates


def _load_channel_config() -> tuple[dict[str, Any], Path | None, str]:
    for candidate in _config_candidates():
        if not candidate.exists():
            continue
        if yaml is None:
            return DEFAULT_CHANNEL_CONFIG, candidate, "pyyaml_not_installed_default_config_used"
        try:
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            return _merge_dict(DEFAULT_CHANNEL_CONFIG, raw), candidate, "loaded"
        except Exception as exc:
            return DEFAULT_CHANNEL_CONFIG, candidate, f"load_failed_default_config_used: {exc}"

    default_path = PROJECT_ROOT / "config" / "bayspec_wavelength_shift_channels.yaml"
    if yaml is not None:
        try:
            default_path.parent.mkdir(parents=True, exist_ok=True)
            default_path.write_text(yaml.safe_dump(DEFAULT_CHANNEL_CONFIG, sort_keys=False), encoding="utf-8")
            return DEFAULT_CHANNEL_CONFIG, default_path, "created_default"
        except Exception:
            pass
    return DEFAULT_CHANNEL_CONFIG, None, "default_in_memory"


BAYSPEC_CHANNEL_CONFIG, CHANNEL_CONFIG_PATH, CHANNEL_CONFIG_STATUS = _load_channel_config()


def configured_sense_root() -> Path:
    hardware = BAYSPEC_CHANNEL_CONFIG.get("hardware", {}) or {}
    environment_name = str(
        hardware.get("sense_root_environment_variable") or "TOUCH_SENSE_ROOT"
    ).strip()
    configured = (
        os.environ.get(environment_name)
        or hardware.get("sense_root")
        or str(DEFAULT_SENSE_ROOT)
    )
    return Path(str(configured)).expanduser()


def configured_sense_export_root() -> Path:
    hardware = BAYSPEC_CHANNEL_CONFIG.get("hardware", {}) or {}
    subdirectory = str(
        hardware.get("sense_export_subdirectory") or "Spectrum_Data"
    ).strip()
    return configured_sense_root() / subdirectory


def configured_device_id() -> str:
    hardware = BAYSPEC_CHANNEL_CONFIG.get("hardware", {}) or {}
    environment_name = str(
        hardware.get("device_id_environment_variable")
        or "TOUCH_BAYSPEC_DEVICE_ID"
    ).strip()
    return str(
        os.environ.get(environment_name)
        or hardware.get("device_id")
        or DEFAULT_DEVICE_ID
    ).strip()


_configured_channel_ids = list((BAYSPEC_CHANNEL_CONFIG.get("channels", {}) or {}).keys())
CHANNEL_ORDER = [channel for channel in DEFAULT_CHANNEL_ORDER if channel in _configured_channel_ids]
CHANNEL_ORDER.extend(channel for channel in _configured_channel_ids if channel not in CHANNEL_ORDER)
if not CHANNEL_ORDER:
    CHANNEL_ORDER = list(DEFAULT_CHANNEL_ORDER)
DEFAULT_CHANNEL = str(BAYSPEC_CHANNEL_CONFIG.get("app", {}).get("default_channel") or "P22")
INTENSITY_RESPONSE_THRESHOLDS = dict(BAYSPEC_CHANNEL_CONFIG.get("response_thresholds", {}))
WAVELENGTH_SHIFT_CONFIG = dict(BAYSPEC_CHANNEL_CONFIG.get("wavelength_shift_demodulation", {}) or {})
WAVELENGTH_SHIFT_THRESHOLDS_PM = dict(
    WAVELENGTH_SHIFT_CONFIG.get("response_thresholds_pm", {}) or {}
)

from src.array_surface.surface_mapper import (  # noqa: E402
    SurfaceConfig,
    build_array_channels,
    map_surface,
    matrices_from_channels,
)
from src.wavelength_shift.demodulator import (  # noqa: E402
    cross_correlation_shift_pm,
    wavelength_shift_metrics,
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


class SpectrumInputError(ValueError):
    """Expected ingest rejection with a stable machine-readable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _safe_float_list(
    value: Any,
    *,
    max_items: int | None = None,
    reject_invalid: bool = False,
) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        if max_items is not None and value.count(",") + value.count(";") + 1 > max_items:
            raise SpectrumInputError("spectrum_point_limit_exceeded")
        parts: Any = value.replace(";", ",").split(",")
    else:
        try:
            parts = iter(value)
        except TypeError:
            parts = [value]
    out: list[float] = []
    for index, item in enumerate(parts):
        if max_items is not None and index >= max_items:
            raise SpectrumInputError("spectrum_point_limit_exceeded")
        number = _safe_float(item)
        if number is None:
            if reject_invalid:
                raise SpectrumInputError("spectrum_contains_nonfinite_or_nonnumeric_value")
            continue
        out.append(number)
    return out


ACCEPTED_SPECTRUM_NORMALIZATION_BASELINES = frozenset(
    {
        "stable_current_session_startup_baseline",
        "stable_post_release_recovery_baseline",
        "stable_post_release_recovery_baseline_with_warning",
    }
)


def normalize_spectrum_to_baseline_ratio(
    current_wavelength_nm: list[float],
    current_intensity_counts: list[float],
    baseline_wavelength_nm: list[float],
    baseline_intensity_counts: list[float],
    *,
    minimum_reference_counts: float = 100.0,
    minimum_valid_fraction: float = 0.80,
) -> dict[str, Any]:
    """Align a no-contact reference and compute the auditable I/I0 spectrum."""

    current_x = np.asarray(current_wavelength_nm, dtype=float).reshape(-1)
    current_y = np.asarray(current_intensity_counts, dtype=float).reshape(-1)
    baseline_x = np.asarray(baseline_wavelength_nm, dtype=float).reshape(-1)
    baseline_y = np.asarray(baseline_intensity_counts, dtype=float).reshape(-1)
    total_points = int(current_x.size)
    base = {
        "ok": False,
        "method": "no_contact_baseline_ratio",
        "normalized_intensity_ratio": [],
        "normalization_reference_intensity_counts": [],
        "valid_point_count": 0,
        "total_point_count": total_points,
        "valid_fraction": 0.0,
    }
    if (
        total_points < 2
        or current_y.size != current_x.size
        or baseline_x.size < 2
        or baseline_y.size != baseline_x.size
    ):
        return {**base, "status": "invalid_spectrum_shape"}
    if not (
        np.all(np.isfinite(current_x))
        and np.all(np.isfinite(current_y))
        and np.all(np.isfinite(baseline_x))
        and np.all(np.isfinite(baseline_y))
    ):
        return {**base, "status": "nonfinite_spectrum_value"}

    baseline_delta = np.diff(baseline_x)
    if np.all(baseline_delta < 0.0):
        baseline_x = baseline_x[::-1]
        baseline_y = baseline_y[::-1]
    elif not np.all(baseline_delta > 0.0):
        return {**base, "status": "invalid_baseline_wavelength_axis"}

    current_delta = np.diff(current_x)
    if not (np.all(current_delta > 0.0) or np.all(current_delta < 0.0)):
        return {**base, "status": "invalid_current_wavelength_axis"}

    overlap = (current_x >= baseline_x[0]) & (current_x <= baseline_x[-1])
    aligned_reference = np.interp(current_x, baseline_x, baseline_y)
    reference_floor = max(float(minimum_reference_counts), EPSILON)
    valid = (
        overlap
        & np.isfinite(aligned_reference)
        & (np.abs(aligned_reference) >= reference_floor)
    )
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / max(total_points, 1)
    details = {
        **base,
        "valid_point_count": valid_count,
        "valid_fraction": float(valid_fraction),
        "minimum_reference_counts": reference_floor,
        "minimum_valid_fraction": float(minimum_valid_fraction),
    }
    if valid_fraction < max(0.0, min(float(minimum_valid_fraction), 1.0)):
        return {**details, "status": "insufficient_valid_reference_points"}

    ratio = np.ones_like(current_y, dtype=float)
    ratio[valid] = current_y[valid] / aligned_reference[valid]
    if not np.all(np.isfinite(ratio)):
        return {**details, "status": "nonfinite_normalized_value"}
    status = (
        "ready"
        if valid_count == total_points
        else "ready_with_invalid_reference_points"
    )
    return {
        **details,
        "ok": True,
        "status": status,
        "normalized_intensity_ratio": ratio.astype(float).tolist(),
        "normalization_reference_intensity_counts": (
            aligned_reference.astype(float).tolist()
        ),
    }


def _median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _mad_std(values: list[float]) -> float | None:
    center = _median(values)
    if center is None:
        return None
    deviations = [abs(value - center) for value in values if math.isfinite(value)]
    mad = _median(deviations)
    return None if mad is None else 1.4826 * mad


def _channel_config(channel_id: str) -> dict[str, Any]:
    config = dict(BAYSPEC_CHANNEL_CONFIG.get("channels", {}).get(channel_id, {}))
    single_p22 = BAYSPEC_CHANNEL_CONFIG.get("single_p22_existing", {}) or {}
    single_channels = single_p22.get("channels", {}) or {}
    if channel_id == "P22" and single_channels.get("P22"):
        config = _merge_dict(config, dict(single_channels["P22"]))
        config["profile"] = single_p22.get("profile") or "single_p22_existing"
    return config


def _usable_measured_wavelength(channel_config: dict[str, Any]) -> float | None:
    measured = _safe_float(channel_config.get("measured_wavelength_nm"))
    if measured is None:
        return None
    measured_status = str(channel_config.get("measured_wavelength_status") or "").strip()
    if measured_status in {"provisional_manual", "approved"}:
        return measured
    status = str(channel_config.get("match_status") or channel_config.get("measured_wavelength_match_status") or "").strip()
    blocked = {"missing_peak", "extra_peak", "manual_review_required", "invalid", "invalid_for_voltage_drop_rule"}
    if status and status not in {"matched", "matched_with_warning"}:
        if status in blocked or "missing" in status:
            return None
    return measured


def _peak_source_for_config(channel_config: dict[str, Any], measured: float | None, target: float | None) -> str:
    if measured is not None:
        if channel_config.get("measured_wavelength_status") == "provisional_manual":
            return "manual_measured_wavelength_override"
        return "measured_peak_config"
    if target is not None:
        return "target_wavelength_plan"
    return "pixel_index_fallback"


def _array_wavelength_plan_payload() -> dict[str, Any]:
    plan = BAYSPEC_CHANNEL_CONFIG.get("array_wavelength_plan", {}) or {}
    channels = BAYSPEC_CHANNEL_CONFIG.get("channels", {}) or {}
    return {
        "wavelength_unit": plan.get("wavelength_unit") or "nm",
        "wavelength_start_nm": _safe_float(plan.get("wavelength_start_nm"), 1540.0),
        "wavelength_stop_nm": _safe_float(plan.get("wavelength_stop_nm"), 1580.0),
        "wavelength_spacing_nm": _safe_float(plan.get("wavelength_spacing_nm"), 5.0),
        "number_of_fbg_peaks": int(_safe_float(plan.get("number_of_fbg_peaks"), 9) or 9),
        "status": plan.get("status") or "preliminary_target_plan",
        "note": plan.get("note")
        or "Target wavelengths are preliminary and should be replaced by measured peak wavelengths after fabrication.",
        "target_wavelengths_nm": {
            channel_id: _safe_float(config.get("target_wavelength_nm"))
            for channel_id, config in channels.items()
            if _safe_float(config.get("target_wavelength_nm")) is not None
        },
    }


def _config_float(section: str, key: str, default: float) -> float:
    value = _safe_float(BAYSPEC_CHANNEL_CONFIG.get(section, {}).get(key))
    return default if value is None else value


def _config_int(section: str, key: str, default: int) -> int:
    value = _safe_float(BAYSPEC_CHANNEL_CONFIG.get(section, {}).get(key))
    return default if value is None else int(value)


def _threshold(key: str, default: float) -> float:
    value = _safe_float(INTENSITY_RESPONSE_THRESHOLDS.get(key))
    return default if value is None else value


def _pick(payload: dict[str, Any], names: list[str]) -> Any:
    lower = {str(key).lower(): key for key in payload}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return payload.get(key)
    return None


def _now() -> float:
    return time.time()


def _format_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _strip_spectrum(record: dict[str, Any], include_spectrum: bool) -> dict[str, Any]:
    clean = dict(record)
    if not include_spectrum:
        clean.pop("wavelength_nm", None)
        clean.pop("intensity", None)
        clean.pop("display_intensity", None)
        clean.pop("overlay_intensity", None)
        clean.pop("normalized_intensity_ratio", None)
        clean.pop("normalization_reference_intensity_counts", None)
    return clean


def _peak_map_status() -> dict[str, Any]:
    channels = BAYSPEC_CHANNEL_CONFIG.get("channels", {}) or {}
    measured = []
    provisional = []
    review = []
    target_only = []
    for channel_id in channels:
        config = _channel_config(channel_id)
        target = _safe_float(config.get("target_wavelength_nm"))
        measured_nm = _usable_measured_wavelength(config)
        status = str(config.get("match_status") or config.get("measured_wavelength_match_status") or "").strip()
        if measured_nm is not None:
            measured.append(channel_id)
            if config.get("measured_wavelength_status") == "provisional_manual":
                provisional.append(channel_id)
        elif target is not None:
            target_only.append(channel_id)
        if status and status not in {"matched", "matched_with_warning"}:
            review.append(channel_id)
    if provisional:
        state = "manual provisional"
    elif review:
        state = "review_required"
    elif measured:
        state = "measured"
    else:
        state = "target"
    return {
        "available": bool(BAYSPEC_CHANNEL_CONFIG.get("auto_peak_finder", {}).get("enabled", True)),
        "state": state,
        "measured_channels": measured,
        "manual_provisional_channels": provisional,
        "target_only_channels": target_only,
        "review_channels": review,
        "display": f"Peak map: {state}",
    }


class BaySpecWavelengthShiftBridge:
    """In-memory buffer for BaySpec spectra and Bragg wavelength shifts."""

    def __init__(
        self,
        max_records_per_channel: int = 1200,
        max_spectrum_records_per_channel: int = 128,
        max_channel_buffers: int = 32,
        max_channels_per_payload: int = 32,
        max_spectrum_points: int = 16384,
        max_channel_id_length: int = 64,
    ) -> None:
        self.max_records_per_channel = max(1, int(max_records_per_channel))
        self.max_spectrum_records_per_channel = max(
            1,
            min(
                int(max_spectrum_records_per_channel),
                self.max_records_per_channel,
            ),
        )
        self.max_channel_buffers = max(1, int(max_channel_buffers))
        self.max_channels_per_payload = max(
            1,
            min(int(max_channels_per_payload), self.max_channel_buffers),
        )
        self.max_spectrum_points = max(16, int(max_spectrum_points))
        self.max_channel_id_length = max(8, int(max_channel_id_length))
        self.records_by_channel: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_records_per_channel)
        )
        # The mixed trace is a UI convenience, not a second long-term archive.
        # Keeping it at the same bound prevents evicted per-channel records from
        # remaining alive through this deque for many additional hours.
        self.all_records: deque[dict[str, Any]] = deque(
            maxlen=self.max_records_per_channel
        )
        self.baseline_intensity_by_channel: dict[str, float] = {}
        self.baseline_wavelength_by_channel: dict[str, float] = {}
        self.baseline_wavelength_noise_pm_by_channel: dict[str, float] = {}
        self.baseline_wavelength_sample_count_by_channel: dict[str, int] = {}
        self.baseline_spectrum_by_channel: dict[str, dict[str, list[float]]] = {}
        self.baseline_spectrum_sample_count_by_channel: dict[str, int] = {}
        self.baseline_spectrum_noise_ratio_by_channel: dict[str, float] = {}
        self.baseline_spectrum_drift_ratio_by_channel: dict[str, float] = {}
        self.baseline_spectrum_span_sec_by_channel: dict[str, float] = {}
        self.baseline_spectrum_status_by_channel: dict[str, str] = {}
        self.baseline_spectrum_semantic_role_by_channel: dict[str, str] = {}
        self.trusted_baseline_anchor_spectrum_by_channel: dict[
            str, dict[str, list[float]]
        ] = {}
        self.baseline_anchor_comparison_by_channel: dict[str, dict[str, Any]] = {}
        self.previous_tracked_wavelength_by_channel: dict[str, float] = {}
        self.baseline_noise_by_channel: dict[str, float] = {}
        self.baseline_noise_ratio_by_channel: dict[str, float] = {}
        self.baseline_sample_count_by_channel: dict[str, int] = {}
        self.baseline_status_by_channel: dict[str, str] = {}
        self.baseline_method_by_channel: dict[str, str] = defaultdict(
            lambda: str(BAYSPEC_CHANNEL_CONFIG.get("baseline", {}).get("default_mode") or "rolling_median")
        )
        self.baseline_candidates_by_channel: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_records_per_channel)
        )
        self.global_candidate_baseline_by_id: dict[str, float] = {}
        self.global_candidate_baseline_noise_pm_by_id: dict[str, float] = {}
        self.global_candidate_baseline_frame_count = 0
        self.global_candidate_baseline_frame_id: int | None = None
        self.global_candidate_baseline_timestamp: float | None = None
        self.lock = threading.RLock()
        self.started_at = _now()
        self.first_timestamp_by_channel: dict[str, float] = {}
        self.frame_counter = 0
        self.ingest_rejection_counts: dict[str, int] = defaultdict(int)
        self._sense_status_cache: dict[str, Any] = {"running": None, "method": "tasklist_cached"}
        self._sense_status_checked_at = 0.0
        self._status_io_lock = threading.Lock()
        self._latest_export_status_cache: Path | None = None
        self._latest_export_status_checked_at = 0.0

    @staticmethod
    def _evict_heavy_spectral_payload(record: dict[str, Any]) -> None:
        """Keep scalar history while releasing old full-spectrum payloads."""

        for field in ("wavelength_nm", "intensity", "spectrum_peaks"):
            record.pop(field, None)
        record["full_spectrum_retained"] = False
        record["spectrum_payload_evicted"] = True

    def _prune_channel_spectrum_history(
        self,
        records: deque[dict[str, Any]],
    ) -> None:
        if len(records) <= self.max_spectrum_records_per_channel:
            return
        stale_record = records[-self.max_spectrum_records_per_channel - 1]
        self._evict_heavy_spectral_payload(stale_record)

    def reset(self, keep_baseline: bool = True) -> dict[str, Any]:
        with self.lock:
            self.records_by_channel.clear()
            self.all_records.clear()
            self.baseline_candidates_by_channel.clear()
            # A buffer reset starts a new temporal sequence even when the
            # calibrated baseline is intentionally retained.
            self.previous_tracked_wavelength_by_channel.clear()
            if not keep_baseline:
                self.baseline_intensity_by_channel.clear()
                self.baseline_wavelength_by_channel.clear()
                self.baseline_wavelength_noise_pm_by_channel.clear()
                self.baseline_wavelength_sample_count_by_channel.clear()
                self.baseline_spectrum_by_channel.clear()
                self.baseline_spectrum_sample_count_by_channel.clear()
                self.baseline_spectrum_noise_ratio_by_channel.clear()
                self.baseline_spectrum_drift_ratio_by_channel.clear()
                self.baseline_spectrum_span_sec_by_channel.clear()
                self.baseline_spectrum_status_by_channel.clear()
                self.baseline_spectrum_semantic_role_by_channel.clear()
                self.trusted_baseline_anchor_spectrum_by_channel.clear()
                self.baseline_anchor_comparison_by_channel.clear()
                self.baseline_noise_by_channel.clear()
                self.baseline_noise_ratio_by_channel.clear()
                self.baseline_sample_count_by_channel.clear()
                self.baseline_status_by_channel.clear()
                self.baseline_method_by_channel.clear()
                self.global_candidate_baseline_by_id.clear()
                self.global_candidate_baseline_noise_pm_by_id.clear()
                self.global_candidate_baseline_frame_count = 0
                self.global_candidate_baseline_frame_id = None
                self.global_candidate_baseline_timestamp = None
            self.first_timestamp_by_channel.clear()
            return {
                "ok": True,
                "buffered_records": 0,
                "keep_baseline": keep_baseline,
                "baseline_intensity_channels": sorted(self.baseline_intensity_by_channel),
                "baseline_wavelength_channels": sorted(self.baseline_wavelength_by_channel),
                "baseline_spectrum_sample_count_by_channel": dict(
                    sorted(self.baseline_spectrum_sample_count_by_channel.items())
                ),
                "baseline_spectrum_noise_ratio_by_channel": dict(
                    sorted(self.baseline_spectrum_noise_ratio_by_channel.items())
                ),
                "baseline_spectrum_drift_ratio_by_channel": dict(
                    sorted(self.baseline_spectrum_drift_ratio_by_channel.items())
                ),
                "baseline_spectrum_span_sec_by_channel": dict(
                    sorted(self.baseline_spectrum_span_sec_by_channel.items())
                ),
                "baseline_spectrum_status_by_channel": dict(
                    sorted(self.baseline_spectrum_status_by_channel.items())
                ),
                "baseline_anchor_comparison_by_channel": dict(
                    sorted(self.baseline_anchor_comparison_by_channel.items())
                ),
                "global_candidate_baseline_ready": bool(
                    len(self.global_candidate_baseline_by_id) == 9
                ),
            }

    def status(self) -> dict[str, Any]:
        # Filesystem discovery and process inspection can be slow on a large
        # Sense export tree. Keep those operations outside the bridge data lock
        # so health/status requests cannot stall live-frame ingestion.
        latest_file, sense_process = self._status_external_snapshot()
        with self.lock:
            return {
                "ok": True,
                "app": "TOUCH",
                "demodulation_mode": "current_optical_force_runtime",
                "recognition_scope": "optical_contact_position_and_continuous_fz",
                "primary_signal": "512_point_bayspec_full_spectrum_time_series",
                "diagnostic_spectrum_scope": "global_9fbg_wavelength_intensity_area_shape",
                "derived_signals": [
                    "delta_wavelength_pm = (tracked_wavelength_nm - baseline_wavelength_nm) * 1000",
                    "absolute_shift_pm = abs(delta_wavelength_pm)",
                    "shift_response_ratio = absolute_shift_pm / visualization_full_scale_pm",
                ],
                "response_output": (
                    "continuous optical Fz estimate from the deployed same-day joint "
                    "nine-FBG runtime; "
                    "PX6D is training supervision and diagnostics only"
                ),
                "wavelength_shift_config": dict(WAVELENGTH_SHIFT_CONFIG),
                "channel_config_path": _format_path(CHANNEL_CONFIG_PATH),
                "channel_config_status": CHANNEL_CONFIG_STATUS,
                "auto_peak_discovery": {
                    "available": bool(BAYSPEC_CHANNEL_CONFIG.get("auto_peak_finder", {}).get("enabled", True)),
                    "config": dict(BAYSPEC_CHANNEL_CONFIG.get("auto_peak_finder", {})),
                },
                "peak_map_status": _peak_map_status(),
                "single_p22_existing": dict(BAYSPEC_CHANNEL_CONFIG.get("single_p22_existing", {})),
                "p22_spectral_feature_mode": dict(BAYSPEC_CHANNEL_CONFIG.get("p22_spectral_feature_mode", {})),
                "default_channel": None,
                "carrier_channel_id": DEFAULT_CHANNEL,
                "carrier_channel_role": "full_spectrum_transport_for_current_runtime",
                "global_candidate_ids": [
                    f"FBG{index:02d}" for index in range(1, 10)
                ],
                "physical_channel_mapping_final": False,
                "real_3x3_enabled": False,
                "enabled_transport_channels": [
                    channel
                    for channel, config in BAYSPEC_CHANNEL_CONFIG.get("channels", {}).items()
                    if config.get("enabled")
                ],
                "not_pd_voltage": True,
                "calibrated_physical_output": False,
                "array_mode": ARRAY_MODE,
                "device_id_hint": configured_device_id(),
                "sense_root": str(configured_sense_root()),
                "sense_root_exists": configured_sense_root().exists(),
                "sense_export_root": str(configured_sense_export_root()),
                "sense_process": sense_process,
                "latest_export_file": _format_path(latest_file),
                "channels_seen": sorted(self.records_by_channel.keys()),
                "history_policy": {
                    "scalar_records_per_channel": self.max_records_per_channel,
                    "mixed_trace_records": self.all_records.maxlen,
                    "full_spectrum_records_per_channel": (
                        self.max_spectrum_records_per_channel
                    ),
                    "full_spectrum_records_retained_by_channel": {
                        channel: sum(
                            1
                            for record in records
                            if record.get("full_spectrum_retained")
                        )
                        for channel, records in sorted(self.records_by_channel.items())
                    },
                },
                "ingest_limits": {
                    "max_channel_buffers": self.max_channel_buffers,
                    "max_channels_per_payload": self.max_channels_per_payload,
                    "max_spectrum_points": self.max_spectrum_points,
                    "max_channel_id_length": self.max_channel_id_length,
                },
                "ingest_rejection_counts": dict(sorted(self.ingest_rejection_counts.items())),
                "baseline_intensity_channels": sorted(self.baseline_intensity_by_channel),
                "baseline_wavelength_channels": sorted(self.baseline_wavelength_by_channel),
                "baseline_wavelength_noise_pm_by_channel": dict(
                    sorted(self.baseline_wavelength_noise_pm_by_channel.items())
                ),
                "baseline_wavelength_sample_count_by_channel": dict(
                    sorted(self.baseline_wavelength_sample_count_by_channel.items())
                ),
                "baseline_noise_by_channel": dict(sorted(self.baseline_noise_by_channel.items())),
                "baseline_noise_ratio_by_channel": dict(sorted(self.baseline_noise_ratio_by_channel.items())),
                "baseline_sample_count_by_channel": dict(sorted(self.baseline_sample_count_by_channel.items())),
                "baseline_spectrum_sample_count_by_channel": dict(
                    sorted(self.baseline_spectrum_sample_count_by_channel.items())
                ),
                "baseline_spectrum_noise_ratio_by_channel": dict(
                    sorted(self.baseline_spectrum_noise_ratio_by_channel.items())
                ),
                "baseline_spectrum_drift_ratio_by_channel": dict(
                    sorted(self.baseline_spectrum_drift_ratio_by_channel.items())
                ),
                "baseline_spectrum_span_sec_by_channel": dict(
                    sorted(self.baseline_spectrum_span_sec_by_channel.items())
                ),
                "baseline_spectrum_status_by_channel": dict(
                    sorted(self.baseline_spectrum_status_by_channel.items())
                ),
                "baseline_spectrum_semantic_role_by_channel": dict(
                    sorted(self.baseline_spectrum_semantic_role_by_channel.items())
                ),
                "baseline_status_by_channel": dict(sorted(self.baseline_status_by_channel.items())),
                "baseline_method_by_channel": dict(sorted(self.baseline_method_by_channel.items())),
                "global_candidate_baseline": {
                    "ready": len(self.global_candidate_baseline_by_id) == 9,
                    "candidate_wavelength_nm": dict(
                        sorted(self.global_candidate_baseline_by_id.items())
                    ),
                    "candidate_noise_pm": dict(
                        sorted(self.global_candidate_baseline_noise_pm_by_id.items())
                    ),
                    "frame_count": self.global_candidate_baseline_frame_count,
                    "frame_id": self.global_candidate_baseline_frame_id,
                    "timestamp": self.global_candidate_baseline_timestamp,
                    "scope": "display_and_diagnostics_only",
                },
                "buffered_records": len(self.all_records),
                "uptime_sec": round(_now() - self.started_at, 3),
                "direct_sdk_note": (
                    "Direct BaySpec USB20BS SDK frames are handled by the backend SDK helper. "
                    "This bridge tracks Bragg wavelengths from SDK, Sense exports, or HTTP JSON spectra."
                ),
            }

    def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"ok": False, "reason": "payload must be a JSON object"}

        source = str(payload.get("source") or "bayspec_sense2020")
        device_id = str(
            payload.get("device_id")
            or payload.get("device")
            or configured_device_id()
        )
        source_timestamp = _safe_float(payload.get("timestamp"))
        timestamp = _now() if source_timestamp is None else source_timestamp
        channels = payload.get("channels")
        if not isinstance(channels, list):
            channels = [payload]
        input_channel_count = len(channels)
        channels_to_process = channels[: self.max_channels_per_payload]
        rejections: list[dict[str, Any]] = []
        if input_channel_count > self.max_channels_per_payload:
            rejections.append(
                {
                    "reason": "channel_count_limit_exceeded",
                    "rejected_count": input_channel_count - self.max_channels_per_payload,
                }
            )

        records: list[dict[str, Any]] = []
        with self.lock:
            reserved_channel_ids = set(self.records_by_channel)
            accepted_channel_ids: set[str] = set()
            for index, channel_payload in enumerate(channels_to_process):
                if not isinstance(channel_payload, dict):
                    rejections.append(
                        {"channel_index": index, "reason": "channel_payload_must_be_object"}
                    )
                    continue
                channel_id = str(
                    _pick(
                        channel_payload,
                        [
                            "channel_id",
                            "channel",
                            "fbg_channel",
                            "fbng_channel",
                            "sensor_id",
                            "name",
                        ],
                    )
                    or f"CH{index + 1}"
                ).strip()
                if not channel_id:
                    channel_id = f"CH{index + 1}"
                if len(channel_id) > self.max_channel_id_length:
                    rejections.append(
                        {
                            "channel_index": index,
                            "reason": "channel_id_too_long",
                        }
                    )
                    continue
                if channel_id in accepted_channel_ids:
                    rejections.append(
                        {
                            "channel_index": index,
                            "channel_id": channel_id,
                            "reason": "duplicate_channel_id_in_frame",
                        }
                    )
                    continue
                if (
                    channel_id not in reserved_channel_ids
                    and len(reserved_channel_ids) >= self.max_channel_buffers
                ):
                    rejections.append(
                        {
                            "channel_index": index,
                            "channel_id": channel_id,
                            "reason": "channel_buffer_limit_exceeded",
                        }
                    )
                    continue
                try:
                    record = self._normalize_channel(
                        channel_payload,
                        timestamp=timestamp,
                        source=source,
                        device_id=device_id,
                        default_channel=channel_id,
                    )
                except SpectrumInputError as exc:
                    rejections.append(
                        {
                            "channel_index": index,
                            "channel_id": channel_id,
                            "reason": exc.reason,
                        }
                    )
                    continue
                if record is None:
                    rejections.append(
                        {
                            "channel_index": index,
                            "channel_id": channel_id,
                            "reason": "no_valid_spectrum_or_wavelength_data",
                        }
                    )
                    continue
                records.append(record)
                accepted_channel_ids.add(channel_id)
                reserved_channel_ids.add(channel_id)
            for rejection in rejections:
                reason = str(rejection.get("reason") or "unknown_ingest_rejection")
                amount = int(rejection.get("rejected_count") or 1)
                self.ingest_rejection_counts[reason] += amount
            if records:
                self.frame_counter += 1
                source_frame_id = payload.get("frame_id")
                for record in records:
                    record["frame_id"] = self.frame_counter
                    if source_frame_id is not None:
                        record["source_frame_id"] = source_frame_id
                    channel_records = self.records_by_channel[record["channel_id"]]
                    channel_records.append(record)
                    self._prune_channel_spectrum_history(channel_records)
                    self.all_records.append(record)

        return {
            "ok": bool(records),
            "records_ingested": len(records),
            "input_channel_count": input_channel_count,
            "records_rejected": sum(int(item.get("rejected_count") or 1) for item in rejections),
            "rejections": rejections,
            "demodulation_mode": "fbg_wavelength_shift",
            "records": [_strip_spectrum(record, include_spectrum=False) for record in records],
            "reason": None if records else "no valid spectrum or wavelength data found",
        }

    @staticmethod
    def _response_metrics(intensity_value: float | None, baseline_intensity: float | None) -> dict[str, Any]:
        """Secondary intensity QA metrics; not the primary response in this edition."""
        relative_intensity = None
        attenuation_ratio = None
        delta_intensity = None
        intensity_loss_db = None
        response_level = "baseline_required"
        polarity_status = "baseline_required"
        if intensity_value is None or baseline_intensity is None or abs(baseline_intensity) <= EPSILON:
            return {
                "relative_intensity": relative_intensity,
                "attenuation_ratio": attenuation_ratio,
                "delta_intensity_counts": delta_intensity,
                "intensity_loss_db": intensity_loss_db,
                "response_level": response_level,
                "polarity_status": polarity_status,
            }

        relative_intensity = intensity_value / baseline_intensity
        attenuation_ratio = 1.0 - relative_intensity
        delta_intensity = intensity_value - baseline_intensity
        intensity_loss_db = -10.0 * math.log10(max(relative_intensity, EPSILON))
        polarity_status = "expected_attenuation"
        if relative_intensity > _threshold("intensity_rise_relative_threshold", 1.05):
            response_level = "intensity_rise_anomaly"
            polarity_status = "intensity_rise_anomaly"
        elif attenuation_ratio < _threshold("no_contact_max_attenuation", 0.05):
            response_level = "no_contact"
        elif attenuation_ratio < _threshold("light_press_max_attenuation", 0.30):
            response_level = "light_press"
        elif attenuation_ratio < _threshold("normal_press_max_attenuation", 0.70):
            response_level = "normal_press"
        else:
            response_level = "hard_press"
        return {
            "relative_intensity": relative_intensity,
            "attenuation_ratio": attenuation_ratio,
            "delta_intensity_counts": delta_intensity,
            "intensity_loss_db": intensity_loss_db,
            "response_level": response_level,
            "polarity_status": polarity_status,
        }

    def _track_wavelength_response(
        self,
        *,
        channel_id: str,
        wavelength_nm: list[float],
        spectrum_intensity: list[float],
        peak_wavelength_nm: float | None,
        centroid_wavelength_nm: float | None,
        parabolic_peak_wavelength_nm: float | None,
        demodulation_wavelength_nm: float | None,
        update_history: bool,
    ) -> dict[str, Any]:
        baseline_wavelength = self.baseline_wavelength_by_channel.get(channel_id)
        baseline_spectrum = self.baseline_spectrum_by_channel.get(channel_id) or {}
        fallback_method = str(WAVELENGTH_SHIFT_CONFIG.get("fallback_method") or "weighted_centroid")
        fallback_wavelength = (
            centroid_wavelength_nm
            if fallback_method == "weighted_centroid" and centroid_wavelength_nm is not None
            else parabolic_peak_wavelength_nm
            if fallback_method == "parabolic_peak_fit" and parabolic_peak_wavelength_nm is not None
            else peak_wavelength_nm
        )
        tracked_wavelength = fallback_wavelength
        tracking_method = fallback_method
        correlation = None
        correlation_valid = False
        correlation_reason = "baseline_spectrum_required"

        if (
            str(WAVELENGTH_SHIFT_CONFIG.get("primary_method") or "cross_correlation") == "cross_correlation"
            and baseline_wavelength is not None
            and wavelength_nm
            and spectrum_intensity
            and baseline_spectrum.get("wavelength_nm")
            and baseline_spectrum.get("intensity")
        ):
            correlation = cross_correlation_shift_pm(
                wavelength_nm,
                spectrum_intensity,
                baseline_spectrum["wavelength_nm"],
                baseline_spectrum["intensity"],
                center_nm=float(demodulation_wavelength_nm or baseline_wavelength),
                half_width_nm=float(WAVELENGTH_SHIFT_CONFIG.get("search_half_width_nm") or 0.8),
                max_shift_pm=float(WAVELENGTH_SHIFT_CONFIG.get("maximum_correlation_shift_pm") or 500.0),
            )
            correlation_reason = str(correlation.get("reason") or "")
            minimum_correlation = float(WAVELENGTH_SHIFT_CONFIG.get("minimum_cross_correlation") or 0.65)
            coefficient = _safe_float(correlation.get("correlation_coefficient"))
            correlation_valid = bool(
                correlation.get("valid")
                and coefficient is not None
                and coefficient >= minimum_correlation
            )
            if correlation_valid:
                tracked_wavelength = baseline_wavelength + float(correlation["shift_pm"]) / 1000.0
                tracking_method = "baseline_spectrum_cross_correlation"
                correlation_reason = ""

        metrics = wavelength_shift_metrics(
            tracked_wavelength,
            baseline_wavelength,
            thresholds_pm=WAVELENGTH_SHIFT_THRESHOLDS_PM,
            visualization_full_scale_pm=float(
                WAVELENGTH_SHIFT_CONFIG.get("visualization_full_scale_pm") or 500.0
            ),
            baseline_noise_pm=self.baseline_wavelength_noise_pm_by_channel.get(channel_id),
        )
        quality_flags = list(metrics.get("quality_flags") or [])
        if correlation is not None and not correlation_valid:
            quality_flags.append("cross_correlation_low_or_unavailable")

        estimator_disagreement_pm = None
        if centroid_wavelength_nm is not None and parabolic_peak_wavelength_nm is not None:
            estimator_disagreement_pm = abs(centroid_wavelength_nm - parabolic_peak_wavelength_nm) * 1000.0
            if estimator_disagreement_pm > float(
                WAVELENGTH_SHIFT_CONFIG.get("estimator_disagreement_warning_pm") or 80.0
            ):
                quality_flags.append("wavelength_estimator_disagreement")

        previous = self.previous_tracked_wavelength_by_channel.get(channel_id)
        frame_jump_pm = None
        if tracked_wavelength is not None and previous is not None:
            frame_jump_pm = abs(tracked_wavelength - previous) * 1000.0
            if frame_jump_pm > float(WAVELENGTH_SHIFT_CONFIG.get("frame_jump_warning_pm") or 300.0):
                quality_flags.append("wavelength_frame_jump")
        if update_history and tracked_wavelength is not None and metrics.get("valid"):
            self.previous_tracked_wavelength_by_channel[channel_id] = float(tracked_wavelength)

        return {
            **metrics,
            "tracked_wavelength_nm": tracked_wavelength,
            "wavelength_tracking_method": tracking_method,
            "cross_correlation_valid": correlation_valid,
            "cross_correlation_coefficient": _safe_float(
                (correlation or {}).get("correlation_coefficient")
            ),
            "cross_correlation_reason": correlation_reason or None,
            "estimator_disagreement_pm": estimator_disagreement_pm,
            "frame_to_frame_jump_pm": frame_jump_pm,
            "quality_flags": sorted(set(quality_flags)),
        }

    def _normalize_channel(
        self,
        channel_payload: dict[str, Any],
        timestamp: float,
        source: str,
        device_id: str,
        default_channel: str,
    ) -> dict[str, Any] | None:
        channel_id = str(
            _pick(channel_payload, ["channel_id", "channel", "fbg_channel", "fbng_channel", "sensor_id", "name"])
            or default_channel
        ).strip()

        wavelength_input = _pick(
            channel_payload,
            ["wavelength_nm", "wavelength", "wavelengths", "lambda_nm"],
        )
        intensity_input = _pick(
            channel_payload,
            ["intensity", "intensities", "counts", "spectrum_counts", "signal"],
        )
        wavelength_nm = _safe_float_list(
            wavelength_input,
            max_items=self.max_spectrum_points,
            reject_invalid=True,
        )
        spectrum_intensity = _safe_float_list(
            intensity_input,
            max_items=self.max_spectrum_points,
            reject_invalid=True,
        )
        display_intensity = _safe_float_list(
            channel_payload.get("display_intensity"),
            max_items=self.max_spectrum_points,
            reject_invalid=True,
        )
        overlay_intensity = _safe_float_list(
            channel_payload.get("overlay_intensity"),
            max_items=self.max_spectrum_points,
            reject_invalid=True,
        )
        if (
            wavelength_input is not None
            and intensity_input is not None
            and len(wavelength_nm) != len(spectrum_intensity)
        ):
            raise SpectrumInputError("spectrum_axis_length_mismatch")
        if display_intensity and len(display_intensity) != len(spectrum_intensity):
            raise SpectrumInputError("display_spectrum_length_mismatch")
        if overlay_intensity and len(overlay_intensity) != len(spectrum_intensity):
            raise SpectrumInputError("overlay_spectrum_length_mismatch")
        peak_wavelength = _safe_float(
            _pick(channel_payload, ["peak_wavelength_nm", "peak_nm", "lambda_peak_nm", "wavelength_peak"])
        )
        intensity_value = _safe_float(
            _pick(channel_payload, ["intensity_counts", "peak_intensity", "peak_counts", "amplitude", "power"])
        )
        peak_pixel_index = _safe_float(_pick(channel_payload, ["peak_pixel_index", "peak_pixel"]))
        peak_marker_wavelength = peak_wavelength
        centroid_wavelength = _safe_float(_pick(channel_payload, ["centroid_wavelength_nm"]))
        window_integrated_intensity = _safe_float(_pick(channel_payload, ["window_integrated_intensity_counts"]))
        window_mean_intensity = _safe_float(_pick(channel_payload, ["window_mean_intensity_counts"]))
        parabolic_peak_wavelength = _safe_float(_pick(channel_payload, ["parabolic_peak_wavelength_nm"]))
        parabolic_peak_intensity = _safe_float(_pick(channel_payload, ["parabolic_peak_intensity_counts"]))
        selected_peak_extraction_method = _pick(channel_payload, ["peak_extraction_method"])
        intensity_extraction_method = _pick(channel_payload, ["intensity_extraction_method"])
        spectral_feature_mode = _pick(channel_payload, ["spectral_feature_mode"])
        selected_spectral_feature_mode = _pick(channel_payload, ["selected_spectral_feature_mode"])
        spectral_feature_mode_status = _pick(channel_payload, ["spectral_feature_mode_status"])
        channel_config = _channel_config(channel_id)
        target_wavelength = _safe_float(channel_config.get("target_wavelength_nm"))
        measured_wavelength = _usable_measured_wavelength(channel_config)
        demodulation_wavelength = measured_wavelength if measured_wavelength is not None else target_wavelength
        peak_source = _peak_source_for_config(channel_config, measured_wavelength, target_wavelength)
        measured_wavelength_source = channel_config.get("measured_wavelength_source")
        measured_wavelength_status = channel_config.get("measured_wavelength_status")
        clean_no_contact_confirmed = channel_config.get("clean_no_contact_confirmed")
        approval_status = channel_config.get("approval_status")
        manual_wavelength_provisional = measured_wavelength_status == "provisional_manual"
        search_half_width = (
            _safe_float(channel_config.get("search_half_width_nm"))
            or _safe_float(channel_config.get("peak_search_half_width_nm"))
            or 1.0
        )
        has_wavelength_grid = bool(wavelength_nm and spectrum_intensity and len(wavelength_nm) == len(spectrum_intensity))
        peak_axis_type = str(
            _pick(channel_payload, ["peak_axis_type"])
            or ("wavelength_nm" if has_wavelength_grid else "pixel_index")
        )
        spectrum_x_unit = str(
            _pick(channel_payload, ["spectrum_x_unit", "x_unit"])
            or ("wavelength_nm" if has_wavelength_grid else "pixel_index")
        )
        peak_selection_method = str(
            _pick(channel_payload, ["peak_selection_method", "selection_method"])
            or "provided_or_global_peak"
        )

        selected = self._select_channel_peak(channel_id, wavelength_nm, spectrum_intensity)
        if selected is not None:
            peak_wavelength = selected["peak_wavelength_nm"]
            intensity_value = selected["intensity_counts"]
            peak_pixel_index = float(selected["peak_pixel_index"])
            peak_selection_method = selected["peak_selection_method"]
            peak_axis_type = "wavelength_nm"
            spectrum_x_unit = "wavelength_nm"
            peak_marker_wavelength = _safe_float(selected.get("peak_marker_wavelength_nm"))
            centroid_wavelength = _safe_float(selected.get("centroid_wavelength_nm"))
            window_integrated_intensity = _safe_float(selected.get("window_integrated_intensity_counts"))
            window_mean_intensity = _safe_float(selected.get("window_mean_intensity_counts"))
            parabolic_peak_wavelength = _safe_float(selected.get("parabolic_peak_wavelength_nm"))
            parabolic_peak_intensity = _safe_float(selected.get("parabolic_peak_intensity_counts"))
            selected_peak_extraction_method = selected.get("peak_extraction_method")
            intensity_extraction_method = selected.get("intensity_extraction_method") or selected_peak_extraction_method
            spectral_feature_mode = selected.get("spectral_feature_mode")
            selected_spectral_feature_mode = selected.get("selected_spectral_feature_mode")
            spectral_feature_mode_status = selected.get("spectral_feature_mode_status")
        elif peak_wavelength is None and wavelength_nm and spectrum_intensity and len(wavelength_nm) == len(spectrum_intensity):
            peak_index = max(range(len(spectrum_intensity)), key=spectrum_intensity.__getitem__)
            peak_wavelength = wavelength_nm[peak_index]
            peak_marker_wavelength = peak_wavelength
            intensity_value = spectrum_intensity[peak_index]
            peak_pixel_index = float(peak_index)
            peak_selection_method = "global_spectrum_peak"
            peak_axis_type = "wavelength_nm"
            spectrum_x_unit = "wavelength_nm"
        elif intensity_value is None and spectrum_intensity:
            peak_index = max(range(len(spectrum_intensity)), key=spectrum_intensity.__getitem__)
            intensity_value = spectrum_intensity[peak_index]
            peak_pixel_index = float(peak_index)
            peak_selection_method = "global_spectrum_max_value"
            peak_wavelength = None
            peak_axis_type = "pixel_index"
            spectrum_x_unit = "pixel_index"
            peak_source = "pixel_index_fallback"

        if intensity_value is None:
            return None

        peak_offset_reference = peak_marker_wavelength if peak_marker_wavelength is not None else peak_wavelength
        peak_offset_nm = None
        if peak_offset_reference is not None and demodulation_wavelength is not None:
            peak_offset_nm = abs(float(peak_offset_reference) - float(demodulation_wavelength))

        self._update_rolling_baseline_before_response(channel_id, timestamp, intensity_value)
        baseline_intensity = self.baseline_intensity_by_channel.get(channel_id)
        baseline_wavelength = self.baseline_wavelength_by_channel.get(channel_id)
        baseline_noise = self.baseline_noise_by_channel.get(channel_id)
        baseline_noise_ratio = self.baseline_noise_ratio_by_channel.get(channel_id)
        baseline_sample_count = self.baseline_sample_count_by_channel.get(channel_id)
        baseline_status = self.baseline_status_by_channel.get(channel_id, "baseline_required")
        baseline_method = self.baseline_method_by_channel.get(channel_id)
        metrics = self._response_metrics(intensity_value, baseline_intensity)
        if baseline_intensity is not None:
            self._update_rolling_baseline_after_response(
                channel_id,
                timestamp,
                intensity_value,
                str(metrics["response_level"]),
                _safe_float(metrics["attenuation_ratio"]),
            )
            baseline_intensity = self.baseline_intensity_by_channel.get(channel_id)
            baseline_noise = self.baseline_noise_by_channel.get(channel_id)
            baseline_noise_ratio = self.baseline_noise_ratio_by_channel.get(channel_id)
            baseline_sample_count = self.baseline_sample_count_by_channel.get(channel_id)
            baseline_status = self.baseline_status_by_channel.get(channel_id, baseline_status)
            metrics = self._response_metrics(intensity_value, baseline_intensity)

        relative_intensity = metrics["relative_intensity"]
        attenuation_ratio = metrics["attenuation_ratio"]
        delta_intensity = metrics["delta_intensity_counts"]
        intensity_loss_db = metrics["intensity_loss_db"]
        shift = self._track_wavelength_response(
            channel_id=channel_id,
            wavelength_nm=wavelength_nm,
            spectrum_intensity=spectrum_intensity,
            peak_wavelength_nm=peak_wavelength,
            centroid_wavelength_nm=centroid_wavelength,
            parabolic_peak_wavelength_nm=parabolic_peak_wavelength,
            demodulation_wavelength_nm=demodulation_wavelength,
            update_history=True,
        )
        response_level = str(shift["response_level"])
        polarity_status = str(shift["shift_direction"])

        qa_flags, qa_status = self._quality_flags(
            intensity_value=intensity_value,
            spectrum_intensity=spectrum_intensity,
            relative_intensity=relative_intensity,
            intensity_loss_db=intensity_loss_db,
            baseline_intensity=baseline_intensity,
            baseline_noise_ratio=baseline_noise_ratio,
            baseline_sample_count=baseline_sample_count,
            baseline_wavelength=baseline_wavelength,
            baseline_wavelength_noise_pm=self.baseline_wavelength_noise_pm_by_channel.get(channel_id),
            baseline_wavelength_sample_count=self.baseline_wavelength_sample_count_by_channel.get(channel_id),
            response_level=response_level,
            peak_pixel_index=peak_pixel_index,
            peak_axis_type=peak_axis_type,
            peak_selection_method=peak_selection_method,
            peak_offset_nm=peak_offset_nm,
            manual_wavelength_provisional=manual_wavelength_provisional,
        )
        qa_flags = sorted(set(qa_flags + list(shift.get("quality_flags") or [])))
        shift_warning_flags = {
            "baseline_wavelength_required",
            "cross_correlation_low_or_unavailable",
            "wavelength_estimator_disagreement",
            "wavelength_frame_jump",
        }
        if qa_status != "invalid" and any(flag in shift_warning_flags for flag in qa_flags):
            qa_status = "warning"
        if qa_status == "invalid":
            response_level = "uncertain"

        first_timestamp = self.first_timestamp_by_channel.setdefault(channel_id, timestamp)
        record = {
            "timestamp": timestamp,
            "relative_time_sec": timestamp - first_timestamp,
            "ingested_at": _now(),
            "source": source,
            "device_id": device_id,
            "channel_id": channel_id,
            "demodulation_mode": "fbg_wavelength_shift",
            "intensity_counts": intensity_value,
            "baseline_intensity_counts": baseline_intensity,
            "baseline_noise": baseline_noise,
            "baseline_noise_ratio": baseline_noise_ratio,
            "baseline_sample_count": baseline_sample_count,
            "baseline_status": baseline_status,
            "baseline_method": baseline_method,
            "delta_intensity_counts": delta_intensity,
            "relative_intensity": relative_intensity,
            "attenuation_ratio": attenuation_ratio,
            "intensity_loss_db": intensity_loss_db,
            "response_level": response_level,
            "response_basis": "bragg_wavelength_shift_from_no_contact_baseline",
            "response_interpretation": (
                "uncalibrated Bragg wavelength shift; strain and temperature are not decoupled"
            ),
            "attenuation_percent": attenuation_ratio * 100.0 if attenuation_ratio is not None else None,
            "polarity_status": polarity_status,
            "peak_wavelength_nm": peak_wavelength,
            "baseline_wavelength_nm": baseline_wavelength,
            "tracked_wavelength_nm": shift.get("tracked_wavelength_nm"),
            "delta_wavelength_nm": shift.get("delta_wavelength_nm"),
            "delta_wavelength_pm": shift.get("delta_wavelength_pm"),
            "absolute_shift_pm": shift.get("absolute_shift_pm"),
            "shift_direction": shift.get("shift_direction"),
            "wavelength_shift_response_ratio": shift.get("wavelength_shift_response_ratio"),
            "response_value": shift.get("wavelength_shift_response_ratio"),
            "wavelength_tracking_method": shift.get("wavelength_tracking_method"),
            "cross_correlation_valid": shift.get("cross_correlation_valid"),
            "cross_correlation_coefficient": shift.get("cross_correlation_coefficient"),
            "cross_correlation_reason": shift.get("cross_correlation_reason"),
            "wavelength_estimator_disagreement_pm": shift.get("estimator_disagreement_pm"),
            "frame_to_frame_jump_pm": shift.get("frame_to_frame_jump_pm"),
            "baseline_wavelength_noise_pm": self.baseline_wavelength_noise_pm_by_channel.get(channel_id),
            "temperature_strain_decoupled": False,
            "temperature_c": _safe_float(_pick(channel_payload, ["temperature_c", "temperature", "temp_c"])),
            "integration_ms": _safe_float(_pick(channel_payload, ["integration_ms", "integration", "integ_ms"])),
            "peak_pixel_index": peak_pixel_index,
            "peak_marker_wavelength_nm": peak_marker_wavelength,
            "centroid_wavelength_nm": centroid_wavelength,
            "window_integrated_intensity_counts": window_integrated_intensity,
            "window_mean_intensity_counts": window_mean_intensity,
            "parabolic_peak_wavelength_nm": parabolic_peak_wavelength,
            "parabolic_peak_intensity_counts": parabolic_peak_intensity,
            "peak_extraction_method": selected_peak_extraction_method,
            "intensity_extraction_method": intensity_extraction_method or selected_peak_extraction_method,
            "spectral_feature_mode": spectral_feature_mode,
            "selected_spectral_feature_mode": selected_spectral_feature_mode,
            "spectral_feature_mode_status": spectral_feature_mode_status,
            "target_wavelength_nm": target_wavelength,
            "measured_wavelength_nm": measured_wavelength,
            "demodulation_wavelength_nm": demodulation_wavelength,
            "measured_wavelength_source": measured_wavelength_source,
            "measured_wavelength_status": measured_wavelength_status,
            "clean_no_contact_confirmed": clean_no_contact_confirmed,
            "approval_status": approval_status,
            "peak_offset_nm": peak_offset_nm,
            "peak_offset_reference_wavelength_nm": peak_offset_reference,
            "peak_source": peak_source,
            "measured_wavelength_match_status": channel_config.get("match_status")
            or channel_config.get("measured_wavelength_match_status"),
            "peak_search_half_width_nm": search_half_width if demodulation_wavelength is not None else None,
            "peak_selection_method": peak_selection_method,
            "peak_axis_type": peak_axis_type,
            "spectrum_x_unit": spectrum_x_unit,
            "qa_flags": qa_flags,
            "qa_status": qa_status,
            "fast_record_frame_index": _safe_float(_pick(channel_payload, ["fast_record_frame_index", "frame_index"])),
            "fast_record_frame_count": _safe_float(_pick(channel_payload, ["fast_record_frame_count", "frame_count"])),
            "spectrum_points": min(len(wavelength_nm), len(spectrum_intensity))
            if wavelength_nm and spectrum_intensity
            else len(spectrum_intensity),
            "full_spectrum_retained": bool(wavelength_nm and spectrum_intensity),
            "spectrum_payload_evicted": False,
        }
        candidate_peaks = self._extract_candidate_spectrum_peaks(wavelength_nm, spectrum_intensity)
        if candidate_peaks:
            record["spectrum_peaks"] = candidate_peaks
            record["spectrum_peak_profile"] = "current_real_9fbg_candidate"
            record["spectrum_peak_mapping_status"] = "pending_labelled_point_press_confirmation"
            record["spectral_evidence_semantics"] = "mixed_wavelength_intensity_shape"
            record["hybrid_spectral_response_available"] = True
        if wavelength_nm:
            record["wavelength_nm"] = wavelength_nm
        if spectrum_intensity:
            record["intensity"] = spectrum_intensity
        if display_intensity:
            record["display_intensity"] = display_intensity
        if overlay_intensity:
            record["overlay_intensity"] = overlay_intensity
        processing_payload = channel_payload.get("spectrum_processing")
        if isinstance(processing_payload, dict):
            record["model_input_source"] = str(
                channel_payload.get("model_input_source")
                or processing_payload.get("model_input_source")
                or "raw_intensity"
            )
            record["spectrum_processing"] = {
                "steps": [
                    str(value)
                    for value in (processing_payload.get("steps") or [])
                ][:8],
                "warnings": [
                    str(value)
                    for value in (processing_payload.get("warnings") or [])
                ][:8],
                "raw_retained": bool(processing_payload.get("raw_retained")),
                "model_input_source": str(
                    processing_payload.get("model_input_source")
                    or "raw_intensity"
                ),
                "display_input_source": str(
                    processing_payload.get("display_input_source")
                    or "display_intensity"
                ),
                "background_reference_ready": bool(
                    processing_payload.get("background_reference_ready")
                ),
                "raw_roughness": _safe_float(
                    processing_payload.get("raw_roughness")
                ),
                "display_roughness": _safe_float(
                    processing_payload.get("display_roughness")
                ),
                "frame_count": _safe_float(processing_payload.get("frame_count")),
                "normalization_requested": bool(
                    processing_payload.get("normalization_requested")
                ),
            }
        self._refresh_full_spectrum_normalization(record)
        return record

    def _refresh_full_spectrum_normalization(
        self,
        record: dict[str, Any],
    ) -> None:
        config = dict(BAYSPEC_CHANNEL_CONFIG.get("spectrum_normalization", {}) or {})
        enabled = bool(config.get("enabled", True))
        channel_id = str(record.get("channel_id") or "")
        baseline_status = self.baseline_spectrum_status_by_channel.get(channel_id)
        metadata = {
            "enabled": enabled,
            "method": str(
                config.get("method") or "no_contact_baseline_ratio"
            ),
            "status": "disabled" if not enabled else "waiting_for_baseline",
            "output_field": str(
                config.get("output_field") or "normalized_intensity_ratio"
            ),
            "reference_status": baseline_status,
            "reference_semantic_role": self.baseline_spectrum_semantic_role_by_channel.get(
                channel_id,
                "post_press_release_recovery_no_contact",
            ),
            "reference_sample_count": int(
                self.baseline_spectrum_sample_count_by_channel.get(channel_id, 0)
            ),
            "reference_noise_ratio": self.baseline_spectrum_noise_ratio_by_channel.get(
                channel_id
            ),
            "reference_drift_ratio": self.baseline_spectrum_drift_ratio_by_channel.get(
                channel_id
            ),
            "raw_retained": True,
            "model_input_source": "raw_intensity",
            "applied_to_model_input": False,
        }
        record.pop("normalized_intensity_ratio", None)
        record.pop("normalization_reference_intensity_counts", None)
        if not enabled:
            record["spectrum_normalization"] = metadata
            return
        if baseline_status not in ACCEPTED_SPECTRUM_NORMALIZATION_BASELINES:
            record["spectrum_normalization"] = metadata
            return

        baseline = self.baseline_spectrum_by_channel.get(channel_id) or {}
        result = normalize_spectrum_to_baseline_ratio(
            _safe_float_list(record.get("wavelength_nm") or []),
            _safe_float_list(record.get("intensity") or []),
            _safe_float_list(baseline.get("wavelength_nm") or []),
            _safe_float_list(baseline.get("intensity") or []),
            minimum_reference_counts=(
                _safe_float(config.get("minimum_reference_counts"), 100.0)
                or 100.0
            ),
            minimum_valid_fraction=(
                _safe_float(config.get("minimum_valid_fraction"), 0.80)
                or 0.80
            ),
        )
        metadata.update(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "normalized_intensity_ratio",
                    "normalization_reference_intensity_counts",
                }
            }
        )
        if result.get("ok"):
            record["normalized_intensity_ratio"] = list(
                result["normalized_intensity_ratio"]
            )
        record["spectrum_normalization"] = metadata

    def _prune_baseline_candidates(self, channel_id: str, timestamp: float) -> list[float]:
        rolling_window = _config_float("baseline", "rolling_window_sec", 2.0)
        candidates = self.baseline_candidates_by_channel[channel_id]
        while candidates and timestamp - candidates[0][0] > rolling_window:
            candidates.popleft()
        return [value for _, value in candidates]

    def _append_baseline_candidate(self, channel_id: str, timestamp: float, intensity_value: float) -> None:
        candidates = self.baseline_candidates_by_channel[channel_id]
        if candidates and abs(candidates[-1][0] - timestamp) <= EPSILON:
            candidates[-1] = (timestamp, intensity_value)
            return
        candidates.append((timestamp, intensity_value))

    def _set_baseline_from_values(
        self,
        channel_id: str,
        values: list[float],
        method: str,
        keep_existing_when_empty: bool = True,
    ) -> None:
        clean = [value for value in values if math.isfinite(value)]
        if not clean:
            if not keep_existing_when_empty:
                self.baseline_status_by_channel[channel_id] = "baseline_required"
            return
        center = _median(clean)
        noise = _mad_std(clean) or 0.0
        if center is None or abs(center) <= EPSILON:
            self.baseline_status_by_channel[channel_id] = "invalid_baseline"
            return
        ratio = abs(noise / center)
        warning_ratio = _config_float("baseline", "noise_warning_ratio", 0.04)
        fail_ratio = _config_float("baseline", "noise_fail_ratio", 0.12)
        if ratio >= fail_ratio:
            status = "baseline_unstable"
        elif ratio >= warning_ratio:
            status = "baseline_warning"
        else:
            status = "ok"
        self.baseline_intensity_by_channel[channel_id] = float(center)
        self.baseline_noise_by_channel[channel_id] = float(noise)
        self.baseline_noise_ratio_by_channel[channel_id] = float(ratio)
        self.baseline_sample_count_by_channel[channel_id] = len(clean)
        self.baseline_status_by_channel[channel_id] = status
        self.baseline_method_by_channel[channel_id] = method

    def _set_wavelength_baseline_from_records(
        self,
        channel_id: str,
        records: list[dict[str, Any]],
        *,
        replace_trusted_session_anchor: bool = False,
    ) -> bool:
        wavelengths: list[float] = []
        for record in records:
            value = _safe_float(
                record.get("centroid_wavelength_nm")
                or record.get("parabolic_peak_wavelength_nm")
                or record.get("peak_marker_wavelength_nm")
                or record.get("peak_wavelength_nm")
            )
            if value is not None:
                wavelengths.append(value)
        center = _median(wavelengths)
        if center is None:
            return False
        noise_nm = _mad_std(wavelengths) or 0.0

        spectrum_records: list[tuple[np.ndarray, np.ndarray, float | None]] = []
        for record in records:
            x = _safe_float_list(record.get("wavelength_nm") or [])
            y = _safe_float_list(record.get("intensity") or [])
            if x and y and len(x) == len(y):
                x_array = np.asarray(x, dtype=float)
                y_array = np.asarray(y, dtype=float)
                if x_array.size >= 16 and np.all(np.diff(x_array) > 0.0):
                    spectrum_records.append(
                        (x_array, y_array, _safe_float(record.get("timestamp")))
                    )
        if spectrum_records:
            reference_x = spectrum_records[-1][0]
            aligned = np.vstack(
                [
                    np.interp(reference_x, x_values, y_values)
                    for x_values, y_values, _timestamp in spectrum_records
                ]
            )
            median_spectrum = np.median(aligned, axis=0)
            scale = max(float(np.mean(np.abs(median_spectrum))), EPSILON)
            frame_rms = np.sqrt(np.mean((aligned - median_spectrum) ** 2, axis=1))
            noise_ratio = float(np.median(frame_rms) / scale)
            drift_ratio = float(
                np.sqrt(np.mean((aligned[-1] - aligned[0]) ** 2)) / scale
            )
            timestamps = [
                timestamp
                for _x_values, _y_values, timestamp in spectrum_records
                if timestamp is not None
            ]
            span_sec = float(max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
            sample_count = int(aligned.shape[0])
            minimum_samples = _config_int("baseline", "full_spectrum_minimum_samples", 20)
            minimum_span_sec = _config_float("baseline", "full_spectrum_minimum_span_sec", 0.6)
            noise_warning = _config_float(
                "baseline", "full_spectrum_noise_warning_ratio", 0.01
            )
            noise_fail = _config_float(
                "baseline", "full_spectrum_noise_fail_ratio", 0.04
            )
            drift_warning = _config_float(
                "baseline", "full_spectrum_drift_warning_ratio", 0.015
            )
            drift_fail = _config_float(
                "baseline", "full_spectrum_drift_fail_ratio", 0.06
            )
            if sample_count < minimum_samples or span_sec < minimum_span_sec:
                spectrum_status = "insufficient_recovery_baseline_frames"
            elif noise_ratio >= noise_fail or drift_ratio >= drift_fail:
                spectrum_status = "unstable_post_release_recovery_baseline"
            elif noise_ratio >= noise_warning or drift_ratio >= drift_warning:
                spectrum_status = "stable_post_release_recovery_baseline_with_warning"
            else:
                spectrum_status = "stable_post_release_recovery_baseline"

            accepted_internal_statuses = {
                "stable_post_release_recovery_baseline",
                "stable_post_release_recovery_baseline_with_warning",
            }
            had_trusted_anchor = (
                channel_id in self.trusted_baseline_anchor_spectrum_by_channel
            )
            anchor_assessment = self._assess_recovery_baseline_against_anchor(
                channel_id,
                reference_x,
                median_spectrum,
            )
            if (
                replace_trusted_session_anchor
                and had_trusted_anchor
                and spectrum_status in accepted_internal_statuses
            ):
                previous_status = str(anchor_assessment.get("status") or "unknown")
                self.trusted_baseline_anchor_spectrum_by_channel[channel_id] = {
                    "wavelength_nm": reference_x.astype(float).tolist(),
                    "intensity": median_spectrum.astype(float).tolist(),
                }
                anchor_assessment = {
                    **anchor_assessment,
                    "status": "trusted_anchor_replaced_by_operator_attestation",
                    "previous_status": previous_status,
                }
            self.baseline_anchor_comparison_by_channel[channel_id] = anchor_assessment
            if spectrum_status in accepted_internal_statuses and anchor_assessment[
                "status"
            ] == "recovery_residual_detected":
                self.baseline_spectrum_sample_count_by_channel[channel_id] = sample_count
                self.baseline_spectrum_noise_ratio_by_channel[channel_id] = noise_ratio
                self.baseline_spectrum_drift_ratio_by_channel[channel_id] = drift_ratio
                self.baseline_spectrum_span_sec_by_channel[channel_id] = span_sec
                self.baseline_spectrum_status_by_channel[channel_id] = (
                    "recovery_residual_detected"
                )
                self.baseline_spectrum_semantic_role_by_channel[channel_id] = str(
                    BAYSPEC_CHANNEL_CONFIG.get("baseline", {}).get("semantic_role")
                    or "post_press_release_recovery_no_contact"
                )
                return False

            if spectrum_status not in accepted_internal_statuses:
                # Preserve the legacy single-spectrum wavelength-shift path,
                # but keep the static classifier blocked until the multiframe
                # baseline contract is satisfied.
                self.baseline_wavelength_by_channel[channel_id] = float(center)
                self.baseline_wavelength_noise_pm_by_channel[channel_id] = float(
                    noise_nm * 1000.0
                )
                self.baseline_wavelength_sample_count_by_channel[channel_id] = len(
                    wavelengths
                )
                self.baseline_spectrum_by_channel[channel_id] = {
                    "wavelength_nm": reference_x.astype(float).tolist(),
                    "intensity": median_spectrum.astype(float).tolist(),
                }
                self.baseline_spectrum_sample_count_by_channel[channel_id] = sample_count
                self.baseline_spectrum_noise_ratio_by_channel[channel_id] = noise_ratio
                self.baseline_spectrum_drift_ratio_by_channel[channel_id] = drift_ratio
                self.baseline_spectrum_span_sec_by_channel[channel_id] = span_sec
                self.baseline_spectrum_status_by_channel[channel_id] = spectrum_status
                self.baseline_spectrum_semantic_role_by_channel[channel_id] = str(
                    BAYSPEC_CHANNEL_CONFIG.get("baseline", {}).get("semantic_role")
                    or "post_press_release_recovery_no_contact"
                )
                return False

            if anchor_assessment["status"] == "trusted_anchor_initialized":
                self.trusted_baseline_anchor_spectrum_by_channel[channel_id] = {
                    "wavelength_nm": reference_x.astype(float).tolist(),
                    "intensity": median_spectrum.astype(float).tolist(),
                }
            elif anchor_assessment["status"].endswith("with_warning"):
                spectrum_status = "stable_post_release_recovery_baseline_with_warning"

            self.baseline_wavelength_by_channel[channel_id] = float(center)
            self.baseline_wavelength_noise_pm_by_channel[channel_id] = float(
                noise_nm * 1000.0
            )
            self.baseline_wavelength_sample_count_by_channel[channel_id] = len(wavelengths)
            self.baseline_spectrum_by_channel[channel_id] = {
                "wavelength_nm": reference_x.astype(float).tolist(),
                "intensity": median_spectrum.astype(float).tolist(),
            }
            self.baseline_spectrum_sample_count_by_channel[channel_id] = sample_count
            self.baseline_spectrum_noise_ratio_by_channel[channel_id] = noise_ratio
            self.baseline_spectrum_drift_ratio_by_channel[channel_id] = drift_ratio
            self.baseline_spectrum_span_sec_by_channel[channel_id] = span_sec
            self.baseline_spectrum_status_by_channel[channel_id] = spectrum_status
            self.baseline_spectrum_semantic_role_by_channel[channel_id] = str(
                BAYSPEC_CHANNEL_CONFIG.get("baseline", {}).get("semantic_role")
                or "post_press_release_recovery_no_contact"
            )
        else:
            self.baseline_wavelength_by_channel[channel_id] = float(center)
            self.baseline_wavelength_noise_pm_by_channel[channel_id] = float(
                noise_nm * 1000.0
            )
            self.baseline_wavelength_sample_count_by_channel[channel_id] = len(wavelengths)
        return True

    def _assess_recovery_baseline_against_anchor(
        self,
        channel_id: str,
        wavelength_nm: np.ndarray,
        candidate_spectrum: np.ndarray,
    ) -> dict[str, Any]:
        """Detect a stable but locally deformed recovery spectrum.

        The first accepted baseline in an acquisition session is the trusted
        anchor. Later baseline attempts are gain-normalized before comparison,
        so a common optical gain change is tolerated while localized spectral
        residuals remain visible.
        """

        config = BAYSPEC_CHANNEL_CONFIG.get("baseline", {}).get(
            "trusted_session_anchor", {}
        )
        if not bool(config.get("enabled", True)):
            return {"status": "trusted_anchor_check_disabled"}
        anchor_payload = self.trusted_baseline_anchor_spectrum_by_channel.get(channel_id)
        if not anchor_payload:
            return {
                "status": "trusted_anchor_initialized",
                "common_gain_ratio": 1.0,
                "normalized_shape_rms": 0.0,
                "normalized_shape_peak": 0.0,
                "shape_correlation": 1.0,
            }

        anchor_x = np.asarray(anchor_payload.get("wavelength_nm") or [], dtype=float)
        anchor_y = np.asarray(anchor_payload.get("intensity") or [], dtype=float)
        if anchor_x.size < 16 or anchor_x.size != anchor_y.size:
            return {"status": "trusted_anchor_invalid"}
        aligned_anchor = np.interp(wavelength_nm, anchor_x, anchor_y)
        scale = max(float(np.mean(np.abs(aligned_anchor))), EPSILON)
        valid = np.isfinite(aligned_anchor) & np.isfinite(candidate_spectrum)
        valid &= np.abs(aligned_anchor) >= scale * 0.05
        if int(np.sum(valid)) < 16:
            return {"status": "trusted_anchor_invalid"}

        ratios = candidate_spectrum[valid] / np.maximum(
            np.abs(aligned_anchor[valid]), EPSILON
        )
        common_gain = float(np.median(ratios))
        if not math.isfinite(common_gain) or common_gain <= EPSILON:
            return {"status": "trusted_anchor_invalid"}
        corrected = candidate_spectrum / common_gain
        residual = (corrected - aligned_anchor) / scale
        rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
        peak = float(np.max(np.abs(residual[valid])))
        anchor_std = float(np.std(aligned_anchor[valid]))
        corrected_std = float(np.std(corrected[valid]))
        correlation = (
            float(np.corrcoef(aligned_anchor[valid], corrected[valid])[0, 1])
            if anchor_std > EPSILON and corrected_std > EPSILON
            else 1.0 if rms <= EPSILON else 0.0
        )

        warning_rms = float(config.get("normalized_shape_rms_warning", 0.015))
        fail_rms = float(config.get("normalized_shape_rms_fail", 0.03))
        warning_peak = float(config.get("normalized_shape_peak_warning", 0.08))
        fail_peak = float(config.get("normalized_shape_peak_fail", 0.18))
        warning_correlation = float(config.get("shape_correlation_warning", 0.999))
        fail_correlation = float(config.get("shape_correlation_fail", 0.995))
        if rms >= fail_rms or peak >= fail_peak or correlation <= fail_correlation:
            status = "recovery_residual_detected"
        elif (
            rms >= warning_rms
            or peak >= warning_peak
            or correlation <= warning_correlation
        ):
            status = "recovery_consistent_with_trusted_anchor_with_warning"
        else:
            status = "recovery_consistent_with_trusted_anchor"
        return {
            "status": status,
            "common_gain_ratio": common_gain,
            "normalized_shape_rms": rms,
            "normalized_shape_peak": peak,
            "shape_correlation": correlation,
            "thresholds": {
                "normalized_shape_rms_warning": warning_rms,
                "normalized_shape_rms_fail": fail_rms,
                "normalized_shape_peak_warning": warning_peak,
                "normalized_shape_peak_fail": fail_peak,
                "shape_correlation_warning": warning_correlation,
                "shape_correlation_fail": fail_correlation,
            },
        }

    def _update_rolling_baseline_before_response(self, channel_id: str, timestamp: float, intensity_value: float) -> None:
        method = self.baseline_method_by_channel[channel_id]
        if method not in {"rolling_median", "manual_latest"}:
            return
        if channel_id in self.baseline_intensity_by_channel:
            return
        self._append_baseline_candidate(channel_id, timestamp, intensity_value)
        values = self._prune_baseline_candidates(channel_id, timestamp)
        min_samples = _config_int("baseline", "rolling_min_samples", 20)
        if len(values) >= min_samples and method == "rolling_median":
            self._set_baseline_from_values(channel_id, values, method="rolling_median")
        elif channel_id not in self.baseline_intensity_by_channel:
            self.baseline_status_by_channel[channel_id] = f"baseline_collecting_{len(values)}/{min_samples}"

    def _update_rolling_baseline_after_response(
        self,
        channel_id: str,
        timestamp: float,
        intensity_value: float,
        response_level: str,
        attenuation_ratio: float | None,
    ) -> None:
        if self.baseline_method_by_channel[channel_id] != "rolling_median":
            return
        update_limit = _config_float("baseline", "update_when_attenuation_below", 0.04)
        if response_level not in {"no_contact", "baseline_required"}:
            return
        if attenuation_ratio is not None and abs(attenuation_ratio) > update_limit:
            return
        self._append_baseline_candidate(channel_id, timestamp, intensity_value)
        values = self._prune_baseline_candidates(channel_id, timestamp)
        if len(values) >= _config_int("baseline", "rolling_min_samples", 20):
            self._set_baseline_from_values(channel_id, values, method="rolling_median")

    def _quality_flags(
        self,
        *,
        intensity_value: float,
        spectrum_intensity: list[float],
        relative_intensity: float | None,
        intensity_loss_db: float | None,
        baseline_intensity: float | None,
        baseline_noise_ratio: float | None,
        baseline_sample_count: int | None,
        baseline_wavelength: float | None,
        baseline_wavelength_noise_pm: float | None,
        baseline_wavelength_sample_count: int | None,
        response_level: str,
        peak_pixel_index: float | None,
        peak_axis_type: str,
        peak_selection_method: str,
        peak_offset_nm: float | None = None,
        manual_wavelength_provisional: bool = False,
    ) -> tuple[list[str], str]:
        flags: list[str] = []
        quality = BAYSPEC_CHANNEL_CONFIG.get("quality", {})
        dark_max = _safe_float(quality.get("dark_count_max")) or 20.0
        saturation_min = _safe_float(quality.get("saturation_count_min")) or 65000.0
        saturation_fraction_warning = _safe_float(quality.get("saturation_fraction_warning")) or 0.02
        low_signal_relative_warning = _safe_float(quality.get("low_signal_relative_warning")) or 0.05
        edge_margin = int(_safe_float(quality.get("peak_edge_margin_points")) or 3)

        if intensity_value <= dark_max:
            flags.append("dark_or_zero_intensity")
        if manual_wavelength_provisional:
            flags.append("manual_measured_wavelength_provisional")
        if baseline_wavelength is None:
            flags.append("baseline_wavelength_not_ready")
        elif baseline_wavelength_sample_count is not None and baseline_wavelength_sample_count < 1:
            flags.append("baseline_wavelength_sample_count_low")
        if (
            baseline_wavelength_noise_pm is not None
            and baseline_wavelength_noise_pm
            >= float(WAVELENGTH_SHIFT_CONFIG.get("baseline_noise_warning_pm") or 8.0)
        ):
            flags.append("baseline_wavelength_noise_warning")
        if baseline_noise_ratio is not None:
            warning_ratio = _config_float("baseline", "noise_warning_ratio", 0.04)
            fail_ratio = _config_float("baseline", "noise_fail_ratio", 0.12)
            if baseline_noise_ratio >= fail_ratio:
                flags.append("baseline_noise_high_invalid")
            elif baseline_noise_ratio >= warning_ratio:
                flags.append("baseline_noise_warning")
        if relative_intensity is not None:
            if not math.isfinite(relative_intensity) or relative_intensity < 0:
                flags.append("invalid_relative_intensity")
            elif relative_intensity < low_signal_relative_warning:
                flags.append("low_signal_warning")
        if intensity_loss_db is not None and not math.isfinite(intensity_loss_db):
            flags.append("invalid_loss_db")
        if response_level == "intensity_rise_anomaly":
            flags.append("intensity_rise_anomaly")

        if spectrum_intensity:
            saturation_count = sum(1 for value in spectrum_intensity if value >= saturation_min)
            if saturation_count / max(len(spectrum_intensity), 1) >= saturation_fraction_warning:
                flags.append("saturation_warning")
            if peak_pixel_index is not None:
                peak_idx = int(round(float(peak_pixel_index)))
                if peak_idx <= edge_margin or peak_idx >= len(spectrum_intensity) - edge_margin - 1:
                    flags.append("peak_near_search_edge")

        if peak_axis_type == "pixel_index":
            flags.append("wavelength_grid_missing")
            flags.append("using_pixel_index_fallback")
            flags.append("pixel_peak_fallback")
        if "fallback" in peak_selection_method and "wavelength" in peak_selection_method:
            flags.append("peak_window_fallback")
        if "p22_peak_not_found_near_manual_wavelength" in peak_selection_method:
            flags.append("p22_peak_not_found_near_manual_wavelength")
        if "p22_peak_not_found_near_candidate_wavelength" in peak_selection_method:
            flags.append("p22_peak_not_found_near_candidate_wavelength")
        if "peak_or_dip_requires_clean_no_contact_confirmation" in peak_selection_method:
            flags.append("peak_or_dip_requires_clean_no_contact_confirmation")
        if peak_offset_nm is not None and math.isfinite(peak_offset_nm):
            if peak_offset_nm > 0.8:
                flags.append("peak_offset_high")
            elif peak_offset_nm > 0.3:
                flags.append("peak_offset_warning")

        invalid_flags = {
            "dark_or_zero_intensity",
            "baseline_noise_high_invalid",
            "invalid_relative_intensity",
            "invalid_loss_db",
            "p22_peak_not_found_near_manual_wavelength",
            "p22_peak_not_found_near_candidate_wavelength",
        }
        if any(flag in invalid_flags for flag in flags):
            return sorted(set(flags)), "invalid"
        warning_flags = {
            "baseline_wavelength_not_ready",
            "baseline_wavelength_sample_count_low",
            "baseline_wavelength_noise_warning",
            "baseline_noise_warning",
            "low_signal_warning",
            "saturation_warning",
            "intensity_rise_anomaly",
            "peak_near_search_edge",
            "wavelength_grid_missing",
            "using_pixel_index_fallback",
            "pixel_peak_fallback",
            "peak_window_fallback",
            "peak_offset_warning",
            "peak_offset_high",
            "peak_or_dip_requires_clean_no_contact_confirmation",
        }
        if any(flag in warning_flags for flag in flags):
            return sorted(set(flags)), "warning"
        if "manual_measured_wavelength_provisional" in flags:
            return sorted(set(flags)), "ok_with_manual_wavelength"
        return [], "ok"

    def _extract_candidate_spectrum_peaks(
        self,
        wavelength_nm: list[float],
        spectrum_intensity: list[float],
    ) -> list[dict[str, Any]]:
        """Extract all current real nine-FBG candidates from one full spectrum.

        These are spectral evidence markers, not an enabled real 3x3 channel
        map. Channel identities remain provisional until labelled point-press
        captures confirm the wavelength-order fabrication mapping.
        """
        profile = BAYSPEC_CHANNEL_CONFIG.get("current_real_9fbg_candidate", {}) or {}
        channels = profile.get("channels", {}) or {}
        if not wavelength_nm or len(wavelength_nm) != len(spectrum_intensity):
            return []
        half_width = _safe_float(profile.get("search_half_width_nm")) or 1.0
        peak_finder = BAYSPEC_CHANNEL_CONFIG.get("auto_peak_finder", {}) or {}
        min_abs_intensity = float(
            _safe_float(peak_finder.get("min_abs_peak_intensity_counts")) or 100.0
        )
        min_prominence_ratio = float(
            _safe_float(peak_finder.get("min_prominence_ratio")) or 0.03
        )
        min_match_snr = float(
            _safe_float(peak_finder.get("min_match_peak_snr")) or 0.75
        )
        width_min_nm = float(
            _safe_float(peak_finder.get("peak_width_min_nm")) or 0.02
        )
        width_max_nm = float(
            _safe_float(peak_finder.get("peak_width_max_nm")) or 2.0
        )
        edge_margin_points = max(
            1,
            int(
                _safe_float(
                    (BAYSPEC_CHANNEL_CONFIG.get("quality", {}) or {}).get(
                        "peak_edge_margin_points"
                    )
                )
                or 3
            ),
        )
        spectrum_dynamic_range = max(
            EPSILON,
            max(float(value) for value in spectrum_intensity)
            - min(float(value) for value in spectrum_intensity),
        )
        peaks: list[dict[str, Any]] = []
        provisional_channel_ids = [
            "P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"
        ]
        for candidate_index, provisional_channel_id in enumerate(
            provisional_channel_ids, start=1
        ):
            channel = channels.get(provisional_channel_id, {}) or {}
            candidate_id = str(
                channel.get("candidate_id") or f"FBG{candidate_index:02d}"
            )
            candidate = _safe_float(channel.get("candidate_measured_wavelength_nm"))
            if candidate is None:
                continue
            indices = [
                index
                for index, wavelength in enumerate(wavelength_nm)
                if abs(float(wavelength) - candidate) <= half_width
            ]
            if len(indices) < 3:
                continue
            peak_index = max(indices, key=lambda index: spectrum_intensity[index])
            peak_marker = float(wavelength_nm[peak_index])
            peak_height = float(spectrum_intensity[peak_index])
            local_values = [float(spectrum_intensity[index]) for index in indices]
            local_baseline = sorted(local_values)[max(0, int(0.10 * (len(local_values) - 1)))]
            prominence_counts = max(0.0, peak_height - local_baseline)
            prominence_ratio = prominence_counts / spectrum_dynamic_range
            local_noise_counts = _mad_std(
                [value for value in local_values if value <= local_baseline + 0.5 * prominence_counts]
            ) or 0.0
            peak_snr = prominence_counts / max(1.0, float(local_noise_counts))
            local_peak_position = indices.index(peak_index)
            peak_near_edge = (
                local_peak_position < edge_margin_points
                or local_peak_position >= len(indices) - edge_margin_points
            )
            half_height = local_baseline + 0.5 * prominence_counts
            above_half = [
                index for index in indices if float(spectrum_intensity[index]) >= half_height
            ]
            peak_width_nm = (
                float(wavelength_nm[max(above_half)]) - float(wavelength_nm[min(above_half)])
                if len(above_half) >= 2
                else 0.0
            )
            peak_width_abnormal = not (width_min_nm <= peak_width_nm <= width_max_nm)
            quality_flags: list[str] = []
            if peak_height <= min_abs_intensity:
                quality_flags.append("peak_intensity_too_low")
            if prominence_ratio < min_prominence_ratio:
                quality_flags.append("peak_prominence_low")
            if peak_snr < min_match_snr:
                quality_flags.append("peak_snr_low")
            if peak_near_edge:
                quality_flags.append("peak_near_search_edge")
            if peak_width_abnormal:
                quality_flags.append("peak_width_abnormal")
            candidate_valid = not any(
                flag in quality_flags
                for flag in (
                    "peak_intensity_too_low",
                    "peak_prominence_low",
                    "peak_snr_low",
                    "peak_near_search_edge",
                )
            )
            positive = [max(float(spectrum_intensity[index]) - local_baseline, 0.0) for index in indices]
            weight_sum = sum(positive)
            centroid = (
                sum(float(wavelength_nm[index]) * weight for index, weight in zip(indices, positive)) / weight_sum
                if weight_sum > EPSILON
                else peak_marker
            )
            parabolic = peak_marker
            if 0 < peak_index < len(spectrum_intensity) - 1:
                y_left = float(spectrum_intensity[peak_index - 1])
                y_mid = float(spectrum_intensity[peak_index])
                y_right = float(spectrum_intensity[peak_index + 1])
                denominator = y_left - 2.0 * y_mid + y_right
                if abs(denominator) > EPSILON:
                    delta = max(-1.0, min(1.0, 0.5 * (y_left - y_right) / denominator))
                    step_nm = (float(wavelength_nm[peak_index + 1]) - float(wavelength_nm[peak_index - 1])) / 2.0
                    parabolic = peak_marker + delta * step_nm
            baseline_reference = self.global_candidate_baseline_by_id.get(
                candidate_id, candidate
            )
            baseline_is_live = candidate_id in self.global_candidate_baseline_by_id
            candidate_delta_pm = (parabolic - baseline_reference) * 1000.0
            peaks.append(
                {
                    "candidate_id": candidate_id,
                    "provisional_channel_id": provisional_channel_id,
                    "channel_id": provisional_channel_id,
                    "candidate_measured_wavelength_nm": candidate,
                    "tracked_wavelength_nm": parabolic,
                    "candidate_reference_wavelength_nm": baseline_reference,
                    "candidate_delta_wavelength_pm": candidate_delta_pm,
                    "candidate_absolute_shift_pm": abs(candidate_delta_pm),
                    "candidate_reference_status": (
                        "session_global_no_contact_baseline"
                        if baseline_is_live
                        else "provisional_no_contact_reference"
                    ),
                    "candidate_baseline_noise_pm": self.global_candidate_baseline_noise_pm_by_id.get(
                        candidate_id
                    ),
                    "peak_wavelength_nm": parabolic,
                    "peak_marker_wavelength_nm": peak_marker,
                    "centroid_wavelength_nm": centroid,
                    "peak_pixel_index": peak_index,
                    "intensity_counts": peak_height,
                    "local_baseline_counts": local_baseline,
                    "local_integrated_area_counts": sum(positive),
                    "peak_prominence_counts": prominence_counts,
                    "peak_prominence_ratio": prominence_ratio,
                    "peak_snr": peak_snr,
                    "peak_width_nm": peak_width_nm,
                    "peak_near_search_edge": peak_near_edge,
                    "quality_flags": quality_flags,
                    "spectral_feature_mode": "mixed_wavelength_intensity_shape",
                    "mapping_basis": profile.get("mapping_basis"),
                    "approval_status": profile.get("status"),
                    "valid": candidate_valid,
                    "enabled": False,
                    "candidate_mapping": True,
                    "physical_channel_mapping_final": False,
                }
            )
        return peaks

    def _select_channel_peak(
        self,
        channel_id: str,
        wavelength_nm: list[float],
        spectrum_intensity: list[float],
    ) -> dict[str, Any] | None:
        channel_config = _channel_config(channel_id)
        target = _safe_float(channel_config.get("target_wavelength_nm"))
        measured = _usable_measured_wavelength(channel_config)
        demodulation_wavelength = measured if measured is not None else target
        if demodulation_wavelength is None or not wavelength_nm or not spectrum_intensity:
            return None
        if len(wavelength_nm) != len(spectrum_intensity):
            return None
        search_half_width = (
            _safe_float(channel_config.get("search_half_width_nm"))
            or _safe_float(channel_config.get("peak_search_half_width_nm"))
            or 1.0
        )
        top_n = int(_safe_float(channel_config.get("peak_top_n_average")) or 5)
        extraction_cfg = BAYSPEC_CHANNEL_CONFIG.get("peak_extraction_method", {}) or {}
        configured_method = channel_config.get("peak_extraction_method") or extraction_cfg.get("default")
        extraction_method = str(configured_method or "window_integrated_intensity")
        allowed_methods = set(extraction_cfg.get("options") or [])
        if allowed_methods and extraction_method not in allowed_methods:
            extraction_method = "window_integrated_intensity"
        feature_cfg = BAYSPEC_CHANNEL_CONFIG.get("p22_spectral_feature_mode", {}) or {}
        feature_mode = str(channel_config.get("p22_spectral_feature_mode") or feature_cfg.get("mode") or "auto")
        if feature_mode not in set(feature_cfg.get("options") or ["peak", "dip", "auto"]):
            feature_mode = "auto"

        window_indices = [
            index
            for index, wavelength in enumerate(wavelength_nm)
            if abs(wavelength - demodulation_wavelength) <= search_half_width
        ]
        method = "fixed_channel_measured_wavelength_window" if measured is not None else "fixed_channel_target_wavelength_window"
        if not window_indices:
            nearest = min(range(len(wavelength_nm)), key=lambda index: abs(wavelength_nm[index] - demodulation_wavelength))
            if abs(float(wavelength_nm[nearest]) - float(demodulation_wavelength)) > search_half_width:
                window_indices = [nearest]
                if channel_config.get("measured_wavelength_status") == "provisional_manual":
                    method = "p22_peak_not_found_near_manual_wavelength"
                elif measured is not None:
                    method = "p22_peak_not_found_near_candidate_wavelength"
                else:
                    method = "nearest_target_wavelength_outside_grid"
            else:
                start = max(0, nearest - 5)
                stop = min(len(wavelength_nm), nearest + 6)
                window_indices = list(range(start, stop))
                method = "nearest_measured_wavelength_fallback_window" if measured is not None else "nearest_target_wavelength_fallback_window"

        max_index = max(window_indices, key=lambda index: spectrum_intensity[index])
        min_index = min(window_indices, key=lambda index: spectrum_intensity[index])
        window_values = [float(spectrum_intensity[index]) for index in window_indices]
        window_median = _median(window_values) or 0.0
        peak_contrast = max(float(spectrum_intensity[max_index]) - window_median, 0.0)
        dip_contrast = max(window_median - float(spectrum_intensity[min_index]), 0.0)
        feature_mode_status = "configured"
        if feature_mode == "auto":
            selected_feature_mode = "dip" if dip_contrast > peak_contrast else "peak"
            contrast_scale = max(peak_contrast, dip_contrast, EPSILON)
            feature_mode_status = (
                "peak_or_dip_requires_clean_no_contact_confirmation"
                if abs(peak_contrast - dip_contrast) / contrast_scale < 0.20
                else f"auto_selected_{selected_feature_mode}"
            )
        else:
            selected_feature_mode = feature_mode
        peak_index = min_index if selected_feature_mode == "dip" else max_index
        peak_marker_wavelength = wavelength_nm[peak_index]
        top_values = sorted((spectrum_intensity[index] for index in window_indices), reverse=selected_feature_mode != "dip")
        top_count = min(top_n, len(top_values))
        top5_mean_intensity = sum(top_values[:top_count]) / top_count if top_count else spectrum_intensity[peak_index]
        window_integrated_intensity = sum(window_values)
        window_mean_intensity = window_integrated_intensity / len(window_values) if window_values else float(spectrum_intensity[peak_index])
        if selected_feature_mode == "dip":
            positive_weights = [max(window_median - float(spectrum_intensity[index]), 0.0) for index in window_indices]
        else:
            positive_weights = [max(float(spectrum_intensity[index]), 0.0) for index in window_indices]
        weight_sum = sum(positive_weights)
        if weight_sum > EPSILON:
            centroid_wavelength = sum(wavelength_nm[index] * weight for index, weight in zip(window_indices, positive_weights)) / weight_sum
            weighted_centroid_intensity = sum(float(spectrum_intensity[index]) * weight for index, weight in zip(window_indices, positive_weights)) / weight_sum
        else:
            centroid_wavelength = peak_marker_wavelength
            weighted_centroid_intensity = float(spectrum_intensity[peak_index])

        parabolic_peak_wavelength = peak_marker_wavelength
        parabolic_peak_intensity = float(spectrum_intensity[peak_index])
        if 0 < peak_index < len(spectrum_intensity) - 1:
            y_left = float(spectrum_intensity[peak_index - 1])
            y_mid = float(spectrum_intensity[peak_index])
            y_right = float(spectrum_intensity[peak_index + 1])
            denominator = y_left - 2.0 * y_mid + y_right
            if abs(denominator) > EPSILON:
                delta = max(-1.0, min(1.0, 0.5 * (y_left - y_right) / denominator))
                left_wavelength = float(wavelength_nm[peak_index - 1])
                right_wavelength = float(wavelength_nm[peak_index + 1])
                step_nm = (right_wavelength - left_wavelength) / 2.0
                parabolic_peak_wavelength = float(wavelength_nm[peak_index]) + delta * step_nm
                parabolic_peak_intensity = y_mid - 0.25 * (y_left - y_right) * delta

        if extraction_method in {"top_peak_max", "top_peak_marker"}:
            intensity_counts = float(spectrum_intensity[peak_index])
            selected_wavelength = peak_marker_wavelength
            method_suffix = f"{selected_feature_mode}_top_peak_max"
        elif extraction_method in {"top5_mean", "top5_mean_peak_marker"}:
            intensity_counts = top5_mean_intensity
            selected_wavelength = peak_marker_wavelength
            method_suffix = f"{selected_feature_mode}_top{top_count}_mean_peak_marker"
        elif extraction_method == "weighted_centroid_intensity":
            intensity_counts = weighted_centroid_intensity
            selected_wavelength = centroid_wavelength
            method_suffix = f"{selected_feature_mode}_weighted_centroid_intensity"
        elif extraction_method == "parabolic_peak_fit":
            intensity_counts = parabolic_peak_intensity
            selected_wavelength = parabolic_peak_wavelength
            method_suffix = f"{selected_feature_mode}_parabolic_peak_fit"
        else:
            intensity_counts = window_mean_intensity
            selected_wavelength = float(demodulation_wavelength)
            method_suffix = f"{selected_feature_mode}_window_integrated_intensity_mean"
        if feature_mode_status == "peak_or_dip_requires_clean_no_contact_confirmation":
            method_suffix = f"{method_suffix}_peak_or_dip_requires_clean_no_contact_confirmation"
        return {
            "peak_pixel_index": peak_index,
            "peak_wavelength_nm": selected_wavelength,
            "intensity_counts": intensity_counts,
            "peak_selection_method": f"{method}_{method_suffix}",
            "peak_extraction_method": extraction_method,
            "intensity_extraction_method": extraction_method,
            "peak_marker_wavelength_nm": peak_marker_wavelength,
            "centroid_wavelength_nm": centroid_wavelength,
            "window_integrated_intensity_counts": window_integrated_intensity,
            "window_mean_intensity_counts": window_mean_intensity,
            "top5_mean_intensity_counts": top5_mean_intensity,
            "parabolic_peak_wavelength_nm": parabolic_peak_wavelength,
            "parabolic_peak_intensity_counts": parabolic_peak_intensity,
            "target_wavelength_nm": target,
            "measured_wavelength_nm": measured,
            "demodulation_wavelength_nm": demodulation_wavelength,
            "peak_source": _peak_source_for_config(channel_config, measured, target),
            "spectral_feature_mode": feature_mode,
            "selected_spectral_feature_mode": selected_feature_mode,
            "spectral_feature_mode_status": feature_mode_status,
        }

    def _refresh_baseline_metrics(self, record: dict[str, Any]) -> None:
        channel_id = str(record.get("channel_id") or "")
        intensity_value = _safe_float(record.get("intensity_counts"))
        peak_wavelength = _safe_float(record.get("peak_wavelength_nm"))
        centroid_wavelength = _safe_float(record.get("centroid_wavelength_nm"))
        parabolic_peak_wavelength = _safe_float(record.get("parabolic_peak_wavelength_nm"))
        peak_marker_wavelength = _safe_float(record.get("peak_marker_wavelength_nm"))
        demodulation_wavelength = _safe_float(record.get("demodulation_wavelength_nm"))
        peak_pixel_index = _safe_float(record.get("peak_pixel_index"))
        peak_axis_type = str(record.get("peak_axis_type") or "pixel_index")
        peak_selection_method = str(record.get("peak_selection_method") or "")
        manual_wavelength_provisional = record.get("measured_wavelength_status") == "provisional_manual"
        spectrum_intensity = _safe_float_list(record.get("intensity") or [])
        baseline_intensity = self.baseline_intensity_by_channel.get(channel_id)
        baseline_wavelength = self.baseline_wavelength_by_channel.get(channel_id)
        baseline_noise = self.baseline_noise_by_channel.get(channel_id)
        baseline_noise_ratio = self.baseline_noise_ratio_by_channel.get(channel_id)
        baseline_sample_count = self.baseline_sample_count_by_channel.get(channel_id)
        baseline_status = self.baseline_status_by_channel.get(channel_id, "baseline_required")
        baseline_method = self.baseline_method_by_channel.get(channel_id)

        metrics = self._response_metrics(intensity_value, baseline_intensity)
        relative_intensity = metrics["relative_intensity"]
        attenuation_ratio = metrics["attenuation_ratio"]
        delta_intensity = metrics["delta_intensity_counts"]
        intensity_loss_db = metrics["intensity_loss_db"]
        shift = self._track_wavelength_response(
            channel_id=channel_id,
            wavelength_nm=_safe_float_list(record.get("wavelength_nm") or []),
            spectrum_intensity=spectrum_intensity,
            peak_wavelength_nm=peak_wavelength,
            centroid_wavelength_nm=centroid_wavelength,
            parabolic_peak_wavelength_nm=parabolic_peak_wavelength,
            demodulation_wavelength_nm=demodulation_wavelength,
            update_history=False,
        )
        response_level = str(shift["response_level"])
        polarity_status = str(shift["shift_direction"])
        peak_offset_nm = None
        peak_offset_reference = peak_marker_wavelength if peak_marker_wavelength is not None else peak_wavelength
        if peak_offset_reference is not None and demodulation_wavelength is not None:
            peak_offset_nm = abs(float(peak_offset_reference) - float(demodulation_wavelength))

        qa_flags, qa_status = self._quality_flags(
            intensity_value=float(intensity_value) if intensity_value is not None else 0.0,
            spectrum_intensity=spectrum_intensity,
            relative_intensity=relative_intensity,
            intensity_loss_db=intensity_loss_db,
            baseline_intensity=baseline_intensity,
            baseline_noise_ratio=baseline_noise_ratio,
            baseline_sample_count=baseline_sample_count,
            baseline_wavelength=baseline_wavelength,
            baseline_wavelength_noise_pm=self.baseline_wavelength_noise_pm_by_channel.get(channel_id),
            baseline_wavelength_sample_count=self.baseline_wavelength_sample_count_by_channel.get(channel_id),
            response_level=response_level,
            peak_pixel_index=peak_pixel_index,
            peak_axis_type=peak_axis_type,
            peak_selection_method=peak_selection_method,
            peak_offset_nm=peak_offset_nm,
            manual_wavelength_provisional=manual_wavelength_provisional,
        )
        qa_flags = sorted(set(qa_flags + list(shift.get("quality_flags") or [])))
        if qa_status != "invalid" and any(
            flag
            in {
                "baseline_wavelength_required",
                "cross_correlation_low_or_unavailable",
                "wavelength_estimator_disagreement",
                "wavelength_frame_jump",
            }
            for flag in qa_flags
        ):
            qa_status = "warning"
        if qa_status == "invalid":
            response_level = "uncertain"

        record.update(
            {
                "baseline_intensity_counts": baseline_intensity,
                "baseline_noise": baseline_noise,
                "baseline_noise_ratio": baseline_noise_ratio,
                "baseline_sample_count": baseline_sample_count,
                "baseline_status": baseline_status,
                "baseline_method": baseline_method,
                "delta_intensity_counts": delta_intensity,
                "relative_intensity": relative_intensity,
                "attenuation_ratio": attenuation_ratio,
                "intensity_loss_db": intensity_loss_db,
                "observed_intensity_counts": intensity_value,
                "observed_relative_intensity": relative_intensity,
                "observed_attenuation_ratio": attenuation_ratio,
                "observed_loss_db": intensity_loss_db,
                "observed_wavelength_shift_response_ratio": shift.get(
                    "wavelength_shift_response_ratio"
                ),
                "local_response_estimate": shift.get("wavelength_shift_response_ratio"),
                "coupling_compensated": False,
                "coupling_status": "uncalibrated_mechanically_coupled_wavelength_shift",
                "coupling_sources": (BAYSPEC_CHANNEL_CONFIG.get("array_coupling", {}) or {}).get("coupling_sources") or [],
                "coupling_roles": ["observed_shift"]
                if (shift.get("wavelength_shift_response_ratio") or 0.0) >= 0.02
                else [],
                "possible_cross_fiber_coupling": False,
                "possible_same_fiber_coupling": False,
                "local_response_estimate_available": False,
                "response_level": response_level,
                "response_basis": "bragg_wavelength_shift_from_no_contact_baseline",
                "response_interpretation": (
                    "uncalibrated Bragg wavelength shift; strain and temperature are not decoupled"
                ),
                "attenuation_percent": attenuation_ratio * 100.0 if attenuation_ratio is not None else None,
                "polarity_status": polarity_status,
                "baseline_wavelength_nm": baseline_wavelength,
                "tracked_wavelength_nm": shift.get("tracked_wavelength_nm"),
                "delta_wavelength_nm": shift.get("delta_wavelength_nm"),
                "delta_wavelength_pm": shift.get("delta_wavelength_pm"),
                "absolute_shift_pm": shift.get("absolute_shift_pm"),
                "shift_direction": shift.get("shift_direction"),
                "wavelength_shift_response_ratio": shift.get("wavelength_shift_response_ratio"),
                "response_value": shift.get("wavelength_shift_response_ratio"),
                "wavelength_tracking_method": shift.get("wavelength_tracking_method"),
                "cross_correlation_valid": shift.get("cross_correlation_valid"),
                "cross_correlation_coefficient": shift.get("cross_correlation_coefficient"),
                "cross_correlation_reason": shift.get("cross_correlation_reason"),
                "wavelength_estimator_disagreement_pm": shift.get("estimator_disagreement_pm"),
                "frame_to_frame_jump_pm": shift.get("frame_to_frame_jump_pm"),
                "baseline_wavelength_noise_pm": self.baseline_wavelength_noise_pm_by_channel.get(channel_id),
                "temperature_strain_decoupled": False,
                "peak_offset_nm": peak_offset_nm,
                "peak_offset_reference_wavelength_nm": peak_offset_reference,
                "qa_flags": qa_flags,
                "qa_status": qa_status,
            }
        )
        self._refresh_full_spectrum_normalization(record)

    def _refresh_channel_records(self, channels: set[str]) -> None:
        for channel in channels:
            for record in self.records_by_channel.get(channel, []):
                self._refresh_baseline_metrics(record)

    def _surface_config(self) -> SurfaceConfig:
        section = BAYSPEC_CHANNEL_CONFIG.get("surface", {})
        return SurfaceConfig(
            grid_size=int(_safe_float(section.get("surface_grid_size"), 80) or 80),
            sigma=float(_safe_float(section.get("surface_sigma"), 0.65) or 0.65),
            clip_min=float(_safe_float(section.get("surface_clip_min"), 0.0) or 0.0),
            clip_max=float(_safe_float(section.get("surface_clip_max"), 1.0) or 1.0),
            active_threshold=float(_safe_float(section.get("active_threshold"), 0.05) or 0.05),
            active_absolute_threshold=float(_safe_float(section.get("active_absolute_threshold"), 0.10) or 0.10),
            active_relative_threshold=float(_safe_float(section.get("active_relative_threshold"), 0.25) or 0.25),
            surface_input_mode=str(section.get("surface_input_mode") or "raw_coupled_response_surface"),
        )

    def _recent_records_for_baseline(
        self,
        channel_id: str,
        minimum_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        records = list(self.records_by_channel.get(channel_id, []))
        if not records:
            return []
        configured_minimum = max(
            _config_int("baseline", "rolling_min_samples", 20),
            _config_int("baseline", "full_spectrum_minimum_samples", 20),
        )
        required_samples = max(
            configured_minimum,
            int(minimum_samples) if minimum_samples is not None else configured_minimum,
        )
        latest_timestamp = _safe_float(records[-1].get("timestamp"))
        if latest_timestamp is None:
            return records[-required_samples:]
        rolling_window = _config_float("baseline", "rolling_window_sec", 2.0)
        recent = []
        for record in reversed(records):
            record_timestamp = _safe_float(record.get("timestamp"))
            if record_timestamp is None:
                continue
            age = latest_timestamp - record_timestamp
            if age < -EPSILON:
                continue
            if age > rolling_window:
                break
            recent.append(record)
        recent.reverse()
        # Direct SDK throughput can be slower than the nominal integration rate.
        # Keep the time-window preference, but never freeze an undersized model
        # baseline when enough current-session records already exist.
        if len(recent) < required_samples:
            return records[-required_samples:]
        return recent

    def build_array_frame(
        self,
        latest_by_channel: dict[str, dict[str, Any]],
        timestamp: float | None = None,
        mode: str = "single_point_fallback",
        include_surface_grid: bool = True,
    ) -> dict[str, Any]:
        reference_record = max(
            latest_by_channel.values(),
            key=lambda record: _safe_float(record.get("timestamp"), float("-inf")),
            default={},
        )
        frame_id = reference_record.get("frame_id") if reference_record else None
        if frame_id is None:
            frame_id = self.frame_counter
        frame_timestamp = timestamp
        if frame_timestamp is None:
            frame_timestamp = _safe_float(reference_record.get("timestamp")) if reference_record else None
        if frame_timestamp is None:
            frame_timestamp = _now()
        latest_frame_ids = {
            record.get("frame_id")
            for record in latest_by_channel.values()
            if record.get("frame_id") is not None
        }
        frame_sync_status = "synced" if len(latest_frame_ids) <= 1 else "latest_snapshot_mixed_frames"
        channels = build_array_channels(
            latest_by_channel=latest_by_channel,
            channel_configs=BAYSPEC_CHANNEL_CONFIG.get("channels", {}),
            channel_order=CHANNEL_ORDER,
        )
        surface = map_surface(channels, config=self._surface_config())
        matrices = matrices_from_channels(channels)
        coupling_cfg = BAYSPEC_CHANNEL_CONFIG.get("array_coupling", {}) or {}
        coupling_sources = coupling_cfg.get("coupling_sources") or [
            "shared_elastomer_mechanical_coupling",
            "fingertip_contact_area_coverage",
        ]
        surface_metrics = surface.get("surface_metrics") or {}
        observed_changed = surface_metrics.get("observed_changed_channels") or []
        primary_observed = surface_metrics.get("primary_observed_channel")
        secondary_observed = surface_metrics.get("secondary_observed_channels") or surface_metrics.get("secondary_changed_channels") or []
        active_count = int(surface.get("active_channel_count") or 0)
        valid_record_count = sum(1 for ch in channels if ch.get("valid"))
        p22_has_valid_record = any(ch.get("channel_id") == "P22" and ch.get("valid") for ch in channels)
        if p22_has_valid_record and valid_record_count == 1:
            frame_mode = "p22_fallback"
        elif p22_has_valid_record and active_count <= 0:
            frame_mode = "p22_fallback"
        elif valid_record_count <= 0:
            frame_mode = "no_valid_channel"
        else:
            frame_mode = mode
        surface_grid = surface.get("surface_grid") if include_surface_grid else None
        return {
            "timestamp": frame_timestamp,
            "frame_id": frame_id,
            "surface_frame_id": frame_id,
            "trace_frame_id": frame_id,
            "spectrum_frame_id": frame_id,
            "frame_sync_status": frame_sync_status,
            "last_update_timestamp": frame_timestamp,
            "mode": frame_mode,
            "surface_display_mode": "surface",
            "coupling_view": "raw_coupled_response",
            "coupling_status": str(
                coupling_cfg.get("coupling_status")
                or "uncalibrated_mechanically_coupled_wavelength_shift"
            ),
            "coupling_compensated": False,
            "coupling_sources": coupling_sources,
            "coupling_model_note": (
                "Wavelength-shift mode shows raw Bragg displacement from contact-area coverage "
                "and shared-elastomer mechanical coupling. It does not reuse the intensity "
                "edition's same-fiber attenuation cascade."
            ),
            "observed_changed_channels": observed_changed,
            "primary_observed_channel": primary_observed,
            "secondary_observed_channels": secondary_observed,
            "possible_cross_fiber_coupling": bool(surface_metrics.get("possible_cross_fiber_coupling")),
            "possible_same_fiber_coupling": bool(surface_metrics.get("possible_same_fiber_coupling")),
            "local_response_estimate_available": bool(surface_metrics.get("local_response_estimate_available")),
            "array_mode": ARRAY_MODE,
            "wavelength_plan": _array_wavelength_plan_payload(),
            "channels": channels,
            "array_response_3x3": matrices["array_response_3x3"],
            "array_quality_3x3": matrices["array_quality_3x3"],
            "valid_channel_mask_3x3": matrices["valid_channel_mask_3x3"],
            "surface_grid": surface_grid,
            "grid_x": surface.get("grid_x"),
            "grid_y": surface.get("grid_y"),
            "surface_metrics": surface_metrics,
            "surface_note": surface.get("surface_note"),
            "surface_mode": surface.get("surface_mode"),
            "surface_title": "Raw coupled Bragg wavelength-shift surface",
            "surface_subtitle": (
                "Uncalibrated wavelength displacement. Strain and temperature are not decoupled."
            ),
            "fallback_note": "Single-point P22 wavelength-shift fallback. 3x3 array not yet enabled."
            if frame_mode == "p22_fallback"
            else None,
        }

    def set_global_candidate_baseline(
        self,
        minimum_frames: int | None = None,
    ) -> dict[str, Any]:
        """Freeze a display baseline from complete recent nine-candidate frames.

        This baseline supports the operator trace and candidate fingerprint. It
        does not replace the stricter full-spectrum baseline used by the hybrid
        model runtime.
        """

        baseline_config = BAYSPEC_CHANNEL_CONFIG.get(
            "global_candidate_baseline", {}
        ) or {}
        required_frames = max(
            3,
            int(
                minimum_frames
                or baseline_config.get("minimum_unique_frames")
                or 30
            ),
        )
        expected_ids = [f"FBG{index:02d}" for index in range(1, 10)]
        with self.lock:
            records = list(self.records_by_channel.get("P22", []))
            recent = records[-required_frames:]
            if len(recent) < required_frames:
                return {
                    "ok": False,
                    "reason": "insufficient_unique_global_candidate_frames",
                    "required_frames": required_frames,
                    "available_frames": len(recent),
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                }

            frame_ids = [record.get("frame_id") for record in recent]
            if (
                any(frame_id is None for frame_id in frame_ids)
                or len(set(frame_ids)) != len(frame_ids)
                or frame_ids != sorted(frame_ids)
            ):
                return {
                    "ok": False,
                    "reason": "global_baseline_frames_not_unique_monotonic",
                    "required_frames": required_frames,
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                }

            integration_values = {
                float(value)
                for record in recent
                if (value := _safe_float(record.get("integration_ms"))) is not None
            }
            device_ids = {
                str(record.get("device_id"))
                for record in recent
                if record.get("device_id") not in {None, ""}
            }
            if len(integration_values) > 1 or len(device_ids) > 1:
                return {
                    "ok": False,
                    "reason": "global_baseline_acquisition_context_changed",
                    "integration_values_ms": sorted(integration_values),
                    "device_ids": sorted(device_ids),
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                }

            ingest_times = [
                float(value)
                for record in recent
                if (value := _safe_float(record.get("ingested_at"))) is not None
            ]
            latest_ingest_age_sec = (
                max(0.0, _now() - max(ingest_times)) if ingest_times else float("inf")
            )
            maximum_latest_ingest_age_sec = float(
                _safe_float(baseline_config.get("maximum_latest_ingest_age_sec"))
                or 2.5
            )
            minimum_ingest_span_sec = float(
                _safe_float(baseline_config.get("minimum_ingest_span_sec"))
                or 0.5
            )
            ingest_span_sec = (
                max(ingest_times) - min(ingest_times) if len(ingest_times) >= 2 else 0.0
            )
            source_values = {
                str(record.get("source") or "").strip()
                for record in recent
                if str(record.get("source") or "").strip()
            }
            if latest_ingest_age_sec > maximum_latest_ingest_age_sec:
                return {
                    "ok": False,
                    "reason": "global_baseline_source_frames_stale",
                    "latest_ingest_age_sec": latest_ingest_age_sec,
                    "maximum_latest_ingest_age_sec": maximum_latest_ingest_age_sec,
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                }
            if ingest_span_sec < minimum_ingest_span_sec:
                return {
                    "ok": False,
                    "reason": "global_baseline_capture_span_too_short",
                    "ingest_span_sec": ingest_span_sec,
                    "minimum_ingest_span_sec": minimum_ingest_span_sec,
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                }
            if bool(baseline_config.get("require_single_source", True)) and len(source_values) != 1:
                return {
                    "ok": False,
                    "reason": "global_baseline_acquisition_source_changed",
                    "sources": sorted(source_values),
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                }

            values_by_id: dict[str, list[float]] = {
                candidate_id: [] for candidate_id in expected_ids
            }
            for record in recent:
                peaks = [
                    peak
                    for peak in (record.get("spectrum_peaks") or [])
                    if isinstance(peak, dict)
                ]
                observed_ids = [str(peak.get("candidate_id") or "") for peak in peaks]
                if observed_ids != expected_ids or any(
                    peak.get("valid") is not True for peak in peaks
                ):
                    return {
                        "ok": False,
                        "reason": "incomplete_or_invalid_global_candidate_frame",
                        "frame_id": record.get("frame_id"),
                        "observed_candidate_ids": observed_ids,
                        "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                    }
                for peak in peaks:
                    tracked = _safe_float(peak.get("tracked_wavelength_nm"))
                    if tracked is None:
                        return {
                            "ok": False,
                            "reason": "nonfinite_global_candidate_wavelength",
                            "frame_id": record.get("frame_id"),
                            "candidate_id": peak.get("candidate_id"),
                            "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                        }
                    values_by_id[str(peak["candidate_id"])].append(tracked)

            candidate_baseline = {
                candidate_id: float(_median(values) or 0.0)
                for candidate_id, values in values_by_id.items()
            }
            candidate_noise_pm = {
                candidate_id: float((_mad_std(values) or 0.0) * 1000.0)
                for candidate_id, values in values_by_id.items()
            }
            candidate_range_pm = {
                candidate_id: float((max(values) - min(values)) * 1000.0)
                for candidate_id, values in values_by_id.items()
            }
            candidate_max_jump_pm = {
                candidate_id: float(
                    max(
                        (
                            abs(current - previous) * 1000.0
                            for previous, current in zip(values, values[1:])
                        ),
                        default=0.0,
                    )
                )
                for candidate_id, values in values_by_id.items()
            }
            noise_limit_pm = float(
                _safe_float(
                    baseline_config.get("maximum_candidate_noise_pm_warning")
                )
                or 8.0
            )
            range_limit_pm = float(
                _safe_float(baseline_config.get("maximum_candidate_range_pm"))
                or 30.0
            )
            jump_limit_pm = float(
                _safe_float(
                    baseline_config.get("maximum_candidate_frame_jump_pm")
                )
                or 15.0
            )
            unstable_candidates = {
                candidate_id: {
                    "noise_pm": candidate_noise_pm[candidate_id],
                    "range_pm": candidate_range_pm[candidate_id],
                    "max_jump_pm": candidate_max_jump_pm[candidate_id],
                }
                for candidate_id in expected_ids
                if candidate_noise_pm[candidate_id] > noise_limit_pm
                or candidate_range_pm[candidate_id] > range_limit_pm
                or candidate_max_jump_pm[candidate_id] > jump_limit_pm
            }
            if bool(baseline_config.get("reject_unstable_baseline", True)) and unstable_candidates:
                return {
                    "ok": False,
                    "reason": "global_candidate_baseline_unstable",
                    "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                    "required_frames": required_frames,
                    "limits": {
                        "noise_pm": noise_limit_pm,
                        "range_pm": range_limit_pm,
                        "max_jump_pm": jump_limit_pm,
                    },
                    "unstable_candidates": unstable_candidates,
                }
            self.global_candidate_baseline_by_id = candidate_baseline
            self.global_candidate_baseline_noise_pm_by_id = candidate_noise_pm
            self.global_candidate_baseline_frame_count = required_frames
            self.global_candidate_baseline_frame_id = int(frame_ids[-1])
            self.global_candidate_baseline_timestamp = float(
                _safe_float(recent[-1].get("timestamp")) or _now()
            )

            for record in records:
                wavelength_nm = record.get("wavelength_nm")
                intensity = record.get("intensity")
                if (
                    isinstance(wavelength_nm, list)
                    and isinstance(intensity, list)
                    and len(wavelength_nm) == len(intensity)
                ):
                    record["spectrum_peaks"] = self._extract_candidate_spectrum_peaks(
                        wavelength_nm,
                        intensity,
                    )

            return {
                "ok": True,
                "baseline_set": True,
                "scope": CURRENT_GLOBAL_BASELINE_SCOPE,
                "baseline_role": "display_and_diagnostics_only",
                "candidate_ids": expected_ids,
                "candidate_wavelength_nm": dict(sorted(candidate_baseline.items())),
                "candidate_noise_pm": dict(sorted(candidate_noise_pm.items())),
                "candidate_range_pm": dict(sorted(candidate_range_pm.items())),
                "candidate_max_jump_pm": dict(
                    sorted(candidate_max_jump_pm.items())
                ),
                "frame_count": required_frames,
                "frame_id": self.global_candidate_baseline_frame_id,
                "timestamp": self.global_candidate_baseline_timestamp,
            }

    def set_baseline(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        spectrum_attempted_channels: set[str] = set()
        replace_trusted_session_anchor = (
            payload.get("replace_trusted_session_anchor") is True
        )
        minimum_recent_samples_raw = payload.get("minimum_recent_samples")
        try:
            minimum_recent_samples = (
                max(1, int(minimum_recent_samples_raw))
                if minimum_recent_samples_raw is not None
                else None
            )
        except (TypeError, ValueError):
            minimum_recent_samples = None
        with self.lock:
            requested_method = str(payload.get("baseline_method") or payload.get("method") or "frozen_baseline")
            intensity_map = payload.get("baseline_intensity_by_channel") or payload.get("baselines_intensity_counts")
            if isinstance(intensity_map, dict):
                for channel, value in intensity_map.items():
                    number = _safe_float(value)
                    if number is not None:
                        channel_key = str(channel)
                        self._set_baseline_from_values(channel_key, [number], method=requested_method)

            wavelength_map = payload.get("baseline_wavelength_by_channel") or payload.get("baselines_wavelength_nm")
            if isinstance(wavelength_map, dict):
                for channel, value in wavelength_map.items():
                    number = _safe_float(value)
                    if number is not None:
                        channel_key = str(channel)
                        self.baseline_wavelength_by_channel[channel_key] = number
                        self.baseline_wavelength_noise_pm_by_channel[channel_key] = 0.0
                        self.baseline_wavelength_sample_count_by_channel[channel_key] = 1

            channel_id = payload.get("channel_id") or payload.get("channel")
            if channel_id:
                channel = str(channel_id)
                intensity = _safe_float(payload.get("baseline_intensity_counts"))
                wavelength = _safe_float(payload.get("baseline_wavelength_nm"))
                recent_records = self._recent_records_for_baseline(
                    channel,
                    minimum_samples=minimum_recent_samples,
                )
                wavelength_baseline_accepted = True
                if recent_records:
                    spectrum_attempted_channels.add(channel)
                    wavelength_baseline_accepted = self._set_wavelength_baseline_from_records(
                        channel,
                        recent_records,
                        replace_trusted_session_anchor=replace_trusted_session_anchor,
                    )
                if intensity is not None and wavelength_baseline_accepted:
                    self._set_baseline_from_values(channel, [float(intensity)], method=requested_method)
                elif recent_records and wavelength_baseline_accepted:
                    values = [
                        value
                        for record in recent_records
                        if (value := _safe_float(record.get("intensity_counts"))) is not None
                    ]
                    self._set_baseline_from_values(channel, values, method=requested_method)
                if wavelength is not None:
                    if wavelength_baseline_accepted:
                        self.baseline_wavelength_by_channel[channel] = float(wavelength)
                        self.baseline_wavelength_noise_pm_by_channel[channel] = 0.0
                        self.baseline_wavelength_sample_count_by_channel[channel] = 1
                self._refresh_channel_records({channel})
            elif not intensity_map and not wavelength_map:
                changed_channels: set[str] = set()
                for channel, records in self.records_by_channel.items():
                    if not records:
                        continue
                    recent_records = self._recent_records_for_baseline(
                        channel,
                        minimum_samples=minimum_recent_samples,
                    )
                    spectrum_attempted_channels.add(channel)
                    wavelength_baseline_accepted = self._set_wavelength_baseline_from_records(
                        channel,
                        recent_records,
                        replace_trusted_session_anchor=replace_trusted_session_anchor,
                    )
                    values = [
                        value
                        for record in recent_records
                        if (value := _safe_float(record.get("intensity_counts"))) is not None
                    ]
                    if values and wavelength_baseline_accepted:
                        self._set_baseline_from_values(channel, values, method=requested_method)
                        changed_channels.add(channel)
                    if wavelength_baseline_accepted:
                        changed_channels.add(channel)
                self._refresh_channel_records(changed_channels)
            elif intensity_map or wavelength_map:
                changed_channels = set(self.baseline_intensity_by_channel) | set(self.baseline_wavelength_by_channel)
                self._refresh_channel_records(changed_channels)

        if channel_id:
            requested_channel = str(channel_id)
            baseline_set = requested_channel in self.baseline_wavelength_by_channel
            missing_reason = (
                None
                if baseline_set
                else f"no baseline wavelength provided and no latest spectrum for {requested_channel}"
            )
        else:
            baseline_set = bool(self.baseline_wavelength_by_channel)
            missing_reason = (
                None
                if baseline_set
                else "no baseline wavelength provided and no latest spectrum available"
            )

        model_baseline_channel = str(channel_id or DEFAULT_CHANNEL)
        model_baseline_status = self.baseline_spectrum_status_by_channel.get(
            model_baseline_channel
        )
        current_runtime_spectrum_baseline_ready = (
            model_baseline_status in ACCEPTED_SPECTRUM_NORMALIZATION_BASELINES
        )
        return {
            "ok": bool(baseline_set),
            "baseline_set": bool(baseline_set),
            "reason": missing_reason,
            "baseline_source": "provided_or_latest_bragg_wavelength",
            "baseline_semantics": "post_press_release_recovery_no_contact",
            "current_runtime_spectrum_baseline_ready": current_runtime_spectrum_baseline_ready,
            "current_runtime_spectrum_baseline_channel": model_baseline_channel,
            "current_runtime_spectrum_baseline_status": model_baseline_status,
            "current_runtime_spectrum_baseline_rejected": bool(
                model_baseline_channel in spectrum_attempted_channels
                and not current_runtime_spectrum_baseline_ready
            ),
            "baseline_intensity_by_channel": dict(sorted(self.baseline_intensity_by_channel.items())),
            "baseline_wavelength_by_channel": dict(sorted(self.baseline_wavelength_by_channel.items())),
            "baseline_wavelength_noise_pm_by_channel": dict(
                sorted(self.baseline_wavelength_noise_pm_by_channel.items())
            ),
            "baseline_wavelength_sample_count_by_channel": dict(
                sorted(self.baseline_wavelength_sample_count_by_channel.items())
            ),
            "baseline_noise_by_channel": dict(sorted(self.baseline_noise_by_channel.items())),
            "baseline_noise_ratio_by_channel": dict(sorted(self.baseline_noise_ratio_by_channel.items())),
            "baseline_sample_count_by_channel": dict(sorted(self.baseline_sample_count_by_channel.items())),
            "baseline_spectrum_sample_count_by_channel": dict(
                sorted(self.baseline_spectrum_sample_count_by_channel.items())
            ),
            "baseline_spectrum_noise_ratio_by_channel": dict(
                sorted(self.baseline_spectrum_noise_ratio_by_channel.items())
            ),
            "baseline_spectrum_drift_ratio_by_channel": dict(
                sorted(self.baseline_spectrum_drift_ratio_by_channel.items())
            ),
            "baseline_spectrum_span_sec_by_channel": dict(
                sorted(self.baseline_spectrum_span_sec_by_channel.items())
            ),
            "baseline_spectrum_status_by_channel": dict(
                sorted(self.baseline_spectrum_status_by_channel.items())
            ),
            "baseline_spectrum_semantic_role_by_channel": dict(
                sorted(self.baseline_spectrum_semantic_role_by_channel.items())
            ),
            "trusted_session_anchor_replaced": any(
                str(item.get("status") or "")
                == "trusted_anchor_replaced_by_operator_attestation"
                for item in self.baseline_anchor_comparison_by_channel.values()
            ),
            "baseline_anchor_comparison_by_channel": dict(
                sorted(self.baseline_anchor_comparison_by_channel.items())
            ),
            "baseline_status_by_channel": dict(sorted(self.baseline_status_by_channel.items())),
            "baseline_method_by_channel": dict(sorted(self.baseline_method_by_channel.items())),
            "demodulation_mode": "fbg_wavelength_shift",
        }

    def set_runtime_recovery_spectrum_baseline(
        self,
        channel_id: str,
        wavelength_nm: list[float] | np.ndarray,
        intensity: list[float] | np.ndarray,
        *,
        sample_count: int,
        span_sec: float,
        shape_motion_rms: float | None = None,
        common_gain_motion: float | None = None,
        policy: str = "multi_evidence_release_then_spectral_stationarity",
    ) -> dict[str, Any]:
        """Install a full-spectrum runtime baseline confirmed by the release guard.

        This path deliberately does not replace the trusted session anchor. It
        updates the operational model reference after a contact/release cycle,
        while preserving the initial anchor for diagnostics and drift review.
        """

        channel = str(channel_id)
        wavelength = np.asarray(wavelength_nm, dtype=float)
        spectrum = np.asarray(intensity, dtype=float)
        if (
            wavelength.ndim != 1
            or spectrum.shape != wavelength.shape
            or wavelength.size < 16
            or not np.all(np.isfinite(wavelength))
            or not np.all(np.isfinite(spectrum))
            or not np.all(np.diff(wavelength) > 0.0)
            or float(np.max(np.abs(spectrum))) <= EPSILON
        ):
            return {
                "ok": False,
                "status": "runtime_recovery_baseline_invalid",
                "channel_id": channel,
            }

        with self.lock:
            self.baseline_spectrum_by_channel[channel] = {
                "wavelength_nm": wavelength.astype(float).tolist(),
                "intensity": spectrum.astype(float).tolist(),
            }
            self.baseline_spectrum_sample_count_by_channel[channel] = max(
                1, int(sample_count)
            )
            self.baseline_spectrum_noise_ratio_by_channel[channel] = float(
                max(0.0, shape_motion_rms or 0.0)
            )
            self.baseline_spectrum_drift_ratio_by_channel[channel] = float(
                max(0.0, common_gain_motion or 0.0)
            )
            self.baseline_spectrum_span_sec_by_channel[channel] = float(
                max(0.0, span_sec)
            )
            self.baseline_spectrum_status_by_channel[channel] = (
                "stable_post_release_recovery_baseline"
            )
            self.baseline_spectrum_semantic_role_by_channel[channel] = (
                "automatic_multi_evidence_post_release_stationary_recovery"
            )

            records = self.records_by_channel.get(channel)
            latest = records[-1] if records else None
            if latest is not None:
                baseline_wavelength = _safe_float(
                    latest.get("centroid_wavelength_nm")
                    or latest.get("parabolic_peak_wavelength_nm")
                    or latest.get("peak_marker_wavelength_nm")
                    or latest.get("peak_wavelength_nm")
                )
                if baseline_wavelength is not None:
                    self.baseline_wavelength_by_channel[channel] = baseline_wavelength
                    self.baseline_wavelength_noise_pm_by_channel[channel] = 0.0
                    self.baseline_wavelength_sample_count_by_channel[channel] = max(
                        1, int(sample_count)
                    )
                baseline_intensity = _safe_float(latest.get("intensity_counts"))
                if baseline_intensity is not None:
                    self._set_baseline_from_values(
                        channel,
                        [baseline_intensity],
                        method="automatic_stationary_release_recovery",
                    )
                self._refresh_channel_records({channel})

        return {
            "ok": True,
            "status": "runtime_recovery_baseline_set",
            "channel_id": channel,
            "sample_count": max(1, int(sample_count)),
            "span_sec": float(max(0.0, span_sec)),
            "shape_motion_rms": shape_motion_rms,
            "common_gain_motion": common_gain_motion,
            "policy": policy,
            "trusted_session_anchor_preserved": True,
            "baseline_semantics": (
                "automatic_multi_evidence_post_release_stationary_recovery"
            ),
        }

    def set_runtime_startup_spectrum_baseline(
        self,
        channel_id: str,
        wavelength_nm: list[float] | np.ndarray,
        intensity: list[float] | np.ndarray,
        *,
        sample_count: int,
        span_sec: float,
        shape_motion_rms: float | None = None,
        common_gain_motion: float | None = None,
        policy: str = "current_session_stable_five_frame_median",
    ) -> dict[str, Any]:
        """Install the first trusted no-contact reference of a live session."""

        result = self.set_runtime_recovery_spectrum_baseline(
            channel_id,
            wavelength_nm,
            intensity,
            sample_count=sample_count,
            span_sec=span_sec,
            shape_motion_rms=shape_motion_rms,
            common_gain_motion=common_gain_motion,
            policy=policy,
        )
        if not result.get("ok"):
            return {
                **result,
                "status": "runtime_startup_baseline_invalid",
            }

        channel = str(channel_id)
        wavelength = np.asarray(wavelength_nm, dtype=float)
        spectrum = np.asarray(intensity, dtype=float)
        with self.lock:
            self.baseline_spectrum_status_by_channel[channel] = (
                "stable_current_session_startup_baseline"
            )
            self.baseline_spectrum_semantic_role_by_channel[channel] = (
                "automatic_current_session_startup_no_contact"
            )
            self.trusted_baseline_anchor_spectrum_by_channel[channel] = {
                "wavelength_nm": wavelength.astype(float).tolist(),
                "intensity": spectrum.astype(float).tolist(),
            }
            self.baseline_anchor_comparison_by_channel[channel] = {
                "status": "trusted_anchor_initialized",
                "common_gain_ratio": 1.0,
                "normalized_shape_rms": 0.0,
                "normalized_shape_peak": 0.0,
                "shape_correlation": 1.0,
            }

        return {
            **result,
            "status": "runtime_startup_baseline_set",
            "policy": policy,
            "trusted_session_anchor_preserved": False,
            "trusted_session_anchor_initialized": True,
            "baseline_semantics": "automatic_current_session_startup_no_contact",
        }

    def latest(self, channel_id: str | None = None, include_spectrum: bool = False) -> Any:
        with self.lock:
            if channel_id:
                records = self.records_by_channel.get(str(channel_id))
                if not records:
                    return None
                return _strip_spectrum(records[-1], include_spectrum)
            return {
                channel: _strip_spectrum(records[-1], include_spectrum)
                for channel, records in sorted(self.records_by_channel.items())
                if records
            }

    def spectral_model_input(self, channel_id: str = "P22") -> dict[str, Any]:
        """Return a synchronized current/baseline full-spectrum pair for ML inference."""

        with self.lock:
            records = self.records_by_channel.get(str(channel_id))
            latest = dict(records[-1]) if records else None
            baseline_payload = self.baseline_spectrum_by_channel.get(str(channel_id))
            baseline = dict(baseline_payload) if baseline_payload else None
            if baseline is not None:
                baseline["wavelength_nm"] = list(baseline.get("wavelength_nm") or [])
                baseline["intensity"] = list(baseline.get("intensity") or [])
            if latest is not None:
                latest["wavelength_nm"] = list(latest.get("wavelength_nm") or [])
                latest["intensity"] = list(latest.get("intensity") or [])
            current_ready = bool(
                latest
                and latest.get("wavelength_nm")
                and latest.get("intensity")
                and len(latest["wavelength_nm"]) == len(latest["intensity"])
            )
            spectrum_status = self.baseline_spectrum_status_by_channel.get(str(channel_id))
            baseline_payload_available = bool(
                baseline
                and baseline.get("wavelength_nm")
                and baseline.get("intensity")
                and len(baseline["wavelength_nm"]) == len(baseline["intensity"])
            )
            baseline_ready = bool(
                baseline_payload_available
                and spectrum_status in ACCEPTED_SPECTRUM_NORMALIZATION_BASELINES
            )
            return {
                "ok": current_ready and baseline_ready,
                "channel_id": str(channel_id),
                "current_ready": current_ready,
                "baseline_ready": baseline_ready,
                "latest": latest,
                "baseline": baseline,
                "baseline_spectrum_sample_count": int(
                    self.baseline_spectrum_sample_count_by_channel.get(str(channel_id), 0)
                ),
                "baseline_spectrum_noise_ratio": self.baseline_spectrum_noise_ratio_by_channel.get(
                    str(channel_id)
                ),
                "baseline_spectrum_drift_ratio": self.baseline_spectrum_drift_ratio_by_channel.get(
                    str(channel_id)
                ),
                "baseline_spectrum_span_sec": self.baseline_spectrum_span_sec_by_channel.get(
                    str(channel_id)
                ),
                "baseline_spectrum_status": spectrum_status,
                "baseline_spectrum_semantic_role": self.baseline_spectrum_semantic_role_by_channel.get(
                    str(channel_id), "post_press_release_recovery_no_contact"
                ),
                "reason": None
                if current_ready and baseline_ready
                else "current_full_spectrum_required"
                if not current_ready
                else spectrum_status or "current_no_contact_full_spectrum_baseline_required",
            }

    def trace(self, channel_id: str | None = None, limit: int = 500, include_spectrum: bool = False) -> dict[str, Any]:
        limit = max(1, min(int(limit), 20000))
        with self.lock:
            if channel_id:
                records = list(self.records_by_channel.get(str(channel_id), []))[-limit:]
            else:
                records = list(self.all_records)[-limit:]
            return {
                "ok": True,
                "channel_id": channel_id,
                "count": len(records),
                "records": [_strip_spectrum(record, include_spectrum) for record in records],
            }

    def frame(self, channel_id: str = "P22", trace_limit: int = 600, include_spectrum: bool = True) -> dict[str, Any]:
        with self.lock:
            selected_latest = self.latest(channel_id=channel_id, include_spectrum=include_spectrum)
            if selected_latest is None:
                latest_map = self.latest(include_spectrum=False)
                if isinstance(latest_map, dict) and latest_map:
                    channel_id = sorted(latest_map)[0]
                    selected_latest = self.latest(channel_id=channel_id, include_spectrum=include_spectrum)
            grid = []
            latest_map = self.latest(include_spectrum=False)
            latest_by_channel = latest_map if isinstance(latest_map, dict) else {}
            array_frame = self.build_array_frame(
                latest_by_channel=latest_by_channel,
                timestamp=(selected_latest or {}).get("timestamp") if isinstance(selected_latest, dict) else None,
                include_surface_grid=True,
            )
            by_channel = {channel["channel_id"]: channel for channel in array_frame["channels"]}
            for channel in CHANNEL_ORDER:
                record = by_channel.get(channel) or {}
                grid.append(
                    {
                        "channel_id": channel,
                        "intensity_counts": record.get("intensity_counts"),
                        "relative_intensity": record.get("relative_intensity"),
                        "attenuation_ratio": record.get("attenuation_ratio"),
                        "tracked_wavelength_nm": record.get("tracked_wavelength_nm"),
                        "baseline_wavelength_nm": record.get("baseline_wavelength_nm"),
                        "delta_wavelength_pm": record.get("delta_wavelength_pm"),
                        "absolute_shift_pm": record.get("absolute_shift_pm"),
                        "shift_direction": record.get("shift_direction"),
                        "wavelength_shift_response_ratio": record.get(
                            "wavelength_shift_response_ratio"
                        ),
                        "response_level": record.get("response_level", "no_data"),
                        "enabled": record.get("enabled", False),
                        "valid": record.get("valid", False),
                        "qa_status": record.get("qa_status", "no_data"),
                        "x": record.get("x"),
                        "y": record.get("y"),
                    }
                )
            trace = self.trace(channel_id=channel_id, limit=trace_limit, include_spectrum=False)

        frame_id = array_frame.get("frame_id")
        timestamp = array_frame.get("timestamp")
        if isinstance(selected_latest, dict):
            selected_latest = dict(selected_latest)
            spectrum_counts = selected_latest.get("intensity") or selected_latest.get("spectrum_counts")
            wavelength_grid = selected_latest.get("wavelength_nm") or selected_latest.get("spectrum_wavelength_nm")
            has_spectrum = isinstance(spectrum_counts, list) and len(spectrum_counts) > 0
            selected_latest.setdefault("frame_id", frame_id)
            selected_latest.setdefault("timestamp", timestamp)
            selected_latest["surface_frame_id"] = frame_id
            selected_latest["trace_frame_id"] = frame_id
            if has_spectrum:
                selected_latest["spectrum_frame_id"] = frame_id
                selected_latest["frame_sync_status"] = "synced"
                if not wavelength_grid:
                    selected_latest["spectrum_unavailable_reason"] = "Pixel-axis spectrum. Wavelength calibration missing."
            else:
                selected_latest["frame_sync_status"] = "spectrum_missing"
                selected_latest["spectrum_unavailable_reason"] = "Spectrum unavailable for this frame"

        return {
            "ok": True,
            "mode": "bayspec_fbg_wavelength_shift",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "selected_channel": channel_id,
            "latest": selected_latest,
            "trace": trace["records"],
            "channel_grid": grid,
            "array_frame": array_frame,
            "surface_grid": array_frame.get("surface_grid"),
            "surface_metrics": array_frame.get("surface_metrics"),
            "surface_note": array_frame.get("surface_note"),
            "status": self.status(),
        }

    def ingest_csv_text(
        self,
        text: str,
        channel_id: str = "P22",
        device_id: str | None = None,
        source: str = "bayspec_sense2020_csv",
    ) -> dict[str, Any]:
        rows = list(csv.reader(io.StringIO(text)))
        payloads = self._csv_rows_to_payloads(
            rows,
            channel_id=channel_id,
            device_id=str(device_id or configured_device_id()),
            source=source,
        )
        if not payloads:
            return {
                "ok": False,
                "reason": "CSV has no usable spectrum or wavelength columns",
                "records_ingested": 0,
            }
        total = 0
        last: dict[str, Any] | None = None
        for payload in payloads:
            result = self.ingest(payload)
            total += int(result.get("records_ingested") or 0)
            last = result
        return {
            "ok": total > 0,
            "records_ingested": total,
            "last_result": last,
            "demodulation_mode": "fbg_wavelength_shift",
        }

    def _csv_rows_to_payloads(
        self,
        rows: list[list[str]],
        channel_id: str,
        device_id: str,
        source: str,
    ) -> list[dict[str, Any]]:
        rows = [row for row in rows if row and any(cell.strip() for cell in row)]
        if not rows:
            return []

        sense_payload = self._sense_export_rows_to_payload(rows, channel_id=channel_id, device_id=device_id, source=source)
        if sense_payload is not None:
            return [sense_payload]

        header = [cell.strip().lower() for cell in rows[0]]
        data_rows = rows[1:] if any(any(token in cell for token in ["wavelength", "intensity", "counts", "channel"]) for cell in header) else rows
        wavelength_idx = self._find_column(header, ["wavelength", "lambda", "nm"])
        intensity_idx = self._find_column(header, ["intensity", "counts", "signal", "power", "value"])
        channel_idx = self._find_column(header, ["channel", "channel_id", "sensor"])
        time_idx = self._find_column(header, ["time", "timestamp"])

        if intensity_idx is None and len(data_rows[0]) >= 2:
            intensity_idx = 1
        if wavelength_idx is None and len(data_rows[0]) >= 2 and channel_idx is None and time_idx is None:
            wavelength_idx = 0

        if wavelength_idx is not None and intensity_idx is not None and channel_idx is None:
            wavelength: list[float] = []
            intensity: list[float] = []
            for row in data_rows:
                if len(row) <= max(wavelength_idx, intensity_idx):
                    continue
                w = _safe_float(row[wavelength_idx])
                signal = _safe_float(row[intensity_idx])
                if w is not None and signal is not None:
                    wavelength.append(w)
                    intensity.append(signal)
            if not intensity:
                return []
            return [
                {
                    "source": source,
                    "device_id": device_id,
                    "channels": [
                        {
                            "channel_id": channel_id,
                            "wavelength_nm": wavelength,
                            "intensity": intensity,
                        }
                    ],
                }
            ]

        payloads: list[dict[str, Any]] = []
        if intensity_idx is None:
            return payloads
        for row in data_rows:
            if len(row) <= intensity_idx:
                continue
            signal = _safe_float(row[intensity_idx])
            if signal is None:
                continue
            record_channel = channel_id
            if channel_idx is not None and len(row) > channel_idx and row[channel_idx].strip():
                record_channel = row[channel_idx].strip()
            timestamp = _safe_float(row[time_idx]) if time_idx is not None and len(row) > time_idx else None
            payload: dict[str, Any] = {
                "source": source,
                "device_id": device_id,
                "channels": [{"channel_id": record_channel, "intensity_counts": signal}],
            }
            if timestamp is not None:
                payload["timestamp"] = timestamp
            payloads.append(payload)
        return payloads

    def ingest_export_file(
        self,
        path: str | Path,
        channel_id: str = "P22",
        source: str = "bayspec_sense2020_export_file",
    ) -> dict[str, Any]:
        export_path = Path(path)
        if export_path.suffix.lower() == ".dat":
            return self.ingest_fast_record_dat(export_path, channel_id=channel_id, source=source)
        text = export_path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_csv_text(text, channel_id=channel_id, source=source)

    def ingest_fast_record_dat(
        self,
        path: str | Path,
        channel_id: str = "P22",
        source: str = "bayspec_sense2020_fast_record_dat",
    ) -> dict[str, Any]:
        dat_path = Path(path)
        try:
            sequence = read_sense_fast_dat(dat_path)
        except (OSError, ValueError) as exc:
            return {"ok": False, "reason": str(exc), "records_ingested": 0}
        frame_count = sequence.layout.frame_count
        intensity = [float(value) for value in sequence.spectra[-1]]
        if not intensity:
            return {"ok": False, "reason": "DAT frame has no intensity data", "records_ingested": 0}

        peak_index = max(range(len(intensity)), key=intensity.__getitem__)
        sorted_values = sorted(intensity, reverse=True)
        top_count = min(5, len(sorted_values))
        peak_intensity = sum(sorted_values[:top_count]) / top_count
        wavelength_grid = self._latest_wavelength_grid(
            dat_path.parent,
            expected_points=len(intensity),
        )
        channel_payload: dict[str, Any] = {
            "channel_id": channel_id,
            "intensity_counts": peak_intensity,
            "intensity": intensity,
            "fast_record_frame_index": frame_count - 1,
            "fast_record_frame_count": frame_count,
            "peak_pixel_index": peak_index,
        }
        if wavelength_grid and len(wavelength_grid) == len(intensity):
            channel_payload["wavelength_nm"] = wavelength_grid
            channel_payload["peak_wavelength_nm"] = wavelength_grid[peak_index]
            channel_payload["peak_axis_type"] = "wavelength_nm"
            channel_payload["spectrum_x_unit"] = "wavelength_nm"
        else:
            channel_payload["spectrum_x_unit"] = "pixel_index"
            channel_payload["peak_axis_type"] = "pixel_index"
            channel_payload["peak_selection_method"] = "fast_record_pixel_peak_no_wavelength_grid"

        result = self.ingest(
            {
                "source": source,
                "device_id": configured_device_id(),
                "channels": [channel_payload],
            }
        )
        result["source_file"] = str(dat_path)
        result["dat_frame_count"] = frame_count
        result["dat_frame_index"] = frame_count - 1
        result["dat_header_bytes"] = sequence.layout.prefix_words * 2
        result["dat_record_words"] = sequence.layout.record_words
        result["dat_auxiliary_words_per_frame"] = (
            sequence.layout.record_words - sequence.layout.spectrum_words
        )
        result["dat_trailing_words_ignored"] = sequence.layout.trailing_words
        result["dat_layout_correlation"] = (
            sequence.layout.median_adjacent_correlation
        )
        result["dat_layout_score_margin"] = sequence.layout.score_margin
        result["dat_parser"] = sequence.layout.name
        return result

    def _latest_wavelength_grid(self, root: Path, expected_points: int = 512) -> list[float] | None:
        try:
            csv_files = sorted(root.glob("Spectrum_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            csv_files = []
        for csv_path in csv_files[:5]:
            try:
                rows = list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8", errors="ignore"))))
            except OSError:
                continue
            header_idx = None
            for idx, row in enumerate(rows):
                if len(row) >= 2 and row[0].strip().lower() in {"wl", "wavelength"}:
                    header_idx = idx
                    break
            if header_idx is None:
                continue
            wavelengths: list[float] = []
            for row in rows[header_idx + 1 :]:
                if len(row) < 2:
                    continue
                wavelength = _safe_float(row[0])
                if wavelength is None:
                    if wavelengths:
                        break
                    continue
                wavelengths.append(wavelength)
                if len(wavelengths) >= expected_points:
                    break
            if len(wavelengths) == expected_points:
                return wavelengths
        return None

    def _sense_export_rows_to_payload(
        self,
        rows: list[list[str]],
        channel_id: str,
        device_id: str,
        source: str,
    ) -> dict[str, Any] | None:
        metadata: dict[str, str] = {}
        spectrum_header_idx: int | None = None
        peak_header_idx: int | None = None
        data_count: int | None = None

        for idx, row in enumerate(rows):
            first = row[0].strip().lower() if row else ""
            second = row[1].strip().lower() if len(row) > 1 else ""
            if len(row) >= 2 and first not in {"wl", "peak_wl"}:
                metadata[first] = row[1].strip()
            if first == "data_count":
                number = _safe_float(row[1] if len(row) > 1 else None)
                if number is not None:
                    data_count = max(0, int(number))
            if spectrum_header_idx is None and first in {"wl", "wavelength"} and (
                "power" in second or "intensity" in second or "count" in second
            ):
                spectrum_header_idx = idx
            if peak_header_idx is None and first in {"peak_wl", "peak wavelength", "peak_wavelength"}:
                peak_header_idx = idx

        if spectrum_header_idx is None and peak_header_idx is None:
            return None

        spectrum_wavelengths: list[float] = []
        spectrum_intensities: list[float] = []
        if spectrum_header_idx is not None:
            previous_wavelength: float | None = None
            for row in rows[spectrum_header_idx + 1 :]:
                if data_count is not None and len(spectrum_wavelengths) >= data_count:
                    break
                if len(row) < 2:
                    continue
                first = row[0].strip().lower()
                if first.startswith("peak_") or first == "peak_count":
                    break
                wavelength = _safe_float(row[0])
                intensity = _safe_float(row[1])
                if wavelength is None or intensity is None:
                    if spectrum_wavelengths:
                        break
                    continue
                if previous_wavelength is not None and wavelength < previous_wavelength - 1e-6:
                    break
                spectrum_wavelengths.append(wavelength)
                spectrum_intensities.append(intensity)
                previous_wavelength = wavelength

        peak_rows: list[tuple[float, float, float | None]] = []
        if peak_header_idx is not None:
            for row in rows[peak_header_idx + 1 :]:
                if len(row) < 2:
                    continue
                wavelength = _safe_float(row[0])
                power = _safe_float(row[1])
                fwhm = _safe_float(row[2]) if len(row) > 2 else None
                if wavelength is None or power is None:
                    if peak_rows:
                        break
                    continue
                peak_rows.append((wavelength, power, fwhm))

        channel_payload: dict[str, Any] = {"channel_id": channel_id}
        if spectrum_wavelengths and spectrum_intensities:
            channel_payload["wavelength_nm"] = spectrum_wavelengths
            channel_payload["intensity"] = spectrum_intensities

        if peak_rows:
            peak_wavelength, peak_power, peak_fwhm = max(peak_rows, key=lambda item: item[1])
            channel_payload["peak_wavelength_nm"] = peak_wavelength
            channel_payload["intensity_counts"] = peak_power
            if peak_fwhm is not None:
                channel_payload["peak_fwhm"] = peak_fwhm

        if not spectrum_intensities and "intensity_counts" not in channel_payload:
            return None

        device = metadata.get("device_sn") or metadata.get("device sn") or device_id
        integration = _safe_float(metadata.get("integration_time(ms)") or metadata.get("integration_time") or metadata.get("integration"))
        temperature = _safe_float(metadata.get("device_temperature") or metadata.get("temperature"))
        if integration is not None:
            channel_payload["integration_ms"] = integration
        if temperature is not None:
            channel_payload["temperature_c"] = temperature

        return {
            "source": source,
            "device_id": device,
            "channels": [channel_payload],
        }

    @staticmethod
    def _find_column(header: list[str], candidates: list[str]) -> int | None:
        for idx, name in enumerate(header):
            clean = name.strip().lower()
            for candidate in candidates:
                if candidate in clean:
                    return idx
        return None

    def latest_export_file(self, root: str | Path | None = None) -> Path | None:
        search_root = Path(root) if root else configured_sense_export_root()
        if not search_root.exists():
            return None
        latest: Path | None = None
        latest_mtime = float("-inf")
        for pattern in ("*.csv", "*.txt", "*.tsv", "*.dat"):
            for path in search_root.rglob(pattern):
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    # Sense may finalize an export by renaming it while this
                    # scan is in progress. Skip the transient path and retry on
                    # the next watcher/status pass.
                    continue
                if modified > latest_mtime:
                    latest = path
                    latest_mtime = modified
        return latest

    def ingest_latest_export(self, root: str | Path | None = None, channel_id: str = "P22") -> dict[str, Any]:
        latest = self.latest_export_file(root=root)
        if latest is None:
            return {
                "ok": False,
                "reason": "no CSV/TXT/TSV/DAT export found",
                "export_root": str(root or configured_sense_export_root()),
            }
        result = self.ingest_export_file(latest, channel_id=channel_id, source="bayspec_sense2020_export_file")
        result["source_file"] = str(latest)
        return result

    def _sense_process_status_cached(self) -> dict[str, Any]:
        now = _now()
        if now - self._sense_status_checked_at < 2.0:
            return dict(self._sense_status_cache)
        self._sense_status_cache = self._sense_process_status()
        self._sense_status_checked_at = now
        return dict(self._sense_status_cache)

    def _status_external_snapshot(self) -> tuple[Path | None, dict[str, Any]]:
        now = _now()
        with self._status_io_lock:
            if now - self._latest_export_status_checked_at >= 2.0:
                self._latest_export_status_cache = self.latest_export_file()
                self._latest_export_status_checked_at = now
            latest_file = self._latest_export_status_cache
            sense_process = self._sense_process_status_cached()
        return latest_file, sense_process

    @staticmethod
    def _sense_process_status() -> dict[str, Any]:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Sense 2020 V1633 20160627.exe"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                creationflags=creationflags,
            )
            output = result.stdout or ""
            # tasklist truncates long image names and may omit the .exe suffix.
            running = "Sense 2020 V1633 20160627" in output
            return {"running": running, "method": "tasklist_cached_hidden"}
        except Exception as exc:
            return {"running": None, "method": "tasklist_cached_hidden", "error": str(exc)}


bridge = BaySpecWavelengthShiftBridge()
