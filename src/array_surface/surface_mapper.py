"""Continuous Bragg wavelength-shift response surface mapping.

The surface maps normalized absolute wavelength displacement. Neighboring
responses may reflect fingertip contact-area coverage and shared-elastomer
mechanical coupling. It is not a calibrated force or pressure reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CHANNEL_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
DISPLAY_ROWS = [
    ["P11", "P21", "P31"],
    ["P12", "P22", "P32"],
    ["P13", "P23", "P33"],
]
DEFAULT_COORDS = {
    "P11": (-1.0, 1.0),
    "P12": (-1.0, 0.0),
    "P13": (-1.0, -1.0),
    "P21": (0.0, 1.0),
    "P22": (0.0, 0.0),
    "P23": (0.0, -1.0),
    "P31": (1.0, 1.0),
    "P32": (1.0, 0.0),
    "P33": (1.0, -1.0),
}


@dataclass(frozen=True)
class SurfaceConfig:
    grid_size: int = 80
    sigma: float = 0.65
    clip_min: float = 0.0
    clip_max: float = 1.0
    active_threshold: float = 0.05
    active_absolute_threshold: float = 0.10
    active_relative_threshold: float = 0.25
    surface_input_mode: str = "raw_coupled_response_surface"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _channel_position(channel_id: str, channel_config: dict[str, Any]) -> tuple[float, float]:
    default_x, default_y = DEFAULT_COORDS.get(channel_id, (0.0, 0.0))
    x = _safe_float(channel_config.get("x"), default_x)
    y = _safe_float(channel_config.get("y"), default_y)
    return float(x), float(y)


def response_value(record: dict[str, Any] | None, surface_input_mode: str = "raw_coupled_response_surface") -> float:
    """Return a clipped 0..1 response for surface mapping."""
    if not record:
        return 0.0
    for key in (
        "observed_wavelength_shift_response_ratio",
        "wavelength_shift_response_ratio",
        "response_value",
    ):
        shift_response = _safe_float(record.get(key))
        if shift_response is not None:
            return max(0.0, min(1.0, shift_response))
    if surface_input_mode == "coupling_compensated_surface":
        attenuation = _safe_float(record.get("local_response_estimate"))
        if attenuation is not None:
            return max(0.0, min(1.0, attenuation))
    attenuation = _safe_float(record.get("observed_attenuation_ratio"))
    if attenuation is None:
        attenuation = _safe_float(record.get("attenuation_ratio"))
    if attenuation is not None:
        return max(0.0, min(1.0, attenuation))
    loss_db = _safe_float(record.get("observed_loss_db"))
    if loss_db is None:
        loss_db = _safe_float(record.get("intensity_loss_db"))
    if loss_db is not None:
        return max(0.0, min(1.0, loss_db / 12.0))
    return 0.0


def build_array_channels(
    latest_by_channel: dict[str, dict[str, Any]],
    channel_configs: dict[str, dict[str, Any]],
    channel_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize the 3x3 channel records used by the frontend and recorder."""
    channels: list[dict[str, Any]] = []
    order = channel_order or CHANNEL_ORDER
    for channel_id in order:
        cfg = dict(channel_configs.get(channel_id, {}))
        record = dict(latest_by_channel.get(channel_id) or {})
        enabled = bool(cfg.get("enabled"))
        valid = bool(enabled and record.get("intensity_counts") is not None and record.get("qa_status") != "invalid")
        x, y = _channel_position(channel_id, cfg)
        observed_intensity = record.get("intensity_counts")
        observed_relative = record.get("relative_intensity")
        observed_attenuation = record.get("attenuation_ratio")
        observed_loss = record.get("intensity_loss_db")
        shift_response = record.get("wavelength_shift_response_ratio")
        observed_shift_response = record.get("observed_wavelength_shift_response_ratio")
        local_response = record.get("local_response_estimate")
        if local_response is None:
            local_response = shift_response
        channels.append(
            {
                "channel_id": channel_id,
                "display_name": cfg.get("display_name") or channel_id,
                "enabled": enabled,
                "valid": valid,
                "x": x,
                "y": y,
                "target_wavelength_nm": cfg.get("target_wavelength_nm"),
                "measured_wavelength_nm": record.get("measured_wavelength_nm") or cfg.get("measured_wavelength_nm"),
                "demodulation_wavelength_nm": record.get("demodulation_wavelength_nm")
                or cfg.get("measured_wavelength_nm")
                or cfg.get("target_wavelength_nm"),
                "search_half_width_nm": cfg.get("search_half_width_nm"),
                "peak_wavelength_nm": record.get("peak_wavelength_nm"),
                "tracked_wavelength_nm": record.get("tracked_wavelength_nm"),
                "baseline_wavelength_nm": record.get("baseline_wavelength_nm"),
                "delta_wavelength_pm": record.get("delta_wavelength_pm"),
                "absolute_shift_pm": record.get("absolute_shift_pm"),
                "shift_direction": record.get("shift_direction"),
                "wavelength_tracking_method": record.get("wavelength_tracking_method"),
                "cross_correlation_coefficient": record.get("cross_correlation_coefficient"),
                "wavelength_shift_response_ratio": shift_response,
                "observed_wavelength_shift_response_ratio": observed_shift_response
                if observed_shift_response is not None
                else shift_response,
                "peak_pixel_index": record.get("peak_pixel_index"),
                "intensity_counts": observed_intensity,
                "baseline_intensity_counts": record.get("baseline_intensity_counts"),
                "relative_intensity": observed_relative,
                "attenuation_ratio": observed_attenuation,
                "intensity_loss_db": observed_loss,
                "observed_intensity_counts": observed_intensity,
                "observed_relative_intensity": observed_relative,
                "observed_attenuation_ratio": observed_attenuation,
                "observed_loss_db": observed_loss,
                "local_response_estimate": local_response,
                "coupling_compensated": bool(record.get("coupling_compensated") or False),
                "coupling_status": record.get("coupling_status")
                or "uncalibrated_mechanically_coupled_wavelength_shift",
                "coupling_sources": record.get("coupling_sources") or [],
                "coupling_roles": record.get("coupling_roles") or [],
                "possible_cross_fiber_coupling": bool(record.get("possible_cross_fiber_coupling") or False),
                "possible_same_fiber_coupling": bool(record.get("possible_same_fiber_coupling") or False),
                "local_response_estimate_available": bool(record.get("local_response_estimate_available") or False),
                "response_level": record.get("response_level") if record else "no_data",
                "qa_status": record.get("qa_status") if record else "no_data",
                "qa_flags": record.get("qa_flags") or [],
                "response_value": response_value(record),
            }
        )
    return channels


def _empty_grid(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def map_surface(
    channels: list[dict[str, Any]],
    config: SurfaceConfig | None = None,
) -> dict[str, Any]:
    """Map valid channels onto a continuous wavelength-shift response surface."""
    cfg = config or SurfaceConfig()
    size = max(8, int(cfg.grid_size))
    sigma = max(0.05, float(cfg.sigma))
    enabled_channel_count = sum(1 for ch in channels if ch.get("enabled"))
    valid_channels = [ch for ch in channels if ch.get("valid")]
    response_channels = [ch for ch in valid_channels if response_value(ch, cfg.surface_input_mode) > 0]
    xs = [-1.25 + (2.5 * i) / (size - 1) for i in range(size)]
    ys = [1.25 - (2.5 * j) / (size - 1) for j in range(size)]
    if not response_channels:
        grid = _empty_grid(size)
    else:
        grid = []
        for y in ys:
            row: list[float] = []
            for x in xs:
                weighted_sum = 0.0
                weight_sum = 0.0
                peak_envelope = 0.0
                for channel in response_channels:
                    dx = x - float(channel.get("x") or 0.0)
                    dy = y - float(channel.get("y") or 0.0)
                    weight = math.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma)))
                    response = response_value(channel, cfg.surface_input_mode)
                    weighted_sum += response * weight
                    weight_sum += weight
                    peak_envelope = max(peak_envelope, response * weight)
                # Normalize overlapping kernels so the visual surface cannot
                # exceed the strongest measured channel. The support factor
                # preserves a local Gaussian footprint for single-point P22.
                normalized = weighted_sum / weight_sum if weight_sum > 1e-12 else 0.0
                support = min(1.0, weight_sum)
                value = max(peak_envelope, normalized * support)
                row.append(max(cfg.clip_min, min(cfg.clip_max, value)))
            grid.append(row)
    metrics = surface_metrics(
        grid,
        xs,
        ys,
        valid_channels,
        cfg.active_threshold,
        cfg.active_absolute_threshold,
        cfg.active_relative_threshold,
        cfg.surface_input_mode,
        enabled_channel_count,
    )
    p22_only = len(valid_channels) == 1 and valid_channels[0].get("channel_id") == "P22"
    return {
        "surface_grid": grid,
        "grid_x": xs,
        "grid_y": ys,
        "surface_metrics": metrics,
        "enabled_channel_count": enabled_channel_count,
        "valid_channel_count": len(valid_channels),
        "responding_channel_count": metrics.get("responding_channel_count", 0),
        "responding_channel_ids": metrics.get("responding_channel_ids", []),
        "active_channel_count": metrics.get("responding_channel_count", 0),
        "surface_input_mode": cfg.surface_input_mode,
        "surface_mode": "p22_fallback" if p22_only else "array_surface",
        "surface_note": "single-point P22 fallback, raw coupled response semantics reserved for array mode"
        if p22_only
        else "raw coupled Bragg wavelength-shift surface",
    }


def surface_metrics(
    grid: list[list[float]],
    xs: list[float],
    ys: list[float],
    valid_channels: list[dict[str, Any]],
    active_threshold: float,
    active_absolute_threshold: float = 0.10,
    active_relative_threshold: float = 0.25,
    surface_input_mode: str = "raw_coupled_response_surface",
    enabled_channel_count: int | None = None,
) -> dict[str, Any]:
    flat = [value for row in grid for value in row]
    total = sum(flat)
    peak = max(flat) if flat else 0.0
    mean = total / max(len(flat), 1)
    threshold = max(float(active_threshold), float(active_absolute_threshold), float(active_relative_threshold) * peak)
    active = [value for value in flat if value >= threshold]
    active_area = len(active) / max(len(flat), 1)
    if total > 1e-12:
        cx = 0.0
        cy = 0.0
        spread_sum = 0.0
        for j, row in enumerate(grid):
            for i, value in enumerate(row):
                cx += xs[i] * value
                cy += ys[j] * value
        cx /= total
        cy /= total
        for j, row in enumerate(grid):
            for i, value in enumerate(row):
                spread_sum += ((xs[i] - cx) ** 2 + (ys[j] - cy) ** 2) * value
        spread = math.sqrt(spread_sum / total)
        probs = [value / total for value in flat if value > 1e-12]
        entropy = -sum(p * math.log(p) for p in probs) / math.log(max(len(flat), 2))
    else:
        cx = 0.0
        cy = 0.0
        spread = 0.0
        entropy = 0.0
    left = sum(value for row in grid for i, value in enumerate(row) if xs[i] < 0)
    right = sum(value for row in grid for i, value in enumerate(row) if xs[i] > 0)
    top = sum(value for j, row in enumerate(grid) if ys[j] > 0 for value in row)
    bottom = sum(value for j, row in enumerate(grid) if ys[j] < 0 for value in row)
    dominant = max(valid_channels, key=lambda ch: response_value(ch, surface_input_mode), default={})
    changed_channels = [
        ch for ch in valid_channels if response_value(ch, surface_input_mode) >= active_threshold
    ]
    primary = max(
        changed_channels,
        key=lambda ch: response_value(ch, surface_input_mode),
        default={},
    )
    primary_id = primary.get("channel_id") or "none"
    secondary = [ch.get("channel_id") for ch in changed_channels if ch.get("channel_id") != primary_id]
    dominant_peak = response_value(primary, surface_input_mode)
    coupled_sum = sum(response_value(ch, surface_input_mode) for ch in changed_channels)
    coupling_statuses = {
        str(ch.get("coupling_status") or "uncalibrated_mechanically_coupled_wavelength_shift")
        for ch in valid_channels
    }
    if "debug_independent_simulation" in coupling_statuses:
        coupling_status = "debug_independent_simulation"
    elif coupling_statuses == {"coupling_compensated"}:
        coupling_status = "coupling_compensated"
    else:
        coupling_status = "uncalibrated_mechanically_coupled_wavelength_shift"
    responding_channel_ids = [str(ch.get("channel_id")) for ch in changed_channels if ch.get("channel_id")]
    if enabled_channel_count is None:
        enabled_channel_count = sum(1 for ch in valid_channels if ch.get("enabled"))
    valid_channel_count = len(valid_channels)
    responding_channel_count = len(changed_channels)
    if responding_channel_count == 0:
        event_interpretation = "no_contact / baseline / no active contact"
    elif responding_channel_count == 1:
        event_interpretation = "single responding Bragg peak within coupled tactile surface"
    else:
        event_interpretation = "distributed fingertip / mechanically coupled wavelength-shift response"
    possible_cross_fiber = any(bool(ch.get("possible_cross_fiber_coupling")) for ch in changed_channels)
    possible_same_fiber = any(bool(ch.get("possible_same_fiber_coupling")) for ch in changed_channels)
    local_response_available = any(bool(ch.get("local_response_estimate_available")) for ch in changed_channels)
    coupling_sources = sorted(
        {
            str(source)
            for ch in changed_channels
            for source in (ch.get("coupling_sources") or [])
            if source
        }
    )
    quality_counts = {str(ch.get("qa_status") or "no_data") for ch in valid_channels}
    quality_status = "ok" if quality_counts in (set(), {"ok"}) else "warning"
    return {
        "surface_peak": peak,
        "surface_mean": mean,
        "surface_area_active": active_area,
        "surface_area_active_percent": active_area * 100.0,
        "surface_active_threshold_used": threshold,
        "surface_active_absolute_threshold": float(active_absolute_threshold),
        "surface_active_relative_threshold": float(active_relative_threshold),
        "surface_centroid_x": cx,
        "surface_centroid_y": cy,
        "surface_spread": spread,
        "surface_entropy": entropy,
        "left_right_asymmetry": (right - left) / max(right + left, 1e-12),
        "top_bottom_asymmetry": (top - bottom) / max(top + bottom, 1e-12),
        "dominant_channel": dominant.get("channel_id") or "none",
        "enabled_channel_count": enabled_channel_count,
        "responding_channel_count": responding_channel_count,
        "responding_channel_ids": responding_channel_ids,
        "valid_channel_count": valid_channel_count,
        "active_channel_count": responding_channel_count,
        "quality_status": quality_status,
        "surface_input_mode": surface_input_mode,
        "coupling_status": coupling_status,
        "coupling_sources": coupling_sources,
        "observed_changed_channels": responding_channel_ids,
        "primary_observed_channel": primary_id,
        "secondary_changed_channels": secondary,
        "secondary_observed_channels": secondary,
        "possible_cross_fiber_coupling": possible_cross_fiber,
        "possible_same_fiber_coupling": possible_same_fiber,
        "local_response_estimate_available": local_response_available,
        "num_changed_peaks": len(changed_channels),
        "dominant_peak_attenuation": dominant_peak,
        "coupled_peak_attenuation_sum": coupled_sum,
        "dominant_peak_shift_response": dominant_peak,
        "coupled_shift_response_sum": coupled_sum,
        "event_interpretation": event_interpretation,
    }


def matrices_from_channels(channels: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {channel["channel_id"]: channel for channel in channels}
    response: list[list[float | None]] = []
    quality: list[list[str]] = []
    mask: list[list[bool]] = []
    for row in DISPLAY_ROWS:
        response.append([response_value(by_id.get(channel_id)) if by_id.get(channel_id, {}).get("valid") else None for channel_id in row])
        quality.append([str(by_id.get(channel_id, {}).get("qa_status") or "no_data") for channel_id in row])
        mask.append([bool(by_id.get(channel_id, {}).get("valid")) for channel_id in row])
    return {
        "array_response_3x3": response,
        "array_quality_3x3": quality,
        "valid_channel_mask_3x3": mask,
    }
