"""Validated configuration loading for the isolated mFBG intensity profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DISPLAY_ROWS = (
    ("P11", "P21", "P31"),
    ("P12", "P22", "P32"),
    ("P13", "P23", "P33"),
)
CHANNEL_ORDER = tuple(channel for row in DISPLAY_ROWS for channel in row)


@dataclass(frozen=True)
class MfbgChannelConfig:
    channel_id: str
    x: float
    y: float
    fiber_id: str
    fiber_order: int
    target_wavelength_nm: float
    measured_wavelength_nm: float | None
    search_half_width_nm: float
    integration_half_width_nm: float
    response_polarity: str
    enabled_for_real_demodulation: bool

    @property
    def demodulation_wavelength_nm(self) -> float:
        if self.measured_wavelength_nm is not None:
            return self.measured_wavelength_nm
        return self.target_wavelength_nm


@dataclass(frozen=True)
class MfbgIntensityProfile:
    config_path: Path
    raw: dict[str, Any]
    profile_id: str
    display_name: str
    status: str
    channels: dict[str, MfbgChannelConfig]
    channel_order: tuple[str, ...]
    display_rows: tuple[tuple[str, ...], ...]
    responding_threshold_ratio: float
    baseline_minimum_frames: int
    real_3x3_enabled: bool

    def summary(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "status": self.status,
            "demodulation_mode": "spectral_window_intensity",
            "primary_metrics": [
                "intensity_counts",
                "relative_intensity",
                "attenuation_ratio",
                "loss_db",
            ],
            "wavelength_shift_role": "auxiliary_diagnostic_only",
            "channel_order": list(self.channel_order),
            "display_rows": [list(row) for row in self.display_rows],
            "channel_count": len(self.channels),
            "real_3x3_enabled": self.real_3x3_enabled,
            "calibrated_force": False,
            "force_N_output": False,
            "calibrated_pressure_output": False,
            "surface_semantics": "raw_coupled_optical_attenuation_proxy",
            "config_path": str(self.config_path),
        }


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "mfbg_intensity_3x3.yaml"


def _as_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    return result


def _parse_rows(raw_rows: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw_rows, list):
        raise ValueError("array.display_rows must be a list")
    rows = tuple(tuple(str(channel) for channel in row) for row in raw_rows)
    if rows != DISPLAY_ROWS:
        raise ValueError(
            "mFBG display orientation must be "
            "P11/P21/P31, P12/P22/P32, P13/P23/P33"
        )
    return rows


def load_profile(path: str | Path | None = None) -> MfbgIntensityProfile:
    config_path = Path(path).resolve() if path is not None else default_config_path()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    profile_payload = payload.get("profile") or {}
    array_payload = payload.get("array") or {}
    boundaries = payload.get("physical_boundaries") or {}
    baseline_payload = payload.get("baseline") or {}
    raw_channels = payload.get("channels") or {}

    display_rows = _parse_rows(array_payload.get("display_rows"))
    channel_order = tuple(str(value) for value in array_payload.get("channel_order") or ())
    if channel_order != CHANNEL_ORDER:
        raise ValueError("array.channel_order does not match the physical display rows")
    if set(raw_channels) != set(CHANNEL_ORDER):
        raise ValueError("mFBG intensity profile must configure exactly P11 through P33")

    channels: dict[str, MfbgChannelConfig] = {}
    for channel_id in CHANNEL_ORDER:
        item = raw_channels[channel_id] or {}
        measured = item.get("measured_wavelength_nm")
        channels[channel_id] = MfbgChannelConfig(
            channel_id=channel_id,
            x=_as_float(item.get("x"), field=f"{channel_id}.x"),
            y=_as_float(item.get("y"), field=f"{channel_id}.y"),
            fiber_id=str(item.get("fiber_id") or ""),
            fiber_order=int(item.get("fiber_order") or 0),
            target_wavelength_nm=_as_float(
                item.get("target_wavelength_nm"),
                field=f"{channel_id}.target_wavelength_nm",
            ),
            measured_wavelength_nm=(
                _as_float(measured, field=f"{channel_id}.measured_wavelength_nm")
                if measured is not None
                else None
            ),
            search_half_width_nm=_as_float(
                item.get("search_half_width_nm"),
                field=f"{channel_id}.search_half_width_nm",
            ),
            integration_half_width_nm=_as_float(
                item.get("integration_half_width_nm"),
                field=f"{channel_id}.integration_half_width_nm",
            ),
            response_polarity=str(item.get("response_polarity") or "attenuation"),
            enabled_for_real_demodulation=bool(
                item.get("enabled_for_real_demodulation", False)
            ),
        )

    return MfbgIntensityProfile(
        config_path=config_path,
        raw=payload,
        profile_id=str(profile_payload.get("profile_id") or "mfbg_intensity_3x3"),
        display_name=str(profile_payload.get("display_name") or "mFBG 3x3"),
        status=str(profile_payload.get("status") or "configuration_loaded"),
        channels=channels,
        channel_order=channel_order,
        display_rows=display_rows,
        responding_threshold_ratio=_as_float(
            array_payload.get("responding_threshold_ratio", 0.05),
            field="array.responding_threshold_ratio",
        ),
        baseline_minimum_frames=max(
            1, int(baseline_payload.get("minimum_frames") or 1)
        ),
        real_3x3_enabled=bool(boundaries.get("real_3x3_enabled", False)),
    )
