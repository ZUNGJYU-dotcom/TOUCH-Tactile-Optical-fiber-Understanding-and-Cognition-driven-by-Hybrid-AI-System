"""Isolated API for the future mFBG 3x3 optical-intensity profile."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from bridge import bridge
from src.mfbg_intensity import (
    MfbgIntensityDemodulator,
    frame_to_channel_rows,
    frame_to_wide_row,
    load_profile,
)
from src.mfbg_intensity.profiles import profile_registry


def _config_path() -> Path:
    override = os.environ.get("TOUCH_MFBG_INTENSITY_CONFIG")
    if override:
        return Path(override).resolve()
    app_root = Path(
        os.environ.get(
            "BAYSPEC_WAVELENGTH_APP_ROOT",
            Path(__file__).resolve().parents[1],
        )
    ).resolve()
    candidates = (
        app_root / "config" / "mfbg_intensity_3x3.yaml",
        app_root.parent / "config" / "mfbg_intensity_3x3.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


MFBG_INTENSITY_ENGINE = MfbgIntensityDemodulator(
    profile=load_profile(_config_path())
)
router = APIRouter(prefix="/api/mfbg-intensity", tags=["mFBG intensity"])


def _spectrum_from_record(record: dict[str, Any] | None) -> tuple[Any, Any]:
    record = record or {}
    return record.get("wavelength_nm"), record.get("intensity")


def _runtime_activation() -> dict[str, Any]:
    profile = MFBG_INTENSITY_ENGINE.profile
    enabled_ids = [
        channel_id
        for channel_id in profile.channel_order
        if profile.channels[channel_id].enabled_for_real_demodulation
    ]
    ready = bool(profile.real_3x3_enabled and len(enabled_ids) == len(profile.channel_order))
    reason = (
        None
        if ready
        else "mfbg_real_3x3_disabled"
        if not profile.real_3x3_enabled
        else "mfbg_real_channel_map_incomplete"
    )
    return {
        "real_runtime_ready": ready,
        "real_3x3_enabled": profile.real_3x3_enabled,
        "real_enabled_channel_ids": enabled_ids,
        "real_enabled_channel_count": len(enabled_ids),
        "required_channel_count": len(profile.channel_order),
        "reason": reason,
    }


def _analysis_gate(*, diagnostic_preview: bool) -> dict[str, Any] | None:
    activation = _runtime_activation()
    if activation["real_runtime_ready"] or diagnostic_preview:
        return None
    return {
        "ok": False,
        "reason": activation["reason"],
        "operator_eligible": False,
        "recording_eligible": False,
        "diagnostic_preview_available": True,
        "required_action": (
            "Enable the measured nine-channel mFBG configuration, or explicitly "
            "request diagnostic_preview=true for non-operational inspection."
        ),
        "runtime_activation": activation,
    }


def _tag_frame(
    frame_payload: dict[str, Any],
    *,
    diagnostic_preview: bool,
) -> dict[str, Any]:
    activation = _runtime_activation()
    operator_eligible = bool(activation["real_runtime_ready"] and not diagnostic_preview)
    frame_payload.update(
        {
            "diagnostic_preview": bool(diagnostic_preview),
            "operator_eligible": operator_eligible,
            "recording_eligible": operator_eligible,
            "runtime_activation": activation,
        }
    )
    return frame_payload


@router.get("/profiles")
def profiles() -> dict[str, Any]:
    return {
        "ok": True,
        **profile_registry(MFBG_INTENSITY_ENGINE.profile),
        "runtime_activation": _runtime_activation(),
    }


@router.get("/profile")
def profile() -> dict[str, Any]:
    return {
        "ok": True,
        **MFBG_INTENSITY_ENGINE.profile_summary(),
        "runtime_activation": _runtime_activation(),
        "analysis_contract": "fail_closed_unless_real_ready_or_explicit_diagnostic_preview",
    }


@router.get("/frame")
def frame() -> dict[str, Any]:
    latest = MFBG_INTENSITY_ENGINE.latest_frame()
    return {
        "ok": latest is not None,
        "frame": latest,
        "reason": None if latest is not None else "no_mfbg_intensity_frame",
    }


@router.post("/analyze-spectrum")
async def analyze_spectrum(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
        diagnostic_preview = bool(payload.get("diagnostic_preview", False))
        blocked = _analysis_gate(diagnostic_preview=diagnostic_preview)
        if blocked is not None:
            return blocked
        spectrum = payload.get("spectrum") or payload
        frame_payload = _tag_frame(
            MFBG_INTENSITY_ENGINE.analyze_spectrum(
                spectrum.get("wavelength_nm") or [],
                spectrum.get("intensity_counts")
                or spectrum.get("intensity")
                or [],
                timestamp=payload.get("timestamp"),
                source=str(payload.get("source") or "manual_api"),
                include_spectrum=payload.get("include_spectrum"),
            ),
            diagnostic_preview=diagnostic_preview,
        )
    except (TypeError, ValueError) as exc:
        return {"ok": False, "reason": str(exc)}
    return {"ok": True, "frame": frame_payload}


@router.post("/analyze-latest-bayspec-frame")
def analyze_latest_bayspec_frame(
    diagnostic_preview: bool = Query(default=False),
) -> dict[str, Any]:
    blocked = _analysis_gate(diagnostic_preview=diagnostic_preview)
    if blocked is not None:
        return blocked
    record = bridge.latest(channel_id="P22", include_spectrum=True)
    wavelength_nm, intensity = _spectrum_from_record(record)
    if not wavelength_nm or not intensity:
        return {
            "ok": False,
            "reason": "no_full_spectrum_available_from_bayspec_transport",
        }
    frame_payload = _tag_frame(
        MFBG_INTENSITY_ENGINE.analyze_spectrum(
            wavelength_nm,
            intensity,
            timestamp=record.get("timestamp"),
            source=str(record.get("source") or "bayspec_full_spectrum_transport"),
        ),
        diagnostic_preview=diagnostic_preview,
    )
    return {"ok": True, "frame": frame_payload}


@router.post("/baseline")
async def baseline(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
        diagnostic_preview = bool(payload.get("diagnostic_preview", False))
        blocked = _analysis_gate(diagnostic_preview=diagnostic_preview)
        if blocked is not None:
            return blocked
        result = MFBG_INTENSITY_ENGINE.set_baseline(
            payload.get("wavelength_nm") or [],
            payload.get("spectra")
            or payload.get("intensity_frames")
            or [],
        )
    except (TypeError, ValueError) as exc:
        return {"ok": False, "reason": str(exc)}
    result["diagnostic_preview"] = diagnostic_preview
    result["operator_eligible"] = bool(
        _runtime_activation()["real_runtime_ready"] and not diagnostic_preview
    )
    return result


@router.post("/baseline-from-recent-bayspec-frames")
def baseline_from_recent_bayspec_frames(
    diagnostic_preview: bool = Query(default=False),
) -> dict[str, Any]:
    blocked = _analysis_gate(diagnostic_preview=diagnostic_preview)
    if blocked is not None:
        return blocked
    required = MFBG_INTENSITY_ENGINE.profile.baseline_minimum_frames
    trace = bridge.trace(
        channel_id="P22",
        limit=max(required, 60),
        include_spectrum=True,
    )
    records = [
        record
        for record in (trace.get("records") or [])
        if record.get("wavelength_nm") and record.get("intensity")
    ]
    if len(records) < required:
        return {
            "ok": False,
            "reason": "insufficient_recent_full_spectrum_frames",
            "frame_count": len(records),
            "minimum_frames": required,
        }
    reference_axis = records[-1]["wavelength_nm"]
    compatible = [
        record["intensity"]
        for record in records
        if record.get("wavelength_nm") == reference_axis
    ]
    result = MFBG_INTENSITY_ENGINE.set_baseline(reference_axis, compatible)
    result["source"] = "recent_bayspec_full_spectrum_transport"
    result["diagnostic_preview"] = diagnostic_preview
    result["operator_eligible"] = bool(
        _runtime_activation()["real_runtime_ready"] and not diagnostic_preview
    )
    return result


@router.post("/reset")
def reset() -> dict[str, Any]:
    return MFBG_INTENSITY_ENGINE.reset()


@router.get("/recording-preview")
def recording_preview(
    diagnostic_preview: bool = Query(default=False),
) -> dict[str, Any]:
    blocked = _analysis_gate(diagnostic_preview=diagnostic_preview)
    if blocked is not None:
        return blocked
    latest = MFBG_INTENSITY_ENGINE.latest_frame()
    if latest is None:
        return {"ok": False, "reason": "no_mfbg_intensity_frame"}
    return {
        "ok": True,
        "diagnostic_preview": diagnostic_preview,
        "recording_eligible": bool(
            _runtime_activation()["real_runtime_ready"] and not diagnostic_preview
        ),
        "wide_row": frame_to_wide_row(latest),
        "channel_rows": frame_to_channel_rows(latest),
        "raw_spectrum_retained": bool(latest.get("raw_spectrum")),
        "recommended_streams": [
            "raw_spectrum",
            "mfbg_channel_intensity",
            "mfbg_response_frame",
            "force_sensor_reference",
        ],
    }
