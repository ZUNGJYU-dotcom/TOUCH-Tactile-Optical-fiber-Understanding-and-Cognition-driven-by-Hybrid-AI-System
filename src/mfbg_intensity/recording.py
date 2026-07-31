"""Recorder adapters for mFBG intensity frames."""

from __future__ import annotations

import json
from typing import Any


def frame_to_channel_rows(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one tidy row per channel for analysis-friendly CSV output."""
    rows: list[dict[str, Any]] = []
    for channel in frame.get("channels") or []:
        rows.append(
            {
                "timestamp": frame.get("timestamp"),
                "frame_id": frame.get("frame_id"),
                "profile_id": frame.get("profile_id"),
                "source": frame.get("source"),
                "channel_id": channel.get("channel_id"),
                "fiber_id": channel.get("fiber_id"),
                "target_wavelength_nm": channel.get("target_wavelength_nm"),
                "measured_wavelength_nm": channel.get("measured_wavelength_nm"),
                "tracked_wavelength_nm": channel.get("tracked_wavelength_nm"),
                "intensity_counts": channel.get("intensity_counts"),
                "baseline_intensity_counts": channel.get(
                    "baseline_intensity_counts"
                ),
                "relative_intensity": channel.get("relative_intensity"),
                "attenuation_ratio": channel.get("attenuation_ratio"),
                "loss_db": channel.get("loss_db"),
                "responding": channel.get("responding"),
                "qa_status": channel.get("qa_status"),
                "qa_flags": "|".join(channel.get("qa_flags") or []),
            }
        )
    return rows


def frame_to_wide_row(frame: dict[str, Any]) -> dict[str, Any]:
    """Return one flattened row per spectrum frame for synchronized capture."""
    row: dict[str, Any] = {
        "timestamp": frame.get("timestamp"),
        "frame_id": frame.get("frame_id"),
        "profile_id": frame.get("profile_id"),
        "source": frame.get("source"),
        "baseline_ready": frame.get("baseline_ready"),
        "responding_channel_ids": "|".join(
            frame.get("responding_channel_ids") or []
        ),
        "contact_regions_json": json.dumps(
            frame.get("contact_regions") or [],
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "qa_status": frame.get("qa_status"),
        "qa_flags": "|".join(frame.get("qa_flags") or []),
    }
    for channel in frame.get("channels") or []:
        prefix = str(channel.get("channel_id") or "unknown")
        for key in (
            "intensity_counts",
            "baseline_intensity_counts",
            "relative_intensity",
            "attenuation_ratio",
            "loss_db",
            "tracked_wavelength_nm",
        ):
            row[f"{prefix}_{key}"] = channel.get(key)
    return row
