"""Map dynamic spectral predictions to the physical 3x3 twin surface."""

from __future__ import annotations

from typing import Any

import numpy as np


ARRAY_DISPLAY_ROWS = (
    ("P11", "P21", "P31"),
    ("P12", "P22", "P32"),
    ("P13", "P23", "P33"),
)
ARRAY_DISPLAY_ORDER = tuple(channel for row in ARRAY_DISPLAY_ROWS for channel in row)
ARRAY_CHANNEL_COORDS = {
    "P11": (-1.0, 1.0),
    "P21": (0.0, 1.0),
    "P31": (1.0, 1.0),
    "P12": (-1.0, 0.0),
    "P22": (0.0, 0.0),
    "P32": (1.0, 0.0),
    "P13": (-1.0, -1.0),
    "P23": (0.0, -1.0),
    "P33": (1.0, -1.0),
}
RESPONSE_DEFORMATION_PROXY = {
    "light": 0.28,
    "normal": 0.58,
    "hard": 0.92,
}
RESPONDING_THRESHOLD = 0.055


def _inactive_proxy(status: str) -> dict[str, Any]:
    return {
        "active": False,
        "status": status,
        "position_id": None,
        "response_level": "no_contact",
        "deformation_proxy": 0.0,
        "surface_grid": [[0.0, 0.0, 0.0] for _ in range(3)],
        "surface_metrics": {
            "surface_peak": 0.0,
            "surface_mean": 0.0,
            "surface_area_active": 0.0,
            "surface_centroid_x": 0.0,
            "surface_centroid_y": 0.0,
            "surface_spread": 0.24,
            "dominant_channel": None,
            "responding_channel_count": 0,
            "responding_channel_ids": [],
        },
        "display_layout": [list(row) for row in ARRAY_DISPLAY_ROWS],
        "visualization_semantics": (
            "single_finger_contact_patch_from_temporal_spectral_prediction"
        ),
        "physical_output_semantics": "uncalibrated_manual_response_level",
    }


def dynamic_prediction_to_twin_proxy(prediction: dict[str, Any] | None) -> dict[str, Any]:
    """Create one orientation-safe continuous contact patch.

    The result is a visualization proxy. It deliberately maps the categorical
    response level to a monotonic deformation amplitude and does not represent
    calibrated physical force or a measured pressure field.
    """

    if not isinstance(prediction, dict) or prediction.get("ready") is not True:
        return _inactive_proxy("prediction_not_ready")
    if prediction.get("contact", {}).get("label") != "contact":
        return _inactive_proxy(str(prediction.get("operational_state") or "no_contact"))

    position_id = str((prediction.get("position") or {}).get("label") or "")
    response_level = str(
        (prediction.get("response_level") or {}).get("label") or ""
    ).lower()
    coordinate = ARRAY_CHANNEL_COORDS.get(position_id)
    peak = RESPONSE_DEFORMATION_PROXY.get(response_level)
    if coordinate is None or peak is None:
        return _inactive_proxy("position_or_response_unavailable")

    sigma = 0.78 + 0.12 * peak
    values_by_channel: dict[str, float] = {}
    for channel_id in ARRAY_DISPLAY_ORDER:
        point = ARRAY_CHANNEL_COORDS[channel_id]
        distance_squared = (
            (point[0] - coordinate[0]) ** 2 + (point[1] - coordinate[1]) ** 2
        )
        values_by_channel[channel_id] = float(
            peak * np.exp(-distance_squared / (2.0 * sigma * sigma))
        )
    grid = [
        [values_by_channel[channel_id] for channel_id in row]
        for row in ARRAY_DISPLAY_ROWS
    ]
    flat = np.asarray(grid, dtype=float).ravel()
    responding = [
        channel_id
        for channel_id in ARRAY_DISPLAY_ORDER
        if values_by_channel[channel_id] >= RESPONDING_THRESHOLD
    ]
    return {
        "active": True,
        "status": "active_contact",
        "position_id": position_id,
        "response_level": response_level,
        "deformation_proxy": peak,
        "surface_grid": grid,
        "surface_metrics": {
            "surface_peak": peak,
            "surface_mean": float(np.mean(flat)),
            "surface_area_active": len(responding) / len(ARRAY_DISPLAY_ORDER),
            "surface_centroid_x": coordinate[0],
            "surface_centroid_y": coordinate[1],
            "surface_spread": sigma,
            "dominant_channel": position_id,
            "responding_channel_count": len(responding),
            "responding_channel_ids": responding,
        },
        "channel_values": values_by_channel,
        "display_layout": [list(row) for row in ARRAY_DISPLAY_ROWS],
        "visualization_semantics": (
            "single_finger_contact_patch_from_temporal_spectral_prediction"
        ),
        "physical_output_semantics": "uncalibrated_manual_response_level",
    }


__all__ = [
    "ARRAY_CHANNEL_COORDS",
    "ARRAY_DISPLAY_ORDER",
    "ARRAY_DISPLAY_ROWS",
    "RESPONSE_DEFORMATION_PROXY",
    "dynamic_prediction_to_twin_proxy",
]
