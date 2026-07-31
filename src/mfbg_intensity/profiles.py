"""Sensor-profile registry without changing the retained ordinary-FBG runtime."""

from __future__ import annotations

from typing import Any

from .config import MfbgIntensityProfile


def profile_registry(mfbg_profile: MfbgIntensityProfile) -> dict[str, Any]:
    return {
        "active_runtime_profile": "ordinary_fbg_hybrid_spectral",
        "future_primary_profile": mfbg_profile.profile_id,
        "profiles": {
            "ordinary_fbg_hybrid_spectral": {
                "sensor_family": "ordinary_fbg",
                "demodulation_mode": "hybrid_wavelength_intensity_shape",
                "status": "retained_unchanged",
                "runtime_active": True,
                "config_file": "config/hybrid_spectrum_channels.yaml",
            },
            mfbg_profile.profile_id: {
                **mfbg_profile.summary(),
                "runtime_active": False,
                "activation_gate": (
                    "measured nine-channel wavelengths and real baseline required"
                ),
            },
        },
        "profile_isolation": True,
        "ordinary_fbg_logic_modified": False,
    }
