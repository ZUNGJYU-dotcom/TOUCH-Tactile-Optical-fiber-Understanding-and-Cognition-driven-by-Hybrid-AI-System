"""FBG Bragg-wavelength tracking and shift demodulation."""

from .demodulator import (
    cross_correlation_shift_pm,
    estimate_peak_wavelengths,
    wavelength_shift_metrics,
)

__all__ = [
    "cross_correlation_shift_pm",
    "estimate_peak_wavelengths",
    "wavelength_shift_metrics",
]
