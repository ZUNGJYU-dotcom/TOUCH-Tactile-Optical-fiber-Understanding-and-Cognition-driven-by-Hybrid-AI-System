"""mFBG 3x3 optical-intensity demodulation profile.

This package is intentionally separate from the retained ordinary-FBG
wavelength-shift and hybrid-spectrum runtime.
"""

from .config import MfbgIntensityProfile, load_profile
from .demodulator import MfbgIntensityDemodulator
from .recording import frame_to_channel_rows, frame_to_wide_row

__all__ = [
    "MfbgIntensityDemodulator",
    "MfbgIntensityProfile",
    "frame_to_channel_rows",
    "frame_to_wide_row",
    "load_profile",
]
