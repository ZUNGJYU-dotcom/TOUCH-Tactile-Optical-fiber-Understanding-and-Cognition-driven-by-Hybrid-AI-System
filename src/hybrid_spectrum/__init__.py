"""Full-spectrum data, feature and model utilities."""

from .dataset import SpectrumSegment, load_segments, save_segment
from .artifacts import load_verified_model_bundle, verify_model_artifact
from .features import PeakWindow, extract_feature_rows, load_peak_windows

__all__ = [
    "PeakWindow",
    "SpectrumSegment",
    "extract_feature_rows",
    "load_peak_windows",
    "load_verified_model_bundle",
    "load_segments",
    "save_segment",
    "verify_model_artifact",
]
