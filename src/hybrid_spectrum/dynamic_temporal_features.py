"""Compatibility exports for training code using the legacy module name."""

from .runtime_temporal_features import (
    SUMMARY_FEATURE_BLOCK_ORDER,
    temporal_summary_features,
)


__all__ = ["SUMMARY_FEATURE_BLOCK_ORDER", "temporal_summary_features"]
