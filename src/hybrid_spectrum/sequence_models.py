"""Spectral-axis neural models and future temporal-sequence building blocks.

The current Sense exports contain one static spectrum per CSV.  Consequently,
``SmallSpectral1DCNN`` operates along the wavelength axis.  The temporal model
classes are reusable once continuous labelled frame sequences are available;
they must not be reported as temporal results for the current snapshot data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .sense_static_dataset import (
    SenseSpectrumRecord,
    StaticFeatureDataset,
    baseline_for_record,
    resample_spectrum,
)


EPSILON = 1.0e-8
SPECTRAL_CHANNEL_NAMES = (
    "current_log_shape",
    "baseline_log_ratio",
    "log_ratio_derivative",
)


@dataclass(frozen=True)
class SpectralMultiviewData:
    """Aligned multiview spectra plus source identity and baseline metadata."""

    values: np.ndarray
    wavelength_nm: np.ndarray
    channel_names: tuple[str, ...]
    file_ids: tuple[str, ...]
    baseline_reference_mode: tuple[str, ...]


@dataclass(frozen=True)
class ChannelStandardizer:
    """Training-only channel statistics for ``[N, C, L]`` tensors."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ChannelStandardizer":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 3 or array.shape[0] == 0:
            raise ValueError("values must have shape [samples, channels, wavelength]")
        mean = np.mean(array, axis=(0, 2), keepdims=True)
        scale = np.std(array, axis=(0, 2), keepdims=True)
        scale = np.where(scale > EPSILON, scale, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32
        )


def _safe_sample_standardize(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    return centered / max(scale, EPSILON)


def build_spectral_multiview_data(
    records: Iterable[SenseSpectrumRecord],
    feature_dataset: StaticFeatureDataset,
) -> SpectralMultiviewData:
    """Build three wavelength-axis views without using class labels.

    The first channel removes per-frame optical gain while retaining spectral
    shape.  The second and third channels preserve baseline-relative magnitude
    and local shape changes.  A second training-only standardization step is
    still required for each evaluation split.
    """

    source_records = tuple(records)
    if source_records != feature_dataset.records:
        raise ValueError("record order must match the static feature dataset")
    grid = np.asarray(feature_dataset.common_wavelength_nm, dtype=float)
    rows: list[np.ndarray] = []
    modes: list[str] = []
    for record in source_records:
        current = np.maximum(resample_spectrum(record, grid), 0.0)
        baseline, mode = baseline_for_record(
            record,
            feature_dataset.reference_baseline_clusters,
            strategy=feature_dataset.baseline_reference_strategy,
        )
        baseline = np.maximum(np.asarray(baseline, dtype=float), 0.0)
        current_log = np.log1p(current)
        current_log_shape = _safe_sample_standardize(current_log)
        log_ratio = current_log - np.log1p(baseline)
        log_ratio_derivative = np.gradient(log_ratio, grid)
        rows.append(
            np.stack(
                [current_log_shape, log_ratio, log_ratio_derivative],
                axis=0,
            ).astype(np.float32)
        )
        modes.append(mode)
    return SpectralMultiviewData(
        values=np.stack(rows, axis=0),
        wavelength_nm=grid,
        channel_names=SPECTRAL_CHANNEL_NAMES,
        file_ids=tuple(record.file_id for record in source_records),
        baseline_reference_mode=tuple(modes),
    )


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by optional-runtime checks
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None


if nn is not None:

    class _ConvBlock(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
            super().__init__(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            )


    class SmallSpectral1DCNN(nn.Module):
        """Compact CNN that retains coarse wavelength location information."""

        def __init__(
            self,
            in_channels: int,
            num_classes: int,
            pooled_length: int = 12,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                _ConvBlock(in_channels, 16, 11),
                _ConvBlock(16, 32, 7),
                _ConvBlock(32, 64, 5),
                nn.AdaptiveAvgPool1d(pooled_length),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * pooled_length, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.encoder(values))


    class _TemporalResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            padding = dilation * 2
            self.layers = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=5,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
            self.activation = nn.GELU()

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            return self.activation(values + self.layers(values))


    class TemporalTCN(nn.Module):
        """TCN for future ``[batch, features, time]`` acquisition windows."""

        def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(in_channels, 48, kernel_size=1)
            self.temporal = nn.Sequential(
                _TemporalResidualBlock(48, 1, dropout),
                _TemporalResidualBlock(48, 2, dropout),
                _TemporalResidualBlock(48, 4, dropout),
                _TemporalResidualBlock(48, 8, dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(48, num_classes),
            )

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.temporal(self.input_projection(values)))


    class SmallTemporal1DCNN(nn.Module):
        """Compact temporal CNN for ``[batch, frame_features, time]`` windows."""

        def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 48, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Conv1d(48, 64, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.encoder(values))


    class TemporalCNNLSTM(nn.Module):
        """CNN-LSTM for future ``[batch, features, time]`` windows."""

        def __init__(self, in_channels: int, num_classes: int, hidden_size: int = 64) -> None:
            super().__init__()
            self.frontend = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),
            )
            self.lstm = nn.LSTM(
                input_size=32,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, num_classes)

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            sequence = self.frontend(values).transpose(1, 2)
            output, _ = self.lstm(sequence)
            return self.head(output[:, -1, :])

else:

    class _TorchUnavailable:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise ImportError(
                "PyTorch is not installed. Install requirements-ml.txt in an isolated environment."
            )

    SmallSpectral1DCNN = _TorchUnavailable
    SmallTemporal1DCNN = _TorchUnavailable
    TemporalTCN = _TorchUnavailable
    TemporalCNNLSTM = _TorchUnavailable


__all__ = [
    "ChannelStandardizer",
    "SPECTRAL_CHANNEL_NAMES",
    "SmallSpectral1DCNN",
    "SmallTemporal1DCNN",
    "SpectralMultiviewData",
    "TemporalCNNLSTM",
    "TemporalTCN",
    "build_spectral_multiview_data",
    "torch_available",
]
