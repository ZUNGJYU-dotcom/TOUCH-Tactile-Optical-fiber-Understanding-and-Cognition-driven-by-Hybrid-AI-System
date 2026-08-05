"""Lightweight spectroscopy transforms used by the deployed runtime.

This module deliberately excludes dataset loaders, cross-validation, metrics,
and model training.  Keeping the frozen application pointed here prevents the
research pipeline from becoming a transitive desktop dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


SPECTRAL_BLOCK_SIZE = 64
GLOBAL_FEATURE_START = 256


class OrthogonalSignalCorrection(BaseEstimator, TransformerMixin):
    """Apply train-fitted orthogonal signal correction at inference time."""

    def __init__(self, n_components: int = 1, epsilon: float = 1.0e-12) -> None:
        self.n_components = int(n_components)
        self.epsilon = float(epsilon)

    def fit(
        self,
        values: np.ndarray,
        target: np.ndarray,
    ) -> "OrthogonalSignalCorrection":
        matrix = np.asarray(values, dtype=float)
        response = np.asarray(target, dtype=float).reshape(-1)
        if matrix.ndim != 2 or len(matrix) != len(response):
            raise ValueError("OSC requires a 2-D matrix aligned with the response")

        self.mean_ = np.mean(matrix, axis=0)
        residual = matrix - self.mean_
        centered_response = response - np.mean(response)
        response_energy = float(centered_response @ centered_response)
        self.orthogonal_weights_: list[np.ndarray] = []
        self.orthogonal_loadings_: list[np.ndarray] = []

        if response_energy <= self.epsilon:
            self.n_components_ = 0
            return self

        for _ in range(max(0, self.n_components)):
            predictive_weight = residual.T @ centered_response / response_energy
            predictive_energy = float(predictive_weight @ predictive_weight)
            if predictive_energy <= self.epsilon:
                break
            predictive_score = residual @ predictive_weight / predictive_energy
            score_energy = float(predictive_score @ predictive_score)
            if score_energy <= self.epsilon:
                break
            loading = residual.T @ predictive_score / score_energy
            orthogonal_weight = loading - predictive_weight * (
                float(predictive_weight @ loading) / predictive_energy
            )
            orthogonal_norm = float(np.linalg.norm(orthogonal_weight))
            if orthogonal_norm <= self.epsilon:
                break
            orthogonal_weight /= orthogonal_norm
            orthogonal_score = residual @ orthogonal_weight
            orthogonal_energy = float(orthogonal_score @ orthogonal_score)
            if orthogonal_energy <= self.epsilon:
                break
            orthogonal_loading = residual.T @ orthogonal_score / orthogonal_energy
            residual -= np.outer(orthogonal_score, orthogonal_loading)
            self.orthogonal_weights_.append(orthogonal_weight)
            self.orthogonal_loadings_.append(orthogonal_loading)

        self.n_components_ = len(self.orthogonal_weights_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("OSC transformer must be fitted before transform")
        residual = np.asarray(values, dtype=float) - self.mean_
        for weight, loading in zip(
            self.orthogonal_weights_, self.orthogonal_loadings_
        ):
            score = residual @ weight
            residual -= np.outer(score, loading)
        return residual


def standard_normal_variate(
    values: np.ndarray,
    *,
    robust: bool = False,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Apply row-wise SNV, optionally using median and MAD."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("SNV input must be a 2D matrix")
    if robust:
        center = np.median(matrix, axis=1, keepdims=True)
        scale = 1.4826 * np.median(
            np.abs(matrix - center), axis=1, keepdims=True
        )
    else:
        center = np.mean(matrix, axis=1, keepdims=True)
        scale = np.std(matrix, axis=1, keepdims=True)
    scale = np.where(scale > epsilon, scale, 1.0)
    normalized = (matrix - center) / scale
    return np.nan_to_num(normalized, copy=False).astype(np.float32, copy=False)


def _savgol(values: np.ndarray, derivative: int) -> np.ndarray:
    width = values.shape[1]
    window = min(9, width if width % 2 else width - 1)
    if window < 5:
        return np.zeros_like(values, dtype=np.float32)
    filtered = savgol_filter(
        values,
        window_length=window,
        polyorder=2,
        deriv=derivative,
        axis=1,
        mode="interp",
    )
    return np.asarray(filtered, dtype=np.float32)


def _strict_spectrum_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] < 264:
        raise ValueError("strict spectrum features must have at least 264 columns")
    matrix = matrix[:, :264]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("strict spectrum features contain non-finite values")
    return matrix


def response_raw_features(features: np.ndarray) -> np.ndarray:
    """Return the validated absolute response view (128 bins + globals)."""

    matrix = _strict_spectrum_matrix(features)
    return np.concatenate((matrix[:, :128], matrix[:, 256:264]), axis=1).astype(
        np.float32, copy=False
    )


def literature_snv_sg_features(features: np.ndarray) -> np.ndarray:
    """Return the row-local SNV and Savitzky-Golay feature view."""

    matrix = _strict_spectrum_matrix(features)
    log_ratio = matrix[:, 0:64]
    shape_delta = matrix[:, 64:128]
    globals_ = matrix[:, 256:264]
    return np.concatenate(
        (
            _savgol(log_ratio, 0),
            standard_normal_variate(log_ratio),
            standard_normal_variate(shape_delta, robust=True),
            _savgol(shape_delta, 1),
            _savgol(shape_delta, 2),
            globals_,
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def compact_spectral_signals(features: np.ndarray) -> np.ndarray:
    """Reduce four 64-bin blocks to 32 local bands plus eight globals."""

    matrix = _strict_spectrum_matrix(features)
    blocks: list[np.ndarray] = []
    for start in range(0, GLOBAL_FEATURE_START, SPECTRAL_BLOCK_SIZE):
        block = matrix[:, start : start + SPECTRAL_BLOCK_SIZE]
        blocks.append(block.reshape(len(matrix), 8, 8).mean(axis=2))
    blocks.append(matrix[:, GLOBAL_FEATURE_START:264])
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def causal_temporal_summary(
    signals: np.ndarray,
    groups: np.ndarray,
    sample_index: np.ndarray,
    *,
    window_frames: int = 5,
) -> np.ndarray:
    """Return causal mean, std, delta, and slope within each session."""

    values = np.asarray(signals, dtype=np.float32)
    groups = np.asarray(groups, dtype=str)
    sample_index = np.asarray(sample_index, dtype=int)
    if values.ndim != 2:
        raise ValueError("temporal signals must be 2D")
    if len(values) != len(groups) or len(values) != len(sample_index):
        raise ValueError("temporal arrays have inconsistent row counts")
    window = max(2, int(window_frames))
    result = np.zeros((len(values), values.shape[1] * 4), dtype=np.float32)

    for group in sorted(set(groups.tolist())):
        selected = np.flatnonzero(groups == group)
        ordered = selected[np.argsort(sample_index[selected], kind="stable")]
        sequence = values[ordered]
        for position, row_index in enumerate(ordered):
            start = max(0, position - window + 1)
            segment = sequence[start : position + 1]
            mean = np.mean(segment, axis=0)
            std = np.std(segment, axis=0)
            delta = (
                sequence[position] - sequence[position - 1]
                if position > 0
                else np.zeros(values.shape[1], dtype=np.float32)
            )
            if len(segment) > 1:
                x = np.arange(len(segment), dtype=np.float32)
                x -= np.mean(x)
                denominator = float(np.dot(x, x))
                slope = np.dot(x, segment - mean) / max(denominator, 1.0e-8)
            else:
                slope = np.zeros(values.shape[1], dtype=np.float32)
            result[row_index] = np.concatenate((mean, std, delta, slope))
    return result


def literature_runtime_contact_features(
    feature_history: np.ndarray,
    *,
    temporal_window_frames: int = 5,
) -> np.ndarray:
    """Build the causal 488-feature contact row used by live inference."""

    matrix = _strict_spectrum_matrix(feature_history)
    window = max(2, int(temporal_window_frames))
    matrix = matrix[-window:]
    hybrid = literature_snv_sg_features(matrix[-1:])
    compact = compact_spectral_signals(matrix)
    temporal = causal_temporal_summary(
        compact,
        np.full(len(compact), "runtime", dtype="<U7"),
        np.arange(len(compact), dtype=int),
        window_frames=window,
    )[-1:]
    return np.concatenate((hybrid, temporal), axis=1).astype(
        np.float32, copy=False
    )
