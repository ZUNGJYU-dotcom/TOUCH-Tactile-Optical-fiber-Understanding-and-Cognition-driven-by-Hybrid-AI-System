"""Literature-guided, leakage-safe models for cross-date FBG spectra.

The module deliberately keeps every transformation causal or row-local.  No
statistics from a held-out acquisition date are used to normalize that date.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPECTRAL_BLOCK_SIZE = 64
GLOBAL_FEATURE_START = 256
POSITION_ORDER = (
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
)


@dataclass(frozen=True)
class StrictCrossDateDataset:
    features: np.ndarray
    feature_names: np.ndarray
    wavelength_nm: np.ndarray
    force_fz_n: np.ndarray
    contact_target: np.ndarray
    position_target: np.ndarray
    force_mask: np.ndarray
    contact_mask: np.ndarray
    position_mask: np.ndarray
    group_id: np.ndarray
    acquisition_date: np.ndarray
    sample_index: np.ndarray
    elapsed_time_sec: np.ndarray


@dataclass(frozen=True)
class FeatureView:
    values: np.ndarray
    names: np.ndarray


class LiteratureGuidedPositionEnsemble:
    """Average absolute-response and normalized-response position evidence.

    The wrapper deliberately accepts the strict 264-feature spectrum row.  It
    owns the two row-local transforms used during validation, which keeps the
    live adapter and the offline benchmark on one feature contract.
    """

    def __init__(
        self,
        response_model: Any,
        normalized_model: Any,
        *,
        response_weight: float = 0.5,
    ) -> None:
        weight = float(response_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("response_weight must be between zero and one")
        if not np.array_equal(response_model.classes_, normalized_model.classes_):
            raise ValueError("position ensemble class orders do not match")
        self.response_model = response_model
        self.normalized_model = normalized_model
        self.response_weight = weight
        self.classes_ = np.asarray(response_model.classes_)
        self.n_features_in_ = 264

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = _strict_spectrum_matrix(values)
        raw = self.response_model.predict_proba(response_raw_features(matrix))
        normalized = self.normalized_model.predict_proba(
            literature_snv_sg_features(matrix)
        )
        return (
            self.response_weight * np.asarray(raw, dtype=float)
            + (1.0 - self.response_weight)
            * np.asarray(normalized, dtype=float)
        )

    def predict(self, values: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(values)
        return self.classes_[np.argmax(probability, axis=1)]


class OrthogonalSignalCorrection(BaseEstimator, TransformerMixin):
    """Remove train-fitted spectral variance orthogonal to a response.

    The correction directions are learned only from the training rows supplied
    to ``fit``.  ``transform`` therefore remains valid for a held-out date and
    never reads statistics from that target date.
    """

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


def _archive_mask(payload: Mapping[str, np.ndarray], stem: str) -> np.ndarray:
    for key in (f"{stem}_training_mask", f"{stem}_mask"):
        if key in payload:
            return np.asarray(payload[key], dtype=bool)
    raise KeyError(f"missing {stem} training mask")


def load_strict_cross_date_datasets(
    datasets_by_date: Mapping[str, Path],
) -> StrictCrossDateDataset:
    """Load strict per-date datasets without changing labels or force range."""

    if len(datasets_by_date) < 2:
        raise ValueError("cross-date validation requires at least two dates")

    chunks: dict[str, list[np.ndarray]] = {
        "features": [],
        "force_fz_n": [],
        "contact_target": [],
        "position_target": [],
        "force_mask": [],
        "contact_mask": [],
        "position_mask": [],
        "group_id": [],
        "acquisition_date": [],
        "sample_index": [],
        "elapsed_time_sec": [],
    }
    reference_names: np.ndarray | None = None
    reference_wavelength: np.ndarray | None = None

    for date, path in sorted(datasets_by_date.items()):
        with np.load(Path(path), allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            names = np.asarray(payload["feature_names"], dtype=str)
            wavelength = np.asarray(payload["wavelength_nm"], dtype=float)
            if features.ndim != 2 or features.shape[1] < 264:
                raise ValueError(f"{path} is not a strict 264-feature dataset")
            if not np.all(np.isfinite(features)):
                raise ValueError(f"{path} contains non-finite spectral features")
            if reference_names is None:
                reference_names = names
                reference_wavelength = wavelength
            else:
                if not np.array_equal(reference_names, names):
                    raise ValueError("feature schemas differ across dates")
                if not np.allclose(reference_wavelength, wavelength):
                    raise ValueError("wavelength grids differ across dates")

            rows = len(features)
            session = np.asarray(payload["session_id"], dtype=str)
            chunks["features"].append(features)
            chunks["force_fz_n"].append(
                np.asarray(payload["force_fz_n"], dtype=float)
            )
            chunks["contact_target"].append(
                np.asarray(payload["contact_target"], dtype=int)
            )
            chunks["position_target"].append(
                np.asarray(payload["position_target"], dtype=str)
            )
            chunks["force_mask"].append(_archive_mask(payload, "force"))
            chunks["contact_mask"].append(_archive_mask(payload, "contact"))
            chunks["position_mask"].append(_archive_mask(payload, "position"))
            chunks["group_id"].append(
                np.asarray([f"{date}::{value}" for value in session], dtype=str)
            )
            chunks["acquisition_date"].append(
                np.full(rows, str(date), dtype=f"<U{max(8, len(str(date)))}")
            )
            chunks["sample_index"].append(
                np.asarray(payload["capture_index"], dtype=int)
            )
            chunks["elapsed_time_sec"].append(
                np.asarray(payload["elapsed_time_sec"], dtype=float)
            )

    assert reference_names is not None
    assert reference_wavelength is not None
    return StrictCrossDateDataset(
        features=np.concatenate(chunks["features"]).astype(np.float32, copy=False),
        feature_names=reference_names,
        wavelength_nm=reference_wavelength,
        force_fz_n=np.concatenate(chunks["force_fz_n"]),
        contact_target=np.concatenate(chunks["contact_target"]),
        position_target=np.concatenate(chunks["position_target"]),
        force_mask=np.concatenate(chunks["force_mask"]),
        contact_mask=np.concatenate(chunks["contact_mask"]),
        position_mask=np.concatenate(chunks["position_mask"]),
        group_id=np.concatenate(chunks["group_id"]),
        acquisition_date=np.concatenate(chunks["acquisition_date"]),
        sample_index=np.concatenate(chunks["sample_index"]),
        elapsed_time_sec=np.concatenate(chunks["elapsed_time_sec"]),
    )


def standard_normal_variate(
    values: np.ndarray,
    *,
    robust: bool = False,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Apply row-wise SNV, with a median/MAD option for transient outliers."""

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


def _block_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index + 1:03d}" for index in range(count)]


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
    """Return the validated row-local SNV/Savitzky-Golay feature view."""

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


def literature_runtime_contact_features(
    feature_history: np.ndarray,
    *,
    temporal_window_frames: int = 5,
) -> np.ndarray:
    """Build the causal 488-feature contact row used by the live runtime."""

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


def compact_spectral_signals(features: np.ndarray) -> np.ndarray:
    """Reduce four 64-bin blocks to 32 local bands plus eight globals."""

    matrix = np.asarray(features, dtype=np.float32)
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
    """Return causal mean/std/delta/slope without crossing session boundaries."""

    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("temporal signals must be 2D")
    if len(values) != len(groups) or len(values) != len(sample_index):
        raise ValueError("temporal arrays have inconsistent row counts")
    window = max(2, int(window_frames))
    result = np.zeros((len(values), values.shape[1] * 4), dtype=np.float32)

    for group in sorted(set(np.asarray(groups, dtype=str).tolist())):
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


def build_literature_feature_views(
    dataset: StrictCrossDateDataset,
    *,
    temporal_window_frames: int = 5,
) -> Mapping[str, FeatureView]:
    """Build leakage-safe spectral views motivated by spectroscopy literature."""

    matrix = _strict_spectrum_matrix(dataset.features)
    hybrid = literature_snv_sg_features(matrix)
    hybrid_names = np.asarray(
        _block_names("sg_log_ratio", 64)
        + _block_names("snv_log_ratio", 64)
        + _block_names("robust_snv_shape_delta", 64)
        + _block_names("sg_shape_delta_d1", 64)
        + _block_names("sg_shape_delta_d2", 64)
        + matrix_feature_names(dataset.feature_names, 256, 264),
        dtype=str,
    )

    compact = compact_spectral_signals(matrix)
    temporal = causal_temporal_summary(
        compact,
        dataset.group_id,
        dataset.sample_index,
        window_frames=temporal_window_frames,
    )
    temporal_names = np.asarray(
        [
            f"causal_{stat}_compact_{index + 1:02d}"
            for stat in ("mean", "std", "delta", "slope")
            for index in range(compact.shape[1])
        ],
        dtype=str,
    )

    return {
        "reference_full264": FeatureView(
            matrix,
            np.asarray(dataset.feature_names[:264], dtype=str),
        ),
        "response_raw136": FeatureView(
            response_raw_features(matrix),
            np.asarray(
                matrix_feature_names(dataset.feature_names, 0, 128)
                + matrix_feature_names(dataset.feature_names, 256, 264),
                dtype=str,
            ),
        ),
        "literature_snv_sg328": FeatureView(hybrid, hybrid_names),
        "literature_snv_sg_temporal488": FeatureView(
            np.concatenate((hybrid, temporal), axis=1).astype(
                np.float32, copy=False
            ),
            np.concatenate((hybrid_names, temporal_names)),
        ),
    }


def matrix_feature_names(names: np.ndarray, start: int, stop: int) -> list[str]:
    return np.asarray(names[start:stop], dtype=str).tolist()


def equal_group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(np.asarray(groups, dtype=str).tolist())
    weights = np.asarray([1.0 / counts[str(group)] for group in groups], dtype=float)
    return weights / max(float(np.mean(weights)), 1.0e-12)


def balanced_group_indices(
    groups: np.ndarray,
    *,
    maximum_rows_per_group: int = 160,
) -> np.ndarray:
    """Deterministically cap long sessions for estimators without sample weights."""

    selected: list[int] = []
    groups = np.asarray(groups, dtype=str)
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        if len(indices) <= maximum_rows_per_group:
            selected.extend(indices.tolist())
        else:
            positions = np.linspace(
                0, len(indices) - 1, maximum_rows_per_group, dtype=int
            )
            selected.extend(indices[positions].tolist())
    return np.asarray(sorted(selected), dtype=int)


def make_extra_trees_classifier(
    *,
    estimators: int = 128,
    minimum_leaf_samples: int = 2,
    seed: int = 42,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(estimators),
        min_samples_leaf=int(minimum_leaf_samples),
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=int(seed),
    )


def make_extra_trees_regressor(
    *,
    estimators: int = 160,
    minimum_leaf_samples: int = 2,
    seed: int = 42,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=int(estimators),
        min_samples_leaf=int(minimum_leaf_samples),
        max_features=0.8,
        n_jobs=-1,
        random_state=int(seed),
    )


def make_pls_regressor(
    feature_count: int,
    *,
    components: int = 8,
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pls",
                PLSRegression(
                    n_components=max(1, min(int(components), feature_count)),
                    scale=False,
                    max_iter=500,
                ),
            ),
        ]
    )


def make_ridge_regressor(*, alpha: float = 100.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def make_osc_ridge_regressor(
    *,
    alpha: float = 100.0,
    osc_components: int = 1,
) -> Pipeline:
    return Pipeline(
        [
            ("osc", OrthogonalSignalCorrection(n_components=osc_components)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def nonnegative_prediction(prediction: np.ndarray) -> np.ndarray:
    """Enforce physical non-negativity without an artificial upper-force cap."""

    return np.maximum(np.asarray(prediction, dtype=float).reshape(-1), 0.0)


def fit_predict_pls(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_groups: np.ndarray,
    test_features: np.ndarray,
    *,
    components: int = 8,
    maximum_rows_per_group: int = 160,
) -> tuple[Pipeline, np.ndarray]:
    selected = balanced_group_indices(
        train_groups,
        maximum_rows_per_group=maximum_rows_per_group,
    )
    model = make_pls_regressor(train_features.shape[1], components=components)
    model.fit(train_features[selected], train_target[selected])
    return model, nonnegative_prediction(model.predict(test_features))


def fit_predict_ridge(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_groups: np.ndarray,
    test_features: np.ndarray,
    *,
    alpha: float = 100.0,
    maximum_rows_per_group: int = 160,
) -> tuple[Pipeline, np.ndarray]:
    selected = balanced_group_indices(
        train_groups,
        maximum_rows_per_group=maximum_rows_per_group,
    )
    model = make_ridge_regressor(alpha=alpha)
    model.fit(train_features[selected], train_target[selected])
    return model, nonnegative_prediction(model.predict(test_features))


def fit_supervised_osc(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    *,
    components: int = 1,
) -> tuple[OrthogonalSignalCorrection, np.ndarray, np.ndarray]:
    """Fit OSC on training rows and transform train/test with that fit."""

    transformer = OrthogonalSignalCorrection(n_components=components)
    transformed_train = transformer.fit_transform(train_features, train_target)
    transformed_test = transformer.transform(test_features)
    return transformer, transformed_train, transformed_test


def fit_predict_osc_ridge(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_groups: np.ndarray,
    test_features: np.ndarray,
    *,
    alpha: float = 100.0,
    osc_components: int = 1,
    maximum_rows_per_group: int = 160,
) -> tuple[Pipeline, np.ndarray]:
    selected = balanced_group_indices(
        train_groups,
        maximum_rows_per_group=maximum_rows_per_group,
    )
    model = make_osc_ridge_regressor(
        alpha=alpha,
        osc_components=osc_components,
    )
    model.fit(train_features[selected], train_target[selected])
    return model, nonnegative_prediction(model.predict(test_features))


def learn_group_oof_blend_weight(
    *,
    tree_features: np.ndarray,
    latent_features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    folds: int = 4,
    estimators: int = 72,
    minimum_leaf_samples: int = 2,
    pls_components: int = 8,
    maximum_rows_per_group: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    """Learn a convex tree/PLS weight using training-session OOF predictions."""

    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(int(folds), len(unique_groups)))
    tree_oof = np.full(len(target), np.nan, dtype=float)
    latent_oof = np.full(len(target), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train, validation) in enumerate(
        splitter.split(tree_features, target, groups), start=1
    ):
        if set(groups[train]).intersection(groups[validation]):
            raise RuntimeError("group leakage in OOF blend training")
        tree = make_extra_trees_regressor(
            estimators=estimators,
            minimum_leaf_samples=minimum_leaf_samples,
            seed=seed + fold,
        )
        tree.fit(
            tree_features[train],
            target[train],
            sample_weight=equal_group_weights(groups[train]),
        )
        tree_oof[validation] = nonnegative_prediction(
            tree.predict(tree_features[validation])
        )
        _, latent_prediction = fit_predict_pls(
            latent_features[train],
            target[train],
            groups[train],
            latent_features[validation],
            components=pls_components,
            maximum_rows_per_group=maximum_rows_per_group,
        )
        latent_oof[validation] = latent_prediction
        fold_rows.append(
            {
                "fold": fold,
                "train_groups": int(len(np.unique(groups[train]))),
                "validation_groups": int(len(np.unique(groups[validation]))),
            }
        )

    valid = np.isfinite(tree_oof) & np.isfinite(latent_oof)
    if not np.all(valid):
        raise RuntimeError("OOF blend predictions are incomplete")
    candidates = np.linspace(0.0, 1.0, 41)
    losses = [
        mean_absolute_error(
            target,
            weight * tree_oof + (1.0 - weight) * latent_oof,
        )
        for weight in candidates
    ]
    best_index = int(np.argmin(losses))
    weight = float(candidates[best_index])
    blended = nonnegative_prediction(
        weight * tree_oof + (1.0 - weight) * latent_oof
    )
    tree_calibration = fit_affine_calibration(tree_oof, target)
    blend_calibration = fit_affine_calibration(blended, target)
    return {
        "tree_weight": weight,
        "latent_weight": 1.0 - weight,
        "oof_mae_n": float(mean_absolute_error(target, blended)),
        "oof_tree_mae_n": float(mean_absolute_error(target, tree_oof)),
        "oof_latent_mae_n": float(mean_absolute_error(target, latent_oof)),
        "tree_calibration_slope": tree_calibration[0],
        "tree_calibration_intercept_n": tree_calibration[1],
        "blend_calibration_slope": blend_calibration[0],
        "blend_calibration_intercept_n": blend_calibration[1],
        "folds": fold_rows,
    }


def fit_affine_calibration(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    minimum_slope: float = 0.5,
    maximum_slope: float = 2.0,
    maximum_absolute_intercept_n: float = 0.75,
) -> tuple[float, float]:
    """Fit target = slope * prediction + intercept on training OOF rows."""

    estimate = np.asarray(prediction, dtype=float).reshape(-1)
    reference = np.asarray(target, dtype=float).reshape(-1)
    if len(estimate) < 2 or np.std(estimate) <= 1.0e-12:
        return 1.0, 0.0
    slope, intercept = np.polyfit(estimate, reference, 1)
    slope = float(np.clip(slope, minimum_slope, maximum_slope))
    intercept = float(
        np.clip(
            intercept,
            -maximum_absolute_intercept_n,
            maximum_absolute_intercept_n,
        )
    )
    return slope, intercept


def apply_affine_calibration(
    prediction: np.ndarray,
    *,
    slope: float,
    intercept_n: float,
) -> np.ndarray:
    return nonnegative_prediction(float(slope) * prediction + float(intercept_n))


def classification_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    *,
    labels: list[Any],
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(true, predicted, labels=labels, average="macro", zero_division=0)
        ),
    }
    recalls = recall_score(
        true,
        predicted,
        labels=labels,
        average=None,
        zero_division=0,
    )
    for label, recall in zip(labels, recalls):
        metrics[f"recall_{label}"] = float(recall)
    return metrics


def group_voting_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    labels: list[str],
) -> dict[str, float]:
    group_true: list[str] = []
    group_predicted: list[str] = []
    for group in sorted(set(np.asarray(groups, dtype=str).tolist())):
        selected = groups == group
        group_true.append(Counter(true[selected].tolist()).most_common(1)[0][0])
        group_predicted.append(
            Counter(predicted[selected].tolist()).most_common(1)[0][0]
        )
    return {
        "group_voting_accuracy": float(
            accuracy_score(group_true, group_predicted)
        ),
        "group_voting_macro_f1": float(
            f1_score(
                group_true,
                group_predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def regression_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    *,
    low_force_threshold_n: float = 0.10,
) -> dict[str, float]:
    reference = np.asarray(true, dtype=float)
    estimate = nonnegative_prediction(predicted)
    residual = estimate - reference
    if len(reference) > 1 and np.std(reference) > 1.0e-12 and np.std(estimate) > 1.0e-12:
        correlation = float(np.corrcoef(reference, estimate)[0, 1])
        slope = float(np.polyfit(reference, estimate, 1)[0])
    else:
        correlation = float("nan")
        slope = float("nan")
    low_force = reference <= float(low_force_threshold_n)
    within_5n = reference <= 5.0
    above_5n = reference > 5.0
    result = {
        "mae_n": float(mean_absolute_error(reference, estimate)),
        "rmse_n": float(np.sqrt(mean_squared_error(reference, estimate))),
        "r2": float(r2_score(reference, estimate)),
        "correlation": correlation,
        "slope": slope,
        "bias_n": float(np.mean(residual)),
        "prediction_max_n": float(np.max(estimate)),
        "reference_max_n": float(np.max(reference)),
        "low_force_false_response_mean_n": float(
            np.mean(estimate[low_force]) if np.any(low_force) else np.nan
        ),
        "low_force_false_response_p95_n": float(
            np.percentile(estimate[low_force], 95) if np.any(low_force) else np.nan
        ),
    }
    if np.any(within_5n):
        result["mae_0_5n_n"] = float(
            mean_absolute_error(reference[within_5n], estimate[within_5n])
        )
    if np.any(above_5n):
        result["mae_above_5n_n"] = float(
            mean_absolute_error(reference[above_5n], estimate[above_5n])
        )
        result["rows_above_5n"] = int(np.sum(above_5n))
    else:
        result["mae_above_5n_n"] = float("nan")
        result["rows_above_5n"] = 0
    return result
