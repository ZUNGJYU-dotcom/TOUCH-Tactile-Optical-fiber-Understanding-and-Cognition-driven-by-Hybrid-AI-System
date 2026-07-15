"""Current-session ordinal calibration for manual light/normal/hard levels."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
LEVEL_ORDER = ("light", "normal", "hard")
CORE_FEATURE_NAMES = (
    "shift_abs_mean_pm",
    "shift_abs_max_pm",
    "normalized_residual_peak",
    "normalized_residual_rms",
)


def extract_response_core_features(engineered: Mapping[str, Any]) -> dict[str, float]:
    shifts = np.asarray(
        [
            float(engineered[f"fbg{index:02d}_fused_common_mode_corrected_shift_pm"])
            for index in range(1, 10)
        ],
        dtype=float,
    )
    values = {
        "shift_abs_mean_pm": float(np.mean(np.abs(shifts))),
        "shift_abs_max_pm": float(np.max(np.abs(shifts))),
        "normalized_residual_peak": float(engineered["global_normalized_residual_peak"]),
        "normalized_residual_rms": float(engineered["global_normalized_residual_rms"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("response calibration features contain non-finite values")
    return values


class PerPositionOrdinalCalibrator:
    """Map four monotonic response features to a position-specific level score."""

    schema_version = "per_position_ordinal_response_calibration_v1"

    def __init__(
        self,
        *,
        anchors: Mapping[str, Mapping[str, Sequence[float]]],
        baseline_token: str | None,
        quality: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.anchors = {
            str(position): {
                str(feature): tuple(float(value) for value in values)
                for feature, values in feature_map.items()
            }
            for position, feature_map in anchors.items()
        }
        self.baseline_token = str(baseline_token) if baseline_token is not None else None
        self.quality = deepcopy(dict(quality or {}))

    @classmethod
    def fit(
        cls,
        samples: Iterable[Mapping[str, Any]],
        *,
        baseline_token: str | None,
        required_positions: Sequence[str] = POSITION_ORDER,
    ) -> "PerPositionOrdinalCalibrator":
        grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        for sample in samples:
            position = str(sample["position"])
            level = str(sample["level"])
            if position not in required_positions or level not in LEVEL_ORDER:
                continue
            features = sample.get("features")
            if not isinstance(features, Mapping):
                raise ValueError("calibration sample has no feature mapping")
            grouped[(position, level)].append(
                {name: float(features[name]) for name in CORE_FEATURE_NAMES}
            )

        missing = [
            f"{position}:{level}"
            for position in required_positions
            for level in LEVEL_ORDER
            if not grouped[(position, level)]
        ]
        if missing:
            raise ValueError(f"calibration is incomplete: {missing}")

        anchors: dict[str, dict[str, tuple[float, float, float]]] = {}
        quality: dict[str, dict[str, Any]] = {}
        for position in required_positions:
            feature_anchors: dict[str, tuple[float, float, float]] = {}
            raw_anchors: dict[str, list[float]] = {}
            monotonic_features: list[str] = []
            corrected_features: list[str] = []
            for feature in CORE_FEATURE_NAMES:
                raw = np.asarray(
                    [
                        np.median(
                            [row[feature] for row in grouped[(position, level)]]
                        )
                        for level in LEVEL_ORDER
                    ],
                    dtype=float,
                )
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"non-finite anchors for {position}:{feature}")
                raw_anchors[feature] = raw.tolist()
                if bool(np.all(np.diff(raw) > 0.0)):
                    monotonic_features.append(feature)
                else:
                    corrected_features.append(feature)
                ordered = np.sort(raw)
                span = max(float(ordered[-1] - ordered[0]), 1e-9)
                epsilon = span * 1e-6
                ordered[1] = max(ordered[1], ordered[0] + epsilon)
                ordered[2] = max(ordered[2], ordered[1] + epsilon)
                feature_anchors[feature] = tuple(float(value) for value in ordered)
            anchors[position] = feature_anchors
            monotonic_fraction = len(monotonic_features) / len(CORE_FEATURE_NAMES)
            quality[position] = {
                "status": (
                    "ready"
                    if monotonic_fraction == 1.0
                    else "ready_with_monotonicity_warning"
                ),
                "monotonic_feature_fraction": monotonic_fraction,
                "monotonic_features": monotonic_features,
                "corrected_features": corrected_features,
                "raw_label_order_anchors": raw_anchors,
                "sample_count_by_level": {
                    level: len(grouped[(position, level)]) for level in LEVEL_ORDER
                },
            }
        return cls(anchors=anchors, baseline_token=baseline_token, quality=quality)

    @staticmethod
    def _feature_score(value: float, anchors: Sequence[float]) -> float:
        return float(np.interp(float(value), np.asarray(anchors), [0.0, 1.0, 2.0]))

    def predict(
        self,
        position: str,
        features: Mapping[str, Any],
        *,
        baseline_token: str | None,
    ) -> dict[str, Any]:
        if self.baseline_token is not None and str(baseline_token) != self.baseline_token:
            return {
                "ok": False,
                "status": "baseline_mismatch_calibration_invalidated",
                "label": None,
                "position": position,
                "calibration_scope": "current_session_position_specific",
            }
        feature_anchors = self.anchors.get(str(position))
        if feature_anchors is None:
            return {
                "ok": False,
                "status": "position_not_calibrated",
                "label": None,
                "position": position,
                "calibration_scope": "current_session_position_specific",
            }
        feature_scores = {
            feature: self._feature_score(float(features[feature]), feature_anchors[feature])
            for feature in CORE_FEATURE_NAMES
        }
        score = float(np.mean(list(feature_scores.values())))
        level_index = int(np.digitize(score, [0.5, 1.5]))
        label = LEVEL_ORDER[level_index]
        boundary_distance = min(abs(score - 0.5), abs(score - 1.5))
        margin = min(1.0, boundary_distance / 0.5)
        position_quality = deepcopy(self.quality.get(position, {}))
        return {
            "ok": True,
            "status": "ready",
            "label": label,
            "ordinal_score": score,
            "margin": margin,
            "confidence": margin,
            "confidence_source": "ordinal_calibration_margin_not_probability",
            "feature_scores": feature_scores,
            "features": {name: float(features[name]) for name in CORE_FEATURE_NAMES},
            "position": position,
            "position_calibration_quality": position_quality,
            "calibration_scope": "current_session_position_specific",
            "force_semantics": "approximate_manual_response_level_not_force_N",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_token": self.baseline_token,
            "level_order": list(LEVEL_ORDER),
            "feature_names": list(CORE_FEATURE_NAMES),
            "anchors": {
                position: {
                    feature: list(values) for feature, values in feature_map.items()
                }
                for position, feature_map in self.anchors.items()
            },
            "quality": deepcopy(self.quality),
            "semantics": "current_session_manual_response_calibration_not_force_N",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerPositionOrdinalCalibrator":
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported calibration schema")
        return cls(
            anchors=payload["anchors"],
            baseline_token=payload.get("baseline_token"),
            quality=payload.get("quality"),
        )


__all__ = [
    "CORE_FEATURE_NAMES",
    "LEVEL_ORDER",
    "POSITION_ORDER",
    "PerPositionOrdinalCalibrator",
    "extract_response_core_features",
]
