"""Runtime adapter for the deployed optical-only ordinary-FBG model."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import warnings

import joblib
import numpy as np

from .baseline_relative_features import extract_baseline_relative_features
from .features import PeakWindow, load_peak_windows
from .joint_nine_fbg_features import extract_joint_nine_fbg_features
from .runtime_literature_features import (
    literature_runtime_contact_features,
    literature_snv_sg_features,
    response_raw_features,
)
from .runtime_baseline_guard import RuntimeBaselineRecoveryGuard
from .runtime_channel_response import (
    build_inferred_contact_probability_surface,
    build_observed_coupled_spectral_response,
    summarize_coupled_contact_signature,
)
from .runtime_spectral_features import (
    baseline_relative_frame_feature_names,
    extract_baseline_relative_frame_features,
)
from .runtime_temporal_features import temporal_summary_features


MODEL_SCHEMA = "ordinary_fbg_optical_only_force_candidate_v2"
FEATURE_SCHEMA = "nine_peak_shift_intensity_shape_temporal_summary_483"
JOINT_NINE_FBG_RUNTIME_SCHEMA = "same_day_joint_nine_fbg_v4"
POSITION_ORDER = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)
DISPLAY_ROWS = (
    ("P11", "P21", "P31"),
    ("P12", "P22", "P32"),
    ("P13", "P23", "P33"),
)
POSITION_COORDINATES = {
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


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=float)
    if vector.ndim != 1 or vector.size < 5:
        raise ValueError(f"{name} must be a one-dimensional spectrum")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _class_probabilities(model: Any, features: np.ndarray) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but .* was fitted with "
                "feature names"
            ),
            category=UserWarning,
        )
        probabilities = np.asarray(model.predict_proba(features), dtype=float)[0]
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("classifier returned non-finite probabilities")
    classes = tuple(str(value) for value in model.classes_)
    return {
        label: float(np.clip(probability, 0.0, 1.0))
        for label, probability in zip(classes, probabilities, strict=True)
    }


def _set_single_thread(model: Any) -> None:
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    for child_name in ("response_model", "normalized_model"):
        child = getattr(model, child_name, None)
        if child is not None and child is not model:
            _set_single_thread(child)


def _model_input_feature_count(model: Any) -> int:
    direct = getattr(model, "n_features_in_", None)
    if direct is not None:
        return int(direct)
    # A Pipeline may begin with a custom transformer that intentionally does
    # not expose n_features_in_. A fitted downstream sklearn step still carries
    # the exact input contract for the complete pipeline.
    for _, step in getattr(model, "steps", ()):
        count = getattr(step, "n_features_in_", None)
        if count is not None:
            return int(count)
    return -1


def _model_predict(model: Any, features: np.ndarray) -> np.ndarray:
    """Predict from the runtime NumPy contract without sklearn name chatter."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but .* was fitted with "
                "feature names"
            ),
            category=UserWarning,
        )
        return np.asarray(model.predict(features))


def _top_probability(
    probabilities: dict[str, float],
) -> tuple[str | None, float, float]:
    if not probabilities:
        return None, 0.0, 0.0
    ordered = sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    )
    label, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return label, float(confidence), float(confidence - runner_up)


class AllSourceOpticalForceAdapter:
    """Run the grouped-evaluated all-source model from optical spectra only."""

    def __init__(
        self,
        bundle: dict[str, Any],
        peak_windows: Iterable[PeakWindow],
        *,
        window_seconds: float = 2.0,
        runtime_recovery_config: dict[str, Any] | None = None,
        runtime_gate_config: dict[str, Any] | None = None,
    ) -> None:
        if bundle.get("schema_version") != MODEL_SCHEMA:
            raise ValueError("unsupported all-source candidate model schema")
        if bundle.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("unexpected all-source feature schema")
        if float(window_seconds) <= 0:
            raise ValueError("window_seconds must be positive")

        self.bundle = bundle
        self.peak_windows = tuple(peak_windows)
        if len(self.peak_windows) != 9:
            raise ValueError("all-source runtime requires nine peak windows")
        self.window_seconds = float(window_seconds)
        self.runtime_frame_feature_names = baseline_relative_frame_feature_names(
            self.peak_windows
        )

        isolation = bundle.get("runtime_model_isolation")
        self.current_only_runtime = bool(
            isinstance(isolation, Mapping)
            and isolation.get("legacy_fallback_enabled") is False
        )
        self.all_feature_names: tuple[str, ...] = ()
        self.contact_task: Mapping[str, Any] | None = None
        self.position_task: Mapping[str, Any] | None = None
        self.force_task: Mapping[str, Any] | None = None
        self.contact_model: Any = None
        self.position_model: Any = None
        self.force_model: Any = None
        self.contact_indices = np.asarray([], dtype=int)
        self.position_indices = np.asarray([], dtype=int)
        self.force_indices = np.asarray([], dtype=int)

        if self.current_only_runtime:
            payload = bundle.get("literature_guided_classification")
            if not isinstance(payload, Mapping):
                raise ValueError("current-only runtime requires its active payload")
            if payload.get("schema_version") != JOINT_NINE_FBG_RUNTIME_SCHEMA:
                raise ValueError("current-only runtime schema is not supported")
            if "tasks" in bundle or "task_metadata" in bundle:
                raise ValueError("current-only runtime contains legacy task models")
            if "all_feature_names" in bundle:
                raise ValueError("current-only runtime contains a legacy feature contract")
        else:
            self.all_feature_names = tuple(
                str(value) for value in bundle["all_feature_names"]
            )
            if len(self.all_feature_names) != 483:
                raise ValueError("all-source runtime expects 483 temporal features")

            tasks = bundle.get("tasks", {})
            self.contact_task = tasks["contact"]
            self.position_task = tasks["position"]
            self.force_task = tasks["force_fz"]
            self.contact_model = self.contact_task["model"]
            self.position_model = self.position_task["model"]
            self.force_model = self.force_task["model"]
            for model in (self.contact_model, self.position_model, self.force_model):
                _set_single_thread(model)

            self.contact_indices = np.asarray(
                self.contact_task["feature_indices"], dtype=int
            )
            self.position_indices = np.asarray(
                self.position_task["feature_indices"], dtype=int
            )
            self.force_indices = np.asarray(
                self.force_task["feature_indices"], dtype=int
            )
            self._validate_contract()

        self.classification_model_source = "legacy_all_source_temporal_v1"
        self.force_model_source = "legacy_all_source_current_frame_v1"
        self.literature_classification_enabled = False
        self.literature_force_enabled = False
        self.literature_static_feature_names: tuple[str, ...] = ()
        self.literature_bin_count = 64
        self.literature_temporal_window_frames = 5
        self.literature_contact_feature_view = "legacy_temporal483"
        self.literature_position_feature_view = "legacy_temporal483"
        self.literature_force_feature_view = "legacy_current40"
        self.literature_contact_feature_indices = np.asarray([], dtype=int)
        self.literature_position_feature_indices = np.asarray([], dtype=int)
        self.literature_force_feature_indices = np.asarray([], dtype=int)
        self.literature_runtime_schema_version = ""
        self.literature_runtime_decision_contract: dict[str, Any] = {}
        self._configure_literature_guided_classification(
            bundle.get("literature_guided_classification")
        )
        if self.current_only_runtime and not self.literature_classification_enabled:
            raise ValueError("current-only runtime did not activate its model heads")

        force_contract = bundle.get("force_calibration_contract", {})
        gate = force_contract.get("optical_contact_gate", {})
        force_range = force_contract.get(
            "training_range_n",
            force_contract.get("prediction_clip_range_n", (0.0, 5.0)),
        )
        self.contact_threshold = float(gate.get("probability_threshold", 0.75))
        self.no_contact_force_n = float(gate.get("no_contact_output_n", 0.0))
        self.force_min_n = float(force_range[0])
        self.force_calibrated_max_n = float(force_range[1])

        gate_config = dict(runtime_gate_config or {})
        decision_config = self.literature_runtime_decision_contract
        self.runtime_gate_enabled = bool(gate_config.get("enabled", True))
        self.contact_off_threshold = float(
            decision_config.get(
                "contact_probability_off",
                gate_config.get("contact_probability_off", 0.55),
            )
        )
        self.position_confidence_min = float(
            decision_config.get(
                "position_switch_probability",
                gate_config.get("position_confidence_min", 0.45),
            )
        )
        self.position_margin_min = float(
            decision_config.get(
                "position_margin_min",
                gate_config.get("position_margin_min", 0.08),
            )
        )
        self.joint_fast_contact_probability_on = float(
            decision_config.get("joint_fast_contact_probability_on", 0.90)
        )
        self.joint_fast_position_confidence_min = float(
            decision_config.get("joint_fast_position_confidence_min", 0.85)
        )
        self.joint_fast_position_margin_min = float(
            decision_config.get("joint_fast_position_margin_min", 0.75)
        )
        self.joint_fast_minimum_baseline_distance = float(
            decision_config.get("joint_fast_minimum_baseline_distance", 0.006)
        )
        self.model_consensus_contact_probability_on = float(
            gate_config.get(
                "model_consensus_contact_probability_on",
                0.70,
            )
        )
        self.model_consensus_minimum_baseline_distance = float(
            gate_config.get(
                "model_consensus_minimum_baseline_distance",
                0.0025,
            )
        )
        self.model_consensus_contact_arm_frames = int(
            gate_config.get("model_consensus_contact_arm_frames", 2)
        )
        self.coupled_signature_enabled = bool(
            gate_config.get("coupled_signature_enabled", False)
        )
        self.coupled_signature_low_response_threshold = float(
            gate_config.get("coupled_signature_low_response_threshold", 0.005)
        )
        self.coupled_signature_nominal_response_threshold = float(
            gate_config.get(
                "coupled_signature_nominal_response_threshold", 0.010
            )
        )
        self.coupled_signature_minimum_low_response_channels = int(
            gate_config.get(
                "coupled_signature_minimum_low_response_channels", 5
            )
        )
        self.coupled_signature_minimum_nominal_response_channels = int(
            gate_config.get(
                "coupled_signature_minimum_nominal_response_channels", 3
            )
        )
        self.coupled_signature_model_contact_probability = float(
            gate_config.get(
                "coupled_signature_model_contact_probability", 0.97
            )
        )
        self.coupled_signature_model_position_confidence = float(
            gate_config.get(
                "coupled_signature_model_position_confidence", 0.45
            )
        )
        self.coupled_signature_model_position_margin = float(
            gate_config.get("coupled_signature_model_position_margin", 0.08)
        )
        self.visual_position_fallback_enabled = bool(
            gate_config.get("visual_position_fallback_enabled", True)
        )
        self.visual_position_confidence_min = float(
            gate_config.get("visual_position_confidence_min", 0.18)
        )
        self.visual_position_margin_min = float(
            gate_config.get("visual_position_margin_min", 0.015)
        )
        self.visual_position_confirm_frames = int(
            gate_config.get("visual_position_confirm_frames", 2)
        )
        self.visual_position_settle_sec = float(
            gate_config.get("visual_position_settle_sec", 0.75)
        )
        self.visual_position_switch_confidence_min = float(
            gate_config.get("visual_position_switch_confidence_min", 0.55)
        )
        self.visual_position_switch_margin_min = float(
            gate_config.get("visual_position_switch_margin_min", 0.15)
        )
        self.visual_position_switch_frames = int(
            gate_config.get("visual_position_switch_frames", 8)
        )
        self.visual_contact_probability_on = float(
            gate_config.get("visual_contact_probability_on", 0.35)
        )
        self.visual_contact_probability_off = float(
            gate_config.get("visual_contact_probability_off", 0.20)
        )
        self.visual_force_on_n = float(
            gate_config.get("visual_force_on_n", 0.08)
        )
        self.visual_force_off_n = float(
            gate_config.get("visual_force_off_n", 0.03)
        )
        self.visual_force_full_scale_n = float(
            gate_config.get("visual_force_full_scale_n", 2.5)
        )
        self.visual_force_gamma = float(
            gate_config.get("visual_force_gamma", 0.55)
        )
        self.visual_deformation_floor = float(
            gate_config.get("visual_deformation_floor", 0.12)
        )
        self.visual_contact_arm_frames = int(
            gate_config.get("visual_contact_arm_frames", 2)
        )
        self.position_ema_alpha = float(
            decision_config.get(
                "position_probability_ema_alpha",
                gate_config.get("position_probability_ema_alpha", 0.55),
            )
        )
        self.position_hold_sec = float(
            gate_config.get("position_hold_sec", 0.75)
        )
        self.position_switch_frames = int(
            decision_config.get(
                "position_switch_confirmation_frames",
                gate_config.get("position_switch_frames", 2),
            )
        )
        self.release_ambiguous_frames = int(
            gate_config.get("release_ambiguous_frames", 2)
        )
        self.ambiguous_quiet_release_sec = float(
            gate_config.get("ambiguous_quiet_release_sec", 1.0)
        )
        self.release_near_baseline_frames = int(
            gate_config.get("release_near_baseline_frames", 2)
        )
        self.baseline_release_distance = float(
            gate_config.get("baseline_release_distance", 0.0050)
        )
        self.require_baseline_separation = bool(
            gate_config.get("require_baseline_separation", True)
        )
        self.activity_memory_sec = float(
            gate_config.get("activity_memory_sec", 1.25)
        )
        self.activity_shape_motion_rms = float(
            gate_config.get("activity_shape_motion_rms", 0.0060)
        )
        self.activity_common_gain_motion = float(
            gate_config.get("activity_common_gain_motion", 0.0060)
        )
        if not 0.0 <= self.contact_off_threshold <= self.contact_threshold:
            raise ValueError("runtime contact off threshold is invalid")
        if not 0.0 <= self.position_confidence_min <= 1.0:
            raise ValueError("runtime position confidence threshold is invalid")
        if not 0.0 <= self.position_margin_min <= 1.0:
            raise ValueError("runtime position margin threshold is invalid")
        if not 0.0 <= self.model_consensus_contact_probability_on <= 1.0:
            raise ValueError(
                "model consensus contact probability threshold is invalid"
            )
        if self.model_consensus_minimum_baseline_distance < 0.0:
            raise ValueError(
                "model consensus baseline-distance threshold is invalid"
            )
        if self.model_consensus_contact_arm_frames < 1:
            raise ValueError(
                "model consensus contact confirmation frame count is invalid"
            )
        if not (
            0.0
            <= self.coupled_signature_low_response_threshold
            <= self.coupled_signature_nominal_response_threshold
        ):
            raise ValueError("coupled signature response thresholds are invalid")
        if not (
            1
            <= self.coupled_signature_minimum_nominal_response_channels
            <= self.coupled_signature_minimum_low_response_channels
            <= len(self.peak_windows)
        ):
            raise ValueError("coupled signature channel counts are invalid")
        if not 0.0 <= self.coupled_signature_model_contact_probability <= 1.0:
            raise ValueError("coupled signature contact probability is invalid")
        if not 0.0 <= self.coupled_signature_model_position_confidence <= 1.0:
            raise ValueError("coupled signature position confidence is invalid")
        if not 0.0 <= self.coupled_signature_model_position_margin <= 1.0:
            raise ValueError("coupled signature position margin is invalid")
        if not 0.0 <= self.joint_fast_contact_probability_on <= 1.0:
            raise ValueError("joint fast contact probability is invalid")
        if not 0.0 <= self.joint_fast_position_confidence_min <= 1.0:
            raise ValueError("joint fast position confidence is invalid")
        if not 0.0 <= self.joint_fast_position_margin_min <= 1.0:
            raise ValueError("joint fast position margin is invalid")
        if self.joint_fast_minimum_baseline_distance < 0.0:
            raise ValueError("joint fast baseline distance is invalid")
        if not 0.0 <= self.visual_position_confidence_min <= 1.0:
            raise ValueError("visual position confidence threshold is invalid")
        if not 0.0 <= self.visual_position_margin_min <= 1.0:
            raise ValueError("visual position margin threshold is invalid")
        if self.visual_position_confirm_frames < 1:
            raise ValueError("visual position confirmation frame count is invalid")
        if (
            self.visual_position_settle_sec < 0.0
            or not 0.0
            <= self.visual_position_switch_confidence_min
            <= 1.0
            or not 0.0 <= self.visual_position_switch_margin_min <= 1.0
            or self.visual_position_switch_frames < 1
        ):
            raise ValueError("visual position episode-lock configuration is invalid")
        if not (
            0.0
            <= self.visual_contact_probability_off
            <= self.visual_contact_probability_on
            <= self.contact_threshold
        ):
            raise ValueError("visual contact probability thresholds are invalid")
        if not (
            self.force_min_n
            <= self.visual_force_off_n
            <= self.visual_force_on_n
            < self.visual_force_full_scale_n
        ):
            raise ValueError("visual force thresholds are invalid")
        if self.force_calibrated_max_n <= self.force_min_n:
            raise ValueError("calibrated force range is invalid")
        if not 0.0 < self.visual_force_gamma <= 1.0:
            raise ValueError("visual force gamma is invalid")
        if not 0.0 <= self.visual_deformation_floor < 1.0:
            raise ValueError("visual deformation floor is invalid")
        if self.visual_contact_arm_frames < 1:
            raise ValueError("visual contact arm frame count is invalid")
        if not 0.0 < self.position_ema_alpha <= 1.0:
            raise ValueError("runtime position EMA alpha is invalid")
        if self.position_switch_frames < 1:
            raise ValueError("runtime position switch frame count is invalid")
        if (
            self.release_ambiguous_frames < 1
            or self.ambiguous_quiet_release_sec < 0.0
            or self.release_near_baseline_frames < 1
            or self.baseline_release_distance < 0.0
        ):
            raise ValueError("runtime release configuration is invalid")

        recovery_config = dict(runtime_recovery_config or {})
        self.allow_model_baseline_reanchor = bool(
            recovery_config.get("allow_model_baseline_reanchor", False)
        )
        for key in (
            "contact_arm_physical_frames",
            "max_position_confidence",
            "quiet_hold_sec",
            "stationary_rest_candidate_delay_sec",
        ):
            if key in gate_config:
                recovery_config[key] = gate_config[key]
        self._runtime_baseline_recovery = RuntimeBaselineRecoveryGuard(
            recovery_config
        )

        self.baseline_wavelength_nm: np.ndarray | None = None
        self.baseline_intensity: np.ndarray | None = None
        self.frame_features: deque[np.ndarray] = deque()
        self.frame_times_sec: deque[float] = deque()
        self.spectrum_feature_history: deque[np.ndarray] = deque(
            maxlen=self.literature_temporal_window_frames
        )
        self._contact_latched = False
        self._model_consensus_contact_frames = 0
        self._visual_contact_latched = False
        self._visual_activation_frames = 0
        self._ambiguous_quiet_frames = 0
        self._ambiguous_quiet_started_at_sec: float | None = None
        self._quiet_no_contact_hint = False
        self._near_baseline_quiet_frames = 0
        self._last_activity_timestamp_sec: float | None = None
        self._position_probability_ema: dict[str, float] = {}
        self._visual_position_probability_ema: dict[str, float] = {}
        self._stable_position_id: str | None = None
        self._stable_position_timestamp_sec: float | None = None
        self._pending_position_id: str | None = None
        self._pending_position_frames = 0
        self._provisional_visual_position_id: str | None = None
        self._visual_position_locked_at_sec: float | None = None
        self._pending_visual_position_id: str | None = None
        self._pending_visual_position_frames = 0
        self._pending_runtime_baseline_update: dict[str, Any] | None = None

    @classmethod
    def from_paths(
        cls,
        model_path: Path,
        peak_config_path: Path,
        *,
        window_seconds: float = 2.0,
        runtime_recovery_config: dict[str, Any] | None = None,
        runtime_gate_config: dict[str, Any] | None = None,
        expected_runtime_schema: str | None = None,
        expected_dataset_id: str | None = None,
        require_current_only: bool = False,
    ) -> "AllSourceOpticalForceAdapter":
        bundle = joblib.load(Path(model_path))
        if not isinstance(bundle, dict):
            raise TypeError("all-source candidate artifact must contain a mapping")
        payload = bundle.get("literature_guided_classification")
        payload_schema = (
            str(payload.get("schema_version") or "")
            if isinstance(payload, Mapping)
            else ""
        )
        payload_dataset = (
            str(payload.get("dataset_id") or "")
            if isinstance(payload, Mapping)
            else ""
        )
        if expected_runtime_schema and payload_schema != expected_runtime_schema:
            raise ValueError("deployed runtime schema does not match this edition")
        if expected_dataset_id and payload_dataset != expected_dataset_id:
            raise ValueError("deployed runtime dataset does not match this edition")
        isolation = bundle.get("runtime_model_isolation")
        if require_current_only and not (
            isinstance(isolation, Mapping)
            and isolation.get("legacy_fallback_enabled") is False
            and isolation.get("active_runtime_schema") == expected_runtime_schema
            and isolation.get("active_dataset_id") == expected_dataset_id
        ):
            raise ValueError("deployed runtime is not an isolated current-only bundle")
        adapter = cls(
            bundle,
            load_peak_windows(Path(peak_config_path)),
            window_seconds=window_seconds,
            runtime_recovery_config=runtime_recovery_config,
            runtime_gate_config=runtime_gate_config,
        )
        if require_current_only and not adapter.current_only_runtime:
            raise ValueError("deployed runtime unexpectedly enabled a legacy fallback")
        return adapter

    def _validate_contract(self) -> None:
        if any(
            task is None
            for task in (self.contact_task, self.position_task, self.force_task)
        ):
            raise ValueError("legacy runtime task contract is incomplete")
        all_names = np.asarray(self.all_feature_names, dtype=object)
        for task_name, task, indices, expected_count in (
            ("contact", self.contact_task, self.contact_indices, 483),
            ("position", self.position_task, self.position_indices, 483),
            ("force_fz", self.force_task, self.force_indices, 40),
        ):
            if len(indices) != expected_count:
                raise ValueError(f"{task_name} feature count is invalid")
            if np.any(indices < 0) or np.any(indices >= len(all_names)):
                raise ValueError(f"{task_name} feature indices are out of range")
            expected_names = tuple(str(value) for value in task["feature_names"])
            selected_names = tuple(str(value) for value in all_names[indices])
            if expected_names != selected_names:
                raise ValueError(f"{task_name} feature-name contract mismatch")
            model_count = _model_input_feature_count(task["model"])
            if model_count != expected_count:
                raise ValueError(f"{task_name} model feature count mismatch")
        if not all(
            self.all_feature_names[index].startswith("last__")
            for index in self.force_indices
        ):
            raise ValueError("force model must use current-frame last__ features")
        expected_frame_names = tuple(
            name.removeprefix("last__")
            for name in self.force_task["feature_names"]
        )
        if expected_frame_names != self.runtime_frame_feature_names:
            raise ValueError("legacy live frame feature-name contract mismatch")

    def _configure_literature_guided_classification(
        self,
        payload: Any,
    ) -> None:
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise TypeError("literature-guided classification payload must be a mapping")
        schema_version = str(payload.get("schema_version") or "")
        if schema_version not in {
            "literature_guided_contact_position_v1",
            "literature_guided_contact_position_force_v2",
            "high_sensitivity_300us_baseline_relative_v3",
            JOINT_NINE_FBG_RUNTIME_SCHEMA,
        }:
            raise ValueError("unsupported literature-guided classification schema")

        names = tuple(str(value) for value in payload["static_feature_names"])
        expected_static_count = (
            339 if schema_version == JOINT_NINE_FBG_RUNTIME_SCHEMA else 264
        )
        if len(names) != expected_static_count:
            raise ValueError(
                "literature-guided static feature contract must contain "
                f"{expected_static_count} names"
            )
        bin_count = int(payload.get("bin_count", 64))
        temporal_frames = int(payload.get("temporal_window_frames", 5))
        if bin_count != 64 or temporal_frames < 2:
            raise ValueError("invalid literature-guided runtime feature configuration")

        contact_payload = payload["contact"]
        position_payload = payload["position"]
        contact_model = contact_payload["model"]
        position_model = position_payload["model"]
        contact_view = str(contact_payload.get("feature_view") or "")
        position_view = str(position_payload.get("feature_view") or "")

        if schema_version == JOINT_NINE_FBG_RUNTIME_SCHEMA:
            expected_contact_view = (
                "baseline_relative_264_plus_nine_fbg_joint_339"
            )
            expected_position_view = "nine_fbg_joint_75"
            contact_indices = np.asarray(
                contact_payload.get("feature_indices"), dtype=int
            )
            position_indices = np.asarray(
                position_payload.get("feature_indices"), dtype=int
            )
            expected_contact_count = len(contact_indices)
            expected_position_count = len(position_indices)
        elif schema_version == "literature_guided_contact_position_v1":
            expected_contact_view = "literature_snv_sg_temporal488"
            expected_position_view = "reference_full264"
            expected_contact_count = 488
            expected_position_count = 264
        elif schema_version == "literature_guided_contact_position_force_v2":
            expected_contact_view = "response_raw136"
            expected_position_view = "response_raw136"
            expected_contact_count = 136
            expected_position_count = 136
        else:
            expected_contact_view = "baseline_relative_264"
            expected_position_view = "baseline_relative_264"
            expected_contact_count = 264
            expected_position_count = 264
        if contact_view not in {expected_contact_view, ""}:
            raise ValueError("literature-guided contact feature view is invalid")
        if position_view not in {
            expected_position_view,
            "response_raw136+literature_snv_sg328",
            "",
        }:
            raise ValueError("literature-guided position feature view is invalid")
        if schema_version == JOINT_NINE_FBG_RUNTIME_SCHEMA:
            for task_name, task_payload, indices in (
                ("contact", contact_payload, contact_indices),
                ("position", position_payload, position_indices),
            ):
                if indices.ndim != 1 or len(indices) == 0:
                    raise ValueError(
                        f"joint runtime {task_name} feature indices are invalid"
                    )
                if np.any(indices < 0) or np.any(indices >= len(names)):
                    raise ValueError(
                        f"joint runtime {task_name} feature indices are out of range"
                    )
                expected_names = tuple(
                    str(value) for value in task_payload["feature_names"]
                )
                selected_names = tuple(names[index] for index in indices)
                if expected_names != selected_names:
                    raise ValueError(
                        f"joint runtime {task_name} feature-name contract mismatch"
                    )
        if _model_input_feature_count(contact_model) != expected_contact_count:
            raise ValueError(
                "literature-guided contact model feature count is invalid"
            )
        if _model_input_feature_count(position_model) != expected_position_count:
            raise ValueError(
                "literature-guided position model feature count is invalid"
            )

        self.contact_model = contact_model
        self.position_model = position_model
        _set_single_thread(self.contact_model)
        _set_single_thread(self.position_model)
        self.literature_static_feature_names = names
        self.literature_bin_count = bin_count
        self.literature_temporal_window_frames = temporal_frames
        self.literature_contact_feature_view = expected_contact_view
        self.literature_position_feature_view = expected_position_view
        self.literature_runtime_schema_version = schema_version
        self.literature_runtime_decision_contract = dict(
            payload.get("runtime_decision_contract") or {}
        )
        if schema_version == JOINT_NINE_FBG_RUNTIME_SCHEMA:
            self.literature_contact_feature_indices = contact_indices
            self.literature_position_feature_indices = position_indices
        if schema_version in {
            "literature_guided_contact_position_force_v2",
            "high_sensitivity_300us_baseline_relative_v3",
            JOINT_NINE_FBG_RUNTIME_SCHEMA,
        }:
            force_payload = payload.get("force_fz")
            if not isinstance(force_payload, Mapping):
                raise ValueError("optical runtime payload requires force_fz")
            force_model = force_payload["model"]
            force_view = str(force_payload.get("feature_view") or "")
            if schema_version == JOINT_NINE_FBG_RUNTIME_SCHEMA:
                expected_force_view = (
                    "baseline_relative_264_plus_nine_fbg_joint_339"
                )
                force_indices = np.asarray(
                    force_payload.get("feature_indices"), dtype=int
                )
                if force_indices.ndim != 1 or len(force_indices) == 0:
                    raise ValueError("joint runtime force feature indices are invalid")
                if np.any(force_indices < 0) or np.any(
                    force_indices >= len(names)
                ):
                    raise ValueError(
                        "joint runtime force feature indices are out of range"
                    )
                expected_force_names = tuple(
                    str(value) for value in force_payload["feature_names"]
                )
                if expected_force_names != tuple(
                    names[index] for index in force_indices
                ):
                    raise ValueError(
                        "joint runtime force feature-name contract mismatch"
                    )
                expected_force_count = len(force_indices)
            else:
                expected_force_view = (
                    "baseline_relative_264"
                    if schema_version
                    == "high_sensitivity_300us_baseline_relative_v3"
                    else "literature_snv_sg328"
                )
                expected_force_count = (
                    264
                    if schema_version
                    == "high_sensitivity_300us_baseline_relative_v3"
                    else 328
                )
            if force_view != expected_force_view:
                raise ValueError("optical force feature view is invalid")
            if _model_input_feature_count(force_model) != expected_force_count:
                raise ValueError(
                    "optical force model feature count is invalid"
                )
            self.force_model = force_model
            _set_single_thread(self.force_model)
            self.literature_force_feature_view = force_view
            self.literature_force_enabled = True
            if schema_version == JOINT_NINE_FBG_RUNTIME_SCHEMA:
                self.literature_force_feature_indices = force_indices
                self.force_model_source = "same_day_joint_nine_fbg_force_v4"
                self.classification_model_source = (
                    "same_day_joint_nine_fbg_contact_position_v4"
                )
            elif schema_version == "high_sensitivity_300us_baseline_relative_v3":
                self.force_model_source = (
                    "high_sensitivity_300us_baseline_relative_extra_trees_v3"
                )
                self.classification_model_source = (
                    "high_sensitivity_300us_baseline_relative_contact_position_v3"
                )
            else:
                self.force_model_source = "literature_guided_three_date_osc_ridge_v1"
                self.classification_model_source = (
                    "literature_guided_three_date_response_raw_v2"
                )
        else:
            self.classification_model_source = "literature_guided_cross_date_v1"
        self.literature_classification_enabled = True

    def clear(self) -> None:
        self.baseline_wavelength_nm = None
        self.baseline_intensity = None
        self._pending_runtime_baseline_update = None
        self._runtime_baseline_recovery.reset()
        self._reset_decision_state()
        self._clear_temporal_history()

    def _clear_temporal_history(self) -> None:
        self.frame_features.clear()
        self.frame_times_sec.clear()
        self.spectrum_feature_history.clear()

    def clear_history(self, *, confirmed_rest: bool = False) -> None:
        self._clear_temporal_history()
        self._pending_runtime_baseline_update = None
        self._runtime_baseline_recovery.reset()
        if self.baseline_intensity is not None:
            self._runtime_baseline_recovery.prime_physical_spectrum(
                self.baseline_intensity
            )
        if confirmed_rest:
            self._runtime_baseline_recovery.confirm_rest()
        self._reset_decision_state()

    def consume_pending_runtime_baseline_update(self) -> dict[str, Any] | None:
        """Return a one-shot baseline candidate for transactional bridge commit."""

        pending = self._pending_runtime_baseline_update
        self._pending_runtime_baseline_update = None
        if pending is None:
            return None
        copied = dict(pending)
        copied["wavelength_nm"] = np.asarray(
            pending["wavelength_nm"], dtype=float
        ).copy()
        copied["intensity"] = np.asarray(
            pending["intensity"], dtype=float
        ).copy()
        return copied

    def _reset_decision_state(self) -> None:
        self._contact_latched = False
        self._model_consensus_contact_frames = 0
        self._visual_contact_latched = False
        self._visual_activation_frames = 0
        self._ambiguous_quiet_frames = 0
        self._ambiguous_quiet_started_at_sec = None
        self._quiet_no_contact_hint = False
        self._near_baseline_quiet_frames = 0
        self._last_activity_timestamp_sec = None
        self._position_probability_ema.clear()
        self._visual_position_probability_ema.clear()
        self._stable_position_id = None
        self._stable_position_timestamp_sec = None
        self._pending_position_id = None
        self._pending_position_frames = 0
        self._provisional_visual_position_id = None
        self._visual_position_locked_at_sec = None
        self._pending_visual_position_id = None
        self._pending_visual_position_frames = 0

    def set_baseline(
        self,
        wavelength_nm: Iterable[float],
        intensity: Iterable[float],
    ) -> None:
        wavelength = _finite_vector(wavelength_nm, "baseline wavelength")
        baseline = _finite_vector(intensity, "baseline intensity")
        if wavelength.shape != baseline.shape:
            raise ValueError("baseline wavelength and intensity must align")
        if np.any(np.diff(wavelength) <= 0):
            raise ValueError("baseline wavelength must be strictly increasing")
        self.baseline_wavelength_nm = wavelength
        self.baseline_intensity = baseline
        self.clear_history(confirmed_rest=True)

    def _smoothed_position_probabilities(
        self,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        if not probabilities:
            return {}
        alpha = self.position_ema_alpha
        if not self._position_probability_ema:
            smoothed = dict(probabilities)
        else:
            smoothed = {
                label: alpha * float(probability)
                + (1.0 - alpha)
                * float(self._position_probability_ema.get(label, 0.0))
                for label, probability in probabilities.items()
            }
        total = max(sum(smoothed.values()), 1.0e-12)
        self._position_probability_ema = {
            label: float(value / total) for label, value in smoothed.items()
        }
        return dict(self._position_probability_ema)

    def _smoothed_visual_position_probabilities(
        self,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        if not probabilities:
            return {}
        alpha = self.position_ema_alpha
        if not self._visual_position_probability_ema:
            smoothed = dict(probabilities)
        else:
            smoothed = {
                label: alpha * float(probability)
                + (1.0 - alpha)
                * float(self._visual_position_probability_ema.get(label, 0.0))
                for label, probability in probabilities.items()
            }
        total = max(sum(smoothed.values()), 1.0e-12)
        self._visual_position_probability_ema = {
            label: float(value / total) for label, value in smoothed.items()
        }
        return dict(self._visual_position_probability_ema)

    def _spectrum_on_baseline_grid(
        self,
        wavelength_nm: Iterable[float],
        intensity: Iterable[float],
    ) -> np.ndarray:
        if self.baseline_wavelength_nm is None:
            raise RuntimeError("baseline is not configured")
        wavelength = _finite_vector(wavelength_nm, "current wavelength")
        current = _finite_vector(intensity, "current intensity")
        if wavelength.shape != current.shape:
            raise ValueError("current wavelength and intensity must align")
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        current = current[order]
        wavelength, unique_indices = np.unique(wavelength, return_index=True)
        current = current[unique_indices]
        baseline_grid = self.baseline_wavelength_nm
        if (
            wavelength[0] > baseline_grid[0]
            or wavelength[-1] < baseline_grid[-1]
        ):
            raise ValueError("current wavelength grid does not cover the baseline grid")
        if (
            wavelength.shape == baseline_grid.shape
            and np.allclose(wavelength, baseline_grid, rtol=0.0, atol=1.0e-9)
        ):
            return current
        return np.interp(baseline_grid, wavelength, current)

    def _temporal_features(self) -> tuple[np.ndarray, int, float]:
        values = np.asarray(self.frame_features, dtype=np.float32)
        times = np.asarray(self.frame_times_sec, dtype=float)
        frame_count = len(values)
        duration_sec = float(times[-1] - times[0]) if frame_count > 1 else 0.0
        temporal_available = frame_count >= 2 and duration_sec > 0.0
        if frame_count == 1:
            values = np.repeat(values, 2, axis=0)
        summary = temporal_summary_features(values[None, :, :])[0]
        context = np.asarray(
            [
                np.log1p(float(frame_count)),
                duration_sec,
                1.0 if temporal_available else 0.0,
            ],
            dtype=np.float32,
        )
        features = np.concatenate((summary.astype(np.float32), context))
        if features.shape != (483,):
            raise RuntimeError("runtime temporal feature shape mismatch")
        return features, frame_count, duration_sec

    def _surface_proxy(
        self,
        position_id: str | None,
        position_probabilities: Mapping[str, float],
        raw_position_probabilities: Mapping[str, float],
        drive_force_n: float,
        visual_active: bool,
        *,
        semantic_contact_active: bool,
        observed_response: Mapping[str, Any],
    ) -> dict[str, Any]:
        force_ratio = float(
            np.clip(
                (drive_force_n - self.visual_force_off_n)
                / max(
                    self.visual_force_full_scale_n - self.visual_force_off_n,
                    1.0e-9,
                ),
                0.0,
                1.0,
            )
        )
        deformation = float(
            np.clip(
                self.visual_deformation_floor
                + (1.0 - self.visual_deformation_floor)
                * force_ratio**self.visual_force_gamma,
                0.0,
                1.0,
            )
        )
        active = bool(
            visual_active
            and position_id in POSITION_COORDINATES
        )
        inferred_surface = build_inferred_contact_probability_surface(
            position_probabilities,
            accepted_position_id=position_id,
            deformation=deformation,
            active=active,
            position_order=POSITION_ORDER,
            display_rows=DISPLAY_ROWS,
            position_coordinates=POSITION_COORDINATES,
            probability_source="trained_position_predict_proba_ema",
            probability_semantics=(
                "runtime_position_posterior_ema_scaled_by_optical_force_"
                "not_measured_pressure"
            ),
        )
        metrics = dict(inferred_surface["surface_metrics"])
        metrics["coupling_status"] = (
            "trained_position_probability_from_coupled_spectrum"
            if active
            else "optical_model_no_contact"
        )
        return {
            "active": active,
            "visual_active": active,
            "semantic_contact_active": bool(semantic_contact_active),
            "position_id": position_id if active else None,
            "semantic_position_id": position_id if active else None,
            "inferred_dominant_channel": inferred_surface["dominant_channel"],
            "drive_force_n": float(drive_force_n) if active else 0.0,
            "drive_ratio": force_ratio if active else 0.0,
            "drive_full_scale_n": self.visual_force_full_scale_n,
            "drive_source": (
                "semantic_contact_continuous_optical_force"
                if active and semantic_contact_active
                else ("low_force_visual_gate" if active else "none")
            ),
            "deformation_proxy": deformation if active else 0.0,
            "map_kind": inferred_surface["map_kind"],
            "map_source": inferred_surface["map_source"],
            "map_semantics": inferred_surface["map_semantics"],
            "inferred_contact_probability_grid": inferred_surface[
                "probability_grid"
            ],
            "inferred_contact_probabilities": inferred_surface[
                "probabilities"
            ],
            "raw_inferred_contact_probabilities": {
                channel_id: float(
                    np.clip(
                        float(raw_position_probabilities.get(channel_id, 0.0)),
                        0.0,
                        1.0,
                    )
                )
                for channel_id in POSITION_ORDER
            },
            "observed_coupled_spectral_response": dict(observed_response),
            "surface_grid": inferred_surface["surface_grid"],
            "surface_metrics": metrics,
        }

    def update(
        self,
        wavelength_nm: Iterable[float],
        intensity: Iterable[float],
        *,
        source_timestamp_sec: float | None = None,
    ) -> dict[str, Any]:
        if (
            self.baseline_wavelength_nm is None
            or self.baseline_intensity is None
        ):
            return {
                "ok": False,
                "status": "baseline_required",
                "reason": "set a stable no-contact spectrum baseline",
            }
        started = time.perf_counter()
        try:
            current = self._spectrum_on_baseline_grid(
                wavelength_nm, intensity
            )
            frame_matrix, frame_names, frame_components, component_names = (
                extract_baseline_relative_frame_features(
                    self.baseline_wavelength_nm,
                    current,
                    self.baseline_intensity,
                    self.peak_windows,
                )
            )
            if tuple(frame_names) != self.runtime_frame_feature_names:
                raise ValueError("single-frame feature-name contract mismatch")
            observed_coupled_response = (
                build_observed_coupled_spectral_response(
                    frame_matrix[0],
                    frame_names,
                    self.peak_windows,
                    DISPLAY_ROWS,
                )
            )

            strict_feature_row: np.ndarray | None = None
            if self.literature_classification_enabled:
                if (
                    self.literature_runtime_schema_version
                    == JOINT_NINE_FBG_RUNTIME_SCHEMA
                ):
                    strict_matrix, strict_names, _ = (
                        extract_joint_nine_fbg_features(
                            current[None, :],
                            self.baseline_intensity,
                            self.baseline_wavelength_nm,
                            self.peak_windows,
                            bin_count=self.literature_bin_count,
                        )
                    )
                else:
                    strict_matrix, strict_names, _ = (
                        extract_baseline_relative_features(
                            current[None, :],
                            self.baseline_intensity,
                            self.baseline_wavelength_nm,
                            bin_count=self.literature_bin_count,
                        )
                    )
                if tuple(strict_names) != self.literature_static_feature_names:
                    raise ValueError(
                        "literature-guided static feature-name contract mismatch"
                    )
                strict_feature_row = strict_matrix[0].astype(np.float32)

            timestamp = (
                float(source_timestamp_sec)
                if source_timestamp_sec is not None
                and np.isfinite(float(source_timestamp_sec))
                else time.monotonic()
            )
            if self.frame_times_sec:
                previous_timestamp = self.frame_times_sec[-1]
                if timestamp < previous_timestamp - 1.0e-3:
                    # A real clock regression marks a new acquisition segment.
                    # Keeping the old temporal window would blend pre-reconnect
                    # contact evidence into the new stream and retain stale
                    # position/deformation state.
                    self.clear_history()
                elif timestamp <= previous_timestamp:
                    # Some exporters only provide coarse timestamps. Preserve
                    # adjacent frames from the same segment without creating a
                    # zero-duration temporal window.
                    timestamp = previous_timestamp + 1.0e-6
            if (
                self.frame_times_sec
                and timestamp - self.frame_times_sec[-1]
                > max(5.0, 2.5 * self.window_seconds)
            ):
                self.clear_history()
            self.frame_features.append(frame_matrix[0].astype(np.float32))
            self.frame_times_sec.append(timestamp)
            if strict_feature_row is not None:
                self.spectrum_feature_history.append(strict_feature_row)
            while (
                len(self.frame_times_sec) > 1
                and timestamp - self.frame_times_sec[0] > self.window_seconds
            ):
                self.frame_times_sec.popleft()
                self.frame_features.popleft()

            temporal, history_frames, history_duration_sec = (
                self._temporal_features()
            )
            row = temporal[None, :]
            if self.literature_classification_enabled:
                if strict_feature_row is None or not self.spectrum_feature_history:
                    raise RuntimeError("literature-guided feature history is unavailable")
                strict_input = strict_feature_row[None, :]
                if (
                    self.literature_contact_feature_view
                    == "literature_snv_sg_temporal488"
                ):
                    contact_input = literature_runtime_contact_features(
                        np.asarray(
                            self.spectrum_feature_history,
                            dtype=np.float32,
                        ),
                        temporal_window_frames=(
                            self.literature_temporal_window_frames
                        ),
                    )
                elif self.literature_contact_feature_view == "response_raw136":
                    contact_input = response_raw_features(strict_input)
                elif self.literature_contact_feature_view == "baseline_relative_264":
                    contact_input = strict_input
                elif self.literature_contact_feature_view == (
                    "baseline_relative_264_plus_nine_fbg_joint_339"
                ):
                    contact_input = strict_input[
                        :, self.literature_contact_feature_indices
                    ]
                else:  # pragma: no cover - guarded at bundle load time
                    raise RuntimeError(
                        "unsupported literature-guided contact feature view"
                    )

                if self.literature_position_feature_view == "reference_full264":
                    position_input = strict_input
                elif self.literature_position_feature_view == "response_raw136":
                    position_input = response_raw_features(strict_input)
                elif self.literature_position_feature_view == "baseline_relative_264":
                    position_input = strict_input
                elif self.literature_position_feature_view == "nine_fbg_joint_75":
                    position_input = strict_input[
                        :, self.literature_position_feature_indices
                    ]
                else:  # pragma: no cover - guarded at bundle load time
                    raise RuntimeError(
                        "unsupported literature-guided position feature view"
                    )
            else:
                contact_input = row[:, self.contact_indices]
                position_input = row[:, self.position_indices]
            contact_probabilities = _class_probabilities(
                self.contact_model, contact_input
            )
            if "contact" not in contact_probabilities:
                positive_probability = next(
                    (
                        contact_probabilities[label]
                        for label in ("1", "True", "true")
                        if label in contact_probabilities
                    ),
                    None,
                )
                negative_probability = next(
                    (
                        contact_probabilities[label]
                        for label in ("0", "False", "false")
                        if label in contact_probabilities
                    ),
                    None,
                )
                if positive_probability is not None:
                    contact_probabilities = {
                        "no_contact": float(
                            1.0 - positive_probability
                            if negative_probability is None
                            else negative_probability
                        ),
                        "contact": float(positive_probability),
                    }
            contact_probability = float(
                contact_probabilities.get("contact", 0.0)
            )
            raw_contact_active = contact_probability >= self.contact_threshold
            raw_contact_label = (
                "contact" if raw_contact_active else "no_contact"
            )

            raw_position_probabilities = _class_probabilities(
                self.position_model, position_input
            )
            if raw_position_probabilities:
                (
                    raw_position_id,
                    raw_position_confidence,
                    raw_position_margin,
                ) = _top_probability(raw_position_probabilities)
            else:
                raw_position_id = str(
                    _model_predict(self.position_model, position_input)[0]
                )
                raw_position_confidence = 1.0
                raw_position_margin = 1.0

            spatially_credible = bool(
                raw_position_id in POSITION_COORDINATES
                and raw_position_confidence >= self.position_confidence_min
                and raw_position_margin >= self.position_margin_min
            )
            coupled_signature = summarize_coupled_contact_signature(
                observed_coupled_response,
                low_response_threshold=(
                    self.coupled_signature_low_response_threshold
                ),
                nominal_response_threshold=(
                    self.coupled_signature_nominal_response_threshold
                ),
                minimum_low_response_channels=(
                    self.coupled_signature_minimum_low_response_channels
                ),
                minimum_nominal_response_channels=(
                    self.coupled_signature_minimum_nominal_response_channels
                ),
                expected_channel_count=len(self.peak_windows),
            )
            measured_coupled_contact_pattern = bool(
                coupled_signature["multichannel_pattern"]
            )
            model_supported_coupled_contact_pattern = bool(
                coupled_signature["all_channels_present"]
                and coupled_signature["low_response_channel_count"]
                >= self.coupled_signature_minimum_low_response_channels
                and contact_probability
                >= self.coupled_signature_model_contact_probability
                and raw_position_confidence
                >= self.coupled_signature_model_position_confidence
                and raw_position_margin
                >= self.coupled_signature_model_position_margin
            )
            coupled_contact_signature_credible = bool(
                not self.runtime_gate_enabled
                or not self.coupled_signature_enabled
                or measured_coupled_contact_pattern
                or model_supported_coupled_contact_pattern
            )
            coupled_signature.update(
                {
                    "enabled": self.coupled_signature_enabled,
                    "measured_contact_pattern": (
                        measured_coupled_contact_pattern
                    ),
                    "model_supported_contact_pattern": (
                        model_supported_coupled_contact_pattern
                    ),
                    "credible": coupled_contact_signature_credible,
                    "model_contact_probability_threshold": (
                        self.coupled_signature_model_contact_probability
                    ),
                    "model_position_confidence_threshold": (
                        self.coupled_signature_model_position_confidence
                    ),
                    "model_position_margin_threshold": (
                        self.coupled_signature_model_position_margin
                    ),
                }
            )
            observed_coupled_response["joint_contact_signature"] = dict(
                coupled_signature
            )
            # The position model is trained only on contact samples, so its
            # confidence cannot provide an independent no-contact hint. A
            # timed optical-rest decision from the gate may, however, seed the
            # existing quiet-baseline recovery state machine on later frames.
            external_no_contact_hint = (
                True if self._quiet_no_contact_hint else None
            )
            recovery, recovered_baseline = (
                self._runtime_baseline_recovery.observe(
                    current,
                    physical_frame=True,
                    external_no_contact_hint=external_no_contact_hint,
                    release_event_probability=None,
                    position_confidence=raw_position_confidence,
                    contact_probability=contact_probability,
                    contact_label=raw_contact_label,
                    baseline_spectrum=self.baseline_intensity,
                    timestamp_sec=timestamp,
                )
            )
            shape_motion_rms = float(
                recovery.get("shape_motion_rms") or 0.0
            )
            common_gain_motion = float(
                recovery.get("common_gain_motion") or 0.0
            )
            fresh_activity = bool(
                shape_motion_rms >= self.activity_shape_motion_rms
                or common_gain_motion >= self.activity_common_gain_motion
            )
            if fresh_activity:
                self._last_activity_timestamp_sec = timestamp
            activity_recent = bool(
                self._last_activity_timestamp_sec is not None
                and timestamp - self._last_activity_timestamp_sec
                <= self.activity_memory_sec
            )

            model_baseline_reanchored = bool(
                recovered_baseline is not None
                and self.allow_model_baseline_reanchor
            )
            if model_baseline_reanchored:
                self.baseline_intensity = np.asarray(
                    recovered_baseline, dtype=float
                )
                self._pending_runtime_baseline_update = {
                    "wavelength_nm": self.baseline_wavelength_nm.copy(),
                    "intensity": self.baseline_intensity.copy(),
                    "sample_count": recovery.get(
                        "stable_release_physical_frames"
                    ),
                    "span_sec": recovery.get("quiet_elapsed_sec"),
                    "shape_motion_rms": recovery.get("shape_motion_rms"),
                    "common_gain_motion": recovery.get("common_gain_motion"),
                    "policy": recovery.get("policy"),
                }
                self._clear_temporal_history()
                self._reset_decision_state()
            elif recovered_baseline is not None:
                # Keep the classifier in the same baseline-relative feature
                # space used during training. The recovery guard may still
                # latch rest and refresh its motion reference, but it cannot
                # move the model's session baseline.
                recovery = {
                    **recovery,
                    "model_baseline_reanchor_blocked": True,
                    "model_baseline_policy": "fixed_clean_session_baseline",
                }
                self._clear_temporal_history()
                self._reset_decision_state()

            baseline_distance = recovery.get("baseline_distance")
            baseline_distance_growth = recovery.get("baseline_distance_growth")
            slow_baseline_departure = bool(
                recovery.get("slow_baseline_departure")
            )
            baseline_separated = bool(
                baseline_distance is not None
                and float(baseline_distance)
                >= self._runtime_baseline_recovery.minimum_contact_baseline_distance
            )
            model_consensus_baseline_separated = bool(
                baseline_distance is not None
                and float(baseline_distance)
                >= self.model_consensus_minimum_baseline_distance
            )
            near_runtime_baseline = bool(
                baseline_distance is not None
                and float(baseline_distance) <= self.baseline_release_distance
            )
            confirmed_release_transition = bool(
                recovery.get("release_transition_detected")
            )
            persistent_contact_plateau = bool(
                self._contact_latched
                and baseline_separated
                and not near_runtime_baseline
                and not confirmed_release_transition
            )

            quiet_ambiguous = bool(
                not fresh_activity
                and not spatially_credible
                and shape_motion_rms
                <= self._runtime_baseline_recovery.max_shape_motion_rms
                and common_gain_motion
                <= self._runtime_baseline_recovery.max_common_gain_motion
            )
            if self._contact_latched and quiet_ambiguous:
                self._ambiguous_quiet_frames += 1
                if self._ambiguous_quiet_started_at_sec is None:
                    self._ambiguous_quiet_started_at_sec = timestamp
            else:
                self._ambiguous_quiet_frames = 0
                self._ambiguous_quiet_started_at_sec = None

            ambiguous_quiet_elapsed_sec = (
                max(0.0, timestamp - self._ambiguous_quiet_started_at_sec)
                if self._ambiguous_quiet_started_at_sec is not None
                else 0.0
            )
            ambiguous_quiet_release = bool(
                self._contact_latched
                and quiet_ambiguous
                and not activity_recent
                and not persistent_contact_plateau
                and ambiguous_quiet_elapsed_sec
                >= self.ambiguous_quiet_release_sec
            )
            model_consensus_frame_evidence = bool(
                contact_probability
                >= self.model_consensus_contact_probability_on
                and spatially_credible
                and model_consensus_baseline_separated
                and coupled_contact_signature_credible
            )
            if model_consensus_frame_evidence:
                self._model_consensus_contact_frames = min(
                    self.model_consensus_contact_arm_frames,
                    self._model_consensus_contact_frames + 1,
                )
            else:
                self._model_consensus_contact_frames = 0
            model_consensus_contact_evidence = bool(
                model_consensus_frame_evidence
                and self._model_consensus_contact_frames
                >= self.model_consensus_contact_arm_frames
            )
            joint_fast_contact_evidence = bool(
                self.literature_runtime_schema_version
                == JOINT_NINE_FBG_RUNTIME_SCHEMA
                and raw_contact_active
                and raw_position_id in POSITION_COORDINATES
                and baseline_distance is not None
                and float(baseline_distance)
                >= self.joint_fast_minimum_baseline_distance
                and (
                    coupled_contact_signature_credible
                    or (
                        contact_probability
                        >= self.joint_fast_contact_probability_on
                        and raw_position_confidence
                        >= self.joint_fast_position_confidence_min
                        and raw_position_margin
                        >= self.joint_fast_position_margin_min
                    )
                )
            )
            if self._contact_latched and near_runtime_baseline and not fresh_activity:
                self._near_baseline_quiet_frames += 1
            else:
                self._near_baseline_quiet_frames = 0

            contact_on_evidence = bool(
                (
                    raw_contact_active
                    and (
                        baseline_separated
                        or not self.require_baseline_separation
                    )
                    and (
                        fresh_activity
                        or activity_recent
                        or (
                            slow_baseline_departure
                            and spatially_credible
                        )
                    )
                    and coupled_contact_signature_credible
                )
                or model_consensus_contact_evidence
                or joint_fast_contact_evidence
            )
            confirmed_rest_suppression = bool(
                recovery.get("suppress_contact")
                and recovery.get("runtime_rest_latched")
            )
            if not self.runtime_gate_enabled:
                self._contact_latched = raw_contact_active
            elif recovered_baseline is not None:
                self._contact_latched = False
            elif confirmed_release_transition:
                self._contact_latched = False
                self._quiet_no_contact_hint = True
            elif contact_on_evidence and (
                joint_fast_contact_evidence or not confirmed_rest_suppression
            ) and not (
                recovery.get("suppress_contact")
                and not (
                    fresh_activity
                    or activity_recent
                    or slow_baseline_departure
                    or model_consensus_contact_evidence
                    or joint_fast_contact_evidence
                )
            ):
                self._contact_latched = True
                if (
                    fresh_activity
                    or model_consensus_contact_evidence
                    or joint_fast_contact_evidence
                ):
                    self._quiet_no_contact_hint = False
            elif (
                (
                    contact_probability < self.contact_off_threshold
                    and not baseline_separated
                )
                or (
                    confirmed_rest_suppression
                    and not persistent_contact_plateau
                )
                or self._near_baseline_quiet_frames
                >= self.release_near_baseline_frames
                or ambiguous_quiet_release
            ):
                self._contact_latched = False
                if ambiguous_quiet_release:
                    self._quiet_no_contact_hint = True
            contact_active = bool(self._contact_latched)
            contact_label = "contact" if contact_active else "no_contact"

            if self.literature_force_enabled:
                if strict_feature_row is None:
                    raise RuntimeError(
                        "literature-guided force feature row is unavailable"
                    )
                if self.literature_force_feature_view == "baseline_relative_264":
                    force_input = strict_feature_row[None, :]
                elif self.literature_force_feature_view == (
                    "baseline_relative_264_plus_nine_fbg_joint_339"
                ):
                    force_input = strict_feature_row[None, :][
                        :, self.literature_force_feature_indices
                    ]
                else:
                    force_input = literature_snv_sg_features(
                        strict_feature_row[None, :]
                    )
            else:
                force_input = row[:, self.force_indices]
            raw_force_n = float(_model_predict(self.force_model, force_input)[0])
            if not np.isfinite(raw_force_n):
                raise RuntimeError("force model returned a non-finite estimate")
            # Compression force cannot be negative, but there is intentionally no
            # software upper clip. The training range is reported separately so
            # callers can distinguish measured-range estimates from extrapolation.
            raw_force_n = float(max(self.force_min_n, raw_force_n))
            force_range_status = (
                "above_calibrated_range"
                if raw_force_n > self.force_calibrated_max_n
                else "within_calibrated_range"
            )

            visual_position_probabilities = (
                self._smoothed_visual_position_probabilities(
                    raw_position_probabilities
                )
                if raw_position_probabilities
                else {}
            )
            (
                visual_candidate_id,
                visual_candidate_confidence,
                visual_candidate_margin,
            ) = _top_probability(visual_position_probabilities)
            visual_candidate_credible = bool(
                visual_candidate_id in POSITION_COORDINATES
                and visual_candidate_confidence
                >= self.visual_position_confidence_min
                and visual_candidate_margin
                >= self.visual_position_margin_min
            )
            visual_signal_evidence = bool(
                fresh_activity
                or activity_recent
                or (
                    slow_baseline_departure
                    and visual_candidate_credible
                )
            )
            strong_spectral_rest_visual_unlock = bool(
                recovery.get("runtime_rest_latched")
                and recovery.get("rest_unlock_strong_spectral_support")
                and raw_position_confidence
                >= self._runtime_baseline_recovery.rest_unlock_position_confidence_min
                and coupled_contact_signature_credible
            )
            fast_visual_contact_unlock = bool(
                confirmed_rest_suppression
                and fresh_activity
                and baseline_separated
                and visual_candidate_credible
                and raw_position_confidence
                >= self._runtime_baseline_recovery.rest_unlock_position_confidence_min
                and contact_probability
                >= self.visual_contact_probability_on
                and raw_force_n >= self.visual_force_on_n
                and coupled_contact_signature_credible
            )
            confirmed_rest_visual_unlock = bool(
                strong_spectral_rest_visual_unlock
                or fast_visual_contact_unlock
            )
            persistent_visual_plateau = bool(
                self._visual_contact_latched
                and baseline_separated
                and not near_runtime_baseline
                and not confirmed_release_transition
                and contact_probability
                > self.visual_contact_probability_off
                and raw_force_n > self.visual_force_off_n
                and not confirmed_rest_suppression
            )
            visual_activation_evidence = bool(
                visual_candidate_credible
                and contact_probability
                >= self.visual_contact_probability_on
                and raw_force_n >= self.visual_force_on_n
                and (
                    baseline_separated
                    or not self.require_baseline_separation
                )
                and visual_signal_evidence
                and coupled_contact_signature_credible
                and (
                    not bool(recovery.get("suppress_contact"))
                    or confirmed_rest_visual_unlock
                )
            )
            if not self.runtime_gate_enabled:
                self._visual_contact_latched = bool(
                    contact_active or visual_activation_evidence
                )
                self._visual_activation_frames = int(
                    self._visual_contact_latched
                )
            elif recovered_baseline is not None:
                self._visual_contact_latched = False
                self._visual_activation_frames = 0
            elif confirmed_release_transition:
                self._visual_contact_latched = False
                self._visual_activation_frames = 0
            elif contact_active:
                self._visual_contact_latched = True
                self._visual_activation_frames = self.visual_contact_arm_frames
            elif fast_visual_contact_unlock and visual_activation_evidence:
                self._visual_contact_latched = True
                self._visual_activation_frames = self.visual_contact_arm_frames
            elif visual_activation_evidence:
                self._visual_activation_frames = min(
                    self.visual_contact_arm_frames,
                    self._visual_activation_frames + 1,
                )
                if (
                    self._visual_activation_frames
                    >= self.visual_contact_arm_frames
                ):
                    self._visual_contact_latched = True
            else:
                self._visual_activation_frames = 0
                if (
                    near_runtime_baseline
                    or (
                        confirmed_rest_suppression
                        and not persistent_visual_plateau
                    )
                    or ambiguous_quiet_release
                    or not baseline_separated
                    or (
                        not activity_recent
                        and not slow_baseline_departure
                        and not persistent_visual_plateau
                    )
                    or (
                        contact_probability
                        <= self.visual_contact_probability_off
                        and raw_force_n <= self.visual_force_off_n
                    )
                ):
                    self._visual_contact_latched = False
            visual_contact_active = bool(
                contact_active or self._visual_contact_latched
            )

            position_probabilities: dict[str, float] = {}
            position_id: str | None = None
            position_confidence = 0.0
            position_margin = 0.0
            position_held = False
            joint_fast_initial_position_accept = False
            if contact_active and raw_position_probabilities:
                position_probabilities = self._smoothed_position_probabilities(
                    raw_position_probabilities
                )
                (
                    candidate_position_id,
                    candidate_position_confidence,
                    candidate_position_margin,
                ) = _top_probability(position_probabilities)
                candidate_credible = bool(
                    candidate_position_id in POSITION_COORDINATES
                    and candidate_position_confidence
                    >= self.position_confidence_min
                    and candidate_position_margin >= self.position_margin_min
                )
                if candidate_credible:
                    candidate_is_stable = bool(
                        candidate_position_id == self._stable_position_id
                    )
                    if not candidate_is_stable:
                        joint_fast_initial_position_accept = bool(
                            self.literature_runtime_schema_version
                            == JOINT_NINE_FBG_RUNTIME_SCHEMA
                            and self._stable_position_id is None
                            and joint_fast_contact_evidence
                            and candidate_position_confidence
                            >= self.joint_fast_position_confidence_min
                            and candidate_position_margin
                            >= self.joint_fast_position_margin_min
                        )
                        if joint_fast_initial_position_accept:
                            self._stable_position_id = candidate_position_id
                            self._stable_position_timestamp_sec = timestamp
                            self._pending_position_id = None
                            self._pending_position_frames = 0
                        elif self._pending_position_id == candidate_position_id:
                            self._pending_position_frames += 1
                        else:
                            self._pending_position_id = candidate_position_id
                            self._pending_position_frames = 1
                        if not joint_fast_initial_position_accept and (
                            self._pending_position_frames
                            >= self.position_switch_frames
                        ):
                            self._stable_position_id = candidate_position_id
                            self._stable_position_timestamp_sec = timestamp
                            self._pending_position_id = None
                            self._pending_position_frames = 0
                        else:
                            position_held = bool(
                                self._stable_position_id in POSITION_COORDINATES
                            )
                    else:
                        self._stable_position_timestamp_sec = timestamp
                        self._pending_position_id = None
                        self._pending_position_frames = 0
                    if (
                        self._stable_position_id in POSITION_COORDINATES
                        and (
                            not position_held
                            or (
                                self._stable_position_timestamp_sec is not None
                                and timestamp
                                - self._stable_position_timestamp_sec
                                <= self.position_hold_sec
                            )
                        )
                    ):
                        position_id = self._stable_position_id
                        position_confidence = float(
                            position_probabilities.get(position_id, 0.0)
                        )
                        competing = max(
                            (
                                value
                                for label, value in position_probabilities.items()
                                if label != position_id
                            ),
                            default=0.0,
                        )
                        position_margin = float(position_confidence - competing)
                elif (
                    self._stable_position_id in POSITION_COORDINATES
                    and self._stable_position_timestamp_sec is not None
                    and timestamp - self._stable_position_timestamp_sec
                    <= self.position_hold_sec
                ):
                    position_id = self._stable_position_id
                    position_confidence = float(
                        position_probabilities.get(position_id, 0.0)
                    )
                    competing = max(
                        (
                            value
                            for label, value in position_probabilities.items()
                            if label != position_id
                        ),
                        default=0.0,
                    )
                    position_margin = float(position_confidence - competing)
                    position_held = True
                    self._pending_position_id = None
                    self._pending_position_frames = 0
            if not contact_active:
                self._position_probability_ema.clear()
                self._stable_position_id = None
                self._stable_position_timestamp_sec = None
                self._pending_position_id = None
                self._pending_position_frames = 0
            if not visual_contact_active and not visual_activation_evidence:
                self._provisional_visual_position_id = None
                self._visual_position_locked_at_sec = None
                self._pending_visual_position_id = None
                self._pending_visual_position_frames = 0
                self._visual_position_probability_ema.clear()

            visual_position_in_settle_window = bool(
                self._provisional_visual_position_id in POSITION_COORDINATES
                and self._visual_position_locked_at_sec is not None
                and timestamp - self._visual_position_locked_at_sec
                <= self.visual_position_settle_sec
            )
            visual_position_switch_strong = bool(
                visual_candidate_credible
                and visual_candidate_confidence
                >= self.visual_position_switch_confidence_min
                and visual_candidate_margin
                >= self.visual_position_switch_margin_min
            )
            visual_position_pending_required_frames = 0
            if (
                position_id in POSITION_COORDINATES
                and self._provisional_visual_position_id is None
            ):
                # The formal position already passed its own multi-frame gate.
                self._provisional_visual_position_id = position_id
                self._visual_position_locked_at_sec = timestamp
                self._pending_visual_position_id = None
                self._pending_visual_position_frames = 0
            elif (
                self.visual_position_fallback_enabled
                and visual_candidate_credible
                and (
                    contact_active
                    or visual_activation_evidence
                    or persistent_visual_plateau
                )
            ):
                if visual_candidate_id == self._provisional_visual_position_id:
                    self._pending_visual_position_id = None
                    self._pending_visual_position_frames = 0
                else:
                    switch_allowed = bool(
                        self._provisional_visual_position_id is None
                        or visual_position_in_settle_window
                        or visual_position_switch_strong
                    )
                    visual_position_pending_required_frames = (
                        self.visual_position_confirm_frames
                        if self._provisional_visual_position_id is None
                        or visual_position_in_settle_window
                        else self.visual_position_switch_frames
                    )
                    if not switch_allowed:
                        self._pending_visual_position_id = None
                        self._pending_visual_position_frames = 0
                    elif self._pending_visual_position_id == visual_candidate_id:
                        self._pending_visual_position_frames += 1
                    else:
                        self._pending_visual_position_id = visual_candidate_id
                        self._pending_visual_position_frames = 1
                    if (
                        switch_allowed
                        and self._pending_visual_position_frames
                        >= visual_position_pending_required_frames
                    ):
                        self._provisional_visual_position_id = visual_candidate_id
                        self._visual_position_locked_at_sec = timestamp
                        self._pending_visual_position_id = None
                        self._pending_visual_position_frames = 0
            elif (
                visual_contact_active
                and position_id is None
                and (
                    not persistent_visual_plateau
                    or not visual_candidate_credible
                )
            ):
                # A stationary press can become spatially ambiguous for several
                # frames even though the full spectrum remains contact-like.
                # Preserve the established contact-episode position until the
                # release gate clears visual_contact_active.
                self._pending_visual_position_id = None
                self._pending_visual_position_frames = 0

            visual_position_id = (
                self._provisional_visual_position_id
                if visual_contact_active
                else None
            )
            visual_position_confidence = 0.0
            visual_position_margin = 0.0
            visual_position_provisional = bool(
                visual_position_id in POSITION_COORDINATES
                and visual_position_id != position_id
            )
            if (
                visual_contact_active
                and visual_position_id in POSITION_COORDINATES
                and visual_position_probabilities
            ):
                visual_position_confidence = float(
                    visual_position_probabilities.get(visual_position_id, 0.0)
                )
                competing = max(
                    (
                        value
                        for label, value in visual_position_probabilities.items()
                        if label != visual_position_id
                    ),
                    default=0.0,
                )
                visual_position_margin = float(
                    visual_position_confidence - competing
                )

            estimated_force_n = (
                raw_force_n if contact_active else self.no_contact_force_n
            )
            visual_drive_force_n = (
                raw_force_n
                if visual_contact_active
                else self.no_contact_force_n
            )
            twin = self._surface_proxy(
                visual_position_id,
                visual_position_probabilities,
                raw_position_probabilities,
                visual_drive_force_n,
                visual_contact_active,
                semantic_contact_active=contact_active,
                observed_response=observed_coupled_response,
            )
            twin["position_source"] = (
                "formal_position"
                if position_id is not None
                else (
                    (
                        "provisional_low_force_visual_position"
                        if not contact_active
                        else "provisional_low_confidence_position"
                    )
                    if visual_position_provisional
                    else "none"
                )
            )
            contact_margin = abs(
                contact_probability - self.contact_threshold
            )
            uncertainty_reasons: list[str] = []
            if contact_margin < 0.10:
                uncertainty_reasons.append("contact_probability_near_gate")
            if raw_contact_active and not contact_active:
                uncertainty_reasons.append("contact_suppressed_by_runtime_gate")
            if contact_active and position_id is None:
                uncertainty_reasons.append("position_not_stable_enough")
            elif contact_active and position_confidence < 0.60:
                uncertainty_reasons.append("position_confidence_low")

            latency_ms = (time.perf_counter() - started) * 1000.0
            return {
                "ok": True,
                "status": "ready",
                "schema_version": MODEL_SCHEMA,
                "recognition_source": self.classification_model_source,
                "classification_model_source": self.classification_model_source,
                "force_model_source": self.force_model_source,
                "runtime_input": "optical_spectrum_time_series",
                "force_sensor_is_runtime_input": False,
                "contact": {
                    "label": contact_label,
                    "confidence": (
                        contact_probability
                        if contact_active
                        else 1.0 - contact_probability
                    ),
                    "contact_probability": contact_probability,
                    "probability_threshold": self.contact_threshold,
                    "probabilities": contact_probabilities,
                    "raw_model_label": raw_contact_label,
                },
                "position": {
                    "label": position_id,
                    "confidence": position_confidence,
                    "margin": position_margin,
                    "probabilities": position_probabilities,
                    "raw_label": raw_position_id,
                    "raw_confidence": raw_position_confidence,
                    "raw_margin": raw_position_margin,
                    "raw_probabilities": raw_position_probabilities,
                    "accepted": bool(position_id),
                    "held_from_previous_frame": position_held,
                    "confidence_threshold": self.position_confidence_min,
                    "margin_threshold": self.position_margin_min,
                    "visual_label": visual_position_id,
                    "visual_confidence": visual_position_confidence,
                    "visual_margin": visual_position_margin,
                    "visual_probabilities": visual_position_probabilities,
                    "visual_fallback_used": visual_position_provisional,
                    "visual_confidence_threshold": (
                        self.visual_position_confidence_min
                    ),
                    "visual_margin_threshold": self.visual_position_margin_min,
                    "visual_confirm_frames": self.visual_position_confirm_frames,
                    "visual_active": visual_contact_active,
                },
                "estimated_force_fz_n": float(estimated_force_n),
                "continuous_force_fz_n": float(raw_force_n),
                "force_fz": {
                    "estimated_n": float(estimated_force_n),
                    "continuous_estimated_n": float(raw_force_n),
                    "visual_drive_n": float(visual_drive_force_n),
                    "raw_estimated_n": raw_force_n,
                    "unit": "N",
                    "gated": not contact_active,
                    "continuous_trace_before_contact_gate": True,
                    "clip_range_n": [
                        self.force_min_n,
                        None,
                    ],
                    "calibrated_range_n": [
                        self.force_min_n,
                        self.force_calibrated_max_n,
                    ],
                    "upper_limit_applied": False,
                    "range_status": force_range_status,
                    "outside_calibrated_range": (
                        force_range_status == "above_calibrated_range"
                    ),
                    "runtime_input": "optical_spectrum_time_series",
                    "calibration_supervision": "PX6D Fz",
                    "model_source": self.force_model_source,
                },
                "digital_twin": twin,
                "observed_coupled_spectral_response": (
                    observed_coupled_response
                ),
                "uncertainty": {
                    "review_needed": bool(uncertainty_reasons),
                    "reasons": uncertainty_reasons,
                },
                "runtime_contact_gate": {
                    "enabled": self.runtime_gate_enabled,
                    "raw_contact_active": raw_contact_active,
                    "contact_latched": contact_active,
                    "visual_contact_active": visual_contact_active,
                    "visual_contact_latched": self._visual_contact_latched,
                    "visual_contact_arm_frames": self.visual_contact_arm_frames,
                    "visual_activation_frames": self._visual_activation_frames,
                    "visual_activation_evidence": (
                        visual_activation_evidence
                    ),
                    "visual_signal_evidence": visual_signal_evidence,
                    "visual_position_credible": visual_candidate_credible,
                    "visual_candidate_position": visual_candidate_id,
                    "visual_candidate_confidence": (
                        visual_candidate_confidence
                    ),
                    "visual_candidate_margin": visual_candidate_margin,
                    "visual_position_ambiguity_rejected": bool(
                        visual_contact_active
                        and position_id is None
                        and not visual_candidate_credible
                    ),
                    "visual_contact_probability_on": (
                        self.visual_contact_probability_on
                    ),
                    "visual_contact_probability_off": (
                        self.visual_contact_probability_off
                    ),
                    "visual_force_on_n": self.visual_force_on_n,
                    "visual_force_off_n": self.visual_force_off_n,
                    "visual_force_full_scale_n": (
                        self.visual_force_full_scale_n
                    ),
                    "spatially_credible": spatially_credible,
                    "model_consensus_contact_probability_on": (
                        self.model_consensus_contact_probability_on
                    ),
                    "model_consensus_minimum_baseline_distance": (
                        self.model_consensus_minimum_baseline_distance
                    ),
                    "model_consensus_contact_arm_frames": (
                        self.model_consensus_contact_arm_frames
                    ),
                    "model_consensus_contact_frames": (
                        self._model_consensus_contact_frames
                    ),
                    "model_consensus_baseline_separated": (
                        model_consensus_baseline_separated
                    ),
                    "model_consensus_frame_evidence": (
                        model_consensus_frame_evidence
                    ),
                    "model_consensus_contact_evidence": (
                        model_consensus_contact_evidence
                    ),
                    "joint_fast_contact_evidence": (
                        joint_fast_contact_evidence
                    ),
                    "joint_fast_contact_probability_on": (
                        self.joint_fast_contact_probability_on
                    ),
                    "joint_fast_position_confidence_min": (
                        self.joint_fast_position_confidence_min
                    ),
                    "joint_fast_position_margin_min": (
                        self.joint_fast_position_margin_min
                    ),
                    "joint_fast_minimum_baseline_distance": (
                        self.joint_fast_minimum_baseline_distance
                    ),
                    "coupled_contact_signature_enabled": (
                        self.coupled_signature_enabled
                    ),
                    "coupled_contact_signature_credible": (
                        coupled_contact_signature_credible
                    ),
                    "coupled_contact_measured_pattern": (
                        measured_coupled_contact_pattern
                    ),
                    "coupled_contact_model_supported_pattern": (
                        model_supported_coupled_contact_pattern
                    ),
                    "coupled_contact_signature": dict(coupled_signature),
                    "confirmed_rest_suppression": (
                        confirmed_rest_suppression
                    ),
                    "confirmed_release_transition": (
                        confirmed_release_transition
                    ),
                    "confirmed_rest_visual_unlock": (
                        confirmed_rest_visual_unlock
                    ),
                    "strong_spectral_rest_visual_unlock": (
                        strong_spectral_rest_visual_unlock
                    ),
                    "fast_visual_contact_unlock": (
                        fast_visual_contact_unlock
                    ),
                    "fresh_spectral_activity": fresh_activity,
                    "spectral_activity_recent": activity_recent,
                    "quiet_spatially_ambiguous": quiet_ambiguous,
                    "ambiguous_quiet_frames": self._ambiguous_quiet_frames,
                    "ambiguous_quiet_elapsed_sec": (
                        ambiguous_quiet_elapsed_sec
                    ),
                    "ambiguous_quiet_release_sec": (
                        self.ambiguous_quiet_release_sec
                    ),
                    "ambiguous_quiet_is_release_evidence": (
                        ambiguous_quiet_release
                    ),
                    "quiet_no_contact_hint": self._quiet_no_contact_hint,
                    "persistent_contact_plateau": persistent_contact_plateau,
                    "persistent_visual_plateau": persistent_visual_plateau,
                    "baseline_separated": baseline_separated,
                    "baseline_distance": baseline_distance,
                    "baseline_distance_growth": baseline_distance_growth,
                    "slow_baseline_departure": slow_baseline_departure,
                    "minimum_contact_baseline_distance": (
                        self._runtime_baseline_recovery.minimum_contact_baseline_distance
                    ),
                    "near_runtime_baseline": near_runtime_baseline,
                    "near_baseline_quiet_frames": (
                        self._near_baseline_quiet_frames
                    ),
                    "baseline_release_distance": self.baseline_release_distance,
                    "position_switch_frames": self.position_switch_frames,
                    "joint_fast_initial_position_accept": (
                        joint_fast_initial_position_accept
                    ),
                    "pending_position_id": self._pending_position_id,
                    "pending_position_frames": self._pending_position_frames,
                    "visual_position_confirm_frames": (
                        self.visual_position_confirm_frames
                    ),
                    "visual_position_settle_sec": self.visual_position_settle_sec,
                    "visual_position_in_settle_window": (
                        visual_position_in_settle_window
                    ),
                    "visual_position_switch_confidence_min": (
                        self.visual_position_switch_confidence_min
                    ),
                    "visual_position_switch_margin_min": (
                        self.visual_position_switch_margin_min
                    ),
                    "visual_position_switch_frames": (
                        self.visual_position_switch_frames
                    ),
                    "visual_position_switch_strong_evidence": (
                        visual_position_switch_strong
                    ),
                    "visual_position_pending_required_frames": (
                        visual_position_pending_required_frames
                    ),
                    "provisional_visual_position_id": (
                        self._provisional_visual_position_id
                    ),
                    "visual_position_locked_at_sec": (
                        self._visual_position_locked_at_sec
                    ),
                    "pending_visual_position_id": (
                        self._pending_visual_position_id
                    ),
                    "pending_visual_position_frames": (
                        self._pending_visual_position_frames
                    ),
                    "shape_motion_rms": shape_motion_rms,
                    "common_gain_motion": common_gain_motion,
                    "activity_shape_motion_rms": (
                        self.activity_shape_motion_rms
                    ),
                    "activity_common_gain_motion": (
                        self.activity_common_gain_motion
                    ),
                    "runtime_reference_reanchored": bool(
                        recovered_baseline is not None
                    ),
                    "model_baseline_reanchored": model_baseline_reanchored,
                    "model_baseline_reanchor_allowed": (
                        self.allow_model_baseline_reanchor
                    ),
                    "baseline_recovery": recovery,
                    "policy": (
                        "model_probability_plus_baseline_separation_plus_"
                        "nine_fbg_joint_signature_plus_spectral_change_or_"
                        "two_frame_model_consensus_"
                        "plus_multiframe_contact_and_position_confirmation_"
                        "plus_timed_quiet_ambiguous_release"
                    ),
                },
                "frame_response_components": {
                    name: float(value)
                    for name, value in zip(
                        component_names,
                        frame_components[0],
                        strict=True,
                    )
                },
                "history_frames": history_frames,
                "history_duration_sec": history_duration_sec,
                "temporal_window_sec": self.window_seconds,
                "inference_latency_ms": latency_ms,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "inference_error",
                "reason": f"{type(exc).__name__}: {exc}",
            }


__all__ = [
    "AllSourceOpticalForceAdapter",
    "DISPLAY_ROWS",
    "FEATURE_SCHEMA",
    "MODEL_SCHEMA",
    "POSITION_COORDINATES",
    "POSITION_ORDER",
]
