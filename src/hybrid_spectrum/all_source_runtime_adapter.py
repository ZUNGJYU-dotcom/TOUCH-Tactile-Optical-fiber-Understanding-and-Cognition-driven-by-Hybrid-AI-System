"""Beta runtime adapter for the all-source optical-only ordinary-FBG model."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any, Iterable

import joblib
import numpy as np

from .dynamic_sequence_dataset import extract_baseline_relative_frame_features
from .dynamic_temporal_features import temporal_summary_features
from .features import PeakWindow, load_peak_windows


MODEL_SCHEMA = "ordinary_fbg_optical_only_force_candidate_v2"
FEATURE_SCHEMA = "nine_peak_shift_intensity_shape_temporal_summary_483"
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
    probabilities = np.asarray(model.predict_proba(features), dtype=float)[0]
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


class AllSourceOpticalForceAdapter:
    """Run the grouped-evaluated all-source model from optical spectra only."""

    def __init__(
        self,
        bundle: dict[str, Any],
        peak_windows: Iterable[PeakWindow],
        *,
        window_seconds: float = 2.0,
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

        force_contract = bundle.get("force_calibration_contract", {})
        gate = force_contract.get("optical_contact_gate", {})
        force_range = force_contract.get("prediction_clip_range_n", (0.0, 5.0))
        self.contact_threshold = float(gate.get("probability_threshold", 0.75))
        self.no_contact_force_n = float(gate.get("no_contact_output_n", 0.0))
        self.force_min_n = float(force_range[0])
        self.force_max_n = float(force_range[1])

        self.baseline_wavelength_nm: np.ndarray | None = None
        self.baseline_intensity: np.ndarray | None = None
        self.frame_features: deque[np.ndarray] = deque()
        self.frame_times_sec: deque[float] = deque()

    @classmethod
    def from_paths(
        cls,
        model_path: Path,
        peak_config_path: Path,
        *,
        window_seconds: float = 2.0,
    ) -> "AllSourceOpticalForceAdapter":
        bundle = joblib.load(Path(model_path))
        if not isinstance(bundle, dict):
            raise TypeError("all-source candidate artifact must contain a mapping")
        return cls(
            bundle,
            load_peak_windows(Path(peak_config_path)),
            window_seconds=window_seconds,
        )

    def _validate_contract(self) -> None:
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
            model_count = int(getattr(task["model"], "n_features_in_", -1))
            if model_count != expected_count:
                raise ValueError(f"{task_name} model feature count mismatch")
        if not all(
            self.all_feature_names[index].startswith("last__")
            for index in self.force_indices
        ):
            raise ValueError("force model must use current-frame last__ features")

    def clear(self) -> None:
        self.baseline_wavelength_nm = None
        self.baseline_intensity = None
        self.frame_features.clear()
        self.frame_times_sec.clear()

    def clear_history(self) -> None:
        self.frame_features.clear()
        self.frame_times_sec.clear()

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
        self.clear_history()

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
        estimated_force_n: float,
        contact_active: bool,
    ) -> dict[str, Any]:
        if not contact_active or position_id not in POSITION_COORDINATES:
            grid = [[0.0, 0.0, 0.0] for _ in range(3)]
            return {
                "active": False,
                "position_id": None,
                "deformation_proxy": 0.0,
                "surface_grid": grid,
                "surface_metrics": {
                    "surface_peak": 0.0,
                    "surface_mean": 0.0,
                    "surface_area_active": 0.0,
                    "surface_centroid_x": 0.0,
                    "surface_centroid_y": 0.0,
                    "surface_spread": 0.0,
                    "dominant_channel": None,
                    "coupling_status": "optical_model_no_contact",
                },
            }

        force_ratio = float(
            np.clip(
                (estimated_force_n - self.force_min_n)
                / max(self.force_max_n - self.force_min_n, 1.0e-9),
                0.0,
                1.0,
            )
        )
        deformation = float(
            np.clip(0.10 + 0.90 * force_ratio**0.65, 0.0, 1.0)
        )
        sigma = 0.68 + 0.18 * np.sqrt(force_ratio)
        center_x, center_y = POSITION_COORDINATES[position_id]
        values: dict[str, float] = {}
        for channel_id, (x_value, y_value) in POSITION_COORDINATES.items():
            distance_squared = (
                (x_value - center_x) ** 2 + (y_value - center_y) ** 2
            )
            values[channel_id] = float(
                deformation
                * np.exp(-distance_squared / (2.0 * sigma * sigma))
            )
        grid = [
            [values[channel_id] for channel_id in row]
            for row in DISPLAY_ROWS
        ]
        flat = np.asarray(grid, dtype=float)
        weights = flat.ravel()
        points = [
            POSITION_COORDINATES[channel_id]
            for row in DISPLAY_ROWS
            for channel_id in row
        ]
        weight_sum = max(float(np.sum(weights)), 1.0e-9)
        centroid_x = float(
            sum(weight * point[0] for weight, point in zip(weights, points))
            / weight_sum
        )
        centroid_y = float(
            sum(weight * point[1] for weight, point in zip(weights, points))
            / weight_sum
        )
        return {
            "active": True,
            "position_id": position_id,
            "deformation_proxy": deformation,
            "surface_grid": grid,
            "surface_metrics": {
                "surface_peak": float(np.max(flat)),
                "surface_mean": float(np.mean(flat)),
                "surface_area_active": float(np.mean(flat >= 0.055)),
                "surface_centroid_x": centroid_x,
                "surface_centroid_y": centroid_y,
                "surface_spread": float(sigma),
                "dominant_channel": position_id,
                "coupling_status": "optical_model_continuous_force_proxy",
            },
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
            frame_matrix, frame_names, _, _ = (
                extract_baseline_relative_frame_features(
                    self.baseline_wavelength_nm,
                    current,
                    self.baseline_intensity,
                    self.peak_windows,
                )
            )
            expected_last_names = tuple(
                name.removeprefix("last__")
                for name in self.force_task["feature_names"]
            )
            if tuple(frame_names) != expected_last_names:
                raise ValueError("single-frame feature-name contract mismatch")

            timestamp = (
                float(source_timestamp_sec)
                if source_timestamp_sec is not None
                and np.isfinite(float(source_timestamp_sec))
                else time.monotonic()
            )
            if self.frame_times_sec and timestamp <= self.frame_times_sec[-1]:
                timestamp = self.frame_times_sec[-1] + 1.0e-6
            if (
                self.frame_times_sec
                and timestamp - self.frame_times_sec[-1]
                > max(5.0, 2.5 * self.window_seconds)
            ):
                self.clear_history()
            self.frame_features.append(frame_matrix[0].astype(np.float32))
            self.frame_times_sec.append(timestamp)
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
            contact_probabilities = _class_probabilities(
                self.contact_model, row[:, self.contact_indices]
            )
            contact_probability = float(
                contact_probabilities.get("contact", 0.0)
            )
            contact_active = contact_probability >= self.contact_threshold
            contact_label = "contact" if contact_active else "no_contact"

            position_probabilities = _class_probabilities(
                self.position_model, row[:, self.position_indices]
            )
            if position_probabilities:
                position_id = max(
                    position_probabilities,
                    key=position_probabilities.get,
                )
                position_confidence = float(
                    position_probabilities[position_id]
                )
            else:
                position_id = str(
                    self.position_model.predict(
                        row[:, self.position_indices]
                    )[0]
                )
                position_confidence = 1.0
            if not contact_active:
                position_id = None
                position_confidence = 0.0

            raw_force_n = float(
                self.force_model.predict(row[:, self.force_indices])[0]
            )
            raw_force_n = float(
                np.clip(raw_force_n, self.force_min_n, self.force_max_n)
            )
            estimated_force_n = (
                raw_force_n if contact_active else self.no_contact_force_n
            )
            twin = self._surface_proxy(
                position_id, estimated_force_n, contact_active
            )
            contact_margin = abs(
                contact_probability - self.contact_threshold
            )
            uncertainty_reasons: list[str] = []
            if contact_margin < 0.10:
                uncertainty_reasons.append("contact_probability_near_gate")
            if contact_active and position_confidence < 0.60:
                uncertainty_reasons.append("position_confidence_low")

            latency_ms = (time.perf_counter() - started) * 1000.0
            return {
                "ok": True,
                "status": "ready",
                "schema_version": MODEL_SCHEMA,
                "recognition_source": "ordinary_fbg_all_data_beta_v1",
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
                },
                "position": {
                    "label": position_id,
                    "confidence": position_confidence,
                    "probabilities": position_probabilities,
                },
                "estimated_force_fz_n": float(estimated_force_n),
                "force_fz": {
                    "estimated_n": float(estimated_force_n),
                    "raw_estimated_n": raw_force_n,
                    "unit": "N",
                    "gated": not contact_active,
                    "clip_range_n": [
                        self.force_min_n,
                        self.force_max_n,
                    ],
                    "runtime_input": "optical_spectrum_time_series",
                    "calibration_supervision": "PX6D Fz",
                },
                "digital_twin": twin,
                "uncertainty": {
                    "review_needed": bool(uncertainty_reasons),
                    "reasons": uncertainty_reasons,
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
