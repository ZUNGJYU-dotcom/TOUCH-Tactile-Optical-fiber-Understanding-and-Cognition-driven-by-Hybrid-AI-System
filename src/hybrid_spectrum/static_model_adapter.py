"""Runtime adapter from one full spectrum to digital-twin position and level."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .sense_static_dataset import extract_snapshot_feature_vectors


class StaticSpectralPredictor:
    """Load a trained hierarchical bundle and expose a stable prediction API."""

    def __init__(self, bundle_path: Path) -> None:
        bundle = joblib.load(bundle_path)
        if bundle.get("schema_version") != "sense_static_hierarchical_bundle_v1":
            raise ValueError("unsupported static spectral model bundle")
        self.bundle_path = bundle_path.resolve()
        self.bundle_sha256 = hashlib.sha256(self.bundle_path.read_bytes()).hexdigest()
        self.bundle = bundle
        self.grid = np.asarray(bundle["common_wavelength_nm"], dtype=float)
        self.fallback_baseline = np.asarray(bundle["fallback_baseline_spectrum"], dtype=float)
        self.peak_windows = tuple(bundle["peak_windows"])
        self.full_spectrum_bins = int(bundle["full_spectrum_bins"])
        self.models = bundle["models"]
        self.digital_twin_contract = bundle["digital_twin_contract"]

    @staticmethod
    def _validate_arrays(
        wavelength_nm: Iterable[float],
        intensity_counts: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        wavelength = np.asarray(list(wavelength_nm), dtype=float)
        intensity = np.asarray(list(intensity_counts), dtype=float)
        if wavelength.ndim != 1 or intensity.ndim != 1 or wavelength.size != intensity.size:
            raise ValueError("wavelength and intensity must be equal-length one-dimensional arrays")
        if wavelength.size < 16:
            raise ValueError("spectrum has too few points")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(intensity)):
            raise ValueError("spectrum contains non-finite values")
        if not np.all(np.diff(wavelength) > 0.0):
            raise ValueError("wavelength grid must be strictly increasing")
        return wavelength, intensity

    @staticmethod
    def _predict_with_probabilities(
        model_payload: dict[str, Any],
        matrix: np.ndarray,
        *,
        model_override: Any | None = None,
    ) -> dict[str, Any]:
        model = model_override if model_override is not None else model_payload["model"]
        predicted = str(model.predict(matrix)[0])
        probabilities: dict[str, float] = {}
        confidence: float | None = None
        margin: float | None = None
        if hasattr(model, "predict_proba"):
            values = np.asarray(model.predict_proba(matrix)[0], dtype=float)
            classes = [str(value) for value in model.classes_]
            probabilities = {
                label: float(values[index]) for index, label in enumerate(classes)
            }
            ordered = np.sort(values)[::-1]
            confidence = float(ordered[0])
            margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else 1.0
        return {
            "label": predicted,
            "confidence": confidence,
            "margin": margin,
            "probabilities": probabilities,
            "confidence_source": (
                "uncalibrated_predict_proba" if probabilities else "unavailable"
            ),
            "probability_calibrated": False,
            "review_needed": bool(
                confidence is None
                or margin is None
                or confidence < 0.80
                or margin < 0.20
            ),
            "model_id": model_payload["model_id"],
            "feature_set": model_payload["feature_set"],
        }

    def predict(
        self,
        wavelength_nm: Iterable[float],
        intensity_counts: Iterable[float],
        *,
        baseline_wavelength_nm: Iterable[float] | None = None,
        baseline_intensity_counts: Iterable[float] | None = None,
        allow_fallback_baseline: bool = False,
    ) -> dict[str, Any]:
        wavelength, intensity = self._validate_arrays(wavelength_nm, intensity_counts)
        current = np.interp(self.grid, wavelength, intensity)
        if baseline_wavelength_nm is None or baseline_intensity_counts is None:
            if not allow_fallback_baseline:
                raise ValueError("a current no-contact baseline is required for live prediction")
            baseline = self.fallback_baseline.copy()
            baseline_source = "training_fallback_debug_only"
        else:
            baseline_wavelength, baseline_intensity = self._validate_arrays(
                baseline_wavelength_nm,
                baseline_intensity_counts,
            )
            baseline = np.interp(self.grid, baseline_wavelength, baseline_intensity)
            baseline_source = "runtime_no_contact_baseline"

        engineered, full_hybrid = extract_snapshot_feature_vectors(
            self.grid,
            current,
            baseline,
            self.peak_windows,
            self.full_spectrum_bins,
        )
        feature_rows = {
            "engineered": engineered,
            "full_hybrid": full_hybrid,
            "current_shape": full_hybrid,
        }

        def task_matrix(task_id: str) -> np.ndarray:
            payload = self.models[task_id]
            row = feature_rows[payload["feature_set"]]
            return np.asarray(
                [[row[column] for column in payload["feature_columns"]]],
                dtype=float,
            )

        contact = self._predict_with_probabilities(
            self.models["contact_detector"],
            task_matrix("contact_detector"),
        )
        output: dict[str, Any] = {
            "schema_version": "sense_static_hierarchical_prediction_v1",
            "model_bundle_sha256": self.bundle_sha256,
            "baseline_source": baseline_source,
            "contact": contact,
            "position": None,
            "force_level": None,
            "force_model_scope": None,
            "digital_twin": {
                "active": False,
                "position_id": None,
                "center_x": None,
                "center_y": None,
                "force_level": None,
                "deformation_proxy": 0.0,
            },
            "deployment_status": self.bundle["deployment_status"],
            "uncertainty": {
                "review_needed": bool(contact["review_needed"]),
                "reasons": ["contact_probability_requires_review"]
                if contact["review_needed"]
                else [],
                "policy": "diagnostic_only_does_not_change_prediction",
            },
        }
        if contact["label"] != "contact":
            return output

        position = self._predict_with_probabilities(
            self.models["position_classifier"],
            task_matrix("position_classifier"),
        )
        conditioned_payload = self.models.get("position_conditioned_force_classifier")
        if conditioned_payload is not None:
            position_model = conditioned_payload["models_by_position"].get(position["label"])
        else:
            position_model = None
        if position_model is not None:
            force = self._predict_with_probabilities(
                conditioned_payload,
                task_matrix("position_conditioned_force_classifier"),
                model_override=position_model,
            )
            force_model_scope = f"position_conditioned:{position['label']}"
        else:
            force = self._predict_with_probabilities(
                self.models["manual_force_classifier"],
                task_matrix("manual_force_classifier"),
            )
            force_model_scope = "global_manual_fallback"
        coordinates = self.digital_twin_contract["position_coordinates"][position["label"]]
        deformation = float(
            self.digital_twin_contract["deformation_proxy"][force["label"]]
        )
        output["position"] = position
        output["force_level"] = force
        output["force_model_scope"] = force_model_scope
        uncertainty_reasons = []
        if contact["review_needed"]:
            uncertainty_reasons.append("contact_probability_requires_review")
        if position["review_needed"]:
            uncertainty_reasons.append("position_probability_requires_review")
        if force["review_needed"]:
            uncertainty_reasons.append("response_level_probability_requires_review")
        output["uncertainty"] = {
            "review_needed": bool(uncertainty_reasons),
            "reasons": uncertainty_reasons,
            "policy": "diagnostic_only_does_not_change_prediction",
        }
        output["digital_twin"] = {
            "active": True,
            "position_id": position["label"],
            "center_x": float(coordinates[0]),
            "center_y": float(coordinates[1]),
            "force_level": force["label"],
            "deformation_proxy": deformation,
        }
        return output
