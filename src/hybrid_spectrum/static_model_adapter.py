"""Runtime adapter from one full spectrum to digital-twin position and level."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .sense_static_dataset import extract_snapshot_feature_vectors
from .session_level_calibration import extract_response_core_features
from .spatial_fingerprint import spatial_fingerprint_from_engineered


class StaticSpectralPredictor:
    """Load a trained hierarchical bundle and expose a stable prediction API."""

    # The original contact model only sees the absolute current spectral shape.
    # Live sessions can shift that domain even when the sensor is released.  Keep
    # it as supporting evidence, but require a baseline-relative change before it
    # may activate the digital twin.  The strong thresholds sit just outside the
    # observed post-release no-contact envelope in the current data set.
    CONTACT_GATE_THRESHOLDS = {
        "strong_residual_rms": 0.040,
        "strong_residual_peak": 0.200,
        "strong_derivative_energy": 0.00046,
        "strong_shape_correlation_max": 0.99845,
        "strong_secondary_rms_floor": 0.025,
        "supporting_residual_rms": 0.018,
        "supporting_residual_peak": 0.100,
        "supporting_derivative_energy": 0.00009,
        "supporting_shape_correlation_max": 0.99970,
    }

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
        confidence_source = (
            str(
                model_payload.get("confidence_source")
                or getattr(model, "confidence_source", "uncalibrated_predict_proba")
            )
            if probabilities
            else "unavailable"
        )
        result = {
            "label": predicted,
            "confidence": confidence,
            "margin": margin,
            "probabilities": probabilities,
            "confidence_source": confidence_source,
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
        if hasattr(model, "predict_diagnostics"):
            result["ensemble_diagnostics"] = model.predict_diagnostics(matrix)[0]
        return result

    @classmethod
    def _baseline_relative_contact_evidence(
        cls,
        engineered: dict[str, float],
    ) -> dict[str, Any]:
        """Summarize physically meaningful current-versus-baseline change."""

        thresholds = cls.CONTACT_GATE_THRESHOLDS

        def finite_value(name: str, fallback: float) -> float:
            value = float(engineered.get(name, fallback))
            return value if np.isfinite(value) else fallback

        residual_rms = finite_value("global_normalized_residual_rms", 0.0)
        residual_peak = finite_value("global_normalized_residual_peak", 0.0)
        derivative_energy = finite_value("global_derivative_residual_energy", 0.0)
        shape_correlation = finite_value("global_shape_correlation", 1.0)
        secondary_floor_passed = (
            residual_rms >= thresholds["strong_secondary_rms_floor"]
        )
        strong_reasons = []
        if residual_rms >= thresholds["strong_residual_rms"]:
            strong_reasons.append("residual_rms")
        if secondary_floor_passed and residual_peak >= thresholds["strong_residual_peak"]:
            strong_reasons.append("residual_peak")
        if (
            secondary_floor_passed
            and derivative_energy >= thresholds["strong_derivative_energy"]
        ):
            strong_reasons.append("derivative_energy")
        if (
            secondary_floor_passed
            and shape_correlation <= thresholds["strong_shape_correlation_max"]
        ):
            strong_reasons.append("shape_correlation")

        supporting_reasons = []
        supporting_floor_passed = residual_rms >= thresholds["supporting_residual_rms"]
        if supporting_floor_passed and residual_peak >= thresholds["supporting_residual_peak"]:
            supporting_reasons.append("residual_peak")
        if (
            supporting_floor_passed
            and derivative_energy >= thresholds["supporting_derivative_energy"]
        ):
            supporting_reasons.append("derivative_energy")
        if (
            supporting_floor_passed
            and shape_correlation <= thresholds["supporting_shape_correlation_max"]
        ):
            supporting_reasons.append("shape_correlation")

        return {
            "strong_contact_evidence": bool(strong_reasons),
            "supporting_contact_evidence": bool(supporting_reasons),
            "strong_reasons": strong_reasons,
            "supporting_reasons": supporting_reasons,
            "metrics": {
                "global_normalized_residual_rms": residual_rms,
                "global_normalized_residual_peak": residual_peak,
                "global_derivative_residual_energy": derivative_energy,
                "global_shape_correlation": shape_correlation,
            },
            "thresholds": dict(thresholds),
            "semantics": "current_full_spectrum_relative_to_runtime_no_contact_baseline",
        }

    @staticmethod
    def _resolve_contact_decision(
        model_contact: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Fuse the legacy classifier with a conservative physical change gate."""

        model_snapshot = {
            "label": model_contact.get("label"),
            "confidence": model_contact.get("confidence"),
            "margin": model_contact.get("margin"),
            "probabilities": dict(model_contact.get("probabilities") or {}),
            "confidence_source": model_contact.get("confidence_source"),
        }
        model_says_contact = model_snapshot["label"] == "contact"
        strong = bool(evidence.get("strong_contact_evidence"))
        supporting = bool(evidence.get("supporting_contact_evidence"))
        resolved = dict(model_contact)

        if strong:
            resolved_label = "contact"
            decision_source = (
                "model_and_baseline_relative_spectral_change"
                if model_says_contact
                else "baseline_relative_spectral_change_gate"
            )
        elif model_says_contact and supporting:
            resolved_label = "contact"
            decision_source = "model_supported_by_baseline_relative_change"
        else:
            resolved_label = "no_contact"
            decision_source = (
                "model_contact_suppressed_without_baseline_change"
                if model_says_contact
                else "model_no_contact_with_baseline_gate"
            )

        disagreement = resolved_label != model_snapshot["label"]
        resolved.update(
            {
                "label": resolved_label,
                "decision_source": decision_source,
                "baseline_relative_evidence": evidence,
                "raw_model_prediction": model_snapshot,
                "model_rule_disagreement": disagreement,
            }
        )
        if disagreement:
            # A threshold decision is not a calibrated probability.  Clear the
            # legacy probability rather than showing a contradictory 100% value.
            resolved.update(
                {
                    "confidence": None,
                    "margin": None,
                    "probabilities": {},
                    "confidence_source": "baseline_relative_physical_gate_not_probability",
                    "probability_calibrated": False,
                    "review_needed": False,
                }
            )
        elif resolved_label == "contact" and supporting:
            resolved["confidence_source"] = (
                f"{resolved.get('confidence_source', 'unavailable')}_with_baseline_evidence"
            )
        return resolved

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
        response_calibration_features = extract_response_core_features(engineered)
        feature_rows = {
            "engineered": engineered,
            "full_hybrid": full_hybrid,
            "current_shape": full_hybrid,
            "spatial_fingerprint": spatial_fingerprint_from_engineered(engineered),
        }

        def task_matrix(task_id: str) -> np.ndarray:
            payload = self.models[task_id]
            row = feature_rows[payload["feature_set"]]
            return np.asarray(
                [[row[column] for column in payload["feature_columns"]]],
                dtype=float,
            )

        model_contact = self._predict_with_probabilities(
            self.models["contact_detector"],
            task_matrix("contact_detector"),
        )
        contact_evidence = self._baseline_relative_contact_evidence(engineered)
        contact = self._resolve_contact_decision(model_contact, contact_evidence)
        contact_diagnostic_flags = (
            ["contact_model_rule_disagreement"]
            if contact["model_rule_disagreement"]
            else []
        )
        output: dict[str, Any] = {
            "schema_version": "sense_static_hierarchical_prediction_v1",
            "model_bundle_sha256": self.bundle_sha256,
            "baseline_source": baseline_source,
            "response_calibration_features": response_calibration_features,
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
                "diagnostic_flags": contact_diagnostic_flags,
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
            "diagnostic_flags": contact_diagnostic_flags,
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
