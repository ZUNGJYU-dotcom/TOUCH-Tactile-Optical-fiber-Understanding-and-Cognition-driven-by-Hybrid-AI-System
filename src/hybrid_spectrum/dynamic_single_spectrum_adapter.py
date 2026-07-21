"""Runtime adapter for a low-latency, one-physical-spectrum shadow candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .dynamic_sequence_dataset import extract_baseline_relative_frame_features
from .dynamic_single_spectrum import baseline_relative_spectral_views


SCHEMA_VERSION = "dynamic_single_spectrum_fast_candidate_v1"


def _aligned_probability(model: Any, values: np.ndarray, labels: list[str]) -> np.ndarray:
    probability = np.asarray(model.predict_proba(values), dtype=float)
    aligned = np.zeros((len(values), len(labels)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        label_text = str(label)
        if label_text not in labels:
            raise ValueError(f"model returned unexpected class: {label_text}")
        aligned[:, labels.index(label_text)] = probability[:, source_index]
    return aligned


def load_dynamic_single_spectrum_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dynamic single-spectrum candidate bundle")
    required = {
        "models",
        "label_order",
        "peak_windows",
        "frame_feature_names",
        "spectral_view_names",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"candidate bundle is missing fields: {missing}")
    return bundle


class DynamicSingleSpectrumShadowAdapter:
    """Predict contact, position, and response from one measured spectrum."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path).resolve()
        self.bundle = load_dynamic_single_spectrum_bundle(self.model_path)

    def predict(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        baseline_spectrum: np.ndarray,
    ) -> dict[str, Any]:
        wavelength = np.asarray(wavelength_nm, dtype=float)
        current = np.asarray(spectrum, dtype=float)
        baseline = np.asarray(baseline_spectrum, dtype=float)
        if current.shape != wavelength.shape or baseline.shape != wavelength.shape:
            raise ValueError("wavelength, spectrum, and baseline must share one dimension")
        engineered, feature_names, _, _ = extract_baseline_relative_frame_features(
            wavelength,
            current,
            baseline,
            self.bundle["peak_windows"],
        )
        expected_names = tuple(self.bundle["frame_feature_names"])
        if tuple(feature_names) != expected_names:
            raise ValueError("runtime engineered feature order differs from training")
        spectral = baseline_relative_spectral_views(current, baseline)
        expected_views = tuple(self.bundle["spectral_view_names"])
        if spectral.shape[1] != len(expected_views):
            raise ValueError("runtime spectral view count differs from training")
        flattened = spectral.reshape(len(spectral), -1)

        models = self.bundle["models"]
        label_order = self.bundle["label_order"]
        contact_labels = list(label_order["contact"])
        position_labels = list(label_order["position"])
        response_labels = list(label_order["response_level"])
        contact_probability = _aligned_probability(
            models["contact"], flattened, contact_labels
        )[0]
        position_probability = _aligned_probability(
            models["position"], engineered, position_labels
        )[0]
        response_probability = _aligned_probability(
            models["response_level"], flattened, response_labels
        )[0]

        contact_index = int(np.argmax(contact_probability))
        position_index = int(np.argmax(position_probability))
        response_index = int(np.argmax(response_probability))
        contact_label = contact_labels[contact_index]
        is_contact = contact_label == "contact"
        return {
            "ok": True,
            "candidate_only": True,
            "deployment_ready": False,
            "schema_version": SCHEMA_VERSION,
            "input_semantics": "one_baseline_relative_physical_spectrum",
            "contact_label": contact_label,
            "contact_confidence": float(contact_probability[contact_index]),
            "position_label": position_labels[position_index] if is_contact else "",
            "position_confidence": float(position_probability[position_index]),
            "response_level": response_labels[response_index] if is_contact else "no_contact",
            "response_confidence": float(response_probability[response_index]),
            "probabilities": {
                "contact": dict(zip(contact_labels, contact_probability.tolist())),
                "position": dict(zip(position_labels, position_probability.tolist())),
                "response_level": dict(zip(response_labels, response_probability.tolist())),
            },
            "response_level_semantics": "approximate_manual_response_level_not_force_N",
        }
