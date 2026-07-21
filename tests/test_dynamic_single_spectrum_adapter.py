from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from src.hybrid_spectrum.dynamic_sequence_dataset import (
    extract_baseline_relative_frame_features,
)
from src.hybrid_spectrum.dynamic_single_spectrum import (
    baseline_relative_spectral_views,
)
from src.hybrid_spectrum.dynamic_single_spectrum_adapter import (
    DynamicSingleSpectrumShadowAdapter,
    SCHEMA_VERSION,
)
from src.hybrid_spectrum.features import PeakWindow


class DynamicSingleSpectrumAdapterTests(unittest.TestCase):
    def test_saved_bundle_predicts_one_spectrum(self) -> None:
        wavelength = np.linspace(1540.0, 1550.0, 64)
        baseline = 1000.0 + 200.0 * np.exp(-((wavelength - 1545.0) / 0.35) ** 2)
        spectra = np.vstack([baseline, baseline * 0.98, baseline * 0.82, baseline * 0.60])
        peak_windows = [PeakWindow("FBG01", "P22", 1545.0, 0.8)]
        engineered, names, _, _ = extract_baseline_relative_frame_features(
            wavelength, spectra, baseline, peak_windows
        )
        views = baseline_relative_spectral_views(spectra, baseline)
        flattened = views.reshape(len(views), -1)
        contact = ExtraTreesClassifier(n_estimators=8, random_state=1).fit(
            flattened, ["no_contact", "no_contact", "contact", "contact"]
        )
        position = ExtraTreesClassifier(n_estimators=8, random_state=2).fit(
            engineered, ["P22", "P22", "P21", "P21"]
        )
        response = ExtraTreesClassifier(n_estimators=8, random_state=3).fit(
            flattened, ["light", "light", "normal", "hard"]
        )
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "models": {
                "contact": contact,
                "position": position,
                "response_level": response,
            },
            "label_order": {
                "contact": ["no_contact", "contact"],
                "position": ["P21", "P22"],
                "response_level": ["light", "normal", "hard"],
            },
            "peak_windows": peak_windows,
            "frame_feature_names": names,
            "spectral_view_names": (
                "log_intensity_ratio",
                "normalized_shape_residual",
                "wavelength_derivative_residual",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.joblib"
            joblib.dump(bundle, path)
            result = DynamicSingleSpectrumShadowAdapter(path).predict(
                wavelength, spectra[-1], baseline
            )
        self.assertTrue(result["ok"])
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["deployment_ready"])
        self.assertEqual(result["contact_label"], "contact")
        self.assertEqual(result["response_level"], "hard")


if __name__ == "__main__":
    unittest.main()
