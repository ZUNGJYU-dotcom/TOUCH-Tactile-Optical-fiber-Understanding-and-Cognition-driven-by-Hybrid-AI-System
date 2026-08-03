from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_spectrum.all_source_runtime_adapter import (  # noqa: E402
    AllSourceOpticalForceAdapter,
    POSITION_ORDER,
)
from hybrid_spectrum.features import load_peak_windows  # noqa: E402


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "ordinary_fbg_optical_only_force_candidate.joblib"
)
PEAK_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml"


class _FixedClassifier:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.classes_ = np.asarray(tuple(probabilities), dtype=object)
        values = np.asarray(tuple(probabilities.values()), dtype=float)
        self._probabilities = values / np.sum(values)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(
            self._probabilities[None, :], len(features), axis=0
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        label = self.classes_[int(np.argmax(self._probabilities))]
        return np.repeat(label, len(features))


class _FixedRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


def _position_probabilities(label: str, confidence: float = 0.92) -> dict[str, float]:
    remainder = (1.0 - confidence) / (len(POSITION_ORDER) - 1)
    return {
        position_id: confidence if position_id == label else remainder
        for position_id in POSITION_ORDER
    }


@pytest.fixture()
def runtime_adapter() -> tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray]:
    if not MODEL_PATH.exists():
        pytest.skip("all-source runtime candidate is unavailable")
    adapter = AllSourceOpticalForceAdapter.from_paths(
        MODEL_PATH,
        PEAK_CONFIG_PATH,
        runtime_recovery_config={
            "enabled": True,
            "minimum_contact_baseline_distance": 0.006,
            "max_shape_motion_rms": 0.0045,
            "max_common_gain_motion": 0.003,
            "activity_shape_motion_rms": 0.006,
            "activity_common_gain_motion": 0.006,
            "quiet_hold_sec": 0.4,
            "minimum_quiet_physical_frames": 2,
            "contact_arm_physical_frames": 1,
        },
        runtime_gate_config={
            "enabled": True,
            "contact_probability_off": 0.55,
            "position_confidence_min": 0.45,
            "position_margin_min": 0.08,
            "visual_position_fallback_enabled": True,
            "visual_position_confidence_min": 0.18,
            "visual_position_margin_min": 0.015,
            "position_probability_ema_alpha": 0.55,
            "position_hold_sec": 0.75,
            "position_switch_frames": 2,
            "require_baseline_separation": True,
            "baseline_release_distance": 0.005,
            "release_near_baseline_frames": 2,
            "release_ambiguous_frames": 2,
            "ambiguous_quiet_release_sec": 1.0,
            "activity_memory_sec": 1.25,
            "activity_shape_motion_rms": 0.006,
            "activity_common_gain_motion": 0.006,
        },
    )
    windows = load_peak_windows(PEAK_CONFIG_PATH)
    wavelength = np.linspace(
        min(window.center_nm - window.half_width_nm for window in windows) - 0.5,
        max(window.center_nm + window.half_width_nm for window in windows) + 0.5,
        512,
    )
    baseline = np.full_like(wavelength, 3500.0)
    for window in windows:
        baseline += 1800.0 * np.exp(
            -0.5 * ((wavelength - window.center_nm) / 0.14) ** 2
        )
    adapter.contact_model = _FixedClassifier(
        {"contact": 0.98, "no_contact": 0.02}
    )
    adapter.position_model = _FixedClassifier(_position_probabilities("P11"))
    adapter.force_model = _FixedRegressor(2.0)
    adapter.set_baseline(wavelength, baseline)
    return adapter, wavelength, baseline


def test_small_stable_offset_cannot_activate_contact(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    result = adapter.update(
        wavelength,
        baseline * 1.003,
        source_timestamp_sec=0.2,
    )

    assert result["ok"] is True
    assert result["contact"]["raw_model_label"] == "contact"
    assert result["runtime_contact_gate"]["baseline_separated"] is False
    assert result["contact"]["label"] == "no_contact"
    assert result["position"]["label"] is None
    assert result["estimated_force_fz_n"] == 0.0


def test_first_physical_press_is_compared_with_baseline(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    adapter.position_model = _FixedClassifier(_position_probabilities("P12"))
    result = adapter.update(
        wavelength,
        baseline * 0.985,
        source_timestamp_sec=0.2,
    )

    assert result["contact"]["label"] == "contact"
    assert result["position"]["label"] == "P12"
    assert result["runtime_contact_gate"]["fresh_spectral_activity"] is True
    assert result["runtime_contact_gate"]["baseline_separated"] is True


def test_near_baseline_release_clears_stale_model_latch(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    pressed = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.2,
    )
    assert pressed["contact"]["label"] == "contact"

    adapter.update(wavelength, baseline, source_timestamp_sec=0.4)
    adapter.update(wavelength, baseline, source_timestamp_sec=0.6)
    released = adapter.update(wavelength, baseline, source_timestamp_sec=0.8)

    assert released["contact"]["raw_model_label"] == "contact"
    assert released["runtime_contact_gate"]["near_runtime_baseline"] is True
    assert released["contact"]["label"] == "no_contact"
    assert released["estimated_force_fz_n"] == 0.0


def test_position_requires_two_frames_before_switching(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    first = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.2,
    )
    assert first["position"]["label"] == "P11"

    adapter.position_model = _FixedClassifier(_position_probabilities("P33"))
    transient = adapter.update(
        wavelength,
        baseline * 0.975,
        source_timestamp_sec=0.4,
    )
    switched = adapter.update(
        wavelength,
        baseline * 0.97,
        source_timestamp_sec=0.6,
    )

    assert transient["position"]["label"] == "P11"
    assert transient["position"]["held_from_previous_frame"] is True
    assert transient["runtime_contact_gate"]["pending_position_id"] == "P33"
    assert switched["position"]["label"] == "P33"
    assert switched["position"]["held_from_previous_frame"] is False


def test_baseline_reset_clears_contact_position_and_force(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    pressed = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.2,
    )
    assert pressed["contact"]["label"] == "contact"

    adapter.set_baseline(wavelength, baseline)
    reset = adapter.update(
        wavelength,
        baseline,
        source_timestamp_sec=0.4,
    )

    assert reset["contact"]["label"] == "no_contact"
    assert reset["position"]["label"] is None
    assert reset["estimated_force_fz_n"] == 0.0


def test_gradual_press_can_activate_after_accumulated_baseline_change(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    result = None
    for index in range(1, 9):
        result = adapter.update(
            wavelength,
            baseline * (1.0 - 0.001 * index),
            source_timestamp_sec=0.2 * index,
        )

    assert result is not None
    assert result["runtime_contact_gate"]["baseline_separated"] is True
    assert result["runtime_contact_gate"]["slow_baseline_departure"] is True
    assert result["contact"]["label"] == "contact"


def test_position_confidence_alone_cannot_activate_contact(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    recovery = {
        "shape_motion_rms": 0.0002,
        "common_gain_motion": 0.0002,
        "baseline_distance": 0.02,
        "baseline_distance_growth": 0.0,
        "slow_baseline_departure": False,
        "suppress_contact": False,
    }
    monkeypatch.setattr(
        adapter._runtime_baseline_recovery,
        "observe",
        lambda *_args, **_kwargs: (recovery, None),
    )

    result = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.2,
    )

    assert result["contact"]["raw_model_label"] == "contact"
    assert result["runtime_contact_gate"]["spatially_credible"] is True
    assert result["runtime_contact_gate"]["fresh_spectral_activity"] is False
    assert result["runtime_contact_gate"]["slow_baseline_departure"] is False
    assert result["contact"]["label"] == "no_contact"
    assert result["position"]["label"] is None
    assert result["estimated_force_fz_n"] == 0.0


def test_low_position_confidence_does_not_release_held_contact(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    pressed_spectrum = baseline * 0.98
    first = adapter.update(
        wavelength,
        pressed_spectrum,
        source_timestamp_sec=0.2,
    )
    assert first["contact"]["label"] == "contact"

    adapter.position_model = _FixedClassifier(
        _position_probabilities("P32", confidence=0.22)
    )
    held = None
    for frame_index in range(1, 5):
        held = adapter.update(
            wavelength,
            pressed_spectrum,
            source_timestamp_sec=0.2 + 0.2 * frame_index,
        )

    assert held is not None
    assert held["runtime_contact_gate"]["ambiguous_quiet_frames"] >= 2
    assert held["runtime_contact_gate"]["ambiguous_quiet_is_release_evidence"] is False
    assert held["contact"]["label"] == "contact"


def test_prolonged_quiet_ambiguous_residual_releases_contact(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    pressed_spectrum = baseline * 0.98
    pressed = adapter.update(
        wavelength,
        pressed_spectrum,
        source_timestamp_sec=0.2,
    )
    assert pressed["contact"]["label"] == "contact"

    adapter.position_model = _FixedClassifier(
        _position_probabilities("P32", confidence=0.22)
    )
    released = None
    release_evidence_seen = False
    for timestamp in (0.4, 0.8, 1.2, 1.6, 1.8):
        released = adapter.update(
            wavelength,
            pressed_spectrum,
            source_timestamp_sec=timestamp,
        )
        release_evidence_seen = bool(
            release_evidence_seen
            or released["runtime_contact_gate"][
                "ambiguous_quiet_is_release_evidence"
            ]
        )

    assert released is not None
    assert released["runtime_contact_gate"]["fresh_spectral_activity"] is False
    assert released["runtime_contact_gate"]["spectral_activity_recent"] is False
    assert release_evidence_seen is True
    assert released["runtime_contact_gate"]["quiet_no_contact_hint"] is True
    assert released["contact"]["label"] == "no_contact"
    assert released["digital_twin"]["active"] is False


def test_provisional_visual_position_does_not_follow_quiet_jitter(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    adapter.position_model = _FixedClassifier(
        _position_probabilities("P32", confidence=0.22)
    )
    first = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.2,
    )
    assert first["position"]["visual_label"] == "P32"

    adapter.position_model = _FixedClassifier(
        _position_probabilities("P11", confidence=0.22)
    )
    quiet = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.4,
    )

    assert quiet["runtime_contact_gate"]["fresh_spectral_activity"] is False
    assert quiet["position"]["visual_label"] == "P32"
    assert quiet["digital_twin"]["position_id"] == "P32"


def test_verified_contact_can_use_provisional_visual_position(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    adapter.position_model = _FixedClassifier(
        _position_probabilities("P32", confidence=0.22)
    )

    result = adapter.update(
        wavelength,
        baseline * 0.98,
        source_timestamp_sec=0.2,
    )

    assert result["contact"]["label"] == "contact"
    assert result["position"]["accepted"] is False
    assert result["position"]["label"] is None
    assert result["position"]["visual_label"] == "P32"
    assert result["position"]["visual_fallback_used"] is True
    assert result["digital_twin"]["active"] is True
    assert result["digital_twin"]["position_id"] == "P32"
    assert (
        result["digital_twin"]["position_source"]
        == "provisional_low_confidence_position"
    )


def test_runtime_recovered_baseline_is_exposed_once_for_bridge_commit(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    recovered = baseline * 1.002
    recovery = {
        "shape_motion_rms": 0.0004,
        "common_gain_motion": 0.0003,
        "baseline_distance": 0.002,
        "suppress_contact": True,
        "stable_release_physical_frames": 5,
        "quiet_elapsed_sec": 1.2,
        "policy": "test_release_recovery",
    }
    monkeypatch.setattr(
        adapter._runtime_baseline_recovery,
        "observe",
        lambda *_args, **_kwargs: (recovery, recovered.copy()),
    )

    result = adapter.update(
        wavelength,
        baseline,
        source_timestamp_sec=0.2,
    )
    pending = adapter.consume_pending_runtime_baseline_update()

    assert result["runtime_contact_gate"]["runtime_reference_reanchored"] is True
    assert pending is not None
    assert np.allclose(pending["wavelength_nm"], wavelength)
    assert np.allclose(pending["intensity"], recovered)
    assert pending["sample_count"] == 5
    assert pending["span_sec"] == pytest.approx(1.2)
    assert pending["policy"] == "test_release_recovery"
    assert adapter.consume_pending_runtime_baseline_update() is None


def test_explicit_baseline_replacement_discards_pending_runtime_candidate(
    runtime_adapter: tuple[AllSourceOpticalForceAdapter, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, wavelength, baseline = runtime_adapter
    recovery = {
        "shape_motion_rms": 0.0,
        "common_gain_motion": 0.0,
        "baseline_distance": 0.0,
        "suppress_contact": True,
        "stable_release_physical_frames": 4,
        "quiet_elapsed_sec": 1.0,
        "policy": "test_release_recovery",
    }
    monkeypatch.setattr(
        adapter._runtime_baseline_recovery,
        "observe",
        lambda *_args, **_kwargs: (recovery, baseline * 1.001),
    )
    adapter.update(wavelength, baseline, source_timestamp_sec=0.2)

    adapter.set_baseline(wavelength, baseline)

    assert adapter.consume_pending_runtime_baseline_update() is None
