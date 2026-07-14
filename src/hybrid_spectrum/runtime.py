"""Runtime baseline, spectral QA, and model-deployment guards."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .features import (
    GLOBAL_ARRAY_CANDIDATE_IDS,
    PeakWindow,
    extract_frame_features,
)
from .dataset import SpectrumSegment


GLOBAL_ARRAY_RECOGNITION_SCOPE = "global_3x3_hybrid_spectral_fingerprint"
GLOBAL_ARRAY_REQUIRED_FEATURE_SUFFIXES = (
    "quality_fused_shift_pm",
    "height_ratio",
    "area_ratio",
    "delta_fwhm_pm",
    "shape_correlation",
    "peak_valid",
)


def feature_schema_sha256(columns: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def has_global_array_feature_scope(
    columns: Iterable[str],
    candidate_ids: Iterable[str] = GLOBAL_ARRAY_CANDIDATE_IDS,
) -> bool:
    """Require mixed wavelength, intensity, area, shape, and QA features for all peaks."""

    normalized = {str(column).lower() for column in columns}
    return all(
        all(
            f"{candidate_id.lower()}_{suffix}" in normalized
            for suffix in GLOBAL_ARRAY_REQUIRED_FEATURE_SUFFIXES
        )
        for candidate_id in candidate_ids
    )


@dataclass(frozen=True)
class FrameQualityDecision:
    status: str
    spectral_input_valid: bool
    peak_valid_count: int
    baseline_peak_valid_count: int
    fused_reliable_count: int
    cross_correlation_reliable_count: int
    fused_common_mode_shift_pm: float | None
    maximum_fused_corrected_shift_pm: float | None
    drift_status: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PredictionReadinessDecision:
    prediction_allowed: bool
    status: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    model_id: str | None
    evaluation_validity: str | None
    feature_schema_match: bool
    configuration_match: bool
    model_artifact_integrity_verified: bool
    physical_mapping_approved: bool
    real_contact_acceptance_passed: bool
    real_model_deployable: bool
    feature_provenance_verified: bool
    held_out_session_evaluation_passed: bool
    global_array_feature_scope_verified: bool


@dataclass(frozen=True)
class BaselineCompatibilityDecision:
    compatible: bool
    status: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    reference_trial_id: str
    reference_device_id: str | None
    current_device_id: str | None
    reference_integration_ms: float | None
    current_integration_ms: float | None
    reference_integrity_verified: bool
    reference_baseline_quality_status: str | None
    reference_backend_session_started_at: float | None
    current_backend_session_started_at: float | None


class OnlineBaselineAccumulator:
    """Collect a monotonic set of unique no-contact frames for one session."""

    def __init__(self, minimum_frames: int = 60, expected_points: int = 512) -> None:
        if minimum_frames < 3:
            raise ValueError("minimum_frames must be at least 3")
        self.minimum_frames = int(minimum_frames)
        self.expected_points = int(expected_points)
        self._wavelength_nm: np.ndarray | None = None
        self._spectra: list[np.ndarray] = []
        self._frame_ids: list[int] = []
        self._timestamps: list[float] = []
        self.duplicate_frame_count = 0

    @property
    def accepted_frame_count(self) -> int:
        return len(self._spectra)

    @property
    def ready(self) -> bool:
        return self.accepted_frame_count >= self.minimum_frames

    @property
    def wavelength_nm(self) -> np.ndarray:
        if self._wavelength_nm is None:
            raise RuntimeError("baseline wavelength grid is not available")
        return self._wavelength_nm.copy()

    @property
    def baseline_spectrum(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("baseline is not ready")
        return np.median(np.vstack(self._spectra), axis=0)

    @property
    def baseline_noise_counts(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("baseline is not ready")
        spectra = np.vstack(self._spectra)
        center = np.median(spectra, axis=0)
        return 1.4826 * np.median(np.abs(spectra - center), axis=0)

    def add_frame(
        self,
        frame_id: int,
        timestamp: float,
        wavelength_nm: np.ndarray,
        intensity_counts: np.ndarray,
    ) -> bool:
        frame_id = int(frame_id)
        timestamp = float(timestamp)
        wavelength = np.asarray(wavelength_nm, dtype=float)
        intensity = np.asarray(intensity_counts, dtype=float)
        if self._frame_ids and frame_id == self._frame_ids[-1]:
            self.duplicate_frame_count += 1
            return False
        if self._frame_ids and frame_id < self._frame_ids[-1]:
            raise ValueError("frame_id moved backwards")
        if self._timestamps and timestamp <= self._timestamps[-1]:
            raise ValueError("timestamp is not strictly increasing")
        if wavelength.shape != intensity.shape or wavelength.size != self.expected_points:
            raise ValueError("spectrum does not match the expected frame shape")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(intensity)):
            raise ValueError("spectrum contains non-finite values")
        if not np.all(np.diff(wavelength) > 0):
            raise ValueError("wavelength grid is not strictly increasing")
        if self._wavelength_nm is None:
            self._wavelength_nm = wavelength.copy()
        elif not np.allclose(wavelength, self._wavelength_nm, atol=1.0e-9, rtol=0.0):
            raise ValueError("wavelength grid changed during baseline collection")
        self._frame_ids.append(frame_id)
        self._timestamps.append(timestamp)
        self._spectra.append(intensity.copy())
        return True


def evaluate_baseline_compatibility(
    reference: SpectrumSegment,
    *,
    current_trial_id: str,
    current_wavelength_nm: np.ndarray,
    current_frame_ids: np.ndarray,
    current_timestamps: np.ndarray,
    current_context: dict[str, Any],
    integration_tolerance_ms: float = 1.0e-6,
    maximum_baseline_age_sec: float = 120.0,
    maximum_backend_session_start_delta_sec: float = 1.0,
    require_integrity_manifest: bool = True,
    require_operator_no_contact_attestation: bool = True,
    require_baseline_quality_pass: bool = True,
) -> BaselineCompatibilityDecision:
    """Reject a baseline that is not from the same acquisition context."""

    blockers: list[str] = []
    warnings: list[str] = []
    reference_metadata = reference.metadata
    reference_device = reference_metadata.get("device_id")
    current_device = current_context.get("device_id")
    reference_integration = reference_metadata.get("integration_ms")
    current_integration = current_context.get("integration_ms")
    reference_source = reference_metadata.get("data_source")
    current_source = current_context.get("source")
    reference_profile = reference_metadata.get("spectrum_peak_profile")
    current_profile = current_context.get("spectrum_peak_profile")
    reference_quality_status = reference_metadata.get("baseline_quality_status")
    reference_session_started_at = reference_metadata.get(
        "backend_session_started_at_epoch"
    )
    current_session_started_at = current_context.get("backend_session_started_at_epoch")
    if current_session_started_at is None:
        try:
            current_session_started_at = float(current_context["timestamp"]) - float(
                current_context["relative_time_sec"]
            )
        except (KeyError, TypeError, ValueError):
            current_session_started_at = None

    if reference.phase != "no_contact" or reference.label != "no_contact":
        blockers.append("baseline_reference_is_not_no_contact")
    if require_integrity_manifest and not reference.integrity_verified:
        blockers.append("baseline_integrity_manifest_not_verified")
    if require_operator_no_contact_attestation and not bool(
        reference_metadata.get("operator_attested_no_contact", False)
    ):
        blockers.append("baseline_no_contact_not_operator_attested")
    if require_baseline_quality_pass and reference_quality_status != "pass":
        blockers.append("baseline_quality_not_passed")
    if reference.trial_id != str(current_trial_id):
        blockers.append("baseline_trial_id_mismatch")
    if reference.baseline_spectrum is None:
        blockers.append("baseline_spectrum_missing")
    if not reference_device or not current_device:
        blockers.append("device_identity_missing")
    elif str(reference_device) != str(current_device):
        blockers.append("baseline_device_mismatch")
    if not reference_source or not current_source:
        blockers.append("acquisition_source_missing")
    elif str(reference_source) != str(current_source):
        blockers.append("baseline_acquisition_source_mismatch")
    try:
        integration_delta = abs(float(reference_integration) - float(current_integration))
    except (TypeError, ValueError):
        blockers.append("integration_time_missing_or_invalid")
    else:
        if integration_delta > integration_tolerance_ms:
            blockers.append("baseline_integration_time_mismatch")
    if reference_profile and current_profile and str(reference_profile) != str(current_profile):
        blockers.append("baseline_peak_profile_mismatch")
    elif not reference_profile or not current_profile:
        warnings.append("peak_profile_identity_incomplete")
    if reference_session_started_at is None or current_session_started_at is None:
        blockers.append("backend_session_identity_missing")
    elif (
        abs(float(reference_session_started_at) - float(current_session_started_at))
        > maximum_backend_session_start_delta_sec
    ):
        blockers.append("baseline_backend_session_mismatch")

    wavelength = np.asarray(current_wavelength_nm, dtype=float)
    frame_ids = np.asarray(current_frame_ids, dtype=np.int64)
    timestamps = np.asarray(current_timestamps, dtype=float)
    if wavelength.shape != reference.wavelength_nm.shape or not np.allclose(
        wavelength,
        reference.wavelength_nm,
        atol=1.0e-9,
        rtol=0.0,
    ):
        blockers.append("baseline_wavelength_grid_mismatch")
    if frame_ids.size == 0 or timestamps.size == 0:
        blockers.append("current_capture_has_no_frames")
    else:
        if int(frame_ids[0]) <= int(reference.frame_ids[-1]):
            blockers.append("baseline_frame_sequence_not_continuous")
        if float(timestamps[0]) <= float(reference.timestamps[-1]):
            blockers.append("baseline_timestamp_not_before_current_capture")
        elif (
            float(timestamps[0]) - float(reference.timestamps[-1])
            > maximum_baseline_age_sec
        ):
            blockers.append("baseline_reference_stale")

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    return BaselineCompatibilityDecision(
        compatible=not blockers,
        status="compatible" if not blockers else "incompatible",
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        reference_trial_id=reference.trial_id,
        reference_device_id=(
            str(reference_device) if reference_device is not None else None
        ),
        current_device_id=(str(current_device) if current_device is not None else None),
        reference_integration_ms=(
            float(reference_integration) if reference_integration is not None else None
        ),
        current_integration_ms=(
            float(current_integration) if current_integration is not None else None
        ),
        reference_integrity_verified=bool(reference.integrity_verified),
        reference_baseline_quality_status=(
            str(reference_quality_status) if reference_quality_status is not None else None
        ),
        reference_backend_session_started_at=(
            float(reference_session_started_at)
            if reference_session_started_at is not None
            else None
        ),
        current_backend_session_started_at=(
            float(current_session_started_at)
            if current_session_started_at is not None
            else None
        ),
    )


def evaluate_spectral_frame(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    baseline_spectrum: np.ndarray,
    peak_windows: Iterable[PeakWindow],
    *,
    minimum_valid_peaks: int = 9,
    expected_no_contact: bool = False,
    maximum_no_contact_common_mode_shift_pm: float = 10.0,
    maximum_no_contact_corrected_shift_pm: float = 10.0,
) -> tuple[FrameQualityDecision, dict[str, float]]:
    features = extract_frame_features(
        np.asarray(wavelength_nm, dtype=float),
        np.asarray(spectrum, dtype=float),
        np.asarray(baseline_spectrum, dtype=float),
        peak_windows,
    )
    peak_valid_count = sum(
        bool(value)
        for key, value in features.items()
        if key.endswith("_peak_valid") and not key.endswith("_baseline_peak_valid")
    )
    baseline_peak_valid_count = sum(
        bool(value)
        for key, value in features.items()
        if key.endswith("_baseline_peak_valid")
    )
    fused_reliable_count = sum(
        bool(value)
        for key, value in features.items()
        if key.endswith("_quality_fused_shift_reliable")
    )
    cross_reliable_count = sum(
        bool(value)
        for key, value in features.items()
        if key.endswith("_cross_correlation_reliable")
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if peak_valid_count < minimum_valid_peaks:
        blockers.append("insufficient_valid_peaks")
    if baseline_peak_valid_count < minimum_valid_peaks:
        blockers.append("insufficient_valid_baseline_peaks")
    if fused_reliable_count < minimum_valid_peaks:
        blockers.append("insufficient_fused_reliable_peaks")
    if cross_reliable_count < minimum_valid_peaks:
        warnings.append("supporting_cross_correlation_coverage_low")

    common_mode_value = features.get("global_fused_common_mode_shift_pm")
    fused_common_mode_shift_pm = (
        float(common_mode_value)
        if common_mode_value is not None and np.isfinite(common_mode_value)
        else None
    )
    corrected_values = [
        float(value)
        for key, value in features.items()
        if key.endswith("_fused_common_mode_corrected_shift_pm")
        and np.isfinite(value)
    ]
    maximum_corrected = (
        float(max(abs(value) for value in corrected_values))
        if corrected_values
        else None
    )
    drift_status = "not_assessed_during_unknown_contact_state"
    if expected_no_contact:
        drift_status = "pass"
        if fused_common_mode_shift_pm is None:
            blockers.append("no_contact_common_mode_unavailable")
            drift_status = "fail"
        elif abs(fused_common_mode_shift_pm) > maximum_no_contact_common_mode_shift_pm:
            blockers.append("no_contact_common_mode_shift_above_gate")
            drift_status = "fail"
        if maximum_corrected is None:
            blockers.append("no_contact_corrected_shift_unavailable")
            drift_status = "fail"
        elif maximum_corrected > maximum_no_contact_corrected_shift_pm:
            blockers.append("no_contact_candidate_residual_above_gate")
            drift_status = "fail"

    status = "fail" if blockers else ("warning" if warnings else "pass")
    return (
        FrameQualityDecision(
            status=status,
            spectral_input_valid=not blockers,
            peak_valid_count=peak_valid_count,
            baseline_peak_valid_count=baseline_peak_valid_count,
            fused_reliable_count=fused_reliable_count,
            cross_correlation_reliable_count=cross_reliable_count,
            fused_common_mode_shift_pm=fused_common_mode_shift_pm,
            maximum_fused_corrected_shift_pm=maximum_corrected,
            drift_status=drift_status,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        ),
        features,
    )


def evaluate_prediction_readiness(
    model_bundle: dict[str, Any],
    current_feature_columns: Iterable[str],
    frame_quality: FrameQualityDecision,
    *,
    baseline_ready: bool,
    physical_mapping_approved: bool,
    current_config_sha256: str | None,
    model_artifact_integrity_verified: bool,
) -> PredictionReadinessDecision:
    current_feature_columns = list(current_feature_columns)
    metadata = model_bundle.get("metadata", {}) or {}
    model_id = metadata.get("model_id")
    evaluation_validity = metadata.get(
        "evaluation_validity", model_bundle.get("evaluation_validity")
    )
    real_model_deployable = bool(
        metadata.get(
            "real_model_deployable", model_bundle.get("real_model_deployable", False)
        )
    )
    expected_schema_hash = metadata.get("feature_schema_sha256")
    current_schema_hash = feature_schema_sha256(current_feature_columns)
    feature_schema_match = bool(
        expected_schema_hash and expected_schema_hash == current_schema_hash
    )
    expected_config_hash = metadata.get("config_sha256")
    configuration_match = bool(
        expected_config_hash
        and current_config_sha256
        and expected_config_hash == current_config_sha256
    )
    real_contact_acceptance_passed = bool(
        metadata.get("real_contact_acceptance_passed", False)
    )
    feature_provenance_verified = bool(metadata.get("feature_provenance_verified", False))
    held_out_session_evaluation_passed = bool(
        metadata.get("held_out_session_evaluation_passed", False)
    )
    metadata_candidate_ids = tuple(
        str(value) for value in metadata.get("global_array_candidate_ids", [])
    )
    global_array_feature_scope_verified = bool(
        metadata.get("recognition_scope") == GLOBAL_ARRAY_RECOGNITION_SCOPE
        and metadata.get("global_array_feature_scope_verified", False)
        and metadata_candidate_ids == GLOBAL_ARRAY_CANDIDATE_IDS
        and has_global_array_feature_scope(current_feature_columns)
    )
    data_sources = {str(value) for value in metadata.get("data_sources", [])}
    blockers: list[str] = []
    warnings: list[str] = list(frame_quality.warnings)
    if model_bundle.get("model") is None:
        blockers.append("model_object_missing")
    if metadata.get("bundle_schema_version") != "hybrid_spectral_model_bundle_v2":
        blockers.append("unsupported_model_bundle_schema")
    if not feature_schema_match:
        blockers.append("feature_schema_mismatch")
    if not configuration_match:
        blockers.append("configuration_fingerprint_mismatch")
    if not model_artifact_integrity_verified:
        blockers.append("model_artifact_integrity_not_verified")
    if not feature_provenance_verified:
        blockers.append("feature_provenance_not_verified")
    if not held_out_session_evaluation_passed:
        blockers.append("held_out_session_evaluation_not_passed")
    if not global_array_feature_scope_verified:
        blockers.append("global_array_feature_scope_not_verified")
    if evaluation_validity not in {"grouped_by_trial_id", "grouped_by_backend_session"}:
        blockers.append("model_evaluation_not_formal_grouped_real")
    if any("synthetic" in source.lower() for source in data_sources):
        blockers.append("synthetic_training_data_present")
    if not real_model_deployable:
        blockers.append("model_not_marked_real_deployable")
    if not real_contact_acceptance_passed:
        blockers.append("real_contact_acceptance_not_passed")
    if not physical_mapping_approved or not bool(metadata.get("physical_mapping_final")):
        blockers.append("physical_channel_mapping_not_approved")
    if not baseline_ready:
        blockers.append("baseline_not_ready")
    if not frame_quality.spectral_input_valid:
        blockers.extend(frame_quality.blockers)
    blockers = list(dict.fromkeys(blockers))
    return PredictionReadinessDecision(
        prediction_allowed=not blockers,
        status="ready" if not blockers else "prediction_blocked",
        blockers=tuple(blockers),
        warnings=tuple(dict.fromkeys(warnings)),
        model_id=str(model_id) if model_id is not None else None,
        evaluation_validity=(
            str(evaluation_validity) if evaluation_validity is not None else None
        ),
        feature_schema_match=feature_schema_match,
        configuration_match=configuration_match,
        model_artifact_integrity_verified=model_artifact_integrity_verified,
        physical_mapping_approved=physical_mapping_approved,
        real_contact_acceptance_passed=real_contact_acceptance_passed,
        real_model_deployable=real_model_deployable,
        feature_provenance_verified=feature_provenance_verified,
        held_out_session_evaluation_passed=held_out_session_evaluation_passed,
        global_array_feature_scope_verified=global_array_feature_scope_verified,
    )
