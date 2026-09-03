"""Runtime-only baseline recovery state machine for the deployed TOUCH model."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class RuntimeBaselineRecoveryGuard:
    """Re-anchor only after a multi-evidence release and a quiet spectrum.

    Spectral stationarity is deliberately not sufficient by itself: a held
    fingertip can also be stationary. The state machine first arms on credible
    contact, then requires a release transition supported by at least two of
    the learned release output, raw no-contact hint, contact-probability drop,
    spatial ambiguity, and recovery toward the existing baseline. Only the
    physical spectra collected during the following quiet interval can become
    the new runtime baseline.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.enabled = bool(values.get("enabled", True))
        self.release_probability_threshold = float(
            values.get("release_probability_threshold", 0.40)
        )
        self.max_position_confidence = float(
            values.get("max_position_confidence", 0.45)
        )
        self.max_shape_motion_rms = float(
            values.get("max_shape_motion_rms", 0.0035)
        )
        self.max_common_gain_motion = float(
            values.get("max_common_gain_motion", 0.0030)
        )
        self.activity_shape_motion_rms = float(
            values.get("activity_shape_motion_rms", 0.0060)
        )
        self.activity_common_gain_motion = float(
            values.get("activity_common_gain_motion", 0.0060)
        )
        self.quiet_hold_sec = float(values.get("quiet_hold_sec", 0.5))
        self.minimum_quiet_physical_frames = int(
            values.get("minimum_quiet_physical_frames", 2)
        )
        self.release_candidate_timeout_sec = float(
            values.get("release_candidate_timeout_sec", 12.0)
        )
        self.contact_probability_arm = float(
            values.get("contact_probability_arm", 0.65)
        )
        self.contact_arm_physical_frames = int(
            values.get("contact_arm_physical_frames", 2)
        )
        self.minimum_contact_baseline_distance = float(
            values.get("minimum_contact_baseline_distance", 0.0060)
        )
        self.minimum_contact_distance_growth = float(
            values.get("minimum_contact_distance_growth", 0.0005)
        )
        self.no_contact_probability_support = float(
            values.get("no_contact_probability_support", 0.55)
        )
        self.minimum_recovery_fraction = float(
            values.get("minimum_recovery_fraction", 0.20)
        )
        self.minimum_contact_probability_drop = float(
            values.get("minimum_contact_probability_drop", 0.15)
        )
        self.minimum_spatial_contact_confidence = float(
            values.get("minimum_spatial_contact_confidence", 0.60)
        )
        self.minimum_slow_release_recovery_fraction = float(
            values.get("minimum_slow_release_recovery_fraction", 0.45)
        )
        self.release_activity_memory_sec = float(
            values.get("release_activity_memory_sec", 2.0)
        )
        self.stationary_rest_fallback_enabled = bool(
            values.get("stationary_rest_fallback_enabled", True)
        )
        self.stationary_rest_candidate_delay_sec = float(
            values.get("stationary_rest_candidate_delay_sec", 1.0)
        )
        self.stationary_rest_max_baseline_distance = float(
            values.get("stationary_rest_max_baseline_distance", 0.0050)
        )
        self.rest_unlock_position_confidence_min = float(
            values.get("rest_unlock_position_confidence_min", 0.35)
        )
        self.rest_unlock_strong_baseline_distance = float(
            values.get("rest_unlock_strong_baseline_distance", 0.0140)
        )
        self.locked_rest_reanchor_min_distance = float(
            values.get("locked_rest_reanchor_min_distance", 0.0040)
        )
        self.locked_rest_reanchor_max_distance = float(
            values.get("locked_rest_reanchor_max_distance", 0.0250)
        )
        self.fallback_requires_new_contact_after_reanchor = bool(
            values.get("fallback_requires_new_contact_after_reanchor", True)
        )
        self.recontact_probability_threshold = float(
            values.get("recontact_probability_threshold", 0.75)
        )
        self.recontact_distance_growth_fraction = float(
            values.get("recontact_distance_growth_fraction", 0.15)
        )
        self.recontact_distance_growth_absolute = float(
            values.get("recontact_distance_growth_absolute", 0.0040)
        )
        self.auto_update_runtime_baseline = bool(
            values.get("auto_update_runtime_baseline", True)
        )
        self.residual_release_enabled = bool(
            values.get("residual_release_enabled", False)
        )
        self.residual_release_minimum_recovery_fraction = float(
            values.get("residual_release_minimum_recovery_fraction", 0.60)
        )
        self.residual_release_max_position_confidence = float(
            values.get("residual_release_max_position_confidence", 0.35)
        )
        self.residual_release_max_baseline_distance = float(
            values.get("residual_release_max_baseline_distance", 0.0250)
        )
        self.residual_release_minimum_physical_frames = int(
            values.get("residual_release_minimum_physical_frames", 2)
        )
        self.residual_release_hold_sec = float(
            values.get("residual_release_hold_sec", 0.18)
        )

        # Backward-compatible frame mode is kept for older unit/config callers.
        self._legacy_frame_mode = bool(
            "quiet_hold_sec" not in values
            and (
                "suppress_after_physical_frames" in values
                or "reanchor_after_physical_frames" in values
            )
        )
        self.suppress_after_physical_frames = int(
            values.get("suppress_after_physical_frames", self.minimum_quiet_physical_frames)
        )
        self.reanchor_after_physical_frames = int(
            values.get("reanchor_after_physical_frames", self.minimum_quiet_physical_frames)
        )
        if self._legacy_frame_mode:
            self.minimum_quiet_physical_frames = self.reanchor_after_physical_frames
            self.quiet_hold_sec = 0.0
        if (
            self.minimum_quiet_physical_frames < 1
            or self.contact_arm_physical_frames < 1
            or self.quiet_hold_sec < 0.0
            or self.release_candidate_timeout_sec <= 0.0
            or self.minimum_contact_distance_growth < 0.0
            or self.stationary_rest_max_baseline_distance < 0.0
            or not 0.0 <= self.rest_unlock_position_confidence_min <= 1.0
            or self.rest_unlock_strong_baseline_distance < 0.0
            or self.locked_rest_reanchor_min_distance < 0.0
            or self.locked_rest_reanchor_max_distance
            < self.locked_rest_reanchor_min_distance
            or not 0.0
            <= self.residual_release_minimum_recovery_fraction
            <= 1.0
            or not 0.0 <= self.residual_release_max_position_confidence <= 1.0
            or self.residual_release_max_baseline_distance < 0.0
            or self.residual_release_minimum_physical_frames < 1
            or self.residual_release_hold_sec < 0.0
        ):
            raise ValueError("runtime baseline recovery configuration is invalid")

        self._spectra: deque[np.ndarray] = deque(
            maxlen=max(64, self.minimum_quiet_physical_frames * 4)
        )
        self._reanchor_count = 0
        self.reset(keep_reanchor_count=True)

    def reset(self, *, keep_reanchor_count: bool = False) -> None:
        fallback_locked = bool(
            getattr(self, "_fallback_locked_until_contact", False)
            if keep_reanchor_count
            else False
        )
        self._spectra.clear()
        self._last_physical_spectrum: np.ndarray | None = None
        self._last_timestamp_sec: float | None = None
        self._observation_started_at_sec: float | None = None
        self._last_activity_timestamp_sec: float | None = None
        self._stable_release_frames = 0
        self._quiet_started_at_sec: float | None = None
        self._quiet_elapsed_sec = 0.0
        self._active_contact_frames = 0
        self._contact_armed = False
        self._contact_peak_baseline_distance = 0.0
        self._contact_peak_probability = 0.0
        self._previous_baseline_distance: float | None = None
        self._release_candidate = False
        self._candidate_kind: str | None = None
        self._release_candidate_started_at_sec: float | None = None
        self._release_evidence: tuple[str, ...] = ()
        self._candidate_min_baseline_distance: float | None = None
        self._residual_release_frames = 0
        self._residual_release_started_at_sec: float | None = None
        self._residual_release_elapsed_sec = 0.0
        self._fallback_locked_until_contact = fallback_locked
        self._locked_recontact_evidence: tuple[str, ...] = ()
        if not keep_reanchor_count:
            self._reanchor_count = 0

    def prime_physical_spectrum(self, spectrum: np.ndarray) -> None:
        """Use a confirmed rest spectrum as the next motion reference."""

        reference = np.asarray(spectrum, dtype=float)
        if reference.ndim != 1 or reference.size < 5:
            raise ValueError("runtime motion reference must be one-dimensional")
        if not np.all(np.isfinite(reference)):
            raise ValueError("runtime motion reference contains non-finite values")
        self._last_physical_spectrum = reference.copy()

    def confirm_rest(self) -> None:
        """Latch a newly accepted baseline as trusted no-contact state."""

        self._clear_release_candidate()
        self._active_contact_frames = 0
        self._contact_armed = False
        self._contact_peak_baseline_distance = 0.0
        self._contact_peak_probability = 0.0
        self._fallback_locked_until_contact = True
        self._locked_recontact_evidence = ()

    @staticmethod
    def _motion(
        previous: np.ndarray | None,
        current: np.ndarray,
    ) -> tuple[float, float]:
        if previous is None:
            return 0.0, 0.0
        eps = 1.0e-6
        delta = np.log(np.maximum(current, eps)) - np.log(np.maximum(previous, eps))
        common_gain = float(np.median(delta))
        shape_motion = float(np.sqrt(np.mean(np.square(delta - common_gain))))
        return shape_motion, abs(common_gain)

    @staticmethod
    def _baseline_distance(
        baseline: np.ndarray | None,
        current: np.ndarray,
    ) -> float | None:
        if baseline is None or baseline.shape != current.shape:
            return None
        eps = 1.0e-6
        delta = np.log(np.maximum(current, eps)) - np.log(np.maximum(baseline, eps))
        return float(np.sqrt(np.mean(np.square(delta))))

    def _clear_release_candidate(self) -> None:
        self._release_candidate = False
        self._candidate_kind = None
        self._release_candidate_started_at_sec = None
        self._release_evidence = ()
        self._candidate_min_baseline_distance = None
        self._stable_release_frames = 0
        self._quiet_started_at_sec = None
        self._quiet_elapsed_sec = 0.0
        self._spectra.clear()
        self._clear_residual_release_evidence()

    def _clear_residual_release_evidence(self) -> None:
        self._residual_release_frames = 0
        self._residual_release_started_at_sec = None
        self._residual_release_elapsed_sec = 0.0

    def snapshot(
        self,
        *,
        external_no_contact_hint: bool | None = None,
        release_event_probability: float | None = None,
        position_confidence: float | None = None,
        contact_probability: float | None = None,
        contact_label: str | None = None,
        shape_motion_rms: float | None = None,
        common_gain_motion: float | None = None,
        baseline_distance: float | None = None,
        baseline_distance_growth: float | None = None,
        slow_baseline_departure: bool = False,
        recovery_fraction: float | None = None,
        spatial_contact_support: bool = False,
        slow_release_transition: bool = False,
        stationary_rest_baseline_safe: bool = False,
        locked_rest_drift_safe: bool = False,
        rest_unlock_spatial_support: bool = False,
        rest_unlock_strong_spectral_support: bool = False,
        suppress_contact: bool = False,
        reanchored: bool = False,
    ) -> dict[str, Any]:
        quiet_progress = (
            min(1.0, self._quiet_elapsed_sec / self.quiet_hold_sec)
            if self.quiet_hold_sec > 0.0
            else min(
                1.0,
                self._stable_release_frames
                / max(1, self.minimum_quiet_physical_frames),
            )
        )
        return {
            "enabled": self.enabled,
            "external_no_contact_hint": external_no_contact_hint,
            "release_event_probability": release_event_probability,
            "position_confidence": position_confidence,
            "contact_probability": contact_probability,
            "contact_label": contact_label,
            "shape_motion_rms": shape_motion_rms,
            "common_gain_motion": common_gain_motion,
            "baseline_distance": baseline_distance,
            "baseline_distance_growth": baseline_distance_growth,
            "slow_baseline_departure": bool(slow_baseline_departure),
            "contact_peak_baseline_distance": self._contact_peak_baseline_distance,
            "recovery_fraction": recovery_fraction,
            "spatial_contact_support": bool(spatial_contact_support),
            "slow_release_transition": bool(slow_release_transition),
            "stationary_rest_baseline_safe": bool(
                stationary_rest_baseline_safe
            ),
            "stationary_rest_max_baseline_distance": (
                self.stationary_rest_max_baseline_distance
            ),
            "locked_rest_drift_safe": bool(locked_rest_drift_safe),
            "locked_rest_reanchor_distance_range": [
                self.locked_rest_reanchor_min_distance,
                self.locked_rest_reanchor_max_distance,
            ],
            "rest_unlock_spatial_support": bool(
                rest_unlock_spatial_support
            ),
            "rest_unlock_position_confidence_min": (
                self.rest_unlock_position_confidence_min
            ),
            "rest_unlock_strong_spectral_support": bool(
                rest_unlock_strong_spectral_support
            ),
            "rest_unlock_strong_baseline_distance": (
                self.rest_unlock_strong_baseline_distance
            ),
            "contact_armed": self._contact_armed,
            "active_contact_physical_frames": self._active_contact_frames,
            "release_transition_detected": bool(
                self._release_candidate
                and self._candidate_kind in {"post_release", "residual_recovery"}
            ),
            "recovery_candidate_kind": self._candidate_kind,
            "release_evidence": list(self._release_evidence),
            "residual_release_enabled": self.residual_release_enabled,
            "residual_release_physical_frames": self._residual_release_frames,
            "residual_release_elapsed_sec": self._residual_release_elapsed_sec,
            "residual_release_minimum_recovery_fraction": (
                self.residual_release_minimum_recovery_fraction
            ),
            "residual_release_max_position_confidence": (
                self.residual_release_max_position_confidence
            ),
            "residual_release_max_baseline_distance": (
                self.residual_release_max_baseline_distance
            ),
            "residual_release_minimum_physical_frames": (
                self.residual_release_minimum_physical_frames
            ),
            "residual_release_hold_sec": self.residual_release_hold_sec,
            "stable_release_candidate": self._release_candidate,
            "stable_release_physical_frames": self._stable_release_frames,
            "quiet_elapsed_sec": self._quiet_elapsed_sec,
            "quiet_hold_sec": self.quiet_hold_sec,
            "quiet_progress": quiet_progress,
            "minimum_quiet_physical_frames": self.minimum_quiet_physical_frames,
            "suppress_contact": bool(suppress_contact),
            "runtime_reference_reanchored": bool(reanchored),
            "runtime_reference_reanchor_count": self._reanchor_count,
            "auto_update_runtime_baseline": self.auto_update_runtime_baseline,
            "stationary_rest_fallback_enabled": (
                self.stationary_rest_fallback_enabled
            ),
            "fallback_locked_until_new_contact": (
                self._fallback_locked_until_contact
            ),
            "runtime_rest_latched": self._fallback_locked_until_contact,
            "locked_recontact_evidence": list(self._locked_recontact_evidence),
            "policy": "multi_evidence_release_then_spectral_stationarity",
        }

    def observe(
        self,
        spectrum: np.ndarray,
        *,
        physical_frame: bool,
        external_no_contact_hint: bool | None,
        release_event_probability: float | None,
        position_confidence: float | None,
        contact_probability: float | None = None,
        contact_label: str | None = None,
        baseline_spectrum: np.ndarray | None = None,
        timestamp_sec: float | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        current = np.asarray(spectrum, dtype=float)
        if not self.enabled or not physical_frame:
            return self.snapshot(
                external_no_contact_hint=external_no_contact_hint,
                release_event_probability=release_event_probability,
                position_confidence=position_confidence,
                contact_probability=contact_probability,
                contact_label=contact_label,
            ), None

        if timestamp_sec is not None and np.isfinite(timestamp_sec):
            now_sec = float(timestamp_sec)
        elif self._last_timestamp_sec is None:
            now_sec = 0.0
        else:
            now_sec = self._last_timestamp_sec + 1.0
        if self._last_timestamp_sec is not None and now_sec <= self._last_timestamp_sec:
            now_sec = self._last_timestamp_sec + 1.0e-3
        if self._observation_started_at_sec is None:
            self._observation_started_at_sec = now_sec

        shape_motion, common_gain_motion = self._motion(
            self._last_physical_spectrum,
            current,
        )
        baseline_distance = self._baseline_distance(
            np.asarray(baseline_spectrum, dtype=float)
            if baseline_spectrum is not None
            else None,
            current,
        )
        baseline_distance_growth = (
            float(baseline_distance - self._previous_baseline_distance)
            if baseline_distance is not None
            and self._previous_baseline_distance is not None
            else None
        )
        stable = bool(
            shape_motion <= self.max_shape_motion_rms
            and common_gain_motion <= self.max_common_gain_motion
        )
        activity = bool(
            shape_motion >= self.activity_shape_motion_rms
            or common_gain_motion >= self.activity_common_gain_motion
        )
        if activity:
            self._last_activity_timestamp_sec = now_sec

        probability = (
            float(contact_probability)
            if contact_probability is not None
            else None
        )
        normalized_label = str(contact_label or "").strip().lower()
        model_contact = bool(
            normalized_label == "contact"
            and probability is not None
            and probability >= self.contact_probability_arm
        )
        spatial_contact_support = bool(
            position_confidence is not None
            and float(position_confidence)
            >= self.minimum_spatial_contact_confidence
        )
        rest_unlock_spatial_support = bool(
            position_confidence is not None
            and float(position_confidence)
            >= self.rest_unlock_position_confidence_min
        )
        rest_unlock_strong_spectral_support = bool(
            baseline_distance is not None
            and baseline_distance >= self.rest_unlock_strong_baseline_distance
        )
        separated_from_baseline = bool(
            baseline_distance is None
            or baseline_distance >= self.minimum_contact_baseline_distance
        )
        slow_baseline_departure = bool(
            separated_from_baseline
            and baseline_distance_growth is not None
            and baseline_distance_growth >= self.minimum_contact_distance_growth
        )
        activity_recent_for_recontact = bool(
            self._last_activity_timestamp_sec is not None
            and now_sec - self._last_activity_timestamp_sec
            <= self.release_activity_memory_sec
        )
        locked_recontact_evidence: list[str] = []
        if activity_recent_for_recontact:
            locked_recontact_evidence.append("recent_full_spectrum_activity")
        if separated_from_baseline:
            locked_recontact_evidence.append("separated_from_runtime_baseline")
        if slow_baseline_departure:
            locked_recontact_evidence.append("slow_baseline_departure")
        if model_contact:
            locked_recontact_evidence.append("contact_model_support")
        if spatial_contact_support:
            locked_recontact_evidence.append("spatial_contact_support")
        self._locked_recontact_evidence = tuple(locked_recontact_evidence)

        # Once a quiet spectrum has been accepted as the runtime rest state,
        # model probability alone must not unlock contact. Residual spectral
        # offsets can remain stationary and look contact-like to the temporal
        # model. Require a fresh physical change relative to the new baseline,
        # baseline separation, and model support before leaving no-contact.
        # Position confidence is intentionally excluded here: the position
        # classifier is trained on contact samples and always has to choose a
        # location, even for a no-contact spectrum.
        credible_contact = bool(
            (
                model_contact
                or (
                    normalized_label == "contact"
                    and rest_unlock_strong_spectral_support
                )
            )
            and separated_from_baseline
            and (
                not self._fallback_locked_until_contact
                or (
                    (
                        rest_unlock_spatial_support
                        or rest_unlock_strong_spectral_support
                    )
                    and (
                        activity_recent_for_recontact
                        or slow_baseline_departure
                    )
                )
            )
        )
        if not self._release_candidate:
            if credible_contact:
                self._active_contact_frames += 1
            elif normalized_label == "no_contact":
                self._active_contact_frames = 0
            else:
                self._active_contact_frames = max(0, self._active_contact_frames - 1)
            if self._active_contact_frames >= self.contact_arm_physical_frames:
                self._contact_armed = True
                self._fallback_locked_until_contact = False

        if self._contact_armed:
            if baseline_distance is not None:
                self._contact_peak_baseline_distance = max(
                    self._contact_peak_baseline_distance,
                    baseline_distance,
                )
            if probability is not None:
                self._contact_peak_probability = max(
                    self._contact_peak_probability,
                    probability,
                )

        recovery_fraction: float | None = None
        if self._contact_peak_baseline_distance > 1.0e-9 and baseline_distance is not None:
            recovery_fraction = max(
                0.0,
                min(
                    1.0,
                    (
                        self._contact_peak_baseline_distance - baseline_distance
                    )
                    / self._contact_peak_baseline_distance,
                ),
            )
        probability_drop = (
            max(0.0, self._contact_peak_probability - probability)
            if probability is not None
            else 0.0
        )
        release_like = bool(
            release_event_probability is not None
            and float(release_event_probability)
            >= self.release_probability_threshold
        )
        model_no_contact = bool(
            normalized_label == "no_contact"
            or (
                probability is not None
                and probability <= 1.0 - self.no_contact_probability_support
            )
        )
        spatially_ambiguous = bool(
            position_confidence is None
            or float(position_confidence) <= self.max_position_confidence
        )
        recovered_toward_baseline = bool(
            recovery_fraction is not None
            and recovery_fraction >= self.minimum_recovery_fraction
        )
        contact_probability_dropped = bool(
            probability_drop >= self.minimum_contact_probability_drop
        )
        slow_release_transition = bool(
            self._contact_armed
            and recovery_fraction is not None
            and recovery_fraction
            >= self.minimum_slow_release_recovery_fraction
            and model_no_contact
        )
        activity_recent = bool(
            self._last_activity_timestamp_sec is not None
            and now_sec - self._last_activity_timestamp_sec
            <= self.release_activity_memory_sec
        )

        # The contact classifier can stay saturated after a physical release.
        # Treat recovery of the complete spectrum as an independent release
        # signal, but only after a real contact peak and only while the spectrum
        # is quiet and spatially ambiguous. A held plateau has near-zero
        # recovery, so low position confidence alone cannot clear it.
        residual_release_frame_evidence = bool(
            self.residual_release_enabled
            and self._contact_armed
            and stable
            and not activity_recent
            and recovery_fraction is not None
            and recovery_fraction
            >= self.residual_release_minimum_recovery_fraction
            and position_confidence is not None
            and float(position_confidence)
            <= self.residual_release_max_position_confidence
            and baseline_distance is not None
            and baseline_distance <= self.residual_release_max_baseline_distance
        )
        if not self._release_candidate and residual_release_frame_evidence:
            if self._residual_release_started_at_sec is None:
                self._residual_release_started_at_sec = now_sec
            self._residual_release_frames += 1
            self._residual_release_elapsed_sec = max(
                0.0,
                now_sec - self._residual_release_started_at_sec,
            )
        elif not self._release_candidate:
            self._clear_residual_release_evidence()
        residual_release_confirmed = bool(
            residual_release_frame_evidence
            and self._residual_release_frames
            >= self.residual_release_minimum_physical_frames
            and self._residual_release_elapsed_sec
            >= self.residual_release_hold_sec
        )

        evidence: list[str] = []
        if release_like:
            evidence.append("learned_release_event")
        if external_no_contact_hint is True:
            evidence.append("raw_no_contact_hint")
        if model_no_contact:
            evidence.append("contact_model_no_contact")
        if contact_probability_dropped:
            evidence.append("contact_probability_drop")
        if spatially_ambiguous:
            evidence.append("spatial_confidence_low")
        if recovered_toward_baseline:
            evidence.append("spectrum_recovered_toward_baseline")
        if slow_release_transition:
            evidence.append("slow_release_after_contact_peak")

        stationary_rest_fallback = False
        locked_rest_drift_fallback = False
        stationary_rest_baseline_safe = False
        locked_rest_drift_safe = False
        if self._legacy_frame_mode:
            release_transition = bool(
                external_no_contact_hint is True
                and release_like
                and spatially_ambiguous
            )
        else:
            # Correlated classifier outputs are not physical release evidence.
            # A temporary contact-model dip can happen during a stationary
            # press, so probability drop + no-contact label must never arm a
            # baseline update unless the spectrum also recovers or the learned
            # release detector observes a release event.
            # A stationary held press can relax slightly after its force peak,
            # and the contact classifier can briefly dip at the same time. Only
            # the stronger configured recovery fraction may support an automatic
            # release; the lower threshold remains diagnostic evidence only.
            recovery_supported_release = bool(
                recovery_fraction is not None
                and recovery_fraction
                >= self.minimum_slow_release_recovery_fraction
                and model_no_contact
            )
            primary_release_evidence = bool(
                release_like or recovery_supported_release
            )
            release_transition = bool(
                self._contact_armed
                and (
                    (
                        activity_recent
                        and primary_release_evidence
                        and len(evidence) >= 2
                    )
                    or slow_release_transition
                    or residual_release_confirmed
                )
            )
            quiet_since_activity = bool(
                self._last_activity_timestamp_sec is None
                or now_sec - self._last_activity_timestamp_sec
                >= self.stationary_rest_candidate_delay_sec
            )
            stationary_rest_baseline_safe = bool(
                baseline_distance is not None
                and baseline_distance
                <= self.stationary_rest_max_baseline_distance
            )
            locked_rest_drift_safe = bool(
                self._fallback_locked_until_contact
                and baseline_distance is not None
                and self.locked_rest_reanchor_min_distance
                <= baseline_distance
                <= self.locked_rest_reanchor_max_distance
                and not rest_unlock_spatial_support
                and not rest_unlock_strong_spectral_support
            )
            near_baseline_rest_fallback = bool(
                self.stationary_rest_fallback_enabled
                and not self._fallback_locked_until_contact
                and stable
                and stationary_rest_baseline_safe
                and (external_no_contact_hint is True or model_no_contact)
                and quiet_since_activity
            )
            locked_rest_drift_fallback = bool(
                self.stationary_rest_fallback_enabled
                and self.auto_update_runtime_baseline
                and locked_rest_drift_safe
                and stable
                and quiet_since_activity
            )
            stationary_rest_fallback = bool(
                near_baseline_rest_fallback or locked_rest_drift_fallback
            )
            if stationary_rest_fallback and not release_transition:
                release_transition = True
                if locked_rest_drift_fallback:
                    evidence.extend(
                        [
                            "locked_rest_drift_recovery",
                            "full_spectrum_stationary",
                            "position_below_rest_unlock_threshold",
                        ]
                    )
                else:
                    evidence.extend(
                        [
                            "stationary_rest_fallback",
                            "full_spectrum_stationary",
                            "near_existing_runtime_baseline",
                        ]
                    )
            if residual_release_confirmed:
                evidence.extend(
                    [
                        "residual_spectral_recovery",
                        "full_spectrum_stationary",
                        "no_recent_spectral_activity",
                        "position_confidence_low",
                    ]
                )
        if release_transition and not self._release_candidate:
            self._release_candidate = True
            self._candidate_kind = (
                "residual_recovery"
                if residual_release_confirmed
                else (
                    "locked_rest_drift_recovery"
                    if locked_rest_drift_fallback
                    else (
                        "stationary_rest_recovery"
                        if stationary_rest_fallback
                        else "post_release"
                    )
                )
            )
            self._release_candidate_started_at_sec = now_sec
            self._release_evidence = tuple(evidence)
            self._candidate_min_baseline_distance = baseline_distance

        if self._release_candidate and baseline_distance is not None:
            self._candidate_min_baseline_distance = (
                baseline_distance
                if self._candidate_min_baseline_distance is None
                else min(self._candidate_min_baseline_distance, baseline_distance)
            )

        if self._release_candidate:
            distance_growth = False
            if (
                baseline_distance is not None
                and self._candidate_min_baseline_distance is not None
            ):
                growth = baseline_distance - self._candidate_min_baseline_distance
                distance_growth = bool(
                    growth >= self.recontact_distance_growth_absolute
                    and baseline_distance
                    >= self._candidate_min_baseline_distance
                    * (1.0 + self.recontact_distance_growth_fraction)
                )
            slow_joint_signature_recontact = bool(
                slow_baseline_departure
                and spatial_contact_support
                and model_contact
            )
            strong_recontact = bool(
                (activity or slow_joint_signature_recontact)
                and normalized_label == "contact"
                and probability is not None
                and probability >= self.recontact_probability_threshold
                and (
                    self._candidate_kind != "locked_rest_drift_recovery"
                    or rest_unlock_spatial_support
                )
                and not release_like
                and distance_growth
            )
            if strong_recontact:
                self._clear_release_candidate()
                self._clear_residual_release_evidence()
            elif stable:
                if self._quiet_started_at_sec is None:
                    self._quiet_started_at_sec = now_sec
                self._stable_release_frames += 1
                self._quiet_elapsed_sec = max(
                    0.0,
                    now_sec - self._quiet_started_at_sec,
                )
                self._spectra.append(current.copy())
            else:
                self._stable_release_frames = 0
                self._quiet_started_at_sec = None
                self._quiet_elapsed_sec = 0.0
                self._spectra.clear()

            if (
                self._release_candidate_started_at_sec is not None
                and now_sec - self._release_candidate_started_at_sec
                > self.release_candidate_timeout_sec
                and self._quiet_elapsed_sec < self.quiet_hold_sec
            ):
                self._clear_release_candidate()

        if self._legacy_frame_mode:
            suppress_contact = bool(
                self._stable_release_frames >= self.suppress_after_physical_frames
            )
            reanchored = bool(
                self._stable_release_frames >= self.reanchor_after_physical_frames
                and len(self._spectra) >= self.reanchor_after_physical_frames
            )
        else:
            quiet_complete = bool(
                self._release_candidate
                and self._quiet_elapsed_sec >= self.quiet_hold_sec
                and self._stable_release_frames
                >= self.minimum_quiet_physical_frames
            )
            suppress_contact = bool(
                quiet_complete or self._fallback_locked_until_contact
            )
            reanchored = bool(quiet_complete and self.auto_update_runtime_baseline)

        recovered_baseline: np.ndarray | None = None
        if reanchored:
            recovered_baseline = np.median(
                np.stack(tuple(self._spectra), axis=0),
                axis=0,
            )
            self._reanchor_count += 1
            if self.fallback_requires_new_contact_after_reanchor:
                self._fallback_locked_until_contact = True

        state = self.snapshot(
            external_no_contact_hint=external_no_contact_hint,
            release_event_probability=release_event_probability,
            position_confidence=position_confidence,
            contact_probability=probability,
            contact_label=normalized_label or None,
            shape_motion_rms=shape_motion,
            common_gain_motion=common_gain_motion,
            baseline_distance=baseline_distance,
            baseline_distance_growth=baseline_distance_growth,
            slow_baseline_departure=slow_baseline_departure,
            recovery_fraction=recovery_fraction,
            spatial_contact_support=spatial_contact_support,
            slow_release_transition=slow_release_transition,
            stationary_rest_baseline_safe=(
                baseline_distance is not None
                and baseline_distance
                <= self.stationary_rest_max_baseline_distance
            ),
            locked_rest_drift_safe=(
                self._fallback_locked_until_contact
                and baseline_distance is not None
                and self.locked_rest_reanchor_min_distance
                <= baseline_distance
                <= self.locked_rest_reanchor_max_distance
                and not rest_unlock_spatial_support
                and not rest_unlock_strong_spectral_support
            ),
            rest_unlock_spatial_support=rest_unlock_spatial_support,
            rest_unlock_strong_spectral_support=(
                rest_unlock_strong_spectral_support
            ),
            suppress_contact=suppress_contact,
            reanchored=reanchored,
        )

        self._last_physical_spectrum = current.copy()
        self._last_timestamp_sec = now_sec
        self._previous_baseline_distance = baseline_distance
        if reanchored:
            self.reset(keep_reanchor_count=True)
            self._last_physical_spectrum = current.copy()
            self._last_timestamp_sec = now_sec
        return state, recovered_baseline


__all__ = ["RuntimeBaselineRecoveryGuard"]
