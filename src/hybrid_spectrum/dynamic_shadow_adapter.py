"""Live/replay adapter for the non-primary dynamic temporal shadow candidate."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .dynamic_sequence_dataset import extract_baseline_relative_frame_features
from .dynamic_temporal_features import SUMMARY_FEATURE_BLOCK_ORDER, temporal_summary_features
from .dynamic_twin_mapping import dynamic_prediction_to_twin_proxy
from .features import PeakWindow, load_peak_windows


SUPPORTED_SCHEMA_VERSIONS = {
    "dynamic_temporal_shadow_candidate_v1",
    "dynamic_temporal_shadow_candidate_v2",
    "dynamic_temporal_shadow_candidate_v3",
}


def _aligned_probability(model: Any, values: np.ndarray, labels: list[str]) -> np.ndarray:
    probability = np.asarray(model.predict_proba(values), dtype=float)
    aligned = np.zeros((len(values), len(labels)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        aligned[:, labels.index(str(label))] = probability[:, source_index]
    return aligned


def _positive_class_probability(
    model: Any,
    values: np.ndarray,
    positive_class: Any = 1,
) -> np.ndarray:
    probability = np.asarray(model.predict_proba(values), dtype=float)
    classes = list(model.classes_)
    matching = [
        index
        for index, label in enumerate(classes)
        if label == positive_class or str(label) == str(positive_class)
    ]
    if len(matching) != 1:
        raise ValueError("release-event model is missing its positive class")
    return probability[:, matching[0]]


class ReleaseResidualGuard:
    """Latch a confirmed hard-to-release transition as physical no-contact.

    The guard is intentionally conservative and is only enabled for a bundle
    that contains grouped validation evidence. A new baseline always clears
    the latch. At runtime, a confirmed quiet no-contact interval followed by
    sustained high-confidence contact can also re-arm the next press without
    carrying release residual into the new trial.
    """

    def __init__(self, config: dict[str, Any] | None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.hard_arm_probability = float(
            self.config.get("hard_arm_probability", 0.50)
        )
        self.hard_arm_frames = int(self.config.get("hard_arm_frames", 20))
        self.hard_exit_probability = float(
            self.config.get("hard_exit_probability", 0.30)
        )
        self.hard_exit_frames = int(self.config.get("hard_exit_frames", 1))
        self.release_event_probability = float(
            self.config.get("release_event_probability", 0.40)
        )
        self.auto_rearm_enabled = bool(
            self.config.get("auto_rearm_enabled", True)
        )
        self.auto_rearm_quiet_frames = int(
            self.config.get("auto_rearm_quiet_frames", 3)
        )
        self.auto_rearm_contact_frames = int(
            self.config.get("auto_rearm_contact_frames", 3)
        )
        self.auto_rearm_contact_probability = float(
            self.config.get("auto_rearm_contact_probability", 0.85)
        )
        self.auto_rearm_max_release_probability = float(
            self.config.get("auto_rearm_max_release_probability", 0.30)
        )
        if (
            self.hard_arm_frames < 1
            or self.hard_exit_frames < 1
            or self.auto_rearm_quiet_frames < 1
            or self.auto_rearm_contact_frames < 1
        ):
            raise ValueError("release guard frame counts must be positive")
        self.reset()

    def reset(self) -> None:
        self._hard_run = 0
        self._hard_exit_run = 0
        self._armed = False
        self._latched = False
        self._quiet_no_contact_run = 0
        self._recontact_run = 0
        self._auto_rearm_ready = False

    def update(
        self,
        *,
        hard_probability: float,
        release_event_probability: float | None,
        contact_probability: float | None = None,
        raw_contact_label: str | None = None,
    ) -> dict[str, Any]:
        hard_probability = float(hard_probability)
        if not self.enabled:
            return self.snapshot(
                release_event_probability=release_event_probability,
                just_latched=False,
            )
        if self._latched:
            just_rearmed = self._update_auto_rearm(
                contact_probability=contact_probability,
                raw_contact_label=raw_contact_label,
                release_event_probability=release_event_probability,
            )
            return self.snapshot(
                release_event_probability=release_event_probability,
                just_latched=False,
                just_rearmed=just_rearmed,
            )

        if hard_probability >= self.hard_arm_probability:
            self._hard_run += 1
        else:
            self._hard_run = 0
        if self._hard_run >= self.hard_arm_frames:
            self._armed = True

        if self._armed and hard_probability < self.hard_exit_probability:
            self._hard_exit_run += 1
        else:
            self._hard_exit_run = 0

        event_probability = (
            float(release_event_probability)
            if release_event_probability is not None
            else 0.0
        )
        just_latched = bool(
            self._armed
            and self._hard_exit_run >= self.hard_exit_frames
            and event_probability >= self.release_event_probability
        )
        if just_latched:
            self._latched = True
        return self.snapshot(
            release_event_probability=release_event_probability,
            just_latched=just_latched,
        )

    def _update_auto_rearm(
        self,
        *,
        contact_probability: float | None,
        raw_contact_label: str | None,
        release_event_probability: float | None,
    ) -> bool:
        if not self.auto_rearm_enabled:
            return False

        label = str(raw_contact_label or "").strip().lower()
        probability = (
            float(contact_probability)
            if contact_probability is not None
            else None
        )
        if label == "no_contact":
            self._quiet_no_contact_run += 1
            self._recontact_run = 0
            if self._quiet_no_contact_run >= self.auto_rearm_quiet_frames:
                self._auto_rearm_ready = True
            return False

        if not self._auto_rearm_ready:
            self._recontact_run = 0
            return False

        release_probability = (
            float(release_event_probability)
            if release_event_probability is not None
            else 0.0
        )
        strong_recontact = bool(
            label == "contact"
            and probability is not None
            and probability >= self.auto_rearm_contact_probability
            and release_probability <= self.auto_rearm_max_release_probability
        )
        self._recontact_run = self._recontact_run + 1 if strong_recontact else 0
        if self._recontact_run < self.auto_rearm_contact_frames:
            return False

        self._latched = False
        self._armed = False
        self._hard_run = 0
        self._hard_exit_run = 0
        self._quiet_no_contact_run = 0
        self._recontact_run = 0
        self._auto_rearm_ready = False
        return True

    def snapshot(
        self,
        *,
        release_event_probability: float | None,
        just_latched: bool,
        just_rearmed: bool = False,
    ) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "armed": self._armed,
            "release_latched": self._latched,
            "just_latched": bool(just_latched),
            "just_rearmed": bool(just_rearmed),
            "hard_evidence_frames": self._hard_run,
            "hard_exit_frames": self._hard_exit_run,
            "release_event_probability": release_event_probability,
            "auto_rearm_enabled": self.auto_rearm_enabled,
            "auto_rearm_ready": self._auto_rearm_ready,
            "auto_rearm_quiet_frames": self._quiet_no_contact_run,
            "auto_rearm_contact_frames": self._recontact_run,
            "baseline_reset_required_after_latch": bool(
                self.config.get("baseline_reset_required_after_latch", True)
            ),
            "validation_scope": self.config.get(
                "validation_scope",
                "not_available",
            ),
        }


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
        self.no_contact_probability_support = float(
            values.get("no_contact_probability_support", 0.55)
        )
        self.minimum_recovery_fraction = float(
            values.get("minimum_recovery_fraction", 0.20)
        )
        self.minimum_contact_probability_drop = float(
            values.get("minimum_contact_probability_drop", 0.15)
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
        self._fallback_locked_until_contact = fallback_locked
        self._locked_recontact_evidence: tuple[str, ...] = ()
        if not keep_reanchor_count:
            self._reanchor_count = 0

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
        recovery_fraction: float | None = None,
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
            "contact_peak_baseline_distance": self._contact_peak_baseline_distance,
            "recovery_fraction": recovery_fraction,
            "contact_armed": self._contact_armed,
            "active_contact_physical_frames": self._active_contact_frames,
            "release_transition_detected": bool(
                self._release_candidate and self._candidate_kind == "post_release"
            ),
            "recovery_candidate_kind": self._candidate_kind,
            "release_evidence": list(self._release_evidence),
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
        separated_from_baseline = bool(
            baseline_distance is None
            or baseline_distance >= self.minimum_contact_baseline_distance
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
        if model_contact:
            locked_recontact_evidence.append("contact_model_support")
        self._locked_recontact_evidence = tuple(locked_recontact_evidence)

        # Once a quiet spectrum has been accepted as the runtime rest state,
        # model probability alone must not unlock contact. Residual spectral
        # offsets can remain stationary and look contact-like to the temporal
        # model. Require a fresh physical change relative to the new baseline,
        # baseline separation, and model support before leaving no-contact.
        credible_contact = bool(
            model_contact
            and separated_from_baseline
            and (
                not self._fallback_locked_until_contact
                or activity_recent_for_recontact
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
        activity_recent = bool(
            self._last_activity_timestamp_sec is not None
            and now_sec - self._last_activity_timestamp_sec
            <= self.release_activity_memory_sec
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

        stationary_rest_fallback = False
        if self._legacy_frame_mode:
            release_transition = bool(
                external_no_contact_hint is True
                and release_like
                and spatially_ambiguous
            )
        else:
            primary_release_evidence = bool(
                release_like or recovered_toward_baseline or model_no_contact
            )
            release_transition = bool(
                self._contact_armed
                and activity_recent
                and primary_release_evidence
                and len(evidence) >= 2
            )
            quiet_since_activity = bool(
                self._last_activity_timestamp_sec is None
                or now_sec - self._last_activity_timestamp_sec
                >= self.stationary_rest_candidate_delay_sec
            )
            stationary_rest_fallback = bool(
                self.stationary_rest_fallback_enabled
                and not self._fallback_locked_until_contact
                and stable
                and external_no_contact_hint is True
                and spatially_ambiguous
                and quiet_since_activity
            )
            if stationary_rest_fallback and not release_transition:
                release_transition = True
                evidence.extend(
                    [
                        "stationary_rest_fallback",
                        "full_spectrum_stationary",
                    ]
                )
        if release_transition and not self._release_candidate:
            self._release_candidate = True
            self._candidate_kind = (
                "stationary_rest_recovery"
                if stationary_rest_fallback
                else "post_release"
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
            strong_recontact = bool(
                activity
                and normalized_label == "contact"
                and probability is not None
                and probability >= self.recontact_probability_threshold
                and not release_like
                and distance_growth
            )
            if strong_recontact:
                self._clear_release_candidate()
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
            recovery_fraction=recovery_fraction,
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


def load_dynamic_shadow_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path.resolve())
    if not isinstance(bundle, dict):
        raise TypeError("dynamic shadow artifact must be a mapping")
    schema_version = bundle.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported dynamic shadow artifact schema")
    if bundle.get("status") != "shadow_only_not_primary":
        raise ValueError("dynamic candidate is missing its shadow-only safety marker")
    if bundle.get("deployment_ready") is not False:
        raise ValueError("dynamic shadow artifact must remain deployment blocked")
    if tuple(bundle.get("summary_feature_block_order", ())) != SUMMARY_FEATURE_BLOCK_ORDER:
        raise ValueError("temporal summary feature contract mismatch")
    required_models = {
        "contact_extra_trees",
        "position_extra_trees",
        "position_rbf_svm",
        "response_extra_trees",
        "response_rbf_svm",
    }
    if not required_models.issubset(bundle.get("models", {})):
        raise ValueError("dynamic shadow artifact is incomplete")
    if schema_version in {
        "dynamic_temporal_shadow_candidate_v2",
        "dynamic_temporal_shadow_candidate_v3",
    }:
        if "release_event_extra_trees" not in bundle.get("models", {}):
            raise ValueError("release-aware dynamic shadow artifact is incomplete")
        if not bundle.get("release_guard", {}).get("enabled", False):
            raise ValueError("v2 dynamic shadow artifact is missing its release guard")
        release_validation = bundle.get("release_guard_grouped_cv", {})
        if release_validation.get("evaluation_validity") != (
            "grouped_by_capture_group_and_file_id"
        ):
            raise ValueError("v2 release guard is missing grouped validation evidence")
        if int(release_validation.get("unsafe_early_trigger_sequence_count", -1)) != 0:
            raise ValueError("v2 release guard has unsafe pre-release triggers")
    if schema_version == "dynamic_temporal_shadow_candidate_v3":
        if "position_factorized" not in bundle.get("models", {}):
            raise ValueError("v3 dynamic shadow artifact is missing factorized position")
        if bundle.get("position_inference_mode") != (
            "factorized_row_column_probability_product"
        ):
            raise ValueError("v3 factorized position contract is incomplete")
    return bundle


class DynamicTemporalShadowAdapter:
    """Convert consecutive spectra into a shadow-only position/level prediction."""

    def __init__(
        self,
        bundle: dict[str, Any],
        peak_windows: Iterable[PeakWindow],
        runtime_recovery_config: dict[str, Any] | None = None,
    ) -> None:
        self.bundle = bundle
        self.peak_windows = tuple(peak_windows)
        self.time_steps = int(bundle["time_steps"])
        self._feature_history: deque[np.ndarray] = deque(maxlen=self.time_steps)
        self._wavelength_nm: np.ndarray | None = None
        self._baseline_spectrum: np.ndarray | None = None
        self._frame_counter = 0
        self._release_guard = ReleaseResidualGuard(bundle.get("release_guard"))
        recovery_config = dict(bundle.get("runtime_baseline_recovery") or {})
        recovery_config.update(dict(runtime_recovery_config or {}))
        self._prime_temporal_history_with_baseline = bool(
            recovery_config.get("prime_temporal_history_with_baseline", False)
        )
        try:
            configured_preroll_frames = int(
                recovery_config.get("baseline_preroll_frames", self.time_steps)
            )
        except (TypeError, ValueError):
            configured_preroll_frames = self.time_steps
        self._baseline_preroll_frames = max(
            0,
            min(self.time_steps, configured_preroll_frames),
        )
        self._runtime_baseline_recovery = RuntimeBaselineRecoveryGuard(
            recovery_config
        )
        self._runtime_baseline_revision = 0
        self._pending_runtime_baseline_update: dict[str, Any] | None = None
        self._last_ready_output: dict[str, Any] | None = None
        # Thread-pool startup dominates latency for one-row ExtraTrees inference.
        # One worker preserves predictions and is substantially faster live.
        for model in bundle["models"].values():
            if hasattr(model, "set_runtime_n_jobs"):
                model.set_runtime_n_jobs(1)
            elif hasattr(model, "n_jobs"):
                model.n_jobs = 1

    @classmethod
    def from_paths(
        cls,
        model_path: Path,
        peak_config_path: Path,
        runtime_recovery_config: dict[str, Any] | None = None,
    ) -> "DynamicTemporalShadowAdapter":
        return cls(
            load_dynamic_shadow_bundle(model_path),
            load_peak_windows(peak_config_path.resolve()),
            runtime_recovery_config=runtime_recovery_config,
        )

    @property
    def baseline_ready(self) -> bool:
        return self._baseline_spectrum is not None

    def _prime_feature_history_from_baseline(self) -> int:
        """Seed a no-contact pre-roll so the next physical frame is inferable."""
        self._feature_history.clear()
        if (
            not self._prime_temporal_history_with_baseline
            or self._baseline_preroll_frames <= 0
            or self._wavelength_nm is None
            or self._baseline_spectrum is None
        ):
            return 0
        features, names, _, _ = extract_baseline_relative_frame_features(
            self._wavelength_nm,
            self._baseline_spectrum,
            self._baseline_spectrum,
            self.peak_windows,
        )
        expected_names = tuple(self.bundle["frame_feature_names"])
        if names != expected_names:
            raise ValueError(
                "baseline pre-roll feature order differs from the trained artifact"
            )
        baseline_features = np.asarray(features[0], dtype=np.float32)
        for _ in range(self._baseline_preroll_frames):
            self._feature_history.append(baseline_features.copy())
        return len(self._feature_history)

    def consume_pending_runtime_baseline_update(self) -> dict[str, Any] | None:
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

    def set_baseline(
        self,
        wavelength_nm: np.ndarray,
        baseline_spectrum: np.ndarray,
    ) -> None:
        wavelength = np.asarray(wavelength_nm, dtype=float)
        baseline = np.asarray(baseline_spectrum, dtype=float)
        if wavelength.ndim != 1 or baseline.shape != wavelength.shape:
            raise ValueError("baseline and wavelength must be aligned one-dimensional arrays")
        if not np.all(np.diff(wavelength) > 0.0):
            raise ValueError("wavelength grid must be strictly increasing")
        self._wavelength_nm = wavelength.copy()
        self._baseline_spectrum = baseline.copy()
        self._prime_feature_history_from_baseline()
        self._frame_counter = 0
        self._release_guard.reset()
        self._runtime_baseline_recovery.reset()
        self._runtime_baseline_revision = 0
        self._pending_runtime_baseline_update = None
        self._last_ready_output = None

    def clear(self) -> None:
        self._feature_history.clear()
        self._frame_counter = 0
        self._release_guard.reset()
        self._runtime_baseline_recovery.reset()
        self._runtime_baseline_revision = 0
        self._pending_runtime_baseline_update = None
        self._last_ready_output = None

    def _warming_output(self, status: str) -> dict[str, Any]:
        output = {
            "status": status,
            "ready": False,
            "mode": "shadow_only_not_primary",
            "history_frames": len(self._feature_history),
            "required_frames": self.time_steps,
            "contact": None,
            "position": None,
            "response_level": None,
            "response_level_semantics": "approximate_manual_level_not_force_N",
            "release_guard": self._release_guard.snapshot(
                release_event_probability=None,
                just_latched=False,
            ),
            "runtime_baseline_recovery": self._runtime_baseline_recovery.snapshot(),
            "runtime_baseline_revision": self._runtime_baseline_revision,
            "runtime_inference_policy": "single_sample_tree_inference_n_jobs_1",
            "baseline_preroll_enabled": self._prime_temporal_history_with_baseline,
            "baseline_preroll_frames": min(
                len(self._feature_history),
                self._baseline_preroll_frames,
            ),
        }
        output["digital_twin_proxy"] = dynamic_prediction_to_twin_proxy(output)
        return output

    def update(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        *,
        run_inference: bool = True,
        physical_frame: bool = True,
        external_no_contact_hint: bool | None = None,
        source_timestamp_sec: float | None = None,
    ) -> dict[str, Any]:
        if self._wavelength_nm is None or self._baseline_spectrum is None:
            return self._warming_output("baseline_required")
        wavelength = np.asarray(wavelength_nm, dtype=float)
        current = np.asarray(spectrum, dtype=float)
        if wavelength.shape != self._wavelength_nm.shape or not np.allclose(
            wavelength, self._wavelength_nm, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("live wavelength grid differs from the baseline grid")
        if current.shape != wavelength.shape:
            raise ValueError("spectrum does not match the wavelength grid")
        features, names, _, _ = extract_baseline_relative_frame_features(
            wavelength,
            current,
            self._baseline_spectrum,
            self.peak_windows,
        )
        expected_names = tuple(self.bundle["frame_feature_names"])
        if names != expected_names:
            raise ValueError("live frame feature order differs from the trained artifact")
        self._feature_history.append(np.asarray(features[0], dtype=np.float32))
        self._frame_counter += 1
        if len(self._feature_history) < self.time_steps:
            return self._warming_output("window_warming_up")
        if not run_inference:
            if self._last_ready_output is None:
                return self._warming_output("inference_stride_hold")
            held = deepcopy(self._last_ready_output)
            held.update(
                {
                    "status": "shadow_stride_hold",
                    "ready": True,
                    "frame_counter": self._frame_counter,
                    "history_frames": len(self._feature_history),
                    "held_from_previous_unique_frame": True,
                }
            )
            return held

        window = np.stack(tuple(self._feature_history), axis=0)[None, :, :]
        summary = temporal_summary_features(window)
        labels = self.bundle["label_order"]
        models = self.bundle["models"]
        contact_probability = _aligned_probability(
            models["contact_extra_trees"], summary, list(labels["contact"])
        )[0]
        contact_index = int(np.argmax(contact_probability))
        contact_label = str(labels["contact"][contact_index])
        release_event_probability: float | None = None
        release_model = models.get("release_event_extra_trees")
        if release_model is not None:
            release_event_probability = float(
                _positive_class_probability(release_model, summary, positive_class=1)[0]
            )
        output: dict[str, Any] = {
            "status": "shadow_ready",
            "ready": True,
            "mode": "shadow_only_not_primary",
            "history_frames": len(self._feature_history),
            "required_frames": self.time_steps,
            "frame_counter": self._frame_counter,
            "contact": {
                "label": contact_label,
                "confidence": float(contact_probability[contact_index]),
                "probabilities": {
                    label: float(contact_probability[index])
                    for index, label in enumerate(labels["contact"])
                },
            },
            "position": None,
            "response_level": None,
            "confidence_note": "uncalibrated model probabilities",
            "response_level_semantics": "approximate_manual_level_not_force_N",
            "release_guard": self._release_guard.snapshot(
                release_event_probability=None,
                just_latched=False,
            ),
            "runtime_inference_policy": "single_sample_tree_inference_n_jobs_1",
            "runtime_baseline_revision": self._runtime_baseline_revision,
            "baseline_preroll_enabled": self._prime_temporal_history_with_baseline,
            "baseline_preroll_frames": min(
                len(self._feature_history),
                self._baseline_preroll_frames,
            ),
        }
        if contact_label == "no_contact":
            output["release_guard"] = self._release_guard.update(
                hard_probability=0.0,
                release_event_probability=release_event_probability,
                contact_probability=float(
                    contact_probability[list(labels["contact"]).index("contact")]
                ),
                raw_contact_label=contact_label,
            )
            output["operational_state"] = "no_contact"
        elif self.bundle.get("position_inference_mode") == (
            "factorized_row_column_probability_product"
        ):
            position_probability = _aligned_probability(
                models["position_factorized"],
                summary,
                list(labels["position"]),
            )[0]
            position_inference_mode = "factorized_row_column_probability_product"
        else:
            position_probability = 0.5 * _aligned_probability(
                models["position_extra_trees"], summary, list(labels["position"])
            )[0] + 0.5 * _aligned_probability(
                models["position_rbf_svm"], summary, list(labels["position"])
            )[0]
            position_inference_mode = "direct_nine_class_soft_vote"
        if contact_label != "no_contact":
            response_probability = 0.5 * _aligned_probability(
                models["response_extra_trees"], summary, list(labels["response_level"])
            )[0] + 0.5 * _aligned_probability(
                models["response_rbf_svm"], summary, list(labels["response_level"])
            )[0]
            position_index = int(np.argmax(position_probability))
            response_index = int(np.argmax(response_probability))
            output["position"] = {
                "label": str(labels["position"][position_index]),
                "confidence": float(position_probability[position_index]),
                "inference_mode": position_inference_mode,
                "probabilities": {
                    label: float(position_probability[index])
                    for index, label in enumerate(labels["position"])
                },
            }
            output["response_level"] = {
                "label": str(labels["response_level"][response_index]),
                "confidence": float(response_probability[response_index]),
                "probabilities": {
                    label: float(response_probability[index])
                    for index, label in enumerate(labels["response_level"])
                },
            }
            hard_label_index = list(labels["response_level"]).index("hard")
            release_guard = self._release_guard.update(
                hard_probability=float(response_probability[hard_label_index]),
                release_event_probability=release_event_probability,
                contact_probability=float(
                    contact_probability[list(labels["contact"]).index("contact")]
                ),
                raw_contact_label=contact_label,
            )
            output["release_guard"] = release_guard
            if release_guard["release_latched"]:
                output["raw_prediction_before_release_guard"] = {
                    "contact": output["contact"],
                    "position": output["position"],
                    "response_level": output["response_level"],
                }
                output["contact"] = {
                    "label": "no_contact",
                    "confidence": None,
                    "probabilities": {},
                    "decision_source": "confirmed_release_residual_latch",
                }
                output["position"] = None
                output["response_level"] = None
                output["status"] = "released_residual_latched"
                output["operational_state"] = "no_contact_after_confirmed_release"
            else:
                output["operational_state"] = "active_contact"

        raw_position = output.get("position")
        position_confidence = (
            float(raw_position.get("confidence"))
            if isinstance(raw_position, dict)
            and raw_position.get("confidence") is not None
            else None
        )
        recovery, recovered_baseline = self._runtime_baseline_recovery.observe(
            current,
            physical_frame=physical_frame,
            external_no_contact_hint=external_no_contact_hint,
            release_event_probability=release_event_probability,
            position_confidence=position_confidence,
            contact_probability=float(
                contact_probability[list(labels["contact"]).index("contact")]
            ),
            contact_label=contact_label,
            baseline_spectrum=self._baseline_spectrum,
            timestamp_sec=source_timestamp_sec,
        )
        output["runtime_baseline_recovery"] = recovery
        if recovered_baseline is not None:
            raw_prediction = {
                "contact": output.get("contact"),
                "position": output.get("position"),
                "response_level": output.get("response_level"),
            }
            self._baseline_spectrum = recovered_baseline.copy()
            self._pending_runtime_baseline_update = {
                "wavelength_nm": self._wavelength_nm.copy(),
                "intensity": recovered_baseline.copy(),
                "sample_count": recovery.get("stable_release_physical_frames"),
                "span_sec": recovery.get("quiet_elapsed_sec"),
                "shape_motion_rms": recovery.get("shape_motion_rms"),
                "common_gain_motion": recovery.get("common_gain_motion"),
                "policy": recovery.get("policy"),
            }
            self._prime_feature_history_from_baseline()
            self._frame_counter = 0
            self._release_guard.reset()
            self._last_ready_output = None
            self._runtime_baseline_revision += 1
            warmed = self._warming_output("runtime_reference_reanchored")
            warmed["raw_prediction_before_runtime_reanchor"] = raw_prediction
            warmed["runtime_baseline_recovery"] = recovery
            warmed["runtime_baseline_revision"] = self._runtime_baseline_revision
            return warmed

        if recovery.get("suppress_contact") and output.get("operational_state") == "active_contact":
            output["raw_prediction_before_runtime_rest_gate"] = {
                "contact": output.get("contact"),
                "position": output.get("position"),
                "response_level": output.get("response_level"),
            }
            output["contact"] = {
                "label": "no_contact",
                "confidence": None,
                "probabilities": {},
                "decision_source": "stable_post_release_runtime_rest_gate",
            }
            output["position"] = None
            output["response_level"] = None
            output["status"] = "runtime_rest_recovery_hold"
            output["operational_state"] = "no_contact_during_runtime_reference_recovery"
        output["digital_twin_proxy"] = dynamic_prediction_to_twin_proxy(output)
        self._last_ready_output = deepcopy(output)
        return output


__all__ = [
    "DynamicTemporalShadowAdapter",
    "ReleaseResidualGuard",
    "RuntimeBaselineRecoveryGuard",
    "load_dynamic_shadow_bundle",
]
