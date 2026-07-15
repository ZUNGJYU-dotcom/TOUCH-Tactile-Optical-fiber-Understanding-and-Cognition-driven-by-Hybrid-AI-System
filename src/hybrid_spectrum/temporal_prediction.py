"""Frame-aware temporal stabilization for static spectral predictions."""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import threading
from typing import Any, Hashable


class TemporalStaticPredictionStabilizer:
    """Debounce contact and vote position/level over unique spectral frames.

    The class does not infer new labels. It only stabilizes consecutive outputs
    from one model. Duplicate HTTP polls are ignored, and a new runtime baseline
    clears all previous evidence.
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        minimum_contact_frames: int = 3,
        release_frames: int = 2,
        minimum_position_support: float = 0.60,
        minimum_level_support: float = 0.60,
    ) -> None:
        if window_size < 3:
            raise ValueError("window_size must be at least 3")
        if not 1 <= minimum_contact_frames <= window_size:
            raise ValueError("minimum_contact_frames must be within the window")
        if not 1 <= release_frames <= window_size:
            raise ValueError("release_frames must be within the window")
        self.window_size = int(window_size)
        self.minimum_contact_frames = int(minimum_contact_frames)
        self.release_frames = int(release_frames)
        self.minimum_position_support = float(minimum_position_support)
        self.minimum_level_support = float(minimum_level_support)
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=self.window_size)
        self._baseline_token: Hashable | None = None
        self._last_frame_id: Hashable | None = None
        self._stable_contact = "no_contact"
        self._stable_position: str | None = None
        self._stable_level: str | None = None
        self._last_output = self._empty_output("not_initialized")

    def _empty_output(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "ready": status == "stable_no_contact",
            "contact_label": "no_contact",
            "position_label": None,
            "force_label": None,
            "position_support": None,
            "force_support": None,
            "contact_votes": 0,
            "history_unique_frames": len(self._history),
            "window_size": self.window_size,
            "minimum_contact_frames": self.minimum_contact_frames,
            "release_frames": self.release_frames,
            "semantics": "unique_frame_temporal_vote_diagnostic_only",
        }

    def reset(self, reason: str = "manual_reset") -> dict[str, Any]:
        with self._lock:
            self._reset_unlocked(reason)
            return deepcopy(self._last_output)

    def _reset_unlocked(self, reason: str) -> None:
        self._history.clear()
        self._last_frame_id = None
        self._stable_contact = "no_contact"
        self._stable_position = None
        self._stable_level = None
        self._last_output = self._empty_output(reason)

    @staticmethod
    def _component_label(prediction: dict[str, Any], key: str) -> str | None:
        component = prediction.get(key)
        if not isinstance(component, dict):
            return None
        label = component.get("label")
        return str(label) if label is not None else None

    @staticmethod
    def _majority_with_latest_tie(values: list[str]) -> tuple[str | None, float | None]:
        if not values:
            return None, None
        counts = Counter(values)
        maximum = max(counts.values())
        tied = {label for label, count in counts.items() if count == maximum}
        selected = next(label for label in reversed(values) if label in tied)
        return selected, counts[selected] / len(values)

    def update(
        self,
        *,
        frame_id: Hashable,
        prediction: dict[str, Any],
        baseline_token: Hashable,
        timestamp: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            if baseline_token != self._baseline_token:
                self._baseline_token = baseline_token
                self._reset_unlocked("baseline_changed_history_reset")
            if frame_id == self._last_frame_id:
                result = deepcopy(self._last_output)
                result["duplicate_frame_ignored"] = True
                return result

            contact = self._component_label(prediction, "contact") or "no_contact"
            position = self._component_label(prediction, "position")
            level = self._component_label(prediction, "force_level")
            position_component = prediction.get("position")
            diagnostics = (
                position_component.get("ensemble_diagnostics")
                if isinstance(position_component, dict)
                else None
            )
            agreement = (
                diagnostics.get("agreement_fraction")
                if isinstance(diagnostics, dict)
                else None
            )
            self._history.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "contact": contact,
                    "position": position,
                    "force": level,
                    "position_agreement": agreement,
                }
            )
            self._last_frame_id = frame_id

            recent = list(self._history)
            contact_rows = [row for row in recent if row["contact"] == "contact"]
            trailing_no_contact = 0
            for row in reversed(recent):
                if row["contact"] != "no_contact":
                    break
                trailing_no_contact += 1

            was_stable_contact = self._stable_contact == "contact"
            if was_stable_contact:
                stable_contact = trailing_no_contact < self.release_frames
            else:
                stable_contact = len(contact_rows) >= self.minimum_contact_frames

            if was_stable_contact and not stable_contact:
                # A confirmed release is a trial boundary. Retain only the
                # trailing release frames so the next press must earn fresh
                # contact, position, and level votes.
                release_rows = recent[-trailing_no_contact:]
                self._history.clear()
                self._history.extend(release_rows)
                recent = list(self._history)
                contact_rows = []

            if not stable_contact:
                self._stable_contact = "no_contact"
                self._stable_position = None
                self._stable_level = None
                status = (
                    "stable_no_contact"
                    if contact == "no_contact"
                    else "contact_warming_up"
                )
                output = self._empty_output(status)
                output.update(
                    {
                        "raw_contact_label": contact,
                        "raw_position_label": position,
                        "raw_force_label": level,
                        "contact_votes": len(contact_rows),
                        "history_unique_frames": len(recent),
                        "duplicate_frame_ignored": False,
                    }
                )
                self._last_output = output
                return deepcopy(output)

            position_values = [
                str(row["position"])
                for row in contact_rows
                if row.get("position") is not None
            ]
            level_values = [
                str(row["force"])
                for row in contact_rows
                if row.get("force") is not None
            ]
            selected_position, position_support = self._majority_with_latest_tie(
                position_values
            )
            selected_level, level_support = self._majority_with_latest_tie(level_values)
            position_ready = bool(
                selected_position is not None
                and position_support is not None
                and position_support >= self.minimum_position_support
            )
            level_ready = bool(
                selected_level is not None
                and level_support is not None
                and level_support >= self.minimum_level_support
            )
            self._stable_contact = "contact"
            self._stable_position = selected_position if position_ready else None
            self._stable_level = selected_level if level_ready else None
            status = "stable_contact" if position_ready and level_ready else "contact_uncertain"
            output = {
                "status": status,
                "ready": status == "stable_contact",
                "contact_label": "contact",
                "position_label": self._stable_position,
                "force_label": self._stable_level,
                "position_support": position_support,
                "force_support": level_support,
                "contact_votes": len(contact_rows),
                "history_unique_frames": len(recent),
                "window_size": self.window_size,
                "minimum_contact_frames": self.minimum_contact_frames,
                "release_frames": self.release_frames,
                "raw_contact_label": contact,
                "raw_position_label": position,
                "raw_force_label": level,
                "raw_position_agreement": agreement,
                "duplicate_frame_ignored": False,
                "semantics": "unique_frame_temporal_vote_diagnostic_only",
            }
            self._last_output = output
            return deepcopy(output)


__all__ = ["TemporalStaticPredictionStabilizer"]
