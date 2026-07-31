"""Auditable display-only processing for BaySpec spectrum frames.

The trained recognition pipeline continues to receive the original ``intensity``
array.  This module produces separate display and overlay arrays so visual
enhancement cannot silently change model features or saved raw evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - packaged runtime always includes yaml
    yaml = None


DEFAULT_SETTINGS: dict[str, Any] = {
    "integration_us": 5000,
    "update_overlay_data": True,
    "subtract_background": False,
    "baseline_correction": True,
    "spectrum_smoothing": True,
    "normalize_spectrum": True,
    "smoothing_window": 7,
    "baseline_window": 41,
    "baseline_percentile": 18.0,
    "model_input_source": "raw_intensity",
}


def _odd_window(value: Any, *, minimum: int, maximum: int) -> int:
    number = max(minimum, min(int(value), maximum))
    return number if number % 2 else number + 1


class SpectrumDisplayProcessor:
    """Build a readable display spectrum while retaining raw counts."""

    def __init__(
        self,
        config_path: Path | None = None,
        user_settings_path: Path | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.config_path = Path(config_path) if config_path else None
        self.user_settings_path = (
            Path(user_settings_path)
            if user_settings_path is not None
            else self._default_user_settings_path()
        )
        self.settings = dict(DEFAULT_SETTINGS)
        self.background_reference: np.ndarray | None = None
        self.overlay_spectrum: np.ndarray | None = None
        self.last_raw_spectrum: np.ndarray | None = None
        self.last_display_spectrum: np.ndarray | None = None
        self.frame_count = 0
        self._load_defaults()
        self._load_user_settings()

    @staticmethod
    def _default_user_settings_path() -> Path:
        local_root = Path(
            os.environ.get("LOCALAPPDATA")
            or (Path.home() / "AppData" / "Local")
        )
        return local_root / "TOUCH" / "spectrum_processing.json"

    def _load_defaults(self) -> None:
        if self.config_path is None or not self.config_path.exists() or yaml is None:
            return
        try:
            payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            configured = payload.get("spectrum_processing", payload)
            if isinstance(configured, dict):
                self.settings.update(configured)
        except Exception:
            # A bad optional display config must never break hardware acquisition.
            return
        self.settings = self._validated_settings(self.settings)

    def _load_user_settings(self) -> None:
        path = self.user_settings_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.settings.update(payload)
                self.settings = self._validated_settings(self.settings)
        except Exception:
            return

    @staticmethod
    def _validated_settings(settings: dict[str, Any]) -> dict[str, Any]:
        validated = dict(DEFAULT_SETTINGS)
        validated.update(settings)
        validated["integration_us"] = max(
            1000, min(int(validated.get("integration_us", 5000)), 1_000_000)
        )
        for name in (
            "update_overlay_data",
            "subtract_background",
            "baseline_correction",
            "spectrum_smoothing",
            "normalize_spectrum",
        ):
            validated[name] = bool(validated.get(name))
        validated["smoothing_window"] = _odd_window(
            validated.get("smoothing_window", 7),
            minimum=5,
            maximum=31,
        )
        validated["baseline_window"] = _odd_window(
            validated.get("baseline_window", 41),
            minimum=11,
            maximum=151,
        )
        validated["baseline_percentile"] = max(
            1.0, min(float(validated.get("baseline_percentile", 18.0)), 45.0)
        )
        # This is intentionally fixed. Display controls must not redirect the
        # trained model to processed data.
        validated["model_input_source"] = "raw_intensity"
        return validated

    def _persist_user_settings(self) -> None:
        path = self.user_settings_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.settings, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # Read-only environments still retain settings for this session.
            pass

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = set(DEFAULT_SETTINGS) - {"model_input_source"}
        with self.lock:
            merged = dict(self.settings)
            for key, value in updates.items():
                if key in allowed:
                    merged[key] = value
            self.settings = self._validated_settings(merged)
            self._persist_user_settings()
            return self.status()

    def reset_session(self) -> None:
        with self.lock:
            self.overlay_spectrum = None
            self.last_raw_spectrum = None
            self.last_display_spectrum = None
            self.frame_count = 0

    def capture_background(self) -> dict[str, Any]:
        with self.lock:
            if self.last_raw_spectrum is None:
                return {
                    "ok": False,
                    "status": "no_spectrum_frame",
                    "message": "Acquire a dark frame before capturing background.",
                }
            self.background_reference = self.last_raw_spectrum.copy()
            return {
                "ok": True,
                "status": "background_captured",
                "points": int(self.background_reference.size),
            }

    def clear_background(self) -> dict[str, Any]:
        with self.lock:
            self.background_reference = None
            return {"ok": True, "status": "background_cleared"}

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if values.size < 3:
            return values.copy()
        window = min(_odd_window(window, minimum=3, maximum=151), values.size)
        if window % 2 == 0:
            window = max(3, window - 1)
        pad = window // 2
        padded = np.pad(values, pad, mode="edge")
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if values.size < 5:
            return values.copy()
        # Seven-point Savitzky-Golay coefficients retain narrow FBG peaks
        # better than a boxcar average. Wider selections apply a second gentle
        # moving-average pass instead of inventing unsupported vendor behavior.
        padded = np.pad(values, 3, mode="edge")
        coefficients = np.asarray([-2, 3, 6, 7, 6, 3, -2], dtype=float) / 21.0
        smoothed = np.convolve(padded, coefficients, mode="valid")
        if window > 7:
            smoothed = SpectrumDisplayProcessor._moving_average(
                smoothed,
                min(window - 4, 11),
            )
        return smoothed

    @staticmethod
    def _baseline(values: np.ndarray, window: int, percentile: float) -> np.ndarray:
        if values.size < 5:
            return np.full_like(values, float(np.min(values)))
        window = min(window, values.size)
        if window % 2 == 0:
            window = max(3, window - 1)
        pad = window // 2
        padded = np.pad(values, pad, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        floor = np.percentile(windows, percentile, axis=-1)
        return SpectrumDisplayProcessor._moving_average(
            floor,
            min(window, 31),
        )

    def process(self, raw_counts: list[float] | np.ndarray) -> dict[str, Any]:
        raw = np.asarray(raw_counts, dtype=float).reshape(-1)
        if raw.size == 0 or not np.all(np.isfinite(raw)):
            raise ValueError("raw spectrum must contain finite values")

        with self.lock:
            settings = dict(self.settings)
            self.last_raw_spectrum = raw.copy()
            display = raw.copy()
            steps: list[str] = []
            warnings: list[str] = []

            if settings["subtract_background"]:
                if (
                    self.background_reference is not None
                    and self.background_reference.size == display.size
                ):
                    display = np.maximum(
                        display - self.background_reference,
                        0.0,
                    )
                    steps.append("background_subtraction")
                else:
                    warnings.append("background_reference_required")

            if settings["baseline_correction"]:
                baseline = self._baseline(
                    display,
                    int(settings["baseline_window"]),
                    float(settings["baseline_percentile"]),
                )
                display = np.maximum(display - baseline, 0.0)
                steps.append("baseline_correction")

            if settings["spectrum_smoothing"]:
                display = np.maximum(
                    self._smooth(display, int(settings["smoothing_window"])),
                    0.0,
                )
                steps.append("spectrum_smoothing")

            overlay = (
                self.overlay_spectrum.copy()
                if self.overlay_spectrum is not None
                and self.overlay_spectrum.size == display.size
                else None
            )
            if settings["update_overlay_data"]:
                self.overlay_spectrum = display.copy()

            self.last_display_spectrum = display.copy()
            self.frame_count += 1
            raw_noise = float(np.median(np.abs(np.diff(raw)))) if raw.size > 1 else 0.0
            display_noise = (
                float(np.median(np.abs(np.diff(display)))) if display.size > 1 else 0.0
            )
            return {
                "display_intensity": display.tolist(),
                "overlay_intensity": overlay.tolist() if overlay is not None else [],
                "spectrum_processing": {
                    "steps": steps,
                    "warnings": warnings,
                    "raw_retained": True,
                    "model_input_source": "raw_intensity",
                    "display_input_source": "display_intensity",
                    "background_reference_ready": self.background_reference is not None,
                    "raw_roughness": raw_noise,
                    "display_roughness": display_noise,
                    "frame_count": self.frame_count,
                    "normalization_requested": bool(
                        settings["normalize_spectrum"]
                    ),
                },
            }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "settings": dict(self.settings),
                "background_reference_ready": self.background_reference is not None,
                "background_reference_points": (
                    int(self.background_reference.size)
                    if self.background_reference is not None
                    else 0
                ),
                "overlay_ready": self.overlay_spectrum is not None,
                "last_frame_points": (
                    int(self.last_raw_spectrum.size)
                    if self.last_raw_spectrum is not None
                    else 0
                ),
                "processed_frame_count": self.frame_count,
                "raw_spectrum_retained": True,
                "model_input_source": "raw_intensity",
                "display_input_source": "display_intensity",
                "config_path": str(self.config_path) if self.config_path else None,
                "user_settings_path": str(self.user_settings_path),
            }
