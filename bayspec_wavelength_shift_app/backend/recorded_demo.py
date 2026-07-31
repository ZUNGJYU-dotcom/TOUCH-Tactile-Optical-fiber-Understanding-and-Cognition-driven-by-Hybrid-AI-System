"""Recorded-spectrum reference library for the TOUCH demo runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


POSITION_PATHS = {
    "vertical_slide_p11_p12_p13": ("P11", "P12", "P13"),
    "horizontal_slide_p11_p21_p31": ("P11", "P21", "P31"),
    "diagonal_slide_p11_p22_p33": ("P11", "P22", "P33"),
}
DIRECT_SCENARIO_POSITION = {
    "center_press": "P22",
    "p21_contact": "P21",
    "p12_contact": "P12",
    "p32_contact": "P32",
    "off_center_fingertip_contact": "P23",
    "broad_fingertip_contact": "P22",
}


class RecordedDemoLibrary:
    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = Path(asset_dir)
        self.npz_path = self.asset_dir / "recorded_demo_20260731.npz"
        self.metadata_path = self.asset_dir / "recorded_demo_20260731.json"
        self._lock = Lock()
        self._arrays: dict[str, np.ndarray] | None = None
        self._metadata: dict[str, Any] | None = None

    def _load(self) -> None:
        if self._arrays is not None:
            return
        with self._lock:
            if self._arrays is not None:
                return
            if not self.npz_path.exists() or not self.metadata_path.exists():
                raise FileNotFoundError(
                    "Recorded demo asset is missing. Run scripts/build_recorded_demo_templates.py."
                )
            with np.load(self.npz_path, allow_pickle=False) as payload:
                self._arrays = {key: np.asarray(payload[key]) for key in payload.files}
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        try:
            self._load()
        except Exception as exc:
            return {
                "available": False,
                "status": "recorded_demo_asset_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        assert self._arrays is not None
        return {
            "available": True,
            "status": "recorded_real_spectrum_ready",
            "dataset_id": (self._metadata or {}).get("dataset_id"),
            "capture_date": (self._metadata or {}).get("capture_date"),
            "positions": [str(value) for value in self._arrays["position_order"]],
            "frames_per_position": int(self._arrays["spectra"].shape[1]),
            "spectrum_points": int(self._arrays["spectra"].shape[2]),
            "peak_discovery": (self._metadata or {}).get("peak_discovery"),
        }

    def _position_index(self, position: str) -> int:
        assert self._arrays is not None
        values = [str(value) for value in self._arrays["position_order"]]
        return values.index(position)

    def _nearest_force_frame(
        self, position: str, target_force_n: float, prefer_release: bool = False
    ) -> int:
        assert self._arrays is not None
        position_index = self._position_index(position)
        force = self._arrays["force_fz_n"][position_index]
        peak_index = int(np.argmax(force))
        if prefer_release:
            candidates = np.arange(peak_index, force.size)
        else:
            candidates = np.arange(0, max(peak_index + 1, 1))
        local = int(np.argmin(np.abs(force[candidates] - float(target_force_n))))
        return int(candidates[local])

    @staticmethod
    def _path_position(path: tuple[str, ...], step: int, step_count: int = 12) -> str:
        phase = min(max(int(step), 0), step_count - 1) / max(step_count - 1, 1)
        return path[min(len(path) - 1, int(round(phase * (len(path) - 1))))]

    def reference_frame(
        self, scenario: str, step: int, desired_force_n: float | None = None
    ) -> dict[str, Any]:
        self._load()
        assert self._arrays is not None
        metadata = self._metadata or {}
        step = max(0, int(step))
        kind = "recorded_action_sequence"

        if scenario == "no_contact":
            position = "NO_CONTACT"
            frame_index = step % self._arrays["spectra"].shape[1]
        elif scenario in DIRECT_SCENARIO_POSITION:
            position = DIRECT_SCENARIO_POSITION[scenario]
            frame_index = min(step, self._arrays["spectra"].shape[1] - 1)
            if scenario in {"off_center_fingertip_contact", "broad_fingertip_contact"}:
                kind = "recorded_position_sequence_composite"
        elif scenario in POSITION_PATHS:
            position = self._path_position(POSITION_PATHS[scenario], step)
            target = 3.5 if desired_force_n is None else desired_force_n
            frame_index = self._nearest_force_frame(position, target)
            kind = "recorded_multi_position_composite_not_measured_slide"
        elif scenario == "tap":
            position = "P22"
            target = 0.0 if desired_force_n is None else desired_force_n
            frame_index = self._nearest_force_frame(
                position, target, prefer_release=step >= 3
            )
            kind = "recorded_contact_frames_composite_not_measured_tap"
        elif scenario == "release":
            position = "P22"
            target = 0.0 if desired_force_n is None else desired_force_n
            frame_index = self._nearest_force_frame(position, target, prefer_release=True)
            kind = "recorded_release_branch_reference"
        else:
            position = "P22"
            frame_index = min(step, self._arrays["spectra"].shape[1] - 1)
            kind = "recorded_position_sequence_composite"

        position_index = self._position_index(position)
        session = (metadata.get("sessions") or {}).get(position, {})
        spectrum = self._arrays["spectra"][position_index, frame_index]
        force_n = float(self._arrays["force_fz_n"][position_index, frame_index])
        return {
            "source_position": position,
            "source_session_id": session.get("session_id"),
            "source_capture_index": int(
                self._arrays["capture_index"][position_index, frame_index]
            ),
            "source_elapsed_time_sec": float(
                self._arrays["elapsed_time_sec"][position_index, frame_index]
            ),
            "reference_force_fz_n": force_n,
            "response_ratio": max(0.0, min(1.0, force_n / 5.0)),
            "wavelength_nm": self._arrays["wavelength_nm"],
            "intensity": spectrum,
            "baseline_intensity": self._arrays["baseline_intensity"],
            "peak_reference_wavelength_nm": self._arrays[
                "peak_reference_wavelength_nm"
            ],
            "demo_reference_kind": kind,
            "dataset_id": metadata.get("dataset_id"),
            "capture_date": metadata.get("capture_date"),
        }

    def spectrum_payload(
        self,
        reference: dict[str, Any],
        channels: list[dict[str, Any]],
        frame_id: int,
        timestamp: float,
    ) -> dict[str, Any]:
        wavelengths = np.asarray(reference["wavelength_nm"], dtype=np.float64)
        intensity = np.asarray(reference["intensity"], dtype=np.float64)
        baseline = np.asarray(reference["baseline_intensity"], dtype=np.float64)
        peak_references = np.asarray(
            reference["peak_reference_wavelength_nm"], dtype=np.float64
        )
        channel_by_id = {
            str(channel.get("channel_id")): channel for channel in channels
        }
        peaks: list[dict[str, Any]] = []
        dominant_channel = str(reference.get("source_position") or "P22")
        if dominant_channel == "NO_CONTACT":
            dominant_channel = "P22"

        channel_order = [str(value) for value in self._arrays["channel_order"]]
        for peak_order, (channel_id, reference_wavelength) in enumerate(
            zip(channel_order, peak_references, strict=True),
            start=1,
        ):
            indices = np.flatnonzero(np.abs(wavelengths - reference_wavelength) <= 0.85)
            if indices.size == 0:
                continue
            peak_index = int(indices[int(np.argmax(intensity[indices]))])
            baseline_peak_index = int(indices[int(np.argmax(baseline[indices]))])
            tracked = float(wavelengths[peak_index])
            baseline_wavelength = float(wavelengths[baseline_peak_index])
            current = float(intensity[peak_index])
            baseline_counts = float(max(baseline[baseline_peak_index], 1e-12))
            relative = current / baseline_counts
            shift_pm = (tracked - baseline_wavelength) * 1000.0
            channel = channel_by_id.get(channel_id, {})
            peaks.append(
                {
                    "candidate_id": f"FBG{peak_order:02d}",
                    "provisional_channel_id": channel_id,
                    "channel_id": channel_id,
                    "target_wavelength_nm": float(reference_wavelength),
                    "candidate_reference_wavelength_nm": float(reference_wavelength),
                    "baseline_wavelength_nm": baseline_wavelength,
                    "tracked_wavelength_nm": tracked,
                    "peak_wavelength_nm": tracked,
                    "delta_wavelength_nm": shift_pm / 1000.0,
                    "delta_wavelength_pm": shift_pm,
                    "absolute_shift_pm": abs(shift_pm),
                    "shift_direction": "stable" if abs(shift_pm) < 5.0 else ("red_shift" if shift_pm > 0 else "blue_shift"),
                    "intensity_counts": current,
                    "baseline_intensity_counts": baseline_counts,
                    "relative_intensity": relative,
                    "attenuation_ratio": 1.0 - relative,
                    "intensity_loss_db": -10.0 * math.log10(max(relative, 1e-12)),
                    "wavelength_tracking_method": "recorded_frame_local_peak_max",
                    "peak_assignment_method": "automatic_no_contact_discovery_then_local_tracking",
                    "candidate_mapping": True,
                    "physical_channel_mapping_final": False,
                    "mapping_basis": "ascending_wavelength_order_after_automatic_discovery",
                    "wavelength_shift_response_ratio": channel.get(
                        "wavelength_shift_response_ratio"
                    ),
                    "observed_wavelength_shift_response_ratio": channel.get(
                        "observed_wavelength_shift_response_ratio"
                    ),
                    "local_response_estimate": channel.get("local_response_estimate"),
                    "dominant": channel_id == dominant_channel,
                    "qa_status": "recorded_reference",
                    "simulated": False,
                }
            )

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "mode": "recorded_real_spectrum_reference",
            "spectrum_type": "recorded real BaySpec 512-point spectrum",
            "axis_type": "wavelength_nm",
            "wavelength_nm": [round(float(value), 4) for value in wavelengths],
            "intensity": [round(float(value), 2) for value in intensity],
            "peaks": peaks,
            "dominant_channel": dominant_channel,
            "selected_channel": dominant_channel,
            "frame_render_semantics": "replace_previous_spectrum",
            "source_note": (
                f"Recorded 2026-07-31 synchronized spectrum: "
                f"{reference.get('source_session_id')} frame "
                f"{reference.get('source_capture_index')}"
            ),
            "source_session_id": reference.get("source_session_id"),
            "source_capture_index": reference.get("source_capture_index"),
            "source_elapsed_time_sec": reference.get("source_elapsed_time_sec"),
            "source_position": reference.get("source_position"),
            "reference_force_fz_n": reference.get("reference_force_fz_n"),
            "demo_reference_kind": reference.get("demo_reference_kind"),
            "dataset_id": reference.get("dataset_id"),
            "capture_date": reference.get("capture_date"),
            "intensity_modulation_enabled": True,
            "peak_height_mode": "recorded_counts",
            "spectrum_peak_profile": "recorded_auto_discovered_9fbg",
            "spectrum_peak_mapping_status": "wavelength_order_assignment",
            "peak_assignment_method": "automatic_no_contact_discovery_then_local_tracking",
            "physical_channel_mapping_final": False,
        }
