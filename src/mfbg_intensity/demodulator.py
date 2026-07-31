"""Nine-channel mFBG spectral-window optical-intensity demodulation."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Iterable

import numpy as np

from src.array_surface.surface_mapper import SurfaceConfig, map_surface

from .config import MfbgChannelConfig, MfbgIntensityProfile, load_profile


def _mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def _finite_spectrum(
    wavelength_nm: Iterable[Any],
    intensity_counts: Iterable[Any],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(wavelength_nm), dtype=float)
    y = np.asarray(list(intensity_counts), dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("wavelength_nm and intensity_counts must be equal 1D arrays")
    if x.size < 3:
        raise ValueError("spectrum requires at least three points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("spectrum contains non-finite values")
    order = np.argsort(x)
    return x[order], y[order]


class MfbgIntensityDemodulator:
    """Stateful baseline-aware mFBG intensity demodulator.

    Each channel is measured from a tracked, fixed-width spectral window. The
    resulting attenuation is a coupled optical response, not a force or
    pressure estimate.
    """

    def __init__(
        self,
        profile: MfbgIntensityProfile | None = None,
    ) -> None:
        self.profile = profile or load_profile()
        self._lock = threading.RLock()
        self._baseline: dict[str, dict[str, float]] = {}
        self._latest_frame: dict[str, Any] | None = None
        self._frame_id = 0

    @property
    def baseline_ready(self) -> bool:
        with self._lock:
            return len(self._baseline) == len(self.profile.channel_order)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._baseline.clear()
            self._latest_frame = None
            self._frame_id = 0
        return {"ok": True, "baseline_ready": False, "frame_id": 0}

    def profile_summary(self) -> dict[str, Any]:
        result = self.profile.summary()
        result.update(
            {
                "baseline_ready": self.baseline_ready,
                "baseline_minimum_frames": self.profile.baseline_minimum_frames,
                "same_fiber_directed_paths": (
                    self.profile.raw.get("coupling", {}).get(
                        "same_fiber_directed_paths", {}
                    )
                ),
                "coupling_interpretation": (
                    self.profile.raw.get("coupling", {}).get("interpretation")
                ),
            }
        )
        return result

    def latest_frame(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._latest_frame is None else dict(self._latest_frame)

    def set_baseline(
        self,
        wavelength_nm: Iterable[Any],
        spectra: Iterable[Iterable[Any]],
    ) -> dict[str, Any]:
        frames = list(spectra)
        required = self.profile.baseline_minimum_frames
        if len(frames) < required:
            return {
                "ok": False,
                "reason": "insufficient_baseline_frames",
                "frame_count": len(frames),
                "minimum_frames": required,
            }

        extracted: dict[str, list[float]] = {
            channel_id: [] for channel_id in self.profile.channel_order
        }
        failures: dict[str, int] = {
            channel_id: 0 for channel_id in self.profile.channel_order
        }
        for raw_frame in frames:
            x, y = _finite_spectrum(wavelength_nm, raw_frame)
            for channel_id in self.profile.channel_order:
                feature = self._extract_channel(x, y, self.profile.channels[channel_id])
                if feature["valid"]:
                    extracted[channel_id].append(float(feature["intensity_counts"]))
                else:
                    failures[channel_id] += 1

        minimum_intensity = float(
            self.profile.raw.get("baseline", {}).get(
                "minimum_intensity_counts", 1.0
            )
        )
        candidate: dict[str, dict[str, float]] = {}
        invalid_channels: list[str] = []
        for channel_id, values in extracted.items():
            array = np.asarray(values, dtype=float)
            if array.size < required or float(np.median(array)) <= minimum_intensity:
                invalid_channels.append(channel_id)
                continue
            candidate[channel_id] = {
                "intensity_counts": float(np.median(array)),
                "noise_counts": _mad_sigma(array),
                "sample_count": float(array.size),
            }

        if invalid_channels:
            return {
                "ok": False,
                "reason": "invalid_baseline_channels",
                "invalid_channel_ids": invalid_channels,
                "extraction_failures": failures,
                "frame_count": len(frames),
            }

        with self._lock:
            self._baseline = candidate
        return {
            "ok": True,
            "baseline_ready": True,
            "frame_count": len(frames),
            "channels": candidate,
        }

    def analyze_spectrum(
        self,
        wavelength_nm: Iterable[Any],
        intensity_counts: Iterable[Any],
        *,
        timestamp: float | None = None,
        source: str = "manual_api",
        include_spectrum: bool | None = None,
    ) -> dict[str, Any]:
        x, y = _finite_spectrum(wavelength_nm, intensity_counts)
        retain_spectrum = (
            bool(
                self.profile.raw.get("demodulation", {}).get(
                    "retain_raw_spectrum_in_frame", True
                )
            )
            if include_spectrum is None
            else bool(include_spectrum)
        )
        with self._lock:
            baseline = {
                channel_id: dict(item) for channel_id, item in self._baseline.items()
            }

        channel_records: list[dict[str, Any]] = []
        for channel_id in self.profile.channel_order:
            feature = self._extract_channel(
                x,
                y,
                self.profile.channels[channel_id],
            )
            channel_records.append(
                self._apply_baseline(
                    feature,
                    self.profile.channels[channel_id],
                    baseline.get(channel_id),
                )
            )

        frame = self._build_frame(
            channel_records,
            timestamp=float(timestamp if timestamp is not None else time.time()),
            source=source,
        )
        if retain_spectrum:
            frame["raw_spectrum"] = {
                "wavelength_nm": x.tolist(),
                "intensity_counts": y.tolist(),
                "point_count": int(x.size),
            }

        with self._lock:
            self._frame_id += 1
            frame["frame_id"] = self._frame_id
            self._latest_frame = frame
        return frame

    def _extract_channel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        channel: MfbgChannelConfig,
    ) -> dict[str, Any]:
        config = self.profile.raw.get("demodulation", {})
        minimum_points = int(config.get("minimum_window_points", 3))
        reference = channel.demodulation_wavelength_nm
        search_mask = np.abs(x - reference) <= channel.search_half_width_nm
        search_x = x[search_mask]
        search_y = y[search_mask]
        base = {
            "channel_id": channel.channel_id,
            "target_wavelength_nm": channel.target_wavelength_nm,
            "measured_wavelength_nm": channel.measured_wavelength_nm,
            "demodulation_wavelength_nm": reference,
            "fiber_id": channel.fiber_id,
            "fiber_order": channel.fiber_order,
            "x": channel.x,
            "y": channel.y,
            "response_polarity": channel.response_polarity,
            "enabled_for_real_demodulation": channel.enabled_for_real_demodulation,
            "qa_flags": [],
        }
        if search_x.size < minimum_points:
            base.update(
                {
                    "valid": False,
                    "qa_status": "invalid",
                    "qa_flags": ["insufficient_points_in_search_window"],
                    "point_count": int(search_x.size),
                }
            )
            return base

        edge_fraction = float(config.get("local_baseline_edge_fraction", 0.2))
        edge_count = max(1, int(round(search_y.size * edge_fraction)))
        edge_values = np.concatenate((search_y[:edge_count], search_y[-edge_count:]))
        local_baseline = float(np.median(edge_values))
        feature_mode = str(config.get("feature_mode", "peak"))
        if feature_mode == "dip":
            weights = np.clip(local_baseline - search_y, 0.0, None)
            marker_index = int(np.argmin(search_y))
            feature_amplitude = local_baseline - float(np.min(search_y))
        else:
            weights = np.clip(search_y - local_baseline, 0.0, None)
            marker_index = int(np.argmax(search_y))
            feature_amplitude = float(np.max(search_y)) - local_baseline

        if float(np.sum(weights)) > 1.0e-12:
            tracked = float(np.sum(search_x * weights) / np.sum(weights))
        else:
            tracked = float(search_x[marker_index])
        maximum_offset = float(config.get("maximum_tracking_offset_nm", 0.5))
        tracking_offset = tracked - reference
        if abs(tracking_offset) > maximum_offset:
            tracked = reference + math.copysign(maximum_offset, tracking_offset)
            base["qa_flags"].append("peak_tracking_offset_limited")

        integration_mask = np.abs(x - tracked) <= channel.integration_half_width_nm
        integration_x = x[integration_mask]
        integration_y = y[integration_mask]
        if integration_x.size < minimum_points:
            base.update(
                {
                    "valid": False,
                    "qa_status": "invalid",
                    "qa_flags": base["qa_flags"]
                    + ["insufficient_points_in_integration_window"],
                    "point_count": int(integration_x.size),
                }
            )
            return base

        span_nm = float(integration_x[-1] - integration_x[0])
        integrated = float(np.trapezoid(integration_y, integration_x))
        primary_intensity = (
            integrated / span_nm if span_nm > 1.0e-12 else float(np.mean(integration_y))
        )
        noise = _mad_sigma(edge_values)
        snr = feature_amplitude / max(noise, 1.0e-12)
        minimum_snr = float(config.get("minimum_peak_snr_warning", 3.0))
        if snr < minimum_snr:
            base["qa_flags"].append("peak_snr_low")

        base.update(
            {
                "valid": True,
                "qa_status": "warning" if base["qa_flags"] else "ok",
                "feature_mode": feature_mode,
                "point_count": int(integration_x.size),
                "tracked_wavelength_nm": tracked,
                "tracking_offset_pm": (tracked - reference) * 1000.0,
                "peak_marker_wavelength_nm": float(search_x[marker_index]),
                "peak_marker_intensity_counts": float(search_y[marker_index]),
                "local_baseline_counts": local_baseline,
                "peak_snr": snr,
                "intensity_counts": primary_intensity,
                "integrated_intensity_count_nm": integrated,
                "integration_span_nm": span_nm,
                "intensity_extraction_method": "tracked_window_integrated_mean",
            }
        )
        return base

    def _apply_baseline(
        self,
        feature: dict[str, Any],
        channel: MfbgChannelConfig,
        baseline: dict[str, float] | None,
    ) -> dict[str, Any]:
        result = dict(feature)
        result["baseline_ready"] = baseline is not None
        if not result.get("valid"):
            return result
        if baseline is None:
            result.update(
                {
                    "baseline_intensity_counts": None,
                    "baseline_noise_counts": None,
                    "relative_intensity": None,
                    "attenuation_ratio": None,
                    "attenuation_percent": None,
                    "loss_db": None,
                    "response_value": 0.0,
                    "responding": False,
                    "response_state": "baseline_required",
                }
            )
            result["qa_flags"] = list(result.get("qa_flags") or []) + [
                "baseline_not_ready"
            ]
            result["qa_status"] = "warning"
            return result

        epsilon = float(
            self.profile.raw.get("demodulation", {}).get("epsilon", 1.0e-12)
        )
        baseline_intensity = max(float(baseline["intensity_counts"]), epsilon)
        current_intensity = float(result["intensity_counts"])
        ratio = current_intensity / baseline_intensity
        attenuation = 1.0 - ratio
        loss_db = -10.0 * math.log10(max(ratio, epsilon))
        response_value = max(0.0, min(1.0, attenuation))
        responding = response_value >= self.profile.responding_threshold_ratio
        rise_warning = float(
            self.profile.raw.get("demodulation", {}).get(
                "intensity_rise_warning_ratio", 1.05
            )
        )
        if ratio > rise_warning:
            result["qa_flags"] = list(result.get("qa_flags") or []) + [
                "intensity_rise_anomaly"
            ]
            result["qa_status"] = "warning"

        result.update(
            {
                "baseline_intensity_counts": baseline_intensity,
                "baseline_noise_counts": float(baseline.get("noise_counts", 0.0)),
                "relative_intensity": ratio,
                "attenuation_ratio": attenuation,
                "attenuation_percent": attenuation * 100.0,
                "loss_db": loss_db,
                "response_value": response_value,
                "responding": responding,
                "response_state": "optical_response" if responding else "no_contact",
            }
        )
        return result

    def _build_frame(
        self,
        channel_records: list[dict[str, Any]],
        *,
        timestamp: float,
        source: str,
    ) -> dict[str, Any]:
        valid = [record for record in channel_records if record.get("valid")]
        responding = [record for record in valid if record.get("responding")]
        dominant = (
            max(responding, key=lambda record: float(record.get("response_value") or 0.0))
            if responding
            else None
        )
        channel_map = {record["channel_id"]: record for record in channel_records}
        regions = self._contact_regions(channel_map)
        surface_channels = []
        for record in channel_records:
            surface_channels.append(
                {
                    **record,
                    "enabled": bool(record.get("valid")),
                    "valid": bool(record.get("valid")),
                    "response_value": float(record.get("response_value") or 0.0),
                }
            )
        surface = map_surface(
            surface_channels,
            SurfaceConfig(
                surface_input_mode="raw_coupled_response_surface",
                active_threshold=self.profile.responding_threshold_ratio,
            ),
        )
        real_enabled_channel_ids = [
            channel_id
            for channel_id in self.profile.channel_order
            if self.profile.channels[channel_id].enabled_for_real_demodulation
        ]
        surface_metrics = dict(surface["surface_metrics"])
        surface_metrics.update(
            {
                "configured_channel_count": len(self.profile.channel_order),
                "analyzed_channel_count": len(valid),
                "real_enabled_channel_count": len(real_enabled_channel_ids),
            }
        )
        all_flags = sorted(
            {
                flag
                for record in channel_records
                for flag in (record.get("qa_flags") or [])
            }
        )
        invalid_count = sum(not bool(record.get("valid")) for record in channel_records)
        qa_status = (
            "invalid"
            if invalid_count == len(channel_records)
            else "warning"
            if all_flags
            else "ok"
        )
        return {
            "profile_id": self.profile.profile_id,
            "mode": "mfbg_intensity_3x3",
            "timestamp": timestamp,
            "source": source,
            "baseline_ready": self.baseline_ready,
            "channel_order": list(self.profile.channel_order),
            "display_rows": [list(row) for row in self.profile.display_rows],
            "channels": channel_records,
            "channel_map": channel_map,
            "configured_channel_count": len(self.profile.channel_order),
            "analyzed_channel_count": len(valid),
            "real_enabled_channel_ids": real_enabled_channel_ids,
            "real_enabled_channel_count": len(real_enabled_channel_ids),
            "intensity_vector": [
                record.get("intensity_counts") for record in channel_records
            ],
            "baseline_intensity_vector": [
                record.get("baseline_intensity_counts") for record in channel_records
            ],
            "relative_intensity_vector": [
                record.get("relative_intensity") for record in channel_records
            ],
            "attenuation_vector": [
                record.get("attenuation_ratio") for record in channel_records
            ],
            "loss_db_vector": [record.get("loss_db") for record in channel_records],
            "responding_channel_ids": [
                record["channel_id"] for record in responding
            ],
            "responding_channel_count": len(responding),
            "dominant_channel": dominant["channel_id"] if dominant else None,
            "peak_attenuation_ratio": (
                float(dominant["response_value"]) if dominant else 0.0
            ),
            "contact_regions": regions,
            "contact_region_count": len(regions),
            "multi_region_interface_ready": True,
            "dense_reconstruction_status": "interface_only_pending_real_training_data",
            "surface_grid": surface["surface_grid"],
            "grid_x": surface["grid_x"],
            "grid_y": surface["grid_y"],
            "surface_metrics": surface_metrics,
            "surface_semantics": "raw_coupled_optical_attenuation_proxy",
            "coupling_interpretation": (
                "mixed mechanical and optical coupling, not independent force pixels"
            ),
            "qa_status": qa_status,
            "qa_flags": all_flags,
            "runtime_activation_status": (
                "real_3x3_enabled"
                if self.profile.real_3x3_enabled
                else "disabled_pending_measured_wavelengths_and_baseline"
            ),
            "real_3x3_enabled": self.profile.real_3x3_enabled,
            "calibrated_force": False,
            "force_N_output": False,
            "calibrated_pressure_output": False,
        }

    def _contact_regions(
        self,
        channel_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        active = {
            channel_id
            for channel_id, record in channel_map.items()
            if record.get("responding")
        }
        if not active:
            return []

        coords = {
            channel_id: (
                int(round(self.profile.channels[channel_id].x)),
                int(round(self.profile.channels[channel_id].y)),
            )
            for channel_id in self.profile.channel_order
        }
        regions: list[dict[str, Any]] = []
        while active:
            seed = min(active)
            queue = [seed]
            active.remove(seed)
            members: list[str] = []
            while queue:
                current = queue.pop()
                members.append(current)
                cx, cy = coords[current]
                neighbors = [
                    candidate
                    for candidate in list(active)
                    if max(
                        abs(coords[candidate][0] - cx),
                        abs(coords[candidate][1] - cy),
                    )
                    <= 1
                ]
                for candidate in neighbors:
                    active.remove(candidate)
                    queue.append(candidate)

            weights = np.asarray(
                [
                    max(float(channel_map[channel_id].get("response_value") or 0.0), 0.0)
                    for channel_id in members
                ],
                dtype=float,
            )
            weight_sum = float(np.sum(weights))
            if weight_sum > 1.0e-12:
                center_x = float(
                    sum(coords[channel_id][0] * weight for channel_id, weight in zip(members, weights))
                    / weight_sum
                )
                center_y = float(
                    sum(coords[channel_id][1] * weight for channel_id, weight in zip(members, weights))
                    / weight_sum
                )
            else:
                center_x = float(np.mean([coords[channel_id][0] for channel_id in members]))
                center_y = float(np.mean([coords[channel_id][1] for channel_id in members]))
            regions.append(
                {
                    "region_id": f"R{len(regions) + 1}",
                    "channel_ids": sorted(members),
                    "center_x": center_x,
                    "center_y": center_y,
                    "peak_response": max(
                        float(channel_map[channel_id].get("response_value") or 0.0)
                        for channel_id in members
                    ),
                    "semantics": "coupled_optical_response_region",
                }
            )
        return regions
