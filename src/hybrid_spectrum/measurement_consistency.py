"""Offline optical-force measurement consistency analysis.

The synchronized capture timeline is authoritative. PX6D ``force_fz_n`` is the
mechanical reference; ``optical_estimated_fz_n`` is the optical-only estimate.
The force reference is never treated as a runtime model input.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRACE_FIELDS = (
    "capture_index",
    "timeline_timestamp_epoch_sec",
    "elapsed_time_sec",
    "reference_fz_n",
    "optical_estimated_fz_n",
    "optical_raw_estimated_fz_n",
    "analysis_estimated_fz_n",
    "analysis_raw_estimated_fz_n",
    "analysis_estimate_source",
    "analysis_model_source",
    "recorded_runtime_optical_estimated_fz_n",
    "recorded_runtime_optical_raw_estimated_fz_n",
    "recorded_runtime_model_inference_latency_ms",
    "recorded_runtime_model_source",
    "optical_force_estimate_gated",
    "contact_label",
    "predicted_position_label",
    "sync_offset_ms",
    "calibration_sync_ok",
    "model_inference_latency_ms",
    "model_source",
    "cycle_id",
    "force_phase",
)


@dataclass(frozen=True)
class MeasurementAnalysisConfig:
    contact_on_n: float = 0.15
    contact_off_n: float = 0.08
    minimum_active_duration_sec: float = 0.40
    baseline_window_sec: float = 0.75
    recovery_window_sec: float = 0.75
    plateau_fraction_of_peak: float = 0.80
    maximum_lag_sec: float = 2.0
    minimum_paired_samples: int = 8
    minimum_repeatability_cycles: int = 3
    normalized_cycle_points: int = 101
    replay_baseline_strategy: str = "session_initial_stable_median"
    replay_baseline_frame_count: int = 10
    replay_baseline_minimum_stable_frames: int = 5
    replay_baseline_stability_mad_multiplier: float = 3.5

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "MeasurementAnalysisConfig":
        values = dict(payload or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


def load_measurement_config(path: Path | None = None) -> MeasurementAnalysisConfig:
    if path is None:
        return MeasurementAnalysisConfig()
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return MeasurementAnalysisConfig.from_mapping(
        payload.get("measurement_analysis", payload)
    )


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _model_force_fields(model: dict[str, Any]) -> dict[str, Any]:
    force = dict(model.get("force_fz") or {})
    uncertainty = dict(model.get("uncertainty") or {})
    estimated = _safe_float(model.get("estimated_force_fz_n"))
    if estimated is None:
        estimated = _safe_float(force.get("estimated_n"))
    raw_estimated = _safe_float(force.get("raw_estimated_n"))
    return {
        "optical_estimated_fz_n": estimated,
        "optical_raw_estimated_fz_n": raw_estimated,
        "optical_force_estimate_gated": _safe_bool(force.get("gated")),
        "model_inference_latency_ms": _safe_float(
            model.get("inference_latency_ms")
        ),
        "model_source": model.get("model_source") or model.get("recognition_source"),
        "contact_label": (model.get("contact") or {}).get("label"),
        "predicted_position_label": (model.get("position") or {}).get("label"),
        "optical_force_review_needed": _safe_bool(
            uncertainty.get("review_needed")
        ),
    }


def _jsonl_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                capture_index = int(payload["capture_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            force = dict(payload.get("px6d_reference") or {})
            model = dict(payload.get("tactile_response") or {})
            row = {
                "capture_index": capture_index,
                "timeline_timestamp_epoch_sec": _safe_float(
                    payload.get("timeline_timestamp_epoch_sec")
                ),
                "elapsed_time_sec": _safe_float(payload.get("elapsed_time_sec")),
                "reference_fz_n": _safe_float(
                    payload.get("force_fz_n", force.get("force_fz_n"))
                ),
                "sync_offset_ms": _safe_float(force.get("sync_offset_ms")),
                "calibration_sync_ok": _safe_bool(
                    force.get("calibration_sync_ok")
                ),
            }
            row.update(_model_force_fields(model))
            rows[capture_index] = row
    return rows


def load_measurement_trace(session_dir: Path) -> list[dict[str, Any]]:
    """Load one synchronized capture, including historical JSONL fallback."""

    session_dir = Path(session_dir)
    json_rows = _jsonl_rows(session_dir / "synchronized_frames.jsonl")
    summary_path = session_dir / "frame_summary.csv"
    rows: dict[int, dict[str, Any]] = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle):
                try:
                    capture_index = int(source["capture_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                row = {
                    "capture_index": capture_index,
                    "timeline_timestamp_epoch_sec": _safe_float(
                        source.get("timeline_timestamp_epoch_sec")
                    ),
                    "elapsed_time_sec": _safe_float(source.get("elapsed_time_sec")),
                    "reference_fz_n": _safe_float(
                        source.get("force_fz_n")
                        or source.get("conditioned_reference_fz_n")
                    ),
                    "optical_estimated_fz_n": _safe_float(
                        source.get("optical_estimated_fz_n")
                    ),
                    "optical_raw_estimated_fz_n": _safe_float(
                        source.get("optical_raw_estimated_fz_n")
                    ),
                    "optical_force_estimate_gated": _safe_bool(
                        source.get("optical_force_estimate_gated")
                    ),
                    "contact_label": source.get("predicted_contact_label"),
                    "predicted_position_label": source.get(
                        "predicted_position_label"
                    ),
                    "sync_offset_ms": _safe_float(source.get("sync_offset_ms")),
                    "calibration_sync_ok": _safe_bool(
                        source.get("calibration_sync_ok")
                    ),
                    "model_inference_latency_ms": _safe_float(
                        source.get("model_inference_latency_ms")
                    ),
                    "model_source": source.get("model_source"),
                }
                fallback = json_rows.get(capture_index, {})
                for key, value in fallback.items():
                    if row.get(key) is None or row.get(key) == "":
                        row[key] = value
                rows[capture_index] = row
    else:
        rows = json_rows
    return [rows[index] for index in sorted(rows)]


def _numeric_array(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get(field) is None else float(row[field]) for row in rows],
        dtype=float,
    )


def _apply_estimate_source(
    rows: list[dict[str, Any]],
    estimate_overlay: dict[int, dict[str, Any]] | None,
    estimate_source_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Select exactly one estimate source while retaining recorded evidence.

    Missing overlay frames deliberately stay missing. Falling back to the
    recorded runtime value on individual frames would create a scientifically
    invalid mixed curve.
    """

    source_info = dict(estimate_source_info or {})
    source = str(source_info.get("source") or "recorded_runtime")
    source_info.setdefault(
        "label", "Historical recorded runtime (capture-time model)"
    )
    source_info.setdefault(
        "evaluation_validity",
        "historical_capture_time_output_not_current_model",
    )
    source_info.setdefault("provenance", {})
    overlay = estimate_overlay if estimate_overlay is not None else None

    for row in rows:
        row["recorded_runtime_optical_estimated_fz_n"] = row.get(
            "optical_estimated_fz_n"
        )
        row["recorded_runtime_optical_raw_estimated_fz_n"] = row.get(
            "optical_raw_estimated_fz_n"
        )
        row["recorded_runtime_model_inference_latency_ms"] = row.get(
            "model_inference_latency_ms"
        )
        row["recorded_runtime_model_source"] = row.get("model_source")

        capture_index = int(row["capture_index"])
        selected = overlay.get(capture_index) if overlay is not None else None
        if overlay is None:
            selected = {
                "estimated_fz_n": row.get(
                    "recorded_runtime_optical_estimated_fz_n"
                ),
                "raw_estimated_fz_n": row.get(
                    "recorded_runtime_optical_raw_estimated_fz_n"
                ),
                "contact_label": row.get("contact_label"),
                "position_label": row.get("predicted_position_label"),
                "inference_latency_ms": row.get(
                    "recorded_runtime_model_inference_latency_ms"
                ),
                "model_source": row.get("recorded_runtime_model_source"),
            }

        selected = dict(selected or {})
        row["analysis_estimated_fz_n"] = _safe_float(
            selected.get("estimated_fz_n")
        )
        row["analysis_raw_estimated_fz_n"] = _safe_float(
            selected.get("raw_estimated_fz_n")
        )
        row["analysis_estimate_source"] = source
        row["analysis_model_source"] = selected.get("model_source")

        # Preserve the original public fields as aliases for existing reports
        # and scripts. They now always represent the selected analysis source.
        row["optical_estimated_fz_n"] = row["analysis_estimated_fz_n"]
        row["optical_raw_estimated_fz_n"] = row[
            "analysis_raw_estimated_fz_n"
        ]
        row["model_inference_latency_ms"] = _safe_float(
            selected.get("inference_latency_ms")
        )
        row["model_source"] = selected.get("model_source")
        if selected.get("contact_label") not in (None, ""):
            row["contact_label"] = selected.get("contact_label")
        if selected.get("position_label") not in (None, ""):
            row["predicted_position_label"] = selected.get("position_label")

    return source_info


def _finite_stat(values: np.ndarray, operation: str) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    functions = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "p90": lambda x: np.percentile(x, 90),
        "p95": lambda x: np.percentile(x, 95),
    }
    return float(functions[operation](finite))


def _regression_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[valid]
    estimate = estimate[valid]
    if reference.size == 0:
        return {
            "paired_sample_count": 0,
            "mae_n": None,
            "rmse_n": None,
            "bias_n": None,
            "r2": None,
            "pearson_r": None,
            "trend_pearson_r": None,
            "linear_slope_pred_vs_reference": None,
            "linear_intercept_n": None,
            "reference_amplitude_p95_p05_n": None,
            "estimate_amplitude_p95_p05_n": None,
            "amplitude_ratio_p95_p05": None,
        }
    error = estimate - reference
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    reference_std = float(np.std(reference))
    estimate_std = float(np.std(estimate))
    reference_amplitude = float(np.percentile(reference, 95) - np.percentile(reference, 5))
    estimate_amplitude = float(np.percentile(estimate, 95) - np.percentile(estimate, 5))
    if reference.size >= 2 and reference_std > 1e-12:
        slope, intercept = np.polyfit(reference, estimate, 1)
    else:
        slope, intercept = math.nan, math.nan
    reference_delta = np.diff(reference)
    estimate_delta = np.diff(estimate)
    delta_valid = np.isfinite(reference_delta) & np.isfinite(estimate_delta)
    reference_delta = reference_delta[delta_valid]
    estimate_delta = estimate_delta[delta_valid]
    trend_correlation = None
    if (
        reference_delta.size >= 3
        and float(np.std(reference_delta)) > 1e-12
        and float(np.std(estimate_delta)) > 1e-12
    ):
        trend_correlation = float(np.corrcoef(reference_delta, estimate_delta)[0, 1])
    return {
        "paired_sample_count": int(reference.size),
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(error**2))),
        "bias_n": float(np.mean(error)),
        "r2": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 1e-12
            else None
        ),
        "pearson_r": (
            float(np.corrcoef(reference, estimate)[0, 1])
            if reference.size >= 3
            and reference_std > 1e-12
            and estimate_std > 1e-12
            else None
        ),
        "trend_pearson_r": trend_correlation,
        "linear_slope_pred_vs_reference": (
            float(slope) if math.isfinite(float(slope)) else None
        ),
        "linear_intercept_n": (
            float(intercept) if math.isfinite(float(intercept)) else None
        ),
        "reference_amplitude_p95_p05_n": reference_amplitude,
        "estimate_amplitude_p95_p05_n": estimate_amplitude,
        "amplitude_ratio_p95_p05": (
            estimate_amplitude / reference_amplitude
            if reference_amplitude > 1e-12
            else None
        ),
    }


def estimate_frame_lag(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    median_frame_interval_sec: float,
    maximum_lag_sec: float,
) -> dict[str, Any]:
    """Estimate positive lag when the optical estimate trails PX6D."""

    if not math.isfinite(median_frame_interval_sec) or median_frame_interval_sec <= 0:
        return {"lag_frames": None, "lag_ms": None, "correlation": None}
    maximum_frames = max(0, int(round(maximum_lag_sec / median_frame_interval_sec)))
    maximum_frames = min(maximum_frames, max(0, len(reference) // 3))
    best: tuple[float, int] | None = None
    for lag in range(-maximum_frames, maximum_frames + 1):
        if lag >= 0:
            x = reference[: len(reference) - lag or None]
            y = estimate[lag:]
        else:
            x = reference[-lag:]
            y = estimate[: len(estimate) + lag]
        valid = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(valid)) < 4:
            continue
        x = x[valid]
        y = y[valid]
        if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            continue
        correlation = float(np.corrcoef(x, y)[0, 1])
        if best is None or correlation > best[0]:
            best = (correlation, lag)
    if best is None:
        return {"lag_frames": None, "lag_ms": None, "correlation": None}
    return {
        "lag_frames": int(best[1]),
        "lag_ms": float(best[1] * median_frame_interval_sec * 1000.0),
        "correlation": float(best[0]),
    }


def _lag_aligned(
    reference: np.ndarray, estimate: np.ndarray, lag_frames: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if lag_frames is None or lag_frames == 0:
        return reference, estimate
    if lag_frames > 0:
        return reference[:-lag_frames], estimate[lag_frames:]
    return reference[-lag_frames:], estimate[:lag_frames]


def _interpolate_finite(values: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(values)
    if int(np.sum(valid)) < 2:
        return None
    indices = np.arange(len(values), dtype=float)
    return np.interp(indices, indices[valid], values[valid])


def _coefficient_of_variation(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2 or abs(float(np.mean(array))) <= 1e-12:
        return None
    return float(np.std(array, ddof=1) / abs(np.mean(array)))


def _segment_cycles(
    elapsed: np.ndarray,
    signal: np.ndarray,
    config: MeasurementAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    filled = _interpolate_finite(signal)
    if filled is None or len(filled) < 3:
        return [], []
    segments: list[tuple[int, int]] = []
    active = False
    start = 0
    for index, value in enumerate(filled):
        if not active and value >= config.contact_on_n:
            start = index
            active = True
        elif active and value <= config.contact_off_n:
            if elapsed[index] - elapsed[start] >= config.minimum_active_duration_sec:
                segments.append((start, index))
            active = False
    if active and elapsed[-1] - elapsed[start] >= config.minimum_active_duration_sec:
        segments.append((start, len(filled) - 1))

    rows: list[dict[str, Any]] = []
    for cycle_id, (start, stop) in enumerate(segments, start=1):
        baseline_start = int(
            np.searchsorted(elapsed, elapsed[start] - config.baseline_window_sec)
        )
        baseline_slice = filled[baseline_start:start]
        baseline = float(np.median(baseline_slice)) if baseline_slice.size else 0.0
        active_signal = filled[start : stop + 1]
        peak_local = int(np.argmax(active_signal))
        peak_index = start + peak_local
        peak = float(filled[peak_index])
        amplitude = max(peak - baseline, 1e-12)
        plateau_threshold = baseline + config.plateau_fraction_of_peak * amplitude
        plateau_indices = np.flatnonzero(active_signal >= plateau_threshold) + start
        plateau = (
            float(np.median(filled[plateau_indices]))
            if plateau_indices.size
            else peak
        )
        recovery_index = min(
            len(filled) - 1,
            int(np.searchsorted(elapsed, elapsed[stop] + config.recovery_window_sec)),
        )
        residual = float(filled[recovery_index] - baseline)
        recovery_ratio = float(np.clip(1.0 - abs(residual) / amplitude, 0.0, 1.0))
        rows.append(
            {
                "cycle_id": cycle_id,
                "start_index": start,
                "peak_index": peak_index,
                "stop_index": stop,
                "recovery_index": recovery_index,
                "start_time_sec": float(elapsed[start]),
                "peak_time_sec": float(elapsed[peak_index]),
                "stop_time_sec": float(elapsed[stop]),
                "active_duration_sec": float(elapsed[stop] - elapsed[start]),
                "baseline_fz_n": baseline,
                "peak_fz_n": peak,
                "plateau_fz_n": plateau,
                "residual_fz_n": residual,
                "release_recovery_ratio": recovery_ratio,
                "rise_time_sec": float(elapsed[peak_index] - elapsed[start]),
                "release_time_sec": float(elapsed[stop] - elapsed[peak_index]),
                "plateau_sample_count": int(plateau_indices.size),
            }
        )
    return rows, segments


def _annotate_phases(
    rows: list[dict[str, Any]],
    cycles: list[dict[str, Any]],
    elapsed: np.ndarray,
) -> None:
    for row in rows:
        row["cycle_id"] = None
        row["force_phase"] = "baseline"
    for cycle in cycles:
        cycle_id = int(cycle["cycle_id"])
        start = int(cycle["start_index"])
        peak = int(cycle["peak_index"])
        stop = int(cycle["stop_index"])
        recovery = int(cycle["recovery_index"])
        for index in range(start, peak + 1):
            rows[index]["cycle_id"] = cycle_id
            rows[index]["force_phase"] = "loading"
        for index in range(peak + 1, stop + 1):
            rows[index]["cycle_id"] = cycle_id
            rows[index]["force_phase"] = "unloading"
        for index in range(stop + 1, recovery + 1):
            rows[index]["cycle_id"] = cycle_id
            rows[index]["force_phase"] = "recovery"


def _cycle_repeatability(
    signal: np.ndarray,
    cycle_rows: list[dict[str, Any]],
    points: int,
) -> dict[str, Any]:
    waveforms: list[np.ndarray] = []
    for cycle in cycle_rows:
        start = int(cycle["start_index"])
        stop = int(cycle["stop_index"])
        segment = signal[start : stop + 1]
        segment = _interpolate_finite(segment)
        if segment is None or len(segment) < 3:
            continue
        baseline = float(cycle["baseline_fz_n"])
        amplitude = max(float(cycle["peak_fz_n"]) - baseline, 1e-12)
        normalized = (segment - baseline) / amplitude
        waveforms.append(
            np.interp(
                np.linspace(0.0, 1.0, points),
                np.linspace(0.0, 1.0, len(normalized)),
                normalized,
            )
        )
    correlations: list[float] = []
    for first in range(len(waveforms)):
        for second in range(first + 1, len(waveforms)):
            if np.std(waveforms[first]) > 1e-12 and np.std(waveforms[second]) > 1e-12:
                correlations.append(
                    float(np.corrcoef(waveforms[first], waveforms[second])[0, 1])
                )
    return {
        "cycle_count": len(cycle_rows),
        "usable_waveform_count": len(waveforms),
        "peak_force_cv": _coefficient_of_variation(
            cycle["peak_fz_n"] for cycle in cycle_rows
        ),
        "plateau_force_cv": _coefficient_of_variation(
            cycle["plateau_fz_n"] for cycle in cycle_rows
        ),
        "mean_pairwise_waveform_correlation": (
            float(np.mean(correlations)) if correlations else None
        ),
        "normalized_waveforms": [waveform.tolist() for waveform in waveforms],
    }


def analyze_measurement_session(
    session_dir: Path,
    config: MeasurementAnalysisConfig | None = None,
    *,
    estimate_overlay: dict[int, dict[str, Any]] | None = None,
    estimate_source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or MeasurementAnalysisConfig()
    rows = load_measurement_trace(session_dir)
    if not rows:
        raise ValueError(f"No synchronized measurement rows found in {session_dir}")
    source_info = _apply_estimate_source(
        rows,
        estimate_overlay,
        estimate_source_info,
    )

    elapsed = _numeric_array(rows, "elapsed_time_sec")
    reference = _numeric_array(rows, "reference_fz_n")
    estimate = _numeric_array(rows, "analysis_estimated_fz_n")
    raw_estimate = _numeric_array(rows, "analysis_raw_estimated_fz_n")
    recorded_estimate = _numeric_array(
        rows, "recorded_runtime_optical_estimated_fz_n"
    )
    intervals = np.diff(elapsed)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    median_interval = float(np.median(intervals)) if intervals.size else math.nan
    cadence = {
        "frame_interval_median_sec": (
            median_interval if math.isfinite(median_interval) else None
        ),
        "frame_interval_p90_sec": _finite_stat(intervals, "p90"),
        "acquisition_rate_hz": (
            float(1.0 / median_interval)
            if math.isfinite(median_interval) and median_interval > 0
            else None
        ),
        "model_inference_latency_median_ms": _finite_stat(
            _numeric_array(rows, "model_inference_latency_ms"), "median"
        ),
        "recorded_runtime_inference_latency_median_ms": _finite_stat(
            _numeric_array(
                rows, "recorded_runtime_model_inference_latency_ms"
            ),
            "median",
        ),
    }

    valid_reference = int(np.sum(np.isfinite(reference)))
    valid_estimate = int(np.sum(np.isfinite(estimate)))
    paired = int(np.sum(np.isfinite(reference) & np.isfinite(estimate)))
    if valid_reference == 0:
        comparison_status = "skipped_force_reference_not_recorded"
    elif paired < config.minimum_paired_samples:
        comparison_status = "insufficient_paired_samples"
    else:
        comparison_status = "completed"

    direct_metrics = _regression_metrics(reference, estimate)
    recorded_runtime_metrics = _regression_metrics(reference, recorded_estimate)
    lag = estimate_frame_lag(
        reference,
        estimate,
        median_frame_interval_sec=median_interval,
        maximum_lag_sec=config.maximum_lag_sec,
    )
    aligned_reference, aligned_estimate = _lag_aligned(
        reference, estimate, lag.get("lag_frames")
    )
    lag_metrics = _regression_metrics(aligned_reference, aligned_estimate)

    cycle_signal = reference if valid_reference >= config.minimum_paired_samples else estimate
    cycle_basis = "PX6D_reference_Fz" if cycle_signal is reference else "optical_estimated_Fz"
    cycle_rows, _ = _segment_cycles(elapsed, cycle_signal, config)
    _annotate_phases(rows, cycle_rows, elapsed)
    for cycle in cycle_rows:
        peak_index = int(cycle["peak_index"])
        cycle["optical_estimated_peak_fz_n"] = (
            float(estimate[peak_index]) if np.isfinite(estimate[peak_index]) else None
        )
        cycle["optical_peak_error_n"] = (
            float(estimate[peak_index] - reference[peak_index])
            if np.isfinite(estimate[peak_index]) and np.isfinite(reference[peak_index])
            else None
        )
    repeatability = _cycle_repeatability(
        cycle_signal, cycle_rows, config.normalized_cycle_points
    )
    repeatability["status"] = (
        "descriptive_repeatability_available"
        if len(cycle_rows) >= config.minimum_repeatability_cycles
        else "insufficient_repeated_cycles"
    )
    repeatability["cycle_basis"] = cycle_basis

    sync_offsets = np.abs(_numeric_array(rows, "sync_offset_ms"))
    sync_ok = [row.get("calibration_sync_ok") for row in rows]
    summary = {
        "schema_version": "touch_optical_force_measurement_v2",
        "session_dir": str(Path(session_dir).resolve()),
        "semantics": {
            "reference": "PX6D conditioned compression Fz in N",
            "estimate": str(
                source_info.get("label")
                or "optical-only model estimated Fz in N"
            ),
            "estimate_source": source_info.get("source"),
            "evaluation_validity": source_info.get("evaluation_validity"),
            "force_sensor_is_runtime_model_input": False,
            "claim_boundary": "comparison and calibration evidence, not force certification",
        },
        "config": asdict(config),
        "data": {
            "row_count": len(rows),
            "valid_reference_count": valid_reference,
            "valid_optical_estimate_count": valid_estimate,
            "paired_count": paired,
            "comparison_status": comparison_status,
            "analysis_estimate_source": source_info.get("source"),
            "analysis_estimate_label": source_info.get("label"),
            "evaluation_validity": source_info.get("evaluation_validity"),
            "estimate_provenance": source_info.get("provenance", {}),
            "model_sources": sorted(
                {str(row["model_source"]) for row in rows if row.get("model_source")}
            ),
            "recorded_runtime_model_sources": sorted(
                {
                    str(row["recorded_runtime_model_source"])
                    for row in rows
                    if row.get("recorded_runtime_model_source")
                }
            ),
        },
        "cadence": cadence,
        "synchronization": {
            "absolute_sync_offset_median_ms": _finite_stat(sync_offsets, "median"),
            "absolute_sync_offset_p95_ms": _finite_stat(sync_offsets, "p95"),
            "absolute_sync_offset_max_ms": _finite_stat(sync_offsets, "max"),
            "calibration_sync_ok_ratio": (
                float(sum(value is True for value in sync_ok) / len(sync_ok))
                if sync_ok
                else None
            ),
        },
        "direct_comparison": direct_metrics,
        "comparisons": {
            "selected_estimate": {
                "source": source_info.get("source"),
                "label": source_info.get("label"),
                "evaluation_validity": source_info.get(
                    "evaluation_validity"
                ),
                **direct_metrics,
            },
            "recorded_runtime": {
                "source": "recorded_runtime",
                "label": "Historical recorded runtime (capture-time model)",
                "evaluation_validity": (
                    "historical_capture_time_output_not_current_model"
                ),
                **recorded_runtime_metrics,
            },
        },
        "lag": lag,
        "lag_compensated_comparison": lag_metrics,
        "repeatability": {
            key: value
            for key, value in repeatability.items()
            if key != "normalized_waveforms"
        },
    }
    return {
        "summary": summary,
        "trace_rows": rows,
        "cycle_rows": cycle_rows,
        "normalized_cycle_waveforms": repeatability["normalized_waveforms"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fields = list(fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _report_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    data = summary["data"]
    cadence = summary["cadence"]
    direct = summary["direct_comparison"]
    lag = summary["lag"]
    compensated = summary["lag_compensated_comparison"]
    repeatability = summary["repeatability"]
    return f"""# TOUCH Optical-Force Measurement Report

## Scope

This report compares the optical-only Fz estimate with the timestamp-aligned
PX6D compression Fz reference. PX6D is calibration supervision and is not a
runtime model input. The result is measurement evidence, not force
certification.

## Data readiness

- Comparison status: `{data['comparison_status']}`
- Estimate source: `{data['analysis_estimate_source']}`
- Evaluation validity: `{data['evaluation_validity']}`
- Timeline rows: {data['row_count']}
- Paired optical/PX6D samples: {data['paired_count']}
- Median acquisition interval: {_format_metric(cadence['frame_interval_median_sec'])} s
- Effective acquisition rate: {_format_metric(cadence['acquisition_rate_hz'])} Hz
- Median model inference latency: {_format_metric(cadence['model_inference_latency_median_ms'])} ms

Acquisition cadence, model inference latency, and display refresh rate are
reported separately. A high UI frame rate does not imply a higher BaySpec
physical sampling rate.

## Paired force comparison

- Direct MAE: {_format_metric(direct['mae_n'])} N
- Direct RMSE: {_format_metric(direct['rmse_n'])} N
- Direct bias: {_format_metric(direct['bias_n'])} N
- Direct R2: {_format_metric(direct['r2'])}
- Direct Pearson r: {_format_metric(direct['pearson_r'])}
- Estimated optical lag: {_format_metric(lag['lag_ms'])} ms ({_format_metric(lag['lag_frames'], 0)} frames)
- Lag-compensated MAE: {_format_metric(compensated['mae_n'])} N
- Lag-compensated RMSE: {_format_metric(compensated['rmse_n'])} N

## Repeated-action evidence

- Status: `{repeatability['status']}`
- Segmented cycles: {repeatability['cycle_count']}
- Peak-force CV: {_format_metric(repeatability['peak_force_cv'])}
- Plateau-force CV: {_format_metric(repeatability['plateau_force_cv'])}
- Mean normalized waveform correlation: {_format_metric(repeatability['mean_pairwise_waveform_correlation'])}
- Segmentation basis: `{repeatability['cycle_basis']}`

The repeated-curve view is inspired by the reviewed industrial demonstration:
it makes force tracking, consistency, lag, and release recovery visible. It
does not copy the demonstration's claimed sampling rate or imply equivalent
hardware performance.
"""


def _plot_measurement(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["trace_rows"]
    elapsed = _numeric_array(rows, "elapsed_time_sec")
    reference = _numeric_array(rows, "reference_fz_n")
    estimate = _numeric_array(rows, "analysis_estimated_fz_n")
    summary = result["summary"]
    waveforms = result["normalized_cycle_waveforms"]

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(elapsed, reference, color="#0072B2", linewidth=2.0, label="PX6D reference Fz")
    axes[0].plot(
        elapsed,
        estimate,
        color="#D55E00",
        linewidth=1.8,
        label=str(summary["data"]["analysis_estimate_label"]),
    )
    axes[0].set_ylabel("Fz (N)")
    axes[0].set_title("Timestamp-aligned optical-force measurement")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(alpha=0.18)

    error = estimate - reference
    axes[1].plot(elapsed, error, color="#6A51A3", linewidth=1.4)
    axes[1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[1].set_ylabel("Estimate error (N)")
    axes[1].set_title(
        "Direct error | lag = "
        + _format_metric(summary["lag"]["lag_ms"])
        + " ms"
    )
    axes[1].grid(alpha=0.18)

    if waveforms:
        normalized_time = np.linspace(0.0, 100.0, len(waveforms[0]))
        for index, waveform in enumerate(waveforms, start=1):
            axes[2].plot(
                normalized_time,
                waveform,
                linewidth=1.5,
                alpha=0.72,
                label=f"Cycle {index}",
            )
        axes[2].set_ylabel("Normalized Fz")
        axes[2].set_xlabel("Normalized action time (%)")
        axes[2].set_title("Repeated action consistency")
        if len(waveforms) <= 8:
            axes[2].legend(loc="best", frameon=False, ncol=2)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No repeated force cycles available",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            color="#607487",
        )
        axes[2].set_axis_off()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def write_measurement_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "measurement_trace.csv", result["trace_rows"], TRACE_FIELDS)
    cycle_fields = list(result["cycle_rows"][0]) if result["cycle_rows"] else [
        "cycle_id",
        "start_time_sec",
        "peak_time_sec",
        "stop_time_sec",
        "peak_fz_n",
        "plateau_fz_n",
        "release_recovery_ratio",
    ]
    _write_csv(output_dir / "measurement_cycles.csv", result["cycle_rows"], cycle_fields)
    (output_dir / "measurement_summary.json").write_text(
        # Keep this machine-readable artifact ASCII-only so Windows PowerShell
        # 5 can parse Chinese paths without relying on its legacy text codec.
        json.dumps(result["summary"], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "measurement_report.md").write_text(
        _report_markdown(result), encoding="utf-8"
    )
    _plot_measurement(result, output_dir / "measurement_consistency.png")
