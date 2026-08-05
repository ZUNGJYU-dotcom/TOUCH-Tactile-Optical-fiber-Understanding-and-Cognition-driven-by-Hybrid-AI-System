import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.hybrid_spectrum.measurement_consistency import (
    MeasurementAnalysisConfig,
    analyze_measurement_session,
    estimate_frame_lag,
    load_measurement_trace,
    write_measurement_artifacts,
)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_cycles(cycle_count: int = 3) -> np.ndarray:
    cycle = np.concatenate(
        (
            np.zeros(5),
            np.linspace(0.0, 1.0, 6)[1:],
            np.ones(5),
            np.linspace(1.0, 0.0, 6)[1:],
            np.zeros(5),
        )
    )
    return np.tile(cycle, cycle_count)


def test_estimate_frame_lag_positive_when_optical_trails_reference():
    reference = _synthetic_cycles(4)
    estimate = np.concatenate((np.zeros(2), reference[:-2]))
    result = estimate_frame_lag(
        reference,
        estimate,
        median_frame_interval_sec=0.1,
        maximum_lag_sec=1.0,
    )
    assert result["lag_frames"] == 2
    assert result["lag_ms"] == 200.0
    assert result["correlation"] > 0.99


def test_analyze_measurement_session_reports_cycles_lag_and_force_metrics(
    tmp_path: Path,
):
    reference = _synthetic_cycles(3)
    estimate = np.concatenate((np.zeros(2), reference[:-2])) * 0.95 + 0.02
    rows = []
    for index, (truth, prediction) in enumerate(zip(reference, estimate)):
        rows.append(
            {
                "capture_index": index,
                "timeline_timestamp_epoch_sec": 1000.0 + 0.1 * index,
                "elapsed_time_sec": 0.1 * index,
                "force_fz_n": truth,
                "optical_estimated_fz_n": prediction,
                "optical_raw_estimated_fz_n": prediction,
                "optical_force_estimate_gated": False,
                "predicted_contact_label": "contact" if prediction > 0.1 else "no_contact",
                "predicted_position_label": "P22" if prediction > 0.1 else "",
                "sync_offset_ms": 12.0,
                "calibration_sync_ok": True,
                "model_inference_latency_ms": 3.0,
                "model_source": "test_model",
            }
        )
    _write_summary(tmp_path / "frame_summary.csv", rows)
    result = analyze_measurement_session(tmp_path)
    summary = result["summary"]
    assert summary["data"]["comparison_status"] == "completed"
    assert summary["lag"]["lag_frames"] == 2
    assert summary["lag_compensated_comparison"]["mae_n"] < 0.04
    assert summary["repeatability"]["cycle_count"] == 3
    assert summary["repeatability"]["status"] == "descriptive_repeatability_available"
    assert summary["cadence"]["acquisition_rate_hz"] == pytest.approx(10.0)
    assert summary["direct_comparison"]["pearson_r"] > 0.8
    assert summary["lag_compensated_comparison"]["linear_slope_pred_vs_reference"] == pytest.approx(
        0.95, abs=0.03
    )
    assert summary["lag_compensated_comparison"]["amplitude_ratio_p95_p05"] == pytest.approx(
        0.95, abs=0.03
    )
    assert all(row["force_phase"] for row in result["trace_rows"])


def test_historical_jsonl_backfills_optical_force_fields(tmp_path: Path):
    _write_summary(
        tmp_path / "frame_summary.csv",
        [
            {
                "capture_index": 0,
                "timeline_timestamp_epoch_sec": 10.0,
                "elapsed_time_sec": 0.0,
                "force_fz_n": 0.6,
                "model_source": "historical_model",
            }
        ],
    )
    payload = {
        "capture_index": 0,
        "timeline_timestamp_epoch_sec": 10.0,
        "elapsed_time_sec": 0.0,
        "force_fz_n": 0.6,
        "tactile_response": {
            "estimated_force_fz_n": 0.55,
            "inference_latency_ms": 4.2,
            "model_source": "historical_model",
            "force_fz": {"raw_estimated_n": 0.58, "gated": False},
            "contact": {"label": "contact"},
            "position": {"label": "P21"},
        },
    }
    (tmp_path / "synchronized_frames.jsonl").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    rows = load_measurement_trace(tmp_path)
    assert rows[0]["optical_estimated_fz_n"] == 0.55
    assert rows[0]["optical_raw_estimated_fz_n"] == 0.58
    assert rows[0]["predicted_position_label"] == "P21"
    assert rows[0]["model_inference_latency_ms"] == 4.2


def test_missing_force_reference_is_explicitly_skipped(tmp_path: Path):
    rows = [
        {
            "capture_index": index,
            "timeline_timestamp_epoch_sec": 100.0 + index * 0.2,
            "elapsed_time_sec": index * 0.2,
            "optical_estimated_fz_n": float(index) / 10.0,
            "model_source": "optical_only",
        }
        for index in range(12)
    ]
    _write_summary(tmp_path / "frame_summary.csv", rows)
    result = analyze_measurement_session(tmp_path)
    assert (
        result["summary"]["data"]["comparison_status"]
        == "skipped_force_reference_not_recorded"
    )
    assert result["summary"]["direct_comparison"]["mae_n"] is None
    assert result["summary"]["semantics"]["force_sensor_is_runtime_model_input"] is False


def test_selected_overlay_never_mixes_missing_frames_with_recorded_runtime(
    tmp_path: Path,
):
    rows = []
    for index in range(12):
        rows.append(
            {
                "capture_index": index,
                "timeline_timestamp_epoch_sec": 100.0 + index * 0.1,
                "elapsed_time_sec": index * 0.1,
                "force_fz_n": float(index) / 10.0,
                "optical_estimated_fz_n": 9.0,
                "optical_raw_estimated_fz_n": 9.5,
                "model_source": "old_capture_model",
            }
        )
    _write_summary(tmp_path / "frame_summary.csv", rows)
    overlay = {
        index: {
            "estimated_fz_n": float(index) / 10.0,
            "raw_estimated_fz_n": float(index) / 10.0,
            "model_source": "grouped_oof",
        }
        for index in range(6)
    }
    result = analyze_measurement_session(
        tmp_path,
        MeasurementAnalysisConfig(minimum_paired_samples=2),
        estimate_overlay=overlay,
        estimate_source_info={
            "source": "grouped_oof",
            "label": "Grouped OOF",
            "evaluation_validity": "formal_grouped_oof_by_session_id",
            "provenance": {"fold_ids": ["fold_1"]},
        },
    )

    trace = result["trace_rows"]
    assert trace[5]["analysis_estimated_fz_n"] == pytest.approx(0.5)
    assert trace[6]["analysis_estimated_fz_n"] is None
    assert trace[6]["recorded_runtime_optical_estimated_fz_n"] == 9.0
    assert result["summary"]["data"]["paired_count"] == 6
    assert result["summary"]["data"]["analysis_estimate_source"] == "grouped_oof"
    assert (
        result["summary"]["comparisons"]["recorded_runtime"]["mae_n"]
        != result["summary"]["direct_comparison"]["mae_n"]
    )


def test_write_measurement_artifacts_creates_shareable_outputs(tmp_path: Path):
    reference = _synthetic_cycles(1)
    rows = [
        {
            "capture_index": index,
            "timeline_timestamp_epoch_sec": 100.0 + index * 0.1,
            "elapsed_time_sec": index * 0.1,
            "force_fz_n": value,
            "optical_estimated_fz_n": value * 0.9,
            "sync_offset_ms": 5.0,
            "calibration_sync_ok": True,
        }
        for index, value in enumerate(reference)
    ]
    session = tmp_path / "session"
    _write_summary(session / "frame_summary.csv", rows)
    result = analyze_measurement_session(
        session,
        MeasurementAnalysisConfig(minimum_repeatability_cycles=1),
    )
    output = tmp_path / "output"
    write_measurement_artifacts(result, output)
    assert (output / "measurement_trace.csv").is_file()
    assert (output / "measurement_cycles.csv").is_file()
    assert (output / "measurement_summary.json").is_file()
    assert (output / "measurement_report.md").is_file()
    assert (output / "measurement_consistency.png").stat().st_size > 10_000


def test_measurement_summary_is_portable_with_non_ascii_session_path(
    tmp_path: Path,
):
    session = tmp_path / "\u666e\u901aFBG" / "\u529b\u5b66\u6807\u5b9a"
    _write_summary(
        session / "frame_summary.csv",
        [
            {
                "capture_index": 0,
                "timeline_timestamp_epoch_sec": 100.0,
                "elapsed_time_sec": 0.0,
                "force_fz_n": 0.0,
                "optical_estimated_fz_n": 0.0,
            }
        ],
    )
    result = analyze_measurement_session(session)
    output = tmp_path / "output"
    write_measurement_artifacts(result, output)

    raw = (output / "measurement_summary.json").read_bytes()
    assert all(byte < 128 for byte in raw)
    summary = json.loads(raw.decode("ascii"))
    assert summary["session_dir"] == str(session.resolve())
