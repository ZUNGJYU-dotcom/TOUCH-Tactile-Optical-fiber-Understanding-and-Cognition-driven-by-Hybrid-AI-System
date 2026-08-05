import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.hybrid_spectrum.measurement_estimate_sources import (
    _recorded_baseline,
    load_grouped_oof_evidence,
    recorded_runtime_evidence,
    resolve_measurement_estimate_evidence,
)


def _baseline_frames(
    *,
    frame_count: int = 6,
    recorded_level: float = 40.0,
    outlier_index: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    wavelength = np.linspace(1540.0, 1541.0, 8)
    recorded = np.full_like(wavelength, recorded_level)
    frames: dict[int, dict[str, np.ndarray]] = {}
    for index in range(frame_count):
        level = 100.0 + index * 0.1
        if index == outlier_index:
            level = 180.0
        frames[index] = {
            "wavelength": wavelength.copy(),
            "intensity": np.full_like(wavelength, level),
            "baseline": recorded.copy(),
        }
    return frames


def _trace_rows(count: int = 10) -> list[dict[str, object]]:
    return [
        {
            "capture_index": index,
            "elapsed_time_sec": index * 0.2,
            "reference_fz_n": index * 0.1,
            "optical_estimated_fz_n": 8.0,
            "optical_raw_estimated_fz_n": 8.5,
            "model_source": "recorded_old_model",
        }
        for index in range(count)
    ]


def _write_oof(path: Path, group_id: str, count: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "group_id",
        "sample_index",
        "elapsed_time_sec",
        "true_force_n",
        "gated_force_n",
        "raw_optical_force_n",
        "contact_gate_active",
        "fold_id",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "group_id": group_id,
                    "sample_index": index,
                    "elapsed_time_sec": index * 0.2,
                    "true_force_n": index * 0.1,
                    "gated_force_n": index * 0.1 + 0.02,
                    "raw_optical_force_n": index * 0.1 + 0.03,
                    "contact_gate_active": index > 0,
                    "fold_id": "fold_2",
                }
            )


def test_replay_baseline_defaults_to_session_local_spectra() -> None:
    wavelength, baseline, info = _recorded_baseline(
        _baseline_frames(),
        baseline_frame_count=6,
    )

    assert wavelength.shape == baseline.shape
    assert np.median(baseline) == pytest.approx(100.25)
    assert info["method"] == "session_initial_stable_median"
    assert info["recorded_baseline_available"] is True
    assert info["recorded_baseline_policy"] == "diagnostic_only"
    assert info["force_sensor_used_for_baseline_selection"] is False
    assert info["recorded_baseline_mean_absolute_difference_counts"] > 60.0
    assert info["recorded_baseline_normalized_rms_difference"] > 0.5
    assert info["recorded_baseline_consistency_status"] == "mismatch_warning"
    assert "recorded_baseline_mismatch" in info["quality_flags"]


def test_replay_baseline_rejects_initial_spectral_outlier() -> None:
    _, baseline, info = _recorded_baseline(
        _baseline_frames(frame_count=8, outlier_index=3),
        baseline_frame_count=8,
        minimum_stable_frames=5,
    )

    assert np.median(baseline) < 101.0
    assert 3 not in info["frame_indices"]
    assert info["candidate_frame_count"] == 8


def test_recorded_replay_baseline_requires_explicit_strategy() -> None:
    _, baseline, info = _recorded_baseline(
        _baseline_frames(recorded_level=42.0),
        baseline_frame_count=6,
        strategy="recorded_baseline_intensity_counts",
    )

    assert np.median(baseline) == pytest.approx(42.0)
    assert info["recorded_baseline_policy"] == "explicit_historical_diagnostic"


def test_replay_baseline_marks_similar_recorded_baseline_consistent() -> None:
    frames = _baseline_frames(recorded_level=100.25)

    _, _, info = _recorded_baseline(frames, baseline_frame_count=6)

    assert info["recorded_baseline_consistency_status"] == "consistent"
    assert info["recorded_baseline_normalized_rms_difference"] < 0.01
    assert info["quality_flags"] == []


def test_grouped_oof_requires_exact_session_and_preserves_provenance(
    tmp_path: Path,
):
    session = tmp_path / "recording_folder"
    session.mkdir()
    (session / "session_metadata.json").write_text(
        json.dumps({"session_id": "session_exact_001"}),
        encoding="utf-8",
    )
    prediction_path = (
        tmp_path / "outputs" / "training_run" / "force_contact_gate_oof_predictions.csv"
    )
    _write_oof(prediction_path, "session_exact_001")

    result = load_grouped_oof_evidence(
        session,
        _trace_rows(),
        tmp_path / "outputs",
    )

    assert result["ok"] is True
    assert result["source"] == "grouped_oof"
    assert result["evaluation_validity"] == "formal_grouped_oof_by_session_id"
    assert result["overlay"][4]["estimated_fz_n"] == pytest.approx(0.42)
    assert result["provenance"]["coverage_ratio"] == pytest.approx(1.0)
    assert result["provenance"]["fold_ids"] == ["fold_2"]
    assert result["provenance"]["prediction_source_file"] == str(
        prediction_path.resolve()
    )


def test_grouped_oof_does_not_use_another_session(tmp_path: Path):
    session = tmp_path / "wanted_session"
    session.mkdir()
    prediction_path = (
        tmp_path / "outputs" / "training_run" / "force_contact_gate_oof_predictions.csv"
    )
    _write_oof(prediction_path, "different_session")

    result = load_grouped_oof_evidence(
        session,
        _trace_rows(),
        tmp_path / "outputs",
    )

    assert result["ok"] is False
    assert result["status"] == "grouped_oof_not_found_for_session"


def test_grouped_oof_keeps_exact_partial_coverage_without_runtime_fallback(
    tmp_path: Path,
):
    session = tmp_path / "partial_session"
    session.mkdir()
    prediction_path = (
        tmp_path / "outputs" / "training_run" / "force_contact_gate_oof_predictions.csv"
    )
    _write_oof(prediction_path, "partial_session", count=7)

    result = load_grouped_oof_evidence(
        session,
        _trace_rows(count=10),
        tmp_path / "outputs",
    )

    assert result["ok"] is True
    assert result["source"] == "grouped_oof"
    assert result["provenance"]["coverage_ratio"] == pytest.approx(0.7)
    assert result["provenance"]["coverage_status"] == "partial_trace_coverage"
    assert result["provenance"]["unmatched_trace_row_count"] == 3
    assert sorted(result["overlay"]) == list(range(7))


def test_recorded_runtime_is_explicitly_historical():
    result = recorded_runtime_evidence(_trace_rows(3))

    assert result["ok"] is True
    assert result["source"] == "recorded_runtime"
    assert result["evaluation_validity"] == (
        "historical_capture_time_output_not_current_model"
    )
    assert result["label"] == (
        "Historical recorded runtime (capture-time model)"
    )
    assert result["overlay"][0]["estimated_fz_n"] == 8.0
    assert result["provenance"]["model_sources"] == ["recorded_old_model"]


def test_best_available_prefers_grouped_oof_over_recorded_runtime(tmp_path: Path):
    session = tmp_path / "session_preferred"
    session.mkdir()
    prediction_path = (
        tmp_path / "outputs" / "training_run" / "force_contact_gate_oof_predictions.csv"
    )
    _write_oof(prediction_path, "session_preferred")

    result = resolve_measurement_estimate_evidence(
        session,
        _trace_rows(),
        "best_available",
        outputs_root=tmp_path / "outputs",
    )

    assert result["ok"] is True
    assert result["requested_source"] == "best_available"
    assert result["source"] == "grouped_oof"
    assert [attempt["source"] for attempt in result["resolution_attempts"]] == [
        "grouped_oof"
    ]
    assert result["overlay"][4]["estimated_fz_n"] == pytest.approx(0.42)


def test_best_available_records_fallback_attempts_without_mixing(tmp_path: Path):
    session = tmp_path / "session_runtime_fallback"
    session.mkdir()

    result = resolve_measurement_estimate_evidence(
        session,
        _trace_rows(3),
        "best_available",
        outputs_root=tmp_path / "outputs",
    )

    assert result["ok"] is True
    assert result["source"] == "recorded_runtime"
    assert [attempt["source"] for attempt in result["resolution_attempts"]] == [
        "grouped_oof",
        "current_model_replay",
        "recorded_runtime",
    ]
    assert result["overlay"][0]["estimated_fz_n"] == 8.0


def test_explicit_unconfigured_current_replay_does_not_silently_fallback(
    tmp_path: Path,
):
    session = tmp_path / "session_explicit_replay"
    session.mkdir()

    result = resolve_measurement_estimate_evidence(
        session,
        _trace_rows(3),
        "current_model_replay",
        outputs_root=tmp_path / "outputs",
    )

    assert result["ok"] is False
    assert result["source"] == "current_model_replay"
    assert result["requested_source"] == "current_model_replay"
    assert result["status"] == "current_model_replay_not_configured"
