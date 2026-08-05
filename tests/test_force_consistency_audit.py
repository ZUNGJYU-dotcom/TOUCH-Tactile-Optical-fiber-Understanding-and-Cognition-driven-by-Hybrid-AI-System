from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid_spectrum.force_consistency_audit import (
    build_force_consistency_tables,
    build_stable_plateau_tables,
    force_curve_metrics,
    infer_position_id,
)


def test_infer_position_id_uses_embedded_position() -> None:
    assert infer_position_id("20260803_154521_P13_continuous_trial_008") == "P13"
    assert infer_position_id("unlabeled") is None


def test_force_curve_metrics_reports_scale_and_lag() -> None:
    elapsed = np.arange(20, dtype=float) * 0.1
    reference = np.linspace(0.0, 5.0, 20)
    estimate = reference * 0.8
    metrics = force_curve_metrics(reference, estimate, elapsed)
    assert metrics["mae_n"] > 0.0
    assert abs(metrics["linear_slope_pred_vs_px6d"] - 0.8) < 1.0e-8
    assert abs(metrics["amplitude_ratio_p95_p05"] - 0.8) < 1.0e-8
    assert metrics["pearson_r"] > 0.999


def test_force_curve_metrics_separates_release_grace_from_idle_residual() -> None:
    elapsed = np.arange(10, dtype=float) * 0.25
    reference = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    estimate = np.asarray([0.0, 0.0, 1.0, 1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    metrics = force_curve_metrics(
        reference,
        estimate,
        elapsed,
        release_grace_sec=0.75,
    )
    assert metrics["zero_force_false_response_rate"] > 0.0
    assert metrics["zero_force_false_response_rate_after_grace"] == 0.0
    assert metrics["zero_force_after_grace_frame_count"] == 6


def test_tables_keep_sessions_and_positions_separate() -> None:
    rows = []
    for position_id in ("P11", "P21"):
        for trial in (1, 2):
            for sample in range(12):
                force = sample / 11 * 5.0
                rows.append(
                    {
                        "position_id": position_id,
                        "group_id": f"session_{position_id}_{trial}",
                        "file_id": f"session_{position_id}_{trial}",
                        "fold_id": trial - 1,
                        "sample_index": sample,
                        "elapsed_time_sec": sample * 0.1,
                        "true_force_n": force,
                        "gated_force_n": force * 0.95,
                    }
                )
    sessions, positions = build_force_consistency_tables(pd.DataFrame(rows))
    assert len(sessions) == 4
    assert positions["position_id"].astype(str).tolist() == ["P11", "P21"]
    assert set(positions["evaluation_validity"]) == {
        "formal_grouped_oof_by_session_id"
    }


def test_stable_plateau_tables_exclude_force_transitions() -> None:
    rows = []
    forces = [0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0]
    for trial in (1, 2):
        for sample, force in enumerate(forces):
            rows.append(
                {
                    "position_id": "P11",
                    "group_id": f"session_P11_{trial}",
                    "sample_index": sample,
                    "elapsed_time_sec": sample * 1.0,
                    "true_force_n": force,
                    "gated_force_n": force * 0.9,
                }
            )
    plateaus, summary = build_stable_plateau_tables(
        pd.DataFrame(rows),
        maximum_force_speed_n_per_sec=0.20,
        force_bin_width_n=0.50,
        minimum_frames_per_plateau=2,
    )
    assert set(np.round(plateaus["px6d_mean_n"], 6)) == {1.0, 2.0, 3.0}
    assert int(summary.iloc[0]["session_count"]) == 2
    assert abs(float(summary.iloc[0]["plateau_slope"]) - 0.9) < 1.0e-8
