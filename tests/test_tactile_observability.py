from pathlib import Path

import numpy as np
import pandas as pd

from src.hybrid_spectrum.tactile_observability import (
    derive_force_phase_labels,
    grouped_regression_observability,
    load_aligned_mechanical_targets,
    validate_force_alignment,
)


def test_load_aligned_mechanical_targets_uses_session_and_capture_index(tmp_path: Path):
    groups = np.asarray(["session_b", "session_a", "session_b", "session_a"])
    indices = np.asarray([20, 10, 10, 20])
    for session, offset in (("session_a", 0.0), ("session_b", 100.0)):
        directory = tmp_path / session
        directory.mkdir()
        pd.DataFrame(
            {
                "capture_index": [10, 20],
                "elapsed_time_sec": [0.0, 0.5],
                "force_fz_n": [offset + 1.0, offset + 2.0],
                "fx_filtered_n": [offset + 3.0, offset + 4.0],
            }
        ).to_csv(directory / "force_timeseries.csv", index=False)
    aligned = load_aligned_mechanical_targets(
        capture_root=tmp_path,
        group_id=groups,
        sample_index=indices,
        target_columns=("force_fz_n", "fx_filtered_n"),
    )
    assert aligned.values["force_fz_n"].tolist() == [102.0, 1.0, 101.0, 2.0]
    assert aligned.elapsed_time_sec.tolist() == [0.5, 0.0, 0.0, 0.5]
    assert validate_force_alignment(
        np.asarray([102.0, 1.0, 101.0, 2.0]), aligned.values["force_fz_n"]
    ) == 0.0


def test_derive_force_phase_labels_tracks_loading_hold_and_release():
    labels, slope, valid = derive_force_phase_labels(
        force_fz_n=np.asarray([0.0, 0.0, 0.3, 0.8, 0.8, 0.3, 0.0]),
        elapsed_time_sec=np.arange(7, dtype=float),
        group_id=np.asarray(["s1"] * 7),
        no_contact_threshold_n=0.05,
        slope_threshold_n_per_sec=0.15,
        smoothing_window=1,
    )
    assert labels[0] == "no_contact"
    assert labels[2] == "loading"
    assert labels[4] == "release"
    assert labels[-1] == "no_contact"
    assert np.all(np.isfinite(slope))
    assert np.all(valid)


def test_grouped_regression_observability_keeps_sessions_separate():
    rng = np.random.default_rng(42)
    groups = np.repeat([f"g{index}" for index in range(9)], 30)
    folds = np.repeat(np.arange(9) % 3, 30)
    latent = rng.normal(size=len(groups))
    features = np.column_stack((latent, latent**2, rng.normal(size=len(groups))))
    target = 2.0 * latent + 0.05 * rng.normal(size=len(groups))
    position = np.asarray(["P11", "P22", "P33"] * 90)
    metrics, prediction = grouped_regression_observability(
        features=features,
        feature_names=np.asarray(["latent", "squared", "noise"]),
        target=target,
        mask=np.ones(len(groups), dtype=bool),
        fold_id=folds,
        group_id=groups,
        position_target=position,
        contact_target=np.ones(len(groups), dtype=int),
        estimators=48,
        seed=5,
    )
    assert metrics["session_count"] == 9
    assert metrics["r2"] > 0.90
    assert metrics["skill_over_position_baseline"] > 0.70
    assert np.all(np.isfinite(prediction))
