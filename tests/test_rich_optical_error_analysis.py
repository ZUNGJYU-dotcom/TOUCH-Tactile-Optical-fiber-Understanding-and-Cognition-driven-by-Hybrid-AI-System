from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hybrid_spectrum.rich_optical_error_analysis import (
    PredictionSpec,
    classification_session_audit,
    delta_summary,
    force_session_audit,
    optical_feature_category,
)


def _classification_predictions() -> pd.DataFrame:
    rows = []
    truth = {"session_a": [0, 0], "session_b": [1, 1]}
    baseline = {"session_a": [1, 0], "session_b": [1, 1]}
    candidate = {"session_a": [0, 0], "session_b": [1, 0]}
    for model, view, values in (
        ("extra_trees", "peak", baseline),
        ("lightgbm", "rich", candidate),
    ):
        for group_id, labels in truth.items():
            for sample_index, true_value in enumerate(labels):
                rows.append(
                    {
                        "model_id": model,
                        "feature_view": view,
                        "task": "contact",
                        "group_id": group_id,
                        "sample_index": sample_index,
                        "fold_id": 0 if group_id == "session_a" else 1,
                        "true_value": true_value,
                        "predicted_value": values[group_id][sample_index],
                    }
                )
    return pd.DataFrame(rows)


def test_classification_session_audit_tracks_improved_and_worsened_sessions() -> None:
    audit = classification_session_audit(
        _classification_predictions(),
        task="contact",
        baseline=PredictionSpec("extra_trees", "peak"),
        candidate=PredictionSpec("lightgbm", "rich"),
        value_kind="integer",
    ).set_index("group_id")
    assert audit.loc["session_a", "accuracy_delta"] == pytest.approx(0.5)
    assert audit.loc["session_b", "accuracy_delta"] == pytest.approx(-0.5)
    assert audit.loc["session_a", "no_contact_recall_delta"] == pytest.approx(0.5)
    assert np.isnan(audit.loc["session_a", "active_contact_recall_delta"])
    assert audit.loc["session_b", "active_contact_recall_delta"] == pytest.approx(-0.5)
    summary = delta_summary(audit.reset_index(), "accuracy_delta", higher_is_better=True)
    assert summary["improved_sessions"] == 1
    assert summary["worsened_sessions"] == 1


def test_session_audit_rejects_misaligned_frames() -> None:
    predictions = _classification_predictions()
    drop = (
        (predictions["model_id"] == "lightgbm")
        & (predictions["group_id"] == "session_b")
        & (predictions["sample_index"] == 1)
    )
    with pytest.raises(ValueError, match="do not align"):
        classification_session_audit(
            predictions.loc[~drop],
            task="contact",
            baseline=PredictionSpec("extra_trees", "peak"),
            candidate=PredictionSpec("lightgbm", "rich"),
            value_kind="integer",
        )


def test_force_session_audit_reports_mae_improvement() -> None:
    rows = []
    for model, view, predicted in (
        ("extra_trees", "peak", [0.0, 0.5]),
        ("extra_trees", "rich", [0.0, 0.9]),
    ):
        for sample_index, (true_value, prediction) in enumerate(
            zip([0.0, 1.0], predicted)
        ):
            rows.append(
                {
                    "model_id": model,
                    "feature_view": view,
                    "task": "force_fz",
                    "group_id": "session_a",
                    "sample_index": sample_index,
                    "fold_id": 0,
                    "true_value": true_value,
                    "predicted_value": prediction,
                }
            )
    audit = force_session_audit(
        pd.DataFrame(rows),
        baseline=PredictionSpec("extra_trees", "peak"),
        candidate=PredictionSpec("extra_trees", "rich"),
    )
    assert audit.loc[0, "baseline_mae_n"] == pytest.approx(0.25)
    assert audit.loc[0, "candidate_mae_n"] == pytest.approx(0.05)
    assert audit.loc[0, "mae_improvement_n"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("rich__coupling_log_area_ratio_same_fibre_fibre_1_std", "same_fibre_coupling"),
        ("rich__coupling_shape_rmse_spatial_row_row_2_mean", "cross_fibre_spatial_coupling"),
        ("rich__global_normalized_residual_rms", "distributed_spectral_shape"),
        ("rich__fbg01_delta_centroid_pm", "wavelength_shift"),
        ("spectrum__spectrum_shape_delta_bin_027", "full_spectrum_bins"),
    ],
)
def test_optical_feature_category(feature: str, expected: str) -> None:
    assert optical_feature_category(feature) == expected
