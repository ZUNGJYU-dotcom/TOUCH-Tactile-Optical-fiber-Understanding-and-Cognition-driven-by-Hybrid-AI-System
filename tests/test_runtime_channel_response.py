from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_spectrum.all_source_runtime_adapter import (  # noqa: E402
    DISPLAY_ROWS,
    POSITION_COORDINATES,
    POSITION_ORDER,
)
from hybrid_spectrum.features import load_peak_windows  # noqa: E402
from hybrid_spectrum.runtime_channel_response import (  # noqa: E402
    build_inferred_contact_probability_surface,
    build_observed_coupled_spectral_response,
    summarize_coupled_contact_signature,
)


PEAK_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml"


def _grid_value(grid: list[list[float]], channel_id: str) -> float:
    for row_index, row in enumerate(DISPLAY_ROWS):
        if channel_id in row:
            return float(grid[row_index][row.index(channel_id)])
    raise AssertionError(f"unknown channel {channel_id}")


def test_inferred_surface_uses_current_posterior_without_gaussian_spread() -> None:
    probabilities = {channel_id: 0.01 for channel_id in POSITION_ORDER}
    probabilities.update({"P11": 0.46, "P21": 0.23, "P33": 0.23})

    result = build_inferred_contact_probability_surface(
        probabilities,
        accepted_position_id="P11",
        deformation=0.8,
        active=True,
        position_order=POSITION_ORDER,
        display_rows=DISPLAY_ROWS,
        position_coordinates=POSITION_COORDINATES,
    )

    probability_grid = result["probability_grid"]
    surface_grid = result["surface_grid"]
    assert result["map_kind"] == "inferred_contact_probability"
    assert result["map_source"] == "trained_position_predict_proba"
    assert result["dominant_channel"] == "P11"
    assert _grid_value(surface_grid, "P11") == pytest.approx(0.8)
    assert _grid_value(surface_grid, "P21") == pytest.approx(0.4)
    assert _grid_value(surface_grid, "P33") == pytest.approx(0.4)
    assert _grid_value(surface_grid, "P21") == pytest.approx(
        _grid_value(surface_grid, "P33")
    )
    assert _grid_value(probability_grid, "P21") == pytest.approx(
        _grid_value(probability_grid, "P33")
    )


def test_inactive_surface_is_zero_but_keeps_current_posterior() -> None:
    probabilities = {
        channel_id: 1.0 if channel_id == "P32" else 0.0
        for channel_id in POSITION_ORDER
    }

    result = build_inferred_contact_probability_surface(
        probabilities,
        accepted_position_id="P32",
        deformation=0.9,
        active=False,
        position_order=POSITION_ORDER,
        display_rows=DISPLAY_ROWS,
        position_coordinates=POSITION_COORDINATES,
    )

    assert np.max(np.asarray(result["surface_grid"], dtype=float)) == 0.0
    assert _grid_value(result["probability_grid"], "P32") == pytest.approx(1.0)
    assert result["dominant_channel"] is None


def test_inferred_surface_sanitizes_non_finite_runtime_values() -> None:
    probabilities = {
        "P11": float("nan"),
        "P21": float("inf"),
        "P31": "invalid",
        "P22": 0.75,
    }

    result = build_inferred_contact_probability_surface(
        probabilities,
        accepted_position_id="P22",
        deformation=float("nan"),
        active=True,
        position_order=POSITION_ORDER,
        display_rows=DISPLAY_ROWS,
        position_coordinates=POSITION_COORDINATES,
    )

    probability_grid = np.asarray(result["probability_grid"], dtype=float)
    surface_grid = np.asarray(result["surface_grid"], dtype=float)
    assert np.all(np.isfinite(probability_grid))
    assert np.all(np.isfinite(surface_grid))
    assert _grid_value(result["probability_grid"], "P22") == pytest.approx(1.0)
    assert np.max(surface_grid) == 0.0
    assert result["surface_metrics"]["surface_peak"] == 0.0


def test_observed_nine_peak_evidence_is_current_frame_and_separate() -> None:
    windows = load_peak_windows(PEAK_CONFIG_PATH)
    names: list[str] = []
    values: list[float] = []
    for window in windows:
        prefix = window.candidate_id.lower()
        names.extend(
            [
                f"{prefix}_centroid_shift_pm",
                f"{prefix}_log_area_ratio",
                f"{prefix}_log_height_ratio",
                f"{prefix}_shape_rmse",
            ]
        )
        if window.provisional_channel_id == "P23":
            values.extend([300.0, 0.30, -0.25, 0.12])
        else:
            values.extend([5.0, 0.002, -0.002, 0.001])

    result = build_observed_coupled_spectral_response(
        values,
        names,
        windows,
        DISPLAY_ROWS,
    )

    assert result["kind"] == "observed_coupled_spectral_response"
    assert result["mapping_status"] == "provisional_fbg_to_position_mapping"
    assert result["dominant_channel"] == "P23"
    assert result["responding_channel_ids"] == ["P23"]
    assert _grid_value(result["response_grid"], "P23") > 0.5
    assert _grid_value(result["response_grid"], "P11") < 0.055
    assert "not_independent_force_pixels" in result["semantics"]


def test_quiet_observed_evidence_has_no_dominant_channel() -> None:
    windows = load_peak_windows(PEAK_CONFIG_PATH)
    names = [
        f"{window.candidate_id.lower()}_{suffix}"
        for window in windows
        for suffix in (
            "centroid_shift_pm",
            "log_area_ratio",
            "log_height_ratio",
            "shape_rmse",
        )
    ]

    result = build_observed_coupled_spectral_response(
        np.zeros(len(names), dtype=float),
        names,
        windows,
        DISPLAY_ROWS,
    )

    assert result["dominant_channel"] is None
    assert result["responding_channel_ids"] == []
    assert result["peak_response"] == 0.0


def test_joint_contact_signature_rejects_sparse_peak_disturbance() -> None:
    response = {
        "channels": [
            {
                "provisional_channel_id": channel_id,
                "evidence_score": score,
            }
            for channel_id, score in zip(
                POSITION_ORDER,
                (0.029, 0.015, 0.009, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001),
                strict=True,
            )
        ]
    }

    result = summarize_coupled_contact_signature(
        response,
        low_response_threshold=0.005,
        nominal_response_threshold=0.010,
        minimum_low_response_channels=5,
        minimum_nominal_response_channels=3,
    )

    assert result["all_channels_present"] is True
    assert result["low_response_channel_count"] == 5
    assert result["nominal_response_channel_count"] == 2
    assert result["multichannel_pattern"] is False
    assert "all_nine_gratings" in result["semantics"]


def test_joint_contact_signature_accepts_distributed_nine_peak_response() -> None:
    response = {
        "channels": [
            {
                "provisional_channel_id": channel_id,
                "evidence_score": score,
            }
            for channel_id, score in zip(
                POSITION_ORDER,
                (0.028, 0.020, 0.014, 0.009, 0.008, 0.004, 0.003, 0.002, 0.001),
                strict=True,
            )
        ]
    }

    result = summarize_coupled_contact_signature(
        response,
        low_response_threshold=0.005,
        nominal_response_threshold=0.010,
        minimum_low_response_channels=5,
        minimum_nominal_response_channels=3,
    )

    assert result["low_response_channel_count"] == 5
    assert result["nominal_response_channel_count"] == 3
    assert result["multichannel_pattern"] is True
    assert result["third_response"] == pytest.approx(0.014)
