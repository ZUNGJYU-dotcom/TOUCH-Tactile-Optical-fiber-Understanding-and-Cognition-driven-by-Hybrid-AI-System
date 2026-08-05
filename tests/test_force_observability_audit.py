import numpy as np

from src.hybrid_spectrum.advanced_optical_benchmark import AlignedOpticalDataset
from src.hybrid_spectrum.force_observability_audit import (
    build_session_observability_table,
)


def _dataset() -> AlignedOpticalDataset:
    force = np.tile(np.linspace(0.1, 5.0, 20), 4)
    groups = np.repeat(["normal_a", "normal_b", "normal_c", "weak"], 20)
    sensitivity = np.repeat([1.0, 0.95, 1.05, 0.25], 20)
    spectrum = np.column_stack(
        (
            sensitivity * force,
            0.5 * sensitivity * force,
            0.25 * sensitivity * force,
        )
    ).astype(np.float32)
    count = len(force)
    return AlignedOpticalDataset(
        peak_features=np.zeros((count, 1), dtype=np.float32),
        peak_feature_names=np.asarray(["unused"]),
        spectrum_features=spectrum,
        spectrum_feature_names=np.asarray(
            [
                "global_log_ratio_rms",
                "global_shape_delta_rms",
                "global_intensity_log_ratio",
            ]
        ),
        contact_target=np.ones(count, dtype=int),
        position_target=np.asarray(["P13"] * count),
        force_fz_n=force,
        contact_mask=np.ones(count, dtype=bool),
        position_mask=np.ones(count, dtype=bool),
        force_mask=np.ones(count, dtype=bool),
        fold_id=np.repeat([0, 1, 2, 3], 20),
        group_id=groups,
        sample_index=np.tile(np.arange(20), 4),
    )


def test_observability_audit_marks_low_sensitivity_without_excluding_it():
    table = build_session_observability_table(_dataset())
    weak = table[table["group_id"] == "weak"].iloc[0]
    assert weak["observability_status"] == "low_optical_sensitivity"
    assert weak["sensitivity_ratio_to_position_median"] < 0.30
    assert len(table) == 4
    assert not table["force_sensor_used_as_runtime_input"].any()
