from __future__ import annotations

import numpy as np

from src.hybrid_spectrum.advanced_optical_benchmark import AlignedOpticalDataset
from src.hybrid_spectrum.rich_optical_benchmark import (
    FeatureView,
    build_rich_feature_views,
    grouped_classification,
    grouped_force_regression,
)
from src.hybrid_spectrum.rich_optical_features import RichFeatureCache


def _aligned_dataset() -> AlignedOpticalDataset:
    rows = 12
    peak_names = np.asarray(
        [f"mean__feature_{index}" for index in range(443)]
        + [f"last__feature_{index}" for index in range(40)]
    )
    return AlignedOpticalDataset(
        peak_features=np.zeros((rows, 483), dtype=np.float32),
        peak_feature_names=peak_names,
        spectrum_features=np.zeros((rows, 264), dtype=np.float32),
        spectrum_feature_names=np.asarray(
            [f"spectrum_{index}" for index in range(264)]
        ),
        contact_target=np.asarray([0, 0, 1, 1] * 3),
        position_target=np.asarray(["", "", "P11", "P11"] * 3),
        force_fz_n=np.asarray([0.0, 0.0, 1.0, 1.0] * 3),
        contact_mask=np.ones(rows, dtype=bool),
        position_mask=np.asarray([False, False, True, True] * 3),
        force_mask=np.ones(rows, dtype=bool),
        fold_id=np.repeat(np.arange(3), 4),
        group_id=np.repeat(np.asarray(["g0", "g1", "g2"]), 4),
        sample_index=np.arange(rows),
    )


def test_rich_feature_views_remain_aligned() -> None:
    dataset = _aligned_dataset()
    cache = RichFeatureCache(
        features=np.ones((12, 3), dtype=np.float32),
        feature_names=np.asarray(["a", "b", "c"]),
        group_id=dataset.group_id,
        sample_index=dataset.sample_index,
    )

    views = build_rich_feature_views(dataset, cache)

    assert views["peak_current_40"].values.shape == (12, 40)
    assert views["full_spectrum_192"].values.shape == (12, 192)
    assert views["rich_optical_physics"].values.shape == (12, 3)
    assert views["rich_plus_full_spectrum_192"].values.shape == (12, 195)


def test_grouped_estimators_return_complete_predictions() -> None:
    groups = np.repeat(np.asarray([f"g{index}" for index in range(6)]), 4)
    folds = np.repeat(np.asarray([0, 1, 2, 0, 1, 2]), 4)
    target = np.tile(np.asarray([0, 0, 1, 1]), 6)
    features = np.column_stack(
        (target.astype(float), np.arange(len(target), dtype=float) / 100.0)
    ).astype(np.float32)
    view = FeatureView(features, np.asarray(["signal", "index"]))

    classification, class_prediction = grouped_classification(
        model_id="extra_trees",
        feature_view=view,
        target=target,
        mask=np.ones(len(target), dtype=bool),
        fold_id=folds,
        group_id=groups,
        labels=[0, 1],
        estimators=16,
        minimum_leaf_samples=1,
        seed=12,
    )
    regression, force_prediction = grouped_force_regression(
        model_id="extra_trees",
        feature_view=view,
        target=target.astype(float),
        mask=np.ones(len(target), dtype=bool),
        fold_id=folds,
        group_id=groups,
        estimators=16,
        minimum_leaf_samples=1,
        seed=12,
    )

    assert classification["macro_f1"] == 1.0
    assert np.all(class_prediction != None)  # noqa: E711
    assert regression["mae_n"] < 0.20
    assert np.all(np.isfinite(force_prediction))
