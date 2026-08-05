#!/usr/bin/env python3
"""Deploy cross-date validated contact and position models into Beta.

The current optical force regressor is intentionally retained. The deployment
only promotes classification candidates that passed the strict leave-one-date-
out benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.literature_guided_domain_models import (  # noqa: E402
    LiteratureGuidedPositionEnsemble,
    build_literature_feature_views,
    equal_group_weights,
    load_strict_cross_date_datasets,
    make_extra_trees_classifier,
)


DEFAULT_DATASETS = {
    "20260803": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260803_new_data_only"
    / "ordinary_fbg_px6d_dataset.npz",
    "20260804": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260804_new_batch_only"
    / "ordinary_fbg_px6d_dataset.npz",
}
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "ordinary_fbg_optical_only_force_candidate.joblib"
)
DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_literature_guided_cross_date_20260804"
    / "literature_guided_aggregate_metrics.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_literature_guided_beta_deployment_20260804"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--estimators", type=int, default=128)
    parser.add_argument("--minimum-leaf-samples", type=int, default=2)
    parser.add_argument("--temporal-window-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _metric_row(path: Path, candidate_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("candidate_id") == candidate_id:
                result: dict[str, Any] = {}
                for key, value in row.items():
                    if value in (None, ""):
                        continue
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
                return result
    raise KeyError(f"candidate {candidate_id!r} is absent from {path}")


def _fit_classifier(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    estimators: int,
    minimum_leaf_samples: int,
    seed: int,
) -> tuple[Any, float]:
    model = make_extra_trees_classifier(
        estimators=estimators,
        minimum_leaf_samples=minimum_leaf_samples,
        seed=seed,
    )
    started = time.perf_counter()
    model.fit(
        features,
        target,
        sample_weight=equal_group_weights(groups),
    )
    model.n_jobs = 1
    return model, float(time.perf_counter() - started)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _update_array_digest(digest: Any, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _force_task_fingerprint(task: Any) -> str:
    """Hash the fitted forest structure without joblib's unstable object state."""
    if not isinstance(task, dict):
        raise TypeError("force task must be a mapping")
    model = task["model"]
    digest = hashlib.sha256()
    digest.update(
        f"{type(model).__module__}.{type(model).__qualname__}".encode("utf-8")
    )
    parameters = model.get_params(deep=False)
    digest.update(
        json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    )
    for key in ("feature_indices", "feature_names", "classes"):
        _update_array_digest(digest, task[key])
    for estimator in getattr(model, "estimators_", ()):  # ExtraTrees forest
        tree = estimator.tree_
        for attribute in (
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
        ):
            _update_array_digest(digest, getattr(tree, attribute))
    return digest.hexdigest()


def _force_probe_predictions(task: Any) -> np.ndarray:
    model = task["model"]
    feature_count = int(getattr(model, "n_features_in_", 0))
    if feature_count < 1:
        raise ValueError("force model does not expose a valid feature contract")
    rng = np.random.default_rng(20260804)
    probe = np.vstack(
        [
            np.zeros(feature_count, dtype=np.float64),
            np.ones(feature_count, dtype=np.float64),
            np.linspace(-1.0, 1.0, feature_count, dtype=np.float64),
            rng.normal(0.0, 0.5, feature_count),
            rng.uniform(-2.0, 2.0, feature_count),
        ]
    )
    return np.asarray(model.predict(probe), dtype=np.float64)


def main() -> int:
    args = parse_args()
    missing = [str(path) for path in (*DEFAULT_DATASETS.values(), args.model, args.benchmark) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing deployment input(s): " + ", ".join(missing))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_strict_cross_date_datasets(DEFAULT_DATASETS)
    views = build_literature_feature_views(
        dataset,
        temporal_window_frames=args.temporal_window_frames,
    )

    contact_mask = dataset.contact_mask
    contact_target = np.where(
        dataset.contact_target[contact_mask] == 1,
        "contact",
        "no_contact",
    )
    contact_model, contact_fit_seconds = _fit_classifier(
        views["literature_snv_sg_temporal488"].values[contact_mask],
        contact_target,
        dataset.group_id[contact_mask],
        estimators=args.estimators,
        minimum_leaf_samples=args.minimum_leaf_samples,
        seed=args.seed,
    )

    position_mask = dataset.position_mask
    position_target = dataset.position_target[position_mask]
    raw_model, raw_fit_seconds = _fit_classifier(
        views["response_raw136"].values[position_mask],
        position_target,
        dataset.group_id[position_mask],
        estimators=args.estimators,
        minimum_leaf_samples=args.minimum_leaf_samples,
        seed=args.seed + 1,
    )
    normalized_model, normalized_fit_seconds = _fit_classifier(
        views["literature_snv_sg328"].values[position_mask],
        position_target,
        dataset.group_id[position_mask],
        estimators=args.estimators,
        minimum_leaf_samples=args.minimum_leaf_samples,
        seed=args.seed + 2,
    )
    position_model = LiteratureGuidedPositionEnsemble(
        raw_model,
        normalized_model,
    )

    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict):
        raise TypeError("Beta model artifact must contain a mapping")
    force_task_before = bundle["tasks"]["force_fz"]
    force_fingerprint_before = _force_task_fingerprint(force_task_before)
    force_probe_before = _force_probe_predictions(force_task_before)
    archive_dir = args.model.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / (
        "ordinary_fbg_optical_only_force_candidate_pre_literature_20260804.joblib"
    )
    if not archive_path.exists():
        shutil.copy2(args.model, archive_path)

    contact_validation = _metric_row(
        args.benchmark,
        "contact_literature_snv_sg_temporal488_extra_trees",
    )
    position_validation = _metric_row(
        args.benchmark,
        "position_literature_probability_ensemble",
    )
    bundle["literature_guided_classification"] = {
        "schema_version": "literature_guided_contact_position_v1",
        "created_for": "beta_cross_date_classification_deployment_20260804",
        "static_feature_schema": "baseline_relative_full_spectrum_264",
        "static_feature_names": dataset.feature_names[:264].astype(str).tolist(),
        "bin_count": 64,
        "temporal_window_frames": int(args.temporal_window_frames),
        "contact": {
            "model": contact_model,
            "feature_view": "literature_snv_sg_temporal488",
            "feature_count": 488,
            "candidate_id": "contact_literature_snv_sg_temporal488_extra_trees",
        },
        "position": {
            "model": position_model,
            "feature_view": "response_raw136+literature_snv_sg328",
            "feature_count": 264,
            "candidate_id": "position_literature_probability_ensemble",
        },
        "validation": {
            "evaluation_validity": "leave_one_date_out_grouped_by_session",
            "dates": sorted(np.unique(dataset.acquisition_date).tolist()),
            "contact_mean_macro_f1": contact_validation.get("mean_macro_f1"),
            "contact_worst_macro_f1": contact_validation.get("worst_macro_f1"),
            "position_mean_macro_f1": position_validation.get("mean_macro_f1"),
            "position_worst_macro_f1": position_validation.get("worst_macro_f1"),
            "position_group_voting_accuracy": position_validation.get(
                "mean_group_voting_accuracy"
            ),
        },
        "force_model_replaced": False,
    }
    bundle.setdefault("deployment_metadata", {})[
        "classification_upgrade"
    ] = "literature_guided_cross_date_v1"

    force_task_after = bundle["tasks"]["force_fz"]
    force_fingerprint_after = _force_task_fingerprint(force_task_after)
    force_probe_after = _force_probe_predictions(force_task_after)
    if force_fingerprint_after != force_fingerprint_before or not np.allclose(
        force_probe_after,
        force_probe_before,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("force model changed during classification deployment")

    temporary_path = args.model.with_suffix(".joblib.deploying")
    joblib.dump(bundle, temporary_path, compress=0)
    verified = joblib.load(temporary_path)
    verified_force_task = verified["tasks"]["force_fz"]
    serialized_force_fingerprint = _force_task_fingerprint(verified_force_task)
    serialized_force_probe = _force_probe_predictions(verified_force_task)
    if (
        serialized_force_fingerprint != force_fingerprint_before
        or not np.allclose(
            serialized_force_probe,
            force_probe_before,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("serialized force model structure or predictions changed")
    os.replace(temporary_path, args.model)

    report = {
        "status": "deployed",
        "model_path": str(args.model.resolve()),
        "archive_path": str(archive_path.resolve()),
        "dataset_rows": int(len(dataset.features)),
        "independent_sessions": int(len(np.unique(dataset.group_id))),
        "training_dates": sorted(np.unique(dataset.acquisition_date).tolist()),
        "contact_training_rows": int(np.sum(contact_mask)),
        "position_training_rows": int(np.sum(position_mask)),
        "contact_fit_seconds": contact_fit_seconds,
        "position_raw_fit_seconds": raw_fit_seconds,
        "position_normalized_fit_seconds": normalized_fit_seconds,
        "force_model_replaced": False,
        "force_task_fingerprint_before": force_fingerprint_before,
        "force_task_fingerprint_after": force_fingerprint_after,
        "force_task_fingerprint_serialized": serialized_force_fingerprint,
        "force_probe_max_abs_serialization_delta_n": float(
            np.max(np.abs(serialized_force_probe - force_probe_before))
        ),
        "contact_validation": contact_validation,
        "position_validation": position_validation,
    }
    (output_dir / "deployment_summary.json").write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "deployment_report.md").write_text(
        "# Literature-guided Beta classification deployment\n\n"
        "The cross-date validated contact and position classifiers were deployed "
        "into the Beta bundle. The existing optical force regressor was retained "
        "byte-for-object-hash and was not replaced.\n\n"
        f"- Training rows: {len(dataset.features):,}\n"
        f"- Independent sessions: {len(np.unique(dataset.group_id))}\n"
        f"- Contact macro-F1 (cross-date mean): {float(contact_validation['mean_macro_f1']):.3f}\n"
        f"- Position macro-F1 (cross-date mean): {float(position_validation['mean_macro_f1']):.3f}\n"
        f"- Position file/session vote accuracy: {float(position_validation['mean_group_voting_accuracy']):.3f}\n"
        "- Force model replaced: no\n"
        "- Force upper clip introduced: no\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
