#!/usr/bin/env python3
"""Fit the validated literature-guided models on all audited acquisition dates.

This script creates a candidate artifact only. It never overwrites the model used
by the desktop application; deployment remains a separate, explicit step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.literature_guided_domain_models import (  # noqa: E402
    balanced_group_indices,
    build_literature_feature_views,
    equal_group_weights,
    load_strict_cross_date_datasets,
    make_extra_trees_classifier,
    make_osc_ridge_regressor,
    nonnegative_prediction,
)


DEFAULT_DATASETS = {
    "20260731": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260731_latest_primary"
    / "primary"
    / "ordinary_fbg_px6d_dataset.npz",
    "20260803": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260803_new_data_only"
    / "ordinary_fbg_px6d_dataset.npz",
    "20260804": PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260804_new_batch_only"
    / "ordinary_fbg_px6d_dataset.npz",
}
DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_literature_guided_cross_date_20260731_20260804"
    / "literature_guided_aggregate_metrics.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_literature_guided_cross_date_20260731_20260804"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "ordinary_fbg_three_date_literature_candidate.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date-dataset",
        action="append",
        default=[],
        metavar="DATE=PATH",
        help="Audited strict NPZ; repeat for every acquisition date.",
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--estimators", type=int, default=128)
    parser.add_argument("--minimum-leaf-samples", type=int, default=2)
    parser.add_argument("--temporal-window-frames", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--osc-components", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_datasets(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_DATASETS)
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --date-dataset value: {value!r}")
        date, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        result[date.strip()] = path.resolve()
    if len(result) < 2:
        raise ValueError("at least two acquisition dates are required")
    return result


def metric_row(path: Path, candidate_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("candidate_id") != candidate_id:
                continue
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            return parsed
    raise KeyError(f"candidate {candidate_id!r} is absent from {path}")


def fit_classifier(
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
    model.fit(features, target, sample_weight=equal_group_weights(groups))
    model.n_jobs = 1
    return model, float(time.perf_counter() - started)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    datasets = parse_datasets(args.date_dataset)
    missing = [str(path) for path in (*datasets.values(), args.benchmark) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing training input(s): " + ", ".join(missing))

    data = load_strict_cross_date_datasets(datasets)
    views = build_literature_feature_views(
        data,
        temporal_window_frames=args.temporal_window_frames,
    )

    contact_mask = data.contact_mask
    contact_target = np.where(
        data.contact_target[contact_mask] == 1,
        "contact",
        "no_contact",
    )
    contact_model, contact_fit_seconds = fit_classifier(
        views["response_raw136"].values[contact_mask],
        contact_target,
        data.group_id[contact_mask],
        estimators=args.estimators,
        minimum_leaf_samples=args.minimum_leaf_samples,
        seed=args.seed,
    )

    position_mask = data.position_mask
    position_model, position_fit_seconds = fit_classifier(
        views["response_raw136"].values[position_mask],
        data.position_target[position_mask],
        data.group_id[position_mask],
        estimators=args.estimators,
        minimum_leaf_samples=args.minimum_leaf_samples,
        seed=args.seed + 1,
    )

    force_mask = data.force_mask
    force_groups = data.group_id[force_mask]
    force_selected = balanced_group_indices(force_groups, maximum_rows_per_group=160)
    force_model = make_osc_ridge_regressor(
        alpha=args.ridge_alpha,
        osc_components=args.osc_components,
    )
    force_started = time.perf_counter()
    force_model.fit(
        views["literature_snv_sg328"].values[force_mask][force_selected],
        data.force_fz_n[force_mask][force_selected],
    )
    force_fit_seconds = float(time.perf_counter() - force_started)

    candidates = {
        "contact": "contact_response_raw136_extra_trees",
        "position": "position_response_raw136_extra_trees",
        "force_fz": "force_literature_train_only_osc_ridge",
    }
    validation = {
        task: metric_row(args.benchmark, candidate_id)
        for task, candidate_id in candidates.items()
    }
    acquisition_dates = sorted(np.unique(data.acquisition_date).astype(str).tolist())
    bundle = {
        "schema_version": "ordinary_fbg_three_date_literature_candidate_v1",
        "deployment_status": "candidate_not_deployed",
        "evaluation_validity": "leave_one_date_out_grouped_by_session",
        "training_dates": acquisition_dates,
        "dataset_rows": int(len(data.features)),
        "independent_sessions": int(len(np.unique(data.group_id))),
        "wavelength_nm": data.wavelength_nm,
        "tasks": {
            "contact": {
                "model": contact_model,
                "candidate_id": candidates["contact"],
                "feature_view": "response_raw136",
                "feature_names": views["response_raw136"].names,
                "classes": np.asarray(contact_model.classes_),
                "training_rows": int(np.sum(contact_mask)),
            },
            "position": {
                "model": position_model,
                "candidate_id": candidates["position"],
                "feature_view": "response_raw136",
                "feature_names": views["response_raw136"].names,
                "classes": np.asarray(position_model.classes_),
                "training_rows": int(np.sum(position_mask)),
            },
            "force_fz": {
                "model": force_model,
                "candidate_id": candidates["force_fz"],
                "feature_view": "literature_snv_sg328",
                "feature_names": views["literature_snv_sg328"].names,
                "training_rows": int(np.sum(force_mask)),
                "balanced_training_rows": int(len(force_selected)),
                "output_policy": "nonnegative_without_upper_clip",
            },
        },
        "validation": validation,
        "fit_seconds": {
            "contact": contact_fit_seconds,
            "position": position_fit_seconds,
            "force_fz": force_fit_seconds,
        },
        "source_datasets": {key: str(path.resolve()) for key, path in datasets.items()},
        "quarantine_policy": {
            "20260731": "primary_only; suspected early P11/P12/P21 sessions excluded",
        },
    }

    model_output = args.model_output.resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_output.with_suffix(model_output.suffix + ".training")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(model_output)

    verified = joblib.load(model_output)
    probe_count = min(32, len(data.features))
    contact_probe = verified["tasks"]["contact"]["model"].predict(
        views["response_raw136"].values[:probe_count]
    )
    position_probe = verified["tasks"]["position"]["model"].predict(
        views["response_raw136"].values[:probe_count]
    )
    force_probe = nonnegative_prediction(
        verified["tasks"]["force_fz"]["model"].predict(
            views["literature_snv_sg328"].values[:probe_count]
        )
    )
    if len(contact_probe) != probe_count or len(position_probe) != probe_count:
        raise RuntimeError("serialized classifier prediction contract failed")
    if len(force_probe) != probe_count or not np.all(np.isfinite(force_probe)):
        raise RuntimeError("serialized force prediction contract failed")

    digest = hashlib.sha256(model_output.read_bytes()).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "trained_and_verified_not_deployed",
        "model_path": str(model_output),
        "model_sha256": digest,
        "model_size_mb": model_output.stat().st_size / (1024.0 * 1024.0),
        "training_dates": acquisition_dates,
        "dataset_rows": int(len(data.features)),
        "independent_sessions": int(len(np.unique(data.group_id))),
        "contact_training_rows": int(np.sum(contact_mask)),
        "position_training_rows": int(np.sum(position_mask)),
        "force_training_rows": int(np.sum(force_mask)),
        "force_balanced_training_rows": int(len(force_selected)),
        "validation": validation,
        "quarantine_policy": bundle["quarantine_policy"],
    }
    (output_dir / "three_date_candidate_training_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "three_date_candidate_training_report.md").write_text(
        "# Three-date literature-guided candidate\n\n"
        "The candidate was fitted on all audited 20260731, 20260803, and "
        "20260804 primary sessions after strict leave-one-date-out validation. "
        "It has not been deployed to the desktop application.\n\n"
        f"- Rows: {len(data.features):,}\n"
        f"- Independent sessions: {len(np.unique(data.group_id))}\n"
        f"- Contact training rows: {np.sum(contact_mask):,}\n"
        f"- Position training rows: {np.sum(position_mask):,}\n"
        f"- Force training rows: {np.sum(force_mask):,}\n"
        "- 20260731 policy: primary only; suspected early P11/P12/P21 "
        "sessions remain quarantined.\n"
        "- Force output: nonnegative and without an artificial upper-force cap.\n"
        f"- Artifact SHA-256: `{digest}`\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
