"""Validate the all-source candidate without touching the deployed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_spectrum.all_source_fusion import resolve_project_path
from hybrid_spectrum.all_source_training import load_fusion_arrays, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/ordinary_fbg_all_data_fusion.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/ordinary_fbg_all_data_fusion_20260731_v1/"
            "all_source_fusion_dataset.npz"
        ),
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/ordinary_fbg_all_data_fusion_training_20260731_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(
        args.config.expanduser().resolve().read_text(encoding="utf-8")
    )
    arrays = load_fusion_arrays(args.dataset.expanduser().resolve())
    output_dir = args.training_output.expanduser().resolve()
    split_audit = pd.read_csv(output_dir / "grouped_split_audit.csv")
    candidate_path = (
        output_dir
        / "candidate_models/ordinary_fbg_optical_only_force_candidate.joblib"
    )
    bundle = joblib.load(candidate_path)
    force_gate_predictions = pd.read_csv(
        output_dir / "force_contact_gate_oof_predictions.csv",
        low_memory=False,
    )
    force_gate_metrics = json.loads(
        (output_dir / "force_contact_gate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    protected_path = resolve_project_path(
        PROJECT_ROOT, config["paths"]["protected_deployed_model"]
    )
    training_summary = json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    forbidden = tuple(
        str(value).lower()
        for value in config["force_calibration"]["forbidden_input_patterns"]
    )
    forbidden_features = [
        str(name)
        for name in arrays.feature_names
        if any(pattern in str(name).lower() for pattern in forbidden)
    ]
    formal_group_folds: dict[str, set[int]] = {}
    for group, fold in zip(
        arrays.group_id[arrays.formal_test_eligible],
        arrays.fold_id[arrays.formal_test_eligible],
        strict=True,
    ):
        formal_group_folds.setdefault(str(group), set()).add(int(fold))
    multi_fold_groups = {
        group: sorted(folds)
        for group, folds in formal_group_folds.items()
        if len(folds) != 1
    }
    expected_formal_group_count = int(
        config.get("evaluation", {}).get(
            "expected_formal_group_count", len(formal_group_folds)
        )
    )
    inference_checks: dict[str, Any] = {}
    smoke_indices = np.arange(min(8, len(arrays.features)), dtype=int)
    for task, task_bundle in bundle["tasks"].items():
        indices = np.asarray(task_bundle["feature_indices"], dtype=int)
        prediction = task_bundle["model"].predict(
            arrays.features[smoke_indices][:, indices]
        )
        inference_checks[task] = {
            "feature_count": int(len(indices)),
            "prediction_count": int(len(prediction)),
            "prediction_finite": bool(
                np.all(np.isfinite(prediction.astype(float)))
                if task == "force_fz"
                else True
            ),
        }
    force_contract = dict(bundle["force_calibration_contract"])
    gate_contract = dict(force_contract["optical_contact_gate"])
    gate_threshold = float(gate_contract["probability_threshold"])
    inactive_gate = (
        force_gate_predictions["contact_probability"].to_numpy(dtype=float)
        < gate_threshold
    )
    checks = {
        "feature_matrix_finite": bool(np.all(np.isfinite(arrays.features))),
        "forbidden_force_input_feature_count": len(forbidden_features),
        "force_mask_out_of_range_count": int(
            np.sum(
                arrays.force_mask
                & (
                    (arrays.force_fz_n < 0.0)
                    | (arrays.force_fz_n > 5.0)
                    | ~np.isfinite(arrays.force_fz_n)
                )
            )
        ),
        "formal_group_count": len(formal_group_folds),
        "expected_formal_group_count": expected_formal_group_count,
        "formal_group_multi_fold_count": len(multi_fold_groups),
        "split_group_overlap_count": int(
            split_audit["group_overlap_count"].sum()
        ),
        "blind_source_training_sample_count": int(
            np.sum(arrays.source_role == "blind_audit")
        ),
        "force_sensor_is_runtime_input": bool(
            force_contract["force_sensor_is_runtime_input"]
        ),
        "force_sensor_required_at_inference": bool(
            force_contract["force_sensor_required_at_inference"]
        ),
        "force_runtime_inputs": list(force_contract["runtime_inputs"]),
        "optical_contact_gate_enabled": bool(gate_contract["enabled"]),
        "optical_contact_gate_threshold": gate_threshold,
        "force_gate_test_sample_count": int(len(force_gate_predictions)),
        "force_gate_expected_sample_count": int(
            force_gate_metrics["test_sample_count"]
        ),
        "force_gate_probability_out_of_range_count": int(
            np.sum(
                (force_gate_predictions["contact_probability"] < 0.0)
                | (force_gate_predictions["contact_probability"] > 1.0)
                | ~np.isfinite(force_gate_predictions["contact_probability"])
            )
        ),
        "inactive_gate_nonzero_force_count": int(
            np.sum(
                np.abs(
                    force_gate_predictions.loc[
                        inactive_gate, "gated_force_n"
                    ].to_numpy(dtype=float)
                )
                > 1e-9
            )
        ),
        "force_gate_uses_grouped_evaluation": (
            force_gate_metrics["evaluation_validity"]
            == "formal_grouped_by_session_id"
        ),
        "candidate_inference": inference_checks,
        "protected_model_sha256_current": sha256_file(protected_path),
        "protected_model_sha256_expected": training_summary[
            "protected_model_sha256_before"
        ],
    }
    checks["protected_model_unchanged"] = (
        checks["protected_model_sha256_current"]
        == checks["protected_model_sha256_expected"]
    )
    ok = bool(
        checks["feature_matrix_finite"]
        and checks["forbidden_force_input_feature_count"] == 0
        and checks["force_mask_out_of_range_count"] == 0
        and checks["formal_group_count"]
        == checks["expected_formal_group_count"]
        and checks["formal_group_multi_fold_count"] == 0
        and checks["split_group_overlap_count"] == 0
        and checks["blind_source_training_sample_count"] == 0
        and not checks["force_sensor_is_runtime_input"]
        and not checks["force_sensor_required_at_inference"]
        and checks["force_runtime_inputs"] == ["optical_spectrum_time_series"]
        and checks["optical_contact_gate_enabled"]
        and checks["force_gate_test_sample_count"]
        == checks["force_gate_expected_sample_count"]
        and checks["force_gate_probability_out_of_range_count"] == 0
        and checks["inactive_gate_nonzero_force_count"] == 0
        and checks["force_gate_uses_grouped_evaluation"]
        and all(
            value["prediction_count"] == len(smoke_indices)
            and value["prediction_finite"]
            for value in inference_checks.values()
        )
        and checks["protected_model_unchanged"]
    )
    payload = {
        "schema_version": "ordinary_fbg_all_data_validation_v1",
        "checks": checks,
        "forbidden_features": forbidden_features,
        "multi_fold_groups": multi_fold_groups,
        "ok": ok,
    }
    output_path = output_dir / "validation_summary.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
