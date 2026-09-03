"""Promote an isolated High Sensitivity / 300 us candidate to Beta."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_training_20260831_high_sensitivity_300us_v1"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260831_high_sensitivity_300us_v1"
)
DEPLOYED_PATH = (
    PROJECT_ROOT / "models" / "deployed" / "ordinary_fbg_current_runtime.joblib"
)
DEPLOYMENT_RECORD_PATH = DEPLOYED_PATH.with_suffix(".deployment.json")
ROLLBACK_DIR = PROJECT_ROOT / "models" / "rollback"
RUNTIME_PAYLOAD_SCHEMA = "high_sensitivity_300us_baseline_relative_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_metric(metrics: dict, task: str) -> dict:
    selected = metrics.get("selected_candidates", {}).get(task)
    if not isinstance(selected, dict):
        raise RuntimeError(f"missing selected {task} metrics")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a validated High Sensitivity / 300 us candidate while "
            "preserving the immediately preceding runtime as a rollback."
        )
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--training-date", action="append", dest="training_dates")
    parser.add_argument(
        "--independent-validation-summary",
        type=Path,
        help="Hash-verified blind evaluation summary for this exact candidate.",
    )
    parser.add_argument(
        "--created-for",
        default="TOUCH Beta live hardware evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    candidate_path = candidate_dir / "ordinary_fbg_px6d_candidate_bundle.joblib"
    metrics_path = candidate_dir / "training_metrics.json"
    dataset_path = dataset_dir / "ordinary_fbg_px6d_dataset.npz"
    training_dates = list(args.training_dates or ["20260831"])

    for required in (candidate_path, metrics_path, dataset_path, DEPLOYED_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    candidate = joblib.load(candidate_path)
    if candidate.get("schema_version") != "ordinary_fbg_px6d_candidate_bundle_v1":
        raise RuntimeError("unexpected candidate schema")
    if candidate.get("historical_data_included") is not False:
        raise RuntimeError("candidate must remain isolated from historical data")
    feature_names = tuple(str(value) for value in candidate["feature_names"])
    if len(feature_names) != 264:
        raise RuntimeError("candidate runtime feature contract must contain 264 names")

    candidate_sha256 = _sha256(candidate_path)
    independent_validation = None
    validation_summary_path = None
    if args.independent_validation_summary is not None:
        validation_summary_path = (
            args.independent_validation_summary.expanduser().resolve()
        )
        validation_summary = json.loads(
            validation_summary_path.read_text(encoding="utf-8")
        )
        if validation_summary.get("prediction_frozen_before_answer_access") is not True:
            raise RuntimeError("independent validation was not frozen before unblinding")
        if int(validation_summary.get("training_overlap_count", -1)) != 0:
            raise RuntimeError("independent validation overlaps the training set")
        if str(validation_summary.get("model_sha256") or "") != candidate_sha256:
            raise RuntimeError("independent validation model hash mismatch")
        offline = dict(validation_summary.get("offline") or {})
        independent_validation = {
            "summary_path": str(validation_summary_path),
            "summary_sha256": _sha256(validation_summary_path),
            "session_count": int(validation_summary["session_count"]),
            "training_overlap_count": 0,
            "active_episode_accuracy": float(offline["active_episode_accuracy"]),
            "idle_session_accuracy": float(offline["idle_session_accuracy"]),
            "active_frame_position_accuracy": float(
                offline["active_frame_position_accuracy"]
            ),
            "active_frame_contact_coverage": float(
                offline["active_frame_contact_coverage"]
            ),
            "idle_frame_false_activation_rate": float(
                offline["idle_frame_false_activation_rate"]
            ),
            "force_active_frames": dict(offline["force_active_frames"]),
        }

    model_payloads: dict[str, dict] = {}
    for runtime_name, candidate_name in (
        ("contact", "contact"),
        ("position", "position"),
        ("force_fz", "force"),
    ):
        source = candidate["models"][candidate_name]
        indices = np.asarray(source["feature_indices"], dtype=int)
        if not np.array_equal(indices, np.arange(264, dtype=int)):
            raise RuntimeError(f"{candidate_name} does not use the full 264-feature view")
        estimator = source["estimator"]
        if int(getattr(estimator, "n_features_in_", -1)) != 264:
            raise RuntimeError(f"{candidate_name} estimator feature count mismatch")
        model_payloads[runtime_name] = {
            "model": estimator,
            "model_id": str(source["model_id"]),
            "feature_view": "baseline_relative_264",
        }

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    contact_metrics = _selected_metric(metrics, "contact")
    position_metrics = _selected_metric(metrics, "position")
    force_metrics = _selected_metric(metrics, "force")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        force_values = np.asarray(dataset["force_fz_n"], dtype=float)
    calibrated_force_max_n = float(np.nanmax(force_values))

    deployed = joblib.load(DEPLOYED_PATH)
    if deployed.get("schema_version") != "ordinary_fbg_optical_only_force_candidate_v2":
        raise RuntimeError("unexpected deployed runtime wrapper schema")

    source_sha256 = _sha256(DEPLOYED_PATH)
    ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
    rollback_path = (
        ROLLBACK_DIR
        / f"ordinary_fbg_runtime_before_high_sensitivity_300us_{source_sha256[:12]}.joblib"
    )
    if not rollback_path.exists():
        shutil.copy2(DEPLOYED_PATH, rollback_path)
    if _sha256(rollback_path) != source_sha256:
        raise RuntimeError("rollback artifact hash mismatch")

    deployed["literature_guided_classification"] = {
        "schema_version": RUNTIME_PAYLOAD_SCHEMA,
        "created_for": str(args.created_for),
        "static_feature_schema": "baseline_relative_full_spectrum_264",
        "static_feature_names": feature_names,
        "bin_count": 64,
        "temporal_window_frames": 2,
        "training_dates": training_dates,
        "dataset_id": candidate["dataset_id"],
        "dataset_rows": int(metrics["frame_count"]),
        "independent_sessions": int(metrics["independent_session_count"]),
        "historical_data_included": False,
        "acquisition_contract": {
            "sensor_mode": "High Sensitivity",
            "sensor_mode_code": 0,
            "integration_us": 300,
            "spectrum_points": 512,
        },
        "evaluation_validity": (
            "grouped_out_of_session_plus_hash_frozen_independent_blind_test"
            if independent_validation is not None
            else "grouped_out_of_session_candidate_pending_live_test"
        ),
        "contact": model_payloads["contact"],
        "position": model_payloads["position"],
        "force_fz": model_payloads["force_fz"],
        "validation": {
            "contact_accuracy": float(contact_metrics["accuracy"]),
            "contact_macro_f1": float(contact_metrics["macro_f1"]),
            "no_contact_false_positive_rate": float(
                contact_metrics["no_contact_false_positive_rate"]
            ),
            "active_contact_recall": float(contact_metrics["active_contact_recall"]),
            "position_accuracy": float(position_metrics["accuracy"]),
            "position_macro_f1": float(position_metrics["macro_f1"]),
            "force_mae_n": float(force_metrics["mae_n"]),
            "force_r2": float(force_metrics["r2"]),
            "force_calibration_slope": float(force_metrics["calibration_slope"]),
        },
        "deployment_status": "beta_experimental_live_validation_required",
    }
    if independent_validation is not None:
        deployed["literature_guided_classification"][
            "independent_validation"
        ] = independent_validation

    force_contract = dict(deployed.get("force_calibration_contract") or {})
    force_contract["training_range_n"] = [0.0, calibrated_force_max_n]
    force_contract["prediction_clip_range_n"] = [0.0, None]
    optical_gate = dict(force_contract.get("optical_contact_gate") or {})
    optical_gate["probability_threshold"] = 0.65
    optical_gate["no_contact_output_n"] = 0.0
    force_contract["optical_contact_gate"] = optical_gate
    deployed["force_calibration_contract"] = force_contract

    deployment_metadata = dict(deployed.get("deployment_metadata") or {})
    deployment_metadata.update(
        {
            "release_channel": "beta",
            "runtime_generation": RUNTIME_PAYLOAD_SCHEMA,
            "candidate_model_path": str(candidate_path),
            "candidate_model_sha256": candidate_sha256,
            "source_runtime_sha256": source_sha256,
            "rollback_model_path": str(rollback_path),
            "live_validation_required": True,
        }
    )
    if independent_validation is not None:
        deployment_metadata["independent_validation"] = independent_validation
    deployed["deployment_metadata"] = deployment_metadata

    temporary_path = DEPLOYED_PATH.with_suffix(".joblib.tmp")
    joblib.dump(deployed, temporary_path, compress=3)
    verified = joblib.load(temporary_path)
    if (
        verified.get("literature_guided_classification", {}).get("schema_version")
        != RUNTIME_PAYLOAD_SCHEMA
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("written runtime bundle failed schema verification")
    os.replace(temporary_path, DEPLOYED_PATH)

    deployment_record = {
        "schema_version": "touch_beta_model_deployment_record_v1",
        "runtime_payload_schema": RUNTIME_PAYLOAD_SCHEMA,
        "deployed_model_path": str(DEPLOYED_PATH),
        "deployed_model_sha256": _sha256(DEPLOYED_PATH),
        "candidate_model_sha256": candidate_sha256,
        "rollback_model_path": str(rollback_path),
        "rollback_model_sha256": _sha256(rollback_path) if rollback_path.exists() else None,
        "dataset_id": candidate["dataset_id"],
        "historical_data_included": False,
        "acquisition_contract": {
            "sensor_mode": "High Sensitivity",
            "integration_us": 300,
        },
        "live_validation_required": True,
    }
    if independent_validation is not None:
        deployment_record["independent_validation"] = independent_validation
    DEPLOYMENT_RECORD_PATH.write_text(
        json.dumps(deployment_record, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(deployment_record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
