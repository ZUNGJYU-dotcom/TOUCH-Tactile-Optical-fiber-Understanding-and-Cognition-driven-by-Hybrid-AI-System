"""Promote or roll back the validated same-day joint nine-FBG Beta runtime."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOYED_PATH = ROOT / "models/deployed/ordinary_fbg_current_runtime.joblib"
DEPLOYMENT_RECORD_PATH = DEPLOYED_PATH.with_suffix(".deployment.json")
ROLLBACK_DIR = ROOT / "models/rollback"
DEFAULT_CANDIDATE = (
    ROOT
    / "outputs/ordinary_fbg_runtime_stage_20260902_same_day_joint_v4_fast"
    / "ordinary_fbg_runtime_candidate_v4_fast.joblib"
)
DEFAULT_AUDIT = (
    ROOT
    / "outputs/ordinary_fbg_runtime_replay_20260902_same_day_joint_v4_fast_position_combined"
    / "metrics.json"
)
EXPECTED_WRAPPER_SCHEMA = "ordinary_fbg_optical_only_force_candidate_v2"
EXPECTED_RUNTIME_SCHEMA = "same_day_joint_nine_fbg_v4"
EXPECTED_DATASET_ID = "ordinary_fbg_20260902_same_day_joint_fingerprint_v2"
RELEASE_VERSION = "0.19.24-beta"
BUILD_ID = "beta-20260903-same-day-joint-nine-fbg-v4-fast"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_verified(source: Path, target: Path, expected_hash: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != expected_hash:
            raise RuntimeError(f"existing rollback artifact hash mismatch: {target}")
        return
    temporary = target.with_name(f"{target.name}.tmp")
    shutil.copy2(source, temporary)
    if _sha256(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"rollback copy hash mismatch: {target}")
    os.replace(temporary, target)


def _replace_from_verified_copy(
    source: Path,
    target: Path,
    expected_hash: str,
) -> None:
    temporary = target.with_name(f"{target.name}.restore.tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    if _sha256(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"restore copy hash mismatch: {target}")
    os.replace(temporary, target)


def _model_feature_count(model: Any) -> int:
    value = getattr(model, "n_features_in_", None)
    return int(value) if value is not None else -1


def _strip_legacy_runtime_models(
    wrapper: dict[str, Any],
    runtime: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Convert the staged compatibility wrapper into a current-only Beta bundle."""

    wrapper.pop("tasks", None)
    wrapper.pop("task_metadata", None)
    wrapper.pop("all_feature_names", None)
    wrapper["runtime_model_isolation"] = {
        "mode": "current_only",
        "active_runtime_schema": EXPECTED_RUNTIME_SCHEMA,
        "active_dataset_id": EXPECTED_DATASET_ID,
        "active_task_count": 3,
        "legacy_task_model_count": 0,
        "legacy_fallback_enabled": False,
    }
    wrapper["runtime_inference_policy"] = {
        "contact": "active_payload.contact",
        "position": "active_payload.position",
        "force_fz": "active_payload.force_fz",
        "single_channel_evidence_sufficient": False,
        "legacy_fallback_enabled": False,
        "force_sensor_is_runtime_input": False,
    }
    wrapper["data_contract"] = {
        "dataset_id": EXPECTED_DATASET_ID,
        "source_batches": list(audit["source_batches"]),
        "excluded_source_batches": list(
            runtime.get("excluded_source_batches") or ["blind2"]
        ),
        "historical_data_included": False,
        "independent_blind_evidence": False,
    }


def _validate_candidate(
    candidate_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    stage_path = candidate_path.with_suffix(".stage.json")
    for required in (candidate_path, stage_path, audit_path, DEPLOYED_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)

    candidate_hash = _sha256(candidate_path)
    stage = _load_json(stage_path)
    audit = _load_json(audit_path)
    if stage.get("deployed") is not False:
        raise RuntimeError("candidate stage record is not in the expected staged state")
    if str(stage.get("output_sha256") or "").lower() != candidate_hash:
        raise RuntimeError("candidate hash does not match its stage record")
    if str(stage.get("runtime_schema") or "") != EXPECTED_RUNTIME_SCHEMA:
        raise RuntimeError("candidate stage runtime schema is not supported")
    if str(stage.get("dataset_id") or "") != EXPECTED_DATASET_ID:
        raise RuntimeError("candidate stage dataset is not supported")
    if str(audit.get("candidate_model_sha256") or "").lower() != candidate_hash:
        raise RuntimeError("exact replay was not run against this candidate hash")
    if str(audit.get("runtime_schema") or "") != EXPECTED_RUNTIME_SCHEMA:
        raise RuntimeError("exact replay runtime schema mismatch")
    if str(audit.get("dataset_id") or "") != EXPECTED_DATASET_ID:
        raise RuntimeError("exact replay dataset mismatch")
    if audit.get("candidate_status") != "passed_same_day_exact_replay_not_deployed":
        raise RuntimeError("candidate has not passed the exact runtime replay")

    acceptance = dict(audit.get("acceptance") or {})
    required_acceptance = (
        "same_day_dedicated_idle_zero_false_triggers",
        "same_day_all_labeled_no_contact_zero_false_triggers",
        "same_day_zero_wrong_emitted_position_labels",
        "same_day_raw_joint_position_all_correct",
        "same_day_all_single_position_sessions_correct",
        "same_day_meaningful_contact_p90_delay_below_250_ms",
        "same_day_position_p90_delay_below_350_ms",
        "complete_batch_holdout_idle_zero_false_triggers",
    )
    failed = [name for name in required_acceptance if acceptance.get(name) is not True]
    if failed:
        raise RuntimeError("candidate failed deployment gates: " + ", ".join(failed))

    wrapper = joblib.load(candidate_path)
    if wrapper.get("schema_version") != EXPECTED_WRAPPER_SCHEMA:
        raise RuntimeError("unexpected runtime wrapper schema")
    runtime = wrapper.get("literature_guided_classification")
    if not isinstance(runtime, dict):
        raise RuntimeError("candidate is missing its optical runtime payload")
    if runtime.get("schema_version") != EXPECTED_RUNTIME_SCHEMA:
        raise RuntimeError("candidate runtime payload schema mismatch")
    if runtime.get("dataset_id") != EXPECTED_DATASET_ID:
        raise RuntimeError("candidate runtime payload dataset mismatch")

    feature_names = tuple(str(value) for value in runtime.get("static_feature_names", ()))
    if len(feature_names) != 339 or len(set(feature_names)) != 339:
        raise RuntimeError("candidate must expose 339 unique joint feature names")
    expected_tasks = {
        "contact": ("baseline_relative_264_plus_nine_fbg_joint_339", 339),
        "position": ("nine_fbg_joint_75", 75),
        "force_fz": ("baseline_relative_264_plus_nine_fbg_joint_339", 339),
    }
    for task_name, (expected_view, expected_count) in expected_tasks.items():
        task = runtime.get(task_name)
        if not isinstance(task, dict):
            raise RuntimeError(f"candidate is missing task {task_name}")
        if str(task.get("feature_view") or "") != expected_view:
            raise RuntimeError(f"candidate {task_name} feature view mismatch")
        indices = np.asarray(task.get("feature_indices"), dtype=int)
        names = tuple(str(value) for value in task.get("feature_names", ()))
        if len(indices) != expected_count or len(names) != expected_count:
            raise RuntimeError(f"candidate {task_name} feature count mismatch")
        if _model_feature_count(task.get("model")) != expected_count:
            raise RuntimeError(f"candidate {task_name} estimator contract mismatch")

    return wrapper, runtime, audit, candidate_hash


def _archive_current_runtime() -> tuple[Path, Path | None, str]:
    current_hash = _sha256(DEPLOYED_PATH)
    ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
    rollback_model = (
        ROLLBACK_DIR
        / f"ordinary_fbg_runtime_before_same_day_joint_v4_{current_hash[:12]}.joblib"
    )
    _copy_verified(DEPLOYED_PATH, rollback_model, current_hash)

    rollback_record: Path | None = None
    if DEPLOYMENT_RECORD_PATH.is_file():
        record_hash = _sha256(DEPLOYMENT_RECORD_PATH)
        rollback_record = (
            ROLLBACK_DIR
            / f"ordinary_fbg_runtime_before_same_day_joint_v4_{current_hash[:12]}.deployment.json"
        )
        _copy_verified(DEPLOYMENT_RECORD_PATH, rollback_record, record_hash)
    return rollback_model, rollback_record, current_hash


def _promote(candidate_path: Path, audit_path: Path) -> dict[str, Any]:
    wrapper, runtime, audit, candidate_hash = _validate_candidate(
        candidate_path,
        audit_path,
    )
    rollback_model, rollback_record, previous_hash = _archive_current_runtime()
    deployed_at = datetime.now().astimezone().isoformat(timespec="seconds")

    _strip_legacy_runtime_models(wrapper, runtime, audit)
    runtime["created_for"] = "TOUCH Beta same-day joint nine-FBG live evaluation"
    runtime["deployment_status"] = "beta_deployed_live_validation_required"
    wrapper["created_for"] = "TOUCH Beta optical-only ordinary-FBG runtime"
    wrapper["deployment_metadata"] = {
        "release_channel": "beta",
        "release_version": RELEASE_VERSION,
        "build_id": BUILD_ID,
        "deployed_at_local": deployed_at,
        "runtime_generation": EXPECTED_RUNTIME_SCHEMA,
        "dataset_id": EXPECTED_DATASET_ID,
        "training_sample_count": int(runtime["dataset_rows"]),
        "training_group_count": int(runtime["independent_sessions"]),
        "training_dates": list(runtime.get("training_dates") or ["20260902"]),
        "runtime_input": "optical_spectrum_only",
        "force_sensor_required_at_inference": False,
        "staged_runtime_path": str(candidate_path),
        "staged_runtime_sha256": candidate_hash,
        "exact_runtime_audit_path": str(audit_path),
        "exact_runtime_audit_sha256": _sha256(audit_path),
        "previous_runtime_sha256": previous_hash,
        "rollback_model_path": str(rollback_model),
        "rollback_deployment_record_path": (
            str(rollback_record) if rollback_record is not None else None
        ),
        "live_validation_required": True,
        "deployed": True,
    }

    model_tmp = DEPLOYED_PATH.with_name(f"{DEPLOYED_PATH.name}.tmp")
    model_tmp.unlink(missing_ok=True)
    joblib.dump(wrapper, model_tmp, compress=3)
    verified = joblib.load(model_tmp)
    verified_runtime = verified.get("literature_guided_classification") or {}
    if verified_runtime.get("schema_version") != EXPECTED_RUNTIME_SCHEMA:
        model_tmp.unlink(missing_ok=True)
        raise RuntimeError("written runtime failed schema verification")
    isolation = verified.get("runtime_model_isolation") or {}
    if (
        any(key in verified for key in ("tasks", "task_metadata", "all_feature_names"))
        or isolation.get("legacy_fallback_enabled") is not False
        or isolation.get("legacy_task_model_count") != 0
    ):
        model_tmp.unlink(missing_ok=True)
        raise RuntimeError("written runtime still contains a legacy model path")
    deployed_hash = _sha256(model_tmp)

    exact = dict(audit["exact_runtime_replay"])
    record = {
        "schema_version": "touch_beta_model_deployment_record_v1",
        "runtime_payload_schema": EXPECTED_RUNTIME_SCHEMA,
        "release_version": RELEASE_VERSION,
        "build_id": BUILD_ID,
        "deployed_at_local": deployed_at,
        "deployed_model_path": str(DEPLOYED_PATH),
        "deployed_model_sha256": deployed_hash,
        "candidate_model_path": str(candidate_path),
        "candidate_model_sha256": candidate_hash,
        "dataset_id": EXPECTED_DATASET_ID,
        "historical_data_included": False,
        "source_batches": list(audit["source_batches"]),
        "acquisition_contract": {
            "sensor_mode": "High Sensitivity",
            "integration_us": 300,
        },
        "feature_contract": dict(audit["feature_contract"]),
        "validation_scope": (
            "same_day_exact_replay_plus_leave_complete_acquisition_batch_out"
        ),
        "independent_blind_evidence": False,
        "exact_runtime_validation": {
            "session_count": int(audit["session_count"]),
            "frame_count": int(audit["frame_count"]),
            "dedicated_idle_frames": int(exact["dedicated_idle_frame_count"]),
            "dedicated_idle_false_positive_rate": float(
                exact["dedicated_idle_false_positive_rate"]
            ),
            "active_contact_recall": float(exact["active_contact_recall"]),
            "wrong_emitted_position_frames": int(
                exact["formal_wrong_position_label_frames"]
            ),
            "emitted_position_accuracy": float(
                exact["formal_emitted_position_accuracy"]
            ),
            "contact_p90_delay_ms": float(
                exact["contact_episodes"]["meaningful_first_detection_delay_ms"]["p90"]
            ),
            "position_p90_delay_ms": float(
                exact["position_episodes"]["first_correct_position_delay_ms"]["p90"]
            ),
        },
        "rollback_model_path": str(rollback_model),
        "rollback_model_sha256": previous_hash,
        "rollback_deployment_record_path": (
            str(rollback_record) if rollback_record is not None else None
        ),
        "rollback_command": (
            f'"{sys.executable}" "{Path(__file__).resolve()}" rollback'
        ),
        "live_validation_required": True,
    }
    record_tmp = DEPLOYMENT_RECORD_PATH.with_name(
        f"{DEPLOYMENT_RECORD_PATH.name}.tmp"
    )
    record_tmp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        os.replace(model_tmp, DEPLOYED_PATH)
        os.replace(record_tmp, DEPLOYMENT_RECORD_PATH)
    except BaseException:
        model_tmp.unlink(missing_ok=True)
        record_tmp.unlink(missing_ok=True)
        _replace_from_verified_copy(rollback_model, DEPLOYED_PATH, previous_hash)
        if rollback_record is not None:
            _replace_from_verified_copy(
                rollback_record,
                DEPLOYMENT_RECORD_PATH,
                _sha256(rollback_record),
            )
        raise

    if _sha256(DEPLOYED_PATH) != deployed_hash:
        raise RuntimeError("deployed runtime hash changed after atomic replacement")
    return record


def _rollback() -> dict[str, Any]:
    if not DEPLOYMENT_RECORD_PATH.is_file():
        raise FileNotFoundError(DEPLOYMENT_RECORD_PATH)
    active_record = _load_json(DEPLOYMENT_RECORD_PATH)
    rollback_model = Path(str(active_record["rollback_model_path"])).resolve()
    rollback_hash = str(active_record["rollback_model_sha256"]).lower()
    rollback_record_value = active_record.get("rollback_deployment_record_path")
    rollback_record = (
        Path(str(rollback_record_value)).resolve()
        if rollback_record_value
        else None
    )
    if not rollback_model.is_file() or _sha256(rollback_model) != rollback_hash:
        raise RuntimeError("rollback model is missing or failed hash verification")

    active_hash = _sha256(DEPLOYED_PATH)
    failed_archive = (
        ROLLBACK_DIR
        / f"ordinary_fbg_runtime_replaced_same_day_joint_v4_{active_hash[:12]}.joblib"
    )
    _copy_verified(DEPLOYED_PATH, failed_archive, active_hash)
    active_record_archive = (
        ROLLBACK_DIR
        / f"ordinary_fbg_runtime_replaced_same_day_joint_v4_{active_hash[:12]}.deployment.json"
    )
    _copy_verified(
        DEPLOYMENT_RECORD_PATH,
        active_record_archive,
        _sha256(DEPLOYMENT_RECORD_PATH),
    )

    _replace_from_verified_copy(rollback_model, DEPLOYED_PATH, rollback_hash)
    if rollback_record is not None and rollback_record.is_file():
        _replace_from_verified_copy(
            rollback_record,
            DEPLOYMENT_RECORD_PATH,
            _sha256(rollback_record),
        )
    else:
        _write_json_atomic(
            DEPLOYMENT_RECORD_PATH,
            {
                "schema_version": "touch_beta_model_deployment_record_v1",
                "deployed_model_path": str(DEPLOYED_PATH),
                "deployed_model_sha256": rollback_hash,
                "rollback_recovered_at_local": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            },
        )
    receipt = {
        "status": "rolled_back",
        "restored_model_path": str(DEPLOYED_PATH),
        "restored_model_sha256": rollback_hash,
        "replaced_runtime_archive": str(failed_archive),
        "replaced_runtime_sha256": active_hash,
    }
    _write_json_atomic(ROLLBACK_DIR / "last_same_day_joint_rollback.json", receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "promote"):
        child = subparsers.add_parser(command)
        child.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
        child.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    subparsers.add_parser("rollback")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "rollback":
        result = _rollback()
    else:
        candidate = args.candidate.expanduser().resolve()
        audit = args.audit.expanduser().resolve()
        _, _, _, candidate_hash = _validate_candidate(candidate, audit)
        if args.command == "verify":
            result = {
                "status": "verified_not_deployed",
                "candidate_path": str(candidate),
                "candidate_sha256": candidate_hash,
                "audit_path": str(audit),
            }
        else:
            result = _promote(candidate, audit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
