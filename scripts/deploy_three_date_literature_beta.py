"""Deploy the audited three-date literature-guided models into TOUCH Beta.

The deployed artifact keeps the original all-source tasks as a compatibility
fallback.  The live adapter selects the v2 payload for contact, position, and
optical-only Fz inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "ordinary_fbg_three_date_literature_candidate.joblib"
)
DEFAULT_BETA_MODEL = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "ordinary_fbg_optical_only_force_candidate.joblib"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ordinary_fbg_three_date_literature_beta_deployment"
)
RELEASE_VERSION = "0.19.4-beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--beta-model", type=Path, default=DEFAULT_BETA_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-version", default=RELEASE_VERSION)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_feature_count(task: Mapping[str, Any]) -> int:
    model = task["model"]
    direct = getattr(model, "n_features_in_", None)
    if direct is not None:
        return int(direct)
    for _, step in getattr(model, "steps", ()):
        count = getattr(step, "n_features_in_", None)
        if count is not None:
            return int(count)
    return -1


def _load_training_contract(
    candidate: Mapping[str, Any],
) -> tuple[list[str], float, int, int]:
    source_datasets = candidate.get("source_datasets")
    if not isinstance(source_datasets, Mapping) or not source_datasets:
        raise ValueError("candidate source dataset contract is missing")

    reference_names: np.ndarray | None = None
    force_max_n = 0.0
    row_count = 0
    groups: set[str] = set()
    for date, raw_path in source_datasets.items():
        path = Path(str(raw_path)).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            names = np.asarray(payload["feature_names"], dtype=str)
            if names.shape != (264,):
                raise ValueError(f"{path} has an invalid static feature schema")
            if reference_names is None:
                reference_names = names
            elif not np.array_equal(reference_names, names):
                raise ValueError("training datasets use different feature schemas")

            force = np.asarray(payload["force_fz_n"], dtype=float)
            mask = np.asarray(payload["force_training_mask"], dtype=bool)
            valid = force[mask & np.isfinite(force)]
            if valid.size:
                force_max_n = max(force_max_n, float(np.max(valid)))
            row_count += int(len(force))
            groups.update(
                f"{date}::{value}"
                for value in np.asarray(payload["session_id"], dtype=str)
            )

    if reference_names is None:
        raise RuntimeError("no training feature schema was loaded")
    return reference_names.tolist(), force_max_n, row_count, len(groups)


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    if (
        candidate.get("schema_version")
        != "ordinary_fbg_three_date_literature_candidate_v1"
    ):
        raise ValueError("unsupported three-date candidate schema")
    tasks = candidate.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("candidate tasks are missing")
    expected = {
        "contact": ("response_raw136", 136),
        "position": ("response_raw136", 136),
        "force_fz": ("literature_snv_sg328", 328),
    }
    for task_name, (feature_view, feature_count) in expected.items():
        task = tasks.get(task_name)
        if not isinstance(task, Mapping):
            raise ValueError(f"candidate task {task_name!r} is missing")
        if str(task.get("feature_view")) != feature_view:
            raise ValueError(f"candidate task {task_name!r} feature view mismatch")
        if _model_feature_count(task) != feature_count:
            raise ValueError(f"candidate task {task_name!r} feature count mismatch")
        if len(task.get("feature_names", ())) != feature_count:
            raise ValueError(f"candidate task {task_name!r} names mismatch")
    if candidate["tasks"]["force_fz"].get("output_policy") != (
        "nonnegative_without_upper_clip"
    ):
        raise ValueError("candidate force output policy is unsafe")


def _runtime_payload(
    candidate: Mapping[str, Any],
    static_feature_names: list[str],
) -> dict[str, Any]:
    tasks = candidate["tasks"]
    return {
        "schema_version": "literature_guided_contact_position_force_v2",
        "created_for": "beta_three_date_deployment_20260804",
        "static_feature_schema": "baseline_relative_full_spectrum_264",
        "static_feature_names": static_feature_names,
        "bin_count": 64,
        "temporal_window_frames": 5,
        "training_dates": list(candidate["training_dates"]),
        "dataset_rows": int(candidate["dataset_rows"]),
        "independent_sessions": int(candidate["independent_sessions"]),
        "evaluation_validity": candidate["evaluation_validity"],
        "contact": {
            "model": tasks["contact"]["model"],
            "feature_view": "response_raw136",
            "feature_count": 136,
            "candidate_id": tasks["contact"]["candidate_id"],
        },
        "position": {
            "model": tasks["position"]["model"],
            "feature_view": "response_raw136",
            "feature_count": 136,
            "candidate_id": tasks["position"]["candidate_id"],
        },
        "force_fz": {
            "model": tasks["force_fz"]["model"],
            "feature_view": "literature_snv_sg328",
            "feature_count": 328,
            "candidate_id": tasks["force_fz"]["candidate_id"],
            "output_policy": "nonnegative_without_upper_clip",
        },
        "validation": candidate["validation"],
        "force_model_replaced": True,
    }


def main() -> int:
    args = parse_args()
    candidate_path = args.candidate.resolve()
    beta_model_path = args.beta_model.resolve()
    output_dir = args.output_dir.resolve()
    if not candidate_path.exists() or not beta_model_path.exists():
        raise FileNotFoundError("candidate or Beta model artifact is missing")

    candidate = joblib.load(candidate_path)
    beta_bundle = joblib.load(beta_model_path)
    if not isinstance(candidate, Mapping) or not isinstance(beta_bundle, dict):
        raise TypeError("model artifacts must contain mappings")
    _validate_candidate(candidate)
    if beta_bundle.get("schema_version") != (
        "ordinary_fbg_optical_only_force_candidate_v2"
    ):
        raise ValueError("unsupported Beta base model schema")

    static_names, force_max_n, source_rows, source_groups = (
        _load_training_contract(candidate)
    )
    if source_rows != int(candidate["dataset_rows"]):
        raise ValueError("candidate row count does not match source datasets")
    if source_groups != int(candidate["independent_sessions"]):
        raise ValueError("candidate session count does not match source datasets")

    archive_dir = beta_model_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / (
        "ordinary_fbg_optical_only_force_candidate_pre_three_date_"
        "20260804_0_19_3_beta.joblib"
    )
    if not archive_path.exists():
        shutil.copy2(beta_model_path, archive_path)

    beta_bundle["literature_guided_classification"] = _runtime_payload(
        candidate,
        static_names,
    )
    force_contract = beta_bundle.setdefault("force_calibration_contract", {})
    force_contract["training_range_n"] = [0.0, float(force_max_n)]
    force_contract["prediction_clip_range_n"] = [0.0, None]
    force_contract["prediction_upper_clip_applied"] = False
    force_contract["runtime_inputs"] = ["optical_spectrum_time_series"]
    force_contract["force_sensor_is_runtime_input"] = False
    force_contract["force_sensor_required_at_inference"] = False

    deployment = beta_bundle.setdefault("deployment_metadata", {})
    deployment.update(
        {
            "release_channel": "beta",
            "release_version": str(args.release_version),
            "build_id": "beta-unified-operator-frame-v19-4-20260804",
            "deployed_at_local": datetime.now().astimezone().isoformat(),
            "source_output": (
                "ordinary_fbg_literature_guided_cross_date_20260731_20260804"
            ),
            "source_sha256": _sha256(candidate_path),
            "training_sample_count": int(candidate["dataset_rows"]),
            "training_group_count": int(candidate["independent_sessions"]),
            "training_dates": list(candidate["training_dates"]),
            "runtime_input": "optical_spectrum_time_series_only",
            "force_sensor_required_at_inference": False,
            "classification_upgrade": "literature_guided_three_date_v2",
            "force_upgrade": "literature_guided_three_date_osc_ridge_v1",
            "force_prediction_upper_clip": False,
        }
    )

    temporary_path = beta_model_path.with_suffix(".joblib.deploying")
    joblib.dump(beta_bundle, temporary_path, compress=0)
    verified = joblib.load(temporary_path)
    payload = verified.get("literature_guided_classification", {})
    if payload.get("schema_version") != (
        "literature_guided_contact_position_force_v2"
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("serialized Beta payload verification failed")
    _validate_candidate(
        {
            **candidate,
            "tasks": {
                "contact": {
                    **candidate["tasks"]["contact"],
                    "model": payload["contact"]["model"],
                },
                "position": {
                    **candidate["tasks"]["position"],
                    "model": payload["position"]["model"],
                },
                "force_fz": {
                    **candidate["tasks"]["force_fz"],
                    "model": payload["force_fz"]["model"],
                },
            },
        }
    )
    os.replace(temporary_path, beta_model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "deployed_and_serialization_verified",
        "release_version": str(args.release_version),
        "model_path": str(beta_model_path),
        "model_sha256": _sha256(beta_model_path),
        "model_size_mb": beta_model_path.stat().st_size / (1024.0 * 1024.0),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "archive_path": str(archive_path),
        "training_dates": list(candidate["training_dates"]),
        "dataset_rows": int(candidate["dataset_rows"]),
        "independent_sessions": int(candidate["independent_sessions"]),
        "training_force_range_n": [0.0, float(force_max_n)],
        "force_prediction_upper_clip": False,
        "validation": candidate["validation"],
    }
    (output_dir / "deployment_summary.json").write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "deployment_report.md").write_text(
        "# Three-date literature-guided Beta deployment\n\n"
        "The audited 20260731, 20260803, and 20260804 models were deployed "
        "to the Beta runtime after serialization verification. Contact and "
        "position use the 136-feature absolute response view; optical Fz uses "
        "the 328-feature SNV/Savitzky-Golay/OSC Ridge path. The existing "
        "release and baseline gates remain active.\n\n"
        f"- Rows: {int(candidate['dataset_rows']):,}\n"
        f"- Independent sessions: {int(candidate['independent_sessions'])}\n"
        f"- Observed training force range: 0-{force_max_n:.3f} N\n"
        "- Force output: nonnegative, no artificial upper clip\n"
        "- Force sensor at runtime: not required\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
