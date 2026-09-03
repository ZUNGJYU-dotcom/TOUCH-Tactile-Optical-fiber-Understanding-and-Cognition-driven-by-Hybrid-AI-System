"""Stage, but do not deploy, the same-day nine-FBG runtime candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.all_source_runtime_adapter import (  # noqa: E402
    JOINT_NINE_FBG_RUNTIME_SCHEMA,
)


DEFAULT_TEMPLATE = ROOT / "models/deployed/ordinary_fbg_current_runtime.joblib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_feature_count(model: Any) -> int:
    value = getattr(model, "n_features_in_", None)
    return int(value) if value is not None else -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("stress_dir", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_dir = args.candidate_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    stress_dir = args.stress_dir.resolve()
    template_path = args.template.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite staged runtime: {output_path}")

    candidate_path = candidate_dir / "ordinary_fbg_px6d_candidate_bundle.joblib"
    metrics_path = candidate_dir / "training_metrics.json"
    dataset_path = dataset_dir / "ordinary_fbg_px6d_dataset.npz"
    manifest_path = dataset_dir / "ordinary_fbg_px6d_dataset_manifest.json"
    stress_path = stress_dir / "stress_metrics.json"
    for path in (
        candidate_path,
        metrics_path,
        dataset_path,
        manifest_path,
        stress_path,
        template_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    candidate = joblib.load(candidate_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    if candidate.get("schema_version") != "ordinary_fbg_px6d_candidate_bundle_v1":
        raise RuntimeError("unexpected candidate bundle schema")
    if candidate.get("dataset_id") != manifest.get("dataset_id"):
        raise RuntimeError("candidate and dataset IDs do not match")
    if stress.get("dataset_id") != manifest.get("dataset_id"):
        raise RuntimeError("stress audit and dataset IDs do not match")
    if stress.get("blind2_excluded") is not True:
        raise RuntimeError("Blind2 exclusion is not proven")
    if stress.get("random_frame_split_used") is not False:
        raise RuntimeError("random frame splitting is not allowed")

    stress_contact = stress["contact"]
    stress_position = stress["position"]
    if float(stress_contact["runtime_dedicated_idle_false_positive_rate"]) != 0.0:
        raise RuntimeError("staged runtime requires zero dedicated-idle false positives")
    if int(stress_contact["runtime_dedicated_idle_false_activation_episodes"]) != 0:
        raise RuntimeError("staged runtime requires zero idle false-activation episodes")
    if int(stress_contact["meaningful_episode_detected_count"]) != int(
        stress_contact["meaningful_episode_count"]
    ):
        raise RuntimeError("not every meaningful contact episode was detected")
    if float(stress_position["single_label_session_accuracy"]) != 1.0:
        raise RuntimeError("single-label session position accuracy must be 100%")
    if float(stress_position["runtime_frame_accuracy"]) < 0.99:
        raise RuntimeError("runtime position frame accuracy is below 99%")

    feature_names = tuple(str(value) for value in candidate["feature_names"])
    if len(feature_names) != 339 or len(set(feature_names)) != 339:
        raise RuntimeError("joint runtime requires 339 unique feature names")
    expected_views = {
        "contact": "baseline_relative_264_plus_nine_fbg_joint_339",
        "position": "nine_fbg_joint_75",
        "force": "baseline_relative_264_plus_nine_fbg_joint_339",
    }
    runtime_names = {"contact": "contact", "position": "position", "force": "force_fz"}
    tasks: dict[str, dict[str, Any]] = {}
    for candidate_name, runtime_name in runtime_names.items():
        source = candidate["models"][candidate_name]
        view = str(source["feature_set"])
        if view != expected_views[candidate_name]:
            raise RuntimeError(
                f"selected {candidate_name} view is {view}, expected "
                f"{expected_views[candidate_name]}"
            )
        indices = np.asarray(source["feature_indices"], dtype=int)
        if indices.ndim != 1 or len(indices) == 0:
            raise RuntimeError(f"selected {candidate_name} indices are invalid")
        if np.any(indices < 0) or np.any(indices >= len(feature_names)):
            raise RuntimeError(f"selected {candidate_name} indices are out of range")
        estimator = source["estimator"]
        if _model_feature_count(estimator) != len(indices):
            raise RuntimeError(f"selected {candidate_name} model contract mismatch")
        tasks[runtime_name] = {
            "model": estimator,
            "model_id": str(source["model_id"]),
            "feature_view": view,
            "feature_indices": indices,
            "feature_names": tuple(feature_names[index] for index in indices),
        }

    with np.load(dataset_path, allow_pickle=False) as dataset:
        calibrated_force_max_n = float(np.max(dataset["force_fz_n"]))

    wrapper = joblib.load(template_path)
    if wrapper.get("schema_version") != "ordinary_fbg_optical_only_force_candidate_v2":
        raise RuntimeError("unexpected runtime template schema")
    wrapper["literature_guided_classification"] = {
        "schema_version": JOINT_NINE_FBG_RUNTIME_SCHEMA,
        "created_for": "isolated same-day nine-FBG runtime replay candidate",
        "static_feature_schema": "baseline_relative_full_spectrum_plus_joint_nine_fbg_339",
        "static_feature_names": feature_names,
        "bin_count": 64,
        "temporal_window_frames": 2,
        "training_dates": ["20260902"],
        "dataset_id": manifest["dataset_id"],
        "dataset_rows": int(manifest["frame_count"]),
        "independent_sessions": int(manifest["session_count"]),
        "historical_data_included": False,
        "excluded_source_batches": ["blind2"],
        "baseline_contract": {
            "training_mode": "session_initial_force_confirmed_quiet_median",
            "runtime_mode": "current_session_settled_multi_frame_median",
            "minimum_training_frames": int(
                min(row["baseline_frame_count"] for row in manifest["sessions"])
            ),
        },
        "acquisition_contract": {
            "sensor_mode": "High Sensitivity",
            "sensor_mode_code": 0,
            "integration_us": 300,
            "spectrum_points": 512,
        },
        "runtime_decision_contract": {
            "contact_probability_on": 0.80,
            "contact_probability_off": 0.50,
            "contact_release_confirmation_frames": 2,
            "joint_fast_contact_probability_on": 0.90,
            "joint_fast_position_confidence_min": 0.85,
            "joint_fast_position_margin_min": 0.75,
            "joint_fast_minimum_baseline_distance": 0.006,
            "position_switch_probability": 0.50,
            "position_margin_min": 0.0,
            "position_probability_ema_alpha": 1.0,
            "position_switch_confirmation_frames": 2,
        },
        "evaluation_validity": "leave_complete_acquisition_batch_out_plus_cross_batch_shuffle",
        "contact": tasks["contact"],
        "position": tasks["position"],
        "force_fz": tasks["force_fz"],
        "validation": {
            "runtime_dedicated_idle_false_positive_rate": float(
                stress_contact["runtime_dedicated_idle_false_positive_rate"]
            ),
            "runtime_dedicated_idle_false_activation_episodes": int(
                stress_contact["runtime_dedicated_idle_false_activation_episodes"]
            ),
            "meaningful_contact_episode_detection_rate": float(
                stress_contact["meaningful_episode_detected_count"]
                / stress_contact["meaningful_episode_count"]
            ),
            "runtime_position_frame_accuracy": float(
                stress_position["runtime_frame_accuracy"]
            ),
            "runtime_position_macro_f1": float(
                stress_position["runtime_macro_f1"]
            ),
            "single_label_session_accuracy": float(
                stress_position["single_label_session_accuracy"]
            ),
            "force_mae_n": float(
                metrics["selected_candidates"]["force"]["mae_n"]
            ),
            "force_r2": float(metrics["selected_candidates"]["force"]["r2"]),
        },
        "deployment_status": "candidate_staged_not_deployed_pending_exact_runtime_replay",
    }

    force_contract = dict(wrapper.get("force_calibration_contract") or {})
    force_contract["training_range_n"] = [0.0, calibrated_force_max_n]
    force_contract["prediction_clip_range_n"] = [0.0, None]
    optical_gate = dict(force_contract.get("optical_contact_gate") or {})
    optical_gate["probability_threshold"] = 0.80
    optical_gate["no_contact_output_n"] = 0.0
    force_contract["optical_contact_gate"] = optical_gate
    wrapper["force_calibration_contract"] = force_contract

    metadata = dict(wrapper.get("deployment_metadata") or {})
    metadata.update(
        {
            "release_channel": "staged_beta_candidate",
            "runtime_generation": JOINT_NINE_FBG_RUNTIME_SCHEMA,
            "candidate_model_path": str(candidate_path),
            "candidate_model_sha256": _sha256(candidate_path),
            "dataset_sha256": _sha256(dataset_path),
            "stress_metrics_sha256": _sha256(stress_path),
            "template_runtime_sha256": _sha256(template_path),
            "live_validation_required": True,
            "deployed": False,
        }
    )
    wrapper["deployment_metadata"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, output_path, compress=3)
    verified = joblib.load(output_path)
    payload = verified.get("literature_guided_classification") or {}
    if payload.get("schema_version") != JOINT_NINE_FBG_RUNTIME_SCHEMA:
        raise RuntimeError("staged runtime schema verification failed")
    record = {
        "schema_version": "touch_same_day_joint_runtime_stage_v1",
        "deployed": False,
        "runtime_schema": JOINT_NINE_FBG_RUNTIME_SCHEMA,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "dataset_id": manifest["dataset_id"],
        "feature_count": len(feature_names),
        "task_feature_counts": {
            name: len(task["feature_indices"]) for name, task in tasks.items()
        },
        "stress_metrics_path": str(stress_path),
    }
    output_path.with_suffix(".stage.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
