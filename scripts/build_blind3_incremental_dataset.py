"""Append unblinded Blind3 sessions to the strict ordinary-FBG dataset.

Blind3 was evaluated before its answer folders were opened, but its file naming
was not obfuscated.  It is therefore useful labelled training data, not an
independent blind benchmark.  This builder keeps that provenance explicit and
never modifies either source dataset.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.baseline_relative_features import (  # noqa: E402
    extract_baseline_relative_features,
)
from src.hybrid_spectrum.px6d_session_dataset import POSITION_ORDER  # noqa: E402


DATASET_FILENAME = "ordinary_fbg_px6d_dataset.npz"
MANIFEST_FILENAME = "ordinary_fbg_px6d_dataset_manifest.json"
FRAME_MANIFEST_FILENAME = "ordinary_fbg_px6d_frame_manifest.csv"
SOURCE_INVENTORY_FILENAME = "ordinary_fbg_px6d_source_inventory.json"
EXPECTED_AUDIT_SCHEMA = "touch_blind_multiepisode_record_audit_v1"
EXPECTED_FEATURE_COUNT = 264
NO_CONTACT_MAX_FORCE_N = 0.03
CONTACT_MIN_FORCE_N = 0.10
POSITION_MIN_FORCE_N = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contact_targets(
    force_fz_n: np.ndarray,
    *,
    active_session: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create conservative contact labels while retaining true idle drift."""

    force = np.asarray(force_fz_n, dtype=float)
    target = np.full(force.shape, -1, dtype=np.int8)
    finite = np.isfinite(force)
    if not active_session:
        target[finite] = 0
    else:
        target[finite & (force <= NO_CONTACT_MAX_FORCE_N)] = 0
        target[finite & (force >= CONTACT_MIN_FORCE_N)] = 1
    return target, target >= 0


def _fold_for_session(
    *,
    ordered_labels: tuple[str, ...],
    idle_index: int,
) -> int:
    """Spread complete sessions across five fixed folds without frame leakage."""

    if len(ordered_labels) > 1:
        sequence_key = tuple(ordered_labels)
        first_sequence = tuple(POSITION_ORDER)
        return 0 if sequence_key == first_sequence else 1
    if len(ordered_labels) == 1:
        return 2 + POSITION_ORDER.index(ordered_labels[0]) % 3
    return 2 + idle_index % 3


def _load_audited_labels(audit_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = audit_dir / "evaluation_summary.json"
    frames_path = audit_dir / "frame_comparison.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != EXPECTED_AUDIT_SCHEMA:
        raise ValueError("unexpected Blind3 audit schema")
    if summary.get("prediction_frozen_before_answer_access") is not True:
        raise ValueError("Blind3 predictions were not proven frozen before unblinding")
    expected_hash = str((summary.get("outputs") or {}).get(frames_path.name) or "")
    actual_hash = _sha256(frames_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("Blind3 audited frame table hash mismatch")

    frames = pd.read_csv(frames_path, encoding="utf-8-sig")
    required = {
        "session_id",
        "capture_index",
        "elapsed_time_sec",
        "reference_force_fz_n",
        "expected_position",
        "session_order",
    }
    missing = sorted(required - set(frames.columns))
    if missing:
        raise ValueError(f"Blind3 audit lacks required columns: {missing}")
    if frames.duplicated(["session_id", "capture_index"]).any():
        raise ValueError("Blind3 audit has duplicate session/capture keys")
    labels = set(frames["expected_position"].fillna("none").astype(str))
    unknown = sorted(labels - {"none", *POSITION_ORDER})
    if unknown:
        raise ValueError(f"Blind3 audit has unknown expected positions: {unknown}")
    return frames, summary


def _answer_labels(summary: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for folder_name in summary.get("answer_folder_names") or []:
        if "-" not in folder_name:
            raise ValueError(f"answer folder lacks suffix: {folder_name}")
        session_id, suffix = folder_name.rsplit("-", 1)
        if suffix.lower() == "nocontact":
            labels: tuple[str, ...] = ()
        else:
            labels = tuple(token.upper() for token in suffix.split("_"))
            if any(label not in POSITION_ORDER for label in labels):
                raise ValueError(f"answer folder has unknown position: {folder_name}")
        mapping[session_id] = labels
    return mapping


def _load_session_spectra(
    session_dir: Path,
    expected_capture_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    jsonl_path = session_dir / "synchronized_frames.jsonl"
    frames: dict[int, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            frames[int(row["capture_index"])] = row

    expected = [int(value) for value in expected_capture_indices]
    if set(frames) != set(expected):
        missing = sorted(set(expected) - set(frames))
        extra = sorted(set(frames) - set(expected))
        raise ValueError(
            f"{session_dir.name}: raw/audit capture mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    intensities: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    wavelengths: list[np.ndarray] = []
    trial_id = ""
    for capture_index in expected:
        row = frames[capture_index]
        spectrum = row.get("spectrum") or {}
        intensities.append(np.asarray(spectrum.get("intensity_counts"), dtype=float))
        baselines.append(
            np.asarray(spectrum.get("baseline_intensity_counts"), dtype=float)
        )
        wavelengths.append(np.asarray(spectrum.get("wavelength_nm"), dtype=float))
        trial_id = str(row.get("trial_id") or trial_id)

    intensity = np.vstack(intensities)
    baseline_matrix = np.vstack(baselines)
    wavelength_matrix = np.vstack(wavelengths)
    if intensity.shape[1] != 512:
        raise ValueError(f"{session_dir.name}: expected 512 spectrum points")
    if not np.all(np.isfinite(intensity)):
        raise ValueError(f"{session_dir.name}: non-finite spectrum intensity")
    if not np.allclose(baseline_matrix, baseline_matrix[0], rtol=0.0, atol=0.0):
        raise ValueError(f"{session_dir.name}: runtime baseline changed within session")
    if not np.allclose(wavelength_matrix, wavelength_matrix[0], rtol=0.0, atol=1e-9):
        raise ValueError(f"{session_dir.name}: wavelength axis changed within session")
    return intensity, baseline_matrix[0], wavelength_matrix[0], trial_id


def _session_rows(
    *,
    blind_root: Path,
    audited_frames: pd.DataFrame,
    answers: dict[str, tuple[str, ...]],
    base_wavelength_nm: np.ndarray,
    base_feature_names: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "force_fz_n": [],
        "contact_target": [],
        "position_target": [],
        "session_id": [],
        "trial_id": [],
        "capture_index": [],
        "elapsed_time_sec": [],
        "fold_id": [],
        "force_training_mask": [],
        "contact_training_mask": [],
        "position_training_mask": [],
        "release_tail_excluded": [],
    }
    session_manifest: list[dict[str, Any]] = []
    frame_rows: list[pd.DataFrame] = []
    source_files: list[dict[str, Any]] = []
    idle_index = 0

    ordered_sessions = (
        audited_frames[["session_id", "session_order"]]
        .drop_duplicates()
        .sort_values("session_order")
    )
    for session in ordered_sessions.itertuples(index=False):
        session_id = str(session.session_id)
        session_dir = blind_root / session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(session_dir)
        session_metadata = json.loads(
            (session_dir / "session_metadata.json").read_text(encoding="utf-8")
        )
        frame = (
            audited_frames[audited_frames["session_id"] == session_id]
            .copy()
            .sort_values("capture_index")
            .reset_index(drop=True)
        )
        capture_index = frame["capture_index"].to_numpy(dtype=np.int32)
        intensity, baseline, wavelength_nm, trial_id = _load_session_spectra(
            session_dir,
            capture_index,
        )
        if not np.allclose(wavelength_nm, base_wavelength_nm, rtol=0.0, atol=1e-6):
            raise ValueError(f"{session_id}: wavelength axis differs from base dataset")
        features, feature_names, _ = extract_baseline_relative_features(
            intensity,
            baseline,
            wavelength_nm,
            bin_count=64,
        )
        if features.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(f"{session_id}: unexpected feature count")
        if tuple(base_feature_names.astype(str)) != feature_names:
            raise ValueError(f"{session_id}: feature schema differs from base dataset")

        force = np.maximum(
            pd.to_numeric(frame["reference_force_fz_n"], errors="coerce").to_numpy(
                dtype=float
            ),
            0.0,
        )
        expected_position = frame["expected_position"].fillna("none").astype(str)
        ordered_labels = answers.get(session_id)
        if ordered_labels is None:
            raise ValueError(f"{session_id}: missing answer-folder provenance")
        active_session = bool(ordered_labels)
        if active_session != bool((expected_position != "none").any()):
            raise ValueError(f"{session_id}: answer/audited episode mismatch")

        contact_target, contact_mask = _contact_targets(
            force,
            active_session=active_session,
        )
        position_mask = (expected_position.to_numpy() != "none") & (
            force >= POSITION_MIN_FORCE_N
        )
        position_target = np.where(
            position_mask,
            expected_position.to_numpy(dtype=str),
            "",
        ).astype("<U16")
        force_mask = np.isfinite(force)
        fold_id = _fold_for_session(
            ordered_labels=ordered_labels,
            idle_index=idle_index,
        )
        if not active_session:
            idle_index += 1

        frame_count = len(frame)
        arrays["features"].append(features.astype(np.float32))
        arrays["force_fz_n"].append(force.astype(np.float32))
        arrays["contact_target"].append(contact_target)
        arrays["position_target"].append(position_target)
        arrays["session_id"].append(np.full(frame_count, session_id, dtype="<U96"))
        arrays["trial_id"].append(np.full(frame_count, trial_id, dtype="<U64"))
        arrays["capture_index"].append(capture_index)
        arrays["elapsed_time_sec"].append(
            frame["elapsed_time_sec"].to_numpy(dtype=np.float32)
        )
        arrays["fold_id"].append(np.full(frame_count, fold_id, dtype=np.int8))
        arrays["force_training_mask"].append(force_mask)
        arrays["contact_training_mask"].append(contact_mask)
        arrays["position_training_mask"].append(position_mask)
        arrays["release_tail_excluded"].append(
            np.zeros(frame_count, dtype=bool)
        )

        baseline_hash = hashlib.sha256(baseline.astype(np.float64).tobytes()).hexdigest()
        position_label = (
            "no_contact"
            if not ordered_labels
            else ordered_labels[0]
            if len(ordered_labels) == 1
            else "mixed_sequence"
        )
        session_manifest.append(
            {
                "session_id": session_id,
                "formal_group_id": session_id,
                "trial_id": trial_id,
                "position_label": position_label,
                "ordered_position_labels": list(ordered_labels),
                "started_at_epoch_sec": float(
                    session_metadata["started_at_epoch_sec"]
                ),
                "selection_role": "primary",
                "qa_status": "pass_unblinded_training_only",
                "finding_codes": ["filename_label_not_obfuscated"],
                "fold_id": fold_id,
                "frame_count": frame_count,
                "baseline_frame_count": 1,
                "baseline_mode": "recorded_runtime_baseline",
                "baseline_sha256": baseline_hash,
                "release_tail_excluded_frames": 0,
                "force_training_frames": int(force_mask.sum()),
                "contact_training_frames": int(contact_mask.sum()),
                "position_training_frames": int(position_mask.sum()),
                "force_min_n": float(np.nanmin(force)),
                "force_max_n": float(np.nanmax(force)),
                "session_directory": str(session_dir),
                "provenance_role": "formerly_blind_now_unblinded_training_data",
            }
        )

        frame_rows.append(
            pd.DataFrame(
                {
                    "session_id": session_id,
                    "formal_group_id": session_id,
                    "trial_id": trial_id,
                    "capture_index": capture_index,
                    "elapsed_time_sec": frame["elapsed_time_sec"].to_numpy(
                        dtype=float
                    ),
                    "fold_id": fold_id,
                    "force_fz_n": force,
                    "contact_target": contact_target,
                    "position_target": position_target,
                    "force_training_eligible": force_mask,
                    "contact_training_eligible": contact_mask,
                    "position_training_eligible": position_mask,
                    "release_tail_excluded": False,
                }
            )
        )
        for source_name in (
            "session_metadata.json",
            "frame_summary.csv",
            "synchronized_frames.jsonl",
        ):
            source_path = session_dir / source_name
            source_files.append(
                {
                    "source_group": "blind3_unblinded",
                    "relative_path": f"{session_id}/{source_name}",
                    "size_bytes": source_path.stat().st_size,
                    "sha256": _sha256(source_path),
                }
            )

    joined = {
        name: np.concatenate(parts, axis=0)
        for name, parts in arrays.items()
    }
    return joined, session_manifest, pd.concat(frame_rows, ignore_index=True), source_files


def build_incremental_dataset(
    *,
    base_dataset_dir: Path,
    blind_root: Path,
    audit_dir: Path,
    output_dir: Path,
    config_path: Path,
    dataset_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output_dir}")

    base_manifest_path = base_dataset_dir / MANIFEST_FILENAME
    base_npz_path = base_dataset_dir / DATASET_FILENAME
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("historical_data_included") is not False:
        raise ValueError("base dataset does not prove historical-data isolation")
    if base_manifest.get("selection_role") != "primary":
        raise ValueError("base dataset is not a primary-role dataset")
    base = np.load(base_npz_path, allow_pickle=False)
    audited_frames, audit_summary = _load_audited_labels(audit_dir)
    answers = _answer_labels(audit_summary)

    new, new_sessions, new_frame_manifest, source_files = _session_rows(
        blind_root=blind_root,
        audited_frames=audited_frames,
        answers=answers,
        base_wavelength_nm=base["wavelength_nm"],
        base_feature_names=base["feature_names"],
    )
    base_sessions = set(base["session_id"].astype(str))
    new_session_ids = set(new["session_id"].astype(str))
    overlap = sorted(base_sessions & new_session_ids)
    if overlap:
        raise ValueError(f"base/Blind3 session overlap: {overlap}")

    combined: dict[str, np.ndarray] = {
        "wavelength_nm": base["wavelength_nm"],
        "feature_names": base["feature_names"],
    }
    for key in (
        "features",
        "force_fz_n",
        "contact_target",
        "position_target",
        "session_id",
        "trial_id",
        "capture_index",
        "elapsed_time_sec",
        "fold_id",
        "force_training_mask",
        "contact_training_mask",
        "position_training_mask",
        "release_tail_excluded",
    ):
        combined[key] = np.concatenate([base[key], new[key]], axis=0)

    provenance = {
        "base_dataset_id": base_manifest["dataset_id"],
        "base_batch_content_sha256": base_manifest["batch_content_sha256"],
        "base_dataset_npz_sha256": _sha256(base_npz_path),
        "blind3_audit_summary_sha256": _sha256(
            audit_dir / "evaluation_summary.json"
        ),
        "blind3_frame_comparison_sha256": _sha256(
            audit_dir / "frame_comparison.csv"
        ),
        "blind3_source_files": source_files,
    }
    batch_hash = _canonical_hash(provenance)

    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / DATASET_FILENAME, **combined)

    base_frame_manifest = pd.read_csv(
        base_dataset_dir / FRAME_MANIFEST_FILENAME,
        encoding="utf-8-sig",
    )
    pd.concat(
        [base_frame_manifest, new_frame_manifest],
        ignore_index=True,
    ).to_csv(
        output_dir / FRAME_MANIFEST_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )

    inventory = {
        "schema_version": "ordinary_fbg_px6d_source_inventory_v2",
        "dataset_id": dataset_id,
        "selection_role": "primary",
        "source_policy": "strict_base_plus_unblinded_blind3_incremental",
        "hash_algorithm": "sha256",
        "batch_content_sha256": batch_hash,
        "base_inventory_path": str(
            base_dataset_dir / SOURCE_INVENTORY_FILENAME
        ),
        "base_inventory_sha256": _sha256(
            base_dataset_dir / SOURCE_INVENTORY_FILENAME
        ),
        "incremental_source_file_count": len(source_files),
        "incremental_source_total_bytes": int(
            sum(row["size_bytes"] for row in source_files)
        ),
        "files": source_files,
    }
    (output_dir / SOURCE_INVENTORY_FILENAME).write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = copy.deepcopy(base_manifest)
    manifest.update(
        {
            "schema_version": "ordinary_fbg_px6d_dataset_v3_incremental",
            "dataset_id": dataset_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_policy": "strict_base_plus_unblinded_blind3_incremental",
            "historical_data_included": False,
            "historical_data_policy": (
                "only_same-day High Sensitivity 300 us strict base and "
                "formerly-Blind3 sessions are included"
            ),
            "selection_role": "primary",
            "selection_rule": (
                "all strict v0.19.19 base sessions plus all 14 unblinded "
                "Blind3 sessions; complete session_id groups remain intact"
            ),
            "reference_validity": (
                "training_and_grouped_oof_only; Blind3 filenames were not "
                "obfuscated and Blind3 is no longer an independent blind test"
            ),
            "source_inventory_path": str(
                output_dir / SOURCE_INVENTORY_FILENAME
            ),
            "batch_content_sha256": batch_hash,
            "config_path": str(config_path),
            "formal_split_requirement": "grouped_by_complete_session_id",
            "formal_group_field": "session_id",
            "frame_count": int(len(combined["features"])),
            "session_count": int(len(base_sessions | new_session_ids)),
            "spectrum_points": int(len(combined["wavelength_nm"])),
            "feature_count": int(combined["features"].shape[1]),
            "force_training_frames": int(
                combined["force_training_mask"].sum()
            ),
            "contact_training_frames": int(
                combined["contact_training_mask"].sum()
            ),
            "position_training_frames": int(
                combined["position_training_mask"].sum()
            ),
            "release_tail_excluded_frames": int(
                combined["release_tail_excluded"].sum()
            ),
            "sessions": [*base_manifest["sessions"], *new_sessions],
            "incremental_training_provenance": {
                "source_name": "Blind3",
                "source_root": str(blind_root),
                "source_session_count": len(new_sessions),
                "source_frame_count": int(len(new["features"])),
                "label_source": (
                    "answer-folder suffixes plus PX6D episode segmentation "
                    "frozen before answer access"
                ),
                "filename_obfuscation": False,
                "independent_blind_evaluation_valid": False,
                "pretraining_replay_only": True,
                "prediction_frozen_before_answer_access": True,
                "frozen_frames_sha256": audit_summary.get(
                    "frozen_frames_sha256"
                ),
                "frozen_predictions_sha256": audit_summary.get(
                    "frozen_predictions_sha256"
                ),
                "audited_frame_table_sha256": provenance[
                    "blind3_frame_comparison_sha256"
                ],
                "baseline_policy": "recorded_runtime_baseline_per_session",
                "contact_thresholds_n": {
                    "no_contact_max": NO_CONTACT_MAX_FORCE_N,
                    "contact_min": CONTACT_MIN_FORCE_N,
                },
                "position_min_force_n": POSITION_MIN_FORCE_N,
                "fold_policy": (
                    "complete sessions only; two ordered sequences in folds "
                    "0/1, single-position and idle sessions spread over 2/3/4"
                ),
            },
        }
    )
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "base_session_count": len(base_sessions),
        "added_session_count": len(new_session_ids),
        "session_count": manifest["session_count"],
        "base_frame_count": int(len(base["features"])),
        "added_frame_count": int(len(new["features"])),
        "frame_count": manifest["frame_count"],
        "batch_content_sha256": batch_hash,
        "force_max_n": float(np.nanmax(combined["force_fz_n"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a strict base plus unblinded Blind3 training dataset."
    )
    parser.add_argument("--base-dataset-dir", type=Path, required=True)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    result = build_incremental_dataset(
        base_dataset_dir=args.base_dataset_dir.resolve(),
        blind_root=args.blind_root.resolve(),
        audit_dir=args.audit_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        config_path=args.config.resolve(),
        dataset_id=args.dataset_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
