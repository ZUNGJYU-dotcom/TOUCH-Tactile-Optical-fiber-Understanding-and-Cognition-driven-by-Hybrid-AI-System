"""Replay anonymous position captures through the deployed TOUCH Beta runtime.

The default mode freezes predictions before answer access. The explicit
post-unblind mode is reserved for deployment verification after labels are
known and marks its artifacts accordingly so they cannot be mistaken for
independent blind evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.all_source_runtime_adapter import (  # noqa: E402
    AllSourceOpticalForceAdapter,
)
from src.hybrid_spectrum.px6d_session_dataset import (  # noqa: E402
    POSITION_ORDER,
    _load_session_frame_matrix,
    discover_sessions,
)


DEFAULT_TRAINING_MANIFEST = (
    ROOT
    / "outputs"
    / "ordinary_fbg_px6d_strict_20260902_v01919_joint_signature_v1"
    / "ordinary_fbg_px6d_dataset_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/deployed/ordinary_fbg_current_runtime.joblib",
    )
    parser.add_argument(
        "--peak-config",
        type=Path,
        default=ROOT / "config/hybrid_spectrum_channels.yaml",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "config/runtime_contact_state_beta.yaml",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=DEFAULT_TRAINING_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-force-min-n", type=float, default=0.25)
    parser.add_argument(
        "--user-confirmed-unseen",
        action="store_true",
        help="Record the operator's confirmation that blind sessions were not trained.",
    )
    parser.add_argument(
        "--post-unblind-deployment-validation",
        action="store_true",
        help=(
            "Mark this replay as answer-known deployment validation rather than "
            "independent pre-unblind evidence."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_sections(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        dict(payload.get("runtime_baseline_recovery") or {}),
        dict(payload.get("all_source_runtime_gate") or {}),
    )


def _majority(votes: Iterable[str]) -> tuple[str, int, int, float, float]:
    counts = Counter(value for value in votes if value in POSITION_ORDER)
    if not counts:
        return "none", 0, 0, 0.0, 0.0
    rank = {label: index for index, label in enumerate(POSITION_ORDER)}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], rank[item[0]]))
    winner, winner_count = ordered[0]
    runner_count = ordered[1][1] if len(ordered) > 1 else 0
    total = sum(counts.values())
    return (
        winner,
        winner_count,
        runner_count,
        float(winner_count / total),
        float((winner_count - runner_count) / total),
    )


def _training_overlap(
    training_manifest: Path,
    blind_session_ids: set[str],
) -> tuple[list[str], int]:
    payload = json.loads(training_manifest.read_text(encoding="utf-8"))
    training_ids = {
        str(row.get("session_id") or "")
        for row in payload.get("sessions") or []
        if str(row.get("session_id") or "")
    }
    return sorted(training_ids & blind_session_ids), len(training_ids)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty result table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _capture_session_id(descriptor: Any) -> str:
    """Use the immutable capture directory name as the anonymous record key."""

    return str(descriptor.session_dir.name)


def main() -> int:
    args = parse_args()
    blind_root = args.blind_root.resolve()
    model_path = args.model.resolve()
    peak_config_path = args.peak_config.resolve()
    runtime_config_path = args.runtime_config.resolve()
    training_manifest_path = args.training_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen blind predictions: {output_dir}"
        )

    sessions = tuple(
        sorted(
            discover_sessions(blind_root),
            key=lambda item: (item.started_at_epoch_sec, item.session_id),
        )
    )
    if not sessions:
        raise RuntimeError(f"no capture sessions found under {blind_root}")
    non_anonymous = [
        item.session_id for item in sessions if item.position_label != "unlabeled"
    ]
    if non_anonymous:
        raise RuntimeError(
            "blind source contains non-anonymous position labels: "
            + ", ".join(non_anonymous)
        )

    blind_ids = {
        identifier
        for item in sessions
        for identifier in (_capture_session_id(item), str(item.session_id))
    }
    overlap, training_session_count = _training_overlap(
        training_manifest_path, blind_ids
    )
    if overlap:
        raise RuntimeError(
            "blind session IDs overlap the training manifest: " + ", ".join(overlap)
        )

    recovery_config, gate_config = _runtime_sections(runtime_config_path)
    adapter = AllSourceOpticalForceAdapter.from_paths(
        model_path,
        peak_config_path,
        runtime_recovery_config=recovery_config,
        runtime_gate_config=gate_config,
    )

    frame_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    source_fingerprints: list[dict[str, Any]] = []
    for session_index, descriptor in enumerate(sessions, start=1):
        capture_session_id = _capture_session_id(descriptor)
        summary, wavelength, intensity = _load_session_frame_matrix(
            descriptor, expected_points=512
        )
        baseline_frame_count = min(5, len(intensity))
        if baseline_frame_count < 1:
            raise RuntimeError(f"empty spectrum session: {capture_session_id}")
        baseline = np.median(intensity[:baseline_frame_count], axis=0)
        adapter.set_baseline(wavelength, baseline)

        visual_votes: list[str] = []
        all_visual_votes: list[str] = []
        raw_votes: list[str] = []
        formal_votes: list[str] = []
        active_frame_count = 0
        visual_active_count = 0
        all_visual_active_count = 0
        active_force_values: list[float] = []
        active_estimated_force_values: list[float] = []
        for row_index, current in enumerate(intensity):
            row = summary.iloc[row_index]
            force_n = float(row["force_fz_n"])
            timestamp = float(row["elapsed_time_sec"])
            result = adapter.update(
                wavelength,
                current,
                source_timestamp_sec=timestamp,
            )
            if not result.get("ok"):
                raise RuntimeError(
                    f"runtime failed for {capture_session_id} frame {row_index}: "
                    f"{result}"
                )

            position = dict(result.get("position") or {})
            digital_twin = dict(result.get("digital_twin") or {})
            contact = dict(result.get("contact") or {})
            gate = dict(result.get("runtime_contact_gate") or {})
            raw_position = str(position.get("raw_label") or "none")
            formal_position = str(position.get("label") or "none")
            visual_position = str(position.get("visual_label") or "none")
            visual_active = bool(digital_twin.get("visual_active"))
            is_active_reference = force_n >= args.active_force_min_n
            estimated_force = result.get("estimated_force_fz_n")
            estimated_force_n = (
                float(estimated_force) if estimated_force is not None else None
            )

            if visual_active:
                all_visual_active_count += 1
                all_visual_votes.append(visual_position)
            if is_active_reference:
                active_frame_count += 1
                active_force_values.append(force_n)
                raw_votes.append(raw_position)
                formal_votes.append(formal_position)
                if visual_active:
                    visual_active_count += 1
                    visual_votes.append(visual_position)
                if estimated_force_n is not None and np.isfinite(estimated_force_n):
                    active_estimated_force_values.append(estimated_force_n)

            frame_rows.append(
                {
                    "session_id": capture_session_id,
                    "metadata_session_id": str(descriptor.session_id),
                    "capture_index": int(row["capture_index"]),
                    "elapsed_time_sec": timestamp,
                    "reference_force_fz_n": force_n,
                    "active_reference_frame": is_active_reference,
                    "contact_probability": float(
                        contact.get("contact_probability") or 0.0
                    ),
                    "raw_position": raw_position,
                    "raw_position_confidence": float(
                        position.get("raw_confidence") or 0.0
                    ),
                    "raw_position_probabilities": json.dumps(
                        dict(position.get("raw_probabilities") or {}),
                        sort_keys=True,
                    ),
                    "formal_position": formal_position,
                    "visual_position": visual_position,
                    "visual_position_confidence": float(
                        position.get("visual_confidence") or 0.0
                    ),
                    "visual_position_margin": float(
                        position.get("visual_margin") or 0.0
                    ),
                    "visual_position_probabilities": json.dumps(
                        dict(position.get("visual_probabilities") or {}),
                        sort_keys=True,
                    ),
                    "visual_active": visual_active,
                    "estimated_force_fz_n": estimated_force_n,
                    "baseline_distance": float(gate.get("baseline_distance") or 0.0),
                    "coupled_contact_signature_credible": bool(
                        gate.get("coupled_contact_signature_credible")
                    ),
                }
            )

        visual = _majority(all_visual_votes)
        active_reference_visual = _majority(visual_votes)
        raw = _majority(raw_votes)
        formal = _majority(formal_votes)
        prediction_rows.append(
            {
                "session_order": session_index,
                "session_id": capture_session_id,
                "metadata_session_id": str(descriptor.session_id),
                "predicted_position": visual[0],
                "primary_vote_source": "visual_position_when_visual_active_all_frames",
                "active_force_min_n": args.active_force_min_n,
                "total_frames": len(intensity),
                "active_reference_frames": active_frame_count,
                "visual_active_frames": all_visual_active_count,
                "active_reference_visual_active_frames": visual_active_count,
                "active_visual_coverage": (
                    float(visual_active_count / active_frame_count)
                    if active_frame_count
                    else 0.0
                ),
                "winner_votes": visual[1],
                "runner_up_votes": visual[2],
                "winner_share": visual[3],
                "winner_margin_share": visual[4],
                "raw_majority_position": raw[0],
                "raw_winner_share": raw[3],
                "formal_majority_position": formal[0],
                "formal_winner_share": formal[3],
                "active_reference_visual_majority_position": active_reference_visual[0],
                "active_reference_visual_winner_share": active_reference_visual[3],
                "reference_force_mean_n": (
                    float(np.mean(active_force_values))
                    if active_force_values
                    else None
                ),
                "reference_force_max_n": (
                    float(np.max(active_force_values))
                    if active_force_values
                    else None
                ),
                "estimated_force_mean_n": (
                    float(np.mean(active_estimated_force_values))
                    if active_estimated_force_values
                    else None
                ),
                "visual_vote_counts": json.dumps(
                    dict(Counter(all_visual_votes)), sort_keys=True
                ),
                "active_reference_visual_vote_counts": json.dumps(
                    dict(Counter(visual_votes)), sort_keys=True
                ),
                "raw_vote_counts": json.dumps(dict(Counter(raw_votes)), sort_keys=True),
                "formal_vote_counts": json.dumps(
                    dict(Counter(formal_votes)), sort_keys=True
                ),
            }
        )
        spectrum_path = descriptor.session_dir / "spectrum_timeseries.csv"
        source_fingerprints.append(
            {
                "session_id": capture_session_id,
                "metadata_session_id": str(descriptor.session_id),
                "spectrum_timeseries_sha256": _sha256(spectrum_path),
            }
        )
        print(
            f"[{session_index:02d}/{len(sessions):02d}] "
            f"{capture_session_id} -> {visual[0]} "
            f"({visual[1]}/{sum(Counter(all_visual_votes).values())} visual votes)"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    frames_path = output_dir / "frames.csv"
    predictions_path = output_dir / "blind_predictions.csv"
    _write_csv(frames_path, frame_rows)
    _write_csv(predictions_path, prediction_rows)
    predictions_sha256 = _sha256(predictions_path)
    post_unblind = bool(args.post_unblind_deployment_validation)
    manifest = {
        "schema_version": "touch_anonymous_position_runtime_replay_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer_accessed": post_unblind,
        "evaluation_phase": (
            "post_unblind_deployment_validation"
            if post_unblind
            else "pre_unblind_frozen_prediction"
        ),
        "independent_blind_evidence": not post_unblind,
        "deployment_validation_only": post_unblind,
        "blind_root": str(blind_root),
        "session_count": len(sessions),
        "session_ids": [_capture_session_id(item) for item in sessions],
        "metadata_session_ids": [str(item.session_id) for item in sessions],
        "source_fingerprints": source_fingerprints,
        "user_confirmed_sessions_not_used_for_training": bool(
            args.user_confirmed_unseen
        ),
        "training_overlap_audit": {
            "method": "exact session_id intersection",
            "training_manifest_path": str(training_manifest_path),
            "training_manifest_sha256": _sha256(training_manifest_path),
            "training_session_count": training_session_count,
            "overlap_count": len(overlap),
            "overlap_session_ids": overlap,
        },
        "prediction_rule": {
            "baseline": "median of first 5 captured spectra",
            "active_reference_frame": (
                f"PX6D force_fz_n >= {args.active_force_min_n:g} N"
            ),
            "primary_vote": (
                "majority visual_position among all captured frames where "
                "digital_twin.visual_active is true; no PX6D label gating"
            ),
            "reference_only_diagnostic": (
                "PX6D-active frame votes are retained as diagnostics and do not "
                "select the primary prediction"
            ),
            "tie_break": "POSITION_ORDER",
            "answer_dependent_tuning": False if not post_unblind else None,
        },
        "runtime_artifacts": {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "peak_config_path": str(peak_config_path),
            "peak_config_sha256": _sha256(peak_config_path),
            "runtime_config_path": str(runtime_config_path),
            "runtime_config_sha256": _sha256(runtime_config_path),
        },
        "outputs": {
            "frames_csv": frames_path.name,
            "frames_sha256": _sha256(frames_path),
            "predictions_csv": predictions_path.name,
            "predictions_sha256": predictions_sha256,
        },
    }
    manifest_path = output_dir / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"predictions_sha256={predictions_sha256}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
