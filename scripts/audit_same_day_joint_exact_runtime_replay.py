"""Replay the same-day joint nine-FBG candidate through the real adapter.

This audit deliberately keeps acquisition order within each session.  The
separate stress audit owns cross-batch row shuffling because shuffling frames
through a stateful contact gate would destroy the physical time series.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.all_source_runtime_adapter import (  # noqa: E402
    AllSourceOpticalForceAdapter,
    JOINT_NINE_FBG_RUNTIME_SCHEMA,
)


POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
EXPECTED_SOURCE_BATCHES = {
    "regular_v01916",
    "regular_v01919",
    "blind1",
    "blind3",
    "blind4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--peak-config", type=Path, default=ROOT / "config/hybrid_spectrum_channels.yaml")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "config/runtime_contact_state_beta.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-batch",
        action="append",
        default=[],
        help="Optional source-batch filter. Repeat for multiple batches.",
    )
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Optional exact session filter. Repeat for multiple sessions.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Smoke-test limit after source filtering.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many sessions.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_sections(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        dict(payload.get("runtime_baseline_recovery") or {}),
        dict(payload.get("all_source_runtime_gate") or {}),
    )


def _load_frames(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "synchronized_frames.jsonl"
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            spectrum = row.get("spectrum") or {}
            wavelength = np.asarray(spectrum.get("wavelength_nm"), dtype=float)
            intensity = np.asarray(spectrum.get("intensity_counts"), dtype=float)
            if wavelength.shape != intensity.shape or wavelength.size != 512:
                raise ValueError(
                    f"invalid spectrum in {path} line {line_number}: "
                    f"{wavelength.shape} vs {intensity.shape}"
                )
            frames.append(
                {
                    "capture_index": int(row["capture_index"]),
                    "elapsed_time_sec": float(row["elapsed_time_sec"]),
                    "wavelength_nm": wavelength,
                    "intensity_counts": intensity,
                }
            )
    frames.sort(key=lambda row: row["capture_index"])
    return frames


def _dataset_indices_by_session(session_ids: np.ndarray) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, session_id in enumerate(session_ids.astype(str)):
        buckets[session_id].append(index)
    return {
        session_id: np.asarray(indices, dtype=int)
        for session_id, indices in buckets.items()
    }


def _activation_episode_count(values: Iterable[bool]) -> int:
    episodes = 0
    previous = False
    for value in values:
        current = bool(value)
        episodes += int(current and not previous)
        previous = current
    return episodes


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _macro_f1(confusion: dict[str, Counter[str]]) -> float:
    scores: list[float] = []
    for label in POSITION_ORDER:
        true_positive = int(confusion[label][label])
        false_positive = sum(
            int(confusion[truth][label])
            for truth in POSITION_ORDER
            if truth != label
        )
        false_negative = sum(
            int(count)
            for prediction, count in confusion[label].items()
            if prediction != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "ordinary_fbg_px6d_dataset_manifest.json"
    dataset_path = dataset_dir / "ordinary_fbg_px6d_dataset.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("random_frame_split_used"):
        raise RuntimeError("exact replay refuses random frame splits")
    excluded = {str(value).lower() for value in manifest.get("excluded_source_batches", [])}
    if "blind2" not in excluded:
        raise RuntimeError("Blind2 exclusion is not recorded in the dataset manifest")

    sessions = list(manifest["sessions"])
    included_batches = {str(row["source_batch"]) for row in sessions}
    if "blind2" in {value.lower() for value in included_batches}:
        raise RuntimeError("Blind2 must not enter exact runtime replay")
    if not EXPECTED_SOURCE_BATCHES.issubset(included_batches):
        missing = sorted(EXPECTED_SOURCE_BATCHES - included_batches)
        raise RuntimeError(f"missing required same-day source batches: {missing}")
    requested_batches = {str(value) for value in args.source_batch}
    if requested_batches:
        sessions = [row for row in sessions if row["source_batch"] in requested_batches]
    requested_sessions = {str(value) for value in args.session_id}
    if requested_sessions:
        sessions = [row for row in sessions if row["session_id"] in requested_sessions]
        found_sessions = {str(row["session_id"]) for row in sessions}
        if found_sessions != requested_sessions:
            missing = sorted(requested_sessions - found_sessions)
            raise RuntimeError(f"requested sessions not found: {missing}")
    sessions.sort(key=lambda row: (row["fold_id"], row["started_at_epoch_sec"], row["session_id"]))
    if args.max_sessions is not None:
        if args.max_sessions < 1:
            raise ValueError("--max-sessions must be positive")
        sessions = sessions[: args.max_sessions]
    if not sessions:
        raise RuntimeError("no sessions selected")

    arrays = np.load(dataset_path, allow_pickle=True)
    by_session = _dataset_indices_by_session(arrays["session_id"])
    recovery_config, gate_config = _runtime_sections(args.runtime_config.resolve())
    adapter = AllSourceOpticalForceAdapter.from_paths(
        model_path,
        args.peak_config.resolve(),
        runtime_recovery_config=recovery_config,
        runtime_gate_config=gate_config,
    )
    if adapter.literature_runtime_schema_version != JOINT_NINE_FBG_RUNTIME_SCHEMA:
        raise RuntimeError("candidate is not the joint nine-FBG v4 runtime schema")

    frame_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = {
        label: Counter() for label in POSITION_ORDER
    }
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.perf_counter()

    for session_number, descriptor in enumerate(sessions, start=1):
        session_id = str(descriptor["session_id"])
        source_batch = str(descriptor["source_batch"])
        dataset_indices = by_session.get(session_id)
        if dataset_indices is None:
            raise RuntimeError(f"session missing from NPZ: {session_id}")
        frames = _load_frames(Path(descriptor["session_directory"]))
        captures = arrays["capture_index"][dataset_indices].astype(int)
        frame_captures = np.asarray([row["capture_index"] for row in frames], dtype=int)
        if not np.array_equal(captures, frame_captures):
            raise RuntimeError(f"capture order mismatch: {session_id}")

        baseline_capture_indices = {
            int(value) for value in descriptor["baseline_capture_indices"]
        }
        baseline_rows = [
            row["intensity_counts"]
            for row in frames
            if row["capture_index"] in baseline_capture_indices
        ]
        if len(baseline_rows) != int(descriptor["baseline_frame_count"]):
            raise RuntimeError(f"baseline frame mismatch: {session_id}")
        baseline = np.median(np.stack(baseline_rows), axis=0)
        adapter.set_baseline(frames[0]["wavelength_nm"], baseline)

        truth_contact = arrays["contact_target"][dataset_indices].astype(int)
        truth_position = arrays["position_target"][dataset_indices].astype(str)
        contact_mask = arrays["contact_training_mask"][dataset_indices].astype(bool)
        position_mask = arrays["position_training_mask"][dataset_indices].astype(bool)
        force = arrays["force_fz_n"][dataset_indices].astype(float)
        predicted_contact: list[bool] = []
        session_position_votes: Counter[str] = Counter()
        counts: Counter[str] = Counter()

        for local_index, (frame, dataset_index) in enumerate(
            zip(frames, dataset_indices, strict=True)
        ):
            result = adapter.update(
                frame["wavelength_nm"],
                frame["intensity_counts"],
                source_timestamp_sec=frame["elapsed_time_sec"],
            )
            if not result.get("ok"):
                raise RuntimeError(
                    f"runtime failure {session_id} frame {local_index}: {result}"
                )
            contact_active = result["contact"]["label"] == "contact"
            position_label = str(result["position"].get("label") or "")
            raw_position = str(result["position"].get("raw_label") or "")
            predicted_contact.append(contact_active)
            counts["frames"] += 1

            if contact_mask[local_index] and truth_contact[local_index] == 0:
                counts["no_contact_frames"] += 1
                counts["no_contact_false_positive"] += int(contact_active)
                source_counts[source_batch]["no_contact_frames"] += 1
                source_counts[source_batch]["no_contact_false_positive"] += int(contact_active)
            if contact_mask[local_index] and truth_contact[local_index] == 1:
                counts["contact_frames"] += 1
                counts["contact_true_positive"] += int(contact_active)
                source_counts[source_batch]["contact_frames"] += 1
                source_counts[source_batch]["contact_true_positive"] += int(contact_active)
            if position_mask[local_index] and truth_position[local_index] in POSITION_ORDER:
                truth_label = truth_position[local_index]
                counts["position_frames"] += 1
                counts["position_correct"] += int(position_label == truth_label)
                source_counts[source_batch]["position_frames"] += 1
                source_counts[source_batch]["position_correct"] += int(position_label == truth_label)
                confusion[truth_label][position_label or "none"] += 1
                if contact_active and position_label in POSITION_ORDER:
                    session_position_votes[position_label] += 1

            gate = dict(result.get("runtime_contact_gate") or {})
            recovery = dict(gate.get("baseline_recovery") or {})
            coupled_signature = dict(gate.get("coupled_contact_signature") or {})
            frame_rows.append(
                {
                    "source_batch": source_batch,
                    "fold_id": int(descriptor["fold_id"]),
                    "session_id": session_id,
                    "capture_index": int(frame["capture_index"]),
                    "elapsed_time_sec": float(frame["elapsed_time_sec"]),
                    "truth_contact": int(truth_contact[local_index]),
                    "contact_training_mask": int(contact_mask[local_index]),
                    "truth_position": truth_position[local_index],
                    "position_training_mask": int(position_mask[local_index]),
                    "reference_force_fz_n": float(force[local_index]),
                    "runtime_contact": int(contact_active),
                    "contact_probability": float(result["contact"]["contact_probability"]),
                    "runtime_position": position_label,
                    "raw_position": raw_position,
                    "raw_position_confidence": float(result["position"].get("raw_confidence") or 0.0),
                    "raw_position_margin": float(result["position"].get("raw_margin") or 0.0),
                    "position_confidence": float(result["position"].get("confidence") or 0.0),
                    "position_margin": float(result["position"].get("margin") or 0.0),
                    "estimated_force_fz_n": float(result["estimated_force_fz_n"]),
                    "inference_latency_ms": float(result.get("inference_latency_ms") or 0.0),
                    "raw_contact_active": int(bool(gate.get("raw_contact_active"))),
                    "contact_latched": int(bool(gate.get("contact_latched"))),
                    "baseline_distance": float(gate.get("baseline_distance") or 0.0),
                    "baseline_separated": int(bool(gate.get("baseline_separated"))),
                    "near_runtime_baseline": int(bool(gate.get("near_runtime_baseline"))),
                    "fresh_spectral_activity": int(bool(gate.get("fresh_spectral_activity"))),
                    "spectral_activity_recent": int(bool(gate.get("spectral_activity_recent"))),
                    "slow_baseline_departure": int(bool(gate.get("slow_baseline_departure"))),
                    "confirmed_rest_suppression": int(bool(gate.get("confirmed_rest_suppression"))),
                    "model_consensus_contact_evidence": int(bool(gate.get("model_consensus_contact_evidence"))),
                    "coupled_signature_credible": int(bool(gate.get("coupled_contact_signature_credible"))),
                    "coupled_low_response_channel_count": int(coupled_signature.get("low_response_channel_count") or 0),
                    "coupled_nominal_response_channel_count": int(coupled_signature.get("nominal_response_channel_count") or 0),
                    "runtime_rest_latched": int(bool(recovery.get("runtime_rest_latched"))),
                    "recovery_suppress_contact": int(bool(recovery.get("suppress_contact"))),
                }
            )

        dedicated_idle = bool(
            descriptor["position_label"] == "unlabeled"
            and not descriptor["ordered_position_labels"]
            and int(descriptor["position_training_frames"]) == 0
        )
        false_episodes = _activation_episode_count(predicted_contact) if dedicated_idle else 0
        majority_position = session_position_votes.most_common(1)[0][0] if session_position_votes else ""
        single_position = (
            str(descriptor["position_label"])
            if str(descriptor["position_label"]) in POSITION_ORDER
            else ""
        )
        session_rows.append(
            {
                "source_batch": source_batch,
                "fold_id": int(descriptor["fold_id"]),
                "session_id": session_id,
                "position_label": descriptor["position_label"],
                "dedicated_idle": int(dedicated_idle),
                "frame_count": counts["frames"],
                "no_contact_frame_count": counts["no_contact_frames"],
                "no_contact_false_positive_count": counts["no_contact_false_positive"],
                "contact_frame_count": counts["contact_frames"],
                "contact_true_positive_count": counts["contact_true_positive"],
                "position_frame_count": counts["position_frames"],
                "position_correct_count": counts["position_correct"],
                "dedicated_idle_false_activation_episodes": false_episodes,
                "majority_position": majority_position,
                "single_position_correct": int(bool(single_position) and majority_position == single_position),
            }
        )
        if args.progress_every > 0 and (
            session_number % args.progress_every == 0 or session_number == len(sessions)
        ):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "progress_sessions": session_number,
                        "total_sessions": len(sessions),
                        "frames": len(frame_rows),
                        "elapsed_sec": round(elapsed, 1),
                    }
                ),
                flush=True,
            )

    dedicated_idle_rows = [row for row in session_rows if row["dedicated_idle"]]
    dedicated_idle_ids = {row["session_id"] for row in dedicated_idle_rows}
    dedicated_idle_frames = [row for row in frame_rows if row["session_id"] in dedicated_idle_ids]
    no_contact_rows = [
        row for row in frame_rows
        if row["contact_training_mask"] and row["truth_contact"] == 0
    ]
    contact_rows = [
        row for row in frame_rows
        if row["contact_training_mask"] and row["truth_contact"] == 1
    ]
    position_rows = [row for row in frame_rows if row["position_training_mask"]]
    single_position_sessions = [
        row for row in session_rows if row["position_label"] in POSITION_ORDER
    ]

    source_metrics: dict[str, Any] = {}
    for source_batch in sorted({row["source_batch"] for row in session_rows}):
        counts = source_counts[source_batch]
        source_metrics[source_batch] = {
            "session_count": sum(row["source_batch"] == source_batch for row in session_rows),
            "no_contact_false_positive_rate": _safe_ratio(
                counts["no_contact_false_positive"], counts["no_contact_frames"]
            ),
            "contact_recall": _safe_ratio(
                counts["contact_true_positive"], counts["contact_frames"]
            ),
            "position_frame_accuracy": _safe_ratio(
                counts["position_correct"], counts["position_frames"]
            ),
        }

    confusion_payload = {
        truth: {prediction: int(count) for prediction, count in sorted(counts.items())}
        for truth, counts in confusion.items()
    }
    metrics = {
        "schema_version": "touch_same_day_joint_exact_runtime_replay_v1",
        "evaluation_status": "exact_adapter_replay_not_deployed",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(manifest_path),
        "blind2_excluded": True,
        "random_frame_split_used": False,
        "runtime_schema": adapter.literature_runtime_schema_version,
        "session_count": len(session_rows),
        "frame_count": len(frame_rows),
        "dedicated_idle_session_count": len(dedicated_idle_rows),
        "dedicated_idle_frame_count": len(dedicated_idle_frames),
        "dedicated_idle_false_positive_frames": sum(row["runtime_contact"] for row in dedicated_idle_frames),
        "dedicated_idle_false_positive_rate": float(np.mean([row["runtime_contact"] for row in dedicated_idle_frames])) if dedicated_idle_frames else None,
        "dedicated_idle_false_activation_episodes": sum(row["dedicated_idle_false_activation_episodes"] for row in dedicated_idle_rows),
        "all_labeled_no_contact_frame_count": len(no_contact_rows),
        "all_labeled_no_contact_false_positive_rate": float(np.mean([row["runtime_contact"] for row in no_contact_rows])) if no_contact_rows else None,
        "active_contact_frame_count": len(contact_rows),
        "active_contact_recall": float(np.mean([row["runtime_contact"] for row in contact_rows])) if contact_rows else None,
        "position_frame_count": len(position_rows),
        "position_frame_accuracy": float(np.mean([row["runtime_position"] == row["truth_position"] for row in position_rows])) if position_rows else None,
        "position_macro_f1": _macro_f1(confusion),
        "single_position_session_count": len(single_position_sessions),
        "single_position_session_accuracy": float(np.mean([row["single_position_correct"] for row in single_position_sessions])) if single_position_sessions else None,
        "source_metrics": source_metrics,
        "position_confusion": confusion_payload,
        "elapsed_sec": float(time.perf_counter() - started),
        "temporal_replay_note": "Frames preserve original within-session order so the real stateful gate is tested.",
        "shuffle_note": "Cross-batch order invariance is evaluated by the companion static stress audit; shuffled stateful replay is physically invalid.",
    }
    _rows_to_csv(output_dir / "frames.csv", frame_rows)
    _rows_to_csv(output_dir / "sessions.csv", session_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
