"""Post-unblind cross-batch shuffled replay of all ordinary-FBG blind captures.

Sessions from Blind1-Blind4 are pooled and globally shuffled before inference.
Frames inside a session remain chronological because the deployed runtime uses
contact hysteresis and position episode locking. Answers are loaded only after
the unscored runtime outputs have been written and hashed. The resulting score
is a regression/deployment audit, not new independent blind evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
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
    _load_session_recorded_baseline,
    discover_sessions,
)


DEFAULT_TEST_ROOT = Path(
    "E:/重要文档/实验/柔性传感/光纤/Micro-FBG/普通FBG/data/test data"
)
DEFAULT_TRAINING_MANIFEST = (
    ROOT
    / "outputs/ordinary_fbg_px6d_20260902_v01922_blind3_unblinded_incremental_v1"
    / "ordinary_fbg_px6d_dataset_manifest.json"
)
DEFAULT_BLIND3_AUDIT = (
    ROOT / "outputs/blind3_record_comparison_20260902_v01922_v1"
)
ANSWER_SUFFIX = re.compile(
    r"^(?P<stem>.+)-(?P<label>uncontact|no[_-]?contact|p(?:11|12|13|21|22|23|31|32|33))$",
    re.IGNORECASE,
)
TIMESTAMP_PREFIX = re.compile(r"^(\d{8}_\d{6})_")
BLIND2_OVERRIDES = {
    "20260902_192225_unlabeled_continuous_px6d_fz_reference_trial_017": "P22",
    "20260902_192252_unlabeled_continuous_px6d_fz_reference_trial_018": "P32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
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
        "--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST
    )
    parser.add_argument("--blind3-audit", type=Path, default=DEFAULT_BLIND3_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=2026090274)
    parser.add_argument("--active-force-min-n", type=float, default=0.25)
    parser.add_argument(
        "--baseline-mode",
        choices=("both", "recorded", "first5_median"),
        default="both",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _runtime_sections(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        dict(payload.get("runtime_baseline_recovery") or {}),
        dict(payload.get("all_source_runtime_gate") or {}),
    )


def _majority(values: Iterable[str]) -> tuple[str, int, int, float]:
    counts = Counter(value for value in values if value in POSITION_ORDER)
    if not counts:
        return "none", 0, 0, 0.0
    rank = {label: index for index, label in enumerate(POSITION_ORDER)}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], rank[item[0]]))
    winner, winner_count = ordered[0]
    runner_count = ordered[1][1] if len(ordered) > 1 else 0
    return winner, winner_count, runner_count, winner_count / sum(counts.values())


def _normalise_answer_label(raw: str) -> str:
    value = raw.lower()
    if value in {"uncontact", "nocontact", "no_contact", "no-contact"}:
        return "none"
    label = value.upper()
    if label not in POSITION_ORDER:
        raise ValueError(f"unknown answer label: {raw}")
    return label


def _timestamp(value: str) -> str:
    match = TIMESTAMP_PREFIX.match(value)
    if not match:
        raise ValueError(f"missing timestamp prefix: {value}")
    return match.group(1)


def _single_answer_mapping(
    batch: str,
    source_ids: list[str],
    answer_root: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    entries: list[dict[str, str]] = []
    for path in sorted(answer_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        match = ANSWER_SUFFIX.match(path.name)
        if not match:
            raise ValueError(f"unrecognized answer folder: {path.name}")
        entries.append(
            {
                "folder": path.name,
                "stem": match.group("stem"),
                "timestamp": _timestamp(match.group("stem")),
                "label": _normalise_answer_label(match.group("label")),
            }
        )
    if len(entries) != len(source_ids):
        raise RuntimeError(
            f"{batch}: answer/source count mismatch {len(entries)} vs {len(source_ids)}"
        )

    overrides = BLIND2_OVERRIDES if batch == "blind2" else {}
    mapping: dict[str, dict[str, str]] = {}
    consumed: set[str] = set()

    def assign(source_id: str, candidates: list[dict[str, str]], method: str) -> bool:
        candidates = [row for row in candidates if row["folder"] not in consumed]
        override = overrides.get(source_id)
        if override:
            candidates = [row for row in candidates if row["label"] == override]
        if len(candidates) != 1:
            return False
        row = candidates[0]
        mapping[source_id] = {**row, "match_method": method}
        consumed.add(row["folder"])
        return True

    for source_id in source_ids:
        exact = [row for row in entries if row["stem"] == source_id]
        assign(source_id, exact, "exact_source_id")
    for source_id in source_ids:
        if source_id in mapping:
            continue
        timestamp_matches = [
            row for row in entries if row["timestamp"] == _timestamp(source_id)
        ]
        assign(source_id, timestamp_matches, "timestamp_prefix")
    for source_id in source_ids:
        if source_id in mapping:
            continue
        override = overrides.get(source_id)
        if not override:
            continue
        remaining = [
            row
            for row in entries
            if row["folder"] not in consumed and row["label"] == override
        ]
        if not assign(
            source_id,
            remaining,
            "documented_unique_unmatched_blind2_answer_correction",
        ):
            raise RuntimeError(f"{batch}: override did not resolve {source_id}")

    missing = sorted(set(source_ids) - set(mapping))
    unused = sorted(row["folder"] for row in entries if row["folder"] not in consumed)
    if missing or unused:
        raise RuntimeError(f"{batch}: unresolved answers missing={missing}, unused={unused}")
    provenance = [
        {
            "batch": batch,
            "session_id": source_id,
            "answer_folder": mapping[source_id]["folder"],
            "expected_label": mapping[source_id]["label"],
            "match_method": mapping[source_id]["match_method"],
        }
        for source_id in source_ids
    ]
    return mapping, provenance


def _load_blind3_labels(audit_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = audit_dir / "evaluation_summary.json"
    frames_path = audit_dir / "frame_comparison.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "touch_blind_multiepisode_record_audit_v1":
        raise ValueError("unexpected Blind3 audit schema")
    if summary.get("prediction_frozen_before_answer_access") is not True:
        raise ValueError("Blind3 label audit lacks pre-unblind provenance")
    expected_hash = str((summary.get("outputs") or {}).get(frames_path.name) or "")
    if not expected_hash or _sha256(frames_path) != expected_hash:
        raise RuntimeError("Blind3 audited frame-label hash mismatch")
    frames = pd.read_csv(frames_path, encoding="utf-8-sig")
    required = {"session_id", "capture_index", "expected_position"}
    missing = sorted(required - set(frames.columns))
    if missing:
        raise ValueError(f"Blind3 label audit missing columns: {missing}")
    return frames[list(required)].copy(), summary


def _training_ids(path: Path) -> tuple[set[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for row in payload.get("sessions") or []:
        session_id = str(row.get("session_id") or "")
        session_dir = Path(str(row.get("session_directory") or "")).name
        if session_id:
            ids.add(session_id)
        if session_dir:
            ids.add(session_dir)
    return ids, payload


def _force_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(reference) & np.isfinite(estimate)
    x = reference[mask]
    y = estimate[mask]
    if len(x) < 2:
        return {"n": int(len(x)), "mae_n": None, "pearson_r": None, "slope": None}
    error = y - x
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
    return {
        "n": int(len(x)),
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_r": correlation,
        "slope": float(slope),
        "intercept_n": float(intercept),
    }


def _summarize_scope(
    frames: pd.DataFrame,
    sessions: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    active = frames["expected_position"] != "none"
    displayed = frames["visual_active"]
    dedicated_idle_ids = set(
        sessions.loc[sessions["intended_position_count"] == 0, "record_key"]
    )
    dedicated_idle = frames["record_key"].isin(dedicated_idle_ids)
    active_session_ids = set(
        sessions.loc[sessions["intended_position_count"] > 0, "record_key"]
    )
    inactive_in_active_sessions = (
        frames["record_key"].isin(active_session_ids) & ~active
    )
    wrong_display = active & displayed & (
        frames["visual_position"] != frames["expected_position"]
    )
    correct_display = active & displayed & (
        frames["visual_position"] == frames["expected_position"]
    )
    active_frames = frames.loc[active]
    return {
        "session_count": int(len(sessions)),
        "active_session_count": int((sessions["intended_position_count"] > 0).sum()),
        "idle_session_count": int((sessions["intended_position_count"] == 0).sum()),
        "session_exact_accuracy": float(sessions["session_exact_correct"].mean()),
        "active_episode_count": int(len(episodes)),
        "active_episode_accuracy": float(episodes["correct"].mean()) if len(episodes) else None,
        "dedicated_idle_session_accuracy": float(
            sessions.loc[
                sessions["intended_position_count"] == 0, "session_exact_correct"
            ].mean()
        ),
        "dedicated_idle_frames": int(dedicated_idle.sum()),
        "dedicated_idle_false_activation_rate": float(
            frames.loc[dedicated_idle, "visual_active"].mean()
        ),
        "active_reference_frames": int(active.sum()),
        "active_visual_coverage": float(displayed[active].mean()),
        "active_frame_correct_rate_including_misses": float(correct_display[active].mean()),
        "position_accuracy_when_visual_active": float(
            (~wrong_display[active & displayed]).mean()
        ) if int((active & displayed).sum()) else None,
        "wrong_displayed_position_frames_when_active": int(wrong_display.sum()),
        "inactive_frames_inside_active_sessions": int(inactive_in_active_sessions.sum()),
        "inactive_region_visual_activation_rate": float(
            frames.loc[inactive_in_active_sessions, "visual_active"].mean()
        ),
        "force_active_frames": _force_metrics(
            active_frames["reference_force_fz_n"].to_numpy(dtype=float),
            active_frames["estimated_force_fz_n"].to_numpy(dtype=float),
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    test_root = args.test_root.resolve()
    model_path = args.model.resolve()
    peak_config_path = args.peak_config.resolve()
    runtime_config_path = args.runtime_config.resolve()
    training_manifest_path = args.training_manifest.resolve()
    training_ids, training_manifest = _training_ids(training_manifest_path)

    batch_specs = (
        ("blind", test_root / "blind", test_root / "answer"),
        ("blind2", test_root / "blind2", test_root / "answer2"),
        ("blind3", test_root / "blind3", test_root / "answer3"),
        ("blind4", test_root / "blind4", test_root / "answer4"),
    )
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for batch, blind_root, answer_root in batch_specs:
        descriptors = tuple(
            sorted(
                discover_sessions(blind_root),
                key=lambda item: (item.started_at_epoch_sec, item.session_dir.name),
            )
        )
        if not descriptors:
            raise RuntimeError(f"{batch}: no sessions")
        for descriptor in descriptors:
            if descriptor.position_label != "unlabeled":
                raise RuntimeError(f"{batch}: source is not anonymous: {descriptor.session_dir.name}")
            session_id = descriptor.session_dir.name
            record_key = f"{batch}/{session_id}"
            overlap = session_id in training_ids or str(descriptor.session_id) in training_ids
            record = {
                "batch": batch,
                "blind_root": blind_root,
                "answer_root": answer_root,
                "descriptor": descriptor,
                "session_id": session_id,
                "metadata_session_id": str(descriptor.session_id),
                "record_key": record_key,
                "training_overlap": overlap,
            }
            records.append(record)
            inventory.append(
                {
                    "batch": batch,
                    "record_key": record_key,
                    "session_id": session_id,
                    "metadata_session_id": str(descriptor.session_id),
                    "training_overlap": overlap,
                    "spectrum_sha256": _sha256(
                        descriptor.session_dir / "spectrum_timeseries.csv"
                    ),
                }
            )

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(records)
    order_rows = [
        {
            "global_order": index,
            "batch": record["batch"],
            "record_key": record["record_key"],
            "session_id": record["session_id"],
            "training_overlap": record["training_overlap"],
        }
        for index, record in enumerate(records, start=1)
    ]
    order_hash = _canonical_sha256(order_rows)
    _write_csv(output_dir / "cross_batch_shuffle_order.csv", order_rows)

    baseline_modes = (
        ("recorded", "first5_median")
        if args.baseline_mode == "both"
        else (args.baseline_mode,)
    )
    recovery_config, gate_config = _runtime_sections(runtime_config_path)
    frame_rows: list[dict[str, Any]] = []
    unscored_sessions: list[dict[str, Any]] = []
    for baseline_mode in baseline_modes:
        adapter = AllSourceOpticalForceAdapter.from_paths(
            model_path,
            peak_config_path,
            runtime_recovery_config=recovery_config,
            runtime_gate_config=gate_config,
        )
        for global_order, record in enumerate(records, start=1):
            descriptor = record["descriptor"]
            summary, wavelength, intensity = _load_session_frame_matrix(
                descriptor, expected_points=512
            )
            if baseline_mode == "recorded":
                baseline = _load_session_recorded_baseline(
                    descriptor, expected_points=512
                )
                if baseline is None:
                    raise RuntimeError(f"missing recorded baseline: {record['record_key']}")
            else:
                baseline = np.median(intensity[: min(5, len(intensity))], axis=0)
            adapter.set_baseline(wavelength, baseline)

            active_votes: list[str] = []
            for row_index, current in enumerate(intensity):
                source = summary.iloc[row_index]
                result = adapter.update(
                    wavelength,
                    current,
                    source_timestamp_sec=float(source["elapsed_time_sec"]),
                )
                if not result.get("ok"):
                    raise RuntimeError(
                        f"runtime failed: {record['record_key']} frame {row_index}"
                    )
                position = dict(result.get("position") or {})
                twin = dict(result.get("digital_twin") or {})
                contact = dict(result.get("contact") or {})
                gate = dict(result.get("runtime_contact_gate") or {})
                visual_active = bool(twin.get("visual_active"))
                visual_position = str(position.get("visual_label") or "none")
                if visual_active:
                    active_votes.append(visual_position)
                estimated_force = result.get("estimated_force_fz_n")
                frame_rows.append(
                    {
                        "baseline_mode": baseline_mode,
                        "global_order": global_order,
                        "batch": record["batch"],
                        "record_key": record["record_key"],
                        "session_id": record["session_id"],
                        "metadata_session_id": record["metadata_session_id"],
                        "training_overlap": record["training_overlap"],
                        "capture_index": int(source["capture_index"]),
                        "elapsed_time_sec": float(source["elapsed_time_sec"]),
                        "reference_force_fz_n": float(source["force_fz_n"]),
                        "contact_probability": float(contact.get("contact_probability") or 0.0),
                        "visual_active": visual_active,
                        "visual_position": visual_position,
                        "visual_position_confidence": float(position.get("visual_confidence") or 0.0),
                        "visual_position_margin": float(position.get("visual_margin") or 0.0),
                        "raw_position": str(position.get("raw_label") or "none"),
                        "formal_position": str(position.get("label") or "none"),
                        "estimated_force_fz_n": (
                            float(estimated_force) if estimated_force is not None else np.nan
                        ),
                        "baseline_distance": float(gate.get("baseline_distance") or 0.0),
                        "coupled_contact_signature_credible": bool(
                            gate.get("coupled_contact_signature_credible")
                        ),
                    }
                )
            majority = _majority(active_votes)
            unscored_sessions.append(
                {
                    "baseline_mode": baseline_mode,
                    "global_order": global_order,
                    "batch": record["batch"],
                    "record_key": record["record_key"],
                    "session_id": record["session_id"],
                    "training_overlap": record["training_overlap"],
                    "total_frames": len(intensity),
                    "visual_active_frames": len(active_votes),
                    "unscored_visual_majority": majority[0],
                    "winner_votes": majority[1],
                    "runner_votes": majority[2],
                    "winner_share": majority[3],
                }
            )
            print(
                f"[{baseline_mode} {global_order:02d}/{len(records):02d}] "
                f"{record['batch']} {record['session_id']} -> {majority[0]}"
            )

    frames_path = output_dir / "runtime_frames_unscored.csv"
    sessions_unscored_path = output_dir / "runtime_sessions_unscored.csv"
    _write_csv(frames_path, frame_rows)
    _write_csv(sessions_unscored_path, unscored_sessions)
    inference_manifest = {
        "schema_version": "touch_cross_batch_shuffled_runtime_inference_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_phase": "post_unblind_cross_batch_regression",
        "answers_loaded_during_inference": False,
        "independent_blind_evidence": False,
        "session_pool_count": len(records),
        "batch_counts": dict(Counter(record["batch"] for record in records)),
        "global_shuffle": True,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_order_sha256": order_hash,
        "adjacent_cross_batch_transitions": sum(
            records[index]["batch"] != records[index - 1]["batch"]
            for index in range(1, len(records))
        ),
        "within_session_frame_order": "chronological",
        "session_boundary_policy": "set selected baseline and reset runtime state",
        "baseline_modes": list(baseline_modes),
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_sha256": _sha256(runtime_config_path),
        "training_manifest_path": str(training_manifest_path),
        "training_manifest_sha256": _sha256(training_manifest_path),
        "training_manifest_session_count": len(training_manifest.get("sessions") or []),
        "training_overlap_by_batch": {
            batch: sum(row["batch"] == batch and row["training_overlap"] for row in inventory)
            for batch, _, _ in batch_specs
        },
        "source_inventory": inventory,
        "outputs": {
            frames_path.name: _sha256(frames_path),
            sessions_unscored_path.name: _sha256(sessions_unscored_path),
            "cross_batch_shuffle_order.csv": _sha256(
                output_dir / "cross_batch_shuffle_order.csv"
            ),
        },
    }
    inference_manifest_path = output_dir / "inference_manifest.json"
    inference_manifest_path.write_text(
        json.dumps(inference_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Load answers only after inference artifacts and hashes exist.
    answer_provenance: list[dict[str, str]] = []
    answer_mappings: dict[str, dict[str, dict[str, str]]] = {}
    for batch, blind_root, answer_root in batch_specs:
        if batch == "blind3":
            continue
        source_ids = sorted(path.name for path in blind_root.iterdir() if path.is_dir())
        mapping, provenance = _single_answer_mapping(batch, source_ids, answer_root)
        answer_mappings[batch] = mapping
        answer_provenance.extend(provenance)
    blind3_labels, blind3_summary = _load_blind3_labels(args.blind3_audit.resolve())
    blind3_lookup = {
        (str(row.session_id), int(row.capture_index)): str(row.expected_position)
        for row in blind3_labels.itertuples(index=False)
    }

    frames = pd.DataFrame(frame_rows)
    expected: list[str] = []
    intended: dict[str, tuple[str, ...]] = {}
    for record in records:
        record_key = record["record_key"]
        if record["batch"] == "blind3":
            labels = blind3_labels.loc[
                blind3_labels["session_id"].astype(str) == record["session_id"],
                "expected_position",
            ].astype(str)
            ordered: list[str] = []
            prior = "none"
            for label in labels:
                if label != "none" and label != prior:
                    ordered.append(label)
                prior = label
            intended[record_key] = tuple(ordered)
        else:
            label = answer_mappings[record["batch"]][record["session_id"]]["label"]
            intended[record_key] = () if label == "none" else (label,)

    for row in frames.itertuples(index=False):
        if row.batch == "blind3":
            key = (str(row.session_id), int(row.capture_index))
            if key not in blind3_lookup:
                raise RuntimeError(f"Blind3 label missing: {key}")
            expected.append(blind3_lookup[key])
        else:
            label = answer_mappings[str(row.batch)][str(row.session_id)]["label"]
            expected.append(
                label
                if label != "none" and float(row.reference_force_fz_n) >= args.active_force_min_n
                else "none"
            )
    frames["expected_position"] = expected

    session_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for (baseline_mode, record_key), group in frames.groupby(
        ["baseline_mode", "record_key"], sort=False
    ):
        group = group.sort_values("capture_index")
        labels = intended[record_key]
        visual_votes = group.loc[group["visual_active"], "visual_position"].astype(str)
        majority = _majority(visual_votes)
        if not labels:
            session_exact = majority[0] == "none"
        elif len(labels) == 1:
            session_exact = majority[0] == labels[0]
            active_group = group[group["expected_position"] == labels[0]]
            unit_majority = _majority(
                active_group.loc[active_group["visual_active"], "visual_position"]
            )
            episode_rows.append(
                {
                    "baseline_mode": baseline_mode,
                    "batch": group["batch"].iloc[0],
                    "record_key": record_key,
                    "session_id": group["session_id"].iloc[0],
                    "episode_index": 1,
                    "expected_position": labels[0],
                    "predicted_position": unit_majority[0],
                    "correct": unit_majority[0] == labels[0],
                    "active_frames": len(active_group),
                    "visual_active_frames": int(active_group["visual_active"].sum()),
                    "training_overlap": bool(group["training_overlap"].iloc[0]),
                }
            )
        else:
            unit_results: list[bool] = []
            for episode_index, label in enumerate(labels, start=1):
                active_group = group[group["expected_position"] == label]
                unit_majority = _majority(
                    active_group.loc[active_group["visual_active"], "visual_position"]
                )
                correct = unit_majority[0] == label
                unit_results.append(correct)
                episode_rows.append(
                    {
                        "baseline_mode": baseline_mode,
                        "batch": group["batch"].iloc[0],
                        "record_key": record_key,
                        "session_id": group["session_id"].iloc[0],
                        "episode_index": episode_index,
                        "expected_position": label,
                        "predicted_position": unit_majority[0],
                        "correct": correct,
                        "active_frames": len(active_group),
                        "visual_active_frames": int(active_group["visual_active"].sum()),
                        "training_overlap": bool(group["training_overlap"].iloc[0]),
                    }
                )
            session_exact = all(unit_results)
        session_rows.append(
            {
                "baseline_mode": baseline_mode,
                "global_order": int(group["global_order"].iloc[0]),
                "batch": group["batch"].iloc[0],
                "record_key": record_key,
                "session_id": group["session_id"].iloc[0],
                "training_overlap": bool(group["training_overlap"].iloc[0]),
                "intended_positions": "_".join(labels) if labels else "none",
                "intended_position_count": len(labels),
                "visual_majority": majority[0],
                "visual_active_frames": int(group["visual_active"].sum()),
                "total_frames": len(group),
                "session_exact_correct": session_exact,
            }
        )

    sessions = pd.DataFrame(session_rows)
    episodes = pd.DataFrame(episode_rows)
    scored_frames_path = output_dir / "scored_runtime_frames.csv"
    scored_sessions_path = output_dir / "scored_sessions.csv"
    episodes_path = output_dir / "scored_episodes.csv"
    frames.to_csv(scored_frames_path, index=False, encoding="utf-8-sig")
    sessions.to_csv(scored_sessions_path, index=False, encoding="utf-8-sig")
    episodes.to_csv(episodes_path, index=False, encoding="utf-8-sig")
    _write_csv(output_dir / "answer_mapping_provenance.csv", answer_provenance)

    summary_modes: dict[str, Any] = {}
    for baseline_mode in baseline_modes:
        mode_frames = frames[frames["baseline_mode"] == baseline_mode]
        mode_sessions = sessions[sessions["baseline_mode"] == baseline_mode]
        mode_episodes = episodes[episodes["baseline_mode"] == baseline_mode]
        by_batch = {}
        for batch, _, _ in batch_specs:
            by_batch[batch] = _summarize_scope(
                mode_frames[mode_frames["batch"] == batch],
                mode_sessions[mode_sessions["batch"] == batch],
                mode_episodes[mode_episodes["batch"] == batch],
            )
        no_training_overlap = ~mode_frames["training_overlap"]
        no_training_sessions = ~mode_sessions["training_overlap"]
        no_training_episodes = ~mode_episodes["training_overlap"]
        summary_modes[baseline_mode] = {
            "overall": _summarize_scope(mode_frames, mode_sessions, mode_episodes),
            "no_exact_training_session_overlap": _summarize_scope(
                mode_frames[no_training_overlap],
                mode_sessions[no_training_sessions],
                mode_episodes[no_training_episodes],
            ),
            "training_seen_blind3": _summarize_scope(
                mode_frames[~no_training_overlap],
                mode_sessions[~no_training_sessions],
                mode_episodes[~no_training_episodes],
            ),
            "by_batch": by_batch,
        }

    confusion_rows: list[dict[str, Any]] = []
    for baseline_mode in baseline_modes:
        subset = frames[
            (frames["baseline_mode"] == baseline_mode)
            & (frames["expected_position"] != "none")
            & frames["visual_active"]
        ]
        for truth in POSITION_ORDER:
            for prediction in POSITION_ORDER:
                confusion_rows.append(
                    {
                        "baseline_mode": baseline_mode,
                        "expected_position": truth,
                        "displayed_position": prediction,
                        "frame_count": int(
                            (
                                (subset["expected_position"] == truth)
                                & (subset["visual_position"] == prediction)
                            ).sum()
                        ),
                    }
                )
    _write_csv(output_dir / "active_frame_confusion.csv", confusion_rows)

    summary = {
        "schema_version": "touch_cross_batch_shuffled_blind_regression_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_phase": "post_unblind_cross_batch_regression",
        "independent_blind_evidence": False,
        "reason": (
            "all answers are already known and Blind3 is part of training; "
            "this audit detects deployment regressions and cross-batch domain shift"
        ),
        "session_pool_count": len(records),
        "global_shuffle": {
            "seed": args.shuffle_seed,
            "order_sha256": order_hash,
            "adjacent_cross_batch_transitions": inference_manifest[
                "adjacent_cross_batch_transitions"
            ],
            "possible_adjacent_transitions": len(records) - 1,
        },
        "training_overlap_by_batch": inference_manifest["training_overlap_by_batch"],
        "strict_evidence_notes": {
            "blind4": "independent pre-unblind evidence exists from model selection",
            "blind_and_blind2": "no exact training-session overlap but answers informed prior development",
            "blind3": "all 14 sessions are included in the deployed training dataset",
        },
        "answer_mapping_notes": {
            "answers_loaded_after_unscored_outputs_were_hashed": True,
            "blind2_known_filename_correction": BLIND2_OVERRIDES,
            "blind3_frame_labels_source": str(args.blind3_audit.resolve()),
            "blind3_frame_labels_sha256": _sha256(
                args.blind3_audit.resolve() / "frame_comparison.csv"
            ),
            "blind3_original_audit_prediction_frozen_before_answer_access": bool(
                blind3_summary.get("prediction_frozen_before_answer_access")
            ),
        },
        "baseline_modes": summary_modes,
        "outputs": {
            inference_manifest_path.name: _sha256(inference_manifest_path),
            scored_frames_path.name: _sha256(scored_frames_path),
            scored_sessions_path.name: _sha256(scored_sessions_path),
            episodes_path.name: _sha256(episodes_path),
            "answer_mapping_provenance.csv": _sha256(
                output_dir / "answer_mapping_provenance.csv"
            ),
            "active_frame_confusion.csv": _sha256(
                output_dir / "active_frame_confusion.csv"
            ),
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# TOUCH 全 Blind 跨批次乱序回放",
        "",
        f"- 共 {len(records)} 个会话；全局随机种子 `{args.shuffle_seed}`。",
        f"- 相邻 {len(records) - 1} 次切换中，{inference_manifest['adjacent_cross_batch_transitions']} 次跨越 Blind 批次。",
        "- 本次为答案已知后的部署回归，不是新的独立盲测。Blind3 已进入训练集。",
        "",
    ]
    for mode in baseline_modes:
        overall = summary_modes[mode]["overall"]
        external = summary_modes[mode]["no_exact_training_session_overlap"]
        lines.extend(
            [
                f"## {mode}",
                "",
                f"- 总体会话严格正确率：{overall['session_exact_accuracy']:.2%}；按压片段正确率：{overall['active_episode_accuracy']:.2%}。",
                f"- 纯空载误触率：{overall['dedicated_idle_false_activation_rate']:.2%}；有效按压覆盖率：{overall['active_visual_coverage']:.2%}。",
                f"- 激活后的有效按压位置准确率：{overall['position_accuracy_when_visual_active']:.2%}；错误位置帧：{overall['wrong_displayed_position_frames_when_active']}。",
                f"- 排除训练重叠会话后：会话严格正确率 {external['session_exact_accuracy']:.2%}，按压片段正确率 {external['active_episode_accuracy']:.2%}。",
                "",
                "| 批次 | 会话严格正确率 | 按压片段正确率 | 空载会话正确率 | 空载误触帧率 | 激活后位置准确率 | 错位帧 | 训练重叠 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for batch, _, _ in batch_specs:
            batch_metrics = summary_modes[mode]["by_batch"][batch]
            overlap_note = (
                "14/14 会话"
                if batch == "blind3"
                else "无精确会话重叠"
            )
            lines.append(
                "| "
                f"{batch} | "
                f"{batch_metrics['session_exact_accuracy']:.2%} | "
                f"{batch_metrics['active_episode_accuracy']:.2%} | "
                f"{batch_metrics['dedicated_idle_session_accuracy']:.2%} | "
                f"{batch_metrics['dedicated_idle_false_activation_rate']:.2%} | "
                f"{batch_metrics['position_accuracy_when_visual_active']:.2%} | "
                f"{batch_metrics['wrong_displayed_position_frames_when_active']} | "
                f"{overlap_note} |"
            )
        lines.append("")

    if "first5_median" in baseline_modes:
        first5_episodes = episodes[
            (episodes["baseline_mode"] == "first5_median")
            & ~episodes["correct"]
        ]
        lines.extend(
            [
                "## first5_median 整段位置误判",
                "",
                "下列条目按一个有效按压片段的多数显示位置判定；帧级短暂跳变不在此表中。",
                "",
                "| 批次 | 会话 | 片段 | 正确位置 | 多数显示位置 | 训练重叠 |",
                "| --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for row in first5_episodes.itertuples(index=False):
            lines.append(
                "| "
                f"{row.batch} | `{row.session_id}` | {row.episode_index} | "
                f"{row.expected_position} | {row.predicted_position} | "
                f"{'是' if row.training_overlap else '否'} |"
            )
        if first5_episodes.empty:
            lines.append("| - | - | - | - | - | - |")
        lines.extend(
            [
                "",
                "## 结论边界",
                "",
                "- Blind4 保留了揭晓答案前的独立盲测证据；本报告中的再次汇总属于部署回归。",
                "- Blind3 的 14 个会话已用于当前模型训练，其结果不能作为独立泛化准确率。",
                "- Blind1/Blind2 没有精确训练会话重叠，但答案曾参与开发判断，也不能重新包装成全新盲测。",
                "- `first5_median` 明显减少错位帧和空载误触，但 Blind2 仍存在采集域或基线域偏移，不能用总体准确率掩盖。",
                "- 下一轮若用这些历史 Blind 数据继续训练，应另采完全隔离的 Blind5，并在揭晓答案前冻结预测结果。",
                "",
            ]
        )
    (output_dir / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary_modes, ensure_ascii=False))
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
