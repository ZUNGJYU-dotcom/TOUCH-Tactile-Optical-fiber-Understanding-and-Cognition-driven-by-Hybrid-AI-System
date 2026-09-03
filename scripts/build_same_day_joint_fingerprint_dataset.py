"""Build a cleaned same-day ordinary-FBG dataset with nine-peak fingerprints.

The five validation folds are complete acquisition blocks: regular v0.19.16,
regular v0.19.19, Blind1, Blind3, and Blind4. Blind2 is explicitly excluded.
No adjacent frames from one session can cross a validation boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.features import load_peak_windows  # noqa: E402
from src.hybrid_spectrum.joint_nine_fbg_features import (  # noqa: E402
    extract_joint_nine_fbg_features,
)
from src.hybrid_spectrum.px6d_session_dataset import POSITION_ORDER  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "E:/重要文档/实验/柔性传感/光纤/Micro-FBG/普通FBG/data"
)
DEFAULT_BLIND3_AUDIT = ROOT / "outputs/blind3_record_comparison_20260902_v01922_v1"
EXPECTED_POINTS = 512
MIN_QUIET_BASELINE_FRAMES = 3
QUIET_BASELINE_SEARCH_FRAMES = 96
NO_CONTACT_MAX_N = 0.04
CONTACT_MIN_N = 0.08
POSITION_MIN_N = 0.12
SOURCE_FOLDS = {
    "regular_v01916": 0,
    "regular_v01919": 1,
    "blind1": 2,
    "blind3": 3,
    "blind4": 4,
}
ANSWER_SUFFIX = re.compile(
    r"^(?P<session>.+)-(?P<label>uncontact|no[_-]?contact|p(?:11|12|13|21|22|23|31|32|33)(?:_p(?:11|12|13|21|22|23|31|32|33))*)$",
    re.IGNORECASE,
)
TIMESTAMP_PREFIX = re.compile(r"^(\d{8}_\d{6})_")


@dataclass(frozen=True)
class SourceSession:
    session_dir: Path
    session_id: str
    source_batch: str
    fold_id: int
    ordered_labels: tuple[str, ...]
    expected_by_capture: dict[int, str] | None
    label_source: str


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _software_version(metadata: dict[str, Any]) -> str:
    provenance = metadata.get("provenance") or {}
    start = provenance.get("start") or {}
    software = start.get("software") or {}
    return str(software.get("version") or "")


def _answer_mapping(blind_root: Path, answer_root: Path) -> dict[str, tuple[str, ...]]:
    sessions = {path.name for path in blind_root.iterdir() if path.is_dir()}
    by_timestamp: dict[str, list[str]] = {}
    for session_id in sessions:
        match = TIMESTAMP_PREFIX.match(session_id)
        if match:
            by_timestamp.setdefault(match.group(1), []).append(session_id)

    mapping: dict[str, tuple[str, ...]] = {}
    for answer_dir in sorted(path for path in answer_root.iterdir() if path.is_dir()):
        match = ANSWER_SUFFIX.match(answer_dir.name)
        if not match:
            raise ValueError(f"unrecognized answer folder: {answer_dir.name}")
        answer_session = match.group("session")
        if answer_session in sessions:
            session_id = answer_session
        else:
            timestamp = TIMESTAMP_PREFIX.match(answer_session)
            candidates = by_timestamp.get(timestamp.group(1), []) if timestamp else []
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot uniquely map answer folder {answer_dir.name}: {candidates}"
                )
            session_id = candidates[0]
        raw_label = match.group("label").lower().replace("-", "_")
        if raw_label in {"uncontact", "no_contact", "nocontact"}:
            labels: tuple[str, ...] = ()
        else:
            labels = tuple(token.upper() for token in raw_label.split("_"))
            if any(label not in POSITION_ORDER for label in labels):
                raise ValueError(f"invalid answer labels: {answer_dir.name}")
        if session_id in mapping:
            raise ValueError(f"duplicate answer mapping for {session_id}")
        mapping[session_id] = labels
    missing = sorted(sessions - set(mapping))
    extra = sorted(set(mapping) - sessions)
    if missing or extra:
        raise ValueError(f"answer mapping mismatch; missing={missing}, extra={extra}")
    return mapping


def _load_blind3_frame_labels(audit_dir: Path) -> dict[str, dict[int, str]]:
    summary_path = audit_dir / "evaluation_summary.json"
    frames_path = audit_dir / "frame_comparison.csv"
    summary = _load_json(summary_path)
    expected_hash = str((summary.get("outputs") or {}).get(frames_path.name) or "")
    if not expected_hash or _sha256(frames_path) != expected_hash:
        raise ValueError("Blind3 audited frame-label hash mismatch")
    frames = pd.read_csv(frames_path, encoding="utf-8-sig")
    required = {"session_id", "capture_index", "expected_position"}
    if missing := sorted(required - set(frames.columns)):
        raise ValueError(f"Blind3 audit missing columns: {missing}")
    output: dict[str, dict[int, str]] = {}
    for session_id, group in frames.groupby("session_id", sort=False):
        labels = {
            int(row.capture_index): str(row.expected_position or "none")
            for row in group.itertuples(index=False)
        }
        output[str(session_id)] = labels
    return output


def _regular_sources(new_data_root: Path) -> list[SourceSession]:
    output: list[SourceSession] = []
    version_to_batch = {
        "0.19.16-beta": "regular_v01916",
        "0.19.19-beta": "regular_v01919",
    }
    for metadata_path in sorted(new_data_root.glob("20260902_*/session_metadata.json")):
        metadata = _load_json(metadata_path)
        version = _software_version(metadata)
        source_batch = version_to_batch.get(version)
        if source_batch is None:
            continue
        position = str(metadata.get("position_label") or "unlabeled")
        if position == "unlabeled":
            note = str(metadata.get("operator_note") or "").strip().lower()
            if note not in {"no-contact", "no_contact", "nocontact"}:
                raise ValueError(
                    f"{metadata_path.parent.name}: unlabeled session lacks no-contact note"
                )
            labels: tuple[str, ...] = ()
        else:
            if position not in POSITION_ORDER:
                raise ValueError(f"unexpected regular position label: {position}")
            labels = (position,)
        session_id = str(metadata.get("session_id") or metadata_path.parent.name)
        output.append(
            SourceSession(
                session_dir=metadata_path.parent,
                session_id=session_id,
                source_batch=source_batch,
                fold_id=SOURCE_FOLDS[source_batch],
                ordered_labels=labels,
                expected_by_capture=None,
                label_source="metadata_position_plus_operator_no_contact",
            )
        )
    return output


def _blind_sources(
    *,
    test_root: Path,
    blind_name: str,
    answer_name: str,
    source_batch: str,
    expected_by_session: dict[str, dict[int, str]] | None = None,
) -> list[SourceSession]:
    blind_root = test_root / blind_name
    answers = _answer_mapping(blind_root, test_root / answer_name)
    output = []
    for session_dir in sorted(path for path in blind_root.iterdir() if path.is_dir()):
        session_id = session_dir.name
        expected = (
            expected_by_session.get(session_id)
            if expected_by_session is not None
            else None
        )
        if expected_by_session is not None and expected is None:
            raise ValueError(f"{session_id}: missing audited Blind3 frame labels")
        output.append(
            SourceSession(
                session_dir=session_dir,
                session_id=session_id,
                source_batch=source_batch,
                fold_id=SOURCE_FOLDS[source_batch],
                ordered_labels=answers[session_id],
                expected_by_capture=expected,
                label_source=(
                    "answer_suffix_plus_audited_frame_episode"
                    if expected is not None
                    else "answer_suffix_plus_px6d_active_interval"
                ),
            )
        )
    return output


def _load_session_frame_matrix(
    source: SourceSession,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, bool]:
    summary = pd.read_csv(source.session_dir / "frame_summary.csv", encoding="utf-8-sig")
    required_summary = {"capture_index", "elapsed_time_sec", "force_fz_n"}
    if missing := sorted(required_summary - set(summary.columns)):
        raise ValueError(f"{source.session_id}: missing summary columns {missing}")
    summary = summary.sort_values("capture_index").reset_index(drop=True)
    if summary["capture_index"].duplicated().any():
        raise ValueError(f"{source.session_id}: duplicate capture indices")

    spectrum = pd.read_csv(
        source.session_dir / "spectrum_timeseries.csv",
        usecols=(
            "capture_index",
            "wavelength_nm",
            "intensity_counts",
            "baseline_intensity_counts",
        ),
        encoding="utf-8-sig",
    )
    spectrum = spectrum.sort_values(["capture_index", "wavelength_nm"])
    counts = spectrum.groupby("capture_index", sort=True).size()
    if not bool((counts == EXPECTED_POINTS).all()):
        raise ValueError(f"{source.session_id}: spectrum frames are not 512 points")
    spectrum_captures = counts.index.to_numpy(dtype=int)
    summary_captures = summary["capture_index"].to_numpy(dtype=int)
    if not np.array_equal(spectrum_captures, summary_captures):
        raise ValueError(f"{source.session_id}: spectrum/summary capture mismatch")

    frame_count = len(summary)
    intensity = spectrum["intensity_counts"].to_numpy(dtype=float).reshape(
        frame_count, EXPECTED_POINTS
    )
    wavelength = spectrum["wavelength_nm"].to_numpy(dtype=float).reshape(
        frame_count, EXPECTED_POINTS
    )
    baseline_matrix = spectrum["baseline_intensity_counts"].to_numpy(
        dtype=float
    ).reshape(frame_count, EXPECTED_POINTS)
    if not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(wavelength)):
        raise ValueError(f"{source.session_id}: non-finite optical data")
    baseline = baseline_matrix[0]
    if not np.all(np.isfinite(baseline)) or float(np.max(np.abs(baseline))) <= 0.0:
        raise ValueError(f"{source.session_id}: invalid recorded baseline")
    baseline_changed = not np.allclose(
        baseline_matrix,
        baseline[None, :],
        rtol=0.0,
        atol=1.0e-9,
    )
    if not np.allclose(wavelength, wavelength[0][None, :], rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{source.session_id}: wavelength axis changed within session")
    return summary, wavelength[0], intensity, baseline, baseline_changed


def _clean_force_and_labels(
    source: SourceSession,
    summary: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    raw_force = pd.to_numeric(summary["force_fz_n"], errors="coerce").to_numpy(
        dtype=float
    )
    finite = np.isfinite(raw_force)
    if not bool(finite.all()):
        raise ValueError(f"{source.session_id}: missing PX6D force reference")
    raw_force = np.maximum(raw_force, 0.0)
    active_session = bool(source.ordered_labels)
    if active_session:
        initial_count = min(len(raw_force), max(12, int(np.ceil(0.03 * len(raw_force)))))
        initial = raw_force[:initial_count]
        force_offset = max(0.0, float(np.quantile(initial, 0.20)))
        cleaned_force = np.maximum(raw_force - force_offset, 0.0)
    else:
        force_offset = float(np.median(raw_force))
        cleaned_force = np.zeros_like(raw_force)

    smoothed = (
        pd.Series(cleaned_force)
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    captures = summary["capture_index"].to_numpy(dtype=int)
    if source.expected_by_capture is None:
        expected = np.full(len(summary), "none", dtype="<U16")
        if len(source.ordered_labels) == 1:
            expected[:] = source.ordered_labels[0]
        elif len(source.ordered_labels) > 1:
            raise ValueError(f"{source.session_id}: multi-position session lacks frame labels")
    else:
        try:
            expected = np.asarray(
                [source.expected_by_capture[int(index)] for index in captures],
                dtype="<U16",
            )
        except KeyError as exc:
            raise ValueError(
                f"{source.session_id}: audited frame label missing for {exc.args[0]}"
            ) from exc

    contact_target = np.full(len(summary), -1, dtype=np.int8)
    if not active_session:
        contact_target[:] = 0
    else:
        contact_target[smoothed <= NO_CONTACT_MAX_N] = 0
        contact_target[smoothed >= CONTACT_MIN_N] = 1
    contact_mask = contact_target >= 0
    position_mask = (expected != "none") & (smoothed >= POSITION_MIN_N)
    position_target = np.where(position_mask, expected, "").astype("<U16")
    return (
        cleaned_force.astype(np.float32),
        contact_target,
        contact_mask,
        position_target,
        position_mask,
        force_offset,
    )


def _select_session_quiet_baseline(
    intensity: np.ndarray,
    contact_target: np.ndarray,
    contact_mask: np.ndarray,
    *,
    requested_frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if requested_frame_count < MIN_QUIET_BASELINE_FRAMES:
        raise ValueError(
            "quiet baseline frame count must be at least "
            f"{MIN_QUIET_BASELINE_FRAMES}"
        )
    search_count = min(len(intensity), QUIET_BASELINE_SEARCH_FRAMES)
    quiet = np.flatnonzero(
        contact_mask[:search_count] & (contact_target[:search_count] == 0)
    )
    selected = quiet[:requested_frame_count]
    if len(selected) < MIN_QUIET_BASELINE_FRAMES:
        raise ValueError(
            "fewer than three force-confirmed quiet frames are available "
            "near the start of the session"
        )
    baseline = np.median(intensity[selected], axis=0)
    if not np.all(np.isfinite(baseline)) or float(np.max(np.abs(baseline))) <= 0.0:
        raise ValueError("session quiet baseline is invalid")
    return baseline.astype(float), selected.astype(np.int32)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output_dir}")
    data_root = args.data_root.resolve()
    test_root = data_root / "test data"
    blind3_labels = _load_blind3_frame_labels(args.blind3_audit.resolve())
    sources = _regular_sources(data_root / "new data")
    sources.extend(
        _blind_sources(
            test_root=test_root,
            blind_name="blind",
            answer_name="answer",
            source_batch="blind1",
        )
    )
    sources.extend(
        _blind_sources(
            test_root=test_root,
            blind_name="blind3",
            answer_name="answer3",
            source_batch="blind3",
            expected_by_session=blind3_labels,
        )
    )
    sources.extend(
        _blind_sources(
            test_root=test_root,
            blind_name="blind4",
            answer_name="answer4",
            source_batch="blind4",
        )
    )
    session_ids = [source.session_id for source in sources]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("same-day sources contain duplicate session IDs")
    source_counts = pd.Series(
        [source.source_batch for source in sources], dtype=str
    ).value_counts().to_dict()
    expected_counts = {
        "regular_v01916": 55,
        "regular_v01919": 55,
        "blind1": 14,
        "blind3": 14,
        "blind4": 23,
    }
    if source_counts != expected_counts:
        raise ValueError(f"unexpected same-day source counts: {source_counts}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    baseline_config = dict(config.get("baseline") or {})
    baseline_mode = str(baseline_config.get("mode") or "")
    if baseline_mode != "session_initial_force_confirmed_quiet_median":
        raise ValueError(f"unsupported baseline mode: {baseline_mode!r}")
    baseline_frame_count = int(baseline_config.get("frame_count") or 16)
    peak_windows = load_peak_windows(args.peak_config.resolve())
    blocks: dict[str, list[np.ndarray]] = {
        key: []
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
        )
    }
    session_manifest: list[dict[str, Any]] = []
    frame_manifests: list[pd.DataFrame] = []
    source_inventory: list[dict[str, Any]] = []
    reference_wavelength: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    feature_sets: dict[str, tuple[int, ...]] | None = None

    for ordinal, source in enumerate(sources, start=1):
        summary, wavelength, intensity, recorded_baseline, baseline_changed = (
            _load_session_frame_matrix(source)
        )
        if reference_wavelength is None:
            reference_wavelength = wavelength.copy()
        elif not np.allclose(reference_wavelength, wavelength, rtol=0.0, atol=1.0e-9):
            intensity = np.vstack(
                [np.interp(reference_wavelength, wavelength, row) for row in intensity]
            )
            recorded_baseline = np.interp(
                reference_wavelength, wavelength, recorded_baseline
            )
            wavelength = reference_wavelength
        (
            force,
            contact_target,
            contact_mask,
            position_target,
            position_mask,
            force_offset,
        ) = _clean_force_and_labels(source, summary)
        baseline, baseline_indices = _select_session_quiet_baseline(
            intensity,
            contact_target,
            contact_mask,
            requested_frame_count=baseline_frame_count,
        )
        features, current_names, current_sets = extract_joint_nine_fbg_features(
            intensity,
            baseline,
            wavelength,
            peak_windows,
            bin_count=64,
        )
        if feature_names is None:
            feature_names = current_names
            feature_sets = current_sets
        elif feature_names != current_names:
            raise ValueError("feature schema changed between sessions")

        if source.ordered_labels and int(position_mask.sum()) < 10:
            raise ValueError(f"{source.session_id}: fewer than 10 active position frames")
        frame_count = len(summary)
        trial_id = str(_load_json(source.session_dir / "session_metadata.json").get("trial_id") or "")
        joint_rms_index = current_names.index("joint_response_rms")
        joint_rms = features[:, joint_rms_index]
        position_label = (
            "unlabeled"
            if not source.ordered_labels
            else source.ordered_labels[0]
            if len(source.ordered_labels) == 1
            else "mixed_sequence"
        )
        finding_codes = []
        if baseline_changed:
            finding_codes.append("recorded_baseline_changed_after_first_frame")
        session_manifest.append(
            {
                "session_id": source.session_id,
                "formal_group_id": source.session_id,
                "trial_id": trial_id,
                "position_label": position_label,
                "ordered_position_labels": list(source.ordered_labels),
                "started_at_epoch_sec": float(
                    _load_json(source.session_dir / "session_metadata.json")[
                        "started_at_epoch_sec"
                    ]
                ),
                "source_batch": source.source_batch,
                "selection_role": "primary",
                "qa_status": "pass_cleaned_same_day",
                "finding_codes": finding_codes,
                "fold_id": source.fold_id,
                "frame_count": frame_count,
                "baseline_frame_count": int(len(baseline_indices)),
                "baseline_capture_indices": baseline_indices.tolist(),
                "baseline_mode": baseline_mode,
                "baseline_sha256": hashlib.sha256(
                    baseline.astype(np.float64).tobytes()
                ).hexdigest(),
                "recorded_baseline_sha256": hashlib.sha256(
                    recorded_baseline.astype(np.float64).tobytes()
                ).hexdigest(),
                "recorded_baseline_changed": baseline_changed,
                "force_zero_offset_n": force_offset,
                "force_training_frames": frame_count,
                "contact_training_frames": int(contact_mask.sum()),
                "position_training_frames": int(position_mask.sum()),
                "force_min_n": float(np.min(force)),
                "force_max_n": float(np.max(force)),
                "joint_response_rms_p99": float(np.quantile(joint_rms, 0.99)),
                "session_directory": str(source.session_dir),
                "label_source": source.label_source,
            }
        )
        captures = summary["capture_index"].to_numpy(dtype=np.int32)
        elapsed = summary["elapsed_time_sec"].to_numpy(dtype=np.float32)
        blocks["features"].append(features.astype(np.float32))
        blocks["force_fz_n"].append(force)
        blocks["contact_target"].append(contact_target)
        blocks["position_target"].append(position_target)
        blocks["session_id"].append(
            np.full(frame_count, source.session_id, dtype="<U96")
        )
        blocks["trial_id"].append(np.full(frame_count, trial_id, dtype="<U64"))
        blocks["capture_index"].append(captures)
        blocks["elapsed_time_sec"].append(elapsed)
        blocks["fold_id"].append(
            np.full(frame_count, source.fold_id, dtype=np.int8)
        )
        blocks["force_training_mask"].append(np.ones(frame_count, dtype=bool))
        blocks["contact_training_mask"].append(contact_mask)
        blocks["position_training_mask"].append(position_mask)
        blocks["release_tail_excluded"].append(np.zeros(frame_count, dtype=bool))
        frame_manifests.append(
            pd.DataFrame(
                {
                    "session_id": source.session_id,
                    "source_batch": source.source_batch,
                    "capture_index": captures,
                    "elapsed_time_sec": elapsed,
                    "fold_id": source.fold_id,
                    "force_fz_n": force,
                    "contact_target": contact_target,
                    "position_target": position_target,
                    "contact_training_eligible": contact_mask,
                    "position_training_eligible": position_mask,
                }
            )
        )
        for name in (
            "session_metadata.json",
            "frame_summary.csv",
            "spectrum_timeseries.csv",
        ):
            path = source.session_dir / name
            source_inventory.append(
                {
                    "source_batch": source.source_batch,
                    "session_id": source.session_id,
                    "file_name": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        print(
            f"[{ordinal:03d}/{len(sources):03d}] {source.source_batch} "
            f"{source.session_id} frames={frame_count} position={int(position_mask.sum())}"
        )

    if reference_wavelength is None or feature_names is None or feature_sets is None:
        raise RuntimeError("same-day dataset is empty")
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in blocks.items()}
    inventory_payload = {
        "schema_version": "ordinary_fbg_same_day_source_inventory_v1",
        "dataset_id": args.dataset_id,
        "excluded_sources": ["blind2"],
        "files": source_inventory,
    }
    batch_hash = _canonical_sha256(inventory_payload)
    manifest = {
        "schema_version": "ordinary_fbg_same_day_joint_dataset_v1",
        "dataset_id": args.dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_role": "primary",
        "source_policy": "same_day_regular_plus_blind1_blind3_blind4_excluding_blind2",
        "historical_data_included": False,
        "training_date": "20260902",
        "formal_split_strategy": "leave_complete_acquisition_batch_out",
        "formal_group_field": "session_id",
        "random_frame_split_used": False,
        "selection_rule": "all_cleaned_same_day_sessions_except_blind2",
        "batch_content_sha256": batch_hash,
        "session_count": len(session_manifest),
        "frame_count": int(len(arrays["features"])),
        "spectrum_points": int(len(reference_wavelength)),
        "feature_count": int(arrays["features"].shape[1]),
        "feature_sets": {
            name: list(indices) for name, indices in feature_sets.items()
        },
        "force_training_frames": int(arrays["force_training_mask"].sum()),
        "contact_training_frames": int(arrays["contact_training_mask"].sum()),
        "position_training_frames": int(arrays["position_training_mask"].sum()),
        "release_tail_excluded_frames": 0,
        "source_batch_counts": source_counts,
        "source_fold_mapping": SOURCE_FOLDS,
        "excluded_source_batches": ["blind2"],
        "label_cleaning_contract": {
            "idle_session_policy": "operator_or_answer_no_contact_forces_all_frames_to_zero",
            "active_force_zeroing": "subtract_initial_20th_percentile_then_clip_at_zero",
            "no_contact_max_n": NO_CONTACT_MAX_N,
            "contact_min_n": CONTACT_MIN_N,
            "position_min_n": POSITION_MIN_N,
            "ambiguous_contact_frames_excluded": True,
        },
        "feature_contract": {
            "input_scope": "all_512_spectrum_points_plus_all_nine_fbg_joint_fingerprint",
            "single_channel_contact_evidence_sufficient": False,
            "nine_fbg_joint_feature_count": len(feature_sets["nine_fbg_joint_75"]),
        },
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "sessions": session_manifest,
    }

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "ordinary_fbg_px6d_dataset.npz",
        wavelength_nm=reference_wavelength,
        feature_names=np.asarray(feature_names, dtype=str),
        **arrays,
    )
    (output_dir / "ordinary_fbg_px6d_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ordinary_fbg_px6d_source_inventory.json").write_text(
        json.dumps(inventory_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.concat(frame_manifests, ignore_index=True).to_csv(
        output_dir / "ordinary_fbg_px6d_frame_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "ok": True,
        "dataset_id": args.dataset_id,
        "output_dir": str(output_dir),
        "session_count": len(session_manifest),
        "frame_count": manifest["frame_count"],
        "feature_count": manifest["feature_count"],
        "source_batch_counts": source_counts,
        "force_training_frames": manifest["force_training_frames"],
        "contact_training_frames": manifest["contact_training_frames"],
        "position_training_frames": manifest["position_training_frames"],
        "batch_content_sha256": batch_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--blind3-audit", type=Path, default=DEFAULT_BLIND3_AUDIT)
    parser.add_argument(
        "--peak-config",
        type=Path,
        default=ROOT / "config/hybrid_spectrum_channels.yaml",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    return parser.parse_args()


def main() -> int:
    result = build_dataset(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
