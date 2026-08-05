"""Build leakage-safe ordinary-FBG spectrum and PX6D force datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from .baseline_relative_features import extract_baseline_relative_features


EPSILON = 1.0e-9
POSITION_ORDER = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)


@dataclass(frozen=True)
class SessionDescriptor:
    session_dir: Path
    session_id: str
    trial_id: str
    position_label: str
    qa_status: str
    finding_codes: tuple[str, ...]
    started_at_epoch_sec: float = 0.0


@dataclass(frozen=True)
class OrdinaryFbgPx6dDataset:
    wavelength_nm: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    force_fz_n: np.ndarray
    contact_target: np.ndarray
    position_target: np.ndarray
    session_id: np.ndarray
    trial_id: np.ndarray
    capture_index: np.ndarray
    elapsed_time_sec: np.ndarray
    fold_id: np.ndarray
    force_training_mask: np.ndarray
    contact_training_mask: np.ndarray
    position_training_mask: np.ndarray
    release_tail_excluded: np.ndarray
    session_manifest: tuple[dict[str, Any], ...]
    feature_sets: Mapping[str, tuple[int, ...]]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_qa_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("session_id") or ""): row
        for row in payload.get("results") or []
        if str(row.get("session_id") or "")
    }


def discover_sessions(
    capture_root: Path,
    qa_summary_path: Path | None = None,
) -> tuple[SessionDescriptor, ...]:
    qa_by_session = _read_qa_results(qa_summary_path)
    descriptors: list[SessionDescriptor] = []
    for metadata_path in sorted(capture_root.rglob("session_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_id = str(metadata.get("session_id") or metadata_path.parent.name)
        qa = qa_by_session.get(session_id, {})
        findings = tuple(
            str(item.get("code") or "")
            for item in qa.get("findings") or []
            if str(item.get("code") or "")
        )
        descriptors.append(
            SessionDescriptor(
                session_dir=metadata_path.parent,
                session_id=session_id,
                trial_id=str(metadata.get("trial_id") or ""),
                position_label=str(metadata.get("position_label") or ""),
                qa_status=str(qa.get("qa_status") or "not_audited"),
                finding_codes=findings,
                started_at_epoch_sec=float(
                    metadata.get("started_at_epoch_sec") or 0.0
                ),
            )
        )
    return tuple(descriptors)


def filter_session_descriptors(
    descriptors: Iterable[SessionDescriptor],
    data_config: Mapping[str, Any],
) -> tuple[SessionDescriptor, ...]:
    """Restrict a shared capture root to explicitly named collection batches."""
    source = tuple(descriptors)
    configured = data_config.get("include_session_id_prefixes")
    if configured is None:
        return source
    if isinstance(configured, str):
        prefixes = (configured.strip(),)
    else:
        prefixes = tuple(str(value).strip() for value in configured)
    prefixes = tuple(value for value in prefixes if value)
    if not prefixes:
        raise ValueError("include_session_id_prefixes must not be empty")
    selected = tuple(
        descriptor
        for descriptor in source
        if descriptor.session_id.startswith(prefixes)
    )
    if not selected:
        raise ValueError(
            "no sessions matched include_session_id_prefixes: "
            + ", ".join(prefixes)
        )
    return selected


def session_has_force_reference(descriptor: SessionDescriptor) -> bool:
    """Return whether a capture contains at least one finite PX6D Fz value."""

    summary_path = descriptor.session_dir / "frame_summary.csv"
    if not summary_path.is_file():
        return False
    force = pd.to_numeric(
        pd.read_csv(summary_path, usecols=("force_fz_n",))["force_fz_n"],
        errors="coerce",
    ).to_numpy(dtype=float)
    return bool(np.any(np.isfinite(force)))


def split_primary_and_challenge_sessions(
    descriptors: Iterable[SessionDescriptor],
    selection_config: Mapping[str, Any],
) -> tuple[tuple[SessionDescriptor, ...], tuple[SessionDescriptor, ...]]:
    """Select the newest requested sessions without mixing roles.

    Positions listed in ``latest_n_by_position`` contribute only their newest
    N sessions to the primary dataset. Earlier sessions from those positions
    become an isolated manual-review set. All unlisted positions remain
    primary when ``include_all_other_positions`` is enabled.
    """
    source = tuple(descriptors)
    mode = str(selection_config.get("mode") or "all_sessions")
    if mode == "all_sessions":
        return tuple(
            sorted(source, key=lambda row: (row.started_at_epoch_sec, row.session_id))
        ), ()
    if mode != "latest_n_by_position":
        raise ValueError(f"unsupported primary selection mode: {mode}")

    latest_counts = {
        str(position): int(count)
        for position, count in dict(
            selection_config.get("latest_n_by_position") or {}
        ).items()
    }
    include_other = bool(
        selection_config.get("include_all_other_positions", True)
    )
    by_position: dict[str, list[SessionDescriptor]] = defaultdict(list)
    for descriptor in source:
        by_position[descriptor.position_label].append(descriptor)

    primary: list[SessionDescriptor] = []
    challenge: list[SessionDescriptor] = []
    for position, rows in sorted(by_position.items()):
        ordered = sorted(
            rows,
            key=lambda row: (row.started_at_epoch_sec, row.session_id),
        )
        if position not in latest_counts:
            if include_other:
                primary.extend(ordered)
            else:
                challenge.extend(ordered)
            continue
        keep_count = latest_counts[position]
        if keep_count <= 0:
            raise ValueError(
                f"latest session count must be positive for {position}"
            )
        if len(ordered) < keep_count:
            raise ValueError(
                f"{position} has only {len(ordered)} sessions; "
                f"{keep_count} latest sessions were requested"
            )
        split_index = len(ordered) - keep_count
        challenge.extend(ordered[:split_index])
        primary.extend(ordered[split_index:])

    return (
        tuple(
            sorted(
                primary,
                key=lambda row: (
                    row.position_label,
                    row.started_at_epoch_sec,
                    row.session_id,
                ),
            )
        ),
        tuple(
            sorted(
                challenge,
                key=lambda row: (
                    row.position_label,
                    row.started_at_epoch_sec,
                    row.session_id,
                ),
            )
        ),
    )


def validate_strict_source_contract(
    capture_root: Path,
    descriptors: Iterable[SessionDescriptor],
    data_config: Mapping[str, Any],
) -> None:
    """Reject accidental mixing of sessions from another collection batch."""
    resolved_root = capture_root.resolve()
    expected_root_name = str(
        data_config.get("required_capture_root_name") or ""
    ).strip()
    if expected_root_name and resolved_root.name != expected_root_name:
        raise ValueError(
            "capture root does not match the strict batch contract: "
            f"expected name {expected_root_name!r}, got {resolved_root.name!r}"
        )

    required_prefix = str(
        data_config.get("required_session_id_prefix") or ""
    ).strip()
    require_qa = bool(data_config.get("require_qa_for_every_session", False))
    session_ids: list[str] = []
    errors: list[str] = []
    for descriptor in descriptors:
        session_ids.append(descriptor.session_id)
        try:
            descriptor.session_dir.resolve().relative_to(resolved_root)
        except ValueError:
            errors.append(
                f"{descriptor.session_id}: session directory is outside capture root"
            )
        if required_prefix and not descriptor.session_id.startswith(required_prefix):
            errors.append(
                f"{descriptor.session_id}: expected session prefix {required_prefix!r}"
            )
        if require_qa and descriptor.qa_status == "not_audited":
            errors.append(f"{descriptor.session_id}: no matching QA result")

    duplicates = sorted(
        session_id
        for session_id in set(session_ids)
        if session_ids.count(session_id) > 1
    )
    if duplicates:
        errors.append("duplicate session_id values: " + ", ".join(duplicates))
    if errors:
        raise ValueError(
            "strict source contract failed; historical or unaudited data may "
            "have been mixed into the batch:\n- "
            + "\n- ".join(errors)
        )


def assign_session_folds(
    descriptors: Iterable[SessionDescriptor],
    *,
    n_splits: int,
    random_seed: int,
) -> dict[str, int]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for descriptor in descriptors:
        stratum = (
            "no_contact"
            if descriptor.position_label == "unlabeled"
            else descriptor.position_label
        )
        by_stratum[stratum].append(descriptor.session_id)
    too_small = {
        label: len(session_ids)
        for label, session_ids in by_stratum.items()
        if len(session_ids) < n_splits
    }
    if too_small:
        raise ValueError(
            "each position stratum must contain at least n_splits sessions: "
            + ", ".join(f"{key}={value}" for key, value in sorted(too_small.items()))
        )
    rng = np.random.default_rng(random_seed)
    assignments: dict[str, int] = {}
    for label in sorted(by_stratum):
        session_ids = np.asarray(sorted(by_stratum[label]), dtype=str)
        order = rng.permutation(len(session_ids))
        for index, source_index in enumerate(order):
            assignments[str(session_ids[source_index])] = index % n_splits
    return assignments


def _load_session_frame_matrix(
    descriptor: SessionDescriptor,
    expected_points: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    summary = pd.read_csv(
        descriptor.session_dir / "frame_summary.csv",
        usecols=("capture_index", "elapsed_time_sec", "force_fz_n"),
    ).sort_values("capture_index")
    spectrum = pd.read_csv(
        descriptor.session_dir / "spectrum_timeseries.csv",
        usecols=(
            "capture_index",
            "point_index",
            "wavelength_nm",
            "intensity_counts",
        ),
    ).sort_values(["capture_index", "point_index"])
    frame_counts = spectrum.groupby("capture_index", sort=True).size()
    if len(frame_counts) != len(summary) or not np.all(
        frame_counts.to_numpy() == expected_points
    ):
        raise ValueError(
            f"{descriptor.session_id} does not contain exactly "
            f"{expected_points} spectrum points per frame"
        )
    capture_order = summary["capture_index"].to_numpy(dtype=int)
    spectrum_capture_order = frame_counts.index.to_numpy(dtype=int)
    if not np.array_equal(capture_order, spectrum_capture_order):
        raise ValueError(
            f"{descriptor.session_id} spectrum and summary frames are misaligned"
        )
    intensity = spectrum["intensity_counts"].to_numpy(dtype=float).reshape(
        len(summary), expected_points
    )
    wavelength_nm = (
        spectrum["wavelength_nm"]
        .to_numpy(dtype=float)
        .reshape(len(summary), expected_points)[0]
    )
    return summary, wavelength_nm, intensity


def _load_session_recorded_baseline(
    descriptor: SessionDescriptor,
    expected_points: int,
) -> np.ndarray | None:
    """Load the fixed baseline stored with the first complete spectrum frame.

    Some recordings contain a later baseline reset.  A force estimator must not
    follow that changing column because doing so can normalize away the applied
    load.  The first frame is the operator-established pre-recording reference;
    later values are intentionally ignored.
    """

    path = descriptor.session_dir / "spectrum_timeseries.csv"
    try:
        spectrum = pd.read_csv(
            path,
            usecols=(
                "capture_index",
                "point_index",
                "baseline_intensity_counts",
            ),
        ).sort_values(["capture_index", "point_index"])
    except (FileNotFoundError, ValueError):
        return None
    if spectrum.empty:
        return None

    first_capture = int(spectrum["capture_index"].iloc[0])
    first = spectrum.loc[spectrum["capture_index"] == first_capture]
    if len(first) != expected_points:
        return None
    if first["point_index"].nunique() != expected_points:
        return None

    baseline = pd.to_numeric(
        first["baseline_intensity_counts"], errors="coerce"
    ).to_numpy(dtype=float)
    if baseline.shape != (expected_points,):
        return None
    if not np.all(np.isfinite(baseline)):
        return None
    if float(np.max(np.abs(baseline))) <= EPSILON:
        return None
    return baseline


def build_dataset(
    capture_root: Path,
    config: Mapping[str, Any],
    *,
    qa_summary_path: Path | None = None,
    selection_role: str = "primary",
) -> OrdinaryFbgPx6dDataset:
    all_descriptors = discover_sessions(capture_root, qa_summary_path)
    data_config = dict(config.get("data") or {})
    all_descriptors = filter_session_descriptors(all_descriptors, data_config)
    if not all_descriptors:
        raise ValueError(f"no sessions found below {capture_root}")
    baseline_config = dict(config.get("baseline") or {})
    label_config = dict(config.get("labels") or {})
    quality_config = dict(config.get("quality") or {})
    feature_config = dict(config.get("features") or {})
    evaluation_config = dict(config.get("evaluation") or {})

    if str(data_config.get("source_policy") or "") == "strict_single_capture_root":
        validate_strict_source_contract(
            capture_root,
            all_descriptors,
            data_config,
        )
    primary_descriptors, challenge_descriptors = (
        split_primary_and_challenge_sessions(
            all_descriptors,
            dict(data_config.get("primary_selection") or {}),
        )
    )
    if selection_role == "primary":
        descriptors = primary_descriptors
        assign_folds = True
    elif selection_role == "challenge":
        descriptors = challenge_descriptors
        assign_folds = False
    elif selection_role == "all":
        descriptors = all_descriptors
        assign_folds = True
    else:
        raise ValueError(
            "selection_role must be one of: primary, challenge, all"
        )
    if not descriptors:
        raise ValueError(
            f"no sessions were selected for role {selection_role!r}"
        )
    # This builder is the strict PX6D-supervised dataset. Captures made
    # without the force sensor remain available to the fusion builder, but
    # must never be interpreted here as zero-newton supervision.
    descriptors = tuple(
        descriptor
        for descriptor in descriptors
        if session_has_force_reference(descriptor)
    )
    if not descriptors:
        raise ValueError(
            f"no force-referenced sessions were selected for role {selection_role!r}"
        )
    expected_points = int(data_config.get("expected_spectrum_points", 512))
    bin_count = int(feature_config.get("downsample_bins", 64))
    no_contact_max = float(label_config.get("no_contact_max_force_n", 0.03))
    contact_min = float(label_config.get("contact_min_force_n", 0.10))
    position_min = float(label_config.get("position_min_force_n", 0.25))
    release_warning = str(
        quality_config.get(
            "release_tail_warning_code", "release_recovery_residual"
        )
    )
    release_fraction = float(
        quality_config.get("release_tail_exclusion_fraction", 0.15)
    )
    if assign_folds:
        fold_assignments = assign_session_folds(
            descriptors,
            n_splits=int(evaluation_config.get("folds", 5)),
            random_seed=int(evaluation_config.get("random_seed", 42)),
        )
    else:
        fold_assignments = {
            descriptor.session_id: -1 for descriptor in descriptors
        }

    feature_blocks: list[np.ndarray] = []
    force_blocks: list[np.ndarray] = []
    contact_blocks: list[np.ndarray] = []
    position_blocks: list[np.ndarray] = []
    session_blocks: list[np.ndarray] = []
    trial_blocks: list[np.ndarray] = []
    capture_blocks: list[np.ndarray] = []
    elapsed_blocks: list[np.ndarray] = []
    fold_blocks: list[np.ndarray] = []
    force_mask_blocks: list[np.ndarray] = []
    contact_mask_blocks: list[np.ndarray] = []
    position_mask_blocks: list[np.ndarray] = []
    release_blocks: list[np.ndarray] = []
    session_manifest: list[dict[str, Any]] = []
    reference_wavelength: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    feature_sets: Mapping[str, tuple[int, ...]] | None = None

    for descriptor in descriptors:
        if descriptor.qa_status == "fail":
            continue
        summary, wavelength_nm, intensity = _load_session_frame_matrix(
            descriptor, expected_points
        )
        if reference_wavelength is None:
            reference_wavelength = wavelength_nm
        elif not np.allclose(reference_wavelength, wavelength_nm, atol=1.0e-9):
            intensity = np.asarray(
                [
                    np.interp(reference_wavelength, wavelength_nm, frame)
                    for frame in intensity
                ],
                dtype=float,
            )
            wavelength_nm = reference_wavelength

        force = summary["force_fz_n"].to_numpy(dtype=float)
        frame_count = len(force)
        search_count = max(
            int(baseline_config.get("minimum_frames", 5)),
            int(np.ceil(frame_count * float(baseline_config.get("search_fraction", 0.20)))),
        )
        search_count = min(frame_count, search_count)
        baseline_candidates = np.flatnonzero(
            (np.arange(frame_count) < search_count)
            & (
                force
                <= float(baseline_config.get("maximum_force_n", no_contact_max))
            )
        )
        minimum_baseline_frames = int(
            baseline_config.get("minimum_frames", 5)
        )
        baseline_mode = "initial_low_force"
        if len(baseline_candidates) < minimum_baseline_frames:
            fallback_count = min(
                search_count,
                int(baseline_config.get("fallback_lowest_force_frames", 10)),
            )
            baseline_candidates = np.argsort(force[:search_count])[:fallback_count]
            baseline_mode = "initial_lowest_force_fallback"
        baseline = np.median(intensity[baseline_candidates], axis=0)
        features, current_names, current_sets = extract_baseline_relative_features(
            intensity,
            baseline,
            wavelength_nm,
            bin_count=bin_count,
        )
        if feature_names is None:
            feature_names = current_names
            feature_sets = current_sets
        elif feature_names != current_names:
            raise ValueError("feature schema changed between sessions")

        release_excluded = np.zeros(frame_count, dtype=bool)
        if release_warning in descriptor.finding_codes:
            release_count = max(1, int(np.ceil(frame_count * release_fraction)))
            release_excluded[-release_count:] = True
        finite = np.isfinite(force) & np.all(np.isfinite(features), axis=1)
        valid = finite & ~release_excluded

        contact_target = np.full(frame_count, -1, dtype=np.int8)
        contact_target[force <= no_contact_max] = 0
        contact_target[
            (force >= contact_min) & (descriptor.position_label != "unlabeled")
        ] = 1
        position_target = np.full(frame_count, "", dtype="<U16")
        position_target[
            (force >= position_min)
            & (descriptor.position_label != "unlabeled")
        ] = descriptor.position_label

        force_training_mask = valid
        contact_training_mask = valid & (contact_target >= 0)
        position_training_mask = valid & (position_target != "")
        baseline_noise_ratio = float(
            np.median(
                np.std(intensity[baseline_candidates], axis=0)
                / np.maximum(np.abs(baseline), 1.0)
            )
        )
        session_manifest.append(
            {
                "session_id": descriptor.session_id,
                "formal_group_id": descriptor.session_id,
                "trial_id": descriptor.trial_id,
                "position_label": descriptor.position_label,
                "started_at_epoch_sec": descriptor.started_at_epoch_sec,
                "selection_role": selection_role,
                "qa_status": descriptor.qa_status,
                "finding_codes": list(descriptor.finding_codes),
                "fold_id": fold_assignments[descriptor.session_id],
                "frame_count": frame_count,
                "baseline_frame_count": int(len(baseline_candidates)),
                "baseline_mode": baseline_mode,
                "baseline_noise_ratio": baseline_noise_ratio,
                "release_tail_excluded_frames": int(np.sum(release_excluded)),
                "force_training_frames": int(np.sum(force_training_mask)),
                "contact_training_frames": int(np.sum(contact_training_mask)),
                "position_training_frames": int(np.sum(position_training_mask)),
                "force_min_n": float(np.min(force)),
                "force_max_n": float(np.max(force)),
                "session_directory": str(descriptor.session_dir),
            }
        )

        feature_blocks.append(features.astype(np.float32))
        force_blocks.append(force.astype(np.float32))
        contact_blocks.append(contact_target)
        position_blocks.append(position_target)
        session_blocks.append(
            np.full(frame_count, descriptor.session_id, dtype="<U96")
        )
        trial_blocks.append(
            np.full(frame_count, descriptor.trial_id, dtype="<U64")
        )
        capture_blocks.append(summary["capture_index"].to_numpy(dtype=np.int32))
        elapsed_blocks.append(
            summary["elapsed_time_sec"].to_numpy(dtype=np.float32)
        )
        fold_blocks.append(
            np.full(
                frame_count,
                fold_assignments[descriptor.session_id],
                dtype=np.int8,
            )
        )
        force_mask_blocks.append(force_training_mask)
        contact_mask_blocks.append(contact_training_mask)
        position_mask_blocks.append(position_training_mask)
        release_blocks.append(release_excluded)

    if not feature_blocks or reference_wavelength is None:
        raise ValueError("no QA-eligible sessions were available")
    assert feature_names is not None
    assert feature_sets is not None
    return OrdinaryFbgPx6dDataset(
        wavelength_nm=reference_wavelength.astype(np.float64),
        features=np.concatenate(feature_blocks, axis=0),
        feature_names=feature_names,
        force_fz_n=np.concatenate(force_blocks),
        contact_target=np.concatenate(contact_blocks),
        position_target=np.concatenate(position_blocks),
        session_id=np.concatenate(session_blocks),
        trial_id=np.concatenate(trial_blocks),
        capture_index=np.concatenate(capture_blocks),
        elapsed_time_sec=np.concatenate(elapsed_blocks),
        fold_id=np.concatenate(fold_blocks),
        force_training_mask=np.concatenate(force_mask_blocks),
        contact_training_mask=np.concatenate(contact_mask_blocks),
        position_training_mask=np.concatenate(position_mask_blocks),
        release_tail_excluded=np.concatenate(release_blocks),
        session_manifest=tuple(session_manifest),
        feature_sets=feature_sets,
    )


def save_dataset(
    dataset: OrdinaryFbgPx6dDataset,
    output_dir: Path,
    *,
    source_root: Path,
    config_path: Path,
    qa_summary_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "ordinary_fbg_px6d_dataset.npz"
    np.savez_compressed(
        dataset_path,
        wavelength_nm=dataset.wavelength_nm,
        features=dataset.features,
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        force_fz_n=dataset.force_fz_n,
        contact_target=dataset.contact_target,
        position_target=dataset.position_target,
        session_id=dataset.session_id,
        trial_id=dataset.trial_id,
        capture_index=dataset.capture_index,
        elapsed_time_sec=dataset.elapsed_time_sec,
        fold_id=dataset.fold_id,
        force_training_mask=dataset.force_training_mask,
        contact_training_mask=dataset.contact_training_mask,
        position_training_mask=dataset.position_training_mask,
        release_tail_excluded=dataset.release_tail_excluded,
    )
    config_payload = load_config(config_path)
    data_config = dict(config_payload.get("data") or {})
    selection_roles = {
        str(session.get("selection_role") or "unspecified")
        for session in dataset.session_manifest
    }
    if len(selection_roles) != 1:
        raise ValueError(
            "a saved dataset must contain exactly one selection role"
        )
    selection_role = next(iter(selection_roles))
    base_dataset_id = str(
        data_config.get("dataset_id")
        or "ordinary_fbg_px6d_unspecified_batch"
    )
    dataset_id = {
        "primary": base_dataset_id,
        "challenge": f"{base_dataset_id}_earlier_quarantine",
        "all": f"{base_dataset_id}_all_sessions_debug",
    }.get(selection_role, f"{base_dataset_id}_{selection_role}")
    source_inventory: list[dict[str, Any]] = []
    provenance_files = tuple(
        str(name)
        for name in data_config.get("provenance_files")
        or (
            "session_metadata.json",
            "frame_summary.csv",
            "spectrum_timeseries.csv",
        )
    )
    batch_digest = hashlib.sha256()
    for session in sorted(
        dataset.session_manifest,
        key=lambda row: str(row["session_id"]),
    ):
        session_dir = Path(str(session["session_directory"]))
        for file_name in provenance_files:
            source_path = session_dir / file_name
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"provenance source file is missing: {source_path}"
                )
            file_digest = hashlib.sha256()
            with source_path.open("rb") as source_handle:
                for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                    file_digest.update(block)
            relative_path = source_path.resolve().relative_to(
                source_root.resolve()
            ).as_posix()
            digest_hex = file_digest.hexdigest()
            size_bytes = source_path.stat().st_size
            source_inventory.append(
                {
                    "relative_path": relative_path,
                    "size_bytes": size_bytes,
                    "sha256": digest_hex,
                }
            )
            batch_digest.update(relative_path.encode("utf-8"))
            batch_digest.update(b"\0")
            batch_digest.update(str(size_bytes).encode("ascii"))
            batch_digest.update(b"\0")
            batch_digest.update(digest_hex.encode("ascii"))
            batch_digest.update(b"\n")

    inventory_path = output_dir / "ordinary_fbg_px6d_source_inventory.json"
    inventory_payload = {
        "schema_version": "ordinary_fbg_px6d_source_inventory_v1",
        "dataset_id": dataset_id,
        "selection_role": selection_role,
        "source_policy": str(data_config.get("source_policy") or ""),
        "source_root": str(source_root.resolve()),
        "hash_algorithm": "sha256",
        "source_file_count": len(source_inventory),
        "source_total_bytes": int(
            sum(row["size_bytes"] for row in source_inventory)
        ),
        "batch_content_sha256": batch_digest.hexdigest(),
        "files": source_inventory,
    }
    inventory_path.write_text(
        json.dumps(inventory_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = output_dir / "ordinary_fbg_px6d_dataset_manifest.json"
    manifest = {
        "schema_version": "ordinary_fbg_px6d_dataset_manifest_v2",
        "dataset_id": inventory_payload["dataset_id"],
        "source_policy": inventory_payload["source_policy"],
        "source_root": str(source_root.resolve()),
        "historical_data_included": False,
        "selection_role": selection_role,
        "selection_rule": dict(data_config.get("primary_selection") or {}),
        "reference_validity": (
            "formal_grouped_training"
            if selection_role == "primary"
            else "diagnostic_only_suspected_label_position_error"
        ),
        "source_inventory_path": str(inventory_path),
        "batch_content_sha256": inventory_payload["batch_content_sha256"],
        "config_path": str(config_path),
        "qa_summary_path": str(qa_summary_path) if qa_summary_path else None,
        "formal_split_requirement": (
            "grouped_by_session_id"
            if selection_role == "primary"
            else "quarantine_manual_review_only_no_model_fit"
        ),
        "formal_group_field": "session_id",
        "frame_count": int(len(dataset.force_fz_n)),
        "session_count": int(len(dataset.session_manifest)),
        "spectrum_points": int(len(dataset.wavelength_nm)),
        "feature_count": int(dataset.features.shape[1]),
        "feature_sets": {
            key: list(indices) for key, indices in dataset.feature_sets.items()
        },
        "force_training_frames": int(np.sum(dataset.force_training_mask)),
        "contact_training_frames": int(np.sum(dataset.contact_training_mask)),
        "position_training_frames": int(np.sum(dataset.position_training_mask)),
        "release_tail_excluded_frames": int(
            np.sum(dataset.release_tail_excluded)
        ),
        "sessions": list(dataset.session_manifest),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    frame_manifest = pd.DataFrame(
        {
            "session_id": dataset.session_id,
            "formal_group_id": dataset.session_id,
            "trial_id": dataset.trial_id,
            "capture_index": dataset.capture_index,
            "elapsed_time_sec": dataset.elapsed_time_sec,
            "fold_id": dataset.fold_id,
            "force_fz_n": dataset.force_fz_n,
            "contact_target": dataset.contact_target,
            "position_target": dataset.position_target,
            "force_training_eligible": dataset.force_training_mask,
            "contact_training_eligible": dataset.contact_training_mask,
            "position_training_eligible": dataset.position_training_mask,
            "release_tail_excluded": dataset.release_tail_excluded,
        }
    )
    frame_manifest.to_csv(
        output_dir / "ordinary_fbg_px6d_frame_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "source_inventory_path": str(inventory_path),
        "dataset_id": manifest["dataset_id"],
        "batch_content_sha256": manifest["batch_content_sha256"],
        "frame_count": manifest["frame_count"],
        "session_count": manifest["session_count"],
        "feature_count": manifest["feature_count"],
        "release_tail_excluded_frames": manifest[
            "release_tail_excluded_frames"
        ],
    }


__all__ = [
    "OrdinaryFbgPx6dDataset",
    "POSITION_ORDER",
    "SessionDescriptor",
    "assign_session_folds",
    "build_dataset",
    "discover_sessions",
    "extract_baseline_relative_features",
    "filter_session_descriptors",
    "session_has_force_reference",
    "load_config",
    "save_dataset",
    "split_primary_and_challenge_sessions",
    "validate_strict_source_contract",
]
