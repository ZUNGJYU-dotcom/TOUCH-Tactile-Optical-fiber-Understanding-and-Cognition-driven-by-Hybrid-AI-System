"""On-disk format for grouped full-spectrum capture segments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VALID_LABELS = (
    "no_contact",
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
)


@dataclass
class SpectrumSegment:
    path: Path
    metadata: dict[str, Any]
    wavelength_nm: np.ndarray
    intensity_counts: np.ndarray
    frame_ids: np.ndarray
    timestamps: np.ndarray
    baseline_spectrum: np.ndarray | None
    integrity_verified: bool = False
    segment_fingerprint_sha256: str | None = None

    @property
    def trial_id(self) -> str:
        return str(self.metadata["trial_id"])

    @property
    def segment_id(self) -> str:
        return str(self.metadata["segment_id"])

    @property
    def label(self) -> str:
        return str(self.metadata["label"])

    @property
    def phase(self) -> str:
        return str(self.metadata["phase"])

    @property
    def training_eligible(self) -> bool:
        return bool(self.metadata.get("training_eligible", False))


def _validate_arrays(
    wavelength_nm: np.ndarray,
    intensity_counts: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    if wavelength_nm.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    if intensity_counts.ndim != 2:
        raise ValueError("intensity_counts must have shape [frames, spectrum_points]")
    if intensity_counts.shape[1] != wavelength_nm.size:
        raise ValueError("spectrum width does not match wavelength grid")
    if intensity_counts.shape[0] != frame_ids.size or frame_ids.size != timestamps.size:
        raise ValueError("frame metadata length does not match spectrum frame count")
    if intensity_counts.shape[0] == 0:
        raise ValueError("capture segment has no frames")
    if not np.all(np.isfinite(wavelength_nm)) or not np.all(np.isfinite(intensity_counts)):
        raise ValueError("capture segment contains non-finite spectrum values")
    if np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("wavelength grid must be strictly increasing")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("frame_ids must be strictly increasing")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_durable(path: Path, content: str, encoding: str = "utf-8") -> None:
    with path.open("w", encoding=encoding, newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _segment_fingerprint(file_hashes: dict[str, str]) -> str:
    canonical = "\n".join(
        f"{name}:{file_hashes[name]}" for name in sorted(file_hashes)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_integrity_manifest(segment_dir: Path) -> tuple[bool, str | None]:
    manifest_path = segment_dir / "segment_manifest.json"
    if not manifest_path.exists():
        return False, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != "hybrid_spectrum_segment_manifest_v1":
        raise ValueError(f"unsupported segment manifest schema: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"segment manifest has no file inventory: {manifest_path}")
    verified_hashes: dict[str, str] = {}
    for name, expected in files.items():
        if not isinstance(expected, dict):
            raise ValueError(f"invalid manifest entry for {name}: {manifest_path}")
        file_path = segment_dir / name
        if not file_path.exists():
            raise ValueError(f"manifest file is missing: {file_path}")
        expected_size = int(expected.get("size_bytes", -1))
        if file_path.stat().st_size != expected_size:
            raise ValueError(f"manifest size mismatch: {file_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != expected.get("sha256"):
            raise ValueError(f"manifest SHA256 mismatch: {file_path}")
        verified_hashes[name] = actual_hash
    fingerprint = _segment_fingerprint(verified_hashes)
    if fingerprint != manifest.get("segment_fingerprint_sha256"):
        raise ValueError(f"segment fingerprint mismatch: {manifest_path}")
    return True, fingerprint


def _validate_stored_contract(segment: SpectrumSegment) -> None:
    metadata = segment.metadata
    required = ("trial_id", "segment_id", "label", "phase")
    missing = [name for name in required if not str(metadata.get(name, "")).strip()]
    if missing:
        raise ValueError(f"segment metadata is missing required fields: {', '.join(missing)}")
    if segment.label not in VALID_LABELS:
        raise ValueError(f"stored segment has unsupported label: {segment.label}")
    expected_values = {
        "num_frames": int(segment.intensity_counts.shape[0]),
        "spectrum_points": int(segment.wavelength_nm.size),
    }
    for field, expected in expected_values.items():
        if field in metadata and int(metadata[field]) != expected:
            raise ValueError(f"stored {field} does not match spectrum archive")
    baseline_available = segment.baseline_spectrum is not None
    if "baseline_spectrum_available" in metadata:
        if bool(metadata["baseline_spectrum_available"]) != baseline_available:
            raise ValueError("stored baseline availability does not match spectrum archive")
    if "frame_id_start" in metadata and int(metadata["frame_id_start"]) != int(
        segment.frame_ids[0]
    ):
        raise ValueError("stored frame_id_start does not match spectrum archive")
    if "frame_id_stop" in metadata and int(metadata["frame_id_stop"]) != int(
        segment.frame_ids[-1]
    ):
        raise ValueError("stored frame_id_stop does not match spectrum archive")
    frames_path = segment.path / "frames.csv"
    if not frames_path.exists():
        if segment.integrity_verified:
            raise ValueError(f"verified segment has no frames.csv: {segment.path}")
        return
    with frames_path.open("r", encoding="utf-8-sig", newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    if len(frame_rows) != segment.frame_ids.size:
        raise ValueError("frames.csv row count does not match spectrum archive")
    for index, row in enumerate(frame_rows):
        if int(row["frame_index"]) != index:
            raise ValueError("frames.csv frame_index is not contiguous")
        if int(row["frame_id"]) != int(segment.frame_ids[index]):
            raise ValueError("frames.csv frame_id does not match spectrum archive")
        if not np.isclose(
            float(row["timestamp"]),
            float(segment.timestamps[index]),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("frames.csv timestamp does not match spectrum archive")


def save_segment(
    output_root: Path,
    metadata: dict[str, Any],
    wavelength_nm: Iterable[float],
    intensity_counts: Iterable[Iterable[float]],
    frame_ids: Iterable[int],
    timestamps: Iterable[float],
    baseline_spectrum: Iterable[float] | None = None,
) -> Path:
    """Save one acquisition segment without mixing trial groups."""

    label = str(metadata.get("label", ""))
    if label not in VALID_LABELS:
        raise ValueError(f"unsupported label: {label}")
    trial_id = str(metadata.get("trial_id", "")).strip()
    segment_id = str(metadata.get("segment_id", "")).strip()
    if not trial_id or not segment_id:
        raise ValueError("trial_id and segment_id are required")

    wavelength = np.asarray(list(wavelength_nm), dtype=float)
    spectra = np.asarray(list(intensity_counts), dtype=float)
    ids = np.asarray(list(frame_ids), dtype=np.int64)
    times = np.asarray(list(timestamps), dtype=float)
    _validate_arrays(wavelength, spectra, ids, times)

    baseline = None if baseline_spectrum is None else np.asarray(list(baseline_spectrum), dtype=float)
    if baseline is not None and baseline.shape != wavelength.shape:
        raise ValueError("baseline spectrum must match wavelength grid")
    if baseline is not None and not np.all(np.isfinite(baseline)):
        raise ValueError("baseline spectrum contains non-finite values")

    trial_dir = output_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = trial_dir / segment_id
    if segment_dir.exists():
        raise FileExistsError(f"segment already exists and will not be overwritten: {segment_dir}")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{segment_id}.incomplete-", dir=str(trial_dir))
    )
    archive_payload: dict[str, np.ndarray] = {
        "wavelength_nm": wavelength,
        "intensity_counts": spectra,
        "frame_ids": ids,
        "timestamps": times,
    }
    if baseline is not None:
        archive_payload["baseline_spectrum"] = baseline
    try:
        archive_path = staging_dir / "spectra.npz"
        with archive_path.open("wb") as archive_handle:
            np.savez_compressed(archive_handle, **archive_payload)
            archive_handle.flush()
            os.fsync(archive_handle.fileno())

        stored_metadata = dict(metadata)
        stored_metadata.update(
            {
                "schema_version": "1.1",
                "spectrum_points": int(wavelength.size),
                "num_frames": int(spectra.shape[0]),
                "wavelength_min_nm": float(wavelength[0]),
                "wavelength_max_nm": float(wavelength[-1]),
                "baseline_spectrum_available": baseline is not None,
                "group_key": "trial_id",
                "random_frame_split_allowed": False,
                "integrity_manifest_available": True,
            }
        )
        _write_text_durable(
            staging_dir / "metadata.json",
            json.dumps(stored_metadata, indent=2, ensure_ascii=False) + "\n",
        )
        frames_path = staging_dir / "frames.csv"
        with frames_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["frame_index", "frame_id", "timestamp"]
            )
            writer.writeheader()
            for index, (frame_id, timestamp) in enumerate(zip(ids, times, strict=True)):
                writer.writerow(
                    {
                        "frame_index": index,
                        "frame_id": int(frame_id),
                        "timestamp": f"{float(timestamp):.9f}",
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())

        file_names = ("spectra.npz", "metadata.json", "frames.csv")
        file_hashes = {name: sha256_file(staging_dir / name) for name in file_names}
        manifest = {
            "manifest_schema_version": "hybrid_spectrum_segment_manifest_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_id": trial_id,
            "segment_id": segment_id,
            "files": {
                name: {
                    "sha256": file_hashes[name],
                    "size_bytes": int((staging_dir / name).stat().st_size),
                }
                for name in file_names
            },
            "segment_fingerprint_sha256": _segment_fingerprint(file_hashes),
        }
        _write_text_durable(
            staging_dir / "segment_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        staging_dir.rename(segment_dir)
        return segment_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def load_segment(segment_path: Path) -> SpectrumSegment:
    segment_dir = segment_path.parent if segment_path.name == "metadata.json" else segment_path
    metadata_path = segment_dir / "metadata.json"
    archive_path = segment_dir / "spectra.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"segment metadata is missing: {metadata_path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"segment spectrum archive is missing: {archive_path}")
    integrity_verified, segment_fingerprint = _verify_integrity_manifest(segment_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if bool(metadata.get("integrity_manifest_available", False)) and not integrity_verified:
        raise ValueError(f"segment integrity manifest is required but missing: {segment_dir}")
    with np.load(archive_path) as archive:
        baseline = (
            np.asarray(archive["baseline_spectrum"], dtype=float)
            if "baseline_spectrum" in archive.files
            else None
        )
        segment = SpectrumSegment(
            path=segment_dir,
            metadata=metadata,
            wavelength_nm=np.asarray(archive["wavelength_nm"], dtype=float),
            intensity_counts=np.asarray(archive["intensity_counts"], dtype=float),
            frame_ids=np.asarray(archive["frame_ids"], dtype=np.int64),
            timestamps=np.asarray(archive["timestamps"], dtype=float),
            baseline_spectrum=baseline,
            integrity_verified=integrity_verified,
            segment_fingerprint_sha256=segment_fingerprint,
        )
    _validate_arrays(
        segment.wavelength_nm,
        segment.intensity_counts,
        segment.frame_ids,
        segment.timestamps,
    )
    if segment.baseline_spectrum is not None:
        if segment.baseline_spectrum.shape != segment.wavelength_nm.shape:
            raise ValueError("stored baseline spectrum does not match wavelength grid")
        if not np.all(np.isfinite(segment.baseline_spectrum)):
            raise ValueError("stored baseline spectrum contains non-finite values")
    _validate_stored_contract(segment)
    return segment


def load_segments(
    root: Path,
    include_ineligible: bool = False,
    strict: bool = True,
) -> list[SpectrumSegment]:
    segments: list[SpectrumSegment] = []
    if not root.exists():
        return segments
    for metadata_path in sorted(root.rglob("metadata.json")):
        try:
            segment = load_segment(metadata_path)
            if not include_ineligible and not segment.training_eligible:
                continue
            segments.append(segment)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            if strict:
                raise
    return segments


def validate_segment_collection(segments: Iterable[SpectrumSegment]) -> dict[str, Any]:
    """Validate cross-segment trial/session and baseline-reference integrity."""

    segment_list = list(segments)
    if not segment_list:
        raise ValueError("segment collection is empty")
    by_trial: dict[str, list[SpectrumSegment]] = defaultdict(list)
    seen_segment_keys: set[tuple[str, str]] = set()
    for segment in segment_list:
        key = (segment.trial_id, segment.segment_id)
        if key in seen_segment_keys:
            raise ValueError(
                f"duplicate trial/segment identity: {segment.trial_id}/{segment.segment_id}"
            )
        seen_segment_keys.add(key)
        by_trial[segment.trial_id].append(segment)

    baseline_reference_count = 0
    context_fields = (
        "data_source",
        "device_id",
        "integration_ms",
        "spectrum_peak_profile",
        "backend_session_started_at_epoch",
    )
    for trial_id, trial_segments in by_trial.items():
        ordered = sorted(trial_segments, key=lambda segment: float(segment.timestamps[0]))
        reference_grid = ordered[0].wavelength_nm
        reference_context = {
            field: ordered[0].metadata.get(field) for field in context_fields
        }
        missing_context = [
            field
            for field, value in reference_context.items()
            if value is None or str(value).strip() == ""
        ]
        if missing_context:
            raise ValueError(
                f"trial {trial_id} is missing acquisition context: {', '.join(missing_context)}"
            )

        baseline_by_fingerprint: dict[str, SpectrumSegment] = {}
        for segment in ordered:
            if segment.wavelength_nm.shape != reference_grid.shape or not np.allclose(
                segment.wavelength_nm, reference_grid, atol=1.0e-9, rtol=0.0
            ):
                raise ValueError(f"trial {trial_id} changes wavelength grid between segments")
            for field in context_fields:
                expected = reference_context[field]
                actual = segment.metadata.get(field)
                if field in {"integration_ms", "backend_session_started_at_epoch"}:
                    try:
                        matches = bool(
                            np.isclose(float(actual), float(expected), rtol=0.0, atol=1.0e-6)
                        )
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = str(actual) == str(expected)
                if not matches:
                    raise ValueError(
                        f"trial {trial_id} changes {field} between acquisition segments"
                    )
            if segment.phase == "no_contact":
                if segment.label != "no_contact":
                    raise ValueError(
                        f"trial {trial_id}/{segment.segment_id} has inconsistent no_contact phase label"
                    )
                if not segment.segment_fingerprint_sha256:
                    raise ValueError(
                        f"trial {trial_id}/{segment.segment_id} baseline fingerprint is missing"
                    )
                baseline_by_fingerprint[segment.segment_fingerprint_sha256] = segment
            elif segment.label == "no_contact":
                raise ValueError(
                    f"trial {trial_id}/{segment.segment_id} uses no_contact label outside no_contact phase"
                )

        for previous, current in zip(ordered, ordered[1:], strict=False):
            if int(current.frame_ids[0]) <= int(previous.frame_ids[-1]):
                raise ValueError(
                    f"trial {trial_id} has overlapping or reversed frame ranges between "
                    f"{previous.segment_id} and {current.segment_id}"
                )
            if float(current.timestamps[0]) <= float(previous.timestamps[-1]):
                raise ValueError(
                    f"trial {trial_id} has overlapping or reversed timestamps between "
                    f"{previous.segment_id} and {current.segment_id}"
                )

        for segment in ordered:
            if segment.phase not in {"contact", "release"}:
                continue
            reference_fingerprint = str(
                segment.metadata.get(
                    "baseline_reference_segment_fingerprint_sha256", ""
                )
                or ""
            )
            baseline_segment = baseline_by_fingerprint.get(reference_fingerprint)
            if baseline_segment is None:
                raise ValueError(
                    f"trial {trial_id}/{segment.segment_id} does not reference a verified "
                    "no-contact baseline from the same trial"
                )
            if segment.baseline_spectrum is None or baseline_segment.baseline_spectrum is None:
                raise ValueError(
                    f"trial {trial_id}/{segment.segment_id} is missing stored baseline spectrum"
                )
            if not np.allclose(
                segment.baseline_spectrum,
                baseline_segment.baseline_spectrum,
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    f"trial {trial_id}/{segment.segment_id} baseline spectrum does not match "
                    "its referenced baseline segment"
                )
            baseline_reference_count += 1

    return {
        "status": "pass",
        "trial_count": len(by_trial),
        "segment_count": len(segment_list),
        "baseline_reference_count": baseline_reference_count,
        "validated_context_fields": list(context_fields),
        "frame_ranges_non_overlapping": True,
        "timestamps_non_overlapping": True,
        "wavelength_grid_consistent_within_trial": True,
        "baseline_references_verified_within_trial": True,
    }
