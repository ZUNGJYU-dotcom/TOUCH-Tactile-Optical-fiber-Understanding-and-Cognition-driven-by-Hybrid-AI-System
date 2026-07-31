"""Build traceable demo frames from the 2026-07-31 synchronized captures.

The generated asset intentionally stores recorded 512-point spectra without
interpolation.  Each spectrum keeps the capture index, elapsed time, source
session and synchronized PX6D Fz reference used by the demo runtime.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.discovery import discover_wavelength_order_peaks


POSITION_ORDER = [
    "NO_CONTACT",
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
]
CHANNEL_ORDER = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"]
PRIMARY_LATEST_N = {"P11": 5, "P12": 5, "P21": 5}
PEAK_DISCOVERY_RANGE_NM = (1526.5, 1561.5)


@dataclass(frozen=True)
class SessionCandidate:
    path: Path
    position: str
    capture_indices: np.ndarray
    elapsed_time_sec: np.ndarray
    force_fz_n: np.ndarray
    score: float


def _project_root() -> Path:
    return PROJECT_ROOT


def _default_data_root() -> Path:
    return _project_root().parents[1] / "data" / "new data"


def _default_output_dir() -> Path:
    return _project_root() / "bayspec_wavelength_shift_app" / "assets" / "demo"


def _position_from_name(name: str) -> str:
    match = re.search(r"_(P(?:11|12|13|21|22|23|31|32|33))_", name)
    return match.group(1) if match else "NO_CONTACT"


def _read_force_timeline(session_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capture_indices: list[int] = []
    elapsed: list[float] = []
    force: list[float] = []
    with (session_dir / "force_timeseries.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            capture_indices.append(int(row["capture_index"]))
            elapsed.append(float(row["elapsed_time_sec"]))
            force.append(max(0.0, float(row.get("force_fz_n") or 0.0)))
    if not capture_indices:
        raise ValueError(f"No force frames in {session_dir}")
    return (
        np.asarray(capture_indices, dtype=np.int32),
        np.asarray(elapsed, dtype=np.float64),
        np.asarray(force, dtype=np.float64),
    )


def _quality_score(force: np.ndarray) -> float:
    edge_count = max(5, int(round(force.size * 0.10)))
    head = float(median(force[:edge_count].tolist()))
    tail = float(median(force[-edge_count:].tolist()))
    peak = float(np.max(force))
    overshoot_penalty = max(0.0, peak - 5.6) * 2.5
    incomplete_penalty = 2.0 if peak < 4.5 else 0.0
    return abs(peak - 5.0) + 2.5 * head + 2.5 * tail + overshoot_penalty + incomplete_penalty


def _session_candidate(session_dir: Path) -> SessionCandidate:
    position = _position_from_name(session_dir.name)
    capture_indices, elapsed, force = _read_force_timeline(session_dir)
    score = 0.0 if position == "NO_CONTACT" else _quality_score(force)
    return SessionCandidate(session_dir, position, capture_indices, elapsed, force, score)


def _choose_sessions(data_root: Path) -> dict[str, SessionCandidate]:
    by_position: dict[str, list[SessionCandidate]] = {key: [] for key in POSITION_ORDER}
    for session_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if not session_dir.name.startswith("20260731_"):
            continue
        if not (session_dir / "spectrum_timeseries.csv").exists():
            continue
        if not (session_dir / "force_timeseries.csv").exists():
            continue
        candidate = _session_candidate(session_dir)
        by_position[candidate.position].append(candidate)

    selected: dict[str, SessionCandidate] = {}
    for position in POSITION_ORDER:
        candidates = sorted(by_position[position], key=lambda item: item.path.name)
        if not candidates:
            raise ValueError(f"No synchronized capture found for {position}")
        if position == "NO_CONTACT":
            selected[position] = candidates[-1]
            continue
        latest_n = PRIMARY_LATEST_N.get(position)
        eligible = candidates[-latest_n:] if latest_n else candidates
        selected[position] = min(eligible, key=lambda item: (item.score, item.path.name))
    return selected


def _sample_indices(candidate: SessionCandidate, frame_count: int) -> np.ndarray:
    positions = np.linspace(0, candidate.capture_indices.size - 1, frame_count)
    sampled = np.rint(positions).astype(np.int32)
    return candidate.capture_indices[sampled]


def _read_selected_spectra(
    session_dir: Path, selected_capture_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    wanted = {int(value) for value in selected_capture_indices.tolist()}
    points: dict[int, list[tuple[int, float, float]]] = {value: [] for value in wanted}
    with (session_dir / "spectrum_timeseries.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            capture_index = int(row["capture_index"])
            if capture_index not in wanted:
                continue
            points[capture_index].append(
                (
                    int(row["point_index"]),
                    float(row["wavelength_nm"]),
                    float(row["intensity_counts"]),
                )
            )

    wavelength_grid: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    for capture_index in selected_capture_indices.tolist():
        rows = sorted(points[int(capture_index)], key=lambda item: item[0])
        if len(rows) != 512:
            raise ValueError(
                f"Expected 512 spectrum points for {session_dir.name} frame "
                f"{capture_index}, found {len(rows)}"
            )
        wavelengths = np.asarray([row[1] for row in rows], dtype=np.float64)
        intensities = np.asarray([row[2] for row in rows], dtype=np.float32)
        if wavelength_grid is None:
            wavelength_grid = wavelengths
        elif not np.allclose(wavelength_grid, wavelengths, atol=1e-6, rtol=0.0):
            raise ValueError(f"Wavelength grid changed within {session_dir.name}")
        spectra.append(intensities)
    assert wavelength_grid is not None
    return wavelength_grid, np.stack(spectra, axis=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_peak_centers(
    wavelengths: np.ndarray, baseline_intensity: np.ndarray
) -> tuple[np.ndarray, dict]:
    discovery = discover_wavelength_order_peaks(
        wavelengths,
        baseline_intensity,
        wavelength_min_nm=PEAK_DISCOVERY_RANGE_NM[0],
        wavelength_max_nm=PEAK_DISCOVERY_RANGE_NM[1],
        expected_peak_count=len(CHANNEL_ORDER),
        smoothing_window_points=7,
        smoothing_polynomial_order=2,
        minimum_peak_distance_nm=2.5,
        minimum_absolute_prominence_counts=1000.0,
        minimum_prominence_fraction_of_range=0.03,
        minimum_peak_snr=5.0,
        minimum_edge_margin_nm=0.35,
        minimum_peak_width_nm=0.05,
        maximum_peak_width_nm=2.0,
    )
    if discovery.status not in {"pass", "pass_with_extra_candidates"}:
        raise ValueError(
            "Automatic nine-FBG discovery failed for the recorded no-contact "
            f"baseline: {discovery.status} ({len(discovery.selected_peaks)}/9 peaks)"
        )
    if len(discovery.selected_peaks) != len(CHANNEL_ORDER):
        raise ValueError(
            "Automatic nine-FBG discovery did not return exactly nine ordered peaks"
        )
    centers = np.asarray(
        [peak.refined_wavelength_nm for peak in discovery.selected_peaks],
        dtype=np.float64,
    )
    audit = {
        "method": "automatic_no_contact_wavelength_order_discovery",
        "status": discovery.status,
        "search_range_nm": list(PEAK_DISCOVERY_RANGE_NM),
        "expected_peak_count": len(CHANNEL_ORDER),
        "detected_peak_count": len(discovery.detected_peaks),
        "selected_peak_count": len(discovery.selected_peaks),
        "detection_threshold_counts": discovery.detection_threshold_counts,
        "noise_sigma_counts": discovery.noise_sigma_counts,
        "channel_assignments": [
            {
                "channel_id": channel_id,
                "wavelength_order_index": index + 1,
                "reference_wavelength_nm": peak.refined_wavelength_nm,
                "marker_wavelength_nm": peak.marker_wavelength_nm,
                "prominence_counts": peak.prominence_counts,
                "peak_snr": peak.peak_snr,
                "peak_width_nm": peak.width_nm,
            }
            for index, (channel_id, peak) in enumerate(
                zip(CHANNEL_ORDER, discovery.selected_peaks, strict=True)
            )
        ],
    }
    return centers, audit


def build_assets(data_root: Path, output_dir: Path, frame_count: int = 50) -> dict:
    selected = _choose_sessions(data_root)
    all_spectra: list[np.ndarray] = []
    all_force: list[np.ndarray] = []
    all_capture_indices: list[np.ndarray] = []
    all_elapsed: list[np.ndarray] = []
    shared_wavelengths: np.ndarray | None = None
    session_metadata: dict[str, dict] = {}

    for position in POSITION_ORDER:
        candidate = selected[position]
        capture_indices = _sample_indices(candidate, frame_count)
        wavelengths, spectra = _read_selected_spectra(candidate.path, capture_indices)
        if shared_wavelengths is None:
            shared_wavelengths = wavelengths
        elif not np.allclose(shared_wavelengths, wavelengths, atol=1e-6, rtol=0.0):
            raise ValueError(f"Wavelength grid differs for selected {position} session")

        force_by_capture = {
            int(index): float(value)
            for index, value in zip(candidate.capture_indices, candidate.force_fz_n)
        }
        elapsed_by_capture = {
            int(index): float(value)
            for index, value in zip(candidate.capture_indices, candidate.elapsed_time_sec)
        }
        sampled_force = np.asarray(
            [force_by_capture[int(index)] for index in capture_indices], dtype=np.float32
        )
        sampled_elapsed = np.asarray(
            [elapsed_by_capture[int(index)] for index in capture_indices], dtype=np.float32
        )
        all_spectra.append(spectra)
        all_force.append(sampled_force)
        all_capture_indices.append(capture_indices.astype(np.int32))
        all_elapsed.append(sampled_elapsed)
        session_metadata[position] = {
            "session_id": candidate.path.name,
            "relative_source_path": candidate.path.name,
            "source_frame_count": int(candidate.capture_indices.size),
            "sampled_frame_count": int(frame_count),
            "sampled_capture_indices": [int(value) for value in capture_indices],
            "force_peak_n": float(np.max(candidate.force_fz_n)),
            "force_start_median_n": float(median(candidate.force_fz_n[: max(5, candidate.force_fz_n.size // 10)].tolist())),
            "force_end_median_n": float(median(candidate.force_fz_n[-max(5, candidate.force_fz_n.size // 10) :].tolist())),
            "selection_score": float(candidate.score),
            "spectrum_sha256": _sha256(candidate.path / "spectrum_timeseries.csv"),
            "force_sha256": _sha256(candidate.path / "force_timeseries.csv"),
        }

    assert shared_wavelengths is not None
    spectra_array = np.stack(all_spectra, axis=0).astype(np.float32)
    force_array = np.stack(all_force, axis=0).astype(np.float32)
    capture_array = np.stack(all_capture_indices, axis=0).astype(np.int32)
    elapsed_array = np.stack(all_elapsed, axis=0).astype(np.float32)
    baseline_intensity = np.median(spectra_array[0], axis=0).astype(np.float32)
    peak_centers, peak_discovery = _baseline_peak_centers(
        shared_wavelengths,
        baseline_intensity,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "recorded_demo_20260731.npz"
    metadata_path = output_dir / "recorded_demo_20260731.json"
    np.savez_compressed(
        npz_path,
        position_order=np.asarray(POSITION_ORDER, dtype="U16"),
        channel_order=np.asarray(CHANNEL_ORDER, dtype="U8"),
        wavelength_nm=shared_wavelengths.astype(np.float64),
        spectra=spectra_array,
        force_fz_n=force_array,
        capture_index=capture_array,
        elapsed_time_sec=elapsed_array,
        baseline_intensity=baseline_intensity,
        peak_reference_wavelength_nm=peak_centers,
    )

    metadata = {
        "schema_version": "touch_recorded_demo_frames_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_date": "2026-07-31",
        "dataset_id": "ordinary_fbg_px6d_20260731_latest_primary_v1",
        "source_root": str(data_root),
        "asset_file": npz_path.name,
        "position_order": POSITION_ORDER,
        "channel_order": CHANNEL_ORDER,
        "frames_per_position": frame_count,
        "spectrum_points_per_frame": 512,
        "spectrum_semantics": "recorded real BaySpec 512-point spectrum",
        "force_semantics": "same-frame synchronized PX6D Fz reference",
        "surface_semantics": "force-scaled visual proxy driven by recorded reference frame",
        "peak_discovery": peak_discovery,
        "peak_assignment_semantics": (
            "P11-P13, P21-P23 and P31-P33 are assigned in ascending wavelength "
            "order after automatic discovery on the recorded no-contact baseline."
        ),
        "composite_action_note": (
            "Slides, broad contact and tap reuse only recorded position spectra and are "
            "identified by the runtime as recorded-reference composite demonstrations; "
            "they are not represented as dedicated measured motion trials."
        ),
        "selection_policy": {
            "P11_P12_P21": "best quality session among latest five captures",
            "other_positions": "best quality session among all 2026-07-31 captures",
            "no_contact": "latest complete no-contact capture",
        },
        "sessions": session_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata["npz_sha256"] = _sha256(npz_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"npz": str(npz_path), "metadata": str(metadata_path), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    result = build_assets(args.data_root.resolve(), args.output_dir.resolve(), args.frames)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
