"""Freeze anonymous-capture predictions from a training candidate bundle.

The command deliberately has no answer-directory argument.  It evaluates the
raw nine-FBG fingerprint, records the already-saved online response beside it,
and writes hashes before a separate unblinding/scoring step is allowed.
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

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.baseline_relative_features import (  # noqa: E402
    extract_baseline_relative_features,
)


POSITION_ORDER = (
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
EXPECTED_POINTS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-force-min-n", type=float, default=0.25)
    parser.add_argument(
        "--user-confirmed-unseen",
        action="store_true",
        help="Record that the anonymous sessions were not used for training.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty result table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _positive_contact(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value) == 1
    return str(value).strip().lower() in {"1", "true", "contact", "active"}


def _positive_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=object)
    positive = [index for index, value in enumerate(classes) if _positive_contact(value)]
    if len(positive) != 1:
        raise ValueError(f"contact estimator has ambiguous classes: {classes.tolist()}")
    return probabilities[:, positive[0]]


def _position_probability_rows(
    estimator: Any,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    labels = np.asarray(estimator.predict(features), dtype=object).astype(str)
    if not hasattr(estimator, "predict_proba"):
        return labels, np.ones(len(labels), dtype=float), [
            {label: 1.0} for label in labels
        ]
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=object).astype(str)
    if probabilities.shape != (len(features), len(classes)):
        raise ValueError("position probability shape does not match estimator classes")
    rows = [
        {label: float(value) for label, value in zip(classes, row, strict=True)}
        for row in probabilities
    ]
    confidence = np.max(probabilities, axis=1)
    return labels, confidence, rows


def _load_session(
    session_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    metadata = json.loads(
        (session_dir / "session_metadata.json").read_text(encoding="utf-8")
    )
    if str(metadata.get("position_label") or "").strip().lower() != "unlabeled":
        raise ValueError(f"{session_dir.name}: source is not anonymous")

    summary = pd.read_csv(session_dir / "frame_summary.csv").sort_values(
        "capture_index"
    )
    records: dict[int, dict[str, Any]] = {}
    with (session_dir / "synchronized_frames.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                records[int(row["capture_index"])] = row
    capture_indices = summary["capture_index"].to_numpy(dtype=int)
    if set(capture_indices.tolist()) != set(records):
        raise ValueError(f"{session_dir.name}: summary/raw frame mismatch")

    intensities: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    wavelengths: list[np.ndarray] = []
    for capture_index in capture_indices:
        spectrum = dict(records[int(capture_index)].get("spectrum") or {})
        intensities.append(np.asarray(spectrum.get("intensity_counts"), dtype=float))
        baselines.append(
            np.asarray(spectrum.get("baseline_intensity_counts"), dtype=float)
        )
        wavelengths.append(np.asarray(spectrum.get("wavelength_nm"), dtype=float))
    intensity = np.vstack(intensities)
    baseline_matrix = np.vstack(baselines)
    wavelength_matrix = np.vstack(wavelengths)
    if intensity.shape != (len(summary), EXPECTED_POINTS):
        raise ValueError(f"{session_dir.name}: expected 512 points per frame")
    if baseline_matrix.shape != intensity.shape or wavelength_matrix.shape != intensity.shape:
        raise ValueError(f"{session_dir.name}: incomplete baseline/wavelength arrays")
    if not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(baseline_matrix)):
        raise ValueError(f"{session_dir.name}: non-finite spectrum values")
    if not np.allclose(baseline_matrix, baseline_matrix[0], rtol=0.0, atol=0.0):
        raise ValueError(f"{session_dir.name}: recorded baseline changed within session")
    if not np.allclose(wavelength_matrix, wavelength_matrix[0], rtol=0.0, atol=1e-9):
        raise ValueError(f"{session_dir.name}: wavelength axis changed within session")
    return summary.reset_index(drop=True), wavelength_matrix[0], intensity, baseline_matrix[0]


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


def main() -> int:
    args = parse_args()
    blind_root = args.blind_root.resolve()
    bundle_path = args.candidate_bundle.resolve()
    training_manifest_path = args.training_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen blind predictions: {output_dir}"
        )

    session_dirs = tuple(sorted(path for path in blind_root.iterdir() if path.is_dir()))
    if not session_dirs:
        raise RuntimeError(f"no capture sessions found under {blind_root}")
    blind_ids = {path.name for path in session_dirs}
    overlap, training_session_count = _training_overlap(
        training_manifest_path, blind_ids
    )
    if overlap:
        raise RuntimeError(
            "blind session IDs overlap the training manifest: " + ", ".join(overlap)
        )

    bundle = joblib.load(bundle_path)
    if bundle.get("schema_version") != "ordinary_fbg_px6d_candidate_bundle_v1":
        raise ValueError("unsupported candidate bundle schema")
    bundle_wavelength = np.asarray(bundle["wavelength_nm"], dtype=float)
    bundle_names = tuple(np.asarray(bundle["feature_names"], dtype=str).tolist())
    task_models = dict(bundle["models"])
    if set(task_models) != {"contact", "position", "force"}:
        raise ValueError("candidate bundle must contain contact, position, and force")

    frame_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    source_fingerprints: list[dict[str, Any]] = []
    for session_order, session_dir in enumerate(session_dirs, start=1):
        summary, wavelength, intensity, baseline = _load_session(session_dir)
        if not np.allclose(wavelength, bundle_wavelength, rtol=0.0, atol=1e-6):
            raise ValueError(f"{session_dir.name}: wavelength axis differs from model")
        features, feature_names, _ = extract_baseline_relative_features(
            intensity, baseline, wavelength, bin_count=64
        )
        if feature_names != bundle_names:
            raise ValueError(f"{session_dir.name}: feature schema differs from model")

        selected: dict[str, np.ndarray] = {}
        for task, payload in task_models.items():
            indices = np.asarray(payload["feature_indices"], dtype=int)
            selected[task] = features[:, indices]

        contact_estimator = task_models["contact"]["estimator"]
        contact_probability = _positive_probability(
            contact_estimator, selected["contact"]
        )
        contact_label = np.asarray(
            [
                "contact" if _positive_contact(value) else "no_contact"
                for value in contact_estimator.predict(selected["contact"])
            ],
            dtype=object,
        )
        position_label, position_confidence, position_probabilities = (
            _position_probability_rows(
                task_models["position"]["estimator"], selected["position"]
            )
        )
        force_estimate = np.asarray(
            task_models["force"]["estimator"].predict(selected["force"]),
            dtype=float,
        )

        optical_contact = contact_label == "contact"
        optical_votes = position_label[optical_contact].tolist()
        optical_majority = _majority(optical_votes)
        reference_force = pd.to_numeric(
            summary["force_fz_n"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        raw_force = pd.to_numeric(
            summary.get("fz_raw_n", pd.Series([np.nan] * len(summary))),
            errors="coerce",
        ).to_numpy(dtype=float)
        zeroed_force = pd.to_numeric(
            summary.get("fz_zeroed_n", pd.Series([np.nan] * len(summary))),
            errors="coerce",
        ).to_numpy(dtype=float)
        signed_reference_force = pd.to_numeric(
            summary.get("reference_fz_n", pd.Series([np.nan] * len(summary))),
            errors="coerce",
        ).to_numpy(dtype=float)
        conditioned_reference_force = pd.to_numeric(
            summary.get(
                "conditioned_reference_fz_n",
                pd.Series([np.nan] * len(summary)),
            ),
            errors="coerce",
        ).to_numpy(dtype=float)
        reference_active = reference_force >= args.active_force_min_n
        reference_majority = _majority(position_label[reference_active].tolist())

        online_position = summary.get(
            "display_position_label", pd.Series(["none"] * len(summary))
        ).fillna("none").astype(str).to_numpy()
        online_active = np.isin(online_position, POSITION_ORDER)
        online_majority = _majority(online_position[online_active].tolist())
        force_error = force_estimate[reference_active] - reference_force[reference_active]

        for index, row in summary.iterrows():
            frame_rows.append(
                {
                    "session_id": session_dir.name,
                    "session_order": session_order,
                    "capture_index": int(row["capture_index"]),
                    "elapsed_time_sec": float(row["elapsed_time_sec"]),
                    "px6d_fz_raw_n": float(raw_force[index]),
                    "px6d_fz_zeroed_n": float(zeroed_force[index]),
                    "px6d_signed_reference_fz_n": float(
                        signed_reference_force[index]
                    ),
                    "px6d_conditioned_reference_fz_n": float(
                        conditioned_reference_force[index]
                    ),
                    "reference_force_fz_n": float(reference_force[index]),
                    "active_reference_frame": bool(reference_active[index]),
                    "candidate_contact_label": str(contact_label[index]),
                    "candidate_contact_probability": float(
                        contact_probability[index]
                    ),
                    "candidate_position_label": str(position_label[index]),
                    "candidate_position_confidence": float(
                        position_confidence[index]
                    ),
                    "candidate_position_probabilities": json.dumps(
                        position_probabilities[index], sort_keys=True
                    ),
                    "candidate_optical_active_position": (
                        str(position_label[index]) if optical_contact[index] else "none"
                    ),
                    "candidate_force_fz_n": float(force_estimate[index]),
                    "record_predicted_contact_label": str(
                        row.get("predicted_contact_label", "") or "none"
                    ),
                    "record_predicted_position_label": str(
                        row.get("predicted_position_label", "") or "none"
                    ),
                    "record_display_position_label": str(
                        row.get("display_position_label", "") or "none"
                    ),
                    "record_display_optical_force_n": (
                        float(row["display_optical_force_n"])
                        if pd.notna(row.get("display_optical_force_n"))
                        else None
                    ),
                }
            )

        prediction_rows.append(
            {
                "session_order": session_order,
                "session_id": session_dir.name,
                "predicted_position": optical_majority[0],
                "primary_vote_source": "candidate_position_when_candidate_contact_is_positive",
                "total_frames": len(summary),
                "candidate_contact_frames": int(np.count_nonzero(optical_contact)),
                "candidate_contact_share": float(np.mean(optical_contact)),
                "winner_votes": optical_majority[1],
                "runner_up_votes": optical_majority[2],
                "winner_share": optical_majority[3],
                "winner_margin_share": optical_majority[4],
                "candidate_vote_counts": json.dumps(
                    dict(Counter(optical_votes)), sort_keys=True
                ),
                "reference_active_frames": int(np.count_nonzero(reference_active)),
                "reference_active_position": reference_majority[0],
                "reference_active_winner_share": reference_majority[3],
                "reference_force_max_n": float(np.max(reference_force)),
                "reference_active_force_mae_n": (
                    float(np.mean(np.abs(force_error))) if len(force_error) else None
                ),
                "record_online_position": online_majority[0],
                "record_online_active_frames": int(np.count_nonzero(online_active)),
                "record_online_winner_share": online_majority[3],
            }
        )
        source_fingerprints.append(
            {
                "session_id": session_dir.name,
                "session_metadata_sha256": _sha256(
                    session_dir / "session_metadata.json"
                ),
                "frame_summary_sha256": _sha256(session_dir / "frame_summary.csv"),
                "synchronized_frames_sha256": _sha256(
                    session_dir / "synchronized_frames.jsonl"
                ),
            }
        )
        print(
            f"[{session_order:02d}/{len(session_dirs):02d}] {session_dir.name} "
            f"-> {optical_majority[0]} "
            f"({optical_majority[1]}/{len(optical_votes)} optical-contact votes)"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    frames_path = output_dir / "frames.csv"
    predictions_path = output_dir / "blind_predictions.csv"
    _write_csv(frames_path, frame_rows)
    _write_csv(predictions_path, prediction_rows)
    manifest = {
        "schema_version": "touch_blind_candidate_predictions_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer_accessed": False,
        "blind_root": str(blind_root),
        "session_count": len(session_dirs),
        "session_ids": [path.name for path in session_dirs],
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
            "baseline": "fixed baseline_intensity_counts recorded with each session",
            "feature_input": "all-nine-FBG baseline-relative 264-feature fingerprint",
            "primary_contact": "candidate contact estimator prediction; no PX6D gating",
            "primary_position": "majority candidate position among candidate-contact-positive frames",
            "reference_diagnostic": (
                f"PX6D force_fz_n >= {args.active_force_min_n:g} N; diagnostics only"
            ),
            "tie_break": "POSITION_ORDER",
            "answer_dependent_tuning": False,
        },
        "candidate_artifact": {
            "path": str(bundle_path),
            "sha256": _sha256(bundle_path),
            "schema_version": str(bundle["schema_version"]),
            "dataset_id": str(bundle["dataset_id"]),
            "selected_models": {
                task: {
                    "model_id": str(payload["model_id"]),
                    "feature_set": str(payload["feature_set"]),
                }
                for task, payload in task_models.items()
            },
        },
        "outputs": {
            "frames_csv": frames_path.name,
            "frames_sha256": _sha256(frames_path),
            "predictions_csv": predictions_path.name,
            "predictions_sha256": _sha256(predictions_path),
        },
    }
    manifest_path = output_dir / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"predictions_sha256={manifest['outputs']['predictions_sha256']}")
    print(f"frames_sha256={manifest['outputs']['frames_sha256']}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
