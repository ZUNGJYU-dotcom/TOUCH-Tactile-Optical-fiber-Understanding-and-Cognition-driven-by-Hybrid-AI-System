"""Evaluate unlabeled Sense DAT files with the deployed temporal spectrum model.

The script deliberately does not infer ground truth from filenames. Predictions,
source hashes, and the model hash are frozen first so labels can be revealed and
scored later without tuning against the blind set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_opening


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    DynamicSequenceRecord,
    extract_dynamic_frame_features,
    load_dynamic_config,
    load_reference_wavelength_grid,
)
from src.hybrid_spectrum.dynamic_temporal_features import (  # noqa: E402
    temporal_summary_features,
)
from src.hybrid_spectrum.features import load_peak_windows  # noqa: E402
from src.hybrid_spectrum.sense_fast_dat import read_sense_fast_dat  # noqa: E402


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
RESPONSE_ORDER = ("light", "normal", "hard")
RESPONSE_COLORS = {
    "no_contact": "#A9B3BC",
    "light": "#56B4E9",
    "normal": "#E6A63B",
    "hard": "#C65D57",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "dynamic_temporal_shadow_candidate_v3_compact_runtime_pos240.joblib",
    )
    parser.add_argument(
        "--sequence-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=768)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_probability(
    model: Any,
    values: np.ndarray,
    labels: Iterable[str],
) -> np.ndarray:
    ordered = tuple(str(label) for label in labels)
    raw = np.asarray(model.predict_proba(values), dtype=float)
    aligned = np.zeros((len(values), len(ordered)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        text = str(label)
        if text in ordered:
            aligned[:, ordered.index(text)] = raw[:, source_index]
    return aligned


def positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=float)
    classes = [str(value) for value in model.classes_]
    for candidate in ("1", "True", "release", "positive"):
        if candidate in classes:
            return raw[:, classes.index(candidate)]
    if raw.shape[1] == 2:
        return raw[:, 1]
    return np.max(raw, axis=1)


def temporal_probabilities(
    frame_features: np.ndarray,
    bundle: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    time_steps = int(bundle["time_steps"])
    window_count = len(frame_features) - time_steps + 1
    if window_count <= 0:
        raise ValueError("blind file is shorter than the temporal model window")
    labels = bundle["label_order"]
    models = bundle["models"]
    contact_blocks: list[np.ndarray] = []
    position_blocks: list[np.ndarray] = []
    response_blocks: list[np.ndarray] = []
    release_blocks: list[np.ndarray] = []
    for start in range(0, window_count, batch_size):
        stop = min(window_count, start + batch_size)
        windows = np.stack(
            [frame_features[index : index + time_steps] for index in range(start, stop)],
            axis=0,
        )
        summary = temporal_summary_features(windows)
        contact_blocks.append(
            aligned_probability(models["contact_extra_trees"], summary, labels["contact"])
        )
        position_blocks.append(
            aligned_probability(models["position_factorized"], summary, labels["position"])
        )
        response_blocks.append(
            0.5
            * aligned_probability(
                models["response_extra_trees"], summary, labels["response_level"]
            )
            + 0.5
            * aligned_probability(
                models["response_rbf_svm"], summary, labels["response_level"]
            )
        )
        release_model = models.get("release_event_extra_trees")
        release_blocks.append(
            positive_probability(release_model, summary)
            if release_model is not None
            else np.zeros(len(summary), dtype=float)
        )
    return {
        "contact": np.vstack(contact_blocks),
        "position": np.vstack(position_blocks),
        "response": np.vstack(response_blocks),
        "release": np.concatenate(release_blocks),
    }


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def sustained_label_path(labels: np.ndarray, active: np.ndarray) -> str:
    path: list[str] = []
    for start, end in runs(active):
        index = start
        while index < end:
            label = str(labels[index])
            next_index = index + 1
            while next_index < end and str(labels[next_index]) == label:
                next_index += 1
            if next_index - index >= 5 and (not path or path[-1] != label):
                path.append(label)
            index = next_index
    return ">".join(path) if path else "none"


def weighted_vote(
    probabilities: np.ndarray,
    weights: np.ndarray,
    labels: tuple[str, ...],
) -> tuple[str | None, float | None, dict[str, float]]:
    if probabilities.size == 0 or not np.any(weights > 0.0):
        return None, None, {label: 0.0 for label in labels}
    score = np.sum(probabilities * weights[:, None], axis=0)
    denominator = max(float(np.sum(score)), 1.0e-12)
    normalized = score / denominator
    index = int(np.argmax(normalized))
    return (
        labels[index],
        float(normalized[index]),
        {label: float(normalized[i]) for i, label in enumerate(labels)},
    )


def terminal_artifact_mask(response_score: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(response_score), dtype=bool)
    if len(response_score) < 30:
        return mask
    previous = response_score[-26:-1]
    terminal = float(response_score[-1])
    reference = max(float(np.median(previous)), 1.0e-6)
    if terminal >= max(0.35, 3.0 * reference) and response_score[-2] < 0.60 * terminal:
        mask[-1] = True
    return mask


def plot_file(
    path: Path,
    *,
    time_axis: np.ndarray,
    response_score: np.ndarray,
    window_time: np.ndarray,
    contact_probability: np.ndarray,
    position_probability: np.ndarray,
    response_probability: np.ndarray,
    sustained_contact: np.ndarray,
    terminal_artifact: np.ndarray,
) -> None:
    response_labels = np.asarray(RESPONSE_ORDER)[np.argmax(response_probability, axis=1)]
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_axis, response_score, color="#17324D", linewidth=1.25)
    axes[0].set_ylabel("audit response score")
    axes[0].text(
        0.01,
        0.94,
        "Audit score only; not a trained-model probability",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        color="#607487",
    )

    axes[1].plot(window_time, contact_probability, color="#168AAD", linewidth=1.2)
    axes[1].axhline(0.5, color="#778899", linewidth=0.8, linestyle="--")
    axes[1].fill_between(
        window_time,
        0.0,
        1.0,
        where=sustained_contact,
        color="#90C8AC",
        alpha=0.18,
        step="mid",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("contact probability")

    for label_index, label in enumerate(RESPONSE_ORDER):
        axes[2].plot(
            window_time,
            response_probability[:, label_index],
            label=label,
            color=RESPONSE_COLORS[label],
            linewidth=1.0,
        )
    dominant_position = np.asarray(POSITION_ORDER)[np.argmax(position_probability, axis=1)]
    top_position_confidence = np.max(position_probability, axis=1)
    axes[2].plot(
        window_time,
        top_position_confidence,
        color="#594A42",
        linewidth=0.8,
        alpha=0.6,
        label="top position confidence",
    )
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("class probability")
    axes[2].set_xlabel("estimated time (s; 40 ms/frame)")
    axes[2].legend(ncol=4, fontsize=8, loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.18)
        if np.any(terminal_artifact):
            artifact_time = time_axis[np.flatnonzero(terminal_artifact)[0]]
            axis.axvline(artifact_time, color="#CC6677", linestyle=":", linewidth=1.0)
    figure.suptitle(
        f"Blind inference | {path.stem}\n"
        f"position labels are model outputs; response path: "
        f"{sustained_label_path(response_labels, sustained_contact)}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True)
    source_paths = sorted(input_root.rglob("*.dat"), key=lambda item: item.name.lower())
    if not source_paths:
        raise FileNotFoundError(f"no DAT files found below {input_root}")

    config = load_dynamic_config(args.sequence_config.resolve())
    wavelength = load_reference_wavelength_grid(Path(config["_reference_spectrum_csv"]))
    peak_windows = load_peak_windows(Path(config["_peak_config_path"]))
    model_path = args.model.resolve()
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != "dynamic_temporal_shadow_candidate_v3":
        raise ValueError("blind evaluation expects the deployed v3 temporal bundle")
    model_hash = sha256(model_path)
    frame_interval = float(config["acquisition"]["frame_interval_sec"])
    time_steps = int(bundle["time_steps"])

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    started = time.perf_counter()

    for file_index, source_path in enumerate(source_paths, start=1):
        decoded = read_sense_fast_dat(source_path)
        spectra = np.asarray(decoded.spectra, dtype=float)
        timestamps = np.arange(len(spectra), dtype=float) * frame_interval
        record = DynamicSequenceRecord(
            path=source_path,
            file_id=source_path.relative_to(input_root).as_posix(),
            capture_group="BLIND",
            position_label="UNREVEALED",
            wavelength_nm=wavelength,
            spectra=spectra,
            timestamps_sec=timestamps,
            layout=decoded.layout,
        )
        (
            frame_features,
            feature_names,
            _response_components,
            _component_names,
            response_score,
            baseline_frames,
        ) = extract_dynamic_frame_features(record, peak_windows, config)
        if tuple(feature_names) != tuple(bundle["frame_feature_names"]):
            raise ValueError("blind feature order differs from the deployed artifact")
        probability = temporal_probabilities(
            frame_features,
            bundle,
            batch_size=max(32, int(args.batch_size)),
        )
        endpoint_frames = np.arange(time_steps - 1, len(spectra), dtype=int)
        window_time = endpoint_frames.astype(float) * frame_interval
        contact_labels = tuple(str(value) for value in bundle["label_order"]["contact"])
        contact_index = contact_labels.index("contact")
        contact_probability = probability["contact"][:, contact_index]
        raw_contact = contact_probability >= 0.5
        sustained_contact = binary_opening(
            binary_closing(raw_contact, structure=np.ones(3, dtype=bool)),
            structure=np.ones(5, dtype=bool),
        )
        response_labels = np.asarray(RESPONSE_ORDER)[
            np.argmax(probability["response"], axis=1)
        ]
        position_labels = np.asarray(POSITION_ORDER)[
            np.argmax(probability["position"], axis=1)
        ]
        artifact_frames = terminal_artifact_mask(response_score)
        artifact_windows = artifact_frames[endpoint_frames]
        valid_window = ~artifact_windows
        active = sustained_contact & valid_window
        weights = contact_probability * active.astype(float)
        predicted_position, position_confidence, position_vote = weighted_vote(
            probability["position"], weights, POSITION_ORDER
        )
        predicted_response, response_confidence, response_vote = weighted_vote(
            probability["response"], weights, RESPONSE_ORDER
        )
        active_runs = runs(active)
        level_path = sustained_label_path(response_labels, active)
        qualified_levels: list[str] = []
        for label in RESPONSE_ORDER:
            label_mask = active & (response_labels == label)
            if any(end - start >= 5 for start, end in runs(label_mask)):
                qualified_levels.append(label)
        maximum_sustained_level = qualified_levels[-1] if qualified_levels else None
        tail_count = max(10, int(round(0.10 * len(valid_window))))
        tail_slice = slice(max(0, len(valid_window) - tail_count), len(valid_window))
        tail_valid = valid_window[tail_slice]
        tail_contact_fraction = (
            float(np.mean(raw_contact[tail_slice][tail_valid]))
            if np.any(tail_valid)
            else None
        )
        active_fraction = float(np.mean(active))
        data_status = (
            "no_sustained_contact"
            if not np.any(active)
            else "multi_bout_or_sequence"
            if len(active_runs) > 1 or ">" in level_path
            else "single_sustained_contact"
        )
        quality_flags: list[str] = []
        if np.any(artifact_frames):
            quality_flags.append("terminal_single_frame_discontinuity_excluded_from_summary")
        if decoded.layout.score_margin < 1.0e-4:
            quality_flags.append("dat_layout_low_score_margin")
        if tail_contact_fraction is not None and tail_contact_fraction > 0.30:
            quality_flags.append("contact_residual_at_file_tail")
        if float(np.mean(raw_contact[: max(5, baseline_frames - time_steps + 1)])) > 0.30:
            quality_flags.append("initial_baseline_predicted_as_contact")

        summary_rows.append(
            {
                "blind_id": f"B{file_index:02d}",
                "file_id": record.file_id,
                "source_sha256": sha256(source_path),
                "frame_count": len(spectra),
                "duration_sec_estimated": round(float(timestamps[-1]), 3),
                "dat_record_words": decoded.layout.record_words,
                "layout_adjacent_correlation": decoded.layout.median_adjacent_correlation,
                "baseline_frame_count": baseline_frames,
                "prediction_status": data_status,
                "predicted_position": predicted_position,
                "position_vote_confidence": position_confidence,
                "predicted_dominant_response": predicted_response,
                "response_vote_confidence": response_confidence,
                "maximum_sustained_response": maximum_sustained_level,
                "predicted_response_path": level_path,
                "sustained_contact_bout_count": len(active_runs),
                "sustained_contact_fraction": active_fraction,
                "tail_contact_fraction": tail_contact_fraction,
                "max_release_event_probability": float(np.max(probability["release"])),
                "terminal_artifact_frames": int(np.count_nonzero(artifact_frames)),
                "quality_flags": ";".join(quality_flags),
                **{f"position_vote_{key}": value for key, value in position_vote.items()},
                **{f"response_vote_{key}": value for key, value in response_vote.items()},
                "ground_truth_position": "",
                "ground_truth_response": "",
                "ground_truth_sequence": "",
            }
        )
        source_manifest.append(
            {
                "blind_id": f"B{file_index:02d}",
                "file_id": record.file_id,
                "source_sha256": summary_rows[-1]["source_sha256"],
                "source_size_bytes": source_path.stat().st_size,
                "model_sha256": model_hash,
                "truth_status": "unrevealed",
            }
        )
        for local_index, endpoint_frame in enumerate(endpoint_frames):
            prediction_rows.append(
                {
                    "blind_id": f"B{file_index:02d}",
                    "file_id": record.file_id,
                    "window_end_frame": int(endpoint_frame),
                    "time_sec_estimated": float(window_time[local_index]),
                    "contact_probability": float(contact_probability[local_index]),
                    "raw_contact_label": "contact" if raw_contact[local_index] else "no_contact",
                    "sustained_contact": bool(active[local_index]),
                    "position_label": str(position_labels[local_index]),
                    "position_confidence": float(
                        np.max(probability["position"][local_index])
                    ),
                    "response_label": str(response_labels[local_index]),
                    "response_confidence": float(
                        np.max(probability["response"][local_index])
                    ),
                    "release_event_probability": float(
                        probability["release"][local_index]
                    ),
                    "terminal_artifact_excluded": bool(artifact_windows[local_index]),
                }
            )
        plot_path = plot_dir / f"B{file_index:02d}_{source_path.stem}.png"
        plot_file(
            plot_path,
            time_axis=timestamps,
            response_score=response_score,
            window_time=window_time,
            contact_probability=contact_probability,
            position_probability=probability["position"],
            response_probability=probability["response"],
            sustained_contact=active,
            terminal_artifact=artifact_frames,
        )
        plot_paths.append(plot_path)

    write_rows(output_dir / "blind_source_manifest.csv", source_manifest)
    write_rows(output_dir / "blind_file_predictions.csv", summary_rows)
    write_rows(output_dir / "blind_window_predictions.csv", prediction_rows)

    images = [plt.imread(path) for path in plot_paths]
    columns = 2
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 5.3))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, image, plot_path in zip(axes_array, images, plot_paths):
        axis.imshow(image)
        axis.set_title(plot_path.stem.split("_", 1)[0], fontsize=9)
        axis.axis("off")
    for axis in axes_array[len(images) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "blind_contact_sheet.png", dpi=120)
    plt.close(figure)

    elapsed = time.perf_counter() - started
    metadata = {
        "schema_version": "touch_blind_temporal_evaluation_v1",
        "input_root": str(input_root),
        "independent_file_count": len(source_paths),
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "model_schema": bundle["schema_version"],
        "model_status": bundle.get("status"),
        "deployment_ready": bundle.get("deployment_ready"),
        "time_steps": time_steps,
        "estimated_frame_interval_sec": frame_interval,
        "ground_truth_status": "unrevealed",
        "accuracy_status": "not_computable_before_unblinding",
        "runtime_sec": elapsed,
    }
    (output_dir / "blind_evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table_lines = [
        "| Blind ID | Predicted position | Dominant level | Max sustained | Response path | Contact bouts | Tail contact | Flags |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        tail = row["tail_contact_fraction"]
        table_lines.append(
            "| {blind_id} | {predicted_position} | {predicted_dominant_response} | "
            "{maximum_sustained_response} | {predicted_response_path} | "
            "{sustained_contact_bout_count} | {tail:.1%} | {quality_flags} |".format(
                **row,
                tail=float(tail) if tail is not None else float("nan"),
            )
        )
    report = [
        "# Blind temporal spectrum evaluation",
        "",
        "## Evaluation boundary",
        "",
        f"- Frozen blind files: {len(source_paths)} independent DAT files.",
        f"- Model: `{model_path.name}` (`{model_hash[:12]}...`).",
        "- Ground-truth labels were not present in paths or filenames and remain unrevealed.",
        "- Accuracy, macro-F1, and per-class recall are intentionally not reported before unblinding.",
        "- The 512 wavelengths inside each spectrum are spatial/spectral samples, not time samples.",
        "- Temporal input is 20 consecutive spectrum frames (estimated 0.8 s).",
        "- Position and light/normal/hard probabilities are uncalibrated model outputs; levels are not force_N.",
        "",
        "## Frozen predictions",
        "",
        *table_lines,
        "",
        "## Interpretation",
        "",
        "`predicted_response_path` preserves sustained level changes inside a file instead of forcing each long DAT to one state. `tail_contact_fraction` is a direct residual check: a high value means the model still calls the released tail contact-like. Isolated terminal discontinuities are retained in the window CSV but excluded from file-level voting and explicitly flagged.",
        "",
        "## Unblinding protocol",
        "",
        "Fill only the three empty ground-truth columns in `blind_file_predictions.csv`, or provide a separate blind-id mapping. Re-scoring must use the frozen source/model hashes and must not retrain or retune thresholds first.",
    ]
    (output_dir / "blind_test_evaluation_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    print(f"files={len(source_paths)} windows={len(prediction_rows)} runtime_sec={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
