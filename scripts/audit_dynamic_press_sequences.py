"""Audit and auto-label ordered no-contact/light/normal/hard/release DAT files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    DynamicFeatureSequence,
    load_dynamic_config,
    load_dynamic_feature_sequences,
)


STAGE_COLORS = {
    "no_contact": "#9AA0A6",
    "light": "#56B4E9",
    "normal": "#E69F00",
    "hard": "#D55E00",
    "release": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dynamic_sequence_audit_20260714_v1",
    )
    return parser.parse_args()


def _plot_sequence(sequence: DynamicFeatureSequence, path: Path) -> None:
    time = sequence.record.timestamps_sec
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time, sequence.response_score, color="#17324D", linewidth=1.5)
    axes[0].set_ylabel("mixed response score")
    for segment in sequence.stage_segments:
        start = time[segment.start_frame]
        end_index = min(segment.end_frame, len(time) - 1)
        end = time[end_index]
        axes[0].axvspan(start, end, color=STAGE_COLORS[segment.label], alpha=0.15)
        axes[0].axvspan(
            time[segment.stable_start_frame],
            time[min(segment.stable_end_frame, len(time) - 1)],
            color=STAGE_COLORS[segment.label],
            alpha=0.22,
        )
        axes[0].text(
            0.5 * (start + end),
            0.96,
            segment.label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )

    for index, name in enumerate(sequence.response_component_names):
        values = sequence.response_components[:, index]
        scale = max(float(np.percentile(np.abs(values), 98.0)), 1.0e-9)
        axes[1].plot(time, values / scale, linewidth=1.0, label=name)
    axes[1].set_ylabel("component / p98")
    axes[1].legend(ncol=2, fontsize=7, loc="upper left")

    shift_indices = [
        index
        for index, name in enumerate(sequence.feature_names)
        if name.endswith("_centroid_shift_pm")
    ]
    for index in shift_indices:
        axes[2].plot(
            time,
            sequence.feature_matrix[:, index],
            linewidth=0.85,
            alpha=0.8,
            label=sequence.feature_names[index].split("_")[0].upper(),
        )
    axes[2].axhline(0.0, color="#666666", linewidth=0.7)
    axes[2].set_ylabel("centroid shift (pm)")
    axes[2].set_xlabel("estimated time (s; 40 ms/frame)")
    axes[2].legend(ncol=9, fontsize=6, loc="upper center")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{sequence.record.capture_group} / {sequence.record.position_label} | "
        f"{sequence.record.file_id} | {sequence.segmentation_status}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    plot_dir = output_dir / "file_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    config = load_dynamic_config(args.config)
    sequences = load_dynamic_feature_sequences(config)

    manifest_rows = []
    segment_rows = []
    frame_rows = []
    feature_blocks = []
    file_index_blocks = []
    frame_index_blocks = []
    for file_index, sequence in enumerate(sequences):
        plot_path = plot_dir / (
            f"{sequence.record.capture_group}_{sequence.record.position_label}.png"
        )
        _plot_sequence(sequence, plot_path)
        manifest_rows.append(
            {
                "file_id": sequence.record.file_id,
                "capture_group": sequence.record.capture_group,
                "position_label": sequence.record.position_label,
                "frame_count": len(sequence.record.spectra),
                "duration_sec_estimated": sequence.record.timestamps_sec[-1],
                "dat_layout": sequence.record.layout.name,
                "dat_record_words": sequence.record.layout.record_words,
                "dat_trailing_words_ignored": sequence.record.layout.trailing_words,
                "layout_median_adjacent_correlation": sequence.record.layout.median_adjacent_correlation,
                "baseline_frame_count": sequence.baseline_frame_count,
                "release_observed": sequence.release_observed,
                "release_recovered": sequence.release_recovered,
                "release_recovery_ratio": sequence.release_recovery_ratio,
                "segmentation_status": sequence.segmentation_status,
                "quality_flags": ";".join(sequence.quality_flags),
            }
        )
        labels = np.full(len(sequence.record.spectra), "transition", dtype=object)
        stable = np.zeros(len(labels), dtype=bool)
        for segment in sequence.stage_segments:
            labels[segment.start_frame : segment.end_frame] = segment.label
            if segment.training_eligible:
                stable[segment.stable_start_frame : segment.stable_end_frame] = True
            segment_rows.append(
                {
                    "file_id": sequence.record.file_id,
                    "capture_group": sequence.record.capture_group,
                    "position_label": sequence.record.position_label,
                    "stage_label": segment.label,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "start_time_sec_estimated": segment.start_frame
                    * float(config["acquisition"]["frame_interval_sec"]),
                    "end_time_sec_estimated": segment.end_frame
                    * float(config["acquisition"]["frame_interval_sec"]),
                    "stable_start_frame": segment.stable_start_frame,
                    "stable_end_frame": segment.stable_end_frame,
                    "mean_response": segment.mean_response,
                    "median_response": segment.median_response,
                    "training_eligible": segment.training_eligible,
                    "quality_flag": segment.quality_flag,
                }
            )
        for frame_index in range(len(labels)):
            frame_rows.append(
                {
                    "file_id": sequence.record.file_id,
                    "capture_group": sequence.record.capture_group,
                    "position_label": sequence.record.position_label,
                    "frame_index": frame_index,
                    "time_sec_estimated": sequence.record.timestamps_sec[frame_index],
                    "stage_label": labels[frame_index],
                    "stable_training_frame": bool(stable[frame_index]),
                    "response_score_labeling_only": sequence.response_score[frame_index],
                }
            )
        feature_blocks.append(sequence.feature_matrix)
        file_index_blocks.append(np.full(len(labels), file_index, dtype=np.int16))
        frame_index_blocks.append(np.arange(len(labels), dtype=np.int32))

    def write_rows(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows(output_dir / "dynamic_sequence_manifest.csv", manifest_rows)
    write_rows(output_dir / "dynamic_stage_segments.csv", segment_rows)
    write_rows(output_dir / "dynamic_frame_labels.csv", frame_rows)
    np.savez_compressed(
        output_dir / "dynamic_frame_features.npz",
        X_frames=np.vstack(feature_blocks),
        file_indices=np.concatenate(file_index_blocks),
        frame_indices=np.concatenate(frame_index_blocks),
        feature_names=np.asarray(sequences[0].feature_names),
        file_ids=np.asarray([sequence.record.file_id for sequence in sequences]),
        capture_groups=np.asarray(
            [sequence.record.capture_group for sequence in sequences]
        ),
        position_labels=np.asarray(
            [sequence.record.position_label for sequence in sequences]
        ),
    )

    ordered_plots = [
        plot_dir / f"G{group}_{position}.png"
        for group in (1, 2, 3)
        for position in ("P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33")
        if (plot_dir / f"G{group}_{position}.png").exists()
    ]
    if ordered_plots:
        images = [plt.imread(path) for path in ordered_plots]
        figure, axes = plt.subplots(9, 3, figsize=(18, 25))
        for axis, image, path in zip(axes.ravel(), images, ordered_plots):
            axis.imshow(image)
            axis.set_title(path.stem, fontsize=8)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "segmentation_contact_sheet.png", dpi=130)
        plt.close(figure)

    release_observed_count = sum(sequence.release_observed for sequence in sequences)
    release_recovered_count = sum(sequence.release_recovered for sequence in sequences)
    warning_count = sum(bool(sequence.quality_flags) for sequence in sequences)
    report = [
        "# Dynamic press-sequence audit",
        "",
        "- Input: 27 independent Sense DAT files (3 capture groups x 9 positions).",
        "- Known order per file: no_contact -> light -> normal -> hard -> release.",
        "- Light/normal/hard are approximate manual response levels, not force_N.",
        "- The 512 wavelength samples inside a frame are not treated as time samples.",
        "- Formal split key is the original file and capture group; random frame split is prohibited.",
        "",
        "## DAT parser correction",
        "",
        "The new files use 513 uint16 words per record: 512 spectral points plus one auxiliary recorder word. The previous 512-word size heuristic caused a one-pixel artificial drift per decoded frame. Layout selection now uses adjacent-spectrum continuity and ignores only incomplete trailing words.",
        "",
        "## Audit summary",
        "",
        f"- Total decoded frames: {sum(len(sequence.record.spectra) for sequence in sequences)}",
        f"- Clear release drop observed: {release_observed_count}/27",
        f"- Release recovered close to initial baseline: {release_recovered_count}/27",
        f"- Files with one or more warnings: {warning_count}/27",
        "- Release frames are excluded from no-contact and press-level training.",
        "- Only stable central portions of no_contact/light/normal/hard segments are training eligible.",
        "",
        "## Limitations",
        "",
        "Stage boundaries are auto-refined from the operator-specified order and mixed spectral change. They should be visually reviewed before any model is promoted. The 40 ms frame interval is an acquisition-setting estimate because the DAT records do not contain an independently verified timestamp for every frame.",
    ]
    (output_dir / "dynamic_sequence_audit_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(f"sequences={len(sequences)} total_frames={sum(len(s.record.spectra) for s in sequences)}")
    print(f"release_observed={release_observed_count} release_recovered={release_recovered_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
