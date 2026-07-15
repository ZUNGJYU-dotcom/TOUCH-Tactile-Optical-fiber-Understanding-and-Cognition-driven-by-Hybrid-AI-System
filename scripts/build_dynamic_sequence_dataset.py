"""Build boundary-safe temporal windows from ordered Sense DAT sequences."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    build_dynamic_window_dataset,
    load_dynamic_config,
    load_dynamic_feature_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dynamic_sequence_dataset_20260714_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_dynamic_config(args.config)
    sequences = load_dynamic_feature_sequences(config)
    dataset = build_dynamic_window_dataset(sequences, config)

    np.savez_compressed(
        output_dir / "dynamic_sequence_windows.npz",
        X=dataset.values,
        feature_names=np.asarray(dataset.feature_names),
        stage_labels=dataset.stage_labels,
        contact_labels=dataset.contact_labels,
        position_labels=dataset.position_labels,
        file_ids=dataset.file_ids,
        capture_groups=dataset.capture_groups,
        window_start_frames=dataset.window_start_frames,
        window_end_frames=dataset.window_end_frames,
        sequence_quality_flags=dataset.sequence_quality_flags,
    )
    rows = []
    for index in range(len(dataset.values)):
        rows.append(
            {
                "window_id": index,
                "file_id": dataset.file_ids[index],
                "capture_group": dataset.capture_groups[index],
                "position_label": dataset.position_labels[index],
                "stage_label": dataset.stage_labels[index],
                "contact_label": dataset.contact_labels[index],
                "window_start_frame": int(dataset.window_start_frames[index]),
                "window_end_frame": int(dataset.window_end_frames[index]),
                "quality_flags": dataset.sequence_quality_flags[index],
            }
        )
    with (output_dir / "dynamic_window_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "shape": list(dataset.values.shape),
        "independent_dat_files": len(set(dataset.file_ids.tolist())),
        "capture_groups": sorted(set(dataset.capture_groups.tolist())),
        "stage_counts": dict(Counter(dataset.stage_labels.tolist())),
        "capture_group_counts": dict(Counter(dataset.capture_groups.tolist())),
        "contact_position_counts": dict(
            Counter(
                dataset.position_labels[dataset.contact_labels == "contact"].tolist()
            )
        ),
        "window_length_frames": int(dataset.values.shape[1]),
        "frame_feature_count": int(dataset.values.shape[2]),
        "estimated_window_duration_sec": float(
            dataset.values.shape[1] * config["acquisition"]["frame_interval_sec"]
        ),
        "split_strategy": config["windowing"]["split_strategy"],
        "random_frame_split_allowed": False,
        "release_windows_included": False,
        "response_score_used_as_model_input": False,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Dynamic sequence dataset",
        "",
        f"- Shape: `{tuple(dataset.values.shape)}` = windows x time x frame features.",
        f"- Independent source DAT files: {summary['independent_dat_files']}.",
        f"- Capture groups: {', '.join(summary['capture_groups'])}.",
        f"- Estimated window duration: {summary['estimated_window_duration_sec']:.2f} s.",
        "- Every window lies inside one stable no_contact/light/normal/hard plateau.",
        "- Release and transition regions are excluded.",
        "- The mixed response score is used for boundary auditing only, never as model input.",
        "- Formal evaluation must leave out whole capture groups and original DAT files.",
        "- Light/normal/hard are approximate manual response levels, not force_N.",
    ]
    (output_dir / "dynamic_sequence_dataset_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
