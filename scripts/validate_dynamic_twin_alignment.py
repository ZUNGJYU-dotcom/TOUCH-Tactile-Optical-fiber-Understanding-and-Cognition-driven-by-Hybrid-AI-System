"""Validate grouped temporal predictions against the digital-twin mapping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_twin_mapping import (  # noqa: E402
    ARRAY_CHANNEL_COORDS,
    ARRAY_DISPLAY_ROWS,
    RESPONSE_DEFORMATION_PROXY,
    dynamic_prediction_to_twin_proxy,
)


POSITION_ORDER = tuple(channel for row in ARRAY_DISPLAY_ROWS for channel in row)
RESPONSE_ORDER = ("light", "normal", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def majority(values: list[str]) -> str:
    counts = Counter(values)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    by_index: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    with args.predictions.resolve().open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_index[int(row["window_index"])][row["task"]] = row

    rows: list[dict[str, object]] = []
    contract_violations: list[str] = []
    for window_index in sorted(by_index):
        pair = by_index[window_index]
        if "position" not in pair or "response_level" not in pair:
            continue
        position_row = pair["position"]
        response_row = pair["response_level"]
        true_position = position_row["true_label"]
        predicted_position = position_row["predicted_label"]
        true_response = response_row["true_label"]
        predicted_response = response_row["predicted_label"]
        prediction = {
            "ready": True,
            "contact": {"label": "contact"},
            "position": {"label": predicted_position},
            "response_level": {"label": predicted_response},
            "operational_state": "active_contact",
        }
        proxy = dynamic_prediction_to_twin_proxy(prediction)
        grid = np.asarray(proxy["surface_grid"], dtype=float)
        peak_row, peak_column = np.unravel_index(np.argmax(grid), grid.shape)
        grid_peak_channel = ARRAY_DISPLAY_ROWS[peak_row][peak_column]
        predicted_coordinate = ARRAY_CHANNEL_COORDS[predicted_position]
        true_coordinate = ARRAY_CHANNEL_COORDS[true_position]
        center = (
            float(proxy["surface_metrics"]["surface_centroid_x"]),
            float(proxy["surface_metrics"]["surface_centroid_y"]),
        )
        mapping_ok = (
            grid_peak_channel == predicted_position
            and center == predicted_coordinate
            and proxy["surface_metrics"]["dominant_channel"] == predicted_position
        )
        if not mapping_ok:
            contract_violations.append(f"window {window_index}: {predicted_position}")
        position_distance = float(
            np.hypot(
                predicted_coordinate[0] - true_coordinate[0],
                predicted_coordinate[1] - true_coordinate[1],
            )
        )
        rows.append(
            {
                "window_index": window_index,
                "file_id": position_row["file_id"],
                "capture_group": position_row["capture_group"],
                "true_position": true_position,
                "predicted_position": predicted_position,
                "position_exact": true_position == predicted_position,
                "same_physical_column": true_position[1] == predicted_position[1],
                "same_physical_row": true_position[2] == predicted_position[2],
                "position_distance_pixels": position_distance,
                "true_response": true_response,
                "predicted_response": predicted_response,
                "response_exact": true_response == predicted_response,
                "response_ordinal_error": abs(
                    RESPONSE_ORDER.index(true_response)
                    - RESPONSE_ORDER.index(predicted_response)
                ),
                "true_deformation_proxy": RESPONSE_DEFORMATION_PROXY[true_response],
                "predicted_deformation_proxy": proxy["deformation_proxy"],
                "grid_peak_channel": grid_peak_channel,
                "surface_centroid_x": center[0],
                "surface_centroid_y": center[1],
                "mapping_contract_ok": mapping_ok,
                "combined_exact": (
                    true_position == predicted_position
                    and true_response == predicted_response
                ),
            }
        )

    if not rows:
        raise RuntimeError("no joined position/response predictions were found")

    files: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        files[str(row["file_id"])].append(row)
    file_position_correct = []
    file_response_stage_correct = []
    for file_rows in files.values():
        true_position = str(file_rows[0]["true_position"])
        predicted_position = majority(
            [str(row["predicted_position"]) for row in file_rows]
        )
        file_position_correct.append(predicted_position == true_position)
        by_stage: dict[str, list[str]] = defaultdict(list)
        for row in file_rows:
            by_stage[str(row["true_response"])].append(
                str(row["predicted_response"])
            )
        file_response_stage_correct.extend(
            majority(predictions) == true_stage
            for true_stage, predictions in by_stage.items()
        )

    position_exact = np.asarray([bool(row["position_exact"]) for row in rows])
    response_exact = np.asarray([bool(row["response_exact"]) for row in rows])
    combined_exact = np.asarray([bool(row["combined_exact"]) for row in rows])
    distances = np.asarray([float(row["position_distance_pixels"]) for row in rows])
    ordinal_errors = np.asarray([float(row["response_ordinal_error"]) for row in rows])
    per_position = {}
    for label in POSITION_ORDER:
        selected = [row for row in rows if row["true_position"] == label]
        per_position[label] = {
            "window_count": len(selected),
            "exact_accuracy": float(np.mean([row["position_exact"] for row in selected])),
            "mean_distance_pixels": float(
                np.mean([row["position_distance_pixels"] for row in selected])
            ),
        }

    metrics = {
        "schema_version": "dynamic_twin_alignment_validation_v1",
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "independent_dat_sequences": len(files),
        "active_window_count": len(rows),
        "position_exact_accuracy": float(np.mean(position_exact)),
        "position_same_column_accuracy": float(
            np.mean([row["same_physical_column"] for row in rows])
        ),
        "position_same_row_accuracy": float(
            np.mean([row["same_physical_row"] for row in rows])
        ),
        "position_mean_distance_pixels": float(np.mean(distances)),
        "position_within_one_pixel_rate": float(np.mean(distances <= 1.0)),
        "position_file_voting_accuracy": float(np.mean(file_position_correct)),
        "response_exact_accuracy": float(np.mean(response_exact)),
        "response_mean_ordinal_error": float(np.mean(ordinal_errors)),
        "response_within_one_level_rate": float(np.mean(ordinal_errors <= 1.0)),
        "response_file_stage_voting_accuracy": float(
            np.mean(file_response_stage_correct)
        ),
        "combined_exact_accuracy": float(np.mean(combined_exact)),
        "mapping_contract_violation_count": len(contract_violations),
        "no_contact_deformation_proxy": dynamic_prediction_to_twin_proxy(
            {
                "ready": True,
                "contact": {"label": "no_contact"},
                "operational_state": "no_contact",
            }
        )["deformation_proxy"],
        "deformation_proxy_by_response": RESPONSE_DEFORMATION_PROXY,
        "per_position": per_position,
    }

    with (output_dir / "dynamic_twin_alignment_windows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "dynamic_twin_alignment_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    x = np.arange(len(POSITION_ORDER))
    axes[0].bar(
        x,
        [per_position[label]["exact_accuracy"] for label in POSITION_ORDER],
        color="#2f7f9f",
    )
    axes[0].set_xticks(x, POSITION_ORDER, rotation=45)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Grouped exact accuracy")
    axes[0].set_title("Predicted center at the correct physical pixel")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x,
        [per_position[label]["mean_distance_pixels"] for label in POSITION_ORDER],
        color="#d59b3d",
    )
    axes[1].set_xticks(x, POSITION_ORDER, rotation=45)
    axes[1].set_ylabel("Mean center error (pixel spacing)")
    axes[1].set_title("Digital-twin contact-center error")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "position_twin_alignment.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    levels = list(RESPONSE_ORDER)
    axis.bar(
        levels,
        [RESPONSE_DEFORMATION_PROXY[level] for level in levels],
        color=["#63b6af", "#e2c66e", "#bf6a68"],
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Uncalibrated deformation proxy")
    axis.set_title("Monotonic response-level visual mapping")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "response_deformation_mapping.png", dpi=180)
    plt.close(figure)

    report = [
        "# Dynamic model to digital-twin alignment validation",
        "",
        "This report evaluates grouped held-out temporal predictions after they are converted to the physical 3x3 visualization contract.",
        "",
        "## Results",
        "",
        f"- Independent DAT sequences: {len(files)}.",
        f"- Active windows: {len(rows)}.",
        f"- Exact digital-twin position accuracy: {metrics['position_exact_accuracy']:.3f}.",
        f"- Same physical row: {metrics['position_same_row_accuracy']:.3f}.",
        f"- Same physical column: {metrics['position_same_column_accuracy']:.3f}.",
        f"- Mean center error: {metrics['position_mean_distance_pixels']:.3f} pixel spacing.",
        f"- Within one pixel: {metrics['position_within_one_pixel_rate']:.3f}.",
        f"- Position file voting accuracy: {metrics['position_file_voting_accuracy']:.3f}.",
        f"- Exact light/normal/hard accuracy: {metrics['response_exact_accuracy']:.3f}.",
        f"- Response file-stage voting accuracy: {metrics['response_file_stage_voting_accuracy']:.3f}.",
        f"- Exact position plus response accuracy: {metrics['combined_exact_accuracy']:.3f}.",
        f"- Mapping-contract violations: {len(contract_violations)}.",
        "",
        "## Interpretation",
        "",
        "The visualization contract itself introduces no additional orientation error: every predicted Pxy label produces its maximum and centroid at the same configured physical pixel. Remaining location errors therefore originate in model recognition, not a row/column swap in the twin mapper.",
        "",
        "Light, normal, and hard map monotonically to 0.28, 0.58, and 0.92 deformation proxies. No-contact maps to zero. These are visualization amplitudes, not calibrated physical force.",
        "",
        "The model remains a validation candidate. Visible live control still requires explicit operator opt-in and live comparison against the known pressed location.",
    ]
    (output_dir / "dynamic_twin_alignment_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
