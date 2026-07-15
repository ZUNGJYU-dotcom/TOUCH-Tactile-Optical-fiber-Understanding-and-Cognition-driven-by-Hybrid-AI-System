"""Replay selected real DAT sequences through the complete dynamic shadow adapter."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    load_dynamic_config,
    load_dynamic_feature_sequences,
)
from src.hybrid_spectrum.dynamic_shadow_adapter import (  # noqa: E402
    DynamicTemporalShadowAdapter,
    load_dynamic_shadow_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "dynamic_temporal_shadow_candidate_v2.joblib",
    )
    parser.add_argument(
        "--sequence-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument(
        "--peak-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    model_path = args.model.resolve()
    bundle = load_dynamic_shadow_bundle(model_path)
    config = load_dynamic_config(args.sequence_config.resolve())
    sequences = load_dynamic_feature_sequences(config)
    selected_cases = (("G1", "P22"), ("G1", "P21"))
    rows: list[dict[str, Any]] = []
    all_latencies: list[float] = []

    for capture_group, position_label in selected_cases:
        sequence = next(
            item
            for item in sequences
            if item.record.capture_group == capture_group
            and item.record.position_label == position_label
        )
        adapter = DynamicTemporalShadowAdapter.from_paths(
            model_path,
            args.peak_config.resolve(),
        )
        adapter.set_baseline(
            sequence.record.wavelength_nm,
            sequence.baseline_spectrum,
        )
        release_start = next(
            segment.start_frame
            for segment in sequence.stage_segments
            if segment.label == "release"
        )
        latch_frame: int | None = None
        latencies: list[float] = []
        last_result: dict[str, Any] | None = None
        for frame_index, spectrum in enumerate(sequence.record.spectra):
            started = time.perf_counter()
            last_result = adapter.update(sequence.record.wavelength_nm, spectrum)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if frame_index >= adapter.time_steps - 1:
                latencies.append(elapsed_ms)
                all_latencies.append(elapsed_ms)
            if last_result.get("release_guard", {}).get("just_latched"):
                latch_frame = frame_index
                break

        rows.append(
            {
                "file_id": sequence.record.file_id,
                "capture_group": capture_group,
                "position_label": position_label,
                "release_start_frame": release_start,
                "release_latch_frame": latch_frame,
                "release_delay_frames": (
                    None if latch_frame is None else latch_frame - release_start
                ),
                "release_delay_sec": (
                    None
                    if latch_frame is None
                    else (latch_frame - release_start) * 0.04
                ),
                "unsafe_early_release": bool(
                    latch_frame is not None and latch_frame < release_start
                ),
                "segmentation_release_observed": sequence.release_observed,
                "segmentation_release_recovered": sequence.release_recovered,
                "final_adapter_status": (
                    None if last_result is None else last_result.get("status")
                ),
                "median_latency_ms_per_frame": (
                    statistics.median(latencies) if latencies else None
                ),
                "p95_latency_ms_per_frame": percentile(latencies, 95.0),
                "runtime_inference_policy": (
                    None
                    if last_result is None
                    else last_result.get("runtime_inference_policy")
                ),
            }
        )

    grouped = bundle["release_guard_grouped_cv"]
    model_schema = str(bundle["schema_version"])
    summary = {
        "schema_version": "dynamic_shadow_runtime_validation_v1",
        "model_path": str(model_path),
        "model_schema": model_schema,
        "status": bundle["status"],
        "deployment_ready": bundle["deployment_ready"],
        "grouped_release_validation": grouped,
        "real_spectrum_runtime_cases": rows,
        "runtime_latency": {
            "median_ms_per_frame": (
                statistics.median(all_latencies) if all_latencies else None
            ),
            "p95_ms_per_frame": percentile(all_latencies, 95.0),
            "estimated_source_frame_interval_ms": 40.0,
            "meets_every_frame_40ms_budget": bool(
                all_latencies and percentile(all_latencies, 95.0) <= 40.0
            ),
            "recommendation": (
                "evaluate_shadow_every_frame"
                if all_latencies and percentile(all_latencies, 95.0) <= 40.0
                else "evaluate_shadow_every_2_frames_and_smooth_visual_state"
            ),
        },
        "decision": "keep_shadow_only_not_primary",
    }
    with (output_dir / "runtime_spectral_replay_samples.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "runtime_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Dynamic shadow runtime validation",
        "",
        "The complete runtime path was exercised with real DAT spectra: nine-peak feature extraction, 20-frame temporal summary, contact/position/response inference, and release-residual guard.",
        "",
        "## Results",
        "",
        f"- Grouped release detection: {grouped['detected_release_sequence_count']}/27 sequences.",
        f"- Grouped unsafe early triggers: {grouped['unsafe_early_trigger_sequence_count']}.",
        f"- Grouped post-guard release false-contact rate: {grouped['v2_release_false_contact_rate_after_guard']:.3f}.",
        f"- G1/P22 real-spectrum latch delay: {rows[0]['release_delay_sec']:.2f} s.",
        f"- G1/P21 latch: {rows[1]['release_latch_frame']} (release evidence was not clear in the source sequence).",
        f"- Runtime median latency: {summary['runtime_latency']['median_ms_per_frame']:.1f} ms/frame.",
        f"- Runtime p95 latency: {summary['runtime_latency']['p95_ms_per_frame']:.1f} ms/frame.",
        "",
        "## Decision",
        "",
        f"Keep the {model_schema} bundle shadow-only. It must not drive the operator UI yet. The validated runtime recommendation is `{summary['runtime_latency']['recommendation']}`, followed by live validation.",
        "",
        "Four sequences remain unresolved in grouped testing. Additional independent release captures are required, including release directly from light and normal levels. Light/normal/hard are approximate manual response levels, not calibrated physical force.",
    ]
    (output_dir / "runtime_validation_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["runtime_latency"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
