"""Validate saved one-spectrum shadow runtime without deploying it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    load_dynamic_config,
    load_dynamic_feature_sequences,
)
from src.hybrid_spectrum.dynamic_single_spectrum import (  # noqa: E402
    STABLE_STAGE_ORDER,
    stable_live_frame_indices,
)
from src.hybrid_spectrum.dynamic_single_spectrum_adapter import (  # noqa: E402
    DynamicSingleSpectrumShadowAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models"
            / "candidates"
            / "dynamic_single_spectrum_fast_candidate_v1.joblib"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-frame-stride", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    config = load_dynamic_config(args.config.resolve())
    sequences = load_dynamic_feature_sequences(config)
    adapter = DynamicSingleSpectrumShadowAdapter(args.model.resolve())

    rows = []
    latency_ms = []
    probability_errors = []
    baseline_predictions = []
    for sequence in sequences:
        baseline_result = adapter.predict(
            sequence.record.wavelength_nm,
            sequence.baseline_spectrum,
            sequence.baseline_spectrum,
        )
        baseline_predictions.append(str(baseline_result["contact_label"]))
        for segment in sequence.stage_segments:
            if segment.label not in STABLE_STAGE_ORDER or not segment.training_eligible:
                continue
            indices = stable_live_frame_indices(
                segment.stable_start_frame,
                segment.stable_end_frame,
                args.live_frame_stride,
            )
            for frame_index in indices:
                started = time.perf_counter_ns()
                result = adapter.predict(
                    sequence.record.wavelength_nm,
                    sequence.record.spectra[int(frame_index)],
                    sequence.baseline_spectrum,
                )
                elapsed = (time.perf_counter_ns() - started) / 1.0e6
                latency_ms.append(elapsed)
                for task, values in result["probabilities"].items():
                    total = float(sum(float(value) for value in values.values()))
                    if not np.isfinite(total) or abs(total - 1.0) > 1.0e-6:
                        probability_errors.append(
                            {
                                "file_id": sequence.record.file_id,
                                "frame_index": int(frame_index),
                                "task": task,
                                "probability_sum": total,
                            }
                        )
                rows.append(
                    {
                        "file_id": sequence.record.file_id,
                        "capture_group": sequence.record.capture_group,
                        "frame_index": int(frame_index),
                        "true_stage": segment.label,
                        "true_position": (
                            ""
                            if segment.label == "no_contact"
                            else sequence.record.position_label
                        ),
                        "predicted_contact": result["contact_label"],
                        "predicted_position": result["position_label"],
                        "predicted_response_level": result["response_level"],
                        "contact_confidence": result["contact_confidence"],
                        "position_confidence": result["position_confidence"],
                        "response_confidence": result["response_confidence"],
                        "runtime_latency_ms": elapsed,
                    }
                )

    with (output_dir / "runtime_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline_no_contact_count = sum(
        label == "no_contact" for label in baseline_predictions
    )
    summary = {
        "schema_version": "dynamic_single_spectrum_runtime_validation_v1",
        "model_path": str(args.model.resolve()),
        "deployment_changed": False,
        "num_independent_dat_files": len(sequences),
        "num_runtime_frames": len(rows),
        "baseline_self_predictions": {
            "no_contact": baseline_no_contact_count,
            "total": len(baseline_predictions),
        },
        "probability_sum_error_count": len(probability_errors),
        "probability_sum_errors": probability_errors,
        "runtime_latency_ms": {
            "p50": float(np.percentile(latency_ms, 50.0)),
            "p95": float(np.percentile(latency_ms, 95.0)),
            "p99": float(np.percentile(latency_ms, 99.0)),
            "maximum": float(max(latency_ms)),
        },
        "physical_acquisition_interval_sec": 0.40,
        "estimated_first_output_p95_sec": float(
            0.40 + np.percentile(latency_ms, 95.0) / 1000.0
        ),
        "formal_accuracy_source": (
            "outputs/dynamic_single_spectrum_algorithm_benchmark_20260716_v1"
        ),
        "warning": "These fitted-model runtime predictions are contract checks, not formal accuracy estimates.",
    }
    (output_dir / "runtime_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Dynamic single-spectrum runtime validation",
        "",
        f"- Independent DAT files exercised: {len(sequences)}.",
        f"- Stable runtime frames exercised: {len(rows)}.",
        f"- Baseline-as-input predicted no-contact: {baseline_no_contact_count}/{len(baseline_predictions)} files.",
        f"- Probability normalization errors: {len(probability_errors)}.",
        f"- End-to-end adapter latency: p50 {summary['runtime_latency_ms']['p50']:.2f} ms, p95 {summary['runtime_latency_ms']['p95']:.2f} ms.",
        f"- Estimated first output including one 0.40 s acquisition: {summary['estimated_first_output_p95_sec']:.3f} s.",
        "- This validation does not deploy or enable the candidate.",
        "- Accuracy must be read from grouped out-of-fold results, not these fitted-model runtime checks.",
        "- Light/normal/hard remain approximate manual response levels, not force_N.",
    ]
    (output_dir / "runtime_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
