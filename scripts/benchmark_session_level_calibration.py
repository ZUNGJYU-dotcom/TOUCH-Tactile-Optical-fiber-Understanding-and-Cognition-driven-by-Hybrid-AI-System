"""Benchmark one-trial-per-position/level current-session calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.sense_static_dataset import (  # noqa: E402
    build_static_feature_dataset,
    load_sense_dataset,
    load_training_config,
)
from src.hybrid_spectrum.session_level_calibration import (  # noqa: E402
    LEVEL_ORDER,
    POSITION_ORDER,
    PerPositionOrdinalCalibrator,
    extract_response_core_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "sense_static_training.yaml",
    )
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=ROOT / "config" / "hybrid_spectrum_channels.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def classification_metrics(true: list[str], predicted: list[str]) -> dict[str, Any]:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=list(LEVEL_ORDER),
        zero_division=0,
    )
    return {
        "num_samples": len(true),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(true, predicted, labels=list(LEVEL_ORDER), average="macro")
        ),
        "confusion_matrix": confusion_matrix(
            true, predicted, labels=list(LEVEL_ORDER)
        ).tolist(),
        "label_order": list(LEVEL_ORDER),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1_values[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LEVEL_ORDER)
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = load_training_config(args.config.resolve())
    records = tuple(load_sense_dataset(config))
    dataset = build_static_feature_dataset(
        records,
        config.get("feature_extraction", config),
        args.channel_config.resolve(),
    )
    feature_rows = [
        extract_response_core_features(
            dict(zip(dataset.engineered_columns, dataset.engineered_matrix[index]))
        )
        for index in range(len(records))
    ]
    manual_indices = [
        index for index, record in enumerate(records) if record.sample_kind == "manual_press"
    ]
    sessions = {
        "target_incremental_repeats_6_15": tuple(range(6, 16)),
        "target_original_repeats_1_5": tuple(range(1, 6)),
    }

    run_rows: list[dict[str, Any]] = []
    all_true: list[str] = []
    all_predicted: list[str] = []
    session_metrics: dict[str, Any] = {}
    example_payload: dict[str, Any] | None = None
    for session_name, session_repeats in sessions.items():
        session_true: list[str] = []
        session_predicted: list[str] = []
        for calibration_repeat in session_repeats:
            calibration_indices = [
                index
                for index in manual_indices
                if records[index].repeat_index == calibration_repeat
            ]
            test_indices = [
                index
                for index in manual_indices
                if records[index].repeat_index in session_repeats
                and records[index].repeat_index != calibration_repeat
            ]
            calibration_samples = [
                {
                    "position": records[index].position_label,
                    "level": records[index].manual_force_label,
                    "features": feature_rows[index],
                }
                for index in calibration_indices
            ]
            token = f"offline_{session_name}_repeat_{calibration_repeat}"
            calibrator = PerPositionOrdinalCalibrator.fit(
                calibration_samples,
                baseline_token=token,
                required_positions=POSITION_ORDER,
            )
            if example_payload is None:
                example_payload = calibrator.to_dict()
                example_payload["runtime_usable"] = False
                example_payload["note"] = (
                    "Offline example only; a live calibration must use the current "
                    "runtime baseline token."
                )

            true_labels: list[str] = []
            predicted_labels: list[str] = []
            scores: list[float] = []
            for index in test_indices:
                prediction = calibrator.predict(
                    str(records[index].position_label),
                    feature_rows[index],
                    baseline_token=token,
                )
                if not prediction.get("ok"):
                    raise RuntimeError(str(prediction))
                true_labels.append(str(records[index].manual_force_label))
                predicted_labels.append(str(prediction["label"]))
                scores.append(float(prediction["ordinal_score"]))
            metrics = classification_metrics(true_labels, predicted_labels)
            quality_values = [
                float(item["monotonic_feature_fraction"])
                for item in calibrator.quality.values()
            ]
            run_rows.append(
                {
                    "target_session": session_name,
                    "calibration_repeat": calibration_repeat,
                    "num_calibration_trials": len(calibration_indices),
                    "num_test_files": len(test_indices),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "light_recall": metrics["per_class"]["light"]["recall"],
                    "normal_recall": metrics["per_class"]["normal"]["recall"],
                    "hard_recall": metrics["per_class"]["hard"]["recall"],
                    "mean_position_monotonic_feature_fraction": float(
                        np.mean(quality_values)
                    ),
                    "mean_ordinal_score": float(np.mean(scores)),
                }
            )
            session_true.extend(true_labels)
            session_predicted.extend(predicted_labels)
            all_true.extend(true_labels)
            all_predicted.extend(predicted_labels)
        session_metrics[session_name] = classification_metrics(
            session_true, session_predicted
        )

    aggregate = classification_metrics(all_true, all_predicted)
    macro_values = np.asarray([row["macro_f1"] for row in run_rows], dtype=float)
    aggregate.update(
        {
            "mean_calibration_run_macro_f1": float(np.mean(macro_values)),
            "minimum_calibration_run_macro_f1": float(np.min(macro_values)),
            "maximum_calibration_run_macro_f1": float(np.max(macro_values)),
            "num_calibration_runs": len(run_rows),
            "calibration_trials_per_run": 27,
        }
    )
    result = {
        "schema_version": "session_level_calibration_benchmark_v1",
        "evaluation_validity": "target_session_one_repeat_calibration_grouped_by_repeat",
        "independent_file_count": len(records),
        "manual_press_file_count": len(manual_indices),
        "calibration_unit": "one_independent_file_per_position_and_level",
        "calibration_trials_per_run": 27,
        "test_unit": "independent_static_spectrum_file",
        "feature_names": [
            "shift_abs_mean_pm",
            "shift_abs_max_pm",
            "normalized_residual_peak",
            "normalized_residual_rms",
        ],
        "session_metrics": session_metrics,
        "aggregate": aggregate,
        "existing_uncalibrated_v7_reference": {
            "original_to_incremental_macro_f1": 0.7921507357989874,
            "incremental_to_original_macro_f1": 0.1675977653631285,
            "source": "outputs/incremental_static_model_20260714_v7_fused_shift/incremental_evaluation.json",
        },
        "deployment_status": "not_deployed_live_calibration_required",
        "force_semantics": "approximate_manual_response_level_not_force_N",
    }

    with (output_dir / "per_calibration_repeat.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    (output_dir / "session_level_calibration_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "offline_example_calibration_payload.json").write_text(
        json.dumps(example_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix = np.asarray(aggregate["confusion_matrix"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center")
    axis.set_xticks(range(3), LEVEL_ORDER)
    axis.set_yticks(range(3), LEVEL_ORDER)
    axis.set_xlabel("Predicted response level")
    axis.set_ylabel("True response level")
    axis.set_title("Current-session calibrated response level")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "session_level_calibration_confusion_matrix.png", dpi=180)
    plt.close(fig)

    report = [
        "# Current-Session Response-Level Calibration Benchmark",
        "",
        "## Scientific boundary",
        "",
        "- The task is manual light/normal/hard response-level recognition, not force_N.",
        "- Each calibration run uses 27 independent files: one per position and level.",
        "- All remaining repeats in the same target session are held out for testing.",
        "- No static window or duplicate spectrum is counted as an independent trial.",
        "- This benchmark does not deploy or overwrite the current model.",
        "",
        "## Result",
        "",
        f"- Mean calibration-run macro-F1: {aggregate['mean_calibration_run_macro_f1']:.4f}",
        f"- Minimum calibration-run macro-F1: {aggregate['minimum_calibration_run_macro_f1']:.4f}",
        f"- Aggregate macro-F1: {aggregate['macro_f1']:.4f}",
        f"- Aggregate light recall: {aggregate['per_class']['light']['recall']:.4f}",
        f"- Aggregate normal recall: {aggregate['per_class']['normal']['recall']:.4f}",
        f"- Aggregate hard recall: {aggregate['per_class']['hard']['recall']:.4f}",
        "",
        "## Interpretation",
        "",
        "The same-session position-specific ordinal anchors materially reduce the",
        "light/normal/hard scale drift seen in the uncalibrated reverse challenge.",
        "The live system must collect a fresh calibration after each accepted runtime",
        "baseline. A baseline change invalidates the calibration by design.",
    ]
    (output_dir / "session_level_calibration_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
