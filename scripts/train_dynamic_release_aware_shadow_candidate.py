"""Add a conservatively validated release-residual guard to the v1 shadow model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    DynamicFeatureSequence,
    load_dynamic_config,
    load_dynamic_feature_sequences,
)
from src.hybrid_spectrum.dynamic_shadow_adapter import (  # noqa: E402
    ReleaseResidualGuard,
    _aligned_probability,
    load_dynamic_shadow_bundle,
)
from src.hybrid_spectrum.dynamic_temporal_features import (  # noqa: E402
    temporal_summary_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-bundle",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "dynamic_temporal_shadow_candidate_v1.joblib",
    )
    parser.add_argument(
        "--sequence-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "dynamic_temporal_shadow_candidate_v2.joblib",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--response-trees", type=int, default=500)
    parser.add_argument("--release-trees", type=int, default=500)
    return parser.parse_args()


def response_tree(seed: int, tree_count: int = 500) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(tree_count),
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def response_svm(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    C=4.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def release_event_model(seed: int, tree_count: int = 500) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(tree_count),
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def rolling_block(
    sequence: DynamicFeatureSequence,
    time_steps: int,
) -> dict[str, Any]:
    windows = np.lib.stride_tricks.sliding_window_view(
        sequence.feature_matrix,
        time_steps,
        axis=0,
    ).transpose(0, 2, 1)
    end_frames = np.arange(time_steps - 1, len(sequence.feature_matrix))
    release_segment = next(
        segment for segment in sequence.stage_segments if segment.label == "release"
    )
    return {
        "sequence": sequence,
        "summary": temporal_summary_features(windows),
        "end_frames": end_frames,
        "release_start": int(release_segment.start_frame),
    }


def event_training_data(
    blocks: list[dict[str, Any]],
    *,
    excluded_group: str | None,
    horizon_frames: int,
    stride_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for block in blocks:
        sequence = block["sequence"]
        if excluded_group is not None and sequence.record.capture_group == excluded_group:
            continue
        end_frames = block["end_frames"]
        release_start = int(block["release_start"])
        eligible = np.flatnonzero(end_frames < release_start + horizon_frames)[
            ::stride_frames
        ]
        values.append(block["summary"][eligible])
        labels.append(
            (
                (end_frames[eligible] >= release_start)
                & (end_frames[eligible] < release_start + horizon_frames)
            ).astype(int)
        )
    return np.vstack(values), np.concatenate(labels)


def release_probability(model: Any, values: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(values), dtype=float)
    positive_index = next(
        index
        for index, label in enumerate(model.classes_)
        if label == 1 or str(label) == "1"
    )
    return probability[:, positive_index]


def run_release_guard(
    block: dict[str, Any],
    response_probability: np.ndarray,
    event_probability: np.ndarray,
    guard_config: dict[str, Any],
) -> dict[str, Any]:
    guard = ReleaseResidualGuard(guard_config)
    end_frames = block["end_frames"]
    release_start = int(block["release_start"])
    trigger_frame: int | None = None
    for index, frame in enumerate(end_frames):
        state = guard.update(
            hard_probability=float(response_probability[index]),
            release_event_probability=float(event_probability[index]),
        )
        if state["just_latched"]:
            trigger_frame = int(frame)
            break

    release_frames = end_frames[end_frames >= release_start]
    if trigger_frame is None:
        remaining_false_contact = len(release_frames)
        detection_delay_frames = None
        result = "missed_release"
    elif trigger_frame < release_start:
        remaining_false_contact = 0
        detection_delay_frames = trigger_frame - release_start
        result = "unsafe_early_trigger"
    else:
        remaining_false_contact = int(np.count_nonzero(release_frames < trigger_frame))
        detection_delay_frames = trigger_frame - release_start
        result = "release_latched"

    sequence = block["sequence"]
    return {
        "file_id": sequence.record.file_id,
        "capture_group": sequence.record.capture_group,
        "position_label": sequence.record.position_label,
        "release_observed_by_segmentation": sequence.release_observed,
        "release_recovered_by_segmentation": sequence.release_recovered,
        "release_recovery_ratio": sequence.release_recovery_ratio,
        "release_start_frame": release_start,
        "release_trigger_frame": trigger_frame,
        "release_detection_delay_frames": detection_delay_frames,
        "release_detection_delay_sec": (
            None
            if detection_delay_frames is None
            else detection_delay_frames * 0.04
        ),
        "release_window_count": len(release_frames),
        "release_false_contact_frames_after_guard": remaining_false_contact,
        "guard_result": result,
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    base_path = args.base_bundle.resolve()
    base = load_dynamic_shadow_bundle(base_path)
    if base["schema_version"] != "dynamic_temporal_shadow_candidate_v1":
        raise ValueError("release-aware packaging requires the v1 base bundle")
    config_path = args.sequence_config.resolve()
    config = load_dynamic_config(config_path)
    sequences = load_dynamic_feature_sequences(config)
    time_steps = int(base["time_steps"])
    if any(sequence.feature_names != tuple(base["frame_feature_names"]) for sequence in sequences):
        raise ValueError("raw sequence features do not match the v1 bundle contract")
    blocks = [rolling_block(sequence, time_steps) for sequence in sequences]

    source = np.load(Path(base["source_dataset_path"]))
    stable_values = np.asarray(source["X"], dtype=np.float32)
    stable_summary = temporal_summary_features(stable_values)
    stable_stage = source["stage_labels"].astype(str)
    stable_groups = source["capture_groups"].astype(str)
    response_labels = list(base["label_order"]["response_level"])
    response_mask = np.isin(stable_stage, response_labels)

    guard_config = {
        "enabled": True,
        "hard_arm_probability": 0.50,
        "hard_arm_frames": 24,
        "hard_exit_probability": 0.45,
        "hard_exit_frames": 1,
        "release_event_probability": 0.40,
        "event_horizon_frames": 30,
        "event_training_stride_frames": 2,
        "baseline_reset_required_after_latch": True,
        "validation_scope": "no_contact_to_light_to_normal_to_hard_to_release_only",
    }

    rows: list[dict[str, Any]] = []
    for fold_index, held_group in enumerate(("G1", "G2", "G3")):
        train = response_mask & (stable_groups != held_group)
        response_seed = 1100 + fold_index
        event_seed = 1200 + fold_index
        tree = response_tree(response_seed, args.response_trees).fit(
            stable_summary[train], stable_stage[train]
        )
        svm = response_svm(response_seed).fit(
            stable_summary[train], stable_stage[train]
        )
        event_x, event_y = event_training_data(
            blocks,
            excluded_group=held_group,
            horizon_frames=int(guard_config["event_horizon_frames"]),
            stride_frames=int(guard_config["event_training_stride_frames"]),
        )
        event_model = release_event_model(event_seed, args.release_trees).fit(
            event_x, event_y
        )
        for block in blocks:
            sequence = block["sequence"]
            if sequence.record.capture_group != held_group:
                continue
            values = block["summary"]
            response_probability = 0.5 * _aligned_probability(
                tree,
                values,
                response_labels,
            ) + 0.5 * _aligned_probability(
                svm,
                values,
                response_labels,
            )
            event_probability = release_probability(event_model, values)
            result = run_release_guard(
                block,
                response_probability[:, response_labels.index("hard")],
                event_probability,
                guard_config,
            )
            result["held_out_group"] = held_group
            rows.append(result)

    detected = [row for row in rows if row["guard_result"] == "release_latched"]
    early = [row for row in rows if row["guard_result"] == "unsafe_early_trigger"]
    missed = [row for row in rows if row["guard_result"] == "missed_release"]
    total_release_frames = int(sum(row["release_window_count"] for row in rows))
    false_after_guard = int(
        sum(row["release_false_contact_frames_after_guard"] for row in rows)
    )
    delays = [int(row["release_detection_delay_frames"]) for row in detected]
    metrics = {
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "split_strategy": "leave_one_complete_capture_group_out_G1_G2_G3",
        "independent_sequence_count": len(rows),
        "detected_release_sequence_count": len(detected),
        "missed_release_sequence_count": len(missed),
        "unsafe_early_trigger_sequence_count": len(early),
        "release_sequence_detection_rate": len(detected) / len(rows),
        "median_detection_delay_frames": float(np.median(delays)) if delays else None,
        "maximum_detection_delay_frames": max(delays) if delays else None,
        "median_detection_delay_sec": float(np.median(delays) * 0.04) if delays else None,
        "maximum_detection_delay_sec": max(delays) * 0.04 if delays else None,
        "raw_v1_release_false_contact_frames": total_release_frames,
        "v2_release_false_contact_frames_after_guard": false_after_guard,
        "raw_v1_release_false_contact_rate": 1.0,
        "v2_release_false_contact_rate_after_guard": (
            false_after_guard / total_release_frames
        ),
        "guard_config": guard_config,
        "missed_files": [row["file_id"] for row in missed],
        "early_trigger_files": [row["file_id"] for row in early],
        "threshold_selection_note": (
            "prototype thresholds selected on the current grouped audit; "
            "independent live validation is still required"
        ),
    }
    if early:
        raise RuntimeError(
            "release guard validation produced unsafe pre-release triggers: "
            + ", ".join(row["file_id"] for row in early)
        )

    all_event_x, all_event_y = event_training_data(
        blocks,
        excluded_group=None,
        horizon_frames=int(guard_config["event_horizon_frames"]),
        stride_frames=int(guard_config["event_training_stride_frames"]),
    )
    final_event_model = release_event_model(
        args.random_state + 2000, args.release_trees
    ).fit(
        all_event_x,
        all_event_y,
    )
    models = dict(base["models"])
    models["release_event_extra_trees"] = final_event_model
    blockers = list(base.get("deployment_blockers", ()))
    blockers.extend(
        [
            "release_guard_not_detected_in_all_27_grouped_sequences",
            "release_guard_validated_only_for_standard_hard_then_release_sequence",
            "release_guard_thresholds_require_independent_live_validation",
            "new_trial_requires_fresh_baseline_after_release_latch",
        ]
    )
    bundle = dict(base)
    bundle.update(
        {
            "schema_version": "dynamic_temporal_shadow_candidate_v2",
            "status": "shadow_only_not_primary",
            "deployment_ready": False,
            "deployment_blockers": sorted(set(blockers)),
            "models": models,
            "release_guard": guard_config,
            "release_guard_grouped_cv": metrics,
            "sequence_config_path": str(config_path),
            "sequence_config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "release_guard_model_tree_count": int(args.release_trees),
            "release_validation_response_tree_count": int(args.response_trees),
        }
    )
    model_output = args.model_output.resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output, compress=3)
    joblib.dump(bundle, output_dir / model_output.name, compress=3)

    with (output_dir / "release_guard_grouped_predictions.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "release_guard_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    labels = [f"{row['capture_group']}/{row['position_label']}" for row in rows]
    plotted_delays = [
        row["release_detection_delay_sec"]
        if row["release_detection_delay_sec"] is not None
        else np.nan
        for row in rows
    ]
    colors = [
        "#198c67" if row["guard_result"] == "release_latched" else "#c85a4a"
        for row in rows
    ]
    axis.bar(np.arange(len(rows)), plotted_delays, color=colors)
    axis.set_xticks(np.arange(len(rows)), labels=labels, rotation=60, ha="right")
    axis.set_ylabel("Release latch delay (s)")
    axis.set_title("Grouped offline release-residual guard validation")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "release_guard_detection_delay.png", dpi=180)
    plt.close(figure)

    report = [
        "# Dynamic release-aware shadow candidate v2",
        "",
        "This remains an offline/shadow-only candidate and does not drive the operator UI or digital twin.",
        "",
        "## Grouped full-sequence replay",
        "",
        f"- Independent DAT sequences: {len(rows)}.",
        "- Formal split: leave one complete capture group out; no source DAT crosses train and test.",
        f"- Confirmed release latches: {len(detected)}/{len(rows)}.",
        f"- Unsafe pre-release triggers: {len(early)}.",
        f"- Median release delay: {metrics['median_detection_delay_sec']:.2f} s.",
        f"- Maximum release delay: {metrics['maximum_detection_delay_sec']:.2f} s.",
        f"- Raw v1 release false-contact rate: {metrics['raw_v1_release_false_contact_rate']:.3f}.",
        f"- v2 post-guard release false-contact rate: {metrics['v2_release_false_contact_rate_after_guard']:.3f}.",
        f"- Release-event trees: {args.release_trees}; validation response trees: {args.response_trees}.",
        "",
        "## Interpretation",
        "",
        "The guard separates a confirmed hard-to-release event from a low absolute response. This prevents recovered optical residuals from being interpreted as continued pressure after a detected release.",
        "",
        f"The {len(missed)} missed sequences do not provide sufficiently consistent release evidence under grouped testing. They remain review cases; the software must not claim that they were solved.",
        "",
        "## Boundaries",
        "",
        "- Validated sequence order: no_contact -> light -> normal -> hard -> release.",
        "- Releasing directly from light or normal has not been independently collected or validated.",
        "- A fresh baseline is required before a new trial after the release latch.",
        "- Light/normal/hard are approximate manual response levels, not force_N.",
        "- This candidate remains deployment blocked pending live shadow validation and more independent sequences.",
    ]
    (output_dir / "release_guard_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(model_output)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
