"""Build a non-deployed hierarchical candidate from grouped OOF evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_live_cadence_models import (  # noqa: E402
    CONTACT_ORDER,
    POSITION_ORDER,
    RESPONSE_ORDER,
    make_summary_model,
    metrics_for,
)
from src.hybrid_spectrum.dynamic_temporal_features import (  # noqa: E402
    SUMMARY_FEATURE_BLOCK_ORDER,
)
from src.hybrid_spectrum.live_cadence_dataset import (  # noqa: E402
    build_live_cadence_dataset,
    causal_summary_features,
)
from src.hybrid_spectrum.live_cadence_models import (  # noqa: E402
    set_single_thread_prediction,
)


POSITION_ENSEMBLE = (
    "factorized_extra_trees",
    "lightgbm_summary",
    "logistic_summary",
)
RESPONSE_ENSEMBLE = (
    "extra_trees_summary",
    "lightgbm_summary",
    "ordinal_logistic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dynamic_sequence_audit_20260714_v2",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "live_cadence_algorithm_benchmark_20260716_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "candidates"
        / "live_cadence_hierarchical_shadow_candidate_v1.joblib",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def priority_majority_vote(values: np.ndarray) -> np.ndarray:
    """Majority vote with the first model as deterministic all-disagree fallback."""

    output: list[str] = []
    for row in np.asarray(values).astype(str):
        labels, counts = np.unique(row, return_counts=True)
        if int(np.max(counts)) == 1:
            output.append(str(row[0]))
        else:
            output.append(str(labels[int(np.argmax(counts))]))
    return np.asarray(output)


def ensemble_oof(
    predictions: pd.DataFrame,
    *,
    task: str,
    history_frames: int,
    models: tuple[str, ...],
    label_order: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = predictions[
        (predictions["task"] == task)
        & (predictions["history_frames"] == history_frames)
        & predictions["model_id"].isin(models)
    ]
    index_columns = [
        "file_id",
        "target_frame_index",
        "capture_group",
        "true_label",
    ]
    pivot = source.pivot_table(
        index=index_columns,
        columns="model_id",
        values="predicted_label",
        aggfunc="first",
    ).dropna(subset=list(models))
    output = pivot.reset_index()[index_columns].copy()
    output["predicted_label"] = priority_majority_vote(pivot[list(models)].to_numpy())
    truth = output["true_label"].to_numpy(str)
    predicted = output["predicted_label"].to_numpy(str)
    metrics = metrics_for(truth, predicted, label_order)
    folds = []
    for group, group_rows in output.groupby("capture_group"):
        fold = metrics_for(
            group_rows["true_label"].to_numpy(str),
            group_rows["predicted_label"].to_numpy(str),
            label_order,
        )
        fold["test_group"] = str(group)
        folds.append(fold)
    metrics.update(
        {
            "task": task,
            "history_frames": history_frames,
            "ensemble_members": list(models),
            "tie_break": "first_member_when_all_models_disagree",
            "minimum_capture_group_macro_f1": float(
                min(fold["macro_f1"] for fold in folds)
            ),
            "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
            "evaluation_validity": "grouped_by_capture_group_and_file_id",
            "folds": folds,
        }
    )
    return output, metrics


def save_confusion(path: Path, metrics: dict[str, Any]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["label_order"]
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{metrics['task']} grouped OOF ensemble")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def result_lookup(metrics_payload: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(row["task"]), str(row["model_id"]), int(row["history_frames"])): row
        for row in metrics_payload["results"]
        if "missing_reason" not in row
    }


def summed_compute_p95(
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    task: str,
    history: int,
    models: tuple[str, ...],
) -> float:
    return float(
        sum(float(lookup[(task, model, history)]["compute_latency_ms_p95"]) for model in models)
    )


def write_report(
    path: Path,
    contact: dict[str, Any],
    position: dict[str, Any],
    response: dict[str, Any],
    combined: dict[str, Any],
    latency: dict[str, float],
    model_path: Path,
) -> None:
    lines = [
        "# Live-cadence hierarchical shadow candidate",
        "",
        "- Status: shadow only; not deployed and not allowed to replace the current desktop model.",
        "- Data: 27 independent dynamic DAT files; blind-test files were not used.",
        "- Evaluation: leave-one-capture-group-out G1/G2/G3 with original file_id exclusivity.",
        "- Live input assumption: one real physical spectrum approximately every 0.40 s; no interpolated spectra are treated as measurements.",
        "- Response labels are approximate light/normal/hard levels, not force_N.",
        "",
        "## Candidate structure",
        "",
        "| Output | Physical history | Model | Macro-F1 | Min-group macro-F1 | Compute p95 |",
        "|---|---:|---|---:|---:|---:|",
        f"| Contact | 1 frame | ExtraTrees | {contact['macro_f1']:.4f} | {contact['minimum_capture_group_macro_f1']:.4f} | {latency['contact_compute_p95_ms']:.2f} ms |",
        f"| Position | 1 frame | Factorized ExtraTrees + LightGBM + Logistic hard vote | {position['macro_f1']:.4f} | {position['minimum_capture_group_macro_f1']:.4f} | {latency['position_compute_p95_ms']:.2f} ms |",
        f"| Response level | 2 frames | ExtraTrees + LightGBM + ordinal Logistic hard vote | {response['macro_f1']:.4f} | {response['minimum_capture_group_macro_f1']:.4f} | {latency['response_compute_p95_ms']:.2f} ms |",
        "",
        f"- Combined exact position + response-level accuracy on common OOF targets: {combined['exact_accuracy']:.4f}.",
        "- First contact/position output becomes possible after the first physical spectrum (nominal cold-start 0.40 s).",
        "- Response level becomes possible after two physical spectra (nominal cold-start 0.80 s).",
        "- Once warm, all outputs update at the physical acquisition period of about 0.40 s.",
        "",
        "## Comparison with current long-window candidate",
        "",
        "The current 20 x 40 ms model has position macro-F1 about 0.911 and response-level macro-F1 about 0.862, but live operation fills its dense history using interpolation between much slower physical spectra. The new candidate keeps nearly the same grouped-CV accuracy using one or two actual physical frames. This removes the false assumption that interpolation creates additional measurements.",
        "",
        "## Safety limits",
        "",
        "- Ensemble membership was selected after inspecting the same grouped CV and therefore still needs a fresh independent capture session before primary deployment.",
        "- Stable no-contact and stable press plateaus are evaluated; onset transition and release-residual behavior are not yet a complete deployment gate.",
        "- Only 27 independent dynamic files are available, despite hundreds of causal target windows.",
        "- No hardware or UI code was changed in this step.",
        "",
        f"Candidate artifact: `{model_path}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    confusion_dir = output_dir / "confusion_matrices"
    confusion_dir.mkdir()
    benchmark_dir = args.benchmark_dir.resolve()
    predictions = pd.read_csv(benchmark_dir / "grouped_predictions.csv")
    benchmark_metrics = json.loads(
        (benchmark_dir / "live_cadence_model_metrics.json").read_text(encoding="utf-8")
    )
    lookup = result_lookup(benchmark_metrics)

    contact_rows = predictions[
        (predictions["task"] == "contact")
        & (predictions["history_frames"] == 1)
        & (predictions["model_id"] == "extra_trees_summary")
    ].copy()
    contact_metrics = metrics_for(
        contact_rows["true_label"].to_numpy(str),
        contact_rows["predicted_label"].to_numpy(str),
        CONTACT_ORDER,
    )
    contact_folds = []
    for group, group_rows in contact_rows.groupby("capture_group"):
        fold = metrics_for(
            group_rows["true_label"].to_numpy(str),
            group_rows["predicted_label"].to_numpy(str),
            CONTACT_ORDER,
        )
        fold["test_group"] = str(group)
        contact_folds.append(fold)
    contact_metrics.update(
        {
            "task": "contact",
            "history_frames": 1,
            "minimum_capture_group_macro_f1": min(
                fold["macro_f1"] for fold in contact_folds
            ),
            "folds": contact_folds,
            "evaluation_validity": "grouped_by_capture_group_and_file_id",
        }
    )
    position_oof, position_metrics = ensemble_oof(
        predictions,
        task="position",
        history_frames=1,
        models=POSITION_ENSEMBLE,
        label_order=POSITION_ORDER,
    )
    response_oof, response_metrics = ensemble_oof(
        predictions,
        task="response_level",
        history_frames=2,
        models=RESPONSE_ENSEMBLE,
        label_order=RESPONSE_ORDER,
    )
    combined_rows = position_oof.rename(
        columns={"true_label": "true_position", "predicted_label": "predicted_position"}
    ).merge(
        response_oof.rename(
            columns={"true_label": "true_response", "predicted_label": "predicted_response"}
        ),
        on=["file_id", "target_frame_index", "capture_group"],
        how="inner",
    )
    combined_rows["exact_match"] = (
        (combined_rows["true_position"] == combined_rows["predicted_position"])
        & (combined_rows["true_response"] == combined_rows["predicted_response"])
    )
    combined_metrics = {
        "common_target_count": int(len(combined_rows)),
        "exact_accuracy": float(combined_rows["exact_match"].mean()),
        "per_capture_group": {
            str(group): float(rows["exact_match"].mean())
            for group, rows in combined_rows.groupby("capture_group")
        },
    }
    latency = {
        "contact_compute_p95_ms": float(
            lookup[("contact", "extra_trees_summary", 1)]["compute_latency_ms_p95"]
        ),
        "position_compute_p95_ms": summed_compute_p95(
            lookup, "position", 1, POSITION_ENSEMBLE
        ),
        "response_compute_p95_ms": summed_compute_p95(
            lookup, "response_level", 2, RESPONSE_ENSEMBLE
        ),
    }

    dataset_one = build_live_cadence_dataset(args.audit_dir, 1)
    dataset_two = build_live_cadence_dataset(args.audit_dir, 2)
    summary_one = causal_summary_features(dataset_one.values)
    summary_two = causal_summary_features(dataset_two.values)
    contact_model = make_summary_model(
        "extra_trees_summary", "contact", args.random_state + 10
    ).fit(summary_one, dataset_one.contact_labels)
    contact_mask_one = dataset_one.contact_labels == "contact"
    position_models = []
    for index, model_id in enumerate(POSITION_ENSEMBLE):
        model = make_summary_model(
            model_id, "position", args.random_state + 100 + index
        ).fit(
            summary_one[contact_mask_one],
            dataset_one.position_labels[contact_mask_one],
        )
        position_models.append((model_id, set_single_thread_prediction(model)))
    contact_mask_two = dataset_two.contact_labels == "contact"
    response_models = []
    for index, model_id in enumerate(RESPONSE_ENSEMBLE):
        model = make_summary_model(
            model_id, "response_level", args.random_state + 200 + index
        ).fit(
            summary_two[contact_mask_two],
            dataset_two.stage_labels[contact_mask_two],
        )
        response_models.append((model_id, set_single_thread_prediction(model)))
    set_single_thread_prediction(contact_model)

    bundle = {
        "schema_version": "live_cadence_hierarchical_shadow_candidate_v1",
        "status": "shadow_only_not_primary",
        "deployment_ready": False,
        "hardware_opened_during_build": False,
        "blind_test_used": False,
        "source_audit_dir": str(args.audit_dir.resolve()),
        "independent_dat_file_count": 27,
        "split_strategy": "leave_one_capture_group_out_G1_G2_G3",
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "source_frame_interval_sec": 0.04,
        "live_frame_interval_sec": 0.40,
        "frame_feature_names": dataset_one.feature_names,
        "summary_feature_block_order": SUMMARY_FEATURE_BLOCK_ORDER,
        "contact": {
            "history_frames": 1,
            "model_id": "extra_trees_summary",
            "model": contact_model,
            "label_order": CONTACT_ORDER,
            "metrics": contact_metrics,
        },
        "position": {
            "history_frames": 1,
            "models": position_models,
            "vote": "hard_majority_first_member_all_disagree_fallback",
            "label_order": POSITION_ORDER,
            "metrics": position_metrics,
        },
        "response_level": {
            "history_frames": 2,
            "models": response_models,
            "vote": "hard_majority_first_member_all_disagree_fallback",
            "label_order": RESPONSE_ORDER,
            "metrics": response_metrics,
        },
        "combined_metrics": combined_metrics,
        "latency": latency,
        "known_limitations": [
            "candidate membership selected on the same grouped CV evidence",
            "only 27 independent dynamic DAT files",
            "release residual is not yet a complete deployment gate",
            "not evaluated on paused blind-test files",
        ],
    }
    model_path = args.model_output.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)

    position_oof.to_csv(output_dir / "position_ensemble_oof_predictions.csv", index=False)
    response_oof.to_csv(output_dir / "response_ensemble_oof_predictions.csv", index=False)
    combined_rows.to_csv(output_dir / "combined_oof_predictions.csv", index=False)
    metrics_payload = {
        "contact": contact_metrics,
        "position": position_metrics,
        "response_level": response_metrics,
        "combined": combined_metrics,
        "latency": latency,
        "candidate_model": str(model_path),
        "deployment_ready": False,
    }
    (output_dir / "shadow_candidate_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_confusion(confusion_dir / "contact.png", contact_metrics)
    save_confusion(confusion_dir / "position.png", position_metrics)
    save_confusion(confusion_dir / "response_level.png", response_metrics)
    write_report(
        output_dir / "shadow_candidate_report.md",
        contact_metrics,
        position_metrics,
        response_metrics,
        combined_metrics,
        latency,
        model_path,
    )
    manifest = [
        {
            "artifact": "candidate_model",
            "path": str(model_path),
            "status": "shadow_only_not_primary",
        },
        {
            "artifact": "grouped_metrics",
            "path": str(output_dir / "shadow_candidate_metrics.json"),
            "status": "generated",
        },
    ]
    with (output_dir / "artifact_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"saved candidate: {model_path}")
    print(f"saved report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
