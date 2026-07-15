"""Audit runtime-model sensitivity to current-session and stale baselines.

This script is read-only with respect to source data and deployed models.  It
compares a deployed bundle with a candidate bundle, then writes an explicit
deployment decision instead of silently replacing the live model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.sense_static_dataset import (  # noqa: E402
    assert_dataset_manifest_stable,
    build_static_feature_dataset,
    dataset_source_manifest,
    extract_snapshot_feature_vectors,
    load_sense_dataset,
    load_training_config,
)
from src.hybrid_spectrum.spatial_fingerprint import (  # noqa: E402
    spatial_fingerprint_from_engineered,
)
from src.hybrid_spectrum.static_model_adapter import (  # noqa: E402
    StaticSpectralPredictor,
)


POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
FORCE_ORDER = ["light", "normal", "hard"]
EXPECTED_DEPLOYED_SHA256 = "CEB367AC583152DAFEB33091E92C70CD34417CE54E281DBCFC6FA6D909225251"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "sense_static_training.yaml"
    )
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=ROOT / "config" / "hybrid_spectrum_channels.yaml",
    )
    parser.add_argument(
        "--deployed-bundle",
        type=Path,
        default=ROOT / "models" / "static_spectral_recognition_bundle.joblib",
    )
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def model_matrix(rows: dict[str, list[dict[str, float]]], payload: dict[str, Any]) -> np.ndarray:
    source = rows[payload["feature_set"]]
    return np.asarray(
        [[row[column] for column in payload["feature_columns"]] for row in source],
        dtype=float,
    )


def build_rows(
    bundle: dict[str, Any],
    records: list[Any],
    baseline_spectrum: np.ndarray,
) -> dict[str, list[dict[str, float]]]:
    grid = np.asarray(bundle["common_wavelength_nm"], dtype=float)
    baseline_grid = np.asarray(baseline_spectrum, dtype=float)
    if baseline_grid.shape != grid.shape:
        raise ValueError("baseline and bundle wavelength grid shapes differ")
    rows: dict[str, list[dict[str, float]]] = {
        "engineered": [],
        "full_hybrid": [],
        "current_shape": [],
        "spatial_fingerprint": [],
    }
    for record in records:
        current = np.interp(grid, record.wavelength_nm, record.intensity_counts)
        engineered, hybrid = extract_snapshot_feature_vectors(
            grid,
            current,
            baseline_grid,
            bundle["peak_windows"],
            int(bundle["full_spectrum_bins"]),
        )
        rows["engineered"].append(engineered)
        rows["full_hybrid"].append(hybrid)
        rows["current_shape"].append(hybrid)
        rows["spatial_fingerprint"].append(
            spatial_fingerprint_from_engineered(engineered)
        )
    return rows


def predict_bundle(bundle: dict[str, Any], rows: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    models = bundle["models"]
    contact_payload = models["contact_detector"]
    raw_contact = contact_payload["model"].predict(model_matrix(rows, contact_payload))
    contact = []
    for label, engineered in zip(raw_contact, rows["engineered"], strict=True):
        evidence = StaticSpectralPredictor._baseline_relative_contact_evidence(engineered)
        resolved = StaticSpectralPredictor._resolve_contact_decision(
            {
                "label": str(label),
                "confidence": None,
                "margin": None,
                "probabilities": {},
                "confidence_source": "batch_baseline_audit",
                "probability_calibrated": False,
                "review_needed": False,
            },
            evidence,
        )
        contact.append(resolved["label"])

    position_payload = models["position_classifier"]
    position_matrix = model_matrix(rows, position_payload)
    position_model = position_payload["model"]
    position = np.asarray(position_model.predict(position_matrix), dtype=object)
    if hasattr(position_model, "predict_diagnostics"):
        diagnostics = position_model.predict_diagnostics(position_matrix)
        unanimous = np.asarray([bool(row["unanimous"]) for row in diagnostics])
        agreement = np.asarray([float(row["agreement_fraction"]) for row in diagnostics])
    else:
        unanimous = np.ones(position.size, dtype=bool)
        agreement = np.ones(position.size, dtype=float)

    if "position_conditioned_force_classifier" in models:
        force_payload = models["position_conditioned_force_classifier"]
        force_matrix = model_matrix(rows, force_payload)
        force = np.empty(position.size, dtype=object)
        scopes = np.empty(position.size, dtype=object)
        for position_id in np.unique(position):
            mask = position == position_id
            force[mask] = force_payload["models_by_position"][str(position_id)].predict(
                force_matrix[mask]
            )
            scopes[mask] = f"position_conditioned:{position_id}"
    else:
        force_payload = models["manual_force_classifier"]
        force = np.asarray(
            force_payload["model"].predict(model_matrix(rows, force_payload)),
            dtype=object,
        )
        scopes = np.asarray(["global_manual_fallback"] * position.size, dtype=object)

    return {
        "contact": np.asarray(contact, dtype=object),
        "position": position,
        "force": force,
        "force_scope": scopes,
        "position_unanimous": unanimous,
        "position_agreement": agreement,
    }


def task_metrics(
    truth_position: np.ndarray,
    truth_force: np.ndarray,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    position = prediction["position"]
    force = prediction["force"]
    force_recalls = recall_score(
        truth_force,
        force,
        labels=FORCE_ORDER,
        average=None,
        zero_division=0,
    )
    return {
        "contact_recall": float(np.mean(prediction["contact"] == "contact")),
        "position_accuracy": float(accuracy_score(truth_position, position)),
        "position_macro_f1": float(
            f1_score(truth_position, position, labels=POSITION_ORDER, average="macro", zero_division=0)
        ),
        "position_unanimous_fraction": float(np.mean(prediction["position_unanimous"])),
        "position_unanimous_accuracy": (
            float(
                np.mean(
                    position[prediction["position_unanimous"]]
                    == truth_position[prediction["position_unanimous"]]
                )
            )
            if np.any(prediction["position_unanimous"])
            else None
        ),
        "force_accuracy": float(accuracy_score(truth_force, force)),
        "force_macro_f1": float(
            f1_score(truth_force, force, labels=FORCE_ORDER, average="macro", zero_division=0)
        ),
        "light_recall": float(force_recalls[0]),
        "normal_recall": float(force_recalls[1]),
        "hard_recall": float(force_recalls[2]),
        "force_scope_counts": dict(Counter(prediction["force_scope"])),
    }


def select_assigned_predictions(
    predictions_by_cluster: dict[str, dict[str, Any]],
    assigned_cluster_ids: list[str],
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for field in (
        "contact",
        "position",
        "force",
        "force_scope",
        "position_unanimous",
        "position_agreement",
    ):
        selected[field] = np.asarray(
            [
                predictions_by_cluster[cluster_id][field][index]
                for index, cluster_id in enumerate(assigned_cluster_ids)
            ]
        )
    return selected


def write_summary_plot(rows: list[dict[str, Any]], destination: Path) -> None:
    metrics = ["position_macro_f1", "force_macro_f1", "hard_recall"]
    labels = [f"{row['model']}\n{row['baseline_scenario']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(12.0, len(rows) * 1.25), 6.2))
    colors = ["#2f7f9f", "#56a67a", "#d6a23a"]
    for index, metric in enumerate(metrics):
        axis.bar(
            x + (index - 1) * width,
            [float(row[metric]) for row in rows],
            width,
            label=metric,
            color=colors[index],
        )
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(x, labels=labels, rotation=35, ha="right")
    axis.set_ylabel("Score")
    axis.set_title("Runtime bundle sensitivity to no-contact baseline selection")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = load_training_config(args.config.resolve())
    manifest_before = dataset_source_manifest(config)
    records = tuple(load_sense_dataset(config))
    manifest_after = dataset_source_manifest(config)
    assert_dataset_manifest_stable(manifest_before, manifest_after)
    dataset = build_static_feature_dataset(
        records,
        config.get("feature_extraction", config),
        args.channel_config.resolve(),
    )
    manual_indices = np.asarray(
        [index for index, record in enumerate(records) if record.sample_kind == "manual_press"],
        dtype=int,
    )
    manual_records = [records[index] for index in manual_indices]
    truth_position = np.asarray([record.position_label for record in manual_records], dtype=object)
    truth_force = np.asarray([record.manual_force_label for record in manual_records], dtype=object)
    reference_clusters = list(dataset.reference_baseline_clusters)
    assigned_cluster_ids = [
        dataset.baseline_reference_mode[index].split("_nearest_trusted")[0]
        for index in manual_indices
    ]

    bundle_paths = {
        "deployed": args.deployed_bundle.resolve(),
        "candidate": args.candidate_bundle.resolve(),
    }
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    all_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    bundles: dict[str, dict[str, Any]] = {}
    for model_name, bundle_path in bundle_paths.items():
        bundle = joblib.load(bundle_path)
        bundles[model_name] = bundle
        predictions_by_cluster: dict[str, dict[str, Any]] = {}
        for cluster in reference_clusters:
            rows = build_rows(bundle, manual_records, cluster.median_spectrum)
            prediction = predict_bundle(bundle, rows)
            predictions_by_cluster[cluster.cluster_id] = prediction
            metrics = task_metrics(truth_position, truth_force, prediction)
            summary_rows.append(
                {
                    "model": model_name,
                    "baseline_scenario": cluster.cluster_id,
                    **metrics,
                }
            )
            for index, record in enumerate(manual_records):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "baseline_scenario": cluster.cluster_id,
                        "file_id": record.file_id,
                        "true_position": truth_position[index],
                        "predicted_position": prediction["position"][index],
                        "position_correct": truth_position[index] == prediction["position"][index],
                        "position_agreement": prediction["position_agreement"][index],
                        "true_force": truth_force[index],
                        "predicted_force": prediction["force"][index],
                        "force_correct": truth_force[index] == prediction["force"][index],
                        "force_scope": prediction["force_scope"][index],
                        "contact_prediction": prediction["contact"][index],
                    }
                )

        assigned = select_assigned_predictions(predictions_by_cluster, assigned_cluster_ids)
        assigned_metrics = task_metrics(truth_position, truth_force, assigned)
        position_stable = np.asarray(
            [
                len(
                    {
                        predictions_by_cluster[cluster.cluster_id]["position"][index]
                        for cluster in reference_clusters
                    }
                )
                == 1
                for index in range(len(manual_records))
            ]
        )
        force_stable = np.asarray(
            [
                len(
                    {
                        predictions_by_cluster[cluster.cluster_id]["force"][index]
                        for cluster in reference_clusters
                    }
                )
                == 1
                for index in range(len(manual_records))
            ]
        )
        assigned_metrics.update(
            {
                "position_same_prediction_all_baselines": float(np.mean(position_stable)),
                "force_same_prediction_all_baselines": float(np.mean(force_stable)),
            }
        )
        summary_rows.append(
            {
                "model": model_name,
                "baseline_scenario": "assigned_current_session_reference",
                **assigned_metrics,
            }
        )
        all_predictions[model_name] = predictions_by_cluster

    summary_columns = sorted({key for row in summary_rows for key in row})
    with (output_dir / "baseline_sensitivity_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "baseline_sensitivity_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    write_summary_plot(summary_rows, output_dir / "baseline_sensitivity_comparison.png")

    deployed_hash = sha256(bundle_paths["deployed"])
    candidate_assigned = next(
        row
        for row in summary_rows
        if row["model"] == "candidate"
        and row["baseline_scenario"] == "assigned_current_session_reference"
    )
    formal = bundles["candidate"].get("incremental_evaluation") or {}
    formal_position = formal.get("position_original_to_incremental_challenge") or {}
    formal_force = formal.get("force_original_to_incremental_challenge") or {}
    hard_recall = float(
        ((formal_force.get("per_class") or {}).get("hard") or {}).get("recall", 0.0)
    )
    checks = {
        "deployed_bundle_unchanged": deployed_hash == EXPECTED_DEPLOYED_SHA256,
        "candidate_uses_global_force_head": (
            "position_conditioned_force_classifier" not in bundles["candidate"]["models"]
        ),
        "formal_position_macro_f1_at_least_0_70": float(formal_position.get("macro_f1", 0.0)) >= 0.70,
        "formal_force_macro_f1_at_least_0_75": float(formal_force.get("macro_f1", 0.0)) >= 0.75,
        "formal_hard_recall_at_least_0_90": hard_recall >= 0.90,
        "position_prediction_stability_across_baselines_at_least_0_75": float(
            candidate_assigned.get("position_same_prediction_all_baselines", 0.0)
        )
        >= 0.75,
        "force_prediction_stability_across_baselines_at_least_0_65": float(
            candidate_assigned.get("force_same_prediction_all_baselines", 0.0)
        )
        >= 0.65,
    }
    decision = {
        "schema_version": "runtime_candidate_baseline_sensitivity_decision_v1",
        "deployed_bundle_path": str(bundle_paths["deployed"]),
        "deployed_bundle_sha256": deployed_hash,
        "candidate_bundle_path": str(bundle_paths["candidate"]),
        "candidate_bundle_sha256": sha256(bundle_paths["candidate"]),
        "source_csv_count": len(records),
        "manual_press_csv_count": len(manual_records),
        "trusted_baseline_cluster_ids": [cluster.cluster_id for cluster in reference_clusters],
        "checks": checks,
        "deployment_ready": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "decision_policy": "candidate_only_until_all_offline_checks_and_live_validation_pass",
        "formal_cross_batch_position": formal_position,
        "formal_cross_batch_force": formal_force,
        "assigned_baseline_resubstitution_warning": (
            "assigned-current-session scores use records included in the final candidate fit; "
            "they verify feature/runtime consistency but are not a generalization estimate"
        ),
    }
    (output_dir / "candidate_integration_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "source_dataset_manifest.json").write_text(
        json.dumps(manifest_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Runtime candidate baseline-sensitivity audit",
        "",
        "- This is the ordinary-FBG BaySpec full-spectrum recognition edition.",
        "- Each CSV is one static spectrum snapshot; 512 values are wavelength bins, not time steps.",
        f"- Independent static CSV files: {len(records)}; manual press files: {len(manual_records)}.",
        "- Current deployed model was not replaced by this audit.",
        "- light / normal / hard are approximate manual response levels, not force_N.",
        "",
        "## Result",
        "",
        f"- Deployment ready: **{decision['deployment_ready']}**",
        f"- Failed checks: {', '.join(decision['failed_checks']) or 'none'}",
        f"- Formal candidate position macro-F1: {float(formal_position.get('macro_f1', 0.0)):.4f}",
        f"- Formal candidate force macro-F1: {float(formal_force.get('macro_f1', 0.0)):.4f}",
        f"- Formal candidate hard recall: {hard_recall:.4f}",
        f"- Candidate position same-prediction rate across trusted baselines: {float(candidate_assigned.get('position_same_prediction_all_baselines', 0.0)):.4f}",
        f"- Candidate force same-prediction rate across trusted baselines: {float(candidate_assigned.get('force_same_prediction_all_baselines', 0.0)):.4f}",
        "",
        "## Interpretation",
        "",
        "A stable, current-session post-release no-contact baseline is part of the model input contract.",
        "Changing to an old session baseline is intentionally audited as a stale-baseline stress test.",
        "The assigned-current-session score is a runtime consistency check, not a formal generalization score, because the final candidate is fitted on those files.",
        "A candidate remains undeployed whenever baseline sensitivity or live validation is unresolved.",
    ]
    (output_dir / "baseline_sensitivity_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
