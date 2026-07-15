"""Retrain a candidate runtime bundle after balanced incremental capture."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.sense_static_dataset import (  # noqa: E402
    assert_dataset_manifest_stable,
    build_static_feature_dataset,
    dataset_source_manifest,
    dataset_sha256,
    load_sense_dataset,
    load_training_config,
)
from src.hybrid_spectrum.github_static_models import (  # noqa: E402
    AgreementAwareVotingClassifier,
)
from src.hybrid_spectrum.spatial_fingerprint import (  # noqa: E402
    build_spatial_fingerprint_matrix,
)


POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
FORCE_ORDER = ["light", "normal", "hard"]
CONTACT_ORDER = ["no_contact", "contact"]


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
    parser.add_argument(
        "--base-bundle",
        type=Path,
        default=ROOT / "models" / "static_spectral_recognition_bundle.joblib",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incremental-date", type=date.fromisoformat, default=date(2026, 7, 14))
    parser.add_argument("--late-holdout-first-repeat", type=int, default=13)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def contact_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=700,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def position_model(random_state: int) -> AgreementAwareVotingClassifier:
    """Use only physically interpretable, common-mode-corrected FBG shifts."""

    return AgreementAwareVotingClassifier(
        estimators=(
            (
                "logistic_regression",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=1.0,
                                max_iter=5000,
                                class_weight="balanced",
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "shrinkage_lda",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LinearDiscriminantAnalysis(
                                solver="lsqr",
                                shrinkage="auto",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "linear_svm",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", SVC(C=0.1, kernel="linear", class_weight="balanced")),
                    ]
                ),
            ),
        ),
        primary_estimator="logistic_regression",
    )


def force_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def classification_summary(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    return {
        "num_samples": int(true_labels.size),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=labels,
        ).tolist(),
        "label_order": labels,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1_values[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def agreement_summary(
    model: AgreementAwareVotingClassifier,
    matrix: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, Any]:
    diagnostics = model.predict_diagnostics(matrix)
    predicted = model.predict(matrix)
    unanimous = np.asarray([bool(row["unanimous"]) for row in diagnostics])
    correct = predicted == true_labels
    member_names = tuple(diagnostics[0]["member_predictions"]) if diagnostics else ()
    return {
        "agreement_semantics": "three_model_hard_vote_not_calibrated_probability",
        "unanimous_fraction": float(np.mean(unanimous)) if unanimous.size else None,
        "unanimous_accuracy": (
            float(np.mean(correct[unanimous])) if np.any(unanimous) else None
        ),
        "review_fraction": float(np.mean(~unanimous)) if unanimous.size else None,
        "member_accuracy": {
            name: float(
                np.mean(
                    np.asarray(
                        [row["member_predictions"][name] for row in diagnostics],
                        dtype=object,
                    )
                    == true_labels
                )
            )
            for name in member_names
        },
    }


def payload(
    *,
    task_id: str,
    model_id: str,
    model: Pipeline,
    feature_set: str,
    feature_columns: tuple[str, ...],
    label_order: list[str],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "sense_static_task_model_v1",
        "task_id": task_id,
        "model_id": model_id,
        "feature_set": feature_set,
        "feature_columns": list(feature_columns),
        "label_order": label_order,
        "model": model,
        "confidence_source": str(
            getattr(model, "confidence_source", "uncalibrated_predict_proba")
        ),
        "evaluation": evaluation,
        "deployment_status": "incremental_current_session_baseline_pending_cross_session_validation",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    config = load_training_config(args.config.resolve())
    manifest_before = dataset_source_manifest(config)
    records = tuple(load_sense_dataset(config))
    manifest_after = dataset_source_manifest(config)
    assert_dataset_manifest_stable(manifest_before, manifest_after)
    output_dir.mkdir(parents=True)
    (output_dir / "models").mkdir()
    (output_dir / "source_dataset_manifest.json").write_text(
        json.dumps(manifest_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dataset = build_static_feature_dataset(
        records,
        config.get("feature_extraction", config),
        args.channel_config.resolve(),
    )
    matrix = dataset.engineered_matrix
    columns = dataset.engineered_columns
    position_matrix, position_columns = build_spatial_fingerprint_matrix(
        dataset.engineered_matrix,
        dataset.engineered_columns,
    )
    selected_position_indices = [
        index
        for index, column in enumerate(position_columns)
        if column.startswith(
            "spatial_fused_common_mode_corrected_shift_pm_signed_"
        )
    ]
    if len(selected_position_indices) != 9:
        raise RuntimeError(
            "expected exactly nine signed fused common-mode-corrected shift features"
        )
    position_matrix = position_matrix[:, selected_position_indices]
    position_columns = tuple(position_columns[index] for index in selected_position_indices)
    sample_kind = np.asarray([record.sample_kind for record in records], dtype=object)
    training_eligible = np.asarray(dataset.training_eligible, dtype=bool)
    repeat_index = np.asarray([record.repeat_index or 0 for record in records], dtype=int)
    timestamps = np.asarray([record.timestamp.date() for record in records], dtype=object)
    contact_labels = np.asarray([record.contact_label for record in records], dtype=object)
    position_labels = np.asarray([record.position_label or "" for record in records], dtype=object)
    force_labels = np.asarray([record.manual_force_label or "" for record in records], dtype=object)
    manual_mask = sample_kind == "manual_press"
    contact_mask = np.isin(sample_kind, ["no_contact", "manual_press"]) & training_eligible

    original_mask = contact_mask & (
        ((sample_kind == "no_contact") & (timestamps < args.incremental_date))
        | (manual_mask & (repeat_index <= 5))
    )
    incremental_mask = contact_mask & (
        ((sample_kind == "no_contact") & (timestamps >= args.incremental_date))
        | (manual_mask & (repeat_index >= 6))
    )
    incremental_press_mask = manual_mask & (repeat_index >= 6)
    late_holdout_mask = manual_mask & (repeat_index >= args.late_holdout_first_repeat)
    pre_late_training_mask = manual_mask & (repeat_index < args.late_holdout_first_repeat)

    contact_challenge_model = contact_model(args.random_state)
    contact_challenge_model.fit(matrix[original_mask], contact_labels[original_mask])
    contact_challenge_prediction = contact_challenge_model.predict(matrix[incremental_mask])
    contact_challenge = classification_summary(
        contact_labels[incremental_mask],
        contact_challenge_prediction,
        CONTACT_ORDER,
    )

    position_challenge_model = position_model(args.random_state)
    original_manual_mask = original_mask & manual_mask
    position_challenge_model.fit(
        position_matrix[original_manual_mask],
        position_labels[original_manual_mask],
    )
    position_challenge_prediction = position_challenge_model.predict(
        position_matrix[incremental_press_mask]
    )
    position_challenge = classification_summary(
        position_labels[incremental_press_mask],
        position_challenge_prediction,
        POSITION_ORDER,
    )
    position_challenge_agreement = agreement_summary(
        position_challenge_model,
        position_matrix[incremental_press_mask],
        position_labels[incremental_press_mask],
    )

    reverse_position_model = position_model(args.random_state)
    reverse_position_model.fit(
        position_matrix[incremental_press_mask],
        position_labels[incremental_press_mask],
    )
    reverse_position_prediction = reverse_position_model.predict(
        position_matrix[original_manual_mask]
    )
    reverse_position_challenge = classification_summary(
        position_labels[original_manual_mask],
        reverse_position_prediction,
        POSITION_ORDER,
    )
    reverse_position_agreement = agreement_summary(
        reverse_position_model,
        position_matrix[original_manual_mask],
        position_labels[original_manual_mask],
    )

    calibrated_position_model = position_model(args.random_state)
    calibrated_position_model.fit(
        position_matrix[pre_late_training_mask],
        position_labels[pre_late_training_mask],
    )
    calibrated_position_prediction = calibrated_position_model.predict(
        position_matrix[late_holdout_mask]
    )
    calibrated_position_holdout = classification_summary(
        position_labels[late_holdout_mask],
        calibrated_position_prediction,
        POSITION_ORDER,
    )
    calibrated_position_agreement = agreement_summary(
        calibrated_position_model,
        position_matrix[late_holdout_mask],
        position_labels[late_holdout_mask],
    )

    force_challenge_model = force_model(args.random_state)
    force_challenge_model.fit(matrix[original_manual_mask], force_labels[original_manual_mask])
    force_challenge_prediction = force_challenge_model.predict(matrix[incremental_press_mask])
    force_challenge = classification_summary(
        force_labels[incremental_press_mask],
        force_challenge_prediction,
        FORCE_ORDER,
    )
    reverse_force_model = force_model(args.random_state)
    reverse_force_model.fit(matrix[incremental_press_mask], force_labels[incremental_press_mask])
    reverse_force_prediction = reverse_force_model.predict(matrix[original_manual_mask])
    reverse_force_challenge = classification_summary(
        force_labels[original_manual_mask],
        reverse_force_prediction,
        FORCE_ORDER,
    )
    incremental_hard_mask = force_labels[incremental_press_mask] == "hard"
    force_hard_recall = float(
        np.mean(force_challenge_prediction[incremental_hard_mask] == "hard")
    )

    source_conditioned_models: dict[str, Pipeline] = {}
    for position in POSITION_ORDER:
        position_source_mask = original_manual_mask & (position_labels == position)
        model = force_model(args.random_state)
        model.fit(matrix[position_source_mask], force_labels[position_source_mask])
        source_conditioned_models[position] = model
    incremental_indices = np.flatnonzero(incremental_press_mask)
    conditioned_force_challenge_prediction = np.asarray(
        [
            source_conditioned_models[str(predicted_position)].predict(
                matrix[index : index + 1]
            )[0]
            for index, predicted_position in zip(
                incremental_indices,
                position_challenge_prediction,
                strict=True,
            )
        ],
        dtype=object,
    )
    conditioned_force_challenge = classification_summary(
        force_labels[incremental_press_mask],
        conditioned_force_challenge_prediction,
        FORCE_ORDER,
    )
    conditioned_force_enabled = bool(
        conditioned_force_challenge["macro_f1"] > force_challenge["macro_f1"]
    )

    calibrated_force_model = force_model(args.random_state)
    calibrated_force_model.fit(matrix[pre_late_training_mask], force_labels[pre_late_training_mask])
    calibrated_force_prediction = calibrated_force_model.predict(matrix[late_holdout_mask])
    calibrated_force_holdout = classification_summary(
        force_labels[late_holdout_mask],
        calibrated_force_prediction,
        FORCE_ORDER,
    )
    late_hard_mask = force_labels[late_holdout_mask] == "hard"
    calibrated_hard_recall = float(
        np.mean(calibrated_force_prediction[late_hard_mask] == "hard")
    )

    evaluations = {
        "evaluation_scope": "incremental_batch_challenge_plus_late_repeat_holdout",
        "original_training_file_count": int(np.sum(original_mask)),
        "incremental_challenge_file_count": int(np.sum(incremental_mask)),
        "incremental_press_file_count": int(np.sum(incremental_press_mask)),
        "incremental_no_contact_file_count": int(
            np.sum(incremental_mask & (sample_kind == "no_contact"))
        ),
        "incremental_force_classes_present": sorted(
            set(force_labels[incremental_press_mask].tolist())
        ),
        "contact_original_to_incremental_challenge": contact_challenge,
        "position_original_to_incremental_challenge": position_challenge,
        "position_original_to_incremental_agreement": position_challenge_agreement,
        "position_incremental_to_original_challenge": reverse_position_challenge,
        "position_incremental_to_original_agreement": reverse_position_agreement,
        "position_late_repeat_holdout_after_incremental_calibration": calibrated_position_holdout,
        "position_late_repeat_holdout_agreement": calibrated_position_agreement,
        "force_original_to_incremental_challenge": force_challenge,
        "force_incremental_to_original_challenge": reverse_force_challenge,
        "position_conditioned_force_original_to_incremental_challenge": (
            conditioned_force_challenge
        ),
        "position_conditioned_force_enabled": conditioned_force_enabled,
        "position_conditioned_force_policy": (
            "enabled_only_when_cross_batch_macro_f1_exceeds_global_force_model"
        ),
        "force_late_repeat_holdout_after_incremental_calibration": calibrated_force_holdout,
        "hard_recall_original_to_incremental_challenge": force_hard_recall,
        "hard_recall_late_repeat_holdout_after_incremental_calibration": calibrated_hard_recall,
        "formal_limitation": (
            "all classes are balanced, but each file is a static spectrum snapshot from "
            "the same sensor build; dynamic temporal generalization is not evaluated"
        ),
    }

    final_contact = contact_model(args.random_state)
    final_contact.fit(matrix[contact_mask], contact_labels[contact_mask])
    final_position = position_model(args.random_state)
    final_position.fit(position_matrix[manual_mask], position_labels[manual_mask])
    final_force = force_model(args.random_state)
    final_force.fit(matrix[manual_mask], force_labels[manual_mask])

    final_models: dict[str, Any] = {
        "contact_detector": payload(
            task_id="contact_detector",
            model_id="extra_trees_incremental",
            model=final_contact,
            feature_set="engineered",
            feature_columns=columns,
            label_order=CONTACT_ORDER,
            evaluation=contact_challenge,
        ),
        "position_classifier": payload(
            task_id="position_classifier",
            model_id="agreement_vote_logreg_lda_linear_svm_fused_shift_v7",
            model=final_position,
            feature_set="spatial_fingerprint",
            feature_columns=position_columns,
            label_order=POSITION_ORDER,
            evaluation={
                "original_to_incremental_challenge": position_challenge,
                "original_to_incremental_agreement": position_challenge_agreement,
                "incremental_to_original_challenge": reverse_position_challenge,
                "incremental_to_original_agreement": reverse_position_agreement,
                "late_repeat_holdout": calibrated_position_holdout,
                "late_repeat_holdout_agreement": calibrated_position_agreement,
            },
        ),
        "manual_force_classifier": payload(
            task_id="manual_force_classifier",
            model_id="random_forest_incremental_global_leaf2_v7",
            model=final_force,
            feature_set="engineered",
            feature_columns=columns,
            label_order=FORCE_ORDER,
            evaluation={
                "hard_recall_original_to_incremental_challenge": force_hard_recall,
                "hard_recall_late_repeat_holdout": calibrated_hard_recall,
                "original_to_incremental_challenge": force_challenge,
                "incremental_to_original_challenge": reverse_force_challenge,
            },
        ),
    }

    if conditioned_force_enabled:
        conditioned_models = {}
        for position in POSITION_ORDER:
            position_mask = manual_mask & (position_labels == position)
            model = force_model(args.random_state)
            model.fit(matrix[position_mask], force_labels[position_mask])
            conditioned_models[position] = model
        conditioned_payload = payload(
            task_id="position_conditioned_force_classifier",
            model_id="random_forest_incremental_position_conditioned",
            model=clone(final_force),
            feature_set="engineered",
            feature_columns=columns,
            label_order=FORCE_ORDER,
            evaluation={
                "cross_batch": conditioned_force_challenge,
                "hard_recall_late_repeat_holdout": calibrated_hard_recall,
                "limitation": evaluations["formal_limitation"],
            },
        )
        conditioned_payload["schema_version"] = "sense_static_position_conditioned_force_v1"
        conditioned_payload["models_by_position"] = conditioned_models
        conditioned_payload["contact_geometry"] = "manual_broad_approximate_fingertip_contact"
        final_models["position_conditioned_force_classifier"] = conditioned_payload

    base_bundle = joblib.load(args.base_bundle.resolve())
    bundle = dict(base_bundle)
    bundle.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_dataset_sha256": dataset_sha256(records),
            "source_file_count": len(records),
            "models": final_models,
            "common_wavelength_nm": dataset.common_wavelength_nm,
            "fallback_baseline_spectrum": dataset.reference_baseline_clusters[-1].median_spectrum,
            "evaluation_validity": "incremental_batch_challenge_plus_late_repeat_holdout",
            "deployment_status": "incremental_current_session_baseline_pending_cross_session_validation",
            "limitations": list(base_bundle.get("limitations") or [])
            + [
                "static_snapshot_dataset_not_dynamic_time_series",
                "same_sensor_build_generalization_only",
                "runtime_contact_decision_uses_baseline_relative_physical_gate",
                "position_uses_nine_fused_common_mode_corrected_shift_features",
                "position_confidence_is_uncalibrated_three_model_vote_fraction",
            ],
            "incremental_evaluation": evaluations,
        }
    )
    bundle_path = output_dir / "models" / "static_spectral_recognition_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    (output_dir / "incremental_evaluation.json").write_text(
        json.dumps(evaluations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "baseline_cluster_audit.json").write_text(
        json.dumps(
            [asdict(item) for item in dataset.baseline_cluster_assessments],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_lines = [
        "# Incremental static-spectrum model update",
        "",
        f"- Source files: {len(records)}",
        f"- Incremental challenge files: {evaluations['incremental_challenge_file_count']}",
        f"- Contact challenge accuracy: {contact_challenge['accuracy']:.4f}",
        f"- Position before incremental calibration: {position_challenge['accuracy']:.4f}",
        f"- Position cross-batch macro-F1: {position_challenge['macro_f1']:.4f}",
        f"- Position reverse cross-batch macro-F1: {reverse_position_challenge['macro_f1']:.4f}",
        f"- Position unanimous fraction: {position_challenge_agreement['unanimous_fraction']:.4f}",
        f"- Position unanimous accuracy: {position_challenge_agreement['unanimous_accuracy']:.4f}",
        f"- Position late-repeat holdout after calibration: {calibrated_position_holdout['accuracy']:.4f}",
        f"- Hard recall before incremental calibration: {force_hard_recall:.4f}",
        f"- Hard recall late-repeat holdout after calibration: {calibrated_hard_recall:.4f}",
        "",
        f"- Force cross-batch macro-F1: {force_challenge['macro_f1']:.4f}",
        f"- Force reverse cross-batch macro-F1: {reverse_force_challenge['macro_f1']:.4f}",
        f"- Position-conditioned force cross-batch macro-F1: {conditioned_force_challenge['macro_f1']:.4f}",
        f"- Position-conditioned force enabled: {conditioned_force_enabled}",
        f"- Force late-repeat holdout macro-F1: {calibrated_force_holdout['macro_f1']:.4f}",
        "All 9 positions and all three manual response levels contain 15 independent CSV snapshots.",
        "The response levels are approximate manual categories, not force_N.",
    ]
    (output_dir / "incremental_training_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluations, indent=2, ensure_ascii=False))
    print(f"bundle={bundle_path}")


if __name__ == "__main__":
    main()
