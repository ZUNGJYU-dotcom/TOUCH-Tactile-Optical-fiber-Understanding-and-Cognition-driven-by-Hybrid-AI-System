"""Summarize grouped OOF results for Blind3 incremental training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]


def _force_metrics(force: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = mask & np.isfinite(force) & np.isfinite(predicted)
    if not selected.any():
        return {"frame_count": 0}
    truth = force[selected]
    estimate = predicted[selected]
    variable = len(truth) > 1 and float(np.std(truth)) > 1.0e-12
    return {
        "frame_count": int(selected.sum()),
        "mae_n": float(mean_absolute_error(truth, estimate)),
        "rmse_n": float(np.sqrt(mean_squared_error(truth, estimate))),
        "r2": float(r2_score(truth, estimate)) if variable else None,
        "slope": float(np.polyfit(truth, estimate, 1)[0]) if variable else None,
        "maximum_reference_n": float(np.max(truth)),
        "maximum_prediction_n": float(np.max(estimate)),
    }


def _section(frame: pd.DataFrame) -> dict[str, Any]:
    contact_true = pd.to_numeric(frame["contact_true"], errors="coerce")
    contact = frame[contact_true.notna()].copy()
    truth_contact = pd.to_numeric(contact["contact_true"]).astype(int).to_numpy()
    predicted_contact = pd.to_numeric(contact["contact_predicted"]).astype(int).to_numpy()
    contact_matrix = confusion_matrix(truth_contact, predicted_contact, labels=[0, 1])

    position = frame[frame["position_true"].astype(str) != ""].copy()
    truth_position = position["position_true"].astype(str).to_numpy()
    predicted_position = position["position_predicted"].astype(str).to_numpy()
    mistakes = Counter(
        (truth, predicted)
        for truth, predicted in zip(truth_position, predicted_position, strict=True)
        if truth != predicted
    )

    force = pd.to_numeric(frame["force_fz_n"], errors="coerce").to_numpy(dtype=float)
    force_prediction = pd.to_numeric(
        frame["force_fz_predicted_n"], errors="coerce"
    ).to_numpy(dtype=float)
    return {
        "frame_count": int(len(frame)),
        "contact": {
            "accuracy": float(accuracy_score(truth_contact, predicted_contact)),
            "macro_f1": float(f1_score(truth_contact, predicted_contact, average="macro")),
            "no_contact_false_positive_rate": float(
                contact_matrix[0, 1] / max(int(contact_matrix[0].sum()), 1)
            ),
            "contact_recall": float(
                contact_matrix[1, 1] / max(int(contact_matrix[1].sum()), 1)
            ),
            "confusion_matrix": contact_matrix.tolist(),
        },
        "position": {
            "frame_count": int(len(position)),
            "accuracy": float(accuracy_score(truth_position, predicted_position)),
            "macro_f1": float(
                f1_score(
                    truth_position,
                    predicted_position,
                    labels=POSITION_ORDER,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_by_position": {
                label: float(
                    np.mean(predicted_position[truth_position == label] == label)
                )
                for label in POSITION_ORDER
            },
            "top_confusions": [
                {"truth": truth, "predicted": predicted, "frame_count": count}
                for (truth, predicted), count in mistakes.most_common(12)
            ],
        },
        "force": {
            "all": _force_metrics(force, force_prediction, np.ones(len(frame), dtype=bool)),
            "active": _force_metrics(force, force_prediction, force >= 0.10),
            "above_5_n": _force_metrics(force, force_prediction, force > 5.0),
        },
    }


def _episode_majority_metrics(
    oof: pd.DataFrame,
    audited_frames_path: Path,
    incremental_ids: set[str],
) -> dict[str, Any]:
    audited = pd.read_csv(
        audited_frames_path,
        encoding="utf-8-sig",
        keep_default_na=False,
        usecols=["session_id", "capture_index", "expected_position"],
    )
    audited = audited[audited["session_id"].astype(str).isin(incremental_ids)]
    merged = audited.merge(
        oof[["session_id", "capture_index", "position_predicted"]],
        on=["session_id", "capture_index"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for session_id, session in merged.groupby("session_id", sort=False):
        session = session.sort_values("capture_index").reset_index(drop=True)
        run_start: int | None = None
        run_label = "none"
        for index, label in enumerate(session["expected_position"].astype(str)):
            if label != run_label:
                if run_start is not None and run_label != "none":
                    subset = session.iloc[run_start:index]
                    votes = [value for value in subset["position_predicted"] if value]
                    predicted = Counter(votes).most_common(1)[0][0] if votes else "none"
                    rows.append(
                        {
                            "session_id": str(session_id),
                            "expected_position": run_label,
                            "predicted_position": predicted,
                            "frame_count": int(len(subset)),
                        }
                    )
                run_start = index if label != "none" else None
                run_label = label
        if run_start is not None and run_label != "none":
            subset = session.iloc[run_start:]
            votes = [value for value in subset["position_predicted"] if value]
            predicted = Counter(votes).most_common(1)[0][0] if votes else "none"
            rows.append(
                {
                    "session_id": str(session_id),
                    "expected_position": run_label,
                    "predicted_position": predicted,
                    "frame_count": int(len(subset)),
                }
            )
    correct = sum(
        row["expected_position"] == row["predicted_position"] for row in rows
    )
    return {
        "episode_count": len(rows),
        "correct_episode_count": int(correct),
        "accuracy": float(correct / len(rows)) if rows else None,
        "errors": [
            row
            for row in rows
            if row["expected_position"] != row["predicted_position"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--previous-training-dir", type=Path, required=True)
    parser.add_argument("--pretraining-audit", type=Path, required=True)
    args = parser.parse_args()

    training_dir = args.training_dir.resolve()
    manifest = json.loads(
        (args.dataset_dir.resolve() / "ordinary_fbg_px6d_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    incremental_ids = {
        str(session["session_id"])
        for session in manifest["sessions"]
        if session.get("provenance_role")
        == "formerly_blind_now_unblinded_training_data"
    }
    idle_ids = {
        str(session["session_id"])
        for session in manifest["sessions"]
        if session.get("provenance_role")
        == "formerly_blind_now_unblinded_training_data"
        and session.get("position_label") == "no_contact"
    }
    oof = pd.read_csv(
        training_dir / "grouped_oof_predictions.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    previous_oof = pd.read_csv(
        args.previous_training_dir.resolve() / "grouped_oof_predictions.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    incremental_mask = oof["session_id"].astype(str).isin(incremental_ids)
    idle = oof[oof["session_id"].astype(str).isin(idle_ids)]
    pretraining = json.loads(args.pretraining_audit.resolve().read_text(encoding="utf-8"))
    training_metrics = json.loads(
        (training_dir / "training_metrics.json").read_text(encoding="utf-8")
    )

    result = {
        "schema_version": "blind3_incremental_grouped_oof_summary_v1",
        "dataset_id": manifest["dataset_id"],
        "evaluation_boundary": (
            "Blind3 filenames were not obfuscated. Results are complete-session "
            "grouped OOF diagnostics, not independent blind-test evidence."
        ),
        "selected_models": {
            task: row["model_id"]
            for task, row in training_metrics["selected_candidates"].items()
        },
        "previous_base_oof": _section(previous_oof),
        "new_model_base_oof": _section(oof[~incremental_mask]),
        "blind3_grouped_oof": _section(oof[incremental_mask]),
        "blind3_grouped_oof_episode_majority": _episode_majority_metrics(
            oof,
            args.pretraining_audit.resolve().parent / "frame_comparison.csv",
            incremental_ids,
        ),
        "combined_grouped_oof": _section(oof),
        "blind3_known_idle_session_false_positive_rate": float(
            (pd.to_numeric(idle["contact_predicted"], errors="coerce") == 1).mean()
        ),
        "pretraining_blind3_replay": pretraining["offline"],
    }
    output_json = training_dir / "blind3_incremental_validation_summary.json"
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    blind3 = result["blind3_grouped_oof"]
    base_old = result["previous_base_oof"]
    base_new = result["new_model_base_oof"]
    high = blind3["force"]["above_5_n"]
    episodes = result["blind3_grouped_oof_episode_majority"]
    report = [
        "# Blind3 Incremental Training Validation",
        "",
        "## Evaluation boundary",
        "",
        "- Blind3 filenames were not obfuscated, so Blind3 is labelled training data.",
        "- Metrics below are grouped out-of-fold by complete session, not an independent blind test.",
        "- The pre-training replay remains useful only as a frozen baseline comparison.",
        "",
        "## Blind3 grouped OOF",
        "",
        f"- Position accuracy: {blind3['position']['accuracy']:.4f}; macro-F1: {blind3['position']['macro_f1']:.4f}.",
        f"- Press-episode majority accuracy: {episodes['correct_episode_count']}/{episodes['episode_count']} ({episodes['accuracy']:.4f}).",
        f"- Contact accuracy: {blind3['contact']['accuracy']:.4f}; no-contact FPR: {blind3['contact']['no_contact_false_positive_rate']:.4f}.",
        f"- Known-idle-session false-positive rate: {result['blind3_known_idle_session_false_positive_rate']:.4f}.",
        f"- Active-force MAE: {blind3['force']['active']['mae_n']:.3f} N; R2: {blind3['force']['active']['r2']:.3f}.",
        "",
        "## Base-session regression check",
        "",
        f"- Position accuracy: {base_old['position']['accuracy']:.4f} -> {base_new['position']['accuracy']:.4f}.",
        f"- No-contact FPR: {base_old['contact']['no_contact_false_positive_rate']:.4f} -> {base_new['contact']['no_contact_false_positive_rate']:.4f}.",
        f"- Force MAE: {base_old['force']['all']['mae_n']:.3f} N -> {base_new['force']['all']['mae_n']:.3f} N.",
        "",
        "## High-force limitation",
        "",
        f"- Blind3 frames above 5 N: {high['frame_count']}.",
        f"- Above-5-N MAE: {high['mae_n']:.3f} N; maximum prediction: {high['maximum_prediction_n']:.3f} N for a {high['maximum_reference_n']:.3f} N reference.",
        "- Tree regression still saturates at the high-force edge and is not validated as unbounded.",
        "",
        "## Deployment decision",
        "",
        "- Candidate only. Do not replace the current Beta model without live no-contact, release, and nine-position checks.",
        "- A new filename-obfuscated, untouched acquisition set is still required for a formal blind claim.",
    ]
    (training_dir / "blind3_incremental_validation_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "summary": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
