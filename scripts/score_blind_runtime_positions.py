"""Score frozen TOUCH blind predictions after the answer set is revealed."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


POSITION_ORDER = (
    "none",
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
)
ANSWER_PATTERN = re.compile(
    r"^(?P<stem>.+)-(?P<label>uncontact|no[_-]?contact|p(?:11|12|13|21|22|23|31|32|33))$",
    re.IGNORECASE,
)
TIMESTAMP_PATTERN = re.compile(r"^(?P<timestamp>\d{8}_\d{6})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-ordered-fallback",
        action="store_true",
        help=(
            "Use answer-directory order only when strict session/timestamp "
            "matching fails, and record every fallback in the score artifact."
        ),
    )
    parser.add_argument(
        "--allow-post-unblind-deployment-validation",
        action="store_true",
        help=(
            "Score an answer-known replay only when its manifest explicitly "
            "marks post-unblind deployment validation."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_key(name: str) -> str:
    match = TIMESTAMP_PATTERN.match(name)
    if not match:
        raise ValueError(f"session name has no timestamp prefix: {name}")
    return match.group("timestamp")


def _answer_entries(answer_root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in sorted(answer_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        match = ANSWER_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"unrecognized answer folder name: {path.name}")
        raw_label = match.group("label").lower()
        entries.append(
            {
                "folder_name": path.name,
                "answer_stem": match.group("stem"),
                "timestamp_key": _timestamp_key(path.name),
                "expected_position": (
                    "none"
                    if raw_label in {"uncontact", "nocontact", "no_contact", "no-contact"}
                    else raw_label.upper()
                ),
            }
        )
    if not entries:
        raise RuntimeError(f"no answer folders found under {answer_root}")
    return entries


def _match_answer(
    session_id: str,
    answers: list[dict[str, str]],
) -> tuple[dict[str, str], str]:
    exact = [item for item in answers if item["answer_stem"] == session_id]
    if len(exact) == 1:
        return exact[0], "exact_session_id"
    timestamp_key = _timestamp_key(session_id)
    timestamp_matches = [
        item for item in answers if item["timestamp_key"] == timestamp_key
    ]
    if len(timestamp_matches) == 1:
        return timestamp_matches[0], "timestamp_prefix_fallback"
    raise RuntimeError(
        f"could not uniquely match {session_id}: "
        f"exact={len(exact)}, timestamp={len(timestamp_matches)}"
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _per_class_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label in POSITION_ORDER:
        support = sum(row["expected_position"] == label for row in rows)
        predicted = sum(row["predicted_position"] == label for row in rows)
        true_positive = sum(
            row["expected_position"] == label
            and row["predicted_position"] == label
            for row in rows
        )
        precision = _ratio(true_positive, predicted)
        recall = _ratio(true_positive, support)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0.0
            else None
        )
        result[label] = {
            "support": support,
            "predicted": predicted,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def main() -> int:
    args = parse_args()
    prediction_dir = args.prediction_dir.resolve()
    answer_root = args.answer_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite blind score: {output_dir}")

    predictions_path = prediction_dir / "blind_predictions.csv"
    manifest_path = prediction_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation_phase = str(manifest.get("evaluation_phase") or "")
    post_unblind = (
        manifest.get("answer_accessed") is True
        and evaluation_phase == "post_unblind_deployment_validation"
    )
    if manifest.get("answer_accessed") is False:
        post_unblind = False
    elif not (post_unblind and args.allow_post_unblind_deployment_validation):
        raise RuntimeError(
            "prediction manifest is not pre-unblind; pass the explicit "
            "post-unblind validation flag only for a correctly marked replay"
        )
    expected_hash = str(
        ((manifest.get("outputs") or {}).get("predictions_sha256") or "")
    )
    actual_hash = _sha256(predictions_path)
    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError("frozen prediction hash does not match its manifest")

    with predictions_path.open("r", newline="", encoding="utf-8-sig") as stream:
        predictions = list(csv.DictReader(stream))
    answers = _answer_entries(answer_root)
    if len(predictions) != len(answers):
        raise RuntimeError(
            f"prediction/answer count mismatch: {len(predictions)} vs {len(answers)}"
        )

    scored_rows: list[dict[str, Any]] = []
    used_answer_folders: set[str] = set()
    for prediction_index, prediction in enumerate(predictions):
        session_id = str(prediction["session_id"])
        try:
            answer, match_method = _match_answer(session_id, answers)
        except RuntimeError:
            if not args.allow_ordered_fallback:
                raise
            answer = answers[prediction_index]
            match_method = "explicit_ordered_fallback"
        answer_folder = answer["folder_name"]
        if answer_folder in used_answer_folders:
            raise RuntimeError(f"answer folder matched more than once: {answer_folder}")
        used_answer_folders.add(answer_folder)
        predicted_position = str(prediction["predicted_position"])
        expected_position = answer["expected_position"]
        scored_rows.append(
            {
                "session_order": int(prediction["session_order"]),
                "session_id": session_id,
                "answer_folder": answer_folder,
                "answer_match_method": match_method,
                "expected_position": expected_position,
                "predicted_position": predicted_position,
                "correct": predicted_position == expected_position,
                "active_reference_frames": int(
                    prediction["active_reference_frames"]
                ),
                "visual_active_frames": int(prediction["visual_active_frames"]),
                "winner_share": float(prediction["winner_share"]),
                "winner_margin_share": float(prediction["winner_margin_share"]),
            }
        )

    total = len(scored_rows)
    correct = sum(row["correct"] for row in scored_rows)
    active_rows = [row for row in scored_rows if row["expected_position"] != "none"]
    idle_rows = [row for row in scored_rows if row["expected_position"] == "none"]
    true_positive = sum(
        row["expected_position"] != "none"
        and row["predicted_position"] != "none"
        for row in scored_rows
    )
    true_negative = sum(
        row["expected_position"] == "none"
        and row["predicted_position"] == "none"
        for row in scored_rows
    )
    false_positive = sum(
        row["expected_position"] == "none"
        and row["predicted_position"] != "none"
        for row in scored_rows
    )
    false_negative = sum(
        row["expected_position"] != "none"
        and row["predicted_position"] == "none"
        for row in scored_rows
    )
    confusion = {
        expected: {
            predicted: sum(
                row["expected_position"] == expected
                and row["predicted_position"] == predicted
                for row in scored_rows
            )
            for predicted in POSITION_ORDER
        }
        for expected in POSITION_ORDER
    }
    per_class = _per_class_metrics(scored_rows)
    supported_f1 = [
        metrics["f1"]
        for metrics in per_class.values()
        if metrics["support"] and metrics["f1"] is not None
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    scored_path = output_dir / "scored_predictions.csv"
    with scored_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)

    summary = {
        "schema_version": "touch_position_runtime_evaluation_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_phase": (
            "post_unblind_deployment_validation"
            if post_unblind
            else "pre_unblind_scoring"
        ),
        "prediction_frozen_before_answer_access": not post_unblind,
        "independent_blind_evidence": not post_unblind,
        "deployment_validation_only": post_unblind,
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": _sha256(manifest_path),
        "frozen_predictions_sha256": actual_hash,
        "answer_root": str(answer_root),
        "answer_folder_names": [item["folder_name"] for item in answers],
        "ordered_answer_fallback_allowed": bool(args.allow_ordered_fallback),
        "ordered_answer_fallback_count": sum(
            row["answer_match_method"] == "explicit_ordered_fallback"
            for row in scored_rows
        ),
        "session_count": total,
        "correct_sessions": correct,
        "session_accuracy": _ratio(correct, total),
        "active_position_sessions": len(active_rows),
        "active_position_correct": sum(row["correct"] for row in active_rows),
        "active_position_accuracy": _ratio(
            sum(row["correct"] for row in active_rows), len(active_rows)
        ),
        "idle_sessions": len(idle_rows),
        "idle_correct": sum(row["correct"] for row in idle_rows),
        "idle_accuracy": _ratio(
            sum(row["correct"] for row in idle_rows), len(idle_rows)
        ),
        "contact_detection": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "sensitivity": _ratio(true_positive, true_positive + false_negative),
            "specificity": _ratio(true_negative, true_negative + false_positive),
        },
        "supported_class_macro_f1": (
            float(sum(supported_f1) / len(supported_f1)) if supported_f1 else None
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "matching_notes": [
            {
                "session_id": row["session_id"],
                "answer_folder": row["answer_folder"],
                "method": row["answer_match_method"],
            }
            for row in scored_rows
            if row["answer_match_method"] != "exact_session_id"
        ],
        "outputs": {
            "scored_predictions_csv": scored_path.name,
            "scored_predictions_sha256": _sha256(scored_path),
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"accuracy={correct}/{total} ({summary['session_accuracy']:.1%}); "
        f"active={summary['active_position_correct']}/{len(active_rows)}; "
        f"idle={summary['idle_correct']}/{len(idle_rows)}"
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
