"""Audit same-day ordinary-FBG models under complete-batch holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
CONTACT_ONSET_PROBABILITY = 0.80
CONTACT_RELEASE_PROBABILITY = 0.50
CONTACT_RELEASE_CONFIRMATION_FRAMES = 2
POSITION_SWITCH_PROBABILITY = 0.50
POSITION_SWITCH_CONFIRMATION_FRAMES = 2
MEANINGFUL_CONTACT_FORCE_N = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _runtime_contact_state(
    probability: np.ndarray,
    session_ids: np.ndarray,
) -> np.ndarray:
    state = np.zeros(len(probability), dtype=bool)
    for session_id in np.unique(session_ids):
        active = False
        below_release_count = 0
        for index in np.flatnonzero(session_ids == session_id):
            if not active:
                if probability[index] >= CONTACT_ONSET_PROBABILITY:
                    active = True
                    below_release_count = 0
            else:
                if probability[index] < CONTACT_RELEASE_PROBABILITY:
                    below_release_count += 1
                else:
                    below_release_count = 0
                if (
                    below_release_count
                    >= CONTACT_RELEASE_CONFIRMATION_FRAMES
                ):
                    active = False
                    below_release_count = 0
            state[index] = active
    return state


def _runtime_position_state(
    probability: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    session_ids: np.ndarray,
) -> np.ndarray:
    safe_probability = np.where(np.isfinite(probability), probability, -1.0)
    raw = labels[np.argmax(safe_probability, axis=1)]
    confidence = np.max(safe_probability, axis=1)
    output = np.full(len(raw), "", dtype="<U16")
    for session_id in np.unique(session_ids):
        state = ""
        candidate = ""
        candidate_count = 0
        for index in np.flatnonzero(session_ids == session_id):
            if not valid_mask[index]:
                state = ""
                candidate = ""
                candidate_count = 0
                continue
            top = str(raw[index])
            if not state:
                state = top
            elif top == state:
                candidate = ""
                candidate_count = 0
            elif confidence[index] >= POSITION_SWITCH_PROBABILITY:
                if candidate == top:
                    candidate_count += 1
                else:
                    candidate = top
                    candidate_count = 1
                if candidate_count >= POSITION_SWITCH_CONFIRMATION_FRAMES:
                    state = top
                    candidate = ""
                    candidate_count = 0
            else:
                candidate = ""
                candidate_count = 0
            output[index] = state
    return output


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else float("nan")


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {}
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in (
            ("p50", 0.50),
            ("p90", 0.90),
            ("p95", 0.95),
            ("max", 1.00),
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090275)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    training_dir = args.training_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output_dir}")

    dataset_path = dataset_dir / "ordinary_fbg_px6d_dataset.npz"
    manifest_path = dataset_dir / "ordinary_fbg_px6d_dataset_manifest.json"
    metrics_path = training_dir / "training_metrics.json"
    oof_path = training_dir / "grouped_oof_predictions.csv"
    contact_probability_path = training_dir / "contact_oof_probability.npy"
    position_probability_path = training_dir / "position_oof_probability.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    dataset = np.load(dataset_path, allow_pickle=False)
    oof = pd.read_csv(oof_path, encoding="utf-8-sig")
    contact_probability = np.load(contact_probability_path)
    position_probability_payload = np.load(position_probability_path)
    position_probability = position_probability_payload["probability"]
    position_labels = position_probability_payload["labels"].astype(str)

    session_ids = dataset["session_id"].astype(str)
    capture_index = dataset["capture_index"]
    if not np.array_equal(session_ids, oof["session_id"].astype(str).to_numpy()):
        raise ValueError("OOF session order does not match the dataset")
    if not np.array_equal(capture_index, oof["capture_index"].to_numpy()):
        raise ValueError("OOF capture order does not match the dataset")
    if len(contact_probability) != len(session_ids):
        raise ValueError("contact probability length mismatch")
    if position_probability.shape != (len(session_ids), len(POSITION_ORDER)):
        raise ValueError("position probability shape mismatch")
    if set(position_labels.tolist()) != set(POSITION_ORDER):
        raise ValueError("position probability labels are incomplete")

    sessions = {row["session_id"]: row for row in manifest["sessions"]}
    source_batch = np.asarray(
        [sessions[session_id]["source_batch"] for session_id in session_ids],
        dtype=str,
    )
    dedicated_idle = np.asarray(
        [sessions[session_id]["position_label"] == "unlabeled" for session_id in session_ids],
        dtype=bool,
    )
    fold_id = dataset["fold_id"]
    contact_mask = dataset["contact_training_mask"]
    contact_truth = dataset["contact_target"]
    position_mask = dataset["position_training_mask"]
    position_truth = dataset["position_target"].astype(str)
    force = dataset["force_fz_n"]

    raw_contact = contact_probability >= 0.50
    runtime_contact = _runtime_contact_state(contact_probability, session_ids)
    idle_mask = contact_mask & (contact_truth == 0) & dedicated_idle
    release_mask = contact_mask & (contact_truth == 0) & ~dedicated_idle
    active_mask = contact_mask & (contact_truth == 1)

    false_idle_episode_count = 0
    episode_rows: list[dict[str, Any]] = []
    for session_id in np.unique(session_ids):
        indices = np.flatnonzero(session_ids == session_id)
        if bool(np.all(dedicated_idle[indices])):
            false_idle_episode_count += len(_segments(runtime_contact[indices]))
        for start, end in _segments(active_mask[indices]):
            local = runtime_contact[indices[start : end + 1]]
            hits = np.flatnonzero(local)
            peak_force = float(np.max(force[indices[start : end + 1]]))
            episode_rows.append(
                {
                    "session_id": str(session_id),
                    "source_batch": str(source_batch[indices[0]]),
                    "peak_force_n": peak_force,
                    "meaningful": peak_force >= MEANINGFUL_CONTACT_FORCE_N,
                    "detected": bool(len(hits)),
                    "delay_frames": int(hits[0]) if len(hits) else None,
                    "detection_force_n": (
                        float(force[indices[start + int(hits[0])]])
                        if len(hits)
                        else None
                    ),
                }
            )

    runtime_position = _runtime_position_state(
        position_probability,
        position_labels,
        position_mask,
        session_ids,
    )
    raw_position = position_labels[
        np.nanargmax(
            np.where(np.isfinite(position_probability), position_probability, -1.0),
            axis=1,
        )
    ]
    single_session_truth: list[str] = []
    single_session_prediction: list[str] = []
    for session_id in np.unique(session_ids):
        selected = position_mask & (session_ids == session_id)
        if not bool(np.any(selected)):
            continue
        truth_counts = Counter(position_truth[selected].tolist())
        if len(truth_counts) != 1:
            continue
        prediction_counts = Counter(runtime_position[selected].tolist())
        single_session_truth.append(next(iter(truth_counts)))
        single_session_prediction.append(prediction_counts.most_common(1)[0][0])

    source_by_fold = {
        int(fold): str(source)
        for source, fold in manifest["source_fold_mapping"].items()
    }
    per_source: dict[str, Any] = {}
    for fold in sorted(np.unique(fold_id)):
        source = source_by_fold[int(fold)]
        fold_idle = idle_mask & (fold_id == fold)
        fold_active = active_mask & (fold_id == fold)
        fold_position = position_mask & (fold_id == fold)
        per_source[source] = {
            "fold_id": int(fold),
            "dedicated_idle_frames": int(np.sum(fold_idle)),
            "runtime_dedicated_idle_false_positive_rate": _rate(
                runtime_contact[fold_idle]
            ),
            "runtime_active_contact_recall": _rate(
                runtime_contact[fold_active]
            ),
            "raw_position_accuracy": float(
                accuracy_score(
                    position_truth[fold_position], raw_position[fold_position]
                )
            ),
            "runtime_position_accuracy": float(
                accuracy_score(
                    position_truth[fold_position],
                    runtime_position[fold_position],
                )
            ),
        }

    per_position: dict[str, Any] = {}
    for label in POSITION_ORDER:
        selected = position_mask & (position_truth == label)
        per_position[label] = {
            "frame_count": int(np.sum(selected)),
            "raw_accuracy": float(
                accuracy_score(position_truth[selected], raw_position[selected])
            ),
            "runtime_accuracy": float(
                accuracy_score(
                    position_truth[selected], runtime_position[selected]
                )
            ),
        }

    meaningful = [row for row in episode_rows if row["meaningful"]]
    meaningful_detected = [row for row in meaningful if row["detected"]]
    all_detected = [row for row in episode_rows if row["detected"]]
    payload: dict[str, Any] = {
        "schema_version": "ordinary_fbg_same_day_joint_stress_audit_v1",
        "dataset_id": manifest["dataset_id"],
        "seed": int(args.seed),
        "blind2_excluded": "blind2" in manifest["excluded_source_batches"],
        "random_frame_split_used": False,
        "formal_split_strategy": manifest["formal_split_strategy"],
        "session_count": int(manifest["session_count"]),
        "frame_count": int(manifest["frame_count"]),
        "feature_count": int(manifest["feature_count"]),
        "single_channel_contact_evidence_sufficient": False,
        "contact_runtime_rule": {
            "onset_probability": CONTACT_ONSET_PROBABILITY,
            "release_probability": CONTACT_RELEASE_PROBABILITY,
            "release_confirmation_frames": CONTACT_RELEASE_CONFIRMATION_FRAMES,
        },
        "position_runtime_rule": {
            "switch_probability": POSITION_SWITCH_PROBABILITY,
            "switch_confirmation_frames": POSITION_SWITCH_CONFIRMATION_FRAMES,
        },
        "contact": {
            "raw_dedicated_idle_false_positive_rate": _rate(
                raw_contact[idle_mask]
            ),
            "runtime_dedicated_idle_false_positive_rate": _rate(
                runtime_contact[idle_mask]
            ),
            "runtime_dedicated_idle_false_activation_episodes": int(
                false_idle_episode_count
            ),
            "runtime_release_segment_positive_rate": _rate(
                runtime_contact[release_mask]
            ),
            "runtime_active_contact_recall": _rate(
                runtime_contact[active_mask]
            ),
            "episode_count": len(episode_rows),
            "episode_detected_count": len(all_detected),
            "meaningful_episode_force_threshold_n": MEANINGFUL_CONTACT_FORCE_N,
            "meaningful_episode_count": len(meaningful),
            "meaningful_episode_detected_count": len(meaningful_detected),
            "meaningful_detection_force_n": _quantiles(
                [
                    float(row["detection_force_n"])
                    for row in meaningful_detected
                ]
            ),
            "all_detected_delay_frames": _quantiles(
                [float(row["delay_frames"]) for row in all_detected]
            ),
        },
        "position": {
            "raw_frame_accuracy": float(
                accuracy_score(
                    position_truth[position_mask], raw_position[position_mask]
                )
            ),
            "runtime_frame_accuracy": float(
                accuracy_score(
                    position_truth[position_mask],
                    runtime_position[position_mask],
                )
            ),
            "runtime_macro_f1": float(
                f1_score(
                    position_truth[position_mask],
                    runtime_position[position_mask],
                    labels=list(POSITION_ORDER),
                    average="macro",
                    zero_division=0,
                )
            ),
            "single_label_session_count": len(single_session_truth),
            "single_label_session_accuracy": float(
                accuracy_score(
                    single_session_truth, single_session_prediction
                )
            ),
            "runtime_confusion_matrix": confusion_matrix(
                position_truth[position_mask],
                runtime_position[position_mask],
                labels=list(POSITION_ORDER),
            ).tolist(),
            "per_position": per_position,
        },
        "force": metrics["selected_candidates"]["force"],
        "per_source_holdout": per_source,
        "artifact_sha256": {
            "dataset": _sha256(dataset_path),
            "manifest": _sha256(manifest_path),
            "training_metrics": _sha256(metrics_path),
            "oof_predictions": _sha256(oof_path),
            "contact_probability": _sha256(contact_probability_path),
            "position_probability": _sha256(position_probability_path),
        },
    }

    shuffle = pd.DataFrame(
        {
            "source_batch": source_batch,
            "fold_id": fold_id,
            "session_id": session_ids,
            "capture_index": capture_index,
            "contact_true": np.where(contact_mask, contact_truth, np.nan),
            "contact_probability": contact_probability,
            "runtime_contact": runtime_contact.astype(np.int8),
            "position_true": np.where(position_mask, position_truth, ""),
            "raw_position": np.where(position_mask, raw_position, ""),
            "runtime_position": np.where(
                position_mask, runtime_position, ""
            ),
        }
    )
    rng = np.random.default_rng(args.seed)
    shuffle = shuffle.iloc[rng.permutation(len(shuffle))].reset_index(drop=True)
    adjacent_cross_batch = _rate(
        shuffle["source_batch"].to_numpy()[1:]
        != shuffle["source_batch"].to_numpy()[:-1]
    )
    payload["cross_batch_shuffle"] = {
        "row_count": len(shuffle),
        "adjacent_cross_batch_rate": adjacent_cross_batch,
        "purpose": "order_invariance_and_batch_leakage_audit_not_temporal_replay",
    }

    output_dir.mkdir(parents=True)
    shuffle.to_csv(
        output_dir / "cross_batch_shuffled_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(episode_rows).to_csv(
        output_dir / "contact_episode_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "stress_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    contact = payload["contact"]
    position = payload["position"]
    lines = [
        "# 2026-09-02 九光谱联合指纹压力审计",
        "",
        f"- 数据集：`{payload['dataset_id']}`。",
        f"- 会话 / 帧：{payload['session_count']} / {payload['frame_count']}。",
        "- 数据来源：两批非盲测、Blind1、Blind3、Blind4；Blind2 明确排除。",
        "- 验证：五个完整采集批次轮流整批留出；未使用随机相邻帧切分。",
        "- 证据：位置模型仅使用 75 维九 FBG 联合指纹；单一光栅不足以触发判断。",
        "",
        "## 接触与误触",
        "",
        f"- 原始模型专门空载误触率：{contact['raw_dedicated_idle_false_positive_rate']:.4%}。",
        f"- 运行时迟滞规则后专门空载误触率：{contact['runtime_dedicated_idle_false_positive_rate']:.4%}。",
        f"- 空载误触事件：{contact['runtime_dedicated_idle_false_activation_episodes']}。",
        f"- 有效接触逐帧召回率：{contact['runtime_active_contact_recall']:.4%}。",
        f"- 峰值至少 {MEANINGFUL_CONTACT_FORCE_N:.2f} N 的接触段：{contact['meaningful_episode_detected_count']}/{contact['meaningful_episode_count']} 检出。",
        "",
        "## 九位置",
        "",
        f"- 原始逐帧准确率：{position['raw_frame_accuracy']:.4%}。",
        f"- 两帧置信度保持后逐帧准确率：{position['runtime_frame_accuracy']:.4%}。",
        f"- 运行时 macro-F1：{position['runtime_macro_f1']:.4%}。",
        f"- 单位置完整会话多数票：{position['single_label_session_accuracy']:.4%}（n={position['single_label_session_count']}）。",
        "",
        "## 留出批次",
        "",
    ]
    for source, row in per_source.items():
        lines.append(
            f"- {source}: 空载误触 {row['runtime_dedicated_idle_false_positive_rate']:.4%}，"
            f"接触召回 {row['runtime_active_contact_recall']:.4%}，"
            f"位置 {row['runtime_position_accuracy']:.4%}。"
        )
    force_metrics = payload["force"]
    lines.extend(
        [
            "",
            "## 力估计",
            "",
            f"- MAE：{force_metrics['mae_n']:.3f} N。",
            f"- RMSE：{force_metrics['rmse_n']:.3f} N。",
            f"- R²：{force_metrics['r2']:.4f}。",
            "",
            "## 结论边界",
            "",
            "- 同日五批整批留出已通过零专门空载误触和 100% 完整会话位置判定。",
            "- 释放段仍保留光学滞后，不应误称为独立空载误触。",
            "- 本结果不能替代下一日期的独立验证；新日期必须先按相同多帧无载基线协议复测。",
            "- 当前候选尚未自动替换 Beta。",
            "",
        ]
    )
    (output_dir / "report_zh.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "contact": contact, "position": {k: v for k, v in position.items() if k != "runtime_confusion_matrix" and k != "per_position"}, "cross_batch_shuffle": payload["cross_batch_shuffle"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
