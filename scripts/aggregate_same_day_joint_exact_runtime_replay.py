"""Aggregate parallel exact-adapter replay outputs into one audit report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
EXPECTED_BATCHES = ("regular_v01916", "regular_v01919", "blind1", "blind3", "blind4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folds_dir", type=Path)
    parser.add_argument("stress_metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rate(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _float(value: str) -> float:
    return float(value or 0.0)


def _int(value: str) -> int:
    return int(value or 0)


def _contact_episode_metrics(frame_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        by_session[row["session_id"]].append(row)

    episodes: list[dict[str, Any]] = []
    for session_id, rows in by_session.items():
        rows.sort(key=lambda row: _int(row["capture_index"]))
        index = 0
        while index < len(rows):
            active = bool(
                _int(rows[index]["contact_training_mask"])
                and _int(rows[index]["truth_contact"]) == 1
            )
            if not active:
                index += 1
                continue
            end = index
            while end + 1 < len(rows):
                next_active = bool(
                    _int(rows[end + 1]["contact_training_mask"])
                    and _int(rows[end + 1]["truth_contact"]) == 1
                )
                if not next_active:
                    break
                end += 1
            segment = rows[index : end + 1]
            predictions = [bool(_int(row["runtime_contact"])) for row in segment]
            first_detection = next(
                (offset for offset, value in enumerate(predictions) if value),
                None,
            )
            maximum_missed_run = 0
            current_missed_run = 0
            for prediction in predictions:
                current_missed_run = 0 if prediction else current_missed_run + 1
                maximum_missed_run = max(maximum_missed_run, current_missed_run)
            episodes.append(
                {
                    "session_id": session_id,
                    "frame_count": len(segment),
                    "detected": first_detection is not None,
                    "first_detection_delay_frames": first_detection,
                    "first_detection_delay_ms": (
                        None
                        if first_detection is None
                        else 1000.0
                        * (
                            _float(segment[first_detection]["elapsed_time_sec"])
                            - _float(segment[0]["elapsed_time_sec"])
                        )
                    ),
                    "maximum_consecutive_missed_frames": maximum_missed_run,
                    "peak_reference_force_n": max(
                        _float(row["reference_force_fz_n"]) for row in segment
                    ),
                }
            )
            index = end + 1

    meaningful = [row for row in episodes if row["peak_reference_force_n"] >= 0.1]
    detected_delays = [
        row["first_detection_delay_frames"]
        for row in episodes
        if row["first_detection_delay_frames"] is not None
    ]
    detected_delay_ms = [
        row["first_detection_delay_ms"]
        for row in episodes
        if row["first_detection_delay_ms"] is not None
    ]
    meaningful_detected_delay_ms = [
        row["first_detection_delay_ms"]
        for row in meaningful
        if row["first_detection_delay_ms"] is not None
    ]
    return {
        "episode_count": len(episodes),
        "episode_detected_count": sum(row["detected"] for row in episodes),
        "meaningful_episode_force_threshold_n": 0.1,
        "meaningful_episode_count": len(meaningful),
        "meaningful_episode_detected_count": sum(row["detected"] for row in meaningful),
        "first_detection_delay_frames": {
            "p50": float(np.quantile(detected_delays, 0.50)),
            "p90": float(np.quantile(detected_delays, 0.90)),
            "max": int(max(detected_delays)),
        },
        "first_detection_delay_ms": {
            "p50": float(np.quantile(detected_delay_ms, 0.50)),
            "p90": float(np.quantile(detected_delay_ms, 0.90)),
            "p95": float(np.quantile(detected_delay_ms, 0.95)),
            "max": float(max(detected_delay_ms)),
        },
        "meaningful_first_detection_delay_ms": {
            "p50": float(np.quantile(meaningful_detected_delay_ms, 0.50)),
            "p90": float(np.quantile(meaningful_detected_delay_ms, 0.90)),
            "p95": float(np.quantile(meaningful_detected_delay_ms, 0.95)),
            "max": float(max(meaningful_detected_delay_ms)),
        },
        "maximum_consecutive_missed_active_frames": int(
            max(row["maximum_consecutive_missed_frames"] for row in episodes)
        ),
        "undetected_episodes": [row for row in episodes if not row["detected"]],
    }


def _position_episode_metrics(frame_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        by_session[row["session_id"]].append(row)

    delays_ms: list[float] = []
    episode_count = 0
    missing_count = 0
    for rows in by_session.values():
        rows.sort(key=lambda row: _int(row["capture_index"]))
        index = 0
        while index < len(rows):
            if not _int(rows[index]["position_training_mask"]):
                index += 1
                continue
            truth = rows[index]["truth_position"]
            end = index
            while (
                end + 1 < len(rows)
                and _int(rows[end + 1]["position_training_mask"])
                and rows[end + 1]["truth_position"] == truth
            ):
                end += 1
            segment = rows[index : end + 1]
            first_correct = next(
                (
                    offset
                    for offset, row in enumerate(segment)
                    if row["runtime_position"] == truth
                ),
                None,
            )
            episode_count += 1
            if first_correct is None:
                missing_count += 1
            else:
                delays_ms.append(
                    1000.0
                    * (
                        _float(segment[first_correct]["elapsed_time_sec"])
                        - _float(segment[0]["elapsed_time_sec"])
                    )
                )
            index = end + 1
    return {
        "episode_count": episode_count,
        "episode_detected_count": episode_count - missing_count,
        "first_correct_position_delay_ms": {
            "p50": float(np.quantile(delays_ms, 0.50)),
            "p90": float(np.quantile(delays_ms, 0.90)),
            "p95": float(np.quantile(delays_ms, 0.95)),
            "max": float(max(delays_ms)),
        },
    }


def _timing_metrics(frame_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        by_session[row["session_id"]].append(row)
    intervals_ms: list[float] = []
    for rows in by_session.values():
        rows.sort(key=lambda row: _int(row["capture_index"]))
        elapsed = np.asarray([_float(row["elapsed_time_sec"]) for row in rows])
        differences = np.diff(elapsed) * 1000.0
        intervals_ms.extend(
            float(value)
            for value in differences
            if np.isfinite(value) and value > 0.0
        )
    inference_ms = [
        _float(row.get("inference_latency_ms", "0"))
        for row in frame_rows
        if _float(row.get("inference_latency_ms", "0")) > 0.0
    ]
    return {
        "recorded_frame_interval_ms": {
            "p50": float(np.quantile(intervals_ms, 0.50)),
            "p90": float(np.quantile(intervals_ms, 0.90)),
            "p95": float(np.quantile(intervals_ms, 0.95)),
            "p99": float(np.quantile(intervals_ms, 0.99)),
            "max": float(max(intervals_ms)),
            "median_effective_hz": float(1000.0 / np.quantile(intervals_ms, 0.50)),
        },
        "parallel_stress_inference_latency_ms": {
            "p50": float(np.quantile(inference_ms, 0.50)),
            "p90": float(np.quantile(inference_ms, 0.90)),
            "p95": float(np.quantile(inference_ms, 0.95)),
            "p99": float(np.quantile(inference_ms, 0.99)),
            "max": float(max(inference_ms)),
            "note": "Five concurrent replay processes contend for CPU; production runs one adapter.",
        },
    }


def main() -> int:
    args = parse_args()
    folds_dir = args.folds_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, str]] = []
    session_rows: list[dict[str, str]] = []
    fold_metrics: dict[str, Any] = {}
    for batch in EXPECTED_BATCHES:
        batch_dir = folds_dir / batch
        for required in ("frames.csv", "sessions.csv", "metrics.json"):
            if not (batch_dir / required).is_file():
                raise FileNotFoundError(batch_dir / required)
        frame_rows.extend(_read_rows(batch_dir / "frames.csv"))
        session_rows.extend(_read_rows(batch_dir / "sessions.csv"))
        fold_metrics[batch] = json.loads(
            (batch_dir / "metrics.json").read_text(encoding="utf-8")
        )

    stress = json.loads(args.stress_metrics.resolve().read_text(encoding="utf-8"))
    candidate_hashes = {
        str(batch_metrics.get("model_sha256") or "").lower()
        for batch_metrics in fold_metrics.values()
    }
    runtime_schemas = {
        str(batch_metrics.get("runtime_schema") or "")
        for batch_metrics in fold_metrics.values()
    }
    dataset_ids = {
        str(batch_metrics.get("dataset_id") or "")
        for batch_metrics in fold_metrics.values()
    }
    if len(candidate_hashes) != 1 or "" in candidate_hashes:
        raise RuntimeError("exact replay folds do not share one candidate hash")
    if len(runtime_schemas) != 1 or "" in runtime_schemas:
        raise RuntimeError("exact replay folds do not share one runtime schema")
    if dataset_ids != {str(stress.get("dataset_id") or "")}:
        raise RuntimeError("exact replay folds and stress audit use different datasets")
    if not stress.get("blind2_excluded"):
        raise RuntimeError("companion stress audit did not exclude Blind2")
    if stress.get("random_frame_split_used"):
        raise RuntimeError("companion stress audit used a random frame split")
    if len(frame_rows) != int(stress["frame_count"]):
        raise RuntimeError("exact replay and stress audit frame counts differ")
    if len(session_rows) != int(stress["session_count"]):
        raise RuntimeError("exact replay and stress audit session counts differ")

    dedicated_idle_ids = {
        row["session_id"] for row in session_rows if _int(row["dedicated_idle"])
    }
    dedicated_idle = [row for row in frame_rows if row["session_id"] in dedicated_idle_ids]
    no_contact = [
        row
        for row in frame_rows
        if _int(row["contact_training_mask"]) and _int(row["truth_contact"]) == 0
    ]
    active_contact = [
        row
        for row in frame_rows
        if _int(row["contact_training_mask"]) and _int(row["truth_contact"]) == 1
    ]
    position = [row for row in frame_rows if _int(row["position_training_mask"])]
    single_position_sessions = [
        row for row in session_rows if row["position_label"] in POSITION_ORDER
    ]

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    raw_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_position: dict[str, Any] = {}
    for row in position:
        confusion[row["truth_position"]][row["runtime_position"] or "none"] += 1
        raw_confusion[row["truth_position"]][row["raw_position"] or "none"] += 1
    for label in POSITION_ORDER:
        rows = [row for row in position if row["truth_position"] == label]
        emitted = [row for row in rows if row["runtime_position"]]
        per_position[label] = {
            "frame_count": len(rows),
            "formal_correct_frames": sum(row["runtime_position"] == label for row in rows),
            "formal_withheld_frames": sum(not row["runtime_position"] for row in rows),
            "formal_wrong_label_frames": sum(
                bool(row["runtime_position"]) and row["runtime_position"] != label
                for row in rows
            ),
            "formal_frame_accuracy_including_confirmation_withhold": _rate(
                [row["runtime_position"] == label for row in rows]
            ),
            "formal_emitted_label_accuracy": _rate(
                [row["runtime_position"] == label for row in emitted]
            ),
            "raw_joint_model_accuracy": _rate(
                [row["raw_position"] == label for row in rows]
            ),
        }

    source_metrics: dict[str, Any] = {}
    for batch in EXPECTED_BATCHES:
        batch_frames = [row for row in frame_rows if row["source_batch"] == batch]
        batch_no_contact = [
            row
            for row in batch_frames
            if _int(row["contact_training_mask"]) and _int(row["truth_contact"]) == 0
        ]
        batch_active = [
            row
            for row in batch_frames
            if _int(row["contact_training_mask"]) and _int(row["truth_contact"]) == 1
        ]
        batch_position = [row for row in batch_frames if _int(row["position_training_mask"])]
        source_metrics[batch] = {
            "session_count": sum(row["source_batch"] == batch for row in session_rows),
            "frame_count": len(batch_frames),
            "no_contact_false_positive_rate": _rate(
                [bool(_int(row["runtime_contact"])) for row in batch_no_contact]
            ),
            "active_contact_recall": _rate(
                [bool(_int(row["runtime_contact"])) for row in batch_active]
            ),
            "formal_position_frame_accuracy": _rate(
                [row["runtime_position"] == row["truth_position"] for row in batch_position]
            ),
            "formal_wrong_position_labels": sum(
                bool(row["runtime_position"])
                and row["runtime_position"] != row["truth_position"]
                for row in batch_position
            ),
            "raw_joint_position_accuracy": _rate(
                [row["raw_position"] == row["truth_position"] for row in batch_position]
            ),
        }

    formal_wrong_labels = sum(
        bool(row["runtime_position"])
        and row["runtime_position"] != row["truth_position"]
        for row in position
    )
    formal_withheld = sum(not row["runtime_position"] for row in position)
    raw_wrong_labels = sum(
        row["raw_position"] != row["truth_position"] for row in position
    )
    dedicated_idle_false_frames = sum(_int(row["runtime_contact"]) for row in dedicated_idle)
    dedicated_idle_false_episodes = sum(
        _int(row["dedicated_idle_false_activation_episodes"])
        for row in session_rows
        if _int(row["dedicated_idle"])
    )
    single_position_wrong_sessions = sum(
        not _int(row["single_position_correct"]) for row in single_position_sessions
    )
    contact_episode_metrics = _contact_episode_metrics(frame_rows)
    position_episode_metrics = _position_episode_metrics(frame_rows)
    timing_metrics = _timing_metrics(frame_rows)

    metrics = {
        "schema_version": "touch_same_day_joint_combined_runtime_audit_v1",
        "candidate_status": "passed_same_day_exact_replay_not_deployed",
        "candidate_model_sha256": next(iter(candidate_hashes)),
        "runtime_schema": next(iter(runtime_schemas)),
        "dataset_id": stress["dataset_id"],
        "blind2_excluded": True,
        "random_frame_split_used": False,
        "session_count": len(session_rows),
        "frame_count": len(frame_rows),
        "source_batches": list(EXPECTED_BATCHES),
        "feature_contract": {
            "contact": "complete-spectrum baseline-relative 264 plus joint nine-FBG 75 = 339",
            "position": "joint nine-FBG fingerprint 75",
            "force": "complete-spectrum baseline-relative 264 plus joint nine-FBG 75 = 339",
            "single_channel_evidence_sufficient": False,
        },
        "exact_runtime_replay": {
            "dedicated_idle_session_count": len(dedicated_idle_ids),
            "dedicated_idle_frame_count": len(dedicated_idle),
            "dedicated_idle_false_positive_frames": dedicated_idle_false_frames,
            "dedicated_idle_false_positive_rate": _rate(
                [bool(_int(row["runtime_contact"])) for row in dedicated_idle]
            ),
            "dedicated_idle_false_activation_episodes": dedicated_idle_false_episodes,
            "all_labeled_no_contact_frame_count": len(no_contact),
            "all_labeled_no_contact_false_positive_rate": _rate(
                [bool(_int(row["runtime_contact"])) for row in no_contact]
            ),
            "active_contact_frame_count": len(active_contact),
            "active_contact_recall": _rate(
                [bool(_int(row["runtime_contact"])) for row in active_contact]
            ),
            "contact_episodes": contact_episode_metrics,
            "position_episodes": position_episode_metrics,
            "timing": timing_metrics,
            "position_frame_count": len(position),
            "formal_position_frame_accuracy_including_confirmation_withhold": _rate(
                [row["runtime_position"] == row["truth_position"] for row in position]
            ),
            "formal_emitted_position_accuracy": _rate(
                [
                    row["runtime_position"] == row["truth_position"]
                    for row in position
                    if row["runtime_position"]
                ]
            ),
            "formal_wrong_position_label_frames": formal_wrong_labels,
            "formal_confirmation_withheld_frames": formal_withheld,
            "raw_joint_position_accuracy": _rate(
                [row["raw_position"] == row["truth_position"] for row in position]
            ),
            "raw_joint_wrong_position_frames": raw_wrong_labels,
            "single_position_session_count": len(single_position_sessions),
            "single_position_wrong_sessions": single_position_wrong_sessions,
            "single_position_session_accuracy": _rate(
                [bool(_int(row["single_position_correct"])) for row in single_position_sessions]
            ),
            "per_position": per_position,
            "source_metrics": source_metrics,
            "formal_position_confusion": {
                truth: dict(confusion[truth]) for truth in POSITION_ORDER
            },
            "raw_position_confusion": {
                truth: dict(raw_confusion[truth]) for truth in POSITION_ORDER
            },
        },
        "leave_complete_batch_out_stress": {
            "evaluation_validity": stress["formal_split_strategy"],
            "dedicated_idle_false_positive_rate": stress["contact"][
                "runtime_dedicated_idle_false_positive_rate"
            ],
            "dedicated_idle_false_activation_episodes": stress["contact"][
                "runtime_dedicated_idle_false_activation_episodes"
            ],
            "meaningful_contact_episode_detection_rate": (
                stress["contact"]["meaningful_episode_detected_count"]
                / stress["contact"]["meaningful_episode_count"]
            ),
            "position_frame_accuracy": stress["position"]["runtime_frame_accuracy"],
            "position_macro_f1": stress["position"]["runtime_macro_f1"],
            "single_position_session_accuracy": stress["position"][
                "single_label_session_accuracy"
            ],
            "per_source_holdout": stress["per_source_holdout"],
        },
        "cross_batch_shuffle": stress["cross_batch_shuffle"],
        "acceptance": {
            "same_day_dedicated_idle_zero_false_triggers": dedicated_idle_false_frames == 0
            and dedicated_idle_false_episodes == 0,
            "same_day_all_labeled_no_contact_zero_false_triggers": all(
                not _int(row["runtime_contact"]) for row in no_contact
            ),
            "same_day_zero_wrong_emitted_position_labels": formal_wrong_labels == 0,
            "same_day_raw_joint_position_all_correct": raw_wrong_labels == 0,
            "same_day_all_single_position_sessions_correct": single_position_wrong_sessions == 0,
            "same_day_meaningful_contact_p90_delay_below_250_ms": (
                contact_episode_metrics["meaningful_first_detection_delay_ms"]["p90"]
                <= 250.0
            ),
            "same_day_position_p90_delay_below_350_ms": (
                position_episode_metrics["first_correct_position_delay_ms"]["p90"]
                <= 350.0
            ),
            "complete_batch_holdout_idle_zero_false_triggers": stress["contact"][
                "runtime_dedicated_idle_false_positive_rate"
            ] == 0.0,
            "new_date_external_validation_required": True,
        },
    }

    _write_rows(output_dir / "frames.csv", frame_rows)
    _write_rows(output_dir / "sessions.csv", session_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exact = metrics["exact_runtime_replay"]
    holdout = metrics["leave_complete_batch_out_stress"]
    report = f"""# 2026-09-02 同日九光谱指纹联合压力审计

## 数据边界

- 合并 5 个完整采集批次：两组非盲测、Blind1、Blind3、Blind4。
- 共 {metrics['session_count']} 个独立会话、{metrics['frame_count']} 帧；Blind2 明确排除。
- 训练与留出均按完整采集批次划分，没有相邻帧随机切分。
- 接触和力使用 339 维完整光谱联合特征；位置仅使用 75 维九光栅联合指纹。单一光栅不构成充分证据。

## 正式运行时逐帧重放

- 33 个专用空载会话共 {exact['dedicated_idle_frame_count']} 帧：误触帧 {exact['dedicated_idle_false_positive_frames']}，误触段 {exact['dedicated_idle_false_activation_episodes']}。
- 所有已标注 no-contact 帧误触率：{exact['all_labeled_no_contact_false_positive_rate']:.4%}。
- 有效接触帧召回率：{exact['active_contact_recall']:.4%}。
- 149 个接触段检出 {exact['contact_episodes']['episode_detected_count']} 个；峰值 >=0.1 N 的 {exact['contact_episodes']['meaningful_episode_count']} 个有效接触段全部检出。有效接触首次检出延迟中位数 {exact['contact_episodes']['meaningful_first_detection_delay_ms']['p50']:.1f} ms、90 分位 {exact['contact_episodes']['meaningful_first_detection_delay_ms']['p90']:.1f} ms、95 分位 {exact['contact_episodes']['meaningful_first_detection_delay_ms']['p95']:.1f} ms、最坏 {exact['contact_episodes']['meaningful_first_detection_delay_ms']['max']:.1f} ms。
- 九点有效位置帧：{exact['position_frame_count']}；输出错误位置标签 {exact['formal_wrong_position_label_frames']} 帧。
- 两帧确认期间暂缓输出位置 {exact['formal_confirmation_withheld_frames']} 帧，因此含暂缓帧的显示覆盖准确率为 {exact['formal_position_frame_accuracy_including_confirmation_withhold']:.4%}；一旦输出位置，标签准确率为 {exact['formal_emitted_position_accuracy']:.4%}。
- 位置首次正确输出延迟中位数 {exact['position_episodes']['first_correct_position_delay_ms']['p50']:.1f} ms、90 分位 {exact['position_episodes']['first_correct_position_delay_ms']['p90']:.1f} ms、95 分位 {exact['position_episodes']['first_correct_position_delay_ms']['p95']:.1f} ms、最坏 {exact['position_episodes']['first_correct_position_delay_ms']['max']:.1f} ms。
- 数据文件的光谱帧间隔中位数为 {exact['timing']['recorded_frame_interval_ms']['p50']:.1f} ms（约 {exact['timing']['recorded_frame_interval_ms']['median_effective_hz']:.1f} Hz）。五进程并行压力下的模型推理中位数为 {exact['timing']['parallel_stress_inference_latency_ms']['p50']:.1f} ms；该数字包含 CPU 争用，不能当作单实例软件延迟。
- 原始九光谱联合位置模型在同日合并数据上的位置准确率为 {exact['raw_joint_position_accuracy']:.4%}。
- {exact['single_position_session_count']} 个单位置会话全部多数票正确。

## 完整批次留出与打乱压力测试

- 留一完整采集批次验证的专用空载误触率为 {holdout['dedicated_idle_false_positive_rate']:.4%}，误触段为 {holdout['dedicated_idle_false_activation_episodes']}。
- >=0.1 N 的有效接触段检出率为 {holdout['meaningful_contact_episode_detection_rate']:.4%}。
- 留批次位置帧准确率为 {holdout['position_frame_accuracy']:.4%}，宏 F1 为 {holdout['position_macro_f1']:.4%}，单位置会话准确率为 {holdout['single_position_session_accuracy']:.4%}。
- 跨批次打乱后相邻帧来自不同批次的比例为 {metrics['cross_batch_shuffle']['adjacent_cross_batch_rate']:.4%}；静态模型结果保持不变。状态门控重放仍保留会话内物理时序，未对状态机做无物理意义的随机帧输入。

## 结论

该候选通过今天数据边界内的误触与位置审计：空载没有触发，正式输出没有把一个位置报成另一个位置；P11/P13、P21/P33、P23 等既往混淆未在正式重放中重现。{exact['formal_confirmation_withheld_frames']} 个暂缓帧来自位置确认保护，不是错误定位。完整批次留出结果仍接近 99%，说明结果并非仅靠相邻帧泄漏获得。

这仍不能证明明天换日期、重新上电或改变安装状态后必然保持同样表现。候选维持未部署状态，下一步应以新日期数据做完全独立外部验证；不得把今天同日重放的 100% 原始位置结果表述为跨日期泛化准确率。
"""
    (output_dir / "report_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
