"""Replay synchronized PX6D sessions through the deployed TOUCH runtime.

The audit is read-only for source captures and model artifacts. It measures
idle false activation and active-contact spatial accuracy after the exact live
runtime gate, including its multi-frame position confirmation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.all_source_runtime_adapter import (  # noqa: E402
    AllSourceOpticalForceAdapter,
)
from src.hybrid_spectrum.px6d_session_dataset import (  # noqa: E402
    POSITION_ORDER,
    _load_session_frame_matrix,
    _load_session_recorded_baseline,
    discover_sessions,
)


DEFAULT_CAPTURE_ROOT = Path(
    "E:/重要文档/实验/柔性传感/光纤/Micro-FBG/普通FBG/data/new data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/deployed/ordinary_fbg_current_runtime.joblib",
    )
    parser.add_argument(
        "--peak-config",
        type=Path,
        default=ROOT / "config/hybrid_spectrum_channels.yaml",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "config/runtime_contact_state.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/runtime_position_replay_audit",
    )
    parser.add_argument("--sessions-per-position", type=int, default=2)
    parser.add_argument("--idle-force-max-n", type=float, default=0.03)
    parser.add_argument("--active-force-min-n", type=float, default=0.25)
    return parser.parse_args()


def _runtime_sections(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        dict(payload.get("runtime_baseline_recovery") or {}),
        dict(payload.get("all_source_runtime_gate") or {}),
    )


def _latest_sessions(capture_root: Path, count: int) -> tuple[Any, ...]:
    by_position: dict[str, list[Any]] = defaultdict(list)
    for descriptor in discover_sessions(capture_root):
        if descriptor.position_label in POSITION_ORDER:
            by_position[descriptor.position_label].append(descriptor)
    selected: list[Any] = []
    for position in POSITION_ORDER:
        rows = sorted(
            by_position[position],
            key=lambda item: (item.started_at_epoch_sec, item.session_id),
        )
        selected.extend(rows[-count:])
    return tuple(selected)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_config, gate_config = _runtime_sections(args.runtime_config.resolve())
    adapter = AllSourceOpticalForceAdapter.from_paths(
        args.model.resolve(),
        args.peak_config.resolve(),
        runtime_recovery_config=recovery_config,
        runtime_gate_config=gate_config,
    )
    sessions = _latest_sessions(args.capture_root.resolve(), args.sessions_per_position)
    if len(sessions) != len(POSITION_ORDER) * args.sessions_per_position:
        raise RuntimeError("not every position has the requested number of sessions")

    frame_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for baseline_mode in ("recorded", "first5_median"):
        for descriptor in sessions:
            summary, wavelength, intensity = _load_session_frame_matrix(
                descriptor, expected_points=512
            )
            recorded_baseline = _load_session_recorded_baseline(
                descriptor, expected_points=512
            )
            if baseline_mode == "recorded":
                if recorded_baseline is None:
                    raise RuntimeError(f"missing recorded baseline: {descriptor.session_id}")
                baseline = recorded_baseline
            else:
                baseline = np.median(intensity[: min(5, len(intensity))], axis=0)

            adapter.set_baseline(wavelength, baseline)
            force = summary["force_fz_n"].to_numpy(dtype=float)
            elapsed = summary["elapsed_time_sec"].to_numpy(dtype=float)
            counts: Counter[str] = Counter()
            predicted: Counter[str] = Counter()
            for index, (current, force_n, timestamp) in enumerate(
                zip(intensity, force, elapsed, strict=True)
            ):
                result = adapter.update(
                    wavelength,
                    current,
                    source_timestamp_sec=float(timestamp),
                )
                if not result.get("ok"):
                    raise RuntimeError(
                        f"runtime failed for {descriptor.session_id} frame {index}: {result}"
                    )
                truth_state = (
                    "idle"
                    if force_n <= args.idle_force_max_n
                    else "active"
                    if force_n >= args.active_force_min_n
                    else "transition"
                )
                raw_position = str(result["position"].get("raw_label") or "none")
                formal_position = str(result["position"].get("label") or "none")
                visual_position = str(result["position"].get("visual_label") or "none")
                visual_active = bool(result["digital_twin"].get("visual_active"))
                formal_contact = result["contact"].get("label") == "contact"
                raw_contact = bool(result["runtime_contact_gate"].get("raw_contact_active"))
                counts[f"{truth_state}_frames"] += 1
                if truth_state == "idle":
                    counts["idle_raw_contact"] += int(raw_contact)
                    counts["idle_formal_contact"] += int(formal_contact)
                    counts["idle_visual_active"] += int(visual_active)
                elif truth_state == "active":
                    counts["active_raw_correct"] += int(
                        raw_position == descriptor.position_label
                    )
                    counts["active_formal_correct"] += int(
                        formal_position == descriptor.position_label
                    )
                    counts["active_visual_correct"] += int(
                        visual_position == descriptor.position_label and visual_active
                    )
                    counts["active_visual_covered"] += int(visual_active)
                    counts["active_visual_p23"] += int(
                        visual_position == "P23" and visual_active
                    )
                    predicted[visual_position if visual_active else "none"] += 1
                    confusion[baseline_mode][descriptor.position_label][
                        visual_position if visual_active else "none"
                    ] += 1
                frame_rows.append(
                    {
                        "baseline_mode": baseline_mode,
                        "session_id": descriptor.session_id,
                        "true_position": descriptor.position_label,
                        "capture_index": int(summary.iloc[index]["capture_index"]),
                        "elapsed_time_sec": float(timestamp),
                        "force_fz_n": float(force_n),
                        "truth_state": truth_state,
                        "raw_contact_probability": float(
                            result["contact"]["contact_probability"]
                        ),
                        "raw_position": raw_position,
                        "raw_position_confidence": float(
                            result["position"]["raw_confidence"]
                        ),
                        "formal_position": formal_position,
                        "visual_position": visual_position,
                        "visual_active": visual_active,
                        "baseline_distance": float(
                            result["runtime_contact_gate"]["baseline_distance"]
                        ),
                        "fresh_spectral_activity": bool(
                            result["runtime_contact_gate"]["fresh_spectral_activity"]
                        ),
                    }
                )

            active_frames = counts["active_frames"]
            idle_frames = counts["idle_frames"]
            session_rows.append(
                {
                    "baseline_mode": baseline_mode,
                    "session_id": descriptor.session_id,
                    "true_position": descriptor.position_label,
                    "idle_frames": idle_frames,
                    "active_frames": active_frames,
                    "idle_raw_contact_rate": _safe_ratio(
                        counts["idle_raw_contact"], idle_frames
                    ),
                    "idle_formal_contact_rate": _safe_ratio(
                        counts["idle_formal_contact"], idle_frames
                    ),
                    "idle_visual_false_activation_rate": _safe_ratio(
                        counts["idle_visual_active"], idle_frames
                    ),
                    "active_raw_position_accuracy": _safe_ratio(
                        counts["active_raw_correct"], active_frames
                    ),
                    "active_formal_position_accuracy": _safe_ratio(
                        counts["active_formal_correct"], active_frames
                    ),
                    "active_visual_position_accuracy": _safe_ratio(
                        counts["active_visual_correct"], active_frames
                    ),
                    "active_visual_coverage": _safe_ratio(
                        counts["active_visual_covered"], active_frames
                    ),
                    "active_visual_p23_rate": _safe_ratio(
                        counts["active_visual_p23"], active_frames
                    ),
                    "visual_prediction_counts": json.dumps(
                        dict(predicted), ensure_ascii=False, sort_keys=True
                    ),
                }
            )

    for name, rows in (("frames.csv", frame_rows), ("sessions.csv", session_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    aggregate: dict[str, Any] = {}
    for baseline_mode in ("recorded", "first5_median"):
        current = [row for row in frame_rows if row["baseline_mode"] == baseline_mode]
        idle = [row for row in current if row["truth_state"] == "idle"]
        active = [row for row in current if row["truth_state"] == "active"]
        active_non_p23 = [
            row for row in active if row["true_position"] != "P23"
        ]
        aggregate[baseline_mode] = {
            "session_count": sum(
                row["baseline_mode"] == baseline_mode for row in session_rows
            ),
            "idle_frame_count": len(idle),
            "active_frame_count": len(active),
            "idle_visual_false_activation_rate": float(
                np.mean([row["visual_active"] for row in idle])
            ),
            "active_visual_coverage": float(
                np.mean([row["visual_active"] for row in active])
            ),
            "active_visual_position_accuracy": float(
                np.mean(
                    [
                        row["visual_active"]
                        and row["visual_position"] == row["true_position"]
                        for row in active
                    ]
                )
            ),
            "active_visual_p23_rate": float(
                np.mean(
                    [
                        row["visual_active"] and row["visual_position"] == "P23"
                        for row in active
                    ]
                )
            ),
            "non_p23_false_p23_rate": float(
                np.mean(
                    [
                        row["visual_active"]
                        and row["visual_position"] == "P23"
                        for row in active_non_p23
                    ]
                )
            ),
            "visual_confusion": {
                truth: dict(counts)
                for truth, counts in confusion[baseline_mode].items()
            },
        }
    payload = {
        "schema_version": "touch_deployed_runtime_position_replay_audit_v1",
        "model_path": str(args.model.resolve()),
        "capture_root": str(args.capture_root.resolve()),
        "sessions_per_position": args.sessions_per_position,
        "gate_config": gate_config,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
