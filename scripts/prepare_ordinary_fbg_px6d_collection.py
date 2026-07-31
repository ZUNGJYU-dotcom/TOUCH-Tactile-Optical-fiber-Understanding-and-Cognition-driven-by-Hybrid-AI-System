"""Generate a balanced ordinary-FBG plus PX6D collection manifest."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import random
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITION_ORDER = (
    "P11",
    "P21",
    "P31",
    "P12",
    "P22",
    "P32",
    "P13",
    "P23",
    "P33",
)
PILOT_ORDER = (
    "P22",
    "P11",
    "P33",
    "P31",
    "P13",
    "P21",
    "P12",
    "P32",
    "P23",
)
FIELDNAMES = (
    "sequence_index",
    "phase",
    "block_id",
    "order_in_block",
    "trial_id",
    "position_label",
    "action_label",
    "loading_program",
    "expected_duration_sec",
    "baseline_sec",
    "loading_sec",
    "hold_sec",
    "unloading_sec",
    "recovery_sec",
    "target_force_n",
    "maximum_safe_force_n",
    "completion_status",
    "session_output_directory",
    "qa_status",
    "operator_note",
)


def _shuffled_positions(
    rng: random.Random,
    *,
    previous: str | None = None,
) -> list[str]:
    positions = list(POSITION_ORDER)
    for _ in range(100):
        rng.shuffle(positions)
        if previous is None or positions[0] != previous:
            return positions
    if previous is not None and positions[0] == previous:
        positions[0], positions[1] = positions[1], positions[0]
    return positions


def _trial_row(
    *,
    phase: str,
    block_id: str,
    order_in_block: int,
    trial_id: str,
    position_label: str,
    loading_program: str,
    baseline_sec: float,
    loading_sec: float,
    hold_sec: float,
    unloading_sec: float,
    recovery_sec: float,
) -> dict[str, object]:
    return {
        "phase": phase,
        "block_id": block_id,
        "order_in_block": order_in_block,
        "trial_id": trial_id,
        "position_label": position_label,
        "action_label": "continuous_px6d_fz_reference",
        "loading_program": loading_program,
        "expected_duration_sec": (
            baseline_sec + loading_sec + hold_sec + unloading_sec + recovery_sec
        ),
        "baseline_sec": baseline_sec,
        "loading_sec": loading_sec,
        "hold_sec": hold_sec,
        "unloading_sec": unloading_sec,
        "recovery_sec": recovery_sec,
        "target_force_n": "",
        "maximum_safe_force_n": "",
        "completion_status": "pending",
        "session_output_directory": "",
        "qa_status": "not_checked",
        "operator_note": "",
    }


def build_manifest(
    *,
    session_date: str,
    formal_repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    if formal_repeats < 1:
        raise ValueError("formal_repeats must be at least 1")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    rows.append(
        _trial_row(
            phase="baseline_checkpoint",
            block_id="baseline_start",
            order_in_block=1,
            trial_id=f"{session_date}_BASE_START",
            position_label="unlabeled",
            loading_program="no_contact_only",
            baseline_sec=12.0,
            loading_sec=0.0,
            hold_sec=0.0,
            unloading_sec=0.0,
            recovery_sec=0.0,
        )
    )

    for index, position in enumerate(PILOT_ORDER, start=1):
        rows.append(
            _trial_row(
                phase="pilot",
                block_id="pilot_all_positions",
                order_in_block=index,
                trial_id=f"{session_date}_{position}_PILOT",
                position_label=position,
                loading_program="slow_ramp_hold",
                baseline_sec=4.0,
                loading_sec=8.0,
                hold_sec=4.0,
                unloading_sec=4.0,
                recovery_sec=6.0,
            )
        )

    rows.append(
        _trial_row(
            phase="baseline_checkpoint",
            block_id="baseline_after_pilot",
            order_in_block=1,
            trial_id=f"{session_date}_BASE_AFTER_PILOT",
            position_label="unlabeled",
            loading_program="no_contact_only",
            baseline_sec=12.0,
            loading_sec=0.0,
            hold_sec=0.0,
            unloading_sec=0.0,
            recovery_sec=0.0,
        )
    )

    previous_position = PILOT_ORDER[-1]
    for repeat_index in range(1, formal_repeats + 1):
        block_id = f"formal_repeat_{repeat_index:02d}"
        positions = _shuffled_positions(rng, previous=previous_position)
        loading_program = (
            "smooth_ramp_hold"
            if repeat_index % 2
            else "three_plateau_hold"
        )
        for order_index, position in enumerate(positions, start=1):
            if loading_program == "smooth_ramp_hold":
                timing = (4.0, 6.0, 4.0, 3.0, 5.0)
            else:
                timing = (4.0, 9.0, 6.0, 3.0, 5.0)
            rows.append(
                _trial_row(
                    phase="formal",
                    block_id=block_id,
                    order_in_block=order_index,
                    trial_id=f"{session_date}_{position}_R{repeat_index:02d}",
                    position_label=position,
                    loading_program=loading_program,
                    baseline_sec=timing[0],
                    loading_sec=timing[1],
                    hold_sec=timing[2],
                    unloading_sec=timing[3],
                    recovery_sec=timing[4],
                )
            )
        previous_position = positions[-1]
        if repeat_index % 2 == 0 and repeat_index != formal_repeats:
            rows.append(
                _trial_row(
                    phase="baseline_checkpoint",
                    block_id=f"baseline_after_repeat_{repeat_index:02d}",
                    order_in_block=1,
                    trial_id=f"{session_date}_BASE_R{repeat_index:02d}",
                    position_label="unlabeled",
                    loading_program="no_contact_only",
                    baseline_sec=12.0,
                    loading_sec=0.0,
                    hold_sec=0.0,
                    unloading_sec=0.0,
                    recovery_sec=0.0,
                )
            )

    rows.append(
        _trial_row(
            phase="baseline_checkpoint",
            block_id="baseline_end",
            order_in_block=1,
            trial_id=f"{session_date}_BASE_END",
            position_label="unlabeled",
            loading_program="no_contact_only",
            baseline_sec=12.0,
            loading_sec=0.0,
            hold_sec=0.0,
            unloading_sec=0.0,
            recovery_sec=0.0,
        )
    )

    for sequence_index, row in enumerate(rows, start=1):
        row["sequence_index"] = sequence_index
    return rows


def _position_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {position: 0 for position in POSITION_ORDER}
    for row in rows:
        position = str(row["position_label"])
        if position in counts and row["phase"] in {"pilot", "formal"}:
            counts[position] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pilot plus balanced randomized formal collection manifest "
            "for ordinary-FBG full spectra synchronized with PX6D force."
        )
    )
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y%m%d"),
        help="Session date in YYYYMMDD format.",
    )
    parser.add_argument("--formal-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/collection_plans/ordinary_fbg_px6d_<date>.",
    )
    args = parser.parse_args()

    session_date = str(args.date).strip()
    if len(session_date) != 8 or not session_date.isdigit():
        parser.error("--date must use YYYYMMDD")
    seed = int(args.seed if args.seed is not None else session_date)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / "data"
        / "collection_plans"
        / f"ordinary_fbg_px6d_{session_date}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(
        session_date=session_date,
        formal_repeats=args.formal_repeats,
        seed=seed,
    )
    manifest_path = output_dir / "collection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "ordinary_fbg_px6d_collection_plan_v1",
        "session_date": session_date,
        "seed": seed,
        "formal_repeats_per_position": args.formal_repeats,
        "pilot_trials": sum(row["phase"] == "pilot" for row in rows),
        "formal_trials": sum(row["phase"] == "formal" for row in rows),
        "baseline_checkpoints": sum(
            row["phase"] == "baseline_checkpoint" for row in rows
        ),
        "total_trials": len(rows),
        "position_counts_including_pilot": _position_counts(rows),
        "force_target": "continuous_force_fz_n_regression",
        "force_class_labels": False,
        "formal_split_group": "trial_id",
        "manifest_path": str(manifest_path),
    }
    (output_dir / "collection_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
