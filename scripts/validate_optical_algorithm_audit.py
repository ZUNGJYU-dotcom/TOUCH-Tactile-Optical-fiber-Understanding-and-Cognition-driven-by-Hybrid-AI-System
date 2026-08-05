"""Validate the offline optical algorithm and tactile-information audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "optical_algorithm_and_tactile_information_audit_20260802"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def _check(
    checks: list[dict[str, Any]], check_id: str, passed: bool, detail: str
) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})


def main() -> int:
    args = _parse_args()
    required = [
        "algorithm_and_tactile_information_decision.md",
        "optical_tactile_information_report.md",
        "candidate_model_leaderboard.csv",
        "candidate_grouped_predictions.csv",
        "baseline_confound_ablation_leaderboard.csv",
        "baseline_confound_grouped_predictions.csv",
        "baseline_metadata_feature_audit.csv",
        "tactile_observable_inventory_v2.csv",
        "offline_latency_budget.csv",
        "optical_tactile_information_summary.json",
    ]
    checks: list[dict[str, Any]] = []
    missing = [name for name in required if not (args.audit_dir / name).is_file()]
    _check(
        checks,
        "required_artifacts",
        not missing,
        "all required artifacts present" if not missing else f"missing: {missing}",
    )

    if missing:
        result = {"passed": False, "checks": checks}
        (args.audit_dir / "final_validation_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 1

    candidate = pd.read_csv(args.audit_dir / "candidate_model_leaderboard.csv")
    ablation = pd.read_csv(
        args.audit_dir / "baseline_confound_ablation_leaderboard.csv"
    )
    formal = pd.concat(
        [
            candidate[["split_strategy", "evaluation_validity"]],
            ablation[["split_strategy", "evaluation_validity"]],
        ],
        ignore_index=True,
    )
    grouped_only = formal["split_strategy"].str.contains(
        "grouped_by_session_id", case=False, na=False
    ).all()
    no_random = ~formal.astype(str).apply(
        lambda column: column.str.contains("random", case=False, na=False)
    ).any().any()
    _check(
        checks,
        "formal_grouped_split_only",
        bool(grouped_only and no_random),
        "formal rows use immutable grouped session folds; random frame split absent",
    )

    fold_violations = 0
    for filename in (
        "candidate_grouped_predictions.csv",
        "baseline_confound_grouped_predictions.csv",
    ):
        predictions = pd.read_csv(args.audit_dir / filename)
        key_columns = ["model_id", "feature_view", "task", "session_id"]
        fold_counts = predictions.groupby(key_columns, dropna=False)["fold_id"].nunique()
        fold_violations += int((fold_counts > 1).sum())
    _check(
        checks,
        "one_fold_per_session",
        fold_violations == 0,
        f"session/fold assignment violations: {fold_violations}",
    )

    inventory = pd.read_csv(args.audit_dir / "tactile_observable_inventory_v2.csv")
    fz_rows = inventory.loc[inventory["observable"] == "optical-only Fz estimate"]
    fz_semantics_ok = (
        len(fz_rows) == 1
        and fz_rows.iloc[0]["status"]
        == "calibration_target_not_counted_as_new_observable"
    )
    _check(
        checks,
        "fz_semantics",
        bool(fz_semantics_ok),
        "Fz is a calibration target and is not counted as a new tactile observable",
    )

    metadata = pd.read_csv(args.audit_dir / "baseline_metadata_feature_audit.csv")
    qa_only = (metadata["recommended_role"] == "qa_and_drift_monitoring_only").all()
    excluded = metadata["excluded_from_response_only_view"].astype(str).str.lower().eq(
        "true"
    ).all()
    _check(
        checks,
        "baseline_metadata_excluded",
        bool(qa_only and excluded),
        f"QA-only baseline metadata fields: {len(metadata)}",
    )

    latency = pd.read_csv(args.audit_dir / "offline_latency_budget.csv")
    live = latency.loc[latency["component"] == "current_live_end_to_end"]
    offline_boundary_ok = (
        len(live) == 1
        and live.iloc[0]["measurement_status"]
        == "not_measured_hardware_disconnected"
    )
    _check(
        checks,
        "offline_hardware_boundary",
        bool(offline_boundary_ok),
        "live end-to-end latency is explicitly not measured while hardware is disconnected",
    )

    summary = json.loads(
        (args.audit_dir / "optical_tactile_information_summary.json").read_text(
            encoding="utf-8"
        )
    )
    dataset_ok = (
        summary["dataset"]["frame_count"] == 10528
        and summary["dataset"]["independent_session_count"] == 50
        and summary["dataset"]["formal_split"]
        == "immutable grouped_by_session_id"
    )
    _check(
        checks,
        "dataset_identity",
        bool(dataset_ok),
        "10,528 frames from 50 independent latest-primary sessions",
    )

    result = {
        "passed": all(item["passed"] for item in checks),
        "audit_dir": str(args.audit_dir),
        "hardware_connected": False,
        "live_validation_performed": False,
        "deployment_changed": False,
        "checks": checks,
    }
    (args.audit_dir / "final_validation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
