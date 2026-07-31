"""Train grouped optical-only candidates from every valid ordinary-FBG source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_spectrum.all_source_fusion import resolve_project_path
from hybrid_spectrum.all_source_training import (
    default_variants,
    evaluate_force_contact_gate,
    fit_candidate_bundle,
    grouped_cross_validation,
    leaderboard_from_metrics,
    load_fusion_arrays,
    save_training_outputs,
    select_candidate_models,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/ordinary_fbg_all_data_fusion.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/ordinary_fbg_all_data_fusion_20260731_v1/"
            "all_source_fusion_dataset.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/ordinary_fbg_all_data_fusion_training_20260731_v1"
        ),
    )
    parser.add_argument(
        "--reuse-oof",
        action="store_true",
        help="Reuse existing grouped OOF artifacts and only rebuild the candidate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    arrays = load_fusion_arrays(dataset_path)
    protected_model = resolve_project_path(
        PROJECT_ROOT, config["paths"]["protected_deployed_model"]
    )
    protected_hash = sha256_file(protected_model)
    variants = default_variants()
    if args.reuse_oof:
        predictions = pd.read_csv(
            output_dir / "grouped_oof_predictions.csv",
            low_memory=False,
        )
        split_audit = pd.read_csv(output_dir / "grouped_split_audit.csv")
        metrics = json.loads(
            (output_dir / "all_data_model_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        leaderboard = pd.read_csv(
            output_dir / "all_data_model_leaderboard.csv"
        )
    else:
        predictions, split_audit, metrics = grouped_cross_validation(
            arrays, variants, config
        )
        leaderboard = leaderboard_from_metrics(metrics)
    selected = select_candidate_models(leaderboard)
    force_gate_predictions, force_gate_metrics = evaluate_force_contact_gate(
        arrays,
        config,
        predictions,
        selected,
        variants,
    )
    candidate_path = (
        output_dir
        / "candidate_models/ordinary_fbg_optical_only_force_candidate.joblib"
    )
    candidate_summary = fit_candidate_bundle(
        arrays,
        selected,
        variants,
        config,
        metrics,
        force_gate_metrics,
        candidate_path,
    )
    summary = save_training_outputs(
        output_dir=output_dir,
        arrays=arrays,
        config=config,
        dataset_path=dataset_path,
        protected_model_path=protected_model,
        protected_hash_before=protected_hash,
        predictions=predictions,
        split_audit=split_audit,
        metrics=metrics,
        leaderboard=leaderboard,
        force_gate_predictions=force_gate_predictions,
        force_gate_metrics=force_gate_metrics,
        candidate_summary=candidate_summary,
        selected=selected,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
