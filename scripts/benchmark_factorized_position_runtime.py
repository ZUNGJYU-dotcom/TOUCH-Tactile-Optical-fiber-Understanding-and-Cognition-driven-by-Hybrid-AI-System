"""Benchmark compact factorized position models with grouped validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.dynamic_temporal_features import temporal_summary_features  # noqa: E402
from src.hybrid_spectrum.factorized_position import FactorizedPositionClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-counts", default="160,240,320,400")
    parser.add_argument("--seeds", default="42,73,104,135,166")
    parser.add_argument("--latency-repeats", type=int, default=200)
    return parser.parse_args()


def estimator(tree_count: int, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=tree_count,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree_counts = [int(value) for value in args.tree_counts.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]

    with np.load(args.dataset, allow_pickle=True) as data:
        values = np.asarray(data["X"], dtype=np.float32)
        stage = np.asarray(data["stage_labels"]).astype(str)
        position = np.asarray(data["position_labels"]).astype(str)
        groups = np.asarray(data["capture_groups"]).astype(str)
    summary = temporal_summary_features(values)
    active = np.isin(stage, ("light", "normal", "hard"))

    results: list[dict[str, float | int]] = []
    for tree_count in tree_counts:
        macro_scores: list[float] = []
        p21_recalls: list[float] = []
        for seed in seeds:
            truth_parts: list[np.ndarray] = []
            prediction_parts: list[np.ndarray] = []
            for fold_index, test_group in enumerate(("G1", "G2", "G3")):
                train = active & (groups != test_group)
                test = active & (groups == test_group)
                model = FactorizedPositionClassifier(
                    estimator(tree_count, seed + fold_index)
                ).fit(summary[train], position[train])
                truth_parts.append(position[test])
                prediction_parts.append(model.predict(summary[test]))
            truth = np.concatenate(truth_parts)
            predicted = np.concatenate(prediction_parts)
            macro_scores.append(float(f1_score(truth, predicted, average="macro")))
            p21_recalls.append(
                float(recall_score(truth == "P21", predicted == "P21"))
            )

        final_model = FactorizedPositionClassifier(
            estimator(tree_count, seeds[0] + 1000)
        ).fit(summary[active], position[active])
        final_model.set_runtime_n_jobs(1)
        probe = summary[np.flatnonzero(active)[0:1]]
        for _ in range(10):
            final_model.predict_proba(probe)
        latency_samples = []
        for _ in range(args.latency_repeats):
            started = perf_counter()
            final_model.predict_proba(probe)
            latency_samples.append((perf_counter() - started) * 1000.0)
        results.append(
            {
                "tree_count_per_axis": tree_count,
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_std": float(np.std(macro_scores)),
                "macro_f1_min": float(np.min(macro_scores)),
                "p21_recall_mean": float(np.mean(p21_recalls)),
                "p21_recall_min": float(np.min(p21_recalls)),
                "position_only_latency_median_ms": float(np.median(latency_samples)),
                "position_only_latency_p95_ms": float(np.percentile(latency_samples, 95)),
            }
        )

    payload = {
        "schema_version": "factorized_position_tree_latency_sweep_v1",
        "evaluation_validity": "grouped_by_capture_group_and_file_id",
        "independent_dat_sequences": 27,
        "selection_rule": (
            "smallest tree count whose macro-F1 mean is within 0.005 of the best "
            "and whose P21 recall mean is within 0.03 of the best"
        ),
        "results": results,
    }
    best_macro = max(float(row["macro_f1_mean"]) for row in results)
    best_p21 = max(float(row["p21_recall_mean"]) for row in results)
    eligible = [
        row
        for row in results
        if float(row["macro_f1_mean"]) >= best_macro - 0.005
        and float(row["p21_recall_mean"]) >= best_p21 - 0.03
    ]
    payload["recommended_tree_count_per_axis"] = int(
        min(eligible, key=lambda row: int(row["tree_count_per_axis"]))[
            "tree_count_per_axis"
        ]
    )
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
