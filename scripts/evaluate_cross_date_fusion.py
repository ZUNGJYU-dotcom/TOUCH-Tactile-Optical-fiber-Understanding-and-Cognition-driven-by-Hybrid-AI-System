"""Evaluate optical-only models with acquisition dates held out in turn."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hybrid_spectrum.all_source_training import (  # noqa: E402
    CONTACT_CLASSES,
    POSITION_CLASSES,
    _classification_model,
    _regression_model,
    feature_indices,
    load_fusion_arrays,
    source_group_weights,
)


DATE_PATTERN = re.compile(r"^(\d{8})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def acquisition_dates(group_ids: np.ndarray) -> np.ndarray:
    values: list[str] = []
    for group_id in group_ids.astype(str):
        match = DATE_PATTERN.match(group_id)
        values.append(match.group(1) if match else "unknown")
    return np.asarray(values, dtype=str)


def equal_group_mass_weights(groups: np.ndarray, selected: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(groups), dtype=np.float64)
    for group in sorted(set(groups[selected].astype(str).tolist())):
        indices = np.flatnonzero(selected & (groups == group))
        weights[indices] = 1.0 / max(1, len(indices))
    mean = float(np.mean(weights[selected]))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("invalid equal-session-mass weights")
    weights[selected] /= mean
    return weights


def majority_vote_metrics(
    true: np.ndarray, predicted: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    group_true: list[str] = []
    group_predicted: list[str] = []
    for group in sorted(set(groups.astype(str).tolist())):
        selected = groups == group
        group_true.append(Counter(true[selected].tolist()).most_common(1)[0][0])
        group_predicted.append(
            Counter(predicted[selected].tolist()).most_common(1)[0][0]
        )
    return (
        float(accuracy_score(group_true, group_predicted)),
        float(f1_score(group_true, group_predicted, average="macro", zero_division=0)),
    )


def safe_pearson(true: np.ndarray, predicted: np.ndarray) -> float:
    if len(true) < 2 or np.std(true) <= 1.0e-12 or np.std(predicted) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(true, predicted)[0, 1])


def regression_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(true, predicted, 1)
    return {
        "mae_n": float(mean_absolute_error(true, predicted)),
        "rmse_n": float(np.sqrt(mean_squared_error(true, predicted))),
        "r2": float(r2_score(true, predicted)),
        "pearson_r": safe_pearson(true, predicted),
        "slope": float(slope),
        "intercept_n": float(intercept),
    }


def model_specs() -> tuple[tuple[str, str, str, str], ...]:
    specs: list[tuple[str, str, str, str]] = [
        (
            "primary_temporal_extra_trees",
            "latest_primary_only",
            "temporal_fusion",
            "extra_trees",
        ),
        (
            "fused_current_extra_trees",
            "all_sources",
            "current_frame",
            "extra_trees",
        ),
        (
            "fused_temporal_extra_trees",
            "all_sources",
            "temporal_fusion",
            "extra_trees",
        ),
        (
            "fused_temporal_random_forest",
            "all_sources",
            "temporal_fusion",
            "random_forest",
        ),
    ]
    try:
        _classification_model(
            "lightgbm", estimators=1, minimum_leaf_samples=2, random_seed=42
        )
    except ValueError:
        pass
    else:
        specs.append(
            (
                "fused_temporal_lightgbm",
                "all_sources",
                "temporal_fusion",
                "lightgbm",
            )
        )
    return tuple(specs)


def date_holdout_masks(
    *,
    arrays: Any,
    dates: np.ndarray,
    task_mask: np.ndarray,
    test_date: str,
    source_regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    formal = arrays.formal_test_eligible
    formal_train = task_mask & formal & (dates != test_date)
    test = task_mask & formal & (dates == test_date)
    if source_regime == "latest_primary_only":
        train = formal_train
    elif source_regime == "all_sources":
        auxiliary_train = task_mask & ~formal & (dates != test_date)
        train = formal_train | auxiliary_train
    else:
        raise ValueError(f"unknown source regime: {source_regime}")
    return train, test


def evaluate_classification(
    *,
    arrays: Any,
    dates: np.ndarray,
    task: str,
    classes: tuple[str, ...],
    target: np.ndarray,
    task_mask: np.ndarray,
    test_date: str,
    model_id: str,
    source_regime: str,
    feature_view: str,
    family: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, test = date_holdout_masks(
        arrays=arrays,
        dates=dates,
        task_mask=task_mask,
        test_date=test_date,
        source_regime=source_regime,
    )
    train_groups = set(arrays.group_id[train].astype(str).tolist())
    test_groups = set(arrays.group_id[test].astype(str).tolist())
    if train_groups.intersection(test_groups):
        raise RuntimeError("session leakage across the date holdout")
    indices = feature_indices(arrays.feature_names, feature_view)
    model_config = dict(config["models"])
    evaluation = dict(config["evaluation"])
    model = _classification_model(
        family,
        estimators=int(model_config.get("tree_estimators", 180)),
        minimum_leaf_samples=int(model_config.get("minimum_leaf_samples", 2)),
        random_seed=int(evaluation.get("random_seed", 42)),
    )
    weights = source_group_weights(
        arrays.source_role,
        arrays.group_id,
        train,
        dict(config["source_policy"]),
    )
    model.fit(
        arrays.features[train][:, indices], target[train], sample_weight=weights[train]
    )
    predicted = np.asarray(model.predict(arrays.features[test][:, indices]), dtype=str)
    true = target[test].astype(str)
    group_accuracy, group_macro_f1 = majority_vote_metrics(
        true, predicted, arrays.group_id[test]
    )
    recalls = recall_score(
        true, predicted, labels=list(classes), average=None, zero_division=0
    )
    metrics: dict[str, Any] = {
        "task": task,
        "model_id": model_id,
        "source_regime": source_regime,
        "feature_view": feature_view,
        "model_family": family,
        "test_date": test_date,
        "train_dates": ",".join(sorted(set(dates[train].tolist()))),
        "train_frames": int(np.sum(train)),
        "test_frames": int(np.sum(test)),
        "train_sessions": len(train_groups),
        "test_sessions": len(test_groups),
        "auxiliary_train_frames": int(np.sum(train & ~arrays.formal_test_eligible)),
        "auxiliary_train_sessions": len(
            set(arrays.group_id[train & ~arrays.formal_test_eligible].astype(str))
        ),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "file_voting_accuracy": group_accuracy,
        "file_voting_macro_f1": group_macro_f1,
    }
    metrics.update(
        {f"recall_{label}": float(value) for label, value in zip(classes, recalls, strict=True)}
    )
    predictions = pd.DataFrame(
        {
            "task": task,
            "model_id": model_id,
            "test_date": test_date,
            "group_id": arrays.group_id[test],
            "sample_index": arrays.sample_index[test],
            "true_label": true,
            "predicted_label": predicted,
        }
    )
    return metrics, predictions


def fit_contact_gate(
    arrays: Any,
    dates: np.ndarray,
    test_date: str,
    source_regime: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    train, _ = date_holdout_masks(
        arrays=arrays,
        dates=dates,
        task_mask=arrays.contact_mask,
        test_date=test_date,
        source_regime=source_regime,
    )
    labels = np.where(arrays.contact_target == 1, "contact", "no_contact")
    indices = feature_indices(arrays.feature_names, "temporal_fusion")
    family = "lightgbm"
    try:
        model = _classification_model(
            family,
            estimators=int(config["models"].get("tree_estimators", 180)),
            minimum_leaf_samples=int(config["models"].get("minimum_leaf_samples", 2)),
            random_seed=int(config["evaluation"].get("random_seed", 42)),
        )
    except ValueError:
        family = "extra_trees"
        model = _classification_model(
            family,
            estimators=int(config["models"].get("tree_estimators", 180)),
            minimum_leaf_samples=int(config["models"].get("minimum_leaf_samples", 2)),
            random_seed=int(config["evaluation"].get("random_seed", 42)),
        )
    weights = source_group_weights(
        arrays.source_role,
        arrays.group_id,
        train,
        dict(config["source_policy"]),
    )
    model.fit(
        arrays.features[train][:, indices], labels[train], sample_weight=weights[train]
    )
    test = arrays.force_mask & arrays.formal_test_eligible & (dates == test_date)
    probabilities = model.predict_proba(arrays.features[test][:, indices])
    class_names = np.asarray(model.classes_, dtype=str)
    contact_index = int(np.flatnonzero(class_names == "contact")[0])
    return probabilities[:, contact_index].astype(float), np.asarray([family] * np.sum(test))


def evaluate_force(
    *,
    arrays: Any,
    dates: np.ndarray,
    test_date: str,
    model_id: str,
    source_regime: str,
    feature_view: str,
    family: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, test = date_holdout_masks(
        arrays=arrays,
        dates=dates,
        task_mask=arrays.force_mask,
        test_date=test_date,
        source_regime=source_regime,
    )
    train_groups = set(arrays.group_id[train].astype(str).tolist())
    test_groups = set(arrays.group_id[test].astype(str).tolist())
    if train_groups.intersection(test_groups):
        raise RuntimeError("session leakage across the date holdout")
    indices = feature_indices(arrays.feature_names, feature_view)
    model = _regression_model(
        family,
        estimators=int(config["models"].get("tree_estimators", 180)),
        minimum_leaf_samples=int(config["models"].get("minimum_leaf_samples", 2)),
        random_seed=int(config["evaluation"].get("random_seed", 42)),
    )
    weights = source_group_weights(
        arrays.source_role,
        arrays.group_id,
        train,
        dict(config["source_policy"]),
    )
    model.fit(
        arrays.features[train][:, indices], arrays.force_fz_n[train], sample_weight=weights[train]
    )
    raw = np.clip(model.predict(arrays.features[test][:, indices]), 0.0, 5.0)
    contact_probability, gate_family = fit_contact_gate(
        arrays,
        dates,
        test_date,
        source_regime,
        config,
    )
    threshold = float(
        config["force_calibration"]["optical_contact_gate"]["probability_threshold"]
    )
    gated = np.where(contact_probability >= threshold, raw, 0.0)
    true = arrays.force_fz_n[test].astype(float)
    raw_metrics = regression_metrics(true, raw)
    gated_metrics = regression_metrics(true, gated)
    no_contact = true <= float(config["labels"]["no_contact_max_force_n"])
    active = true >= float(config["labels"]["contact_min_force_n"])
    metrics: dict[str, Any] = {
        "task": "force_fz",
        "model_id": model_id,
        "source_regime": source_regime,
        "feature_view": feature_view,
        "model_family": family,
        "contact_gate_family": str(gate_family[0]) if len(gate_family) else "",
        "contact_gate_threshold": threshold,
        "test_date": test_date,
        "train_dates": ",".join(sorted(set(dates[train].tolist()))),
        "train_frames": int(np.sum(train)),
        "test_frames": int(np.sum(test)),
        "train_sessions": len(train_groups),
        "test_sessions": len(test_groups),
        "auxiliary_train_frames": int(np.sum(train & ~arrays.formal_test_eligible)),
        "auxiliary_train_sessions": len(
            set(arrays.group_id[train & ~arrays.formal_test_eligible].astype(str))
        ),
        **{f"raw_{key}": value for key, value in raw_metrics.items()},
        **{f"gated_{key}": value for key, value in gated_metrics.items()},
        "gate_active_recall": float(np.mean(contact_probability[active] >= threshold))
        if np.any(active)
        else float("nan"),
        "gate_no_contact_false_positive_rate": float(
            np.mean(contact_probability[no_contact] >= threshold)
        )
        if np.any(no_contact)
        else float("nan"),
    }
    predictions = pd.DataFrame(
        {
            "task": "force_fz",
            "model_id": model_id,
            "test_date": test_date,
            "group_id": arrays.group_id[test],
            "sample_index": arrays.sample_index[test],
            "elapsed_time_sec": arrays.elapsed_time_sec[test],
            "true_force_n": true,
            "raw_predicted_force_n": raw,
            "contact_probability": contact_probability,
            "gated_predicted_force_n": gated,
        }
    )
    return metrics, predictions


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = metrics.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column not in {"train_frames", "test_frames", "train_sessions", "test_sessions"}]
    rows: list[dict[str, Any]] = []
    for (task, model_id), group in metrics.groupby(["task", "model_id"], sort=True):
        row: dict[str, Any] = {
            "task": task,
            "model_id": model_id,
            "date_holdout_count": int(group["test_date"].nunique()),
        }
        for column in numeric:
            if column in group:
                row[f"mean_{column}"] = float(group[column].mean())
                row[f"worst_{column}"] = float(group[column].max() if "mae" in column or "rmse" in column or "false_positive" in column else group[column].min())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(aggregate: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    specifications = (
        ("contact", "mean_macro_f1", "Contact macro-F1", True),
        ("position", "mean_macro_f1", "Position macro-F1", True),
        ("force_fz", "mean_gated_mae_n", "Force gated MAE (N)", False),
    )
    for axis, (task, field, title, higher_better) in zip(axes, specifications, strict=True):
        rows = aggregate[aggregate["task"] == task].dropna(subset=[field]).copy()
        rows = rows.sort_values(field, ascending=not higher_better)
        axis.barh(rows["model_id"], rows[field], color="#2C9AB7")
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Bidirectional acquisition-date holdout", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def dataframe_to_markdown(frame: pd.DataFrame, float_digits: int | None = None) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)) and float_digits is not None:
            text = f"{float(value):.{float_digits}f}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(render(value) for value in record) + " |")
    return "\n".join(rows)


def write_report(
    output: Path,
    dates: list[str],
    inventory: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    lines = [
        "# Cross-date fusion validation",
        "",
        "- Runtime inputs are optical features only.",
        "- PX6D Fz is supervision and evaluation reference only.",
        "- Every split holds out a complete acquisition date; session IDs never cross train/test.",
        "- Frames are repeated observations, not independent experiments.",
        "- This is candidate validation and does not deploy a model.",
        "",
        f"Acquisition dates: {', '.join(dates)}.",
        "",
        "## Inventory",
        "",
        dataframe_to_markdown(inventory),
        "",
        "## Bidirectional date-holdout summary",
        "",
        dataframe_to_markdown(aggregate, float_digits=4),
        "",
        "## Interpretation",
        "",
        "Grouped five-fold results quantify out-of-session performance within the pooled data. The date-holdout results are the stricter domain-shift check: a model is trained on one acquisition date and evaluated on the other without seeing any session from that date. A pooled score should not be treated as cross-date generalization when the corresponding date-holdout result is weak.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")) or {}
    arrays = load_fusion_arrays(args.dataset.resolve())
    dates = acquisition_dates(arrays.group_id)
    formal_dates = sorted(set(dates[arrays.formal_test_eligible].tolist()))
    configured_dates = [str(value) for value in config["evaluation"].get("acquisition_dates", [])]
    if configured_dates and formal_dates != sorted(configured_dates):
        raise ValueError(
            f"formal dates {formal_dates} do not match configured dates {configured_dates}"
        )
    if len(formal_dates) < 2:
        raise ValueError("cross-date validation requires at least two acquisition dates")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    for date in formal_dates:
        selected = dates == date
        formal_selected = arrays.formal_test_eligible & selected
        auxiliary_selected = ~arrays.formal_test_eligible & selected
        inventory_rows.append(
            {
                "acquisition_date": date,
                "all_sessions": int(len(set(arrays.group_id[selected].tolist()))),
                "all_frames": int(np.sum(selected)),
                "formal_sessions": int(
                    len(set(arrays.group_id[formal_selected].tolist()))
                ),
                "formal_frames": int(np.sum(formal_selected)),
                "auxiliary_sessions": int(
                    len(set(arrays.group_id[auxiliary_selected].tolist()))
                ),
                "auxiliary_frames": int(np.sum(auxiliary_selected)),
                "contact_frames": int(np.sum(selected & arrays.contact_mask)),
                "position_frames": int(np.sum(selected & arrays.position_mask)),
                "force_frames": int(np.sum(selected & arrays.force_mask)),
            }
        )
    inventory = pd.DataFrame(inventory_rows)

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    specs = model_specs()
    contact_target = np.where(arrays.contact_target == 1, "contact", "no_contact")
    for test_date in formal_dates:
        for model_id, source_regime, feature_view, family in specs:
            for task, classes, target, task_mask in (
                ("contact", CONTACT_CLASSES, contact_target, arrays.contact_mask),
                ("position", POSITION_CLASSES, arrays.position_target, arrays.position_mask),
            ):
                metrics, predictions = evaluate_classification(
                    arrays=arrays,
                    dates=dates,
                    task=task,
                    classes=classes,
                    target=target,
                    task_mask=task_mask,
                    test_date=test_date,
                    model_id=model_id,
                    source_regime=source_regime,
                    feature_view=feature_view,
                    family=family,
                    config=config,
                )
                all_metrics.append(metrics)
                all_predictions.append(predictions)
            metrics, predictions = evaluate_force(
                arrays=arrays,
                dates=dates,
                test_date=test_date,
                model_id=model_id,
                source_regime=source_regime,
                feature_view=feature_view,
                family=family,
                config=config,
            )
            all_metrics.append(metrics)
            all_predictions.append(predictions)

    metrics_frame = pd.DataFrame(all_metrics)
    aggregate = aggregate_metrics(metrics_frame)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    inventory.to_csv(output_dir / "cross_date_inventory.csv", index=False, encoding="utf-8-sig")
    metrics_frame.to_csv(output_dir / "cross_date_metrics_by_holdout.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(output_dir / "cross_date_model_summary.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(output_dir / "cross_date_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "cross_date_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "ordinary_fbg_cross_date_validation_v1",
                "acquisition_dates": formal_dates,
                "split_strategy": "bidirectional_acquisition_date_holdout_grouped_by_session_id",
                "runtime_force_input_allowed": False,
                "metrics_by_holdout": all_metrics,
                "aggregate": aggregate.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(aggregate, output_dir / "cross_date_model_summary.png")
    write_report(
        output_dir / "cross_date_validation_report.md",
        formal_dates,
        inventory,
        aggregate,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dates": formal_dates,
                "session_count": int(len(set(arrays.group_id[arrays.formal_test_eligible].tolist()))),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
