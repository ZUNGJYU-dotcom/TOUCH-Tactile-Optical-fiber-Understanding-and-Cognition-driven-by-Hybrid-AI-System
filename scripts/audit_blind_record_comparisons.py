"""Audit frozen runtime replay against synchronized Record-time evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.force_consistency_audit import (  # noqa: E402
    force_curve_metrics,
)


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
POSITION_COLORS = {
    "none": "#9aa9b7",
    "P11": "#146c94",
    "P12": "#1f9e89",
    "P13": "#66a61e",
    "P21": "#e6ab02",
    "P22": "#d95f02",
    "P23": "#e76f51",
    "P31": "#7570b3",
    "P32": "#8e5ea2",
    "P33": "#b24745",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-force-min-n", type=float, default=0.25)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in POSITION_ORDER else "none"


def _majority(values: Iterable[str]) -> tuple[str, int, int, float]:
    counts = Counter(value for value in values if value in POSITION_ORDER)
    if not counts:
        return "none", 0, 0, 0.0
    rank = {label: index for index, label in enumerate(POSITION_ORDER)}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], rank[item[0]]))
    winner, winner_count = ordered[0]
    runner_count = ordered[1][1] if len(ordered) > 1 else 0
    return winner, winner_count, runner_count, winner_count / sum(counts.values())


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_percentile(values: pd.Series, percentile: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.percentile(numeric, percentile)) if len(numeric) else None


def _edge_time(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    edge: str,
) -> float | None:
    selected = pd.to_numeric(
        frame.loc[mask, "elapsed_time_sec"], errors="coerce"
    ).to_numpy(dtype=float)
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return None
    if edge == "first":
        return float(np.min(selected))
    if edge == "last":
        return float(np.max(selected))
    raise ValueError(f"unsupported edge: {edge}")


def _time_offset_ms(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return float((value - reference) * 1000.0)


def _transition_count(values: Iterable[str]) -> int:
    transitions = 0
    previous: str | None = None
    for value in values:
        current = _label(value)
        if previous is not None and current != previous:
            transitions += 1
        previous = current
    return transitions


def _basic_force_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[valid]
    estimate = estimate[valid]
    if not len(reference):
        return {
            "sample_count": 0,
            "mae_n": None,
            "rmse_n": None,
            "bias_n": None,
            "r2": None,
            "pearson_r": None,
            "slope": None,
            "intercept_n": None,
        }
    error = estimate - reference
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    correlation = None
    if len(reference) >= 3 and np.std(reference) > 1.0e-12 and np.std(estimate) > 1.0e-12:
        correlation = float(np.corrcoef(reference, estimate)[0, 1])
    slope = intercept = None
    if len(reference) >= 2 and np.std(reference) > 1.0e-12:
        fitted = np.polyfit(reference, estimate, 1)
        slope, intercept = float(fitted[0]), float(fitted[1])
    return {
        "sample_count": int(len(reference)),
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(error**2))),
        "bias_n": float(np.mean(error)),
        "r2": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 1.0e-12
            else None
        ),
        "pearson_r": correlation,
        "slope": slope,
        "intercept_n": intercept,
    }


def _curve_metrics(
    frame: pd.DataFrame,
    estimate_column: str,
) -> dict[str, Any]:
    selected = frame[
        np.isfinite(frame["reference_force_fz_n"])
        & np.isfinite(frame[estimate_column])
        & np.isfinite(frame["elapsed_time_sec"])
    ]
    if selected.empty:
        return {}
    try:
        return force_curve_metrics(
            selected["reference_force_fz_n"].to_numpy(dtype=float),
            selected[estimate_column].to_numpy(dtype=float),
            selected["elapsed_time_sec"].to_numpy(dtype=float),
            release_grace_sec=0.75,
        )
    except ValueError:
        return _basic_force_metrics(
            selected["reference_force_fz_n"].to_numpy(dtype=float),
            selected[estimate_column].to_numpy(dtype=float),
        )


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _confusion(
    session_table: pd.DataFrame,
    predicted_column: str,
) -> tuple[np.ndarray, list[str]]:
    labels = [
        label
        for label in POSITION_ORDER
        if label in set(session_table["expected_position"])
        or label in set(session_table[predicted_column])
    ]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: offset for offset, label in enumerate(labels)}
    for row in session_table.itertuples(index=False):
        matrix[index[row.expected_position], index[getattr(row, predicted_column)]] += 1
    return matrix, labels


def _draw_confusion(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Expected")
    axis.set_title(title, loc="left", fontweight="bold")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(matrix[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() / 2 else "#17324d",
                    fontweight="bold",
                )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _position_figure(session_table: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    offline_matrix, labels = _confusion(session_table, "offline_predicted_position")
    online_matrix, online_labels = _confusion(session_table, "online_predicted_position")
    _draw_confusion(axes[0, 0], offline_matrix, labels, "Complete offline runtime replay")
    _draw_confusion(axes[0, 1], online_matrix, online_labels, "Recorded online predictions")

    active = session_table[session_table["expected_position"] != "none"].copy()
    x = np.arange(len(active))
    axes[1, 0].bar(
        x - 0.2,
        active["offline_active_correct_rate"],
        width=0.4,
        color="#235789",
        label="Offline replay",
    )
    axes[1, 0].bar(
        x + 0.2,
        active["online_active_correct_rate_when_available"],
        width=0.4,
        color="#f06449",
        label="Online, when saved",
    )
    axes[1, 0].set_xticks(x, active["session_order"].astype(int), rotation=0)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("Blind session order")
    axes[1, 0].set_ylabel("Correct active-frame fraction")
    axes[1, 0].set_title("Frame-level location stability", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", color="#dce6ef", linewidth=0.8)

    axes[1, 1].bar(
        x - 0.2,
        active["offline_active_visual_coverage"],
        width=0.4,
        color="#2a9d8f",
        label="Offline display coverage",
    )
    axes[1, 1].bar(
        x + 0.2,
        active["online_active_prediction_coverage"],
        width=0.4,
        color="#e9c46a",
        label="Online prediction saved",
    )
    axes[1, 1].set_xticks(x, active["session_order"].astype(int), rotation=0)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("Blind session order")
    axes[1, 1].set_ylabel("Fraction of PX6D-active frames")
    axes[1, 1].set_title("Visibility and Record inference coverage", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", color="#dce6ef", linewidth=0.8)

    figure.suptitle(
        "Blind2 position audit: answer, online Record, and complete replay",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _force_trace_figure(frame_table: pd.DataFrame, session_table: pd.DataFrame, output_path: Path) -> None:
    active_sessions = session_table[session_table["expected_position"] != "none"]
    figure, axes = plt.subplots(6, 3, figsize=(17, 19), sharey=True)
    for axis, session in zip(axes.flat, active_sessions.itertuples(index=False), strict=True):
        frame = frame_table[frame_table["session_id"] == session.session_id]
        axis.plot(
            frame["elapsed_time_sec"],
            frame["reference_force_fz_n"],
            color="#173f67",
            linewidth=2.0,
            label="PX6D Fz",
        )
        axis.plot(
            frame["elapsed_time_sec"],
            frame["offline_optical_force_n"],
            color="#e76f51",
            linewidth=1.5,
            label="Offline optical",
        )
        online = frame[frame["online_prediction_available"]]
        axis.scatter(
            online["elapsed_time_sec"],
            online["online_display_optical_force_n"],
            color="#21a6b5",
            edgecolor="white",
            linewidth=0.25,
            s=14,
            zorder=3,
            label="Recorded online",
        )
        correct = session.offline_predicted_position == session.expected_position
        title_color = "#1f6f54" if correct else "#a23b3b"
        axis.set_title(
            f"#{int(session.session_order):02d} {session.expected_position} | "
            f"offline {session.offline_predicted_position} | online {session.online_predicted_position}",
            loc="left",
            fontsize=10,
            fontweight="bold",
            color=title_color,
        )
        axis.grid(color="#e1e9f0", linewidth=0.7)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.977),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Blind2 synchronized force traces",
        y=0.997,
        fontsize=18,
        fontweight="bold",
        color="#17324d",
    )
    figure.subplots_adjust(top=0.945, bottom=0.04, hspace=0.48, wspace=0.20)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _state_timing_figure(session_table: pd.DataFrame, output_path: Path) -> None:
    active = session_table[session_table["expected_position"] != "none"].copy()
    idle = session_table[session_table["expected_position"] == "none"].copy()
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    x_idle = np.arange(len(idle))
    axes[0, 0].bar(
        x_idle - 0.2,
        idle["offline_idle_false_activation_rate"],
        width=0.4,
        color="#235789",
        label="Offline replay",
    )
    axes[0, 0].bar(
        x_idle + 0.2,
        idle["online_idle_false_activation_rate_when_available"],
        width=0.4,
        color="#f06449",
        label="Online, when saved",
    )
    axes[0, 0].set_xticks(x_idle, idle["session_order"].astype(int))
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_xlabel("Idle session order")
    axes[0, 0].set_ylabel("False-active fraction")
    axes[0, 0].set_title("Idle false activation", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    x_active = np.arange(len(active))
    axes[0, 1].bar(
        x_active - 0.2,
        active["offline_contact_onset_offset_ms"],
        width=0.4,
        color="#2a9d8f",
        label="Offline replay",
    )
    axes[0, 1].bar(
        x_active + 0.2,
        active["online_contact_onset_offset_ms"],
        width=0.4,
        color="#e9c46a",
        label="Online, saved frames",
    )
    axes[0, 1].axhline(0, color="#657786", linewidth=1)
    axes[0, 1].set_xticks(x_active, active["session_order"].astype(int))
    axes[0, 1].set_xlabel("Active session order")
    axes[0, 1].set_ylabel("Detected onset - PX6D onset (ms)")
    axes[0, 1].set_title("Contact onset timing", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].bar(
        x_active - 0.2,
        active["offline_contact_release_offset_ms"],
        width=0.4,
        color="#235789",
        label="Offline replay",
    )
    axes[1, 0].bar(
        x_active + 0.2,
        active["online_contact_release_offset_ms"],
        width=0.4,
        color="#f06449",
        label="Online, saved frames",
    )
    axes[1, 0].axhline(0, color="#657786", linewidth=1)
    axes[1, 0].set_xticks(x_active, active["session_order"].astype(int))
    axes[1, 0].set_xlabel("Active session order")
    axes[1, 0].set_ylabel("Last detected - PX6D release (ms)")
    axes[1, 0].set_title("Contact release timing", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        x_active - 0.2,
        active["offline_active_position_transitions"],
        width=0.4,
        color="#2a9d8f",
        label="Offline replay",
    )
    axes[1, 1].bar(
        x_active + 0.2,
        active["online_active_position_transitions"],
        width=0.4,
        color="#e9c46a",
        label="Online, saved frames",
    )
    axes[1, 1].set_xticks(x_active, active["session_order"].astype(int))
    axes[1, 1].set_xlabel("Active session order")
    axes[1, 1].set_ylabel("Label transitions")
    axes[1, 1].set_title("Within-contact position stability", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", color="#e1e9f0", linewidth=0.7)
    figure.suptitle(
        "Blind2 runtime state and timing audit",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _record_diagnostics_figure(session_table: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(session_table))
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].bar(x, session_table["capture_rate_hz"], color="#235789")
    axes[0, 0].set_ylabel("Captured frames/s")
    axes[0, 0].set_title("Record capture cadence", loc="left", fontweight="bold")

    axes[0, 1].bar(x, session_table["online_prediction_coverage"], color="#e9c46a")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Saved prediction fraction")
    axes[0, 1].set_title("Online inference retention", loc="left", fontweight="bold")

    axes[1, 0].bar(
        x - 0.2,
        session_table["online_inference_latency_median_ms"],
        width=0.4,
        color="#2a9d8f",
        label="Median",
    )
    axes[1, 0].bar(
        x + 0.2,
        session_table["online_inference_latency_p95_ms"],
        width=0.4,
        color="#e76f51",
        label="p95",
    )
    axes[1, 0].set_ylabel("Inference latency (ms)")
    axes[1, 0].set_title("Same-frame inference latency", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        x,
        session_table["maximum_absolute_force_sync_offset_ms"],
        color="#8e5ea2",
    )
    axes[1, 1].set_ylabel("Maximum |offset| (ms)")
    axes[1, 1].set_title("PX6D synchronization bound", loc="left", fontweight="bold")
    for axis in axes.flat:
        axis.set_xticks(x, session_table["session_order"].astype(int))
        axis.set_xlabel("Blind session order")
        axis.grid(axis="y", color="#e1e9f0", linewidth=0.7)
    figure.suptitle(
        "Blind2 Record acquisition diagnostics",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _force_summary_figure(frame_table: pd.DataFrame, session_table: pd.DataFrame, output_path: Path) -> None:
    active_ids = set(
        session_table.loc[
            session_table["expected_position"] != "none", "session_id"
        ]
    )
    active = frame_table[frame_table["session_id"].isin(active_ids)].copy()
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    axes[0].scatter(
        active["reference_force_fz_n"],
        active["offline_optical_force_n"],
        s=8,
        alpha=0.28,
        color="#2a9d8f",
        label="Offline replay",
    )
    online = active[active["online_prediction_available"]]
    axes[0].scatter(
        online["reference_force_fz_n"],
        online["online_display_optical_force_n"],
        s=10,
        alpha=0.45,
        color="#e76f51",
        label="Recorded online",
    )
    maximum = float(
        np.nanmax(
            [
                active["reference_force_fz_n"].max(),
                active["offline_optical_force_n"].max(),
                online["online_display_optical_force_n"].max(),
            ]
        )
    )
    axes[0].plot([0, maximum], [0, maximum], "--", color="#657786", linewidth=1)
    axes[0].set_xlim(0, maximum * 1.03)
    axes[0].set_ylim(0, maximum * 1.03)
    axes[0].set_xlabel("PX6D Fz (N)")
    axes[0].set_ylabel("Optical estimate (N)")
    axes[0].set_title("Frame-level force agreement", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    x = np.arange(len(session_table[session_table["expected_position"] != "none"]))
    sessions = session_table[session_table["expected_position"] != "none"]
    axes[1].bar(x - 0.2, sessions["offline_force_mae_n"], width=0.4, color="#235789", label="Offline")
    axes[1].bar(x + 0.2, sessions["online_force_mae_n"], width=0.4, color="#f06449", label="Online")
    axes[1].set_xticks(x, sessions["session_order"].astype(int))
    axes[1].set_xlabel("Blind session order")
    axes[1].set_ylabel("Force MAE (N)")
    axes[1].set_title("Per-session force error", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    axes[2].bar(x - 0.2, sessions["offline_force_pearson_r"], width=0.4, color="#2a9d8f", label="Offline")
    axes[2].bar(x + 0.2, sessions["online_force_pearson_r"], width=0.4, color="#e9c46a", label="Online")
    axes[2].axhline(0.85, color="#a23b3b", linestyle="--", linewidth=1, label="r = 0.85")
    axes[2].set_xticks(x, sessions["session_order"].astype(int))
    axes[2].set_ylim(-1.0, 1.05)
    axes[2].set_xlabel("Blind session order")
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title("Per-session force trend", loc="left", fontweight="bold")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(color="#e1e9f0", linewidth=0.7)
    figure.suptitle(
        "Blind2 optical force vs PX6D reference",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    blind_root = args.blind_root.resolve()
    prediction_dir = args.prediction_dir.resolve()
    score_dir = args.score_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output_dir}")

    prediction_manifest_path = prediction_dir / "prediction_manifest.json"
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    frozen_frames_path = prediction_dir / "frames.csv"
    expected_frame_hash = str(
        ((prediction_manifest.get("outputs") or {}).get("frames_sha256") or "")
    )
    if not expected_frame_hash or _sha256(frozen_frames_path) != expected_frame_hash:
        raise RuntimeError("frozen frame table hash mismatch")

    score_summary_path = score_dir / "evaluation_summary.json"
    score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
    scored = pd.read_csv(score_dir / "scored_predictions.csv", encoding="utf-8-sig")
    frozen = pd.read_csv(frozen_frames_path, encoding="utf-8-sig")
    frozen["active_reference_frame"] = frozen["active_reference_frame"].map(_truthy)
    frozen["visual_active"] = frozen["visual_active"].map(_truthy)
    for column in (
        "reference_force_fz_n",
        "estimated_force_fz_n",
        "elapsed_time_sec",
        "visual_position_confidence",
        "visual_position_margin",
    ):
        frozen[column] = pd.to_numeric(frozen[column], errors="coerce")

    all_frames: list[pd.DataFrame] = []
    session_rows: list[dict[str, Any]] = []
    for score in scored.sort_values("session_order").itertuples(index=False):
        session_dir = blind_root / score.session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(session_dir)
        online = pd.read_csv(session_dir / "frame_summary.csv", encoding="utf-8-sig")
        offline = frozen[frozen["session_id"] == score.session_id].copy()
        if offline.empty:
            raise RuntimeError(f"missing frozen frames for {score.session_id}")
        if online["capture_index"].duplicated().any() or offline["capture_index"].duplicated().any():
            raise RuntimeError(f"duplicate capture index in {score.session_id}")

        online_columns = [
            "capture_index",
            "elapsed_time_sec",
            "force_fz_n",
            "sync_offset_ms",
            "model_ready",
            "capture_response_source",
            "capture_response_frame_match",
            "display_contact_active",
            "display_position_label",
            "display_position_confidence",
            "display_position_margin",
            "display_optical_force_n",
            "formal_position_label",
            "raw_position_label",
            "optical_estimated_fz_n",
            "model_inference_latency_ms",
        ]
        missing = sorted(set(online_columns) - set(online.columns))
        if missing:
            raise ValueError(f"{score.session_id} lacks Record fields: {missing}")
        online = online[online_columns].copy()
        online = online.rename(
            columns={
                "elapsed_time_sec": "record_elapsed_time_sec",
                "force_fz_n": "record_reference_force_fz_n",
                "model_ready": "online_model_ready",
                "display_contact_active": "online_display_contact_active",
                "display_position_label": "online_display_position_label",
                "display_position_confidence": "online_display_position_confidence",
                "display_position_margin": "online_display_position_margin",
                "display_optical_force_n": "online_display_optical_force_n",
                "formal_position_label": "online_formal_position_label",
                "raw_position_label": "online_raw_position_label",
                "optical_estimated_fz_n": "online_optical_estimated_fz_n",
                "model_inference_latency_ms": "online_model_inference_latency_ms",
            }
        )
        frame = offline.merge(online, on="capture_index", how="inner", validate="one_to_one")
        if len(frame) != len(offline) or len(frame) != len(online):
            raise RuntimeError(f"incomplete frame join for {score.session_id}")
        force_difference = np.abs(
            frame["reference_force_fz_n"].to_numpy(dtype=float)
            - pd.to_numeric(frame["record_reference_force_fz_n"], errors="coerce").to_numpy(dtype=float)
        )
        if np.nanmax(force_difference) > 1.0e-9:
            raise RuntimeError(f"reference force mismatch in {score.session_id}")

        frame["expected_position"] = str(score.expected_position)
        frame["session_order"] = int(score.session_order)
        frame["online_prediction_available"] = frame["online_model_ready"].map(_truthy)
        frame["online_display_active"] = (
            frame["online_prediction_available"]
            & frame["online_display_contact_active"].map(_truthy)
        )
        frame["offline_display_position"] = [
            _label(value) if active else "none"
            for value, active in zip(frame["visual_position"], frame["visual_active"], strict=True)
        ]
        frame["online_display_position"] = [
            _label(value) if active else "none"
            for value, active in zip(
                frame["online_display_position_label"],
                frame["online_display_active"],
                strict=True,
            )
        ]
        frame["online_raw_position_norm"] = [
            _label(value) if available else "none"
            for value, available in zip(
                frame["online_raw_position_label"],
                frame["online_prediction_available"],
                strict=True,
            )
        ]
        frame["online_formal_position_norm"] = [
            _label(value) if available else "none"
            for value, available in zip(
                frame["online_formal_position_label"],
                frame["online_prediction_available"],
                strict=True,
            )
        ]
        frame["offline_optical_force_n"] = pd.to_numeric(
            frame["estimated_force_fz_n"], errors="coerce"
        )
        frame["online_display_optical_force_n"] = pd.to_numeric(
            frame["online_display_optical_force_n"], errors="coerce"
        )
        frame["online_offline_display_agree"] = np.where(
            frame["online_prediction_available"],
            frame["online_display_position"] == frame["offline_display_position"],
            False,
        )
        frame["offline_answer_correct"] = (
            frame["visual_active"]
            & (frame["offline_display_position"] == frame["expected_position"])
        )
        frame["online_answer_correct"] = (
            frame["online_display_active"]
            & (frame["online_display_position"] == frame["expected_position"])
        )
        frame["online_force_error_n"] = (
            frame["online_display_optical_force_n"] - frame["reference_force_fz_n"]
        )
        frame["offline_force_error_n"] = (
            frame["offline_optical_force_n"] - frame["reference_force_fz_n"]
        )
        metadata = _load_metadata(session_dir / "session_metadata.json")
        start_provenance = dict(
            ((metadata.get("provenance") or {}).get("start") or {})
        )
        software_provenance = dict(start_provenance.get("software") or {})
        model_provenance = dict(start_provenance.get("model") or {})
        baseline_provenance = dict(start_provenance.get("baseline") or {})
        baseline_attestation = dict(
            baseline_provenance.get("operator_attestation") or {}
        )
        baseline_force_evidence = dict(
            baseline_attestation.get("force_evidence") or {}
        )
        baseline_attested_at = baseline_attestation.get("attested_at_epoch_sec")
        baseline_age_sec = None
        if baseline_attested_at is not None and metadata.get("started_at_epoch_sec") is not None:
            baseline_age_sec = float(metadata["started_at_epoch_sec"]) - float(
                baseline_attested_at
            )

        active_mask = frame["active_reference_frame"]
        offline_votes = frame.loc[active_mask & frame["visual_active"], "offline_display_position"]
        online_votes = frame.loc[
            active_mask & frame["online_display_active"], "online_display_position"
        ]
        if score.expected_position == "none":
            offline_votes = frame.loc[frame["visual_active"], "offline_display_position"]
            online_votes = frame.loc[frame["online_display_active"], "online_display_position"]
        offline_majority = _majority(offline_votes)
        online_majority = _majority(online_votes)

        offline_force = _curve_metrics(frame, "offline_optical_force_n")
        online_force = _curve_metrics(
            frame[frame["online_prediction_available"]],
            "online_display_optical_force_n",
        )
        active_count = int(active_mask.sum())
        online_active_available = int(
            (active_mask & frame["online_prediction_available"]).sum()
        )
        online_active_display = int((active_mask & frame["online_display_active"]).sum())
        offline_active_display = int((active_mask & frame["visual_active"]).sum())
        comparable = frame["online_prediction_available"]
        offline_correct_count = int(
            (active_mask & frame["offline_answer_correct"]).sum()
        )
        online_correct_count = int(
            (active_mask & frame["online_answer_correct"]).sum()
        )
        online_available_active_mask = active_mask & frame["online_prediction_available"]
        idle_mask = ~active_mask
        offline_idle_false_rate = (
            float(frame.loc[idle_mask, "visual_active"].mean())
            if idle_mask.any()
            else None
        )
        online_idle_available_mask = idle_mask & frame["online_prediction_available"]
        online_idle_false_rate = (
            float(frame.loc[online_idle_available_mask, "online_display_active"].mean())
            if online_idle_available_mask.any()
            else None
        )

        reference_onset = _edge_time(frame, active_mask, edge="first")
        reference_release = _edge_time(frame, active_mask, edge="last")
        offline_onset = _edge_time(frame, frame["visual_active"], edge="first")
        offline_release = _edge_time(frame, frame["visual_active"], edge="last")
        online_onset = _edge_time(
            frame,
            frame["online_prediction_available"] & frame["online_display_active"],
            edge="first",
        )
        online_release = _edge_time(
            frame,
            frame["online_prediction_available"] & frame["online_display_active"],
            edge="last",
        )
        elapsed = pd.to_numeric(frame["elapsed_time_sec"], errors="coerce")
        if reference_onset is None:
            precontact_mask = pd.Series(False, index=frame.index)
        else:
            precontact_mask = elapsed < reference_onset
        if reference_release is None:
            postrelease_mask = pd.Series(False, index=frame.index)
        else:
            postrelease_mask = elapsed > reference_release + 0.75
        offline_expected_onset = _edge_time(
            frame,
            frame["visual_active"]
            & (frame["offline_display_position"] == score.expected_position),
            edge="first",
        )
        online_expected_onset = _edge_time(
            frame,
            frame["online_prediction_available"]
            & frame["online_display_active"]
            & (frame["online_display_position"] == score.expected_position),
            edge="first",
        )

        offline_active_labels = frame.loc[active_mask, "offline_display_position"]
        online_active_labels = frame.loc[
            online_available_active_mask, "online_display_position"
        ]
        offline_raw_majority = _majority(frame.loc[active_mask, "raw_position"])
        offline_formal_majority = _majority(frame.loc[active_mask, "formal_position"])
        online_raw_majority = _majority(
            frame.loc[online_available_active_mask, "online_raw_position_norm"]
        )
        online_formal_majority = _majority(
            frame.loc[online_available_active_mask, "online_formal_position_norm"]
        )
        capture_sources = Counter(str(value) for value in frame["capture_response_source"])
        session_rows.append(
            {
                "session_order": int(score.session_order),
                "session_id": score.session_id,
                "expected_position": score.expected_position,
                "offline_predicted_position": offline_majority[0],
                "offline_session_correct": offline_majority[0] == score.expected_position,
                "online_predicted_position": online_majority[0],
                "online_session_correct": online_majority[0] == score.expected_position,
                "total_frames": int(len(frame)),
                "active_reference_frames": active_count,
                "offline_active_visual_frames": offline_active_display,
                "offline_active_visual_coverage": _safe_ratio(offline_active_display, active_count),
                "offline_active_correct_frames": offline_correct_count,
                "offline_active_correct_rate": _safe_ratio(
                    offline_correct_count, active_count
                ),
                "offline_raw_majority_position": offline_raw_majority[0],
                "offline_formal_majority_position": offline_formal_majority[0],
                "offline_raw_active_correct_rate": _safe_ratio(
                    int((active_mask & (frame["raw_position"] == score.expected_position)).sum()),
                    active_count,
                ),
                "offline_formal_active_correct_rate": _safe_ratio(
                    int((active_mask & (frame["formal_position"] == score.expected_position)).sum()),
                    active_count,
                ),
                "online_prediction_frames": int(frame["online_prediction_available"].sum()),
                "online_prediction_coverage": float(frame["online_prediction_available"].mean()),
                "online_active_prediction_frames": online_active_available,
                "online_active_prediction_coverage": _safe_ratio(online_active_available, active_count),
                "online_active_visual_frames": online_active_display,
                "online_active_visual_coverage": _safe_ratio(online_active_display, active_count),
                "online_active_visual_coverage_when_available": _safe_ratio(
                    online_active_display, online_active_available
                ),
                "online_active_correct_frames": online_correct_count,
                "online_active_correct_rate": _safe_ratio(
                    online_correct_count, active_count
                ),
                "online_active_correct_rate_when_available": _safe_ratio(
                    online_correct_count, online_active_available
                ),
                "online_raw_majority_position": online_raw_majority[0],
                "online_formal_majority_position": online_formal_majority[0],
                "online_raw_active_correct_rate_when_available": _safe_ratio(
                    int(
                        (
                            online_available_active_mask
                            & (frame["online_raw_position_norm"] == score.expected_position)
                        ).sum()
                    ),
                    online_active_available,
                ),
                "online_formal_active_correct_rate_when_available": _safe_ratio(
                    int(
                        (
                            online_available_active_mask
                            & (frame["online_formal_position_norm"] == score.expected_position)
                        ).sum()
                    ),
                    online_active_available,
                ),
                "online_offline_display_agreement": (
                    float(frame.loc[comparable, "online_offline_display_agree"].mean())
                    if comparable.any()
                    else None
                ),
                "deferred_response_frames": int(
                    (frame["capture_response_source"] == "deferred_for_high_rate_capture").sum()
                ),
                "same_frame_inference_frames": int(
                    frame["capture_response_source"].isin(
                        {"same_frame_runtime_inference", "same_frame_runtime_cache"}
                    ).sum()
                ),
                "capture_response_source_counts": json.dumps(
                    dict(capture_sources), sort_keys=True
                ),
                "online_inference_latency_median_ms": _finite_percentile(
                    frame.loc[comparable, "online_model_inference_latency_ms"], 50
                ),
                "online_inference_latency_p95_ms": _finite_percentile(
                    frame.loc[comparable, "online_model_inference_latency_ms"], 95
                ),
                "capture_rate_hz": metadata.get("captured_frame_rate_hz"),
                "maximum_absolute_force_sync_offset_ms": metadata.get(
                    "maximum_absolute_sync_offset_ms"
                ),
                "alignment_status": (
                    (metadata.get("alignment_audit") or {}).get("status")
                ),
                "record_software_version": software_provenance.get("version"),
                "record_build_id": software_provenance.get("build_id"),
                "record_model_sha256": model_provenance.get("model_bundle_sha256"),
                "record_model_loaded": model_provenance.get("loaded"),
                "record_baseline_token": baseline_provenance.get("token"),
                "record_baseline_status": baseline_provenance.get("status"),
                "record_baseline_age_sec": baseline_age_sec,
                "record_baseline_attested_force_fz_n": baseline_force_evidence.get(
                    "force_fz_n"
                ),
                "record_baseline_force_tare_status": baseline_force_evidence.get(
                    "tare_status"
                ),
                "reference_force_mean_n": _finite_percentile(
                    frame.loc[active_mask, "reference_force_fz_n"], 50
                ),
                "reference_force_max_n": _finite_percentile(
                    frame["reference_force_fz_n"], 100
                ),
                "baseline_distance_median": _finite_percentile(
                    frame["baseline_distance"], 50
                ),
                "baseline_distance_p95": _finite_percentile(
                    frame["baseline_distance"], 95
                ),
                "offline_idle_false_activation_rate": offline_idle_false_rate,
                "online_idle_false_activation_rate_when_available": online_idle_false_rate,
                "offline_idle_position_counts": json.dumps(
                    dict(
                        Counter(
                            frame.loc[
                                idle_mask & frame["visual_active"],
                                "offline_display_position",
                            ]
                        )
                    ),
                    sort_keys=True,
                ),
                "online_idle_position_counts": json.dumps(
                    dict(
                        Counter(
                            frame.loc[
                                online_idle_available_mask
                                & frame["online_display_active"],
                                "online_display_position",
                            ]
                        )
                    ),
                    sort_keys=True,
                ),
                "offline_precontact_false_activation_rate": (
                    float(frame.loc[precontact_mask, "visual_active"].mean())
                    if precontact_mask.any()
                    else None
                ),
                "online_precontact_false_activation_rate_when_available": (
                    float(
                        frame.loc[
                            precontact_mask & frame["online_prediction_available"],
                            "online_display_active",
                        ].mean()
                    )
                    if (precontact_mask & frame["online_prediction_available"]).any()
                    else None
                ),
                "offline_postrelease_false_activation_rate": (
                    float(frame.loc[postrelease_mask, "visual_active"].mean())
                    if postrelease_mask.any()
                    else None
                ),
                "online_postrelease_false_activation_rate_when_available": (
                    float(
                        frame.loc[
                            postrelease_mask & frame["online_prediction_available"],
                            "online_display_active",
                        ].mean()
                    )
                    if (postrelease_mask & frame["online_prediction_available"]).any()
                    else None
                ),
                "reference_contact_onset_sec": reference_onset,
                "reference_contact_release_sec": reference_release,
                "offline_contact_onset_sec": offline_onset,
                "offline_contact_release_sec": offline_release,
                "online_contact_onset_sec": online_onset,
                "online_contact_release_sec": online_release,
                "offline_expected_position_onset_sec": offline_expected_onset,
                "online_expected_position_onset_sec": online_expected_onset,
                "offline_contact_onset_offset_ms": _time_offset_ms(
                    offline_onset, reference_onset
                ),
                "offline_contact_release_offset_ms": _time_offset_ms(
                    offline_release, reference_release
                ),
                "online_contact_onset_offset_ms": _time_offset_ms(
                    online_onset, reference_onset
                ),
                "online_contact_release_offset_ms": _time_offset_ms(
                    online_release, reference_release
                ),
                "offline_expected_position_onset_offset_ms": _time_offset_ms(
                    offline_expected_onset, reference_onset
                ),
                "online_expected_position_onset_offset_ms": _time_offset_ms(
                    online_expected_onset, reference_onset
                ),
                "offline_first_frame_active": bool(frame["visual_active"].iloc[0]),
                "offline_first_frame_position": frame["offline_display_position"].iloc[0],
                "online_first_saved_active": (
                    bool(frame.loc[comparable, "online_display_active"].iloc[0])
                    if comparable.any()
                    else False
                ),
                "online_first_saved_position": (
                    frame.loc[comparable, "online_display_position"].iloc[0]
                    if comparable.any()
                    else "none"
                ),
                "offline_active_position_transitions": _transition_count(
                    offline_active_labels
                ),
                "online_active_position_transitions": _transition_count(
                    online_active_labels
                ),
                "offline_active_position_counts": json.dumps(
                    dict(Counter(offline_active_labels)), sort_keys=True
                ),
                "online_active_position_counts": json.dumps(
                    dict(Counter(online_active_labels)), sort_keys=True
                ),
                "offline_force_mae_n": offline_force.get("mae_n"),
                "offline_force_rmse_n": offline_force.get("rmse_n"),
                "offline_force_pearson_r": offline_force.get("pearson_r"),
                "offline_force_slope": offline_force.get("linear_slope_pred_vs_px6d"),
                "offline_force_lag_ms": offline_force.get("lag_ms"),
                "offline_release_false_response_rate": offline_force.get(
                    "zero_force_false_response_rate_after_grace"
                ),
                "online_force_mae_n": online_force.get("mae_n"),
                "online_force_rmse_n": online_force.get("rmse_n"),
                "online_force_pearson_r": online_force.get("pearson_r"),
                "online_force_slope": online_force.get("linear_slope_pred_vs_px6d"),
                "online_force_lag_ms": online_force.get("lag_ms"),
                "online_release_false_response_rate": online_force.get(
                    "zero_force_false_response_rate_after_grace"
                ),
            }
        )
        all_frames.append(frame)

    frame_table = pd.concat(all_frames, ignore_index=True)
    session_table = pd.DataFrame(session_rows).sort_values("session_order")
    active_sessions = session_table[session_table["expected_position"] != "none"]
    idle_sessions = session_table[session_table["expected_position"] == "none"]
    active_frames = frame_table[
        frame_table["session_id"].isin(active_sessions["session_id"])
    ]
    online_force_frames = active_frames[active_frames["online_prediction_available"]]
    position_summary_rows: list[dict[str, Any]] = []
    for position in POSITION_ORDER[1:]:
        position_sessions = active_sessions[
            active_sessions["expected_position"] == position
        ]
        if position_sessions.empty:
            continue
        position_frames = frame_table[
            frame_table["session_id"].isin(position_sessions["session_id"])
        ]
        position_active = position_frames["active_reference_frame"]
        position_online = (
            position_active & position_frames["online_prediction_available"]
        )
        offline_force_metrics = _basic_force_metrics(
            position_frames["reference_force_fz_n"].to_numpy(dtype=float),
            position_frames["offline_optical_force_n"].to_numpy(dtype=float),
        )
        online_force_metrics = _basic_force_metrics(
            position_frames.loc[position_online, "reference_force_fz_n"].to_numpy(
                dtype=float
            ),
            position_frames.loc[
                position_online, "online_display_optical_force_n"
            ].to_numpy(dtype=float),
        )
        position_summary_rows.append(
            {
                "expected_position": position,
                "session_count": int(len(position_sessions)),
                "offline_session_accuracy": float(
                    position_sessions["offline_session_correct"].mean()
                ),
                "online_session_accuracy": float(
                    position_sessions["online_session_correct"].mean()
                ),
                "offline_active_frame_accuracy": _safe_ratio(
                    int(
                        position_frames.loc[
                            position_active, "offline_answer_correct"
                        ].sum()
                    ),
                    int(position_active.sum()),
                ),
                "online_active_frame_accuracy_when_available": _safe_ratio(
                    int(
                        position_frames.loc[
                            position_online, "online_answer_correct"
                        ].sum()
                    ),
                    int(position_online.sum()),
                ),
                "online_active_prediction_coverage": _safe_ratio(
                    int(position_online.sum()), int(position_active.sum())
                ),
                "offline_force_mae_n": offline_force_metrics.get("mae_n"),
                "offline_force_pearson_r": offline_force_metrics.get("pearson_r"),
                "offline_force_slope": offline_force_metrics.get("slope"),
                "online_force_mae_n": online_force_metrics.get("mae_n"),
                "online_force_pearson_r": online_force_metrics.get("pearson_r"),
                "online_force_slope": online_force_metrics.get("slope"),
            }
        )
    position_summary_table = pd.DataFrame(position_summary_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    frame_output = output_dir / "frame_comparison.csv"
    session_output = output_dir / "session_comparison.csv"
    frame_table.to_csv(frame_output, index=False, encoding="utf-8-sig")
    session_table.to_csv(session_output, index=False, encoding="utf-8-sig")

    idle_output = output_dir / "idle_session_diagnostics.csv"
    active_output = output_dir / "active_session_diagnostics.csv"
    disagreement_output = output_dir / "online_offline_disagreement_frames.csv"
    source_output = output_dir / "record_response_source_summary.csv"
    position_summary_output = output_dir / "per_position_summary.csv"
    idle_sessions.to_csv(idle_output, index=False, encoding="utf-8-sig")
    active_sessions.to_csv(active_output, index=False, encoding="utf-8-sig")
    disagreement_columns = [
        "session_order",
        "session_id",
        "capture_index",
        "elapsed_time_sec",
        "reference_force_fz_n",
        "expected_position",
        "offline_display_position",
        "online_display_position",
        "raw_position",
        "formal_position",
        "online_raw_position_norm",
        "online_formal_position_norm",
        "contact_probability",
        "baseline_distance",
        "visual_position_confidence",
        "visual_position_margin",
        "online_display_position_confidence",
        "online_display_position_margin",
        "offline_optical_force_n",
        "online_display_optical_force_n",
        "capture_response_source",
        "online_model_inference_latency_ms",
    ]
    disagreement = frame_table[
        frame_table["online_prediction_available"]
        & ~frame_table["online_offline_display_agree"]
    ][disagreement_columns]
    disagreement.to_csv(disagreement_output, index=False, encoding="utf-8-sig")
    source_summary = (
        frame_table.groupby("capture_response_source", dropna=False)
        .agg(
            frame_count=("capture_index", "size"),
            session_count=("session_id", "nunique"),
            model_ready_frames=("online_prediction_available", "sum"),
            median_inference_latency_ms=("online_model_inference_latency_ms", "median"),
            p95_inference_latency_ms=(
                "online_model_inference_latency_ms",
                lambda values: _finite_percentile(values, 95),
            ),
        )
        .reset_index()
    )
    source_summary["frame_share"] = source_summary["frame_count"] / len(frame_table)
    source_summary.to_csv(source_output, index=False, encoding="utf-8-sig")
    position_summary_table.to_csv(
        position_summary_output, index=False, encoding="utf-8-sig"
    )

    position_figure = output_dir / "position_comparison.png"
    force_summary_figure = output_dir / "force_summary.png"
    force_trace_figure = output_dir / "force_trace_sheet.png"
    state_timing_figure = output_dir / "state_timing_comparison.png"
    record_diagnostics_figure = output_dir / "record_capture_diagnostics.png"
    _position_figure(session_table, position_figure)
    _force_summary_figure(frame_table, session_table, force_summary_figure)
    _force_trace_figure(frame_table, session_table, force_trace_figure)
    _state_timing_figure(session_table, state_timing_figure)
    _record_diagnostics_figure(session_table, record_diagnostics_figure)

    offline_global_force = _basic_force_metrics(
        active_frames["reference_force_fz_n"].to_numpy(dtype=float),
        active_frames["offline_optical_force_n"].to_numpy(dtype=float),
    )
    online_global_force = _basic_force_metrics(
        online_force_frames["reference_force_fz_n"].to_numpy(dtype=float),
        online_force_frames["online_display_optical_force_n"].to_numpy(dtype=float),
    )
    online_available = frame_table["online_prediction_available"]
    active_reference = frame_table["active_reference_frame"]
    online_active_available = online_available & active_reference
    idle_session_mask = frame_table["session_id"].isin(idle_sessions["session_id"])
    online_idle_available = online_available & idle_session_mask
    offline_active_raw_correct = active_reference & (
        frame_table["raw_position"] == frame_table["expected_position"]
    )
    offline_active_formal_correct = active_reference & (
        frame_table["formal_position"] == frame_table["expected_position"]
    )
    online_active_raw_correct = online_active_available & (
        frame_table["online_raw_position_norm"] == frame_table["expected_position"]
    )
    online_active_formal_correct = online_active_available & (
        frame_table["online_formal_position_norm"] == frame_table["expected_position"]
    )
    baseline_tokens = sorted(
        {
            str(value)
            for value in session_table["record_baseline_token"].dropna()
            if str(value)
        }
    )
    record_model_hashes = sorted(
        {
            str(value)
            for value in session_table["record_model_sha256"].dropna()
            if str(value)
        }
    )
    summary = {
        "schema_version": "touch_blind_record_comparison_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "blind_root": str(blind_root),
        "frozen_prediction_manifest": str(prediction_manifest_path),
        "frozen_prediction_manifest_sha256": _sha256(prediction_manifest_path),
        "score_summary": str(score_summary_path),
        "score_summary_sha256": _sha256(score_summary_path),
        "answer_integrity": {
            "ordered_fallback_count": score_summary.get(
                "ordered_answer_fallback_count", 0
            ),
            "formal_evidence_status": (
                "provisional_due_to_answer_directory_mapping"
                if score_summary.get("ordered_answer_fallback_count", 0)
                else "strict_session_match"
            ),
        },
        "capture_integrity": {
            "session_count": int(len(session_table)),
            "frame_count": int(len(frame_table)),
            "all_sessions_aligned": bool(
                (session_table["alignment_status"] == "aligned").all()
            ),
            "capture_rate_hz_median": float(session_table["capture_rate_hz"].median()),
            "capture_rate_hz_min": float(session_table["capture_rate_hz"].min()),
            "capture_rate_hz_max": float(session_table["capture_rate_hz"].max()),
            "maximum_force_sync_offset_ms": float(
                session_table["maximum_absolute_force_sync_offset_ms"].max()
            ),
            "online_prediction_frames": int(online_available.sum()),
            "online_prediction_coverage": float(online_available.mean()),
            "same_frame_inference_frames": int(
                frame_table["capture_response_source"].isin(
                    {"same_frame_runtime_inference", "same_frame_runtime_cache"}
                ).sum()
            ),
            "deferred_response_frames": int(
                (frame_table["capture_response_source"] == "deferred_for_high_rate_capture").sum()
            ),
            "response_source_counts": dict(
                Counter(str(value) for value in frame_table["capture_response_source"])
            ),
            "online_inference_latency_median_ms": _finite_percentile(
                frame_table.loc[
                    online_available, "online_model_inference_latency_ms"
                ],
                50,
            ),
            "online_inference_latency_p95_ms": _finite_percentile(
                frame_table.loc[
                    online_available, "online_model_inference_latency_ms"
                ],
                95,
            ),
        },
        "record_provenance": {
            "software_versions": sorted(
                {
                    str(value)
                    for value in session_table["record_software_version"].dropna()
                    if str(value)
                }
            ),
            "model_sha256_values": record_model_hashes,
            "single_model_across_sessions": len(record_model_hashes) == 1,
            "baseline_tokens": baseline_tokens,
            "single_shared_baseline_across_sessions": len(baseline_tokens) == 1,
            "baseline_age_sec_min": _finite_percentile(
                session_table["record_baseline_age_sec"], 0
            ),
            "baseline_age_sec_max": _finite_percentile(
                session_table["record_baseline_age_sec"], 100
            ),
            "baseline_status_values": sorted(
                {
                    str(value)
                    for value in session_table["record_baseline_status"].dropna()
                    if str(value)
                }
            ),
            "offline_replay_baseline_policy": "per_session_median_first_5_frames",
        },
        "session_position": {
            "overall_count": int(len(session_table)),
            "offline_correct": int(session_table["offline_session_correct"].sum()),
            "offline_accuracy": float(session_table["offline_session_correct"].mean()),
            "online_correct": int(session_table["online_session_correct"].sum()),
            "online_accuracy": float(session_table["online_session_correct"].mean()),
            "active_count": int(len(active_sessions)),
            "offline_active_correct": int(active_sessions["offline_session_correct"].sum()),
            "offline_active_accuracy": float(active_sessions["offline_session_correct"].mean()),
            "online_active_correct": int(active_sessions["online_session_correct"].sum()),
            "online_active_accuracy": float(active_sessions["online_session_correct"].mean()),
            "idle_count": int(len(idle_sessions)),
            "offline_idle_correct": int(idle_sessions["offline_session_correct"].sum()),
            "online_idle_correct": int(idle_sessions["online_session_correct"].sum()),
        },
        "frame_position": {
            "active_reference_frames": int(frame_table["active_reference_frame"].sum()),
            "offline_active_visual_coverage": float(
                frame_table.loc[
                    frame_table["active_reference_frame"], "visual_active"
                ].mean()
            ),
            "offline_active_correct_rate": float(
                frame_table.loc[
                    frame_table["active_reference_frame"], "offline_answer_correct"
                ].mean()
            ),
            "online_active_prediction_coverage": float(
                frame_table.loc[
                    frame_table["active_reference_frame"], "online_prediction_available"
                ].mean()
            ),
            "online_active_correct_rate_all_reference_frames": float(
                frame_table.loc[
                    frame_table["active_reference_frame"], "online_answer_correct"
                ].mean()
            ),
            "online_active_correct_rate_when_prediction_available": _safe_ratio(
                int(frame_table.loc[online_active_available, "online_answer_correct"].sum()),
                int(online_active_available.sum()),
            ),
            "online_active_visual_coverage_when_prediction_available": _safe_ratio(
                int(frame_table.loc[online_active_available, "online_display_active"].sum()),
                int(online_active_available.sum()),
            ),
            "offline_raw_active_correct_rate": _safe_ratio(
                int(offline_active_raw_correct.sum()), int(active_reference.sum())
            ),
            "offline_formal_active_correct_rate": _safe_ratio(
                int(offline_active_formal_correct.sum()), int(active_reference.sum())
            ),
            "online_raw_active_correct_rate_when_prediction_available": _safe_ratio(
                int(online_active_raw_correct.sum()), int(online_active_available.sum())
            ),
            "online_formal_active_correct_rate_when_prediction_available": _safe_ratio(
                int(online_active_formal_correct.sum()), int(online_active_available.sum())
            ),
            "online_offline_display_agreement_when_available": float(
                frame_table.loc[
                    online_available, "online_offline_display_agree"
                ].mean()
            ),
            "offline_idle_false_activation_rate": float(
                frame_table.loc[
                    frame_table["session_id"].isin(idle_sessions["session_id"]),
                    "visual_active",
                ].mean()
            ),
            "online_idle_false_activation_rate_when_available": float(
                frame_table.loc[
                    online_idle_available,
                    "online_display_active",
                ].mean()
            ),
        },
        "state_timing": {
            "offline_contact_onset_offset_median_ms": _finite_percentile(
                active_sessions["offline_contact_onset_offset_ms"], 50
            ),
            "online_contact_onset_offset_median_ms": _finite_percentile(
                active_sessions["online_contact_onset_offset_ms"], 50
            ),
            "offline_contact_release_offset_median_ms": _finite_percentile(
                active_sessions["offline_contact_release_offset_ms"], 50
            ),
            "online_contact_release_offset_median_ms": _finite_percentile(
                active_sessions["online_contact_release_offset_ms"], 50
            ),
            "offline_active_position_transition_count": int(
                active_sessions["offline_active_position_transitions"].sum()
            ),
            "online_active_position_transition_count_saved_frames": int(
                active_sessions["online_active_position_transitions"].sum()
            ),
        },
        "force_comparison": {
            "scope": "all_frames_from_answered_active_position_sessions",
            "offline_complete_replay": offline_global_force,
            "online_recorded_available_frames": online_global_force,
            "force_sensor_used_as_runtime_input": False,
        },
        "outputs": {
            "frame_comparison_csv": frame_output.name,
            "frame_comparison_sha256": _sha256(frame_output),
            "session_comparison_csv": session_output.name,
            "session_comparison_sha256": _sha256(session_output),
            "active_session_diagnostics_csv": active_output.name,
            "idle_session_diagnostics_csv": idle_output.name,
            "per_position_summary_csv": position_summary_output.name,
            "record_response_source_summary_csv": source_output.name,
            "online_offline_disagreement_frames_csv": disagreement_output.name,
            "position_figure": position_figure.name,
            "force_summary_figure": force_summary_figure.name,
            "force_trace_figure": force_trace_figure.name,
            "state_timing_figure": state_timing_figure.name,
            "record_capture_diagnostics_figure": record_diagnostics_figure.name,
        },
    }
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = output_dir / "audit_report.md"
    wrong = session_table[~session_table["offline_session_correct"]]
    wrong_active = wrong[wrong["expected_position"] != "none"]
    wrong_idle = wrong[wrong["expected_position"] == "none"]
    report_path.write_text(
        "\n".join(
            [
                "# TOUCH Blind2 Record Comparison",
                "",
                f"- Sessions: {len(session_table)}; frames: {len(frame_table)}.",
                f"- Offline session accuracy: {summary['session_position']['offline_correct']}/{len(session_table)}.",
                f"- Online recorded session accuracy: {summary['session_position']['online_correct']}/{len(session_table)}.",
                f"- Online inference saved for {summary['capture_integrity']['online_prediction_coverage']:.1%} of captured frames; deferred rows remain explicitly missing.",
                f"- Online active-frame correctness is {summary['frame_position']['online_active_correct_rate_when_prediction_available']:.1%} when a same-frame prediction was saved, versus {summary['frame_position']['online_active_correct_rate_all_reference_frames']:.1%} if missing frames are included in the denominator.",
                f"- Offline/online idle false-active rates are {summary['frame_position']['offline_idle_false_activation_rate']:.1%} and {summary['frame_position']['online_idle_false_activation_rate_when_available']:.1%}; the online value is conditional on saved predictions.",
                f"- Record cadence is {summary['capture_integrity']['capture_rate_hz_median']:.2f} Hz median; inference latency is {summary['capture_integrity']['online_inference_latency_median_ms']:.1f} ms median and {summary['capture_integrity']['online_inference_latency_p95_ms']:.1f} ms p95.",
                f"- Record used {len(summary['record_provenance']['model_sha256_values'])} model hash and {len(summary['record_provenance']['baseline_tokens'])} shared baseline token across all sessions; baseline age increased from {summary['record_provenance']['baseline_age_sec_min']:.1f} s to {summary['record_provenance']['baseline_age_sec_max']:.1f} s.",
                f"- Offline optical force: MAE {offline_global_force['mae_n']:.3f} N, r={offline_global_force['pearson_r']:.3f}, slope={offline_global_force['slope']:.3f}.",
                f"- Online recorded optical force: MAE {online_global_force['mae_n']:.3f} N, r={online_global_force['pearson_r']:.3f}, slope={online_global_force['slope']:.3f}.",
                "",
                "## Incorrect active-position session predictions",
                "",
                *[
                    f"- #{int(row.session_order):02d}: expected {row.expected_position}, predicted {row.offline_predicted_position}; online {row.online_predicted_position}."
                    for row in wrong_active.itertuples(index=False)
                ],
                "",
                "## Idle false-activation sessions",
                "",
                *[
                    f"- #{int(row.session_order):02d}: offline majority {row.offline_predicted_position}, online majority {row.online_predicted_position}; offline false-active {row.offline_idle_false_activation_rate:.1%}, online conditional false-active {row.online_idle_false_activation_rate_when_available:.1%}."
                    for row in wrong_idle.itertuples(index=False)
                ],
                "",
                "## Record comparison scope",
                "",
                "Online fields are historical outputs captured by Record. Deferred rows stay missing. Complete replay independently processes every raw spectrum from a reset per-session baseline, so disagreement can reveal sparse online inference, shared live-state carryover, or a model/runtime difference.",
                "",
                "## Evidence boundary",
                "",
                "The frozen runtime predictions were created before answer access. The score is provisional because two answer folders required an explicit ordered fallback after a duplicated trial_017 answer directory was detected. Blind2 is evaluation-only and is not used to fit or retune the deployed model.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
