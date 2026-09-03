"""Score frozen TOUCH blind predictions, including ordered multi-press sessions.

The script never runs the model.  It verifies the pre-unblind hashes, then uses
the frozen per-frame predictions together with PX6D-defined contact episodes to
score answer folders that contain either one position, no contact, or an
ordered list of positions such as ``...-p11_p21_p31``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITIONS = ("P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33")
CONFUSION_LABELS = ("none",) + POSITIONS
POSITION_PATTERN = re.compile(r"P(?:11|12|13|21|22|23|31|32|33)$", re.IGNORECASE)
NO_CONTACT_LABELS = {"uncontact", "nocontact", "no_contact", "no-contact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    text = str(value).strip().upper()
    return text if text in POSITIONS else "none"


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_percentile(values: Iterable[Any], percentile: float) -> float | None:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.percentile(numeric, percentile)) if numeric.size else None


def _majority(values: Iterable[Any]) -> tuple[str, int, int, float, float]:
    labels = [_label(value) for value in values]
    labels = [value for value in labels if value != "none"]
    if not labels:
        return "none", 0, 0, 0.0, 0.0
    counts = Counter(labels)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], POSITIONS.index(item[0])))
    winner, winner_count = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    total = len(labels)
    return (
        winner,
        int(winner_count),
        int(total),
        float(winner_count / total),
        float((winner_count - runner_up) / total),
    )


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask.astype(bool)):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active else index - 1
            runs.append((start, end))
            start = None
    return runs


def _force_metrics(reference: Iterable[Any], estimate: Iterable[Any]) -> dict[str, Any]:
    actual = pd.to_numeric(pd.Series(list(reference)), errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(pd.Series(list(estimate)), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[finite]
    predicted = predicted[finite]
    if not actual.size:
        return {"n": 0, "mae_n": None, "rmse_n": None, "pearson_r": None, "slope": None, "intercept_n": None}
    error = predicted - actual
    pearson = None
    if actual.size >= 2 and np.std(actual) > 1.0e-12 and np.std(predicted) > 1.0e-12:
        pearson = float(np.corrcoef(actual, predicted)[0, 1])
    slope = intercept = None
    if actual.size >= 2 and np.std(actual) > 1.0e-12:
        slope, intercept = (float(value) for value in np.polyfit(actual, predicted, 1))
    return {
        "n": int(actual.size),
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_r": pearson,
        "slope": slope,
        "intercept_n": intercept,
    }


def _parse_answer_suffix(suffix: str) -> list[str]:
    lowered = suffix.strip().lower()
    if lowered in NO_CONTACT_LABELS:
        return []
    tokens = suffix.split("_")
    if not tokens or any(POSITION_PATTERN.fullmatch(token) is None for token in tokens):
        raise ValueError(f"unrecognized answer suffix: {suffix}")
    return [token.upper() for token in tokens]


def _answer_mapping(answer_root: Path, session_ids: list[str]) -> dict[str, dict[str, Any]]:
    folders = sorted((path for path in answer_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if len(folders) != len(session_ids):
        raise RuntimeError(f"answer/session count mismatch: {len(folders)} vs {len(session_ids)}")
    mapping: dict[str, dict[str, Any]] = {}
    used: set[Path] = set()
    for session_id in session_ids:
        prefix = f"{session_id}-"
        matches = [path for path in folders if path.name.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"answer match for {session_id}: {len(matches)} folders")
        folder = matches[0]
        suffix = folder.name[len(prefix) :]
        mapping[session_id] = {
            "folder_name": folder.name,
            "labels": _parse_answer_suffix(suffix),
        }
        used.add(folder)
    if len(used) != len(folders):
        raise RuntimeError("one or more answer folders were not uniquely consumed")
    return mapping


def _confusion(rows: pd.DataFrame, predicted_column: str) -> dict[str, dict[str, int]]:
    return {
        expected: {
            predicted: int(
                ((rows["expected_position"] == expected) & (rows[predicted_column] == predicted)).sum()
            )
            for predicted in CONFUSION_LABELS
        }
        for expected in CONFUSION_LABELS
    }


def _first_correct_delay_ms(frame: pd.DataFrame, mask: np.ndarray, column: str, expected: str) -> float | None:
    indexes = np.flatnonzero(mask)
    if not indexes.size:
        return None
    onset = float(frame.iloc[indexes[0]]["elapsed_time_sec"])
    correct = indexes[frame.iloc[indexes][column].to_numpy() == expected]
    if not correct.size:
        return None
    return float((float(frame.iloc[correct[0]]["elapsed_time_sec"]) - onset) * 1000.0)


def _plot_confusion(units: pd.DataFrame, path: Path, audit_name: str) -> None:
    active = units[units["expected_position"] != "none"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "offline_predicted_position", "Frozen offline replay"),
        (axes[1], "online_predicted_position", "Saved Record runtime"),
    ):
        matrix = np.array(
            [
                [int(((active["expected_position"] == expected) & (active[column] == predicted)).sum()) for predicted in CONFUSION_LABELS]
                for expected in POSITIONS
            ]
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_xticks(range(len(CONFUSION_LABELS)), CONFUSION_LABELS, rotation=45, ha="right")
        axis.set_yticks(range(len(POSITIONS)), POSITIONS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Answer")
        axis.set_title(title, loc="left", fontweight="bold")
        for row in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row, column_index]
                if value:
                    axis.text(column_index, row, str(value), ha="center", va="center", color="#17324d")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{audit_name} active-episode position confusion", fontsize=18, fontweight="bold", color="#17324d")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_force(frame: pd.DataFrame, path: Path, audit_name: str) -> None:
    active = frame[frame["expected_position"] != "none"]
    figure, axes = plt.subplots(3, 3, figsize=(15, 14), sharex=True, sharey=True, constrained_layout=True)
    for axis, position in zip(axes.flat, POSITIONS, strict=True):
        subset = active[active["expected_position"] == position]
        axis.scatter(
            subset["reference_force_fz_n"],
            subset["offline_optical_force_n"],
            s=11,
            alpha=0.45,
            color="#2a9d8f",
            edgecolors="none",
        )
        metrics = _force_metrics(subset["reference_force_fz_n"], subset["offline_optical_force_n"])
        maximum = max(
            1.0,
            float(pd.to_numeric(subset["reference_force_fz_n"], errors="coerce").max()),
            float(pd.to_numeric(subset["offline_optical_force_n"], errors="coerce").max()),
        )
        axis.plot([0, maximum], [0, maximum], linestyle="--", color="#7b8794", linewidth=1)
        axis.set_title(position, loc="left", fontweight="bold")
        axis.text(
            0.98,
            0.04,
            f"MAE={metrics['mae_n']:.2f} N\nr={metrics['pearson_r']:.3f}" if metrics["pearson_r"] is not None else f"MAE={metrics['mae_n']:.2f} N",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#516b84",
        )
        axis.grid(color="#e1e9f0", linewidth=0.7)
    for axis in axes[-1, :]:
        axis.set_xlabel("PX6D Fz (N)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Optical estimate (N)")
    figure.suptitle(f"{audit_name} frozen optical-force agreement", fontsize=18, fontweight="bold", color="#17324d")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_sequences(frame: pd.DataFrame, session_table: pd.DataFrame, path: Path, audit_name: str) -> None:
    sequence_sessions = session_table[session_table["answer_position_count"] > 1]
    if sequence_sessions.empty:
        return
    figure, axes = plt.subplots(len(sequence_sessions), 1, figsize=(16, 4.6 * len(sequence_sessions)), squeeze=False, constrained_layout=True)
    colors = dict(zip(POSITIONS, plt.get_cmap("tab10").colors[: len(POSITIONS)], strict=True))
    for axis, session in zip(axes.flat, sequence_sessions.itertuples(index=False), strict=True):
        subset = frame[frame["session_id"] == session.session_id]
        axis.plot(subset["elapsed_time_sec"], subset["reference_force_fz_n"], color="#17324d", linewidth=1.7, label="PX6D Fz")
        axis.plot(subset["elapsed_time_sec"], subset["offline_optical_force_n"], color="#e76f51", linewidth=1.2, alpha=0.9, label="Optical estimate")
        for position in POSITIONS:
            marked = subset[subset["expected_position"] == position]
            if not marked.empty:
                axis.axvspan(float(marked["elapsed_time_sec"].min()), float(marked["elapsed_time_sec"].max()), color=colors[position], alpha=0.10)
                axis.text(float(marked["elapsed_time_sec"].median()), axis.get_ylim()[1] * 0.92, position, ha="center", va="top", fontsize=8, color=colors[position])
        wrong = subset[(subset["expected_position"] != "none") & (subset["offline_display_position"] != subset["expected_position"])]
        axis.scatter(wrong["elapsed_time_sec"], wrong["reference_force_fz_n"], marker="x", s=18, color="#a23b3b", label="Position mismatch")
        axis.set_title(f"Session {int(session.session_order)}", loc="left", fontweight="bold")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
        axis.grid(color="#e1e9f0", linewidth=0.7)
        axis.legend(frameon=False, ncol=3)
    figure.suptitle(f"{audit_name} ordered nine-position sequences", fontsize=18, fontweight="bold", color="#17324d")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    blind_root = args.blind_root.resolve()
    prediction_dir = args.prediction_dir.resolve()
    answer_root = args.answer_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output_dir}")

    manifest_path = prediction_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("answer_accessed") is not False:
        raise RuntimeError("prediction manifest is not marked pre-unblind")
    frames_path = prediction_dir / "frames.csv"
    predictions_path = prediction_dir / "blind_predictions.csv"
    expected_frames_hash = str(((manifest.get("outputs") or {}).get("frames_sha256") or ""))
    expected_predictions_hash = str(((manifest.get("outputs") or {}).get("predictions_sha256") or ""))
    if _sha256(frames_path) != expected_frames_hash:
        raise RuntimeError("frozen frame hash mismatch")
    if _sha256(predictions_path) != expected_predictions_hash:
        raise RuntimeError("frozen prediction hash mismatch")

    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig").sort_values("session_order")
    session_ids = predictions["session_id"].astype(str).tolist()
    answers = _answer_mapping(answer_root, session_ids)
    frozen = pd.read_csv(frames_path, encoding="utf-8-sig")
    frozen["active_reference_frame"] = frozen["active_reference_frame"].map(_truthy)
    if "visual_active" not in frozen and "candidate_contact_label" in frozen:
        frozen["visual_active"] = (
            frozen["candidate_contact_label"].astype(str).str.lower() == "contact"
        )
    if "visual_position" not in frozen and "candidate_position_label" in frozen:
        frozen["visual_position"] = frozen["candidate_position_label"]
    if "estimated_force_fz_n" not in frozen and "candidate_force_fz_n" in frozen:
        frozen["estimated_force_fz_n"] = frozen["candidate_force_fz_n"]
    required_frozen = {"visual_active", "visual_position", "estimated_force_fz_n"}
    missing_frozen = sorted(required_frozen - set(frozen.columns))
    if missing_frozen:
        raise ValueError(f"frozen prediction fields missing: {missing_frozen}")
    frozen["visual_active"] = frozen["visual_active"].map(_truthy)

    unit_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    for prediction in predictions.itertuples(index=False):
        session_id = str(prediction.session_id)
        session_order = int(prediction.session_order)
        session_dir = blind_root / session_id
        offline = frozen[frozen["session_id"] == session_id].copy().sort_values("capture_index").reset_index(drop=True)
        online = pd.read_csv(session_dir / "frame_summary.csv", encoding="utf-8-sig")
        required_online = {
            "capture_index",
            "force_fz_n",
            "model_ready",
            "capture_response_source",
            "capture_response_frame_match",
            "model_inference_latency_ms",
            "display_contact_active",
            "display_position_label",
            "display_position_confidence",
            "display_position_margin",
            "display_optical_force_n",
            "formal_position_label",
            "raw_position_label",
        }
        missing = sorted(required_online - set(online.columns))
        if missing:
            raise ValueError(f"{session_id} lacks Record fields: {missing}")
        online = online[list(required_online)].copy()
        online = online.rename(
            columns={
                "force_fz_n": "record_reference_force_fz_n",
                "model_ready": "online_model_ready",
                "display_contact_active": "online_display_contact_active",
                "display_position_label": "online_display_position_label",
                "display_position_confidence": "online_display_position_confidence",
                "display_position_margin": "online_display_position_margin",
                "display_optical_force_n": "online_display_optical_force_n",
                "formal_position_label": "online_formal_position_label",
                "raw_position_label": "online_raw_position_label",
                "model_inference_latency_ms": "online_model_inference_latency_ms",
            }
        )
        if offline["capture_index"].duplicated().any() or online["capture_index"].duplicated().any():
            raise RuntimeError(f"duplicate capture_index in {session_id}")
        frame = offline.merge(online, on="capture_index", how="inner", validate="one_to_one")
        if len(frame) != len(offline) or len(frame) != len(online):
            raise RuntimeError(f"incomplete frozen/Record join for {session_id}")
        force_delta = np.abs(
            pd.to_numeric(frame["reference_force_fz_n"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(frame["record_reference_force_fz_n"], errors="coerce").to_numpy(dtype=float)
        )
        if np.nanmax(force_delta) > 1.0e-9:
            raise RuntimeError(f"reference force mismatch in {session_id}")

        frame["session_order"] = session_order
        frame["offline_display_position"] = [
            _label(value) if active else "none"
            for value, active in zip(frame["visual_position"], frame["visual_active"], strict=True)
        ]
        frame["online_prediction_available"] = frame["online_model_ready"].map(_truthy)
        frame["online_display_active"] = frame["online_prediction_available"] & frame["online_display_contact_active"].map(_truthy)
        frame["online_display_position"] = [
            _label(value) if active else "none"
            for value, active in zip(frame["online_display_position_label"], frame["online_display_active"], strict=True)
        ]
        frame["offline_optical_force_n"] = pd.to_numeric(frame["estimated_force_fz_n"], errors="coerce")
        frame["online_display_optical_force_n"] = pd.to_numeric(frame["online_display_optical_force_n"], errors="coerce")

        answer = answers[session_id]
        expected_labels = list(answer["labels"])
        active_mask = frame["active_reference_frame"].to_numpy(dtype=bool)
        runs = _contiguous_runs(active_mask)
        expected = np.full(len(frame), "none", dtype=object)
        units: list[tuple[np.ndarray, str, int | None, int | None]] = []
        if not expected_labels:
            units.append((np.ones(len(frame), dtype=bool), "none", None, None))
        elif len(expected_labels) == 1:
            if not runs:
                raise RuntimeError(f"active answer has no PX6D episode: {session_id}")
            expected[active_mask] = expected_labels[0]
            units.append((active_mask.copy(), expected_labels[0], runs[0][0], runs[-1][1]))
        else:
            if len(runs) != len(expected_labels):
                raise RuntimeError(
                    f"{session_id}: {len(expected_labels)} answer positions but {len(runs)} frozen PX6D episodes"
                )
            for label, (start, end) in zip(expected_labels, runs, strict=True):
                mask = np.zeros(len(frame), dtype=bool)
                mask[start : end + 1] = True
                expected[mask] = label
                units.append((mask, label, start, end))
        frame["expected_position"] = expected
        frame["offline_answer_correct"] = frame["offline_display_position"] == frame["expected_position"]
        frame["online_answer_correct"] = np.where(
            frame["online_prediction_available"],
            frame["online_display_position"] == frame["expected_position"],
            False,
        )

        first_unit_index = len(unit_rows)
        for local_index, (unit_mask, expected_position, start, end) in enumerate(units, start=1):
            if expected_position == "none":
                offline_votes = frame.loc[unit_mask & frame["visual_active"].to_numpy(dtype=bool), "offline_display_position"]
                online_votes = frame.loc[unit_mask & frame["online_display_active"].to_numpy(dtype=bool), "online_display_position"]
            else:
                offline_votes = frame.loc[unit_mask & frame["visual_active"].to_numpy(dtype=bool), "offline_display_position"]
                online_votes = frame.loc[unit_mask & frame["online_display_active"].to_numpy(dtype=bool), "online_display_position"]
            offline_majority = _majority(offline_votes)
            online_majority = _majority(online_votes)
            available_mask = unit_mask & frame["online_prediction_available"].to_numpy(dtype=bool)
            unit_rows.append(
                {
                    "evaluation_unit": len(unit_rows) + 1,
                    "session_order": session_order,
                    "session_id": session_id,
                    "answer_folder": answer["folder_name"],
                    "unit_index_in_session": local_index,
                    "unit_type": "idle_session" if expected_position == "none" else ("ordered_episode" if len(expected_labels) > 1 else "single_position_session"),
                    "expected_position": expected_position,
                    "offline_predicted_position": offline_majority[0],
                    "offline_correct": offline_majority[0] == expected_position,
                    "offline_vote_count": offline_majority[2],
                    "offline_winner_share": offline_majority[3],
                    "offline_winner_margin_share": offline_majority[4],
                    "online_predicted_position": online_majority[0],
                    "online_correct": online_majority[0] == expected_position,
                    "online_vote_count": online_majority[2],
                    "online_winner_share": online_majority[3],
                    "online_winner_margin_share": online_majority[4],
                    "reference_frames": int(unit_mask.sum()),
                    "offline_active_frames": int((unit_mask & frame["visual_active"].to_numpy(dtype=bool)).sum()),
                    "online_available_frames": int(available_mask.sum()),
                    "online_active_frames": int((unit_mask & frame["online_display_active"].to_numpy(dtype=bool)).sum()),
                    "offline_frame_accuracy": float(frame.loc[unit_mask, "offline_answer_correct"].mean()),
                    "online_frame_accuracy_when_available": float(frame.loc[available_mask, "online_answer_correct"].mean()) if available_mask.any() else None,
                    "episode_start_capture_index": int(frame.iloc[start]["capture_index"]) if start is not None else None,
                    "episode_end_capture_index": int(frame.iloc[end]["capture_index"]) if end is not None else None,
                    "episode_start_sec": float(frame.iloc[start]["elapsed_time_sec"]) if start is not None else None,
                    "episode_end_sec": float(frame.iloc[end]["elapsed_time_sec"]) if end is not None else None,
                    "offline_first_correct_delay_ms": _first_correct_delay_ms(frame, unit_mask, "offline_display_position", expected_position) if expected_position != "none" else None,
                    "online_first_correct_delay_ms": _first_correct_delay_ms(frame, available_mask, "online_display_position", expected_position) if expected_position != "none" else None,
                }
            )

        metadata = json.loads((session_dir / "session_metadata.json").read_text(encoding="utf-8"))
        session_units = unit_rows[first_unit_index:]
        idle_mask = frame["expected_position"].to_numpy() == "none"
        online_idle_available = idle_mask & frame["online_prediction_available"].to_numpy(dtype=bool)
        offline_force = _force_metrics(frame["reference_force_fz_n"], frame["offline_optical_force_n"])
        online_force_mask = frame["online_prediction_available"].to_numpy(dtype=bool)
        online_force = _force_metrics(
            frame.loc[online_force_mask, "reference_force_fz_n"],
            frame.loc[online_force_mask, "online_display_optical_force_n"],
        )
        provenance = (((metadata.get("provenance") or {}).get("start") or {}))
        session_rows.append(
            {
                "session_order": session_order,
                "session_id": session_id,
                "answer_folder": answer["folder_name"],
                "answer_positions": "_".join(expected_labels) if expected_labels else "none",
                "answer_position_count": len(expected_labels),
                "px6d_episode_count": len(runs),
                "evaluation_unit_count": len(session_units),
                "offline_correct_units": sum(bool(row["offline_correct"]) for row in session_units),
                "offline_session_exact": all(bool(row["offline_correct"]) for row in session_units),
                "online_correct_units": sum(bool(row["online_correct"]) for row in session_units),
                "online_session_exact": all(bool(row["online_correct"]) for row in session_units),
                "total_frames": len(frame),
                "active_reference_frames": int(active_mask.sum()),
                "offline_active_frame_accuracy": float(frame.loc[active_mask, "offline_answer_correct"].mean()) if active_mask.any() else None,
                "online_active_frame_accuracy_when_available": float(frame.loc[active_mask & online_force_mask, "online_answer_correct"].mean()) if (active_mask & online_force_mask).any() else None,
                "online_prediction_coverage": float(frame["online_prediction_available"].mean()),
                "offline_idle_false_activation_rate": float(frame.loc[idle_mask, "visual_active"].mean()) if idle_mask.any() else None,
                "online_idle_false_activation_rate_when_available": float(frame.loc[online_idle_available, "online_display_active"].mean()) if online_idle_available.any() else None,
                "offline_force_mae_n": offline_force["mae_n"],
                "offline_force_pearson_r": offline_force["pearson_r"],
                "offline_force_slope": offline_force["slope"],
                "online_force_mae_n": online_force["mae_n"],
                "online_force_pearson_r": online_force["pearson_r"],
                "online_force_slope": online_force["slope"],
                "online_inference_latency_median_ms": _finite_percentile(frame.loc[online_force_mask, "online_model_inference_latency_ms"], 50),
                "online_inference_latency_p95_ms": _finite_percentile(frame.loc[online_force_mask, "online_model_inference_latency_ms"], 95),
                "capture_rate_hz": metadata.get("captured_frame_rate_hz"),
                "capture_response_source_counts": json.dumps(dict(Counter(frame["capture_response_source"].astype(str))), sort_keys=True),
                "record_software_version": ((provenance.get("software") or {}).get("version")),
                "record_model_sha256": ((provenance.get("model") or {}).get("model_bundle_sha256")),
                "record_baseline_token": ((provenance.get("baseline") or {}).get("token")),
            }
        )
        all_frames.append(frame)

    frame_table = pd.concat(all_frames, ignore_index=True)
    unit_table = pd.DataFrame(unit_rows)
    session_table = pd.DataFrame(session_rows)
    active_units = unit_table[unit_table["expected_position"] != "none"]
    idle_units = unit_table[unit_table["expected_position"] == "none"]
    active_frames = frame_table["expected_position"] != "none"
    idle_frames = ~active_frames
    online_available = frame_table["online_prediction_available"]
    online_active_available = active_frames & online_available
    online_idle_available = idle_frames & online_available

    per_position_rows: list[dict[str, Any]] = []
    for position in POSITIONS:
        units = active_units[active_units["expected_position"] == position]
        frames = frame_table[frame_table["expected_position"] == position]
        online_frames = frames[frames["online_prediction_available"]]
        offline_force = _force_metrics(frames["reference_force_fz_n"], frames["offline_optical_force_n"])
        online_force = _force_metrics(online_frames["reference_force_fz_n"], online_frames["online_display_optical_force_n"])
        per_position_rows.append(
            {
                "position": position,
                "episode_count": len(units),
                "offline_episode_accuracy": float(units["offline_correct"].mean()) if len(units) else None,
                "online_episode_accuracy": float(units["online_correct"].mean()) if len(units) else None,
                "offline_active_frame_accuracy": float(frames["offline_answer_correct"].mean()) if len(frames) else None,
                "online_active_frame_accuracy_when_available": float(online_frames["online_answer_correct"].mean()) if len(online_frames) else None,
                "online_active_prediction_coverage": _safe_ratio(len(online_frames), len(frames)),
                "offline_force_mae_n": offline_force["mae_n"],
                "offline_force_pearson_r": offline_force["pearson_r"],
                "offline_force_slope": offline_force["slope"],
                "online_force_mae_n": online_force["mae_n"],
                "online_force_pearson_r": online_force["pearson_r"],
                "online_force_slope": online_force["slope"],
            }
        )
    per_position_table = pd.DataFrame(per_position_rows)

    offline_force_all = _force_metrics(frame_table["reference_force_fz_n"], frame_table["offline_optical_force_n"])
    offline_force_active = _force_metrics(frame_table.loc[active_frames, "reference_force_fz_n"], frame_table.loc[active_frames, "offline_optical_force_n"])
    online_force_all = _force_metrics(frame_table.loc[online_available, "reference_force_fz_n"], frame_table.loc[online_available, "online_display_optical_force_n"])
    online_force_active = _force_metrics(frame_table.loc[online_active_available, "reference_force_fz_n"], frame_table.loc[online_active_available, "online_display_optical_force_n"])

    output_dir.mkdir(parents=True, exist_ok=False)
    unit_path = output_dir / "evaluation_units.csv"
    frame_path = output_dir / "frame_comparison.csv"
    session_path = output_dir / "session_summary.csv"
    position_path = output_dir / "per_position_summary.csv"
    unit_table.to_csv(unit_path, index=False, encoding="utf-8-sig")
    frame_table.to_csv(frame_path, index=False, encoding="utf-8-sig")
    session_table.to_csv(session_path, index=False, encoding="utf-8-sig")
    per_position_table.to_csv(position_path, index=False, encoding="utf-8-sig")

    audit_name = blind_root.name
    confusion_path = output_dir / "position_confusion.png"
    force_path = output_dir / "force_agreement.png"
    sequence_path = output_dir / "ordered_sequences.png"
    _plot_confusion(unit_table, confusion_path, audit_name)
    _plot_force(frame_table, force_path, audit_name)
    _plot_sequences(frame_table, session_table, sequence_path, audit_name)

    summary = {
        "schema_version": "touch_blind_multiepisode_record_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_frozen_before_answer_access": True,
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": _sha256(manifest_path),
        "frozen_frames_sha256": _sha256(frames_path),
        "frozen_predictions_sha256": _sha256(predictions_path),
        "model_sha256": (
            (manifest.get("runtime_artifacts") or {}).get("model_sha256")
            or (manifest.get("candidate_artifact") or {}).get("sha256")
        ),
        "training_overlap_count": ((manifest.get("training_overlap_audit") or {}).get("overlap_count")),
        "answer_root": str(answer_root),
        "answer_folder_names": [answers[session_id]["folder_name"] for session_id in session_ids],
        "segmentation_rule": "contiguous frozen active_reference_frame runs; threshold inherited from pre-unblind evaluator",
        "session_count": len(session_table),
        "sequence_session_count": int((session_table["answer_position_count"] > 1).sum()),
        "evaluation_unit_count": len(unit_table),
        "active_episode_count": len(active_units),
        "idle_session_count": len(idle_units),
        "offline": {
            "unit_accuracy": float(unit_table["offline_correct"].mean()),
            "active_episode_accuracy": float(active_units["offline_correct"].mean()),
            "idle_session_accuracy": float(idle_units["offline_correct"].mean()),
            "active_frame_position_accuracy": float(frame_table.loc[active_frames, "offline_answer_correct"].mean()),
            "active_frame_contact_coverage": float(frame_table.loc[active_frames, "visual_active"].mean()),
            "idle_frame_false_activation_rate": float(frame_table.loc[idle_frames, "visual_active"].mean()),
            "session_exact_accuracy": float(session_table["offline_session_exact"].mean()),
            "force_all_frames": offline_force_all,
            "force_active_frames": offline_force_active,
        },
        "online_record": {
            "unit_accuracy": float(unit_table["online_correct"].mean()),
            "active_episode_accuracy": float(active_units["online_correct"].mean()),
            "idle_session_accuracy": float(idle_units["online_correct"].mean()),
            "prediction_coverage": float(online_available.mean()),
            "active_frame_position_accuracy_when_available": float(frame_table.loc[online_active_available, "online_answer_correct"].mean()),
            "active_frame_contact_coverage_when_available": float(frame_table.loc[online_active_available, "online_display_active"].mean()),
            "idle_frame_false_activation_rate_when_available": float(frame_table.loc[online_idle_available, "online_display_active"].mean()),
            "session_exact_accuracy": float(session_table["online_session_exact"].mean()),
            "inference_latency_median_ms": _finite_percentile(frame_table.loc[online_available, "online_model_inference_latency_ms"], 50),
            "inference_latency_p95_ms": _finite_percentile(frame_table.loc[online_available, "online_model_inference_latency_ms"], 95),
            "force_all_available_frames": online_force_all,
            "force_active_available_frames": online_force_active,
        },
        "offline_unit_confusion": _confusion(unit_table, "offline_predicted_position"),
        "online_unit_confusion": _confusion(unit_table, "online_predicted_position"),
        "outputs": {},
    }
    for path in (unit_path, frame_path, session_path, position_path, confusion_path, force_path):
        summary["outputs"][path.name] = _sha256(path)
    if sequence_path.exists():
        summary["outputs"][sequence_path.name] = _sha256(sequence_path)

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_dir / "audit_report_zh.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {audit_name} 严格盲测报告",
                "",
                "## 证据边界",
                "",
                f"- 模型推理在读取答案前已冻结；预测哈希：`{summary['frozen_predictions_sha256']}`。",
                f"- 冻结逐帧结果哈希：`{summary['frozen_frames_sha256']}`。",
                f"- 与训练清单精确会话重叠：{summary['training_overlap_count']}。",
                "- 解盲后未重新推理、未改模型、未改阈值。连续会话按冻结的 PX6D 有效帧切成连续片段。",
                "",
                "## 位置识别",
                "",
                f"- 共 {summary['evaluation_unit_count']} 个评价单元，其中 {summary['active_episode_count']} 个按压片段、{summary['idle_session_count']} 个空载会话。",
                f"- 冻结离线片段准确率：{summary['offline']['active_episode_accuracy']:.1%}；空载会话准确率：{summary['offline']['idle_session_accuracy']:.1%}。",
                f"- 冻结离线有效按压帧位置准确率：{summary['offline']['active_frame_position_accuracy']:.1%}；接触覆盖率：{summary['offline']['active_frame_contact_coverage']:.1%}。",
                f"- Record 在线保存片段准确率：{summary['online_record']['active_episode_accuracy']:.1%}；在线帧覆盖率：{summary['online_record']['prediction_coverage']:.1%}。",
                f"- Record 在线可用帧位置准确率：{summary['online_record']['active_frame_position_accuracy_when_available']:.1%}。",
                "",
                "## 光学估力",
                "",
                f"- 冻结离线有效按压帧：MAE {offline_force_active['mae_n']:.3f} N，r {offline_force_active['pearson_r']:.3f}，斜率 {offline_force_active['slope']:.3f}。",
                f"- Record 在线可用按压帧：MAE {online_force_active['mae_n']:.3f} N，r {online_force_active['pearson_r']:.3f}，斜率 {online_force_active['slope']:.3f}。",
                "",
                "## Record 性能",
                "",
                f"- 保存的实时识别覆盖率：{summary['online_record']['prediction_coverage']:.1%}。",
                f"- 模型推理延迟中位数：{summary['online_record']['inference_latency_median_ms']:.1f} ms；P95：{summary['online_record']['inference_latency_p95_ms']:.1f} ms。",
                "",
                "详细逐片段、逐帧、逐会话及逐位置结果见同目录 CSV 与图。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["outputs"][report_path.name] = _sha256(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"offline active episodes={int(active_units['offline_correct'].sum())}/{len(active_units)} "
        f"({summary['offline']['active_episode_accuracy']:.1%}); "
        f"idle={int(idle_units['offline_correct'].sum())}/{len(idle_units)}; "
        f"online active episodes={int(active_units['online_correct'].sum())}/{len(active_units)}"
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
