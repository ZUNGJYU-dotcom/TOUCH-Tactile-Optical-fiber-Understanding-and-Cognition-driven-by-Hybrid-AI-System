"""Audit synchronized ordinary-FBG and PX6D capture sessions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_ROOT = PROJECT_ROOT / "data" / "px6d_synchronized"
AXIS_FIELDS = (
    "fx_raw_n",
    "fy_raw_n",
    "fz_raw_n",
    "mx_raw_nm",
    "my_raw_nm",
    "mz_raw_nm",
    "fx_zeroed_n",
    "fy_zeroed_n",
    "fz_zeroed_n",
    "mx_zeroed_nm",
    "my_zeroed_nm",
    "mz_zeroed_nm",
    "fx_filtered_n",
    "fy_filtered_n",
    "fz_filtered_n",
    "mx_filtered_nm",
    "my_filtered_nm",
    "mz_filtered_nm",
)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "ok"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _status_from(findings: list[dict[str, str]]) -> str:
    severities = {item["severity"] for item in findings}
    if "error" in severities:
        return "fail"
    if "warning" in severities:
        return "usable_with_warning"
    return "pass"


def finding(
    code: str,
    severity: str,
    message: str,
) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def audit_session(
    session_dir: Path,
    *,
    minimum_frames: int,
    maximum_sync_offset_ms: float,
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.is_file():
        return {
            "session_directory": str(session_dir),
            "qa_status": "fail",
            "findings": [
                finding(
                    "missing_session_metadata",
                    "error",
                    "session_metadata.json is missing",
                )
            ],
        }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "session_directory": str(session_dir),
            "qa_status": "fail",
            "findings": [
                finding(
                    "invalid_session_metadata",
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            ],
        }

    selected = set(metadata.get("selected_outputs") or [])
    if not selected:
        files = metadata.get("files") or {}
        if files.get("spectrum_timeseries_csv"):
            selected.add("spectrum")
        if files.get("tactile_response_timeseries_csv"):
            selected.add("response")
        if files.get("force_timeseries_csv"):
            selected.add("force")
    expected = {
        "spectrum": session_dir / "spectrum_timeseries.csv",
        "response": session_dir / "tactile_response_timeseries.csv",
        "force": session_dir / "force_timeseries.csv",
    }
    for stream in selected:
        path = expected.get(stream)
        if path is not None and not path.is_file():
            findings.append(
                finding(
                    f"missing_{stream}_csv",
                    "error",
                    f"{path.name} is missing although {stream} was selected",
                )
            )

    summary_path = session_dir / "frame_summary.csv"
    if not summary_path.is_file():
        findings.append(
            finding(
                "missing_frame_summary",
                "error",
                "frame_summary.csv is missing",
            )
        )
        summary_fields: list[str] = []
        summary_rows: list[dict[str, str]] = []
    else:
        summary_fields, summary_rows = read_csv(summary_path)

    frame_count = len(summary_rows)
    if frame_count < minimum_frames:
        findings.append(
            finding(
                "insufficient_frames",
                "error",
                f"captured {frame_count} frames; require at least {minimum_frames}",
            )
        )
    required_summary_fields = {
        "capture_index",
        "timeline_timestamp_epoch_sec",
        "elapsed_time_sec",
        "position_label",
        "trial_id",
    }
    missing_summary_fields = sorted(required_summary_fields - set(summary_fields))
    if missing_summary_fields:
        findings.append(
            finding(
                "missing_summary_fields",
                "error",
                f"missing fields: {', '.join(missing_summary_fields)}",
            )
        )

    capture_indices: list[int] = []
    timestamps: list[float] = []
    for row in summary_rows:
        try:
            capture_indices.append(int(row.get("capture_index", "")))
        except ValueError:
            pass
        timestamp = finite_float(row.get("timeline_timestamp_epoch_sec"))
        if timestamp is not None:
            timestamps.append(timestamp)
    if len(capture_indices) != frame_count or len(set(capture_indices)) != frame_count:
        findings.append(
            finding(
                "invalid_capture_index",
                "error",
                "capture_index is missing or duplicated",
            )
        )
    if timestamps and any(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        findings.append(
            finding(
                "non_monotonic_timestamps",
                "error",
                "canonical timeline timestamps are not strictly increasing",
            )
        )

    spectrum_frame_count = 0
    minimum_spectrum_points = None
    maximum_spectrum_points = None
    if "spectrum" in selected and expected["spectrum"].is_file():
        spectrum_fields, spectrum_rows = read_csv(expected["spectrum"])
        required_spectrum = {
            "capture_index",
            "point_index",
            "wavelength_nm",
            "intensity_counts",
        }
        missing = sorted(required_spectrum - set(spectrum_fields))
        if missing:
            findings.append(
                finding(
                    "missing_spectrum_fields",
                    "error",
                    f"missing fields: {', '.join(missing)}",
                )
            )
        points_by_frame: dict[str, int] = {}
        invalid_spectrum_values = 0
        for row in spectrum_rows:
            key = str(row.get("capture_index") or "")
            points_by_frame[key] = points_by_frame.get(key, 0) + 1
            if finite_float(row.get("wavelength_nm")) is None or finite_float(
                row.get("intensity_counts")
            ) is None:
                invalid_spectrum_values += 1
        spectrum_frame_count = len(points_by_frame)
        point_counts = list(points_by_frame.values())
        minimum_spectrum_points = min(point_counts) if point_counts else 0
        maximum_spectrum_points = max(point_counts) if point_counts else 0
        if spectrum_frame_count != frame_count:
            findings.append(
                finding(
                    "spectrum_frame_count_mismatch",
                    "error",
                    f"spectrum frames={spectrum_frame_count}, timeline frames={frame_count}",
                )
            )
        if minimum_spectrum_points < 128:
            findings.append(
                finding(
                    "insufficient_spectrum_points",
                    "error",
                    f"minimum spectrum points per frame={minimum_spectrum_points}",
                )
            )
        if invalid_spectrum_values:
            findings.append(
                finding(
                    "invalid_spectrum_values",
                    "error",
                    f"{invalid_spectrum_values} spectrum rows contain non-finite values",
                )
            )

    force_frame_count = 0
    valid_force_ratio = None
    sync_pass_ratio = None
    maximum_observed_sync_offset_ms = None
    force_statistics: dict[str, float | None] = {}
    if "force" in selected and expected["force"].is_file():
        force_fields, force_rows = read_csv(expected["force"])
        required_force = {
            "capture_index",
            "force_timestamp_epoch_sec",
            "sync_offset_ms",
            "calibration_sync_ok",
            "force_fz_n",
            *AXIS_FIELDS,
        }
        missing = sorted(required_force - set(force_fields))
        if missing:
            findings.append(
                finding(
                    "missing_force_fields",
                    "error",
                    f"missing fields: {', '.join(missing)}",
                )
            )
        force_frame_count = len(force_rows)
        if force_frame_count != frame_count:
            findings.append(
                finding(
                    "force_frame_count_mismatch",
                    "error",
                    f"force frames={force_frame_count}, timeline frames={frame_count}",
                )
            )
        force_values: list[float] = []
        axis_rows_valid = 0
        sync_passes = 0
        offsets: list[float] = []
        for row in force_rows:
            force_value = finite_float(row.get("force_fz_n"))
            if force_value is not None and force_value >= 0.0:
                force_values.append(force_value)
            if all(finite_float(row.get(field)) is not None for field in AXIS_FIELDS):
                axis_rows_valid += 1
            offset = finite_float(row.get("sync_offset_ms"))
            if offset is not None:
                offsets.append(abs(offset))
            if truthy(row.get("calibration_sync_ok")) and (
                offset is None or abs(offset) <= maximum_sync_offset_ms
            ):
                sync_passes += 1
        valid_force_ratio = (
            min(len(force_values), axis_rows_valid) / force_frame_count
            if force_frame_count
            else 0.0
        )
        sync_pass_ratio = (
            sync_passes / force_frame_count if force_frame_count else 0.0
        )
        maximum_observed_sync_offset_ms = max(offsets) if offsets else None
        if valid_force_ratio < 0.95:
            findings.append(
                finding(
                    "invalid_force_coverage",
                    "error",
                    f"valid six-axis force coverage={valid_force_ratio:.1%}",
                )
            )
        if sync_pass_ratio < 0.95:
            findings.append(
                finding(
                    "force_sync_coverage_low",
                    "error",
                    f"calibration sync pass ratio={sync_pass_ratio:.1%}",
                )
            )

        if force_values:
            edge_count = max(1, int(round(len(force_values) * 0.15)))
            start_median = median(force_values[:edge_count])
            end_median = median(force_values[-edge_count:])
            p95 = percentile(force_values, 0.95)
            maximum = max(force_values)
            dynamic_range = max(0.0, (p95 or 0.0) - start_median)
            force_statistics = {
                "start_baseline_median_n": start_median,
                "end_recovery_median_n": end_median,
                "force_p95_n": p95,
                "force_max_n": maximum,
                "dynamic_range_n": dynamic_range,
            }
            if metadata.get("position_label") != "unlabeled":
                if dynamic_range < 0.03:
                    findings.append(
                        finding(
                            "insufficient_force_excursion",
                            "warning",
                            f"force dynamic range is only {dynamic_range:.4f} N",
                        )
                    )
                recovery_limit = max(0.03, 0.15 * max(dynamic_range, 0.03))
                if end_median - start_median > recovery_limit:
                    findings.append(
                        finding(
                            "release_recovery_residual",
                            "warning",
                            (
                                f"end baseline exceeds start by "
                                f"{end_median - start_median:.4f} N"
                            ),
                        )
                    )

    metadata_position = str(metadata.get("position_label") or "")
    metadata_trial = str(metadata.get("trial_id") or "")
    if metadata_position not in {
        "unlabeled",
        "P11",
        "P21",
        "P31",
        "P12",
        "P22",
        "P32",
        "P13",
        "P23",
        "P33",
    }:
        findings.append(
            finding(
                "invalid_position_label",
                "error",
                f"unexpected position_label={metadata_position!r}",
            )
        )
    if not metadata_trial or metadata_trial == "trial_001":
        findings.append(
            finding(
                "non_unique_trial_id_risk",
                "warning",
                f"trial_id={metadata_trial or 'missing'}",
            )
        )

    metadata_status = str(metadata.get("capture_status") or "")
    if metadata_status not in {"complete", "capture_stopped"}:
        findings.append(
            finding(
                "capture_not_complete",
                "error",
                f"capture_status={metadata_status or 'missing'}",
            )
        )
    qa_status = _status_from(findings)
    session_id = str(metadata.get("session_id") or session_dir.name)
    return {
        "session_directory": str(session_dir),
        "session_id": session_id,
        "formal_group_id": session_id,
        "trial_id": metadata_trial,
        "position_trial_key": (
            f"{metadata_position}:{metadata_trial}"
            if metadata_position and metadata_trial
            else ""
        ),
        "position_label": metadata_position,
        "selected_outputs": sorted(selected),
        "qa_status": qa_status,
        "frame_count": frame_count,
        "spectrum_frame_count": spectrum_frame_count,
        "minimum_spectrum_points": minimum_spectrum_points,
        "maximum_spectrum_points": maximum_spectrum_points,
        "force_frame_count": force_frame_count,
        "valid_force_ratio": valid_force_ratio,
        "sync_pass_ratio": sync_pass_ratio,
        "maximum_observed_sync_offset_ms": maximum_observed_sync_offset_ms,
        "force_statistics": force_statistics,
        "findings": findings,
    }


def discover_sessions(path: Path) -> list[Path]:
    if (path / "session_metadata.json").is_file():
        return [path]
    return sorted(
        candidate.parent
        for candidate in path.rglob("session_metadata.json")
        if candidate.parent.is_dir()
    )


def write_reports(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_counts = Counter(
        str(row.get("trial_id") or "")
        for row in results
        if str(row.get("trial_id") or "")
    )
    duplicate_trial_ids = sorted(
        trial_id for trial_id, count in trial_counts.items() if count > 1
    )
    for row in results:
        trial_id = str(row.get("trial_id") or "")
        row["trial_id_occurrences"] = trial_counts.get(trial_id, 0)
    summary = {
        "schema_version": "ordinary_fbg_px6d_collection_audit_v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_count": len(results),
        "unique_session_count": len(
            {str(row.get("session_id") or "") for row in results}
        ),
        "pass_count": sum(row["qa_status"] == "pass" for row in results),
        "warning_count": sum(
            row["qa_status"] == "usable_with_warning" for row in results
        ),
        "fail_count": sum(row["qa_status"] == "fail" for row in results),
        "formal_split_requirement": "grouped_by_session_id",
        "formal_group_field": "formal_group_id",
        "trial_id_semantics": "record_only_not_a_unique_group_key",
        "duplicate_trial_id_values": duplicate_trial_ids,
        "force_target": "continuous_force_fz_n",
        "results": results,
    }
    (output_dir / "collection_qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "session_id",
        "formal_group_id",
        "trial_id",
        "trial_id_occurrences",
        "position_trial_key",
        "position_label",
        "qa_status",
        "frame_count",
        "spectrum_frame_count",
        "minimum_spectrum_points",
        "force_frame_count",
        "valid_force_ratio",
        "sync_pass_ratio",
        "maximum_observed_sync_offset_ms",
        "start_baseline_median_n",
        "end_recovery_median_n",
        "force_p95_n",
        "force_max_n",
        "dynamic_range_n",
        "finding_codes",
        "session_directory",
    ]
    with (output_dir / "collection_qa_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            force = result.get("force_statistics") or {}
            writer.writerow(
                {
                    **{key: result.get(key) for key in fields},
                    **{key: force.get(key) for key in force},
                    "finding_codes": ";".join(
                        item["code"] for item in result.get("findings") or []
                    ),
                }
            )

    lines = [
        "# Ordinary FBG + PX6D Collection QA",
        "",
        f"- Sessions: {summary['session_count']}",
        f"- Pass: {summary['pass_count']}",
        f"- Usable with warning: {summary['warning_count']}",
        f"- Fail: {summary['fail_count']}",
        "- Force target: continuous `force_fz_n` in N",
        "- Formal evaluation split: grouped by unique `session_id`",
        "- `trial_id` is retained as a record label and is not a unique split key",
        (
            "- Reused trial ID values: "
            + (
                ", ".join(f"`{value}`" for value in duplicate_trial_ids)
                if duplicate_trial_ids
                else "none"
            )
        ),
        "",
        "## Sessions",
        "",
    ]
    for result in results:
        lines.append(
            f"### {result.get('session_id')} - "
            f"{result['qa_status']}"
        )
        lines.append("")
        lines.append(
            f"Position `{result.get('position_label')}`; "
            f"trial `{result.get('trial_id')}`; "
            f"frames {result.get('frame_count')}; "
            f"sync pass {result.get('sync_pass_ratio')}."
        )
        for item in result.get("findings") or []:
            lines.append(
                f"- {item['severity']}: `{item['code']}` - {item['message']}"
            )
        if not result.get("findings"):
            lines.append("- No QA findings.")
        lines.append("")
    (output_dir / "collection_qa_report.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit ordinary-FBG plus PX6D synchronized capture sessions."
    )
    parser.add_argument("capture_path", type=Path, nargs="?", default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--minimum-frames", type=int, default=20)
    parser.add_argument("--maximum-sync-offset-ms", type=float, default=250.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    capture_path = args.capture_path.expanduser().resolve()
    sessions = discover_sessions(capture_path)
    if not sessions:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "no_session_metadata_found",
                    "capture_path": str(capture_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    results = [
        audit_session(
            session,
            minimum_frames=max(1, args.minimum_frames),
            maximum_sync_offset_ms=max(0.0, args.maximum_sync_offset_ms),
        )
        for session in sessions
    ]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else capture_path
        / f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    write_reports(results, output_dir)
    payload = {
        "ok": all(result["qa_status"] != "fail" for result in results),
        "output_dir": str(output_dir),
        "session_count": len(results),
        "pass_count": sum(result["qa_status"] == "pass" for result in results),
        "warning_count": sum(
            result["qa_status"] == "usable_with_warning" for result in results
        ),
        "fail_count": sum(result["qa_status"] == "fail" for result in results),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
