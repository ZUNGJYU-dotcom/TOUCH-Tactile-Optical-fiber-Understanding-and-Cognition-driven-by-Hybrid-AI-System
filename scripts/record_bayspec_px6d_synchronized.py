"""Record BaySpec full-spectrum frames with timestamp-aligned PX6D reference force."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "px6d_synchronized"


def request_json(url: str, *, method: str = "GET", timeout: float = 8.0) -> dict:
    request = Request(url, method=method)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("ok") is False:
        raise RuntimeError(payload.get("detail") or payload.get("status") or "API request failed")
    return payload


def frame_key(payload: dict) -> tuple:
    latest = payload.get("latest") or {}
    return (
        latest.get("source"),
        latest.get("frame_id"),
        latest.get("ingested_at") or latest.get("timestamp"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture full BaySpec spectra with synchronized PX6D Fz labels."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8630")
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--position", default="unlabeled")
    parser.add_argument("--action", default="static_press")
    parser.add_argument("--trial-id", default="trial_001")
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--tare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-missing-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write diagnostic spectrum frames even when no valid tared PX6D label is available.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{session_name}_{args.position}_{args.trial_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_path = output_dir / "synchronized_frames.jsonl"
    summary_path = output_dir / "frame_summary.csv"
    metadata_path = output_dir / "session_metadata.json"

    base_url = args.base_url.rstrip("/")
    request_json(f"{base_url}/api/px6d/start", method="POST")
    if args.tare:
        request_json(f"{base_url}/api/px6d/tare?duration_sec=1.0", method="POST", timeout=4.0)

    started_epoch = time.time()
    deadline = time.monotonic() + max(1.0, args.duration_sec)
    captured = 0
    skipped_duplicates = 0
    missing_spectrum_polls = 0
    missing_force = 0
    last_key = None
    summary_rows: list[dict] = []

    try:
        with frames_path.open("w", encoding="utf-8") as jsonl:
            while time.monotonic() < deadline:
                payload = request_json(
                    f"{base_url}/api/global_spectrum_frame"
                    "?trace_limit=1&include_spectrum=true"
                    "&include_shadow=false&include_dynamic_shadow=false"
                    "&temporal_validation_mode=false",
                    timeout=8.0,
                )
                latest = payload.get("latest")
                if not isinstance(latest, dict):
                    missing_spectrum_polls += 1
                    time.sleep(max(0.01, args.poll_interval_sec))
                    continue
                key = frame_key(payload)
                if key == last_key:
                    skipped_duplicates += 1
                    time.sleep(max(0.01, args.poll_interval_sec))
                    continue
                wavelengths = latest.get("wavelength_nm") or []
                intensities = latest.get("intensity") or []
                if not wavelengths or not intensities or len(wavelengths) != len(intensities):
                    missing_spectrum_polls += 1
                    time.sleep(max(0.01, args.poll_interval_sec))
                    continue
                force = payload.get("px6d_reference") or {}
                force_ready = bool(force.get("ok") and force.get("tare_ready"))
                if not force_ready:
                    missing_force += 1
                    if not args.allow_missing_force:
                        time.sleep(max(0.01, args.poll_interval_sec))
                        continue
                last_key = key
                row = {
                    "capture_index": captured,
                    "capture_timestamp_epoch_sec": time.time(),
                    "position_label": args.position,
                    "action_label": args.action,
                    "trial_id": args.trial_id,
                    "spectrum_source": latest.get("source"),
                    "spectrum_frame_id": latest.get("frame_id"),
                    "spectrum_timestamp_epoch_sec": latest.get("ingested_at")
                    or latest.get("timestamp"),
                    "wavelength_nm": wavelengths,
                    "intensity_counts": intensities,
                    "spectrum_peaks": latest.get("spectrum_peaks") or [],
                    "px6d_reference": force,
                }
                jsonl.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                raw = force.get("raw") or {}
                summary_rows.append(
                    {
                        "capture_index": captured,
                        "spectrum_frame_id": latest.get("frame_id"),
                        "spectrum_timestamp_epoch_sec": row["spectrum_timestamp_epoch_sec"],
                        "spectrum_points": min(len(wavelengths), len(intensities)),
                        "px6d_sync_ok": force.get("ok"),
                        "px6d_tare_ready": force.get("tare_ready"),
                        "px6d_sync_method": force.get("sync_method"),
                        "px6d_sync_offset_ms": force.get("sync_offset_ms"),
                        "px6d_sample_count": force.get("sample_count"),
                        "force_timestamp_epoch_sec": force.get(
                            "force_timestamp_epoch_sec"
                        ),
                        "fx_raw_n": raw.get("fx_n"),
                        "fy_raw_n": raw.get("fy_n"),
                        "fz_raw_n": raw.get("fz_n"),
                        "mx_raw_nm": raw.get("mx_nm"),
                        "my_raw_nm": raw.get("my_nm"),
                        "mz_raw_nm": raw.get("mz_nm"),
                        "reference_fz_n": force.get("reference_fz_n"),
                        "reference_fz_display_n": force.get("reference_fz_display_n"),
                        "position_label": args.position,
                        "action_label": args.action,
                        "trial_id": args.trial_id,
                    }
                )
                captured += 1
                time.sleep(max(0.01, args.poll_interval_sec))
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        error_message = f"{type(exc).__name__}: {exc}"
    else:
        error_message = None

    fieldnames = list(summary_rows[0]) if summary_rows else [
        "capture_index",
        "spectrum_frame_id",
        "spectrum_timestamp_epoch_sec",
        "spectrum_points",
        "px6d_sync_ok",
        "px6d_tare_ready",
        "px6d_sync_method",
        "px6d_sync_offset_ms",
        "px6d_sample_count",
        "force_timestamp_epoch_sec",
        "fx_raw_n",
        "fy_raw_n",
        "fz_raw_n",
        "mx_raw_nm",
        "my_raw_nm",
        "mz_raw_nm",
        "reference_fz_n",
        "reference_fz_display_n",
        "position_label",
        "action_label",
        "trial_id",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    capture_status = (
        "complete"
        if error_message is None and captured > 0
        else "no_valid_spectrum_frames"
        if error_message is None
        else "capture_error"
    )
    metadata = {
        "capture_status": capture_status,
        "session_name": session_name,
        "started_at_epoch_sec": started_epoch,
        "ended_at_epoch_sec": time.time(),
        "requested_duration_sec": args.duration_sec,
        "position_label": args.position,
        "action_label": args.action,
        "trial_id": args.trial_id,
        "operator_note": args.operator_note,
        "captured_unique_spectrum_frames": captured,
        "skipped_duplicate_spectrum_frames": skipped_duplicates,
        "polls_without_valid_spectrum": missing_spectrum_polls,
        "frames_missing_px6d_reference": missing_force,
        "software_tare_requested": args.tare,
        "allow_missing_force": args.allow_missing_force,
        "force_semantics": "PX6D_reference_Fz_not_optical_force_prediction",
        "synchronization": "host_epoch_timestamp_window_median",
        "error": error_message,
        "files": {
            "full_frames_jsonl": frames_path.name,
            "frame_summary_csv": summary_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    session_ok = error_message is None and captured > 0
    print(json.dumps({"ok": session_ok, "output_dir": str(output_dir), **metadata}, ensure_ascii=False, indent=2))
    return 0 if session_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
