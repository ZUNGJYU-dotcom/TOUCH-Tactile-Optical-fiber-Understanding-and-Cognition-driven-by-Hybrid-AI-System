"""Check whether TOUCH is ready for ordinary-FBG plus PX6D collection."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "px6d_synchronized"


def request_json(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 5.0,
) -> dict[str, Any]:
    request = Request(url, method=method)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return payload


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def check(
    name: str,
    *,
    passed: bool,
    required: bool,
    detail: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else ("fail" if required else "warning"),
        "required": required,
        "detail": detail,
    }


def writable_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=".touch_preflight_",
            suffix=".tmp",
            dir=path,
            delete=True,
            encoding="utf-8",
        ) as handle:
            handle.write("TOUCH preflight")
            handle.flush()
        return True, str(path)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_preflight(
    *,
    base_url: str,
    output_root: Path,
    maximum_optical_age_sec: float,
    require_baseline: bool,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    checks: list[dict[str, Any]] = []
    try:
        health = request_json(f"{base_url}/api/health")
        px6d = request_json(f"{base_url}/api/px6d/status")
        capture = request_json(f"{base_url}/api/px6d_capture/status")
        frame = request_json(
            f"{base_url}/api/frame?channel=P22&include_spectrum=false"
        )
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "ready_for_pilot_capture": False,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "checks": [
                check(
                    "touch_backend",
                    passed=False,
                    required=True,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            ],
        }

    checks.append(
        check(
            "touch_backend",
            passed=bool(health.get("ok")),
            required=True,
            detail=(
                f"version={health.get('version')}, build={health.get('build_id')}"
            ),
        )
    )
    profile = str(health.get("active_runtime_sensor_profile") or "")
    checks.append(
        check(
            "ordinary_fbg_profile",
            passed=profile == "ordinary_fbg_hybrid_spectral",
            required=True,
            detail=f"active_runtime_sensor_profile={profile or 'missing'}",
        )
    )
    checks.append(
        check(
            "recorder_idle",
            passed=not bool(capture.get("running"))
            and not bool(capture.get("start_in_progress")),
            required=True,
            detail=f"capture_status={capture.get('capture_status')}",
        )
    )

    directory_ok, directory_detail = writable_directory(output_root)
    checks.append(
        check(
            "output_directory_writable",
            passed=directory_ok,
            required=True,
            detail=directory_detail,
        )
    )

    checks.extend(
        [
            check(
                "px6d_running",
                passed=bool(px6d.get("running") and px6d.get("worker_alive")),
                required=True,
                detail=(
                    f"port={px6d.get('port')}, "
                    f"lifecycle={px6d.get('lifecycle_status')}"
                ),
            ),
            check(
                "px6d_connected",
                passed=bool(px6d.get("connected")),
                required=True,
                detail=(
                    f"port={px6d.get('port')}, "
                    f"last_error={px6d.get('last_error') or 'none'}"
                ),
            ),
            check(
                "px6d_sample_fresh",
                passed=bool(px6d.get("sample_fresh")),
                required=True,
                detail=(
                    f"sample_age_sec={px6d.get('last_sample_age_sec')}, "
                    f"observed_hz={px6d.get('observed_sample_hz')}"
                ),
            ),
            check(
                "px6d_software_zero",
                passed=bool(px6d.get("tare_ready")),
                required=True,
                detail=(
                    f"tare_status={px6d.get('tare_status')}, "
                    f"tare_fz_std_n={px6d.get('tare_fz_std_n')}"
                ),
            ),
        ]
    )

    sdk_live = frame.get("sdk_live") or {}
    export_watcher = frame.get("export_watcher") or {}
    optical_source_running = bool(
        sdk_live.get("active")
        and sdk_live.get("worker_alive")
        and sdk_live.get("process_running")
    ) or bool(export_watcher.get("active"))
    latest = frame.get("latest") or {}
    timestamp = finite_float(frame.get("timestamp") or latest.get("timestamp"))
    optical_age_sec = time.time() - timestamp if timestamp is not None else None
    spectrum_points = int(latest.get("spectrum_points") or 0)
    source = str(latest.get("source") or "")
    source_is_real = bool(source) and not any(
        marker in source.lower() for marker in ("demo", "synthetic", "simulated")
    )
    frame_fresh = bool(
        timestamp is not None
        and optical_age_sec is not None
        and 0.0 <= optical_age_sec <= maximum_optical_age_sec
    )
    checks.extend(
        [
            check(
                "optical_source_running",
                passed=optical_source_running,
                required=True,
                detail=(
                    f"sdk_active={sdk_live.get('active')}, "
                    f"watch_active={export_watcher.get('active')}"
                ),
            ),
            check(
                "real_spectrum_frame",
                passed=bool(frame_fresh and source_is_real and spectrum_points >= 128),
                required=True,
                detail=(
                    f"source={source or 'missing'}, frame_age_sec={optical_age_sec}, "
                    f"spectrum_points={spectrum_points}"
                ),
            ),
        ]
    )

    baseline_status = str(latest.get("baseline_status") or "unknown")
    baseline_ready = baseline_status not in {
        "",
        "unknown",
        "baseline_required",
        "not_ready",
    }
    checks.append(
        check(
            "optical_baseline",
            passed=baseline_ready,
            required=require_baseline,
            detail=f"baseline_status={baseline_status}",
        )
    )
    required_failures = [
        item["name"]
        for item in checks
        if item["required"] and item["status"] != "pass"
    ]
    warnings = [
        item["name"] for item in checks if item["status"] == "warning"
    ]
    return {
        "ok": not required_failures,
        "ready_for_pilot_capture": not required_failures,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "output_root": str(output_root),
        "required_failures": required_failures,
        "warnings": warnings,
        "checks": checks,
        "collection_contract": {
            "optical_input": "ordinary_FBG_full_spectrum",
            "force_reference": "PX6D_all_six_axes",
            "primary_force_target": "continuous_force_fz_n",
            "position_target": "P11_to_P33",
            "force_class_labels": False,
            "formal_split_group": "trial_id",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check TOUCH readiness before an ordinary-FBG plus PX6D pilot."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8640")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--maximum-optical-age-sec", type=float, default=1.5)
    parser.add_argument(
        "--require-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat optical baseline readiness as a required check.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    result = run_preflight(
        base_url=args.base_url,
        output_root=args.output_root.expanduser().resolve(),
        maximum_optical_age_sec=max(0.1, args.maximum_optical_age_sec),
        require_baseline=args.require_baseline,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output is not None:
        path = args.json_output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0 if result["ready_for_pilot_capture"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
