"""Start the app's selectable, time-aligned TOUCH data recorder from the CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "px6d_synchronized"
AVAILABLE_OUTPUTS = ("spectrum", "response", "force")


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 8.0,
) -> dict:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("ok") is False:
        raise RuntimeError(
            result.get("reason")
            or result.get("detail")
            or result.get("status")
            or "API request failed"
        )
    return result


def parse_outputs(value: str) -> list[str]:
    selected = []
    for item in str(value or "").split(","):
        normalized = item.strip().lower()
        if normalized in AVAILABLE_OUTPUTS and normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise argparse.ArgumentTypeError(
            "select one or more of: spectrum,response,force"
        )
    return [name for name in AVAILABLE_OUTPUTS if name in selected]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture selectable full-spectrum, tactile-response, and PX6D force "
            "streams on one canonical timeline."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8640")
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--position", default="unlabeled")
    parser.add_argument("--action", default="static_press")
    parser.add_argument("--trial-id", default="trial_001")
    parser.add_argument("--operator-note", default="")
    parser.add_argument(
        "--outputs",
        type=parse_outputs,
        default=list(AVAILABLE_OUTPUTS),
        help="Comma-separated subset of spectrum,response,force (default: all).",
    )
    parser.add_argument("--tare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    selected = list(args.outputs)
    try:
        if "force" in selected:
            request_json(f"{base_url}/api/px6d/start", method="POST")
            if args.tare:
                request_json(
                    f"{base_url}/api/px6d/tare?duration_sec=1.0",
                    method="POST",
                    timeout=4.0,
                )
        started = request_json(
            f"{base_url}/api/px6d_capture/start",
            method="POST",
            payload={
                "position_label": args.position,
                "action_label": args.action,
                "trial_id": args.trial_id,
                "operator_note": args.operator_note,
                "output_root": str(args.output_root.expanduser().resolve()),
                "selected_outputs": selected,
            },
        )
        deadline = time.monotonic() + max(0.1, float(args.duration_sec))
        while time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
        stopped = request_json(
            f"{base_url}/api/px6d_capture/stop",
            method="POST",
            timeout=8.0,
        )
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        try:
            request_json(
                f"{base_url}/api/px6d_capture/stop",
                method="POST",
                timeout=4.0,
            )
        except Exception:
            pass
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1

    result = {
        "ok": int(stopped.get("captured_timeline_frames") or 0) > 0,
        "selected_outputs": selected,
        "output_directory": stopped.get("output_directory"),
        "captured_timeline_frames": stopped.get("captured_timeline_frames"),
        "capture_status": stopped.get("capture_status"),
        "timeline_basis": stopped.get("timeline_basis"),
        "output_files": stopped.get("output_files"),
        "start_status": started.get("status"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
