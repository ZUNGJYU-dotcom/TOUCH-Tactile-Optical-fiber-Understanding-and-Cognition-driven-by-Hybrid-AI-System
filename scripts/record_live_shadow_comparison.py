"""Record deployed and candidate static-spectrum predictions side by side.

The script never deploys a model and never controls the digital twin.  Baseline
capture is opt-in and additionally requires ``--confirm-sensor-released``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_spectrum.live_shadow_validation import (  # noqa: E402
    flatten_shadow_frame,
    write_shadow_validation_artifacts,
)


POSITION_ORDER = ("P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33")
FORCE_ORDER = ("light", "normal", "hard")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8640")
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--poll-interval-ms", type=int, default=100)
    parser.add_argument("--expected-contact", choices=("no_contact", "contact"))
    parser.add_argument("--expected-position", choices=POSITION_ORDER)
    parser.add_argument("--expected-force", choices=FORCE_ORDER)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / f"live_shadow_validation_{timestamp}",
    )
    parser.add_argument("--set-baseline-first", action="store_true")
    parser.add_argument("--confirm-sensor-released", action="store_true")
    parser.add_argument("--baseline-minimum-frames", type=int, default=30)
    args = parser.parse_args()
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")
    if args.poll_interval_ms < 20:
        parser.error("--poll-interval-ms must be at least 20")
    if args.set_baseline_first and not args.confirm_sensor_released:
        parser.error(
            "--set-baseline-first requires --confirm-sensor-released; "
            "a pressed spectrum must not become the no-contact baseline"
        )
    if args.expected_contact == "no_contact" and (
        args.expected_position is not None or args.expected_force is not None
    ):
        parser.error("no_contact capture cannot have an expected position or level")
    return args


def request_json(url: str, *, method: str = "GET", timeout: float = 15.0) -> dict[str, Any]:
    request = Request(url, method=method, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local API by design
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    baseline_result = None
    if args.set_baseline_first:
        query = urlencode({"minimum_frames": args.baseline_minimum_frames})
        baseline_result = request_json(
            f"{base_url}/api/global_candidate_baseline?{query}", method="POST"
        )

    endpoint = (
        f"{base_url}/api/global_spectrum_frame"
        "?trace_limit=8&include_spectrum=false&include_shadow=true"
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_frame_ids: set[Any] = set()
    deadline = time.monotonic() + args.duration_sec
    while time.monotonic() < deadline:
        try:
            frame = request_json(endpoint)
            latest = frame.get("latest") if isinstance(frame.get("latest"), dict) else {}
            frame_id = latest.get("frame_id", frame.get("frame_id"))
            if frame_id not in seen_frame_ids:
                seen_frame_ids.add(frame_id)
                records.append(
                    flatten_shadow_frame(
                        frame,
                        captured_at=datetime.now(timezone.utc).isoformat(),
                        expected_contact=args.expected_contact,
                        expected_position=args.expected_position,
                        expected_force=args.expected_force,
                    )
                )
        except Exception as exc:  # continue capture while preserving the error trail
            errors.append(f"{type(exc).__name__}: {exc}")
        time.sleep(args.poll_interval_ms / 1000.0)

    summary = write_shadow_validation_artifacts(
        args.output_dir,
        records,
        capture_errors=errors,
        baseline_result=baseline_result,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
