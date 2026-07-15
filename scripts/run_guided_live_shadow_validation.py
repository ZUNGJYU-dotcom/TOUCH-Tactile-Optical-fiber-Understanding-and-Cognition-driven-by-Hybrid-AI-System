"""Guide an operator through independent live position/level validation trials.

The deployed model is never modified. The v7 candidate and its temporal vote
are requested explicitly from the local API and written as shadow evidence.
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

from src.hybrid_spectrum.guided_live_validation import (  # noqa: E402
    LEVEL_ORDER,
    POSITION_ORDER,
    build_trial_plan,
    write_guided_validation_artifacts,
)
from src.hybrid_spectrum.live_shadow_validation import flatten_shadow_frame  # noqa: E402


def _csv_choice(value: str, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(parsed) - set(allowed))
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated subset of {allowed}; unknown={unknown}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8640")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order", choices=("randomized", "blocked"), default="randomized")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--positions",
        default=POSITION_ORDER,
        type=lambda value: _csv_choice(value, POSITION_ORDER, "positions"),
    )
    parser.add_argument(
        "--levels",
        default=LEVEL_ORDER,
        type=lambda value: _csv_choice(value, LEVEL_ORDER, "levels"),
    )
    parser.add_argument("--contact-sec", type=float, default=4.0)
    parser.add_argument("--release-sec", type=float, default=2.5)
    parser.add_argument("--poll-interval-ms", type=int, default=80)
    parser.add_argument("--baseline-minimum-frames", type=int, default=30)
    parser.add_argument("--baseline-timeout-sec", type=float, default=90.0)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--load-calibration-to-shadow", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / f"guided_live_shadow_validation_{stamp}",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.contact_sec <= 0 or args.release_sec <= 0:
        parser.error("capture durations must be positive")
    if args.poll_interval_ms < 20:
        parser.error("--poll-interval-ms must be at least 20")
    if args.start_at < 1:
        parser.error("--start-at must be positive")
    return args


def request_json(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 20.0,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, method=method, headers=headers, data=data)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local API
        return json.loads(response.read().decode("utf-8"))


def wait_for_clean_baseline(
    base_url: str,
    *,
    minimum_frames: int,
    timeout_sec: float,
) -> dict[str, Any]:
    request_json(f"{base_url}/api/reset?keep_baseline=false", method="POST")
    deadline = time.monotonic() + timeout_sec
    result: dict[str, Any] = {"ok": False, "reason": "not_attempted"}
    query = urlencode({"minimum_frames": minimum_frames})
    while time.monotonic() < deadline:
        result = request_json(
            f"{base_url}/api/global_candidate_baseline?{query}",
            method="POST",
            timeout=30.0,
        )
        if result.get("ok"):
            return result
        time.sleep(1.0)
    raise RuntimeError(
        "clean current-session baseline was not accepted: "
        + str(result.get("reason") or result)
    )


def capture_stage(
    *,
    endpoint: str,
    duration_sec: float,
    minimum_unique_frames: int,
    seen_frame_ids: set[Any],
    trial: dict[str, Any],
    phase: str,
    poll_interval_ms: int,
    errors: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration_sec
    hard_deadline = deadline + max(5.0, duration_sec)
    while (time.monotonic() < deadline or len(records) < minimum_unique_frames) and (
        time.monotonic() < hard_deadline
    ):
        try:
            frame = request_json(endpoint)
            latest = frame.get("latest") if isinstance(frame.get("latest"), dict) else {}
            frame_id = latest.get("frame_id", frame.get("frame_id"))
            if frame_id is not None and frame_id not in seen_frame_ids:
                seen_frame_ids.add(frame_id)
                is_contact = phase == "contact"
                record = flatten_shadow_frame(
                    frame,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    expected_contact="contact" if is_contact else "no_contact",
                    expected_position=trial["position"] if is_contact else None,
                    expected_force=trial["force_level"] if is_contact else None,
                )
                record.update(
                    {
                        "trial_index": trial["trial_index"],
                        "trial_id": trial["trial_id"],
                        "repeat": trial["repeat"],
                        "trial_role": trial["trial_role"],
                        "phase": phase,
                        "trial_position": trial["position"],
                        "trial_force_level": trial["force_level"],
                    }
                )
                records.append(record)
        except Exception as exc:
            errors.append(
                f"{trial['trial_id']}:{phase}:{type(exc).__name__}: {exc}"
            )
        time.sleep(poll_interval_ms / 1000.0)
    return records


def main() -> int:
    args = parse_args()
    plan = build_trial_plan(
        repeats=args.repeats,
        positions=args.positions,
        levels=args.levels,
        order=args.order,
        seed=args.seed,
    )
    run_plan = [trial for trial in plan if int(trial["trial_index"]) >= args.start_at]
    metadata: dict[str, Any] = {
        "base_url": args.base_url,
        "order": args.order,
        "seed": args.seed,
        "repeats": args.repeats,
        "contact_sec": args.contact_sec,
        "release_sec": args.release_sec,
        "mode": "plan_only" if args.plan_only else "interactive_live_capture",
        "load_calibration_to_shadow": args.load_calibration_to_shadow,
    }
    if args.plan_only:
        summary = write_guided_validation_artifacts(
            args.output_dir,
            trial_plan=plan,
            records=[],
            run_metadata=metadata,
        )
        print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2))
        return 0

    base_url = args.base_url.rstrip("/")
    health = request_json(f"{base_url}/api/health")
    sdk = request_json(f"{base_url}/api/sdk/status").get("sdk_live", {})
    candidate = (
        health.get("trained_static_spectral_model", {}).get("shadow_candidate", {})
    )
    metadata.update(
        {
            "app": health.get("app"),
            "candidate_id": candidate.get("candidate_id"),
            "candidate_hash": candidate.get("model_bundle_sha256"),
            "candidate_runtime_role": candidate.get("runtime_role"),
        }
    )
    if not sdk.get("active") or sdk.get("freshness") != "live":
        raise RuntimeError("BaySpec SDK must already be active and live")

    print("\nGuided static-spectrum validation")
    print("- 9 approximate fingertip positions, not independent force pixels")
    print("- light/normal/hard are manual response levels, not force_N")
    print("- v7 remains shadow-only and cannot drive the digital twin")
    confirmation = input(
        "\nRelease the sensor completely, wait for recovery, then type RELEASED: "
    ).strip()
    if confirmation != "RELEASED":
        raise RuntimeError("baseline cancelled because RELEASED was not confirmed")

    baseline_result = wait_for_clean_baseline(
        base_url,
        minimum_frames=args.baseline_minimum_frames,
        timeout_sec=args.baseline_timeout_sec,
    )
    metadata["baseline_status"] = baseline_result.get(
        "static_model_spectrum_baseline", {}
    ).get("status")
    print(f"Baseline accepted: {metadata['baseline_status']}")

    endpoint = (
        f"{base_url}/api/global_spectrum_frame"
        "?trace_limit=8&include_spectrum=false&include_shadow=true"
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_frame_ids: set[Any] = set()
    calibration_loaded = False
    try:
        for trial in run_plan:
            print(
                f"\n[{trial['trial_index']}/{len(plan)}] "
                f"{trial['position']} / {trial['force_level']}"
            )
            input("Release fully, wait for recovery, then press Enter: ")
            records.extend(
                capture_stage(
                    endpoint=endpoint,
                    duration_sec=args.release_sec,
                    minimum_unique_frames=3,
                    seen_frame_ids=seen_frame_ids,
                    trial=trial,
                    phase="pre_release",
                    poll_interval_ms=args.poll_interval_ms,
                    errors=errors,
                )
            )
            input(
                f"Press and hold {trial['position']} at {trial['force_level']} level; "
                "press Enter when stable: "
            )
            records.extend(
                capture_stage(
                    endpoint=endpoint,
                    duration_sec=args.contact_sec,
                    minimum_unique_frames=5,
                    seen_frame_ids=seen_frame_ids,
                    trial=trial,
                    phase="contact",
                    poll_interval_ms=args.poll_interval_ms,
                    errors=errors,
                )
            )
            input("Release completely, then press Enter: ")
            records.extend(
                capture_stage(
                    endpoint=endpoint,
                    duration_sec=args.release_sec,
                    minimum_unique_frames=3,
                    seen_frame_ids=seen_frame_ids,
                    trial=trial,
                    phase="post_release",
                    poll_interval_ms=args.poll_interval_ms,
                    errors=errors,
                )
            )
            partial_summary = write_guided_validation_artifacts(
                args.output_dir,
                trial_plan=plan,
                records=records,
                baseline_result=baseline_result,
                capture_errors=errors,
                run_metadata=metadata,
            )
            calibration_status = partial_summary.get("session_level_calibration", {})
            if (
                args.load_calibration_to_shadow
                and not calibration_loaded
                and calibration_status.get("ok")
            ):
                calibration_payload = json.loads(
                    (args.output_dir / "session_level_calibration_candidate.json").read_text(
                        encoding="utf-8"
                    )
                )
                load_result = request_json(
                    f"{base_url}/api/shadow/session_level_calibration",
                    method="POST",
                    payload={
                        "calibration": calibration_payload,
                        "source": str(args.output_dir),
                        "trial_count": calibration_status.get("trial_sample_count"),
                    },
                )
                if not load_result.get("ok"):
                    raise RuntimeError(
                        "shadow session calibration load failed: " + str(load_result)
                    )
                calibration_loaded = True
                metadata["shadow_session_calibration_loaded"] = True
                print("Current-session level calibration loaded in shadow mode.")
    except KeyboardInterrupt:
        errors.append("capture_interrupted_by_operator")

    summary = write_guided_validation_artifacts(
        args.output_dir,
        trial_plan=plan,
        records=records,
        baseline_result=baseline_result,
        capture_errors=errors,
        run_metadata=metadata,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
