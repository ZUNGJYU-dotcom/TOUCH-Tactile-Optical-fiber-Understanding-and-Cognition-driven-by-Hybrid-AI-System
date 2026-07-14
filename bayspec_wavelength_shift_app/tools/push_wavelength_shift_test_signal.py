"""Push a synthetic P22 spectrum whose Bragg peak moves over time.

This is an API smoke source, not a substitute for BaySpec hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request


LAMBDA0_NM = 1546.89


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def spectrum(center_nm: float) -> tuple[list[float], list[float]]:
    wavelengths: list[float] = []
    counts: list[float] = []
    for index in range(361):
        wavelength = 1546.0 + index * 0.005
        value = 900.0 + 30000.0 * math.exp(-0.5 * ((wavelength - center_nm) / 0.055) ** 2)
        wavelengths.append(round(wavelength, 6))
        counts.append(round(value, 3))
    return wavelengths, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8630")
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--set-baseline", action="store_true")
    args = parser.parse_args()

    ingest_url = f"{args.base_url.rstrip('/')}/api/ingest"
    baseline_url = f"{args.base_url.rstrip('/')}/api/baseline"
    started = time.time()
    baseline_set = False
    sample_index = 0
    period = 1.0 / max(args.rate_hz, 0.1)

    while time.time() - started < args.duration_sec:
        elapsed = time.time() - started
        envelope = 0.0 if elapsed < 2.0 else max(0.0, math.sin((elapsed - 2.0) * 0.8)) ** 2
        shift_pm = 320.0 * envelope
        center_nm = LAMBDA0_NM + shift_pm / 1000.0
        wavelengths, counts = spectrum(center_nm)
        result = post_json(
            ingest_url,
            {
                "source": "synthetic_wavelength_shift_smoke",
                "channels": [
                    {
                        "channel_id": "P22",
                        "wavelength_nm": wavelengths,
                        "intensity": counts,
                        "intensity_counts": max(counts),
                        "peak_wavelength_nm": center_nm,
                        "integration_ms": 40.0,
                    }
                ],
            },
        )
        if args.set_baseline and not baseline_set and elapsed >= 1.0:
            post_json(baseline_url, {"channel_id": "P22"})
            baseline_set = True
        if sample_index % max(1, int(args.rate_hz)) == 0:
            record = (result.get("records") or [{}])[0]
            print(
                f"t={elapsed:5.2f}s lambdaB={record.get('tracked_wavelength_nm')} "
                f"delta={record.get('delta_wavelength_pm')} pm level={record.get('response_level')}"
            )
        sample_index += 1
        time.sleep(period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
