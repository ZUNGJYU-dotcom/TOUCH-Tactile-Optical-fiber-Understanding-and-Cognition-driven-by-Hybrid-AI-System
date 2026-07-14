"""Discover the current real nine-FBG cluster from repeated full spectra.

This workflow intentionally separates spectral feature discovery from physical
channel approval. Wavelength-order channel IDs are candidates until a labelled
P11-P33 press sequence confirms the fabrication order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
import yaml


CHANNEL_ORDER = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"]


def _odd_window(points: int, requested: int, minimum: int = 5) -> int:
    value = min(requested, points - 1 if points % 2 == 0 else points)
    value = max(minimum, value)
    if value % 2 == 0:
        value -= 1
    return max(3, value)


def load_long_spectra(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: dict[int, list[tuple[int, float, float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = int(row["frame_id"])
            frames.setdefault(frame_id, []).append(
                (
                    int(row["sample_index"]),
                    float(row["wavelength_nm"]),
                    float(row["intensity_counts"]),
                )
            )
    if not frames:
        raise ValueError(f"No spectrum frames found in {path}")
    ordered_frames = []
    wavelength = None
    for frame_id in sorted(frames):
        rows = sorted(frames[frame_id], key=lambda item: item[0])
        current_wavelength = np.asarray([item[1] for item in rows], dtype=float)
        counts = np.asarray([item[2] for item in rows], dtype=float)
        if wavelength is None:
            wavelength = current_wavelength
        if len(current_wavelength) != len(wavelength) or not np.allclose(
            current_wavelength, wavelength, atol=1e-9, rtol=0
        ):
            continue
        ordered_frames.append(counts)
    if wavelength is None or not ordered_frames:
        raise ValueError("No wavelength-grid-consistent frames found")
    return wavelength, np.vstack(ordered_frames), np.asarray(sorted(frames)[: len(ordered_frames)])


def parabolic_center(wavelength: np.ndarray, counts: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(counts) - 1:
        return float(wavelength[index])
    x = wavelength[index - 1 : index + 2]
    y = counts[index - 1 : index + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if not np.isfinite(a) or not np.isfinite(b) or a >= 0 or abs(a) < 1e-12:
        return float(wavelength[index])
    center = float(-b / (2.0 * a))
    return center if float(x[0]) <= center <= float(x[-1]) else float(wavelength[index])


def select_nine_peak_cluster(
    wavelength: np.ndarray,
    average: np.ndarray,
    start_nm: float,
    stop_nm: float,
    expected_first_peak_nm: float = 1528.0,
    first_peak_tolerance_nm: float = 1.5,
) -> tuple[list[int], dict]:
    step_nm = float(np.median(np.diff(wavelength)))
    broad_window = _odd_window(len(average), int(round(10.0 / step_nm)))
    local_window = _odd_window(len(average), 5)
    broad = savgol_filter(average, broad_window, 2, mode="interp")
    residual = average - broad
    smoothed = savgol_filter(residual, local_window, 2, mode="interp")
    raw_smoothed = savgol_filter(average, local_window, 2, mode="interp")
    positive = smoothed[smoothed > 0]
    prominence = max(500.0, float(np.percentile(positive, 35)) if positive.size else 500.0)
    distance_points = max(1, int(round(2.8 / step_nm)))
    all_indices, properties = find_peaks(smoothed, prominence=prominence, distance=distance_points)
    widths = peak_widths(smoothed, all_indices, rel_height=0.5)[0] * step_nm
    raw_prominence = max(1000.0, float(np.ptp(raw_smoothed)) * 0.025)
    raw_indices, _ = find_peaks(
        raw_smoothed,
        prominence=raw_prominence,
        distance=distance_points,
    )
    raw_gate_points = max(2, int(round(0.45 / step_nm)))

    candidate_by_raw_index: dict[int, dict] = {}
    for order, index in enumerate(all_indices):
        if raw_indices.size == 0:
            continue
        nearest_raw = int(raw_indices[np.argmin(np.abs(raw_indices - index))])
        if abs(nearest_raw - int(index)) > raw_gate_points:
            continue
        wl = float(wavelength[nearest_raw])
        width_nm = float(widths[order])
        if start_nm <= wl <= stop_nm and 0.25 <= width_nm <= 2.0:
            candidate = {
                "index": nearest_raw,
                "wavelength_nm": wl,
                "width_nm": width_nm,
                "prominence": float(properties["prominences"][order]),
            }
            previous = candidate_by_raw_index.get(nearest_raw)
            if previous is None or candidate["prominence"] > previous["prominence"]:
                candidate_by_raw_index[nearest_raw] = candidate
    candidates = list(candidate_by_raw_index.values())
    candidates.sort(key=lambda item: item["wavelength_nm"])
    if len(candidates) < 9:
        raise ValueError(
            f"Only {len(candidates)} narrow stable candidates found between "
            f"{start_nm:.3f} and {stop_nm:.3f} nm"
        )

    best = None
    for offset in range(len(candidates) - 8):
        group = candidates[offset : offset + 9]
        centers = np.asarray([item["wavelength_nm"] for item in group])
        first_peak_offset_nm = abs(float(centers[0]) - expected_first_peak_nm)
        if first_peak_offset_nm > first_peak_tolerance_nm:
            continue
        spacing = np.diff(centers)
        if np.any(spacing < 2.5) or np.any(spacing > 5.5):
            continue
        spacing_cv = float(np.std(spacing) / max(np.mean(spacing), 1e-12))
        prominence_score = float(np.mean([item["prominence"] for item in group]))
        anchor_penalty = first_peak_offset_nm / max(first_peak_tolerance_nm, 1e-12)
        score = prominence_score / (1.0 + 8.0 * spacing_cv + anchor_penalty)
        if best is None or score > best[0]:
            best = (score, group, spacing_cv, first_peak_offset_nm)
    if best is None:
        raise ValueError(
            "No physically continuous nine-peak cluster satisfied the spacing "
            f"and {expected_first_peak_nm:.3f} nm first-peak anchor rules"
        )
    group = best[1]
    audit = {
        "wavelength_step_nm": step_nm,
        "broad_background_window_points": broad_window,
        "minimum_prominence_counts": prominence,
        "all_detected_peak_count": int(len(all_indices)),
        "raw_local_maximum_count": int(len(raw_indices)),
        "narrow_candidates_in_search_band": int(len(candidates)),
        "cluster_spacing_nm": [float(value) for value in np.diff([item["wavelength_nm"] for item in group])],
        "cluster_spacing_cv": float(best[2]),
        "expected_first_peak_nm": float(expected_first_peak_nm),
        "first_peak_tolerance_nm": float(first_peak_tolerance_nm),
        "selected_first_peak_offset_nm": float(best[3]),
        "search_start_nm": start_nm,
        "search_stop_nm": stop_nm,
    }
    return [item["index"] for item in group], {"broad": broad, "smoothed": smoothed, "audit": audit}


def analyze_peak(
    wavelength: np.ndarray,
    frames: np.ndarray,
    average: np.ndarray,
    candidate_index: int,
) -> dict:
    center_guess = float(wavelength[candidate_index])
    local = np.flatnonzero(np.abs(wavelength - center_guess) <= 1.2)
    if local.size < 3:
        raise ValueError(f"Insufficient local samples near {center_guess:.4f} nm")
    frame_centers = []
    frame_heights = []
    frame_areas = []
    for counts in frames:
        local_index = int(local[np.argmax(counts[local])])
        center = parabolic_center(wavelength, counts, local_index)
        local_counts = counts[local]
        local_baseline = float(np.percentile(local_counts, 10))
        corrected = np.clip(local_counts - local_baseline, 0.0, None)
        frame_centers.append(center)
        frame_heights.append(float(counts[local_index]))
        frame_areas.append(float(np.trapezoid(corrected, wavelength[local])))

    average_local_index = int(local[np.argmax(average[local])])
    average_center = parabolic_center(wavelength, average, average_local_index)
    average_local = average[local]
    local_baseline = float(np.percentile(average_local, 10))
    corrected = np.clip(average_local - local_baseline, 0.0, None)
    centroid = float(np.sum(wavelength[local] * corrected) / max(np.sum(corrected), 1e-12))
    peak_height = float(average[average_local_index])
    half_height = local_baseline + 0.5 * (peak_height - local_baseline)
    above = wavelength[local][average_local >= half_height]
    fwhm = float(above[-1] - above[0]) if above.size >= 2 else math.nan
    centers = np.asarray(frame_centers)
    heights = np.asarray(frame_heights)
    areas = np.asarray(frame_areas)
    return {
        "sample_peak_wavelength_nm": center_guess,
        "recommended_wavelength_nm": float(np.median(centers)),
        "average_parabolic_wavelength_nm": average_center,
        "average_weighted_centroid_nm": centroid,
        "wavelength_std_pm": float(np.std(centers) * 1000.0),
        "wavelength_p2p_pm": float((np.max(centers) - np.min(centers)) * 1000.0),
        "mean_peak_height_counts": float(np.mean(heights)),
        "peak_height_std_counts": float(np.std(heights)),
        "peak_height_cv": float(np.std(heights) / max(np.mean(heights), 1e-12)),
        "mean_local_area_counts_nm": float(np.mean(areas)),
        "local_area_cv": float(np.std(areas) / max(np.mean(areas), 1e-12)),
        "average_fwhm_nm": fwhm,
        "valid_frame_count": int(len(centers)),
    }


def write_outputs(
    output_dir: Path,
    wavelength: np.ndarray,
    frames: np.ndarray,
    peak_indices: list[int],
    helper: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    average = np.mean(frames, axis=0)
    rows = []
    for channel_id, index in zip(CHANNEL_ORDER, peak_indices):
        row = {"channel_id": channel_id, **analyze_peak(wavelength, frames, average, index)}
        row.update(
            {
                "mapping_basis": "ascending_fabrication_order_candidate",
                "approval_status": "pending_labelled_point_press_confirmation",
                "config_applied": False,
            }
        )
        rows.append(row)

    with (output_dir / "live_nine_fbg_peak_candidates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "profile": "current_real_9fbg_candidate",
        "status": "candidate_pending_labelled_point_press_confirmation",
        "source": "100-frame live BaySpec no-contact spectrum",
        "channel_order_basis": "ascending fabrication order candidate",
        "do_not_apply_automatically": True,
        "channels": {
            row["channel_id"]: {
                "enabled": False,
                "candidate_measured_wavelength_nm": round(row["recommended_wavelength_nm"], 6),
                "wavelength_std_pm": round(row["wavelength_std_pm"], 3),
                "approval_status": row["approval_status"],
            }
            for row in rows
        },
    }
    (output_dir / "candidate_nine_fbg_channel_map.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "cluster_selection_audit.json").write_text(
        json.dumps(helper["audit"], indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(15, 6.8), dpi=160)
    ax.plot(wavelength, average, color="#2369a8", linewidth=1.6, label="100-frame mean spectrum")
    ax.plot(wavelength, helper["broad"], color="#8b98a5", linewidth=1.0, alpha=0.75, label="broad background")
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(rows)))
    for color, row in zip(colors, rows):
        x = row["recommended_wavelength_nm"]
        y = float(np.interp(x, wavelength, average))
        ax.scatter([x], [y], s=38, color=color, zorder=4)
        ax.annotate(
            f"{row['channel_id']}\n{x:.3f} nm",
            (x, y),
            xytext=(0, 16 if CHANNEL_ORDER.index(row["channel_id"]) % 2 == 0 else 31),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#16324a",
        )
    ax.axvspan(1526.5, 1561.5, color="#d9f0f2", alpha=0.22, label="user-confirmed FBG cluster band")
    ax.set_title("Live BaySpec spectrum: candidate real nine-FBG cluster")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Optical intensity (counts)")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "live_nine_fbg_peak_candidates.png", bbox_inches="tight")
    plt.close(fig)

    p22 = next(row for row in rows if row["channel_id"] == "P22")
    lines = [
        "# Live nine-FBG cluster discovery",
        "",
        "- Input: 100 live no-contact BaySpec spectra, 512 wavelength-calibrated samples each.",
        "- User-confirmed physical clue: the real FBG peak cluster starts near 1528 nm.",
        "- Method: broad-background removal, narrow-peak filtering, continuous nine-peak spacing check, then per-frame local parabolic tracking.",
        "- Mapping status: wavelength-order candidate only; P11-P33 identity still requires labelled point-press confirmation.",
        "- No measured force or force_N is produced.",
        "",
        "## Candidate wavelengths",
        "",
        "| Channel candidate | Wavelength (nm) | Stability std (pm) | Peak height CV | Area CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['channel_id']} | {row['recommended_wavelength_nm']:.6f} | "
            f"{row['wavelength_std_pm']:.3f} | {row['peak_height_cv']:.4f} | {row['local_area_cv']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Important interpretation",
            "",
            f"The fifth wavelength-order candidate (P22 candidate) is {p22['recommended_wavelength_nm']:.6f} nm, not the old provisional 1546.89 nm value.",
            "Because real pressing changes wavelength, amplitude, area, and spectral shape together, the final channel recognizer must use labelled full-spectrum fingerprints rather than wavelength alone.",
            "This candidate map is intentionally not applied to production configuration.",
        ]
    )
    (output_dir / "live_nine_fbg_peak_discovery_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster-start-nm", type=float, default=1526.5)
    parser.add_argument("--cluster-stop-nm", type=float, default=1561.5)
    parser.add_argument("--expected-first-peak-nm", type=float, default=1528.0)
    parser.add_argument("--first-peak-tolerance-nm", type=float, default=1.5)
    args = parser.parse_args()
    wavelength, frames, _frame_ids = load_long_spectra(args.input)
    average = np.mean(frames, axis=0)
    peak_indices, helper = select_nine_peak_cluster(
        wavelength,
        average,
        args.cluster_start_nm,
        args.cluster_stop_nm,
        args.expected_first_peak_nm,
        args.first_peak_tolerance_nm,
    )
    write_outputs(args.output, wavelength, frames, peak_indices, helper)
    print(
        json.dumps(
            {
                "ok": True,
                "input_frames": int(frames.shape[0]),
                "spectrum_points": int(frames.shape[1]),
                "candidate_peak_count": len(peak_indices),
                "output_dir": str(args.output),
                "config_applied": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
