from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / "new data"
MODEL_ROOT = PROJECT_ROOT / "outputs" / "ordinary_fbg_all_data_fusion_training_20260731_v1"
DEFAULT_SESSION_ID = "20260731_104454_P22_continuous_px6d_fz_reference_2"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "paper_figures_origin_20260801"

POSITION_ORDER = ["P11", "P21", "P31", "P12", "P22", "P32", "P13", "P23", "P33"]
GRID_ORDER = [["P11", "P21", "P31"], ["P12", "P22", "P32"], ["P13", "P23", "P33"]]

COLORS = {
    "teal": "#167C80",
    "cyan": "#28A8C2",
    "coral": "#D4675D",
    "amber": "#D6A42C",
    "gray": "#6B7F8F",
    "light_gray": "#B8C5CF",
    "ink": "#102236",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-data publication figures in Origin.")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show-origin", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def choose_force_band(frame: pd.DataFrame, target_n: float, count: int = 12) -> pd.Index:
    force = pd.to_numeric(frame["force_fz_n"], errors="coerce")
    return (force - target_n).abs().nsmallest(count).index


def load_representative_session(session_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(session_dir / "frame_summary.csv")
    spectrum = pd.read_csv(session_dir / "spectrum_timeseries.csv")
    if frame.empty or spectrum.empty:
        raise RuntimeError(f"Representative session is empty: {session_dir}")
    if spectrum["capture_index"].nunique() != frame["capture_index"].nunique():
        raise RuntimeError("Frame and spectrum capture counts do not match.")
    return frame, spectrum


def build_spectral_sources(
    frame: pd.DataFrame, spectrum: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    intensity = spectrum.pivot(index="capture_index", columns="point_index", values="intensity_counts").sort_index()
    ratio = spectrum.pivot(
        index="capture_index", columns="point_index", values="normalized_intensity_ratio"
    ).sort_index()
    wavelength = (
        spectrum.groupby("point_index")["wavelength_nm"].median().reindex(intensity.columns).to_numpy(dtype=float)
    )
    baseline_counts = (
        spectrum.groupby("point_index")["baseline_intensity_counts"]
        .median()
        .reindex(intensity.columns)
        .to_numpy(dtype=float)
    )

    indexed_frame = frame.set_index("capture_index").reindex(intensity.index)
    force = pd.to_numeric(indexed_frame["force_fz_n"], errors="coerce").fillna(0.0)
    elapsed = pd.to_numeric(indexed_frame["elapsed_time_sec"], errors="coerce")

    baseline_mask = (force <= 0.05) & (elapsed <= 8.0)
    if int(baseline_mask.sum()) < 8:
        baseline_mask = force <= force.quantile(0.10)

    medium_rows = choose_force_band(indexed_frame.reset_index(), 2.5)
    high_rows = choose_force_band(indexed_frame.reset_index(), 5.0)
    medium_capture = indexed_frame.reset_index().iloc[medium_rows]["capture_index"]
    high_capture = indexed_frame.reset_index().iloc[high_rows]["capture_index"]

    baseline_curve = intensity.loc[baseline_mask].median(axis=0).to_numpy(dtype=float)
    medium_curve = intensity.loc[intensity.index.isin(medium_capture)].median(axis=0).to_numpy(dtype=float)
    high_curve = intensity.loc[intensity.index.isin(high_capture)].median(axis=0).to_numpy(dtype=float)

    prominence = max(100.0, 0.03 * float(np.nanmax(baseline_curve) - np.nanmin(baseline_curve)))
    candidates, props = find_peaks(baseline_curve, prominence=prominence, distance=15)
    if len(candidates) < 9:
        candidates, props = find_peaks(baseline_curve, prominence=100.0, distance=10)
    if len(candidates) < 9:
        raise RuntimeError(f"Only {len(candidates)} spectral peaks were found; nine are required.")
    best = np.argsort(props["prominences"])[-9:]
    peak_indices = np.sort(candidates[best])

    spectral_df = pd.DataFrame(
        {
            "wavelength_nm": wavelength,
            "no_contact_counts": baseline_curve,
            "medium_force_counts": medium_curve,
            "high_force_counts": high_curve,
            "medium_relative_change_percent": 100.0
            * (medium_curve - baseline_curve)
            / np.maximum(np.abs(baseline_curve), 1e-9),
            "high_relative_change_percent": 100.0
            * (high_curve - baseline_curve)
            / np.maximum(np.abs(baseline_curve), 1e-9),
            "high_minus_baseline_counts": high_curve - baseline_curve,
        }
    )
    peak_df = pd.DataFrame(
        {
            "peak_id": [f"FBG{i}" for i in range(1, 10)],
            "point_index": peak_indices,
            "wavelength_nm": wavelength[peak_indices],
            "baseline_intensity_counts": baseline_curve[peak_indices],
        }
    )

    valid_points = np.isfinite(baseline_counts) & (baseline_counts > np.nanmedian(baseline_counts))
    ratio_values = ratio.to_numpy(dtype=float)
    spectral_change = np.sqrt(np.nanmean(np.square(ratio_values[:, valid_points] - 1.0), axis=1)) * 100.0
    dynamics_df = pd.DataFrame(
        {
            "capture_index": intensity.index.to_numpy(dtype=int),
            "elapsed_time_s": elapsed.to_numpy(dtype=float),
            "measured_fz_n": force.to_numpy(dtype=float),
            "spectral_shape_change_rms_percent": spectral_change,
            "sync_offset_ms": pd.to_numeric(indexed_frame["sync_offset_ms"], errors="coerce").to_numpy(dtype=float),
        }
    )

    pearson = float(np.corrcoef(dynamics_df["measured_fz_n"], dynamics_df["spectral_shape_change_rms_percent"])[0, 1])
    spearman = float(spearmanr(dynamics_df["measured_fz_n"], dynamics_df["spectral_shape_change_rms_percent"]).statistic)
    metadata = {
        "baseline_frame_count": int(baseline_mask.sum()),
        "medium_force_median_n": float(force.loc[intensity.index.isin(medium_capture)].median()),
        "high_force_median_n": float(force.loc[intensity.index.isin(high_capture)].median()),
        "spectral_force_pearson_r": pearson,
        "spectral_force_spearman_rho": spearman,
        "spectral_feature_formula": "100 * RMS(normalized_intensity_ratio - 1) over baseline-above-median wavelengths",
        "native_timing_preserved": True,
    }
    return spectral_df, peak_df, dynamics_df, metadata


def build_position_sources(oof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    oof = pd.read_csv(oof_path, low_memory=False)
    position = oof[(oof["task"] == "position") & (oof["model_id"] == "all_sources_temporal_extra_trees")].copy()
    if position.empty:
        raise RuntimeError("Grouped OOF position predictions were not found.")
    counts = confusion_matrix(position["true_label"], position["predicted_label"], labels=POSITION_ORDER)
    support = counts.sum(axis=1, keepdims=True)
    percent = np.divide(counts, support, out=np.zeros_like(counts, dtype=float), where=support > 0) * 100.0
    counts_df = pd.DataFrame(counts, index=POSITION_ORDER, columns=POSITION_ORDER)
    percent_df = pd.DataFrame(percent, index=POSITION_ORDER, columns=POSITION_ORDER)
    meta = {
        "accuracy": float(accuracy_score(position["true_label"], position["predicted_label"])),
        "macro_recall": float(np.mean(np.diag(percent) / 100.0)),
        "test_window_count": int(len(position)),
        "test_session_count": int(position["group_id"].nunique()),
    }
    return counts_df, percent_df, meta


def build_force_sources(gate_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    force = pd.read_csv(gate_path)
    required = ["true_force_n", "gated_force_n", "raw_optical_force_n", "group_id", "fold_id"]
    missing = [column for column in required if column not in force]
    if missing:
        raise RuntimeError(f"Force OOF predictions are missing columns: {missing}")
    force = force.dropna(subset=["true_force_n", "gated_force_n"]).copy()
    force["residual_n"] = force["gated_force_n"] - force["true_force_n"]
    metrics = {
        "mae_n": float(mean_absolute_error(force["true_force_n"], force["gated_force_n"])),
        "rmse_n": float(math.sqrt(mean_squared_error(force["true_force_n"], force["gated_force_n"]))),
        "r2": float(r2_score(force["true_force_n"], force["gated_force_n"])),
        "test_window_count": int(len(force)),
        "test_session_count": int(force["group_id"].nunique()),
        "split_strategy": "grouped_by_session_id",
    }
    return force, metrics


def save_sources(
    source_dir: Path,
    spectral: pd.DataFrame,
    peaks: pd.DataFrame,
    dynamics: pd.DataFrame,
    cm_counts: pd.DataFrame,
    cm_percent: pd.DataFrame,
    force: pd.DataFrame,
) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    spectral.to_csv(source_dir / "fig1_real_spectral_fingerprint.csv", index=False, encoding="utf-8-sig")
    peaks.to_csv(source_dir / "fig1_detected_real_peaks.csv", index=False, encoding="utf-8-sig")
    dynamics.to_csv(source_dir / "fig2_real_synchronized_optical_force.csv", index=False, encoding="utf-8-sig")
    cm_counts.to_csv(source_dir / "fig3_position_confusion_counts_grouped_oof.csv", encoding="utf-8-sig")
    cm_percent.to_csv(source_dir / "fig3_position_confusion_row_percent_grouped_oof.csv", encoding="utf-8-sig")
    force.to_csv(source_dir / "fig4_force_regression_grouped_oof.csv", index=False, encoding="utf-8-sig")


def style_label(label: Any, size: int = 14, color: str | None = None, bold: bool = False) -> None:
    label.set_int("font", 0)
    label.set_int("fsize", int(size))
    label.set_int("underline", 0)
    label.set_int("bold", int(bold))
    label.set_int("u", 0)
    label.set_int("b", int(bold))
    if color:
        label.color = color


def set_data_label(
    label: Any,
    x: float,
    y: float,
    size: int = 14,
    color: str | None = None,
    bold: bool = False,
) -> None:
    label.set_int("attach", 2)
    label.set_float("x1", float(x))
    label.set_float("y1", float(y))
    style_label(label, size=size, color=color, bold=bold)


def remove_legend(layer: Any) -> None:
    legend = layer.label("Legend")
    if legend:
        legend.remove()


def style_axis_titles(layer: Any, names: tuple[str, ...] = ("XB", "YL", "YR")) -> None:
    for name in names:
        label = layer.label(name)
        if label:
            style_label(label, size=18, color=COLORS["ink"])


def style_colorbar(layer: Any) -> None:
    colorbar = layer.label("SPECTRUM1")
    if colorbar:
        style_label(colorbar, size=13, color=COLORS["ink"])
    try:
        layer.lt_exec("SPECTRUM1.font=0; SPECTRUM1.fsize=15; SPECTRUM1.u=0;")
    except Exception:
        pass


def reverse_colormap(plot: Any) -> None:
    try:
        plot.layer.SetNumProp(plot._format_property("cmap.reverse"), 1)
    except Exception as exc:
        print(f"Warning: could not reverse colormap: {exc}")


def export_graph(page: Any, figure_dir: Path, stem: str, width: int = 3600) -> list[str]:
    exported: list[str] = []
    page.lt_exec("page -FLS;")
    for extension, export_type, ratio in [("png", "png", 0), ("svg", "svg", 100), ("emf", "emf", 100)]:
        target = figure_dir / f"{stem}.{extension}"
        try:
            page.save_fig(str(target), type=export_type, replace=True, width=width if extension == "png" else 0, ratio=ratio)
            if target.exists():
                exported.append(str(target))
        except Exception as exc:
            print(f"Warning: could not export {target.name}: {exc}")
    return exported


def add_heatmap_annotations(layer: Any, values: np.ndarray, labels: list[str]) -> None:
    n = len(labels)
    for row in range(n):
        y = n - row
        for col in range(n):
            value = float(values[row, col])
            text_color = "#FFFFFF" if value >= 55.0 else COLORS["ink"]
            text = layer.add_label(f"{value:.1f}", col + 1, y)
            set_data_label(text, col + 1, y, size=10, color=text_color, bold=value >= 95.0)
    for col, label in enumerate(labels, start=1):
        text = layer.add_label(label, col, n + 0.72)
        set_data_label(text, col, n + 0.72, size=10, color=COLORS["ink"], bold=True)
        text.set_float("rotate", 45)
    for row, label in enumerate(labels):
        text = layer.add_label(label, 0.15, n - row)
        set_data_label(text, 0.15, n - row, size=10, color=COLORS["ink"], bold=True)


def validate_figure_inputs(
    spectral: pd.DataFrame,
    peaks: pd.DataFrame,
    dynamics: pd.DataFrame,
    cm_percent: pd.DataFrame,
    force: pd.DataFrame,
    spatial_recall: pd.DataFrame,
) -> None:
    if len(peaks) != 9:
        raise ValueError(f"Expected nine real spectral peaks, found {len(peaks)}.")
    if spectral[["wavelength_nm", "no_contact_counts", "medium_force_counts", "high_force_counts"]].isna().any().any():
        raise ValueError("Figure 1 source contains missing values.")
    if dynamics[["elapsed_time_s", "spectral_shape_change_rms_percent", "measured_fz_n"]].isna().any().any():
        raise ValueError("Figure 2 source contains missing values.")
    if cm_percent.shape != (9, 9) or not np.allclose(cm_percent.sum(axis=1), 100.0, atol=1e-6):
        raise ValueError("Grouped position confusion matrix is not a valid 9x9 row-normalized matrix.")
    if force[["true_force_n", "gated_force_n"]].isna().any().any():
        raise ValueError("Figure 4 source contains missing force values.")
    if spatial_recall.shape != (3, 3):
        raise ValueError("Spatial recall map must be 3x3.")


def build_origin_project(
    output_dir: Path,
    spectral: pd.DataFrame,
    peaks: pd.DataFrame,
    dynamics: pd.DataFrame,
    cm_percent: pd.DataFrame,
    force: pd.DataFrame,
    spatial_recall: pd.DataFrame,
    spectral_meta: dict[str, float],
    position_meta: dict[str, float],
    force_meta: dict[str, float],
    show_origin: bool,
) -> dict[str, list[str]]:
    import originpro as op

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, list[str]] = {}
    try:
        op.set_show(show_origin)
        op.new(asksave=False)

        ws1 = op.new_sheet("w", lname="Fig1_Spectrum")
        ws1.from_df(
            spectral[
                [
                    "wavelength_nm",
                    "no_contact_counts",
                    "medium_relative_change_percent",
                    "high_relative_change_percent",
                ]
            ]
        )
        graph1 = op.new_graph(lname="Fig1_Real_9FBG", template="DoubleY")
        layer1, layer1_delta = graph1[0], graph1[1]
        baseline_plot = layer1.add_plot(ws1, coly=1, colx=0, type="l")
        baseline_plot.color = COLORS["gray"]
        baseline_plot.set_cmd("-w 1.8")
        medium_plot = layer1_delta.add_plot(ws1, coly=2, colx=0, type="l")
        medium_plot.color = COLORS["amber"]
        medium_plot.set_cmd("-w 2.0", "-d 3")
        high_plot = layer1_delta.add_plot(ws1, coly=3, colx=0, type="l")
        high_plot.color = COLORS["coral"]
        high_plot.set_cmd("-w 2.4")
        layer1.rescale()
        layer1_delta.rescale("x")
        layer1.axis("x").title = "Wavelength (nm)"
        layer1.axis("y").title = "No-contact intensity (counts)"
        layer1_delta.axis("y2").title = "Relative spectral change (%)"
        style_axis_titles(layer1, ("XB", "YL"))
        style_axis_titles(layer1_delta, ("YR",))
        remove_legend(layer1)
        remove_legend(layer1_delta)
        layer1.lt_exec("layer.x.inc=20;")
        ymin = float(spectral[["no_contact_counts", "medium_force_counts", "high_force_counts"]].min().min())
        ymax = float(spectral[["no_contact_counts", "medium_force_counts", "high_force_counts"]].max().max())
        yrange = max(1.0, ymax - ymin)
        layer1.set_ylim(max(0.0, ymin - 0.05 * yrange), ymax + 0.22 * yrange)
        for index, peak in peaks.iterrows():
            level = index % 3
            x = float(peak["wavelength_nm"]) + (-0.75 + 0.75 * level)
            y = float(peak["baseline_intensity_counts"]) + (0.055 + 0.065 * level) * yrange
            label = layer1.add_label(str(peak["peak_id"]), x, y)
            set_data_label(label, x, y, size=10, color=COLORS["ink"], bold=True)
        label_x = float(spectral["wavelength_nm"].quantile(0.86))
        delta_low = float(
            min(
                spectral["medium_relative_change_percent"].min(),
                spectral["high_relative_change_percent"].min(),
            )
        )
        delta_high = float(
            max(
                spectral["medium_relative_change_percent"].max(),
                spectral["high_relative_change_percent"].max(),
            )
        )
        delta_margin = max(1.0, 0.08 * (delta_high - delta_low))
        layer1_delta.set_ylim(delta_low - delta_margin, delta_high + 3.2 * delta_margin)
        medium_y = delta_high + 2.25 * delta_margin
        high_y = delta_high + 0.65 * delta_margin
        medium_label = layer1_delta.add_label(f"Change at {spectral_meta['medium_force_median_n']:.2f} N", label_x, medium_y)
        high_label = layer1_delta.add_label(f"Change at {spectral_meta['high_force_median_n']:.2f} N", label_x, high_y)
        set_data_label(medium_label, label_x, medium_y, size=11, color=COLORS["amber"], bold=True)
        set_data_label(high_label, label_x, high_y, size=11, color=COLORS["coral"], bold=True)
        exports["fig1"] = export_graph(graph1, figure_dir, "Fig1_real_9fbg_spectral_fingerprint")

        ws2 = op.new_sheet("w", lname="Fig2_Dynamics")
        ws2.from_df(dynamics[["elapsed_time_s", "spectral_shape_change_rms_percent", "measured_fz_n"]])
        graph2 = op.new_graph(lname="Fig2_Sync_Dynamics", template="DoubleY")
        left, right = graph2[0], graph2[1]
        optical_plot = left.add_plot(ws2, coly=1, colx=0, type="l")
        force_plot = right.add_plot(ws2, coly=2, colx=0, type="l")
        optical_plot.color = COLORS["teal"]
        optical_plot.set_cmd("-w 2.2")
        force_plot.color = COLORS["coral"]
        force_plot.set_cmd("-w 2.2")
        left.rescale()
        right.rescale("x")
        left.axis("x").title = "Time (s)"
        left.axis("y").title = "Spectral shape change, RMS (%)"
        right.axis("y2").title = "Measured Fz (N)"
        style_axis_titles(left, ("XB", "YL"))
        style_axis_titles(right, ("YR",))
        remove_legend(left)
        remove_legend(right)
        t_min = float(dynamics["elapsed_time_s"].min())
        t_max = float(dynamics["elapsed_time_s"].max())
        t_range = max(1.0, t_max - t_min)
        left.set_xlim(t_min, t_max)
        right.set_xlim(t_min, t_max)
        left.lt_exec("layer.x.inc=10;")
        optical_max = float(dynamics["spectral_shape_change_rms_percent"].max())
        force_max = float(dynamics["measured_fz_n"].max())
        optical_label = left.add_label("Full-spectrum change", t_min + 0.14 * t_range, optical_max * 0.91)
        set_data_label(optical_label, t_min + 0.14 * t_range, optical_max * 0.91, size=12, color=COLORS["teal"], bold=True)
        force_label = right.add_label("Measured Fz", t_min + 0.14 * t_range, force_max * 0.91)
        set_data_label(force_label, t_min + 0.14 * t_range, force_max * 0.91, size=12, color=COLORS["coral"], bold=True)
        stats_x = t_min + 0.72 * t_range
        stats_y = optical_max * 0.87
        annotation = left.add_label(
            f"r = {spectral_meta['spectral_force_pearson_r']:.3f} | rho = {spectral_meta['spectral_force_spearman_rho']:.3f}",
            stats_x,
            stats_y,
        )
        set_data_label(annotation, stats_x, stats_y, size=11, color=COLORS["ink"])
        exports["fig2"] = export_graph(graph2, figure_dir, "Fig2_synchronized_optical_force_dynamics")

        matrix3 = op.new_sheet("m", lname="Fig3_CM")
        cm_values = cm_percent.to_numpy(dtype=float)
        matrix3.from_np(cm_values[::-1])
        matrix3.xymap = (1.0, 9.0, 1.0, 9.0)
        graph3 = op.new_graph(lname="Fig3_Position_CM", template="heatmap")
        layer3 = graph3[0]
        heat3 = layer3.add_plot(matrix3, colz=0)
        heat3.colormap = "Viridis.PAL"
        z = heat3.zlevels
        z["minors"] = 0
        z["levels"] = [0, 20, 40, 60, 80, 100]
        heat3.zlevels = z
        layer3.rescale("z")
        layer3.set_xlim(0.0, 10.0)
        layer3.set_ylim(0.0, 10.0)
        layer3.axis("x").title = "Predicted position"
        layer3.axis("y").title = "True position"
        style_axis_titles(layer3, ("XB", "YL"))
        layer3.lt_exec("axis -ps X L 0; axis -ps Y L 0; run.section(Standard,spectrum);")
        try:
            layer3.lt_exec('SPECTRUM1.title$="Recall (%)"; SPECTRUM1.title=1;')
        except Exception:
            pass
        style_colorbar(layer3)
        add_heatmap_annotations(layer3, cm_values, POSITION_ORDER)
        exports["fig3"] = export_graph(graph3, figure_dir, "Fig3_position_recognition_confusion_matrix")

        ws4 = op.new_sheet("w", lname="Fig4_Force")
        ws4.from_df(force[["true_force_n", "gated_force_n", "residual_n", "fold_id"]])
        lo = float(min(force["true_force_n"].min(), force["gated_force_n"].min(), 0.0))
        hi = float(max(force["true_force_n"].max(), force["gated_force_n"].max()))
        margin = max(0.25, 0.04 * (hi - lo))
        edges = np.linspace(lo - margin, hi + margin, 81)
        density, x_edges, y_edges = np.histogram2d(
            force["true_force_n"].to_numpy(dtype=float),
            force["gated_force_n"].to_numpy(dtype=float),
            bins=[edges, edges],
        )
        log_density = np.log1p(density.T)
        pd.DataFrame(
            density,
            index=pd.IntervalIndex.from_breaks(x_edges),
            columns=pd.IntervalIndex.from_breaks(y_edges),
        ).to_csv(output_dir / "source_data" / "fig4_force_regression_density_counts.csv")
        matrix4 = op.new_sheet("m", lname="Fig4_Density")
        matrix4.from_np(log_density)
        matrix4.xymap = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
        graph4 = op.new_graph(lname="Fig4_Force_Regression", template="heatmap")
        layer4 = graph4[0]
        heat4 = layer4.add_plot(matrix4, colz=0)
        heat4.colormap = "GreyBlue.PAL"
        layer4.rescale("z")
        z4 = heat4.zlevels
        z4["minors"] = 0
        positive_density = log_density[log_density > 0]
        density_scale_max = max(1.0, float(np.percentile(positive_density, 99.0)))
        z4["levels"] = np.linspace(0.0, density_scale_max, 6).tolist()
        heat4.zlevels = z4
        layer4.lt_exec("run.section(Standard,spectrum);")
        try:
            layer4.lt_exec('SPECTRUM1.title$="Window density"; SPECTRUM1.title=1;')
        except Exception:
            pass
        style_colorbar(layer4)
        layer4.set_xlim(lo - margin, hi + margin)
        layer4.set_ylim(lo - margin, hi + margin)
        identity = layer4.add_line(lo - margin, lo - margin, hi + margin, hi + margin)
        identity.color = COLORS["coral"]
        identity.width = 2.2
        identity.type = 2
        layer4.axis("x").title = "Measured Fz (N)"
        layer4.axis("y").title = "Optically estimated Fz (N)"
        style_axis_titles(layer4, ("XB", "YL"))
        remove_legend(layer4)
        layer4.lt_exec("legend -d;")
        layer4.lt_exec("layer.x.inc=1; layer.y.inc=1;")
        data_range = max(1.0, (hi + margin) - (lo - margin))
        stats_x = lo - margin + 0.18 * data_range
        stats_y = hi + margin - 0.11 * data_range
        stats4 = layer4.add_label(
            f"MAE = {force_meta['mae_n']:.3f} N\n"
            f"RMSE = {force_meta['rmse_n']:.3f} N\n"
            f"R2 = {force_meta['r2']:.3f}",
            stats_x,
            stats_y,
        )
        set_data_label(stats4, stats_x, stats_y, size=12, color=COLORS["ink"], bold=True)
        exports["fig4"] = export_graph(graph4, figure_dir, "Fig4_optical_force_regression_validation")

        matrix5 = op.new_sheet("m", lname="Fig5_Recall")
        recall_values = spatial_recall.to_numpy(dtype=float)
        matrix5.from_np(recall_values[::-1])
        matrix5.xymap = (1.0, 3.0, 1.0, 3.0)
        graph5 = op.new_graph(lname="Fig5_Spatial_Recall", template="heatmap")
        layer5 = graph5[0]
        heat5 = layer5.add_plot(matrix5, colz=0)
        heat5.colormap = "Viridis.PAL"
        z5 = heat5.zlevels
        z5["minors"] = 0
        z5["levels"] = [90, 92, 94, 96, 98, 100]
        heat5.zlevels = z5
        layer5.rescale("z")
        layer5.set_xlim(0.25, 3.75)
        layer5.set_ylim(0.25, 3.75)
        layer5.lt_exec("axis -ps X L 0; axis -ps Y L 0; run.section(Standard,spectrum);")
        for axis_title in ("XB", "YL", "YR"):
            label = layer5.label(axis_title)
            if label:
                label.remove()
        try:
            layer5.lt_exec('SPECTRUM1.title$="Recall (%)"; SPECTRUM1.title=1;')
        except Exception:
            pass
        style_colorbar(layer5)
        for r, row in enumerate(GRID_ORDER):
            y = 3 - r
            for c, label_text in enumerate(row):
                x = c + 1
                value = float(recall_values[r, c])
                label = layer5.add_label(f"{label_text}\n{value:.1f}%", x, y)
                set_data_label(label, x, y, size=15, color="#FFFFFF" if value >= 97 else COLORS["ink"], bold=True)
        exports["fig5"] = export_graph(graph5, figure_dir, "Fig5_spatial_position_recall_map", width=2600)

        project_path = output_dir / "TOUCH_real_data_publication_figures_20260801.opju"
        op.save(str(project_path))
    finally:
        op.exit()
    return exports


def make_contact_sheet(figure_dir: Path, output_path: Path) -> None:
    files = [
        figure_dir / "Fig1_real_9fbg_spectral_fingerprint.png",
        figure_dir / "Fig2_synchronized_optical_force_dynamics.png",
        figure_dir / "Fig3_position_recognition_confusion_matrix.png",
        figure_dir / "Fig4_optical_force_regression_validation.png",
        figure_dir / "Fig5_spatial_position_recall_map.png",
    ]
    images = [Image.open(path).convert("RGB") for path in files if path.exists()]
    if not images:
        return
    tile_w, tile_h = 1100, 720
    sheet = Image.new("RGB", (tile_w * 2, tile_h * 3), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for index, image in enumerate(images):
        image.thumbnail((tile_w - 60, tile_h - 80), Image.Resampling.LANCZOS)
        x0 = (index % 2) * tile_w + (tile_w - image.width) // 2
        y0 = (index // 2) * tile_h + 48 + (tile_h - 80 - image.height) // 2
        sheet.paste(image, (x0, y0))
        draw.text(((index % 2) * tile_w + 24, (index // 2) * tile_h + 12), f"Figure {index + 1}", fill=COLORS["ink"], font=font)
    sheet.save(output_path, quality=95)


def write_reports(
    output_dir: Path,
    session_dir: Path,
    spectral_meta: dict[str, float],
    position_meta: dict[str, float],
    force_meta: dict[str, float],
    sources: list[Path],
    exports: dict[str, list[str]],
) -> None:
    manifest = {
        "generated_by": "scripts/build_origin_publication_figures.py",
        "origin_version": "Origin 2024 / OriginPro 10.1",
        "data_policy": "real measured data only; no synthetic values in publication metrics",
        "representative_session": session_dir.name,
        "evaluation_validity": "formal grouped-by-session OOF evaluation",
        "source_files": [source_record(path) for path in sources],
        "spectral_dynamics_metrics": spectral_meta,
        "position_metrics": position_meta,
        "force_metrics": force_meta,
        "exports": exports,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    captions = f"""# Publication figure captions

## Figure 1. Real nine-FBG spectral fingerprint under synchronized loading
The no-contact spectrum and force-dependent relative spectral changes from a real P22 session are shown at approximately {spectral_meta['medium_force_median_n']:.2f} N and {spectral_meta['high_force_median_n']:.2f} N. Relative change is computed wavelength by wavelength against the no-contact spectrum. FBG1-FBG9 denote the nine detected spectral peaks in ascending wavelength order; they are not asserted to be a final spatial P11-P33 wavelength map.

## Figure 2. Synchronized optical response and mechanical reference
The optical trace is the RMS change of the full normalized spectrum, while the mechanical trace is the synchronized PX6D Fz reference. Native acquisition timestamps and their physical delay are retained. Pearson r = {spectral_meta['spectral_force_pearson_r']:.3f}; Spearman rho = {spectral_meta['spectral_force_spearman_rho']:.3f}. This single-session panel illustrates response dynamics rather than population-level generalization.

## Figure 3. Grouped out-of-fold tactile position recognition
Row-normalized confusion matrix for nine positions. Evaluation is grouped by session, preventing frames from the same recording from entering both training and test folds. Window accuracy = {position_meta['accuracy']:.3f} over {int(position_meta['test_window_count'])} windows from {int(position_meta['test_session_count'])} independent test sessions.

## Figure 4. Optical estimation of normal force
Measured PX6D Fz is compared with the contact-gated, optical-only estimate for grouped out-of-fold predictions. The heatmap uses all test windows and encodes log-transformed window density; the color scale is clipped at its 99th percentile for legibility while the full unmodified count table remains in `source_data/`. The dashed line denotes identity. MAE = {force_meta['mae_n']:.3f} N, RMSE = {force_meta['rmse_n']:.3f} N, and R2 = {force_meta['r2']:.3f}. PX6D force is supervision and validation reference only; it is not an inference input.

## Figure 5. Spatial distribution of position recall
Per-position recall arranged using the physical 3x3 display convention: top row P11-P21-P31, middle row P12-P22-P32, and bottom row P13-P23-P33.
"""
    (output_dir / "figure_captions.md").write_text(captions, encoding="utf-8")

    readme = f"""# Origin publication figure pack

This pack contains five editable Origin figures generated from real synchronized 9-FBG spectra, PX6D Fz references, and formal grouped out-of-fold predictions.

## Contents
- `TOUCH_real_data_publication_figures_20260801.opju`: editable Origin project.
- `source_data/`: traceable CSV tables used by every figure.
- `figures/`: Origin-exported PNG, SVG, and EMF files.
- `contact_sheet.png`: quick visual index.
- `figure_manifest.json`: input hashes, metrics, and provenance.
- `figure_captions.md`: manuscript-ready caption draft.

## Scientific boundaries
- All plotted spectra and synchronized force traces are real measurements.
- Model evaluation uses session-grouped out-of-fold predictions; random frame splitting is not used.
- FBG1-FBG9 in Figure 1 are wavelength-ordered detected peaks, not a claimed final spatial wavelength assignment.
- The PX6D Fz signal is a calibration reference, not an input to the optical inference model.
- Figure 2 preserves the native optical-mechanical timing offset instead of shifting traces to make them look better.

Representative real session: `{session_dir.name}`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    session_dir = DATA_ROOT / args.session_id
    frame_path = session_dir / "frame_summary.csv"
    spectrum_path = session_dir / "spectrum_timeseries.csv"
    oof_path = MODEL_ROOT / "grouped_oof_predictions.csv"
    gate_path = MODEL_ROOT / "force_contact_gate_oof_predictions.csv"
    for path in [frame_path, spectrum_path, oof_path, gate_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    frame, spectrum = load_representative_session(session_dir)
    spectral_df, peak_df, dynamics_df, spectral_meta = build_spectral_sources(frame, spectrum)
    cm_counts, cm_percent, position_meta = build_position_sources(oof_path)
    force_df, force_meta = build_force_sources(gate_path)

    recall = pd.Series(np.diag(cm_percent.to_numpy()), index=POSITION_ORDER)
    spatial_recall = pd.DataFrame(
        [[recall[label] for label in row] for row in GRID_ORDER],
        index=["top", "middle", "bottom"],
        columns=["left", "center", "right"],
    )
    validate_figure_inputs(spectral_df, peak_df, dynamics_df, cm_percent, force_df, spatial_recall)
    spatial_recall.to_csv(source_dir / "fig5_spatial_position_recall_percent.csv", encoding="utf-8-sig")

    save_sources(source_dir, spectral_df, peak_df, dynamics_df, cm_counts, cm_percent, force_df)
    exports = build_origin_project(
        output_dir,
        spectral_df,
        peak_df,
        dynamics_df,
        cm_percent,
        force_df,
        spatial_recall,
        spectral_meta,
        position_meta,
        force_meta,
        args.show_origin,
    )
    make_contact_sheet(output_dir / "figures", output_dir / "contact_sheet.png")
    write_reports(
        output_dir,
        session_dir,
        spectral_meta,
        position_meta,
        force_meta,
        [frame_path, spectrum_path, oof_path, gate_path],
        exports,
    )
    expected_exports = {
        output_dir / "figures" / f"{stem}.{extension}"
        for stem in [
            "Fig1_real_9fbg_spectral_fingerprint",
            "Fig2_synchronized_optical_force_dynamics",
            "Fig3_position_recognition_confusion_matrix",
            "Fig4_optical_force_regression_validation",
            "Fig5_spatial_position_recall_map",
        ]
        for extension in ["png", "svg", "emf"]
    }
    missing_exports = sorted(str(path) for path in expected_exports if not path.exists() or path.stat().st_size == 0)
    qa = {
        "real_data_only": True,
        "grouped_evaluation": True,
        "nine_real_peaks_detected": len(peak_df) == 9,
        "confusion_rows_sum_to_100": bool(np.allclose(cm_percent.sum(axis=1), 100.0, atol=1e-6)),
        "missing_or_empty_exports": missing_exports,
        "passed": not missing_exports,
    }
    (output_dir / "figure_qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if missing_exports:
        raise RuntimeError(f"Publication export QA failed: {missing_exports}")
    print(json.dumps({"output_dir": str(output_dir), "exports": exports}, indent=2))


if __name__ == "__main__":
    main()
