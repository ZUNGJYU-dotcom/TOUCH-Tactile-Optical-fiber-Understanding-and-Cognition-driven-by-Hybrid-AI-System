"""Train the latency-first one-spectrum pipeline as an undeployed shadow artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dynamic_sequence_models import POSITION_ORDER, RESPONSE_ORDER  # noqa: E402
from src.hybrid_spectrum.dynamic_sequence_dataset import (  # noqa: E402
    extract_baseline_relative_frame_features,
    load_dynamic_config,
    load_dynamic_feature_sequences,
)
from src.hybrid_spectrum.dynamic_single_spectrum import (  # noqa: E402
    build_dynamic_single_spectrum_dataset,
)
from src.hybrid_spectrum.dynamic_single_spectrum_adapter import (  # noqa: E402
    DynamicSingleSpectrumShadowAdapter,
    SCHEMA_VERSION,
)
from src.hybrid_spectrum.features import load_peak_windows  # noqa: E402


CONTACT_ORDER = ["no_contact", "contact"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "sense_dynamic_sequence.yaml",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models"
            / "candidates"
            / "dynamic_single_spectrum_fast_candidate_v1.joblib"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-frame-stride", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=240,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    model_path = args.model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_dynamic_config(args.config.resolve())
    sequences = load_dynamic_feature_sequences(config)
    dataset = build_dynamic_single_spectrum_dataset(
        sequences,
        live_frame_stride=args.live_frame_stride,
    )
    contact_values = dataset.spectral_views.reshape(len(dataset.spectral_views), -1)
    contact_labels = dataset.contact_labels
    contact_mask = dataset.stage_labels != "no_contact"
    position_values = dataset.engineered_features[contact_mask]
    position_labels = dataset.position_labels[contact_mask]
    response_values = contact_values[contact_mask]
    response_labels = dataset.stage_labels[contact_mask]

    models = {
        "contact": _extra_trees(args.random_state),
        "position": _extra_trees(args.random_state + 1),
        "response_level": _extra_trees(args.random_state + 2),
    }
    training_seconds = {}
    for task, values, labels in (
        ("contact", contact_values, contact_labels),
        ("position", position_values, position_labels),
        ("response_level", response_values, response_labels),
    ):
        started = time.perf_counter()
        models[task].fit(values, labels)
        training_seconds[task] = time.perf_counter() - started
        models[task].n_jobs = 1

    peak_config_path = Path(config["_peak_config_path"])
    peak_windows = load_peak_windows(peak_config_path)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "status": "offline_shadow_candidate_not_deployed",
        "deployment_ready": False,
        "deployment_blockers": [
            "only_27_independent_DAT_sequences",
            "independent_live_shadow_validation_not_completed",
            "single_spectrum_and_temporal_metrics_not_directly_interchangeable",
            "probabilities_not_calibrated",
        ],
        "models": models,
        "input_type": {
            "contact": "baseline_relative_spectral_multiview_flat",
            "position": "engineered_baseline_relative_features",
            "response_level": "baseline_relative_spectral_multiview_flat",
        },
        "label_order": {
            "contact": CONTACT_ORDER,
            "position": POSITION_ORDER,
            "response_level": RESPONSE_ORDER,
        },
        "peak_windows": peak_windows,
        "frame_feature_names": dataset.feature_names,
        "spectral_view_names": dataset.spectral_view_names,
        "live_frame_stride": int(dataset.live_frame_stride),
        "independent_dat_files": int(len(set(dataset.file_ids.tolist()))),
        "stable_training_frames": int(len(dataset.stage_labels)),
        "training_seconds": training_seconds,
        "split_evidence": "outputs/dynamic_single_spectrum_algorithm_benchmark_20260716_v1",
        "sequence_config_path": str(args.config.resolve()),
        "sequence_config_sha256": _sha256(args.config.resolve()),
        "peak_config_path": str(peak_config_path.resolve()),
        "peak_config_sha256": _sha256(peak_config_path.resolve()),
        "response_level_semantics": "approximate_manual_response_level_not_force_N",
    }
    joblib.dump(bundle, model_path, compress=3)

    sequence = sequences[0]
    sample_index = int(dataset.frame_indices[0])
    adapter = DynamicSingleSpectrumShadowAdapter(model_path)
    prediction = adapter.predict(
        sequence.record.wavelength_nm,
        sequence.record.spectra[sample_index],
        sequence.baseline_spectrum,
    )
    if not prediction.get("ok"):
        raise RuntimeError("saved shadow candidate failed its load/predict smoke test")
    feature_matrix, feature_names, _, _ = extract_baseline_relative_frame_features(
        sequence.record.wavelength_nm,
        sequence.record.spectra[sample_index],
        sequence.baseline_spectrum,
        peak_windows,
    )
    if feature_matrix.shape != (1, len(dataset.feature_names)):
        raise AssertionError("runtime engineered feature shape differs from training")
    if tuple(feature_names) != tuple(dataset.feature_names):
        raise AssertionError("runtime engineered feature names differ from training")

    summary = {
        "model_path": str(model_path),
        "model_size_mb": model_path.stat().st_size / (1024.0 * 1024.0),
        "schema_version": SCHEMA_VERSION,
        "deployment_ready": False,
        "independent_dat_files": bundle["independent_dat_files"],
        "stable_training_frames": bundle["stable_training_frames"],
        "training_seconds": training_seconds,
        "load_predict_smoke_test": prediction,
    }
    (output_dir / "shadow_candidate_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Dynamic single-spectrum fast shadow candidate",
        "",
        f"- Artifact: `{model_path}`",
        f"- Size: {summary['model_size_mb']:.3f} MB.",
        f"- Independent DAT files: {bundle['independent_dat_files']}.",
        f"- Stable training frames: {bundle['stable_training_frames']}.",
        "- Contact and response use baseline-relative 3 x 512 spectral views.",
        "- Position uses 40 engineered peak/intensity/shape features.",
        "- The artifact is not deployed and is not marked deployment-ready.",
        "- It requires independent live shadow validation before replacing the current model.",
        "- Light/normal/hard are approximate manual response levels, not force_N.",
    ]
    (output_dir / "shadow_candidate_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
