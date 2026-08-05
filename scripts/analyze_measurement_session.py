"""Analyze one synchronized TOUCH optical/PX6D measurement session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.measurement_consistency import (  # noqa: E402
    analyze_measurement_session,
    load_measurement_config,
    load_measurement_trace,
    write_measurement_artifacts,
)
from src.hybrid_spectrum.measurement_estimate_sources import (  # noqa: E402
    EVIDENCE_SOURCES,
    resolve_measurement_estimate_evidence,
)


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "ordinary_fbg_optical_only_force_candidate.joblib"
)
DEFAULT_PEAK_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_spectrum_channels.yaml"
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "config" / "runtime_contact_state.yaml"


def _load_yaml_section(path: Path, section: str) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = payload.get(section, {}) if isinstance(payload, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "measurement_analysis.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--estimate-source",
        choices=EVIDENCE_SOURCES,
        default="best_available",
        help=(
            "Force-estimate evidence. best_available prefers grouped OOF, "
            "then current-model replay, then recorded runtime."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--peak-config",
        type=Path,
        default=DEFAULT_PEAK_CONFIG_PATH,
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG_PATH,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "measurement_sessions" / args.session_dir.name
    )
    analysis_config = load_measurement_config(args.config)
    trace_rows = load_measurement_trace(args.session_dir)
    evidence = resolve_measurement_estimate_evidence(
        args.session_dir,
        trace_rows,
        args.estimate_source,
        outputs_root=args.outputs_root,
        model_path=args.model_path,
        peak_config_path=args.peak_config,
        runtime_recovery_config=_load_yaml_section(
            args.runtime_config,
            "runtime_baseline_recovery",
        ),
        runtime_gate_config=_load_yaml_section(
            args.runtime_config,
            "all_source_runtime_gate",
        ),
        baseline_frame_count=analysis_config.replay_baseline_frame_count,
        baseline_strategy=analysis_config.replay_baseline_strategy,
        baseline_minimum_stable_frames=(
            analysis_config.replay_baseline_minimum_stable_frames
        ),
        baseline_stability_mad_multiplier=(
            analysis_config.replay_baseline_stability_mad_multiplier
        ),
    )
    if not evidence.get("ok"):
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 2
    result = analyze_measurement_session(
        args.session_dir,
        analysis_config,
        estimate_overlay=evidence.get("overlay"),
        estimate_source_info=evidence,
    )
    write_measurement_artifacts(result, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "requested_estimate_source": args.estimate_source,
                "selected_estimate_source": evidence.get("source"),
                "evaluation_validity": evidence.get("evaluation_validity"),
                "comparison_status": result["summary"]["data"][
                    "comparison_status"
                ],
                "paired_count": result["summary"]["data"]["paired_count"],
                "cycle_count": result["summary"]["repeatability"]["cycle_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
