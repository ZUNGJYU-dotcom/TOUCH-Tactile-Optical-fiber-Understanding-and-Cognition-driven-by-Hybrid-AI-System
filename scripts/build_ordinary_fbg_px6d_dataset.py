"""Build a grouped ordinary-FBG spectrum and PX6D force dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.px6d_session_dataset import (  # noqa: E402
    build_dataset,
    load_config,
    save_dataset,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ordinary_fbg_px6d_training.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe ordinary-FBG full-spectrum features with "
            "continuous PX6D Fz targets."
        )
    )
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--qa-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-role",
        choices=("primary", "challenge", "all"),
        default="primary",
        help=(
            "Build the latest-session primary set, the isolated earlier-session "
            "manual-review quarantine, or an all-session debug set."
        ),
    )
    args = parser.parse_args()

    capture_root = args.capture_root.expanduser().resolve()
    qa_summary = args.qa_summary.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = load_config(config_path)
    dataset = build_dataset(
        capture_root,
        config,
        qa_summary_path=qa_summary,
        selection_role=args.selection_role,
    )
    payload = save_dataset(
        dataset,
        output_dir,
        source_root=capture_root,
        config_path=config_path,
        qa_summary_path=qa_summary,
    )
    payload["ok"] = True
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
