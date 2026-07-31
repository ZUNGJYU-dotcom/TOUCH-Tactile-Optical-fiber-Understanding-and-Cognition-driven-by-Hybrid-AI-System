"""Build the unified ordinary-FBG optical dataset with masked Fz supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.all_source_fusion import (  # noqa: E402
    build_all_source_dataset,
    load_fusion_config,
    resolve_project_path,
    save_all_source_dataset,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "ordinary_fbg_all_data_fusion.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "ordinary_fbg_all_data_fusion_20260731_v1"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = load_fusion_config(config_path)
    dataset = build_all_source_dataset(PROJECT_ROOT, config)
    protected_model = resolve_project_path(
        PROJECT_ROOT, config["paths"]["protected_deployed_model"]
    )
    result = save_all_source_dataset(
        dataset,
        output_dir,
        config_path=config_path,
        protected_model_path=protected_model,
    )
    result["ok"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
