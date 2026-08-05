"""Merge audited PX6D collection summaries without changing raw captures."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--session-prefix",
        action="append",
        default=[],
        help="Optional session-id prefix; repeat to keep multiple batches.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_summaries(
    paths: list[Path], prefixes: tuple[str, ...]
) -> dict[str, Any]:
    by_session: dict[str, dict[str, Any]] = {}
    source_files: list[str] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        source_files.append(str(resolved))
        for row in payload.get("results") or []:
            session_id = str(row.get("session_id") or "")
            if not session_id:
                continue
            if prefixes and not session_id.startswith(prefixes):
                continue
            previous = by_session.get(session_id)
            if previous is not None and previous != row:
                raise ValueError(
                    f"conflicting QA records for session {session_id}"
                )
            by_session[session_id] = row
    if not by_session:
        raise ValueError("no QA records matched the requested session prefixes")
    results = [by_session[key] for key in sorted(by_session)]
    statuses = Counter(str(row.get("qa_status") or "not_audited") for row in results)
    dates = Counter(str(row["session_id"])[:8] for row in results)
    return {
        "schema_version": "ordinary_fbg_px6d_collection_audit_merged_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary_files": source_files,
        "session_prefix_filter": list(prefixes),
        "session_count": len(results),
        "unique_session_count": len(results),
        "pass_count": int(statuses.get("pass", 0)),
        "warning_count": int(statuses.get("usable_with_warning", 0)),
        "fail_count": int(statuses.get("fail", 0)),
        "session_count_by_date": dict(sorted(dates.items())),
        "formal_split_requirement": "grouped_by_session_id",
        "formal_group_field": "formal_group_id",
        "force_target": "continuous_force_fz_n",
        "results": results,
    }


def main() -> int:
    args = parse_args()
    prefixes = tuple(str(value) for value in args.session_prefix)
    payload = merge_summaries(args.input, prefixes)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
