#!/usr/bin/env python3
"""Apply bounded post-H1 field-day cleanup to the active calendar assignment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_INPUT_JSON = YEAR_DIR / "checkpoints" / "fd04a-fd19c-calendar-assignment-2026-05-13.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "post-h1-cleanup-calendar-assignment-2026-05-13.json"
DEFAULT_REPORT_JSON = YEAR_DIR / "checkpoints" / "post-h1-cleanup-calendar-assignment-2026-05-13-report.json"
DEFAULT_REPORT_MD = YEAR_DIR / "checkpoints" / "post-h1-cleanup-calendar-assignment-2026-05-13-report.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "post-h1-cleanup-calendar-assignment-2026-05-13-manifest.json"

BOGUS_SOURCE_DATE = "2026-07-18"
BOGUS_TARGET_DATE = "2026-07-12"
BOGUS_CANDIDATE_ID = "block-bogus_mores_lodge_tempest"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def assignment_by_date(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("date")): row for row in payload.get("assignments") or []}


def candidate_ids(field_day: dict[str, Any]) -> set[str]:
    return {str(loop.get("candidate_id")) for loop in field_day.get("loops") or [] if loop.get("candidate_id")}


def normalize_field_day_for_date(field_day: dict[str, Any], assignment: dict[str, Any], draft_day_number: int | None) -> dict[str, Any]:
    normalized = copy.deepcopy(field_day)
    if draft_day_number is not None:
        normalized["draft_day_number"] = draft_day_number
    normalized["day_type"] = assignment.get("day_type")
    p90_bound = ((assignment.get("field_day") or {}).get("p90_bound_minutes")) or normalized.get("p90_bound_minutes")
    if p90_bound:
        normalized["p90_bound_minutes"] = int(p90_bound)
        normalized["stress"] = round(float(normalized.get("p90_minutes") or 0) / int(p90_bound), 3)
    return normalized


def move_bogus_to_reserve_slot(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaned = copy.deepcopy(payload)
    by_date = assignment_by_date(cleaned)
    source = by_date.get(BOGUS_SOURCE_DATE)
    target = by_date.get(BOGUS_TARGET_DATE)
    if not source or not target:
        missing = [date for date, row in ((BOGUS_SOURCE_DATE, source), (BOGUS_TARGET_DATE, target)) if row is None]
        raise ValueError(f"missing_calendar_dates: {', '.join(missing)}")

    source_field_day = copy.deepcopy(source.get("field_day") or {})
    target_field_day = copy.deepcopy(target.get("field_day") or {})
    if BOGUS_CANDIDATE_ID not in candidate_ids(source_field_day):
        raise ValueError(f"source_date_does_not_contain_bogus_18: {BOGUS_SOURCE_DATE}")

    source_draft = (source.get("field_day") or {}).get("draft_day_number")
    target_draft = (target.get("field_day") or {}).get("draft_day_number")
    source["field_day"] = normalize_field_day_for_date(target_field_day, source, source_draft)
    target["field_day"] = normalize_field_day_for_date(source_field_day, target, target_draft)

    report = {
        "schema": "boise_trails_post_h1_field_day_cleanup_v1",
        "generated_at": now_iso(),
        "status": "passed",
        "accepted_changes": [
            {
                "change_id": "move-bogus-18-off-final-day",
                "reason": "Keep the final challenge day as the operational buffer by moving the existing certified Bogus route card into the July 12 reserve slot.",
                "route_candidate_id": BOGUS_CANDIDATE_ID,
                "from_date": BOGUS_SOURCE_DATE,
                "to_date": BOGUS_TARGET_DATE,
                "final_buffer_date": BOGUS_SOURCE_DATE,
            }
        ],
        "pushed_back_changes": [
            {
                "change_id": "invent-real-availability-windows",
                "reason": "No authoritative per-date personal availability file exists in the active inputs. The field-day layer now exposes dated p90 bounds and marks weekday/weekend as context only, but it does not invent real hard-stop windows.",
            }
        ],
    }
    cleaned["source_files"] = {
        **(cleaned.get("source_files") or {}),
        "post_h1_cleanup_input": display_path(DEFAULT_INPUT_JSON),
    }
    cleaned["known_gaps"] = list(cleaned.get("known_gaps") or [])
    gap = "Post-H1 cleanup moved Bogus route 18 from 2026-07-18 to 2026-07-12 so the final day remains a buffer."
    if gap not in cleaned["known_gaps"]:
        cleaned["known_gaps"].append(gap)
    return cleaned, report


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Post-H1 Field-Day Cleanup",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Accepted Changes",
        "",
    ]
    for row in report["accepted_changes"]:
        lines.append(f"- `{row['change_id']}`: moved `{row['route_candidate_id']}` from `{row['from_date']}` to `{row['to_date']}`; `{row['final_buffer_date']}` is now the final buffer date.")
    lines.extend(["", "## Pushback", ""])
    for row in report["pushed_back_changes"]:
        lines.append(f"- `{row['change_id']}`: {row['reason']}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = read_json(args.input_json)
    cleaned, report = move_bogus_to_reserve_slot(payload)
    write_json(args.output_json, cleaned)
    write_json(args.report_json, report)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="post_h1_field_day_cleanup",
        inputs=[args.input_json],
        outputs=[args.output_json, args.report_json, args.report_md],
        command="python years/2026/scripts/post_h1_field_day_cleanup.py",
        metadata={
            "schema": report["schema"],
            "status": report["status"],
            "accepted_change_count": len(report["accepted_changes"]),
            "pushed_back_change_count": len(report["pushed_back_changes"]),
        },
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.report_json)}")
    print(f"Wrote {display_path(args.report_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], **manifest["metadata"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
