#!/usr/bin/env python3
"""Summarize phone field progress and produce a planner-state patch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "outputs" / "private" / "progress" / "field-progress-latest.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "outputs" / "private" / "progress" / "field-progress-latest.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


def int_ids(values: list[str]) -> list[int]:
    return sorted(int(value) for value in values)


def official_segment_ids(official_geojson: dict[str, Any]) -> list[str]:
    ids = []
    for feature in official_geojson.get("features") or []:
        seg_id = (feature.get("properties") or {}).get("segId")
        if seg_id is not None:
            ids.append(str(seg_id))
    return normalized_ids(ids)


def route_index(field_tool_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(route.get("outing_id")): route for route in field_tool_data.get("routes") or []}


def completed_segments_from_progress(
    routes_by_id: dict[str, dict[str, Any]],
    progress: dict[str, Any],
) -> list[str]:
    completed = set(normalized_ids(progress.get("completed_segment_ids")))
    for outing_id in normalized_ids(progress.get("completed_outing_ids")):
        route = routes_by_id.get(outing_id)
        if route:
            completed.update(normalized_ids(route.get("segment_ids")))
    completed.difference_update(normalized_ids(progress.get("missed_segment_ids")))
    completed.difference_update(normalized_ids(progress.get("blocked_segment_ids")))
    return normalized_ids(completed)


def available_routes(
    field_tool_data: dict[str, Any],
    completed_outing_ids: set[str],
) -> list[dict[str, Any]]:
    routes = []
    for route in field_tool_data.get("routes") or []:
        if str(route.get("outing_id")) in completed_outing_ids:
            continue
        if (route.get("validation") or {}).get("passed") is not True:
            continue
        routes.append(route)
    return routes


def today_options(
    routes: list[dict[str, Any]],
    remaining_ids: set[str],
    time_filters: list[int],
) -> dict[str, list[dict[str, Any]]]:
    options: dict[str, list[dict[str, Any]]] = {}
    for minutes in time_filters:
        rows = []
        for route in routes:
            p75 = int(route.get("door_to_door_minutes_p75") or 0)
            segment_ids = set(normalized_ids(route.get("segment_ids")))
            new_segments = sorted(segment_ids & remaining_ids, key=lambda item: (len(item), item))
            if not new_segments or p75 > minutes:
                continue
            rows.append(
                {
                    "outing_id": route.get("outing_id"),
                    "label": route.get("label"),
                    "trailhead": route.get("trailhead"),
                    "door_to_door_minutes_p75": p75,
                    "new_segment_count": len(new_segments),
                    "new_segment_ids": new_segments,
                    "official_miles": route.get("official_miles"),
                    "on_foot_miles": route.get("on_foot_miles"),
                    "gpx_href": route.get("gpx_href"),
                }
            )
        rows.sort(key=lambda row: (-int(row["new_segment_count"]), int(row["door_to_door_minutes_p75"])))
        options[str(minutes)] = rows[:10]
    return options


def build_progress_report(
    field_tool_data: dict[str, Any],
    official_geojson: dict[str, Any],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    progress = progress or {}
    target_ids = set(official_segment_ids(official_geojson))
    routes_by_id = route_index(field_tool_data)
    completed_outing_ids = set(normalized_ids(progress.get("completed_outing_ids")))
    missed_ids = set(normalized_ids(progress.get("missed_segment_ids")))
    blocked_ids = set(normalized_ids(progress.get("blocked_segment_ids")))
    completed_ids = set(completed_segments_from_progress(routes_by_id, progress))
    remaining_ids = target_ids - completed_ids - blocked_ids
    routes = available_routes(field_tool_data, completed_outing_ids)
    available_remaining_ids = set()
    for route in routes:
        available_remaining_ids.update(set(normalized_ids(route.get("segment_ids"))) & remaining_ids)
    missing_remaining_ids = remaining_ids - available_remaining_ids
    time_filters = [int(value) for value in field_tool_data.get("time_filters_minutes") or [60, 90, 120, 180, 240, 360]]
    baseline = field_tool_data.get("certified_baseline") or {}
    report = {
        "schema": "boise_trails_field_progress_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_schema": field_tool_data.get("schema"),
        "certified_baseline": baseline,
        "summary": {
            "official_segment_count": len(target_ids),
            "completed_outing_count": len(completed_outing_ids),
            "completed_segment_count": len(completed_ids),
            "missed_segment_count": len(missed_ids),
            "blocked_segment_count": len(blocked_ids),
            "remaining_segment_count": len(remaining_ids),
            "available_remaining_segment_count": len(available_remaining_ids),
            "missing_remaining_segment_count": len(missing_remaining_ids),
            "remaining_coverage_preserved": len(missing_remaining_ids) == 0,
            "certified_baseline_status": baseline.get("status"),
            "original_target_still_possible_from_menu": len(missing_remaining_ids) == 0 and baseline.get("status") == "passed",
        },
        "completed_outing_ids": normalized_ids(completed_outing_ids),
        "completed_segment_ids": normalized_ids(completed_ids),
        "missed_segment_ids": normalized_ids(missed_ids),
        "blocked_segment_ids": normalized_ids(blocked_ids),
        "remaining_segment_ids": normalized_ids(remaining_ids),
        "missing_remaining_segment_ids": normalized_ids(missing_remaining_ids),
        "today_options_by_minutes": today_options(routes, remaining_ids, time_filters),
        "private_state_patch": {
            "completed_segment_ids": int_ids(normalized_ids(completed_ids)),
            "blocked_segment_ids": int_ids(normalized_ids(blocked_ids)),
            "_source": "field_progress_report.py; review before merging into 2026-planner-state.private.json",
        },
        "caveats": [
            "This report verifies remaining coverage against the current field menu; it does not regenerate the full MILP calendar certificate.",
            "Missed segment ids are intentionally subtracted from completed outing credit.",
            "Apply the private_state_patch only after confirming the Strava/GPS activity really completed each segment end-to-end.",
        ],
    }
    return report


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Field Progress Report",
        "",
        f"- Certified baseline: `{summary.get('certified_baseline_status')}`",
        f"- Official target segments: {summary['official_segment_count']}",
        f"- Completed segments: {summary['completed_segment_count']}",
        f"- Remaining segments: {summary['remaining_segment_count']}",
        f"- Missing remaining segments from current menu: {summary['missing_remaining_segment_count']}",
        f"- Remaining coverage preserved: {summary['remaining_coverage_preserved']}",
        "",
        "## Best Options By Time",
        "",
    ]
    for minutes, rows in report["today_options_by_minutes"].items():
        lines.append(f"### <= {minutes} minutes")
        if not rows:
            lines.append("")
            lines.append("No remaining outing fits this window.")
            lines.append("")
            continue
        lines.extend(["", "| Outing | Trailhead | P75 | New segments |", "|---|---|---:|---:|"])
        for row in rows[:5]:
            lines.append(
                f"| {row.get('label') or row.get('outing_id')} | {row.get('trailhead')} | "
                f"{row.get('door_to_door_minutes_p75')} | {row.get('new_segment_count')} |"
            )
        lines.append("")
    if report["missing_remaining_segment_ids"]:
        lines.extend(["## Missing Remaining Segment IDs", "", ", ".join(report["missing_remaining_segment_ids"]), ""])
    lines.extend(["## Caveats", ""])
    lines.extend(f"- {item}" for item in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = read_json(args.progress_json) if args.progress_json else {}
    report = build_progress_report(
        read_json(args.field_tool_data_json),
        read_json(args.official_geojson),
        progress,
    )
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
