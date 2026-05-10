#!/usr/bin/env python3
"""Generate a human-executable field-day layer over certified route cards.

This exporter does not replace route-card certification. It takes the dated
field-day assignment proof and overlays each loop with the currently certified
phone route-card metadata when a match exists. Unmatched loops stay visible as
promotion gaps.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_ASSIGNMENT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-calendar-assignment-2026-05-06.json"
DEFAULT_FIELD_TOOL_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "human-executable-field-day-layer-2026-05-10.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "human-executable-field-day-layer-2026-05-10.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "human-executable-field-day-layer-2026-05-10-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def trail_set_key(trails: list[Any] | None) -> tuple[str, ...]:
    return tuple(sorted(normalize_key(trail) for trail in trails or [] if normalize_key(trail)))


def int_segment_ids(values: list[Any] | None) -> list[int]:
    result = []
    for value in values or []:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def route_card_ref(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "outing_id": route.get("outing_id"),
        "label": route.get("label"),
        "candidate_ids": list(route.get("candidate_ids") or []),
        "gpx_href": route.get("gpx_href"),
        "validation_passed": bool((route.get("validation") or {}).get("passed")),
    }


def build_route_card_index(field_tool_payload: dict[str, Any]) -> dict[str, Any]:
    by_candidate_id: dict[str, dict[str, Any]] = {}
    by_trailhead_and_trails: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    by_segment_set: dict[tuple[int, ...], dict[str, Any]] = {}

    for route in field_tool_payload.get("routes") or []:
        for candidate_id in route.get("candidate_ids") or []:
            by_candidate_id[normalize_key(candidate_id)] = route
        by_trailhead_and_trails[
            (normalize_key(route.get("trailhead")), trail_set_key(route.get("trails")))
        ] = route
        segment_key = tuple(int_segment_ids(route.get("segment_ids")))
        if segment_key:
            by_segment_set[segment_key] = route

    return {
        "by_candidate_id": by_candidate_id,
        "by_trailhead_and_trails": by_trailhead_and_trails,
        "by_segment_set": by_segment_set,
    }


def find_route_card(loop: dict[str, Any], index: dict[str, Any]) -> dict[str, Any] | None:
    candidate_id = normalize_key(loop.get("candidate_id"))
    if candidate_id and candidate_id in index["by_candidate_id"]:
        return index["by_candidate_id"][candidate_id]

    trailhead_trails_key = (
        normalize_key(loop.get("trailhead")),
        trail_set_key(loop.get("trail_names")),
    )
    if trailhead_trails_key in index["by_trailhead_and_trails"]:
        return index["by_trailhead_and_trails"][trailhead_trails_key]

    segment_key = tuple(int_segment_ids(loop.get("segment_ids")))
    if segment_key and segment_key in index["by_segment_set"]:
        return index["by_segment_set"][segment_key]

    return None


def public_loop(loop: dict[str, Any], route: dict[str, Any] | None) -> dict[str, Any]:
    status = "certified_route_card" if route else "needs_route_card_promotion"
    return {
        "loop_id": loop.get("loop_id"),
        "source": loop.get("source"),
        "candidate_id": loop.get("candidate_id"),
        "label": loop.get("label"),
        "trailhead": loop.get("trailhead"),
        "trail_names": list(loop.get("trail_names") or []),
        "segment_count": int(loop.get("segment_count") or 0),
        "official_miles": round(float(loop.get("official_miles") or 0.0), 2),
        "on_foot_miles": round(float(loop.get("on_foot_miles") or 0.0), 2),
        "p75_minutes": int(loop.get("p75_minutes") or 0),
        "p90_minutes": int(loop.get("p90_minutes") or 0),
        "validation_passed": bool(loop.get("validation_passed")),
        "manual_design_hold": bool(loop.get("manual_design_hold")),
        "certification_status": status,
        "route_card_ref": route_card_ref(route) if route else None,
    }


def field_day_execution_status(loops: list[dict[str, Any]]) -> str:
    if any(loop["certification_status"] != "certified_route_card" for loop in loops):
        return "needs_route_card_promotion"
    if len(loops) > 1:
        return "needs_day_gpx_validation"
    return "executable_route_card"


def build_field_day_layer(
    assignments_payload: dict[str, Any],
    field_tool_payload: dict[str, Any],
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    index = build_route_card_index(field_tool_payload)
    field_days = []
    certified_loop_count = 0
    promotion_gap_count = 0
    total_loop_count = 0

    for assignment in assignments_payload.get("assignments") or []:
        field_day = assignment.get("field_day") or {}
        loops = []
        for loop in field_day.get("loops") or []:
            route = find_route_card(loop, index)
            loop_row = public_loop(loop, route)
            loops.append(loop_row)
            total_loop_count += 1
            if route:
                certified_loop_count += 1
            else:
                promotion_gap_count += 1

        segment_summary = field_day.get("segment_summary") or {}
        field_days.append(
            {
                "date": assignment.get("date"),
                "weekday_name": assignment.get("weekday_name"),
                "day_type": assignment.get("day_type"),
                "constraints": list(assignment.get("constraints") or []),
                "draft_day_number": field_day.get("draft_day_number"),
                "field_day_id": field_day.get("field_day_id"),
                "p75_minutes": int(field_day.get("p75_minutes") or 0),
                "p90_minutes": int(field_day.get("p90_minutes") or 0),
                "p90_bound_minutes": int(field_day.get("p90_bound_minutes") or 0),
                "stress": field_day.get("stress"),
                "drive_minutes": int(field_day.get("drive_minutes") or 0),
                "between_drive_minutes": int(field_day.get("between_drive_minutes") or 0),
                "loop_count": len(loops),
                "transfer_count": max(len(loops) - 1, 0),
                "official_miles": round(float(segment_summary.get("official_miles") or 0.0), 2),
                "on_foot_miles": round(float(field_day.get("on_foot_miles") or 0.0), 2),
                "segment_count": int(segment_summary.get("segment_count") or 0),
                "segment_ids": int_segment_ids(segment_summary.get("segment_ids")),
                "execution_status": field_day_execution_status(loops),
                "loops": loops,
            }
        )

    audit = assignments_payload.get("audit") or {}
    summary = {
        "field_day_count": len(field_days),
        "loop_count": total_loop_count,
        "multi_start_day_count": sum(1 for day in field_days if day["loop_count"] > 1),
        "total_p75_minutes": sum(day["p75_minutes"] for day in field_days),
        "max_p90_minutes": max((day["p90_minutes"] for day in field_days), default=0),
        "total_between_drive_minutes": sum(day["between_drive_minutes"] for day in field_days),
        "certified_route_card_loop_count": certified_loop_count,
        "needs_route_card_promotion_loop_count": promotion_gap_count,
        "official_segment_count": int(audit.get("official_segment_count") or 0),
        "covered_segment_count": int(audit.get("covered_segment_count") or 0),
        "missing_segment_count": int(audit.get("missing_segment_count") or 0),
        "assignment_audit_passed": bool(audit.get("passed")),
        "field_tool_baseline_status": (field_tool_payload.get("certified_baseline") or {}).get("status"),
    }

    return {
        "schema": "boise_trails_human_executable_field_day_layer_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "objective": "Human-executable field-day layer over certified route cards.",
        "source_files": source_files or {},
        "summary": summary,
        "publication_status": (
            "needs_route_card_promotion"
            if promotion_gap_count
            else "needs_day_gpx_validation"
        ),
        "field_days": field_days,
        "known_gaps": [
            "Loop-level route-card certification gaps must be promoted before publication.",
            "Multi-loop days still need day-level GPX export and validation.",
            "Day-of Ridge to Rivers conditions, closures, heat, water, and parking checks still apply.",
        ],
    }


def render_markdown(layer: dict[str, Any]) -> str:
    summary = layer.get("summary") or {}
    lines = [
        "# Human-Executable Field-Day Layer",
        "",
        f"Generated: {layer.get('generated_at', '')}",
        "",
        "Objective: group certified route cards into day-level execution bundles while keeping promotion gaps visible.",
        "",
        "## Summary",
        "",
        f"- Field days: {summary.get('field_day_count', 0)}",
        f"- Loops: {summary.get('loop_count', 0)}",
        f"- Multi-start days: {summary.get('multi_start_day_count', 0)}",
        f"- Coverage: {summary.get('covered_segment_count', 0)}/{summary.get('official_segment_count', 0)} official segments",
        f"- Total p75: {summary.get('total_p75_minutes', 0)} min",
        f"- Total between-start drive: {summary.get('total_between_drive_minutes', 0)} min",
        f"- Certified route-card loops: {summary.get('certified_route_card_loop_count', 0)}",
        f"- Needs route-card promotion: {summary.get('needs_route_card_promotion_loop_count', 0)}",
        f"- Publication status: `{layer.get('publication_status')}`",
        "",
        "## Field Days",
        "",
        "| Date | Weekday | Type | P75 | P90 / bound | Loops | Transfer min | Official mi | On-foot mi | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for day in layer.get("field_days") or []:
        lines.append(
            "| {date} | {weekday} | {day_type} | {p75} | {p90} / {bound} | {loops} | {transfer} | {official:.2f} | {on_foot:.2f} | {status} |".format(
                date=day.get("date"),
                weekday=day.get("weekday_name"),
                day_type=day.get("day_type"),
                p75=day.get("p75_minutes"),
                p90=day.get("p90_minutes"),
                bound=day.get("p90_bound_minutes"),
                loops=day.get("loop_count"),
                transfer=day.get("between_drive_minutes"),
                official=float(day.get("official_miles") or 0.0),
                on_foot=float(day.get("on_foot_miles") or 0.0),
                status=day.get("execution_status"),
            )
        )

    lines.extend(["", "## Loop Certification Detail", ""])
    for day in layer.get("field_days") or []:
        lines.append(f"### {day.get('date')} {day.get('weekday_name')}")
        lines.append("")
        for loop in day.get("loops") or []:
            lines.append(
                f"- `{loop.get('label')}` from `{loop.get('trailhead')}` - `{loop.get('certification_status')}`"
            )
        lines.append("")

    if layer.get("known_gaps"):
        lines.extend(["## Known Gaps", ""])
        for gap in layer["known_gaps"]:
            lines.append(f"- {gap}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-json", type=Path, default=DEFAULT_ASSIGNMENT_JSON)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments_payload = read_json(args.assignment_json)
    field_tool_payload = read_json(args.field_tool_json)
    layer = build_field_day_layer(
        assignments_payload,
        field_tool_payload,
        source_files={
            "calendar_assignment": display_path(args.assignment_json),
            "field_tool_data": display_path(args.field_tool_json),
        },
    )
    write_json(args.output_json, layer)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(layer), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="human-executable-field-day-layer-2026-05-10",
        inputs=[args.assignment_json, args.field_tool_json],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/export_field_day_layer.py",
        metadata={"schema": layer["schema"], "publication_status": layer["publication_status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps(layer["summary"], indent=2))


if __name__ == "__main__":
    main()
