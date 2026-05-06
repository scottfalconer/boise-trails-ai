#!/usr/bin/env python3
"""Probe p90 feasibility from a configured manual access anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_execution_gpx import load_official_segment_index  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    build_performance_profile,
    candidate_from_trail_group,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    merge_planning_trailheads,
    read_json,
)
from p90_segment_split_probe import probe_candidate_row, segment_trail  # noqa: E402


DEFAULT_MANUAL_DESIGN_JSON = YEAR_DIR / "inputs" / "personal" / "2026-harlow-spring-manual-route-design-v1.json"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "manual-access-anchor-probe-harlow-west-2026-05-06"
DEFAULT_AREA_ID = "package10_harlows_spring_split_probe"
DEFAULT_ALTERNATIVE_ID = "10A"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def anchor_trailhead(anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": anchor["name"],
        "lat": float(anchor["lat"]),
        "lon": float(anchor["lon"]),
        "has_parking": anchor.get("has_parking") is True,
        "parking_confidence": anchor.get("parking_confidence"),
        "parking_minutes": int(anchor.get("parking_minutes") or 8),
        "source": anchor.get("source"),
        "field_ready": anchor.get("field_ready") is True,
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    p90_bound_minutes: int,
    field_ready: bool,
) -> dict[str, Any]:
    under_bound = [row for row in rows if int(row["door_to_door_p90_minutes"]) <= p90_bound_minutes]
    under_bound_ready = [
        row
        for row in under_bound
        if row["route_status"] == "graph_validated" and row["track_validation_passed"]
    ]
    return {
        "probe_count": len(rows),
        "p90_bound_minutes": p90_bound_minutes,
        "under_p90_bound_count": len(under_bound),
        "under_p90_bound_track_valid_graph_validated_count": len(under_bound_ready),
        "over_p90_bound_count": len(rows) - len(under_bound),
        "track_validation_passed_count": sum(1 for row in rows if row["track_validation_passed"]),
        "graph_validated_count": sum(1 for row in rows if row["route_status"] == "graph_validated"),
        "field_ready": field_ready,
        "field_ready_blocker": None if field_ready else "manual_anchor_parking_access_not_verified",
    }


def find_area_and_alternative(
    manual_design: dict[str, Any],
    area_id: str,
    alternative_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    area = next(
        (item for item in manual_design.get("areas") or [] if str(item.get("area_id")) == area_id),
        None,
    )
    if not area:
        raise ValueError(f"Manual design area not found: {area_id}")
    alternative = next(
        (item for item in area.get("alternatives") or [] if str(item.get("alternative_id")) == alternative_id),
        None,
    )
    if not alternative:
        raise ValueError(f"Manual design alternative not found: {alternative_id}")
    anchor_id = alternative.get("start_anchor_id")
    anchor = next(
        (item for item in area.get("anchors") or [] if str(item.get("anchor_id")) == str(anchor_id)),
        None,
    )
    if not anchor:
        raise ValueError(f"Manual design anchor not found: {anchor_id}")
    return area, alternative, anchor


def build_report(
    *,
    manual_design: dict[str, Any],
    state: dict[str, Any],
    manual_design_json_path: Path,
    state_json_path: Path,
    official_geojson_path: Path,
    connector_geojson: Path,
    dem_tif: Path,
    dem_summary_json: Path,
    area_id: str,
    alternative_id: str,
) -> dict[str, Any]:
    area, alternative, anchor = find_area_and_alternative(manual_design, area_id, alternative_id)
    trailhead = anchor_trailhead(anchor)
    state = merge_planning_trailheads(state, [trailhead])
    official_segments, _ = load_official_segments(official_geojson_path)
    segments_by_id = {segment["seg_id"]: segment for segment in official_segments}
    official_index = load_official_segment_index(official_geojson_path)
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    dem_context = load_dem_context(dem_tif, dem_summary_json)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
        activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
        activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
        segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
    )
    rows = []
    for seg_id in alternative.get("required_segment_ids") or []:
        segment = segments_by_id[int(seg_id)]
        candidate = candidate_from_trail_group(
            [segment_trail(segment)],
            state,
            performance_profile,
            connector_graph,
            candidate_type="manual_access_anchor_single_segment_probe",
            elevation_sampler=dem_context["sampler"],
        )
        rows.append(probe_candidate_row(candidate, segment, official_index, connector_graph))
    availability = state.get("availability_model") or {}
    max_bound = max(
        int(availability.get("weekday_max_minutes") or 0),
        int(availability.get("weekend_max_minutes") or 0),
    )
    field_ready = trailhead["field_ready"] is True and trailhead.get("parking_confidence") != "manual_required"
    return {
        "objective": "probe p90-bounded single-segment loops from a manual access anchor",
        "source_files": {
            "manual_design_json": display_path(manual_design_json_path),
            "state_json": display_path(state_json_path),
            "official_geojson": display_path(official_geojson_path),
            "connector_geojson": display_path(connector_geojson),
        },
        "manual_area": {
            "area_id": area.get("area_id"),
            "alternative_id": alternative.get("alternative_id"),
            "title": alternative.get("title"),
            "anchor_id": alternative.get("start_anchor_id"),
            "anchor_name": anchor.get("name"),
            "parking_confidence": anchor.get("parking_confidence"),
            "field_ready": anchor.get("field_ready") is True,
        },
        "summary": summarize_rows(rows, p90_bound_minutes=max_bound, field_ready=field_ready),
        "probe_rows": sorted(rows, key=lambda row: (row["door_to_door_p90_minutes"], row["trail_name"], row["seg_id"])),
        "caveats": [
            "This is conditional route math. The access anchor is not field-ready until parking/signage/access is verified.",
            "Rows are single-official-segment probes, not necessarily enjoyable final outings.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    area = report["manual_area"]
    lines = [
        "# Manual Access Anchor Probe",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Anchor",
        "",
        f"- Area: {area['area_id']}",
        f"- Alternative: {area['alternative_id']} - {area['title']}",
        f"- Anchor: {area['anchor_name']}",
        f"- Parking confidence: {area['parking_confidence']}",
        f"- Field ready: {area['field_ready']}",
        "",
        "## Summary",
        "",
        f"- Probe count: {summary['probe_count']}",
        f"- P90 bound: {summary['p90_bound_minutes']} min",
        f"- Under p90 bound: {summary['under_p90_bound_count']}",
        f"- Under p90 bound with graph validation and continuous track: {summary['under_p90_bound_track_valid_graph_validated_count']}",
        f"- Field ready: {summary['field_ready']}",
        f"- Field-ready blocker: {summary['field_ready_blocker']}",
        "",
        "## Probe Rows",
        "",
        "| Segment | Trail | P90 | P75 | Official | On foot | Track |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report["probe_rows"]:
        lines.append(
            f"| {row['seg_id']} | {row['trail_name']} | {row['door_to_door_p90_minutes']} | "
            f"{row['door_to_door_p75_minutes']} | {row['official_miles']} | "
            f"{row['on_foot_miles']} | {row['track_validation_passed']} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--area-id", default=DEFAULT_AREA_ID)
    parser.add_argument("--alternative-id", default=DEFAULT_ALTERNATIVE_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(
        manual_design=read_json(args.manual_design_json),
        state=read_json(args.state_json),
        manual_design_json_path=args.manual_design_json,
        state_json_path=args.state_json,
        official_geojson_path=args.official_geojson,
        connector_geojson=args.connector_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
        area_id=args.area_id,
        alternative_id=args.alternative_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
