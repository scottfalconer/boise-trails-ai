#!/usr/bin/env python3
"""Probe whether missing p90-bound segments can be split into smaller loops."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from export_execution_gpx import (  # noqa: E402
    candidate_track_coordinates,
    load_official_segment_index,
    validate_track_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    build_performance_profile,
    candidate_from_trail_group,
    center_point,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_trailheads_from_geojson,
    merge_planning_trailheads,
    read_json,
    round_miles,
    slugify,
)


DEFAULT_GAP_JSON = YEAR_DIR / "checkpoints" / "p90-completion-gap-analysis-2026-05-06.json"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-segment-split-probe-2026-05-06"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def segment_trail(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "trail_name": segment["trail_name"],
        "segments": [segment],
        "remaining_segment_ids": [segment["seg_id"]],
        "official_miles": segment["official_miles"],
        "direction_counts": {segment["direction"]: 1},
        "start": segment["start"],
        "end": segment["end"],
        "center": segment.get("center") or center_point(segment["coordinates"]),
    }


def summarize_probe_rows(
    rows: list[dict[str, Any]],
    *,
    max_bound_minutes: int,
    weekend_bound_minutes: int,
) -> dict[str, Any]:
    under_max = [row for row in rows if row["door_to_door_p90_minutes"] <= max_bound_minutes]
    under_weekend = [row for row in rows if row["door_to_door_p90_minutes"] <= weekend_bound_minutes]
    under_max_field_ready = [
        row
        for row in under_max
        if row["route_status"] == "graph_validated" and row["track_validation_passed"]
    ]
    still_over = [row for row in rows if row["door_to_door_p90_minutes"] > max_bound_minutes]
    by_trail = Counter(str(row.get("trail_name") or "unknown") for row in rows)
    still_over_by_trail = Counter(str(row.get("trail_name") or "unknown") for row in still_over)
    return {
        "probe_count": len(rows),
        "input_missing_segment_count": len(rows),
        "under_max_bound_count": len(under_max),
        "under_max_bound_track_valid_graph_validated_count": len(under_max_field_ready),
        "under_weekend_bound_count": len(under_weekend),
        "still_over_max_bound_count": len(still_over),
        "graph_validated_count": sum(1 for row in rows if row["route_status"] == "graph_validated"),
        "track_validation_passed_count": sum(1 for row in rows if row["track_validation_passed"]),
        "trail_count": len(by_trail),
        "still_over_max_bound_trail_count": len(still_over_by_trail),
        "max_bound_minutes": max_bound_minutes,
        "weekend_bound_minutes": weekend_bound_minutes,
    }


def probe_candidate_row(
    candidate: dict[str, Any],
    segment: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any],
) -> dict[str, Any]:
    time_estimates = candidate.get("time_estimates_minutes") or {}
    track = candidate_track_coordinates(
        candidate,
        official_index,
        connector_graph=connector_graph,
        densify_source_lines=True,
    )
    track_validation = validate_track_segments([track])
    candidate_id = f"single-segment-{segment['seg_id']}-{slugify(segment['seg_name'])}"
    return {
        "candidate_id": candidate_id,
        "seg_id": segment["seg_id"],
        "seg_name": segment["seg_name"],
        "trail_name": segment["trail_name"],
        "direction": segment["direction"],
        "official_miles": round_miles(segment["official_miles"]),
        "route_status": candidate.get("route_status"),
        "validation_passed": all(
            (candidate.get("validation") or {}).get(key) is True
            for key in [
                "segment_coverage_passed",
                "ascent_direction_passed",
                "return_path_graph_validated",
            ]
        ),
        "track_validation_passed": track_validation["passed"],
        "track_validation": track_validation,
        "trailhead": (candidate.get("trailhead") or {}).get("name"),
        "on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
        "official_repeat_miles": candidate.get("official_repeat_miles"),
        "connector_miles": candidate.get("connector_miles"),
        "road_miles": candidate.get("road_miles"),
        "door_to_door_p75_minutes": time_estimates.get("door_to_door_p75"),
        "door_to_door_p90_minutes": time_estimates.get("door_to_door_p90"),
        "moving_effort_p75_minutes": time_estimates.get("moving_effort_p75"),
        "route_finding_penalty_minutes": time_estimates.get("route_finding_penalty"),
        "ascent_ft": (candidate.get("effort") or {}).get("ascent_ft"),
        "grade_adjusted_miles": (candidate.get("effort") or {}).get("grade_adjusted_miles"),
        "flags": candidate.get("less_optimal_flags") or [],
    }


def build_report(
    *,
    gap_report: dict[str, Any],
    official_geojson_path: Path,
    state: dict[str, Any],
    trailheads_geojson: Path,
    private_parking_anchors_geojson: Path,
    connector_geojson: Path,
    dem_tif: Path,
    dem_summary_json: Path,
) -> dict[str, Any]:
    public_trailheads = load_trailheads_from_geojson(trailheads_geojson)
    private_trailheads = load_trailheads_from_geojson(private_parking_anchors_geojson)
    state = merge_planning_trailheads(state, [*public_trailheads, *private_trailheads])
    official_segments, _ = load_official_segments(official_geojson_path)
    official_index = load_official_segment_index(official_geojson_path)
    segments_by_id = {segment["seg_id"]: segment for segment in official_segments}
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    dem_context = load_dem_context(dem_tif, dem_summary_json)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
        activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
        activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
        segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
    )
    missing_segment_ids = [
        int(row["seg_id"])
        for row in gap_report.get("missing_under_max_bound_segments") or []
        if int(row["seg_id"]) in segments_by_id
    ]
    rows = []
    for seg_id in missing_segment_ids:
        segment = segments_by_id[seg_id]
        candidate = candidate_from_trail_group(
            [segment_trail(segment)],
            state,
            performance_profile,
            connector_graph,
            candidate_type="single_segment_p90_probe",
            elevation_sampler=dem_context["sampler"],
        )
        rows.append(probe_candidate_row(candidate, segment, official_index, connector_graph))
    availability = state.get("availability_model") or {}
    max_bound = max(int(availability.get("weekday_max_minutes") or 0), int(availability.get("weekend_max_minutes") or 0))
    weekend_bound = int(availability.get("weekend_max_minutes") or 0)
    under_max_ids = {
        row["seg_id"]
        for row in rows
        if row["door_to_door_p90_minutes"] <= max_bound
        and row["route_status"] == "graph_validated"
        and row["track_validation_passed"]
    }
    still_missing_ids = sorted(set(missing_segment_ids) - under_max_ids)
    return {
        "objective": "test whether p90-missing official segments can be split into single-segment legal loops",
        "source_files": {
            "gap_report_json": display_path(DEFAULT_GAP_JSON),
            "official_geojson": display_path(official_geojson_path),
            "state_json": display_path(DEFAULT_STATE_JSON),
            "connector_geojson": display_path(connector_geojson),
            "trailheads_geojson": display_path(trailheads_geojson),
            "private_parking_anchors_geojson": display_path(private_parking_anchors_geojson),
            "dem_tif": display_path(dem_tif),
        },
        "summary": summarize_probe_rows(rows, max_bound_minutes=max_bound, weekend_bound_minutes=weekend_bound)
        | {
            "newly_bounded_track_valid_segment_count": len(under_max_ids),
            "still_missing_after_single_segment_probe_count": len(still_missing_ids),
            "still_missing_after_single_segment_probe_ids": still_missing_ids,
        },
        "probe_rows": sorted(rows, key=lambda row: (-row["door_to_door_p90_minutes"], row["trail_name"], row["seg_id"])),
        "caveats": [
            "These are diagnostic split probes, not promoted field-menu outings.",
            "Every probe completes one official segment and returns to the same parked car, but route quality may be poor because it can require long access/return for tiny official credit.",
            "Rows still over the max p90 bound need better access anchors, different route design, or an explicit personal-bound exception.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# P90 Segment Split Probe",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Input missing segments: {summary['input_missing_segment_count']}",
        f"- Single-segment probes under max bound ({summary['max_bound_minutes']} min): {summary['under_max_bound_count']}",
        f"- Under max bound with graph validation and continuous track: {summary['under_max_bound_track_valid_graph_validated_count']}",
        f"- Single-segment probes under weekend bound ({summary['weekend_bound_minutes']} min): {summary['under_weekend_bound_count']}",
        f"- Still over max bound: {summary['still_over_max_bound_count']}",
        f"- Graph-validated probes: {summary['graph_validated_count']}",
        f"- Track-validation passed probes: {summary['track_validation_passed_count']}",
        f"- Still missing after probe ids: {', '.join(str(seg_id) for seg_id in summary['still_missing_after_single_segment_probe_ids'])}",
        "",
        "## Probe Rows",
        "",
        "| Segment | Trail | P90 | P75 | Official | On foot | Trailhead | Status | Flags |",
        "|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in report["probe_rows"]:
        lines.append(
            f"| {row['seg_id']} | {row['trail_name']} | {row['door_to_door_p90_minutes']} | "
            f"{row['door_to_door_p75_minutes']} | {row['official_miles']} | "
            f"{row['on_foot_miles']} | {row['trailhead']} | {row['route_status']} | "
            f"{', '.join(row['flags'])} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-json", type=Path, default=DEFAULT_GAP_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--private-parking-anchors-geojson", type=Path, default=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(
        gap_report=read_json(args.gap_json),
        official_geojson_path=args.official_geojson,
        state=read_json(args.state_json),
        trailheads_geojson=args.trailheads_geojson,
        private_parking_anchors_geojson=args.private_parking_anchors_geojson,
        connector_geojson=args.connector_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
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
