#!/usr/bin/env python3
"""Assemble one route candidate per human route block.

This pass is deliberately different from ``block_route_candidate_pass.py``.
Instead of selecting many pre-existing route-menu candidates, it takes the
block inventory as the unit of work and tries to build one continuous
single-car candidate for each block using the existing connector graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_route_candidate_pass import (  # noqa: E402
    PALETTE,
    line_feature,
    multiline_feature,
    parking_feature,
    render_html as render_map_html,
    split_coords_on_gaps,
)
from export_execution_gpx import (  # noqa: E402
    candidate_segment_coordinates,
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
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    as_int_set,
    build_performance_profile,
    candidate_from_trail_group,
    center_point,
    get_outing_model,
    group_remaining_by_trail,
    haversine_miles,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    load_trailheads_from_geojson,
    merge_planning_trailheads,
    normalize_name,
    read_json,
    reverse_trail_orientation,
    round_miles,
    shortest_connector_path,
    trail_is_reversible,
)
from route_block_planner import build_block_index, normalize_trail_name  # noqa: E402


DEFAULT_BLOCKS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-blocks-v1.json"
DEFAULT_STATE_PATH = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "block-assembled-route-pass-v1"
TOTAL_OFFICIAL_MILES = 164.43


def connector_or_gap_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
) -> float:
    mapped = shortest_connector_path(start, end, connector_graph, snap_tolerance_miles)
    if mapped:
        return float(mapped["distance_miles"])
    return haversine_miles(start, end) * 1.25


def trail_options(trail: dict[str, Any]) -> list[dict[str, Any]]:
    options = [trail]
    if trail_is_reversible(trail):
        options.append(reverse_trail_orientation(trail))
    return options


def order_trails_by_connector_cost(
    trails: list[dict[str, Any]],
    start_point: tuple[float, float],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
) -> list[dict[str, Any]]:
    """Nearest-neighbor ordering using graph distance, not straight-line distance."""
    remaining = list(trails)
    ordered: list[dict[str, Any]] = []
    current = start_point
    while remaining:
        best_index = 0
        best_trail = remaining[0]
        best_distance = float("inf")
        for index, trail in enumerate(remaining):
            for option in trail_options(trail):
                distance = connector_or_gap_distance(
                    current,
                    option["start"],
                    connector_graph,
                    snap_tolerance_miles,
                )
                if distance < best_distance:
                    best_index = index
                    best_trail = option
                    best_distance = distance
        remaining.pop(best_index)
        ordered.append(best_trail)
        current = best_trail["end"]
    return ordered


def block_center(trails: list[dict[str, Any]]) -> tuple[float, float]:
    points = [trail["center"] for trail in trails]
    return center_point(points)


def preferred_trailhead_for_block(
    block: dict[str, Any],
    trails: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any] | None:
    trailheads = state.get("trailheads") or []
    if not trailheads:
        return None
    preferred_names = [normalize_name(str(value)) for value in block.get("preferred_trailheads") or []]
    preferred_matches = []
    for trailhead in trailheads:
        trailhead_name = normalize_name(str(trailhead.get("name") or ""))
        if any(
            preferred and (preferred in trailhead_name or trailhead_name in preferred)
            for preferred in preferred_names
        ):
            preferred_matches.append(trailhead)
    candidates = preferred_matches or trailheads
    center = block_center(trails)
    return min(
        candidates,
        key=lambda trailhead: haversine_miles(
            center,
            (float(trailhead["lon"]), float(trailhead["lat"])),
        ),
    )


def load_planning_context(args: argparse.Namespace) -> dict[str, Any]:
    state = load_state(args.state_json)
    public_trailheads = load_trailheads_from_geojson(args.trailheads_geojson)
    state = merge_planning_trailheads(state, public_trailheads)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    completed_ids = as_int_set(state.get("completed_segment_ids"))
    blocked_ids = as_int_set(state.get("blocked_segment_ids"))
    blocked_trails = {normalize_name(name) for name in state.get("blocked_trail_names") or []}
    remaining_segments = [
        segment
        for segment in official_segments
        if segment["seg_id"] not in completed_ids
        and segment["seg_id"] not in blocked_ids
        and normalize_name(segment["trail_name"]) not in blocked_trails
    ]
    trails = group_remaining_by_trail(remaining_segments)
    performance_profile = build_performance_profile(
        state=state,
        strava_activity_details_dir=args.strava_activity_details_dir,
        activity_summary_csv=args.activity_summary_csv,
        activity_detail_summary_csv=args.activity_detail_summary_csv,
        segment_perf_csv=args.segment_perf_csv,
    )
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    dem_context = load_dem_context(args.dem_tif, args.dem_summary_json)
    return {
        "state": state,
        "official_segments": official_segments,
        "trails": trails,
        "performance_profile": performance_profile,
        "connector_graph": connector_graph,
        "elevation_sampler": dem_context["sampler"],
        "source_datasets": {
            "official_geojson": str(args.official_geojson),
            "blocks_json": str(args.blocks_json),
            "state_json": str(args.state_json),
            "connector_geojson": str(args.connector_geojson),
            "trailheads_geojson": str(args.trailheads_geojson),
            "dem_tif": str(args.dem_tif),
        },
    }


def block_acceptance_status(
    block: dict[str, Any],
    candidate: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> tuple[str, list[str]]:
    if not candidate:
        return "unbuilt", ["no_candidate_built"]

    reasons = []
    official = float(candidate.get("official_new_miles") or 0.0)
    total = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
    ratio = total / official if official else 99.0
    preferred_ratio = float(acceptance.get("preferred_max_on_foot_to_official_ratio") or 1.6)
    min_official = float(acceptance.get("min_official_miles_unless_geography_locked") or 5.0)

    if candidate.get("route_status") != "graph_validated":
        reasons.append("not_graph_validated")
    if not candidate.get("validation", {}).get("ascent_direction_passed"):
        reasons.append("ascent_direction_not_validated")
    if official < min_official and not block.get("geography_locked"):
        reasons.append("below_min_official_miles")
    if ratio > preferred_ratio and not block.get("geography_locked"):
        reasons.append("ratio_above_preferred_limit")
    if block.get("status") == "boundary_review":
        reasons.append("boundary_review_block")

    if not reasons:
        return "schedule_candidate_after_gpx", []
    if reasons == ["boundary_review_block"]:
        return "manual_review_candidate", reasons
    return "needs_manual_gpx_review", reasons


def build_block_routes(
    blocks_config: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    trail_to_block, duplicates = build_block_index(blocks_config)
    _ = trail_to_block
    trail_by_name = {normalize_trail_name(trail["trail_name"]): trail for trail in context["trails"]}
    acceptance = blocks_config.get("acceptance_criteria") or {}
    snap_tolerance = float(
        get_outing_model(context["state"]).get("mapped_connector_snap_tolerance_miles", 0.02)
    )
    routes = []
    missing_configured_trails: dict[str, list[str]] = {}
    for block_index, block in enumerate(blocks_config.get("blocks") or [], start=1):
        block_trails = []
        missing = []
        for trail_name in block.get("trail_names") or []:
            trail = trail_by_name.get(normalize_trail_name(str(trail_name)))
            if trail:
                block_trails.append(trail)
            else:
                missing.append(str(trail_name))
        if missing:
            missing_configured_trails[str(block["block_id"])] = missing
        if not block_trails:
            continue
        start_trailhead = preferred_trailhead_for_block(block, block_trails, context["state"])
        start_point = (
            (float(start_trailhead["lon"]), float(start_trailhead["lat"]))
            if start_trailhead
            else block_center(block_trails)
        )
        ordered = order_trails_by_connector_cost(
            sorted(block_trails, key=lambda item: item["trail_name"]),
            start_point,
            context["connector_graph"],
            snap_tolerance,
        )
        candidate = candidate_from_trail_group(
            ordered,
            context["state"],
            context["performance_profile"],
            context["connector_graph"],
            candidate_type="assembled_route_block",
            elevation_sampler=context["elevation_sampler"],
        )
        candidate["candidate_id"] = f"block-{block['block_id']}"
        candidate["block_id"] = str(block["block_id"])
        candidate["block_name"] = block.get("name")
        candidate["block_status"] = block.get("status")
        candidate["block_rationale"] = block.get("rationale")
        candidate["planned_order_start_trailhead"] = start_trailhead.get("name") if start_trailhead else None
        status, reasons = block_acceptance_status(block, candidate, acceptance)
        official = float(candidate.get("official_new_miles") or 0.0)
        total = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
        routes.append(
            {
                "route_number": block_index,
                "candidate_id": candidate["candidate_id"],
                "block_id": candidate["block_id"],
                "block_name": candidate["block_name"],
                "block_status": candidate["block_status"],
                "planning_status": status,
                "planning_reasons": reasons,
                "trail_names": candidate.get("trail_names") or [],
                "official_miles": round_miles(official),
                "on_foot_miles": round_miles(total),
                "ratio": round(total / official, 2) if official else None,
                "total_minutes": candidate.get("total_minutes"),
                "trailhead": (candidate.get("trailhead") or {}).get("name"),
                "route_status": candidate.get("route_status"),
                "less_optimal_flags": candidate.get("less_optimal_flags") or [],
                "segment_ids": candidate.get("segment_ids") or [],
                "candidate": candidate,
            }
        )

    covered_segments = {int(seg_id) for route in routes for seg_id in route.get("segment_ids") or []}
    total_on_foot = sum(float(route.get("on_foot_miles") or 0.0) for route in routes)
    route_rows = [{key: value for key, value in route.items() if key != "candidate"} for route in routes]
    return {
        "planning_status": "block_assembled_route_pass",
        "source_datasets": context["source_datasets"],
        "summary": {
            "assembled_route_count": len(routes),
            "covered_segment_count": len(covered_segments),
            "total_on_foot_miles": round_miles(total_on_foot),
            "planwide_on_foot_to_official_ratio": round(total_on_foot / TOTAL_OFFICIAL_MILES, 2),
            "routes_under_1_official_mile": sum(1 for route in routes if float(route["official_miles"]) < 1.0),
            "routes_under_2_official_miles": sum(1 for route in routes if float(route["official_miles"]) < 2.0),
            "graph_validated_route_count": sum(1 for route in routes if route["route_status"] == "graph_validated"),
            "schedule_candidate_after_gpx_count": sum(
                1 for route in routes if route["planning_status"] == "schedule_candidate_after_gpx"
            ),
            "manual_review_route_count": sum(
                1 for route in routes if route["planning_status"] != "schedule_candidate_after_gpx"
            ),
            "duplicate_configured_trail_count": len(duplicates),
            "missing_configured_trail_count": sum(len(items) for items in missing_configured_trails.values()),
        },
        "routes": route_rows,
        "candidate_index": {route["candidate_id"]: route["candidate"] for route in routes},
        "missing_configured_trails": missing_configured_trails,
        "caveats": [
            "This is the first graph-aware block assembly pass. It reduces car-hop fragmentation by building from human route blocks, but it still needs GPX/local review before being called final.",
            "Boundary-review blocks remain review candidates even when graph validation passes.",
            "Routes with ratios above the preferred threshold need either manual loop improvement, an explicit necessary-grinder label, or a split/merge decision.",
        ],
    }


def build_map_data(
    route_pass: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    route_features = []
    official_features = []
    parking_features = []
    validations = []
    candidate_index = route_pass.get("candidate_index") or {}
    for route in route_pass.get("routes") or []:
        candidate = candidate_index[str(route["candidate_id"])]
        color = PALETTE[(int(route["route_number"]) - 1) % len(PALETTE)]
        coords = candidate_track_coordinates(candidate, official_index, connector_graph=connector_graph)
        source_validation = validate_track_segments([coords], max_gap_miles=0.1)
        rendered_parts = split_coords_on_gaps(coords, max_gap_miles=0.1)
        render_validation = validate_track_segments(rendered_parts, max_gap_miles=0.1)
        props = {
            "kind": "route",
            "route_number": route["route_number"],
            "candidate_id": route["candidate_id"],
            "block_name": route["block_name"],
            "title": ", ".join(route["trail_names"]),
            "official_miles": route["official_miles"],
            "on_foot_miles": route["on_foot_miles"],
            "trailhead": route["trailhead"],
            "color": color,
            "source_gap_warning_count": len(source_validation["failures"]),
        }
        route_feature = multiline_feature(rendered_parts, props)
        if route_feature:
            route_features.append(route_feature)
        parking = parking_feature(candidate, props)
        if parking:
            parking_features.append(parking)
        for segment in candidate.get("segments") or []:
            seg_coords = candidate_segment_coordinates(candidate, segment, official_index)
            feature = line_feature(
                seg_coords,
                {
                    **props,
                    "kind": "official_segment",
                    "seg_id": segment.get("seg_id"),
                    "segment_name": segment.get("seg_name") or segment.get("trail_name"),
                },
            )
            if feature:
                official_features.append(feature)
        validations.append(
            {
                "candidate_id": route["candidate_id"],
                "source_gap_warning": not source_validation["passed"],
                "source_max_gap_miles": source_validation["max_trackpoint_gap_miles"],
                "rendered_passed": source_validation["passed"] and render_validation["passed"],
                "rendered_failures": render_validation["failures"],
            }
        )
    return {
        "summary": {
            "selected_route_count": route_pass["summary"]["assembled_route_count"],
            "covered_segment_count": route_pass["summary"]["covered_segment_count"],
            "total_on_foot_miles": route_pass["summary"]["total_on_foot_miles"],
            "planwide_on_foot_to_official_ratio": route_pass["summary"][
                "planwide_on_foot_to_official_ratio"
            ],
        },
        "routes": route_pass["routes"],
        "feature_collections": {
            "routes": {"type": "FeatureCollection", "features": route_features},
            "official_segments": {"type": "FeatureCollection", "features": official_features},
            "parking": {"type": "FeatureCollection", "features": parking_features},
        },
        "map_validation": {
            "rendered_passed": all(item["rendered_passed"] for item in validations),
            "source_gap_warning_count": sum(1 for item in validations if item["source_gap_warning"]),
            "route_validations": validations,
        },
    }


def render_markdown(route_pass: dict[str, Any]) -> str:
    summary = route_pass["summary"]
    lines = [
        "# 2026 Block-Assembled Route Pass v1",
        "",
        "Status: graph-aware block assembly draft. This is the first pass that builds from route blocks instead of choosing many small pre-existing outings.",
        "",
        "## Summary",
        "",
        f"- Assembled routes: {summary['assembled_route_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / 251",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- Planwide on-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Routes under 1 official mile: {summary['routes_under_1_official_mile']}",
        f"- Routes under 2 official miles: {summary['routes_under_2_official_miles']}",
        f"- Graph-validated routes: {summary['graph_validated_route_count']}",
        f"- Schedule candidates after GPX: {summary['schedule_candidate_after_gpx_count']}",
        f"- Manual-review routes: {summary['manual_review_route_count']}",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in route_pass.get("caveats") or [])
    lines.extend(
        [
            "",
            "## Routes",
            "",
            "| # | Block | Status | Route | Trailhead | Official mi | On-foot mi | Ratio | Minutes | Reasons |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for route in route_pass.get("routes") or []:
        reasons = ", ".join(route.get("planning_reasons") or [])
        lines.append(
            f"| {route['route_number']} | {route['block_name']} | {route['planning_status']} | "
            f"{', '.join(route['trail_names'])} | {route.get('trailhead') or ''} | "
            f"{route['official_miles']} | {route['on_foot_miles']} | {route['ratio']} | "
            f"{route.get('total_minutes') or ''} | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks-json", type=Path, default=DEFAULT_BLOCKS_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--strava-activity-details-dir", type=Path, default=DEFAULT_STRAVA_DETAILS_DIR)
    parser.add_argument("--activity-summary-csv", type=Path, default=DEFAULT_ACTIVITY_SUMMARY_CSV)
    parser.add_argument("--activity-detail-summary-csv", type=Path, default=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV)
    parser.add_argument("--segment-perf-csv", type=Path, default=DEFAULT_SEGMENT_PERF_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blocks_config = read_json(args.blocks_json)
    context = load_planning_context(args)
    route_pass = build_block_routes(blocks_config, context)
    official_index = load_official_segment_index(args.official_geojson)
    map_data = build_map_data(route_pass, official_index, context["connector_graph"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_json_path = args.output_dir / f"{args.basename}-map-data.json"
    map_html_path = args.output_dir / f"{args.basename}-map.html"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, route_pass)
    md_path.write_text(render_markdown(route_pass), encoding="utf-8")
    write_json(map_json_path, map_data)
    map_html_path.write_text(render_map_html(map_data), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[
                args.blocks_json,
                args.state_json,
                args.official_geojson,
                args.connector_geojson,
                args.trailheads_geojson,
            ],
            outputs=[json_path, md_path, map_json_path, map_html_path],
            command="block_route_assembler.py",
            metadata={
                "assembled_route_count": route_pass["summary"]["assembled_route_count"],
                "covered_segment_count": route_pass["summary"]["covered_segment_count"],
                "rendered_map_passed": map_data["map_validation"]["rendered_passed"],
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {map_html_path}")
    print(f"Wrote {map_json_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
