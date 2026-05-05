#!/usr/bin/env python3
"""Combine validated block-package components when it improves the route plan."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_route_assembler import load_planning_context  # noqa: E402
from block_route_candidate_pass import (  # noqa: E402
    PALETTE,
    line_feature,
    multiline_feature,
    parking_feature,
    render_html,
    split_coords_on_gaps,
)
from export_execution_gpx import (  # noqa: E402
    candidate_segment_coordinates,
    candidate_track_coordinates,
    load_candidate_index,
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
    candidate_from_trail_group,
    group_remaining_by_trail,
    load_official_segments,
    order_trails_nearest_neighbor,
    read_json,
    round_miles,
    slugify,
)
from route_block_planner import normalize_trail_name  # noqa: E402


DEFAULT_BLOCKS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-blocks-v1.json"
DEFAULT_STATE_PATH = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-route-candidate-pass-v1.json"
DEFAULT_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-day-package-pass-v1.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "block-combo-route-pass-v1"
TOTAL_OFFICIAL_MILES = 164.43


def candidate_segment_ids(candidate: dict[str, Any]) -> set[int]:
    return {int(segment_id) for segment_id in candidate.get("segment_ids") or []}


def source_component_ids(candidate: dict[str, Any]) -> list[str]:
    return list(candidate.get("source_component_candidate_ids") or [candidate["candidate_id"]])


def build_combo_candidate(
    components: list[dict[str, Any]],
    original_candidate_index: dict[str, dict[str, Any]],
    trail_by_name: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    trails = []
    seen_trails = set()
    for component in components:
        candidate = original_candidate_index[str(component["candidate_id"])]
        for trail_name in candidate.get("trail_names") or []:
            key = normalize_trail_name(str(trail_name))
            if key in seen_trails:
                continue
            trail = trail_by_name.get(key)
            if not trail:
                return None
            trails.append(trail)
            seen_trails.add(key)
    if len(trails) < 2:
        return None
    start_point = trails[0]["start"]
    ordered = order_trails_nearest_neighbor(trails, start_point)
    candidate = candidate_from_trail_group(
        ordered,
        context["state"],
        context["performance_profile"],
        context["connector_graph"],
        candidate_type="block_component_combo",
        elevation_sampler=context["elevation_sampler"],
    )
    source_ids = [str(component["candidate_id"]) for component in components]
    candidate["candidate_id"] = "combo-" + slugify("-".join(source_ids))
    candidate["source_component_candidate_ids"] = source_ids
    return candidate


def combo_is_acceptable(
    combo_candidate: dict[str, Any],
    components: list[dict[str, Any]],
    max_extra_miles_per_saved_route: float,
) -> bool:
    if combo_candidate.get("route_status") != "graph_validated":
        return False
    original_miles = sum(float(component.get("on_foot_miles") or 0.0) for component in components)
    combo_miles = float(combo_candidate.get("estimated_total_on_foot_miles") or 0.0)
    saved_routes = max(0, len(components) - 1)
    return combo_miles <= original_miles + max_extra_miles_per_saved_route * saved_routes


def generate_combo_candidates(
    package: dict[str, Any],
    original_candidate_index: dict[str, dict[str, Any]],
    trail_by_name: dict[str, dict[str, Any]],
    context: dict[str, Any],
    max_combo_size: int,
    max_extra_miles_per_saved_route: float,
) -> list[dict[str, Any]]:
    components = package.get("components") or []
    combos = []
    for size in range(2, min(max_combo_size, len(components)) + 1):
        for component_tuple in itertools.combinations(components, size):
            combo = build_combo_candidate(
                list(component_tuple),
                original_candidate_index,
                trail_by_name,
                context,
            )
            if not combo:
                continue
            if not combo_is_acceptable(combo, list(component_tuple), max_extra_miles_per_saved_route):
                continue
            combo["block_id"] = package["block_id"]
            combo["block_name"] = package["block_name"]
            combo["combo_original_on_foot_miles"] = round_miles(
                sum(float(component.get("on_foot_miles") or 0.0) for component in component_tuple)
            )
            combo["combo_saved_component_routes"] = len(component_tuple) - 1
            combos.append(combo)
    unique: dict[str, dict[str, Any]] = {}
    for combo in combos:
        key = "-".join(sorted(str(value) for value in combo["source_component_candidate_ids"]))
        current = unique.get(key)
        if current is None or float(combo["estimated_total_on_foot_miles"]) < float(
            current["estimated_total_on_foot_miles"]
        ):
            unique[key] = combo
    return list(unique.values())


def local_candidate_cost(candidate: dict[str, Any], route_count_weight: float) -> float:
    on_foot = float(candidate.get("estimated_total_on_foot_miles") or candidate.get("on_foot_miles") or 0.0)
    ratio = on_foot / float(candidate.get("official_new_miles") or candidate.get("official_miles") or 1.0)
    return on_foot + route_count_weight + max(0.0, ratio - 2.2)


def select_package_candidates(
    original_components: list[dict[str, Any]],
    combo_candidates: list[dict[str, Any]],
    original_candidate_index: dict[str, dict[str, Any]],
    route_count_weight: float,
) -> list[dict[str, Any]]:
    pool = []
    for component in original_components:
        candidate = dict(original_candidate_index[str(component["candidate_id"])])
        candidate["block_id"] = component.get("block_id")
        candidate["block_name"] = component.get("block_name")
        pool.append(candidate)
    pool.extend(combo_candidates)
    segment_ids = sorted({segment_id for candidate in pool for segment_id in candidate_segment_ids(candidate)})
    segment_index = {segment_id: index for index, segment_id in enumerate(segment_ids)}
    coverage = np.zeros((len(segment_ids), len(pool)))
    for candidate_index, candidate in enumerate(pool):
        for segment_id in candidate_segment_ids(candidate):
            coverage[segment_index[segment_id], candidate_index] = 1.0
    costs = np.array([local_candidate_cost(candidate, route_count_weight) for candidate in pool])
    result = milp(
        c=costs,
        constraints=LinearConstraint(
            coverage,
            lb=np.ones(len(segment_ids)),
            ub=np.full(len(segment_ids), np.inf),
        ),
        bounds=Bounds(0, 1),
        integrality=np.ones(len(pool)),
        options={"time_limit": 20, "mip_rel_gap": 0.0},
    )
    if not result.success:
        return [original_candidate_index[str(component["candidate_id"])] for component in original_components]
    return [candidate for candidate, value in zip(pool, result.x) if value > 0.5]


def route_row(
    route_number: int,
    candidate: dict[str, Any],
    package: dict[str, Any],
) -> dict[str, Any]:
    official = float(candidate.get("official_new_miles") or 0.0)
    on_foot = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
    return {
        "route_number": route_number,
        "candidate_id": candidate.get("candidate_id"),
        "block_id": package["block_id"],
        "block_name": package["block_name"],
        "trail_names": candidate.get("trail_names") or [],
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "total_minutes": candidate.get("total_minutes"),
        "trailhead": (candidate.get("trailhead") or {}).get("name"),
        "route_status": candidate.get("route_status"),
        "less_optimal_flags": candidate.get("less_optimal_flags") or [],
        "segment_ids": candidate.get("segment_ids") or [],
        "source_component_candidate_ids": source_component_ids(candidate),
        "is_combo": bool(candidate.get("source_component_candidate_ids")),
    }


def build_combo_route_pass(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    original_candidate_index: dict[str, dict[str, Any]],
    trail_by_name: dict[str, dict[str, Any]],
    context: dict[str, Any],
    max_combo_size: int = 5,
    max_extra_miles_per_saved_route: float = 2.0,
    route_count_weight: float = 2.0,
) -> dict[str, Any]:
    selected_routes = []
    generated_combo_index = {}
    combo_count = 0
    original_component_count = 0
    for package in package_pass.get("packages") or []:
        components = [
            {
                **component,
                "block_id": package["block_id"],
                "block_name": package["block_name"],
            }
            for component in package.get("components") or []
        ]
        original_component_count += len(components)
        combos = generate_combo_candidates(
            package,
            original_candidate_index,
            trail_by_name,
            context,
            max_combo_size=max_combo_size,
            max_extra_miles_per_saved_route=max_extra_miles_per_saved_route,
        )
        for combo in combos:
            generated_combo_index[str(combo["candidate_id"])] = combo
        chosen = select_package_candidates(
            components,
            combos,
            original_candidate_index,
            route_count_weight=route_count_weight,
        )
        combo_count += sum(1 for candidate in chosen if candidate.get("source_component_candidate_ids"))
        selected_routes.extend((package, candidate) for candidate in chosen)

    rows = [
        route_row(index, candidate, package)
        for index, (package, candidate) in enumerate(selected_routes, start=1)
    ]
    covered_segments = {int(segment_id) for row in rows for segment_id in row.get("segment_ids") or []}
    total_on_foot = sum(float(row["on_foot_miles"]) for row in rows)
    return {
        "planning_status": "block_combo_route_pass",
        "source_route_pass_status": route_pass.get("planning_status"),
        "summary": {
            "selected_route_count": len(rows),
            "original_component_route_count": original_component_count,
            "selected_combo_route_count": combo_count,
            "covered_segment_count": len(covered_segments),
            "total_on_foot_miles": round_miles(total_on_foot),
            "planwide_on_foot_to_official_ratio": round(total_on_foot / TOTAL_OFFICIAL_MILES, 2),
            "routes_under_1_official_mile": sum(1 for row in rows if float(row["official_miles"]) < 1),
            "routes_under_2_official_miles": sum(1 for row in rows if float(row["official_miles"]) < 2),
        },
        "routes": rows,
        "candidate_index": generated_combo_index,
        "caveats": [
            "This pass combines validated route components inside each route block when the combined route saves starts without adding more than the configured mileage allowance.",
            "Original graph-validated components remain available as fallback where combining would create a worse route.",
            "This is still a route-design draft; packages with high ratios or multiple trailheads need local GPX review before final scheduling.",
        ],
    }


def build_map_data(
    combo_pass: dict[str, Any],
    plan: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_index = load_candidate_index(plan)
    candidate_index.update(combo_pass.get("candidate_index") or {})
    route_features = []
    official_features = []
    parking_features = []
    validations = []
    for route in combo_pass.get("routes") or []:
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
                "rendered_passed": render_validation["passed"],
                "rendered_failures": render_validation["failures"],
            }
        )
    return {
        "summary": combo_pass["summary"],
        "routes": combo_pass["routes"],
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


def render_markdown(combo_pass: dict[str, Any]) -> str:
    summary = combo_pass["summary"]
    lines = [
        "# 2026 Block Combo Route Pass v1",
        "",
        "Status: block-package component combination draft.",
        "",
        "## Summary",
        "",
        f"- Selected routes: {summary['selected_route_count']}",
        f"- Original component routes: {summary['original_component_route_count']}",
        f"- Selected combo routes: {summary['selected_combo_route_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / 251",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- On-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Routes under 1 official mile: {summary['routes_under_1_official_mile']}",
        f"- Routes under 2 official miles: {summary['routes_under_2_official_miles']}",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in combo_pass.get("caveats") or [])
    lines.extend(
        [
            "",
            "## Routes",
            "",
            "| # | Block | Type | Route | Trailhead | Official mi | On-foot mi | Ratio | Replaces |",
            "|---:|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for route in combo_pass.get("routes") or []:
        route_type = "combo" if route.get("is_combo") else "component"
        replaces = ", ".join(route.get("source_component_candidate_ids") or [])
        lines.append(
            f"| {route['route_number']} | {route['block_name']} | {route_type} | "
            f"{', '.join(route['trail_names'])} | {route.get('trailhead') or ''} | "
            f"{route['official_miles']} | {route['on_foot_miles']} | {route['ratio']} | {replaces} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--route-pass-json", type=Path, default=DEFAULT_ROUTE_PASS_JSON)
    parser.add_argument("--package-pass-json", type=Path, default=DEFAULT_PACKAGE_PASS_JSON)
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
    parser.add_argument("--max-combo-size", type=int, default=5)
    parser.add_argument("--max-extra-miles-per-saved-route", type=float, default=2.0)
    parser.add_argument("--route-count-weight", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.plan_json)
    route_pass = read_json(args.route_pass_json)
    package_pass = read_json(args.package_pass_json)
    context = load_planning_context(args)
    official_segments, _meta = load_official_segments(args.official_geojson)
    trails = group_remaining_by_trail(official_segments)
    trail_by_name = {normalize_trail_name(trail["trail_name"]): trail for trail in trails}
    original_candidate_index = load_candidate_index(plan)
    combo_pass = build_combo_route_pass(
        route_pass,
        package_pass,
        original_candidate_index,
        trail_by_name,
        context,
        max_combo_size=args.max_combo_size,
        max_extra_miles_per_saved_route=args.max_extra_miles_per_saved_route,
        route_count_weight=args.route_count_weight,
    )
    official_index = load_official_segment_index(args.official_geojson)
    map_data = build_map_data(combo_pass, plan, official_index, context["connector_graph"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_json_path = args.output_dir / f"{args.basename}-map-data.json"
    map_html_path = args.output_dir / f"{args.basename}-map.html"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, combo_pass)
    md_path.write_text(render_markdown(combo_pass), encoding="utf-8")
    write_json(map_json_path, map_data)
    map_html_path.write_text(render_html(map_data), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[args.plan_json, args.route_pass_json, args.package_pass_json, args.blocks_json],
            outputs=[json_path, md_path, map_json_path, map_html_path],
            command="block_combo_route_pass.py",
            metadata={
                "selected_route_count": combo_pass["summary"]["selected_route_count"],
                "covered_segment_count": combo_pass["summary"]["covered_segment_count"],
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
