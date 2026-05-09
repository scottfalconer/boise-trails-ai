#!/usr/bin/env python3
"""Select the best block route from combo-package and assembled-block passes."""

from __future__ import annotations

import argparse
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
from block_combo_route_pass import build_map_data, render_html  # noqa: E402
from export_execution_gpx import load_official_segment_index  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    load_connector_graph,
    load_official_segments,
    read_json,
)
from route_block_planner import build_block_index, normalize_trail_name  # noqa: E402


DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_BLOCKS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-blocks-v1.json"
DEFAULT_COMBO_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-combo-route-pass-v1.json"
DEFAULT_COMBO_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-combo-day-package-pass-v1.json"
DEFAULT_ASSEMBLED_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-assembled-route-pass-v1.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "block-hybrid-route-pass-v1"
TOTAL_OFFICIAL_MILES = 164.43


def round_miles(value: float) -> float:
    return round(float(value), 2)


def should_use_assembled(
    package: dict[str, Any],
    assembled_route: dict[str, Any] | None,
    max_extra_miles_per_saved_route: float,
    required_segment_ids: set[int] | None = None,
) -> tuple[bool, str]:
    if not assembled_route:
        return False, "no_assembled_route"
    if assembled_route.get("route_status") != "graph_validated":
        return False, "assembled_not_graph_validated"
    if required_segment_ids:
        assembled_segment_ids = {int(segment_id) for segment_id in assembled_route.get("segment_ids") or []}
        if not required_segment_ids.issubset(assembled_segment_ids):
            return False, "assembled_does_not_cover_component_segments"
    package_miles = float(package.get("on_foot_miles") or 0.0)
    assembled_miles = float(assembled_route.get("on_foot_miles") or 0.0)
    saved_routes = max(0, int(package.get("component_route_count") or 0) - 1)
    if assembled_miles <= package_miles:
        return True, "assembled_reduces_mileage"
    if saved_routes and assembled_miles <= package_miles + max_extra_miles_per_saved_route * saved_routes:
        return True, "assembled_reduces_route_starts_with_acceptable_extra_miles"
    return False, "component_package_is_more_efficient"


def package_segment_ids(
    package: dict[str, Any],
    combo_rows_by_id: dict[str, dict[str, Any]],
) -> set[int]:
    segment_ids = set()
    for component in package.get("components") or []:
        source_route = combo_rows_by_id.get(str(component["candidate_id"]), component)
        segment_ids.update(int(segment_id) for segment_id in source_route.get("segment_ids") or [])
    return segment_ids


def segment_index_by_id(official_segments: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(segment["seg_id"]): segment for segment in official_segments}


def candidate_cross_block_miles(
    candidate: dict[str, Any],
    official_segment_index: dict[int, dict[str, Any]],
    trail_to_block: dict[str, dict[str, Any]],
) -> tuple[float, int]:
    block_id = str(candidate.get("block_id") or "")
    miles = 0.0
    count = 0
    for segment_id in candidate.get("segment_ids") or []:
        official_segment = official_segment_index[int(segment_id)]
        block = trail_to_block.get(normalize_trail_name(str(official_segment["trail_name"])))
        if block and str(block["block_id"]) != block_id:
            miles += float(official_segment.get("official_miles") or 0.0)
            count += 1
    return miles, count


def global_candidate_cost(
    candidate: dict[str, Any],
    official_segment_index: dict[int, dict[str, Any]],
    trail_to_block: dict[str, dict[str, Any]],
    route_count_weight: float,
    cross_block_mile_penalty: float,
) -> float:
    official = float(candidate.get("official_miles") or candidate.get("official_new_miles") or 0.0)
    on_foot = float(candidate.get("on_foot_miles") or candidate.get("estimated_total_on_foot_miles") or 0.0)
    cross_miles, _cross_count = candidate_cross_block_miles(candidate, official_segment_index, trail_to_block)
    tiny_penalty = 8.0 if official < 1.0 else 0.0
    small_penalty = 4.0 if official < 2.0 else 0.0
    source_bonus = -2.0 if candidate.get("route_source") == "assembled_block_route" else 0.0
    return on_foot + route_count_weight + tiny_penalty + small_penalty + cross_miles * cross_block_mile_penalty + source_bonus


def global_candidate_pool(
    combo_route_pass: dict[str, Any],
    assembled_pass: dict[str, Any],
    combo_package_pass: dict[str, Any] | None = None,
    max_extra_assembled_miles: float = 0.0,
    official_segment_index: dict[int, dict[str, Any]] | None = None,
    trail_to_block: dict[str, dict[str, Any]] | None = None,
    allow_cross_block_candidates: bool = False,
) -> list[dict[str, Any]]:
    pool = []
    seen = set()
    package_by_block = {
        str(package["block_id"]): package
        for package in (combo_package_pass or {}).get("packages") or []
    }
    combo_rows_by_id = {
        str(route["candidate_id"]): route
        for route in combo_route_pass.get("routes") or []
    }
    for route in combo_route_pass.get("routes") or []:
        if route.get("route_status") != "graph_validated":
            continue
        candidate = {
            **route,
            "route_source": "combo_package_component",
            "selection_reason": "global_set_cover_selected_component",
        }
        if official_segment_index and trail_to_block and not allow_cross_block_candidates:
            cross_miles, _cross_count = candidate_cross_block_miles(candidate, official_segment_index, trail_to_block)
            if cross_miles > 0:
                continue
        pool.append(candidate)
        seen.add(str(candidate["candidate_id"]))
    for route in assembled_pass.get("routes") or []:
        if route.get("route_status") != "graph_validated":
            continue
        package = package_by_block.get(str(route.get("block_id")))
        if package:
            package_miles = float(package.get("on_foot_miles") or 0.0)
            assembled_miles = float(route.get("on_foot_miles") or 0.0)
            package_cross_miles = 0.0
            if official_segment_index and trail_to_block:
                for component in package.get("components") or []:
                    source_route = combo_rows_by_id.get(str(component["candidate_id"]), component)
                    cross_miles, _cross_count = candidate_cross_block_miles(
                        source_route,
                        official_segment_index,
                        trail_to_block,
                    )
                    package_cross_miles += cross_miles
            package_segments = package_segment_ids(package, combo_rows_by_id)
            assembled_segments = {int(segment_id) for segment_id in route.get("segment_ids") or []}
            assembled_adds_needed_segments = not assembled_segments.issubset(package_segments)
            clean_package_comparison = package_cross_miles == 0.0 and not assembled_adds_needed_segments
            if clean_package_comparison and assembled_miles > package_miles + max_extra_assembled_miles:
                continue
        candidate = {
            **route,
            "route_source": "assembled_block_route",
            "selection_reason": "global_set_cover_selected_block_route",
        }
        if official_segment_index and trail_to_block and not allow_cross_block_candidates:
            cross_miles, _cross_count = candidate_cross_block_miles(candidate, official_segment_index, trail_to_block)
            if cross_miles > 0:
                continue
        if str(candidate["candidate_id"]) in seen:
            continue
        pool.append(candidate)
    return pool


def select_global_candidates(
    pool: list[dict[str, Any]],
    official_segments: list[dict[str, Any]],
    blocks_config: dict[str, Any],
    route_count_weight: float,
    cross_block_mile_penalty: float,
    time_limit_seconds: int,
    required_segment_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    official_segment_ids = {int(segment["seg_id"]) for segment in official_segments}
    all_segment_ids = sorted(required_segment_ids or official_segment_ids)
    unknown_required_ids = sorted(set(all_segment_ids) - official_segment_ids)
    if unknown_required_ids:
        raise ValueError(f"Required segment ids are not in official data: {unknown_required_ids}")
    segment_row_index = {segment_id: index for index, segment_id in enumerate(all_segment_ids)}
    coverage = np.zeros((len(all_segment_ids), len(pool)))
    for candidate_index, candidate in enumerate(pool):
        for segment_id in candidate.get("segment_ids") or []:
            segment_id = int(segment_id)
            if segment_id in segment_row_index:
                coverage[segment_row_index[segment_id], candidate_index] = 1.0
    official_index = segment_index_by_id(official_segments)
    trail_to_block, _duplicates = build_block_index(blocks_config)
    costs = np.array(
        [
            global_candidate_cost(
                candidate,
                official_index,
                trail_to_block,
                route_count_weight=route_count_weight,
                cross_block_mile_penalty=cross_block_mile_penalty,
            )
            for candidate in pool
        ]
    )
    result = milp(
        c=costs,
        constraints=LinearConstraint(
            coverage,
            lb=np.ones(len(all_segment_ids)),
            ub=np.full(len(all_segment_ids), np.inf),
        ),
        bounds=Bounds(0, 1),
        integrality=np.ones(len(pool)),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"Hybrid global set cover failed: {result.message}")
    return [candidate for candidate, value in zip(pool, result.x) if value > 0.5]


def build_global_hybrid_route_pass(
    combo_route_pass: dict[str, Any],
    assembled_pass: dict[str, Any],
    official_segments: list[dict[str, Any]],
    blocks_config: dict[str, Any],
    combo_package_pass: dict[str, Any] | None = None,
    required_segment_ids: set[int] | None = None,
    max_extra_assembled_miles: float = 0.0,
    allow_cross_block_candidates: bool = False,
    route_count_weight: float = 4.0,
    cross_block_mile_penalty: float = 12.0,
    time_limit_seconds: int = 60,
) -> dict[str, Any]:
    official_index = segment_index_by_id(official_segments)
    official_segment_ids = set(official_index)
    target_segment_ids = set(required_segment_ids or official_segment_ids)
    target_official_miles = sum(
        float(official_index[segment_id].get("official_miles") or 0.0)
        for segment_id in target_segment_ids
        if segment_id in official_index
    )
    trail_to_block, _duplicates = build_block_index(blocks_config)
    pool = global_candidate_pool(
        combo_route_pass,
        assembled_pass,
        combo_package_pass=combo_package_pass,
        max_extra_assembled_miles=max_extra_assembled_miles,
        official_segment_index=official_index,
        trail_to_block=trail_to_block,
        allow_cross_block_candidates=allow_cross_block_candidates,
    )
    selected = select_global_candidates(
        pool,
        official_segments,
        blocks_config,
        route_count_weight=route_count_weight,
        cross_block_mile_penalty=cross_block_mile_penalty,
        time_limit_seconds=time_limit_seconds,
        required_segment_ids=target_segment_ids,
    )
    rows = []
    selection_decisions = []
    for route_number, route in enumerate(selected, start=1):
        row = row_from_route(
            route_number,
            route,
            str(route["block_id"]),
            str(route["block_name"]),
            str(route["route_source"]),
            str(route["selection_reason"]),
        )
        cross_miles, cross_count = candidate_cross_block_miles(row, official_index, trail_to_block)
        row["cross_block_official_miles"] = round_miles(cross_miles)
        row["cross_block_segment_count"] = cross_count
        rows.append(row)
        selection_decisions.append(
            {
                "candidate_id": row["candidate_id"],
                "block_name": row["block_name"],
                "selected_source": row["route_source"],
                "selection_reason": row["selection_reason"],
                "on_foot_miles": row["on_foot_miles"],
                "cross_block_official_miles": row["cross_block_official_miles"],
            }
        )
    covered_segments = {int(segment_id) for row in rows for segment_id in row.get("segment_ids") or []}
    covered_required_segments = covered_segments & target_segment_ids
    total_on_foot = sum(float(row["on_foot_miles"]) for row in rows)
    return {
        "planning_status": "block_hybrid_route_pass",
        "summary": {
            "selected_route_count": len(rows),
            "assembled_block_route_count": sum(1 for row in rows if row["route_source"] == "assembled_block_route"),
            "combo_component_route_count": sum(1 for row in rows if row["route_source"] == "combo_package_component"),
            "target_segment_count": len(target_segment_ids),
            "covered_segment_count": len(covered_required_segments),
            "covered_route_segment_count": len(covered_segments),
            "target_official_miles": round_miles(target_official_miles),
            "total_on_foot_miles": round_miles(total_on_foot),
            "planwide_on_foot_to_official_ratio": round(total_on_foot / target_official_miles, 2)
            if target_official_miles
            else None,
            "routes_under_1_official_mile": sum(1 for row in rows if float(row["official_miles"]) < 1),
            "routes_under_2_official_miles": sum(1 for row in rows if float(row["official_miles"]) < 2),
            "non_graph_validated_route_count": sum(1 for row in rows if row.get("route_status") != "graph_validated"),
            "cross_block_official_miles": round_miles(sum(float(row["cross_block_official_miles"]) for row in rows)),
            "cross_block_route_count": sum(1 for row in rows if float(row["cross_block_official_miles"]) > 0),
        },
        "routes": rows,
        "selection_decisions": selection_decisions,
        "candidate_index": {
            **(combo_route_pass.get("candidate_index") or {}),
            **(assembled_pass.get("candidate_index") or {}),
        },
        "caveats": [
            "This pass solves a global set-cover over selected combo components and graph-validated assembled block routes.",
            "The objective penalizes cross-block sweep routes so natural trail-system blocks win when mileage is comparable.",
            "Cross-block sweep candidates are excluded by default; enable allow-cross-block-candidates only for diagnostic sweeps.",
            "Assembled block routes are only eligible when they do not add on-foot miles relative to the component package, unless the max-extra-assembled-miles setting explicitly allows that tradeoff.",
            "Day-of conditions, signage, and final GPX continuity still need checking before executing a route.",
        ],
    }


def required_segment_ids_from_plan(plan: dict[str, Any], official_segments: list[dict[str, Any]]) -> set[int]:
    """Return the active required segment set from the planner output.

    Clean-start plans list every official segment as remaining. After progress is
    applied, `remaining_trails` is the source of truth for the active route menu.
    """

    required_ids = {
        int(segment_id)
        for trail in plan.get("remaining_trails") or []
        for segment_id in trail.get("remaining_segment_ids") or []
    }
    if required_ids:
        return required_ids
    return {int(segment["seg_id"]) for segment in official_segments}


def row_from_route(
    route_number: int,
    route: dict[str, Any],
    block_id: str,
    block_name: str,
    source: str,
    selection_reason: str,
) -> dict[str, Any]:
    official = float(route.get("official_miles") or route.get("official_new_miles") or 0.0)
    on_foot = float(route.get("on_foot_miles") or route.get("estimated_total_on_foot_miles") or 0.0)
    return {
        "route_number": route_number,
        "candidate_id": route.get("candidate_id"),
        "block_id": block_id,
        "block_name": block_name,
        "route_source": source,
        "selection_reason": selection_reason,
        "trail_names": route.get("trail_names") or [],
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "total_minutes": route.get("total_minutes"),
        "trailhead": route.get("trailhead"),
        "route_status": route.get("route_status"),
        "less_optimal_flags": route.get("less_optimal_flags") or [],
        "segment_ids": route.get("segment_ids") or [],
        "source_component_candidate_ids": route.get("source_component_candidate_ids") or [route.get("candidate_id")],
        "is_hybrid_assembled": source == "assembled_block_route",
    }


def build_hybrid_route_pass(
    combo_route_pass: dict[str, Any],
    combo_package_pass: dict[str, Any],
    assembled_pass: dict[str, Any],
    max_extra_miles_per_saved_route: float = 2.0,
) -> dict[str, Any]:
    combo_rows_by_id = {
        str(route["candidate_id"]): route
        for route in combo_route_pass.get("routes") or []
    }
    assembled_by_block = {
        str(route["block_id"]): route
        for route in assembled_pass.get("routes") or []
    }
    selected_rows = []
    decisions = []
    for package in combo_package_pass.get("packages") or []:
        block_id = str(package["block_id"])
        block_name = str(package["block_name"])
        assembled_route = assembled_by_block.get(block_id)
        required_segment_ids = package_segment_ids(package, combo_rows_by_id)
        use_assembled, reason = should_use_assembled(
            package,
            assembled_route,
            max_extra_miles_per_saved_route=max_extra_miles_per_saved_route,
            required_segment_ids=required_segment_ids,
        )
        if use_assembled and assembled_route:
            selected_rows.append(
                row_from_route(
                    len(selected_rows) + 1,
                    assembled_route,
                    block_id,
                    block_name,
                    "assembled_block_route",
                    reason,
                )
            )
        else:
            for component in package.get("components") or []:
                source_route = combo_rows_by_id.get(str(component["candidate_id"]), component)
                selected_rows.append(
                    row_from_route(
                        len(selected_rows) + 1,
                        source_route,
                        block_id,
                        block_name,
                        "combo_package_component",
                        reason,
                    )
                )
        decisions.append(
            {
                "block_id": block_id,
                "block_name": block_name,
                "selected_source": "assembled_block_route" if use_assembled else "combo_package_components",
                "selection_reason": reason,
                "combo_component_count": package.get("component_route_count"),
                "combo_on_foot_miles": package.get("on_foot_miles"),
                "assembled_on_foot_miles": assembled_route.get("on_foot_miles") if assembled_route else None,
                "assembled_route_status": assembled_route.get("route_status") if assembled_route else None,
            }
        )
    covered_segments = {int(segment_id) for row in selected_rows for segment_id in row.get("segment_ids") or []}
    total_on_foot = sum(float(row["on_foot_miles"]) for row in selected_rows)
    return {
        "planning_status": "block_hybrid_route_pass",
        "summary": {
            "selected_route_count": len(selected_rows),
            "assembled_block_route_count": sum(1 for row in selected_rows if row["route_source"] == "assembled_block_route"),
            "combo_component_route_count": sum(1 for row in selected_rows if row["route_source"] == "combo_package_component"),
            "covered_segment_count": len(covered_segments),
            "total_on_foot_miles": round_miles(total_on_foot),
            "planwide_on_foot_to_official_ratio": round(total_on_foot / TOTAL_OFFICIAL_MILES, 2),
            "routes_under_1_official_mile": sum(1 for row in selected_rows if float(row["official_miles"]) < 1),
            "routes_under_2_official_miles": sum(1 for row in selected_rows if float(row["official_miles"]) < 2),
            "non_graph_validated_route_count": sum(1 for row in selected_rows if row.get("route_status") != "graph_validated"),
        },
        "routes": selected_rows,
        "selection_decisions": decisions,
        "candidate_index": {
            **(combo_route_pass.get("candidate_index") or {}),
            **(assembled_pass.get("candidate_index") or {}),
        },
        "caveats": [
            "This pass chooses assembled one-block routes only when they are graph-validated and reduce mileage, or when they save route starts with a small mileage tradeoff.",
            "Combo package components remain selected where one-route block assembly is draft-only or creates excessive dead mileage.",
            "This is the current best route-design surface for human loop/block review; day-of conditions and signage still need checking.",
        ],
    }


def render_markdown(hybrid_pass: dict[str, Any]) -> str:
    summary = hybrid_pass["summary"]
    lines = [
        "# 2026 Block Hybrid Route Pass v1",
        "",
        "Status: best current block-route selection from combo packages and assembled block routes.",
        "",
        "## Summary",
        "",
        f"- Selected routes: {summary['selected_route_count']}",
        f"- Assembled block routes: {summary['assembled_block_route_count']}",
        f"- Combo component routes: {summary['combo_component_route_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / {summary.get('target_segment_count', 251)}",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- On-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Routes under 1 official mile: {summary['routes_under_1_official_mile']}",
        f"- Routes under 2 official miles: {summary['routes_under_2_official_miles']}",
        f"- Non-graph-validated routes: {summary['non_graph_validated_route_count']}",
        "",
        "## Routes",
        "",
        "| # | Block | Source | Route | Trailhead | Official mi | On-foot mi | Ratio | Why |",
        "|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for route in hybrid_pass.get("routes") or []:
        lines.append(
            f"| {route['route_number']} | {route['block_name']} | {route['route_source']} | "
            f"{', '.join(route.get('trail_names') or [])} | {route.get('trailhead') or ''} | "
            f"{route['official_miles']} | {route['on_foot_miles']} | {route['ratio']} | {route['selection_reason']} |"
        )
    lines.extend(["", "## Selection Decisions", ""])
    for decision in hybrid_pass.get("selection_decisions") or []:
        if "combo_on_foot_miles" in decision:
            detail = f"combo={decision['combo_on_foot_miles']}, assembled={decision['assembled_on_foot_miles']}"
        else:
            detail = f"on_foot={decision['on_foot_miles']}, cross_block={decision['cross_block_official_miles']}"
        lines.append(
            f"- {decision['block_name']}: {decision['selected_source']} "
            f"({decision['selection_reason']}; {detail})"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--blocks-json", type=Path, default=DEFAULT_BLOCKS_JSON)
    parser.add_argument("--combo-route-pass-json", type=Path, default=DEFAULT_COMBO_ROUTE_PASS_JSON)
    parser.add_argument("--combo-package-pass-json", type=Path, default=DEFAULT_COMBO_PACKAGE_PASS_JSON)
    parser.add_argument("--assembled-pass-json", type=Path, default=DEFAULT_ASSEMBLED_PASS_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-extra-miles-per-saved-route", type=float, default=2.0)
    parser.add_argument("--route-count-weight", type=float, default=4.0)
    parser.add_argument("--cross-block-mile-penalty", type=float, default=12.0)
    parser.add_argument("--max-extra-assembled-miles", type=float, default=0.0)
    parser.add_argument("--allow-cross-block-candidates", action="store_true")
    parser.add_argument("--time-limit-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.plan_json)
    combo_route_pass = read_json(args.combo_route_pass_json)
    combo_package_pass = read_json(args.combo_package_pass_json)
    assembled_pass = read_json(args.assembled_pass_json)
    blocks_config = read_json(args.blocks_json)
    official_segments, _meta = load_official_segments(args.official_geojson)
    required_segment_ids = required_segment_ids_from_plan(plan, official_segments)
    hybrid_pass = build_global_hybrid_route_pass(
        combo_route_pass,
        assembled_pass,
        official_segments,
        blocks_config,
        combo_package_pass=combo_package_pass,
        required_segment_ids=required_segment_ids,
        max_extra_assembled_miles=args.max_extra_assembled_miles,
        allow_cross_block_candidates=args.allow_cross_block_candidates,
        route_count_weight=args.route_count_weight,
        cross_block_mile_penalty=args.cross_block_mile_penalty,
        time_limit_seconds=args.time_limit_seconds,
    )
    official_index = load_official_segment_index(args.official_geojson)
    connector_graph = (
        load_connector_graph(args.connector_geojson, official_segments=official_segments)
        if args.connector_geojson.exists()
        else None
    )
    map_data = build_map_data(hybrid_pass, plan, official_index, connector_graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_json_path = args.output_dir / f"{args.basename}-map-data.json"
    map_html_path = args.output_dir / f"{args.basename}-map.html"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, hybrid_pass)
    md_path.write_text(render_markdown(hybrid_pass), encoding="utf-8")
    write_json(map_json_path, map_data)
    map_html_path.write_text(render_html(map_data), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[
                args.plan_json,
                args.blocks_json,
                args.combo_route_pass_json,
                args.combo_package_pass_json,
                args.assembled_pass_json,
            ],
            outputs=[json_path, md_path, map_json_path, map_html_path],
            command="block_hybrid_route_pass.py",
            metadata={
                "selected_route_count": hybrid_pass["summary"]["selected_route_count"],
                "covered_segment_count": hybrid_pass["summary"]["covered_segment_count"],
                "total_on_foot_miles": hybrid_pass["summary"]["total_on_foot_miles"],
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
