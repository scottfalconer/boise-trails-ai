#!/usr/bin/env python3
"""Optimize route-card credit ownership against the physical traversal graph."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from cluster_level_repricing_audit import (  # noqa: E402
    UnionFind,
    display_path,
    edge_key,
    float_value,
    int_value,
    normalized_ids,
    official_segment_miles,
    optimize_set_cover,
    route_index,
    route_key,
    route_label,
    route_metrics,
    rounded,
    sort_id,
    sum_route_metrics,
    write_json,
)
from latent_credit_delta_repricing_audit import field_day_order  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "ownership-reassignment-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "ownership-reassignment-optimization-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "ownership-reassignment-optimization-audit-2026-05-12-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment.get("seg_id")): segment for segment in official_segments if segment.get("seg_id") is not None}


def segment_brief(
    segment_id: str,
    official_by_id: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
) -> dict[str, Any]:
    segment = official_by_id.get(str(segment_id), {})
    return {
        "seg_id": str(segment_id),
        "trail_name": segment.get("trail_name"),
        "seg_name": segment.get("seg_name"),
        "direction": segment.get("direction"),
        "official_miles": rounded(segment_miles.get(str(segment_id), segment.get("official_miles", 0.0))),
    }


def route_repeat_index(
    route_repeat_audit: dict[str, Any],
    routes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup = route_index(routes)
    rows: dict[str, dict[str, Any]] = {}
    for row in route_repeat_audit.get("routes") or []:
        route = lookup.get(str(row.get("outing_id") or "")) or lookup.get(str(row.get("label") or ""))
        if route:
            rows[route_key(route)] = row
    return rows


def route_claims(routes: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {route_key(route): set(normalized_ids(route.get("segment_ids") or [])) for route in routes}


def segment_claimants(claims_by_route: dict[str, set[str]]) -> dict[str, set[str]]:
    claims: dict[str, set[str]] = {}
    for key, segment_ids in claims_by_route.items():
        for segment_id in segment_ids:
            claims.setdefault(segment_id, set()).add(key)
    return claims


def physical_coverage_by_route(
    routes: list[dict[str, Any]],
    repeat_rows_by_route: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for route in routes:
        key = route_key(route)
        claimed_ids = set(normalized_ids(route.get("segment_ids") or []))
        repeat_row = repeat_rows_by_route.get(key) or {}
        actual_full_ids = set(normalized_ids(repeat_row.get("actual_full_segment_ids") or []))
        coverage[key] = claimed_ids | actual_full_ids
    return coverage


def segment_coverers(coverage_by_route: dict[str, set[str]]) -> dict[str, set[str]]:
    coverers: dict[str, set[str]] = {}
    for key, segment_ids in coverage_by_route.items():
        for segment_id in segment_ids:
            coverers.setdefault(segment_id, set()).add(key)
    return coverers


def add_edge(
    edges: dict[tuple[str, str], dict[str, Any]],
    left: str,
    right: str,
    *,
    segment_id: str,
    reason: str,
    segment_miles: dict[str, float],
) -> None:
    if left == right:
        return
    key = edge_key(left, right)
    edge = edges.setdefault(
        key,
        {
            "left": key[0],
            "right": key[1],
            "segment_ids": set(),
            "reasons": set(),
            "official_miles": 0.0,
        },
    )
    if segment_id not in edge["segment_ids"]:
        edge["official_miles"] += segment_miles.get(segment_id, 0.0)
    edge["segment_ids"].add(segment_id)
    edge["reasons"].add(reason)


def ownership_edges(
    routes: list[dict[str, Any]],
    claims_by_segment: dict[str, set[str]],
    coverers_by_segment: dict[str, set[str]],
    segment_miles: dict[str, float],
) -> list[dict[str, Any]]:
    route_keys = {route_key(route) for route in routes}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for segment_id, current_owners in claims_by_segment.items():
        coverers = coverers_by_segment.get(segment_id, set()) & route_keys
        related = (current_owners & route_keys) | coverers
        if len(related) < 2:
            continue
        for left, right in itertools.combinations(sorted(related, key=sort_id), 2):
            reason = (
                "current_owner_alternative_cover"
                if left in current_owners or right in current_owners
                else "shared_physical_cover"
            )
            add_edge(edges, left, right, segment_id=segment_id, reason=reason, segment_miles=segment_miles)
    rows = []
    for edge in edges.values():
        rows.append(
            {
                "left": edge["left"],
                "right": edge["right"],
                "reasons": sorted(edge["reasons"]),
                "segment_ids": normalized_ids(edge["segment_ids"]),
                "official_miles": rounded(edge["official_miles"]),
                "weight": rounded(edge["official_miles"]),
            }
        )
    return sorted(rows, key=lambda row: (-float_value(row["weight"]), row["left"], row["right"]))


def components_from_edges(routes: list[dict[str, Any]], edge_rows: list[dict[str, Any]]) -> list[list[str]]:
    keys = [route_key(route) for route in routes]
    union_find = UnionFind(keys)
    for edge in edge_rows:
        union_find.union(edge["left"], edge["right"])
    groups: dict[str, list[str]] = {}
    for key in keys:
        groups.setdefault(union_find.find(key), []).append(key)
    return sorted([sorted(group, key=sort_id) for group in groups.values()], key=lambda group: (-len(group), group[0]))


def order_index(route_key_value: str, order: dict[str, dict[str, Any]]) -> int:
    row = order.get(route_key_value) or {}
    if row.get("order_index") is None:
        return 10_000
    return int(row["order_index"])


def choose_optimized_owner(
    segment_id: str,
    selected_keys: set[str],
    coverers_by_segment: dict[str, set[str]],
    routes_by_key: dict[str, dict[str, Any]],
    order: dict[str, dict[str, Any]],
) -> str | None:
    coverers = selected_keys & coverers_by_segment.get(segment_id, set())
    if not coverers:
        return None
    return min(
        coverers,
        key=lambda key: (
            order_index(key, order),
            float_value(routes_by_key[key].get("on_foot_miles")),
            key,
        ),
    )


def assignment_reason(optimized_owner: str, current_owners: set[str], order: dict[str, dict[str, Any]]) -> str:
    if optimized_owner in current_owners:
        return "current_owner_selected"
    if not current_owners:
        return "unclaimed_physical_segment"
    earliest_current = min((order_index(key, order) for key in current_owners), default=10_000)
    if order_index(optimized_owner, order) < earliest_current:
        return "earlier_selected_physical_coverer"
    return "current_owner_not_selected"


def build_segment_assignments(
    target_ids: list[str],
    selected_keys: set[str],
    claims_by_segment: dict[str, set[str]],
    coverers_by_segment: dict[str, set[str]],
    routes_by_key: dict[str, dict[str, Any]],
    order: dict[str, dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for segment_id in target_ids:
        current_owners = claims_by_segment.get(segment_id, set())
        optimized_owner = choose_optimized_owner(segment_id, selected_keys, coverers_by_segment, routes_by_key, order)
        coverers = coverers_by_segment.get(segment_id, set())
        rows.append(
            {
                **segment_brief(segment_id, official_by_id, segment_miles),
                "current_owner_route_keys": sorted(current_owners, key=sort_id),
                "current_owner_labels": [route_label(routes_by_key[key]) for key in sorted(current_owners, key=sort_id)],
                "optimized_owner_route_key": optimized_owner,
                "optimized_owner_label": route_label(routes_by_key[optimized_owner]) if optimized_owner else None,
                "physical_coverer_route_keys": sorted(coverers, key=sort_id),
                "physical_coverer_labels": [route_label(routes_by_key[key]) for key in sorted(coverers, key=sort_id)],
                "reassigned": optimized_owner is not None and optimized_owner not in current_owners,
                "assignment_reason": assignment_reason(optimized_owner, current_owners, order)
                if optimized_owner
                else "missing_selected_physical_coverer",
            }
        )
    return rows


def route_impact_rows(
    component_keys: list[str],
    selected_keys: set[str],
    assignments: list[dict[str, Any]],
    claims_by_route: dict[str, set[str]],
    routes_by_key: dict[str, dict[str, Any]],
    order: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    optimized_by_route: dict[str, set[str]] = {}
    for row in assignments:
        owner = row.get("optimized_owner_route_key")
        if owner:
            optimized_by_route.setdefault(str(owner), set()).add(str(row["seg_id"]))
    rows = []
    for key in component_keys:
        route = routes_by_key[key]
        current_ids = claims_by_route.get(key, set())
        optimized_ids = optimized_by_route.get(key, set())
        lost_ids = current_ids - optimized_ids
        gained_ids = optimized_ids - current_ids
        selected = key in selected_keys
        if not selected and current_ids:
            status = "removed_proven"
        elif selected and lost_ids and optimized_ids:
            status = "shrunk_unpriced"
        elif selected and not current_ids and optimized_ids:
            status = "new_credit_owner_same_physical_route"
        elif selected and gained_ids:
            status = "expanded_owner_same_physical_route"
        elif selected:
            status = "unchanged_selected"
        else:
            status = "not_selected_no_current_credit"
        if status in {"unchanged_selected", "not_selected_no_current_credit"} and not gained_ids:
            continue
        replacement_owner_keys = sorted(
            {
                str(row["optimized_owner_route_key"])
                for row in assignments
                if str(row["seg_id"]) in lost_ids and row.get("optimized_owner_route_key")
            },
            key=sort_id,
        )
        route_order = order_index(key, order)
        owner_orders = [order_index(owner, order) for owner in replacement_owner_keys]
        if not replacement_owner_keys:
            order_status = "not_applicable"
        elif all(owner_order <= route_order for owner_order in owner_orders):
            order_status = "current_calendar_skip_ready"
        else:
            order_status = "requires_calendar_reorder"
        rows.append(
            {
                "route_key": key,
                "route": route_metrics(route),
                "schedule": order.get(key),
                "selected_physical_route": selected,
                "status": status,
                "current_credit_segment_ids": normalized_ids(current_ids),
                "optimized_credit_segment_ids": normalized_ids(optimized_ids),
                "lost_credit_segment_ids": normalized_ids(lost_ids),
                "gained_credit_segment_ids": normalized_ids(gained_ids),
                "replacement_owner_route_keys": replacement_owner_keys,
                "replacement_owner_labels": [route_label(routes_by_key[owner]) for owner in replacement_owner_keys],
                "replacement_order_status": order_status,
                "saved_on_foot_miles": rounded(route.get("on_foot_miles")) if status == "removed_proven" else 0.0,
                "saved_p75_minutes": int_value(route.get("door_to_door_minutes_p75")) if status == "removed_proven" else 0,
                "saved_p90_minutes": int_value(route.get("door_to_door_minutes_p90")) if status == "removed_proven" else 0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["status"] != "removed_proven",
            row["status"] != "shrunk_unpriced",
            -float_value(row["saved_on_foot_miles"]),
            row["route"]["label"],
        ),
    )


def build_component_rows(
    components: list[list[str]],
    edge_rows: list[dict[str, Any]],
    routes_by_key: dict[str, dict[str, Any]],
    claims_by_route: dict[str, set[str]],
    claims_by_segment: dict[str, set[str]],
    coverage_by_route: dict[str, set[str]],
    coverers_by_segment: dict[str, set[str]],
    order: dict[str, dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
) -> list[dict[str, Any]]:
    edges_by_component_key = {}
    for edge in edge_rows:
        edge_nodes = {edge["left"], edge["right"]}
        for component in components:
            if edge_nodes <= set(component):
                edges_by_component_key.setdefault(tuple(component), []).append(edge)
                break
    rows = []
    for index, component in enumerate(components, start=1):
        component_routes = [routes_by_key[key] for key in component]
        target_ids = normalized_ids(
            segment_id
            for key in component
            for segment_id in claims_by_route.get(key, set())
        )
        candidates = [
            {
                **route_metrics(routes_by_key[key]),
                "coverage_ids": normalized_ids(coverage_by_route.get(key, set()) & set(target_ids)),
            }
            for key in component
        ]
        optimization = optimize_set_cover(candidates, target_ids) if target_ids else {
            "status": "no_target_segments",
            "exact": True,
            "selected_route_keys": [],
            "missing_segment_ids": [],
            "visited_states": 0,
        }
        current = sum_route_metrics(component_routes)
        if optimization["status"] == "incomplete_candidate_coverage":
            selected_keys = set(component)
            selected_routes = component_routes
            skipped_routes: list[dict[str, Any]] = []
            optimized = current
        else:
            selected_keys = set(optimization.get("selected_route_keys") or [])
            selected_routes = [routes_by_key[key] for key in component if key in selected_keys]
            skipped_routes = [routes_by_key[key] for key in component if key not in selected_keys]
            optimized = sum_route_metrics(selected_routes)
        assignments = build_segment_assignments(
            target_ids,
            selected_keys,
            claims_by_segment,
            coverers_by_segment,
            routes_by_key,
            order,
            official_by_id,
            segment_miles,
        )
        reassigned = [row for row in assignments if row["reassigned"]]
        impacts = route_impact_rows(
            component,
            selected_keys,
            assignments,
            claims_by_route,
            routes_by_key,
            order,
        )
        removed = [row for row in impacts if row["status"] == "removed_proven"]
        shrink = [row for row in impacts if row["status"] == "shrunk_unpriced"]
        skip_ready_removed = [
            row for row in removed if row["replacement_order_status"] == "current_calendar_skip_ready"
        ]
        savings = {
            "on_foot_miles": rounded(current["on_foot_miles"] - optimized["on_foot_miles"]),
            "door_to_door_minutes_p75": current["door_to_door_minutes_p75"] - optimized["door_to_door_minutes_p75"],
            "door_to_door_minutes_p90": current["door_to_door_minutes_p90"] - optimized["door_to_door_minutes_p90"],
        }
        skip_ready_savings = {
            "on_foot_miles": rounded(sum(float_value(row["saved_on_foot_miles"]) for row in skip_ready_removed)),
            "door_to_door_minutes_p75": sum(int_value(row["saved_p75_minutes"]) for row in skip_ready_removed),
            "door_to_door_minutes_p90": sum(int_value(row["saved_p90_minutes"]) for row in skip_ready_removed),
        }
        component_edges = edges_by_component_key.get(tuple(component), [])
        rows.append(
            {
                "component_id": f"C{index:02d}",
                "route_count": len(component),
                "segment_count": len(target_ids),
                "route_keys": component,
                "routes": [route_metrics(route) for route in component_routes],
                "edge_count": len(component_edges),
                "edge_weight": rounded(sum(float_value(edge["weight"]) for edge in component_edges)),
                "edges": component_edges,
                "target_segment_ids": target_ids,
                "candidate_loop_policy": "existing_certified_no_shuttle_route_cards_only",
                "optimization_status": optimization["status"],
                "optimization_exact": optimization["exact"],
                "visited_states": optimization["visited_states"],
                "current": current,
                "optimized": optimized,
                "order_free_savings": savings,
                "current_calendar_skip_ready_savings": skip_ready_savings,
                "selected_routes": [route_metrics(route) for route in selected_routes],
                "skipped_routes": [route_metrics(route) for route in skipped_routes],
                "segment_assignments": assignments,
                "reassigned_segments": reassigned,
                "reassigned_segment_count": len(reassigned),
                "reassigned_official_miles": rounded(
                    sum(segment_miles.get(str(row["seg_id"]), 0.0) for row in reassigned)
                ),
                "route_impacts": impacts,
                "removed_routes": removed,
                "partial_shrink_routes": shrink,
                "skip_ready_removed_routes": skip_ready_removed,
                "missing_segment_ids": optimization.get("missing_segment_ids") or [],
            }
        )
    return rows


def build_ownership_reassignment_optimization_audit(
    field_tool_data: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = {route_key(route): route for route in routes}
    repeat_rows_by_route = route_repeat_index(route_repeat_audit, routes)
    claims_by_route = route_claims(routes)
    claims_by_segment = segment_claimants(claims_by_route)
    coverage_by_route = physical_coverage_by_route(routes, repeat_rows_by_route)
    coverers_by_segment = segment_coverers(coverage_by_route)
    official_by_id = segment_index(official_segments)
    segment_miles = official_segment_miles(official_segments, field_tool_data)
    edge_rows = ownership_edges(routes, claims_by_segment, coverers_by_segment, segment_miles)
    components = components_from_edges(routes, edge_rows)
    order = field_day_order(field_tool_data)
    component_rows = build_component_rows(
        components,
        edge_rows,
        routes_by_key,
        claims_by_route,
        claims_by_segment,
        coverage_by_route,
        coverers_by_segment,
        order,
        official_by_id,
        segment_miles,
    )
    relevant_components = [
        row
        for row in component_rows
        if row["route_count"] > 1 and (row["reassigned_segment_count"] or row["removed_routes"] or row["partial_shrink_routes"])
    ]
    savings_components = [
        row for row in relevant_components if float_value(row["order_free_savings"]["on_foot_miles"]) > 0.01
    ]
    credit_only_components = [
        row
        for row in relevant_components
        if not float_value(row["order_free_savings"]["on_foot_miles"]) > 0.01 and row["reassigned_segment_count"]
    ]
    exact_components = [row for row in relevant_components if row["optimization_exact"]]
    current = {
        "route_count": sum(row["current"]["route_count"] for row in relevant_components),
        "on_foot_miles": rounded(sum(float_value(row["current"]["on_foot_miles"]) for row in relevant_components)),
        "door_to_door_minutes_p75": sum(row["current"]["door_to_door_minutes_p75"] for row in relevant_components),
        "door_to_door_minutes_p90": sum(row["current"]["door_to_door_minutes_p90"] for row in relevant_components),
    }
    optimized = {
        "route_count": sum(row["optimized"]["route_count"] for row in relevant_components),
        "on_foot_miles": rounded(sum(float_value(row["optimized"]["on_foot_miles"]) for row in relevant_components)),
        "door_to_door_minutes_p75": sum(row["optimized"]["door_to_door_minutes_p75"] for row in relevant_components),
        "door_to_door_minutes_p90": sum(row["optimized"]["door_to_door_minutes_p90"] for row in relevant_components),
    }
    skip_ready_savings = {
        "on_foot_miles": rounded(
            sum(float_value(row["current_calendar_skip_ready_savings"]["on_foot_miles"]) for row in relevant_components)
        ),
        "door_to_door_minutes_p75": sum(
            row["current_calendar_skip_ready_savings"]["door_to_door_minutes_p75"] for row in relevant_components
        ),
        "door_to_door_minutes_p90": sum(
            row["current_calendar_skip_ready_savings"]["door_to_door_minutes_p90"] for row in relevant_components
        ),
    }
    total_reassigned_segments = [
        assignment
        for row in relevant_components
        for assignment in row["reassigned_segments"]
    ]
    removed_routes = [impact for row in relevant_components for impact in row["removed_routes"]]
    shrink_routes = [impact for row in relevant_components for impact in row["partial_shrink_routes"]]
    status = (
        "ownership_reassignment_reduces_existing_loop_work"
        if savings_components and len(exact_components) == len(relevant_components)
        else "ownership_reassignment_changes_credit_only"
        if credit_only_components or total_reassigned_segments
        else "no_ownership_reassignment_value_found"
    )
    return {
        "schema": "boise_trails_ownership_reassignment_optimization_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "parameters": {
            "physical_traversal_source": "route-repeat audit actual_full_segment_ids plus current route-card claims",
            "credit_ownership_source": "field packet route segment_ids",
            "assignment_policy": "select minimum on-foot route-card cover, then assign each segment to the earliest selected physical coverer",
            "objective": "minimize total on-foot miles, tie-break by p75 minutes and route count",
        },
        "scope": {
            "proves": [
                "which route cards physically complete each currently claimed official segment",
                "which credit-owner changes are needed for the minimum existing-route-card physical cover",
                "which route cards can be removed outright inside an order-free recertified plan",
            ],
            "does_not_prove": [
                "official BTC progress before challenge-window activity validation",
                "field-ready replacement cards for partial shrink rows",
                "same-calendar skipping when replacement owner routes are scheduled later",
                "a global optimum over newly generated trail-system loops",
            ],
        },
        "summary": {
            "route_count": len(routes),
            "claimed_segment_count": len(claims_by_segment),
            "physically_covered_segment_count": len(coverers_by_segment),
            "ownership_edge_count": len(edge_rows),
            "component_count": len(component_rows),
            "relevant_component_count": len(relevant_components),
            "exact_optimized_relevant_component_count": len(exact_components),
            "components_with_order_free_savings_count": len(savings_components),
            "credit_only_reassignment_component_count": len(credit_only_components),
            "reassigned_segment_count": len(total_reassigned_segments),
            "reassigned_official_miles": rounded(
                sum(float_value(assignment["official_miles"]) for assignment in total_reassigned_segments)
            ),
            "order_free_removed_route_count": len(removed_routes),
            "current_calendar_skip_ready_removed_route_count": len(
                [row for row in removed_routes if row["replacement_order_status"] == "current_calendar_skip_ready"]
            ),
            "requires_calendar_reorder_removed_route_count": len(
                [row for row in removed_routes if row["replacement_order_status"] == "requires_calendar_reorder"]
            ),
            "partial_shrink_route_count": len(shrink_routes),
            "current_relevant_route_count": current["route_count"],
            "optimized_relevant_route_count": optimized["route_count"],
            "order_free_saved_route_count": current["route_count"] - optimized["route_count"],
            "current_relevant_on_foot_miles": current["on_foot_miles"],
            "optimized_relevant_on_foot_miles": optimized["on_foot_miles"],
            "order_free_saved_on_foot_miles": rounded(current["on_foot_miles"] - optimized["on_foot_miles"]),
            "order_free_saved_p75_minutes": current["door_to_door_minutes_p75"]
            - optimized["door_to_door_minutes_p75"],
            "order_free_saved_p90_minutes": current["door_to_door_minutes_p90"]
            - optimized["door_to_door_minutes_p90"],
            "current_calendar_skip_ready_saved_on_foot_miles": skip_ready_savings["on_foot_miles"],
            "current_calendar_skip_ready_saved_p75_minutes": skip_ready_savings["door_to_door_minutes_p75"],
            "current_calendar_skip_ready_saved_p90_minutes": skip_ready_savings["door_to_door_minutes_p90"],
        },
        "edges": edge_rows,
        "components": component_rows,
        "components_with_order_free_savings": sorted(
            savings_components,
            key=lambda row: -float_value(row["order_free_savings"]["on_foot_miles"]),
        ),
        "credit_only_reassignment_components": sorted(
            credit_only_components,
            key=lambda row: -float_value(row["reassigned_official_miles"]),
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Ownership Reassignment Optimization Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Ownership graph edges: {summary['ownership_edge_count']}",
        f"- Relevant components: {summary['relevant_component_count']}",
        f"- Exact optimized relevant components: {summary['exact_optimized_relevant_component_count']}",
        f"- Reassigned official segments: {summary['reassigned_segment_count']} ({summary['reassigned_official_miles']:.2f} mi)",
        f"- Order-free route cards: {summary['current_relevant_route_count']} -> {summary['optimized_relevant_route_count']} ({summary['order_free_saved_route_count']} removed)",
        f"- Order-free on-foot miles: {summary['current_relevant_on_foot_miles']:.2f} -> {summary['optimized_relevant_on_foot_miles']:.2f} ({summary['order_free_saved_on_foot_miles']:.2f} mi saved)",
        f"- Order-free p75/p90 saved: {summary['order_free_saved_p75_minutes']} / {summary['order_free_saved_p90_minutes']} min",
        f"- Current-calendar skip-ready savings: {summary['current_calendar_skip_ready_saved_on_foot_miles']:.2f} mi, {summary['current_calendar_skip_ready_saved_p75_minutes']} p75 min, {summary['current_calendar_skip_ready_saved_p90_minutes']} p90 min",
        f"- Partial shrink routes needing regenerated cards: {summary['partial_shrink_route_count']}",
        "",
        "## Components With Order-Free Savings",
        "",
    ]
    savings_components = audit.get("components_with_order_free_savings") or []
    if savings_components:
        lines.extend(
            [
                "| Component | Routes | Reassigned ids | Removed routes | Saved on-foot mi | Saved p75 | Calendar note |",
                "|---|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in savings_components:
            removed = ", ".join(impact["route"]["label"] for impact in row["removed_routes"])
            reorder_count = len(
                [
                    impact
                    for impact in row["removed_routes"]
                    if impact["replacement_order_status"] == "requires_calendar_reorder"
                ]
            )
            skip_ready_count = len(row["skip_ready_removed_routes"])
            if reorder_count and skip_ready_count:
                calendar_note = f"{skip_ready_count} skip-ready, {reorder_count} need reorder"
            elif reorder_count:
                calendar_note = f"{reorder_count} need reorder"
            elif skip_ready_count:
                calendar_note = f"{skip_ready_count} skip-ready"
            else:
                calendar_note = "order-free only"
            lines.append(
                f"| {row['component_id']} | {row['route_count']} | {row['reassigned_segment_count']} | {removed} | {float_value(row['order_free_savings']['on_foot_miles']):.2f} | {row['order_free_savings']['door_to_door_minutes_p75']} | {calendar_note} |"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Partial Shrink Credit Moves", ""])
    shrink_rows = [
        impact
        for component in audit.get("components") or []
        for impact in component.get("partial_shrink_routes") or []
    ]
    if shrink_rows:
        lines.extend(
            [
                "| Route | Lost credit ids | Retained credit ids | Replacement owner(s) | Current on-foot mi |",
                "|---|---|---|---|---:|",
            ]
        )
        for impact in shrink_rows[:20]:
            lines.append(
                f"| {impact['route']['label']} | {', '.join(impact['lost_credit_segment_ids'])} | {', '.join(impact['optimized_credit_segment_ids'])} | {', '.join(impact['replacement_owner_labels'])} | {float_value(impact['route']['on_foot_miles']):.2f} |"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- This audit separates physical traversal from credit ownership; a route can physically stay unchanged while some official segment credit moves to a different route card.",
            "- Order-free savings require a recertified calendar/field-packet promotion before becoming an executable menu change.",
            "- Current-calendar skip-ready savings are only counted when the replacement owner route is already scheduled no later than the removed route.",
            "- Partial shrink rows are deliberately unpriced until a regenerated replacement route card exists.",
            "- This audit does not mark BTC progress or replace challenge-window activity validation.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    audit = build_ownership_reassignment_optimization_audit(
        read_json(args.field_tool_data_json),
        read_json(args.route_repeat_audit_json),
        official_segments,
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "route_repeat_audit_json": display_path(args.route_repeat_audit_json),
            "official_geojson": display_path(args.official_geojson),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="ownership-reassignment-optimization-audit-2026-05-12",
        inputs=[args.field_tool_data_json, args.route_repeat_audit_json, args.official_geojson],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/ownership_reassignment_optimization_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "no_ownership_reassignment_value_found" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
