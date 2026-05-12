#!/usr/bin/env python3
"""Find latent/repeat route clusters and optimize current no-shuttle loop sets."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_LATENT_CREDIT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "field-latent-credit-audit-2026-05-11.json"
DEFAULT_ROUTE_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "cluster-level-repricing-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "cluster-level-repricing-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "cluster-level-repricing-audit-2026-05-12-manifest.json"


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


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def int_value(value: Any) -> int:
    return int(round(float_value(value)))


def rounded(value: Any, digits: int = 2) -> float:
    return round(float_value(value), digits)


def sort_id(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def normalized_ids(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("route_key") or route.get("label") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_metrics(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_key": route_key(route),
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "candidate_ids": route.get("candidate_ids") or [],
        "trailhead": route.get("trailhead"),
        "segment_ids": normalized_ids(route.get("segment_ids") or []),
        "official_miles": rounded(route.get("official_miles")),
        "on_foot_miles": rounded(route.get("on_foot_miles")),
        "door_to_door_minutes_p75": int_value(route.get("door_to_door_minutes_p75")),
        "door_to_door_minutes_p90": int_value(route.get("door_to_door_minutes_p90")),
    }


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for route in routes:
        keys = [route_key(route), str(route.get("outing_id") or ""), str(route.get("label") or ""), route_label(route)]
        for candidate_id in route.get("candidate_ids") or []:
            keys.append(str(candidate_id))
        for key in keys:
            if key:
                index[key] = route
    return index


def claimed_segment_index(routes: list[dict[str, Any]]) -> dict[str, set[str]]:
    claims: dict[str, set[str]] = {}
    for route in routes:
        key = route_key(route)
        for segment_id in normalized_ids(route.get("segment_ids") or []):
            claims.setdefault(segment_id, set()).add(key)
    return claims


def official_segment_miles(official_segments: list[dict[str, Any]], field_tool_data: dict[str, Any]) -> dict[str, float]:
    miles: dict[str, float] = {}
    for segment in official_segments:
        segment_id = str(segment.get("seg_id") or "")
        if segment_id:
            miles[segment_id] = float_value(segment.get("official_miles"))
    for route in field_tool_data.get("routes") or []:
        route_ids = normalized_ids(route.get("segment_ids") or [])
        if len(route_ids) == 1 and float_value(route.get("official_miles")):
            miles.setdefault(route_ids[0], float_value(route.get("official_miles")))
    return miles


def haversine_miles(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1 = left
    lon2, lat2 = right
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_miles * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parking_point(route: dict[str, Any]) -> tuple[float, float] | None:
    parking = route.get("parking") or {}
    if parking.get("lon") is None or parking.get("lat") is None:
        return None
    return float(parking["lon"]), float(parking["lat"])


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right), key=sort_id))


def add_edge(
    edges: dict[tuple[str, str], dict[str, Any]],
    left: str,
    right: str,
    *,
    reason: str,
    latent_miles: float = 0.0,
    repeat_miles: float = 0.0,
    proximity_weight: float = 0.0,
    segment_ids: list[str] | None = None,
    distance_miles: float | None = None,
) -> None:
    if left == right:
        return
    key = edge_key(left, right)
    edge = edges.setdefault(
        key,
        {
            "left": key[0],
            "right": key[1],
            "reasons": [],
            "latent_official_miles": 0.0,
            "repeat_official_miles": 0.0,
            "trailhead_proximity_weight": 0.0,
            "segment_ids": set(),
            "trailhead_distance_miles": None,
        },
    )
    if reason not in edge["reasons"]:
        edge["reasons"].append(reason)
    edge["latent_official_miles"] += float_value(latent_miles)
    edge["repeat_official_miles"] += float_value(repeat_miles)
    edge["trailhead_proximity_weight"] = max(float_value(edge["trailhead_proximity_weight"]), proximity_weight)
    edge["segment_ids"].update(normalized_ids(segment_ids or []))
    if distance_miles is not None:
        current = edge.get("trailhead_distance_miles")
        edge["trailhead_distance_miles"] = (
            rounded(distance_miles, 3) if current is None else min(float_value(current), rounded(distance_miles, 3))
        )


def latent_edges(
    latent_audit: dict[str, Any],
    routes_by_key: dict[str, dict[str, Any]],
    segment_miles: dict[str, float],
    edges: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for source_review in latent_audit.get("reconciled_routes") or []:
        source_route = routes_by_key.get(str(source_review.get("outing_id") or "")) or routes_by_key.get(
            str(source_review.get("label") or "")
        )
        if not source_route:
            continue
        source_key = route_key(source_route)
        for segment in source_review.get("segments") or []:
            if segment.get("status") != "reconciled_owned_elsewhere":
                continue
            segment_id = str(segment.get("seg_id") or "")
            if not segment_id:
                continue
            miles = float_value(segment.get("official_miles")) or segment_miles.get(segment_id, 0.0)
            for owner in segment.get("claimed_by_other_routes") or []:
                owner_route = routes_by_key.get(str(owner.get("outing_id") or "")) or routes_by_key.get(
                    str(owner.get("label") or "")
                )
                if owner_route:
                    add_edge(
                        edges,
                        source_key,
                        route_key(owner_route),
                        reason="latent_owned_elsewhere",
                        latent_miles=miles,
                        segment_ids=[segment_id],
                    )


def repeat_edges(
    route_repeat_audit: dict[str, Any],
    routes_by_key: dict[str, dict[str, Any]],
    claims: dict[str, set[str]],
    segment_miles: dict[str, float],
    edges: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for row in route_repeat_audit.get("routes") or []:
        source_route = routes_by_key.get(str(row.get("outing_id") or "")) or routes_by_key.get(str(row.get("label") or ""))
        if not source_route:
            continue
        source_key = route_key(source_route)
        for segment_id in normalized_ids(row.get("declared_repeat_segment_ids") or []):
            for owner_key in claims.get(segment_id, set()):
                if owner_key != source_key:
                    add_edge(
                        edges,
                        source_key,
                        owner_key,
                        reason="declared_official_repeat",
                        repeat_miles=segment_miles.get(segment_id, 0.0),
                        segment_ids=[segment_id],
                    )


def trailhead_edges(
    routes: list[dict[str, Any]],
    edges: dict[tuple[str, str], dict[str, Any]],
    *,
    proximity_threshold_miles: float,
) -> None:
    for left, right in itertools.combinations(routes, 2):
        left_key = route_key(left)
        right_key = route_key(right)
        left_name = str(left.get("trailhead") or "").strip().lower()
        right_name = str(right.get("trailhead") or "").strip().lower()
        left_point = parking_point(left)
        right_point = parking_point(right)
        distance = haversine_miles(left_point, right_point) if left_point and right_point else None
        same_name = bool(left_name and right_name and left_name == right_name)
        near = proximity_threshold_miles > 0 and distance is not None and distance <= proximity_threshold_miles
        if not same_name and not near:
            continue
        proximity_weight = (
            1.0
            if same_name
            else max(0.0, 1.0 - float_value(distance) / proximity_threshold_miles)
        )
        add_edge(
            edges,
            left_key,
            right_key,
            reason="shared_or_near_trailhead",
            proximity_weight=proximity_weight,
            distance_miles=distance,
        )


def normalized_edges(edges: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for edge in edges.values():
        rows.append(
            {
                "left": edge["left"],
                "right": edge["right"],
                "reasons": sorted(edge["reasons"]),
                "latent_official_miles": rounded(edge["latent_official_miles"]),
                "repeat_official_miles": rounded(edge["repeat_official_miles"]),
                "trailhead_proximity_weight": rounded(edge["trailhead_proximity_weight"], 3),
                "weight": rounded(
                    float_value(edge["latent_official_miles"])
                    + float_value(edge["repeat_official_miles"])
                    + float_value(edge["trailhead_proximity_weight"]),
                    3,
                ),
                "segment_ids": normalized_ids(edge["segment_ids"]),
                "trailhead_distance_miles": edge.get("trailhead_distance_miles"),
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
    return sorted(
        [sorted(group, key=sort_id) for group in groups.values()],
        key=lambda group: (-len(group), group[0]),
    )


def greedy_cover(candidates: list[dict[str, Any]], target_bits: int) -> tuple[float, tuple[int, ...], int]:
    covered = 0
    selected: list[int] = []
    total = 0.0
    while covered & target_bits != target_bits:
        best_index = None
        best_score = None
        for index, candidate in enumerate(candidates):
            new_bits = int(candidate["coverage_bits"]) & target_bits & ~covered
            if not new_bits:
                continue
            new_count = new_bits.bit_count()
            score = (float_value(candidate["on_foot_miles"]) / new_count, float_value(candidate["on_foot_miles"]))
            if best_score is None or score < best_score:
                best_index = index
                best_score = score
        if best_index is None:
            return math.inf, tuple(selected), covered
        selected.append(best_index)
        total += float_value(candidates[best_index]["on_foot_miles"])
        covered |= int(candidates[best_index]["coverage_bits"])
    return total, tuple(selected), covered


def optimize_set_cover(
    candidates: list[dict[str, Any]],
    target_ids: list[str],
    *,
    max_states: int = 500_000,
) -> dict[str, Any]:
    segment_bit = {segment_id: 1 << index for index, segment_id in enumerate(target_ids)}
    target_bits = (1 << len(target_ids)) - 1
    prepared = []
    for candidate in candidates:
        coverage_bits = 0
        for segment_id in candidate["coverage_ids"]:
            if segment_id in segment_bit:
                coverage_bits |= segment_bit[segment_id]
        if coverage_bits:
            prepared.append({**candidate, "coverage_bits": coverage_bits})
    covered_by_any = 0
    for candidate in prepared:
        covered_by_any |= int(candidate["coverage_bits"])
    if covered_by_any & target_bits != target_bits:
        missing = [
            segment_id
            for segment_id, bit in segment_bit.items()
            if not (covered_by_any & bit)
        ]
        return {
            "status": "incomplete_candidate_coverage",
            "selected_route_keys": [],
            "missing_segment_ids": missing,
            "exact": False,
            "visited_states": 0,
        }

    coverers: dict[int, list[int]] = {}
    for segment_id, bit in segment_bit.items():
        coverers[bit] = [
            index
            for index, candidate in enumerate(prepared)
            if int(candidate["coverage_bits"]) & bit
        ]
    greedy_cost, greedy_selected, _covered = greedy_cover(prepared, target_bits)
    best_cost = greedy_cost
    best_selected = greedy_selected
    best_tuple = (
        greedy_cost,
        sum(int_value(prepared[index]["door_to_door_minutes_p75"]) for index in greedy_selected),
        len(greedy_selected),
    )
    memo: dict[int, float] = {}
    visited = 0
    aborted = False

    def choose_uncovered_bit(covered: int) -> int:
        uncovered_bits = target_bits & ~covered
        bits = [1 << index for index in range(len(target_ids)) if uncovered_bits & (1 << index)]
        return min(bits, key=lambda bit: len(coverers[bit]))

    def search(covered: int, selected: tuple[int, ...], cost: float) -> None:
        nonlocal best_cost, best_selected, best_tuple, visited, aborted
        if aborted:
            return
        visited += 1
        if visited > max_states:
            aborted = True
            return
        if cost > best_cost + 1e-9:
            return
        prior = memo.get(covered)
        if prior is not None and prior <= cost + 1e-9:
            return
        memo[covered] = cost
        if covered & target_bits == target_bits:
            p75 = sum(int_value(prepared[index]["door_to_door_minutes_p75"]) for index in selected)
            score = (cost, p75, len(selected))
            if score < best_tuple:
                best_tuple = score
                best_cost = cost
                best_selected = selected
            return
        bit = choose_uncovered_bit(covered)
        options = sorted(
            coverers[bit],
            key=lambda index: (
                float_value(prepared[index]["on_foot_miles"])
                / max(1, (int(prepared[index]["coverage_bits"]) & target_bits & ~covered).bit_count()),
                float_value(prepared[index]["on_foot_miles"]),
            ),
        )
        selected_set = set(selected)
        for index in options:
            if index in selected_set:
                continue
            new_bits = int(prepared[index]["coverage_bits"]) & target_bits & ~covered
            if not new_bits:
                continue
            search(
                covered | int(prepared[index]["coverage_bits"]),
                (*selected, index),
                cost + float_value(prepared[index]["on_foot_miles"]),
            )

    search(0, tuple(), 0.0)
    selected_rows = [prepared[index] for index in best_selected]
    return {
        "status": "optimized_exact" if not aborted else "optimized_greedy_due_to_state_cap",
        "exact": not aborted,
        "selected_route_keys": [row["route_key"] for row in selected_rows],
        "missing_segment_ids": [],
        "visited_states": visited,
    }


def sum_route_metrics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "route_count": len(routes),
        "on_foot_miles": rounded(sum(float_value(route.get("on_foot_miles")) for route in routes)),
        "door_to_door_minutes_p75": sum(int_value(route.get("door_to_door_minutes_p75")) for route in routes),
        "door_to_door_minutes_p90": sum(int_value(route.get("door_to_door_minutes_p90")) for route in routes),
    }


def build_component_rows(
    components: list[list[str]],
    edge_rows: list[dict[str, Any]],
    routes_by_key: dict[str, dict[str, Any]],
    repeat_rows_by_route: dict[str, dict[str, Any]],
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
            for route in component_routes
            for segment_id in normalized_ids(route.get("segment_ids") or [])
        )
        candidates = []
        for route in component_routes:
            key = route_key(route)
            repeat_row = repeat_rows_by_route.get(key) or {}
            claimed_ids = set(normalized_ids(route.get("segment_ids") or []))
            actual_full_ids = set(normalized_ids(repeat_row.get("actual_full_segment_ids") or []))
            candidates.append(
                {
                    **route_metrics(route),
                    "coverage_ids": normalized_ids((claimed_ids | actual_full_ids) & set(target_ids)),
                }
            )
        optimization = optimize_set_cover(candidates, target_ids)
        selected_keys = set(optimization.get("selected_route_keys") or [])
        current = sum_route_metrics(component_routes)
        if optimization["status"] == "incomplete_candidate_coverage":
            selected_routes = component_routes
            skipped_routes: list[dict[str, Any]] = []
            optimized = current
        else:
            selected_routes = [routes_by_key[key] for key in component if key in selected_keys]
            skipped_routes = [routes_by_key[key] for key in component if key not in selected_keys]
            optimized = sum_route_metrics(selected_routes)
        savings = {
            "on_foot_miles": rounded(current["on_foot_miles"] - optimized["on_foot_miles"]),
            "door_to_door_minutes_p75": current["door_to_door_minutes_p75"] - optimized["door_to_door_minutes_p75"],
            "door_to_door_minutes_p90": current["door_to_door_minutes_p90"] - optimized["door_to_door_minutes_p90"],
        }
        component_edges = edges_by_component_key.get(tuple(component), [])
        rows.append(
            {
                "component_id": f"C{index:02d}",
                "route_count": len(component),
                "segment_count": len(target_ids),
                "route_keys": component,
                "routes": [route_metrics(route) for route in component_routes],
                "cluster_label": " / ".join(route.get("label") or route_key(route) for route in component_routes[:4]),
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
                "savings": savings,
                "selected_routes": [route_metrics(route) for route in selected_routes],
                "skipped_routes": [route_metrics(route) for route in skipped_routes],
                "missing_segment_ids": optimization.get("missing_segment_ids") or [],
            }
        )
    return rows


def build_cluster_level_repricing_audit(
    field_tool_data: dict[str, Any],
    latent_audit: dict[str, Any],
    route_repeat_audit: dict[str, Any],
    official_segments: list[dict[str, Any]],
    *,
    proximity_threshold_miles: float = 0.25,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = {route_key(route): route for route in routes}
    route_lookup = route_index(routes)
    claims = claimed_segment_index(routes)
    segment_miles = official_segment_miles(official_segments, field_tool_data)
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    latent_edges(latent_audit, route_lookup, segment_miles, edges)
    repeat_edges(route_repeat_audit, route_lookup, claims, segment_miles, edges)
    trailhead_edges(routes, edges, proximity_threshold_miles=proximity_threshold_miles)
    edge_rows = normalized_edges(edges)
    components = components_from_edges(routes, edge_rows)
    component_rows = build_component_rows(
        components,
        edge_rows,
        routes_by_key,
        {
            str(row.get("outing_id") or row.get("route_key")): row
            for row in route_repeat_audit.get("routes") or []
        },
    )
    relevant_components = [row for row in component_rows if row["route_count"] > 1]
    savings_components = [row for row in relevant_components if float_value(row["savings"]["on_foot_miles"]) > 0.01]
    exact_components = [row for row in relevant_components if row["optimization_exact"]]
    total_current = sum_route_metrics(
        [route for row in relevant_components for route in row["routes"]]
    )
    total_optimized = {
        "route_count": sum(row["optimized"]["route_count"] for row in relevant_components),
        "on_foot_miles": rounded(sum(row["optimized"]["on_foot_miles"] for row in relevant_components)),
        "door_to_door_minutes_p75": sum(row["optimized"]["door_to_door_minutes_p75"] for row in relevant_components),
        "door_to_door_minutes_p90": sum(row["optimized"]["door_to_door_minutes_p90"] for row in relevant_components),
    }
    total_savings = {
        "on_foot_miles": rounded(total_current["on_foot_miles"] - total_optimized["on_foot_miles"]),
        "door_to_door_minutes_p75": total_current["door_to_door_minutes_p75"]
        - total_optimized["door_to_door_minutes_p75"],
        "door_to_door_minutes_p90": total_current["door_to_door_minutes_p90"]
        - total_optimized["door_to_door_minutes_p90"],
    }
    status = (
        "optimized_existing_loop_clusters"
        if savings_components and len(exact_components) == len(relevant_components)
        else "optimized_with_search_limits"
        if savings_components
        else "no_existing_loop_cluster_savings"
    )
    return {
        "schema": "boise_trails_cluster_level_repricing_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "source_files": source_files or {},
        "parameters": {
            "proximity_threshold_miles": proximity_threshold_miles,
            "candidate_loop_policy": "existing certified no-shuttle field-packet route cards only",
            "objective": "minimize total on-foot miles, tie-break by p75 minutes and route count",
        },
        "scope": {
            "proves": [
                "connected components induced by latent credit, declared official repeat, and shared/near trailheads",
                "minimum-cost cover within the current certified no-shuttle route-card candidate set",
                "which current route cards are redundant inside those existing-loop clusters",
            ],
            "does_not_prove": [
                "a global optimum over newly generated trail-system loops",
                "field-ready replacement cards for clusters where the current candidate set is insufficient",
                "official BTC progress before challenge-window activity validation",
            ],
        },
        "summary": {
            "route_count": len(routes),
            "edge_count": len(edge_rows),
            "component_count": len(component_rows),
            "multi_route_component_count": len(relevant_components),
            "exact_optimized_component_count": len(exact_components),
            "components_with_existing_loop_savings_count": len(savings_components),
            "current_cluster_route_count": total_current["route_count"],
            "optimized_cluster_route_count": total_optimized["route_count"],
            "saved_route_count": total_current["route_count"] - total_optimized["route_count"],
            "current_cluster_on_foot_miles": total_current["on_foot_miles"],
            "optimized_cluster_on_foot_miles": total_optimized["on_foot_miles"],
            "saved_on_foot_miles": total_savings["on_foot_miles"],
            "saved_p75_minutes": total_savings["door_to_door_minutes_p75"],
            "saved_p90_minutes": total_savings["door_to_door_minutes_p90"],
        },
        "edges": edge_rows,
        "components": component_rows,
        "components_with_savings": sorted(
            savings_components,
            key=lambda row: -float_value(row["savings"]["on_foot_miles"]),
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Cluster-Level Repricing Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Graph edges: {summary['edge_count']}",
        f"- Multi-route components: {summary['multi_route_component_count']}",
        f"- Exact optimized components: {summary['exact_optimized_component_count']}",
        f"- Components with existing-loop savings: {summary['components_with_existing_loop_savings_count']}",
        f"- Existing-loop route cards: {summary['current_cluster_route_count']} -> {summary['optimized_cluster_route_count']} ({summary['saved_route_count']} removed)",
        f"- Existing-loop on-foot miles: {summary['current_cluster_on_foot_miles']:.2f} -> {summary['optimized_cluster_on_foot_miles']:.2f} ({summary['saved_on_foot_miles']:.2f} mi saved)",
        f"- Existing-loop p75 minutes saved: {summary['saved_p75_minutes']}",
        f"- Existing-loop p90 minutes saved: {summary['saved_p90_minutes']}",
        "",
        "## Existing-Loop Savings Components",
        "",
    ]
    savings_components = audit.get("components_with_savings") or []
    if savings_components:
        lines.extend(
            [
                "| Component | Routes | Segments | Saved routes | Saved on-foot mi | Saved p75 | Selected | Skipped |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in savings_components:
            selected = ", ".join(route["label"] for route in row["selected_routes"])
            skipped = ", ".join(route["label"] for route in row["skipped_routes"])
            lines.append(
                f"| {row['component_id']} | {row['route_count']} | {row['segment_count']} | {row['current']['route_count'] - row['optimized']['route_count']} | {float_value(row['savings']['on_foot_miles']):.2f} | {row['savings']['door_to_door_minutes_p75']} | {selected} | {skipped} |"
            )
    else:
        lines.append("- None in the current certified route-card candidate set.")
    lines.extend(["", "## Largest Components", ""])
    lines.extend(
        [
            "| Component | Routes | Edge weight | Current on-foot mi | Optimized on-foot mi | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(audit.get("components") or [], key=lambda item: (-item["route_count"], -float_value(item["edge_weight"])))[:12]:
        if row["route_count"] <= 1:
            continue
        lines.append(
            f"| {row['component_id']} | {row['route_count']} | {float_value(row['edge_weight']):.2f} | {float_value(row['current']['on_foot_miles']):.2f} | {float_value(row['optimized']['on_foot_miles']):.2f} | {row['optimization_status']} |"
        )
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- This is a cluster-level optimizer over existing certified no-shuttle route cards.",
            "- It is order-free cluster repricing, so promoting its skipped-route set would require a new calendar assignment and full field-packet recertification.",
            "- It does not generate new Harlow/Avimor, Freestone/Military, Hulls/Crestline, Bogus, or Cartwright/Polecat loops.",
            "- If a cluster still has high optimized on-foot cost, that is a candidate-universe problem: generate new no-shuttle loops for the whole component, then rerun this audit.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--latent-credit-audit-json", type=Path, default=DEFAULT_LATENT_CREDIT_AUDIT_JSON)
    parser.add_argument("--route-repeat-audit-json", type=Path, default=DEFAULT_ROUTE_REPEAT_AUDIT_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--proximity-threshold-miles", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_tool_data = read_json(args.field_tool_data_json)
    latent_audit = read_json(args.latent_credit_audit_json)
    route_repeat_audit = read_json(args.route_repeat_audit_json)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    audit = build_cluster_level_repricing_audit(
        field_tool_data,
        latent_audit,
        route_repeat_audit,
        official_segments,
        proximity_threshold_miles=args.proximity_threshold_miles,
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "latent_credit_audit_json": display_path(args.latent_credit_audit_json),
            "route_repeat_audit_json": display_path(args.route_repeat_audit_json),
            "official_geojson": display_path(args.official_geojson),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="cluster-level-repricing-audit-2026-05-12",
        inputs=[
            args.field_tool_data_json,
            args.latent_credit_audit_json,
            args.route_repeat_audit_json,
            args.official_geojson,
        ],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/cluster_level_repricing_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] == "no_existing_loop_cluster_savings" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
