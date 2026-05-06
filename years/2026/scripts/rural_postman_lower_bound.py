#!/usr/bin/env python3
"""Compute a Rural-Postman-style lower bound for official required segments.

This is intentionally a lower bound, not a route generator. It ignores parking,
trailhead access, day splits, heat, signage, and route-finding friction. For the
parity add-on it uses straight-line distances between odd required-graph
endpoints, which is optimistic compared with any real trail/road path and is
therefore safe as a mathematical lower bound.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    load_connector_graph,
    nearest_connector_node_for_graph,
)

DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = (
    YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
)
DEFAULT_CURRENT_PLAN_JSON = YEAR_DIR / "checkpoints" / "route-efficiency-audit-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "rural-postman-lower-bound-2026-05-06"
MILES_PER_FOOT = 1.0 / 5280.0


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def line_parts(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    coords = geometry.get("coordinates") or []
    if not coords:
        return []
    if geometry.get("type") == "LineString":
        return [[(float(lon), float(lat)) for lon, lat, *_ in coords]]
    if geometry.get("type") == "MultiLineString":
        return [
            [(float(lon), float(lat)) for lon, lat, *_ in part]
            for part in coords
            if len(part) >= 2
        ]
    return []


def line_length_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def official_feature_miles(props: dict[str, Any], parts: list[list[tuple[float, float]]]) -> float:
    length_ft = props.get("LengthFt")
    if length_ft is not None:
        return float(length_ft) * MILES_PER_FOOT
    return sum(line_length_miles(part) for part in parts)


def required_edges_from_geojson(official_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        parts = line_parts(feature.get("geometry") or {})
        if not parts:
            continue
        feature_miles = official_feature_miles(props, parts)
        part_lengths = [line_length_miles(part) for part in parts]
        total_part_length = sum(part_lengths)
        for part_index, coords in enumerate(parts):
            if len(coords) < 2:
                continue
            if len(parts) == 1 or total_part_length <= 0:
                official_miles = feature_miles
            else:
                official_miles = feature_miles * (part_lengths[part_index] / total_part_length)
            edges.append(
                {
                    "seg_id": int(props["segId"]),
                    "seg_name": str(props.get("segName") or props.get("segId")),
                    "direction": props.get("direction") or "both",
                    "start": coords[0],
                    "end": coords[-1],
                    "coordinates": coords,
                    "official_miles": official_miles,
                    "geometry_miles": part_lengths[part_index],
                    "part_index": part_index,
                    "part_count": len(parts),
                }
            )
    return edges


def snap_endpoint_nodes(
    edges: list[dict[str, Any]],
    snap_tolerance_miles: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    endpoints: list[tuple[float, float]] = []
    endpoint_refs: list[tuple[int, str]] = []
    for edge_index, edge in enumerate(edges):
        endpoint_refs.append((edge_index, "start_node"))
        endpoints.append(edge["start"])
        endpoint_refs.append((edge_index, "end_node"))
        endpoints.append(edge["end"])

    uf = UnionFind(len(endpoints))
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            if haversine_miles(endpoints[left], endpoints[right]) <= snap_tolerance_miles:
                uf.union(left, right)

    root_to_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(endpoints)):
        root_to_members[uf.find(index)].append(index)

    root_to_node_id = {root: node_id for node_id, root in enumerate(sorted(root_to_members))}
    node_info: dict[int, dict[str, Any]] = {}
    for root, members in root_to_members.items():
        node_id = root_to_node_id[root]
        lon = sum(endpoints[index][0] for index in members) / len(members)
        lat = sum(endpoints[index][1] for index in members) / len(members)
        max_snap = max(haversine_miles((lon, lat), endpoints[index]) for index in members)
        node_info[node_id] = {
            "lon": lon,
            "lat": lat,
            "endpoint_count": len(members),
            "max_snap_distance_miles": max_snap,
        }

    snapped_edges = [dict(edge) for edge in edges]
    for endpoint_index, (edge_index, key) in enumerate(endpoint_refs):
        snapped_edges[edge_index][key] = root_to_node_id[uf.find(endpoint_index)]
    return snapped_edges, node_info


def required_graph(edges: list[dict[str, Any]]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for edge in edges:
        graph.add_edge(
            edge["start_node"],
            edge["end_node"],
            weight=edge["official_miles"],
            seg_id=edge["seg_id"],
            seg_name=edge["seg_name"],
        )
    return graph


def odd_node_ids(graph: nx.MultiGraph) -> list[int]:
    return sorted(node for node, degree in graph.degree() if degree % 2 == 1)


def straight_line_min_matching(
    odd_nodes: list[int],
    node_info: dict[int, dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    if not odd_nodes:
        return 0.0, []
    if len(odd_nodes) % 2:
        raise ValueError("Odd node count must be even")
    complete = nx.Graph()
    for index, left in enumerate(odd_nodes):
        left_point = (node_info[left]["lon"], node_info[left]["lat"])
        for right in odd_nodes[index + 1 :]:
            right_point = (node_info[right]["lon"], node_info[right]["lat"])
            distance = haversine_miles(left_point, right_point)
            # max_weight_matching maximizes, so negate the distance.
            complete.add_edge(left, right, weight=-distance, distance=distance)
    matching = nx.algorithms.matching.max_weight_matching(complete, maxcardinality=True, weight="weight")
    pairs = []
    total = 0.0
    for left, right in sorted((min(a, b), max(a, b)) for a, b in matching):
        distance = float(complete[left][right]["distance"])
        total += distance
        pairs.append({"left_node": left, "right_node": right, "straight_line_miles": round(distance, 4)})
    return total, pairs


def connector_official_segments_from_required_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build official-repeat connector records without flattening multipart features."""
    return [
        {
            "seg_id": edge["seg_id"],
            "seg_name": edge["seg_name"],
            "trail_name": edge["seg_name"],
            "official_miles": edge["official_miles"],
            "direction": edge.get("direction") or "both",
            "coordinates": edge["coordinates"],
        }
        for edge in edges
    ]


def connector_components(connector_graph: dict[str, Any]) -> dict[tuple[float, float], int]:
    graph = connector_graph.get("graph") or {}
    nodes = set(connector_graph.get("nodes") or [])
    adjacency: dict[tuple[float, float], set[tuple[float, float]]] = defaultdict(set)
    for node, edges in graph.items():
        nodes.add(node)
        for edge in edges:
            other = edge["to"]
            nodes.add(other)
            adjacency[node].add(other)
            adjacency[other].add(node)

    component_by_node: dict[tuple[float, float], int] = {}
    for component_id, start in enumerate(sorted(nodes)):
        if start in component_by_node:
            continue
        stack = [start]
        component_by_node[start] = component_id
        while stack:
            node = stack.pop()
            for other in adjacency.get(node, set()):
                if other in component_by_node:
                    continue
                component_by_node[other] = component_id
                stack.append(other)
    return component_by_node


def directed_distances_to_targets(
    connector_graph: dict[str, Any],
    start: tuple[float, float],
    targets: set[tuple[float, float]],
) -> dict[tuple[float, float], float]:
    if not targets:
        return {}
    graph = connector_graph.get("graph") or {}
    found: dict[tuple[float, float], float] = {}
    queue: list[tuple[float, int, tuple[float, float]]] = [(0.0, 0, start)]
    distances: dict[tuple[float, float], float] = {start: 0.0}
    push_count = 1
    remaining = set(targets)
    while queue and remaining:
        distance, _index, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue
        if node in remaining:
            found[node] = distance
            remaining.remove(node)
            if not remaining:
                break
        for edge in graph.get(node, []):
            next_node = edge["to"]
            next_distance = distance + float(edge["distance"])
            if next_distance >= distances.get(next_node, math.inf):
                continue
            distances[next_node] = next_distance
            heapq.heappush(queue, (next_distance, push_count, next_node))
            push_count += 1
    return found


def connector_graph_min_matching(
    odd_nodes: list[int],
    node_info: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_feet: float,
) -> dict[str, Any]:
    if not connector_graph:
        return {
            "available": False,
            "reason": "connector_graph_missing",
            "quality_checks": {"connector_graph_loaded": False},
        }
    if not odd_nodes:
        return {
            "available": True,
            "connector_parity_add_on_miles": 0.0,
            "pairs": [],
            "snapped_odd_node_count": 0,
            "unsnapped_odd_nodes": [],
            "component_summaries": [],
            "quality_checks": {
                "connector_graph_loaded": True,
                "all_odd_nodes_snapped": True,
                "all_connector_components_even": True,
                "full_connector_matching_found": True,
            },
        }

    snap_tolerance_miles = snap_tolerance_feet * MILES_PER_FOOT
    snapped: dict[int, dict[str, Any]] = {}
    unsnapped: list[dict[str, Any]] = []
    for node_id in odd_nodes:
        point = (node_info[node_id]["lon"], node_info[node_id]["lat"])
        nearest = nearest_connector_node_for_graph(point, connector_graph, snap_tolerance_miles)
        if not nearest:
            unsnapped.append({"node": node_id, "lon": point[0], "lat": point[1]})
            continue
        snapped[node_id] = {
            "connector_node": nearest[0],
            "snap_miles": float(nearest[1]),
            "lon": point[0],
            "lat": point[1],
        }

    component_by_connector_node = connector_components(connector_graph)
    odd_by_component: dict[int, list[int]] = defaultdict(list)
    for node_id, snap in snapped.items():
        component_id = component_by_connector_node.get(snap["connector_node"])
        if component_id is None:
            unsnapped.append(
                {
                    "node": node_id,
                    "lon": snap["lon"],
                    "lat": snap["lat"],
                    "reason": "snapped_connector_node_not_in_component_index",
                }
            )
            continue
        snap["component_id"] = component_id
        odd_by_component[component_id].append(node_id)

    all_pairs: list[dict[str, Any]] = []
    component_summaries = []
    total = 0.0
    all_components_even = True
    full_matching_found = True
    for component_id, component_odd_nodes in sorted(odd_by_component.items()):
        component_odd_nodes = sorted(component_odd_nodes)
        target_nodes = {snapped[node_id]["connector_node"] for node_id in component_odd_nodes}
        directed: dict[int, dict[tuple[float, float], float]] = {}
        for node_id in component_odd_nodes:
            directed[node_id] = directed_distances_to_targets(
                connector_graph,
                snapped[node_id]["connector_node"],
                target_nodes,
            )

        component_complete = nx.Graph()
        for index, left in enumerate(component_odd_nodes):
            left_snap = snapped[left]
            for right in component_odd_nodes[index + 1 :]:
                right_snap = snapped[right]
                left_to_right = directed[left].get(right_snap["connector_node"])
                right_to_left = directed[right].get(left_snap["connector_node"])
                directional_options = []
                if left_to_right is not None:
                    directional_options.append(("left_to_right", left_to_right))
                if right_to_left is not None:
                    directional_options.append(("right_to_left", right_to_left))
                if not directional_options:
                    continue
                direction, graph_distance = min(directional_options, key=lambda item: item[1])
                distance = graph_distance + left_snap["snap_miles"] + right_snap["snap_miles"]
                component_complete.add_edge(
                    left,
                    right,
                    weight=-distance,
                    distance=distance,
                    graph_distance=graph_distance,
                    direction=direction,
                )

        component_even = len(component_odd_nodes) % 2 == 0
        all_components_even = all_components_even and component_even
        matching = nx.algorithms.matching.max_weight_matching(
            component_complete,
            maxcardinality=True,
            weight="weight",
        )
        expected_pairs = len(component_odd_nodes) // 2 if component_even else None
        component_full = component_even and len(matching) == expected_pairs
        full_matching_found = full_matching_found and component_full
        component_total = 0.0
        component_pairs = []
        for left, right in sorted((min(a, b), max(a, b)) for a, b in matching):
            edge = component_complete[left][right]
            distance = float(edge["distance"])
            component_total += distance
            pair = {
                "left_node": left,
                "right_node": right,
                "connector_graph_miles": round(distance, 4),
                "graph_path_miles": round(float(edge["graph_distance"]), 4),
                "direction_used_for_lower_bound": edge["direction"],
                "component_id": component_id,
            }
            component_pairs.append(pair)
            all_pairs.append(pair)
        total += component_total
        component_summaries.append(
            {
                "component_id": component_id,
                "odd_node_count": len(component_odd_nodes),
                "pair_count": len(matching),
                "expected_pair_count": expected_pairs,
                "full_matching_found": component_full,
                "connector_parity_miles": round(component_total, 4),
                "missing_pair_edges": max(0, (expected_pairs or 0) - len(matching)),
            }
        )

    quality_checks = {
        "connector_graph_loaded": True,
        "all_odd_nodes_snapped": not unsnapped,
        "all_connector_components_even": all_components_even,
        "full_connector_matching_found": full_matching_found and not unsnapped,
        "connector_parity_nonnegative": total >= 0,
    }
    return {
        "available": bool(quality_checks["full_connector_matching_found"]),
        "connector_parity_add_on_miles": round(total, 4) if quality_checks["full_connector_matching_found"] else None,
        "pairs": all_pairs,
        "snapped_odd_node_count": len(snapped),
        "unsnapped_odd_nodes": unsnapped,
        "component_summaries": component_summaries,
        "quality_checks": quality_checks,
        "snap_tolerance_feet": snap_tolerance_feet,
    }


def component_summaries(graph: nx.MultiGraph, odd_nodes: list[int]) -> list[dict[str, Any]]:
    odd_set = set(odd_nodes)
    summaries = []
    for index, nodes in enumerate(sorted(nx.connected_components(graph), key=lambda item: (len(item), sorted(item)[0]), reverse=True), 1):
        subgraph = graph.subgraph(nodes)
        edge_miles = sum(float(data.get("weight") or 0.0) for *_nodes, data in subgraph.edges(data=True))
        summaries.append(
            {
                "component_index": index,
                "node_count": len(nodes),
                "required_edge_count": subgraph.number_of_edges(),
                "official_miles": round(edge_miles, 2),
                "odd_node_count": len(set(nodes) & odd_set),
            }
        )
    return summaries


def build_report(
    official_geojson: dict[str, Any],
    *,
    snap_tolerance_feet: float,
    connector_graph: dict[str, Any] | None = None,
    connector_snap_tolerance_feet: float = 250.0,
    current_plan_audit: dict[str, Any] | None = None,
    official_segments_path: Path | None = None,
    connector_geojson_path: Path | None = None,
    current_plan_audit_path: Path | None = None,
) -> dict[str, Any]:
    edges = required_edges_from_geojson(official_geojson)
    snap_tolerance_miles = snap_tolerance_feet * MILES_PER_FOOT
    snapped_edges, node_info = snap_endpoint_nodes(edges, snap_tolerance_miles)
    graph = required_graph(snapped_edges)
    odd_nodes = odd_node_ids(graph)
    parity_miles, matching_pairs = straight_line_min_matching(odd_nodes, node_info)
    connector_matching = connector_graph_min_matching(
        odd_nodes,
        node_info,
        connector_graph,
        connector_snap_tolerance_feet,
    )
    official_miles = sum(float(edge["official_miles"]) for edge in snapped_edges)
    lower_bound = official_miles + parity_miles
    connector_lower_bound = None
    connector_parity = connector_matching.get("connector_parity_add_on_miles")
    if connector_parity is not None:
        connector_lower_bound = official_miles + float(connector_parity)
    direction_counts = Counter(edge.get("direction") or "both" for edge in snapped_edges)
    current = {}
    if current_plan_audit:
        totals = ((current_plan_audit.get("summary") or {}).get("runnable_field_packet_totals") or {})
        current_on_foot = totals.get("on_foot_miles")
        if current_on_foot is not None:
            current = {
                "on_foot_miles": float(current_on_foot),
                "gap_to_lower_bound_miles": round(float(current_on_foot) - lower_bound, 2),
                "ratio_to_lower_bound": round(float(current_on_foot) / lower_bound, 3) if lower_bound else None,
                "gap_to_connector_lower_bound_miles": round(float(current_on_foot) - connector_lower_bound, 2)
                if connector_lower_bound is not None
                else None,
                "ratio_to_connector_lower_bound": round(float(current_on_foot) / connector_lower_bound, 3)
                if connector_lower_bound
                else None,
            }

    return {
        "objective": "Rural-Postman-style lower bounds for official required segments with optional real connector graph costs",
        "method": {
            "required_edges": "2026 official on-foot challenge segments",
            "base_miles": "sum of official LengthFt values",
            "parity_add_on": "minimum perfect matching over odd-degree required-graph endpoints using straight-line distances",
            "connector_graph_parity_add_on": (
                "minimum perfect matching over odd required endpoints snapped to the connector graph, "
                "using shortest legal connector/road/official-repeat paths inside each connector component"
            ),
            "why_this_is_a_lower_bound": [
                "Every required official segment must be traversed at least once.",
                "Any closed single-car outing collection must add traversal that pairs odd required-graph endpoints.",
                "Straight-line distance between paired odd endpoints is no longer than any real trail, road, or connector path.",
                "The calculation ignores parking access, route splitting, ascent direction, field navigation, and day-of constraints, so it is intentionally optimistic.",
            ],
            "connector_graph_scope": [
                "The connector-graph lower bound is stronger than the straight-line lower bound because odd-endpoint pairing uses the loaded legal connector graph.",
                "It is still a lower bound, not a route: it ignores parking access, drive time, day splits, hard stops, and route-finding complexity.",
                "It is conditional on the connector overlay being complete and correctly filtering private, no-foot, and non-real graph artifacts.",
                "Directional connector edges are handled optimistically by using the cheaper reachable direction for each parity pair.",
            ],
            "not_claimed": [
                "Not an executable route.",
                "Not a proof that the current route set is globally optimal.",
                "Not a day-of closure, signage, or parking validation.",
            ],
            "snap_tolerance_feet": snap_tolerance_feet,
            "connector_snap_tolerance_feet": connector_snap_tolerance_feet,
        },
        "inputs": {
            "official_segments_geojson": display_path(official_segments_path),
            "connector_geojson": display_path(connector_geojson_path),
            "current_plan_audit_json": display_path(current_plan_audit_path),
            "official_feature_count": len(official_geojson.get("features") or []),
            "last_updated_utc": official_geojson.get("lastUpdatedUTC"),
            "connector_graph": {
                "loaded": bool(connector_graph),
                "path": display_path(Path(connector_graph["path"])) if connector_graph and connector_graph.get("path") else None,
                "node_count": len(connector_graph.get("nodes") or []) if connector_graph else 0,
                "adjacency_node_count": len(connector_graph.get("graph") or {}) if connector_graph else 0,
                "feature_count": connector_graph.get("feature_count") if connector_graph else None,
                "official_repeat_segment_count": connector_graph.get("official_segment_count") if connector_graph else None,
                "connector_class_counts": connector_graph.get("connector_class_counts") if connector_graph else {},
            },
        },
        "summary": {
            "required_segment_count": len({edge["seg_id"] for edge in snapped_edges}),
            "required_edge_part_count": len(snapped_edges),
            "official_miles": round(official_miles, 2),
            "required_graph_node_count": graph.number_of_nodes(),
            "required_graph_component_count": nx.number_connected_components(graph),
            "odd_node_count": len(odd_nodes),
            "straight_line_parity_add_on_miles": round(parity_miles, 2),
            "rural_postman_lower_bound_miles": round(lower_bound, 2),
            "lower_bound_ratio_to_official": round(lower_bound / official_miles, 3) if official_miles else None,
            "connector_graph_parity_add_on_miles": round(float(connector_parity), 2)
            if connector_parity is not None
            else None,
            "connector_graph_lower_bound_miles": round(connector_lower_bound, 2)
            if connector_lower_bound is not None
            else None,
            "connector_graph_lower_bound_ratio_to_official": round(connector_lower_bound / official_miles, 3)
            if connector_lower_bound
            else None,
            "direction_counts": dict(sorted(direction_counts.items())),
            "current_plan_comparison": current,
        },
        "component_summaries": component_summaries(graph, odd_nodes),
        "matching": {
            "pair_count": len(matching_pairs),
            "total_straight_line_miles": round(parity_miles, 4),
            "pairs": matching_pairs,
        },
        "connector_graph_matching": connector_matching,
        "quality_checks": {
            "odd_node_count_even": len(odd_nodes) % 2 == 0,
            "matching_pair_count_expected": len(matching_pairs) == len(odd_nodes) // 2,
            "official_miles_positive": official_miles > 0,
            "lower_bound_at_least_official_miles": lower_bound >= official_miles,
            "connector_graph_lower_bound_available": bool(connector_matching.get("available")),
            "connector_graph_lower_bound_at_least_straight_line_lower_bound": (
                connector_lower_bound is not None and connector_lower_bound >= lower_bound
            ),
        },
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    current = summary.get("current_plan_comparison") or {}
    lines = [
        "# Rural-Postman Lower Bound",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Method",
        "",
        f"- Required edges: {report['method']['required_edges']}",
        f"- Base miles: {report['method']['base_miles']}",
        f"- Parity add-on: {report['method']['parity_add_on']}",
        f"- Endpoint snap tolerance: {report['method']['snap_tolerance_feet']} ft",
        "",
        "This is a mathematical lower bound, not a runnable route. It is deliberately optimistic.",
        "",
        "## Result",
        "",
        f"- Official required miles: {summary['official_miles']}",
        f"- Required graph nodes: {summary['required_graph_node_count']}",
        f"- Required graph components: {summary['required_graph_component_count']}",
        f"- Odd required-graph nodes: {summary['odd_node_count']}",
        f"- Straight-line parity add-on: {summary['straight_line_parity_add_on_miles']} mi",
        f"- Rural-postman-style lower bound: {summary['rural_postman_lower_bound_miles']} mi",
        f"- Lower-bound ratio to official miles: {summary['lower_bound_ratio_to_official']}x",
    ]
    if summary.get("connector_graph_lower_bound_miles") is not None:
        connector_inputs = (report.get("inputs") or {}).get("connector_graph") or {}
        lines.extend(
            [
                "",
                "## Connector-Graph Lower Bound",
                "",
                f"- Connector graph: {connector_inputs.get('node_count')} nodes, "
                f"{connector_inputs.get('feature_count')} connector features, "
                f"{connector_inputs.get('official_repeat_segment_count')} official-repeat segments",
                f"- Connector classes: {connector_inputs.get('connector_class_counts')}",
                f"- Connector parity add-on: {summary['connector_graph_parity_add_on_miles']} mi",
                f"- Connector-graph lower bound: {summary['connector_graph_lower_bound_miles']} mi",
                f"- Connector-graph lower-bound ratio to official miles: "
                f"{summary['connector_graph_lower_bound_ratio_to_official']}x",
                "",
                "This is still a lower bound, not a field route. It uses the loaded connector graph for parity costs, "
                "but it does not include parking access, drive time, day splits, hard stops, or route-finding complexity.",
            ]
        )
    else:
        connector_matching = report.get("connector_graph_matching") or {}
        lines.extend(
            [
                "",
                "## Connector-Graph Lower Bound",
                "",
                "No complete connector-graph lower bound was produced.",
                f"Reason: {connector_matching.get('reason') or connector_matching.get('quality_checks')}",
            ]
        )
    if current:
        lines.extend(
            [
                "",
                "## Current Plan Comparison",
                "",
                f"- Current field-packet on-foot miles: {current['on_foot_miles']}",
                f"- Current gap above lower bound: {current['gap_to_lower_bound_miles']} mi",
                f"- Current / lower-bound ratio: {current['ratio_to_lower_bound']}x",
            ]
        )
        if current.get("gap_to_connector_lower_bound_miles") is not None:
            lines.extend(
                [
                    f"- Current gap above connector-graph lower bound: "
                    f"{current['gap_to_connector_lower_bound_miles']} mi",
                    f"- Current / connector-graph lower-bound ratio: "
                    f"{current['ratio_to_connector_lower_bound']}x",
                ]
            )
    lines.extend(
        [
            "",
            "## Why This Is A Lower Bound",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["method"]["why_this_is_a_lower_bound"])
    if summary.get("connector_graph_lower_bound_miles") is not None:
        lines.extend(
            [
                "",
                "## Connector-Graph Scope",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in report["method"]["connector_graph_scope"])
    lines.extend(
        [
            "",
            "## Quality Checks",
            "",
            "| Check | Passed |",
            "|---|---:|",
        ]
    )
    for key, value in report["quality_checks"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Largest Required Components",
            "",
            "| Component | Nodes | Edges | Official mi | Odd nodes |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for component in report["component_summaries"][:12]:
        lines.append(
            f"| {component['component_index']} | {component['node_count']} | "
            f"{component['required_edge_count']} | {component['official_miles']} | {component['odd_node_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-segments-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--current-plan-audit-json", type=Path, default=DEFAULT_CURRENT_PLAN_JSON)
    parser.add_argument("--snap-tolerance-feet", type=float, default=50.0)
    parser.add_argument("--connector-snap-tolerance-feet", type=float, default=250.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_geojson = read_json(args.official_segments_geojson)
    required_edges = required_edges_from_geojson(official_geojson)
    connector_graph = None
    if args.connector_geojson.exists():
        connector_graph = load_connector_graph(
            args.connector_geojson,
            official_segments=connector_official_segments_from_required_edges(required_edges),
        )
    current_plan_audit = read_json(args.current_plan_audit_json) if args.current_plan_audit_json.exists() else None
    report = build_report(
        official_geojson,
        snap_tolerance_feet=args.snap_tolerance_feet,
        connector_graph=connector_graph,
        connector_snap_tolerance_feet=args.connector_snap_tolerance_feet,
        current_plan_audit=current_plan_audit,
        official_segments_path=args.official_segments_geojson,
        connector_geojson_path=args.connector_geojson if args.connector_geojson.exists() else None,
        current_plan_audit_path=args.current_plan_audit_json if args.current_plan_audit_json.exists() else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
