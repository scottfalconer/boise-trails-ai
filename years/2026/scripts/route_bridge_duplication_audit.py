#!/usr/bin/env python3
"""Audit topology bridge duplication across field-packet route cards."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    load_connector_graph,
    load_official_segments,
    shortest_connector_path,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "route-bridge-duplication-audit-2026-05-26.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "route-bridge-duplication-audit-2026-05-26.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "route-bridge-duplication-audit-2026-05-26-manifest.json"
DEFAULT_ENDPOINT_TOLERANCE_MILES = 15 / 1609.344
DEFAULT_NEAR_BRIDGE_MIN_DETOUR_MILES = 0.25
DEFAULT_CONNECTOR_SNAP_TOLERANCE_MILES = 0.03


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


def sort_id(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def normalized_ids(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("route_key") or route.get("label") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_ref(route: dict[str, Any] | None) -> dict[str, Any]:
    if not route:
        return {"route_key": None, "outing_id": None, "label": None, "candidate_ids": []}
    return {
        "route_key": route_key(route),
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "candidate_ids": [str(candidate_id) for candidate_id in route.get("candidate_ids") or []],
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


def segment_id(segment: dict[str, Any]) -> str:
    return str(segment.get("seg_id") or segment.get("segId") or "")


def segment_coordinates(segment: dict[str, Any]) -> list[tuple[float, float]]:
    coords = segment.get("coordinates") or []
    return [(float(coord[0]), float(coord[1])) for coord in coords if len(coord) >= 2]


def segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {segment_id(segment): segment for segment in official_segments if segment_id(segment)}


def segment_brief(segment_by_id: dict[str, dict[str, Any]], seg_id: str) -> dict[str, Any]:
    segment = segment_by_id.get(str(seg_id), {})
    return {
        "seg_id": str(seg_id),
        "seg_name": segment.get("seg_name"),
        "trail_name": segment.get("trail_name"),
        "direction": segment.get("direction"),
        "official_miles": round(float_value(segment.get("official_miles")), 3),
    }


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_endpoint_graph(
    official_segments: list[dict[str, Any]],
    *,
    endpoint_tolerance_miles: float,
) -> dict[str, Any]:
    endpoint_rows = []
    for segment in official_segments:
        seg_id = segment_id(segment)
        coords = segment_coordinates(segment)
        if not seg_id or len(coords) < 2:
            continue
        endpoint_rows.append({"seg_id": seg_id, "endpoint": "start", "point": coords[0]})
        endpoint_rows.append({"seg_id": seg_id, "endpoint": "end", "point": coords[-1]})
    union_find = UnionFind(len(endpoint_rows))
    for left_index, left in enumerate(endpoint_rows):
        for right_index in range(left_index + 1, len(endpoint_rows)):
            right = endpoint_rows[right_index]
            if haversine_miles(left["point"], right["point"]) <= endpoint_tolerance_miles:
                union_find.union(left_index, right_index)

    root_to_node: dict[int, str] = {}
    endpoint_node: dict[tuple[str, str], str] = {}
    node_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for index, endpoint in enumerate(endpoint_rows):
        root = union_find.find(index)
        node_id = root_to_node.setdefault(root, f"N{len(root_to_node) + 1}")
        endpoint_node[(endpoint["seg_id"], endpoint["endpoint"])] = node_id
        node_points[node_id].append(endpoint["point"])

    segment_edges = {}
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in official_segments:
        seg_id = segment_id(segment)
        if not seg_id:
            continue
        start_node = endpoint_node.get((seg_id, "start"))
        end_node = endpoint_node.get((seg_id, "end"))
        if not start_node or not end_node:
            continue
        miles = float_value(segment.get("official_miles"))
        segment_edges[seg_id] = {
            "seg_id": seg_id,
            "start_node": start_node,
            "end_node": end_node,
            "official_miles": miles,
        }
        edge = {"seg_id": seg_id, "to": end_node, "official_miles": miles}
        reverse = {"seg_id": seg_id, "to": start_node, "official_miles": miles}
        adjacency[start_node].append(edge)
        if str(segment.get("direction") or "both") == "both":
            adjacency[end_node].append(reverse)
        else:
            adjacency[end_node].append(reverse)

    node_coordinates = {}
    for node_id, points in node_points.items():
        node_coordinates[node_id] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    return {
        "adjacency": dict(adjacency),
        "segment_edges": segment_edges,
        "node_coordinates": node_coordinates,
        "node_count": len(node_coordinates),
        "edge_count": len(segment_edges),
    }


def shortest_official_path(
    graph: dict[str, Any],
    start_node: str,
    end_node: str,
    *,
    avoid_segment_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    avoided = set(avoid_segment_ids or set())
    queue: list[tuple[float, int, str]] = [(0.0, 0, start_node)]
    distances = {start_node: 0.0}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    push_count = 1
    while queue:
        distance, _count, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue
        if node == end_node:
            path_segments = []
            cursor = node
            while cursor in previous:
                prior, edge = previous[cursor]
                path_segments.append(edge["seg_id"])
                cursor = prior
            path_segments.reverse()
            return {
                "distance_miles": distance,
                "official_repeat_segment_ids": path_segments,
                "path_source": "official_segment_graph",
            }
        for edge in graph.get("adjacency", {}).get(node, []):
            if str(edge.get("seg_id")) in avoided:
                continue
            next_node = str(edge["to"])
            next_distance = distance + float_value(edge.get("official_miles"))
            if next_distance >= distances.get(next_node, math.inf):
                continue
            distances[next_node] = next_distance
            previous[next_node] = (node, edge)
            heapq.heappush(queue, (next_distance, push_count, next_node))
            push_count += 1
    return None


def connected_components(graph: dict[str, Any], *, avoid_segment_ids: set[str] | None = None) -> dict[str, int]:
    avoided = set(avoid_segment_ids or set())
    components = {}
    component_index = 0
    for node in graph.get("node_coordinates", {}):
        if node in components:
            continue
        component_index += 1
        stack = [node]
        components[node] = component_index
        while stack:
            current = stack.pop()
            for edge in graph.get("adjacency", {}).get(current, []):
                if str(edge.get("seg_id")) in avoided:
                    continue
                next_node = str(edge["to"])
                if next_node not in components:
                    components[next_node] = component_index
                    stack.append(next_node)
    return components


def local_xy_miles(point: tuple[float, float], origin_lat: float) -> tuple[float, float]:
    lon, lat = point
    return (lon * 69.172 * math.cos(math.radians(origin_lat)), lat * 69.0)


def point_projection_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    px, py = local_xy_miles(point, point[1])
    ax, ay = local_xy_miles(start, point[1])
    bx, by = local_xy_miles(end, point[1])
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest = (ax + t * dx, ay + t * dy)
    return math.hypot(px - nearest[0], py - nearest[1]), t


def mid_segment_junction_proof(
    official_segments: list[dict[str, Any]],
    *,
    endpoint_tolerance_miles: float,
) -> dict[str, Any]:
    junctions = []
    endpoint_rows = []
    for segment in official_segments:
        seg_id = segment_id(segment)
        coords = segment_coordinates(segment)
        if seg_id and len(coords) >= 2:
            endpoint_rows.append((seg_id, "start", coords[0]))
            endpoint_rows.append((seg_id, "end", coords[-1]))
    for endpoint_seg_id, endpoint_name, point in endpoint_rows:
        for interior_segment in official_segments:
            interior_seg_id = segment_id(interior_segment)
            if not interior_seg_id or interior_seg_id == endpoint_seg_id:
                continue
            coords = segment_coordinates(interior_segment)
            if len(coords) < 2:
                continue
            if (
                haversine_miles(point, coords[0]) <= endpoint_tolerance_miles
                or haversine_miles(point, coords[-1]) <= endpoint_tolerance_miles
            ):
                continue
            for part_index, (start, end) in enumerate(zip(coords, coords[1:])):
                distance, fraction = point_projection_to_segment(point, start, end)
                if distance <= endpoint_tolerance_miles and 0.0001 < fraction < 0.9999:
                    junctions.append(
                        {
                            "endpoint_segment_id": endpoint_seg_id,
                            "endpoint": endpoint_name,
                            "interior_segment_id": interior_seg_id,
                            "interior_part_index": part_index,
                            "distance_miles": round(distance, 5),
                            "interior_fraction": round(fraction, 4),
                        }
                    )
                    break
    unique = {
        (
            row["endpoint_segment_id"],
            row["endpoint"],
            row["interior_segment_id"],
        ): row
        for row in junctions
    }
    rows = sorted(unique.values(), key=lambda row: (sort_id(row["endpoint_segment_id"]), row["endpoint"]))
    return {
        "status": "proved_no_mid_segment_junctions" if not rows else "incomplete_until_virtual_nodes_inserted",
        "endpoint_tolerance_miles": endpoint_tolerance_miles,
        "mid_segment_junction_count": len(rows),
        "junctions": rows,
    }


def route_claims(routes: list[dict[str, Any]]) -> dict[str, set[str]]:
    claims: dict[str, set[str]] = {}
    for route in routes:
        key = route_key(route)
        for seg_id in normalized_ids(route.get("segment_ids") or []):
            claims.setdefault(seg_id, set()).add(key)
    return claims


def declared_repeat_ids(route: dict[str, Any]) -> set[str]:
    ids = set()
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    ids.update(normalized_ids(reconciliation.get("declared_owned_elsewhere_segment_ids") or []))
    for row in reconciliation.get("segments_owned_elsewhere") or []:
        if row.get("seg_id") is not None:
            ids.add(str(row["seg_id"]))
    for cue in route.get("wayfinding_cues") or []:
        ids.update(normalized_ids(cue.get("official_repeat_segment_ids") or []))
    return ids


def owner_refs_for_repeated_segment(
    route: dict[str, Any],
    repeated_segment_id: str,
    routes_by_key: dict[str, dict[str, Any]],
    claims: dict[str, set[str]],
) -> list[dict[str, Any]]:
    refs = []
    seen = set()
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    for row in reconciliation.get("segments_owned_elsewhere") or []:
        if str(row.get("seg_id") or "") != str(repeated_segment_id):
            continue
        for owner in row.get("owned_by_routes") or []:
            owner_route = None
            for key in (owner.get("route_key"), owner.get("outing_id"), owner.get("label")):
                if key and str(key) in routes_by_key:
                    owner_route = routes_by_key[str(key)]
                    break
            ref = route_ref(owner_route)
            if not ref["route_key"]:
                ref = {
                    "route_key": str(owner.get("route_key") or owner.get("outing_id") or owner.get("label") or ""),
                    "outing_id": owner.get("outing_id"),
                    "label": owner.get("label"),
                    "candidate_ids": owner.get("candidate_ids") or [],
                }
            if ref["route_key"] and ref["route_key"] not in seen:
                refs.append(ref)
                seen.add(ref["route_key"])
    receiver_key = route_key(route)
    for owner_key in sorted(claims.get(str(repeated_segment_id), set()), key=sort_id):
        if owner_key == receiver_key or owner_key in seen:
            continue
        owner_route = routes_by_key.get(owner_key)
        refs.append(route_ref(owner_route))
        seen.add(owner_key)
    return refs


def route_matches_waiver(route: dict[str, Any], route_key_or_id: str | None) -> bool:
    if not route_key_or_id:
        return True
    candidate = str(route_key_or_id)
    return candidate in {
        route_key(route),
        str(route.get("outing_id") or ""),
        str(route.get("label") or ""),
        route_label(route),
        *(str(item) for item in route.get("candidate_ids") or []),
    }


def matching_waiver(
    *,
    bridge_segment_id: str,
    owner_routes: list[dict[str, Any]],
    receiver_route: dict[str, Any],
    waivers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    accepted_statuses = {"accepted_unavoidable_bridge", "accepted_unavoidable"}
    owner_keys = {str(owner.get("route_key") or owner.get("outing_id") or "") for owner in owner_routes}
    for waiver in waivers:
        if str(waiver.get("status") or "") not in accepted_statuses:
            continue
        if str(waiver.get("bridge_segment_id") or waiver.get("seg_id") or "") != str(bridge_segment_id):
            continue
        owner_key = str(waiver.get("owner_route_key") or waiver.get("owner_outing_id") or "")
        if owner_key and owner_key not in owner_keys:
            continue
        if not route_matches_waiver(receiver_route, waiver.get("receiver_route_key") or waiver.get("receiver_outing_id")):
            continue
        return waiver
    return None


def certified_repair_exists(
    *,
    bridge_segment_id: str,
    receiver_route: dict[str, Any],
    certified_repair_candidates: list[dict[str, Any]],
) -> bool:
    accepted_statuses = {"certified_replacement", "accepted_replacement", "field_ready_replacement"}
    for candidate in certified_repair_candidates:
        if str(candidate.get("status") or "") not in accepted_statuses:
            continue
        if str(candidate.get("bridge_segment_id") or candidate.get("seg_id") or "") != str(bridge_segment_id):
            continue
        if route_matches_waiver(receiver_route, candidate.get("replaces_receiver_route_key") or candidate.get("receiver_route_key")):
            return True
    return False


def chained_credit_ids_for_bridge(
    receiver: dict[str, Any],
    graph: dict[str, Any],
    bridge_segment_id: str,
) -> list[str]:
    bridge = graph.get("segment_edges", {}).get(str(bridge_segment_id))
    if not bridge:
        return []
    components = connected_components(graph, avoid_segment_ids={str(bridge_segment_id)})
    endpoint_components = {
        components.get(bridge["start_node"]),
        components.get(bridge["end_node"]),
    }
    chained = []
    for claimed_id in normalized_ids(receiver.get("segment_ids") or []):
        edge = graph.get("segment_edges", {}).get(claimed_id)
        if not edge:
            continue
        if components.get(edge["start_node"]) in endpoint_components or components.get(edge["end_node"]) in endpoint_components:
            chained.append(claimed_id)
    return normalized_ids(chained)


def connector_alternate_path(
    segment_by_id: dict[str, dict[str, Any]],
    bridge_segment_id: str,
    connector_graph: dict[str, Any] | None,
    *,
    connector_snap_tolerance_miles: float,
) -> dict[str, Any] | None:
    if not connector_graph:
        return None
    coords = segment_coordinates(segment_by_id.get(str(bridge_segment_id), {}))
    if len(coords) < 2:
        return None
    return shortest_connector_path(
        coords[0],
        coords[-1],
        connector_graph,
        connector_snap_tolerance_miles,
        avoid_official_segment_ids={int(bridge_segment_id)},
    )


def alternate_path_for_bridge(
    segment_by_id: dict[str, dict[str, Any]],
    graph: dict[str, Any],
    bridge_segment_id: str,
    connector_graph: dict[str, Any] | None,
    *,
    connector_snap_tolerance_miles: float,
) -> dict[str, Any] | None:
    connector_path = connector_alternate_path(
        segment_by_id,
        bridge_segment_id,
        connector_graph,
        connector_snap_tolerance_miles=connector_snap_tolerance_miles,
    )
    if connector_path:
        return {**connector_path, "path_source": "connector_graph"}
    edge = graph.get("segment_edges", {}).get(str(bridge_segment_id))
    if not edge:
        return None
    return shortest_official_path(
        graph,
        edge["start_node"],
        edge["end_node"],
        avoid_segment_ids={str(bridge_segment_id)},
    )


def repair_candidates_for_finding(
    *,
    classification: str,
    bridge_segment_id: str,
    owner_routes: list[dict[str, Any]],
    receiver_route: dict[str, Any],
    chained_credit_segment_ids: list[str],
    alternate_path: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    owner_keys = [str(owner.get("route_key") or owner.get("outing_id") or "") for owner in owner_routes]
    candidates = [
        {
            "candidate_type": "owner_route_extension",
            "candidate_status": "needs_route_generation",
            "owner_route_keys": owner_keys,
            "receiver_route_key": route_key(receiver_route),
            "add_credit_segment_ids_to_owner": chained_credit_segment_ids,
            "drop_credit_segment_ids_from_receiver": chained_credit_segment_ids,
            "required_proof": "generated route card must preserve full official coverage and pass field-route validation",
        },
        {
            "candidate_type": "receiver_reroute_avoiding_bridge",
            "candidate_status": "quantified_detour_available" if alternate_path else "no_alternate_path_found",
            "receiver_route_key": route_key(receiver_route),
            "avoid_official_segment_ids": [str(bridge_segment_id)],
            "alternate_path_source": (alternate_path or {}).get("path_source"),
            "alternate_distance_miles": round(float_value((alternate_path or {}).get("distance_miles")), 3)
            if alternate_path
            else None,
            "alternate_official_repeat_segment_ids": normalized_ids((alternate_path or {}).get("official_repeat_segment_ids") or []),
        },
        {
            "candidate_type": "cluster_recomposition",
            "candidate_status": "needs_cluster_candidate_generation",
            "route_keys": normalized_ids([*owner_keys, route_key(receiver_route)]),
            "target_segment_ids": normalized_ids(
                [
                    *chained_credit_segment_ids,
                    bridge_segment_id,
                    *normalized_ids(receiver_route.get("segment_ids") or []),
                ]
            ),
            "required_proof": "candidate cluster must reduce cross-day bridge duplicate miles without losing official coverage",
        },
    ]
    if classification == "strict_bridge":
        candidates[1]["required_decision"] = "strict bridge has no alternate path in the current graph; prefer ownership/cluster repair or waiver"
    return candidates


def summarize_alternate_path(alternate_path: dict[str, Any] | None) -> dict[str, Any] | None:
    if not alternate_path:
        return None
    return {
        "path_source": alternate_path.get("path_source"),
        "distance_miles": round(float_value(alternate_path.get("distance_miles")), 3),
        "connector_miles": round(float_value(alternate_path.get("connector_miles")), 3),
        "official_repeat_miles": round(float_value(alternate_path.get("official_repeat_miles")), 3),
        "connector_names": alternate_path.get("connector_names") or [],
        "connector_classes": alternate_path.get("connector_classes") or [],
        "official_repeat_segment_ids": normalized_ids(alternate_path.get("official_repeat_segment_ids") or []),
        "connector_edge_count": len(alternate_path.get("connector_edges") or []),
        "path_coordinate_count": len(alternate_path.get("path_coordinates") or []),
        "snap_start_miles": round(float_value(alternate_path.get("snap_start_miles")), 3),
        "snap_end_miles": round(float_value(alternate_path.get("snap_end_miles")), 3),
    }


def build_bridge_duplication_audit(
    field_tool_data: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    endpoint_tolerance_miles: float = DEFAULT_ENDPOINT_TOLERANCE_MILES,
    near_bridge_min_detour_miles: float = DEFAULT_NEAR_BRIDGE_MIN_DETOUR_MILES,
    connector_snap_tolerance_miles: float = DEFAULT_CONNECTOR_SNAP_TOLERANCE_MILES,
    bridge_duplication_waivers: list[dict[str, Any]] | None = None,
    certified_repair_candidates: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    routes_by_key = route_index(routes)
    claims = route_claims(routes)
    segment_by_id = segment_index(official_segments)
    graph = build_endpoint_graph(official_segments, endpoint_tolerance_miles=endpoint_tolerance_miles)
    junction_proof = mid_segment_junction_proof(official_segments, endpoint_tolerance_miles=endpoint_tolerance_miles)
    waivers = list(field_tool_data.get("bridge_duplication_waivers") or [])
    waivers.extend(bridge_duplication_waivers or [])
    certified_repairs = list(field_tool_data.get("certified_bridge_repair_candidates") or [])
    certified_repairs.extend(certified_repair_candidates or [])
    findings = []
    seen = set()
    for receiver in routes:
        receiver_key = route_key(receiver)
        for bridge_id in declared_repeat_ids(receiver):
            if bridge_id not in segment_by_id or bridge_id not in claims:
                continue
            owner_routes = owner_refs_for_repeated_segment(receiver, bridge_id, routes_by_key, claims)
            if not owner_routes:
                continue
            key = (receiver_key, bridge_id, tuple(sorted(str(owner.get("route_key") or "") for owner in owner_routes)))
            if key in seen:
                continue
            seen.add(key)
            bridge_miles = float_value(segment_by_id[bridge_id].get("official_miles"))
            alternate = alternate_path_for_bridge(
                segment_by_id,
                graph,
                bridge_id,
                connector_graph,
                connector_snap_tolerance_miles=connector_snap_tolerance_miles,
            )
            detour_distance = float_value((alternate or {}).get("distance_miles")) if alternate else None
            detour_added = detour_distance - bridge_miles if detour_distance is not None else None
            chained_ids = chained_credit_ids_for_bridge(receiver, graph, bridge_id)
            if not chained_ids:
                classification = "tail_opportunity"
                severity = "informational"
            elif alternate is None:
                classification = "strict_bridge"
                severity = "optimization_debt"
            elif detour_added is not None and detour_added >= near_bridge_min_detour_miles:
                classification = "near_bridge"
                severity = "optimization_warning"
            else:
                classification = "tail_opportunity"
                severity = "informational"
            waiver = matching_waiver(
                bridge_segment_id=bridge_id,
                owner_routes=owner_routes,
                receiver_route=receiver,
                waivers=waivers,
            )
            has_certified_repair = certified_repair_exists(
                bridge_segment_id=bridge_id,
                receiver_route=receiver,
                certified_repair_candidates=certified_repairs,
            )
            waived = waiver is not None
            graduation_status = (
                "accepted_unavoidable_bridge"
                if waived
                else "informational"
                if classification == "tail_opportunity"
                else "blocking_failure"
                if classification == "strict_bridge" and has_certified_repair
                else "optimization_debt"
            )
            findings.append(
                {
                    "code": "cross_day_bridge_duplication",
                    "classification": classification,
                    "severity": "waived_optimization_debt" if waived else severity,
                    "graduation_status": graduation_status,
                    "waived": waived,
                    "waiver": waiver,
                    "owner_routes": owner_routes,
                    "receiver_route": route_ref(receiver),
                    "bridge_segment": segment_brief(segment_by_id, bridge_id),
                    "duplicate_bridge_miles": round(bridge_miles, 3),
                    "chained_credit_segment_ids": chained_ids,
                    "chained_credit_segments": [segment_brief(segment_by_id, seg_id) for seg_id in chained_ids],
                    "alternate_path": summarize_alternate_path(alternate),
                    "detour_distance_miles": round(detour_distance, 3) if detour_distance is not None else None,
                    "detour_added_miles": round(detour_added, 3) if detour_added is not None else None,
                    "repair_candidates": repair_candidates_for_finding(
                        classification=classification,
                        bridge_segment_id=bridge_id,
                        owner_routes=owner_routes,
                        receiver_route=receiver,
                        chained_credit_segment_ids=chained_ids,
                        alternate_path=alternate,
                    ),
                }
            )
    findings.sort(
        key=lambda row: (
            {"strict_bridge": 0, "near_bridge": 1, "tail_opportunity": 2}.get(row["classification"], 9),
            -len(row.get("chained_credit_segment_ids") or []),
            -float_value(row.get("detour_added_miles") if row.get("detour_added_miles") is not None else 9999),
            row["receiver_route"].get("label") or "",
        )
    )
    strict = [row for row in findings if row["classification"] == "strict_bridge"]
    near = [row for row in findings if row["classification"] == "near_bridge"]
    tail = [row for row in findings if row["classification"] == "tail_opportunity"]
    unwaived_strict = [row for row in strict if not row.get("waived")]
    graduated_blocking = [row for row in findings if row.get("graduation_status") == "blocking_failure"]
    return {
        "schema": "boise_trails_route_bridge_duplication_audit_v1",
        "generated_at": generated_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "actionable_bridge_debt" if unwaived_strict or near else "clear",
        "source_files": source_files or {},
        "parameters": {
            "endpoint_tolerance_miles": endpoint_tolerance_miles,
            "near_bridge_min_detour_miles": near_bridge_min_detour_miles,
            "connector_snap_tolerance_miles": connector_snap_tolerance_miles,
            "near_bridge_has_upper_detour_cap": False,
        },
        "graph": {
            "node_count": graph["node_count"],
            "official_edge_count": graph["edge_count"],
        },
        "mid_segment_junction_proof": junction_proof,
        "summary": {
            "route_count": len(routes),
            "finding_count": len(findings),
            "strict_bridge_count": len(strict),
            "strict_bridge_count_unwaived": len(unwaived_strict),
            "near_bridge_count": len(near),
            "tail_opportunity_count": len(tail),
            "graduated_blocking_strict_bridge_count": len(graduated_blocking),
            "duplicate_bridge_miles": round(sum(float_value(row.get("duplicate_bridge_miles")) for row in findings), 3),
            "unwaived_strict_duplicate_bridge_miles": round(
                sum(float_value(row.get("duplicate_bridge_miles")) for row in unwaived_strict),
                3,
            ),
            "repair_candidate_count": sum(len(row.get("repair_candidates") or []) for row in findings),
            "mid_segment_junction_count": junction_proof["mid_segment_junction_count"],
        },
        "findings": findings,
        "strict_bridges": strict,
        "near_bridges": near,
        "tail_opportunities": tail,
        "graduated_blocking_findings": graduated_blocking,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Route Bridge Duplication Audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Findings: {summary['finding_count']}",
        f"- Strict bridges: {summary['strict_bridge_count']} ({summary['strict_bridge_count_unwaived']} unwaived)",
        f"- Near bridges: {summary['near_bridge_count']}",
        f"- Tail opportunities: {summary['tail_opportunity_count']}",
        f"- Duplicate bridge miles: {summary['duplicate_bridge_miles']:.3f}",
        f"- Mid-segment junctions: {summary['mid_segment_junction_count']}",
        "",
        "## Bridge Findings",
        "",
    ]
    findings = audit.get("findings") or []
    if findings:
        lines.extend(
            [
                "| Class | Receiver | Bridge | Owners | Chained credits | Detour added mi | Status |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in findings:
            owners = ", ".join(str(owner.get("label") or owner.get("route_key")) for owner in row.get("owner_routes") or [])
            bridge = row.get("bridge_segment") or {}
            detour = row.get("detour_added_miles")
            detour_text = "" if detour is None else f"{float(detour):.3f}"
            lines.append(
                "| {classification} | {receiver} | {seg_id} {seg_name} | {owners} | {chained} | {detour} | {status} |".format(
                    classification=row.get("classification"),
                    receiver=(row.get("receiver_route") or {}).get("label"),
                    seg_id=bridge.get("seg_id"),
                    seg_name=bridge.get("seg_name"),
                    owners=owners,
                    chained=len(row.get("chained_credit_segment_ids") or []),
                    detour=detour_text,
                    status=row.get("graduation_status"),
                )
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Mid-Segment Junction Proof", ""])
    proof = audit.get("mid_segment_junction_proof") or {}
    lines.append(f"- Status: `{proof.get('status')}`")
    lines.append(f"- Count: {proof.get('mid_segment_junction_count')}")
    if proof.get("junctions"):
        lines.append("- Junctions found; endpoint-only bridge analysis may have false negatives until virtual nodes are inserted.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- This audit detects cross-day bridge duplication and produces repair candidate specs.",
            "- It does not by itself certify replacement route cards; generated candidates still need coverage, GPX, route-card, access, and field-valid cue review.",
            "- Near-bridges are ranked open-ended by detour cost; there is no upper detour cap.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--endpoint-tolerance-miles", type=float, default=DEFAULT_ENDPOINT_TOLERANCE_MILES)
    parser.add_argument("--near-bridge-min-detour-miles", type=float, default=DEFAULT_NEAR_BRIDGE_MIN_DETOUR_MILES)
    parser.add_argument("--connector-snap-tolerance-miles", type=float, default=DEFAULT_CONNECTOR_SNAP_TOLERANCE_MILES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_tool_data = read_json(args.field_tool_data_json)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    audit = build_bridge_duplication_audit(
        field_tool_data,
        official_segments=official_segments,
        connector_graph=connector_graph,
        endpoint_tolerance_miles=args.endpoint_tolerance_miles,
        near_bridge_min_detour_miles=args.near_bridge_min_detour_miles,
        connector_snap_tolerance_miles=args.connector_snap_tolerance_miles,
        source_files={
            "field_tool_data_json": display_path(args.field_tool_data_json),
            "official_geojson": display_path(args.official_geojson),
            "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
            "connector_geojson": display_path(args.connector_geojson),
        },
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="route-bridge-duplication-audit-2026-05-26",
        inputs=[args.field_tool_data_json, args.official_geojson, args.connector_geojson],
        outputs=[args.output_json, args.output_md],
        command="python years/2026/scripts/route_bridge_duplication_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["summary"]["graduated_blocking_strict_bridge_count"] and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
