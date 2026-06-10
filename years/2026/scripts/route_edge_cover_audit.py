#!/usr/bin/env python3
"""Audit route cards as closed required-edge tours from a fixed depot."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    load_connector_graph,
    load_official_segments,
    shortest_connector_path,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "route-edge-cover-audit-2026-05-26.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "route-edge-cover-audit-2026-05-26.md"
MILES_PER_FOOT = 1.0 / 5280.0


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def normalized_ids(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None}, key=lambda item: (len(item), item))


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


def line_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def parse_gpx_track_segments(path: Path) -> list[list[tuple[float, float]]]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (ET.ParseError, FileNotFoundError):
        return []
    segments = []
    for trkseg in root.findall(".//{*}trkseg"):
        points = []
        for trkpt in trkseg.findall("{*}trkpt"):
            lon = trkpt.get("lon")
            lat = trkpt.get("lat")
            if lon is not None and lat is not None:
                points.append((float(lon), float(lat)))
        if len(points) >= 2:
            segments.append(points)
    return segments


def route_gpx_path(route: dict[str, Any], packet_dir: Path) -> Path | None:
    href = route.get("audit_gpx_href") or route.get("gpx_href")
    if not href:
        return None
    path = Path(str(href))
    return path if path.is_absolute() else packet_dir / path


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("route_code") or route.get("outing_id") or "unknown-route")
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment.get("seg_id")): segment for segment in official_segments if segment.get("seg_id") is not None}


def segment_coords(segment: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(lon), float(lat)) for lon, lat in segment.get("coordinates") or []]


def segment_miles(segment: dict[str, Any]) -> float:
    if segment.get("official_miles") is not None:
        return float(segment["official_miles"])
    length_ft = segment.get("LengthFt") or segment.get("length_ft")
    if length_ft is not None:
        return float(length_ft) * MILES_PER_FOOT
    return line_miles(segment_coords(segment))


def snap_required_edges(
    required_segments: list[dict[str, Any]],
    tolerance_miles: float,
) -> tuple[nx.MultiGraph, dict[int, dict[str, Any]]]:
    endpoints: list[tuple[float, float]] = []
    endpoint_refs: list[tuple[str, int]] = []
    for edge_index, segment in enumerate(required_segments):
        coords = segment_coords(segment)
        if len(coords) < 2:
            continue
        endpoint_refs.append(("start", edge_index))
        endpoints.append(coords[0])
        endpoint_refs.append(("end", edge_index))
        endpoints.append(coords[-1])
    uf = UnionFind(len(endpoints))
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            if haversine_miles(endpoints[left], endpoints[right]) <= tolerance_miles:
                uf.union(left, right)
    root_to_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(endpoints)):
        root_to_members[uf.find(index)].append(index)
    root_to_node = {root: node_id for node_id, root in enumerate(sorted(root_to_members))}
    node_info: dict[int, dict[str, Any]] = {}
    for root, members in root_to_members.items():
        node_id = root_to_node[root]
        lon = sum(endpoints[index][0] for index in members) / len(members)
        lat = sum(endpoints[index][1] for index in members) / len(members)
        node_info[node_id] = {"lon": lon, "lat": lat, "point": (lon, lat), "endpoint_count": len(members)}

    edge_nodes: dict[int, dict[str, int]] = defaultdict(dict)
    for endpoint_index, (side, edge_index) in enumerate(endpoint_refs):
        edge_nodes[edge_index][side] = root_to_node[uf.find(endpoint_index)]

    graph = nx.MultiGraph()
    for edge_index, segment in enumerate(required_segments):
        nodes = edge_nodes.get(edge_index) or {}
        if "start" not in nodes or "end" not in nodes:
            continue
        graph.add_edge(
            nodes["start"],
            nodes["end"],
            weight=segment_miles(segment),
            seg_id=str(segment.get("seg_id")),
        )
    return graph, node_info


def straight_line_min_matching(odd_nodes: list[int], node_info: dict[int, dict[str, Any]]) -> float:
    if len(odd_nodes) < 2:
        return 0.0
    complete = nx.Graph()
    for index, left in enumerate(odd_nodes):
        for right in odd_nodes[index + 1 :]:
            distance = haversine_miles(node_info[left]["point"], node_info[right]["point"])
            complete.add_edge(left, right, weight=-distance, distance=distance)
    matching = nx.algorithms.matching.max_weight_matching(complete, maxcardinality=True, weight="weight")
    return sum(float(complete[left][right]["distance"]) for left, right in matching)


def point_pair_distance(
    left: tuple[float, float],
    right: tuple[float, float],
    connector_graph: dict[str, Any] | None,
) -> float:
    if connector_graph:
        path = shortest_connector_path(left, right, connector_graph, 0.08)
        if path and path.get("distance_miles") is not None:
            return float(path["distance_miles"])
    return haversine_miles(left, right)


def component_attachment_miles(
    graph: nx.MultiGraph,
    node_info: dict[int, dict[str, Any]],
    depot_point: tuple[float, float] | None,
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
) -> tuple[float, int]:
    if not graph.number_of_nodes() or depot_point is None:
        return 0.0, 0
    components = [set(component) for component in nx.connected_components(graph)]
    labels = ["depot", *range(len(components))]
    complete = nx.Graph()
    for label in labels:
        complete.add_node(label)

    def points_for(label: str | int) -> list[tuple[float, float]]:
        if label == "depot":
            return [depot_point]
        return [node_info[node]["point"] for node in components[int(label)]]

    for index, left_label in enumerate(labels):
        for right_label in labels[index + 1 :]:
            left_points = points_for(left_label)
            right_points = points_for(right_label)
            best = min(
                point_pair_distance(left, right, connector_graph)
                for left in left_points
                for right in right_points
            )
            if best <= snap_tolerance_miles:
                best = 0.0
            complete.add_edge(left_label, right_label, weight=best)
    if complete.number_of_nodes() <= 1:
        return 0.0, len(components)
    tree = nx.minimum_spanning_tree(complete, weight="weight")
    return sum(float(data.get("weight") or 0.0) for _left, _right, data in tree.edges(data=True)), len(components)


def route_depot_point(route: dict[str, Any], track_segments: list[list[tuple[float, float]]]) -> tuple[float, float] | None:
    parking = route.get("parking") or {}
    if parking.get("lon") is not None and parking.get("lat") is not None:
        return (float(parking["lon"]), float(parking["lat"]))
    for segment in track_segments:
        if segment:
            return segment[0]
    return None


def phase_reset_failures(route: dict[str, Any]) -> list[dict[str, Any]]:
    required_ids = set(normalized_ids(route.get("segment_ids") or []))
    credited: set[str] = set()
    failures = []
    for cue in route.get("wayfinding_cues") or []:
        cue_type = str(cue.get("cue_type") or "")
        if cue_type == "car_pass_connector" and credited:
            remaining = required_ids - credited
            if remaining:
                failures.append(
                    {
                        "code": "depot_revisit_before_required_edges_cleared",
                        "seq": cue.get("seq"),
                        "cue_type": cue_type,
                        "remaining_segment_ids": normalized_ids(remaining),
                        "message": "The route revisits the depot/trailhead before all required edges are credited.",
                    }
                )
        credited.update(normalized_ids(cue.get("official_segment_ids") or []))
    return failures


def phase_reset_is_direction_constrained(route: dict[str, Any], failure: dict[str, Any]) -> bool:
    """Treat some depot revisits as advisory when one-way credit makes them intrinsic.

    The undirected edge-cover check is intentionally strict for normal routes.
    For ascent-only route cards, though, a parking point can be the forced
    uphill end of more than one required edge. In that case a car pass before
    the final edge is a directed-routing warning, not proof that a shorter
    same-activity ordering exists.
    """

    direction_evidence = route.get("segment_direction_evidence") or {}
    if not direction_evidence:
        return False
    required_ids = set(normalized_ids(route.get("segment_ids") or []))
    remaining_ids = set(normalized_ids(failure.get("remaining_segment_ids") or []))
    credited_before_failure = required_ids - remaining_ids
    relevant_ids = credited_before_failure | remaining_ids
    if not relevant_ids or relevant_ids != required_ids:
        return False
    return all(
        (direction_evidence.get(segment_id) or {}).get("direction_rule") == "ascent"
        for segment_id in relevant_ids
    )


def audit_route(
    route: dict[str, Any],
    *,
    official_by_id: dict[str, dict[str, Any]],
    packet_dir: Path,
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
) -> dict[str, Any]:
    label = route_label(route)
    required_ids = normalized_ids(route.get("segment_ids") or [])
    required_segments = [official_by_id[seg_id] for seg_id in required_ids if seg_id in official_by_id]
    gpx_path = route_gpx_path(route, packet_dir)
    track_segments = parse_gpx_track_segments(gpx_path) if gpx_path else []
    generated_miles = sum(line_miles(segment) for segment in track_segments)
    if not gpx_path or not track_segments:
        missing_gpx_failures = [
            {
                "code": "missing_route_quality_gpx",
                "message": "No route GPX was available for edge-cover quality analysis.",
            }
        ]
    else:
        missing_gpx_failures = []
    required_graph, node_info = snap_required_edges(required_segments, snap_tolerance_miles)
    odd_nodes = sorted(node for node, degree in required_graph.degree() if degree % 2 == 1)
    required_miles = sum(segment_miles(segment) for segment in required_segments)
    depot_point = route_depot_point(route, track_segments)
    parity_miles = straight_line_min_matching(odd_nodes, node_info)
    attachment_miles, component_count = component_attachment_miles(
        required_graph,
        node_info,
        depot_point,
        connector_graph,
        snap_tolerance_miles,
    )
    phase_resets = phase_reset_failures(route)
    hard_failures = list(missing_gpx_failures)
    advisory_findings = []
    for failure in phase_resets:
        if phase_reset_is_direction_constrained(route, failure):
            advisory = dict(failure)
            advisory["severity"] = "advisory"
            advisory["message"] = (
                "The route revisits the depot before all required edges are credited, "
                "but every required edge in this route is ascent-constrained; keep "
                "advisory until a directed split-route replacement or stronger "
                "directed route proof improves it."
            )
            advisory_findings.append(advisory)
        elif component_count <= 1:
            hard_failures.append(failure)
        else:
            advisory = dict(failure)
            advisory["severity"] = "advisory"
            advisory["message"] = (
                "The route revisits the depot before all required edges are credited, "
                "but the required-edge subgraph is disconnected; keep advisory until "
                "a same-depot replacement or connector-graph proof is concrete."
            )
            advisory_findings.append(advisory)
    lower_bound = required_miles + parity_miles + attachment_miles
    ratio = generated_miles / lower_bound if lower_bound else None
    status = "failed" if hard_failures else "passed"
    quality_status = "failed" if hard_failures else "advisory" if advisory_findings else "passed"
    route_quality = {
        "status": quality_status,
        "required_miles": round(required_miles, 2),
        "generated_miles": round(generated_miles, 2),
        "lower_bound_miles": round(lower_bound, 2),
        "efficiency_ratio": round(ratio, 3) if ratio is not None else None,
        "required_component_count": component_count,
        "odd_required_node_count": len(odd_nodes),
        "parity_add_on_miles": round(parity_miles, 2),
        "depot_component_attachment_miles": round(attachment_miles, 2),
    }
    return {
        "label": label,
        "outing_id": route.get("outing_id"),
        "candidate_ids": [str(value) for value in route.get("candidate_ids") or []],
        "trailhead": route.get("trailhead"),
        "audit_status": status,
        "hard_failures": hard_failures,
        "advisory_findings": advisory_findings,
        "route_quality": route_quality,
        "gpx_path": display_path(gpx_path),
        "segment_ids": required_ids,
    }


def build_route_edge_cover_audit(
    field_tool_data: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    packet_dir: Path,
    connector_graph: dict[str, Any] | None = None,
    connector_graph_path: Path | None = None,
    snap_tolerance_feet: float = 100.0,
) -> dict[str, Any]:
    if connector_graph is None and connector_graph_path and connector_graph_path.exists():
        connector_graph = load_connector_graph(connector_graph_path, official_segments=official_segments)
    official_by_id = segment_index(official_segments)
    snap_tolerance_miles = snap_tolerance_feet * MILES_PER_FOOT
    routes = [
        audit_route(
            route,
            official_by_id=official_by_id,
            packet_dir=packet_dir,
            connector_graph=connector_graph,
            snap_tolerance_miles=snap_tolerance_miles,
        )
        for route in field_tool_data.get("routes") or []
    ]
    failed_routes = [route for route in routes if route["audit_status"] != "passed"]
    phase_reset_count = sum(
        1
        for route in routes
        for failure in route.get("hard_failures") or []
        if failure.get("code") == "depot_revisit_before_required_edges_cleared"
    )
    advisory_routes = [route for route in routes if route.get("advisory_findings")]
    phase_reset_advisory_count = sum(
        1
        for route in routes
        for finding in route.get("advisory_findings") or []
        if finding.get("code") == "depot_revisit_before_required_edges_cleared"
    )
    missing_gpx_count = sum(
        1
        for route in routes
        for failure in route.get("hard_failures") or []
        if failure.get("code") == "missing_route_quality_gpx"
    )
    status = "failed" if failed_routes else "passed"
    return {
        "schema": "boise_trails_route_edge_cover_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "objective": "verify one-car route cards behave as closed required-edge tours from their selected depot",
        "status": status,
        "summary": {
            "route_count": len(routes),
            "failed_route_count": len(failed_routes),
            "missing_gpx_route_count": missing_gpx_count,
            "phase_reset_failure_count": phase_reset_count,
            "advisory_route_count": len(advisory_routes),
            "phase_reset_advisory_count": phase_reset_advisory_count,
        },
        "parameters": {
            "snap_tolerance_feet": snap_tolerance_feet,
        },
        "connector_graph": {
            "path": display_path(connector_graph_path),
            "loaded": bool(connector_graph),
        },
        "failed_routes": failed_routes,
        "advisory_routes": advisory_routes,
        "routes": routes,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Route Edge-Cover Audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Routes: {summary['route_count']}",
        f"- Failed routes: {summary['failed_route_count']}",
        f"- Depot phase-reset failures: {summary['phase_reset_failure_count']}",
        f"- Advisory routes: {summary.get('advisory_route_count', 0)}",
        f"- Advisory depot phase resets: {summary.get('phase_reset_advisory_count', 0)}",
        f"- Missing GPX routes: {summary['missing_gpx_route_count']}",
        "",
        "## Failed Routes",
        "",
    ]
    if not audit.get("failed_routes"):
        lines.append("No hard failures.")
    for route in audit.get("failed_routes") or []:
        quality = route.get("route_quality") or {}
        lines.append(
            f"- {route['label']}: {route.get('hard_failures') or []}; "
            f"generated {quality.get('generated_miles')} mi / lower bound {quality.get('lower_bound_miles')} mi"
        )
    if audit.get("advisory_routes"):
        lines.extend(["", "## Advisory Routes", ""])
        for route in audit.get("advisory_routes") or []:
            quality = route.get("route_quality") or {}
            lines.append(
                f"- {route['label']}: {route.get('advisory_findings') or []}; "
                f"generated {quality.get('generated_miles')} mi / lower bound {quality.get('lower_bound_miles')} mi"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--snap-tolerance-feet", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    official_segments, _official_geojson = load_official_segments(args.official_geojson)
    audit = build_route_edge_cover_audit(
        read_json(args.field_tool_data_json),
        official_segments=official_segments,
        packet_dir=args.packet_dir,
        connector_graph_path=args.connector_geojson,
        snap_tolerance_feet=args.snap_tolerance_feet,
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
