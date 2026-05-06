#!/usr/bin/env python3
"""Persist reusable connector-graph shortest-path distances between planning nodes."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    load_connector_graph,
    load_official_segments,
    load_trailheads_from_geojson,
    nearest_connector_node_for_graph,
)
from rural_postman_lower_bound import (  # noqa: E402
    MILES_PER_FOOT,
    odd_node_ids,
    read_json,
    required_edges_from_geojson,
    required_graph,
    snap_endpoint_nodes,
)


DEFAULT_OUTPUT_JSON = YEAR_DIR / "derived" / "connector-matrix" / "connector-shortest-path-matrix-2026-05-06.json"
DEFAULT_PRIVATE_PARKING_ANCHORS = YEAR_DIR / "inputs" / "personal" / "private" / "strava-parking-anchors-v1.geojson"
DEFAULT_PRIVATE_OUTPUT_JSON = (
    YEAR_DIR / "inputs" / "personal" / "private" / "connector-shortest-path-matrix-with-strava-parking-v1.json"
)


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def official_odd_endpoint_nodes(official_geojson: dict[str, Any], snap_tolerance_feet: float) -> list[dict[str, Any]]:
    edges = required_edges_from_geojson(official_geojson)
    snapped_edges, node_info = snap_endpoint_nodes(edges, snap_tolerance_feet * MILES_PER_FOOT)
    graph = required_graph(snapped_edges)
    odd_nodes = odd_node_ids(graph)
    return [
        {
            "node_id": f"official_odd_{node_id}",
            "node_type": "official_odd_endpoint",
            "source_node_id": node_id,
            "lon": node_info[node_id]["lon"],
            "lat": node_info[node_id]["lat"],
        }
        for node_id in odd_nodes
    ]


def trailhead_nodes(path: Path, *, private: bool = False) -> list[dict[str, Any]]:
    nodes = []
    for index, trailhead in enumerate(load_trailheads_from_geojson(path), start=1):
        nodes.append(
            {
                "node_id": f"{'private_' if private else ''}trailhead_{index:03d}",
                "node_type": "private_parking_anchor" if private else "public_trailhead",
                "name": trailhead["name"],
                "lon": float(trailhead["lon"]),
                "lat": float(trailhead["lat"]),
                "parking_confidence": trailhead.get("parking_confidence"),
                "source": trailhead.get("source"),
            }
        )
    return nodes


def snap_nodes_to_connector_graph(
    nodes: list[dict[str, Any]],
    connector_graph: dict[str, Any],
    snap_tolerance_miles: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapped = []
    unsnapped = []
    for node in nodes:
        nearest = nearest_connector_node_for_graph((float(node["lon"]), float(node["lat"])), connector_graph, snap_tolerance_miles)
        if not nearest:
            unsnapped.append(node)
            continue
        copied = dict(node)
        copied["connector_node"] = [nearest[0][0], nearest[0][1]]
        copied["snap_miles"] = round(float(nearest[1]), 4)
        snapped.append(copied)
    return snapped, unsnapped


def dijkstra_to_targets(
    graph: dict[tuple[float, float], list[dict[str, Any]]],
    source: tuple[float, float],
    targets: set[tuple[float, float]],
) -> dict[tuple[float, float], float]:
    queue: list[tuple[float, int, tuple[float, float]]] = [(0.0, 0, source)]
    distances = {source: 0.0}
    found = {}
    remaining = set(targets)
    push_count = 1
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


def build_matrix(
    nodes: list[dict[str, Any]],
    connector_graph: dict[str, Any],
    snap_tolerance_feet: float,
) -> dict[str, Any]:
    snapped, unsnapped = snap_nodes_to_connector_graph(nodes, connector_graph, snap_tolerance_feet * MILES_PER_FOOT)
    target_nodes = {
        (float(node["connector_node"][0]), float(node["connector_node"][1]))
        for node in snapped
    }
    rows = []
    graph = connector_graph.get("graph") or {}
    for source in snapped:
        source_connector = (float(source["connector_node"][0]), float(source["connector_node"][1]))
        distances = dijkstra_to_targets(graph, source_connector, target_nodes)
        for target in snapped:
            if source["node_id"] == target["node_id"]:
                continue
            target_connector = (float(target["connector_node"][0]), float(target["connector_node"][1]))
            graph_distance = distances.get(target_connector)
            if graph_distance is None:
                continue
            total = graph_distance + float(source["snap_miles"]) + float(target["snap_miles"])
            rows.append(
                {
                    "source_node_id": source["node_id"],
                    "target_node_id": target["node_id"],
                    "distance_miles": round(total, 4),
                    "graph_miles": round(graph_distance, 4),
                    "source_snap_miles": source["snap_miles"],
                    "target_snap_miles": target["snap_miles"],
                }
            )
    return {
        "nodes": snapped,
        "unsnapped_nodes": unsnapped,
        "rows": rows,
        "summary": {
            "node_count": len(nodes),
            "snapped_node_count": len(snapped),
            "unsnapped_node_count": len(unsnapped),
            "matrix_row_count": len(rows),
            "directed": True,
            "snap_tolerance_feet": snap_tolerance_feet,
        },
    }


def matrix_payload(
    official_geojson: Path,
    connector_geojson: Path,
    public_trailheads_geojson: Path,
    *,
    private_parking_anchors: Path | None = None,
    include_private: bool = False,
    endpoint_snap_tolerance_feet: float = 50.0,
    connector_snap_tolerance_feet: float = 250.0,
) -> dict[str, Any]:
    official_data = read_json(official_geojson)
    official_segments, _meta = load_official_segments(official_geojson)
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    if connector_graph is None:
        raise ValueError(f"Connector graph did not load: {connector_geojson}")
    nodes = [
        *official_odd_endpoint_nodes(official_data, endpoint_snap_tolerance_feet),
        *trailhead_nodes(public_trailheads_geojson),
    ]
    if include_private and private_parking_anchors and private_parking_anchors.exists():
        nodes.extend(trailhead_nodes(private_parking_anchors, private=True))
    matrix = build_matrix(nodes, connector_graph, connector_snap_tolerance_feet)
    matrix.update(
        {
            "dataset": "connector-shortest-path-matrix-2026-05-06",
            "privacy": "private_exact_coordinates" if include_private else "public_trailheads_and_official_endpoints_only",
            "source_datasets": {
                "official_geojson": display_path(official_geojson),
                "connector_geojson": display_path(connector_geojson),
                "public_trailheads_geojson": display_path(public_trailheads_geojson),
                "private_parking_anchors_geojson": display_path(private_parking_anchors)
                if include_private and private_parking_anchors
                else None,
            },
            "connector_graph": {
                "node_count": len(connector_graph.get("nodes") or []),
                "feature_count": connector_graph.get("feature_count"),
                "official_repeat_segment_count": connector_graph.get("official_segment_count"),
                "connector_class_counts": connector_graph.get("connector_class_counts"),
            },
        }
    )
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--public-trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--private-parking-anchors-geojson", type=Path, default=DEFAULT_PRIVATE_PARKING_ANCHORS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--private-output-json", type=Path, default=DEFAULT_PRIVATE_OUTPUT_JSON)
    parser.add_argument("--endpoint-snap-tolerance-feet", type=float, default=50.0)
    parser.add_argument("--connector-snap-tolerance-feet", type=float, default=250.0)
    parser.add_argument("--write-private", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    public_payload = matrix_payload(
        args.official_geojson,
        args.connector_geojson,
        args.public_trailheads_geojson,
        endpoint_snap_tolerance_feet=args.endpoint_snap_tolerance_feet,
        connector_snap_tolerance_feet=args.connector_snap_tolerance_feet,
    )
    write_json(args.output_json, public_payload)
    print(f"Wrote {args.output_json}")
    print(json.dumps(public_payload["summary"], indent=2))

    if args.write_private:
        private_payload = matrix_payload(
            args.official_geojson,
            args.connector_geojson,
            args.public_trailheads_geojson,
            private_parking_anchors=args.private_parking_anchors_geojson,
            include_private=True,
            endpoint_snap_tolerance_feet=args.endpoint_snap_tolerance_feet,
            connector_snap_tolerance_feet=args.connector_snap_tolerance_feet,
        )
        write_json(args.private_output_json, private_payload)
        print(f"Wrote {args.private_output_json}")
        print(json.dumps(private_payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
