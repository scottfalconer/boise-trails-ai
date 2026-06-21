#!/usr/bin/env python3
"""Audit current route cards for same-anchor spur splits.

This catches the Sheep Camp failure class: a route reaches the junction for a
required spur, but that spur is preserved as a separate car-to-car outing from
the same parking anchor.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, haversine_miles, load_official_segments  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "same-anchor-spur-split-audit-2026-06-20.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "same-anchor-spur-split-audit-2026-06-20.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def round_miles(value: float) -> float:
    return round(float(value), 2)


def normalized_anchor(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def route_label(route: dict[str, Any]) -> str:
    return str(route.get("route_code") or route.get("label") or route.get("outing_id") or "")


def route_anchor(route: dict[str, Any]) -> str:
    return normalized_anchor(route.get("trailhead_display") or route.get("trailhead"))


def route_parking_point(route: dict[str, Any]) -> tuple[float, float] | None:
    parking = route.get("parking") or {}
    if parking.get("lon") is None or parking.get("lat") is None:
        return None
    return (float(parking["lon"]), float(parking["lat"]))


def same_anchor(left: dict[str, Any], right: dict[str, Any], *, parking_threshold_miles: float) -> bool:
    if route_anchor(left) and route_anchor(left) == route_anchor(right):
        return True
    left_point = route_parking_point(left)
    right_point = route_parking_point(right)
    if left_point and right_point:
        return haversine_miles(left_point, right_point) <= parking_threshold_miles
    return False


def coordinate_path_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def densify_coordinates(
    coords: list[tuple[float, float]],
    *,
    max_gap_miles: float = 0.02,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    densified = [coords[0]]
    for start, end in zip(coords, coords[1:]):
        distance = haversine_miles(start, end)
        steps = max(1, int(math.ceil(distance / max_gap_miles)))
        for step in range(1, steps + 1):
            fraction = step / steps
            point = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            if haversine_miles(densified[-1], point) > 0.000001:
                densified.append(point)
    return densified


def gpx_track_coordinates(path: Path) -> list[tuple[float, float]]:
    tree = ET.parse(path)
    root = tree.getroot()
    coords: list[tuple[float, float]] = []
    for element in root.iter():
        if not element.tag.endswith("trkpt"):
            continue
        lon = element.attrib.get("lon")
        lat = element.attrib.get("lat")
        if lon is None or lat is None:
            continue
        coords.append((float(lon), float(lat)))
    return coords


def local_gpx_path(route: dict[str, Any], packet_dir: Path) -> Path | None:
    href = str(route.get("gpx_href") or "")
    if not href:
        return None
    path = packet_dir / href
    return path if path.exists() else None


def load_route_tracks(field_tool_data: dict[str, Any], packet_dir: Path) -> dict[str, list[tuple[float, float]]]:
    tracks: dict[str, list[tuple[float, float]]] = {}
    for route in field_tool_data.get("routes") or []:
        label = route_label(route)
        path = local_gpx_path(route, packet_dir)
        if not label or path is None:
            continue
        tracks[label] = densify_coordinates(gpx_track_coordinates(path))
    return tracks


def endpoint_key(point: tuple[float, float]) -> str:
    return f"{point[0]:.6f},{point[1]:.6f}"


def official_by_id(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment["seg_id"]): segment for segment in official_segments}


def route_segments(route: dict[str, Any], official_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        official_index[str(segment_id)]
        for segment_id in route.get("segment_ids") or []
        if str(segment_id) in official_index
    ]


def min_distance_to_track(point: tuple[float, float], track: list[tuple[float, float]]) -> float:
    if not track:
        return float("inf")
    return min(haversine_miles(point, candidate) for candidate in track)


def network_nodes_and_edges(segments: list[dict[str, Any]]) -> tuple[set[str], list[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    for segment in segments:
        coords = segment.get("coordinates") or []
        if len(coords) < 2:
            continue
        start = endpoint_key(tuple(coords[0]))
        end = endpoint_key(tuple(coords[-1]))
        nodes.update([start, end])
        edges.append((start, end))
    return nodes, edges


def network_connected(nodes: set[str], edges: list[tuple[str, str]]) -> bool:
    if not nodes:
        return False
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen = set()
    stack = [next(iter(nodes))]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(sorted(adjacency.get(node, set()) - seen))
    return seen == nodes


def contact_nodes(
    host_track: list[tuple[float, float]],
    spur_segments: list[dict[str, Any]],
    *,
    endpoint_threshold_miles: float,
) -> list[dict[str, Any]]:
    contacts: dict[str, dict[str, Any]] = {}
    for segment in spur_segments:
        coords = segment.get("coordinates") or []
        if len(coords) < 2:
            continue
        for endpoint_name, point in [("start", tuple(coords[0])), ("end", tuple(coords[-1]))]:
            distance = min_distance_to_track(point, host_track)
            if distance > endpoint_threshold_miles:
                continue
            key = endpoint_key(point)
            entry = contacts.setdefault(
                key,
                {
                    "endpoint": endpoint_name,
                    "segment_ids": [],
                    "distance_miles": round(distance, 4),
                },
            )
            entry["distance_miles"] = min(float(entry["distance_miles"]), round(distance, 4))
            entry["segment_ids"].append(str(segment["seg_id"]))
    return [
        {**value, "segment_ids": sorted(set(value["segment_ids"]))}
        for value in contacts.values()
    ]


def route_official_miles(segments: list[dict[str, Any]]) -> float:
    return sum(float(segment.get("official_miles") or 0.0) for segment in segments)


def route_has_ascent_segments(segments: list[dict[str, Any]]) -> bool:
    return any(str(segment.get("direction") or "") == "ascent" for segment in segments)


def build_audit(
    field_tool_data: dict[str, Any],
    official_segments: list[dict[str, Any]],
    route_tracks: dict[str, list[tuple[float, float]]],
    *,
    endpoint_threshold_miles: float = 0.045,
    parking_threshold_miles: float = 0.10,
    min_savings_miles: float = 0.25,
) -> dict[str, Any]:
    official_index = official_by_id(official_segments)
    routes = [
        route
        for route in field_tool_data.get("routes") or []
        if route.get("field_ready") is True and route.get("segment_ids")
    ]
    findings: list[dict[str, Any]] = []
    advisory_count = 0
    for host in routes:
        host_label = route_label(host)
        host_track = route_tracks.get(host_label) or []
        if not host_track:
            continue
        host_segment_ids = {str(value) for value in host.get("segment_ids") or []}
        for spur in routes:
            spur_label = route_label(spur)
            if spur_label == host_label:
                continue
            if not same_anchor(host, spur, parking_threshold_miles=parking_threshold_miles):
                continue
            spur_segment_ids = {str(value) for value in spur.get("segment_ids") or []}
            if host_segment_ids & spur_segment_ids:
                continue
            if float(spur.get("on_foot_miles") or 0.0) >= float(host.get("on_foot_miles") or 0.0):
                continue
            spur_segments = route_segments(spur, official_index)
            if not spur_segments:
                continue
            nodes, edges = network_nodes_and_edges(spur_segments)
            if not network_connected(nodes, edges):
                advisory_count += 1
                continue
            contacts = contact_nodes(
                host_track,
                spur_segments,
                endpoint_threshold_miles=endpoint_threshold_miles,
            )
            if len(contacts) != 1:
                continue
            official_miles = route_official_miles(spur_segments)
            estimated_incremental = official_miles * 2.0
            separate_on_foot = float(spur.get("on_foot_miles") or 0.0)
            savings = separate_on_foot - estimated_incremental
            if savings < min_savings_miles:
                continue
            findings.append(
                {
                    "host_route": {
                        "outing_id": host.get("outing_id"),
                        "label": host_label,
                        "route_name": host.get("route_name"),
                        "trailhead": host.get("trailhead_display") or host.get("trailhead"),
                        "segment_ids": [str(value) for value in host.get("segment_ids") or []],
                        "on_foot_miles": host.get("on_foot_miles"),
                    },
                    "spur_route": {
                        "outing_id": spur.get("outing_id"),
                        "label": spur_label,
                        "route_name": spur.get("route_name"),
                        "trailhead": spur.get("trailhead_display") or spur.get("trailhead"),
                        "segment_ids": [str(value) for value in spur.get("segment_ids") or []],
                        "official_miles": round_miles(official_miles),
                        "on_foot_miles": spur.get("on_foot_miles"),
                        "has_ascent_segments": route_has_ascent_segments(spur_segments),
                    },
                    "contact": contacts[0],
                    "estimated_incremental_out_and_back_miles": round_miles(estimated_incremental),
                    "estimated_saved_on_foot_miles": round_miles(savings),
                    "status": "blocking_same_anchor_spur_split",
                    "reason": (
                        "The host route reaches exactly one endpoint of the separate route's official "
                        "segment network from the same parking anchor; clearing it as an out-and-back "
                        "spur appears materially cheaper than preserving a separate car-to-car outing."
                    ),
                }
            )
    findings = sorted(
        findings,
        key=lambda item: (
            -float(item["estimated_saved_on_foot_miles"]),
            item["host_route"]["label"],
            item["spur_route"]["label"],
        ),
    )
    return {
        "schema": "boise_trails_same_anchor_spur_split_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not findings else "failed",
        "summary": {
            "route_count": len(routes),
            "finding_count": len(findings),
            "advisory_disconnected_candidate_count": advisory_count,
            "endpoint_threshold_miles": endpoint_threshold_miles,
            "parking_threshold_miles": parking_threshold_miles,
            "min_savings_miles": min_savings_miles,
        },
        "findings": findings,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Same-Anchor Spur Split Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Routes reviewed: {summary['route_count']}",
        f"- Blocking findings: {summary['finding_count']}",
        f"- Endpoint threshold: {summary['endpoint_threshold_miles']} mi",
        f"- Parking threshold: {summary['parking_threshold_miles']} mi",
        f"- Minimum savings threshold: {summary['min_savings_miles']} mi",
        "",
    ]
    if not audit.get("findings"):
        lines.extend(
            [
                "No current field-packet route has a material same-anchor spur split.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(["## Findings", ""])
    for finding in audit["findings"]:
        host = finding["host_route"]
        spur = finding["spur_route"]
        lines.extend(
            [
                f"### {host['label']} -> {spur['label']}",
                "",
                f"- Host: {host['route_name']} from {host['trailhead']}",
                f"- Separate spur card: {spur['route_name']} ({', '.join(spur['segment_ids'])})",
                f"- Separate on-foot miles: {spur['on_foot_miles']}",
                f"- Estimated incremental out-and-back miles: {finding['estimated_incremental_out_and_back_miles']}",
                f"- Estimated saved on-foot miles: {finding['estimated_saved_on_foot_miles']}",
                f"- Reason: {finding['reason']}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--endpoint-threshold-miles", type=float, default=0.045)
    parser.add_argument("--parking-threshold-miles", type=float, default=0.10)
    parser.add_argument("--min-savings-miles", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_json)
    official_segments, _meta = load_official_segments(args.official_geojson)
    route_tracks = load_route_tracks(field_tool_data, args.packet_dir)
    audit = build_audit(
        field_tool_data,
        official_segments,
        route_tracks,
        endpoint_threshold_miles=args.endpoint_threshold_miles,
        parking_threshold_miles=args.parking_threshold_miles,
        min_savings_miles=args.min_savings_miles,
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
