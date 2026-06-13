#!/usr/bin/env python3
"""Audit exported field-packet routes as a headless field walker.

This is intentionally a product-level audit: it reads the exported phone packet
and Nav GPX, then checks whether a runner could discover each non-credit access
or connector leg from the visible cue text. It is not a replacement for the
optimizer's internal route proof.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import (  # noqa: E402
    connector_class_for_properties,
    iter_line_parts,
    read_json,
    unsafe_connector_access_reasons,
)


DEFAULT_FIELD_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_FIELD_TOOL_DATA_JSON = DEFAULT_FIELD_PACKET_DIR / "field-tool-data.json"
DEFAULT_OFFICIAL_GEOJSON = (
    YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
)
DEFAULT_CONNECTOR_GEOJSON = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "routing-connectors-2026-05-04"
    / "combined_r2r_osm_connectors.geojson"
)
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "field-route-walkthrough-audit-2026-05-06.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "field-route-walkthrough-audit-2026-05-06.md"
TRAIL_NUMBER_RE = re.compile(r"#\s*([0-9]+[A-Z]?)\b", re.IGNORECASE)
GENERIC_NAME_WORDS = {
    "trail",
    "trails",
    "loop",
    "loops",
    "the",
}
MOVEMENT_CUE_TYPES = {
    "start_access",
    "official_segment_start",
    "follow_official_segment",
    "junction_turn",
    "connector_named_trail",
    "connector_road",
    "repeat_official_noncredit",
    "exit_access",
    "return_to_car",
}


class TrailEdge:
    def __init__(
        self,
        *,
        edge_id: str,
        name: str,
        normalized_name: str,
        signposts: set[str],
        source_class: str,
        coords: list[tuple[float, float]],
        segment_id: str | None = None,
        direction: str = "both",
        raw_properties: dict[str, Any] | None = None,
    ) -> None:
        self.edge_id = str(edge_id)
        self.name = str(name or "")
        self.normalized_name = str(normalized_name or "")
        self.signposts = set(signposts or set())
        self.source_class = str(source_class or "unknown_connector")
        self.coords = [(float(lon), float(lat)) for lon, lat in coords]
        self.segment_id = str(segment_id) if segment_id is not None else None
        self.direction = str(direction or "both")
        self.raw_properties = dict(raw_properties or {})
        lons = [point[0] for point in self.coords] or [0.0]
        lats = [point[1] for point in self.coords] or [0.0]
        self.bbox = (min(lons), min(lats), max(lons), max(lats))


class TrailGraph:
    def __init__(self, edges: list[TrailEdge], cell_degrees: float = 0.001) -> None:
        self.edges = list(edges)
        self.cell_degrees = float(cell_degrees)
        self.buckets: dict[tuple[int, int], list[tuple[TrailEdge, int, tuple[float, float], tuple[float, float]]]] = {}
        for edge in self.edges:
            for segment_index, (start, end) in enumerate(zip(edge.coords, edge.coords[1:])):
                min_lon = min(start[0], end[0])
                max_lon = max(start[0], end[0])
                min_lat = min(start[1], end[1])
                max_lat = max(start[1], end[1])
                min_x = math.floor(min_lon / self.cell_degrees)
                max_x = math.floor(max_lon / self.cell_degrees)
                min_y = math.floor(min_lat / self.cell_degrees)
                max_y = math.floor(max_lat / self.cell_degrees)
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        self.buckets.setdefault((x, y), []).append((edge, segment_index, start, end))

    def candidate_segments(
        self, point: tuple[float, float]
    ) -> list[tuple[TrailEdge, int, tuple[float, float], tuple[float, float]]]:
        lon, lat = point
        x = math.floor(lon / self.cell_degrees)
        y = math.floor(lat / self.cell_degrees)
        seen = set()
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for edge, segment_index, start, end in self.buckets.get((x + dx, y + dy), []):
                    key = (edge.edge_id, segment_index)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((edge, segment_index, start, end))
        return candidates


class SegmentSpatialIndex:
    def __init__(self, track_segments: list[list[tuple[float, float]]], cell_degrees: float = 0.001) -> None:
        self.cell_degrees = float(cell_degrees)
        self.buckets: dict[tuple[int, int], list[tuple[tuple[float, float], tuple[float, float]]]] = {}
        self.segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for track in track_segments:
            for start, end in zip(track, track[1:]):
                self.segments.append((start, end))
                min_lon = min(start[0], end[0])
                max_lon = max(start[0], end[0])
                min_lat = min(start[1], end[1])
                max_lat = max(start[1], end[1])
                min_x = math.floor(min_lon / self.cell_degrees)
                max_x = math.floor(max_lon / self.cell_degrees)
                min_y = math.floor(min_lat / self.cell_degrees)
                max_y = math.floor(max_lat / self.cell_degrees)
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        self.buckets.setdefault((x, y), []).append((start, end))

    def candidate_segments(self, point: tuple[float, float]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        lon, lat = point
        x = math.floor(lon / self.cell_degrees)
        y = math.floor(lat / self.cell_degrees)
        seen = set()
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for start, end in self.buckets.get((x + dx, y + dy), []):
                    key = (start, end)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((start, end))
        return candidates

    def distance_to_point(self, point: tuple[float, float]) -> float:
        candidates = self.candidate_segments(point)
        if not candidates:
            return float("inf")
        return min(
            point_to_segment_distance_miles(point, start, end, point[1])
            for start, end in candidates
        )


class OfficialSegment:
    def __init__(
        self,
        *,
        segment_id: str,
        name: str,
        direction: str,
        parts: list[list[tuple[float, float]]],
    ) -> None:
        self.segment_id = str(segment_id)
        self.name = str(name or "")
        self.direction = str(direction or "both")
        self.parts = parts


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


def local_xy_miles(point: tuple[float, float], origin_lat: float) -> tuple[float, float]:
    lon, lat = point
    return (lon * 69.172 * math.cos(math.radians(origin_lat)), lat * 69.0)


def point_to_segment_distance_miles(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    origin_lat: float,
) -> float:
    lon_scale = 69.172 * math.cos(math.radians(origin_lat))
    lat_scale = 69.0
    px, py = point[0] * lon_scale, point[1] * lat_scale
    ax, ay = start[0] * lon_scale, start[1] * lat_scale
    bx, by = end[0] * lon_scale, end[1] * lat_scale
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_to_polyline_distance_miles(point: tuple[float, float], coords: list[tuple[float, float]]) -> float:
    if not coords:
        return float("inf")
    if len(coords) == 1:
        return haversine_miles(point, coords[0])
    return min(
        point_to_segment_distance_miles(point, start, end, point[1])
        for start, end in zip(coords, coords[1:])
    )


def normalize_name(value: Any) -> str:
    text = str(value or "").replace("’", "'").lower()
    text = TRAIL_NUMBER_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    words = [word for word in text.split() if word not in GENERIC_NAME_WORDS]
    return " ".join(words)


def extract_signposts(value: Any) -> set[str]:
    return {match.upper() for match in TRAIL_NUMBER_RE.findall(str(value or ""))}


def trail_name_from_segment_name(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+\d+$", "", text).strip() or text


def edge_label(name: str, signposts: set[str]) -> str:
    if signposts and name and not str(name).strip().startswith("#"):
        return f"#{sorted(signposts)[0]} {name}"
    if name:
        return str(name)
    if signposts:
        return f"#{sorted(signposts)[0]}"
    return "unnamed edge"


def feature_name(props: dict[str, Any]) -> str:
    trail_id = str(props.get("TrailID") or "").strip()
    name = (
        props.get("TrailName")
        or props.get("Name")
        or props.get("SystemName")
        or props.get("segName")
        or props.get("segment_name")
        or props.get("name")
        or ""
    )
    if trail_id and name and not str(name).startswith("#"):
        return f"#{trail_id} {name}"
    return str(name or "")


def edges_from_official_geojson(data: dict[str, Any]) -> tuple[list[TrailEdge], dict[str, OfficialSegment]]:
    edges: list[TrailEdge] = []
    official_index: dict[str, OfficialSegment] = {}
    for feature_index, feature in enumerate(data.get("features") or []):
        props = feature.get("properties") or {}
        seg_id = props.get("segId") or props.get("seg_id")
        if seg_id is None:
            continue
        segment_id = str(seg_id)
        segment_name = str(props.get("segName") or props.get("segment_name") or "")
        trail_name = str(props.get("trail_name") or props.get("TrailName") or "").strip()
        if not trail_name:
            trail_name = trail_name_from_segment_name(segment_name)
        direction = str(props.get("direction") or props.get("direction_rule") or "both")
        parts = iter_line_parts(feature.get("geometry") or {})
        official_index[segment_id] = OfficialSegment(
            segment_id=segment_id,
            name=trail_name,
            direction=direction,
            parts=parts,
        )
        for part_index, coords in enumerate(parts):
            signposts = extract_signposts(trail_name) | extract_signposts(segment_name)
            name = edge_label(trail_name, signposts) if signposts else trail_name
            edges.append(
                TrailEdge(
                    edge_id=f"official:{segment_id}:{feature_index}:{part_index}",
                    name=name,
                    normalized_name=normalize_name(name),
                    signposts=signposts,
                    source_class="official_segment",
                    coords=coords,
                    segment_id=segment_id,
                    direction=direction,
                    raw_properties=props,
                )
            )
    return edges, official_index


def edges_from_connector_geojson(data: dict[str, Any]) -> list[TrailEdge]:
    edges: list[TrailEdge] = []
    for feature_index, feature in enumerate(data.get("features") or []):
        props = feature.get("properties") or {}
        unsafe_reasons = unsafe_connector_access_reasons(props)
        source_class = "private_or_blocked" if unsafe_reasons else connector_class_for_properties(props, "connector")
        raw_name = feature_name(props) or f"connector {props.get('OBJECTID') or feature_index}"
        signposts = extract_signposts(raw_name)
        for part_index, coords in enumerate(iter_line_parts(feature.get("geometry") or {})):
            edges.append(
                TrailEdge(
                    edge_id=f"connector:{feature_index}:{part_index}",
                    name=raw_name,
                    normalized_name=normalize_name(raw_name),
                    signposts=signposts,
                    source_class=source_class,
                    coords=coords,
                    raw_properties=props,
                )
            )
    return edges


def parse_gpx_track_segments(path: Path) -> list[list[tuple[float, float]]]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (ET.ParseError, FileNotFoundError):
        return []
    segments = []
    for trkseg in root.findall(".//{*}trkseg"):
        points = []
        for trkpt in trkseg.findall("{*}trkpt"):
            lat = trkpt.get("lat")
            lon = trkpt.get("lon")
            if lat is None or lon is None:
                continue
            points.append((float(lon), float(lat)))
        if len(points) >= 2:
            segments.append(points)
    return segments


def flatten_track_segments(track_segments: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    return [point for segment in track_segments for point in segment]


def resample_track_segments_for_matching(
    track_segments: list[list[tuple[float, float]]],
    spacing_miles: float = 0.01,
) -> list[list[tuple[float, float]]]:
    sampled_segments: list[list[tuple[float, float]]] = []
    for segment in track_segments:
        if len(segment) < 2:
            continue
        sampled = [segment[0]]
        distance_since_keep = 0.0
        last_point = segment[0]
        for point in segment[1:]:
            distance_since_keep += haversine_miles(last_point, point)
            last_point = point
            if distance_since_keep >= spacing_miles:
                sampled.append(point)
                distance_since_keep = 0.0
        if sampled[-1] != segment[-1]:
            sampled.append(segment[-1])
        sampled_segments.append(sampled)
    return sampled_segments


def bbox_for_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in points] or [0.0]
    lats = [point[1] for point in points] or [0.0]
    return (min(lons), min(lats), max(lons), max(lats))


def expand_bbox(bbox: tuple[float, float, float, float], margin_degrees: float) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        min_lon - margin_degrees,
        min_lat - margin_degrees,
        max_lon + margin_degrees,
        max_lat + margin_degrees,
    )


def bboxes_intersect(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


def filter_edges_for_track(
    graph_edges: list[TrailEdge],
    track_segments: list[list[tuple[float, float]]],
    margin_degrees: float = 0.025,
) -> list[TrailEdge]:
    route_bbox = expand_bbox(bbox_for_points(flatten_track_segments(track_segments)), margin_degrees)
    return [edge for edge in graph_edges if bboxes_intersect(edge.bbox, route_bbox)]


def line_length_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(a, b) for a, b in zip(coords, coords[1:]))


def route_intervals(track_segments: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    intervals = []
    route_miles = 0.0
    for segment_index, segment in enumerate(track_segments):
        for point_index, (start, end) in enumerate(zip(segment, segment[1:])):
            distance = haversine_miles(start, end)
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            intervals.append(
                {
                    "segment_index": segment_index,
                    "point_index": point_index,
                    "start": start,
                    "end": end,
                    "midpoint": midpoint,
                    "distance": distance,
                    "route_miles_start": route_miles,
                    "route_miles_end": route_miles + distance,
                }
            )
            route_miles += distance
    return intervals


def closest_edge(
    point: tuple[float, float],
    graph: TrailGraph,
    preferred_edge_ids: set[str] | None = None,
) -> tuple[TrailEdge | None, float, int | None]:
    measured: list[tuple[float, TrailEdge, int]] = []
    for edge, segment_index, start, end in graph.candidate_segments(point):
        distance = point_to_segment_distance_miles(point, start, end, point[1])
        measured.append((distance, edge, segment_index))
    if not measured:
        return None, float("inf"), None
    best_distance = min(distance for distance, _edge, _segment_index in measured)
    near = [
        (distance, edge, segment_index)
        for distance, edge, segment_index in measured
        if distance <= best_distance + 0.01
    ]
    distance, edge, segment_index = min(
        near,
        key=lambda item: (
            0 if preferred_edge_ids and item[1].edge_id in preferred_edge_ids else 1,
            edge_match_priority(item[1]),
            item[0],
        ),
    )
    return edge, distance, segment_index


def edge_match_priority(edge: TrailEdge) -> int:
    if edge.source_class == "official_segment":
        return 0
    if edge.source_class == "r2r_trail":
        return 1
    if edge.source_class in {"osm_path_footway", "osm_public_road"}:
        return 2
    return 3


def matched_edge_groups(
    track_segments: list[list[tuple[float, float]]],
    graph_edges: list[TrailEdge] | TrailGraph,
    snap_tolerance_miles: float,
    preferred_text: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = graph_edges if isinstance(graph_edges, TrailGraph) else TrailGraph(graph_edges)
    preferred_signposts = extract_signposts(preferred_text) if preferred_text else set()
    preferred_normalized_text = normalized_text_blob(preferred_text) if preferred_text else ""
    preferred_normalized_tokens = preferred_normalized_text.split() if preferred_normalized_text else []
    preferred_edge_ids = {
        edge.edge_id
        for edge in graph.edges
        if preferred_text
        and text_mentions_edge_precomputed(
            preferred_signposts,
            preferred_normalized_text,
            edge,
            preferred_normalized_tokens,
        )
    }
    groups: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for interval in route_intervals(track_segments):
        edge, distance, edge_segment_index = closest_edge(interval["midpoint"], graph, preferred_edge_ids=preferred_edge_ids)
        traversal_direction = (
            edge_traversal_direction(edge, interval["start"], interval["end"], edge_segment_index) if edge else None
        )
        if edge is None or distance > snap_tolerance_miles:
            unmatched.append(
                {
                    "code": "unmatched_route_geometry",
                    "severity": "error",
                    "message": f"Route interval could not be matched to an allowed named edge within {snap_tolerance_miles:.3f} mi.",
                    "coordinate": list(interval["midpoint"]),
                    "distance_miles": round(distance, 4) if distance != float("inf") else None,
                }
            )
            edge = None
            traversal_direction = None
        if (
            not groups
            or groups[-1].get("edge_id") != (edge.edge_id if edge else None)
            or groups[-1].get("traversal_direction") != traversal_direction
        ):
            groups.append(
                {
                    "edge_id": edge.edge_id if edge else None,
                    "name": edge.name if edge else "unmatched route geometry",
                    "signposts": sorted(edge.signposts) if edge else [],
                    "source_class": edge.source_class if edge else "unmatched",
                    "segment_id": edge.segment_id if edge else None,
                    "direction": edge.direction if edge else None,
                    "traversal_direction": traversal_direction,
                    "distance_miles": 0.0,
                    "route_miles_start": interval["route_miles_start"],
                    "route_miles_end": interval["route_miles_end"],
                    "start_coordinate": list(interval["start"]),
                    "end_coordinate": list(interval["end"]),
                    "_edge": edge,
                }
            )
        groups[-1]["distance_miles"] += interval["distance"]
        groups[-1]["route_miles_end"] = interval["route_miles_end"]
        groups[-1]["end_coordinate"] = list(interval["end"])
    for group in groups:
        group["distance_miles"] = round(float(group["distance_miles"]), 3)
        group["route_miles_start"] = round(float(group["route_miles_start"]), 3)
        group["route_miles_end"] = round(float(group["route_miles_end"]), 3)
    return groups, unmatched


def cue_text_values(route: dict[str, Any]) -> list[str]:
    values = []
    for step in route.get("turn_by_turn_steps") or []:
        values.extend([str(step.get("title") or ""), str(step.get("detail") or "")])
    for cue in route.get("wayfinding_cues") or []:
        values.extend(
            [
                str(cue.get("compact") or ""),
                str(cue.get("display_detail") or ""),
                str(cue.get("action") or ""),
                str(cue.get("target") or ""),
                str(cue.get("until") or ""),
            ]
        )
        values.extend(str(item) for item in cue.get("signed_as") or [])
        values.extend(str(item) for item in cue.get("landmarks") or [])
    return [value for value in values if value.strip()]


def start_access_text(route: dict[str, Any]) -> str:
    if route.get("wayfinding_cues"):
        values = []
        for cue in route.get("wayfinding_cues") or []:
            if cue.get("official_segment_ids"):
                break
            values.extend(
                [
                    str(cue.get("compact") or ""),
                    str(cue.get("display_detail") or ""),
                    str(cue.get("target") or ""),
                    str(cue.get("until") or ""),
                ]
            )
            values.extend(str(item) for item in cue.get("signed_as") or [])
        return " ".join(values)
    values = []
    for step in route.get("turn_by_turn_steps") or []:
        if str(step.get("kind") or "").lower() in {"park", "access", "start", "trailhead"}:
            values.extend([str(step.get("title") or ""), str(step.get("detail") or "")])
    return " ".join(values)


def normalized_text_blob(values: list[str] | str) -> str:
    if isinstance(values, str):
        values = [values]
    return normalize_name(" ".join(values))


def normalized_tokens(value: str) -> list[str]:
    return normalize_name(value).split()


def token_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    left_base = left[:-1] if len(left) > 3 and left.endswith("s") else left
    right_base = right[:-1] if len(right) > 3 and right.endswith("s") else right
    return bool(left_base and left_base == right_base)


def token_sequence_mentioned(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(0, len(haystack) - len(needle) + 1):
        if all(token_matches(left, right) for left, right in zip(needle, haystack[start : start + len(needle)])):
            return True
    return False


def text_mentions_edge(text: str, edge: TrailEdge) -> bool:
    if not text:
        return False
    raw_text = str(text)
    signposts = extract_signposts(raw_text)
    if edge.signposts and edge.signposts & signposts:
        return True
    normalized = normalized_text_blob(raw_text)
    return text_mentions_edge_precomputed(signposts, normalized, edge)


def text_mentions_edge_precomputed(
    signposts: set[str],
    normalized_text: str,
    edge: TrailEdge,
    normalized_tokens_cache: list[str] | None = None,
) -> bool:
    if edge.signposts and edge.signposts & signposts:
        return True
    normalized = normalized_text
    if edge.normalized_name and edge.normalized_name in normalized:
        return True
    return token_sequence_mentioned(
        normalized_tokens(edge.name),
        normalized_tokens_cache if normalized_tokens_cache is not None else normalized.split(),
    )


def cue_vocabulary_summary(route: dict[str, Any]) -> dict[str, Any]:
    values = cue_text_values(route)
    signposts = sorted({f"#{value}" for value in extract_signposts(" ".join(values))})
    names = sorted({normalize_name(value) for value in values if normalize_name(value)})
    return {
        "signposts": signposts,
        "normalized_text_count": len(names),
        "sample_normalized_text": names[:20],
    }


def route_declares_gap(route: dict[str, Any]) -> bool:
    text = " ".join(cue_text_values(route)).lower()
    if any(token in text for token in ("re-park", "repark", "multi-start", "manual hold", "manual field check")):
        return True
    for cue in route.get("wayfinding_cues") or []:
        cue_type = str(cue.get("cue_type") or "")
        if cue_type in {"connector_named_trail", "connector_road", "repeat_official_noncredit", "exit_access", "start_access"}:
            if cue.get("signed_as") or cue.get("road_name") or cue.get("landmarks"):
                return True
    return False


def failure(
    code: str,
    message: str,
    *,
    coordinate: tuple[float, float] | list[float] | None = None,
    expected_cue_text_hint: str | None = None,
    severity: str = "error",
) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if coordinate is not None:
        item["coordinate"] = list(coordinate)
    if expected_cue_text_hint:
        item["expected_cue_text_hint"] = expected_cue_text_hint
    return item


def dedupe_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in failures:
        key = (
            item.get("code"),
            item.get("message"),
            item.get("expected_cue_text_hint"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def named_nonofficial_group(group: dict[str, Any]) -> bool:
    edge = group.get("_edge")
    return bool(edge and edge.source_class != "official_segment" and (edge.name or edge.signposts) and not is_generic_connector_name(edge.name))


def is_generic_connector_name(name: str) -> bool:
    text = str(name or "").strip().lower()
    return bool(
        re.fullmatch(r"osm\s+(path|footway|service|track|unclassified|residential|primary|secondary|tertiary)\s+connector\s+\d+", text)
        or re.fullmatch(r"connector[_\s-]*\d+", text)
        or text == "connector"
    )


def official_group_claimed(group: dict[str, Any], claimed_segment_ids: set[str]) -> bool:
    edge = group.get("_edge")
    return bool(edge and edge.source_class == "official_segment" and edge.segment_id in claimed_segment_ids)


def unsafe_edge_reasons(edge: TrailEdge, strict: bool = False) -> list[str]:
    reasons = []
    if edge.source_class == "private_or_blocked":
        reasons.append("private_or_blocked")
    for key in ("access", "foot"):
        value = str(edge.raw_properties.get(key) or "").strip().lower()
        if value in {"no", "private"}:
            reasons.append(f"{key}={value}")
    if strict and edge.source_class == "unknown_connector":
        reasons.append("unknown_connector")
    return sorted(set(reasons))


def sample_polyline(coords: list[tuple[float, float]], spacing_miles: float = 0.01) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return list(coords)
    samples = [coords[0]]
    for start, end in zip(coords, coords[1:]):
        distance = haversine_miles(start, end)
        steps = max(1, int(math.ceil(distance / spacing_miles)))
        for step in range(1, steps + 1):
            fraction = step / steps
            samples.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
    return samples


def project_miles_along_polyline(point: tuple[float, float], coords: list[tuple[float, float]]) -> tuple[float, float]:
    best_along = 0.0
    best_distance = float("inf")
    along = 0.0
    for start, end in zip(coords, coords[1:]):
        segment_miles = haversine_miles(start, end)
        origin_lat = point[1]
        px, py = local_xy_miles(point, origin_lat)
        ax, ay = local_xy_miles(start, origin_lat)
        bx, by = local_xy_miles(end, origin_lat)
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            fraction = 0.0
            distance = math.hypot(px - ax, py - ay)
        else:
            fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
            distance = math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))
        if distance < best_distance:
            best_distance = distance
            best_along = along + fraction * segment_miles
        along += segment_miles
    return best_along, best_distance


def point_fraction_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    origin_lat: float,
) -> tuple[float, float]:
    px, py = local_xy_miles(point, origin_lat)
    ax, ay = local_xy_miles(start, origin_lat)
    bx, by = local_xy_miles(end, origin_lat)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return 0.0, math.hypot(px - ax, py - ay)
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    distance = math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))
    return fraction, distance


def edge_traversal_direction(
    edge: TrailEdge | None,
    start: tuple[float, float],
    end: tuple[float, float],
    segment_index: int | None = None,
) -> str | None:
    if not edge or len(edge.coords) < 2:
        return None
    if segment_index is not None and 0 <= segment_index < len(edge.coords) - 1:
        edge_start = edge.coords[segment_index]
        edge_end = edge.coords[segment_index + 1]
        origin_lat = (start[1] + end[1] + edge_start[1] + edge_end[1]) / 4
        start_fraction, _start_gap = point_fraction_on_segment(start, edge_start, edge_end, origin_lat)
        end_fraction, _end_gap = point_fraction_on_segment(end, edge_start, edge_end, origin_lat)
        if abs(end_fraction - start_fraction) >= 1e-6:
            return "forward" if end_fraction > start_fraction else "reverse"

        sx, sy = local_xy_miles(start, origin_lat)
        ex, ey = local_xy_miles(end, origin_lat)
        ax, ay = local_xy_miles(edge_start, origin_lat)
        bx, by = local_xy_miles(edge_end, origin_lat)
        dot = (ex - sx) * (bx - ax) + (ey - sy) * (by - ay)
        if abs(dot) >= 1e-9:
            return "forward" if dot > 0 else "reverse"
        return "stationary"
    start_mile, _start_gap = project_miles_along_polyline(start, edge.coords)
    end_mile, _end_gap = project_miles_along_polyline(end, edge.coords)
    if abs(end_mile - start_mile) < 1e-6:
        return "stationary"
    return "forward" if end_mile > start_mile else "reverse"


def route_distance_for_point(
    point: tuple[float, float],
    track_segments: list[list[tuple[float, float]]],
) -> tuple[float, float]:
    best_route_miles = 0.0
    best_distance = float("inf")
    route_miles = 0.0
    for segment in track_segments:
        for start, end in zip(segment, segment[1:]):
            segment_miles = haversine_miles(start, end)
            origin_lat = point[1]
            px, py = local_xy_miles(point, origin_lat)
            ax, ay = local_xy_miles(start, origin_lat)
            bx, by = local_xy_miles(end, origin_lat)
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                fraction = 0.0
                distance = math.hypot(px - ax, py - ay)
            else:
                fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                distance = math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))
            if distance < best_distance:
                best_distance = distance
                best_route_miles = route_miles + fraction * segment_miles
            route_miles += segment_miles
    return best_route_miles, best_distance


def official_parts(entry: Any) -> list[list[tuple[float, float]]]:
    if hasattr(entry, "parts"):
        return entry.parts
    if hasattr(entry, "coords"):
        return [entry.coords]
    return []


def official_direction(entry: Any) -> str:
    return str(getattr(entry, "direction", "both") or "both")


def official_name(entry: Any) -> str:
    return str(getattr(entry, "name", "") or getattr(entry, "trail_name", "") or "")


def allowed_geometry_directions_for_segment(route: dict[str, Any], segment_id: str) -> set[str]:
    evidence = (route.get("segment_direction_evidence") or {}).get(str(segment_id)) or {}
    direction = str(evidence.get("allowed_geometry_direction") or "").strip().lower()
    if direction in {"forward", "reverse"}:
        return {direction}
    if direction in {"both", "either"}:
        return {"forward", "reverse"}
    direction_cue = str(evidence.get("direction_cue") or "").lower()
    if "opposite official geometry" in direction_cue:
        return {"reverse"}
    return {"forward"}


def segment_coverage_ratio(
    official_entry: Any,
    track_segments: list[list[tuple[float, float]]],
    snap_tolerance_miles: float,
    track_index: SegmentSpatialIndex | None = None,
) -> float:
    samples = []
    for part in official_parts(official_entry):
        samples.extend(sample_polyline(part))
    if not samples:
        return 0.0
    index = track_index or SegmentSpatialIndex(track_segments)
    covered = 0
    for point in samples:
        if index.distance_to_point(point) <= snap_tolerance_miles:
            covered += 1
    return covered / len(samples)


def audit_official_coverage(
    route: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    official_index: dict[str, Any],
    snap_tolerance_miles: float,
    min_coverage_ratio: float,
    matched_groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    failures = []
    track_index = SegmentSpatialIndex(track_segments)
    for segment_id in sorted({str(item) for item in route.get("segment_ids") or []}):
        official_entry = official_index.get(segment_id)
        if not official_entry:
            continue
        coverage = segment_coverage_ratio(official_entry, track_segments, snap_tolerance_miles, track_index)
        if coverage < min_coverage_ratio:
            failures.append(
                failure(
                    "claimed_segment_not_covered",
                    f"Route claims official segment {segment_id} ({official_name(official_entry)}) but Nav GPX covers only {coverage:.0%} within tolerance.",
                    expected_cue_text_hint="Rebuild the Nav GPX so it covers the full official segment endpoint-to-endpoint, or remove this segment from the route claim.",
                )
            )
        if official_direction(official_entry).lower() in {"ascent", "oneway", "forward"}:
            allowed_directions = allowed_geometry_directions_for_segment(route, segment_id)
            segment_groups = [
                group for group in matched_groups or []
                if str(group.get("segment_id") or "") == segment_id
            ]
            group_directions = {str(group.get("traversal_direction") or "") for group in segment_groups}
            if group_directions & allowed_directions:
                continue
            disallowed_directions = {"forward", "reverse"} - allowed_directions
            if group_directions & disallowed_directions:
                failures.append(
                    failure(
                        "direction_rule_violated",
                        f"Route traverses direction-specific segment {segment_id} ({official_name(official_entry)}) opposite the official geometry direction.",
                        expected_cue_text_hint="Reverse this route leg, add explicit ascent-direction evidence, or split the outing so the direction-specific segment is traversed in the allowed direction.",
                    )
                )
                continue
            for part in official_parts(official_entry):
                if len(part) < 2:
                    continue
                start_mile, start_gap = route_distance_for_point(part[0], track_segments)
                end_mile, end_gap = route_distance_for_point(part[-1], track_segments)
                fallback_direction = "forward" if start_mile < end_mile else "reverse"
                if (
                    start_gap <= snap_tolerance_miles
                    and end_gap <= snap_tolerance_miles
                    and fallback_direction not in allowed_directions
                ):
                    failures.append(
                        failure(
                            "direction_rule_violated",
                            f"Route traverses direction-specific segment {segment_id} ({official_name(official_entry)}) opposite the official geometry direction.",
                            expected_cue_text_hint="Reverse this route leg, add explicit ascent-direction evidence, or split the outing so the direction-specific segment is traversed in the allowed direction.",
                        )
                    )
                    break
    return failures


def graph_node_key(point: tuple[float, float], precision: int = 5) -> tuple[float, float]:
    return (round(point[0], precision), round(point[1], precision))


def node_degree_counts(graph_edges: list[TrailEdge]) -> Counter[tuple[float, float]]:
    counts: Counter[tuple[float, float]] = Counter()
    for edge in graph_edges:
        if len(edge.coords) < 2:
            continue
        counts[graph_node_key(edge.coords[0])] += 1
        counts[graph_node_key(edge.coords[-1])] += 1
    return counts


def generic_osm_connector_edge(edge: TrailEdge | None) -> bool:
    if not edge:
        return False
    normalized = normalize_name(edge.name)
    return normalized.startswith(("osm service connector", "osm path connector", "osm track connector"))


def generic_connector_is_covered_by_adjacent_named_text(
    all_text: str,
    previous_edge: TrailEdge | None,
    current_edge: TrailEdge,
    next_edge: TrailEdge | None,
) -> bool:
    if not generic_osm_connector_edge(current_edge):
        return False
    for adjacent in (previous_edge, next_edge):
        if not adjacent or generic_osm_connector_edge(adjacent):
            continue
        if text_mentions_edge(all_text, adjacent):
            return True
    return False


def audit_route_walkthrough(
    route: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    graph_edges: list[TrailEdge] | TrailGraph,
    official_index: dict[str, Any],
    *,
    snap_tolerance_miles: float = 0.015,
    max_gap_miles: float = 0.05,
    min_coverage_ratio: float = 0.95,
    strict: bool = False,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    graph = graph_edges if isinstance(graph_edges, TrailGraph) else TrailGraph(graph_edges)
    all_text = " ".join(cue_text_values(route))
    matching_track_segments = resample_track_segments_for_matching(track_segments)
    groups, unmatched_failures = matched_edge_groups(matching_track_segments, graph, snap_tolerance_miles, preferred_text=all_text)
    failures.extend(unmatched_failures)

    for left, right in zip(track_segments, track_segments[1:]):
        if not left or not right:
            continue
        gap = haversine_miles(left[-1], right[0])
        if gap > max_gap_miles and not route_declares_gap(route):
            failures.append(
                failure(
                    "hidden_track_gap",
                    f"Nav GPX has an unexplained inter-track-segment gap of {gap:.2f} mi.",
                    coordinate=right[0],
                    expected_cue_text_hint="Declare this as a named connector, road connector, re-park boundary, or manual field hold.",
                )
            )

    claimed_segment_ids = {str(item) for item in route.get("segment_ids") or []}
    start_text = start_access_text(route)
    first_official_index = next(
        (index for index, group in enumerate(groups) if official_group_claimed(group, claimed_segment_ids)),
        None,
    )
    last_official_index = None
    for index, group in enumerate(groups):
        if official_group_claimed(group, claimed_segment_ids):
            last_official_index = index

    if first_official_index is not None:
        for group in groups[:first_official_index]:
            edge = group.get("_edge")
            if not edge or not named_nonofficial_group(group):
                continue
            if not text_mentions_edge(start_text, edge):
                failures.append(
                    failure(
                        "start_access_missing_named_edge",
                        f"Route traverses {edge.name} before the first claimed official segment, but the start/access cue does not name it.",
                        coordinate=group.get("start_coordinate"),
                        expected_cue_text_hint=f"Add a start_access cue such as: Follow {edge.name} until the signed first official-segment junction.",
                    )
                )

    for index, group in enumerate(groups):
        edge = group.get("_edge")
        if not edge:
            continue
        reasons = unsafe_edge_reasons(edge, strict=strict)
        if reasons and edge.source_class != "official_segment":
            failures.append(
                failure(
                    "blocked_connector_used",
                    f"Route traverses {edge.name} with unsafe connector access: {', '.join(reasons)}.",
                    coordinate=group.get("start_coordinate"),
                    expected_cue_text_hint="Replace this connector with a public foot-legal trail/road or mark the route as blocked.",
                )
            )
        is_access_or_connector = (
            edge.source_class != "official_segment"
            and (first_official_index is None or index > first_official_index)
            and (last_official_index is None or index < last_official_index)
        )
        if is_access_or_connector and named_nonofficial_group(group) and not text_mentions_edge(all_text, edge):
            failures.append(
                failure(
                    "named_connector_not_cued",
                    f"Route traverses named connector/access edge {edge.name}, but the phone cue text does not name it.",
                    coordinate=group.get("start_coordinate"),
                    expected_cue_text_hint=f"Add a connector cue naming {edge.name}.",
                )
            )

    degrees = node_degree_counts(graph.edges)
    for index, (previous, current) in enumerate(zip(groups, groups[1:])):
        previous_edge = previous.get("_edge")
        current_edge = current.get("_edge")
        if not previous_edge or not current_edge or previous_edge.edge_id == current_edge.edge_id:
            continue
        next_group = groups[index + 2] if index + 2 < len(groups) else {}
        next_edge = next_group.get("_edge") if isinstance(next_group, dict) else None
        node_key = graph_node_key(tuple(current.get("start_coordinate") or current_edge.coords[0]))
        if (
            degrees.get(node_key, 0) >= 3
            and not text_mentions_edge(all_text, current_edge)
            and not generic_connector_is_covered_by_adjacent_named_text(
                all_text,
                previous_edge,
                current_edge,
                next_edge,
            )
        ):
            failures.append(
                failure(
                    "ambiguous_decision_point",
                    f"At a multi-option junction, the cue text does not name outgoing edge {current_edge.name}.",
                    coordinate=current.get("start_coordinate"),
                    expected_cue_text_hint=f"At this junction, explicitly take {current_edge.name}.",
                )
            )

    failures.extend(
        audit_official_coverage(
            route,
            track_segments,
            official_index,
            snap_tolerance_miles,
            min_coverage_ratio,
            groups,
        )
    )

    public_groups = [
        {
            key: value
            for key, value in group.items()
            if key != "_edge"
        }
        for group in groups
    ]
    failures = dedupe_failures(failures)
    return {
        "outing_id": route.get("outing_id"),
        "label": route.get("label"),
        "trailhead": route.get("trailhead"),
        "passed": not failures,
        "traversed_edge_summary": public_groups,
        "cue_vocabulary_summary": cue_vocabulary_summary(route),
        "failures": failures,
    }


def load_graph_edges(official_geojson: dict[str, Any], connector_geojson: dict[str, Any] | None = None) -> tuple[list[TrailEdge], dict[str, OfficialSegment]]:
    official_edges, official_index = edges_from_official_geojson(official_geojson)
    connector_edges = edges_from_connector_geojson(connector_geojson or {"features": []})
    return official_edges + connector_edges, official_index


def build_walkthrough_audit(
    *,
    field_tool_data: dict[str, Any],
    packet_dir: Path,
    official_geojson: dict[str, Any],
    connector_geojson: dict[str, Any] | None = None,
    route_id: str | None = None,
    snap_tolerance_miles: float = 0.015,
    max_gap_miles: float = 0.05,
    min_coverage_ratio: float = 0.95,
    strict: bool = False,
) -> dict[str, Any]:
    graph_edges, official_index = load_graph_edges(official_geojson, connector_geojson)
    route_reports = []
    for route in field_tool_data.get("routes") or []:
        if route_id and route_id not in {str(route.get("outing_id")), str(route.get("label"))}:
            continue
        gpx_href = route.get("gpx_href") or route.get("nav_gpx_href")
        if not gpx_href:
            route_reports.append(
                {
                    "outing_id": route.get("outing_id"),
                    "label": route.get("label"),
                    "trailhead": route.get("trailhead"),
                    "passed": False,
                    "traversed_edge_summary": [],
                    "cue_vocabulary_summary": cue_vocabulary_summary(route),
                    "failures": [failure("missing_nav_gpx", "Route has no navigation GPX href.")],
                }
            )
            continue
        track_segments = parse_gpx_track_segments(packet_dir / str(gpx_href))
        if not track_segments:
            route_reports.append(
                {
                    "outing_id": route.get("outing_id"),
                    "label": route.get("label"),
                    "trailhead": route.get("trailhead"),
                    "passed": False,
                    "traversed_edge_summary": [],
                    "cue_vocabulary_summary": cue_vocabulary_summary(route),
                    "failures": [failure("empty_nav_gpx", f"Could not parse track segments from {gpx_href}.")],
                }
            )
            continue
        local_edges = filter_edges_for_track(graph_edges, track_segments)
        route_reports.append(
            audit_route_walkthrough(
                route,
                track_segments,
                TrailGraph(local_edges),
                official_index,
                snap_tolerance_miles=snap_tolerance_miles,
                max_gap_miles=max_gap_miles,
                min_coverage_ratio=min_coverage_ratio,
                strict=strict,
            )
        )
    failed_routes = [route for route in route_reports if not route.get("passed")]
    failure_counts = Counter(
        failure_item.get("code")
        for route in failed_routes
        for failure_item in route.get("failures") or []
    )
    return {
        "schema": "boise_trails_field_route_walkthrough_audit_v1",
        "status": "passed" if not failed_routes else "failed",
        "settings": {
            "snap_tolerance_miles": snap_tolerance_miles,
            "max_gap_miles": max_gap_miles,
            "min_coverage_ratio": min_coverage_ratio,
            "strict": strict,
            "route_id": route_id,
        },
        "summary": {
            "route_count": len(route_reports),
            "passed_route_count": len(route_reports) - len(failed_routes),
            "failed_route_count": len(failed_routes),
            "graph_edge_count": len(graph_edges),
            "official_segment_count": len(official_index),
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        "routes": sorted(route_reports, key=lambda item: str(item.get("label") or item.get("outing_id") or "")),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Field Route Walkthrough Audit",
        "",
        f"Status: **{report.get('status')}**",
        "",
        "## Summary",
        "",
        f"- Routes: {summary.get('route_count')}",
        f"- Passed routes: {summary.get('passed_route_count')}",
        f"- Failed routes: {summary.get('failed_route_count')}",
        f"- Graph edges: {summary.get('graph_edge_count')}",
        f"- Official segments: {summary.get('official_segment_count')}",
        f"- Failure counts: `{json.dumps(summary.get('failure_counts') or {}, sort_keys=True)}`",
        "",
    ]
    failed = [route for route in report.get("routes") or [] if not route.get("passed")]
    if failed:
        lines.extend(["## Failures", ""])
        for route in failed:
            lines.append(f"### {route.get('label') or route.get('outing_id')} - {route.get('trailhead') or ''}")
            for item in route.get("failures") or []:
                lines.append(f"- `{item.get('code')}`: {item.get('message')}")
                if item.get("expected_cue_text_hint"):
                    lines.append(f"  - Hint: {item.get('expected_cue_text_hint')}")
            lines.append("")
    else:
        lines.extend(["## Failures", "", "None.", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_FIELD_PACKET_DIR)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--route-id", help="Audit one outing id or label, for example 1B.")
    parser.add_argument("--snap-tolerance-miles", type=float, default=0.015)
    parser.add_argument("--max-gap-miles", type=float, default=0.05)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--strict", action="store_true", help="Also fail matched unknown connector classes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    field_tool_data = read_json(args.field_tool_data_json)
    official_geojson = read_json(args.official_geojson)
    connector_geojson = read_json(args.connector_geojson) if args.connector_geojson and args.connector_geojson.exists() else None
    report = build_walkthrough_audit(
        field_tool_data=field_tool_data,
        packet_dir=args.packet_dir,
        official_geojson=official_geojson,
        connector_geojson=connector_geojson,
        route_id=args.route_id,
        snap_tolerance_miles=args.snap_tolerance_miles,
        max_gap_miles=args.max_gap_miles,
        min_coverage_ratio=args.min_coverage_ratio,
        strict=args.strict,
    )
    write_report(report, args.output_json, args.output_md)
    summary = report.get("summary") or {}
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "route_count": summary.get("route_count"),
                "passed_route_count": summary.get("passed_route_count"),
                "failed_route_count": summary.get("failed_route_count"),
                "failure_counts": summary.get("failure_counts"),
            },
            indent=2,
        )
    )
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
