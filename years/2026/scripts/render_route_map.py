#!/usr/bin/env python3
"""Render topology-aware static navigation maps from a GPX track.

The renderer is intentionally offline and deterministic. It uses the ordered GPX
track as the route truth, detects repeated route edges/nodes after local metric
projection, offsets repeated traversals into visible lanes, and writes both SVG
and PNG outputs for phone or print use.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_PALETTE = "#2563eb,#0891b2,#16a34a,#f59e0b,#dc2626,#7c3aed"
GPX_NS = "{http://www.topografix.com/GPX/1/1}"


@dataclass(frozen=True)
class GeoPoint:
    lon: float
    lat: float
    ele_m: float | None = None


@dataclass(frozen=True)
class MetricPoint:
    x: float
    y: float


@dataclass(frozen=True)
class Waypoint:
    name: str
    description: str
    geo: GeoPoint
    metric: MetricPoint | None = None


@dataclass(frozen=True)
class ContextLine:
    name: str
    points: tuple[MetricPoint, ...]
    source: str = ""


@dataclass
class RouteSegment:
    index: int
    start: MetricPoint
    end: MetricPoint
    distance_m: float
    cumulative_start_m: float
    cumulative_end_m: float
    edge_key: tuple[tuple[int, int], tuple[int, int]]
    lane_index: int = 0
    lane_count: int = 1
    offset_m: float = 0.0
    repeated: bool = False

    @property
    def midpoint(self) -> MetricPoint:
        return MetricPoint((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    @property
    def angle_rad(self) -> float:
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)

    @property
    def progress_mid(self) -> float:
        total = max(self.cumulative_end_m, 1.0)
        return (self.cumulative_start_m + self.cumulative_end_m) / 2 / total


@dataclass
class RepeatedNode:
    key: tuple[int, int]
    point: MetricPoint
    visit_indices: list[int]
    cumulative_m: list[float]
    score: float = 0.0
    near_waypoint: bool = False


@dataclass
class DenseArea:
    area_id: str
    bbox: tuple[float, float, float, float]
    center: MetricPoint
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class WaypointRoutePosition:
    waypoint: Waypoint
    cumulative_m: float
    point: MetricPoint
    angle_rad: float
    distance_from_route_m: float
    route_order: int
    nearest_route_index: int


@dataclass(frozen=True)
class WaypointSnapCandidate:
    route_m: float
    route_index: int
    point: MetricPoint
    distance_from_waypoint_m: float


@dataclass(frozen=True)
class SnappedWaypoint:
    waypoint: Waypoint
    selected: WaypointSnapCandidate
    candidates: tuple[WaypointSnapCandidate, ...]
    ambiguous: bool
    segment_name: str | None = None


@dataclass(frozen=True)
class NavCue:
    cue_number: int
    route_m: float
    route_mile: float
    route_index: int
    point: MetricPoint
    geo: GeoPoint
    kind: str
    action: str
    label: str
    source_waypoint_name: str | None
    segment_name: str | None
    repeated_place_id: str | None
    repeated_pass_number: int | None
    ambiguous: bool = False


@dataclass(frozen=True)
class NapkinCue:
    cue_number: int
    route_mile: float
    distance_from_previous_miles: float
    point: MetricPoint
    geo: GeoPoint
    action: str
    display_action: str
    turn_confidence: float
    label: str
    elevation_delta_ft: float | None
    elevation_gain_ft: float | None
    elevation_loss_ft: float | None
    elevation_hint: str
    repeated_place_id: str | None
    repeated_pass_number: int | None
    same_as_cue_number: int | None
    needs_review: bool
    review_reasons: tuple[str, ...]


@dataclass
class RouteAnalysis:
    source_gpx: str
    route_name: str
    geo_points: list[GeoPoint]
    points: list[MetricPoint]
    waypoints: list[Waypoint]
    segments: list[RouteSegment]
    repeated_edge_count: int
    repeated_edge_traversal_count: int
    repeated_nodes: list[RepeatedNode]
    dense_areas: list[DenseArea]
    total_distance_m: float
    bbox: tuple[float, float, float, float]
    projection_origin: GeoPoint
    context_lines: list[ContextLine]


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1400
    height: int = 1000
    dpi: int = 180
    line_width: float = 7.0
    arrow_spacing_m: float = 320.0
    overlap_offset_m: float = 8.0
    min_repeated_stretch_m: float = 50.0
    max_lanes: int = 3
    edge_snap_m: float = 8.0
    node_snap_m: float = 16.0
    label_density: str = "normal"
    basemap_style: str = "muted"
    inset_mode: str = "auto"
    embed_insets: bool = False
    max_insets: int = 3
    inset_size: int = 280
    render_mode: str = "field"
    profile: str = "overview"
    field_route_color: str = "#1d4ed8"
    route_color_mode: str = "solid"
    max_arrows: int = 28
    max_pass_labels_overview: int = 12
    max_pass_labels_inset: int = 25
    simplify_tolerance_m: float = 10.0
    label_mode: str = "nav-cues"
    show_cue_labels: bool = True
    show_segment_labels: bool = False
    show_parking_marker: bool = False
    waypoint_snap_tolerance_m: float = 35.0
    palette: tuple[str, ...] = tuple(DEFAULT_PALETTE.split(","))
    marker_json: Path | None = None
    context_geojson: Path | None = None

    @property
    def max_repeated_node_labels(self) -> int:
        return {"low": 5, "normal": 10, "high": 18}.get(self.label_density, 10)

    @property
    def max_waypoint_labels(self) -> int:
        return {"low": 5, "normal": 12, "high": 24}.get(self.label_density, 12)


def parse_hex_color(value: str) -> tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {value!r}")
    return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def interpolate_color(palette: tuple[str, ...], progress: float) -> str:
    colors = [parse_hex_color(color) for color in palette]
    if not colors:
        colors = [parse_hex_color("#2563eb"), parse_hex_color("#dc2626")]
    if len(colors) == 1:
        return rgb_to_hex(colors[0])
    clamped = max(0.0, min(1.0, progress))
    scaled = clamped * (len(colors) - 1)
    left_index = min(int(math.floor(scaled)), len(colors) - 2)
    fraction = scaled - left_index
    left = colors[left_index]
    right = colors[left_index + 1]
    mixed = tuple(int(round(left[i] + (right[i] - left[i]) * fraction)) for i in range(3))
    return rgb_to_hex(mixed)


def strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if strip_namespace(child.tag) == name and child.text:
            return child.text.strip()
    return ""


def parse_gpx(path: Path) -> tuple[str, list[GeoPoint], list[Waypoint]]:
    tree = ET.parse(path)
    root = tree.getroot()
    metadata = root.find(f"{GPX_NS}metadata")
    route_name = child_text(metadata if metadata is not None else root, "name") or path.stem

    waypoints: list[Waypoint] = []
    for element in root.iter():
        if strip_namespace(element.tag) != "wpt":
            continue
        lat = element.attrib.get("lat")
        lon = element.attrib.get("lon")
        if lat is None or lon is None:
            continue
        waypoints.append(
            Waypoint(
                name=child_text(element, "name") or "Waypoint",
                description=child_text(element, "desc"),
                geo=GeoPoint(lon=float(lon), lat=float(lat), ele_m=float(child_text(element, "ele")) if child_text(element, "ele") else None),
            )
        )

    points: list[GeoPoint] = []
    track_names: list[str] = []
    for track in [item for item in root.iter() if strip_namespace(item.tag) == "trk"]:
        name = child_text(track, "name")
        if name:
            track_names.append(name)
        for point in track.iter():
            if strip_namespace(point.tag) != "trkpt":
                continue
            lat = point.attrib.get("lat")
            lon = point.attrib.get("lon")
            if lat is not None and lon is not None:
                ele = child_text(point, "ele")
                points.append(GeoPoint(lon=float(lon), lat=float(lat), ele_m=float(ele) if ele else None))

    if not points:
        for point in root.iter():
            if strip_namespace(point.tag) != "rtept":
                continue
            lat = point.attrib.get("lat")
            lon = point.attrib.get("lon")
            if lat is not None and lon is not None:
                ele = child_text(point, "ele")
                points.append(GeoPoint(lon=float(lon), lat=float(lat), ele_m=float(ele) if ele else None))

    if len(points) < 2:
        raise ValueError(f"GPX route must contain at least two track or route points: {path}")
    if track_names:
        route_name = track_names[0]
    return route_name, points, waypoints


def projection_origin(points: list[GeoPoint], waypoints: list[Waypoint]) -> GeoPoint:
    all_points = points + [waypoint.geo for waypoint in waypoints]
    return GeoPoint(
        lon=sum(point.lon for point in all_points) / len(all_points),
        lat=sum(point.lat for point in all_points) / len(all_points),
    )


def project(point: GeoPoint, origin: GeoPoint) -> MetricPoint:
    meters_per_degree_lat = 111_132.92
    meters_per_degree_lon = 111_412.84 * math.cos(math.radians(origin.lat))
    return MetricPoint(
        x=(point.lon - origin.lon) * meters_per_degree_lon,
        y=(point.lat - origin.lat) * meters_per_degree_lat,
    )


def distance(left: MetricPoint, right: MetricPoint) -> float:
    return math.hypot(right.x - left.x, right.y - left.y)


def quantize(point: MetricPoint, snap_m: float) -> tuple[int, int]:
    snap = max(snap_m, 0.1)
    return (int(round(point.x / snap)), int(round(point.y / snap)))


def edge_key(left: MetricPoint, right: MetricPoint, snap_m: float) -> tuple[tuple[int, int], tuple[int, int]]:
    a = quantize(left, snap_m)
    b = quantize(right, snap_m)
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def route_bbox(points: list[MetricPoint]) -> tuple[float, float, float, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def expand_bbox(
    bbox: tuple[float, float, float, float],
    padding_m: float,
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bbox
    if max_x - min_x < 1:
        min_x -= 10
        max_x += 10
    if max_y - min_y < 1:
        min_y -= 10
        max_y += 10
    return min_x - padding_m, min_y - padding_m, max_x + padding_m, max_y + padding_m


def build_segments(points: list[MetricPoint], config: RenderConfig) -> tuple[list[RouteSegment], float]:
    segments: list[RouteSegment] = []
    cumulative = 0.0
    for index, (left, right) in enumerate(zip(points, points[1:])):
        segment_distance = distance(left, right)
        if segment_distance <= 0:
            continue
        key = edge_key(left, right, config.edge_snap_m)
        segments.append(
            RouteSegment(
                index=index,
                start=left,
                end=right,
                distance_m=segment_distance,
                cumulative_start_m=cumulative,
                cumulative_end_m=cumulative + segment_distance,
                edge_key=key,
            )
        )
        cumulative += segment_distance
    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for segment in segments:
        edge_counts[segment.edge_key] = edge_counts.get(segment.edge_key, 0) + 1
    seen: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for segment in segments:
        count = edge_counts[segment.edge_key]
        occurrence = seen.get(segment.edge_key, 0)
        seen[segment.edge_key] = occurrence + 1
        segment.lane_count = count
        segment.lane_index = occurrence
        segment.repeated = count > 1
        if count > 1:
            segment.offset_m = (occurrence - (count - 1) / 2.0) * config.overlap_offset_m
    return segments, cumulative


def angle_delta(left: float, right: float) -> float:
    delta = abs((right - left + math.pi) % (2 * math.pi) - math.pi)
    return delta


def find_repeated_nodes(
    points: list[MetricPoint],
    segments: list[RouteSegment],
    config: RenderConfig,
    waypoints: list[Waypoint],
) -> list[RepeatedNode]:
    cumulative_by_point = [0.0]
    for segment in segments:
        cumulative_by_point.append(segment.cumulative_end_m)
    visits: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(points):
        key = quantize(point, config.node_snap_m)
        if key not in visits:
            visits[key] = []
        if visits[key] and index - visits[key][-1] <= 2:
            continue
        visits[key].append(index)

    repeated: list[RepeatedNode] = []
    min_separation_m = max(40.0, config.node_snap_m * 2)
    for key, indices in visits.items():
        if len(indices) < 2:
            continue
        distinct_cumulative: list[float] = []
        distinct_indices: list[int] = []
        for index in indices:
            cumulative = cumulative_by_point[min(index, len(cumulative_by_point) - 1)]
            if not distinct_cumulative or abs(cumulative - distinct_cumulative[-1]) >= min_separation_m:
                distinct_cumulative.append(cumulative)
                distinct_indices.append(index)
        if len(distinct_indices) < 2:
            continue
        avg_x = sum(points[index].x for index in distinct_indices) / len(distinct_indices)
        avg_y = sum(points[index].y for index in distinct_indices) / len(distinct_indices)
        node_point = MetricPoint(avg_x, avg_y)
        direction_bins: set[int] = set()
        turn_angles: list[float] = []
        for index in distinct_indices:
            if index > 0:
                prev_point = points[index - 1]
                direction_bins.add(int(round(math.atan2(node_point.y - prev_point.y, node_point.x - prev_point.x) / (math.pi / 4))))
            if index < len(points) - 1:
                next_point = points[index + 1]
                direction_bins.add(int(round(math.atan2(next_point.y - node_point.y, next_point.x - node_point.x) / (math.pi / 4))))
            if 0 < index < len(points) - 1:
                inbound = math.atan2(node_point.y - points[index - 1].y, node_point.x - points[index - 1].x)
                outbound = math.atan2(points[index + 1].y - node_point.y, points[index + 1].x - node_point.x)
                turn_angles.append(angle_delta(inbound, outbound))
        near_important_waypoint = any(
            waypoint.metric and distance(node_point, waypoint.metric) <= max(config.node_snap_m * 1.3, 20.0)
            for waypoint in important_waypoints(waypoints)
        )
        is_decision_node = (
            near_important_waypoint
            or (len(direction_bins) >= 4 and turn_angles and max(turn_angles) >= math.radians(90))
        )
        if not is_decision_node:
            continue
        max_turn = max(turn_angles) if turn_angles else 0.0
        score = (100.0 if near_important_waypoint else 0.0) + len(distinct_indices) * 3 + len(direction_bins) + max_turn
        repeated.append(
            RepeatedNode(
                key=key,
                point=node_point,
                visit_indices=distinct_indices,
                cumulative_m=distinct_cumulative,
                score=score,
                near_waypoint=near_important_waypoint,
            )
        )

    repeated.sort(key=lambda node: (-node.score, node.cumulative_m[0]))
    selected: list[RepeatedNode] = []
    min_label_spacing_m = max(160.0, config.node_snap_m * 8)
    for node in repeated:
        if len(selected) >= config.max_repeated_node_labels:
            break
        if any(distance(node.point, other.point) < min_label_spacing_m for other in selected):
            continue
        selected.append(node)
    return selected


def detect_dense_areas(
    segments: list[RouteSegment],
    repeated_nodes: list[RepeatedNode],
    config: RenderConfig,
) -> list[DenseArea]:
    candidates: list[tuple[MetricPoint, float, str]] = []
    for node in repeated_nodes:
        candidates.append((node.point, 4.0 + len(node.visit_indices), "repeated intersection"))
    for segment in segments:
        if segment.repeated:
            candidates.append((segment.midpoint, 2.5 + segment.lane_count, "repeated trail segment"))

    for left, center, right in zip(segments, segments[1:], segments[2:]):
        turn = angle_delta(left.angle_rad, right.angle_rad)
        if turn > math.radians(70):
            candidates.append((center.start, 1.0 + turn, "sharp turn"))

    if config.inset_mode == "off" or not candidates:
        return []

    cluster_radius_m = 180.0
    clusters: list[dict[str, Any]] = []
    for point, score, reason in candidates:
        target = None
        for cluster in clusters:
            if distance(point, cluster["center"]) <= cluster_radius_m:
                target = cluster
                break
        if target is None:
            target = {"points": [], "score": 0.0, "reasons": set(), "center": point}
            clusters.append(target)
        target["points"].append(point)
        target["score"] += score
        target["reasons"].add(reason)
        target["center"] = MetricPoint(
            sum(item.x for item in target["points"]) / len(target["points"]),
            sum(item.y for item in target["points"]) / len(target["points"]),
        )

    dense_areas: list[DenseArea] = []
    for index, cluster in enumerate(sorted(clusters, key=lambda item: -item["score"])[: config.max_insets], start=1):
        points = cluster["points"]
        bbox = expand_bbox(route_bbox(points), 140.0)
        dense_areas.append(
            DenseArea(
                area_id=f"inset-{index}",
                bbox=bbox,
                center=cluster["center"],
                score=round(cluster["score"], 2),
                reasons=sorted(cluster["reasons"]),
            )
        )
    return dense_areas


def analyze_gpx(path: Path, config: RenderConfig) -> RouteAnalysis:
    route_name, geo_points, raw_waypoints = parse_gpx(path)
    origin = projection_origin(geo_points, raw_waypoints)
    points = [project(point, origin) for point in geo_points]
    waypoints = [
        Waypoint(
            name=waypoint.name,
            description=waypoint.description,
            geo=waypoint.geo,
            metric=project(waypoint.geo, origin),
        )
        for waypoint in raw_waypoints
    ]
    if config.marker_json:
        waypoints.extend(load_extra_markers(config.marker_json, origin))
    context_lines = load_context_geojson(config.context_geojson, origin) if config.context_geojson else []
    segments, total_distance_m = build_segments(points, config)
    if not segments:
        raise ValueError("GPX route did not contain any non-zero-length segments.")
    repeated_nodes = find_repeated_nodes(points, segments, config, waypoints)
    dense_areas = detect_dense_areas(segments, repeated_nodes, config)
    repeated_edge_keys = {segment.edge_key for segment in segments if segment.repeated}
    bbox_points = points + [waypoint.metric for waypoint in waypoints if waypoint.metric]
    bbox = expand_bbox(route_bbox([point for point in bbox_points if point is not None]), 80.0)
    return RouteAnalysis(
        source_gpx=str(path),
        route_name=route_name,
        geo_points=geo_points,
        points=points,
        waypoints=waypoints,
        segments=segments,
        repeated_edge_count=len(repeated_edge_keys),
        repeated_edge_traversal_count=len([segment for segment in segments if segment.repeated]),
        repeated_nodes=repeated_nodes,
        dense_areas=dense_areas,
        total_distance_m=total_distance_m,
        bbox=bbox,
        projection_origin=origin,
        context_lines=context_lines,
    )


def load_extra_markers(path: Path, origin: GeoPoint) -> list[Waypoint]:
    data = json.loads(path.read_text(encoding="utf-8"))
    markers = data.get("markers") if isinstance(data, dict) else data
    if not isinstance(markers, list):
        raise ValueError("Marker JSON must be a list or an object with a markers list.")
    waypoints: list[Waypoint] = []
    for marker in markers:
        if "lat" not in marker or "lon" not in marker:
            continue
        geo = GeoPoint(lon=float(marker["lon"]), lat=float(marker["lat"]))
        waypoints.append(
            Waypoint(
                name=str(marker.get("name") or marker.get("label") or "Marker"),
                description=str(marker.get("description") or marker.get("type") or ""),
                geo=geo,
                metric=project(geo, origin),
            )
        )
    return waypoints


def iter_geojson_line_parts(geometry: dict[str, Any]) -> list[list[Any]]:
    geom_type = geometry.get("type")
    if geom_type == "LineString":
        return [geometry.get("coordinates", [])]
    if geom_type == "MultiLineString":
        return list(geometry.get("coordinates", []))
    return []


def load_context_geojson(path: Path, origin: GeoPoint) -> list[ContextLine]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features") if isinstance(data, dict) and data.get("type") == "FeatureCollection" else []
    if not isinstance(features, list):
        raise ValueError("Context GeoJSON must be a FeatureCollection.")
    lines: list[ContextLine] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if not isinstance(geometry, dict):
            continue
        for part_index, coordinates in enumerate(iter_geojson_line_parts(geometry), start=1):
            points: list[MetricPoint] = []
            for coordinate in coordinates:
                if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
                    continue
                points.append(project(GeoPoint(lon=float(coordinate[0]), lat=float(coordinate[1])), origin))
            if len(points) < 2:
                continue
            name = str(properties.get("name") or properties.get("segName") or properties.get("trail_name") or f"context-{index}")
            suffix = f"-part-{part_index}" if part_index > 1 else ""
            lines.append(ContextLine(name=f"{name}{suffix}", points=tuple(points), source=str(path)))
    return lines


class Transform:
    def __init__(self, bbox: tuple[float, float, float, float], viewport: tuple[float, float, float, float]) -> None:
        self.min_x, self.min_y, self.max_x, self.max_y = bbox
        self.vx, self.vy, self.vw, self.vh = viewport
        width = max(self.max_x - self.min_x, 1.0)
        height = max(self.max_y - self.min_y, 1.0)
        self.scale = min(self.vw / width, self.vh / height)
        self.pad_x = (self.vw - width * self.scale) / 2
        self.pad_y = (self.vh - height * self.scale) / 2

    def point(self, point: MetricPoint) -> tuple[float, float]:
        x = self.vx + self.pad_x + (point.x - self.min_x) * self.scale
        y = self.vy + self.pad_y + (self.max_y - point.y) * self.scale
        return x, y

    def length(self, meters: float) -> float:
        return meters * self.scale


def offset_points(segment: RouteSegment) -> tuple[MetricPoint, MetricPoint]:
    if not segment.offset_m:
        return segment.start, segment.end
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    length = math.hypot(dx, dy)
    if length <= 0:
        return segment.start, segment.end
    nx = -dy / length
    ny = dx / length
    offset = segment.offset_m
    return (
        MetricPoint(segment.start.x + nx * offset, segment.start.y + ny * offset),
        MetricPoint(segment.end.x + nx * offset, segment.end.y + ny * offset),
    )


def shifted_point_on_segment(segment: RouteSegment, fraction: float) -> MetricPoint:
    start, end = offset_points(segment)
    return MetricPoint(
        x=start.x + (end.x - start.x) * fraction,
        y=start.y + (end.y - start.y) * fraction,
    )


def point_to_segment_distance(point: MetricPoint, start: MetricPoint, end: MetricPoint) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return distance(point, start)
    fraction = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    clamped = max(0.0, min(1.0, fraction))
    projected = MetricPoint(start.x + dx * clamped, start.y + dy * clamped)
    return distance(point, projected)


def project_point_to_segment(
    point: MetricPoint,
    segment: RouteSegment,
) -> tuple[MetricPoint, float, float]:
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return segment.start, 0.0, distance(point, segment.start)
    fraction = ((point.x - segment.start.x) * dx + (point.y - segment.start.y) * dy) / length_sq
    clamped = max(0.0, min(1.0, fraction))
    projected = MetricPoint(segment.start.x + dx * clamped, segment.start.y + dy * clamped)
    return projected, clamped, distance(point, projected)


def nearest_route_point_index(points: list[MetricPoint], target: MetricPoint) -> int:
    return min(range(len(points)), key=lambda index: distance(points[index], target))


def simplify_metric_points(
    points: list[MetricPoint],
    tolerance_m: float,
    preserve_indices: set[int] | None = None,
) -> list[MetricPoint]:
    if len(points) <= 2 or tolerance_m <= 0:
        return points
    preserve = preserve_indices or set()

    def simplify_range(start_index: int, end_index: int, output: list[int]) -> None:
        if end_index <= start_index + 1:
            return
        forced = [index for index in preserve if start_index < index < end_index]
        if forced:
            cursor = start_index
            for forced_index in sorted(forced):
                simplify_range(cursor, forced_index, output)
                output.append(forced_index)
                cursor = forced_index
            simplify_range(cursor, end_index, output)
            return
        start = points[start_index]
        end = points[end_index]
        farthest_index = start_index
        farthest_distance = -1.0
        for index in range(start_index + 1, end_index):
            candidate_distance = point_to_segment_distance(points[index], start, end)
            if candidate_distance > farthest_distance:
                farthest_distance = candidate_distance
                farthest_index = index
        if farthest_distance > tolerance_m:
            simplify_range(start_index, farthest_index, output)
            output.append(farthest_index)
            simplify_range(farthest_index, end_index, output)

    indices = [0]
    simplify_range(0, len(points) - 1, indices)
    indices.append(len(points) - 1)
    deduped = sorted(set(indices))
    return [points[index] for index in deduped]


def field_ribbon_points(analysis: RouteAnalysis, config: RenderConfig) -> list[MetricPoint]:
    preserve_indices = {0, len(analysis.points) - 1}
    for cue in navigation_cues(analysis, config):
        preserve_indices.add(cue.route_index)
    return simplify_metric_points(analysis.points, config.simplify_tolerance_m, preserve_indices)


def cue_boundaries(analysis: RouteAnalysis, config: RenderConfig) -> list[int]:
    boundaries = [0, len(analysis.points) - 1]
    for cue in navigation_cues(analysis, config):
        boundaries.append(cue.route_index)
    return sorted(set(boundaries))


def field_ribbon_parts(analysis: RouteAnalysis, config: RenderConfig) -> list[tuple[list[MetricPoint], str]]:
    if config.route_color_mode == "solid":
        return [(field_ribbon_points(analysis, config), config.field_route_color)]
    if config.route_color_mode == "gradient":
        points = field_ribbon_points(analysis, config)
        parts: list[tuple[list[MetricPoint], str]] = []
        for index, (start, end) in enumerate(zip(points, points[1:])):
            progress = index / max(len(points) - 2, 1)
            parts.append(([start, end], interpolate_color(config.palette, progress)))
        return parts
    boundaries = cue_boundaries(analysis, config)
    parts: list[tuple[list[MetricPoint], str]] = []
    for part_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end <= start:
            continue
        segment_points = analysis.points[start : end + 1]
        preserve = {0, len(segment_points) - 1}
        simplified = simplify_metric_points(segment_points, config.simplify_tolerance_m, preserve)
        if len(simplified) < 2:
            continue
        if config.route_color_mode == "gradient":
            progress = start / max(len(analysis.points) - 1, 1)
            color = interpolate_color(config.palette, progress)
        else:
            color = config.palette[part_index % len(config.palette)]
        parts.append((simplified, color))
    if not parts:
        parts.append((field_ribbon_points(analysis, config), config.field_route_color))
    return parts


@dataclass(frozen=True)
class RepeatedStretch:
    start_index: int
    end_index: int
    distance_m: float


def repeated_stretches(analysis: RouteAnalysis, config: RenderConfig) -> list[RepeatedStretch]:
    stretches: list[RepeatedStretch] = []
    start_index: int | None = None
    distance_m = 0.0
    last_index = 0
    for segment in analysis.segments:
        if segment.repeated:
            if start_index is None:
                start_index = segment.index
                distance_m = 0.0
            distance_m += segment.distance_m
            last_index = segment.index + 1
        else:
            if start_index is not None and distance_m >= config.min_repeated_stretch_m:
                stretches.append(RepeatedStretch(start_index, last_index, distance_m))
            start_index = None
            distance_m = 0.0
    if start_index is not None and distance_m >= config.min_repeated_stretch_m:
        stretches.append(RepeatedStretch(start_index, last_index, distance_m))
    return stretches


def field_arrow_targets(analysis: RouteAnalysis, config: RenderConfig, detail: bool = False) -> list[float]:
    spacing = max(config.arrow_spacing_m * (0.65 if detail else 1.0), 80.0)
    cues = navigation_cues(analysis, config)
    cue_cumulatives = [cue.route_m for cue in cues if 0 < cue.route_m < analysis.total_distance_m]
    boundaries = [0.0] + cue_cumulatives + [analysis.total_distance_m]
    targets: list[float] = []
    if cue_cumulatives:
        for start, end in zip(boundaries, boundaries[1:]):
            leg = end - start
            if leg < 80:
                continue
            targets.append(start + leg * 0.55)
            if leg > max(spacing * 2.2, 850.0):
                targets.append(start + leg * 0.82)
    else:
        target = spacing
        while target < analysis.total_distance_m:
            targets.append(target)
            target += spacing
    if detail:
        for cue in cues:
            for offset in (-35.0, 45.0):
                target = cue.route_m + offset
                if 0 < target < analysis.total_distance_m:
                    targets.append(target)
        for stretch in repeated_stretches(analysis, config):
            start_cum = analysis.segments[stretch.start_index].cumulative_start_m
            end_cum = analysis.segments[min(stretch.end_index - 1, len(analysis.segments) - 1)].cumulative_end_m
            targets.extend([start_cum + 25.0, end_cum - 25.0])
    cleaned: list[float] = []
    for target in sorted(set(round(value, 1) for value in targets)):
        if target <= 0 or target >= analysis.total_distance_m:
            continue
        if cleaned and target - cleaned[-1] < (90 if detail else 150):
            continue
        cleaned.append(target)
    limit = config.max_arrows if not detail else max(config.max_arrows, 90)
    return cleaned[:limit]


def draw_arrow_png(
    draw: ImageDraw.ImageDraw,
    transform: Transform,
    point: MetricPoint,
    angle_rad: float,
    color: str,
    size_px: float,
) -> None:
    x, y = transform.point(point)
    size = max(size_px, 8.0)
    wing = size * 0.55
    points = [
        (x + math.cos(angle_rad) * size, y - math.sin(angle_rad) * size),
        (x - math.cos(angle_rad) * size * 0.45 + math.cos(angle_rad + math.pi / 2) * wing,
         y + math.sin(angle_rad) * size * 0.45 - math.sin(angle_rad + math.pi / 2) * wing),
        (x - math.cos(angle_rad) * size * 0.45 + math.cos(angle_rad - math.pi / 2) * wing,
         y + math.sin(angle_rad) * size * 0.45 - math.sin(angle_rad - math.pi / 2) * wing),
    ]
    draw.polygon(points, fill=color, outline="#ffffff")


def draw_chevron_png(
    draw: ImageDraw.ImageDraw,
    transform: Transform,
    point: MetricPoint,
    angle_rad: float,
    color: str,
    size_px: float,
) -> None:
    x, y = transform.point(point)
    size = max(size_px, 9.0)
    wing = size * 0.65
    tip = (x + math.cos(angle_rad) * size, y - math.sin(angle_rad) * size)
    left = (
        x - math.cos(angle_rad) * size * 0.2 + math.cos(angle_rad + math.pi / 2) * wing,
        y + math.sin(angle_rad) * size * 0.2 - math.sin(angle_rad + math.pi / 2) * wing,
    )
    right = (
        x - math.cos(angle_rad) * size * 0.2 + math.cos(angle_rad - math.pi / 2) * wing,
        y + math.sin(angle_rad) * size * 0.2 - math.sin(angle_rad - math.pi / 2) * wing,
    )
    draw.line([left, tip, right], fill="#ffffff", width=5, joint="curve")
    draw.line([left, tip, right], fill=color, width=3, joint="curve")


def svg_arrow_polygon(transform: Transform, point: MetricPoint, angle_rad: float, size_px: float) -> str:
    x, y = transform.point(point)
    size = max(size_px, 8.0)
    wing = size * 0.55
    points = [
        (x + math.cos(angle_rad) * size, y - math.sin(angle_rad) * size),
        (x - math.cos(angle_rad) * size * 0.45 + math.cos(angle_rad + math.pi / 2) * wing,
         y + math.sin(angle_rad) * size * 0.45 - math.sin(angle_rad + math.pi / 2) * wing),
        (x - math.cos(angle_rad) * size * 0.45 + math.cos(angle_rad - math.pi / 2) * wing,
         y + math.sin(angle_rad) * size * 0.45 - math.sin(angle_rad - math.pi / 2) * wing),
    ]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def svg_chevron_polyline(transform: Transform, point: MetricPoint, angle_rad: float, size_px: float) -> str:
    x, y = transform.point(point)
    size = max(size_px, 9.0)
    wing = size * 0.65
    tip = (x + math.cos(angle_rad) * size, y - math.sin(angle_rad) * size)
    left = (
        x - math.cos(angle_rad) * size * 0.2 + math.cos(angle_rad + math.pi / 2) * wing,
        y + math.sin(angle_rad) * size * 0.2 - math.sin(angle_rad + math.pi / 2) * wing,
    )
    right = (
        x - math.cos(angle_rad) * size * 0.2 + math.cos(angle_rad - math.pi / 2) * wing,
        y + math.sin(angle_rad) * size * 0.2 - math.sin(angle_rad - math.pi / 2) * wing,
    )
    return " ".join(f"{px:.1f},{py:.1f}" for px, py in [left, tip, right])


def route_progress(segment: RouteSegment, total_distance_m: float, fraction: float = 0.5) -> float:
    if total_distance_m <= 0:
        return 0
    return (segment.cumulative_start_m + segment.distance_m * fraction) / total_distance_m


def arrow_fractions(segment: RouteSegment, config: RenderConfig, dense: bool) -> list[float]:
    spacing = config.arrow_spacing_m * (0.55 if dense or segment.repeated else 1.0)
    if segment.distance_m < max(12.0, spacing * 0.35):
        return [0.5] if dense or segment.repeated else []
    count = max(1, int(segment.distance_m / max(spacing, 1.0)))
    return [(index + 1) / (count + 1) for index in range(count)]


def segment_point(segment: RouteSegment, fraction: float, apply_lane_offset: bool = True) -> MetricPoint:
    start, end = offset_points(segment) if apply_lane_offset else (segment.start, segment.end)
    return MetricPoint(
        x=start.x + (end.x - start.x) * fraction,
        y=start.y + (end.y - start.y) * fraction,
    )


def point_at_cumulative(
    analysis: RouteAnalysis,
    cumulative_m: float,
    apply_lane_offset: bool = True,
) -> tuple[MetricPoint, float]:
    target = max(0.0, min(cumulative_m, analysis.total_distance_m))
    for segment in analysis.segments:
        if segment.cumulative_start_m <= target <= segment.cumulative_end_m:
            fraction = (target - segment.cumulative_start_m) / max(segment.distance_m, 1.0)
            return segment_point(segment, fraction, apply_lane_offset), segment.angle_rad
    last = analysis.segments[-1]
    return segment_point(last, 1.0, apply_lane_offset), last.angle_rad


def waypoint_route_positions(analysis: RouteAnalysis) -> list[WaypointRoutePosition]:
    positions: list[WaypointRoutePosition] = []
    for waypoint in important_waypoints(analysis.waypoints):
        if not waypoint.metric:
            continue
        best: tuple[float, float, RouteSegment] | None = None
        for segment in analysis.segments:
            dx = segment.end.x - segment.start.x
            dy = segment.end.y - segment.start.y
            length_sq = dx * dx + dy * dy
            if length_sq <= 0:
                continue
            fraction = ((waypoint.metric.x - segment.start.x) * dx + (waypoint.metric.y - segment.start.y) * dy) / length_sq
            clamped = max(0.0, min(1.0, fraction))
            projected = segment_point(segment, clamped, apply_lane_offset=False)
            off_route_m = distance(waypoint.metric, projected)
            cumulative = segment.cumulative_start_m + segment.distance_m * clamped
            if best is None or off_route_m < best[0]:
                best = (off_route_m, cumulative, segment)
        if best is None:
            continue
        off_route_m, cumulative, segment = best
        route_point, route_angle = point_at_cumulative(analysis, cumulative, apply_lane_offset=False)
        nearest_index = nearest_route_point_index(analysis.points, route_point)
        positions.append(
            WaypointRoutePosition(
                waypoint=waypoint,
                cumulative_m=cumulative,
                point=route_point,
                angle_rad=route_angle,
                distance_from_route_m=off_route_m,
                route_order=segment.index,
                nearest_route_index=nearest_index,
            )
        )
    positions.sort(key=lambda item: (item.cumulative_m, item.waypoint.name))
    deduped: list[WaypointRoutePosition] = []
    for position in positions:
        if deduped and abs(position.cumulative_m - deduped[-1].cumulative_m) < 8 and position.waypoint.name == deduped[-1].waypoint.name:
            continue
        deduped.append(position)
    return deduped


def is_segment_waypoint_name(name: str) -> bool:
    upper = name.upper().strip()
    return upper.startswith("SEG ") or upper.startswith("SEGMENT ")


def parse_segment_name(name: str) -> str | None:
    if not is_segment_waypoint_name(name):
        return None
    parts = name.split(maxsplit=2)
    if len(parts) >= 3 and parts[1].isdigit():
        return parts[2].strip()
    return name


def nav_waypoint_kind(name: str) -> str | None:
    upper = name.upper()
    if is_segment_waypoint_name(name):
        return None
    if "PARK" in upper or "START" in upper:
        return "start"
    if "RETURN" in upper or "FINISH" in upper:
        return "finish"
    if upper.startswith("CUE") or " TURN" in upper or upper.startswith("TURN") or "JUNCTION" in upper:
        return "decision"
    if "CHECKPOINT" in upper or "WATER" in upper or "AID" in upper:
        return "checkpoint"
    return None


def cue_source_sort_key(waypoint: Waypoint) -> tuple[int, int, str]:
    kind = nav_waypoint_kind(waypoint.name)
    if kind == "start":
        return (0, 0, waypoint.name)
    if kind == "finish":
        return (9, 0, waypoint.name)
    for token in waypoint.name.replace("/", " ").split():
        if token.isdigit():
            return (1, int(token), waypoint.name)
    return (2, 0, waypoint.name)


def marker_number(cue: NavCue) -> str:
    return str(cue.cue_number)


def legacy_waypoint_marker_number(name: str, fallback: int) -> str:
    upper = name.upper()
    if "PARK" in upper or "START" in upper:
        return "P"
    if "RETURN" in upper or "FINISH" in upper:
        return "F"
    for token in upper.replace("/", " ").split():
        if token.isdigit():
            return token
    return str(fallback)


def cue_label(name: str) -> str:
    cleaned = name
    for prefix in ("PARK/START", "RETURN TO CAR", "CUE", "TURN"):
        cleaned = cleaned.replace(prefix, "")
    tokens = cleaned.split()
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    return " ".join(tokens)[:36]


def snap_waypoint_candidates(
    analysis: RouteAnalysis,
    waypoint: Waypoint,
    tolerance_m: float,
) -> tuple[WaypointSnapCandidate, ...]:
    if not waypoint.metric:
        return tuple()
    all_candidates: list[WaypointSnapCandidate] = []
    for segment in analysis.segments:
        projected, fraction, off_route_m = project_point_to_segment(waypoint.metric, segment)
        all_candidates.append(
            WaypointSnapCandidate(
                route_m=segment.cumulative_start_m + segment.distance_m * fraction,
                route_index=segment.index,
                point=projected,
                distance_from_waypoint_m=off_route_m,
            )
        )
    all_candidates.sort(key=lambda item: (item.distance_from_waypoint_m, item.route_m))
    near = [candidate for candidate in all_candidates if candidate.distance_from_waypoint_m <= tolerance_m]
    if not near and all_candidates:
        near = [all_candidates[0]]
    # Collapse adjacent raw-GPX candidates from the same physical pass.
    collapsed: list[WaypointSnapCandidate] = []
    for candidate in sorted(near, key=lambda item: item.route_m):
        if collapsed and abs(candidate.route_m - collapsed[-1].route_m) < 25:
            if candidate.distance_from_waypoint_m < collapsed[-1].distance_from_waypoint_m:
                collapsed[-1] = candidate
            continue
        collapsed.append(candidate)
    return tuple(collapsed)


def choose_monotonic_candidate(
    candidates: tuple[WaypointSnapCandidate, ...],
    previous_route_m: float,
    prefer_end: bool = False,
    prefer_start: bool = False,
) -> WaypointSnapCandidate:
    if not candidates:
        raise ValueError("Cannot choose from empty waypoint snap candidates.")
    if prefer_start:
        return min(candidates, key=lambda item: (item.route_m, item.distance_from_waypoint_m))
    if prefer_end:
        return max(candidates, key=lambda item: (item.route_m, -item.distance_from_waypoint_m))
    forward = [candidate for candidate in candidates if candidate.route_m > previous_route_m + 10]
    if forward:
        return min(forward, key=lambda item: (item.route_m, item.distance_from_waypoint_m))
    return min(candidates, key=lambda item: (abs(item.route_m - previous_route_m), item.distance_from_waypoint_m))


def route_geo_at_index(analysis: RouteAnalysis, route_index: int) -> GeoPoint:
    index = max(0, min(route_index, len(analysis.geo_points) - 1))
    return analysis.geo_points[index]


def route_geo_at_cumulative(analysis: RouteAnalysis, cumulative_m: float) -> GeoPoint:
    target = max(0.0, min(cumulative_m, analysis.total_distance_m))
    for segment in analysis.segments:
        if segment.cumulative_start_m <= target <= segment.cumulative_end_m:
            fraction = (target - segment.cumulative_start_m) / max(segment.distance_m, 1.0)
            start = analysis.geo_points[max(0, min(segment.index, len(analysis.geo_points) - 1))]
            end = analysis.geo_points[max(0, min(segment.index + 1, len(analysis.geo_points) - 1))]
            return GeoPoint(
                lon=start.lon + (end.lon - start.lon) * fraction,
                lat=start.lat + (end.lat - start.lat) * fraction,
                ele_m=(
                    start.ele_m + (end.ele_m - start.ele_m) * fraction
                    if start.ele_m is not None and end.ele_m is not None
                    else None
                ),
            )
    return analysis.geo_points[-1]


def repeated_place_for_cue(
    cue_point: MetricPoint,
    cue_route_m: float,
    repeated_nodes: list[RepeatedNode],
) -> tuple[str | None, int | None]:
    nearest: tuple[int, RepeatedNode, float] | None = None
    for index, node in enumerate(repeated_nodes, start=1):
        dist = distance(cue_point, node.point)
        if dist > 45:
            continue
        if nearest is None or dist < nearest[2]:
            nearest = (index, node, dist)
    if nearest is None:
        return None, None
    place_index, node, _dist = nearest
    pass_number = min(
        range(1, len(node.cumulative_m) + 1),
        key=lambda idx: abs(node.cumulative_m[idx - 1] - cue_route_m),
    )
    return f"J{place_index:02d}", pass_number


def nav_cue_action(kind: str, label: str) -> str:
    if kind == "start":
        return "Start at car"
    if kind == "finish":
        return "Return to car / finish"
    if kind == "checkpoint":
        return f"Pass {label}".strip()
    return f"Navigate to {label}".strip()


def snap_source_waypoint(
    analysis: RouteAnalysis,
    config: RenderConfig,
    waypoint: Waypoint,
    previous_route_m: float = -1.0,
    prefer_start: bool = False,
    prefer_end: bool = False,
) -> SnappedWaypoint | None:
    candidates = snap_waypoint_candidates(analysis, waypoint, config.waypoint_snap_tolerance_m)
    if not candidates:
        return None
    selected = choose_monotonic_candidate(
        candidates,
        previous_route_m,
        prefer_start=prefer_start,
        prefer_end=prefer_end,
    )
    return SnappedWaypoint(
        waypoint=waypoint,
        selected=selected,
        candidates=candidates,
        ambiguous=len(candidates) > 1,
        segment_name=parse_segment_name(waypoint.name),
    )


def build_navigation_cues(analysis: RouteAnalysis, config: RenderConfig) -> tuple[list[NavCue], list[SnappedWaypoint]]:
    snapped: list[SnappedWaypoint] = []
    nav_waypoints = [waypoint for waypoint in analysis.waypoints if nav_waypoint_kind(waypoint.name)]
    nav_waypoints.sort(key=cue_source_sort_key)
    previous_route_m = -1.0
    cue_specs: list[tuple[float, int, MetricPoint, str, str, str | None, str | None, bool]] = []

    start_waypoint = next((waypoint for waypoint in nav_waypoints if nav_waypoint_kind(waypoint.name) == "start"), None)
    finish_waypoint = next((waypoint for waypoint in nav_waypoints if nav_waypoint_kind(waypoint.name) == "finish"), None)
    cue_specs.append((0.0, 0, analysis.points[0], "start", "START / CAR", start_waypoint.name if start_waypoint else None, None, False))
    previous_route_m = 0.0

    for waypoint in nav_waypoints:
        kind = nav_waypoint_kind(waypoint.name)
        if kind in {"start", "finish"}:
            continue
        snapped_waypoint = snap_source_waypoint(analysis, config, waypoint, previous_route_m)
        if not snapped_waypoint:
            continue
        selected = snapped_waypoint.selected
        snapped.append(snapped_waypoint)
        label = cue_label(waypoint.name) or waypoint.name
        cue_specs.append((selected.route_m, selected.route_index, selected.point, kind or "decision", label, waypoint.name, None, snapped_waypoint.ambiguous))
        previous_route_m = max(previous_route_m, selected.route_m)

    if start_waypoint and start_waypoint.metric:
        start_snap = snap_source_waypoint(analysis, config, start_waypoint, prefer_start=True)
        if start_snap:
            snapped.append(start_snap)
    if finish_waypoint and finish_waypoint.metric:
        finish_snap = snap_source_waypoint(analysis, config, finish_waypoint, analysis.total_distance_m, prefer_end=True)
        if finish_snap:
            snapped.append(finish_snap)

    for waypoint in analysis.waypoints:
        if not is_segment_waypoint_name(waypoint.name):
            continue
        segment_snap = snap_source_waypoint(analysis, config, waypoint)
        if segment_snap:
            snapped.append(segment_snap)

    cue_specs.append(
        (
            analysis.total_distance_m,
            len(analysis.points) - 1,
            analysis.points[-1],
            "finish",
            "RETURN / FINISH",
            finish_waypoint.name if finish_waypoint else None,
            None,
            False,
        )
    )
    cue_specs.sort(key=lambda item: (item[0], 0 if item[3] == "start" else 1))
    deduped_specs: list[tuple[float, int, MetricPoint, str, str, str | None, str | None, bool]] = []
    for spec in cue_specs:
        if deduped_specs and abs(spec[0] - deduped_specs[-1][0]) < 8 and spec[3] == deduped_specs[-1][3]:
            continue
        deduped_specs.append(spec)

    cues: list[NavCue] = []
    for cue_number_value, (route_m, route_index, point, kind, label, source_name, segment_name, ambiguous) in enumerate(deduped_specs, start=1):
        repeated_place_id, repeated_pass_number = repeated_place_for_cue(point, route_m, analysis.repeated_nodes)
        cues.append(
            NavCue(
                cue_number=cue_number_value,
                route_m=route_m,
                route_mile=route_m / 1609.344,
                route_index=route_index,
                point=point,
                geo=route_geo_at_cumulative(analysis, route_m),
                kind=kind,
                action=nav_cue_action(kind, label),
                label=label,
                source_waypoint_name=source_name,
                segment_name=segment_name,
                repeated_place_id=repeated_place_id,
                repeated_pass_number=repeated_pass_number,
                ambiguous=ambiguous,
            )
        )
    return cues, snapped


def navigation_cues(analysis: RouteAnalysis, config: RenderConfig) -> list[NavCue]:
    return build_navigation_cues(analysis, config)[0]


def snapped_waypoints(analysis: RouteAnalysis, config: RenderConfig) -> list[SnappedWaypoint]:
    return build_navigation_cues(analysis, config)[1]


def signed_angle_delta(left: float, right: float) -> float:
    return (right - left + math.pi) % math.tau - math.pi


def turn_action_for_cue(analysis: RouteAnalysis, cue: NavCue) -> tuple[str, float]:
    if cue.kind == "start":
        return "START / CAR", 1.0
    if cue.kind == "finish":
        return "RETURN / FINISH", 1.0
    if len(analysis.segments) < 2:
        return "DECISION", 0.0
    inbound_index = max(0, min(cue.route_index - 1, len(analysis.segments) - 1))
    outbound_index = max(0, min(cue.route_index + 1, len(analysis.segments) - 1))
    inbound = analysis.segments[inbound_index].angle_rad
    outbound = analysis.segments[outbound_index].angle_rad
    degrees = math.degrees(signed_angle_delta(inbound, outbound))
    abs_degrees = abs(degrees)
    direction = "LEFT" if degrees > 0 else "RIGHT"
    if abs_degrees < 25:
        return "GO STRAIGHT", 0.72
    if abs_degrees < 60:
        return f"KEEP {direction}", 0.68
    if abs_degrees < 135:
        return f"TURN {direction}", 0.82
    return "TURN AROUND", 0.7


def turn_action_from_points(previous: MetricPoint | None, current: MetricPoint, next_point: MetricPoint | None, kind: str) -> tuple[str, float]:
    if kind == "start":
        return "START / CAR", 1.0
    if kind == "finish":
        return "RETURN / FINISH", 1.0
    if previous is None or next_point is None:
        return "DECISION", 0.0
    inbound = math.atan2(current.y - previous.y, current.x - previous.x)
    outbound = math.atan2(next_point.y - current.y, next_point.x - current.x)
    degrees = math.degrees(signed_angle_delta(inbound, outbound))
    abs_degrees = abs(degrees)
    direction = "LEFT" if degrees > 0 else "RIGHT"
    if abs_degrees < 22:
        return "GO STRAIGHT", 0.72
    if abs_degrees < 65:
        return f"KEEP {direction}", 0.74
    if abs_degrees < 145:
        return f"TURN {direction}", 0.84
    return "TURN AROUND", 0.72


def cue_elevation_stats(
    analysis: RouteAnalysis,
    start_m: float,
    end_m: float,
) -> tuple[float | None, float | None, float | None]:
    if end_m <= start_m:
        return 0.0, 0.0, 0.0
    net_m = 0.0
    gain_m = 0.0
    loss_m = 0.0
    saw_overlap = False
    for segment in analysis.segments:
        overlap_start = max(start_m, segment.cumulative_start_m)
        overlap_end = min(end_m, segment.cumulative_end_m)
        if overlap_end <= overlap_start:
            continue
        start_geo = analysis.geo_points[max(0, min(segment.index, len(analysis.geo_points) - 1))]
        end_geo = analysis.geo_points[max(0, min(segment.index + 1, len(analysis.geo_points) - 1))]
        if start_geo.ele_m is None or end_geo.ele_m is None:
            return None, None, None
        fraction = (overlap_end - overlap_start) / max(segment.distance_m, 1.0)
        delta_m = (end_geo.ele_m - start_geo.ele_m) * fraction
        net_m += delta_m
        if delta_m >= 0:
            gain_m += delta_m
        else:
            loss_m += abs(delta_m)
        saw_overlap = True
    if not saw_overlap:
        return None, None, None
    meters_to_feet = 3.28084
    return net_m * meters_to_feet, gain_m * meters_to_feet, loss_m * meters_to_feet


def elevation_hint(delta_ft: float | None, gain_ft: float | None, loss_ft: float | None, distance_miles: float) -> str:
    if delta_ft is None or gain_ft is None or loss_ft is None:
        return "ELEV REVIEW"
    if distance_miles <= 0.03:
        return "FLAT"
    if gain_ft > 60 and loss_ft > 60:
        return "ROLLING"
    grade_ft_per_mile = delta_ft / max(distance_miles, 0.05)
    if grade_ft_per_mile >= 350 or delta_ft >= 180:
        return "STEEP UP"
    if grade_ft_per_mile <= -350 or delta_ft <= -180:
        return "STEEP DOWN"
    if delta_ft >= 35:
        return "UP"
    if delta_ft <= -35:
        return "DOWN"
    return "FLAT"


def trail_target_token(label: str) -> str:
    return next((token.strip(".,;:") for token in label.split() if token.startswith("#")), "")


def trail_target_name(label: str) -> str:
    token = trail_target_token(label)
    if not token:
        return label
    return label.replace(token, "", 1).strip(" -:") or token


def field_action_for_cue(cue: NavCue, geometry_action: str) -> str:
    if cue.kind == "start":
        return "START / CAR"
    if cue.kind == "finish":
        return "RETURN / FINISH"
    token = trail_target_token(cue.label)
    if token:
        return f"TAKE {token}"
    if geometry_action == "TURN AROUND":
        return "DECISION"
    return geometry_action


def build_napkin_cues(analysis: RouteAnalysis, config: RenderConfig) -> list[NapkinCue]:
    cues = navigation_cues(analysis, config)
    first_at_place: dict[tuple[int, int], int] = {}
    napkin_cues: list[NapkinCue] = []
    previous_route_m = 0.0
    for index, cue in enumerate(cues):
        previous_cue = cues[index - 1] if index > 0 else None
        next_cue = cues[index + 1] if index < len(cues) - 1 else None
        geometry_action, confidence = turn_action_from_points(
            previous_cue.point if previous_cue else None,
            cue.point,
            next_cue.point if next_cue else None,
            cue.kind,
        )
        if geometry_action == "GO STRAIGHT" and cue.kind not in {"start", "finish"}:
            local_action, local_confidence = turn_action_for_cue(analysis, cue)
            if local_action != "GO STRAIGHT":
                geometry_action, confidence = local_action, min(local_confidence, 0.72)
        action = field_action_for_cue(cue, geometry_action)
        distance_miles = max(0.0, (cue.route_m - previous_route_m) / 1609.344)
        delta_ft, gain_ft, loss_ft = cue_elevation_stats(analysis, previous_route_m, cue.route_m)
        hint = elevation_hint(delta_ft, gain_ft, loss_ft, distance_miles)
        place_key = quantize(cue.point, max(config.node_snap_m * 1.5, 24.0))
        same_as = first_at_place.get(place_key)
        if same_as is None:
            first_at_place[place_key] = cue.cue_number
        reasons: list[str] = []
        if confidence < 0.7 and cue.kind not in {"start", "finish"}:
            reasons.append("low_turn_confidence")
        if cue.ambiguous:
            reasons.append("ambiguous_waypoint_snap")
        if delta_ft is None:
            reasons.append("missing_elevation")
        if same_as is not None:
            reasons.append("repeated_junction")
        if geometry_action == "TURN AROUND" and cue.kind not in {"start", "finish"}:
            reasons.append("geometry_turn_review")
        display_action = action
        if same_as is not None and cue.kind not in {"start", "finish"}:
            display_action = f"SAME JCT / {action}"
        napkin_cues.append(
            NapkinCue(
                cue_number=cue.cue_number,
                route_mile=cue.route_mile,
                distance_from_previous_miles=distance_miles,
                point=cue.point,
                geo=cue.geo,
                action=action,
                display_action=display_action,
                turn_confidence=confidence,
                label=cue.label,
                elevation_delta_ft=delta_ft,
                elevation_gain_ft=gain_ft,
                elevation_loss_ft=loss_ft,
                elevation_hint=hint,
                repeated_place_id=cue.repeated_place_id or (f"P{place_key[0]}_{place_key[1]}" if same_as is not None else None),
                repeated_pass_number=cue.repeated_pass_number,
                same_as_cue_number=same_as,
                needs_review=bool(reasons),
                review_reasons=tuple(sorted(set(reasons))),
            )
        )
        previous_route_m = cue.route_m
    return napkin_cues


def point_near_repeated_node(point: MetricPoint, repeated_nodes: list[RepeatedNode]) -> bool:
    return any(distance(point, node.point) < 80.0 for node in repeated_nodes)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_muted_basemap_png(draw: ImageDraw.ImageDraw, width: int, height: int, style: str) -> None:
    background = "#f7f8f4" if style == "context" else ("#f3f4ef" if style in {"muted", "topo"} else "#ffffff")
    draw.rectangle([0, 0, width, height], fill=background)
    if style in {"none", "context"}:
        return
    grid_color = "#d8ddd3" if style in {"muted", "topo"} else "#eeeeee"
    spacing = 100
    for x in range(0, width + spacing, spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height + spacing, spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    if style == "topo":
        for y in range(60, height, 120):
            points = [(x, y + math.sin(x / 85.0 + y / 120.0) * 16) for x in range(-20, width + 20, 28)]
            draw.line(points, fill="#c8d5c0", width=1)


def svg_muted_basemap(width: int, height: int, style: str) -> str:
    background = "#f7f8f4" if style == "context" else ("#f3f4ef" if style in {"muted", "topo"} else "#ffffff")
    grid_color = "#d8ddd3" if style in {"muted", "topo"} else "#eeeeee"
    items = [f'<rect width="{width}" height="{height}" fill="{background}"/>']
    if style in {"none", "context"}:
        return "\n".join(items)
    for x in range(0, width + 100, 100):
        items.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{grid_color}" stroke-width="1"/>')
    for y in range(0, height + 100, 100):
        items.append(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{grid_color}" stroke-width="1"/>')
    if style == "topo":
        for y in range(60, height, 120):
            points = " ".join(
                f"{x},{y + math.sin(x / 85.0 + y / 120.0) * 16:.1f}" for x in range(-20, width + 20, 28)
            )
            items.append(f'<polyline points="{points}" fill="none" stroke="#c8d5c0" stroke-width="1"/>')
    return "\n".join(items)


def draw_route_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
    include_labels: bool = True,
) -> None:
    line_width = max(1, int(round(transform.length(config.line_width / transform.scale if transform.scale else config.line_width))))
    line_width = max(3, int(round(config.line_width)))
    casing = line_width + 4
    for segment in analysis.segments:
        start, end = offset_points(segment)
        p1 = transform.point(start)
        p2 = transform.point(end)
        draw.line([p1, p2], fill="#ffffff", width=casing)
    for segment in analysis.segments:
        start, end = offset_points(segment)
        p1 = transform.point(start)
        p2 = transform.point(end)
        color = interpolate_color(config.palette, route_progress(segment, analysis.total_distance_m))
        draw.line([p1, p2], fill=color, width=line_width)

    for segment in analysis.segments:
        dense = point_near_repeated_node(segment.midpoint, analysis.repeated_nodes)
        for fraction in arrow_fractions(segment, config, dense):
            point = shifted_point_on_segment(segment, fraction)
            color = interpolate_color(config.palette, route_progress(segment, analysis.total_distance_m, fraction))
            draw_arrow_png(draw, transform, point, segment.angle_rad, color, size_px=max(8.0, config.line_width * 1.4))

    if include_labels:
        draw_markers_png(draw, analysis, transform, config)


def draw_field_route_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
    include_labels: bool = True,
) -> None:
    line_width = max(5, int(round(config.line_width)))
    casing = line_width + 6
    route_points = [transform.point(point) for point in field_ribbon_points(analysis, config)]
    if len(route_points) >= 2:
        draw.line(route_points, fill="#ffffff", width=casing, joint="curve")
    for points, color in field_ribbon_parts(analysis, config):
        screen_points = [transform.point(point) for point in points]
        if len(screen_points) >= 2:
            draw.line(screen_points, fill=color, width=line_width, joint="curve")

    for cumulative in field_arrow_targets(analysis, config, detail=config.profile == "detail"):
        point, angle = point_at_cumulative(analysis, cumulative, apply_lane_offset=False)
        draw_chevron_png(draw, transform, point, angle, "#111827", size_px=max(10.0, config.line_width * 1.55))

    if include_labels:
        draw_field_markers_png(draw, analysis, transform, config, navigation_cues(analysis, config))


def line_intersects_bbox(points: tuple[MetricPoint, ...], bbox: tuple[float, float, float, float]) -> bool:
    min_x, min_y, max_x, max_y = bbox
    return any(min_x <= point.x <= max_x and min_y <= point.y <= max_y for point in points)


def draw_context_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    bbox: tuple[float, float, float, float],
) -> None:
    for line in analysis.context_lines:
        if not line_intersects_bbox(line.points, bbox):
            continue
        draw.line([transform.point(point) for point in line.points], fill="#aab5a6", width=2)


def label_box_overlaps(box: tuple[float, float, float, float], boxes: list[tuple[float, float, float, float]]) -> bool:
    left, top, right, bottom = box
    for other_left, other_top, other_right, other_bottom in boxes:
        if right < other_left or left > other_right or bottom < other_top or top > other_bottom:
            continue
        return True
    return False


def draw_label_png(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    used_boxes: list[tuple[float, float, float, float]],
    fill: str = "#111827",
    outline: str = "#ffffff",
    size: int = 12,
) -> bool:
    label_font = font(size=size, bold=True)
    bbox = draw.textbbox((x, y), text, font=label_font)
    padded = (bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3)
    if label_box_overlaps(padded, used_boxes):
        return False
    draw.rounded_rectangle(padded, radius=4, fill=outline, outline="#cbd5e1")
    draw.text((x, y), text, font=label_font, fill=fill)
    used_boxes.append(padded)
    return True


def draw_candidate_label_png(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor_x: float,
    anchor_y: float,
    used_boxes: list[tuple[float, float, float, float]],
    fill: str = "#111827",
    size: int = 11,
) -> bool:
    offsets = [
        (16, -10), (16, 6), (16, -26), (-16, -10), (-16, 6), (-16, -26), (0, 18), (0, -30)
    ]
    label_font = font(size=size, bold=True)
    for dx, dy in offsets:
        x = anchor_x + dx
        y = anchor_y + dy
        bbox = draw.textbbox((x, y), text, font=label_font)
        if dx < 0:
            width = bbox[2] - bbox[0]
            x = anchor_x + dx - width
            bbox = draw.textbbox((x, y), text, font=label_font)
        padded = (bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3)
        if label_box_overlaps(padded, used_boxes):
            continue
        draw.line([(anchor_x, anchor_y), (x, y + (bbox[3] - bbox[1]) / 2)], fill="#64748b", width=1)
        draw.rounded_rectangle(padded, radius=4, fill="#ffffff", outline="#cbd5e1")
        draw.text((x, y), text, font=label_font, fill=fill)
        used_boxes.append(padded)
        return True
    return False


def draw_markers_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
) -> None:
    used_boxes: list[tuple[float, float, float, float]] = []
    start = analysis.points[0]
    finish = analysis.points[-1]
    for label, point, color in [("S", start, "#16a34a"), ("F", finish, "#dc2626")]:
        x, y = transform.point(point)
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=color, outline="#ffffff", width=3)
        draw.text((x - 4, y - 7), label, fill="#ffffff", font=font(13, bold=True))
        draw_label_png(draw, "Start" if label == "S" else "Finish", x + 14, y - 9, used_boxes, fill=color)

    for node in analysis.repeated_nodes[: config.max_repeated_node_labels]:
        base_x, base_y = transform.point(node.point)
        for pass_number, cumulative in enumerate(node.cumulative_m[:5], start=1):
            angle = (pass_number - 1) * math.tau / max(len(node.cumulative_m[:5]), 1)
            x = base_x + math.cos(angle) * 22
            y = base_y + math.sin(angle) * 22
            color = interpolate_color(config.palette, cumulative / max(analysis.total_distance_m, 1))
            draw.ellipse([x - 9, y - 9, x + 9, y + 9], fill=color, outline="#ffffff", width=2)
            draw.text((x - 4, y - 6), str(pass_number), fill="#ffffff", font=font(11, bold=True))

    labeled_waypoints = important_waypoints(analysis.waypoints)[: config.max_waypoint_labels]
    for waypoint in labeled_waypoints:
        if not waypoint.metric:
            continue
        x, y = transform.point(waypoint.metric)
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#111827", outline="#ffffff", width=2)
        draw_label_png(draw, waypoint.name[:32], x + 8, y - 8, used_boxes, size=11)


def draw_field_markers_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
    cues: list[NavCue],
) -> None:
    if config.label_mode == "segment-waypoints":
        draw_legacy_waypoint_markers_png(draw, analysis, transform, config)
        return
    used_boxes: list[tuple[float, float, float, float]] = []
    offsets = cue_screen_offsets(cues[: config.max_waypoint_labels], transform)
    for cue in cues[: config.max_waypoint_labels]:
        base_x, base_y = transform.point(cue.point)
        dx, dy = offsets.get(cue.cue_number, (0.0, 0.0))
        x, y = base_x + dx, base_y + dy
        number = marker_number(cue)
        fill = "#111827"
        if cue.kind == "start":
            fill = "#16a34a"
        elif cue.kind == "finish":
            fill = "#dc2626"
        radius = 13 if cue.kind in {"start", "finish"} else 11
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline="#ffffff", width=3)
        text_x = x - (4 if len(number) == 1 else 7)
        draw.text((text_x, y - 7), number, fill="#ffffff", font=font(12, bold=True))
        label = cue.label
        if config.show_cue_labels:
            if label and cue.kind not in {"start", "finish"}:
                draw_candidate_label_png(draw, label, x, y, used_boxes, fill="#111827", size=11)
            elif cue.kind in {"start", "finish"}:
                draw_candidate_label_png(draw, label, x, y, used_boxes, fill=fill, size=11)
    if not cues:
        for label, point, color in [("P", analysis.points[0], "#16a34a"), ("F", analysis.points[-1], "#dc2626")]:
            x, y = transform.point(point)
            draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=color, outline="#ffffff", width=3)
            draw.text((x - 4, y - 7), label, fill="#ffffff", font=font(13, bold=True))
    if config.show_segment_labels:
        for snapped in [item for item in snapped_waypoints(analysis, config) if is_segment_waypoint_name(item.waypoint.name)][: config.max_waypoint_labels]:
            x, y = transform.point(snapped.selected.point)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="#64748b", outline="#ffffff", width=1)
            label = snapped.segment_name or snapped.waypoint.name
            draw_candidate_label_png(draw, label[:28], x, y, used_boxes, fill="#64748b", size=10)
    if config.show_parking_marker:
        draw_parking_marker_png(draw, analysis, transform, cues)


def draw_legacy_waypoint_markers_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
) -> None:
    used_boxes: list[tuple[float, float, float, float]] = []
    for cue_index, position in enumerate(waypoint_route_positions(analysis)[: config.max_waypoint_labels], start=1):
        x, y = transform.point(position.point)
        number = legacy_waypoint_marker_number(position.waypoint.name, cue_index)
        fill = "#111827" if number not in {"P", "F"} else ("#16a34a" if number == "P" else "#dc2626")
        radius = 13 if number in {"P", "F"} else 11
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline="#ffffff", width=3)
        text_x = x - (4 if len(number) == 1 else 7)
        draw.text((text_x, y - 7), number, fill="#ffffff", font=font(12, bold=True))
        label = cue_label(position.waypoint.name)
        if label and number not in {"P", "F"}:
            draw_candidate_label_png(draw, label, x, y, used_boxes, fill="#111827", size=11)


def cue_screen_offsets(cues: list[NavCue], transform: Transform) -> dict[int, tuple[float, float]]:
    groups: list[list[NavCue]] = []
    anchors: list[tuple[float, float]] = []
    for cue in cues:
        x, y = transform.point(cue.point)
        target_index: int | None = None
        for index, (anchor_x, anchor_y) in enumerate(anchors):
            if math.hypot(x - anchor_x, y - anchor_y) <= 24:
                target_index = index
                break
        if target_index is None:
            groups.append([cue])
            anchors.append((x, y))
        else:
            groups[target_index].append(cue)
    offsets: dict[int, tuple[float, float]] = {}
    for group in groups:
        if len(group) == 1:
            offsets[group[0].cue_number] = (0.0, 0.0)
            continue
        if len(group) == 2:
            pair_offsets = [(-14.0, -10.0), (14.0, 10.0)]
        else:
            radius = 20.0
            pair_offsets = [
                (math.cos(index * math.tau / len(group)) * radius, math.sin(index * math.tau / len(group)) * radius)
                for index in range(len(group))
            ]
        for cue, offset in zip(group, pair_offsets):
            offsets[cue.cue_number] = offset
    return offsets


def start_cue_point(cues: list[NavCue], analysis: RouteAnalysis) -> MetricPoint:
    start = next((cue for cue in cues if cue.kind == "start"), None)
    return start.point if start else analysis.points[0]


def draw_parking_marker_png(
    draw: ImageDraw.ImageDraw,
    analysis: RouteAnalysis,
    transform: Transform,
    cues: list[NavCue],
) -> None:
    x, y = transform.point(start_cue_point(cues, analysis))
    x += 2
    y += 34
    draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill="#111827", outline="#ffffff", width=3)
    draw.text((x - 5, y - 8), "P", fill="#ffffff", font=font(15, bold=True))


def important_waypoints(waypoints: list[Waypoint]) -> list[Waypoint]:
    priority_terms = ("PARK", "START", "FINISH", "RETURN", "WATER", "AID", "CHECKPOINT", "CUE")
    ranked = []
    for waypoint in waypoints:
        name = waypoint.name.upper()
        score = 0
        for index, term in enumerate(priority_terms):
            if term in name:
                score = len(priority_terms) - index
                break
        ranked.append((score, waypoint.name, waypoint))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked if item[0] > 0]


def render_png_scene(
    analysis: RouteAnalysis,
    config: RenderConfig,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    include_labels: bool = True,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw_muted_basemap_png(draw, width, height, config.basemap_style)
    transform = Transform(bbox, (18, 18, width - 36, height - 36))
    draw_context_png(draw, analysis, transform, bbox)
    if config.render_mode == "audit":
        draw_route_png(draw, analysis, transform, config, include_labels=include_labels)
    else:
        draw_field_route_png(draw, analysis, transform, config, include_labels=include_labels)
    return image


def draw_inset_previews_png(
    image: Image.Image,
    analysis: RouteAnalysis,
    config: RenderConfig,
) -> None:
    if not analysis.dense_areas:
        return
    draw = ImageDraw.Draw(image)
    margin = 18
    gap = 14
    size = min(config.inset_size, max(160, (image.height - margin * 2) // max(len(analysis.dense_areas), 1) - gap))
    for index, area in enumerate(analysis.dense_areas):
        x = image.width - size - margin
        y = margin + index * (size + gap)
        inset = render_png_scene(analysis, config, area.bbox, size, size, include_labels=True)
        image.paste(inset, (x, y))
        draw.rectangle([x, y, x + size, y + size], outline="#111827", width=2)
        draw.rounded_rectangle([x + 8, y + 8, x + 82, y + 30], radius=4, fill="#111827")
        draw.text((x + 14, y + 12), f"Inset {index + 1}", fill="#ffffff", font=font(12, bold=True))


def detail_letter(index: int) -> str:
    return chr(ord("A") + index)


def draw_detail_boxes_png(image: Image.Image, analysis: RouteAnalysis) -> None:
    if not analysis.dense_areas:
        return
    draw = ImageDraw.Draw(image)
    transform = Transform(analysis.bbox, (18, 18, image.width - 36, image.height - 36))
    used_boxes: list[tuple[float, float, float, float]] = []
    for index, area in enumerate(analysis.dense_areas):
        p1 = transform.point(MetricPoint(area.bbox[0], area.bbox[1]))
        p2 = transform.point(MetricPoint(area.bbox[2], area.bbox[3]))
        left, right = sorted((p1[0], p2[0]))
        top, bottom = sorted((p1[1], p2[1]))
        draw.rectangle([left, top, right, bottom], outline="#334155", width=1)
        draw_candidate_label_png(draw, f"Detail {detail_letter(index)}", left, top, used_boxes, size=12)


def render_png(analysis: RouteAnalysis, config: RenderConfig, output_path: Path) -> None:
    image = render_png_scene(analysis, config, analysis.bbox, config.width, config.height, include_labels=True)
    if config.embed_insets:
        draw_inset_previews_png(image, analysis, config)
    else:
        draw_detail_boxes_png(image, analysis)
    image.save(output_path, dpi=(config.dpi, config.dpi))


def svg_scene_elements(
    analysis: RouteAnalysis,
    config: RenderConfig,
    bbox: tuple[float, float, float, float],
    viewport: tuple[float, float, float, float],
    clip_id: str | None = None,
    include_labels: bool = True,
) -> list[str]:
    transform = Transform(bbox, viewport)
    attrs = f' clip-path="url(#{clip_id})"' if clip_id else ""
    elements = [f'<g class="route-scene"{attrs}>']
    line_width = max(config.line_width, 3)
    for segment in analysis.segments:
        start, end = offset_points(segment)
        x1, y1 = transform.point(start)
        x2, y2 = transform.point(end)
        lane_class = " repeated-edge-lane" if segment.repeated else ""
        elements.append(
            f'<line class="route-casing{lane_class}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#ffffff" stroke-width="{line_width + 4:.1f}" stroke-linecap="round"/>'
        )
    for segment in analysis.segments:
        start, end = offset_points(segment)
        x1, y1 = transform.point(start)
        x2, y2 = transform.point(end)
        color = interpolate_color(config.palette, route_progress(segment, analysis.total_distance_m))
        lane_class = " repeated-edge-lane" if segment.repeated else ""
        elements.append(
            f'<line class="route-segment{lane_class}" data-order="{segment.index}" data-lane="{segment.lane_index + 1}" '
            f'data-lanes="{segment.lane_count}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{line_width:.1f}" stroke-linecap="round"/>'
        )
    for segment in analysis.segments:
        dense = point_near_repeated_node(segment.midpoint, analysis.repeated_nodes)
        for fraction in arrow_fractions(segment, config, dense):
            point = shifted_point_on_segment(segment, fraction)
            color = interpolate_color(config.palette, route_progress(segment, analysis.total_distance_m, fraction))
            polygon = svg_arrow_polygon(transform, point, segment.angle_rad, max(8.0, config.line_width * 1.4))
            elements.append(
                f'<polygon class="direction-arrow" points="{polygon}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )
    if include_labels:
        elements.extend(svg_marker_elements(analysis, transform, config))
    elements.append("</g>")
    return elements


def svg_field_scene_elements(
    analysis: RouteAnalysis,
    config: RenderConfig,
    bbox: tuple[float, float, float, float],
    viewport: tuple[float, float, float, float],
    clip_id: str | None = None,
    include_labels: bool = True,
) -> list[str]:
    transform = Transform(bbox, viewport)
    attrs = f' clip-path="url(#{clip_id})"' if clip_id else ""
    elements = [f'<g class="field-route-scene"{attrs}>']
    line_width = max(config.line_width, 5)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.point(point) for point in field_ribbon_points(analysis, config)))
    elements.append(
        f'<polyline class="route-casing field-route-casing" points="{points}" fill="none" stroke="#ffffff" '
        f'stroke-width="{line_width + 6:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for part_index, (part_points, color) in enumerate(field_ribbon_parts(analysis, config), start=1):
        part = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.point(point) for point in part_points))
        elements.append(
            f'<polyline class="route-segment field-route" data-leg="{part_index}" points="{part}" fill="none" '
            f'stroke="{color}" stroke-width="{line_width:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    for cumulative in field_arrow_targets(analysis, config, detail=config.profile == "detail"):
        point, angle = point_at_cumulative(analysis, cumulative, apply_lane_offset=False)
        polyline = svg_chevron_polyline(transform, point, angle, max(10.0, config.line_width * 1.55))
        elements.append(
            f'<polyline class="direction-arrow field-direction-arrow" points="{polyline}" fill="none" stroke="#ffffff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<polyline class="direction-arrow field-direction-arrow" points="{polyline}" fill="none" stroke="#111827" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    if include_labels:
        elements.extend(svg_field_marker_elements(analysis, transform, config, navigation_cues(analysis, config)))
    elements.append("</g>")
    return elements


def svg_context_elements(
    analysis: RouteAnalysis,
    transform: Transform,
    bbox: tuple[float, float, float, float],
    clip_id: str | None = None,
) -> list[str]:
    attrs = f' clip-path="url(#{clip_id})"' if clip_id else ""
    elements = [f'<g class="context-layer"{attrs}>']
    for line in analysis.context_lines:
        if not line_intersects_bbox(line.points, bbox):
            continue
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.point(point) for point in line.points))
        elements.append(
            f'<polyline class="context-line" data-name="{escape_xml(line.name)}" points="{points}" '
            f'fill="none" stroke="#aab5a6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity=".72"/>'
        )
    elements.append("</g>")
    return elements


def svg_scene_for_mode(
    analysis: RouteAnalysis,
    config: RenderConfig,
    bbox: tuple[float, float, float, float],
    viewport: tuple[float, float, float, float],
    clip_id: str | None = None,
    include_labels: bool = True,
) -> list[str]:
    if config.render_mode == "audit":
        return svg_scene_elements(analysis, config, bbox, viewport, clip_id=clip_id, include_labels=include_labels)
    return svg_field_scene_elements(analysis, config, bbox, viewport, clip_id=clip_id, include_labels=include_labels)


def svg_marker_elements(analysis: RouteAnalysis, transform: Transform, config: RenderConfig) -> list[str]:
    elements: list[str] = []
    for label, point, color, text in [
        ("S", analysis.points[0], "#16a34a", "Start"),
        ("F", analysis.points[-1], "#dc2626", "Finish"),
    ]:
        x, y = transform.point(point)
        elements.append(
            f'<g class="{"start" if label == "S" else "finish"}-marker">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" stroke="#ffffff" stroke-width="3"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">{label}</text>'
            f'<text x="{x + 16:.1f}" y="{y - 10:.1f}" class="marker-label" fill="{color}">{escape_xml(text)}</text>'
            f"</g>"
        )
    for node_index, node in enumerate(analysis.repeated_nodes[: config.max_repeated_node_labels], start=1):
        base_x, base_y = transform.point(node.point)
        elements.append(f'<g class="repeated-intersection" data-node="{node_index}">')
        for pass_number, cumulative in enumerate(node.cumulative_m[:5], start=1):
            angle = (pass_number - 1) * math.tau / max(len(node.cumulative_m[:5]), 1)
            x = base_x + math.cos(angle) * 22
            y = base_y + math.sin(angle) * 22
            color = interpolate_color(config.palette, cumulative / max(analysis.total_distance_m, 1))
            elements.append(
                f'<g class="pass-order-label" data-pass="{pass_number}">'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" font-weight="800" fill="#ffffff">{pass_number}</text>'
                f'</g>'
            )
        elements.append("</g>")
    for waypoint in important_waypoints(analysis.waypoints)[: config.max_waypoint_labels]:
        if not waypoint.metric:
            continue
        x, y = transform.point(waypoint.metric)
        elements.append(
            f'<g class="waypoint-label">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#111827" stroke="#ffffff" stroke-width="2"/>'
            f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" class="marker-label">{escape_xml(waypoint.name[:32])}</text>'
            f'</g>'
        )
    return elements


def svg_field_marker_elements(
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
    cues: list[NavCue],
) -> list[str]:
    if config.label_mode == "segment-waypoints":
        return svg_legacy_waypoint_marker_elements(analysis, transform, config)
    elements: list[str] = []
    offsets = cue_screen_offsets(cues[: config.max_waypoint_labels], transform)
    for cue in cues[: config.max_waypoint_labels]:
        base_x, base_y = transform.point(cue.point)
        dx, dy = offsets.get(cue.cue_number, (0.0, 0.0))
        x, y = base_x + dx, base_y + dy
        number = marker_number(cue)
        fill = "#111827"
        if cue.kind == "start":
            fill = "#16a34a"
        elif cue.kind == "finish":
            fill = "#dc2626"
        radius = 13 if cue.kind in {"start", "finish"} else 11
        label = cue.label
        elements.append(
            f'<g class="cue-step-label" data-cue="{escape_xml(number)}" data-kind="{escape_xml(cue.kind)}" '
            f'data-name="{escape_xml(label)}" data-source-waypoint="{escape_xml(cue.source_waypoint_name or "")}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="#ffffff" stroke-width="3"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">{escape_xml(number)}</text>'
        )
        if label and config.show_cue_labels:
            elements.append(f'<text x="{x + 15:.1f}" y="{y - 8:.1f}" class="marker-label">{escape_xml(label)}</text>')
        elements.append("</g>")
    if not cues:
        for label, point, color, text in [
            ("P", analysis.points[0], "#16a34a", "Park/start"),
            ("F", analysis.points[-1], "#dc2626", "Finish"),
        ]:
            x, y = transform.point(point)
            elements.append(
                f'<g class="cue-step-label">'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" stroke="#ffffff" stroke-width="3"/>'
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">{label}</text>'
                f'<text x="{x + 16:.1f}" y="{y - 10:.1f}" class="marker-label" fill="{color}">{escape_xml(text)}</text>'
                f"</g>"
            )
    if config.show_segment_labels:
        for snapped in [item for item in snapped_waypoints(analysis, config) if is_segment_waypoint_name(item.waypoint.name)][: config.max_waypoint_labels]:
            x, y = transform.point(snapped.selected.point)
            label = snapped.segment_name or snapped.waypoint.name
            elements.append(
                f'<g class="segment-secondary-label" data-source-waypoint="{escape_xml(snapped.waypoint.name)}">'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#64748b" stroke="#ffffff" stroke-width="1"/>'
                f'<text x="{x + 7:.1f}" y="{y - 6:.1f}" class="marker-label" fill="#64748b">{escape_xml(label[:28])}</text>'
                f'</g>'
            )
    if config.show_parking_marker:
        elements.extend(svg_parking_marker_elements(analysis, transform, cues))
    return elements


def svg_legacy_waypoint_marker_elements(
    analysis: RouteAnalysis,
    transform: Transform,
    config: RenderConfig,
) -> list[str]:
    elements: list[str] = []
    for cue_index, position in enumerate(waypoint_route_positions(analysis)[: config.max_waypoint_labels], start=1):
        x, y = transform.point(position.point)
        number = legacy_waypoint_marker_number(position.waypoint.name, cue_index)
        fill = "#111827" if number not in {"P", "F"} else ("#16a34a" if number == "P" else "#dc2626")
        radius = 13 if number in {"P", "F"} else 11
        label = cue_label(position.waypoint.name)
        elements.append(
            f'<g class="cue-step-label legacy-waypoint-label" data-cue="{escape_xml(number)}" data-name="{escape_xml(position.waypoint.name)}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="#ffffff" stroke-width="3"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">{escape_xml(number)}</text>'
        )
        if label and number not in {"P", "F"}:
            elements.append(f'<text x="{x + 15:.1f}" y="{y - 8:.1f}" class="marker-label">{escape_xml(label)}</text>')
        elements.append("</g>")
    return elements


def svg_parking_marker_elements(
    analysis: RouteAnalysis,
    transform: Transform,
    cues: list[NavCue],
) -> list[str]:
    x, y = transform.point(start_cue_point(cues, analysis))
    x += 2
    y += 34
    return [
        '<g class="parking-marker">',
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="#111827" stroke="#ffffff" stroke-width="3"/>',
        f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="15" font-weight="900" fill="#ffffff">P</text>',
        "</g>",
    ]


def napkin_bbox(analysis: RouteAnalysis) -> tuple[float, float, float, float]:
    return expand_bbox(route_bbox(analysis.points), 110.0)


def draw_napkin_header_png(draw: ImageDraw.ImageDraw, analysis: RouteAnalysis, width: int) -> None:
    title = analysis.route_name
    draw.text((42, 28), title[:42], fill="#111827", font=font(34, bold=True))
    draw.text((42, 76), "Schematic field guide - not to scale. Follow cue order.", fill="#475569", font=font(18))


def action_chip_text(cue: NapkinCue) -> str:
    if cue.cue_number == 1:
        return "START / CAR"
    if cue.action == "RETURN / FINISH":
        return "RETURN / FINISH"
    action = cue.display_action.replace("TURN LEFT", "LEFT").replace("TURN RIGHT", "RIGHT")
    action = action.replace("KEEP LEFT", "KEEP L").replace("KEEP RIGHT", "KEEP R")
    action = action.replace("GO STRAIGHT", "STRAIGHT")
    action = action.replace("SAME JCT / ", "SAME JCT · ")
    hint = "" if cue.elevation_hint in {"FLAT", "ELEV REVIEW"} else f" · {cue.elevation_hint}"
    trail_token = next((token for token in cue.label.split() if token.startswith("#")), "")
    trail = f" · {trail_token}" if trail_token and trail_token not in action else ""
    return f"{action}{hint}{trail}"


def napkin_transform(bbox: tuple[float, float, float, float], viewport: tuple[float, float, float, float]) -> Transform:
    transform = Transform(bbox, viewport)
    transform.pad_y = min(transform.pad_y, 24.0)
    return transform


def draw_napkin_map_png(analysis: RouteAnalysis, config: RenderConfig, output_path: Path) -> None:
    width, height = config.width, config.height
    image = Image.new("RGB", (width, height), "#fbfaf4")
    draw = ImageDraw.Draw(image)
    draw_napkin_header_png(draw, analysis, width)
    bbox = napkin_bbox(analysis)
    transform = napkin_transform(bbox, (54, 140, width - 108, height - 220))
    draw_context_png(draw, analysis, transform, bbox)

    ribbon = simplify_metric_points(analysis.points, max(config.simplify_tolerance_m, 22.0), {0, len(analysis.points) - 1})
    route_points = [transform.point(point) for point in ribbon]
    if len(route_points) >= 2:
        draw.line(route_points, fill="#ffffff", width=int(config.line_width + 9), joint="curve")
        draw.line(route_points, fill=config.field_route_color, width=int(config.line_width), joint="curve")
    for cumulative in field_arrow_targets(analysis, config):
        point, angle = point_at_cumulative(analysis, cumulative, apply_lane_offset=False)
        draw_chevron_png(draw, transform, point, angle, "#111827", 13)

    cues = build_napkin_cues(analysis, config)
    nav_cues = navigation_cues(analysis, config)
    cue_lookup = {cue.cue_number: cue for cue in nav_cues}
    offsets = cue_screen_offsets(nav_cues[: config.max_waypoint_labels], transform)
    used_boxes: list[tuple[float, float, float, float]] = []
    for cue in cues[: config.max_waypoint_labels]:
        base_x, base_y = transform.point(cue.point)
        dx, dy = offsets.get(cue.cue_number, (0.0, 0.0))
        x, y = base_x + dx, base_y + dy
        fill = "#111827"
        if cue.cue_number == 1:
            fill = "#16a34a"
        elif cue.action == "RETURN / FINISH":
            fill = "#dc2626"
        radius = 17 if cue.cue_number in {1, len(cues)} else 15
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline="#ffffff", width=4)
        draw.text((x - (5 if cue.cue_number < 10 else 10), y - 9), str(cue.cue_number), fill="#ffffff", font=font(16, bold=True))
        chip = action_chip_text(cue)
        if cue.cue_number in {1, len(cues)} or cue.needs_review or cue.cue_number <= 8:
            draw_candidate_label_png(draw, chip, x, y, used_boxes, fill=fill if cue.cue_number in {1, len(cues)} else "#111827", size=13)
    draw_parking_marker_png(draw, analysis, transform, nav_cues)
    image.save(output_path, dpi=(config.dpi, config.dpi))


def svg_napkin_map(analysis: RouteAnalysis, config: RenderConfig, output_path: Path) -> None:
    width, height = config.width, config.height
    bbox = napkin_bbox(analysis)
    transform = napkin_transform(bbox, (54, 140, width - 108, height - 220))
    ribbon = simplify_metric_points(analysis.points, max(config.simplify_tolerance_m, 22.0), {0, len(analysis.points) - 1})
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.point(point) for point in ribbon))
    cues = build_napkin_cues(analysis, config)
    nav_cues = navigation_cues(analysis, config)
    offsets = cue_screen_offsets(nav_cues[: config.max_waypoint_labels], transform)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{escape_xml(analysis.route_name)} napkin field guide</title>",
        '<rect width="100%" height="100%" fill="#fbfaf4"/>',
        f'<text x="42" y="60" font-size="34" font-weight="900" fill="#111827">{escape_xml(analysis.route_name[:42])}</text>',
        '<text x="42" y="94" font-size="18" fill="#475569">Schematic field guide - not to scale. Follow cue order.</text>',
    ]
    elements.extend(svg_context_elements(analysis, transform, bbox))
    elements.append(
        f'<polyline class="napkin-route-halo" points="{points}" fill="none" stroke="#ffffff" stroke-width="{config.line_width + 9:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    elements.append(
        f'<polyline class="napkin-route" points="{points}" fill="none" stroke="{config.field_route_color}" stroke-width="{config.line_width:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for cumulative in field_arrow_targets(analysis, config):
        point, angle = point_at_cumulative(analysis, cumulative, apply_lane_offset=False)
        polyline = svg_chevron_polyline(transform, point, angle, 13)
        elements.append(f'<polyline class="napkin-direction-arrow" points="{polyline}" fill="none" stroke="#111827" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    for cue in cues[: config.max_waypoint_labels]:
        base_x, base_y = transform.point(cue.point)
        dx, dy = offsets.get(cue.cue_number, (0.0, 0.0))
        x, y = base_x + dx, base_y + dy
        fill = "#111827"
        if cue.cue_number == 1:
            fill = "#16a34a"
        elif cue.action == "RETURN / FINISH":
            fill = "#dc2626"
        chip = action_chip_text(cue)
        elements.append(
            f'<g class="napkin-cue" data-cue="{cue.cue_number}" data-action="{escape_xml(cue.display_action)}" data-elevation="{escape_xml(cue.elevation_hint)}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{fill}" stroke="#ffffff" stroke-width="4"/>'
            f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="16" font-weight="900" fill="#ffffff">{cue.cue_number}</text>'
            f'<text x="{x + 22:.1f}" y="{y - 8:.1f}" class="napkin-label" font-size="13" font-weight="800" fill="{fill}" paint-order="stroke" stroke="#ffffff" stroke-width="4">{escape_xml(chip)}</text>'
            '</g>'
        )
    elements.extend(svg_parking_marker_elements(analysis, transform, nav_cues))
    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def svg_detail_boxes(analysis: RouteAnalysis, width: int, height: int) -> list[str]:
    if not analysis.dense_areas:
        return []
    transform = Transform(analysis.bbox, (18, 18, width - 36, height - 36))
    elements: list[str] = ['<g class="detail-panel-boxes">']
    for index, area in enumerate(analysis.dense_areas):
        p1 = transform.point(MetricPoint(area.bbox[0], area.bbox[1]))
        p2 = transform.point(MetricPoint(area.bbox[2], area.bbox[3]))
        left, right = sorted((p1[0], p2[0]))
        top, bottom = sorted((p1[1], p2[1]))
        label = f"Detail {detail_letter(index)}"
        elements.append(
            f'<rect class="detail-box" x="{left:.1f}" y="{top:.1f}" width="{right - left:.1f}" height="{bottom - top:.1f}" '
            f'fill="none" stroke="#334155" stroke-width="1"/>'
        )
        elements.append(
            f'<text class="detail-box-label" x="{left + 6:.1f}" y="{top - 6:.1f}" font-size="12" '
            f'font-weight="800" fill="#111827" paint-order="stroke" stroke="#ffffff" stroke-width="4">{label}</text>'
        )
    elements.append("</g>")
    return elements


def escape_xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(analysis: RouteAnalysis, config: RenderConfig, output_path: Path) -> None:
    width = config.width
    height = config.height
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape_xml(analysis.route_name)} route navigation map</title>",
        f"<desc>Topology-aware route map with continuous progress color, arrows, repeated-edge lanes, repeated-intersection pass labels, and start/finish markers.</desc>",
        "<style>",
        ".marker-label{font:700 12px system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round;fill:#111827}",
        ".route-segment{mix-blend-mode:multiply}.repeated-edge-lane{stroke-linecap:round}.direction-arrow{opacity:.96}.inset-label{font:800 12px system-ui,sans-serif;fill:#fff}",
        "</style>",
        svg_muted_basemap(width, height, config.basemap_style),
    ]
    overview_transform = Transform(analysis.bbox, (18, 18, width - 36, height - 36))
    elements.extend(svg_context_elements(analysis, overview_transform, analysis.bbox))
    elements.extend(svg_scene_for_mode(analysis, config, analysis.bbox, (18, 18, width - 36, height - 36)))
    if analysis.dense_areas and config.embed_insets:
        elements.append("<defs>")
        for index, _area in enumerate(analysis.dense_areas, start=1):
            size = min(config.inset_size, max(160, (height - 36) // max(len(analysis.dense_areas), 1) - 14))
            x = width - size - 18
            y = 18 + (index - 1) * (size + 14)
            elements.append(f'<clipPath id="inset-clip-{index}"><rect x="{x}" y="{y}" width="{size}" height="{size}" rx="4"/></clipPath>')
        elements.append("</defs>")
        for index, area in enumerate(analysis.dense_areas, start=1):
            size = min(config.inset_size, max(160, (height - 36) // max(len(analysis.dense_areas), 1) - 14))
            x = width - size - 18
            y = 18 + (index - 1) * (size + 14)
            elements.append(f'<rect class="inset-frame" x="{x}" y="{y}" width="{size}" height="{size}" rx="4" fill="#f8fafc" stroke="#111827" stroke-width="2"/>')
            inset_transform = Transform(area.bbox, (x, y, size, size))
            elements.extend(svg_context_elements(analysis, inset_transform, area.bbox, clip_id=f"inset-clip-{index}"))
            elements.extend(svg_scene_for_mode(analysis, config, area.bbox, (x, y, size, size), clip_id=f"inset-clip-{index}", include_labels=True))
            elements.append(f'<rect x="{x + 8}" y="{y + 8}" width="74" height="22" rx="4" fill="#111827"/>')
            elements.append(f'<text class="inset-label" x="{x + 15}" y="{y + 23}">Inset {index}</text>')
    elif analysis.dense_areas:
        elements.extend(svg_detail_boxes(analysis, width, height))
    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_inset_outputs(analysis: RouteAnalysis, config: RenderConfig, output_dir: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, area in enumerate(analysis.dense_areas, start=1):
        letter = detail_letter(index - 1).lower()
        png_path = output_dir / f"detail-{letter}.png"
        svg_path = output_dir / f"detail-{letter}.svg"
        detail_config = RenderConfig(
            width=config.inset_size,
            height=config.inset_size,
            dpi=config.dpi,
            line_width=config.line_width + 2,
            arrow_spacing_m=max(config.arrow_spacing_m * 0.6, 120.0),
            overlap_offset_m=config.overlap_offset_m,
            min_repeated_stretch_m=config.min_repeated_stretch_m,
            max_lanes=config.max_lanes,
            edge_snap_m=config.edge_snap_m,
            node_snap_m=config.node_snap_m,
            label_density="high",
            basemap_style=config.basemap_style,
            inset_mode="off",
            embed_insets=False,
            max_insets=0,
            inset_size=config.inset_size,
            render_mode=config.render_mode,
            profile="detail",
            field_route_color=config.field_route_color,
            route_color_mode=config.route_color_mode,
            max_arrows=max(config.max_arrows, 90),
            max_pass_labels_overview=config.max_pass_labels_overview,
            max_pass_labels_inset=config.max_pass_labels_inset,
            simplify_tolerance_m=max(config.simplify_tolerance_m * 0.6, 4.0),
            label_mode=config.label_mode,
            show_cue_labels=config.show_cue_labels,
            show_segment_labels=config.show_segment_labels,
            show_parking_marker=config.show_parking_marker,
            waypoint_snap_tolerance_m=config.waypoint_snap_tolerance_m,
            palette=config.palette,
            marker_json=config.marker_json,
            context_geojson=config.context_geojson,
        )
        image = render_png_scene(analysis, detail_config, area.bbox, config.inset_size, config.inset_size, include_labels=True)
        image.save(png_path, dpi=(config.dpi, config.dpi))
        render_svg_for_bbox(analysis, detail_config, area.bbox, svg_path, title=f"{analysis.route_name} Detail {detail_letter(index - 1)}")
        outputs.append(
            {
                "area_id": f"detail-{letter}",
                "png": str(png_path),
                "svg": str(svg_path),
                "score": area.score,
                "reasons": area.reasons,
            }
        )
    return outputs


def render_svg_for_bbox(
    analysis: RouteAnalysis,
    config: RenderConfig,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    title: str,
) -> None:
    width = config.width
    height = config.height
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{escape_xml(title)}</title>",
        "<style>.marker-label{font:700 12px system-ui,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round;fill:#111827}</style>",
        svg_muted_basemap(width, height, config.basemap_style),
    ]
    transform = Transform(bbox, (18, 18, width - 36, height - 36))
    elements.extend(svg_context_elements(analysis, transform, bbox))
    elements.extend(svg_scene_for_mode(analysis, config, bbox, (18, 18, width - 36, height - 36)))
    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_metadata(
    analysis: RouteAnalysis,
    config: RenderConfig,
    output_dir: Path,
    overview_png: Path,
    overview_svg: Path,
    inset_outputs: list[dict[str, Any]],
    cue_sheets: dict[str, str],
) -> Path:
    cues = navigation_cues(analysis, config)
    snapped = snapped_waypoints(analysis, config)
    segment_snaps = [item for item in snapped if is_segment_waypoint_name(item.waypoint.name)]
    segment_waypoint_count = len([waypoint for waypoint in analysis.waypoints if is_segment_waypoint_name(waypoint.name)])
    visible_cues = cues[: config.max_waypoint_labels] if config.label_mode == "nav-cues" else []
    cue_count = len(cues)
    arrow_count = 0 if config.render_mode == "audit" else len(field_arrow_targets(analysis, config))
    rendered_pass_label_count = len(analysis.repeated_nodes) if config.render_mode == "audit" else 0
    omitted_label_count = max(0, cue_count - config.max_waypoint_labels)
    omitted_segment_label_count = 0 if config.show_segment_labels else segment_waypoint_count
    data = {
        "source_gpx": analysis.source_gpx,
        "route_name": analysis.route_name,
        "outputs": {
            "overview_png": str(overview_png),
            "overview_svg": str(overview_svg),
            "insets": inset_outputs,
        },
        "config": {
            "width": config.width,
            "height": config.height,
            "dpi": config.dpi,
            "line_width": config.line_width,
            "palette": list(config.palette),
            "arrow_spacing_m": config.arrow_spacing_m,
            "overlap_offset_m": config.overlap_offset_m,
            "min_repeated_stretch_m": config.min_repeated_stretch_m,
            "max_lanes": config.max_lanes,
            "edge_snap_m": config.edge_snap_m,
            "node_snap_m": config.node_snap_m,
            "label_density": config.label_density,
            "inset_mode": config.inset_mode,
            "embed_insets": config.embed_insets,
            "basemap_style": config.basemap_style,
            "render_mode": config.render_mode,
            "profile": config.profile,
            "field_route_color": config.field_route_color,
            "route_color_mode": config.route_color_mode,
            "max_arrows": config.max_arrows,
            "simplify_tolerance_m": config.simplify_tolerance_m,
            "label_mode": config.label_mode,
            "show_cue_labels": config.show_cue_labels,
            "show_segment_labels": config.show_segment_labels,
            "show_parking_marker": config.show_parking_marker,
            "waypoint_snap_tolerance_m": config.waypoint_snap_tolerance_m,
            "context_geojson": str(config.context_geojson) if config.context_geojson else None,
        },
        "analysis": {
            "point_count": len(analysis.points),
            "segment_count": len(analysis.segments),
            "total_distance_miles": round(analysis.total_distance_m / 1609.344, 3),
            "repeated_edge_count": analysis.repeated_edge_count,
            "repeated_edge_traversal_count": analysis.repeated_edge_traversal_count,
            "repeated_intersection_count": len(analysis.repeated_nodes),
            "dense_area_count": len(analysis.dense_areas),
            "context_line_count": len(analysis.context_lines),
            "arrow_count_overview": arrow_count,
            "cue_count": cue_count,
            "navigation_cue_count": cue_count,
            "segment_waypoint_count": segment_waypoint_count,
            "primary_marker_mode": config.label_mode,
            "segment_labels_rendered": config.show_segment_labels,
            "snapped_waypoint_count": len(snapped),
            "ambiguous_waypoint_count": len([item for item in snapped if item.ambiguous]),
            "visible_marker_numbers": [cue.cue_number for cue in visible_cues],
            "repeated_stretch_count": len(repeated_stretches(analysis, config)),
            "rendered_pass_label_count": rendered_pass_label_count,
            "omitted_label_count": omitted_label_count,
            "omitted_segment_label_count": omitted_segment_label_count,
            "detail_panel_count": len(analysis.dense_areas),
            "cue_sheet_outputs": cue_sheets,
            "waypoint_snap_candidates": [
                {
                    "waypoint_name": item.waypoint.name,
                    "selected_route_miles": round(item.selected.route_m / 1609.344, 3),
                    "selected_route_index": item.selected.route_index,
                    "ambiguous": item.ambiguous,
                    "segment_name": item.segment_name,
                    "candidates": [
                        {
                            "route_miles": round(candidate.route_m / 1609.344, 3),
                            "route_index": candidate.route_index,
                            "distance_from_waypoint_m": round(candidate.distance_from_waypoint_m, 1),
                        }
                        for candidate in item.candidates
                    ],
                }
                for item in snapped
            ],
            "segment_waypoints": [
                {
                    "waypoint_name": item.waypoint.name,
                    "segment_name": item.segment_name,
                    "route_miles": round(item.selected.route_m / 1609.344, 3),
                    "ambiguous": item.ambiguous,
                }
                for item in segment_snaps
            ],
            "detail_panels": [
                {
                    "area_id": f"detail-{detail_letter(index).lower()}",
                    "score": area.score,
                    "reasons": area.reasons,
                }
                for index, area in enumerate(analysis.dense_areas)
            ],
        },
    }
    metadata_path = output_dir / "route-map-metadata.json"
    metadata_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def cue_sheet_rows(analysis: RouteAnalysis, config: RenderConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue in navigation_cues(analysis, config):
        rows.append(
            {
                "cue_number": cue.cue_number,
                "map_label": cue.cue_number,
                "mile": round(cue.route_mile, 3),
                "route_meters": round(cue.route_m, 1),
                "action": cue.action,
                "label": cue.label,
                "latitude": round(cue.geo.lat, 7),
                "longitude": round(cue.geo.lon, 7),
                "route_index": cue.route_index,
                "kind": cue.kind,
                "source_waypoint_name": cue.source_waypoint_name or "",
                "segment_name": cue.segment_name or "",
                "repeated_place_id": cue.repeated_place_id or "",
                "repeated_pass_number": cue.repeated_pass_number or "",
                "ambiguous": cue.ambiguous,
            }
        )
    return rows


def write_cue_sheets(analysis: RouteAnalysis, config: RenderConfig, output_dir: Path) -> dict[str, str]:
    rows = cue_sheet_rows(analysis, config)
    json_path = output_dir / "nav-cues.json"
    csv_path = output_dir / "nav-cues.csv"
    md_path = output_dir / "nav-cues.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "cue_number",
        "map_label",
        "mile",
        "route_meters",
        "action",
        "label",
        "latitude",
        "longitude",
        "route_index",
        "kind",
        "source_waypoint_name",
        "segment_name",
        "repeated_place_id",
        "repeated_pass_number",
        "ambiguous",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {analysis.route_name} Navigation Cues", "", "| Cue | Mile | Action | Source |", "| ---: | ---: | --- | --- |"]
    for row in rows:
        source = row["source_waypoint_name"] or row["segment_name"] or ""
        lines.append(f"| {row['cue_number']} | {row['mile']} | {row['action']} | {source} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    legacy_json_path = output_dir / "cue-sheet.json"
    legacy_csv_path = output_dir / "cue-sheet.csv"
    legacy_md_path = output_dir / "cue-sheet.md"
    shutil.copyfile(json_path, legacy_json_path)
    shutil.copyfile(csv_path, legacy_csv_path)
    shutil.copyfile(md_path, legacy_md_path)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "legacy_json": str(legacy_json_path),
        "legacy_csv": str(legacy_csv_path),
        "legacy_md": str(legacy_md_path),
    }


def validate_navigation_cues(cues: list[NavCue], config: RenderConfig) -> None:
    if config.label_mode != "nav-cues":
        return
    if len(cues) < 2:
        raise ValueError("Navigation cue mode requires at least start and finish cues.")
    if cues[0].kind != "start":
        raise ValueError("First visible navigation cue must be route start/car.")
    if cues[-1].kind != "finish":
        raise ValueError("Final visible navigation cue must be finish/return-to-car.")
    previous_route_m = -1.0
    for expected_number, cue in enumerate(cues, start=1):
        if cue.cue_number != expected_number:
            raise ValueError("Navigation cue numbers must be sequential in route order.")
        if cue.route_m < previous_route_m:
            raise ValueError("Navigation cue route distances must be monotonic.")
        previous_route_m = cue.route_m
        if cue.source_waypoint_name and is_segment_waypoint_name(cue.source_waypoint_name):
            raise ValueError("Segment metadata waypoint was promoted to a primary navigation cue.")


def napkin_cue_rows(cues: list[NapkinCue]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue in cues:
        rows.append(
            {
                "cue_number": cue.cue_number,
                "distance_from_previous_miles": round(cue.distance_from_previous_miles, 2),
                "action": cue.display_action,
                "elevation_hint": cue.elevation_hint,
                "label": cue.label,
                "same_as_cue_number": cue.same_as_cue_number or "",
                "needs_review": cue.needs_review,
                "route_mile": round(cue.route_mile, 3),
                "lat": round(cue.geo.lat, 7),
                "lon": round(cue.geo.lon, 7),
                "elevation_delta_ft": round(cue.elevation_delta_ft, 1) if cue.elevation_delta_ft is not None else "",
                "turn_confidence": round(cue.turn_confidence, 2),
                "review_reasons": ";".join(cue.review_reasons),
            }
        )
    return rows


def write_napkin_cue_files(analysis: RouteAnalysis, config: RenderConfig, output_dir: Path) -> dict[str, str]:
    cues = build_napkin_cues(analysis, config)
    rows = napkin_cue_rows(cues)
    csv_path = output_dir / "napkin-cues.csv"
    md_path = output_dir / "napkin-cues.md"
    review_path = output_dir / "napkin-review.json"
    fieldnames = [
        "cue_number",
        "distance_from_previous_miles",
        "action",
        "elevation_hint",
        "label",
        "same_as_cue_number",
        "needs_review",
        "route_mile",
        "lat",
        "lon",
        "elevation_delta_ft",
        "turn_confidence",
        "review_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {analysis.route_name} Napkin Field Guide",
        "",
        "Schematic field guide - not to scale. Follow cue order.",
        "",
    ]
    for cue in cues:
        lines.append(f"{cue.cue_number}. {action_chip_text(cue)}")
        detail = f"{cue.distance_from_previous_miles:.2f} mi from previous cue"
        if cue.label and cue.cue_number not in {1, len(cues)}:
            detail += f" - {cue.label}"
        lines.append(f"   {detail}.")
        if cue.same_as_cue_number:
            lines.append(f"   same junction as cue {cue.same_as_cue_number}; choose this pass's action.")
        if cue.needs_review:
            lines.append(f"   Review: {', '.join(cue.review_reasons)}.")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    review = {
        "profile": "napkin",
        "source_of_truth": "GPX/navigation cue renderer",
        "not_to_scale": True,
        "cue_count": len(cues),
        "low_turn_confidence_cues": [row for row in rows if "low_turn_confidence" in str(row["review_reasons"])],
        "repeated_junction_cues": [row for row in rows if "repeated_junction" in str(row["review_reasons"])],
        "ambiguous_waypoint_cues": [row for row in rows if "ambiguous_waypoint_snap" in str(row["review_reasons"])],
        "missing_elevation_cues": [row for row in rows if "missing_elevation" in str(row["review_reasons"])],
    }
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path), "review_json": str(review_path)}


def decision_instruction(cue: NapkinCue, previous_cue: NapkinCue | None = None) -> str:
    if cue.cue_number == 1:
        return "Start at the car / parking point. Start the GPX before leaving."
    if cue.action == "RETURN / FINISH":
        return "Return to the car and stop the GPX."
    prefix = ""
    action = cue.action
    if cue.display_action.startswith("SAME JCT / "):
        prefix = f"Same junction as cue {cue.same_as_cue_number}: "
        action = cue.display_action.replace("SAME JCT / ", "", 1)
    token = trail_target_token(cue.label)
    target_name = trail_target_name(cue.label)
    previous_token = trail_target_token(previous_cue.label) if previous_cue else ""
    previous_name = trail_target_name(previous_cue.label) if previous_cue else ""
    origin = ""
    if previous_cue and previous_cue.cue_number == 1:
        origin = " from the car"
    elif previous_token:
        origin = f" after {previous_token}"
    elif previous_name and previous_name not in {"START / CAR", "RETURN / FINISH"}:
        origin = f" after {previous_name}"
    if action.startswith("TAKE ") and token:
        if target_name and target_name != token:
            return f"{prefix}Take signed {token} toward {target_name}{origin}."
        return f"{prefix}Take signed {token}{origin}."
    if action == "DECISION":
        return f"{prefix}Confirm the next signed trail before continuing."
    return f"{prefix}{action.title()}."


def field_decision_rows(cues: list[NapkinCue]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        previous_cue = cues[index - 1] if index > 0 else None
        rows.append(
            {
                "cue_number": cue.cue_number,
                "title": action_chip_text(cue),
                "instruction": decision_instruction(cue, previous_cue),
                "distance_from_previous_miles": round(cue.distance_from_previous_miles, 2),
                "route_mile": round(cue.route_mile, 3),
                "elevation_hint": cue.elevation_hint,
                "label": cue.label,
                "same_as_cue_number": cue.same_as_cue_number or "",
                "needs_review": cue.needs_review,
                "review_reasons": list(cue.review_reasons),
                "latitude": round(cue.geo.lat, 7),
                "longitude": round(cue.geo.lon, 7),
            }
        )
    return rows


def write_field_decision_files(cues: list[NapkinCue], analysis: RouteAnalysis, output_dir: Path) -> dict[str, str]:
    rows = field_decision_rows(cues)
    json_path = output_dir / "field-decisions.json"
    md_path = output_dir / "field-decisions.md"
    html_path = output_dir / "field-decisions.html"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# {analysis.route_name} Field Decisions",
        "",
        "Use this as the quick field aid. The GPX remains the source of truth for exact track geometry.",
        "",
    ]
    for row in rows:
        md_lines.append(f"## {row['cue_number']}. {row['title']}")
        md_lines.append("")
        md_lines.append(str(row["instruction"]))
        md_lines.append("")
        md_lines.append(
            f"- Since last cue: {row['distance_from_previous_miles']} mi"
            + (f"; elevation: {row['elevation_hint']}" if row["elevation_hint"] != "ELEV REVIEW" else "")
        )
        if row["same_as_cue_number"]:
            md_lines.append(f"- Same physical place as cue {row['same_as_cue_number']}; use this cue's instruction for this pass.")
        if row["needs_review"]:
            md_lines.append(f"- Review flag: {', '.join(row['review_reasons'])}")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    card_html = []
    for row in rows:
        review = ""
        if row["needs_review"]:
            review = f"<p class=\"review\">Review: {html.escape(', '.join(row['review_reasons']))}</p>"
        same_place = ""
        if row["same_as_cue_number"]:
            same_place = f"<p class=\"same\">Same place as cue {row['same_as_cue_number']}; follow this pass.</p>"
        elevation = "" if row["elevation_hint"] == "ELEV REVIEW" else f"<span>{html.escape(str(row['elevation_hint']))}</span>"
        card_html.append(
            "\n".join(
                [
                    f"<article class=\"cue-card\" id=\"cue-{row['cue_number']}\">",
                    f"  <div class=\"cue-num\">{row['cue_number']}</div>",
                    "  <div>",
                    f"    <h2>{html.escape(str(row['title']))}</h2>",
                    f"    <p class=\"instruction\">{html.escape(str(row['instruction']))}</p>",
                    f"    <p class=\"meta\"><span>{row['distance_from_previous_miles']} mi since last cue</span>{elevation}</p>",
                    same_place,
                    review,
                    "  </div>",
                    "</article>",
                ]
            )
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(analysis.route_name)} Field Decisions</title>
  <style>
    body {{ margin: 0; background: #f8fafc; color: #111827; font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 16px; background: rgba(248, 250, 252, .96); border-bottom: 1px solid #dbe3ef; }}
    h1 {{ margin: 0; font-size: 22px; line-height: 1.15; }}
    header p {{ margin: 6px 0 0; color: #475569; font-size: 14px; }}
    main {{ display: grid; gap: 12px; padding: 12px; max-width: 720px; margin: 0 auto 32px; }}
    .cue-card {{ display: grid; grid-template-columns: 46px 1fr; gap: 12px; padding: 14px; background: white; border: 1px solid #d7dee8; border-radius: 8px; box-shadow: 0 1px 2px rgba(15, 23, 42, .08); }}
    .cue-num {{ width: 38px; height: 38px; border-radius: 999px; background: #111827; color: white; display: grid; place-items: center; font-weight: 900; }}
    .cue-card:first-child .cue-num {{ background: #16a34a; }}
    .cue-card:last-child .cue-num {{ background: #dc2626; }}
    h2 {{ margin: 0 0 8px; font-size: 19px; line-height: 1.2; }}
    .instruction {{ margin: 0; font-size: 18px; font-weight: 650; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 0; color: #475569; font-size: 14px; }}
    .meta span {{ padding: 3px 8px; background: #eef2f7; border-radius: 999px; }}
    .same {{ margin: 10px 0 0; color: #92400e; font-weight: 700; }}
    .review {{ margin: 10px 0 0; color: #9f1239; font-size: 14px; }}
  </style>
</head>
<body>
  <header>
    <h1>What to do next</h1>
    <p>{html.escape(analysis.route_name)}. Cue numbers are field order, not challenge segment order.</p>
  </header>
  <main>
{chr(10).join(card_html)}
  </main>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    return {"field_decisions_json": str(json_path), "field_decisions_md": str(md_path), "field_decisions_html": str(html_path)}


def write_napkin_outputs(analysis: RouteAnalysis, config: RenderConfig, output_dir: Path) -> dict[str, Any]:
    png_path = output_dir / "napkin-map.png"
    svg_path = output_dir / "napkin-map.svg"
    draw_napkin_map_png(analysis, config, png_path)
    svg_napkin_map(analysis, config, svg_path)
    cue_files = write_napkin_cue_files(analysis, config, output_dir)
    decision_files = write_field_decision_files(build_napkin_cues(analysis, config), analysis, output_dir)
    return {"png": str(png_path), "svg": str(svg_path), **cue_files, **decision_files}




def render_route_map(input_gpx: Path, output_dir: Path, config: RenderConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_gpx(input_gpx, config)
    validate_navigation_cues(navigation_cues(analysis, config), config)
    overview_png = output_dir / "route-overview.png"
    overview_svg = output_dir / "route-overview.svg"
    render_png(analysis, config, overview_png)
    render_svg(analysis, config, overview_svg)
    inset_outputs = write_inset_outputs(analysis, config, output_dir)
    legacy_png = output_dir / "route-map.png"
    legacy_svg = output_dir / "route-map.svg"
    shutil.copyfile(overview_png, legacy_png)
    shutil.copyfile(overview_svg, legacy_svg)
    cue_sheets = write_cue_sheets(analysis, config, output_dir)
    napkin_outputs = write_napkin_outputs(analysis, config, output_dir) if config.profile == "napkin" else {}
    metadata_path = write_metadata(analysis, config, output_dir, overview_png, overview_svg, inset_outputs, cue_sheets)
    return {
        "analysis": analysis,
        "overview_png": overview_png,
        "overview_svg": overview_svg,
        "metadata": metadata_path,
        "insets": inset_outputs,
        "cue_sheets": cue_sheets,
        "napkin": napkin_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpx", type=Path, help="Input GPX track.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--profile", default="overview", choices=["overview", "field-packet", "imagegen-helper", "napkin", "audit"], help="Rendering profile.")
    parser.add_argument("--style", choices=["muted", "context", "topo", "none"], help="Legacy alias for --basemap.")
    parser.add_argument("--basemap", default="context", choices=["context", "topo", "none", "muted"], help="Basemap style.")
    parser.add_argument("--insets", default="auto", choices=["auto", "off"], help="Inset generation mode.")
    parser.add_argument("--detail-panels", choices=["auto", "off"], help="Alias for --insets.")
    parser.add_argument("--embed-insets", action="store_true", help="Embed detail panels inside the overview instead of writing separate files only.")
    parser.add_argument("--mode", default="field", choices=["field", "audit"], help="Field mode favors readable cue maps; audit mode shows dense repeated-pass diagnostics.")
    parser.add_argument("--width", type=int, default=1400, help="Overview image width in pixels.")
    parser.add_argument("--height", type=int, default=1000, help="Overview image height in pixels.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI metadata.")
    parser.add_argument("--line-width", type=float, default=7.0, help="Route line width in pixels.")
    parser.add_argument("--field-route-color", default="#1d4ed8", help="Solid route color used by --mode field.")
    parser.add_argument("--route-color-mode", default="solid", choices=["solid", "gradient", "cue-legs"], help="How field route ribbon color is assigned.")
    parser.add_argument("--max-arrows", type=int, default=28, help="Hard cap for overview chevrons in field mode.")
    parser.add_argument("--simplify-tolerance", type=float, default=10.0, help="Route-ribbon simplification tolerance in meters for --mode field.")
    parser.add_argument("--palette", default=DEFAULT_PALETTE, help="Comma-separated hex colors for route progress gradient.")
    parser.add_argument("--arrow-spacing", type=float, default=320.0, help="Base arrow spacing in route meters.")
    parser.add_argument("--overlap-offset", "--overlap-offset-m", dest="overlap_offset", type=float, default=8.0, help="Parallel lane offset for repeated traversals, in meters.")
    parser.add_argument("--min-repeated-stretch-m", type=float, default=50.0, help="Minimum repeated route stretch to report/draw as meaningful.")
    parser.add_argument("--max-lanes", type=int, default=3, help="Maximum repeated-stretch lanes to draw in field/detail modes.")
    parser.add_argument("--max-pass-labels-overview", type=int, default=12, help="Maximum repeated-pass labels on overview maps.")
    parser.add_argument("--max-pass-labels-inset", type=int, default=25, help="Maximum repeated-pass labels on detail panels.")
    parser.add_argument("--label-mode", default="nav-cues", choices=["nav-cues", "segment-waypoints"], help="Primary marker source. Default renders route-order navigation cues.")
    parser.add_argument("--show-cue-labels", default="true", choices=["true", "false"], help="Show cue text labels next to primary cue markers.")
    parser.add_argument("--show-segment-labels", default="false", choices=["true", "false"], help="Show segment metadata as small secondary labels.")
    parser.add_argument("--show-parking-marker", default="false", choices=["true", "false"], help="Show a P parking marker near the route start.")
    parser.add_argument("--waypoint-snap-tolerance", type=float, default=35.0, help="Maximum waypoint-to-route snap distance before falling back to nearest route position.")
    parser.add_argument("--label-density", default="normal", choices=["low", "normal", "high"], help="How many labels to place.")
    parser.add_argument("--edge-snap", type=float, default=8.0, help="Repeated-edge snap tolerance in meters.")
    parser.add_argument("--node-snap", type=float, default=16.0, help="Repeated-intersection snap tolerance in meters.")
    parser.add_argument("--inset-size", type=int, default=280, help="Inset panel size in pixels.")
    parser.add_argument("--max-insets", type=int, default=3, help="Maximum number of automatic inset panels.")
    parser.add_argument("--markers-json", type=Path, help="Optional extra markers JSON.")
    parser.add_argument("--context-geojson", type=Path, help="Optional muted trail/road context GeoJSON FeatureCollection.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RenderConfig:
    palette = tuple(color.strip() for color in args.palette.split(",") if color.strip())
    if not palette:
        raise ValueError("--palette must contain at least one color")
    for color in palette:
        parse_hex_color(color)
    parse_hex_color(args.field_route_color)
    basemap_style = args.style or args.basemap
    inset_mode = args.detail_panels or args.insets
    render_mode = "audit" if args.profile == "audit" else args.mode
    label_density = "high" if args.profile == "field-packet" else args.label_density
    width = args.width
    height = args.height
    if args.profile == "field-packet" and args.width == 1400 and args.height == 1000:
        width = 1170
        height = 2532
    if args.profile == "imagegen-helper" and args.width == 1400 and args.height == 1000:
        width = 1024
        height = 1280
    if args.profile == "napkin" and args.width == 1400 and args.height == 1000:
        width = 1024
        height = 1536
    inset_mode = "off" if args.profile == "imagegen-helper" else inset_mode
    inset_mode = "off" if args.profile == "napkin" else inset_mode
    line_width = args.line_width
    arrow_spacing = args.arrow_spacing
    max_arrows = args.max_arrows
    simplify_tolerance = args.simplify_tolerance
    show_cue_labels = args.show_cue_labels == "true"
    show_parking_marker = args.show_parking_marker == "true"
    if args.profile == "imagegen-helper":
        label_density = "high"
        line_width = 10.0 if args.line_width == 7.0 else args.line_width
        arrow_spacing = 420.0 if args.arrow_spacing == 320.0 else args.arrow_spacing
        max_arrows = 12 if args.max_arrows == 28 else args.max_arrows
        simplify_tolerance = 18.0 if args.simplify_tolerance == 10.0 else args.simplify_tolerance
        show_cue_labels = False if args.show_cue_labels == "true" else show_cue_labels
        show_parking_marker = True if args.show_parking_marker == "false" else show_parking_marker
    if args.profile == "napkin":
        label_density = "high"
        line_width = 12.0 if args.line_width == 7.0 else args.line_width
        arrow_spacing = 300.0 if args.arrow_spacing == 320.0 else args.arrow_spacing
        max_arrows = 16 if args.max_arrows == 28 else args.max_arrows
        simplify_tolerance = 28.0 if args.simplify_tolerance == 10.0 else args.simplify_tolerance
        show_cue_labels = False if args.show_cue_labels == "true" else show_cue_labels
        show_parking_marker = True if args.show_parking_marker == "false" else show_parking_marker
    return RenderConfig(
        width=width,
        height=height,
        dpi=args.dpi,
        line_width=line_width,
        arrow_spacing_m=arrow_spacing,
        overlap_offset_m=args.overlap_offset,
        min_repeated_stretch_m=args.min_repeated_stretch_m,
        max_lanes=args.max_lanes,
        edge_snap_m=args.edge_snap,
        node_snap_m=args.node_snap,
        label_density=label_density,
        basemap_style=basemap_style,
        inset_mode=inset_mode,
        embed_insets=args.embed_insets,
        max_insets=args.max_insets,
        inset_size=args.inset_size,
        render_mode=render_mode,
        profile=args.profile,
        field_route_color=args.field_route_color,
        route_color_mode=args.route_color_mode,
        max_arrows=max_arrows,
        max_pass_labels_overview=args.max_pass_labels_overview,
        max_pass_labels_inset=args.max_pass_labels_inset,
        simplify_tolerance_m=simplify_tolerance,
        label_mode=args.label_mode,
        show_cue_labels=show_cue_labels,
        show_segment_labels=args.show_segment_labels == "true",
        show_parking_marker=show_parking_marker,
        waypoint_snap_tolerance_m=args.waypoint_snap_tolerance,
        palette=palette,
        marker_json=args.markers_json,
        context_geojson=args.context_geojson,
    )


def main() -> int:
    args = parse_args()
    result = render_route_map(args.gpx, args.output, config_from_args(args))
    print(f"Wrote {result['overview_png']}")
    print(f"Wrote {result['overview_svg']}")
    print(f"Wrote {result['metadata']}")
    for inset in result["insets"]:
        print(f"Wrote {inset['png']}")
        print(f"Wrote {inset['svg']}")
    for napkin_output in result.get("napkin", {}).values():
        print(f"Wrote {napkin_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
