#!/usr/bin/env python3
"""Audit field-packet GPX tracks against published trail management rules."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_OFFICIAL_GEOJSON = (
    YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
)
DEFAULT_R2R_TRAILS_GEOJSON = (
    YEAR_DIR / "inputs" / "open-data" / "r2r-trails-2026-05-04" / "boise_parks_trails_open_data.geojson"
)
DEFAULT_RULES_JSON = YEAR_DIR / "inputs" / "open-data" / "special-management-rules-2026.json"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "special-management-rule-audit-2026-05-22.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "special-management-rule-audit-2026-05-22.md"
DEFAULT_OFFICIAL_SEGMENT_SNAP_TOLERANCE_MILES = 0.03
DEFAULT_OPEN_TRAIL_MODE_SNAP_TOLERANCE_MILES = 0.0075


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def line_parts(geometry: dict[str, Any] | None) -> list[list[tuple[float, float]]]:
    if not geometry:
        return []
    coords = geometry.get("coordinates") or []
    if geometry.get("type") == "LineString":
        raw_parts = [coords]
    elif geometry.get("type") == "MultiLineString":
        raw_parts = coords
    else:
        return []
    parts = []
    for raw_part in raw_parts:
        part = []
        for coord in raw_part or []:
            if len(coord) >= 2:
                part.append((float(coord[0]), float(coord[1])))
        if len(part) >= 2:
            parts.append(part)
    return parts


def bbox_for_part(part: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in part]
    lats = [point[1] for point in part]
    return (min(lons), min(lats), max(lons), max(lats))


def point_near_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
    margin_miles: float,
) -> bool:
    lon, lat = point
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_margin = margin_miles / 69.0
    lon_margin = margin_miles / max(1.0, 69.172 * math.cos(math.radians(lat)))
    return (
        min_lon - lon_margin <= lon <= max_lon + lon_margin
        and min_lat - lat_margin <= lat <= max_lat + lat_margin
    )


def official_segment_index(official_geojson: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = props.get("segId") or props.get("seg_id")
        if seg_id is None:
            continue
        segment_name = str(props.get("segName") or props.get("segment_name") or "")
        trail_name = segment_name.rsplit(" ", 1)[0] if segment_name.rsplit(" ", 1)[-1].isdigit() else segment_name
        parts = line_parts(feature.get("geometry") or {})
        index[str(seg_id)] = {
            "segment_id": str(seg_id),
            "segment_name": segment_name,
            "trail_name": str(props.get("trail_name") or props.get("TrailName") or trail_name),
            "parts": parts,
            "part_bboxes": [bbox_for_part(part) for part in parts],
            "properties": props,
        }
    return index


def open_trail_features(open_trails_geojson: dict[str, Any] | None) -> list[dict[str, Any]]:
    features = []
    for feature in (open_trails_geojson or {}).get("features") or []:
        props = feature.get("properties") or {}
        parts = line_parts(feature.get("geometry") or {})
        if not parts:
            continue
        features.append(
            {
                "trail_id": str(props.get("TrailID") or ""),
                "trail_name": str(props.get("TrailName") or props.get("Name") or ""),
                "name": str(props.get("Name") or props.get("TrailName") or ""),
                "properties": props,
                "parts": parts,
                "part_bboxes": [bbox_for_part(part) for part in parts],
            }
        )
    return features


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
    px, py = local_xy_miles(point, origin_lat)
    ax, ay = local_xy_miles(start, origin_lat)
    bx, by = local_xy_miles(end, origin_lat)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


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


def point_to_polyline_distance_miles(point: tuple[float, float], coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return float("inf")
    return min(
        point_to_segment_distance_miles(point, start, end, point[1])
        for start, end in zip(coords, coords[1:])
    )


def interval_direction_on_part(
    start: tuple[float, float],
    end: tuple[float, float],
    part: list[tuple[float, float]],
) -> str:
    start_mile, _start_gap = project_miles_along_polyline(start, part)
    end_mile, _end_gap = project_miles_along_polyline(end, part)
    if abs(end_mile - start_mile) < 1e-6:
        return "stationary"
    return "forward" if end_mile > start_mile else "reverse"


def segment_traversal_directions(
    segment: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    *,
    snap_tolerance_miles: float = 0.03,
    min_matched_miles: float = 0.10,
) -> dict[str, float]:
    direction_miles: dict[str, float] = {}
    parts = segment.get("parts") or []
    for track in track_segments:
        for start, end in zip(track, track[1:]):
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            matches = [
                (point_to_polyline_distance_miles(midpoint, part), part)
                for part in parts
            ]
            if not matches:
                continue
            distance, part = min(matches, key=lambda item: item[0])
            if distance > snap_tolerance_miles:
                continue
            direction = interval_direction_on_part(start, end, part)
            if direction not in {"forward", "reverse"}:
                continue
            direction_miles[direction] = direction_miles.get(direction, 0.0) + haversine_miles(start, end)
    return {
        direction: round(miles, 3)
        for direction, miles in direction_miles.items()
        if miles >= min_matched_miles
    }


def directional_rule_segment_ids(rules: list[dict[str, Any]]) -> list[str]:
    ids = []
    for rule in rules:
        if str(rule.get("rule_type") or "") != "directional_segment_traversal":
            continue
        ids.extend(str(segment_id) for segment_id in (rule.get("segment_direction_overrides") or {}).keys())
    return sorted(set(ids), key=lambda item: (len(item), item))


def closest_special_segment_match(
    midpoint: tuple[float, float],
    official_index: dict[str, dict[str, Any]],
    segment_ids: list[str],
    snap_tolerance_miles: float = 0.03,
) -> tuple[str | None, list[tuple[float, float]] | None, float]:
    best_segment_id = None
    best_part = None
    best_distance = float("inf")
    for segment_id in segment_ids:
        segment = official_index.get(str(segment_id)) or {}
        for part, bbox in zip(segment.get("parts") or [], segment.get("part_bboxes") or []):
            if not point_near_bbox(midpoint, bbox, snap_tolerance_miles):
                continue
            distance = point_to_polyline_distance_miles(midpoint, part)
            if distance < best_distance:
                best_segment_id = str(segment_id)
                best_part = part
                best_distance = distance
    return best_segment_id, best_part, best_distance


def closest_open_trail_match(
    midpoint: tuple[float, float],
    features: list[dict[str, Any]],
    snap_tolerance_miles: float = 0.03,
) -> tuple[dict[str, Any] | None, list[tuple[float, float]] | None, float]:
    best_feature = None
    best_part = None
    best_distance = float("inf")
    for feature in features:
        for part, bbox in zip(feature.get("parts") or [], feature.get("part_bboxes") or []):
            if not point_near_bbox(midpoint, bbox, snap_tolerance_miles):
                continue
            distance = point_to_polyline_distance_miles(midpoint, part)
            if distance < best_distance:
                best_feature = feature
                best_part = part
                best_distance = distance
    return best_feature, best_part, best_distance


def rule_targets_open_feature(rule: dict[str, Any], feature: dict[str, Any]) -> bool:
    trail_ids = {str(item).lower() for item in rule.get("trail_ids") or []}
    if trail_ids and str(feature.get("trail_id") or "").lower() in trail_ids:
        return True
    trail_names = {str(item).lower() for item in rule.get("trail_names") or []}
    feature_names = {
        str(feature.get("trail_name") or "").lower(),
        str(feature.get("name") or "").lower(),
    }
    return bool(trail_names & feature_names)


def rule_disambiguates_open_feature(rule: dict[str, Any], feature: dict[str, Any]) -> bool:
    trail_ids = {str(item).lower() for item in rule.get("disambiguation_trail_ids") or []}
    if trail_ids and str(feature.get("trail_id") or "").lower() in trail_ids:
        return True
    trail_names = {str(item).lower() for item in rule.get("disambiguation_trail_names") or []}
    feature_names = {
        str(feature.get("trail_name") or "").lower(),
        str(feature.get("name") or "").lower(),
    }
    return bool(trail_names & feature_names)


def matched_rule_open_trail_miles(
    track_segments: list[list[tuple[float, float]]],
    features: list[dict[str, Any]],
    rule: dict[str, Any],
    *,
    snap_tolerance_miles: float = 0.03,
    min_matched_miles: float = 0.10,
) -> float:
    matched_miles = 0.0
    if not features:
        return matched_miles
    target_features = [feature for feature in features if rule_targets_open_feature(rule, feature)]
    if not target_features:
        return matched_miles
    disambiguation_features = [
        feature for feature in features if rule_disambiguates_open_feature(rule, feature)
    ]
    candidate_features = target_features + [
        feature for feature in disambiguation_features if feature not in target_features
    ]
    for track in track_segments:
        for start, end in zip(track, track[1:]):
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            feature, _part, distance = closest_open_trail_match(
                midpoint,
                candidate_features,
                snap_tolerance_miles=snap_tolerance_miles,
            )
            if not feature or distance > snap_tolerance_miles:
                continue
            if rule_targets_open_feature(rule, feature):
                matched_miles += haversine_miles(start, end)
    return round(matched_miles, 3) if matched_miles >= min_matched_miles else 0.0


def matched_special_segment_directions(
    track_segments: list[list[tuple[float, float]]],
    official_index: dict[str, dict[str, Any]],
    rules: list[dict[str, Any]],
    *,
    snap_tolerance_miles: float = 0.03,
    min_matched_miles: float = 0.10,
) -> dict[str, dict[str, float]]:
    segment_ids = directional_rule_segment_ids(rules)
    direction_miles: dict[str, dict[str, float]] = {}
    if not segment_ids:
        return direction_miles
    for track in track_segments:
        for start, end in zip(track, track[1:]):
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            segment_id, part, distance = closest_special_segment_match(
                midpoint,
                official_index,
                segment_ids,
                snap_tolerance_miles=snap_tolerance_miles,
            )
            if not segment_id or not part or distance > snap_tolerance_miles:
                continue
            direction = interval_direction_on_part(start, end, part)
            if direction not in {"forward", "reverse"}:
                continue
            direction_miles.setdefault(segment_id, {})
            direction_miles[segment_id][direction] = (
                direction_miles[segment_id].get(direction, 0.0) + haversine_miles(start, end)
            )
    return {
        segment_id: {
            direction: round(miles, 3)
            for direction, miles in values.items()
            if miles >= min_matched_miles
        }
        for segment_id, values in direction_miles.items()
        if any(miles >= min_matched_miles for miles in values.values())
    }


def route_label(route: dict[str, Any]) -> str:
    return str(route.get("label") or route.get("outing_id") or "unknown-route")


def failure(
    *,
    code: str,
    route: dict[str, Any],
    rule: dict[str, Any],
    segment_id: str,
    observed_directions: dict[str, float],
    allowed_directions: list[str],
    segment_name: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "error",
        "route_label": route_label(route),
        "rule_id": rule.get("rule_id"),
        "segment_id": str(segment_id),
        "segment_name": segment_name,
        "observed_directions": observed_directions,
        "allowed_directions": list(allowed_directions),
        "message": (
            f"{route_label(route)} traverses {segment_name} ({segment_id}) as "
            f"{sorted(observed_directions)} but published rule requires "
            f"{rule.get('required_direction_label') or allowed_directions}."
        ),
        "source_url": rule.get("source_url"),
    }


def mode_failure(
    *,
    code: str,
    route: dict[str, Any],
    rule: dict[str, Any],
    matched_miles: float,
    activity_type: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "error",
        "route_label": route_label(route),
        "rule_id": rule.get("rule_id"),
        "matched_miles": matched_miles,
        "activity_type": activity_type,
        "message": (
            f"{route_label(route)} traverses about {matched_miles:.2f} mi of "
            f"{rule.get('trail_name') or rule.get('trail_number') or 'a restricted trail'} "
            f"while planned as {activity_type}; published rule says {rule.get('foot_rule') or 'this mode is not allowed'}."
        ),
        "source_url": rule.get("source_url"),
    }


def audit_route_against_special_management_rules(
    route: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    official_index: dict[str, dict[str, Any]],
    rules: list[dict[str, Any]],
    open_features: list[dict[str, Any]] | None = None,
    *,
    activity_type: str = "on_foot",
    snap_tolerance_miles: float = DEFAULT_OFFICIAL_SEGMENT_SNAP_TOLERANCE_MILES,
    open_trail_snap_tolerance_miles: float = DEFAULT_OPEN_TRAIL_MODE_SNAP_TOLERANCE_MILES,
) -> dict[str, Any]:
    failures = []
    checked_segments = []
    date_context_required = []
    matched_directions = matched_special_segment_directions(
        track_segments,
        official_index,
        rules,
        snap_tolerance_miles=snap_tolerance_miles,
    )
    for rule in rules:
        rule_type = str(rule.get("rule_type") or "")
        if rule_type == "date_use_restriction":
            route_segment_ids = {str(item) for item in route.get("segment_ids") or []}
            rule_segment_ids = {str(item) for item in rule.get("segment_ids") or []}
            rule_applies = bool(route_segment_ids & rule_segment_ids)
            if not rule_applies:
                matched_miles = matched_rule_open_trail_miles(
                    track_segments,
                    open_features or [],
                    rule,
                    snap_tolerance_miles=open_trail_snap_tolerance_miles,
                )
                rule_applies = matched_miles > 0
            if rule_applies and not route.get("planned_date"):
                date_context_required.append(rule.get("rule_id"))
            if rule_applies and route.get("planned_date") and activity_type in (rule.get("prohibited_activity_types_on_odd_days") or []):
                day = datetime.fromisoformat(str(route["planned_date"])).day
                if day % 2 == 1:
                    failures.append(
                        mode_failure(
                            code="special_management_date_use_violated",
                            route=route,
                            rule=rule,
                            matched_miles=0.0,
                            activity_type=activity_type,
                        )
                    )
            continue
        if rule_type == "mode_restriction":
            if activity_type not in [str(item) for item in rule.get("prohibited_activity_types") or []]:
                continue
            matched_miles = matched_rule_open_trail_miles(
                track_segments,
                open_features or [],
                rule,
                snap_tolerance_miles=open_trail_snap_tolerance_miles,
            )
            if matched_miles > 0:
                failures.append(
                    mode_failure(
                        code="special_management_mode_violated",
                        route=route,
                        rule=rule,
                        matched_miles=matched_miles,
                        activity_type=activity_type,
                    )
                )
            continue
        if rule_type != "directional_segment_traversal":
            continue
        for segment_id, allowed in (rule.get("segment_direction_overrides") or {}).items():
            segment = official_index.get(str(segment_id))
            if not segment:
                continue
            observed_directions = matched_directions.get(str(segment_id)) or {}
            if not observed_directions:
                continue
            allowed_directions = [str(item) for item in allowed]
            checked_segments.append(
                {
                    "rule_id": rule.get("rule_id"),
                    "segment_id": str(segment_id),
                    "segment_name": segment.get("segment_name"),
                    "observed_directions": observed_directions,
                    "allowed_directions": allowed_directions,
                }
            )
            disallowed = {
                direction: miles
                for direction, miles in observed_directions.items()
                if direction not in allowed_directions
            }
            if disallowed:
                failures.append(
                    failure(
                        code="special_management_direction_violated",
                        route=route,
                        rule=rule,
                        segment_id=str(segment_id),
                        observed_directions=disallowed,
                        allowed_directions=allowed_directions,
                        segment_name=str(segment.get("segment_name") or ""),
                    )
                )
    return {
        "label": route.get("label"),
        "outing_id": route.get("outing_id"),
        "passed": not failures,
        "checked_segments": checked_segments,
        "date_context_required": date_context_required,
        "failures": failures,
    }


def build_special_management_audit(
    *,
    field_tool_data: dict[str, Any],
    official_geojson: dict[str, Any],
    rules_config: dict[str, Any],
    packet_dir: Path,
    open_trails_geojson: dict[str, Any] | None = None,
    official_segment_snap_tolerance_miles: float = DEFAULT_OFFICIAL_SEGMENT_SNAP_TOLERANCE_MILES,
    open_trail_snap_tolerance_miles: float = DEFAULT_OPEN_TRAIL_MODE_SNAP_TOLERANCE_MILES,
) -> dict[str, Any]:
    official_index = official_segment_index(official_geojson)
    open_features = open_trail_features(open_trails_geojson)
    rules = list(rules_config.get("rules") or [])
    activity_type = str(rules_config.get("activity_type") or "on_foot")
    route_reports = []
    for route in field_tool_data.get("routes") or []:
        gpx_href = route.get("gpx_href")
        track_segments = parse_gpx_track_segments(packet_dir / str(gpx_href)) if gpx_href else []
        route_reports.append(
            audit_route_against_special_management_rules(
                route,
                track_segments,
                official_index,
                rules,
                open_features,
                activity_type=activity_type,
                snap_tolerance_miles=official_segment_snap_tolerance_miles,
                open_trail_snap_tolerance_miles=open_trail_snap_tolerance_miles,
            )
        )
    failed_routes = [route for route in route_reports if not route.get("passed")]
    failure_counts: dict[str, int] = {}
    for route in failed_routes:
        for item in route.get("failures") or []:
            code = str(item.get("code") or "unknown")
            failure_counts[code] = failure_counts.get(code, 0) + 1
    return {
        "schema": "boise_trails_special_management_rule_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failed_routes else "failed",
        "source_rules": {
            "schema": rules_config.get("schema"),
            "source_date": rules_config.get("source_date"),
            "rule_count": len(rules),
        },
        "summary": {
            "route_count": len(route_reports),
            "passed_route_count": len(route_reports) - len(failed_routes),
            "failed_route_count": len(failed_routes),
            "failure_counts": failure_counts,
            "date_context_required_route_count": len(
                [route for route in route_reports if route.get("date_context_required")]
            ),
        },
        "routes": route_reports,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Special Management Rule Audit - 2026-05-22",
        "",
        f"- Status: `{audit.get('status')}`",
        f"- Routes: {audit.get('summary', {}).get('route_count')}",
        f"- Failed routes: {audit.get('summary', {}).get('failed_route_count')}",
        f"- Failure counts: `{json.dumps(audit.get('summary', {}).get('failure_counts') or {}, sort_keys=True)}`",
        "",
        "## Failed Routes",
        "",
    ]
    failed = [route for route in audit.get("routes") or [] if not route.get("passed")]
    if not failed:
        lines.append("- None.")
    for route in failed:
        lines.append(f"- `{route.get('label') or route.get('outing_id')}`")
        for item in route.get("failures") or []:
            target = (
                f"segment `{item.get('segment_id')}`"
                if item.get("segment_id") is not None
                else f"matched `{float(item.get('matched_miles') or 0):.2f}` mi"
            )
            lines.append(
                f"  - `{item.get('code')}` `{item.get('rule_id')}` {target}: "
                f"{item.get('message')}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--open-trails-geojson", type=Path, default=DEFAULT_R2R_TRAILS_GEOJSON)
    parser.add_argument("--rules-json", type=Path, default=DEFAULT_RULES_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_special_management_audit(
        field_tool_data=read_json(args.field_tool_data_json),
        official_geojson=read_json(args.official_geojson),
        rules_config=read_json(args.rules_json),
        packet_dir=args.packet_dir,
        open_trails_geojson=read_json(args.open_trails_geojson) if args.open_trails_geojson.exists() else None,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
