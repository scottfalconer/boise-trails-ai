#!/usr/bin/env python3
"""Export a phone-first field packet from the current outing menu map data."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_day_packager import (  # noqa: E402
    build_outing_menu,
    format_miles,
    format_minutes,
    outing_time_bucket_sort,
)
from export_execution_gpx import haversine_miles, validate_track_segments  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON,
    load_connector_graph,
    load_official_segments,
    read_json,
    shortest_connector_path,
)
from field_route_walkthrough_audit import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON as DEFAULT_WALKTHROUGH_CONNECTOR_GEOJSON,
    DEFAULT_OFFICIAL_GEOJSON as DEFAULT_WALKTHROUGH_OFFICIAL_GEOJSON,
    TrailGraph,
    filter_edges_for_track,
    load_graph_edges,
    matched_edge_groups,
    named_nonofficial_group,
    official_group_claimed,
    resample_track_segments_for_matching,
    start_access_text,
    text_mentions_edge,
)


DEFAULT_CANONICAL_MAP_DATA_JSON = (
    YEAR_DIR
    / "outputs"
    / "private"
    / "2026-outing-menu-map-data.json"
)
DEFAULT_PUBLIC_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_MAP_DATA_JSON = DEFAULT_CANONICAL_MAP_DATA_JSON
DEFAULT_MAP_HTML = REPO_ROOT / "outing-menu-map.html"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_BASENAME = "phone-field-packet"
DEFAULT_CERTIFICATE_JSON = YEAR_DIR / "checkpoints" / "p90-responsible-relaxed-certificate-2026-05-06.json"
DEFAULT_TRAILHEAD_CANDIDATES = (
    YEAR_DIR / "inputs" / "open-data" / "city-parks-facilities-2026-05-04" / "trailhead_candidates.geojson"
)
DEFAULT_MAX_GAP_MILES = 0.05
DEFAULT_MAX_PARKING_GAP_MILES = 0.35
GPX_ZIP_NAME = "all-field-packet-gpx.zip"
FIELD_TOOL_DATA_NAME = "field-tool-data.json"
TIME_FILTER_MINUTES = [60, 90, 120, 180, 240, 360]
COMPLETED_STORAGE_KEY = "fieldPacketCompletedOutings"
ACTIVE_STORAGE_KEY = "fieldPacketActiveOuting"
GPX_PATH_KEYS = ("gpx_path", "cue_gpx_path", "audit_gpx_path")
GPX_HREF_KEYS = ("gpx_href", "cue_gpx_href", "audit_gpx_href")
PRIVATE_LITERAL_PATTERNS = (
    "/Users/scott",
    "outputs/private",
    "GETAthleteDashboard",
    "access_token",
    "refresh_token",
    "client_secret",
)
PRIVATE_REGEX_PATTERNS = (
    re.compile(r"\b911\s+n\.?\s+18th\b", re.IGNORECASE),
    re.compile(r"\b911\s+north\s+18th\b", re.IGNORECASE),
)
BLOCKED_CONNECTOR_TOKENS = (
    "private",
    "access_no",
    "access=no",
    "foot_no",
    "foot=no",
    "no_foot",
    "non_real",
    "graph_artifact",
)
SIGNPOST_TRAIL_NUMBERS = {
    "hippie shake": "50",
    "hippie shake trail": "50",
    "who now loop": "51",
    "who now loop trail": "51",
    "kemper's ridge": "52",
    "kemper's ridge trail": "52",
    "buena vista": "53",
    "buena vista trail": "53",
    "harrison hollow": "57",
    "harrison hollow trail": "57",
    "harrison ridge": "58",
    "harrison ridge trail": "58",
    "lower hulls gulch": "29",
    "lower hulls gulch trail": "29",
    "polecat loop": "81",
    "polecat loop trail": "81",
    "around the mountain": "98",
    "around the mountain trail": "98",
    "bucktail": "20A",
    "bucktail trail": "20A",
}
TRAIL_NUMBER_RE = re.compile(r"#\s*([0-9]+[A-Z]?)\b", re.IGNORECASE)
MANUAL_SIGNPOST_NOTES = {
    ("1", "Harrison Hollow"): [
        "At repeated #57 Harrison Hollow / #51 Who Now / #52 Kemper's Ridge junctions, confirm the next signed trail before dropping downhill.",
    ],
    ("1B", "Harrison Hollow"): [
        "Early junction: do not keep climbing #57 Harrison Hollow. Turn toward #52 Kemper's Ridge / #51 Who Now first.",
        "After Kemper's Ridge, take #50 Hippie Shake. Do not drop onto #51 Who Now unless the GPX says you are completing that segment.",
    ],
}
MANUAL_ACCESS_HINTS = {
    ("harrison hollow trailhead", "who now loop trail"): {
        "title": "Start on #57 Harrison Hollow (AWT)",
        "signed_as": ["#57 Harrison Hollow (AWT)"],
        "target": "#51 Who Now Loop",
        "until": "signed junction with #51 Who Now Loop",
        "avoid": ["do not start on unsigned social trails"],
        "detail": (
            "From the car, take #57 Harrison Hollow (AWT) uphill from Harrison Hollow Trailhead, "
            "then use the signed access/connector toward #51 Who Now Loop. #51 does not begin at "
            "the parking lot; this named access trail/road is part of the route even though it is not "
            "official challenge credit."
        ),
    },
    ("harrison hollow trailhead", "who now loop"): {
        "title": "Start on #57 Harrison Hollow (AWT)",
        "signed_as": ["#57 Harrison Hollow (AWT)"],
        "target": "#51 Who Now Loop",
        "until": "signed junction with #51 Who Now Loop",
        "avoid": ["do not start on unsigned social trails"],
        "detail": (
            "From the car, take #57 Harrison Hollow (AWT) uphill from Harrison Hollow Trailhead, "
            "then use the signed access/connector toward #51 Who Now Loop. #51 does not begin at "
            "the parking lot; this named access trail/road is part of the route even though it is not "
            "official challenge credit."
        ),
    },
}

def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:90] or "outing"


def stable_json_sha256(data: dict[str, Any]) -> str:
    """Hash the effective map payload without relying on file formatting."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lookup_text(value: Any) -> str:
    text = str(value or "").replace("’", "'").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def trail_label_from_id_and_name(trail_id: Any, trail_name: Any) -> str:
    name = str(trail_name or "").strip()
    identifier = str(trail_id or "").strip()
    if identifier and name and not name.startswith("#"):
        return f"#{identifier} {name}"
    if name:
        return name
    if identifier:
        return f"#{identifier}"
    return ""


def public_source_label(source_path: Path | None) -> str:
    if not source_path:
        return "in-memory-map-data"
    return source_path.name


def source_metadata_for_map_data(map_data: dict[str, Any], source_path: Path | None = None) -> dict[str, Any]:
    progress = map_data.get("progress") or {}
    metadata = {
        "canonical_data_role": "2026-outing-menu-map-data",
        "source_label": public_source_label(source_path),
        "map_data_sha256": stable_json_sha256(map_data),
        "package_count": len(map_data.get("packages") or []),
        "component_route_count": int((map_data.get("summary") or {}).get("component_route_count") or 0),
        "completed_segment_count_at_export": len(progress.get("completed_segment_ids") or []),
        "blocked_segment_count_at_export": len(progress.get("blocked_segment_ids") or []),
    }
    if source_path and source_path.exists() and source_path.is_file():
        metadata["source_file_sha256"] = file_sha256(source_path)
    return metadata


def normalized_segment_ids(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    return [str(value) for value in values or [] if value is not None]


def extract_map_data_from_html(html: str) -> dict[str, Any]:
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find embedded DATA payload in outing map HTML.")
    return json.loads(match.group(1))


def load_map_data(map_html: Path | None, map_data_json: Path | None) -> tuple[dict[str, Any], Path]:
    if map_data_json and map_data_json.exists():
        return read_json(map_data_json), map_data_json
    explicit_html = map_html and map_html != DEFAULT_MAP_HTML
    if explicit_html and map_html.exists():
        return extract_map_data_from_html(map_html.read_text(encoding="utf-8")), map_html
    if DEFAULT_PUBLIC_MAP_DATA_JSON.exists():
        return read_json(DEFAULT_PUBLIC_MAP_DATA_JSON), DEFAULT_PUBLIC_MAP_DATA_JSON
    if map_html and map_html.exists():
        return extract_map_data_from_html(map_html.read_text(encoding="utf-8")), map_html
    raise FileNotFoundError("No map HTML or map-data JSON source exists for the mobile field packet.")


def known_field_menu_regression_failures(map_data: dict[str, Any]) -> list[str]:
    failures = []
    for package in map_data.get("packages") or []:
        if int(package.get("package_number") or 0) != 1:
            continue
        components = list(package.get("components") or [])
        if len(components) != 1:
            continue
        component = components[0]
        if str(component.get("candidate_id") or "") != "block-hillside_harrison_frontside":
            continue
        if float(component.get("total_minutes") or 0) < 240:
            continue
        failures.append(
            "Package 1 collapsed into block-hillside_harrison_frontside. "
            "Use the canonical executable outing split with 1A West Climb and 1B Harrison Hollow."
        )
    return failures


def validate_field_menu_source(map_data: dict[str, Any]) -> None:
    failures = known_field_menu_regression_failures(map_data)
    if failures:
        raise ValueError("Field menu source regression detected:\n" + "\n".join(failures))


def route_parts(feature: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry = feature.get("geometry") or {}
    geometry_type = geometry.get("type")
    if geometry_type == "LineString":
        raw_parts = [geometry.get("coordinates") or []]
    elif geometry_type == "MultiLineString":
        raw_parts = geometry.get("coordinates") or []
    else:
        return []
    parts: list[list[tuple[float, float]]] = []
    for raw_part in raw_parts:
        part = []
        for coord in raw_part or []:
            if isinstance(coord, list | tuple) and len(coord) >= 2:
                part.append((float(coord[0]), float(coord[1])))
        if len(part) >= 2:
            parts.append(part)
    return parts


def feature_point(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        return None
    coords = geometry.get("coordinates") or []
    if len(coords) < 2:
        return None
    return (float(coords[0]), float(coords[1]))


def midpoint(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not coords:
        return None
    return coords[len(coords) // 2]


def coord_key(coord: tuple[float, float]) -> str:
    return f"{coord[0]:.5f},{coord[1]:.5f}"


def first_point(track_segments: list[list[tuple[float, float]]]) -> tuple[float, float] | None:
    for segment in track_segments:
        if segment:
            return segment[0]
    return None


def last_point(track_segments: list[list[tuple[float, float]]]) -> tuple[float, float] | None:
    for segment in reversed(track_segments):
        if segment:
            return segment[-1]
    return None


def track_distance_miles(track_segments: list[list[tuple[float, float]]] | None) -> float:
    total = 0.0
    for segment in track_segments or []:
        for index in range(1, len(segment)):
            total += haversine_miles(segment[index - 1], segment[index])
    return total


def cumulative_track_points(track_segments: list[list[tuple[float, float]]] | None) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    total = 0.0
    for segment in track_segments or []:
        prior: tuple[float, float] | None = None
        for point in segment:
            point = (float(point[0]), float(point[1]))
            if prior is not None:
                total += haversine_miles(prior, point)
            points.append({"point": point, "mile": total})
            prior = point
    return points


def flatten_track_segments(track_segments: list[list[tuple[float, float]]] | None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for segment in track_segments or []:
        for point in segment:
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    return points


def route_miles_near_point(
    route_points: list[dict[str, Any]],
    point: tuple[float, float],
    tolerance_miles: float = 0.03,
) -> list[float]:
    if not route_points:
        return []
    matches = [
        float(item["mile"])
        for item in route_points
        if haversine_miles(point, item["point"]) <= tolerance_miles
    ]
    if matches:
        return matches
    nearest = min(route_points, key=lambda item: haversine_miles(point, item["point"]))
    if haversine_miles(point, nearest["point"]) <= 0.12:
        return [float(nearest["mile"])]
    return []


def official_endpoint_route_miles(
    segment: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    route_points: list[dict[str, Any]],
) -> list[float]:
    feature = official_index.get(str(segment.get("seg_id") or ""))
    if not feature:
        return []
    miles: list[float] = []
    for part in route_parts(feature):
        if len(part) < 2:
            continue
        miles.extend(route_miles_near_point(route_points, part[0]))
        miles.extend(route_miles_near_point(route_points, part[-1]))
    return miles


def non_credit_gaps_for_cue(
    cue: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    official_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    segments = cue.get("segments") or []
    route_points = cumulative_track_points(track_segments)
    total_miles = route_points[-1]["mile"] if route_points else 0.0
    result = {
        "start_access_gap_miles": 0.0,
        "return_access_gap_miles": 0.0,
    }
    if not segments or not route_points:
        return result
    first_miles = official_endpoint_route_miles(segments[0], official_index, route_points)
    if first_miles:
        result["start_access_gap_miles"] = round(max(0.0, min(first_miles)), 3)
    last_miles = official_endpoint_route_miles(segments[-1], official_index, route_points)
    if last_miles:
        result["return_access_gap_miles"] = round(max(0.0, total_miles - max(last_miles)), 3)
    return result


def bearing_degrees(start: tuple[float, float], end: tuple[float, float]) -> float | None:
    if start == end:
        return None
    lon1 = math.radians(start[0])
    lat1 = math.radians(start[1])
    lon2 = math.radians(end[0])
    lat2 = math.radians(end[1])
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    if x == 0 and y == 0:
        return None
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def signed_bearing_delta(incoming: float, outgoing: float) -> float:
    return (outgoing - incoming + 540) % 360 - 180


def absolute_bearing_delta(first: float, second: float) -> float:
    return abs(signed_bearing_delta(first, second))


def turn_phrase(delta: float) -> str | None:
    abs_delta = abs(delta)
    if abs_delta < 25:
        return "continue straight"
    if abs_delta < 60:
        return "bear right" if delta > 0 else "bear left"
    if abs_delta < 140:
        return "turn right" if delta > 0 else "turn left"
    if abs_delta < 170:
        return "make a sharp right" if delta > 0 else "make a sharp left"
    return "turn around"


def nearby_track_point(
    points: list[tuple[float, float]],
    start_index: int,
    step: int,
    min_distance_miles: float = 0.01,
) -> tuple[float, float] | None:
    if not points:
        return None
    anchor = points[start_index]
    index = start_index + step
    best: tuple[float, float] | None = None
    while 0 <= index < len(points):
        candidate = points[index]
        if candidate != anchor:
            best = candidate
            if haversine_miles(anchor, candidate) >= min_distance_miles:
                return candidate
        index += step
    return best


def endpoint_candidates(parts: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for part in parts:
        if len(part) < 2:
            continue
        candidates.append({"point": part[0], "neighbor": part[1]})
        candidates.append({"point": part[-1], "neighbor": part[-2]})
    return candidates


def transition_geometry(
    prior_group: dict[str, Any],
    next_group: dict[str, Any],
    official_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not official_index:
        return None
    prior_segments = prior_group.get("segments") or []
    next_segments = next_group.get("segments") or []
    if not prior_segments or not next_segments:
        return None
    prior_feature = official_index.get(str(prior_segments[-1].get("seg_id") or ""))
    next_feature = official_index.get(str(next_segments[0].get("seg_id") or ""))
    if not prior_feature or not next_feature:
        return None
    prior_candidates = endpoint_candidates(route_parts(prior_feature))
    next_candidates = endpoint_candidates(route_parts(next_feature))
    if not prior_candidates or not next_candidates:
        return None

    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for prior in prior_candidates:
        for next_candidate in next_candidates:
            distance = haversine_miles(prior["point"], next_candidate["point"])
            if best is None or distance < best[0]:
                best = (distance, prior, next_candidate)
    if best is None:
        return None
    _, prior, next_candidate = best
    midpoint_point = (
        (prior["point"][0] + next_candidate["point"][0]) / 2,
        (prior["point"][1] + next_candidate["point"][1]) / 2,
    )
    return {
        "point": midpoint_point,
        "next_bearing": bearing_degrees(next_candidate["point"], next_candidate["neighbor"]),
    }


def turn_phrase_for_transition(
    prior_group: dict[str, Any],
    next_group: dict[str, Any],
    track_segments: list[list[tuple[float, float]]] | None,
    official_index: dict[str, dict[str, Any]] | None,
) -> str | None:
    transition = transition_geometry(prior_group, next_group, official_index)
    points = flatten_track_segments(track_segments)
    if not transition or len(points) < 3:
        return None

    target = transition["point"]
    nearest_index = min(range(len(points)), key=lambda index: haversine_miles(target, points[index]))
    if haversine_miles(target, points[nearest_index]) > 0.12:
        return None

    before = nearby_track_point(points, nearest_index, -1)
    after = nearby_track_point(points, nearest_index, 1)
    if not before or not after:
        return None
    incoming = bearing_degrees(before, points[nearest_index])
    outgoing = bearing_degrees(points[nearest_index], after)
    if incoming is None or outgoing is None:
        return None

    expected_next_bearing = transition.get("next_bearing")
    if expected_next_bearing is not None and absolute_bearing_delta(outgoing, expected_next_bearing) > 70:
        return None
    return turn_phrase(signed_bearing_delta(incoming, outgoing))


def densify_segment(
    coords: list[tuple[float, float]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    dense = [coords[0]]
    target_gap = max_gap_miles * 0.75
    for left, right in zip(coords, coords[1:]):
        gap = haversine_miles(left, right)
        steps = max(1, math.ceil(gap / target_gap)) if target_gap > 0 else 1
        for step in range(1, steps + 1):
            ratio = step / steps
            dense.append(
                (
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                )
            )
    return dense


def densify_track_segments(
    track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> list[list[tuple[float, float]]]:
    return [densify_segment(segment, max_gap_miles=max_gap_miles) for segment in track_segments]


def load_default_connector_graph() -> dict[str, Any] | None:
    try:
        official_segments, _metadata = load_official_segments(DEFAULT_OFFICIAL_GEOJSON)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None
    return load_connector_graph(DEFAULT_CONNECTOR_GEOJSON, official_segments=official_segments)


def append_deduped_track_point(target: list[tuple[float, float]], coord: Any) -> None:
    point = (float(coord[0]), float(coord[1]))
    if target and haversine_miles(target[-1], point) < 0.000001:
        return
    target.append(point)


def stitch_inter_segment_track_gaps(
    track_segments: list[list[tuple[float, float]]],
    connector_graph: dict[str, Any] | None,
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
    stitch_snap_tolerance_miles: float = 0.03,
) -> list[list[tuple[float, float]]]:
    """Insert explicit graph connector geometry between GPX parts when possible."""

    if not track_segments or not connector_graph:
        return track_segments
    stitched: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for segment in track_segments:
        if not segment:
            continue
        if not current:
            current = list(segment)
            continue
        gap = haversine_miles(current[-1], segment[0])
        if gap <= max_gap_miles:
            for point in segment:
                append_deduped_track_point(current, point)
            continue
        stitch = shortest_connector_path(
            current[-1],
            segment[0],
            connector_graph,
            stitch_snap_tolerance_miles,
        )
        if not stitch:
            stitched.append(current)
            current = list(segment)
            continue
        for point in stitch.get("path_coordinates") or []:
            append_deduped_track_point(current, point)
        for point in segment:
            append_deduped_track_point(current, point)
    if current:
        stitched.append(current)
    return stitched


def inter_segment_gap_count(
    track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> int:
    count = 0
    for left, right in zip(track_segments, track_segments[1:]):
        if left and right and haversine_miles(left[-1], right[0]) > max_gap_miles:
            count += 1
    return count


def source_gap_repair_summary(
    raw_track_segments: list[list[tuple[float, float]]],
    stitched_track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
) -> dict[str, Any]:
    raw_gap_count = inter_segment_gap_count(raw_track_segments, max_gap_miles=max_gap_miles)
    remaining_gap_count = inter_segment_gap_count(stitched_track_segments, max_gap_miles=max_gap_miles)
    repaired_count = max(0, raw_gap_count - remaining_gap_count)
    return {
        "raw_inter_segment_gap_count": raw_gap_count,
        "repaired_inter_segment_gap_count": repaired_count,
        "remaining_inter_segment_gap_count": remaining_gap_count,
        "repair_method": "graph_connector_stitch" if repaired_count else None,
    }


def indexed_features(
    map_data: dict[str, Any],
    collection_name: str,
    property_name: str,
) -> dict[str, list[dict[str, Any]]]:
    collection = (map_data.get("feature_collections") or {}).get(collection_name) or {}
    index: dict[str, list[dict[str, Any]]] = {}
    for feature in collection.get("features") or []:
        value = (feature.get("properties") or {}).get(property_name)
        if value is None:
            continue
        index.setdefault(str(value), []).append(feature)
    return index


def official_segment_index(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    collection = (map_data.get("feature_collections") or {}).get("official_segments") or {}
    index = {}
    for feature in collection.get("features") or []:
        props = feature.get("properties") or {}
        segment_id = props.get("seg_id") or props.get("segment_id")
        if segment_id is not None:
            index[str(segment_id)] = feature
    return index


def load_trailhead_access_index(path: Path = DEFAULT_TRAILHEAD_CANDIDATES) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = read_json(path)
    index: dict[str, dict[str, Any]] = {}
    for feature in data.get("features") or []:
        props = feature.get("properties") or {}
        point = feature_point(feature)
        name = props.get("facility_name") or props.get("name") or props.get("trailhead")
        open_label = trail_label_from_id_and_name(
            props.get("nearest_open_trail_id"),
            props.get("nearest_open_trail_name"),
        )
        metadata = {
            "nearest_open_trail_id": props.get("nearest_open_trail_id"),
            "nearest_open_trail_name": props.get("nearest_open_trail_name"),
            "nearest_open_trail_label": open_label,
            "nearest_open_trail_distance_miles": props.get("nearest_open_trail_distance_miles"),
            "nearest_official_trail_name": props.get("nearest_official_trail_name"),
            "nearest_official_segment_id": props.get("nearest_official_segment_id"),
            "nearest_official_distance_miles": props.get("nearest_official_distance_miles"),
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "")}
        if not metadata:
            continue
        if name:
            index[lookup_text(name)] = metadata
        if point:
            index[f"{point[1]:.6f},{point[0]:.6f}"] = metadata
    return index


def trailhead_access_for_parking(
    parking: dict[str, Any],
    trailhead_access_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not trailhead_access_index:
        return {}
    name = parking.get("name")
    if name:
        by_name = trailhead_access_index.get(lookup_text(name))
        if by_name:
            return by_name
    if parking.get("lat") is not None and parking.get("lon") is not None:
        return trailhead_access_index.get(f"{float(parking['lat']):.6f},{float(parking['lon']):.6f}", {})
    return {}


def enrich_parking_with_trailhead_access(
    parking: dict[str, Any],
    trailhead_access_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    metadata = trailhead_access_for_parking(parking, trailhead_access_index)
    if not metadata:
        return parking
    enriched = dict(parking)
    for key, value in metadata.items():
        if enriched.get(key) in (None, ""):
            enriched[key] = value
    return enriched


def parking_for_outing(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    parking_by_candidate: dict[str, list[dict[str, Any]]],
    trailhead_access_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        trailhead = cue.get("trailhead") or {}
        if trailhead.get("lat") is not None and trailhead.get("lon") is not None:
            parking = {
                "name": trailhead.get("name") or outing.get("trailhead") or "Parking/start",
                "lon": float(trailhead["lon"]),
                "lat": float(trailhead["lat"]),
                "has_parking": trailhead.get("has_parking"),
                "has_restroom": trailhead.get("has_restroom"),
                "has_water": trailhead.get("has_water"),
                "water_confidence": trailhead.get("water_confidence"),
                "nearest_open_trail_name": trailhead.get("nearest_open_trail_name"),
                "nearest_open_trail_label": trailhead.get("nearest_open_trail_label"),
            }
            return enrich_parking_with_trailhead_access(parking, trailhead_access_index)
        for feature in parking_by_candidate.get(str(candidate_id), []):
            point = feature_point(feature)
            if point:
                props = feature.get("properties") or {}
                parking = {
                    "name": props.get("name") or props.get("trailhead") or outing.get("trailhead") or "Parking/start",
                    "lon": point[0],
                    "lat": point[1],
                    "has_parking": props.get("has_parking"),
                    "has_restroom": props.get("has_restroom"),
                    "has_water": props.get("has_water"),
                    "water_confidence": props.get("water_confidence"),
                    "nearest_open_trail_name": props.get("nearest_open_trail_name"),
                    "nearest_open_trail_label": props.get("nearest_open_trail_label"),
                }
                return enrich_parking_with_trailhead_access(parking, trailhead_access_index)
    return None


def connector_failures_for_outing(outing: dict[str, Any], route_cues: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        links = list(cue.get("between_links") or [])
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            links.append(return_to_car)
        for link in links:
            values = []
            values.extend(str(item).lower() for item in link.get("connector_classes") or [])
            values.extend(str(item).lower() for item in link.get("connector_names") or [])
            joined = " ".join(values)
            if any(token in joined for token in BLOCKED_CONNECTOR_TOKENS):
                failures.append({"code": "blocked_connector_class", "candidate_id": str(candidate_id), "value": joined})
    return failures


def link_declares_field_gap(link: dict[str, Any]) -> bool:
    if not link:
        return False
    if link.get("intentional_repark") or link.get("manual_day_of_access_hold"):
        return True
    if link.get("connector_names") or link.get("signpost_labels"):
        return True
    if link.get("connector_miles") or link.get("road_miles") or link.get("official_repeat_miles"):
        return True
    classes = {str(item) for item in link.get("connector_classes") or []}
    return bool(classes & {"r2r_trail", "osm_path_footway", "osm_public_road", "official_repeat"})


def link_declares_track_break(link: dict[str, Any]) -> bool:
    if not link:
        return False
    return bool(
        link.get("intentional_repark")
        or link.get("multi_start_boundary")
        or link.get("manual_day_of_access_hold")
    )


def declared_gap_links_for_outing(outing: dict[str, Any], route_cues: dict[str, Any]) -> list[dict[str, Any]]:
    declared = []
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        for link in cue.get("between_links") or []:
            if link_declares_track_break(link):
                declared.append(link)
        return_to_car = cue.get("return_to_car") or {}
        if link_declares_track_break(return_to_car):
            declared.append(return_to_car)
    return declared


def track_segments_for_outing(
    outing: dict[str, Any],
    routes_by_candidate: dict[str, list[dict[str, Any]]],
) -> list[list[tuple[float, float]]]:
    segments = []
    for candidate_id in outing.get("candidate_ids") or []:
        for feature in routes_by_candidate.get(str(candidate_id), []):
            segments.extend(route_parts(feature))
    return segments


def turn_waypoints(track_segments: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    waypoints = []
    seen = set()
    for segment in track_segments:
        for index in range(1, len(segment) - 1):
            if coord_key(segment[index - 1]) == coord_key(segment[index + 1]):
                key = coord_key(segment[index])
                if key in seen:
                    continue
                seen.add(key)
                waypoints.append(
                    {
                        "name": "TURN",
                        "description": "Double-back or return point. Follow the phone card and route line.",
                        "lon": segment[index][0],
                        "lat": segment[index][1],
                    }
                )
    return waypoints


def segment_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    waypoints = []
    seen = set()
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        for segment in cue.get("segments") or []:
            segment_id = str(segment.get("seg_id"))
            if not segment_id or segment_id in seen:
                continue
            feature = official_index.get(segment_id)
            if not feature:
                continue
            parts = route_parts(feature)
            point = midpoint(parts[0]) if parts else None
            if not point:
                continue
            seen.add(segment_id)
            direction = str(segment.get("direction_rule") or "").lower()
            prefix = "ASCENT" if direction == "ascent" else "SEG"
            order = segment.get("order") or len(waypoints) + 1
            name = f"{prefix} {order} {segment.get('segment_name') or segment.get('trail_name') or segment_id}"
            description = (
                f"{segment.get('direction_cue') or 'Follow route direction.'} "
                f"Official {format_miles(segment.get('official_miles'))} mi."
            )
            effort = segment_effort_sentence(segment)
            if effort:
                description += f" {effort}."
            signpost = signpost_sentence([segment.get("trail_name")], prefix="Signpost")
            if signpost:
                description += f" {signpost}."
            waypoints.append({"name": name, "description": description.strip(), "lon": point[0], "lat": point[1]})
    return waypoints


def cue_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    waypoints = []
    last_trail_key = None
    cue_number = 1
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        for segment in cue.get("segments") or []:
            trail_name = segment.get("trail_name") or "route"
            trail_key = signpost_key(trail_name) or normalized_trail_text(trail_name).lower()
            if trail_key == last_trail_key:
                continue
            segment_id = str(segment.get("seg_id"))
            feature = official_index.get(segment_id)
            if not feature:
                continue
            parts = route_parts(feature)
            point = midpoint(parts[0]) if parts else None
            if not point:
                continue
            signpost = signpost_label(trail_name)
            cue_label = signpost or normalized_trail_text(trail_name) or "route"
            direction = str(segment.get("direction_rule") or "").lower()
            direction_note = segment.get("direction_cue") or "Follow the GPX line."
            if direction == "ascent":
                direction_note = "ASCENT REQUIRED. " + direction_note
            name = f"CUE {cue_number:02d} {cue_label}"
            signpost_note = f" Signpost: {signpost}." if signpost else ""
            description = (
                f"Follow {cue_label}. First official segment here: "
                f"{segment.get('segment_name') or trail_name}.{signpost_note} {direction_note} "
                "This is a navigation cue, not the full segment-credit audit."
            )
            waypoints.append({"name": name, "description": description.strip(), "lon": point[0], "lat": point[1]})
            last_trail_key = trail_key
            cue_number += 1
    return waypoints


def build_navigation_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    parking: dict[str, Any] | None,
    track_segments: list[list[tuple[float, float]]],
    logistics: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    waypoints = []
    if parking:
        waypoints.append(
            {
                "name": f"PARK/START {parking['name']}",
                "description": "Park here and start this outing.",
                "lon": parking["lon"],
                "lat": parking["lat"],
            }
        )
    if logistics:
        waypoints.extend(logistics_waypoints(logistics))
    waypoints.extend(cue_waypoints(outing, route_cues, official_index))
    final = last_point(track_segments)
    if final:
        waypoints.append(
            {
                "name": "RETURN TO CAR",
                "description": "Route endpoint / return-to-car point.",
                "lon": final[0],
                "lat": final[1],
            }
        )
    return waypoints


def build_audit_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    parking: dict[str, Any] | None,
    track_segments: list[list[tuple[float, float]]],
    logistics: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    waypoints = []
    if parking:
        waypoints.append(
            {
                "name": f"PARK/START {parking['name']}",
                "description": "Park here and start this outing.",
                "lon": parking["lon"],
                "lat": parking["lat"],
            }
        )
    if logistics:
        waypoints.extend(logistics_waypoints(logistics))
    waypoints.extend(turn_waypoints(track_segments))
    final = last_point(track_segments)
    if final:
        waypoints.append(
            {
                "name": "RETURN TO CAR",
                "description": "Route endpoint / return-to-car point.",
                "lon": final[0],
                "lat": final[1],
            }
        )
    waypoints.extend(segment_waypoints(outing, route_cues, official_index))
    return waypoints


def gpx_description(outing: dict[str, Any]) -> str:
    return (
        f"{outing.get('label')}. {outing.get('trailhead')} | "
        f"Official {format_miles(outing.get('official_miles'))} mi; "
        f"On-foot {format_miles(outing.get('on_foot_miles'))} mi; "
        f"Door-to-door p75 {format_minutes(outing.get('total_minutes'))}; "
        f"Segments left {outing.get('remaining_segment_count')} / {len(outing.get('segment_ids') or [])}. "
        "Check current Ridge to Rivers signage and conditions before leaving."
    )


def segment_effort_sentence(segment: dict[str, Any]) -> str:
    parts = []
    if segment.get("estimated_moving_minutes"):
        parts.append(f"est. {format_minutes(segment.get('estimated_moving_minutes'))}")
    if segment.get("estimated_moving_minutes_p75"):
        parts.append(f"p75 {format_minutes(segment.get('estimated_moving_minutes_p75'))}")
    if segment.get("ascent_ft"):
        parts.append(f"{int(round(float(segment.get('ascent_ft') or 0)))} ft climb")
    return "; ".join(parts)


def render_gpx(
    name: str,
    description: str,
    track_segments: list[list[tuple[float, float]]],
    waypoints: list[dict[str, Any]],
) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="boise-trails-ai" xmlns="http://www.topografix.com/GPX/1/1">',
        "  <metadata>",
        f"    <name>{escape(name)}</name>",
        f"    <desc>{escape(description)}</desc>",
        "  </metadata>",
    ]
    for waypoint in waypoints:
        lines.extend(
            [
                f'  <wpt lat="{waypoint["lat"]:.6f}" lon="{waypoint["lon"]:.6f}">',
                f"    <name>{escape(str(waypoint['name']))}</name>",
                f"    <desc>{escape(str(waypoint.get('description') or ''))}</desc>",
                "  </wpt>",
            ]
        )
    if track_segments:
        lines.extend(["  <trk>", f"    <name>{escape(name)}</name>", f"    <desc>{escape(description)}</desc>"])
        for coords in track_segments:
            lines.append("    <trkseg>")
            for lon, lat in coords:
                lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}" />')
            lines.append("    </trkseg>")
        lines.append("  </trk>")
    lines.extend(["</gpx>", ""])
    return "\n".join(lines)


def parking_navigation_url(parking: dict[str, Any] | None) -> str | None:
    if not parking:
        return None
    return f"https://www.google.com/maps/dir/?api=1&destination={parking['lat']:.6f},{parking['lon']:.6f}"


def validate_outing_export(
    outing: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    parking: dict[str, Any] | None,
    route_cues: dict[str, Any],
    max_gap_miles: float,
    max_parking_gap_miles: float,
) -> dict[str, Any]:
    validation = validate_track_segments(track_segments, max_gap_miles=max_gap_miles)
    failures = list(validation.get("failures") or [])
    declared_gap_links = declared_gap_links_for_outing(outing, route_cues)
    declared_gap_index = 0
    declared_gap_count = 0
    for segment_index, (left, right) in enumerate(zip(track_segments, track_segments[1:])):
        if not left or not right:
            continue
        gap = haversine_miles(left[-1], right[0])
        if gap > max_gap_miles:
            if declared_gap_index < len(declared_gap_links):
                declared_gap_index += 1
                declared_gap_count += 1
                continue
            failures.append(
                {
                    "code": "unexplained_inter_segment_gap",
                    "track_segment_index": segment_index,
                    "gap_miles": round(gap, 4),
                    "max_allowed_gap_miles": max_gap_miles,
                }
            )
    if parking:
        parking_point = (parking["lon"], parking["lat"])
        start = first_point(track_segments)
        end = last_point(track_segments)
        start_gap = haversine_miles(parking_point, start) if start else None
        end_gap = haversine_miles(parking_point, end) if end else None
        if start_gap is None or start_gap > max_parking_gap_miles:
            failures.append(
                {
                    "code": "start_not_near_parking",
                    "gap_miles": round(start_gap or 0, 4),
                    "max_allowed_gap_miles": max_parking_gap_miles,
                }
            )
        if end_gap is None or end_gap > max_parking_gap_miles:
            failures.append(
                {
                    "code": "end_not_near_parking",
                    "gap_miles": round(end_gap or 0, 4),
                    "max_allowed_gap_miles": max_parking_gap_miles,
                }
            )
    else:
        failures.append({"code": "missing_parking"})
    failures.extend(connector_failures_for_outing(outing, route_cues))
    return {
        **validation,
        "passed": not failures,
        "failures": failures,
        "max_allowed_parking_gap_miles": max_parking_gap_miles,
        "declared_inter_segment_gap_count": declared_gap_count,
    }


def html_escape(value: Any) -> str:
    return escape(str(value or ""), {'"': "&quot;"})


def summarized_names(values: list[Any], limit: int = 8) -> str:
    names = unique_nonempty_text(values)
    if not names:
        return ""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", +{len(names) - limit} more"


def unique_nonempty_text(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalized_trail_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("’", "'").strip())


def signpost_key(value: Any) -> str:
    text = normalized_trail_text(value).lower()
    text = TRAIL_NUMBER_RE.sub("", text)
    text = re.sub(r"\btrail\s*(\d+)?\b", "trail", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—,:;")
    return text


def signpost_label(value: Any) -> str:
    text = normalized_trail_text(value)
    if not text:
        return ""
    number_match = TRAIL_NUMBER_RE.search(text)
    number = number_match.group(1).upper() if number_match else SIGNPOST_TRAIL_NUMBERS.get(signpost_key(text))
    if not number:
        return ""
    name = TRAIL_NUMBER_RE.sub("", text).strip(" -–—,:;")
    if not name:
        return f"#{number}"
    return f"#{number} {name}"


def signpost_labels(values: list[Any]) -> list[str]:
    labels = []
    seen = set()
    for value in values:
        label = signpost_label(value)
        key = re.sub(r"\btrail\b", "", label.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def signpost_sentence(values: list[Any], prefix: str = "Signpost cues") -> str:
    labels = signpost_labels(values)
    if not labels:
        return ""
    return f"{prefix}: {'; '.join(labels)}"


def aggregate_logistics(
    cues: list[dict[str, Any]],
    parking: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    car_passes: list[dict[str, Any]] = []
    known_water: list[dict[str, Any]] = []
    for cue in cues:
        logistics = cue.get("logistics") or {}
        car_passes.extend(logistics.get("car_passes") or [])
        known_water.extend(logistics.get("known_water") or [])
    trailhead_names = {
        str((cue.get("trailhead") or {}).get("name"))
        for cue in cues
        if (cue.get("trailhead") or {}).get("name")
    }
    if len(cues) > 1 and len(trailhead_names) == 1 and parking:
        car_passes.insert(
            0,
            {
                "name": "Back at car between route components",
                "inter_component": True,
                "mile_from_start": None,
                "distance_to_car_miles": 0,
                "lon": parking["lon"],
                "lat": parking["lat"],
            },
        )
    if parking and parking.get("has_water") is True:
        known_water.append(
            {
                "name": parking.get("name") or "Parking/start",
                "location": "parking/start",
                "confidence": parking.get("water_confidence") or "verified",
                "lon": parking["lon"],
                "lat": parking["lat"],
            }
        )
    deduped_water = []
    seen_water = set()
    for item in known_water:
        key = (
            str(item.get("name") or ""),
            str(item.get("location") or ""),
            round(float(item.get("lon") or 0), 5),
            round(float(item.get("lat") or 0), 5),
        )
        if key in seen_water:
            continue
        seen_water.add(key)
        deduped_water.append(item)
    return {"car_passes": car_passes, "known_water": deduped_water}


def car_pass_sentence(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No mid-route car pass detected."
    pieces = []
    for item in items:
        if item.get("inter_component"):
            pieces.append("Back at car between route components.")
        else:
            pieces.append(f"Pass by car again near mile {format_miles(item.get('mile_from_start'))}.")
    return " ".join(pieces)


def water_sentence(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    return "; ".join(
        f"{item.get('name') or 'Water'} · {item.get('location') or 'location'} · {item.get('confidence') or 'verified'}"
        for item in items
    )


def logistics_section_html(logistics: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    car_passes = logistics.get("car_passes") or []
    known_water = logistics.get("known_water") or []
    if car_passes:
        lines.append(f"<b>Car:</b> {html_escape(car_pass_sentence(car_passes))}")
    if known_water:
        lines.append(f"<b>Known water:</b> {html_escape(water_sentence(known_water))}")
    if not lines:
        return ""
    return f"<section><h3>Field logistics</h3><p>{'<br>'.join(lines)}</p></section>"


def logistics_waypoints(logistics: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    waypoints = []
    for index, car_pass in enumerate(logistics.get("car_passes") or [], start=1):
        if car_pass.get("lon") is None or car_pass.get("lat") is None:
            continue
        description = (
            "Back at the parked car between route components."
            if car_pass.get("inter_component")
            else f"Pass by the parked car again near mile {format_miles(car_pass.get('mile_from_start'))}."
        )
        waypoints.append(
            {
                "name": f"CAR PASS {index}",
                "description": description,
                "lon": float(car_pass["lon"]),
                "lat": float(car_pass["lat"]),
            }
        )
    for water in logistics.get("known_water") or []:
        if water.get("lon") is None or water.get("lat") is None:
            continue
        waypoints.append(
            {
                "name": f"WATER {water.get('name') or 'Water'}",
                "description": (
                    f"Known water: {water.get('name') or 'Water'}; "
                    f"{water.get('location') or 'location'}; {water.get('confidence') or 'verified'}."
                ),
                "lon": float(water["lon"]),
                "lat": float(water["lat"]),
            }
        )
    return waypoints


def manual_signpost_notes(route: dict[str, Any]) -> list[str]:
    outing = route.get("outing") or {}
    key = (str(outing.get("label") or ""), str(outing.get("trailhead") or ""))
    return MANUAL_SIGNPOST_NOTES.get(key, [])


def connector_detail(link: dict[str, Any]) -> str:
    pieces = []
    distance = link.get("distance_miles")
    raw_connector_names = unique_nonempty_text([str(name) for name in link.get("connector_names") or []])
    connector_names = summarized_names(raw_connector_names)
    distance_miles = float(distance or 0)
    if len(raw_connector_names) > 2 and distance_miles:
        pieces.append(f"Use the named connector trail/road for {format_miles(distance)} mi")
    elif connector_names and distance_miles:
        pieces.append(f"Use {connector_names} for {format_miles(distance)} mi")
    elif connector_names:
        pieces.append(f"Use {connector_names}")
    elif distance_miles > 0.02:
        pieces.append(f"Short link: {format_miles(distance)} mi")
    signpost = signpost_sentence(
        link.get("connector_names") or [link.get("from_trail"), link.get("to_trail")],
        prefix="Look for signs",
    )
    if signpost:
        pieces.append(signpost)
    return ". ".join(pieces) + ("." if pieces else "")


def display_trail(value: Any) -> str:
    return signpost_label(value) or normalized_trail_text(value) or "the next trail"


def plain_trail(value: Any) -> str:
    return normalized_trail_text(value) or "the previous trail"


def sentence_list(values: list[str]) -> str:
    clean = unique_nonempty_text(values)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def compact_number_list(numbers: list[str]) -> str:
    if not numbers:
        return ""
    if len(numbers) > 2 and all(number.isdigit() for number in numbers):
        ints = [int(number) for number in numbers]
        if ints == list(range(ints[0], ints[-1] + 1)):
            return f"{ints[0]}-{ints[-1]}"
    return sentence_list(numbers)


def segment_credit_label(segments: list[dict[str, Any]]) -> str:
    names = [str(segment.get("segment_name") or segment.get("trail_name") or "") for segment in segments]
    trail_names = unique_nonempty_text([str(segment.get("trail_name") or "") for segment in segments])
    if len(trail_names) == 1:
        trail_name = trail_names[0]
        numbers = []
        for name in names:
            match = re.search(r"(\d+)$", name.strip())
            if not match:
                break
            base = name[: match.start()].strip()
            if normalized_trail_text(base) != normalized_trail_text(trail_name):
                break
            numbers.append(match.group(1))
        else:
            if numbers:
                if len(numbers) == 1:
                    return f"{trail_name} segment {compact_number_list(numbers)}"
                if len(numbers) == 2:
                    return f"both {trail_name} official segments"
                return f"{trail_name} segments {compact_number_list(numbers)}"
    return sentence_list(names)


def trail_groups(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for segment in segments:
        trail_name = segment.get("trail_name") or "official trail"
        if groups and groups[-1]["trail_name"] == trail_name:
            groups[-1]["segments"].append(segment)
        else:
            groups.append({"trail_name": trail_name, "segments": [segment]})
    return groups


def segment_completion_sentence(segments: list[dict[str, Any]], *, final_group: bool) -> str:
    credit = segment_credit_label(segments)
    if not credit:
        return ""
    return f"This earns: {credit}."


def group_effort_sentence(segments: list[dict[str, Any]]) -> str:
    official = sum(float(segment.get("official_miles") or 0) for segment in segments)
    ascent = sum(float(segment.get("ascent_ft") or 0) for segment in segments)
    minutes = sum(float(segment.get("estimated_moving_minutes_p75") or segment.get("estimated_moving_minutes") or 0) for segment in segments)
    parts = []
    if official:
        parts.append(f"{format_miles(official)} official mi")
    if minutes:
        parts.append(f"~{round(minutes)} min moving")
    if ascent:
        parts.append(f"{round(ascent)} ft climb")
    if not parts:
        return ""
    return "Section estimate: " + ", ".join(parts)


def ascent_warning_sentence(segments: list[dict[str, Any]]) -> str:
    ascent_segments = [
        str(segment.get("segment_name") or segment.get("trail_name") or "")
        for segment in segments
        if str(segment.get("direction_rule") or "").lower() == "ascent"
    ]
    if not ascent_segments:
        return ""
    return f"ASCENT REQUIRED on {sentence_list(ascent_segments)}."


def link_for_group_transition(links: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return links[index] if index < len(links) else {}


def turn_title(phrase: str | None, trail_label: str) -> str:
    if not phrase:
        return f"Turn onto {trail_label}"
    if phrase == "continue straight":
        return f"Continue onto {trail_label}"
    return f"{phrase.capitalize()} onto {trail_label}"


def turn_detail_action(phrase: str | None, trail_label: str) -> str:
    if not phrase:
        return f"turn onto {trail_label}"
    if phrase == "continue straight":
        return f"continue straight onto {trail_label}"
    return f"{phrase} onto {trail_label}"


def trail_navigation_steps_for_cue(
    cue: dict[str, Any],
    parking: dict[str, Any],
    track_segments: list[list[tuple[float, float]]] | None = None,
    official_index: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    segments = cue.get("segments") or []
    groups = trail_groups(segments)
    if not groups:
        return []

    steps: list[dict[str, str]] = []
    links = list(cue.get("between_links") or [])
    trailhead_name = parking.get("name") or (cue.get("trailhead") or {}).get("name") or "the car"

    for index, group in enumerate(groups):
        trail_name = group["trail_name"]
        trail_label = display_trail(trail_name)
        is_first = index == 0
        is_last = index == len(groups) - 1
        next_group = groups[index + 1] if not is_last else None
        prior_group = groups[index - 1] if not is_first else None
        detail_parts: list[str] = []

        if is_first:
            title = f"Take {trail_label}"
            detail_parts.append(f"Follow {trail_label}.")
        else:
            link = link_for_group_transition(links, index - 1)
            from_trail = link.get("from_trail") or (prior_group or {}).get("trail_name") or "previous trail"
            to_trail = link.get("to_trail") or trail_name
            from_plain = plain_trail(from_trail)
            to_plain = plain_trail(to_trail)
            to_label = display_trail(to_trail)
            phrase = turn_phrase_for_transition(prior_group or {}, group, track_segments, official_index)
            title = turn_title(phrase, to_label)
            action = turn_detail_action(phrase, to_label)
            detail_parts.append(f"At the signed junction with {to_plain}, {action}.")
            link_detail = connector_detail(link)
            if link_detail:
                detail_parts.append(link_detail)

        completion = segment_completion_sentence(group["segments"], final_group=is_last)
        if completion:
            detail_parts.append(completion)
        effort = group_effort_sentence(group["segments"])
        if effort:
            detail_parts.append(effort + ".")
        ascent_warning = ascent_warning_sentence(group["segments"])
        if ascent_warning:
            detail_parts.append(ascent_warning)
        if next_group:
            next_label = display_trail(next_group["trail_name"])
            detail_parts.append(f"Next decision: look for signs to {next_label}.")

        steps.append({"kind": "navigate", "title": title, "detail": " ".join(detail_parts)})

    return steps


def access_navigation_step(
    parking: dict[str, Any],
    first_trail: Any,
    access: dict[str, Any],
    start_access_gap_miles: float = 0.0,
) -> dict[str, str]:
    first_trail_label = display_trail(first_trail)
    first_key = signpost_key(first_trail) or lookup_text(first_trail)
    trailhead_key = lookup_text(parking.get("name"))
    manual_hint = MANUAL_ACCESS_HINTS.get((trailhead_key, first_key))
    if manual_hint:
        return {
            "kind": "access",
            "title": manual_hint["title"],
            "detail": manual_hint["detail"],
        }

    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    if start_label and signpost_key(start_label) != first_key:
        access_miles = max(float(access.get("mapped_access_miles") or 0), float(start_access_gap_miles or 0))
        detail = (
            f"From the car, take {start_label} toward {first_trail_label}. "
            "This access trail/road is part of the route even though it is not official challenge credit."
        )
        if access_miles > 0.05:
            detail += f" Follow the GPX access line for about {format_miles(access_miles)} mi."
        first_signpost = signpost_sentence([start_label, first_trail], prefix="Watch for signs")
        if first_signpost:
            detail += " " + first_signpost + "."
        return {
            "kind": "access",
            "title": f"Start on {start_label}",
            "detail": detail,
        }

    access_detail = f"From the car, head toward {first_trail_label}."
    access_miles = max(float(access.get("mapped_access_miles") or 0), float(start_access_gap_miles or 0))
    if access_miles > 0.05:
        access_detail += (
            f" Follow the GPX access line for about {format_miles(access_miles)} mi before official credit starts."
        )
    first_signpost = signpost_sentence([first_trail], prefix="Look for signs")
    if first_signpost:
        access_detail += " " + first_signpost + "."
    return {
        "kind": "access",
        "title": f"Leave car toward {first_trail_label}",
        "detail": access_detail,
    }


def return_navigation_step(
    parking: dict[str, Any],
    last_trail: Any,
    return_to_car: dict[str, Any],
    return_access_gap_miles: float = 0.0,
) -> dict[str, str]:
    last_key = signpost_key(last_trail) or lookup_text(last_trail)
    last_label = display_trail(last_trail)
    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    return_names = summarized_names(return_to_car.get("connector_names") or [])
    access_miles = max(
        float(return_access_gap_miles or 0),
        float(return_to_car.get("connector_miles") or 0),
        float(return_to_car.get("road_miles") or 0),
    )

    if access_miles > 0.05 or return_names:
        if return_names:
            title = f"Return via {return_names}"
        elif start_label and signpost_key(start_label) != last_key:
            title = f"Return via {start_label}"
        else:
            title = "Follow connector/access back to car"
        parking_name = parking.get("name") or "the parking point"
        detail = f"After {last_label}, you are not fully back at the car. "
        if return_names:
            detail += f"Use {return_names} back toward {parking_name}."
        elif start_label and signpost_key(start_label) != last_key:
            detail += f"Follow the signed connector/access back toward {start_label} and descend to {parking_name}."
        else:
            detail += f"Follow the GPX connector/access line back to {parking_name}."
        if access_miles > 0.05:
            detail += f" Expect about {format_miles(access_miles)} mi after the final official-credit segment."
        detail += " This return leg is part of the route even though it is not official challenge credit."
        return {"kind": "return", "title": title, "detail": detail}

    return_names = summarized_names(return_to_car.get("connector_names") or [])
    detail = return_to_car.get("description") or "Follow the GPX line back to the car."
    if "geometry tolerance" in detail:
        detail = "You should be back at the parking point."
    if return_names:
        detail += f" Return via {return_names}."
    return_parts = []
    if float(return_to_car.get("official_repeat_miles") or 0):
        return_parts.append(f"{format_miles(return_to_car.get('official_repeat_miles'))} mi repeat official")
    if float(return_to_car.get("connector_miles") or 0):
        return_parts.append(f"{format_miles(return_to_car.get('connector_miles'))} mi connector")
    if float(return_to_car.get("road_miles") or 0):
        return_parts.append(f"{format_miles(return_to_car.get('road_miles'))} mi road")
    if return_parts:
        detail += f" Return leg: {sentence_list(return_parts)}."
    return {"kind": "return", "title": "Return to car", "detail": detail}


WAYFINDING_TYPE_LABELS = {
    "start_access": "START/ACCESS",
    "official_segment_start": "OFFICIAL START",
    "follow_official_segment": "FOLLOW",
    "junction_turn": "JCT",
    "connector_named_trail": "CONNECTOR",
    "connector_road": "ROAD",
    "repeat_official_noncredit": "REPEAT",
    "exit_access": "EXIT",
    "return_to_car": "RETURN",
    "manual_field_check": "HOLD",
}


def cue_type_label(cue_type: str) -> str:
    return WAYFINDING_TYPE_LABELS.get(str(cue_type or ""), str(cue_type or "CUE").upper())


def cue_signed_text(values: list[Any]) -> str:
    return " / ".join(unique_nonempty_text(values))


def wayfinding_compact(cue: dict[str, Any]) -> str:
    signed = cue_signed_text(cue.get("signed_as") or [])
    label = cue_type_label(str(cue.get("cue_type") or "cue"))
    action = str(cue.get("action") or "FOLLOW").upper()
    show_action = "" if action == label else action
    if signed and show_action:
        signed_part = f" {show_action} {signed}"
    elif signed:
        signed_part = f" {signed}"
    elif show_action:
        signed_part = f" {show_action}"
    else:
        signed_part = ""
    until = f" UNTIL {cue.get('until')}" if cue.get("until") else ""
    return (
        f"{int(cue.get('seq') or 0):02d} {float(cue.get('cum_miles') or 0):.2f} mi "
        f"(+{float(cue.get('leg_miles') or 0):.2f}) {label}"
        f"{signed_part}{until}."
    )


def wayfinding_display_detail(cue: dict[str, Any]) -> str:
    signed = cue_signed_text(cue.get("signed_as") or [])
    action = str(cue.get("action") or "FOLLOW").title()
    target = cue.get("target")
    until = cue.get("until")
    parts = []
    if signed and target:
        parts.append(f"{action} {signed} toward {target}.")
    elif signed:
        parts.append(f"{action} {signed}.")
    elif target:
        parts.append(f"{action} toward {target}.")
    else:
        parts.append(action + ".")
    if until:
        parts.append(f"UNTIL {until}.")
    verify = cue.get("verify")
    if verify:
        parts.append(f"VERIFY: {verify}.")
    avoid = cue.get("avoid") or cue.get("negative_cues") or []
    if avoid:
        parts.append(f"IGNORE: {sentence_list(avoid)}.")
    note = cue.get("note")
    if note:
        parts.append(str(note))
    return " ".join(parts)


def make_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    leg_miles: float,
    cue_type: str,
    action: str,
    signed_as: list[Any] | None = None,
    target: Any = None,
    until: Any = None,
    verify: Any = None,
    avoid: list[Any] | None = None,
    confidence: str = "planner",
    note: Any = None,
    official_segment_ids: list[Any] | None = None,
) -> dict[str, Any]:
    cue = {
        "seq": seq,
        "cum_miles": round(float(cum_miles or 0), 2),
        "leg_miles": round(float(leg_miles or 0), 2),
        "cue_type": cue_type,
        "action": action,
        "signed_as": unique_nonempty_text(signed_as or []),
        "target": normalized_trail_text(target),
        "until": normalized_trail_text(until),
        "verify": normalized_trail_text(verify),
        "avoid": unique_nonempty_text(avoid or []),
        "confidence": confidence,
        "note": normalized_trail_text(note),
        "official_segment_ids": normalized_segment_ids(official_segment_ids or []),
    }
    cue = {key: value for key, value in cue.items() if value not in (None, "", [])}
    cue["compact"] = wayfinding_compact(cue)
    cue["display_detail"] = wayfinding_display_detail(cue)
    return cue


def access_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    parking: dict[str, Any],
    first_trail: Any,
    access: dict[str, Any],
    start_access_gap_miles: float,
) -> dict[str, Any]:
    first_trail_label = display_trail(first_trail)
    first_key = signpost_key(first_trail) or lookup_text(first_trail)
    trailhead_key = lookup_text(parking.get("name"))
    manual_hint = MANUAL_ACCESS_HINTS.get((trailhead_key, first_key))
    access_miles = max(float(access.get("mapped_access_miles") or 0), float(start_access_gap_miles or 0))
    if manual_hint:
        return make_wayfinding_cue(
            seq=seq,
            cum_miles=cum_miles,
            leg_miles=access_miles,
            cue_type="start_access",
            action="FOLLOW",
            signed_as=manual_hint.get("signed_as") or [],
            target=manual_hint.get("target") or first_trail_label,
            until=manual_hint.get("until") or f"signed junction with {first_trail_label}",
            verify=f"watch for signs: {cue_signed_text(manual_hint.get('signed_as') or [])}",
            avoid=manual_hint.get("avoid") or [],
            confidence="field_check_needed",
            note="This access leg is not official challenge credit.",
        )

    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    if start_label and signpost_key(start_label) != first_key:
        return make_wayfinding_cue(
            seq=seq,
            cum_miles=cum_miles,
            leg_miles=access_miles,
            cue_type="start_access",
            action="FOLLOW",
            signed_as=[start_label],
            target=first_trail_label,
            until=f"signed junction with {first_trail_label}",
            verify=signpost_sentence([start_label], prefix="watch for signs").replace(".", ""),
            confidence="planner",
            note="This access leg is not official challenge credit.",
        )

    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=access_miles,
        cue_type="official_segment_start" if access_miles <= 0.05 else "start_access",
        action="FOLLOW",
        signed_as=[first_trail_label],
        target=first_trail_label,
        until=f"signed {first_trail_label} route / first official segment",
        verify=signpost_sentence([first_trail], prefix="watch for signs").replace(".", ""),
        confidence="planner" if access_miles <= 0.05 else "field_check_needed",
    )


def link_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    link: dict[str, Any],
    next_trail: Any,
) -> dict[str, Any] | None:
    names = unique_nonempty_text((link.get("signpost_labels") or []) + (link.get("connector_names") or []))
    distance = float(link.get("distance_miles") or link.get("connector_miles") or link.get("road_miles") or 0)
    if not names and distance <= 0.01:
        return None
    classes = {str(item) for item in link.get("connector_classes") or []}
    cue_type = "connector_road" if "osm_public_road" in classes or float(link.get("road_miles") or 0) else "connector_named_trail"
    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=distance,
        cue_type=cue_type,
        action="FOLLOW",
        signed_as=names or ["connector/access"],
        target=display_trail(next_trail),
        until=f"signed junction with {display_trail(next_trail)}",
        verify=signpost_sentence(names, prefix="watch for signs").replace(".", "") if names else "",
        confidence="planner" if names else "field_check_needed",
        note="Connector mileage does not count as new official challenge credit.",
    )


def official_group_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    group: dict[str, Any],
    next_group: dict[str, Any] | None,
    first_group: bool,
    phrase: str | None = None,
) -> dict[str, Any]:
    trail_label = display_trail(group["trail_name"])
    official_miles = sum(float(segment.get("official_miles") or 0) for segment in group.get("segments") or [])
    segment_ids = [segment.get("seg_id") for segment in group.get("segments") or []]
    if next_group:
        until = f"signed junction with {display_trail(next_group['trail_name'])}"
        target = display_trail(next_group["trail_name"])
    else:
        until = f"end of {trail_label} for this route"
        target = "return to car"
    if first_group:
        cue_type = "follow_official_segment"
        action = "FOLLOW"
    else:
        cue_type = "junction_turn"
        action = (phrase or "TAKE").upper()
    notes = [
        segment_completion_sentence(group.get("segments") or [], final_group=not next_group),
        group_effort_sentence(group.get("segments") or []),
        ascent_warning_sentence(group.get("segments") or []),
    ]
    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=official_miles,
        cue_type=cue_type,
        action=action,
        signed_as=[trail_label],
        target=target,
        until=until,
        verify=signpost_sentence([trail_label], prefix="watch for signs").replace(".", ""),
        confidence="planner",
        note=" ".join(note for note in notes if note),
        official_segment_ids=segment_ids,
    )


def return_wayfinding_cue(
    *,
    seq: int,
    cum_miles: float,
    parking: dict[str, Any],
    last_trail: Any,
    return_to_car: dict[str, Any],
    return_access_gap_miles: float,
) -> dict[str, Any]:
    return_names = unique_nonempty_text(return_to_car.get("connector_names") or [])
    start_label = parking.get("nearest_open_trail_label") or parking.get("nearest_open_trail_name")
    start_label = signpost_label(start_label) or normalized_trail_text(start_label)
    signed_as = return_names or ([start_label] if start_label else [display_trail(last_trail)])
    access_miles = max(
        float(return_access_gap_miles or 0),
        float(return_to_car.get("connector_miles") or 0),
        float(return_to_car.get("road_miles") or 0),
        float(return_to_car.get("official_repeat_miles") or 0),
    )
    return make_wayfinding_cue(
        seq=seq,
        cum_miles=cum_miles,
        leg_miles=access_miles,
        cue_type="return_to_car" if access_miles <= 0.05 else "exit_access",
        action="FOLLOW",
        signed_as=signed_as,
        target=parking.get("name") or "parked car",
        until="parked car / trailhead",
        verify=f"finish at {parking.get('name') or 'the parked car'}",
        confidence="planner" if return_names or start_label else "field_check_needed",
        note="Return leg does not count as new official challenge credit." if access_miles > 0.05 else "",
    )


def refresh_wayfinding_text(cue: dict[str, Any]) -> None:
    cue["compact"] = wayfinding_compact(cue)
    cue["display_detail"] = wayfinding_display_detail(cue)


def add_names_to_wayfinding_cue(cue: dict[str, Any], names: list[str], note_prefix: str) -> None:
    clean_names = unique_nonempty_text(names)
    if not clean_names:
        return
    cue["signed_as"] = unique_nonempty_text((cue.get("signed_as") or []) + clean_names)
    note = str(cue.get("note") or "").strip()
    addition = f"{note_prefix}: {sentence_list(clean_names)}."
    if addition not in note:
        cue["note"] = " ".join(part for part in [note, addition] if part)
    refresh_wayfinding_text(cue)


def append_names_to_step(step: dict[str, str], names: list[str], prefix: str) -> None:
    clean_names = unique_nonempty_text(names)
    if not clean_names:
        return
    addition = f"{prefix}: {sentence_list(clean_names)}."
    detail = str(step.get("detail") or "")
    if addition not in detail:
        step["detail"] = " ".join(part for part in [detail, addition] if part)


def first_cue_index_before_official(route: dict[str, Any]) -> int | None:
    cues = route.get("wayfinding_cues") or []
    for index, cue in enumerate(cues):
        if cue.get("official_segment_ids"):
            return 0 if index else None
        if str(cue.get("cue_type") or "") in {"start_access", "official_segment_start"}:
            return index
    return 0 if cues else None


def connector_cue_index_for_mile(route: dict[str, Any], route_miles: float | None) -> int | None:
    cues = route.get("wayfinding_cues") or []
    if not cues:
        return None
    connector_indexes = [
        index
        for index, cue in enumerate(cues)
        if str(cue.get("cue_type") or "") in {"connector_named_trail", "connector_road", "repeat_official_noncredit"}
    ]
    if connector_indexes:
        if route_miles is None:
            return connector_indexes[0]
        return min(
            connector_indexes,
            key=lambda index: abs(
                float(cues[index].get("cum_miles") or 0)
                + float(cues[index].get("leg_miles") or 0) / 2
                - float(route_miles or 0)
            ),
        )
    if route_miles is not None:
        return min(
            range(len(cues)),
            key=lambda index: abs(float(cues[index].get("cum_miles") or 0) - float(route_miles or 0)),
        )
    return 0


def enrich_route_with_walkthrough_edge_names(
    route: dict[str, Any],
    track_segments: list[list[tuple[float, float]]],
    graph_edges: list[Any],
    *,
    snap_tolerance_miles: float = 0.015,
) -> dict[str, Any]:
    if not track_segments or not graph_edges or not route.get("wayfinding_cues"):
        return route
    local_edges = filter_edges_for_track(graph_edges, track_segments)
    if not local_edges:
        return route
    all_text = " ".join(
        [
            str(step.get("title") or "") + " " + str(step.get("detail") or "")
            for step in route.get("turn_by_turn_steps") or []
        ]
        + [
            " ".join(
                [
                    str(cue.get("compact") or ""),
                    str(cue.get("display_detail") or ""),
                    str(cue.get("target") or ""),
                    str(cue.get("until") or ""),
                    " ".join(str(item) for item in cue.get("signed_as") or []),
                ]
            )
            for cue in route.get("wayfinding_cues") or []
        ]
    )
    groups, _unmatched = matched_edge_groups(
        resample_track_segments_for_matching(track_segments),
        TrailGraph(local_edges),
        snap_tolerance_miles,
        preferred_text=all_text,
    )
    claimed_segment_ids = {
        str(item)
        for item in (
            route.get("segment_ids")
            or (route.get("outing") or {}).get("remaining_segment_ids")
            or (route.get("outing") or {}).get("segment_ids")
            or []
        )
    }
    first_official_index = next(
        (index for index, group in enumerate(groups) if official_group_claimed(group, claimed_segment_ids)),
        None,
    )
    last_official_index = None
    for index, group in enumerate(groups):
        if official_group_claimed(group, claimed_segment_ids):
            last_official_index = index

    start_names: list[str] = []
    if first_official_index is not None:
        route_start_text = start_access_text(route)
        for group in groups[:first_official_index]:
            edge = group.get("_edge")
            if edge and named_nonofficial_group(group) and not text_mentions_edge(route_start_text, edge):
                start_names.append(edge.name)
    if start_names:
        cue_index = first_cue_index_before_official(route)
        if cue_index is not None:
            add_names_to_wayfinding_cue(
                route["wayfinding_cues"][cue_index],
                start_names,
                "Access also follows",
            )
        for step in route.get("turn_by_turn_steps") or []:
            if str(step.get("kind") or "").lower() == "access":
                append_names_to_step(step, start_names, "Access also follows")
                break

    if first_official_index is not None and last_official_index is not None:
        for group_index, group in enumerate(groups):
            edge = group.get("_edge")
            is_between_claimed_officials = first_official_index < group_index < last_official_index
            if not edge or not is_between_claimed_officials or not named_nonofficial_group(group):
                continue
            if text_mentions_edge(all_text, edge):
                continue
            cue_index = connector_cue_index_for_mile(route, group.get("route_miles_start"))
            if cue_index is not None:
                add_names_to_wayfinding_cue(
                    route["wayfinding_cues"][cue_index],
                    [edge.name],
                    "Connector also uses",
                )
            for step in route.get("turn_by_turn_steps") or []:
                if str(step.get("kind") or "").lower() == "navigate":
                    append_names_to_step(step, [edge.name], "Connector also uses")
                    break
            all_text += " " + edge.name
    return route


def build_wayfinding_cues(route: dict[str, Any]) -> list[dict[str, Any]]:
    parking = route.get("parking") or {}
    cue_list = route.get("route_cues") or []
    track_segments = route.get("_track_segments") or []
    official_index = route.get("_official_segment_index") or {}
    cues: list[dict[str, Any]] = []
    seq = 1
    cum_miles = 0.0
    for cue in cue_list:
        segments = cue.get("segments") or []
        groups = trail_groups(segments)
        if not groups:
            continue
        first_trail = groups[0]["trail_name"]
        gaps = non_credit_gaps_for_cue(cue, track_segments, official_index)
        access_cue = access_wayfinding_cue(
            seq=seq,
            cum_miles=cum_miles,
            parking=parking,
            first_trail=first_trail,
            access=cue.get("start_access") or {},
            start_access_gap_miles=float(gaps.get("start_access_gap_miles") or 0),
        )
        cues.append(access_cue)
        seq += 1
        cum_miles += float(access_cue.get("leg_miles") or 0)
        links = list(cue.get("between_links") or [])
        for index, group in enumerate(groups):
            is_first = index == 0
            prior_group = groups[index - 1] if not is_first else None
            if not is_first:
                link = link_for_group_transition(links, index - 1)
                connector_cue = link_wayfinding_cue(seq=seq, cum_miles=cum_miles, link=link, next_trail=group["trail_name"])
                if connector_cue:
                    cues.append(connector_cue)
                    seq += 1
                    cum_miles += float(connector_cue.get("leg_miles") or 0)
            next_group = groups[index + 1] if index + 1 < len(groups) else None
            phrase = turn_phrase_for_transition(prior_group or {}, group, track_segments, official_index) if prior_group else None
            follow_cue = official_group_wayfinding_cue(
                seq=seq,
                cum_miles=cum_miles,
                group=group,
                next_group=next_group,
                first_group=is_first,
                phrase=phrase,
            )
            cues.append(follow_cue)
            seq += 1
            cum_miles += float(follow_cue.get("leg_miles") or 0)
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            last_trail = groups[-1]["trail_name"]
            return_cue = return_wayfinding_cue(
                seq=seq,
                cum_miles=cum_miles,
                parking=parking,
                last_trail=last_trail,
                return_to_car=return_to_car,
                return_access_gap_miles=float(gaps.get("return_access_gap_miles") or 0),
            )
            cues.append(return_cue)
            seq += 1
            cum_miles += float(return_cue.get("leg_miles") or 0)
    return cues


def build_turn_by_turn_steps(route: dict[str, Any]) -> list[dict[str, str]]:
    outing = route["outing"]
    parking = route.get("parking") or {}
    cues = route.get("route_cues") or []
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    track_segments = route.get("_track_segments") or []
    official_index = route.get("_official_segment_index") or {}
    steps = [
        {
            "kind": "park",
            "title": f"Park/start at {parking.get('name') or outing.get('trailhead') or 'planned parking'}",
            "detail": "Start the GPX before leaving the car. The track should begin and end at this parking point.",
        }
    ]
    notes = manual_signpost_notes(route)
    if notes:
        steps.append({"kind": "checkpoint", "title": "Key checkpoints", "detail": " ".join(notes)})
    for cue in cues:
        segments = cue.get("segments") or []
        first_segment = segments[0] if segments else {}
        first_trail = first_segment.get("trail_name") or cue.get("title") or "the first official segment"
        access = cue.get("start_access") or {}
        gaps = non_credit_gaps_for_cue(cue, track_segments, official_index)
        steps.append(access_navigation_step(parking, first_trail, access, gaps.get("start_access_gap_miles") or 0))
        steps.extend(trail_navigation_steps_for_cue(cue, parking, track_segments, official_index))
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            last_segment = segments[-1] if segments else {}
            last_trail = last_segment.get("trail_name") or cue.get("title") or "route"
            steps.append(return_navigation_step(parking, last_trail, return_to_car, gaps.get("return_access_gap_miles") or 0))
    return steps


def navigation_quality_for_route(route: dict[str, Any]) -> dict[str, Any]:
    cues = route.get("route_cues") or []
    track_segments = route.get("_track_segments") or []
    official_index = route.get("_official_segment_index") or {}
    gap_records = [
        non_credit_gaps_for_cue(cue, track_segments, official_index)
        for cue in cues
    ]
    max_start_gap = max((float(record.get("start_access_gap_miles") or 0) for record in gap_records), default=0.0)
    max_return_gap = max((float(record.get("return_access_gap_miles") or 0) for record in gap_records), default=0.0)
    return {
        "start_access_gap_miles": round(max_start_gap, 3),
        "return_access_gap_miles": round(max_return_gap, 3),
        "non_credit_start_access_required": max_start_gap > 0.05,
        "non_credit_return_access_required": max_return_gap > 0.05,
    }


def render_card(route: dict[str, Any]) -> str:
    outing = route["outing"]
    parking = route.get("parking") or {}
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    effort = route_effort_summary(route)
    time_estimates = route_time_estimate_summary(route)
    completion_safe = (route.get("completion_safety") or {}).get(
        "normal_completion_preserves_remaining_menu_coverage"
    )
    completion_safe_value = "false" if completion_safe is False else "true"
    segment_ids = " ".join(normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids")))
    nav_url = route.get("parking_navigation_url")
    nav_link = (
        f'<a class="secondary" href="{html_escape(nav_url)}">Open parking in Google Maps</a>'
        if nav_url
        else '<span class="secondary disabled">Parking navigation unavailable</span>'
    )
    warnings = ""
    if not route["validation"]["passed"]:
        warnings = '<p class="warning">GPX validation failed. Do not use this route in the field until reviewed.</p>'
    wayfinding_cues = route.get("wayfinding_cues") or build_wayfinding_cues(route)
    steps_html = "".join(
        f'<li class="{html_escape(cue.get("cue_type"))}" tabindex="0">'
        f'<b><span class="cue-code">{int(cue.get("seq") or 0):02d} {float(cue.get("cum_miles") or 0):.2f} mi '
        f'(+{float(cue.get("leg_miles") or 0):.2f})</span> {html_escape(cue_type_label(str(cue.get("cue_type") or "")))}</b>'
        f'<span>{html_escape(cue.get("display_detail") or wayfinding_display_detail(cue))}</span></li>'
        for cue in wayfinding_cues
    )
    logistics_html = logistics_section_html(logistics)
    return f"""
    <article class="card" id="{html_escape(outing['outing_id'])}" data-outing-id="{html_escape(outing['outing_id'])}" data-minutes="{int(outing.get('total_minutes') or 0)}" data-completion-safe="{completion_safe_value}" data-segment-ids="{html_escape(segment_ids)}">
      <div class="card-head">
        <h2>{html_escape(outing['label'])}. {html_escape(outing['trailhead'])}</h2>
      </div>
      <div class="stats">
        <div><b>Door to door p75</b><strong>{html_escape(format_minutes(outing.get('total_minutes')))}</strong></div>
        <div><b>On foot</b><strong>{html_escape(format_miles(outing.get('on_foot_miles')))} mi</strong></div>
        <div><b>Official</b><strong>{html_escape(format_miles(outing.get('official_miles')))} mi</strong></div>
        <div><b>Door to door p90</b><strong>{html_escape(format_minutes(time_estimates.get('door_to_door_minutes_p90')))}</strong></div>
        <div><b>Segments</b><strong>{html_escape(outing.get('remaining_segment_count'))} / {len(outing.get('segment_ids') or [])}</strong></div>
        <div><b>Climb</b><strong>{int(round(effort.get('ascent_ft') or 0))} ft</strong></div>
      </div>
      <div class="actions">
        <a href="{html_escape(route['gpx_href'])}" download>Open Nav GPX</a>
        <a class="secondary" href="live-map.html?outing={html_escape(outing['outing_id'])}">Open Live Map</a>
        {nav_link}
        <button type="button" class="active-button" data-active-action="pin">Pin active</button>
        <button type="button" class="done-button" data-complete-action="mark">Mark done</button>
        <button type="button" class="undo-button" data-complete-action="undo">Undo done</button>
      </div>
      {warnings}
      <section><h3>PARK/START</h3><p>Park/start at {html_escape(parking.get('name') or outing.get('trailhead'))}</p></section>
      {logistics_html}
      <section><h3>Trails</h3><p>{html_escape(', '.join(outing.get('trails') or []))}</p></section>
      <section><h3>Field Cue Sheet</h3><p class="cue-help">What to do next: Tap the cue you are working on to keep your place.</p><ol class="steps decision-cards">{steps_html}</ol></section>
    </article>
    """


def render_index(manifest: dict[str, Any]) -> str:
    cards = "\n".join(render_card(route) for route in manifest["routes"])
    manual_count = manifest["summary"]["manual_hold_count"]
    zip_href = manifest["summary"].get("gpx_zip_href") or f"gpx/{GPX_ZIP_NAME}"
    all_segment_ids = {
        segment_id
        for route in manifest["routes"]
        for segment_id in normalized_segment_ids(
            route.get("outing", {}).get("remaining_segment_ids") or route.get("outing", {}).get("segment_ids")
        )
    }
    certified = manifest.get("certified_baseline") or {}
    official_segment_count = certified.get("official_segment_count") or len(all_segment_ids)
    menu_on_foot_miles = sum(
        float((route.get("outing") or route).get("on_foot_miles") or 0)
        for route in manifest["routes"]
    )
    gpx_status = "GPX passed" if manifest["summary"].get("gpx_validation_passed") else "GPX needs review"
    field_menu_text = (
        f"{len(manifest['routes'])} runnable outings · "
        f"{len(all_segment_ids)}/{official_segment_count} official segments · "
        f"{format_miles(menu_on_foot_miles)} on foot · {gpx_status}"
    )
    filter_labels = {60: "&le;1h", 90: "&le;90m", 120: "&le;2h", 180: "&le;3h", 240: "&le;4h", 360: "&le;6h"}
    filter_buttons = "\n      ".join(
        ["""<button type="button" class="active" data-filter="all">All</button>"""]
        + [
            f'<button type="button" data-filter="{minutes}">{filter_labels.get(minutes, f"&le;{minutes}m")}</button>'
            for minutes in TIME_FILTER_MINUTES
        ]
    )
    manual_note = (
        f"<p>{manual_count} manual-design outing(s) are intentionally hidden from GPX export until the route is redesigned.</p>"
        if manual_count
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#111827">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Trails Packet">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="manifest" href="manifest.webmanifest">
  <link rel="apple-touch-icon" href="icons/icon-192.png">
  <link rel="icon" href="icons/icon-192.png">
  <title>Phone Field Packet</title>
  <style>
    body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f7f2; color:#111827; }}
    header {{ padding:14px 14px 10px; background:rgba(255,255,255,.97); border-bottom:1px solid #d7ddd4; }}
    h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:0; }}
    p {{ margin:4px 0; color:#475467; line-height:1.4; }}
    .top-grid {{ display:grid; grid-template-columns:1fr; gap:8px; margin-top:10px; }}
    .status-panel {{ border:1px solid #d7ddd4; border-radius:8px; padding:8px; background:#f9fafb; }}
    .status-panel b {{ color:#111827; }}
    .filters {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; margin-top:10px; }}
    button {{ min-height:34px; border:1px solid #d7ddd4; border-radius:6px; background:#fff; font-weight:700; }}
    button.active {{ background:#111827; color:#fff; border-color:#111827; }}
    .utility-actions {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; }}
    .utility-actions a,.utility-actions button {{ display:flex; align-items:center; justify-content:center; min-height:38px; padding:0 10px; border-radius:6px; border:1px solid #d7ddd4; background:#fff; color:#111827; font-weight:800; text-decoration:none; }}
    .quick-list {{ margin:10px 0 0; padding:8px; border:1px solid #d7ddd4; border-radius:8px; background:#fff; }}
    .quick-list h2 {{ margin:0 0 4px; color:#111827; font-size:14px; }}
    main {{ padding:10px; display:grid; gap:10px; }}
    .card {{ overflow:hidden; border:1px solid #d7ddd4; border-radius:8px; background:#fff; box-shadow:0 1px 4px rgba(15,23,42,.08); }}
    .card.active-outing {{ border:2px solid #2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.12); }}
    .card.completed {{ opacity:.48; }}
    body.hide-completed .card.completed {{ display:none !important; }}
    .card-head {{ padding:12px; background:#111827; color:#fff; }}
    .card.active-outing .card-head {{ background:#1d4ed8; }}
    .card-head span {{ display:block; color:#cbd5e1; font-size:12px; text-transform:uppercase; font-weight:800; }}
    h2 {{ margin:3px 0 0; font-size:19px; line-height:1.15; letter-spacing:0; }}
    .stats {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; padding:10px 12px; }}
    .stats div {{ border:1px solid #e5e7eb; border-radius:6px; padding:7px; background:#f9fafb; }}
    .stats b {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .stats strong {{ display:block; margin-top:2px; font-size:15px; }}
    .actions {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 12px 12px; }}
    .actions a,.actions span,.actions button {{ display:flex; align-items:center; justify-content:center; min-height:40px; border-radius:6px; font-weight:800; text-decoration:none; border:1px solid #d7ddd4; }}
    .actions a:first-child {{ background:#2563eb; color:#fff; }}
    .done-button {{ background:#166534; color:#fff; border-color:#166534 !important; }}
    .active-button {{ background:#dbeafe; color:#1e3a8a; border-color:#93c5fd !important; }}
    .card.active-outing .active-button {{ background:#1d4ed8; color:#fff; border-color:#1d4ed8 !important; }}
    .undo-button {{ display:none !important; background:#fff; color:#111827; }}
    .card.completed .done-button {{ display:none !important; }}
    .card.completed .undo-button {{ display:flex !important; }}
    .secondary {{ border:1px solid #d7ddd4; color:#111827; background:#fff; }}
    .disabled {{ color:#667085; }}
    section {{ padding:10px 12px; border-top:1px solid #eef0ed; }}
    h3 {{ margin:0 0 6px; font-size:12px; text-transform:uppercase; letter-spacing:0; color:#111827; }}
    .segment {{ border:1px solid #e5e7eb; border-radius:6px; padding:7px; margin:5px 0; background:#fff; }}
    .segment.ascent {{ border-color:#b45309; background:#fff7ed; }}
    .segment b {{ display:block; font-size:13px; }}
    .segment span {{ display:block; margin-top:2px; color:#475467; font-size:12px; line-height:1.35; }}
    .steps {{ margin:0; padding:0; list-style:none; counter-reset:step; display:grid; gap:7px; }}
    .steps li {{ counter-increment:step; position:relative; padding:8px 8px 8px 36px; border:1px solid #e5e7eb; border-radius:6px; background:#fff; }}
    .steps li::before {{ content:counter(step); position:absolute; left:8px; top:8px; display:grid; place-items:center; width:20px; height:20px; border-radius:999px; background:#111827; color:#fff; font-size:11px; font-weight:900; }}
    .steps li.connector {{ background:#f0f9ff; border-color:#bae6fd; }}
    .steps li.ascent {{ background:#fff7ed; border-color:#fdba74; }}
    .steps li.return {{ background:#f0fdf4; border-color:#bbf7d0; }}
    .steps li.car-pass {{ background:#fefce8; border-color:#fde68a; }}
    .steps li.water {{ background:#eff6ff; border-color:#bfdbfe; }}
    .steps li.checkpoint {{ background:#fff7ed; border-color:#fdba74; }}
    .steps li.current-step {{ border-color:#2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.14); background:#eff6ff; }}
    .steps b {{ display:block; font-size:13px; }}
    .steps span {{ display:block; margin-top:2px; color:#475467; font-size:12px; line-height:1.35; }}
    .decision-cards li b {{ font-size:15px; }}
    .decision-cards li span {{ font-size:13px; }}
    .cue-code {{ display:block; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:#6b7280; margin-bottom:3px; }}
    .cue-help {{ margin:0 0 8px; font-size:13px; color:#475467; }}
    .signpost-notes {{ margin:6px 0 0; padding-left:18px; color:#475467; font-size:12px; line-height:1.35; }}
    .warning {{ margin:10px 12px; padding:8px; border-left:4px solid #b45309; background:#fff7ed; color:#7c2d12; }}
    body.screenshot header,.screenshot .utility-actions,.screenshot .filters,.screenshot .actions {{ display:none !important; }}
    body.screenshot main {{ padding:0; }}
    body.screenshot .card {{ display:none !important; border:0; border-radius:0; box-shadow:none; }}
    body.screenshot .card.active-outing:not(.completed), body.screenshot .card:not(.completed):first-of-type {{ display:block !important; }}
    @media (min-width:760px) {{ main {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Phone Field Packet</h1>
    <p>Open one outing, send the Nav GPX to your navigation app, then use the card for parking, turn-by-turn cues, and return-to-car notes.</p>
    <div class="top-grid">
      <div class="status-panel"><b>Offline-ready</b> after the first full load. In Safari, Share &rarr; Add to Home Screen for the app-style launcher. <span id="offline-status">Checking offline cache...</span></div>
      <div class="status-panel"><b>Field menu</b> <span id="field-menu-summary">{html_escape(field_menu_text)}</span></div>
      <div class="status-panel"><b>Progress</b> <span id="remaining-segment-count">{len(all_segment_ids)}</span> of <span id="total-segment-count">{len(all_segment_ids)}</span> official segments remain in this field menu. <span id="completed-outing-count">0</span> outing(s) marked done on this phone.</div>
      <div class="utility-actions">
        <a href="live-map.html">Live GPS map</a>
        <button type="button" id="completed-toggle">Hide completed</button>
        <button type="button" id="screenshot-toggle">Screenshot mode</button>
        <button type="button" id="clear-active">Clear active</button>
        <button type="button" id="export-progress">Export progress</button>
        <button type="button" id="reset-completed">Reset progress</button>
      </div>
    </div>
    <div class="quick-list"><h2>Today&apos;s best options</h2><p id="best-today-copy">Use the time buttons to pick what fits the door-to-door window. Mark completed outings so they disappear from the active field list.</p></div>
    {manual_note}
    <div class="filters">
      {filter_buttons}
    </div>
  </header>
  <main>{cards}</main>
  <script>
    const STORAGE_KEY = "{COMPLETED_STORAGE_KEY}";
    const ACTIVE_KEY = "{ACTIVE_STORAGE_KEY}";
    const buttons = [...document.querySelectorAll("button[data-filter]")];
    const cards = [...document.querySelectorAll(".card")];
    const cardContainer = document.querySelector("main");
    const completedToggle = document.getElementById("completed-toggle");
    const screenshotToggle = document.getElementById("screenshot-toggle");
    const clearActive = document.getElementById("clear-active");
    const exportProgress = document.getElementById("export-progress");
    const resetCompleted = document.getElementById("reset-completed");
    const offlineStatus = document.getElementById("offline-status");
    const remainingSegmentCount = document.getElementById("remaining-segment-count");
    const totalSegmentCount = document.getElementById("total-segment-count");
    const completedOutingCount = document.getElementById("completed-outing-count");
    const bestTodayCopy = document.getElementById("best-today-copy");

    function segmentIdsForCard(card) {{
      return (card.dataset.segmentIds || "").split(" ").filter(Boolean);
    }}

    function allSegmentSet() {{
      const ids = new Set();
      cards.forEach(card => segmentIdsForCard(card).forEach(id => ids.add(id)));
      return ids;
    }}

    function completedSet() {{
      try {{
        return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
      }} catch (error) {{
        return new Set();
      }}
    }}

    function saveCompleted(set) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...set]));
    }}

    function completedSegmentSet() {{
      const completed = completedSet();
      const ids = new Set();
      cards.forEach(card => {{
        if (completed.has(card.dataset.outingId)) {{
          segmentIdsForCard(card).forEach(id => ids.add(id));
        }}
      }});
      return ids;
    }}

    function activeOutingId() {{
      return localStorage.getItem(ACTIVE_KEY) || "";
    }}

    function saveActive(id) {{
      if (id) {{
        localStorage.setItem(ACTIVE_KEY, id);
      }} else {{
        localStorage.removeItem(ACTIVE_KEY);
      }}
    }}

    function syncCompletedState() {{
      const completed = completedSet();
      cards.forEach(card => card.classList.toggle("completed", completed.has(card.dataset.outingId)));
      applyFilter();
    }}

    function syncActiveState(scrollToActive = false) {{
      const activeId = activeOutingId();
      let activeCard = null;
      cards.forEach(card => {{
        const isActive = activeId && card.dataset.outingId === activeId;
        card.classList.toggle("active-outing", Boolean(isActive));
        const activeButton = card.querySelector('[data-active-action="pin"]');
        if (activeButton) {{
          activeButton.textContent = isActive ? "Active" : "Pin active";
        }}
        if (isActive) {{
          activeCard = card;
        }}
      }});
      if (activeCard) {{
        cardContainer.prepend(activeCard);
      }}
      if (clearActive) {{
        clearActive.disabled = !activeId;
      }}
      applyFilter();
      if (scrollToActive && activeCard) {{
        activeCard.scrollIntoView({{ block: "start" }});
      }}
    }}

    function activeFilter() {{
      return document.querySelector("button[data-filter].active")?.dataset.filter || "all";
    }}
    function activeFilterLabel() {{
      return document.querySelector("button[data-filter].active")?.textContent?.trim() || "All";
    }}

    function updateDailyStatus() {{
      const completed = completedSet();
      const allSegments = allSegmentSet();
      const completedSegments = completedSegmentSet();
      const remaining = Math.max(allSegments.size - completedSegments.size, 0);
      if (remainingSegmentCount) {{
        remainingSegmentCount.textContent = String(remaining);
      }}
      if (totalSegmentCount) {{
        totalSegmentCount.textContent = String(allSegments.size);
      }}
      if (completedOutingCount) {{
        completedOutingCount.textContent = String(completed.size);
      }}
      const filter = activeFilter();
      const candidates = cards
        .filter(card => !completed.has(card.dataset.outingId))
        .filter(card => filter === "all" || Number(card.dataset.minutes || 0) <= Number(filter))
        .map(card => {{
          const newSegments = segmentIdsForCard(card).filter(id => !completedSegments.has(id)).length;
          return {{
            card,
            completionSafe: card.dataset.completionSafe !== "false",
            newSegments,
            minutes: Number(card.dataset.minutes || 0),
            title: card.querySelector("h2")?.textContent?.trim() || card.dataset.outingId,
          }};
        }})
        .filter(item => item.newSegments > 0)
        .sort((left, right) => Number(right.completionSafe) - Number(left.completionSafe) || right.newSegments - left.newSegments || left.minutes - right.minutes);
      if (!bestTodayCopy) {{
        return;
      }}
      if (!candidates.length) {{
        bestTodayCopy.textContent = filter === "all"
          ? "No remaining runnable outings in this packet."
          : `No remaining runnable outings fit the ${{activeFilterLabel()}} door-to-door window.`;
        return;
      }}
      const best = candidates[0];
      const windowText = filter === "all" ? "the current list" : `${{activeFilterLabel()}} door to door`;
      const safetyText = best.completionSafe ? "completion-safe in the current menu" : "needs recertification review";
      bestTodayCopy.textContent = `Best today for ${{windowText}}: ${{best.title}} · ${{best.minutes}} min p75 · ${{best.newSegments}} new official segment(s) · ${{safetyText}}. Pin it before you start.`;
    }}

    function applyFilter() {{
      const filter = activeFilter();
      cards.forEach(card => {{
        const minutes = Number(card.dataset.minutes || 0);
        const isActive = card.classList.contains("active-outing");
        card.style.display = isActive || filter === "all" || minutes <= Number(filter) ? "" : "none";
      }});
      updateDailyStatus();
    }}

    buttons.forEach(button => button.addEventListener("click", () => {{
      buttons.forEach(item => item.classList.toggle("active", item === button));
      applyFilter();
    }}));

    cards.forEach(card => {{
      card.querySelector('[data-complete-action="mark"]')?.addEventListener("click", () => {{
        const completed = completedSet();
        completed.add(card.dataset.outingId);
        saveCompleted(completed);
        syncCompletedState();
      }});
      card.querySelector('[data-complete-action="undo"]')?.addEventListener("click", () => {{
        const completed = completedSet();
        completed.delete(card.dataset.outingId);
        saveCompleted(completed);
        syncCompletedState();
      }});
      card.querySelector('[data-active-action="pin"]')?.addEventListener("click", () => {{
        saveActive(card.dataset.outingId);
        syncActiveState(true);
      }});
      card.querySelectorAll(".steps li").forEach(step => {{
        step.addEventListener("click", () => {{
          card.querySelectorAll(".steps li.current-step").forEach(item => item.classList.remove("current-step"));
          step.classList.add("current-step");
        }});
        step.addEventListener("keydown", event => {{
          if (event.key === "Enter" || event.key === " ") {{
            event.preventDefault();
            step.click();
          }}
        }});
      }});
    }});

    clearActive.addEventListener("click", () => {{
      saveActive("");
      syncActiveState();
    }});

    completedToggle.addEventListener("click", () => {{
      document.body.classList.toggle("hide-completed");
      completedToggle.textContent = document.body.classList.contains("hide-completed") ? "Show completed" : "Hide completed";
    }});

    screenshotToggle.addEventListener("click", () => {{
      document.body.classList.toggle("screenshot");
    }});

    resetCompleted.addEventListener("click", () => {{
      localStorage.removeItem(STORAGE_KEY);
      syncCompletedState();
    }});

    exportProgress.addEventListener("click", () => {{
      const payload = {{
        schema: "boise_trails_phone_progress_v1",
        exported_at: new Date().toISOString(),
        completed_outing_ids: [...completedSet()].sort(),
        completed_segment_ids: [...completedSegmentSet()].sort((left, right) => left.length - right.length || left.localeCompare(right)),
        missed_segment_ids: [],
        note: "Review the activity before applying this to private planner state. Add missed_segment_ids for any planned segment not actually completed end-to-end."
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "boise-trails-progress.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }});

    if ("serviceWorker" in navigator) {{
      navigator.serviceWorker.register("service-worker.js").then(registration => {{
        offlineStatus.textContent = "Offline cache installed.";
        return registration;
      }}).catch(() => {{
        offlineStatus.textContent = "Offline cache unavailable; keep GPX downloaded.";
      }});
    }} else {{
      offlineStatus.textContent = "Offline cache unsupported; keep GPX downloaded.";
    }}

    window.addEventListener("online", () => offlineStatus.textContent = "Online.");
    window.addEventListener("offline", () => offlineStatus.textContent = "Offline. Cached cards and GPX remain available.");
    syncCompletedState();
    syncActiveState();
  </script>
</body>
</html>
"""


def render_live_map_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#0f172a">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Trail Live Map">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="manifest" href="manifest.webmanifest">
  <link rel="apple-touch-icon" href="icons/icon-192.png">
  <link rel="icon" href="icons/icon-192.png">
  <title>Live GPS Route Map</title>
  <style>
    :root { color-scheme: light; }
    html { height:100%; overflow:hidden; }
    * { box-sizing: border-box; }
    body { margin:0; height:100%; overflow:hidden; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#e8ece4; color:#111827; }
    .app { height:100dvh; min-height:100vh; overflow:hidden; display:grid; grid-template-rows:auto minmax(0,1fr) auto; }
    header { padding:calc(10px + env(safe-area-inset-top)) 12px 10px; background:#0f172a; color:#fff; box-shadow:0 2px 12px rgba(15,23,42,.18); z-index:2; }
    h1 { margin:0 0 8px; font-size:19px; letter-spacing:0; }
    .controls { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; }
    select,button { min-height:38px; border:1px solid #cbd5e1; border-radius:7px; background:#fff; color:#111827; font-weight:800; font-size:14px; }
    select { width:100%; min-width:0; padding:0 8px; }
    button { padding:0 10px; }
    .button-row { margin-top:8px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
    .button-row button.active { background:#2563eb; color:#fff; border-color:#2563eb; }
    .status { margin-top:8px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
    .status div { border:1px solid rgba(255,255,255,.18); border-radius:7px; padding:6px; background:rgba(255,255,255,.08); min-width:0; }
    .status b { display:block; font-size:10px; text-transform:uppercase; color:#cbd5e1; }
    .status span { display:block; margin-top:2px; font-size:13px; font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .map-shell { position:relative; min-height:0; overflow:hidden; background:#f7f6ef; }
    svg { width:100%; height:100%; display:block; touch-action:none; }
    .grid-line { stroke:#d8ded2; stroke-width:1; vector-effect:non-scaling-stroke; }
    .route-halo { fill:none; stroke:#fff; stroke-width:18; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-line { fill:none; stroke:#2563eb; stroke-width:8; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-context { fill:none; stroke:#94a3b8; stroke-width:5; stroke-linecap:round; stroke-linejoin:round; opacity:.42; vector-effect:non-scaling-stroke; }
    .route-context-gradient { opacity:.24; }
    .active-halo { fill:none; stroke:#fff; stroke-width:22; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .active-line { fill:none; stroke:#2563eb; stroke-width:10; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-slice { fill:none; stroke-width:8; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .route-slice.napkin { stroke-width:12; }
    .cue-leg { fill:none; stroke-width:9; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .chevron { fill:none; stroke:#111827; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .direction-arrow { fill:#111827; stroke:#fff; stroke-width:2.5; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .cue-dot { fill:#111827; stroke:#fff; stroke-width:3; vector-effect:non-scaling-stroke; }
    .cue-dot.active { fill:#2563eb; }
    .cue-dot.next { fill:#16a34a; }
    .cue-dot.context-marker { fill:#64748b; opacity:.28; stroke-width:2; }
    .cue-anchor { fill:#fff; stroke:#111827; stroke-width:2.2; vector-effect:non-scaling-stroke; }
    .cue-anchor.active { stroke:#2563eb; }
    .cue-anchor.next { stroke:#16a34a; }
    .cue-callout-line { stroke:#fff; stroke-width:7; stroke-linecap:round; vector-effect:non-scaling-stroke; }
    .cue-callout-line.dark { stroke:#111827; stroke-width:1.5; }
    .cue-label { fill:#fff; font-size:13px; font-weight:900; text-anchor:middle; dominant-baseline:central; pointer-events:none; }
    .cue-label.context-label { display:none; }
    .leg-tag { fill:#111827; font-size:15px; font-weight:950; paint-order:stroke; stroke:#fff; stroke-width:6; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .parking-dot { fill:#166534; stroke:#fff; stroke-width:4; vector-effect:non-scaling-stroke; }
    .finish-dot { fill:#b91c1c; stroke:#fff; stroke-width:4; vector-effect:non-scaling-stroke; }
    .marker-tag { fill:#111827; font-size:13px; font-weight:900; paint-order:stroke; stroke:#fff; stroke-width:5; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .user-accuracy { fill:#2563eb; opacity:.12; stroke:#2563eb; stroke-width:1; vector-effect:non-scaling-stroke; }
    .user-dot { fill:#0ea5e9; stroke:#fff; stroke-width:5; vector-effect:non-scaling-stroke; }
    .user-heading { fill:#0f172a; stroke:#fff; stroke-width:2; vector-effect:non-scaling-stroke; }
    .map-leg-banner { position:absolute; left:10px; right:10px; top:10px; z-index:1; border:1px solid #bfdbfe; border-radius:8px; background:rgba(239,246,255,.94); color:#111827; padding:8px 10px; font-size:13px; font-weight:850; line-height:1.25; box-shadow:0 2px 10px rgba(15,23,42,.10); }
    .map-leg-banner b { color:#1d4ed8; }
    .gap-warning { position:absolute; left:10px; right:10px; top:10px; z-index:1; border:1px solid #fed7aa; border-radius:8px; background:#fff7ed; color:#9a3412; padding:8px 10px; font-size:13px; font-weight:800; box-shadow:0 2px 10px rgba(15,23,42,.12); }
    .map-tools { position:absolute; right:10px; bottom:calc(10px + env(safe-area-inset-bottom)); display:grid; gap:6px; }
    .map-tools button { min-width:44px; min-height:44px; box-shadow:0 2px 8px rgba(15,23,42,.16); }
    footer { padding:8px 12px calc(8px + env(safe-area-inset-bottom)); background:rgba(255,255,255,.96); border-top:1px solid #d7ddd4; }
    .cue-card { min-height:46px; border:1px solid #d7ddd4; border-radius:8px; padding:8px; background:#fff; }
    .cue-card b { display:block; font-size:12px; text-transform:uppercase; color:#475467; }
    .cue-card span { display:block; margin-top:2px; color:#111827; font-size:14px; font-weight:800; line-height:1.25; }
    .cue-controls { margin-top:8px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
    .cue-controls button { min-height:36px; font-size:13px; }
    .note { margin-top:6px; color:#667085; font-size:12px; line-height:1.35; }
    @media (min-width:800px) {
      .app { grid-template-columns:380px 1fr; grid-template-rows:1fr auto; }
      header { grid-row:1 / span 2; }
      .button-row,.status { grid-template-columns:repeat(2,minmax(0,1fr)); }
      footer { grid-column:2; }
    }
    @media (max-width:900px) {
      header { padding:calc(8px + env(safe-area-inset-top)) 10px 8px; }
      h1 { font-size:17px; margin-bottom:6px; }
      select,button { min-height:34px; font-size:13px; }
      .button-row,.status { margin-top:6px; gap:5px; }
      .status div { padding:5px; }
      .note { display:none; }
      .map-leg-banner { font-size:12px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Live GPS Route Map</h1>
      <div class="controls">
        <select id="route-select" aria-label="Choose outing"></select>
        <button type="button" id="locate-button">Start GPS</button>
      </div>
      <div class="button-row" aria-label="Route style">
        <button type="button" class="active" data-style="ribbon">Ribbon</button>
        <button type="button" data-style="cue-legs">Cue legs</button>
        <button type="button" data-style="napkin">Napkin</button>
      </div>
      <div class="status">
        <div><b>Distance to route</b><span id="distance-to-route">--</span></div>
        <div><b>GPS accuracy</b><span id="gps-accuracy">--</span></div>
        <div><b>Progress</b><span id="route-progress">--</span></div>
      </div>
      <p class="note">Field cue-leg map. The blue ribbon is the active cue-to-cue leg; muted lines are surrounding route context. Use the GitHub Pages HTTPS URL or a local web server; direct file:// cannot load GPX data.</p>
    </header>
    <main class="map-shell">
      <svg id="map-svg" role="img" aria-label="Live route map">
        <g id="grid-layer"></g>
        <g id="route-layer"></g>
        <g id="marker-layer"></g>
        <g id="user-layer"></g>
      </svg>
      <div id="map-leg-banner" class="map-leg-banner" hidden></div>
      <div id="map-warning" class="gap-warning" hidden></div>
      <div class="map-tools">
        <button type="button" id="fit-button">Fit</button>
        <button type="button" id="zoom-in">+</button>
        <button type="button" id="zoom-out">-</button>
      </div>
    </main>
    <footer>
      <div class="cue-card">
        <b>Active leg / nearest cue</b>
        <span id="nearest-cue">Load an outing, then start GPS.</span>
        <div class="cue-controls">
          <button type="button" id="previous-cue">Prev cue</button>
          <button type="button" id="fit-leg">Fit leg</button>
          <button type="button" id="next-cue">Next cue</button>
        </div>
      </div>
    </footer>
  </div>
  <script>
    const FIELD_DATA_URL = "__FIELD_TOOL_DATA__";
    const ACTIVE_KEY = "__ACTIVE_KEY__";
    const state = {
      routes: [],
      route: null,
      trackSegments: [],
      waypoints: [],
      displayedSegments: [],
      projectedSegments: [],
      projected: [],
      routePositions: [],
      cumulativeM: [],
      totalRouteM: 0,
      gapWarnings: [],
      viewBox: null,
      baseViewBox: null,
      style: "ribbon",
      activeCueIndex: 0,
      watchId: null,
      user: null
    };
    const activePointers = new Map();
    const gesture = { mode: null, lastPoint: null, lastCenter: null, lastDistance: 0 };
    const svg = document.getElementById("map-svg");
    const routeSelect = document.getElementById("route-select");
    const routeLayer = document.getElementById("route-layer");
    const markerLayer = document.getElementById("marker-layer");
    const userLayer = document.getElementById("user-layer");
    const gridLayer = document.getElementById("grid-layer");
    const mapWarning = document.getElementById("map-warning");
    const mapLegBanner = document.getElementById("map-leg-banner");
    const distanceToRoute = document.getElementById("distance-to-route");
    const gpsAccuracy = document.getElementById("gps-accuracy");
    const routeProgress = document.getElementById("route-progress");
    const nearestCue = document.getElementById("nearest-cue");
    const locateButton = document.getElementById("locate-button");
    const previousCue = document.getElementById("previous-cue");
    const nextCue = document.getElementById("next-cue");
    const fitLegButton = document.getElementById("fit-leg");

    function miles(meters) { return meters / 1609.344; }
    function metersFromMiles(value) { return Number(value || 0) * 1609.344; }
    function fmtDistance(meters) {
      if (!Number.isFinite(meters)) return "--";
      if (meters < 160) return `${Math.round(meters)} m`;
      return `${miles(meters).toFixed(2)} mi`;
    }
    function fmtProgress(routeM) {
      const total = state.totalRouteM || 0;
      if (!total) return "--";
      return `${miles(Math.min(routeM, total)).toFixed(2)} / ${miles(total).toFixed(2)} mi`;
    }
    function escapeText(value) {
      return String(value ?? "").replace(/[&<>"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
    }
    function gpxNodes(xml, localName) {
      return [...xml.getElementsByTagNameNS("*", localName)];
    }
    function parseGpx(text) {
      const xml = new DOMParser().parseFromString(text, "application/xml");
      const trackSegments = gpxNodes(xml, "trkseg").map(segment => (
        [...segment.getElementsByTagNameNS("*", "trkpt")].map(node => ({
          lat: Number(node.getAttribute("lat")),
          lon: Number(node.getAttribute("lon"))
        })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon))
      )).filter(segment => segment.length > 0);
      if (!trackSegments.length) {
        const fallbackTrack = [...xml.getElementsByTagNameNS("*", "trkpt")].map(node => ({
          lat: Number(node.getAttribute("lat")),
          lon: Number(node.getAttribute("lon"))
        })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon));
        if (fallbackTrack.length) trackSegments.push(fallbackTrack);
      }
      const waypoints = gpxNodes(xml, "wpt").map(node => ({
        lat: Number(node.getAttribute("lat")),
        lon: Number(node.getAttribute("lon")),
        name: node.getElementsByTagNameNS("*", "name")[0]?.textContent || "",
        desc: node.getElementsByTagNameNS("*", "desc")[0]?.textContent || ""
      })).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon));
      return { trackSegments, waypoints };
    }
    function boundsFor(points) {
      const lats = points.map(point => point.lat);
      const lons = points.map(point => point.lon);
      return {
        minLat: Math.min(...lats),
        maxLat: Math.max(...lats),
        minLon: Math.min(...lons),
        maxLon: Math.max(...lons)
      };
    }
    function makeProjector(points) {
      const bounds = boundsFor(points);
      const lat0 = ((bounds.minLat + bounds.maxLat) / 2) * Math.PI / 180;
      const cosLat = Math.max(Math.cos(lat0), 0.01);
      return point => ({
        x: (point.lon - bounds.minLon) * 111320 * cosLat,
        y: (bounds.maxLat - point.lat) * 110540
      });
    }
    function pathFor(points) {
      if (!points.length) return "";
      return points.map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
    }
    function pathForSegments(segments) {
      return segments.map(pathFor).filter(Boolean).join(" ");
    }
    function distance(a, b) {
      return Math.hypot(a.x - b.x, a.y - b.y);
    }
    function perpendicularDistance(point, lineStart, lineEnd) {
      const dx = lineEnd.x - lineStart.x;
      const dy = lineEnd.y - lineStart.y;
      const len2 = dx * dx + dy * dy;
      if (!len2) return distance(point, lineStart);
      const t = Math.max(0, Math.min(1, ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / len2));
      return distance(point, { x: lineStart.x + dx * t, y: lineStart.y + dy * t });
    }
    function simplifyPolyline(points, tolerance = 6) {
      if (points.length <= 2) return points;
      let maxDistance = 0;
      let splitIndex = 0;
      for (let index = 1; index < points.length - 1; index += 1) {
        const candidateDistance = perpendicularDistance(points[index], points[0], points[points.length - 1]);
        if (candidateDistance > maxDistance) {
          maxDistance = candidateDistance;
          splitIndex = index;
        }
      }
      if (maxDistance <= tolerance) return [points[0], points[points.length - 1]];
      const left = simplifyPolyline(points.slice(0, splitIndex + 1), tolerance);
      const right = simplifyPolyline(points.slice(splitIndex), tolerance);
      return [...left.slice(0, -1), ...right];
    }
    function refreshDisplaySegments() {
      state.displayedSegments = state.projectedSegments.map(segment => (
        simplifyPolyline(segment, state.style === "napkin" ? 10 : 5)
      ));
    }
    function buildSegmentCumulative(projectedSegments) {
      const segmentCumulative = [];
      const routePositions = [];
      let total = 0;
      for (let segmentIndex = 0; segmentIndex < projectedSegments.length; segmentIndex += 1) {
        const segment = projectedSegments[segmentIndex];
        const values = [];
        for (let pointIndex = 0; pointIndex < segment.length; pointIndex += 1) {
          if (pointIndex > 0) total += distance(segment[pointIndex - 1], segment[pointIndex]);
          values.push(total);
          segment[pointIndex].routeM = total;
          segment[pointIndex].segmentIndex = segmentIndex;
          segment[pointIndex].pointIndex = pointIndex;
          routePositions.push({ ...segment[pointIndex] });
        }
        segmentCumulative.push(values);
      }
      return { segmentCumulative, routePositions, totalRouteM: total };
    }
    function positionForRouteM(routeM) {
      if (!state.projectedSegments.length) return null;
      const target = Math.max(0, Math.min(routeM, state.totalRouteM || 0));
      for (let segmentIndex = 0; segmentIndex < state.projectedSegments.length; segmentIndex += 1) {
        const segment = state.projectedSegments[segmentIndex];
        const cumulative = state.cumulativeM[segmentIndex] || [];
        if (!segment.length) continue;
        if (target <= cumulative[cumulative.length - 1] || segmentIndex === state.projectedSegments.length - 1) {
          for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
            if (cumulative[pointIndex] >= target) {
              const before = cumulative[pointIndex - 1];
              const after = cumulative[pointIndex];
              const t = after === before ? 0 : (target - before) / (after - before);
              const a = segment[pointIndex - 1];
              const b = segment[pointIndex];
              return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, routeM: target, segmentIndex, pointIndex };
            }
          }
          const last = segment[segment.length - 1];
          return { ...last, routeM: target, segmentIndex, pointIndex: segment.length - 1 };
        }
      }
      const lastSegment = state.projectedSegments[state.projectedSegments.length - 1];
      const last = lastSegment[lastSegment.length - 1];
      return { ...last, routeM: target, segmentIndex: state.projectedSegments.length - 1, pointIndex: lastSegment.length - 1 };
    }
    function displayedRoutePositionForM(routeM) {
      if (!state.displayedSegments.length) return null;
      const target = Math.max(0, Math.min(routeM, state.totalRouteM || 0));
      let fallback = null;
      for (let segmentIndex = 0; segmentIndex < state.displayedSegments.length; segmentIndex += 1) {
        const segment = state.displayedSegments[segmentIndex];
        if (!segment.length) continue;
        if (!fallback) fallback = { ...segment[0], angle: 0 };
        for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
          const a = segment[pointIndex - 1];
          const b = segment[pointIndex];
          const aM = a.routeM || 0;
          const bM = b.routeM || 0;
          if (target < Math.min(aM, bM) || target > Math.max(aM, bM)) continue;
          const span = bM - aM;
          const t = span === 0 ? 0 : (target - aM) / span;
          return {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            routeM: target,
            angle: Math.atan2(b.y - a.y, b.x - a.x),
            segmentIndex,
            pointIndex
          };
        }
        fallback = { ...segment[segment.length - 1], angle: fallback.angle || 0 };
      }
      return fallback;
    }
    function tangentForRouteM(routeM) {
      const position = positionForRouteM(routeM);
      if (!position) return null;
      const segment = state.projectedSegments[position.segmentIndex] || [];
      const next = segment[Math.min(position.pointIndex + 1, segment.length - 1)];
      const previous = segment[Math.max(position.pointIndex - 1, 0)];
      if (!next || !previous) return null;
      return Math.atan2(next.y - previous.y, next.x - previous.x);
    }
    function interpolateRoutePoint(a, b, routeM) {
      const span = (b.routeM || 0) - (a.routeM || 0);
      const t = span === 0 ? 0 : (routeM - (a.routeM || 0)) / span;
      return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        routeM
      };
    }
    function projectPointToRoute(point) {
      if (!state.projectedSegments.length) return null;
      const projectedUser = state.project(point);
      let best = null;
      for (let segmentIndex = 0; segmentIndex < state.projectedSegments.length; segmentIndex += 1) {
        const segment = state.projectedSegments[segmentIndex];
        const cumulative = state.cumulativeM[segmentIndex] || [];
        for (let pointIndex = 1; pointIndex < segment.length; pointIndex += 1) {
          const a = segment[pointIndex - 1];
          const b = segment[pointIndex];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len2 = dx * dx + dy * dy || 1;
          const t = Math.max(0, Math.min(1, ((projectedUser.x - a.x) * dx + (projectedUser.y - a.y) * dy) / len2));
          const snap = { x: a.x + t * dx, y: a.y + t * dy };
          const dist = distance(projectedUser, snap);
          if (!best || dist < best.distanceM) {
            const legM = distance(a, b) * t;
            best = { point: projectedUser, snap, distanceM: dist, routeM: cumulative[pointIndex - 1] + legM, segmentIndex, pointIndex };
          }
        }
      }
      return best;
    }
    function cueForRouteM(routeM) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return null;
      const next = cues.find(cue => metersFromMiles(cue.cum_miles) >= routeM - 30);
      return next || cues[cues.length - 1];
    }
    function cueLabel(cue) {
      if (!cue) return "No cue data for this route.";
      const seq = String(cue.seq || "").padStart(2, "0");
      const type = String(cue.cue_type || "cue").replaceAll("_", " ").toUpperCase();
      const action = cue.action ? `${cue.action}: ` : "";
      const target = cue.target ? ` toward ${cue.target}` : "";
      return `${seq} ${type} - ${action}${(cue.signed_as || []).join(" / ")}${target}`;
    }
    function cueSeq(cue, fallbackIndex) {
      return String(cue?.seq || fallbackIndex + 1).padStart(2, "0");
    }
    function cueRouteM(cue) {
      return Math.max(0, Math.min(metersFromMiles(cue?.cum_miles), state.totalRouteM || 0));
    }
    function activeLegRange(index = state.activeCueIndex) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return { startM: 0, endM: state.totalRouteM || 0, cue: null, nextCue: null, index: 0, nextIndex: null };
      const clamped = Math.max(0, Math.min(index, cues.length - 1));
      const cue = cues[clamped];
      const startM = cueRouteM(cue);
      let endM = state.totalRouteM || 0;
      let nextIndex = null;
      for (let candidateIndex = clamped + 1; candidateIndex < cues.length; candidateIndex += 1) {
        const candidateM = cueRouteM(cues[candidateIndex]);
        if (candidateM > startM + 8) {
          endM = candidateM;
          nextIndex = candidateIndex;
          break;
        }
      }
      if (endM <= startM + 8 && (state.totalRouteM || 0) > startM + 8) endM = state.totalRouteM;
      return { startM, endM, cue, nextCue: nextIndex === null ? null : cues[nextIndex], index: clamped, nextIndex };
    }
    function cueIndexForRouteM(routeM) {
      const cues = state.route?.wayfinding_cues || [];
      if (!cues.length) return 0;
      let active = 0;
      for (let index = 0; index < cues.length; index += 1) {
        if (cueRouteM(cues[index]) <= routeM + 25) active = index;
      }
      return active;
    }
    function updateActiveCuePanel() {
      const cues = state.route?.wayfinding_cues || [];
      const leg = activeLegRange();
      if (!cues.length || !leg.cue) {
        nearestCue.textContent = "No cue data for this route.";
      } else {
        const currentSeq = cueSeq(leg.cue, leg.index);
        const nextSeq = leg.nextCue ? cueSeq(leg.nextCue, leg.nextIndex) : "finish";
        const legMiles = miles(Math.max(0, leg.endM - leg.startM)).toFixed(2);
        nearestCue.textContent = `Cue ${currentSeq} -> ${nextSeq} · +${legMiles} mi · ${cueLabel(leg.cue)}`;
      }
      previousCue.disabled = !cues.length || state.activeCueIndex <= 0;
      nextCue.disabled = !cues.length || state.activeCueIndex >= cues.length - 1;
      updateMapLegBanner();
    }
    function updateMapLegBanner() {
      const leg = activeLegRange();
      if (!leg.cue) {
        mapLegBanner.hidden = true;
        mapLegBanner.textContent = "";
        return;
      }
      const fromSeq = cueSeq(leg.cue, leg.index);
      const toSeq = leg.nextCue ? cueSeq(leg.nextCue, leg.nextIndex) : "finish";
      const signedAs = (leg.cue.signed_as || []).join(" / ") || "active route";
      const target = leg.cue.target ? ` to ${escapeText(leg.cue.target)}` : "";
      const until = leg.cue.until ? ` until ${escapeText(leg.cue.until)}` : "";
      mapLegBanner.hidden = false;
      mapLegBanner.innerHTML = `<b>FOLLOW ${fromSeq} -> ${toSeq}</b>: ${escapeText(signedAs)}${target}${until}.`;
    }
    function setActiveCueIndex(index, options = {}) {
      const cues = state.route?.wayfinding_cues || [];
      state.activeCueIndex = cues.length ? Math.max(0, Math.min(index, cues.length - 1)) : 0;
      updateActiveCuePanel();
      if (options.fit) fitActiveLeg(Boolean(state.user));
      if (options.render !== false) render();
    }
    function setViewBox(box) {
      state.viewBox = box;
      svg.setAttribute("viewBox", `${box.x} ${box.y} ${box.w} ${box.h}`);
    }
    function svgPointFromClient(clientX, clientY) {
      const box = state.viewBox || state.baseViewBox;
      const rect = svg.getBoundingClientRect();
      if (!box || !rect.width || !rect.height) return null;
      return {
        x: box.x + ((clientX - rect.left) / rect.width) * box.w,
        y: box.y + ((clientY - rect.top) / rect.height) * box.h
      };
    }
    function panViewBox(deltaX, deltaY) {
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      setViewBox({ x: box.x + deltaX, y: box.y + deltaY, w: box.w, h: box.h });
    }
    function zoomAt(scale, center) {
      const box = state.viewBox || state.baseViewBox;
      if (!box || !center || !Number.isFinite(scale) || scale <= 0) return;
      const nextW = Math.max(40, Math.min(box.w * scale, 20000));
      const nextH = Math.max(40, Math.min(box.h * scale, 20000));
      const widthScale = nextW / box.w;
      const heightScale = nextH / box.h;
      setViewBox({
        x: center.x - (center.x - box.x) * widthScale,
        y: center.y - (center.y - box.y) * heightScale,
        w: nextW,
        h: nextH
      });
    }
    function pointerDistance(a, b) {
      return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    }
    function pointerCenter(a, b) {
      return { clientX: (a.clientX + b.clientX) / 2, clientY: (a.clientY + b.clientY) / 2 };
    }
    function resetGestureFromPointers() {
      const pointers = [...activePointers.values()];
      if (pointers.length === 1) {
        gesture.mode = "pan";
        gesture.lastPoint = svgPointFromClient(pointers[0].clientX, pointers[0].clientY);
        gesture.lastCenter = null;
        gesture.lastDistance = 0;
      } else if (pointers.length >= 2) {
        const center = pointerCenter(pointers[0], pointers[1]);
        gesture.mode = "pinch";
        gesture.lastPoint = null;
        gesture.lastCenter = svgPointFromClient(center.clientX, center.clientY);
        gesture.lastDistance = pointerDistance(pointers[0], pointers[1]);
      } else {
        gesture.mode = null;
        gesture.lastPoint = null;
        gesture.lastCenter = null;
        gesture.lastDistance = 0;
      }
    }
    function mapUnitsPerPixel() {
      const box = state.viewBox || state.baseViewBox;
      const width = svg.clientWidth || 1;
      const height = svg.clientHeight || 1;
      if (!box) return 1;
      return Math.max(box.w / width, box.h / height);
    }
    function fitRoute(includeUser = false) {
      const points = [...state.projected];
      if (includeUser && state.user) points.push(state.project(state.user));
      if (!points.length) return;
      const xs = points.map(point => point.x);
      const ys = points.map(point => point.y);
      const width = Math.max(Math.max(...xs) - Math.min(...xs), 120);
      const height = Math.max(Math.max(...ys) - Math.min(...ys), 120);
      const pad = Math.max(width, height) * 0.18 + 80;
      const box = { x: Math.min(...xs) - pad, y: Math.min(...ys) - pad, w: width + pad * 2, h: height + pad * 2 };
      state.baseViewBox = box;
      setViewBox(box);
    }
    function fitActiveLeg(includeUser = false) {
      const leg = activeLegRange();
      const legSegments = segmentsForRouteRange(leg.startM, leg.endM);
      const points = legSegments.flat();
      if (includeUser && state.user) points.push(state.project(state.user));
      if (!points.length) return fitRoute(includeUser);
      const xs = points.map(point => point.x);
      const ys = points.map(point => point.y);
      const width = Math.max(Math.max(...xs) - Math.min(...xs), 70);
      const height = Math.max(Math.max(...ys) - Math.min(...ys), 70);
      const pad = Math.max(width, height) * 0.35 + 70;
      setViewBox({ x: Math.min(...xs) - pad, y: Math.min(...ys) - pad, w: width + pad * 2, h: height + pad * 2 });
    }
    function zoom(factor) {
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      zoomAt(factor, { x: box.x + box.w / 2, y: box.y + box.h / 2 });
    }
    function drawGrid() {
      const box = state.viewBox || state.baseViewBox;
      if (!box) return;
      const spacing = Math.max(100, Math.round(Math.max(box.w, box.h) / 6 / 50) * 50);
      const lines = [];
      for (let x = Math.floor(box.x / spacing) * spacing; x < box.x + box.w; x += spacing) {
        lines.push(`<line class="grid-line" x1="${x}" y1="${box.y}" x2="${x}" y2="${box.y + box.h}" />`);
      }
      for (let y = Math.floor(box.y / spacing) * spacing; y < box.y + box.h; y += spacing) {
        lines.push(`<line class="grid-line" x1="${box.x}" y1="${y}" x2="${box.x + box.w}" y2="${y}" />`);
      }
      gridLayer.innerHTML = lines.join("");
    }
    const MAX_OVERVIEW_CHEVRONS = 18;
    function chevrons(maxCount = MAX_OVERVIEW_CHEVRONS, startM = 0, endM = state.totalRouteM) {
      const total = state.totalRouteM || 0;
      const span = Math.max(0, Math.min(endM, total) - Math.max(0, startM));
      if (!span) return "";
      const spacing = Math.max(120, span / Math.max(maxCount, 1));
      const items = [];
      for (let target = startM + spacing; target < endM; target += spacing) {
        const center = positionForRouteM(target);
        const angle = tangentForRouteM(target);
        if (!center || angle === null) continue;
        const size = state.style === "napkin" ? 20 : 14;
        const wing = Math.PI * 0.78;
        const p1 = { x: center.x - Math.cos(angle - wing) * size, y: center.y - Math.sin(angle - wing) * size };
        const p2 = { x: center.x - Math.cos(angle + wing) * size, y: center.y - Math.sin(angle + wing) * size };
        items.push(`<path class="chevron" d="M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} L ${center.x.toFixed(1)} ${center.y.toFixed(1)} L ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}" />`);
      }
      return items.join("");
    }
    function activeLegArrows(startM, endM) {
      const span = Math.max(0, endM - startM);
      if (span < 80) return "";
      const unit = mapUnitsPerPixel();
      const arrowSpacing = state.style === "napkin" ? 115 : 145;
      const inset = Math.min(90, span * 0.16);
      const items = [];
      let count = 0;
      for (let target = startM + inset; target < endM - inset && count < 28; target += arrowSpacing) {
        const sample = displayedRoutePositionForM(target);
        if (!sample || sample.angle === null) continue;
        const center = sample;
        const angle = sample.angle;
        const size = (state.style === "napkin" ? 19 : 15) * unit;
        const baseAngle = angle - Math.PI;
        const tip = { x: center.x + Math.cos(angle) * size * 0.72, y: center.y + Math.sin(angle) * size * 0.72 };
        const left = { x: center.x + Math.cos(baseAngle - 0.48) * size, y: center.y + Math.sin(baseAngle - 0.48) * size };
        const right = { x: center.x + Math.cos(baseAngle + 0.48) * size, y: center.y + Math.sin(baseAngle + 0.48) * size };
        items.push(`<path class="direction-arrow" d="M ${tip.x.toFixed(1)} ${tip.y.toFixed(1)} L ${left.x.toFixed(1)} ${left.y.toFixed(1)} L ${right.x.toFixed(1)} ${right.y.toFixed(1)} Z" />`);
        count += 1;
      }
      return items.join("");
    }
    const ROUTE_GRADIENT_STOPS = [
      { at: 0, color: [37, 99, 235] },
      { at: 0.55, color: [124, 58, 237] },
      { at: 1, color: [220, 38, 38] }
    ];
    function routeColorAt(progress) {
      const p = Math.max(0, Math.min(1, progress));
      for (let index = 1; index < ROUTE_GRADIENT_STOPS.length; index += 1) {
        const left = ROUTE_GRADIENT_STOPS[index - 1];
        const right = ROUTE_GRADIENT_STOPS[index];
        if (p <= right.at) {
          const local = right.at === left.at ? 0 : (p - left.at) / (right.at - left.at);
          const color = left.color.map((value, channel) => Math.round(value + (right.color[channel] - value) * local));
          return `rgb(${color[0]} ${color[1]} ${color[2]})`;
        }
      }
      const last = ROUTE_GRADIENT_STOPS[ROUTE_GRADIENT_STOPS.length - 1].color;
      return `rgb(${last[0]} ${last[1]} ${last[2]})`;
    }
    function segmentsForRouteRange(startM, endM) {
      const output = [];
      for (const segment of state.displayedSegments) {
        const slice = [];
        for (let index = 1; index < segment.length; index += 1) {
          const a = segment[index - 1];
          const b = segment[index];
          const aM = a.routeM || 0;
          const bM = b.routeM || 0;
          if (bM < startM || aM > endM) continue;
          const start = Math.max(aM, startM);
          const end = Math.min(bM, endM);
          const startPoint = start === aM ? a : interpolateRoutePoint(a, b, start);
          const endPoint = end === bM ? b : interpolateRoutePoint(a, b, end);
          if (!slice.length || distance(slice[slice.length - 1], startPoint) > 0.1) slice.push(startPoint);
          slice.push(endPoint);
        }
        if (slice.length > 1) output.push(slice);
      }
      return output;
    }
    function drawProgressRibbon() {
      const total = state.totalRouteM || 1;
      const defs = [];
      const slices = [];
      let gradientIndex = 0;
      for (const segment of segmentsForRouteRange(0, state.totalRouteM)) {
        for (let index = 1; index < segment.length; index += 1) {
          const a = segment[index - 1];
          const b = segment[index];
          const gradientId = `route-gradient-${gradientIndex}`;
          const startColor = routeColorAt((a.routeM || 0) / total);
          const endColor = routeColorAt((b.routeM || 0) / total);
          const modeClass = state.style === "napkin" ? " napkin" : "";
          defs.push(`<linearGradient id="${gradientId}" gradientUnits="userSpaceOnUse" x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}"><stop offset="0%" stop-color="${startColor}" /><stop offset="100%" stop-color="${endColor}" /></linearGradient>`);
          slices.push(`<path class="route-slice${modeClass}" stroke="url(#${gradientId})" d="M ${a.x.toFixed(1)} ${a.y.toFixed(1)} L ${b.x.toFixed(1)} ${b.y.toFixed(1)}" />`);
          gradientIndex += 1;
        }
      }
      return `<defs>${defs.join("")}</defs>${slices.join("")}`;
    }
    function drawRoute() {
      const cueLegColors = ["#2563eb", "#06b6d4", "#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7", "#0f766e"];
      const visibleSegments = segmentsForRouteRange(0, state.totalRouteM);
      const fullPath = pathForSegments(visibleSegments);
      let routeHtml = `<path class="route-context" d="${fullPath}" />`;
      if (state.style === "cue-legs") {
        const cueStops = (state.route?.wayfinding_cues || []).map(cue => metersFromMiles(cue.cum_miles));
        const stops = [...new Set([0, ...cueStops, state.totalRouteM])].filter(value => Number.isFinite(value) && value <= state.totalRouteM).sort((a, b) => a - b);
        routeHtml += `<g class="route-context-gradient">`;
        for (let index = 1; index < stops.length; index += 1) {
          const legSegments = segmentsForRouteRange(stops[index - 1], stops[index]);
          routeHtml += legSegments.map(leg => `<path class="cue-leg" stroke="${cueLegColors[index % cueLegColors.length]}" d="${pathFor(leg)}" />`).join("");
        }
        routeHtml += `</g>`;
      } else if (state.style === "ribbon") {
        routeHtml += `<g class="route-context-gradient">${drawProgressRibbon()}</g>`;
      }
      const leg = activeLegRange();
      const activeSegments = segmentsForRouteRange(leg.startM, leg.endM);
      const activePath = pathForSegments(activeSegments);
      if (activePath) {
        routeHtml += `<path class="active-halo" d="${activePath}" /><path class="active-line" d="${activePath}" />`;
      }
      routeLayer.innerHTML = routeHtml + activeLegArrows(leg.startM, leg.endM);
    }
    function drawMarkers() {
      const placed = [];
      const leg = activeLegRange();
      const unit = mapUnitsPerPixel();
      const cueMarkers = (state.route?.wayfinding_cues || [])
        .slice(0, 24)
        .map((cue, index) => {
          const cueM = Math.min(metersFromMiles(cue.cum_miles), state.totalRouteM);
          const point = displayedRoutePositionForM(cueM) || positionForRouteM(cueM);
          if (!point) return "";
          const nearby = placed.filter(existing => distance(existing, point) < 30).length;
          placed.push(point);
          const number = String(cue.seq || index + 1).padStart(2, "0");
          const title = escapeText(cue.compact || cueLabel(cue));
          const isActive = index === state.activeCueIndex;
          const isNext = index === leg.nextIndex;
          const isCallout = isActive || isNext;
          const calloutAngle = (point.angle || 0) + (isActive ? -Math.PI / 2 : Math.PI / 2);
          const calloutDistance = isCallout ? 44 * unit : nearby ? 24 * unit : 0;
          const fallbackAngle = nearby * Math.PI * 0.75;
          const markerAngle = isCallout ? calloutAngle : fallbackAngle;
          const marker = { x: point.x + Math.cos(markerAngle) * calloutDistance, y: point.y + Math.sin(markerAngle) * calloutDistance };
          const roleClass = isActive ? " active" : isNext ? " next" : " context-marker";
          const labelClass = isActive || isNext ? "cue-label" : "cue-label context-label";
          const anchorClass = isActive ? " active" : isNext ? " next" : "";
          const radiusForCue = (isActive || isNext ? 16 : 6) * unit;
          const anchorRadius = (isActive || isNext ? 5 : 0) * unit;
          const labelFontSize = (isActive || isNext ? 16 : 10) * unit;
          const tagFontSize = 18 * unit;
          const tagStrokeWidth = 6 * unit;
          const legTag = isActive || isNext
            ? `<text class="leg-tag" x="${(marker.x + 22 * unit).toFixed(1)}" y="${(marker.y - 21 * unit).toFixed(1)}" font-size="${tagFontSize.toFixed(1)}" stroke-width="${tagStrokeWidth.toFixed(1)}">${isActive ? "FROM" : "NEXT"} ${escapeText(number)}</text>`
            : "";
          const anchor = anchorRadius
            ? `<circle class="cue-anchor${anchorClass}" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${anchorRadius.toFixed(1)}" />`
            : "";
          return `<g><title>${title}</title>${anchor}<line class="cue-callout-line" x1="${point.x.toFixed(1)}" y1="${point.y.toFixed(1)}" x2="${marker.x.toFixed(1)}" y2="${marker.y.toFixed(1)}" /><line class="cue-callout-line dark" x1="${point.x.toFixed(1)}" y1="${point.y.toFixed(1)}" x2="${marker.x.toFixed(1)}" y2="${marker.y.toFixed(1)}" /><circle class="cue-dot${roleClass}" cx="${marker.x.toFixed(1)}" cy="${marker.y.toFixed(1)}" r="${radiusForCue.toFixed(1)}" /><text class="${labelClass}" x="${marker.x.toFixed(1)}" y="${marker.y.toFixed(1)}" font-size="${labelFontSize.toFixed(1)}">${escapeText(number)}</text>${legTag}</g>`;
        });
      const first = displayedRoutePositionForM(0) || positionForRouteM(0);
      const last = displayedRoutePositionForM(state.totalRouteM) || positionForRouteM(state.totalRouteM);
      const sameStartFinish = first && last && distance(first, last) < 25;
      const endpointMarkers = sameStartFinish
        ? [
            `<circle class="parking-dot" cx="${first.x.toFixed(1)}" cy="${first.y.toFixed(1)}" r="17"><title>START / FINISH / CAR</title></circle>`,
            `<circle class="finish-dot" cx="${(first.x + 18).toFixed(1)}" cy="${(first.y + 4).toFixed(1)}" r="9"><title>FINISH</title></circle>`,
            `<text class="marker-tag" x="${(first.x + 24).toFixed(1)}" y="${(first.y + 28).toFixed(1)}">START/FINISH</text>`
          ]
        : [
            first ? `<circle class="parking-dot" cx="${first.x.toFixed(1)}" cy="${first.y.toFixed(1)}" r="17"><title>START / CAR</title></circle><text class="marker-tag" x="${(first.x + 20).toFixed(1)}" y="${(first.y + 5).toFixed(1)}">START</text>` : "",
            last ? `<circle class="finish-dot" cx="${last.x.toFixed(1)}" cy="${last.y.toFixed(1)}" r="15"><title>FINISH</title></circle><text class="marker-tag" x="${(last.x + 20).toFixed(1)}" y="${(last.y + 5).toFixed(1)}">FINISH</text>` : ""
          ];
      markerLayer.innerHTML = [
        ...endpointMarkers,
        ...cueMarkers
      ].join("");
    }
    function drawUser() {
      userLayer.innerHTML = "";
      if (!state.user) return;
      const point = state.project(state.user);
      const accuracy = Math.max(Number(state.user.accuracy || 0), 8);
      const heading = Number.isFinite(state.user.heading) ? state.user.heading : null;
      const headingTriangle = heading === null ? "" : (() => {
        const angle = (heading - 90) * Math.PI / 180;
        const tip = { x: point.x + Math.cos(angle) * 22, y: point.y + Math.sin(angle) * 22 };
        const left = { x: point.x + Math.cos(angle + 2.5) * 13, y: point.y + Math.sin(angle + 2.5) * 13 };
        const right = { x: point.x + Math.cos(angle - 2.5) * 13, y: point.y + Math.sin(angle - 2.5) * 13 };
        return `<path class="user-heading" d="M ${tip.x.toFixed(1)} ${tip.y.toFixed(1)} L ${left.x.toFixed(1)} ${left.y.toFixed(1)} L ${right.x.toFixed(1)} ${right.y.toFixed(1)} Z" />`;
      })();
      userLayer.innerHTML = `<circle class="user-accuracy" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${accuracy.toFixed(1)}" />${headingTriangle}<circle class="user-dot" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="10" />`;
    }
    function render() {
      drawGrid();
      drawRoute();
      drawMarkers();
      drawUser();
    }
    async function loadRoute(route) {
      state.route = route;
      localStorage.setItem(ACTIVE_KEY, route.outing_id);
      routeSelect.value = route.outing_id;
      nearestCue.textContent = "Loading GPX...";
      const response = await fetch(route.gpx_href);
      const gpx = parseGpx(await response.text());
      state.trackSegments = gpx.trackSegments;
      state.waypoints = gpx.waypoints;
      const flatTrack = state.trackSegments.flat();
      const points = [...flatTrack, ...state.waypoints];
      if (!flatTrack.length || !points.length) {
        nearestCue.textContent = "No track geometry found for this outing.";
        return;
      }
      state.project = makeProjector(points);
      state.projectedSegments = state.trackSegments.map(segment => segment.map(state.project));
      const metrics = buildSegmentCumulative(state.projectedSegments);
      state.cumulativeM = metrics.segmentCumulative;
      state.routePositions = metrics.routePositions;
      state.totalRouteM = metrics.totalRouteM;
      state.displayedSegments = state.projectedSegments.map(segment => simplifyPolyline(segment, state.style === "napkin" ? 10 : 5));
      state.activeCueIndex = 0;
      state.gapWarnings = [];
      for (let index = 1; index < state.projectedSegments.length; index += 1) {
        const prior = state.projectedSegments[index - 1][state.projectedSegments[index - 1].length - 1];
        const next = state.projectedSegments[index][0];
        const gap = distance(prior, next);
        if (gap > 80) state.gapWarnings.push(`${fmtDistance(gap)} between GPX track parts ${index} and ${index + 1}`);
      }
      const plannedMeters = route.on_foot_miles ? metersFromMiles(route.on_foot_miles) : null;
      if (plannedMeters && Math.abs(plannedMeters - state.totalRouteM) > metersFromMiles(0.35)) {
        state.gapWarnings.push(`Nav GPX length ${miles(state.totalRouteM).toFixed(2)} mi differs from route card ${Number(route.on_foot_miles).toFixed(2)} mi`);
      }
      state.projected = state.routePositions;
      if (state.gapWarnings.length) {
        mapWarning.hidden = false;
        mapWarning.textContent = `Route review needed: ${state.gapWarnings.join("; ")}. Do not treat gaps as runnable connectors.`;
      } else {
        mapWarning.hidden = true;
        mapWarning.textContent = "";
      }
      setActiveCueIndex(cueIndexForRouteM(0), { render: false });
      fitActiveLeg(false);
      routeProgress.textContent = fmtProgress(0);
      render();
    }
    async function boot() {
      const response = await fetch(FIELD_DATA_URL);
      const data = await response.json();
      state.routes = data.routes || [];
      routeSelect.innerHTML = state.routes.map(route => `<option value="${escapeText(route.outing_id)}">${escapeText(route.label)}. ${escapeText(route.trailhead)}</option>`).join("");
      const params = new URLSearchParams(window.location.search);
      const preferred = params.get("outing") || localStorage.getItem(ACTIVE_KEY) || state.routes[0]?.outing_id;
      const route = state.routes.find(item => item.outing_id === preferred) || state.routes[0];
      if (route) await loadRoute(route);
    }
    routeSelect.addEventListener("change", () => {
      const route = state.routes.find(item => item.outing_id === routeSelect.value);
      if (route) loadRoute(route);
    });
    document.querySelectorAll("[data-style]").forEach(button => button.addEventListener("click", () => {
      document.querySelectorAll("[data-style]").forEach(item => item.classList.toggle("active", item === button));
      state.style = button.dataset.style;
      refreshDisplaySegments();
      render();
    }));
    document.getElementById("fit-button").addEventListener("click", () => { fitRoute(Boolean(state.user)); render(); });
    fitLegButton.addEventListener("click", () => { fitActiveLeg(Boolean(state.user)); render(); });
    previousCue.addEventListener("click", () => setActiveCueIndex(state.activeCueIndex - 1, { fit: true }));
    nextCue.addEventListener("click", () => setActiveCueIndex(state.activeCueIndex + 1, { fit: true }));
    document.getElementById("zoom-in").addEventListener("click", () => { zoom(0.72); render(); });
    document.getElementById("zoom-out").addEventListener("click", () => { zoom(1.38); render(); });
    svg.addEventListener("pointerdown", event => {
      svg.setPointerCapture(event.pointerId);
      activePointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
      resetGestureFromPointers();
      event.preventDefault();
    });
    svg.addEventListener("pointermove", event => {
      if (!activePointers.has(event.pointerId)) return;
      activePointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });
      const pointers = [...activePointers.values()];
      if (pointers.length === 1 && gesture.mode === "pan") {
        const current = svgPointFromClient(event.clientX, event.clientY);
        if (current && gesture.lastPoint) {
          panViewBox(gesture.lastPoint.x - current.x, gesture.lastPoint.y - current.y);
          gesture.lastPoint = svgPointFromClient(event.clientX, event.clientY);
          render();
        }
      } else if (pointers.length >= 2) {
        const centerClient = pointerCenter(pointers[0], pointers[1]);
        const currentDistance = pointerDistance(pointers[0], pointers[1]);
        const center = svgPointFromClient(centerClient.clientX, centerClient.clientY);
        if (gesture.mode !== "pinch" || !gesture.lastDistance || !gesture.lastCenter) {
          resetGestureFromPointers();
        } else if (currentDistance > 12 && center) {
          zoomAt(gesture.lastDistance / currentDistance, center);
          const recentered = svgPointFromClient(centerClient.clientX, centerClient.clientY);
          if (recentered && gesture.lastCenter) {
            panViewBox(gesture.lastCenter.x - recentered.x, gesture.lastCenter.y - recentered.y);
          }
          gesture.lastDistance = currentDistance;
          gesture.lastCenter = svgPointFromClient(centerClient.clientX, centerClient.clientY);
          render();
        }
      }
      event.preventDefault();
    });
    svg.addEventListener("pointerup", event => {
      activePointers.delete(event.pointerId);
      resetGestureFromPointers();
      event.preventDefault();
    });
    svg.addEventListener("pointercancel", event => {
      activePointers.delete(event.pointerId);
      resetGestureFromPointers();
      event.preventDefault();
    });
    svg.addEventListener("wheel", event => {
      const center = svgPointFromClient(event.clientX, event.clientY);
      zoomAt(event.deltaY < 0 ? 0.82 : 1.22, center);
      render();
      event.preventDefault();
    }, { passive: false });
    locateButton.addEventListener("click", () => {
      if (!navigator.geolocation) {
        nearestCue.textContent = "Geolocation is not available in this browser.";
        return;
      }
      if (state.watchId !== null) {
        navigator.geolocation.clearWatch(state.watchId);
        state.watchId = null;
        locateButton.textContent = "Start GPS";
        return;
      }
      locateButton.textContent = "Stop GPS";
      state.watchId = navigator.geolocation.watchPosition(position => {
        state.user = {
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          accuracy: position.coords.accuracy,
          heading: position.coords.heading
        };
        const nearest = projectPointToRoute(state.user);
        if (nearest) {
          distanceToRoute.textContent = fmtDistance(nearest.distanceM);
          routeProgress.textContent = fmtProgress(nearest.routeM);
        }
        gpsAccuracy.textContent = fmtDistance(position.coords.accuracy);
        render();
      }, error => {
        nearestCue.textContent = `GPS unavailable: ${error.message}`;
        locateButton.textContent = "Start GPS";
      }, { enableHighAccuracy: true, maximumAge: 4000, timeout: 15000 });
    });
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("service-worker.js").catch(() => {});
    }
    boot().catch(error => { nearestCue.textContent = `Unable to load field data: ${error.message}`; });
  </script>
</body>
</html>
""".replace("__FIELD_TOOL_DATA__", FIELD_TOOL_DATA_NAME).replace("__ACTIVE_KEY__", ACTIVE_STORAGE_KEY)


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", binascii.crc32(tag + payload) & 0xFFFFFFFF)
    )


def solid_png_bytes(size: int, rgb: tuple[int, int, int] = (17, 24, 39)) -> bytes:
    """Build a simple standards-compliant RGB PNG without optional dependencies."""
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )


def render_web_manifest() -> str:
    manifest = {
        "name": "Boise Trails Field Packet",
        "short_name": "Trails Packet",
        "description": "Offline field cards and GPX routes for Boise Trails Challenge outings.",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "orientation": "portrait",
        "theme_color": "#111827",
        "background_color": "#f5f7f2",
        "icons": [
            {
                "src": "icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable",
            },
            {
                "src": "icons/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable",
            },
        ],
    }
    return json.dumps(manifest, indent=2) + "\n"


def render_service_worker(precache_urls: list[str], cache_name: str) -> str:
    urls = ["./"] + unique_nonempty_text(precache_urls)
    return f"""const CACHE_NAME = "{cache_name}";
const PRECACHE_URLS = {json.dumps(urls, indent=2)};
const NETWORK_FIRST_URLS = new Set([
  "index.html",
  "live-map.html",
  "field-tool-data.json",
  "manifest.json",
  "manifest.webmanifest"
]);

function normalizedCacheKey(request) {{
  const requestUrl = new URL(request.url);
  requestUrl.search = '';
  return requestUrl.href;
}}

function shouldUseNetworkFirst(request) {{
  const requestUrl = new URL(request.url);
  const filename = requestUrl.pathname.split('/').pop() || 'index.html';
  return NETWORK_FIRST_URLS.has(filename) || requestUrl.pathname.includes('/gpx/');
}}

self.addEventListener('install', event => {{
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
}});

self.addEventListener('activate', event => {{
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
    )).then(() => self.clients.claim())
  );
}});

self.addEventListener('fetch', event => {{
  if (event.request.method !== 'GET') {{
    return;
  }}
  const cacheKey = normalizedCacheKey(event.request);
  if (shouldUseNetworkFirst(event.request)) {{
    event.respondWith(
      fetch(event.request).then(response => {{
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(cacheKey, copy));
        return response;
      }}).catch(() => caches.match(cacheKey).then(cached => cached || caches.match('./index.html')))
    );
    return;
  }}
  event.respondWith(
    caches.match(event.request).then(cached => {{
      if (cached) {{
        return cached;
      }}
      return fetch(event.request).then(response => {{
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        return response;
      }}).catch(() => caches.match('./index.html'));
    }})
  );
}});
"""


def write_pwa_assets(
    output_dir: Path,
    routes: list[dict[str, Any]],
    zip_href: str,
    extra_precache_urls: list[str] | None = None,
) -> list[Path]:
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    icon_192 = icons_dir / "icon-192.png"
    icon_512 = icons_dir / "icon-512.png"
    icon_192.write_bytes(solid_png_bytes(192))
    icon_512.write_bytes(solid_png_bytes(512))
    web_manifest_path = output_dir / "manifest.webmanifest"
    web_manifest_path.write_text(render_web_manifest(), encoding="utf-8")
    precache_urls = [
        "index.html",
        "manifest.json",
        "manifest.webmanifest",
        "icons/icon-192.png",
        "icons/icon-512.png",
        zip_href,
    ]
    precache_urls.extend(extra_precache_urls or [])
    for route in routes:
        precache_urls.extend(route[key] for key in GPX_HREF_KEYS)
    service_worker_path = output_dir / "service-worker.js"
    digest = hashlib.sha256()
    for route in routes:
        for key in GPX_PATH_KEYS:
            digest.update(Path(route[key]).read_bytes())
    digest.update((output_dir / zip_href).read_bytes())
    for href in extra_precache_urls or []:
        extra_path = output_dir / href
        if extra_path.exists() and extra_path.is_file():
            digest.update(extra_path.read_bytes())
    cache_suffix = digest.hexdigest().replace("911", "nineoneone")[:18]
    cache_name = f"boise-trails-field-packet-v{len(routes)}-{cache_suffix}"
    service_worker_path.write_text(
        strip_trailing_whitespace(render_service_worker(precache_urls, cache_name)),
        encoding="utf-8",
    )
    return [icon_192, icon_512, web_manifest_path, service_worker_path]


def write_gpx_zip(gpx_dir: Path, routes: list[dict[str, Any]]) -> Path:
    zip_path = gpx_dir / GPX_ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for route in routes:
            for key in GPX_PATH_KEYS:
                gpx_path = Path(route[key])
                archive.write(gpx_path, arcname=str(gpx_path.relative_to(gpx_dir)))
    return zip_path


def public_safety_check(output_dir: Path, extra_forbidden: list[str] | None = None) -> list[str]:
    forbidden = list(PRIVATE_LITERAL_PATTERNS)
    forbidden.extend(extra_forbidden or [])
    failures = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token and token in text:
                failures.append(f"{path}: contains {token}")
        for pattern in PRIVATE_REGEX_PATTERNS:
            if pattern.search(text):
                failures.append(f"{path}: contains private address pattern")
    return failures


def load_default_certificate_data(path: Path = DEFAULT_CERTIFICATE_JSON) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def apply_progress_to_map_data(map_data: dict[str, Any], progress_data: dict[str, Any] | None) -> dict[str, Any]:
    if not progress_data:
        return map_data
    updated = json.loads(json.dumps(map_data))
    outings = build_outing_menu(updated)
    outings_by_id = {str(outing.get("outing_id")): outing for outing in outings}
    completed_ids = {
        int(seg_id)
        for seg_id in (updated.get("progress") or {}).get("completed_segment_ids") or []
        if seg_id is not None
    }
    completed_ids.update(
        int(seg_id)
        for seg_id in progress_data.get("completed_segment_ids") or []
        if seg_id is not None
    )
    for outing_id in normalized_segment_ids(progress_data.get("completed_outing_ids")):
        outing = outings_by_id.get(outing_id)
        if not outing:
            continue
        completed_ids.update(int(seg_id) for seg_id in outing.get("segment_ids") or [] if seg_id is not None)
    missed_ids = {int(seg_id) for seg_id in progress_data.get("missed_segment_ids") or [] if seg_id is not None}
    blocked_ids = {int(seg_id) for seg_id in progress_data.get("blocked_segment_ids") or [] if seg_id is not None}
    completed_ids.difference_update(missed_ids)
    completed_ids.difference_update(blocked_ids)
    progress = dict(updated.get("progress") or {})
    progress["completed_segment_ids"] = sorted(completed_ids)
    progress["blocked_segment_ids"] = sorted(set(progress.get("blocked_segment_ids") or []) | blocked_ids)
    if missed_ids:
        progress["missed_segment_ids"] = sorted(missed_ids)
    progress["completed_outing_ids"] = normalized_segment_ids(progress_data.get("completed_outing_ids"))
    updated["progress"] = progress
    return updated


def certified_baseline_from_certificate(
    certificate_data: dict[str, Any] | None,
    fallback_segment_count: int,
) -> dict[str, Any]:
    if not certificate_data:
        return {
            "status": "not_loaded",
            "profile_id": None,
            "official_segment_count": fallback_segment_count,
            "covered_segment_count": fallback_segment_count,
            "missing_segment_count": None,
            "field_day_count": None,
            "total_p75_minutes": None,
            "total_on_foot_miles": None,
            "max_on_foot_miles": None,
            "max_p90_minutes": None,
            "day_gpx_validation_passed": None,
            "actual_max_day_trackpoint_gap_miles": None,
        }
    segment_set = certificate_data.get("segment_set") or {}
    field_days = certificate_data.get("field_days") or {}
    gpx_validation = certificate_data.get("gpx_validation") or {}
    profile = certificate_data.get("profile") or {}
    bounds = profile.get("bounds") or {}
    return {
        "status": certificate_data.get("certificate_status"),
        "profile_id": profile.get("profile_id"),
        "official_segment_count": segment_set.get("official_segment_count", fallback_segment_count),
        "covered_segment_count": segment_set.get("selected_calendar_segment_count")
        or segment_set.get("selected_plan_segment_count"),
        "missing_segment_count": segment_set.get("missing_segment_count"),
        "field_day_count": field_days.get("field_day_count"),
        "total_p75_minutes": field_days.get("total_p75_minutes"),
        "total_on_foot_miles": field_days.get("total_on_foot_miles"),
        "max_on_foot_miles": field_days.get("max_on_foot_miles"),
        "max_p90_minutes": field_days.get("max_p90_minutes"),
        "weekday_p90_minutes": bounds.get("weekday_p90_minutes"),
        "weekend_p90_minutes": bounds.get("weekend_p90_minutes"),
        "max_on_foot_miles_per_field_day": bounds.get("max_on_foot_miles_per_field_day"),
        "day_gpx_validation_passed": gpx_validation.get("day_track_validation_passed"),
        "actual_max_day_trackpoint_gap_miles": gpx_validation.get("actual_max_day_trackpoint_gap_miles"),
    }


def route_effort_summary(route: dict[str, Any]) -> dict[str, Any]:
    segment_ascent_ft = 0.0
    segment_descent_ft = 0.0
    segment_grade_adjusted_miles = 0.0
    segment_moving_p50 = 0.0
    segment_moving_p75 = 0.0
    cue_ascent_ft = 0.0
    cue_descent_ft = 0.0
    cue_grade_adjusted_miles = 0.0
    cue_moving_p50 = 0.0
    cue_moving_p75 = 0.0
    seen_segments: set[str] = set()
    used_segment_effort = False
    for cue in route.get("route_cues") or []:
        effort = cue.get("effort") or {}
        estimates = cue.get("time_estimates_minutes") or {}
        cue_ascent_ft += float(effort.get("ascent_ft") or 0)
        cue_descent_ft += float(effort.get("descent_ft") or 0)
        cue_grade_adjusted_miles += float(effort.get("grade_adjusted_miles") or 0)
        cue_moving_p50 += float(estimates.get("moving_effort_p50") or estimates.get("moving_p50") or 0)
        cue_moving_p75 += float(estimates.get("moving_effort_p75") or estimates.get("moving_p75") or 0)
        for segment in cue.get("segments") or []:
            segment_id = str(segment.get("seg_id") or segment.get("segment_id") or "")
            if segment_id and segment_id in seen_segments:
                continue
            if segment_id:
                seen_segments.add(segment_id)
            if segment.get("ascent_ft") is not None or segment.get("descent_ft") is not None:
                used_segment_effort = True
            segment_ascent_ft += float(segment.get("ascent_ft") or 0)
            segment_descent_ft += float(segment.get("descent_ft") or 0)
            segment_grade_adjusted_miles += float(segment.get("grade_adjusted_miles") or 0)
            segment_moving_p50 += float(segment.get("estimated_moving_minutes") or 0)
            segment_moving_p75 += float(segment.get("estimated_moving_minutes_p75") or 0)
    ascent_ft = segment_ascent_ft if used_segment_effort else cue_ascent_ft
    descent_ft = segment_descent_ft if used_segment_effort else cue_descent_ft
    grade_adjusted_miles = segment_grade_adjusted_miles if used_segment_effort else cue_grade_adjusted_miles
    moving_p50 = cue_moving_p50 or segment_moving_p50
    moving_p75 = cue_moving_p75 or segment_moving_p75
    elevation_source = "dem" if ascent_ft or descent_ft or grade_adjusted_miles else "unavailable"
    return {
        "ascent_ft": int(round(ascent_ft)),
        "descent_ft": int(round(descent_ft)),
        "grade_adjusted_miles": round(grade_adjusted_miles, 2) if grade_adjusted_miles else None,
        "estimated_moving_minutes_p50": int(round(moving_p50)) if moving_p50 else None,
        "estimated_moving_minutes_p75": int(round(moving_p75)) if moving_p75 else None,
        "elevation_source": elevation_source,
    }


def route_time_estimate_summary(route: dict[str, Any]) -> dict[str, Any]:
    outing = route.get("outing") or {}
    p75 = int(round(float(outing.get("total_minutes") or 0)))
    p90_values = []
    for cue in route.get("route_cues") or []:
        estimates = cue.get("time_estimates_minutes") or {}
        if estimates.get("door_to_door_p90") is not None:
            p90_values.append(float(estimates["door_to_door_p90"]))
    if len(p90_values) == 1:
        p90 = int(round(p90_values[0]))
    elif p90_values:
        p90 = max(p75, int(round(sum(p90_values))))
    else:
        p90 = int(math.ceil(p75 * 1.12)) if p75 else None
    return {
        "door_to_door_minutes_p75": p75 or None,
        "door_to_door_minutes_p90": p90,
    }


def route_field_tool_record(route: dict[str, Any], completion_safety: dict[str, Any] | None = None) -> dict[str, Any]:
    outing = route["outing"]
    parking = route.get("parking") or {}
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    validation = route.get("validation") or {}
    segment_ids = normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
    time_estimates = route_time_estimate_summary(route)
    return {
        "outing_id": outing.get("outing_id"),
        "label": outing.get("label"),
        "block_name": outing.get("block_name"),
        "trailhead": outing.get("trailhead"),
        "candidate_ids": [str(candidate_id) for candidate_id in outing.get("candidate_ids") or []],
        "trails": outing.get("trails") or [],
        "segment_ids": segment_ids,
        "remaining_segment_count": len(segment_ids),
        "official_miles": outing.get("official_miles"),
        "on_foot_miles": outing.get("on_foot_miles"),
        "door_to_door_minutes_p75": time_estimates.get("door_to_door_minutes_p75"),
        "door_to_door_minutes_p90": time_estimates.get("door_to_door_minutes_p90"),
        "effort": route_effort_summary(route),
        "time_bucket": outing.get("time_bucket"),
        "gpx_href": route.get("gpx_href"),
        "parking_navigation_url": route.get("parking_navigation_url"),
        "parking": {
            "name": parking.get("name"),
            "lat": parking.get("lat"),
            "lon": parking.get("lon"),
            "has_parking": parking.get("has_parking"),
            "has_restroom": parking.get("has_restroom"),
            "has_water": parking.get("has_water"),
            "water_confidence": parking.get("water_confidence"),
            "nearest_open_trail_name": parking.get("nearest_open_trail_name"),
            "nearest_open_trail_label": parking.get("nearest_open_trail_label"),
        },
        "logistics": {
            "car_pass_count": len(logistics.get("car_passes") or []),
            "known_water_count": len(logistics.get("known_water") or []),
            "has_car_pass": bool(logistics.get("car_passes")),
            "has_known_water": bool(logistics.get("known_water")),
        },
        "validation": {
            "passed": validation.get("passed"),
            "max_trackpoint_gap_miles": validation.get("max_trackpoint_gap_miles"),
            "max_allowed_gap_miles": validation.get("max_allowed_gap_miles"),
            "max_allowed_parking_gap_miles": validation.get("max_allowed_parking_gap_miles"),
            "failures": validation.get("failures") or [],
        },
        "navigation_quality": route.get("navigation_quality") or {},
        "source_gap_repair": route.get("source_gap_repair") or {},
        "completion_safety": completion_safety or route.get("completion_safety") or {},
        "segment_direction_evidence": route.get("segment_direction_evidence") or {},
        "turn_by_turn_steps": route.get("turn_by_turn_steps") or [],
        "wayfinding_cues": route.get("wayfinding_cues") or [],
    }


def completion_safety_by_outing(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_segment_ids = {
        segment_id
        for route in routes
        for segment_id in normalized_segment_ids(
            route.get("outing", {}).get("remaining_segment_ids") or route.get("outing", {}).get("segment_ids")
        )
    }
    by_outing: dict[str, dict[str, Any]] = {}
    for route in routes:
        outing = route.get("outing") or {}
        outing_id = str(outing.get("outing_id") or "")
        completed_by_this_route = set(
            normalized_segment_ids(outing.get("remaining_segment_ids") or outing.get("segment_ids"))
        )
        remaining_after = all_segment_ids - completed_by_this_route
        available_after = set()
        for other in routes:
            other_outing = other.get("outing") or {}
            if str(other_outing.get("outing_id") or "") == outing_id:
                continue
            available_after.update(
                normalized_segment_ids(other_outing.get("remaining_segment_ids") or other_outing.get("segment_ids"))
            )
        missing_after = sorted(remaining_after - available_after, key=lambda value: (len(value), value))
        by_outing[outing_id] = {
            "normal_completion_preserves_remaining_menu_coverage": not missing_after,
            "missing_remaining_segment_ids_after_completion": missing_after,
        }
    return by_outing


def build_field_tool_data(
    manifest: dict[str, Any],
    certificate_data: dict[str, Any] | None = None,
    map_data: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_segment_ids = sorted(
        {
            segment_id
            for route in manifest["routes"]
            for segment_id in normalized_segment_ids(
                route.get("outing", {}).get("remaining_segment_ids") or route.get("outing", {}).get("segment_ids")
            )
        },
        key=lambda value: (len(value), value),
    )
    progress = (map_data or {}).get("progress") or {}
    completed_segment_ids = normalized_segment_ids(progress.get("completed_segment_ids"))
    blocked_segment_ids = normalized_segment_ids(progress.get("blocked_segment_ids"))
    safety_by_outing = completion_safety_by_outing(manifest["routes"])
    return {
        "schema": "boise_trails_field_tool_data_v1",
        "source": source_metadata or source_metadata_for_map_data(map_data or {}),
        "time_filters_minutes": TIME_FILTER_MINUTES,
        "certified_baseline": certified_baseline_from_certificate(certificate_data, len(all_segment_ids)),
        "progress": {
            "completed_segment_ids_at_export": completed_segment_ids,
            "blocked_segment_ids_at_export": blocked_segment_ids,
            "remaining_segment_count_at_start": len(set(all_segment_ids) - set(blocked_segment_ids)),
        },
        "summary": {
            "runnable_outing_count": manifest["summary"]["runnable_outing_count"],
            "manual_hold_count": manifest["summary"]["manual_hold_count"],
            "gpx_validation_passed": manifest["summary"]["gpx_validation_passed"],
            "segment_count_in_field_menu": len(all_segment_ids),
            "gpx_zip_href": manifest["summary"].get("gpx_zip_href"),
        },
        "routes": [
            route_field_tool_record(route, safety_by_outing.get(str(route.get("outing_id"))))
            for route in manifest["routes"]
        ],
        "manual_holds": manifest.get("manual_holds") or [],
    }


def load_default_walkthrough_graph_edges() -> list[Any]:
    if not DEFAULT_WALKTHROUGH_OFFICIAL_GEOJSON.exists():
        return []
    official_geojson = read_json(DEFAULT_WALKTHROUGH_OFFICIAL_GEOJSON)
    connector_geojson = (
        read_json(DEFAULT_WALKTHROUGH_CONNECTOR_GEOJSON)
        if DEFAULT_WALKTHROUGH_CONNECTOR_GEOJSON.exists()
        else {"features": []}
    )
    graph_edges, _official_index = load_graph_edges(official_geojson, connector_geojson)
    return graph_edges


def segment_direction_evidence_for_route(route: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for cue in route.get("route_cues") or []:
        for segment in cue.get("segments") or []:
            segment_id = str(segment.get("seg_id") or "")
            if not segment_id:
                continue
            direction_rule = str(segment.get("direction_rule") or "")
            direction_cue = str(segment.get("direction_cue") or "")
            allowed_geometry_direction = None
            if "opposite official geometry" in direction_cue.lower():
                allowed_geometry_direction = "reverse"
            elif direction_rule.lower() in {"ascent", "oneway", "forward"}:
                allowed_geometry_direction = "forward"
            if direction_rule or direction_cue or allowed_geometry_direction:
                evidence[segment_id] = {
                    key: value
                    for key, value in {
                        "direction_rule": direction_rule,
                        "direction_cue": direction_cue,
                        "allowed_geometry_direction": allowed_geometry_direction,
                    }.items()
                    if value
                }
    return evidence


def export_field_packet(
    map_data: dict[str, Any],
    output_dir: Path,
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
    max_parking_gap_miles: float = DEFAULT_MAX_PARKING_GAP_MILES,
    certificate_data: dict[str, Any] | None = None,
    progress_data: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
    trailhead_access_index: dict[str, dict[str, Any]] | None = None,
    walkthrough_graph_edges: list[Any] | None = None,
) -> dict[str, Any]:
    map_data = apply_progress_to_map_data(map_data, progress_data)
    computed_source_metadata = source_metadata_for_map_data(map_data)
    effective_source_metadata = {**computed_source_metadata, **(source_metadata or {})}
    effective_source_metadata["map_data_sha256"] = computed_source_metadata["map_data_sha256"]
    effective_source_metadata["completed_segment_count_at_export"] = computed_source_metadata[
        "completed_segment_count_at_export"
    ]
    effective_source_metadata["blocked_segment_count_at_export"] = computed_source_metadata[
        "blocked_segment_count_at_export"
    ]
    validate_field_menu_source(map_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_manifest in output_dir.glob("*-artifact-manifest.json"):
        stale_manifest.unlink()
    gpx_dir = output_dir / "gpx"
    gpx_dir.mkdir(parents=True, exist_ok=True)
    for stale_gpx in gpx_dir.rglob("*.gpx"):
        stale_gpx.unlink()
    for stale_zip in gpx_dir.glob("*.zip"):
        stale_zip.unlink()
    navigation_gpx_dir = gpx_dir / "navigation"
    cue_gpx_dir = gpx_dir / "cues"
    audit_gpx_dir = gpx_dir / "audit"
    for directory in (navigation_gpx_dir, cue_gpx_dir, audit_gpx_dir):
        directory.mkdir(parents=True, exist_ok=True)
    routes_by_candidate = indexed_features(map_data, "routes", "candidate_id")
    parking_by_candidate = indexed_features(map_data, "parking", "candidate_id")
    if trailhead_access_index is None:
        trailhead_access_index = {}
    segments_by_id = official_segment_index(map_data)
    route_cues = map_data.get("route_cues") or {}
    if walkthrough_graph_edges is None:
        walkthrough_graph_edges = load_default_walkthrough_graph_edges()
    connector_graph = load_default_connector_graph()
    outings = build_outing_menu(map_data)
    runnable = [outing for outing in outings if not outing.get("manual_design_hold") and outing.get("remaining_segment_ids")]
    manual_holds = [outing for outing in outings if outing.get("manual_design_hold") and outing.get("remaining_segment_ids")]
    routes = []
    for outing in sorted(
        runnable,
        key=lambda item: (
            outing_time_bucket_sort(item["time_bucket"]),
            int(item.get("total_minutes") or 0),
            str(item.get("label") or ""),
        ),
    ):
        raw_track_segments = track_segments_for_outing(outing, routes_by_candidate)
        stitched_track_segments = stitch_inter_segment_track_gaps(
            raw_track_segments,
            connector_graph,
            max_gap_miles=max_gap_miles,
        )
        source_gap_repair = source_gap_repair_summary(
            raw_track_segments,
            stitched_track_segments,
            max_gap_miles=max_gap_miles,
        )
        track_segments = densify_track_segments(
            stitched_track_segments,
            max_gap_miles=max_gap_miles,
        )
        parking = parking_for_outing(outing, route_cues, parking_by_candidate, trailhead_access_index)
        validation = validate_outing_export(
            outing,
            track_segments,
            parking,
            route_cues,
            max_gap_miles=max_gap_miles,
            max_parking_gap_miles=max_parking_gap_miles,
        )
        title = f"{outing['label']} {outing['trailhead']}"
        slug = slugify(f"{outing['label']}-{outing['trailhead']}-{'-'.join(outing.get('trails') or [])}")
        description = gpx_description(outing)
        cue_list = [route_cues[str(candidate_id)] for candidate_id in outing.get("candidate_ids") or [] if str(candidate_id) in route_cues]
        logistics = aggregate_logistics(cue_list, parking)
        navigation_waypoints = build_navigation_waypoints(
            outing,
            route_cues,
            segments_by_id,
            parking,
            track_segments,
            logistics,
        )
        cue_only_waypoints = []
        if parking:
            cue_only_waypoints.append(
                {
                    "name": f"PARK/START {parking['name']}",
                    "description": "Park here and start this outing.",
                    "lon": parking["lon"],
                    "lat": parking["lat"],
                }
            )
        cue_only_waypoints.extend(logistics_waypoints(logistics))
        cue_only_waypoints.extend(cue_waypoints(outing, route_cues, segments_by_id))
        final = last_point(track_segments)
        if final:
            cue_only_waypoints.append(
                {
                    "name": "RETURN TO CAR",
                    "description": "Route endpoint / return-to-car point.",
                    "lon": final[0],
                    "lat": final[1],
                }
            )
        audit_waypoints = build_audit_waypoints(
            outing,
            route_cues,
            segments_by_id,
            parking,
            track_segments,
            logistics,
        )
        gpx_path = navigation_gpx_dir / f"{slug}.gpx"
        cue_gpx_path = cue_gpx_dir / f"{slug}.gpx"
        audit_gpx_path = audit_gpx_dir / f"{slug}.gpx"
        gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(title, description, track_segments, navigation_waypoints)),
            encoding="utf-8",
        )
        cue_gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(f"{title} cues", description, [], cue_only_waypoints)),
            encoding="utf-8",
        )
        audit_gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(f"{title} audit", description, track_segments, audit_waypoints)),
            encoding="utf-8",
        )
        route = {
            "outing_id": outing["outing_id"],
            "label": outing["label"],
            "outing": outing,
            "parking": parking,
            "logistics": logistics,
            "parking_navigation_url": parking_navigation_url(parking),
            "gpx_path": str(gpx_path),
            "gpx_href": f"gpx/navigation/{gpx_path.name}",
            "cue_gpx_path": str(cue_gpx_path),
            "cue_gpx_href": f"gpx/cues/{cue_gpx_path.name}",
            "audit_gpx_path": str(audit_gpx_path),
            "audit_gpx_href": f"gpx/audit/{audit_gpx_path.name}",
            "validation": validation,
            "route_cues": cue_list,
            "waypoint_count": len(navigation_waypoints),
            "navigation_waypoint_count": len(navigation_waypoints),
            "cue_waypoint_count": len(cue_only_waypoints),
            "audit_waypoint_count": len(audit_waypoints),
            "track_segment_count": len(track_segments),
            "source_gap_repair": source_gap_repair,
            "_track_segments": track_segments,
            "_official_segment_index": segments_by_id,
        }
        route["navigation_quality"] = navigation_quality_for_route(route)
        route["turn_by_turn_steps"] = build_turn_by_turn_steps(route)
        route["wayfinding_cues"] = build_wayfinding_cues(route)
        route["segment_direction_evidence"] = segment_direction_evidence_for_route(route)
        enrich_route_with_walkthrough_edge_names(route, track_segments, walkthrough_graph_edges)
        routes.append(route)
    safety_by_outing = completion_safety_by_outing(routes)
    for route in routes:
        route["completion_safety"] = safety_by_outing.get(str(route.get("outing_id")), {})
    zip_path = write_gpx_zip(gpx_dir, routes)
    zip_href = f"gpx/{zip_path.name}"
    manifest = {
        "summary": {
            "runnable_outing_count": len(runnable),
            "manual_hold_count": len(manual_holds),
            "gpx_count": len(routes) * len(GPX_PATH_KEYS),
            "navigation_gpx_count": len(routes),
            "cue_gpx_count": len(routes),
            "audit_gpx_count": len(routes),
            "gpx_zip_href": zip_href,
            "gpx_validation_passed": all(route["validation"]["passed"] for route in routes),
            "failed_gpx_count": len([route for route in routes if not route["validation"]["passed"]]),
            "max_gap_miles": max_gap_miles,
            "max_parking_gap_miles": max_parking_gap_miles,
        },
        "routes": routes,
        "manual_holds": manual_holds,
    }
    if certificate_data is None:
        certificate_data = load_default_certificate_data()
    field_tool_data = build_field_tool_data(
        manifest,
        certificate_data=certificate_data,
        map_data=map_data,
        source_metadata=effective_source_metadata,
    )
    manifest["certified_baseline"] = field_tool_data["certified_baseline"]
    manifest["summary"]["field_tool_data_href"] = FIELD_TOOL_DATA_NAME
    manifest["summary"]["map_data_sha256"] = field_tool_data["source"]["map_data_sha256"]
    (output_dir / "index.html").write_text(strip_trailing_whitespace(render_index(manifest)), encoding="utf-8")
    (output_dir / FIELD_TOOL_DATA_NAME).write_text(json.dumps(field_tool_data, indent=2) + "\n", encoding="utf-8")
    live_map_path = output_dir / "live-map.html"
    live_map_path.write_text(strip_trailing_whitespace(render_live_map_html()), encoding="utf-8")
    pwa_paths = write_pwa_assets(
        output_dir,
        routes,
        zip_href,
        extra_precache_urls=[FIELD_TOOL_DATA_NAME, "live-map.html"],
    )
    public_manifest = {
        **manifest,
        "routes": [
            {
                **{
                    key: value
                    for key, value in route.items()
                    if key not in {"route_cues", "_track_segments", "_official_segment_index", *GPX_PATH_KEYS}
                },
                "gpx_path": route["gpx_href"],
                "cue_gpx_path": route["cue_gpx_href"],
                "audit_gpx_path": route["audit_gpx_href"],
            }
            for route in routes
        ],
    }
    public_manifest["summary"]["pwa_artifacts"] = [
        "manifest.webmanifest",
        "service-worker.js",
        "icons/icon-192.png",
        "icons/icon-512.png",
        FIELD_TOOL_DATA_NAME,
        "live-map.html",
    ]
    (output_dir / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    safety_failures = public_safety_check(output_dir)
    if safety_failures:
        raise ValueError("Public safety check failed:\n" + "\n".join(safety_failures))
    manifest["pwa_paths"] = [str(live_map_path), *[str(path) for path in pwa_paths]]
    manifest["gpx_zip_path"] = str(zip_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-html", type=Path, default=DEFAULT_MAP_HTML)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-gap-miles", type=float, default=DEFAULT_MAX_GAP_MILES)
    parser.add_argument("--max-parking-gap-miles", type=float, default=DEFAULT_MAX_PARKING_GAP_MILES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_data, source_path = load_map_data(args.map_html, args.map_data_json)
    manifest = export_field_packet(
        map_data,
        args.output_dir,
        max_gap_miles=args.max_gap_miles,
        max_parking_gap_miles=args.max_parking_gap_miles,
        progress_data=read_json(args.progress_json) if args.progress_json else None,
        source_metadata=source_metadata_for_map_data(map_data, source_path),
        trailhead_access_index=load_trailhead_access_index(),
    )
    artifact_manifest_dir = YEAR_DIR / "outputs" / "private" / "field-packet"
    artifact_manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path = artifact_manifest_dir / f"{DEFAULT_BASENAME}-artifact-manifest.json"
    outputs = [
        args.output_dir / "index.html",
        args.output_dir / "manifest.json",
        args.output_dir / FIELD_TOOL_DATA_NAME,
        args.output_dir / "manifest.webmanifest",
        args.output_dir / "service-worker.js",
        args.output_dir / "icons" / "icon-192.png",
        args.output_dir / "icons" / "icon-512.png",
        args.output_dir / "gpx" / GPX_ZIP_NAME,
    ]
    outputs.extend(Path(route["gpx_path"]) for route in manifest["routes"])
    inputs = [source_path]
    if args.progress_json:
        inputs.append(args.progress_json)
    write_manifest(
        artifact_manifest_path,
        build_artifact_manifest(
            run_id=str(map_data.get("run_id") or DEFAULT_BASENAME),
            inputs=inputs,
            outputs=outputs,
            command="export_mobile_field_packet.py",
            metadata={
                "runnable_outing_count": manifest["summary"]["runnable_outing_count"],
                "gpx_validation_passed": manifest["summary"]["gpx_validation_passed"],
            },
        ),
    )
    print(f"Wrote {manifest['summary']['gpx_count']} GPX files to {args.output_dir / 'gpx'}")
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {args.output_dir / 'manifest.json'}")
    print(f"Wrote {artifact_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
