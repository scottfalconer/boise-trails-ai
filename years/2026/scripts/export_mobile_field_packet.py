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
from personal_route_planner import read_json  # noqa: E402


DEFAULT_MAP_DATA_JSON = (
    YEAR_DIR
    / "outputs"
    / "private"
    / "route-blocks"
    / "block-hybrid-day-package-pass-v1-map-data.json"
)
DEFAULT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_BASENAME = "phone-field-packet"
DEFAULT_MAX_GAP_MILES = 0.05
DEFAULT_MAX_PARKING_GAP_MILES = 0.35
GPX_ZIP_NAME = "all-field-packet-gpx.zip"
COMPLETED_STORAGE_KEY = "fieldPacketCompletedOutings"
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
    ("1B", "Harrison Hollow"): [
        "Early junction: do not keep climbing #57 Harrison Hollow. Turn toward #52 Kemper's Ridge / #51 Who Now first.",
        "After Kemper's Ridge, take #50 Hippie Shake. Do not drop onto #51 Who Now unless the GPX says you are completing that segment.",
    ],
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:90] or "outing"


def extract_map_data_from_html(html: str) -> dict[str, Any]:
    match = re.search(r"const DATA = (.*?);\nconst map =", html, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find embedded DATA payload in outing map HTML.")
    return json.loads(match.group(1))


def load_map_data(map_html: Path | None, map_data_json: Path | None) -> tuple[dict[str, Any], Path]:
    if map_html and map_html.exists():
        return extract_map_data_from_html(map_html.read_text(encoding="utf-8")), map_html
    if map_data_json and map_data_json.exists():
        return read_json(map_data_json), map_data_json
    raise FileNotFoundError("No map HTML or map-data JSON source exists for the mobile field packet.")


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


def parking_for_outing(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    parking_by_candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for candidate_id in outing.get("candidate_ids") or []:
        cue = route_cues.get(str(candidate_id)) or {}
        trailhead = cue.get("trailhead") or {}
        if trailhead.get("lat") is not None and trailhead.get("lon") is not None:
            return {
                "name": trailhead.get("name") or outing.get("trailhead") or "Parking/start",
                "lon": float(trailhead["lon"]),
                "lat": float(trailhead["lat"]),
                "has_parking": trailhead.get("has_parking"),
                "has_restroom": trailhead.get("has_restroom"),
                "has_water": trailhead.get("has_water"),
                "water_confidence": trailhead.get("water_confidence"),
            }
        for feature in parking_by_candidate.get(str(candidate_id), []):
            point = feature_point(feature)
            if point:
                props = feature.get("properties") or {}
                return {
                    "name": props.get("name") or props.get("trailhead") or outing.get("trailhead") or "Parking/start",
                    "lon": point[0],
                    "lat": point[1],
                    "has_parking": props.get("has_parking"),
                    "has_restroom": props.get("has_restroom"),
                    "has_water": props.get("has_water"),
                    "water_confidence": props.get("water_confidence"),
                }
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


def route_signpost_labels(route: dict[str, Any]) -> list[str]:
    values = []
    for cue in route.get("route_cues") or []:
        for segment in cue.get("segments") or []:
            values.append(segment.get("trail_name"))
        for link in cue.get("between_links") or []:
            values.extend(link.get("connector_names") or [])
            values.append(link.get("from_trail"))
            values.append(link.get("to_trail"))
        return_to_car = cue.get("return_to_car") or {}
        values.extend(return_to_car.get("connector_names") or [])
    return signpost_labels(values)


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
        return "No verified water in planner data."
    return "; ".join(
        f"{item.get('name') or 'Water'} · {item.get('location') or 'location'} · {item.get('confidence') or 'verified'}"
        for item in items
    )


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
    if distance is not None:
        pieces.append(f"{format_miles(distance)} mi")
    connector_names = summarized_names(link.get("connector_names") or [])
    if connector_names:
        pieces.append(f"via {connector_names}")
    signpost = signpost_sentence(link.get("connector_names") or [link.get("from_trail"), link.get("to_trail")])
    if signpost:
        pieces.append(signpost)
    mile_parts = []
    if float(link.get("official_repeat_miles") or 0):
        mile_parts.append(f"{format_miles(link.get('official_repeat_miles'))} repeat official")
    if float(link.get("connector_miles") or 0):
        mile_parts.append(f"{format_miles(link.get('connector_miles'))} connector")
    if float(link.get("road_miles") or 0):
        mile_parts.append(f"{format_miles(link.get('road_miles'))} road")
    if mile_parts:
        pieces.append(" / ".join(mile_parts))
    return ". ".join(pieces) + ("." if pieces else "")


def build_turn_by_turn_steps(route: dict[str, Any]) -> list[dict[str, str]]:
    outing = route["outing"]
    parking = route.get("parking") or {}
    cues = route.get("route_cues") or []
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    steps = [
        {
            "kind": "park",
            "title": f"Park/start at {parking.get('name') or outing.get('trailhead') or 'planned parking'}",
            "detail": "Start the GPX before leaving the car. The track should begin and end at this parking point.",
        }
    ]
    if logistics.get("car_passes"):
        steps.append(
            {
                "kind": "car-pass",
                "title": "Pass by car again",
                "detail": car_pass_sentence(logistics.get("car_passes") or []),
            }
        )
    steps.append(
        {
            "kind": "water",
            "title": "Known water",
            "detail": water_sentence(logistics.get("known_water") or []),
        }
    )
    for cue in cues:
        segments = cue.get("segments") or []
        first_segment = segments[0] if segments else {}
        first_trail = first_segment.get("trail_name") or cue.get("title") or "the first official segment"
        access = cue.get("start_access") or {}
        access_bits = []
        if access.get("confidence"):
            access_bits.append(f"snap confidence {access['confidence']}")
        if access.get("mapped_access_miles") is not None:
            access_bits.append(f"mapped access {format_miles(access.get('mapped_access_miles'))} mi")
        if access.get("direct_gap_miles") is not None:
            access_bits.append(f"direct gap {format_miles(access.get('direct_gap_miles'))} mi")
        access_detail = "Follow the GPX line from the car to the first official segment."
        if access_bits:
            access_detail += " " + "; ".join(access_bits) + "."
        first_signpost = signpost_sentence([first_trail], prefix="Signpost")
        if first_signpost:
            access_detail += " " + first_signpost + "."
        steps.append(
            {
                "kind": "access",
                "title": "Get from parking to the route",
                "detail": f"{access_detail} Start with {first_trail}.",
            }
        )
        between_links = list(cue.get("between_links") or [])
        link_index = 0
        for index, segment in enumerate(segments):
            direction = str(segment.get("direction_rule") or "").lower()
            segment_name = segment.get("segment_name") or segment.get("trail_name") or "official segment"
            trail_name = segment.get("trail_name") or "official trail"
            detail = (
                f"{trail_name} · official {format_miles(segment.get('official_miles'))} mi. "
                f"{segment.get('direction_cue') or 'Follow the GPX line.'}"
            )
            effort = segment_effort_sentence(segment)
            if effort:
                detail += f" {effort}."
            signpost = signpost_sentence([trail_name], prefix="Signpost")
            if signpost:
                detail += " " + signpost + "."
            if direction == "ascent":
                detail = "ASCENT REQUIRED. " + detail
            steps.append(
                {
                    "kind": "ascent" if direction == "ascent" else "official",
                    "title": f"Complete {segment_name}",
                    "detail": detail.strip(),
                }
            )
            next_segment = segments[index + 1] if index + 1 < len(segments) else None
            if next_segment and next_segment.get("trail_name") != segment.get("trail_name"):
                link = between_links[link_index] if link_index < len(between_links) else {}
                link_index += 1
                to_trail = link.get("to_trail") or next_segment.get("trail_name") or "next trail"
                from_trail = link.get("from_trail") or segment.get("trail_name") or trail_name
                detail = connector_detail(link) or "Follow the GPX connector line to the next official trail."
                steps.append(
                    {
                        "kind": "connector",
                        "title": f"Connector to {to_trail}",
                        "detail": f"From {from_trail}: {detail}",
                    }
                )
        for link in between_links[link_index:]:
            to_trail = link.get("to_trail") or "next trail"
            from_trail = link.get("from_trail") or "previous trail"
            detail = connector_detail(link) or "Follow the GPX connector line."
            steps.append(
                {
                    "kind": "connector",
                    "title": f"Connector to {to_trail}",
                    "detail": f"From {from_trail}: {detail}",
                }
            )
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            return_names = summarized_names(return_to_car.get("connector_names") or [])
            detail = return_to_car.get("description") or "Follow the GPX line back to the car."
            if return_names:
                detail += f" Return via {return_names}."
            detail += (
                f" Repeat official {format_miles(return_to_car.get('official_repeat_miles'))} mi; "
                f"connector {format_miles(return_to_car.get('connector_miles'))} mi; "
                f"road {format_miles(return_to_car.get('road_miles'))} mi."
            )
            steps.append({"kind": "return", "title": "Return to car", "detail": detail})
    return steps


def render_card(route: dict[str, Any]) -> str:
    outing = route["outing"]
    parking = route.get("parking") or {}
    logistics = route.get("logistics") or {"car_passes": [], "known_water": []}
    nav_url = route.get("parking_navigation_url")
    nav_link = (
        f'<a class="secondary" href="{html_escape(nav_url)}">Navigate to parking</a>'
        if nav_url
        else '<span class="secondary disabled">Parking navigation unavailable</span>'
    )
    cue_gpx_link = f'<a class="secondary" href="{html_escape(route["cue_gpx_href"])}" download>Cue GPX</a>'
    audit_gpx_link = f'<a class="secondary" href="{html_escape(route["audit_gpx_href"])}" download>Audit GPX</a>'
    cues = route.get("route_cues") or []
    segment_rows = []
    for cue in cues:
        for segment in cue.get("segments") or []:
            direction = str(segment.get("direction_rule") or "").lower()
            css = "segment ascent" if direction == "ascent" else "segment"
            label = "ASCENT" if direction == "ascent" else "SEG"
            signpost = signpost_sentence([segment.get("trail_name")], prefix="Signpost")
            segment_detail = segment.get("direction_cue") or "Follow GPX line."
            effort = segment_effort_sentence(segment)
            if effort:
                segment_detail += " " + effort + "."
            if signpost:
                segment_detail += " " + signpost
            segment_rows.append(
                f'<div class="{css}"><b>{label} {html_escape(segment.get("order"))}: '
                f'{html_escape(segment.get("segment_name") or segment.get("trail_name"))}</b>'
                f'<span>{html_escape(segment_detail)}</span></div>'
            )
    return_html = []
    for cue in cues:
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            return_html.append(
                f'<p>{html_escape(return_to_car.get("description") or "Follow the GPX line back to the car.")}'
                f'<br>Repeat official: {html_escape(format_miles(return_to_car.get("official_repeat_miles")))} mi'
                f' · connector: {html_escape(format_miles(return_to_car.get("connector_miles")))} mi'
                f' · road: {html_escape(format_miles(return_to_car.get("road_miles")))} mi</p>'
            )
    warnings = ""
    if not route["validation"]["passed"]:
        warnings = '<p class="warning">GPX validation failed. Do not use this route in the field until reviewed.</p>'
    steps = route.get("turn_by_turn_steps") or build_turn_by_turn_steps(route)
    steps_html = "".join(
        f'<li class="{html_escape(step.get("kind"))}"><b>{html_escape(step.get("title"))}</b>'
        f'<span>{html_escape(step.get("detail"))}</span></li>'
        for step in steps
    )
    signpost_labels_html = ""
    labels = route_signpost_labels(route)
    notes = manual_signpost_notes(route)
    if labels or notes:
        label_html = f'<p><b>Watch for:</b> {html_escape("; ".join(labels))}</p>' if labels else ""
        notes_html = "".join(f"<li>{html_escape(note)}</li>" for note in notes)
        signpost_labels_html = (
            "<section><h3>Signpost cues</h3>"
            f"{label_html}"
            f"{f'<ul class=\"signpost-notes\">{notes_html}</ul>' if notes_html else ''}"
            "</section>"
        )
    return f"""
    <article class="card" id="{html_escape(outing['outing_id'])}" data-outing-id="{html_escape(outing['outing_id'])}" data-minutes="{int(outing.get('total_minutes') or 0)}">
      <div class="card-head">
        <span>Phone run card</span>
        <h2>{html_escape(outing['label'])}. {html_escape(outing['trailhead'])}</h2>
      </div>
      <div class="stats">
        <div><b>Door to door p75</b><strong>{html_escape(format_minutes(outing.get('total_minutes')))}</strong></div>
        <div><b>On foot</b><strong>{html_escape(format_miles(outing.get('on_foot_miles')))} mi</strong></div>
        <div><b>Official</b><strong>{html_escape(format_miles(outing.get('official_miles')))} mi</strong></div>
        <div><b>Segments</b><strong>{html_escape(outing.get('remaining_segment_count'))} / {len(outing.get('segment_ids') or [])}</strong></div>
      </div>
      <div class="actions">
        <a href="{html_escape(route['gpx_href'])}" download>Open Nav GPX</a>
        {nav_link}
        {cue_gpx_link}
        {audit_gpx_link}
        <button type="button" class="done-button" data-complete-action="mark">Mark done</button>
        <button type="button" class="undo-button" data-complete-action="undo">Undo done</button>
      </div>
      {warnings}
      <section><h3>PARK/START</h3><p>{html_escape(parking.get('name') or outing.get('trailhead'))}</p></section>
      <section><h3>Water / car access</h3><p><b>Car:</b> {html_escape(car_pass_sentence(logistics.get('car_passes') or []))}<br><b>Known water:</b> {html_escape(water_sentence(logistics.get('known_water') or []))}</p></section>
      <section><h3>Trails</h3><p>{html_escape(', '.join(outing.get('trails') or []))}</p></section>
      {signpost_labels_html}
      <section><h3>Turn-by-turn from car</h3><ol class="steps">{steps_html}</ol></section>
      <section><h3>Official segment order</h3>{''.join(segment_rows) or '<p>Follow the GPX line.</p>'}</section>
      <section><h3>Return to car</h3>{''.join(return_html) or '<p>Follow the GPX line back to parking.</p>'}</section>
      <section><h3>Before leaving</h3><p>Check current Ridge to Rivers conditions/signage. The GPX is for navigation, not day-of closure clearance.</p></section>
    </article>
    """


def render_index(manifest: dict[str, Any]) -> str:
    cards = "\n".join(render_card(route) for route in manifest["routes"])
    manual_count = manifest["summary"]["manual_hold_count"]
    zip_href = manifest["summary"].get("gpx_zip_href") or f"gpx/{GPX_ZIP_NAME}"
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
    .filters {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:6px; margin-top:10px; }}
    button {{ min-height:34px; border:1px solid #d7ddd4; border-radius:6px; background:#fff; font-weight:700; }}
    button.active {{ background:#111827; color:#fff; border-color:#111827; }}
    .utility-actions {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; }}
    .utility-actions a,.utility-actions button {{ display:flex; align-items:center; justify-content:center; min-height:38px; padding:0 10px; border-radius:6px; border:1px solid #d7ddd4; background:#fff; color:#111827; font-weight:800; text-decoration:none; }}
    .quick-list {{ margin:10px 0 0; padding:8px; border:1px solid #d7ddd4; border-radius:8px; background:#fff; }}
    .quick-list h2 {{ margin:0 0 4px; color:#111827; font-size:14px; }}
    main {{ padding:10px; display:grid; gap:10px; }}
    .card {{ overflow:hidden; border:1px solid #d7ddd4; border-radius:8px; background:#fff; box-shadow:0 1px 4px rgba(15,23,42,.08); }}
    .card.completed {{ opacity:.48; }}
    body.hide-completed .card.completed {{ display:none !important; }}
    .card-head {{ padding:12px; background:#111827; color:#fff; }}
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
    .steps b {{ display:block; font-size:13px; }}
    .steps span {{ display:block; margin-top:2px; color:#475467; font-size:12px; line-height:1.35; }}
    .signpost-notes {{ margin:6px 0 0; padding-left:18px; color:#475467; font-size:12px; line-height:1.35; }}
    .warning {{ margin:10px 12px; padding:8px; border-left:4px solid #b45309; background:#fff7ed; color:#7c2d12; }}
    body.screenshot header,.screenshot .utility-actions,.screenshot .filters,.screenshot .actions {{ display:none !important; }}
    body.screenshot main {{ padding:0; }}
    body.screenshot .card {{ display:none !important; border:0; border-radius:0; box-shadow:none; }}
    body.screenshot .card:not(.completed):first-of-type {{ display:block !important; }}
    @media (min-width:760px) {{ main {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Phone Field Packet</h1>
    <p>Open one outing, send the Nav GPX to your navigation app, then use the card for parking, turn-by-turn cues, and return-to-car notes. Cue GPX is marker-only; Audit GPX keeps dense segment-credit markers out of the field view.</p>
    <div class="top-grid">
      <div class="status-panel"><b>Offline-ready</b> after the first full load. In Safari, Share &rarr; Add to Home Screen for the app-style launcher. <span id="offline-status">Checking offline cache...</span></div>
      <div class="utility-actions">
        <a href="{html_escape(zip_href)}" download>Download all GPX</a>
        <button type="button" id="completed-toggle">Hide completed</button>
        <button type="button" id="screenshot-toggle">Screenshot mode</button>
        <button type="button" id="reset-completed">Reset progress</button>
      </div>
    </div>
    <div class="quick-list"><h2>Today&apos;s best options</h2><p>Use the time buttons to pick what fits the door-to-door window. Mark completed outings so they disappear from the active field list.</p></div>
    {manual_note}
    <div class="filters">
      <button type="button" class="active" data-filter="all">All</button>
      <button type="button" data-filter="90">&le;90m</button>
      <button type="button" data-filter="120">&le;2h</button>
      <button type="button" data-filter="180">&le;3h</button>
      <button type="button" data-filter="240">&le;4h</button>
    </div>
  </header>
  <main>{cards}</main>
  <script>
    const STORAGE_KEY = "{COMPLETED_STORAGE_KEY}";
    const buttons = [...document.querySelectorAll("button[data-filter]")];
    const cards = [...document.querySelectorAll(".card")];
    const completedToggle = document.getElementById("completed-toggle");
    const screenshotToggle = document.getElementById("screenshot-toggle");
    const resetCompleted = document.getElementById("reset-completed");
    const offlineStatus = document.getElementById("offline-status");

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

    function syncCompletedState() {{
      const completed = completedSet();
      cards.forEach(card => card.classList.toggle("completed", completed.has(card.dataset.outingId)));
      applyFilter();
    }}

    function activeFilter() {{
      return document.querySelector("button[data-filter].active")?.dataset.filter || "all";
    }}

    function applyFilter() {{
      const filter = activeFilter();
      cards.forEach(card => {{
        const minutes = Number(card.dataset.minutes || 0);
        card.style.display = filter === "all" || minutes <= Number(filter) ? "" : "none";
      }});
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
  </script>
</body>
</html>
"""


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


def write_pwa_assets(output_dir: Path, routes: list[dict[str, Any]], zip_href: str) -> list[Path]:
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
    for route in routes:
        precache_urls.extend(route[key] for key in GPX_HREF_KEYS)
    service_worker_path = output_dir / "service-worker.js"
    digest = hashlib.sha256()
    for route in routes:
        for key in GPX_PATH_KEYS:
            digest.update(Path(route[key]).read_bytes())
    digest.update((output_dir / zip_href).read_bytes())
    cache_name = f"boise-trails-field-packet-v{len(routes)}-{digest.hexdigest()[:12]}"
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


def export_field_packet(
    map_data: dict[str, Any],
    output_dir: Path,
    max_gap_miles: float = DEFAULT_MAX_GAP_MILES,
    max_parking_gap_miles: float = DEFAULT_MAX_PARKING_GAP_MILES,
) -> dict[str, Any]:
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
    segments_by_id = official_segment_index(map_data)
    route_cues = map_data.get("route_cues") or {}
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
        track_segments = densify_track_segments(
            track_segments_for_outing(outing, routes_by_candidate),
            max_gap_miles=max_gap_miles,
        )
        parking = parking_for_outing(outing, route_cues, parking_by_candidate)
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
        }
        route["turn_by_turn_steps"] = build_turn_by_turn_steps(route)
        routes.append(route)
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
    (output_dir / "index.html").write_text(strip_trailing_whitespace(render_index(manifest)), encoding="utf-8")
    pwa_paths = write_pwa_assets(output_dir, routes, zip_href)
    public_manifest = {
        **manifest,
        "routes": [
            {
                **{key: value for key, value in route.items() if key not in {"route_cues", *GPX_PATH_KEYS}},
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
    ]
    (output_dir / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    safety_failures = public_safety_check(output_dir)
    if safety_failures:
        raise ValueError("Public safety check failed:\n" + "\n".join(safety_failures))
    manifest["pwa_paths"] = [str(path) for path in pwa_paths]
    manifest["gpx_zip_path"] = str(zip_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-html", type=Path, default=DEFAULT_MAP_HTML)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
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
    )
    artifact_manifest_dir = YEAR_DIR / "outputs" / "private" / "field-packet"
    artifact_manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path = artifact_manifest_dir / f"{DEFAULT_BASENAME}-artifact-manifest.json"
    outputs = [
        args.output_dir / "index.html",
        args.output_dir / "manifest.json",
        args.output_dir / "manifest.webmanifest",
        args.output_dir / "service-worker.js",
        args.output_dir / "icons" / "icon-192.png",
        args.output_dir / "icons" / "icon-512.png",
        args.output_dir / "gpx" / GPX_ZIP_NAME,
    ]
    outputs.extend(Path(route["gpx_path"]) for route in manifest["routes"])
    write_manifest(
        artifact_manifest_path,
        build_artifact_manifest(
            run_id=str(map_data.get("run_id") or DEFAULT_BASENAME),
            inputs=[source_path],
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
