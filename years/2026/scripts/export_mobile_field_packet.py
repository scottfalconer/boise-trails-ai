#!/usr/bin/env python3
"""Export a phone-first field packet from the current outing menu map data."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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
            waypoints.append({"name": name, "description": description.strip(), "lon": point[0], "lat": point[1]})
    return waypoints


def build_waypoints(
    outing: dict[str, Any],
    route_cues: dict[str, Any],
    official_index: dict[str, dict[str, Any]],
    parking: dict[str, Any] | None,
    track_segments: list[list[tuple[float, float]]],
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
        f"Door-to-door {format_minutes(outing.get('total_minutes'))}; "
        f"Segments left {outing.get('remaining_segment_count')} / {len(outing.get('segment_ids') or [])}. "
        "Check current Ridge to Rivers signage and conditions before leaving."
    )


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
    lines.extend(["  <trk>", f"    <name>{escape(name)}</name>", f"    <desc>{escape(description)}</desc>"])
    for coords in track_segments:
        lines.append("    <trkseg>")
        for lon, lat in coords:
            lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}" />')
        lines.append("    </trkseg>")
    lines.extend(["  </trk>", "</gpx>", ""])
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


def connector_detail(link: dict[str, Any]) -> str:
    pieces = []
    distance = link.get("distance_miles")
    if distance is not None:
        pieces.append(f"{format_miles(distance)} mi")
    connector_names = summarized_names(link.get("connector_names") or [])
    if connector_names:
        pieces.append(f"via {connector_names}")
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
    steps = [
        {
            "kind": "park",
            "title": f"Park/start at {parking.get('name') or outing.get('trailhead') or 'planned parking'}",
            "detail": "Start the GPX before leaving the car. The track should begin and end at this parking point.",
        }
    ]
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
    nav_url = route.get("parking_navigation_url")
    nav_link = (
        f'<a class="secondary" href="{html_escape(nav_url)}">Navigate to parking</a>'
        if nav_url
        else '<span class="secondary disabled">Parking navigation unavailable</span>'
    )
    cues = route.get("route_cues") or []
    segment_rows = []
    for cue in cues:
        for segment in cue.get("segments") or []:
            direction = str(segment.get("direction_rule") or "").lower()
            css = "segment ascent" if direction == "ascent" else "segment"
            label = "ASCENT" if direction == "ascent" else "SEG"
            segment_rows.append(
                f'<div class="{css}"><b>{label} {html_escape(segment.get("order"))}: '
                f'{html_escape(segment.get("segment_name") or segment.get("trail_name"))}</b>'
                f'<span>{html_escape(segment.get("direction_cue") or "Follow GPX line.")}</span></div>'
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
    return f"""
    <article class="card" id="{html_escape(outing['outing_id'])}" data-minutes="{int(outing.get('total_minutes') or 0)}">
      <div class="card-head">
        <span>Phone run card</span>
        <h2>{html_escape(outing['label'])}. {html_escape(outing['trailhead'])}</h2>
      </div>
      <div class="stats">
        <div><b>Door to door</b><strong>{html_escape(format_minutes(outing.get('total_minutes')))}</strong></div>
        <div><b>On foot</b><strong>{html_escape(format_miles(outing.get('on_foot_miles')))} mi</strong></div>
        <div><b>Official</b><strong>{html_escape(format_miles(outing.get('official_miles')))} mi</strong></div>
        <div><b>Segments</b><strong>{html_escape(outing.get('remaining_segment_count'))} / {len(outing.get('segment_ids') or [])}</strong></div>
      </div>
      <div class="actions">
        <a href="{html_escape(route['gpx_href'])}" download>Open GPX</a>
        {nav_link}
      </div>
      {warnings}
      <section><h3>PARK/START</h3><p>{html_escape(parking.get('name') or outing.get('trailhead'))}</p></section>
      <section><h3>Trails</h3><p>{html_escape(', '.join(outing.get('trails') or []))}</p></section>
      <section><h3>Turn-by-turn from car</h3><ol class="steps">{steps_html}</ol></section>
      <section><h3>Official segment order</h3>{''.join(segment_rows) or '<p>Follow the GPX line.</p>'}</section>
      <section><h3>Return to car</h3>{''.join(return_html) or '<p>Follow the GPX line back to parking.</p>'}</section>
      <section><h3>Before leaving</h3><p>Check current Ridge to Rivers conditions/signage. The GPX is for navigation, not day-of closure clearance.</p></section>
    </article>
    """


def render_index(manifest: dict[str, Any]) -> str:
    cards = "\n".join(render_card(route) for route in manifest["routes"])
    manual_count = manifest["summary"]["manual_hold_count"]
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
  <title>Phone Field Packet</title>
  <style>
    body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f7f2; color:#111827; }}
    header {{ position:sticky; top:0; z-index:2; padding:14px 14px 10px; background:rgba(255,255,255,.97); border-bottom:1px solid #d7ddd4; }}
    h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:0; }}
    p {{ margin:4px 0; color:#475467; line-height:1.4; }}
    .filters {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; margin-top:10px; }}
    button {{ min-height:34px; border:1px solid #d7ddd4; border-radius:6px; background:#fff; font-weight:700; }}
    button.active {{ background:#111827; color:#fff; border-color:#111827; }}
    main {{ padding:10px; display:grid; gap:10px; }}
    .card {{ overflow:hidden; border:1px solid #d7ddd4; border-radius:8px; background:#fff; box-shadow:0 1px 4px rgba(15,23,42,.08); }}
    .card-head {{ padding:12px; background:#111827; color:#fff; }}
    .card-head span {{ display:block; color:#cbd5e1; font-size:12px; text-transform:uppercase; font-weight:800; }}
    h2 {{ margin:3px 0 0; font-size:19px; line-height:1.15; letter-spacing:0; }}
    .stats {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; padding:10px 12px; }}
    .stats div {{ border:1px solid #e5e7eb; border-radius:6px; padding:7px; background:#f9fafb; }}
    .stats b {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .stats strong {{ display:block; margin-top:2px; font-size:15px; }}
    .actions {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 12px 12px; }}
    .actions a,.actions span {{ display:flex; align-items:center; justify-content:center; min-height:40px; border-radius:6px; font-weight:800; text-decoration:none; }}
    .actions a:first-child {{ background:#2563eb; color:#fff; }}
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
    .steps b {{ display:block; font-size:13px; }}
    .steps span {{ display:block; margin-top:2px; color:#475467; font-size:12px; line-height:1.35; }}
    .warning {{ margin:10px 12px; padding:8px; border-left:4px solid #b45309; background:#fff7ed; color:#7c2d12; }}
    @media (min-width:760px) {{ main {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Phone Field Packet</h1>
    <p>Open one outing, send the GPX to your navigation app, then use the card for parking, segment order, and return-to-car notes.</p>
    {manual_note}
    <div class="filters">
      <button type="button" class="active" data-filter="all">All</button>
      <button type="button" data-filter="120">&le;2h</button>
      <button type="button" data-filter="180">&le;3h</button>
      <button type="button" data-filter="240">&le;4h</button>
    </div>
  </header>
  <main>{cards}</main>
  <script>
    const buttons = [...document.querySelectorAll("button[data-filter]")];
    const cards = [...document.querySelectorAll(".card")];
    buttons.forEach(button => button.addEventListener("click", () => {{
      buttons.forEach(item => item.classList.toggle("active", item === button));
      const filter = button.dataset.filter;
      cards.forEach(card => {{
        const minutes = Number(card.dataset.minutes || 0);
        card.style.display = filter === "all" || minutes <= Number(filter) ? "" : "none";
      }});
    }}));
  </script>
</body>
</html>
"""


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


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
    for stale_gpx in gpx_dir.glob("*.gpx"):
        stale_gpx.unlink()
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
        waypoints = build_waypoints(outing, route_cues, segments_by_id, parking, track_segments)
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
        gpx_path = gpx_dir / f"{slug}.gpx"
        description = gpx_description(outing)
        gpx_path.write_text(
            strip_trailing_whitespace(render_gpx(title, description, track_segments, waypoints)),
            encoding="utf-8",
        )
        cue_list = [route_cues[str(candidate_id)] for candidate_id in outing.get("candidate_ids") or [] if str(candidate_id) in route_cues]
        route = {
            "outing_id": outing["outing_id"],
            "label": outing["label"],
            "outing": outing,
            "parking": parking,
            "parking_navigation_url": parking_navigation_url(parking),
            "gpx_path": str(gpx_path),
            "gpx_href": f"gpx/{gpx_path.name}",
            "validation": validation,
            "route_cues": cue_list,
            "waypoint_count": len(waypoints),
            "track_segment_count": len(track_segments),
        }
        route["turn_by_turn_steps"] = build_turn_by_turn_steps(route)
        routes.append(
            route
        )
    manifest = {
        "summary": {
            "runnable_outing_count": len(runnable),
            "manual_hold_count": len(manual_holds),
            "gpx_count": len(routes),
            "gpx_validation_passed": all(route["validation"]["passed"] for route in routes),
            "failed_gpx_count": len([route for route in routes if not route["validation"]["passed"]]),
            "max_gap_miles": max_gap_miles,
            "max_parking_gap_miles": max_parking_gap_miles,
        },
        "routes": routes,
        "manual_holds": manual_holds,
    }
    (output_dir / "index.html").write_text(strip_trailing_whitespace(render_index(manifest)), encoding="utf-8")
    public_manifest = {
        **manifest,
        "routes": [
            {
                **{key: value for key, value in route.items() if key not in {"route_cues", "gpx_path"}},
                "gpx_path": route["gpx_href"],
            }
            for route in routes
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    safety_failures = public_safety_check(output_dir)
    if safety_failures:
        raise ValueError("Public safety check failed:\n" + "\n".join(safety_failures))
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
    outputs = [args.output_dir / "index.html", args.output_dir / "manifest.json"]
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
