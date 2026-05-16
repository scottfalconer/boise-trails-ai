#!/usr/bin/env python3
"""Generate runner-perspective frame-shift audits for current field routes."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
FIELD_TOOL_DATA = REPO_ROOT / "docs/field-packet/field-tool-data.json"
R2R_GEOJSON = REPO_ROOT / "years/2026/inputs/open-data/r2r-trails-2026-05-04/boise_parks_trails_open_data.geojson"
CONNECTORS_GEOJSON = REPO_ROOT / "years/2026/inputs/open-data/routing-connectors-2026-05-04/combined_r2r_osm_connectors.geojson"
OFFICIAL_SEGMENTS = REPO_ROOT / "years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson"
OUTPUT_DIR = REPO_ROOT / "years/2026/checkpoints/runner-perspective-frame-shift-2026-05-10"
FRAME_LOG = Path.home() / ".codex/debug/frame-shift/frames.jsonl"

Point = tuple[float, float]  # lat, lon


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "route"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def feature_name(props: dict[str, Any]) -> str:
    return clean_text(props.get("TrailName") or props.get("Name") or props.get("highway") or "unnamed feature")


def iter_lines(geometry: dict[str, Any]) -> list[list[Point]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if geom_type == "LineString":
        return [[(float(lat), float(lon)) for lon, lat, *_ in coords]]
    if geom_type == "MultiLineString":
        return [[(float(lat), float(lon)) for lon, lat, *_ in line] for line in coords]
    return []


def load_features(paths: list[Path]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    seen = set()
    for path in paths:
        payload = load_json(path)
        for feature in payload.get("features") or []:
            props = feature.get("properties") or {}
            name = feature_name(props)
            if not name:
                continue
            for line in iter_lines(feature.get("geometry") or {}):
                if len(line) < 2:
                    continue
                key = (name, tuple((round(lat, 6), round(lon, 6)) for lat, lon in line[:3]))
                if key in seen:
                    continue
                seen.add(key)
                lats = [point[0] for point in line]
                lons = [point[1] for point in line]
                features.append(
                    {
                        "name": name,
                        "props": props,
                        "line": line,
                        "bbox": (min(lats), min(lons), max(lats), max(lons)),
                        "source": clean_text(props.get("source") or "ridge_to_rivers_open_data"),
                    }
                )
    return features


def parse_gpx(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    waypoints = []
    for wpt in root.findall("g:wpt", ns):
        name = wpt.findtext("g:name", default="", namespaces=ns)
        desc = wpt.findtext("g:desc", default="", namespaces=ns)
        waypoints.append(
            {
                "name": clean_text(name),
                "desc": clean_text(desc),
                "lat": float(wpt.attrib["lat"]),
                "lon": float(wpt.attrib["lon"]),
            }
        )
    track_points = []
    for trkpt in root.findall(".//g:trkpt", ns):
        track_points.append((float(trkpt.attrib["lat"]), float(trkpt.attrib["lon"])))
    return {"waypoints": waypoints, "track_points": track_points}


def meters_per_degree_lat() -> float:
    return 111_320.0


def to_xy(point: Point, origin_lat: float) -> tuple[float, float]:
    lat, lon = point
    x = lon * meters_per_degree_lat() * math.cos(math.radians(origin_lat))
    y = lat * meters_per_degree_lat()
    return x, y


def distance_m(a: Point, b: Point) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    h = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def interpolate_track(track_points: list[Point], distance_from_start_m: float) -> Point | None:
    if not track_points:
        return None
    if distance_from_start_m <= 0:
        return track_points[0]
    covered = 0.0
    for a, b in zip(track_points, track_points[1:]):
        segment_m = distance_m(a, b)
        if segment_m <= 0:
            continue
        if covered + segment_m >= distance_from_start_m:
            ratio = (distance_from_start_m - covered) / segment_m
            return (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)
        covered += segment_m
    return track_points[-1]


def point_segment_distance_m(point: Point, a: Point, b: Point) -> float:
    origin_lat = point[0]
    px, py = to_xy(point, origin_lat)
    ax, ay = to_xy(a, origin_lat)
    bx, by = to_xy(b, origin_lat)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def point_line_distance_m(point: Point, line: list[Point]) -> float:
    best = float("inf")
    for a, b in zip(line, line[1:]):
        best = min(best, point_segment_distance_m(point, a, b))
    return best


def point_feature_distance_m(point: Point, feature: dict[str, Any]) -> float:
    return point_line_distance_m(point, feature["line"])


def route_feature_distance_m(track_points: list[Point], feature: dict[str, Any]) -> float:
    if not track_points:
        return float("inf")
    # Dense enough for these local route GPX files, while bounded for 30 routes.
    sample_step = max(1, len(track_points) // 300)
    best = float("inf")
    for point in track_points[::sample_step]:
        best = min(best, point_feature_distance_m(point, feature))
    return best


def is_vehicle_feature(props: dict[str, Any]) -> bool:
    highway = clean_text(props.get("highway")).lower()
    return highway in {
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "track",
        "road",
    }


def is_path_feature(props: dict[str, Any]) -> bool:
    highway = clean_text(props.get("highway")).lower()
    if highway in {"path", "footway", "cycleway", "bridleway", "steps"}:
        return True
    name = feature_name(props)
    return name.startswith("#") or "trail" in name.lower()


def nearby_features(point: Point | None, features: list[dict[str, Any]], limit: int = 8, radius_m: float = 175.0) -> list[dict[str, Any]]:
    if point is None:
        return []
    hits = []
    seen = set()
    lat, lon = point
    lat_pad = radius_m / meters_per_degree_lat()
    lon_pad = radius_m / (meters_per_degree_lat() * max(0.2, math.cos(math.radians(lat))))
    for feature in features:
        name = feature["name"]
        if name in seen:
            continue
        min_lat, min_lon, max_lat, max_lon = feature["bbox"]
        if lat < min_lat - lat_pad or lat > max_lat + lat_pad or lon < min_lon - lon_pad or lon > max_lon + lon_pad:
            continue
        dist = point_feature_distance_m(point, feature)
        if dist <= radius_m:
            props = feature["props"]
            seen.add(name)
            hits.append(
                {
                    "name": name,
                    "distance_m": round(dist),
                    "source": feature["source"],
                    "highway": clean_text(props.get("highway")),
                    "condition": clean_text(props.get("Condition")),
                    "level_of_use": clean_text(props.get("LevelOfUse")),
                    "all_weather": clean_text(props.get("AllWeather")),
                    "vehicle": is_vehicle_feature(props),
                    "path": is_path_feature(props),
                }
            )
    hits.sort(key=lambda item: (item["distance_m"], item["name"]))
    return hits[:limit]


def waypoint_for_cue(gpx: dict[str, Any], seq: int) -> Point | None:
    prefix = f"CUE {seq:02d}"
    for waypoint in gpx["waypoints"]:
        if waypoint["name"].startswith(prefix):
            return (waypoint["lat"], waypoint["lon"])
    return None


def point_for_cue(gpx: dict[str, Any], cue: dict[str, Any]) -> Point | None:
    route_miles = cue.get("route_miles")
    if route_miles is not None:
        try:
            return interpolate_track(gpx["track_points"], float(route_miles) * 1609.344)
        except (TypeError, ValueError):
            pass
    return waypoint_for_cue(gpx, int(cue["seq"]))


def named_waypoint(gpx: dict[str, Any], prefix: str) -> Point | None:
    for waypoint in gpx["waypoints"]:
        if waypoint["name"].startswith(prefix):
            return (waypoint["lat"], waypoint["lon"])
    return None


def waypoint_desc(gpx: dict[str, Any], prefix: str) -> str:
    for waypoint in gpx["waypoints"]:
        if waypoint["name"].startswith(prefix):
            return waypoint["desc"]
    return ""


def route_gpx_path(route: dict[str, Any]) -> Path:
    href = route.get("gpx_href") or ""
    return REPO_ROOT / "docs/field-packet" / href


def signed_as(cue: dict[str, Any]) -> str:
    names = cue.get("signed_as") or []
    return " / ".join(clean_text(name) for name in names if clean_text(name)) or "the active route line"


def route_car_context(route: dict[str, Any], point_nearby: list[dict[str, Any]], is_endpoint: bool = False) -> str:
    parking = route.get("parking") or {}
    vehicle_hits = [hit for hit in point_nearby if hit["vehicle"]]
    if is_endpoint:
        if parking.get("has_parking") is True:
            return "Cars are plausible at the parked-start surface; expect the route to begin with a parking/trailhead orientation problem, not just a trail problem."
        if "road" in clean_text(parking.get("name")).lower() or "parking anchor" in clean_text(parking.get("name")).lower():
            return "Cars or road-edge ambiguity are plausible because the start is a road or anchor-style access point; treat exact parking legality as a separate proof."
    if vehicle_hits:
        roads = ", ".join(hit["name"] for hit in vehicle_hits[:3])
        return f"Vehicle movement may be audible or visible near {roads}; do not mistake the road/driveway line for the trail branch."
    return "The local route data does not prove car traffic at this point; treat cars/road noise as field-only unless a road or parking surface is visibly present."


def visual_inference(cue: dict[str, Any], nearby: list[dict[str, Any]], route: dict[str, Any], endpoint: bool = False) -> str:
    other_paths = [hit for hit in nearby if hit["path"] and hit["name"] not in signed_as(cue)]
    visible_names = ", ".join(hit["name"] for hit in other_paths[:5])
    car_context = route_car_context(route, nearby, endpoint)
    if endpoint:
        base = f"Stand still first: identify the parked car, the route line, and the first/last signed trail before thinking about segment credit. {car_context}"
    elif other_paths:
        base = f"Runner frame: this is a visual branch, not a JSON row. Expect nearby trail choices in the local data ({visible_names}); use the signed cue and active leg before committing."
    else:
        base = "Runner frame: the immediate job is to keep the current trail until the named junction/landmark, with no extra branch proven by local data at this checkpoint."
    return base


def likely_visual_field(cue: dict[str, Any], nearby: list[dict[str, Any]], route: dict[str, Any], endpoint: bool = False) -> str:
    path_hits = [hit for hit in nearby if hit["path"]]
    vehicle_hits = [hit for hit in nearby if hit["vehicle"]]
    feature_hits = [hit for hit in nearby if not hit["path"] and not hit["vehicle"]]
    pieces = []
    if endpoint:
        pieces.append("car/parking orientation first")
    if path_hits:
        pieces.append("mapped trail/path choices near you: " + ", ".join(hit["name"] for hit in path_hits[:5]))
    elif feature_hits:
        pieces.append("mapped named route features near you: " + ", ".join(hit["name"] for hit in feature_hits[:5]))
    else:
        pieces.append("no mapped side trail/path inside the local audit radius")
    if vehicle_hits:
        pieces.append("vehicle corridor or service/residential road context: " + ", ".join(hit["name"] for hit in vehicle_hits[:4]))
    else:
        pieces.append("no vehicle corridor is proven near this checkpoint by the local overlay")
    if cue:
        pieces.append(f"the branch to privilege is `{signed_as(cue)}` until `{clean_text(cue.get('until'))}`")
    else:
        pieces.append("the branch to privilege is the first or final packet cue, not a nearby plausible line")
    pieces.append("actual visibility, signs, and car movement remain field/imagery proof gaps")
    return "; ".join(pieces) + "."


def checkpoint_risk(cue: dict[str, Any], nearby: list[dict[str, Any]], endpoint: bool = False) -> str:
    note = clean_text(cue.get("note"))
    warning = clean_text(cue.get("field_warning"))
    generic_connector = any("OSM " in name for name in cue.get("signed_as") or [])
    many_nearby = len([hit for hit in nearby if hit["path"]]) >= 4
    risks = []
    if endpoint:
        risks.append("start/finish access can fail even when route geometry passes")
    if warning or "Overlap warning" in note:
        risks.append("overlapping GPS line can make the correct direction ambiguous")
    if generic_connector:
        risks.append("generic OSM connector name may not exist on signs")
    if many_nearby:
        risks.append("multiple nearby trail lines can lure a tired runner onto a plausible wrong branch")
    if "Reverse direction would be steep" in note:
        risks.append("wrong-direction choice has meaningful climb penalty")
    return "; ".join(risks) or "main risk is ordinary trail-sign confirmation, not a known route-data blocker"


def access_status(route: dict[str, Any]) -> str:
    parking = route.get("parking") or {}
    name = clean_text(parking.get("name"))
    if "Strava parking anchor" in name:
        return "private-history parking anchor; usable as planning evidence but still public-proof limited"
    if "road" in name.lower() or "probe" in name.lower() or "roadside" in name.lower():
        return "parking/access proof-sensitive road or probe anchor"
    if parking.get("has_parking") is True:
        return "known-or-mapped parking in packet data"
    return "parking evidence incomplete in packet data"


def route_status(route: dict[str, Any]) -> str:
    validation = route.get("validation") or {}
    if not validation.get("passed"):
        return "coverage_gap"
    if route.get("wayfinding_cues"):
        return "needs_visual_proof"
    return "needs_cue_repair"


def route_audit(route: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
    gpx_path = route_gpx_path(route)
    gpx = parse_gpx(gpx_path)
    start_point = named_waypoint(gpx, "PARK/START")
    finish_point = named_waypoint(gpx, "RETURN TO CAR") or start_point
    checkpoints = []

    start_nearby = nearby_features(start_point, features)
    checkpoints.append(
        {
            "kind": "start",
            "label": "Start",
            "point_role": waypoint_desc(gpx, "PARK/START") or "Park here and start this outing.",
            "model_frame": "The packet proves the route has a start coordinate and a first cue.",
            "runner_frame": visual_inference({}, start_nearby, route, endpoint=True),
            "likely_visual_field": likely_visual_field({}, start_nearby, route, endpoint=True),
            "nearby": start_nearby,
            "decision": "Before moving, find the first signed trail or road-access line and confirm the car is parked where the return cue can actually resolve.",
            "risk": checkpoint_risk({}, start_nearby, endpoint=True),
            "evidence": "cue GPX waypoint plus local R2R/OSM overlay; no live imagery or field photo in this audit",
        }
    )

    for cue in route.get("wayfinding_cues") or []:
        point = point_for_cue(gpx, cue)
        near = nearby_features(point, features)
        checkpoints.append(
            {
                "kind": clean_text(cue.get("cue_type")) or "cue",
                "label": f"Cue {int(cue['seq']):02d}: {clean_text(cue.get('action'))} {signed_as(cue)}",
                "point_role": clean_text(cue.get("until")),
                "model_frame": f"The packet says `{clean_text(cue.get('compact'))}`.",
                "runner_frame": visual_inference(cue, near, route),
                "likely_visual_field": likely_visual_field(cue, near, route),
                "nearby": near,
                "decision": f"Follow {signed_as(cue)} until {clean_text(cue.get('until'))}; target is {clean_text(cue.get('target')) or 'the next route state'}.",
                "risk": checkpoint_risk(cue, near),
                "evidence": "field-packet cue plus cue GPX waypoint plus local R2R/OSM overlay; sightlines are inferred, not field-observed",
            }
        )

    finish_nearby = nearby_features(finish_point, features)
    checkpoints.append(
        {
            "kind": "finish",
            "label": "Finish / return to car",
            "point_role": waypoint_desc(gpx, "RETURN TO CAR") or "Route endpoint / return-to-car point.",
            "model_frame": "The packet endpoint closes the loop.",
            "runner_frame": visual_inference({}, finish_nearby, route, endpoint=True),
            "likely_visual_field": likely_visual_field({}, finish_nearby, route, endpoint=True),
            "nearby": finish_nearby,
            "decision": "Do not stop the BTC recording early; finish the actual return-to-car leg and then save/upload according to the BTC workflow.",
            "risk": checkpoint_risk({}, finish_nearby, endpoint=True),
            "evidence": "cue GPX return waypoint plus local R2R/OSM overlay; no official completion proof without the eventual activity geometry",
        }
    )

    route_nearby = []
    seen = set()
    for checkpoint in checkpoints:
        for hit in checkpoint["nearby"]:
            if hit["name"] in seen:
                continue
            seen.add(hit["name"])
            route_nearby.append(hit)
    route_nearby.sort(key=lambda item: (item["distance_m"], item["name"]))

    return {
        "label": route["label"],
        "outing_id": route.get("outing_id"),
        "trailhead": route.get("trailhead"),
        "parking": route.get("parking") or {},
        "trails": route.get("trails") or [],
        "official_miles": route.get("official_miles"),
        "on_foot_miles": route.get("on_foot_miles"),
        "door_to_door_minutes_p75": route.get("door_to_door_minutes_p75"),
        "door_to_door_minutes_p90": route.get("door_to_door_minutes_p90"),
        "wayfinding_cue_count": len(route.get("wayfinding_cues") or []),
        "segment_count": len(route.get("segment_ids") or []),
        "gpx": str(gpx_path.relative_to(REPO_ROOT)),
        "access_status": access_status(route),
        "human_validity_status": route_status(route),
        "frame_decision": "needs-proof",
        "frame_decision_reason": "The packet can support a model-to-runner visualization audit, but literal sightlines, signs, car movement, and trail-in-distance claims remain field/imagery proof gaps.",
        "checkpoints": checkpoints,
        "route_line_nearby_features": route_nearby[:20],
    }


def render_nearby(nearby: list[dict[str, Any]]) -> str:
    if not nearby:
        return "None within the audit radius from local overlay data."
    parts = []
    for hit in nearby[:6]:
        detail = hit["name"]
        if hit.get("highway"):
            detail += f" ({hit['highway']})"
        detail += f" ~{hit['distance_m']}m"
        parts.append(detail)
    return "; ".join(parts)


def render_route_markdown(audit: dict[str, Any]) -> str:
    route_title = f"{audit['label']} - {audit['trailhead']}"
    lines = [
        f"# Runner-Perspective Frame Shift: {route_title}",
        "",
        "## Frame Contract",
        "",
        f"- Route card: `{audit['label']}` / outing `{audit['outing_id']}`.",
        f"- Field use: choose the right physical branch from parked car back to parked car while preserving official segment credit.",
        f"- Evidence used: `docs/field-packet/field-tool-data.json`, `{audit['gpx']}`, R2R open data, OSM connector overlay, official 2026 segment source.",
        "- Evidence not used: live field photos, Street View, current day-of signage, current mud/closure report, actual runner sightline.",
        f"- Frame decision: `{audit['frame_decision']}`. {audit['frame_decision_reason']}",
        f"- Access status: {audit['access_status']}.",
        f"- Human-validity status for this audit: `{audit['human_validity_status']}`.",
        "",
        "## Route Snapshot",
        "",
        f"- Trails: {', '.join(audit['trails'])}.",
        f"- Official miles: {audit['official_miles']}; on-foot miles: {audit['on_foot_miles']}.",
        f"- Door-to-door: p75 {audit['door_to_door_minutes_p75']} min; p90 {audit['door_to_door_minutes_p90']} min.",
        f"- Segment count: {audit['segment_count']}; wayfinding cue count: {audit['wayfinding_cue_count']}.",
        "",
        "## Start-End-Junction Frame Shifts",
        "",
    ]
    for checkpoint in audit["checkpoints"]:
        lines.extend(
            [
                f"### {checkpoint['label']}",
                "",
                f"- Physical role: {checkpoint['point_role']}",
                f"- Model frame: {checkpoint['model_frame']}",
                f"- Runner frame: {checkpoint['runner_frame']}",
                f"- Likely visual field: {checkpoint['likely_visual_field']}",
                f"- Nearby trails/roads from local overlays: {render_nearby(checkpoint['nearby'])}",
                f"- Decision as runner: {checkpoint['decision']}",
                f"- Wrong-layer risk: {checkpoint['risk']}",
                f"- Evidence boundary: {checkpoint['evidence']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Whole-Route Frame Shift",
            "",
            "- Original frame: a route card can be valid because coverage, GPX continuity, and cue exports exist.",
            "- Runner frame: the route is only executable if each visible branch, road edge, return leg, overlap, and indistinct connector can be recognized while tired and moving.",
            "- Adjacent frame checked: literal sightline proof from imagery or field photos. This audit does not have that proof, so it keeps sightline claims bounded.",
            "- Adjacent frame checked: route-card certification. This audit reads route-card cues but does not rerun the full certification chain.",
            "- Adjacent frame checked: day-of legality. This audit does not replace current Ridge to Rivers condition/signage checks.",
            "",
            "## Adversarial Failure Stories",
            "",
            "- The GPX line is correct, but the runner starts on the first official-credit label instead of the actual access leg from the car.",
            "- A side trail or road line near a cue looks plausible in the distance, and the runner follows the visual line instead of the signed cue/active leg.",
            "- A generic OSM connector or repeat leg has no field sign, so the runner needs the active GPX leg rather than a sign name.",
            "- Overlap or double-back geometry causes the map to look like one line, hiding the fact that the active direction changed.",
            "- The route answers the coverage question but misses the day-of condition question: closure, mud, heat, or signage can still block execution.",
            "",
            "## Route-Line Nearby Features",
            "",
        ]
    )
    if audit["route_line_nearby_features"]:
        for hit in audit["route_line_nearby_features"]:
            kind = "vehicle" if hit["vehicle"] else "path" if hit["path"] else "feature"
            highway = f", highway={hit['highway']}" if hit.get("highway") else ""
            lines.append(f"- {hit['name']} (~{hit['distance_m']}m, {kind}{highway}, source={hit['source']})")
    else:
        lines.append("- No route-line nearby features found in the local overlay threshold.")
    lines.append("")
    lines.extend(
        [
            "## Required Next Proof",
            "",
            "- For literal `what do I see?` confidence: inspect current imagery or field photos for the checkpoint and verify signs/road visibility.",
            "- Before running: check current Ridge to Rivers conditions, closures, special-management direction rules, heat, and water.",
            "- Before claiming challenge credit: validate the eventual BTC activity geometry against official full-segment coverage and ascent direction.",
            "",
        ]
    )
    return "\n".join(lines)


def render_index(audits: list[dict[str, Any]]) -> str:
    lines = [
        "# Runner-Perspective Frame-Shift Audit Index",
        "",
        "This checkpoint decomposes the current 2026 field-packet routes into start, cue/junction, and finish decision points, then reframes each point from model artifact logic into runner-on-trail logic.",
        "",
        "Important boundary: this is a local-data visual reasoning audit. It does not contain live field photos, current Street View, or day-of Ridge to Rivers condition checks, so literal sightline claims remain proof-gated.",
        "",
        "## Inputs",
        "",
        "- `docs/field-packet/field-tool-data.json`",
        "- `docs/field-packet/gpx/official/*.gpx`",
        "- `years/2026/inputs/open-data/r2r-trails-2026-05-04/boise_parks_trails_open_data.geojson`",
        "- `years/2026/inputs/open-data/routing-connectors-2026-05-04/combined_r2r_osm_connectors.geojson`",
        "- `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson`",
        "",
        "## Method",
        "",
        "For every current field-packet route, the audit reads the route card, cue GPX waypoints, and nearby named trail/road features. It then records a frame shift at the parked start, every packet cue/decision point, and the return-to-car endpoint.",
        "",
        "The key frame shift is: `the model sees a valid route artifact; the runner sees signs, branches, roads, other trail lines, overlap, fatigue, and uncertainty`.",
        "",
        "## Routes",
        "",
        "| Route | Trailhead | Cues | Access status | Audit status | File |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for audit in audits:
        filename = f"{slugify(audit['label'])}-{slugify(str(audit['trailhead']))}.md"
        lines.append(
            f"| {audit['label']} | {audit['trailhead']} | {audit['wayfinding_cue_count']} | {audit['access_status']} | {audit['human_validity_status']} | [{filename}](route-audits/{filename}) |"
        )
    lines.extend(
        [
            "",
            "## Cross-Route Findings",
            "",
            "- The dominant gap is literal sightline proof: local route data can identify nearby named branches and road features, but cannot prove what the runner can visually see at trail speed.",
            "- Routes with generic OSM connectors, private-history anchors, road/probe starts, overlap warnings, or many nearby named trails deserve the first field-photo or imagery pass.",
            "- This audit intentionally does not promote or reject routes; it changes the proof burden from `does the route artifact exist?` to `can the runner choose correctly at each visible decision point?`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_frame_logs(audits: list[dict[str, Any]]) -> None:
    FRAME_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now()
    with FRAME_LOG.open("a", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(
                json.dumps(
                    {
                        "timestamp_utc": timestamp,
                        "cwd": str(REPO_ROOT),
                        "artifact_type": "route guide",
                        "decision": audit["frame_decision"],
                        "original_assumption": f"{audit['label']} can be assessed from route-card/GPX validity.",
                        "challenge": "The user asked what the runner sees at start, finish, junctions, and decision points.",
                        "new_frame": "Assess each route checkpoint as an on-trail branch/visibility/signage problem, with sightlines proof-gated.",
                        "resulting_perspective": "Generated a per-route runner-perspective audit with nearby trail/road context and explicit visual proof gaps.",
                        "widened_options": [
                            "route-card validity",
                            "runner checkpoint decision",
                            "literal imagery/field-photo sightline proof",
                        ],
                        "frame_iterations": [
                            "artifact validity",
                            "runner-on-trail decision points",
                            "sightline proof boundary",
                        ],
                        "evidence_level": "supplemental connector data plus route-card cues; live sightline unknown",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def write_outputs(audits: list[dict[str, Any]], output_dir: Path) -> None:
    route_dir = output_dir / "route-audits"
    route_dir.mkdir(parents=True, exist_ok=True)
    for audit in audits:
        filename = f"{slugify(audit['label'])}-{slugify(str(audit['trailhead']))}.md"
        (route_dir / filename).write_text(render_route_markdown(audit), encoding="utf-8")
    (output_dir / "index.md").write_text(render_index(audits), encoding="utf-8")
    manifest = {
        "generated_at": utc_now(),
        "route_count": len(audits),
        "checkpoint_count": sum(len(audit["checkpoints"]) for audit in audits),
        "inputs": [
            str(FIELD_TOOL_DATA.relative_to(REPO_ROOT)),
            str(R2R_GEOJSON.relative_to(REPO_ROOT)),
            str(CONNECTORS_GEOJSON.relative_to(REPO_ROOT)),
            str(OFFICIAL_SEGMENTS.relative_to(REPO_ROOT)),
        ],
        "outputs": ["index.md", "route-audits/*.md"],
        "routes": [
            {
                "label": audit["label"],
                "trailhead": audit["trailhead"],
                "cue_count": audit["wayfinding_cue_count"],
                "checkpoint_count": len(audit["checkpoints"]),
                "frame_decision": audit["frame_decision"],
                "human_validity_status": audit["human_validity_status"],
            }
            for audit in audits
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-frame-log", action="store_true", help="Do not append frame-shift debug logs.")
    args = parser.parse_args(argv)

    field_tool_data = load_json(FIELD_TOOL_DATA)
    features = load_features([R2R_GEOJSON, CONNECTORS_GEOJSON])
    audits = [route_audit(route, features) for route in field_tool_data.get("routes") or []]
    write_outputs(audits, args.output_dir)
    if not args.skip_frame_log:
        write_frame_logs(audits)
    print(f"Wrote {len(audits)} route audits to {args.output_dir}")
    print(f"Checkpoint count: {sum(len(audit['checkpoints']) for audit in audits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
