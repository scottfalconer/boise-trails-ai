#!/usr/bin/env python3
"""Export a side-by-side HTML map for the selected calendar runbook."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from export_execution_gpx import (  # noqa: E402
    candidate_segment_coordinates,
    candidate_track_coordinates,
    load_candidate_index,
    load_official_segment_index,
    validate_track_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    load_connector_graph,
    load_official_segments,
    read_json,
)


DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_RUNBOOK_JSON = YEAR_DIR / "outputs" / "private" / "2026-personal-ideal-plan.json"
DEFAULT_OUTPUT_HTML = YEAR_DIR / "outputs" / "private" / "2026-personal-ideal-plan-map.html"
DEFAULT_OUTPUT_DATA = YEAR_DIR / "outputs" / "private" / "2026-personal-ideal-plan-map-data.json"

DAY_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#0f766e",
    "#b45309",
    "#7c3aed",
    "#15803d",
]


def feature(
    geometry_type: str,
    coordinates: list[Any],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": geometry_type,
            "coordinates": coordinates,
        },
        "properties": properties,
    }


def line_feature(
    coords: list[tuple[float, float]] | list[list[float]],
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    if len(coords) < 2:
        return None
    return feature(
        "LineString",
        [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in coords],
        properties,
    )


def multiline_feature(
    parts: list[list[tuple[float, float]]],
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    usable_parts = [part for part in parts if len(part) >= 2]
    if not usable_parts:
        return None
    if len(usable_parts) == 1:
        return line_feature(usable_parts[0], properties)
    return feature(
        "MultiLineString",
        [
            [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in part]
            for part in usable_parts
        ],
        properties,
    )


def point_feature(lon: float, lat: float, properties: dict[str, Any]) -> dict[str, Any]:
    return feature("Point", [round(float(lon), 6), round(float(lat), 6)], properties)


def round_number(value: Any, digits: int = 2) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    if number.is_integer():
        return int(number)
    return round(number, digits)


def outing_title(outing: dict[str, Any]) -> str:
    names = outing.get("trail_names") or []
    return ", ".join(str(name) for name in names) or str(outing.get("outing_id") or "outing")


def day_title(day: dict[str, Any]) -> str:
    return str(day.get("date") or "unknown day")


def split_coords_on_gaps(
    coords: list[tuple[float, float]],
    max_gap_miles: float = 0.1,
) -> list[list[tuple[float, float]]]:
    if not coords:
        return []
    parts: list[list[tuple[float, float]]] = [[coords[0]]]
    for left, right in zip(coords, coords[1:]):
        if haversine_miles(left, right) > max_gap_miles:
            parts.append([right])
        else:
            parts[-1].append(right)
    return [part for part in parts if len(part) >= 2]


def build_plan_map_data(
    runbook: dict[str, Any],
    plan: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_index = load_candidate_index(plan)
    on_foot_features: list[dict[str, Any]] = []
    official_features: list[dict[str, Any]] = []
    drive_features: list[dict[str, Any]] = []
    trailhead_features_by_key: dict[str, dict[str, Any]] = {}
    days: list[dict[str, Any]] = []
    route_validations: list[dict[str, Any]] = []

    for day_index, day in enumerate(runbook.get("days") or []):
        color = DAY_PALETTE[day_index % len(DAY_PALETTE)]
        day_id = day_title(day)
        day_outings = []

        for outing_index, outing in enumerate(day.get("outings") or []):
            outing_id = str(outing.get("outing_id"))
            candidate = candidate_index.get(outing_id)
            if not candidate:
                route_validations.append(
                    {
                        "day": day_id,
                        "outing_id": outing_id,
                        "source_gap_warning": True,
                        "source_max_trackpoint_gap_miles": None,
                        "source_failures": [{"code": "candidate_not_found"}],
                        "rendered_track_passed": False,
                        "rendered_track_segment_count": 0,
                        "rendered_max_trackpoint_gap_miles": None,
                        "rendered_failures": [{"code": "candidate_not_found"}],
                    }
                )
                continue

            route_name = outing_title(outing)
            route_coords = candidate_track_coordinates(
                candidate,
                official_index,
                connector_graph=connector_graph,
            )
            source_validation = validate_track_segments([route_coords], max_gap_miles=0.1)
            route_parts = split_coords_on_gaps(route_coords, max_gap_miles=0.1)
            render_validation = validate_track_segments(route_parts, max_gap_miles=0.1)
            gap_count = len(source_validation.get("failures") or [])
            route_props = {
                "kind": "on_foot_route",
                "date": day_id,
                "day_index": day_index,
                "outing_index": outing_index,
                "outing_id": outing_id,
                "title": route_name,
                "trailhead": outing.get("trailhead"),
                "route_label": outing.get("route_label"),
                "official_miles": round_number(outing.get("new_official_miles")),
                "on_foot_miles": round_number(outing.get("estimated_total_on_foot_miles")),
                "ascent_ft": round_number(outing.get("ascent_ft"), 0),
                "color": color,
                "map_gap_warning_count": gap_count,
            }
            route_feature = multiline_feature(route_parts, route_props)
            if route_feature:
                on_foot_features.append(route_feature)
            route_validations.append(
                {
                    "day": day_id,
                    "outing_id": outing_id,
                    "source_gap_warning": not source_validation["passed"],
                    "source_max_trackpoint_gap_miles": source_validation[
                        "max_trackpoint_gap_miles"
                    ],
                    "source_failures": source_validation["failures"],
                    "rendered_track_passed": render_validation["passed"],
                    "rendered_track_segment_count": render_validation["track_segment_count"],
                    "rendered_max_trackpoint_gap_miles": render_validation[
                        "max_trackpoint_gap_miles"
                    ],
                    "rendered_failures": render_validation["failures"],
                }
            )

            for segment in candidate.get("segments") or []:
                segment_coords = candidate_segment_coordinates(candidate, segment, official_index)
                segment_feature = line_feature(
                    segment_coords,
                    {
                        **route_props,
                        "kind": "official_segment",
                        "seg_id": segment.get("seg_id"),
                        "segment_name": segment.get("seg_name") or segment.get("trail_name"),
                    },
                )
                if segment_feature:
                    official_features.append(segment_feature)

            park = outing.get("park") or {}
            if park.get("lat") is not None and park.get("lon") is not None:
                trailhead_name = str(park.get("trailhead") or outing.get("trailhead") or "Trailhead")
                key = f"{trailhead_name}:{float(park['lat']):.6f}:{float(park['lon']):.6f}"
                trailhead_features_by_key[key] = point_feature(
                    float(park["lon"]),
                    float(park["lat"]),
                    {
                        "kind": "trailhead",
                        "name": trailhead_name,
                        "parking_confidence": park.get("parking_confidence"),
                        "can_park": bool(park.get("can_park")),
                        "facility_status": park.get("facility_status"),
                    },
                )

            day_outings.append(
                {
                    "outing_id": outing_id,
                    "title": route_name,
                    "trailhead": outing.get("trailhead"),
                    "route_label": outing.get("route_label"),
                    "official_miles": round_number(outing.get("new_official_miles")),
                    "on_foot_miles": round_number(outing.get("estimated_total_on_foot_miles")),
                    "ascent_ft": round_number(outing.get("ascent_ft"), 0),
                    "map_gap_warning_count": gap_count,
                }
            )

        for drive_index, leg in enumerate(((day.get("day_transport") or {}).get("legs")) or []):
            coords = ((leg.get("geometry") or {}).get("coordinates")) or []
            drive_feature = line_feature(
                coords,
                {
                    "kind": "drive",
                    "date": day_id,
                    "day_index": day_index,
                    "drive_index": drive_index,
                    "leg_type": leg.get("leg_type"),
                    "from": leg.get("from"),
                    "to": leg.get("to"),
                    "distance_miles": round_number(leg.get("distance_miles")),
                    "duration_minutes": round_number(leg.get("duration_minutes"), 0),
                    "color": color,
                },
            )
            if drive_feature:
                drive_features.append(drive_feature)

        days.append(
            {
                "date": day_id,
                "color": color,
                "official_miles": round_number(day.get("official_new_miles")),
                "on_foot_miles": round_number(day.get("estimated_total_on_foot_miles")),
                "ascent_ft": round_number(day.get("ascent_ft"), 0),
                "realistic_total_minutes": round_number(day.get("realistic_total_minutes"), 0),
                "requires_normal_cap_exception": bool(day.get("requires_normal_cap_exception")),
                "normal_cap_exception_minutes": round_number(day.get("normal_cap_exception_minutes"), 0),
                "trailheads": day.get("trailheads") or [],
                "outings": day_outings,
            }
        )

    source_gap_warnings = [
        validation for validation in route_validations if validation.get("source_gap_warning")
    ]
    render_failures = [
        validation for validation in route_validations if not validation.get("rendered_track_passed")
    ]
    return {
        "profile_name": runbook.get("profile_name"),
        "run_id": runbook.get("run_id"),
        "summary": runbook.get("summary") or {},
        "audit": runbook.get("audit") or {},
        "source": {
            "runbook_json": str(DEFAULT_RUNBOOK_JSON),
            "plan_json": str(DEFAULT_PLAN_JSON),
        },
        "days": days,
        "feature_collections": {
            "on_foot_routes": {"type": "FeatureCollection", "features": on_foot_features},
            "official_segments": {"type": "FeatureCollection", "features": official_features},
            "drives": {"type": "FeatureCollection", "features": drive_features},
            "trailheads": {
                "type": "FeatureCollection",
                "features": list(trailhead_features_by_key.values()),
            },
        },
        "map_validation": {
            "route_count": len(route_validations),
            "rendered_route_geometry_validation_passed": not render_failures,
            "rendered_failed_route_count": len(render_failures),
            "source_route_gap_warning_count": len(source_gap_warnings),
            "route_validations": route_validations,
        },
    }


def render_html(map_data: dict[str, Any]) -> str:
    payload = json.dumps(map_data, separators=(",", ":"))
    escaped_title = html.escape(str(map_data.get("profile_name") or "2026 personal plan"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2026 Boise Trails Plan Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIINfQkBRjzRoR2DIN+mqMZcV6tYaGNcJyk="
    crossorigin="">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8f5;
      --panel: #ffffff;
      --line: #d7ddd4;
      --text: #1f2933;
      --muted: #667085;
      --accent: #2563eb;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(360px, 430px) minmax(0, 1fr);
      height: 100vh;
      min-height: 100vh;
      overflow: hidden;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }}
    header {{
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
    }}
    .metric {{
      min-height: 54px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfa;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      text-transform: uppercase;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 17px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }}
    button, label.toggle {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      min-height: 34px;
      border-radius: 6px;
      padding: 7px 9px;
      font-size: 13px;
      line-height: 1.2;
      cursor: pointer;
    }}
    button.active {{
      border-color: var(--accent);
      color: #174ea6;
      background: #eef4ff;
    }}
    label.toggle {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      cursor: default;
    }}
    .day-list {{
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding: 12px;
    }}
    .day {{
      border: 1px solid var(--line);
      border-left-width: 6px;
      border-radius: 6px;
      margin-bottom: 10px;
      background: #fff;
    }}
    .day.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
    }}
    .day button {{
      width: 100%;
      border: 0;
      border-radius: 0;
      display: block;
      text-align: left;
      padding: 11px 12px;
      background: transparent;
      min-height: 0;
    }}
    .day-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
      font-size: 14px;
      line-height: 1.35;
    }}
    .day-meta {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .exception {{
      color: var(--danger);
      font-weight: 700;
    }}
    .outings {{
      border-top: 1px solid var(--line);
      padding: 8px 12px 11px;
    }}
    .outing {{
      margin: 7px 0;
      font-size: 12px;
      line-height: 1.35;
    }}
    .outing strong {{
      display: block;
      font-size: 12px;
    }}
    .outing span {{
      color: var(--muted);
    }}
    main {{
      position: relative;
      min-width: 0;
      height: 100vh;
      min-height: 100vh;
      overflow: hidden;
    }}
    #map {{
      position: absolute;
      inset: 0;
      overflow: hidden;
      background: #e4eadf;
    }}
    .leaflet-container {{
      height: 100%;
      width: 100%;
      overflow: hidden;
      background: #e4eadf;
    }}
    .leaflet-pane,
    .leaflet-tile,
    .leaflet-marker-icon,
    .leaflet-marker-shadow,
    .leaflet-tile-container,
    .leaflet-pane > svg,
    .leaflet-pane > canvas,
    .leaflet-zoom-box,
    .leaflet-image-layer,
    .leaflet-layer {{
      position: absolute;
      left: 0;
      top: 0;
    }}
    .leaflet-tile,
    .leaflet-marker-icon,
    .leaflet-marker-shadow {{
      user-select: none;
      -webkit-user-drag: none;
    }}
    .leaflet-tile {{
      visibility: hidden;
    }}
    .leaflet-tile-loaded {{
      visibility: inherit;
    }}
    .leaflet-pane {{ z-index: 400; }}
    .leaflet-tile-pane {{ z-index: 200; }}
    .leaflet-overlay-pane {{ z-index: 400; }}
    .leaflet-shadow-pane {{ z-index: 500; }}
    .leaflet-marker-pane {{ z-index: 600; }}
    .leaflet-tooltip-pane {{ z-index: 650; }}
    .leaflet-popup-pane {{ z-index: 700; }}
    .leaflet-map-pane canvas {{ z-index: 100; }}
    .leaflet-map-pane svg {{ z-index: 200; }}
    .leaflet-top,
    .leaflet-bottom {{
      position: absolute;
      z-index: 1000;
      pointer-events: none;
    }}
    .leaflet-top {{ top: 0; }}
    .leaflet-right {{ right: 0; }}
    .leaflet-bottom {{ bottom: 0; }}
    .leaflet-left {{ left: 0; }}
    .leaflet-control {{
      position: relative;
      z-index: 800;
      float: left;
      clear: both;
      pointer-events: auto;
    }}
    .leaflet-right .leaflet-control {{ float: right; }}
    .leaflet-top .leaflet-control {{ margin-top: 10px; }}
    .leaflet-bottom .leaflet-control {{ margin-bottom: 10px; }}
    .leaflet-left .leaflet-control {{ margin-left: 10px; }}
    .leaflet-right .leaflet-control {{ margin-right: 10px; }}
    .leaflet-bar {{
      border: 1px solid rgba(15, 23, 42, 0.2);
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.16);
      overflow: hidden;
    }}
    .leaflet-bar a {{
      display: block;
      width: 28px;
      height: 28px;
      line-height: 28px;
      border-bottom: 1px solid rgba(15, 23, 42, 0.18);
      background: #fff;
      color: #111827;
      text-align: center;
      text-decoration: none;
      font-weight: 700;
    }}
    .leaflet-bar a:last-child {{
      border-bottom: 0;
    }}
    .leaflet-control-attribution {{
      background: rgba(255, 255, 255, 0.84);
      padding: 0 5px;
      color: #374151;
      font-size: 11px;
    }}
    .legend {{
      position: absolute;
      right: 16px;
      bottom: 20px;
      z-index: 500;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
      font-size: 12px;
      line-height: 1.4;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0;
    }}
    .sample {{
      width: 30px;
      height: 0;
      border-top: 4px solid #2563eb;
    }}
    .sample.full {{ border-top-width: 3px; opacity: 0.5; }}
    .sample.drive {{ border-top: 3px dashed #6b7280; }}
    .sample-arrow {{
      display: inline-block;
      width: 0;
      height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-bottom: 12px solid #1f2937;
      transform: rotate(90deg);
    }}
    .dir-arrow-wrap {{ background: transparent; border: 0; }}
    .dir-arrow {{
      width: 0;
      height: 0;
      border-left: 6px solid transparent;
      border-right: 6px solid transparent;
      border-bottom: 14px solid var(--route-color, #1f2937);
      filter: drop-shadow(0 0 2px #fff) drop-shadow(0 0 2px #fff);
      transform-origin: 50% 50%;
    }}
    .parking-marker-wrap {{ background: transparent; border: 0; }}
    .parking-marker {{
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #111827;
      color: #fff;
      border: 2px solid #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.32);
    }}
    .path-marker-wrap {{ background: transparent; border: 0; }}
    .path-marker {{
      min-width: 24px;
      height: 24px;
      border-radius: 999px;
      background: #fff;
      color: #111827;
      border: 2px solid #111827;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 6px;
      font-weight: 800;
      font-size: 11px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.25);
    }}
    .path-marker.turn {{ background: #fef3c7; }}
    @media (max-width: 860px) {{
      .app {{ grid-template-columns: 1fr; grid-template-rows: minmax(260px, 46vh) minmax(0, 1fr); }}
      aside {{ order: 2; border-right: 0; border-top: 1px solid var(--line); min-height: 0; }}
      main {{ order: 1; height: auto; min-height: 0; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>2026 Personal Plan Map</h1>
        <p class="subtitle">{escaped_title} with official segments, full on-foot routes, trailheads, and day-level drives.</p>
      </header>
      <section class="metrics" id="metrics"></section>
      <section class="controls">
        <button type="button" id="showAll" class="active">All days</button>
        <label class="toggle"><input id="officialToggle" type="checkbox" checked> Official</label>
        <label class="toggle"><input id="fullRouteToggle" type="checkbox" checked> Full route</label>
        <label class="toggle"><input id="driveToggle" type="checkbox"> Drives</label>
      </section>
      <section class="day-list" id="dayList"></section>
    </aside>
    <main>
      <div id="map"></div>
      <div class="legend">
        <div class="legend-row"><span class="sample"></span><span>Official challenge segment</span></div>
        <div class="legend-row"><span class="sample full"></span><span>Full on-foot route</span></div>
        <div class="legend-row"><span class="sample-arrow"></span><span>Selected day route line, arrows, and turn markers</span></div>
        <div class="legend-row"><span class="parking-marker">P</span><span>Parking / start trailhead</span></div>
        <div class="legend-row"><span class="sample drive"></span><span>Drive leg</span></div>
      </div>
    </main>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""></script>
  <script>
    const PLAN_MAP_DATA = {payload};
    const state = {{
      selectedDay: null,
      showOfficial: true,
      showFull: true,
      showDrives: false
    }};
    const map = L.map("map", {{
      preferCanvas: true,
      fadeAnimation: false,
      markerZoomAnimation: false
    }});
    const baseLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
      maxZoom: 19,
      maxNativeZoom: 18,
      updateWhenIdle: true,
      updateWhenZooming: false,
      keepBuffer: 4,
      attribution: "Tiles &copy; Esri"
    }}).addTo(map);
    baseLayer.on("tileerror", event => {{
      if (event.tile) event.tile.style.visibility = "hidden";
    }});

    const officialLayer = L.layerGroup().addTo(map);
    const routeLayer = L.layerGroup().addTo(map);
    const arrowLayer = L.layerGroup().addTo(map);
    const driveLayer = L.layerGroup().addTo(map);
    const markerLayer = L.layerGroup().addTo(map);

    function fmt(value, suffix = "") {{
      if (value === null || value === undefined) return "n/a";
      return `${{value}}${{suffix}}`;
    }}

    function matchesDay(feature) {{
      return !state.selectedDay || feature.properties.date === state.selectedDay || feature.properties.kind === "trailhead";
    }}

    function popupHtml(props) {{
      if (props.kind === "trailhead") {{
        return `<strong>${{props.name}}</strong><br>Parking: ${{props.can_park ? "yes" : "unknown"}}<br>Confidence: ${{props.parking_confidence || "unknown"}}`;
      }}
      if (props.kind === "drive") {{
        return `<strong>${{props.leg_type}}</strong><br>${{props.from || ""}} to ${{props.to || ""}}<br>${{fmt(props.distance_miles, " mi")}}, ${{fmt(props.duration_minutes, " min")}}`;
      }}
      if (props.kind === "official_segment") {{
        return `<strong>${{props.segment_name || props.title}}</strong><br>${{props.date}}<br>${{props.title}}`;
      }}
      const warning = props.map_gap_warning_count ? `<br>Map gaps split: ${{props.map_gap_warning_count}}` : "";
      return `<strong>${{props.title}}</strong><br>${{props.date}}<br>${{fmt(props.official_miles, " official mi")}}, ${{fmt(props.on_foot_miles, " total mi")}}<br>${{fmt(props.ascent_ft, " ft ascent")}}${{warning}}`;
    }}

    function addGeoJson(collection, targetLayer, styleFn, pointFn) {{
      L.geoJSON(collection, {{
        filter: matchesDay,
        style: styleFn,
        pointToLayer: pointFn,
        onEachFeature: (feature, layer) => layer.bindPopup(popupHtml(feature.properties || {{}}))
      }}).addTo(targetLayer);
    }}

    function parkingIcon() {{
      return L.divIcon({{
        className: "parking-marker-wrap",
        iconSize: [26, 26],
        iconAnchor: [13, 13],
        html: '<div class="parking-marker">P</div>'
      }});
    }}

    function pathMarkerIcon(label, extraClass = "") {{
      return L.divIcon({{
        className: "path-marker-wrap",
        iconSize: [32, 26],
        iconAnchor: [16, 13],
        html: `<div class="path-marker ${{extraClass}}">${{label}}</div>`
      }});
    }}

    function routeParts(feature) {{
      const geom = feature.geometry || {{}};
      if (geom.type === "LineString") return [geom.coordinates || []];
      if (geom.type === "MultiLineString") return geom.coordinates || [];
      return [];
    }}

    const OUT_AND_BACK_OFFSET_METERS = 10;

    function coordKey(pt) {{
      return `${{Number(pt[0]).toFixed(5)}},${{Number(pt[1]).toFixed(5)}}`;
    }}

    function orientedSegmentKey(a, b) {{
      return `${{coordKey(a)}}>${{coordKey(b)}}`;
    }}

    function segmentKey(a, b) {{
      const left = coordKey(a);
      const right = coordKey(b);
      return left < right ? `${{left}}|${{right}}` : `${{right}}|${{left}}`;
    }}

    function opposingSegmentKeys(feature) {{
      const oriented = new Set();
      routeParts(feature).forEach(rawPart => {{
        const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
        for (let i = 1; i < part.length; i += 1) {{
          oriented.add(orientedSegmentKey(part[i - 1], part[i]));
        }}
      }});
      const opposing = new Set();
      routeParts(feature).forEach(rawPart => {{
        const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
        for (let i = 1; i < part.length; i += 1) {{
          const left = part[i - 1];
          const right = part[i];
          if (oriented.has(orientedSegmentKey(right, left))) opposing.add(segmentKey(left, right));
        }}
      }});
      return opposing;
    }}

    function normalMeters(a, b) {{
      const meanLat = ((a[1] + b[1]) / 2) * Math.PI / 180;
      const dx = (b[0] - a[0]) * 111320 * Math.cos(meanLat);
      const dy = (b[1] - a[1]) * 110540;
      const length = Math.sqrt(dx * dx + dy * dy);
      if (length <= 0) return null;
      return [-dy / length, dx / length];
    }}

    function offsetPoint(pt, normal) {{
      const latRad = pt[1] * Math.PI / 180;
      const lonScale = Math.max(1, 111320 * Math.cos(latRad));
      return [pt[0] + normal[0] / lonScale, pt[1] + normal[1] / 110540];
    }}

    function offsetLinePart(part, opposing) {{
      if (!opposing || !opposing.size) return part;
      return part.map((pt, index) => {{
        let nx = 0;
        let ny = 0;
        let count = 0;
        if (index > 0 && opposing.has(segmentKey(part[index - 1], pt))) {{
          const normal = normalMeters(part[index - 1], pt);
          if (normal) {{ nx += normal[0]; ny += normal[1]; count += 1; }}
        }}
        if (index < part.length - 1 && opposing.has(segmentKey(pt, part[index + 1]))) {{
          const normal = normalMeters(pt, part[index + 1]);
          if (normal) {{ nx += normal[0]; ny += normal[1]; count += 1; }}
        }}
        if (!count) return pt;
        const length = Math.sqrt(nx * nx + ny * ny);
        if (length <= 0) return pt;
        return offsetPoint(pt, [nx / length * OUT_AND_BACK_OFFSET_METERS, ny / length * OUT_AND_BACK_OFFSET_METERS]);
      }});
    }}

    function segmentMeters(a, b) {{
      const meanLat = ((a[1] + b[1]) / 2) * Math.PI / 180;
      const dx = (b[0] - a[0]) * 111320 * Math.cos(meanLat);
      const dy = (b[1] - a[1]) * 110540;
      return Math.sqrt(dx * dx + dy * dy);
    }}

    function bearingDegrees(a, b) {{
      const meanLat = ((a[1] + b[1]) / 2) * Math.PI / 180;
      const dx = (b[0] - a[0]) * Math.cos(meanLat);
      const dy = b[1] - a[1];
      return Math.atan2(dx, dy) * 180 / Math.PI;
    }}

    function pointAlong(part, targetMeters) {{
      let walked = 0;
      for (let i = 1; i < part.length; i += 1) {{
        const prev = part[i - 1];
        const next = part[i];
        const length = segmentMeters(prev, next);
        if (length <= 0) continue;
        if (walked + length >= targetMeters) {{
          const ratio = (targetMeters - walked) / length;
          return {{
            point: [prev[0] + (next[0] - prev[0]) * ratio, prev[1] + (next[1] - prev[1]) * ratio],
            bearing: bearingDegrees(prev, next),
          }};
        }}
        walked += length;
      }}
      return null;
    }}

    function drawRouteLines(collection) {{
      const selected = Boolean(state.selectedDay);
      (collection.features || []).filter(matchesDay).forEach(feature => {{
        const color = feature.properties.color || "#2563eb";
        routeParts(feature).forEach(rawPart => {{
          const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
          if (part.length < 2) return;
          const latlngs = part.map(pt => [pt[1], pt[0]]);
          if (selected) {{
            L.polyline(latlngs, {{
              color: "#111827",
              weight: 11,
              opacity: 0.25,
              lineCap: "round",
              lineJoin: "round"
            }}).addTo(routeLayer);
            L.polyline(latlngs, {{
              color: "#ffffff",
              weight: 8,
              opacity: 0.92,
              lineCap: "round",
              lineJoin: "round"
            }}).addTo(routeLayer);
            L.polyline(latlngs, {{
              color: color,
              weight: 5,
              opacity: 1,
              lineCap: "round",
              lineJoin: "round"
            }}).bindPopup(popupHtml(feature.properties || {{}})).addTo(routeLayer);
          }} else {{
            L.polyline(latlngs, {{
              color: color,
              weight: 3,
              opacity: 0.38,
              lineCap: "round",
              lineJoin: "round"
            }}).bindPopup(popupHtml(feature.properties || {{}})).addTo(routeLayer);
          }}
        }});
      }});
    }}

    function drawDirectionArrows(collection) {{
      const selected = Boolean(state.selectedDay);
      (collection.features || []).filter(matchesDay).forEach(feature => {{
        const color = feature.properties.color || "#1f2937";
        routeParts(feature).forEach(rawPart => {{
          const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
          if (part.length < 2) return;
          let total = 0;
          for (let i = 1; i < part.length; i += 1) total += segmentMeters(part[i - 1], part[i]);
          if (total <= 0) return;
          const arrowCount = selected ? Math.max(1, Math.min(4, Math.floor(total / 1800))) : Math.max(1, Math.min(5, Math.floor(total / 1600)));
          const spacing = total / (arrowCount + 1);
          for (let i = 1; i <= arrowCount; i += 1) {{
            const placed = pointAlong(part, spacing * i);
            if (!placed) continue;
            L.marker([placed.point[1], placed.point[0]], {{
              interactive: false,
              keyboard: false,
              icon: L.divIcon({{
                className: "dir-arrow-wrap",
                iconSize: [18, 18],
                iconAnchor: [9, 9],
                html: `<div class="dir-arrow" style="--route-color:${{color}}; transform:rotate(${{placed.bearing}}deg);"></div>`
              }})
            }}).addTo(arrowLayer);
          }}
        }});
      }});
    }}

    function drawRouteCues(collection) {{
      (collection.features || []).filter(matchesDay).forEach(feature => {{
        routeParts(feature).forEach(rawPart => {{
          const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
          if (part.length < 2) return;
          for (let i = 1; i < part.length - 1; i += 1) {{
            if (coordKey(part[i - 1]) === coordKey(part[i + 1])) {{
              L.marker([part[i][1], part[i][0]], {{ icon: pathMarkerIcon("TURN", "turn") }}).addTo(arrowLayer);
            }}
          }}
        }});
      }});
    }}

    function renderLayers() {{
      officialLayer.clearLayers();
      routeLayer.clearLayers();
      arrowLayer.clearLayers();
      driveLayer.clearLayers();
      markerLayer.clearLayers();
      if (state.showFull) {{
        drawRouteLines(PLAN_MAP_DATA.feature_collections.on_foot_routes);
        drawDirectionArrows(PLAN_MAP_DATA.feature_collections.on_foot_routes);
        if (state.selectedDay) drawRouteCues(PLAN_MAP_DATA.feature_collections.on_foot_routes);
      }}
      if (state.showOfficial && !(state.selectedDay && state.showFull)) {{
        addGeoJson(PLAN_MAP_DATA.feature_collections.official_segments, officialLayer, feature => ({{
          color: feature.properties.color || "#2563eb",
          weight: 5,
          opacity: 0.95
        }}));
      }}
      if (state.showDrives) {{
        addGeoJson(PLAN_MAP_DATA.feature_collections.drives, driveLayer, feature => ({{
          color: "#6b7280",
          weight: 3,
          opacity: 0.65,
          dashArray: "7 7"
        }}));
      }}
      addGeoJson(
        PLAN_MAP_DATA.feature_collections.trailheads,
        markerLayer,
        null,
        (feature, latlng) => L.marker(latlng, {{ icon: parkingIcon() }})
      );
      fitVisible();
    }}

    function visibleLayers() {{
      const layers = [];
      [officialLayer, routeLayer, arrowLayer, driveLayer, markerLayer].forEach(group => {{
        group.eachLayer(layer => layers.push(layer));
      }});
      return layers;
    }}

    function invalidateMapSize() {{
      map.invalidateSize(false);
    }}

    function fitVisible() {{
      invalidateMapSize();
      const group = L.featureGroup(visibleLayers());
      if (group.getLayers().length) {{
        map.fitBounds(group.getBounds(), {{ padding: [28, 28], maxZoom: state.selectedDay ? 14 : 11 }});
      }} else {{
        map.setView([43.63, -116.2], 11);
      }}
    }}

    function renderMetrics() {{
      const summary = PLAN_MAP_DATA.summary || {{}};
      const audit = PLAN_MAP_DATA.audit || {{}};
      const metrics = [
        ["Segments", `${{summary.scheduled_segments || "n/a"}}`],
        ["Official miles", `${{summary.scheduled_official_miles || "n/a"}}`],
        ["On-foot miles", `${{summary.scheduled_total_on_foot_miles || "n/a"}}`],
        ["Ascent", `${{summary.scheduled_ascent_ft || "n/a"}} ft`],
        ["Days", `${{summary.scheduled_days || PLAN_MAP_DATA.days.length}}`],
        ["Validation", audit.execution_validation_passed ? "passed" : "check"]
      ];
      document.getElementById("metrics").innerHTML = metrics.map(([label, value]) =>
        `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`
      ).join("");
    }}

    function renderDays() {{
      const list = document.getElementById("dayList");
      list.innerHTML = PLAN_MAP_DATA.days.map(day => {{
        const active = state.selectedDay === day.date ? " active" : "";
        const exception = day.requires_normal_cap_exception
          ? `<span class="exception">+${{day.normal_cap_exception_minutes}} min</span>`
          : "";
        const outings = day.outings.map(outing => `
          <div class="outing">
            <strong>${{outing.title}}</strong>
            <span>${{outing.route_label || "route"}} · ${{fmt(outing.official_miles, " official mi")}} · ${{fmt(outing.on_foot_miles, " total mi")}} · ${{fmt(outing.ascent_ft, " ft")}}${{outing.map_gap_warning_count ? " · map gaps " + outing.map_gap_warning_count : ""}}</span>
          </div>
        `).join("");
        return `
          <article class="day${{active}}" style="border-left-color: ${{day.color}}">
            <button type="button" data-day="${{day.date}}">
              <div class="day-title"><span>${{day.date}}</span><span>${{fmt(day.realistic_total_minutes, " min")}}</span></div>
              <div class="day-meta">${{fmt(day.official_miles, " official mi")}} · ${{fmt(day.on_foot_miles, " total mi")}} · ${{fmt(day.ascent_ft, " ft ascent")}} ${{exception}}</div>
            </button>
            <div class="outings">${{outings}}</div>
          </article>
        `;
      }}).join("");
      list.querySelectorAll("button[data-day]").forEach(button => {{
        button.addEventListener("click", () => {{
          state.selectedDay = button.dataset.day;
          document.getElementById("showAll").classList.remove("active");
          renderDays();
          renderLayers();
        }});
      }});
    }}

    document.getElementById("showAll").addEventListener("click", event => {{
      state.selectedDay = null;
      event.currentTarget.classList.add("active");
      renderDays();
      renderLayers();
    }});
    document.getElementById("officialToggle").addEventListener("change", event => {{
      state.showOfficial = event.currentTarget.checked;
      renderLayers();
    }});
    document.getElementById("fullRouteToggle").addEventListener("change", event => {{
      state.showFull = event.currentTarget.checked;
      renderLayers();
    }});
    document.getElementById("driveToggle").addEventListener("change", event => {{
      state.showDrives = event.currentTarget.checked;
      renderLayers();
    }});

    renderMetrics();
    renderDays();
    renderLayers();
    window.addEventListener("resize", () => invalidateMapSize());
    if ("ResizeObserver" in window) {{
      let resizeQueued = false;
      const resizeObserver = new ResizeObserver(() => {{
        if (resizeQueued) return;
        resizeQueued = true;
        requestAnimationFrame(() => {{
          resizeQueued = false;
          invalidateMapSize();
        }});
      }});
      resizeObserver.observe(document.querySelector("main"));
    }}
    requestAnimationFrame(() => {{
      invalidateMapSize();
      fitVisible();
    }});
    setTimeout(() => {{
      invalidateMapSize();
      fitVisible();
    }}, 250);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runbook-json", type=Path, default=DEFAULT_RUNBOOK_JSON)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_OUTPUT_DATA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runbook = read_json(args.runbook_json)
    plan = read_json(args.plan_json)
    official_index = load_official_segment_index(args.official_geojson)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_meta = ((plan.get("source_datasets") or {}).get("connector_geojson") or {})
    connector_path = Path(str(connector_meta.get("path"))) if connector_meta.get("path") else None
    connector_graph = (
        load_connector_graph(connector_path, official_segments=official_segments)
        if connector_path and connector_path.exists()
        else None
    )
    map_data = build_plan_map_data(
        runbook,
        plan,
        official_index,
        connector_graph=connector_graph,
    )

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_data.write_text(json.dumps(map_data, indent=2) + "\n", encoding="utf-8")
    args.output_html.write_text(render_html(map_data), encoding="utf-8")

    manifest_path = args.output_html.with_name(f"{args.output_html.stem}-artifact-manifest.json")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=str(runbook.get("run_id") or plan.get("run_id") or "plan-map"),
            inputs=[args.runbook_json, args.plan_json, args.official_geojson],
            outputs=[args.output_html, args.output_data],
            command="export_plan_map.py",
            metadata={
                "profile_name": runbook.get("profile_name"),
                "rendered_route_geometry_validation_passed": map_data["map_validation"][
                    "rendered_route_geometry_validation_passed"
                ],
                "rendered_failed_route_count": map_data["map_validation"][
                    "rendered_failed_route_count"
                ],
                "source_route_gap_warning_count": map_data["map_validation"][
                    "source_route_gap_warning_count"
                ],
            },
        ),
    )
    print(f"Wrote {args.output_html}")
    print(f"Wrote {args.output_data}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
