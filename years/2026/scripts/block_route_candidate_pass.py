#!/usr/bin/env python3
"""Select a graph-validated block route pass from existing route-menu candidates."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


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
from route_block_planner import build_block_index, normalize_trail_name  # noqa: E402


DEFAULT_BLOCKS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-blocks-v1.json"
DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "block-route-candidate-pass-v1"
PALETTE = [
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
]


def candidate_cost(candidate: dict[str, Any], route_count_weight: float) -> float:
    official = float(candidate.get("official_new_miles") or 0.0)
    on_foot = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
    ratio = on_foot / official if official else 99.0
    small_penalty = 8.0 if official < 2.0 else 0.0
    tiny_penalty = 12.0 if official < 1.0 else 0.0
    ratio_penalty = max(0.0, ratio - 1.8) * 5.0
    status_penalty = 10000.0 if candidate.get("route_status") != "graph_validated" else 0.0
    return route_count_weight + on_foot + small_penalty + tiny_penalty + ratio_penalty + status_penalty


def select_candidates(
    candidates: list[dict[str, Any]],
    route_count_weight: float = 40.0,
    time_limit_seconds: int = 60,
) -> list[dict[str, Any]]:
    graph_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("route_status") == "graph_validated"
        and candidate.get("segment_ids")
    ]
    segment_ids = sorted(
        {int(segment_id) for candidate in graph_candidates for segment_id in candidate.get("segment_ids") or []}
    )
    segment_index = {segment_id: index for index, segment_id in enumerate(segment_ids)}
    coverage = np.zeros((len(segment_ids), len(graph_candidates)))
    for candidate_index, candidate in enumerate(graph_candidates):
        for segment_id in candidate.get("segment_ids") or []:
            coverage[segment_index[int(segment_id)], candidate_index] = 1.0
    costs = np.array([candidate_cost(candidate, route_count_weight) for candidate in graph_candidates])
    result = milp(
        c=costs,
        constraints=LinearConstraint(
            coverage,
            lb=np.ones(len(segment_ids)),
            ub=np.full(len(segment_ids), np.inf),
        ),
        bounds=Bounds(0, 1),
        integrality=np.ones(len(graph_candidates)),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.01},
    )
    if not result.success:
        raise RuntimeError(f"Candidate set cover failed: {result.message}")
    return [candidate for candidate, value in zip(graph_candidates, result.x) if value > 0.5]


def block_for_candidate(
    candidate: dict[str, Any],
    trail_to_block: dict[str, dict[str, Any]],
) -> str:
    block_miles: dict[str, float] = {}
    for segment in candidate.get("segments") or []:
        block = trail_to_block.get(normalize_trail_name(str(segment.get("trail_name") or "")))
        block_id = str(block["block_id"]) if block else "unassigned"
        block_miles[block_id] = block_miles.get(block_id, 0.0) + float(
            segment.get("official_miles") or 0.0
        )
    if block_miles:
        return max(block_miles.items(), key=lambda item: (item[1], item[0]))[0]

    block_counts: dict[str, int] = {}
    for trail_name in candidate.get("trail_names") or []:
        block = trail_to_block.get(normalize_trail_name(str(trail_name)))
        block_id = str(block["block_id"]) if block else "unassigned"
        block_counts[block_id] = block_counts.get(block_id, 0) + 1
    if not block_counts:
        return "unassigned"
    return max(block_counts.items(), key=lambda item: (item[1], item[0]))[0]


def summarize_selection(
    selected: list[dict[str, Any]],
    blocks_config: dict[str, Any],
) -> dict[str, Any]:
    trail_to_block, duplicates = build_block_index(blocks_config)
    block_names = {str(block["block_id"]): block.get("name") for block in blocks_config.get("blocks") or []}
    unique_segments = {int(segment_id) for candidate in selected for segment_id in candidate.get("segment_ids") or []}
    block_rows: dict[str, dict[str, Any]] = {}
    route_rows = []
    for index, candidate in enumerate(sorted(selected, key=lambda item: (-float(item.get("official_new_miles") or 0), str(item.get("candidate_id")))), start=1):
        block_id = block_for_candidate(candidate, trail_to_block)
        official = float(candidate.get("official_new_miles") or 0.0)
        on_foot = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
        route = {
            "route_number": index,
            "candidate_id": candidate.get("candidate_id"),
            "block_id": block_id,
            "block_name": block_names.get(block_id, block_id),
            "trail_names": candidate.get("trail_names") or [],
            "official_miles": round(official, 2),
            "on_foot_miles": round(on_foot, 2),
            "ratio": round(on_foot / official, 2) if official else None,
            "total_minutes": candidate.get("total_minutes"),
            "trailhead": (candidate.get("trailhead") or {}).get("name"),
            "route_status": candidate.get("route_status"),
            "less_optimal_flags": candidate.get("less_optimal_flags") or [],
            "segment_ids": candidate.get("segment_ids") or [],
        }
        route_rows.append(route)
        block = block_rows.setdefault(
            block_id,
            {
                "block_id": block_id,
                "block_name": block_names.get(block_id, block_id),
                "route_count": 0,
                "official_miles_raw_sum": 0.0,
                "on_foot_miles": 0.0,
                "segment_ids": set(),
            },
        )
        block["route_count"] += 1
        block["official_miles_raw_sum"] += official
        block["on_foot_miles"] += on_foot
        block["segment_ids"].update(int(segment_id) for segment_id in candidate.get("segment_ids") or [])
    for block in block_rows.values():
        block["official_miles_raw_sum"] = round(block["official_miles_raw_sum"], 2)
        block["on_foot_miles"] = round(block["on_foot_miles"], 2)
        block["segment_count"] = len(block["segment_ids"])
        block["segment_ids"] = sorted(block["segment_ids"])

    total_on_foot = sum(float(candidate.get("estimated_total_on_foot_miles") or 0.0) for candidate in selected)
    return {
        "planning_status": "candidate_route_pass_from_existing_graph_menu",
        "summary": {
            "selected_route_count": len(selected),
            "covered_segment_count": len(unique_segments),
            "total_on_foot_miles": round(total_on_foot, 2),
            "planwide_on_foot_to_official_ratio": round(total_on_foot / 164.43, 2),
            "routes_under_1_official_mile": sum(1 for candidate in selected if float(candidate.get("official_new_miles") or 0) < 1),
            "routes_under_2_official_miles": sum(1 for candidate in selected if float(candidate.get("official_new_miles") or 0) < 2),
            "duplicate_configured_trail_count": len(duplicates),
        },
        "routes": route_rows,
        "blocks": sorted(block_rows.values(), key=lambda item: item["block_name"]),
        "caveats": [
            "This is the best route pass selected from the existing graph-validated route-menu candidates.",
            "It is usable as an evidence-backed candidate baseline, but it is not yet the final normal-human loop plan because several blocks still require custom GPX route design.",
            "The selected route count and remaining small outings show the current candidate universe is too fragmented for the final objective.",
        ],
    }


def line_feature(coords: list[tuple[float, float]], properties: dict[str, Any]) -> dict[str, Any] | None:
    if len(coords) < 2:
        return None
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in coords],
        },
        "properties": properties,
    }


def point_feature(lon: float, lat: float, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [round(float(lon), 6), round(float(lat), 6)],
        },
        "properties": properties,
    }


def parking_feature(candidate: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any] | None:
    trailhead = candidate.get("trailhead") or {}
    if trailhead.get("lon") is None or trailhead.get("lat") is None:
        return None
    name = trailhead.get("name") or properties.get("trailhead") or "Trailhead"
    return point_feature(
        float(trailhead["lon"]),
        float(trailhead["lat"]),
        {
            **properties,
            "kind": "parking",
            "name": name,
            "trailhead": name,
            "has_parking": trailhead.get("has_parking"),
            "has_restroom": trailhead.get("has_restroom"),
            "has_water": trailhead.get("has_water"),
            "water_confidence": trailhead.get("water_confidence"),
            "parking_minutes": trailhead.get("parking_minutes"),
            "source": trailhead.get("source"),
        },
    )


def split_coords_on_gaps(coords: list[tuple[float, float]], max_gap_miles: float = 0.1) -> list[list[tuple[float, float]]]:
    if not coords:
        return []
    parts = [[coords[0]]]
    for left, right in zip(coords, coords[1:]):
        if haversine_miles(left, right) > max_gap_miles:
            parts.append([right])
        else:
            parts[-1].append(right)
    return [part for part in parts if len(part) >= 2]


def multiline_feature(parts: list[list[tuple[float, float]]], properties: dict[str, Any]) -> dict[str, Any] | None:
    usable = [part for part in parts if len(part) >= 2]
    if not usable:
        return None
    if len(usable) == 1:
        return line_feature(usable[0], properties)
    return {
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [
                [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in part]
                for part in usable
            ],
        },
        "properties": properties,
    }


def build_map_data(
    route_pass: dict[str, Any],
    plan: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_index = load_candidate_index(plan)
    route_features = []
    official_features = []
    parking_features = []
    validations = []
    for route in route_pass.get("routes") or []:
        candidate = candidate_index[str(route["candidate_id"])]
        color = PALETTE[(int(route["route_number"]) - 1) % len(PALETTE)]
        coords = candidate_track_coordinates(candidate, official_index, connector_graph=connector_graph)
        source_validation = validate_track_segments([coords], max_gap_miles=0.1)
        rendered_parts = split_coords_on_gaps(coords, max_gap_miles=0.1)
        render_validation = validate_track_segments(rendered_parts, max_gap_miles=0.1)
        props = {
            "kind": "route",
            "route_number": route["route_number"],
            "candidate_id": route["candidate_id"],
            "block_name": route["block_name"],
            "title": ", ".join(route["trail_names"]),
            "official_miles": route["official_miles"],
            "on_foot_miles": route["on_foot_miles"],
            "trailhead": route["trailhead"],
            "color": color,
            "source_gap_warning_count": len(source_validation["failures"]),
        }
        route_feature = multiline_feature(rendered_parts, props)
        if route_feature:
            route_features.append(route_feature)
        parking = parking_feature(candidate, props)
        if parking:
            parking_features.append(parking)
        for segment in candidate.get("segments") or []:
            seg_coords = candidate_segment_coordinates(candidate, segment, official_index)
            feature = line_feature(
                seg_coords,
                {
                    **props,
                    "kind": "official_segment",
                    "seg_id": segment.get("seg_id"),
                    "segment_name": segment.get("seg_name") or segment.get("trail_name"),
                },
            )
            if feature:
                official_features.append(feature)
        validations.append(
            {
                "candidate_id": route["candidate_id"],
                "source_gap_warning": not source_validation["passed"],
                "source_max_gap_miles": source_validation["max_trackpoint_gap_miles"],
                "rendered_passed": render_validation["passed"],
                "rendered_failures": render_validation["failures"],
            }
        )
    return {
        "summary": route_pass["summary"],
        "routes": route_pass["routes"],
        "feature_collections": {
            "routes": {"type": "FeatureCollection", "features": route_features},
            "official_segments": {"type": "FeatureCollection", "features": official_features},
            "parking": {"type": "FeatureCollection", "features": parking_features},
        },
        "map_validation": {
            "rendered_passed": all(item["rendered_passed"] for item in validations),
            "source_gap_warning_count": sum(1 for item in validations if item["source_gap_warning"]),
            "route_validations": validations,
        },
    }


def render_html(map_data: dict[str, Any]) -> str:
    payload = json.dumps(map_data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2026 Block Route Candidate Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#1f2933; }}
    .app {{ display:grid; grid-template-columns: 420px minmax(0,1fr); min-height:100vh; }}
    aside {{ overflow:auto; border-right:1px solid #d7ddd4; background:#fff; }}
    header {{ padding:16px; border-bottom:1px solid #d7ddd4; }}
    h1 {{ margin:0 0 6px; font-size:20px; letter-spacing:0; }}
    p {{ margin:0; color:#667085; font-size:13px; line-height:1.45; }}
    .metrics {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; padding:12px 16px; border-bottom:1px solid #d7ddd4; }}
    .metric {{ border:1px solid #d7ddd4; border-radius:6px; padding:8px; }}
    .metric span {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:3px; font-size:16px; }}
    .route {{ border-bottom:1px solid #e5e7eb; padding:10px 14px; cursor:pointer; }}
    .route.active {{ background:#eef6ff; border-left:4px solid #2563eb; padding-left:10px; }}
    .route strong {{ display:block; font-size:13px; line-height:1.35; }}
    .route span {{ color:#667085; font-size:12px; line-height:1.35; }}
    .selected-panel {{ display:none; margin:12px 16px; border:2px solid #111827; border-radius:6px; padding:10px; background:#fff; }}
    .selected-panel.active {{ display:block; }}
    .selected-panel span {{ display:block; color:#667085; font-size:11px; font-weight:700; text-transform:uppercase; }}
    .selected-panel strong {{ display:block; margin-top:3px; font-size:15px; line-height:1.25; }}
    .selected-panel p {{ margin-top:6px; color:#344054; font-size:12px; }}
    .legend {{ margin:0 16px 12px; border:1px solid #d7ddd4; border-radius:6px; padding:8px 9px; color:#475467; font-size:12px; line-height:1.4; }}
    .legend-arrow {{ display:inline-block; width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-bottom:12px solid #1f2937; vertical-align:-1px; margin-right:6px; transform:rotate(90deg); }}
    .dir-arrow-wrap {{ background:transparent; border:0; }}
    .dir-arrow {{ width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:14px solid var(--route-color,#1f2937); filter:drop-shadow(0 0 2px #fff) drop-shadow(0 0 2px #fff); transform-origin:50% 50%; }}
    .parking-marker-wrap {{ background:transparent; border:0; }}
    .parking-marker {{ width:22px; height:22px; border-radius:50%; background:#111827; color:#fff; border:2px solid #fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; box-shadow:0 2px 8px rgba(15,23,42,.32); }}
    .path-marker-wrap {{ background:transparent; border:0; }}
    .path-marker {{ min-width:24px; height:24px; border-radius:999px; background:#fff; color:#111827; border:2px solid #111827; display:flex; align-items:center; justify-content:center; padding:0 6px; font-weight:800; font-size:11px; box-shadow:0 2px 8px rgba(15,23,42,.25); }}
    .path-marker.turn {{ background:#fef3c7; }}
    .map-summary {{ display:none; position:absolute; top:16px; right:16px; z-index:500; max-width:360px; border:2px solid #111827; border-radius:6px; padding:10px 12px; background:rgba(255,255,255,.96); box-shadow:0 10px 30px rgba(15,23,42,.18); }}
    .map-summary.active {{ display:block; }}
    .map-summary span {{ display:block; color:#667085; font-size:11px; font-weight:700; text-transform:uppercase; }}
    .map-summary strong {{ display:block; margin-top:2px; font-size:15px; line-height:1.25; }}
    .map-summary p {{ margin-top:6px; color:#344054; font-size:12px; }}
    .leaflet-tooltip.parking-label {{ background:#111827; color:#fff; border:0; border-radius:4px; padding:4px 6px; box-shadow:0 2px 8px rgba(15,23,42,.28); font-size:12px; font-weight:700; }}
    .leaflet-tooltip.parking-label::before {{ display:none; }}
    #map {{ min-height:100vh; }}
    button {{ margin:12px 16px; min-height:34px; border:1px solid #d7ddd4; background:#fff; border-radius:6px; padding:7px 9px; cursor:pointer; }}
    @media (max-width:860px) {{ .app {{ grid-template-columns:1fr; }} #map {{ min-height:55vh; }} }}
  </style>
</head>
<body>
<div class="app">
  <aside>
    <header>
      <h1>Block Route Candidate Pass</h1>
      <p>Existing graph-validated route candidates selected for full coverage. This is a baseline map before custom GPX block design.</p>
    </header>
    <div class="metrics" id="metrics"></div>
    <div class="selected-panel" id="selectedPanel"></div>
    <div class="legend"><span class="legend-arrow"></span>Selected routes show one clear cased line, directional arrows, and turn markers for double-backs. <strong>P</strong> markers show where to park/start.</div>
    <button id="fitAll" type="button">Fit all</button>
    <div id="routes"></div>
  </aside>
  <main><div id="map"><div class="map-summary" id="mapSummary"></div></div></main>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = {payload};
const map = L.map("map", {{ preferCanvas:true }});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{ maxZoom:19, attribution:"&copy; OpenStreetMap contributors" }}).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const officialLayer = L.layerGroup().addTo(map);
const parkingLayer = L.layerGroup().addTo(map);
const arrowLayer = L.layerGroup().addTo(map);
function fmt(v,s="") {{ return v === null || v === undefined ? "n/a" : `${{v}}${{s}}`; }}
function esc(value) {{
  return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function popup(props) {{
  if (props.kind === "parking") {{
    return `<strong>Park: ${{props.name || props.trailhead}}</strong><br>${{props.block_name || ""}}<br>Parking: ${{props.has_parking === false ? "unknown" : "yes"}}<br>Prep: ${{fmt(props.parking_minutes," min")}}`;
  }}
  return `<strong>${{props.title || props.segment_name}}</strong><br>${{props.block_name || ""}}<br>${{fmt(props.official_miles," official mi")}} · ${{fmt(props.on_foot_miles," total mi")}}`;
}}
function parkingIcon() {{
  return L.divIcon({{
    className:"parking-marker-wrap",
    iconSize:[26,26],
    iconAnchor:[13,13],
    html:'<div class="parking-marker">P</div>'
  }});
}}
function pathMarkerIcon(label, extraClass="") {{
  return L.divIcon({{
    className:"path-marker-wrap",
    iconSize:[32,26],
    iconAnchor:[16,13],
    html:`<div class="path-marker ${{extraClass}}">${{esc(label)}}</div>`
  }});
}}
function routeById(id) {{
  return (DATA.routes || []).find(route => String(route.candidate_id) === String(id));
}}
function parkingNames(filterId) {{
  const features = ((DATA.feature_collections.parking || {{ features:[] }}).features || [])
    .filter(feature => !filterId || String(feature.properties.candidate_id) === String(filterId));
  return [...new Set(features.map(feature => feature.properties.name || feature.properties.trailhead).filter(Boolean))];
}}
function shortParkingName(name) {{
  return String(name || "Park here").replace(/ Parking\\/Trailhead| Trailhead| Trail Access Point/g, "");
}}
function selectedHtml(route) {{
  if (!route) return "";
  const parks = parkingNames(route.candidate_id);
  const parkText = parks.length ? parks.map(shortParkingName).join(" + ") : (route.trailhead || "parking TBD");
  return `<span>Selected route</span><strong>${{esc(route.route_number)}}. ${{esc(route.trail_names.join(", "))}}</strong><p><b>Park/start:</b> ${{esc(parkText)}}<br><b>Block:</b> ${{esc(route.block_name)}}<br><b>Run:</b> ${{esc(route.official_miles)}} official mi · ${{esc(route.on_foot_miles)}} total mi</p>`;
}}
function updateSelection(filterId) {{
  const route = filterId ? routeById(filterId) : null;
  const panel = document.getElementById("selectedPanel");
  const summary = document.getElementById("mapSummary");
  panel.innerHTML = route ? selectedHtml(route) : "";
  summary.innerHTML = route ? selectedHtml(route) : "";
  panel.classList.toggle("active", Boolean(route));
  summary.classList.toggle("active", Boolean(route));
  document.querySelectorAll(".route").forEach(el => el.classList.toggle("active", Boolean(filterId) && String(el.dataset.id) === String(filterId)));
}}
function routeParts(feature) {{
  const geom = feature.geometry || {{}};
  if (geom.type === "LineString") return [geom.coordinates || []];
  if (geom.type === "MultiLineString") return geom.coordinates || [];
  return [];
}}
const OUT_AND_BACK_OFFSET_METERS = 10;
function coordKey(pt) {{ return `${{Number(pt[0]).toFixed(5)}},${{Number(pt[1]).toFixed(5)}}`; }}
function orientedSegmentKey(a,b) {{ return `${{coordKey(a)}}>${{coordKey(b)}}`; }}
function segmentKey(a,b) {{
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
function normalMeters(a,b) {{
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
function segmentMeters(a,b) {{
  const meanLat = ((a[1] + b[1]) / 2) * Math.PI / 180;
  const dx = (b[0] - a[0]) * 111320 * Math.cos(meanLat);
  const dy = (b[1] - a[1]) * 110540;
  return Math.sqrt(dx * dx + dy * dy);
}}
function bearingDegrees(a,b) {{
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
function drawRouteLines(collection, filterFn, selected=false) {{
  (collection.features || []).filter(filterFn).forEach(feature => {{
    const color = feature.properties.color || "#1f2937";
    routeParts(feature).forEach(rawPart => {{
      const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
      if (part.length < 2) return;
      const latlngs = part.map(pt => [pt[1], pt[0]]);
      if (selected) {{
        L.polyline(latlngs, {{ color:"#111827", weight:11, opacity:.25, lineCap:"round", lineJoin:"round" }}).addTo(routeLayer);
        L.polyline(latlngs, {{ color:"#ffffff", weight:8, opacity:.92, lineCap:"round", lineJoin:"round" }}).addTo(routeLayer);
        L.polyline(latlngs, {{ color:color, weight:5, opacity:1, lineCap:"round", lineJoin:"round" }}).bindPopup(popup(feature.properties || {{}})).addTo(routeLayer);
      }} else {{
        L.polyline(latlngs, {{ color:color, weight:4, opacity:.72, lineCap:"round", lineJoin:"round" }}).bindPopup(popup(feature.properties || {{}})).addTo(routeLayer);
      }}
    }});
  }});
}}
function drawDirectionArrows(collection, filterFn, selected=false) {{
  (collection.features || []).filter(filterFn).forEach(feature => {{
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
          interactive:false,
          keyboard:false,
          icon:L.divIcon({{
            className:"dir-arrow-wrap",
            iconSize:[18,18],
            iconAnchor:[9,9],
            html:`<div class="dir-arrow" style="--route-color:${{color}}; transform:rotate(${{placed.bearing}}deg);"></div>`
          }})
        }}).addTo(arrowLayer);
      }}
    }});
  }});
}}
function drawRouteCues(collection, filterFn) {{
  (collection.features || []).filter(filterFn).forEach(feature => {{
    routeParts(feature).forEach(rawPart => {{
      const part = (rawPart || []).filter(pt => Array.isArray(pt) && pt.length >= 2);
      if (part.length < 2) return;
      for (let i = 1; i < part.length - 1; i += 1) {{
        if (coordKey(part[i - 1]) === coordKey(part[i + 1])) {{
          L.marker([part[i][1], part[i][0]], {{ icon:pathMarkerIcon("TURN", "turn") }}).addTo(arrowLayer);
        }}
      }}
    }});
  }});
}}
function draw(filterId=null) {{
  const selected = Boolean(filterId);
  routeLayer.clearLayers(); officialLayer.clearLayers(); parkingLayer.clearLayers(); arrowLayer.clearLayers();
  drawRouteLines(DATA.feature_collections.routes, f => !filterId || f.properties.candidate_id === filterId, selected);
  if (!selected) {{
    L.geoJSON(DATA.feature_collections.official_segments, {{
      filter:f => !filterId || f.properties.candidate_id === filterId,
      style:f => ({{ color:f.properties.color, weight:6, opacity:.9 }}),
      onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))
    }}).addTo(officialLayer);
  }}
  L.geoJSON(DATA.feature_collections.parking || {{ type:"FeatureCollection", features:[] }}, {{
    filter:f => !filterId || f.properties.candidate_id === filterId,
    pointToLayer:(f,latlng)=>L.marker(latlng, {{ icon:parkingIcon() }}),
    onEachFeature:(f,l)=>{{
      l.bindPopup(popup(f.properties));
      if (filterId) l.bindTooltip(shortParkingName(f.properties.name || f.properties.trailhead), {{ permanent:true, direction:"top", offset:[0,-15], className:"parking-label" }});
    }}
  }}).addTo(parkingLayer);
  drawDirectionArrows(DATA.feature_collections.routes, f => !filterId || f.properties.candidate_id === filterId, selected);
  if (selected) drawRouteCues(DATA.feature_collections.routes, f => f.properties.candidate_id === filterId);
  updateSelection(filterId);
  fit();
}}
function fit() {{
  const layers=[]; routeLayer.eachLayer(l=>layers.push(l)); officialLayer.eachLayer(l=>layers.push(l)); parkingLayer.eachLayer(l=>layers.push(l));
  if (layers.length) map.fitBounds(L.featureGroup(layers).getBounds(), {{ padding:[24,24], maxZoom:14 }});
}}
document.getElementById("metrics").innerHTML = [
  ["Routes", DATA.summary.selected_route_count],
  ["Segments", DATA.summary.covered_segment_count],
  ["On-foot", `${{DATA.summary.total_on_foot_miles}} mi`],
  ["Ratio", `${{DATA.summary.planwide_on_foot_to_official_ratio}}x`]
].map(([a,b])=>`<div class="metric"><span>${{a}}</span><strong>${{b}}</strong></div>`).join("");
document.getElementById("routes").innerHTML = DATA.routes.map(r => `<div class="route" data-id="${{r.candidate_id}}"><strong>${{r.route_number}}. ${{r.trail_names.join(", ")}}</strong><span>${{r.block_name}} · ${{r.official_miles}} official · ${{r.on_foot_miles}} total · ${{r.trailhead || ""}}</span></div>`).join("");
document.querySelectorAll(".route").forEach(el => el.addEventListener("click", () => draw(el.dataset.id)));
document.getElementById("fitAll").addEventListener("click", () => draw());
draw();
</script>
</body>
</html>
"""


def render_markdown(route_pass: dict[str, Any]) -> str:
    summary = route_pass["summary"]
    lines = [
        "# 2026 Block Route Candidate Pass v1",
        "",
        "Status: existing graph-candidate baseline, not yet the final normal-human loop plan.",
        "",
        "## Summary",
        "",
        f"- Selected routes: {summary['selected_route_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / 251",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- Planwide on-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Routes under 1 official mile: {summary['routes_under_1_official_mile']}",
        f"- Routes under 2 official miles: {summary['routes_under_2_official_miles']}",
        "",
        "## Caveat",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in route_pass.get("caveats") or [])
    lines.extend(
        [
            "",
            "## Routes",
            "",
            "| # | Block | Route | Trailhead | Official mi | On-foot mi | Ratio | Minutes | Flags |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for route in route_pass.get("routes") or []:
        flags = ", ".join(route.get("less_optimal_flags") or [])
        lines.append(
            f"| {route['route_number']} | {route['block_name']} | {', '.join(route['trail_names'])} | "
            f"{route.get('trailhead') or ''} | {route['official_miles']} | {route['on_foot_miles']} | "
            f"{route['ratio']} | {route.get('total_minutes') or ''} | {flags} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--blocks-json", type=Path, default=DEFAULT_BLOCKS_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--route-count-weight", type=float, default=40.0)
    parser.add_argument("--optimizer-time-limit-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.plan_json)
    blocks_config = read_json(args.blocks_json)
    selected = select_candidates(
        plan["route_menu"]["all_candidates"],
        route_count_weight=args.route_count_weight,
        time_limit_seconds=args.optimizer_time_limit_seconds,
    )
    route_pass = summarize_selection(selected, blocks_config)
    official_index = load_official_segment_index(args.official_geojson)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_meta = ((plan.get("source_datasets") or {}).get("connector_geojson") or {})
    connector_path = Path(str(connector_meta.get("path"))) if connector_meta.get("path") else None
    connector_graph = (
        load_connector_graph(connector_path, official_segments=official_segments)
        if connector_path and connector_path.exists()
        else None
    )
    map_data = build_map_data(route_pass, plan, official_index, connector_graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_json_path = args.output_dir / f"{args.basename}-map-data.json"
    map_html_path = args.output_dir / f"{args.basename}-map.html"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, route_pass)
    md_path.write_text(render_markdown(route_pass), encoding="utf-8")
    write_json(map_json_path, map_data)
    map_html_path.write_text(render_html(map_data), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=str(plan.get("run_id") or args.basename),
            inputs=[args.plan_json, args.blocks_json, args.official_geojson],
            outputs=[json_path, md_path, map_json_path, map_html_path],
            command="block_route_candidate_pass.py",
            metadata={
                "selected_route_count": route_pass["summary"]["selected_route_count"],
                "covered_segment_count": route_pass["summary"]["covered_segment_count"],
                "rendered_map_passed": map_data["map_validation"]["rendered_passed"],
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {map_html_path}")
    print(f"Wrote {map_json_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
