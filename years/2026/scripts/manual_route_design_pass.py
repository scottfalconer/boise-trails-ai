#!/usr/bin/env python3
"""Write manual route-design reports for held route areas."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from export_execution_gpx import (  # noqa: E402
    candidate_track_coordinates,
    load_official_segment_index,
    render_gpx_segments,
    validate_track_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    build_performance_profile,
    candidate_from_trail_group,
    group_remaining_by_trail,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    read_json,
    round_miles,
)


DEFAULT_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1.json"
DEFAULT_MANUAL_DESIGN_JSON = YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json"
DEFAULT_STATE_PATH = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "package16-manual-route-design-v1"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "route"


def target_range_text(value: list[Any] | None) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "n/a"
    return f"{round_miles(float(value[0]))}-{round_miles(float(value[1]))}"


def find_package(package_pass: dict[str, Any], area: dict[str, Any]) -> dict[str, Any] | None:
    for package in package_pass.get("packages") or []:
        if area.get("package_number") is not None and str(package.get("package_number")) == str(area["package_number"]):
            return package
        if area.get("block_id") and str(package.get("block_id")) == str(area["block_id"]):
            return package
    return None


def components_for_ids(package: dict[str, Any] | None, candidate_ids: list[str]) -> list[dict[str, Any]]:
    if not package:
        return []
    wanted = {str(candidate_id) for candidate_id in candidate_ids}
    return [
        component
        for component in package.get("components") or []
        if str(component.get("candidate_id")) in wanted
    ]


def forced_anchor_state(base_state: dict[str, Any], area: dict[str, Any], alternative: dict[str, Any]) -> dict[str, Any] | None:
    anchor_id = alternative.get("start_anchor_id")
    if not anchor_id:
        return None
    anchor = next((item for item in area.get("anchors") or [] if item.get("anchor_id") == anchor_id), None)
    if not anchor:
        return None
    state = dict(base_state)
    state["trailheads"] = [
        {
            "name": anchor.get("name"),
            "lat": anchor.get("lat"),
            "lon": anchor.get("lon"),
            "has_parking": bool(anchor.get("has_parking")),
            "parking_confidence": anchor.get("parking_confidence") or "manual_required",
            "parking_minutes": base_state.get("parking_minutes") or 8,
            "source": anchor.get("source") or "manual_route_design_anchor",
        }
    ]
    return state


def candidate_probe_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_status": candidate.get("route_status"),
        "official_miles": candidate.get("official_new_miles"),
        "on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
        "total_minutes": candidate.get("total_minutes"),
        "official_repeat_miles": candidate.get("official_repeat_miles"),
        "connector_miles": candidate.get("connector_miles"),
        "road_miles": candidate.get("road_miles"),
        "ascent_direction_passed": (candidate.get("validation") or {}).get("ascent_direction_passed"),
        "return_path_graph_validated": (candidate.get("validation") or {}).get("return_path_graph_validated"),
        "trailhead_snap_confidence": (candidate.get("validation") or {}).get("trailhead_snap_confidence"),
        "less_optimal_flags": candidate.get("less_optimal_flags") or [],
        "planned_traversal_direction": (candidate.get("direction_validation") or {}).get("planned_traversal_direction") or {},
        "trailhead": (candidate.get("trailhead") or {}).get("name"),
        "route_is_probe_not_field_ready": True,
    }


def build_probe_candidate_objects(
    manual_design: dict[str, Any],
    state_json: Path,
    official_geojson: Path,
    connector_geojson: Path,
    dem_tif: Path,
    dem_summary_json: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base_state = load_state(state_json)
    official_segments, _official_meta = load_official_segments(official_geojson)
    segment_by_id = {int(segment["seg_id"]): segment for segment in official_segments}
    connector_graph = load_connector_graph(connector_geojson, official_segments=official_segments)
    elevation_sampler = load_dem_context(dem_tif, dem_summary_json)["sampler"]
    probes: dict[str, dict[str, Any]] = {}
    for area in manual_design.get("areas") or []:
        for alternative in area.get("alternatives") or []:
            required_ids = [int(item) for item in alternative.get("required_segment_ids") or []]
            state = forced_anchor_state(base_state, area, alternative)
            if not required_ids or not state:
                continue
            missing = [seg_id for seg_id in required_ids if seg_id not in segment_by_id]
            if missing:
                probes[str(alternative.get("alternative_id"))] = {
                    "error": "missing_segments",
                    "missing_segment_ids": missing,
                }
                continue
            performance_profile = build_performance_profile(
                state=state,
                strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
                activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
                activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
                segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
            )
            trails = group_remaining_by_trail([segment_by_id[seg_id] for seg_id in required_ids])
            candidate = candidate_from_trail_group(
                trails,
                state,
                performance_profile,
                connector_graph,
                candidate_type="manual_route_design_probe",
                elevation_sampler=elevation_sampler,
            )
            probes[str(alternative.get("alternative_id"))] = candidate
    return probes, {"connector_graph": connector_graph}


def summarize_probe_candidates(probe_candidate_objects: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries = {}
    for alternative_id, candidate in probe_candidate_objects.items():
        if candidate.get("error"):
            summaries[alternative_id] = candidate
        else:
            summaries[alternative_id] = candidate_probe_summary(candidate)
    return summaries


def build_probe_candidates(
    manual_design: dict[str, Any],
    state_json: Path,
    official_geojson: Path,
    connector_geojson: Path,
    dem_tif: Path,
    dem_summary_json: Path,
) -> dict[str, dict[str, Any]]:
    probe_objects, _context = build_probe_candidate_objects(
        manual_design,
        state_json,
        official_geojson,
        connector_geojson,
        dem_tif,
        dem_summary_json,
    )
    return summarize_probe_candidates(probe_objects)


def build_design_report(
    package_pass: dict[str, Any],
    manual_design: dict[str, Any],
    probe_candidates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probe_candidates = probe_candidates or {}
    areas = []
    for area in manual_design.get("areas") or []:
        package = find_package(package_pass, area)
        demoted_components = components_for_ids(package, area.get("demote_candidate_ids") or [])
        kept_components = components_for_ids(package, area.get("keep_candidate_ids") or [])
        current_on_foot = sum(float(component.get("on_foot_miles") or 0.0) for component in demoted_components)
        current_official = sum(float(component.get("official_miles") or 0.0) for component in demoted_components)
        acceptance_target_on_foot = max(0.0, current_on_foot - 8.0) if current_on_foot else None
        alternatives = []
        for alternative in area.get("alternatives") or []:
            alternative_id = str(alternative.get("alternative_id"))
            target_range = alternative.get("target_on_foot_miles_range")
            target_min = float(target_range[0]) if isinstance(target_range, list) and len(target_range) == 2 else None
            target_max = float(target_range[1]) if isinstance(target_range, list) and len(target_range) == 2 else None
            probe = probe_candidates.get(alternative_id)
            alternatives.append(
                {
                    **alternative,
                    "target_on_foot_miles_text": target_range_text(target_range),
                    "probe": probe,
                    "beats_current_placeholder_by_8_miles_if": (
                        target_max is not None
                        and acceptance_target_on_foot is not None
                        and target_max <= acceptance_target_on_foot
                    ),
                    "could_beat_placeholder_if_actual_near_low_end": (
                        target_min is not None
                        and acceptance_target_on_foot is not None
                        and target_min <= acceptance_target_on_foot
                    ),
                }
            )
        default_alternatives = [
            alternative
            for alternative in alternatives
            if alternative.get("status") == "manual_gpx_required"
            and alternative.get("probe")
            and not alternative["probe"].get("error")
        ]
        default_probe_on_foot = sum(float((alternative.get("probe") or {}).get("on_foot_miles") or 0.0) for alternative in default_alternatives)
        default_probe_official = sum(float((alternative.get("probe") or {}).get("official_miles") or 0.0) for alternative in default_alternatives)
        default_probe_minutes = sum(int((alternative.get("probe") or {}).get("total_minutes") or 0) for alternative in default_alternatives)
        default_probe_valid = bool(default_alternatives) and all(
            (alternative.get("probe") or {}).get("route_status") == "graph_validated"
            and (alternative.get("probe") or {}).get("ascent_direction_passed") is True
            for alternative in default_alternatives
        )
        improvement = current_on_foot - default_probe_on_foot if current_on_foot and default_probe_on_foot else None
        areas.append(
            {
                "area_id": area.get("area_id"),
                "title": area.get("title"),
                "status": area.get("status"),
                "decision": area.get("decision"),
                "package_number": area.get("package_number"),
                "block_id": area.get("block_id"),
                "current_demoted_official_miles": round_miles(current_official),
                "current_demoted_on_foot_miles": round_miles(current_on_foot),
                "acceptance_target_on_foot_miles": round_miles(acceptance_target_on_foot)
                if acceptance_target_on_foot is not None
                else None,
                "default_split_probe": {
                    "alternative_ids": [alternative.get("alternative_id") for alternative in default_alternatives],
                    "official_miles": round_miles(default_probe_official),
                    "on_foot_miles": round_miles(default_probe_on_foot),
                    "door_to_door_minutes_if_separate_outings": default_probe_minutes,
                    "improvement_vs_current_on_foot_miles": round_miles(improvement) if improvement is not None else None,
                    "passes_probe_acceptance": bool(
                        default_probe_valid and improvement is not None and improvement >= 8.0
                    ),
                    "note": "Probe totals assume the listed alternatives are separate parked-start outings, not one continuous activity.",
                },
                "demoted_components": demoted_components,
                "kept_components": kept_components,
                "anchors": area.get("anchors") or [],
                "alternatives": alternatives,
                "acceptance_gates": area.get("acceptance_gates") or [],
                "preflight_notes": area.get("preflight_notes") or [],
                "source_links": area.get("source_links") or [],
                "recommendation": (
                    "Keep the kept components in the normal outing menu. Build manual GPX for the priority alternatives; "
                    "do not schedule the demoted placeholder until it passes the acceptance gates."
                ),
            }
        )
    return {
        "planning_status": "manual_route_design_pass",
        "summary": {
            "manual_area_count": len(areas),
            "held_candidate_count": sum(len(area.get("demoted_components") or []) for area in areas),
            "kept_candidate_count": sum(len(area.get("kept_components") or []) for area in areas),
        },
        "areas": areas,
    }


def coords_feature(
    coords: list[tuple[float, float]],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lon, lat in coords]},
        "properties": properties,
    }


def parking_feature(candidate: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any] | None:
    trailhead = candidate.get("trailhead") or {}
    if trailhead.get("lon") is None or trailhead.get("lat") is None:
        return None
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(trailhead["lon"]), float(trailhead["lat"])],
        },
        "properties": {
            **properties,
            "kind": "parking",
            "name": trailhead.get("name"),
            "parking_confidence": trailhead.get("parking_confidence") or "manual_required",
            "has_parking": trailhead.get("has_parking"),
        },
    }


def render_route_map_html(title: str, geojson: dict[str, Any]) -> str:
    payload = json.dumps(geojson, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #map {{ min-height:100vh; }}
    .panel {{ position:absolute; z-index:500; top:14px; left:14px; max-width:360px; background:#fff; border:2px solid #111827; border-radius:6px; padding:10px 12px; box-shadow:0 10px 30px rgba(15,23,42,.18); }}
    .panel h1 {{ margin:0 0 5px; font-size:16px; }}
    .panel p {{ margin:4px 0; color:#344054; font-size:12px; line-height:1.35; }}
    .parking {{ width:22px; height:22px; border-radius:50%; background:#111827; color:#fff; border:2px solid #fff; display:flex; align-items:center; justify-content:center; font-weight:800; box-shadow:0 2px 8px rgba(15,23,42,.32); }}
  </style>
</head>
<body>
<div id="map"><div class="panel"><h1>{html.escape(title)}</h1><p>Accepted 16A split probe routes. These remain parking/access-manual until the lower Sweet/Dry access is verified.</p></div></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = {payload};
const map = L.map("map", {{ preferCanvas:true }});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{ maxZoom:19, attribution:"&copy; OpenStreetMap contributors" }}).addTo(map);
const colors = {{"16A-1":"#2563eb","16A-2":"#dc2626","16A-3":"#b45309"}};
function popup(props) {{
  return `<strong>${{props.alternative_id || props.name}}</strong><br>${{props.title || ""}}<br>${{props.official_miles || "n/a"}} official mi · ${{props.on_foot_miles || "n/a"}} on-foot mi<br>Status: ${{props.route_status || props.parking_confidence || "n/a"}}`;
}}
function parkingIcon() {{
  return L.divIcon({{ className:"", iconSize:[26,26], iconAnchor:[13,13], html:'<div class="parking">P</div>' }});
}}
const layers = [];
L.geoJSON(DATA, {{
  style: feature => ({{
    color: colors[feature.properties.alternative_id] || "#111827",
    weight: feature.properties.kind === "route" ? 5 : 2,
    opacity: .95,
  }}),
  pointToLayer: (feature, latlng) => L.marker(latlng, {{ icon:parkingIcon() }}),
  onEachFeature: (feature, layer) => {{
    layer.bindPopup(popup(feature.properties || {{}}));
    layers.push(layer);
  }},
}}).addTo(map);
if (layers.length) map.fitBounds(L.featureGroup(layers).getBounds(), {{ padding:[24,24], maxZoom:14 }});
</script>
</body>
</html>
"""


def write_probe_route_artifacts(
    report: dict[str, Any],
    probe_candidate_objects: dict[str, dict[str, Any]],
    official_geojson: Path,
    connector_graph: dict[str, Any],
    output_dir: Path,
    basename: str,
    max_gap_miles: float = 0.05,
) -> list[Path]:
    official_index = load_official_segment_index(official_geojson)
    output_paths: list[Path] = []
    accepted_ids = {
        alternative_id
        for area in report.get("areas") or []
        for alternative_id in (area.get("default_split_probe") or {}).get("alternative_ids") or []
    }
    features: list[dict[str, Any]] = []
    artifact_records: dict[str, dict[str, Any]] = {}
    gpx_dir = output_dir / f"{basename}-gpx"
    gpx_dir.mkdir(parents=True, exist_ok=True)
    for area in report.get("areas") or []:
        for alternative in area.get("alternatives") or []:
            alternative_id = str(alternative.get("alternative_id"))
            if alternative_id not in accepted_ids:
                continue
            candidate = probe_candidate_objects.get(alternative_id) or {}
            if not candidate or candidate.get("error"):
                continue
            coords = candidate_track_coordinates(
                candidate,
                official_index,
                connector_graph=connector_graph,
                densify_source_lines=True,
            )
            validation = validate_track_segments([coords], max_gap_miles=max_gap_miles)
            slug = slugify(f"{basename}-{alternative_id}-{alternative.get('title')}")
            gpx_path = gpx_dir / f"{slug}.gpx"
            gpx_path.write_text(
                render_gpx_segments(f"{alternative_id} {alternative.get('title')}", [coords]),
                encoding="utf-8",
            )
            output_paths.append(gpx_path)
            props = {
                "kind": "route",
                "area_id": area.get("area_id"),
                "alternative_id": alternative_id,
                "title": alternative.get("title"),
                "route_status": candidate.get("route_status"),
                "official_miles": candidate.get("official_new_miles"),
                "on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
                "total_minutes": candidate.get("total_minutes"),
                "gpx_path": str(gpx_path),
                "validation_passed": validation["passed"],
                "max_trackpoint_gap_miles": validation["max_trackpoint_gap_miles"],
            }
            features.append(coords_feature(coords, props))
            parking = parking_feature(candidate, props)
            if parking:
                features.append(parking)
            artifact_records[alternative_id] = {
                "gpx_path": str(gpx_path),
                "track_validation": validation,
            }
    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path = output_dir / f"{basename}-accepted-routes.geojson"
    map_path = output_dir / f"{basename}-accepted-routes-map.html"
    geojson_path.write_text(json.dumps(geojson, indent=2) + "\n", encoding="utf-8")
    map_path.write_text(render_route_map_html("Package 16A Accepted Split Probe Routes", geojson), encoding="utf-8")
    output_paths.extend([geojson_path, map_path])
    report["generated_route_artifacts"] = {
        "accepted_alternative_ids": sorted(accepted_ids),
        "geojson_path": str(geojson_path),
        "map_html_path": str(map_path),
        "gpx_dir": str(gpx_dir),
        "routes": artifact_records,
        "all_track_validations_passed": all(
            record["track_validation"]["passed"] for record in artifact_records.values()
        )
        if artifact_records
        else False,
    }
    for area in report.get("areas") or []:
        split_probe = area.get("default_split_probe") or {}
        if (
            set(split_probe.get("alternative_ids") or []) <= set(artifact_records)
            and split_probe.get("passes_probe_acceptance") is True
            and report["generated_route_artifacts"]["all_track_validations_passed"]
        ):
            area["status"] = "accepted_split_probe_parking_manual"
            area["current_good_route"] = {
                "alternative_ids": split_probe.get("alternative_ids") or [],
                "official_miles": split_probe.get("official_miles"),
                "on_foot_miles": split_probe.get("on_foot_miles"),
                "door_to_door_minutes_if_separate_outings": split_probe.get(
                    "door_to_door_minutes_if_separate_outings"
                ),
                "remaining_blocker": "day-of roadside parking capacity/signage and current trail conditions",
            }
            area["recommendation"] = (
                "Use the accepted split probe as the current best 16A route: run 16A-1 and 16A-2 as two "
                "separate parked-start outings from the Dry Creek / Sweet Connie roadside parking after day-of capacity, signage, and conditions checks."
            )
        for alternative in area.get("alternatives") or []:
            alternative_id = str(alternative.get("alternative_id"))
            if alternative_id in artifact_records:
                alternative["generated_route_artifact"] = artifact_records[alternative_id]
                alternative["route_design_status"] = "gpx_generated_parking_manual"
    return output_paths


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Manual Route Design Pass v1",
        "",
        "Status: route-design holds that should not be treated as runnable outing-menu rows yet.",
        "",
        "## Summary",
        "",
        f"- Manual areas: {report['summary']['manual_area_count']}",
        f"- Held candidate components: {report['summary']['held_candidate_count']}",
        f"- Kept runnable components: {report['summary']['kept_candidate_count']}",
        "",
    ]
    for area in report.get("areas") or []:
        lines.extend(
            [
                f"## {area.get('title')}",
                "",
                f"- Status: {area.get('status')}",
                f"- Decision: {area.get('decision')}",
                f"- Current held official miles: {area.get('current_demoted_official_miles')}",
                f"- Current held on-foot miles: {area.get('current_demoted_on_foot_miles')}",
                f"- Acceptance target: <= {area.get('acceptance_target_on_foot_miles')} on-foot miles for the held work",
                f"- Recommendation: {area.get('recommendation')}",
                "",
                "### Current Best Split Probe",
                "",
            ]
        )
        split_probe = area.get("default_split_probe") or {}
        if split_probe.get("alternative_ids"):
            lines.extend(
                [
                    f"- Alternatives: {', '.join(split_probe.get('alternative_ids') or [])}",
                    f"- Official miles: {split_probe.get('official_miles')}",
                    f"- On-foot miles: {split_probe.get('on_foot_miles')}",
                    f"- Door-to-door minutes if run separately: {split_probe.get('door_to_door_minutes_if_separate_outings')}",
                    f"- Improvement vs current 16A placeholder: {split_probe.get('improvement_vs_current_on_foot_miles')} on-foot miles",
                    f"- Probe acceptance passed: {split_probe.get('passes_probe_acceptance')}",
                    f"- Note: {split_probe.get('note')}",
                    "",
                ]
            )
        current_good_route = area.get("current_good_route") or {}
        if current_good_route:
            lines.extend(
                [
                    "### Current Good 16A Route",
                    "",
                    f"- Route: {', '.join(current_good_route.get('alternative_ids') or [])}",
                    f"- Official miles: {current_good_route.get('official_miles')}",
                    f"- On-foot miles: {current_good_route.get('on_foot_miles')}",
                    f"- Door-to-door minutes if run separately: {current_good_route.get('door_to_door_minutes_if_separate_outings')}",
                    f"- Remaining blocker: {current_good_route.get('remaining_blocker')}",
                    "",
                ]
            )
        else:
            lines.extend(["- No default split probe was generated.", ""])
        artifacts = report.get("generated_route_artifacts") or {}
        if artifacts:
            lines.extend(
                [
                    "### Generated Route Artifacts",
                    "",
                    f"- Accepted alternatives: {', '.join(artifacts.get('accepted_alternative_ids') or [])}",
                    f"- GPX folder: `{artifacts.get('gpx_dir')}`",
                    f"- GeoJSON: `{artifacts.get('geojson_path')}`",
                    f"- Map: `{artifacts.get('map_html_path')}`",
                    f"- Track continuity passed: {artifacts.get('all_track_validations_passed')}",
                    "",
                ]
            )
        lines.extend(
            [
                "### Runnable/Kept Components",
                "",
                "| Candidate | Trailhead | Official mi | On-foot mi |",
                "|---|---|---:|---:|",
            ]
        )
        for component in area.get("kept_components") or []:
            lines.append(
                f"| {component.get('candidate_id')} | {component.get('trailhead')} | "
                f"{component.get('official_miles')} | {component.get('on_foot_miles')} |"
            )
        if not area.get("kept_components"):
            lines.append("| n/a | n/a | n/a | n/a |")
        lines.extend(
            [
                "",
                "### Held Placeholder Components",
                "",
                "| Candidate | Trailhead | Official mi | On-foot mi | Flags |",
                "|---|---|---:|---:|---|",
            ]
        )
        for component in area.get("demoted_components") or []:
            lines.append(
                f"| {component.get('candidate_id')} | {component.get('trailhead')} | "
                f"{component.get('official_miles')} | {component.get('on_foot_miles')} | "
                f"{', '.join(component.get('less_optimal_flags') or [])} |"
            )
        lines.extend(
            [
                "",
                "### Alternatives To Build",
                "",
                "| Alternative | Status | Target official | Target on-foot | Segment IDs | Gate posture |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for alternative in area.get("alternatives") or []:
            probe = alternative.get("probe") or {}
            gate_posture = (
                "target max passes 8-mile improvement gate"
                if alternative.get("beats_current_placeholder_by_8_miles_if")
                else "actual GPX must come in near low end to pass 8-mile improvement gate"
                if alternative.get("could_beat_placeholder_if_actual_near_low_end")
                else "comparison only or not enough target data"
            )
            lines.append(
                f"| {alternative.get('alternative_id')}: {alternative.get('title')} | "
                f"{alternative.get('route_design_status') or alternative.get('status')} | {alternative.get('target_official_miles', 'n/a')} | "
                f"{alternative.get('target_on_foot_miles_text')} | "
                f"{', '.join(str(item) for item in alternative.get('required_segment_ids') or [])} | "
                f"{gate_posture} |"
            )
            if probe and not probe.get("error"):
                lines.append(
                    f"| probe | {probe.get('route_status')} | {probe.get('official_miles')} | "
                    f"{probe.get('on_foot_miles')} | {', '.join(str(item) for item in alternative.get('required_segment_ids') or [])} | "
                    f"ascent={probe.get('ascent_direction_passed')}, trailhead_snap={probe.get('trailhead_snap_confidence')} |"
                )
        lines.extend(["", "### Acceptance Gates", ""])
        lines.extend(f"- {gate}" for gate in area.get("acceptance_gates") or [])
        if area.get("preflight_notes"):
            lines.extend(["", "### Preflight Notes", ""])
            lines.extend(f"- {note}" for note in area.get("preflight_notes") or [])
        if area.get("source_links"):
            lines.extend(["", "### Source Links", ""])
            lines.extend(f"- {link}" for link in area.get("source_links") or [])
        lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-pass-json", type=Path, default=DEFAULT_PACKAGE_PASS_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_pass = read_json(args.package_pass_json)
    manual_design = read_json(args.manual_design_json)
    probe_candidate_objects, probe_context = build_probe_candidate_objects(
        manual_design,
        args.state_json,
        args.official_geojson,
        args.connector_geojson,
        args.dem_tif,
        args.dem_summary_json,
    )
    probe_candidates = summarize_probe_candidates(probe_candidate_objects)
    report = build_design_report(package_pass, manual_design, probe_candidates=probe_candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    route_artifact_paths = write_probe_route_artifacts(
        report,
        probe_candidate_objects,
        args.official_geojson,
        probe_context["connector_graph"],
        args.output_dir,
        args.basename,
    )
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[
                args.package_pass_json,
                args.manual_design_json,
                args.state_json,
                args.official_geojson,
                args.connector_geojson,
                args.dem_tif,
                args.dem_summary_json,
            ],
            outputs=[json_path, md_path, *route_artifact_paths],
            command="manual_route_design_pass.py",
            metadata=report["summary"],
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
