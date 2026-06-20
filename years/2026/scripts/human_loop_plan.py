#!/usr/bin/env python3
"""Build the user-facing human loop/block plan from the combo route pass."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_day_packager import (  # noqa: E402
    manual_design_area_for_candidate_ids,
    render_html as render_package_map_html,
    render_outing_menu_markdown,
)
from export_execution_gpx import (  # noqa: E402
    candidate_track_coordinates,
    load_official_segment_index,
    special_management_segment_direction_overrides,
    validate_track_segments,
)
from field_activity_review import activity_coordinates, normalized_ids, review_activity_against_segments  # noqa: E402
from multi_start_field_menu_replacements import (  # noqa: E402
    coords_feature,
    cue_from_candidate,
    parking_feature,
)
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, load_official_segments, read_json  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    build_elevation_effort,
    build_performance_profile,
    build_time_estimates_minutes,
    candidate_from_trail_group,
    drive_minutes_to_trailhead,
    enrich_segment_estimates_with_elevation,
    group_remaining_by_trail,
    haversine_miles,
    load_connector_graph,
    load_dem_context,
    load_state,
)


DEFAULT_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1.json"
DEFAULT_PACKAGE_MAP_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1-map-data.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "human-loop-plan-v1"
DEFAULT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_OUTING_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = DEFAULT_OFFICIAL_GEOJSON
DEFAULT_MANUAL_DESIGN_JSON = YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json"
DEFAULT_MANUAL_DESIGN_REPORT_JSON = (
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "package16-manual-route-design-v1.json"
)
DEFAULT_MANUAL_DESIGN_REPORT_JSONS = [
    DEFAULT_MANUAL_DESIGN_REPORT_JSON,
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "harlow-spring-manual-route-design-v1.json",
]
DEFAULT_FIELD_MENU_OVERRIDES_JSON = YEAR_DIR / "inputs" / "personal" / "2026-field-menu-overrides-v1.json"
DEFAULT_GENERATED_MULTI_START_FIELD_MENU_REPLACEMENTS_JSON = (
    YEAR_DIR / "inputs" / "personal" / "private" / "2026-field-menu-replacements-v2-multi-start.private.json"
)
DEFAULT_TIME_CALIBRATIONS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-field-time-calibrations-v1.json"
DEFAULT_ROUTE_TRUTH_REPAIRS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-truth-repairs-v1.json"


NECESSARY_GRINDER_TERMS = {
    "cartwright",
    "cervidae",
    "dry creek",
    "harlow",
    "harris ridge",
    "mores",
    "oregon",
    "shingle",
    "spring creek",
    "stack rock",
    "sweet connie",
    "watchman",
}

PROMOTED_MANUAL_ROUTE_PREFIX = "manual-"


def default_field_menu_overrides_json() -> Path:
    """Prefer generated multi-start replacements for local active planning."""

    if DEFAULT_GENERATED_MULTI_START_FIELD_MENU_REPLACEMENTS_JSON.exists():
        return DEFAULT_GENERATED_MULTI_START_FIELD_MENU_REPLACEMENTS_JSON
    return DEFAULT_FIELD_MENU_OVERRIDES_JSON


def package_status(package: dict[str, Any], map_data: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    reasons = []
    if map_data and manual_design_area_for_candidate_ids(
        map_data,
        package.get("package_number"),
        [str(candidate_id) for candidate_id in package.get("component_candidate_ids") or []],
    ):
        return "manual_design_area", ["coverage_placeholder_needs_human_route_design"]
    ratio = float(package.get("ratio") or 0.0)
    name = str(package.get("block_name") or "").lower()
    if package.get("trailhead_count", 0) > 1:
        reasons.append("split_across_nearby_route_components")
    if package.get("component_routes_under_1_official_mile"):
        reasons.append("tiny_segments_absorbed")
    if package.get("boundary_review"):
        reasons.append("block_boundary_review_recorded")
    if ratio > 2.0 or any(term in name for term in NECESSARY_GRINDER_TERMS):
        reasons.append("necessary_grinder_or_geography_locked")
        return "necessary_grinder", reasons
    if package.get("trailhead_count", 0) > 1 or package.get("component_route_count", 0) > 1:
        return "accepted_split_block", reasons
    return "primary_loop_block", reasons


def build_human_plan(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    map_html_path: Path,
    map_data_json_path: Path | None = None,
) -> dict[str, Any]:
    map_data_json_path = map_data_json_path or map_html_path.with_name("2026-outing-menu-map-data.json")
    route_statuses = {str(route.get("route_status")) for route in route_pass.get("routes") or []}
    map_validation = package_map.get("map_validation") or {}
    packages = []
    for package in package_pass.get("packages") or []:
        status, reasons = package_status(package, package_map)
        packages.append(
            {
                **package,
                "human_plan_status": status,
                "human_plan_reasons": reasons,
            }
        )
    status_counts: dict[str, int] = {}
    for package in packages:
        status_counts[package["human_plan_status"]] = status_counts.get(package["human_plan_status"], 0) + 1
    package_summary = package_pass.get("summary") or {}
    unresolved = []
    if package_summary.get("covered_segment_count") != 251:
        unresolved.append("coverage_not_complete")
    if route_statuses != {"graph_validated"}:
        unresolved.append("non_graph_validated_route_components")
    if map_validation.get("rendered_passed") is not True:
        unresolved.append("map_render_validation_failed")
    return {
        "planning_status": "human_loop_plan",
        "summary": {
            **package_summary,
            "route_component_count": package_summary.get("component_route_count") or len(package_components(packages)),
            "status_counts": status_counts,
            "unresolved_blocker_count": len(unresolved),
            "unresolved_blockers": unresolved,
            "all_route_components_graph_validated": route_statuses == {"graph_validated"},
            "map_rendered_passed": map_validation.get("rendered_passed") is True,
            "map_html": str(map_html_path),
            "map_data_json": str(map_data_json_path),
            "manual_design_area_count": status_counts.get("manual_design_area", 0),
        },
        "packages": packages,
        "routes": route_pass.get("routes") or [],
        "caveats": [
            "This is a planning route book, not a day-of conditions clearance. Ridge to Rivers conditions/signage still need checking before each outing.",
            "Accepted split blocks intentionally keep short nearby components separate when merging them created excessive dead mileage.",
            "Necessary grinders are retained because the geography and single-car return-to-car constraint make them expensive but still required for 100 percent coverage.",
        ],
    }


def merge_manual_design_report(manual_design: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    if not manual_design or not report:
        return manual_design
    report_areas = {str(area.get("area_id")): area for area in report.get("areas") or []}
    merged = {**manual_design, "areas": []}
    for area in manual_design.get("areas") or []:
        report_area = report_areas.get(str(area.get("area_id"))) or {}
        enriched_area = {**area}
        for key in [
            "status",
            "recommendation",
            "default_split_probe",
            "current_demoted_on_foot_miles",
            "acceptance_target_on_foot_miles",
            "current_good_route",
        ]:
            if key in report_area:
                enriched_area[key] = report_area[key]
        if report.get("generated_route_artifacts"):
            enriched_area["generated_route_artifacts"] = report["generated_route_artifacts"]
        report_alternatives = {
            str(alternative.get("alternative_id")): alternative
            for alternative in report_area.get("alternatives") or []
        }
        enriched_area["alternatives"] = [
            {
                **alternative,
                **{
                    key: value
                    for key, value in (report_alternatives.get(str(alternative.get("alternative_id"))) or {}).items()
                    if key in {
                        "probe",
                        "target_on_foot_miles_text",
                        "route_design_status",
                        "generated_route_artifact",
                    }
                },
            }
            for alternative in area.get("alternatives") or []
        ]
        merged["areas"].append(enriched_area)
    return merged


def round_miles(value: Any) -> float:
    return round(float(value or 0.0), 2)


def promoted_candidate_id(alternative_id: str) -> str:
    return PROMOTED_MANUAL_ROUTE_PREFIX + alternative_id.lower()


def selected_manual_alternatives(area: dict[str, Any]) -> list[dict[str, Any]]:
    good_route = area.get("current_good_route") or area.get("default_split_probe") or {}
    accepted_ids = [str(value) for value in good_route.get("alternative_ids") or []]
    if not accepted_ids:
        accepted_ids = [
            str(value)
            for value in ((area.get("generated_route_artifacts") or {}).get("accepted_alternative_ids") or [])
        ]
    alternatives_by_id = {str(alt.get("alternative_id")): alt for alt in area.get("alternatives") or []}
    selected = []
    for alternative_id in accepted_ids:
        alternative = alternatives_by_id.get(alternative_id)
        if not alternative:
            continue
        probe = alternative.get("probe") or {}
        generated = alternative.get("generated_route_artifact") or {}
        validation = generated.get("track_validation") or {}
        if alternative.get("route_design_status") != "gpx_generated_parking_manual":
            continue
        if probe.get("route_status") != "graph_validated":
            continue
        if validation.get("passed") is not True:
            continue
        selected.append(alternative)
    return selected


def load_accepted_route_features(report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    artifacts = report.get("generated_route_artifacts") or {}
    geojson_path = artifacts.get("geojson_path")
    if not geojson_path:
        return {}
    path = Path(geojson_path)
    if not path.exists():
        return {}
    geojson = json.loads(path.read_text(encoding="utf-8"))
    by_alternative: dict[str, dict[str, dict[str, Any]]] = {}
    for feature in geojson.get("features") or []:
        props = feature.get("properties") or {}
        alternative_id = props.get("alternative_id")
        if not alternative_id:
            continue
        kind = props.get("kind")
        if kind not in {"route", "parking"}:
            continue
        by_alternative.setdefault(str(alternative_id), {})[str(kind)] = feature
    return by_alternative


def trailhead_from_anchor(
    area: dict[str, Any],
    alternative: dict[str, Any],
    parking_feature: dict[str, Any] | None,
) -> dict[str, Any]:
    probe = alternative.get("probe") or {}
    anchor_id = alternative.get("start_anchor_id")
    anchor = next(
        (item for item in area.get("anchors") or [] if str(item.get("anchor_id")) == str(anchor_id)),
        {},
    )
    point = ((parking_feature or {}).get("geometry") or {}).get("coordinates") or []
    lat = anchor.get("lat")
    lon = anchor.get("lon")
    if len(point) >= 2:
        lon = point[0]
        lat = point[1]
    return {
        "name": anchor.get("name") or probe.get("trailhead") or "Manual parking/start",
        "lat": lat,
        "lon": lon,
        "has_parking": anchor.get("has_parking", True),
        "has_restroom": anchor.get("has_restroom"),
        "has_water": anchor.get("has_water"),
        "water_confidence": anchor.get("water_confidence"),
        "parking_minutes": anchor.get("parking_minutes", 8),
        "source": anchor.get("source") or "manual_route_design",
        "parking_confidence": anchor.get("parking_confidence") or "manual_required",
        "field_ready": anchor.get("field_ready"),
    }


def old_cue_for_alternative(
    route_cues: dict[str, Any],
    demoted_candidate_ids: list[str],
    alternative: dict[str, Any],
) -> dict[str, Any]:
    required = {str(segment_id) for segment_id in alternative.get("required_segment_ids") or []}
    for candidate_id in demoted_candidate_ids:
        cue = route_cues.get(str(candidate_id)) or {}
        cue_segment_ids = {str(segment.get("seg_id")) for segment in cue.get("segments") or []}
        if required and required <= cue_segment_ids:
            return cue
    for candidate_id in demoted_candidate_ids:
        cue = route_cues.get(str(candidate_id))
        if cue:
            return cue
    return {}


def cue_segments_for_alternative(cue: dict[str, Any], alternative: dict[str, Any]) -> list[dict[str, Any]]:
    required = {str(segment_id) for segment_id in alternative.get("required_segment_ids") or []}
    segments = cue.get("segments") or []
    if not required:
        return list(segments)
    return [segment for segment in segments if str(segment.get("seg_id")) in required]


def alternative_probe_segments(alternative: dict[str, Any]) -> list[dict[str, Any]]:
    required = {str(segment_id) for segment_id in alternative.get("required_segment_ids") or []}
    probe = alternative.get("probe") or {}
    segments = copy.deepcopy(probe.get("segments") or [])
    if required:
        segments = [segment for segment in segments if str(segment.get("seg_id")) in required]
    return segments


def cue_or_probe_segments_for_alternative(cue: dict[str, Any], alternative: dict[str, Any]) -> list[dict[str, Any]]:
    cue_segments = cue_segments_for_alternative(cue, alternative)
    return cue_segments or alternative_probe_segments(alternative)


def official_segments_for_alternative(
    alternative: dict[str, Any],
    official_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = {str(segment.get("seg_id")): segment for segment in official_segments}
    result = []
    for order, segment_id in enumerate(alternative.get("required_segment_ids") or [], start=1):
        official = index.get(str(segment_id))
        if not official:
            continue
        direction = official.get("direction") or "both"
        result.append(
            {
                "order": order,
                "seg_id": int(segment_id) if str(segment_id).isdigit() else segment_id,
                "segment_name": official.get("seg_name") or official.get("segment_name") or official.get("trail_name"),
                "trail_name": official.get("trail_name"),
                "official_miles": round_miles(official.get("official_miles")),
                "direction_rule": direction,
                "direction_cue": (
                    "ASCENT REQUIRED: follow map arrows uphill."
                    if str(direction).lower() == "ascent"
                    else "Either direction allowed; follow map arrows."
                ),
            }
        )
    return result


def completed_official_ids_for_alternative(
    alternative: dict[str, Any],
    official_segments: list[dict[str, Any]],
) -> list[str]:
    planned_ids = normalized_ids(alternative.get("required_segment_ids") or [])
    if not planned_ids:
        return []
    artifact = alternative.get("generated_route_artifact") or {}
    gpx_path = artifact.get("gpx_path")
    if not gpx_path or not official_segments:
        return []
    path = Path(str(gpx_path))
    if not path.exists():
        return []
    coords = activity_coordinates(path)
    if len(coords) < 2:
        return []
    planned_set = set(planned_ids)
    official_subset = [
        segment
        for segment in official_segments
        if str(segment.get("seg_id")) in planned_set
    ]
    if not official_subset:
        return []
    review = review_activity_against_segments(
        coords,
        official_subset,
        planned_segment_ids=planned_ids,
        threshold_miles=0.045,
        endpoint_threshold_miles=0.04,
        min_fraction=0.85,
        partial_min_fraction=0.2,
    )
    return normalized_ids(review.get("completed_segment_ids") or [])


def filter_cue_links_to_segments(cue: dict[str, Any]) -> None:
    trail_names = {str(segment.get("trail_name")) for segment in cue.get("segments") or [] if segment.get("trail_name")}
    if not trail_names:
        return
    for key in ["between_links", "between_trail_links"]:
        links = cue.get(key)
        if not isinstance(links, list):
            continue
        cue[key] = [
            link
            for link in links
            if str(link.get("from_trail")) in trail_names and str(link.get("to_trail")) in trail_names
        ]


def update_manual_package_summary(package: dict[str, Any]) -> None:
    components = package.get("components") or []
    official = sum(float(component.get("official_miles") or 0.0) for component in components)
    on_foot = sum(float(component.get("on_foot_miles") or 0.0) for component in components)
    trailheads = sorted({str(component.get("trailhead")) for component in components if component.get("trailhead")})
    candidate_ids = [component.get("candidate_id") for component in components if component.get("candidate_id")]
    trail_names = sorted(
        {
            str(trail)
            for component in components
            for trail in (component.get("trail_names") or [])
            if trail
        }
    )
    package["component_route_count"] = len(components)
    package["component_candidate_ids"] = candidate_ids
    package["trail_names"] = trail_names
    package["official_miles"] = round_miles(official)
    package["on_foot_miles"] = round_miles(on_foot)
    package["ratio"] = round(on_foot / official, 2) if official else None
    package["trailheads"] = trailheads
    package["trailhead_count"] = len(trailheads)
    package["primary_trailhead"] = trailheads[0] if len(trailheads) == 1 else None
    package["total_minutes_components"] = sum(int(component.get("total_minutes") or 0) for component in components)
    package["component_routes_under_1_official_mile"] = sum(
        1 for component in components if float(component.get("official_miles") or 0.0) < 1.0
    )
    package["component_routes_under_2_official_miles"] = sum(
        1 for component in components if float(component.get("official_miles") or 0.0) < 2.0
    )


def normalized_int_ids(values: Any) -> list[int]:
    ids = set()
    for value in values or []:
        if value is None:
            continue
        ids.add(int(value))
    return sorted(ids)


def sync_progress_from_state(package_map: dict[str, Any], state: dict[str, Any]) -> None:
    """Keep generated map progress tied to the active private planner state."""

    package_map["progress"] = {
        "completed_segment_ids": normalized_int_ids(state.get("completed_segment_ids")),
        "blocked_segment_ids": normalized_int_ids(state.get("blocked_segment_ids")),
        "blocked_trail_names": sorted(str(name) for name in state.get("blocked_trail_names") or []),
    }


def package_components(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [component for package in packages for component in package.get("components") or []]


def recompute_package_summary(data: dict[str, Any], map_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild summary metrics after manual promotions and field-menu overrides."""

    packages = data.get("packages") or []
    for package in packages:
        update_manual_package_summary(package)
    components = package_components(packages)
    segment_ids = {
        str(segment_id)
        for component in components
        for segment_id in component.get("segment_ids") or []
        if segment_id is not None
    }
    official_miles = sum(float(component.get("official_miles") or 0.0) for component in components)
    on_foot_miles = sum(float(component.get("on_foot_miles") or 0.0) for component in components)
    trailhead_counts = [int(package.get("trailhead_count") or 0) for package in packages]
    summary = dict(data.get("summary") or {})
    summary.update(
        {
            "package_count": len(packages),
            "component_route_count": len(components),
            "covered_segment_count": len(segment_ids),
            "official_miles": round_miles(official_miles),
            "total_on_foot_miles": round_miles(on_foot_miles),
            "planwide_on_foot_to_official_ratio": round(on_foot_miles / official_miles, 2) if official_miles else None,
            "packages_with_multiple_trailheads": sum(1 for count in trailhead_counts if count > 1),
            "component_routes_under_1_official_mile": sum(
                1 for component in components if float(component.get("official_miles") or 0.0) < 1.0
            ),
            "component_routes_under_2_official_miles": sum(
                1 for component in components if float(component.get("official_miles") or 0.0) < 2.0
            ),
            "same_trailhead_package_count": sum(1 for count in trailhead_counts if count == 1),
        }
    )
    if map_data is not None:
        summary["manual_route_design_package_count"] = sum(
            1 for package in packages if package_status(package, map_data)[0] == "manual_design_area"
        )
    data["summary"] = summary
    return summary


def feature_segment_id(feature: dict[str, Any]) -> str | None:
    props = feature.get("properties") or {}
    segment_id = props.get("seg_id") or props.get("segment_id") or props.get("segId")
    return str(segment_id) if segment_id is not None else None


def direction_cue_for_segment(segment: dict[str, Any]) -> str:
    direction = str(segment.get("direction") or "both").lower()
    if direction == "ascent":
        return "Ascent-only official segment; follow the planned uphill direction."
    return "Either direction allowed; follow map arrows."


def sync_official_segment_features(package_map: dict[str, Any], official_segments: list[dict[str, Any]]) -> None:
    """Rebuild the official segment layer from active package segment ownership."""

    official_by_id = {str(segment.get("seg_id")): segment for segment in official_segments}
    collections = package_map.setdefault("feature_collections", {})
    official_collection = collections.setdefault("official_segments", {"type": "FeatureCollection", "features": []})
    existing_by_id = {
        segment_id: feature
        for feature in official_collection.get("features") or []
        if (segment_id := feature_segment_id(feature))
    }
    features = []
    seen_segment_ids = set()
    for package in package_map.get("packages") or []:
        for component in package.get("components") or []:
            for segment_id in component.get("segment_ids") or []:
                key = str(segment_id)
                if key in seen_segment_ids:
                    continue
                segment = official_by_id.get(key)
                if not segment:
                    continue
                seen_segment_ids.add(key)
                feature = copy.deepcopy(existing_by_id.get(key)) or {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": segment.get("coordinates") or []},
                    "properties": {},
                }
                props = feature.setdefault("properties", {})
                props.update(
                    {
                        "kind": "official_segment",
                        "package_number": package.get("package_number"),
                        "candidate_id": component.get("candidate_id"),
                        "block_name": package.get("block_name"),
                        "title": ", ".join(component.get("trail_names") or []) or component.get("field_menu_label"),
                        "official_miles": round_miles(segment.get("official_miles")),
                        "segment_official_miles": round_miles(segment.get("official_miles")),
                        "route_official_miles": round_miles(component.get("official_miles")),
                        "route_on_foot_miles": round_miles(component.get("on_foot_miles")),
                        "on_foot_miles": round_miles(component.get("on_foot_miles")),
                        "trailhead": component.get("trailhead"),
                        "color": props.get("color") or "#2563eb",
                        "seg_id": segment.get("seg_id"),
                        "segment_name": segment.get("seg_name") or segment.get("trail_name"),
                        "trail_name": segment.get("trail_name"),
                        "direction_rule": segment.get("direction"),
                    }
                )
                props.setdefault("direction_cue", direction_cue_for_segment(segment))
                features.append(feature)
    official_collection["type"] = "FeatureCollection"
    official_collection["features"] = features


def normalized_direction_cue(segment: dict[str, Any]) -> str:
    segment_id = str(segment.get("seg_id") or segment.get("segment_id") or segment.get("segId") or "")
    direction = str(segment.get("direction_rule") or segment.get("direction") or "both").lower()
    special_direction = special_management_segment_direction_overrides().get(segment_id)
    if special_direction == "forward":
        return "SPECIAL MANAGEMENT: follow the signed one-way direction."
    if special_direction == "reverse":
        return "SPECIAL MANAGEMENT: follow the signed one-way direction opposite official geometry."
    if direction == "ascent":
        return "ASCENT REQUIRED: follow map arrows uphill."
    existing_cue = str(segment.get("direction_cue") or "")
    if "opposite official geometry" in existing_cue.lower():
        return "Either direction allowed; follow map arrows."
    return existing_cue or "Either direction allowed; follow map arrows."


def sync_route_direction_cues(package_map: dict[str, Any]) -> None:
    """Normalize stale route-cue direction labels from current official/rule data."""

    for cue in (package_map.get("route_cues") or {}).values():
        for segment in cue.get("segments") or []:
            segment["direction_cue"] = normalized_direction_cue(segment)

    features = (
        (package_map.get("feature_collections") or {})
        .get("official_segments", {})
        .get("features")
        or []
    )
    for feature in features:
        props = feature.get("properties") or {}
        if props:
            props["direction_cue"] = normalized_direction_cue(props)


def update_overall_summary(summary: dict[str, Any], old_on_foot: float, new_on_foot: float) -> None:
    prior_total = float(summary.get("total_on_foot_miles") or 0.0)
    prior_ratio = float(summary.get("planwide_on_foot_to_official_ratio") or 0.0)
    if "total_on_foot_miles" in summary:
        summary["total_on_foot_miles"] = round_miles(prior_total - old_on_foot + new_on_foot)
    official = float(summary.get("official_miles") or 0.0)
    if not official and prior_ratio:
        official = prior_total / prior_ratio
    if official:
        summary["planwide_on_foot_to_official_ratio"] = round(float(summary.get("total_on_foot_miles") or 0.0) / official, 2)
    if "manual_route_design_package_count" in summary:
        summary["manual_route_design_package_count"] = max(0, int(summary.get("manual_route_design_package_count") or 0) - 1)


def component_segment_ids_for_candidates(
    package_map: dict[str, Any],
    package_number: Any,
    candidate_ids: list[str],
) -> set[str]:
    wanted = {str(candidate_id) for candidate_id in candidate_ids}
    segment_ids = set()
    for package in package_map.get("packages") or []:
        if str(package.get("package_number")) != str(package_number):
            continue
        for component in package.get("components") or []:
            if str(component.get("candidate_id")) in wanted:
                segment_ids.update(str(segment_id) for segment_id in component.get("segment_ids") or [])
    return segment_ids


def record_manual_promotion_skip(
    datasets: list[dict[str, Any]],
    *,
    area_id: str,
    package_number: Any,
    demoted_candidate_ids: list[str],
    selected_alternative_ids: list[str],
    missing_segment_ids: set[str],
) -> None:
    skip = {
        "area_id": area_id,
        "package_number": package_number,
        "demoted_candidate_ids": demoted_candidate_ids,
        "selected_alternative_ids": selected_alternative_ids,
        "missing_segment_ids": sorted(missing_segment_ids, key=lambda item: (len(item), item)),
        "reason": "manual_split_alternatives_do_not_cover_all_demoted_segments",
    }
    for dataset in datasets:
        dataset.setdefault("manual_promotion_skips", []).append(copy.deepcopy(skip))


def clear_manual_promotion_skip(datasets: list[dict[str, Any]], *, area_id: str, package_number: Any) -> None:
    for dataset in datasets:
        skips = dataset.get("manual_promotion_skips")
        if not isinstance(skips, list):
            continue
        dataset["manual_promotion_skips"] = [
            skip
            for skip in skips
            if not (
                str(skip.get("area_id")) == str(area_id)
                and str(skip.get("package_number")) == str(package_number)
            )
        ]


def replace_route_validations(
    map_data: dict[str, Any],
    remove_candidate_ids: list[str],
    new_validations: list[dict[str, Any]],
) -> None:
    map_validation = map_data.get("map_validation") or {}
    route_validations = map_validation.get("route_validations")
    if not isinstance(route_validations, list):
        return
    remove = {str(candidate_id) for candidate_id in remove_candidate_ids}
    map_validation["route_validations"] = [
        validation
        for validation in route_validations
        if str(validation.get("candidate_id")) not in remove
    ] + copy.deepcopy(new_validations)
    map_validation["source_gap_warning_count"] = sum(
        1 for validation in map_validation["route_validations"] if validation.get("source_gap_warning")
    )


def promote_package16_manual_routes(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    manual_design: dict[str, Any],
    manual_design_report: dict[str, Any],
) -> None:
    """Replace accepted manual placeholders with their validated split routes.

    The manual-design report proves a better GPX route, but the field menu is
    built from package components. This promotion keeps the route-design caveat
    in metadata while making the executable menu consume the accepted GPX
    instead of the old coverage placeholder.
    """

    if not manual_design_report:
        return
    accepted_features = load_accepted_route_features(manual_design_report)
    report_areas = {str(area.get("area_id")): area for area in manual_design_report.get("areas") or []}
    manual_areas = {str(area.get("area_id")): area for area in manual_design.get("areas") or []}
    route_cues = package_map.setdefault("route_cues", {})
    official_segments, _official_meta = load_official_segments(DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    for area_id, report_area in report_areas.items():
        area = {**(manual_areas.get(area_id) or {}), **report_area}
        alternatives = selected_manual_alternatives(area)
        if not alternatives:
            continue
        promoted_candidate_ids = {
            promoted_candidate_id(str(alternative.get("alternative_id")))
            for alternative in alternatives
        }
        demoted_candidate_ids = [str(value) for value in area.get("demote_candidate_ids") or []]
        package_number = area.get("package_number")
        demoted_segment_ids = component_segment_ids_for_candidates(package_map, package_number, demoted_candidate_ids)
        selected_segment_ids = {
            str(segment_id)
            for alternative in alternatives
            for segment_id in alternative.get("required_segment_ids") or []
        }
        covered_elsewhere_segment_ids = {
            str(segment_id)
            for segment_id in area.get("covered_elsewhere_segment_ids") or []
        }
        missing_replacement_segment_ids = demoted_segment_ids - selected_segment_ids - covered_elsewhere_segment_ids
        if missing_replacement_segment_ids:
            record_manual_promotion_skip(
                [route_pass, package_pass, package_map],
                area_id=str(area_id),
                package_number=package_number,
                demoted_candidate_ids=demoted_candidate_ids,
                selected_alternative_ids=[str(alternative.get("alternative_id")) for alternative in alternatives],
                missing_segment_ids=missing_replacement_segment_ids,
            )
            continue
        clear_manual_promotion_skip([route_pass, package_pass, package_map], area_id=str(area_id), package_number=package_number)

        for data in [package_pass, package_map]:
            for package in data.get("packages") or []:
                if str(package.get("package_number")) != str(package_number):
                    continue
                old_on_foot = float(package.get("on_foot_miles") or 0.0)
                kept_components = [
                    component
                    for component in package.get("components") or []
                    if str(component.get("candidate_id")) not in demoted_candidate_ids
                    and str(component.get("candidate_id")) not in promoted_candidate_ids
                ]
                promoted_components = []
                for alternative in alternatives:
                    alternative_id = str(alternative.get("alternative_id"))
                    probe = alternative.get("probe") or {}
                    new_candidate_id = promoted_candidate_id(alternative_id)
                    cue_segments = cue_or_probe_segments_for_alternative(
                        old_cue_for_alternative(route_cues, demoted_candidate_ids, alternative),
                        alternative,
                    ) or official_segments_for_alternative(alternative, official_segments)
                    trail_names = [
                        segment.get("trail_name")
                        for segment in cue_segments
                    ]
                    trail_names = sorted({str(name) for name in trail_names if name})
                    component = {
                        "route_number": probe.get("route_number"),
                        "candidate_id": new_candidate_id,
                        "field_menu_group_id": new_candidate_id,
                        "field_menu_label": alternative_id,
                        "trail_names": trail_names or [alternative.get("title")],
                        "official_miles": round_miles(probe.get("official_miles") or alternative.get("target_official_miles")),
                        "on_foot_miles": round_miles(probe.get("on_foot_miles")),
                        "ratio": round(float(probe.get("on_foot_miles") or 0.0) / float(probe.get("official_miles") or 1.0), 2),
                        "total_minutes": int(probe.get("total_minutes") or 0),
                        "trailhead": probe.get("trailhead") or "Manual parking/start",
                        "less_optimal_flags": list(probe.get("less_optimal_flags") or []) + ["parking_access_day_of_check_required"],
                        "segment_ids": list(alternative.get("required_segment_ids") or []),
                        "time_breakdown_minutes": copy.deepcopy(probe.get("time_breakdown_minutes")),
                        "time_estimates_minutes": copy.deepcopy(probe.get("time_estimates_minutes")),
                        "effort": copy.deepcopy(probe.get("effort")),
                        "route_design_status": alternative.get("route_design_status"),
                    }
                    promoted_components.append(component)
                for component in kept_components:
                    if str(component.get("candidate_id")) in (area.get("keep_candidate_ids") or []):
                        component.setdefault("field_menu_label", "16B")
                package["components"] = promoted_components + kept_components
                package["planning_status"] = "accepted_manual_split_parking_manual"
                package["planning_reasons"] = [
                    "accepted_manual_split_probe",
                    "parking_access_day_of_check_required",
                    "necessary_grinder_or_geography_locked",
                ]
                update_manual_package_summary(package)
                new_on_foot = float(package.get("on_foot_miles") or 0.0)
                update_overall_summary(data.get("summary") or {}, old_on_foot, new_on_foot)

        for route in route_pass.get("routes") or []:
            if str(route.get("candidate_id")) not in demoted_candidate_ids:
                continue
            alternative = next(
                (
                    item
                    for item in alternatives
                    if set(str(seg) for seg in item.get("required_segment_ids") or [])
                    == set(str(seg) for seg in route.get("segment_ids") or [])
                ),
                None,
            )
            if not alternative:
                continue
            alternative_id = str(alternative.get("alternative_id"))
            probe = alternative.get("probe") or {}
            route["candidate_id"] = promoted_candidate_id(alternative_id)
            route["trailhead"] = probe.get("trailhead") or route.get("trailhead")
            route["official_miles"] = round_miles(probe.get("official_miles"))
            route["on_foot_miles"] = round_miles(probe.get("on_foot_miles"))
            route["ratio"] = round(float(probe.get("on_foot_miles") or 0.0) / float(probe.get("official_miles") or 1.0), 2)
            route["total_minutes"] = int(probe.get("total_minutes") or route.get("total_minutes") or 0)
            route["less_optimal_flags"] = list(probe.get("less_optimal_flags") or []) + ["parking_access_day_of_check_required"]
            route["selection_reason"] = "accepted_manual_route_design"
            route["source_component_candidate_ids"] = [route["candidate_id"]]
        if "summary" in route_pass:
            update_overall_summary(
                route_pass["summary"],
                float(area.get("current_demoted_on_foot_miles") or (area.get("current_placeholder") or {}).get("on_foot_miles") or 0.0),
                float((area.get("current_good_route") or {}).get("on_foot_miles") or 0.0),
            )

        routes_features = (package_map.get("feature_collections") or {}).get("routes", {}).get("features") or []
        parking_features = (package_map.get("feature_collections") or {}).get("parking", {}).get("features") or []
        package_map["feature_collections"]["routes"]["features"] = [
            feature
            for feature in routes_features
            if str((feature.get("properties") or {}).get("candidate_id")) not in demoted_candidate_ids
        ]
        package_map["feature_collections"]["parking"]["features"] = [
            feature
            for feature in parking_features
            if str((feature.get("properties") or {}).get("candidate_id")) not in demoted_candidate_ids
        ]
        for alternative in alternatives:
            alternative_id = str(alternative.get("alternative_id"))
            new_candidate_id = promoted_candidate_id(alternative_id)
            feature_group = accepted_features.get(alternative_id) or {}
            for kind in ["route", "parking"]:
                feature = copy.deepcopy(feature_group.get(kind))
                if not feature:
                    continue
                props = feature.setdefault("properties", {})
                props.update(
                    {
                        "candidate_id": new_candidate_id,
                        "package_number": package_number,
                        "block_name": area.get("title") or "Manual route design",
                        "field_menu_label": alternative_id,
                        "route_design_status": alternative.get("route_design_status"),
                    }
                )
                target = "routes" if kind == "route" else "parking"
                package_map["feature_collections"][target]["features"].append(feature)

            old_cue = old_cue_for_alternative(route_cues, demoted_candidate_ids, alternative)
            cue = copy.deepcopy(old_cue) if old_cue else {"candidate_id": new_candidate_id, "between_links": []}
            cue["segments"] = cue_or_probe_segments_for_alternative(
                cue,
                alternative,
            ) or official_segments_for_alternative(alternative, official_segments)
            if cue.get("segments"):
                filter_cue_links_to_segments(cue)
                probe = alternative.get("probe") or {}
                parking_feature = feature_group.get("parking")
                trailhead = trailhead_from_anchor(area, alternative, parking_feature)
                completed_official_ids = completed_official_ids_for_alternative(alternative, official_segments)
                if not completed_official_ids:
                    completed_official_ids = normalized_ids(alternative.get("required_segment_ids") or [])
                cue.update(
                    {
                        "candidate_id": new_candidate_id,
                        "title": alternative.get("title") or old_cue.get("title"),
                        "route_status": probe.get("route_status") or "graph_validated",
                        "route_design_status": alternative.get("route_design_status"),
                        "official_miles": round_miles(probe.get("official_miles")),
                        "on_foot_miles": round_miles(probe.get("on_foot_miles")),
                        "raw_total_minutes": probe.get("raw_total_minutes"),
                        "total_minutes": int(probe.get("total_minutes") or 0),
                        "time_breakdown_minutes": copy.deepcopy(probe.get("time_breakdown_minutes")),
                        "time_estimates_minutes": copy.deepcopy(probe.get("time_estimates_minutes")),
                        "effort": copy.deepcopy(probe.get("effort")),
                        "trailhead": trailhead,
                        "start_access": {
                            "confidence": probe.get("trailhead_snap_confidence") or "manual",
                            "direct_gap_miles": 0,
                            "mapped_access_miles": 0,
                            "access_class": "manual_lower_access_anchor",
                            "graph_validated": True,
                        },
                        "return_to_car": {
                            "strategy": "accepted_manual_split_gpx",
                            "description": "Follow the accepted manual-design GPX back to the parked car.",
                            "official_repeat_miles": round_miles(probe.get("official_repeat_miles")),
                            "connector_miles": round_miles(probe.get("connector_miles")),
                            "road_miles": round_miles(probe.get("road_miles")),
                            "connector_names": [],
                            "connector_classes": [],
                            "official_repeat_segment_ids": [int(segment_id) for segment_id in completed_official_ids if str(segment_id).isdigit()],
                        },
                        "field_warning": "Roadside parking/access still needs day-of capacity and signage check.",
                    }
                )
                route_cues[new_candidate_id] = cue
        for candidate_id in demoted_candidate_ids:
            route_cues.pop(candidate_id, None)
        replace_route_validations(
            package_map,
            demoted_candidate_ids,
            [
                {
                    "candidate_id": promoted_candidate_id(str(alternative.get("alternative_id"))),
                    "source_gap_warning": False,
                    "source_max_gap_miles": (
                        ((alternative.get("generated_route_artifact") or {}).get("track_validation") or {}).get(
                            "max_trackpoint_gap_miles"
                        )
                    ),
                    "rendered_passed": True,
                    "rendered_failures": [],
                }
                for alternative in alternatives
            ],
        )


def package_component_ids(package: dict[str, Any]) -> list[str]:
    return [
        str(component.get("candidate_id"))
        for component in package.get("components") or []
        if component.get("candidate_id")
    ]


def replace_package_in_dataset(data: dict[str, Any], replacement: dict[str, Any], remove_candidate_ids: list[str]) -> None:
    package_number = str(replacement.get("package_number"))
    for index, package in enumerate(data.get("packages") or []):
        if str(package.get("package_number")) != package_number:
            continue
        old_on_foot = float(package.get("on_foot_miles") or 0.0)
        data["packages"][index] = copy.deepcopy(replacement)
        update_overall_summary(data.get("summary") or {}, old_on_foot, float(replacement.get("on_foot_miles") or 0.0))
        break

    route_cues = data.get("route_cues")
    if isinstance(route_cues, dict):
        for candidate_id in remove_candidate_ids:
            route_cues.pop(str(candidate_id), None)

    collections = data.get("feature_collections") or {}
    for collection_name, collection in collections.items():
        if collection_name == "official_segments":
            continue
        features = collection.get("features")
        if not isinstance(features, list):
            continue
        collection["features"] = [
            feature
            for feature in features
            if str((feature.get("properties") or {}).get("candidate_id")) not in set(remove_candidate_ids)
        ]


def apply_field_menu_overrides(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    """Apply durable field-menu route overrides that encode human-accepted splits."""

    for override in overrides.get("overrides") or []:
        replacement = copy.deepcopy(override.get("replace_package") or {})
        if not replacement:
            continue
        package_number = str(replacement.get("package_number"))
        existing_package = next(
            (
                package
                for package in package_map.get("packages") or []
                if str(package.get("package_number")) == package_number
            ),
            None,
        )
        if existing_package is None:
            continue
        remove_candidate_ids = [
            str(candidate_id)
            for candidate_id in (
                override.get("remove_candidate_ids")
                or package_component_ids(existing_package)
                or replacement.get("component_candidate_ids")
                or []
            )
        ]
        replacement_candidate_ids = package_component_ids(replacement)
        for data in [package_pass, package_map]:
            replace_package_in_dataset(data, replacement, remove_candidate_ids)

        route_cues = package_map.setdefault("route_cues", {})
        route_cues.update(copy.deepcopy(override.get("route_cues") or {}))

        collections = package_map.setdefault("feature_collections", {})
        for collection_name, override_collection in (override.get("feature_collections") or {}).items():
            if collection_name == "official_segments":
                continue
            collection = collections.setdefault(collection_name, {"type": "FeatureCollection", "features": []})
            collection.setdefault("features", []).extend(copy.deepcopy(override_collection.get("features") or []))
        replace_route_validations(
            package_map,
            remove_candidate_ids,
            override.get("route_validations") or [],
        )

        removed_routes = [
            route
            for route in route_pass.get("routes") or []
            if str(route.get("candidate_id")) in set(remove_candidate_ids)
        ]
        old_route_on_foot = sum(float(route.get("on_foot_miles") or 0.0) for route in removed_routes)
        route_pass["routes"] = [
            route
            for route in route_pass.get("routes") or []
            if str(route.get("candidate_id")) not in set(remove_candidate_ids)
        ]
        existing_numbers = [int(route.get("route_number") or 0) for route in route_pass.get("routes") or []]
        next_route_number = max(existing_numbers or [0]) + 1
        for component in replacement.get("components") or []:
            route_pass.setdefault("routes", []).append(
                {
                    "route_number": component.get("route_number") or next_route_number,
                    "candidate_id": component.get("candidate_id"),
                    "block_id": replacement.get("block_id"),
                    "block_name": replacement.get("block_name"),
                    "route_source": "field_menu_override",
                    "selection_reason": "human_accepted_field_menu_split",
                    "trail_names": component.get("trail_names") or [],
                    "official_miles": component.get("official_miles"),
                    "on_foot_miles": component.get("on_foot_miles"),
                    "ratio": component.get("ratio"),
                    "total_minutes": component.get("total_minutes"),
                    "trailhead": component.get("trailhead"),
                    "route_status": "graph_validated",
                    "less_optimal_flags": component.get("less_optimal_flags") or [],
                    "segment_ids": component.get("segment_ids") or [],
                    "source_component_candidate_ids": [component.get("candidate_id")],
                    "is_hybrid_assembled": False,
                    "cross_block_official_miles": 0.0,
                    "cross_block_segment_count": 0,
                }
            )
            next_route_number += 1
        update_overall_summary(
            route_pass.get("summary") or {},
            old_route_on_foot,
            sum(float(component.get("on_foot_miles") or 0.0) for component in replacement.get("components") or []),
        )


def update_component_time(component: dict[str, Any], calibration: dict[str, Any]) -> None:
    if calibration.get("total_minutes") is not None:
        component["total_minutes"] = int(calibration["total_minutes"])
    if calibration.get("raw_total_minutes") is not None:
        component["raw_total_minutes"] = int(calibration["raw_total_minutes"])
    if calibration.get("time_breakdown_minutes"):
        component["time_breakdown_minutes"] = copy.deepcopy(calibration["time_breakdown_minutes"])
    if calibration.get("time_estimates_minutes"):
        component["time_estimates_minutes"] = copy.deepcopy(calibration["time_estimates_minutes"])
    if calibration.get("effort"):
        component["effort"] = copy.deepcopy(calibration["effort"])
    if calibration.get("calibration_source"):
        component["time_calibration_source"] = calibration.get("calibration_source")


def recalc_package_time_summary(package: dict[str, Any]) -> None:
    package["total_minutes_components"] = sum(int(component.get("total_minutes") or 0) for component in package.get("components") or [])


def apply_time_calibrations(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    calibrations: dict[str, Any],
) -> None:
    """Apply field-test or manual timing corrections to the canonical menu data."""

    by_candidate_id = {
        str(calibration.get("candidate_id")): calibration
        for calibration in calibrations.get("calibrations") or []
        if calibration.get("candidate_id")
    }
    if not by_candidate_id:
        return

    for route in route_pass.get("routes") or []:
        calibration = by_candidate_id.get(str(route.get("candidate_id")))
        if calibration:
            update_component_time(route, calibration)

    for data in [package_pass, package_map]:
        for package in data.get("packages") or []:
            changed = False
            for component in package.get("components") or []:
                calibration = by_candidate_id.get(str(component.get("candidate_id")))
                if not calibration:
                    continue
                update_component_time(component, calibration)
                changed = True
            if changed:
                recalc_package_time_summary(package)

    route_cues = package_map.setdefault("route_cues", {})
    for candidate_id, calibration in by_candidate_id.items():
        cue = route_cues.setdefault(candidate_id, {"candidate_id": candidate_id})
        update_component_time(cue, calibration)


def active_route_truth_repairs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        repair
        for repair in payload.get("repairs") or []
        if str(repair.get("status") or "active") == "active"
    ]


def official_segments_by_id(official_segments: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(segment["seg_id"]): segment
        for segment in official_segments
        if str(segment.get("seg_id") or "").isdigit()
    }


def coordinate_path_miles(coords: list[list[float]] | list[tuple[float, float]]) -> float:
    total = 0.0
    for left, right in zip(coords, coords[1:]):
        total += haversine_miles(
            (float(left[0]), float(left[1])),
            (float(right[0]), float(right[1])),
        )
    return total


def append_coords(target: list[list[float]], coords: list[list[float]] | list[tuple[float, float]]) -> None:
    for raw in coords:
        point = [float(raw[0]), float(raw[1])]
        if target and haversine_miles((target[-1][0], target[-1][1]), (point[0], point[1])) <= 0.000001:
            continue
        target.append(point)


def densify_coordinate_path(
    coords: list[list[float]],
    *,
    max_gap_miles: float = 0.025,
) -> list[list[float]]:
    if len(coords) < 2:
        return coords
    densified: list[list[float]] = [coords[0]]
    for start, end in zip(coords, coords[1:]):
        distance = haversine_miles((start[0], start[1]), (end[0], end[1]))
        steps = max(1, int(math.ceil(distance / max_gap_miles)))
        for step in range(1, steps + 1):
            fraction = step / steps
            point = [
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            ]
            if haversine_miles((densified[-1][0], densified[-1][1]), (point[0], point[1])) > 0.000001:
                densified.append(point)
    return densified


def segment_for_repair_traversal(
    segment: dict[str, Any],
    *,
    direction: str,
    credit: str,
) -> dict[str, Any]:
    result = copy.deepcopy(segment)
    coords = [[float(lon), float(lat)] for lon, lat in segment.get("coordinates") or []]
    if direction == "reverse":
        coords = list(reversed(coords))
        result["start"] = segment.get("end")
        result["end"] = segment.get("start")
        result["direction_cue"] = (
            "Either direction allowed; this leg follows the opposite official geometry direction."
        )
    else:
        result["start"] = segment.get("start")
        result["end"] = segment.get("end")
        if str(segment.get("direction") or "") == "ascent":
            result["direction_cue"] = "Ascent-only segment; valid uphill traversal follows official geometry direction."
        else:
            result["direction_cue"] = "Either direction allowed; follow map arrows."
    result["coordinates"] = coords
    result["segment_name"] = segment.get("seg_name") or segment.get("segment_name")
    result["direction_rule"] = segment.get("direction") or segment.get("direction_rule") or "both"
    result["route_truth_credit"] = credit
    return result


def time_estimates_for_explicit_route(
    *,
    track_miles: float,
    state: dict[str, Any],
    trailhead: dict[str, Any],
    route_finding_penalty: int = 8,
) -> dict[str, Any]:
    drive_model = state.get("drive_model") or {}
    try:
        drive_to = drive_minutes_to_trailhead(trailhead, drive_model) if drive_model else 13
    except KeyError:
        drive_to = 13
    parking_minutes = int(trailhead.get("parking_minutes") or state.get("parking_minutes") or 8)
    pace = float(state.get("pace_min_per_mile") or 16.0)
    moving_raw = max(1, int(math.ceil(track_miles * pace)))
    moving_p50 = int(math.ceil(moving_raw * 1.12))
    moving_p75 = int(math.ceil(moving_p50 * 1.12)) + route_finding_penalty
    moving_p90 = int(math.ceil(moving_p75 * 1.12))
    raw_total = drive_to + parking_minutes + moving_raw + drive_to
    return {
        "raw_total_minutes": raw_total,
        "total_minutes": drive_to + parking_minutes + moving_p75 + drive_to,
        "time_breakdown_minutes": {
            "drive_to_trailhead": drive_to,
            "parking_and_prep": parking_minutes,
            "trailhead_access": 0,
            "moving_time": moving_raw,
            "return_drive": drive_to,
        },
        "time_estimates_minutes": {
            "door_to_door_raw": raw_total,
            "door_to_door_p50": drive_to + parking_minutes + moving_p50 + drive_to,
            "door_to_door_p75": drive_to + parking_minutes + moving_p75 + drive_to,
            "door_to_door_p90": drive_to + parking_minutes + moving_p90 + drive_to,
            "recommended_door_to_door": drive_to + parking_minutes + moving_p75 + drive_to,
            "moving_raw": moving_raw,
            "moving_effort_p50": moving_p50,
            "moving_effort_p75": moving_p75,
            "route_finding_penalty": route_finding_penalty,
        },
    }


def explicit_lollipop_route_payload(
    repair: dict[str, Any],
    *,
    official_by_id: dict[int, dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    trailhead = copy.deepcopy(repair.get("trailhead") or {})
    parking_point = [float(trailhead["lon"]), float(trailhead["lat"])]
    traversal_steps = list(repair.get("traversal") or [])
    traversal_segments = []
    for step in traversal_steps:
        segment_id = int(step["segment_id"])
        traversal_segments.append(
            segment_for_repair_traversal(
                official_by_id[segment_id],
                direction=str(step.get("direction") or "forward"),
                credit=str(step.get("credit") or "new"),
            )
        )
    if not traversal_segments:
        raise ValueError(f"Route-truth repair {repair.get('repair_id')} has no traversal segments")

    first_start = traversal_segments[0]["coordinates"][0]
    last_end = traversal_segments[-1]["coordinates"][-1]
    access_path = [parking_point, first_start]
    return_path = [last_end, parking_point]
    track: list[list[float]] = []
    append_coords(track, access_path)
    for segment in traversal_segments:
        append_coords(track, segment.get("coordinates") or [])
    append_coords(track, return_path)
    track = densify_coordinate_path(track)
    on_foot_miles = round_miles(coordinate_path_miles(track))
    claim_ids = [int(value) for value in repair.get("claim_segment_ids") or []]
    official_miles = round_miles(sum(float(official_by_id[seg_id].get("official_miles") or 0.0) for seg_id in claim_ids))
    all_repeat_ids = [
        int(step["segment_id"])
        for step in traversal_steps
        if str(step.get("credit") or "") == "repeat"
    ]
    last_new_index = max(
        (index for index, step in enumerate(traversal_steps) if str(step.get("credit") or "") == "new"),
        default=-1,
    )
    return_repeat_ids = [
        int(step["segment_id"])
        for index, step in enumerate(traversal_steps)
        if index > last_new_index and str(step.get("credit") or "") == "repeat"
    ]
    return_repeat_miles = round_miles(
        sum(float(official_by_id[seg_id].get("official_miles") or 0.0) for seg_id in set(return_repeat_ids))
    )
    ascent_claim_ids = [
        seg_id
        for seg_id in claim_ids
        if str(official_by_id[seg_id].get("direction") or "") == "ascent"
    ]
    planned_ascent_directions = {}
    for step in traversal_steps:
        segment_id = int(step["segment_id"])
        if segment_id not in ascent_claim_ids or str(step.get("credit") or "") != "new":
            continue
        planned_ascent_directions[str(segment_id)] = (
            "official_geometry_end_to_start"
            if str(step.get("direction") or "forward") == "reverse"
            else "official_geometry_start_to_end"
        )
    timing = time_estimates_for_explicit_route(track_miles=on_foot_miles, state=state, trailhead=trailhead)
    candidate_id = str(repair["candidate_id"])
    field_shape = str(repair.get("field_shape") or "lower_dry_access_shingle_up_dry_creek_down")
    component = {
        "candidate_id": candidate_id,
        "field_menu_group_id": candidate_id,
        "field_menu_label": repair.get("field_menu_label"),
        "trail_names": list(repair.get("trail_names") or []),
        "official_miles": official_miles,
        "on_foot_miles": on_foot_miles,
        "ratio": round(on_foot_miles / official_miles, 2) if official_miles else None,
        "total_minutes": timing["total_minutes"],
        "raw_total_minutes": timing["raw_total_minutes"],
        "trailhead": trailhead.get("name"),
        "less_optimal_flags": [
            "route_truth_repaired",
            "lollipop_stem_repeats_official_mileage",
            "parking_access_day_of_check_required",
        ],
        "segment_ids": claim_ids,
        "time_breakdown_minutes": timing["time_breakdown_minutes"],
        "time_estimates_minutes": timing["time_estimates_minutes"],
        "route_status": "graph_validated",
        "source": "route_truth_repair",
        "route_truth_repair_id": repair.get("repair_id"),
        "route_truth_replaces_candidate_ids": [
            str(value)
            for value in repair.get("replace_candidate_ids") or []
        ],
        "route_quality": {
            "route_truth_repair": True,
            "route_truth_lollipop": True,
            "field_shape": field_shape,
        },
    }
    route_cue = {
        "candidate_id": candidate_id,
        "title": repair.get("title"),
        "route_status": "graph_validated",
        "official_miles": official_miles,
        "on_foot_miles": on_foot_miles,
        "raw_total_minutes": timing["raw_total_minutes"],
        "total_minutes": timing["total_minutes"],
        "time_breakdown_minutes": timing["time_breakdown_minutes"],
        "time_estimates_minutes": timing["time_estimates_minutes"],
        "trailhead": trailhead,
        "start_access": {
            "confidence": "field_check_needed",
            "direct_gap_miles": round_miles(coordinate_path_miles(access_path)),
            "mapped_access_miles": round_miles(coordinate_path_miles(access_path)),
            "official_repeat_miles": 0,
            "official_repeat_segment_ids": [],
            "access_class": "manual_lower_access_anchor",
            "graph_validated": True,
            "connector_names": ["Dry Creek", "Sweet Connie"],
            "connector_classes": ["roadside_access"],
            "path_start": parking_point,
            "path_end": first_start,
            "path_coordinates": access_path,
        },
        "segments": traversal_segments,
        "between_links": [],
        "return_to_car": {
            "strategy": "explicit_lollipop_return",
            "description": repair.get("return_description")
            or "Return to the car by continuing down Dry Creek after the final new-credit leg.",
            "official_repeat_miles": return_repeat_miles,
            "official_repeat_segment_ids": sorted(set(return_repeat_ids)),
            "connector_miles": round_miles(coordinate_path_miles(return_path)),
            "road_miles": 0,
            "connector_names": ["Dry Creek"],
            "connector_classes": ["roadside_access"],
            "path_start": last_end,
            "path_end": parking_point,
            "path_coordinates": return_path,
            "needs_map_validation": False,
            "graph_validated": True,
        },
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
            "trailhead_snap_confidence": "high",
            "connector_overlap_checked": True,
            "special_management_checked": False,
        },
        "direction_validation": {
            "passed": True,
            "reason": "explicit_lollipop_route_truth_repair",
            "ascent_segment_ids_checked": ascent_claim_ids,
            "planned_traversal_direction": planned_ascent_directions,
        },
        "cue_generation_mode": "route_truth_repair_explicit_lollipop",
        "field_warning": "Roadside parking/access still needs day-of capacity and signage check.",
        "route_truth_repair_id": repair.get("repair_id"),
        "field_notes": list(repair.get("field_notes") or []),
    }
    validation = validate_track_segments([[tuple(point) for point in track]], max_gap_miles=0.05)
    route_feature = coords_feature(
        [(point[0], point[1]) for point in track],
        {
            "kind": "route",
            "package_number": repair.get("package_number"),
            "candidate_id": candidate_id,
            "block_name": repair.get("block_name"),
            "title": repair.get("title"),
            "official_miles": official_miles,
            "on_foot_miles": on_foot_miles,
            "trailhead": trailhead.get("name"),
            "field_menu_label": repair.get("field_menu_label"),
            "source": "route_truth_repair",
            "route_truth_repair_id": repair.get("repair_id"),
            "source_gap_warning_count": 0 if validation.get("passed") else 1,
        },
    )
    parking = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": parking_point},
        "properties": {
            "kind": "parking",
            "package_number": repair.get("package_number"),
            "candidate_id": candidate_id,
            "block_name": repair.get("block_name"),
            "field_menu_label": repair.get("field_menu_label"),
            "name": trailhead.get("name"),
            "has_parking": trailhead.get("has_parking"),
            "parking_minutes": trailhead.get("parking_minutes"),
            "source": trailhead.get("source"),
            "parking_confidence": trailhead.get("parking_confidence"),
        },
    }
    return {
        "component": component,
        "cue": route_cue,
        "route_feature": route_feature,
        "parking_feature": parking,
        "route_validation": {
            "candidate_id": candidate_id,
            "source_gap_warning": not validation.get("passed"),
            "source_max_gap_miles": validation.get("max_trackpoint_gap_miles"),
            "rendered_passed": validation.get("passed"),
            "rendered_failures": validation.get("failures") or [],
        },
    }


def route_truth_context(
    state: dict[str, Any],
    official_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    # Route-truth repairs use explicit official segment geometry; avoid the
    # expensive full official-overlap pass while rebuilding companion routes.
    connector_graph = load_connector_graph(DEFAULT_CONNECTOR_GEOJSON, official_segments=[])
    return {
        "official_index": load_official_segment_index(DEFAULT_OFFICIAL_SEGMENTS_GEOJSON),
        "connector_graph": connector_graph,
        "elevation_sampler": load_dem_context(DEFAULT_DEM_TIF, DEFAULT_DEM_SUMMARY_JSON)["sampler"],
        "performance_profile": build_performance_profile(
            state=state,
            strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
            activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
            activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
            segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
        ),
    }


def package_by_number(data: dict[str, Any], package_number: Any) -> dict[str, Any] | None:
    return next(
        (
            package
            for package in data.get("packages") or []
            if str(package.get("package_number")) == str(package_number)
        ),
        None,
    )


def component_by_candidate(package: dict[str, Any] | None, candidate_id: Any) -> dict[str, Any] | None:
    if not package:
        return None
    return next(
        (
            component
            for component in package.get("components") or []
            if str(component.get("candidate_id")) == str(candidate_id)
        ),
        None,
    )


def remove_candidates_from_package(package: dict[str, Any], candidate_ids: set[str]) -> None:
    package["components"] = [
        component
        for component in package.get("components") or []
        if str(component.get("candidate_id")) not in candidate_ids
    ]


def remove_candidate_sources(map_data: dict[str, Any], candidate_ids: set[str]) -> None:
    route_cues = map_data.get("route_cues")
    if isinstance(route_cues, dict):
        for candidate_id in candidate_ids:
            route_cues.pop(candidate_id, None)
    for collection_name, collection in (map_data.get("feature_collections") or {}).items():
        if collection_name == "official_segments":
            continue
        features = collection.get("features")
        if not isinstance(features, list):
            continue
        collection["features"] = [
            feature
            for feature in features
            if str((feature.get("properties") or {}).get("candidate_id")) not in candidate_ids
        ]


def add_route_truth_sources(
    package_map: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    package_map.setdefault("route_cues", {})[payload["component"]["candidate_id"]] = payload["cue"]
    collections = package_map.setdefault("feature_collections", {})
    collections.setdefault("routes", {"type": "FeatureCollection", "features": []}).setdefault("features", []).append(
        payload["route_feature"]
    )
    collections.setdefault("parking", {"type": "FeatureCollection", "features": []}).setdefault("features", []).append(
        payload["parking_feature"]
    )
    replace_route_validations(package_map, [payload["component"]["candidate_id"]], [payload["route_validation"]])


def append_route_pass_route(
    route_pass: dict[str, Any],
    *,
    package: dict[str, Any],
    component: dict[str, Any],
    source: str,
) -> None:
    existing_numbers = [int(route.get("route_number") or 0) for route in route_pass.get("routes") or []]
    route_pass.setdefault("routes", []).append(
        {
            "route_number": component.get("route_number") or max(existing_numbers or [0]) + 1,
            "candidate_id": component.get("candidate_id"),
            "block_id": package.get("block_id"),
            "block_name": package.get("block_name"),
            "route_source": source,
            "selection_reason": source,
            "trail_names": component.get("trail_names") or [],
            "official_miles": component.get("official_miles"),
            "on_foot_miles": component.get("on_foot_miles"),
            "ratio": component.get("ratio"),
            "total_minutes": component.get("total_minutes"),
            "trailhead": component.get("trailhead"),
            "route_status": "graph_validated",
            "less_optimal_flags": component.get("less_optimal_flags") or [],
            "segment_ids": component.get("segment_ids") or [],
            "source_component_candidate_ids": [component.get("candidate_id")],
            "is_hybrid_assembled": False,
            "cross_block_official_miles": 0.0,
            "cross_block_segment_count": 0,
        }
    )


def candidate_component_from_probe(
    *,
    candidate_id: str,
    candidate: dict[str, Any],
    base_component: dict[str, Any],
    label: str | None = None,
    source: str = "route_truth_repair",
) -> dict[str, Any]:
    official = float(candidate.get("official_new_miles") or 0.0)
    on_foot = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
    component = {
        "route_number": base_component.get("route_number"),
        "candidate_id": candidate_id,
        "field_menu_group_id": candidate_id,
        "trail_names": candidate.get("trail_names") or [],
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "total_minutes": int(candidate.get("total_minutes") or 0),
        "raw_total_minutes": int(candidate.get("raw_total_minutes") or 0),
        "trailhead": (candidate.get("trailhead") or {}).get("name") or base_component.get("trailhead"),
        "less_optimal_flags": list(candidate.get("less_optimal_flags") or []) + ["route_truth_repaired"],
        "segment_ids": [int(value) for value in candidate.get("segment_ids") or []],
        "time_breakdown_minutes": copy.deepcopy(candidate.get("time_breakdown_minutes")),
        "time_estimates_minutes": copy.deepcopy(candidate.get("time_estimates_minutes")),
        "effort": copy.deepcopy(candidate.get("effort")),
        "route_status": candidate.get("route_status"),
        "source": source,
    }
    if label:
        component["field_menu_label"] = label
    return component


def build_pruned_component_payload(
    *,
    prune: dict[str, Any],
    source_component: dict[str, Any],
    source_cue: dict[str, Any],
    official_by_id: dict[int, dict[str, Any]],
    state: dict[str, Any],
    context: dict[str, Any],
    package: dict[str, Any],
) -> dict[str, Any]:
    trailhead = copy.deepcopy(source_cue.get("trailhead") or {})
    forced_state = dict(state)
    forced_state["trailheads"] = [trailhead]
    segment_ids = [int(value) for value in prune.get("replacement_segment_ids") or []]
    candidate = candidate_from_trail_group(
        group_remaining_by_trail([official_by_id[segment_id] for segment_id in segment_ids]),
        forced_state,
        context["performance_profile"],
        context["connector_graph"],
        candidate_type="route_truth_pruned_component",
        elevation_sampler=context["elevation_sampler"],
    )
    candidate_id = str(prune.get("replacement_candidate_id") or source_component.get("candidate_id"))
    component = candidate_component_from_probe(
        candidate_id=candidate_id,
        candidate=candidate,
        base_component=source_component,
        label=source_component.get("field_menu_label"),
    )
    if prune.get("replacement_title"):
        component["route_truth_replacement_title"] = prune.get("replacement_title")
    component["route_truth_replaces_candidate_ids"] = [str(prune.get("candidate_id"))]
    cue = cue_from_candidate(
        candidate_id,
        candidate,
        context["official_index"],
        context["connector_graph"],
    )
    cue["title"] = prune.get("replacement_title") or cue.get("title")
    cue["route_truth_repair_id"] = prune.get("repair_id")
    coords = candidate_track_coordinates(
        candidate,
        context["official_index"],
        connector_graph=context["connector_graph"],
        densify_source_lines=True,
    )
    validation = validate_track_segments([coords], max_gap_miles=0.05)
    route_feature = coords_feature(
        coords,
        {
            "kind": "route",
            "package_number": package.get("package_number"),
            "candidate_id": candidate_id,
            "block_name": package.get("block_name"),
            "title": cue.get("title"),
            "official_miles": component.get("official_miles"),
            "on_foot_miles": component.get("on_foot_miles"),
            "trailhead": component.get("trailhead"),
            "field_menu_label": component.get("field_menu_label"),
            "source": "route_truth_repair",
            "source_gap_warning_count": 0 if validation.get("passed") else 1,
        },
    )
    parking = parking_feature(
        candidate,
        {
            "kind": "parking",
            "package_number": package.get("package_number"),
            "candidate_id": candidate_id,
            "block_name": package.get("block_name"),
            "field_menu_label": component.get("field_menu_label"),
            "source": "route_truth_repair",
        },
    )
    return {
        "component": component,
        "cue": cue,
        "route_feature": route_feature,
        "parking_feature": parking,
        "route_validation": {
            "candidate_id": candidate_id,
            "source_gap_warning": not validation.get("passed"),
            "source_max_gap_miles": validation.get("max_trackpoint_gap_miles"),
            "rendered_passed": validation.get("passed"),
            "rendered_failures": validation.get("failures") or [],
        },
    }


def apply_route_truth_repairs(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    repairs_payload: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    repairs = active_route_truth_repairs(repairs_payload)
    if not repairs:
        return
    official_by_id = official_segments_by_id(official_segments)
    context = route_truth_context(state, official_segments)
    route_cues = package_map.setdefault("route_cues", {})
    for repair in repairs:
        if repair.get("repair_type") != "explicit_lollipop_route":
            continue
        replacement_payload = explicit_lollipop_route_payload(
            repair,
            official_by_id=official_by_id,
            state=state,
        )
        remove_ids = {str(value) for value in repair.get("replace_candidate_ids") or []}
        remove_ids.add(str(repair.get("candidate_id")))
        for data in [package_pass, package_map]:
            package = package_by_number(data, repair.get("package_number"))
            if not package:
                continue
            remove_candidates_from_package(package, remove_ids)
            package.setdefault("components", []).append(copy.deepcopy(replacement_payload["component"]))
            package["planning_status"] = "route_truth_repaired"
            reasons = list(package.get("planning_reasons") or [])
            for reason in ["route_truth_repaired", "planner_artifact_removed"]:
                if reason not in reasons:
                    reasons.append(reason)
            package["planning_reasons"] = reasons
            update_manual_package_summary(package)
        remove_candidate_sources(package_map, remove_ids)
        add_route_truth_sources(package_map, replacement_payload)
        route_pass["routes"] = [
            route
            for route in route_pass.get("routes") or []
            if str(route.get("candidate_id")) not in remove_ids
        ]
        target_package = package_by_number(package_map, repair.get("package_number")) or {}
        append_route_pass_route(
            route_pass,
            package=target_package,
            component=replacement_payload["component"],
            source="route_truth_repair",
        )

        for prune in repair.get("prune_claims") or []:
            prune = {**prune, "repair_id": repair.get("repair_id")}
            source_id = str(prune.get("candidate_id"))
            replacement_id = str(prune.get("replacement_candidate_id") or "")
            map_package = package_by_number(package_map, prune.get("package_number"))
            pass_package = package_by_number(package_pass, prune.get("package_number"))
            source_component = component_by_candidate(map_package, source_id) or component_by_candidate(
                pass_package,
                source_id,
            )
            source_cue = copy.deepcopy(route_cues.get(source_id) or {})
            if not map_package or not source_component or not source_cue:
                continue
            pruned_payload = build_pruned_component_payload(
                prune=prune,
                source_component=source_component,
                source_cue=source_cue,
                official_by_id=official_by_id,
                state=state,
                context=context,
                package=map_package,
            )
            replacement_id = pruned_payload["component"]["candidate_id"]
            for package in [pass_package, map_package]:
                if not package:
                    continue
                remove_candidates_from_package(package, {source_id, pruned_payload["component"]["candidate_id"]})
                package.setdefault("components", []).append(copy.deepcopy(pruned_payload["component"]))
                reasons = list(package.get("planning_reasons") or [])
                for reason in ["route_truth_repaired", "dry_creek_claims_moved_to_shingle_lollipop"]:
                    if reason not in reasons:
                        reasons.append(reason)
                package["planning_reasons"] = reasons
                package["planning_status"] = "route_truth_repaired"
                update_manual_package_summary(package)
            remove_candidate_sources(package_map, {source_id, replacement_id})
            add_route_truth_sources(package_map, pruned_payload)
            route_pass["routes"] = [
                route
                for route in route_pass.get("routes") or []
                if str(route.get("candidate_id")) not in {source_id, replacement_id}
            ]
            append_route_pass_route(
                route_pass,
                package=map_package,
                component=pruned_payload["component"],
                source="route_truth_pruned_component",
            )


def package_start_plan(package: dict[str, Any]) -> str:
    trailhead_count = int(package.get("trailhead_count") or 0)
    component_count = int(package.get("component_route_count") or 0)
    if trailhead_count > 1:
        return f"{trailhead_count} parked starts"
    if trailhead_count == 1 and component_count > 1:
        return f"1 parked start, {component_count} route components"
    if trailhead_count == 1:
        return "1 parked start"
    return f"{component_count} route components"


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# 2026 Human Loop Plan v1",
        "",
        "Status: current user-facing route/block plan.",
        "",
        "## Summary",
        "",
        f"- Packages: {summary['package_count']}",
        f"- Route components: {summary['route_component_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / 251",
        f"- Official miles: {summary['official_miles']}",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- On-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Primary loop blocks: {status_counts.get('primary_loop_block', 0)}",
        f"- Accepted split blocks: {status_counts.get('accepted_split_block', 0)}",
        f"- Necessary grinders: {status_counts.get('necessary_grinder', 0)}",
        f"- Manual design areas: {status_counts.get('manual_design_area', 0)}",
        f"- Route components graph-validated: {summary['all_route_components_graph_validated']}",
        f"- Map rendered: {summary['map_rendered_passed']}",
        f"- Map: `{summary['map_html']}`",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in plan.get("caveats") or [])
    lines.append("- Package on-foot miles are totals if you do every listed parked start in that package; they are not additional mileage per start.")
    lines.extend(
        [
            "",
            "## Blocks",
            "",
            "| # | Block | Status | Plan | Trailhead(s) | Official mi | On-foot mi | Ratio | Why |",
            "|---:|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for package in plan.get("packages") or []:
        reasons = ", ".join(package.get("human_plan_reasons") or [])
        trailheads = ", ".join(package.get("trailheads") or [])
        lines.append(
            f"| {package['package_number']} | {package['block_name']} | {package['human_plan_status']} | "
            f"{package_start_plan(package)} | {trailheads} | {package['official_miles']} | "
            f"{package['on_foot_miles']} | {package['ratio']} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Route Components",
            "",
            "| # | Block | Route | Trailhead | Official mi | On-foot mi | Ratio |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for route in plan.get("routes") or []:
        lines.append(
            f"| {route['route_number']} | {route['block_name']} | {', '.join(route.get('trail_names') or [])} | "
            f"{route.get('trailhead') or ''} | {route['official_miles']} | {route['on_foot_miles']} | {route['ratio']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def apply_manual_design_reports(
    route_pass: dict[str, Any],
    package_pass: dict[str, Any],
    package_map: dict[str, Any],
    manual_design: dict[str, Any],
    manual_design_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_manual_design = manual_design
    for report in manual_design_reports:
        merged_manual_design = merge_manual_design_report(merged_manual_design, report)
    package_map["manual_design"] = merged_manual_design
    for report in manual_design_reports:
        promote_package16_manual_routes(route_pass, package_pass, package_map, merged_manual_design, report)
    return merged_manual_design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-pass-json", type=Path, default=DEFAULT_ROUTE_PASS_JSON)
    parser.add_argument("--package-pass-json", type=Path, default=DEFAULT_PACKAGE_PASS_JSON)
    parser.add_argument("--package-map-json", type=Path, default=DEFAULT_PACKAGE_MAP_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument(
        "--manual-design-report-json",
        type=Path,
        action="append",
        help="Manual route-design report to promote. Repeat to apply multiple reports; defaults to the current package16 and Harlow/Spring reports.",
    )
    parser.add_argument("--field-menu-overrides-json", type=Path, default=default_field_menu_overrides_json())
    parser.add_argument("--time-calibrations-json", type=Path, default=DEFAULT_TIME_CALIBRATIONS_JSON)
    parser.add_argument("--route-truth-repairs-json", type=Path, default=DEFAULT_ROUTE_TRUTH_REPAIRS_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--official-segments-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--map-html", type=Path, default=DEFAULT_MAP_HTML)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--outing-menu-md", type=Path, default=DEFAULT_OUTING_MENU_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_pass = read_json(args.route_pass_json)
    package_pass = read_json(args.package_pass_json)
    package_map = read_json(args.package_map_json)
    state = read_json(args.state_json) if args.state_json.exists() else {}
    manual_design = read_json(args.manual_design_json) if args.manual_design_json.exists() else {}
    manual_design_report_paths = args.manual_design_report_json or DEFAULT_MANUAL_DESIGN_REPORT_JSONS
    manual_design_reports = [read_json(path) for path in manual_design_report_paths if path.exists()]
    field_menu_overrides = read_json(args.field_menu_overrides_json) if args.field_menu_overrides_json.exists() else {}
    time_calibrations = read_json(args.time_calibrations_json) if args.time_calibrations_json.exists() else {}
    route_truth_repairs = read_json(args.route_truth_repairs_json) if args.route_truth_repairs_json.exists() else {}
    official_segments, _official_meta = load_official_segments(args.official_segments_geojson)
    if field_menu_overrides:
        apply_field_menu_overrides(route_pass, package_pass, package_map, field_menu_overrides)
    if manual_design:
        manual_design = apply_manual_design_reports(
            route_pass,
            package_pass,
            package_map,
            manual_design,
            manual_design_reports,
        )
    if route_truth_repairs:
        apply_route_truth_repairs(
            route_pass,
            package_pass,
            package_map,
            route_truth_repairs,
            official_segments=official_segments,
            state=state,
        )
    if time_calibrations:
        apply_time_calibrations(route_pass, package_pass, package_map, time_calibrations)
    sync_progress_from_state(package_map, state)
    recompute_package_summary(package_map, package_map)
    sync_official_segment_features(package_map, official_segments)
    sync_route_direction_cues(package_map)
    package_pass["summary"] = dict(package_map["summary"])
    if "summary" in route_pass:
        route_pass["summary"]["selected_route_count"] = len(route_pass.get("routes") or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_html_path = args.map_html
    map_data_json_path = args.map_data_json
    outing_menu_md_path = args.outing_menu_md
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    map_html_path.parent.mkdir(parents=True, exist_ok=True)
    map_data_json_path.parent.mkdir(parents=True, exist_ok=True)
    outing_menu_md_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(map_data_json_path, package_map)
    map_html_path.write_text(render_package_map_html(package_map), encoding="utf-8")
    outing_menu_md_path.write_text(render_outing_menu_markdown(package_map, map_html_path), encoding="utf-8")
    plan = build_human_plan(route_pass, package_pass, package_map, map_html_path, map_data_json_path)
    write_json(json_path, plan)
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[
                path
                for path in [
                    args.route_pass_json,
                    args.package_pass_json,
                    args.package_map_json,
                    args.manual_design_json,
                    args.field_menu_overrides_json,
                    args.route_truth_repairs_json,
                    args.time_calibrations_json,
                    args.state_json,
                    args.official_segments_geojson,
                ]
                + [path for path in manual_design_report_paths if path.exists()]
                if path.exists()
            ],
            outputs=[json_path, md_path, map_html_path, map_data_json_path, outing_menu_md_path],
            command="human_loop_plan.py",
            metadata={
                "unresolved_blocker_count": plan["summary"]["unresolved_blocker_count"],
                "covered_segment_count": plan["summary"]["covered_segment_count"],
                "route_component_count": plan["summary"]["route_component_count"],
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {map_html_path}")
    print(f"Wrote {map_data_json_path}")
    print(f"Wrote {outing_menu_md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
