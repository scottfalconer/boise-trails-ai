#!/usr/bin/env python3
"""Build the user-facing human loop/block plan from the combo route pass."""

from __future__ import annotations

import argparse
import copy
import json
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
from personal_route_planner import read_json  # noqa: E402


DEFAULT_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_PACKAGE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1.json"
DEFAULT_PACKAGE_MAP_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-day-package-pass-v1-map-data.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "human-loop-plan-v1"
DEFAULT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_OUTING_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_MANUAL_DESIGN_JSON = YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json"
DEFAULT_MANUAL_DESIGN_REPORT_JSON = (
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "package16-manual-route-design-v1.json"
)
DEFAULT_FIELD_MENU_OVERRIDES_JSON = YEAR_DIR / "inputs" / "personal" / "2026-field-menu-overrides-v1.json"
DEFAULT_TIME_CALIBRATIONS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-field-time-calibrations-v1.json"


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
    unresolved = []
    if package_pass.get("summary", {}).get("covered_segment_count") != 251:
        unresolved.append("coverage_not_complete")
    if route_statuses != {"graph_validated"}:
        unresolved.append("non_graph_validated_route_components")
    if map_validation.get("rendered_passed") is not True:
        unresolved.append("map_render_validation_failed")
    return {
        "planning_status": "human_loop_plan",
        "summary": {
            **(package_pass.get("summary") or {}),
            "route_component_count": route_pass.get("summary", {}).get("selected_route_count"),
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
    route_cues = package_map.get("route_cues") or {}
    for area_id, report_area in report_areas.items():
        area = {**(manual_areas.get(area_id) or {}), **report_area}
        alternatives = selected_manual_alternatives(area)
        if not alternatives:
            continue
        demoted_candidate_ids = [str(value) for value in area.get("demote_candidate_ids") or []]
        package_number = area.get("package_number")

        for data in [package_pass, package_map]:
            for package in data.get("packages") or []:
                if str(package.get("package_number")) != str(package_number):
                    continue
                old_on_foot = float(package.get("on_foot_miles") or 0.0)
                kept_components = [
                    component
                    for component in package.get("components") or []
                    if str(component.get("candidate_id")) not in demoted_candidate_ids
                ]
                promoted_components = []
                for alternative in alternatives:
                    alternative_id = str(alternative.get("alternative_id"))
                    probe = alternative.get("probe") or {}
                    new_candidate_id = promoted_candidate_id(alternative_id)
                    cue_segments = cue_segments_for_alternative(
                        old_cue_for_alternative(route_cues, demoted_candidate_ids, alternative),
                        alternative,
                    )
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
            if old_cue:
                cue = copy.deepcopy(old_cue)
                cue["segments"] = cue_segments_for_alternative(cue, alternative)
                filter_cue_links_to_segments(cue)
                probe = alternative.get("probe") or {}
                parking_feature = feature_group.get("parking")
                trailhead = trailhead_from_anchor(area, alternative, parking_feature)
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
            {},
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-pass-json", type=Path, default=DEFAULT_ROUTE_PASS_JSON)
    parser.add_argument("--package-pass-json", type=Path, default=DEFAULT_PACKAGE_PASS_JSON)
    parser.add_argument("--package-map-json", type=Path, default=DEFAULT_PACKAGE_MAP_JSON)
    parser.add_argument("--manual-design-json", type=Path, default=DEFAULT_MANUAL_DESIGN_JSON)
    parser.add_argument("--manual-design-report-json", type=Path, default=DEFAULT_MANUAL_DESIGN_REPORT_JSON)
    parser.add_argument("--field-menu-overrides-json", type=Path, default=DEFAULT_FIELD_MENU_OVERRIDES_JSON)
    parser.add_argument("--time-calibrations-json", type=Path, default=DEFAULT_TIME_CALIBRATIONS_JSON)
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
    manual_design = read_json(args.manual_design_json) if args.manual_design_json.exists() else {}
    manual_design_report = read_json(args.manual_design_report_json) if args.manual_design_report_json.exists() else {}
    field_menu_overrides = read_json(args.field_menu_overrides_json) if args.field_menu_overrides_json.exists() else {}
    time_calibrations = read_json(args.time_calibrations_json) if args.time_calibrations_json.exists() else {}
    if field_menu_overrides:
        apply_field_menu_overrides(route_pass, package_pass, package_map, field_menu_overrides)
    if manual_design:
        manual_design = merge_manual_design_report(manual_design, manual_design_report)
        package_map["manual_design"] = manual_design
        promote_package16_manual_routes(route_pass, package_pass, package_map, manual_design, manual_design_report)
    if time_calibrations:
        apply_time_calibrations(route_pass, package_pass, package_map, time_calibrations)
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
                    args.time_calibrations_json,
                ]
                + ([args.manual_design_report_json] if args.manual_design_report_json.exists() else [])
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
