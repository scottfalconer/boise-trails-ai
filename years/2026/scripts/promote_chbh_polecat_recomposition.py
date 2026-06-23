#!/usr/bin/env python3
"""Promote the CHBH/Polecat recomposition into the private field-menu source.

The promotion deliberately edits the ignored private replacement source, not
generated packet artifacts. It absorbs CHBH and Polecat Loop 5 into the lower
36th Street Chute card because, after drive/repark and prep are included, that
beats keeping a separate Cartwright mini-card. It also removes those claimed
segments from the Polecat core and Peggy's / Cartwright cards.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from export_execution_gpx import validate_track_segments  # noqa: E402
from freestone_cluster_route_generation_experiment import build_generated_route  # noqa: E402
from multi_start_field_menu_replacements import (  # noqa: E402
    feature_collections_for_candidates,
    recompute_package,
    route_cue_index,
    route_validations_for_candidates,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    build_time_estimates_minutes,
    drive_minutes_to_trailhead,
    elevation_gain_loss_for_line,
    haversine_miles,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    round_miles,
)


DEFAULT_CURRENT_MAP_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_FIELD_MENU_REPLACEMENTS_JSON = (
    YEAR_DIR
    / "inputs"
    / "personal"
    / "private"
    / "2026-field-menu-replacements-v2-multi-start.private.json"
)
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_JSON = (
    YEAR_DIR
    / "outputs"
    / "private"
    / "chbh-polecat-recomposition-2026-06-23"
    / "chbh-polecat-promotion-summary.json"
)

REPAIR_ID = "chbh-polecat-recomposition-2026-06-23"
ROUTE_TRUTH_SOURCE = "chbh_polecat_recomposition_repair"

PARKING_CANDIDATE_FALLBACKS = {
    "multi-start-1a-1a-ms-04-1-36th-street-chute": [
        "route-truth-1a-1-chute-chbh-polecat5",
    ],
    "multi-start-5-5-ms-04-2-polecat-loop-doe-ridge-quick-draw": [
        "route-truth-5b-polecat-minus-p5",
        "review-seed-5c-chbh-polecat5",
    ],
    "block-cartwright_peggy_interface": [
        "route-truth-6-cartwright-peggy-minus-chbh",
        "route-truth-5b-polecat-minus-p5",
    ],
}

GENERATED_ROUTES = {
    "route-truth-1a-1-chute-chbh-polecat5": {
        "package_number": 1,
        "field_menu_label": "1A-1",
        "title": "36th Street Chute / CHBH / Polecat Loop 5",
        "trail_names": ["36th Street Chute", "CHBH Connector", "Polecat Loop"],
        "parking_candidate_id": "multi-start-1a-1a-ms-04-1-36th-street-chute",
        "ordered_segment_ids": ["1482", "1516", "1601"],
        "strategy": "template_sequence_greedy",
        "route_truth_replaces_candidate_ids": [
            "multi-start-1a-1a-ms-04-1-36th-street-chute",
        ],
    },
    "route-truth-5b-polecat-minus-p5": {
        "package_number": 5,
        "field_menu_label": "5B",
        "title": "Polecat Loop / Doe Ridge / Quick Draw without Polecat 5",
        "trail_names": ["Polecat Loop", "Doe Ridge", "Quick Draw"],
        "parking_candidate_id": "multi-start-5-5-ms-04-2-polecat-loop-doe-ridge-quick-draw",
        "ordered_segment_ids": ["1599", "1602", "1600", "1598", "1603", "1604", "1541", "1610"],
        "strategy": "nearest_segment_greedy",
        "route_truth_replaces_candidate_ids": [
            "multi-start-5-5-ms-04-2-polecat-loop-doe-ridge-quick-draw",
        ],
    },
    "review-seed-5c-chbh-polecat5": {
        "package_number": 5,
        "field_menu_label": "5C",
        "title": "CHBH Connector and Polecat Loop 5 from Cartwright",
        "trail_names": ["CHBH Connector", "Polecat Loop"],
        "parking_candidate_id": "multi-start-5-5-ms-04-2-polecat-loop-doe-ridge-quick-draw",
        "ordered_segment_ids": ["1516", "1601"],
        "strategy": "nearest_segment_greedy",
    },
    "route-truth-6-cartwright-peggy-minus-chbh": {
        "package_number": 6,
        "field_menu_label": "6",
        "title": "Peggy's / Chukar Butte / Cartwright without CHBH",
        "trail_names": ["Peggy's Trail", "Chukar Butte Trail", "Cartwright Connector", "Cartwright Ridge"],
        "parking_candidate_id": "block-cartwright_peggy_interface",
        "ordered_segment_ids": ["1597", "1519", "1520", "1521", "1709", "1509", "1508"],
        "strategy": "template_sequence_greedy",
        "route_truth_replaces_candidate_ids": [
            "block-cartwright_peggy_interface",
        ],
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def sort_id(value: Any) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def normalized_ids(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def int_value(value: Any) -> int:
    return int(round(float(value or 0)))


def ceil_minutes(value: float) -> int:
    return int(math.ceil(max(0.0, value) - 1e-9))


def package_for_number(map_data: dict[str, Any], package_number: Any) -> dict[str, Any]:
    for package in map_data.get("packages") or []:
        if str(package.get("package_number")) == str(package_number):
            return package
    raise KeyError(f"Missing package {package_number}")


def component_by_candidate(package: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for component in package.get("components") or []:
        if str(component.get("candidate_id")) == candidate_id:
            return component
    raise KeyError(f"Missing component {candidate_id} in package {package.get('package_number')}")


def replacement_route_cues(replacements: dict[str, Any] | None) -> dict[str, Any]:
    cues: dict[str, Any] = {}
    for override in (replacements or {}).get("overrides") or []:
        for candidate_id, cue in (override.get("route_cues") or {}).items():
            cues.setdefault(str(candidate_id), cue)
    return cues


def parking_for_candidate(
    current_map: dict[str, Any],
    candidate_id: str,
    replacements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cues: dict[str, Any] = {}
    cues.update(current_map.get("route_cues") or {})
    cues.update(replacement_route_cues(replacements))
    candidate_ids = [candidate_id, *PARKING_CANDIDATE_FALLBACKS.get(candidate_id, [])]
    for cue_candidate_id in candidate_ids:
        cue = cues.get(cue_candidate_id) or {}
        trailhead = copy.deepcopy(cue.get("trailhead") or {})
        if trailhead.get("lat") is not None and trailhead.get("lon") is not None:
            trailhead.setdefault("parking_minutes", 8)
            return trailhead
    raise ValueError(f"Missing trailhead coordinates for {candidate_id}")


def point_path_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def route_finding_penalty(candidate: dict[str, Any]) -> int:
    penalty = 8
    repeat = float(candidate.get("official_repeat_miles") or 0)
    connector = float(candidate.get("connector_miles") or 0)
    if repeat >= 2.0:
        penalty += 6
    elif repeat >= 1.0:
        penalty += 4
    elif repeat >= 0.5:
        penalty += 2
    if connector >= 4.0:
        penalty += 4
    elif connector >= 2.0:
        penalty += 3
    elif connector >= 1.0:
        penalty += 2
    cue_complexity = candidate.get("cue_complexity") or {}
    if int(cue_complexity.get("connector_or_return_cue_count") or 0) >= 3:
        penalty += 2
    if candidate.get("ascent_direction_validation", {}).get("ascent_segment_ids"):
        penalty += 2
    if float(candidate.get("direct_gap_fallback_miles") or 0) > 0:
        penalty += 2
    return min(penalty, 22)


def track_effort(
    coords: list[tuple[float, float]],
    *,
    official_miles: float,
    raw_moving_minutes: int,
    elevation_sampler: Any,
) -> dict[str, Any]:
    if not elevation_sampler:
        return {
            "ascent_ft": None,
            "descent_ft": None,
            "grade_adjusted_miles": None,
            "estimated_moving_minutes_p50": raw_moving_minutes,
            "estimated_moving_minutes_p75": ceil_minutes(raw_moving_minutes * 1.12),
            "heat_risk": "unknown",
            "effort_score": raw_moving_minutes,
            "elevation_source": "unavailable",
        }
    ascent, descent, sampled = elevation_gain_loss_for_line(coords, elevation_sampler)
    if not sampled:
        return {
            "ascent_ft": None,
            "descent_ft": None,
            "grade_adjusted_miles": None,
            "estimated_moving_minutes_p50": raw_moving_minutes,
            "estimated_moving_minutes_p75": ceil_minutes(raw_moving_minutes * 1.12),
            "heat_risk": "unknown",
            "effort_score": raw_moving_minutes,
            "elevation_source": "dem_no_valid_samples",
        }
    effort_score = raw_moving_minutes + ascent / 100
    return {
        "ascent_ft": round(ascent),
        "descent_ft": round(descent),
        "grade_adjusted_miles": round_miles(official_miles + ascent / 1000),
        "estimated_moving_minutes_p50": raw_moving_minutes,
        "estimated_moving_minutes_p75": ceil_minutes(effort_score * 1.12),
        "heat_risk": "unknown",
        "effort_score": ceil_minutes(effort_score),
        "elevation_source": "dem",
    }


def price_generated_candidate(
    candidate: dict[str, Any],
    *,
    state: dict[str, Any],
    trailhead: dict[str, Any],
    elevation_sampler: Any,
) -> dict[str, Any]:
    coords = [(float(lon), float(lat)) for lon, lat in candidate["track_coordinates"]]
    track_miles = round_miles(point_path_miles(coords))
    raw_moving = ceil_minutes(track_miles * float(state.get("pace_min_per_mile") or 16.0))
    effort = track_effort(
        coords,
        official_miles=float(candidate.get("official_miles") or 0),
        raw_moving_minutes=raw_moving,
        elevation_sampler=elevation_sampler,
    )
    drive_to = drive_minutes_to_trailhead(trailhead, state.get("drive_model") or {})
    parking_minutes = int(trailhead.get("parking_minutes") or state.get("parking_minutes") or 8)
    penalty = route_finding_penalty(candidate)
    estimates = build_time_estimates_minutes(
        drive_to=drive_to,
        parking_minutes=parking_minutes,
        raw_moving_minutes=raw_moving,
        effort=effort,
        route_finding_penalty_minutes=penalty,
    )
    return {
        "track_miles": track_miles,
        "raw_moving_minutes": raw_moving,
        "drive_to": drive_to,
        "parking_minutes": parking_minutes,
        "route_finding_penalty": penalty,
        "effort": effort,
        "time_estimates_minutes": estimates,
        "time_breakdown_minutes": {
            "drive_to_trailhead": drive_to,
            "parking_and_prep": parking_minutes,
            "trailhead_access": 0,
            "moving_time": raw_moving,
            "return_drive": drive_to,
        },
    }


def segment_payload(
    segment_id: str,
    *,
    official_by_id: dict[str, dict[str, Any]],
    reversed_direction: bool,
) -> dict[str, Any]:
    segment = copy.deepcopy(official_by_id[segment_id])
    coords = list(reversed(segment["coordinates"])) if reversed_direction else list(segment["coordinates"])
    direction = str(segment.get("direction") or "both")
    segment["coordinates"] = [[float(lon), float(lat)] for lon, lat in coords]
    segment["direction_rule"] = direction
    if direction == "ascent":
        segment["direction_cue"] = (
            "Ascent-only segment; valid uphill traversal is opposite official geometry direction."
            if reversed_direction
            else "Ascent-only segment; valid uphill traversal follows official geometry direction."
        )
    else:
        segment["direction_cue"] = "Either direction allowed; follow map arrows."
    return segment


def link_to_cue_link(row: dict[str, Any], previous_segment: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "source": row.get("path_source") or "mapped_graph",
        "from_segment_id": previous_segment.get("seg_id") if previous_segment else None,
        "to_segment_id": row.get("to_segment_id"),
        "from_trail": previous_segment.get("trail_name") if previous_segment else None,
        "to_trail": row.get("to_trail_name"),
        "distance_miles": row.get("link_distance_miles"),
        "connector_miles": row.get("connector_miles"),
        "official_repeat_miles": row.get("official_repeat_miles"),
        "official_repeat_segment_ids": copy.deepcopy(row.get("official_repeat_segment_ids") or []),
        "connector_names": copy.deepcopy(row.get("connector_names") or []),
        "connector_classes": copy.deepcopy(row.get("connector_classes") or []),
        "path_coordinates": copy.deepcopy(row.get("path_coordinates") or []),
        "graph_validated": row.get("path_source") != "direct_gap_fallback",
    }


def route_cue_for_generated(
    *,
    candidate_id: str,
    route_def: dict[str, Any],
    candidate: dict[str, Any],
    pricing: dict[str, Any],
    trailhead: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    link_rows = list(candidate.get("link_rows") or [])
    segment_rows = [row for row in link_rows if row.get("to_segment_id") != "return_to_car"]
    segments = []
    previous = None
    between_links = []
    for index, row in enumerate(segment_rows):
        segment_id = str(row["to_segment_id"])
        segment = segment_payload(
            segment_id,
            official_by_id=official_by_id,
            reversed_direction=bool(row.get("segment_reversed")),
        )
        if index > 0 and float(row.get("link_distance_miles") or 0) > 0:
            link = link_to_cue_link(row, previous)
            segment["pre_connector_link"] = copy.deepcopy(link)
            between_links.append(link)
        segments.append(segment)
        previous = segment
    start_row = segment_rows[0] if segment_rows else {}
    return_row = next((row for row in link_rows if row.get("to_segment_id") == "return_to_car"), {})
    validation = candidate.get("coverage_validation") or {}
    direct_gap = float(candidate.get("direct_gap_fallback_miles") or 0)
    return {
        "candidate_id": candidate_id,
        "title": route_def["title"],
        "route_status": "graph_validated",
        "official_miles": candidate.get("official_miles"),
        "on_foot_miles": pricing["track_miles"],
        "raw_total_minutes": pricing["time_estimates_minutes"]["door_to_door_raw"],
        "total_minutes": pricing["time_estimates_minutes"]["door_to_door_p75"],
        "time_breakdown_minutes": pricing["time_breakdown_minutes"],
        "time_estimates_minutes": pricing["time_estimates_minutes"],
        "effort": pricing["effort"],
        "trailhead": copy.deepcopy(trailhead),
        "start_access": {
            "confidence": "high" if float(start_row.get("link_distance_miles") or 0) <= 0.1 else "medium",
            "direct_gap_miles": 0.0 if start_row.get("path_source") != "direct_gap_fallback" else start_row.get("link_distance_miles"),
            "mapped_access_miles": start_row.get("link_distance_miles"),
            "official_repeat_miles": start_row.get("official_repeat_miles") or 0,
            "official_repeat_segment_ids": copy.deepcopy(start_row.get("official_repeat_segment_ids") or []),
            "access_class": "mapped" if float(start_row.get("link_distance_miles") or 0) > 0.1 else "direct",
            "graph_validated": start_row.get("path_source") != "direct_gap_fallback",
            "connector_names": copy.deepcopy(start_row.get("connector_names") or []),
            "connector_classes": copy.deepcopy(start_row.get("connector_classes") or []),
            "path_coordinates": copy.deepcopy(start_row.get("path_coordinates") or []),
        },
        "segments": segments,
        "between_links": between_links,
        "return_to_car": {
            "strategy": "mapped_generated_return"
            if return_row.get("path_source") != "direct_gap_fallback"
            else "direct_gap_generated_return",
            "description": "Return by the generated graph route back to the parked trailhead.",
            "official_repeat_miles": return_row.get("official_repeat_miles") or 0,
            "official_repeat_segment_ids": copy.deepcopy(return_row.get("official_repeat_segment_ids") or []),
            "connector_miles": return_row.get("connector_miles") or 0,
            "road_miles": 0.0,
            "connector_names": copy.deepcopy(return_row.get("connector_names") or []),
            "connector_classes": copy.deepcopy(return_row.get("connector_classes") or []),
            "path_coordinates": copy.deepcopy(return_row.get("path_coordinates") or []),
            "needs_map_validation": return_row.get("path_source") == "direct_gap_fallback",
            "graph_validated": return_row.get("path_source") != "direct_gap_fallback",
        },
        "validation": {
            "segment_coverage_passed": validation.get("status") == "covers_template_segment_set",
            "ascent_direction_passed": (candidate.get("ascent_direction_validation") or {}).get("status")
            != "failed_ascent_direction",
            "return_path_graph_validated": return_row.get("path_source") != "direct_gap_fallback",
            "trailhead_snap_confidence": "medium" if direct_gap else "high",
            "connector_overlap_checked": True,
            "special_management_checked": False,
        },
        "direction_validation": {
            "passed": (candidate.get("ascent_direction_validation") or {}).get("status") != "failed_ascent_direction",
            "reason": (candidate.get("ascent_direction_validation") or {}).get("status"),
            "ascent_segment_ids_checked": [
                int(value)
                for value in (candidate.get("ascent_direction_validation") or {}).get("ascent_segment_ids") or []
            ],
            "planned_traversal_direction": {
                str(row["to_segment_id"]): (
                    "official_geometry_end_to_start"
                    if row.get("segment_reversed")
                    else "official_geometry_start_to_end"
                )
                for row in segment_rows
                if official_by_id.get(str(row.get("to_segment_id")), {}).get("direction") == "ascent"
            },
        },
        "cue_generation_mode": "chbh_polecat_recomposition_repair",
        "route_truth_repair_id": REPAIR_ID,
        "field_notes": [
            "Generated as part of the CHBH/Polecat recomposition repair.",
            "Current conditions, heat, water, and day-of signage still require same-day checks.",
        ],
    }


def route_feature(candidate_id: str, route_def: dict[str, Any], candidate: dict[str, Any], pricing: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon), float(lat)] for lon, lat in candidate["track_coordinates"]],
        },
        "properties": {
            "kind": "route",
            "package_number": package.get("package_number"),
            "candidate_id": candidate_id,
            "block_name": package.get("block_name"),
            "title": route_def["title"],
            "official_miles": candidate.get("official_miles"),
            "on_foot_miles": pricing["track_miles"],
            "trailhead": route_def.get("trailhead_name"),
            "field_menu_label": route_def.get("field_menu_label"),
            "source": ROUTE_TRUTH_SOURCE,
            "route_truth_repair_id": REPAIR_ID,
            "source_gap_warning_count": 1 if float(candidate.get("direct_gap_fallback_miles") or 0) > 0.05 else 0,
        },
    }


def parking_feature(candidate_id: str, route_def: dict[str, Any], trailhead: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(trailhead["lon"]), float(trailhead["lat"])]},
        "properties": {
            "kind": "parking",
            "package_number": package.get("package_number"),
            "candidate_id": candidate_id,
            "block_name": package.get("block_name"),
            "field_menu_label": route_def.get("field_menu_label"),
            "name": trailhead.get("name"),
            "has_parking": trailhead.get("has_parking"),
            "has_restroom": trailhead.get("has_restroom"),
            "has_water": trailhead.get("has_water"),
            "water_confidence": trailhead.get("water_confidence"),
            "parking_minutes": trailhead.get("parking_minutes"),
            "source": trailhead.get("source"),
            "parking_confidence": trailhead.get("parking_confidence"),
        },
    }


def component_for_generated(
    candidate_id: str,
    route_def: dict[str, Any],
    candidate: dict[str, Any],
    pricing: dict[str, Any],
    route_number: int,
    trailhead: dict[str, Any],
) -> dict[str, Any]:
    official = float(candidate.get("official_miles") or 0)
    on_foot = float(pricing["track_miles"])
    flags = ["route_truth_repaired", "chbh_polecat_recomposition"]
    if float(candidate.get("official_repeat_miles") or 0) > 0:
        flags.append("requires_official_repeat_to_get_back_to_car")
    if float(candidate.get("direct_gap_fallback_miles") or 0) > 0:
        flags.append("direct_gap_reviewed_generated_connector")
    if official and on_foot / official > 2.0:
        flags.append("low_official_to_total_mileage_ratio")
    return {
        "route_number": route_number,
        "candidate_id": candidate_id,
        "field_menu_group_id": candidate_id,
        "field_menu_label": route_def.get("field_menu_label"),
        "trail_names": route_def.get("trail_names") or [],
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "total_minutes": int(pricing["time_estimates_minutes"]["door_to_door_p75"]),
        "raw_total_minutes": int(pricing["time_estimates_minutes"]["door_to_door_raw"]),
        "trailhead": trailhead.get("name"),
        "less_optimal_flags": flags,
        "segment_ids": [int(value) for value in candidate.get("official_segment_ids") or []],
        "time_breakdown_minutes": pricing["time_breakdown_minutes"],
        "time_estimates_minutes": pricing["time_estimates_minutes"],
        "effort": pricing["effort"],
        "route_status": "graph_validated",
        "source": ROUTE_TRUTH_SOURCE,
        "route_truth_repair_id": REPAIR_ID,
        "route_truth_replaces_candidate_ids": list(route_def.get("route_truth_replaces_candidate_ids") or []),
        "route_quality": {
            "route_truth_repair": True,
            "chbh_polecat_recomposition": True,
            "direct_gap_fallback_miles": candidate.get("direct_gap_fallback_miles"),
        },
    }


def generated_payloads(
    *,
    current_map: dict[str, Any],
    replacements: dict[str, Any],
    state: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    connector_graph: dict[str, Any],
    elevation_sampler: Any,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for candidate_id, route_def in GENERATED_ROUTES.items():
        parking = parking_for_candidate(current_map, route_def["parking_candidate_id"], replacements)
        candidate = build_generated_route(
            variant_id=candidate_id,
            strategy=route_def["strategy"],
            ordered_segment_ids=route_def["ordered_segment_ids"],
            parking=parking,
            official_by_id=official_by_id,
            connector_graph=connector_graph,
        )
        pricing = price_generated_candidate(
            candidate,
            state=state,
            trailhead=parking,
            elevation_sampler=elevation_sampler,
        )
        route_def = {**route_def, "trailhead_name": parking.get("name")}
        package = package_for_number(current_map, route_def["package_number"])
        validation = validate_track_segments([candidate["track_coordinates"]], max_gap_miles=0.05)
        payloads[candidate_id] = {
            "definition": route_def,
            "candidate": candidate,
            "pricing": pricing,
            "cue": route_cue_for_generated(
                candidate_id=candidate_id,
                route_def=route_def,
                candidate=candidate,
                pricing=pricing,
                trailhead=parking,
                official_by_id=official_by_id,
            ),
            "route_feature": route_feature(candidate_id, route_def, candidate, pricing, package),
            "parking_feature": parking_feature(candidate_id, route_def, parking, package),
            "route_validation": {
                "candidate_id": candidate_id,
                "source_gap_warning": float(candidate.get("direct_gap_fallback_miles") or 0) > 0.05,
                "source_max_gap_miles": candidate.get("direct_gap_fallback_miles"),
                "rendered_passed": validation.get("passed"),
                "rendered_failures": validation.get("failures") or [],
            },
        }
    return payloads


def merge_feature_collections(*collections_list: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for collections in collections_list:
        for name, collection in (collections or {}).items():
            target = merged.setdefault(name, {"type": collection.get("type") or "FeatureCollection", "features": []})
            target.setdefault("features", []).extend(copy.deepcopy(collection.get("features") or []))
    return merged


def build_override_for_package1(current_map: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    package = copy.deepcopy(package_for_number(current_map, 1))
    candidate_ids = [str(component.get("candidate_id")) for component in package.get("components") or []]
    keep_ids = [
        "multi-start-1a-1a-ms-04-2-full-sail-trail-buena-vista-trail-bob-smylie",
        "combo-who-now-loop-trail-harrison-ridge-harrison-hollow-kempers-ridge-trail-hippie-shake-trail",
    ]
    kept_components = [
        copy.deepcopy(component_by_candidate(package, candidate_id))
        for candidate_id in keep_ids
        if any(str(component.get("candidate_id")) == candidate_id for component in package.get("components") or [])
    ]
    candidate_id = "route-truth-1a-1-chute-chbh-polecat5"
    new_component = component_for_generated(
        candidate_id,
        payloads[candidate_id]["definition"],
        payloads[candidate_id]["candidate"],
        payloads[candidate_id]["pricing"],
        13,
        payloads[candidate_id]["cue"]["trailhead"],
    )
    package["components"] = [new_component, *kept_components]
    package["planning_status"] = "route_truth_repaired"
    reasons = list(package.get("planning_reasons") or [])
    for reason in [
        "route_truth_repaired",
        "chbh_and_polecat5_absorbed_into_36th_chute",
        "drive_repark_cost_reviewed",
    ]:
        if reason not in reasons:
            reasons.append(reason)
    package["planning_reasons"] = reasons
    package = recompute_package(package, add_multi_start_reasons=False)
    cues = route_cue_index(current_map)
    route_cues = {
        candidate_id: copy.deepcopy(payloads[candidate_id]["cue"]),
        **{keep_id: copy.deepcopy(cues[keep_id]) for keep_id in keep_ids if keep_id in cues},
    }
    existing_features = feature_collections_for_candidates(current_map, keep_ids)
    new_features = {
        "routes": {"type": "FeatureCollection", "features": [copy.deepcopy(payloads[candidate_id]["route_feature"])]},
        "parking": {"type": "FeatureCollection", "features": [copy.deepcopy(payloads[candidate_id]["parking_feature"])]},
    }
    return {
        "repair_id": REPAIR_ID,
        "package_number": 1,
        "reason": "Absorb CHBH and Polecat 5 into the lower 36th Street Chute card after drive/repark cost review.",
        "remove_candidate_ids": candidate_ids,
        "replace_package": package,
        "route_cues": route_cues,
        "feature_collections": merge_feature_collections(existing_features, new_features),
        "route_validations": route_validations_for_candidates(current_map, keep_ids)
        + [copy.deepcopy(payloads[candidate_id]["route_validation"])],
    }


def build_override_for_package5(current_map: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    package = copy.deepcopy(package_for_number(current_map, 5))
    candidate_ids = [str(component.get("candidate_id")) for component in package.get("components") or []]
    barn_id = "multi-start-5-5-ms-04-1-barn-owl"
    barn_component = copy.deepcopy(component_by_candidate(package, barn_id))
    new_components = [
        barn_component,
        component_for_generated(
            "route-truth-5b-polecat-minus-p5",
            payloads["route-truth-5b-polecat-minus-p5"]["definition"],
            payloads["route-truth-5b-polecat-minus-p5"]["candidate"],
            payloads["route-truth-5b-polecat-minus-p5"]["pricing"],
            92,
            payloads["route-truth-5b-polecat-minus-p5"]["cue"]["trailhead"],
        ),
    ]
    package["components"] = new_components
    package["planning_status"] = "route_truth_repaired"
    reasons = list(package.get("planning_reasons") or [])
    for reason in [
        "route_truth_repaired",
        "chbh_moved_out_of_package_6",
        "polecat5_claim_moved_to_36th_chute_card",
        "drive_repark_cost_reviewed",
    ]:
        if reason not in reasons:
            reasons.append(reason)
    package["planning_reasons"] = reasons
    package = recompute_package(package, add_multi_start_reasons=False)
    cues = route_cue_index(current_map)
    route_cues = {barn_id: copy.deepcopy(cues[barn_id])}
    for candidate_id in ["route-truth-5b-polecat-minus-p5"]:
        route_cues[candidate_id] = copy.deepcopy(payloads[candidate_id]["cue"])
    existing_features = feature_collections_for_candidates(current_map, [barn_id])
    new_features = {
        "routes": {
            "type": "FeatureCollection",
            "features": [
                copy.deepcopy(payloads["route-truth-5b-polecat-minus-p5"]["route_feature"]),
            ],
        },
        "parking": {
            "type": "FeatureCollection",
            "features": [
                copy.deepcopy(payloads["route-truth-5b-polecat-minus-p5"]["parking_feature"]),
            ],
        },
    }
    return {
        "repair_id": REPAIR_ID,
        "package_number": 5,
        "reason": "Remove Polecat Loop 5 from the Polecat core card after moving it to 36th Chute.",
        "remove_candidate_ids": candidate_ids,
        "replace_package": package,
        "route_cues": route_cues,
        "feature_collections": merge_feature_collections(existing_features, new_features),
        "route_validations": route_validations_for_candidates(current_map, [barn_id])
        + [copy.deepcopy(payloads["route-truth-5b-polecat-minus-p5"]["route_validation"])],
    }


def build_override_for_package6(current_map: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    package = copy.deepcopy(package_for_number(current_map, 6))
    candidate_ids = [str(component.get("candidate_id")) for component in package.get("components") or []]
    candidate_id = "route-truth-6-cartwright-peggy-minus-chbh"
    package["components"] = [
        component_for_generated(
            candidate_id,
            payloads[candidate_id]["definition"],
            payloads[candidate_id]["candidate"],
            payloads[candidate_id]["pricing"],
            10,
            payloads[candidate_id]["cue"]["trailhead"],
        )
    ]
    package["planning_status"] = "route_truth_repaired"
    stale_reasons = {"chbh_claim_moved_to_package_5_cartwright_mini_card"}
    reasons = [reason for reason in (package.get("planning_reasons") or []) if reason not in stale_reasons]
    for reason in ["route_truth_repaired", "chbh_claim_moved_to_36th_chute_card"]:
        if reason not in reasons:
            reasons.append(reason)
    package["planning_reasons"] = reasons
    package = recompute_package(package, add_multi_start_reasons=False)
    return {
        "repair_id": REPAIR_ID,
        "package_number": 6,
        "reason": "Remove CHBH Connector from the Peggy's / Cartwright mega-card.",
        "remove_candidate_ids": candidate_ids,
        "replace_package": package,
        "route_cues": {candidate_id: copy.deepcopy(payloads[candidate_id]["cue"])},
        "feature_collections": {
            "routes": {"type": "FeatureCollection", "features": [copy.deepcopy(payloads[candidate_id]["route_feature"])]},
            "parking": {"type": "FeatureCollection", "features": [copy.deepcopy(payloads[candidate_id]["parking_feature"])]},
        },
        "route_validations": [copy.deepcopy(payloads[candidate_id]["route_validation"])],
    }


def transfer_minutes_between(a: dict[str, Any], b: dict[str, Any], drive_model: dict[str, Any]) -> int:
    miles = haversine_miles((float(a["lon"]), float(a["lat"])), (float(b["lon"]), float(b["lat"])))
    drive_miles = miles * float(drive_model["straight_line_factor"])
    return max(ceil_minutes(drive_miles * float(drive_model["minutes_per_mile"])), int(drive_model["minimum_one_way_minutes"]))


def summarize_decision(
    *,
    current_map: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    state: dict[str, Any],
    previous_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package5 = package_for_number(current_map, 5)
    package6 = package_for_number(current_map, 6)
    package1 = package_for_number(current_map, 1)
    try:
        chute = component_by_candidate(package1, "multi-start-1a-1a-ms-04-1-36th-street-chute")
        current5b = component_by_candidate(package5, "multi-start-5-5-ms-04-2-polecat-loop-doe-ridge-quick-draw")
        current6 = component_by_candidate(package6, "block-cartwright_peggy_interface")
    except KeyError:
        if previous_summary:
            summary = copy.deepcopy(previous_summary)
            summary["rerun_note"] = (
                "The active private menu is already repaired; retaining the original "
                "pre-repair comparison metrics for audit continuity."
            )
            return summary
        chute = payloads["route-truth-1a-1-chute-chbh-polecat5"]["pricing"]
        current5b = payloads["route-truth-5b-polecat-minus-p5"]["pricing"]
        current6 = payloads["route-truth-6-cartwright-peggy-minus-chbh"]["pricing"]
    lower_seed = payloads["route-truth-1a-1-chute-chbh-polecat5"]
    cart_mini = payloads["review-seed-5c-chbh-polecat5"]
    five_minus = payloads["route-truth-5b-polecat-minus-p5"]
    six_minus = payloads["route-truth-6-cartwright-peggy-minus-chbh"]
    drive_model = state.get("drive_model") or {}
    transfer = transfer_minutes_between(lower_seed["cue"]["trailhead"], cart_mini["cue"]["trailhead"], drive_model)
    current_scope_p75 = int_value(chute.get("total_minutes")) + int_value(current5b.get("total_minutes")) + int_value(current6.get("total_minutes"))
    current_scope_miles = round_miles(
        float(chute.get("on_foot_miles") or 0)
        + float(current5b.get("on_foot_miles") or 0)
        + float(current6.get("on_foot_miles") or 0)
    )
    chute_absorb_scope_p75 = (
        int_value(lower_seed["pricing"]["time_estimates_minutes"]["door_to_door_p75"])
        + int_value(five_minus["pricing"]["time_estimates_minutes"]["door_to_door_p75"])
        + int_value(six_minus["pricing"]["time_estimates_minutes"]["door_to_door_p75"])
    )
    chute_absorb_scope_miles = round_miles(
        float(lower_seed["pricing"]["track_miles"])
        + float(five_minus["pricing"]["track_miles"])
        + float(six_minus["pricing"]["track_miles"])
    )
    cartwright_mini_scope_p75 = (
        int_value(chute.get("total_minutes"))
        + int_value(five_minus["pricing"]["time_estimates_minutes"]["door_to_door_p75"])
        + int_value(cart_mini["pricing"]["time_estimates_minutes"]["door_to_door_p75"])
        + int_value(six_minus["pricing"]["time_estimates_minutes"]["door_to_door_p75"])
    )
    cartwright_mini_scope_miles = round_miles(
        float(chute.get("on_foot_miles") or 0)
        + float(five_minus["pricing"]["track_miles"])
        + float(cart_mini["pricing"]["track_miles"])
        + float(six_minus["pricing"]["track_miles"])
    )
    return {
        "current_scope": {
            "description": "Current Chute + Polecat 5B + Route 6 scope from canonical private menu",
            "route_count": 3,
            "on_foot_miles": current_scope_miles,
            "p75_minutes": current_scope_p75,
        },
        "selected_chute_absorption": {
            "description": "Move CHBH + Polecat 5 into the lower 36th Chute outing",
            "route_count": 3,
            "on_foot_miles": chute_absorb_scope_miles,
            "p75_minutes": chute_absorb_scope_p75,
            "delta_miles_vs_current": round_miles(chute_absorb_scope_miles - current_scope_miles),
            "delta_p75_vs_current": chute_absorb_scope_p75 - current_scope_p75,
        },
        "cartwright_mini_card_review_seed": {
            "description": "Keep Chute alone, split CHBH + Polecat 5 into a Cartwright mini-card",
            "route_count": 4,
            "on_foot_miles": cartwright_mini_scope_miles,
            "p75_minutes": cartwright_mini_scope_p75,
            "delta_miles_vs_current": round_miles(cartwright_mini_scope_miles - current_scope_miles),
            "delta_p75_vs_current": cartwright_mini_scope_p75 - current_scope_p75,
        },
        "same_day_transfer": {
            "lower_36th_to_cartwright_drive_minutes": transfer,
            "note": "The Cartwright mini-card avoids lower-36th absorption but adds a fourth route card; same-day Chute plus Cartwright work would add this transfer.",
        },
        "selected_repair": "selected_chute_absorption",
        "selection_reason": "After drive/repark and prep are counted, the Chute absorption is shorter and lower p75 than a separate Cartwright CHBH/Polecat mini-card, while still shrinking Route 6.",
    }


def upsert_repair_overrides(replacements: dict[str, Any], overrides: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(replacements)
    kept = [
        override
        for override in result.get("overrides") or []
        if str(override.get("repair_id") or "") != REPAIR_ID
    ]
    kept.extend(overrides)
    result["overrides"] = kept
    result.setdefault("summary", {})["chbh_polecat_recomposition_repair"] = {
        "repair_id": REPAIR_ID,
        "packages_replaced": [1, 5, 6],
        "selected_repair": "chute_absorption",
    }
    result["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return result


def build_promotion(args: argparse.Namespace) -> dict[str, Any]:
    current_map = read_json(args.current_map_json)
    replacements = read_json(args.field_menu_replacements_json)
    previous_summary = None
    if args.output_json.exists():
        previous_output = read_json(args.output_json)
        previous_summary = previous_output.get("summary")
    state = load_state(args.state_json)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    official_by_id = {str(segment["seg_id"]): segment for segment in official_segments}
    connector_graph = load_connector_graph(args.connector_geojson, official_segments=official_segments)
    elevation_sampler = load_dem_context(args.dem_tif, args.dem_summary_json)["sampler"]
    payloads = generated_payloads(
        current_map=current_map,
        replacements=replacements,
        state=state,
        official_by_id=official_by_id,
        connector_graph=connector_graph,
        elevation_sampler=elevation_sampler,
    )
    overrides = [
        build_override_for_package1(current_map, payloads),
        build_override_for_package5(current_map, payloads),
        build_override_for_package6(current_map, payloads),
    ]
    updated_replacements = upsert_repair_overrides(replacements, overrides)
    summary = summarize_decision(
        current_map=current_map,
        payloads=payloads,
        state=state,
        previous_summary=previous_summary,
    )
    return {
        "updated_replacements": updated_replacements,
        "summary": summary,
        "payload_summary": {
            candidate_id: {
                "official_miles": payload["candidate"].get("official_miles"),
                "on_foot_miles": payload["pricing"]["track_miles"],
                "p75_minutes": payload["pricing"]["time_estimates_minutes"]["door_to_door_p75"],
                "p90_minutes": payload["pricing"]["time_estimates_minutes"]["door_to_door_p90"],
                "official_repeat_miles": payload["candidate"].get("official_repeat_miles"),
                "connector_miles": payload["candidate"].get("connector_miles"),
                "direct_gap_fallback_miles": payload["candidate"].get("direct_gap_fallback_miles"),
                "route_finding_penalty": payload["pricing"]["route_finding_penalty"],
                "ascent_ft": payload["pricing"]["effort"].get("ascent_ft"),
                "descent_ft": payload["pricing"]["effort"].get("descent_ft"),
            }
            for candidate_id, payload in payloads.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-map-json", type=Path, default=DEFAULT_CURRENT_MAP_JSON)
    parser.add_argument("--field-menu-replacements-json", type=Path, default=DEFAULT_FIELD_MENU_REPLACEMENTS_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    promotion = build_promotion(args)
    if not args.dry_run:
        write_json(args.field_menu_replacements_json, promotion["updated_replacements"])
    write_json(
        args.output_json,
        {
            "schema": "boise_trails_chbh_polecat_promotion_summary_v1",
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "status": "dry_run" if args.dry_run else "private_replacement_source_updated",
            "source_files": {
                "current_map_json": display_path(args.current_map_json),
                "field_menu_replacements_json": display_path(args.field_menu_replacements_json),
                "official_geojson": display_path(args.official_geojson),
                "connector_geojson": display_path(args.connector_geojson),
            },
            "summary": promotion["summary"],
            "payload_summary": promotion["payload_summary"],
        },
    )
    print(f"Wrote {display_path(args.output_json)}")
    if not args.dry_run:
        print(f"Updated {display_path(args.field_menu_replacements_json)}")
    print(json.dumps(promotion["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
