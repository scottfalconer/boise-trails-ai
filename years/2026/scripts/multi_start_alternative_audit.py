#!/usr/bin/env python3
"""Audit field-menu outings for one-transfer multi-start alternatives.

This is a review tool, not a field-packet publisher. It may use private
Strava-derived parking anchors in private outputs, then writes a public-safe
checkpoint with private coordinates stripped.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
    DEFAULT_ACTIVITY_SUMMARY_CSV,
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
    DEFAULT_SEGMENT_PERF_CSV,
    DEFAULT_STRAVA_DETAILS_DIR,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    build_performance_profile,
    candidate_from_trail_group,
    ceil_minutes,
    drive_minutes_to_trailhead,
    get_outing_model,
    group_remaining_by_trail,
    haversine_miles,
    iter_line_parts,
    line_length_miles,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    load_trailheads_from_geojson,
    normalize_name,
    read_json,
    reverse_trail_orientation,
    round_miles,
    shortest_connector_path,
    trail_is_reversible,
)


DEFAULT_FIELD_MENU_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_MANUAL_DESIGN_JSONS = [
    YEAR_DIR / "inputs" / "personal" / "2026-manual-route-designs-v1.json",
]
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_PARKING_REVIEW_DECISIONS_JSON = (
    YEAR_DIR / "inputs" / "personal" / "private" / "parking-anchor-review-decisions-2026-05-08.json"
)
DEFAULT_PRIVATE_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "multi-start-alternatives"
DEFAULT_PUBLIC_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "multi-start-alternative-audit-2026-05-08"

PROMISING_MIN_SAVINGS_MILES = 0.75
PROMISING_MAX_ELAPSED_WORSE_MINUTES = 20
SUBSTANTIAL_SAVINGS_MIN_MILES = 2.0
SUBSTANTIAL_SAVINGS_MAX_ELAPSED_WORSE_MINUTES = 45
SUBSTANTIAL_SAVINGS_MAX_MINUTES_PER_MILE = 15.0
DEFAULT_REFILL_REPARK_MINUTES = 8
DEFAULT_MAX_PARTITIONS_PER_OUTING = 24
DEFAULT_MAX_ANCHORS_PER_COMPONENT = 3
DEFAULT_MAX_STREET_PROBES_PER_OUTING = 10
DEFAULT_CERTIFIABLE_ANCHOR_CONNECTOR_BUDGET_MILES = 1.0
ASSUMED_VEHICLE_ROAD_PARKING_MAX_MILES = 0.10
ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE = "assumed_paved_road_parking_within_0_10_mile"
KNOWN_BOGUS_TRAILHEAD_NAMES = {
    "Nordic Lodge Parking Area",
    "Pioneer Lodge Parking Area",
    "Simplot Lodge Parking Area",
}
BOGUS_BASIN_TRAIL_NAMES = {
    "Around the Mountain",
    "Around the Mountain Trail",
    "Brewer's Byway Extension",
    "Brewers Byway",
    "Deer Point Trail",
    "Eastside",
    "Elk Meadows Trail",
    "Lodge Trail",
    "Mores Mtn Interpretive",
    "Mores Mountain Interpretive Trail",
    "Shindig",
    "Sunshine XC",
    "Tempest Trail",
    "The Face Trail",
}

PUBLIC_ROAD_HIGHWAYS = {
    "primary",
    "secondary",
    "residential",
    "service",
    "unclassified",
    "tertiary",
    "living_street",
    "track",
}
PAVED_BY_DEFAULT_HIGHWAYS = {
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "living_street",
    "unclassified",
}
PAVED_SURFACES = {
    "asphalt",
    "concrete",
    "concrete:lanes",
    "concrete:plates",
    "paved",
}
UNPAVED_SURFACES = {
    "compacted",
    "dirt",
    "earth",
    "fine_gravel",
    "grass",
    "gravel",
    "ground",
    "mud",
    "pebblestone",
    "sand",
    "unpaved",
}
EXCLUDED_STREET_PROBE_HIGHWAYS = {
    "motorway",
    "trunk",
    "footway",
    "path",
    "cycleway",
    "steps",
}

CONFIDENCE_RANKS = {
    "source_validated_trailhead": 90,
    "source_verified_roadside_plus_strava_seen": 88,
    "source_verified_roadside": 86,
    "osm_amenity_parking_fee_no_capacity_36_source_checked": 84,
    "osm_amenity_parking_capacity_16_source_checked": 84,
    "osm_amenity_parking_source_checked": 82,
    "osm_amenity_parking_near_official_start": 80,
    "user_review_confirmed_paved_road_parking": 78,
    "strava_reused_prior_challenge_window": 76,
    "strava_seen_prior_challenge_window": 72,
    "inferred_from_trailhead_layer": 64,
    "user_configured_trailhead": 62,
    "strava_single_prior_challenge_window": 50,
    ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE: 48,
    "street_parking_probe_manual_required": 28,
    "manual_required": 20,
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def confidence_rank(anchor: dict[str, Any]) -> int:
    confidence = str(anchor.get("parking_confidence") or "")
    if confidence in CONFIDENCE_RANKS:
        return CONFIDENCE_RANKS[confidence]
    for prefix, rank in CONFIDENCE_RANKS.items():
        if confidence.startswith(prefix):
            return rank
    return 40


def anchor_id(anchor: dict[str, Any], prefix: str = "anchor") -> str:
    if anchor.get("anchor_id"):
        return str(anchor["anchor_id"])
    if anchor.get("facility_id") is not None:
        return f"facility-{anchor['facility_id']}"
    return f"{prefix}-{slugify(str(anchor.get('name') or 'unknown'))}"


def normalized_anchor(anchor: dict[str, Any], *, source_type: str | None = None) -> dict[str, Any]:
    result = dict(anchor)
    result["anchor_id"] = anchor_id(result, prefix=source_type or "anchor")
    result["name"] = str(result.get("name") or result.get("facility_name") or result["anchor_id"])
    result["lat"] = float(result["lat"])
    result["lon"] = float(result["lon"])
    result["has_parking"] = result.get("has_parking") is not False
    result["parking_minutes"] = int(result.get("parking_minutes") or 8)
    result["parking_confidence"] = result.get("parking_confidence") or "manual_required"
    result["source"] = result.get("source") or source_type or "unknown"
    result["source_type"] = result.get("source_type") or source_type or result["source"]
    result["field_ready"] = anchor_field_ready(result)
    result.setdefault("privacy", "public")
    return result


def anchor_key(anchor: dict[str, Any]) -> tuple[float, float]:
    return (round(float(anchor["lon"]), 5), round(float(anchor["lat"]), 5))


def merge_parking_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[float, float], dict[str, Any]] = {}
    for raw_anchor in anchors:
        if raw_anchor.get("lat") is None or raw_anchor.get("lon") is None:
            continue
        anchor = normalized_anchor(raw_anchor)
        key = anchor_key(anchor)
        current = deduped.get(key)
        if current is None or confidence_rank(anchor) > confidence_rank(current):
            deduped[key] = anchor
    return sorted(
        deduped.values(),
        key=lambda item: (-confidence_rank(item), str(item.get("name") or "")),
    )


def load_parking_review_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    if "anchors" in data and isinstance(data["anchors"], dict):
        return data["anchors"]
    if isinstance(data, dict):
        return data
    return {}


def apply_parking_review_decisions(
    anchors: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not decisions:
        return anchors
    reviewed = []
    for anchor in anchors:
        result = dict(anchor)
        decision = decisions.get(str(result.get("anchor_id") or ""))
        if decision:
            value = str(decision.get("decision") or "").lower()
            result["parking_review_decision"] = value
            result["parking_review_notes"] = decision.get("notes") or ""
            result["parking_review_updated_at"] = decision.get("updatedAt")
            if value == "yes":
                result["field_ready"] = True
                if result.get("parking_confidence") == ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE:
                    result["parking_confidence"] = "user_review_confirmed_paved_road_parking"
            elif value == "no":
                result["field_ready"] = False
                result["parking_rejected"] = True
            else:
                result["field_ready"] = False
        reviewed.append(result)
    return reviewed


def anchor_field_ready(anchor: dict[str, Any]) -> bool:
    if anchor.get("parking_rejected") is True:
        return False
    if anchor.get("has_parking") is not True:
        return False
    source_type = str(anchor.get("source_type") or "").lower()
    source = str(anchor.get("source") or "").lower()
    confidence = str(anchor.get("parking_confidence") or "").lower()
    if source_type == "private_strava_anchor" or source == "strava_activity_endpoint_cluster":
        return True
    if anchor.get("field_ready") is False:
        return False
    return any(
        token in confidence
        for token in [
            "source_validated",
            "source_verified",
            "osm_amenity",
            "strava_reused",
            "strava_seen",
            "strava_single",
            "inferred_from_trailhead_layer",
            "user_configured",
        ]
    )


def manual_anchor_trailheads(paths: list[Path]) -> list[dict[str, Any]]:
    anchors = []
    for path in paths:
        if not path.exists():
            continue
        data = read_json(path)
        for area in data.get("areas") or []:
            for anchor in area.get("anchors") or []:
                anchors.append(
                    {
                        "anchor_id": anchor.get("anchor_id"),
                        "name": anchor.get("name"),
                        "lat": anchor.get("lat"),
                        "lon": anchor.get("lon"),
                        "parking_minutes": int(anchor.get("parking_minutes") or 8),
                        "has_parking": anchor.get("has_parking") is True,
                        "parking_confidence": anchor.get("parking_confidence") or "manual_required",
                        "source": anchor.get("source") or "manual_route_design",
                        "source_type": "manual_route_design",
                        "field_ready": anchor.get("field_ready") is True,
                        "allowed_for_trails": anchor.get("allowed_for_trails") or [],
                        "manual_area_id": area.get("area_id"),
                        "privacy": "public",
                    }
                )
    return anchors


def load_parking_anchors(
    *,
    public_trailheads_geojson: Path,
    private_parking_anchors_geojson: Path,
    manual_design_jsons: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = [
        normalized_anchor(item, source_type="public_trailhead")
        for item in load_trailheads_from_geojson(public_trailheads_geojson)
    ]
    private = [
        normalized_anchor(item | {"privacy": "private_exact_coordinates"}, source_type="private_strava_anchor")
        for item in load_trailheads_from_geojson(private_parking_anchors_geojson)
    ]
    manual = [normalized_anchor(item, source_type="manual_route_design") for item in manual_anchor_trailheads(manual_design_jsons)]
    merged = merge_parking_anchors([*public, *private, *manual])
    return merged, {
        "public_trailhead_count": len(public),
        "private_strava_anchor_count": len(private),
        "manual_anchor_count": len(manual),
        "merged_anchor_count": len(merged),
        "parking_confidence_counts": dict(Counter(anchor.get("parking_confidence") for anchor in merged)),
    }


def feature_name(props: dict[str, Any]) -> str:
    return str(
        props.get("TrailName")
        or props.get("Name")
        or props.get("SystemName")
        or f"OSM {props.get('highway') or 'road'} connector"
    )


def feature_is_allowed_street_parking_probe(props: dict[str, Any]) -> bool:
    if str(props.get("source") or "").lower() != "openstreetmap":
        return False
    highway = str(props.get("highway") or "").lower()
    if highway in EXCLUDED_STREET_PROBE_HIGHWAYS or highway not in PUBLIC_ROAD_HIGHWAYS:
        return False
    access_values = {
        str(props.get(key) or "").lower()
        for key in ["access", "foot", "vehicle", "motor_vehicle"]
        if props.get(key) is not None
    }
    if any(value in {"no", "private"} for value in access_values):
        return False
    surface = str(props.get("surface") or "").lower()
    if surface in UNPAVED_SURFACES:
        return False
    if surface in PAVED_SURFACES:
        return True
    if highway in PAVED_BY_DEFAULT_HIGHWAYS:
        return True
    return False


def point_distance_to_segments(point: tuple[float, float], segments: list[dict[str, Any]]) -> float:
    distances = []
    for segment in segments:
        for key in ["start", "end", "center"]:
            if segment.get(key):
                distances.append(haversine_miles(point, segment[key]))
    return min(distances) if distances else math.inf


def point_distance_to_segment_access_points(point: tuple[float, float], segments: list[dict[str, Any]]) -> float:
    distances = []
    for segment in segments:
        if segment.get("start"):
            distances.append(haversine_miles(point, segment["start"]))
        if segment.get("end") and segment.get("direction") != "ascent":
            distances.append(haversine_miles(point, segment["end"]))
    return min(distances) if distances else math.inf


def street_parking_probes_for_segments(
    connector_geojson: dict[str, Any],
    segments: list[dict[str, Any]],
    *,
    max_distance_miles: float = ASSUMED_VEHICLE_ROAD_PARKING_MAX_MILES,
    limit: int = DEFAULT_MAX_STREET_PROBES_PER_OUTING,
) -> list[dict[str, Any]]:
    probes = []
    seen_names: set[str] = set()
    for feature in connector_geojson.get("features") or []:
        props = feature.get("properties") or {}
        if not feature_is_allowed_street_parking_probe(props):
            continue
        parts = iter_line_parts(feature.get("geometry") or {})
        if not parts:
            continue
        coords = parts[0]
        if not coords:
            continue
        midpoint = coords[len(coords) // 2]
        distance = point_distance_to_segment_access_points(midpoint, segments)
        if distance > max_distance_miles:
            continue
        name = feature_name(props)
        if name in seen_names:
            continue
        seen_names.add(name)
        probes.append(
            {
                "anchor_id": f"street-probe-{slugify(name)}",
                "name": f"{name} road-parking anchor",
                "lat": float(midpoint[1]),
                "lon": float(midpoint[0]),
                "has_parking": True,
                "parking_minutes": 8,
                "parking_confidence": ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
                "source": "openstreetmap_public_road_probe",
                "source_type": "assumed_paved_road_parking",
                "field_ready": False,
                "privacy": "public",
                "highway": props.get("highway"),
                "surface": props.get("surface"),
                "distance_to_required_segment_miles": round_miles(distance),
                "max_assumed_walk_to_road_miles": ASSUMED_VEHICLE_ROAD_PARKING_MAX_MILES,
            }
        )
    return sorted(
        probes,
        key=lambda item: (float(item["distance_to_required_segment_miles"]), item["name"]),
    )[:limit]


def segment_trail_order(segment_ids: list[int], segments_by_id: dict[int, dict[str, Any]]) -> list[str]:
    order = []
    seen = set()
    for seg_id in segment_ids:
        segment = segments_by_id.get(int(seg_id))
        if not segment:
            continue
        name = segment["trail_name"]
        if name not in seen:
            seen.add(name)
            order.append(name)
    return order


def segment_ids_for_trail_subset(
    segment_ids: list[int],
    segments_by_id: dict[int, dict[str, Any]],
    trail_names: set[str],
) -> list[int]:
    return [
        int(seg_id)
        for seg_id in segment_ids
        if int(seg_id) in segments_by_id and segments_by_id[int(seg_id)]["trail_name"] in trail_names
    ]


def trail_objects_for_segment_ids(
    segment_ids: list[int],
    segments_by_id: dict[int, dict[str, Any]],
    *,
    preferred_trail_order: list[str],
) -> list[dict[str, Any]]:
    segments = [segments_by_id[int(seg_id)] for seg_id in segment_ids if int(seg_id) in segments_by_id]
    trails = group_remaining_by_trail(segments)
    order_index = {name: index for index, name in enumerate(preferred_trail_order)}
    return sorted(trails, key=lambda trail: order_index.get(trail["trail_name"], len(order_index)))


def generate_trail_partitions(
    trail_names: list[str],
    *,
    max_subset_size: int = 3,
    limit: int = DEFAULT_MAX_PARTITIONS_PER_OUTING,
) -> list[tuple[set[str], set[str]]]:
    if len(trail_names) < 2:
        return []
    partitions = []
    seen = set()
    all_names = set(trail_names)
    max_size = min(max_subset_size, len(trail_names) - 1)
    for size in range(1, max_size + 1):
        for subset_tuple in itertools.combinations(trail_names, size):
            subset = set(subset_tuple)
            rest = all_names - subset
            if not rest:
                continue
            key_parts = sorted([tuple(sorted(subset)), tuple(sorted(rest))])
            key = tuple(key_parts)
            if key in seen:
                continue
            seen.add(key)
            partitions.append((subset, rest))
            if len(partitions) >= limit:
                return partitions
    return partitions


def forced_anchor_state(base_state: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(base_state)
    state["trailheads"] = [
        {
            "name": anchor["name"],
            "lat": float(anchor["lat"]),
            "lon": float(anchor["lon"]),
            "parking_minutes": int(anchor.get("parking_minutes") or base_state.get("parking_minutes") or 8),
            "has_parking": anchor_has_usable_parking(anchor),
            "source": anchor.get("source"),
            "parking_confidence": anchor.get("parking_confidence"),
            "privacy": anchor.get("privacy"),
        }
    ]
    return state


def anchor_has_usable_parking(anchor: dict[str, Any]) -> bool:
    if anchor.get("parking_rejected") is True:
        return False
    if anchor.get("has_parking") is True:
        return True
    if anchor.get("field_ready") is True:
        return True
    return False


def trail_variant_sequences(
    trails: list[dict[str, Any]],
    *,
    max_variants: int = 16,
) -> list[list[dict[str, Any]]]:
    variants = []
    if len(trails) <= 3:
        orders = itertools.permutations(trails)
        for order in orders:
            option_lists = [
                [trail, reverse_trail_orientation(trail)] if trail_is_reversible(trail) else [trail]
                for trail in order
            ]
            for choices in itertools.product(*option_lists):
                variants.append(list(choices))
                if len(variants) >= max_variants:
                    return variants
    else:
        variants.append(list(trails))
        reverse_order = [
            reverse_trail_orientation(trail) if trail_is_reversible(trail) else trail
            for trail in reversed(trails)
        ]
        variants.append(reverse_order)
    return variants


def anchor_distance_to_component(anchor: dict[str, Any], segments: list[dict[str, Any]]) -> float:
    return point_distance_to_segments((float(anchor["lon"]), float(anchor["lat"])), segments)


def anchor_is_assumed_road_parking(anchor: dict[str, Any]) -> bool:
    return (
        anchor.get("parking_confidence") == ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE
        or anchor.get("source_type") in {"assumed_vehicle_road_parking", "assumed_paved_road_parking"}
    )


def anchor_is_certifiable_parking(anchor: dict[str, Any]) -> bool:
    if anchor.get("parking_rejected") is True:
        return False
    if anchor.get("has_parking") is not True:
        return False
    if anchor_is_assumed_road_parking(anchor):
        return False
    confidence = str(anchor.get("parking_confidence") or "").lower()
    if confidence in {"manual_required", "street_parking_probe_manual_required"}:
        return False
    return anchor_field_ready(anchor)


def anchor_allowed_for_component(anchor: dict[str, Any], segments: list[dict[str, Any]]) -> bool:
    return anchor_allowed_for_trail_names(
        anchor,
        [str(segment.get("trail_name") or "") for segment in segments],
    )


def anchor_allowed_for_trail_names(anchor: dict[str, Any], trail_names: list[str]) -> bool:
    allowed = anchor.get("allowed_for_trails") or []
    if not allowed:
        return True
    allowed_names = {normalize_name(str(name)) for name in allowed}
    component_names = {normalize_name(str(name)) for name in trail_names}
    return bool(allowed_names & component_names)


def component_is_bogus_basin(segments: list[dict[str, Any]]) -> bool:
    return any(str(segment.get("trail_name") or "") in BOGUS_BASIN_TRAIL_NAMES for segment in segments)


def anchor_is_known_bogus_trailhead(anchor: dict[str, Any]) -> bool:
    return str(anchor.get("name") or "") in KNOWN_BOGUS_TRAILHEAD_NAMES


def ranked_anchors_for_component(
    anchors: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_MAX_ANCHORS_PER_COMPONENT,
    certifiable_connector_budget_miles: float = DEFAULT_CERTIFIABLE_ANCHOR_CONNECTOR_BUDGET_MILES,
) -> list[dict[str, Any]]:
    ranked = []
    bogus_basin_component = component_is_bogus_basin(segments)
    for anchor in anchors:
        if anchor.get("parking_rejected") is True:
            continue
        if bogus_basin_component and not anchor_is_known_bogus_trailhead(anchor):
            continue
        if not anchor_allowed_for_component(anchor, segments):
            continue
        if anchor_is_assumed_road_parking(anchor):
            distance = point_distance_to_segment_access_points(
                (float(anchor["lon"]), float(anchor["lat"])),
                segments,
            )
            if distance > ASSUMED_VEHICLE_ROAD_PARKING_MAX_MILES:
                continue
        else:
            distance = anchor_distance_to_component(anchor, segments)
        enriched = anchor | {
            "distance_to_component_miles": round_miles(distance),
            "access_search_ring": (
                "certifiable_parking_anchor"
                if anchor_is_certifiable_parking(anchor)
                else "closest_graph_valid_anchor"
            ),
            "certifiable_parking_anchor": anchor_is_certifiable_parking(anchor),
        }
        ranked.append((distance, -confidence_rank(anchor), enriched))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]["name"]))
    selected = [item[2] for item in ranked[:limit]]
    selected_ids = {str(anchor.get("anchor_id") or "") for anchor in selected}
    for distance, _rank, anchor in ranked:
        if not anchor.get("certifiable_parking_anchor"):
            continue
        if distance > certifiable_connector_budget_miles:
            continue
        anchor_id_value = str(anchor.get("anchor_id") or "")
        if anchor_id_value not in selected_ids:
            selected.append(anchor)
            selected_ids.add(anchor_id_value)
        break
    return selected


def best_candidate_for_component(
    *,
    segment_ids: list[int],
    anchor: dict[str, Any],
    base_state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any],
    elevation_sampler: Any,
    segments_by_id: dict[int, dict[str, Any]],
    preferred_trail_order: list[str],
) -> dict[str, Any] | None:
    trails = trail_objects_for_segment_ids(
        segment_ids,
        segments_by_id,
        preferred_trail_order=preferred_trail_order,
    )
    if not trails:
        return None
    state = forced_anchor_state(base_state, anchor)
    best: dict[str, Any] | None = None
    for variant in trail_variant_sequences(trails):
        candidate = candidate_from_trail_group(
            variant,
            state,
            performance_profile,
            connector_graph,
            candidate_type="multi_start_component_probe",
            elevation_sampler=elevation_sampler,
        )
        if best is None or candidate_sort_key(candidate) < candidate_sort_key(best):
            best = candidate
    return best


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int]:
    return (
        float(candidate.get("estimated_total_on_foot_miles") or math.inf),
        int(candidate.get("total_minutes") or 999999),
    )


def component_activity_minutes(component: dict[str, Any]) -> int:
    breakdown = component.get("time_breakdown_minutes") or {}
    total = int(component.get("total_minutes") or 0)
    subtract = sum(
        int(breakdown.get(key) or 0)
        for key in ["drive_to_trailhead", "parking_and_prep", "return_drive"]
    )
    return max(0, total - subtract)


def drive_minutes_between_anchors(
    left: dict[str, Any],
    right: dict[str, Any],
    drive_model: dict[str, Any],
) -> int:
    distance = haversine_miles(
        (float(left["lon"]), float(left["lat"])),
        (float(right["lon"]), float(right["lat"])),
    )
    minutes = distance * float(drive_model.get("straight_line_factor") or 1.2) * float(
        drive_model.get("minutes_per_mile") or 2.0
    )
    return max(math.ceil(minutes), int(drive_model.get("minimum_one_way_minutes") or 4))


def parking_blockers_for_anchor(anchor: dict[str, Any]) -> list[str]:
    blockers = []
    confidence = str(anchor.get("parking_confidence") or "")
    review_decision = str(anchor.get("parking_review_decision") or "").lower()
    if review_decision == "no":
        blockers.append("user rejected parking anchor")
    if review_decision == "maybe":
        blockers.append("user marked parking maybe")
    if confidence == "street_parking_probe_manual_required":
        blockers.append("street parking requires manual check")
    if confidence == "manual_required":
        blockers.append("parking/access requires manual verification")
    if confidence == ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE:
        blockers.append("assumed paved-road parking requires manual check")
    if confidence == "inferred_from_trailhead_layer" and anchor.get("field_ready") is not True:
        blockers.append("parking inferred from trailhead layer")
    if anchor.get("field_ready") is False:
        blockers.append("anchor is not field-ready")
    return sorted(set(blockers))


def classify_alternative(
    *,
    baseline_on_foot_miles: float,
    baseline_elapsed_p75_minutes: int,
    alternative_on_foot_miles: float,
    alternative_elapsed_p75_minutes: int,
    component_on_foot_miles: list[float],
    parking_blockers: list[str],
    car_access_benefit: str,
) -> dict[str, Any]:
    savings = round_miles(baseline_on_foot_miles - alternative_on_foot_miles)
    elapsed_delta = int(alternative_elapsed_p75_minutes - baseline_elapsed_p75_minutes)
    substantial_savings_tradeoff = (
        car_access_benefit == "refill_bail"
        and savings >= SUBSTANTIAL_SAVINGS_MIN_MILES
        and elapsed_delta <= SUBSTANTIAL_SAVINGS_MAX_ELAPSED_WORSE_MINUTES
        and elapsed_delta <= savings * SUBSTANTIAL_SAVINGS_MAX_MINUTES_PER_MILE
    )
    if savings >= PROMISING_MIN_SAVINGS_MILES and elapsed_delta <= PROMISING_MAX_ELAPSED_WORSE_MINUTES:
        underlying_status = "promising"
    elif substantial_savings_tradeoff:
        underlying_status = "promising"
    else:
        underlying_status = "not_worth_it"

    status = (
        "needs_parking_check"
        if parking_blockers and underlying_status != "not_worth_it"
        else underlying_status
    )
    if status == "needs_parking_check":
        recommendation = f"Review parking before promoting; underlying opportunity is {underlying_status}."
    elif status == "promising":
        recommendation = "Review as a lower-foot-mile multi-start option."
    else:
        recommendation = "Keep current outing unless there is a non-routing reason to split."
    return {
        "status": status,
        "underlying_status": underlying_status,
        "recommendation": recommendation,
        "on_foot_savings_miles": savings,
        "elapsed_delta_minutes": elapsed_delta,
    }


def summarize_candidate_component(
    *,
    candidate: dict[str, Any],
    segment_ids: list[int],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    time_estimates = candidate.get("time_estimates_minutes") or {}
    return {
        "segment_ids": segment_ids,
        "trail_names": candidate.get("trail_names") or [],
        "start_anchor": anchor,
        "official_miles": candidate.get("official_new_miles"),
        "on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
        "p75_total_minutes_if_standalone": candidate.get("total_minutes"),
        "p75_activity_minutes_without_home_drive": component_activity_minutes(candidate),
        "p75_moving_minutes": time_estimates.get("moving_effort_p75"),
        "route_finding_penalty_minutes": time_estimates.get("route_finding_penalty"),
        "parking_minutes": (candidate.get("time_breakdown_minutes") or {}).get("parking_and_prep", 8),
        "drive_to_trailhead_minutes": (candidate.get("time_breakdown_minutes") or {}).get("drive_to_trailhead"),
        "return_drive_minutes": (candidate.get("time_breakdown_minutes") or {}).get("return_drive"),
        "parking_confidence": anchor.get("parking_confidence"),
        "parking_blockers": parking_blockers_for_anchor(anchor),
        "route_status": candidate.get("route_status"),
        "less_optimal_flags": candidate.get("less_optimal_flags") or [],
    }


def elapsed_for_order(
    components: list[dict[str, Any]],
    *,
    drive_model: dict[str, Any],
    refill_repark_minutes: int,
) -> dict[str, Any]:
    first, second = components
    first_anchor = first["start_anchor"]
    second_anchor = second["start_anchor"]
    mid_drive = drive_minutes_between_anchors(first_anchor, second_anchor, drive_model)
    elapsed = (
        int(first["drive_to_trailhead_minutes"] or 0)
        + int(first["parking_minutes"] or 8)
        + int(first["p75_activity_minutes_without_home_drive"] or 0)
        + mid_drive
        + refill_repark_minutes
        + int(second["p75_activity_minutes_without_home_drive"] or 0)
        + int(second["return_drive_minutes"] or 0)
    )
    return {
        "component_order": [component["trail_names"] for component in components],
        "elapsed_p75_minutes": elapsed,
        "mid_drive_minutes": mid_drive,
    }


def build_alternative(
    *,
    outing_label: str,
    partition_index: int,
    baseline: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    drive_model: dict[str, Any],
    refill_repark_minutes: int = DEFAULT_REFILL_REPARK_MINUTES,
) -> dict[str, Any]:
    order_options = [
        elapsed_for_order([left, right], drive_model=drive_model, refill_repark_minutes=refill_repark_minutes),
        elapsed_for_order([right, left], drive_model=drive_model, refill_repark_minutes=refill_repark_minutes),
    ]
    selected_order = min(order_options, key=lambda item: item["elapsed_p75_minutes"])
    ordered_components = [left, right]
    if selected_order["component_order"] == [right["trail_names"], left["trail_names"]]:
        ordered_components = [right, left]
    on_foot = round_miles(sum(float(component.get("on_foot_miles") or 0.0) for component in ordered_components))
    official = round_miles(sum(float(component.get("official_miles") or 0.0) for component in ordered_components))
    component_miles = [float(component.get("on_foot_miles") or 0.0) for component in ordered_components]
    anchor_ids = [component["start_anchor"]["anchor_id"] for component in ordered_components]
    car_access_benefit = "refill_bail"
    same_anchor = anchor_ids[0] == anchor_ids[1]
    if same_anchor:
        car_access_benefit = "none"
    parking_blockers = sorted(
        {
            blocker
            for component in ordered_components
            for blocker in component.get("parking_blockers") or []
        }
    )
    classification = classify_alternative(
        baseline_on_foot_miles=float(baseline["on_foot_miles"]),
        baseline_elapsed_p75_minutes=int(baseline["elapsed_p75_minutes"]),
        alternative_on_foot_miles=on_foot,
        alternative_elapsed_p75_minutes=int(selected_order["elapsed_p75_minutes"]),
        component_on_foot_miles=component_miles,
        parking_blockers=parking_blockers,
        car_access_benefit=car_access_benefit,
    )
    if same_anchor:
        classification["status"] = "not_worth_it"
        classification["underlying_status"] = "not_worth_it"
        classification["recommendation"] = "Rejected: both components use the same parked start."
    return {
        "alternative_id": f"{slugify(outing_label).upper()}-MS-{partition_index:02d}",
        "status": classification["status"],
        "underlying_status": classification["underlying_status"],
        "recommendation": classification["recommendation"],
        "components": ordered_components,
        "official_miles": official,
        "on_foot_miles": on_foot,
        "elapsed_p75_minutes": int(selected_order["elapsed_p75_minutes"]),
        "mid_drive_minutes": int(selected_order["mid_drive_minutes"]),
        "refill_repark_minutes": refill_repark_minutes,
        "on_foot_savings_miles": classification["on_foot_savings_miles"],
        "elapsed_delta_minutes": classification["elapsed_delta_minutes"],
        "car_access_benefit": car_access_benefit,
        "parking_blockers": parking_blockers,
    }


def connector_corridor_between_anchors(
    *,
    replacement_anchor: dict[str, Any],
    tie_in_anchor: dict[str, Any],
    connector_graph: dict[str, Any] | None,
    base_state: dict[str, Any],
) -> dict[str, Any]:
    outing_model = get_outing_model(base_state)
    replacement_point = (float(replacement_anchor["lon"]), float(replacement_anchor["lat"]))
    tie_in_point = (float(tie_in_anchor["lon"]), float(tie_in_anchor["lat"]))
    mapped = shortest_connector_path(
        replacement_point,
        tie_in_point,
        connector_graph,
        float(outing_model.get("mapped_connector_snap_tolerance_miles", 0.02)),
    )
    if mapped:
        one_way = float(mapped["distance_miles"])
        return {
            "source": "mapped_graph",
            "graph_validated": True,
            "one_way_miles": round_miles(one_way),
            "round_trip_miles": round_miles(one_way * 2),
            "one_way_connector_miles": mapped["connector_miles"],
            "one_way_official_repeat_miles": mapped["official_repeat_miles"],
            "connector_names": mapped["connector_names"],
            "connector_classes": mapped.get("connector_classes", []),
            "connector_edges": mapped.get("connector_edges", []),
            "official_repeat_segment_ids": mapped["official_repeat_segment_ids"],
        }

    direct_gap = haversine_miles(replacement_point, tie_in_point)
    return {
        "source": "direct_gap_estimate",
        "graph_validated": direct_gap <= ASSUMED_VEHICLE_ROAD_PARKING_MAX_MILES,
        "one_way_miles": round_miles(direct_gap),
        "round_trip_miles": round_miles(direct_gap * 2),
        "one_way_connector_miles": round_miles(direct_gap),
        "one_way_official_repeat_miles": 0,
        "connector_names": [],
        "connector_classes": [],
        "connector_edges": [],
        "official_repeat_segment_ids": [],
    }


def classify_certifiable_anchor_repair(
    *,
    connector_budget_passed: bool,
    graph_validated: bool,
    adjusted_savings_miles: float,
    adjusted_elapsed_delta_minutes: int,
) -> str:
    if not graph_validated:
        return "needs_connector_graph"
    if not connector_budget_passed:
        return "connector_budget_exceeded"
    if adjusted_savings_miles <= 0:
        return "not_worth_it_connector_tax"
    if (
        adjusted_savings_miles >= PROMISING_MIN_SAVINGS_MILES
        and adjusted_elapsed_delta_minutes <= PROMISING_MAX_ELAPSED_WORSE_MINUTES
    ):
        return "redesign_candidate"
    if adjusted_savings_miles >= SUBSTANTIAL_SAVINGS_MIN_MILES:
        return "human_cost_research"
    return "not_worth_it_connector_tax"


def repair_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int]:
    status_rank = {
        "redesign_candidate": 0,
        "human_cost_research": 1,
        "connector_budget_exceeded": 2,
        "needs_connector_graph": 3,
        "not_worth_it_connector_tax": 4,
    }.get(str(candidate.get("status") or ""), 9)
    return (
        status_rank,
        -float(candidate.get("adjusted_on_foot_savings_miles") or 0.0),
        int(candidate.get("adjusted_elapsed_delta_minutes") or 999999),
    )


def certifiable_anchor_repair_candidates(
    *,
    baseline: dict[str, Any],
    alternative: dict[str, Any],
    anchors: list[dict[str, Any]],
    connector_graph: dict[str, Any] | None,
    base_state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_budget_miles: float = DEFAULT_CERTIFIABLE_ANCHOR_CONNECTOR_BUDGET_MILES,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if not alternative.get("parking_blockers"):
        return []
    if alternative.get("underlying_status") == "not_worth_it":
        return []

    baseline_overhead = round_miles(
        max(float(baseline.get("on_foot_miles") or 0.0) - float(baseline.get("official_miles") or 0.0), 0.0)
    )
    fallback_p75_pace = float(performance_profile.get("fallback_pace_min_per_mile") or 16.0) * 1.12
    repairs = []
    for component in alternative.get("components") or []:
        blocked_anchor = component.get("start_anchor") or {}
        if not parking_blockers_for_anchor(blocked_anchor):
            continue
        component_trail_names = [str(name) for name in component.get("trail_names") or []]
        blocked_point = (float(blocked_anchor["lon"]), float(blocked_anchor["lat"]))
        replacement_options = []
        for replacement_anchor in anchors:
            if str(replacement_anchor.get("anchor_id") or "") == str(blocked_anchor.get("anchor_id") or ""):
                continue
            if not anchor_is_certifiable_parking(replacement_anchor):
                continue
            if not anchor_allowed_for_trail_names(replacement_anchor, component_trail_names):
                continue
            direct_gap = haversine_miles(
                (float(replacement_anchor["lon"]), float(replacement_anchor["lat"])),
                blocked_point,
            )
            if direct_gap > connector_budget_miles:
                continue
            replacement_options.append((direct_gap, -confidence_rank(replacement_anchor), replacement_anchor))
        replacement_options.sort(key=lambda item: (item[0], item[1], item[2]["name"]))
        for _direct_gap, _rank, replacement_anchor in replacement_options[:limit]:
            corridor = connector_corridor_between_anchors(
                replacement_anchor=replacement_anchor,
                tie_in_anchor=blocked_anchor,
                connector_graph=connector_graph,
                base_state=base_state,
            )
            round_trip = float(corridor["round_trip_miles"])
            adjusted_on_foot = round_miles(float(alternative.get("on_foot_miles") or 0.0) + round_trip)
            adjusted_savings = round_miles(float(baseline.get("on_foot_miles") or 0.0) - adjusted_on_foot)
            extra_minutes = ceil_minutes(round_trip * fallback_p75_pace)
            adjusted_delta = int(alternative.get("elapsed_delta_minutes") or 0) + extra_minutes
            connector_budget_passed = float(corridor["one_way_miles"]) <= connector_budget_miles
            status = classify_certifiable_anchor_repair(
                connector_budget_passed=connector_budget_passed,
                graph_validated=bool(corridor["graph_validated"]),
                adjusted_savings_miles=adjusted_savings,
                adjusted_elapsed_delta_minutes=adjusted_delta,
            )
            repairs.append(
                {
                    "status": status,
                    "replacement_ready": False,
                    "promotion_gate": "requires_regenerated_route_source_gpx_cues_p75_p90_and_certification_audits",
                    "failure_class": "wrong_anchor_or_missing_access_evidence_not_goal_failure",
                    "blocked_anchor": blocked_anchor,
                    "replacement_anchor": replacement_anchor,
                    "tie_in_waypoint": {
                        "name": blocked_anchor.get("name"),
                        "anchor_id": blocked_anchor.get("anchor_id"),
                    },
                    "component_trail_names": component_trail_names,
                    "corridor": corridor,
                    "connector_budget_miles": connector_budget_miles,
                    "connector_budget_passed": connector_budget_passed,
                    "connector_tax_less_than_baseline_overhead": round_trip < baseline_overhead,
                    "baseline_overhead_miles": baseline_overhead,
                    "adjusted_on_foot_miles": adjusted_on_foot,
                    "adjusted_on_foot_savings_miles": adjusted_savings,
                    "extra_p75_minutes": extra_minutes,
                    "adjusted_elapsed_delta_minutes": adjusted_delta,
                }
            )
    repairs.sort(key=repair_sort_key)
    return repairs[:limit]


def label_for_component(package: dict[str, Any], component: dict[str, Any], component_index: int) -> str:
    explicit = component.get("field_menu_label") or component.get("label")
    if explicit:
        return str(explicit)
    package_number = package.get("package_number")
    if len(package.get("components") or []) > 1 and package_number is not None:
        suffix = chr(ord("A") + component_index)
        return f"{package_number}{suffix}"
    return str(package_number or component.get("candidate_id") or "")


def baseline_for_component(package: dict[str, Any], component: dict[str, Any], component_index: int) -> dict[str, Any]:
    label = label_for_component(package, component, component_index)
    elapsed = (
        component.get("total_minutes")
        or (component.get("time_estimates_minutes") or {}).get("door_to_door_p75")
        or 0
    )
    return {
        "outing_id": component.get("outing_id"),
        "label": label,
        "package_number": package.get("package_number"),
        "candidate_id": component.get("candidate_id"),
        "trailhead": component.get("trailhead"),
        "trail_names": component.get("trail_names") or [],
        "segment_ids": [int(value) for value in component.get("segment_ids") or []],
        "official_miles": float(component.get("official_miles") or 0.0),
        "on_foot_miles": float(component.get("on_foot_miles") or 0.0),
        "elapsed_p75_minutes": int(elapsed or 0),
    }


def best_component_from_anchors(
    *,
    segment_ids: list[int],
    anchors: list[dict[str, Any]],
    base_state: dict[str, Any],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any],
    elevation_sampler: Any,
    segments_by_id: dict[int, dict[str, Any]],
    preferred_trail_order: list[str],
    cache: dict[tuple[tuple[int, ...], str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    segments = [segments_by_id[seg_id] for seg_id in segment_ids if seg_id in segments_by_id]
    best_summary = None
    for anchor in ranked_anchors_for_component(anchors, segments):
        cache_key = (tuple(sorted(segment_ids)), anchor["anchor_id"])
        if cache_key not in cache:
            cache[cache_key] = best_candidate_for_component(
                segment_ids=segment_ids,
                anchor=anchor,
                base_state=base_state,
                performance_profile=performance_profile,
                connector_graph=connector_graph,
                elevation_sampler=elevation_sampler,
                segments_by_id=segments_by_id,
                preferred_trail_order=preferred_trail_order,
            )
        candidate = cache[cache_key]
        if not candidate:
            continue
        summary = summarize_candidate_component(candidate=candidate, segment_ids=segment_ids, anchor=anchor)
        if best_summary is None or (
            float(summary.get("on_foot_miles") or math.inf),
            int(summary.get("p75_total_minutes_if_standalone") or 999999),
        ) < (
            float(best_summary.get("on_foot_miles") or math.inf),
            int(best_summary.get("p75_total_minutes_if_standalone") or 999999),
        ):
            best_summary = summary
    return best_summary


def audit_outing_component(
    *,
    package: dict[str, Any],
    component: dict[str, Any],
    component_index: int,
    base_anchors: list[dict[str, Any]],
    connector_geojson: dict[str, Any],
    base_state: dict[str, Any],
    parking_review_decisions: dict[str, dict[str, Any]],
    performance_profile: dict[str, Any],
    connector_graph: dict[str, Any],
    elevation_sampler: Any,
    segments_by_id: dict[int, dict[str, Any]],
    max_partitions: int,
    certifiable_anchor_connector_budget_miles: float,
) -> dict[str, Any]:
    baseline = baseline_for_component(package, component, component_index)
    segment_ids = baseline["segment_ids"]
    trail_order = segment_trail_order(segment_ids, segments_by_id)
    required_segments = [segments_by_id[seg_id] for seg_id in segment_ids if seg_id in segments_by_id]
    street_probes = street_parking_probes_for_segments(connector_geojson, required_segments)
    anchors = apply_parking_review_decisions(
        merge_parking_anchors([*base_anchors, *street_probes]),
        parking_review_decisions,
    )
    cache: dict[tuple[tuple[int, ...], str], dict[str, Any] | None] = {}
    alternatives = []
    for index, (left_trails, right_trails) in enumerate(
        generate_trail_partitions(trail_order, limit=max_partitions),
        1,
    ):
        left_ids = segment_ids_for_trail_subset(segment_ids, segments_by_id, left_trails)
        right_ids = segment_ids_for_trail_subset(segment_ids, segments_by_id, right_trails)
        if not left_ids or not right_ids:
            continue
        left = best_component_from_anchors(
            segment_ids=left_ids,
            anchors=anchors,
            base_state=base_state,
            performance_profile=performance_profile,
            connector_graph=connector_graph,
            elevation_sampler=elevation_sampler,
            segments_by_id=segments_by_id,
            preferred_trail_order=trail_order,
            cache=cache,
        )
        right = best_component_from_anchors(
            segment_ids=right_ids,
            anchors=anchors,
            base_state=base_state,
            performance_profile=performance_profile,
            connector_graph=connector_graph,
            elevation_sampler=elevation_sampler,
            segments_by_id=segments_by_id,
            preferred_trail_order=trail_order,
            cache=cache,
        )
        if not left or not right:
            continue
        alternative = build_alternative(
            outing_label=baseline["label"],
            partition_index=index,
            baseline=baseline,
            left=left,
            right=right,
            drive_model=base_state.get("drive_model") or {},
        )
        repairs = certifiable_anchor_repair_candidates(
            baseline=baseline,
            alternative=alternative,
            anchors=anchors,
            connector_graph=connector_graph,
            base_state=base_state,
            performance_profile=performance_profile,
            connector_budget_miles=certifiable_anchor_connector_budget_miles,
        )
        if repairs:
            alternative["certifiable_anchor_repair_candidates"] = repairs
        if sorted(left_ids + right_ids) != sorted(segment_ids):
            alternative["status"] = "not_worth_it"
            alternative["recommendation"] = "Rejected: segment coverage mismatch."
        alternatives.append(alternative)

    alternatives = sorted(
        alternatives,
        key=lambda item: (
            status_sort_key(str(item.get("status"))),
            -float(item.get("on_foot_savings_miles") or 0.0),
            int(item.get("elapsed_delta_minutes") or 0),
        ),
    )[:3]
    return {
        "outing_id": baseline.get("outing_id"),
        "label": baseline["label"],
        "package_number": baseline["package_number"],
        "candidate_id": baseline["candidate_id"],
        "baseline": baseline,
        "street_probe_count": len(street_probes),
        "partition_count_considered": len(generate_trail_partitions(trail_order, limit=max_partitions)),
        "alternatives": alternatives,
        "best_status": alternatives[0]["status"] if alternatives else "not_evaluated",
    }


def status_sort_key(status: str) -> int:
    return {
        "promising": 0,
        "needs_parking_check": 1,
        "not_worth_it": 2,
        "not_evaluated": 4,
    }.get(status, 9)


def build_report(
    *,
    field_menu_json: Path,
    official_geojson: Path,
    state_json: Path,
    public_trailheads_geojson: Path,
    private_parking_anchors_geojson: Path,
    manual_design_jsons: list[Path],
    parking_review_decisions_json: Path | None,
    connector_geojson_path: Path,
    dem_tif: Path,
    dem_summary_json: Path,
    max_partitions_per_outing: int = DEFAULT_MAX_PARTITIONS_PER_OUTING,
    certifiable_anchor_connector_budget_miles: float = DEFAULT_CERTIFIABLE_ANCHOR_CONNECTOR_BUDGET_MILES,
    route_labels: set[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    field_menu = read_json(field_menu_json)
    official_segments, official_meta = load_official_segments(official_geojson)
    segments_by_id = {int(segment["seg_id"]): segment for segment in official_segments}
    base_state = load_state(state_json)
    performance_profile = build_performance_profile(
        state=base_state,
        strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
        activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
        activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
        segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
    )
    connector_graph = load_connector_graph(connector_geojson_path, official_segments=official_segments)
    connector_geojson = read_json(connector_geojson_path)
    elevation_sampler = load_dem_context(dem_tif, dem_summary_json)["sampler"]
    base_anchors, anchor_summary = load_parking_anchors(
        public_trailheads_geojson=public_trailheads_geojson,
        private_parking_anchors_geojson=private_parking_anchors_geojson,
        manual_design_jsons=manual_design_jsons,
    )
    parking_review_decisions = load_parking_review_decisions(parking_review_decisions_json)
    outings = []
    for package in field_menu.get("packages") or []:
        for component_index, component in enumerate(package.get("components") or []):
            if len(component.get("segment_ids") or []) < 2:
                continue
            component_label = label_for_component(package, component, component_index)
            if route_labels and component_label not in route_labels:
                continue
            outings.append(
                audit_outing_component(
                    package=package,
                    component=component,
                    component_index=component_index,
                    base_anchors=base_anchors,
                    connector_geojson=connector_geojson,
                    base_state=base_state,
                    parking_review_decisions=parking_review_decisions,
                    performance_profile=performance_profile,
                    connector_graph=connector_graph,
                    elevation_sampler=elevation_sampler,
                    segments_by_id=segments_by_id,
                    max_partitions=max_partitions_per_outing,
                    certifiable_anchor_connector_budget_miles=certifiable_anchor_connector_budget_miles,
                )
            )
    all_alternatives = [
        alternative
        for outing in outings
        for alternative in outing.get("alternatives") or []
    ]
    status_counts = Counter(alternative.get("status") for alternative in all_alternatives)
    repair_candidates = [
        repair
        for alternative in all_alternatives
        for repair in alternative.get("certifiable_anchor_repair_candidates") or []
    ]
    repair_status_counts = Counter(repair.get("status") for repair in repair_candidates)
    candidate_outings = [
        outing
        for outing in outings
        if any(
            alternative.get("status") in {"promising", "needs_parking_check"}
            for alternative in outing.get("alternatives") or []
        )
    ]
    return {
        "objective": "evaluate field-menu outings for one-transfer multi-start alternatives with parking-anchor review",
        "generated_at": generated_at,
        "filters": {
            "route_labels": sorted(route_labels) if route_labels else None,
        },
        "source_files": {
            "field_menu_json": display_path(field_menu_json),
            "official_geojson": display_path(official_geojson),
            "state_json": display_path(state_json),
            "public_trailheads_geojson": display_path(public_trailheads_geojson),
            "private_parking_anchors_geojson": display_path(private_parking_anchors_geojson),
            "manual_design_jsons": [display_path(path) for path in manual_design_jsons],
            "parking_review_decisions_json": display_path(parking_review_decisions_json)
            if parking_review_decisions_json and parking_review_decisions_json.exists()
            else None,
            "connector_geojson": display_path(connector_geojson_path),
            "dem_tif": display_path(dem_tif),
        },
        "policy": {
            "max_mid_session_car_transfers": 1,
            "refill_repark_minutes": DEFAULT_REFILL_REPARK_MINUTES,
            "promising_min_on_foot_savings_miles": PROMISING_MIN_SAVINGS_MILES,
            "promising_max_elapsed_worse_minutes": PROMISING_MAX_ELAPSED_WORSE_MINUTES,
            "substantial_savings_min_on_foot_savings_miles": SUBSTANTIAL_SAVINGS_MIN_MILES,
            "substantial_savings_max_elapsed_worse_minutes": SUBSTANTIAL_SAVINGS_MAX_ELAPSED_WORSE_MINUTES,
            "substantial_savings_max_minutes_per_mile_saved": SUBSTANTIAL_SAVINGS_MAX_MINUTES_PER_MILE,
            "road_parking_policy": ASSUMED_VEHICLE_ROAD_PARKING_CONFIDENCE,
            "road_parking_max_assumed_walk_miles": ASSUMED_VEHICLE_ROAD_PARKING_MAX_MILES,
            "road_parking_access_basis": "trailhead_or_segment_access_endpoint_not_segment_center",
            "road_parking_field_ready_policy": "assumed_paved_road_parking_always_needs_manual_check",
            "certifiable_anchor_connector_budget_miles": certifiable_anchor_connector_budget_miles,
            "certifiable_anchor_policy": (
                "when a road/probe anchor blocks a promising split, price a mapped connector "
                "from the nearest source/user-certified parking anchor to that blocked tie-in waypoint"
            ),
            "private_strava_anchor_field_ready_policy": (
                "private Strava-derived anchors with parking are prior real user parking, "
                "not generic parking-review blockers"
            ),
            "bogus_basin_parking_policy": "only_known_trailhead_or_lodge_parking_anchors",
            "known_bogus_trailhead_names": sorted(KNOWN_BOGUS_TRAILHEAD_NAMES),
            "parking_review_decision_policy": "yes clears generic paved-road blockers; maybe remains needs-parking-check; no is excluded",
        },
        "anchor_summary": anchor_summary,
        "parking_review_summary": {
            "decision_count": len(parking_review_decisions),
            "decision_counts": dict(
                Counter(str(item.get("decision") or "unknown").lower() for item in parking_review_decisions.values())
            ),
        },
        "official_meta": {
            **official_meta,
            "path": display_path(official_geojson),
        },
        "summary": {
            "outing_component_count": len(outings),
            "alternative_count": len(all_alternatives),
            "candidate_outing_count": len(candidate_outings),
            "status_counts": dict(sorted(status_counts.items())),
            "certifiable_anchor_repair_candidate_count": len(repair_candidates),
            "certifiable_anchor_repair_status_counts": dict(sorted(repair_status_counts.items())),
            "redesign_candidate_count": repair_status_counts.get("redesign_candidate", 0),
            "west_climb_candidate_found": any(
                str(outing.get("candidate_id"))
                == "combo-full-sail-trail-buena-vista-trail-bob-smylie-36th-street-chute"
                and any(
                    alternative.get("status") in {"promising", "needs_parking_check"}
                    for alternative in outing.get("alternatives") or []
                )
                for outing in outings
            ),
        },
        "outings": outings,
        "alternatives": all_alternatives,
        "caveats": [
            "This audit is review-only and does not change the field packet, map, or GPX outputs.",
            "Car transfer minutes count toward elapsed p75 but never toward on-foot or official challenge miles.",
            "Unreviewed or maybe public paved-road anchors are assumed parking within 0.10 mile for route math, but remain parking-review blockers until manually accepted.",
            "Certifiable-anchor repair candidates are review-only waypoint-corridor price checks; they are not route replacements until regenerated source, GPX, cues, p75/p90, and certification audits pass.",
            "Bogus Basin alternatives use only known lodge/trailhead parking anchors, not road shoulder or cat-track probes.",
            "Private Strava-derived anchors are treated as prior real user parking; exact coordinates are only present in the private report.",
        ],
    }


def public_safe_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    result = {
        "anchor_id": anchor.get("anchor_id"),
        "name": anchor.get("name"),
        "parking_confidence": anchor.get("parking_confidence"),
        "source": anchor.get("source"),
        "source_type": anchor.get("source_type"),
        "field_ready": anchor.get("field_ready"),
        "privacy": anchor.get("privacy"),
        "distance_to_component_miles": anchor.get("distance_to_component_miles"),
    }
    if anchor.get("privacy") == "private_exact_coordinates":
        result["name"] = "Private Strava parking anchor"
    return {key: value for key, value in result.items() if value is not None}


def public_safe_report(value: Any) -> Any:
    if isinstance(value, list):
        return [public_safe_report(item) for item in value]
    if not isinstance(value, dict):
        return value
    if {"lat", "lon"}.issubset(value.keys()) and (
        value.get("parking_confidence") or value.get("anchor_id") or value.get("source")
    ):
        return public_safe_anchor(value)
    sanitized = {}
    for key, child in value.items():
        if key in {"lat", "lon", "coordinates", "geometry", "raw_activity_ids", "activity_ids"}:
            continue
        sanitized[key] = public_safe_report(child)
    return sanitized


def render_md(report: dict[str, Any], *, public_safe: bool = False) -> str:
    summary = report["summary"]
    lines = [
        "# Multi-Start Alternative Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Outing components evaluated: {summary['outing_component_count']}",
        f"- Alternatives retained for review: {summary['alternative_count']}",
        f"- Outings with review candidates: {summary['candidate_outing_count']}",
        f"- Status counts: {json.dumps(summary['status_counts'], sort_keys=True)}",
        f"- Certifiable-anchor repair candidates: {summary.get('certifiable_anchor_repair_candidate_count', 0)}",
        f"- Certifiable-anchor repair status counts: {json.dumps(summary.get('certifiable_anchor_repair_status_counts', {}), sort_keys=True)}",
        f"- West Climb candidate found: {summary['west_climb_candidate_found']}",
        "",
        "## Review Candidates",
        "",
        "| Outing | Status | Baseline on-foot | Alt on-foot | Savings | Delta min | Benefit | Components | Blockers |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    candidate_rows = []
    for outing in report.get("outings") or []:
        baseline = outing.get("baseline") or {}
        for alternative in outing.get("alternatives") or []:
            if alternative.get("status") == "not_worth_it":
                continue
            components = " / ".join(
                "+".join(component.get("trail_names") or [])
                for component in alternative.get("components") or []
            )
            blockers = "; ".join(alternative.get("parking_blockers") or [])
            candidate_rows.append(
                (
                    status_sort_key(str(alternative.get("status"))),
                    str(outing.get("label") or ""),
                    f"| {outing.get('label')} | {alternative.get('status')} | "
                    f"{round_miles(float(baseline.get('on_foot_miles') or 0.0))} | "
                    f"{alternative.get('on_foot_miles')} | "
                    f"{alternative.get('on_foot_savings_miles')} | "
                    f"{alternative.get('elapsed_delta_minutes')} | "
                    f"{alternative.get('car_access_benefit')} | {components} | {blockers} |",
                )
            )
    for _status, _label, line in sorted(candidate_rows)[:40]:
        lines.append(line)
    if not candidate_rows:
        lines.append("| n/a | n/a |  |  |  |  |  |  |  |")
    repair_rows = []
    for outing in report.get("outings") or []:
        for alternative in outing.get("alternatives") or []:
            for repair in alternative.get("certifiable_anchor_repair_candidates") or []:
                blocked_anchor = repair.get("blocked_anchor") or {}
                replacement_anchor = repair.get("replacement_anchor") or {}
                corridor = repair.get("corridor") or {}
                repair_rows.append(
                    (
                        repair_sort_key(repair),
                        f"| {outing.get('label')} | {alternative.get('alternative_id')} | "
                        f"{repair.get('status')} | {blocked_anchor.get('name')} | "
                        f"{replacement_anchor.get('name')} | {corridor.get('one_way_miles')} | "
                        f"{corridor.get('round_trip_miles')} | {repair.get('adjusted_on_foot_savings_miles')} | "
                        f"{repair.get('adjusted_elapsed_delta_minutes')} | "
                        f"{repair.get('connector_budget_passed')} | {corridor.get('graph_validated')} |",
                    )
                )
    lines.extend(
        [
            "",
            "## Certifiable Anchor Repair Candidates",
            "",
            "| Outing | Alternative | Status | Blocked tie-in | Certifiable anchor | One-way mi | Round-trip tax | Adjusted savings | Adjusted delta min | Budget pass | Graph valid |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for _sort_key, line in sorted(repair_rows)[:40]:
        lines.append(line)
    if not repair_rows:
        lines.append("| n/a | n/a | n/a |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in report.get("caveats") or []:
        lines.append(f"- {caveat}")
    if public_safe:
        lines.append("- This Markdown is public-safe and omits private exact coordinates.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-menu-json", type=Path, default=DEFAULT_FIELD_MENU_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--public-trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--private-parking-anchors-geojson", type=Path, default=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON)
    parser.add_argument("--manual-design-json", type=Path, action="append", default=None)
    parser.add_argument("--parking-review-decisions-json", type=Path, default=DEFAULT_PARKING_REVIEW_DECISIONS_JSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--private-output-dir", type=Path, default=DEFAULT_PRIVATE_OUTPUT_DIR)
    parser.add_argument("--public-output-dir", type=Path, default=DEFAULT_PUBLIC_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-partitions-per-outing", type=int, default=DEFAULT_MAX_PARTITIONS_PER_OUTING)
    parser.add_argument("--route-label", action="append", default=None)
    parser.add_argument(
        "--certifiable-anchor-connector-budget-miles",
        type=float,
        default=DEFAULT_CERTIFIABLE_ANCHOR_CONNECTOR_BUDGET_MILES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_design_jsons = args.manual_design_json or DEFAULT_MANUAL_DESIGN_JSONS
    report = build_report(
        field_menu_json=args.field_menu_json,
        official_geojson=args.official_geojson,
        state_json=args.state_json,
        public_trailheads_geojson=args.public_trailheads_geojson,
        private_parking_anchors_geojson=args.private_parking_anchors_geojson,
        manual_design_jsons=manual_design_jsons,
        parking_review_decisions_json=args.parking_review_decisions_json,
        connector_geojson_path=args.connector_geojson,
        dem_tif=args.dem_tif,
        dem_summary_json=args.dem_summary_json,
        max_partitions_per_outing=args.max_partitions_per_outing,
        certifiable_anchor_connector_budget_miles=args.certifiable_anchor_connector_budget_miles,
        route_labels=set(args.route_label or []) or None,
    )
    private_json = args.private_output_dir / f"{args.basename}.json"
    private_md = args.private_output_dir / f"{args.basename}.md"
    public_json = args.public_output_dir / f"{args.basename}.json"
    public_md = args.public_output_dir / f"{args.basename}.md"
    write_json(private_json, report)
    private_md.write_text(render_md(report, public_safe=False), encoding="utf-8")
    public_report = public_safe_report(report)
    write_json(public_json, public_report)
    public_md.write_text(render_md(public_report, public_safe=True), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id=args.basename,
        command="python years/2026/scripts/multi_start_alternative_audit.py",
        inputs=[
            args.field_menu_json,
            args.official_geojson,
            args.state_json,
            args.public_trailheads_geojson,
            args.private_parking_anchors_geojson,
            args.parking_review_decisions_json,
            args.connector_geojson,
        ],
        outputs=[private_json, private_md, public_json, public_md],
        metadata={"report_generated_at": report["generated_at"]},
    )
    write_manifest(args.private_output_dir / f"{args.basename}-artifact-manifest.json", manifest)
    print(f"Wrote {display_path(private_json)}")
    print(f"Wrote {display_path(public_json)}")


if __name__ == "__main__":
    main()
