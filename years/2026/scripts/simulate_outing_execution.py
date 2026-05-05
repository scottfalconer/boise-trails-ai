#!/usr/bin/env python3
"""Simulate full outing execution from car logistics through return home."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_STATE_PATH,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    METERS_PER_MILE,
    ceil_minutes,
    classify_bucket,
    get_drive_model,
    haversine_miles,
    load_state,
    normalize_name,
    read_json,
    write_json,
)


DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "personal-route-menu.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "experiments" / "2026-05-04-outing-execution-simulation"
RouteProvider = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
BUCKET_ORDER = ["under_1_hour", "one_to_two_hours", "two_to_three_hours", "three_to_four_hours", "four_plus_hours"]


def load_parking_index(trailheads_geojson: Path | None) -> dict[str, dict[str, Any]]:
    if not trailheads_geojson or not trailheads_geojson.exists():
        return {}
    data = read_json(trailheads_geojson)
    index = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        name = props.get("facility_name") or props.get("FacilityName")
        if not name:
            continue
        index[normalize_name(str(name))] = {
            "name": name,
            "lat": props.get("lat"),
            "lon": props.get("lon"),
            "facility_id": props.get("facility_id"),
            "facility_status": props.get("facility_status"),
            "address": props.get("address"),
            "has_parking": props.get("has_parking"),
            "has_restroom": props.get("has_restroom"),
            "has_water": props.get("has_water"),
            "parking_confidence": props.get("parking_confidence"),
            "source": "city_parks_facilities",
            "nearest_official_segment_id": props.get("nearest_official_segment_id"),
            "nearest_official_trail_name": props.get("nearest_official_trail_name"),
            "nearest_official_distance_miles": props.get("nearest_official_distance_miles"),
            "nearest_open_trail_id": props.get("nearest_open_trail_id"),
            "nearest_open_trail_name": props.get("nearest_open_trail_name"),
            "nearest_open_trail_distance_miles": props.get("nearest_open_trail_distance_miles"),
        }
    return index


def origin_from_drive_model(drive_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": drive_model.get("origin_label") or "Planning origin",
        "lat": float(drive_model["origin_lat"]),
        "lon": float(drive_model["origin_lon"]),
    }


def fallback_drive_route(
    origin: dict[str, Any],
    destination: dict[str, Any],
    drive_model: dict[str, Any],
    reason: str = "drive_model_only",
) -> dict[str, Any]:
    origin_point = (float(origin["lon"]), float(origin["lat"]))
    destination_point = (float(destination["lon"]), float(destination["lat"]))
    distance_miles = haversine_miles(origin_point, destination_point) * float(
        drive_model.get("straight_line_factor", 1.25)
    )
    duration = max(
        ceil_minutes(distance_miles * float(drive_model.get("minutes_per_mile", 2.2))),
        int(drive_model.get("minimum_one_way_minutes", 5)),
    )
    return {
        "provider": "drive_model",
        "route_validated": False,
        "validation_reason": reason,
        "distance_miles": round(distance_miles, 2),
        "duration_minutes": duration,
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [float(origin["lon"]), float(origin["lat"])],
                [float(destination["lon"]), float(destination["lat"])],
            ],
        },
    }


def osrm_drive_route(
    origin: dict[str, Any],
    destination: dict[str, Any],
    drive_model: dict[str, Any],
    base_url: str,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError:
        return fallback_drive_route(origin, destination, drive_model, reason="requests_unavailable")

    coord_pair = (
        f"{float(origin['lon'])},{float(origin['lat'])};"
        f"{float(destination['lon'])},{float(destination['lat'])}"
    )
    url = (
        base_url.rstrip("/")
        + "/route/v1/driving/"
        + coord_pair
        + "?overview=simplified&geometries=geojson&steps=false"
    )
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return fallback_drive_route(origin, destination, drive_model, reason=f"osrm_error:{type(exc).__name__}")

    routes = data.get("routes") or []
    if not routes:
        return fallback_drive_route(origin, destination, drive_model, reason="osrm_no_route")
    route = routes[0]
    return {
        "provider": "osrm",
        "route_validated": True,
        "validation_reason": "osrm_route_found",
        "distance_miles": round(float(route.get("distance") or 0) / METERS_PER_MILE, 2),
        "duration_minutes": ceil_minutes(float(route.get("duration") or 0) / 60),
        "geometry": route.get("geometry"),
    }


def build_route_provider(
    mode: str,
    drive_model: dict[str, Any],
    osrm_base_url: str,
) -> RouteProvider:
    cache: dict[tuple[str, str, float, float, float, float], dict[str, Any]] = {}

    def cached(route_func: RouteProvider, origin: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
        key = (
            mode,
            str(origin.get("name") or ""),
            round(float(origin["lat"]), 6),
            round(float(origin["lon"]), 6),
            round(float(destination["lat"]), 6),
            round(float(destination["lon"]), 6),
        )
        if key not in cache:
            cache[key] = route_func(origin, destination)
        return cache[key]

    if mode == "model":
        model_provider = lambda origin, destination: fallback_drive_route(
            origin, destination, drive_model, reason="drive_model_requested"
        )
        return lambda origin, destination: cached(model_provider, origin, destination)

    osrm_provider = lambda origin, destination: osrm_drive_route(
        origin, destination, drive_model, base_url=osrm_base_url
    )
    return lambda origin, destination: cached(osrm_provider, origin, destination)


def enrich_trailhead(
    trailhead: dict[str, Any],
    parking_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(trailhead)
    source_record = parking_index.get(normalize_name(str(trailhead.get("name") or "")))
    if source_record:
        enriched = {**source_record, **enriched}
        enriched["source"] = source_record.get("source")
        enriched["has_parking"] = source_record.get("has_parking")
        enriched["facility_status"] = source_record.get("facility_status")
        enriched["parking_confidence"] = source_record.get("parking_confidence")
    return enriched


def parking_is_validated(trailhead: dict[str, Any]) -> bool:
    status = str(trailhead.get("facility_status") or "Open").lower()
    return bool(trailhead.get("has_parking") is True and "open" in status)


def simulate_candidate_outing(
    candidate: dict[str, Any],
    drive_model: dict[str, Any],
    route_provider: RouteProvider,
    parking_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parking_index = parking_index or {}
    origin = origin_from_drive_model(drive_model)
    trailhead = enrich_trailhead(candidate["trailhead"], parking_index)
    drive_to = route_provider(origin, trailhead)
    drive_home = route_provider(trailhead, origin)
    validation = candidate.get("validation") or {}
    return_to_car = candidate.get("return_to_car") or {}
    trailhead_snap = validation.get("trailhead_snap") or {}
    trailhead_access = candidate.get("trailhead_access") or {}

    checks = {
        "drive_to_trailhead_validated": bool(drive_to.get("route_validated")),
        "parking_validated": parking_is_validated(trailhead),
        "access_to_trail_validated": (
            validation.get("trailhead_snap_confidence") in {"high", "medium"}
            and bool(trailhead_snap.get("graph_validated"))
        ),
        "official_segment_traversal_validated": bool(
            validation.get("segment_coverage_passed")
            and validation.get("ascent_direction_passed")
        ),
        "return_to_car_validated": bool(
            return_to_car.get("graph_validated")
            and not return_to_car.get("needs_map_validation")
        ),
        "drive_home_validated": bool(drive_home.get("route_validated")),
        "day_of_special_management_checked": bool(validation.get("special_management_checked")),
    }

    blocking_reasons = []
    if not checks["drive_to_trailhead_validated"] or not checks["drive_home_validated"]:
        blocking_reasons.append("drive_route_not_validated")
    if not checks["parking_validated"]:
        blocking_reasons.append("parking_not_source_validated")
    if not checks["access_to_trail_validated"]:
        blocking_reasons.append("trailhead_access_not_graph_validated")
    if not checks["official_segment_traversal_validated"]:
        blocking_reasons.append("official_segment_or_ascent_validation_failed")
    if not checks["return_to_car_validated"]:
        blocking_reasons.append("return_to_car_not_graph_validated")

    if any(reason in blocking_reasons for reason in ["drive_route_not_validated", "parking_not_source_validated"]):
        execution_status = "blocked_by_logistics"
    elif blocking_reasons:
        execution_status = "blocked_by_route_validation"
    else:
        execution_status = "simulated_ready"

    time_breakdown = candidate.get("time_breakdown_minutes", {})
    parking_minutes = int(time_breakdown.get("parking_and_prep") or 0)
    moving_minutes = int(time_breakdown.get("moving_time") or 0)
    simulated_total_minutes = (
        int(drive_to.get("duration_minutes") or 0)
        + parking_minutes
        + moving_minutes
        + int(drive_home.get("duration_minutes") or 0)
    )
    official_new_miles = float(candidate.get("official_new_miles") or 0)
    effort = candidate.get("effort") or {}
    simulated_efficiency = (
        official_new_miles / simulated_total_minutes if simulated_total_minutes else 0
    )

    legs = [
        {
            "leg_type": "drive_to_trailhead",
            "from": origin["name"],
            "to": trailhead["name"],
            **drive_to,
        },
        {
            "leg_type": "park",
            "trailhead": trailhead["name"],
            "lat": trailhead.get("lat"),
            "lon": trailhead.get("lon"),
            "can_park": checks["parking_validated"],
            "parking_confidence": trailhead.get("parking_confidence"),
            "facility_status": trailhead.get("facility_status"),
            "address": trailhead.get("address"),
            "source": trailhead.get("source"),
            "parking_minutes": parking_minutes,
        },
        {
            "leg_type": "access_to_trail",
            "trailhead": trailhead["name"],
            "snap_confidence": validation.get("trailhead_snap_confidence"),
            "direct_gap_miles": trailhead_snap.get("direct_gap_miles"),
            "mapped_access_miles": trailhead_snap.get("mapped_access_miles"),
            "one_way_miles": trailhead_access.get("one_way_miles"),
            "round_trip_miles": trailhead_access.get("round_trip_miles"),
            "round_trip_connector_miles": trailhead_access.get("round_trip_connector_miles"),
            "round_trip_official_repeat_miles": trailhead_access.get("round_trip_official_repeat_miles"),
            "access_minutes_charged_in_moving_time": time_breakdown.get("trailhead_access"),
            "source": trailhead_access.get("source"),
            "connector_names": trailhead_access.get("connector_names") or [],
            "validated": checks["access_to_trail_validated"],
        },
        {
            "leg_type": "run_official_route",
            "trail_names": candidate.get("trail_names"),
            "segment_count": len(candidate.get("segment_ids") or []),
            "official_new_miles": candidate.get("official_new_miles"),
            "official_repeat_miles": candidate.get("official_repeat_miles"),
            "connector_miles": candidate.get("connector_miles"),
            "road_miles": candidate.get("road_miles"),
            "estimated_total_on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
            "ascent_ft": effort.get("ascent_ft"),
            "descent_ft": effort.get("descent_ft"),
            "grade_adjusted_miles": effort.get("grade_adjusted_miles"),
            "effort_score": effort.get("effort_score"),
            "elevation_source": effort.get("elevation_source"),
            "moving_minutes": moving_minutes,
            "validated": checks["official_segment_traversal_validated"],
        },
        {
            "leg_type": "return_to_car",
            "trailhead": trailhead["name"],
            "strategy": return_to_car.get("strategy"),
            "official_repeat_miles": return_to_car.get("official_repeat_miles"),
            "connector_miles": return_to_car.get("connector_miles"),
            "endpoint_gap_miles": return_to_car.get("endpoint_gap_miles"),
            "connector_names": return_to_car.get("connector_names") or [],
            "official_repeat_segment_ids": return_to_car.get("official_repeat_segment_ids") or [],
            "validated": checks["return_to_car_validated"],
            "needs_map_validation": return_to_car.get("needs_map_validation"),
        },
        {
            "leg_type": "drive_home",
            "from": trailhead["name"],
            "to": origin["name"],
            **drive_home,
        },
    ]

    return {
        "candidate_id": candidate.get("candidate_id"),
        "trail_names": candidate.get("trail_names"),
        "segment_ids": candidate.get("segment_ids") or [],
        "candidate_route_status": candidate.get("route_status"),
        "execution_status": execution_status,
        "field_ready": execution_status == "simulated_ready"
        and checks["day_of_special_management_checked"],
        "blocking_reasons": blocking_reasons,
        "day_of_checks_remaining": []
        if checks["day_of_special_management_checked"]
        else ["current_r2r_conditions_and_special_management"],
        "validation": checks,
        "planner_total_minutes": candidate.get("total_minutes"),
        "simulated_total_minutes": simulated_total_minutes,
        "simulated_efficiency_score": round(simulated_efficiency, 4),
        "legs": legs,
    }


def build_execution_ready_menu(outings: list[dict[str, Any]]) -> dict[str, Any]:
    def official_new_miles(outing: dict[str, Any]) -> float:
        if outing.get("legs") and len(outing["legs"]) > 3:
            return float(outing["legs"][3].get("official_new_miles") or 0)
        return float(outing.get("official_new_miles") or 0)

    ready_by_bucket: dict[str, list[dict[str, Any]]] = {}
    blocked_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for outing in outings:
        bucket = outing.get("time_bucket") or "unbucketed"
        if outing.get("execution_status") == "simulated_ready":
            ready_by_bucket.setdefault(bucket, []).append(outing)
        else:
            blocked_by_bucket.setdefault(bucket, []).append(outing)

    for bucket_outings in ready_by_bucket.values():
        bucket_outings.sort(
            key=lambda item: (
                -float(item.get("simulated_efficiency_score") or 0),
                -official_new_miles(item),
                int(item.get("simulated_total_minutes") or 99999),
                item.get("trail_names") or [],
            )
        )
    for bucket_outings in blocked_by_bucket.values():
        bucket_outings.sort(
            key=lambda item: (
                item.get("blocking_reasons") or [],
                item.get("trail_names") or [],
            )
        )

    return {
        "recommended_by_bucket": {
            bucket: bucket_outings[0]
            for bucket, bucket_outings in sorted(ready_by_bucket.items())
            if bucket_outings
        },
        "ready_by_bucket": dict(sorted(ready_by_bucket.items())),
        "blocked_by_bucket": dict(sorted(blocked_by_bucket.items())),
        "ready_counts_by_bucket": {
            bucket: len(bucket_outings)
            for bucket, bucket_outings in sorted(ready_by_bucket.items())
        },
        "blocked_counts_by_bucket": {
            bucket: len(bucket_outings)
            for bucket, bucket_outings in sorted(blocked_by_bucket.items())
        },
    }


def outing_trailhead_point(outing: dict[str, Any]) -> dict[str, Any]:
    park = outing["legs"][1]
    return {
        "name": park["trailhead"],
        "lat": float(park["lat"]),
        "lon": float(park["lon"]),
    }


def combo_has_unique_segments(outings: tuple[dict[str, Any], ...]) -> bool:
    seen: set[int] = set()
    for outing in outings:
        segment_ids = {int(seg_id) for seg_id in outing.get("segment_ids") or []}
        if seen & segment_ids:
            return False
        seen |= segment_ids
    return True


def build_combined_ready_menu(
    outings: list[dict[str, Any]],
    drive_model: dict[str, Any],
    route_provider: RouteProvider,
    max_outings_per_combo: int = 3,
    max_source_outings: int = 12,
) -> dict[str, Any]:
    ready = [
        outing
        for outing in outings
        if outing.get("execution_status") == "simulated_ready"
        and outing.get("segment_ids")
    ]
    ready.sort(
        key=lambda item: (
            -float(item.get("simulated_efficiency_score") or 0),
            -float(item["legs"][3].get("official_new_miles") or 0),
            int(item.get("simulated_total_minutes") or 99999),
            item.get("trail_names") or [],
        )
    )
    ready = ready[:max_source_outings]
    combos_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for combo_size in range(2, max_outings_per_combo + 1):
        for combo in combinations(ready, combo_size):
            if not combo_has_unique_segments(combo):
                continue
            interdrive_minutes = 0
            interdrive_miles = 0.0
            interdrive_legs = []
            for left, right in zip(combo, combo[1:]):
                left_point = outing_trailhead_point(left)
                right_point = outing_trailhead_point(right)
                if left_point["name"] == right_point["name"]:
                    route = {
                        "provider": "same_trailhead",
                        "route_validated": True,
                        "distance_miles": 0.0,
                        "duration_minutes": 0,
                    }
                else:
                    route = route_provider(left_point, right_point)
                interdrive_minutes += int(route.get("duration_minutes") or 0)
                interdrive_miles += float(route.get("distance_miles") or 0)
                interdrive_legs.append(
                    {
                        "from": left_point["name"],
                        "to": right_point["name"],
                        "provider": route.get("provider"),
                        "route_validated": bool(route.get("route_validated")),
                        "distance_miles": route.get("distance_miles"),
                        "duration_minutes": route.get("duration_minutes"),
                    }
                )
            if any(not leg["route_validated"] for leg in interdrive_legs):
                continue

            first_drive = int(combo[0]["legs"][0].get("duration_minutes") or 0)
            last_drive = int(combo[-1]["legs"][-1].get("duration_minutes") or 0)
            parking_minutes = sum(int(outing["legs"][1].get("parking_minutes") or 0) for outing in combo)
            moving_minutes = sum(int(outing["legs"][3].get("moving_minutes") or 0) for outing in combo)
            simulated_total = (
                first_drive + parking_minutes + moving_minutes + interdrive_minutes + last_drive
            )
            official_miles = sum(float(outing["legs"][3].get("official_new_miles") or 0) for outing in combo)
            total_foot = sum(float(outing["legs"][3].get("estimated_total_on_foot_miles") or 0) for outing in combo)
            bucket = classify_bucket(simulated_total)
            combo_record = {
                "combo_id": "+".join(str(outing.get("candidate_id")) for outing in combo),
                "outing_ids": [outing.get("candidate_id") for outing in combo],
                "outing_count": len(combo),
                "time_bucket": bucket,
                "trail_names": [trail for outing in combo for trail in outing.get("trail_names", [])],
                "trailheads": [outing["legs"][1]["trailhead"] for outing in combo],
                "official_new_miles": round(official_miles, 2),
                "estimated_total_on_foot_miles": round(total_foot, 2),
                "simulated_total_minutes": simulated_total,
                "simulated_efficiency_score": round(official_miles / simulated_total, 4)
                if simulated_total
                else 0,
                "first_drive_minutes": first_drive,
                "interdrive_minutes": interdrive_minutes,
                "interdrive_miles": round(interdrive_miles, 2),
                "return_home_minutes": last_drive,
                "interdrive_legs": interdrive_legs,
            }
            combos_by_bucket.setdefault(bucket, []).append(combo_record)

    for bucket_combos in combos_by_bucket.values():
        bucket_combos.sort(
            key=lambda item: (
                -item["simulated_efficiency_score"],
                -item["official_new_miles"],
                item["simulated_total_minutes"],
                item["trail_names"],
            )
        )

    return {
        "recommended_by_bucket": {
            bucket: combos[0]
            for bucket, combos in sorted(combos_by_bucket.items())
            if combos
        },
        "combos_by_bucket": dict(sorted(combos_by_bucket.items())),
        "combo_counts_by_bucket": {
            bucket: len(combos)
            for bucket, combos in sorted(combos_by_bucket.items())
        },
    }


def build_same_car_ready_menu(
    outings: list[dict[str, Any]],
    max_outings_per_combo: int = 4,
) -> dict[str, Any]:
    ready_by_trailhead: dict[str, list[dict[str, Any]]] = {}
    for outing in outings:
        if outing.get("execution_status") != "simulated_ready" or not outing.get("segment_ids"):
            continue
        trailhead = outing["legs"][1]["trailhead"]
        ready_by_trailhead.setdefault(trailhead, []).append(outing)

    combos_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for trailhead, trailhead_outings in ready_by_trailhead.items():
        if len(trailhead_outings) < 2:
            continue
        trailhead_outings.sort(
            key=lambda item: (
                -float(item.get("simulated_efficiency_score") or 0),
                -float(item["legs"][3].get("official_new_miles") or 0),
                int(item.get("simulated_total_minutes") or 99999),
                item.get("trail_names") or [],
            )
        )
        for combo_size in range(2, max_outings_per_combo + 1):
            for combo in combinations(trailhead_outings, combo_size):
                if not combo_has_unique_segments(combo):
                    continue
                first_drive = int(combo[0]["legs"][0].get("duration_minutes") or 0)
                parking_minutes = int(combo[0]["legs"][1].get("parking_minutes") or 0)
                return_home = int(combo[0]["legs"][-1].get("duration_minutes") or 0)
                moving_minutes = sum(int(outing["legs"][3].get("moving_minutes") or 0) for outing in combo)
                official_miles = sum(float(outing["legs"][3].get("official_new_miles") or 0) for outing in combo)
                total_foot = sum(float(outing["legs"][3].get("estimated_total_on_foot_miles") or 0) for outing in combo)
                simulated_total = first_drive + parking_minutes + moving_minutes + return_home
                bucket = classify_bucket(simulated_total)
                combo_record = {
                    "recommendation_type": "same_parked_car",
                    "combo_id": "+".join(str(outing.get("candidate_id")) for outing in combo),
                    "outing_ids": [outing.get("candidate_id") for outing in combo],
                    "outing_count": len(combo),
                    "time_bucket": bucket,
                    "trail_names": [trail for outing in combo for trail in outing.get("trail_names", [])],
                    "trailheads": [trailhead],
                    "official_new_miles": round(official_miles, 2),
                    "estimated_total_on_foot_miles": round(total_foot, 2),
                    "simulated_total_minutes": simulated_total,
                    "simulated_efficiency_score": round(official_miles / simulated_total, 4)
                    if simulated_total
                    else 0,
                    "first_drive_minutes": first_drive,
                    "parking_minutes": parking_minutes,
                    "moving_minutes": moving_minutes,
                    "interdrive_minutes": 0,
                    "interdrive_miles": 0.0,
                    "return_home_minutes": return_home,
                }
                combos_by_bucket.setdefault(bucket, []).append(combo_record)

    for bucket_combos in combos_by_bucket.values():
        bucket_combos.sort(
            key=lambda item: (
                -item["simulated_efficiency_score"],
                -item["official_new_miles"],
                item["simulated_total_minutes"],
                item["trail_names"],
            )
        )

    return {
        "recommended_by_bucket": {
            bucket: combos[0]
            for bucket, combos in sorted(combos_by_bucket.items())
            if combos
        },
        "combos_by_bucket": dict(sorted(combos_by_bucket.items())),
        "combo_counts_by_bucket": {
            bucket: len(combos)
            for bucket, combos in sorted(combos_by_bucket.items())
        },
    }


def build_best_executable_menu(
    execution_menu: dict[str, Any],
    combined_menu: dict[str, Any],
    same_car_menu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options_by_bucket: dict[str, list[dict[str, Any]]] = {}
    same_car_menu = same_car_menu or {}

    for bucket, outing in (execution_menu.get("recommended_by_bucket") or {}).items():
        run = outing["legs"][3]
        drive_to = outing["legs"][0]
        drive_home = outing["legs"][-1]
        park = outing["legs"][1]
        options_by_bucket.setdefault(bucket, []).append(
            {
                "recommendation_type": "single_outing",
                "id": outing.get("candidate_id"),
                "bucket": bucket,
                "trail_names": outing.get("trail_names") or [],
                "trailheads": [park.get("trailhead")],
                "official_new_miles": float(run.get("official_new_miles") or 0),
                "estimated_total_on_foot_miles": float(run.get("estimated_total_on_foot_miles") or 0),
                "simulated_total_minutes": int(outing.get("simulated_total_minutes") or 0),
                "simulated_efficiency_score": float(outing.get("simulated_efficiency_score") or 0),
                "drive_minutes": int(drive_to.get("duration_minutes") or 0),
                "return_home_minutes": int(drive_home.get("duration_minutes") or 0),
                "interdrive_minutes": 0,
                "source": outing,
            }
        )

    for bucket, combo in (combined_menu.get("recommended_by_bucket") or {}).items():
        options_by_bucket.setdefault(bucket, []).append(
            {
                "recommendation_type": "combined_multi_stop",
                "id": combo.get("combo_id"),
                "bucket": bucket,
                "trail_names": combo.get("trail_names") or [],
                "trailheads": combo.get("trailheads") or [],
                "official_new_miles": float(combo.get("official_new_miles") or 0),
                "estimated_total_on_foot_miles": float(combo.get("estimated_total_on_foot_miles") or 0),
                "simulated_total_minutes": int(combo.get("simulated_total_minutes") or 0),
                "simulated_efficiency_score": float(combo.get("simulated_efficiency_score") or 0),
                "drive_minutes": int(combo.get("first_drive_minutes") or 0),
                "return_home_minutes": int(combo.get("return_home_minutes") or 0),
                "interdrive_minutes": int(combo.get("interdrive_minutes") or 0),
                "source": combo,
            }
        )

    for bucket, combo in (same_car_menu.get("recommended_by_bucket") or {}).items():
        options_by_bucket.setdefault(bucket, []).append(
            {
                "recommendation_type": "same_parked_car",
                "id": combo.get("combo_id"),
                "bucket": bucket,
                "trail_names": combo.get("trail_names") or [],
                "trailheads": combo.get("trailheads") or [],
                "official_new_miles": float(combo.get("official_new_miles") or 0),
                "estimated_total_on_foot_miles": float(combo.get("estimated_total_on_foot_miles") or 0),
                "simulated_total_minutes": int(combo.get("simulated_total_minutes") or 0),
                "simulated_efficiency_score": float(combo.get("simulated_efficiency_score") or 0),
                "drive_minutes": int(combo.get("first_drive_minutes") or 0),
                "return_home_minutes": int(combo.get("return_home_minutes") or 0),
                "interdrive_minutes": 0,
                "source": combo,
            }
        )

    complexity_rank = {
        "single_outing": 0,
        "same_parked_car": 1,
        "combined_multi_stop": 2,
    }
    for options in options_by_bucket.values():
        options.sort(
            key=lambda item: (
                -item["simulated_efficiency_score"],
                -item["official_new_miles"],
                item["simulated_total_minutes"],
                complexity_rank.get(item["recommendation_type"], 9),
                item["trail_names"],
            )
        )

    recommended = {
        bucket: options[0]
        for bucket, options in sorted(
            options_by_bucket.items(),
            key=lambda item: BUCKET_ORDER.index(item[0]) if item[0] in BUCKET_ORDER else len(BUCKET_ORDER),
        )
        if options
    }
    return {
        "recommended_by_bucket": recommended,
        "options_by_bucket": dict(
            sorted(
                options_by_bucket.items(),
                key=lambda item: BUCKET_ORDER.index(item[0]) if item[0] in BUCKET_ORDER else len(BUCKET_ORDER),
            )
        ),
        "missing_buckets": [bucket for bucket in BUCKET_ORDER if bucket not in recommended],
    }


def build_single_car_menu(
    execution_menu: dict[str, Any],
    same_car_menu: dict[str, Any],
) -> dict[str, Any]:
    return build_best_executable_menu(
        execution_menu=execution_menu,
        same_car_menu=same_car_menu,
        combined_menu={"recommended_by_bucket": {}},
    )


def selected_candidates(plan: dict[str, Any], candidate_set: str) -> list[dict[str, Any]]:
    if candidate_set == "primary":
        candidates_by_bucket = plan["route_menu"]["primary_candidates_by_bucket"]
        return [
            {**candidate, "time_bucket": bucket}
            for bucket, candidate in candidates_by_bucket.items()
        ]
    if candidate_set == "validated-primary":
        candidates_by_bucket = plan["route_menu"].get("primary_validated_candidates_by_bucket") or {}
        return [
            {**candidate, "time_bucket": bucket}
            for bucket, candidate in candidates_by_bucket.items()
        ]
    if candidate_set == "graph-validated":
        return [
            candidate
            for candidate in plan["route_menu"]["all_candidates"]
            if candidate.get("route_status") == "graph_validated"
        ]
    return plan["route_menu"]["all_candidates"]


def render_markdown(simulation: dict[str, Any]) -> str:
    execution_menu = simulation.get("execution_ready_menu") or {}
    best_menu = simulation.get("best_executable_menu") or {}
    single_car_menu = simulation.get("single_car_menu") or {}
    lines = [
        "# Outing Execution Simulation",
        "",
        f"Generated: {simulation['generated_at']}",
        "",
        "This simulates the full loop: drive to the trailhead, park, access the official trail, run the candidate, return to the parked car, and drive home.",
        "",
        "## Summary",
        "",
        f"- Candidate set: {simulation['candidate_set']}",
        f"- Drive routing mode: {simulation['drive_routing_mode']}",
        f"- Outings simulated: {len(simulation['outings'])}",
        f"- Simulated ready: {sum(1 for outing in simulation['outings'] if outing['execution_status'] == 'simulated_ready')}",
        f"- Blocked: {sum(1 for outing in simulation['outings'] if outing['execution_status'] != 'simulated_ready')}",
        "",
        "## Best Executable Recommendations",
        "",
        "| Bucket | Type | Route | Sim total | Official mi | Total foot mi | Trailheads | Drive / interdrive / home | Efficiency |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for bucket, recommendation in (best_menu.get("recommended_by_bucket") or {}).items():
        lines.append(
            f"| {bucket} "
            f"| {recommendation['recommendation_type']} "
            f"| {', '.join(recommendation['trail_names'])} "
            f"| {recommendation['simulated_total_minutes']} "
            f"| {recommendation['official_new_miles']:.2f} "
            f"| {recommendation['estimated_total_on_foot_miles']:.2f} "
            f"| {' -> '.join(recommendation['trailheads'])} "
            f"| {recommendation['drive_minutes']} / {recommendation['interdrive_minutes']} / {recommendation['return_home_minutes']} "
            f"| {recommendation['simulated_efficiency_score']} |"
        )
    if best_menu.get("missing_buckets"):
        lines.extend(
            [
                "",
                "No executable recommendation currently exists for: "
                + ", ".join(best_menu["missing_buckets"])
                + ".",
            ]
        )

    if single_car_menu.get("recommended_by_bucket"):
        lines.extend(
            [
                "",
                "## Best Single-Car Recommendations",
                "",
                "| Bucket | Type | Route | Sim total | Official mi | Total foot mi | Trailhead | Efficiency |",
                "|---|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for bucket, recommendation in single_car_menu["recommended_by_bucket"].items():
            lines.append(
                f"| {bucket} "
                f"| {recommendation['recommendation_type']} "
                f"| {', '.join(recommendation['trail_names'])} "
                f"| {recommendation['simulated_total_minutes']} "
                f"| {recommendation['official_new_miles']:.2f} "
                f"| {recommendation['estimated_total_on_foot_miles']:.2f} "
                f"| {' -> '.join(recommendation['trailheads'])} "
                f"| {recommendation['simulated_efficiency_score']} |"
            )
        if single_car_menu.get("missing_buckets"):
            lines.extend(
                [
                    "",
                    "No single-car recommendation currently exists for: "
                    + ", ".join(single_car_menu["missing_buckets"])
                    + ".",
                ]
            )

    lines.extend(
        [
            "",
            "## Single Ready Route Winners",
            "",
            "| Bucket | Route | Sim total | Official mi | Total foot mi | Trailhead | Drive each way | Efficiency |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for bucket, outing in (execution_menu.get("recommended_by_bucket") or {}).items():
        drive_to = outing["legs"][0]
        run = outing["legs"][3]
        park = outing["legs"][1]
        lines.append(
            f"| {bucket} "
            f"| {', '.join(outing['trail_names'])} "
            f"| {outing['simulated_total_minutes']} "
            f"| {run['official_new_miles']} "
            f"| {run['estimated_total_on_foot_miles']} "
            f"| {park['trailhead']} "
            f"| {drive_to['duration_minutes']} "
            f"| {outing['simulated_efficiency_score']} |"
        )
    missing_buckets = [
        bucket
        for bucket in ["under_1_hour", "one_to_two_hours", "two_to_three_hours", "three_to_four_hours", "four_plus_hours"]
        if bucket not in (execution_menu.get("recommended_by_bucket") or {})
    ]
    if missing_buckets:
        lines.extend(
            [
                "",
                "No simulated-ready route currently exists for: "
                + ", ".join(missing_buckets)
                + ".",
            ]
        )
    combined_menu = simulation.get("combined_ready_menu") or {}
    same_car_menu = simulation.get("same_car_ready_menu") or {}
    if same_car_menu.get("recommended_by_bucket"):
        lines.extend(
            [
                "",
                "## Same Parked-Car Routes",
                "",
                "| Bucket | Route | Sim total | Official mi | Total foot mi | Trailhead | Efficiency |",
                "|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for bucket, combo in same_car_menu["recommended_by_bucket"].items():
            lines.append(
                f"| {bucket} "
                f"| {', '.join(combo['trail_names'])} "
                f"| {combo['simulated_total_minutes']} "
                f"| {combo['official_new_miles']} "
                f"| {combo['estimated_total_on_foot_miles']} "
                f"| {' -> '.join(combo['trailheads'])} "
                f"| {combo['simulated_efficiency_score']} |"
            )
    if combined_menu.get("recommended_by_bucket"):
        lines.extend(
            [
                "",
                "## Combined Ready Routes",
                "",
                "| Bucket | Route | Sim total | Official mi | Total foot mi | Trailheads | Interdrive | Efficiency |",
                "|---|---|---:|---:|---:|---|---:|---:|",
            ]
        )
        for bucket, combo in combined_menu["recommended_by_bucket"].items():
            lines.append(
                f"| {bucket} "
                f"| {', '.join(combo['trail_names'])} "
                f"| {combo['simulated_total_minutes']} "
                f"| {combo['official_new_miles']} "
                f"| {combo['estimated_total_on_foot_miles']} "
                f"| {' -> '.join(combo['trailheads'])} "
                f"| {combo['interdrive_minutes']} "
                f"| {combo['simulated_efficiency_score']} |"
            )
    lines.extend(
        [
            "",
        "## Outings",
        "",
            "| Bucket | Route | Status | Sim total | Drive | Parking | Access | Return | Home | Blocking reasons |",
            "|---|---|---|---:|---|---|---|---|---|---|",
        ]
    )
    for outing in simulation["outings"]:
        drive_to = outing["legs"][0]
        park = outing["legs"][1]
        access = outing["legs"][2]
        ret = outing["legs"][4]
        home = outing["legs"][5]
        lines.append(
            f"| {outing.get('time_bucket', '')} "
            f"| {', '.join(outing['trail_names'])} "
            f"| {outing['execution_status']} "
            f"| {outing['simulated_total_minutes']} "
            f"| {drive_to['duration_minutes']} min / {drive_to['distance_miles']} mi / {drive_to['provider']} "
            f"| {'yes' if park['can_park'] else 'no'} "
            f"| {'yes' if access['validated'] else 'no'} "
            f"| {'yes' if ret['validated'] else 'no'} "
            f"| {'yes' if home['route_validated'] else 'no'} "
            f"| {', '.join(outing['blocking_reasons']) or 'none'} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.append("- `simulated_ready` means drive, parking, trail access, official traversal, return-to-car, and drive-home checks passed with the available data.")
    lines.append("- `field_ready` remains false until current Ridge to Rivers conditions, closures, and special-management rules are checked for the actual date.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument(
        "--candidate-set",
        choices=["primary", "validated-primary", "graph-validated", "all"],
        default="graph-validated",
    )
    parser.add_argument("--drive-routing", choices=["osrm", "model"], default="osrm")
    parser.add_argument("--osrm-base-url", default="https://router.project-osrm.org")
    parser.add_argument("--max-combo-source-outings", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "outing_execution.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_DIR / "outing_execution.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.plan_json)
    state = load_state(args.state)
    drive_model = get_drive_model(state)
    parking_index = load_parking_index(args.trailheads_geojson)
    route_provider = build_route_provider(args.drive_routing, drive_model, args.osrm_base_url)
    outings = [
        {
            **simulate_candidate_outing(
                candidate,
                drive_model,
                route_provider=route_provider,
                parking_index=parking_index,
            ),
            "time_bucket": candidate.get("time_bucket"),
        }
        for candidate in selected_candidates(plan, args.candidate_set)
    ]
    run_id = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    simulation = {
        "run_id": run_id,
        "generated_at": run_id,
        "candidate_set": args.candidate_set,
        "drive_routing_mode": args.drive_routing,
        "source_plan": str(args.plan_json),
        "source_plan_run_id": plan.get("run_id") or plan.get("generated_at"),
        "source_trailheads": str(args.trailheads_geojson),
        "outings": outings,
    }
    simulation["execution_ready_menu"] = build_execution_ready_menu(outings)
    simulation["same_car_ready_menu"] = build_same_car_ready_menu(outings)
    simulation["single_car_menu"] = build_single_car_menu(
        simulation["execution_ready_menu"],
        simulation["same_car_ready_menu"],
    )
    simulation["combined_ready_menu"] = build_combined_ready_menu(
            outings,
            drive_model,
            route_provider=route_provider,
            max_source_outings=args.max_combo_source_outings,
    )
    simulation["best_executable_menu"] = build_best_executable_menu(
        simulation["execution_ready_menu"],
        simulation["combined_ready_menu"],
        same_car_menu=simulation["same_car_ready_menu"],
    )
    write_json(args.output_json, simulation)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(simulation))
    manifest_path = args.output_json.parent / f"{args.output_json.stem}-artifact-manifest.json"
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=run_id,
            inputs=[args.plan_json, args.state, args.trailheads_geojson],
            outputs=[args.output_json, args.output_md],
            command="simulate_outing_execution.py",
            metadata={
                "candidate_set": args.candidate_set,
                "drive_routing_mode": args.drive_routing,
                "source_plan_run_id": simulation["source_plan_run_id"],
            },
        ),
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {manifest_path}")
    print(
        "Outings: "
        f"{sum(1 for outing in outings if outing['execution_status'] == 'simulated_ready')} ready, "
        f"{sum(1 for outing in outings if outing['execution_status'] != 'simulated_ready')} blocked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
