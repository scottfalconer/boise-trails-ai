#!/usr/bin/env python3
"""Package selected route candidates into human route-block review units."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from block_route_candidate_pass import (  # noqa: E402
    PALETTE,
    line_feature,
    multiline_feature,
    parking_feature,
    point_feature,
    split_coords_on_gaps,
)
from export_execution_gpx import (  # noqa: E402
    best_connector_link_for_oriented_segment,
    candidate_segment_coordinates,
    candidate_segments_for_track,
    candidate_track_coordinates,
    load_candidate_index,
    load_official_segment_index,
    safe_between_trail_links_for_candidate,
    special_management_segment_direction_overrides,
    validate_track_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    load_connector_graph,
    load_official_segments,
    read_json,
    round_miles,
    shortest_connector_path,
)


DEFAULT_BLOCKS_JSON = YEAR_DIR / "inputs" / "personal" / "2026-route-blocks-v1.json"
DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-route-candidate-pass-v1.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_BASENAME = "block-day-package-pass-v1"
DEFAULT_R2R_TRAILS_GEOJSON = (
    YEAR_DIR
    / "inputs"
    / "open-data"
    / "r2r-trails-2026-05-04"
    / "boise_parks_trails_open_data.geojson"
)
CAR_PASS_RADIUS_MILES = 0.08
CAR_PASS_ENDPOINT_BUFFER_MILES = 0.5
CAR_PASS_MIN_SEPARATION_MILES = 0.35
SIGNPOST_TRAIL_NUMBERS = {
    "hippie shake": "50",
    "hippie shake trail": "50",
    "who now loop": "51",
    "who now loop trail": "51",
    "kemper's ridge": "52",
    "kemper's ridge trail": "52",
    "buena vista": "53",
    "buena vista trail": "53",
    "harrison hollow": "57",
    "harrison hollow trail": "57",
    "lower hulls gulch": "29",
    "lower hulls gulch trail": "29",
    "polecat loop": "81",
    "polecat loop trail": "81",
    "around the mountain": "98",
    "around the mountain trail": "98",
    "bucktail": "20A",
    "bucktail trail": "20A",
}

_ROUTE_CUE_OFFICIAL_INDEX: dict[int, dict[str, Any]] | None = None


def route_cue_official_index() -> dict[int, dict[str, Any]]:
    global _ROUTE_CUE_OFFICIAL_INDEX
    if _ROUTE_CUE_OFFICIAL_INDEX is None:
        _ROUTE_CUE_OFFICIAL_INDEX = load_official_segment_index(DEFAULT_OFFICIAL_GEOJSON)
    return _ROUTE_CUE_OFFICIAL_INDEX
TRAIL_NUMBER_RE = re.compile(r"#\s*([0-9]+[A-Z]?)\b", re.IGNORECASE)
GENERIC_ROUTE_NAME_KEYS = {
    "connector",
    "connection",
    "access",
    "access trail",
    "osm path connector",
    "osm service connector",
    "trail",
}
TRAILHEAD_AREA_PATTERNS = (
    (re.compile(r"\b(simplot|pioneer|freddy|bogus)\b", re.IGNORECASE), "Bogus Basin"),
    (re.compile(r"\b(miller\s*gulch|dry creek|sweet connie)\b", re.IGNORECASE), "Dry Creek"),
    (re.compile(r"\bavimor|spring valley creek\b", re.IGNORECASE), "Avimor / Harlow"),
    (re.compile(r"\bcervidae|arrow rock\b", re.IGNORECASE), "Cervidae / Arrow Rock"),
    (re.compile(r"\bfull sail|36th\b", re.IGNORECASE), "Full Sail / N 36th St"),
)
_R2R_NAME_INDEX_CACHE: dict[str, Any] | None = None


def normalized_trail_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("’", "'").strip())


def signpost_key(value: Any) -> str:
    text = normalized_trail_text(value).lower()
    text = TRAIL_NUMBER_RE.sub("", text)
    text = re.sub(r"\btrail\s*(\d+)?\b", "trail", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—,:;")
    return text


def signpost_label(value: Any) -> str:
    text = normalized_trail_text(value)
    if not text:
        return ""
    number_match = TRAIL_NUMBER_RE.search(text)
    number = number_match.group(1).upper() if number_match else SIGNPOST_TRAIL_NUMBERS.get(signpost_key(text))
    if not number:
        return ""
    name = TRAIL_NUMBER_RE.sub("", text).strip(" -–—,:;")
    if not name:
        return f"#{number}"
    return f"#{number} {name}"


def signpost_labels(values: list[Any]) -> list[str]:
    labels = []
    seen = set()
    for value in values:
        label = signpost_label(value)
        key = re.sub(r"\btrail\b", "", label.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def route_name_key(value: Any) -> str:
    text = normalized_trail_text(value).lower()
    text = TRAIL_NUMBER_RE.sub("", text)
    text = re.sub(r"\([^)]*\)", "", text)
    replacements = {
        "’": "'",
        "&": " and ",
        "kestral": "kestrel",
        "hull's": "hulls",
        "brewer's": "brewers",
        "mtn": "mountain",
        "mountain interpretive trail": "mountain interpretive",
        "street": "st",
        "st.": "st",
        "extension": "ext",
        "tribes trail": "tribes",
        "trail": "",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def route_name_variants(value: Any) -> list[str]:
    raw = normalized_trail_text(value)
    variants = [raw]
    variants.extend(part.strip() for part in re.split(r"\s*[-/]\s*", raw) if part.strip())
    if " - " in raw:
        variants.append(raw.split(" - ", 1)[0])
    keys = []
    for variant in variants:
        key = route_name_key(variant)
        if key and key not in keys:
            keys.append(key)
    return keys


def load_r2r_route_name_index(path: Path = DEFAULT_R2R_TRAILS_GEOJSON) -> dict[str, Any]:
    global _R2R_NAME_INDEX_CACHE
    if _R2R_NAME_INDEX_CACHE is not None:
        return _R2R_NAME_INDEX_CACHE
    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    if path.exists():
        for feature in read_json(path).get("features") or []:
            props = feature.get("properties") or {}
            trail_name = normalized_trail_text(props.get("TrailName"))
            if not trail_name:
                continue
            record = {
                "trail_name": trail_name,
                "trail_subsystem": normalized_trail_text(props.get("TrailSubSystem")),
                "system_name": normalized_trail_text(props.get("SystemName")),
                "source": str(path),
            }
            for key in route_name_variants(trail_name):
                if record not in by_key[key]:
                    by_key[key].append(record)
    _R2R_NAME_INDEX_CACHE = {"by_key": by_key, "source": str(path)}
    return _R2R_NAME_INDEX_CACHE


def preferred_r2r_record(records: list[dict[str, str]], key: str) -> dict[str, str] | None:
    if not records or key in GENERIC_ROUTE_NAME_KEYS:
        return None
    candidates = [record for record in records if route_name_key(record.get("trail_name")) == key] or records
    subsystems = {
        record.get("trail_subsystem")
        for record in candidates
        if record.get("trail_subsystem")
    }
    if len(subsystems) != 1:
        return None

    def score(record: dict[str, str]) -> tuple[bool, bool, int, str]:
        trail_name = record.get("trail_name") or ""
        lower_name = trail_name.lower()
        return (
            "(" in trail_name or ")" in trail_name,
            "dog on-leash" in lower_name or "stm" in lower_name,
            len(trail_name),
            lower_name,
        )

    return sorted(candidates, key=score)[0]


def r2r_record_for_trail(trail_name: Any, r2r_index: dict[str, Any] | None = None) -> dict[str, str] | None:
    index = r2r_index or load_r2r_route_name_index()
    by_key = index.get("by_key") or {}
    for key in route_name_variants(trail_name):
        records = by_key.get(key) or []
        if len(records) == 1:
            return records[0]
        if len(records) > 1:
            record = preferred_r2r_record(records, key)
            if record:
                return record
    return None


def major_system_name(record: dict[str, str] | None) -> str:
    if not record:
        return ""
    subsystem = normalized_trail_text(record.get("trail_subsystem"))
    system = normalized_trail_text(record.get("system_name"))
    if subsystem == "Bogus Basin Area":
        return "Bogus Basin"
    return subsystem or system


def clean_trailhead_area_name(trailhead: Any) -> str:
    text = normalized_trail_text(trailhead)
    if not text:
        return ""
    for pattern, area in TRAILHEAD_AREA_PATTERNS:
        if pattern.search(text):
            return area
    text = re.sub(r"\b(?:parking area|parking|trailhead|road-parking anchor|prior parking anchor|osm)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:private|strava|anchor)\b", "", text, flags=re.IGNORECASE)
    text = text.replace("MillerGulch", "Miller Gulch")
    text = re.sub(r"\s+", " ", text).strip(" ,-/")
    return text


def common_trail_name(trail_name: Any, r2r_index: dict[str, Any] | None = None) -> str:
    record = r2r_record_for_trail(trail_name, r2r_index)
    if record:
        return record["trail_name"]
    text = normalized_trail_text(trail_name)
    text = text.replace("Kestral", "Kestrel")
    return text


def human_route_name(
    trail_names: list[Any] | tuple[Any, ...] | None,
    trailhead: Any = "",
    *,
    r2r_index: dict[str, Any] | None = None,
) -> dict[str, str]:
    trails = unique_nonempty(list(trail_names or []))
    index = r2r_index or load_r2r_route_name_index()
    first_trail = trails[0] if trails else ""
    primary = common_trail_name(first_trail, index) if first_trail else ""
    primary_key = route_name_key(primary)

    first_record = r2r_record_for_trail(first_trail, index) if first_trail else None
    area = ""
    source = "fallback"
    if first_record and primary_key not in GENERIC_ROUTE_NAME_KEYS:
        area = major_system_name(first_record)
        source = "r2r_starting_trail_subsystem"
    if not area:
        for trail in trails:
            record = r2r_record_for_trail(trail, index)
            if record and route_name_key(record.get("trail_name")) not in GENERIC_ROUTE_NAME_KEYS:
                area = major_system_name(record)
                source = "r2r_first_known_trail_subsystem"
                break
    if not area:
        area = clean_trailhead_area_name(trailhead)
        source = "trailhead_fallback" if area else "trail_name_fallback"
    if not primary and trails:
        primary = common_trail_name(trails[0], index)
    if not primary and area:
        primary = "Route"

    if area and primary:
        if route_name_key(area) == route_name_key(primary):
            name = primary if source == "trailhead_fallback" else area
        else:
            name = f"{area}: {primary}"
    else:
        name = primary or area or "Route"
    return {
        "route_name": name,
        "route_name_source": source,
        "route_name_area": area,
        "route_name_primary_trail": primary,
    }


def apply_human_route_names_to_map_data(map_data: dict[str, Any]) -> dict[str, Any]:
    r2r_index = load_r2r_route_name_index()
    for package in map_data.get("packages") or []:
        package_route_names = []
        for component in package.get("components") or []:
            route_name = human_route_name(
                component.get("trail_names") or [],
                component.get("trailhead") or package.get("primary_trailhead") or "",
                r2r_index=r2r_index,
            )
            component.update(route_name)
            if route_name["route_name"] not in package_route_names:
                package_route_names.append(route_name["route_name"])
        if package_route_names:
            package["route_names"] = package_route_names
    map_data.setdefault("route_name_source", {})["r2r_open_data"] = r2r_index.get("source")
    return map_data


def trailhead_distance_miles(left: str | None, right: str | None, routes: list[dict[str, Any]]) -> float | None:
    if not left or not right or left == right:
        return 0.0
    points: dict[str, tuple[float, float]] = {}
    for route in routes:
        trailhead = route.get("trailhead")
        trailhead_data = route.get("_candidate", {}).get("trailhead") or {}
        if trailhead and trailhead_data.get("lon") is not None and trailhead_data.get("lat") is not None:
            points[str(trailhead)] = (float(trailhead_data["lon"]), float(trailhead_data["lat"]))
    if left not in points or right not in points:
        return None
    return haversine_miles(points[left], points[right])


def package_status(package: dict[str, Any], acceptance: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    preferred_ratio = float(acceptance.get("preferred_max_on_foot_to_official_ratio") or 1.6)
    if package["trailhead_count"] > int(acceptance.get("max_normal_trailheads_per_day") or 1):
        reasons.append("multiple_trailheads_need_route_design")
    if package["ratio"] and package["ratio"] > preferred_ratio:
        reasons.append("ratio_above_preferred_limit")
    if package["component_routes_under_1_official_mile"]:
        reasons.append("contains_absorbed_sub1_components")
    if package["component_routes_under_2_official_miles"]:
        reasons.append("contains_absorbed_sub2_components")
    if package["boundary_review"]:
        reasons.append("boundary_review_block")
    if not reasons:
        return "schedule_candidate_after_gpx", []
    if package["trailhead_count"] == 1 and not package["boundary_review"]:
        return "same_trailhead_package_after_gpx", reasons
    return "needs_manual_route_design", reasons


def build_packages(
    route_pass: dict[str, Any],
    blocks_config: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_index = candidate_index or {}
    blocks = {str(block["block_id"]): block for block in blocks_config.get("blocks") or []}
    routes_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in route_pass.get("routes") or []:
        candidate = candidate_index.get(str(route.get("candidate_id")))
        enriched = dict(route)
        if candidate:
            enriched["_candidate"] = candidate
        routes_by_block[str(route.get("block_id") or "unassigned")].append(enriched)

    packages = []
    block_order = [str(block["block_id"]) for block in blocks_config.get("blocks") or []]
    for block_id in [*block_order, *sorted(set(routes_by_block) - set(block_order))]:
        routes = sorted(routes_by_block.get(block_id) or [], key=lambda route: int(route.get("route_number") or 0))
        if not routes:
            continue
        block = blocks.get(block_id, {"block_id": block_id, "name": block_id})
        official = sum(float(route.get("official_miles") or 0.0) for route in routes)
        on_foot = sum(float(route.get("on_foot_miles") or 0.0) for route in routes)
        trailheads = sorted({str(route.get("trailhead")) for route in routes if route.get("trailhead")})
        primary_trailhead = trailheads[0] if len(trailheads) == 1 else None
        package = {
            "package_number": len(packages) + 1,
            "block_id": block_id,
            "block_name": block.get("name"),
            "boundary_review": block.get("status") == "boundary_review",
            "component_route_count": len(routes),
            "component_candidate_ids": [route.get("candidate_id") for route in routes],
            "trail_names": sorted({trail for route in routes for trail in route.get("trail_names") or []}),
            "official_miles": round_miles(official),
            "on_foot_miles": round_miles(on_foot),
            "ratio": round(on_foot / official, 2) if official else None,
            "trailheads": trailheads,
            "trailhead_count": len(trailheads),
            "primary_trailhead": primary_trailhead,
            "total_minutes_components": sum(int(route.get("total_minutes") or 0) for route in routes),
            "component_routes_under_1_official_mile": sum(
                1 for route in routes if float(route.get("official_miles") or 0.0) < 1.0
            ),
            "component_routes_under_2_official_miles": sum(
                1 for route in routes if float(route.get("official_miles") or 0.0) < 2.0
            ),
            "components": [
                {
                    key: route.get(key)
                    for key in [
                        "route_number",
                        "candidate_id",
                        "trail_names",
                        "official_miles",
                        "on_foot_miles",
                        "ratio",
                        "total_minutes",
                        "trailhead",
                        "less_optimal_flags",
                        "segment_ids",
                        "time_breakdown_minutes",
                    ]
                }
                for route in routes
            ],
        }
        status, reasons = package_status(package, blocks_config.get("acceptance_criteria") or {})
        package["planning_status"] = status
        package["planning_reasons"] = reasons
        packages.append(package)

    official = sum(float(package["official_miles"]) for package in packages)
    on_foot = sum(float(package["on_foot_miles"]) for package in packages)
    return {
        "planning_status": "block_day_package_pass",
        "summary": {
            "package_count": len(packages),
            "component_route_count": sum(package["component_route_count"] for package in packages),
            "covered_segment_count": route_pass.get("summary", {}).get("covered_segment_count"),
            "official_miles": round_miles(official),
            "total_on_foot_miles": round_miles(on_foot),
            "planwide_on_foot_to_official_ratio": round(on_foot / official, 2) if official else None,
            "packages_with_multiple_trailheads": sum(1 for package in packages if package["trailhead_count"] > 1),
            "component_routes_under_1_official_mile": sum(
                package["component_routes_under_1_official_mile"] for package in packages
            ),
            "component_routes_under_2_official_miles": sum(
                package["component_routes_under_2_official_miles"] for package in packages
            ),
            "same_trailhead_package_count": sum(1 for package in packages if package["trailhead_count"] == 1),
            "manual_route_design_package_count": sum(
                1 for package in packages if package["planning_status"] == "needs_manual_route_design"
            ),
        },
        "packages": packages,
        "caveats": [
            "This packages validated component routes into route-block review units so the plan is reviewed as trail systems, not as car-hop errands.",
            "A package is not automatically one calendar day. Packages with multiple parked starts can be run same-day with a re-park or split across days.",
            "It does not yet rebuild continuous custom GPX for every package; packages with multiple trailheads or high ratios still need manual route design.",
            "Sub-1 and sub-2-mile components are no longer standalone day decisions in this artifact; they are explicitly absorbed into block packages.",
        ],
    }


def unique_nonempty(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalized_segment_ids(values: list[Any] | tuple[Any, ...] | None) -> list[int]:
    result = []
    for value in values or []:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def coordinate_pair(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def coordinate_path(path_coordinates: Any) -> list[list[float]]:
    return [point for point in (coordinate_pair(value) for value in path_coordinates or []) if point]


def path_payload(path_coordinates: Any) -> dict[str, Any]:
    points = [point for point in (coordinate_pair(value) for value in path_coordinates or []) if point]
    if len(points) < 2:
        return {}
    return {"path_coordinates": points, "path_start": points[0], "path_end": points[-1]}


def segment_direction_cue(candidate: dict[str, Any], segment: dict[str, Any]) -> str:
    seg_id = str(segment.get("seg_id"))
    special_direction = special_management_segment_direction_overrides().get(seg_id)
    if special_direction == "forward":
        return "SPECIAL MANAGEMENT: follow the signed one-way direction."
    if special_direction == "reverse":
        return "SPECIAL MANAGEMENT: follow the signed one-way direction opposite official geometry."
    planned = (
        ((candidate.get("direction_validation") or {}).get("planned_traversal_direction") or {}).get(seg_id)
    )
    orientation = (candidate.get("route_orientation") or {}).get("direction")
    if segment.get("direction") == "ascent":
        if planned == "official_geometry_end_to_start":
            return "ASCENT REQUIRED: follow map arrows opposite official geometry."
        if planned == "official_geometry_start_to_end":
            return "ASCENT REQUIRED: follow map arrows uphill."
        return "ASCENT REQUIRED: verify current signage before running."
    if orientation == "reversed":
        return "Either direction allowed; planned opposite official geometry."
    return "Either direction allowed; follow map arrows."


def connector_cue(link: dict[str, Any]) -> dict[str, Any]:
    cue = {
        "source": link.get("source"),
        "from_segment_id": link.get("from_segment_id"),
        "to_segment_id": link.get("to_segment_id"),
        "from_trail": link.get("from_trail"),
        "to_trail": link.get("to_trail"),
        "distance_miles": round_miles(link.get("distance_miles") or 0),
        "connector_miles": round_miles(link.get("connector_miles") or 0),
        "official_repeat_miles": round_miles(link.get("official_repeat_miles") or 0),
        "official_repeat_segment_ids": normalized_segment_ids(link.get("official_repeat_segment_ids")),
        "earned_segment_ids_before_link": normalized_segment_ids(link.get("earned_segment_ids_before_link")),
        "avoided_unearned_segment_ids": normalized_segment_ids(link.get("avoided_unearned_segment_ids")),
        "connector_names": unique_nonempty(link.get("connector_names") or []),
        "signpost_labels": signpost_labels(link.get("connector_names") or [link.get("from_trail"), link.get("to_trail")]),
        "connector_classes": unique_nonempty(link.get("connector_classes") or []),
    }
    cue = {key: value for key, value in cue.items() if value not in (None, "", [])}
    cue.update(path_payload(link.get("path_coordinates")))
    return cue


def track_points_with_cumulative_miles(parts: list[list[tuple[float, float]]]) -> list[tuple[tuple[float, float], float]]:
    points: list[tuple[tuple[float, float], float]] = []
    total = 0.0
    for part in parts:
        previous = None
        for point in part:
            if previous is not None:
                total += haversine_miles(previous, point)
            points.append((point, total))
            previous = point
    return points


def car_passes_for_track_parts(
    parts: list[list[tuple[float, float]]],
    trailhead: dict[str, Any],
    radius_miles: float = CAR_PASS_RADIUS_MILES,
    endpoint_buffer_miles: float = CAR_PASS_ENDPOINT_BUFFER_MILES,
    min_separation_miles: float = CAR_PASS_MIN_SEPARATION_MILES,
) -> list[dict[str, Any]]:
    track_points = track_points_with_cumulative_miles(parts)
    if len(track_points) < 3 or trailhead.get("lon") is None or trailhead.get("lat") is None:
        return []
    parking_point = (float(trailhead["lon"]), float(trailhead["lat"]))
    total_miles = track_points[-1][1] if track_points else 0.0
    passes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for point, mile_from_start in track_points:
        if mile_from_start < endpoint_buffer_miles or total_miles - mile_from_start < endpoint_buffer_miles:
            continue
        distance_to_car = haversine_miles(parking_point, point)
        if distance_to_car > radius_miles:
            continue
        if current is None or mile_from_start - float(current["last_mile"]) > min_separation_miles:
            current = {
                "name": "Pass by car again",
                "mile_from_start": round_miles(mile_from_start),
                "distance_to_car_miles": round_miles(distance_to_car),
                "lon": point[0],
                "lat": point[1],
                "last_mile": mile_from_start,
            }
            passes.append(current)
            continue
        current["last_mile"] = mile_from_start
        if distance_to_car < float(current["distance_to_car_miles"]):
            current.update(
                {
                    "mile_from_start": round_miles(mile_from_start),
                    "distance_to_car_miles": round_miles(distance_to_car),
                    "lon": point[0],
                    "lat": point[1],
                }
            )
    for index, item in enumerate(passes, start=1):
        item.pop("last_mile", None)
        item["pass_number"] = index
    return passes


def known_water_points(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    trailhead = candidate.get("trailhead") or {}
    if trailhead.get("has_water") is not True:
        return []
    if trailhead.get("lon") is None or trailhead.get("lat") is None:
        return []
    return [
        {
            "name": trailhead.get("name") or "Parking/start",
            "location": "parking/start",
            "confidence": trailhead.get("water_confidence") or trailhead.get("source") or "verified",
            "lon": float(trailhead["lon"]),
            "lat": float(trailhead["lat"]),
        }
    ]


def logistics_for_candidate(
    candidate_id: str,
    candidate: dict[str, Any],
    props: dict[str, Any],
    rendered_parts: list[list[tuple[float, float]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trailhead = candidate.get("trailhead") or {}
    car_passes = car_passes_for_track_parts(rendered_parts, trailhead)
    planned_on_foot_miles = float(props.get("on_foot_miles") or 0)
    if planned_on_foot_miles > 0:
        latest_useful_car_pass_mile = max(0.0, planned_on_foot_miles - CAR_PASS_ENDPOINT_BUFFER_MILES)
        car_passes = [
            car_pass
            for car_pass in car_passes
            if float(car_pass.get("mile_from_start") or 0) <= latest_useful_car_pass_mile
        ]
    water_points = known_water_points(candidate)
    features = []
    for car_pass in car_passes:
        features.append(
            point_feature(
                float(car_pass["lon"]),
                float(car_pass["lat"]),
                {
                    **props,
                    "kind": "car_pass",
                    "candidate_id": candidate_id,
                    "name": car_pass["name"],
                    "pass_number": car_pass["pass_number"],
                    "mile_from_start": car_pass["mile_from_start"],
                    "distance_to_car_miles": car_pass["distance_to_car_miles"],
                },
            )
        )
    for water in water_points:
        features.append(
            point_feature(
                float(water["lon"]),
                float(water["lat"]),
                {
                    **props,
                    "kind": "water",
                    "candidate_id": candidate_id,
                    "name": water["name"],
                    "location": water["location"],
                    "confidence": water["confidence"],
                },
            )
        )
    return {"car_passes": car_passes, "known_water": water_points}, features


def route_cue(
    candidate: dict[str, Any],
    route: dict[str, Any],
    connector_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trailhead = candidate.get("trailhead") or {}
    trailhead_access = candidate.get("trailhead_access") or {}
    return_to_car = candidate.get("return_to_car") or {}
    outbound = trailhead_access.get("outbound_path_coordinates") or []
    current = None
    if outbound:
        last = outbound[-1]
        if isinstance(last, list | tuple) and len(last) >= 2:
            current = (float(last[0]), float(last[1]))
    raw_segments = list(candidate.get("segments") or [])
    official_index = route_cue_official_index()
    if all(int(segment.get("seg_id")) in official_index for segment in raw_segments if segment.get("seg_id") is not None):
        cue_segments = candidate_segments_for_track(candidate, official_index, current)
    else:
        cue_segments = raw_segments
    segments = []
    required_segment_ids = {
        int(segment.get("seg_id"))
        for segment in cue_segments
        if str(segment.get("seg_id") or "").isdigit()
    }
    earned_segment_ids: set[int] = set()
    link_by_to_segment_id = {
        str(link.get("to_segment_id")): link
        for link in ((candidate.get("inter_segment_links") or {}).get("links") or [])
        if link.get("to_segment_id") is not None
    }
    computed_between_links: list[dict[str, Any]] = []
    previous_trail = None
    previous_segment: dict[str, Any] | None = None
    for index, segment in enumerate(cue_segments, start=1):
        seg_id = int(segment["seg_id"]) if str(segment.get("seg_id") or "").isdigit() else None
        stored_link = link_by_to_segment_id.get(str(segment.get("seg_id")))
        segment_coords = coordinate_path(segment.get("coordinates"))
        selected_link = stored_link if previous_trail is not None else None
        if seg_id is not None and seg_id in official_index:
            segment_coords, selected_link = best_connector_link_for_oriented_segment(
                candidate=candidate,
                segment=segment,
                official_index=official_index,
                current=current,
                previous_segment=previous_segment,
                stored_link=stored_link if previous_trail is not None else None,
                connector_graph=connector_graph,
                stitch_snap_tolerance_miles=0.02,
                avoid_official_segment_ids=required_segment_ids - earned_segment_ids,
                earned_segment_ids_before_link=earned_segment_ids,
            )
        row = {
            "order": index,
            "seg_id": segment.get("seg_id"),
            "segment_name": segment.get("seg_name") or segment.get("trail_name"),
            "trail_name": segment.get("trail_name"),
            "signpost_label": signpost_label(segment.get("trail_name")),
            "official_miles": round_miles(segment.get("official_miles") or 0),
            "direction_rule": segment.get("direction"),
            "direction_cue": segment_direction_cue(candidate, segment),
            "estimated_moving_minutes": segment.get("estimated_moving_minutes"),
            "estimated_moving_minutes_p75": segment.get("estimated_moving_minutes_p75"),
            "ascent_ft": segment.get("ascent_ft"),
            "descent_ft": segment.get("descent_ft"),
            "grade_adjusted_miles": segment.get("grade_adjusted_miles"),
            "elevation_source": segment.get("elevation_source"),
        }
        if segment_coords:
            row["coordinates"] = [[float(lon), float(lat)] for lon, lat in segment_coords]
            row["start"] = row["coordinates"][0]
            row["end"] = row["coordinates"][-1]
        if selected_link and previous_segment is not None:
            row["pre_connector_link"] = connector_cue(selected_link)
            if previous_trail is not None and segment.get("trail_name") != previous_trail:
                computed_between_links.append(selected_link)
        if segment_coords:
            current = segment_coords[-1]
        if seg_id is not None:
            earned_segment_ids.add(seg_id)
        previous_trail = segment.get("trail_name")
        previous_segment = segment
        segments.append(row)
    inter_segment_links = (
        computed_between_links
        if connector_graph is not None and required_segment_ids
        else ((candidate.get("inter_segment_links") or {}).get("links")) or []
    )
    if connector_graph is not None and required_segment_ids:
        between_links = inter_segment_links
    else:
        between_links = inter_segment_links or (
            safe_between_trail_links_for_candidate(candidate, official_index, connector_graph)
            if connector_graph
            else (((candidate.get("between_trail_links") or {}).get("links")) or [])
        )
    trailhead_point = None
    if trailhead.get("lon") is not None and trailhead.get("lat") is not None:
        trailhead_point = (float(trailhead["lon"]), float(trailhead["lat"]))
    if current is not None and trailhead_point is not None and connector_graph is not None:
        mapped_return = shortest_connector_path(
            current,
            trailhead_point,
            connector_graph,
            0.02,
        )
        if mapped_return:
            return_to_car = {
                "strategy": "mapped_direct_to_trailhead",
                "description": "Return by the shortest graph-routed trail/path network back to the parked trailhead.",
                "official_repeat_miles": mapped_return.get("official_repeat_miles"),
                "official_repeat_segment_ids": mapped_return.get("official_repeat_segment_ids") or [],
                "connector_miles": mapped_return.get("connector_miles"),
                "road_miles": 0,
                "connector_names": mapped_return.get("connector_names") or [],
                "connector_classes": mapped_return.get("connector_classes") or [],
                "path_coordinates": mapped_return.get("path_coordinates") or [],
            }
    return_cue = {
        "strategy": return_to_car.get("strategy"),
        "description": return_to_car.get("description"),
        "official_repeat_miles": round_miles(return_to_car.get("official_repeat_miles") or 0),
        "official_repeat_segment_ids": normalized_segment_ids(return_to_car.get("official_repeat_segment_ids")),
        "connector_miles": round_miles(return_to_car.get("connector_miles") or 0),
        "road_miles": round_miles(return_to_car.get("road_miles") or 0),
        "connector_names": unique_nonempty(return_to_car.get("connector_names") or []),
        "connector_classes": unique_nonempty(return_to_car.get("connector_classes") or []),
    }
    return_cue.update(path_payload(return_to_car.get("path_coordinates")))
    start_access_cue = {
        "confidence": (candidate.get("validation") or {}).get("trailhead_snap_confidence"),
        "direct_gap_miles": round_miles(trailhead_access.get("direct_gap_miles") or 0),
        "mapped_access_miles": round_miles(trailhead_access.get("mapped_access_miles") or 0),
        "official_repeat_miles": round_miles(
            trailhead_access.get("one_way_official_repeat_miles")
            or trailhead_access.get("official_repeat_miles")
            or 0
        ),
        "official_repeat_segment_ids": normalized_segment_ids(trailhead_access.get("official_repeat_segment_ids")),
        "access_class": trailhead_access.get("access_class"),
        "graph_validated": trailhead_access.get("graph_validated"),
    }
    start_access_cue.update(path_payload(trailhead_access.get("path_coordinates")))
    return {
        "candidate_id": candidate.get("candidate_id"),
        "title": ", ".join(candidate.get("trail_names") or []),
        "route_status": candidate.get("route_status"),
        "official_miles": route.get("official_miles") or candidate.get("official_new_miles"),
        "on_foot_miles": route.get("on_foot_miles") or candidate.get("estimated_total_on_foot_miles"),
        "raw_total_minutes": candidate.get("raw_total_minutes") or route.get("raw_total_minutes"),
        "time_estimates_minutes": candidate.get("time_estimates_minutes") or route.get("time_estimates_minutes"),
        "total_minutes": route.get("total_minutes") or candidate.get("total_minutes"),
        "trailhead": {
            "name": trailhead.get("name") or route.get("trailhead"),
            "lat": trailhead.get("lat"),
            "lon": trailhead.get("lon"),
            "has_parking": trailhead.get("has_parking"),
            "has_restroom": trailhead.get("has_restroom"),
            "has_water": trailhead.get("has_water"),
            "water_confidence": trailhead.get("water_confidence"),
            "parking_minutes": trailhead.get("parking_minutes"),
            "source": trailhead.get("source"),
            "parking_confidence": trailhead.get("parking_confidence"),
        },
        "start_access": start_access_cue,
        "segments": segments,
        "between_links": [
            connector_cue(link)
            for link in between_links
        ],
        "return_to_car": return_cue,
        "validation": {
            "ascent_direction_passed": ((candidate.get("validation") or {}).get("ascent_direction_passed")),
            "return_path_graph_validated": ((candidate.get("validation") or {}).get("return_path_graph_validated")),
            "trailhead_snap_confidence": ((candidate.get("validation") or {}).get("trailhead_snap_confidence")),
        },
    }


def build_map_data(
    package_pass: dict[str, Any],
    source_route_pass: dict[str, Any],
    plan: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_index = load_candidate_index(plan) if plan.get("route_menu") else {}
    candidate_index.update(source_route_pass.get("candidate_index") or {})
    package_by_candidate = {
        str(candidate_id): package
        for package in package_pass.get("packages") or []
        for candidate_id in package.get("component_candidate_ids") or []
    }
    route_lookup = {
        str(route["candidate_id"]): route for route in source_route_pass.get("routes") or []
    }
    route_features = []
    official_features = []
    parking_features = []
    logistics_features = []
    route_cues = {}
    validations = []
    for candidate_id, package in package_by_candidate.items():
        candidate = candidate_index[candidate_id]
        route = route_lookup.get(candidate_id) or {}
        color = PALETTE[(int(package["package_number"]) - 1) % len(PALETTE)]
        coords = candidate_track_coordinates(
            candidate,
            official_index,
            connector_graph=connector_graph,
            densify_source_lines=True,
        )
        source_validation = validate_track_segments([coords], max_gap_miles=0.1)
        rendered_parts = split_coords_on_gaps(coords, max_gap_miles=0.1)
        render_validation = validate_track_segments(rendered_parts, max_gap_miles=0.1)
        props = {
            "kind": "route",
            "package_number": package["package_number"],
            "candidate_id": candidate_id,
            "block_name": package["block_name"],
            "title": ", ".join(candidate.get("trail_names") or []),
            "official_miles": route.get("official_miles"),
            "on_foot_miles": route.get("on_foot_miles"),
            "trailhead": route.get("trailhead"),
            "color": color,
            "source_gap_warning_count": len(source_validation["failures"]),
        }
        route_feature = multiline_feature(rendered_parts, props)
        if route_feature:
            route_features.append(route_feature)
        parking = parking_feature(candidate, props)
        if parking:
            parking_features.append(parking)
        cue = route_cue(candidate, route, connector_graph=connector_graph)
        cue["logistics"], new_logistics_features = logistics_for_candidate(
            candidate_id,
            candidate,
            props,
            rendered_parts,
        )
        route_cues[candidate_id] = cue
        logistics_features.extend(new_logistics_features)
        for segment in candidate.get("segments") or []:
            seg_coords = candidate_segment_coordinates(candidate, segment, official_index)
            feature = line_feature(
                seg_coords,
                {
                    **props,
                    "kind": "official_segment",
                    "seg_id": segment.get("seg_id"),
                    "segment_name": segment.get("seg_name") or segment.get("trail_name"),
                    "trail_name": segment.get("trail_name"),
                    "segment_official_miles_raw": segment.get("official_miles"),
                    "direction_rule": segment.get("direction"),
                    "direction_cue": segment_direction_cue(candidate, segment),
                },
            )
            if feature:
                official_features.append(feature)
        validations.append(
            {
                "candidate_id": candidate_id,
                "source_gap_warning": not source_validation["passed"],
                "source_max_gap_miles": source_validation["max_trackpoint_gap_miles"],
                "rendered_passed": source_validation["passed"] and render_validation["passed"],
                "rendered_failures": render_validation["failures"],
            }
        )
    map_data = {
        "summary": package_pass["summary"],
        "progress": {
            "completed_segment_ids": sorted(int(item) for item in (plan.get("state_inputs") or {}).get("completed_segment_ids") or []),
            "blocked_segment_ids": sorted(int(item) for item in (plan.get("state_inputs") or {}).get("blocked_segment_ids") or []),
        },
        "packages": package_pass["packages"],
        "feature_collections": {
            "routes": {"type": "FeatureCollection", "features": route_features},
            "official_segments": {"type": "FeatureCollection", "features": official_features},
            "parking": {"type": "FeatureCollection", "features": parking_features},
            "logistics": {"type": "FeatureCollection", "features": logistics_features},
        },
        "route_cues": route_cues,
        "map_validation": {
            "rendered_passed": all(item["rendered_passed"] for item in validations),
            "source_gap_warning_count": sum(1 for item in validations if item["source_gap_warning"]),
            "route_validations": validations,
        },
    }
    return apply_human_route_names_to_map_data(map_data)


def render_markdown(package_pass: dict[str, Any]) -> str:
    summary = package_pass["summary"]
    lines = [
        "# 2026 Route Package Pass v1",
        "",
        "Status: reviewable route-package plan built from existing graph-validated route components.",
        "",
        "## Summary",
        "",
        f"- Packages: {summary['package_count']}",
        f"- Route components absorbed: {summary['component_route_count']}",
        f"- Covered segments: {summary['covered_segment_count']} / 251",
        f"- Official miles: {summary['official_miles']}",
        f"- Total on-foot miles: {summary['total_on_foot_miles']}",
        f"- On-foot/official ratio: {summary['planwide_on_foot_to_official_ratio']}x",
        f"- Packages with multiple trailheads: {summary['packages_with_multiple_trailheads']}",
        f"- Sub-1-mile components absorbed: {summary['component_routes_under_1_official_mile']}",
        f"- Sub-2-mile components absorbed: {summary['component_routes_under_2_official_miles']}",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in package_pass.get("caveats") or [])
    lines.extend(
        [
            "",
            "## Packages",
            "",
            "| # | Block | Status | Trailhead(s) | Plan | Official mi | On-foot mi | Ratio | Reasons |",
            "|---:|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for package in package_pass.get("packages") or []:
        reasons = ", ".join(package.get("planning_reasons") or [])
        trailheads = ", ".join(package.get("trailheads") or [])
        lines.append(
            f"| {package['package_number']} | {package['block_name']} | {package['planning_status']} | "
            f"{trailheads} | {package_start_plan(package)} | {package['official_miles']} | "
            f"{package['on_foot_miles']} | {package['ratio']} | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


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


def short_parking_name(name: str | None) -> str:
    value = str(name or "Park here")
    for suffix in (" Parking/Trailhead", " Trailhead", " Trail Access Point"):
        value = value.replace(suffix, "")
    return value


def format_miles(value: Any) -> str:
    try:
        return str(round(float(value), 2))
    except (TypeError, ValueError):
        return "n/a"


def format_minutes(value: Any) -> str:
    try:
        minutes = round(float(value))
    except (TypeError, ValueError):
        return "time n/a"
    if minutes <= 0:
        return "time n/a"
    hours = minutes // 60
    mins = minutes % 60
    if not hours:
        return f"{mins} min"
    if not mins:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def outing_time_bucket(total_minutes: int | float | None) -> str:
    minutes = float(total_minutes or 0)
    if minutes <= 120:
        return "2 hours or less"
    if minutes <= 180:
        return "2-3 hours"
    if minutes <= 240:
        return "3-4 hours"
    return "4+ hours"


def outing_time_bucket_sort(bucket: str) -> int:
    return {
        "2 hours or less": 1,
        "2-3 hours": 2,
        "3-4 hours": 3,
        "4+ hours": 4,
    }.get(bucket, 99)


def manual_design_areas(map_data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((map_data.get("manual_design") or {}).get("areas") or []))


def manual_design_area_for_candidate_ids(
    map_data: dict[str, Any],
    package_number: int | str | None,
    candidate_ids: list[str],
) -> dict[str, Any] | None:
    candidates = {str(candidate_id) for candidate_id in candidate_ids}
    for area in manual_design_areas(map_data):
        if area.get("package_number") is not None and str(area["package_number"]) != str(package_number):
            continue
        demoted = {str(candidate_id) for candidate_id in area.get("demote_candidate_ids") or []}
        if candidates & demoted:
            return area
    return None


def official_segment_props_by_id(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    collection = (map_data.get("feature_collections") or {}).get("official_segments") or {}
    index: dict[str, dict[str, Any]] = {}
    for feature in collection.get("features") or []:
        props = feature.get("properties") or {}
        segment_id = props.get("seg_id") or props.get("segment_id") or props.get("segId")
        if segment_id is not None:
            index[str(segment_id)] = props
    return index


def official_segment_miles(props: dict[str, Any]) -> float:
    for key in ("segment_official_miles_raw", "segment_official_miles", "official_miles", "LengthMi", "length_miles"):
        value = props.get(key)
        if value is not None:
            return float(value)
    length_ft = props.get("LengthFt") or props.get("length_ft")
    if length_ft is not None:
        return float(length_ft) / 5280.0
    return 0.0


def official_miles_for_segment_ids(
    segment_ids: list[str],
    official_props_by_id: dict[str, dict[str, Any]],
) -> float | None:
    miles = 0.0
    found = False
    for segment_id in segment_ids:
        props = official_props_by_id.get(str(segment_id))
        if not props:
            continue
        segment_miles = official_segment_miles(props)
        if segment_miles <= 0:
            continue
        found = True
        miles += segment_miles
    return round_miles(miles) if found else None


def trails_for_segment_ids(
    segment_ids: list[str],
    official_props_by_id: dict[str, dict[str, Any]],
    fallback_trails: list[str],
) -> list[str]:
    trails: list[str] = []
    for segment_id in segment_ids:
        props = official_props_by_id.get(str(segment_id)) or {}
        trail = props.get("trail_name") or props.get("TrailName") or props.get("trailName")
        if trail and trail not in trails:
            trails.append(str(trail))
    return trails or fallback_trails


def build_outing_menu(map_data: dict[str, Any]) -> list[dict[str, Any]]:
    completed = {str(item) for item in (map_data.get("progress") or {}).get("completed_segment_ids") or []}
    official_props_by_id = official_segment_props_by_id(map_data)
    r2r_index = load_r2r_route_name_index()
    outings: list[dict[str, Any]] = []
    for package in map_data.get("packages") or []:
        starts: list[dict[str, Any]] = []
        for component in package.get("components") or []:
            if component.get("accepted_replacement_status") == "active" and component.get("accepted_anchor_label"):
                trailhead = str(component["accepted_anchor_label"])
            else:
                trailhead = short_parking_name(component.get("trailhead") or "parking TBD")
            group_key = str(component.get("field_menu_group_id") or trailhead)
            group = next((item for item in starts if item["group_key"] == group_key), None)
            if not group:
                group = {
                    "group_key": group_key,
                    "trailhead": trailhead,
                    "label_override": component.get("field_menu_label"),
                    "official_miles": 0.0,
                    "on_foot_miles": 0.0,
                    "total_minutes": 0,
                    "trails": [],
                    "candidate_ids": [],
                    "segment_ids": [],
                    "accepted_replacement_ids": [],
                    "start_justifications": [],
                    "route_card_statuses": [],
                    "packet_visibilities": [],
                    "certified_route_card_values": [],
                    "requires_field_walkthrough_values": [],
                    "cue_generation_modes": [],
                    "anchor_to_credit_endpoint_distance_miles": [],
                    "credit_endpoint_used": [],
                    "route_quality_values": [],
                }
                starts.append(group)
            group["official_miles"] += float(component.get("official_miles") or 0)
            group["on_foot_miles"] += float(component.get("on_foot_miles") or 0)
            group["total_minutes"] += int(component.get("total_minutes") or 0)
            if component.get("candidate_id"):
                group["candidate_ids"].append(str(component["candidate_id"]))
            for segment_id in component.get("segment_ids") or []:
                value = str(segment_id)
                if value not in group["segment_ids"]:
                    group["segment_ids"].append(value)
            for trail in component.get("trail_names") or []:
                if trail not in group["trails"]:
                    group["trails"].append(trail)
            if component.get("accepted_replacement_id"):
                group["accepted_replacement_ids"].append(str(component["accepted_replacement_id"]))
            if component.get("start_justification"):
                group["start_justifications"].append(str(component["start_justification"]))
            if component.get("route_card_status"):
                group["route_card_statuses"].append(str(component["route_card_status"]))
            if component.get("packet_visibility"):
                group["packet_visibilities"].append(str(component["packet_visibility"]))
            if component.get("certified_route_card") is not None:
                group["certified_route_card_values"].append(bool(component["certified_route_card"]))
            if component.get("requires_field_walkthrough") is not None:
                group["requires_field_walkthrough_values"].append(bool(component["requires_field_walkthrough"]))
            if component.get("cue_generation_mode"):
                group["cue_generation_modes"].append(str(component["cue_generation_mode"]))
            if component.get("anchor_to_credit_endpoint_distance_miles") is not None:
                group["anchor_to_credit_endpoint_distance_miles"].append(component["anchor_to_credit_endpoint_distance_miles"])
            if component.get("credit_endpoint_used"):
                group["credit_endpoint_used"].append(str(component["credit_endpoint_used"]))
            if component.get("route_quality"):
                group["route_quality_values"].append(component["route_quality"])
        for index, start in enumerate(starts):
            segment_ids = list(start["segment_ids"])
            remaining_segment_ids = [segment_id for segment_id in segment_ids if segment_id not in completed]
            if segment_ids and not remaining_segment_ids:
                continue
            display_segment_ids = remaining_segment_ids or segment_ids
            official_miles = official_miles_for_segment_ids(display_segment_ids, official_props_by_id)
            if official_miles is None:
                official_miles = round_miles(start["official_miles"])
            display_trails = trails_for_segment_ids(display_segment_ids, official_props_by_id, start["trails"])
            label = str(
                start.get("label_override")
                or f"{package['package_number']}{chr(65 + index) if len(starts) > 1 else ''}"
            )
            route_name = human_route_name(display_trails, start["trailhead"], r2r_index=r2r_index)
            manual_area = manual_design_area_for_candidate_ids(
                map_data,
                package.get("package_number"),
                start["candidate_ids"],
            )
            outings.append(
                {
                    "outing_id": f"{package['package_number']}-{index + 1}",
                    "label": label,
                    "route_code": label,
                    **route_name,
                    "package_number": package["package_number"],
                    "package_start_count": len(starts),
                    "block_name": package.get("block_name") or f"Package {package['package_number']}",
                    "trailhead": start["trailhead"],
                    "trails": display_trails,
                    "official_miles": official_miles,
                    "on_foot_miles": round_miles(start["on_foot_miles"]),
                    "total_minutes": start["total_minutes"],
                    "candidate_ids": start["candidate_ids"],
                    "segment_ids": segment_ids,
                    "remaining_segment_ids": remaining_segment_ids,
                    "remaining_segment_count": len(remaining_segment_ids),
                    "time_bucket": outing_time_bucket(start["total_minutes"]),
                    "manual_design_hold": bool(manual_area),
                    "manual_design_area_id": manual_area.get("area_id") if manual_area else None,
                    "manual_design_status": manual_area.get("status") if manual_area else None,
                    "manual_design_decision": manual_area.get("decision") if manual_area else None,
                    "accepted_replacement_id": start["accepted_replacement_ids"][0]
                    if start["accepted_replacement_ids"]
                    else None,
                    "start_justification": start["start_justifications"][0]
                    if start["start_justifications"]
                    else None,
                    "route_card_status": start["route_card_statuses"][0] if start["route_card_statuses"] else None,
                    "packet_visibility": start["packet_visibilities"][0] if start["packet_visibilities"] else None,
                    "certified_route_card": all(start["certified_route_card_values"])
                    if start["certified_route_card_values"]
                    else None,
                    "requires_field_walkthrough": any(start["requires_field_walkthrough_values"])
                    if start["requires_field_walkthrough_values"]
                    else None,
                    "cue_generation_mode": start["cue_generation_modes"][0]
                    if start["cue_generation_modes"]
                    else None,
                    "anchor_to_credit_endpoint_distance_miles": min(start["anchor_to_credit_endpoint_distance_miles"])
                    if start["anchor_to_credit_endpoint_distance_miles"]
                    else None,
                    "credit_endpoint_used": start["credit_endpoint_used"][0] if start["credit_endpoint_used"] else None,
                    "route_quality": start["route_quality_values"][0]
                    if len(start["route_quality_values"]) == 1
                    else {},
                }
            )
    return sorted(
        outings,
        key=lambda outing: (
            outing_time_bucket_sort(outing["time_bucket"]),
            int(outing.get("total_minutes") or 0),
            str(outing.get("label") or ""),
        ),
    )


def visible_manual_design_areas(map_data: dict[str, Any], outings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    held_area_ids = {
        str(outing.get("manual_design_area_id"))
        for outing in outings
        if outing.get("manual_design_hold") and outing.get("remaining_segment_ids")
    }
    if not held_area_ids:
        return []
    return [area for area in manual_design_areas(map_data) if str(area.get("area_id")) in held_area_ids]


def render_outing_menu_markdown(map_data: dict[str, Any], map_html_path: Path | None = None) -> str:
    outings = build_outing_menu(map_data)
    normal_outings = [outing for outing in outings if not outing.get("manual_design_hold")]
    manual_outings = [outing for outing in outings if outing.get("manual_design_hold")]
    manual_areas = visible_manual_design_areas(map_data, outings)
    remaining_segments = len({segment_id for outing in outings for segment_id in outing.get("remaining_segment_ids") or []})
    summary = map_data.get("summary") or {}
    ratio = summary.get("planwide_on_foot_to_official_ratio")
    if ratio is None:
        official = float(summary.get("official_miles") or 0)
        on_foot = float(summary.get("total_on_foot_miles") or 0)
        ratio = round(on_foot / official, 2) if official else None
    ratio_text = f"{ratio}x" if ratio is not None else "unknown"
    lines = [
        "# 2026 Outing Menu",
        "",
        "Status: written companion to the canonical outing map.",
        "",
        "Use this like the map: pick the door-to-door time you actually have, choose one parked-start outing, then check current trail conditions and signage before leaving.",
        "",
        "## Summary",
        "",
        f"- Open runnable outings: {len(normal_outings)}",
        f"- Manual design holds: {len(manual_outings)}",
        f"- Remaining official segments represented: {remaining_segments}",
        f"- Full-plan official miles: {summary.get('official_miles')}",
        f"- Full-plan on-foot miles: {summary.get('total_on_foot_miles')}",
        f"- Full-plan on-foot/official ratio: {ratio_text}",
    ]
    if map_html_path:
        lines.append(f"- Map: `{map_html_path}`")
    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "- Each row is one executable parked-start outing, not a calendar day and not a multi-start package.",
            "- Door-to-door time uses the planner's configured origin estimate, including drive, parking/prep, access, route/return movement, and drive home.",
            "- Completed outings are omitted when all official segment IDs in that outing are already in `completed_segment_ids`.",
            "- If a row says it belongs to a package with multiple starts, pair it with the related start only when today's time allows.",
            "- Manual design holds are not runnable menu items yet. They record coverage placeholders that need a better human route before scheduling.",
        ]
    )
    if manual_areas:
        lines.extend(["", "## Manual Design Areas", ""])
        for area in manual_areas:
            placeholder = area.get("current_placeholder") or {}
            lines.extend(
                [
                    f"### {area.get('title') or area.get('area_id')}",
                    "",
                    f"Decision: {area.get('decision')}",
                    "",
                    "| Current placeholder | Door-to-door | Official mi | On-foot mi | Why held |",
                    "|---|---:|---:|---:|---|",
                    (
                        f"| {placeholder.get('label', 'hold')} from {placeholder.get('trailhead', 'unknown')} | "
                        f"{format_minutes(placeholder.get('door_to_door_minutes'))} | "
                        f"{format_miles(placeholder.get('official_miles'))} | "
                        f"{format_miles(placeholder.get('on_foot_miles'))} | "
                        f"{placeholder.get('reason', 'manual design required')} |"
                    ),
                    "",
                ]
            )
            split_probe = area.get("default_split_probe") or {}
            if split_probe.get("alternative_ids"):
                lines.extend(
                    [
                        "Current best split probe:",
                        f"- Alternatives: {', '.join(split_probe.get('alternative_ids') or [])}",
                        f"- Official miles: {format_miles(split_probe.get('official_miles'))}",
                        f"- On-foot miles: {format_miles(split_probe.get('on_foot_miles'))}",
                        f"- Door-to-door if run separately: {format_minutes(split_probe.get('door_to_door_minutes_if_separate_outings'))}",
                        f"- Improvement vs current 16A placeholder: {format_miles(split_probe.get('improvement_vs_current_on_foot_miles'))} on-foot miles",
                        f"- Probe acceptance passed: {split_probe.get('passes_probe_acceptance')}",
                        "",
                    ]
                )
            lines.extend(
                [
                    "| Alternative | Status | Target official | Target on-foot | Required segments | Notes |",
                    "|---|---|---:|---:|---|---|",
                ]
            )
            for alternative in area.get("alternatives") or []:
                target_range = alternative.get("target_on_foot_miles_range") or []
                if len(target_range) == 2:
                    target_text = f"{format_miles(target_range[0])}-{format_miles(target_range[1])}"
                else:
                    target_text = "n/a"
                probe = alternative.get("probe") or {}
                probe_text = ""
                if probe and not probe.get("error"):
                    probe_text = (
                        f" Probe: {probe.get('route_status')}, {format_miles(probe.get('on_foot_miles'))} on-foot mi, "
                        f"ascent={probe.get('ascent_direction_passed')}."
                    )
                notes = "; ".join(alternative.get("design_notes") or [])
                lines.append(
                    f"| {alternative.get('alternative_id')}: {alternative.get('title')} | "
                    f"{alternative.get('route_design_status') or alternative.get('status')} | {format_miles(alternative.get('target_official_miles'))} | "
                    f"{target_text} | {', '.join(str(item) for item in alternative.get('required_segment_ids') or [])} | "
                    f"{notes}{probe_text} |"
                )
            if area.get("acceptance_gates"):
                lines.extend(["", "Acceptance gates:"])
                lines.extend(f"- {gate}" for gate in area.get("acceptance_gates") or [])
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outing in normal_outings:
        by_bucket[outing["time_bucket"]].append(outing)
    for bucket in sorted(by_bucket, key=outing_time_bucket_sort):
        lines.extend(
            [
                "",
                f"## {bucket}",
                "",
                "| Outing | Door-to-door | Park/start | Official mi | On-foot mi | Remaining segs | Route package | Trails |",
                "|---|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for outing in by_bucket[bucket]:
            related = (
                f"Package {outing['package_number']} ({outing['package_start_count']} starts)"
                if outing["package_start_count"] > 1
                else f"Package {outing['package_number']}"
            )
            outing_name = outing.get("route_name") or outing["label"]
            code_text = f" ({outing['label']})" if outing.get("label") and outing.get("label") != outing_name else ""
            lines.append(
                f"| {outing_name}{code_text} | {format_minutes(outing['total_minutes'])} | {outing['trailhead']} | "
                f"{format_miles(outing['official_miles'])} | {format_miles(outing['on_foot_miles'])} | "
                f"{outing['remaining_segment_count']} / {len(outing['segment_ids'])} | "
                f"{related}: {outing['block_name']} | {', '.join(outing.get('trails') or [])} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_html(map_data: dict[str, Any]) -> str:
    apply_human_route_names_to_map_data(map_data)
    payload = json.dumps(map_data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2026 Outing Menu Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#1f2933; }}
    .app {{ display:grid; grid-template-columns: 430px minmax(0,1fr); min-height:100vh; }}
    aside {{ overflow:auto; border-right:1px solid #d7ddd4; background:#fff; }}
    header {{ padding:16px; border-bottom:1px solid #d7ddd4; }}
    h1 {{ margin:0 0 6px; font-size:20px; letter-spacing:0; }}
    p {{ margin:0; color:#667085; font-size:13px; line-height:1.45; }}
    .metrics {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; padding:12px 16px; border-bottom:1px solid #d7ddd4; }}
    .metric {{ border:1px solid #d7ddd4; border-radius:6px; padding:8px; }}
    .metric span {{ display:block; color:#667085; font-size:11px; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:3px; font-size:16px; }}
    #manualDesignAreas {{ padding:0 16px 4px; }}
    .manual-card {{ border:2px solid #b45309; border-radius:6px; margin:0 0 10px; padding:10px; background:#fff7ed; }}
    .manual-card strong {{ display:block; color:#7c2d12; font-size:13px; line-height:1.3; }}
    .manual-card span {{ display:block; color:#7c2d12; font-size:12px; line-height:1.4; }}
    .manual-card ul {{ margin:7px 0 0 18px; padding:0; color:#344054; font-size:12px; line-height:1.35; }}
    #packages {{ padding:10px 10px 16px; }}
    .package {{ border:1px solid #d7ddd4; border-radius:6px; margin:0 0 10px; padding:10px 12px; cursor:pointer; background:#fff; }}
    .package.active {{ background:#eef6ff; border-color:#2563eb; box-shadow:inset 4px 0 0 #2563eb; }}
    .package strong {{ display:block; font-size:13px; line-height:1.35; }}
    .package span {{ color:#667085; font-size:12px; line-height:1.35; }}
    .package-meta {{ display:block; margin-top:3px; }}
    .start-list {{ margin-top:8px; border-top:1px solid #e5e7eb; padding-top:6px; }}
    .start-row {{ display:flex; justify-content:space-between; gap:8px; padding:3px 0; color:#344054; font-size:12px; line-height:1.3; }}
    .start-row b {{ color:#111827; font-weight:700; }}
    .start-row em {{ color:#667085; font-style:normal; white-space:nowrap; }}
    .selected-panel {{ display:none; margin:12px 16px; border:2px solid #111827; border-radius:6px; padding:10px; background:#fff; }}
    .selected-panel.active {{ display:block; }}
    .selected-panel span {{ display:block; color:#667085; font-size:11px; font-weight:700; text-transform:uppercase; }}
    .selected-panel strong {{ display:block; margin-top:3px; font-size:15px; line-height:1.25; }}
    .selected-panel p {{ margin-top:6px; color:#344054; font-size:12px; }}
    .route-card {{ border:2px solid #111827; border-radius:8px; background:#fff; color:#111827; overflow:hidden; }}
    .route-card-header {{ padding:11px 12px; background:#111827; color:#fff; }}
    .route-card-header span {{ color:#cbd5e1; }}
    .route-card-header strong {{ color:#fff; font-size:18px; line-height:1.15; }}
    .route-card-header em {{ display:block; margin-top:3px; color:#cbd5e1; font-style:normal; font-size:12px; line-height:1.25; }}
    .route-card-body {{ padding:11px 12px 12px; }}
    .route-card-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin:10px 0; }}
    .route-card-kv {{ border:1px solid #d7ddd4; border-radius:6px; padding:7px; background:#f9fafb; }}
    .route-card-kv b {{ display:block; font-size:11px; color:#667085; text-transform:uppercase; }}
    .route-card-kv span {{ display:block; margin-top:2px; color:#111827; font-size:13px; font-weight:800; text-transform:none; }}
    .route-card-section {{ margin-top:10px; padding-top:9px; border-top:1px solid #e5e7eb; }}
    .route-card-section h2 {{ margin:0 0 6px; font-size:12px; letter-spacing:0; text-transform:uppercase; color:#111827; }}
    .route-card-section p {{ margin:4px 0; color:#344054; font-size:12px; line-height:1.35; }}
    .cue-list {{ display:grid; gap:4px; }}
    .cue-row {{ display:grid; grid-template-columns:24px minmax(0,1fr); gap:7px; align-items:start; border:1px solid #e5e7eb; border-radius:6px; padding:6px; background:#fff; }}
    .cue-row i {{ display:flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:999px; background:#111827; color:#fff; font-style:normal; font-size:11px; font-weight:800; }}
    .cue-row b {{ display:block; font-size:12px; color:#111827; line-height:1.25; }}
    .cue-row span {{ display:block; margin-top:1px; color:#475467; font-size:11px; line-height:1.3; text-transform:none; }}
    .cue-row.ascent {{ border-color:#b45309; background:#fff7ed; }}
    .cue-row.ascent i {{ background:#b45309; }}
    .card-note {{ border-left:4px solid #2563eb; padding:7px 8px; background:#eff6ff; color:#1e3a8a; font-size:12px; line-height:1.35; }}
    .legend {{ margin:0 16px 12px; border:1px solid #d7ddd4; border-radius:6px; padding:8px 9px; color:#475467; font-size:12px; line-height:1.4; }}
    .legend-arrow {{ display:inline-block; width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-bottom:12px solid #1f2937; vertical-align:-1px; margin-right:6px; transform:rotate(90deg); }}
    .time-filters {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; margin:0 16px 10px; }}
    .time-filter {{ margin:0; min-height:30px; border:1px solid #d7ddd4; background:#fff; border-radius:6px; padding:5px 7px; cursor:pointer; font-size:12px; }}
    .time-filter.active {{ background:#111827; border-color:#111827; color:#fff; }}
    .dir-arrow-wrap {{ background:transparent; border:0; }}
    .dir-arrow {{ width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:14px solid var(--route-color,#1f2937); filter:drop-shadow(0 0 2px #fff) drop-shadow(0 0 2px #fff); transform-origin:50% 50%; }}
    .parking-marker-wrap {{ background:transparent; border:0; }}
    .parking-marker {{ width:22px; height:22px; border-radius:50%; background:#111827; color:#fff; border:2px solid #fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; box-shadow:0 2px 8px rgba(15,23,42,.32); }}
    .logistics-marker-wrap {{ background:transparent; border:0; }}
    .logistics-marker {{ min-width:26px; height:24px; border-radius:999px; color:#111827; border:2px solid #fff; display:flex; align-items:center; justify-content:center; padding:0 6px; font-weight:900; font-size:10px; box-shadow:0 2px 9px rgba(15,23,42,.35); }}
    .logistics-marker.car-pass {{ background:#fde68a; }}
    .logistics-marker.water {{ background:#bae6fd; }}
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
    @media (max-width:860px) {{ .app {{ grid-template-columns:1fr; }} #map {{ min-height:55vh; }} .route-card-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media print {{ aside {{ border-right:0; }} #packages,.time-filters,#fitAll,.legend,#manualDesignAreas {{ display:none !important; }} .app {{ display:block; }} #map {{ min-height:45vh; }} .selected-panel {{ margin:0; border:0; }} }}
  </style>
</head>
<body>
<div class="app">
  <aside>
    <header>
      <h1>Outing Menu Map</h1>
      <p>Executable parked-start outings grouped by related trail package, filtered by door-to-door time and remaining challenge progress.</p>
    </header>
    <div class="metrics" id="metrics"></div>
    <div class="selected-panel" id="selectedPanel"></div>
    <div class="legend"><span class="legend-arrow"></span>Each card is one door-to-door outing from one parking/start location. Completed outings are hidden after progress is updated. Selected outings show one clear cased line with arrows, turn markers for double-backs, P markers for where to park/start, CAR markers for mid-route car access, and W markers for verified water.</div>
    <div id="manualDesignAreas"></div>
    <div class="time-filters" id="timeFilters">
      <button class="time-filter active" type="button" data-filter="all">All</button>
      <button class="time-filter" type="button" data-filter="120">≤2h d2d</button>
      <button class="time-filter" type="button" data-filter="180">≤3h d2d</button>
      <button class="time-filter" type="button" data-filter="240">≤4h d2d</button>
      <button class="time-filter" type="button" data-filter="over240">4h+ d2d</button>
    </div>
    <button id="fitAll" type="button">Fit all</button>
    <div id="packages"></div>
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
const logisticsLayer = L.layerGroup().addTo(map);
const arrowLayer = L.layerGroup().addTo(map);
function fmt(v,s="") {{ return v === null || v === undefined ? "n/a" : `${{v}}${{s}}`; }}
function esc(value) {{
  return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}
function popup(props) {{
  if (props.kind === "parking") {{
    return `<strong>Park: ${{props.name || props.trailhead}}</strong><br>${{props.block_name || ""}}<br>Parking: ${{props.has_parking === false ? "unknown" : "yes"}}<br>Water: ${{yesNoUnknown(props.has_water)}}<br>Prep: ${{fmt(props.parking_minutes," min")}}`;
  }}
  if (props.kind === "car_pass") {{
    return `<strong>Pass by car again</strong><br>${{props.block_name || ""}}<br>Approx mile ${{fmt(props.mile_from_start)}} from start · ${{fmt(props.distance_to_car_miles," mi")}} from parked car`;
  }}
  if (props.kind === "water") {{
    return `<strong>Known water: ${{props.name || "water"}}</strong><br>${{props.block_name || ""}}<br>${{props.location || "location"}} · ${{props.confidence || "verified"}}`;
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
function logisticsIcon(kind) {{
  const label = kind === "water" ? "W" : "CAR";
  const extraClass = kind === "water" ? "water" : "car-pass";
  return L.divIcon({{
    className:"logistics-marker-wrap",
    iconSize:[38,26],
    iconAnchor:[19,13],
    html:`<div class="logistics-marker ${{extraClass}}">${{label}}</div>`
  }});
}}
function shortParkingName(name) {{
  return String(name || "Park here").replace(/ Parking\\/Trailhead| Trailhead| Trail Access Point/g, "");
}}
function formatMiles(value) {{
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return String(Math.round(number * 100) / 100);
}}
function formatMinutes(value) {{
  const minutes = Number(value);
  if (!Number.isFinite(minutes) || minutes <= 0) return "time n/a";
  const rounded = Math.round(minutes);
  const hours = Math.floor(rounded / 60);
  const mins = rounded % 60;
  if (!hours) return `${{mins}} min`;
  if (!mins) return `${{hours}}h`;
  return `${{hours}}h ${{mins}}m`;
}}
function outingById(id) {{
  return (DATA.outings || []).find(outing => String(outing.outing_id) === String(id));
}}
function featureInOuting(feature, outing) {{
  return !outing || (outing.candidate_ids || []).includes(String(feature.properties.candidate_id));
}}
function visibleCandidateIds() {{
  return new Set((DATA.outings || []).filter(outingMatchesFilter).flatMap(outing => outing.candidate_ids || []));
}}
function manualDesignAreas() {{
  return DATA.manual_design?.areas || [];
}}
function manualDesignAreaForCandidateIds(packageNumber, candidateIds) {{
  const candidates = new Set((candidateIds || []).map(String));
  return manualDesignAreas().find(area => {{
    if (area.package_number !== undefined && String(area.package_number) !== String(packageNumber)) return false;
    return (area.demote_candidate_ids || []).some(candidateId => candidates.has(String(candidateId)));
  }}) || null;
}}
function featureInCurrentMenu(feature) {{
  return visibleCandidateIds().has(String(feature.properties.candidate_id));
}}
function parkingNamesForOuting(outing) {{
  const features = ((DATA.feature_collections.parking || {{ features:[] }}).features || [])
    .filter(feature => featureInOuting(feature, outing));
  return [...new Set(features.map(feature => feature.properties.name || feature.properties.trailhead).filter(Boolean))];
}}
function yesNoUnknown(value) {{
  if (value === true) return "yes";
  if (value === false) return "no";
  return "unknown";
}}
function parkingDetailsText(trailhead) {{
  const parts = [`Parking: ${{yesNoUnknown(trailhead.has_parking)}}`];
  if (trailhead.has_restroom === true) parts.push("Restroom: yes");
  if (trailhead.has_water === true) parts.push("Water: yes");
  return parts.join(" · ");
}}
function startAccessText(access) {{
  const miles = Number(access.mapped_access_miles);
  if (Number.isFinite(miles) && miles > 0.05) {{
    return `Route starts about ${{formatMiles(miles)}} mi from parking.`;
  }}
  return "Route starts at or very near the parking point.";
}}
function routeCuesForOuting(outing) {{
  return (outing.candidate_ids || []).map(candidateId => DATA.route_cues?.[candidateId]).filter(Boolean);
}}
function routeLogisticsForOuting(outing) {{
  const cues = routeCuesForOuting(outing);
  const carPasses = [];
  const knownWater = [];
  cues.forEach(cue => {{
    const logistics = cue.logistics || {{}};
    (logistics.car_passes || []).forEach(item => carPasses.push(item));
    (logistics.known_water || []).forEach(item => knownWater.push(item));
  }});
  const trailheads = [...new Set(cues.map(cue => cue.trailhead?.name).filter(Boolean))];
  if (cues.length > 1 && trailheads.length === 1) {{
    carPasses.unshift({{
      name:"Back at car between route components",
      inter_component:true,
      mile_from_start:null,
      distance_to_car_miles:0,
    }});
  }}
  return {{ carPasses, knownWater }};
}}
function carPassText(items) {{
  if (!items.length) return "No mid-route car pass detected.";
  return items.map(item => item.inter_component
    ? "Back at car between route components."
    : `Pass by car again near mile ${{formatMiles(item.mile_from_start)}}.`
  ).join(" ");
}}
function waterText(items) {{
  if (!items.length) return "";
  return items.map(item => `${{item.name || "Water"}} · ${{item.location || "location"}} · ${{item.confidence || "verified"}}`).join("; ");
}}
function waterLineHtml(items) {{
  const text = waterText(items);
  return text ? `<br><b>Known water:</b> ${{esc(text)}}` : "";
}}
function classForSegment(segment) {{
  return segment.direction_rule === "ascent" ? "cue-row ascent" : "cue-row";
}}
function segmentCueHtml(segment) {{
  const minutes = segment.estimated_moving_minutes ? ` · est. ${{segment.estimated_moving_minutes}} min` : "";
  const p75 = segment.estimated_moving_minutes_p75 ? ` / p75 ${{segment.estimated_moving_minutes_p75}}` : "";
  const ascent = segment.ascent_ft ? ` · ${{Math.round(Number(segment.ascent_ft))}} ft climb` : "";
  const signpost = segment.signpost_label ? ` · Signpost: ${{segment.signpost_label}}` : "";
  return `<div class="${{classForSegment(segment)}}"><i>${{esc(segment.order)}}</i><div><b>${{esc(segment.segment_name || segment.trail_name)}}</b><span>${{esc(formatMiles(segment.official_miles))}} official mi · ${{esc(segment.direction_cue || "Follow map arrows.")}}${{esc(signpost)}}${{esc(minutes)}}${{esc(p75)}}${{esc(ascent)}}</span></div></div>`;
}}
function connectorText(link) {{
  const names = (link.connector_names || []).slice(0, 4).join(", ");
  const signs = (link.signpost_labels || []).slice(0, 4).join("; ");
  const classes = (link.connector_classes || []).join(", ");
  const nameText = names ? ` via ${{names}}` : "";
  const signText = signs ? ` · Signpost cues: ${{signs}}` : "";
  const classText = classes ? ` (${{classes}})` : "";
  return `${{link.from_trail || "route"}} to ${{link.to_trail || "next trail"}}: ${{formatMiles(link.distance_miles)}} mi${{nameText}}${{signText}}${{classText}}`;
}}
function routeCueHtml(cue, cueIndex, totalCues) {{
  const trailhead = cue.trailhead || {{}};
  const access = cue.start_access || {{}};
  const returnToCar = cue.return_to_car || {{}};
  const logistics = cue.logistics || {{}};
  const returnNames = (returnToCar.connector_names || []).slice(0, 6).join(", ");
  const titlePrefix = totalCues > 1 ? `<h2>Route component ${{cueIndex + 1}}</h2>` : "";
  const waterHtml = waterLineHtml(logistics.known_water || []);
  const between = (cue.between_links || []).length
    ? `<div class="route-card-section"><h2>Connector moves</h2><p>${{(cue.between_links || []).map(connectorText).map(esc).join("<br>")}}</p></div>`
    : "";
  return `<div class="route-card-section">${{titlePrefix}}<h2>Park and start</h2><p><b>${{esc(trailhead.name || "Parking TBD")}}</b><br>${{esc(parkingDetailsText(trailhead))}}<br>${{esc(startAccessText(access))}}</p></div><div class="route-card-section"><h2>Car / water access</h2><p><b>Car:</b> ${{esc(carPassText(logistics.car_passes || []))}}${{waterHtml}}</p></div><div class="route-card-section"><h2>Route checkpoints</h2><div class="cue-list">${{(cue.segments || []).map(segmentCueHtml).join("")}}</div></div>${{between}}<div class="route-card-section"><h2>Return to car</h2><p>${{esc(returnToCar.description || "Follow mapped route back to parking.")}}<br>Repeat official: ${{esc(formatMiles(returnToCar.official_repeat_miles))}} mi · connector: ${{esc(formatMiles(returnToCar.connector_miles))}} mi · road: ${{esc(formatMiles(returnToCar.road_miles))}} mi${{returnNames ? `<br>Return via: ${{esc(returnNames)}}` : ""}}</p></div>`;
}}
function selectedHtml(outing) {{
  if (!outing) return "";
  const parks = parkingNamesForOuting(outing);
  const parkText = parks.length ? parks.map(shortParkingName).join(" + ") : outing.trailhead;
  const cues = routeCuesForOuting(outing);
  const logistics = routeLogisticsForOuting(outing);
  const waterHtml = waterLineHtml(logistics.knownWater);
  const routeName = outing.route_name || outing.label;
  const codeText = outing.label && outing.label !== routeName ? ` · ${{outing.label}}` : "";
  const pairNote = outing.package_start_count > 1
    ? `<p><b>Related package:</b> ${{esc(outing.package_number)}} has ${{esc(outing.package_start_count)}} starts. This card is only the selected parked-start outing.</p>`
    : "";
  const cueHtml = cues.length
    ? cues.map((cue, index) => routeCueHtml(cue, index, cues.length)).join("")
    : `<div class="route-card-section"><h2>Route cues</h2><p>No cue data found for this outing. Use the selected map line and GPX before running.</p></div>`;
  return `<div class="route-card"><div class="route-card-header"><span>Selected outing</span><strong>${{esc(routeName)}}</strong><em>${{esc(outing.trailhead)}}${{esc(codeText)}}</em></div><div class="route-card-body"><div class="route-card-grid"><div class="route-card-kv"><b>Door to door p75</b><span>${{esc(formatMinutes(outing.total_minutes))}}</span></div><div class="route-card-kv"><b>On foot</b><span>${{esc(formatMiles(outing.on_foot_miles))}} mi</span></div><div class="route-card-kv"><b>Official credit</b><span>${{esc(formatMiles(outing.official_miles))}} mi</span></div><div class="route-card-kv"><b>Segments left</b><span>${{esc(outing.remaining_segment_count)}} / ${{esc(outing.segment_ids.length)}}</span></div></div><div class="card-note"><b>Map:</b> selected route is isolated on the map. P marks parking/start. Arrows show travel direction. TURN markers show double-backs or return points. CAR marks a mid-route car pass; W marks verified water.</div><div class="route-card-section"><h2>Outing</h2><p><b>Route package:</b> ${{esc(outing.block_name)}}<br><b>Park/start:</b> ${{esc(parkText || "parking TBD")}}<br><b>Car access:</b> ${{esc(carPassText(logistics.carPasses))}}${{waterHtml}}<br><b>Trails:</b> ${{esc(outing.trails.join(", "))}}</p>${{pairNote}}</div>${{cueHtml}}</div></div>`;
}}
function selectedMapSummaryHtml(outing) {{
  if (!outing) return "";
  const parks = parkingNamesForOuting(outing);
  const parkText = parks.length ? parks.map(shortParkingName).join(" + ") : outing.trailhead;
  const logistics = routeLogisticsForOuting(outing);
  const waterHtml = waterLineHtml(logistics.knownWater);
  const routeName = outing.route_name || outing.label;
  const codeText = outing.label && outing.label !== routeName ? ` · ${{outing.label}}` : "";
  return `<span>Selected outing</span><strong>${{esc(routeName)}}</strong><p>${{esc(outing.trailhead)}}${{esc(codeText)}}<br><b>Park:</b> ${{esc(parkText || "parking TBD")}}<br><b>Door to door p75:</b> ${{esc(formatMinutes(outing.total_minutes))}} · <b>On foot:</b> ${{esc(formatMiles(outing.on_foot_miles))}} mi<br><b>Official:</b> ${{esc(formatMiles(outing.official_miles))}} mi · <b>Segments:</b> ${{esc(outing.remaining_segment_count)}} / ${{esc(outing.segment_ids.length)}}<br><b>Car access:</b> ${{esc(carPassText(logistics.carPasses))}}${{waterHtml}}<br>Follow the selected line arrows. P marks parking/start.</p>`;
}}
function updateSelection(outingId) {{
  const outing = outingId ? outingById(outingId) : null;
  const panel = document.getElementById("selectedPanel");
  const summary = document.getElementById("mapSummary");
  panel.innerHTML = outing ? selectedHtml(outing) : "";
  summary.innerHTML = outing ? selectedMapSummaryHtml(outing) : "";
  panel.classList.toggle("active", Boolean(outing));
  summary.classList.toggle("active", Boolean(outing));
  document.querySelectorAll(".package").forEach(el => el.classList.toggle("active", Boolean(outingId) && String(el.dataset.id) === String(outingId)));
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
function drawLogisticsMarkers(collection, filterFn) {{
  (collection.features || []).filter(filterFn).forEach(feature => {{
    const geometry = feature.geometry || {{}};
    const coords = geometry.coordinates || [];
    if (geometry.type !== "Point" || coords.length < 2) return;
    L.marker([coords[1], coords[0]], {{ icon:logisticsIcon(feature.properties.kind) }})
      .bindPopup(popup(feature.properties || {{}}))
      .bindTooltip(feature.properties.kind === "water" ? "Water" : "Pass by car", {{ permanent:true, direction:"top", offset:[0,-14], className:"parking-label" }})
      .addTo(logisticsLayer);
  }});
}}
function draw(outingId=null) {{
  const outing = outingId ? outingById(outingId) : null;
  const selected = Boolean(outing);
  routeLayer.clearLayers(); officialLayer.clearLayers(); parkingLayer.clearLayers(); logisticsLayer.clearLayers(); arrowLayer.clearLayers();
  drawRouteLines(DATA.feature_collections.routes, f => selected ? featureInOuting(f, outing) : featureInCurrentMenu(f), selected);
  if (!selected) {{
    L.geoJSON(DATA.feature_collections.official_segments, {{
      filter:f => featureInCurrentMenu(f),
      style:f => ({{ color:f.properties.color, weight:6, opacity:.9 }}),
      onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))
    }}).addTo(officialLayer);
  }}
  L.geoJSON(DATA.feature_collections.parking || {{ type:"FeatureCollection", features:[] }}, {{
    filter:f => selected ? featureInOuting(f, outing) : featureInCurrentMenu(f),
    pointToLayer:(f,latlng)=>L.marker(latlng, {{ icon:parkingIcon() }}),
    onEachFeature:(f,l)=>{{
      l.bindPopup(popup(f.properties));
      if (outing) l.bindTooltip(shortParkingName(f.properties.name || f.properties.trailhead), {{ permanent:true, direction:"top", offset:[0,-15], className:"parking-label" }});
    }}
  }}).addTo(parkingLayer);
  drawLogisticsMarkers(DATA.feature_collections.logistics || {{ type:"FeatureCollection", features:[] }}, f => selected ? featureInOuting(f, outing) : featureInCurrentMenu(f));
  drawDirectionArrows(DATA.feature_collections.routes, f => selected ? featureInOuting(f, outing) : featureInCurrentMenu(f), selected);
  if (selected) drawRouteCues(DATA.feature_collections.routes, f => featureInOuting(f, outing));
  updateSelection(outingId);
  fit();
}}
function fit() {{
  const layers=[]; routeLayer.eachLayer(l=>layers.push(l)); officialLayer.eachLayer(l=>layers.push(l)); parkingLayer.eachLayer(l=>layers.push(l)); logisticsLayer.eachLayer(l=>layers.push(l));
  if (layers.length) {{
    const bounds = L.featureGroup(layers).getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, {{ padding:[24,24], maxZoom:14 }});
  }}
}}
function parkedStarts(package) {{
  const groups = [];
  (package.components || []).forEach(component => {{
    const trailhead = shortParkingName(component.trailhead || "parking TBD");
    const groupKey = String(component.field_menu_group_id || trailhead);
    let group = groups.find(item => item.groupKey === groupKey);
    if (!group) {{
      group = {{ groupKey, trailhead, labelOverride: component.field_menu_label || null, routeNames:[], onFoot:0, official:0, minutes:0, trails:[], candidateIds:[], segmentIds:[] }};
      groups.push(group);
    }}
    group.onFoot += Number(component.on_foot_miles || 0);
    group.official += Number(component.official_miles || 0);
    group.minutes += Number(component.total_minutes || 0);
    if (component.candidate_id) group.candidateIds.push(String(component.candidate_id));
    (component.segment_ids || []).forEach(segmentId => {{
      const value = String(segmentId);
      if (!group.segmentIds.includes(value)) group.segmentIds.push(value);
    }});
    (component.trail_names || []).forEach(trail => {{
      if (!group.trails.includes(trail)) group.trails.push(trail);
    }});
    if (component.route_name && !group.routeNames.includes(component.route_name)) group.routeNames.push(component.route_name);
  }});
  return groups;
}}
function buildOutings() {{
  const outings = [];
  const completed = new Set((DATA.progress?.completed_segment_ids || []).map(String));
  (DATA.packages || []).forEach(package => {{
    const starts = parkedStarts(package);
    starts.forEach((start, index) => {{
      const segmentIds = [...new Set((start.segmentIds || []).map(String))];
      const remainingSegmentIds = segmentIds.filter(segId => !completed.has(segId));
      const manualArea = manualDesignAreaForCandidateIds(package.package_number, start.candidateIds);
      const routeCode = start.labelOverride || `${{package.package_number}}${{starts.length > 1 ? String.fromCharCode(65 + index) : ""}}`;
      const routeName = start.routeNames[0] || start.trails[0] || start.trailhead || routeCode;
      outings.push({{
        outing_id: `${{package.package_number}}-${{index + 1}}`,
        label: routeCode,
        route_code: routeCode,
        route_name: routeName,
        package_number: package.package_number,
        package_start_count: starts.length,
        block_name: package.block_name,
        trailhead: start.trailhead,
        trails: start.trails,
        official_miles: start.official,
        on_foot_miles: start.onFoot,
        total_minutes: start.minutes,
        candidate_ids: start.candidateIds,
        segment_ids: segmentIds,
        remaining_segment_ids: remainingSegmentIds,
        remaining_segment_count: remainingSegmentIds.length,
        completed: segmentIds.length > 0 && remainingSegmentIds.length === 0,
        manual_design_hold: Boolean(manualArea),
        manual_design_area_id: manualArea?.area_id || null,
        manual_design_status: manualArea?.status || null,
        manual_design_decision: manualArea?.decision || null,
      }});
    }});
  }});
  return outings;
}}
DATA.outings = buildOutings();
let activeTimeFilter = "all";
function remainingSegmentCount() {{
  return new Set(DATA.outings.flatMap(outing => outing.remaining_segment_ids || [])).size;
}}
function openRunnableOutings() {{
  return DATA.outings.filter(outing => !outing.completed && !outing.manual_design_hold);
}}
function heldManualOutings() {{
  return DATA.outings.filter(outing => !outing.completed && outing.manual_design_hold);
}}
document.getElementById("metrics").innerHTML = [
  ["Open outings", openRunnableOutings().length],
  ["Manual holds", heldManualOutings().length],
  ["Remaining segs", remainingSegmentCount()],
  ["On-foot", `${{DATA.summary.total_on_foot_miles}} mi`]
].map(([a,b])=>`<div class="metric"><span>${{a}}</span><strong>${{b}}</strong></div>`).join("");
function outingListHtml(outing) {{
  const pairText = outing.package_start_count > 1 ? ` · package ${{outing.package_number}} has ${{outing.package_start_count}} starts` : "";
  const progressText = outing.remaining_segment_count === outing.segment_ids.length ? "" : ` · ${{outing.remaining_segment_count}}/${{outing.segment_ids.length}} segments left`;
  const logistics = routeLogisticsForOuting(outing);
  const carText = logistics.carPasses.length ? " · car access" : "";
  const waterBadge = logistics.knownWater.length ? " · water" : "";
  const routeName = outing.route_name || outing.label;
  const codeText = outing.label && outing.label !== routeName ? ` · ${{outing.label}}` : "";
  return `<div class="package" data-id="${{esc(outing.outing_id)}}"><strong>${{esc(routeName)}}</strong><span class="package-meta">${{esc(outing.trailhead)}}${{esc(codeText)}} · ${{esc(formatMinutes(outing.total_minutes))}} door-to-door · ${{esc(formatMiles(outing.official_miles))}} official · ${{esc(formatMiles(outing.on_foot_miles))}} on foot${{esc(progressText)}}${{esc(pairText)}}${{esc(carText)}}${{esc(waterBadge)}}</span><div class="start-list"><div class="start-row"><span>${{esc(outing.block_name)}}</span><em>package</em></div><div class="start-row"><span>${{esc(outing.trails.join(", "))}}</span><em>trails</em></div></div></div>`;
}}
function outingMatchesFilter(outing) {{
  if (outing.completed) return false;
  if (outing.manual_design_hold) return false;
  const minutes = Number(outing.total_minutes || 0);
  if (activeTimeFilter === "all") return true;
  if (activeTimeFilter === "over240") return minutes > 240;
  return minutes <= Number(activeTimeFilter);
}}
function manualDesignAreaHtml(area) {{
  const placeholder = area.current_placeholder || {{}};
  const splitProbe = area.default_split_probe || null;
  const splitProbeHtml = splitProbe && splitProbe.alternative_ids?.length
    ? `<span>Best current split probe: ${{esc(splitProbe.alternative_ids.join(" + "))}} · ${{esc(formatMiles(splitProbe.on_foot_miles))}} mi on foot · improves by ${{esc(formatMiles(splitProbe.improvement_vs_current_on_foot_miles))}} mi · accepted=${{esc(splitProbe.passes_probe_acceptance)}}</span>`
    : "";
  const alternatives = (area.alternatives || []).slice(0, 3).map(alt => {{
    const target = Array.isArray(alt.target_on_foot_miles_range) && alt.target_on_foot_miles_range.length === 2
      ? `${{formatMiles(alt.target_on_foot_miles_range[0])}}-${{formatMiles(alt.target_on_foot_miles_range[1])}} mi`
      : "mileage TBD";
    const probe = alt.probe && !alt.probe.error
      ? ` · probe ${{formatMiles(alt.probe.on_foot_miles)}} mi, ${{alt.probe.route_status}}`
      : "";
    return `<li>${{esc(alt.alternative_id)}}: ${{esc(alt.title)}} · ${{esc(alt.status)}} · target ${{esc(target)}}${{esc(probe)}}</li>`;
  }}).join("");
  return `<div class="manual-card"><strong>${{esc(area.title || area.area_id)}}</strong><span>${{esc(area.decision || "Manual route design required before scheduling.")}}</span><span>Held placeholder: ${{esc(placeholder.label || "hold")}} from ${{esc(placeholder.trailhead || "unknown")}} · ${{esc(formatMinutes(placeholder.door_to_door_minutes))}} · ${{esc(formatMiles(placeholder.on_foot_miles))}} mi on foot for ${{esc(formatMiles(placeholder.official_miles))}} official.</span>${{splitProbeHtml}}${{alternatives ? `<ul>${{alternatives}}</ul>` : ""}}</div>`;
}}
function renderManualDesignAreas() {{
  const heldAreaIds = new Set(heldManualOutings().map(outing => outing.manual_design_area_id).filter(Boolean));
  const areas = manualDesignAreas().filter(area => heldAreaIds.has(area.area_id));
  document.getElementById("manualDesignAreas").innerHTML = areas.length ? areas.map(manualDesignAreaHtml).join("") : "";
}}
function renderOutingList() {{
  const visible = DATA.outings.filter(outingMatchesFilter);
  document.getElementById("packages").innerHTML = visible.length
    ? visible.map(outingListHtml).join("")
    : '<div class="package"><strong>No outings in this time bucket</strong><span class="package-meta">Try a larger time window.</span></div>';
  document.querySelectorAll(".package[data-id]").forEach(el => el.addEventListener("click", () => draw(el.dataset.id)));
}}
renderManualDesignAreas();
document.querySelectorAll(".time-filter").forEach(button => button.addEventListener("click", () => {{
  activeTimeFilter = button.dataset.filter;
  document.querySelectorAll(".time-filter").forEach(item => item.classList.toggle("active", item === button));
  renderOutingList();
  draw();
}}));
renderOutingList();
document.getElementById("fitAll").addEventListener("click", () => draw());
draw();
</script>
</body>
</html>
"""


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-pass-json", type=Path, default=DEFAULT_ROUTE_PASS_JSON)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--blocks-json", type=Path, default=DEFAULT_BLOCKS_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument(
        "--write-map-html",
        action="store_true",
        help="Write a diagnostic package map HTML. The normal user-facing map is written by human_loop_plan.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_pass = read_json(args.route_pass_json)
    plan = read_json(args.plan_json)
    blocks_config = read_json(args.blocks_json)
    candidate_index = load_candidate_index(plan)
    candidate_index.update(route_pass.get("candidate_index") or {})
    package_pass = build_packages(route_pass, blocks_config, candidate_index)
    official_index = load_official_segment_index(args.official_geojson)
    official_segments, _meta = load_official_segments(args.official_geojson)
    connector_meta = ((plan.get("source_datasets") or {}).get("connector_geojson") or {})
    connector_path = Path(str(connector_meta.get("path"))) if connector_meta.get("path") else None
    connector_graph = (
        load_connector_graph(connector_path, official_segments=official_segments)
        if connector_path and connector_path.exists()
        else None
    )
    map_data = build_map_data(package_pass, route_pass, plan, official_index, connector_graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    map_json_path = args.output_dir / f"{args.basename}-map-data.json"
    map_html_path = args.output_dir / f"{args.basename}-map.html"
    manifest_path = args.output_dir / f"{args.basename}-artifact-manifest.json"
    write_json(json_path, package_pass)
    md_path.write_text(render_markdown(package_pass), encoding="utf-8")
    write_json(map_json_path, map_data)
    outputs = [json_path, md_path, map_json_path]
    if args.write_map_html:
        map_html_path.write_text(render_html(map_data), encoding="utf-8")
        outputs.append(map_html_path)
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=args.basename,
            inputs=[args.route_pass_json, args.plan_json, args.blocks_json, args.official_geojson],
            outputs=outputs,
            command="block_day_packager.py",
            metadata={
                "package_count": package_pass["summary"]["package_count"],
                "component_route_count": package_pass["summary"]["component_route_count"],
                "rendered_map_passed": map_data["map_validation"]["rendered_passed"],
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {map_json_path}")
    if args.write_map_html:
        print(f"Wrote {map_html_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
