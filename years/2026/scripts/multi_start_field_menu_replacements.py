#!/usr/bin/env python3
"""Promote accepted multi-start alternatives into field-menu replacements.

The public base field-menu rule file must not carry private Strava-derived
parking coordinates. This script merges the public route rules with selected
private multi-start replacements and writes an ignored private replacement
source that can be passed to human_loop_plan.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from export_execution_gpx import (  # noqa: E402
    candidate_track_coordinates,
    load_official_segment_index,
    validate_track_segments,
)
from export_mobile_field_packet import densify_track_segments  # noqa: E402
from multi_start_alternative_audit import (  # noqa: E402
    DEFAULT_PRIVATE_OUTPUT_DIR,
    DEFAULT_STATE_JSON,
    best_candidate_for_component,
    display_path,
    segment_trail_order,
)
from p90_relaxed_drive_day_gpx_export import stitch_remaining_track_gaps  # noqa: E402
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
    haversine_miles,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    read_json,
    round_miles,
)


DEFAULT_BASE_OVERRIDES_JSON = YEAR_DIR / "inputs" / "personal" / "2026-field-menu-overrides-v1.json"
DEFAULT_CURRENT_MAP_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_MULTI_START_AUDIT_JSON = (
    DEFAULT_PRIVATE_OUTPUT_DIR / "multi-start-alternative-audit-2026-05-08.json"
)
DEFAULT_OUTPUT_JSON = (
    YEAR_DIR
    / "inputs"
    / "personal"
    / "private"
    / "2026-field-menu-replacements-v2-multi-start.private.json"
)
DEFAULT_SEGMENT_PROMOTIONS_JSON = (
    YEAR_DIR / "inputs" / "personal" / "2026-cross-package-segment-promotions-v1.json"
)
DEFAULT_MANIFEST_JSON = (
    YEAR_DIR
    / "inputs"
    / "personal"
    / "private"
    / "2026-field-menu-replacements-v2-multi-start-artifact-manifest.json"
)

OVERRIDE_SOURCE = "multi_start_field_menu_replacement"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def selected_alternatives_from_audit(audit: dict[str, Any]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for outing in audit.get("outings") or []:
        alternatives = [
            alternative
            for alternative in outing.get("alternatives") or []
            if alternative.get("status") == "promising"
            and not alternative.get("parking_blockers")
            and alternative.get("alternative_id")
        ]
        if not alternatives:
            continue
        alternatives.sort(
            key=lambda item: (
                -float(item.get("on_foot_savings_miles") or 0.0),
                int(item.get("elapsed_delta_minutes") or 0),
                str(item.get("alternative_id")),
            )
        )
        selected[str(outing.get("label"))] = str(alternatives[0]["alternative_id"])
    return selected


def selected_alternatives_from_arg(raw_values: list[str] | None, audit: dict[str, Any]) -> dict[str, str]:
    if not raw_values:
        return selected_alternatives_from_audit(audit)
    selected: dict[str, str] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Expected LABEL=ALTERNATIVE_ID, got {raw!r}")
        label, alternative_id = raw.split("=", 1)
        selected[label.strip()] = alternative_id.strip()
    return selected


def index_audit_outings(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(outing.get("label")): outing for outing in audit.get("outings") or []}


def package_for_number(map_data: dict[str, Any], package_number: Any) -> dict[str, Any]:
    for package in map_data.get("packages") or []:
        if str(package.get("package_number")) == str(package_number):
            return package
    raise KeyError(f"Package {package_number} not found in current map data")


def component_for_candidate(package: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for component in package.get("components") or []:
        if str(component.get("candidate_id")) == str(candidate_id):
            return component
    raise KeyError(f"Candidate {candidate_id} not found in package {package.get('package_number')}")


def package_contains_candidate(package: dict[str, Any], candidate_id: str) -> bool:
    return any(
        str(component.get("candidate_id")) == str(candidate_id)
        for component in package.get("components") or []
    )


def generated_components_for_alternative(
    package: dict[str, Any],
    alternative_id: str,
) -> list[dict[str, Any]]:
    return [
        component
        for component in package.get("components") or []
        if str(component.get("source") or "") == OVERRIDE_SOURCE
        and str(component.get("multi_start_alternative_id") or "") == alternative_id
    ]


def route_number_base_from_generated_components(components: list[dict[str, Any]]) -> int:
    route_numbers = sorted(
        int(component.get("route_number"))
        for component in components
        if str(component.get("route_number") or "").isdigit()
    )
    if not route_numbers:
        return 0
    first = route_numbers[0]
    return first // 10 if first >= 10 else first


def fallback_replacement_packages(base_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry["replace_package"]
        for entry in base_overrides.get("overrides") or []
        if isinstance(entry.get("replace_package"), dict)
    ]


def package_source_for_replacement(
    *,
    current_map: dict[str, Any],
    fallback_packages: list[dict[str, Any]],
    package_number: Any,
    baseline_candidate_id: str,
    alternative_id: str,
) -> tuple[dict[str, Any], str]:
    try:
        current_package = package_for_number(current_map, package_number)
    except KeyError:
        current_package = None
    if current_package and package_contains_candidate(current_package, baseline_candidate_id):
        return current_package, "baseline"
    for package in fallback_packages:
        if str(package.get("package_number")) != str(package_number):
            continue
        if package_contains_candidate(package, baseline_candidate_id):
            return package, "baseline"
        if generated_components_for_alternative(package, alternative_id):
            return package, "already_replaced"
    if current_package and generated_components_for_alternative(current_package, alternative_id):
        return current_package, "already_replaced"
    if current_package:
        component_for_candidate(current_package, baseline_candidate_id)
        return current_package, "baseline"
    raise KeyError(f"Package {package_number} not found in current map data or fallback overrides")


def candidate_id_for_component(label: str, alternative_id: str, index: int, candidate: dict[str, Any]) -> str:
    return slugify(f"multi-start-{label}-{alternative_id}-{index}-{candidate.get('candidate_id')}")


def field_menu_label(base_label: str, component_count: int, index: int) -> str | None:
    if component_count <= 1:
        return None
    if re.fullmatch(r"\d+[A-Z]", base_label):
        return f"{base_label}-{index}"
    return None


def label_kept_components_from_original(package: dict[str, Any], kept_components: list[dict[str, Any]]) -> None:
    try:
        package_number = int(package.get("package_number"))
    except (TypeError, ValueError):
        return
    original_index_by_candidate = {
        str(component.get("candidate_id")): index
        for index, component in enumerate(package.get("components") or [])
        if component.get("candidate_id")
    }
    for fallback_index, component in enumerate(kept_components):
        if not component.get("field_menu_label"):
            original_index = original_index_by_candidate.get(str(component.get("candidate_id")), fallback_index)
            component["field_menu_label"] = f"{package_number}{chr(65 + original_index)}"
        if not component.get("field_menu_group_id"):
            component["field_menu_group_id"] = str(component.get("candidate_id"))


def component_from_candidate(
    *,
    candidate_id: str,
    candidate: dict[str, Any],
    base_component: dict[str, Any],
    label: str,
    alternative_id: str,
    component_count: int,
    component_index: int,
) -> dict[str, Any]:
    official = float(candidate.get("official_new_miles") or 0.0)
    on_foot = float(candidate.get("estimated_total_on_foot_miles") or 0.0)
    generated_label = field_menu_label(label, component_count, component_index)
    component = {
        "route_number": int(base_component.get("route_number") or 0) * 10 + component_index,
        "candidate_id": candidate_id,
        "field_menu_group_id": candidate_id,
        "trail_names": candidate.get("trail_names") or [],
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "total_minutes": int(candidate.get("total_minutes") or 0),
        "raw_total_minutes": int(candidate.get("raw_total_minutes") or 0),
        "trailhead": (candidate.get("trailhead") or {}).get("name") or "Parking/start",
        "less_optimal_flags": list(candidate.get("less_optimal_flags") or []),
        "segment_ids": [int(value) for value in candidate.get("segment_ids") or []],
        "time_breakdown_minutes": copy.deepcopy(candidate.get("time_breakdown_minutes")),
        "time_estimates_minutes": copy.deepcopy(candidate.get("time_estimates_minutes")),
        "effort": copy.deepcopy(candidate.get("effort")),
        "route_status": candidate.get("route_status"),
        "multi_start_alternative_id": alternative_id,
        "source": OVERRIDE_SOURCE,
    }
    if generated_label:
        component["field_menu_label"] = generated_label
    return component


def cue_from_candidate(candidate_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    start_access = candidate.get("trailhead_access") or {}
    planned_directions = (
        (candidate.get("direction_validation") or {}).get("planned_traversal_direction")
        or {}
    )
    segments = []
    for raw_segment in candidate.get("segments") or []:
        segment = copy.deepcopy(raw_segment)
        direction = str(segment.get("direction") or segment.get("direction_rule") or "both")
        segment["direction_rule"] = direction
        if direction == "ascent":
            planned_direction = planned_directions.get(str(segment.get("seg_id")))
            if planned_direction == "official_geometry_end_to_start":
                segment["direction_cue"] = "Ascent-only segment; valid uphill traversal is opposite official geometry direction."
            elif planned_direction == "official_geometry_start_to_end":
                segment["direction_cue"] = "Ascent-only segment; valid uphill traversal follows official geometry direction."
            else:
                segment["direction_cue"] = "Ascent-only segment; verify signed uphill direction."
        else:
            segment.setdefault("direction_cue", "Either direction allowed; follow map arrows.")
        segments.append(segment)
    return {
        "candidate_id": candidate_id,
        "title": ", ".join(candidate.get("trail_names") or []),
        "route_status": candidate.get("route_status"),
        "official_miles": candidate.get("official_new_miles"),
        "on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
        "raw_total_minutes": candidate.get("raw_total_minutes"),
        "total_minutes": candidate.get("total_minutes"),
        "time_breakdown_minutes": copy.deepcopy(candidate.get("time_breakdown_minutes")),
        "time_estimates_minutes": copy.deepcopy(candidate.get("time_estimates_minutes")),
        "effort": copy.deepcopy(candidate.get("effort")),
        "trailhead": copy.deepcopy(candidate.get("trailhead")),
        "start_access": {
            "confidence": start_access.get("snap_confidence")
            or (candidate.get("validation") or {}).get("trailhead_snap_confidence"),
            "direct_gap_miles": start_access.get("direct_gap_miles"),
            "mapped_access_miles": start_access.get("one_way_miles"),
            "official_repeat_miles": start_access.get("one_way_official_repeat_miles")
            or start_access.get("official_repeat_miles")
            or 0,
            "official_repeat_segment_ids": copy.deepcopy(start_access.get("official_repeat_segment_ids") or []),
            "access_class": (candidate.get("validation") or {}).get("trailhead_snap", {}).get("access_class"),
            "graph_validated": bool(start_access.get("graph_validated")),
            "connector_names": copy.deepcopy(start_access.get("connector_names") or []),
            "connector_classes": copy.deepcopy(start_access.get("connector_classes") or []),
        },
        "segments": segments,
        "between_links": copy.deepcopy((candidate.get("between_trail_links") or {}).get("links") or []),
        "return_to_car": copy.deepcopy(candidate.get("return_to_car") or {}),
        "validation": copy.deepcopy(candidate.get("validation") or {}),
        "direction_validation": copy.deepcopy(candidate.get("direction_validation") or {}),
    }


def coords_feature(coords: list[tuple[float, float]], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon), float(lat)] for lon, lat in coords],
        },
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
            "has_parking": trailhead.get("has_parking"),
            "has_restroom": trailhead.get("has_restroom"),
            "has_water": trailhead.get("has_water"),
            "water_confidence": trailhead.get("water_confidence"),
            "parking_minutes": trailhead.get("parking_minutes"),
            "source": trailhead.get("source"),
            "parking_confidence": trailhead.get("parking_confidence"),
        },
    }


def track_for_candidate(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any],
    max_gap_miles: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    raw = candidate_track_coordinates(
        candidate,
        official_index,
        connector_graph=connector_graph,
        densify_source_lines=True,
    )
    stitched = stitch_remaining_track_gaps(
        raw,
        connector_graph=connector_graph,
        max_gap_miles=max_gap_miles,
    )
    dense = densify_track_segments([stitched], max_gap_miles=max_gap_miles)[0]
    validation = validate_track_segments([dense], max_gap_miles=max_gap_miles)
    return dense, validation


def route_properties(
    *,
    package: dict[str, Any],
    candidate_id: str,
    candidate: dict[str, Any],
    label: str,
    alternative_id: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "route",
        "package_number": package.get("package_number"),
        "candidate_id": candidate_id,
        "block_name": package.get("block_name"),
        "title": ", ".join(candidate.get("trail_names") or []),
        "official_miles": candidate.get("official_new_miles"),
        "on_foot_miles": candidate.get("estimated_total_on_foot_miles"),
        "trailhead": (candidate.get("trailhead") or {}).get("name"),
        "field_menu_label": field_menu_label(label, 2, 1),
        "multi_start_alternative_id": alternative_id,
        "source": OVERRIDE_SOURCE,
        "source_gap_warning_count": 0 if validation.get("passed") else 1,
    }


def recompute_package(package: dict[str, Any], *, add_multi_start_reasons: bool = True) -> dict[str, Any]:
    result = copy.deepcopy(package)
    components = result.get("components") or []
    official = sum(float(component.get("official_miles") or 0.0) for component in components)
    on_foot = sum(float(component.get("on_foot_miles") or 0.0) for component in components)
    trailheads = sorted({str(component.get("trailhead")) for component in components if component.get("trailhead")})
    result["component_route_count"] = len(components)
    result["component_candidate_ids"] = [component.get("candidate_id") for component in components]
    result["trail_names"] = sorted({trail for component in components for trail in component.get("trail_names") or []})
    result["official_miles"] = round_miles(official)
    result["on_foot_miles"] = round_miles(on_foot)
    result["ratio"] = round(on_foot / official, 2) if official else None
    result["trailheads"] = trailheads
    result["trailhead_count"] = len(trailheads)
    result["primary_trailhead"] = trailheads[0] if len(trailheads) == 1 else None
    result["total_minutes_components"] = sum(int(component.get("total_minutes") or 0) for component in components)
    result["component_routes_under_1_official_mile"] = [
        component.get("candidate_id")
        for component in components
        if float(component.get("official_miles") or 0.0) < 1.0
    ]
    result["component_routes_under_2_official_miles"] = [
        component.get("candidate_id")
        for component in components
        if float(component.get("official_miles") or 0.0) < 2.0
    ]
    if add_multi_start_reasons:
        reasons = list(result.get("planning_reasons") or [])
        for reason in ["accepted_multi_start_split", "mid_route_car_bailout_or_refill"]:
            if reason not in reasons:
                reasons.append(reason)
        result["planning_status"] = "accepted_multi_start_split"
        result["planning_reasons"] = reasons
    return result


def official_segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment["seg_id"]): segment for segment in official_segments}


def route_cue_index(map_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate_id): cue
        for candidate_id, cue in (map_data.get("route_cues") or {}).items()
        if isinstance(cue, dict)
    }


def candidate_ids_for_package(package: dict[str, Any]) -> list[str]:
    return [
        str(component.get("candidate_id"))
        for component in package.get("components") or []
        if component.get("candidate_id")
    ]


def feature_collections_for_candidates(
    current_map: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, Any]:
    candidate_set = set(candidate_ids)
    collections: dict[str, Any] = {}
    for collection_name, collection in (current_map.get("feature_collections") or {}).items():
        if collection_name == "official_segments":
            continue
        features = [
            copy.deepcopy(feature)
            for feature in collection.get("features") or []
            if str((feature.get("properties") or {}).get("candidate_id")) in candidate_set
        ]
        if features:
            collections[collection_name] = {"type": collection.get("type") or "FeatureCollection", "features": features}
    return collections


def route_validations_for_candidates(
    current_map: dict[str, Any],
    candidate_ids: list[str],
) -> list[dict[str, Any]]:
    candidate_set = set(candidate_ids)
    map_validation = current_map.get("map_validation") or {}
    route_validations = map_validation.get("route_validations") if isinstance(map_validation, dict) else []
    return [
        copy.deepcopy(validation)
        for validation in route_validations or []
        if str(validation.get("candidate_id")) in candidate_set
    ]


def baseline_replacement_entry(
    *,
    current_map: dict[str, Any],
    package_number: str,
    reason: str,
) -> dict[str, Any]:
    package = copy.deepcopy(package_for_number(current_map, package_number))
    candidate_ids = candidate_ids_for_package(package)
    cues = route_cue_index(current_map)
    return {
        "package_number": package.get("package_number"),
        "replacement_type": "segment_ownership_baseline_promotion",
        "reason": reason,
        "remove_candidate_ids": candidate_ids,
        "replace_package": recompute_package(package, add_multi_start_reasons=False),
        "route_cues": {
            candidate_id: copy.deepcopy(cues[candidate_id])
            for candidate_id in candidate_ids
            if candidate_id in cues
        },
        "feature_collections": feature_collections_for_candidates(current_map, candidate_ids),
        "route_validations": route_validations_for_candidates(current_map, candidate_ids),
    }


def replacement_entry_for_package(
    replacement_entries: list[dict[str, Any]],
    *,
    current_map: dict[str, Any],
    package_number: str,
    reason: str,
) -> dict[str, Any]:
    entry = next(
        (
            item
            for item in replacement_entries
            if str((item.get("replace_package") or {}).get("package_number")) == package_number
        ),
        None,
    )
    if entry:
        return entry
    entry = baseline_replacement_entry(
        current_map=current_map,
        package_number=package_number,
        reason=reason,
    )
    replacement_entries.append(entry)
    return entry


def entry_uses_multi_start_recompute(entry: dict[str, Any]) -> bool:
    return str(entry.get("replacement_type") or "") != "segment_ownership_baseline_promotion"


def recompute_entry_package(entry: dict[str, Any]) -> None:
    package = entry.get("replace_package") or {}
    package.update(
        recompute_package(
            package,
            add_multi_start_reasons=entry_uses_multi_start_recompute(entry),
        )
    )


def remove_candidate_from_replacement_entry(
    entry: dict[str, Any],
    candidate_id: str,
    *,
    reason: str | None = None,
) -> None:
    replacement_package = entry.get("replace_package") or {}
    components = replacement_package.get("components") or []
    replacement_package["components"] = [
        component
        for component in components
        if str(component.get("candidate_id")) != candidate_id
    ]
    route_cues = entry.get("route_cues") or {}
    route_cues.pop(candidate_id, None)
    entry["route_cues"] = route_cues
    for collection in (entry.get("feature_collections") or {}).values():
        features = collection.get("features")
        if not isinstance(features, list):
            continue
        collection["features"] = [
            feature
            for feature in features
            if str((feature.get("properties") or {}).get("candidate_id")) != candidate_id
        ]
    entry["route_validations"] = [
        validation
        for validation in entry.get("route_validations") or []
        if str(validation.get("candidate_id")) != candidate_id
    ]
    removed = entry.setdefault("segment_ownership_removed_candidate_ids", [])
    if candidate_id not in removed:
        removed.append(candidate_id)
    if reason:
        notes = entry.setdefault("segment_ownership_removal_notes", [])
        if reason not in notes:
            notes.append(reason)
    recompute_entry_package(entry)


def segment_row_from_source(
    *,
    segment_id: str,
    promotion: dict[str, Any],
    current_map: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
    target_cue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for cue in [target_cue] if target_cue else []:
        for segment in cue.get("segments") or []:
            if str(segment.get("seg_id")) == segment_id:
                return copy.deepcopy(segment)
    cues = route_cue_index(current_map)
    source_candidate_id = str(((promotion.get("from") or {}).get("candidate_id")) or "")
    candidate_ids = [source_candidate_id] if source_candidate_id else []
    candidate_ids.extend(candidate_id for candidate_id in sorted(cues) if candidate_id not in candidate_ids)
    for candidate_id in candidate_ids:
        cue = cues.get(candidate_id) or {}
        for segment in cue.get("segments") or []:
            if str(segment.get("seg_id")) == segment_id:
                return copy.deepcopy(segment)
    official = official_by_id.get(segment_id)
    if not official:
        raise ValueError(f"Segment {segment_id} was not found in official data or existing route cues")
    direction = str(official.get("direction") or "both")
    row = {
        "seg_id": int(segment_id) if segment_id.isdigit() else segment_id,
        "seg_name": official.get("seg_name"),
        "trail_name": official.get("trail_name"),
        "official_miles": round_miles(official.get("official_miles")),
        "direction": direction,
        "direction_rule": direction,
    }
    if direction == "ascent":
        row["direction_cue"] = "Ascent-only segment; verify signed uphill direction."
    else:
        row["direction_cue"] = "Either direction allowed; follow map arrows."
    return row


def insert_segment_row(
    segments: list[dict[str, Any]],
    segment_row: dict[str, Any],
    insert_after_segment_id: Any | None = None,
) -> list[dict[str, Any]]:
    segment_id = str(segment_row.get("seg_id"))
    filtered = [segment for segment in segments if str(segment.get("seg_id")) != segment_id]
    if insert_after_segment_id is None:
        return filtered + [segment_row]
    insert_after = str(insert_after_segment_id)
    result: list[dict[str, Any]] = []
    inserted = False
    for segment in filtered:
        result.append(segment)
        if str(segment.get("seg_id")) == insert_after:
            result.append(segment_row)
            inserted = True
    if not inserted:
        result.append(segment_row)
    return result


def promotion_evidence_passed(promotion: dict[str, Any], repo_root: Path) -> bool:
    evidence = promotion.get("evidence") or {}
    review_path_value = evidence.get("activity_review_json")
    segment_id = str(promotion.get("segment_id"))
    if review_path_value:
        review_path = Path(str(review_path_value))
        if not review_path.is_absolute():
            review_path = repo_root / review_path
        review = read_json(review_path)
        row = next(
            (item for item in review.get("segment_reviews") or [] if str(item.get("seg_id")) == segment_id),
            None,
        )
        if not row:
            return False
        if str(row.get("completion_status")) != str(evidence.get("required_status") or "completed"):
            return False
        if float(row.get("match_fraction") or 0.0) < float(evidence.get("min_match_fraction") or 0.85):
            return False
        if evidence.get("requires_endpoints_ok", True) and row.get("endpoints_ok") is not True:
            return False
        if evidence.get("requires_direction_ok", True) and row.get("direction_ok") is not True:
            return False
        return True

    latent_path_value = evidence.get("field_latent_credit_audit_json") or evidence.get("latent_credit_audit_json")
    if not latent_path_value:
        return False
    latent_path = Path(str(latent_path_value))
    if not latent_path.is_absolute():
        latent_path = repo_root / latent_path
    audit = read_json(latent_path)
    route_key = str(evidence.get("route_key") or "")
    outing_id = str(evidence.get("outing_id") or "")
    row = next(
        (
            route
            for route in audit.get("route_reviews") or []
            if (not route_key or str(route.get("route_key")) == route_key)
            and (not outing_id or str(route.get("outing_id")) == outing_id)
            and segment_id in {str(value) for value in route.get("latent_completed_segment_ids") or []}
        ),
        None,
    )
    if not row:
        return False
    if evidence.get("requires_audit_status_passed", True) and str(row.get("audit_status")) not in {
        "passed",
        "reconciled",
    }:
        return False
    segment_row = next(
        (item for item in row.get("segments") or [] if str(item.get("seg_id")) == segment_id),
        None,
    )
    if not segment_row:
        return False
    required_status = str(evidence.get("required_status") or "reconciled_owned_elsewhere")
    if str(segment_row.get("status")) != required_status:
        return False
    return True


def add_segment_to_component(component: dict[str, Any], segment_row: dict[str, Any]) -> None:
    segment_id = segment_row.get("seg_id")
    segment_ids = [str(value) for value in component.get("segment_ids") or []]
    already_present = str(segment_id) in segment_ids
    if str(segment_id) not in segment_ids:
        component.setdefault("segment_ids", []).append(int(segment_id) if str(segment_id).isdigit() else segment_id)
    official = float(component.get("official_miles") or 0.0)
    if not already_present:
        official += float(segment_row.get("official_miles") or 0.0)
    component["official_miles"] = round_miles(official)
    on_foot = float(component.get("on_foot_miles") or 0.0)
    component["ratio"] = round(on_foot / official, 2) if official else None
    trail_name = segment_row.get("trail_name")
    if trail_name and str(trail_name) not in [str(value) for value in component.get("trail_names") or []]:
        component.setdefault("trail_names", []).append(str(trail_name))


def update_cue_for_segment_promotion(
    cue: dict[str, Any],
    *,
    segment_row: dict[str, Any],
    insert_after_segment_id: Any | None,
    promotion: dict[str, Any],
) -> None:
    segment_id = str(segment_row.get("seg_id"))
    cue["segments"] = insert_segment_row(cue.get("segments") or [], segment_row, insert_after_segment_id)
    cue["official_miles"] = round_miles(
        sum(float(segment.get("official_miles") or 0.0) for segment in cue.get("segments") or [])
    )
    for leg_key in ["start_access", "return_to_car"]:
        leg = cue.get(leg_key)
        if not isinstance(leg, dict):
            continue
        leg["official_repeat_segment_ids"] = [
            segment
            for segment in leg.get("official_repeat_segment_ids") or []
            if str(segment) != segment_id
        ]
    for link in cue.get("between_links") or []:
        if not isinstance(link, dict):
            continue
        link["official_repeat_segment_ids"] = [
            segment
            for segment in link.get("official_repeat_segment_ids") or []
            if str(segment) != segment_id
        ]
    title_names = []
    for segment in cue.get("segments") or []:
        name = segment.get("trail_name")
        if name and str(name) not in title_names:
            title_names.append(str(name))
    if title_names:
        cue["title"] = ", ".join(title_names)
    notes = list(cue.get("segment_ownership_promotion_notes") or [])
    reason = promotion.get("reason")
    if reason and reason not in notes:
        notes.append(reason)
    cue["segment_ownership_promotion_notes"] = notes


def promotion_target_status(
    promotion: dict[str, Any],
    current_map: dict[str, Any],
    replacement_entries: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    target = promotion.get("to") or {}
    package_number = str(target.get("package_number"))
    candidate_id = str(target.get("candidate_id"))
    packages = list(current_map.get("packages") or [])
    packages.extend(
        copy.deepcopy(entry.get("replace_package"))
        for entry in replacement_entries
        if isinstance(entry.get("replace_package"), dict)
    )
    matching_packages = [
        item
        for item in packages
        if str(item.get("package_number")) == package_number
    ]
    if not matching_packages:
        return False, "target_package_not_in_current_map"
    if not any(
        str(component.get("candidate_id")) == candidate_id
        for package in matching_packages
        for component in package.get("components") or []
    ):
        return False, "target_candidate_not_in_current_package"
    return True, None


def apply_segment_ownership_promotions(
    replacement_entries: list[dict[str, Any]],
    promotions_payload: dict[str, Any],
    *,
    current_map: dict[str, Any],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    promotions = [
        promotion
        for promotion in promotions_payload.get("promotions") or []
        if promotion.get("status") == "promoted"
    ]
    if not promotions:
        return [], []
    official_by_id = official_segment_index(context["official_segments"])
    applied = []
    skipped = []
    for promotion in promotions:
        target_exists, skip_reason = promotion_target_status(promotion, current_map, replacement_entries)
        if not target_exists:
            skipped.append(
                {
                    "segment_id": promotion.get("segment_id"),
                    "status": "skipped",
                    "reason": skip_reason,
                    "from": copy.deepcopy(promotion.get("from") or {}),
                    "to": copy.deepcopy(promotion.get("to") or {}),
                }
            )
            continue
        if not promotion_evidence_passed(promotion, REPO_ROOT):
            raise ValueError(f"Promotion evidence failed for segment {promotion.get('segment_id')}")
        target = promotion.get("to") or {}
        package_number = str(target.get("package_number"))
        candidate_id = str(target.get("candidate_id"))
        entry = replacement_entry_for_package(
            replacement_entries,
            current_map=current_map,
            package_number=package_number,
            reason=f"Segment ownership promotion target for {candidate_id}.",
        )
        replacement_package = entry.get("replace_package") or {}
        component = next(
            (
                item
                for item in replacement_package.get("components") or []
                if str(item.get("candidate_id")) == candidate_id
            ),
            None,
        )
        if not component:
            raise ValueError(f"No component {candidate_id} found for promotion segment {promotion.get('segment_id')}")
        route_cue = (entry.get("route_cues") or {}).get(candidate_id)
        segment_id = str(promotion.get("segment_id"))
        segment_row = segment_row_from_source(
            segment_id=segment_id,
            promotion=promotion,
            current_map=current_map,
            official_by_id=official_by_id,
            target_cue=route_cue,
        )
        add_segment_to_component(component, segment_row)
        if route_cue:
            update_cue_for_segment_promotion(
                route_cue,
                segment_row=segment_row,
                insert_after_segment_id=target.get("insert_after_segment_id"),
                promotion=promotion,
            )
        recompute_entry_package(entry)
        for collection in (entry.get("feature_collections") or {}).values():
            for feature in collection.get("features") or []:
                props = feature.get("properties") or {}
                if str(props.get("candidate_id")) != candidate_id:
                    continue
                props["official_miles"] = component.get("official_miles")
                props["title"] = ", ".join(component.get("trail_names") or []) or props.get("title")
        source = promotion.get("from") or {}
        source_action = str(promotion.get("source_action") or source.get("source_action") or "keep")
        source_candidate_id = str(source.get("candidate_id") or "")
        source_package_number = str(source.get("package_number") or "")
        if source_action == "remove_route_card" and source_candidate_id and source_package_number:
            source_entry = replacement_entry_for_package(
                replacement_entries,
                current_map=current_map,
                package_number=source_package_number,
                reason=f"Segment ownership promotion source removal for {source_candidate_id}.",
            )
            remove_candidate_from_replacement_entry(
                source_entry,
                source_candidate_id,
                reason=promotion.get("reason"),
            )
        entry.setdefault("segment_ownership_promotions", []).append(copy.deepcopy(promotion))
        applied.append(copy.deepcopy(promotion))
    return applied, skipped


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    base_state = load_state(args.state_json)
    return {
        "official_segments": official_segments,
        "segments_by_id": {int(segment["seg_id"]): segment for segment in official_segments},
        "base_state": base_state,
        "performance_profile": build_performance_profile(
            state=base_state,
            strava_activity_details_dir=DEFAULT_STRAVA_DETAILS_DIR,
            activity_summary_csv=DEFAULT_ACTIVITY_SUMMARY_CSV,
            activity_detail_summary_csv=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
            segment_perf_csv=DEFAULT_SEGMENT_PERF_CSV,
        ),
        "connector_graph": load_connector_graph(args.connector_geojson, official_segments=official_segments),
        "elevation_sampler": load_dem_context(args.dem_tif, args.dem_summary_json)["sampler"],
        "official_index": load_official_segment_index(args.official_geojson),
    }


def find_alternative(outing: dict[str, Any], alternative_id: str) -> dict[str, Any]:
    for alternative in outing.get("alternatives") or []:
        if str(alternative.get("alternative_id")) == alternative_id:
            return alternative
    raise KeyError(f"Alternative {alternative_id} not found for outing {outing.get('label')}")


def build_replacement_entry(
    *,
    label: str,
    alternative_id: str,
    current_map: dict[str, Any],
    fallback_packages: list[dict[str, Any]],
    audit_outing: dict[str, Any],
    context: dict[str, Any],
    max_gap_miles: float,
) -> dict[str, Any]:
    alternative = find_alternative(audit_outing, alternative_id)
    baseline = audit_outing.get("baseline") or {}
    baseline_candidate_id = str(baseline.get("candidate_id"))
    package, package_source = package_source_for_replacement(
        current_map=current_map,
        fallback_packages=fallback_packages,
        package_number=baseline.get("package_number"),
        baseline_candidate_id=baseline_candidate_id,
        alternative_id=alternative_id,
    )
    if package_source == "already_replaced":
        existing_generated = generated_components_for_alternative(package, alternative_id)
        existing_generated_ids = {
            str(component.get("candidate_id"))
            for component in existing_generated
            if component.get("candidate_id")
        }
        base_component = {"route_number": route_number_base_from_generated_components(existing_generated)}
        kept_components = [
            copy.deepcopy(component)
            for component in package.get("components") or []
            if str(component.get("candidate_id")) not in existing_generated_ids
        ]
    else:
        base_component = component_for_candidate(package, baseline_candidate_id)
        kept_components = [
            copy.deepcopy(component)
            for component in package.get("components") or []
            if str(component.get("candidate_id")) != baseline_candidate_id
        ]
    preferred_order = segment_trail_order(
        [int(value) for value in baseline.get("segment_ids") or []],
        context["segments_by_id"],
    )
    replacement_components: list[dict[str, Any]] = []
    route_cues: dict[str, Any] = {}
    route_features: list[dict[str, Any]] = []
    parking_features: list[dict[str, Any]] = []
    route_validations: list[dict[str, Any]] = []
    component_count = len(alternative.get("components") or [])
    for index, component_summary in enumerate(alternative.get("components") or [], 1):
        candidate = best_candidate_for_component(
            segment_ids=[int(value) for value in component_summary.get("segment_ids") or []],
            anchor=component_summary.get("start_anchor") or {},
            base_state=context["base_state"],
            performance_profile=context["performance_profile"],
            connector_graph=context["connector_graph"],
            elevation_sampler=context["elevation_sampler"],
            segments_by_id=context["segments_by_id"],
            preferred_trail_order=preferred_order,
        )
        if not candidate:
            raise ValueError(f"Could not build candidate for {alternative_id} component {index}")
        candidate_id = candidate_id_for_component(label, alternative_id, index, candidate)
        coords, validation = track_for_candidate(
            candidate,
            context["official_index"],
            context["connector_graph"],
            max_gap_miles=max_gap_miles,
        )
        if not validation.get("passed"):
            raise ValueError(f"{candidate_id} GPX validation failed: {validation.get('failures')}")
        component = component_from_candidate(
            candidate_id=candidate_id,
            candidate=candidate,
            base_component=base_component,
            label=label,
            alternative_id=alternative_id,
            component_count=component_count,
            component_index=index,
        )
        replacement_components.append(component)
        route_cues[candidate_id] = cue_from_candidate(candidate_id, candidate)
        props = route_properties(
            package=package,
            candidate_id=candidate_id,
            candidate=candidate,
            label=label,
            alternative_id=alternative_id,
            validation=validation,
        )
        if "field_menu_label" in component:
            props["field_menu_label"] = component["field_menu_label"]
        route_features.append(coords_feature(coords, props))
        parking = parking_feature(candidate, props)
        if parking:
            parking_features.append(parking)
        route_validations.append(
            {
                "candidate_id": candidate_id,
                "source_gap_warning": False,
                "source_max_gap_miles": validation.get("max_trackpoint_gap_miles"),
                "rendered_passed": True,
                "rendered_failures": [],
            }
        )

    label_kept_components_from_original(package, kept_components)
    new_components = replacement_components + kept_components

    replacement_package = recompute_package({**package, "components": new_components})
    return {
        "package_number": package.get("package_number"),
        "reason": (
            f"Accepted multi-start alternative {alternative_id}: "
            f"{alternative.get('on_foot_savings_miles')} mi less on foot, "
            f"{alternative.get('elapsed_delta_minutes')} min p75 delta, "
            "with mid-route car access/bailout value."
        ),
        "remove_candidate_ids": [baseline_candidate_id],
        "replace_package": replacement_package,
        "route_cues": route_cues,
        "feature_collections": {
            "routes": {"type": "FeatureCollection", "features": route_features},
            "parking": {"type": "FeatureCollection", "features": parking_features},
        },
        "route_validations": route_validations,
    }


def output_summary(replacement_entries: list[dict[str, Any]]) -> dict[str, Any]:
    added = [
        entry
        for entry in replacement_entries
        if str(entry.get("reason") or "").startswith("Accepted multi-start")
    ]
    return {
        "multi_start_replacement_count": len(added),
        "packages_replaced": [entry.get("package_number") for entry in added],
        "new_candidate_count": sum(
            len((entry.get("replace_package") or {}).get("components") or [])
            for entry in added
        ),
    }


def build_replacement_payload(args: argparse.Namespace) -> dict[str, Any]:
    base = read_json(args.base_overrides_json) if args.base_overrides_json.exists() else {"overrides": []}
    existing_generated = read_json(args.output_json) if args.output_json.exists() else {"overrides": []}
    current_map = read_json(args.current_map_json)
    audit = read_json(args.multi_start_audit_json)
    promotions_payload = read_json(args.segment_promotions_json) if args.segment_promotions_json.exists() else {}
    selected = selected_alternatives_from_arg(args.selected_alternative, audit)
    context = build_context(args)
    audit_by_label = index_audit_outings(audit)
    fallback_packages = fallback_replacement_packages(base) + fallback_replacement_packages(existing_generated)
    replacement_entries = copy.deepcopy(base.get("overrides") or [])
    for label, alternative_id in selected.items():
        if label not in audit_by_label:
            raise KeyError(f"Outing label {label} not found in audit")
        replacement_entries.append(
            build_replacement_entry(
                label=label,
                alternative_id=alternative_id,
                current_map=current_map,
                fallback_packages=fallback_packages,
                audit_outing=audit_by_label[label],
                context=context,
                max_gap_miles=args.max_gap_miles,
            )
        )
    applied_promotions, skipped_promotions = apply_segment_ownership_promotions(
        replacement_entries,
        promotions_payload,
        current_map=current_map,
        context=context,
    )
    return {
        "description": (
            "Private generated field-menu replacements: public route rules plus "
            "multi-start alternatives selected by runnable-cost heuristics."
        ),
        "privacy": "private_exact_parking_coordinates",
        "source_files": {
            "base_overrides_json": display_path(args.base_overrides_json),
            "current_map_json": display_path(args.current_map_json),
            "multi_start_audit_json": display_path(args.multi_start_audit_json),
            "official_geojson": display_path(args.official_geojson),
            "connector_geojson": display_path(args.connector_geojson),
            "dem_tif": display_path(args.dem_tif),
        },
        "selected_alternatives": selected,
        "segment_ownership_promotions": applied_promotions,
        "segment_ownership_promotion_skips": skipped_promotions,
        "summary": output_summary(replacement_entries),
        "overrides": replacement_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-overrides-json", type=Path, default=DEFAULT_BASE_OVERRIDES_JSON)
    parser.add_argument("--current-map-json", type=Path, default=DEFAULT_CURRENT_MAP_JSON)
    parser.add_argument("--multi-start-audit-json", type=Path, default=DEFAULT_MULTI_START_AUDIT_JSON)
    parser.add_argument("--segment-promotions-json", type=Path, default=DEFAULT_SEGMENT_PROMOTIONS_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--max-gap-miles", type=float, default=0.05)
    parser.add_argument(
        "--selected-alternative",
        action="append",
        help="Manually select LABEL=ALTERNATIVE_ID values instead of auto-promoting promising no-blocker alternatives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_replacement_payload(args)
    write_json(args.output_json, payload)
    write_manifest(
        args.manifest_json,
        build_artifact_manifest(
            run_id="multi-start-field-menu-replacements-v2",
            command="python years/2026/scripts/multi_start_field_menu_replacements.py",
            inputs=[
                args.base_overrides_json,
                args.current_map_json,
                args.multi_start_audit_json,
                args.segment_promotions_json,
                args.state_json,
                args.official_geojson,
                args.connector_geojson,
                args.dem_tif,
                args.dem_summary_json,
            ],
            outputs=[args.output_json],
            metadata={
                "selected_alternatives": payload["selected_alternatives"],
                "segment_ownership_promotions": payload["segment_ownership_promotions"],
                "segment_ownership_promotion_skips": payload["segment_ownership_promotion_skips"],
                "summary": payload["summary"],
            },
        ),
    )
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
