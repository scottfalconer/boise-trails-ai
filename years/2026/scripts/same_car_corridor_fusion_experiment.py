#!/usr/bin/env python3
"""Probe same-parked-car fusions for repeated paid access/return corridors."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_CLUSTER_AUDIT_JSON = YEAR_DIR / "checkpoints" / "cluster-route-optimization-audit-2026-05-12.json"
DEFAULT_REPEAT_AUDIT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = (
    YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
)
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "same-car-corridor-fusion-experiment-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "same-car-corridor-fusion-experiment-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "same-car-corridor-fusion-experiment-2026-05-12-manifest.json"

START_CUE_TYPES = {"start_access", "official_segment_start"}
RETURN_CUE_TYPES = {"exit_access", "return_to_car"}
NON_OFFICIAL_CUE_TYPES = {
    "start_access",
    "connector_named_trail",
    "connector_road",
    "repeat_official_noncredit",
    "exit_access",
    "return_to_car",
}

PROBES = [
    {
        "probe_id": "dry-creek-fd09a-10b",
        "title": "Dry Creek single-return probe",
        "route_labels": ["FD09A", "10B"],
        "corridor_kind": "return",
        "strategy": "drop_intermediate_return_corridor",
        "candidate_route_name": "FD09A + 10B single parked-car Dry Creek route",
        "fusion_scope": "requested_fusion_A",
        "candidate_route_spec": [
            "Start at Dry Creek Parking Area/Trailhead.",
            "Use the current FD09A access and Barn Owl official leg.",
            "Do not return fully to the car after Barn Owl.",
            "Generate a continuous Red Tail / Dry Creek transition to the 10B Bitterbrush/Currant leg.",
            "Use the current 10B official/connector middle legs, then return once to the car.",
        ],
    },
    {
        "probe_id": "freestone-fd19c-fd04a-3-fd20a",
        "title": "Freestone Mountain Cove access-corridor probe",
        "route_labels": ["FD19C", "FD04A", "3", "FD20A"],
        "corridor_kind": "access",
        "strategy": "single_access_corridor_lower_bound",
        "candidate_route_name": "Freestone / Shane's / Three Bears same-parked-car access bundle",
        "fusion_scope": "explicit_repeated_corridor_not_in_A_B_C_shortlist",
        "candidate_route_spec": [
            "Start at Freestone Creek.",
            "Pay the shared #22C Mountain Cove access corridor once.",
            "Generate a real route ordering across Shane's, Two Point/Femrite/Patrol, Military Reserve, and Three Bears.",
            "Return to the parked car with a certified continuous GPX and rewritten cue sheet.",
        ],
    },
    {
        "probe_id": "cartwright-fd14a-fd14b-fd18a",
        "title": "Cartwright Doe Ridge ownership probe",
        "route_labels": ["FD14A", "FD14B", "FD18A"],
        "tested_route_labels": ["FD14A", "FD14B"],
        "corridor_kind": "access",
        "strategy": "promote_source_claim_and_remove_owner",
        "source_label": "FD14B",
        "remove_labels": ["FD14A"],
        "candidate_route_name": "FD14B with Doe Ridge ownership promoted; FD18A remains separate",
        "fusion_scope": "requested_fusion_B_with_corridor_context",
        "candidate_route_spec": [
            "Keep the current FD14B continuous Cartwright GPX.",
            "Promote Doe Ridge 1541 from FD14B repeat/owned-elsewhere evidence to claimed official credit if recertification accepts it.",
            "Remove FD14A from the active menu after source-card claim/cue promotion.",
            "Leave FD18A as the separate Polecat/Peggy's day unless a later bundle redesign is generated.",
        ],
    },
    {
        "probe_id": "avimor-fd27a-fd27b-fd27c",
        "title": "Avimor Spring Creek microcard probe",
        "route_labels": ["FD27A", "FD27B", "FD27C"],
        "corridor_kind": "access",
        "strategy": "promote_source_claim_and_remove_owner_keep_other",
        "source_label": "FD27B",
        "remove_labels": ["FD27A"],
        "candidate_route_name": "FD27B claims Spring Creek 1; FD27C remains same-day unless fused GPX is generated",
        "fusion_scope": "requested_fusion_C",
        "candidate_route_spec": [
            "Keep the current FD27B Spring Creek GPX.",
            "Promote Spring Creek 1 / segment 1661 from FD27B repeat/owned-elsewhere evidence to claimed official credit.",
            "Remove FD27A from the July 4 Avimor microcard day.",
            "Keep FD27C as a separate same-day loop until a continuous Spring Creek to Whistling Pig GPX is generated.",
        ],
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
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
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    return sorted({str(value) for value in values if value is not None}, key=sort_id)


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def int_value(value: Any) -> int:
    return int(round(float_value(value)))


def rounded(value: Any, digits: int = 2) -> float:
    return round(float_value(value), digits)


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("route_key") or route.get("label") or "")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route_key(route))
    outing_id = route.get("outing_id")
    return f"{outing_id}: {label}" if outing_id and str(outing_id) not in label else label


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for route in routes:
        keys = [route_key(route), str(route.get("outing_id") or ""), str(route.get("label") or ""), route_label(route)]
        for candidate_id in route.get("candidate_ids") or []:
            keys.append(str(candidate_id))
        for key in keys:
            if key:
                index[key] = route
    return index


def repeat_index(repeat_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in repeat_audit.get("routes") or []:
        key = str(row.get("route_key") or row.get("outing_id") or "")
        if key:
            rows[key] = row
    return rows


def official_segment_index(official_geojson: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        segment_id = str(props.get("segId") or "")
        if not segment_id:
            continue
        rows[segment_id] = {
            "segment_id": segment_id,
            "name": props.get("segName"),
            "direction": props.get("direction") or "both",
            "length_ft": props.get("LengthFt"),
            "special_instruction": props.get("specInst") or "",
        }
    return rows


def claimed_ids(route: dict[str, Any]) -> set[str]:
    return set(normalized_ids(route.get("segment_ids") or []))


def cue_repeat_ids(route: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        ids.update(normalized_ids(cue.get("official_repeat_segment_ids") or []))
    return ids


def declared_owned_elsewhere_ids(route: dict[str, Any]) -> set[str]:
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    return set(normalized_ids(reconciliation.get("declared_owned_elsewhere_segment_ids") or []))


def route_card_metrics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    segment_ids = sorted({segment_id for route in routes for segment_id in claimed_ids(route)}, key=sort_id)
    return {
        "route_count": len(routes),
        "labels": [route.get("label") for route in routes],
        "on_foot_miles": rounded(sum(float_value(route.get("on_foot_miles")) for route in routes)),
        "official_miles": rounded(sum(float_value(route.get("official_miles")) for route in routes)),
        "p75_minutes": sum(int_value(route.get("door_to_door_minutes_p75")) for route in routes),
        "p90_minutes": sum(int_value(route.get("door_to_door_minutes_p90")) for route in routes),
        "segment_ids": segment_ids,
        "segment_count": len(segment_ids),
    }


def cue_complexity(routes: list[dict[str, Any]]) -> dict[str, Any]:
    cue_count = 0
    non_official = 0
    overlap_warnings = 0
    field_check_needed = 0
    osm_named = 0
    unique_signed: set[str] = set()
    for route in routes:
        for cue in route.get("wayfinding_cues") or []:
            cue_count += 1
            cue_type = str(cue.get("cue_type") or "")
            if cue_type in NON_OFFICIAL_CUE_TYPES:
                non_official += 1
            text = " ".join(str(value) for value in [cue.get("note"), cue.get("field_warning"), cue.get("display_detail")]).lower()
            if "overlap" in text or "double-back" in text:
                overlap_warnings += 1
            if cue.get("confidence") == "field_check_needed":
                field_check_needed += 1
            signed = [str(item) for item in cue.get("signed_as") or []]
            if any("OSM " in item for item in signed):
                osm_named += 1
            unique_signed.update(item for item in signed if item)
    return {
        "cue_count": cue_count,
        "non_official_cue_count": non_official,
        "overlap_warning_count": overlap_warnings,
        "field_check_needed_cue_count": field_check_needed,
        "osm_named_cue_count": osm_named,
        "unique_signed_name_count": len(unique_signed),
    }


def complexity_delta(current: dict[str, Any], candidate: dict[str, Any]) -> str:
    current_score = (
        int_value(current.get("cue_count"))
        + int_value(current.get("overlap_warning_count"))
        + int_value(current.get("field_check_needed_cue_count"))
        + int_value(current.get("osm_named_cue_count"))
    )
    candidate_score = (
        int_value(candidate.get("cue_count"))
        + int_value(candidate.get("overlap_warning_count"))
        + int_value(candidate.get("field_check_needed_cue_count"))
        + int_value(candidate.get("osm_named_cue_count"))
    )
    if candidate_score < current_score:
        return "simpler"
    if candidate_score > current_score:
        return "worse"
    return "similar"


def first_cue(route: dict[str, Any], cue_types: set[str]) -> dict[str, Any] | None:
    for cue in route.get("wayfinding_cues") or []:
        if cue.get("cue_type") in cue_types:
            return cue
    return None


def last_cue(route: dict[str, Any], cue_types: set[str]) -> dict[str, Any] | None:
    for cue in reversed(route.get("wayfinding_cues") or []):
        if cue.get("cue_type") in cue_types:
            return cue
    return None


def cue_leg_miles(cue: dict[str, Any] | None) -> float:
    if not cue:
        return 0.0
    return float_value(cue.get("route_leg_miles") if cue.get("route_leg_miles") is not None else cue.get("leg_miles"))


def repeated_corridor_row(cluster_audit: dict[str, Any], route_keys: set[str], kind: str) -> dict[str, Any] | None:
    candidates = []
    for row in cluster_audit.get("already_paid_access_corridors") or []:
        if row.get("kind") != kind:
            continue
        row_keys = {str(item.get("route_key") or "") for item in row.get("routes") or []}
        if route_keys <= row_keys or row_keys <= route_keys:
            candidates.append(row)
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (-float_value(row.get("total_corridor_leg_miles")), str(row.get("family_signature"))))[0]


def field_days_for_routes(field_tool_data: dict[str, Any], route_keys: set[str]) -> list[dict[str, Any]]:
    rows = []
    for day in (field_tool_data.get("field_day_layer") or {}).get("field_days") or []:
        loop_keys = {
            str((loop.get("route_card_ref") or {}).get("outing_id") or "")
            for loop in day.get("loops") or []
        }
        if route_keys & loop_keys:
            rows.append(day)
    return rows


def common_field_day(field_tool_data: dict[str, Any], route_keys: set[str]) -> dict[str, Any] | None:
    for day in (field_tool_data.get("field_day_layer") or {}).get("field_days") or []:
        loop_keys = {
            str((loop.get("route_card_ref") or {}).get("outing_id") or "")
            for loop in day.get("loops") or []
        }
        if route_keys <= loop_keys:
            return day
    return None


def loop_schedule_minutes(loop: dict[str, Any], key: str) -> int:
    return int_value(loop.get(f"field_day_schedule_{key}_minutes") or loop.get(f"route_card_door_to_door_{key}_minutes"))


def reprice_day_after_removals(day: dict[str, Any] | None, removed_labels: set[str]) -> dict[str, Any] | None:
    if not day:
        return None
    remaining_loops = [loop for loop in day.get("loops") or [] if str(loop.get("label")) not in removed_labels]
    removed_loops = [loop for loop in day.get("loops") or [] if str(loop.get("label")) in removed_labels]
    if not remaining_loops:
        return None
    removed_p75 = sum(loop_schedule_minutes(loop, "p75") for loop in removed_loops)
    removed_p90 = sum(loop_schedule_minutes(loop, "p90") for loop in removed_loops)
    p75 = max(0, int_value(day.get("field_day_schedule_p75_minutes") or day.get("p75_minutes")) - removed_p75)
    p90 = max(0, int_value(day.get("field_day_schedule_p90_minutes") or day.get("p90_minutes")) - removed_p90)
    p75 = max(p75, max(loop_schedule_minutes(loop, "p75") for loop in remaining_loops))
    p90 = max(p90, max(loop_schedule_minutes(loop, "p90") for loop in remaining_loops))
    return {
        "field_day_id": day.get("field_day_id"),
        "date": day.get("date"),
        "loop_count": len(remaining_loops),
        "remaining_loop_labels": [loop.get("label") for loop in remaining_loops],
        "p75_minutes": p75,
        "p90_minutes": p90,
        "on_foot_miles": rounded(sum(float_value(loop.get("on_foot_miles")) for loop in remaining_loops)),
        "official_miles": rounded(sum(float_value(loop.get("official_miles")) for loop in remaining_loops)),
    }


def direction_status(segment_ids: set[str], official_segments: dict[str, dict[str, Any]], routes: list[dict[str, Any]]) -> dict[str, Any]:
    ascent_ids = sorted(
        [segment_id for segment_id in segment_ids if (official_segments.get(segment_id) or {}).get("direction") == "ascent"],
        key=sort_id,
    )
    if not ascent_ids:
        return {"status": "passed_no_ascent_segments", "ascent_segment_ids": []}
    evidence_by_id: dict[str, Any] = {}
    for route in routes:
        evidence = route.get("segment_direction_evidence") or {}
        if isinstance(evidence, dict):
            evidence_by_id.update(evidence)
    missing = [
        segment_id
        for segment_id in ascent_ids
        if not (evidence_by_id.get(segment_id) or {}).get("allowed_geometry_direction")
    ]
    return {
        "status": "passed" if not missing else "needs_ascent_direction_evidence",
        "ascent_segment_ids": ascent_ids,
        "missing_direction_evidence_segment_ids": missing,
    }


def source_removal_support(
    source: dict[str, Any],
    owners: list[dict[str, Any]],
    repeat_rows: dict[str, dict[str, Any]],
    official_segments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_key = route_key(source)
    source_repeat = repeat_rows.get(source_key) or {}
    source_actual_full = set(normalized_ids(source_repeat.get("actual_full_segment_ids") or []))
    source_repeat_cues = cue_repeat_ids(source)
    source_declared_elsewhere = declared_owned_elsewhere_ids(source)
    owner_claimed = {segment_id for owner in owners for segment_id in claimed_ids(owner)}
    direction = direction_status(owner_claimed, official_segments, [source])
    return {
        "source_route": route_label(source),
        "owner_routes": [route_label(owner) for owner in owners],
        "owner_claimed_segment_ids": normalized_ids(owner_claimed),
        "source_actual_full_segment_ids_include_owner": owner_claimed <= source_actual_full,
        "source_cues_owner_segments_as_repeat": owner_claimed <= source_repeat_cues,
        "source_declares_owner_segments_owned_elsewhere": owner_claimed <= source_declared_elsewhere,
        "ascent_direction_validation": direction,
        "promotion_gate": "requires_route_card_claim_and_cue_promotion"
        if owner_claimed <= source_actual_full
        else "blocked_source_does_not_physically_complete_owner",
    }


def estimated_minutes_after_mile_savings(current_minutes: int, current_miles: float, saved_miles: float) -> int | None:
    if current_minutes <= 0 or current_miles <= 0 or saved_miles <= 0:
        return None
    return max(0, int(round(current_minutes - (current_minutes / current_miles * saved_miles))))


def corridor_savings_lower_bound(corridor: dict[str, Any] | None) -> dict[str, Any]:
    if not corridor:
        return {"status": "missing_corridor_row", "miles": 0.0}
    leg_miles = [float_value(row.get("leg_miles")) for row in corridor.get("routes") or []]
    if len(leg_miles) < 2:
        return {"status": "not_repeated", "miles": 0.0}
    return {
        "status": "lower_bound_duplicate_corridor_savings",
        "miles": rounded(sum(leg_miles) - max(leg_miles)),
        "total_corridor_leg_miles": rounded(sum(leg_miles)),
        "retained_corridor_leg_miles": rounded(max(leg_miles)),
    }


def build_probe(
    probe: dict[str, Any],
    *,
    field_tool_data: dict[str, Any],
    cluster_audit: dict[str, Any],
    repeat_rows: dict[str, dict[str, Any]],
    official_segments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    routes_by_key = route_index(field_tool_data.get("routes") or [])
    route_labels = probe.get("route_labels") or []
    routes = [routes_by_key[str(label)] for label in route_labels if str(label) in routes_by_key]
    missing_labels = [str(label) for label in route_labels if str(label) not in routes_by_key]
    route_keys = {route_key(route) for route in routes}
    tested_labels = probe.get("tested_route_labels") or route_labels
    tested_routes = [routes_by_key[str(label)] for label in tested_labels if str(label) in routes_by_key]
    tested_keys = {route_key(route) for route in tested_routes}
    corridor = repeated_corridor_row(cluster_audit, route_keys, str(probe.get("corridor_kind") or ""))
    current_metrics = route_card_metrics(routes)
    tested_current_metrics = route_card_metrics(tested_routes)
    current_complexity = cue_complexity(routes)
    tested_current_complexity = cue_complexity(tested_routes)
    common_day = common_field_day(field_tool_data, tested_keys)
    route_days = field_days_for_routes(field_tool_data, route_keys)
    duplicate_corridor = corridor_savings_lower_bound(corridor)
    remove_labels = [str(label) for label in probe.get("remove_labels") or []]
    removed_routes = [routes_by_key[label] for label in remove_labels if label in routes_by_key]
    kept_routes = [route for route in tested_routes if str(route.get("label")) not in set(remove_labels)]
    source = routes_by_key.get(str(probe.get("source_label") or ""))
    current_claimed = {segment_id for route in tested_routes for segment_id in claimed_ids(route)}
    candidate_claimed = {segment_id for route in kept_routes for segment_id in claimed_ids(route)}
    if source and removed_routes:
        candidate_claimed.update(segment_id for route in removed_routes for segment_id in claimed_ids(route))
    source_support = source_removal_support(source, removed_routes, repeat_rows, official_segments) if source and removed_routes else None
    if source_support and source_support["source_actual_full_segment_ids_include_owner"]:
        candidate_status = "promotion_candidate_requires_recertification"
    elif source_support:
        candidate_status = "blocked_source_does_not_cover_removed_owner"
    else:
        candidate_status = "paper_only_needs_continuous_gpx_timing_and_coverage"
    if removed_routes:
        candidate_on_foot = route_card_metrics(kept_routes).get("on_foot_miles")
        candidate_official = route_card_metrics(kept_routes).get("official_miles")
        candidate_p75 = route_card_metrics(kept_routes).get("p75_minutes")
        candidate_p90 = route_card_metrics(kept_routes).get("p90_minutes")
        removed_miles = rounded(sum(float_value(route.get("on_foot_miles")) for route in removed_routes))
        removed_p75 = sum(int_value(route.get("door_to_door_minutes_p75")) for route in removed_routes)
        removed_p90 = sum(int_value(route.get("door_to_door_minutes_p90")) for route in removed_routes)
        candidate_complexity = cue_complexity(kept_routes)
    else:
        saved_miles = float_value(duplicate_corridor.get("miles"))
        baseline_p75 = int_value((common_day or {}).get("field_day_schedule_p75_minutes")) or int_value(
            tested_current_metrics.get("p75_minutes")
        )
        baseline_p90 = int_value((common_day or {}).get("field_day_schedule_p90_minutes")) or int_value(
            tested_current_metrics.get("p90_minutes")
        )
        candidate_on_foot = rounded(float_value(current_metrics.get("on_foot_miles")) - saved_miles)
        candidate_official = current_metrics.get("official_miles")
        candidate_p75 = estimated_minutes_after_mile_savings(
            baseline_p75,
            float_value(tested_current_metrics.get("on_foot_miles")),
            saved_miles,
        )
        candidate_p90 = estimated_minutes_after_mile_savings(
            baseline_p90,
            float_value(tested_current_metrics.get("on_foot_miles")),
            saved_miles,
        )
        removed_miles = rounded(saved_miles)
        removed_p75 = baseline_p75 - int_value(candidate_p75)
        removed_p90 = baseline_p90 - int_value(candidate_p90)
        removed_cues = max(1, len(tested_routes) - 1)
        candidate_complexity = dict(tested_current_complexity)
        candidate_complexity["cue_count"] = max(0, int_value(candidate_complexity.get("cue_count")) - removed_cues)
        candidate_complexity["non_official_cue_count"] = max(
            0, int_value(candidate_complexity.get("non_official_cue_count")) - removed_cues
        )
    day_after = reprice_day_after_removals(common_day, set(remove_labels)) if common_day and remove_labels else None
    coverage_status = (
        "coverage_preserved_if_source_claim_promoted"
        if removed_routes and source_support and source_support["source_actual_full_segment_ids_include_owner"]
        else "same_claimed_coverage_needs_new_continuous_gpx_audit"
        if not removed_routes
        else "coverage_blocked"
    )
    direction = direction_status(candidate_claimed or current_claimed, official_segments, kept_routes + ([source] if source else []))
    return {
        "probe_id": probe.get("probe_id"),
        "title": probe.get("title"),
        "fusion_scope": probe.get("fusion_scope"),
        "strategy": probe.get("strategy"),
        "candidate_route_name": probe.get("candidate_route_name"),
        "route_labels": route_labels,
        "tested_route_labels": tested_labels,
        "missing_route_labels": missing_labels,
        "current_separate_cards": current_metrics,
        "current_tested_cards": tested_current_metrics,
        "current_field_day_context": {
            "common_field_day_id": common_day.get("field_day_id") if common_day else None,
            "common_field_day_date": common_day.get("date") if common_day else None,
            "common_field_day_p75_minutes": common_day.get("field_day_schedule_p75_minutes") if common_day else None,
            "common_field_day_p90_minutes": common_day.get("field_day_schedule_p90_minutes") if common_day else None,
            "touched_field_day_count": len(route_days),
            "touched_field_days": [
                {
                    "field_day_id": day.get("field_day_id"),
                    "date": day.get("date"),
                    "loop_labels": [loop.get("label") for loop in day.get("loops") or []],
                    "p75_minutes": day.get("field_day_schedule_p75_minutes"),
                    "p90_minutes": day.get("field_day_schedule_p90_minutes"),
                }
                for day in route_days
            ],
        },
        "duplicate_corridor": {
            "family_signature": corridor.get("family_signature") if corridor else None,
            "kind": corridor.get("kind") if corridor else probe.get("corridor_kind"),
            "same_trailhead": corridor.get("same_trailhead") if corridor else None,
            "same_day_bundle_possible": corridor.get("same_day_bundle_possible") if corridor else None,
            "total_corridor_leg_miles": corridor.get("total_corridor_leg_miles") if corridor else None,
            "lower_bound_saved_miles": duplicate_corridor.get("miles"),
            "routes": [
                {
                    "label": (row.get("route") or {}).get("label"),
                    "leg_miles": row.get("leg_miles"),
                    "cue_seq": row.get("cue_seq"),
                    "signed_as": row.get("signed_as"),
                    "target": row.get("target"),
                    "official_repeat_segment_ids": row.get("official_repeat_segment_ids"),
                    "official_repeat_miles": row.get("official_repeat_miles"),
                }
                for row in (corridor or {}).get("routes") or []
            ],
        },
        "current_cue_complexity": current_complexity,
        "tested_cue_complexity": tested_current_complexity,
        "candidate_fused_route": {
            "status": candidate_status,
            "route_spec": probe.get("candidate_route_spec") or [],
            "kept_route_labels": [route.get("label") for route in kept_routes] if kept_routes else tested_labels,
            "removed_route_labels": remove_labels,
            "on_foot_miles": rounded(candidate_on_foot),
            "official_miles": rounded(candidate_official),
            "p75_minutes": candidate_p75,
            "p90_minutes": candidate_p90,
            "saved_on_foot_miles": rounded(removed_miles),
            "saved_p75_minutes": removed_p75,
            "saved_p90_minutes": removed_p90,
            "pricing_status": "existing_route_card_reprice" if removed_routes else "corridor_scaled_estimate_needs_dem_and_gpx",
            "field_day_after_removal": day_after,
            "official_segment_coverage": {
                "status": coverage_status,
                "candidate_claimed_segment_ids": normalized_ids(candidate_claimed or current_claimed),
                "current_tested_segment_ids": normalized_ids(current_claimed),
            },
            "hidden_repeat_or_latent_credit_status": source_support
            or {
                "status": "no_owner_route_removed",
                "note": "Candidate keeps current claimed segment ownership; hidden repeat check requires generated GPX.",
            },
            "ascent_direction_validation": direction,
            "cue_complexity": candidate_complexity,
            "cue_sheet_assessment": complexity_delta(tested_current_complexity, candidate_complexity),
            "promotion_status": "not_promotable_until_continuous_gpx_p75_cues_coverage_and_recertification"
            if not removed_routes
            else "not_promotable_until_route_card_claim_cue_promotion_and_recertification",
        },
    }


def build_report(
    field_tool_data: dict[str, Any],
    cluster_audit: dict[str, Any],
    repeat_audit: dict[str, Any],
    official_geojson: dict[str, Any],
) -> dict[str, Any]:
    repeat_rows = repeat_index(repeat_audit)
    official_segments = official_segment_index(official_geojson)
    probes = [
        build_probe(
            probe,
            field_tool_data=field_tool_data,
            cluster_audit=cluster_audit,
            repeat_rows=repeat_rows,
            official_segments=official_segments,
        )
        for probe in PROBES
    ]
    promotion_candidates = [
        row
        for row in probes
        if row["candidate_fused_route"]["status"] == "promotion_candidate_requires_recertification"
    ]
    paper_only = [
        row
        for row in probes
        if row["candidate_fused_route"]["status"] == "paper_only_needs_continuous_gpx_timing_and_coverage"
    ]
    return {
        "schema": "boise_trails_same_car_corridor_fusion_experiment_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "promotion_candidates_and_paper_fusions_found",
        "source_files": {
            "field_tool_data_json": display_path(DEFAULT_FIELD_TOOL_DATA_JSON),
            "cluster_route_optimization_audit_json": display_path(DEFAULT_CLUSTER_AUDIT_JSON),
            "route_repeat_optimization_audit_json": display_path(DEFAULT_REPEAT_AUDIT_JSON),
            "official_segments_geojson": display_path(DEFAULT_OFFICIAL_SEGMENTS_GEOJSON),
        },
        "scope": {
            "purpose": "Stop paying repeated access/return corridors twice by probing same-parked-car fusions.",
            "promotion_policy": "This experiment does not mutate the active field packet. Candidates must still pass continuous GPX, p75/p90, cue, segment coverage, ascent-direction, and recertification gates.",
            "pricing_policy": "Ownership-removal candidates use existing certified route-card costs; non-removal corridor fusions are lower-bound estimates until route geometry and DEM timing are generated.",
        },
        "summary": {
            "probe_count": len(probes),
            "promotion_candidate_count": len(promotion_candidates),
            "paper_only_fusion_candidate_count": len(paper_only),
            "total_lower_bound_saved_on_foot_miles": rounded(
                sum(float_value(row["candidate_fused_route"].get("saved_on_foot_miles")) for row in probes)
            ),
            "total_existing_route_card_saved_on_foot_miles": rounded(
                sum(float_value(row["candidate_fused_route"].get("saved_on_foot_miles")) for row in promotion_candidates)
            ),
            "best_promotion_candidates": [
                {
                    "probe_id": row.get("probe_id"),
                    "candidate_route_name": row.get("candidate_route_name"),
                    "removed_route_labels": row["candidate_fused_route"].get("removed_route_labels"),
                    "saved_on_foot_miles": row["candidate_fused_route"].get("saved_on_foot_miles"),
                    "saved_p75_minutes": row["candidate_fused_route"].get("saved_p75_minutes"),
                    "saved_p90_minutes": row["candidate_fused_route"].get("saved_p90_minutes"),
                }
                for row in promotion_candidates
            ],
        },
        "probes": probes,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Same-Car Corridor Fusion Experiment",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Probes: {summary.get('probe_count')}",
        f"- Promotion candidates: {summary.get('promotion_candidate_count')}",
        f"- Paper-only fusion candidates: {summary.get('paper_only_fusion_candidate_count')}",
        f"- Total lower-bound saved on-foot miles: {summary.get('total_lower_bound_saved_on_foot_miles')}",
        f"- Existing route-card saved on-foot miles: {summary.get('total_existing_route_card_saved_on_foot_miles')}",
        "",
        "This is an experiment, not an active packet mutation. Paper-only fusions still need continuous GPX, DEM timing, cue rewrite, coverage, ascent-direction validation, and recertification.",
        "",
        "## Probe Results",
        "",
        "| Probe | Current on-foot | Current p75/p90 cards/day | Duplicate corridor | Candidate status | Candidate on-foot | Candidate p75/p90 cards/day | Savings | Cue sheet | Gate |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in report.get("probes") or []:
        current = row.get("current_tested_cards") or {}
        current_day = row.get("current_field_day_context") or {}
        candidate = row.get("candidate_fused_route") or {}
        candidate_day = candidate.get("field_day_after_removal") or {}
        corridor = row.get("duplicate_corridor") or {}
        gate = candidate.get("promotion_status")
        current_day_text = (
            f"{current_day.get('common_field_day_p75_minutes')}/{current_day.get('common_field_day_p90_minutes')}"
            if current_day.get("common_field_day_p75_minutes")
            else "n/a"
        )
        candidate_day_text = (
            f"{candidate_day.get('p75_minutes')}/{candidate_day.get('p90_minutes')}"
            if candidate_day.get("p75_minutes")
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("title")),
                    str(current.get("on_foot_miles")),
                    f"{current.get('p75_minutes')}/{current.get('p90_minutes')} / {current_day_text}",
                    str(corridor.get("total_corridor_leg_miles")),
                    f"`{candidate.get('status')}`",
                    str(candidate.get("on_foot_miles")),
                    f"{candidate.get('p75_minutes')}/{candidate.get('p90_minutes')} / {candidate_day_text}",
                    f"{candidate.get('saved_on_foot_miles')} mi / {candidate.get('saved_p75_minutes')} p75",
                    str(candidate.get("cue_sheet_assessment")),
                    str(gate),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes", ""])
    for row in report.get("probes") or []:
        candidate = row.get("candidate_fused_route") or {}
        coverage = candidate.get("official_segment_coverage") or {}
        latent = candidate.get("hidden_repeat_or_latent_credit_status") or {}
        direction = candidate.get("ascent_direction_validation") or {}
        lines.append(f"- `{row.get('probe_id')}`: coverage `{coverage.get('status')}`, repeat/latent gate `{latent.get('promotion_gate') or latent.get('status')}`, ascent `{direction.get('status')}`.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--cluster-audit-json", type=Path, default=DEFAULT_CLUSTER_AUDIT_JSON)
    parser.add_argument("--repeat-audit-json", type=Path, default=DEFAULT_REPEAT_AUDIT_JSON)
    parser.add_argument("--official-segments-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        read_json(args.field_tool_data_json),
        read_json(args.cluster_audit_json),
        read_json(args.repeat_audit_json),
        read_json(args.official_segments_geojson),
    )
    report["source_files"] = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "cluster_route_optimization_audit_json": display_path(args.cluster_audit_json),
        "route_repeat_optimization_audit_json": display_path(args.repeat_audit_json),
        "official_segments_geojson": display_path(args.official_segments_geojson),
    }
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id="same_car_corridor_fusion_experiment",
        command="python years/2026/scripts/same_car_corridor_fusion_experiment.py",
        inputs=[
            args.field_tool_data_json,
            args.cluster_audit_json,
            args.repeat_audit_json,
            args.official_segments_geojson,
        ],
        outputs=[args.output_json, args.output_md],
        metadata={"status": report.get("status"), "summary": report.get("summary")},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
