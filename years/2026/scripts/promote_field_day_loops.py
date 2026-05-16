#!/usr/bin/env python3
"""Promote selected field-day loops into canonical route-card source data.

The field-day scheduler can select graph-validated loop candidates before those
loops exist as phone route cards. This script converts the selected loops into
the canonical map-data shape consumed by export_mobile_field_packet.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_forced_anchor_gpx_export as forced_gpx  # noqa: E402
import p90_relaxed_drive_day_gpx_export as day_gpx  # noqa: E402
import multi_start_alternative_audit as multi_start  # noqa: E402
from accepted_route_replacements import (  # noqa: E402
    ACTIVE_STATUS,
    INVESTIGATE_STATUS,
    AcceptedRouteReplacementIndex,
    BLOCKING_STATUSES,
    DEFAULT_ACCEPTED_REPLACEMENTS_JSON,
    candidate_metrics,
    dominance_deltas,
    route_metrics,
    anchor_refs_match,
    normalized_segment_ids as replacement_segment_ids,
)
from block_day_packager import logistics_for_candidate, route_cue, round_miles  # noqa: E402
from block_route_candidate_pass import PALETTE, multiline_feature, parking_feature, split_coords_on_gaps  # noqa: E402
from export_execution_gpx import haversine_miles, load_official_segment_index, validate_track_segments  # noqa: E402
from export_field_day_layer import (  # noqa: E402
    build_route_card_index,
    find_route_card,
    route_card_certification_blockers,
)
from human_loop_plan import (  # noqa: E402
    render_outing_menu_markdown,
    render_package_map_html,
    sync_official_segment_features,
)
from multi_start_field_menu_replacements import (  # noqa: E402
    add_segment_to_component,
    segment_row_from_source,
    update_cue_for_segment_promotion,
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
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    build_performance_profile,
    candidate_from_trail_group,
    group_remaining_by_trail,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    load_state,
    load_trailheads_from_geojson,
    merge_planning_trailheads,
    slugify,
)
from route_block_planner import normalize_trail_name  # noqa: E402


DEFAULT_CALENDAR_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-calendar-assignment-2026-05-06.json"
DEFAULT_BASE_MAP_DATA_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_FIELD_TOOL_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PERSONAL_ROUTE_MENU_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_HYBRID_ROUTE_PASS_JSON = YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
DEFAULT_FORCED_ANCHOR_PROBE_JSON = YEAR_DIR / "checkpoints" / "p90-forced-anchor-probe-2026-05-06.json"
DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "p90-forced-anchor-gpx-export-2026-05-06.json"
DEFAULT_FIELD_PACKET_MANIFEST_JSON = REPO_ROOT / "docs" / "field-packet" / "manifest.json"
DEFAULT_OUTPUT_MAP_DATA_JSON = DEFAULT_BASE_MAP_DATA_JSON
DEFAULT_OUTPUT_MAP_HTML = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map.html"
DEFAULT_OUTPUT_MENU_MD = YEAR_DIR / "outputs" / "private" / "2026-outing-menu.md"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "field-day-loop-promotion-2026-05-11.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "field-day-loop-promotion-2026-05-11.md"
DEFAULT_SEGMENT_PROMOTIONS_JSON = (
    YEAR_DIR / "inputs" / "personal" / "2026-cross-package-segment-promotions-v1.json"
)
DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON = forced_gpx.DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON


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


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def feature_index(map_data: dict[str, Any], collection_name: str) -> dict[str, list[dict[str, Any]]]:
    collection = (map_data.get("feature_collections") or {}).get(collection_name) or {}
    indexed: dict[str, list[dict[str, Any]]] = {}
    for feature in collection.get("features") or []:
        candidate_id = (feature.get("properties") or {}).get("candidate_id")
        if candidate_id is None:
            continue
        indexed.setdefault(str(candidate_id), []).append(feature)
    return indexed


def personal_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in (payload.get("route_menu") or {}).get("all_candidates") or []
        if candidate.get("candidate_id")
    }


def hybrid_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate_id): candidate
        for candidate_id, candidate in (payload.get("candidate_index") or {}).items()
    }


def forced_probe_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("probe_rows") or []:
        candidate_id = str(row.get("candidate_id") or "")
        trailhead = str(row.get("anchor_name") or row.get("trailhead") or "")
        if candidate_id and trailhead:
            index[(candidate_id, trailhead)] = row
            index[(f"{candidate_id}::{trailhead}", trailhead)] = row
    return index


def sync_missing_cue_trailhead_metadata(cue: dict[str, Any], candidate: dict[str, Any]) -> None:
    cue_trailhead = cue.setdefault("trailhead", {})
    candidate_trailhead = candidate.get("trailhead") or {}
    for key in ("parking_confidence", "source", "field_ready"):
        if cue_trailhead.get(key) in (None, "") and candidate_trailhead.get(key) not in (None, ""):
            cue_trailhead[key] = candidate_trailhead.get(key)


def trailhead_display_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def candidate_total_minutes(candidate: dict[str, Any], loop: dict[str, Any]) -> int:
    estimates = candidate.get("time_estimates_minutes") or {}
    return int(
        loop.get("p75_minutes")
        or estimates.get("door_to_door_p75")
        or candidate.get("total_minutes")
        or 0
    )


def loop_segment_ids(candidate: dict[str, Any], existing_route: dict[str, Any] | None = None) -> list[int]:
    if existing_route:
        values = existing_route.get("segment_ids") or []
    else:
        values = candidate.get("segment_ids") or [segment.get("seg_id") for segment in candidate.get("segments") or []]
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def normalized_segment_ids(values: list[Any] | tuple[Any, ...] | None) -> list[int]:
    result = []
    for value in values or []:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def promoted_segment_rows_by_target(
    promotions_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for promotion in promotions_payload.get("promotions") or []:
        if promotion.get("status") != "promoted":
            continue
        target = promotion.get("to") or {}
        candidate_id = str(target.get("candidate_id") or "")
        if not candidate_id:
            continue
        rows.setdefault(candidate_id, []).append(promotion)
    return rows


def removed_source_loop_targets(
    promotions_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for promotion in promotions_payload.get("promotions") or []:
        if promotion.get("status") != "promoted":
            continue
        source = promotion.get("from") or {}
        source_action = str(promotion.get("source_action") or source.get("source_action") or "keep")
        if source_action != "remove_route_card":
            continue
        source_candidate_id = str(source.get("candidate_id") or "")
        target_candidate_id = str((promotion.get("to") or {}).get("candidate_id") or "")
        if not source_candidate_id or not target_candidate_id:
            continue
        row = rows.setdefault(
            source_candidate_id,
            {
                "source_candidate_id": source_candidate_id,
                "target_candidate_id": target_candidate_id,
                "segment_ids": [],
                "reasons": [],
            },
        )
        row["segment_ids"].append(str(promotion.get("segment_id")))
        reason = promotion.get("reason")
        if reason and reason not in row["reasons"]:
            row["reasons"].append(reason)
    return rows


def apply_segment_promotions_to_route_card(
    *,
    component: dict[str, Any],
    cue: dict[str, Any] | None,
    promotions: list[dict[str, Any]],
    base_map_data: dict[str, Any],
    official_by_id: dict[str, dict[str, Any]],
) -> None:
    for promotion in promotions:
        segment_id = str(promotion.get("segment_id"))
        segment_row = segment_row_from_source(
            segment_id=segment_id,
            promotion=promotion,
            current_map=base_map_data,
            official_by_id=official_by_id,
            target_cue=cue,
        )
        add_segment_to_component(component, segment_row)
        if cue is not None:
            update_cue_for_segment_promotion(
                cue,
                segment_row=segment_row,
                insert_after_segment_id=(promotion.get("to") or {}).get("insert_after_segment_id"),
                promotion=promotion,
            )


def route_feature_parts(feature: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry = feature.get("geometry") or {}
    raw_parts = []
    if geometry.get("type") == "LineString":
        raw_parts = [geometry.get("coordinates") or []]
    elif geometry.get("type") == "MultiLineString":
        raw_parts = geometry.get("coordinates") or []
    parts: list[list[tuple[float, float]]] = []
    for raw_part in raw_parts:
        part = []
        for coord in raw_part or []:
            if isinstance(coord, list | tuple) and len(coord) >= 2:
                part.append((float(coord[0]), float(coord[1])))
        if len(part) >= 2:
            parts.append(part)
    return parts


def cumulative_points(parts: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    total = 0.0
    for part in parts:
        prior = None
        for point in part:
            if prior is not None:
                total += haversine_miles(prior, point)
            points.append({"point": point, "mile": total})
            prior = point
    return points


def route_interval_hits(
    route_points: list[dict[str, Any]],
    coords: list[tuple[float, float]],
    *,
    start_mile: float,
    end_mile: float,
    tolerance_miles: float = 0.035,
) -> bool:
    if not route_points or not coords or end_mile <= start_mile:
        return False
    interval_points = [
        item
        for item in route_points
        if start_mile - 0.05 <= float(item.get("mile") or 0) <= end_mile + 0.05
    ]
    if not interval_points:
        return False
    for coord in coords:
        if any(haversine_miles(coord, item["point"]) <= tolerance_miles for item in interval_points):
            return True
    return False


def normalized_name_key(value: Any) -> str:
    text = str(value or "").replace("’", "'").replace("'", "")
    text = re.sub(r"\bmtn\b", "mountain", text, flags=re.IGNORECASE)
    text = normalize_key(text)
    return re.sub(r"\btrail(?:-[0-9]+)?\b", "trail", text).strip("-")


def official_feature_name_keys(feature: dict[str, Any]) -> set[str]:
    props = feature.get("properties") or {}
    values = [
        props.get("trail_name"),
        props.get("segment_name"),
        props.get("seg_name"),
        props.get("title"),
    ]
    return {key for key in (normalized_name_key(value) for value in values) if key}


def leg_name_keys(leg: dict[str, Any]) -> set[str]:
    values = list(leg.get("connector_names") or []) + list(leg.get("signpost_labels") or [])
    return {key for key in (normalized_name_key(value) for value in values) if key}


def name_keys_overlap(left: set[str], right: set[str]) -> bool:
    for left_key in left:
        for right_key in right:
            if left_key == right_key or left_key in right_key or right_key in left_key:
                return True
    return False


def leg_distance_miles(leg: dict[str, Any]) -> float:
    for key in ("distance_miles", "mapped_access_miles", "connector_miles", "road_miles", "official_repeat_miles"):
        value = leg.get(key)
        try:
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def infer_leg_repeat_segment_ids(
    *,
    leg: dict[str, Any],
    route_points: list[dict[str, Any]],
    official_features: list[dict[str, Any]],
    start_mile: float,
    end_mile: float,
    fallback_segment_ids: list[Any] | None = None,
) -> list[int]:
    if normalized_segment_ids(leg.get("official_repeat_segment_ids")):
        return normalized_segment_ids(leg.get("official_repeat_segment_ids"))
    if str(leg.get("strategy") or "") == "out_and_back":
        return normalized_segment_ids(fallback_segment_ids)
    keys = leg_name_keys(leg)
    connector_classes = {str(value) for value in leg.get("connector_classes") or []}
    allow_geometry_only = (
        float(leg.get("official_repeat_miles") or 0) > 0
        and (
            "official_repeat" in connector_classes
            or str(leg.get("strategy") or "") in {"accepted_manual_split_gpx", "mapped_mixed_loop"}
            or not keys
        )
    )
    inferred = []
    for require_name_match in ([True, False] if allow_geometry_only else [True]):
        if inferred:
            break
        if require_name_match and not keys:
            continue
        for feature in official_features:
            props = feature.get("properties") or {}
            segment_id = props.get("seg_id") or props.get("segment_id")
            if segment_id is None:
                continue
            if require_name_match and not name_keys_overlap(keys, official_feature_name_keys(feature)):
                continue
            coords = [coord for part in route_feature_parts(feature) for coord in part]
            if route_interval_hits(route_points, coords, start_mile=start_mile, end_mile=end_mile):
                inferred.append(segment_id)
    if not inferred and allow_geometry_only and fallback_segment_ids:
        inferred.extend(fallback_segment_ids)
    return normalized_segment_ids(inferred)


def enrich_official_repeat_segment_ids(map_data: dict[str, Any]) -> None:
    official_features = ((map_data.get("feature_collections") or {}).get("official_segments") or {}).get("features") or []
    route_features_by_candidate = feature_index(map_data, "routes")
    for candidate_id, cue in (map_data.get("route_cues") or {}).items():
        route_features = route_features_by_candidate.get(str(candidate_id)) or []
        route_points = cumulative_points([part for feature in route_features for part in route_feature_parts(feature)])
        if not route_points:
            continue
        segment_ids = [segment.get("seg_id") for segment in cue.get("segments") or []]
        cursor = 0.0
        start_access = cue.get("start_access") or {}
        if start_access:
            end = cursor + leg_distance_miles(start_access)
            inferred = infer_leg_repeat_segment_ids(
                leg=start_access,
                route_points=route_points,
                official_features=official_features,
                start_mile=cursor,
                end_mile=end,
            )
            if inferred:
                start_access["official_repeat_segment_ids"] = inferred
            cursor = end
        groups: list[dict[str, Any]] = []
        for segment in cue.get("segments") or []:
            trail_name = str(segment.get("trail_name") or segment.get("segment_name") or "")
            official_miles = float(segment.get("official_miles") or 0)
            if groups and groups[-1]["trail_name"] == trail_name:
                groups[-1]["official_miles"] += official_miles
                groups[-1]["segment_ids"].append(segment.get("seg_id"))
            else:
                groups.append({"trail_name": trail_name, "official_miles": official_miles, "segment_ids": [segment.get("seg_id")]})
        links = list(cue.get("between_links") or [])
        for index, group in enumerate(groups):
            cursor += float(group["official_miles"])
            if index >= len(groups) - 1:
                continue
            link = links[index] if index < len(links) else {}
            end = cursor + leg_distance_miles(link)
            fallback_ids = list(group.get("segment_ids") or []) + list(groups[index + 1].get("segment_ids") or [])
            inferred = infer_leg_repeat_segment_ids(
                leg=link,
                route_points=route_points,
                official_features=official_features,
                start_mile=cursor,
                end_mile=end,
                fallback_segment_ids=fallback_ids,
            )
            if inferred:
                link["official_repeat_segment_ids"] = inferred
            cursor = end
        return_to_car = cue.get("return_to_car") or {}
        if return_to_car:
            end = float(route_points[-1].get("mile") or cursor)
            inferred = infer_leg_repeat_segment_ids(
                leg=return_to_car,
                route_points=route_points,
                official_features=official_features,
                start_mile=cursor,
                end_mile=end,
                fallback_segment_ids=segment_ids,
            )
            if inferred:
                return_to_car["official_repeat_segment_ids"] = inferred


def route_segment_ids(route: dict[str, Any] | None) -> set[int]:
    result = set()
    for value in (route or {}).get("segment_ids") or []:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def label_for_loop(loop: dict[str, Any], existing_route: dict[str, Any] | None, loop_index: int) -> str:
    if existing_route and existing_route.get("label"):
        return str(existing_route["label"])
    day_number = int(loop.get("draft_day_number") or 0)
    suffix = chr(ord("A") + loop_index) if loop_index >= 0 else ""
    return f"FD{day_number:02d}{suffix}"


def flat_assignment_loops(calendar: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for assignment in calendar.get("assignments") or []:
        day = assignment.get("field_day") or {}
        for loop_index, raw_loop in enumerate(day.get("loops") or []):
            loop = copy.deepcopy(raw_loop)
            loop["date"] = assignment.get("date")
            loop["draft_day_number"] = day.get("draft_day_number")
            loop["loop_index"] = loop_index
            rows.append(loop)
    return rows


def superset_route_overrides(
    *,
    calendar: dict[str, Any],
    context: "PromotionContext",
    field_tool_payload: dict[str, Any],
    card_index: dict[str, Any],
    accepted_replacements: AcceptedRouteReplacementIndex | None = None,
) -> dict[str, dict[str, Any]]:
    """Map stale selected loops to current certified superseding route cards.

    Accepted route-card promotions can move a segment from one old selected loop
    into another active card. Exact candidate/segment matching then undercounts
    coverage even though the current certified route inventory has the right
    card. When that happens, replace the stale subset loop with the current
    certified superset card.
    """

    loop_rows = []
    for loop in flat_assignment_loops(calendar):
        candidate = context.candidate_for_loop(loop)
        exact_route = find_route_card(loop, card_index, accepted_replacements)
        exact_blockers = route_card_certification_blockers(exact_route, context.args.field_packet_dir) if exact_route else []
        original_segments = set(loop_segment_ids(candidate))
        effective_segments = route_segment_ids(exact_route) if exact_route and not exact_blockers else original_segments
        loop_rows.append(
            {
                "loop": loop,
                "candidate": candidate,
                "exact_route": exact_route if exact_route and not exact_blockers else None,
                "original_segments": original_segments,
                "effective_segments": effective_segments,
            }
        )

    official_ids = {int(segment["seg_id"]) for segment in context.official_segments}
    covered = {segment_id for row in loop_rows for segment_id in row["effective_segments"]}
    missing = sorted(official_ids - covered)
    if not missing:
        return {}

    clean_routes = []
    for route in field_tool_payload.get("routes") or []:
        blockers = route_card_certification_blockers(route, context.args.field_packet_dir)
        if blockers:
            continue
        segments = route_segment_ids(route)
        if segments:
            clean_routes.append((route, segments))

    overrides: dict[str, dict[str, Any]] = {}
    used_route_labels: set[str] = set()
    for missing_id in missing:
        candidate_routes = [
            (route, segments)
            for route, segments in clean_routes
            if missing_id in segments and str(route.get("label")) not in used_route_labels
        ]
        candidate_routes.sort(key=lambda item: (len(item[1]), str(item[0].get("label") or "")))
        for route, route_segments in candidate_routes:
            candidates = [
                row
                for row in loop_rows
                if not row["exact_route"]
                and row["original_segments"]
                and row["original_segments"].issubset(route_segments)
                and route_segments - row["original_segments"]
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda row: (
                    -len(row["original_segments"] & route_segments),
                    int(row["loop"].get("draft_day_number") or 999),
                    int(row["loop"].get("loop_index") or 0),
                )
            )
            selected = candidates[0]["loop"]
            overrides[str(selected["loop_id"])] = route
            used_route_labels.add(str(route.get("label")))
            break
    return overrides


def route_component(
    *,
    loop: dict[str, Any],
    candidate_id: str,
    candidate: dict[str, Any],
    existing_route: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    replacement_candidate = bool(candidate.get("accepted_replacement_id") or candidate.get("accepted_replacement_status"))
    if replacement_candidate:
        official = float((existing_route or {}).get("official_miles") or candidate.get("official_new_miles") or loop.get("official_miles") or 0)
        on_foot = float(
            (existing_route or {}).get("on_foot_miles")
            or candidate.get("estimated_total_on_foot_miles")
            or candidate.get("on_foot_miles")
            or loop.get("on_foot_miles")
            or 0
        )
    else:
        official = float((existing_route or {}).get("official_miles") or loop.get("official_miles") or candidate.get("official_new_miles") or 0)
        on_foot = float(
            (existing_route or {}).get("on_foot_miles")
            or loop.get("on_foot_miles")
            or candidate.get("estimated_total_on_foot_miles")
            or candidate.get("on_foot_miles")
            or 0
        )
    estimates = copy.deepcopy(candidate.get("time_estimates_minutes") or {})
    if existing_route:
        if existing_route.get("door_to_door_minutes_p75") is not None:
            estimates["door_to_door_p75"] = int(existing_route["door_to_door_minutes_p75"])
        if existing_route.get("door_to_door_minutes_p90") is not None:
            estimates["door_to_door_p90"] = int(existing_route["door_to_door_minutes_p90"])
    elif not replacement_candidate and loop.get("p75_minutes") is not None:
        estimates["door_to_door_p75"] = int(loop["p75_minutes"])
        estimates.setdefault("recommended_door_to_door", int(loop["p75_minutes"]))
        if loop.get("p90_minutes") is not None:
            estimates["door_to_door_p90"] = int(loop["p90_minutes"])
    total_minutes = (
        (existing_route or {}).get("door_to_door_minutes_p75")
        or (candidate.get("total_minutes") if replacement_candidate else None)
        or loop.get("p75_minutes")
        or candidate.get("total_minutes")
        or 0
    )
    trailhead = (
        (existing_route or {}).get("trailhead")
        or (candidate.get("trailhead") if replacement_candidate else None)
        or loop.get("trailhead")
        or candidate.get("trailhead")
        or ""
    )
    return {
        "route_number": loop.get("promotion_route_number"),
        "candidate_id": candidate_id,
        "field_menu_group_id": loop["loop_id"],
        "field_menu_label": label,
        "trail_names": list((existing_route or {}).get("trails") or candidate.get("trail_names") or loop.get("trail_names") or []),
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "total_minutes": int(total_minutes),
        "raw_total_minutes": candidate.get("raw_total_minutes"),
        "trailhead": trailhead_display_name(trailhead),
        "less_optimal_flags": list(candidate.get("less_optimal_flags") or candidate.get("flags") or []),
        "segment_ids": loop_segment_ids(candidate, existing_route),
        "time_breakdown_minutes": copy.deepcopy(candidate.get("time_breakdown_minutes")),
        "time_estimates_minutes": estimates,
        "effort": copy.deepcopy(candidate.get("effort") or {
            "ascent_ft": loop.get("ascent_ft"),
            "grade_adjusted_miles": loop.get("grade_adjusted_miles"),
        }),
        "route_status": (existing_route or {}).get("route_status") or candidate.get("route_status") or "graph_validated",
        "source": loop.get("source"),
        "source_loop_id": loop.get("loop_id"),
        "source_candidate_id": loop.get("candidate_id"),
    }


def package_for_day(day: dict[str, Any], components: list[dict[str, Any]]) -> dict[str, Any]:
    segment_ids = sorted({int(seg_id) for component in components for seg_id in component.get("segment_ids") or []})
    trailheads = sorted({str(component.get("trailhead")) for component in components if component.get("trailhead")})
    trail_names = sorted({str(name) for component in components for name in component.get("trail_names") or [] if name})
    official = sum(float(component.get("official_miles") or 0) for component in components)
    on_foot = sum(float(component.get("on_foot_miles") or 0) for component in components)
    return {
        "package_number": 100 + int(day.get("draft_day_number") or 0),
        "block_id": f"field-day-{day.get('draft_day_number')}",
        "block_name": f"Field Day {day.get('draft_day_number')} route-card bundle",
        "source_field_day_id": day.get("field_day_id"),
        "source_date": day.get("date"),
        "component_route_count": len(components),
        "component_candidate_ids": [component["candidate_id"] for component in components],
        "trail_names": trail_names,
        "official_miles": round_miles(official),
        "on_foot_miles": round_miles(on_foot),
        "ratio": round(on_foot / official, 2) if official else None,
        "trailheads": trailheads,
        "trailhead_count": len(trailheads),
        "primary_trailhead": trailheads[0] if len(trailheads) == 1 else None,
        "total_minutes_components": sum(int(component.get("total_minutes") or 0) for component in components),
        "component_routes_under_1_official_mile": sum(1 for component in components if float(component.get("official_miles") or 0) < 1),
        "component_routes_under_2_official_miles": sum(1 for component in components if float(component.get("official_miles") or 0) < 2),
        "segment_ids": segment_ids,
        "components": components,
        "planning_status": "field_day_loop_promoted",
        "planning_reasons": [
            "field_day_assignment_selected_loop",
            "loop_has_graph_validated_candidate_or_existing_certified_card",
            "route_card_source_promoted_for_phone_packet_certification",
        ],
    }


def copy_feature_group(
    features_by_candidate: dict[str, list[dict[str, Any]]],
    candidate_id: str,
    props: dict[str, Any],
) -> list[dict[str, Any]]:
    copied = []
    for feature in features_by_candidate.get(candidate_id, []):
        clone = copy.deepcopy(feature)
        clone_props = clone.setdefault("properties", {})
        original_kind = clone_props.get("kind")
        clone_props.update(props)
        if original_kind and original_kind != "route":
            clone_props["kind"] = original_kind
        copied.append(clone)
    return copied


def replacement_preservation_blocker(record: dict[str, Any]) -> str:
    return f"accepted_replacement_{record.get('status')}_blocks_preservation"


def route_endpoint_distance_for_replacement(
    *,
    record: dict[str, Any],
    anchor: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    anchor_point = (float(anchor["lon"]), float(anchor["lat"]))
    best: tuple[float, str, int] | None = None
    for raw_segment_id in replacement_segment_ids(record.get("target_segment_ids")):
        segment = official_index.get(int(raw_segment_id))
        coords = segment.get("coordinates") if segment else None
        if not coords:
            continue
        direction = str(segment.get("direction") or "both").lower()
        endpoints = [("segment_start", coords[0]), ("segment_end", coords[-1])]
        if direction == "ascent":
            endpoints = [("segment_start", coords[0])]
        for endpoint_name, coord in endpoints:
            distance = haversine_miles(anchor_point, (float(coord[0]), float(coord[1])))
            if best is None or distance < best[0]:
                best = (distance, endpoint_name, int(raw_segment_id))
    if best is None:
        return {
            "anchor_to_credit_endpoint_distance_miles": None,
            "credit_endpoint_used": None,
            "direction_rule_checked": False,
        }
    distance, endpoint_name, segment_id = best
    return {
        "anchor_to_credit_endpoint_distance_miles": round_miles(distance),
        "credit_endpoint_used": endpoint_name,
        "credit_endpoint_segment_id": segment_id,
        "direction_rule_checked": True,
    }


def apply_replacement_metadata_to_component(
    component: dict[str, Any],
    record: dict[str, Any],
    deltas: dict[str, Any] | None = None,
    endpoint: dict[str, Any] | None = None,
) -> None:
    component["accepted_replacement_id"] = record.get("replacement_id")
    component["accepted_replacement_status"] = record.get("status")
    component["accepted_anchor_ref"] = record.get("accepted_anchor_ref")
    component["accepted_anchor_label"] = record.get("public_anchor_label")
    component["route_card_status"] = record.get("route_card_status")
    component["packet_visibility"] = record.get("packet_visibility")
    component["certified_route_card"] = record.get("certified_route_card")
    component["requires_field_walkthrough"] = record.get("requires_field_walkthrough")
    if record.get("start_justification"):
        component["start_justification"] = record.get("start_justification")
    if record.get("status") == ACTIVE_STATUS:
        component["cue_generation_mode"] = "regenerated_for_reanchored_candidate"
    if endpoint:
        component.update(endpoint)
    if deltas:
        component["dominance_deltas"] = deltas


def apply_replacement_metadata_to_cue(
    cue: dict[str, Any],
    record: dict[str, Any],
    endpoint: dict[str, Any] | None = None,
) -> None:
    cue["accepted_replacement_id"] = record.get("replacement_id")
    cue["accepted_replacement_status"] = record.get("status")
    cue["accepted_anchor_ref"] = record.get("accepted_anchor_ref")
    cue["route_card_status"] = record.get("route_card_status")
    cue["packet_visibility"] = record.get("packet_visibility")
    cue["certified_route_card"] = record.get("certified_route_card")
    cue["requires_field_walkthrough"] = record.get("requires_field_walkthrough")
    if record.get("start_justification"):
        cue["start_justification"] = record.get("start_justification")
    if record.get("status") == ACTIVE_STATUS:
        cue["cue_generation_mode"] = "regenerated_for_reanchored_candidate"
    if endpoint:
        cue.update(endpoint)


def accepted_replacement_street_probe_anchors(
    records: list[dict[str, Any]],
    connector_geojson: dict[str, Any],
    official_segments_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen_refs = {
        str(record.get("accepted_anchor_ref") or "")
        for record in records
        if str(record.get("accepted_anchor_ref") or "").startswith("street-probe-")
    }
    if not seen_refs:
        return anchors
    for record in records:
        anchor_ref = str(record.get("accepted_anchor_ref") or "")
        if anchor_ref not in seen_refs:
            continue
        segments = [
            official_segments_by_id[int(segment_id)]
            for segment_id in replacement_segment_ids(record.get("target_segment_ids"))
            if int(segment_id) in official_segments_by_id
        ]
        for anchor in multi_start.street_parking_probes_for_segments(connector_geojson, segments, limit=20):
            if anchor_refs_match(anchor.get("anchor_id"), anchor_ref):
                anchors.append(anchor)
    return anchors


class PromotionContext:
    def __init__(self, args: argparse.Namespace, base_map_data: dict[str, Any]) -> None:
        self.args = args
        self.base_map_data = base_map_data
        self.accepted_replacements = AcceptedRouteReplacementIndex.from_path(args.accepted_replacements_json)
        self.personal_candidates = personal_index(read_json(args.personal_route_menu_json))
        self.hybrid_candidates = hybrid_index(read_json(args.hybrid_route_pass_json))
        self.forced_probe_rows = forced_probe_index(read_json(args.forced_anchor_probe_json))
        self.field_packet_gpx = day_gpx.field_packet_gpx_index(args.field_packet_manifest_json)
        self.forced_anchor_gpx = day_gpx.forced_anchor_gpx_index(args.forced_anchor_gpx_manifest_json)
        self.official_segments, _meta = load_official_segments(args.official_geojson)
        self.official_segments_by_id = {int(segment["seg_id"]): segment for segment in self.official_segments}
        self.trail_by_name = {
            normalize_trail_name(str(trail.get("trail_name") or "")): trail
            for trail in group_remaining_by_trail(self.official_segments)
        }
        self.official_index = load_official_segment_index(args.official_geojson)
        self.connector_graph = load_connector_graph(args.connector_geojson, official_segments=self.official_segments)
        self.route_features_by_candidate = feature_index(base_map_data, "routes")
        self.parking_features_by_candidate = feature_index(base_map_data, "parking")
        self.logistics_features_by_candidate = feature_index(base_map_data, "logistics")
        self._forced_candidate_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._forced_runtime: dict[str, Any] | None = None
        self._fallback_runtime: dict[str, Any] | None = None
        self._replacement_runtime: dict[str, Any] | None = None

    def forced_runtime(self) -> dict[str, Any]:
        if self._forced_runtime is None:
            state = read_json(self.args.state_json)
            dem_context = forced_gpx.load_dem_context(forced_gpx.DEFAULT_DEM_TIF, forced_gpx.DEFAULT_DEM_SUMMARY_JSON)
            performance_profile = forced_gpx.build_performance_profile(
                state=state,
                strava_activity_details_dir=forced_gpx.DEFAULT_STRAVA_DETAILS_DIR,
                activity_summary_csv=forced_gpx.DEFAULT_ACTIVITY_SUMMARY_CSV,
                activity_detail_summary_csv=forced_gpx.DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV,
                segment_perf_csv=forced_gpx.DEFAULT_SEGMENT_PERF_CSV,
            )
            anchors = forced_gpx.forced_probe.load_all_anchors(
                public_trailheads_geojson=forced_gpx.DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
                private_parking_anchors_geojson=forced_gpx.DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
                manual_design_jsons=forced_gpx.forced_probe.DEFAULT_MANUAL_DESIGN_JSONS,
            )
            self._forced_runtime = {
                "state": state,
                "sampler": dem_context["sampler"],
                "performance_profile": performance_profile,
                "anchors": anchors,
            }
        return self._forced_runtime

    def forced_candidate(self, loop: dict[str, Any]) -> dict[str, Any]:
        key = (str(loop.get("candidate_id") or ""), str(loop.get("trailhead") or ""))
        cached = self._forced_candidate_cache.get(key)
        if cached:
            return cached
        runtime = self.forced_runtime()
        seg_id = forced_gpx.segment_id_from_candidate_id(str(loop["candidate_id"]))
        segment = self.official_segments_by_id[seg_id]
        anchor = forced_gpx.anchor_by_name(runtime["anchors"], str(loop["trailhead"]))
        candidate = forced_gpx.candidate_from_trail_group(
            [forced_gpx.segment_trail(segment)],
            forced_gpx.forced_probe.forced_anchor_state(runtime["state"], anchor),
            runtime["performance_profile"],
            self.connector_graph,
            candidate_type="field_day_loop_promotion",
            elevation_sampler=runtime["sampler"],
        )
        probe_row = self.forced_probe_rows.get(key) or {}
        candidate["candidate_id"] = "::".join([str(loop["candidate_id"]), str(loop["trailhead"])])
        candidate["route_status"] = "graph_validated"
        candidate["official_new_miles"] = loop.get("official_miles") or probe_row.get("official_miles")
        candidate["estimated_total_on_foot_miles"] = loop.get("on_foot_miles") or probe_row.get("on_foot_miles")
        candidate["total_minutes"] = loop.get("p75_minutes") or probe_row.get("door_to_door_p75_minutes")
        candidate["time_estimates_minutes"] = {
            "door_to_door_p75": loop.get("p75_minutes") or probe_row.get("door_to_door_p75_minutes"),
            "door_to_door_p90": loop.get("p90_minutes") or probe_row.get("door_to_door_p90_minutes"),
            "recommended_door_to_door": loop.get("p75_minutes") or probe_row.get("door_to_door_p75_minutes"),
            "moving_effort_p75": probe_row.get("moving_effort_p75_minutes"),
            "route_finding_penalty": probe_row.get("route_finding_penalty_minutes"),
        }
        candidate["effort"] = {
            "ascent_ft": loop.get("ascent_ft") or probe_row.get("ascent_ft"),
            "grade_adjusted_miles": loop.get("grade_adjusted_miles") or probe_row.get("grade_adjusted_miles"),
            "estimated_moving_minutes_p75": probe_row.get("moving_effort_p75_minutes"),
            "elevation_source": "dem",
        }
        candidate["validation"] = {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
            "trailhead_snap_confidence": "forced_anchor_probe",
        }
        trailhead = candidate.get("trailhead") or {}
        trailhead["name"] = loop.get("trailhead")
        trailhead["has_parking"] = True
        trailhead["parking_confidence"] = probe_row.get("parking_confidence")
        trailhead["source"] = probe_row.get("anchor_source")
        candidate["trailhead"] = trailhead
        self._forced_candidate_cache[key] = candidate
        return candidate

    def candidate_for_loop(self, loop: dict[str, Any]) -> dict[str, Any]:
        source = loop.get("source")
        candidate_id = str(loop.get("candidate_id") or "")
        if source == "personal_route_menu":
            return self.personal_candidates[candidate_id]
        if source == "hybrid_candidate_index":
            candidate = self.hybrid_candidates.get(candidate_id)
            if candidate:
                return candidate
            return self.rebuild_missing_hybrid_candidate(loop)
        if source == "forced_anchor_probe":
            return self.forced_candidate(loop)
        if source == "canonical_field_menu":
            cue = (self.base_map_data.get("route_cues") or {}).get(candidate_id)
            if not cue:
                raise KeyError(f"Canonical field-menu cue not found: {candidate_id}")
            return {
                "candidate_id": candidate_id,
                "trail_names": cue.get("title", "").split(", ") if cue.get("title") else loop.get("trail_names") or [],
                "trailhead": cue.get("trailhead") or {"name": loop.get("trailhead")},
                "segments": cue.get("segments") or [],
                "segment_ids": [segment.get("seg_id") for segment in cue.get("segments") or []],
                "route_status": cue.get("route_status") or "graph_validated",
                "official_new_miles": cue.get("official_miles"),
                "estimated_total_on_foot_miles": cue.get("on_foot_miles"),
                "total_minutes": cue.get("total_minutes"),
                "raw_total_minutes": cue.get("raw_total_minutes"),
                "time_estimates_minutes": cue.get("time_estimates_minutes"),
                "effort": cue.get("effort"),
                "validation": cue.get("validation"),
            }
        raise ValueError(f"Unsupported loop source: {source}")

    def track_segments_for_loop(self, loop: dict[str, Any]) -> tuple[list[list[tuple[float, float]]], str]:
        try:
            return day_gpx.loop_track_segments(
                loop,
                personal_candidates=self.personal_candidates,
                hybrid_candidates=self.hybrid_candidates,
                field_packet_gpx=self.field_packet_gpx,
                forced_anchor_gpx=self.forced_anchor_gpx,
                official_index=self.official_index,
                connector_graph=self.connector_graph,
            )
        except KeyError:
            candidate = self.candidate_for_loop(loop)
            return day_gpx.candidate_track(candidate, self.official_index, self.connector_graph), "rebuilt_missing_hybrid_candidate"

    def fallback_runtime(self) -> dict[str, Any]:
        if self._fallback_runtime is None:
            state = load_state(self.args.state_json)
            public_trailheads = load_trailheads_from_geojson(self.args.trailheads_geojson)
            state = merge_planning_trailheads(state, public_trailheads)
            dem_context = load_dem_context(self.args.dem_tif, self.args.dem_summary_json)
            performance_profile = build_performance_profile(
                state=state,
                strava_activity_details_dir=self.args.strava_activity_details_dir,
                activity_summary_csv=self.args.activity_summary_csv,
                activity_detail_summary_csv=self.args.activity_detail_summary_csv,
                segment_perf_csv=self.args.segment_perf_csv,
            )
            self._fallback_runtime = {
                "state": state,
                "sampler": dem_context["sampler"],
                "performance_profile": performance_profile,
            }
        return self._fallback_runtime

    def replacement_runtime(self) -> dict[str, Any]:
        if self._replacement_runtime is None:
            runtime = self.fallback_runtime()
            anchors, _summary = multi_start.load_parking_anchors(
                public_trailheads_geojson=self.args.trailheads_geojson,
                private_parking_anchors_geojson=self.args.private_parking_anchors_geojson,
                manual_design_jsons=multi_start.DEFAULT_MANUAL_DESIGN_JSONS,
            )
            connector_geojson = read_json(self.args.connector_geojson)
            street_probe_anchors = accepted_replacement_street_probe_anchors(
                self.accepted_replacements.records,
                connector_geojson,
                self.official_segments_by_id,
            )
            parking_review_decisions = multi_start.load_parking_review_decisions(
                multi_start.DEFAULT_PARKING_REVIEW_DECISIONS_JSON
            )
            anchors = multi_start.apply_parking_review_decisions(
                multi_start.merge_parking_anchors([*anchors, *street_probe_anchors]),
                parking_review_decisions,
            )
            self._replacement_runtime = {
                **runtime,
                "anchors": anchors,
            }
        return self._replacement_runtime

    def accepted_replacement_anchor(self, record: dict[str, Any]) -> dict[str, Any]:
        runtime = self.replacement_runtime()
        for anchor in runtime["anchors"]:
            if anchor_refs_match(anchor.get("anchor_id"), record.get("accepted_anchor_ref")):
                accepted = copy.deepcopy(anchor)
                if record.get("public_anchor_label"):
                    accepted["name"] = record["public_anchor_label"]
                return accepted
        raise ValueError(f"Accepted replacement anchor not found: {record.get('accepted_anchor_ref')}")

    def candidate_for_accepted_replacement(self, record: dict[str, Any], loop: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = self.replacement_runtime()
        anchor = self.accepted_replacement_anchor(record)
        segment_ids = [int(value) for value in replacement_segment_ids(record.get("target_segment_ids"))]
        candidate = multi_start.best_candidate_for_component(
            segment_ids=segment_ids,
            anchor=anchor,
            base_state=runtime["state"],
            performance_profile=runtime["performance_profile"],
            connector_graph=self.connector_graph,
            elevation_sampler=runtime["sampler"],
            segments_by_id=self.official_segments_by_id,
            preferred_trail_order=list(record.get("preferred_trail_order") or loop.get("trail_names") or []),
        )
        if not candidate:
            raise ValueError(f"Could not generate accepted replacement candidate: {record.get('replacement_id')}")
        candidate["candidate_id"] = str(record.get("replacement_candidate_id") or candidate.get("candidate_id") or loop.get("candidate_id"))
        candidate["route_status"] = str(record.get("route_card_status") or "provisional_re_anchored")
        candidate["accepted_replacement_id"] = record.get("replacement_id")
        candidate["accepted_replacement_status"] = record.get("status")
        candidate["accepted_anchor_ref"] = record.get("accepted_anchor_ref")
        candidate["route_card_status"] = record.get("route_card_status")
        candidate["packet_visibility"] = record.get("packet_visibility")
        candidate["certified_route_card"] = record.get("certified_route_card")
        candidate["requires_field_walkthrough"] = record.get("requires_field_walkthrough")
        if record.get("start_justification"):
            candidate["start_justification"] = record.get("start_justification")
        trailhead = candidate.setdefault("trailhead", {})
        trailhead["name"] = record.get("public_anchor_label") or trailhead.get("name")
        trailhead["accepted_anchor_ref"] = record.get("accepted_anchor_ref")
        endpoint = route_endpoint_distance_for_replacement(
            record=record,
            anchor=anchor,
            official_index=self.official_index,
        )
        candidate.update(endpoint)
        return candidate, endpoint

    def track_segments_for_candidate(self, candidate: dict[str, Any]) -> tuple[list[list[tuple[float, float]]], str]:
        return day_gpx.candidate_track(candidate, self.official_index, self.connector_graph), "accepted_replacement_candidate"

    def rebuild_missing_hybrid_candidate(self, loop: dict[str, Any]) -> dict[str, Any]:
        runtime = self.fallback_runtime()
        candidate_id = str(loop.get("candidate_id") or "")
        trail_names = list(loop.get("trail_names") or [])
        ordered_names = sorted(
            trail_names,
            key=lambda name: (
                candidate_id.find(slugify(str(name))) if slugify(str(name)) in candidate_id else 9999,
                trail_names.index(name),
            ),
        )
        trails = []
        for name in ordered_names:
            trail = self.trail_by_name.get(normalize_trail_name(str(name)))
            if not trail:
                raise KeyError(f"Trail not found for fallback candidate: {name}")
            trails.append(trail)
        candidate = candidate_from_trail_group(
            trails,
            runtime["state"],
            runtime["performance_profile"],
            self.connector_graph,
            candidate_type="field_day_missing_hybrid_rebuild",
            elevation_sampler=runtime["sampler"],
        )
        candidate["candidate_id"] = candidate_id
        candidate["route_status"] = "graph_validated"
        candidate["official_new_miles"] = loop.get("official_miles") or candidate.get("official_new_miles")
        candidate["estimated_total_on_foot_miles"] = loop.get("on_foot_miles") or candidate.get("estimated_total_on_foot_miles")
        candidate["total_minutes"] = loop.get("p75_minutes") or candidate.get("total_minutes")
        candidate["time_estimates_minutes"] = {
            **(candidate.get("time_estimates_minutes") or {}),
            "door_to_door_p75": loop.get("p75_minutes"),
            "door_to_door_p90": loop.get("p90_minutes"),
            "recommended_door_to_door": loop.get("p75_minutes"),
        }
        candidate["effort"] = {
            **(candidate.get("effort") or {}),
            "ascent_ft": loop.get("ascent_ft"),
            "grade_adjusted_miles": loop.get("grade_adjusted_miles"),
            "elevation_source": "dem",
        }
        return candidate


def build_promoted_map_data(
    *,
    calendar: dict[str, Any],
    base_map_data: dict[str, Any],
    field_tool_payload: dict[str, Any],
    segment_promotions_payload: dict[str, Any] | None = None,
    context: PromotionContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    card_index = build_route_card_index(field_tool_payload)
    segment_promotions_payload = segment_promotions_payload or {}
    target_segment_promotions = promoted_segment_rows_by_target(segment_promotions_payload)
    removed_source_targets = removed_source_loop_targets(segment_promotions_payload)
    official_by_id = {str(segment["seg_id"]): segment for segment in context.official_segments}
    packages = []
    route_cues: dict[str, dict[str, Any]] = {}
    route_features = []
    parking_features = []
    logistics_features = []
    validations = []
    promotions = []
    promotion_route_number = 1
    previously_certified_count = 0
    newly_promoted_count = 0
    superset_overrides = superset_route_overrides(
        calendar=calendar,
        context=context,
        field_tool_payload=field_tool_payload,
        card_index=card_index,
        accepted_replacements=context.accepted_replacements,
    )
    superset_replacement_count = 0
    skipped_source_loop_count = 0

    for assignment in calendar.get("assignments") or []:
        day = copy.deepcopy(assignment.get("field_day") or {})
        day["date"] = assignment.get("date")
        components = []
        for loop_index, raw_loop in enumerate(day.get("loops") or []):
            loop = copy.deepcopy(raw_loop)
            loop["draft_day_number"] = day.get("draft_day_number")
            source_removal = removed_source_targets.get(str(loop.get("candidate_id") or ""))
            if source_removal:
                skipped_source_loop_count += 1
                promotions.append(
                    {
                        "rank": len(promotions) + 1,
                        "date": assignment.get("date"),
                        "draft_day_number": day.get("draft_day_number"),
                        "loop_id": loop.get("loop_id"),
                        "source": loop.get("source"),
                        "source_candidate_id": loop.get("candidate_id"),
                        "route_card_candidate_id": source_removal["target_candidate_id"],
                        "label": loop.get("label"),
                        "trailhead": loop.get("trailhead"),
                        "official_miles": loop.get("official_miles"),
                        "on_foot_miles": loop.get("on_foot_miles"),
                        "p75_minutes": loop.get("p75_minutes"),
                        "p90_minutes": loop.get("p90_minutes"),
                        "mode": "removed_source_loop_after_segment_ownership_promotion",
                        "skipped_route_card_source": True,
                        "reassigned_segment_ids": source_removal["segment_ids"],
                        "reassigned_to_candidate_id": source_removal["target_candidate_id"],
                        "reasons": source_removal["reasons"],
                    }
                )
                continue
            loop["promotion_route_number"] = promotion_route_number
            candidate = context.candidate_for_loop(loop)
            accepted_replacement = context.accepted_replacements.match_for_loop(loop, candidate)
            replacement_endpoint: dict[str, Any] | None = None
            replacement_deltas: dict[str, Any] | None = None
            if accepted_replacement and accepted_replacement.get("status") == ACTIVE_STATUS:
                existing_route = None
                candidate, replacement_endpoint = context.candidate_for_accepted_replacement(accepted_replacement, loop)
                baseline = route_metrics(accepted_replacement.get("baseline_card_ref") or {})
                replacement_deltas = dominance_deltas(baseline, candidate_metrics(candidate))
                blockers = [replacement_preservation_blocker(accepted_replacement)]
            elif accepted_replacement and accepted_replacement.get("status") == INVESTIGATE_STATUS:
                existing_route = None
                blockers = [replacement_preservation_blocker(accepted_replacement)]
            else:
                existing_route = superset_overrides.get(str(loop.get("loop_id"))) or find_route_card(
                    loop,
                    card_index,
                    context.accepted_replacements,
                )
                blockers = route_card_certification_blockers(existing_route, context.args.field_packet_dir) if existing_route else []
            use_existing = bool(existing_route and not blockers)
            used_superset_override = str(loop.get("loop_id")) in superset_overrides
            route_candidate_id = str((existing_route or {}).get("candidate_ids", [None])[0] or candidate.get("candidate_id") or loop.get("candidate_id"))
            label = label_for_loop(loop, existing_route if use_existing else None, loop_index)
            component = route_component(
                loop=loop,
                candidate_id=route_candidate_id,
                candidate=candidate,
                existing_route=existing_route if use_existing else None,
                label=label,
            )
            if accepted_replacement and accepted_replacement.get("status") in BLOCKING_STATUSES:
                apply_replacement_metadata_to_component(
                    component,
                    accepted_replacement,
                    deltas=replacement_deltas,
                    endpoint=replacement_endpoint,
                )
            cue = (
                copy.deepcopy(context.base_map_data["route_cues"][route_candidate_id])
                if use_existing and route_candidate_id in (context.base_map_data.get("route_cues") or {})
                else None
            )
            applied_segment_promotions = target_segment_promotions.get(route_candidate_id, [])
            if applied_segment_promotions:
                apply_segment_promotions_to_route_card(
                    component=component,
                    cue=cue,
                    promotions=applied_segment_promotions,
                    base_map_data=base_map_data,
                    official_by_id=official_by_id,
                )
            components.append(component)
            props = {
                "kind": "route",
                "route_number": promotion_route_number,
                "package_number": 100 + int(day.get("draft_day_number") or 0),
                "candidate_id": route_candidate_id,
                "block_name": f"Field Day {day.get('draft_day_number')} route-card bundle",
                "title": ", ".join(component.get("trail_names") or []),
                "official_miles": component.get("official_miles"),
                "on_foot_miles": component.get("on_foot_miles"),
                "trailhead": component.get("trailhead"),
                "color": PALETTE[(promotion_route_number - 1) % len(PALETTE)],
                "field_menu_label": label,
            }

            if use_existing and cue is not None:
                sync_missing_cue_trailhead_metadata(cue, candidate)
                route_cues[route_candidate_id] = cue
                route_features.extend(copy_feature_group(context.route_features_by_candidate, route_candidate_id, props))
                parking_features.extend(copy_feature_group(context.parking_features_by_candidate, route_candidate_id, props))
                logistics_features.extend(copy_feature_group(context.logistics_features_by_candidate, route_candidate_id, props))
                track_segments, track_source = context.track_segments_for_loop({**loop, "source": "canonical_field_menu", "candidate_id": route_candidate_id})
                previously_certified_count += 1
                if used_superset_override:
                    superset_replacement_count += 1
            else:
                if accepted_replacement and accepted_replacement.get("status") == ACTIVE_STATUS:
                    track_segments, track_source = context.track_segments_for_candidate(candidate)
                else:
                    track_segments, track_source = context.track_segments_for_loop(loop)
                flat_parts = [part for part in track_segments if len(part) >= 2]
                if len(flat_parts) == 1:
                    rendered_parts = split_coords_on_gaps(flat_parts[0], max_gap_miles=0.1)
                else:
                    rendered_parts = flat_parts
                route_feature = multiline_feature(rendered_parts, props)
                if route_feature:
                    route_features.append(route_feature)
                parking = parking_feature(candidate, props)
                if parking:
                    parking_features.append(parking)
                cue = route_cue(candidate, {**component, "candidate_id": route_candidate_id})
                cue["candidate_id"] = route_candidate_id
                if accepted_replacement and accepted_replacement.get("status") in BLOCKING_STATUSES:
                    apply_replacement_metadata_to_cue(cue, accepted_replacement, endpoint=replacement_endpoint)
                if applied_segment_promotions:
                    apply_segment_promotions_to_route_card(
                        component=component,
                        cue=cue,
                        promotions=applied_segment_promotions,
                        base_map_data=base_map_data,
                        official_by_id=official_by_id,
                    )
                cue["logistics"], new_logistics = logistics_for_candidate(
                    route_candidate_id,
                    candidate,
                    props,
                    rendered_parts,
                )
                route_cues[route_candidate_id] = cue
                logistics_features.extend(new_logistics)
                newly_promoted_count += 1

            validation = validate_track_segments(track_segments, max_gap_miles=0.1)
            validations.append(
                {
                    "candidate_id": route_candidate_id,
                    "source_loop_id": loop.get("loop_id"),
                    "track_source": track_source,
                    "source_gap_warning": not validation["passed"],
                    "source_max_gap_miles": validation.get("max_trackpoint_gap_miles"),
                    "rendered_passed": validation["passed"],
                    "rendered_failures": validation.get("failures") or [],
                }
            )
            promotions.append(
                {
                    "rank": len(promotions) + 1,
                    "date": assignment.get("date"),
                    "draft_day_number": day.get("draft_day_number"),
                    "loop_id": loop.get("loop_id"),
                    "source": loop.get("source"),
                    "source_candidate_id": loop.get("candidate_id"),
                    "route_card_candidate_id": route_candidate_id,
                    "label": label,
                    "trailhead": component.get("trailhead"),
                    "official_miles": component.get("official_miles"),
                    "on_foot_miles": component.get("on_foot_miles"),
                    "p75_minutes": component.get("total_minutes"),
                    "p90_minutes": (component.get("time_estimates_minutes") or {}).get("door_to_door_p90"),
                    "mode": (
                        "preserved_existing_certified_superset_replacement"
                        if used_superset_override
                        else "preserved_existing_certified_card"
                        if use_existing
                        else "promoted_candidate_to_route_card_source"
                    ),
                    "superset_replacement": used_superset_override,
                    "track_source": track_source,
                    "certification_blockers_before_promotion": blockers,
                    "accepted_replacement_id": (accepted_replacement or {}).get("replacement_id"),
                    "accepted_replacement_status": (accepted_replacement or {}).get("status"),
                    "dominance_checked": bool(accepted_replacement),
                    "preservation_blocked_reason": (
                        replacement_preservation_blocker(accepted_replacement)
                        if accepted_replacement and accepted_replacement.get("status") in BLOCKING_STATUSES
                        else None
                    ),
                    "route_card_status": component.get("route_card_status"),
                    "packet_visibility": component.get("packet_visibility"),
                    "certified_route_card": component.get("certified_route_card"),
                    "requires_field_walkthrough": component.get("requires_field_walkthrough"),
                    "cue_generation_mode": cue.get("cue_generation_mode") if cue else None,
                    "anchor_to_credit_endpoint_distance_miles": component.get("anchor_to_credit_endpoint_distance_miles"),
                    "credit_endpoint_used": component.get("credit_endpoint_used"),
                    "dominance_deltas": replacement_deltas,
                    "segment_ownership_promotions": [
                        promotion.get("segment_id")
                        for promotion in applied_segment_promotions
                    ],
                }
            )
            promotion_route_number += 1
        packages.append(package_for_day(day, components))

    segment_ids = sorted({seg_id for package in packages for seg_id in package.get("segment_ids") or []})
    total_official_miles = round_miles(sum(float(package.get("official_miles") or 0) for package in packages))
    total_on_foot_miles = round_miles(sum(float(package.get("on_foot_miles") or 0) for package in packages))
    accepted_replacement_blocker_count = sum(
        1
        for promotion in promotions
        if promotion.get("accepted_replacement_status") in BLOCKING_STATUSES
    )
    map_data = {
        "schema": "boise_trails_field_day_promoted_route_card_source_v1",
        "run_id": "field-day-loop-promotion-2026-05-11",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "summary": {
            "source": "field_day_loop_promotion",
            "package_count": len(packages),
            "component_route_count": sum(len(package.get("components") or []) for package in packages),
            "covered_segment_count": len(segment_ids),
            "official_miles": total_official_miles,
            "total_on_foot_miles": total_on_foot_miles,
            "planwide_on_foot_to_official_ratio": round(total_on_foot_miles / total_official_miles, 2)
            if total_official_miles
            else None,
            "previously_certified_loop_count": previously_certified_count,
            "newly_promoted_loop_count": newly_promoted_count,
            "superset_replacement_loop_count": superset_replacement_count,
            "skipped_source_loop_count": skipped_source_loop_count,
            "accepted_replacement_blocker_count": accepted_replacement_blocker_count,
            "segment_ownership_promotion_count": sum(
                len(promotions)
                for promotions in target_segment_promotions.values()
            ),
        },
        "progress": copy.deepcopy(base_map_data.get("progress") or {}),
        "packages": packages,
        "feature_collections": {
            "routes": {"type": "FeatureCollection", "features": route_features},
            "official_segments": {"type": "FeatureCollection", "features": []},
            "parking": {"type": "FeatureCollection", "features": parking_features},
            "logistics": {"type": "FeatureCollection", "features": logistics_features},
        },
        "route_cues": route_cues,
        "map_validation": {
            "rendered_passed": all(item["rendered_passed"] for item in validations),
            "source_gap_warning_count": sum(1 for item in validations if item["source_gap_warning"]),
            "route_validations": validations,
        },
        "source_files": {
            "calendar_assignment_json": display_path(context.args.calendar_json),
            "base_map_data_json": display_path(context.args.base_map_data_json),
            "field_tool_json": display_path(context.args.field_tool_json),
            "accepted_replacements_json": display_path(context.args.accepted_replacements_json),
            "segment_promotions_json": display_path(context.args.segment_promotions_json),
            "personal_route_menu_json": display_path(context.args.personal_route_menu_json),
            "hybrid_route_pass_json": display_path(context.args.hybrid_route_pass_json),
            "forced_anchor_probe_json": display_path(context.args.forced_anchor_probe_json),
        },
    }
    sync_official_segment_features(map_data, context.official_segments)
    enrich_official_repeat_segment_ids(map_data)
    report = {
        "schema": "boise_trails_field_day_loop_promotion_report_v1",
        "generated_at": map_data["generated_at"],
        "objective": "Promote every selected field-day loop into an auditable route-card source unit.",
        "summary": {
            **map_data["summary"],
            "route_card_source_loop_count": len(promotions),
            "track_validation_passed": map_data["map_validation"]["rendered_passed"],
            "source_gap_warning_count": map_data["map_validation"]["source_gap_warning_count"],
            "route_card_promotion_gap_after_source_promotion": 0,
            "superset_replacement_loop_count": superset_replacement_count,
            "skipped_source_loop_count": skipped_source_loop_count,
            "accepted_replacement_blocker_count": accepted_replacement_blocker_count,
            "segment_ownership_promotion_count": sum(
                len(promotions)
                for promotions in target_segment_promotions.values()
            ),
        },
        "ranking_policy": [
            "Rank selected loops in field execution order so earliest scheduled loops are certified first if the run is interrupted.",
            "Preserve already certified route cards only after active or investigate accepted replacements have been checked.",
            "Promote remaining selected candidates only when stored geometry or generated forced-anchor GPX validates.",
        ],
        "source_files": map_data["source_files"],
        "promotions": promotions,
        "validation": map_data["map_validation"],
    }
    return map_data, report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Field-Day Loop Promotion",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "Objective: promote every selected field-day loop into the canonical route-card source so the phone packet can certify each loop instead of showing route-card promotion gaps.",
        "",
        "## Summary",
        "",
        f"- Route-card source loops: {summary.get('route_card_source_loop_count', 0)}",
        f"- Previously certified loops preserved: {summary.get('previously_certified_loop_count', 0)}",
        f"- Newly promoted loops: {summary.get('newly_promoted_loop_count', 0)}",
        f"- Certified superset replacements: {summary.get('superset_replacement_loop_count', 0)}",
        f"- Accepted replacement blockers: {summary.get('accepted_replacement_blocker_count', 0)}",
        f"- Covered official segments: {summary.get('covered_segment_count', 0)}",
        f"- Total p75 source minutes: {sum(int(row.get('p75_minutes') or 0) for row in report.get('promotions') or [])}",
        f"- Track validation passed: `{summary.get('track_validation_passed')}`",
        f"- Source gap warnings: {summary.get('source_gap_warning_count', 0)}",
        "",
        "## Why Promote",
        "",
        "There is no route-quality reason to leave these loops unpromoted once access, timing, geometry, and coverage evidence exists. The blocker was a source-shape gap: selected Field Day candidates were not materialized as canonical route-card source records.",
        "",
        "## Promotion Queue",
        "",
        "| Rank | Date | Day | Label | Source | Candidate | Trailhead | P75 | Mode |",
        "|---:|---|---:|---|---|---|---|---:|---|",
    ]
    for row in report.get("promotions") or []:
        lines.append(
            "| {rank} | {date} | {day} | `{label}` | `{source}` | `{candidate}` | {trailhead} | {p75} | {mode} |".format(
                rank=row.get("rank"),
                date=row.get("date"),
                day=row.get("draft_day_number"),
                label=row.get("label"),
                source=row.get("source"),
                candidate=row.get("source_candidate_id"),
                trailhead=row.get("trailhead") or "",
                p75=row.get("p75_minutes") or "",
                mode=row.get("mode"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_promoted_map_views(map_data: dict[str, Any], map_html_path: Path, menu_md_path: Path) -> None:
    map_html_path.parent.mkdir(parents=True, exist_ok=True)
    menu_md_path.parent.mkdir(parents=True, exist_ok=True)
    map_html_path.write_text(render_package_map_html(map_data), encoding="utf-8")
    menu_md_path.write_text(render_outing_menu_markdown(map_data, map_html_path), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-json", type=Path, default=DEFAULT_CALENDAR_JSON)
    parser.add_argument("--base-map-data-json", type=Path, default=DEFAULT_BASE_MAP_DATA_JSON)
    parser.add_argument("--field-tool-json", type=Path, default=DEFAULT_FIELD_TOOL_JSON)
    parser.add_argument("--segment-promotions-json", type=Path, default=DEFAULT_SEGMENT_PROMOTIONS_JSON)
    parser.add_argument("--field-packet-dir", type=Path, default=REPO_ROOT / "docs" / "field-packet")
    parser.add_argument("--personal-route-menu-json", type=Path, default=DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    parser.add_argument("--hybrid-route-pass-json", type=Path, default=DEFAULT_HYBRID_ROUTE_PASS_JSON)
    parser.add_argument("--forced-anchor-probe-json", type=Path, default=DEFAULT_FORCED_ANCHOR_PROBE_JSON)
    parser.add_argument("--forced-anchor-gpx-manifest-json", type=Path, default=DEFAULT_FORCED_ANCHOR_GPX_MANIFEST_JSON)
    parser.add_argument("--field-packet-manifest-json", type=Path, default=DEFAULT_FIELD_PACKET_MANIFEST_JSON)
    parser.add_argument("--accepted-replacements-json", type=Path, default=DEFAULT_ACCEPTED_REPLACEMENTS_JSON)
    parser.add_argument("--private-parking-anchors-geojson", type=Path, default=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON)
    parser.add_argument("--state-json", type=Path, default=forced_gpx.DEFAULT_STATE_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--trailheads-geojson", type=Path, default=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON)
    parser.add_argument("--strava-activity-details-dir", type=Path, default=DEFAULT_STRAVA_DETAILS_DIR)
    parser.add_argument("--activity-summary-csv", type=Path, default=DEFAULT_ACTIVITY_SUMMARY_CSV)
    parser.add_argument("--activity-detail-summary-csv", type=Path, default=DEFAULT_ACTIVITY_DETAIL_SUMMARY_CSV)
    parser.add_argument("--segment-perf-csv", type=Path, default=DEFAULT_SEGMENT_PERF_CSV)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--output-map-data-json", type=Path, default=DEFAULT_OUTPUT_MAP_DATA_JSON)
    parser.add_argument("--output-map-html", type=Path, default=DEFAULT_OUTPUT_MAP_HTML)
    parser.add_argument("--output-menu-md", type=Path, default=DEFAULT_OUTPUT_MENU_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calendar = read_json(args.calendar_json)
    base_map_data = read_json(args.base_map_data_json)
    field_tool_payload = read_json(args.field_tool_json)
    segment_promotions_payload = (
        read_json(args.segment_promotions_json)
        if args.segment_promotions_json.exists()
        else {"promotions": []}
    )
    context = PromotionContext(args, base_map_data)
    map_data, report = build_promoted_map_data(
        calendar=calendar,
        base_map_data=base_map_data,
        field_tool_payload=field_tool_payload,
        segment_promotions_payload=segment_promotions_payload,
        context=context,
    )
    write_json(args.output_map_data_json, map_data)
    write_promoted_map_views(map_data, args.output_map_html, args.output_menu_md)
    write_json(args.report_json, report)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_map_data_json}")
    print(f"Wrote {args.output_map_html}")
    print(f"Wrote {args.output_menu_md}")
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
