#!/usr/bin/env python3
"""Audit route repeat accounting and optimization pressure in field artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from field_activity_review import (  # noqa: E402
    activity_coordinates,
    normalized_ids,
    review_activity_against_segments,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_CONNECTOR_GEOJSON,
    DEFAULT_DEM_SUMMARY_JSON,
    DEFAULT_DEM_TIF,
    DEFAULT_OFFICIAL_GEOJSON,
    bbox_overlaps,
    coordinate_bbox,
    downsample_coords,
    haversine_miles,
    load_connector_graph,
    load_dem_context,
    load_official_segments,
    point_to_polyline_distance_miles,
    shortest_connector_path,
)


DEFAULT_FIELD_TOOL_DATA_JSON = REPO_ROOT / "docs" / "field-packet" / "field-tool-data.json"
DEFAULT_PACKET_DIR = REPO_ROOT / "docs" / "field-packet"
DEFAULT_OUTPUT_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.json"
DEFAULT_OUTPUT_MD = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12.md"
DEFAULT_MANIFEST_JSON = YEAR_DIR / "checkpoints" / "route-repeat-optimization-audit-2026-05-12-manifest.json"
DEFAULT_ROUTE_PROOF_JSONS = [
    YEAR_DIR / "checkpoints" / "adversarial-route-disproof-2026-05-16.json",
    YEAR_DIR / "checkpoints" / "all-route-adversarial-disproof-2026-05-16.json",
]
DEFAULT_TAIL_OPPORTUNITY_ENDPOINT_THRESHOLD_MILES = 0.03
DEFAULT_TAIL_OPPORTUNITY_MAX_SEGMENT_MILES = 0.35

NON_CREDIT_CUE_TYPES = {
    "start_access",
    "official_segment_start",
    "connector_named_trail",
    "connector_road",
    "repeat_official_noncredit",
    "overlap_repeat",
    "exit_access",
    "return_to_car",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def route_key(route: dict[str, Any]) -> str:
    return str(route.get("outing_id") or route.get("label") or route.get("block_name") or "unknown-route")


def route_label(route: dict[str, Any]) -> str:
    label = str(route.get("label") or route.get("block_name") or route_key(route))
    outing_id = route.get("outing_id")
    if outing_id and str(outing_id) not in label:
        return f"{outing_id}: {label}"
    return label


def route_audit_gpx_path(route: dict[str, Any], packet_dir: Path) -> Path | None:
    href = route.get("audit_gpx_href")
    if not href and route.get("gpx_href"):
        href = "gpx/audit/" + Path(str(route["gpx_href"])).name
    if not href:
        return None
    path = Path(str(href))
    if path.is_absolute():
        return path
    return packet_dir / path


def build_segment_index(official_segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(segment["seg_id"]): segment for segment in official_segments}


def segment_brief(segment_index: dict[str, dict[str, Any]], segment_id: str) -> dict[str, Any]:
    segment = segment_index.get(str(segment_id), {})
    return {
        "seg_id": str(segment_id),
        "seg_name": segment.get("seg_name"),
        "trail_name": segment.get("trail_name"),
        "direction": segment.get("direction"),
        "official_miles": round(float(segment.get("official_miles") or 0), 2),
    }


def sort_id(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def route_index(routes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for route in routes:
        keys = [
            route_key(route),
            str(route.get("outing_id") or ""),
            str(route.get("label") or ""),
            route_label(route),
        ]
        for candidate_id in route.get("candidate_ids") or []:
            keys.append(str(candidate_id))
        for key in keys:
            if key:
                index[key] = route
    return index


def route_claims(routes: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        route_key(route): set(normalized_ids(route.get("segment_ids") or []))
        for route in routes
    }


def segment_endpoints(segment: dict[str, Any] | None) -> list[tuple[float, float]]:
    if not segment:
        return []
    coords = segment.get("coordinates") or []
    if coords:
        return [tuple(coords[0]), tuple(coords[-1])]
    endpoints = []
    for key in ("start", "end"):
        point = segment.get(key)
        if point:
            endpoints.append(tuple(point))
    return endpoints


def min_endpoint_gap_miles(
    segment_index: dict[str, dict[str, Any]],
    left_segment_id: str,
    right_segment_id: str,
) -> float | None:
    left_points = segment_endpoints(segment_index.get(str(left_segment_id)))
    right_points = segment_endpoints(segment_index.get(str(right_segment_id)))
    if not left_points or not right_points:
        return None
    return min(haversine_miles(left, right) for left in left_points for right in right_points)


def owner_route_refs(owner_rows: list[dict[str, Any]], routes_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for owner in owner_rows:
        owner_route = None
        for key in (
            str(owner.get("outing_id") or ""),
            str(owner.get("label") or ""),
            str(owner.get("route_key") or ""),
        ):
            if key and key in routes_by_key:
                owner_route = routes_by_key[key]
                break
        refs.append(
            {
                "route_key": route_key(owner_route) if owner_route else str(owner.get("outing_id") or owner.get("label") or ""),
                "outing_id": owner.get("outing_id") or (owner_route or {}).get("outing_id"),
                "label": owner.get("label") or ((owner_route and route_label(owner_route)) or None),
                "candidate_ids": owner.get("candidate_ids") or ((owner_route or {}).get("candidate_ids") or []),
                "found_in_field_tool_data": owner_route is not None,
            }
        )
    return refs


def cue_pair_context(route: dict[str, Any], repeated_segment_id: str, adjacent_segment_id: str) -> dict[str, Any]:
    repeat_cues = []
    adjacent_cues = []
    for cue in route.get("wayfinding_cues") or []:
        seq = cue.get("seq")
        cue_type = cue.get("cue_type")
        row = {"seq": seq, "cue_type": cue_type}
        if str(repeated_segment_id) in normalized_ids(cue.get("official_repeat_segment_ids") or []):
            repeat_cues.append(row)
        if str(adjacent_segment_id) in normalized_ids(cue.get("official_segment_ids") or []):
            adjacent_cues.append(row)
    nearest = None
    for repeat_cue in repeat_cues:
        for adjacent_cue in adjacent_cues:
            if repeat_cue.get("seq") is None or adjacent_cue.get("seq") is None:
                continue
            try:
                distance = abs(float(repeat_cue["seq"]) - float(adjacent_cue["seq"]))
            except (TypeError, ValueError):
                continue
            candidate = {
                "repeat_cue_seq": repeat_cue["seq"],
                "repeat_cue_type": repeat_cue.get("cue_type"),
                "adjacent_cue_seq": adjacent_cue["seq"],
                "adjacent_cue_type": adjacent_cue.get("cue_type"),
                "cue_seq_delta": distance,
            }
            if nearest is None or distance < nearest["cue_seq_delta"]:
                nearest = candidate
    return {
        "repeat_cues": repeat_cues,
        "adjacent_credit_cues": adjacent_cues,
        "nearest_cue_pair": nearest,
    }


def cross_route_tail_opportunities(
    routes: list[dict[str, Any]],
    segment_index: dict[str, dict[str, Any]],
    *,
    endpoint_threshold_miles: float = DEFAULT_TAIL_OPPORTUNITY_ENDPOINT_THRESHOLD_MILES,
    max_segment_miles: float = DEFAULT_TAIL_OPPORTUNITY_MAX_SEGMENT_MILES,
) -> list[dict[str, Any]]:
    """Find split-route pressure where a repeated owned segment touches a small claimed edge.

    This is an optimization warning, not a route-certification decision. It says the
    owner route might have been able to carry a short adjacent edge, reducing the
    receiver route's later credit or repeat burden.
    """

    routes_by_key = route_index(routes)
    claims_by_route = route_claims(routes)
    rows = []
    seen = set()
    for receiver in routes:
        receiver_key = route_key(receiver)
        receiver_claimed_ids = set(normalized_ids(receiver.get("segment_ids") or []))
        reconciliation = receiver.get("segment_ownership_reconciliation") or {}
        for owned_segment in reconciliation.get("segments_owned_elsewhere") or []:
            repeated_segment_id = str(owned_segment.get("seg_id") or "")
            if not repeated_segment_id or repeated_segment_id not in segment_index:
                continue
            owner_refs = owner_route_refs(owned_segment.get("owned_by_routes") or [], routes_by_key)
            owner_refs_without_adjacent = []
            for adjacent_segment_id in receiver_claimed_ids:
                if adjacent_segment_id == repeated_segment_id or adjacent_segment_id not in segment_index:
                    continue
                adjacent_miles = float_value(segment_index[adjacent_segment_id].get("official_miles"))
                if adjacent_miles > max_segment_miles:
                    continue
                endpoint_gap = min_endpoint_gap_miles(segment_index, repeated_segment_id, adjacent_segment_id)
                if endpoint_gap is None or endpoint_gap > endpoint_threshold_miles:
                    continue
                owner_refs_without_adjacent = [
                    owner
                    for owner in owner_refs
                    if adjacent_segment_id not in claims_by_route.get(str(owner.get("route_key") or ""), set())
                ]
                if not owner_refs_without_adjacent:
                    continue
                key = (
                    receiver_key,
                    repeated_segment_id,
                    adjacent_segment_id,
                    tuple(sorted(str(owner.get("route_key") or "") for owner in owner_refs_without_adjacent)),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "code": "cross_route_tail_opportunity",
                        "severity": "optimization_warning",
                        "receiver_route_key": receiver_key,
                        "receiver_label": route_label(receiver),
                        "receiver_outing_id": receiver.get("outing_id"),
                        "receiver_trailhead": receiver.get("trailhead"),
                        "repeated_owned_segment": segment_brief(segment_index, repeated_segment_id),
                        "adjacent_candidate_segment": segment_brief(segment_index, adjacent_segment_id),
                        "owner_routes": owner_refs_without_adjacent,
                        "endpoint_gap_miles": round(endpoint_gap, 4),
                        "adjacent_segment_miles": round(adjacent_miles, 2),
                        "worst_case_out_and_back_added_miles": round(adjacent_miles * 2, 2),
                        "cue_context": cue_pair_context(receiver, repeated_segment_id, adjacent_segment_id),
                        "message": (
                            "A later route repeats an official segment owned elsewhere next to a short claimed "
                            "segment. Review whether the owner route should carry that adjacent edge instead."
                        ),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            -float_value((row.get("repeated_owned_segment") or {}).get("official_miles")),
            float_value(row.get("adjacent_segment_miles")),
            row.get("receiver_label") or "",
            (row.get("adjacent_candidate_segment") or {}).get("seg_id") or "",
        ),
    )


def candidate_segments_for_activity(
    activity_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    include_ids: set[str],
    threshold_miles: float,
) -> list[dict[str, Any]]:
    if len(activity_coords) < 2:
        return official_segments
    activity_bbox = coordinate_bbox(activity_coords)
    origin_lat = sum(point[1] for point in activity_coords) / len(activity_coords)
    lat_buffer = threshold_miles / 69.0
    lon_buffer = threshold_miles / max(1e-6, 69.172 * math.cos(math.radians(origin_lat)))
    candidates = []
    for segment in official_segments:
        segment_id = str(segment["seg_id"])
        segment_bbox = segment.get("bbox") or coordinate_bbox(segment["coordinates"])
        if segment_id in include_ids or bbox_overlaps(activity_bbox, segment_bbox, lon_buffer, lat_buffer):
            candidates.append(segment)
    return candidates


def cumulative_points(coords: list[tuple[float, float]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    total = 0.0
    prior = None
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if prior is not None:
            total += haversine_miles(prior, point)
        points.append({"point": point, "mile": total})
        prior = point
    return points


def polyline_distance_miles(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_miles(left, right) for left, right in zip(coords, coords[1:]))


def point_at_mile(points: list[dict[str, Any]], target_mile: float) -> tuple[float, float] | None:
    if not points:
        return None
    target = max(0.0, min(float(target_mile or 0), float(points[-1].get("mile") or 0)))
    prior = points[0]
    for item in points[1:]:
        left_mile = float(prior.get("mile") or 0)
        right_mile = float(item.get("mile") or 0)
        if right_mile >= target:
            if right_mile == left_mile:
                return item["point"]
            ratio = (target - left_mile) / (right_mile - left_mile)
            left = prior["point"]
            right = item["point"]
            return (
                left[0] + (right[0] - left[0]) * ratio,
                left[1] + (right[1] - left[1]) * ratio,
            )
        prior = item
    return points[-1]["point"]


def interval_coordinates(
    activity_coords: list[tuple[float, float]],
    *,
    start_mile: float,
    end_mile: float,
) -> list[tuple[float, float]]:
    points = cumulative_points(activity_coords)
    if not points or end_mile <= start_mile:
        return []
    result = []
    start_point = point_at_mile(points, start_mile)
    end_point = point_at_mile(points, end_mile)
    if start_point:
        result.append(start_point)
    result.extend(
        item["point"]
        for item in points
        if start_mile < float(item.get("mile") or 0) < end_mile
    )
    if end_point:
        result.append(end_point)
    return result


def dedupe_adjacent_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if deduped and haversine_miles(deduped[-1], point) < 0.001:
            continue
        deduped.append(point)
    return deduped


def replace_interval_coordinates(
    activity_coords: list[tuple[float, float]],
    *,
    start_mile: float,
    end_mile: float,
    replacement_coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    points = cumulative_points(activity_coords)
    if not points:
        return activity_coords
    track_total = float(points[-1].get("mile") or 0)
    repaired: list[tuple[float, float]] = []
    repaired.extend(interval_coordinates(activity_coords, start_mile=0.0, end_mile=start_mile))
    repaired.extend(replacement_coords)
    repaired.extend(interval_coordinates(activity_coords, start_mile=end_mile, end_mile=track_total))
    return dedupe_adjacent_coords(repaired)


def official_endpoint_miss_ids(
    coords: list[tuple[float, float]],
    *,
    segment_ids: set[str],
    official_segments: list[dict[str, Any]],
    endpoint_tolerance_miles: float = 0.04,
) -> set[str]:
    if len(coords) < 2:
        return set(segment_ids)
    segment_candidates = [
        segment
        for segment in official_segments
        if str(segment.get("seg_id")) in segment_ids
    ]
    missing: set[str] = set()
    for segment in segment_candidates:
        segment_id = str(segment.get("seg_id"))
        segment_coords = segment.get("coordinates") or []
        if len(segment_coords) < 2:
            continue
        for endpoint in (segment_coords[0], segment_coords[-1]):
            if point_to_polyline_distance_miles(endpoint, coords) > endpoint_tolerance_miles:
                missing.add(segment_id)
                break
    return missing


def official_geometry_miss_ids(
    coords: list[tuple[float, float]],
    *,
    segment_ids: set[str],
    official_segments: list[dict[str, Any]],
    threshold_miles: float = 0.045,
    min_fraction: float = 0.95,
) -> set[str]:
    if len(coords) < 2:
        return set(segment_ids)
    segment_candidates = [
        segment
        for segment in official_segments
        if str(segment.get("seg_id")) in segment_ids
    ]
    missing: set[str] = set()
    for segment in segment_candidates:
        official_coords = densify_coords(segment.get("coordinates") or [], max_gap_miles=0.02)
        if not official_coords:
            continue
        near_count = sum(
            1
            for point in official_coords
            if point_to_polyline_distance_miles(point, coords) <= threshold_miles
        )
        if near_count / len(official_coords) < min_fraction:
            missing.add(str(segment.get("seg_id")))
    return missing


def cue_interval(route: dict[str, Any], cue: dict[str, Any]) -> tuple[float, float] | None:
    if cue.get("route_miles") is None and cue.get("cum_miles") is None:
        return None
    start = float_value(cue.get("route_miles") if cue.get("route_miles") is not None else cue.get("cum_miles"))
    length = float_value(
        cue.get("route_leg_miles") if cue.get("route_leg_miles") is not None else cue.get("leg_miles")
    )
    if length <= 0:
        return None
    return start, start + length


def coordinate_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def cue_source_path_coords(cue: dict[str, Any]) -> list[tuple[float, float]]:
    coords = [
        point
        for point in (coordinate_pair(value) for value in cue.get("source_path_coordinates") or [])
        if point is not None
    ]
    return coords if len(coords) >= 2 else []


def cue_review_coords(
    activity_coords: list[tuple[float, float]],
    cue: dict[str, Any],
) -> list[tuple[float, float]]:
    source_coords = cue_source_path_coords(cue)
    if source_coords:
        return source_coords
    interval = cue_interval({}, cue)
    if not interval:
        return []
    return interval_coordinates(activity_coords, start_mile=interval[0], end_mile=interval[1])


def post_credit_repeat_savings_threshold(current_miles: float) -> float:
    return max(0.10, float(current_miles or 0) * 0.10)


def cue_route_leg_miles(cue: dict[str, Any]) -> float:
    for key in ("route_leg_miles", "leg_miles", "official_repeat_miles"):
        value = float_value(cue.get(key))
        if value > 0:
            return value
    return 0.0


def rounded_point_key(point: tuple[float, float] | None) -> tuple[float, float] | None:
    if point is None:
        return None
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def already_credited_source(repeat_ids: set[str], completed_at_export_ids: set[str]) -> str:
    if repeat_ids and repeat_ids <= completed_at_export_ids:
        return "completed_at_export"
    if repeat_ids & completed_at_export_ids:
        return "mixed"
    return "prior_route_cue"


def cue_credit_segment_ids(cue: dict[str, Any], claimed_segment_ids: set[str]) -> set[str]:
    ids = set(normalized_ids(cue.get("official_segment_ids") or []))
    if ids:
        return ids
    if str(cue.get("cue_type") or "") in {"follow_official_segment", "junction_turn"}:
        return set(claimed_segment_ids)
    return set()


def alternate_path_avoiding_repeats(
    *,
    start_point: tuple[float, float] | None,
    end_point: tuple[float, float] | None,
    repeated_segment_ids: set[str],
    connector_graph: dict[str, Any] | None,
    snap_tolerance_miles: float,
    cache: dict[tuple[Any, ...], dict[str, Any] | None],
) -> dict[str, Any] | None:
    if not connector_graph or start_point is None or end_point is None:
        return None
    avoid_ids = {
        int(segment_id)
        for segment_id in repeated_segment_ids
        if str(segment_id).isdigit()
    }
    key = (
        rounded_point_key(start_point),
        rounded_point_key(end_point),
        round(float(snap_tolerance_miles or 0), 4),
        tuple(sorted(avoid_ids)),
    )
    if key not in cache:
        cache[key] = shortest_connector_path(
            start_point,
            end_point,
            connector_graph,
            snap_tolerance_miles,
            avoid_official_segment_ids=avoid_ids,
        )
    return cache[key]


def alternate_path_still_completes_repeated_ids(
    *,
    alternate: dict[str, Any],
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    repeated_segment_ids: set[str],
    official_segments: list[dict[str, Any]],
    elevation_sampler: Any = None,
) -> set[str]:
    repeated_ids = {str(segment_id) for segment_id in repeated_segment_ids}
    if not repeated_ids:
        return set()
    segment_candidates = [
        segment
        for segment in official_segments
        if str(segment.get("seg_id")) in repeated_ids
    ]
    if not segment_candidates:
        return set()
    path_coords = [
        (float(coord[0]), float(coord[1]))
        for coord in alternate.get("path_coordinates") or []
        if len(coord) >= 2
    ]
    coords = [start_point, *path_coords, end_point]
    completed = set()
    for segment in segment_candidates:
        official_coords = densify_coords(segment.get("coordinates") or [], max_gap_miles=0.02)
        if not official_coords:
            continue
        near_count = sum(
            1
            for point in official_coords
            if point_to_polyline_distance_miles(point, coords) <= 0.045
        )
        if near_count / len(official_coords) >= 0.85:
            completed.add(str(segment.get("seg_id")))
    return completed & repeated_ids


def densify_coords(
    coords: list[tuple[float, float]],
    *,
    max_gap_miles: float,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    dense = [coords[0]]
    for left, right in zip(coords, coords[1:]):
        gap = haversine_miles(left, right)
        steps = max(1, math.ceil(gap / max_gap_miles)) if max_gap_miles > 0 else 1
        for step in range(1, steps + 1):
            ratio = step / steps
            dense.append(
                (
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                )
            )
    return dense


def post_credit_repeat_instances(
    route: dict[str, Any],
    activity_coords: list[tuple[float, float]],
    *,
    official_segments: list[dict[str, Any]],
    connector_graph: dict[str, Any] | None,
    completed_at_export_ids: set[str],
    snap_tolerance_miles: float,
    alternate_path_cache: dict[tuple[Any, ...], dict[str, Any] | None],
    elevation_sampler: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard_instances: list[dict[str, Any]] = []
    advisory_instances: list[dict[str, Any]] = []
    credited = set(completed_at_export_ids)
    cumulative = cumulative_points(activity_coords)
    claimed_segment_ids = set(normalized_ids(route.get("segment_ids") or []))
    baseline_missing_endpoint_ids = official_endpoint_miss_ids(
        activity_coords,
        segment_ids=claimed_segment_ids,
        official_segments=official_segments,
    )
    for cue in route.get("wayfinding_cues") or []:
        repeat_ids = set(normalized_ids(cue.get("official_repeat_segment_ids") or []))
        interval = cue_interval(route, cue)
        interval_coords: list[tuple[float, float]] = []
        cue_completed_repeat_ids: set[str] = set()
        if repeat_ids and interval:
            interval_coords = interval_coordinates(activity_coords, start_mile=interval[0], end_mile=interval[1])
            if len(interval_coords) >= 2:
                cue_completed, _partial, _candidate_count = review_completed_segment_ids(
                    interval_coords,
                    official_segments,
                    planned_ids=repeat_ids,
                    threshold_miles=0.045,
                    endpoint_threshold_miles=0.04,
                    min_fraction=0.85,
                    partial_min_fraction=0.2,
                    max_activity_points=1200,
                    elevation_sampler=elevation_sampler,
                )
                cue_completed_repeat_ids = cue_completed & repeat_ids
        already_repeated_ids = cue_completed_repeat_ids & credited
        if already_repeated_ids:
            interval_miles = polyline_distance_miles(interval_coords)
            current_miles = interval_miles if interval_miles > 0 else cue_route_leg_miles(cue)
            start_point = point_at_mile(cumulative, interval[0]) if interval else None
            end_point = point_at_mile(cumulative, interval[1]) if interval else None
            base = {
                "seq": cue.get("seq"),
                "cue_type": cue.get("cue_type"),
                "repeated_segment_ids": normalized_ids(already_repeated_ids),
                "already_credited_source": already_credited_source(already_repeated_ids, completed_at_export_ids),
                "current_route_leg_miles": round(float(current_miles or 0), 4),
                "source_route_leg_miles": round(float(cue_route_leg_miles(cue) or 0), 4),
                "savings_threshold_miles": round(post_credit_repeat_savings_threshold(current_miles), 4),
            }
            if not connector_graph:
                advisory_instances.append({**base, "reason": "connector_graph_unavailable"})
            elif not interval or start_point is None or end_point is None:
                advisory_instances.append({**base, "reason": "repeat_cue_anchor_unavailable"})
            else:
                alternate = alternate_path_avoiding_repeats(
                    start_point=start_point,
                    end_point=end_point,
                    repeated_segment_ids=already_repeated_ids,
                    connector_graph=connector_graph,
                    snap_tolerance_miles=snap_tolerance_miles,
                    cache=alternate_path_cache,
                )
                if not alternate:
                    advisory_instances.append(
                        {**base, "reason": "repeat_exit_no_alternate_graph_path_proven"}
                    )
                else:
                    still_completed_ids = alternate_path_still_completes_repeated_ids(
                        alternate=alternate,
                        start_point=start_point,
                        end_point=end_point,
                        repeated_segment_ids=already_repeated_ids,
                        official_segments=official_segments,
                        elevation_sampler=elevation_sampler,
                    )
                    if still_completed_ids:
                        advisory_instances.append(
                            {
                                **base,
                                "reason": "alternate_geometry_still_completes_repeated_segments",
                                "alternate_completed_repeat_ids": normalized_ids(still_completed_ids),
                            }
                        )
                        credited.update(cue_credit_segment_ids(cue, claimed_segment_ids))
                        continue
                    path_coords = [
                        (float(coord[0]), float(coord[1]))
                        for coord in alternate.get("path_coordinates") or []
                        if len(coord) >= 2
                    ]
                    replacement_coords = dedupe_adjacent_coords([start_point, *path_coords, end_point])
                    alternate_miles = polyline_distance_miles(replacement_coords)
                    simulated_coords = replace_interval_coordinates(
                        activity_coords,
                        start_mile=interval[0],
                        end_mile=interval[1],
                        replacement_coords=replacement_coords,
                    )
                    newly_missing_endpoint_ids = official_endpoint_miss_ids(
                        simulated_coords,
                        segment_ids=claimed_segment_ids,
                        official_segments=official_segments,
                    ) - baseline_missing_endpoint_ids
                    newly_missing_geometry_ids = official_geometry_miss_ids(
                        simulated_coords,
                        segment_ids=claimed_segment_ids,
                        official_segments=official_segments,
                    ) - official_geometry_miss_ids(
                        activity_coords,
                        segment_ids=claimed_segment_ids,
                        official_segments=official_segments,
                    )
                    if newly_missing_endpoint_ids:
                        advisory_instances.append(
                            {
                                **base,
                                "reason": "alternate_would_drop_claimed_endpoint_coverage",
                                "would_miss_claimed_segment_ids": normalized_ids(newly_missing_endpoint_ids),
                            }
                        )
                        credited.update(cue_credit_segment_ids(cue, claimed_segment_ids))
                        continue
                    if newly_missing_geometry_ids:
                        advisory_instances.append(
                            {
                                **base,
                                "reason": "alternate_would_drop_claimed_geometry_coverage",
                                "would_miss_claimed_segment_ids": normalized_ids(newly_missing_geometry_ids),
                            }
                        )
                        credited.update(cue_credit_segment_ids(cue, claimed_segment_ids))
                        continue
                    savings = current_miles - alternate_miles
                    instance = {
                        **base,
                        "alternate_distance_miles": round(alternate_miles, 4),
                        "graph_alternate_distance_miles": round(float_value(alternate.get("distance_miles")), 4),
                        "estimated_savings_miles": round(savings, 4),
                        "alternate_connector_miles": round(float_value(alternate.get("connector_miles")), 4),
                        "alternate_official_repeat_miles": round(float_value(alternate.get("official_repeat_miles")), 4),
                        "alternate_connector_names": alternate.get("connector_names") or [],
                        "alternate_connector_classes": alternate.get("connector_classes") or [],
                    }
                    savings_threshold = max(
                        post_credit_repeat_savings_threshold(current_miles),
                        float_value(route.get("on_foot_miles")) * 0.10,
                    )
                    instance["savings_threshold_miles"] = round(savings_threshold, 4)
                    if savings >= savings_threshold:
                        hard_instances.append(instance)
                    else:
                        advisory_instances.append(
                            {
                                **instance,
                                "reason": "alternate_savings_below_route_threshold",
                            }
                        )
        credited.update(cue_credit_segment_ids(cue, claimed_segment_ids))
    return hard_instances, advisory_instances


def cue_text(cue: dict[str, Any]) -> str:
    parts = []
    for key in ("action", "display_detail", "note", "field_warning", "compact"):
        if cue.get(key):
            parts.append(str(cue[key]))
    return " ".join(parts).lower()


def cue_repeat_text_ok(cue: dict[str, Any]) -> bool:
    text = cue_text(cue)
    mentions_repeat = "repeat" in text
    mentions_no_credit = (
        "no new credit" in text
        or "not official challenge credit" in text
        or "does not count" in text
    )
    return mentions_repeat and mentions_no_credit


def declared_repeat_rows(route: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cue in route.get("wayfinding_cues") or []:
        repeat_ids = normalized_ids(cue.get("official_repeat_segment_ids") or [])
        if not repeat_ids:
            continue
        repeat_miles = cue.get("official_repeat_miles")
        rows.append(
            {
                "seq": cue.get("seq"),
                "cue_type": cue.get("cue_type"),
                "official_repeat_segment_ids": repeat_ids,
                "official_repeat_miles": round(float_value(repeat_miles), 2),
                "repeat_miles_missing_or_zero": repeat_miles is None or float_value(repeat_miles) <= 0,
                "cue_text_missing": not cue_repeat_text_ok(cue),
            }
        )
    return rows


def declared_owned_elsewhere_ids(route: dict[str, Any]) -> set[str]:
    reconciliation = route.get("segment_ownership_reconciliation") or {}
    return set(normalized_ids(reconciliation.get("declared_owned_elsewhere_segment_ids") or []))


def review_completed_segment_ids(
    coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    *,
    planned_ids: set[str],
    threshold_miles: float,
    endpoint_threshold_miles: float | None,
    min_fraction: float,
    partial_min_fraction: float,
    max_activity_points: int,
    elevation_sampler: Any = None,
) -> tuple[set[str], set[str], int]:
    review_coords = downsample_coords(coords, max_points=max_activity_points)
    candidates = candidate_segments_for_activity(
        coords,
        official_segments,
        planned_ids,
        threshold_miles,
    )
    review = review_activity_against_segments(
        review_coords,
        candidates,
        planned_segment_ids=planned_ids,
        threshold_miles=threshold_miles,
        endpoint_threshold_miles=endpoint_threshold_miles,
        min_fraction=min_fraction,
        partial_min_fraction=partial_min_fraction,
        elevation_sampler=elevation_sampler,
    )
    return set(review["completed_segment_ids"]), set(review["partial_segment_ids"]), len(candidates)


def non_credit_full_segment_ids(
    route: dict[str, Any],
    activity_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    *,
    threshold_miles: float,
    endpoint_threshold_miles: float | None,
    min_fraction: float,
    partial_min_fraction: float,
    max_activity_points: int,
    elevation_sampler: Any = None,
) -> set[str]:
    completed: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        if str(cue.get("cue_type") or "") not in NON_CREDIT_CUE_TYPES:
            continue
        interval_coords = cue_review_coords(activity_coords, cue)
        if len(interval_coords) < 2:
            continue
        cue_completed, _partial, _candidate_count = review_completed_segment_ids(
            interval_coords,
            official_segments,
            planned_ids=set(normalized_ids(route.get("segment_ids") or [])),
            threshold_miles=threshold_miles,
            endpoint_threshold_miles=endpoint_threshold_miles,
            min_fraction=min_fraction,
            partial_min_fraction=partial_min_fraction,
            max_activity_points=max_activity_points,
            elevation_sampler=elevation_sampler,
        )
        completed.update(cue_completed)
    return completed


def post_credit_hidden_self_repeat_ids(
    route: dict[str, Any],
    activity_coords: list[tuple[float, float]],
    official_segments: list[dict[str, Any]],
    *,
    declared_repeat: set[str],
    threshold_miles: float,
    endpoint_threshold_miles: float | None,
    min_fraction: float,
    partial_min_fraction: float,
    max_activity_points: int,
    elevation_sampler: Any = None,
) -> set[str]:
    claimed = set(normalized_ids(route.get("segment_ids") or []))
    credited: set[str] = set()
    hidden: set[str] = set()
    for cue in route.get("wayfinding_cues") or []:
        if str(cue.get("cue_type") or "") in NON_CREDIT_CUE_TYPES:
            interval_coords = cue_review_coords(activity_coords, cue)
            if len(interval_coords) >= 2:
                cue_completed, _partial, _candidate_count = review_completed_segment_ids(
                    interval_coords,
                    official_segments,
                    planned_ids=claimed,
                    threshold_miles=threshold_miles,
                    endpoint_threshold_miles=endpoint_threshold_miles,
                    min_fraction=min_fraction,
                    partial_min_fraction=partial_min_fraction,
                    max_activity_points=max_activity_points,
                    elevation_sampler=elevation_sampler,
                )
                hidden.update((cue_completed & credited) - declared_repeat)
        credited.update(cue_credit_segment_ids(cue, claimed))
    return hidden


def warning_rows_for_route(
    route: dict[str, Any],
    *,
    declared_repeat_miles: float,
    non_credit_miles: float,
    ratio: float | None,
) -> list[dict[str, Any]]:
    warnings = []
    if non_credit_miles > 5:
        warnings.append(
            {
                "code": "high_non_credit_miles",
                "severity": "optimization_warning",
                "value": round(non_credit_miles, 2),
                "message": "More than 5 miles of this outing is non-credit movement.",
            }
        )
    if ratio is not None and ratio > 3:
        warnings.append(
            {
                "code": "high_on_foot_to_official_ratio",
                "severity": "optimization_warning",
                "value": round(ratio, 2),
                "message": "On-foot miles are more than 3x official miles.",
            }
        )
    if declared_repeat_miles > 2:
        warnings.append(
            {
                "code": "high_declared_repeat_miles",
                "severity": "optimization_warning",
                "value": round(declared_repeat_miles, 2),
                "message": "Declared official repeat mileage is above 2 miles.",
            }
        )
    return warnings


def route_proof_is_accepted(proof: dict[str, Any]) -> bool:
    checks = proof.get("checks") or {}
    return (
        proof.get("status") == "accepted_current"
        and checks.get("gpx_continuity_passed") is True
        and checks.get("current_route_has_p75_time") is True
        and checks.get("current_route_has_dem_effort") is True
        and checks.get("no_better_exact_generated_candidate") is True
        and checks.get("no_dominant_boundary_recombination") is True
        and checks.get("no_dominant_global_optimizer_replacement") is True
    )


def route_proof_index(route_proofs: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for registry in route_proofs or []:
        for proof in registry.get("proofs") or []:
            if not route_proof_is_accepted(proof):
                continue
            for candidate_id in proof.get("candidate_ids") or []:
                index[str(candidate_id)] = proof
            if proof.get("candidate_id"):
                index[str(proof["candidate_id"])] = proof
    return index


def close_proofed_warning_rows(
    warning_rows: list[dict[str, Any]],
    route_proofs: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proof_index = route_proof_index(route_proofs)
    open_rows = []
    closed_rows = []
    for warning in warning_rows:
        proof = None
        for candidate_id in warning.get("candidate_ids") or []:
            proof = proof_index.get(str(candidate_id))
            if proof:
                break
        if not proof:
            open_rows.append(warning)
            continue
        closed_rows.append(
            {
                **warning,
                "proof_status": "closed_by_route_disproof",
                "route_proof_candidate_id": proof.get("candidate_id"),
                "route_proof_area": proof.get("area"),
            }
        )
    return open_rows, closed_rows


def advisory_closure(status: str, warning_rows: list[dict[str, Any]], closed_warning_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    warning_counts: dict[str, int] = {}
    for warning in warning_rows:
        code = str(warning.get("code") or "unknown")
        warning_counts[code] = warning_counts.get(code, 0) + 1
    hard_failed = status != "passed"
    closed_count = len(closed_warning_rows or [])
    closure_status = "blocked_by_hard_failures"
    if not hard_failed:
        closure_status = "closed_by_route_disproof" if not warning_rows and closed_count else "closed_non_blocking_optimization_backlog"
    return {
        "status": closure_status,
        "warning_count": len(warning_rows),
        "closed_warning_count": closed_count,
        "warning_counts": dict(sorted(warning_counts.items())),
        "blocking_policy": (
            "Route-repeat audit blocks only on missing GPX, hidden self-repeat, latent credit without ownership/repeat decision, unpriced repeat, or proven avoidable post-credit repeat. "
            "High ratio, high non-credit, high declared-repeat, and same-trailhead bundle rows are optimization pressure signals."
        ),
        "closure_reason": (
            "Hard repeat-accounting failures remain unresolved; optimization warnings are secondary until the hard failures are fixed."
            if hard_failed
            else "No repeat-accounting hard failures remain and all optimization warnings have a current adversarial disproof record."
            if not warning_rows and closed_count
            else "No repeat-accounting hard failures remain. Optimization warnings are intentionally carried forward to ownership, repeat-productivity, same-car, and route-efficiency audits without blocking route promotion by themselves."
        ),
    }


def same_trailhead_warnings(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trailhead: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        trailhead = str(route.get("trailhead") or "").strip()
        if not trailhead:
            continue
        by_trailhead.setdefault(trailhead.lower(), []).append(route)
    warnings = []
    for key, group in sorted(by_trailhead.items()):
        if len(group) < 2:
            continue
        labels = [route_label(route) for route in group]
        warnings.append(
            {
                "code": "same_trailhead_bundle_candidate",
                "severity": "optimization_warning",
                "trailhead": group[0].get("trailhead") or key,
                "route_count": len(group),
                "labels": labels,
                "message": "Multiple cards use the same trailhead and may be candidates for no-extra-drive bundling.",
            }
        )
    return warnings


def audit_route(
    route: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    segment_index: dict[str, dict[str, Any]],
    packet_dir: Path,
    connector_graph: dict[str, Any] | None,
    completed_at_export_ids: set[str],
    snap_tolerance_miles: float,
    alternate_path_cache: dict[tuple[Any, ...], dict[str, Any] | None],
    threshold_miles: float,
    endpoint_threshold_miles: float | None,
    min_fraction: float,
    partial_min_fraction: float,
    max_activity_points: int,
    elevation_sampler: Any = None,
) -> dict[str, Any]:
    claimed = set(normalized_ids(route.get("segment_ids") or []))
    repeat_rows = declared_repeat_rows(route)
    declared_repeat = {segment_id for row in repeat_rows for segment_id in row["official_repeat_segment_ids"]}
    owned_elsewhere = declared_owned_elsewhere_ids(route)
    unpriced_repeat_rows = [
        row for row in repeat_rows if row["repeat_miles_missing_or_zero"] or row["cue_text_missing"]
    ]
    unpriced_repeat_ids = {segment_id for row in unpriced_repeat_rows for segment_id in row["official_repeat_segment_ids"]}
    gpx_path = route_audit_gpx_path(route, packet_dir)
    base = {
        "route_key": route_key(route),
        "outing_id": route.get("outing_id"),
        "label": route_label(route),
        "trailhead": route.get("trailhead"),
        "candidate_ids": route.get("candidate_ids") or [],
        "gpx_path": str(gpx_path) if gpx_path else None,
        "claimed_segment_ids": normalized_ids(claimed),
        "declared_repeat_segment_ids": normalized_ids(declared_repeat),
        "declared_owned_elsewhere_segment_ids": normalized_ids(owned_elsewhere),
        "official_miles": round(float_value(route.get("official_miles")), 2),
        "on_foot_miles": round(float_value(route.get("on_foot_miles")), 2),
    }
    official_miles = float_value(route.get("official_miles"))
    on_foot_miles = float_value(route.get("on_foot_miles"))
    non_credit_miles = round(max(0.0, on_foot_miles - official_miles), 2)
    ratio = round(on_foot_miles / official_miles, 2) if official_miles else None
    declared_repeat_miles = round(sum(float_value(row.get("official_repeat_miles")) for row in repeat_rows), 2)
    if not gpx_path or not gpx_path.exists():
        return {
            **base,
            "audit_status": "missing_gpx",
            "actual_full_segment_ids": [],
            "non_credit_full_segment_ids": [],
            "hidden_self_repeat_ids": [],
            "latent_credit_ids": [],
            "unpriced_repeat_ids": normalized_ids(unpriced_repeat_ids),
            "avoidable_post_credit_repeat_ids": [],
            "avoidable_post_credit_repeat_instances": [],
            "post_credit_repeat_advisories": [],
            "unpriced_repeat_rows": unpriced_repeat_rows,
            "optimization_warnings": warning_rows_for_route(
                route,
                declared_repeat_miles=declared_repeat_miles,
                non_credit_miles=non_credit_miles,
                ratio=ratio,
            ),
            "non_credit_miles": non_credit_miles,
            "on_foot_to_official_ratio": ratio,
            "declared_repeat_miles": declared_repeat_miles,
        }

    activity_coords = activity_coordinates(gpx_path)
    actual_full, partial_ids, candidate_count = review_completed_segment_ids(
        activity_coords,
        official_segments,
        planned_ids=claimed | declared_repeat | owned_elsewhere,
        threshold_miles=threshold_miles,
        endpoint_threshold_miles=endpoint_threshold_miles,
        min_fraction=min_fraction,
        partial_min_fraction=partial_min_fraction,
        max_activity_points=max_activity_points,
        elevation_sampler=elevation_sampler,
    )
    non_credit_full = non_credit_full_segment_ids(
        route,
        activity_coords,
        official_segments,
        threshold_miles=threshold_miles,
        endpoint_threshold_miles=endpoint_threshold_miles,
        min_fraction=min_fraction,
        partial_min_fraction=partial_min_fraction,
        max_activity_points=max_activity_points,
        elevation_sampler=elevation_sampler,
    )
    hidden_self_repeat_ids = post_credit_hidden_self_repeat_ids(
        route,
        activity_coords,
        official_segments,
        declared_repeat=declared_repeat,
        threshold_miles=threshold_miles,
        endpoint_threshold_miles=endpoint_threshold_miles,
        min_fraction=min_fraction,
        partial_min_fraction=partial_min_fraction,
        max_activity_points=max_activity_points,
        elevation_sampler=elevation_sampler,
    )
    latent_credit_ids = actual_full - claimed - declared_repeat - owned_elsewhere
    avoidable_repeats, post_credit_repeat_advisories = post_credit_repeat_instances(
        route,
        activity_coords,
        official_segments=official_segments,
        connector_graph=connector_graph,
        completed_at_export_ids=completed_at_export_ids,
        snap_tolerance_miles=snap_tolerance_miles,
        alternate_path_cache=alternate_path_cache,
        elevation_sampler=elevation_sampler,
    )
    avoidable_repeat_ids = {
        segment_id
        for instance in avoidable_repeats
        for segment_id in normalized_ids(instance.get("repeated_segment_ids") or [])
    }
    hard_failure_ids = hidden_self_repeat_ids | latent_credit_ids | unpriced_repeat_ids | avoidable_repeat_ids
    status = "failed" if hard_failure_ids else "passed"
    return {
        **base,
        "audit_status": status,
        "activity_point_count": len(activity_coords),
        "candidate_segment_count": candidate_count,
        "actual_full_segment_ids": normalized_ids(actual_full),
        "partial_segment_ids": normalized_ids(partial_ids),
        "non_credit_full_segment_ids": normalized_ids(non_credit_full),
        "hidden_self_repeat_ids": normalized_ids(hidden_self_repeat_ids),
        "latent_credit_ids": normalized_ids(latent_credit_ids),
        "unpriced_repeat_ids": normalized_ids(unpriced_repeat_ids),
        "avoidable_post_credit_repeat_ids": normalized_ids(avoidable_repeat_ids),
        "avoidable_post_credit_repeat_instances": avoidable_repeats,
        "post_credit_repeat_advisories": post_credit_repeat_advisories,
        "unpriced_repeat_rows": unpriced_repeat_rows,
        "segments": {
            "hidden_self_repeat": [segment_brief(segment_index, segment_id) for segment_id in normalized_ids(hidden_self_repeat_ids)],
            "latent_credit": [segment_brief(segment_index, segment_id) for segment_id in normalized_ids(latent_credit_ids)],
            "unpriced_repeat": [segment_brief(segment_index, segment_id) for segment_id in normalized_ids(unpriced_repeat_ids)],
            "avoidable_post_credit_repeat": [
                segment_brief(segment_index, segment_id)
                for segment_id in normalized_ids(avoidable_repeat_ids)
            ],
        },
        "optimization_warnings": warning_rows_for_route(
            route,
            declared_repeat_miles=declared_repeat_miles,
            non_credit_miles=non_credit_miles,
            ratio=ratio,
        ),
        "non_credit_miles": non_credit_miles,
        "on_foot_to_official_ratio": ratio,
        "declared_repeat_miles": declared_repeat_miles,
    }


def build_route_repeat_optimization_audit(
    field_tool_data: dict[str, Any],
    *,
    official_segments: list[dict[str, Any]],
    packet_dir: Path,
    connector_graph_path: Path | None = None,
    threshold_miles: float = 0.045,
    endpoint_threshold_miles: float | None = None,
    min_fraction: float = 0.85,
    partial_min_fraction: float = 0.2,
    max_activity_points: int = 1200,
    tail_opportunity_endpoint_threshold_miles: float = DEFAULT_TAIL_OPPORTUNITY_ENDPOINT_THRESHOLD_MILES,
    tail_opportunity_max_segment_miles: float = DEFAULT_TAIL_OPPORTUNITY_MAX_SEGMENT_MILES,
    elevation_sampler: Any = None,
    source_files: dict[str, str] | None = None,
    route_proofs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    routes = field_tool_data.get("routes") or []
    segment_index = build_segment_index(official_segments)
    connector_graph = (
        load_connector_graph(connector_graph_path, official_segments=official_segments)
        if connector_graph_path and connector_graph_path.exists()
        else None
    )
    completed_at_export_ids = set(
        normalized_ids((field_tool_data.get("progress") or {}).get("completed_segment_ids_at_export") or [])
    )
    alternate_path_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
    snap_tolerance_miles = 0.02
    route_rows = [
        audit_route(
            route,
            official_segments=official_segments,
            segment_index=segment_index,
            packet_dir=packet_dir,
            connector_graph=connector_graph,
            completed_at_export_ids=completed_at_export_ids,
            snap_tolerance_miles=snap_tolerance_miles,
            alternate_path_cache=alternate_path_cache,
            threshold_miles=threshold_miles,
            endpoint_threshold_miles=endpoint_threshold_miles,
            min_fraction=min_fraction,
            partial_min_fraction=partial_min_fraction,
            max_activity_points=max_activity_points,
            elevation_sampler=elevation_sampler,
        )
        for route in routes
    ]
    bundle_warnings = same_trailhead_warnings(routes)
    tail_opportunities = cross_route_tail_opportunities(
        routes,
        segment_index,
        endpoint_threshold_miles=tail_opportunity_endpoint_threshold_miles,
        max_segment_miles=tail_opportunity_max_segment_miles,
    )
    for warning in bundle_warnings:
        for row in route_rows:
            if row.get("label") in warning["labels"]:
                row.setdefault("optimization_warnings", []).append(warning)
    rows_by_route_key = {row["route_key"]: row for row in route_rows}
    for opportunity in tail_opportunities:
        row = rows_by_route_key.get(str(opportunity.get("receiver_route_key") or ""))
        if row:
            row.setdefault("optimization_warnings", []).append(
                {
                    "code": opportunity["code"],
                    "severity": opportunity["severity"],
                    "repeated_owned_segment_id": opportunity["repeated_owned_segment"]["seg_id"],
                    "adjacent_candidate_segment_id": opportunity["adjacent_candidate_segment"]["seg_id"],
                    "adjacent_segment_miles": opportunity["adjacent_segment_miles"],
                    "endpoint_gap_miles": opportunity["endpoint_gap_miles"],
                    "owner_routes": opportunity["owner_routes"],
                    "message": opportunity["message"],
                }
            )

    failed_routes = [row for row in route_rows if row["audit_status"] != "passed"]
    missing_gpx_routes = [row for row in route_rows if row["audit_status"] == "missing_gpx"]
    hidden_self_repeat_ids = {segment_id for row in route_rows for segment_id in row["hidden_self_repeat_ids"]}
    latent_credit_ids = {segment_id for row in route_rows for segment_id in row["latent_credit_ids"]}
    unpriced_repeat_ids = {segment_id for row in route_rows for segment_id in row["unpriced_repeat_ids"]}
    avoidable_repeat_ids = {
        segment_id
        for row in route_rows
        for segment_id in row.get("avoidable_post_credit_repeat_ids", [])
    }
    avoidable_repeat_instances = [
        {"label": row["label"], "candidate_ids": row["candidate_ids"], **instance}
        for row in route_rows
        for instance in row.get("avoidable_post_credit_repeat_instances") or []
    ]
    post_credit_repeat_advisories = [
        {"label": row["label"], "candidate_ids": row["candidate_ids"], **instance}
        for row in route_rows
        for instance in row.get("post_credit_repeat_advisories") or []
    ]
    warning_rows = [
        {"label": row["label"], "candidate_ids": row["candidate_ids"], **warning}
        for row in route_rows
        for warning in row.get("optimization_warnings") or []
    ]
    open_warning_rows, closed_warning_rows = close_proofed_warning_rows(warning_rows, route_proofs)
    high_repeat_burden = sorted(
        [
            {
                "label": row["label"],
                "outing_id": row.get("outing_id"),
                "candidate_ids": row.get("candidate_ids") or [],
                "official_miles": row["official_miles"],
                "on_foot_miles": row["on_foot_miles"],
                "non_credit_miles": row["non_credit_miles"],
                "on_foot_to_official_ratio": row["on_foot_to_official_ratio"],
                "declared_repeat_miles": row["declared_repeat_miles"],
            }
            for row in route_rows
        ],
        key=lambda row: float(row["non_credit_miles"] or 0),
        reverse=True,
    )
    status = "failed" if failed_routes or missing_gpx_routes else "passed"
    return {
        "schema": "boise_trails_route_repeat_optimization_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "summary": {
            "route_count": len(routes),
            "failed_route_count": len(failed_routes),
            "missing_gpx_route_count": len(missing_gpx_routes),
            "hidden_self_repeat_segment_count": len(hidden_self_repeat_ids),
            "latent_credit_segment_count": len(latent_credit_ids),
            "unpriced_repeat_segment_count": len(unpriced_repeat_ids),
            "avoidable_post_credit_repeat_segment_count": len(avoidable_repeat_ids),
            "avoidable_post_credit_repeat_instance_count": len(avoidable_repeat_instances),
            "post_credit_repeat_advisory_count": len(post_credit_repeat_advisories),
            "optimization_warning_count": len(open_warning_rows),
            "total_optimization_warning_count": len(warning_rows),
            "closed_optimization_warning_count": len(closed_warning_rows),
            "high_non_credit_route_count": sum(1 for row in route_rows if float(row["non_credit_miles"] or 0) > 5),
            "high_ratio_route_count": sum(1 for row in route_rows if row["on_foot_to_official_ratio"] is not None and row["on_foot_to_official_ratio"] > 3),
            "high_declared_repeat_route_count": sum(1 for row in route_rows if float(row["declared_repeat_miles"] or 0) > 2),
            "same_trailhead_bundle_warning_count": len(bundle_warnings),
            "cross_route_tail_opportunity_count": len(tail_opportunities),
        },
        "parameters": {
            "threshold_miles": threshold_miles,
            "endpoint_threshold_miles": endpoint_threshold_miles or threshold_miles,
            "min_fraction": min_fraction,
            "partial_min_fraction": partial_min_fraction,
            "max_activity_points": max_activity_points,
            "post_credit_repeat_min_savings_miles": 0.10,
            "post_credit_repeat_savings_ratio": 0.10,
            "connector_snap_tolerance_miles": snap_tolerance_miles,
            "tail_opportunity_endpoint_threshold_miles": tail_opportunity_endpoint_threshold_miles,
            "tail_opportunity_max_segment_miles": tail_opportunity_max_segment_miles,
        },
        "source_files": source_files or {},
        "connector_graph": {
            "path": display_path(connector_graph_path) if connector_graph_path else None,
            "exists": bool(connector_graph_path and connector_graph_path.exists()),
            "loaded": bool(connector_graph),
        },
        "hard_failures": {
            "hidden_self_repeat_segment_ids": normalized_ids(hidden_self_repeat_ids),
            "latent_credit_segment_ids": normalized_ids(latent_credit_ids),
            "unpriced_repeat_segment_ids": normalized_ids(unpriced_repeat_ids),
            "avoidable_post_credit_repeat_segment_ids": normalized_ids(avoidable_repeat_ids),
        },
        "advisory_closure": advisory_closure(status, open_warning_rows, closed_warning_rows),
        "avoidable_post_credit_repeat_instances": avoidable_repeat_instances,
        "post_credit_repeat_advisories": post_credit_repeat_advisories,
        "optimization_warnings": open_warning_rows,
        "closed_optimization_warnings": closed_warning_rows,
        "same_trailhead_bundle_warnings": bundle_warnings,
        "cross_route_tail_opportunities": tail_opportunities,
        "high_repeat_burden": high_repeat_burden,
        "failed_routes": failed_routes,
        "routes": route_rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    failures = audit["hard_failures"]
    lines = [
        "# Route Repeat Optimization Audit",
        "",
        f"Generated: {audit.get('generated_at')}",
        f"Status: `{audit.get('status')}`",
        "",
        "## Summary",
        "",
        f"- Routes audited: {summary['route_count']}",
        f"- Failed routes: {summary['failed_route_count']}",
        f"- Hidden self-repeat segments: {summary['hidden_self_repeat_segment_count']}",
        f"- Latent credit segments without ownership/repeat decision: {summary['latent_credit_segment_count']}",
        f"- Unpriced repeat segments: {summary['unpriced_repeat_segment_count']}",
        f"- Avoidable post-credit repeat instances: {summary.get('avoidable_post_credit_repeat_instance_count', 0)}",
        f"- Post-credit repeat advisories: {summary.get('post_credit_repeat_advisory_count', 0)}",
        f"- Open optimization warnings: {summary['optimization_warning_count']}",
        f"- Closed optimization warnings: {summary.get('closed_optimization_warning_count', 0)} of {summary.get('total_optimization_warning_count', summary['optimization_warning_count'])}",
        f"- High non-credit routes (>5 mi): {summary['high_non_credit_route_count']}",
        f"- High ratio routes (>3x): {summary['high_ratio_route_count']}",
        f"- High declared-repeat routes (>2 mi): {summary['high_declared_repeat_route_count']}",
        f"- Same-trailhead bundle warnings: {summary['same_trailhead_bundle_warning_count']}",
        f"- Cross-route tail opportunities: {summary.get('cross_route_tail_opportunity_count', 0)}",
        "",
        "## Hard Failures",
        "",
        f"- Hidden self-repeat ids: {failures['hidden_self_repeat_segment_ids'] or []}",
        f"- Latent credit ids: {failures['latent_credit_segment_ids'] or []}",
        f"- Unpriced repeat ids: {failures['unpriced_repeat_segment_ids'] or []}",
        f"- Avoidable post-credit repeat ids: {failures.get('avoidable_post_credit_repeat_segment_ids') or []}",
        "",
        "## Advisory Closure",
        "",
        f"- Closure status: `{audit.get('advisory_closure', {}).get('status')}`",
        f"- Warning count: {audit.get('advisory_closure', {}).get('warning_count')}",
        f"- Blocking policy: {audit.get('advisory_closure', {}).get('blocking_policy')}",
        f"- Closure reason: {audit.get('advisory_closure', {}).get('closure_reason')}",
        "",
        "## Highest Non-Credit Burden",
        "",
        "| Label | Official mi | On-foot mi | Non-credit mi | Ratio | Declared repeat mi |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit.get("high_repeat_burden", [])[:15]:
        ratio = "" if row.get("on_foot_to_official_ratio") is None else f"{float(row['on_foot_to_official_ratio']):.2f}"
        lines.append(
            "| {label} | {official:.2f} | {on_foot:.2f} | {non_credit:.2f} | {ratio} | {repeat:.2f} |".format(
                label=row.get("label"),
                official=float(row.get("official_miles") or 0),
                on_foot=float(row.get("on_foot_miles") or 0),
                non_credit=float(row.get("non_credit_miles") or 0),
                ratio=ratio,
                repeat=float(row.get("declared_repeat_miles") or 0),
            )
        )
    if audit.get("cross_route_tail_opportunities"):
        lines.extend(
            [
                "",
                "## Cross-Route Tail Opportunities",
                "",
                "| Receiver | Repeated owned segment | Adjacent candidate | Owner route(s) | Endpoint gap mi | Adjacent mi |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for row in audit["cross_route_tail_opportunities"][:25]:
            repeated = row.get("repeated_owned_segment") or {}
            adjacent = row.get("adjacent_candidate_segment") or {}
            owner_labels = ", ".join(
                str(owner.get("label") or owner.get("outing_id") or owner.get("route_key") or "unknown")
                for owner in row.get("owner_routes") or []
            )
            lines.append(
                "| {receiver} | {repeat_id} {repeat_name} | {adjacent_id} {adjacent_name} | {owners} | {gap:.4f} | {miles:.2f} |".format(
                    receiver=row.get("receiver_label"),
                    repeat_id=repeated.get("seg_id"),
                    repeat_name=repeated.get("seg_name"),
                    adjacent_id=adjacent.get("seg_id"),
                    adjacent_name=adjacent.get("seg_name"),
                    owners=owner_labels,
                    gap=float(row.get("endpoint_gap_miles") or 0),
                    miles=float(row.get("adjacent_segment_miles") or 0),
                )
            )
    if audit.get("failed_routes"):
        lines.extend(["", "## Failed Routes", ""])
        for row in audit["failed_routes"]:
            lines.append(f"### {row['label']}")
            if row["hidden_self_repeat_ids"]:
                lines.append("- Hidden self-repeat: " + ", ".join(row["hidden_self_repeat_ids"]))
            if row["latent_credit_ids"]:
                lines.append("- Latent credit without ownership decision: " + ", ".join(row["latent_credit_ids"]))
            if row["unpriced_repeat_ids"]:
                lines.append("- Unpriced repeat: " + ", ".join(row["unpriced_repeat_ids"]))
            if row.get("avoidable_post_credit_repeat_ids"):
                lines.append(
                    "- Avoidable post-credit repeat: "
                    + ", ".join(row["avoidable_post_credit_repeat_ids"])
                )
            lines.append("")
    if audit.get("post_credit_repeat_advisories"):
        lines.extend(["", "## Post-Credit Repeat Advisories", ""])
        for item in audit["post_credit_repeat_advisories"][:20]:
            lines.append(
                "- {label} cue {seq}: {reason}; repeated {ids}".format(
                    label=item.get("label"),
                    seq=item.get("seq"),
                    reason=item.get("reason"),
                    ids=", ".join(item.get("repeated_segment_ids") or []),
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-tool-data-json", type=Path, default=DEFAULT_FIELD_TOOL_DATA_JSON)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--connector-geojson", type=Path, default=DEFAULT_CONNECTOR_GEOJSON)
    parser.add_argument("--dem-tif", type=Path, default=DEFAULT_DEM_TIF)
    parser.add_argument("--dem-summary-json", type=Path, default=DEFAULT_DEM_SUMMARY_JSON)
    parser.add_argument("--threshold-miles", type=float, default=0.045)
    parser.add_argument("--endpoint-threshold-miles", type=float)
    parser.add_argument("--min-fraction", type=float, default=0.85)
    parser.add_argument("--partial-min-fraction", type=float, default=0.2)
    parser.add_argument("--max-activity-points", type=int, default=1200)
    parser.add_argument(
        "--tail-opportunity-endpoint-threshold-miles",
        type=float,
        default=DEFAULT_TAIL_OPPORTUNITY_ENDPOINT_THRESHOLD_MILES,
        help="Endpoint proximity for cross-route tail opportunity warnings.",
    )
    parser.add_argument(
        "--tail-opportunity-max-segment-miles",
        type=float,
        default=DEFAULT_TAIL_OPPORTUNITY_MAX_SEGMENT_MILES,
        help="Maximum adjacent claimed official segment size to flag as a tail opportunity.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--route-proof-json", action="append", type=Path, dest="route_proof_jsons")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write audit artifacts but return success even when hard failures exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    field_tool_data = read_json(args.field_tool_data_json)
    official_segments, official_metadata = load_official_segments(args.official_geojson)
    dem_context = load_dem_context(args.dem_tif, args.dem_summary_json)
    route_proof_paths = args.route_proof_jsons if args.route_proof_jsons is not None else DEFAULT_ROUTE_PROOF_JSONS
    route_proofs = [
        payload
        for payload in (read_optional_json(path) for path in route_proof_paths)
        if payload is not None
    ]
    source_files = {
        "field_tool_data_json": display_path(args.field_tool_data_json),
        "packet_dir": display_path(args.packet_dir),
        "audit_gpx_glob": display_path(args.packet_dir / "gpx" / "audit" / "*.gpx"),
        "official_geojson": display_path(args.official_geojson),
        "official_last_updated_utc": str(official_metadata.get("lastUpdatedUTC")),
        "connector_geojson": display_path(args.connector_geojson),
        "dem_tif": display_path(args.dem_tif),
        "dem_summary_json": display_path(args.dem_summary_json),
        "route_proof_jsons": [display_path(path) for path in route_proof_paths],
    }
    audit = build_route_repeat_optimization_audit(
        field_tool_data,
        official_segments=official_segments,
        packet_dir=args.packet_dir,
        connector_graph_path=args.connector_geojson,
        threshold_miles=args.threshold_miles,
        endpoint_threshold_miles=args.endpoint_threshold_miles,
        min_fraction=args.min_fraction,
        partial_min_fraction=args.partial_min_fraction,
        max_activity_points=args.max_activity_points,
        tail_opportunity_endpoint_threshold_miles=args.tail_opportunity_endpoint_threshold_miles,
        tail_opportunity_max_segment_miles=args.tail_opportunity_max_segment_miles,
        elevation_sampler=dem_context.get("sampler"),
        source_files=source_files,
        route_proofs=route_proofs,
    )
    write_json(args.output_json, audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    manifest = build_artifact_manifest(
        run_id=args.output_json.stem,
        inputs=[
            display_path(args.field_tool_data_json),
            display_path(args.official_geojson),
            display_path(args.connector_geojson),
        ],
        outputs=[display_path(args.output_json), display_path(args.output_md)],
        command="python years/2026/scripts/route_repeat_optimization_audit.py",
        metadata={"schema": audit["schema"], "status": audit["status"]},
    )
    write_manifest(args.manifest_json, manifest)
    print(f"Wrote {display_path(args.output_json)}")
    print(f"Wrote {display_path(args.output_md)}")
    print(f"Wrote {display_path(args.manifest_json)}")
    print(json.dumps({"status": audit["status"], **audit["summary"]}, indent=2))
    if audit["status"] != "passed" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
