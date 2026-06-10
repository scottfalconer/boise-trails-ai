#!/usr/bin/env python3
"""Export simulated executable on-foot recommendations to GPX tracks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, write_manifest  # noqa: E402
from personal_route_planner import (  # noqa: E402
    DEFAULT_OFFICIAL_GEOJSON,
    haversine_miles,
    iter_line_parts,
    load_connector_graph,
    load_official_segments,
    read_json,
    shortest_connector_path,
)


DEFAULT_PLAN_JSON = YEAR_DIR / "outputs" / "personal-route-menu.json"
DEFAULT_EXECUTION_JSON = (
    YEAR_DIR
    / "experiments"
    / "2026-05-04-outing-execution-simulation"
    / "outing_execution.json"
)
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "gpx" / "executable-routes-2026-05-04"
DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON = YEAR_DIR / "inputs" / "open-data" / "special-management-rules-2026.json"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "route"


def load_official_segment_index(path: Path) -> dict[int, dict[str, Any]]:
    data = read_json(path)
    index = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        seg_id = int(props["segId"])
        geometry = feature.get("geometry") or {}
        parts = iter_line_parts(geometry)
        if geometry.get("type") == "MultiLineString" and len(parts) > 1:
            raise ValueError(
                f"Official segment {seg_id} is a MultiLineString; "
                "normalize multipart official geometry before GPX export"
            )
        index[seg_id] = {
            "seg_id": seg_id,
            "seg_name": props.get("segName"),
            "trail_name": re.sub(r"\s+\d+$", "", str(props.get("segName") or "")),
            "direction": props.get("direction") or "both",
            "coordinates": parts[0] if parts else [],
        }
    return index


def load_candidate_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in plan["route_menu"]["all_candidates"]
    }


def dedupe_append(
    target: list[tuple[float, float]],
    coords: list[tuple[float, float]],
) -> None:
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if target and target[-1] == point:
            continue
        target.append(point)


def densify_coordinates(
    coords: list[tuple[float, float]],
    max_gap_miles: float = 0.03,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return list(coords)
    dense = [coords[0]]
    for left, right in zip(coords, coords[1:]):
        gap = haversine_miles(left, right)
        steps = max(1, int((gap / max_gap_miles) + 0.999999))
        for step in range(1, steps + 1):
            fraction = step / steps
            dense.append(
                (
                    left[0] + (right[0] - left[0]) * fraction,
                    left[1] + (right[1] - left[1]) * fraction,
                )
            )
    return dense


def candidate_segment_coordinates(
    candidate: dict[str, Any],
    segment: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
) -> list[tuple[float, float]]:
    seg_id = int(segment["seg_id"])
    official = official_index[seg_id]
    special_management_direction = special_management_segment_direction_overrides().get(str(seg_id))
    coordinate_source = official if special_management_direction else segment
    coords = [
        (float(coord[0]), float(coord[1]))
        for coord in (coordinate_source.get("coordinates") or official["coordinates"])
    ]
    if special_management_direction == "forward":
        if densify_source_lines:
            return densify_coordinates(coords, max_gap_miles=densify_max_gap_miles)
        return coords
    if special_management_direction == "reverse":
        coords = list(reversed(coords))
        if densify_source_lines:
            return densify_coordinates(coords, max_gap_miles=densify_max_gap_miles)
        return coords
    planned_directions = (
        (candidate.get("direction_validation") or {}).get("planned_traversal_direction")
        or {}
    )
    if planned_directions.get(str(seg_id)) == "official_geometry_end_to_start":
        coords = list(reversed(coords))
    elif (
        (candidate.get("route_orientation") or {}).get("direction") == "reversed"
        and official.get("direction") == "both"
        and not candidate.get("custom_traversal_order")
    ):
        coords = list(reversed(coords))
    if densify_source_lines:
        return densify_coordinates(coords, max_gap_miles=densify_max_gap_miles)
    return coords


_SPECIAL_MANAGEMENT_SEGMENT_DIRECTIONS: dict[str, str] | None = None


def special_management_segment_direction_overrides() -> dict[str, str]:
    global _SPECIAL_MANAGEMENT_SEGMENT_DIRECTIONS
    if _SPECIAL_MANAGEMENT_SEGMENT_DIRECTIONS is not None:
        return _SPECIAL_MANAGEMENT_SEGMENT_DIRECTIONS
    directions: dict[str, str] = {}
    if DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON.exists():
        payload = json.loads(DEFAULT_SPECIAL_MANAGEMENT_RULES_JSON.read_text(encoding="utf-8"))
        for rule in payload.get("rules") or []:
            if str(rule.get("rule_type") or "") != "directional_segment_traversal":
                continue
            for segment_id, allowed in (rule.get("segment_direction_overrides") or {}).items():
                allowed_values = [str(item) for item in allowed or []]
                if allowed_values == ["forward"]:
                    directions[str(segment_id)] = "forward"
                elif allowed_values == ["reverse"]:
                    directions[str(segment_id)] = "reverse"
    _SPECIAL_MANAGEMENT_SEGMENT_DIRECTIONS = directions
    return directions


def planned_segment_direction(candidate: dict[str, Any], seg_id: int) -> str | None:
    planned_directions = (
        (candidate.get("direction_validation") or {}).get("planned_traversal_direction")
        or {}
    )
    direction = planned_directions.get(str(seg_id))
    return str(direction) if direction else None


def should_auto_orient_segment(
    candidate: dict[str, Any],
    official: dict[str, Any],
    seg_id: int,
) -> bool:
    if planned_segment_direction(candidate, seg_id):
        return False
    if (candidate.get("route_orientation") or {}).get("direction") == "reversed":
        return False
    return str(official.get("direction") or "both") == "both"


def orient_path_to_current(
    current: tuple[float, float] | None,
    path: list[tuple[float, float]],
    *,
    enabled: bool = True,
) -> list[tuple[float, float]]:
    if not enabled or current is None or len(path) < 2:
        return path
    start_gap = haversine_miles(current, path[0])
    end_gap = haversine_miles(current, path[-1])
    if end_gap < start_gap:
        return list(reversed(path))
    return path


def path_coordinate_tuples(
    raw_coords: list[Any],
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
) -> list[tuple[float, float]]:
    coords = [(float(coord[0]), float(coord[1])) for coord in raw_coords or []]
    if densify_source_lines:
        return densify_coordinates(coords, max_gap_miles=densify_max_gap_miles)
    return coords


def normalized_connector_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def skipped_connector_name_set(connector_graph: dict[str, Any] | None) -> set[str]:
    skipped = ((connector_graph or {}).get("skipped_connector_names") or {})
    if isinstance(skipped, dict):
        values = skipped.keys()
    else:
        values = skipped
    return {normalized_connector_name(value) for value in values if str(value or "").strip()}


def stored_connector_link_names(link: dict[str, Any]) -> set[str]:
    names = {normalized_connector_name(name) for name in link.get("connector_names") or []}
    for edge in link.get("connector_edges") or []:
        names.add(normalized_connector_name(edge.get("name")))
    return {name for name in names if name}


def connector_names_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    return (len(left) > 4 and left in right) or (len(right) > 4 and right in left)


def matching_skipped_connector_names(
    link: dict[str, Any],
    connector_graph: dict[str, Any] | None,
) -> set[str]:
    skipped_names = skipped_connector_name_set(connector_graph)
    if not skipped_names:
        return set()
    matches = set()
    for link_name in stored_connector_link_names(link):
        if any(connector_names_match(link_name, skipped_name) for skipped_name in skipped_names):
            matches.add(link_name)
    return matches


def stored_connector_link_uses_skipped_name(
    link: dict[str, Any],
    connector_graph: dict[str, Any] | None,
) -> bool:
    return bool(matching_skipped_connector_names(link, connector_graph))


def repaired_connector_link(
    stored_link: dict[str, Any],
    repair: dict[str, Any],
    *,
    replaced_names: set[str],
) -> dict[str, Any]:
    result = dict(stored_link)
    result.update(repair)
    for key in ("from_trail", "to_trail"):
        if stored_link.get(key) is not None:
            result[key] = stored_link.get(key)
    result["source"] = "safe_connector_graph_repair"
    result["replaced_unsafe_connector_names"] = sorted(replaced_names)
    return result


def replacement_for_unsafe_connector_link(
    stored_link: dict[str, Any],
    current: tuple[float, float] | None,
    next_point: tuple[float, float] | None,
    connector_graph: dict[str, Any] | None,
    stitch_snap_tolerance_miles: float,
) -> dict[str, Any] | None:
    if current is None or next_point is None:
        return None
    replaced_names = matching_skipped_connector_names(stored_link, connector_graph)
    if not replaced_names:
        return None
    repair = shortest_connector_path(
        current,
        next_point,
        connector_graph,
        stitch_snap_tolerance_miles,
    )
    if not repair:
        return None
    return repaired_connector_link(stored_link, repair, replaced_names=replaced_names)


def connector_link_from_mapped_path(
    mapped: dict[str, Any],
    *,
    source: str,
    from_segment_id: Any = None,
    to_segment_id: Any = None,
    from_trail: str | None = None,
    to_trail: str | None = None,
    earned_segment_ids_before_link: set[int] | None = None,
    avoided_unearned_segment_ids: set[int] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "from_segment_id": from_segment_id,
        "to_segment_id": to_segment_id,
        "from_trail": from_trail,
        "to_trail": to_trail,
        "distance_miles": mapped.get("distance_miles"),
        "connector_miles": mapped.get("connector_miles"),
        "official_repeat_miles": mapped.get("official_repeat_miles"),
        "connector_names": mapped.get("connector_names") or [],
        "connector_classes": mapped.get("connector_classes") or [],
        "connector_edges": mapped.get("connector_edges") or [],
        "official_repeat_segment_ids": mapped.get("official_repeat_segment_ids") or [],
        "earned_segment_ids_before_link": sorted(earned_segment_ids_before_link or []),
        "avoided_unearned_segment_ids": sorted(avoided_unearned_segment_ids or []),
        "path_coordinates": mapped.get("path_coordinates") or [],
    }


def link_distance_miles(link: dict[str, Any] | None) -> float:
    if not link:
        return float("inf")
    for key in ("distance_miles", "connector_miles"):
        try:
            return float(link.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return float("inf")


def best_connector_link_for_oriented_segment(
    *,
    candidate: dict[str, Any],
    segment: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    current: tuple[float, float] | None,
    previous_segment: dict[str, Any] | None,
    stored_link: dict[str, Any] | None,
    connector_graph: dict[str, Any] | None,
    stitch_snap_tolerance_miles: float,
    avoid_official_segment_ids: set[int] | None,
    earned_segment_ids_before_link: set[int] | None,
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
) -> tuple[list[tuple[float, float]], dict[str, Any] | None]:
    seg_id = int(segment["seg_id"])
    segment_coords = candidate_segment_coordinates(
        candidate,
        segment,
        official_index,
        densify_source_lines=densify_source_lines,
        densify_max_gap_miles=densify_max_gap_miles,
    )
    segment_coords = orient_path_to_current(
        current,
        segment_coords,
        enabled=should_auto_orient_segment(candidate, official_index[seg_id], seg_id),
    )
    if current is None or not segment_coords or connector_graph is None:
        return segment_coords, stored_link
    if haversine_miles(current, segment_coords[0]) <= 0.02:
        return segment_coords, None

    mapped = shortest_connector_path(
        current,
        segment_coords[0],
        connector_graph,
        stitch_snap_tolerance_miles,
        avoid_official_segment_ids=avoid_official_segment_ids,
    )
    if not mapped:
        return segment_coords, stored_link

    mapped_distance = float(mapped.get("distance_miles") or 0.0)
    stored_distance = link_distance_miles(stored_link)
    if mapped_distance <= 0.02:
        return segment_coords, None
    if stored_link and mapped_distance + 0.01 >= stored_distance:
        return segment_coords, stored_link

    return segment_coords, connector_link_from_mapped_path(
        mapped,
        source="mapped_graph_orientation_repair",
        from_segment_id=(previous_segment or {}).get("seg_id"),
        to_segment_id=segment.get("seg_id"),
        from_trail=(previous_segment or {}).get("trail_name"),
        to_trail=segment.get("trail_name"),
        earned_segment_ids_before_link=earned_segment_ids_before_link,
        avoided_unearned_segment_ids=avoid_official_segment_ids,
    )


def safe_between_trail_links_for_candidate(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    stitch_snap_tolerance_miles: float = 0.02,
) -> list[dict[str, Any]]:
    trailhead_access = candidate.get("trailhead_access") or {}
    outbound = path_coordinate_tuples(trailhead_access.get("outbound_path_coordinates") or [])
    current = outbound[-1] if outbound else None
    between_links = list(((candidate.get("between_trail_links") or {}).get("links")) or [])
    next_link_index = 0
    previous_trail = None
    safe_links: list[dict[str, Any]] = []
    for segment in candidate_segments_for_track(candidate, official_index, current):
        trail_name = segment.get("trail_name")
        seg_id = int(segment["seg_id"])
        segment_coords = candidate_segment_coordinates(candidate, segment, official_index)
        segment_coords = orient_path_to_current(
            current,
            segment_coords,
            enabled=should_auto_orient_segment(candidate, official_index[seg_id], seg_id),
        )
        if previous_trail is not None and trail_name != previous_trail:
            stored_link = (
                between_links[next_link_index]
                if next_link_index < len(between_links)
                else {}
            )
            repaired = replacement_for_unsafe_connector_link(
                stored_link,
                current,
                segment_coords[0] if segment_coords else None,
                connector_graph,
                stitch_snap_tolerance_miles,
            )
            safe_links.append(repaired or stored_link)
            next_link_index += 1
        if segment_coords:
            current = segment_coords[-1]
        previous_trail = trail_name
    return safe_links


def oriented_segment_options(
    candidate: dict[str, Any],
    segment: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
) -> list[list[tuple[float, float]]]:
    seg_id = int(segment["seg_id"])
    coords = candidate_segment_coordinates(candidate, segment, official_index)
    if not coords:
        return []
    if special_management_segment_direction_overrides().get(str(seg_id)):
        return [coords]
    official = official_index[seg_id]
    if str(official.get("direction") or "both") == "both":
        reversed_coords = list(reversed(coords))
        if reversed_coords != coords:
            return [coords, reversed_coords]
    return [coords]


def reorder_special_management_group(
    candidate: dict[str, Any],
    segments: list[dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    current: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    remaining = list(segments)
    ordered: list[dict[str, Any]] = []
    while remaining:
        best_index = 0
        best_coords: list[tuple[float, float]] | None = None
        best_gap = float("inf")
        for index, segment in enumerate(remaining):
            for option in oriented_segment_options(candidate, segment, official_index):
                gap = 0.0 if current is None else haversine_miles(current, option[0])
                if gap < best_gap:
                    best_gap = gap
                    best_index = index
                    best_coords = option
        selected = remaining.pop(best_index)
        ordered.append(selected)
        if best_coords:
            current = best_coords[-1]
    return ordered


def candidate_segments_for_track(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]] | None = None,
    current: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    segments = base_candidate_segments_for_track(candidate)
    if official_index is None:
        return segments

    special_direction_segments = set(special_management_segment_direction_overrides())
    if not special_direction_segments:
        return segments

    ordered: list[dict[str, Any]] = []
    index = 0
    while index < len(segments):
        trail_name = segments[index].get("trail_name")
        group = []
        while index < len(segments) and segments[index].get("trail_name") == trail_name:
            group.append(segments[index])
            index += 1
        if any(str(segment.get("seg_id")) in special_direction_segments for segment in group):
            group = reorder_special_management_group(candidate, group, official_index, current)
        for segment in group:
            ordered.append(segment)
            seg_id = int(segment["seg_id"])
            segment_coords = candidate_segment_coordinates(candidate, segment, official_index)
            segment_coords = orient_path_to_current(
                current,
                segment_coords,
                enabled=should_auto_orient_segment(candidate, official_index[seg_id], seg_id),
            )
            if segment_coords:
                current = segment_coords[-1]
    return ordered


def base_candidate_segments_for_track(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    segments = list(candidate.get("segments") or [])
    if (
        (candidate.get("route_orientation") or {}).get("direction") == "reversed"
        and not candidate.get("custom_traversal_order")
    ):
        return list(reversed(segments))
    return segments


def segment_id_sequence(segments: list[dict[str, Any]]) -> list[str]:
    return [str(segment.get("seg_id")) for segment in segments]


def candidate_track_order_changed(
    candidate: dict[str, Any],
    ordered_segments: list[dict[str, Any]],
) -> bool:
    return segment_id_sequence(base_candidate_segments_for_track(candidate)) != segment_id_sequence(
        ordered_segments
    )


def raise_unstitched_gap(candidate: dict[str, Any], from_point: tuple[float, float], to_point: tuple[float, float], gap: float) -> None:
    raise ValueError(
        f"Candidate {candidate.get('candidate_id')} has an unstitched source gap "
        f"of {gap:.4f} mi from {from_point} to {to_point}"
    )


def stitch_remaining_coordinate_gaps(
    coords: list[tuple[float, float]],
    connector_graph: dict[str, Any] | None,
    *,
    stitch_gap_threshold_miles: float,
    stitch_snap_tolerance_miles: float,
    densify_source_lines: bool,
    densify_max_gap_miles: float,
    candidate: dict[str, Any],
    fail_on_unstitched_gap: bool,
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    stitched = [coords[0]]
    for point in coords[1:]:
        gap = haversine_miles(stitched[-1], point)
        if gap > stitch_gap_threshold_miles:
            stitch = (
                shortest_connector_path(
                    stitched[-1],
                    point,
                    connector_graph,
                    stitch_snap_tolerance_miles,
                )
                if connector_graph
                else None
            )
            if stitch:
                dedupe_append(
                    stitched,
                    path_coordinate_tuples(
                        stitch.get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                )
            elif fail_on_unstitched_gap:
                raise_unstitched_gap(candidate, stitched[-1], point, gap)
        dedupe_append(stitched, [point])
    return stitched


def candidate_track_coordinates(
    candidate: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    stitch_gap_threshold_miles: float = 0.05,
    stitch_snap_tolerance_miles: float = 0.02,
    densify_source_lines: bool = False,
    densify_max_gap_miles: float = 0.03,
    fail_on_unstitched_gap: bool = False,
) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    trailhead_access = candidate.get("trailhead_access") or {}
    dedupe_append(
        coords,
        path_coordinate_tuples(
            trailhead_access.get("outbound_path_coordinates") or [],
            densify_source_lines=densify_source_lines,
            densify_max_gap_miles=densify_max_gap_miles,
        ),
    )
    ordered_segments = candidate_segments_for_track(
        candidate,
        official_index,
        coords[-1] if coords else None,
    )
    order_changed = candidate_track_order_changed(candidate, ordered_segments)
    inter_segment_links = (
        []
        if order_changed
        else list(((candidate.get("inter_segment_links") or {}).get("links")) or [])
    )
    link_by_to_segment_id = {
        str(link.get("to_segment_id")): link
        for link in inter_segment_links
        if link.get("to_segment_id") is not None
    }
    required_segment_ids = {
        int(segment.get("seg_id"))
        for segment in ordered_segments
        if str(segment.get("seg_id") or "").isdigit()
    }
    earned_segment_ids: set[int] = set()
    between_links = (
        []
        if inter_segment_links
        else list(((candidate.get("between_trail_links") or {}).get("links")) or [])
    )
    next_link_index = 0
    previous_trail = None
    previous_segment: dict[str, Any] | None = None
    for segment in ordered_segments:
        trail_name = segment.get("trail_name")
        seg_id = int(segment["seg_id"])
        pre_link = link_by_to_segment_id.get(str(segment.get("seg_id")))
        avoid_segment_ids = required_segment_ids - earned_segment_ids
        segment_coords, selected_pre_link = best_connector_link_for_oriented_segment(
            candidate=candidate,
            segment=segment,
            official_index=official_index,
            current=coords[-1] if coords else None,
            previous_segment=previous_segment,
            stored_link=pre_link if previous_trail is not None else None,
            connector_graph=connector_graph,
            stitch_snap_tolerance_miles=stitch_snap_tolerance_miles,
            avoid_official_segment_ids=avoid_segment_ids,
            earned_segment_ids_before_link=earned_segment_ids,
            densify_source_lines=densify_source_lines,
            densify_max_gap_miles=densify_max_gap_miles,
        )
        if previous_trail is not None and selected_pre_link:
            dedupe_append(
                coords,
                path_coordinate_tuples(
                    selected_pre_link.get("path_coordinates") or [],
                    densify_source_lines=densify_source_lines,
                    densify_max_gap_miles=densify_max_gap_miles,
                ),
            )
            segment_coords = orient_path_to_current(
                coords[-1] if coords else None,
                segment_coords,
                enabled=should_auto_orient_segment(candidate, official_index[seg_id], seg_id),
            )
        if previous_trail is not None and trail_name != previous_trail:
            if next_link_index < len(between_links):
                stored_link = between_links[next_link_index]
                segment_coords, selected_link = best_connector_link_for_oriented_segment(
                    candidate=candidate,
                    segment=segment,
                    official_index=official_index,
                    current=coords[-1] if coords else None,
                    previous_segment=previous_segment,
                    stored_link=stored_link,
                    connector_graph=connector_graph,
                    stitch_snap_tolerance_miles=stitch_snap_tolerance_miles,
                    avoid_official_segment_ids=avoid_segment_ids,
                    earned_segment_ids_before_link=earned_segment_ids,
                    densify_source_lines=densify_source_lines,
                    densify_max_gap_miles=densify_max_gap_miles,
                )
                link_is_unsafe = stored_connector_link_uses_skipped_name(
                    selected_link or {},
                    connector_graph,
                )
                link_coords = orient_path_to_current(
                    coords[-1] if coords else None,
                    path_coordinate_tuples(
                        (selected_link or {}).get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                    enabled=not link_is_unsafe,
                )
                current = coords[-1] if coords else None
                segment_gap = (
                    haversine_miles(current, segment_coords[0])
                    if current is not None and segment_coords
                    else float("inf")
                )
                link_gap = (
                    haversine_miles(current, link_coords[0])
                    if current is not None and link_coords
                    else 0.0
                )
                skip_stale_link = (
                    current is not None
                    and segment_coords
                    and link_coords
                    and segment_gap <= max(stitch_gap_threshold_miles * 2.0, 0.1)
                    and segment_gap <= link_gap
                )
                if not link_is_unsafe and not skip_stale_link:
                    dedupe_append(coords, link_coords)
                next_link_index += 1
                segment_coords = orient_path_to_current(
                    coords[-1] if coords else None,
                    segment_coords,
                    enabled=should_auto_orient_segment(candidate, official_index[seg_id], seg_id),
                )
        if coords and segment_coords and haversine_miles(coords[-1], segment_coords[0]) > stitch_gap_threshold_miles:
            gap = haversine_miles(coords[-1], segment_coords[0])
            stitch = shortest_connector_path(
                coords[-1],
                segment_coords[0],
                connector_graph,
                stitch_snap_tolerance_miles,
            )
            if stitch:
                dedupe_append(
                    coords,
                    path_coordinate_tuples(
                        stitch.get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                )
                segment_coords = orient_path_to_current(
                    coords[-1] if coords else None,
                    segment_coords,
                    enabled=should_auto_orient_segment(candidate, official_index[seg_id], seg_id),
                )
            elif fail_on_unstitched_gap:
                raise_unstitched_gap(candidate, coords[-1], segment_coords[0], gap)
        dedupe_append(coords, segment_coords)
        previous_trail = trail_name
        previous_segment = segment
        if str(segment.get("seg_id") or "").isdigit():
            earned_segment_ids.add(int(segment["seg_id"]))

    return_path = path_coordinate_tuples(
        (candidate.get("return_to_car") or {}).get("path_coordinates") or [],
        densify_source_lines=densify_source_lines,
        densify_max_gap_miles=densify_max_gap_miles,
    )
    trailhead = candidate.get("trailhead") or {}
    trailhead_point = None
    if trailhead.get("lon") is not None and trailhead.get("lat") is not None:
        trailhead_point = (float(trailhead["lon"]), float(trailhead["lat"]))
    trailhead_return_path = path_coordinate_tuples(
        trailhead_access.get("return_path_coordinates") or [],
        densify_source_lines=densify_source_lines,
        densify_max_gap_miles=densify_max_gap_miles,
    )
    if coords and trailhead_point and connector_graph:
        direct_return = shortest_connector_path(
            coords[-1],
            trailhead_point,
            connector_graph,
            stitch_snap_tolerance_miles,
        )
        if direct_return:
            return_path = path_coordinate_tuples(
                direct_return.get("path_coordinates") or [],
                densify_source_lines=densify_source_lines,
                densify_max_gap_miles=densify_max_gap_miles,
            )
            trailhead_return_path = []
    if coords and return_path:
        current = coords[-1]
        return_path = orient_path_to_current(current, return_path)
        already_at_trailhead_return = (
            bool(trailhead_return_path)
            and haversine_miles(current, trailhead_return_path[0]) <= stitch_gap_threshold_miles
        )
        if already_at_trailhead_return:
            return_path = []
        elif haversine_miles(current, return_path[0]) > stitch_gap_threshold_miles:
            if haversine_miles(current, return_path[-1]) <= stitch_gap_threshold_miles:
                return_path = list(reversed(return_path))
            else:
                gap = haversine_miles(current, return_path[0])
                stitch = shortest_connector_path(
                    current,
                    return_path[0],
                    connector_graph,
                    stitch_snap_tolerance_miles,
                )
                if stitch:
                    dedupe_append(
                        coords,
                        path_coordinate_tuples(
                            stitch.get("path_coordinates") or [],
                            densify_source_lines=densify_source_lines,
                            densify_max_gap_miles=densify_max_gap_miles,
                        ),
                    )
                elif fail_on_unstitched_gap:
                    raise_unstitched_gap(candidate, current, return_path[0], gap)
    dedupe_append(coords, return_path)
    trailhead_return_path = orient_path_to_current(coords[-1] if coords else None, trailhead_return_path)
    if coords and trailhead_return_path and haversine_miles(coords[-1], trailhead_return_path[0]) > stitch_gap_threshold_miles:
        if haversine_miles(coords[-1], trailhead_return_path[-1]) <= stitch_gap_threshold_miles:
            trailhead_return_path = list(reversed(trailhead_return_path))
        else:
            gap = haversine_miles(coords[-1], trailhead_return_path[0])
            stitch = shortest_connector_path(
                coords[-1],
                trailhead_return_path[0],
                connector_graph,
                stitch_snap_tolerance_miles,
            )
            if stitch:
                dedupe_append(
                    coords,
                    path_coordinate_tuples(
                        stitch.get("path_coordinates") or [],
                        densify_source_lines=densify_source_lines,
                        densify_max_gap_miles=densify_max_gap_miles,
                    ),
                )
            elif fail_on_unstitched_gap:
                raise_unstitched_gap(candidate, coords[-1], trailhead_return_path[0], gap)
    dedupe_append(coords, trailhead_return_path)
    return stitch_remaining_coordinate_gaps(
        coords,
        connector_graph,
        stitch_gap_threshold_miles=stitch_gap_threshold_miles,
        stitch_snap_tolerance_miles=stitch_snap_tolerance_miles,
        densify_source_lines=densify_source_lines,
        densify_max_gap_miles=densify_max_gap_miles,
        candidate=candidate,
        fail_on_unstitched_gap=fail_on_unstitched_gap,
    )


def recommendation_candidate_ids(recommendation: dict[str, Any]) -> list[str]:
    source = recommendation.get("source") or recommendation
    if source.get("outing_ids"):
        return [str(candidate_id) for candidate_id in source["outing_ids"]]
    if recommendation.get("id"):
        return [str(recommendation["id"])]
    if source.get("candidate_id"):
        return [str(source["candidate_id"])]
    return []


def recommendation_track_coordinates(
    recommendation: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    densify_source_lines: bool = False,
) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for candidate_id in recommendation_candidate_ids(recommendation):
        candidate = candidate_index[candidate_id]
        dedupe_append(
            coords,
            candidate_track_coordinates(
                candidate,
                official_index,
                connector_graph=connector_graph,
                densify_source_lines=densify_source_lines,
            ),
        )
    return coords


def recommendation_track_segments(
    recommendation: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    official_index: dict[int, dict[str, Any]],
    connector_graph: dict[str, Any] | None = None,
    densify_source_lines: bool = False,
) -> list[list[tuple[float, float]]]:
    return [
        candidate_track_coordinates(
            candidate_index[candidate_id],
            official_index,
            connector_graph=connector_graph,
            densify_source_lines=densify_source_lines,
        )
        for candidate_id in recommendation_candidate_ids(recommendation)
    ]


def render_gpx_segments(name: str, track_segments: list[list[tuple[float, float]]]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="boise-trails-ai" xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk>",
        f"    <name>{escape(name)}</name>",
    ]
    for coords in track_segments:
        lines.append("    <trkseg>")
        for lon, lat in coords:
            lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}" />')
        lines.append("    </trkseg>")
    lines.extend(
        [
            "  </trk>",
            "</gpx>",
            "",
        ]
    )
    return "\n".join(lines)


def render_gpx(name: str, coords: list[tuple[float, float]]) -> str:
    return render_gpx_segments(name, [coords])


def validate_track_segments(
    track_segments: list[list[tuple[float, float]]],
    max_gap_miles: float = 0.05,
) -> dict[str, Any]:
    failures = []
    max_gap = 0.0
    point_count = 0
    for segment_index, coords in enumerate(track_segments):
        point_count += len(coords)
        if len(coords) < 2:
            failures.append(
                {
                    "code": "track_segment_too_short",
                    "track_segment_index": segment_index,
                    "point_count": len(coords),
                }
            )
            continue
        for point_index, (left, right) in enumerate(zip(coords, coords[1:])):
            gap = haversine_miles(left, right)
            max_gap = max(max_gap, gap)
            if gap > max_gap_miles:
                failures.append(
                    {
                        "code": "max_trackpoint_gap_exceeded",
                        "track_segment_index": segment_index,
                        "point_index": point_index,
                        "gap_miles": round(gap, 4),
                        "max_allowed_gap_miles": max_gap_miles,
                    }
                )
    if not track_segments:
        failures.append({"code": "no_track_segments"})
    return {
        "passed": not failures,
        "track_segment_count": len(track_segments),
        "point_count": point_count,
        "max_trackpoint_gap_miles": round(max_gap, 4),
        "max_allowed_gap_miles": max_gap_miles,
        "failures": failures,
    }


def selected_menu(execution: dict[str, Any], menu_name: str) -> dict[str, Any]:
    if menu_name == "single-car":
        return execution["single_car_menu"]["recommended_by_bucket"]
    if menu_name == "best-executable":
        return execution["best_executable_menu"]["recommended_by_bucket"]
    raise ValueError(f"Unsupported menu: {menu_name}")


def export_recommendation_gpx(
    plan: dict[str, Any],
    execution: dict[str, Any],
    official_index: dict[int, dict[str, Any]],
    output_dir: Path,
    menu_name: str,
    max_gap_miles: float = 0.05,
    connector_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_index = load_candidate_index(plan)
    routes = []
    for bucket, recommendation in selected_menu(execution, menu_name).items():
        track_segments = recommendation_track_segments(
            recommendation,
            candidate_index,
            official_index,
            connector_graph=connector_graph,
            densify_source_lines=True,
        )
        point_count = sum(len(coords) for coords in track_segments)
        validation = validate_track_segments(track_segments, max_gap_miles=max_gap_miles)
        route_name = f"{bucket}: {', '.join(recommendation.get('trail_names') or [])}"
        filename = f"{bucket}-{slugify(str(recommendation.get('id') or route_name))}.gpx"
        output_path = output_dir / filename
        output_path.write_text(render_gpx_segments(route_name, track_segments), encoding="utf-8")
        routes.append(
            {
                "bucket": bucket,
                "recommendation_type": recommendation.get("recommendation_type"),
                "id": recommendation.get("id"),
                "trail_names": recommendation.get("trail_names"),
                "official_new_miles": recommendation.get("official_new_miles"),
                "simulated_total_minutes": recommendation.get("simulated_total_minutes"),
                "track_segment_count": len(track_segments),
                "point_count": point_count,
                "path": str(output_path),
                "gpx_validation": validation,
            }
        )
    return {
        "menu": menu_name,
        "summary": {
            "route_count": len(routes),
            "gpx_validation_passed": all(route["gpx_validation"]["passed"] for route in routes),
            "failed_route_count": len([route for route in routes if not route["gpx_validation"]["passed"]]),
            "max_allowed_gap_miles": max_gap_miles,
        },
        "routes": routes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--execution-json", type=Path, default=DEFAULT_EXECUTION_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--menu", choices=["single-car", "best-executable"], default="single-car")
    parser.add_argument("--max-gap-miles", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = read_json(args.plan_json)
    execution = read_json(args.execution_json)
    official_index = load_official_segment_index(args.official_geojson)
    official_segments, _official_meta = load_official_segments(args.official_geojson)
    connector_meta = ((plan.get("source_datasets") or {}).get("connector_geojson") or {})
    connector_path = Path(str(connector_meta.get("path"))) if connector_meta.get("path") else None
    connector_graph = (
        load_connector_graph(connector_path, official_segments=official_segments)
        if connector_path and connector_path.exists()
        else None
    )
    manifest = export_recommendation_gpx(
        plan,
        execution,
        official_index,
        args.output_dir,
        args.menu,
        max_gap_miles=args.max_gap_miles,
        connector_graph=connector_graph,
    )
    manifest_path = args.output_dir / f"{args.menu}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    artifact_manifest_path = args.output_dir / f"{args.menu}-artifact-manifest.json"
    write_manifest(
        artifact_manifest_path,
        build_artifact_manifest(
            run_id=str(plan.get("run_id") or execution.get("run_id") or args.menu),
            inputs=[args.plan_json, args.execution_json, args.official_geojson],
            outputs=[manifest_path] + [Path(route["path"]) for route in manifest["routes"]],
            command="export_execution_gpx.py",
            metadata={"menu": args.menu},
        ),
    )
    print(f"Wrote {manifest['summary']['route_count']} GPX files to {args.output_dir}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {artifact_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
