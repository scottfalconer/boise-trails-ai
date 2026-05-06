#!/usr/bin/env python3
"""Challenge high-overhead field-menu routes against generated alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_MAP_DATA_JSON = REPO_ROOT / "outing-menu-map-data.json"
DEFAULT_OFFICIAL_SEGMENTS_GEOJSON = (
    YEAR_DIR / "inputs" / "official" / "api-pull-2026-05-04" / "official_foot_segments.geojson"
)
DEFAULT_ROUTE_BLOCKS_DIR = YEAR_DIR / "outputs" / "private" / "route-blocks"
DEFAULT_CANDIDATE_SOURCES = [
    DEFAULT_ROUTE_BLOCKS_DIR / "block-route-candidate-pass-v1.json",
    DEFAULT_ROUTE_BLOCKS_DIR / "block-assembled-route-pass-v1.json",
    DEFAULT_ROUTE_BLOCKS_DIR / "block-hybrid-route-pass-v1.json",
]
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "route-alternative-challenge-2026-05-06"

HIGH_RATIO_THRESHOLD = 2.0
HIGH_OVERHEAD_MILES = 6.0
MEANINGFUL_IMPROVEMENT_MILES = 0.25


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def int_or_none(value: Any) -> int | None:
    number = as_float(value)
    return int(round(number)) if number is not None else None


def normalize_trailhead(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def unique_ints(values: list[Any]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        as_number = as_int(value)
        if as_number is None or as_number in seen:
            continue
        seen.add(as_number)
        result.append(as_number)
    return result


def segment_ids_for(item: dict[str, Any]) -> list[int]:
    direct = unique_ints(item.get("segment_ids") or [])
    if direct:
        return direct
    from_segments = []
    for segment in item.get("segments") or []:
        from_segments.append(segment.get("seg_id") or segment.get("segId"))
    return unique_ints(from_segments)


def trail_names_for(item: dict[str, Any]) -> list[str]:
    direct = [str(name) for name in item.get("trail_names") or [] if name]
    if direct:
        return direct
    seen: set[str] = set()
    names: list[str] = []
    for segment in item.get("segments") or []:
        name = str(segment.get("trail_name") or "")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def official_segment_miles(official_segments_geojson: dict[str, Any]) -> dict[int, float]:
    miles_by_id: dict[int, float] = {}
    for feature in official_segments_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = as_int(props.get("segId") or props.get("seg_id"))
        length_ft = as_float(props.get("LengthFt"))
        if seg_id is not None and length_ft is not None:
            miles_by_id[seg_id] = length_ft / 5280.0
    return miles_by_id


def sum_segment_miles(segment_ids: list[int], segment_miles: dict[int, float]) -> float | None:
    values = [segment_miles[seg_id] for seg_id in segment_ids if seg_id in segment_miles]
    if not values:
        return None
    return sum(values)


def official_miles_for(item: dict[str, Any], segment_miles: dict[int, float]) -> float | None:
    for key in ("official_miles", "official_new_miles"):
        value = as_float(item.get(key))
        if value is not None and value > 0:
            return value
    segment_values = []
    seen: set[int] = set()
    for segment in item.get("segments") or []:
        seg_id = as_int(segment.get("seg_id") or segment.get("segId"))
        if seg_id is None or seg_id in seen:
            continue
        seen.add(seg_id)
        miles = as_float(segment.get("official_miles"))
        if miles is not None:
            segment_values.append(miles)
    if segment_values:
        return sum(segment_values)
    return sum_segment_miles(segment_ids_for(item), segment_miles)


def on_foot_miles_for(item: dict[str, Any]) -> float | None:
    for key in ("on_foot_miles", "estimated_total_on_foot_miles", "total_on_foot_miles"):
        value = as_float(item.get(key))
        if value is not None and value > 0:
            return value
    return None


def candidate_effort_fields(item: dict[str, Any]) -> dict[str, Any]:
    effort = item.get("effort") or {}
    time_estimates = item.get("time_estimates_minutes") or {}
    return {
        "official_repeat_miles": round_or_none(as_float(item.get("official_repeat_miles"))),
        "connector_miles": round_or_none(as_float(item.get("connector_miles"))),
        "road_miles": round_or_none(as_float(item.get("road_miles"))),
        "raw_total_minutes": as_int(item.get("raw_total_minutes")),
        "door_to_door_p50_minutes": as_int(time_estimates.get("door_to_door_p50")),
        "door_to_door_p75_minutes": as_int(time_estimates.get("door_to_door_p75")),
        "door_to_door_p90_minutes": as_int(time_estimates.get("door_to_door_p90")),
        "moving_raw_minutes": as_int(time_estimates.get("moving_raw")),
        "moving_effort_p50_minutes": as_int(
            time_estimates.get("moving_effort_p50") or effort.get("estimated_moving_minutes_p50")
        ),
        "moving_effort_p75_minutes": as_int(
            time_estimates.get("moving_effort_p75") or effort.get("estimated_moving_minutes_p75")
        ),
        "route_finding_penalty_minutes": as_int(time_estimates.get("route_finding_penalty")),
        "ascent_ft": int_or_none(effort.get("ascent_ft")),
        "descent_ft": int_or_none(effort.get("descent_ft")),
        "grade_adjusted_miles": round_or_none(as_float(effort.get("grade_adjusted_miles"))),
        "effort_score": int_or_none(effort.get("effort_score")),
        "elevation_source": effort.get("elevation_source"),
    }


def navigation_path_ready_for_item(item: dict[str, Any]) -> bool:
    trailhead_access = item.get("trailhead_access") or {}
    if float(trailhead_access.get("mapped_access_miles") or 0.0) > 0.05 and not trailhead_access.get(
        "outbound_path_coordinates"
    ):
        return False
    return_to_car = item.get("return_to_car") or {}
    return_distance = sum(
        float(return_to_car.get(key) or 0.0)
        for key in ["official_repeat_miles", "connector_miles", "road_miles"]
    )
    if return_distance > 0.05 and not return_to_car.get("path_coordinates"):
        return False
    for link in ((item.get("between_trail_links") or {}).get("links") or []):
        if float(link.get("distance_miles") or 0.0) > 0.05 and not link.get("path_coordinates"):
            return False
    return True


def effort_fields_from_route_cue(route_cue: dict[str, Any] | None) -> dict[str, Any]:
    if not route_cue:
        return {}
    segments = route_cue.get("segments") or []
    time_estimates = route_cue.get("time_estimates_minutes") or {}
    effort = route_cue.get("effort") or {}
    ascent = sum(float(segment.get("ascent_ft") or 0.0) for segment in segments)
    descent = sum(float(segment.get("descent_ft") or 0.0) for segment in segments)
    grade_adjusted = sum(float(segment.get("grade_adjusted_miles") or 0.0) for segment in segments)
    return {
        "raw_total_minutes": as_int(route_cue.get("raw_total_minutes")),
        "door_to_door_p50_minutes": as_int(time_estimates.get("door_to_door_p50")),
        "door_to_door_p75_minutes": as_int(time_estimates.get("door_to_door_p75")),
        "door_to_door_p90_minutes": as_int(time_estimates.get("door_to_door_p90")),
        "moving_raw_minutes": as_int(time_estimates.get("moving_raw")),
        "moving_effort_p50_minutes": as_int(time_estimates.get("moving_effort_p50")),
        "moving_effort_p75_minutes": as_int(time_estimates.get("moving_effort_p75")),
        "route_finding_penalty_minutes": as_int(time_estimates.get("route_finding_penalty")),
        "ascent_ft": int_or_none(effort.get("ascent_ft")) or (int(round(ascent)) if segments else None),
        "descent_ft": int_or_none(effort.get("descent_ft")) or (int(round(descent)) if segments else None),
        "grade_adjusted_miles": round_or_none(as_float(effort.get("grade_adjusted_miles")))
        or (round(grade_adjusted, 2) if segments else None),
        "elevation_source": effort.get("elevation_source")
        or ("dem" if any(segment.get("elevation_source") == "dem" for segment in segments) else None),
    }


def merge_missing_effort_fields(candidate: dict[str, Any], fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        if value is not None and candidate.get(key) is None:
            candidate[key] = value


def candidate_richness(candidate: dict[str, Any]) -> tuple[int, int, float]:
    effort_keys = [
        "ascent_ft",
        "descent_ft",
        "grade_adjusted_miles",
        "moving_effort_p75_minutes",
        "door_to_door_p75_minutes",
        "route_finding_penalty_minutes",
    ]
    logistics_keys = ["official_repeat_miles", "connector_miles", "road_miles", "raw_total_minutes"]
    return (
        sum(candidate.get(key) is not None for key in effort_keys),
        sum(candidate.get(key) is not None for key in logistics_keys),
        -float(candidate.get("on_foot_miles") or 9999.0),
    )


def normalize_candidate(
    item: dict[str, Any],
    *,
    source_name: str,
    segment_miles: dict[int, float],
    key: str | None = None,
) -> dict[str, Any] | None:
    segment_ids = segment_ids_for(item)
    official_miles = official_miles_for(item, segment_miles)
    on_foot_miles = on_foot_miles_for(item)
    if not segment_ids or official_miles is None or on_foot_miles is None:
        return None
    candidate_id = str(item.get("candidate_id") or key or "")
    total_minutes = as_int(item.get("total_minutes") or item.get("door_to_door_minutes"))
    ratio = on_foot_miles / official_miles if official_miles else None
    normalized = {
        "source": source_name,
        "candidate_id": candidate_id,
        "candidate_type": item.get("candidate_type") or "route",
        "block_id": item.get("block_id"),
        "block_name": item.get("block_name") or "",
        "trailhead": normalize_trailhead(item.get("trailhead")),
        "trail_names": trail_names_for(item),
        "segment_ids": segment_ids,
        "segment_count": len(segment_ids),
        "official_miles": round(official_miles, 2),
        "on_foot_miles": round(on_foot_miles, 2),
        "overhead_miles": round(on_foot_miles - official_miles, 2),
        "ratio": round(ratio, 3) if ratio else None,
        "total_minutes": total_minutes,
        "route_status": item.get("route_status"),
        "navigation_path_ready": navigation_path_ready_for_item(item),
        **candidate_effort_fields(item),
    }
    if normalized.get("door_to_door_p75_minutes") is None and total_minutes is not None:
        normalized["door_to_door_p75_minutes"] = total_minutes
    return normalized


def iter_source_candidates(payload: dict[str, Any], source_name: str, segment_miles: dict[int, float]):
    for item in payload.get("routes") or []:
        candidate = normalize_candidate(item, source_name=source_name, segment_miles=segment_miles)
        if candidate:
            yield candidate
    for key, item in (payload.get("candidate_index") or {}).items():
        candidate = normalize_candidate(item, source_name=source_name, segment_miles=segment_miles, key=key)
        if candidate:
            yield candidate


def load_candidate_universe(source_paths: list[Path], segment_miles: dict[int, float]) -> list[dict[str, Any]]:
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in source_paths:
        payload = read_json(path)
        source_name = path.stem
        for candidate in iter_source_candidates(payload, source_name, segment_miles):
            dedupe_key = (candidate["source"], candidate["candidate_id"])
            existing = candidates_by_key.get(dedupe_key)
            if existing is None or candidate_richness(candidate) > candidate_richness(existing):
                candidates_by_key[dedupe_key] = candidate
    return list(candidates_by_key.values())


def is_manually_challenged(component: dict[str, Any], package: dict[str, Any]) -> bool:
    return component.get("route_design_status") == "gpx_generated_parking_manual" or package.get("planning_status") in {
        "accepted_manual_split_parking_manual",
        "accepted_manual_override",
    }


def selected_components(map_data: dict[str, Any], segment_miles: dict[int, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    route_cues = map_data.get("route_cues") or {}
    for package in map_data.get("packages") or []:
        for component in package.get("components") or []:
            normalized = normalize_candidate(
                component,
                source_name="current_field_menu",
                segment_miles=segment_miles,
                key=component.get("candidate_id"),
            )
            if not normalized:
                continue
            merge_missing_effort_fields(
                normalized,
                effort_fields_from_route_cue(route_cues.get(str(component.get("candidate_id") or ""))),
            )
            normalized.update(
                {
                    "label": str(
                        component.get("field_menu_label")
                        or component.get("label")
                        or package.get("package_number")
                        or ""
                    ),
                    "package_number": package.get("package_number"),
                    "block_name": package.get("block_name") or normalized.get("block_name") or "",
                    "planning_status": package.get("planning_status"),
                    "route_design_status": component.get("route_design_status"),
                    "manually_challenged": is_manually_challenged(component, package),
                }
            )
            rows.append(normalized)
    return rows


def challenge_targets(
    components: list[dict[str, Any]],
    *,
    high_ratio_threshold: float = HIGH_RATIO_THRESHOLD,
    high_overhead_miles: float = HIGH_OVERHEAD_MILES,
) -> list[dict[str, Any]]:
    targets = []
    for component in components:
        if component.get("manually_challenged"):
            continue
        ratio = as_float(component.get("ratio")) or 0.0
        overhead = as_float(component.get("overhead_miles")) or 0.0
        if ratio > high_ratio_threshold or overhead >= high_overhead_miles:
            targets.append(component)
    return sorted(targets, key=lambda row: float(row.get("overhead_miles") or 0.0), reverse=True)


def compact_candidate(candidate: dict[str, Any], target_ids: set[int] | None = None) -> dict[str, Any]:
    row = {
        "source": candidate.get("source"),
        "candidate_id": candidate.get("candidate_id"),
        "trailhead": candidate.get("trailhead"),
        "official_miles": candidate.get("official_miles"),
        "on_foot_miles": candidate.get("on_foot_miles"),
        "overhead_miles": candidate.get("overhead_miles"),
        "ratio": candidate.get("ratio"),
        "total_minutes": candidate.get("total_minutes"),
        "door_to_door_p75_minutes": candidate.get("door_to_door_p75_minutes"),
        "moving_effort_p75_minutes": candidate.get("moving_effort_p75_minutes"),
        "route_finding_penalty_minutes": candidate.get("route_finding_penalty_minutes"),
        "ascent_ft": candidate.get("ascent_ft"),
        "descent_ft": candidate.get("descent_ft"),
        "grade_adjusted_miles": candidate.get("grade_adjusted_miles"),
        "elevation_source": candidate.get("elevation_source"),
        "trail_names": candidate.get("trail_names"),
        "segment_count": candidate.get("segment_count"),
    }
    if target_ids is not None:
        candidate_ids = set(candidate.get("segment_ids") or [])
        overlap = candidate_ids & target_ids
        row["overlap_segment_count"] = len(overlap)
        row["extra_segment_count"] = len(candidate_ids - target_ids)
        row["missing_segment_count"] = len(target_ids - candidate_ids)
    return row


def best_by_on_foot(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("on_foot_miles") or 9999.0),
            int(row.get("total_minutes") or 99999),
            str(row.get("candidate_id") or ""),
        ),
    )[0]


def challenge_target(target: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    target_ids = set(target.get("segment_ids") or [])
    current_id = target.get("candidate_id")
    exact = []
    supersets = []
    overlaps = []
    for candidate in candidates:
        candidate_ids = set(candidate.get("segment_ids") or [])
        if not candidate_ids or candidate_ids == target_ids and candidate.get("source") == "current_field_menu":
            continue
        overlap = candidate_ids & target_ids
        if not overlap:
            continue
        if candidate_ids == target_ids:
            exact.append(candidate)
        elif target_ids <= candidate_ids:
            supersets.append(candidate)
        elif len(overlap) / len(target_ids) >= 0.65:
            overlaps.append(candidate)

    best_exact = best_by_on_foot(exact)
    best_superset = best_by_on_foot(supersets)
    current_on_foot = float(target.get("on_foot_miles") or 0.0)
    status = "no_exact_alternative_in_existing_universe"
    recommendation = "Manual map review still required; no generated route covers the same official segments for direct comparison."
    if best_exact:
        improvement = current_on_foot - float(best_exact.get("on_foot_miles") or 0.0)
        if best_exact.get("candidate_id") != current_id and improvement > MEANINGFUL_IMPROVEMENT_MILES:
            status = "better_exact_candidate_found"
            recommendation = "Replace or re-export this outing from the better exact candidate before calling the route efficient."
        else:
            status = "current_best_exact_candidate_in_existing_universe"
            recommendation = "No generated exact alternative beats the selected route; manual/local map review is still needed for absolute efficiency."
    if best_superset and float(best_superset.get("on_foot_miles") or 9999.0) < current_on_foot - MEANINGFUL_IMPROVEMENT_MILES:
        recommendation = (
            "A generated superset covers this route's segments with fewer on-foot miles; review whether it can replace this outing "
            "and absorb the extra official work."
        )
        if status != "better_exact_candidate_found":
            status = "better_superset_candidate_found"

    overlap_rows = sorted(
        overlaps,
        key=lambda row: (
            -len(set(row.get("segment_ids") or []) & target_ids),
            float(row.get("on_foot_miles") or 9999.0),
        ),
    )[:5]

    return {
        "label": target.get("label"),
        "package_number": target.get("package_number"),
        "candidate_id": current_id,
        "block_name": target.get("block_name"),
        "trailhead": target.get("trailhead"),
        "trail_names": target.get("trail_names"),
        "selected": compact_candidate(target),
        "challenge_status": status,
        "recommendation": recommendation,
        "exact_alternative_count": len(exact),
        "superset_alternative_count": len(supersets),
        "high_overlap_alternative_count": len(overlaps),
        "best_exact": compact_candidate(best_exact, target_ids) if best_exact else None,
        "best_superset": compact_candidate(best_superset, target_ids) if best_superset else None,
        "top_high_overlap": [compact_candidate(row, target_ids) for row in overlap_rows],
    }


def build_report(
    map_data: dict[str, Any],
    official_segments: dict[str, Any],
    candidate_sources: list[Path],
    *,
    high_ratio_threshold: float = HIGH_RATIO_THRESHOLD,
    high_overhead_miles: float = HIGH_OVERHEAD_MILES,
) -> dict[str, Any]:
    segment_miles = official_segment_miles(official_segments)
    components = selected_components(map_data, segment_miles)
    targets = challenge_targets(
        components,
        high_ratio_threshold=high_ratio_threshold,
        high_overhead_miles=high_overhead_miles,
    )
    candidate_universe = load_candidate_universe(candidate_sources, segment_miles)
    challenges = [challenge_target(target, candidate_universe) for target in targets]
    better_exact = [row for row in challenges if row["challenge_status"] == "better_exact_candidate_found"]
    better_superset = [row for row in challenges if row["challenge_status"] == "better_superset_candidate_found"]
    current_best_exact = [
        row for row in challenges if row["challenge_status"] == "current_best_exact_candidate_in_existing_universe"
    ]
    no_exact = [row for row in challenges if row["challenge_status"] == "no_exact_alternative_in_existing_universe"]
    return {
        "objective": "challenge selected high-overhead field-menu outings against the generated candidate universe",
        "target_selection": {
            "ratio_over": high_ratio_threshold,
            "overhead_miles_at_least": high_overhead_miles,
            "manual_routes_excluded": True,
        },
        "source_candidate_files": [str(path) for path in candidate_sources],
        "summary": {
            "selected_component_count": len(components),
            "candidate_universe_count": len(candidate_universe),
            "target_count": len(targets),
            "target_candidate_ids": [row.get("candidate_id") for row in targets],
            "challenged_candidate_ids": [row.get("candidate_id") for row in challenges],
            "better_exact_candidate_count": len(better_exact),
            "better_superset_candidate_count": len(better_superset),
            "current_best_exact_candidate_count": len(current_best_exact),
            "no_exact_alternative_count": len(no_exact),
            "manual_map_review_still_required_count": len(challenges),
            "targets_with_elevation_metrics_count": sum(1 for row in targets if row.get("ascent_ft") is not None),
            "targets_with_p75_time_count": sum(1 for row in targets if row.get("door_to_door_p75_minutes") is not None),
        },
        "challenges": challenges,
        "caveats": [
            "This report only compares already-generated candidates. It does not prove that no better hand-designed GPX exists.",
            "Superset alternatives are not automatic replacements because they may duplicate or move official work from another outing.",
            "A route remains unproven for absolute efficiency until the relevant local-map/GPX design area has also been reviewed.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Route Alternative Challenge",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Selected field-menu components: {summary['selected_component_count']}",
        f"- Generated candidate universe: {summary['candidate_universe_count']}",
        f"- High-overhead targets challenged: {summary['target_count']}",
        f"- Better exact candidates found: {summary['better_exact_candidate_count']}",
        f"- Better superset candidates found: {summary['better_superset_candidate_count']}",
        f"- Current selected route is best exact generated candidate: {summary['current_best_exact_candidate_count']}",
        f"- No exact generated alternative: {summary['no_exact_alternative_count']}",
        f"- Manual map review still required: {summary['manual_map_review_still_required_count']}",
        f"- Targets with DEM elevation metrics: {summary['targets_with_elevation_metrics_count']} / {summary['target_count']}",
        f"- Targets with p75 door-to-door time: {summary['targets_with_p75_time_count']} / {summary['target_count']}",
        "",
        "## Target Results",
        "",
        "| Label | Selected | Status | Best exact | Best superset | Recommendation |",
        "|---|---|---|---|---|---|",
    ]
    for challenge in report["challenges"]:
        selected = challenge["selected"]
        best_exact = challenge.get("best_exact")
        best_superset = challenge.get("best_superset")
        selected_text = (
            f"{selected['official_miles']} official / {selected['on_foot_miles']} on-foot "
            f"({selected['ratio']}x, {selected.get('ascent_ft') or 'n/a'} ft, "
            f"{selected.get('door_to_door_p75_minutes') or selected.get('total_minutes') or 'n/a'} min p75) "
            f"from {selected['trailhead']}"
        )
        exact_text = "none"
        if best_exact:
            exact_text = (
                f"{best_exact['candidate_id']} - {best_exact['official_miles']} official / "
                f"{best_exact['on_foot_miles']} on-foot"
            )
        superset_text = "none"
        if best_superset:
            superset_text = (
                f"{best_superset['candidate_id']} - {best_superset['official_miles']} official / "
                f"{best_superset['on_foot_miles']} on-foot; +{best_superset['extra_segment_count']} segments"
            )
        lines.append(
            f"| {challenge['label']} | {selected_text} | {challenge['challenge_status']} | "
            f"{exact_text} | {superset_text} | {challenge['recommendation']} |"
        )
    lines.extend(["", "## Details", ""])
    for challenge in report["challenges"]:
        lines.extend(
            [
                f"### {challenge['label']} - {challenge['block_name']}",
                "",
                f"- Candidate id: `{challenge['candidate_id']}`",
                f"- Trails: {', '.join(challenge.get('trail_names') or [])}",
                f"- Exact alternatives checked: {challenge['exact_alternative_count']}",
                f"- Superset alternatives checked: {challenge['superset_alternative_count']}",
                f"- High-overlap alternatives checked: {challenge['high_overlap_alternative_count']}",
            ]
        )
        if challenge.get("top_high_overlap"):
            lines.append("- Top high-overlap generated candidates:")
            for row in challenge["top_high_overlap"]:
                lines.append(
                    f"  - `{row['candidate_id']}`: overlaps {row['overlap_segment_count']} segments; "
                    f"{row['official_miles']} official / {row['on_foot_miles']} on-foot from {row['trailhead']}"
                )
        lines.append("")
    lines.extend(["## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data-json", type=Path, default=DEFAULT_MAP_DATA_JSON)
    parser.add_argument("--official-segments-geojson", type=Path, default=DEFAULT_OFFICIAL_SEGMENTS_GEOJSON)
    parser.add_argument("--candidate-source", action="append", type=Path, dest="candidate_sources")
    parser.add_argument("--high-ratio-threshold", type=float, default=HIGH_RATIO_THRESHOLD)
    parser.add_argument("--high-overhead-miles", type=float, default=HIGH_OVERHEAD_MILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_sources = args.candidate_sources or DEFAULT_CANDIDATE_SOURCES
    report = build_report(
        read_json(args.map_data_json),
        read_json(args.official_segments_geojson),
        candidate_sources,
        high_ratio_threshold=args.high_ratio_threshold,
        high_overhead_miles=args.high_overhead_miles,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
