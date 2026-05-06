#!/usr/bin/env python3
"""Find official segments that still lack p90-bounded runnable candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_PERSONAL_ROUTE_MENU_JSON = YEAR_DIR / "outputs" / "private" / "personal-route-menu.json"
DEFAULT_HYBRID_ROUTE_PASS_JSON = (
    YEAR_DIR / "outputs" / "private" / "route-blocks" / "block-hybrid-route-pass-v1.json"
)
DEFAULT_FIELD_MENU_JSON = YEAR_DIR / "outputs" / "private" / "2026-outing-menu-map-data.json"
DEFAULT_STATE_JSON = YEAR_DIR / "inputs" / "personal" / "2026-planner-state.private.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-completion-gap-analysis-2026-05-06"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def official_segments(official_geojson: dict[str, Any]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = as_int(props.get("segId"))
        if seg_id is None:
            continue
        index[seg_id] = {
            "seg_id": seg_id,
            "seg_name": props.get("segName"),
            "trail_name": str(props.get("segName") or "").rsplit(" ", 1)[0],
            "official_miles": round(float(props.get("LengthFt") or 0.0) / 5280.0, 3),
            "direction": props.get("direction") or "both",
        }
    return index


def unique_ints(values: list[Any]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        number = as_int(value)
        if number is None or number in seen:
            continue
        seen.add(number)
        result.append(number)
    return result


def segment_ids_for(item: dict[str, Any]) -> list[int]:
    direct = unique_ints(item.get("segment_ids") or [])
    if direct:
        return direct
    return unique_ints(
        [
            segment.get("seg_id") or segment.get("segId") or segment.get("segment_id")
            for segment in item.get("segments") or []
        ]
    )


def trail_names_for(item: dict[str, Any]) -> list[str]:
    direct = [str(name) for name in item.get("trail_names") or [] if name]
    if direct:
        return direct
    names = []
    seen = set()
    for segment in item.get("segments") or []:
        name = str(segment.get("trail_name") or "")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    title = item.get("title")
    if not names and title:
        names.append(str(title))
    return names


def item_official_miles(item: dict[str, Any], segment_index: dict[int, dict[str, Any]]) -> float:
    for key in ("official_miles", "official_new_miles"):
        value = as_float(item.get(key))
        if value is not None:
            return round(value, 2)
    return round(sum(segment_index[seg_id]["official_miles"] for seg_id in segment_ids_for(item) if seg_id in segment_index), 2)


def item_on_foot_miles(item: dict[str, Any]) -> float | None:
    for key in ("on_foot_miles", "estimated_total_on_foot_miles", "total_on_foot_miles"):
        value = as_float(item.get(key))
        if value is not None:
            return round(value, 2)
    return None


def normalize_trailhead(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def validation_passed(item: dict[str, Any]) -> bool:
    validation = item.get("validation") or {}
    if validation.get("passed") is True:
        return True
    if item.get("route_status") == "graph_validated":
        return True
    required_keys = [
        "segment_coverage_passed",
        "ascent_direction_passed",
        "return_path_graph_validated",
    ]
    return all(validation.get(key) is True for key in required_keys)


def normalize_candidate(
    item: dict[str, Any],
    *,
    source: str,
    segment_index: dict[int, dict[str, Any]],
    candidate_id: str | None = None,
) -> dict[str, Any] | None:
    segment_ids = segment_ids_for(item)
    on_foot_miles = item_on_foot_miles(item)
    if not segment_ids or on_foot_miles is None:
        return None
    time_estimates = item.get("time_estimates_minutes") or {}
    p75 = as_int(time_estimates.get("door_to_door_p75") or item.get("total_minutes"))
    p90 = as_int(time_estimates.get("door_to_door_p90"))
    if p75 is None:
        return None
    if p90 is None:
        p90 = int(round(p75 * 1.12))
    official_miles = item_official_miles(item, segment_index)
    cid = str(candidate_id or item.get("candidate_id") or "")
    return {
        "source": source,
        "candidate_id": cid,
        "label": item.get("label") or item.get("title") or cid,
        "route_status": item.get("route_status"),
        "validation_passed": validation_passed(item),
        "trailhead": normalize_trailhead(item.get("trailhead")),
        "trail_names": trail_names_for(item),
        "segment_ids": sorted(set(segment_ids)),
        "official_miles": official_miles,
        "on_foot_miles": on_foot_miles,
        "door_to_door_p75_minutes": p75,
        "door_to_door_p90_minutes": p90,
        "grade_adjusted_miles": as_float(
            (item.get("effort") or {}).get("grade_adjusted_miles")
            or item.get("grade_adjusted_miles")
        ),
        "ascent_ft": as_int((item.get("effort") or {}).get("ascent_ft") or item.get("ascent_ft")),
        "manual_design_hold": bool(item.get("manual_design_hold")),
    }


def candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return (str(candidate["source"]), str(candidate["candidate_id"]))


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        current = by_key.get(key)
        richness = (
            candidate.get("validation_passed") is True,
            candidate.get("grade_adjusted_miles") is not None,
            -float(candidate.get("door_to_door_p90_minutes") or 99999),
        )
        current_richness = (
            current.get("validation_passed") is True,
            current.get("grade_adjusted_miles") is not None,
            -float(current.get("door_to_door_p90_minutes") or 99999),
        ) if current else None
        if current is None or richness > current_richness:
            by_key[key] = candidate
    return list(by_key.values())


def load_candidates(
    personal_route_menu: dict[str, Any],
    hybrid_route_pass: dict[str, Any],
    field_menu: dict[str, Any],
    segment_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for item in (personal_route_menu.get("route_menu") or {}).get("all_candidates") or []:
        normalized = normalize_candidate(item, source="personal_route_menu", segment_index=segment_index)
        if normalized:
            candidates.append(normalized)
    for item in (hybrid_route_pass.get("candidate_index") or {}).values():
        normalized = normalize_candidate(item, source="hybrid_candidate_index", segment_index=segment_index)
        if normalized:
            candidates.append(normalized)
    for candidate_id, item in (field_menu.get("route_cues") or {}).items():
        normalized = normalize_candidate(
            item,
            source="canonical_field_menu",
            segment_index=segment_index,
            candidate_id=str(candidate_id),
        )
        if normalized:
            route = next(
                (
                    route
                    for route in ((field_menu.get("manifest") or {}).get("routes") or [])
                    if candidate_id in (((route.get("outing") or {}).get("candidate_ids")) or [])
                ),
                None,
            )
            if route:
                normalized["validation_passed"] = ((route.get("validation") or {}).get("passed") is True)
            candidates.append(normalized)
    return dedupe_candidates(candidates)


def usable_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if candidate["validation_passed"] is True
        and candidate["manual_design_hold"] is not True
        and candidate["route_status"] in {None, "graph_validated"}
    ]


def best_candidate_for_segment(
    candidates: list[dict[str, Any]],
    seg_id: int,
) -> dict[str, Any] | None:
    containing = [candidate for candidate in candidates if seg_id in candidate["segment_ids"]]
    if not containing:
        return None
    return min(
        containing,
        key=lambda candidate: (
            candidate["door_to_door_p90_minutes"],
            candidate["door_to_door_p75_minutes"],
            candidate["on_foot_miles"],
            -candidate["official_miles"],
        ),
    )


def solve_maximum_bounded_coverage(
    candidates: list[dict[str, Any]],
    target_segment_ids: list[int],
    p90_bound_minutes: int,
) -> dict[str, Any]:
    bounded = [candidate for candidate in candidates if candidate["door_to_door_p90_minutes"] <= p90_bound_minutes]
    if not bounded:
        return {"success": False, "reason": "no_bounded_candidates"}
    target_set = set(target_segment_ids)
    coverable = {
        seg_id
        for candidate in bounded
        for seg_id in candidate["segment_ids"]
        if seg_id in target_set
    }
    uncovered = set(coverable)
    selected = []
    while uncovered:
        best = max(
            bounded,
            key=lambda candidate: (
                len(set(candidate["segment_ids"]) & uncovered),
                -candidate["door_to_door_p75_minutes"],
                -candidate["door_to_door_p90_minutes"],
                -candidate["on_foot_miles"],
            ),
        )
        newly_covered = set(best["segment_ids"]) & uncovered
        if not newly_covered:
            break
        selected.append(best)
        uncovered -= newly_covered
    covered = sorted(
        {
            seg_id
            for candidate in selected
            for seg_id in candidate["segment_ids"]
            if seg_id in target_set
        }
    )
    return {
        "success": True,
        "selected_candidate_count": len(selected),
        "covered_segment_count": len(covered),
        "missing_segment_count": len(set(target_segment_ids) - set(covered)),
        "total_p75_minutes": int(sum(candidate["door_to_door_p75_minutes"] for candidate in selected)),
        "total_p90_minutes": int(sum(candidate["door_to_door_p90_minutes"] for candidate in selected)),
        "total_on_foot_miles": round(sum(candidate["on_foot_miles"] for candidate in selected), 2),
        "selected_candidates": sorted(
            selected,
            key=lambda candidate: (
                candidate["door_to_door_p90_minutes"],
                candidate["candidate_id"],
            ),
        ),
        "missing_segment_ids": sorted(set(target_segment_ids) - set(covered)),
    }


def build_report(
    *,
    official_geojson: dict[str, Any],
    personal_route_menu: dict[str, Any],
    hybrid_route_pass: dict[str, Any],
    field_menu: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    segment_index = official_segments(official_geojson)
    target_ids = sorted(segment_index)
    availability = state.get("availability_model") or {}
    bounds = {
        "weekday": int(availability.get("weekday_max_minutes") or 0),
        "weekend": int(availability.get("weekend_max_minutes") or 0),
    }
    max_bound = max(bounds.values())
    all_candidates = load_candidates(personal_route_menu, hybrid_route_pass, field_menu, segment_index)
    candidates = usable_candidates(all_candidates)
    all_covered = {seg_id for candidate in candidates for seg_id in candidate["segment_ids"]}
    bounded_by_type: dict[str, dict[str, Any]] = {}
    for name, bound in bounds.items():
        bounded = [candidate for candidate in candidates if candidate["door_to_door_p90_minutes"] <= bound]
        covered = {seg_id for candidate in bounded for seg_id in candidate["segment_ids"]}
        bounded_by_type[name] = {
            "p90_bound_minutes": bound,
            "candidate_count": len(bounded),
            "covered_segment_count": len(covered & set(target_ids)),
            "missing_segment_count": len(set(target_ids) - covered),
            "missing_segment_ids": sorted(set(target_ids) - covered),
        }
    max_bounded = [
        candidate for candidate in candidates if candidate["door_to_door_p90_minutes"] <= max_bound
    ]
    max_bounded_covered = {seg_id for candidate in max_bounded for seg_id in candidate["segment_ids"]}
    missing_under_max = sorted(set(target_ids) - max_bounded_covered)
    missing_rows = []
    by_trail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg_id in missing_under_max:
        best = best_candidate_for_segment(candidates, seg_id)
        segment = segment_index[seg_id]
        row = {
            **segment,
            "best_existing_candidate": {
                key: best.get(key)
                for key in [
                    "source",
                    "candidate_id",
                    "label",
                    "trailhead",
                    "trail_names",
                    "official_miles",
                    "on_foot_miles",
                    "door_to_door_p75_minutes",
                    "door_to_door_p90_minutes",
                    "route_status",
                    "validation_passed",
                ]
            }
            if best
            else None,
        }
        missing_rows.append(row)
        by_trail[segment["trail_name"]].append(row)
    missing_trail_groups = [
        {
            "trail_name": trail_name,
            "segment_count": len(rows),
            "official_miles": round(sum(row["official_miles"] for row in rows), 2),
            "best_existing_p90_min": min(
                (
                    row["best_existing_candidate"]["door_to_door_p90_minutes"]
                    for row in rows
                    if row.get("best_existing_candidate")
                ),
                default=None,
            ),
            "best_existing_p90_max": max(
                (
                    row["best_existing_candidate"]["door_to_door_p90_minutes"]
                    for row in rows
                    if row.get("best_existing_candidate")
                ),
                default=None,
            ),
            "segment_ids": [row["seg_id"] for row in rows],
        }
        for trail_name, rows in sorted(by_trail.items())
    ]
    coverage_solution = solve_maximum_bounded_coverage(candidates, target_ids, max_bound)
    return {
        "objective": "find current official segments that lack graph-validated candidates inside personal p90 bounds",
        "source_files": {
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "personal_route_menu_json": display_path(DEFAULT_PERSONAL_ROUTE_MENU_JSON),
            "hybrid_route_pass_json": display_path(DEFAULT_HYBRID_ROUTE_PASS_JSON),
            "field_menu_json": display_path(DEFAULT_FIELD_MENU_JSON),
            "state_json": display_path(DEFAULT_STATE_JSON),
        },
        "bounds": {
            "weekday_max_minutes": bounds["weekday"],
            "weekend_max_minutes": bounds["weekend"],
            "max_daily_bound_minutes": max_bound,
        },
        "summary": {
            "target_segment_count": len(target_ids),
            "all_usable_candidate_count": len(candidates),
            "all_usable_covered_segment_count": len(all_covered & set(target_ids)),
            "all_usable_missing_segment_count": len(set(target_ids) - all_covered),
            "max_bound_candidate_count": len(max_bounded),
            "max_bound_covered_segment_count": len(max_bounded_covered & set(target_ids)),
            "max_bound_missing_segment_count": len(missing_under_max),
            "missing_trail_count": len(missing_trail_groups),
            "completion_possible_with_existing_bounded_candidates": len(missing_under_max) == 0,
        },
        "bounded_by_day_type": bounded_by_type,
        "maximum_bounded_coverage_solution": coverage_solution,
        "missing_under_max_bound_segments": missing_rows,
        "missing_under_max_bound_trail_groups": missing_trail_groups,
        "caveats": [
            "This analyzes the existing candidate universe only. A missing row means no current graph-validated candidate fits the p90 bound, not that a better hand-designed route is impossible.",
            "The next route-design task should create smaller legal single-car loops for the missing trail groups, or explicitly document why the p90 bound must be relaxed for those segments.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    bounds = report["bounds"]
    lines = [
        "# P90 Completion Gap Analysis",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Target segments: {summary['target_segment_count']}",
        f"- Usable candidates: {summary['all_usable_candidate_count']}",
        f"- Segments covered by any usable candidate: {summary['all_usable_covered_segment_count']}",
        f"- Max p90 bound: {bounds['max_daily_bound_minutes']} min",
        f"- Candidates under max p90 bound: {summary['max_bound_candidate_count']}",
        f"- Segments covered under max p90 bound: {summary['max_bound_covered_segment_count']}",
        f"- Segments missing under max p90 bound: {summary['max_bound_missing_segment_count']}",
        f"- Completion possible with current bounded candidates: {summary['completion_possible_with_existing_bounded_candidates']}",
        "",
        "## Day-Type Coverage",
        "",
        "| Day type | P90 bound | Candidate count | Covered segments | Missing segments |",
        "|---|---:|---:|---:|---:|",
    ]
    for day_type, row in report["bounded_by_day_type"].items():
        lines.append(
            f"| {day_type} | {row['p90_bound_minutes']} | {row['candidate_count']} | "
            f"{row['covered_segment_count']} | {row['missing_segment_count']} |"
        )
    solution = report["maximum_bounded_coverage_solution"]
    lines.extend(
        [
            "",
            "## Best Existing Bounded Coverage",
            "",
        ]
    )
    if solution.get("success"):
        lines.extend(
            [
                f"- Selected candidates: {solution['selected_candidate_count']}",
                f"- Covered segments: {solution['covered_segment_count']}",
                f"- Missing segments: {solution['missing_segment_count']}",
                f"- Total p75 minutes across selected loops: {solution['total_p75_minutes']}",
                f"- Total on-foot miles across selected loops: {solution['total_on_foot_miles']}",
            ]
        )
    else:
        lines.append(f"- Solver failed: {solution.get('reason') or solution.get('message')}")
    lines.extend(
        [
            "",
            "## Missing Trail Groups Under Max P90",
            "",
            "| Trail | Segments | Official mi | Best existing p90 range | Segment ids |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group in report["missing_under_max_bound_trail_groups"]:
        p90_min = group["best_existing_p90_min"]
        p90_max = group["best_existing_p90_max"]
        p90_text = "none" if p90_min is None else f"{p90_min}-{p90_max}"
        lines.append(
            f"| {group['trail_name']} | {group['segment_count']} | {group['official_miles']} | "
            f"{p90_text} | {', '.join(str(seg_id) for seg_id in group['segment_ids'])} |"
        )
    lines.extend(
        [
            "",
            "## Missing Segments",
            "",
            "| Segment | Trail | Official mi | Best existing candidate | P90 | On foot | Trailhead |",
            "|---:|---|---:|---|---:|---:|---|",
        ]
    )
    for row in report["missing_under_max_bound_segments"]:
        best = row.get("best_existing_candidate") or {}
        lines.append(
            f"| {row['seg_id']} | {row['trail_name']} | {row['official_miles']} | "
            f"{best.get('candidate_id') or 'none'} | {best.get('door_to_door_p90_minutes') or ''} | "
            f"{best.get('on_foot_miles') or ''} | {best.get('trailhead') or ''} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--personal-route-menu-json", type=Path, default=DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    parser.add_argument("--hybrid-route-pass-json", type=Path, default=DEFAULT_HYBRID_ROUTE_PASS_JSON)
    parser.add_argument("--field-menu-json", type=Path, default=DEFAULT_FIELD_MENU_JSON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(
        official_geojson=read_json(args.official_geojson),
        personal_route_menu=read_json(args.personal_route_menu_json),
        hybrid_route_pass=read_json(args.hybrid_route_pass_json),
        field_menu=read_json(args.field_menu_json),
        state=read_json(args.state_json),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
