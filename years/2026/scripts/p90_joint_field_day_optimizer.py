#!/usr/bin/env python3
"""Optimize field days directly from the repaired p90 candidate universe."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_repaired_candidate_universe_audit as repaired  # noqa: E402
import p90_repaired_field_day_pack_audit as pack_audit  # noqa: E402
from p90_completion_gap_analyzer import (  # noqa: E402
    load_candidates,
    official_segments,
    usable_candidates,
)
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-joint-field-day-optimizer-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def all_repaired_candidates() -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    official_geojson = read_json(DEFAULT_OFFICIAL_GEOJSON)
    segment_index = official_segments(official_geojson)
    target_ids = sorted(segment_index)
    personal_route_menu = read_json(repaired.DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    hybrid_route_pass = read_json(repaired.DEFAULT_HYBRID_ROUTE_PASS_JSON)
    field_menu = read_json(repaired.DEFAULT_FIELD_MENU_JSON)
    existing = usable_candidates(load_candidates(personal_route_menu, hybrid_route_pass, field_menu, segment_index))
    probes = repaired.strict_probe_candidates(
        read_json(repaired.DEFAULT_SPLIT_PROBE_JSON),
        read_json(repaired.DEFAULT_FORCED_PROBE_JSON),
    )
    return target_ids, existing + probes, field_menu


def candidate_loop_records(
    candidates: list[dict[str, Any]],
    field_menu: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = pack_audit.candidate_key(candidate)
        current = deduped.get(key)
        if current is None or (
            int(candidate["door_to_door_p75_minutes"]),
            int(candidate["door_to_door_p90_minutes"]),
            float(candidate["on_foot_miles"]),
        ) < (
            int(current["door_to_door_p75_minutes"]),
            int(current["door_to_door_p90_minutes"]),
            float(current["on_foot_miles"]),
        ):
            deduped[key] = candidate
    return pack_audit.loop_records(list(deduped.values()), field_menu, state)


def add_field_day_candidate(
    field_days: list[dict[str, Any]],
    *,
    loops: list[dict[str, Any]],
    combo: tuple[int, ...],
    day: dict[str, Any],
    day_type: str,
    bound: int,
) -> None:
    segment_ids = sorted(
        {
            int(seg_id)
            for index in combo
            for seg_id in loops[index]["segment_ids"]
        }
    )
    field_days.append(
        {
            "field_day_id": f"{day_type}-" + "--".join(day["order"]),
            "day_type": day_type,
            "loop_ids": day["order"],
            "segment_ids": segment_ids,
            "loop_count": len(combo),
            "p75_minutes": day["p75_minutes"],
            "p90_minutes": day["p90_minutes"],
            "p90_bound_minutes": bound,
            "stress": round(day["p90_minutes"] / bound, 3),
            "drive_minutes": day["drive_minutes"],
            "between_drive_minutes": day["between_drive_minutes"],
            "on_foot_miles": day["on_foot_miles"],
            "grade_adjusted_miles": day["grade_adjusted_miles"],
            "parking_risk": day["parking_risk"],
        }
    )


def nearby_pairs(
    loops: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    neighbor_limit: int,
) -> dict[int, list[int]]:
    drive_model = state.get("drive_model") or {}
    acceptable_drive = int((state.get("availability_model") or {}).get("acceptable_inter_trailhead_drive_minutes") or 20)
    result: dict[int, list[int]] = {}
    for index, loop in enumerate(loops):
        left = (float(loop["parking"]["lon"]), float(loop["parking"]["lat"]))
        neighbors = []
        for other_index, other in enumerate(loops):
            if other_index == index:
                continue
            right = (float(other["parking"]["lon"]), float(other["parking"]["lat"]))
            minutes = pack_audit.drive_minutes_between(left, right, drive_model, apply_minimum=False)
            if minutes <= acceptable_drive:
                neighbors.append((minutes, other_index))
        result[index] = [other for _minutes, other in sorted(neighbors)[:neighbor_limit]]
    return result


def generate_direct_field_day_candidates(
    loops: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
    connected_expansion_limit: int | None = None,
) -> list[dict[str, Any]]:
    bounds = {"weekday": weekday_bound, "weekend": weekend_bound}
    max_bound = max(bounds.values())
    field_days: list[dict[str, Any]] = []
    neighbors = nearby_pairs(loops, state, neighbor_limit=neighbor_limit)
    neighbor_sets = {index: set(values) for index, values in neighbors.items()}
    neighbor_ranks = {
        index: {neighbor: rank for rank, neighbor in enumerate(values)}
        for index, values in neighbors.items()
    }
    expansion_limit = int(connected_expansion_limit or neighbor_limit)
    seen_combos: set[tuple[int, ...]] = set()
    expandable_combos: set[tuple[int, ...]] = set()

    def combo_min_possible_p90(combo: tuple[int, ...]) -> int:
        return sum(int(loops[index]["internal_p75_minutes"]) + int(loops[index]["p90_delta_minutes"]) for index in combo)

    def maybe_add(combo: tuple[int, ...]) -> bool:
        combo = tuple(sorted(combo))
        if combo in seen_combos:
            return False
        seen_combos.add(combo)
        if combo_min_possible_p90(combo) > max_bound:
            return False
        day = pack_audit.best_field_day_for_combo(loops, combo, state)
        is_expandable = day["p90_minutes"] <= max_bound
        for day_type, bound in bounds.items():
            if day["p90_minutes"] <= bound:
                add_field_day_candidate(field_days, loops=loops, combo=combo, day=day, day_type=day_type, bound=bound)
        if is_expandable:
            expandable_combos.add(combo)
        return is_expandable

    frontier: list[tuple[int, ...]] = []
    for index, loop in enumerate(loops):
        if int(loop["door_to_door_p90_minutes"]) <= max_bound:
            if maybe_add((index,)):
                frontier.append((index,))

    for size in range(2, max_combo_size + 1):
        for left, right_neighbors in neighbors.items():
            limited = [index for index in right_neighbors if index > left]
            for rest in itertools.combinations(limited, size - 1):
                combo = (left, *rest)
                if not all(
                    right in neighbor_sets.get(other_left, set())
                    or other_left in neighbor_sets.get(right, set())
                    for other_left, right in itertools.combinations(combo, 2)
                ):
                    continue
                maybe_add(combo)

    for size in range(2, max_combo_size + 1):
        next_frontier: list[tuple[int, ...]] = []
        for combo in frontier:
            expansion_candidates = sorted(set().union(*(neighbor_sets.get(index, set()) for index in combo)) - set(combo))
            expansion_candidates = sorted(
                expansion_candidates,
                key=lambda next_index: (
                    min(neighbor_ranks.get(index, {}).get(next_index, 9999) for index in combo),
                    combo_min_possible_p90(tuple(sorted((*combo, next_index)))),
                    next_index,
                ),
            )[:expansion_limit]
            for next_index in expansion_candidates:
                next_combo = tuple(sorted((*combo, next_index)))
                if len(next_combo) != size:
                    continue
                if maybe_add(next_combo) or next_combo in expandable_combos:
                    next_frontier.append(next_combo)
        frontier = next_frontier

    return field_days


def solve_direct_field_day_cover(
    field_days: list[dict[str, Any]],
    target_ids: list[int],
    counts: dict[str, int],
) -> dict[str, Any]:
    if not field_days:
        return {"success": False, "reason": "no_field_day_candidates"}
    target_index = {seg_id: row for row, seg_id in enumerate(target_ids)}
    coverable = {
        seg_id
        for field_day in field_days
        for seg_id in field_day["segment_ids"]
        if seg_id in target_index
    }
    if set(target_ids) - coverable:
        return {
            "success": False,
            "reason": "not_all_segments_coverable_by_field_day_candidates",
            "missing_segment_ids": sorted(set(target_ids) - coverable),
        }
    matrix = lil_matrix((len(target_ids) + 2, len(field_days)), dtype=float)
    lower = [1.0] * len(target_ids) + [0.0, 0.0]
    upper = [np.inf] * len(target_ids) + [float(counts["weekday"]), float(counts["weekend"])]
    for col, field_day in enumerate(field_days):
        for seg_id in set(field_day["segment_ids"]):
            if seg_id in target_index:
                matrix[target_index[seg_id], col] = 1.0
        matrix[len(target_ids), col] = 1.0 if field_day["day_type"] == "weekday" else 0.0
        matrix[len(target_ids) + 1, col] = 1.0 if field_day["day_type"] == "weekend" else 0.0
    costs = np.array(
        [
            float(day["p75_minutes"])
            + float(day["stress"]) * 0.01
            + float(day["grade_adjusted_miles"]) * 0.001
            + float(day["on_foot_miles"]) * 0.0001
            + float(day["loop_count"]) * 0.00001
            + float(day["parking_risk"]) * 0.000001
            for day in field_days
        ],
        dtype=float,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(field_days), dtype=int),
        bounds=Bounds(np.zeros(len(field_days)), np.ones(len(field_days))),
        constraints=LinearConstraint(matrix.tocsr(), lb=np.array(lower), ub=np.array(upper)),
        options={"time_limit": 120},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}
    selected = [day for day, value in zip(field_days, result.x) if value >= 0.5]
    covered = sorted(
        {
            seg_id
            for day in selected
            for seg_id in day["segment_ids"]
            if seg_id in target_index
        }
    )
    return {
        "success": True,
        "field_day_count": len(selected),
        "weekday_field_day_count": sum(1 for day in selected if day["day_type"] == "weekday"),
        "weekend_field_day_count": sum(1 for day in selected if day["day_type"] == "weekend"),
        "covered_segment_count": len(covered),
        "missing_segment_count": len(set(target_ids) - set(covered)),
        "missing_segment_ids": sorted(set(target_ids) - set(covered)),
        "total_p75_minutes": int(sum(day["p75_minutes"] for day in selected)),
        "max_p90_stress": round(max(day["stress"] for day in selected), 3) if selected else 0,
        "total_on_foot_miles_day_sum": round(sum(day["on_foot_miles"] for day in selected), 2),
        "selected_field_days": sorted(selected, key=lambda day: (day["day_type"], day["p75_minutes"], day["field_day_id"])),
    }


def solve_direct_field_day_max_coverage(
    field_days: list[dict[str, Any]],
    target_ids: list[int],
    counts: dict[str, int],
    official_miles_by_segment: dict[int, float] | None = None,
    required_segment_ids: list[int] | None = None,
) -> dict[str, Any]:
    if not field_days:
        return {"success": False, "reason": "no_field_day_candidates"}
    official_miles_by_segment = official_miles_by_segment or {}
    required_segment_ids = [int(seg_id) for seg_id in (required_segment_ids or [])]
    target_index = {seg_id: row for row, seg_id in enumerate(target_ids)}
    required_not_in_target = sorted(set(required_segment_ids) - set(target_index))
    if required_not_in_target:
        return {
            "success": False,
            "reason": "required_segments_not_in_target",
            "missing_required_segment_ids": required_not_in_target,
        }
    coverable = {
        seg_id
        for field_day in field_days
        for seg_id in field_day["segment_ids"]
        if seg_id in target_index
    }
    missing_required = sorted(set(required_segment_ids) - coverable)
    if missing_required:
        return {
            "success": False,
            "reason": "required_segments_not_coverable_by_field_day_candidates",
            "missing_required_segment_ids": missing_required,
        }
    day_count = len(field_days)
    segment_count = len(target_ids)
    matrix = lil_matrix((segment_count + 2, day_count + segment_count), dtype=float)
    lower = [-np.inf] * segment_count + [0.0, 0.0]
    upper = [0.0] * segment_count + [float(counts["weekday"]), float(counts["weekend"])]
    for seg_id, row in target_index.items():
        matrix[row, day_count + row] = 1.0
    for col, field_day in enumerate(field_days):
        for seg_id in set(field_day["segment_ids"]):
            if seg_id in target_index:
                matrix[target_index[seg_id], col] = -1.0
        matrix[segment_count, col] = 1.0 if field_day["day_type"] == "weekday" else 0.0
        matrix[segment_count + 1, col] = 1.0 if field_day["day_type"] == "weekend" else 0.0
    day_costs = [
        float(day["p75_minutes"]) * 0.001
        + float(day["stress"]) * 0.0001
        + float(day["grade_adjusted_miles"]) * 0.00001
        + float(day["parking_risk"]) * 0.000001
        for day in field_days
    ]
    segment_rewards = [
        -(100000.0 + float(official_miles_by_segment.get(seg_id, 0.0)) * 1000.0)
        for seg_id in target_ids
    ]
    costs = np.array(day_costs + segment_rewards, dtype=float)
    lower_bounds = np.zeros(day_count + segment_count)
    for seg_id in required_segment_ids:
        lower_bounds[day_count + target_index[seg_id]] = 1.0
    result = milp(
        c=costs,
        integrality=np.ones(day_count + segment_count, dtype=int),
        bounds=Bounds(lower_bounds, np.ones(day_count + segment_count)),
        constraints=LinearConstraint(matrix.tocsr(), lb=np.array(lower), ub=np.array(upper)),
        options={"time_limit": 120},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}
    selected_days = [day for day, value in zip(field_days, result.x[:day_count]) if value >= 0.5]
    covered_ids = [
        seg_id
        for seg_id, value in zip(target_ids, result.x[day_count:])
        if value >= 0.5
    ]
    missing_ids = sorted(set(target_ids) - set(covered_ids))
    return {
        "success": True,
        "field_day_count": len(selected_days),
        "weekday_field_day_count": sum(1 for day in selected_days if day["day_type"] == "weekday"),
        "weekend_field_day_count": sum(1 for day in selected_days if day["day_type"] == "weekend"),
        "covered_segment_count": len(covered_ids),
        "missing_segment_count": len(missing_ids),
        "missing_segment_ids": missing_ids,
        "covered_official_miles": round(sum(float(official_miles_by_segment.get(seg_id, 0.0)) for seg_id in covered_ids), 2),
        "missing_official_miles": round(sum(float(official_miles_by_segment.get(seg_id, 0.0)) for seg_id in missing_ids), 2),
        "total_p75_minutes": int(sum(day["p75_minutes"] for day in selected_days)),
        "max_p90_stress": round(max(day["stress"] for day in selected_days), 3) if selected_days else 0,
        "selected_field_days": sorted(selected_days, key=lambda day: (day["day_type"], day["p75_minutes"], day["field_day_id"])),
    }


def solve_direct_field_day_cover_with_cost(
    field_days: list[dict[str, Any]],
    target_ids: list[int],
    *,
    weekday_limit: int,
    weekend_limit: int,
    cost_mode: str,
) -> dict[str, Any]:
    if not field_days:
        return {"success": False, "reason": "no_field_day_candidates"}
    target_index = {seg_id: row for row, seg_id in enumerate(target_ids)}
    coverable = {
        seg_id
        for field_day in field_days
        for seg_id in field_day["segment_ids"]
        if seg_id in target_index
    }
    if set(target_ids) - coverable:
        return {
            "success": False,
            "reason": "not_all_segments_coverable_by_field_day_candidates",
            "missing_segment_ids": sorted(set(target_ids) - coverable),
        }
    matrix = lil_matrix((len(target_ids) + 2, len(field_days)), dtype=float)
    lower = [1.0] * len(target_ids) + [0.0, 0.0]
    upper = [np.inf] * len(target_ids) + [float(weekday_limit), float(weekend_limit)]
    for col, field_day in enumerate(field_days):
        for seg_id in set(field_day["segment_ids"]):
            if seg_id in target_index:
                matrix[target_index[seg_id], col] = 1.0
        matrix[len(target_ids), col] = 1.0 if field_day["day_type"] == "weekday" else 0.0
        matrix[len(target_ids) + 1, col] = 1.0 if field_day["day_type"] == "weekend" else 0.0
    if cost_mode == "min_weekdays":
        costs = np.array(
            [
                (1.0 if day["day_type"] == "weekday" else 0.0)
                + 0.001
                + float(day["p75_minutes"]) * 0.000001
                for day in field_days
            ],
            dtype=float,
        )
    else:
        costs = np.array(
            [
                1.0
                + float(day["p75_minutes"]) * 0.000001
                + float(day["stress"]) * 0.0000001
                for day in field_days
            ],
            dtype=float,
        )
    result = milp(
        c=costs,
        integrality=np.ones(len(field_days), dtype=int),
        bounds=Bounds(np.zeros(len(field_days)), np.ones(len(field_days))),
        constraints=LinearConstraint(matrix.tocsr(), lb=np.array(lower), ub=np.array(upper)),
        options={"time_limit": 60},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}
    selected = [day for day, value in zip(field_days, result.x) if value >= 0.5]
    return {
        "success": True,
        "field_day_count": len(selected),
        "weekday_field_day_count": sum(1 for day in selected if day["day_type"] == "weekday"),
        "weekend_field_day_count": sum(1 for day in selected if day["day_type"] == "weekend"),
        "total_p75_minutes": int(sum(day["p75_minutes"] for day in selected)),
    }


def relaxed_pressure_diagnostics(
    field_days: list[dict[str, Any]],
    target_ids: list[int],
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "min_total_days_unlimited_day_counts": solve_direct_field_day_cover_with_cost(
            field_days,
            target_ids,
            weekday_limit=100,
            weekend_limit=100,
            cost_mode="min_days",
        ),
        "min_weekdays_with_actual_weekend_count": solve_direct_field_day_cover_with_cost(
            field_days,
            target_ids,
            weekday_limit=100,
            weekend_limit=int(counts["weekend"]),
            cost_mode="min_weekdays",
        ),
    }


def build_scenario(
    *,
    name: str,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
    connected_expansion_limit: int,
    current_rule_compliant: bool,
) -> dict[str, Any]:
    state = read_json(repaired.DEFAULT_STATE_JSON)
    target_ids, candidates, field_menu = all_repaired_candidates()
    segment_index = official_segments(read_json(DEFAULT_OFFICIAL_GEOJSON))
    official_miles_by_segment = {
        seg_id: float(segment_index[seg_id]["official_miles"])
        for seg_id in target_ids
        if seg_id in segment_index
    }
    max_bound = max(weekday_bound, weekend_bound)
    bounded = [
        candidate
        for candidate in candidates
        if int(candidate["door_to_door_p90_minutes"]) <= max_bound
        and candidate.get("validation_passed") is True
        and candidate.get("manual_design_hold") is not True
    ]
    loops, missing_coordinates = candidate_loop_records(bounded, field_menu, state)
    availability = state.get("availability_model") or {}
    counts = pack_audit.date_counts(
        (availability.get("available_dates") or {})["start"],
        (availability.get("available_dates") or {})["end"],
    )
    field_days = generate_direct_field_day_candidates(
        loops,
        state,
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
        connected_expansion_limit=connected_expansion_limit,
    )
    solution = solve_direct_field_day_cover(field_days, target_ids, counts)
    max_coverage = solve_direct_field_day_max_coverage(
        field_days,
        target_ids,
        counts,
        official_miles_by_segment=official_miles_by_segment,
    )
    pressure = relaxed_pressure_diagnostics(field_days, target_ids, counts)
    return {
        "scenario": name,
        "current_rule_compliant": current_rule_compliant,
        "weekday_bound_minutes": weekday_bound,
        "weekend_bound_minutes": weekend_bound,
        "candidate_loop_count": len(loops),
        "missing_coordinate_count": len(missing_coordinates),
        "missing_coordinates": missing_coordinates[:20],
        "field_day_candidate_count": len(field_days),
        "date_counts": counts,
        "max_coverage_solution": max_coverage,
        "pressure_diagnostics": pressure,
        "solution": solution,
        "caveats": [
            "This is a direct field-day set-cover optimization over generated day candidates.",
            "Generated combos include nearby cliques plus capped connected nearby-drive chains; a missed combo is possible.",
            "Drive times use the private straight-line drive model, not OSRM.",
            "Date-specific assignment, including Lower Hulls even-day placement, is not performed here.",
        ],
    }


def build_report(max_combo_size: int, neighbor_limit: int, connected_expansion_limit: int) -> dict[str, Any]:
    state = read_json(repaired.DEFAULT_STATE_JSON)
    availability = state.get("availability_model") or {}
    weekday = int(availability.get("weekday_max_minutes") or 0)
    weekend = int(availability.get("weekend_max_minutes") or 0)
    return {
        "objective": "directly optimize home-to-home field days over the repaired p90 candidate universe",
        "source_files": {
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "state_json": display_path(repaired.DEFAULT_STATE_JSON),
            "field_menu_json": display_path(repaired.DEFAULT_FIELD_MENU_JSON),
            "split_probe_json": display_path(repaired.DEFAULT_SPLIT_PROBE_JSON),
            "forced_probe_json": display_path(repaired.DEFAULT_FORCED_PROBE_JSON),
        },
        "config": {
            "max_combo_size": max_combo_size,
            "neighbor_limit": neighbor_limit,
            "connected_expansion_limit": connected_expansion_limit,
        },
        "scenarios": [
            build_scenario(
                name="strict_current_p90_bounds",
                weekday_bound=weekday,
                weekend_bound=weekend,
                max_combo_size=max_combo_size,
                neighbor_limit=neighbor_limit,
                connected_expansion_limit=connected_expansion_limit,
                current_rule_compliant=True,
            ),
            build_scenario(
                name="shingle_weekday_292_bound",
                weekday_bound=max(weekday, 292),
                weekend_bound=weekend,
                max_combo_size=max_combo_size,
                neighbor_limit=neighbor_limit,
                connected_expansion_limit=connected_expansion_limit,
                current_rule_compliant=False,
            ),
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P90 Joint Field-Day Optimizer",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        "| Scenario | Current-rule compliant | Candidate loops | Field-day candidates | Solution | Max coverage | Field days | Weekday / weekend | Total p75 | Missing |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for scenario in report["scenarios"]:
        solution = scenario["solution"]
        max_coverage = scenario["max_coverage_solution"]
        lines.append(
            f"| {scenario['scenario']} | {scenario['current_rule_compliant']} | "
            f"{scenario['candidate_loop_count']} | {scenario['field_day_candidate_count']} | "
            f"{solution.get('success')} {solution.get('reason') or solution.get('message') or ''} | "
            f"{max_coverage.get('covered_segment_count') or ''}/{max_coverage.get('covered_segment_count') + max_coverage.get('missing_segment_count') if max_coverage.get('success') else ''} | "
            f"{solution.get('field_day_count') or ''} | "
            f"{solution.get('weekday_field_day_count') or ''} / {solution.get('weekend_field_day_count') or ''} | "
            f"{solution.get('total_p75_minutes') or ''} | "
            f"{', '.join(str(seg_id) for seg_id in solution.get('missing_segment_ids') or [])} |"
        )
    lines.extend(["", "## Scenario Details", ""])
    for scenario in report["scenarios"]:
        solution = scenario["solution"]
        max_coverage = scenario["max_coverage_solution"]
        lines.extend(
            [
                f"### {scenario['scenario']}",
                "",
                f"- Weekday/weekend bounds: {scenario['weekday_bound_minutes']} / {scenario['weekend_bound_minutes']} min",
                f"- Candidate loops with coordinates: {scenario['candidate_loop_count']}",
                f"- Field-day candidates generated: {scenario['field_day_candidate_count']}",
                f"- Solution success: {solution.get('success')}",
            ]
        )
        if max_coverage.get("success"):
            lines.extend(
                [
                    f"- Max-coverage schedule: {max_coverage['covered_segment_count']} covered / {max_coverage['missing_segment_count']} missing",
                    f"- Max-coverage official miles: {max_coverage['covered_official_miles']} covered / {max_coverage['missing_official_miles']} missing",
                    f"- Max-coverage field days: {max_coverage['field_day_count']} ({max_coverage['weekday_field_day_count']} weekday / {max_coverage['weekend_field_day_count']} weekend)",
                    f"- Max-coverage total p75 minutes: {max_coverage['total_p75_minutes']}",
                    f"- Max-coverage missing segment ids: {', '.join(str(seg_id) for seg_id in max_coverage['missing_segment_ids'])}",
                ]
            )
        min_days = scenario["pressure_diagnostics"]["min_total_days_unlimited_day_counts"]
        min_weekdays = scenario["pressure_diagnostics"]["min_weekdays_with_actual_weekend_count"]
        if min_days.get("success"):
            lines.append(
                f"- Relaxed minimum field days: {min_days['field_day_count']} "
                f"({min_days['weekday_field_day_count']} weekday / {min_days['weekend_field_day_count']} weekend)"
            )
        if min_weekdays.get("success"):
            lines.append(
                f"- Minimum weekdays with actual weekend count: {min_weekdays['weekday_field_day_count']} "
                f"weekday / {min_weekdays['weekend_field_day_count']} weekend"
            )
        if solution.get("success"):
            lines.extend(
                [
                    f"- Field days: {solution['field_day_count']}",
                    f"- Weekday/weekend field days: {solution['weekday_field_day_count']} / {solution['weekend_field_day_count']}",
                    f"- Total p75 minutes: {solution['total_p75_minutes']}",
                    f"- Max p90 stress: {solution['max_p90_stress']}",
                    f"- Total on-foot day-sum miles: {solution['total_on_foot_miles_day_sum']}",
                ]
            )
        else:
            lines.append(f"- Reason/message: {solution.get('reason') or solution.get('message')}")
        lines.append("")
    lines.extend(
        [
            "## Caveats",
            "",
            "- This audit may prove a better route-selection shape exists, but the Shingle 292-minute scenario is not compliant with the current p90 rule.",
            "- The result is still a field-day set, not an assigned calendar with Lower Hulls date placement.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--neighbor-limit", type=int, default=10)
    parser.add_argument("--connected-expansion-limit", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        max_combo_size=args.max_combo_size,
        neighbor_limit=args.neighbor_limit,
        connected_expansion_limit=args.connected_expansion_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for scenario in report["scenarios"]:
        solution = scenario["solution"]
        print(
            json.dumps(
                {
                    "scenario": scenario["scenario"],
                    "candidate_loop_count": scenario["candidate_loop_count"],
                    "field_day_candidate_count": scenario["field_day_candidate_count"],
                    "success": solution.get("success"),
                    "reason": solution.get("reason"),
                    "field_day_count": solution.get("field_day_count"),
                    "total_p75_minutes": solution.get("total_p75_minutes"),
                    "missing_segment_ids": solution.get("missing_segment_ids"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
