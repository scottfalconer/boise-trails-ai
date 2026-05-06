#!/usr/bin/env python3
"""Pack the repaired p90 candidate universe into home-to-home field days."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import date, timedelta
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
from p90_completion_gap_analyzer import (  # noqa: E402
    load_candidates,
    official_segments,
    usable_candidates,
)
from p90_forced_anchor_probe import (  # noqa: E402
    DEFAULT_MANUAL_DESIGN_JSONS,
    load_all_anchors,
    parking_risk_score,
)
from personal_route_planner import (  # noqa: E402
    DEFAULT_OFFICIAL_GEOJSON,
    DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
    DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
    haversine_miles,
    read_json,
)


DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-repaired-field-day-pack-audit-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def date_counts(start: str, end: str) -> dict[str, int]:
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    counts = {"weekday": 0, "weekend": 0, "total": 0}
    while current <= last:
        counts["weekend" if current.weekday() >= 5 else "weekday"] += 1
        counts["total"] += 1
        current += timedelta(days=1)
    return counts


def drive_minutes_between(
    left: tuple[float, float],
    right: tuple[float, float],
    drive_model: dict[str, Any],
    *,
    apply_minimum: bool = False,
) -> int:
    if left == right:
        return 0
    straight_line = haversine_miles(left, right)
    minutes = straight_line * float(drive_model.get("straight_line_factor") or 1.25) * float(
        drive_model.get("minutes_per_mile") or 2.2
    )
    if apply_minimum and straight_line > 0.05:
        minutes = max(minutes, float(drive_model.get("minimum_one_way_minutes") or 5))
    return int(math.ceil(minutes))


def candidate_key(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("source") or ""),
        str(candidate.get("candidate_id") or ""),
        str(candidate.get("trailhead") or ""),
    )


def exact_set_cover_candidates(
    candidates: list[dict[str, Any]],
    target_ids: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not candidates:
        return {"success": False, "reason": "no_candidates"}, []
    target_index = {seg_id: row for row, seg_id in enumerate(target_ids)}
    matrix = lil_matrix((len(target_ids), len(candidates)), dtype=float)
    for col, candidate in enumerate(candidates):
        for seg_id in set(candidate["segment_ids"]):
            if seg_id in target_index:
                matrix[target_index[seg_id], col] = 1.0
    costs = np.array(
        [
            float(candidate["door_to_door_p75_minutes"])
            + float(candidate["door_to_door_p90_minutes"]) * 0.001
            + float(candidate.get("grade_adjusted_miles") or 0.0) * 0.0001
            + float(candidate["on_foot_miles"]) * 0.00001
            + float(candidate.get("parking_risk") or 1) * 0.000001
            for candidate in candidates
        ],
        dtype=float,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(
            matrix.tocsr(),
            lb=np.ones(len(target_ids)),
            ub=np.full(len(target_ids), np.inf),
        ),
        options={"time_limit": 60},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}, []
    selected = [candidate for candidate, value in zip(candidates, result.x) if value >= 0.5]
    covered = sorted(
        {
            seg_id
            for candidate in selected
            for seg_id in candidate["segment_ids"]
            if seg_id in target_index
        }
    )
    return (
        {
            "success": True,
            "selected_candidate_count": len(selected),
            "covered_segment_count": len(covered),
            "missing_segment_count": len(set(target_ids) - set(covered)),
            "missing_segment_ids": sorted(set(target_ids) - set(covered)),
            "total_p75_minutes": int(sum(candidate["door_to_door_p75_minutes"] for candidate in selected)),
            "total_on_foot_miles": round(sum(candidate["on_foot_miles"] for candidate in selected), 2),
        },
        selected,
    )


def coordinate_lookup(field_menu: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    anchors = load_all_anchors(
        public_trailheads_geojson=DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON,
        private_parking_anchors_geojson=DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON,
        manual_design_jsons=DEFAULT_MANUAL_DESIGN_JSONS,
    )
    for anchor in anchors:
        lookup[str(anchor["name"])] = {
            "lon": float(anchor["lon"]),
            "lat": float(anchor["lat"]),
            "parking_risk": parking_risk_score(anchor),
            "parking_confidence": anchor.get("parking_confidence"),
        }
    for cue in (field_menu.get("route_cues") or {}).values():
        trailhead = cue.get("trailhead") or {}
        if trailhead.get("name") and trailhead.get("lon") is not None and trailhead.get("lat") is not None:
            lookup[str(trailhead["name"])] = {
                "lon": float(trailhead["lon"]),
                "lat": float(trailhead["lat"]),
                "parking_risk": 1,
                "parking_confidence": trailhead.get("parking_confidence"),
            }
    return lookup


def load_repaired_candidates(
    *,
    include_shingle_exception: bool,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any], int]:
    official_geojson = read_json(DEFAULT_OFFICIAL_GEOJSON)
    segment_index = official_segments(official_geojson)
    target_ids = sorted(segment_index)
    state = read_json(repaired.DEFAULT_STATE_JSON)
    availability = state.get("availability_model") or {}
    max_bound = max(
        int(availability.get("weekday_max_minutes") or 0),
        int(availability.get("weekend_max_minutes") or 0),
    )
    personal_route_menu = read_json(repaired.DEFAULT_PERSONAL_ROUTE_MENU_JSON)
    hybrid_route_pass = read_json(repaired.DEFAULT_HYBRID_ROUTE_PASS_JSON)
    field_menu = read_json(repaired.DEFAULT_FIELD_MENU_JSON)
    existing = usable_candidates(load_candidates(personal_route_menu, hybrid_route_pass, field_menu, segment_index))
    probes = repaired.strict_probe_candidates(
        read_json(repaired.DEFAULT_SPLIT_PROBE_JSON),
        read_json(repaired.DEFAULT_FORCED_PROBE_JSON),
    )
    all_candidates = existing + probes
    bounded = [
        candidate
        for candidate in all_candidates
        if int(candidate["door_to_door_p90_minutes"]) <= max_bound
        and candidate.get("validation_passed") is True
        and candidate.get("manual_design_hold") is not True
    ]
    if include_shingle_exception:
        exception = repaired.best_over_bound_candidate_for_segment(all_candidates, 1656, max_bound)
        if exception:
            bounded.append(exception)
    return target_ids, bounded, field_menu, max_bound


def loop_records(
    selected: list[dict[str, Any]],
    field_menu: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = coordinate_lookup(field_menu)
    drive_model = state.get("drive_model") or {}
    home = (float(drive_model["origin_lon"]), float(drive_model["origin_lat"]))
    loops: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for candidate in selected:
        trailhead = str(candidate.get("trailhead") or "")
        location = lookup.get(trailhead)
        if not location:
            missing.append(
                {
                    "source": candidate.get("source"),
                    "candidate_id": candidate.get("candidate_id"),
                    "trailhead": trailhead,
                }
            )
            continue
        parking = (float(location["lon"]), float(location["lat"]))
        home_drive = drive_minutes_between(home, parking, drive_model, apply_minimum=True)
        p75 = int(candidate["door_to_door_p75_minutes"])
        p90 = int(candidate["door_to_door_p90_minutes"])
        loops.append(
            {
                "loop_id": "::".join(candidate_key(candidate)),
                "source": candidate.get("source"),
                "candidate_id": candidate.get("candidate_id"),
                "label": candidate.get("label"),
                "trailhead": trailhead,
                "parking": {"lon": parking[0], "lat": parking[1]},
                "segment_ids": candidate["segment_ids"],
                "official_miles": candidate["official_miles"],
                "on_foot_miles": candidate["on_foot_miles"],
                "grade_adjusted_miles": candidate.get("grade_adjusted_miles") or 0.0,
                "ascent_ft": candidate.get("ascent_ft") or 0,
                "door_to_door_p75_minutes": p75,
                "door_to_door_p90_minutes": p90,
                "home_drive_minutes": home_drive,
                "internal_p75_minutes": max(0, p75 - 2 * home_drive),
                "p90_delta_minutes": max(0, p90 - p75),
                "parking_risk": candidate.get("parking_risk") or location.get("parking_risk") or 1,
                "parking_confidence": candidate.get("parking_confidence") or location.get("parking_confidence"),
            }
        )
    return loops, missing


def field_day_time(
    loops: list[dict[str, Any]],
    order: tuple[int, ...],
    state: dict[str, Any],
) -> dict[str, Any]:
    drive_model = state.get("drive_model") or {}
    home = (float(drive_model["origin_lon"]), float(drive_model["origin_lat"]))
    ordered = [loops[index] for index in order]
    first = ordered[0]["parking"]
    drive = drive_minutes_between(home, (float(first["lon"]), float(first["lat"])), drive_model, apply_minimum=True)
    between = 0
    for left, right in zip(ordered, ordered[1:]):
        left_point = (float(left["parking"]["lon"]), float(left["parking"]["lat"]))
        right_point = (float(right["parking"]["lon"]), float(right["parking"]["lat"]))
        minutes = drive_minutes_between(left_point, right_point, drive_model, apply_minimum=False)
        between += minutes
        drive += minutes
    last = ordered[-1]["parking"]
    drive += drive_minutes_between(home, (float(last["lon"]), float(last["lat"])), drive_model, apply_minimum=True)
    internal_p75 = sum(int(loop["internal_p75_minutes"]) for loop in ordered)
    p75 = drive + internal_p75
    p90 = p75 + sum(int(loop["p90_delta_minutes"]) for loop in ordered)
    return {
        "order": [loop["loop_id"] for loop in ordered],
        "p75_minutes": int(p75),
        "p90_minutes": int(p90),
        "drive_minutes": int(drive),
        "between_drive_minutes": int(between),
        "on_foot_miles": round(sum(float(loop["on_foot_miles"]) for loop in ordered), 2),
        "grade_adjusted_miles": round(sum(float(loop.get("grade_adjusted_miles") or 0.0) for loop in ordered), 2),
        "parking_risk": int(sum(int(loop.get("parking_risk") or 1) for loop in ordered)),
    }


def best_field_day_for_combo(
    loops: list[dict[str, Any]],
    combo: tuple[int, ...],
    state: dict[str, Any],
) -> dict[str, Any]:
    best = None
    for order in itertools.permutations(combo):
        day = field_day_time(loops, order, state)
        if best is None or (
            day["p75_minutes"],
            day["p90_minutes"],
            day["between_drive_minutes"],
        ) < (
            best["p75_minutes"],
            best["p90_minutes"],
            best["between_drive_minutes"],
        ):
            best = day
    assert best is not None
    return best


def generate_field_day_candidates(
    loops: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drive_model = state.get("drive_model") or {}
    acceptable_drive = int((state.get("availability_model") or {}).get("acceptable_inter_trailhead_drive_minutes") or 20)
    bounds = {"weekday": weekday_bound, "weekend": weekend_bound}
    max_bound = max(bounds.values())
    blockers = [
        {
            "loop_id": loop["loop_id"],
            "trailhead": loop["trailhead"],
            "p90_minutes": loop["door_to_door_p90_minutes"],
            "max_bound_minutes": max_bound,
        }
        for loop in loops
        if int(loop["door_to_door_p90_minutes"]) > max_bound
    ]
    candidates = []
    nearby: dict[tuple[int, int], bool] = {}
    for left, right in itertools.combinations(range(len(loops)), 2):
        left_point = (float(loops[left]["parking"]["lon"]), float(loops[left]["parking"]["lat"]))
        right_point = (float(loops[right]["parking"]["lon"]), float(loops[right]["parking"]["lat"]))
        nearby[(left, right)] = (
            drive_minutes_between(left_point, right_point, drive_model, apply_minimum=False) <= acceptable_drive
        )
    for size in range(1, max_combo_size + 1):
        for combo in itertools.combinations(range(len(loops)), size):
            if size > 1 and not all(nearby.get(tuple(sorted(pair)), False) for pair in itertools.combinations(combo, 2)):
                continue
            min_internal = sum(int(loops[index]["internal_p75_minutes"]) for index in combo)
            min_delta = sum(int(loops[index]["p90_delta_minutes"]) for index in combo)
            if min_internal + min_delta > max_bound:
                continue
            day = best_field_day_for_combo(loops, combo, state)
            for day_type, bound in bounds.items():
                if day["p90_minutes"] > bound:
                    continue
                candidates.append(
                    {
                        "field_day_id": f"{day_type}-" + "--".join(day["order"]),
                        "day_type": day_type,
                        "loop_ids": day["order"],
                        "loop_count": len(day["order"]),
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
    return candidates, blockers


def solve_field_day_partition(
    loops: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, Any]:
    if not candidates:
        return {"success": False, "reason": "no_field_day_candidates"}
    loop_to_row = {loop["loop_id"]: row for row, loop in enumerate(loops)}
    matrix = lil_matrix((len(loops) + 2, len(candidates)), dtype=float)
    lower = [1.0] * len(loops) + [0.0, 0.0]
    upper = [1.0] * len(loops) + [float(counts["weekday"]), float(counts["weekend"])]
    for col, candidate in enumerate(candidates):
        for loop_id in candidate["loop_ids"]:
            matrix[loop_to_row[loop_id], col] = 1.0
        matrix[len(loops), col] = 1.0 if candidate["day_type"] == "weekday" else 0.0
        matrix[len(loops) + 1, col] = 1.0 if candidate["day_type"] == "weekend" else 0.0
    costs = np.array(
        [
            float(candidate["p75_minutes"])
            + float(candidate["stress"]) * 0.01
            + float(candidate["grade_adjusted_miles"]) * 0.001
            + float(candidate["on_foot_miles"]) * 0.0001
            + float(candidate["loop_count"]) * 0.00001
            + float(candidate["parking_risk"]) * 0.000001
            for candidate in candidates
        ],
        dtype=float,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(matrix.tocsr(), lb=np.array(lower), ub=np.array(upper)),
        options={"time_limit": 60},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}
    selected = [candidate for candidate, value in zip(candidates, result.x) if value >= 0.5]
    return {
        "success": True,
        "field_day_count": len(selected),
        "weekday_field_day_count": sum(1 for day in selected if day["day_type"] == "weekday"),
        "weekend_field_day_count": sum(1 for day in selected if day["day_type"] == "weekend"),
        "total_p75_minutes": int(sum(day["p75_minutes"] for day in selected)),
        "max_p90_stress": round(max(day["stress"] for day in selected), 3) if selected else 0,
        "total_on_foot_miles_day_sum": round(sum(day["on_foot_miles"] for day in selected), 2),
        "field_days": sorted(selected, key=lambda day: (day["day_type"], day["p75_minutes"], day["field_day_id"])),
    }


def loop_pressure_diagnostics(
    loops: list[dict[str, Any]],
    field_day_candidates: list[dict[str, Any]],
    *,
    weekend_bound: int,
    weekday_count: int,
) -> dict[str, Any]:
    weekday_only_ids = {
        loop["loop_id"]
        for loop in loops
        if int(loop["door_to_door_p90_minutes"]) > weekend_bound
    }
    weekday_candidates = [
        candidate for candidate in field_day_candidates if candidate["day_type"] == "weekday"
    ]
    weekday_only_counts = [
        sum(1 for loop_id in candidate["loop_ids"] if loop_id in weekday_only_ids)
        for candidate in weekday_candidates
    ]
    max_weekday_only_per_field_day = max(weekday_only_counts, default=0)
    min_weekday_days_for_weekday_only = (
        math.ceil(len(weekday_only_ids) / max_weekday_only_per_field_day)
        if max_weekday_only_per_field_day
        else 0
    )
    return {
        "weekday_only_loop_count": len(weekday_only_ids),
        "weekday_count": weekday_count,
        "max_weekday_only_loops_per_field_day_candidate": max_weekday_only_per_field_day,
        "min_weekday_days_for_weekday_only_loops": min_weekday_days_for_weekday_only,
        "weekday_only_capacity_exceeded": min_weekday_days_for_weekday_only > weekday_count,
    }


def build_scenario(
    *,
    name: str,
    include_shingle_exception: bool,
    shingle_weekday_bound_override: int | None,
    max_combo_size: int,
) -> dict[str, Any]:
    state = read_json(repaired.DEFAULT_STATE_JSON)
    target_ids, candidates, field_menu, strict_bound = load_repaired_candidates(
        include_shingle_exception=include_shingle_exception
    )
    set_cover, selected = exact_set_cover_candidates(candidates, target_ids)
    availability = state.get("availability_model") or {}
    weekday_bound = int(availability.get("weekday_max_minutes") or 0)
    weekend_bound = int(availability.get("weekend_max_minutes") or 0)
    if shingle_weekday_bound_override:
        weekday_bound = max(weekday_bound, shingle_weekday_bound_override)
    loops, missing_coordinates = loop_records(selected, field_menu, state)
    counts = date_counts(
        (availability.get("available_dates") or {})["start"],
        (availability.get("available_dates") or {})["end"],
    )
    if not set_cover.get("success"):
        field_day_candidates: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        partition = {"success": False, "reason": "set_cover_failed_before_field_day_packing"}
    elif missing_coordinates:
        field_day_candidates = []
        blockers = []
        partition = {"success": False, "reason": "selected_loop_missing_parking_coordinates"}
    else:
        field_day_candidates, blockers = generate_field_day_candidates(
            loops,
            state,
            weekday_bound=weekday_bound,
            weekend_bound=weekend_bound,
            max_combo_size=max_combo_size,
        )
        if blockers:
            partition = {"success": False, "reason": "selected_loop_exceeds_scenario_p90_bound"}
        else:
            partition = solve_field_day_partition(loops, field_day_candidates, counts)
    pressure = loop_pressure_diagnostics(
        loops,
        field_day_candidates,
        weekend_bound=weekend_bound,
        weekday_count=counts["weekday"],
    )
    return {
        "scenario": name,
        "include_shingle_exception": include_shingle_exception,
        "compliant_with_current_p90_rule": not include_shingle_exception and shingle_weekday_bound_override is None,
        "strict_bound_minutes": strict_bound,
        "weekday_bound_minutes": weekday_bound,
        "weekend_bound_minutes": weekend_bound,
        "date_counts": counts,
        "set_cover": set_cover,
        "selected_loop_count": len(selected),
        "selected_loop_missing_coordinate_count": len(missing_coordinates),
        "selected_loop_missing_coordinates": missing_coordinates[:20],
        "field_day_candidate_count": len(field_day_candidates),
        "oversized_selected_loop_count": len(blockers),
        "oversized_selected_loops": blockers[:20],
        "loop_pressure": pressure,
        "field_day_partition": partition,
        "caveats": [
            "This packs the p75-minimizing set-cover loop selection, not a joint route-selection-and-day-packing optimization.",
            "Drive times use the private straight-line drive model, not OSRM.",
            "The Shingle override scenario is not compliant with the current p90 rule; it is only a what-if.",
        ],
    }


def build_report(max_combo_size: int) -> dict[str, Any]:
    return {
        "objective": "pack repaired p90 candidate-universe loops into home-to-home field days",
        "source_files": {
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "state_json": display_path(repaired.DEFAULT_STATE_JSON),
            "field_menu_json": display_path(repaired.DEFAULT_FIELD_MENU_JSON),
            "split_probe_json": display_path(repaired.DEFAULT_SPLIT_PROBE_JSON),
            "forced_probe_json": display_path(repaired.DEFAULT_FORCED_PROBE_JSON),
            "public_trailheads_geojson": display_path(DEFAULT_TRAILHEAD_CANDIDATES_GEOJSON),
            "private_parking_anchors_geojson": display_path(DEFAULT_PRIVATE_PARKING_ANCHORS_GEOJSON),
            "manual_design_jsons": [display_path(path) for path in DEFAULT_MANUAL_DESIGN_JSONS],
        },
        "config": {"max_combo_size": max_combo_size},
        "scenarios": [
            build_scenario(
                name="strict_current_p90_bounds",
                include_shingle_exception=False,
                shingle_weekday_bound_override=None,
                max_combo_size=max_combo_size,
            ),
            build_scenario(
                name="shingle_exception_current_bounds",
                include_shingle_exception=True,
                shingle_weekday_bound_override=None,
                max_combo_size=max_combo_size,
            ),
            build_scenario(
                name="shingle_exception_weekday_292_bound",
                include_shingle_exception=True,
                shingle_weekday_bound_override=292,
                max_combo_size=max_combo_size,
            ),
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P90 Repaired Field-Day Pack Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        "| Scenario | Current-rule compliant | Set cover | Selected loops | Day candidates | Oversized loops | Field-day partition | Field days | Total p75 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        partition = scenario["field_day_partition"]
        lines.append(
            f"| {scenario['scenario']} | {scenario['compliant_with_current_p90_rule']} | "
            f"{scenario['set_cover'].get('success')} | {scenario['selected_loop_count']} | "
                f"{scenario['field_day_candidate_count']} | {scenario['oversized_selected_loop_count']} | "
            f"{partition.get('success')} {partition.get('reason') or partition.get('message') or ''} | "
            f"{partition.get('field_day_count') or ''} | {partition.get('total_p75_minutes') or ''} |"
        )
    lines.extend(["", "## Scenario Details", ""])
    for scenario in report["scenarios"]:
        partition = scenario["field_day_partition"]
        lines.extend(
            [
                f"### {scenario['scenario']}",
                "",
                f"- Weekday/weekend bounds: {scenario['weekday_bound_minutes']} / {scenario['weekend_bound_minutes']} min",
                f"- Set cover success: {scenario['set_cover'].get('success')}",
                f"- Selected loop count: {scenario['selected_loop_count']}",
                f"- Field-day candidates: {scenario['field_day_candidate_count']}",
                f"- Oversized selected loops: {scenario['oversized_selected_loop_count']}",
                f"- Weekday-only loops / weekdays: {scenario['loop_pressure']['weekday_only_loop_count']} / {scenario['loop_pressure']['weekday_count']}",
                f"- Min weekday days for weekday-only loops: {scenario['loop_pressure']['min_weekday_days_for_weekday_only_loops']}",
                f"- Partition success: {partition.get('success')}",
            ]
        )
        if partition.get("success"):
            lines.extend(
                [
                    f"- Field days: {partition['field_day_count']}",
                    f"- Weekday/weekend field days: {partition['weekday_field_day_count']} / {partition['weekend_field_day_count']}",
                    f"- Total p75 minutes: {partition['total_p75_minutes']}",
                    f"- Max p90 stress: {partition['max_p90_stress']}",
                ]
            )
        else:
            lines.append(f"- Reason/message: {partition.get('reason') or partition.get('message')}")
        if scenario["oversized_selected_loops"]:
            lines.extend(["", "Oversized selected loops:"])
            for blocker in scenario["oversized_selected_loops"][:10]:
                lines.append(
                    f"- `{blocker['loop_id']}`: {blocker['p90_minutes']} min p90 > {blocker['max_bound_minutes']} min"
                )
        lines.append("")
    lines.extend(
        [
            "## Caveats",
            "",
            "- This is a bridge audit between candidate coverage and the final calendar scheduler.",
            "- It does not replace the canonical outing menu or phone field packet.",
            "- The only current-rule-compliant scenario remains infeasible because Shingle `1656` is not covered under the 260-minute p90 bound.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-combo-size", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(max_combo_size=args.max_combo_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for scenario in report["scenarios"]:
        print(
            json.dumps(
                {
                    "scenario": scenario["scenario"],
                    "set_cover_success": scenario["set_cover"].get("success"),
                    "selected_loop_count": scenario["selected_loop_count"],
                    "field_day_candidate_count": scenario["field_day_candidate_count"],
                    "oversized_selected_loop_count": scenario["oversized_selected_loop_count"],
                    "partition_success": scenario["field_day_partition"].get("success"),
                    "partition_reason": scenario["field_day_partition"].get("reason"),
                    "field_day_count": scenario["field_day_partition"].get("field_day_count"),
                    "total_p75_minutes": scenario["field_day_partition"].get("total_p75_minutes"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
