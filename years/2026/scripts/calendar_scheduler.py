#!/usr/bin/env python3
"""Schedule executable outings across the 2026 challenge window."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from artifact_utils import build_artifact_manifest, utc_now_id, write_manifest  # noqa: E402
from day_of_preflight import evaluate_route_preflight  # noqa: E402


DEFAULT_EXECUTION_JSON = (
    YEAR_DIR
    / "experiments"
    / "2026-05-04-outing-execution-simulation"
    / "outing_execution.json"
)
DEFAULT_OUTPUT_DIR = YEAR_DIR / "outputs" / "calendar"

DEFAULT_CONSTRAINTS = {
    "start_time": "07:00",
    "weekday_max_minutes": 120,
    "weekday_normal_max_minutes": None,
    "weekend_max_minutes": 330,
    "long_outing_minutes": 240,
    "rest_days_after_long": 1,
    "max_consecutive_scheduled_days": None,
    "max_consecutive_scheduled_days_target": None,
    "min_open_days": 0,
    "objective_profile": "coverage_then_time",
    "latest_scheduled_date": None,
    "required_completion_segments": None,
    "required_rest_days_before_latest": None,
    "max_weekly_on_foot_miles": None,
    "long_on_foot_day_miles": None,
    "max_long_on_foot_days_per_week": None,
    "final_mop_up_start_date": None,
    "final_mop_up_max_minutes": None,
    "final_mop_up_max_on_foot_miles": None,
    "final_mop_up_max_ascent_ft": None,
    "optimizer_time_limit_seconds": 30,
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def day_budget(day: date, constraints: dict[str, Any]) -> int:
    if day.weekday() >= 5:
        return int(constraints["weekend_max_minutes"])
    return int(constraints["weekday_max_minutes"])


def normal_day_budget(day: date, constraints: dict[str, Any]) -> int | None:
    if day.weekday() >= 5:
        return None
    normal = constraints.get("weekday_normal_max_minutes")
    return int(normal) if normal is not None else None


def official_miles(outing: dict[str, Any]) -> float:
    return float(outing["legs"][3].get("official_new_miles") or 0)


def total_foot_miles(outing: dict[str, Any]) -> float:
    return float(outing["legs"][3].get("estimated_total_on_foot_miles") or 0)


def outing_ascent_ft(outing: dict[str, Any]) -> int:
    return int(round(float(outing["legs"][3].get("ascent_ft") or 0)))


def outing_descent_ft(outing: dict[str, Any]) -> int:
    return int(round(float(outing["legs"][3].get("descent_ft") or 0)))


def outing_grade_adjusted_miles(outing: dict[str, Any]) -> float:
    return float(outing["legs"][3].get("grade_adjusted_miles") or 0)


def outing_minutes(outing: dict[str, Any]) -> int:
    return int(outing.get("simulated_total_minutes") or 0)


def outing_official_to_total_ratio(outing: dict[str, Any]) -> float:
    total = total_foot_miles(outing)
    if total <= 0:
        return 0.0
    return official_miles(outing) / total


def outing_segment_ids(outing: dict[str, Any]) -> set[int]:
    return {int(segment_id) for segment_id in outing.get("segment_ids") or []}


def ready_outings(execution: dict[str, Any]) -> list[dict[str, Any]]:
    outings = [
        outing
        for outing in execution.get("outings") or []
        if outing.get("execution_status") == "simulated_ready"
    ]
    outings.sort(
        key=lambda item: (
            -official_miles(item),
            -float(item.get("simulated_efficiency_score") or 0),
            int(item.get("simulated_total_minutes") or 999999),
            item.get("trail_names") or [],
        )
    )
    return outings


def load_source_plan(execution: dict[str, Any]) -> dict[str, Any]:
    source_plan = execution.get("source_plan")
    if not source_plan:
        return {}
    path = Path(str(source_plan))
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_segment_mile_index(execution: dict[str, Any]) -> dict[int, float]:
    index: dict[int, float] = {}
    source_plan = load_source_plan(execution)
    for candidate in source_plan.get("route_menu", {}).get("all_candidates", []):
        for segment in candidate.get("segments") or []:
            index[int(segment["seg_id"])] = float(segment.get("official_miles") or 0)

    for outing in execution.get("outings") or []:
        for segment_id, miles in (outing.get("segment_miles") or {}).items():
            index[int(segment_id)] = float(miles)

    has_exact_index = bool(index)
    for outing in execution.get("outings") or []:
        segment_ids = outing_segment_ids(outing)
        if not segment_ids:
            continue
        fallback_miles = official_miles(outing) / len(segment_ids) if segment_ids else 0
        for segment_id in segment_ids:
            if has_exact_index and segment_id in index:
                continue
            index[segment_id] = max(index.get(segment_id, 0), fallback_miles)

    return index


def outing_route_for_preflight(outing: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": outing.get("candidate_id"),
        "candidate_id": outing.get("candidate_id"),
        "recommendation_type": "single_outing",
        "trail_names": outing.get("trail_names") or [],
        "trailheads": [outing["legs"][1].get("trailhead")],
    }


def choose_outing_for_day(
    day: date,
    outings: list[dict[str, Any]],
    used_segments: set[int],
    constraints: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    budget = day_budget(day, constraints)
    blocked_ids = []
    for outing in outings:
        outing_id = str(outing.get("candidate_id"))
        segment_ids = {int(segment_id) for segment_id in outing.get("segment_ids") or []}
        if not segment_ids or used_segments & segment_ids:
            continue
        if int(outing.get("simulated_total_minutes") or 0) > budget:
            continue
        preflight = evaluate_route_preflight(
            outing_route_for_preflight(outing),
            run_date=day.isoformat(),
            start_time=str(constraints["start_time"]),
        )
        if preflight["field_status"] == "blocked":
            blocked_ids.append(outing_id)
            continue
        return outing, blocked_ids
    return None, blocked_ids


def preflight_allows_outing_on_day(
    outing: dict[str, Any],
    day_record: dict[str, Any],
    constraints: dict[str, Any],
) -> bool:
    preflight = evaluate_route_preflight(
        outing_route_for_preflight(outing),
        run_date=day_record["date"],
        start_time=str(constraints["start_time"]),
    )
    return preflight["field_status"] != "blocked"


def outing_allowed_by_day_constraints(
    outing: dict[str, Any],
    day_record: dict[str, Any],
    constraints: dict[str, Any],
) -> bool:
    latest_scheduled = constraints.get("latest_scheduled_date")
    if latest_scheduled and str(day_record["date"]) > str(latest_scheduled):
        return False

    mop_up_start = constraints.get("final_mop_up_start_date")
    if not mop_up_start or str(day_record["date"]) < str(mop_up_start):
        return True

    max_minutes = constraints.get("final_mop_up_max_minutes")
    if max_minutes is not None and outing_minutes(outing) > int(max_minutes):
        return False

    max_on_foot = constraints.get("final_mop_up_max_on_foot_miles")
    if max_on_foot is not None and total_foot_miles(outing) > float(max_on_foot):
        return False

    max_ascent = constraints.get("final_mop_up_max_ascent_ft")
    if max_ascent is not None and outing_ascent_ft(outing) > int(max_ascent):
        return False

    return True


def outing_objective_cost(
    outing: dict[str, Any],
    day_index: int,
    day_record: dict[str, Any],
    constraints: dict[str, Any],
) -> float:
    profile = str(constraints.get("objective_profile") or "coverage_then_time")
    minutes = outing_minutes(outing)
    if profile == "coverage_then_time":
        return minutes * 0.001 + day_index * 0.00001

    official = official_miles(outing)
    on_foot = total_foot_miles(outing)
    overhead = max(0.0, on_foot - official)
    ascent = outing_ascent_ft(outing)
    cost = (
        on_foot * 1.0
        + overhead * 1.5
        + minutes * 0.02
        + (ascent / 1000.0) * 0.35
        + day_index * 0.00001
    )
    if outing_official_to_total_ratio(outing) < 0.35:
        cost += 2.0

    mop_up_start = constraints.get("final_mop_up_start_date")
    if mop_up_start and str(day_record["date"]) >= str(mop_up_start):
        if minutes >= int(constraints.get("long_outing_minutes") or 240):
            cost += 20.0
        if on_foot >= 10:
            cost += 15.0

    return cost


def optimize_outing_assignments(
    outings: list[dict[str, Any]],
    days: list[dict[str, Any]],
    constraints: dict[str, Any],
    segment_mile_index: dict[int, float],
) -> tuple[list[tuple[int, int]], set[str], dict[str, Any]]:
    feasible: dict[tuple[int, int], int] = {}
    preflight_blocked: set[str] = set()
    variable_count = 0
    for outing_index, outing in enumerate(outings):
        outing_id = str(outing.get("candidate_id"))
        for day_index, day_record in enumerate(days):
            if outing_minutes(outing) > int(day_record["available_minutes"]):
                continue
            if not outing_allowed_by_day_constraints(outing, day_record, constraints):
                continue
            if not preflight_allows_outing_on_day(outing, day_record, constraints):
                preflight_blocked.add(outing_id)
                continue
            feasible[(outing_index, day_index)] = variable_count
            variable_count += 1

    all_segments = sorted({segment_id for outing in outings for segment_id in outing_segment_ids(outing)})
    segment_variables = {
        segment_id: variable_count + index for index, segment_id in enumerate(all_segments)
    }
    variable_count += len(segment_variables)
    day_used_variables = {
        day_index: variable_count + day_index for day_index in range(len(days))
    }
    variable_count += len(day_used_variables)

    if not feasible:
        return [], preflight_blocked, {"optimizer_status": "no_feasible_outings"}

    objective = np.zeros(variable_count)
    for (outing_index, day_index), variable_index in feasible.items():
        objective[variable_index] = outing_objective_cost(
            outings[outing_index],
            day_index,
            days[day_index],
            constraints,
        )
    for segment_id, variable_index in segment_variables.items():
        objective[variable_index] = -(1000 + float(segment_mile_index.get(segment_id, 0)))
    for variable_index in day_used_variables.values():
        objective[variable_index] = 0.01

    rows: list[dict[int, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(coefficients: dict[int, float], lower: float, upper: float) -> None:
        if coefficients:
            rows.append(coefficients)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

    for outing_index, _outing in enumerate(outings):
        add_constraint(
            {
                variable_index: 1
                for (candidate_outing_index, _day_index), variable_index in feasible.items()
                if candidate_outing_index == outing_index
            },
            -np.inf,
            1,
        )

    for day_index, day_record in enumerate(days):
        add_constraint(
            {
                variable_index: outing_minutes(outings[outing_index])
                for (outing_index, candidate_day_index), variable_index in feasible.items()
                if candidate_day_index == day_index
            },
            -np.inf,
            int(day_record["available_minutes"]),
        )

    for (outing_index, day_index), variable_index in feasible.items():
        add_constraint({variable_index: 1, day_used_variables[day_index]: -1}, -np.inf, 0)

    for segment_id, segment_variable in segment_variables.items():
        coefficients = {segment_variable: 1}
        for (outing_index, _day_index), variable_index in feasible.items():
            if segment_id in outing_segment_ids(outings[outing_index]):
                coefficients[variable_index] = coefficients.get(variable_index, 0) - 1
        add_constraint(coefficients, -np.inf, 0)

    required_completion_segments = constraints.get("required_completion_segments")
    if required_completion_segments is not None:
        required_completion_segments = min(int(required_completion_segments), len(segment_variables))
        if required_completion_segments >= len(segment_variables):
            for segment_id in segment_variables:
                add_constraint(
                    {
                        variable_index: 1
                        for (outing_index, _day_index), variable_index in feasible.items()
                        if segment_id in outing_segment_ids(outings[outing_index])
                    },
                    1,
                    np.inf,
                )
        else:
            add_constraint(
                {variable_index: 1 for variable_index in segment_variables.values()},
                required_completion_segments,
                np.inf,
            )

    rest_days_after_long = int(constraints.get("rest_days_after_long") or 0)
    long_outing_minutes = int(constraints.get("long_outing_minutes") or 0)
    if rest_days_after_long > 0 and long_outing_minutes > 0:
        for (outing_index, day_index), variable_index in feasible.items():
            if outing_minutes(outings[outing_index]) < long_outing_minutes:
                continue
            for rest_day_index in range(
                day_index + 1,
                min(len(days), day_index + 1 + rest_days_after_long),
            ):
                add_constraint(
                    {variable_index: 1, day_used_variables[rest_day_index]: 1},
                    -np.inf,
                    1,
                )

    max_consecutive = constraints.get("max_consecutive_scheduled_days")
    if max_consecutive is not None:
        max_consecutive = int(max_consecutive)
        if max_consecutive > 0:
            for start_index in range(0, len(days) - max_consecutive):
                add_constraint(
                    {
                        day_used_variables[day_index]: 1
                        for day_index in range(start_index, start_index + max_consecutive + 1)
                    },
                    -np.inf,
                    max_consecutive,
                )

    min_open_days = int(constraints.get("min_open_days") or 0)
    if min_open_days > 0:
        add_constraint(
            {variable_index: 1 for variable_index in day_used_variables.values()},
            -np.inf,
            max(0, len(days) - min_open_days),
        )

    constraint_matrix = lil_matrix((len(rows), variable_count))
    for row_index, coefficients in enumerate(rows):
        for variable_index, value in coefficients.items():
            constraint_matrix[row_index, variable_index] = value

    result = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            constraint_matrix.tocsr(),
            np.array(lower_bounds),
            np.array(upper_bounds),
        ),
        options={"time_limit": int(constraints.get("optimizer_time_limit_seconds") or 30)},
    )

    values = result.x if result.x is not None else np.zeros(variable_count)
    selected = [
        key
        for key, variable_index in feasible.items()
        if values[variable_index] > 0.5
    ]
    selected.sort(key=lambda item: (item[1], outing_minutes(outings[item[0]]), outings[item[0]].get("candidate_id")))

    metadata = {
        "optimizer_status": "optimal" if result.success else "time_limited_or_feasible",
        "optimizer_message": str(result.message),
        "objective_value": float(result.fun) if result.fun is not None else None,
        "feasible_assignments": len(feasible),
        "required_completion_segments": constraints.get("required_completion_segments"),
    }
    return selected, preflight_blocked, metadata


def would_exceed_max_consecutive(
    day_status: list[str],
    day_index: int,
    max_consecutive: int | None,
) -> bool:
    if not max_consecutive or max_consecutive <= 0:
        return False
    simulated = list(day_status)
    simulated[day_index] = "scheduled"
    current = 0
    for status in simulated:
        if status == "scheduled":
            current += 1
            if current > max_consecutive:
                return True
        else:
            current = 0
    return False


def assignment_coverage(
    assignments: list[tuple[int, int]],
    outings: list[dict[str, Any]],
    segment_mile_index: dict[int, float],
) -> tuple[int, float, float, int]:
    covered = {segment_id for outing_index, _day_index in assignments for segment_id in outing_segment_ids(outings[outing_index])}
    official = sum(float(segment_mile_index.get(segment_id, 0)) for segment_id in covered)
    on_foot = sum(total_foot_miles(outings[outing_index]) for outing_index, _day_index in assignments)
    minutes = sum(outing_minutes(outings[outing_index]) for outing_index, _day_index in assignments)
    return len(covered), official, -on_foot, -minutes


def greedy_outing_assignments(
    outings: list[dict[str, Any]],
    days: list[dict[str, Any]],
    constraints: dict[str, Any],
    segment_mile_index: dict[int, float],
) -> tuple[list[tuple[int, int]], set[str], dict[str, Any]]:
    used_segments: set[int] = set()
    scheduled_outing_indexes: set[int] = set()
    preflight_blocked: set[str] = set()
    selected: list[tuple[int, int]] = []
    day_status = ["open" for _day in days]
    rest_days_after_long = int(constraints.get("rest_days_after_long") or 0)
    long_outing_minutes = int(constraints.get("long_outing_minutes") or 0)
    max_consecutive = constraints.get("max_consecutive_scheduled_days")
    max_consecutive = int(max_consecutive) if max_consecutive is not None else None
    min_open_days = int(constraints.get("min_open_days") or 0)
    max_scheduled_days = max(0, len(days) - min_open_days) if min_open_days > 0 else None

    for day_index, day_record in enumerate(days):
        if day_status[day_index] != "open":
            continue
        if max_scheduled_days is not None and day_status.count("scheduled") >= max_scheduled_days:
            break
        if would_exceed_max_consecutive(day_status, day_index, max_consecutive):
            continue
        remaining_minutes = int(day_record["available_minutes"])
        while True:
            best_index = None
            best_score = None
            for outing_index, outing in enumerate(outings):
                if outing_index in scheduled_outing_indexes:
                    continue
                if outing_minutes(outing) > remaining_minutes:
                    continue
                if not outing_allowed_by_day_constraints(outing, day_record, constraints):
                    continue
                new_segments = outing_segment_ids(outing) - used_segments
                if not new_segments:
                    continue
                if not preflight_allows_outing_on_day(outing, day_record, constraints):
                    preflight_blocked.add(str(outing.get("candidate_id")))
                    continue
                if rest_days_after_long > 0 and outing_minutes(outing) >= long_outing_minutes:
                    rest_range = range(
                        day_index + 1,
                        min(len(days), day_index + 1 + rest_days_after_long),
                    )
                    if any(day_status[rest_day_index] == "scheduled" for rest_day_index in rest_range):
                        continue
                new_miles = sum(float(segment_mile_index.get(segment_id, 0)) for segment_id in new_segments)
                score = (
                    len(new_segments),
                    new_miles,
                    -outing_objective_cost(outing, day_index, day_record, constraints),
                    float(outing.get("simulated_efficiency_score") or 0),
                    -outing_minutes(outing),
                    str(outing.get("candidate_id")),
                )
                if best_score is None or score > best_score:
                    best_index = outing_index
                    best_score = score
            if best_index is None:
                break
            selected.append((best_index, day_index))
            scheduled_outing_indexes.add(best_index)
            used_segments.update(outing_segment_ids(outings[best_index]))
            remaining_minutes -= outing_minutes(outings[best_index])
            day_status[day_index] = "scheduled"
            if rest_days_after_long > 0 and outing_minutes(outings[best_index]) >= long_outing_minutes:
                for rest_day_index in range(
                    day_index + 1,
                    min(len(days), day_index + 1 + rest_days_after_long),
                ):
                    if day_status[rest_day_index] == "open":
                        day_status[rest_day_index] = "recovery"

    return selected, preflight_blocked, {
        "optimizer_status": "coverage_greedy",
        "optimizer_message": "coverage-first greedy fallback",
        "feasible_assignments": None,
    }


def build_day_schedule_record(
    day_record: dict[str, Any],
    assigned_outings: list[dict[str, Any]],
    used_segments: set[int],
    segment_mile_index: dict[int, float],
) -> dict[str, Any]:
    scheduled_items = []
    day_new_segments: list[int] = []
    day_repeat_segments: list[int] = []
    day_new_miles = 0.0
    for outing in assigned_outings:
        segment_ids = sorted(outing_segment_ids(outing))
        new_segment_ids = [segment_id for segment_id in segment_ids if segment_id not in used_segments]
        repeat_segment_ids = [segment_id for segment_id in segment_ids if segment_id in used_segments]
        used_segments.update(new_segment_ids)
        day_new_segments.extend(new_segment_ids)
        day_repeat_segments.extend(repeat_segment_ids)
        new_miles = sum(float(segment_mile_index.get(segment_id, 0)) for segment_id in new_segment_ids)
        day_new_miles += new_miles
        scheduled_items.append(
            {
                "outing_id": outing.get("candidate_id"),
                "trail_names": outing.get("trail_names") or [],
                "trailhead": outing["legs"][1].get("trailhead"),
                "simulated_total_minutes": outing_minutes(outing),
                "raw_official_miles": round(official_miles(outing), 2),
                "new_official_miles": round(new_miles, 2),
                "estimated_total_on_foot_miles": round(total_foot_miles(outing), 2),
                "ascent_ft": outing_ascent_ft(outing),
                "descent_ft": outing_descent_ft(outing),
                "grade_adjusted_miles": round(outing_grade_adjusted_miles(outing), 2),
                "new_segment_ids": new_segment_ids,
                "repeat_segment_ids": repeat_segment_ids,
            }
        )

    total_minutes = sum(int(item["simulated_total_minutes"]) for item in scheduled_items)
    normal_minutes = day_record.get("normal_available_minutes")
    exception_minutes = (
        max(0, total_minutes - int(normal_minutes))
        if normal_minutes is not None
        else 0
    )
    trailheads = [str(item["trailhead"]) for item in scheduled_items]
    trail_names = [
        trail_name
        for item in scheduled_items
        for trail_name in item["trail_names"]
    ]
    return day_record | {
        "status": "scheduled",
        "reason": None,
        "outing_id": scheduled_items[0]["outing_id"],
        "outing_ids": [str(item["outing_id"]) for item in scheduled_items],
        "outings": scheduled_items,
        "trail_names": trail_names,
        "trailhead": " + ".join(dict.fromkeys(trailheads)),
        "trailheads": list(dict.fromkeys(trailheads)),
        "simulated_total_minutes": total_minutes,
        "normal_available_minutes": normal_minutes,
        "requires_normal_cap_exception": exception_minutes > 0,
        "normal_cap_exception_minutes": exception_minutes,
        "official_new_miles": round(day_new_miles, 2),
        "estimated_total_on_foot_miles": round(
            sum(float(item["estimated_total_on_foot_miles"]) for item in scheduled_items),
            2,
        ),
        "ascent_ft": sum(int(item.get("ascent_ft") or 0) for item in scheduled_items),
        "descent_ft": sum(int(item.get("descent_ft") or 0) for item in scheduled_items),
        "grade_adjusted_miles": round(
            sum(float(item.get("grade_adjusted_miles") or 0) for item in scheduled_items),
            2,
        ),
        "segment_ids": sorted(day_new_segments),
        "repeat_segment_ids": sorted(day_repeat_segments),
        "gpx_recommendation": "export or combine the listed executable outing GPX files",
    }


def build_calendar_schedule(
    execution: dict[str, Any],
    start_date: str,
    end_date: str,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_constraints = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    start = parse_date(start_date)
    end = parse_date(end_date)
    outings = ready_outings(execution)
    segment_mile_index = build_segment_mile_index(execution)
    used_segments: set[int] = set()
    days = [
        {
            "date": day.isoformat(),
            "status": "open",
            "available_minutes": day_budget(day, merged_constraints),
            "normal_available_minutes": normal_day_budget(day, merged_constraints),
            "reason": "no_simulated_ready_outing_assigned_by_optimizer",
        }
        for day in daterange(start, end)
    ]

    milp_assignments, milp_preflight_blocked, milp_metadata = optimize_outing_assignments(
        outings,
        days,
        merged_constraints,
        segment_mile_index,
    )
    greedy_assignments, greedy_preflight_blocked, greedy_metadata = greedy_outing_assignments(
        outings,
        days,
        merged_constraints,
        segment_mile_index,
    )
    if assignment_coverage(greedy_assignments, outings, segment_mile_index) > assignment_coverage(
        milp_assignments,
        outings,
        segment_mile_index,
    ):
        selected_assignments = greedy_assignments
        preflight_blocked = greedy_preflight_blocked
        optimizer_metadata = greedy_metadata | {
            "selected_strategy": "coverage_greedy",
            "milp_status": milp_metadata.get("optimizer_status"),
            "milp_message": milp_metadata.get("optimizer_message"),
        }
    else:
        selected_assignments = milp_assignments
        preflight_blocked = milp_preflight_blocked
        optimizer_metadata = milp_metadata | {"selected_strategy": "milp"}

    assignments_by_day: dict[int, list[dict[str, Any]]] = {}
    for outing_index, day_index in selected_assignments:
        assignments_by_day.setdefault(day_index, []).append(outings[outing_index])

    for day_index, assigned_outings in assignments_by_day.items():
        assigned_outings.sort(key=lambda outing: (outing_minutes(outing), outing.get("candidate_id")))
        days[day_index] = build_day_schedule_record(
            days[day_index],
            assigned_outings,
            used_segments,
            segment_mile_index,
        )

    rest_days_after_long = int(merged_constraints.get("rest_days_after_long") or 0)
    long_outing_minutes = int(merged_constraints.get("long_outing_minutes") or 0)
    if rest_days_after_long > 0 and long_outing_minutes > 0:
        for day_index, day_record in enumerate(days):
            if day_record["status"] != "scheduled":
                continue
            if not any(
                int(item.get("simulated_total_minutes") or 0) >= long_outing_minutes
                for item in day_record.get("outings", [])
            ):
                continue
            for rest_day_index in range(
                day_index + 1,
                min(len(days), day_index + 1 + rest_days_after_long),
            ):
                if days[rest_day_index]["status"] == "open":
                    days[rest_day_index]["status"] = "recovery"
                    days[rest_day_index]["reason"] = "recovery_after_long_outing"

    ready_ids = {str(outing.get("candidate_id")) for outing in outings}
    scheduled_outing_ids = {
        outing_id
        for day in days
        if day["status"] == "scheduled"
        for outing_id in day.get("outing_ids", [])
    }
    unscheduled_ready = sorted(ready_ids - scheduled_outing_ids)
    scheduled_days = [day for day in days if day["status"] == "scheduled"]
    ready_segment_ids = {segment_id for outing in outings for segment_id in outing_segment_ids(outing)}
    return {
        "run_id": utc_now_id(),
        "schedule_type": "optimized_assumption_based_draft",
        "source_execution_run_id": execution.get("run_id") or execution.get("generated_at"),
        "constraints": merged_constraints,
        "optimizer": optimizer_metadata,
        "days": days,
        "scheduled_outing_ids": sorted(scheduled_outing_ids),
        "unscheduled_ready_outing_ids": unscheduled_ready,
        "preflight_blocked_outing_ids": sorted(preflight_blocked),
        "summary": {
            "scheduled_outings": len(scheduled_days),
            "scheduled_executable_units": len(scheduled_outing_ids),
            "scheduled_official_miles": round(sum(float(day["official_new_miles"]) for day in scheduled_days), 2),
            "scheduled_total_on_foot_miles": round(
                sum(float(day["estimated_total_on_foot_miles"]) for day in scheduled_days), 2
            ),
            "scheduled_ascent_ft": sum(int(day.get("ascent_ft") or 0) for day in scheduled_days),
            "scheduled_descent_ft": sum(int(day.get("descent_ft") or 0) for day in scheduled_days),
            "scheduled_grade_adjusted_miles": round(
                sum(float(day.get("grade_adjusted_miles") or 0) for day in scheduled_days),
                2,
            ),
            "scheduled_segments": len(used_segments),
            "ready_segments_available": len(ready_segment_ids),
            "unscheduled_ready_segments": len(ready_segment_ids - used_segments),
            "unscheduled_ready_outings": len(unscheduled_ready),
            "open_or_recovery_days": len([day for day in days if day["status"] != "scheduled"]),
            "normal_cap_exception_days": len(
                [day for day in scheduled_days if day.get("requires_normal_cap_exception")]
            ),
            "normal_cap_exception_minutes": sum(
                int(day.get("normal_cap_exception_minutes") or 0) for day in scheduled_days
            ),
        },
        "caveats": [
            "This is not final without actual availability, recovery preferences, and target completion level.",
            "Only simulated-ready outings are scheduled; draft routes are excluded.",
            "Multiple outings on one day are conservative separate executable loops, each already includes drive, parking, run, return-to-car, and drive home time.",
            "Static preflight rules are applied, but current R2R conditions still need live checking.",
        ],
    }


def render_markdown(schedule: dict[str, Any]) -> str:
    lines = [
        "# 2026 Calendar Schedule Draft",
        "",
        f"- Schedule type: {schedule['schedule_type']}",
        f"- Scheduled outings: {schedule['summary']['scheduled_outings']}",
        f"- Scheduled official miles: {schedule['summary']['scheduled_official_miles']}",
        f"- Scheduled ascent: {schedule['summary'].get('scheduled_ascent_ft')} ft",
        f"- Scheduled segments: {schedule['summary']['scheduled_segments']}",
        f"- Unscheduled ready outings: {schedule['summary']['unscheduled_ready_outings']}",
        "",
        "## Assumptions",
        "",
    ]
    for key, value in schedule["constraints"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Days",
            "",
            "| Date | Status | Route | Minutes | Official mi | Ascent ft | Trailhead |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for day in schedule["days"]:
        route = ", ".join(day.get("trail_names") or [])
        lines.append(
            f"| {day['date']} "
            f"| {day['status']} "
            f"| {route or day.get('reason', '')} "
            f"| {day.get('simulated_total_minutes', '')} "
            f"| {day.get('official_new_miles', '')} "
            f"| {day.get('ascent_ft', '')} "
            f"| {day.get('trailhead', '')} |"
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in schedule["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-json", type=Path, default=DEFAULT_EXECUTION_JSON)
    parser.add_argument("--start-date", default="2026-06-18")
    parser.add_argument("--end-date", default="2026-07-18")
    parser.add_argument("--start-time", default=DEFAULT_CONSTRAINTS["start_time"])
    parser.add_argument("--weekday-max-minutes", type=int, default=DEFAULT_CONSTRAINTS["weekday_max_minutes"])
    parser.add_argument("--weekday-normal-max-minutes", type=int)
    parser.add_argument("--weekend-max-minutes", type=int, default=DEFAULT_CONSTRAINTS["weekend_max_minutes"])
    parser.add_argument("--long-outing-minutes", type=int, default=DEFAULT_CONSTRAINTS["long_outing_minutes"])
    parser.add_argument("--rest-days-after-long", type=int, default=DEFAULT_CONSTRAINTS["rest_days_after_long"])
    parser.add_argument("--max-consecutive-scheduled-days", type=int, default=None)
    parser.add_argument("--max-consecutive-scheduled-days-target", type=int)
    parser.add_argument("--min-open-days", type=int, default=DEFAULT_CONSTRAINTS["min_open_days"])
    parser.add_argument(
        "--objective-profile",
        choices=["coverage_then_time", "minimize_overhead"],
        default=DEFAULT_CONSTRAINTS["objective_profile"],
    )
    parser.add_argument("--latest-scheduled-date")
    parser.add_argument("--required-completion-segments", type=int)
    parser.add_argument("--required-rest-days-before-latest", type=int)
    parser.add_argument("--max-weekly-on-foot-miles", type=float)
    parser.add_argument("--long-on-foot-day-miles", type=float)
    parser.add_argument("--max-long-on-foot-days-per-week", type=int)
    parser.add_argument("--final-mop-up-start-date")
    parser.add_argument("--final-mop-up-max-minutes", type=int)
    parser.add_argument("--final-mop-up-max-on-foot-miles", type=float)
    parser.add_argument("--final-mop-up-max-ascent-ft", type=int)
    parser.add_argument(
        "--optimizer-time-limit-seconds",
        type=int,
        default=DEFAULT_CONSTRAINTS["optimizer_time_limit_seconds"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    execution = load_json(args.execution_json)
    constraints = {
        "start_time": args.start_time,
        "weekday_max_minutes": args.weekday_max_minutes,
        "weekday_normal_max_minutes": args.weekday_normal_max_minutes,
        "weekend_max_minutes": args.weekend_max_minutes,
        "long_outing_minutes": args.long_outing_minutes,
        "rest_days_after_long": args.rest_days_after_long,
        "max_consecutive_scheduled_days": args.max_consecutive_scheduled_days,
        "max_consecutive_scheduled_days_target": args.max_consecutive_scheduled_days_target,
        "min_open_days": args.min_open_days,
        "objective_profile": args.objective_profile,
        "latest_scheduled_date": args.latest_scheduled_date,
        "required_completion_segments": args.required_completion_segments,
        "required_rest_days_before_latest": args.required_rest_days_before_latest,
        "max_weekly_on_foot_miles": args.max_weekly_on_foot_miles,
        "long_on_foot_day_miles": args.long_on_foot_day_miles,
        "max_long_on_foot_days_per_week": args.max_long_on_foot_days_per_week,
        "final_mop_up_start_date": args.final_mop_up_start_date,
        "final_mop_up_max_minutes": args.final_mop_up_max_minutes,
        "final_mop_up_max_on_foot_miles": args.final_mop_up_max_on_foot_miles,
        "final_mop_up_max_ascent_ft": args.final_mop_up_max_ascent_ft,
        "optimizer_time_limit_seconds": args.optimizer_time_limit_seconds,
    }
    schedule = build_calendar_schedule(
        execution,
        start_date=args.start_date,
        end_date=args.end_date,
        constraints=constraints,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.start_date}-to-{args.end_date}-draft"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(schedule, indent=2) + "\n")
    md_path.write_text(render_markdown(schedule))
    manifest_path = args.output_dir / f"{stem}-artifact-manifest.json"
    write_manifest(
        manifest_path,
        build_artifact_manifest(
            run_id=schedule["run_id"],
            inputs=[args.execution_json],
            outputs=[json_path, md_path],
            command="calendar_scheduler.py",
            metadata={
                "start_date": args.start_date,
                "end_date": args.end_date,
                "source_execution_run_id": schedule.get("source_execution_run_id"),
            },
        ),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    print(
        "Schedule: "
        f"{schedule['summary']['scheduled_outings']} outings, "
        f"{schedule['summary']['scheduled_official_miles']} official miles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
