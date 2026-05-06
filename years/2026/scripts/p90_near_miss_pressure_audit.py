#!/usr/bin/env python3
"""Audit day-count pressure for a near-miss p90 sensitivity scenario."""

from __future__ import annotations

import argparse
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

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402
from p90_completion_gap_analyzer import official_segments  # noqa: E402
from personal_route_planner import DEFAULT_OFFICIAL_GEOJSON, read_json  # noqa: E402


DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-near-miss-pressure-audit-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def day_count_delta(required: int | None, available: int) -> int | None:
    if required is None:
        return None
    return max(0, int(required) - int(available))


def selected_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    if not selected:
        return {
            "field_day_count": 0,
            "weekday_field_day_count": 0,
            "weekend_field_day_count": 0,
            "total_p75_minutes": 0,
            "max_p90_minutes": 0,
            "max_p90_stress": 0,
        }
    return {
        "field_day_count": len(selected),
        "weekday_field_day_count": sum(1 for day in selected if day["day_type"] == "weekday"),
        "weekend_field_day_count": sum(1 for day in selected if day["day_type"] == "weekend"),
        "total_p75_minutes": int(sum(day["p75_minutes"] for day in selected)),
        "max_p90_minutes": int(max(day["p90_minutes"] for day in selected)),
        "max_p90_stress": round(max(float(day["stress"]) for day in selected), 3),
    }


def solve_full_cover_with_limits(
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

    costs = []
    for day in field_days:
        weekday = 1.0 if day["day_type"] == "weekday" else 0.0
        weekend = 1.0 if day["day_type"] == "weekend" else 0.0
        if cost_mode == "min_weekdays":
            primary = weekday
            secondary = 0.001
        elif cost_mode == "min_weekends":
            primary = weekend
            secondary = 0.001
        else:
            primary = 1.0
            secondary = 0.0
        costs.append(
            primary
            + secondary
            + float(day["p75_minutes"]) * 0.000001
            + float(day["stress"]) * 0.0000001
        )
    result = milp(
        c=np.array(costs, dtype=float),
        integrality=np.ones(len(field_days), dtype=int),
        bounds=Bounds(np.zeros(len(field_days)), np.ones(len(field_days))),
        constraints=LinearConstraint(matrix.tocsr(), lb=np.array(lower), ub=np.array(upper)),
        options={"time_limit": 120},
    )
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}
    selected = [day for day, value in zip(field_days, result.x) if value >= 0.5]
    summary = selected_summary(selected)
    return {
        "success": True,
        **summary,
        "selected_field_days": sorted(
            selected,
            key=lambda day: (day["day_type"], day["p75_minutes"], day["field_day_id"]),
        ),
    }


def build_field_days(
    *,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
    inter_trailhead_drive_minutes: int | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[str, int], dict[int, float]]:
    state = read_json(joint_optimizer.repaired.DEFAULT_STATE_JSON)
    if inter_trailhead_drive_minutes is not None:
        state.setdefault("availability_model", {})["acceptable_inter_trailhead_drive_minutes"] = int(inter_trailhead_drive_minutes)
    target_ids, candidates, field_menu = joint_optimizer.all_repaired_candidates()
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
    loops, _missing_coordinates = joint_optimizer.candidate_loop_records(bounded, field_menu, state)
    availability = state.get("availability_model") or {}
    counts = joint_optimizer.pack_audit.date_counts(
        (availability.get("available_dates") or {})["start"],
        (availability.get("available_dates") or {})["end"],
    )
    field_days = joint_optimizer.generate_direct_field_day_candidates(
        loops,
        state,
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
    )
    return target_ids, field_days, counts, official_miles_by_segment


def build_report(
    *,
    weekday_bound: int,
    weekend_bound: int,
    max_combo_size: int,
    neighbor_limit: int,
    inter_trailhead_drive_minutes: int | None = None,
) -> dict[str, Any]:
    target_ids, field_days, counts, official_miles_by_segment = build_field_days(
        weekday_bound=weekday_bound,
        weekend_bound=weekend_bound,
        max_combo_size=max_combo_size,
        neighbor_limit=neighbor_limit,
        inter_trailhead_drive_minutes=inter_trailhead_drive_minutes,
    )
    baseline = joint_optimizer.solve_direct_field_day_max_coverage(
        field_days,
        target_ids,
        counts,
        official_miles_by_segment=official_miles_by_segment,
    )
    p75_min_full = joint_optimizer.solve_direct_field_day_cover(field_days, target_ids, counts)
    actual_full = solve_full_cover_with_limits(
        field_days,
        target_ids,
        weekday_limit=counts["weekday"],
        weekend_limit=counts["weekend"],
        cost_mode="min_days",
    )
    min_days = solve_full_cover_with_limits(
        field_days,
        target_ids,
        weekday_limit=100,
        weekend_limit=100,
        cost_mode="min_days",
    )
    min_weekdays = solve_full_cover_with_limits(
        field_days,
        target_ids,
        weekday_limit=100,
        weekend_limit=counts["weekend"],
        cost_mode="min_weekdays",
    )
    min_weekends = solve_full_cover_with_limits(
        field_days,
        target_ids,
        weekday_limit=counts["weekday"],
        weekend_limit=100,
        cost_mode="min_weekends",
    )
    required_weekdays = min_weekdays.get("weekday_field_day_count") if min_weekdays.get("success") else None
    required_weekends = min_weekends.get("weekend_field_day_count") if min_weekends.get("success") else None
    return {
        "objective": "diagnose day-count pressure for the 292/360 near-miss field-day scenario",
        "source_files": {
            "joint_optimizer": display_path(YEAR_DIR / "scripts" / "p90_joint_field_day_optimizer.py"),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
            "state_json": display_path(joint_optimizer.repaired.DEFAULT_STATE_JSON),
        },
        "config": {
            "weekday_bound_minutes": weekday_bound,
            "weekend_bound_minutes": weekend_bound,
            "max_combo_size": max_combo_size,
            "neighbor_limit": neighbor_limit,
            "inter_trailhead_drive_minutes": inter_trailhead_drive_minutes,
        },
        "available_day_counts": counts,
        "field_day_candidate_count": len(field_days),
        "baseline_max_coverage": baseline,
        "p75_min_full_cover": p75_min_full,
        "actual_day_count_full_cover": actual_full,
        "full_cover_pressure": {
            "min_total_days_unlimited_day_types": min_days,
            "min_weekdays_with_actual_weekends": min_weekdays,
            "min_weekends_with_actual_weekdays": min_weekends,
            "extra_weekdays_needed_if_weekends_fixed": day_count_delta(required_weekdays, counts["weekday"]),
            "extra_weekends_needed_if_weekdays_fixed": day_count_delta(required_weekends, counts["weekend"]),
        },
        "interpretation": [
            "The baseline max-coverage solution is the best 31-day coverage found under the generated field-day universe.",
            "If full cover needs extra weekdays or weekends, the route planner needs either more available long days or better route grouping that reduces field-day count.",
            "A segment being individually runnable does not prove it can fit the schedule without displacing higher-value field days.",
        ],
    }


def compact_solution(solution: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "success",
        "field_day_count",
        "weekday_field_day_count",
        "weekend_field_day_count",
        "covered_segment_count",
        "missing_segment_count",
        "missing_segment_ids",
        "covered_official_miles",
        "missing_official_miles",
        "total_p75_minutes",
        "max_p90_minutes",
        "max_p90_stress",
        "reason",
        "message",
    ]
    return {key: solution.get(key) for key in keys if key in solution}


def render_md(report: dict[str, Any]) -> str:
    baseline = report["baseline_max_coverage"]
    pressure = report["full_cover_pressure"]
    actual = report["actual_day_count_full_cover"]
    p75_min = report["p75_min_full_cover"]
    min_days = pressure["min_total_days_unlimited_day_types"]
    min_weekdays = pressure["min_weekdays_with_actual_weekends"]
    min_weekends = pressure["min_weekends_with_actual_weekdays"]
    lines = [
        "# P90 Near-Miss Pressure Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Scenario",
        "",
        f"- P90 bounds: {report['config']['weekday_bound_minutes']} weekday / {report['config']['weekend_bound_minutes']} weekend minutes",
        f"- Inter-trailhead drive limit: {report['config']['inter_trailhead_drive_minutes'] if report['config']['inter_trailhead_drive_minutes'] is not None else 'state default'} minutes",
        f"- Neighbor search limit: {report['config']['neighbor_limit']}",
        f"- Available field days: {report['available_day_counts']['weekday']} weekday / {report['available_day_counts']['weekend']} weekend",
        f"- Generated field-day candidates: {report['field_day_candidate_count']}",
        "",
        "## Result",
        "",
        f"- Full cover with actual day counts: {actual.get('success')}",
        f"- P75-min full cover with actual day counts: {p75_min.get('success')}",
        f"- P75-min full-cover total p75: {p75_min.get('total_p75_minutes') if p75_min.get('success') else 'not solved'}",
        f"- Best 31-day coverage: {baseline.get('covered_segment_count')}/251 segments, {baseline.get('covered_official_miles')} official mi",
        f"- Missing from best 31-day coverage: {', '.join(str(seg_id) for seg_id in baseline.get('missing_segment_ids') or []) or 'none'}",
        f"- Minimum full-cover days if day types are unlimited: {min_days.get('field_day_count') if min_days.get('success') else 'not solved'}",
        f"- Minimum weekdays with 9 weekends fixed: {min_weekdays.get('weekday_field_day_count') if min_weekdays.get('success') else 'not solved'}",
        f"- Minimum weekends with 22 weekdays fixed: {min_weekends.get('weekend_field_day_count') if min_weekends.get('success') else 'not solved'}",
        f"- Extra weekdays needed if weekend count is fixed: {pressure['extra_weekdays_needed_if_weekends_fixed']}",
        f"- Extra weekends needed if weekday count is fixed: {pressure['extra_weekends_needed_if_weekdays_fixed']}",
        "",
        "## Compact Solutions",
        "",
        "```json",
        json.dumps(
            {
                "baseline_max_coverage": compact_solution(baseline),
                "p75_min_full_cover": compact_solution(p75_min),
                "actual_day_count_full_cover": compact_solution(actual),
                "min_total_days_unlimited_day_types": compact_solution(min_days),
                "min_weekdays_with_actual_weekends": compact_solution(min_weekdays),
                "min_weekends_with_actual_weekdays": compact_solution(min_weekends),
            },
            indent=2,
        ),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekday-bound", type=int, default=292)
    parser.add_argument("--weekend-bound", type=int, default=360)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--neighbor-limit", type=int, default=20)
    parser.add_argument("--inter-trailhead-drive-minutes", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        weekday_bound=args.weekday_bound,
        weekend_bound=args.weekend_bound,
        max_combo_size=args.max_combo_size,
        neighbor_limit=args.neighbor_limit,
        inter_trailhead_drive_minutes=args.inter_trailhead_drive_minutes,
    )
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        json.dumps(
            {
                "baseline_covered_segments": report["baseline_max_coverage"].get("covered_segment_count"),
                "baseline_missing_segments": report["baseline_max_coverage"].get("missing_segment_ids"),
                "extra_weekdays_needed_if_weekends_fixed": report["full_cover_pressure"]["extra_weekdays_needed_if_weekends_fixed"],
                "extra_weekends_needed_if_weekdays_fixed": report["full_cover_pressure"]["extra_weekends_needed_if_weekdays_fixed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
