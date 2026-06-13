#!/usr/bin/env python3
"""Extract the strict-current-profile max-coverage field-day fallback plan."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_OPTIMIZER_JSON = YEAR_DIR / "checkpoints" / "p90-joint-field-day-optimizer-wide-2026-05-06.json"
DEFAULT_OFFICIAL_GEOJSON = YEAR_DIR / "inputs" / "official" / "api-pull-2026-06-13" / "official_foot_segments.geojson"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-strict-profile-max-coverage-plan-2026-05-06"


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


def select_strict_scenario(optimizer: dict[str, Any]) -> dict[str, Any]:
    for scenario in optimizer.get("scenarios") or []:
        if scenario.get("scenario") == "strict_current_p90_bounds" and scenario.get("current_rule_compliant") is True:
            return scenario
    raise ValueError("strict_current_p90_bounds scenario not found")


def challenge_dates(start: date = date(2026, 6, 18), end: date = date(2026, 7, 18)) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def day_type_for(value: date) -> str:
    return "weekend" if value.weekday() >= 5 else "weekday"


def touches_lower_hulls(day: dict[str, Any]) -> bool:
    haystack = " ".join(str(value).lower() for value in [day.get("field_day_id"), *(day.get("loop_ids") or [])])
    return "lower-hulls" in haystack or "lower hull" in haystack


def assign_dates(field_days: list[dict[str, Any]], dates: list[date]) -> list[dict[str, Any]]:
    remaining = list(field_days)
    available = list(dates)
    assignments = []

    def take_date(day: dict[str, Any]) -> date:
        for index, candidate in enumerate(available):
            if day_type_for(candidate) != day.get("day_type"):
                continue
            if touches_lower_hulls(day) and candidate.day % 2 != 0:
                continue
            return available.pop(index)
        raise ValueError(f"No date available for field day {day.get('field_day_id')}")

    lower_hulls_first = sorted(remaining, key=lambda item: (not touches_lower_hulls(item), item.get("p90_minutes", 0), item.get("field_day_id", "")))
    for day in lower_hulls_first:
        assigned = take_date(day)
        assignments.append(
            {
                "date": assigned.isoformat(),
                "day_of_month": assigned.day,
                "weekday_name": assigned.strftime("%A"),
                "day_type": day_type_for(assigned),
                "is_even_day": assigned.day % 2 == 0,
                "constraints": ["lower_hulls_even_day_on_foot"] if touches_lower_hulls(day) else [],
                "field_day": day,
            }
        )
    return sorted(assignments, key=lambda row: row["date"])


def official_segment_index(official_geojson: dict[str, Any]) -> dict[int, dict[str, Any]]:
    index = {}
    for feature in official_geojson.get("features") or []:
        props = feature.get("properties") or {}
        seg_id = int(props["segId"])
        seg_name = str(props.get("segName") or "")
        trail_name = seg_name.rsplit(" ", 1)[0] if seg_name.rsplit(" ", 1)[-1].isdigit() else seg_name
        index[seg_id] = {
            "seg_id": seg_id,
            "seg_name": seg_name,
            "trail_name": trail_name,
            "official_miles": round(float(props.get("LengthFt") or 0) / 5280, 2),
        }
    return index


def missing_segment_rows(segment_ids: list[int], official_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [official_index[int(seg_id)] for seg_id in segment_ids]


def build_report(
    *,
    optimizer: dict[str, Any],
    official_geojson: dict[str, Any],
) -> dict[str, Any]:
    scenario = select_strict_scenario(optimizer)
    solution = scenario["max_coverage_solution"]
    official_index = official_segment_index(official_geojson)
    field_days = list(solution.get("selected_field_days") or [])
    assignments = assign_dates(field_days, challenge_dates())
    missing_rows = missing_segment_rows(solution.get("missing_segment_ids") or [], official_index)
    return {
        "objective": "extract strict-current-profile max-coverage fallback field days",
        "source_files": {
            "optimizer_json": display_path(DEFAULT_OPTIMIZER_JSON),
            "official_geojson": display_path(DEFAULT_OFFICIAL_GEOJSON),
        },
        "status": "partial_strict_profile_fallback_not_completion",
        "profile": {
            "scenario": scenario["scenario"],
            "weekday_bound_minutes": scenario["weekday_bound_minutes"],
            "weekend_bound_minutes": scenario["weekend_bound_minutes"],
            "current_rule_compliant": scenario["current_rule_compliant"],
        },
        "summary": {
            "accepted_as_completion_plan": False,
            "reason_not_completion": "strict current profile max-coverage solution misses official segments",
            "field_day_count": solution.get("field_day_count"),
            "weekday_field_day_count": solution.get("weekday_field_day_count"),
            "weekend_field_day_count": solution.get("weekend_field_day_count"),
            "covered_segment_count": solution.get("covered_segment_count"),
            "missing_segment_count": solution.get("missing_segment_count"),
            "covered_official_miles": solution.get("covered_official_miles"),
            "missing_official_miles": solution.get("missing_official_miles"),
            "total_p75_minutes": solution.get("total_p75_minutes"),
            "max_p90_stress": solution.get("max_p90_stress"),
            "lower_hulls_even_day_violation_count": sum(
                1 for row in assignments if row["constraints"] and not row["is_even_day"]
            ),
        },
        "assignments": assignments,
        "missing_segments": missing_rows,
        "known_gaps": [
            "This is not a completion plan because it misses official segments.",
            "It is useful as the strict-profile max-coverage fallback while the 100% profile decision is unresolved.",
            "Day-level GPX export is not generated for this partial fallback in this artifact.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    profile = report["profile"]
    lines = [
        "# Strict Profile Max-Coverage Plan",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Verdict",
        "",
        f"- Accepted as completion plan: {summary['accepted_as_completion_plan']}",
        f"- Status: `{report['status']}`",
        f"- Scenario: `{profile['scenario']}`",
        f"- P90 bounds: {profile['weekday_bound_minutes']} weekday / {profile['weekend_bound_minutes']} weekend",
        "",
        "## Summary",
        "",
        f"- Field days: {summary['field_day_count']} ({summary['weekday_field_day_count']} weekday / {summary['weekend_field_day_count']} weekend)",
        f"- Covered segments: {summary['covered_segment_count']}",
        f"- Missing segments: {summary['missing_segment_count']}",
        f"- Covered official miles: {summary['covered_official_miles']}",
        f"- Missing official miles: {summary['missing_official_miles']}",
        f"- Total p75: {summary['total_p75_minutes']} min",
        f"- Max p90 stress: {summary['max_p90_stress']}",
        "",
        "## Missing Segments",
        "",
        "| Segment | Trail | Official mi |",
        "|---:|---|---:|",
    ]
    for row in report["missing_segments"]:
        lines.append(f"| {row['seg_id']} | {row['seg_name']} | {row['official_miles']} |")
    lines.extend(
        [
            "",
            "## Field Days",
            "",
            "| Date | Type | P75 | P90 | Bound | Official segments | On foot | Field day |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for assignment in report["assignments"]:
        day = assignment["field_day"]
        lines.append(
            f"| {assignment['date']} | {assignment['day_type']} | {day['p75_minutes']} | "
            f"{day['p90_minutes']} | {day['p90_bound_minutes']} | {len(day['segment_ids'])} | "
            f"{day['on_foot_miles']} | {day['field_day_id']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-json", type=Path, default=DEFAULT_OPTIMIZER_JSON)
    parser.add_argument("--official-geojson", type=Path, default=DEFAULT_OFFICIAL_GEOJSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        optimizer=read_json(args.optimizer_json),
        official_geojson=read_json(args.official_geojson),
    )
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
