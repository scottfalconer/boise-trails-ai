#!/usr/bin/env python3
"""Assign the relaxed-drive draft field days to challenge-window dates."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent

DEFAULT_INPUT_JSON = YEAR_DIR / "checkpoints" / "p90-relaxed-drive-draft-field-day-plan-2026-05-06.json"
DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-relaxed-drive-calendar-assignment-2026-05-06"
DEFAULT_START = "2026-06-18"
DEFAULT_END = "2026-07-18"


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


def challenge_dates(start: str, end: str) -> list[dict[str, Any]]:
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    rows = []
    while current <= last:
        rows.append(
            {
                "date": current.isoformat(),
                "day_of_month": current.day,
                "weekday_name": current.strftime("%A"),
                "day_type": "weekend" if current.weekday() >= 5 else "weekday",
                "is_even_day": current.day % 2 == 0,
            }
        )
        current += timedelta(days=1)
    return rows


def day_mentions(day: dict[str, Any], text: str) -> bool:
    haystacks = [day.get("field_day_id") or ""]
    for loop in day.get("loops") or []:
        haystacks.append(str(loop.get("label") or ""))
        haystacks.append(str(loop.get("candidate_id") or ""))
        haystacks.extend(str(name) for name in loop.get("trail_names") or [])
    return text.lower() in " ".join(haystacks).lower()


def schedule_constraints(day: dict[str, Any]) -> list[str]:
    constraints = []
    if day_mentions(day, "lower-hulls") or day_mentions(day, "lower hull"):
        constraints.append("lower_hulls_even_day_on_foot")
    return constraints


def assign_days_to_dates(days: list[dict[str, Any]], dates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining_dates = dates[:]
    assignments: list[dict[str, Any]] = []
    constrained = [day for day in days if schedule_constraints(day)]
    unconstrained = [day for day in days if not schedule_constraints(day)]

    def take_date(day: dict[str, Any]) -> dict[str, Any]:
        constraints = schedule_constraints(day)
        for index, candidate_date in enumerate(remaining_dates):
            if candidate_date["day_type"] != day["day_type"]:
                continue
            if "lower_hulls_even_day_on_foot" in constraints and not candidate_date["is_even_day"]:
                continue
            return remaining_dates.pop(index)
        raise ValueError(f"No date available for day {day['draft_day_number']} with constraints {constraints}")

    for day in sorted(constrained, key=lambda item: item["draft_day_number"]):
        assignment_date = take_date(day)
        assignments.append({**assignment_date, "constraints": schedule_constraints(day), "field_day": day})
    for day in sorted(unconstrained, key=lambda item: item["draft_day_number"]):
        assignment_date = take_date(day)
        assignments.append({**assignment_date, "constraints": [], "field_day": day})
    return sorted(assignments, key=lambda item: item["date"])


def audit_assignments(assignments: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    official_count = int((plan.get("coverage") or {}).get("official_segment_count") or 0)
    covered_ids = sorted(
        {
            seg_id
            for assignment in assignments
            for seg_id in assignment["field_day"]["segment_summary"]["segment_ids"]
        }
    )
    lower_hulls_violations = [
        assignment
        for assignment in assignments
        if "lower_hulls_even_day_on_foot" in assignment["constraints"] and not assignment["is_even_day"]
    ]
    type_violations = [
        assignment
        for assignment in assignments
        if assignment["day_type"] != assignment["field_day"]["day_type"]
    ]
    p90_violations = [
        assignment
        for assignment in assignments
        if int(assignment["field_day"]["p90_minutes"]) > int(assignment["field_day"]["p90_bound_minutes"])
    ]
    return {
        "assigned_day_count": len(assignments),
        "covered_segment_count": len(covered_ids),
        "official_segment_count": official_count,
        "missing_segment_count": official_count - len(covered_ids),
        "day_type_violation_count": len(type_violations),
        "lower_hulls_even_day_violation_count": len(lower_hulls_violations),
        "p90_violation_count": len(p90_violations),
        "passed": (
            len(assignments) == len(plan.get("field_days") or [])
            and len(covered_ids) == official_count
            and not type_violations
            and not lower_hulls_violations
            and not p90_violations
        ),
    }


def build_report(plan: dict[str, Any], *, start: str, end: str) -> dict[str, Any]:
    dates = challenge_dates(start, end)
    assignments = assign_days_to_dates(plan.get("field_days") or [], dates)
    audit = audit_assignments(assignments, plan)
    return {
        "objective": "assign relaxed-drive draft field days to concrete 2026 challenge dates",
        "source_files": {
            "draft_field_day_plan_json": display_path(DEFAULT_INPUT_JSON),
        },
        "challenge_window": {"start": start, "end": end},
        "status": "draft_calendar_needs_human_review",
        "audit": audit,
        "known_gaps": [
            "This is a deterministic date assignment, not a recovery/rest optimizer.",
            "Day-level multi-loop GPX files still need export and validation.",
            "Current Ridge to Rivers conditions/signage are not checked here.",
            "The plan still uses the relaxed 292/360 + 45-minute drive sensitivity profile.",
        ],
        "assignments": assignments,
    }


def render_md(report: dict[str, Any]) -> str:
    audit = report["audit"]
    lines = [
        "# P90 Relaxed-Drive Calendar Assignment",
        "",
        f"Objective: {report['objective']}",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Audit",
        "",
        f"- Assigned days: {audit['assigned_day_count']}",
        f"- Segment coverage: {audit['covered_segment_count']}/{audit['official_segment_count']}",
        f"- Day-type violations: {audit['day_type_violation_count']}",
        f"- Lower Hulls even-day violations: {audit['lower_hulls_even_day_violation_count']}",
        f"- P90 violations: {audit['p90_violation_count']}",
        f"- Passed assignment checks: {audit['passed']}",
        "",
        "## Known Gaps",
        "",
    ]
    lines.extend(f"- {gap}" for gap in report["known_gaps"])
    lines.extend(
        [
            "",
            "## Assigned Field Days",
            "",
            "| Date | Type | Draft # | P90 | Starts | Between drive | Official mi | Primary loops | Constraints |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for assignment in report["assignments"]:
        day = assignment["field_day"]
        loop_labels = "; ".join(str(loop.get("label") or loop.get("candidate_id") or loop.get("loop_id")) for loop in day["loops"])
        constraints = ", ".join(assignment["constraints"])
        lines.append(
            f"| {assignment['date']} {assignment['weekday_name']} | {assignment['day_type']} | "
            f"{day['draft_day_number']} | {day['p90_minutes']} | {day['loop_count']} | "
            f"{day['between_drive_minutes']} | {day['segment_summary']['official_miles']} | "
            f"{loop_labels} | {constraints} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(read_json(args.input_json), start=args.start, end=args.end)
    json_path = args.output_dir / f"{args.basename}.json"
    md_path = args.output_dir / f"{args.basename}.md"
    write_json(json_path, report)
    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["audit"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
