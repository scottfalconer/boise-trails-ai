#!/usr/bin/env python3
"""Run joint field-day optimizer across plausible availability bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_DIR = SCRIPT_DIR.parent
REPO_ROOT = YEAR_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as joint_optimizer  # noqa: E402


DEFAULT_OUTPUT_DIR = YEAR_DIR / "checkpoints"
DEFAULT_BASENAME = "p90-availability-sensitivity-audit-2026-05-06"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def scenario_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "current_260_weekday_180_weekend",
            "weekday_bound": 260,
            "weekend_bound": 180,
            "current_rule_compliant": True,
        },
        {
            "name": "shingle_floor_292_weekday_180_weekend",
            "weekday_bound": 292,
            "weekend_bound": 180,
            "current_rule_compliant": False,
        },
        {
            "name": "292_weekday_240_weekend",
            "weekday_bound": 292,
            "weekend_bound": 240,
            "current_rule_compliant": False,
        },
        {
            "name": "292_weekday_292_weekend",
            "weekday_bound": 292,
            "weekend_bound": 292,
            "current_rule_compliant": False,
        },
        {
            "name": "292_weekday_360_weekend",
            "weekday_bound": 292,
            "weekend_bound": 360,
            "current_rule_compliant": False,
        },
        {
            "name": "320_weekday_240_weekend",
            "weekday_bound": 320,
            "weekend_bound": 240,
            "current_rule_compliant": False,
        },
        {
            "name": "320_weekday_292_weekend",
            "weekday_bound": 320,
            "weekend_bound": 292,
            "current_rule_compliant": False,
        },
        {
            "name": "360_weekday_360_weekend",
            "weekday_bound": 360,
            "weekend_bound": 360,
            "current_rule_compliant": False,
        },
    ]


def scenario_brief(scenario: dict[str, Any]) -> dict[str, Any]:
    solution = scenario.get("solution") or {}
    max_coverage = scenario.get("max_coverage_solution") or {}
    pressure = scenario.get("pressure_diagnostics") or {}
    min_days = pressure.get("min_total_days_unlimited_day_counts") or {}
    min_weekdays = pressure.get("min_weekdays_with_actual_weekend_count") or {}
    return {
        "scenario": scenario["scenario"],
        "weekday_bound_minutes": scenario["weekday_bound_minutes"],
        "weekend_bound_minutes": scenario["weekend_bound_minutes"],
        "current_rule_compliant": scenario["current_rule_compliant"],
        "field_day_candidate_count": scenario["field_day_candidate_count"],
        "feasible": solution.get("success") is True,
        "field_day_count": solution.get("field_day_count"),
        "weekday_field_day_count": solution.get("weekday_field_day_count"),
        "weekend_field_day_count": solution.get("weekend_field_day_count"),
        "total_p75_minutes": solution.get("total_p75_minutes"),
        "max_coverage_segments": max_coverage.get("covered_segment_count"),
        "max_coverage_official_miles": max_coverage.get("covered_official_miles"),
        "max_coverage_missing_segments": max_coverage.get("missing_segment_count"),
        "max_coverage_missing_official_miles": max_coverage.get("missing_official_miles"),
        "relaxed_min_field_days": min_days.get("field_day_count"),
        "relaxed_min_weekdays_with_actual_weekends": min_weekdays.get("weekday_field_day_count"),
        "solution_reason": solution.get("reason") or solution.get("message"),
    }


def build_report(*, max_combo_size: int, neighbor_limit: int, connected_expansion_limit: int) -> dict[str, Any]:
    scenarios = [
        joint_optimizer.build_scenario(
            name=spec["name"],
            weekday_bound=spec["weekday_bound"],
            weekend_bound=spec["weekend_bound"],
            max_combo_size=max_combo_size,
            neighbor_limit=neighbor_limit,
            connected_expansion_limit=connected_expansion_limit,
            current_rule_compliant=spec["current_rule_compliant"],
        )
        for spec in scenario_specs()
    ]
    summaries = [scenario_brief(scenario) for scenario in scenarios]
    feasible = [summary for summary in summaries if summary["feasible"]]
    best_coverage = max(
        summaries,
        key=lambda summary: (
            int(summary.get("max_coverage_segments") or 0),
            float(summary.get("max_coverage_official_miles") or 0.0),
            -int(summary.get("weekday_bound_minutes") or 0),
            -int(summary.get("weekend_bound_minutes") or 0),
        ),
    )
    first_feasible = min(
        feasible,
        key=lambda summary: (
            int(summary["weekday_bound_minutes"]) + int(summary["weekend_bound_minutes"]),
            int(summary["field_day_count"] or 999),
            int(summary["total_p75_minutes"] or 999999),
        ),
        default=None,
    )
    return {
        "objective": "quantify how availability bounds affect repaired joint field-day feasibility",
        "source_files": {
            "joint_optimizer": display_path(YEAR_DIR / "scripts" / "p90_joint_field_day_optimizer.py"),
            "official_geojson": display_path(joint_optimizer.DEFAULT_OFFICIAL_GEOJSON),
            "state_json": display_path(joint_optimizer.repaired.DEFAULT_STATE_JSON),
        },
        "config": {
            "max_combo_size": max_combo_size,
            "neighbor_limit": neighbor_limit,
            "connected_expansion_limit": connected_expansion_limit,
        },
        "summary": {
            "scenario_count": len(summaries),
            "feasible_scenario_count": len(feasible),
            "first_feasible_scenario": first_feasible,
            "best_max_coverage_scenario": best_coverage,
        },
        "scenario_summaries": summaries,
        "scenarios": scenarios,
        "caveats": [
            "This is a sensitivity audit, not a user-approved availability change.",
            "Scenarios other than current_260_weekday_180_weekend are non-compliant with the current personal p90 bounds.",
            "Generated combos are limited by max combo size and neighbor limit.",
        ],
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# P90 Availability Sensitivity Audit",
        "",
        f"Objective: {report['objective']}",
        "",
        "## Summary",
        "",
        f"- Scenarios tested: {summary['scenario_count']}",
        f"- Feasible scenarios: {summary['feasible_scenario_count']}",
    ]
    if summary.get("first_feasible_scenario"):
        first = summary["first_feasible_scenario"]
        lines.append(
            f"- First feasible scenario in this grid: `{first['scenario']}` "
            f"({first['field_day_count']} field days, {first['total_p75_minutes']} p75 min)"
        )
    else:
        lines.append("- First feasible scenario in this grid: none")
    best = summary["best_max_coverage_scenario"]
    lines.extend(
        [
            f"- Best max-coverage scenario: `{best['scenario']}` "
            f"({best['max_coverage_segments']}/251 segments, {best['max_coverage_official_miles']} official mi)",
            "",
            "## Scenario Table",
            "",
            "| Scenario | Weekday | Weekend | Feasible | Field days | P75 min | Max coverage | Missing mi | Relaxed min days | Min weekdays w/ actual weekends |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["scenario_summaries"]:
        lines.append(
            f"| {row['scenario']} | {row['weekday_bound_minutes']} | {row['weekend_bound_minutes']} | "
            f"{row['feasible']} | {row.get('field_day_count') or ''} | {row.get('total_p75_minutes') or ''} | "
            f"{row.get('max_coverage_segments')}/251 | {row.get('max_coverage_missing_official_miles')} | "
            f"{row.get('relaxed_min_field_days') or ''} | {row.get('relaxed_min_weekdays_with_actual_weekends') or ''} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report.get("caveats") or [])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--neighbor-limit", type=int, default=20)
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
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
